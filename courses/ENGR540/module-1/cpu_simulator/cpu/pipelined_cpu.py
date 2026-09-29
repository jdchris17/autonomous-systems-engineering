"""PipelinedCPU: five instructions genuinely in flight at once, using
exactly the machinery this package already built for that purpose --
`Datapath`'s stage methods, `control.py`'s control unit,
`pipeline.py`'s `IF_ID`/`ID_EX`/`EX_MEM`/`MEM_WB` registers, and
`forwarding.py`'s forwarding unit. `cpu.py`'s single-cycle `CPU` is kept
exactly as it is -- not thrown away, not edited -- specifically to serve
as this file's reference implementation: for any straight-line program,
`FinalState(CPU) == FinalState(PipelinedCPU)` is the correctness test
this whole file is built to pass (see `module-8/` for that comparison
run for real).

Cycle execution order matters, and it matters in Python the exact same
way it matters in hardware: every stage's next-state value is computed
from *this* cycle's current pipeline state, and none of the pipeline
registers change until every stage has finished reading the old ones.
`step()` computes `next_if_id`/`next_id_ex`/`next_ex_mem`/`next_mem_wb`/
`next_pc` first, then commits all five in one block at the end --
mutating `self.ex_mem` before `_execute()` has read the *old* `ex_mem`
for forwarding would mean an instruction's own EX stage could observe
its own EX stage's output, which is not what a real pipeline register
does on the clock edge it's driven by.

The one exception to "read old, commit new all at once" is the register
file: `_writeback()` runs first each cycle and writes directly into
`self.registers`, *before* `_decode()` (computing `next_id_ex`) reads
from it. This is deliberate, not an oversight -- it's the same
"write-first" same-cycle behavior a real synchronous register file has,
and it's necessary for correctness, not just faithful hardware
mimicry: a value produced far enough back that its producer has already
fully retired (no longer sitting in `EX_MEM` or `MEM_WB` by the time a
dependent instruction reaches EX) can *only* still resolve correctly
through this same-cycle register file bypass -- forwarding physically
cannot reach a producer that has already left the pipeline.

Control hazards: branches resolve in EX under static not-taken
prediction (fetch always just fetches `pc + 4`, every cycle, with no
predictor). When a branch comes out taken, `hazards.py`'s
`branch_hazard_control` says to flush `IF_ID` and `ID_EX` -- the two
stages that ran ahead of it on the wrong assumption -- and `pc`
redirects to `EX_MEM.branch_target`. `JAL`/`JALR` are the one
remaining, still-documented gap: both are correctly computed (`rd`
gets `pc + 4`), but neither actually redirects `pc` yet -- unconditional
jumps weren't part of this round's ask, only conditional branches were.
Every test/demo either avoids jumps entirely or pads around them.

Data hazards forwarding can't reach: a load's value doesn't exist until
MEMORY runs, so an instruction immediately behind it in the pipeline
can't get a correct value from EX_MEM the way any other RAW dependency
can (see `forwarding.py`'s own docstring on why it skips a load still
in EX_MEM). `hazards.py`'s `detect_load_use_hazard` catches this one
specific case and stalls fetch/decode for a single cycle, inserting a
bubble into `ID_EX` in place of the dependent instruction; by the time
that instruction actually reaches EX, the load has reached MEM_WB and
ordinary forwarding takes over from there.

Structural hazard: MEM stage latency (section 58). By default (no
`data_cache_size_bytes` given) `_memory()` talks straight to
`self.data_memory`, exactly as before -- every load/store still takes
one cycle, and every existing test's exact cycle counts still hold.
Passing `data_cache_size_bytes` builds a real `Cache` in front of
`self.data_memory` and wraps both in a `MemoryHierarchy`; a load or
store then costs `MemoryAccessResult.cycles` instead of an assumed 1,
and `hazards.py`'s `memory_stall_control()` freezes the rest of the
pipeline (`IF_ID`/`ID_EX`/`EX_MEM`, and `pc`) for however many extra
cycles a miss takes. The stalled instruction stays exactly where it is
in `EX_MEM` across all of those cycles -- `MEM_WB` bubbles in the
meantime, since nothing has actually finished MEMORY yet -- and the
cache is only ever asked for the access's result *once*, at the moment
the stall begins; `self._pending_mem_wb`/`self.memory_stall_remaining`
are what remember that result and how much longer to wait, since the
cache itself has no notion of a multi-cycle access already in flight.

A second consequence of that: `step()` always fetches, every cycle,
whatever `pc` currently points at -- there is no "the program has ended,
stop fetching" signal. `len(program)` cycles only gets the *last*
instruction *into* the pipeline, not out the other end; four more
cycles (one per remaining stage it still has to clear) are needed to
drain it, and FETCH keeps running during those, straight into whatever
comes after the program in instruction memory. `load_program()` below
pads the assembled program with harmless real NOPs
(`addi x0, x0, 0` -- writes are discarded, x0 stays zero) specifically
so that drain has something valid to fetch instead of uninitialized
memory decoding into a crash.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

if __package__ in (None, ""):
    # See core/adder.py's module docstring for why this is needed to run
    # this file directly (e.g. VS Code's Run button) rather than via
    # `python -m`. Three .parent calls: pipelined_cpu.py -> cpu/ ->
    # cpu_simulator/ -> module-1/.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.core.adder import ripple_add
from cpu_simulator.core.alu import alu
from cpu_simulator.cpu.analysis.performance import PerformanceCounters
from cpu_simulator.cpu.clock import Clock
from cpu_simulator.cpu.control import generate_control
from cpu_simulator.cpu.datapath import Datapath
from cpu_simulator.cpu.forwarding import REGISTER_VALUE, determine_forwarding, forward_operands
from cpu_simulator.cpu.hazards import (
    HazardControl,
    branch_hazard_control,
    detect_load_use_hazard,
    load_use_hazard_control,
    memory_stall_control,
)
from cpu_simulator.cpu.pc import ProgramCounter
from cpu_simulator.cpu.pipeline import EX_MEM, ID_EX, IF_ID, MEM_WB
from cpu_simulator.cpu.register_file import RegisterFile
from cpu_simulator.cpu.trace import PipelineTrace
from cpu_simulator.isa.assembler import assemble
from cpu_simulator.isa.encoder import encode_i_type_arith
from cpu_simulator.memory.cache import Cache
from cpu_simulator.memory.memory import Memory
from cpu_simulator.memory.memory_hierarchy import MemoryHierarchy

WIDTH = 32
MASK = (1 << WIDTH) - 1
INSTRUCTION_WIDTH = 4
PIPELINE_DEPTH = 5  # IF, ID, EX, MEM, WB

NOP_WORD = encode_i_type_arith("ADDI", rd=0, rs1=0, immediate=0)  # addi x0, x0, 0


def load_program(cpu: "PipelinedCPU", source: str, base_address: int = 0,
                  pad_instructions: int = PIPELINE_DEPTH - 1) -> int:
    """Assemble `source`, write it into `cpu`'s instruction memory, and
    pad it with `pad_instructions` NOPs so the pipeline has something
    valid to fetch while the real program's last instruction drains
    through the remaining stages. Returns the real (unpadded)
    instruction count -- `run(count + pad_instructions)` (or more) is
    what actually lets every instruction finish its own writeback.
    """
    words = assemble(source, base_address=base_address)
    for i, word in enumerate(words):
        cpu.instruction_memory.write_word(base_address + i * INSTRUCTION_WIDTH, word)
    for i in range(pad_instructions):
        pad_address = base_address + (len(words) + i) * INSTRUCTION_WIDTH
        cpu.instruction_memory.write_word(pad_address, NOP_WORD)
    return len(words)


class PipelinedCPU:
    def __init__(self, instruction_memory_bytes: int = 1024, data_memory_bytes: int = 1024,
                 data_cache_size_bytes: int | None = None, data_cache_line_size: int = 16,
                 data_cache_associativity: int = 1, hit_latency: int = 1, miss_latency: int = 20):
        self.pc = ProgramCounter()
        self.clock = Clock()
        self.registers = RegisterFile()
        self.instruction_memory = Memory(size_bytes=instruction_memory_bytes)
        self.data_memory = Memory(size_bytes=data_memory_bytes)
        self.datapath = Datapath(self.instruction_memory, self.data_memory, self.registers)

        # None (the default) preserves the original, cache-free MEM
        # stage exactly -- every load/store costs one cycle, no
        # exceptions, and every cycle count any earlier test/demo
        # asserts still holds. Giving `data_cache_size_bytes` opts into
        # a real `Cache` in front of `self.data_memory` and the latency
        # `_memory()` now models through it -- see the module docstring.
        if data_cache_size_bytes is not None:
            self.data_cache = Cache(size_bytes=data_cache_size_bytes, line_size=data_cache_line_size,
                                     lower_memory=self.data_memory, associativity=data_cache_associativity)
            self.data_hierarchy = MemoryHierarchy(self.data_cache, hit_latency=hit_latency,
                                                   miss_latency=miss_latency)
        else:
            self.data_cache = None
            self.data_hierarchy = None

        self.if_id = IF_ID()
        self.id_ex = ID_EX()
        self.ex_mem = EX_MEM()
        self.mem_wb = MEM_WB()

        # State a multi-cycle memory access needs remembered *across*
        # step() calls -- see the module docstring's "Structural
        # hazard" section for why this can't just be recomputed fresh
        # each cycle the way load-use/branch hazards are.
        self.memory_stall_remaining = 0
        self._pending_mem_wb: MEM_WB | None = None

        self.last_trace: PipelineTrace | None = None
        self.trace_history: list[PipelineTrace] = []
        self.performance = PerformanceCounters()

    def step(self) -> None:
        """One simulated cycle. See the module docstring for why
        writeback runs first (against the register file directly) and
        everything else is computed from old state before any pipeline
        register is overwritten.

        `pre_if_id`/`pre_id_ex`/`pre_ex_mem`/`pre_mem_wb` capture this
        cycle's *starting* pipeline state before any stage runs. No
        stage function ever mutates a pipeline register in place (each
        one returns a fresh dataclass instance), so simply holding these
        references is enough -- they stay exactly the "old" values every
        stage below is entitled to read from, all the way to
        `_build_trace()`, which needs them again afterward to show what
        each stage actually consumed this cycle.
        """
        pre_if_id = self.if_id
        pre_id_ex = self.id_ex
        pre_ex_mem = self.ex_mem
        pre_mem_wb = self.mem_wb

        self._writeback(pre_mem_wb)

        if self.memory_stall_remaining > 0:
            # Already mid-access from an earlier cycle -- don't touch
            # the cache again, just count down. See the module
            # docstring's "Structural hazard" section for why nothing
            # else in the pipeline can move while this is happening.
            self.memory_stall_remaining -= 1
            if self.memory_stall_remaining == 0:
                # The access finishes *this* cycle -- MEM_WB gets the
                # real result, and EX_MEM is free again starting this
                # same cycle (not the next one), so the instruction
                # that's been held in ID_EX this whole time can finally
                # execute into it. Falls through to the normal path
                # below exactly as if nothing had been stalled.
                next_mem_wb = self._pending_mem_wb
                self._pending_mem_wb = None
                memory_busy = False
            else:
                next_mem_wb = MEM_WB()
                memory_busy = True
                forward_a = forward_b = REGISTER_VALUE
        else:
            next_mem_wb, extra_stall_cycles = self._memory(pre_ex_mem)
            if extra_stall_cycles > 0:
                self.memory_stall_remaining = extra_stall_cycles
                self._pending_mem_wb = next_mem_wb
                next_mem_wb = MEM_WB()
                memory_busy = True
                forward_a = forward_b = REGISTER_VALUE
            else:
                memory_busy = False

        if memory_busy:
            next_ex_mem = pre_ex_mem  # the stalled instruction stays exactly where it is
            next_id_ex = pre_id_ex    # EX can't run -- EX_MEM has nowhere to receive its output
            next_if_id = pre_if_id
            next_pc = self.pc.read()
            # `self.memory_stall_remaining` already reflects "cycles
            # still needed after this one" by this point -- it was set
            # fresh (no decrement yet) on the cycle the access started,
            # or already decremented at the top of this same step() on
            # every cycle since.
            stall = memory_stall_control(True, self.memory_stall_remaining)
            branch = HazardControl()
        else:
            next_ex_mem, forward_a, forward_b = self._execute(pre_id_ex, pre_ex_mem, pre_mem_wb)
            next_id_ex_candidate = self._decode(pre_if_id)

            load_use_conflict = detect_load_use_hazard(
                pre_id_ex, next_id_ex_candidate.rs1, next_id_ex_candidate.rs2)
            stall = load_use_hazard_control(load_use_conflict)
            branch = branch_hazard_control(next_ex_mem.valid and next_ex_mem.branch_taken)

            next_id_ex = ID_EX() if (stall.insert_bubble or branch.flush_id_ex) else next_id_ex_candidate

            if branch.flush_if_id:
                next_if_id = IF_ID()
            elif stall.stall_decode:
                next_if_id = pre_if_id
            else:
                next_if_id = self._fetch(self.pc.read())

            if branch.flush_if_id:
                # A branch resolving taken always wins over an unrelated
                # load-use stall for whatever fetch/decode were doing this
                # cycle -- and the two can never actually collide, since
                # `pre_id_ex` can't simultaneously be both a load (what
                # triggers a stall) and a branch (what triggers a flush).
                next_pc = next_ex_mem.branch_target
            elif stall.stall_fetch:
                next_pc = self.pc.read()
            else:
                next_pc, _carry = ripple_add(self.pc.read(), INSTRUCTION_WIDTH, width=WIDTH)

        self.mem_wb = next_mem_wb
        self.ex_mem = next_ex_mem
        self.id_ex = next_id_ex
        self.if_id = next_if_id
        self.pc.jump(next_pc)

        self.clock.tick()

        self._record_performance(pre_mem_wb, next_ex_mem, forward_a, forward_b, stall, branch, memory_busy)

        self.last_trace = self._build_trace(
            pre_if_id, pre_id_ex, pre_ex_mem, pre_mem_wb, forward_a, forward_b, stall, branch)
        self.trace_history.append(self.last_trace)

    def run(self, num_cycles: int) -> None:
        for _ in range(num_cycles):
            self.step()

    def _fetch(self, pc: int) -> IF_ID:
        raw = self.datapath.fetch(pc)
        return IF_ID(valid=True, pc=pc, instruction=raw)

    def _decode(self, if_id: IF_ID) -> ID_EX:
        if not if_id.valid:
            return ID_EX()
        instruction = self.datapath.decode(if_id.instruction)
        control = generate_control(instruction)
        rs1 = instruction.rs1 if instruction.rs1 is not None else 0
        rs2 = instruction.rs2 if instruction.rs2 is not None else 0
        rd = instruction.rd if instruction.rd is not None else 0
        immediate = instruction.immediate if instruction.immediate is not None else 0
        return ID_EX(
            valid=True, pc=if_id.pc, raw=if_id.instruction, mnemonic=instruction.mnemonic,
            rs1=rs1, rs2=rs2,
            rs1_value=self.registers.read(rs1), rs2_value=self.registers.read(rs2),
            rd=rd, immediate=immediate, control=control,
        )

    def _execute(self, id_ex: ID_EX, ex_mem: EX_MEM, mem_wb: MEM_WB) -> tuple[EX_MEM, str, str]:
        if not id_ex.valid:
            return EX_MEM(), REGISTER_VALUE, REGISTER_VALUE

        # id_ex.rs1_value/rs2_value were captured back at decode time --
        # fine under ordinary one-cycle-per-stage timing, where at most
        # one cycle separates decode from execute and forwarding's own
        # one-cycle EX_MEM/MEM_WB window always covers that gap. A
        # memory stall breaks that assumption: an instruction can now
        # sit *held* in ID_EX (unable to execute, since EX_MEM has
        # nowhere to receive its output) across many cycles, and a
        # producer that retires during that freeze passes through
        # MEM_WB's own narrow one-cycle forwarding window and is gone by
        # the time this instruction finally gets to execute -- forwarding
        # can no longer reach it, but the register file (already updated
        # by that producer's writeback) still can. Re-reading here
        # -- right before forwarding, every single time, not just when
        # something was actually held -- costs nothing extra for the
        # ordinary case (decode already read the same fresh values) and
        # is what actually keeps this correct for the held one.
        id_ex = replace(id_ex, rs1_value=self.registers.read(id_ex.rs1),
                         rs2_value=self.registers.read(id_ex.rs2))

        # determine_forwarding() is called directly here (not just
        # through forward_operands() below) because the trace needs the
        # *decision* (which source fed each operand), not only the
        # resulting value -- exactly the split forwarding.py's own
        # docstring calls out these two functions for.
        forward_a = determine_forwarding(id_ex.rs1, ex_mem, mem_wb)
        forward_b = determine_forwarding(id_ex.rs2, ex_mem, mem_wb)
        operand_a, forwarded_rs2 = forward_operands(id_ex, ex_mem, mem_wb)
        if id_ex.mnemonic == "AUIPC":
            # AUIPC's first ALU operand is pc, not a register -- there's
            # no rs1 to forward into, the same special case
            # Datapath._operand_a makes for the single-cycle CPU.
            operand_a = id_ex.pc

        # The ALU's second operand is the immediate for everything
        # *except* R-type/branches; forwarded_rs2 is still needed
        # separately below regardless, since a STORE's value-to-write
        # is always rs2, whatever alu_src says about the ALU's own
        # second operand.
        alu_operand_b = forwarded_rs2 if id_ex.control.alu_src == "REGISTER" else (id_ex.immediate & MASK)

        result = alu(operand_a, alu_operand_b, id_ex.control.alu_op, width=WIDTH)
        branch_taken = self._branch_taken(id_ex.control, result)
        pc_plus_4, _carry = ripple_add(id_ex.pc, INSTRUCTION_WIDTH, width=WIDTH)
        # pc-relative, the same computation cpu.py's _update_pc uses for
        # the single-cycle model -- harmless to compute for every
        # instruction, not just branches, since it's only ever read when
        # branch_taken is True.
        branch_target, _carry = ripple_add(id_ex.pc, id_ex.immediate & MASK, width=WIDTH)

        return EX_MEM(
            valid=True, pc=id_ex.pc, raw=id_ex.raw, mnemonic=id_ex.mnemonic,
            alu_result=result.value, rs2_value=forwarded_rs2, rd=id_ex.rd,
            branch_taken=branch_taken, branch_target=branch_target,
            pc_plus_4=pc_plus_4, control=id_ex.control,
        ), forward_a, forward_b

    def _memory(self, ex_mem: EX_MEM) -> tuple[MEM_WB, int]:
        """Returns the MEM_WB this instruction produces, plus how many
        *extra* cycles (beyond this one) its memory access still needs
        -- 0 for anything that isn't a load/store, for a hit, or
        whenever `self.data_hierarchy` is None (the original,
        cache-free behavior). Only a load/store actually goes through
        `self.data_hierarchy` at all; everything else still just passes
        `alu_result` through unchanged, the same as before section 58.
        """
        if not ex_mem.valid:
            return MEM_WB(), 0

        extra_stall_cycles = 0
        if self.data_hierarchy is not None and ex_mem.control.mem_read:
            result = self.data_hierarchy.load_word(ex_mem.alu_result)
            memory_result = result.value
            extra_stall_cycles = result.cycles - 1
        elif self.data_hierarchy is not None and ex_mem.control.mem_write:
            result = self.data_hierarchy.store_word(ex_mem.alu_result, ex_mem.rs2_value)
            memory_result = ex_mem.alu_result
            extra_stall_cycles = result.cycles - 1
        elif ex_mem.control.mem_read:
            memory_result = self.data_memory.read_word(ex_mem.alu_result)
        elif ex_mem.control.mem_write:
            self.data_memory.write_word(ex_mem.alu_result, ex_mem.rs2_value)
            memory_result = ex_mem.alu_result
        else:
            memory_result = ex_mem.alu_result

        return MEM_WB(
            valid=True, pc=ex_mem.pc, raw=ex_mem.raw, mnemonic=ex_mem.mnemonic,
            memory_result=memory_result, rd=ex_mem.rd,
            pc_plus_4=ex_mem.pc_plus_4, control=ex_mem.control,
        ), extra_stall_cycles

    def _writeback(self, mem_wb: MEM_WB) -> None:
        if not mem_wb.valid or mem_wb.control is None or not mem_wb.control.reg_write:
            return
        value = mem_wb.pc_plus_4 if mem_wb.control.result_src == "PC_PLUS_4" else mem_wb.memory_result
        self.registers.write(mem_wb.rd, value)

    def _branch_taken(self, control, alu_result) -> bool:
        # Same logic as Datapath._branch_taken -- duplicated rather than
        # called, since Datapath's version is a bound method that
        # expects to be invoked as part of its own execute(), not as a
        # standalone helper on pre-computed operands.
        if control.branch is None:
            return False
        if control.branch == "BEQ":
            return bool(alu_result.zero)
        if control.branch == "BNE":
            return not alu_result.zero
        if control.branch in ("BLT", "BLTU"):
            return alu_result.value == 1
        if control.branch in ("BGE", "BGEU"):
            return alu_result.value == 0
        raise ValueError(f"unrecognized branch mnemonic {control.branch!r}")

    def _record_performance(self, pre_mem_wb: MEM_WB, next_ex_mem: EX_MEM,
                             forward_a: str, forward_b: str,
                             stall: HazardControl, branch: HazardControl,
                             memory_busy: bool) -> None:
        """Updates `self.performance` from this cycle's already-computed
        values -- no new decisions made here, just counting ones
        `step()` already made. `pre_mem_wb.valid` (not `next_ex_mem` or
        anything upstream) is what "retired" means: writeback is the
        one stage every real instruction, whatever it does or doesn't
        write, must pass through to be done -- so the instruction-mix
        counters (`alu`/`load`/`store`/`branch_instructions`) classify
        `pre_mem_wb`, the instruction actually retiring this cycle, not
        whatever's mid-flight elsewhere. `branches_taken` is the one
        exception: `MEM_WB` never carries `branch_taken` (only `EX_MEM`
        does), so it's still read off `next_ex_mem` at the moment a
        branch resolves -- every branch that reaches `EX_MEM` also goes
        on to retire (a branch is never itself a flushed, wrong-path
        instruction), so the two measurement points agree on the total
        count regardless.
        """
        self.performance.total_cycles += 1

        if pre_mem_wb.valid:
            self.performance.instructions_retired += 1
            if pre_mem_wb.control.mem_read:
                self.performance.load_instructions += 1
            elif pre_mem_wb.control.mem_write:
                self.performance.store_instructions += 1
            elif pre_mem_wb.control.branch is not None:
                self.performance.branch_instructions += 1
            else:
                self.performance.alu_instructions += 1

        if stall.insert_bubble:
            self.performance.data_hazard_stalls += 1
            self.performance.load_use_stalls += 1
        if memory_busy:
            self.performance.memory_stall_cycles += 1
        if next_ex_mem.valid and next_ex_mem.control is not None and next_ex_mem.control.branch is not None:
            if next_ex_mem.branch_taken:
                self.performance.branches_taken += 1
        if branch.flush_if_id:
            self.performance.branch_flush_cycles += 1
        for source in (forward_a, forward_b):
            if source != REGISTER_VALUE:
                self.performance.forwarding_events += 1

        if self.data_cache is not None:
            # Mirrored, not independently counted -- Cache is the
            # ground truth for its own hit/miss bookkeeping; this just
            # brings it into the same unified report.
            self.performance.cache_accesses = self.data_cache.stats.accesses
            self.performance.cache_hits = self.data_cache.stats.hits
            self.performance.cache_misses = self.data_cache.stats.misses
            self.performance.dirty_writebacks = self.data_cache.stats.dirty_writebacks

    def _build_trace(self, pre_if_id: IF_ID, pre_id_ex: ID_EX, pre_ex_mem: EX_MEM,
                      pre_mem_wb: MEM_WB, forward_a: str, forward_b: str,
                      stall: HazardControl, branch: HazardControl) -> PipelineTrace:
        """Assembles this cycle's PipelineTrace -- pipelined_cpu.py is
        the only thing that sees every stage's state in the same
        instant a cycle commits, the same reasoning cpu.py's step()
        is the sole builder of its own StepTrace (see trace.py).

        `self.if_id` (read *after* the commit above) is deliberately
        what the IF row shows, not `pre_if_id` -- IF's whole point is
        showing what got freshly fetched (or flushed to a bubble, if a
        branch resolved this same cycle) this cycle, which is exactly
        the post-commit value. Every other row shows its *pre*-commit
        state, since that's what each stage actually consumed as input
        this cycle -- see step()'s own docstring for why capturing
        `pre_if_id`/etc. once at the top is sufficient for that.
        """
        return PipelineTrace(
            cycle=self.clock.cycle,
            if_stage=self.if_id,
            id_stage=pre_if_id,
            ex_stage=pre_id_ex,
            mem_stage=pre_ex_mem,
            wb_stage=pre_mem_wb,
            forward_a=forward_a,
            forward_b=forward_b,
            stall=stall.reason is not None,
            stall_reason=stall.reason,
            stall_detail=stall.detail,
            flush=branch.flush_if_id,
            branch_target=self.ex_mem.branch_target if branch.flush_if_id else None,
        )
