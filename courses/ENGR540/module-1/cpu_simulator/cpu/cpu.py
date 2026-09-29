"""The coordinator: runs real, decoded RV32I instructions through
datapath.py's five stages and control.py's control unit. Originally
built in module-7/ (Module 7's own answer to "what runs decoded
instructions," once decoder.py and control.py existed) and moved here
once the target package layout put control.py/datapath.py/cpu.py/
trace.py inside cpu_simulator/cpu/ alongside register_file.py/memory.py/
pc.py -- the same reusable-library location as everything else this
class depends on, rather than a separate consumer folder.

pseudo_cpu.py's PseudoOpCPU, in this same directory, is the original,
pre-decoder integration point (Module 5's artificial pseudo-instruction
tuples) -- kept as the deliberately-unchanged reference module-5/ and
module-6/ still use, not superseded by this file. See pseudo_cpu.py's
own docstring for the full story of why both exist side by side.

step() is deliberately one small function that *calls* five separate
concerns, not one giant function that *is* all five concerns --
"separate concerns rather than having one giant function that knows
everything," per section 29. Swap step() for something that latches
values between stages across cycles instead of calling them all in one
pass, and this is most of the way to a pipelined design (Module 8);
nothing about datapath.py's own stage methods needs to change for that.

update_pc() is deliberately not a datapath method -- picking the next
pc is a sequencing decision (does this instruction fall through, branch,
or jump?), not a data-movement one, and it's the one piece of the
section-29 sketch this file keeps separate exactly as shown.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    # See core/adder.py's module docstring for why this is needed to run
    # this file directly (e.g. VS Code's Run button) rather than via
    # `python -m`. Three .parent calls: cpu.py -> cpu/ -> cpu_simulator/
    # -> module-1/.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.core.adder import ripple_add
from cpu_simulator.cpu.analysis.performance import PerformanceCounters
from cpu_simulator.cpu.clock import Clock
from cpu_simulator.cpu.control import generate_control
from cpu_simulator.cpu.datapath import Datapath
from cpu_simulator.cpu.pc import ProgramCounter
from cpu_simulator.cpu.register_file import RegisterFile
from cpu_simulator.cpu.trace import StepTrace
from cpu_simulator.memory.memory import Memory

WIDTH = 32
MASK = (1 << WIDTH) - 1
INSTRUCTION_WIDTH = 4


class CPU:
    def __init__(self, instruction_memory_bytes: int = 1024, data_memory_bytes: int = 1024):
        self.pc = ProgramCounter()
        self.clock = Clock()
        self.registers = RegisterFile()
        self.instruction_memory = Memory(size_bytes=instruction_memory_bytes)
        self.data_memory = Memory(size_bytes=data_memory_bytes)
        self.datapath = Datapath(self.instruction_memory, self.data_memory, self.registers)
        self.last_trace: StepTrace | None = None
        self.trace_history: list[StepTrace] = []
        self.performance = PerformanceCounters()

    def step(self) -> None:
        """One full cycle: fetch, decode, generate control, execute,
        access memory, write back, then decide the next pc. Also builds
        this cycle's StepTrace -- cpu.py is the only thing that sees
        every intermediate value in a cycle, so it's the only place
        that can assemble one; trace.py only renders what's handed to
        it (see that file's own docstring).
        """
        pc_before = self.pc.read()

        fetched = self.datapath.fetch(pc_before)
        decoded = self.datapath.decode(fetched)
        control = generate_control(decoded)

        rs1_value = self.registers.read(decoded.rs1) if decoded.rs1 is not None else None
        rs2_value = self.registers.read(decoded.rs2) if decoded.rs2 is not None else None

        executed = self.datapath.execute(decoded, control, pc_before)
        memory_result = self.datapath.memory_access(decoded, control, executed)

        # Captured before writeback runs -- a store's value and a
        # trace's "what was in this register during this cycle" both
        # mean the value as of *before* any write this instruction
        # itself performs, the same way real hardware reads happen
        # before a same-cycle write commits.
        mem_write_value = self.registers.read(decoded.rs2) if control.mem_write else None
        mem_address = executed.alu_value if (control.mem_read or control.mem_write) else None
        mem_read_value = memory_result if control.mem_read else None

        pc_plus_4, _carry = ripple_add(pc_before, INSTRUCTION_WIDTH, width=WIDTH)
        writeback_value = self.datapath.writeback(decoded, control, memory_result, pc_plus_4)
        self._update_pc(pc_before, pc_plus_4, decoded, control, executed)

        self.clock.tick()

        self.performance.total_cycles += 1
        self.performance.instructions_retired += 1  # every step() retires exactly one instruction
        if control.mem_read:
            self.performance.load_instructions += 1
        elif control.mem_write:
            self.performance.store_instructions += 1
        elif control.branch is not None:
            self.performance.branch_instructions += 1
            if executed.branch_taken:
                self.performance.branches_taken += 1
        else:
            self.performance.alu_instructions += 1

        self.last_trace = StepTrace(
            cycle=self.clock.cycle,
            pc=pc_before,
            raw=fetched,
            instruction=decoded,
            rs1_value=rs1_value,
            rs2_value=rs2_value,
            control=control,
            operand_a=executed.operand_a,
            operand_b=executed.operand_b,
            alu_result=executed.alu_value,
            branch_taken=executed.branch_taken,
            mem_address=mem_address,
            mem_read_value=mem_read_value,
            mem_write_value=mem_write_value,
            writeback_value=writeback_value,
            next_pc=self.pc.read(),
        )
        self.trace_history.append(self.last_trace)

    def run(self, num_steps: int) -> None:
        for _ in range(num_steps):
            self.step()

    def _update_pc(self, pc_before, pc_plus_4, instruction, control, executed) -> None:
        if control.jump:
            if instruction.mnemonic == "JAL":
                # JAL's target (pc + immediate) is a separate adder in
                # a real datapath, not the main ALU's output -- see
                # control.py's own note that the ALU isn't actually
                # driving anything for JAL. Computed directly here.
                target, _carry = ripple_add(pc_before, instruction.immediate & MASK, width=WIDTH)
            else:  # JALR
                # JALR's target *is* what execute() already computed
                # (rs1 + immediate) -- the main ALU is correctly wired
                # for this one. RISC-V clears the low bit of the result.
                target = executed.alu_value & ~1 & MASK
            self.pc.jump(target)
        elif control.branch is not None and executed.branch_taken:
            target, _carry = ripple_add(pc_before, instruction.immediate & MASK, width=WIDTH)
            self.pc.jump(target)
        else:
            self.pc.jump(pc_plus_4)


def main():
    """The module-6 program, run through real decoding this time
    instead of Module 5's pseudo-instruction shortcut.
    """
    from cpu_simulator.isa.assembler import assemble

    program = """
addi x1, x0, 10
addi x2, x0, 7
add  x3, x1, x2
sub  x4, x3, x2
sw   x4, 0(x0)
    """
    words = assemble(program)

    cpu = CPU()
    for i, word in enumerate(words):
        cpu.instruction_memory.write_word(i * 4, word)

    print("=" * 60)
    print("Running real decoded RV32I instructions")
    print("=" * 60)
    cpu.run(len(words))
    for r in (1, 2, 3, 4):
        print(f"  x{r} = {cpu.registers.read(r)}")
    print(f"  Memory[0] = {cpu.data_memory.read_word(0)}")
    print(f"  pc = {cpu.pc.read():#06x}  cycles = {cpu.clock.cycle}")

    assert cpu.registers.read(1) == 10
    assert cpu.registers.read(2) == 7
    assert cpu.registers.read(3) == 17
    assert cpu.registers.read(4) == 10
    assert cpu.data_memory.read_word(0) == 10
    print()
    print("Matches module-6's architectural target -- reached this time")
    print("by actually decoding and executing each instruction, not by")
    print("running its Module 5 pseudo-op equivalent.")
    print("=" * 60)


if __name__ == "__main__":
    main()
