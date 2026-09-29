"""Per-cycle execution trace: everything cpu.py's step() computes,
captured in one place and rendered two ways -- verbose (nearly an
executable version of the textbook datapath diagram, one section per
stage) and compact (one line per instruction, for a whole run).

StepTrace is deliberately a plain data record, not something that
prints itself. Building the trace (cpu.py's job -- it's the only thing
that ever sees every intermediate value in a cycle) and rendering it
(this file's job) are two different concerns; keeping them separate is
also what makes format_verbose/format_compact independently testable
without running a CPU at all.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    # See core/adder.py's module docstring for why this is needed to run
    # this file directly (e.g. VS Code's Run button) rather than via
    # `python -m`. Three .parent calls: trace.py -> cpu/ ->
    # cpu_simulator/ -> module-1/.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.cpu.control import ControlSignals
from cpu_simulator.cpu.forwarding import EX_MEM_RESULT, MEM_WB_RESULT, REGISTER_VALUE
from cpu_simulator.cpu.pipeline import EX_MEM, ID_EX, IF_ID, MEM_WB
from cpu_simulator.isa.decoder import decode as decode_word
from cpu_simulator.isa.decoder import to_asm
from cpu_simulator.isa.instruction import Instruction


@dataclass
class StepTrace:
    cycle: int
    pc: int
    raw: int
    instruction: Instruction
    rs1_value: int | None
    rs2_value: int | None
    control: ControlSignals
    operand_a: int
    operand_b: int
    alu_result: int
    branch_taken: bool
    mem_address: int | None
    mem_read_value: int | None
    mem_write_value: int | None
    writeback_value: int | None
    next_pc: int


def format_verbose(trace: StepTrace) -> str:
    """One section per pipeline stage -- section 31's own description:
    "nearly an executable version of the textbook datapath diagram."
    """
    inst = trace.instruction
    c = trace.control
    lines = [f"Cycle/Step {trace.cycle}", "-" * 30, ""]

    lines += ["PC", f"{trace.pc:#010x}", ""]
    lines += ["FETCH", f"Instruction: 0x{trace.raw:08X}", ""]

    lines.append("DECODE")
    lines.append(f"Mnemonic: {inst.mnemonic}")
    if inst.rd is not None:
        lines.append(f"rd:  x{inst.rd}")
    if inst.rs1 is not None:
        lines.append(f"rs1: x{inst.rs1}")
    if inst.rs2 is not None:
        lines.append(f"rs2: x{inst.rs2}")
    if inst.immediate is not None:
        lines.append(f"imm: {inst.immediate}")
    lines.append("")

    register_lines = []
    if inst.rs1 is not None:
        register_lines.append(f"x{inst.rs1} = {trace.rs1_value}")
    if inst.rs2 is not None:
        register_lines.append(f"x{inst.rs2} = {trace.rs2_value}")
    if register_lines:
        lines.append("REGISTERS")
        lines += register_lines
        lines.append("")

    lines.append("CONTROL")
    lines.append(f"RegWrite  = {int(c.reg_write)}")
    lines.append(f"MemRead   = {int(c.mem_read)}")
    lines.append(f"MemWrite  = {int(c.mem_write)}")
    lines.append(f"ALUSrc    = {c.alu_src}")
    lines.append(f"ALUOp     = {c.alu_op}")
    lines.append(f"ResultSrc = {c.result_src}")
    if c.branch is not None:
        lines.append(f"Branch    = {c.branch}")
    if c.jump:
        lines.append("Jump      = 1")
    lines.append("")

    lines += ["EXECUTE", f"A = {trace.operand_a}", f"B = {trace.operand_b}",
              f"ALUResult = {trace.alu_result}", ""]

    lines.append("MEMORY")
    if trace.mem_read_value is not None:
        lines.append(f"Read  [{trace.mem_address:#010x}] -> {trace.mem_read_value}")
    elif trace.mem_write_value is not None:
        lines.append(f"Write [{trace.mem_address:#010x}] <- {trace.mem_write_value}")
    else:
        lines.append("No access")
    lines.append("")

    lines.append("WRITEBACK")
    if trace.writeback_value is not None:
        lines.append(f"x{inst.rd} <- {trace.writeback_value}")
    else:
        lines.append("No writeback")
    lines.append("")

    lines += ["NEXT PC", f"{trace.next_pc:#010x}"]

    return "\n".join(lines)


def format_compact(trace: StepTrace) -> str:
    """One line per instruction: PC, mnemonic, operands, and whatever
    actually happened (a register write, a branch decision, a store) --
    "eventually the verbose trace will be too much... for running
    programs."
    """
    inst = trace.instruction
    line = f"PC={trace.pc:#x} {inst.mnemonic} {_compact_operands(inst)}"

    if trace.writeback_value is not None:
        line += f"  x{inst.rd}<-{trace.writeback_value}"
    elif trace.control.branch is not None:
        line += f"  TAKEN -> {trace.next_pc:#x}" if trace.branch_taken else "  NOT TAKEN"
    elif trace.mem_write_value is not None:
        line += f"  MEM[{trace.mem_address:#x}]<-{trace.mem_write_value}"

    return line


def _compact_operands(inst: Instruction) -> str:
    """rd, rs1, rs2, immediate -- whichever of these a format actually
    has, in that order. That single, uniform rule happens to reproduce
    every operand order RV32I assembly conventionally uses (R-type's
    "rd,rs1,rs2", I-type's "rd,rs1,imm", B-type's "rs1,rs2,imm" -- rd
    is simply absent for branches) without hand-listing a format table
    the way to_asm() in decoder.py has to, since a trace line doesn't
    need to match assembly syntax exactly, just be unambiguous.
    """
    parts = []
    if inst.rd is not None:
        parts.append(f"x{inst.rd}")
    if inst.rs1 is not None:
        parts.append(f"x{inst.rs1}")
    if inst.rs2 is not None:
        parts.append(f"x{inst.rs2}")
    if inst.immediate is not None:
        parts.append(str(inst.immediate))
    return ",".join(parts)


@dataclass
class PipelineTrace:
    """One cycle's snapshot of every pipeline stage at once -- the
    pipelined model's answer to StepTrace, which only ever had one
    instruction to describe per cycle. `pipelined_cpu.py`'s
    `_build_trace()` is this file's `StepTrace` (the only thing that
    sees every stage's state in the same instant a cycle commits);
    `format_pipeline()` below only renders what's handed to it, the
    same division of labor as the single-cycle model's two functions.

    `if_stage`/`id_stage`/`ex_stage`/`mem_stage`/`wb_stage` are each
    that pipeline register's contents *as of this cycle's IF/ID/EX/
    MEM/WB row* -- not all five are the same register's "current" vs.
    "previous" value in the same sense, since IF has no pipeline
    register upstream of it. See `PipelinedCPU._build_trace()`'s own
    docstring for exactly which (pre- or post-commit) each one is.
    """
    cycle: int
    if_stage: IF_ID
    id_stage: IF_ID
    ex_stage: ID_EX
    mem_stage: EX_MEM
    wb_stage: MEM_WB
    forward_a: str
    forward_b: str
    stall: bool
    stall_reason: str | None
    flush: bool
    branch_target: int | None


_FORWARD_LABELS = {
    REGISTER_VALUE: "REG",
    EX_MEM_RESULT: "EX/MEM",
    MEM_WB_RESULT: "MEM/WB",
}


def _pipeline_asm(raw: int) -> str:
    """decoder.py's own to_asm(), with the comma-space it uses for
    ordinary assembly listings ("add x5, x6, x7") tightened to the
    comma-only convention section 8's worked trace example uses
    ("add x5,x6,x7") -- a rendering-only difference, not a second
    assembly syntax; to_asm() itself isn't duplicated here.
    """
    return to_asm(decode_word(raw)).replace(", ", ",")


def _stage_line(label: str, valid: bool, pc: int, raw: int) -> str:
    if not valid:
        return f"{label:<7}(bubble)"
    return f"{label:<7}0x{raw:08x}  {_pipeline_asm(raw)}"


def format_pipeline(trace: PipelineTrace) -> str:
    """One line per pipeline stage, oldest instruction (WB) at the
    bottom, newest (IF) at the top -- plus this cycle's forwarding
    decisions and whichever hazard, if either fired, is holding or
    redirecting the pipeline. The two "No" lines expand into an
    explanatory block instead of just flipping to "Yes" when a stall or
    a flush is actually happening, matching section 8's own two worked
    examples of what an active hazard looks like in this trace.
    """
    lines = [f"Cycle {trace.cycle}", ""]
    lines.append(_stage_line("IF", trace.if_stage.valid, trace.if_stage.pc, trace.if_stage.instruction))
    lines.append(_stage_line("ID", trace.id_stage.valid, trace.id_stage.pc, trace.id_stage.instruction))
    lines.append(_stage_line("EX", trace.ex_stage.valid, trace.ex_stage.pc, trace.ex_stage.raw))
    lines.append(_stage_line("MEM", trace.mem_stage.valid, trace.mem_stage.pc, trace.mem_stage.raw))
    lines.append(_stage_line("WB", trace.wb_stage.valid, trace.wb_stage.pc, trace.wb_stage.raw))
    lines.append("")

    lines.append(f"Forward A: {_FORWARD_LABELS[trace.forward_a]}")
    lines.append(f"Forward B: {_FORWARD_LABELS[trace.forward_b]}")

    if trace.stall:
        lines.append(f"STALL: {trace.stall_reason}")
        lines.append("bubble inserted into EX")
    else:
        lines.append("Stall:     No")

    if trace.flush:
        lines.append("BRANCH TAKEN")
        lines.append("flush IF/ID")
        lines.append("flush ID/EX")
        lines.append(f"next PC = {trace.branch_target:#010x}")
    else:
        lines.append("Flush:     No")

    return "\n".join(lines)


def main():
    """Both worked examples: section 31's verbose ADD trace, and
    section 32's compact three-instruction loop trace. Imports cpu.py
    locally -- cpu.py imports StepTrace from this file, so a top-level
    import here would be circular.
    """
    from cpu_simulator.cpu.cpu import CPU
    from cpu_simulator.isa.assembler import assemble

    print("=" * 60)
    print("SECTION 31: verbose trace")
    print("=" * 60)
    cpu = CPU()
    cpu.registers.write(6, 10)
    cpu.registers.write(7, 7)
    cpu.pc.jump(0x20)
    cpu.instruction_memory.write_word(0x20, assemble("add x5, x6, x7")[0])
    cpu.step()
    print(format_verbose(cpu.last_trace))

    print()
    print("=" * 60)
    print("SECTION 32: compact trace")
    print("=" * 60)
    cpu = CPU()
    cpu.registers.write(6, 10)
    cpu.registers.write(7, 7)
    cpu.pc.jump(0x20)
    program = """
add  x5, x6, x7
addi x6, x6, -1
bne  x6, x0, -4
    """
    words = assemble(program)
    for i, word in enumerate(words):
        cpu.instruction_memory.write_word(0x20 + i * 4, word)
    for _ in range(3):
        cpu.step()
        print(format_compact(cpu.last_trace))
    print("=" * 60)


if __name__ == "__main__":
    main()
