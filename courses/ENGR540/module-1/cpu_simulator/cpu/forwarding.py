"""The forwarding unit. One job, per spec: determine the actual
operands entering EX. Not "fix hazards" in general -- that's a bigger
job split across this file (data hazards a bypass can resolve) and
hazards.py (the ones that need a stall instead, not built yet). Keeping
forwarding this narrowly scoped is what lets it stay ignorant of the
ALU, and lets alu.py stay ignorant of pipelining.

Two functions, deliberately separate:

    determine_forwarding()  the decision -- which of three sources
                             should feed a read of register `rs`
    select()                 the mux -- given that decision, produce
                              the actual value

forward_operands() is the convenience wrapper that runs both for both
of EX's operands at once, which is what a pipelined CPU's EX stage
actually calls.

Priority: EX_MEM before MEM_WB. EX_MEM holds whatever instruction is
one cycle closer to EX than MEM_WB's instruction -- it produced its
result more recently, so if both would-be sources have already written
(or are about to write) the same register, EX_MEM's is the one that
should win. Section: "the most recent value should win."
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    # See core/adder.py's module docstring for why this is needed to run
    # this file directly (e.g. VS Code's Run button) rather than via
    # `python -m`. Three .parent calls: forwarding.py -> cpu/ ->
    # cpu_simulator/ -> module-1/.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.cpu.pipeline import EX_MEM, ID_EX, MEM_WB

REGISTER_VALUE = "REGISTER_VALUE"
EX_MEM_RESULT = "EX_MEM_RESULT"
MEM_WB_RESULT = "MEM_WB_RESULT"


def determine_forwarding(rs: int, ex_mem: EX_MEM, mem_wb: MEM_WB) -> str:
    """Which source should feed a read of register `rs`.

    x0 never forwards (`rs == 0`): it's hardwired to zero, nothing
    legitimately "produces" a value for it, and RegisterFile already
    discards writes to it regardless -- forwarding into it would be
    both pointless and, if `rs` were used as a bubble's leftover
    default, actively wrong to act on.

    EX_MEM is skipped for a load still sitting in that stage, even if
    its `rd` matches: a load's `EX_MEM.alu_result` is the *address* it
    is about to read, not the value that will eventually be written --
    that value doesn't exist until MEMORY has run, i.e. until this same
    producer reaches MEM_WB one cycle later. Forwarding it now would
    forward the wrong number entirely. (This still doesn't cover a load
    *immediately* followed by a dependent instruction -- that gap is a
    stall, not something forwarding alone can close, and hazard
    detection isn't built yet.)
    """
    if rs == 0:
        return REGISTER_VALUE
    if (ex_mem.valid and ex_mem.rd == rs and ex_mem.control is not None
            and ex_mem.control.reg_write and not ex_mem.control.mem_read):
        return EX_MEM_RESULT
    if (mem_wb.valid and mem_wb.rd == rs and mem_wb.control is not None
            and mem_wb.control.reg_write):
        return MEM_WB_RESULT
    return REGISTER_VALUE


def select(source: str, register_value: int, ex_mem_result: int, mem_wb_result: int) -> int:
    """The mux itself. No decision-making left in it --
    determine_forwarding() already made the decision; this only acts
    on it, which is what keeps forwarding separate from ALU
    functionality (nothing here computes anything, it only chooses
    between three already-computed values).
    """
    if source == EX_MEM_RESULT:
        return ex_mem_result
    if source == MEM_WB_RESULT:
        return mem_wb_result
    if source == REGISTER_VALUE:
        return register_value
    raise ValueError(f"unrecognized forwarding source {source!r}")


def forward_operands(id_ex: ID_EX, ex_mem: EX_MEM, mem_wb: MEM_WB) -> tuple[int, int]:
    """The forwarding unit's one job, run for both of EX's operands.
    `id_ex` is the instruction about to enter EX; `ex_mem`/`mem_wb` are
    the two candidate more-recent producers.

    The value MEM_WB would supply is whichever of its own two fields
    `writeback()` will actually write -- `pc_plus_4` for a `JAL`/`JALR`
    still finishing its writeback, `memory_result` for everything else
    (already the loaded word for a load, or execute()'s own ALU result
    passed straight through for anything else) -- the exact same
    selection `Datapath.writeback()` makes, just read here one stage
    earlier instead of applied to the register file.
    """
    mem_wb_value = (mem_wb.pc_plus_4
                     if mem_wb.control is not None and mem_wb.control.result_src == "PC_PLUS_4"
                     else mem_wb.memory_result)

    forward_a = determine_forwarding(id_ex.rs1, ex_mem, mem_wb)
    forward_b = determine_forwarding(id_ex.rs2, ex_mem, mem_wb)

    operand_a = select(forward_a, id_ex.rs1_value, ex_mem.alu_result, mem_wb_value)
    operand_b = select(forward_b, id_ex.rs2_value, ex_mem.alu_result, mem_wb_value)
    return operand_a, operand_b
