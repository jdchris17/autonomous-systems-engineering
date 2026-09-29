"""Module 7's control unit: converts a decoded Instruction into the
explicit control signals a datapath would use to configure its muxes,
the register file's write-enable, memory's read/write-enable, and which
operation the ALU runs this cycle.

A structured ControlSignals object, not a loose dict -- a typo in a
dict key ("reg_write" vs "regwrite") fails silently at the point of
use, wherever that happens to be; a typo in a dataclass field fails
immediately, at construction, which is exactly where a control-unit bug
should surface.

Split into two functions, mirroring real processor design:

    main control   looks *only* at opcode. It decides broad behavior --
                    does this instruction write a register? read or
                    write memory? is the ALU's second operand a
                    register or an immediate? -- without knowing or
                    caring what ADD vs SUB vs AND even means. Opcode
                    alone is enough to know "this is an R-type," not
                    enough to know which R-type operation it is.

    ALU control     the reverse: given main control's coarse category
                     plus funct3 (and funct7, for the instructions that
                     have one) -- the same fields decode() already put
                     on the Instruction -- picks the one exact ALU
                     operation (from alu.py's OPERATIONS) the datapath
                     should actually run.

ALU control's funct3/funct7 -> operation tables are not redeclared here.
They're decoder.py's own R_TYPE_MNEMONICS, SHIFT_IMMEDIATE_MNEMONICS,
and I_TYPE_ARITH_MNEMONICS, reused a *fourth* time (encoder.py's
reverse-lookup tables were the second use of this knowledge,
assembler.py's mnemonic-grouping sets were the third). A control unit
that didn't know decode()'s own funct3/funct7 -> mnemonic mapping
wouldn't be mirroring the same hardware decode() already models --
it would just be a second, differently-shaped copy of the same facts,
one more place for the two to quietly disagree.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    # See core/adder.py's module docstring for why this is needed to run
    # this file directly (e.g. VS Code's Run button) rather than via
    # `python -m`. Three .parent calls: control.py -> cpu/ ->
    # cpu_simulator/ -> module-1/.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.isa.decoder import (
    I_TYPE_ARITH_MNEMONICS,
    OPCODE_AUIPC,
    OPCODE_BRANCH,
    OPCODE_I_TYPE_ARITH,
    OPCODE_JAL,
    OPCODE_JALR,
    OPCODE_LOAD,
    OPCODE_LUI,
    OPCODE_R_TYPE,
    OPCODE_STORE,
    R_TYPE_MNEMONICS,
    SHIFT_IMMEDIATE_MNEMONICS,
)
from cpu_simulator.isa.instruction import Instruction


@dataclass
class ControlSignals:
    reg_write: bool
    mem_read: bool
    mem_write: bool
    alu_src: str             # "REGISTER" or "IMMEDIATE" -- the ALU's second operand source
    alu_op: str               # the resolved operation, e.g. "ADD"/"SUB"/"SLT" (one of alu.py's OPERATIONS)
    result_src: str | None    # "ALU", "MEMORY", or "PC_PLUS_4" -- what gets written to rd; None if reg_write is False
    branch: str | None        # the branch mnemonic (BEQ/BNE/...) for the branch unit, or None
    jump: bool


def generate_control(instruction: Instruction) -> ControlSignals:
    """The full control unit: main control's coarse decision, then ALU
    control resolving it into the exact operation the ALU runs.
    """
    main = _generate_main_control(instruction)
    alu_op = _generate_alu_control(main.alu_op_category, instruction)
    return ControlSignals(
        reg_write=main.reg_write,
        mem_read=main.mem_read,
        mem_write=main.mem_write,
        alu_src=main.alu_src,
        alu_op=alu_op,
        result_src=main.result_src,
        branch=main.branch,
        jump=main.jump,
    )


@dataclass
class _MainControlSignals:
    """Everything main control decides before consulting funct3/funct7.
    alu_op_category is the coarse hint ALU control resolves into an
    actual operation -- it is deliberately not itself a real ALU
    operation (alu.py has no "R_TYPE" or "I_TYPE" op).
    """
    reg_write: bool
    mem_read: bool
    mem_write: bool
    alu_src: str
    alu_op_category: str | None  # "ADD" / "R_TYPE" / "I_TYPE" / "BRANCH" / None
    result_src: str | None
    branch: str | None
    jump: bool


def _generate_main_control(instruction: Instruction) -> _MainControlSignals:
    """Looks only at opcode -- never funct3/funct7. "opcode = LOAD"
    alone is enough to know this instruction reads memory and writes a
    register; it takes funct3 to know *which* load (LW vs LB vs LH),
    and this function never needs to ask that question.
    """
    opcode = instruction.opcode

    if opcode == OPCODE_R_TYPE:
        return _MainControlSignals(
            reg_write=True, mem_read=False, mem_write=False,
            alu_src="REGISTER", alu_op_category="R_TYPE",
            result_src="ALU", branch=None, jump=False,
        )
    if opcode == OPCODE_I_TYPE_ARITH:
        return _MainControlSignals(
            reg_write=True, mem_read=False, mem_write=False,
            alu_src="IMMEDIATE", alu_op_category="I_TYPE",
            result_src="ALU", branch=None, jump=False,
        )
    if opcode == OPCODE_LOAD:
        return _MainControlSignals(
            reg_write=True, mem_read=True, mem_write=False,
            alu_src="IMMEDIATE", alu_op_category="ADD",
            result_src="MEMORY", branch=None, jump=False,
        )
    if opcode == OPCODE_STORE:
        return _MainControlSignals(
            reg_write=False, mem_read=False, mem_write=True,
            alu_src="IMMEDIATE", alu_op_category="ADD",
            result_src=None, branch=None, jump=False,
        )
    if opcode == OPCODE_BRANCH:
        return _MainControlSignals(
            reg_write=False, mem_read=False, mem_write=False,
            alu_src="REGISTER", alu_op_category="BRANCH",
            result_src=None, branch=instruction.mnemonic, jump=False,
        )
    if opcode == OPCODE_JAL:
        # rd = pc + 4; the target address (pc + immediate) is a
        # separate adder in a real datapath, not the same ALU output
        # that feeds result_src -- so this instruction's register
        # write-back doesn't route through the ALU at all.
        return _MainControlSignals(
            reg_write=True, mem_read=False, mem_write=False,
            alu_src="IMMEDIATE", alu_op_category=None,
            result_src="PC_PLUS_4", branch=None, jump=True,
        )
    if opcode == OPCODE_JALR:
        # Same rd = pc + 4 as JAL, but the *target* is rs1 + immediate,
        # which the ALU does compute -- just not for result_src, which
        # is still the return address, not the jump target.
        return _MainControlSignals(
            reg_write=True, mem_read=False, mem_write=False,
            alu_src="IMMEDIATE", alu_op_category="ADD",
            result_src="PC_PLUS_4", branch=None, jump=True,
        )
    if opcode == OPCODE_LUI:
        # rd = immediate directly. Routed through the ALU as
        # 0 + immediate anyway, the same reasoning cpu.py's
        # step_load_immediate already uses for LOAD_IMM: there's no
        # dedicated load-immediate circuit, just the adder fed a
        # hardwired zero.
        return _MainControlSignals(
            reg_write=True, mem_read=False, mem_write=False,
            alu_src="IMMEDIATE", alu_op_category="ADD",
            result_src="ALU", branch=None, jump=False,
        )
    if opcode == OPCODE_AUIPC:
        # rd = pc + immediate. Which value feeds the ALU's *first*
        # operand (register file vs pc) is itself a mux a full datapath
        # would need another control signal to select -- out of scope
        # for this file, which only decides alu_src (the *second*
        # operand's source) and which operation the ALU runs.
        return _MainControlSignals(
            reg_write=True, mem_read=False, mem_write=False,
            alu_src="IMMEDIATE", alu_op_category="ADD",
            result_src="ALU", branch=None, jump=False,
        )

    raise ValueError(f"unsupported opcode {opcode:#09b} for control generation")


# R-type and shift-immediate ops read straight off decoder.py's own
# (funct3, funct7) -> mnemonic tables -- for R-type, that mnemonic
# already *is* the alu.py operation name.
_R_TYPE_ALU_OPS = dict(R_TYPE_MNEMONICS)

# Shift-immediate mnemonics carry a trailing "I" (SLLI/SRLI/SRAI) that
# alu.py's operations don't (SLL/SRL/SRA) -- strip it rather than
# hand-maintain a second table that says the same thing differently.
_SHIFT_IMMEDIATE_ALU_OPS = {fields: mnemonic[:-1]
                            for fields, mnemonic in SHIFT_IMMEDIATE_MNEMONICS.items()}

# Same trick for the non-shift I-type arithmetic ops -- funct3 alone,
# no funct7 needed -- but RISC-V's own naming isn't fully regular:
# ADDI/SLTI/XORI/ORI/ANDI are all "base name + trailing I", but SLTIU
# inserts the I *before* the trailing U (SLT + I + U), so a blind
# "strip the last character" gets it wrong ("SLTIU"[:-1] == "SLTI", not
# "SLTU"). Spelled out explicitly rather than guessed at with a pattern
# that only covers 5 of the 6 cases -- the funct3 keys still come from
# I_TYPE_ARITH_MNEMONICS, so this can't drift out of sync with *which*
# funct3 means *which* instruction, only the spelling is hand-written.
_I_TYPE_MNEMONIC_TO_ALU_OP = {
    "ADDI": "ADD", "SLTI": "SLT", "SLTIU": "SLTU",
    "XORI": "XOR", "ORI": "OR", "ANDI": "AND",
}
_I_TYPE_ARITH_ALU_OPS = {funct3: _I_TYPE_MNEMONIC_TO_ALU_OP[mnemonic]
                         for funct3, mnemonic in I_TYPE_ARITH_MNEMONICS.items()}

_SHIFT_IMMEDIATE_FUNCT3 = {0b001, 0b101}

_BRANCH_ALU_OPS = {
    # funct3: the ALU operation the branch unit actually runs to form
    # the comparison. BEQ/BNE subtract and look at the zero flag;
    # BLT/BGE and BLTU/BGEU run the ALU's own SLT/SLTU circuit directly
    # and look at its 0/1 result -- there is no separate "is less than"
    # ALU mode beyond the SLT/SLTU alu.py already has.
    0b000: "SUB",   # BEQ
    0b001: "SUB",   # BNE
    0b100: "SLT",   # BLT
    0b101: "SLT",   # BGE
    0b110: "SLTU",  # BLTU
    0b111: "SLTU",  # BGEU
}


def _generate_alu_control(category: str | None, instruction: Instruction) -> str:
    """Combines main control's coarse category with funct3 (and funct7,
    where the instruction has one) to pick the exact ALU operation.
    """
    if category == "ADD":
        return "ADD"
    if category == "R_TYPE":
        return _R_TYPE_ALU_OPS[(instruction.funct3, instruction.funct7)]
    if category == "I_TYPE":
        if instruction.funct3 in _SHIFT_IMMEDIATE_FUNCT3:
            return _SHIFT_IMMEDIATE_ALU_OPS[(instruction.funct3, instruction.funct7)]
        return _I_TYPE_ARITH_ALU_OPS[instruction.funct3]
    if category == "BRANCH":
        return _BRANCH_ALU_OPS[instruction.funct3]
    if category is None:
        # JAL: the ALU isn't actually driving this instruction's
        # result (result_src is PC_PLUS_4, not ALU) -- alu_op still
        # needs *some* value since ControlSignals.alu_op isn't
        # Optional, so this is an unused placeholder, not a real signal.
        return "ADD"

    raise ValueError(f"unrecognized ALU op category {category!r}")


def main():
    """The two section-26 worked examples: ADD (R-type) and LW (load)."""
    from cpu_simulator.isa.assembler import assemble_line
    from cpu_simulator.isa.decoder import decode

    print("=" * 60)
    print("ADD x5, x6, x7")
    print("=" * 60)
    control = generate_control(decode(assemble_line("add x5, x6, x7")))
    for field in ("reg_write", "mem_read", "mem_write", "alu_src", "alu_op", "result_src"):
        print(f"  {field:<10} = {getattr(control, field)}")
    print()

    print("=" * 60)
    print("LW x5, 12(x2)")
    print("=" * 60)
    control = generate_control(decode(assemble_line("lw x5, 12(x2)")))
    for field in ("reg_write", "mem_read", "mem_write", "alu_src", "alu_op", "result_src"):
        print(f"  {field:<10} = {getattr(control, field)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
