"""The reverse transformation: structured fields back into a raw 32-bit
machine word -- decoder.py's mirror image, one format at a time.

Every lookup table here is built by *inverting* decoder.py's own tables
(R_TYPE_MNEMONICS, I_TYPE_ARITH_MNEMONICS, ...), not by re-declaring the
same funct3/funct7 knowledge a second time. Two independently-typed
tables recording "ADD is funct3=0, funct7=0" would eventually drift out
of sync with each other; one table with its arrows reversed can't.

Assembly is exactly what section 42 describes: shift each field into
its bit position and OR them together.

    (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12)
        | (rd << 7) | opcode

is encode_r_type's entire job. The five immediate-encoding functions
(encode_i_immediate, encode_s_immediate, encode_b_immediate,
encode_u_immediate, encode_j_immediate) are the reverse of decoder.py's
five decode_*_immediate functions -- same reasoning as section 40: a
signed Python int doesn't get scattered back across S/B/J's non-
contiguous bit positions by one generic function any more than it gets
gathered from them by one.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    # See core/adder.py's module docstring for why this is needed to run
    # this file directly (e.g. VS Code's Run button) rather than via
    # `python -m`. Three .parent calls: encoder.py -> isa/ ->
    # cpu_simulator/ -> module-1/.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.isa.decoder import (
    BRANCH_MNEMONICS,
    I_TYPE_ARITH_MNEMONICS,
    JALR_FUNCT3,
    LOAD_MNEMONICS,
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
    STORE_MNEMONICS,
)

# ---------------------------------------------------------------------------
# Reverse lookups: mnemonic -> (funct3[, funct7]), inverted from
# decoder.py's forward tables rather than re-declared.
# ---------------------------------------------------------------------------
_R_TYPE_FIELDS = {mnemonic: fields for fields, mnemonic in R_TYPE_MNEMONICS.items()}
_SHIFT_IMMEDIATE_FIELDS = {mnemonic: fields for fields, mnemonic in SHIFT_IMMEDIATE_MNEMONICS.items()}
_I_TYPE_ARITH_FIELDS = {mnemonic: funct3 for funct3, mnemonic in I_TYPE_ARITH_MNEMONICS.items()}
_LOAD_FIELDS = {mnemonic: funct3 for funct3, mnemonic in LOAD_MNEMONICS.items()}
_STORE_FIELDS = {mnemonic: funct3 for funct3, mnemonic in STORE_MNEMONICS.items()}
_BRANCH_FIELDS = {mnemonic: funct3 for funct3, mnemonic in BRANCH_MNEMONICS.items()}


def encode(mnemonic: str, rd: int | None = None, rs1: int | None = None,
           rs2: int | None = None, immediate: int | None = None) -> int:
    """General dispatcher: any mnemonic decode() can produce, encoded
    back to a raw word. Which of rd/rs1/rs2/immediate are actually used
    depends entirely on the mnemonic -- the same "format decides which
    fields exist" rule Instruction's own fields follow.
    """
    if mnemonic in _R_TYPE_FIELDS:
        return encode_r_type(mnemonic, rd, rs1, rs2)
    if mnemonic in _SHIFT_IMMEDIATE_FIELDS:
        return encode_shift_immediate(mnemonic, rd, rs1, immediate)
    if mnemonic in _I_TYPE_ARITH_FIELDS:
        return encode_i_type_arith(mnemonic, rd, rs1, immediate)
    if mnemonic in _LOAD_FIELDS:
        return encode_load(mnemonic, rd, rs1, immediate)
    if mnemonic == "JALR":
        return encode_jalr(rd, rs1, immediate)
    if mnemonic in _STORE_FIELDS:
        return encode_store(mnemonic, rs1, rs2, immediate)
    if mnemonic in _BRANCH_FIELDS:
        return encode_branch(mnemonic, rs1, rs2, immediate)
    if mnemonic == "JAL":
        return encode_jal(rd, immediate)
    if mnemonic in ("LUI", "AUIPC"):
        return encode_u_type(mnemonic, rd, immediate)

    raise ValueError(f"unknown mnemonic {mnemonic!r}")


# ---------------------------------------------------------------------------
# R-type: funct7 | rs2 | rs1 | funct3 | rd | opcode
# ---------------------------------------------------------------------------
def encode_r_type(mnemonic: str, rd: int, rs1: int, rs2: int) -> int:
    _validate_register("rd", rd)
    _validate_register("rs1", rs1)
    _validate_register("rs2", rs2)
    funct3, funct7 = _R_TYPE_FIELDS[mnemonic]
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | OPCODE_R_TYPE


def encode_add(rd: int, rs1: int, rs2: int) -> int:
    """The section-42 worked example: encode_add(5, 6, 7) == 0x007302B3."""
    return encode_r_type("ADD", rd, rs1, rs2)


# ---------------------------------------------------------------------------
# I-type arithmetic, and its SLLI/SRLI/SRAI shift-immediate special case
# ---------------------------------------------------------------------------
def encode_i_type_arith(mnemonic: str, rd: int, rs1: int, immediate: int) -> int:
    _validate_register("rd", rd)
    _validate_register("rs1", rs1)
    funct3 = _I_TYPE_ARITH_FIELDS[mnemonic]
    return encode_i_immediate(immediate) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | OPCODE_I_TYPE_ARITH


def encode_shift_immediate(mnemonic: str, rd: int, rs1: int, shift_amount: int) -> int:
    """SLLI/SRLI/SRAI: `shift_amount` occupies bits [24:20] directly
    (not sign-extended -- it's an unsigned 0-31 shift count, not a
    signed immediate), with funct7 back in its normal position to
    distinguish SRLI from SRAI.
    """
    _validate_register("rd", rd)
    _validate_register("rs1", rs1)
    if not (0 <= shift_amount <= 31):
        raise ValueError(f"shift_amount must be 0-31, got {shift_amount}")
    funct3, funct7 = _SHIFT_IMMEDIATE_FIELDS[mnemonic]
    return (funct7 << 25) | (shift_amount << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | OPCODE_I_TYPE_ARITH


def encode_load(mnemonic: str, rd: int, rs1: int, immediate: int) -> int:
    _validate_register("rd", rd)
    _validate_register("rs1", rs1)
    funct3 = _LOAD_FIELDS[mnemonic]
    return encode_i_immediate(immediate) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | OPCODE_LOAD


def encode_jalr(rd: int, rs1: int, immediate: int) -> int:
    _validate_register("rd", rd)
    _validate_register("rs1", rs1)
    return encode_i_immediate(immediate) | (rs1 << 15) | (JALR_FUNCT3 << 12) | (rd << 7) | OPCODE_JALR


def encode_i_immediate(immediate: int) -> int:
    """I-type: pack a signed value into bits [31:20], already shifted
    into position -- the reverse of decoder.py's decode_i_immediate.
    """
    return (immediate & 0xFFF) << 20


# ---------------------------------------------------------------------------
# S-type: imm[11:5] | rs2 | rs1 | funct3 | imm[4:0] | opcode
# ---------------------------------------------------------------------------
def encode_store(mnemonic: str, rs1: int, rs2: int, immediate: int) -> int:
    _validate_register("rs1", rs1)
    _validate_register("rs2", rs2)
    funct3 = _STORE_FIELDS[mnemonic]
    return encode_s_immediate(immediate) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | OPCODE_STORE


def encode_s_immediate(immediate: int) -> int:
    """S-type: split imm[11:0] back into imm[11:5] at bits [31:25] and
    imm[4:0] at bits [11:7] -- the reverse of decode_s_immediate.
    """
    bits12 = immediate & 0xFFF
    imm_4_0 = bits12 & 0x1F
    imm_11_5 = (bits12 >> 5) & 0x7F
    return (imm_11_5 << 25) | (imm_4_0 << 7)


# ---------------------------------------------------------------------------
# B-type: imm[12|10:5] | rs2 | rs1 | funct3 | imm[4:1|11] | opcode
# ---------------------------------------------------------------------------
def encode_branch(mnemonic: str, rs1: int, rs2: int, immediate: int) -> int:
    _validate_register("rs1", rs1)
    _validate_register("rs2", rs2)
    funct3 = _BRANCH_FIELDS[mnemonic]
    return encode_b_immediate(immediate) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | OPCODE_BRANCH


def encode_b_immediate(immediate: int) -> int:
    """B-type: scatter imm[12:0] (bit 0 assumed 0 -- branch targets are
    instruction-aligned) back across bits [31], [30:25], [11:8], [7] --
    the reverse of decode_b_immediate.
    """
    bits13 = immediate & 0x1FFF
    imm_12 = (bits13 >> 12) & 0x1
    imm_11 = (bits13 >> 11) & 0x1
    imm_10_5 = (bits13 >> 5) & 0x3F
    imm_4_1 = (bits13 >> 1) & 0xF
    return (imm_12 << 31) | (imm_10_5 << 25) | (imm_4_1 << 8) | (imm_11 << 7)


# ---------------------------------------------------------------------------
# J-type: imm[20|10:1|11|19:12] | rd | opcode
# ---------------------------------------------------------------------------
def encode_jal(rd: int, immediate: int) -> int:
    _validate_register("rd", rd)
    return encode_j_immediate(immediate) | (rd << 7) | OPCODE_JAL


def encode_j_immediate(immediate: int) -> int:
    """J-type: scatter imm[20:0] (bit 0 assumed 0) back across bits
    [31], [30:21], [20], [19:12] -- the reverse of decode_j_immediate.
    """
    bits21 = immediate & 0x1FFFFF
    imm_20 = (bits21 >> 20) & 0x1
    imm_19_12 = (bits21 >> 12) & 0xFF
    imm_11 = (bits21 >> 11) & 0x1
    imm_10_1 = (bits21 >> 1) & 0x3FF
    return (imm_20 << 31) | (imm_10_1 << 21) | (imm_11 << 20) | (imm_19_12 << 12)


# ---------------------------------------------------------------------------
# U-type: imm[31:12] | rd | opcode
# ---------------------------------------------------------------------------
def encode_u_type(mnemonic: str, rd: int, immediate: int) -> int:
    _validate_register("rd", rd)
    opcode = OPCODE_LUI if mnemonic == "LUI" else OPCODE_AUIPC
    return encode_u_immediate(immediate) | (rd << 7) | opcode


def encode_u_immediate(immediate: int) -> int:
    """U-type: the top 20 bits of `immediate`, already in position --
    the reverse of decode_u_immediate. The low 12 bits are dropped, not
    validated as zero, the same "storage truncates" reasoning
    register.py's write_next() masks rather than raises.
    """
    return immediate & 0xFFFFF000


def _validate_register(name: str, value: int) -> None:
    if not (0 <= value <= 31):
        raise ValueError(f"{name} must be 0-31, got {value}")


def main():
    """The section-42 worked example, run end to end."""
    raw = encode_add(5, 6, 7)
    print("=" * 50)
    print("ENCODE add x5, x6, x7")
    print("=" * 50)
    print("fields: ADD, rd=5, rs1=6, rs2=7")
    print(f"encode_add(5, 6, 7) = {raw:#010X}")
    print("=" * 50)
    assert raw == 0x007302B3, "encoder disagrees with the section-39 worked example"
    print("Matches the section-39 worked example (0x007302B3) exactly.")
    print("Both directions built: fields -> machine code, machine code -> fields.")
    print("=" * 50)


if __name__ == "__main__":
    main()
