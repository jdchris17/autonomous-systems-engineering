"""Turning a raw 32-bit word into meaningful fields -- the application
Module 1's bit-mask/shift exercises (bits.py's extract_bits, numbers.py's
twos_complement_to_signed) were actually preparing for. Nothing in this
file uses Python's own >>/&; every field comes out through extract_bits,
and every signed immediate is reconstructed through
twos_complement_to_signed rather than any manual sign-extension trick.

Field positions (fixed for every RV32I format):

    opcode  = bits [6:0]     (always here, decides everything else)
    rd      = bits [11:7]    (R/I/U/J -- not S or B, which have no
                               destination register)
    funct3  = bits [14:12]   (R/I/S/B -- not U or J, which have no
                               sub-opcode field at all)
    rs1     = bits [19:15]   (R/I/S/B -- not U or J)
    rs2     = bits [24:20]   (R/S/B only)
    funct7  = bits [31:25]   (R-type only, as a genuine funct7; I-type
                               shifts reuse this position to distinguish
                               SRLI from SRAI, the same way funct7 tells
                               SRL from SRA apart in R-type)

Everything else -- which bits are the immediate, and in what order they
get reassembled -- differs by format, which is exactly the "meaningful
fields" part: the ISA doesn't hand you a 32-bit immediate laid out
conveniently, it scrambles S/B/J's immediate bits across the word so
that as many fields as possible stay in the *same* position across
formats (rs1, rs2, and the low immediate bits of S/B all sit where
R-type's rs1/rs2/rd sit) -- that's a real hardware decision (it lets one
set of wires feed the register file and ALU regardless of format), not
an arbitrary one.

Only the RV32I subset this course targets is supported: R-type ALU ops,
I-type arithmetic (including the SLLI/SRLI/SRAI shift-immediate special
case), LW, SW, BEQ/BNE/BLT/BGE/BLTU/BGEU, JAL, JALR, LUI, AUIPC. Byte/
halfword loads and stores are deliberately not decoded yet -- "eventually
we can add byte/halfword operations."

Immediate extraction is deliberately five separate functions
(decode_i_immediate, decode_s_immediate, decode_b_immediate,
decode_u_immediate, decode_j_immediate), not one "grab the immediate
bits" function with a format argument. Assuming every format's immediate
sits at one contiguous range would be wrong for three of the five --
S/B/J all scramble their immediate bits across non-adjacent positions
(see each function's own docstring for why), so a single generic
implementation would either be wrong for those three or would have to
smuggle the scrambling logic back in through a pile of conditionals,
which is just this same five-way split with worse names.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    # See core/adder.py's module docstring for why this is needed to run
    # this file directly (e.g. VS Code's Run button) rather than via
    # `python -m`. Three .parent calls: decoder.py -> isa/ ->
    # cpu_simulator/ -> module-1/.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.core.bits import extract_bits
from cpu_simulator.core.numbers import decimal_to_binary, twos_complement_to_signed
from cpu_simulator.isa.instruction import Instruction

# ---------------------------------------------------------------------------
# Opcodes (bits [6:0])
# ---------------------------------------------------------------------------
OPCODE_R_TYPE = 0b0110011       # register-register ALU ops
OPCODE_I_TYPE_ARITH = 0b0010011  # immediate ALU ops (ADDI, ANDI, ..., SLLI, ...)
OPCODE_LOAD = 0b0000011          # LW
OPCODE_STORE = 0b0100011         # SW
OPCODE_BRANCH = 0b1100011        # BEQ, BNE, BLT, BGE, BLTU, BGEU
OPCODE_JAL = 0b1101111
OPCODE_JALR = 0b1100111
OPCODE_LUI = 0b0110111
OPCODE_AUIPC = 0b0010111

# ---------------------------------------------------------------------------
# funct3 (+ funct7 where the opcode alone is ambiguous)
# ---------------------------------------------------------------------------
R_TYPE_MNEMONICS = {
    # (funct3, funct7): mnemonic
    (0b000, 0b0000000): "ADD",
    (0b000, 0b0100000): "SUB",
    (0b001, 0b0000000): "SLL",
    (0b010, 0b0000000): "SLT",
    (0b011, 0b0000000): "SLTU",
    (0b100, 0b0000000): "XOR",
    (0b101, 0b0000000): "SRL",
    (0b101, 0b0100000): "SRA",
    (0b110, 0b0000000): "OR",
    (0b111, 0b0000000): "AND",
}

I_TYPE_ARITH_MNEMONICS = {
    # funct3: mnemonic -- excludes 0b001/0b101, which are SLLI/SRLI/SRAI
    # and need the funct7-style discriminator handled separately below.
    0b000: "ADDI",
    0b010: "SLTI",
    0b011: "SLTIU",
    0b100: "XORI",
    0b110: "ORI",
    0b111: "ANDI",
}

SHIFT_IMMEDIATE_MNEMONICS = {
    # (funct3, funct7): mnemonic -- RV32's shift-immediate instructions
    # only ever shift by 0-31, so the shift amount fits in 5 bits (bits
    # [24:20], the same position R-type's rs2 occupies) with bits
    # [31:25] left over to act as a funct7 exactly like R-type's SRL/SRA.
    (0b001, 0b0000000): "SLLI",
    (0b101, 0b0000000): "SRLI",
    (0b101, 0b0100000): "SRAI",
}

BRANCH_MNEMONICS = {
    0b000: "BEQ",
    0b001: "BNE",
    0b100: "BLT",
    0b101: "BGE",
    0b110: "BLTU",
    0b111: "BGEU",
}

LOAD_MNEMONICS = {
    0b010: "LW",
}

STORE_MNEMONICS = {
    0b010: "SW",
}

JALR_FUNCT3 = 0b000


def decode(raw: int) -> Instruction:
    """Decode one raw 32-bit instruction word."""
    opcode = extract_bits(raw, 0, 7)

    if opcode == OPCODE_R_TYPE:
        return _decode_r_type(raw, opcode)
    if opcode == OPCODE_I_TYPE_ARITH:
        return _decode_i_type_arith(raw, opcode)
    if opcode == OPCODE_LOAD:
        return _decode_load(raw, opcode)
    if opcode == OPCODE_JALR:
        return _decode_jalr(raw, opcode)
    if opcode == OPCODE_STORE:
        return _decode_store(raw, opcode)
    if opcode == OPCODE_BRANCH:
        return _decode_branch(raw, opcode)
    if opcode == OPCODE_JAL:
        return _decode_jal(raw, opcode)
    if opcode in (OPCODE_LUI, OPCODE_AUIPC):
        return _decode_u_type(raw, opcode)

    raise ValueError(f"unsupported opcode {opcode:#09b} in instruction {raw:#010x}")


# ---------------------------------------------------------------------------
# R-type: funct7 | rs2 | rs1 | funct3 | rd | opcode
# ---------------------------------------------------------------------------
def _decode_r_type(raw: int, opcode: int) -> Instruction:
    rd = extract_bits(raw, 7, 5)
    funct3 = extract_bits(raw, 12, 3)
    rs1 = extract_bits(raw, 15, 5)
    rs2 = extract_bits(raw, 20, 5)
    funct7 = extract_bits(raw, 25, 7)

    mnemonic = R_TYPE_MNEMONICS.get((funct3, funct7))
    if mnemonic is None:
        raise ValueError(
            f"unrecognized R-type funct3={funct3:#05b}/funct7={funct7:#09b} "
            f"in instruction {raw:#010x}"
        )
    return Instruction(raw=raw, mnemonic=mnemonic, opcode=opcode, format="R",
                        rd=rd, rs1=rs1, rs2=rs2, funct3=funct3, funct7=funct7)


# ---------------------------------------------------------------------------
# I-type: imm[11:0] | rs1 | funct3 | rd | opcode
# (ADDI/SLTI/SLTIU/XORI/ORI/ANDI, plus SLLI/SRLI/SRAI's shift-amount case)
# ---------------------------------------------------------------------------
def _decode_i_type_arith(raw: int, opcode: int) -> Instruction:
    rd = extract_bits(raw, 7, 5)
    funct3 = extract_bits(raw, 12, 3)
    rs1 = extract_bits(raw, 15, 5)

    if funct3 in (0b001, 0b101):
        shift_amount = extract_bits(raw, 20, 5)
        funct7 = extract_bits(raw, 25, 7)
        mnemonic = SHIFT_IMMEDIATE_MNEMONICS.get((funct3, funct7))
        if mnemonic is None:
            raise ValueError(
                f"unrecognized shift-immediate funct3={funct3:#05b}/"
                f"funct7={funct7:#09b} in instruction {raw:#010x}"
            )
        return Instruction(raw=raw, mnemonic=mnemonic, opcode=opcode, format="I",
                            rd=rd, rs1=rs1, funct3=funct3, funct7=funct7,
                            immediate=shift_amount)

    mnemonic = I_TYPE_ARITH_MNEMONICS.get(funct3)
    if mnemonic is None:
        raise ValueError(f"unrecognized I-type funct3={funct3:#05b} "
                          f"in instruction {raw:#010x}")
    immediate = decode_i_immediate(raw)
    return Instruction(raw=raw, mnemonic=mnemonic, opcode=opcode, format="I",
                        rd=rd, rs1=rs1, funct3=funct3, immediate=immediate)


def _decode_load(raw: int, opcode: int) -> Instruction:
    rd = extract_bits(raw, 7, 5)
    funct3 = extract_bits(raw, 12, 3)
    rs1 = extract_bits(raw, 15, 5)

    mnemonic = LOAD_MNEMONICS.get(funct3)
    if mnemonic is None:
        raise ValueError(f"unrecognized load funct3={funct3:#05b} "
                          f"in instruction {raw:#010x}")
    immediate = decode_i_immediate(raw)
    return Instruction(raw=raw, mnemonic=mnemonic, opcode=opcode, format="I",
                        rd=rd, rs1=rs1, funct3=funct3, immediate=immediate)


def _decode_jalr(raw: int, opcode: int) -> Instruction:
    rd = extract_bits(raw, 7, 5)
    funct3 = extract_bits(raw, 12, 3)
    rs1 = extract_bits(raw, 15, 5)

    if funct3 != JALR_FUNCT3:
        raise ValueError(f"unrecognized JALR funct3={funct3:#05b} "
                          f"in instruction {raw:#010x}")
    immediate = decode_i_immediate(raw)
    return Instruction(raw=raw, mnemonic="JALR", opcode=opcode, format="I",
                        rd=rd, rs1=rs1, funct3=funct3, immediate=immediate)


def decode_i_immediate(raw: int) -> int:
    """I-type: imm[11:0] = raw bits [31:20], contiguous and already in
    the right order -- the simplest of the five, and the only one that
    really is "just grab a range of bits," which is exactly why the
    other four can't share this implementation.
    """
    return _sign_extend(extract_bits(raw, 20, 12), width=12)


# ---------------------------------------------------------------------------
# S-type: imm[11:5] | rs2 | rs1 | funct3 | imm[4:0] | opcode
# No rd -- a store has nothing to write to a register.
# ---------------------------------------------------------------------------
def _decode_store(raw: int, opcode: int) -> Instruction:
    funct3 = extract_bits(raw, 12, 3)
    rs1 = extract_bits(raw, 15, 5)
    rs2 = extract_bits(raw, 20, 5)

    mnemonic = STORE_MNEMONICS.get(funct3)
    if mnemonic is None:
        raise ValueError(f"unrecognized store funct3={funct3:#05b} "
                          f"in instruction {raw:#010x}")

    immediate = decode_s_immediate(raw)
    return Instruction(raw=raw, mnemonic=mnemonic, opcode=opcode, format="S",
                        rs1=rs1, rs2=rs2, funct3=funct3, immediate=immediate)


def decode_s_immediate(raw: int) -> int:
    """S-type: imm[11:5] = raw bits [31:25], imm[4:0] = raw bits [11:7].
    Split in two (not contiguous) so that rs1/rs2/funct3 can sit in the
    same positions as R-type's -- one set of wires feeds the register
    file regardless of format, and this split is the cost of that.
    """
    imm_4_0 = extract_bits(raw, 7, 5)
    imm_11_5 = extract_bits(raw, 25, 7)
    combined = (imm_11_5 << 5) | imm_4_0
    return _sign_extend(combined, width=12)


# ---------------------------------------------------------------------------
# B-type: imm[12|10:5] | rs2 | rs1 | funct3 | imm[4:1|11] | opcode
# ---------------------------------------------------------------------------
def _decode_branch(raw: int, opcode: int) -> Instruction:
    funct3 = extract_bits(raw, 12, 3)
    rs1 = extract_bits(raw, 15, 5)
    rs2 = extract_bits(raw, 20, 5)

    mnemonic = BRANCH_MNEMONICS.get(funct3)
    if mnemonic is None:
        raise ValueError(f"unrecognized branch funct3={funct3:#05b} "
                          f"in instruction {raw:#010x}")

    immediate = decode_b_immediate(raw)
    return Instruction(raw=raw, mnemonic=mnemonic, opcode=opcode, format="B",
                        rs1=rs1, rs2=rs2, funct3=funct3, immediate=immediate)


def decode_b_immediate(raw: int) -> int:
    """B-type: imm[12|11|10:5|4:1], bit 0 always 0 (branch targets are
    instruction-aligned, so it's never stored). The bits are scrambled
    -- not stored in their own value order -- specifically so rs1/rs2/
    funct3 stay in the same positions as every other format: a real
    hardware decision (one set of wires feeds the register file and ALU
    regardless of which format is on the bus), and this scrambling is
    what the decoder pays for it.
    """
    imm_12 = extract_bits(raw, 31, 1)
    imm_11 = extract_bits(raw, 7, 1)
    imm_10_5 = extract_bits(raw, 25, 6)
    imm_4_1 = extract_bits(raw, 8, 4)
    combined = (imm_12 << 12) | (imm_11 << 11) | (imm_10_5 << 5) | (imm_4_1 << 1)
    return _sign_extend(combined, width=13)


# ---------------------------------------------------------------------------
# J-type: imm[20|10:1|11|19:12] | rd | opcode
# ---------------------------------------------------------------------------
def _decode_jal(raw: int, opcode: int) -> Instruction:
    rd = extract_bits(raw, 7, 5)
    immediate = decode_j_immediate(raw)
    return Instruction(raw=raw, mnemonic="JAL", opcode=opcode, format="J",
                        rd=rd, immediate=immediate)


def decode_j_immediate(raw: int) -> int:
    """J-type: imm[20|19:12|11|10:1], bit 0 always 0, same scrambling
    idea as B-type -- this time to keep rd in the same position as
    every other rd-producing format instead of rs1/rs2.
    """
    imm_20 = extract_bits(raw, 31, 1)
    imm_19_12 = extract_bits(raw, 12, 8)
    imm_11 = extract_bits(raw, 20, 1)
    imm_10_1 = extract_bits(raw, 21, 10)
    combined = (imm_20 << 20) | (imm_19_12 << 12) | (imm_11 << 11) | (imm_10_1 << 1)
    return _sign_extend(combined, width=21)


# ---------------------------------------------------------------------------
# U-type: imm[31:12] | rd | opcode
# No funct3, no rs1/rs2 -- LUI/AUIPC only ever produce a value from an
# immediate (plus, for AUIPC, the current pc), nothing else feeds in.
# ---------------------------------------------------------------------------
def _decode_u_type(raw: int, opcode: int) -> Instruction:
    rd = extract_bits(raw, 7, 5)
    mnemonic = "LUI" if opcode == OPCODE_LUI else "AUIPC"
    immediate = decode_u_immediate(raw)
    return Instruction(raw=raw, mnemonic=mnemonic, opcode=opcode, format="U",
                        rd=rd, immediate=immediate)


def decode_u_immediate(raw: int) -> int:
    """U-type: imm[31:12] = raw bits [31:12], contiguous like I-type's,
    but shifted left 12 first -- LUI/AUIPC's immediate always occupies
    the *top* 20 bits of the final 32-bit value, with the low 12
    implicitly zero, not the low 20 bits of a 20-bit number.
    """
    imm_31_12 = extract_bits(raw, 12, 20)
    combined = imm_31_12 << 12
    return _sign_extend(combined, width=32)


def _sign_extend(bits_value: int, width: int) -> int:
    """Interpret an unsigned `width`-bit field as a two's-complement
    signed value -- the exact job numbers.py's twos_complement_to_signed
    was built for. Every scrambled/reassembled immediate above still
    ends up going through this single choke point, rather than each
    format inventing its own sign-extension.
    """
    return twos_complement_to_signed(decimal_to_binary(bits_value, width=width))


def to_asm(instruction: Instruction) -> str:
    """Render a decoded Instruction back as readable RISC-V assembly
    text -- decode()'s mirror image: meaningful fields back into the
    human-readable form a real assembly listing uses. One line per
    format, since each format's operand order is its own convention
    (a store's "register, offset(base)" isn't a load's "register,
    register, register").
    """
    m = instruction.mnemonic.lower()
    fmt = instruction.format

    if fmt == "R":
        return f"{m} x{instruction.rd}, x{instruction.rs1}, x{instruction.rs2}"
    if fmt == "I":
        if instruction.opcode == OPCODE_LOAD:
            return f"{m} x{instruction.rd}, {instruction.immediate}(x{instruction.rs1})"
        return f"{m} x{instruction.rd}, x{instruction.rs1}, {instruction.immediate}"
    if fmt == "S":
        return f"{m} x{instruction.rs2}, {instruction.immediate}(x{instruction.rs1})"
    if fmt == "B":
        return f"{m} x{instruction.rs1}, x{instruction.rs2}, {instruction.immediate}"
    if fmt in ("U", "J"):
        return f"{m} x{instruction.rd}, {instruction.immediate}"

    raise ValueError(f"unrecognized instruction format {fmt!r}")


def main():
    """The section-39 worked example: turn one raw machine word back
    into meaning, field by field, then all the way to assembly text.
    """
    raw = 0x007302B3
    inst = decode(raw)

    print("=" * 50)
    print(f"DECODE {raw:#010X}")
    print("=" * 50)
    print(f"raw:    {inst.raw:#010x}")
    print()
    print(f"opcode: {inst.opcode:07b}")
    print(f"rd:     {inst.rd:05b} = x{inst.rd}")
    print(f"funct3: {inst.funct3:03b}")
    print(f"rs1:    {inst.rs1:05b} = x{inst.rs1}")
    print(f"rs2:    {inst.rs2:05b} = x{inst.rs2}")
    print(f"funct7: {inst.funct7:07b}")
    print()
    print(f"opcode + funct3 + funct7  ->  {inst.mnemonic}")
    print(f"identifies:  {to_asm(inst)}")
    print("=" * 50)
    print("A raw machine word, turned back into meaning.")
    print("=" * 50)


if __name__ == "__main__":
    main()
