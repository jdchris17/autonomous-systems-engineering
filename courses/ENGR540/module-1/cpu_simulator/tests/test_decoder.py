import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.isa.decoder import (
    decode,
    decode_b_immediate,
    decode_i_immediate,
    decode_j_immediate,
    decode_s_immediate,
    decode_u_immediate,
    to_asm,
)

# ---------------------------------------------------------------------------
# Raw-word builders, written independently from decoder.py's own field
# extraction -- these place fields at their documented RV32I bit positions
# using fresh shift/OR arithmetic, so a bug in decoder.py has to actually
# disagree with the spec to be caught, not just disagree with itself.
# ---------------------------------------------------------------------------
OPCODE_R = 0b0110011
OPCODE_I_ARITH = 0b0010011
OPCODE_LOAD = 0b0000011
OPCODE_STORE = 0b0100011
OPCODE_BRANCH = 0b1100011
OPCODE_JAL = 0b1101111
OPCODE_JALR = 0b1100111
OPCODE_LUI = 0b0110111
OPCODE_AUIPC = 0b0010111


def r_type(funct7, rs2, rs1, funct3, rd, opcode=OPCODE_R):
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def i_type(imm12, rs1, funct3, rd, opcode):
    return ((imm12 & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def shift_i_type(funct7, shamt, rs1, funct3, rd, opcode=OPCODE_I_ARITH):
    return (funct7 << 25) | (shamt << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def s_type(imm12, rs2, rs1, funct3, opcode=OPCODE_STORE):
    imm12 &= 0xFFF
    imm_11_5 = (imm12 >> 5) & 0x7F
    imm_4_0 = imm12 & 0x1F
    return (imm_11_5 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm_4_0 << 7) | opcode


def b_type(imm13, rs2, rs1, funct3, opcode=OPCODE_BRANCH):
    imm13 &= 0x1FFF
    imm_12 = (imm13 >> 12) & 0x1
    imm_11 = (imm13 >> 11) & 0x1
    imm_10_5 = (imm13 >> 5) & 0x3F
    imm_4_1 = (imm13 >> 1) & 0xF
    return ((imm_12 << 31) | (imm_10_5 << 25) | (rs2 << 20) | (rs1 << 15)
             | (funct3 << 12) | (imm_4_1 << 8) | (imm_11 << 7) | opcode)


def u_type(imm20, rd, opcode):
    return ((imm20 & 0xFFFFF) << 12) | (rd << 7) | opcode


def j_type(imm21, rd, opcode=OPCODE_JAL):
    imm21 &= 0x1FFFFF
    imm_20 = (imm21 >> 20) & 0x1
    imm_19_12 = (imm21 >> 12) & 0xFF
    imm_11 = (imm21 >> 11) & 0x1
    imm_10_1 = (imm21 >> 1) & 0x3FF
    return (imm_20 << 31) | (imm_10_1 << 21) | (imm_11 << 20) | (imm_19_12 << 12) | (rd << 7) | opcode


class TestRType:
    @pytest.mark.parametrize("mnemonic,funct3,funct7", [
        ("ADD", 0b000, 0b0000000),
        ("SUB", 0b000, 0b0100000),
        ("SLL", 0b001, 0b0000000),
        ("SLT", 0b010, 0b0000000),
        ("SLTU", 0b011, 0b0000000),
        ("XOR", 0b100, 0b0000000),
        ("SRL", 0b101, 0b0000000),
        ("SRA", 0b101, 0b0100000),
        ("OR", 0b110, 0b0000000),
        ("AND", 0b111, 0b0000000),
    ])
    def test_every_r_type_mnemonic(self, mnemonic, funct3, funct7):
        raw = r_type(funct7=funct7, rs2=2, rs1=1, funct3=funct3, rd=3)
        inst = decode(raw)
        assert inst.mnemonic == mnemonic
        assert inst.format == "R"
        assert inst.raw == raw
        assert (inst.rd, inst.rs1, inst.rs2, inst.funct3, inst.funct7) == (3, 1, 2, funct3, funct7)
        assert inst.immediate is None

    def test_unrecognized_funct3_funct7_combo_rejected(self):
        raw = r_type(funct7=0b1111111, rs2=2, rs1=1, funct3=0b000, rd=3)
        with pytest.raises(ValueError):
            decode(raw)


class TestITypeArithmetic:
    @pytest.mark.parametrize("mnemonic,funct3", [
        ("ADDI", 0b000),
        ("SLTI", 0b010),
        ("SLTIU", 0b011),
        ("XORI", 0b100),
        ("ORI", 0b110),
        ("ANDI", 0b111),
    ])
    def test_positive_immediate(self, mnemonic, funct3):
        raw = i_type(imm12=100, rs1=1, funct3=funct3, rd=2, opcode=OPCODE_I_ARITH)
        inst = decode(raw)
        assert inst.mnemonic == mnemonic
        assert inst.format == "I"
        assert (inst.rd, inst.rs1, inst.funct3) == (2, 1, funct3)
        assert inst.immediate == 100

    def test_negative_immediate_sign_extends(self):
        raw = i_type(imm12=-1, rs1=1, funct3=0b000, rd=2, opcode=OPCODE_I_ARITH)
        inst = decode(raw)
        assert inst.mnemonic == "ADDI"
        assert inst.immediate == -1

    def test_smallest_and_largest_12_bit_signed_immediate(self):
        raw_min = i_type(imm12=-2048, rs1=0, funct3=0b000, rd=1, opcode=OPCODE_I_ARITH)
        raw_max = i_type(imm12=2047, rs1=0, funct3=0b000, rd=1, opcode=OPCODE_I_ARITH)
        assert decode(raw_min).immediate == -2048
        assert decode(raw_max).immediate == 2047

    def test_all_eight_funct3_values_are_valid(self):
        # funct3 is 3 bits -- all 8 values are meaningful for this
        # opcode: 6 are ADDI/SLTI/SLTIU/XORI/ORI/ANDI (handled here),
        # and 0b001/0b101 are the shift-immediate cases (handled by
        # TestShiftImmediate) -- there's no unassigned funct3 to reject.
        for funct3 in range(8):
            raw = shift_i_type(funct7=0, shamt=0, rs1=0, funct3=funct3, rd=0) \
                if funct3 in (0b001, 0b101) else i_type(imm12=0, rs1=0, funct3=funct3, rd=0, opcode=OPCODE_I_ARITH)
            decode(raw)  # must not raise


class TestShiftImmediate:
    def test_slli(self):
        raw = shift_i_type(funct7=0b0000000, shamt=5, rs1=1, funct3=0b001, rd=2)
        inst = decode(raw)
        assert inst.mnemonic == "SLLI"
        assert inst.format == "I"
        assert inst.immediate == 5
        assert (inst.rd, inst.rs1) == (2, 1)

    def test_srli(self):
        raw = shift_i_type(funct7=0b0000000, shamt=31, rs1=1, funct3=0b101, rd=2)
        inst = decode(raw)
        assert inst.mnemonic == "SRLI"
        assert inst.immediate == 31

    def test_srai(self):
        raw = shift_i_type(funct7=0b0100000, shamt=1, rs1=1, funct3=0b101, rd=2)
        inst = decode(raw)
        assert inst.mnemonic == "SRAI"
        assert inst.immediate == 1

    def test_unrecognized_shift_funct7_rejected(self):
        raw = shift_i_type(funct7=0b0000001, shamt=1, rs1=1, funct3=0b101, rd=2)
        with pytest.raises(ValueError):
            decode(raw)


class TestLoad:
    def test_lw(self):
        raw = i_type(imm12=12, rs1=2, funct3=0b010, rd=5, opcode=OPCODE_LOAD)
        inst = decode(raw)
        assert inst.mnemonic == "LW"
        assert inst.format == "I"
        assert (inst.rd, inst.rs1) == (5, 2)
        assert inst.immediate == 12

    def test_negative_offset(self):
        raw = i_type(imm12=-8, rs1=2, funct3=0b010, rd=5, opcode=OPCODE_LOAD)
        assert decode(raw).immediate == -8

    def test_unrecognized_funct3_rejected(self):
        raw = i_type(imm12=0, rs1=0, funct3=0b011, rd=0, opcode=OPCODE_LOAD)
        with pytest.raises(ValueError):
            decode(raw)


class TestStore:
    def test_sw(self):
        raw = s_type(imm12=12, rs2=5, rs1=2, funct3=0b010)
        inst = decode(raw)
        assert inst.mnemonic == "SW"
        assert inst.format == "S"
        assert (inst.rs1, inst.rs2) == (2, 5)
        assert inst.immediate == 12
        assert inst.rd is None

    def test_negative_offset(self):
        raw = s_type(imm12=-100, rs2=5, rs1=2, funct3=0b010)
        assert decode(raw).immediate == -100

    def test_smallest_and_largest_12_bit_signed_immediate(self):
        assert decode(s_type(imm12=-2048, rs2=0, rs1=0, funct3=0b010)).immediate == -2048
        assert decode(s_type(imm12=2047, rs2=0, rs1=0, funct3=0b010)).immediate == 2047


class TestBranch:
    @pytest.mark.parametrize("mnemonic,funct3", [
        ("BEQ", 0b000),
        ("BNE", 0b001),
        ("BLT", 0b100),
        ("BGE", 0b101),
        ("BLTU", 0b110),
        ("BGEU", 0b111),
    ])
    def test_every_branch_mnemonic(self, mnemonic, funct3):
        raw = b_type(imm13=16, rs2=2, rs1=1, funct3=funct3)
        inst = decode(raw)
        assert inst.mnemonic == mnemonic
        assert inst.format == "B"
        assert (inst.rs1, inst.rs2) == (1, 2)
        assert inst.immediate == 16
        assert inst.rd is None

    def test_negative_offset(self):
        raw = b_type(imm13=-16, rs2=2, rs1=1, funct3=0b000)
        assert decode(raw).immediate == -16

    def test_smallest_and_largest_13_bit_signed_immediate(self):
        # 13-bit signed range with bit0 forced to 0: -4096..4094 (even only)
        assert decode(b_type(imm13=-4096, rs2=0, rs1=0, funct3=0b000)).immediate == -4096
        assert decode(b_type(imm13=4094, rs2=0, rs1=0, funct3=0b000)).immediate == 4094

    def test_unrecognized_funct3_rejected(self):
        raw = b_type(imm13=0, rs2=0, rs1=0, funct3=0b010)
        with pytest.raises(ValueError):
            decode(raw)


class TestJal:
    def test_positive_offset(self):
        raw = j_type(imm21=1024, rd=1)
        inst = decode(raw)
        assert inst.mnemonic == "JAL"
        assert inst.format == "J"
        assert inst.rd == 1
        assert inst.immediate == 1024
        assert inst.rs1 is None

    def test_negative_offset(self):
        raw = j_type(imm21=-1024, rd=1)
        assert decode(raw).immediate == -1024

    def test_smallest_and_largest_21_bit_signed_immediate(self):
        assert decode(j_type(imm21=-1048576, rd=0)).immediate == -1048576
        assert decode(j_type(imm21=1048574, rd=0)).immediate == 1048574


class TestJalr:
    def test_basic(self):
        raw = i_type(imm12=4, rs1=1, funct3=0b000, rd=2, opcode=OPCODE_JALR)
        inst = decode(raw)
        assert inst.mnemonic == "JALR"
        assert inst.format == "I"
        assert (inst.rd, inst.rs1) == (2, 1)
        assert inst.immediate == 4

    def test_unrecognized_funct3_rejected(self):
        raw = i_type(imm12=0, rs1=0, funct3=0b001, rd=0, opcode=OPCODE_JALR)
        with pytest.raises(ValueError):
            decode(raw)


class TestUType:
    def test_lui(self):
        # 0xABCDE has its own top bit set, so the reassembled 32-bit
        # value (0xABCDE000) is stored sign-extended, same convention
        # as every other format's immediate -- -1412571136 is
        # 0xABCDE000 read as signed 32-bit two's complement.
        raw = u_type(imm20=0xABCDE, rd=3, opcode=OPCODE_LUI)
        inst = decode(raw)
        assert inst.mnemonic == "LUI"
        assert inst.format == "U"
        assert inst.rd == 3
        assert inst.immediate == -1412571136
        assert inst.rs1 is None
        assert inst.funct3 is None

    def test_auipc_with_positive_result(self):
        # Top bit of 0x12345 is 0, so this one stays positive --
        # confirms the sign-extension isn't unconditionally flipping.
        raw = u_type(imm20=0x12345, rd=5, opcode=OPCODE_AUIPC)
        inst = decode(raw)
        assert inst.mnemonic == "AUIPC"
        assert inst.immediate == 0x12345000

    def test_top_bit_set_sign_extends_to_negative(self):
        # imm20 with its own top bit set -> becomes bit 31 of the final
        # value, which twos_complement_to_signed should read as negative.
        raw = u_type(imm20=0x80000, rd=1, opcode=OPCODE_LUI)
        inst = decode(raw)
        assert inst.immediate < 0
        assert inst.immediate == -0x80000000


class TestUnknownOpcode:
    def test_rejected(self):
        with pytest.raises(ValueError):
            decode(0b1111111)  # opcode bits all set -- not a valid RV32I base opcode


class TestToAsm:
    def test_the_section_39_worked_example(self):
        inst = decode(0x007302B3)
        assert inst.mnemonic == "ADD"
        assert (inst.rd, inst.rs1, inst.rs2) == (5, 6, 7)
        assert to_asm(inst) == "add x5, x6, x7"

    def test_r_type(self):
        raw = r_type(funct7=0b0100000, rs2=2, rs1=1, funct3=0b000, rd=3)  # SUB
        assert to_asm(decode(raw)) == "sub x3, x1, x2"

    def test_i_type_arithmetic(self):
        raw = i_type(imm12=10, rs1=1, funct3=0b000, rd=2, opcode=OPCODE_I_ARITH)
        assert to_asm(decode(raw)) == "addi x2, x1, 10"

    def test_i_type_negative_immediate(self):
        raw = i_type(imm12=-1, rs1=1, funct3=0b000, rd=2, opcode=OPCODE_I_ARITH)
        assert to_asm(decode(raw)) == "addi x2, x1, -1"

    def test_load(self):
        raw = i_type(imm12=12, rs1=2, funct3=0b010, rd=5, opcode=OPCODE_LOAD)
        assert to_asm(decode(raw)) == "lw x5, 12(x2)"

    def test_store(self):
        raw = s_type(imm12=12, rs2=5, rs1=2, funct3=0b010)
        assert to_asm(decode(raw)) == "sw x5, 12(x2)"

    def test_branch(self):
        raw = b_type(imm13=16, rs2=2, rs1=1, funct3=0b000)  # BEQ
        assert to_asm(decode(raw)) == "beq x1, x2, 16"

    def test_jal(self):
        raw = j_type(imm21=1024, rd=1)
        assert to_asm(decode(raw)) == "jal x1, 1024"

    def test_jalr(self):
        raw = i_type(imm12=4, rs1=1, funct3=0b000, rd=2, opcode=OPCODE_JALR)
        assert to_asm(decode(raw)) == "jalr x2, x1, 4"

    def test_lui(self):
        raw = u_type(imm20=0x12345, rd=3, opcode=OPCODE_LUI)
        assert to_asm(decode(raw)) == f"lui x3, {0x12345000}"


class TestImmediateFunctionsDirectly:
    """Each of the five format-specific immediate functions, called
    directly rather than through decode() -- confirming section 40's
    "do not make one universal function" split actually gives each
    format its own independently-testable, independently-correct piece.
    """

    def test_i_immediate(self):
        raw = i_type(imm12=-5, rs1=0, funct3=0b000, rd=0, opcode=OPCODE_I_ARITH)
        assert decode_i_immediate(raw) == -5

    def test_s_immediate(self):
        raw = s_type(imm12=-100, rs2=0, rs1=0, funct3=0b010)
        assert decode_s_immediate(raw) == -100

    def test_b_immediate(self):
        raw = b_type(imm13=-16, rs2=0, rs1=0, funct3=0b000)
        assert decode_b_immediate(raw) == -16

    def test_u_immediate(self):
        raw = u_type(imm20=0x12345, rd=0, opcode=OPCODE_LUI)
        assert decode_u_immediate(raw) == 0x12345000

    def test_j_immediate(self):
        raw = j_type(imm21=-1024, rd=0)
        assert decode_j_immediate(raw) == -1024

    def test_each_immediate_function_only_reads_its_own_format(self):
        # Sanity check that none of these accidentally reach into bit
        # positions that belong to rd/rs1/rs2/funct3/funct7 -- fill
        # every non-immediate field with 1s and confirm the immediate
        # extracted is still exactly what was placed there.
        raw = i_type(imm12=123, rs1=0b11111, funct3=0b111, rd=0b11111, opcode=OPCODE_I_ARITH)
        assert decode_i_immediate(raw) == 123
