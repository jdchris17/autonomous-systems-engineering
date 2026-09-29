import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.isa.decoder import decode
from cpu_simulator.isa.encoder import (
    encode,
    encode_add,
    encode_b_immediate,
    encode_branch,
    encode_i_immediate,
    encode_i_type_arith,
    encode_j_immediate,
    encode_jal,
    encode_jalr,
    encode_load,
    encode_r_type,
    encode_s_immediate,
    encode_shift_immediate,
    encode_store,
    encode_u_immediate,
    encode_u_type,
)


class TestEncodeAdd:
    def test_the_section_42_worked_example(self):
        assert encode_add(5, 6, 7) == 0x007302B3

    def test_matches_the_section_39_decode_example(self):
        # The same instruction decoder.py's worked example decoded --
        # confirming encode and decode agree on the same raw word.
        assert encode_add(5, 6, 7) == decode(0x007302B3).raw


class TestEncodeRType:
    @pytest.mark.parametrize("mnemonic", [
        "ADD", "SUB", "SLL", "SLT", "SLTU", "XOR", "SRL", "SRA", "OR", "AND",
    ])
    def test_round_trips_through_decode(self, mnemonic):
        raw = encode_r_type(mnemonic, rd=3, rs1=1, rs2=2)
        inst = decode(raw)
        assert inst.mnemonic == mnemonic
        assert (inst.rd, inst.rs1, inst.rs2) == (3, 1, 2)

    def test_rejects_out_of_range_register(self):
        with pytest.raises(ValueError):
            encode_r_type("ADD", rd=32, rs1=0, rs2=0)
        with pytest.raises(ValueError):
            encode_r_type("ADD", rd=0, rs1=-1, rs2=0)


class TestEncodeShiftImmediate:
    @pytest.mark.parametrize("mnemonic", ["SLLI", "SRLI", "SRAI"])
    def test_round_trips_through_decode(self, mnemonic):
        raw = encode_shift_immediate(mnemonic, rd=2, rs1=1, shift_amount=5)
        inst = decode(raw)
        assert inst.mnemonic == mnemonic
        assert inst.immediate == 5

    def test_rejects_out_of_range_shift_amount(self):
        with pytest.raises(ValueError):
            encode_shift_immediate("SLLI", rd=1, rs1=0, shift_amount=32)


class TestEncodeITypeArithmetic:
    @pytest.mark.parametrize("mnemonic", [
        "ADDI", "SLTI", "SLTIU", "XORI", "ORI", "ANDI",
    ])
    def test_round_trips_positive_immediate(self, mnemonic):
        raw = encode_i_type_arith(mnemonic, rd=2, rs1=1, immediate=100)
        inst = decode(raw)
        assert inst.mnemonic == mnemonic
        assert inst.immediate == 100

    def test_round_trips_negative_immediate(self):
        raw = encode_i_type_arith("ADDI", rd=2, rs1=1, immediate=-1)
        assert decode(raw).immediate == -1

    def test_round_trips_extreme_12_bit_values(self):
        for imm in (-2048, 2047):
            raw = encode_i_type_arith("ADDI", rd=1, rs1=0, immediate=imm)
            assert decode(raw).immediate == imm


class TestEncodeLoadStore:
    def test_lw_round_trips(self):
        raw = encode_load("LW", rd=5, rs1=2, immediate=12)
        inst = decode(raw)
        assert inst.mnemonic == "LW"
        assert (inst.rd, inst.rs1, inst.immediate) == (5, 2, 12)

    def test_lw_negative_offset(self):
        raw = encode_load("LW", rd=5, rs1=2, immediate=-8)
        assert decode(raw).immediate == -8

    def test_sw_round_trips(self):
        raw = encode_store("SW", rs1=2, rs2=5, immediate=12)
        inst = decode(raw)
        assert inst.mnemonic == "SW"
        assert (inst.rs1, inst.rs2, inst.immediate) == (2, 5, 12)

    def test_sw_negative_offset(self):
        raw = encode_store("SW", rs1=2, rs2=5, immediate=-100)
        assert decode(raw).immediate == -100

    def test_sw_extreme_12_bit_values(self):
        for imm in (-2048, 2047):
            raw = encode_store("SW", rs1=0, rs2=0, immediate=imm)
            assert decode(raw).immediate == imm


class TestEncodeJalr:
    def test_round_trips(self):
        raw = encode_jalr(rd=2, rs1=1, immediate=4)
        inst = decode(raw)
        assert inst.mnemonic == "JALR"
        assert (inst.rd, inst.rs1, inst.immediate) == (2, 1, 4)


class TestEncodeBranch:
    @pytest.mark.parametrize("mnemonic", ["BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU"])
    def test_round_trips(self, mnemonic):
        raw = encode_branch(mnemonic, rs1=1, rs2=2, immediate=16)
        inst = decode(raw)
        assert inst.mnemonic == mnemonic
        assert (inst.rs1, inst.rs2, inst.immediate) == (1, 2, 16)

    def test_negative_offset(self):
        raw = encode_branch("BEQ", rs1=1, rs2=2, immediate=-16)
        assert decode(raw).immediate == -16

    def test_extreme_13_bit_values(self):
        for imm in (-4096, 4094):
            raw = encode_branch("BEQ", rs1=0, rs2=0, immediate=imm)
            assert decode(raw).immediate == imm


class TestEncodeJal:
    def test_round_trips(self):
        raw = encode_jal(rd=1, immediate=1024)
        inst = decode(raw)
        assert inst.mnemonic == "JAL"
        assert (inst.rd, inst.immediate) == (1, 1024)

    def test_negative_offset(self):
        raw = encode_jal(rd=1, immediate=-1024)
        assert decode(raw).immediate == -1024

    def test_extreme_21_bit_values(self):
        for imm in (-1048576, 1048574):
            raw = encode_jal(rd=0, immediate=imm)
            assert decode(raw).immediate == imm


class TestEncodeUType:
    def test_lui_round_trips(self):
        raw = encode_u_type("LUI", rd=3, immediate=0x12345000)
        inst = decode(raw)
        assert inst.mnemonic == "LUI"
        assert inst.rd == 3
        assert inst.immediate == 0x12345000

    def test_auipc_round_trips(self):
        raw = encode_u_type("AUIPC", rd=5, immediate=0x00001000)
        inst = decode(raw)
        assert inst.mnemonic == "AUIPC"
        assert inst.immediate == 0x00001000

    def test_negative_immediate_round_trips(self):
        # -1412571136 == 0xABCDE000 read as signed 32-bit -- the same
        # value test_decoder.py's TestUType.test_lui decodes.
        raw = encode_u_type("LUI", rd=3, immediate=-1412571136)
        assert decode(raw).immediate == -1412571136
        assert raw == 0xABCDE1B7  # 0xABCDE000 | (rd=3 << 7) | opcode(0x37)


class TestImmediateFunctionsDirectly:
    def test_i_immediate_matches_decoder(self):
        from cpu_simulator.isa.decoder import decode_i_immediate
        raw_field = encode_i_immediate(-5)
        # Place it in a full I-type word (rs1=0, funct3=0, rd=0, opcode
        # irrelevant to the extraction itself) and decode it back.
        raw = raw_field | 0b0010011
        assert decode_i_immediate(raw) == -5

    def test_s_immediate_matches_decoder(self):
        from cpu_simulator.isa.decoder import decode_s_immediate
        raw = encode_s_immediate(-100) | 0b0100011
        assert decode_s_immediate(raw) == -100

    def test_b_immediate_matches_decoder(self):
        from cpu_simulator.isa.decoder import decode_b_immediate
        raw = encode_b_immediate(-16) | 0b1100011
        assert decode_b_immediate(raw) == -16

    def test_u_immediate_matches_decoder(self):
        from cpu_simulator.isa.decoder import decode_u_immediate
        raw = encode_u_immediate(0x12345000) | 0b0110111
        assert decode_u_immediate(raw) == 0x12345000

    def test_j_immediate_matches_decoder(self):
        from cpu_simulator.isa.decoder import decode_j_immediate
        raw = encode_j_immediate(-1024) | 0b1101111
        assert decode_j_immediate(raw) == -1024


class TestGeneralEncodeDispatcher:
    def test_r_type(self):
        assert encode("ADD", rd=5, rs1=6, rs2=7) == 0x007302B3

    def test_i_type(self):
        assert encode("ADDI", rd=2, rs1=1, immediate=100) == \
            encode_i_type_arith("ADDI", rd=2, rs1=1, immediate=100)

    def test_load(self):
        assert encode("LW", rd=5, rs1=2, immediate=12) == \
            encode_load("LW", rd=5, rs1=2, immediate=12)

    def test_store(self):
        assert encode("SW", rs1=2, rs2=5, immediate=12) == \
            encode_store("SW", rs1=2, rs2=5, immediate=12)

    def test_branch(self):
        assert encode("BEQ", rs1=1, rs2=2, immediate=16) == \
            encode_branch("BEQ", rs1=1, rs2=2, immediate=16)

    def test_jal(self):
        assert encode("JAL", rd=1, immediate=1024) == encode_jal(rd=1, immediate=1024)

    def test_jalr(self):
        assert encode("JALR", rd=2, rs1=1, immediate=4) == \
            encode_jalr(rd=2, rs1=1, immediate=4)

    def test_lui(self):
        assert encode("LUI", rd=3, immediate=0x12345000) == \
            encode_u_type("LUI", rd=3, immediate=0x12345000)

    def test_shift_immediate(self):
        assert encode("SLLI", rd=2, rs1=1, immediate=5) == \
            encode_shift_immediate("SLLI", rd=2, rs1=1, shift_amount=5)

    def test_unknown_mnemonic_rejected(self):
        with pytest.raises(ValueError):
            encode("MUL", rd=1, rs1=2, rs2=3)


class TestFullRoundTrip:
    """decode(encode(...)) recovers the original fields, and
    encode(decode(raw).mnemonic, ...) recovers the original raw word --
    for every mnemonic this package supports. The strongest possible
    check that decoder.py and encoder.py actually agree with each other.
    """

    @pytest.mark.parametrize("mnemonic", [
        "ADD", "SUB", "SLL", "SLT", "SLTU", "XOR", "SRL", "SRA", "OR", "AND",
    ])
    def test_r_type_mnemonics(self, mnemonic):
        raw = encode(mnemonic, rd=5, rs1=6, rs2=7)
        inst = decode(raw)
        raw2 = encode(inst.mnemonic, rd=inst.rd, rs1=inst.rs1, rs2=inst.rs2)
        assert raw2 == raw

    @pytest.mark.parametrize("mnemonic", [
        "ADDI", "SLTI", "SLTIU", "XORI", "ORI", "ANDI",
    ])
    def test_i_type_arith_mnemonics(self, mnemonic):
        raw = encode(mnemonic, rd=5, rs1=6, immediate=-42)
        inst = decode(raw)
        raw2 = encode(inst.mnemonic, rd=inst.rd, rs1=inst.rs1, immediate=inst.immediate)
        assert raw2 == raw

    @pytest.mark.parametrize("mnemonic", ["BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU"])
    def test_branch_mnemonics(self, mnemonic):
        raw = encode(mnemonic, rs1=3, rs2=4, immediate=-200)
        inst = decode(raw)
        raw2 = encode(inst.mnemonic, rs1=inst.rs1, rs2=inst.rs2, immediate=inst.immediate)
        assert raw2 == raw

    def test_jal(self):
        raw = encode("JAL", rd=1, immediate=-50000)
        inst = decode(raw)
        assert encode(inst.mnemonic, rd=inst.rd, immediate=inst.immediate) == raw

    def test_lui(self):
        raw = encode("LUI", rd=7, immediate=0x7FFFF000)
        inst = decode(raw)
        assert encode(inst.mnemonic, rd=inst.rd, immediate=inst.immediate) == raw
