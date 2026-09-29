import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.core.alu import ALUResult, alu

MASK32 = 0xFFFFFFFF
INT32_MAX = 0x7FFFFFFF        # largest positive signed 32-bit value
INT32_MIN = 0x80000000        # bit pattern of the smallest (most negative) signed 32-bit value


class TestAdd:
    def test_zero_plus_zero(self):
        r = alu(0, 0, "ADD")
        assert r.value == 0
        assert r.zero and not r.carry and not r.overflow and not r.negative

    def test_one_plus_one(self):
        r = alu(1, 1, "ADD")
        assert r.value == 2
        assert not r.zero

    def test_255_plus_1_wraps_in_8_bit(self):
        r = alu(255, 1, "ADD", width=8)
        assert r.value == 0
        assert r.zero and r.carry

    def test_0xffffffff_plus_1_carries_but_does_not_overflow(self):
        # Unsigned wraparound (needs a 33rd bit -> carry); but as signed
        # values this is -1 + 1 == 0, entirely unremarkable -> no
        # *signed* overflow. Carry and overflow disagree here.
        r = alu(MASK32, 1, "ADD")
        assert r.value == 0
        assert r.carry
        assert not r.overflow

    def test_int32_max_plus_1_overflows_but_does_not_carry(self):
        # The opposite disagreement: the unsigned sum 0x80000000 still
        # fits in 32 bits (no carry), but as a signed result it just
        # wrapped from the largest positive value to the most negative
        # one -> signed overflow.
        r = alu(INT32_MAX, 1, "ADD")
        assert r.value == INT32_MIN
        assert not r.carry
        assert r.overflow
        assert r.negative  # 0x80000000's own sign bit is 1


class TestSubtract:
    def test_5_minus_3(self):
        assert alu(5, 3, "SUB").value == 2

    def test_3_minus_5_wraps_in_8_bit(self):
        r = alu(3, 5, "SUB", width=8)
        assert r.value == 254  # -2 as an 8-bit two's complement pattern
        assert r.negative

    def test_0_minus_1_wraps_to_all_ones(self):
        r = alu(0, 1, "SUB")
        assert r.value == MASK32  # -1 as a 32-bit two's complement pattern
        assert r.negative

    def test_int32_min_minus_1_overflows(self):
        r = alu(INT32_MIN, 1, "SUB")
        assert r.value == INT32_MAX  # wrapped past the negative end
        assert r.overflow


class TestBitwise:
    def test_and_or_xor_known_pattern(self):
        a, b = 0xFF00FF00, 0xFFFF0000
        assert alu(a, b, "AND").value == (a & b)
        assert alu(a, b, "OR").value == (a | b)
        assert alu(a, b, "XOR").value == (a ^ b)

    def test_and_or_xor_match_python_across_many_values(self):
        for a in range(0, 300, 17):
            for b in range(0, 300, 23):
                assert alu(a, b, "AND").value == (a & b)
                assert alu(a, b, "OR").value == (a | b)
                assert alu(a, b, "XOR").value == (a ^ b)


class TestShifts:
    def test_sll(self):
        assert alu(1, 4, "SLL").value == 0b10000

    def test_sll_overflow_is_dropped(self):
        r = alu(0x80000000, 1, "SLL")
        assert r.value == 0

    def test_srl_zero_fills(self):
        r = alu(0x80000000, 4, "SRL")
        assert r.value == 0x08000000

    def test_sra_sign_extends(self):
        r = alu(0x80000000, 4, "SRA")
        assert r.value == 0xF8000000

    def test_srl_and_sra_agree_on_a_positive_pattern(self):
        # No sign bit set -> logical and arithmetic shifts are identical.
        assert alu(0x40000000, 4, "SRL").value == alu(0x40000000, 4, "SRA").value

    def test_srl_and_sra_disagree_on_a_negative_looking_pattern(self):
        assert alu(0x80000000, 4, "SRL").value != alu(0x80000000, 4, "SRA").value

    def test_shift_amount_only_uses_low_bits(self):
        # For width=32, only the low 5 bits of the shift operand matter --
        # real RISC-V hardware ignores the rest rather than erroring.
        assert alu(1, 4, "SLL").value == alu(1, 4 + 32, "SLL").value


class TestSetLessThan:
    def test_signed_vs_unsigned_disagree_on_0xffffffff(self):
        # As signed, 0xFFFFFFFF is -1, so -1 < 1 is true.
        # As unsigned, 0xFFFFFFFF is the largest possible value, so it's
        # not less than 1. Same bit pattern, opposite answers.
        assert alu(MASK32, 1, "SLT").value == 1
        assert alu(MASK32, 1, "SLTU").value == 0

    def test_slt_ordinary_cases(self):
        assert alu(3, 5, "SLT").value == 1
        assert alu(5, 3, "SLT").value == 0
        assert alu(5, 5, "SLT").value == 0

    def test_sltu_ordinary_cases(self):
        assert alu(3, 5, "SLTU").value == 1
        assert alu(5, 3, "SLTU").value == 0
        assert alu(5, 5, "SLTU").value == 0

    def test_slt_correct_despite_signed_overflow(self):
        # INT32_MIN < 1 is mathematically true; naively comparing raw
        # sign bits after INT32_MIN - 1 overflows would get this wrong
        # (see TestSubtract.test_int32_min_minus_1_overflows) -- SLT
        # must correct for that via the overflow flag.
        assert alu(INT32_MIN, 1, "SLT").value == 1

    def test_slt_sltu_never_set_carry_or_overflow(self):
        r_slt = alu(MASK32, 1, "SLT")
        r_sltu = alu(MASK32, 1, "SLTU")
        assert not r_slt.carry and not r_slt.overflow
        assert not r_sltu.carry and not r_sltu.overflow


class TestFlagsApplyToEveryOperation:
    def test_zero_flag_on_a_non_arithmetic_op(self):
        assert alu(0b1010, 0b1010, "XOR").zero

    def test_negative_flag_on_a_non_arithmetic_op(self):
        assert alu(0x80000000, 0x00000000, "OR").negative

    def test_non_arithmetic_ops_never_set_carry_or_overflow(self):
        for op in ["AND", "OR", "XOR", "SLL", "SRL", "SRA"]:
            r = alu(0xFFFFFFFF, 1, op)
            assert not r.carry
            assert not r.overflow


class TestInputValidation:
    def test_unknown_operation_rejected(self):
        with pytest.raises(ValueError):
            alu(1, 1, "MUL")

    def test_operand_out_of_width_rejected(self):
        with pytest.raises(ValueError):
            alu(256, 0, "ADD", width=8)


class TestArchitecture:
    """Confirms the ALU is actually dispatching to the lower layers
    (adder.py, logic.py) rather than reimplementing each operation with
    Python's own arithmetic/bitwise operators.
    """

    def test_arithmetic_ops_use_ripple_add(self):
        source = inspect.getsource(alu)
        assert "ripple_add" in source

    def test_bitwise_ops_use_the_gate_slice_helper(self):
        source = inspect.getsource(alu)
        assert "_bitwise" in source

    def test_returns_alu_result_dataclass(self):
        r = alu(1, 1, "ADD")
        assert isinstance(r, ALUResult)
