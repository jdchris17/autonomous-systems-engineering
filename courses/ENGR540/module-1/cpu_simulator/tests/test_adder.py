import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.core.adder import MASK32, full_adder, half_adder, ripple_add, subtract
from cpu_simulator.core.numbers import twos_complement_to_signed, signed_to_twos_complement


class TestHalfAdder:
    def test_truth_table(self):
        expected = {
            (0, 0): (0, 0),
            (0, 1): (1, 0),
            (1, 0): (1, 0),
            (1, 1): (0, 1),
        }
        for (a, b), want in expected.items():
            assert half_adder(a, b) == want


class TestFullAdder:
    def test_truth_table(self):
        expected = {
            (0, 0, 0): (0, 0),
            (0, 0, 1): (1, 0),
            (0, 1, 0): (1, 0),
            (0, 1, 1): (0, 1),
            (1, 0, 0): (1, 0),
            (1, 0, 1): (0, 1),
            (1, 1, 0): (0, 1),
            (1, 1, 1): (1, 1),
        }
        for (a, b, cin), want in expected.items():
            assert full_adder(a, b, cin) == want

    def test_built_from_half_adders_not_a_fresh_truth_table(self):
        source = inspect.getsource(full_adder)
        assert "half_adder" in source
        assert source.count("half_adder") >= 2


class TestRippleAdd:
    def test_small_values(self):
        result, carry = ripple_add(5, 7, width=8)
        assert result == 12
        assert carry == 0

    def test_matches_python_plus_within_width(self):
        for a in range(0, 32, 3):
            for b in range(0, 32, 5):
                result, carry = ripple_add(a, b, width=8)
                assert result == (a + b) & 0xFF
                assert carry == ((a + b) >> 8) & 1

    def test_overflow_wraps_and_sets_carry(self):
        result, carry = ripple_add(MASK32, 1, width=32)
        assert result == 0
        assert carry == 1

    def test_carry_in_is_honored(self):
        result, carry = ripple_add(0, 0, width=8, carry_in=1)
        assert result == 1
        assert carry == 0

    def test_rejects_value_that_does_not_fit_width(self):
        with pytest.raises(ValueError):
            ripple_add(256, 0, width=8)
        with pytest.raises(ValueError):
            ripple_add(0, -1, width=8)

    def test_does_not_use_pythons_own_integer_addition(self):
        source = inspect.getsource(ripple_add)
        body = source.split('"""')[-1]  # exclude the docstring, which discusses `a + b` in prose
        assert "a + b" not in body and "b + a" not in body
        assert "full_adder" in body


class TestSubtract:
    def test_positive_result(self):
        assert subtract(10, 3, width=8) == 7

    def test_matches_python_within_width(self):
        for a in range(0, 40, 3):
            for b in range(0, 40, 7):
                assert subtract(a, b, width=8) == (a - b) & 0xFF

    def test_negative_result_wraps_to_twos_complement(self):
        # 5 - 7 = -2, which as an 8-bit two's complement pattern is 254.
        result = subtract(5, 7, width=8)
        assert result == 254
        assert twos_complement_to_signed(f"{result:08b}") == -2

    def test_round_trip_against_signed_twos_complement_helpers(self):
        for a in range(-20, 20):
            for b in range(-20, 20):
                expected = a - b
                lo, hi = -128, 127
                if not (lo <= expected <= hi):
                    continue  # outside 8-bit signed range; not this test's concern
                a_bits = signed_to_twos_complement(a, width=8)
                b_bits = signed_to_twos_complement(b, width=8)
                result = subtract(int(a_bits, 2), int(b_bits, 2), width=8)
                assert twos_complement_to_signed(f"{result:08b}") == expected

    def test_uses_ripple_add_rather_than_a_separate_subtractor(self):
        source = inspect.getsource(subtract)
        assert "ripple_add" in source
        assert " - " not in source.split("\"\"\"")[-1]  # no bare `a - b` in the body

    def test_rejects_value_that_does_not_fit_width(self):
        with pytest.raises(ValueError):
            subtract(256, 0, width=8)
