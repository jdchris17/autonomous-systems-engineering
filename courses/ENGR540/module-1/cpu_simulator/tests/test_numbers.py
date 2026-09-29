import sys
from pathlib import Path

import pytest

# cpu_simulator.core.numbers is imported as a package, not as a bare
# top-level `numbers` -- numbers.py shadows the *stdlib* numbers module
# if its own directory (core/) is ever put on sys.path directly, which
# breaks pytest's own imports (it needs decimal, which needs stdlib
# numbers).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.core.numbers import (
    binary_to_decimal,
    binary_to_hex,
    decimal_to_binary,
    hex_to_binary,
    signed_range,
    signed_to_twos_complement,
    twos_complement_to_signed,
    unsigned_range,
)


class TestDecimalToBinary:
    def test_no_width_gives_minimal_representation(self):
        assert decimal_to_binary(0) == "0"
        assert decimal_to_binary(5) == "101"
        assert decimal_to_binary(42) == "101010"

    def test_width_zero_pads(self):
        assert decimal_to_binary(42, width=8) == "00101010"

    def test_width_exact_fit_no_padding_needed(self):
        assert decimal_to_binary(255, width=8) == "11111111"

    def test_negative_input_rejected(self):
        with pytest.raises(ValueError):
            decimal_to_binary(-1)

    def test_overflow_of_width_rejected(self):
        with pytest.raises(ValueError):
            decimal_to_binary(256, width=8)


class TestBinaryToDecimal:
    def test_round_trips_with_decimal_to_binary(self):
        for n in [0, 1, 42, 255, 1000]:
            assert binary_to_decimal(decimal_to_binary(n)) == n

    def test_leading_zeros_ignored(self):
        assert binary_to_decimal("00101010") == 42

    def test_invalid_string_rejected(self):
        with pytest.raises(ValueError):
            binary_to_decimal("102")
        with pytest.raises(ValueError):
            binary_to_decimal("")


class TestBinaryHexConversion:
    def test_binary_to_hex_pads_to_nibble(self):
        assert binary_to_hex("00101010") == "2A"
        assert binary_to_hex("101010") == "2A"  # 6 bits -> padded to 8

    def test_hex_to_binary_natural_width(self):
        assert hex_to_binary("2A") == "00101010"
        assert hex_to_binary("FF") == "11111111"

    def test_hex_to_binary_explicit_width(self):
        assert hex_to_binary("2A", width=16) == "0000000000101010"

    def test_round_trip(self):
        assert binary_to_hex(hex_to_binary("DEAD")) == "DEAD"


class TestRanges:
    def test_unsigned_range(self):
        assert unsigned_range(8) == (0, 255)
        assert unsigned_range(1) == (0, 1)

    def test_signed_range(self):
        assert signed_range(8) == (-128, 127)
        assert signed_range(1) == (-1, 0)

    def test_width_below_one_rejected(self):
        with pytest.raises(ValueError):
            unsigned_range(0)
        with pytest.raises(ValueError):
            signed_range(0)


class TestTwosComplement:
    def test_positive_values_unchanged_pattern(self):
        assert signed_to_twos_complement(5, width=8) == "00000101"

    def test_negative_one_is_all_ones(self):
        assert signed_to_twos_complement(-1, width=8) == "11111111"

    def test_most_negative_value(self):
        assert signed_to_twos_complement(-128, width=8) == "10000000"

    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            signed_to_twos_complement(128, width=8)
        with pytest.raises(ValueError):
            signed_to_twos_complement(-129, width=8)

    def test_decode_matches_encode(self):
        for n in range(-128, 128):
            assert twos_complement_to_signed(signed_to_twos_complement(n, width=8)) == n

    def test_top_bit_clear_is_nonnegative(self):
        assert twos_complement_to_signed("01111111") == 127

    def test_top_bit_set_is_negative(self):
        assert twos_complement_to_signed("11111111") == -1
        assert twos_complement_to_signed("10000000") == -128
