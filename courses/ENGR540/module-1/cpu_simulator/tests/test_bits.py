import sys
from pathlib import Path

import pytest

# Imported as a package (cpu_simulator.bits) for consistency with
# test_numbers.py -- see the comment there for why numbers.py in
# particular requires this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.core.bits import (
    bit_and,
    bit_or,
    bit_xor,
    clear_bit,
    extract_bits,
    get_bit,
    left_shift,
    right_shift,
    set_bit,
    toggle_bit,
)


class TestLogicOps:
    def test_bit_and(self):
        assert bit_and(0b1100, 0b1010) == 0b1000

    def test_bit_or(self):
        assert bit_or(0b1100, 0b1010) == 0b1110

    def test_bit_xor(self):
        assert bit_xor(0b1100, 0b1010) == 0b0110


class TestShifts:
    def test_left_shift(self):
        assert left_shift(0b0001, 4) == 0b10000

    def test_right_shift_zero_fills(self):
        assert right_shift(0b1000, 3) == 0b1

    def test_right_shift_drops_low_bits(self):
        assert right_shift(0b1011, 1) == 0b101

    def test_negative_amount_rejected(self):
        with pytest.raises(ValueError):
            left_shift(1, -1)
        with pytest.raises(ValueError):
            right_shift(1, -1)

    def test_negative_value_rejected_for_right_shift(self):
        with pytest.raises(ValueError):
            right_shift(-8, 1)


class TestSingleBitOps:
    def test_get_bit(self):
        value = 0b1010
        assert get_bit(value, 0) == 0
        assert get_bit(value, 1) == 1
        assert get_bit(value, 3) == 1

    def test_set_bit(self):
        assert set_bit(0b0000, 2) == 0b0100
        assert set_bit(0b0100, 2) == 0b0100  # already set, no change

    def test_clear_bit(self):
        assert clear_bit(0b1111, 1) == 0b1101
        assert clear_bit(0b1101, 1) == 0b1101  # already clear, no change

    def test_toggle_bit(self):
        assert toggle_bit(0b0000, 0) == 0b0001
        assert toggle_bit(0b0001, 0) == 0b0000

    def test_negative_position_rejected(self):
        with pytest.raises(ValueError):
            get_bit(1, -1)


class TestExtractBits:
    def test_extract_middle_field(self):
        # value = 0b101101 (bit0=1, bit1=0, bit2=1, bit3=1, bit4=0, bit5=1)
        # bits [1:5) = bit4 bit3 bit2 bit1 = 0110
        value = 0b101101
        assert extract_bits(value, start=1, width=4) == 0b0110

    def test_extract_low_field(self):
        assert extract_bits(0b11110000, start=0, width=4) == 0b0000

    def test_extract_high_field(self):
        assert extract_bits(0b11110000, start=4, width=4) == 0b1111

    def test_riscv_style_opcode_extraction(self):
        # A made-up 32-bit word: opcode in bits [0:7), rd in bits [7:12)
        opcode = 0b0110011
        rd = 0b01010
        instruction = (rd << 7) | opcode
        assert extract_bits(instruction, start=0, width=7) == opcode
        assert extract_bits(instruction, start=7, width=5) == rd

    def test_width_below_one_rejected(self):
        with pytest.raises(ValueError):
            extract_bits(0b1010, start=0, width=0)
