import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.memory.memory import Memory


class TestConstruction:
    def test_size_bytes_property(self):
        assert Memory(16).size_bytes == 16

    def test_rejects_nonpositive_size(self):
        with pytest.raises(ValueError):
            Memory(0)
        with pytest.raises(ValueError):
            Memory(-4)

    def test_starts_zeroed(self):
        mem = Memory(8)
        for addr in range(8):
            assert mem.read_byte(addr) == 0


class TestByteAccess:
    def test_write_then_read(self):
        mem = Memory(8)
        mem.write_byte(3, 0xAB)
        assert mem.read_byte(3) == 0xAB

    def test_bytes_are_independent(self):
        mem = Memory(8)
        mem.write_byte(0, 1)
        mem.write_byte(1, 2)
        assert mem.read_byte(0) == 1
        assert mem.read_byte(1) == 2

    def test_write_byte_masks_rather_than_raises(self):
        mem = Memory(8)
        mem.write_byte(0, 0x1FF)  # does not fit in a byte
        assert mem.read_byte(0) == 0x1FF & 0xFF

    def test_out_of_range_address_rejected(self):
        mem = Memory(8)
        with pytest.raises(ValueError):
            mem.read_byte(8)
        with pytest.raises(ValueError):
            mem.read_byte(-1)
        with pytest.raises(ValueError):
            mem.write_byte(8, 0)


class TestWordAccessIsLittleEndian:
    def test_write_word_then_read_word(self):
        mem = Memory(16)
        mem.write_word(0, 0x12345678)
        assert mem.read_word(0) == 0x12345678

    def test_write_word_places_least_significant_byte_first(self):
        mem = Memory(16)
        mem.write_word(0, 0x12345678)
        assert mem.read_byte(0) == 0x78
        assert mem.read_byte(1) == 0x56
        assert mem.read_byte(2) == 0x34
        assert mem.read_byte(3) == 0x12

    def test_read_word_reassembles_little_endian_bytes(self):
        mem = Memory(16)
        mem.write_byte(4, 0x78)
        mem.write_byte(5, 0x56)
        mem.write_byte(6, 0x34)
        mem.write_byte(7, 0x12)
        assert mem.read_word(4) == 0x12345678

    def test_word_access_at_nonzero_offset(self):
        mem = Memory(16)
        mem.write_word(8, 0xDEADBEEF)
        assert mem.read_word(8) == 0xDEADBEEF
        # neighboring bytes untouched
        assert mem.read_byte(7) == 0
        assert mem.read_byte(12) == 0

    def test_write_word_masks_values_wider_than_32_bits(self):
        mem = Memory(16)
        mem.write_word(0, 0x1_00000000 | 0xCAFEBABE)
        assert mem.read_word(0) == 0xCAFEBABE

    def test_word_access_straddling_the_end_is_rejected(self):
        mem = Memory(4)  # exactly one word
        mem.write_word(0, 0xAABBCCDD)  # fits exactly
        with pytest.raises(ValueError):
            mem.read_word(1)  # would need bytes 1..4, but memory ends at 4
        with pytest.raises(ValueError):
            mem.write_word(2, 0)


class TestSeparateInstructionAndDataMemory:
    """Memory itself makes no instruction/data distinction -- the split
    described in this module's docstring comes entirely from
    instantiating it twice. These tests confirm that actually gives you
    two independent address spaces, not two references to shared state.
    """

    def test_two_instances_start_independently_zeroed(self):
        instruction_memory = Memory(size_bytes=64)
        data_memory = Memory(size_bytes=64)
        instruction_memory.write_word(0, 0xDEADBEEF)
        assert data_memory.read_word(0) == 0

    def test_writes_to_one_do_not_appear_in_the_other(self):
        instruction_memory = Memory(size_bytes=64)
        data_memory = Memory(size_bytes=64)
        instruction_memory.write_byte(4, 0xAB)
        data_memory.write_byte(4, 0xCD)
        assert instruction_memory.read_byte(4) == 0xAB
        assert data_memory.read_byte(4) == 0xCD

    def test_instances_can_have_different_sizes(self):
        instruction_memory = Memory(size_bytes=1024)
        data_memory = Memory(size_bytes=256)
        assert instruction_memory.size_bytes == 1024
        assert data_memory.size_bytes == 256
