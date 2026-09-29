import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.cpu.register_file import RegisterFile


class TestX0HardwiredToZero:
    def test_x0_reads_zero_initially(self):
        assert RegisterFile().read(0) == 0

    def test_writes_to_x0_are_discarded(self):
        rf = RegisterFile()
        rf.write(0, 12345)
        assert rf.read(0) == 0

    def test_x0_stays_zero_across_many_writes(self):
        rf = RegisterFile()
        for value in [1, 0xFFFFFFFF, 42, 0]:
            rf.write(0, value)
        assert rf.read(0) == 0


class TestOrdinaryRegisters:
    def test_read_before_write_is_zero(self):
        rf = RegisterFile()
        for r in [1, 5, 31]:
            assert rf.read(r) == 0

    def test_write_then_read(self):
        rf = RegisterFile()
        rf.write(5, 42)
        assert rf.read(5) == 42

    def test_registers_are_independent(self):
        rf = RegisterFile()
        rf.write(1, 100)
        rf.write(2, 200)
        assert rf.read(1) == 100
        assert rf.read(2) == 200

    def test_write_masks_to_32_bits(self):
        rf = RegisterFile()
        rf.write(3, 0x1_FFFFFFFF)
        assert rf.read(3) == 0xFFFFFFFF

    def test_x31_is_a_real_register(self):
        rf = RegisterFile()
        rf.write(31, 99)
        assert rf.read(31) == 99


class TestValidation:
    def test_negative_index_rejected(self):
        rf = RegisterFile()
        with pytest.raises(ValueError):
            rf.read(-1)
        with pytest.raises(ValueError):
            rf.write(-1, 0)

    def test_index_32_and_above_rejected(self):
        rf = RegisterFile()
        with pytest.raises(ValueError):
            rf.read(32)
        with pytest.raises(ValueError):
            rf.write(32, 0)
