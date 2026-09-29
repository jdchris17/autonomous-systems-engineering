import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.cpu.register import (
    Bit,
    Counter,
    GeneralPurposeRegister,
    Register,
)


class TestRegisterTwoPhaseWrite:
    def test_starts_at_reset_value(self):
        assert Register(width=8).read() == 0
        assert Register(width=8, reset_value=5).read() == 5

    def test_write_next_does_not_change_read_until_commit(self):
        reg = Register(width=8)
        reg.write_next(42)
        assert reg.read() == 0  # still the old value -- nothing committed yet

    def test_commit_latches_the_staged_value(self):
        reg = Register(width=8)
        reg.write_next(42)
        reg.commit()
        assert reg.read() == 42

    def test_commit_with_nothing_staged_holds_current_value(self):
        reg = Register(width=8, reset_value=7)
        reg.commit()
        assert reg.read() == 7

    def test_last_write_next_before_commit_wins(self):
        reg = Register(width=8)
        reg.write_next(1)
        reg.write_next(2)
        reg.write_next(3)
        reg.commit()
        assert reg.read() == 3

    def test_reset_restores_reset_value_and_clears_pending(self):
        reg = Register(width=8, reset_value=9)
        reg.write_next(200)
        reg.commit()
        assert reg.read() == 200
        reg.write_next(50)  # staged, not yet committed
        reg.reset()
        assert reg.read() == 9
        reg.commit()
        assert reg.read() == 9  # the pending 50 was cleared by reset(), not committed


class TestRegisterWidthMasking:
    def test_write_next_masks_rather_than_raises(self):
        reg = Register(width=8)
        reg.write_next(300)  # 300 does not fit in 8 bits
        reg.commit()
        assert reg.read() == 300 & 0xFF

    def test_reset_value_is_masked_too(self):
        assert Register(width=8, reset_value=300).read() == 300 & 0xFF

    def test_width_property(self):
        assert Register(width=16).width == 16


class TestBit:
    def test_width_is_one(self):
        assert Bit().width == 1

    def test_masks_to_a_single_bit(self):
        bit = Bit()
        bit.write_next(5)  # 0b101 -- only the low bit should survive
        bit.commit()
        assert bit.read() == 1


class TestGeneralPurposeRegister:
    def test_width_is_32(self):
        assert GeneralPurposeRegister().width == 32

    def test_behaves_like_a_32_bit_register(self):
        reg = GeneralPurposeRegister()
        reg.write_next(0xDEADBEEF)
        reg.commit()
        assert reg.read() == 0xDEADBEEF


class TestCounter:
    def test_increment_is_two_phase(self):
        counter = Counter(width=8)
        counter.increment()
        assert counter.read() == 0  # staged, not committed
        counter.commit()
        assert counter.read() == 1

    def test_repeated_increments(self):
        counter = Counter(width=8)
        for _ in range(5):
            counter.increment()
            counter.commit()
        assert counter.read() == 5

    def test_increment_by_arbitrary_step(self):
        counter = Counter(width=8)
        counter.increment(step=10)
        counter.commit()
        assert counter.read() == 10

    def test_increment_wraps_at_width(self):
        counter = Counter(width=8, reset_value=255)
        counter.increment()
        counter.commit()
        assert counter.read() == 0

    def test_negative_step_decrements(self):
        counter = Counter(width=8, reset_value=5)
        counter.increment(step=-1)
        counter.commit()
        assert counter.read() == 4

    def test_uses_ripple_add(self):
        source = inspect.getsource(Counter.increment)
        assert "ripple_add" in source
