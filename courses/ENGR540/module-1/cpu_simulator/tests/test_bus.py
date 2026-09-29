import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.cpu.bus import Bus


class TestBus:
    def test_starts_at_zero(self):
        assert Bus().read() == 0

    def test_default_width_is_32(self):
        assert Bus().width == 32

    def test_drive_then_read(self):
        bus = Bus()
        bus.drive(0xCAFEBABE)
        assert bus.read() == 0xCAFEBABE

    def test_drive_masks_to_width_rather_than_raises(self):
        bus = Bus(width=8)
        bus.drive(0x1FF)
        assert bus.read() == 0xFF

    def test_custom_width(self):
        bus = Bus(width=4)
        bus.drive(0xFF)
        assert bus.read() == 0xF

    def test_later_drive_overwrites_earlier_one(self):
        bus = Bus()
        bus.drive(1)
        bus.drive(2)
        assert bus.read() == 2

    def test_independent_instances(self):
        a, b = Bus(), Bus()
        a.drive(5)
        assert a.read() == 5
        assert b.read() == 0
