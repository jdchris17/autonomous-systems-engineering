import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.cpu.clock import Clock


class TestClock:
    def test_starts_at_cycle_zero(self):
        assert Clock().cycle == 0

    def test_tick_increments_by_one(self):
        clock = Clock()
        clock.tick()
        assert clock.cycle == 1

    def test_repeated_ticks_accumulate(self):
        clock = Clock()
        for _ in range(10):
            clock.tick()
        assert clock.cycle == 10

    def test_independent_clocks_do_not_share_state(self):
        a, b = Clock(), Clock()
        a.tick()
        a.tick()
        assert a.cycle == 2
        assert b.cycle == 0
