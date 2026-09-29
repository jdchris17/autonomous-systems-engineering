import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from counter_system import run_counter_system


class TestCounterSystem:
    def test_zero_cycles_returns_just_the_reset_value(self):
        assert run_counter_system(0) == [0]

    def test_matches_the_spec_example(self):
        assert run_counter_system(3) == [0, 1, 2, 3]

    def test_counts_up_by_one_each_cycle(self):
        values = run_counter_system(10)
        assert values == list(range(11))

    def test_wraps_at_the_register_width(self):
        # width=2 -> values 0..3, then R + 1 wraps back to 0 via the
        # ALU's own width masking, not a special case in this file.
        assert run_counter_system(5, width=2) == [0, 1, 2, 3, 0, 1]

    def test_width_one_toggles(self):
        assert run_counter_system(4, width=1) == [0, 1, 0, 1, 0]
