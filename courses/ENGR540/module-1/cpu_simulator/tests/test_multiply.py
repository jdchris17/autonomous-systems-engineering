import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.core.multiply import multiply


class TestMultiply:
    def test_13_times_11(self):
        assert multiply(13, 11, width=8) == 143

    def test_zero_cases(self):
        assert multiply(0, 5, width=8) == 0
        assert multiply(5, 0, width=8) == 0

    def test_matches_python_across_many_values(self):
        for a in range(0, 16):
            for b in range(0, 16):
                assert multiply(a, b, width=8) == a * b

    def test_overflow_truncates_to_width(self):
        # 200 * 200 = 40000, which does not fit in 8 bits.
        assert multiply(200, 200, width=8) == (200 * 200) & 0xFF

    def test_trace_records_one_entry_per_bit(self):
        trace = []
        multiply(13, 11, width=8, trace=trace)
        assert len(trace) == 8
        assert [i for i, *_ in trace] == list(range(8))

    def test_trace_partial_products_are_shifted_multiplicand(self):
        trace = []
        multiply(13, 11, width=8, trace=trace)
        for i, bit, partial, _acc in trace:
            expected = (13 << i) & 0xFF if bit else 0
            assert partial == expected

    def test_rejects_value_that_does_not_fit_width(self):
        with pytest.raises(ValueError):
            multiply(256, 0, width=8)


class TestArchitecture:
    def test_does_not_use_pythons_own_multiplication_or_addition(self):
        source = inspect.getsource(multiply)
        body = source.split('"""')[-1]  # exclude the docstring
        assert "*" not in body
        assert "a + b" not in body and "accumulator + " not in body

    def test_uses_ripple_add_and_left_shift(self):
        source = inspect.getsource(multiply)
        assert "ripple_add" in source
        assert "left_shift" in source
