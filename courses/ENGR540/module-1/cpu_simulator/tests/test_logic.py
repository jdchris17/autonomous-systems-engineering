import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.core import logic
from cpu_simulator.core.logic import AND, NAND, NOR, NOT, OR, XOR

BITS = [0, 1]
PAIRS = [(a, b) for a in BITS for b in BITS]


class TestTruthTables:
    def test_not(self):
        assert NOT(0) == 1
        assert NOT(1) == 0

    def test_nand(self):
        expected = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 0}
        for pair, want in expected.items():
            assert NAND(*pair) == want

    def test_and(self):
        expected = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1}
        for pair, want in expected.items():
            assert AND(*pair) == want

    def test_or(self):
        expected = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1}
        for pair, want in expected.items():
            assert OR(*pair) == want

    def test_nor(self):
        expected = {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 0}
        for pair, want in expected.items():
            assert NOR(*pair) == want

    def test_xor(self):
        expected = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}
        for pair, want in expected.items():
            assert XOR(*pair) == want


class TestConsistencyWithPython:
    """Cross-check each gate against Python's own boolean semantics,
    independent of how the gate is implemented internally.
    """

    def test_and_matches_python_and(self):
        for a, b in PAIRS:
            assert AND(a, b) == int(bool(a) and bool(b))

    def test_or_matches_python_or(self):
        for a, b in PAIRS:
            assert OR(a, b) == int(bool(a) or bool(b))

    def test_xor_matches_python_xor(self):
        for a, b in PAIRS:
            assert XOR(a, b) == int(bool(a) != bool(b))

    def test_nand_is_not_and(self):
        for a, b in PAIRS:
            assert NAND(a, b) == NOT(AND(a, b))

    def test_nor_is_not_or(self):
        for a, b in PAIRS:
            assert NOR(a, b) == NOT(OR(a, b))


class TestInvalidInputRejected:
    @pytest.mark.parametrize("gate", [NOT])
    def test_unary_gate_rejects_non_bit(self, gate):
        with pytest.raises(ValueError):
            gate(2)

    @pytest.mark.parametrize("gate", [NAND, AND, OR, NOR, XOR])
    def test_binary_gate_rejects_non_bit(self, gate):
        with pytest.raises(ValueError):
            gate(0, 2)
        with pytest.raises(ValueError):
            gate(2, 0)


class TestAbstractionHierarchy:
    """The point of this module isn't just correct truth tables -- it's
    that derived gates are actually built from lower-level gates in
    source, not reimplemented from scratch with Python operators. These
    tests inspect the source of each derived gate to enforce that.
    """

    def test_primitives_do_not_call_derived_gates(self):
        # Word-boundary match: "AND" is a substring of "NAND", so a plain
        # `in` check would false-positive on NAND's own definition.
        import re

        primitive_source = inspect.getsource(NOT) + inspect.getsource(NAND)
        for derived_name in ["AND", "OR", "NOR", "XOR"]:
            assert not re.search(rf"\b{derived_name}\(", primitive_source)

    def test_and_is_built_from_not_and_nand(self):
        source = inspect.getsource(AND)
        assert "NOT" in source and "NAND" in source

    def test_or_is_built_from_not_and_nand(self):
        source = inspect.getsource(OR)
        assert "NOT" in source and "NAND" in source

    def test_nor_is_built_from_or(self):
        source = inspect.getsource(NOR)
        assert "OR" in source

    def test_xor_is_built_from_and_or_nand(self):
        source = inspect.getsource(XOR)
        assert "AND" in source and "OR" in source and "NAND" in source

    def test_no_gate_uses_python_boolean_or_bitwise_operators(self):
        # Check only the `return` line, not the docstring (whose prose
        # can innocently contain words like "and" -- e.g. XOR's docstring
        # says "when a and b differ").
        forbidden = [" and ", " or ", " not ", "&", "|", "^"]
        for name in ["AND", "OR", "NOR", "XOR"]:
            source = inspect.getsource(getattr(logic, name))
            return_line = next(line for line in source.splitlines() if "return" in line)
            for token in forbidden:
                assert token not in return_line, f"{name} appears to use `{token.strip()}` directly"
