"""Ripple-carry addition and two's-complement subtraction, one layer of
the gates -> adders -> ALU hierarchy at a time. Every function here is
built only from the layer below it:

    half_adder   <- XOR, AND                    (logic.py)
    full_adder   <- two half_adders, OR          (logic.py)
    ripple_add   <- full_adder, chained per bit  (bits.py for bit access)
    subtract     <- ripple_add                   (no separate subtractor)

subtract() in particular is not its own from-scratch bit-borrowing
circuit. Real hardware doesn't build one either -- there's a single adder,
and subtraction is done by inverting one operand and forcing the initial
carry-in to 1:

    A - B  ==  A + NOT(B) + 1

so that's exactly what subtract() does: flip B's bits and hand
ripple_add a carry_in of 1, rather than threading a second, unrelated
code path through the ALU.

Width discipline: Python integers are arbitrary precision; the hardware
this simulates is not. ripple_add only ever produces `width` sum bits by
construction (the loop runs exactly `width` times), but the result is
also explicitly masked before returning -- both because it documents the
wraparound behavior in the code itself, and as a backstop against a bug
in the bit-assembly loop. MASK32 is the named constant for the common
32-bit case; narrower/wider widths compute their own mask the same way.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    # Run directly (e.g. VS Code's Run button, or `python adder.py`) rather
    # than as `python -m cpu_simulator.core.adder`. Python puts this file's
    # own directory (core/) on sys.path in that case, not module-1/ -- so
    # `cpu_simulator.core.bits` can't be found no matter what the terminal's
    # cwd is. Add module-1/ (this file's great-grandparent: adder.py ->
    # core/ -> cpu_simulator/ -> module-1/) so the package import below
    # resolves the same way it does when run as a module.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.core.bits import get_bit, set_bit
from cpu_simulator.core.logic import AND, OR, XOR

MASK32 = 0xFFFFFFFF


def half_adder(a: int, b: int) -> tuple[int, int]:
    """Add two bits with no carry-in. Returns (sum_bit, carry_out)."""
    sum_bit = XOR(a, b)
    carry = AND(a, b)
    return sum_bit, carry


def full_adder(a: int, b: int, carry_in: int) -> tuple[int, int]:
    """Add two bits plus a carry-in. Returns (sum_bit, carry_out).

    Built from two half adders, not a fresh 3-input truth table: the
    first half adder combines a and b, the second folds in carry_in, and
    a carry out of either stage means a carry out overall (at most one
    of the two half-adder carries can fire, since the second stage's
    inputs -- sum1 and carry_in -- can't both be 1 unless sum1 is 1, in
    which case a and b differed, so carry1 is necessarily 0).
    """
    sum1, carry1 = half_adder(a, b)
    sum2, carry2 = half_adder(sum1, carry_in)
    carry_out = OR(carry1, carry2)
    return sum2, carry_out


def ripple_add(a: int, b: int, width: int = 32, carry_in: int = 0) -> tuple[int, int]:
    """Add two `width`-bit unsigned integers by chaining full_adder
    across each bit position, carry rippling from LSB to MSB -- the
    same architecture a real ripple-carry adder circuit has, just built
    from Python function calls instead of gates.

    Deliberately not `return a + b`: that would ask Python's own
    arbitrary-precision integers to do the work, which defeats the
    point of simulating a fixed-width adder circuit. Returns
    (sum, carry_out) -- the final carry out of the MSB, which is the
    unsigned-overflow flag a real ALU would expose.
    """
    mask = _mask_for(width)
    _validate_fits("a", a, mask)
    _validate_fits("b", b, mask)
    if carry_in not in (0, 1):
        raise ValueError(f"carry_in must be 0 or 1, got {carry_in!r}")

    result = 0
    carry = carry_in
    for i in range(width):
        sum_bit, carry = full_adder(get_bit(a, i), get_bit(b, i), carry)
        if sum_bit:
            result = set_bit(result, i)

    result &= mask  # explicit width enforcement -- see module docstring
    return result, carry


def subtract(a: int, b: int, width: int = 32) -> int:
    """A - B via A + NOT(B) + 1, reusing ripple_add rather than
    implementing a second, independent subtractor.

    B is inverted with XOR against the width's all-ones mask, not
    Python's `~` operator -- `~b` sign-extends to a negative,
    infinite-precision integer with no fixed bit pattern, which is
    exactly the "Python protects you from the hardware" trap this
    module's width discipline exists to avoid.
    """
    mask = _mask_for(width)
    _validate_fits("a", a, mask)
    _validate_fits("b", b, mask)
    b_inverted = b ^ mask
    result, _carry_out = ripple_add(a, b_inverted, width=width, carry_in=1)
    return result


def _mask_for(width: int) -> int:
    if width < 1:
        raise ValueError(f"width must be at least 1, got {width}")
    return (1 << width) - 1


def _validate_fits(name: str, value: int, mask: int) -> None:
    if not (0 <= value <= mask):
        raise ValueError(f"{name} does not fit in the given width "
                          f"(0 <= {name} <= {mask}), got {value}")
