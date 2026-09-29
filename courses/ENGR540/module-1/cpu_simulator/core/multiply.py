"""Unsigned multiplication, done the way hardware without a dedicated
multiplier does it: shift-and-add. Not needed for the ALU yet (no MUL
operation is in alu.py's OPERATIONS) -- this file exists because
multiplication decomposing into shifts + addition + control is worth
seeing directly, the same way subtract() turned out to just be addition
with an inverted operand and a forced carry-in.

The algorithm, in terms already built:
    For each bit i of the multiplier, from LSB to MSB:
        if that bit is 1: add (multiplicand << i) into an accumulator
    The accumulator, after all width bits, is the product.

Every piece is something this package already has: get_bit to read the
multiplier one bit at a time, left_shift to form each partial product,
and ripple_add to accumulate them -- no `*`, and no `+` either, since
"add" here means the same width-explicit ripple_add every other file in
this package uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    # See adder.py's module docstring for the full explanation. Three
    # .parent calls: multiply.py -> core/ -> cpu_simulator/ -> module-1/.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.core.adder import ripple_add
from cpu_simulator.core.bits import get_bit, left_shift


def multiply(a: int, b: int, width: int = 32, trace=None) -> int:
    """Unsigned a * b via shift-and-add, result truncated to `width` bits
    (a real fixed-width multiplier drops the bits that overflow the
    result register, same as ripple_add drops carry out of the top bit).

    If `trace` is given a list, each bit's partial product is appended
    to it as (bit_index, bit_value, partial_product, accumulator_after)
    -- the intermediate work multiply_demo.py prints.
    """
    mask = _mask_for(width)
    _validate_fits("a", a, mask)
    _validate_fits("b", b, mask)

    accumulator = 0
    for i in range(width):
        bit = get_bit(b, i)
        partial = left_shift(a, i) & mask if bit else 0
        if bit:
            accumulator, _carry = ripple_add(accumulator, partial, width=width)
        if trace is not None:
            trace.append((i, bit, partial, accumulator))
    return accumulator


def _mask_for(width: int) -> int:
    if width < 1:
        raise ValueError(f"width must be at least 1, got {width}")
    return (1 << width) - 1


def _validate_fits(name: str, value: int, mask: int) -> None:
    if not (0 <= value <= mask):
        raise ValueError(f"{name} does not fit in the given width "
                          f"(0 <= {name} <= {mask}), got {value}")


def main():
    a, b, width = 13, 11, 8

    print("=" * 68)
    print(f"UNSIGNED SHIFT-AND-ADD MULTIPLICATION: {a} x {b}  (width={width})")
    print("=" * 68)
    print(f"{a} = 0b{a:0{width}b}   {b} = 0b{b:0{width}b}")
    print()
    print(f"{'bit i':>6}  {'b[i]':>4}  {'partial (a << i)':>20}  {'accumulator':>14}")

    trace = []
    result = multiply(a, b, width=width, trace=trace)

    highest_set_bit = max((i for i, bit, _p, _acc in trace if bit), default=None)
    for i, bit, partial, acc in trace:
        if bit == 0 and highest_set_bit is not None and i > highest_set_bit:
            continue  # nothing left to show once every remaining bit of b is 0
        print(f"{i:6d}  {bit:4d}  {partial:20d}  {acc:14d}")

    print()
    print(f"Result: {result}  (Python's {a} * {b} = {a * b}, for comparison)")
    assert result == a * b, "shift-and-add result disagreed with Python's own multiplication"
    print("=" * 68)


if __name__ == "__main__":
    main()
