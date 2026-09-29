"""Low-level bit manipulation: the primitive operations a CPU's ALU and
instruction decoder are built from.

Bit numbering convention used throughout: position 0 is the least
significant bit. `right_shift` is a *logical* shift (zero-fills from the
left), not arithmetic (sign-extending) -- Python ints are unbounded, so
there's no sign bit to preserve here; that distinction only matters once
values are given a fixed width, which is numbers.py's job, not this
file's.
"""

from __future__ import annotations


def bit_and(a: int, b: int) -> int:
    return a & b


def bit_or(a: int, b: int) -> int:
    return a | b


def bit_xor(a: int, b: int) -> int:
    return a ^ b


def left_shift(value: int, amount: int) -> int:
    _validate_nonnegative("amount", amount)
    return value << amount


def right_shift(value: int, amount: int) -> int:
    """Logical right shift: zero-fills from the left. Requires a
    non-negative `value` since a negative Python int has no fixed-width
    bit pattern to shift zeros into.
    """
    _validate_nonnegative("amount", amount)
    _validate_nonnegative("value", value)
    return value >> amount


def get_bit(value: int, position: int) -> int:
    """Return the bit at `position` (0 = LSB) as 0 or 1."""
    _validate_nonnegative("position", position)
    return (value >> position) & 1


def set_bit(value: int, position: int) -> int:
    """Return `value` with the bit at `position` forced to 1."""
    _validate_nonnegative("position", position)
    return value | (1 << position)


def clear_bit(value: int, position: int) -> int:
    """Return `value` with the bit at `position` forced to 0."""
    _validate_nonnegative("position", position)
    return value & ~(1 << position)


def toggle_bit(value: int, position: int) -> int:
    """Return `value` with the bit at `position` flipped."""
    _validate_nonnegative("position", position)
    return value ^ (1 << position)


def extract_bits(value: int, start: int, width: int) -> int:
    """Pull out `width` bits starting at bit `start` (0 = LSB) and return
    them right-aligned as an unsigned integer.

    This is the operation instruction decoding is built from: a RISC-V
    opcode is extract_bits(instruction, 0, 7), for instance -- fixed
    bit fields packed into a 32-bit word, pulled out one field at a
    time.
    """
    _validate_nonnegative("start", start)
    if width < 1:
        raise ValueError(f"width must be at least 1, got {width}")
    mask = (1 << width) - 1
    return (value >> start) & mask


def _validate_nonnegative(name: str, n: int) -> None:
    if n < 0:
        raise ValueError(f"{name} must be non-negative, got {n}")
