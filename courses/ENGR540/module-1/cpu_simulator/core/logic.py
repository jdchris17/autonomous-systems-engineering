"""Logical abstraction: boolean gates, built as a hierarchy rather than
each implemented independently.

This is the logical abstraction, not literal transistor physics -- but it
mirrors why real hardware is built the way it is. NAND is a *universal*
gate: every other boolean function can be constructed from NAND alone
(with NOT as the other bedrock primitive, since NOT itself has a
single-input identity that doesn't reduce further in terms of the
operators used here). That's not a coincidence of this exercise -- it's
why NAND (and NOR) gates, not AND/OR gates, are what actually gets
fabricated on silicon: a NAND gate is cheaper to build from transistors
than an AND gate, so everything else is built up from it.

Hierarchy enforced in this file:

    NOT, NAND            -- primitives, implemented directly
    AND, OR               -- derived from NOT + NAND only
    NOR, XOR              -- derived from the gates above

No derived gate below reaches for Python's `and`/`or`/`not` or bitwise
operators directly -- each one is built only from already-defined gates,
so the dependency chain back to NOT/NAND stays visible in the code, not
just in a comment.
"""

from __future__ import annotations


def NOT(a: int) -> int:
    """Primitive: invert a single bit."""
    _validate_bit("a", a)
    return 1 - a


def NAND(a: int, b: int) -> int:
    """Primitive: NOT AND. The universal gate -- everything else in this
    file ultimately bottoms out here (or at NOT).
    """
    _validate_bit("a", a)
    _validate_bit("b", b)
    return 0 if (a == 1 and b == 1) else 1


def AND(a: int, b: int) -> int:
    """AND is just NAND with its own output inverted."""
    return NOT(NAND(a, b))


def OR(a: int, b: int) -> int:
    """De Morgan's law: a OR b == NOT(NOT a AND NOT b), and "NOT ... AND"
    is exactly what NAND already computes -- so OR is NAND fed inverted
    inputs, with no separate AND/NOT-of-the-result step needed.
    """
    return NAND(NOT(a), NOT(b))


def NOR(a: int, b: int) -> int:
    """NOT OR, built from the OR already derived above."""
    return NOT(OR(a, b))


def XOR(a: int, b: int) -> int:
    """a XOR b == (a OR b) AND (a NAND b): true exactly when a and b
    differ -- OR rules out "both 0", NAND rules out "both 1", and AND
    requires both conditions to hold at once. The classic 4-gate XOR.
    """
    return AND(OR(a, b), NAND(a, b))


def _validate_bit(name: str, value: int) -> None:
    if value not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
