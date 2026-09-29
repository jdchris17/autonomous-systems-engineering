"""Registers: fixed-width storage with a two-phase write, mirroring how
synchronous hardware actually updates -- see clock.py for why the two
phases are kept apart in time.

write_next(value) stages a value; nothing observable changes until
commit() runs. read() always returns the last *committed* value, never a
pending one -- that's the entire reason this class has two methods
instead of one plain setter. A real flip-flop's D input isn't readable
as this cycle's Q; a Register has no way to peek at its own pending
write either, on purpose.

Width handling here is deliberately different from bits.py/numbers.py's
"raise on mismatch" philosophy. Those modules validate values that are
*supposed* to already fit a width before doing arithmetic with them --
raising catches a bug upstream, before it corrupts a computation. A
register is the thing that actually *enforces* a width physically: real
storage of `width` bits cannot hold more than `width` bits, and there is
no such thing as a register-overflow exception -- the extra bits just
don't exist. So write_next() masks (`value & (2**width - 1)`) instead of
raising; that mask is the hardware behavior, not a shortcut around it.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    # See core/adder.py's module docstring for why this is needed to run
    # this file directly (e.g. VS Code's Run button) rather than via
    # `python -m`. Three .parent calls: register.py -> cpu/ ->
    # cpu_simulator/ -> module-1/.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.core.adder import ripple_add


class Register:
    """Generic fixed-width register with a two-phase write."""

    def __init__(self, width: int = 32, reset_value: int = 0):
        self._width = width
        self._mask = _mask_for(width)
        self._reset_value = reset_value & self._mask
        self._value = self._reset_value
        self._next = None  # pending write, or None if nothing staged this cycle

    @property
    def width(self) -> int:
        return self._width

    def read(self) -> int:
        """The last *committed* value -- never a pending, uncommitted one."""
        return self._value

    def write_next(self, value: int) -> None:
        """Stage a value to become this register's contents at the next
        commit(). Masked to this register's width -- see the module
        docstring for why that's a mask and not a raise. Calling this
        again before a commit() replaces the pending value; the last
        call before commit() wins.
        """
        self._next = value & self._mask

    def commit(self) -> None:
        """Latch the staged value, if any. If write_next() wasn't
        called since the last commit(), the register holds its current
        value -- the same as a real flip-flop whose D input didn't
        change.
        """
        if self._next is not None:
            self._value = self._next
            self._next = None

    def reset(self) -> None:
        """Immediately restore the reset value. Not staged through
        write_next()/commit() -- a real reset line bypasses the clocked
        D input entirely rather than waiting for the next clock edge.
        """
        self._value = self._reset_value
        self._next = None


class Bit(Register):
    """A single bit of storage: Register with width fixed to 1."""

    def __init__(self, reset_value: int = 0):
        super().__init__(width=1, reset_value=reset_value)


class GeneralPurposeRegister(Register):
    """A plain 32-bit register, named distinctly from the generic
    Register mainly so a future register file built from these reads as
    an intentional array of 32-bit registers (x0-x31), not as leftover
    default-width Registers.
    """

    def __init__(self, reset_value: int = 0):
        super().__init__(width=32, reset_value=reset_value)


class Counter(Register):
    """A register that can stage "current value + step" as its own next
    value, via the same ripple_add the rest of this package uses for
    arithmetic -- not Python's `+`, for the same reason ripple_add
    exists at all. Still two-phase: increment() only calls write_next();
    nothing changes until the next commit().
    """

    def increment(self, step: int = 1) -> None:
        """Stage current + step (mod 2**width) as the next value. A
        negative step decrements: Python's `&` reads a negative int as
        an infinite two's-complement pattern, so masking it to this
        register's width already produces the correct wraparound-
        consistent bit pattern to add.
        """
        next_value, _carry = ripple_add(self.read(), step & self._mask, width=self._width)
        self.write_next(next_value)


def _mask_for(width: int) -> int:
    if width < 1:
        raise ValueError(f"width must be at least 1, got {width}")
    return (1 << width) - 1
