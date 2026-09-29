"""A bus: a shared wire bundle carrying one value between components, at
a fixed width. Somewhat artificial in Python -- integers already pass
between functions and objects effortlessly, no wire bundle required --
so this class only earns its keep by forcing two things Python wouldn't
otherwise force: an explicit width, and an explicit, visible "this value
is now on the bus" step instead of components just handing each other
raw ints directly.

Deliberately tiny, on purpose. drive(value) masks to this bus's width
rather than raising -- the same storage-truncates philosophy
register.py/memory.py use: a bus, like a register, physically cannot
carry more than `width` bits, there's no such thing as a bus-overflow
exception. If this class ever grows tri-state/multiple-driver
semantics, arbitration, or a transfer history, that's a real design
decision to make deliberately later, not something to bolt on
speculatively now while it's still just making width and transfers
visible.
"""

from __future__ import annotations


class Bus:
    def __init__(self, width: int = 32):
        self.width = width
        self._mask = (1 << width) - 1
        self.value = 0

    def drive(self, value: int) -> None:
        """Put `value` on the bus, masked to this bus's width."""
        self.value = value & self._mask

    def read(self) -> int:
        """Read whatever is currently on the bus."""
        return self.value
