"""The program counter, wrapped as its own CPU-specific component rather
than left as register.py's generic Counter/Register machinery.

register.py's Counter already provides an incrementable register, and an
earlier version of this file was just Counter fixed to a 4-byte step.
It's a standalone class here instead, for the same reason register_file.py
doesn't expose Register's write_next()/commit() split: a program counter
has exactly one job, advance by the instruction width or jump to an
exact target, and immediate-update semantics (no separate commit() the
caller has to remember) matches that job better than generic staged
storage does. Register's *width discipline* is still reused, though --
ripple_add for the +4 (not Python's `+`, for the same reason every other
arithmetic in this package goes through it) and an explicit 32-bit mask
on jump(), since a jump target isn't guaranteed to already fit.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    # See core/adder.py's module docstring for why this is needed to run
    # this file directly (e.g. VS Code's Run button) rather than via
    # `python -m`. Three .parent calls: pc.py -> cpu/ -> cpu_simulator/
    # -> module-1/.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.core.adder import ripple_add

WIDTH = 32
INSTRUCTION_WIDTH = 4  # bytes -- RISC-V's fixed 32-bit instruction encoding
MASK = (1 << WIDTH) - 1


class ProgramCounter:
    def __init__(self):
        self.value = 0

    def read(self) -> int:
        return self.value

    def advance(self) -> None:
        """PC_next = PC + 4, the default every cycle that isn't a
        taken branch or jump. Via ripple_add, not Python's `+` --
        wraps correctly at 32 bits, same as every other adder use in
        this package.
        """
        self.value, _carry = ripple_add(self.value, INSTRUCTION_WIDTH, width=WIDTH)

    def jump(self, target: int) -> None:
        """Set PC directly, for taken branches/jumps. Masked to 32
        bits rather than validated -- a jump target lands here as
        physical storage, the same reasoning register.py's
        write_next() masks instead of raising.
        """
        self.value = target & MASK
