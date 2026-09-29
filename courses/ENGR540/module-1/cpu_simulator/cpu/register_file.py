"""RV32I's register file: 32 general-purpose 32-bit registers, x0-x31,
built from register.py's GeneralPurposeRegister rather than a bare list
of ints -- the same width masking and storage discipline every other
register in this package gets, reused instead of reimplemented.

The one RISC-V-specific rule this file exists to enforce: x0 is
hardwired to zero. Every read of x0 returns 0, and every write to x0 is
silently discarded -- not an error, since real RISC-V code routinely
writes to x0 on purpose (it's the conventional target for "compute this
and throw the result away," e.g. how a NOP is encoded). Enforcing that
now, before there's a decoder to feed this register file, means the
eventual instruction-execution code never has to special-case rd == 0
itself.

write() commits immediately rather than exposing the write_next()/
commit() two-phase split GeneralPurposeRegister itself has -- that split
matters when several registers' next values must all be computed against
the *same* starting state before any of them changes (see register.py
and clock.py's docstrings). A register file's read(rs1)/read(rs2)/
write(rd, value) as given here is the simpler contract this module's
spec actually asked for; the two-phase machinery still runs underneath,
it's just not exposed at this layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    # See core/adder.py's module docstring for why this is needed to run
    # this file directly (e.g. VS Code's Run button) rather than via
    # `python -m`. Three .parent calls: register_file.py -> cpu/ ->
    # cpu_simulator/ -> module-1/.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.cpu.register import GeneralPurposeRegister

NUM_REGISTERS = 32


class RegisterFile:
    def __init__(self):
        self._registers = [GeneralPurposeRegister() for _ in range(NUM_REGISTERS)]

    def read(self, register: int) -> int:
        """Read x`register`. x0 always reads as 0, regardless of
        anything ever written to it.
        """
        _validate_index(register)
        if register == 0:
            return 0
        return self._registers[register].read()

    def write(self, register: int, value: int) -> None:
        """Write `value` into x`register`. A write to x0 is silently
        discarded.
        """
        _validate_index(register)
        if register == 0:
            return
        self._registers[register].write_next(value)
        self._registers[register].commit()


def _validate_index(register: int) -> None:
    if not (0 <= register < NUM_REGISTERS):
        raise ValueError(f"register index must be 0-{NUM_REGISTERS - 1}, got {register}")
