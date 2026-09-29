"""The integrated exercise: register + ALU + clock, wired together for
the first time. Nothing here is new machinery -- Register, alu, and
Clock all already exist in module-1/cpu_simulator/, built and tested
independently. The only thing this file adds is wiring them into the
exact three-step cycle their own docstrings already describe:

    1. combinational logic computes the register's next value
       (here: alu(register.read(), 1, "ADD") -- this cycle's already-
       committed value plus one, via the Module 1 ALU, not Python's
       own `+`)
    2. clock.tick()
    3. register.commit() latches the value step 1 computed

The output (0, 1, 2, 3, ...) looks trivial. The pieces producing it are
not: a component that holds a value across time (Register), a component
that computes a new value from an old one (alu), and a component that
defines what "across time" even means (Clock). Wire those three
together and the result is stored state + computation + time -- the
actual skeleton every computer is built from, whether it's this loop or
a real RV32I core with 32 registers and a thousand-gate ALU.
"""

from __future__ import annotations

import sys
from pathlib import Path

# counter_system.py isn't part of the cpu_simulator package -- it's an
# external consumer of it, living in a sibling module-4/ folder. Always
# add module-1/ to sys.path (no __package__ check needed here, unlike
# cpu_simulator's own files: this script is only ever run directly,
# never imported as part of another package).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module-1"))

from cpu_simulator.alu import alu
from cpu_simulator.clock import Clock
from cpu_simulator.register import Register


def run_counter_system(num_cycles: int, width: int = 32) -> list[int]:
    """Wire a Register, the ALU, and a Clock into R_next = R + 1, run
    `num_cycles` clock ticks, and return the value read at the start of
    each cycle -- index 0 is cycle 0, before any tick has happened.
    """
    register = Register(width=width, reset_value=0)
    clock = Clock()

    values = [register.read()]
    for _ in range(num_cycles):
        result = alu(register.read(), 1, "ADD", width=width)
        register.write_next(result.value)   # step 1: computed against *this* cycle's state
        clock.tick()                        # step 2
        register.commit()                   # step 3: only now does read() change
        values.append(register.read())
    return values


def main():
    print("=" * 60)
    print("INTEGRATED SYSTEM: register + ALU + clock")
    print("=" * 60)

    values = run_counter_system(num_cycles=10)
    for cycle, value in enumerate(values):
        print(f"cycle {cycle}: {value}")

    print()
    print("Nothing above is new machinery -- Register, alu, and Clock")
    print("were each built and tested independently in module-1. This")
    print("loop just wires them together: stored state (Register) +")
    print("computation (alu) + time (Clock) is the skeleton of a computer.")
    print("=" * 60)


if __name__ == "__main__":
    main()
