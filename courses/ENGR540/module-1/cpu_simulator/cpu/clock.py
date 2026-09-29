"""An abstract simulation clock -- not nanosecond-accurate electronics,
just the discrete event a synchronous digital circuit organizes itself
around.

The two-phase discipline register.py's Register class implements
(write_next() stages a value, commit() latches it) exists specifically so
that "compute next state" and "the state actually changes" can be kept
apart in time:

    1. combinational logic computes every involved register's next
       value (write_next() calls, each one reading this cycle's
       *current*, already-committed state -- never a value some other
       register already committed earlier in the same cycle)
    2. clock.tick()
    3. every register involved has its commit() called, latching the
       value step 1 computed

Deliberately just a cycle counter here, not an object that owns a list
of registers and calls commit() on all of them itself -- which registers
exist and which clock domain they belong to is a decision for whatever
wires up a circuit (a future datapath/system module), not this class's
job. tick() advancing the count *is* the discrete event; what happens in
response to it is the caller's concern.
"""

from __future__ import annotations


class Clock:
    def __init__(self):
        self.cycle = 0

    def tick(self) -> None:
        """Advance one cycle."""
        self.cycle += 1
