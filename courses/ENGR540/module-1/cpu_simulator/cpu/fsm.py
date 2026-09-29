"""A finite-state machine, built the way sequential control actually
works in hardware rather than as a plain if/elif dispatcher: state lives
explicitly in one place, next-state logic is a pure function of (current
state, inputs), and the state only actually changes on a commit -- the
same write-then-commit split register.py's Register uses, for the same
reason (see clock.py's docstring). A state register genuinely is just a
Register whose stored value happens to mean "which state," so it should
behave the same way.

The example machine here is a single-shot load/process/done cycle:

    IDLE    --start-->    LOAD
    LOAD    --loaded-->   PROCESS
    PROCESS --complete--> DONE
    DONE    (any tick)--> IDLE

`reset` overrides every other rule from any state. Any (state, input)
pair not covered above holds -- the FSM stays where it is, the same way
a real state register does when nothing relevant is asserted that cycle.
"""

from __future__ import annotations

STATES = ("IDLE", "LOAD", "PROCESS", "DONE")
INPUTS = ("start", "loaded", "complete", "reset")


def transition(state: str, input_signal: str | None) -> str:
    """Pure combinational next-state logic -- no side effects, doesn't
    touch any FSM instance. Kept as a standalone function (not a method)
    so it can be tested and reasoned about as the truth table it is,
    independent of anything about how/when it gets committed.
    """
    _validate_state(state)
    if input_signal is not None:
        _validate_input(input_signal)

    if input_signal == "reset":
        return "IDLE"
    if state == "DONE":
        return "IDLE"
    if state == "IDLE" and input_signal == "start":
        return "LOAD"
    if state == "LOAD" and input_signal == "loaded":
        return "PROCESS"
    if state == "PROCESS" and input_signal == "complete":
        return "DONE"
    return state  # hold: nothing relevant asserted this cycle


class FSM:
    """Wraps `transition` with the same two-phase discipline Register
    has: compute_next() stages a result without changing `state`;
    commit() latches it. Real sequential control works this way so that
    every piece of next-state logic in a circuit reads the *same*
    cycle's state, never a value something else already committed this
    same cycle.
    """

    def __init__(self, initial_state: str = "IDLE"):
        _validate_state(initial_state)
        self._state = initial_state
        self._next_state = None

    @property
    def state(self) -> str:
        """The last *committed* state -- never a staged, uncommitted one."""
        return self._state

    def compute_next(self, input_signal: str | None) -> None:
        """Stage transition(current state, input_signal) as the next
        state. Does not change `state` -- the same role write_next()
        plays for a Register.
        """
        self._next_state = transition(self._state, input_signal)

    def commit(self) -> None:
        """Latch the staged state, if compute_next() was called since
        the last commit(). If it wasn't, the FSM holds its state -- the
        same as a Register whose write_next() wasn't called that cycle.
        """
        if self._next_state is not None:
            self._state = self._next_state
            self._next_state = None

    def step(self, input_signal: str | None) -> str:
        """compute_next() then commit() in one call, for callers that
        don't need the two phases kept apart -- e.g. walking through a
        known sequence of inputs one at a time.
        """
        self.compute_next(input_signal)
        self.commit()
        return self._state


def _validate_state(state: str) -> None:
    if state not in STATES:
        raise ValueError(f"unknown state {state!r}; expected one of {STATES}")


def _validate_input(input_signal: str) -> None:
    if input_signal not in INPUTS:
        raise ValueError(f"unknown input {input_signal!r}; expected one of {INPUTS} or None")
