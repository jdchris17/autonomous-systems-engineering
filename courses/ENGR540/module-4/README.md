# Module 4 — Register + ALU + clock

The integrated exercise: wire three independently-built module-1
components together for the first time and watch a counter fall out.

## Files

| File | What it does |
|---|---|
| [`counter_system.py`](counter_system.py) | `run_counter_system(num_cycles, width=32)`. A `Register` starts at 0; each cycle computes `R_next = R + 1` via `alu(register.read(), 1, "ADD")` from `module-1/cpu_simulator/alu.py` — not Python's own `+` — stages it with `write_next()`, ticks a `Clock`, then `commit()`s. `main()` prints `cycle N: value` for 10 cycles, matching the spec's example output exactly. |
| [`test_counter_system.py`](test_counter_system.py) | Checks the exact 3-cycle example from the spec, a longer run, and — since the ALU does the arithmetic — that the register wraps correctly at its own width (`width=2` rolls `0,1,2,3,0,1`; `width=1` toggles `0,1,0,1,0`), for free, with no special-case wraparound logic in this file. |

## Nothing new was built

Every moving part — `Register`, `alu`, `Clock` — already exists in
[`module-1/cpu_simulator/`](../module-1/README.md), built and tested on
its own. This file only wires them into the three-step cycle their own
docstrings already describe: compute the next value against *this*
cycle's committed state (`alu(...)`), `clock.tick()`, then
`register.commit()` latches it — the same write-then-commit discipline
`register.py`/`fsm.py` use, just driven by an explicit loop instead of a
test.

`counter_system.py` lives in its own folder (not inside `cpu_simulator/`)
since it's a consumer of that package, not a new primitive being added
to it — it puts `module-1/` on `sys.path` before importing
`cpu_simulator.*`.

## Why it matters

The output is just `0, 1, 2, 3, ...`. What produces it is a component
that holds a value across time (`Register`), a component that computes a
new value from an old one (`alu`), and a component that defines what
"across time" even means (`Clock`). Stored state + computation + time —
the skeleton of a computer, at the smallest scale that still counts as
one.

## Run it

```
python counter_system.py
```
