"""Architectural performance statistics: the counters a real processor's
performance-monitoring unit would expose, plus the two ratios everything
else in this file's spec is building toward.

`PerformanceCounters` is deliberately pure data -- the same division of
labor `pipeline.py`'s registers and `trace.py`'s `StepTrace`/
`PipelineTrace` already use: something that only sees every intermediate
value in a cycle (`CPU.step()`, `PipelinedCPU.step()`) is the only thing
that can *build* one, so building it stays there; this file only holds
the shape and the two pure functions (`cpi`/`ipc`) that read it back.

Eight fields, matching section 8's own list exactly:

    instructions_retired   a real instruction (not a bubble) completed
                            writeback this cycle -- counted whether or
                            not it actually wrote a register (a store or
                            a branch retires too)
    total_cycles            one clock tick, regardless of what happened
                            during it
    stalls                  any cycle a hazard held `pc`/`IF_ID` in
                            place -- today that's only ever a load-use
                            stall (see `load_use_stalls` below), but the
                            two are kept as separate counters since a
                            future stall source (a structural hazard,
                            say) would count toward this one without
                            being a load-use stall
    load_use_stalls         specifically `hazards.py`'s load-use case
    branch_count             a branch instruction resolved in EX
    branches_taken           of those, how many actually redirected pc
    branch_flushes           IF_ID/ID_EX actually got flushed -- 1:1
                            with `branches_taken` in this design (every
                            taken branch flushes exactly once), but kept
                            distinct for the same forward-looking reason
                            as stalls/load_use_stalls: a predictor that
                            sometimes predicts taken correctly would
                            break that 1:1 relationship without this
                            being a different field
    forwarding_events        one per operand (`forward_a`/`forward_b`)
                            that resolved to something other than the
                            plain register value -- up to two per cycle,
                            each counted separately, since they're two
                            independent mux decisions

`CPU` (single-cycle) only ever increments `instructions_retired`,
`total_cycles`, `branch_count`, and `branches_taken` -- the other four
counters stay at their dataclass default of 0 forever, which is itself
the point: a single-cycle model structurally cannot stall, flush, or
forward, because nothing is ever in flight to stall, flush, or forward
between. `cpi(CPU's counters) == 1.0` always follows directly from that,
not from anything special-cased here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PerformanceCounters:
    instructions_retired: int = 0
    total_cycles: int = 0
    stalls: int = 0
    load_use_stalls: int = 0
    branch_count: int = 0
    branches_taken: int = 0
    branch_flushes: int = 0
    forwarding_events: int = 0


def cpi(counters: PerformanceCounters) -> float:
    """Cycles per instruction -- lower is better. Exactly 1.0 for the
    single-cycle model, always; >= 1.0 for the pipelined model once
    startup/drain and any stall/flush cycles are included, approaching
    1.0 from above as a program's real instruction count grows relative
    to that fixed per-run overhead.
    """
    return counters.total_cycles / counters.instructions_retired


def ipc(counters: PerformanceCounters) -> float:
    """Instructions per cycle -- the reciprocal of cpi(), higher is
    better. `<= 1` for this simple single-issue machine (at most one
    instruction can ever retire in a single cycle, single-cycle or
    pipelined) -- section 8's own bound, which falls directly out of
    `instructions_retired` never being able to exceed `total_cycles`
    for a machine that retires at most one instruction per tick.
    """
    return counters.instructions_retired / counters.total_cycles
