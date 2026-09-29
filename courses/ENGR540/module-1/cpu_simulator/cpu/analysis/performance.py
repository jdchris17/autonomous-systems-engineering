"""Architectural performance statistics: the counters a real processor's
performance-monitoring unit would expose, plus the higher-level metrics
(`cpi`/`ipc`/`cpi_breakdown`) everything else in this file's spec is
building toward.

`PerformanceCounters` is deliberately pure data -- the same division of
labor `pipeline.py`'s registers and `trace.py`'s `StepTrace`/
`PipelineTrace` already use: something that only sees every intermediate
value in a cycle (`CPU.step()`, `PipelinedCPU.step()`) is the only thing
that can *build* one, so building it stays there; this file only holds
the shape and the pure functions that read it back.

Lives in `cpu_simulator/cpu/analysis/` (moved here from `cpu_simulator/
cpu/` in Module 10's repository update) for the same reason `memory.py`
moved out of `cpu/` into `memory/` in Module 9: "performance analysis is
now conceptually separate from processor functionality." `CPU`/
`PipelinedCPU` *use* this file; they don't need to sit next to it.

Section 35's own minimum field list, grouped the same way the spec
groups it:

    instructions_retired   a real instruction (not a bubble) completed
                            writeback this cycle -- counted whether or
                            not it actually wrote a register (a store or
                            a branch retires too)
    total_cycles            one clock tick, regardless of what happened
                            during it

    alu_instructions        retired, and none of the three categories
    load_instructions       below -- every retiring instruction falls
    store_instructions      into exactly one of these four "instruction
    branch_instructions     mix" buckets, `control.mem_read`/`mem_write`/
                            `branch is not None` deciding which, ALU
                            catching everything else (R-type, I-type
                            arithmetic, `LUI`/`AUIPC`, and `JAL`/`JALR`,
                            none of which have their own bucket here)

    data_hazard_stalls      any cycle a *data* hazard held `pc`/`IF_ID`
                            in place -- today that's only ever a
                            load-use stall (see `load_use_stalls`), kept
                            as a separate, broader counter for the same
                            forward-looking reason `load_use_stalls`
                            itself is split out from it: a future data
                            hazard source would count toward this one
                            without being a load-use stall specifically
    load_use_stalls         specifically `hazards.py`'s load-use case
    branch_flush_cycles     `IF_ID`/`ID_EX` actually got flushed by a
                            taken branch -- 1:1 with `branches_taken` in
                            this design, but kept as its own counter for
                            the same reason: a predictor that sometimes
                            predicts taken correctly would break that
                            1:1 relationship without this being separate
    memory_stall_cycles     cycles held by a multi-cycle data-memory
                            access (section 58's cache-backed MEM stage)
                            -- a *structural* hazard, not a data hazard,
                            so counted apart from `data_hazard_stalls`

    cache_accesses          mirrors `PipelinedCPU.data_cache.stats`
    cache_hits              exactly -- `Cache` is the ground truth for
    cache_misses            its own hit/miss bookkeeping, this file
    dirty_writebacks        just brings it into the same unified report
                            rather than re-deriving or re-counting it

Two fields beyond the spec's own minimum list, kept from Module 8
because they're already correct, already tested, and "at minimum" never
meant "and nothing else": `branches_taken` (of `branch_instructions`,
how many actually redirected `pc`) and `forwarding_events` (one per
operand resolved to something other than the plain register value).

`CPU` (single-cycle) only ever moves `instructions_retired`,
`total_cycles`, the four instruction-mix counters, and `branches_taken`
-- every stall/flush/cache counter stays at its dataclass default of 0
forever, which is itself the point: a single-cycle model structurally
cannot stall, flush, forward, or miss a cache it doesn't have, because
nothing is ever in flight between stages to do any of those with.
`cpi(CPU's counters) == 1.0` always follows directly from that, not
from anything special-cased here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PerformanceCounters:
    instructions_retired: int = 0
    total_cycles: int = 0

    alu_instructions: int = 0
    load_instructions: int = 0
    store_instructions: int = 0
    branch_instructions: int = 0

    data_hazard_stalls: int = 0
    load_use_stalls: int = 0
    branch_flush_cycles: int = 0
    memory_stall_cycles: int = 0

    cache_accesses: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    dirty_writebacks: int = 0

    branches_taken: int = 0
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
    better. `<= 1` for this simple single-issue, five-stage machine on
    any long-running workload (at most one instruction can ever retire
    in a single cycle, single-cycle or pipelined) -- section 37's own
    bound, which falls directly out of `instructions_retired` never
    being able to exceed `total_cycles` for a machine that retires at
    most one instruction per tick. A later superscalar design (able to
    retire more than one instruction per cycle) is exactly the case
    where `IPC` could finally exceed 1 -- section 37's own reason for
    tracking both metrics now, ahead of that.
    """
    return counters.instructions_retired / counters.total_cycles


@dataclass
class CpiBreakdown:
    base_cpi: float
    data_hazard_contribution: float
    branch_contribution: float
    memory_contribution: float
    actual_cpi: float


def cpi_breakdown(counters: PerformanceCounters) -> CpiBreakdown:
    """Section 36's own worked shape: a fixed `base_cpi` of 1.0 (the
    single-issue ideal -- one instruction, one cycle, no stalls at all)
    plus one contribution per stall category, each just that category's
    stall cycles spread over the whole run
    (`category_cycles / instructions_retired`), with `actual_cpi` their
    sum. Deliberately *not* forced to equal `cpi(counters)` (which
    divides `total_cycles` by `instructions_retired` directly): pipeline
    fill/drain overhead is real cycles that aren't attributed to any of
    the three named categories here, so for a short-enough run the two
    numbers can legitimately disagree -- this is an attribution tool
    telling you *why* the CPI looks the way it does, not a second way
    of computing the same number.
    """
    base_cpi = 1.0
    data_hazard_contribution = counters.data_hazard_stalls / counters.instructions_retired
    branch_contribution = counters.branch_flush_cycles / counters.instructions_retired
    memory_contribution = counters.memory_stall_cycles / counters.instructions_retired
    return CpiBreakdown(
        base_cpi=base_cpi,
        data_hazard_contribution=data_hazard_contribution,
        branch_contribution=branch_contribution,
        memory_contribution=memory_contribution,
        actual_cpi=base_cpi + data_hazard_contribution + branch_contribution + memory_contribution,
    )
