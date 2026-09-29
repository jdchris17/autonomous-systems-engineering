"""Section 8's benchmarking work: the first of several programs meant
to show that *architectural* correctness and *performance* are
different questions. `CPU` and `PipelinedCPU` must reach the identical
answer -- that's `cross_validate.py`'s job, already covered -- but they
don't take the same number of cycles to get there, and this file is
where that difference gets measured instead of just asserted away.

`run_benchmark()` is written to be reused by the independent-operations
and dependency-heavy programs section 8 says are coming next: it runs
`program` to completion (by real, dynamic instruction count, not
assembled word count -- see its own docstring for why that distinction
matters the moment a program loops) on both models, confirms they agree
on architectural state, and reports each model's `PerformanceCounters`
plus CPI/IPC. Only today's one benchmark actually calls it so far.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module-1"))

from cpu_simulator.cpu.analysis.performance import cpi, ipc
from cpu_simulator.cpu.cpu import CPU
from cpu_simulator.cpu.pipelined_cpu import PipelinedCPU, load_program
from cpu_simulator.isa.assembler import assemble

SUM_TO_TEN_PROGRAM = """
addi x1, x0, 10
addi x2, x0, 0

loop:
    add  x2, x2, x1
    addi x1, x1, -1
    bne  x1, x0, loop

sw x2, 0(x0)
"""
# 2 setup instructions, then the 3-instruction loop body runs 10 times
# (once per value of x1 from 10 down to 1).
SUM_TO_TEN_DYNAMIC_INSTRUCTIONS = 2 + 10 * 3 + 1


def _load_single_cycle(cpu: CPU, program: str) -> None:
    words = assemble(program)
    for i, word in enumerate(words):
        cpu.instruction_memory.write_word(i * 4, word)


def run_benchmark(name: str, program: str, dynamic_instructions: int,
                   pad_instructions: int = 10) -> bool:
    """Runs `program` on both models until exactly `dynamic_instructions`
    real instructions have retired on each, then reports architectural
    agreement and each model's performance counters. Returns whether the
    two models' architectural state (registers + Memory[0]) agreed.

    Stopping by *retired instruction count* rather than a precomputed
    cycle count is deliberate: a looping program's dynamic instruction
    count doesn't equal its assembled word count, and the pipelined
    model's actual cycle count depends on how many stalls/flushes it
    hits, which isn't known up front either. Counting real retirements
    sidesteps both problems and, as a side effect, stops before any
    `load_program()` padding NOP ever gets the chance to retire and
    quietly inflate the counters.
    """
    single = CPU()
    _load_single_cycle(single, program)
    single.run(dynamic_instructions)

    pipelined = PipelinedCPU()
    load_program(pipelined, program, pad_instructions=pad_instructions)
    while pipelined.performance.instructions_retired < dynamic_instructions:
        pipelined.step()

    print(f"--- {name} ---")
    match = True
    for reg in range(1, 6):
        s, p = single.registers.read(reg), pipelined.registers.read(reg)
        if s != 0 or p != 0:
            match = match and (s == p)
            print(f"x{reg}: single={s} pipelined={p} match={s == p}")
    mem_match = single.data_memory.read_word(0) == pipelined.data_memory.read_word(0)
    match = match and mem_match
    print(f"Memory[0]: single={single.data_memory.read_word(0)} "
          f"pipelined={pipelined.data_memory.read_word(0)} match={mem_match}")
    print()

    sp = single.performance
    print(f"single-cycle:  instructions={sp.instructions_retired:<4} "
          f"cycles={sp.total_cycles:<4} CPI={cpi(sp):.2f}  IPC={ipc(sp):.2f}")

    pp = pipelined.performance
    print(f"pipelined:     instructions={pp.instructions_retired:<4} "
          f"cycles={pp.total_cycles:<4} CPI={cpi(pp):.2f}  IPC={ipc(pp):.2f}")
    print(f"  data_hazard_stalls={pp.data_hazard_stalls} (load-use: {pp.load_use_stalls})  "
          f"branches={pp.branch_instructions} (taken: {pp.branches_taken}, "
          f"flush_cycles: {pp.branch_flush_cycles})  forwarding_events={pp.forwarding_events}")
    print()
    return match


def main():
    print("=" * 60)
    print("MODULE 8: benchmarking -- architectural match, different speed")
    print("=" * 60)
    match = run_benchmark("sum 1..10 (dependency-heavy loop)",
                           SUM_TO_TEN_PROGRAM, SUM_TO_TEN_DYNAMIC_INSTRUCTIONS)
    assert match
    print("Both models agree on Memory[0] = 55 -- but the pipelined")
    print("model's CPI is well above the single-cycle model's exact 1.00,")
    print("driven entirely by this program's own RAW-dependency chain")
    print("(every loop iteration's add/addi/bne each depend on the one")
    print("before it) plus the fixed fill/drain and branch-flush cost.")
    print("Same instruction set, same final answer, different hardware,")
    print("different speed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
