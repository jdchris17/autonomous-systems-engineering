""""Priority matters" -- the spec's own worked example -- run on the
real `PipelinedCPU`, with a per-cycle trace of what each pipeline stage
holds. Uses `trace.py`'s `format_pipeline()`, the library's own
pipeline-trace renderer (added alongside `hazards.py`/branch handling)
-- this file used to build its own one-off stage-summary printer before
that existed; now that a real one does, reusing it is the obvious call
rather than keeping two divergent pipeline-trace formats around.

    add  x5, x1, x2      x5 = 10 + 20 = 30
    addi x5, x5, 1       x5 = 31          <- must forward from EX_MEM,
                                              not the stale register file
    sub  x6, x5, x3      x6 = 31 - 1 = 30 <- must forward the *newer*
                                              EX_MEM value, not MEM_WB's
                                              older one still in flight
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module-1"))

from cpu_simulator.cpu.pipelined_cpu import PIPELINE_DEPTH, PipelinedCPU, load_program
from cpu_simulator.cpu.trace import format_pipeline

PROGRAM = """
add  x5, x1, x2
addi x5, x5, 1
sub  x6, x5, x3
"""


def run(cpu: PipelinedCPU | None = None, trace: bool = True) -> PipelinedCPU:
    """Load and run PROGRAM to completion (including drain) on a fresh
    (or given) PipelinedCPU, printing a full pipeline trace each cycle
    by default. Returns the CPU so a caller can inspect final state.
    """
    cpu = cpu or PipelinedCPU()
    cpu.registers.write(1, 10)
    cpu.registers.write(2, 20)
    cpu.registers.write(3, 1)
    n = load_program(cpu, PROGRAM)

    for cycle in range(1, n + PIPELINE_DEPTH):
        cpu.step()
        if trace:
            print(format_pipeline(cpu.last_trace))
            print()
    return cpu


def main():
    print("=" * 60)
    print("MODULE 8: forwarding -- \"priority matters\"")
    print("=" * 60)
    cpu = run()
    print()
    print(f"x5 = {cpu.registers.read(5)} (expect 31)")
    print(f"x6 = {cpu.registers.read(6)} (expect 30)")
    assert cpu.registers.read(5) == 31
    assert cpu.registers.read(6) == 30
    print()
    print("x6 came out 30, not 29 -- the sub read x5's freshest value")
    print("(31, still in EX_MEM) rather than the older one already sitting")
    print("in MEM_WB (30 would have been wrong: 30 - 1 = 29 is not what a")
    print("real, non-forwarding-broken pipeline produces here).")
    print("=" * 60)


if __name__ == "__main__":
    main()
