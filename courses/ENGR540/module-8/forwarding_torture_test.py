"""A forwarding torture test: five instructions, each one depending on
the immediately preceding instruction's result -- the worst case a RAW
hazard can present a pipeline (every possible gap between producer and
consumer collapsed to zero).

    addi x1, x0, 1
    addi x1, x1, 1    depends on the previous instruction, immediately
    addi x1, x1, 1    same
    addi x1, x1, 1    same
    addi x1, x1, 1    same

x1 = 5 is the easy part -- `test_pipelined_cpu.py` already covers
forwarding correctness in isolation. What this file actually measures
is efficiency: a correct forwarding network resolves every one of these
four back-to-back dependencies for free, in the same cycle each
consumer would have entered EX anyway. A design that could only resolve
a RAW hazard by stalling until the producer's value reached the
register file (through writeback) would need real stall cycles for
each of these four dependencies instead. `stalls == 0` here, with the
total cycle count landing exactly on the fill+drain minimum
(`n + PIPELINE_DEPTH - 1`, no stall overhead at all), is what proves
forwarding is actually doing the work instead of a stall quietly
picking up the slack underneath it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module-1"))

from cpu_simulator.cpu.pipelined_cpu import PIPELINE_DEPTH, PipelinedCPU, load_program
from cpu_simulator.cpu.trace import format_pipeline

PROGRAM = """
addi x1, x0, 1
addi x1, x1, 1
addi x1, x1, 1
addi x1, x1, 1
addi x1, x1, 1
"""


def run(cpu: PipelinedCPU | None = None, trace: bool = True) -> PipelinedCPU:
    cpu = cpu or PipelinedCPU()
    n = load_program(cpu, PROGRAM)
    for cycle in range(1, n + PIPELINE_DEPTH):
        cpu.step()
        if trace:
            print(format_pipeline(cpu.last_trace))
            print()
    return cpu


def main():
    print("=" * 60)
    print("MODULE 8: forwarding torture test")
    print("=" * 60)
    cpu = run()
    print(f"x1 = {cpu.registers.read(1)} (expect 5)")
    assert cpu.registers.read(1) == 5
    print()

    ideal_cycles = 5 + PIPELINE_DEPTH - 1
    print(f"stalls = {cpu.performance.data_hazard_stalls} (expect 0 -- forwarding, not stalling)")
    print(f"cycles = {cpu.clock.cycle} (ideal fill+drain minimum: {ideal_cycles})")
    print(f"forwarding_events = {cpu.performance.forwarding_events} "
          f"(expect 4 -- one per chained dependency)")
    assert cpu.performance.data_hazard_stalls == 0
    assert cpu.clock.cycle == ideal_cycles
    assert cpu.performance.forwarding_events == 4
    print()
    print("Four back-to-back RAW dependencies, zero stall cycles -- every")
    print("one of them resolved through forwarding instead. A naive")
    print("stall-until-writeback design would have needed real stall")
    print("cycles for each of these four; this pipeline needed none.")
    print("=" * 60)


if __name__ == "__main__":
    main()
