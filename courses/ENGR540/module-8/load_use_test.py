"""A deliberate load-use test: one program with two back-to-back RAW
dependencies that need *different* resolutions.

    lw  x5, 0(x1)     produces x5
    add x6, x5, x2    consumes x5 immediately -- forwarding can't reach
                      this one (the loaded value doesn't exist until
                      MEMORY runs), so hazards.py must stall
    sub x7, x6, x3    consumes x6 immediately -- by the time this
                      reaches EX, add's result is already sitting in
                      EX_MEM, so this one should resolve through
                      ordinary forwarding, *no* stall needed

If the pipeline can't tell these two cases apart -- stalling both, or
forwarding both -- it either wastes a cycle it didn't need to, or
produces a wrong answer. Getting `stalls == 1` (not 0, not 2) out of
this one program is a real correctness signal about the hazard unit,
not just the final register values.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module-1"))

from cpu_simulator.cpu.pipelined_cpu import PIPELINE_DEPTH, PipelinedCPU, load_program
from cpu_simulator.cpu.trace import format_pipeline

PROGRAM = """
lw  x5, 0(x1)
add x6, x5, x2
sub x7, x6, x3
"""


def run(cpu: PipelinedCPU | None = None, trace: bool = True) -> PipelinedCPU:
    """x1 = 0 (the load's base address), Memory[0] = 100, x2 = 10,
    x3 = 5 -- so x5 = 100, x6 = 110, x7 = 105.
    """
    cpu = cpu or PipelinedCPU()
    cpu.registers.write(1, 0)
    cpu.registers.write(2, 10)
    cpu.registers.write(3, 5)
    cpu.data_memory.write_word(0, 100)
    n = load_program(cpu, PROGRAM)

    for cycle in range(1, n + PIPELINE_DEPTH + 1):  # +1: the stall costs a cycle
        cpu.step()
        if trace:
            print(format_pipeline(cpu.last_trace))
            print()
    return cpu


def main():
    print("=" * 60)
    print("MODULE 8: deliberate load-use test")
    print("=" * 60)
    cpu = run()
    print(f"x5={cpu.registers.read(5)} x6={cpu.registers.read(6)} x7={cpu.registers.read(7)}")
    assert cpu.registers.read(5) == 100
    assert cpu.registers.read(6) == 110
    assert cpu.registers.read(7) == 105
    print()

    print(f"stalls = {cpu.performance.data_hazard_stalls} (expect exactly 1 -- lw -> add only)")
    print(f"load_use_stalls = {cpu.performance.load_use_stalls}")
    assert cpu.performance.data_hazard_stalls == 1
    assert cpu.performance.load_use_stalls == 1
    print()
    print("Exactly one stall: the lw -> add dependency needed it, and the")
    print("add -> sub dependency didn't -- forwarding caught that one on")
    print("its own. The hazard logic is doing real, selective work here,")
    print("not just stalling on every dependency it sees.")
    print("=" * 60)


if __name__ == "__main__":
    main()
