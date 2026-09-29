"""A branch-flush test: the exact program section 8 specs, checking one
extremely important pipeline principle -- speculative work must not
accidentally become architectural state.

    addi x1, x0, 1
    addi x2, x0, 1
    beq  x1, x2, target   taken (x1 == x2)

    addi x3, x0, 99       wrong-path -- fetched under the not-taken
    addi x4, x0, 99       prediction, must never retire

    target:
    addi x5, x0, 42

`hazards_demo.py` already covers a taken branch in general; this file
is narrower on purpose -- it exists to pin down that `x3`/`x4` never
observably become `99`, not just that the final answer (`x5 == 42`)
comes out right. A pipeline that computed the correct final state by
accident (say, by re-flushing the same registers back to a stale value
some other way) could pass a looser check and still have leaked
speculative state for a cycle or two along the way.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module-1"))

from cpu_simulator.cpu.pipelined_cpu import PIPELINE_DEPTH, PipelinedCPU, load_program
from cpu_simulator.cpu.trace import format_pipeline

PROGRAM = """
addi x1, x0, 1
addi x2, x0, 1
beq  x1, x2, target

addi x3, x0, 99
addi x4, x0, 99

target:
addi x5, x0, 42
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
    print("MODULE 8: branch-flush test")
    print("=" * 60)
    cpu = run()
    print(f"x3={cpu.registers.read(3)} x4={cpu.registers.read(4)} x5={cpu.registers.read(5)}")

    assert cpu.registers.read(3) == 0, "x3 must never observe the wrong-path 99"
    assert cpu.registers.read(4) == 0, "x4 must never observe the wrong-path 99"
    assert cpu.registers.read(5) == 42

    # Not just the final values -- confirm neither wrong-path instruction
    # ever reached writeback at all (retiring is what "became
    # architectural state" means here; a register merely *reading* 0
    # isn't the same guarantee as never having been written).
    for t in cpu.trace_history:
        assert not (t.wb_stage.valid and t.wb_stage.rd in (3, 4)), (
            f"cycle {t.cycle}: a wrong-path instruction retired into x{t.wb_stage.rd}"
        )

    print()
    print(f"branch_flush_cycles = {cpu.performance.branch_flush_cycles} (expect 1)")
    assert cpu.performance.branch_flush_cycles == 1
    print()
    print("addi x3,x0,99 and addi x4,x0,99 were fetched under the static")
    print("not-taken prediction, flushed once the branch resolved taken in")
    print("EX, and never retired at all -- speculative work discarded")
    print("before it could ever become architectural state.")
    print("=" * 60)


if __name__ == "__main__":
    main()
