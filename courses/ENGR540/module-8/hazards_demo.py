"""Section 8's two remaining worked examples: a load-use stall and a
taken branch's flush, both rendered with `trace.py`'s new
`format_pipeline()` -- the same per-cycle, per-stage trace this file's
sibling `forwarding_demo.py` now also uses, replacing that file's own
one-off stage-summary printer now that a real one exists in the
library.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module-1"))

from cpu_simulator.cpu.pipelined_cpu import PIPELINE_DEPTH, PipelinedCPU, load_program
from cpu_simulator.cpu.trace import format_pipeline

LOAD_USE_PROGRAM = """
addi x1, x0, 100
sw   x1, 0(x0)
lw   x5, 0(x0)
add  x6, x5, x1
"""

BRANCH_PROGRAM = """
addi x1, x0, 5
addi x2, x0, 5
beq  x1, x2, skip
addi x10, x0, 999
addi x11, x0, 999
skip:
addi x3, x0, 42
"""


def run_load_use_stall(cpu: PipelinedCPU | None = None, trace: bool = True) -> PipelinedCPU:
    """A load immediately followed by a dependent instruction --
    forwarding alone can't bridge this one (the loaded value doesn't
    exist until MEMORY runs), so `hazards.py` stalls fetch/decode for
    one cycle and lets ordinary MEM_WB forwarding catch it from there.
    """
    cpu = cpu or PipelinedCPU()
    n = load_program(cpu, LOAD_USE_PROGRAM)
    for cycle in range(1, n + PIPELINE_DEPTH + 1):  # +1: the stall costs a cycle
        cpu.step()
        if trace:
            print(format_pipeline(cpu.last_trace))
            print()
    return cpu


def run_branch_flush(cpu: PipelinedCPU | None = None, trace: bool = True) -> PipelinedCPU:
    """A taken branch, resolved in EX under static not-taken prediction --
    the two instructions fetch/decode already grabbed on the wrong
    (fall-through) assumption get flushed, and pc redirects to the
    branch's target. addi x10/addi x11 must never actually retire.
    """
    cpu = cpu or PipelinedCPU()
    n = load_program(cpu, BRANCH_PROGRAM)
    for cycle in range(1, n + PIPELINE_DEPTH):
        cpu.step()
        if trace:
            print(format_pipeline(cpu.last_trace))
            print()
    return cpu


def main():
    print("=" * 60)
    print("MODULE 8: load-use hazard -- a stall, then forwarding")
    print("=" * 60)
    cpu = run_load_use_stall()
    print(f"x5 = {cpu.registers.read(5)} (expect 100)")
    print(f"x6 = {cpu.registers.read(6)} (expect 200)")
    assert cpu.registers.read(5) == 100
    assert cpu.registers.read(6) == 200
    assert any(t.stall for t in cpu.trace_history)
    print()

    print("=" * 60)
    print("MODULE 8: branch handling -- resolve in EX, flush on taken")
    print("=" * 60)
    cpu = run_branch_flush()
    print(f"x3  = {cpu.registers.read(3)} (expect 42, the skip target)")
    print(f"x10 = {cpu.registers.read(10)} (expect 0 -- flushed, never executed)")
    print(f"x11 = {cpu.registers.read(11)} (expect 0 -- flushed, never executed)")
    assert cpu.registers.read(3) == 42
    assert cpu.registers.read(10) == 0
    assert cpu.registers.read(11) == 0
    assert any(t.flush for t in cpu.trace_history)
    print()
    print("A load-use hazard bought one cycle with a stall, then let")
    print("forwarding finish the job; a taken branch discarded two")
    print("wrong-path instructions before they could ever retire.")
    print("=" * 60)


if __name__ == "__main__":
    main()
