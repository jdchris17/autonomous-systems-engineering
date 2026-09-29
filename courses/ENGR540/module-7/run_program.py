"""Section 34's milestone: run a real program. Assembles a backward-
branch summation loop, runs it through the full decode-and-execute
pipeline, and confirms Memory[0] == 15. "If that executes from encoded
RV32I machine words, you have a legitimate processor simulator" -- the
spec's own bar for this file to clear.

Every individual capability this program exercises (immediate
arithmetic, register arithmetic, a backward branch, a PC change, a
memory store) already has its own test in test_cpu.py. This file is the
first place all five are exercised by *one* program at once, with a
full compact trace showing exactly how.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module-1"))

from cpu_simulator.cpu.cpu import CPU
from cpu_simulator.cpu.trace import format_compact
from cpu_simulator.isa.assembler import assemble

PROGRAM = """
addi x1, x0, 5
addi x2, x0, 0

loop:
    add  x2, x2, x1
    addi x1, x1, -1
    bne  x1, x0, loop

sw   x2, 0(x0)
"""

# Dynamic instruction count, not the assembled word count: 2 setup
# instructions, then the 3-instruction loop body runs 5 times (once per
# value of x1 from 5 down to 1), then the final store.
NUM_STEPS = 2 + 5 * 3 + 1


def run(cpu: CPU | None = None, trace: bool = True) -> CPU:
    """Load and run PROGRAM on a fresh (or given) CPU, printing a
    compact trace by default. Returns the CPU so a caller can inspect
    its final state.
    """
    cpu = cpu or CPU()
    words = assemble(PROGRAM)
    for i, word in enumerate(words):
        cpu.instruction_memory.write_word(i * 4, word)

    for _ in range(NUM_STEPS):
        cpu.step()
        if trace:
            print(format_compact(cpu.last_trace))
    return cpu


def main():
    print("=" * 60)
    print("SECTION 34: a real program -- 5 + 4 + 3 + 2 + 1")
    print("=" * 60)
    cpu = run()
    print()
    print(f"Memory[0] = {cpu.data_memory.read_word(0)}")
    assert cpu.data_memory.read_word(0) == 15
    print()
    print("5 + 4 + 3 + 2 + 1 = 15, executed from encoded RV32I machine")
    print("words: immediate arithmetic, register arithmetic, a backward")
    print("branch, PC changes, and a memory store, all in one program.")
    print("=" * 60)


if __name__ == "__main__":
    main()
