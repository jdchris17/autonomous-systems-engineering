"""Module 5's first tiny program: the pseudo-instruction datapath
exercise. Everything this file needs -- CPU, the ("ADD", 3, 1, 2)-style
pseudo-instruction format, and its dispatcher (CPU.execute/CPU.run) --
already lives in module-1/cpu_simulator/cpu/pseudo_cpu.py. This file
adds no new machinery; it only writes the program and prints the
result.

The program:

    LOAD_IMM R1, 10
    LOAD_IMM R2, 7
    ADD      R3, R1, R2
    SUB      R4, R3, R2
    STORE    R4, [100]

as pseudo-instruction tuples. The last line's absolute address, [100],
isn't its own addressing mode -- STORE always means
MEM[R[base] + offset], so an absolute address is base=0 (x0, hardwired
to zero) with that address as the offset: ("STORE", 4, 0, 100).

Uses PseudoOpCPU, not cpu_simulator.cpu.cpu.CPU -- that's a different,
later class (Module 7's real decoded-instruction CPU) that didn't exist
when this file was written and isn't what this exercise is about. The
whole point here is running a program *without* any decoder or encoding
existing yet; PseudoOpCPU is deliberately still that same pre-decoder
class, unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module-1"))

from cpu_simulator.cpu.pseudo_cpu import PseudoOpCPU as CPU

PROGRAM = [
    ("LOAD_IMM", 1, 10),   # R1 = 10
    ("LOAD_IMM", 2, 7),    # R2 = 7
    ("ADD", 3, 1, 2),      # R3 = R1 + R2
    ("SUB", 4, 3, 2),      # R4 = R3 - R2
    ("STORE", 4, 0, 100),  # MEM[100] = R4
]


def main():
    cpu = CPU()
    cpu.run(PROGRAM)

    print("=" * 50)
    print("MODULE 5: FIRST TINY PROGRAM")
    print("=" * 50)
    for line in [
        "LOAD_IMM R1, 10",
        "LOAD_IMM R2, 7",
        "ADD      R3, R1, R2",
        "SUB      R4, R3, R2",
        "STORE    R4, [100]",
    ]:
        print(f"  {line}")
    print()
    for reg in [1, 2, 3, 4]:
        print(f"R{reg}         = {cpu.registers.read(reg)}")
    print(f"memory[100] = {cpu.data_memory.read_word(100)}")
    print(f"PC          = {cpu.pc.read()}")
    print("=" * 50)
    print("Stored state (registers, memory) + computation (the ALU) +")
    print("time (five ticks of the clock, five PC advances) -- a")
    print("program is now driving a machine model, not just a function")
    print("call producing a return value.")
    print("=" * 50)


if __name__ == "__main__":
    main()
