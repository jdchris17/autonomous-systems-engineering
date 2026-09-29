"""Module 6's first real RV32I program: text assembled into actual
32-bit machine words, stored in memory, and decoded back to fields.

    addi x1, x0, 10
    addi x2, x0, 7
    add  x3, x1, x2
    sub  x4, x3, x2
    sw   x4, 0(x0)

Deliberately NOT executed from those decoded fields when this file was
written -- driving a datapath off an Instruction the way a real control
unit would was explicitly Module 7's job, which didn't exist yet
("By Module 7, the control unit will drive the actual datapath from
those decoded fields"). Module 7 exists now
(cpu_simulator/cpu/control.py, datapath.py, cpu.py) and can run this
exact program from its decoded fields for real -- see
module-7/run_program.py. This file is kept exactly as originally
written anyway, as the historical record of what was and wasn't
possible at the point in the course where it was built.

What this file runs instead is the pseudo-instruction tuple format
PseudoOpCPU (module-1/cpu_simulator/cpu/pseudo_cpu.py) understands from
Module 5. The program above and the tuple list below are the same five
operations in two different representations -- one real machine code,
one the artificial stand-in that predates a decoder-driven execution
engine:

    real RV32I text          equivalent pseudo-instruction tuple
    -----------------------  -----------------------------------
    addi x1, x0, 10          ("LOAD_IMM", 1, 10)
    addi x2, x0, 7           ("LOAD_IMM", 2, 7)
    add  x3, x1, x2          ("ADD", 3, 1, 2)
    sub  x4, x3, x2          ("SUB", 4, 3, 2)
    sw   x4, 0(x0)           ("STORE", 4, 0, 0)

"addi rd, x0, imm" and "LOAD_IMM" are the same operation for the same
reason PseudoOpCPU's step_load_immediate docstring already gives: real
RISC-V's own "li" pseudo-instruction is literally "addi rd, x0, imm".
Running the tuple form is how this file demonstrates the architectural
result (x1=10, x2=7, x3=17, x4=10, Memory[0]=10) without building a
control unit early.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module-1"))

from cpu_simulator.cpu.pseudo_cpu import PseudoOpCPU as CPU
from cpu_simulator.isa.assembler import assemble
from cpu_simulator.isa.decoder import decode, to_asm
from cpu_simulator.isa.instruction import Instruction
from cpu_simulator.memory.memory import Memory

PROGRAM_ASM = """
addi x1, x0, 10
addi x2, x0, 7
add  x3, x1, x2
sub  x4, x3, x2
sw   x4, 0(x0)
"""

PROGRAM_PSEUDO_OPS = [
    ("LOAD_IMM", 1, 10),
    ("LOAD_IMM", 2, 7),
    ("ADD", 3, 1, 2),
    ("SUB", 4, 3, 2),
    ("STORE", 4, 0, 0),
]


def assemble_and_store(program_asm: str, instruction_memory: Memory,
                        base_address: int = 0) -> list[int]:
    """"Your assembler should encode it. Your memory should store it."""
    words = assemble(program_asm, base_address=base_address)
    for i, word in enumerate(words):
        instruction_memory.write_word(base_address + i * 4, word)
    return words


def fetch_and_decode(instruction_memory: Memory, base_address: int,
                      count: int) -> list[Instruction]:
    """"Your decoder should recover the instruction fields." Reads each
    word back out of memory (not just off the `words` list assemble()
    returned) -- the fields have to survive an actual round trip
    through storage, not just an in-memory Python list.
    """
    return [decode(instruction_memory.read_word(base_address + i * 4))
            for i in range(count)]


def main():
    print("=" * 70)
    print("MODULE 6: A GOOD FIRST ACTUAL RV32I PROGRAM")
    print("=" * 70)
    print(PROGRAM_ASM.strip())
    print()

    instruction_memory = Memory(size_bytes=64)
    words = assemble_and_store(PROGRAM_ASM, instruction_memory)

    print("Assembled and stored in instruction memory:")
    for i, word in enumerate(words):
        print(f"  [{i * 4:#04x}] {word:#010x}")
    print()

    print("Read back out of instruction memory and decoded:")
    instructions = fetch_and_decode(instruction_memory, base_address=0, count=len(words))
    for inst in instructions:
        print(f"  {to_asm(inst):<18} mnemonic={inst.mnemonic:<5} "
              f"rd={inst.rd} rs1={inst.rs1} rs2={inst.rs2} imm={inst.immediate}")
    print()

    print("Not executed from those decoded fields -- that's Module 7's")
    print("control unit. Running the equivalent pseudo-instruction program")
    print("(Module 5's cpu.run()) instead, to reach the architectural result:")
    print()

    cpu = CPU()
    cpu.run(PROGRAM_PSEUDO_OPS)

    for r in (1, 2, 3, 4):
        print(f"  x{r} = {cpu.registers.read(r)}")
    print(f"  Memory[0] = {cpu.data_memory.read_word(0)}")

    assert cpu.registers.read(1) == 10
    assert cpu.registers.read(2) == 7
    assert cpu.registers.read(3) == 17
    assert cpu.registers.read(4) == 10
    assert cpu.data_memory.read_word(0) == 10
    print()
    print("Matches the architectural target exactly.")
    print("=" * 70)


if __name__ == "__main__":
    main()
