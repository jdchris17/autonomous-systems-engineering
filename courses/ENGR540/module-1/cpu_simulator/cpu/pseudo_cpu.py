"""PseudoOpCPU: the *original* integration point, kept intentionally
unchanged now that a real decoded-instruction CPU exists (cpu.py, in
this same directory). This class predates control.py/datapath.py/
decoder.py being wired together at all -- it exposes a few artificial
internal operations (step_alu_op, step_load_word, step_store_word,
step_load_immediate) that a caller invokes directly with plain register
indices and immediates, standing in for what a real decoded instruction
triggers.

run()/execute() accept a temporary pseudo-instruction format -- plain
tuples like ("ADD", 3, 1, 2) meaning R3 = R1 + R2, or ("LOAD", 5, 2, 12)
meaning R5 = MEM[R2 + 12] -- deliberately not real 32-bit RISC-V machine
code. This is the abstraction boundary Module 5 and Module 6 were built
on opposite sides of: Module 5 asks how a CPU moves data through its
datapath once it knows what to do; Module 6 asks how "what to do" gets
encoded into 32 bits in the first place. These tuples let run() exercise
the whole datapath (pc, registers, alu, memory) without any encoding
existing yet -- module-5/ and module-6/ still use exactly this class for
that reason, not because it's been superseded into irrelevance.

This file used to be named cpu.py and its class was named CPU. Renamed
once cpu.py's own real, decoded-instruction-driven CPU existed and
needed that name for itself -- two classes both called "CPU" in the
same package, one pseudo-op and one real, would have been confusing in
exactly the way this course has avoided elsewhere (see register.py's
README note about retiring its own old ProgramCounter for the same
reason). PseudoOpCPU's behavior is completely unchanged from before the
rename; only the name changed, and every test below still passes
unmodified in substance.

Component roles:

    self.pc                 ProgramCounter (cpu/pc.py)
    self.registers           RegisterFile (cpu/register_file.py)
    self.alu                 the alu() function itself (core/alu.py) --
                              there's no ALU *class* to instantiate, alu.py
                              already exposes exactly the function this
                              attribute needs to be
    self.instruction_memory  Memory (memory/memory.py)
    self.data_memory         Memory, a *separate* instance -- see
                              memory.py's docstring on why one Memory
                              class backs two independent address spaces
    self.clock                Clock (cpu/clock.py) -- not in the spec's
                               literal minimum list, but added here since
                               a CPU that can "execute" cycles needs one;
                               see the note below

Why self.clock, when the spec's minimum list didn't include it: every
step_*() method below needs to advance time as a distinct, visible
event (the same argument module-4's integrated exercise made for having
a Clock at all), and register_file.write()/pc.advance() already commit
immediately on their own -- so clock.tick() here isn't gating those
commits the way it does in a component that exposes write_next()/
commit() directly. It's this CPU's own record of "how many cycles have
elapsed," kept for the same reason a real CPU has a cycle counter, not
because anything here is still waiting to be latched.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    # See core/adder.py's module docstring for why this is needed to run
    # this file directly (e.g. VS Code's Run button) rather than via
    # `python -m`. Three .parent calls: cpu.py -> cpu/ -> cpu_simulator/
    # -> module-1/.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.core.alu import OPERATIONS as ALU_OPERATIONS
from cpu_simulator.core.alu import alu
from cpu_simulator.cpu.clock import Clock
from cpu_simulator.cpu.pc import ProgramCounter
from cpu_simulator.cpu.register_file import RegisterFile
from cpu_simulator.memory.memory import Memory

WIDTH = 32
MASK = (1 << WIDTH) - 1


class PseudoOpCPU:
    def __init__(self, instruction_memory_bytes: int = 1024, data_memory_bytes: int = 1024):
        self.pc = ProgramCounter()
        self.registers = RegisterFile()
        self.alu = alu
        self.instruction_memory = Memory(size_bytes=instruction_memory_bytes)
        self.data_memory = Memory(size_bytes=data_memory_bytes)
        self.clock = Clock()

    def step_alu_op(self, operation: str, rd: int, rs1: int, rs2: int):
        """rd = rs1 <operation> rs2, via this CPU's own ALU -- an
        artificial stand-in for what a decoded R-type instruction
        (ADD/SUB/AND/OR/XOR/SLL/SRL/SRA/SLT/SLTU) will eventually
        trigger. Advances pc by the normal instruction width and counts
        one cycle, same as a real single-cycle implementation would.
        """
        a = self.registers.read(rs1)
        b = self.registers.read(rs2)
        result = self.alu(a, b, operation, width=WIDTH)
        self.registers.write(rd, result.value)
        self.pc.advance()
        self.clock.tick()
        return result

    def step_add(self, rd: int, rs1: int, rs2: int):
        """rd = rs1 + rs2. A thin, explicitly-named wrapper over
        step_alu_op, kept for call sites that only ever meant ADD.
        """
        return self.step_alu_op("ADD", rd, rs1, rs2)

    def step_load_immediate(self, rd: int, immediate: int):
        """rd = immediate. Not its own code path -- routed through the
        ALU as rd = 0 + immediate, the same trick real RISC-V uses (its
        "li" pseudo-instruction is literally "addi rd, x0, imm"): there
        is no dedicated load-immediate circuit, just the adder fed a
        hardwired zero as one operand. Unlike step_alu_op, the second
        operand here is the immediate value itself, not a register
        index to read -- the actual distinction real hardware draws
        between R-type (register, register) and I-type (register,
        immediate) instructions.
        """
        result = self.alu(0, immediate & MASK, "ADD", width=WIDTH)
        self.registers.write(rd, result.value)
        self.pc.advance()
        self.clock.tick()
        return result

    def step_load_word(self, rd: int, base: int, offset: int) -> int:
        """rd = data_memory[registers[base] + offset]. The address is
        computed through this CPU's own ALU, same as a real load
        instruction's address-generation stage -- not Python's `+`.
        A negative offset is handled the same way register.py's
        Counter.increment() handles a negative step: masking it to
        32 bits first gives the correct two's-complement pattern for
        the ALU to add.
        """
        base_value = self.registers.read(base)
        address = self.alu(base_value, offset & MASK, "ADD", width=WIDTH).value
        value = self.data_memory.read_word(address)
        self.registers.write(rd, value)
        self.pc.advance()
        self.clock.tick()
        return value

    def step_store_word(self, rs: int, base: int, offset: int) -> None:
        """data_memory[registers[base] + offset] = registers[rs]."""
        base_value = self.registers.read(base)
        address = self.alu(base_value, offset & MASK, "ADD", width=WIDTH).value
        value = self.registers.read(rs)
        self.data_memory.write_word(address, value)
        self.pc.advance()
        self.clock.tick()

    def execute(self, instruction: tuple) -> None:
        """Dispatch one pseudo-instruction tuple to the matching step_*()
        method. Any opcode alu.py already knows (ADD, SUB, AND, OR, XOR,
        SLL, SRL, SRA, SLT, SLTU) is handled generically as a 3-operand
        register-register op -- there's no reason to hand-list each one
        separately when alu.OPERATIONS already is that list. LOAD,
        STORE, and LOAD_IMM have their own shapes (a memory address, or
        an immediate, isn't a register index) and are handled by name.

        An absolute address, as in the "STORE R4, [100]" form of Module
        5's example program, isn't its own addressing mode here -- it's
        base+offset with x0 as the base (("STORE", 4, 0, 100)), the same
        way real RISC-V assemblers synthesize "absolute" addressing from
        base+offset plus the one register that's hardwired to zero.
        """
        opcode, *operands = instruction
        if opcode in ALU_OPERATIONS:
            rd, rs1, rs2 = operands
            self.step_alu_op(opcode, rd, rs1, rs2)
        elif opcode == "LOAD_IMM":
            rd, immediate = operands
            self.step_load_immediate(rd, immediate)
        elif opcode == "LOAD":
            rd, base, offset = operands
            self.step_load_word(rd, base, offset)
        elif opcode == "STORE":
            rs, base, offset = operands
            self.step_store_word(rs, base, offset)
        else:
            raise ValueError(f"unknown pseudo-instruction opcode {opcode!r}")

    def run(self, program: list[tuple]) -> None:
        """Execute a whole pseudo-instruction program in order. No
        branches or jumps exist in this instruction set yet, so "in
        order" just means the list order -- pc still advances
        normally underneath, it's just not consulted to decide what
        runs next.
        """
        for instruction in program:
            self.execute(instruction)
