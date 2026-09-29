"""The datapath: actual movement of data through processor components,
broken into named stages rather than one function that knows
everything. Even though CPU.step() (in cpu.py) still calls every stage
sequentially within one cycle, giving each stage its own method now is
exactly the organization a pipelined design needs later -- a pipeline
is just these same five stages running concurrently on different
instructions at once, with latched values passed between them instead
of plain Python return values. Nothing about these method signatures
would need to change to get there.

Datapath owns the "physical" components a real datapath diagram would
show: instruction memory, data memory, the register file, and the ALU.
It does not own the program counter or the clock -- those are
sequencing/control-flow state, which belongs to cpu.py's CPU (the
coordinator), the only thing that actually knows how to get from one
instruction to the next.

decode() here is a thin wrapper around decoder.py's own decode() --
present as a stage method because the datapath is supposed to represent
every stage explicitly, not because the decoding logic is duplicated
here. It isn't; it's the same decoder.py the rest of this codebase
already uses and tests.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    # See core/adder.py's module docstring for why this is needed to run
    # this file directly (e.g. VS Code's Run button) rather than via
    # `python -m`. Three .parent calls: datapath.py -> cpu/ ->
    # cpu_simulator/ -> module-1/.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.core.alu import alu
from cpu_simulator.cpu.control import ControlSignals
from cpu_simulator.cpu.register_file import RegisterFile
from cpu_simulator.isa.decoder import decode as decode_word
from cpu_simulator.isa.instruction import Instruction
from cpu_simulator.memory.memory import Memory

WIDTH = 32
MASK = (1 << WIDTH) - 1


@dataclass
class ExecuteResult:
    operand_a: int
    operand_b: int
    alu_value: int
    branch_taken: bool


class Datapath:
    def __init__(self, instruction_memory: Memory, data_memory: Memory, registers: RegisterFile):
        self.instruction_memory = instruction_memory
        self.data_memory = data_memory
        self.registers = registers

    def fetch(self, address: int) -> int:
        """Read one raw instruction word out of instruction memory."""
        return self.instruction_memory.read_word(address)

    def decode(self, raw: int) -> Instruction:
        """decoder.py's decode(), exposed as a stage method -- see the
        module docstring for why this doesn't reimplement decoding.
        """
        return decode_word(raw)

    def execute(self, instruction: Instruction, control: ControlSignals, pc: int) -> ExecuteResult:
        """Run the ALU on whichever operands this instruction's control
        signals select, and (for branches) decide whether the
        comparison came out true.
        """
        operand_a = self._operand_a(instruction, pc)
        operand_b = self._operand_b(instruction, control)
        result = alu(operand_a, operand_b, control.alu_op, width=WIDTH)
        return ExecuteResult(operand_a=operand_a, operand_b=operand_b, alu_value=result.value,
                              branch_taken=self._branch_taken(control, result))

    def memory_access(self, instruction: Instruction, control: ControlSignals,
                       executed: ExecuteResult) -> int:
        """Read or write data memory at the address execute() computed.
        Returns the value writeback should use: the loaded word for a
        load, or execute()'s own ALU value passed straight through for
        everything else, so writeback never has to re-derive it.
        """
        if control.mem_read:
            return self.data_memory.read_word(executed.alu_value)
        if control.mem_write:
            self.data_memory.write_word(executed.alu_value, self.registers.read(instruction.rs2))
        return executed.alu_value

    def writeback(self, instruction: Instruction, control: ControlSignals,
                  memory_result: int, pc_plus_4: int) -> int | None:
        """Write the selected result to rd -- if this instruction
        writes a register at all. Returns the value actually written
        (or None if reg_write was False), mainly so a caller building a
        trace of the cycle doesn't have to re-derive which of
        memory_result/pc_plus_4 was used.
        """
        if not control.reg_write:
            return None
        value = pc_plus_4 if control.result_src == "PC_PLUS_4" else memory_result
        self.registers.write(instruction.rd, value)
        return value

    def _operand_a(self, instruction: Instruction, pc: int) -> int:
        if instruction.mnemonic == "AUIPC":
            # AUIPC's first ALU operand is pc, not a register -- the
            # mux control.py's own docstring flagged as out of its
            # scope. Resolved here, where the datapath actually has a
            # pc value on hand to mux in.
            return pc
        if instruction.rs1 is None:
            return 0  # LUI, JAL -- no first register operand at all
        return self.registers.read(instruction.rs1)

    def _operand_b(self, instruction: Instruction, control: ControlSignals) -> int:
        if control.alu_src == "REGISTER":
            return self.registers.read(instruction.rs2)
        return instruction.immediate & MASK  # "IMMEDIATE"

    def _branch_taken(self, control: ControlSignals, alu_result) -> bool:
        """BEQ/BNE read the SUB result's zero flag; BLT/BGE and
        BLTU/BGEU read the SLT/SLTU result's own 0/1 value directly --
        there is no separate "branch comparator" circuit here beyond
        the ALU operations control.py's ALU control already selected.
        """
        if control.branch is None:
            return False
        if control.branch == "BEQ":
            return bool(alu_result.zero)
        if control.branch == "BNE":
            return not alu_result.zero
        if control.branch in ("BLT", "BLTU"):
            return alu_result.value == 1
        if control.branch in ("BGE", "BGEU"):
            return alu_result.value == 0
        raise ValueError(f"unrecognized branch mnemonic {control.branch!r}")
