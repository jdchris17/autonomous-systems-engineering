"""An optional staged execution mode, per section 36 -- built directly
on datapath.py's already-separate stage methods, which is the entire
payoff of section 28 having asked for fetch/decode/execute/
memory_access/writeback as five real methods instead of one function:
nothing needed to change in datapath.py to drive it one stage at a
time instead of all five at once.

Where cpu.py's CPU (section 35's single-cycle model) calls all five
stages inside one step() and ticks the clock once per *instruction*,
MulticycleCPU calls exactly one stage per tick() and ticks the clock
once per *stage* -- five ticks per instruction, cycling through an
explicit

    FETCH -> DECODE -> EXECUTE -> MEMORY -> WRITEBACK -> FETCH -> ...

state machine. "Then your control unit is behaving more like an FSM" --
the same shape module-1/cpu_simulator/cpu/fsm.py already established
(current state stored explicitly, a pure function computing the next
state, the state only actually advancing on a tick) -- deliberately not
that exact FSM class, though: its states (IDLE/LOAD/PROCESS/DONE) and
inputs (start/loaded/complete/reset) are a different, unrelated demo
machine, and forcing this 5-stage cycle through that specific class
would mean bending its API to fit rather than reusing anything real.
What's reused is the *pattern* fsm.py demonstrated, applied to a real
control unit instead of a toy example -- section 36's own description
of what this ties together.

Values that would be a real multicycle CPU's internal latches (the
instruction register, the ALU output register, and so on -- state that
survives *between* stages of the *same* instruction, the same role a
pipeline register would play if these stages ran concurrently instead
of in sequence) are just plain attributes here, set by one stage and
read by a later one. This is deliberately not modeling variable-length
multicycle timing (a real design skips the MEMORY stage entirely for
an R-type, finishing in 4 cycles instead of 5) -- every instruction
here takes exactly five ticks, even when a stage has nothing to do.
That's a real simplification, not an oversight: section 36 asks for
the staged *structure*, not cycle-accurate performance modeling.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module-1"))

from cpu_simulator.core.adder import ripple_add
from cpu_simulator.cpu.clock import Clock
from cpu_simulator.cpu.control import generate_control
from cpu_simulator.cpu.datapath import Datapath
from cpu_simulator.cpu.pc import ProgramCounter
from cpu_simulator.cpu.register_file import RegisterFile
from cpu_simulator.cpu.trace import StepTrace
from cpu_simulator.memory.memory import Memory

WIDTH = 32
MASK = (1 << WIDTH) - 1
INSTRUCTION_WIDTH = 4

STAGES = ("FETCH", "DECODE", "EXECUTE", "MEMORY", "WRITEBACK")


def _next_stage(stage: str) -> str:
    """The whole transition function -- fixed cyclic order, no inputs
    to branch on. Kept as its own pure function anyway (rather than
    inlined into tick()), the same reason fsm.py's transition() is
    separate from FSM: it's independently testable as the truth table
    it is, with no CPU state involved at all.
    """
    return STAGES[(STAGES.index(stage) + 1) % len(STAGES)]


class MulticycleCPU:
    def __init__(self, instruction_memory_bytes: int = 1024, data_memory_bytes: int = 1024):
        self.pc = ProgramCounter()
        self.clock = Clock()
        self.registers = RegisterFile()
        self.instruction_memory = Memory(size_bytes=instruction_memory_bytes)
        self.data_memory = Memory(size_bytes=data_memory_bytes)
        self.datapath = Datapath(self.instruction_memory, self.data_memory, self.registers)

        self.stage = "FETCH"
        self.last_trace: StepTrace | None = None
        self.trace_history: list[StepTrace] = []

        # Inter-stage latches for the instruction currently in flight --
        # written by one stage, read by a later one, the same role a
        # pipeline register plays when stages run concurrently instead
        # of in sequence.
        self._pc_before = None
        self._fetched = None
        self._decoded = None
        self._rs1_value = None
        self._rs2_value = None
        self._control = None
        self._executed = None
        self._memory_result = None
        self._mem_address = None
        self._mem_read_value = None
        self._mem_write_value = None
        self._pc_plus_4 = None

    def tick(self) -> None:
        """Perform the current stage's work, advance the clock, and
        move to the next stage -- one clock.tick() per *stage*, not per
        instruction.
        """
        if self.stage == "FETCH":
            self._fetch()
        elif self.stage == "DECODE":
            self._decode()
        elif self.stage == "EXECUTE":
            self._execute()
        elif self.stage == "MEMORY":
            self._memory()
        elif self.stage == "WRITEBACK":
            self._writeback()

        self.clock.tick()
        self.stage = _next_stage(self.stage)

    def run_instruction(self) -> None:
        """Advance through all five stages: one full instruction, five
        tick() calls, five clock cycles.
        """
        for _ in range(len(STAGES)):
            self.tick()

    def run(self, num_instructions: int) -> None:
        for _ in range(num_instructions):
            self.run_instruction()

    def _fetch(self) -> None:
        self._pc_before = self.pc.read()
        self._fetched = self.datapath.fetch(self._pc_before)

    def _decode(self) -> None:
        self._decoded = self.datapath.decode(self._fetched)
        self._control = generate_control(self._decoded)
        self._rs1_value = (self.registers.read(self._decoded.rs1)
                            if self._decoded.rs1 is not None else None)
        self._rs2_value = (self.registers.read(self._decoded.rs2)
                            if self._decoded.rs2 is not None else None)

    def _execute(self) -> None:
        self._executed = self.datapath.execute(self._decoded, self._control, self._pc_before)

    def _memory(self) -> None:
        control = self._control
        self._memory_result = self.datapath.memory_access(self._decoded, control, self._executed)
        self._mem_write_value = self._rs2_value if control.mem_write else None
        self._mem_address = (self._executed.alu_value
                              if (control.mem_read or control.mem_write) else None)
        self._mem_read_value = self._memory_result if control.mem_read else None

    def _writeback(self) -> None:
        decoded, control, executed = self._decoded, self._control, self._executed
        self._pc_plus_4, _carry = ripple_add(self._pc_before, INSTRUCTION_WIDTH, width=WIDTH)
        writeback_value = self.datapath.writeback(
            decoded, control, self._memory_result, self._pc_plus_4)
        self._update_pc(control, executed)

        self.last_trace = StepTrace(
            cycle=self.clock.cycle + 1,  # +1: this tick()'s own clock.tick() hasn't run yet
            pc=self._pc_before,
            raw=self._fetched,
            instruction=decoded,
            rs1_value=self._rs1_value,
            rs2_value=self._rs2_value,
            control=control,
            operand_a=executed.operand_a,
            operand_b=executed.operand_b,
            alu_result=executed.alu_value,
            branch_taken=executed.branch_taken,
            mem_address=self._mem_address,
            mem_read_value=self._mem_read_value,
            mem_write_value=self._mem_write_value,
            writeback_value=writeback_value,
            next_pc=self.pc.read(),
        )
        self.trace_history.append(self.last_trace)

    def _update_pc(self, control, executed) -> None:
        # Same logic as cpu.py's CPU._update_pc -- see that file for
        # why JAL and JALR compute their targets differently.
        if control.jump:
            if self._decoded.mnemonic == "JAL":
                target, _carry = ripple_add(self._pc_before, self._decoded.immediate & MASK,
                                             width=WIDTH)
            else:  # JALR
                target = executed.alu_value & ~1 & MASK
            self.pc.jump(target)
        elif control.branch is not None and executed.branch_taken:
            target, _carry = ripple_add(self._pc_before, self._decoded.immediate & MASK,
                                         width=WIDTH)
            self.pc.jump(target)
        else:
            self.pc.jump(self._pc_plus_4)


def main():
    from cpu_simulator.cpu.trace import format_verbose
    from cpu_simulator.isa.assembler import assemble

    cpu = MulticycleCPU()
    cpu.instruction_memory.write_word(0, assemble("addi x1, x0, 5")[0])

    print("=" * 60)
    print("MULTICYCLE MODE: one instruction, five ticks")
    print("=" * 60)
    for _ in range(5):
        print(f"stage = {cpu.stage:<10} cycle = {cpu.clock.cycle}")
        cpu.tick()
    print(f"stage = {cpu.stage:<10} cycle = {cpu.clock.cycle}  (back to FETCH)")
    print()
    print(format_verbose(cpu.last_trace))
    print("=" * 60)


if __name__ == "__main__":
    main()
