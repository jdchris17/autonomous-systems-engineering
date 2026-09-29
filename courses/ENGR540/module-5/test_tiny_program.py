import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module-1"))

from cpu_simulator.cpu.pseudo_cpu import PseudoOpCPU as CPU
from tiny_program import PROGRAM


class TestTinyProgram:
    def test_final_register_and_memory_state(self):
        cpu = CPU()
        cpu.run(PROGRAM)
        assert cpu.registers.read(1) == 10
        assert cpu.registers.read(2) == 7
        assert cpu.registers.read(3) == 17
        assert cpu.registers.read(4) == 10
        assert cpu.data_memory.read_word(100) == 10

    def test_pc_and_clock_advance_once_per_instruction(self):
        cpu = CPU()
        cpu.run(PROGRAM)
        assert cpu.pc.read() == len(PROGRAM) * 4
        assert cpu.clock.cycle == len(PROGRAM)

    def test_instruction_memory_is_untouched(self):
        # This program is executed directly as pseudo-instructions --
        # nothing was ever encoded into instruction_memory, since
        # encoding is Module 6's job, not Module 5's.
        cpu = CPU()
        cpu.run(PROGRAM)
        for address in range(0, 64, 4):
            assert cpu.instruction_memory.read_word(address) == 0
