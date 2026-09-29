import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.cpu.pc import ProgramCounter


class TestProgramCounter:
    def test_starts_at_zero(self):
        assert ProgramCounter().read() == 0

    def test_advance_moves_by_instruction_width(self):
        pc = ProgramCounter()
        pc.advance()
        assert pc.read() == 4

    def test_repeated_advance(self):
        pc = ProgramCounter()
        for _ in range(3):
            pc.advance()
        assert pc.read() == 12

    def test_advance_wraps_at_32_bits(self):
        pc = ProgramCounter()
        pc.jump(0xFFFFFFFC)  # 4 bytes short of wrapping
        pc.advance()
        assert pc.read() == 0

    def test_jump_sets_an_exact_target(self):
        pc = ProgramCounter()
        pc.jump(0x1000)
        assert pc.read() == 0x1000

    def test_jump_then_advance(self):
        pc = ProgramCounter()
        pc.jump(0x2000)
        pc.advance()
        assert pc.read() == 0x2004

    def test_jump_masks_a_target_wider_than_32_bits(self):
        pc = ProgramCounter()
        pc.jump(0x1_00000000 | 0xCAFE)
        assert pc.read() == 0xCAFE

    def test_advance_uses_ripple_add(self):
        source = inspect.getsource(ProgramCounter.advance)
        assert "ripple_add" in source
