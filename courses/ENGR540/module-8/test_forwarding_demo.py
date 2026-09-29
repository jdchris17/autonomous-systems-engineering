import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from forwarding_demo import run


class TestPriorityMattersDemo:
    def test_ex_mem_beats_mem_wb(self):
        cpu = run(trace=False)
        assert cpu.registers.read(5) == 31
        assert cpu.registers.read(6) == 30
