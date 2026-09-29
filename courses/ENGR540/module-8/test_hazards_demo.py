import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hazards_demo import run_branch_flush, run_load_use_stall


class TestLoadUseStallDemo:
    def test_stall_then_forward_reaches_the_correct_values(self):
        cpu = run_load_use_stall(trace=False)
        assert cpu.registers.read(5) == 100
        assert cpu.registers.read(6) == 200
        assert any(t.stall for t in cpu.trace_history)


class TestBranchFlushDemo:
    def test_wrong_path_instructions_never_retire(self):
        cpu = run_branch_flush(trace=False)
        assert cpu.registers.read(3) == 42
        assert cpu.registers.read(10) == 0
        assert cpu.registers.read(11) == 0
        assert any(t.flush for t in cpu.trace_history)
