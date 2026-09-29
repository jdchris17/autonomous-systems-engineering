import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from branch_flush_test import run


class TestBranchFlushTest:
    def test_wrong_path_registers_stay_at_their_reset_value(self):
        cpu = run(trace=False)
        assert cpu.registers.read(3) == 0
        assert cpu.registers.read(4) == 0
        assert cpu.registers.read(5) == 42

    def test_wrong_path_instructions_never_retire(self):
        cpu = run(trace=False)
        assert not any(t.wb_stage.valid and t.wb_stage.rd in (3, 4) for t in cpu.trace_history)

    def test_exactly_one_flush(self):
        cpu = run(trace=False)
        assert cpu.performance.branch_flush_cycles == 1
