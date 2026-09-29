import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_use_test import run


class TestLoadUseTest:
    def test_final_values_are_correct(self):
        cpu = run(trace=False)
        assert cpu.registers.read(5) == 100
        assert cpu.registers.read(6) == 110
        assert cpu.registers.read(7) == 105

    def test_exactly_one_stall_the_load_use_pair_not_the_alu_pair(self):
        cpu = run(trace=False)
        assert cpu.performance.data_hazard_stalls == 1
        assert cpu.performance.load_use_stalls == 1

    def test_the_add_to_sub_dependency_resolves_through_forwarding(self):
        cpu = run(trace=False)
        assert cpu.performance.forwarding_events >= 1
