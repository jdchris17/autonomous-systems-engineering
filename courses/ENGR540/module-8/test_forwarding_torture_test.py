import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from forwarding_torture_test import run
from cpu_simulator.cpu.pipelined_cpu import PIPELINE_DEPTH


class TestForwardingTortureTest:
    def test_final_value_is_correct(self):
        cpu = run(trace=False)
        assert cpu.registers.read(1) == 5

    def test_zero_stalls_across_four_back_to_back_dependencies(self):
        cpu = run(trace=False)
        assert cpu.performance.data_hazard_stalls == 0

    def test_cycle_count_hits_the_fill_and_drain_minimum(self):
        cpu = run(trace=False)
        assert cpu.clock.cycle == 5 + (PIPELINE_DEPTH - 1)

    def test_every_dependency_forwards(self):
        cpu = run(trace=False)
        assert cpu.performance.forwarding_events == 4
