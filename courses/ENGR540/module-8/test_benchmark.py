import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import SUM_TO_TEN_DYNAMIC_INSTRUCTIONS, SUM_TO_TEN_PROGRAM, run_benchmark


class TestSumToTenBenchmark:
    def test_both_models_agree_and_reach_the_expected_sum(self):
        assert run_benchmark("sum 1..10", SUM_TO_TEN_PROGRAM,
                              SUM_TO_TEN_DYNAMIC_INSTRUCTIONS) is True

    def test_pipelined_cpi_exceeds_single_cycles_exact_one(self):
        from benchmark import _load_single_cycle
        from cpu_simulator.cpu.analysis.performance import cpi
        from cpu_simulator.cpu.cpu import CPU
        from cpu_simulator.cpu.pipelined_cpu import PipelinedCPU, load_program

        single = CPU()
        _load_single_cycle(single, SUM_TO_TEN_PROGRAM)
        single.run(SUM_TO_TEN_DYNAMIC_INSTRUCTIONS)
        assert cpi(single.performance) == 1.0

        pipelined = PipelinedCPU()
        load_program(pipelined, SUM_TO_TEN_PROGRAM, pad_instructions=10)
        while pipelined.performance.instructions_retired < SUM_TO_TEN_DYNAMIC_INSTRUCTIONS:
            pipelined.step()
        assert cpi(pipelined.performance) > cpi(single.performance)
