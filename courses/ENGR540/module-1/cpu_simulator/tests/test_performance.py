import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.cpu.analysis.performance import PerformanceCounters, cpi, cpi_breakdown, ipc


class TestDefaults:
    def test_every_counter_starts_at_zero(self):
        counters = PerformanceCounters()
        assert counters.instructions_retired == 0
        assert counters.total_cycles == 0
        assert counters.alu_instructions == 0
        assert counters.load_instructions == 0
        assert counters.store_instructions == 0
        assert counters.branch_instructions == 0
        assert counters.data_hazard_stalls == 0
        assert counters.load_use_stalls == 0
        assert counters.branch_flush_cycles == 0
        assert counters.memory_stall_cycles == 0
        assert counters.cache_accesses == 0
        assert counters.cache_hits == 0
        assert counters.cache_misses == 0
        assert counters.dirty_writebacks == 0
        assert counters.branches_taken == 0
        assert counters.forwarding_events == 0


class TestCpiAndIpc:
    def test_one_instruction_per_cycle_is_cpi_one(self):
        counters = PerformanceCounters(instructions_retired=10, total_cycles=10)
        assert cpi(counters) == 1.0
        assert ipc(counters) == 1.0

    def test_cpi_and_ipc_are_reciprocals(self):
        counters = PerformanceCounters(instructions_retired=33, total_cycles=55)
        assert cpi(counters) == 55 / 33
        assert ipc(counters) == 33 / 55
        assert cpi(counters) * ipc(counters) == 1.0

    def test_ipc_never_exceeds_one_for_a_single_issue_machine(self):
        # instructions_retired can never exceed total_cycles -- at most
        # one instruction retires per tick, single-cycle or pipelined.
        counters = PerformanceCounters(instructions_retired=33, total_cycles=55)
        assert ipc(counters) <= 1.0


class TestCpiBreakdown:
    """Section 36's own worked example, reproduced exactly."""

    def test_matches_the_spec_worked_example(self):
        counters = PerformanceCounters(
            instructions_retired=100,
            data_hazard_stalls=7,
            branch_flush_cycles=14,
            memory_stall_cycles=31,
        )
        breakdown = cpi_breakdown(counters)
        assert breakdown.base_cpi == 1.00
        assert breakdown.data_hazard_contribution == 0.07
        assert breakdown.branch_contribution == 0.14
        assert breakdown.memory_contribution == 0.31
        assert round(breakdown.actual_cpi, 2) == 1.52

    def test_no_stalls_at_all_leaves_actual_cpi_at_base(self):
        counters = PerformanceCounters(instructions_retired=10)
        breakdown = cpi_breakdown(counters)
        assert breakdown.data_hazard_contribution == 0.0
        assert breakdown.branch_contribution == 0.0
        assert breakdown.memory_contribution == 0.0
        assert breakdown.actual_cpi == 1.0

    def test_actual_cpi_is_the_sum_of_base_and_every_contribution(self):
        counters = PerformanceCounters(
            instructions_retired=50, data_hazard_stalls=5,
            branch_flush_cycles=10, memory_stall_cycles=20,
        )
        breakdown = cpi_breakdown(counters)
        assert breakdown.actual_cpi == (
            breakdown.base_cpi + breakdown.data_hazard_contribution
            + breakdown.branch_contribution + breakdown.memory_contribution
        )
