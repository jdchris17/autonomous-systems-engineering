import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module-1"))

from cpu_simulator.cpu.cpu import CPU
from cpu_simulator.isa.assembler import assemble

from multicycle import STAGES, MulticycleCPU, _next_stage
from run_program import PROGRAM


def _load(cpu, program: str):
    words = assemble(program)
    for i, word in enumerate(words):
        cpu.instruction_memory.write_word(i * 4, word)
    return len(words)


class TestStageTransitions:
    def test_cycles_through_all_five_stages_in_order(self):
        stage = "FETCH"
        seen = [stage]
        for _ in range(5):
            stage = _next_stage(stage)
            seen.append(stage)
        assert seen == ["FETCH", "DECODE", "EXECUTE", "MEMORY", "WRITEBACK", "FETCH"]

    def test_stages_constant_matches_the_transition_order(self):
        assert STAGES == ("FETCH", "DECODE", "EXECUTE", "MEMORY", "WRITEBACK")


class TestTickAdvancesOneStagePerCall:
    def test_starts_in_fetch(self):
        cpu = MulticycleCPU()
        assert cpu.stage == "FETCH"

    def test_five_ticks_completes_one_instruction_and_returns_to_fetch(self):
        cpu = MulticycleCPU()
        _load(cpu, "addi x1, x0, 5")
        stages_seen = []
        for _ in range(5):
            stages_seen.append(cpu.stage)
            cpu.tick()
        assert stages_seen == ["FETCH", "DECODE", "EXECUTE", "MEMORY", "WRITEBACK"]
        assert cpu.stage == "FETCH"

    def test_clock_advances_once_per_stage_not_once_per_instruction(self):
        cpu = MulticycleCPU()
        _load(cpu, "addi x1, x0, 5")
        cpu.run_instruction()
        assert cpu.clock.cycle == 5

    def test_register_is_not_written_until_writeback_stage(self):
        cpu = MulticycleCPU()
        _load(cpu, "addi x1, x0, 5")
        for _ in range(4):  # FETCH, DECODE, EXECUTE, MEMORY
            cpu.tick()
        assert cpu.registers.read(1) == 0  # not written yet
        cpu.tick()  # WRITEBACK
        assert cpu.registers.read(1) == 5


class TestRunInstructionAndRun:
    def test_run_instruction_runs_exactly_five_ticks(self):
        cpu = MulticycleCPU()
        _load(cpu, "addi x1, x0, 5")
        cpu.run_instruction()
        assert cpu.clock.cycle == 5
        assert cpu.stage == "FETCH"

    def test_run_n_instructions(self):
        cpu = MulticycleCPU()
        n = _load(cpu, "addi x1, x0, 5\naddi x2, x0, 7\nadd x3, x1, x2")
        cpu.run(n)
        assert cpu.registers.read(3) == 12
        assert cpu.clock.cycle == n * 5


class TestBranchesAndMemoryUnderMulticycle:
    def test_backward_branch(self):
        cpu = MulticycleCPU()
        _load(cpu, "addi x5, x0, 3\naddi x5, x5, -1\nbne x5, x0, -4")
        # 1 setup instruction, then the 2-instruction loop body runs 3
        # times (x5: 3->2->1->0, exiting when bne is finally not taken).
        cpu.run(1 + 3 * 2)
        assert cpu.registers.read(5) == 0
        assert cpu.pc.read() == 0xC

    def test_store_and_load(self):
        cpu = MulticycleCPU()
        n = _load(cpu, "addi x1, x0, 42\nsw x1, 0(x0)\nlw x2, 0(x0)")
        cpu.run(n)
        assert cpu.registers.read(2) == 42


class TestTraceIntegration:
    def test_last_trace_is_populated_after_writeback(self):
        cpu = MulticycleCPU()
        _load(cpu, "addi x1, x0, 5")
        cpu.run_instruction()
        assert cpu.last_trace is not None
        assert cpu.last_trace.instruction.mnemonic == "ADDI"
        assert cpu.last_trace.writeback_value == 5

    def test_trace_history_has_one_entry_per_completed_instruction(self):
        cpu = MulticycleCPU()
        n = _load(cpu, "addi x1, x0, 1\naddi x1, x1, 1\naddi x1, x1, 1")
        cpu.run(n)
        assert len(cpu.trace_history) == n

    def test_cycle_number_reflects_ticks_not_instructions(self):
        cpu = MulticycleCPU()
        n = _load(cpu, "addi x1, x0, 1\naddi x1, x1, 1")
        cpu.run(n)
        assert [t.cycle for t in cpu.trace_history] == [5, 10]


class TestAgreesWithSingleCycleCPU:
    """The strongest correctness check: running the exact same program
    through both execution models should reach the exact same
    architectural end state, even though the clock behaves completely
    differently underneath (1 tick/instruction vs 5 ticks/instruction).
    """

    def test_section_34_program_reaches_the_same_result(self):
        single = CPU()
        _load(single, PROGRAM)
        from run_program import NUM_STEPS
        single.run(NUM_STEPS)

        multi = MulticycleCPU()
        n = _load(multi, PROGRAM)
        # Same dynamic instruction count as the single-cycle run.
        multi.run(NUM_STEPS)

        assert multi.data_memory.read_word(0) == single.data_memory.read_word(0) == 15
        assert multi.registers.read(1) == single.registers.read(1)
        assert multi.registers.read(2) == single.registers.read(2)
        assert multi.pc.read() == single.pc.read()

    def test_clock_cycles_differ_by_exactly_5x(self):
        single = CPU()
        _load(single, PROGRAM)
        from run_program import NUM_STEPS
        single.run(NUM_STEPS)

        multi = MulticycleCPU()
        _load(multi, PROGRAM)
        multi.run(NUM_STEPS)

        assert multi.clock.cycle == single.clock.cycle * 5
