import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.cpu.cpu import CPU
from cpu_simulator.cpu.pipelined_cpu import PIPELINE_DEPTH, PipelinedCPU, load_program
from cpu_simulator.isa.assembler import assemble


def _load_single_cycle(cpu: CPU, program: str) -> int:
    words = assemble(program)
    for i, word in enumerate(words):
        cpu.instruction_memory.write_word(i * 4, word)
    return len(words)


class TestPipelineFillAndDrain:
    """With no data hazards, one instruction should complete writeback
    per cycle once the pipeline is full -- filling takes PIPELINE_DEPTH
    - 1 cycles (the first instruction reaches WB on cycle 5), and the
    last instruction of an N-instruction program needs N + 4 cycles
    total to finish its own writeback.
    """

    PROGRAM = """
addi x1, x0, 10
addi x2, x0, 20
addi x3, x0, 30
addi x4, x0, 40
    """

    def test_nothing_has_written_back_before_cycle_five(self):
        cpu = PipelinedCPU()
        n = load_program(cpu, self.PROGRAM)
        cpu.run(4)  # cycles 1-4: pipeline still filling
        assert cpu.registers.read(1) == 0

    def test_first_instruction_writes_back_on_its_fifth_cycle(self):
        cpu = PipelinedCPU()
        load_program(cpu, self.PROGRAM)
        cpu.run(5)
        assert cpu.registers.read(1) == 10

    def test_all_four_instructions_complete_after_n_plus_four_cycles(self):
        cpu = PipelinedCPU()
        n = load_program(cpu, self.PROGRAM)
        cpu.run(n + (PIPELINE_DEPTH - 1))
        assert cpu.registers.read(1) == 10
        assert cpu.registers.read(2) == 20
        assert cpu.registers.read(3) == 30
        assert cpu.registers.read(4) == 40

    def test_pc_and_clock_advance_once_per_cycle_not_per_instruction(self):
        cpu = PipelinedCPU()
        load_program(cpu, self.PROGRAM)
        cpu.run(6)
        assert cpu.pc.read() == 6 * 4
        assert cpu.clock.cycle == 6


class TestBubblesInitially:
    def test_all_four_registers_start_as_bubbles(self):
        cpu = PipelinedCPU()
        assert cpu.if_id.valid is False
        assert cpu.id_ex.valid is False
        assert cpu.ex_mem.valid is False
        assert cpu.mem_wb.valid is False

    def test_first_cycle_populates_only_if_id(self):
        cpu = PipelinedCPU()
        load_program(cpu, "addi x1, x0, 5")
        cpu.step()
        assert cpu.if_id.valid is True
        assert cpu.id_ex.valid is False


class TestThePriorityMattersWorkedExample:
    """The exact spec example: the most recently produced value (in
    EX_MEM) must win over an older one still sitting in MEM_WB.
    """

    PROGRAM = """
add  x5, x1, x2
addi x5, x5, 1
sub  x6, x5, x3
    """

    def test_ex_mem_forwarding_beats_mem_wb_forwarding(self):
        cpu = PipelinedCPU()
        cpu.registers.write(1, 10)
        cpu.registers.write(2, 20)
        cpu.registers.write(3, 1)
        n = load_program(cpu, self.PROGRAM)
        cpu.run(n + (PIPELINE_DEPTH - 1))
        assert cpu.registers.read(5) == 31
        assert cpu.registers.read(6) == 30


class TestLoadForwarding:
    """A load's value only exists after MEMORY runs, so a consumer far
    enough behind it (not immediately adjacent) should still get the
    correct value via MEM_WB forwarding.
    """

    PROGRAM = """
addi x1, x0, 100
sw   x1, 0(x0)
lw   x2, 0(x0)
addi x3, x0, 1
add  x4, x2, x3
    """

    def test_load_result_forwards_correctly_to_a_non_adjacent_consumer(self):
        cpu = PipelinedCPU()
        n = load_program(cpu, self.PROGRAM)
        cpu.run(n + (PIPELINE_DEPTH - 1))
        assert cpu.registers.read(2) == 100
        assert cpu.registers.read(4) == 101


class TestCrossValidationAgainstSingleCycleCPU:
    """FinalState(CPU) == FinalState(PipelinedCPU) for any straight-line
    program -- the correctness test this whole file exists to pass.
    """

    def _run_both(self, program: str, extra_drain_cycles: int = PIPELINE_DEPTH - 1):
        single = CPU()
        n_single = _load_single_cycle(single, program)
        single.run(n_single)

        pipelined = PipelinedCPU()
        n_pipelined = load_program(pipelined, program)
        pipelined.run(n_pipelined + extra_drain_cycles)

        return single, pipelined

    def test_module_6_program(self):
        program = """
addi x1, x0, 10
addi x2, x0, 7
add  x3, x1, x2
sub  x4, x3, x2
sw   x4, 0(x0)
        """
        single, pipelined = self._run_both(program)
        for reg in (1, 2, 3, 4):
            assert single.registers.read(reg) == pipelined.registers.read(reg)
        assert single.data_memory.read_word(0) == pipelined.data_memory.read_word(0)

    def test_load_forwarding_heavy_program(self):
        program = """
addi x1, x0, 100
sw   x1, 0(x0)
lw   x2, 0(x0)
addi x3, x0, 1
add  x4, x2, x3
addi x4, x4, 5
sub  x5, x4, x2
        """
        single, pipelined = self._run_both(program)
        for reg in (1, 2, 3, 4, 5):
            assert single.registers.read(reg) == pipelined.registers.read(reg)

    def test_the_priority_matters_program(self):
        program = """
add  x5, x1, x2
addi x5, x5, 1
sub  x6, x5, x3
        """
        single = CPU()
        single.registers.write(1, 10)
        single.registers.write(2, 20)
        single.registers.write(3, 1)
        n_single = _load_single_cycle(single, program)
        single.run(n_single)

        pipelined = PipelinedCPU()
        pipelined.registers.write(1, 10)
        pipelined.registers.write(2, 20)
        pipelined.registers.write(3, 1)
        n_pipelined = load_program(pipelined, program)
        pipelined.run(n_pipelined + (PIPELINE_DEPTH - 1))

        for reg in (5, 6):
            assert single.registers.read(reg) == pipelined.registers.read(reg)


class TestLoadProgramPadding:
    def test_pads_with_real_nop_words_not_zeros(self):
        from cpu_simulator.cpu.pipelined_cpu import NOP_WORD

        cpu = PipelinedCPU()
        n = load_program(cpu, "addi x1, x0, 1", pad_instructions=3)
        assert n == 1
        for i in range(3):
            pad_address = (n + i) * 4
            assert cpu.instruction_memory.read_word(pad_address) == NOP_WORD

    def test_default_padding_lets_a_single_instruction_program_drain_cleanly(self):
        cpu = PipelinedCPU()
        n = load_program(cpu, "addi x1, x0, 42")
        # Should run without raising, all the way through drain.
        cpu.run(n + (PIPELINE_DEPTH - 1))
        assert cpu.registers.read(1) == 42


class TestLoadUseStall:
    """Forwarding alone can't bridge a load-use hazard -- the value
    doesn't exist until MEMORY runs. hazards.py's stall buys the one
    cycle forwarding then needs to catch it via MEM_WB.
    """

    PROGRAM = """
addi x1, x0, 100
sw   x1, 0(x0)
lw   x5, 0(x0)
add  x6, x5, x1
    """

    def test_dependent_instruction_still_gets_the_correct_value(self):
        cpu = PipelinedCPU()
        n = load_program(cpu, self.PROGRAM)
        cpu.run(n + (PIPELINE_DEPTH - 1) + 1)  # +1: the stall costs a cycle
        assert cpu.registers.read(5) == 100
        assert cpu.registers.read(6) == 200

    def test_a_bubble_actually_enters_id_ex_during_the_stall(self):
        cpu = PipelinedCPU()
        load_program(cpu, self.PROGRAM)
        cpu.run(5)  # the cycle "add x6,x5,x1" would otherwise enter ID_EX
        assert cpu.id_ex.valid is False  # bubble, not "add"

    def test_the_stalled_instruction_is_still_in_if_id_not_lost(self):
        cpu = PipelinedCPU()
        load_program(cpu, self.PROGRAM)
        cpu.run(5)
        assert cpu.if_id.valid is True
        assert cpu.last_trace.stall is True
        assert cpu.last_trace.stall_reason == "load-use dependency on x5"

    def test_pc_does_not_advance_during_the_stall(self):
        cpu = PipelinedCPU()
        load_program(cpu, self.PROGRAM)
        cpu.run(4)
        pc_before_stall = cpu.pc.read()
        cpu.step()  # the stall cycle
        assert cpu.pc.read() == pc_before_stall

    def test_no_stall_needed_once_a_non_adjacent_gap_separates_load_and_use(self):
        # The same load-forwarding program from the cross-validation
        # suite -- one instruction of daylight between the load and its
        # consumer means no stall is needed at all, only forwarding.
        program = """
addi x1, x0, 100
sw   x1, 0(x0)
lw   x2, 0(x0)
addi x3, x0, 1
add  x4, x2, x3
        """
        cpu = PipelinedCPU()
        n = load_program(cpu, program)
        cpu.run(n + (PIPELINE_DEPTH - 1))
        assert not any(t.stall for t in cpu.trace_history)
        assert cpu.registers.read(4) == 101


class TestBranchFlush:
    """A taken branch, resolved in EX under static not-taken prediction,
    must flush the two wrong-path instructions fetch/decode already
    grabbed and redirect pc to the branch's target.
    """

    PROGRAM = """
addi x1, x0, 5
addi x2, x0, 5
beq  x1, x2, skip
addi x10, x0, 999
addi x11, x0, 999
skip:
addi x3, x0, 42
    """

    def test_wrong_path_instructions_never_retire(self):
        cpu = PipelinedCPU()
        n = load_program(cpu, self.PROGRAM)
        cpu.run(n + (PIPELINE_DEPTH - 1))
        assert cpu.registers.read(10) == 0
        assert cpu.registers.read(11) == 0
        assert cpu.registers.read(3) == 42

    def test_pc_redirects_to_the_branch_target(self):
        cpu = PipelinedCPU()
        load_program(cpu, self.PROGRAM)
        cpu.run(5)  # the cycle beq resolves in EX
        assert cpu.last_trace.flush is True
        assert cpu.last_trace.branch_target == 0x14  # "skip:"

    def test_not_taken_branch_flushes_nothing(self):
        cpu = PipelinedCPU()
        load_program(cpu, "beq x1, x2, 16\naddi x3, x0, 1", pad_instructions=6)
        cpu.registers.write(1, 1)  # x1 != x2(=0) -> not taken
        cpu.run(6)
        assert not any(t.flush for t in cpu.trace_history)
        assert cpu.registers.read(3) == 1  # fell through and ran normally

    def test_backward_branch_loop_cross_validates_against_single_cycle_cpu(self):
        # The first branching (non-straight-line) program this pipeline
        # has to get right: a real backward-branch loop, summing 5+4+3+2+1.
        # Needs generous padding -- every taken iteration of the branch
        # costs two flushed wrong-path fetches beyond what a straight-line
        # instruction count would predict.
        program = """
addi x1, x0, 5
addi x2, x0, 0
loop:
    add  x2, x2, x1
    addi x1, x1, -1
    bne  x1, x0, loop
sw   x2, 0(x0)
        """
        single = CPU()
        _load_single_cycle(single, program)
        # Dynamic step count, not the assembled word count: 2 setup
        # instructions, then the 3-instruction loop body runs 5 times.
        single.run(2 + 5 * 3 + 1)

        pipelined = PipelinedCPU()
        load_program(pipelined, program, pad_instructions=40)
        pipelined.run(40)

        assert pipelined.registers.read(1) == single.registers.read(1)
        assert pipelined.registers.read(2) == single.registers.read(2)
        assert pipelined.data_memory.read_word(0) == single.data_memory.read_word(0)
        assert single.registers.read(2) == 15  # 5+4+3+2+1
        # 4 of the loop's 5 iterations take the backward branch; only the
        # last falls through -- so exactly 4 flushes are expected.
        assert sum(1 for t in pipelined.trace_history if t.flush) == 4


class TestPerformanceCounters:
    def test_straight_line_program_has_no_stalls_or_flushes(self):
        cpu = PipelinedCPU()
        n = load_program(cpu, "addi x1, x0, 1\naddi x2, x0, 2\naddi x3, x0, 3")
        cpu.run(n + (PIPELINE_DEPTH - 1))
        assert cpu.performance.instructions_retired == n
        assert cpu.performance.data_hazard_stalls == 0
        assert cpu.performance.load_use_stalls == 0
        assert cpu.performance.branch_flush_cycles == 0

    def test_total_cycles_always_matches_the_clock(self):
        cpu = PipelinedCPU()
        n = load_program(cpu, "addi x1, x0, 1\naddi x2, x0, 2\naddi x3, x0, 3")
        cpu.run(n + (PIPELINE_DEPTH - 1))
        assert cpu.performance.total_cycles == cpu.clock.cycle

    def test_a_load_use_stall_is_counted(self):
        program = """
addi x1, x0, 100
sw   x1, 0(x0)
lw   x5, 0(x0)
add  x6, x5, x1
        """
        cpu = PipelinedCPU()
        n = load_program(cpu, program)
        cpu.run(n + (PIPELINE_DEPTH - 1) + 1)
        assert cpu.performance.data_hazard_stalls == 1
        assert cpu.performance.load_use_stalls == 1

    def test_forwarding_events_are_counted_per_operand(self):
        # The exact "priority matters" example: both x5 reads (ADDI's
        # and SUB's) forward from EX_MEM, x3 doesn't need to forward at
        # all -- 2 total forwarding events.
        program = """
add  x5, x1, x2
addi x5, x5, 1
sub  x6, x5, x3
        """
        cpu = PipelinedCPU()
        cpu.registers.write(1, 10)
        cpu.registers.write(2, 20)
        cpu.registers.write(3, 1)
        n = load_program(cpu, program)
        cpu.run(n + (PIPELINE_DEPTH - 1))
        assert cpu.performance.forwarding_events == 2

    def test_branch_and_flush_counters_on_a_backward_branch_loop(self):
        program = """
addi x1, x0, 5
addi x2, x0, 0
loop:
    add  x2, x2, x1
    addi x1, x1, -1
    bne  x1, x0, loop
sw   x2, 0(x0)
        """
        cpu = PipelinedCPU()
        load_program(cpu, program, pad_instructions=40)
        cpu.run(40)
        assert cpu.performance.branch_instructions == 5
        assert cpu.performance.branches_taken == 4
        assert cpu.performance.branch_flush_cycles == 4  # 1:1 with branches_taken today

    def test_sum_to_ten_benchmark_cpi_exceeds_one_but_ipc_stays_at_most_one(self):
        from cpu_simulator.cpu.analysis.performance import cpi, ipc

        program = """
addi x1, x0, 10
addi x2, x0, 0
loop:
    add  x2, x2, x1
    addi x1, x1, -1
    bne  x1, x0, loop
sw   x2, 0(x0)
        """
        dynamic_instructions = 2 + 10 * 3 + 1
        cpu = PipelinedCPU()
        load_program(cpu, program, pad_instructions=10)
        while cpu.performance.instructions_retired < dynamic_instructions:
            cpu.step()

        assert cpu.data_memory.read_word(0) == 55
        assert cpu.performance.instructions_retired == dynamic_instructions
        assert cpi(cpu.performance) > 1.0  # fill/drain + flush overhead
        assert ipc(cpu.performance) <= 1.0


class TestCacheBackedMemoryStage:
    """Section 58: the MEM stage now goes through a MemoryHierarchy
    when one is configured, and a miss stalls the whole front end for
    however many extra cycles the access needs.
    """

    def test_no_cache_by_default_preserves_original_one_cycle_timing(self):
        cpu = PipelinedCPU()
        assert cpu.data_cache is None
        assert cpu.data_hierarchy is None
        n = load_program(cpu, "sw x0, 0(x0)\nlw x1, 0(x0)")
        cpu.run(n + (PIPELINE_DEPTH - 1))
        assert cpu.performance.memory_stall_cycles == 0
        assert cpu.clock.cycle == n + (PIPELINE_DEPTH - 1)

    def test_a_cache_miss_stalls_for_the_configured_latency(self):
        cpu = PipelinedCPU(data_cache_size_bytes=64, data_cache_line_size=16,
                            hit_latency=1, miss_latency=20)
        n = load_program(cpu, "addi x1, x0, 1\nsw x1, 0(x0)", pad_instructions=25)
        while cpu.performance.instructions_retired < n:
            cpu.step()
        # The store misses (nothing cached yet) -- 19 extra stall cycles
        # beyond the one it would have cost with no cache at all.
        assert cpu.performance.memory_stall_cycles == 19

    def test_a_cache_hit_costs_no_extra_stall(self):
        cpu = PipelinedCPU(data_cache_size_bytes=64, data_cache_line_size=16,
                            hit_latency=1, miss_latency=20)
        program = "addi x1, x0, 1\nsw x1, 0(x0)\nsw x1, 0(x0)"  # second sw hits
        n = load_program(cpu, program, pad_instructions=25)
        while cpu.performance.instructions_retired < n:
            cpu.step()
        assert cpu.performance.memory_stall_cycles == 19  # only the first store misses

    def test_the_front_end_is_frozen_for_the_whole_stall(self):
        cpu = PipelinedCPU(data_cache_size_bytes=64, data_cache_line_size=16,
                            hit_latency=1, miss_latency=5)
        load_program(cpu, "addi x1, x0, 1\nsw x1, 0(x0)\naddi x2, x0, 2", pad_instructions=10)
        cpu.run(6)  # somewhere in the middle of the store's stall
        assert cpu.last_trace.stall is True
        assert "data memory access in progress" in cpu.last_trace.stall_reason
        assert cpu.last_trace.stall_detail == "MEM stage busy"

    def test_architectural_state_matches_the_single_cycle_model_with_a_cache(self):
        # The regression case: two back-to-back stores/loads whose
        # addresses deliberately conflict in a small direct-mapped
        # cache, with an ordinary RAW dependency (addi x2 -> sw x2)
        # immediately followed by an instruction (sw x1 -> stalls) that
        # freezes the consumer of that dependency in ID_EX before it
        # gets a chance to execute. Caught a real bug once: a held
        # ID_EX instruction's decode-time register snapshot going stale
        # while its producer's one-cycle MEM_WB forwarding window came
        # and went during the freeze.
        program = """
addi x1, x0, 11
addi x2, x0, 22
sw   x1, 0(x0)
sw   x2, 64(x0)
lw   x3, 0(x0)
lw   x4, 64(x0)
add  x5, x3, x4
        """
        single = CPU()
        _load_single_cycle(single, program)
        single.run(7)

        pipelined = PipelinedCPU(data_cache_size_bytes=64, data_cache_line_size=16,
                                  hit_latency=1, miss_latency=5)
        load_program(pipelined, program, pad_instructions=40)
        while pipelined.performance.instructions_retired < 7:
            pipelined.step()

        for reg in range(1, 6):
            assert pipelined.registers.read(reg) == single.registers.read(reg)
        assert single.registers.read(5) == 33  # 11 + 22

    def test_held_operands_refresh_from_the_register_file_not_a_stale_decode_snapshot(self):
        # Narrower reproduction of the same bug: addi x2 retires *during*
        # an unrelated instruction's memory stall, while the consumer of
        # x2 is frozen in ID_EX, unable to execute (and therefore unable
        # to catch x2 via forwarding) until the stall clears -- by which
        # point x2's one-cycle MEM_WB forwarding window has already
        # closed. Only a register-file re-read at actual execute time
        # (not the value captured back at decode time) gets this right.
        program = """
addi x1, x0, 11
addi x2, x0, 22
sw   x1, 0(x0)
sw   x2, 64(x0)
        """
        cpu = PipelinedCPU(data_cache_size_bytes=64, data_cache_line_size=16,
                            hit_latency=1, miss_latency=5)
        load_program(cpu, program, pad_instructions=40)
        while cpu.performance.instructions_retired < 4:
            cpu.step()
        assert cpu.data_hierarchy.load_word(64).value == 22

    def test_memory_stalls_are_counted_independently_of_data_hazard_stalls(self):
        cpu = PipelinedCPU(data_cache_size_bytes=64, data_cache_line_size=16,
                            hit_latency=1, miss_latency=10)
        n = load_program(cpu, "addi x1, x0, 1\nsw x1, 0(x0)", pad_instructions=15)
        while cpu.performance.instructions_retired < n:
            cpu.step()
        assert cpu.performance.memory_stall_cycles == 9
        assert cpu.performance.data_hazard_stalls == 0  # no load-use hazard here
        assert cpu.performance.load_use_stalls == 0

    def test_cache_counters_mirror_the_caches_own_statistics(self):
        cpu = PipelinedCPU(data_cache_size_bytes=64, data_cache_line_size=16,
                            hit_latency=1, miss_latency=10)
        # Each sw is 4 individual byte-level cache accesses (see
        # MemoryHierarchy.store_word) -- the first store's first byte
        # misses (installs the line), its other 3 bytes and the whole
        # second store (same, now-resident line) all hit: 1 miss, 7 hits.
        program = "addi x1, x0, 1\nsw x1, 0(x0)\nsw x1, 0(x0)"
        n = load_program(cpu, program, pad_instructions=15)
        while cpu.performance.instructions_retired < n:
            cpu.step()
        assert cpu.performance.cache_accesses == cpu.data_cache.stats.accesses
        assert cpu.performance.cache_hits == cpu.data_cache.stats.hits
        assert cpu.performance.cache_misses == cpu.data_cache.stats.misses
        assert cpu.performance.cache_accesses == 8
        assert cpu.performance.cache_hits == 7
        assert cpu.performance.cache_misses == 1

    def test_dirty_writebacks_are_counted_on_eviction(self):
        cpu = PipelinedCPU(data_cache_size_bytes=64, data_cache_line_size=16,
                            hit_latency=1, miss_latency=5)
        # x1 -> address 0, x1 -> address 64 (conflicts with 0, evicts the
        # dirty line the first store just installed).
        program = "addi x1, x0, 1\nsw x1, 0(x0)\nsw x1, 64(x0)"
        n = load_program(cpu, program, pad_instructions=20)
        while cpu.performance.instructions_retired < n:
            cpu.step()
        assert cpu.performance.dirty_writebacks == 1
        assert cpu.performance.dirty_writebacks == cpu.data_cache.stats.dirty_writebacks

    def test_instruction_mix_counters_classify_the_retiring_instruction(self):
        cpu = PipelinedCPU()
        program = "addi x1, x0, 1\nsw x1, 0(x0)\nlw x2, 0(x0)\nbeq x1, x1, skip\nskip:\nadd x3, x1, x2"
        n = load_program(cpu, program, pad_instructions=20)
        while cpu.performance.instructions_retired < n:
            cpu.step()
        assert cpu.performance.alu_instructions == 2  # addi, add
        assert cpu.performance.load_instructions == 1
        assert cpu.performance.store_instructions == 1
        assert cpu.performance.branch_instructions == 1
        total = (cpu.performance.alu_instructions + cpu.performance.load_instructions
                 + cpu.performance.store_instructions + cpu.performance.branch_instructions)
        assert total == cpu.performance.instructions_retired
