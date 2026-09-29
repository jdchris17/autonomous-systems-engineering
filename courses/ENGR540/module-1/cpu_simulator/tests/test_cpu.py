import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.cpu.cpu import CPU
from cpu_simulator.isa.assembler import assemble


def _load(cpu: CPU, program: str) -> int:
    """Assemble a program and load it into the CPU's instruction
    memory starting at address 0. Returns the instruction count.
    """
    words = assemble(program)
    for i, word in enumerate(words):
        cpu.instruction_memory.write_word(i * 4, word)
    return len(words)


class TestModule6ProgramThroughRealDecoding:
    """The exact module-6 program, this time reached by actually
    decoding and executing each instruction rather than running its
    Module 5 pseudo-op equivalent.
    """

    PROGRAM = """
addi x1, x0, 10
addi x2, x0, 7
add  x3, x1, x2
sub  x4, x3, x2
sw   x4, 0(x0)
    """

    def test_matches_the_architectural_target(self):
        cpu = CPU()
        n = _load(cpu, self.PROGRAM)
        cpu.run(n)
        assert cpu.registers.read(1) == 10
        assert cpu.registers.read(2) == 7
        assert cpu.registers.read(3) == 17
        assert cpu.registers.read(4) == 10
        assert cpu.data_memory.read_word(0) == 10

    def test_pc_and_clock_advance_once_per_instruction(self):
        cpu = CPU()
        n = _load(cpu, self.PROGRAM)
        cpu.run(n)
        assert cpu.pc.read() == n * 4
        assert cpu.clock.cycle == n


class TestBranchLoop:
    PROGRAM = """
addi x5, x0, 3
loop:
    addi x5, x5, -1
    bne  x5, x0, loop
    """

    def test_loop_runs_the_expected_number_of_cycles_and_terminates(self):
        cpu = CPU()
        _load(cpu, self.PROGRAM)
        # 1 setup instruction + 3 loop iterations * 2 instructions each.
        cpu.run(7)
        assert cpu.registers.read(5) == 0
        assert cpu.pc.read() == 0xC  # falls through, past the loop
        assert cpu.clock.cycle == 7

    def test_branch_not_taken_falls_through_to_pc_plus_4(self):
        cpu = CPU()
        cpu.registers.write(1, 1)  # x1 != x2 (x2 defaults to 0) -> BEQ not taken
        _load(cpu, "beq x1, x2, 16")
        cpu.step()
        assert cpu.pc.read() == 4  # not taken -> ordinary pc + 4


class TestJumpAndLink:
    def test_jal_sets_return_address_and_jumps(self):
        cpu = CPU()
        _load(cpu, "jal x1, 100")
        cpu.step()
        assert cpu.registers.read(1) == 4  # return address = pc + 4
        assert cpu.pc.read() == 100

    def test_jal_negative_offset(self):
        # A literal (non-label) displacement encodes the same word
        # regardless of where it's placed, so the instruction is
        # written directly at address 200 rather than assembled there.
        cpu = CPU()
        cpu.pc.jump(200)
        word = assemble("jal x1, -100")[0]
        cpu.instruction_memory.write_word(200, word)
        cpu.step()
        assert cpu.pc.read() == 100


class TestJumpAndLinkRegister:
    def test_jalr_target_is_rs1_plus_immediate(self):
        cpu = CPU()
        cpu.registers.write(2, 40)
        _load(cpu, "jalr x1, x2, 8")
        cpu.step()
        assert cpu.registers.read(1) == 4  # return address
        assert cpu.pc.read() == 48

    def test_jalr_clears_the_low_bit(self):
        cpu = CPU()
        cpu.registers.write(2, 41)  # odd base
        _load(cpu, "jalr x1, x2, 0")
        cpu.step()
        assert cpu.pc.read() == 40  # low bit cleared


class TestLuiAndAuipc:
    def test_lui(self):
        cpu = CPU()
        _load(cpu, "lui x3, 74565")
        cpu.step()
        assert cpu.registers.read(3) == 0x12345000

    def test_auipc_adds_the_current_pc(self):
        cpu = CPU()
        cpu.pc.jump(0x100)
        word = assemble("auipc x3, 4096")[0]
        cpu.instruction_memory.write_word(0x100, word)
        cpu.step()
        assert cpu.registers.read(3) == 0x1000000 + 0x100


class TestX0StaysZero:
    def test_writing_to_x0_is_discarded_through_the_full_pipeline(self):
        cpu = CPU()
        _load(cpu, "addi x0, x0, 5")
        cpu.step()
        assert cpu.registers.read(0) == 0


class TestPerformanceCounters:
    """The single-cycle model structurally can't stall, flush, or
    forward -- nothing is ever in flight to do any of those between --
    so those four counters must stay at 0 no matter what runs.
    """

    def test_one_instruction_retires_every_cycle(self):
        cpu = CPU()
        n = _load(cpu, "addi x1, x0, 1\naddi x1, x1, 1\naddi x1, x1, 1")
        cpu.run(n)
        assert cpu.performance.instructions_retired == n
        assert cpu.performance.total_cycles == n

    def test_cpi_is_always_exactly_one(self):
        from cpu_simulator.cpu.analysis.performance import cpi, ipc

        cpu = CPU()
        n = _load(cpu, "addi x1, x0, 1\naddi x1, x1, 1\naddi x1, x1, 1")
        cpu.run(n)
        assert cpi(cpu.performance) == 1.0
        assert ipc(cpu.performance) == 1.0

    def test_stall_flush_and_forwarding_counters_never_move(self):
        cpu = CPU()
        n = _load(cpu, "addi x1, x0, 1\naddi x1, x1, 1\naddi x1, x1, 1")
        cpu.run(n)
        assert cpu.performance.data_hazard_stalls == 0
        assert cpu.performance.load_use_stalls == 0
        assert cpu.performance.branch_flush_cycles == 0
        assert cpu.performance.forwarding_events == 0
        assert cpu.performance.alu_instructions == n

    def test_branch_instructions_and_taken_count_are_tracked(self):
        cpu = CPU()
        n = _load(cpu, "addi x5, x0, 3\nloop:\n    addi x5, x5, -1\n    bne x5, x0, loop")
        cpu.run(7)  # 1 setup + 3 iterations * 2 instructions
        assert cpu.performance.branch_instructions == 3
        assert cpu.performance.branches_taken == 2  # last iteration falls through
