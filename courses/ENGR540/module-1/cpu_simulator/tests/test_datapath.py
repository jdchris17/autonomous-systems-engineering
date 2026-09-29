import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.cpu.control import generate_control
from cpu_simulator.cpu.datapath import Datapath, ExecuteResult
from cpu_simulator.cpu.register_file import RegisterFile
from cpu_simulator.isa.assembler import assemble_line
from cpu_simulator.isa.decoder import decode
from cpu_simulator.memory.memory import Memory


def _make_datapath():
    instruction_memory = Memory(size_bytes=64)
    data_memory = Memory(size_bytes=64)
    registers = RegisterFile()
    return Datapath(instruction_memory, data_memory, registers)


class TestFetch:
    def test_reads_the_word_at_the_given_address(self):
        dp = _make_datapath()
        word = assemble_line("add x1, x2, x3")
        dp.instruction_memory.write_word(8, word)
        assert dp.fetch(8) == word


class TestDecode:
    def test_delegates_to_decoder_py(self):
        dp = _make_datapath()
        word = assemble_line("add x5, x6, x7")
        inst = dp.decode(word)
        assert inst.mnemonic == "ADD"
        assert (inst.rd, inst.rs1, inst.rs2) == (5, 6, 7)


class TestExecuteRType:
    def test_add(self):
        dp = _make_datapath()
        dp.registers.write(1, 5)
        dp.registers.write(2, 7)
        inst = decode(assemble_line("add x3, x1, x2"))
        control = generate_control(inst)
        result = dp.execute(inst, control, pc=0)
        assert result.alu_value == 12
        assert result.branch_taken is False

    def test_sub(self):
        dp = _make_datapath()
        dp.registers.write(1, 10)
        dp.registers.write(2, 3)
        inst = decode(assemble_line("sub x3, x1, x2"))
        result = dp.execute(inst, generate_control(inst), pc=0)
        assert result.alu_value == 7


class TestExecuteImmediateAndLuiAuipc:
    def test_addi_uses_immediate_not_a_second_register(self):
        dp = _make_datapath()
        dp.registers.write(1, 5)
        inst = decode(assemble_line("addi x2, x1, 10"))
        result = dp.execute(inst, generate_control(inst), pc=0)
        assert result.alu_value == 15

    def test_lui_uses_zero_as_first_operand(self):
        dp = _make_datapath()
        inst = decode(assemble_line("lui x3, 74565"))
        result = dp.execute(inst, generate_control(inst), pc=0x1000)
        assert result.alu_value == 0x12345000  # pc is irrelevant to LUI

    def test_auipc_uses_pc_as_first_operand(self):
        dp = _make_datapath()
        inst = decode(assemble_line("auipc x3, 4096"))  # imm = 0x1000000
        result = dp.execute(inst, generate_control(inst), pc=0x100)
        assert result.alu_value == 0x1000000 + 0x100


class TestExecuteBranches:
    def test_beq_taken_when_equal(self):
        dp = _make_datapath()
        dp.registers.write(1, 5)
        dp.registers.write(2, 5)
        inst = decode(assemble_line("beq x1, x2, 16"))
        result = dp.execute(inst, generate_control(inst), pc=0)
        assert result.branch_taken is True

    def test_beq_not_taken_when_unequal(self):
        dp = _make_datapath()
        dp.registers.write(1, 5)
        dp.registers.write(2, 6)
        inst = decode(assemble_line("beq x1, x2, 16"))
        result = dp.execute(inst, generate_control(inst), pc=0)
        assert result.branch_taken is False

    def test_bne_is_the_opposite_of_beq(self):
        dp = _make_datapath()
        dp.registers.write(1, 5)
        dp.registers.write(2, 6)
        inst = decode(assemble_line("bne x1, x2, 16"))
        result = dp.execute(inst, generate_control(inst), pc=0)
        assert result.branch_taken is True

    def test_blt_taken_when_less(self):
        dp = _make_datapath()
        dp.registers.write(1, 3)
        dp.registers.write(2, 5)
        inst = decode(assemble_line("blt x1, x2, 16"))
        result = dp.execute(inst, generate_control(inst), pc=0)
        assert result.branch_taken is True

    def test_bge_taken_when_equal_or_greater(self):
        dp = _make_datapath()
        dp.registers.write(1, 5)
        dp.registers.write(2, 5)
        inst = decode(assemble_line("bge x1, x2, 16"))
        result = dp.execute(inst, generate_control(inst), pc=0)
        assert result.branch_taken is True

    def test_bltu_treats_operands_as_unsigned(self):
        dp = _make_datapath()
        dp.registers.write(1, 0xFFFFFFFF)  # -1 signed, huge unsigned
        dp.registers.write(2, 1)
        inst = decode(assemble_line("bltu x1, x2, 16"))
        result = dp.execute(inst, generate_control(inst), pc=0)
        assert result.branch_taken is False  # 0xFFFFFFFF is NOT < 1 unsigned

    def test_blt_treats_the_same_pattern_as_signed(self):
        dp = _make_datapath()
        dp.registers.write(1, 0xFFFFFFFF)  # -1 signed
        dp.registers.write(2, 1)
        inst = decode(assemble_line("blt x1, x2, 16"))
        result = dp.execute(inst, generate_control(inst), pc=0)
        assert result.branch_taken is True  # -1 IS < 1 signed


class TestMemoryAccess:
    def test_load_reads_the_computed_address(self):
        dp = _make_datapath()
        dp.data_memory.write_word(12, 0xCAFEBABE)
        dp.registers.write(2, 0)  # base
        inst = decode(assemble_line("lw x5, 12(x2)"))
        control = generate_control(inst)
        executed = dp.execute(inst, control, pc=0)
        result = dp.memory_access(inst, control, executed)
        assert result == 0xCAFEBABE

    def test_store_writes_the_computed_address(self):
        dp = _make_datapath()
        dp.registers.write(1, 0x11223344)  # value to store
        dp.registers.write(2, 0)  # base
        inst = decode(assemble_line("sw x1, 12(x2)"))
        control = generate_control(inst)
        executed = dp.execute(inst, control, pc=0)
        dp.memory_access(inst, control, executed)
        assert dp.data_memory.read_word(12) == 0x11223344

    def test_non_memory_instruction_passes_alu_value_through(self):
        dp = _make_datapath()
        dp.registers.write(1, 5)
        dp.registers.write(2, 7)
        inst = decode(assemble_line("add x3, x1, x2"))
        control = generate_control(inst)
        executed = dp.execute(inst, control, pc=0)
        assert dp.memory_access(inst, control, executed) == executed.alu_value == 12


class TestWriteback:
    def test_alu_result_written_to_rd(self):
        dp = _make_datapath()
        inst = decode(assemble_line("add x3, x1, x2"))
        control = generate_control(inst)
        dp.writeback(inst, control, memory_result=42, pc_plus_4=999)
        assert dp.registers.read(3) == 42

    def test_memory_result_written_for_a_load(self):
        dp = _make_datapath()
        inst = decode(assemble_line("lw x5, 0(x2)"))
        control = generate_control(inst)
        dp.writeback(inst, control, memory_result=0xABCD, pc_plus_4=999)
        assert dp.registers.read(5) == 0xABCD

    def test_pc_plus_4_written_for_jal(self):
        dp = _make_datapath()
        inst = decode(assemble_line("jal x1, 100"))
        control = generate_control(inst)
        dp.writeback(inst, control, memory_result=0, pc_plus_4=4)
        assert dp.registers.read(1) == 4

    def test_store_does_not_write_a_register(self):
        dp = _make_datapath()
        inst = decode(assemble_line("sw x1, 0(x2)"))
        control = generate_control(inst)
        before = [dp.registers.read(r) for r in range(32)]
        dp.writeback(inst, control, memory_result=999, pc_plus_4=999)
        after = [dp.registers.read(r) for r in range(32)]
        assert before == after

    def test_branch_does_not_write_a_register(self):
        dp = _make_datapath()
        inst = decode(assemble_line("beq x1, x2, 16"))
        control = generate_control(inst)
        before = [dp.registers.read(r) for r in range(32)]
        dp.writeback(inst, control, memory_result=999, pc_plus_4=999)
        after = [dp.registers.read(r) for r in range(32)]
        assert before == after
