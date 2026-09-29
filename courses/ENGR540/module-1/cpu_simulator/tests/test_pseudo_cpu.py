import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.core.alu import alu as alu_function
from cpu_simulator.cpu.clock import Clock
from cpu_simulator.memory.memory import Memory
from cpu_simulator.cpu.pc import ProgramCounter
from cpu_simulator.cpu.pseudo_cpu import PseudoOpCPU as CPU
from cpu_simulator.cpu.register_file import RegisterFile


class TestConstruction:
    def test_components_are_the_expected_types(self):
        cpu = CPU()
        assert isinstance(cpu.pc, ProgramCounter)
        assert isinstance(cpu.registers, RegisterFile)
        assert cpu.alu is alu_function
        assert isinstance(cpu.instruction_memory, Memory)
        assert isinstance(cpu.data_memory, Memory)
        assert isinstance(cpu.clock, Clock)

    def test_instruction_and_data_memory_are_separate_instances(self):
        cpu = CPU()
        assert cpu.instruction_memory is not cpu.data_memory
        cpu.instruction_memory.write_word(0, 0xDEADBEEF)
        assert cpu.data_memory.read_word(0) == 0

    def test_memory_sizes_are_configurable(self):
        cpu = CPU(instruction_memory_bytes=256, data_memory_bytes=64)
        assert cpu.instruction_memory.size_bytes == 256
        assert cpu.data_memory.size_bytes == 64

    def test_starts_at_pc_zero_cycle_zero(self):
        cpu = CPU()
        assert cpu.pc.read() == 0
        assert cpu.clock.cycle == 0


class TestStepAdd:
    def test_adds_two_registers(self):
        cpu = CPU()
        cpu.registers.write(1, 5)
        cpu.registers.write(2, 7)
        cpu.step_add(rd=3, rs1=1, rs2=2)
        assert cpu.registers.read(3) == 12

    def test_returns_the_alu_result(self):
        cpu = CPU()
        cpu.registers.write(1, 5)
        cpu.registers.write(2, 7)
        result = cpu.step_add(rd=3, rs1=1, rs2=2)
        assert result.value == 12
        assert not result.zero

    def test_advances_pc_and_clock(self):
        cpu = CPU()
        cpu.step_add(rd=1, rs1=0, rs2=0)
        assert cpu.pc.read() == 4
        assert cpu.clock.cycle == 1

    def test_writing_to_x0_is_discarded(self):
        cpu = CPU()
        cpu.registers.write(1, 5)
        cpu.registers.write(2, 7)
        cpu.step_add(rd=0, rs1=1, rs2=2)
        assert cpu.registers.read(0) == 0

    def test_multiple_steps_accumulate(self):
        cpu = CPU()
        cpu.registers.write(1, 1)
        cpu.step_add(rd=1, rs1=1, rs2=1)   # x1 = 2
        cpu.step_add(rd=1, rs1=1, rs2=1)   # x1 = 4
        cpu.step_add(rd=1, rs1=1, rs2=1)   # x1 = 8
        assert cpu.registers.read(1) == 8
        assert cpu.pc.read() == 12
        assert cpu.clock.cycle == 3


class TestStepLoadStoreWord:
    def test_store_then_load_round_trips(self):
        cpu = CPU()
        cpu.registers.write(1, 0xCAFEBABE)  # value to store
        cpu.registers.write(2, 16)          # base address
        cpu.step_store_word(rs=1, base=2, offset=0)
        cpu.step_load_word(rd=3, base=2, offset=0)
        assert cpu.registers.read(3) == 0xCAFEBABE

    def test_address_is_base_plus_offset(self):
        cpu = CPU()
        cpu.registers.write(1, 0x11223344)
        cpu.registers.write(2, 100)  # base
        cpu.step_store_word(rs=1, base=2, offset=8)
        # written directly at address 108, bypassing step_load_word,
        # to confirm the address arithmetic independently
        assert cpu.data_memory.read_word(108) == 0x11223344

    def test_negative_offset(self):
        cpu = CPU()
        cpu.registers.write(1, 0xAABBCCDD)
        cpu.registers.write(2, 100)  # base
        cpu.step_store_word(rs=1, base=2, offset=-4)
        assert cpu.data_memory.read_word(96) == 0xAABBCCDD

    def test_load_advances_pc_and_clock(self):
        cpu = CPU()
        cpu.registers.write(2, 0)
        cpu.step_load_word(rd=1, base=2, offset=0)
        assert cpu.pc.read() == 4
        assert cpu.clock.cycle == 1

    def test_store_advances_pc_and_clock(self):
        cpu = CPU()
        cpu.registers.write(2, 0)
        cpu.step_store_word(rs=1, base=2, offset=0)
        assert cpu.pc.read() == 4
        assert cpu.clock.cycle == 1

    def test_loading_into_x0_is_discarded(self):
        cpu = CPU()
        cpu.registers.write(2, 0)
        cpu.data_memory.write_word(0, 0x12345678)
        cpu.step_load_word(rd=0, base=2, offset=0)
        assert cpu.registers.read(0) == 0


class TestStepAluOp:
    def test_generalizes_beyond_add(self):
        cpu = CPU()
        cpu.registers.write(1, 0b1100)
        cpu.registers.write(2, 0b1010)
        cpu.step_alu_op("AND", rd=3, rs1=1, rs2=2)
        assert cpu.registers.read(3) == 0b1000

    def test_sub(self):
        cpu = CPU()
        cpu.registers.write(1, 10)
        cpu.registers.write(2, 3)
        cpu.step_alu_op("SUB", rd=3, rs1=1, rs2=2)
        assert cpu.registers.read(3) == 7

    def test_step_add_is_equivalent_to_step_alu_op_add(self):
        a, b = CPU(), CPU()
        a.registers.write(1, 5)
        a.registers.write(2, 7)
        b.registers.write(1, 5)
        b.registers.write(2, 7)
        a.step_add(rd=3, rs1=1, rs2=2)
        b.step_alu_op("ADD", rd=3, rs1=1, rs2=2)
        assert a.registers.read(3) == b.registers.read(3)


class TestStepLoadImmediate:
    def test_sets_the_register(self):
        cpu = CPU()
        cpu.step_load_immediate(rd=1, immediate=10)
        assert cpu.registers.read(1) == 10

    def test_negative_immediate(self):
        cpu = CPU()
        cpu.step_load_immediate(rd=1, immediate=-1)
        assert cpu.registers.read(1) == 0xFFFFFFFF

    def test_advances_pc_and_clock(self):
        cpu = CPU()
        cpu.step_load_immediate(rd=1, immediate=42)
        assert cpu.pc.read() == 4
        assert cpu.clock.cycle == 1

    def test_uses_the_alu_not_a_direct_write(self):
        import inspect

        source = inspect.getsource(CPU.step_load_immediate)
        assert "self.alu(" in source


class TestExecuteAndRun:
    def test_execute_dispatches_add(self):
        cpu = CPU()
        cpu.registers.write(1, 5)
        cpu.registers.write(2, 7)
        cpu.execute(("ADD", 3, 1, 2))
        assert cpu.registers.read(3) == 12

    def test_execute_dispatches_every_alu_operation_generically(self):
        cpu = CPU()
        cpu.registers.write(1, 0xF0)
        cpu.registers.write(2, 0x0F)
        cpu.execute(("XOR", 3, 1, 2))
        assert cpu.registers.read(3) == 0xFF

    def test_execute_dispatches_load_imm(self):
        cpu = CPU()
        cpu.execute(("LOAD_IMM", 1, 99))
        assert cpu.registers.read(1) == 99

    def test_execute_dispatches_load_and_store(self):
        cpu = CPU()
        cpu.registers.write(1, 0xABCD)
        cpu.registers.write(2, 100)  # base
        cpu.execute(("STORE", 1, 2, 12))
        cpu.execute(("LOAD", 3, 2, 12))
        assert cpu.registers.read(3) == 0xABCD

    def test_store_with_x0_base_is_absolute_addressing(self):
        cpu = CPU()
        cpu.registers.write(1, 0x1234)
        cpu.execute(("STORE", 1, 0, 100))
        assert cpu.data_memory.read_word(100) == 0x1234

    def test_unknown_opcode_rejected(self):
        cpu = CPU()
        with pytest.raises(ValueError):
            cpu.execute(("MUL", 1, 2, 3))

    def test_run_executes_a_whole_program_in_order(self):
        cpu = CPU()
        program = [
            ("LOAD_IMM", 1, 10),
            ("LOAD_IMM", 2, 7),
            ("ADD", 3, 1, 2),
            ("SUB", 4, 3, 2),
            ("STORE", 4, 0, 100),
        ]
        cpu.run(program)
        assert cpu.registers.read(1) == 10
        assert cpu.registers.read(2) == 7
        assert cpu.registers.read(3) == 17
        assert cpu.registers.read(4) == 10
        assert cpu.data_memory.read_word(100) == 10
        assert cpu.pc.read() == 20  # 5 instructions * 4 bytes
        assert cpu.clock.cycle == 5
