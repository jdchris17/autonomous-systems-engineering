import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.cpu.control import ControlSignals, generate_control
from cpu_simulator.isa.assembler import assemble_line
from cpu_simulator.isa.decoder import decode


def _control_for(asm: str) -> ControlSignals:
    return generate_control(decode(assemble_line(asm)))


class TestSection26WorkedExamples:
    def test_add(self):
        control = _control_for("add x5, x6, x7")
        assert control.reg_write is True
        assert control.mem_read is False
        assert control.mem_write is False
        assert control.alu_src == "REGISTER"
        assert control.alu_op == "ADD"
        assert control.result_src == "ALU"
        assert control.branch is None
        assert control.jump is False

    def test_load(self):
        control = _control_for("lw x5, 12(x2)")
        assert control.reg_write is True
        assert control.mem_read is True
        assert control.mem_write is False
        assert control.alu_src == "IMMEDIATE"
        assert control.alu_op == "ADD"
        assert control.result_src == "MEMORY"
        assert control.branch is None
        assert control.jump is False


class TestRTypeAluControl:
    @pytest.mark.parametrize("asm,expected_op", [
        ("add x1, x2, x3", "ADD"),
        ("sub x1, x2, x3", "SUB"),
        ("and x1, x2, x3", "AND"),
        ("or x1, x2, x3", "OR"),
        ("xor x1, x2, x3", "XOR"),
        ("sll x1, x2, x3", "SLL"),
        ("srl x1, x2, x3", "SRL"),
        ("sra x1, x2, x3", "SRA"),
        ("slt x1, x2, x3", "SLT"),
        ("sltu x1, x2, x3", "SLTU"),
    ])
    def test_alu_op_resolved_from_funct3_funct7(self, asm, expected_op):
        control = _control_for(asm)
        assert control.alu_op == expected_op
        assert control.alu_src == "REGISTER"
        assert control.reg_write is True
        assert control.result_src == "ALU"


class TestITypeArithmeticAluControl:
    @pytest.mark.parametrize("asm,expected_op", [
        ("addi x1, x2, 10", "ADD"),
        ("slti x1, x2, 10", "SLT"),
        ("sltiu x1, x2, 10", "SLTU"),
        ("xori x1, x2, 10", "XOR"),
        ("ori x1, x2, 10", "OR"),
        ("andi x1, x2, 10", "AND"),
    ])
    def test_alu_op_resolved_from_funct3(self, asm, expected_op):
        control = _control_for(asm)
        assert control.alu_op == expected_op
        assert control.alu_src == "IMMEDIATE"

    @pytest.mark.parametrize("asm,expected_op", [
        ("slli x1, x2, 5", "SLL"),
        ("srli x1, x2, 5", "SRL"),
        ("srai x1, x2, 5", "SRA"),
    ])
    def test_shift_immediate_alu_op_needs_funct3_and_funct7(self, asm, expected_op):
        control = _control_for(asm)
        assert control.alu_op == expected_op


class TestStore:
    def test_sw(self):
        control = _control_for("sw x5, 12(x2)")
        assert control.reg_write is False
        assert control.mem_read is False
        assert control.mem_write is True
        assert control.alu_src == "IMMEDIATE"
        assert control.alu_op == "ADD"
        assert control.result_src is None


class TestBranch:
    @pytest.mark.parametrize("asm,expected_mnemonic,expected_alu_op", [
        ("beq x1, x2, 16", "BEQ", "SUB"),
        ("bne x1, x2, 16", "BNE", "SUB"),
        ("blt x1, x2, 16", "BLT", "SLT"),
        ("bge x1, x2, 16", "BGE", "SLT"),
        ("bltu x1, x2, 16", "BLTU", "SLTU"),
        ("bgeu x1, x2, 16", "BGEU", "SLTU"),
    ])
    def test_branch_signals(self, asm, expected_mnemonic, expected_alu_op):
        control = _control_for(asm)
        assert control.reg_write is False
        assert control.mem_read is False
        assert control.mem_write is False
        assert control.alu_src == "REGISTER"
        assert control.branch == expected_mnemonic
        assert control.alu_op == expected_alu_op
        assert control.jump is False


class TestJumpsAndUpperImmediates:
    def test_jal(self):
        control = _control_for("jal x1, 1024")
        assert control.reg_write is True
        assert control.result_src == "PC_PLUS_4"
        assert control.jump is True
        assert control.branch is None

    def test_jalr(self):
        control = _control_for("jalr x1, x2, 4")
        assert control.reg_write is True
        assert control.alu_src == "IMMEDIATE"
        assert control.alu_op == "ADD"
        assert control.result_src == "PC_PLUS_4"
        assert control.jump is True

    def test_lui(self):
        control = _control_for("lui x3, 74565")
        assert control.reg_write is True
        assert control.alu_src == "IMMEDIATE"
        assert control.alu_op == "ADD"
        assert control.result_src == "ALU"
        assert control.jump is False

    def test_auipc(self):
        control = _control_for("auipc x3, 4096")
        assert control.reg_write is True
        assert control.alu_op == "ADD"
        assert control.result_src == "ALU"


class TestAluControlDrawsOnlyFromFunct3Funct7NotMnemonic:
    """The pedagogical point of section 27: ALU control resolves the
    operation from funct3/funct7, the same fields real hardware would
    have -- not by taking a shortcut through Instruction.mnemonic.
    """

    def test_source_does_not_reference_mnemonic(self):
        import inspect
        from cpu_simulator.cpu.control import _generate_alu_control
        source = inspect.getsource(_generate_alu_control)
        assert "mnemonic" not in source

    def test_two_different_mnemonics_sharing_funct3_are_disambiguated_by_funct7(self):
        # SRL and SRA share funct3 (0b101); only funct7 tells them apart.
        # If ALU control were reading funct3 alone, these would collide.
        srl = _control_for("srl x1, x2, x3")
        sra = _control_for("sra x1, x2, x3")
        assert srl.alu_op == "SRL"
        assert sra.alu_op == "SRA"
        assert srl.alu_op != sra.alu_op


class TestUnsupportedOpcode:
    def test_rejected(self):
        from cpu_simulator.isa.instruction import Instruction

        bogus = Instruction(raw=0, mnemonic="???", opcode=0b1111111, format="R")
        with pytest.raises(ValueError):
            generate_control(bogus)
