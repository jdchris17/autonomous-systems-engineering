import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.isa.instruction import FORMATS, Instruction


class TestInstruction:
    def test_required_fields(self):
        inst = Instruction(raw=0x33, mnemonic="ADD", opcode=0b0110011, format="R")
        assert inst.raw == 0x33
        assert inst.mnemonic == "ADD"
        assert inst.opcode == 0b0110011
        assert inst.format == "R"

    def test_optional_fields_default_to_none(self):
        inst = Instruction(raw=0, mnemonic="LUI", opcode=0b0110111, format="U")
        assert inst.rd is None
        assert inst.rs1 is None
        assert inst.rs2 is None
        assert inst.funct3 is None
        assert inst.funct7 is None
        assert inst.immediate is None

    def test_can_populate_every_field(self):
        inst = Instruction(
            raw=0x002081B3, mnemonic="ADD", opcode=0b0110011, format="R",
            rd=3, rs1=1, rs2=2, funct3=0, funct7=0,
        )
        assert (inst.rd, inst.rs1, inst.rs2, inst.funct3, inst.funct7) == (3, 1, 2, 0, 0)

    def test_formats_constant(self):
        assert FORMATS == ("R", "I", "S", "B", "U", "J")
