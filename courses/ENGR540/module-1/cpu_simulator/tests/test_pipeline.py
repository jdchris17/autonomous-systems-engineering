import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.cpu.control import ControlSignals
from cpu_simulator.cpu.pipeline import EX_MEM, ID_EX, IF_ID, MEM_WB


class TestDefaultsAreBubbles:
    """Every register's default construction -- no arguments at all --
    should be a valid=False bubble. This is what an empty pipeline slot
    at startup, or a stall injecting an empty stage, actually is.
    """

    def test_if_id(self):
        stage = IF_ID()
        assert stage.valid is False
        assert stage.pc == 0
        assert stage.instruction == 0

    def test_id_ex(self):
        stage = ID_EX()
        assert stage.valid is False
        assert stage.control is None

    def test_ex_mem(self):
        stage = EX_MEM()
        assert stage.valid is False
        assert stage.branch_taken is False
        assert stage.control is None

    def test_mem_wb(self):
        stage = MEM_WB()
        assert stage.valid is False
        assert stage.control is None


class TestValidIsTheSoleDiscriminant:
    """valid=False means "ignore everything else in this register,"
    not "every other field happens to be zero." A bubble can carry
    stale/leftover field values (e.g. reused from a flushed
    instruction) and still be correctly treated as empty -- nothing
    downstream should need those other fields to also be zeroed out.
    """

    def test_a_bubble_can_carry_nonzero_fields_and_still_be_a_bubble(self):
        stage = IF_ID(valid=False, pc=0x100, instruction=0x007302B3)
        assert stage.valid is False
        # The other fields are still whatever was set -- valid alone
        # is what a caller is expected to check first.
        assert stage.pc == 0x100
        assert stage.instruction == 0x007302B3

    def test_flushing_is_just_setting_valid_false(self):
        stage = ID_EX(valid=True, pc=0x20, rd=5, rs1_value=10, rs2_value=7)
        flushed = ID_EX(**{**stage.__dict__, "valid": False})
        assert flushed.valid is False
        assert flushed.rd == 5  # untouched -- flushing doesn't need to clear it


class TestEachRegisterCarriesWhatItsNextStageNeeds:
    """Not just "does construction work" -- does each register actually
    hold everything the *next* Datapath stage's method signature needs?
    """

    def test_if_id_carries_what_decode_needs(self):
        # Datapath.decode(raw) needs exactly the raw word.
        stage = IF_ID(valid=True, pc=0x10, instruction=0x007302B3)
        assert stage.instruction == 0x007302B3

    def test_id_ex_carries_what_execute_needs(self):
        # Datapath.execute(instruction, control, pc) needs operand
        # values (not register numbers) and the control signals
        # decode-time control generation already resolved.
        control = ControlSignals(
            reg_write=True, mem_read=False, mem_write=False,
            alu_src="REGISTER", alu_op="ADD", result_src="ALU",
            branch=None, jump=False,
        )
        stage = ID_EX(valid=True, pc=0x20, mnemonic="ADD", rs1=6, rs2=7,
                       rs1_value=10, rs2_value=7, rd=5, immediate=0, control=control)
        assert stage.rs1_value == 10
        assert stage.rs2_value == 7
        assert stage.control.alu_op == "ADD"
        # rs1/rs2 (the numbers, not the values) are what hazard
        # detection will need later -- both are present alongside the
        # values, not replacing them.
        assert (stage.rs1, stage.rs2) == (6, 7)

    def test_ex_mem_carries_what_memory_access_needs(self):
        # Datapath.memory_access(instruction, control, executed) needs
        # the computed address (alu_result) and, for a store, the
        # value to write (rs2_value) -- latched forward, not re-read
        # from the register file at this later stage.
        control = ControlSignals(
            reg_write=False, mem_read=False, mem_write=True,
            alu_src="IMMEDIATE", alu_op="ADD", result_src=None,
            branch=None, jump=False,
        )
        stage = EX_MEM(valid=True, pc=0x24, mnemonic="SW", alu_result=100,
                        rs2_value=42, rd=0, control=control)
        assert stage.alu_result == 100  # the store address
        assert stage.rs2_value == 42    # the value being stored

    def test_mem_wb_carries_what_writeback_needs(self):
        # Datapath.writeback(instruction, control, memory_result,
        # pc_plus_4) needs one resolved value plus pc_plus_4 for the
        # JAL/JALR case, and rd to know where it goes.
        control = ControlSignals(
            reg_write=True, mem_read=True, mem_write=False,
            alu_src="IMMEDIATE", alu_op="ADD", result_src="MEMORY",
            branch=None, jump=False,
        )
        stage = MEM_WB(valid=True, pc=0x28, mnemonic="LW",
                        memory_result=0xCAFEBABE, rd=5, pc_plus_4=0x2C, control=control)
        assert stage.memory_result == 0xCAFEBABE
        assert stage.rd == 5


class TestDataclassEquality:
    """Plain dataclasses compare by value, for free -- worth confirming
    since a future pipeline's stall/flush logic will likely compare a
    register against IF_ID()/ID_EX()/etc. (a fresh bubble) to detect
    "is this stage currently empty."
    """

    def test_two_bubbles_are_equal(self):
        assert IF_ID() == IF_ID()
        assert ID_EX() == ID_EX()
        assert EX_MEM() == EX_MEM()
        assert MEM_WB() == MEM_WB()

    def test_a_populated_register_is_not_equal_to_a_bubble(self):
        assert IF_ID(valid=True, pc=4, instruction=0x33) != IF_ID()
