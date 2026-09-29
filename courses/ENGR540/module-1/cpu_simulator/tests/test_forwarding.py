import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.cpu.control import ControlSignals
from cpu_simulator.cpu.forwarding import (
    EX_MEM_RESULT,
    MEM_WB_RESULT,
    REGISTER_VALUE,
    determine_forwarding,
    forward_operands,
    select,
)
from cpu_simulator.cpu.pipeline import EX_MEM, ID_EX, MEM_WB


def _control(**overrides) -> ControlSignals:
    defaults = dict(
        reg_write=True, mem_read=False, mem_write=False,
        alu_src="REGISTER", alu_op="ADD", result_src="ALU",
        branch=None, jump=False,
    )
    defaults.update(overrides)
    return ControlSignals(**defaults)


class TestDetermineForwarding:
    def test_no_producers_falls_back_to_register_value(self):
        assert determine_forwarding(5, EX_MEM(), MEM_WB()) == REGISTER_VALUE

    def test_ex_mem_producer_wins(self):
        ex_mem = EX_MEM(valid=True, rd=5, control=_control())
        assert determine_forwarding(5, ex_mem, MEM_WB()) == EX_MEM_RESULT

    def test_mem_wb_producer_used_when_ex_mem_does_not_match(self):
        mem_wb = MEM_WB(valid=True, rd=5, control=_control())
        assert determine_forwarding(5, EX_MEM(), mem_wb) == MEM_WB_RESULT

    def test_ex_mem_takes_priority_over_mem_wb(self):
        ex_mem = EX_MEM(valid=True, rd=5, control=_control())
        mem_wb = MEM_WB(valid=True, rd=5, control=_control())
        assert determine_forwarding(5, ex_mem, mem_wb) == EX_MEM_RESULT

    def test_x0_never_forwards(self):
        ex_mem = EX_MEM(valid=True, rd=0, control=_control())
        mem_wb = MEM_WB(valid=True, rd=0, control=_control())
        assert determine_forwarding(0, ex_mem, mem_wb) == REGISTER_VALUE

    def test_non_matching_rd_falls_through(self):
        ex_mem = EX_MEM(valid=True, rd=6, control=_control())
        assert determine_forwarding(5, ex_mem, MEM_WB()) == REGISTER_VALUE

    def test_invalid_stage_never_forwards_even_with_matching_rd(self):
        ex_mem = EX_MEM(valid=False, rd=5, control=_control())
        assert determine_forwarding(5, ex_mem, MEM_WB()) == REGISTER_VALUE

    def test_non_reg_write_producer_never_forwards(self):
        ex_mem = EX_MEM(valid=True, rd=5, control=_control(reg_write=False))
        assert determine_forwarding(5, ex_mem, MEM_WB()) == REGISTER_VALUE

    def test_ex_mem_load_is_skipped_its_alu_result_is_an_address_not_data(self):
        # A load still in EX_MEM hasn't read memory yet -- its
        # alu_result is the address, not the value. Forwarding it would
        # forward the wrong number, so this falls through to MEM_WB.
        ex_mem = EX_MEM(valid=True, rd=5, control=_control(mem_read=True))
        mem_wb = MEM_WB(valid=True, rd=5, control=_control())
        assert determine_forwarding(5, ex_mem, mem_wb) == MEM_WB_RESULT

    def test_ex_mem_load_with_no_mem_wb_producer_falls_all_the_way_to_register(self):
        ex_mem = EX_MEM(valid=True, rd=5, control=_control(mem_read=True))
        assert determine_forwarding(5, ex_mem, MEM_WB()) == REGISTER_VALUE


class TestSelect:
    def test_register_value(self):
        assert select(REGISTER_VALUE, 1, 2, 3) == 1

    def test_ex_mem_result(self):
        assert select(EX_MEM_RESULT, 1, 2, 3) == 2

    def test_mem_wb_result(self):
        assert select(MEM_WB_RESULT, 1, 2, 3) == 3

    def test_unknown_source_rejected(self):
        with pytest.raises(ValueError):
            select("NONSENSE", 1, 2, 3)


class TestForwardOperands:
    def test_no_hazard_uses_id_exs_own_values(self):
        id_ex = ID_EX(valid=True, rs1=1, rs2=2, rs1_value=10, rs2_value=20)
        a, b = forward_operands(id_ex, EX_MEM(), MEM_WB())
        assert (a, b) == (10, 20)

    def test_forwards_from_ex_mem_for_rs1(self):
        id_ex = ID_EX(valid=True, rs1=5, rs2=2, rs1_value=999, rs2_value=20)
        ex_mem = EX_MEM(valid=True, rd=5, alu_result=42, control=_control())
        a, b = forward_operands(id_ex, ex_mem, MEM_WB())
        assert (a, b) == (42, 20)

    def test_forwards_from_mem_wb_for_rs2(self):
        id_ex = ID_EX(valid=True, rs1=1, rs2=6, rs1_value=10, rs2_value=999)
        mem_wb = MEM_WB(valid=True, rd=6, memory_result=77, control=_control())
        a, b = forward_operands(id_ex, EX_MEM(), mem_wb)
        assert (a, b) == (10, 77)

    def test_mem_wb_forwards_pc_plus_4_for_jal(self):
        id_ex = ID_EX(valid=True, rs1=1, rs2=1, rs1_value=999, rs2_value=999)
        mem_wb = MEM_WB(valid=True, rd=1, pc_plus_4=0x100,
                         control=_control(result_src="PC_PLUS_4"))
        a, b = forward_operands(id_ex, EX_MEM(), mem_wb)
        assert a == 0x100 and b == 0x100

    def test_the_priority_matters_worked_example(self):
        # add x5, x1, x2 (retired, its result now in MEM_WB)
        # addi x5, x5, 1 (its result now in EX_MEM -- the newer producer)
        # sub x6, x5, x3 (currently in ID_EX, about to enter EX)
        id_ex = ID_EX(valid=True, rs1=5, rs2=3, rs1_value=0, rs2_value=1)  # rs1_value stale
        ex_mem = EX_MEM(valid=True, rd=5, alu_result=31, control=_control())   # ADDI's result
        mem_wb = MEM_WB(valid=True, rd=5, memory_result=30, control=_control())  # ADD's result
        a, b = forward_operands(id_ex, ex_mem, mem_wb)
        assert a == 31  # the newer (ADDI/EX_MEM) value wins, not ADD's older 30
