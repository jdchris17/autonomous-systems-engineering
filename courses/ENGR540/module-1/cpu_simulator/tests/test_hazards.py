import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.cpu.control import ControlSignals
from cpu_simulator.cpu.hazards import (
    HazardControl,
    branch_hazard_control,
    detect_load_use_hazard,
    load_use_hazard_control,
)
from cpu_simulator.cpu.pipeline import ID_EX


def _control(**overrides) -> ControlSignals:
    defaults = dict(
        reg_write=True, mem_read=False, mem_write=False,
        alu_src="REGISTER", alu_op="ADD", result_src="ALU",
        branch=None, jump=False,
    )
    defaults.update(overrides)
    return ControlSignals(**defaults)


class TestDetectLoadUseHazard:
    def test_no_hazard_when_producer_is_invalid(self):
        producer = ID_EX(valid=False, rd=5, control=_control(mem_read=True))
        assert detect_load_use_hazard(producer, 5, 6) is None

    def test_no_hazard_when_producer_is_not_a_load(self):
        producer = ID_EX(valid=True, rd=5, control=_control(mem_read=False))
        assert detect_load_use_hazard(producer, 5, 6) is None

    def test_hazard_when_load_produces_consumers_rs1(self):
        producer = ID_EX(valid=True, rd=5, control=_control(mem_read=True))
        assert detect_load_use_hazard(producer, 5, 6) == 5

    def test_hazard_when_load_produces_consumers_rs2(self):
        producer = ID_EX(valid=True, rd=5, control=_control(mem_read=True))
        assert detect_load_use_hazard(producer, 6, 5) == 5

    def test_no_hazard_when_rd_matches_neither_operand(self):
        producer = ID_EX(valid=True, rd=5, control=_control(mem_read=True))
        assert detect_load_use_hazard(producer, 6, 7) is None

    def test_x0_never_triggers_a_hazard(self):
        producer = ID_EX(valid=True, rd=0, control=_control(mem_read=True))
        assert detect_load_use_hazard(producer, 0, 0) is None

    def test_no_hazard_when_producer_control_is_none(self):
        producer = ID_EX(valid=True, rd=5, control=None)
        assert detect_load_use_hazard(producer, 5, 6) is None


class TestLoadUseHazardControl:
    def test_no_conflict_produces_an_inert_control(self):
        control = load_use_hazard_control(None)
        assert control == HazardControl()

    def test_a_conflict_stalls_fetch_and_decode_and_inserts_a_bubble(self):
        control = load_use_hazard_control(5)
        assert control.stall_fetch is True
        assert control.stall_decode is True
        assert control.insert_bubble is True
        assert control.flush_if_id is False
        assert control.flush_id_ex is False
        assert control.reason == "load-use dependency on x5"


class TestBranchHazardControl:
    def test_not_taken_produces_an_inert_control(self):
        control = branch_hazard_control(False)
        assert control == HazardControl()

    def test_taken_flushes_if_id_and_id_ex_but_does_not_stall(self):
        control = branch_hazard_control(True)
        assert control.flush_if_id is True
        assert control.flush_id_ex is True
        assert control.stall_fetch is False
        assert control.stall_decode is False
        assert control.insert_bubble is False
