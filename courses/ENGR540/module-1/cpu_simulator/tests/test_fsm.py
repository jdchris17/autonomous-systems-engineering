import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.cpu.fsm import FSM, transition


class TestTransitionTruthTable:
    def test_idle_start_to_load(self):
        assert transition("IDLE", "start") == "LOAD"

    def test_load_loaded_to_process(self):
        assert transition("LOAD", "loaded") == "PROCESS"

    def test_process_complete_to_done(self):
        assert transition("PROCESS", "complete") == "DONE"

    def test_done_returns_to_idle_on_any_input(self):
        assert transition("DONE", None) == "IDLE"
        assert transition("DONE", "start") == "IDLE"
        assert transition("DONE", "loaded") == "IDLE"

    def test_reset_overrides_from_any_state(self):
        for state in ["IDLE", "LOAD", "PROCESS", "DONE"]:
            assert transition(state, "reset") == "IDLE"

    def test_irrelevant_input_holds_current_state(self):
        assert transition("IDLE", "complete") == "IDLE"
        assert transition("LOAD", "start") == "LOAD"
        assert transition("PROCESS", "loaded") == "PROCESS"

    def test_no_input_holds_current_state(self):
        assert transition("IDLE", None) == "IDLE"
        assert transition("LOAD", None) == "LOAD"

    def test_unknown_state_rejected(self):
        with pytest.raises(ValueError):
            transition("HALTED", "start")

    def test_unknown_input_rejected(self):
        with pytest.raises(ValueError):
            transition("IDLE", "explode")


class TestFSMTwoPhaseCommit:
    def test_starts_in_idle_by_default(self):
        assert FSM().state == "IDLE"

    def test_custom_initial_state(self):
        assert FSM(initial_state="PROCESS").state == "PROCESS"

    def test_invalid_initial_state_rejected(self):
        with pytest.raises(ValueError):
            FSM(initial_state="HALTED")

    def test_compute_next_does_not_change_state_until_commit(self):
        machine = FSM()
        machine.compute_next("start")
        assert machine.state == "IDLE"  # still the old state
        machine.commit()
        assert machine.state == "LOAD"

    def test_commit_with_nothing_staged_holds_state(self):
        machine = FSM()
        machine.commit()
        assert machine.state == "IDLE"

    def test_step_combines_both_phases(self):
        machine = FSM()
        assert machine.step("start") == "LOAD"


class TestFullCycleWalkthrough:
    def test_idle_load_process_done_idle(self):
        machine = FSM()
        assert machine.step("start") == "LOAD"
        assert machine.step("loaded") == "PROCESS"
        assert machine.step("complete") == "DONE"
        assert machine.step(None) == "IDLE"

    def test_reset_mid_cycle_returns_to_idle(self):
        machine = FSM()
        machine.step("start")
        machine.step("loaded")
        assert machine.state == "PROCESS"
        assert machine.step("reset") == "IDLE"

    def test_irrelevant_inputs_do_not_advance_the_cycle(self):
        machine = FSM()
        machine.step("complete")  # meaningless while IDLE
        machine.step("loaded")    # meaningless while IDLE
        assert machine.state == "IDLE"
        assert machine.step("start") == "LOAD"
