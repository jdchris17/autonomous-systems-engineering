import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cross_validate import (
    LOAD_FORWARDING_PROGRAM,
    MODULE_6_PROGRAM,
    compare,
    compare_branch_loop,
)


class TestCrossValidation:
    def test_module_6_program_matches(self):
        assert compare(MODULE_6_PROGRAM, range(1, 5), "module-6 program") is True

    def test_load_forwarding_program_matches(self):
        assert compare(LOAD_FORWARDING_PROGRAM, range(1, 6), "load-forwarding program") is True

    def test_branch_loop_matches(self):
        assert compare_branch_loop() is True
