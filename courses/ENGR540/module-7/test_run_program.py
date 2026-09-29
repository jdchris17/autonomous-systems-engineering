import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_program import run


class TestRealProgram:
    def test_sum_5_to_1_equals_15(self):
        cpu = run(trace=False)
        assert cpu.data_memory.read_word(0) == 15

    def test_exercises_every_required_capability(self):
        cpu = run(trace=False)
        # immediate arithmetic + register arithmetic:
        assert cpu.registers.read(1) == 0   # counted down to 0
        assert cpu.registers.read(2) == 15  # accumulated sum
        # backward branch + PC changes: pc ended up past the loop and
        # the store, not stuck inside it or off the end of the program.
        assert cpu.pc.read() == 0x18
        # memory store:
        assert cpu.data_memory.read_word(0) == 15

    def test_takes_the_expected_number_of_cycles(self):
        cpu = run(trace=False)
        assert cpu.clock.cycle == 18  # 2 setup + 5 * 3 loop + 1 store
