"""FinalState(CPU) == FinalState(PipelinedCPU) -- run visibly, not just
asserted in a test. Runs the same programs through both the
single-cycle `CPU` (module-1's reference implementation, one
instruction per cycle, no forwarding or hazard detection needed because
nothing is ever in flight at once) and the five-stage `PipelinedCPU`
(five instructions in flight, correctness resting on `forwarding.py`
and, for the loop below, `hazards.py`'s branch-flush handling too), and
prints a register-by-register comparison. Two completely different
execution disciplines are expected to reach the identical architectural
state for any program -- pipelining is a performance technique, not a
different ISA, and that now holds for branching programs as well as
straight-line ones.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module-1"))

from cpu_simulator.cpu.cpu import CPU
from cpu_simulator.cpu.pipelined_cpu import PIPELINE_DEPTH, PipelinedCPU, load_program
from cpu_simulator.isa.assembler import assemble

MODULE_6_PROGRAM = """
addi x1, x0, 10
addi x2, x0, 7
add  x3, x1, x2
sub  x4, x3, x2
sw   x4, 0(x0)
"""

LOAD_FORWARDING_PROGRAM = """
addi x1, x0, 100
sw   x1, 0(x0)
lw   x2, 0(x0)
addi x3, x0, 1
add  x4, x2, x3
addi x4, x4, 5
sub  x5, x4, x2
"""

BRANCH_LOOP_PROGRAM = """
addi x1, x0, 5
addi x2, x0, 0
loop:
    add  x2, x2, x1
    addi x1, x1, -1
    bne  x1, x0, loop
sw   x2, 0(x0)
"""


def _load_single_cycle(cpu: CPU, program: str) -> int:
    words = assemble(program)
    for i, word in enumerate(words):
        cpu.instruction_memory.write_word(i * 4, word)
    return len(words)


def compare(program: str, registers_to_check: range | list[int], label: str) -> bool:
    single = CPU()
    n_single = _load_single_cycle(single, program)
    single.run(n_single)

    pipelined = PipelinedCPU()
    n_pipelined = load_program(pipelined, program)
    pipelined.run(n_pipelined + (PIPELINE_DEPTH - 1))

    print(f"--- {label} ---")
    all_match = True
    for reg in registers_to_check:
        single_value = single.registers.read(reg)
        pipelined_value = pipelined.registers.read(reg)
        match = single_value == pipelined_value
        all_match = all_match and match
        print(f"x{reg}: single={single_value} pipelined={pipelined_value} match={match}")
    print(f"clock cycles: single={single.clock.cycle} pipelined={pipelined.clock.cycle}")
    print()
    return all_match


def compare_branch_loop() -> bool:
    """The backward-branch summation loop (5+4+3+2+1), which neither
    `compare()`'s word-count-equals-step-count assumption nor its
    default drain padding fit: the pipelined model needs extra cycles
    for every taken branch's flush penalty, and enough padding for `pc`
    to wander before it settles into draining.
    """
    single = CPU()
    _load_single_cycle(single, BRANCH_LOOP_PROGRAM)
    # 2 setup instructions, then the 3-instruction loop body runs 5 times.
    single.run(2 + 5 * 3 + 1)

    pipelined = PipelinedCPU()
    load_program(pipelined, BRANCH_LOOP_PROGRAM, pad_instructions=40)
    pipelined.run(40)

    print("--- backward-branch loop (5+4+3+2+1) ---")
    all_match = True
    for reg in (1, 2):
        single_value = single.registers.read(reg)
        pipelined_value = pipelined.registers.read(reg)
        match = single_value == pipelined_value
        all_match = all_match and match
        print(f"x{reg}: single={single_value} pipelined={pipelined_value} match={match}")
    mem_match = single.data_memory.read_word(0) == pipelined.data_memory.read_word(0)
    all_match = all_match and mem_match
    print(f"Memory[0]: single={single.data_memory.read_word(0)} "
          f"pipelined={pipelined.data_memory.read_word(0)} match={mem_match}")
    flushes = sum(1 for t in pipelined.trace_history if t.flush)
    print(f"clock cycles: single={single.clock.cycle} pipelined={pipelined.clock.cycle} "
          f"(flushes: {flushes})")
    print()
    return all_match


def main():
    print("=" * 60)
    print("MODULE 8: cross-validation -- CPU vs PipelinedCPU")
    print("=" * 60)
    results = [
        compare(MODULE_6_PROGRAM, range(1, 5), "module-6 program"),
        compare(LOAD_FORWARDING_PROGRAM, range(1, 6), "load-forwarding program"),
        compare_branch_loop(),
    ]
    assert all(results)
    print("All three programs reached identical final architectural state")
    print("on both models -- forwarding resolves the RAW hazards, and")
    print("hazards.py's stall/flush logic correctly handles the load-use")
    print("and control hazards straight-line programs never exercise.")
    print("=" * 60)


if __name__ == "__main__":
    main()
