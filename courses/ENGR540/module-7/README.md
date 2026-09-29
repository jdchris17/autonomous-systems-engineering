# Module 7 — The control unit, datapath, and execution trace

Two files live in this folder: [`run_program.py`](run_program.py) and
[`multicycle.py`](multicycle.py), plus their `test_*.py`. The library
code they're built on — `control.py`, `datapath.py`, `cpu.py`,
`trace.py` — was **built here originally, then moved** into
[`module-1/cpu_simulator/cpu/`](../module-1/README.md), once the
course's own target tree showed those files belonging alongside
`register_file.py`/`memory.py`/`pc.py` as reusable library components,
not one-off module-7 code. See module-1's README for the full story
(including the `cpu.py`/`pseudo_cpu.py` naming collision that move
required resolving) and for those four files' own documentation.

`decoder.py` (Module 6) turns a raw word into meaningful fields.
Module 7 is what actually *does* something with them: `control.py`
converts those fields into explicit control signals, `datapath.py`
breaks execution into named stages, `cpu.py` coordinates the two into a
real, running RV32I CPU, and `trace.py` renders exactly what happened
each cycle — the first CPU in this whole build that executes
actually-decoded instructions rather than either hand-written Python
calls (Module 4) or Module 5's pseudo-instruction tuples.

## What's in `cpu_simulator/cpu/` (summary — see module-1's README for the full detail)

- **`control.py`** — `generate_control(instruction) -> ControlSignals`, a structured dataclass (`reg_write`, `mem_read`, `mem_write`, `alu_src`, `alu_op`, `result_src`, `branch`, `jump`). Split into main control (opcode only) and ALU control (funct3/funct7 only), mirroring real processor design. Verified against both spec worked examples exactly. ALU control's lookup tables are `decoder.py`'s own tables, reused rather than redeclared — one naming bug (`SLTIU`'s irregular `SLT`+`I`+`U` spelling breaking a suffix-stripping shortcut) was caught by the test suite and fixed with an explicit lookup.
- **`datapath.py`** — `Datapath`: `fetch()`/`decode()`/`execute()`/`memory_access()`/`writeback()` as five real methods, per section 28's explicit ask — the same organization a pipelined design needs later, since a pipeline is just these five stages running concurrently instead of in sequence. Resolves two gaps `control.py`'s own docstring had left open (`AUIPC`'s `pc`-as-operand, `JAL`'s separately-computed jump target).
- **`cpu.py`** — `CPU`, the coordinator. `step()` is one small function calling five separate concerns, per section 29. Reaches module-6's exact architectural target through *actual* decoding and execution, not Module 5's pseudo-op shortcut, and correctly runs backward-branch loops (both taken and not-taken paths exercised).
- **`trace.py`** — `StepTrace` plus `format_verbose`/`format_compact`. "This should become one of the best components in the project" (section 31) — verified to match both spec worked examples character for character. Every `CPU.step()` call appends to `self.trace_history` automatically.

## [`run_program.py`](run_program.py) — section 34's milestone

Section 34's exact summation program (a backward-branch loop computing
`5 + 4 + 3 + 2 + 1 = 15`, storing the result to `Memory[0]`), run
through the full pipeline and printed as a compact trace — 18 lines,
one per *dynamic* instruction (2 setup + the 3-instruction loop body
running 5 times + 1 store), not the 7 assembled words. Exercises every
capability the spec calls out at once: immediate arithmetic, register
arithmetic, a backward branch, PC changes, and a memory store, all from
encoded RV32I machine words. `run(cpu=None, trace=True)` is importable
directly (used by `test_multicycle.py` for a same-program,
cross-model correctness check).

**Section 35 needed no new file.** "Treat `step()` as execute one
complete instruction, but internally still expose fetch → decode →
execute → memory → writeback" is exactly what `CPU.step()` already
does — one `clock.tick()` per instruction, five separate `datapath.*`
method calls underneath. Nothing to add.

## [`multicycle.py`](multicycle.py) — section 36's optional staged mode

`MulticycleCPU`: the same `Datapath` stage methods and `control.py`
control unit as `CPU`, but `tick()` performs exactly *one* stage and
advances the clock once per **stage**, not once per instruction —
`run_instruction()` takes five `tick()` calls (five clock cycles) to do
what `CPU.step()` does in one. `self.stage` cycles
`FETCH → DECODE → EXECUTE → MEMORY → WRITEBACK → FETCH → ...`, driven by
a small pure transition function (`_next_stage`), the same *shape*
`module-1/cpu_simulator/cpu/fsm.py`'s `FSM`/`transition()` established
— current state stored explicitly, a pure function computing the next
state, the state only actually advancing on a tick. Deliberately not a
literal reuse of that specific `FSM` class, though: its states
(`IDLE`/`LOAD`/`PROCESS`/`DONE`) and inputs are a different, unrelated
demo machine, and forcing this 5-stage cycle through that API would
mean bending it to fit rather than reusing anything real. What carries
over is the *pattern*, applied to a real control unit instead of a toy
example — section 36's own "ties together Module 4 and Module 7."

Values a real multicycle CPU would latch between stages (the fetched
word, the decoded instruction, operand values, the ALU result) are
plain instance attributes, written by one stage and read by a later one
— the same role a pipeline register plays once these stages run
concurrently instead of in sequence. Every instruction takes exactly
five ticks, even when a stage has nothing to do (e.g. `MEMORY` for an
`ADD`) — a real multicycle design varies cycle count by instruction type
(R-type skips `MEMORY` entirely); that's cycle-accurate performance
modeling, which section 36 asks for the *structure* of, not the timing
of, so it's a deliberate simplification, not an oversight.

**Cross-validated against `CPU`, not just tested in isolation**: running
section 34's exact program through both `CPU` (1 tick/instruction) and
`MulticycleCPU` (5 ticks/instruction) reaches the identical final
architectural state — same registers, same `Memory[0]`, same final
`pc` — while `MulticycleCPU.clock.cycle` ends up exactly 5× `CPU`'s.
Two completely different clock disciplines, one correct answer.

## Run it

```
python run_program.py
python multicycle.py
```

(`control.py`/`datapath.py`/`cpu.py`/`trace.py` are run from
`module-1/cpu_simulator/cpu/` now — see that module's README.)
