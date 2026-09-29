# Module 5 — Pseudo-instructions and the datapath

How does a CPU move data through its datapath once it already knows
what to do? Deliberately not asking the other half of that question yet
(how "what to do" gets encoded into 32 bits) — that's Module 6's job,
kept on the other side of a clean abstraction boundary.

## The pseudo-instruction format

Lives in [`module-1/cpu_simulator/cpu/pseudo_cpu.py`](../module-1/cpu_simulator/cpu/pseudo_cpu.py)'s
`PseudoOpCPU` class, not this folder — it's a real extension of that
class, not a one-off script's private format. (That file used to be
named `cpu.py`, renamed once `module-1` grew a *real*,
decoded-instruction-driven `CPU` class in Module 7 that needed the name
`cpu.py`/`CPU` for itself — `PseudoOpCPU`'s own behavior here is
completely unchanged by that rename.) Plain tuples:

| Tuple | Meaning |
|---|---|
| `("ADD", rd, rs1, rs2)` | `R[rd] = R[rs1] + R[rs2]` |
| `("SUB", rd, rs1, rs2)` | `R[rd] = R[rs1] - R[rs2]` |
| *(any other `alu.py` op)* | `AND`/`OR`/`XOR`/`SLL`/`SRL`/`SRA`/`SLT`/`SLTU` all work the same 3-operand way — `execute()` dispatches generically against `alu.OPERATIONS` rather than hand-listing each one |
| `("LOAD_IMM", rd, value)` | `R[rd] = value` |
| `("LOAD", rd, base, offset)` | `R[rd] = MEM[R[base] + offset]` |
| `("STORE", rs, base, offset)` | `MEM[R[base] + offset] = R[rs]` |

`execute(instruction)` dispatches one tuple; `run(program)` runs a
whole list in order.

## Two judgment calls worth knowing about

**`LOAD_IMM` goes through the ALU, not a direct register write.**
`R[rd] = 0 + value`, via `alu(0, value, "ADD")` — the same trick real
RISC-V uses (its `li` pseudo-instruction is literally `addi rd, x0,
imm`). There's no dedicated load-immediate circuit in real hardware,
just the adder fed a hardwired zero, so there isn't one here either.

**Absolute addressing isn't its own addressing mode.** Your example
program's last line, `STORE R4, [100]`, has no base register at all —
but `PseudoOpCPU`'s `STORE` only knows base+offset. The tuple form is
`("STORE", 4, 0, 100)`: base = `x0`, which is hardwired to zero, so
`MEM[R[0] + 100]` is exactly `MEM[100]`. This is the same trick real
RISC-V assemblers use to synthesize "absolute" addressing — there's no
separate hardware path for it either.

## [`tiny_program.py`](tiny_program.py)

Your exact example program from section 37:

```
LOAD_IMM R1, 10
LOAD_IMM R2, 7
ADD      R3, R1, R2
SUB      R4, R3, R2
STORE    R4, [100]
```

Run it:

```
python tiny_program.py
```

Prints the program, then `R1`–`R4`, `memory[100]`, and `PC` — exactly
the inspection list from the spec. Expected result: `R1=10, R2=7,
R3=17, R4=10, memory[100]=10, PC=20` (five instructions × 4 bytes).

`instruction_memory` is untouched throughout — this program is executed
directly as pseudo-instructions, never encoded into machine words.
That's the whole point of the abstraction boundary: `run()` exercises
the complete datapath (pc, registers, alu, memory) with zero encoding
work done, because encoding doesn't exist yet.
