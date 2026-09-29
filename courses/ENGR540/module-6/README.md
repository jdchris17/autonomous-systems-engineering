# Module 6 — A good first actual RV32I program

The section-49 program, assembled into real machine words, stored in
memory, and decoded back to fields:

```
addi x1, x0, 10
addi x2, x0, 7
add  x3, x1, x2
sub  x4, x3, x2
sw   x4, 0(x0)
```

## [`first_program.py`](first_program.py)

```
python first_program.py
```

- **Assembler encodes it**: `assemble_and_store()` runs the program through `module-1/cpu_simulator/isa/assembler.py` and writes each word into a `Memory` instance.
- **Memory stores it**: the words are read back with `read_word()`, not kept around as the Python list `assemble()` returned — they have to survive an actual round trip through storage.
- **Decoder recovers the fields**: `fetch_and_decode()` decodes each word back into an `Instruction`, printed with its mnemonic and operands.

## What this file does *not* do

Decoded fields aren't executed here. When this file was written, driving
a datapath from an `Instruction` the way a real control unit would was
explicitly Module 7's job ("By Module 7, the control unit will drive the
actual datapath from those decoded fields") — that component didn't
exist yet. **Module 7 exists now** (`cpu_simulator/cpu/control.py`,
`datapath.py`, `cpu.py`) and can run this exact program from its decoded
fields for real — see [`module-7/run_program.py`](../module-7/README.md).
This file is kept exactly as originally written anyway, as the
historical record of what was and wasn't possible at this point in the
course.

What ran instead, from Module 5, is `PseudoOpCPU`'s pseudo-instruction
execution path (`run()`), in
[`module-1/cpu_simulator/cpu/pseudo_cpu.py`](../module-1/cpu_simulator/cpu/pseudo_cpu.py)
(that file used to be named `cpu.py`, renamed once Module 7 needed that
name for its own real CPU class — `PseudoOpCPU`'s behavior is
unaffected). The five real instructions above and
`PROGRAM_PSEUDO_OPS`'s five tuples are the same program in two
representations — one real RV32I machine code, one the artificial
stand-in that predates a decoder-driven execution engine:

| real RV32I | pseudo-instruction tuple |
|---|---|
| `addi x1, x0, 10` | `("LOAD_IMM", 1, 10)` |
| `addi x2, x0, 7` | `("LOAD_IMM", 2, 7)` |
| `add x3, x1, x2` | `("ADD", 3, 1, 2)` |
| `sub x4, x3, x2` | `("SUB", 4, 3, 2)` |
| `sw x4, 0(x0)` | `("STORE", 4, 0, 0)` |

`addi rd, x0, imm` and `LOAD_IMM` are the same operation for the same
reason `PseudoOpCPU`'s `step_load_immediate` docstring already gives:
real RISC-V's own `li` pseudo-instruction is literally `addi rd, x0,
imm`. Running the tuple form is how this file reached the architectural
result without building a control unit early — `x1=10, x2=7, x3=17,
x4=10, Memory[0]=10`, matching the spec exactly.

## Tests

[`test_first_program.py`](test_first_program.py) follows section 51's
principle directly: exact known encodings (each of the five words
hand-verified against the encoder before being written down, not just
trusted), exact decoded fields for each one, and `decode(encode(x)) ==
x` round-trip symmetry — plus the literal `add x5, x6, x7 ->
0x007302B3` example from the spec, independent of this program. 8 tests,
all passing.
