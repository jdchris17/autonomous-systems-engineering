# Module 8 — Pipelining, forwarding, hazards, branch handling, and performance

The library code this module is built on lives in
[`module-1/cpu_simulator/cpu/`](../module-1/README.md), following the
project's standing rule: `cpu_simulator/` is the base library and never
depends on any `module-N/` folder, while `module-N/` folders are
consumers of it. `forwarding.py`, `pipelined_cpu.py`, `hazards.py`,
`performance.py`, and the pipeline-trace additions to `trace.py` are
all genuinely CPU-building library code — they live in module-1. This
folder holds only the *outside program builds*: seven runnable demos
plus their tests.

## What's new in `cpu_simulator/cpu/` (summary — see module-1's README for the full detail)

- **`pipeline.py`** — `IF_ID`/`ID_EX`/`EX_MEM`/`MEM_WB`, plain dataclasses with `valid: bool = False` as the bubble mechanism. `EX_MEM` now also carries `branch_target`; `ID_EX`/`EX_MEM`/`MEM_WB` each carry `raw` (the original fetched word) purely so a trace can reconstruct full assembly text at any stage.
- **`forwarding.py`** — `determine_forwarding()` (the decision) and `select()` (the mux). Priority is EX_MEM before MEM_WB — the more recently produced value wins, per the spec's own "priority matters" example. Doesn't solve the load-use hazard (the consumer directly behind a load) — that's `hazards.py`'s job.
- **`hazards.py`** — the hazard unit, new this round. `detect_load_use_hazard()` + `load_use_hazard_control()`: a load immediately followed by its consumer stalls fetch/decode for one cycle and inserts a bubble into `ID_EX`, buying forwarding the cycle it needs to catch the value from MEM_WB. `branch_hazard_control()`: a branch resolved taken in EX (static not-taken prediction) flushes `IF_ID` and `ID_EX`. The two hazard kinds can never collide in the same cycle — a single instruction can't be both a load and a branch.
- **`pipelined_cpu.py`** — `PipelinedCPU` now wires `hazards.py` into `step()`: stalls hold `pc`/`IF_ID` and insert a bubble; a taken branch flushes `IF_ID`/`ID_EX` and redirects `pc` to `EX_MEM.branch_target`. Also builds a `PipelineTrace` every cycle. **Remaining gap**: `JAL`/`JALR` still don't redirect `pc` — only conditional branches do.
- **`trace.py`** — `PipelineTrace`/`format_pipeline()`, new: one line per stage (`IF`/`ID`/`EX`/`MEM`/`WB`, newest to oldest), this cycle's forwarding decisions, and whichever hazard (if any) is active — matching section 8's own worked trace examples, including the stall and branch-flush blocks.
- **`performance.py`** — `PerformanceCounters` plus `cpi()`/`ipc()`. Both `CPU` and `PipelinedCPU` own a `self.performance`, updated every `step()`. `CPU`'s stays at `cpi() == 1.0` always — a single-cycle model structurally can't stall, flush, or forward. (Moved to `cpu_simulator/cpu/analysis/performance.py` and substantially expanded in Module 10 — see [`module-1/README.md`](../module-1/README.md) for the current field list.)

## [`forwarding_demo.py`](forwarding_demo.py) — the "priority matters" worked example

The spec's exact three-instruction example (`add x5,x1,x2` /
`addi x5,x5,1` / `sub x6,x5,x3`, with x1=10, x2=20, x3=1), run on a real
`PipelinedCPU` with a full `format_pipeline()` trace each cycle.
Confirms `x5 == 31` and `x6 == 30` — the second result only comes out
right if forwarding picks the *newer* value still in EX_MEM (31) over
the *older* one already sitting in MEM_WB (30, which would wrongly
yield `x6 = 29`). This file used to build its own one-off stage-summary
printer before `trace.py`'s `format_pipeline()` existed; now that a
real one does, it uses that instead.

## [`hazards_demo.py`](hazards_demo.py) — a load-use stall, and a branch flush

Two worked examples, each printed with a full `format_pipeline()`
trace:

- **Load-use stall**: `lw x5, 0(x0)` immediately followed by
  `add x6, x5, x1`. Forwarding alone can't bridge this — the loaded
  value doesn't exist until MEMORY runs — so `hazards.py` stalls fetch
  and decode for exactly one cycle, inserting a bubble into `ID_EX`;
  ordinary MEM_WB forwarding catches the value one cycle later.
  Confirms `x5 == 100`, `x6 == 200`.
- **Branch flush**: `beq x1, x2, skip` (taken) followed by two
  instructions that must never retire (`addi x10, x0, 999` /
  `addi x11, x0, 999`), then the `skip:` target. Confirms both wrong-path
  registers stay `0` and `x3 == 42`.

## [`cross_validate.py`](cross_validate.py) — CPU vs PipelinedCPU, side by side

`FinalState(CPU) == FinalState(PipelinedCPU)` — run visibly instead of
just asserted. Three programs now: module-6's original 5-instruction
program, a 7-instruction load-forwarding-heavy program, and — new this
round, now that branches actually redirect `pc` and flush — a real
backward-branch summation loop (`5+4+3+2+1`). `compare_branch_loop()`
is its own function rather than going through `compare()`, since a
loop breaks that function's two simplifying assumptions (assembled
word count equals dynamic step count; default drain padding is
enough) — every taken iteration of the branch costs the pipelined
model two flushed wrong-path fetches beyond what a straight-line
instruction count would predict. All three programs reach identical
final architectural state on both models, despite completely different
execution disciplines and total cycle counts.

## [`benchmark.py`](benchmark.py) — architectural match, different speed

Section 8's first performance-comparison program: a backward-branch
loop summing `10 + 9 + ... + 1 = 55`, deliberately dependency-heavy
(every loop iteration's `add`/`addi`/`bne` each depend on the
instruction right before it). `run_benchmark()` runs it on both `CPU`
and `PipelinedCPU`, confirms `Memory[0] == 55` on both, and reports each
model's `PerformanceCounters` plus `CPI`/`IPC`. `CPU` lands at exactly
`CPI = 1.00`; `PipelinedCPU` comes out well above that (`~1.67` for this
program), driven by the RAW-dependency chain plus the fixed fill/drain
and branch-flush cost — architectural correctness and performance are
different questions, and this is where the second one gets measured.

`run_benchmark()` stops each model by counting *retired* instructions
rather than pre-computing a cycle count, specifically so it can be
reused for the independent-operations and dependency-heavy programs
section 8 says are coming next: a looping program's dynamic instruction
count isn't its assembled word count, and the pipelined model's actual
cycle count isn't known until stalls/flushes are accounted for either.
Counting retirements sidesteps both, and stops before any
`load_program()` padding NOP can retire and quietly inflate the count.

## Three regression tests

Purpose-built programs, each pinned down as its own file so a future
regression has one exact, minimal case to fail against instead of a
vague "the pipeline seems wrong somewhere."

### [`load_use_test.py`](load_use_test.py) — distinguishing stall-needed from forward-only

`lw x5,0(x1)` / `add x6,x5,x2` / `sub x7,x6,x3` — two back-to-back RAW
dependencies in one program that need *different* resolutions. The
first (`lw` → `add`) can only be fixed by a stall; the second
(`add` → `sub`) should be resolved by ordinary forwarding, no stall
needed. Asserts `x5=100, x6=110, x7=105` **and** `stalls == 1` exactly
— not 0 (which would mean the load-use hazard was missed and the wrong
value used) and not 2 (which would mean the hazard unit stalls
indiscriminately instead of actually distinguishing the two cases).

### [`forwarding_torture_test.py`](forwarding_torture_test.py) — forwarding under maximum pressure

Five `addi`s, each depending on the one immediately before it —
the worst case a RAW hazard can present (zero gap between producer and
consumer, four times in a row). `x1 == 5` is the easy check;
`stalls == 0` and `cycles == n + PIPELINE_DEPTH - 1` (the theoretical
fill+drain minimum, no stall overhead at all) is what actually proves
forwarding resolved every one of the four dependencies for free, rather
than a stall quietly picking up the slack underneath it.

### [`branch_flush_test.py`](branch_flush_test.py) — speculative work must not become architectural state

The exact program from the spec: `beq x1,x2,target` (taken), with two
wrong-path instructions (`addi x3,x0,99` / `addi x4,x0,99`) that must
never retire. Checks more than the final register read — it walks
`trace_history` and asserts neither wrong-path instruction's `WB` stage
is ever `valid` for `rd` 3 or 4, confirming the flush discarded them
before they could retire at all, not just that something later
happened to leave the registers looking right.

## Run it

```
python forwarding_demo.py
python hazards_demo.py
python cross_validate.py
python benchmark.py
python load_use_test.py
python forwarding_torture_test.py
python branch_flush_test.py
```
