"""Explicit pipeline-register structures: the values that survive
between adjacent pipeline stages when those stages run concurrently on
different instructions instead of sequentially on one, the way
`datapath.py`'s stage methods and `cpu.py`'s `CPU.step()` currently run
them.

Four registers, one per boundary between `Datapath`'s five stages:

    fetch -> IF_ID -> decode -> ID_EX -> execute -> EX_MEM -> memory -> MEM_WB -> writeback

Each dataclass is deliberately built from plain fields, not a nested
`Instruction` object -- real pipeline registers are exactly this: a
fixed set of wires latched on a clock edge, not a Python object
reference. Which specific fields each one carries follows directly from
this package's own datapath: `Datapath.execute()` needs `rs1_value`/
`rs2_value` (not raw register *numbers* -- those were already resolved
to values back in `decode`, which is the whole point of latching values
forward instead of re-reading the register file from a later stage);
`Datapath.memory_access()` needs the address `execute()` computed plus
`rs2_value` again (the value a store writes); `writeback()` needs
`rd` and whichever of the memory result / `pc + 4` `control.result_src`
selects. `ControlSignals` itself rides along unresolved from `ID_EX`
onward, since which of those needs still apply to a given instruction
is exactly what `control.py` already decided once, back in `decode`.

Every register also carries `mnemonic`, `pc`, and (from `ID_EX` onward)
`raw`, none of which anything in `Datapath`'s own stage signatures
strictly requires -- they're carried purely so a pipeline trace
(`trace.py`'s `format_pipeline`, the pipelined equivalent of what that
file already does for the single-cycle model) can say which
instruction, at which address, is sitting in which stage on a given
cycle. `raw` specifically exists so a trace can render a stage's full
assembly text (`decoder.decode(raw)` recovers rs1/rs2/immediate on
demand) even once a stage's own dataclass -- `EX_MEM`/`MEM_WB` -- has
stopped carrying those fields individually, having already resolved
them into `alu_result`/`rs2_value` by that point. It is otherwise
inert: nothing in `_execute`/`_memory`/`_writeback` reads it.

Every register has a `valid` bit, defaulting to `False`. A bubble --
an empty pipeline slot, whether from startup, a flush, or a stall -- is
simply an instance with `valid=False`; every other field is free to
hold whatever its dataclass default happens to be, because nothing
downstream is allowed to look at them until `valid` says to. This is
deliberately not modeled as a real "NOP" instruction manufactured and
pushed through the pipeline -- a fabricated NOP would need a real
opcode, real (harmless) control signals, and would still occupy
register-file bandwidth and hazard-detection logic's attention as if it
were a genuine instruction, when the entire point is that it isn't one.
`valid=False` needs no encoding, no control-signal generation, and every
hazard/forwarding check downstream (not yet built) gets to start with
"if not stage.valid: skip" before it has to reason about anything else
in that register at all.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    # See core/adder.py's module docstring for why this is needed to run
    # this file directly (e.g. VS Code's Run button) rather than via
    # `python -m`. Three .parent calls: pipeline.py -> cpu/ ->
    # cpu_simulator/ -> module-1/.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.cpu.control import ControlSignals


@dataclass
class IF_ID:
    """What fetch() hands to decode(): the raw word and where it came
    from. Nothing decoded yet -- that's decode()'s own job, next.
    """
    valid: bool = False
    pc: int = 0
    instruction: int = 0  # raw 32-bit fetched word


@dataclass
class ID_EX:
    """What decode() (plus control-signal generation) hands to
    execute(). rs1/rs2 are the register *numbers* -- kept alongside
    their already-read values because hazard detection (not yet built)
    needs to compare *numbers* against other stages' rd, while
    execute() itself only ever needs the values.
    """
    valid: bool = False
    pc: int = 0
    raw: int = 0
    mnemonic: str = ""
    rs1: int = 0
    rs2: int = 0
    rs1_value: int = 0
    rs2_value: int = 0
    rd: int = 0
    immediate: int = 0
    control: ControlSignals | None = None


@dataclass
class EX_MEM:
    """What execute() hands to memory_access(): the ALU's result (an
    address, for a load/store; the actual result, for everything else),
    the value a store would write (rs2_value, latched forward rather
    than re-read from the register file at this later stage), and
    whether a branch now in this stage was taken -- plus, when it was,
    `branch_target` (`pc + immediate`, the same pc-relative computation
    `cpu.py`'s `_update_pc` already uses for the single-cycle model) for
    whichever cycle actually redirects `pc` once it's committed.
    """
    valid: bool = False
    pc: int = 0
    raw: int = 0
    mnemonic: str = ""
    alu_result: int = 0
    rs2_value: int = 0
    rd: int = 0
    branch_taken: bool = False
    branch_target: int = 0
    pc_plus_4: int = 0
    control: ControlSignals | None = None


@dataclass
class MEM_WB:
    """What memory_access() hands to writeback(): one already-resolved
    value (the loaded word, or execute()'s own alu_result passed
    through unchanged for a non-memory instruction -- exactly what
    `Datapath.memory_access()` already returns today), plus `pc_plus_4`
    for the one case (`JAL`/`JALR`) where `control.result_src` picks
    that instead.
    """
    valid: bool = False
    pc: int = 0
    raw: int = 0
    mnemonic: str = ""
    memory_result: int = 0
    rd: int = 0
    pc_plus_4: int = 0
    control: ControlSignals | None = None
