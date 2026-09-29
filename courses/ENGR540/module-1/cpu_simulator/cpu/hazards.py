"""The hazard unit. Centralizes the pipeline control decisions that
aren't forwarding -- everything forwarding *can't* fix by itself --
so `pipelined_cpu.py`'s `step()` calls this instead of scattering
stall/bubble/flush logic across itself. Two kinds of hazard, two
detector functions, one shared result shape:

    load-use hazard    a load's value doesn't exist yet when the very
                        next instruction needs it in EX -- forwarding
                        alone can bridge every other RAW dependency
                        (see forwarding.py), but not this one, because
                        there's nothing to forward *from* yet. Needs a
                        one-cycle stall: hold `pc` and `IF_ID`, and
                        insert a bubble where the dependent instruction
                        would have entered `ID_EX`. By the time it
                        actually reaches EX, one cycle later, the load
                        has reached MEM_WB and ordinary MEM_WB
                        forwarding supplies the value -- the stall's
                        only job is to buy that one cycle, not to
                        deliver the value itself.

    control hazard      a taken branch, resolved in EX under static
                        not-taken prediction, means the two
                        instructions fetch already assumed would follow
                        straight-line were wrong. Both get flushed
                        (`IF_ID` and `ID_EX`, the two stages that ran
                        on the wrong-path assumption before EX knew
                        better) and `pc` redirects to the branch's
                        target.

HazardControl is the same small result shape either detector returns --
booleans a caller applies, never reasons about *why* beyond what's
already decided. `reason` exists for the trace, not for control flow:
nothing branches on its text.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    # See core/adder.py's module docstring for why this is needed to run
    # this file directly (e.g. VS Code's Run button) rather than via
    # `python -m`. Three .parent calls: hazards.py -> cpu/ ->
    # cpu_simulator/ -> module-1/.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.cpu.pipeline import ID_EX


@dataclass
class HazardControl:
    stall_fetch: bool = False
    stall_decode: bool = False
    insert_bubble: bool = False
    flush_if_id: bool = False
    flush_id_ex: bool = False
    reason: str | None = None


def detect_load_use_hazard(producer: ID_EX, consumer_rs1: int, consumer_rs2: int) -> int | None:
    """`producer` is whatever currently sits in `ID_EX` -- one stage
    ahead of the instruction now entering decode, i.e. about to enter
    EX this same cycle. Returns the conflicting register number if
    `producer` is a load whose destination the newly-decoded
    instruction reads, else None.

    x0 is excluded via `producer.rd == 0`, the same reasoning
    `forwarding.py`'s `rs == 0` check uses: nothing legitimately
    produces a value for x0, and a consumer that happens to read x0
    never actually needs `producer`'s result regardless of what `rd`
    says.
    """
    if not producer.valid or producer.control is None or not producer.control.mem_read:
        return None
    if producer.rd == 0:
        return None
    if producer.rd == consumer_rs1 or producer.rd == consumer_rs2:
        return producer.rd
    return None


def load_use_hazard_control(conflict_register: int | None) -> HazardControl:
    """The response to `detect_load_use_hazard`'s finding: freeze `pc`
    and `IF_ID` for one cycle, and let a bubble enter `ID_EX` in place
    of the dependent instruction. Nothing here touches `EX_MEM`/
    `MEM_WB` -- the load itself must keep moving forward; only the
    instructions behind it wait.
    """
    if conflict_register is None:
        return HazardControl()
    return HazardControl(
        stall_fetch=True, stall_decode=True, insert_bubble=True,
        reason=f"load-use dependency on x{conflict_register}",
    )


def branch_hazard_control(branch_taken: bool) -> HazardControl:
    """The response to a branch resolving taken in EX: flush the two
    stages that ran ahead of it on the wrong (not-taken) assumption.
    Doesn't carry the redirect target itself -- that's `EX_MEM.
    branch_target`, already sitting on the same pipeline register this
    decision was made from; `pipelined_cpu.py` applies it directly
    rather than round-tripping it through here.
    """
    if not branch_taken:
        return HazardControl()
    return HazardControl(flush_if_id=True, flush_id_ex=True, reason="branch taken")
