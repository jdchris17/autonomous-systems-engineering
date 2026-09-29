"""A structured representation of a decoded instruction -- the output
type decoder.py produces and encoder.py will eventually consume in
reverse. A raw 32-bit word carries meaning entirely by field position;
Instruction is that same meaning made explicit and named, so nothing
downstream (execution, printing, testing) has to re-derive "which bits
mean rs2 here" from scratch.

Six formats share this one dataclass rather than each getting its own
type, because which fields are populated is exactly the information
`format` already carries -- an R-type instruction has rd/rs1/rs2/funct3/
funct7 and no immediate; a U-type has rd/immediate and nothing else.
Rather than model that with six dataclasses (or a class hierarchy),
every field that isn't always present defaults to None, and it's
decoder.py's job to only fill in the ones a given format actually has.
"""

from __future__ import annotations

from dataclasses import dataclass

FORMATS = ("R", "I", "S", "B", "U", "J")


@dataclass
class Instruction:
    raw: int
    mnemonic: str
    opcode: int
    format: str
    rd: int | None = None
    rs1: int | None = None
    rs2: int | None = None
    funct3: int | None = None
    funct7: int | None = None
    immediate: int | None = None
