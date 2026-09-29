"""One abstraction level above encoder.py: text in, machine words out.

    "add x5, x6, x7"  ->  0x007302B3

Not a full industrial assembler -- no directives, no sections, no
pseudo-instructions beyond what decoder.py/encoder.py already know as
real mnemonics. Just enough of a pipeline to turn a short RV32I program
into the words that would live in instruction memory:

    source line -> tokenize -> identify mnemonic -> parse operands
        -> select instruction format -> encode fields -> machine word

The "select instruction format" step doesn't invent new knowledge --
which shape of operands a mnemonic expects (rd, rs1, rs2 for R-type;
rd, offset(rs1) for a load; rs1, rs2, target for a branch; ...) is
read off decoder.py's own mnemonic tables, the third file in this
package to derive from that one source of truth rather than
re-declaring it (encoder.py's reverse-lookup tables were the second).

Labels and two-pass assembly exist for one reason: a branch's encoded
immediate is a PC-relative displacement, not an absolute address, and
you cannot compute "target address minus this instruction's address"
before you know where every label lands -- including ones defined
*after* the branch that uses them (a loop's exit label, typically).
Pass 1 walks the whole source just to learn every label's address;
pass 2 walks it again to actually encode, now able to resolve any
label reference (forward or backward) instead of guessing.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    # See core/adder.py's module docstring for why this is needed to run
    # this file directly (e.g. VS Code's Run button) rather than via
    # `python -m`. Three .parent calls: assembler.py -> isa/ ->
    # cpu_simulator/ -> module-1/.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.isa.decoder import (
    BRANCH_MNEMONICS,
    I_TYPE_ARITH_MNEMONICS,
    LOAD_MNEMONICS,
    R_TYPE_MNEMONICS,
    SHIFT_IMMEDIATE_MNEMONICS,
    STORE_MNEMONICS,
)
from cpu_simulator.isa.encoder import encode

INSTRUCTION_WIDTH = 4  # bytes -- every RV32I instruction, no exceptions

# Which mnemonics belong to which operand shape, read off decoder.py's
# own tables rather than re-declared.
_R_TYPE_SET = set(R_TYPE_MNEMONICS.values())
_SHIFT_IMMEDIATE_SET = set(SHIFT_IMMEDIATE_MNEMONICS.values())
_I_TYPE_ARITH_SET = set(I_TYPE_ARITH_MNEMONICS.values())
_LOAD_SET = set(LOAD_MNEMONICS.values())
_STORE_SET = set(STORE_MNEMONICS.values())
_BRANCH_SET = set(BRANCH_MNEMONICS.values())
_U_TYPE_SET = {"LUI", "AUIPC"}


def assemble(source: str, base_address: int = 0) -> list[int]:
    """Assemble a short RV32I program (one instruction per line, labels
    allowed) into a list of raw 32-bit machine words, in program order.
    `base_address` is where the first instruction is assumed to live --
    it only matters if the program uses labels, since label addresses
    (and therefore branch/jump displacements) are computed from it.
    """
    lines = source.splitlines()
    label_addresses = _first_pass_label_addresses(lines, base_address)
    return _second_pass_encode(lines, label_addresses, base_address)


def assemble_line(line: str) -> int:
    """Assemble a single instruction with no labels involved -- the
    section-43 example: assemble_line("add x5, x6, x7") == 0x007302B3.
    """
    words = assemble(line)
    if len(words) != 1:
        raise ValueError(f"expected exactly one instruction, got {len(words)}")
    return words[0]


def _first_pass_label_addresses(lines: list[str], base_address: int) -> dict[str, int]:
    """Pass 1: walk the source once, recording where every label
    points. Nothing gets encoded here -- this pass only exists to
    answer "what address is this name" before pass 2 needs to know.
    """
    label_addresses: dict[str, int] = {}
    address = base_address
    for line in lines:
        label, mnemonic, _operands = _tokenize(line)
        if label is not None:
            if label in label_addresses:
                raise ValueError(f"label {label!r} defined more than once")
            label_addresses[label] = address
        if mnemonic:
            address += INSTRUCTION_WIDTH
    return label_addresses


def _second_pass_encode(lines: list[str], label_addresses: dict[str, int],
                         base_address: int) -> list[int]:
    """Pass 2: walk the source again, this time actually encoding. Every
    label is already known (from pass 1), so a branch/jump to a label
    defined later in the file resolves exactly the same way as one to a
    label defined earlier -- that's the entire reason this is two passes
    instead of one.
    """
    words = []
    address = base_address
    for line in lines:
        _label, mnemonic, operands = _tokenize(line)
        if not mnemonic:
            continue  # blank line, comment-only line, or label-only line
        kwargs = _parse_operands(mnemonic, operands, label_addresses, address)
        words.append(encode(mnemonic, **kwargs))
        address += INSTRUCTION_WIDTH
    return words


def _tokenize(line: str) -> tuple[str | None, str, list[str]]:
    """One source line -> (label_or_None, mnemonic, operand_strings).
    `mnemonic` is `""` for a blank line, a comment-only line, or a
    label-only line -- all three carry no instruction to encode.
    """
    line = line.split("#", 1)[0].strip()  # strip a trailing comment
    if not line:
        return None, "", []

    label = None
    if ":" in line:
        label_part, _, rest = line.partition(":")
        label = label_part.strip()
        line = rest.strip()
        if not line:
            return label, "", []  # label-only line, e.g. "loop:"

    mnemonic, _, operand_text = line.partition(" ")
    mnemonic = mnemonic.strip().upper()
    operands = [op.strip() for op in operand_text.split(",")] if operand_text.strip() else []
    return label, mnemonic, operands


def _parse_operands(mnemonic: str, operands: list[str],
                     label_addresses: dict[str, int], current_address: int) -> dict:
    """Operand order is read off the same shape to_asm() renders each
    format in (decoder.py's mirror, in reverse) -- e.g. a store's
    "register, offset(base)" isn't a load's "register, register,
    register".
    """
    try:
        if mnemonic in _R_TYPE_SET:
            rd, rs1, rs2 = operands
            return dict(rd=_parse_register(rd), rs1=_parse_register(rs1), rs2=_parse_register(rs2))

        if mnemonic in _SHIFT_IMMEDIATE_SET:
            rd, rs1, shamt = operands
            return dict(rd=_parse_register(rd), rs1=_parse_register(rs1),
                        immediate=_parse_immediate(shamt))

        if mnemonic in _I_TYPE_ARITH_SET or mnemonic == "JALR":
            rd, rs1, imm = operands
            return dict(rd=_parse_register(rd), rs1=_parse_register(rs1),
                        immediate=_parse_immediate(imm))

        if mnemonic in _LOAD_SET:
            rd, offset_base = operands
            imm, rs1 = _parse_offset_register(offset_base)
            return dict(rd=_parse_register(rd), rs1=rs1, immediate=imm)

        if mnemonic in _STORE_SET:
            rs2, offset_base = operands
            imm, rs1 = _parse_offset_register(offset_base)
            return dict(rs1=rs1, rs2=_parse_register(rs2), immediate=imm)

        if mnemonic in _BRANCH_SET:
            rs1, rs2, target = operands
            immediate = _resolve_branch_or_jump_target(target, label_addresses, current_address)
            return dict(rs1=_parse_register(rs1), rs2=_parse_register(rs2), immediate=immediate)

        if mnemonic == "JAL":
            rd, target = operands
            immediate = _resolve_branch_or_jump_target(target, label_addresses, current_address)
            return dict(rd=_parse_register(rd), immediate=immediate)

        if mnemonic in _U_TYPE_SET:
            # LUI/AUIPC's assembly operand is conventionally the 20-bit
            # upper-immediate value itself (e.g. "lui x3, 0x12345"
            # means "load 0x12345 into the top 20 bits"), not the
            # already-shifted 32-bit result decoder.py's Instruction
            # stores -- so it's shifted left 12 here, at the text-
            # parsing boundary, rather than changing what
            # Instruction.immediate means for every other format.
            rd, imm = operands
            return dict(rd=_parse_register(rd), immediate=_parse_immediate(imm) << 12)
    except ValueError as exc:
        raise ValueError(f"{mnemonic} {', '.join(operands)}: {exc}") from exc

    raise ValueError(f"unknown mnemonic {mnemonic!r}")


def _parse_register(token: str) -> int:
    token = token.strip()
    if len(token) < 2 or token[0] not in ("x", "X"):
        raise ValueError(f"expected a register like 'x5', got {token!r}")
    try:
        register = int(token[1:])
    except ValueError:
        raise ValueError(f"expected a register like 'x5', got {token!r}") from None
    if not (0 <= register <= 31):
        raise ValueError(f"register out of range 0-31: {token!r}")
    return register


def _parse_immediate(token: str) -> int:
    token = token.strip()
    try:
        return int(token, 0)  # base 0: auto-detects 0x/0b/0o, plain decimal, and a leading '-'
    except ValueError:
        raise ValueError(f"expected a number, got {token!r}") from None


def _parse_offset_register(token: str) -> tuple[int, int]:
    """"12(x2)" -> (12, 2) -- a load/store's offset(base) operand."""
    token = token.strip()
    if "(" not in token or not token.endswith(")"):
        raise ValueError(f"expected 'offset(register)', got {token!r}")
    imm_part, _, reg_part = token.partition("(")
    return _parse_immediate(imm_part), _parse_register(reg_part.rstrip(")"))


def _resolve_branch_or_jump_target(token: str, label_addresses: dict[str, int],
                                    current_address: int) -> int:
    """A branch/jump operand is either a label name (resolved to a
    PC-relative displacement -- target address minus *this*
    instruction's own address, the actual quantity RV32I's B/J-type
    immediate encodes) or a plain numeric displacement already.
    """
    token = token.strip()
    if token in label_addresses:
        return label_addresses[token] - current_address
    if token[:1] in ("-", "+") or token[:1].isdigit():
        return _parse_immediate(token)
    raise ValueError(f"undefined label {token!r}")


def main():
    """Both worked examples: section 43's single instruction, and
    section 44/45's label loop.
    """
    print("=" * 60)
    print("SECTION 43: add x5, x6, x7")
    print("=" * 60)
    word = assemble_line("add x5, x6, x7")
    print(f"assemble_line('add x5, x6, x7') = {word:#010X}")
    assert word == 0x007302B3
    print()

    print("=" * 60)
    print("SECTION 44/45: a label, two-pass resolved")
    print("=" * 60)
    program = """
loop:
    addi x5, x5, -1
    bne  x5, x0, loop
    """
    lines = program.splitlines()
    labels = _first_pass_label_addresses(lines, base_address=0)
    for name, addr in labels.items():
        print(f"{name} -> {addr:#010x}")
    words = assemble(program)
    for line, word in zip((l for l in lines if l.strip() and not l.strip().endswith(":")), words):
        print(f"{line.strip():<22} {word:#010x}")
    print("=" * 60)


if __name__ == "__main__":
    main()
