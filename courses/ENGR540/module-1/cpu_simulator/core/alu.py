"""The arithmetic-logic unit: one function (`alu`) dispatching to every
operation built up so far -- adder.py's ripple_add for arithmetic,
logic.py's gates for bitwise ops, bits.py's shifts for SLL/SRL, and a
from-scratch sign-extending shift for SRA (bits.py deliberately only
offers a *logical* shift; what an arithmetic shift fills the vacated bits
with is a signed-interpretation question, which is this file's job, the
same way numbers.py -- not bits.py -- owns the unsigned/two's-complement
split).

Real ALUs are physically built as `width` identical 1-bit slices wired in
parallel (a "bit-slice" ALU). AND/OR/XOR mirror that here: they run the
single-bit logic.py gate across every bit position rather than reaching
for bits.py's own bit_and/bit_or/bit_xor. Those exist as a convenience
shortcut for callers who don't care how the bits got combined; this file
does care, because tracing every word-level operation back to a single
NAND-built gate is the entire point of this module's place in the
gates -> adders -> ALU hierarchy.

Flags
-----
zero      Result == 0. Defined for every operation, not just
          arithmetic -- it's just a property of the output word.
negative  The result's own sign bit (bit `width - 1`): how the result
          would read if interpreted as two's complement. Also defined
          for every operation's output.
carry     Only meaningful for ADD/SUB; this ALU reports 0 for every
          other operation (simple ALU designs tie it off rather than
          defining it for logical/shift ops). For ADD it's the adder's
          own carry-out -- unsigned overflow. For SUB it's the
          carry-out of the *same underlying addition* (A + NOT(B) + 1),
          which is why it reads 1 when no borrow occurred and 0 when
          one did, not the other way around.
overflow  Only meaningful for ADD/SUB. *Signed* overflow: true when the
          two operands actually fed to the adder shared a sign but the
          result doesn't match it -- the one case where two's-
          complement addition lies about its own result's sign.

carry and overflow are answering different questions and can disagree in
either direction. INT32_MAX + 1 overflows signed (0x7FFFFFFF + 1 wraps
past the largest positive signed value) but does not carry (the unsigned
sum 0x80000000 still fits in 32 bits). 0xFFFFFFFF + 1 carries (the
unsigned sum needs a 33rd bit) but does not overflow signed (-1 + 1 = 0
is completely unremarkable as a signed result). SLT and SLTU below are
built from exactly this distinction: SLTU(a, b) is NOT(carry) of the
a - b subtraction -- unsigned "less than" only cares whether a borrow
happened. SLT(a, b) is negative XOR overflow of that same subtraction --
signed "less than" needs the overflow-corrected sign, because on signed
overflow the result's raw sign bit is exactly wrong.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    # See adder.py's module docstring for why this is needed to run this
    # file directly (e.g. VS Code's Run button) rather than via `python -m`.
    # Three .parent calls: alu.py -> core/ -> cpu_simulator/ -> module-1/.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.core.adder import ripple_add
from cpu_simulator.core.bits import get_bit, left_shift, right_shift, set_bit
from cpu_simulator.core.logic import AND, NOT, OR, XOR

OPERATIONS = ("ADD", "SUB", "AND", "OR", "XOR", "SLL", "SRL", "SRA", "SLT", "SLTU")


@dataclass(frozen=True)
class ALUResult:
    value: int
    zero: bool
    carry: bool
    overflow: bool
    negative: bool


def alu(a: int, b: int, operation: str, width: int = 32) -> ALUResult:
    """Run one ALU operation on two `width`-bit unsigned values and
    return the result together with its flags. `operation` is one of
    OPERATIONS, spelled the way the eventual RISC-V decoder will name
    them.
    """
    mask = _mask_for(width)
    _validate_fits("a", a, mask)
    _validate_fits("b", b, mask)
    if operation not in OPERATIONS:
        raise ValueError(f"unknown operation {operation!r}; expected one of {OPERATIONS}")

    carry = 0
    overflow = 0

    if operation == "ADD":
        value, carry = ripple_add(a, b, width=width, carry_in=0)
        overflow = _add_overflow(a, b, value, width)
    elif operation == "SUB":
        value, carry, overflow = _subtract_core(a, b, width)
    elif operation == "AND":
        value = _bitwise(AND, a, b, width)
    elif operation == "OR":
        value = _bitwise(OR, a, b, width)
    elif operation == "XOR":
        value = _bitwise(XOR, a, b, width)
    elif operation == "SLL":
        value = left_shift(a, _shift_amount(b, width)) & mask
    elif operation == "SRL":
        value = right_shift(a, _shift_amount(b, width))
    elif operation == "SRA":
        value = _arithmetic_right_shift(a, _shift_amount(b, width), width)
    elif operation == "SLT":
        sub_value, _sub_carry, sub_overflow = _subtract_core(a, b, width)
        sub_negative = get_bit(sub_value, width - 1)
        value = XOR(sub_negative, sub_overflow)
    else:  # SLTU
        _sub_value, sub_carry, _sub_overflow = _subtract_core(a, b, width)
        value = NOT(sub_carry)

    return ALUResult(
        value=value,
        zero=(value == 0),
        carry=bool(carry),
        overflow=bool(overflow),
        negative=bool(get_bit(value, width - 1)),
    )


def _subtract_core(a: int, b: int, width: int) -> tuple[int, int, int]:
    """A - B computed once, shared by SUB, SLT, and SLTU -- all three
    are genuinely the same subtraction, just reading different pieces
    of its output (the difference itself, its carry, or its
    overflow-corrected sign).
    """
    mask = _mask_for(width)
    b_inverted = b ^ mask
    value, carry = ripple_add(a, b_inverted, width=width, carry_in=1)
    overflow = _add_overflow(a, b_inverted, value, width)
    return value, carry, overflow


def _add_overflow(x: int, y: int, result: int, width: int) -> int:
    """Signed overflow of the addition x + y -> result (all `width`-bit
    two's complement), computed with the same combinational logic a real
    adder's overflow detector uses: the operands agreed on sign, but the
    result doesn't match it. Built from logic.py gates, not `==`/`!=`,
    since this is itself a small piece of ALU circuitry.
    """
    sign_x = get_bit(x, width - 1)
    sign_y = get_bit(y, width - 1)
    sign_result = get_bit(result, width - 1)
    same_operand_signs = NOT(XOR(sign_x, sign_y))
    result_sign_differs = XOR(sign_x, sign_result)
    return AND(same_operand_signs, result_sign_differs)


def _bitwise(gate, a: int, b: int, width: int) -> int:
    """Apply a single-bit logic.py gate across every bit position -- the
    bit-slice structure described in this module's docstring.
    """
    result = 0
    for i in range(width):
        if gate(get_bit(a, i), get_bit(b, i)):
            result = set_bit(result, i)
    return result


def _shift_amount(b: int, width: int) -> int:
    """Only the low bits of b that can index a `width`-bit word matter
    as a shift amount -- real hardware (RISC-V included) ignores the
    rest of the register rather than treating a large shift amount as
    an error. Requires `width` to be a power of two (true of every
    width this module is used with: 8/16/32/64) for `width - 1` to work
    as a bitmask.
    """
    return b & (width - 1)


def _arithmetic_right_shift(value: int, amount: int, width: int) -> int:
    """Sign-extending right shift, built from bits.py's logical shift
    plus manual sign extension: shift right with zero-fill, then, if the
    original sign bit was 1, set every bit the shift vacated back to 1.
    """
    sign_bit = get_bit(value, width - 1)
    shifted = right_shift(value, amount)
    if sign_bit:
        for i in range(width - amount, width):
            shifted = set_bit(shifted, i)
    return shifted & _mask_for(width)


def _mask_for(width: int) -> int:
    if width < 1:
        raise ValueError(f"width must be at least 1, got {width}")
    return (1 << width) - 1


def _validate_fits(name: str, value: int, mask: int) -> None:
    if not (0 <= value <= mask):
        raise ValueError(f"{name} does not fit in the given width "
                          f"(0 <= {name} <= {mask}), got {value}")
