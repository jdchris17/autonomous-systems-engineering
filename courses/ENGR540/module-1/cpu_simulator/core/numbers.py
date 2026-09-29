"""Number representation: converting between decimal, binary, and hex, and
between unsigned and signed (two's complement) interpretations of a bit
pattern.

Scope note: a bit pattern has no inherent meaning -- "10000000" is 128
unsigned, -128 signed (two's complement), or just a byte, depending on how
you choose to read it. That's why signed_to_twos_complement/
twos_complement_to_signed are separate from decimal_to_binary/
binary_to_decimal: the unsigned pair treats the string as a plain base-2
numeral, the signed pair treats it as a two's complement encoding. Keeping
them separate instead of one "smart" converter is deliberate -- hardware
doesn't know which interpretation you meant either; a 32-bit adder produces
the same bits whether you were thinking of the operands as signed or
unsigned, and it's the width that changes, not this software.
"""

from __future__ import annotations


def decimal_to_binary(n: int, width: int | None = None) -> str:
    """Convert a non-negative integer to its unsigned binary string.

    Without `width`, produces the minimal representation (no leading
    zeros, except "0" itself). With `width`, zero-pads to exactly that
    many bits, and raises if `n` doesn't fit -- silently truncating a
    value that overflows its width is exactly the kind of bug this
    toolkit exists to make impossible to write by accident.
    """
    if n < 0:
        raise ValueError(f"decimal_to_binary is for unsigned values, got {n}; "
                          f"use signed_to_twos_complement for negative numbers")
    bits = bin(n)[2:]  # strip the "0b" prefix
    if width is None:
        return bits
    if len(bits) > width:
        raise ValueError(f"{n} does not fit in {width} unsigned bits "
                          f"(max is {unsigned_range(width)[1]})")
    return bits.zfill(width)


def binary_to_decimal(bits: str) -> int:
    """Interpret a binary string as an unsigned base-2 numeral."""
    _validate_bit_string(bits)
    return int(bits, 2)


def binary_to_hex(bits: str) -> str:
    """Convert a binary string to uppercase hex, padding on the left to a
    whole number of nibbles first (hardware groups bits into nibbles for
    hex display; a 6-bit value gets a leading zero nibble, not a half
    digit).
    """
    _validate_bit_string(bits)
    padded_width = ((len(bits) + 3) // 4) * 4
    padded = bits.zfill(padded_width)
    hex_digits = padded_width // 4
    return f"{int(padded, 2):0{hex_digits}X}"


def hex_to_binary(hex_digits: str, width: int | None = None) -> str:
    """Convert a hex string to its binary string.

    Without `width`, produces exactly 4 bits per hex digit (the natural
    width for the given digits). With `width`, zero-pads or validates
    that the value fits, same contract as decimal_to_binary.
    """
    n = int(hex_digits, 16)
    natural_width = len(hex_digits) * 4
    if width is None:
        return decimal_to_binary(n, width=natural_width)
    return decimal_to_binary(n, width=width)


def unsigned_range(width: int) -> tuple[int, int]:
    """Inclusive (min, max) representable by `width` unsigned bits."""
    _validate_width(width)
    return (0, 2 ** width - 1)


def signed_range(width: int) -> tuple[int, int]:
    """Inclusive (min, max) representable by `width` two's-complement bits.

    One more negative value than positive: the top bit alone (e.g.
    "1000" for width=4) is -2^(width-1), but there's no positive
    counterpart at +2^(width-1) since that needs a 0 top bit plus all
    the value bits, which is one bit pattern short.
    """
    _validate_width(width)
    return (-(2 ** (width - 1)), 2 ** (width - 1) - 1)


def signed_to_twos_complement(n: int, width: int) -> str:
    """Encode a signed integer as a `width`-bit two's complement string.

    Negative numbers are encoded as 2**width + n (equivalent to invert-
    and-add-one on the unsigned bit pattern, but easier to get right
    starting from the number rather than the bits).
    """
    _validate_width(width)
    lo, hi = signed_range(width)
    if not (lo <= n <= hi):
        raise ValueError(f"{n} does not fit in {width} signed bits (range is [{lo}, {hi}])")
    encoded = n if n >= 0 else (2 ** width + n)
    return decimal_to_binary(encoded, width=width)


def twos_complement_to_signed(bits: str) -> int:
    """Interpret a bit string as a two's complement signed integer.

    Width is taken from len(bits) -- two's complement is meaningless
    without an agreed-upon width, since the same value bits mean
    different things padded to different widths.
    """
    _validate_bit_string(bits)
    width = len(bits)
    unsigned_value = int(bits, 2)
    if bits[0] == "1":  # top bit set -> negative
        return unsigned_value - 2 ** width
    return unsigned_value


def _validate_bit_string(bits: str) -> None:
    if not bits or any(b not in "01" for b in bits):
        raise ValueError(f"not a binary string: {bits!r}")


def _validate_width(width: int) -> None:
    if width < 1:
        raise ValueError(f"width must be at least 1, got {width}")
