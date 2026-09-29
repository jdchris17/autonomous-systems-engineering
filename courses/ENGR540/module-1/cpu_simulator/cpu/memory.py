"""A deliberately tiny, byte-addressable RAM -- matching the future
processor's own addressing (every load/store instruction addresses
individual bytes, even one moving a whole word), and little-endian, also
matching RISC-V.

Internally just a bytearray: Python's bytearray already enforces "each
cell holds exactly one byte" for us (assigning outside 0-255 raises), so
there's no separate width mask to write here the way register.py needs
one -- the storage type itself *is* the width enforcement. What this
file adds on top is address validation (asking for a byte this memory
doesn't physically have is a different kind of error from a value not
fitting a cell -- closer to a real bus fault than a truncation, so it
raises rather than wrapping) and little-endian word assembly/
disassembly, built from bits.py's own primitives (left_shift, bit_or,
extract_bits) rather than Python's int.from_bytes/to_bytes. This is
exactly the byte-order work bits.py's extract_bits was introduced for.

Instruction memory and data memory are conceptually distinct even though
this one class backs both -- a real (simple) datapath keeps them as two
separate address spaces so instruction fetch and load/store never
collide or alias each other. That distinction is made by instantiating
Memory twice, not by anything in this class:

    instruction_memory = Memory(size_bytes=...)
    data_memory = Memory(size_bytes=...)

Two Memory instances share no state -- see
TestSeparateInstructionAndDataMemory in tests/test_memory.py. Unified
(single address space, both roles sharing one Memory instance) versus
Harvard-style separate memories is a real, later design decision, not
one this file makes for you.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    # See core/adder.py's module docstring for why this is needed to run
    # this file directly (e.g. VS Code's Run button) rather than via
    # `python -m`. Three .parent calls: memory.py -> cpu/ ->
    # cpu_simulator/ -> module-1/.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.core.bits import bit_or, extract_bits, left_shift

WORD_SIZE = 4  # bytes


class Memory:
    def __init__(self, size_bytes: int):
        if size_bytes < 1:
            raise ValueError(f"size_bytes must be at least 1, got {size_bytes}")
        self._size = size_bytes
        self._bytes = bytearray(size_bytes)

    @property
    def size_bytes(self) -> int:
        return self._size

    def read_byte(self, address: int) -> int:
        self._validate_address(address)
        return self._bytes[address]

    def write_byte(self, address: int, value: int) -> None:
        """Masks `value` to 8 bits rather than raising -- a byte cell
        can't hold more than a byte, the same storage-truncates
        philosophy register.py's write_next() uses.
        """
        self._validate_address(address)
        self._bytes[address] = value & 0xFF

    def read_word(self, address: int) -> int:
        """Little-endian: the byte at `address` is the *least*
        significant byte of the word.
        """
        self._validate_address(address, span=WORD_SIZE)
        b0 = self.read_byte(address)
        b1 = self.read_byte(address + 1)
        b2 = self.read_byte(address + 2)
        b3 = self.read_byte(address + 3)
        word = bit_or(b0, left_shift(b1, 8))
        word = bit_or(word, left_shift(b2, 16))
        word = bit_or(word, left_shift(b3, 24))
        return word

    def write_word(self, address: int, value: int) -> None:
        """Little-endian: value's least significant byte lands at
        `address`. Each byte is pulled out with extract_bits, whose own
        masking already gives write_word the same truncate-not-raise
        storage behavior write_byte has -- a value wider than 32 bits
        just loses the bits that don't fit in this word.
        """
        self._validate_address(address, span=WORD_SIZE)
        self.write_byte(address, extract_bits(value, 0, 8))
        self.write_byte(address + 1, extract_bits(value, 8, 8))
        self.write_byte(address + 2, extract_bits(value, 16, 8))
        self.write_byte(address + 3, extract_bits(value, 24, 8))

    def _validate_address(self, address: int, span: int = 1) -> None:
        if address < 0 or address + span > self._size:
            raise ValueError(
                f"address {address} (span {span} byte{'s' if span != 1 else ''}) "
                f"is out of bounds for {self._size}-byte memory"
            )
