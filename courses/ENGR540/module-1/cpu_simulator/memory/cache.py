"""A cache, starting direct-mapped but built as the general set/way
shape from the beginning -- section 42's own "even though your first
version is direct mapped, design the interface so you can generalize
it later."

Direct-mapped is just the `associativity=1` special case of a
set-associative cache: every set holds exactly one line (one "way"),
so a memory block can only ever land in one specific line. Rather than
storing `Cache`'s lines as one flat list (a direct-mapped-only shape
that a later associativity > 1 would have to restructure), `self.sets`
is already a list of sets, each set a list of `associativity` many
`CacheLine`s -- direct-mapped is simply what that shape degenerates to
when each set's list has length 1. Nothing about `self.sets` itself
needs to change when associativity later becomes configurable to
something greater than 1; only replacement policy (see `read()`) would.

Address decomposition (section 43) reuses `core/bits.py`'s
`extract_bits`/`bit_and` rather than raw Python bit-twiddling -- the
same "Module 1 bit manipulation becomes real computer architecture"
the spec calls out explicitly. `line_size` and `num_sets` must each be
an exact power of 2 for `offsetBits = log2(LineSize)` and
`indexBits = log2(numSets)` to even be whole numbers, so `__init__`
validates that up front rather than letting `split_address()` produce
nonsense later.

The read path (sections 45-46) is exactly the nine steps the spec lists,
in order: split the address, find the set, check each candidate line
for a valid+matching tag (a hit), and on a miss, compute the missed
line's block-aligned base address, pull the whole line from
`lower_memory`, install it, then return the requested byte either way.

Section 47 is what actually exercises `associativity > 1` for the first
time: a lookup already checked *every* line in the indexed set (not
just line 0), so upgrading to genuinely set-associative needed no
change there -- what it needed was a real answer to "otherwise choose a
victim," which line 0 always being the answer never was. Section 48's
LRU is that real answer: every `CacheLine` now carries `last_used`, a
monotonically increasing `Cache.current_access_number` timestamps
on every hit and every install, and `_select_victim()` picks an invalid
line first (free space beats evicting anything), falling back to
whichever valid line has the smallest `last_used` (genuinely least
recently used) only once every line in the set is occupied. At
`associativity=1` this still degenerates to "evict the one line," the
same as before -- LRU with one candidate has no real decision to make
either, which is exactly why generalizing the policy now doesn't
disturb direct-mapped behavior at all.

`write()` (section 49) is write-allocate, write-back: a hit modifies
the cached byte and sets `dirty`; a miss runs the same victim-selection
and block-install machinery `read()`'s miss path uses (factored into
`_handle_miss()`, shared by both), except that if the victim being
evicted is itself dirty, its *old* block is written back to
`lower_memory` first -- reconstructed from the victim's own `tag` and
the set it came from (`(tag << (offset_bits + index_bits)) |
(set_index << offset_bits)`), since a `CacheLine` itself has no memory
of which set it lives in. Only after that does the newly requested
block get fetched and installed, the byte actually being written lands
in it, and the line is marked dirty again -- a write is a strictly
"heavier" `_handle_miss()` than a read's, never a lighter one.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    # See core/adder.py's module docstring for why this is needed to run
    # this file directly (e.g. VS Code's Run button) rather than via
    # `python -m`. Three .parent calls: cache.py -> memory/ ->
    # cpu_simulator/ -> module-1/.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.core.bits import bit_and, extract_bits
from cpu_simulator.memory.memory import Memory

ADDRESS_WIDTH = 32  # bits -- RV32I


@dataclass
class CacheLine:
    valid: bool = False
    dirty: bool = False
    tag: int = 0
    data: bytearray = field(default_factory=bytearray)
    last_used: int = 0


@dataclass
class CacheStatistics:
    accesses: int = 0
    hits: int = 0
    misses: int = 0
    dirty_writebacks: int = 0


def hit_rate(stats: CacheStatistics) -> float:
    return stats.hits / stats.accesses


def miss_rate(stats: CacheStatistics) -> float:
    return stats.misses / stats.accesses


def _extract_bits_or_zero(value: int, start: int, width: int) -> int:
    """`extract_bits`, except a 0-bit field is 0 rather than an error --
    see `Cache.split_address`'s own docstring for why width 0 is a real,
    legitimate case here, not a bug being papered over.
    """
    if width == 0:
        return 0
    return extract_bits(value, start, width)


def _log2_exact(n: int, what: str) -> int:
    """log2(n), refusing anything that isn't an exact power of 2 --
    `offsetBits`/`indexBits` only make sense as whole numbers, and a
    non-power-of-2 `line_size`/`num_sets` would otherwise silently
    produce a `split_address()` that quietly loses or overlaps bits.
    """
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"{what} must be a power of 2, got {n}")
    return n.bit_length() - 1


class Cache:
    def __init__(self, size_bytes: int, line_size: int, lower_memory: Memory,
                 associativity: int = 1):
        if size_bytes < 1:
            raise ValueError(f"size_bytes must be at least 1, got {size_bytes}")
        if line_size < 1:
            raise ValueError(f"line_size must be at least 1, got {line_size}")
        if size_bytes % line_size != 0:
            raise ValueError(
                f"size_bytes ({size_bytes}) must be a multiple of line_size ({line_size})"
            )

        num_lines = size_bytes // line_size
        if associativity < 1:
            raise ValueError(f"associativity must be at least 1, got {associativity}")
        if num_lines % associativity != 0:
            raise ValueError(
                f"{num_lines} lines ({size_bytes} / {line_size}) isn't a whole "
                f"number of {associativity}-way sets"
            )

        self.size_bytes = size_bytes
        self.line_size = line_size
        self.associativity = associativity
        self.num_lines = num_lines
        self.num_sets = num_lines // associativity
        self.lower_memory = lower_memory

        self.offset_bits = _log2_exact(line_size, "line_size")
        self.index_bits = _log2_exact(self.num_sets, "num_sets (size_bytes / (line_size * associativity))")
        self.tag_bits = ADDRESS_WIDTH - self.offset_bits - self.index_bits
        if self.tag_bits < 0:
            raise ValueError(
                f"offset_bits ({self.offset_bits}) + index_bits ({self.index_bits}) "
                f"exceeds the {ADDRESS_WIDTH}-bit address width"
            )

        # A list of sets, each set a list of `associativity` ways -- see
        # the module docstring for why this shape (not one flat list of
        # `num_lines` lines) is what "design the interface so you can
        # generalize it later" actually means here.
        self.sets: list[list[CacheLine]] = [
            [CacheLine(data=bytearray(line_size)) for _ in range(associativity)]
            for _ in range(self.num_sets)
        ]

        self.stats = CacheStatistics()
        self.current_access_number = 0

    def split_address(self, address: int) -> tuple[int, int, int]:
        """`(tag, set_index, block_offset)` -- the three fields every
        address decomposes into, each pulled out with `extract_bits`
        rather than hand-rolled shifts/masks, the same primitive
        `decoder.py` uses to pull an opcode or a register number out of
        a raw instruction word. `block_offset` is the low `offset_bits`
        bits, `set_index` the next `index_bits` bits, and `tag`
        whatever's left above those.

        `extract_bits` refuses a 0-bit field (there's no such thing as
        "the bits at position N, none of them"), but a 0-bit field is
        exactly what a legitimately degenerate cache configuration
        produces -- `index_bits == 0` for a fully-associative cache (a
        single set needs 0 bits to index), or `offset_bits == 0` for a
        1-byte line. `_extract_bits_or_zero` is the one place that
        degenerate case is handled, so `split_address` itself doesn't
        need three copies of the same width-0 special case.
        """
        block_offset = _extract_bits_or_zero(address, 0, self.offset_bits)
        set_index = _extract_bits_or_zero(address, self.offset_bits, self.index_bits)
        tag = _extract_bits_or_zero(address, self.offset_bits + self.index_bits, self.tag_bits)
        return tag, set_index, block_offset

    def read(self, address: int) -> int:
        """One byte, read through the cache. The nine-step path section
        45 lists, in order: split the address, find the address's one
        candidate *set*, inspect every line in that set for a hit
        (valid **and** tag match -- either alone isn't enough, an
        invalid line's leftover tag from a previous, unrelated block
        must never count), and on a miss, install the missing line from
        `lower_memory` before finally returning the requested byte --
        a hit and a post-install miss both end the same way, reading
        `block_offset` out of whichever line just proved to hold the
        right data.
        """
        tag, set_index, block_offset = self.split_address(address)
        candidate_set = self.sets[set_index]
        access_number = self._next_access_number()
        self.stats.accesses += 1

        for line in candidate_set:
            if line.valid and line.tag == tag:
                self.stats.hits += 1
                line.last_used = access_number
                return line.data[block_offset]

        self.stats.misses += 1
        line = self._handle_miss(candidate_set, set_index, address, tag, access_number)
        return line.data[block_offset]

    def write(self, address: int, value: int) -> None:
        """One byte, written through the cache -- write-allocate,
        write-back (section 49). A hit modifies the cached byte in
        place and marks the line dirty, no `lower_memory` access at
        all. A miss runs the exact same victim-selection and
        block-install `read()`'s miss path uses (`_handle_miss()`,
        shared by both -- a write's miss path is that plus two more
        steps, never a different one), then modifies the requested byte
        in the freshly installed line and marks *that* dirty instead.
        `value` is masked to 8 bits, the same truncate-not-raise
        storage behavior `Memory.write_byte` already has.
        """
        tag, set_index, block_offset = self.split_address(address)
        candidate_set = self.sets[set_index]
        access_number = self._next_access_number()
        self.stats.accesses += 1

        for line in candidate_set:
            if line.valid and line.tag == tag:
                self.stats.hits += 1
                line.last_used = access_number
                line.data[block_offset] = value & 0xFF
                line.dirty = True
                return

        self.stats.misses += 1
        line = self._handle_miss(candidate_set, set_index, address, tag, access_number)
        line.data[block_offset] = value & 0xFF
        line.dirty = True

    def _next_access_number(self) -> int:
        self.current_access_number += 1
        return self.current_access_number

    def _select_victim(self, candidate_set: list[CacheLine]) -> CacheLine:
        """Section 48's own policy, verbatim: an invalid line first (an
        empty line always beats evicting anything real), otherwise
        whichever valid line has the smallest `last_used` -- genuinely
        least recently used. Distinct `last_used` values are guaranteed
        unique across every line that's ever actually been touched (see
        `_next_access_number()`), so `min()` never has a tie to break.
        """
        for line in candidate_set:
            if not line.valid:
                return line
        return min(candidate_set, key=lambda line: line.last_used)

    def _handle_miss(self, candidate_set: list[CacheLine], set_index: int,
                      address: int, tag: int, access_number: int) -> CacheLine:
        """Sections 46 and 49's shared miss machinery: pick a victim,
        write it back to `lower_memory` first if it's dirty (section
        49's own steps 1-2), then fetch and install the actually
        requested block (steps 3-4, and section 46's own read-side
        version of the same). `BlockAddress = Address & ~(LineSize - 1)`
        (via `bit_and` rather than Python's bare `&`) is the address of
        the *first* byte of the line containing `address` -- reading
        `line_size` bytes starting there, not just the one byte actually
        asked for, is what makes this a cache instead of a pass-through:
        the whole point of a line is that nearby bytes arrive for free,
        on the assumption a program that touched `address` is likely to
        touch its neighbors next (spatial locality). The caller (`read()`
        or `write()`) still owns steps 5-6 for a write (modifying the
        requested byte, marking it dirty again) -- this only ever leaves
        a freshly installed, *clean* line behind.
        """
        victim = self._select_victim(candidate_set)
        if victim.valid and victim.dirty:
            self._writeback_dirty_victim(victim, set_index)

        block_address = bit_and(address, ~(self.line_size - 1))
        block_data = bytearray(self.line_size)
        for i in range(self.line_size):
            block_data[i] = self.lower_memory.read_byte(block_address + i)

        victim.valid = True
        victim.dirty = False
        victim.tag = tag
        victim.data = block_data
        victim.last_used = access_number
        return victim

    def _writeback_dirty_victim(self, victim: CacheLine, set_index: int) -> None:
        """Writes a dirty line's *old* block back to `lower_memory`
        before it gets overwritten -- otherwise the only copy of a
        modification a hit ever made (a cache hit never touches
        `lower_memory` at all) would simply vanish when the line is
        evicted. `CacheLine` doesn't remember which set it lives in, so
        the old block's address is reconstructed here from the victim's
        own `tag` plus the `set_index` the caller already knows (the
        inverse of `split_address()`: tag and index back into an
        address, instead of an address apart into tag and index).
        """
        old_block_address = (victim.tag << (self.offset_bits + self.index_bits)) \
            | (set_index << self.offset_bits)
        for i in range(self.line_size):
            self.lower_memory.write_byte(old_block_address + i, victim.data[i])
        self.stats.dirty_writebacks += 1


def format_address_breakdown(cache: Cache, address: int) -> str:
    """Section 44's own debugging output -- "build that inspection
    capability early." A pure renderer, the same split from the actual
    decomposition `trace.py`'s `format_*` functions keep from
    `StepTrace`/`PipelineTrace`: `split_address()` computes, this only
    displays.
    """
    tag, set_index, block_offset = cache.split_address(address)
    return (
        f"Address: {address:#010x}\n"
        f"\n"
        f"Tag:    {tag:#x}\n"
        f"Index:  {set_index}\n"
        f"Offset: {block_offset}"
    )
