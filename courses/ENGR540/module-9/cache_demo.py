"""Module 9's seven sections, worked end to end: split an address into
tag/index/offset, print the debugging breakdown "build early," run real
reads through the cache and watch hits/misses/block-installation/
eviction happen, upgrade to a genuinely set-associative lookup, watch
LRU pick a real victim instead of always evicting way 0, and finally
watch a write-back cache defer touching `lower_memory` at all until a
dirty line actually gets evicted.

Four independent walkthroughs, since they exercise different things:

    run_address_breakdown_demo()  sections 43-44 -- pure address math,
                                    no memory access at all
    run_read_path_demo()          sections 45-46 -- the real read path,
                                    a direct-mapped cache, including a
                                    miss installing a whole line from
                                    `lower_memory`
    run_lru_demo()                sections 47-48 -- a 4-way
                                    set-associative cache, filled, then
                                    forced to actually choose a victim
    run_write_back_demo()         section 49 -- a write hit that never
                                    touches `lower_memory`, and a dirty
                                    eviction that finally does
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module-1"))

from cpu_simulator.memory.cache import Cache, format_address_breakdown, hit_rate, miss_rate
from cpu_simulator.memory.memory import Memory

# Section 44's exact worked example.
EXAMPLE_CACHE_SIZE = 1024
EXAMPLE_LINE_SIZE = 16
EXAMPLE_ASSOCIATIVITY = 1
EXAMPLE_ADDRESS = 0x12345678


def run_address_breakdown_demo() -> Cache:
    """No reads at all -- just the address-decomposition math sections
    43-44 ask for, confirmed against the spec's own numbers.
    """
    backing = Memory(size_bytes=EXAMPLE_CACHE_SIZE)
    cache = Cache(size_bytes=EXAMPLE_CACHE_SIZE, line_size=EXAMPLE_LINE_SIZE,
                   lower_memory=backing, associativity=EXAMPLE_ASSOCIATIVITY)

    print(f"Lines = {cache.num_lines} (expect 64)")
    print(f"Sets  = {cache.num_sets} (expect 64)")
    print(f"OffsetBits = {cache.offset_bits} (expect 4)")
    print(f"IndexBits  = {cache.index_bits} (expect 6)")
    print(f"TagBits    = {cache.tag_bits} (expect 22)")
    assert cache.num_lines == 64
    assert cache.num_sets == 64
    assert cache.offset_bits == 4
    assert cache.index_bits == 6
    assert cache.tag_bits == 22
    print()

    print(format_address_breakdown(cache, EXAMPLE_ADDRESS))
    print()
    return cache


def run_read_path_demo() -> Cache:
    """A small 64-byte, 4-line direct-mapped cache in front of a
    256-byte memory, `memory[address] = address` so every returned
    value doubles as its own correctness check.
    """
    backing = Memory(size_bytes=256)
    for address in range(256):
        backing.write_byte(address, address)
    cache = Cache(size_bytes=64, line_size=16, lower_memory=backing, associativity=1)

    print("--- first read: a miss, installs the whole line ---")
    value = cache.read(0x1A)
    block_address = 0x1A - cache.split_address(0x1A)[2]
    print(f"cache.read(0x1A) = {value} (expect 26)")
    print(f"installed line's block address = {block_address:#x} "
          f"(covers {block_address:#x} through {block_address + 15:#x})")
    assert value == 0x1A
    assert block_address == 0x10
    assert cache.stats.misses == 1 and cache.stats.hits == 0
    print()

    print("--- second read, same line: a hit, no memory access needed ---")
    value = cache.read(0x1F)  # still within 0x10-0x1F, installed for free
    print(f"cache.read(0x1F) = {value} (expect 31)")
    assert value == 0x1F
    assert cache.stats.hits == 1
    print()

    print("--- a conflicting address (same set, different tag): evicts ---")
    conflicting_address = 0x1A + cache.size_bytes  # same set_index, different tag
    value = cache.read(conflicting_address)
    print(f"cache.read({conflicting_address:#x}) = {value} "
          f"(expect {conflicting_address % 256})")
    assert cache.stats.misses == 2
    print()

    print("--- re-reading the original address: evicted, so a miss again ---")
    value = cache.read(0x1A)
    print(f"cache.read(0x1A) = {value} (expect 26)")
    assert cache.stats.misses == 3
    print()

    print(f"final stats: {cache.stats}")
    print(f"hit rate = {hit_rate(cache.stats):.2f}, miss rate = {miss_rate(cache.stats):.2f}")
    return cache


def run_lru_demo() -> Cache:
    """A 4-way set-associative cache with exactly one set (64 bytes,
    16-byte lines, 4-way -- one set holding all four lines): fill every
    way, deliberately re-touch three of the four to make the fourth
    genuinely least-recently-used, then force a fifth distinct access
    into the same set and confirm *that* specific line -- and only that
    one -- gets evicted.
    """
    backing = Memory(size_bytes=1024)
    for address in range(1024):
        backing.write_byte(address, address % 256)
    cache = Cache(size_bytes=64, line_size=16, lower_memory=backing, associativity=4)
    assert cache.num_sets == 1  # one set, four ways -- fully associative here

    print("--- filling all four ways (addresses 0, 16, 32, 48) ---")
    for address in (0, 16, 32, 48):
        cache.read(address)
    print(f"stats: {cache.stats} (expect 4 misses, 0 hits)")
    assert cache.stats.misses == 4
    print()

    print("--- re-touching 0, 32, 48 -- address 16's line is now the LRU ---")
    cache.read(0)
    cache.read(32)
    cache.read(48)
    print(f"stats: {cache.stats}")
    print()

    print("--- a 5th distinct line (address 64) forces a real eviction ---")
    cache.read(64)
    print(f"stats: {cache.stats} (expect a 5th miss)")
    assert cache.stats.misses == 5
    print()

    print("--- 0/32/48 are still resident; 16 was the one evicted ---")
    hits_before = cache.stats.hits
    for address in (0, 32, 48):
        cache.read(address)
    print(f"stats after re-reading 0/32/48: {cache.stats} "
          f"(expect {hits_before + 3} hits total)")
    assert cache.stats.hits == hits_before + 3

    cache.read(16)  # the one that was actually evicted
    print(f"stats after re-reading 16: {cache.stats} (expect one more miss)")
    assert cache.stats.misses == 6
    print()
    return cache


def run_write_back_demo() -> tuple[Cache, Memory]:
    """A write hit modifies the cached byte and marks it dirty, but
    never touches `lower_memory` at all -- the whole point of
    write-back over write-through. Only once that dirty line is
    actually evicted does its modification finally reach memory.
    """
    backing = Memory(size_bytes=256)
    for address in range(256):
        backing.write_byte(address, address % 256)
    cache = Cache(size_bytes=64, line_size=16, lower_memory=backing, associativity=1)

    print("--- write(5, 0xAB): a write-allocate miss ---")
    cache.write(5, 0xAB)
    print(f"stats: {cache.stats} (expect 1 miss)")
    print(f"memory[5] = {backing.read_byte(5)} (still 5 -- write-back is lazy)")
    assert cache.stats.misses == 1
    assert backing.read_byte(5) == 5
    print()

    print("--- read(5): a hit, sees the write's own value ---")
    value = cache.read(5)
    print(f"cache.read(5) = {value} (expect 0xAB = 171)")
    assert value == 0xAB
    assert cache.stats.hits == 1
    print()

    print("--- a conflicting address evicts the dirty line ---")
    conflicting_address = 5 + cache.size_bytes
    cache.read(conflicting_address)
    print(f"memory[5] = {backing.read_byte(5)} (now 0xAB -- written back on eviction)")
    print(f"memory[4] = {backing.read_byte(4)}, memory[6] = {backing.read_byte(6)} "
          f"(the rest of the block, untouched)")
    assert backing.read_byte(5) == 0xAB
    assert backing.read_byte(4) == 4
    assert backing.read_byte(6) == 6
    print()
    return cache, backing


def main():
    print("=" * 60)
    print("MODULE 9: address decomposition")
    print("=" * 60)
    run_address_breakdown_demo()

    print("=" * 60)
    print("MODULE 9: the cache read path")
    print("=" * 60)
    run_read_path_demo()

    print("=" * 60)
    print("MODULE 9: set-associative lookup and LRU replacement")
    print("=" * 60)
    run_lru_demo()

    print("=" * 60)
    print("MODULE 9: write-back")
    print("=" * 60)
    run_write_back_demo()
    print("=" * 60)


if __name__ == "__main__":
    main()
