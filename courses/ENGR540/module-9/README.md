# Module 9 — Caching: address decomposition and the read path

The library code this module is built on lives in
[`module-1/cpu_simulator/memory/`](../module-1/README.md), following the
project's standing rule: `cpu_simulator/` is the base library and never
depends on any `module-N/` folder, while `module-N/` folders are
consumers of it. Section 41's own repository update moved `memory.py`
out of `cpu/` into a new `memory/` package specifically because a cache
sitting in front of RAM is general-purpose memory-hierarchy machinery,
not CPU-specific — "the CPU should **use** a hierarchy. It shouldn't
itself be the hierarchy." `cache.py` lives there too, right alongside
`memory.py`. This folder holds only the *outside program build*:
`cache_demo.py`, plus its test.

## What's new in `cpu_simulator/memory/cache.py` (summary — see module-1's README for the full detail)

- **Address decomposition (sections 43-44)** — `Cache.split_address(address) -> (tag, set_index, block_offset)`, built from `core/bits.py`'s `extract_bits` rather than hand-rolled shifts — "Module 1 bit manipulation becomes real computer architecture," literally. `offset_bits`/`index_bits`/`tag_bits` are computed once in `__init__` (`log2(line_size)`, `log2(num_sets)`, and whatever's left of a 32-bit address), which requires `line_size` and `num_sets` to each be an exact power of 2 — validated at construction, not discovered later as a silently-wrong decomposition. `format_address_breakdown(cache, address)` is the debugging output section 44 asks to "build early," kept as its own pure renderer the same way `trace.py`'s `format_*` functions stay separate from the traces they render.
- **The read path (sections 45-46)** — `Cache.read(address) -> int`, the exact nine-step path the spec lists: split the address, find the set, check every candidate line for **valid and** a matching tag (a hit), and on a miss, compute `BlockAddress = Address & ~(LineSize - 1)` (via `core/bits.py`'s `bit_and`), pull the whole line from a `lower_memory: Memory` the cache now owns from construction, install it, and return the requested byte. `CacheStatistics` (`accesses`/`hits`/`misses`) plus `hit_rate()`/`miss_rate()` track exactly what "update statistics" asks for, mirroring `performance.py`'s counters-plus-pure-ratio-functions shape from Module 8.
- **Known, deliberate limitation**: eviction on a miss always targets way 0 of the set — a real placeholder, not a replacement policy. `associativity=1` (the only case anything here actually exercises) has no replacement *decision* to make (a set with one way only has that one way to evict), so this doesn't matter yet; a real policy (LRU or otherwise) is future work for whenever `associativity > 1` actually gets used. `write()` isn't built yet either — `CacheLine.dirty` exists already, unused until then.

## [`cache_demo.py`](cache_demo.py) — two worked examples

**`run_address_breakdown_demo()`** — section 44's exact configuration
(1024-byte cache, 16-byte lines, direct-mapped) and exact address
(`0x12345678`). Confirms `Lines=64, Sets=64, OffsetBits=4, IndexBits=6,
TagBits=22` match the spec's own numbers, then prints the debugging
breakdown (`Address: / Tag: / Index: / Offset:`) section 44 asks for.
No memory access at all — pure address math.

**`run_read_path_demo()`** — a small 64-byte, 4-line cache in front of
a 256-byte memory where `memory[address] == address`, so every returned
value doubles as its own correctness check. Walks all four read-path
outcomes in sequence: a first read that misses and installs a whole
16-byte line (confirming the installed block's base address matches
section 46's own `& ~(LineSize-1)` alignment), a second read of a
*different* address in that *same* line that hits for free (spatial
locality, made concrete), a conflicting address in the same set that
evicts the line, and a final re-read of the original address that
misses again because of that eviction. Ends by printing
`CacheStatistics` plus `hit_rate()`/`miss_rate()`.

## Run it

```
python cache_demo.py
```
