import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cpu_simulator.memory.cache import Cache, CacheLine, format_address_breakdown, hit_rate, miss_rate
from cpu_simulator.memory.memory import Memory


def _cache(size_bytes=64, line_size=16, associativity=1, memory_size=4096, memory=None):
    memory = memory or Memory(size_bytes=memory_size)
    return Cache(size_bytes=size_bytes, line_size=line_size,
                 lower_memory=memory, associativity=associativity)


class TestCacheLineDefaults:
    def test_a_fresh_line_is_invalid_and_clean(self):
        line = CacheLine()
        assert line.valid is False
        assert line.dirty is False
        assert line.tag == 0
        assert line.data == bytearray()

    def test_each_line_gets_its_own_data_array_not_a_shared_one(self):
        # A mutable dataclass default must be a fresh object per
        # instance -- a shared bytearray() default would let writing
        # one line's data corrupt every other line's.
        a, b = CacheLine(), CacheLine()
        a.data.extend(b"x")
        assert b.data == bytearray()


class TestCacheIsDirectMappedByDefault:
    def test_associativity_defaults_to_one(self):
        assert _cache().associativity == 1

    def test_every_set_holds_exactly_one_way(self):
        assert all(len(s) == 1 for s in _cache().sets)

    def test_num_sets_equals_num_lines_when_direct_mapped(self):
        cache = _cache()
        assert cache.num_lines == 4
        assert cache.num_sets == 4
        assert len(cache.sets) == 4

    def test_every_line_starts_invalid_with_a_correctly_sized_data_array(self):
        for s in _cache().sets:
            for line in s:
                assert line.valid is False
                assert len(line.data) == 16


class TestCacheGeneralizesToSetAssociative:
    def test_a_two_way_cache_has_half_as_many_sets_as_lines(self):
        cache = _cache(associativity=2)
        assert cache.num_lines == 4
        assert cache.num_sets == 2
        assert all(len(s) == 2 for s in cache.sets)

    def test_a_fully_associative_cache_is_one_set_holding_every_line(self):
        cache = _cache(associativity=4)
        assert cache.num_sets == 1
        assert len(cache.sets[0]) == 4


class TestCacheValidation:
    def test_rejects_size_not_a_multiple_of_line_size(self):
        with pytest.raises(ValueError):
            _cache(size_bytes=50, line_size=16)

    def test_rejects_line_count_not_a_whole_number_of_sets(self):
        with pytest.raises(ValueError):
            _cache(associativity=3)  # 4 lines, not divisible by 3

    def test_rejects_non_positive_size(self):
        with pytest.raises(ValueError):
            _cache(size_bytes=0)

    def test_rejects_non_positive_line_size(self):
        with pytest.raises(ValueError):
            _cache(line_size=0)

    def test_rejects_non_positive_associativity(self):
        with pytest.raises(ValueError):
            _cache(associativity=0)

    def test_rejects_a_line_size_that_isnt_a_power_of_two(self):
        with pytest.raises(ValueError):
            _cache(size_bytes=60, line_size=12)

    def test_rejects_a_set_count_that_isnt_a_power_of_two(self):
        # 64 bytes / (16-byte lines * 2-way) = 2 sets -- fine. Force a
        # non-power-of-2 set count instead: 96 bytes / 16 = 6 lines,
        # 6 / 1-way = 6 sets, not a power of 2.
        with pytest.raises(ValueError):
            _cache(size_bytes=96, line_size=16)


class TestSplitAddress:
    """Section 44's exact worked example: a 1024-byte, 16-byte-line,
    direct-mapped cache, address 0x12345678.
    """

    def _example_cache(self):
        return _cache(size_bytes=1024, line_size=16, associativity=1)

    def test_bit_widths_match_the_worked_example(self):
        cache = self._example_cache()
        assert cache.num_lines == 64
        assert cache.num_sets == 64
        assert cache.offset_bits == 4
        assert cache.index_bits == 6
        assert cache.tag_bits == 22

    def test_the_exact_worked_address_decomposes_correctly(self):
        cache = self._example_cache()
        tag, set_index, block_offset = cache.split_address(0x12345678)
        assert block_offset == 0x8
        assert set_index == 0x12345678 >> 4 & 0x3F
        assert tag == 0x12345678 >> 10

    def test_offset_is_always_within_the_line(self):
        cache = self._example_cache()
        for address in (0x0, 0xF, 0x10, 0x1F, 0x12345678):
            _, _, offset = cache.split_address(address)
            assert 0 <= offset < cache.line_size

    def test_set_index_is_always_within_bounds(self):
        cache = self._example_cache()
        for address in (0x0, 0x10, 0x12345678, 0xFFFFFFFF):
            _, set_index, _ = cache.split_address(address)
            assert 0 <= set_index < cache.num_sets


class TestFormatAddressBreakdown:
    def test_matches_the_spec_example_shape(self):
        cache = _cache(size_bytes=1024, line_size=16, associativity=1)
        text = format_address_breakdown(cache, 0x12345678)
        assert text.startswith("Address: 0x12345678")
        assert "Tag:" in text
        assert "Index:" in text
        assert "Offset:" in text


class TestReadPath:
    def _populated_cache(self, memory_size=1024, **kwargs):
        memory = Memory(size_bytes=memory_size)
        for address in range(memory_size):
            memory.write_byte(address, address % 256)
        return _cache(memory=memory, **kwargs), memory

    def test_first_read_of_an_address_is_a_miss(self):
        cache, _ = self._populated_cache()
        cache.read(5)
        assert cache.stats.accesses == 1
        assert cache.stats.hits == 0
        assert cache.stats.misses == 1

    def test_a_second_read_of_the_same_line_is_a_hit(self):
        cache, _ = self._populated_cache()
        cache.read(5)
        cache.read(6)  # same 16-byte line as address 5
        assert cache.stats.hits == 1
        assert cache.stats.misses == 1

    def test_returns_the_correct_byte_on_a_miss_and_a_hit(self):
        cache, _ = self._populated_cache()
        assert cache.read(5) == 5    # miss
        assert cache.read(6) == 6    # hit, same line
        assert cache.read(20) == 20  # different line -> miss

    def test_an_invalid_line_never_counts_as_a_hit_even_with_a_matching_leftover_tag(self):
        cache, _ = self._populated_cache()
        line = cache.sets[0][0]
        line.tag = cache.split_address(5)[0]  # matching tag, but...
        line.valid = False                     # ...never installed
        assert cache.read(5) == 5
        assert cache.stats.misses == 1

    def test_a_conflicting_address_in_the_same_set_evicts_the_line(self):
        cache, _ = self._populated_cache()  # 4 sets, direct-mapped
        conflict_stride = cache.size_bytes  # +size_bytes always maps to the same set
        cache.read(5)
        cache.read(5 + conflict_stride)
        assert cache.stats.misses == 2
        # 5's line was evicted -- reading it again is a fresh miss.
        cache.read(5)
        assert cache.stats.misses == 3

    def test_a_non_conflicting_address_does_not_evict(self):
        cache, _ = self._populated_cache()
        cache.read(5)    # set 0
        cache.read(20)   # a different set
        cache.read(5)    # still cached -- a hit
        assert cache.stats.hits == 1
        assert cache.stats.misses == 2


class TestMissHandling:
    def test_block_address_is_aligned_to_the_line_size(self):
        # Section 46's own example: 0x1234567A with 16-byte lines ->
        # block base ends in ...70. Doesn't need to actually read
        # through the cache -- split_address()'s offset alone recovers
        # BlockAddress = Address & ~(LineSize-1), since the block base
        # is just the address with its own offset subtracted off.
        cache = _cache(size_bytes=1024, line_size=16, associativity=1)
        _, _, offset = cache.split_address(0x1234567A)
        assert offset == 0xA
        block_address = 0x1234567A - offset
        assert block_address == 0x12345670

    def test_reading_one_byte_installs_the_whole_line(self):
        memory = Memory(size_bytes=1024)
        for address in range(1024):
            memory.write_byte(address, address % 256)
        cache = _cache(size_bytes=64, line_size=16, associativity=1, memory=memory)

        cache.read(0x21)  # installs the line covering 0x20-0x2F
        set_index = cache.split_address(0x21)[1]
        line = cache.sets[set_index][0]
        assert line.valid is True
        assert list(line.data) == [b % 256 for b in range(0x20, 0x30)]


class TestCacheStatisticsRates:
    def test_hit_rate_and_miss_rate_are_complementary(self):
        memory = Memory(size_bytes=1024)
        cache = _cache(memory=memory)
        cache.read(5)   # miss
        cache.read(6)   # hit
        cache.read(7)   # hit
        assert cache.stats.accesses == 3
        assert hit_rate(cache.stats) == 2 / 3
        assert miss_rate(cache.stats) == 1 / 3
        assert hit_rate(cache.stats) + miss_rate(cache.stats) == 1.0


class TestSetAssociativeLookup:
    """Section 47: a lookup now compares the requested tag against
    every way in the indexed set, not just one line.
    """

    def _populated_memory(self, size=1024):
        memory = Memory(size_bytes=size)
        for address in range(size):
            memory.write_byte(address, address % 256)
        return memory

    def test_a_hit_in_any_way_of_the_set_counts_as_a_hit(self):
        # 4-way, one set total -- four distinct lines all coexist.
        cache = _cache(size_bytes=64, line_size=16, associativity=4,
                        memory=self._populated_memory())
        cache.read(0)    # way A
        cache.read(16)   # way B
        cache.read(32)   # way C
        assert cache.stats.misses == 3
        # All three should still be resident -- no eviction needed yet
        # (only 3 of 4 ways used).
        assert cache.read(0) == 0 and cache.read(16) == 16 and cache.read(32) == 32
        assert cache.stats.hits == 3

    def test_four_distinct_lines_fit_without_any_eviction(self):
        cache = _cache(size_bytes=64, line_size=16, associativity=4,
                        memory=self._populated_memory())
        for address in (0, 16, 32, 48):
            cache.read(address)
        assert cache.stats.misses == 4
        for address in (0, 16, 32, 48):
            cache.read(address)
        assert cache.stats.hits == 4  # every one still resident


class TestLRUReplacement:
    """Section 48: on replacement, an invalid line first; otherwise the
    valid line with the smallest last_used (genuinely least recently
    used).
    """

    def _four_way_cache(self):
        memory = Memory(size_bytes=1024)
        for address in range(1024):
            memory.write_byte(address, address % 256)
        return _cache(size_bytes=64, line_size=16, associativity=4, memory=memory)

    def test_empty_lines_are_used_before_anything_is_evicted(self):
        cache = self._four_way_cache()
        for address in (0, 16, 32, 48):
            cache.read(address)
        assert cache.stats.misses == 4
        assert all(line.valid for line in cache.sets[0])

    def test_least_recently_used_line_is_evicted_first(self):
        cache = self._four_way_cache()
        # Fill all four ways, in order: 0, 16, 32, 48.
        for address in (0, 16, 32, 48):
            cache.read(address)
        # Touch 0, 32, 48 again (in that order) -- 16 is now the LRU.
        cache.read(0)
        cache.read(32)
        cache.read(48)
        # A 5th distinct line in the same set must evict address 16's line.
        cache.read(64)
        assert cache.stats.misses == 5
        # 0/32/48 are still resident; 16 was evicted.
        hits_before = cache.stats.hits
        cache.read(0); cache.read(32); cache.read(48)
        assert cache.stats.hits == hits_before + 3
        cache.read(16)  # fresh miss -- it was evicted
        assert cache.stats.misses == 6

    def test_touching_a_line_protects_it_from_the_next_eviction(self):
        cache = self._four_way_cache()
        for address in (0, 16, 32, 48):
            cache.read(address)
        cache.read(0)  # 0 is now MRU; 16 is now LRU
        cache.read(64)  # evicts 16, not 0
        assert cache.read(0) == 0
        misses_before = cache.stats.misses
        cache.read(0)
        assert cache.stats.misses == misses_before  # still a hit


class TestFullyAssociativeDoesNotCrash:
    """A regression guard: a fully-associative cache (associativity ==
    num_lines) has exactly one set, so index_bits == 0 -- a genuinely
    valid configuration that `extract_bits` alone can't represent
    (it refuses a 0-bit field), which `split_address` must still handle.
    """

    def test_index_bits_can_legitimately_be_zero(self):
        cache = _cache(size_bytes=64, line_size=16, associativity=4)
        assert cache.index_bits == 0
        assert cache.num_sets == 1

    def test_reads_work_normally_with_zero_index_bits(self):
        memory = Memory(size_bytes=64)
        for address in range(64):
            memory.write_byte(address, address)
        cache = _cache(size_bytes=64, line_size=16, associativity=4, memory=memory)
        assert cache.read(5) == 5
        assert cache.read(5) == 5  # hit, no crash


class TestWriteBack:
    def _cache_and_memory(self, size_bytes=64, line_size=16, associativity=1, memory_size=256):
        memory = Memory(size_bytes=memory_size)
        for address in range(memory_size):
            memory.write_byte(address, address % 256)
        cache = _cache(size_bytes=size_bytes, line_size=line_size,
                        associativity=associativity, memory=memory)
        return cache, memory

    def test_a_write_miss_allocates_the_line_and_marks_it_dirty(self):
        cache, _ = self._cache_and_memory()
        cache.write(5, 0xAB)
        assert cache.stats.misses == 1
        line = cache.sets[cache.split_address(5)[1]][0]
        assert line.valid is True
        assert line.dirty is True
        assert line.data[5 % 16] == 0xAB

    def test_a_write_hit_modifies_the_cached_byte_without_touching_memory(self):
        cache, memory = self._cache_and_memory()
        cache.read(5)          # install the line (clean)
        cache.write(5, 0xCD)   # hit -- modify in place
        assert cache.stats.hits == 1
        assert memory.read_byte(5) == 5  # untouched -- write-back is lazy
        line = cache.sets[cache.split_address(5)[1]][0]
        assert line.dirty is True
        assert line.data[5 % 16] == 0xCD

    def test_evicting_a_clean_line_never_touches_lower_memory(self):
        cache, memory = self._cache_and_memory()
        cache.read(5)  # clean line, never written
        cache.read(5 + cache.size_bytes)  # conflicts, evicts the clean line
        assert memory.read_byte(5) == 5  # unchanged -- nothing to write back

    def test_evicting_a_dirty_line_writes_its_old_block_back_first(self):
        cache, memory = self._cache_and_memory()
        cache.write(5, 0xAB)  # dirty
        assert memory.read_byte(5) == 5  # not yet written back
        cache.read(5 + cache.size_bytes)  # conflicting address -- evicts
        assert memory.read_byte(5) == 0xAB  # written back on eviction
        # The rest of the evicted block is untouched.
        assert memory.read_byte(4) == 4
        assert memory.read_byte(6) == 6

    def test_value_is_masked_to_a_byte(self):
        cache, _ = self._cache_and_memory()
        cache.write(5, 0x1AB)  # 0x1AB doesn't fit in a byte
        line = cache.sets[cache.split_address(5)[1]][0]
        assert line.data[5 % 16] == 0xAB  # low 8 bits only

    def test_a_subsequent_read_sees_the_written_value(self):
        cache, _ = self._cache_and_memory()
        cache.write(10, 42)
        assert cache.read(10) == 42
        assert cache.stats.hits == 1  # the read found the write's own line
