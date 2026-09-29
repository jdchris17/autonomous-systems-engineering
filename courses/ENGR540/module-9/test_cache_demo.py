import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cache_demo import (
    run_address_breakdown_demo,
    run_lru_demo,
    run_read_path_demo,
    run_write_back_demo,
)


class TestAddressBreakdownDemo:
    def test_matches_the_spec_worked_example(self):
        cache = run_address_breakdown_demo()
        assert cache.num_lines == 64
        assert cache.num_sets == 64
        assert cache.offset_bits == 4
        assert cache.index_bits == 6
        assert cache.tag_bits == 22


class TestReadPathDemo:
    def test_ends_with_three_misses_and_one_hit(self):
        cache = run_read_path_demo()
        assert cache.stats.accesses == 4
        assert cache.stats.hits == 1
        assert cache.stats.misses == 3


class TestLRUDemo:
    def test_the_genuinely_least_recently_used_line_is_evicted(self):
        cache = run_lru_demo()
        assert cache.stats.misses == 6  # 4 initial + 1 conflict + 1 re-fetch of the evicted line
        assert cache.stats.hits == 6    # 3 re-touches + 3 confirmations


class TestWriteBackDemo:
    def test_write_back_is_lazy_until_eviction(self):
        cache, memory = run_write_back_demo()
        assert memory.read_byte(5) == 0xAB
        assert cache.stats.misses == 2  # the write-allocate + the evicting read
        assert cache.stats.hits == 1    # the read that saw the write's own value
