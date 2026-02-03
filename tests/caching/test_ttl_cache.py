"""
Tests for the _TTLCache internal class.

Covers:
- Basic get/set operations
- Cache miss on empty cache
- Cache hit after set
- TTL-based expiration
- LRU eviction when cache is full
- Cache clear
- Statistics tracking (hits, misses, evictions, size)
- cache_info snapshot
- Expired entry cleanup on set
- Updating existing keys
- Edge cases: maxsize=1, missing keys, overwrite behavior
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from aragora.caching.decorators import _TTLCache, CacheStats


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def cache() -> _TTLCache:
    """Create a standard TTL cache with reasonable defaults."""
    return _TTLCache(maxsize=5, ttl_seconds=60.0)


@pytest.fixture
def tiny_cache() -> _TTLCache:
    """Create a very small cache for eviction tests."""
    return _TTLCache(maxsize=2, ttl_seconds=60.0)


@pytest.fixture
def short_ttl_cache() -> _TTLCache:
    """Create a cache with very short TTL for expiration tests."""
    return _TTLCache(maxsize=10, ttl_seconds=0.1)


# ===========================================================================
# Test: Basic Get/Set Operations
# ===========================================================================


class TestBasicGetSet:
    """Tests for basic cache get and set operations."""

    def test_get_missing_key(self, cache: _TTLCache):
        """Get on a missing key returns (False, None)."""
        found, value = cache.get("nonexistent")
        assert found is False
        assert value is None

    def test_set_and_get(self, cache: _TTLCache):
        """Set a value and retrieve it."""
        cache.set("key1", "value1")
        found, value = cache.get("key1")
        assert found is True
        assert value == "value1"

    def test_set_multiple_keys(self, cache: _TTLCache):
        """Set multiple keys and retrieve each."""
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)

        found_a, val_a = cache.get("a")
        found_b, val_b = cache.get("b")
        found_c, val_c = cache.get("c")

        assert (found_a, val_a) == (True, 1)
        assert (found_b, val_b) == (True, 2)
        assert (found_c, val_c) == (True, 3)

    def test_overwrite_existing_key(self, cache: _TTLCache):
        """Overwriting an existing key updates the value."""
        cache.set("key", "old")
        cache.set("key", "new")
        found, value = cache.get("key")
        assert found is True
        assert value == "new"

    def test_set_none_value(self, cache: _TTLCache):
        """None can be stored as a cache value."""
        cache.set("key", None)
        found, value = cache.get("key")
        assert found is True
        assert value is None

    def test_set_empty_string(self, cache: _TTLCache):
        """Empty string can be stored as a cache value."""
        cache.set("key", "")
        found, value = cache.get("key")
        assert found is True
        assert value == ""

    def test_set_zero(self, cache: _TTLCache):
        """Zero can be stored as a cache value."""
        cache.set("key", 0)
        found, value = cache.get("key")
        assert found is True
        assert value == 0

    def test_set_false(self, cache: _TTLCache):
        """False can be stored as a cache value."""
        cache.set("key", False)
        found, value = cache.get("key")
        assert found is True
        assert value is False

    def test_set_complex_value(self, cache: _TTLCache):
        """Complex objects can be stored."""
        data = {"nested": [1, 2, {"deep": True}]}
        cache.set("key", data)
        found, value = cache.get("key")
        assert found is True
        assert value == data


# ===========================================================================
# Test: TTL Expiration
# ===========================================================================


class TestTTLExpiration:
    """Tests for TTL-based entry expiration."""

    def test_entry_not_expired_within_ttl(self, cache: _TTLCache):
        """Entry is accessible within TTL window."""
        cache.set("key", "value")
        found, value = cache.get("key")
        assert found is True
        assert value == "value"

    def test_entry_expired_after_ttl(self, short_ttl_cache: _TTLCache):
        """Entry is not accessible after TTL expires."""
        short_ttl_cache.set("key", "value")
        time.sleep(0.15)  # Wait for TTL to expire
        found, value = short_ttl_cache.get("key")
        assert found is False
        assert value is None

    def test_expired_entry_removed_on_get(self, short_ttl_cache: _TTLCache):
        """Expired entry is cleaned up on access."""
        short_ttl_cache.set("key", "value")
        time.sleep(0.15)

        # Access triggers removal
        short_ttl_cache.get("key")
        info = short_ttl_cache.cache_info()
        assert info.size == 0

    def test_mixed_expired_and_valid(self):
        """Valid entries survive while expired ones are removed."""
        cache = _TTLCache(maxsize=10, ttl_seconds=0.1)
        cache.set("short_lived", "old")
        time.sleep(0.15)
        cache.set("long_lived", "new")

        found_short, _ = cache.get("short_lived")
        found_long, val_long = cache.get("long_lived")

        assert found_short is False
        assert found_long is True
        assert val_long == "new"

    def test_expired_cleanup_on_set(self):
        """Expired entries are cleaned up when setting new values."""
        cache = _TTLCache(maxsize=10, ttl_seconds=0.1)
        cache.set("old1", "val1")
        cache.set("old2", "val2")
        time.sleep(0.15)

        # Setting a new value triggers cleanup of expired entries
        cache.set("new", "val_new")
        info = cache.cache_info()
        assert info.size == 1  # Only the new entry remains


# ===========================================================================
# Test: LRU Eviction
# ===========================================================================


class TestLRUEviction:
    """Tests for least-recently-used eviction."""

    def test_eviction_at_maxsize(self, tiny_cache: _TTLCache):
        """Oldest entry is evicted when cache is full."""
        tiny_cache.set("first", 1)
        tiny_cache.set("second", 2)
        tiny_cache.set("third", 3)  # Should evict "first"

        found_first, _ = tiny_cache.get("first")
        found_third, val_third = tiny_cache.get("third")

        assert found_first is False
        assert found_third is True
        assert val_third == 3

    def test_lru_access_updates_order(self, tiny_cache: _TTLCache):
        """Accessing an entry moves it to most-recently-used position."""
        tiny_cache.set("a", 1)
        tiny_cache.set("b", 2)

        # Access "a" to make it most recently used
        tiny_cache.get("a")

        # Add "c" - should evict "b" (LRU), not "a"
        tiny_cache.set("c", 3)

        found_a, _ = tiny_cache.get("a")
        found_b, _ = tiny_cache.get("b")
        found_c, _ = tiny_cache.get("c")

        assert found_a is True
        assert found_b is False
        assert found_c is True

    def test_eviction_count_tracked(self, tiny_cache: _TTLCache):
        """Evictions are counted in stats."""
        tiny_cache.set("a", 1)
        tiny_cache.set("b", 2)
        tiny_cache.set("c", 3)  # Evicts "a"
        tiny_cache.set("d", 4)  # Evicts "b"

        info = tiny_cache.cache_info()
        assert info.evictions == 2

    def test_maxsize_one(self):
        """Cache with maxsize=1 evicts on every new unique key."""
        cache = _TTLCache(maxsize=1, ttl_seconds=60.0)
        cache.set("a", 1)
        cache.set("b", 2)

        found_a, _ = cache.get("a")
        found_b, val_b = cache.get("b")

        assert found_a is False
        assert found_b is True
        assert val_b == 2

    def test_overwrite_does_not_evict(self, tiny_cache: _TTLCache):
        """Overwriting an existing key does not trigger eviction."""
        tiny_cache.set("a", 1)
        tiny_cache.set("b", 2)
        tiny_cache.set("a", 10)  # Update, not new entry

        info = tiny_cache.cache_info()
        assert info.evictions == 0
        assert info.size == 2

    def test_eviction_preserves_most_recent(self):
        """Only least-recently-used entries are evicted."""
        cache = _TTLCache(maxsize=3, ttl_seconds=60.0)

        # Fill cache
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)

        # Access in order: a, b (c is now LRU)
        cache.get("a")
        cache.get("b")

        # Add new entry - should evict "c" (LRU)
        cache.set("d", 4)

        found_a, _ = cache.get("a")
        found_b, _ = cache.get("b")
        found_c, _ = cache.get("c")
        found_d, _ = cache.get("d")

        assert found_a is True
        assert found_b is True
        assert found_c is False
        assert found_d is True


# ===========================================================================
# Test: Cache Clear
# ===========================================================================


class TestCacheClear:
    """Tests for cache clear operation."""

    def test_clear_empties_cache(self, cache: _TTLCache):
        """Clear removes all entries."""
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()

        found_a, _ = cache.get("a")
        found_b, _ = cache.get("b")

        assert found_a is False
        assert found_b is False

    def test_clear_resets_size(self, cache: _TTLCache):
        """Clear resets size to zero."""
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()

        info = cache.cache_info()
        assert info.size == 0

    def test_clear_preserves_stats_counters(self, cache: _TTLCache):
        """Clear does not reset hit/miss/eviction counters."""
        cache.set("a", 1)
        cache.get("a")  # hit
        cache.get("missing")  # miss
        cache.clear()

        info = cache.cache_info()
        assert info.hits == 1
        assert info.misses == 1

    def test_clear_allows_reuse(self, cache: _TTLCache):
        """Cache can be used normally after clear."""
        cache.set("a", 1)
        cache.clear()
        cache.set("b", 2)

        found_b, val_b = cache.get("b")
        assert found_b is True
        assert val_b == 2


# ===========================================================================
# Test: Statistics Tracking
# ===========================================================================


class TestStatistics:
    """Tests for cache statistics tracking."""

    def test_miss_count_incremented(self, cache: _TTLCache):
        """Misses are counted correctly."""
        cache.get("a")
        cache.get("b")
        cache.get("c")

        info = cache.cache_info()
        assert info.misses == 3

    def test_hit_count_incremented(self, cache: _TTLCache):
        """Hits are counted correctly."""
        cache.set("a", 1)
        cache.get("a")
        cache.get("a")

        info = cache.cache_info()
        assert info.hits == 2

    def test_size_tracks_entries(self, cache: _TTLCache):
        """Size reflects number of entries."""
        cache.set("a", 1)
        assert cache.cache_info().size == 1

        cache.set("b", 2)
        assert cache.cache_info().size == 2

        cache.set("c", 3)
        assert cache.cache_info().size == 3

    def test_maxsize_in_stats(self, cache: _TTLCache):
        """Maxsize is reported correctly."""
        info = cache.cache_info()
        assert info.maxsize == 5

    def test_cache_info_returns_snapshot(self, cache: _TTLCache):
        """cache_info returns a snapshot, not a live reference."""
        cache.set("a", 1)
        info1 = cache.cache_info()

        cache.set("b", 2)
        info2 = cache.cache_info()

        # info1 should not have changed
        assert info1.size == 1
        assert info2.size == 2

    def test_cache_info_returns_cache_stats(self, cache: _TTLCache):
        """cache_info returns a CacheStats instance."""
        info = cache.cache_info()
        assert isinstance(info, CacheStats)

    def test_stats_after_expiration(self):
        """Expired entry access counts as a miss."""
        cache = _TTLCache(maxsize=10, ttl_seconds=0.1)
        cache.set("key", "val")
        time.sleep(0.15)
        cache.get("key")

        info = cache.cache_info()
        # One miss for initial set (no get counted as miss), one miss for expired get
        assert info.misses >= 1
