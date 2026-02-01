"""Tests for BoundedTTLDict - bounded, thread-safe TTL cache for bot handlers.

Verifies:
- Maximum size enforcement with LRU eviction
- TTL expiration of entries
- Thread safety under concurrent access
- All dict-like operations (__contains__, get, pop, del, values, items, keys, len)
- cleanup() manual eviction
- Edge cases (empty cache, single item, re-insertion, etc.)
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from aragora.server.handlers.bots.cache import BoundedTTLDict


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_valid_construction(self) -> None:
        cache = BoundedTTLDict(max_size=100, ttl_seconds=60, name="test")
        assert len(cache) == 0
        assert repr(cache) == "BoundedTTLDict(name='test', size=0, max_size=100, ttl_seconds=60)"

    def test_default_name(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=10)
        assert "cache" in repr(cache)

    def test_invalid_max_size_zero(self) -> None:
        with pytest.raises(ValueError, match="max_size must be >= 1"):
            BoundedTTLDict(max_size=0, ttl_seconds=60)

    def test_invalid_max_size_negative(self) -> None:
        with pytest.raises(ValueError, match="max_size must be >= 1"):
            BoundedTTLDict(max_size=-5, ttl_seconds=60)

    def test_invalid_ttl_zero(self) -> None:
        with pytest.raises(ValueError, match="ttl_seconds must be > 0"):
            BoundedTTLDict(max_size=10, ttl_seconds=0)

    def test_invalid_ttl_negative(self) -> None:
        with pytest.raises(ValueError, match="ttl_seconds must be > 0"):
            BoundedTTLDict(max_size=10, ttl_seconds=-1)


# ---------------------------------------------------------------------------
# Basic dict operations
# ---------------------------------------------------------------------------


class TestBasicOperations:
    def test_setitem_getitem(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["a"] = 1
        assert cache["a"] == 1

    def test_getitem_missing_key_raises(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        with pytest.raises(KeyError):
            _ = cache["missing"]

    def test_contains(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["x"] = 42
        assert "x" in cache
        assert "y" not in cache

    def test_delitem(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["k"] = "v"
        del cache["k"]
        assert "k" not in cache

    def test_delitem_missing_key_raises(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        with pytest.raises(KeyError):
            del cache["missing"]

    def test_len(self) -> None:
        cache = BoundedTTLDict(max_size=100, ttl_seconds=60)
        assert len(cache) == 0
        cache["a"] = 1
        cache["b"] = 2
        assert len(cache) == 2

    def test_bool_empty(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        assert not cache

    def test_bool_nonempty(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["a"] = 1
        assert cache

    def test_iter(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["a"] = 1
        cache["b"] = 2
        assert sorted(cache) == ["a", "b"]

    def test_get_existing(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["k"] = "val"
        assert cache.get("k") == "val"

    def test_get_missing_returns_default(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        assert cache.get("missing") is None
        assert cache.get("missing", "fallback") == "fallback"

    def test_pop_existing(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["k"] = "val"
        result = cache.pop("k")
        assert result == "val"
        assert "k" not in cache

    def test_pop_missing_with_default(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        assert cache.pop("missing", "default") == "default"

    def test_pop_missing_without_default_raises(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        with pytest.raises(KeyError):
            cache.pop("missing")

    def test_values(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["a"] = 1
        cache["b"] = 2
        assert sorted(cache.values()) == [1, 2]

    def test_items(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["a"] = 1
        cache["b"] = 2
        assert sorted(cache.items()) == [("a", 1), ("b", 2)]

    def test_keys(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["a"] = 1
        cache["b"] = 2
        assert sorted(cache.keys()) == ["a", "b"]

    def test_overwrite_existing_key(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["a"] = 1
        cache["a"] = 2
        assert cache["a"] == 2
        assert len(cache) == 1

    def test_stores_complex_values(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        value = {"topic": "test", "votes": [1, 2, 3], "nested": {"a": True}}
        cache["k"] = value
        assert cache["k"] == value


# ---------------------------------------------------------------------------
# Max size enforcement (LRU eviction)
# ---------------------------------------------------------------------------


class TestMaxSizeEviction:
    def test_evicts_oldest_when_full(self) -> None:
        cache = BoundedTTLDict(max_size=3, ttl_seconds=60)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        # Cache is at capacity; inserting "d" should evict "a" (oldest)
        cache["d"] = 4
        assert "a" not in cache
        assert cache["b"] == 2
        assert cache["c"] == 3
        assert cache["d"] == 4

    def test_evicts_multiple_if_needed(self) -> None:
        cache = BoundedTTLDict(max_size=2, ttl_seconds=60)
        cache["a"] = 1
        cache["b"] = 2
        # At capacity; inserting should evict oldest
        cache["c"] = 3
        assert "a" not in cache
        assert len(cache) == 2

    def test_max_size_one(self) -> None:
        cache = BoundedTTLDict(max_size=1, ttl_seconds=60)
        cache["a"] = 1
        assert cache["a"] == 1
        cache["b"] = 2
        assert "a" not in cache
        assert cache["b"] == 2
        assert len(cache) == 1

    def test_overwrite_does_not_evict(self) -> None:
        cache = BoundedTTLDict(max_size=3, ttl_seconds=60)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        # Overwriting "a" should not evict anyone
        cache["a"] = 10
        assert len(cache) == 3
        assert cache["a"] == 10
        assert cache["b"] == 2
        assert cache["c"] == 3

    def test_overwrite_refreshes_insertion_order(self) -> None:
        """Re-inserting an existing key moves it to the end (newest)."""
        cache = BoundedTTLDict(max_size=3, ttl_seconds=60)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        # Refresh "a" - now "b" is oldest
        cache["a"] = 10
        cache["d"] = 4
        # "b" should have been evicted (oldest after "a" refresh)
        assert "b" not in cache
        assert cache["a"] == 10
        assert cache["c"] == 3
        assert cache["d"] == 4


# ---------------------------------------------------------------------------
# TTL expiration
# ---------------------------------------------------------------------------


class TestTTLExpiration:
    def _make_cache_with_expired_entry(self) -> BoundedTTLDict:
        """Create a cache and manually backdate an entry to simulate expiration."""
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["fresh"] = "still_good"
        cache["stale"] = "expired"
        # Backdate the "stale" entry by manipulating the internal data
        with cache._lock:
            val, _ = cache._data["stale"]
            cache._data["stale"] = (val, time.monotonic() - 120)  # 2 minutes ago
        return cache

    def test_expired_entry_not_returned_by_getitem(self) -> None:
        cache = self._make_cache_with_expired_entry()
        with pytest.raises(KeyError):
            _ = cache["stale"]

    def test_expired_entry_not_in_contains(self) -> None:
        cache = self._make_cache_with_expired_entry()
        assert "stale" not in cache

    def test_expired_entry_not_returned_by_get(self) -> None:
        cache = self._make_cache_with_expired_entry()
        assert cache.get("stale") is None
        assert cache.get("stale", "default") == "default"

    def test_expired_entry_not_returned_by_pop(self) -> None:
        cache = self._make_cache_with_expired_entry()
        assert cache.pop("stale", "default") == "default"

    def test_expired_entry_pop_no_default_raises(self) -> None:
        cache = self._make_cache_with_expired_entry()
        with pytest.raises(KeyError):
            cache.pop("stale")

    def test_expired_entries_excluded_from_len(self) -> None:
        cache = self._make_cache_with_expired_entry()
        assert len(cache) == 1  # only "fresh" remains

    def test_expired_entries_excluded_from_values(self) -> None:
        cache = self._make_cache_with_expired_entry()
        assert cache.values() == ["still_good"]

    def test_expired_entries_excluded_from_items(self) -> None:
        cache = self._make_cache_with_expired_entry()
        assert cache.items() == [("fresh", "still_good")]

    def test_expired_entries_excluded_from_keys(self) -> None:
        cache = self._make_cache_with_expired_entry()
        assert cache.keys() == ["fresh"]

    def test_expired_entries_excluded_from_bool(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["stale"] = "old"
        with cache._lock:
            val, _ = cache._data["stale"]
            cache._data["stale"] = (val, time.monotonic() - 120)
        assert not cache  # all entries expired

    def test_expired_entries_excluded_from_iter(self) -> None:
        cache = self._make_cache_with_expired_entry()
        assert list(cache) == ["fresh"]

    def test_setitem_cleans_expired_before_size_check(self) -> None:
        """Expired entries are cleaned before LRU eviction, preventing unnecessary eviction."""
        cache = BoundedTTLDict(max_size=2, ttl_seconds=60)
        cache["a"] = 1
        cache["b"] = 2
        # Expire "a"
        with cache._lock:
            val, _ = cache._data["a"]
            cache._data["a"] = (val, time.monotonic() - 120)
        # Insert "c" - should evict expired "a" first, not LRU-evict "b"
        cache["c"] = 3
        assert "a" not in cache
        assert cache["b"] == 2
        assert cache["c"] == 3


# ---------------------------------------------------------------------------
# cleanup() method
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_cleanup_removes_expired(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["a"] = 1
        cache["b"] = 2
        # Expire "a"
        with cache._lock:
            val, _ = cache._data["a"]
            cache._data["a"] = (val, time.monotonic() - 120)
        removed = cache.cleanup()
        assert removed == 1
        assert "a" not in cache
        assert cache["b"] == 2

    def test_cleanup_returns_zero_when_nothing_expired(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["a"] = 1
        assert cache.cleanup() == 0

    def test_cleanup_on_empty_cache(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        assert cache.cleanup() == 0

    def test_cleanup_removes_all_expired(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        for i in range(5):
            cache[f"k{i}"] = i
        # Expire all
        with cache._lock:
            now = time.monotonic() - 120
            for k in list(cache._data):
                val, _ = cache._data[k]
                cache._data[k] = (val, now)
        removed = cache.cleanup()
        assert removed == 5
        assert len(cache) == 0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_writes(self) -> None:
        cache = BoundedTTLDict(max_size=1000, ttl_seconds=60, name="thread_test")
        errors: list[Exception] = []

        def writer(start: int) -> None:
            try:
                for i in range(100):
                    cache[f"t{start}_{i}"] = i
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        # All 1000 keys should fit
        assert len(cache) == 1000

    def test_concurrent_reads_and_writes(self) -> None:
        cache = BoundedTTLDict(max_size=500, ttl_seconds=60, name="rw_test")
        errors: list[Exception] = []

        # Pre-populate
        for i in range(200):
            cache[f"pre_{i}"] = i

        def reader() -> None:
            try:
                for i in range(200):
                    cache.get(f"pre_{i}")
                    _ = f"pre_{i}" in cache
                    cache.items()
            except Exception as e:
                errors.append(e)

        def writer() -> None:
            try:
                for i in range(200):
                    cache[f"new_{i}"] = i
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=reader))
            threads.append(threading.Thread(target=writer))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"

    def test_concurrent_eviction(self) -> None:
        """Concurrent inserts on a small cache should not corrupt state."""
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60, name="eviction_test")
        errors: list[Exception] = []

        def inserter(thread_id: int) -> None:
            try:
                for i in range(50):
                    cache[f"t{thread_id}_{i}"] = i
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=inserter, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert len(cache) <= 10


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_cache_operations(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        assert len(cache) == 0
        assert not cache
        assert list(cache) == []
        assert cache.keys() == []
        assert cache.values() == []
        assert cache.items() == []
        assert cache.get("x") is None
        assert cache.pop("x", None) is None

    def test_single_item_cache(self) -> None:
        cache = BoundedTTLDict(max_size=1, ttl_seconds=60)
        cache["only"] = "one"
        assert len(cache) == 1
        assert cache["only"] == "one"
        assert list(cache) == ["only"]

    def test_none_value(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["k"] = None
        assert "k" in cache
        assert cache["k"] is None
        assert cache.get("k", "default") is None

    def test_empty_string_key(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache[""] = "empty_key"
        assert cache[""] == "empty_key"

    def test_repr_shows_current_state(self) -> None:
        cache = BoundedTTLDict(max_size=5, ttl_seconds=30, name="repr_test")
        cache["a"] = 1
        cache["b"] = 2
        r = repr(cache)
        assert "repr_test" in r
        assert "size=2" in r
        assert "max_size=5" in r
        assert "ttl_seconds=30" in r

    def test_large_number_of_insertions(self) -> None:
        """Ensure cache stays bounded even with many insertions."""
        cache = BoundedTTLDict(max_size=100, ttl_seconds=60)
        for i in range(10_000):
            cache[f"k{i}"] = i
        assert len(cache) == 100
        # Last 100 keys should be present
        for i in range(9_900, 10_000):
            assert f"k{i}" in cache

    def test_pop_returns_value_and_removes(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["a"] = {"data": 42}
        val = cache.pop("a")
        assert val == {"data": 42}
        assert "a" not in cache
        assert len(cache) == 0

    def test_delete_then_reinsert(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["k"] = "v1"
        del cache["k"]
        cache["k"] = "v2"
        assert cache["k"] == "v2"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestLogging:
    def test_lru_eviction_logs_debug(self) -> None:
        cache = BoundedTTLDict(max_size=2, ttl_seconds=60, name="log_test")
        cache["a"] = 1
        cache["b"] = 2
        with patch("aragora.server.handlers.bots.cache.logger") as mock_logger:
            cache["c"] = 3
            mock_logger.debug.assert_called()
            # Check that at least one call mentions the cache name
            calls = [str(c) for c in mock_logger.debug.call_args_list]
            assert any("log_test" in c for c in calls)

    def test_ttl_eviction_logs_debug(self) -> None:
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60, name="ttl_log")
        cache["stale"] = "old"
        with cache._lock:
            val, _ = cache._data["stale"]
            cache._data["stale"] = (val, time.monotonic() - 120)
        with patch("aragora.server.handlers.bots.cache.logger") as mock_logger:
            _ = cache.get("stale")
            mock_logger.debug.assert_called()
