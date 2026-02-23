"""Comprehensive tests for BoundedTTLDict in aragora.server.handlers.bots.cache."""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from aragora.server.handlers.bots.cache import BoundedTTLDict


# ============================================================================
# Construction / Validation
# ============================================================================


class TestConstruction:
    """Tests for BoundedTTLDict.__init__ validation."""

    def test_valid_construction(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60, name="test")
        assert len(cache) == 0

    def test_default_name(self):
        cache = BoundedTTLDict(max_size=5, ttl_seconds=30)
        assert "cache" in repr(cache)

    def test_max_size_zero_raises(self):
        with pytest.raises(ValueError, match="max_size must be >= 1"):
            BoundedTTLDict(max_size=0, ttl_seconds=60)

    def test_max_size_negative_raises(self):
        with pytest.raises(ValueError, match="max_size must be >= 1"):
            BoundedTTLDict(max_size=-5, ttl_seconds=60)

    def test_ttl_zero_raises(self):
        with pytest.raises(ValueError, match="ttl_seconds must be > 0"):
            BoundedTTLDict(max_size=10, ttl_seconds=0)

    def test_ttl_negative_raises(self):
        with pytest.raises(ValueError, match="ttl_seconds must be > 0"):
            BoundedTTLDict(max_size=10, ttl_seconds=-1)

    def test_max_size_one_is_valid(self):
        cache = BoundedTTLDict(max_size=1, ttl_seconds=1)
        assert len(cache) == 0

    def test_large_max_size(self):
        cache = BoundedTTLDict(max_size=1_000_000, ttl_seconds=3600)
        assert len(cache) == 0

    def test_custom_name_in_repr(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60, name="my_cache")
        assert "my_cache" in repr(cache)


# ============================================================================
# __setitem__ / __getitem__
# ============================================================================


class TestSetAndGet:
    """Tests for setting and getting items."""

    def test_set_and_get_basic(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["key1"] = "value1"
        assert cache["key1"] == "value1"

    def test_set_overwrites_existing(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["key1"] = "value1"
        cache["key1"] = "value2"
        assert cache["key1"] == "value2"

    def test_get_missing_key_raises_key_error(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        with pytest.raises(KeyError):
            cache["nonexistent"]

    def test_set_multiple_keys(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        for i in range(5):
            cache[f"key{i}"] = f"value{i}"
        for i in range(5):
            assert cache[f"key{i}"] == f"value{i}"

    def test_set_various_value_types(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["int"] = 42
        cache["float"] = 3.14
        cache["list"] = [1, 2, 3]
        cache["dict"] = {"nested": True}
        cache["none"] = None
        cache["bool"] = True
        assert cache["int"] == 42
        assert cache["float"] == 3.14
        assert cache["list"] == [1, 2, 3]
        assert cache["dict"] == {"nested": True}
        assert cache["none"] is None
        assert cache["bool"] is True

    def test_set_empty_string_key(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache[""] = "empty_key"
        assert cache[""] == "empty_key"

    def test_set_long_key(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        long_key = "k" * 10000
        cache[long_key] = "long"
        assert cache[long_key] == "long"


# ============================================================================
# TTL Expiration
# ============================================================================


class TestTTLExpiration:
    """Tests for time-to-live expiration behavior."""

    def test_expired_key_raises_on_getitem(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=1)
        cache["key1"] = "value1"
        # Manipulate the timestamp to simulate expiration
        with cache._lock:
            val, _ = cache._data["key1"]
            cache._data["key1"] = (val, time.monotonic() - 2)
        with pytest.raises(KeyError):
            cache["key1"]

    def test_expired_key_returns_default_on_get(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=1)
        cache["key1"] = "value1"
        with cache._lock:
            val, _ = cache._data["key1"]
            cache._data["key1"] = (val, time.monotonic() - 2)
        assert cache.get("key1") is None
        assert cache.get("key1", "default") == "default"

    def test_expired_key_not_in_contains(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=1)
        cache["key1"] = "value1"
        assert "key1" in cache
        with cache._lock:
            val, _ = cache._data["key1"]
            cache._data["key1"] = (val, time.monotonic() - 2)
        assert "key1" not in cache

    def test_expired_entries_cleaned_on_len(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=1)
        cache["key1"] = "value1"
        cache["key2"] = "value2"
        with cache._lock:
            for k in list(cache._data.keys()):
                val, _ = cache._data[k]
                cache._data[k] = (val, time.monotonic() - 2)
        assert len(cache) == 0

    def test_non_expired_key_survives(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=3600)
        cache["key1"] = "value1"
        assert cache["key1"] == "value1"
        assert "key1" in cache

    def test_mixed_expired_and_live_keys(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["live"] = "alive"
        cache["dead"] = "gone"
        with cache._lock:
            val, _ = cache._data["dead"]
            cache._data["dead"] = (val, time.monotonic() - 120)
        assert "live" in cache
        assert "dead" not in cache
        assert cache.get("live") == "alive"
        assert cache.get("dead") is None

    def test_real_ttl_expiration(self):
        """Test actual time-based expiration with a very short TTL."""
        cache = BoundedTTLDict(max_size=10, ttl_seconds=1)
        cache["key1"] = "value1"
        assert cache["key1"] == "value1"
        time.sleep(1.1)
        with pytest.raises(KeyError):
            cache["key1"]

    def test_entry_just_before_expiry_is_accessible(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["key1"] = "value1"
        # Set timestamp just barely within TTL
        with cache._lock:
            val, _ = cache._data["key1"]
            cache._data["key1"] = (val, time.monotonic() - 59.9)
        assert cache["key1"] == "value1"

    def test_entry_at_exact_ttl_boundary_is_expired(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["key1"] = "value1"
        with cache._lock:
            val, _ = cache._data["key1"]
            cache._data["key1"] = (val, time.monotonic() - 60)
        with pytest.raises(KeyError):
            cache["key1"]


# ============================================================================
# LRU Eviction
# ============================================================================


class TestLRUEviction:
    """Tests for LRU eviction when cache is full."""

    def test_evicts_oldest_when_full(self):
        cache = BoundedTTLDict(max_size=3, ttl_seconds=60)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        # Full now; adding d should evict a
        cache["d"] = 4
        assert "a" not in cache
        assert cache["b"] == 2
        assert cache["c"] == 3
        assert cache["d"] == 4

    def test_evicts_multiple_when_full(self):
        cache = BoundedTTLDict(max_size=2, ttl_seconds=60)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        # a should be evicted, b and c remain
        assert "a" not in cache
        assert cache["b"] == 2
        assert cache["c"] == 3

    def test_max_size_one_only_keeps_latest(self):
        cache = BoundedTTLDict(max_size=1, ttl_seconds=60)
        cache["a"] = 1
        assert cache["a"] == 1
        cache["b"] = 2
        assert "a" not in cache
        assert cache["b"] == 2

    def test_overwrite_does_not_increase_size(self):
        cache = BoundedTTLDict(max_size=3, ttl_seconds=60)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        # Overwrite b: should NOT evict a
        cache["b"] = 20
        assert cache["a"] == 1
        assert cache["b"] == 20
        assert cache["c"] == 3
        assert len(cache) == 3

    def test_overwrite_moves_key_to_end(self):
        cache = BoundedTTLDict(max_size=3, ttl_seconds=60)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        # Re-insert a: a is now newest
        cache["a"] = 10
        # Adding d: oldest non-'a' entry (b) should be evicted
        cache["d"] = 4
        assert "b" not in cache
        assert cache["a"] == 10
        assert cache["c"] == 3
        assert cache["d"] == 4

    def test_eviction_respects_insertion_order(self):
        cache = BoundedTTLDict(max_size=3, ttl_seconds=60)
        cache["first"] = 1
        cache["second"] = 2
        cache["third"] = 3
        cache["fourth"] = 4
        # "first" was oldest, should be gone
        assert "first" not in cache
        assert len(cache) == 3

    def test_capacity_after_delete_allows_insert_without_eviction(self):
        cache = BoundedTTLDict(max_size=2, ttl_seconds=60)
        cache["a"] = 1
        cache["b"] = 2
        del cache["a"]
        cache["c"] = 3
        # b should still be there since we had room
        assert cache["b"] == 2
        assert cache["c"] == 3


# ============================================================================
# __contains__
# ============================================================================


class TestContains:
    """Tests for __contains__ (in operator)."""

    def test_contains_present_key(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["key1"] = "value1"
        assert "key1" in cache

    def test_contains_missing_key(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        assert "key1" not in cache

    def test_contains_non_string_key_returns_false(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        assert 123 not in cache
        assert None not in cache
        assert 3.14 not in cache
        assert [] not in cache

    def test_contains_after_delete(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["key1"] = "value1"
        del cache["key1"]
        assert "key1" not in cache


# ============================================================================
# __delitem__
# ============================================================================


class TestDelete:
    """Tests for __delitem__."""

    def test_delete_existing_key(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["key1"] = "value1"
        del cache["key1"]
        assert "key1" not in cache

    def test_delete_missing_key_raises(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        with pytest.raises(KeyError):
            del cache["nonexistent"]

    def test_delete_reduces_len(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["a"] = 1
        cache["b"] = 2
        del cache["a"]
        assert len(cache) == 1

    def test_double_delete_raises(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["key1"] = "value1"
        del cache["key1"]
        with pytest.raises(KeyError):
            del cache["key1"]


# ============================================================================
# __len__
# ============================================================================


class TestLen:
    """Tests for __len__."""

    def test_len_empty(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        assert len(cache) == 0

    def test_len_after_inserts(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        for i in range(5):
            cache[f"key{i}"] = i
        assert len(cache) == 5

    def test_len_after_overwrite(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["a"] = 1
        cache["a"] = 2
        assert len(cache) == 1

    def test_len_evicts_expired_on_call(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=1)
        cache["a"] = 1
        cache["b"] = 2
        with cache._lock:
            val, _ = cache._data["a"]
            cache._data["a"] = (val, time.monotonic() - 2)
        # len triggers cleanup, only b remains
        assert len(cache) == 1

    def test_len_at_max_size(self):
        cache = BoundedTTLDict(max_size=3, ttl_seconds=60)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        assert len(cache) == 3

    def test_len_after_eviction(self):
        cache = BoundedTTLDict(max_size=2, ttl_seconds=60)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        assert len(cache) == 2


# ============================================================================
# __bool__
# ============================================================================


class TestBool:
    """Tests for __bool__."""

    def test_empty_is_falsy(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        assert not cache

    def test_nonempty_is_truthy(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["key1"] = "value1"
        assert cache

    def test_bool_false_after_all_expired(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=1)
        cache["key1"] = "value1"
        with cache._lock:
            val, _ = cache._data["key1"]
            cache._data["key1"] = (val, time.monotonic() - 2)
        assert not cache

    def test_bool_true_with_mixed_expired(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["live"] = 1
        cache["dead"] = 2
        with cache._lock:
            val, _ = cache._data["dead"]
            cache._data["dead"] = (val, time.monotonic() - 120)
        assert cache  # still has "live"


# ============================================================================
# __iter__
# ============================================================================


class TestIter:
    """Tests for __iter__."""

    def test_iter_empty(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        assert list(cache) == []

    def test_iter_returns_keys(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        keys = list(cache)
        assert set(keys) == {"a", "b", "c"}

    def test_iter_excludes_expired(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["live"] = 1
        cache["dead"] = 2
        with cache._lock:
            val, _ = cache._data["dead"]
            cache._data["dead"] = (val, time.monotonic() - 120)
        keys = list(cache)
        assert keys == ["live"]

    def test_iter_snapshot_safety(self):
        """Iter returns a snapshot, so modifying during iteration is safe."""
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["a"] = 1
        cache["b"] = 2
        for key in cache:
            # This should not raise
            cache[key] = cache.get(key, 0)


# ============================================================================
# get()
# ============================================================================


class TestGet:
    """Tests for the get() method."""

    def test_get_existing_key(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["key1"] = "value1"
        assert cache.get("key1") == "value1"

    def test_get_missing_key_returns_none(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        assert cache.get("missing") is None

    def test_get_missing_key_returns_default(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        assert cache.get("missing", "fallback") == "fallback"

    def test_get_expired_key_returns_none(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=1)
        cache["key1"] = "value1"
        with cache._lock:
            val, _ = cache._data["key1"]
            cache._data["key1"] = (val, time.monotonic() - 2)
        assert cache.get("key1") is None

    def test_get_expired_key_returns_default(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=1)
        cache["key1"] = "value1"
        with cache._lock:
            val, _ = cache._data["key1"]
            cache._data["key1"] = (val, time.monotonic() - 2)
        assert cache.get("key1", "default") == "default"

    def test_get_with_none_default(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        assert cache.get("missing", None) is None


# ============================================================================
# pop()
# ============================================================================


class TestPop:
    """Tests for the pop() method."""

    def test_pop_existing_key(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["key1"] = "value1"
        assert cache.pop("key1") == "value1"
        assert "key1" not in cache

    def test_pop_missing_key_raises(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        with pytest.raises(KeyError):
            cache.pop("nonexistent")

    def test_pop_missing_key_with_default(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        assert cache.pop("nonexistent", "default") == "default"

    def test_pop_expired_key_raises(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=1)
        cache["key1"] = "value1"
        with cache._lock:
            val, _ = cache._data["key1"]
            cache._data["key1"] = (val, time.monotonic() - 2)
        with pytest.raises(KeyError):
            cache.pop("key1")

    def test_pop_expired_key_with_default(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=1)
        cache["key1"] = "value1"
        with cache._lock:
            val, _ = cache._data["key1"]
            cache._data["key1"] = (val, time.monotonic() - 2)
        assert cache.pop("key1", "default") == "default"

    def test_pop_removes_key(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["key1"] = "value1"
        cache.pop("key1")
        assert len(cache) == 0

    def test_pop_with_none_default(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        assert cache.pop("missing", None) is None


# ============================================================================
# values(), items(), keys()
# ============================================================================


class TestCollections:
    """Tests for values(), items(), and keys() methods."""

    def test_values_returns_list(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["a"] = 1
        cache["b"] = 2
        vals = cache.values()
        assert isinstance(vals, list)
        assert set(vals) == {1, 2}

    def test_values_excludes_expired(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["live"] = 1
        cache["dead"] = 2
        with cache._lock:
            val, _ = cache._data["dead"]
            cache._data["dead"] = (val, time.monotonic() - 120)
        assert cache.values() == [1]

    def test_values_empty_cache(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        assert cache.values() == []

    def test_items_returns_list_of_tuples(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["a"] = 1
        cache["b"] = 2
        items = cache.items()
        assert isinstance(items, list)
        assert set(items) == {("a", 1), ("b", 2)}

    def test_items_excludes_expired(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["live"] = 1
        cache["dead"] = 2
        with cache._lock:
            val, _ = cache._data["dead"]
            cache._data["dead"] = (val, time.monotonic() - 120)
        assert cache.items() == [("live", 1)]

    def test_items_empty_cache(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        assert cache.items() == []

    def test_keys_returns_list(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["a"] = 1
        cache["b"] = 2
        keys = cache.keys()
        assert isinstance(keys, list)
        assert set(keys) == {"a", "b"}

    def test_keys_excludes_expired(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["live"] = 1
        cache["dead"] = 2
        with cache._lock:
            val, _ = cache._data["dead"]
            cache._data["dead"] = (val, time.monotonic() - 120)
        assert cache.keys() == ["live"]

    def test_keys_empty_cache(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        assert cache.keys() == []


# ============================================================================
# clear()
# ============================================================================


class TestClear:
    """Tests for the clear() method."""

    def test_clear_empties_cache(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        for i in range(5):
            cache[f"key{i}"] = i
        cache.clear()
        assert len(cache) == 0

    def test_clear_empties_heap(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["a"] = 1
        cache["b"] = 2
        cache.clear()
        assert len(cache._expiry_heap) == 0

    def test_clear_allows_reinsertion(self):
        cache = BoundedTTLDict(max_size=2, ttl_seconds=60)
        cache["a"] = 1
        cache["b"] = 2
        cache.clear()
        cache["c"] = 3
        cache["d"] = 4
        assert cache["c"] == 3
        assert cache["d"] == 4

    def test_clear_on_empty_cache(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache.clear()  # should not raise
        assert len(cache) == 0


# ============================================================================
# cleanup()
# ============================================================================


class TestCleanup:
    """Tests for the cleanup() method."""

    def test_cleanup_returns_zero_when_nothing_expired(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["a"] = 1
        cache["b"] = 2
        assert cache.cleanup() == 0

    def test_cleanup_returns_count_of_removed(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=1)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        with cache._lock:
            for k in list(cache._data.keys()):
                val, _ = cache._data[k]
                cache._data[k] = (val, time.monotonic() - 2)
        assert cache.cleanup() == 3
        assert len(cache) == 0

    def test_cleanup_partial_expiration(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["live1"] = 1
        cache["dead1"] = 2
        cache["live2"] = 3
        cache["dead2"] = 4
        with cache._lock:
            for k in ["dead1", "dead2"]:
                val, _ = cache._data[k]
                cache._data[k] = (val, time.monotonic() - 120)
        removed = cache.cleanup()
        assert removed == 2
        assert len(cache) == 2

    def test_cleanup_on_empty_returns_zero(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        assert cache.cleanup() == 0


# ============================================================================
# __repr__
# ============================================================================


class TestRepr:
    """Tests for __repr__."""

    def test_repr_format(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60, name="test_repr")
        r = repr(cache)
        assert "BoundedTTLDict" in r
        assert "test_repr" in r
        assert "max_size=10" in r
        assert "ttl_seconds=60" in r

    def test_repr_shows_current_size(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60, name="sz")
        cache["a"] = 1
        cache["b"] = 2
        assert "size=2" in repr(cache)

    def test_repr_empty_cache(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        assert "size=0" in repr(cache)


# ============================================================================
# Heap-based eviction internals
# ============================================================================


class TestHeapEviction:
    """Tests for the min-heap based expiration optimization."""

    def test_heap_grows_on_insert(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["a"] = 1
        assert len(cache._expiry_heap) >= 1

    def test_stale_heap_entries_handled(self):
        """When a key is updated, old heap entries become stale and are skipped."""
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["a"] = 1
        heap_size_after_first = len(cache._expiry_heap)
        cache["a"] = 2  # re-insert: old heap entry becomes stale
        # Heap has at least 2 entries for key "a"
        assert len(cache._expiry_heap) >= heap_size_after_first
        # But the value is correct
        assert cache["a"] == 2

    def test_heap_cleanup_skips_already_deleted_keys(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=1)
        cache["a"] = 1
        # Delete the key directly, heap still has the entry
        with cache._lock:
            del cache._data["a"]
        # Force cleanup via len() - should not raise
        assert len(cache) == 0

    def test_heap_cleanup_skips_updated_keys(self):
        """Heap entry with old timestamp should be skipped if key was updated."""
        cache = BoundedTTLDict(max_size=10, ttl_seconds=2)
        cache["a"] = 1
        # Artificially expire the first entry
        with cache._lock:
            val, _ = cache._data["a"]
            old_ts = time.monotonic() - 3
            cache._data["a"] = (val, old_ts)
        # Now update the key (creates new timestamp and new heap entry)
        cache["a"] = 2
        # The key should be accessible since it was just re-inserted
        assert cache["a"] == 2


# ============================================================================
# Thread Safety
# ============================================================================


class TestThreadSafety:
    """Tests for concurrent access thread safety."""

    def test_concurrent_writes(self):
        cache = BoundedTTLDict(max_size=100, ttl_seconds=60)
        errors = []

        def writer(thread_id):
            try:
                for i in range(50):
                    cache[f"t{thread_id}_k{i}"] = f"v{i}"
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(cache) <= 100

    def test_concurrent_reads_and_writes(self):
        cache = BoundedTTLDict(max_size=50, ttl_seconds=60)
        errors = []

        def writer():
            try:
                for i in range(50):
                    cache[f"key{i}"] = i
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(50):
                    cache.get(f"key{i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=writer))
            threads.append(threading.Thread(target=reader))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_concurrent_deletes(self):
        cache = BoundedTTLDict(max_size=100, ttl_seconds=60)
        for i in range(100):
            cache[f"key{i}"] = i
        errors = []

        def deleter(start):
            for i in range(start, 100, 4):
                try:
                    del cache[f"key{i}"]
                except KeyError:
                    pass  # Another thread may have deleted it
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=deleter, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(cache) == 0

    def test_concurrent_mixed_operations(self):
        cache = BoundedTTLDict(max_size=50, ttl_seconds=60)
        errors = []

        def mixed_ops(thread_id):
            try:
                for i in range(30):
                    key = f"t{thread_id}_k{i}"
                    cache[key] = i
                    _ = cache.get(key)
                    _ = key in cache
                    _ = len(cache)
                    cache.pop(key, None)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=mixed_ops, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_set_after_expired_key_freed_space(self):
        """Expired entries are cleaned before checking capacity."""
        cache = BoundedTTLDict(max_size=2, ttl_seconds=1)
        cache["a"] = 1
        cache["b"] = 2
        # Expire both
        with cache._lock:
            for k in list(cache._data.keys()):
                val, _ = cache._data[k]
                cache._data[k] = (val, time.monotonic() - 2)
        # Now set a new key: expired entries should be cleaned first
        cache["c"] = 3
        assert cache["c"] == 3
        assert len(cache) == 1

    def test_repeated_set_and_delete_cycle(self):
        cache = BoundedTTLDict(max_size=5, ttl_seconds=60)
        for cycle in range(10):
            cache["key"] = cycle
            assert cache["key"] == cycle
            del cache["key"]
        assert len(cache) == 0

    def test_large_number_of_entries(self):
        cache = BoundedTTLDict(max_size=1000, ttl_seconds=60)
        for i in range(1000):
            cache[f"key{i}"] = i
        assert len(cache) == 1000
        # Adding one more evicts the oldest
        cache["overflow"] = "new"
        assert len(cache) == 1000

    def test_all_methods_on_empty_cache(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        assert len(cache) == 0
        assert not cache
        assert list(cache) == []
        assert cache.get("x") is None
        assert cache.pop("x", "d") == "d"
        assert cache.values() == []
        assert cache.items() == []
        assert cache.keys() == []
        assert cache.cleanup() == 0
        cache.clear()

    def test_is_expired_on_missing_key(self):
        """_is_expired returns True for a key not in data."""
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        with cache._lock:
            assert cache._is_expired("nonexistent") is True

    def test_evict_oldest_zero_count(self):
        """_evict_oldest with count <= 0 is a no-op."""
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["a"] = 1
        with cache._lock:
            cache._evict_oldest(0)
            cache._evict_oldest(-1)
        assert "a" in cache

    def test_none_value_stored_correctly(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["key"] = None
        assert cache["key"] is None
        assert cache.get("key") is None
        assert cache.get("key", "sentinel") is None

    def test_pop_then_set_same_key(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["key"] = "first"
        popped = cache.pop("key")
        assert popped == "first"
        cache["key"] = "second"
        assert cache["key"] == "second"

    def test_cleanup_idempotent(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=1)
        cache["a"] = 1
        with cache._lock:
            val, _ = cache._data["a"]
            cache._data["a"] = (val, time.monotonic() - 2)
        first = cache.cleanup()
        second = cache.cleanup()
        assert first == 1
        assert second == 0

    def test_get_default_none_vs_stored_none(self):
        """Distinguish between missing key (returns default) and stored None."""
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        sentinel = object()
        cache["has_none"] = None
        # get with sentinel default: stored None is returned, not sentinel
        assert cache.get("has_none", sentinel) is None
        # Missing key returns sentinel
        assert cache.get("missing", sentinel) is sentinel

    def test_ttl_seconds_as_float_like_int(self):
        """ttl_seconds > 0 check works for integer values."""
        cache = BoundedTTLDict(max_size=5, ttl_seconds=1)
        cache["k"] = "v"
        assert cache["k"] == "v"


# ============================================================================
# _evict_expired internals
# ============================================================================


class TestEvictExpired:
    """Tests for the _evict_expired internal method."""

    def test_evict_expired_removes_all_expired(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=1)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        with cache._lock:
            now = time.monotonic()
            for k in list(cache._data.keys()):
                val, _ = cache._data[k]
                cache._data[k] = (val, now - 2)
            cache._evict_expired()
            assert len(cache._data) == 0

    def test_evict_expired_keeps_live_entries(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        cache["a"] = 1
        cache["b"] = 2
        with cache._lock:
            cache._evict_expired()
            assert len(cache._data) == 2

    def test_evict_expired_handles_empty_heap(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=60)
        with cache._lock:
            cache._expiry_heap.clear()
            cache._evict_expired()  # should not raise

    def test_evict_expired_fallback_scan(self):
        """Entries modified outside of __setitem__ are caught by fallback scan."""
        cache = BoundedTTLDict(max_size=10, ttl_seconds=1)
        # Insert normally
        cache["a"] = 1
        # Clear the heap so only fallback scan catches expiration
        with cache._lock:
            cache._expiry_heap.clear()
            val, _ = cache._data["a"]
            cache._data["a"] = (val, time.monotonic() - 2)
            cache._evict_expired()
            assert "a" not in cache._data


# ============================================================================
# Logging
# ============================================================================


class TestLogging:
    """Tests for logging behavior."""

    def test_eviction_logs_debug(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=1, name="log_test")
        cache["a"] = 1
        with cache._lock:
            val, _ = cache._data["a"]
            cache._data["a"] = (val, time.monotonic() - 2)
        with patch("aragora.server.handlers.bots.cache.logger") as mock_logger:
            cache.cleanup()
            mock_logger.debug.assert_called()

    def test_lru_eviction_logs_debug(self):
        cache = BoundedTTLDict(max_size=1, ttl_seconds=60, name="lru_test")
        cache["a"] = 1
        with patch("aragora.server.handlers.bots.cache.logger") as mock_logger:
            cache["b"] = 2  # triggers LRU eviction of "a"
            mock_logger.debug.assert_called()

    def test_expired_read_logs_debug(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=1, name="read_test")
        cache["a"] = 1
        with cache._lock:
            val, _ = cache._data["a"]
            cache._data["a"] = (val, time.monotonic() - 2)
        with patch("aragora.server.handlers.bots.cache.logger") as mock_logger:
            with pytest.raises(KeyError):
                cache["a"]
            mock_logger.debug.assert_called()

    def test_expired_get_logs_debug(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=1, name="get_test")
        cache["a"] = 1
        with cache._lock:
            val, _ = cache._data["a"]
            cache._data["a"] = (val, time.monotonic() - 2)
        with patch("aragora.server.handlers.bots.cache.logger") as mock_logger:
            cache.get("a")
            mock_logger.debug.assert_called()

    def test_expired_contains_logs_debug(self):
        cache = BoundedTTLDict(max_size=10, ttl_seconds=1, name="contains_test")
        cache["a"] = 1
        with cache._lock:
            val, _ = cache._data["a"]
            cache._data["a"] = (val, time.monotonic() - 2)
        with patch("aragora.server.handlers.bots.cache.logger") as mock_logger:
            _ = "a" in cache
            mock_logger.debug.assert_called()


# ============================================================================
# Module-level exports
# ============================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports_bounded_ttl_dict(self):
        from aragora.server.handlers.bots import cache

        assert "BoundedTTLDict" in cache.__all__

    def test_import_from_module(self):
        from aragora.server.handlers.bots.cache import BoundedTTLDict as Cls

        assert Cls is BoundedTTLDict
