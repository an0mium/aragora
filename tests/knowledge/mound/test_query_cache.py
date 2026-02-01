"""
Comprehensive tests for aragora.knowledge.mound.query_cache module.

Tests cover:
- CacheStats dataclass (hits, misses, evictions, hit_rate, to_dict)
- CacheEntry dataclass (value storage, creation timestamp, access count)
- RequestScopedCache basic ops (get/set, hit/miss tracking)
- LRU eviction at max_size
- get_or_compute (sync) with cache hit and miss
- get_or_compute_async (async) with cache hit and miss
- Invalidation patterns (single key, prefix, clear)
- Disabled cache mode (enabled=False)
- Skip cache mode (skip=True in get_or_compute)
- Context manager lifecycle
- Contextvars-based request scoping
- Global cache access functions
- Key builder functions
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from aragora.knowledge.mound.query_cache import (
    CacheEntry,
    CacheStats,
    RequestScopedCache,
    get_current_cache,
    get_or_compute,
    get_or_compute_async,
    node_key,
    permission_key,
    relationship_key,
    request_cache_context,
    workspace_nodes_key,
)


# =============================================================================
# CacheStats Tests
# =============================================================================


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_default_values(self):
        """Test CacheStats default initialization."""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.compute_time_ms == 0.0

    def test_custom_values(self):
        """Test CacheStats with custom values."""
        stats = CacheStats(hits=10, misses=5, evictions=2, compute_time_ms=150.5)
        assert stats.hits == 10
        assert stats.misses == 5
        assert stats.evictions == 2
        assert stats.compute_time_ms == 150.5

    def test_hit_rate_calculation(self):
        """Test hit_rate property calculation."""
        stats = CacheStats(hits=75, misses=25)
        assert stats.hit_rate == 0.75

    def test_hit_rate_zero_total(self):
        """Test hit_rate when no hits or misses."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_all_hits(self):
        """Test hit_rate when all requests are hits."""
        stats = CacheStats(hits=100, misses=0)
        assert stats.hit_rate == 1.0

    def test_hit_rate_all_misses(self):
        """Test hit_rate when all requests are misses."""
        stats = CacheStats(hits=0, misses=100)
        assert stats.hit_rate == 0.0

    def test_to_dict(self):
        """Test to_dict conversion."""
        stats = CacheStats(hits=80, misses=20, evictions=5, compute_time_ms=123.456)
        result = stats.to_dict()

        assert result["hits"] == 80
        assert result["misses"] == 20
        assert result["evictions"] == 5
        assert result["hit_rate"] == 0.8
        assert result["compute_time_ms"] == 123.46

    def test_to_dict_rounding(self):
        """Test that to_dict properly rounds values."""
        stats = CacheStats(hits=1, misses=3, compute_time_ms=99.999)
        result = stats.to_dict()

        assert result["hit_rate"] == 0.25
        assert result["compute_time_ms"] == 100.0


# =============================================================================
# CacheEntry Tests
# =============================================================================


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_value_storage(self):
        """Test that value is stored correctly."""
        entry = CacheEntry(value="test_value")
        assert entry.value == "test_value"

    def test_value_storage_complex(self):
        """Test storage of complex values."""
        complex_value = {"key": [1, 2, 3], "nested": {"a": "b"}}
        entry = CacheEntry(value=complex_value)
        assert entry.value == complex_value
        assert entry.value is complex_value  # Same reference

    def test_value_storage_none(self):
        """Test that None can be stored as a value."""
        entry = CacheEntry(value=None)
        assert entry.value is None

    def test_created_at_default(self):
        """Test that created_at is set to current time by default."""
        before = time.time()
        entry = CacheEntry(value="test")
        after = time.time()

        assert before <= entry.created_at <= after

    def test_created_at_custom(self):
        """Test custom created_at value."""
        custom_time = 1234567890.0
        entry = CacheEntry(value="test", created_at=custom_time)
        assert entry.created_at == custom_time

    def test_access_count_default(self):
        """Test that access_count defaults to 0."""
        entry = CacheEntry(value="test")
        assert entry.access_count == 0

    def test_access_count_custom(self):
        """Test custom access_count value."""
        entry = CacheEntry(value="test", access_count=5)
        assert entry.access_count == 5

    def test_access_count_mutable(self):
        """Test that access_count can be incremented."""
        entry = CacheEntry(value="test")
        entry.access_count += 1
        assert entry.access_count == 1


# =============================================================================
# RequestScopedCache Basic Operations Tests
# =============================================================================


class TestRequestScopedCacheBasicOps:
    """Tests for RequestScopedCache basic get/set operations."""

    def test_get_set(self):
        """Test basic get and set operations."""
        cache = RequestScopedCache()
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_nonexistent_key(self):
        """Test getting a non-existent key returns None."""
        cache = RequestScopedCache()
        assert cache.get("nonexistent") is None

    def test_set_overwrites(self):
        """Test that set overwrites existing values."""
        cache = RequestScopedCache()
        cache.set("key", "value1")
        cache.set("key", "value2")
        assert cache.get("key") == "value2"

    def test_hit_tracking(self):
        """Test that hits are tracked correctly."""
        cache = RequestScopedCache()
        cache.set("key", "value")

        cache.get("key")
        cache.get("key")
        cache.get("key")

        assert cache.stats.hits == 3

    def test_miss_tracking(self):
        """Test that misses are tracked correctly."""
        cache = RequestScopedCache()

        cache.get("nonexistent1")
        cache.get("nonexistent2")

        assert cache.stats.misses == 2

    def test_hit_and_miss_tracking(self):
        """Test combined hit and miss tracking."""
        cache = RequestScopedCache()
        cache.set("key", "value")

        cache.get("key")  # hit
        cache.get("nonexistent")  # miss
        cache.get("key")  # hit

        assert cache.stats.hits == 2
        assert cache.stats.misses == 1

    def test_size_property(self):
        """Test cache size property."""
        cache = RequestScopedCache()
        assert cache.size == 0

        cache.set("key1", "value1")
        assert cache.size == 1

        cache.set("key2", "value2")
        assert cache.size == 2

    def test_access_count_incremented_on_get(self):
        """Test that access_count is incremented when entry is accessed."""
        cache = RequestScopedCache()
        cache.set("key", "value")

        cache.get("key")
        cache.get("key")
        cache.get("key")

        entry = cache._cache["key"]
        assert entry.access_count == 3


# =============================================================================
# LRU Eviction Tests
# =============================================================================


class TestLRUEviction:
    """Tests for LRU eviction behavior."""

    def test_eviction_at_max_size(self):
        """Test that eviction occurs when cache reaches max_size."""
        cache = RequestScopedCache(max_size=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        assert cache.size == 3

        cache.set("key4", "value4")

        assert cache.size == 3
        assert cache.stats.evictions == 1

    def test_eviction_removes_least_accessed(self):
        """Test that the least accessed entry is evicted."""
        cache = RequestScopedCache(max_size=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key2 and key3, but not key1
        cache.get("key2")
        cache.get("key3")

        cache.set("key4", "value4")

        # key1 should be evicted (lowest access count)
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None
        assert cache.get("key4") is not None
        # key1 was evicted but now accessing it adds a miss
        # We need to check the internal cache
        assert "key1" not in cache._cache

    def test_eviction_multiple(self):
        """Test multiple evictions."""
        cache = RequestScopedCache(max_size=2)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")

        assert cache.size == 2
        assert cache.stats.evictions == 2

    def test_evict_lru_empty_cache(self):
        """Test that _evict_lru handles empty cache gracefully."""
        cache = RequestScopedCache()
        cache._evict_lru()  # Should not raise
        assert cache.stats.evictions == 0


# =============================================================================
# get_or_compute (Sync) Tests
# =============================================================================


class TestGetOrComputeSync:
    """Tests for synchronous get_or_compute method."""

    def test_cache_miss_computes(self):
        """Test that computation occurs on cache miss."""
        cache = RequestScopedCache()
        compute_fn = MagicMock(return_value="computed_value")

        result = cache.get_or_compute("key", compute_fn)

        assert result == "computed_value"
        compute_fn.assert_called_once()

    def test_cache_hit_skips_computation(self):
        """Test that computation is skipped on cache hit."""
        cache = RequestScopedCache()
        compute_fn = MagicMock(return_value="computed_value")

        cache.get_or_compute("key", compute_fn)
        compute_fn.reset_mock()

        result = cache.get_or_compute("key", compute_fn)

        assert result == "computed_value"
        compute_fn.assert_not_called()

    def test_cache_stores_computed_value(self):
        """Test that computed value is stored in cache."""
        cache = RequestScopedCache()

        cache.get_or_compute("key", lambda: "value")

        assert cache.get("key") == "value"

    def test_tracks_compute_time(self):
        """Test that computation time is tracked."""
        cache = RequestScopedCache()

        def slow_compute():
            time.sleep(0.01)
            return "value"

        cache.get_or_compute("key", slow_compute)

        assert cache.stats.compute_time_ms >= 10.0

    def test_tracks_hits_and_misses(self):
        """Test that hits and misses are tracked correctly."""
        cache = RequestScopedCache()

        cache.get_or_compute("key", lambda: "value")  # miss
        cache.get_or_compute("key", lambda: "value")  # hit
        cache.get_or_compute("key2", lambda: "value2")  # miss

        assert cache.stats.misses == 2
        assert cache.stats.hits == 1

    def test_eviction_during_compute(self):
        """Test that eviction occurs during get_or_compute when at capacity."""
        cache = RequestScopedCache(max_size=2)

        cache.get_or_compute("key1", lambda: "value1")
        cache.get_or_compute("key2", lambda: "value2")
        cache.get_or_compute("key3", lambda: "value3")

        assert cache.size == 2
        assert cache.stats.evictions == 1


# =============================================================================
# get_or_compute_async Tests
# =============================================================================


class TestGetOrComputeAsync:
    """Tests for asynchronous get_or_compute_async method."""

    @pytest.mark.asyncio
    async def test_cache_miss_computes(self):
        """Test that async computation occurs on cache miss."""
        cache = RequestScopedCache()
        call_count = 0

        async def compute_fn():
            nonlocal call_count
            call_count += 1
            return "computed_value"

        result = await cache.get_or_compute_async("key", compute_fn)

        assert result == "computed_value"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_cache_hit_skips_computation(self):
        """Test that async computation is skipped on cache hit."""
        cache = RequestScopedCache()
        call_count = 0

        async def compute_fn():
            nonlocal call_count
            call_count += 1
            return "computed_value"

        await cache.get_or_compute_async("key", compute_fn)
        result = await cache.get_or_compute_async("key", compute_fn)

        assert result == "computed_value"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_cache_stores_computed_value(self):
        """Test that async computed value is stored in cache."""
        cache = RequestScopedCache()

        async def compute():
            return "async_value"

        await cache.get_or_compute_async("key", compute)

        assert cache.get("key") == "async_value"

    @pytest.mark.asyncio
    async def test_tracks_compute_time(self):
        """Test that async computation time is tracked."""
        cache = RequestScopedCache()

        async def slow_compute():
            await asyncio.sleep(0.01)
            return "value"

        await cache.get_or_compute_async("key", slow_compute)

        assert cache.stats.compute_time_ms >= 10.0

    @pytest.mark.asyncio
    async def test_tracks_hits_and_misses(self):
        """Test that async hits and misses are tracked."""
        cache = RequestScopedCache()

        async def compute():
            return "value"

        await cache.get_or_compute_async("key1", compute)  # miss
        await cache.get_or_compute_async("key1", compute)  # hit
        await cache.get_or_compute_async("key2", compute)  # miss

        assert cache.stats.misses == 2
        assert cache.stats.hits == 1

    @pytest.mark.asyncio
    async def test_eviction_during_async_compute(self):
        """Test that eviction occurs during async get_or_compute when at capacity."""
        cache = RequestScopedCache(max_size=2)

        async def compute(val):
            return val

        await cache.get_or_compute_async("key1", lambda: compute("value1"))
        await cache.get_or_compute_async("key2", lambda: compute("value2"))
        await cache.get_or_compute_async("key3", lambda: compute("value3"))

        assert cache.size == 2
        assert cache.stats.evictions == 1


# =============================================================================
# Invalidation Tests
# =============================================================================


class TestInvalidation:
    """Tests for cache invalidation methods."""

    def test_invalidate_existing_key(self):
        """Test invalidating an existing key."""
        cache = RequestScopedCache()
        cache.set("key", "value")

        result = cache.invalidate("key")

        assert result is True
        assert cache.get("key") is None

    def test_invalidate_nonexistent_key(self):
        """Test invalidating a non-existent key returns False."""
        cache = RequestScopedCache()

        result = cache.invalidate("nonexistent")

        assert result is False

    def test_invalidate_prefix(self):
        """Test invalidating keys by prefix."""
        cache = RequestScopedCache()
        cache.set("node:123", "value1")
        cache.set("node:456", "value2")
        cache.set("perm:789", "value3")

        count = cache.invalidate_prefix("node:")

        assert count == 2
        assert cache.get("node:123") is None
        assert cache.get("node:456") is None
        assert cache.get("perm:789") == "value3"

    def test_invalidate_prefix_no_matches(self):
        """Test invalidate_prefix with no matching keys."""
        cache = RequestScopedCache()
        cache.set("key1", "value1")

        count = cache.invalidate_prefix("nonexistent:")

        assert count == 0
        assert cache.get("key1") == "value1"

    def test_invalidate_prefix_all_keys(self):
        """Test invalidate_prefix matching all keys."""
        cache = RequestScopedCache()
        cache.set("prefix:1", "value1")
        cache.set("prefix:2", "value2")

        count = cache.invalidate_prefix("prefix:")

        assert count == 2
        assert cache.size == 0

    def test_clear(self):
        """Test clearing all cache entries."""
        cache = RequestScopedCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        cache.clear()

        assert cache.size == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") is None


# =============================================================================
# Disabled Cache Mode Tests
# =============================================================================


class TestDisabledCacheMode:
    """Tests for cache behavior when enabled=False."""

    def test_get_returns_none(self):
        """Test that get always returns None when disabled."""
        cache = RequestScopedCache(enabled=False)
        cache._cache["key"] = CacheEntry(value="value")  # Manually add

        assert cache.get("key") is None

    def test_set_does_nothing(self):
        """Test that set does nothing when disabled."""
        cache = RequestScopedCache(enabled=False)
        cache.set("key", "value")

        assert cache.size == 0

    def test_get_or_compute_always_computes(self):
        """Test that get_or_compute always computes when disabled."""
        cache = RequestScopedCache(enabled=False)
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return "value"

        cache.get_or_compute("key", compute)
        cache.get_or_compute("key", compute)

        assert call_count == 2

    def test_get_or_compute_does_not_cache(self):
        """Test that values are not cached when disabled."""
        cache = RequestScopedCache(enabled=False)

        cache.get_or_compute("key", lambda: "value")

        assert cache.size == 0

    @pytest.mark.asyncio
    async def test_get_or_compute_async_always_computes(self):
        """Test that async get_or_compute always computes when disabled."""
        cache = RequestScopedCache(enabled=False)
        call_count = 0

        async def compute():
            nonlocal call_count
            call_count += 1
            return "value"

        await cache.get_or_compute_async("key", compute)
        await cache.get_or_compute_async("key", compute)

        assert call_count == 2

    def test_disabled_no_stats_tracking(self):
        """Test that stats are not tracked when disabled."""
        cache = RequestScopedCache(enabled=False)

        cache.get("key")
        cache.get("key")

        # Disabled cache doesn't track hits/misses through get
        assert cache.stats.hits == 0
        assert cache.stats.misses == 0

    def test_disabled_tracks_compute_time(self):
        """Test that compute time is still tracked when disabled."""
        cache = RequestScopedCache(enabled=False)

        def slow_compute():
            time.sleep(0.01)
            return "value"

        cache.get_or_compute("key", slow_compute)

        assert cache.stats.compute_time_ms >= 10.0


# =============================================================================
# Skip Cache Mode Tests
# =============================================================================


class TestSkipCacheMode:
    """Tests for skip_cache=True behavior in get_or_compute."""

    def test_skip_cache_always_computes(self):
        """Test that skip_cache=True always computes."""
        cache = RequestScopedCache()
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return f"value_{call_count}"

        cache.get_or_compute("key", compute)  # First call
        result = cache.get_or_compute("key", compute, skip_cache=True)

        assert call_count == 2
        assert result == "value_2"

    def test_skip_cache_still_caches_result(self):
        """Test that skip_cache=True still caches the computed result."""
        cache = RequestScopedCache()

        cache.get_or_compute("key", lambda: "old_value")
        cache.get_or_compute("key", lambda: "new_value", skip_cache=True)

        # The new value should be cached
        assert cache.get("key") == "new_value"

    @pytest.mark.asyncio
    async def test_skip_cache_async_always_computes(self):
        """Test that async skip_cache=True always computes."""
        cache = RequestScopedCache()
        call_count = 0

        async def compute():
            nonlocal call_count
            call_count += 1
            return f"value_{call_count}"

        await cache.get_or_compute_async("key", compute)
        await cache.get_or_compute_async("key", compute, skip_cache=True)

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_skip_cache_async_still_caches_result(self):
        """Test that async skip_cache=True still caches the result."""
        cache = RequestScopedCache()

        async def compute_old():
            return "old_value"

        async def compute_new():
            return "new_value"

        await cache.get_or_compute_async("key", compute_old)
        await cache.get_or_compute_async("key", compute_new, skip_cache=True)

        assert cache.get("key") == "new_value"


# =============================================================================
# Context Manager Lifecycle Tests
# =============================================================================


class TestContextManagerLifecycle:
    """Tests for context manager __enter__ and __exit__ behavior."""

    def test_enter_returns_cache(self):
        """Test that __enter__ returns the cache instance."""
        cache = RequestScopedCache()

        with cache as ctx:
            assert ctx is cache

    def test_exit_clears_cache(self):
        """Test that __exit__ clears the cache."""
        cache = RequestScopedCache()

        with cache:
            cache.set("key", "value")
            assert cache.size == 1

        assert cache.size == 0

    def test_exit_resets_context_var(self):
        """Test that __exit__ resets the context variable."""
        cache = RequestScopedCache()

        with cache:
            assert get_current_cache() is cache

        assert get_current_cache() is None

    def test_nested_contexts(self):
        """Test nested cache contexts."""
        cache1 = RequestScopedCache()
        cache2 = RequestScopedCache()

        with cache1:
            assert get_current_cache() is cache1
            cache1.set("key", "value1")

            with cache2:
                assert get_current_cache() is cache2
                cache2.set("key", "value2")

            assert get_current_cache() is cache1
            assert cache1.get("key") == "value1"

        assert get_current_cache() is None

    def test_exit_on_exception(self):
        """Test that cache is properly cleaned up on exception."""
        cache = RequestScopedCache()

        try:
            with cache:
                cache.set("key", "value")
                raise ValueError("test error")
        except ValueError:
            pass

        assert cache.size == 0
        assert get_current_cache() is None


# =============================================================================
# Contextvars Request Scoping Tests
# =============================================================================


class TestContextvarsScoping:
    """Tests for contextvars-based request scoping."""

    def test_separate_async_tasks_have_separate_caches(self):
        """Test that separate async tasks have separate cache instances."""
        results = []

        async def task1():
            with request_cache_context() as cache:
                cache.set("key", "task1_value")
                await asyncio.sleep(0.01)
                results.append(("task1", cache.get("key")))

        async def task2():
            with request_cache_context() as cache:
                cache.set("key", "task2_value")
                await asyncio.sleep(0.01)
                results.append(("task2", cache.get("key")))

        async def run():
            await asyncio.gather(task1(), task2())

        asyncio.run(run())

        assert ("task1", "task1_value") in results
        assert ("task2", "task2_value") in results

    def test_no_cache_outside_context(self):
        """Test that get_current_cache returns None outside context."""
        assert get_current_cache() is None

    def test_cache_available_inside_context(self):
        """Test that get_current_cache returns cache inside context."""
        with request_cache_context() as cache:
            assert get_current_cache() is cache


# =============================================================================
# Global Cache Access Functions Tests
# =============================================================================


class TestGlobalCacheAccessFunctions:
    """Tests for global cache access functions."""

    def test_get_current_cache_returns_none_by_default(self):
        """Test that get_current_cache returns None when no context."""
        assert get_current_cache() is None

    def test_get_current_cache_returns_active_cache(self):
        """Test that get_current_cache returns the active cache."""
        cache = RequestScopedCache()
        with cache:
            assert get_current_cache() is cache

    def test_get_or_compute_without_context(self):
        """Test that global get_or_compute works without context."""
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return "value"

        result1 = get_or_compute("key", compute)
        result2 = get_or_compute("key", compute)

        assert result1 == "value"
        assert result2 == "value"
        assert call_count == 2  # No caching without context

    def test_get_or_compute_with_context(self):
        """Test that global get_or_compute uses context cache."""
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return "value"

        with request_cache_context():
            result1 = get_or_compute("key", compute)
            result2 = get_or_compute("key", compute)

        assert result1 == "value"
        assert result2 == "value"
        assert call_count == 1  # Cached on second call

    def test_get_or_compute_skip_cache(self):
        """Test that global get_or_compute respects skip_cache."""
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return f"value_{call_count}"

        with request_cache_context():
            get_or_compute("key", compute)
            result = get_or_compute("key", compute, skip_cache=True)

        assert result == "value_2"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_get_or_compute_async_without_context(self):
        """Test that async global get_or_compute_async works without context."""
        call_count = 0

        async def compute():
            nonlocal call_count
            call_count += 1
            return "value"

        result1 = await get_or_compute_async("key", compute)
        result2 = await get_or_compute_async("key", compute)

        assert result1 == "value"
        assert result2 == "value"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_get_or_compute_async_with_context(self):
        """Test that async global get_or_compute_async uses context cache."""
        call_count = 0

        async def compute():
            nonlocal call_count
            call_count += 1
            return "value"

        with request_cache_context():
            result1 = await get_or_compute_async("key", compute)
            result2 = await get_or_compute_async("key", compute)

        assert result1 == "value"
        assert result2 == "value"
        assert call_count == 1


# =============================================================================
# request_cache_context Tests
# =============================================================================


class TestRequestCacheContext:
    """Tests for request_cache_context context manager."""

    def test_yields_cache_instance(self):
        """Test that request_cache_context yields a cache instance."""
        with request_cache_context() as cache:
            assert isinstance(cache, RequestScopedCache)

    def test_custom_max_size(self):
        """Test that custom max_size is respected."""
        with request_cache_context(max_size=5) as cache:
            assert cache.max_size == 5

    def test_custom_enabled(self):
        """Test that custom enabled flag is respected."""
        with request_cache_context(enabled=False) as cache:
            assert cache.enabled is False

    def test_cache_cleared_on_exit(self):
        """Test that cache is cleared when context exits."""
        with request_cache_context() as cache:
            cache.set("key", "value")
            assert cache.size == 1

        # Cache should be cleared after exit
        assert cache.size == 0

    def test_stats_preserved_after_exit(self):
        """Test that stats are preserved after context exit."""
        with request_cache_context() as cache:
            cache.get_or_compute("key", lambda: "value")
            cache.get_or_compute("key", lambda: "value")

        assert cache.stats.hits == 1
        assert cache.stats.misses == 1


# =============================================================================
# Key Builder Functions Tests
# =============================================================================


class TestKeyBuilders:
    """Tests for key builder functions."""

    def test_node_key(self):
        """Test node_key builder."""
        assert node_key("123") == "node:123"
        assert node_key("abc-def") == "node:abc-def"

    def test_permission_key(self):
        """Test permission_key builder."""
        assert permission_key("item1", "user1", "read") == "perm:item1:user1:read"
        assert permission_key("doc:123", "team:456", "write") == "perm:doc:123:team:456:write"

    def test_relationship_key(self):
        """Test relationship_key builder."""
        assert relationship_key("node1", "node2") == "rel:node1:node2"
        assert relationship_key("a", "b") == "rel:a:b"

    def test_workspace_nodes_key_without_type(self):
        """Test workspace_nodes_key without node type."""
        assert workspace_nodes_key("ws123") == "ws_nodes:ws123"

    def test_workspace_nodes_key_with_type(self):
        """Test workspace_nodes_key with node type."""
        assert workspace_nodes_key("ws123", "document") == "ws_nodes:ws123:document"

    def test_key_uniqueness(self):
        """Test that different inputs produce unique keys."""
        keys = {
            node_key("1"),
            node_key("2"),
            permission_key("a", "b", "c"),
            permission_key("a", "b", "d"),
            relationship_key("x", "y"),
            relationship_key("y", "x"),
            workspace_nodes_key("ws1"),
            workspace_nodes_key("ws1", "type1"),
        }
        assert len(keys) == 8


# =============================================================================
# Edge Cases and Additional Coverage Tests
# =============================================================================


class TestEdgeCases:
    """Additional tests for edge cases and full coverage."""

    def test_cache_with_various_value_types(self):
        """Test caching various Python types."""
        cache = RequestScopedCache()

        test_values = [
            ("int", 42),
            ("float", 3.14),
            ("string", "hello"),
            ("list", [1, 2, 3]),
            ("dict", {"a": 1}),
            ("tuple", (1, 2, 3)),
            ("set", {1, 2, 3}),
            ("none", None),
            ("bool", True),
            ("bytes", b"hello"),
        ]

        for key, value in test_values:
            cache.set(key, value)
            retrieved = cache.get(key)
            assert retrieved == value, f"Failed for type {key}"

    def test_empty_string_key(self):
        """Test that empty string can be used as a key."""
        cache = RequestScopedCache()
        cache.set("", "empty_key_value")
        assert cache.get("") == "empty_key_value"

    def test_unicode_keys(self):
        """Test that unicode keys work correctly."""
        cache = RequestScopedCache()
        cache.set("key_with_emoji_\U0001f600", "value")
        cache.set("key_\u4e2d\u6587", "chinese_value")

        assert cache.get("key_with_emoji_\U0001f600") == "value"
        assert cache.get("key_\u4e2d\u6587") == "chinese_value"

    def test_stats_property_returns_same_instance(self):
        """Test that stats property returns the same instance."""
        cache = RequestScopedCache()
        assert cache.stats is cache._stats

    def test_max_size_one(self):
        """Test cache with max_size of 1."""
        cache = RequestScopedCache(max_size=1)

        cache.set("key1", "value1")
        assert cache.size == 1

        cache.set("key2", "value2")
        assert cache.size == 1
        assert cache.get("key2") == "value2"
        assert "key1" not in cache._cache

    def test_callable_returning_none(self):
        """Test get_or_compute with callable returning None."""
        cache = RequestScopedCache()

        result = cache.get_or_compute("key", lambda: None)

        assert result is None
        # None should be cached
        assert "key" in cache._cache
        assert cache._cache["key"].value is None

    def test_stats_hit_rate_precision(self):
        """Test hit_rate calculation precision."""
        stats = CacheStats(hits=1, misses=2)
        assert abs(stats.hit_rate - 0.3333333333) < 0.0001

    def test_context_manager_token_reset(self):
        """Test that token is properly reset after context exit."""
        cache = RequestScopedCache()

        with cache:
            assert cache._token is not None

        assert cache._token is None

    def test_multiple_sequential_contexts(self):
        """Test multiple sequential cache contexts."""
        for i in range(3):
            with request_cache_context() as cache:
                cache.set("key", f"value_{i}")
                assert cache.get("key") == f"value_{i}"

            assert get_current_cache() is None

    @pytest.mark.asyncio
    async def test_async_exception_handling(self):
        """Test that async exceptions don't corrupt cache state."""
        cache = RequestScopedCache()

        async def failing_compute():
            raise ValueError("async error")

        with pytest.raises(ValueError):
            await cache.get_or_compute_async("key", failing_compute)

        # Cache should still be functional
        async def success_compute():
            return "success"

        result = await cache.get_or_compute_async("key2", success_compute)
        assert result == "success"

    def test_sync_exception_handling(self):
        """Test that sync exceptions don't corrupt cache state."""
        cache = RequestScopedCache()

        def failing_compute():
            raise ValueError("sync error")

        with pytest.raises(ValueError):
            cache.get_or_compute("key", failing_compute)

        # Cache should still be functional
        result = cache.get_or_compute("key2", lambda: "success")
        assert result == "success"

    def test_large_number_of_entries(self):
        """Test cache with many entries."""
        cache = RequestScopedCache(max_size=100)

        for i in range(100):
            cache.set(f"key_{i}", f"value_{i}")

        assert cache.size == 100

        # Adding one more should trigger eviction
        cache.set("key_100", "value_100")
        assert cache.size == 100
        assert cache.stats.evictions == 1

    def test_invalidate_prefix_empty_prefix(self):
        """Test invalidate_prefix with empty string prefix."""
        cache = RequestScopedCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Empty prefix should match all keys
        count = cache.invalidate_prefix("")

        assert count == 2
        assert cache.size == 0

    def test_default_max_size_from_constant(self):
        """Test that default max_size comes from CACHE_MAX_SIZE."""
        from aragora.knowledge.mound.query_cache import CACHE_MAX_SIZE

        cache = RequestScopedCache()
        assert cache.max_size == CACHE_MAX_SIZE

    def test_default_enabled_from_constant(self):
        """Test that default enabled comes from CACHE_ENABLED."""
        from aragora.knowledge.mound.query_cache import CACHE_ENABLED

        cache = RequestScopedCache()
        assert cache.enabled == CACHE_ENABLED
