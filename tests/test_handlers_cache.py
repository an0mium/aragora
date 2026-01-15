"""Tests for aragora.server.handlers.cache module."""

import asyncio
import concurrent.futures
import threading
import time

import pytest

from aragora.server.handlers.admin.cache import (
    BoundedTTLCache,
    CACHE_INVALIDATION_MAP,
    clear_cache,
    get_cache_stats,
    get_handler_cache,
    invalidate_agent_cache,
    invalidate_cache,
    invalidate_debate_cache,
    invalidate_leaderboard_cache,
    invalidate_on_event,
    invalidates_cache,
    ttl_cache,
)


class TestBoundedTTLCache:
    """Tests for BoundedTTLCache class."""

    def test_basic_get_set(self):
        """Test basic get/set operations."""
        cache = BoundedTTLCache(max_entries=10)
        cache.set("key1", "value1")
        hit, value = cache.get("key1", ttl_seconds=60)
        assert hit is True
        assert value == "value1"

    def test_cache_miss(self):
        """Test cache miss for non-existent key."""
        cache = BoundedTTLCache(max_entries=10)
        hit, value = cache.get("nonexistent", ttl_seconds=60)
        assert hit is False
        assert value is None

    def test_ttl_expiry(self):
        """Test that entries expire after TTL."""
        cache = BoundedTTLCache(max_entries=10)
        cache.set("key1", "value1")

        # Should hit immediately
        hit, value = cache.get("key1", ttl_seconds=0.1)
        assert hit is True

        # Wait for expiry
        time.sleep(0.15)

        # Should miss after expiry
        hit, value = cache.get("key1", ttl_seconds=0.1)
        assert hit is False

    def test_lru_eviction(self):
        """Test LRU eviction when max entries reached."""
        cache = BoundedTTLCache(max_entries=5, evict_percent=0.2)

        # Fill cache to capacity
        for i in range(5):
            cache.set(f"key{i}", f"value{i}")

        assert len(cache) == 5

        # Add one more to trigger eviction
        cache.set("key5", "value5")

        # At least one entry should have been evicted
        assert len(cache) <= 5

        # Most recent entry should still be there
        hit, value = cache.get("key5", ttl_seconds=60)
        assert hit is True

    def test_update_existing_key(self):
        """Test updating an existing key."""
        cache = BoundedTTLCache(max_entries=10)
        cache.set("key1", "value1")
        cache.set("key1", "value2")

        hit, value = cache.get("key1", ttl_seconds=60)
        assert hit is True
        assert value == "value2"
        assert len(cache) == 1

    def test_clear_all(self):
        """Test clearing all entries."""
        cache = BoundedTTLCache(max_entries=10)
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cleared = cache.clear()
        assert cleared == 2
        assert len(cache) == 0

    def test_clear_by_prefix(self):
        """Test clearing entries by prefix."""
        cache = BoundedTTLCache(max_entries=10)
        cache.set("leaderboard:top10", "data1")
        cache.set("leaderboard:top20", "data2")
        cache.set("agents:list", "data3")

        cleared = cache.clear("leaderboard")
        assert cleared == 2
        assert len(cache) == 1

        hit, _ = cache.get("agents:list", ttl_seconds=60)
        assert hit is True

    def test_stats(self):
        """Test cache statistics."""
        cache = BoundedTTLCache(max_entries=10)
        cache.set("key1", "value1")

        # Generate hits and misses
        cache.get("key1", ttl_seconds=60)  # hit
        cache.get("key1", ttl_seconds=60)  # hit
        cache.get("missing", ttl_seconds=60)  # miss

        stats = cache.stats
        assert stats["entries"] == 1
        assert stats["max_entries"] == 10
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2 / 3)

    def test_contains(self):
        """Test __contains__ operator."""
        cache = BoundedTTLCache(max_entries=10)
        cache.set("key1", "value1")

        assert "key1" in cache
        assert "key2" not in cache


class TestTTLCacheDecorator:
    """Tests for ttl_cache decorator."""

    def test_caches_function_result(self):
        """Test that decorator caches function results."""
        call_count = 0

        @ttl_cache(ttl_seconds=60, key_prefix="test", skip_first=False)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call - should execute function
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call - should return cached result
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Not incremented

        # Different argument - should execute function
        result3 = expensive_function(10)
        assert result3 == 20
        assert call_count == 2

        # Cleanup
        clear_cache("test")

    def test_respects_ttl(self):
        """Test that cached values expire after TTL."""
        call_count = 0

        @ttl_cache(ttl_seconds=0.1, key_prefix="test_ttl", skip_first=False)
        def short_lived(x):
            nonlocal call_count
            call_count += 1
            return x

        short_lived(1)
        assert call_count == 1

        short_lived(1)
        assert call_count == 1

        time.sleep(0.15)

        short_lived(1)
        assert call_count == 2

        clear_cache("test_ttl")

    def test_method_caching(self):
        """Test caching on class methods (skip_first=True)."""
        call_count = 0

        class MyClass:
            @ttl_cache(ttl_seconds=60, key_prefix="method_test", skip_first=True)
            def compute(self, x):
                nonlocal call_count
                call_count += 1
                return x * 3

        obj = MyClass()
        obj.compute(5)
        assert call_count == 1

        obj.compute(5)
        assert call_count == 1

        clear_cache("method_test")


class TestInvalidatesCache:
    """Tests for invalidates_cache decorator."""

    def test_sync_function_invalidation(self):
        """Test cache invalidation after sync function."""
        cache = get_handler_cache()
        cache.set("leaderboard:test", "data")

        @invalidates_cache("elo_updated")
        def update_elo():
            return "updated"

        result = update_elo()
        assert result == "updated"

        # Cache should be invalidated
        hit, _ = cache.get("leaderboard:test", ttl_seconds=60)
        assert hit is False

    @pytest.mark.asyncio
    async def test_async_function_invalidation(self):
        """Test cache invalidation after async function."""
        cache = get_handler_cache()
        cache.set("dashboard_debates:test", "data")

        @invalidates_cache("debate_completed")
        async def complete_debate():
            return "completed"

        result = await complete_debate()
        assert result == "completed"


class TestCacheInvalidation:
    """Tests for cache invalidation functions."""

    def test_invalidate_on_event(self):
        """Test event-driven cache invalidation."""
        cache = get_handler_cache()

        # Set some entries
        cache.set("leaderboard:top10", "data1")
        cache.set("lb_rankings:all", "data2")
        cache.set("unrelated:data", "data3")

        # Invalidate ELO-related caches
        cleared = invalidate_on_event("elo_updated")

        # Leaderboard entries should be cleared
        hit, _ = cache.get("leaderboard:top10", ttl_seconds=60)
        assert hit is False

        # Unrelated entry should remain
        hit, _ = cache.get("unrelated:data", ttl_seconds=60)
        assert hit is True

        cache.clear()

    def test_invalidate_leaderboard_cache(self):
        """Test leaderboard cache invalidation convenience function."""
        cache = get_handler_cache()
        cache.set("leaderboard:test", "data")

        invalidate_leaderboard_cache()

        hit, _ = cache.get("leaderboard:test", ttl_seconds=60)
        assert hit is False

    def test_invalidate_agent_cache_specific(self):
        """Test invalidating specific agent's cache."""
        cache = get_handler_cache()
        cache.set("profile:claude", "claude_data")
        cache.set("profile:gemini", "gemini_data")

        cleared = invalidate_agent_cache("claude")

        # Claude's cache should be cleared
        hit, _ = cache.get("profile:claude", ttl_seconds=60)
        assert hit is False

        # Gemini's cache should remain
        hit, _ = cache.get("profile:gemini", ttl_seconds=60)
        assert hit is True

        cache.clear()

    def test_invalidate_debate_cache_specific(self):
        """Test invalidating specific debate's cache."""
        cache = get_handler_cache()
        cache.set("debate:123:data", "debate_data")
        cache.set("debate:456:data", "other_data")

        cleared = invalidate_debate_cache("123")

        hit, _ = cache.get("debate:123:data", ttl_seconds=60)
        assert hit is False

        hit, _ = cache.get("debate:456:data", ttl_seconds=60)
        assert hit is True

        cache.clear()

    def test_invalidate_cache_by_data_source(self):
        """Test invalidate_cache by data source name."""
        cache = get_handler_cache()
        cache.set("leaderboard:test", "data")

        # "elo" maps to "elo_updated" event
        invalidate_cache("elo")

        hit, _ = cache.get("leaderboard:test", ttl_seconds=60)
        assert hit is False


class TestCacheStats:
    """Tests for cache statistics."""

    def test_get_cache_stats(self):
        """Test getting cache statistics."""
        cache = get_handler_cache()
        cache.clear()

        cache.set("key1", "value1")
        cache.get("key1", ttl_seconds=60)
        cache.get("missing", ttl_seconds=60)

        stats = get_cache_stats()
        assert "entries" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats

        cache.clear()


class TestCacheInvalidationMap:
    """Tests for CACHE_INVALIDATION_MAP configuration."""

    def test_map_has_required_events(self):
        """Test that invalidation map contains expected events."""
        required_events = [
            "elo_updated",
            "match_recorded",
            "debate_completed",
            "agent_updated",
            "memory_updated",
            "consensus_reached",
        ]

        for event in required_events:
            assert event in CACHE_INVALIDATION_MAP

    def test_map_events_have_prefixes(self):
        """Test that all events have at least one prefix."""
        for event, prefixes in CACHE_INVALIDATION_MAP.items():
            assert len(prefixes) > 0, f"Event {event} has no prefixes"
            for prefix in prefixes:
                assert isinstance(prefix, str)
                assert len(prefix) > 0


class TestClearCache:
    """Tests for clear_cache function."""

    def test_clear_cache_all(self):
        """Test clearing all cache entries."""
        cache = get_handler_cache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cleared = clear_cache()
        assert cleared >= 2

    def test_clear_cache_by_prefix(self):
        """Test clearing cache by prefix."""
        cache = get_handler_cache()
        cache.set("test:key1", "value1")
        cache.set("test:key2", "value2")
        cache.set("other:key", "value3")

        cleared = clear_cache("test")
        assert cleared == 2

        hit, _ = cache.get("other:key", ttl_seconds=60)
        assert hit is True

        cache.clear()


# =============================================================================
# Edge Case Tests (F2)
# =============================================================================


class TestBoundedTTLCacheEdgeCases:
    """Edge case tests for BoundedTTLCache."""

    def test_zero_ttl_always_misses(self):
        """Test that zero TTL always results in cache miss."""
        cache = BoundedTTLCache(max_entries=10)
        cache.set("key1", "value1")

        # Even immediate access with TTL=0 should miss
        hit, value = cache.get("key1", ttl_seconds=0)
        assert hit is False

    def test_negative_ttl_always_misses(self):
        """Test that negative TTL always results in cache miss."""
        cache = BoundedTTLCache(max_entries=10)
        cache.set("key1", "value1")

        hit, value = cache.get("key1", ttl_seconds=-1)
        assert hit is False

    def test_exact_max_entries_no_eviction(self):
        """Test that filling to exactly max_entries doesn't trigger eviction."""
        cache = BoundedTTLCache(max_entries=5, evict_percent=0.2)

        for i in range(5):
            cache.set(f"key{i}", f"value{i}")

        assert len(cache) == 5

        # All entries should still be accessible
        for i in range(5):
            hit, value = cache.get(f"key{i}", ttl_seconds=60)
            assert hit is True
            assert value == f"value{i}"

    def test_eviction_removes_oldest_entries(self):
        """Test that eviction removes the oldest entries."""
        cache = BoundedTTLCache(max_entries=5, evict_percent=0.4)

        # Fill cache
        for i in range(5):
            cache.set(f"key{i}", f"value{i}")

        # Access key2 to make it more recently used
        cache.get("key2", ttl_seconds=60)

        # Add new entry to trigger eviction
        cache.set("key5", "value5")

        # key0 and key1 should be evicted (oldest, not accessed)
        hit0, _ = cache.get("key0", ttl_seconds=60)
        hit1, _ = cache.get("key1", ttl_seconds=60)
        hit2, _ = cache.get("key2", ttl_seconds=60)
        hit5, _ = cache.get("key5", ttl_seconds=60)

        assert hit0 is False
        assert hit1 is False
        assert hit2 is True  # Was accessed, should survive
        assert hit5 is True  # Newest, should survive

    def test_empty_cache_clear(self):
        """Test clearing an empty cache."""
        cache = BoundedTTLCache(max_entries=10)
        cleared = cache.clear()
        assert cleared == 0

    def test_empty_prefix_clear(self):
        """Test clearing with prefix that matches nothing."""
        cache = BoundedTTLCache(max_entries=10)
        cache.set("key1", "value1")
        cleared = cache.clear("nonexistent_prefix")
        assert cleared == 0
        assert len(cache) == 1

    def test_large_capacity_eviction(self):
        """Test eviction behavior with larger cache."""
        cache = BoundedTTLCache(max_entries=100, evict_percent=0.1)

        # Fill beyond capacity
        for i in range(110):
            cache.set(f"key{i}", f"value{i}")

        # Should not exceed max_entries
        assert len(cache) <= 100

        # Most recent entries should be accessible
        hit, _ = cache.get("key109", ttl_seconds=60)
        assert hit is True

    def test_single_entry_cache(self):
        """Test cache with max_entries=1."""
        cache = BoundedTTLCache(max_entries=1, evict_percent=1.0)

        cache.set("key1", "value1")
        hit, value = cache.get("key1", ttl_seconds=60)
        assert hit is True
        assert value == "value1"

        cache.set("key2", "value2")
        # key1 should be evicted
        hit1, _ = cache.get("key1", ttl_seconds=60)
        hit2, value2 = cache.get("key2", ttl_seconds=60)
        assert hit1 is False
        assert hit2 is True
        assert value2 == "value2"


class TestCacheThreadSafety:
    """Thread safety tests for BoundedTTLCache."""

    def test_concurrent_reads(self):
        """Test concurrent read operations."""
        cache = BoundedTTLCache(max_entries=100)

        # Populate cache
        for i in range(50):
            cache.set(f"key{i}", f"value{i}")

        errors = []

        def reader(thread_id):
            try:
                for _ in range(100):
                    for i in range(50):
                        hit, value = cache.get(f"key{i}", ttl_seconds=60)
                        if hit and value != f"value{i}":
                            errors.append(f"Thread {thread_id}: Wrong value for key{i}")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"

    def test_concurrent_writes(self):
        """Test concurrent write operations."""
        cache = BoundedTTLCache(max_entries=100)
        errors = []

        def writer(thread_id):
            try:
                for i in range(100):
                    key = f"thread{thread_id}_key{i}"
                    cache.set(key, f"value{i}")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        # Should have some entries (exact count depends on eviction)
        assert len(cache) > 0

    def test_concurrent_read_write(self):
        """Test concurrent read and write operations."""
        cache = BoundedTTLCache(max_entries=100)
        errors = []
        stop_event = threading.Event()

        def writer():
            try:
                i = 0
                while not stop_event.is_set():
                    cache.set(f"key{i % 50}", f"value{i}")
                    i += 1
            except Exception as e:
                errors.append(f"Writer: {e}")

        def reader():
            try:
                while not stop_event.is_set():
                    for i in range(50):
                        cache.get(f"key{i}", ttl_seconds=60)
            except Exception as e:
                errors.append(f"Reader: {e}")

        writer_thread = threading.Thread(target=writer)
        reader_threads = [threading.Thread(target=reader) for _ in range(3)]

        writer_thread.start()
        for t in reader_threads:
            t.start()

        time.sleep(0.5)  # Run for 500ms
        stop_event.set()

        writer_thread.join()
        for t in reader_threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"


class TestAsyncTTLCacheDecorator:
    """Tests for async_ttl_cache decorator."""

    @pytest.mark.asyncio
    async def test_async_cache_basic(self):
        """Test basic async caching."""
        from aragora.server.handlers.admin.cache import async_ttl_cache

        call_count = 0

        @async_ttl_cache(ttl_seconds=60, key_prefix="async_test", skip_first=False)
        async def async_function(x):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return x * 2

        result1 = await async_function(5)
        assert result1 == 10
        assert call_count == 1

        result2 = await async_function(5)
        assert result2 == 10
        assert call_count == 1  # Cached

        result3 = await async_function(10)
        assert result3 == 20
        assert call_count == 2  # Different args

        clear_cache("async_test")

    @pytest.mark.asyncio
    async def test_async_cache_concurrent_calls(self):
        """Test that concurrent calls don't cause issues."""
        from aragora.server.handlers.admin.cache import async_ttl_cache

        call_count = 0

        @async_ttl_cache(ttl_seconds=60, key_prefix="async_concurrent", skip_first=False)
        async def slow_function(x):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return x * 2

        # Make concurrent calls
        results = await asyncio.gather(
            slow_function(5),
            slow_function(5),
            slow_function(5),
        )

        # All should return correct value
        assert all(r == 10 for r in results)

        # Due to race conditions, call_count might be > 1 but should stabilize
        # Subsequent calls should be cached
        call_count_before = call_count
        await slow_function(5)
        assert call_count == call_count_before  # Should be cached now

        clear_cache("async_concurrent")
