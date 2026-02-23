"""Tests for aragora.server.handlers.admin.cache module.

Comprehensive coverage of:
- BoundedTTLCache: get, set, clear, invalidate_containing, eviction, stats, thread safety
- ttl_cache decorator: sync function caching with TTL, key_prefix, skip_first
- async_ttl_cache decorator: async function caching with TTL, key_prefix, skip_first
- invalidates_cache decorator: sync and async event-driven invalidation
- Module-level functions: clear_cache, get_cache_stats, invalidate_on_event,
  invalidate_cache, invalidate_leaderboard_cache, invalidate_agent_cache,
  invalidate_debate_cache
- get_handler_cache / _register_handler_cache: ServiceRegistry integration
- _get_metrics: lazy metric loading
- CACHE_INVALIDATION_MAP: event-to-prefix mappings
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.admin.cache import (
    CACHE_INVALIDATION_MAP,
    BoundedTTLCache,
    async_ttl_cache,
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
    _cache,
    _get_metrics,
    _invalidate_events,
    _noop_metric,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def _clear_global_cache():
    """Clear global cache before and after each test to prevent cross-test pollution."""
    _cache.clear()
    _cache._hits = 0
    _cache._misses = 0
    yield
    _cache.clear()
    _cache._hits = 0
    _cache._misses = 0


# ===========================================================================
# Tests: BoundedTTLCache - Basic Operations
# ===========================================================================


class TestBoundedTTLCacheBasic:
    """Tests for basic get/set operations."""

    def test_get_miss_on_empty_cache(self):
        cache = BoundedTTLCache()
        hit, value = cache.get("nonexistent", 60.0)
        assert hit is False
        assert value is None

    def test_set_and_get_hit(self):
        cache = BoundedTTLCache()
        cache.set("key1", "value1")
        hit, value = cache.get("key1", 60.0)
        assert hit is True
        assert value == "value1"

    def test_set_overwrites_existing_key(self):
        cache = BoundedTTLCache()
        cache.set("key1", "original")
        cache.set("key1", "updated")
        hit, value = cache.get("key1", 60.0)
        assert hit is True
        assert value == "updated"

    def test_get_expired_entry_returns_miss(self):
        cache = BoundedTTLCache()
        cache.set("key1", "value1")
        # Use a very small TTL that has already expired
        time.sleep(0.05)
        hit, value = cache.get("key1", 0.01)
        assert hit is False
        assert value is None

    def test_expired_entry_is_removed(self):
        cache = BoundedTTLCache()
        cache.set("key1", "value1")
        time.sleep(0.05)
        cache.get("key1", 0.01)  # triggers removal
        assert "key1" not in cache

    def test_get_moves_to_end_lru(self):
        cache = BoundedTTLCache(max_entries=10)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        # Access "a" to move it to end
        cache.get("a", 60.0)
        items = cache.items()
        keys = [k for k, _ in items]
        assert keys[-1] == "a"

    def test_set_different_value_types(self):
        cache = BoundedTTLCache()
        cache.set("str", "hello")
        cache.set("int", 42)
        cache.set("list", [1, 2, 3])
        cache.set("dict", {"a": 1})
        cache.set("none", None)

        assert cache.get("str", 60.0) == (True, "hello")
        assert cache.get("int", 60.0) == (True, 42)
        assert cache.get("list", 60.0) == (True, [1, 2, 3])
        assert cache.get("dict", 60.0) == (True, {"a": 1})
        assert cache.get("none", 60.0) == (True, None)

    def test_len_empty_cache(self):
        cache = BoundedTTLCache()
        assert len(cache) == 0

    def test_len_after_sets(self):
        cache = BoundedTTLCache()
        cache.set("a", 1)
        cache.set("b", 2)
        assert len(cache) == 2

    def test_contains_present_key(self):
        cache = BoundedTTLCache()
        cache.set("key1", "value")
        assert "key1" in cache

    def test_contains_absent_key(self):
        cache = BoundedTTLCache()
        assert "missing" not in cache

    def test_items_returns_copy(self):
        cache = BoundedTTLCache()
        cache.set("a", 1)
        cache.set("b", 2)
        items = cache.items()
        assert len(items) == 2
        # Items are (key, (timestamp, value)) tuples
        keys = [k for k, _ in items]
        assert "a" in keys
        assert "b" in keys


# ===========================================================================
# Tests: BoundedTTLCache - Eviction
# ===========================================================================


class TestBoundedTTLCacheEviction:
    """Tests for eviction behavior when max_entries is reached."""

    def test_evicts_oldest_when_full(self):
        cache = BoundedTTLCache(max_entries=3, evict_percent=0.34)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        # Adding a 4th entry should evict the oldest
        cache.set("d", 4)
        assert len(cache) <= 3
        # "a" should be evicted (oldest)
        assert "a" not in cache
        assert "d" in cache

    def test_evict_count_at_least_one(self):
        cache = BoundedTTLCache(max_entries=5, evict_percent=0.0)
        # evict_count = max(1, int(5 * 0.0)) = 1
        for i in range(5):
            cache.set(f"k{i}", i)
        cache.set("overflow", 99)
        # Should have evicted at least 1
        assert len(cache) <= 5

    def test_eviction_preserves_newest(self):
        cache = BoundedTTLCache(max_entries=3, evict_percent=0.5)
        cache.set("old1", 1)
        cache.set("old2", 2)
        cache.set("old3", 3)
        cache.set("new1", 4)
        # newest entry must survive
        hit, value = cache.get("new1", 60.0)
        assert hit is True
        assert value == 4

    def test_large_evict_percent(self):
        cache = BoundedTTLCache(max_entries=10, evict_percent=0.5)
        for i in range(10):
            cache.set(f"k{i}", i)
        cache.set("overflow", 99)
        # 50% eviction = 5 evicted, plus 1 new = 6 entries
        assert len(cache) <= 6

    def test_repeated_evictions(self):
        cache = BoundedTTLCache(max_entries=2, evict_percent=0.5)
        for i in range(10):
            cache.set(f"k{i}", i)
        assert len(cache) <= 2


# ===========================================================================
# Tests: BoundedTTLCache - clear and invalidate
# ===========================================================================


class TestBoundedTTLCacheClear:
    """Tests for clear and invalidate_containing methods."""

    def test_clear_all(self):
        cache = BoundedTTLCache()
        cache.set("a", 1)
        cache.set("b", 2)
        count = cache.clear()
        assert count == 2
        assert len(cache) == 0

    def test_clear_empty_cache(self):
        cache = BoundedTTLCache()
        count = cache.clear()
        assert count == 0

    def test_clear_with_prefix(self):
        cache = BoundedTTLCache()
        cache.set("leaderboard:top10", 1)
        cache.set("leaderboard:all", 2)
        cache.set("dashboard:stats", 3)
        count = cache.clear("leaderboard")
        assert count == 2
        assert len(cache) == 1
        assert "dashboard:stats" in cache

    def test_clear_with_prefix_no_match(self):
        cache = BoundedTTLCache()
        cache.set("a", 1)
        count = cache.clear("nonexistent_prefix")
        assert count == 0
        assert len(cache) == 1

    def test_invalidate_containing(self):
        cache = BoundedTTLCache()
        cache.set("debate:abc123:result", 1)
        cache.set("debate:abc123:votes", 2)
        cache.set("debate:xyz789:result", 3)
        count = cache.invalidate_containing("abc123")
        assert count == 2
        assert len(cache) == 1
        assert "debate:xyz789:result" in cache

    def test_invalidate_containing_no_match(self):
        cache = BoundedTTLCache()
        cache.set("a", 1)
        count = cache.invalidate_containing("zzz")
        assert count == 0
        assert len(cache) == 1

    def test_invalidate_containing_empty_cache(self):
        cache = BoundedTTLCache()
        count = cache.invalidate_containing("anything")
        assert count == 0


# ===========================================================================
# Tests: BoundedTTLCache - Stats
# ===========================================================================


class TestBoundedTTLCacheStats:
    """Tests for cache statistics."""

    def test_stats_initial(self):
        cache = BoundedTTLCache(max_entries=100)
        stats = cache.stats
        assert stats["entries"] == 0
        assert stats["max_entries"] == 100
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0

    def test_stats_after_hits_and_misses(self):
        cache = BoundedTTLCache()
        cache.set("a", 1)
        cache.get("a", 60.0)  # hit
        cache.get("b", 60.0)  # miss
        cache.get("c", 60.0)  # miss
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["hit_rate"] == pytest.approx(1 / 3)

    def test_stats_hit_rate_all_hits(self):
        cache = BoundedTTLCache()
        cache.set("a", 1)
        for _ in range(5):
            cache.get("a", 60.0)
        stats = cache.stats
        assert stats["hits"] == 5
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 1.0

    def test_stats_entries_count(self):
        cache = BoundedTTLCache()
        cache.set("a", 1)
        cache.set("b", 2)
        assert cache.stats["entries"] == 2


# ===========================================================================
# Tests: ttl_cache decorator (sync)
# ===========================================================================


class TestTTLCacheDecorator:
    """Tests for the ttl_cache sync decorator."""

    def test_caches_function_result(self):
        call_count = 0

        @ttl_cache(ttl_seconds=60.0, key_prefix="test", skip_first=False)
        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = expensive_func(5)
        result2 = expensive_func(5)
        assert result1 == 10
        assert result2 == 10
        assert call_count == 1  # Second call used cache

    def test_different_args_not_cached(self):
        call_count = 0

        @ttl_cache(ttl_seconds=60.0, key_prefix="test2", skip_first=False)
        def func(x):
            nonlocal call_count
            call_count += 1
            return x

        func(1)
        func(2)
        assert call_count == 2

    def test_skip_first_true_for_methods(self):
        """With skip_first=True, self is excluded from cache key."""
        call_count = 0

        class MyClass:
            @ttl_cache(ttl_seconds=60.0, key_prefix="method_test", skip_first=True)
            def compute(self, x):
                nonlocal call_count
                call_count += 1
                return x * 3

        obj1 = MyClass()
        obj2 = MyClass()
        result1 = obj1.compute(4)
        result2 = obj2.compute(4)
        assert result1 == 12
        assert result2 == 12
        assert call_count == 1  # Same args (excluding self) = cache hit

    def test_skip_first_false_for_standalone(self):
        call_count = 0

        @ttl_cache(ttl_seconds=60.0, key_prefix="standalone", skip_first=False)
        def standalone(a, b):
            nonlocal call_count
            call_count += 1
            return a + b

        standalone(1, 2)
        standalone(1, 2)
        assert call_count == 1

    def test_kwargs_included_in_cache_key(self):
        call_count = 0

        @ttl_cache(ttl_seconds=60.0, key_prefix="kw", skip_first=False)
        def func(x, option=False):
            nonlocal call_count
            call_count += 1
            return x

        func(1, option=False)
        func(1, option=True)
        assert call_count == 2  # Different kwargs = different cache key

    def test_default_ttl(self):
        """Default TTL is 60 seconds."""
        @ttl_cache(key_prefix="defaults", skip_first=False)
        def func():
            return "result"

        result = func()
        assert result == "result"

    def test_preserves_function_name(self):
        @ttl_cache(ttl_seconds=10.0, key_prefix="wrap", skip_first=False)
        def my_function():
            pass

        assert my_function.__name__ == "my_function"

    def test_key_prefix_used_for_metrics(self):
        """When key_prefix is provided, it's used for metrics recording."""
        @ttl_cache(ttl_seconds=60.0, key_prefix="my_prefix", skip_first=False)
        def func():
            return 42

        # Just verify it runs without error
        func()


# ===========================================================================
# Tests: async_ttl_cache decorator
# ===========================================================================


class TestAsyncTTLCacheDecorator:
    """Tests for the async_ttl_cache decorator."""

    @pytest.mark.asyncio
    async def test_caches_async_result(self):
        call_count = 0

        @async_ttl_cache(ttl_seconds=60.0, key_prefix="async_test", skip_first=False)
        async def expensive_async(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = await expensive_async(5)
        result2 = await expensive_async(5)
        assert result1 == 10
        assert result2 == 10
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_different_args_not_cached_async(self):
        call_count = 0

        @async_ttl_cache(ttl_seconds=60.0, key_prefix="async_diff", skip_first=False)
        async def func(x):
            nonlocal call_count
            call_count += 1
            return x

        await func(1)
        await func(2)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_skip_first_async_method(self):
        call_count = 0

        class AsyncService:
            @async_ttl_cache(ttl_seconds=60.0, key_prefix="async_method", skip_first=True)
            async def query(self, key):
                nonlocal call_count
                call_count += 1
                return f"result:{key}"

        svc1 = AsyncService()
        svc2 = AsyncService()
        r1 = await svc1.query("test")
        r2 = await svc2.query("test")
        assert r1 == "result:test"
        assert r2 == "result:test"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_kwargs_in_key(self):
        call_count = 0

        @async_ttl_cache(ttl_seconds=60.0, key_prefix="async_kw", skip_first=False)
        async def func(x, flag=False):
            nonlocal call_count
            call_count += 1
            return x

        await func(1, flag=False)
        await func(1, flag=True)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_preserves_function_name(self):
        @async_ttl_cache(ttl_seconds=10.0, key_prefix="async_wrap", skip_first=False)
        async def my_async_function():
            pass

        assert my_async_function.__name__ == "my_async_function"


# ===========================================================================
# Tests: invalidates_cache decorator
# ===========================================================================


class TestInvalidatesCacheDecorator:
    """Tests for the invalidates_cache decorator."""

    def test_sync_invalidation(self):
        """Sync function decorated with invalidates_cache clears entries."""
        _cache.set("leaderboard:top10", "data")
        _cache.set("lb_rankings:all", "data")

        @invalidates_cache("elo_updated")
        def update_elo():
            return "done"

        result = update_elo()
        assert result == "done"
        # leaderboard entries should be cleared
        hit, _ = _cache.get("leaderboard:top10", 60.0)
        assert hit is False

    @pytest.mark.asyncio
    async def test_async_invalidation(self):
        """Async function decorated with invalidates_cache clears entries."""
        _cache.set("dashboard_debates:list", "data")

        @invalidates_cache("debate_completed")
        async def complete_debate():
            return "completed"

        result = await complete_debate()
        assert result == "completed"
        hit, _ = _cache.get("dashboard_debates:list", 60.0)
        assert hit is False

    def test_multiple_events(self):
        """Multiple events clear all associated prefixes."""
        _cache.set("leaderboard:top", "data")
        _cache.set("lb_matches:recent", "data")

        @invalidates_cache("elo_updated", "match_recorded")
        def record_match():
            return True

        result = record_match()
        assert result is True
        assert len(_cache) == 0

    def test_invalidation_only_on_success(self):
        """Invalidation happens after successful function execution."""
        _cache.set("leaderboard:top", "data")

        @invalidates_cache("elo_updated")
        def failing_func():
            raise ValueError("failed")

        with pytest.raises(ValueError):
            failing_func()
        # Cache should NOT be cleared because function raised
        hit, _ = _cache.get("leaderboard:top", 60.0)
        assert hit is True

    def test_unknown_event_does_nothing(self):
        """An event not in CACHE_INVALIDATION_MAP clears nothing."""
        _cache.set("some_key", "data")

        @invalidates_cache("nonexistent_event")
        def func():
            return "ok"

        func()
        hit, _ = _cache.get("some_key", 60.0)
        assert hit is True


# ===========================================================================
# Tests: Module-level functions
# ===========================================================================


class TestClearCache:
    """Tests for the clear_cache function."""

    def test_clear_all_entries(self):
        _cache.set("a", 1)
        _cache.set("b", 2)
        count = clear_cache()
        assert count == 2
        assert len(_cache) == 0

    def test_clear_with_prefix(self):
        _cache.set("dashboard:stats", 1)
        _cache.set("dashboard:debates", 2)
        _cache.set("leaderboard:top", 3)
        count = clear_cache("dashboard")
        assert count == 2
        assert len(_cache) == 1

    def test_clear_empty(self):
        count = clear_cache()
        assert count == 0

    def test_clear_no_matching_prefix(self):
        _cache.set("x", 1)
        count = clear_cache("zzz")
        assert count == 0
        assert len(_cache) == 1


class TestGetCacheStats:
    """Tests for the get_cache_stats function."""

    def test_returns_stats_dict(self):
        stats = get_cache_stats()
        assert isinstance(stats, dict)
        assert "entries" in stats
        assert "max_entries" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats

    def test_reflects_global_cache_state(self):
        _cache.set("key", "val")
        _cache.get("key", 60.0)
        stats = get_cache_stats()
        assert stats["entries"] == 1
        assert stats["hits"] == 1


class TestInvalidateOnEvent:
    """Tests for the invalidate_on_event function."""

    def test_known_event_clears_associated_prefixes(self):
        _cache.set("leaderboard:top10", "data")
        _cache.set("agents_list:all", "data")
        cleared = invalidate_on_event("elo_updated")
        assert cleared == 2

    def test_unknown_event_returns_zero(self):
        _cache.set("something", "data")
        cleared = invalidate_on_event("totally_unknown_event")
        assert cleared == 0
        assert len(_cache) == 1

    def test_no_matching_entries(self):
        """Event maps to prefixes, but no cache entries match."""
        cleared = invalidate_on_event("elo_updated")
        assert cleared == 0

    def test_debate_completed_clears_many_prefixes(self):
        """debate_completed has many associated prefixes."""
        for prefix in CACHE_INVALIDATION_MAP["debate_completed"]:
            _cache.set(f"{prefix}:data", "value")
        cleared = invalidate_on_event("debate_completed")
        expected = len(CACHE_INVALIDATION_MAP["debate_completed"])
        assert cleared == expected

    def test_debate_started_clears_dashboard_debates(self):
        _cache.set("dashboard_debates:list", "data")
        _cache.set("dashboard_debates:page2", "data")
        cleared = invalidate_on_event("debate_started")
        assert cleared == 2

    def test_agent_updated_clears_agent_prefixes(self):
        _cache.set("agent_profile:claude", "data")
        _cache.set("agents_list:all", "data")
        _cache.set("lb_introspection:data", "data")
        cleared = invalidate_on_event("agent_updated")
        assert cleared == 3

    def test_memory_updated_clears_memory_prefixes(self):
        _cache.set("analytics_memory:data", "data")
        _cache.set("critique_patterns:data", "data")
        _cache.set("critique_stats:data", "data")
        cleared = invalidate_on_event("memory_updated")
        assert cleared == 3

    def test_consensus_reached_clears_consensus_prefixes(self):
        _cache.set("consensus_stats:data", "data")
        _cache.set("consensus_settled:data", "data")
        _cache.set("consensus_similar:data", "data")
        cleared = invalidate_on_event("consensus_reached")
        assert cleared == 3


class TestInvalidateCache:
    """Tests for the invalidate_cache function (data source to event mapping)."""

    def test_elo_source(self):
        _cache.set("leaderboard:data", "v")
        cleared = invalidate_cache("elo")
        assert cleared >= 1

    def test_memory_source(self):
        _cache.set("analytics_memory:data", "v")
        cleared = invalidate_cache("memory")
        assert cleared >= 1

    def test_debates_source(self):
        _cache.set("dashboard_debates:data", "v")
        cleared = invalidate_cache("debates")
        assert cleared >= 1

    def test_consensus_source(self):
        _cache.set("consensus_stats:data", "v")
        cleared = invalidate_cache("consensus")
        assert cleared >= 1

    def test_agent_source(self):
        _cache.set("agent_profile:data", "v")
        cleared = invalidate_cache("agent")
        assert cleared >= 1

    def test_calibration_source_maps_to_elo(self):
        _cache.set("leaderboard:data", "v")
        cleared = invalidate_cache("calibration")
        assert cleared >= 1

    def test_unknown_source_fallback_to_prefix(self):
        """Unknown data source tries to clear by prefix."""
        _cache.set("custom_prefix:data", "v")
        cleared = invalidate_cache("custom_prefix")
        assert cleared == 1

    def test_unknown_source_no_match(self):
        cleared = invalidate_cache("no_such_source")
        assert cleared == 0


class TestInvalidateLeaderboardCache:
    """Tests for the invalidate_leaderboard_cache convenience function."""

    def test_clears_leaderboard_entries(self):
        _cache.set("leaderboard:top", "data")
        _cache.set("lb_rankings:all", "data")
        cleared = invalidate_leaderboard_cache()
        assert cleared >= 2

    def test_no_entries_returns_zero(self):
        cleared = invalidate_leaderboard_cache()
        assert cleared == 0


class TestInvalidateAgentCache:
    """Tests for the invalidate_agent_cache function."""

    def test_with_agent_name(self):
        _cache.set("agent:claude-opus:profile", "data")
        _cache.set("agent:gpt-4:profile", "data")
        cleared = invalidate_agent_cache("claude-opus")
        assert cleared == 1
        assert len(_cache) == 1

    def test_without_agent_name_clears_all_agent_caches(self):
        _cache.set("agent_profile:claude", "data")
        _cache.set("agents_list:all", "data")
        cleared = invalidate_agent_cache()
        assert cleared >= 2

    def test_agent_name_no_match(self):
        _cache.set("agent:other:data", "v")
        cleared = invalidate_agent_cache("nonexistent_agent")
        assert cleared == 0

    def test_agent_name_none_triggers_event(self):
        _cache.set("agent_profile:x", "v")
        cleared = invalidate_agent_cache(None)
        assert cleared >= 1


class TestInvalidateDebateCache:
    """Tests for the invalidate_debate_cache function."""

    def test_with_debate_id(self):
        _cache.set("debate:abc-123:result", "data")
        _cache.set("debate:abc-123:votes", "data")
        _cache.set("debate:xyz-789:result", "data")
        cleared = invalidate_debate_cache("abc-123")
        assert cleared == 2
        assert len(_cache) == 1

    def test_without_debate_id_clears_all_debate_caches(self):
        _cache.set("dashboard_debates:list", "data")
        _cache.set("dashboard_overview:main", "data")
        cleared = invalidate_debate_cache()
        assert cleared >= 2

    def test_debate_id_no_match(self):
        _cache.set("debate:other:data", "v")
        cleared = invalidate_debate_cache("nonexistent_id")
        assert cleared == 0

    def test_debate_id_none_triggers_event(self):
        _cache.set("dashboard_debates:x", "v")
        cleared = invalidate_debate_cache(None)
        assert cleared >= 1


# ===========================================================================
# Tests: _invalidate_events helper
# ===========================================================================


class TestInvalidateEvents:
    """Tests for the _invalidate_events helper."""

    def test_returns_total_cleared(self):
        _cache.set("leaderboard:data", "v")
        _cache.set("lb_matches:data", "v")
        total = _invalidate_events(("elo_updated", "match_recorded"), "test_func")
        assert total >= 2

    def test_empty_events_tuple(self):
        total = _invalidate_events((), "test_func")
        assert total == 0

    def test_single_event(self):
        _cache.set("dashboard_debates:x", "v")
        total = _invalidate_events(("debate_started",), "test_func")
        assert total == 1


# ===========================================================================
# Tests: CACHE_INVALIDATION_MAP
# ===========================================================================


class TestCacheInvalidationMap:
    """Tests for the CACHE_INVALIDATION_MAP structure."""

    def test_all_keys_are_strings(self):
        for key in CACHE_INVALIDATION_MAP:
            assert isinstance(key, str)

    def test_all_values_are_lists_of_strings(self):
        for key, prefixes in CACHE_INVALIDATION_MAP.items():
            assert isinstance(prefixes, list), f"{key} value is not a list"
            for prefix in prefixes:
                assert isinstance(prefix, str), f"{key} has non-string prefix: {prefix}"

    def test_expected_events_exist(self):
        expected = {"elo_updated", "match_recorded", "debate_completed",
                    "debate_started", "agent_updated", "memory_updated",
                    "consensus_reached"}
        assert expected == set(CACHE_INVALIDATION_MAP.keys())

    def test_elo_updated_prefixes(self):
        prefixes = CACHE_INVALIDATION_MAP["elo_updated"]
        assert "leaderboard" in prefixes
        assert "lb_rankings" in prefixes

    def test_debate_completed_prefixes(self):
        prefixes = CACHE_INVALIDATION_MAP["debate_completed"]
        assert "dashboard_debates" in prefixes
        assert "dashboard_overview" in prefixes


# ===========================================================================
# Tests: get_handler_cache and _register_handler_cache
# ===========================================================================


class TestGetHandlerCache:
    """Tests for get_handler_cache and ServiceRegistry integration."""

    def test_returns_global_cache(self):
        cache = get_handler_cache()
        assert cache is _cache

    def test_returns_bounded_ttl_cache(self):
        cache = get_handler_cache()
        assert isinstance(cache, BoundedTTLCache)

    def test_register_does_not_fail_without_services(self):
        """Even if ServiceRegistry is unavailable, get_handler_cache works."""
        import aragora.server.handlers.admin.cache as cache_mod
        # Reset registration state
        original = cache_mod._handler_cache_registered
        cache_mod._handler_cache_registered = False
        try:
            cache = get_handler_cache()
            assert cache is _cache
        finally:
            cache_mod._handler_cache_registered = original


# ===========================================================================
# Tests: _get_metrics and _noop_metric
# ===========================================================================


class TestGetMetrics:
    """Tests for lazy metric loading."""

    def test_noop_metric_does_nothing(self):
        """_noop_metric accepts a string and returns None."""
        result = _noop_metric("test")
        assert result is None

    def test_get_metrics_returns_callables(self):
        hit_fn, miss_fn = _get_metrics()
        assert callable(hit_fn)
        assert callable(miss_fn)

    def test_get_metrics_returns_tuple(self):
        result = _get_metrics()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_metrics_functions_accept_string_arg(self):
        """Metric functions should accept a string argument without error."""
        hit_fn, miss_fn = _get_metrics()
        hit_fn("test_metric")
        miss_fn("test_metric")


# ===========================================================================
# Tests: Environment configuration
# ===========================================================================


class TestCacheConfiguration:
    """Tests for cache configuration from environment variables."""

    def test_default_max_entries(self):
        """Default max entries is 1000."""
        from aragora.server.handlers.admin.cache import CACHE_MAX_ENTRIES
        assert CACHE_MAX_ENTRIES == int(
            __import__("os").environ.get("ARAGORA_CACHE_MAX_ENTRIES", "1000")
        )

    def test_default_evict_percent(self):
        """Default evict percent is 0.1."""
        from aragora.server.handlers.admin.cache import CACHE_EVICT_PERCENT
        assert CACHE_EVICT_PERCENT == float(
            __import__("os").environ.get("ARAGORA_CACHE_EVICT_PERCENT", "0.1")
        )


# ===========================================================================
# Tests: Edge cases and thread safety
# ===========================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_set_empty_string_key(self):
        cache = BoundedTTLCache()
        cache.set("", "empty_key")
        hit, value = cache.get("", 60.0)
        assert hit is True
        assert value == "empty_key"

    def test_clear_with_empty_prefix(self):
        cache = BoundedTTLCache()
        cache.set("a", 1)
        cache.set("b", 2)
        # Empty prefix matches all keys (all keys start with "")
        count = cache.clear("")
        assert count == 2
        assert len(cache) == 0

    def test_invalidate_containing_empty_string(self):
        cache = BoundedTTLCache()
        cache.set("a", 1)
        cache.set("b", 2)
        # Empty substring is contained in every key
        count = cache.invalidate_containing("")
        assert count == 2

    def test_very_long_key(self):
        cache = BoundedTTLCache()
        long_key = "k" * 10000
        cache.set(long_key, "value")
        hit, value = cache.get(long_key, 60.0)
        assert hit is True
        assert value == "value"

    def test_unicode_keys(self):
        cache = BoundedTTLCache()
        cache.set("key_with_unicode_chars", "value")
        hit, value = cache.get("key_with_unicode_chars", 60.0)
        assert hit is True

    def test_max_entries_one(self):
        """Cache with max_entries=1 always has at most 1 entry."""
        cache = BoundedTTLCache(max_entries=1)
        cache.set("a", 1)
        cache.set("b", 2)
        assert len(cache) == 1
        hit, _ = cache.get("b", 60.0)
        assert hit is True

    def test_zero_ttl_always_misses(self):
        cache = BoundedTTLCache()
        cache.set("key", "value")
        time.sleep(0.001)
        hit, value = cache.get("key", 0.0)
        assert hit is False

    def test_negative_ttl_always_misses(self):
        cache = BoundedTTLCache()
        cache.set("key", "value")
        hit, value = cache.get("key", -1.0)
        assert hit is False

    def test_concurrent_set_and_get(self):
        """Basic thread safety test: concurrent access doesn't crash."""
        import threading
        cache = BoundedTTLCache(max_entries=100)
        errors = []

        def writer():
            try:
                for i in range(50):
                    cache.set(f"w_{i}", i)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(50):
                    cache.get(f"w_{i}", 60.0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_ttl_cache_with_no_args(self):
        """ttl_cache on a function with no args (skip_first=False)."""
        call_count = 0

        @ttl_cache(ttl_seconds=60.0, key_prefix="noargs", skip_first=False)
        def get_constant():
            nonlocal call_count
            call_count += 1
            return 42

        get_constant()
        get_constant()
        assert call_count == 1

    def test_ttl_cache_skip_first_with_no_args(self):
        """ttl_cache with skip_first=True on a no-arg function."""
        @ttl_cache(ttl_seconds=60.0, key_prefix="noargs_skip", skip_first=True)
        def func():
            return "ok"

        # skip_first with no args: cache_args = args (empty tuple since args is empty)
        result = func()
        assert result == "ok"

    def test_cache_stores_none_value(self):
        """Cache can store and retrieve None as a value."""
        cache = BoundedTTLCache()
        cache.set("null_val", None)
        hit, value = cache.get("null_val", 60.0)
        assert hit is True
        assert value is None

    def test_ttl_cache_default_key_prefix_empty(self):
        """When key_prefix is empty, func name is used for metrics."""
        call_count = 0

        @ttl_cache(ttl_seconds=60.0, key_prefix="", skip_first=False)
        def named_func():
            nonlocal call_count
            call_count += 1
            return "result"

        named_func()
        named_func()
        assert call_count == 1


# ===========================================================================
# Tests: TTL expiry behavior
# ===========================================================================


class TestTTLExpiry:
    """Tests for time-based expiry in both cache and decorators."""

    def test_entry_available_before_ttl(self):
        cache = BoundedTTLCache()
        cache.set("k", "v")
        # With a long TTL, entry should be available
        hit, value = cache.get("k", 3600.0)
        assert hit is True

    def test_entry_expired_after_ttl(self):
        cache = BoundedTTLCache()
        cache.set("k", "v")
        time.sleep(0.05)
        hit, _ = cache.get("k", 0.01)
        assert hit is False

    def test_decorator_ttl_expiry(self):
        call_count = 0

        @ttl_cache(ttl_seconds=0.05, key_prefix="expiry_test", skip_first=False)
        def func():
            nonlocal call_count
            call_count += 1
            return "data"

        func()
        assert call_count == 1
        time.sleep(0.1)
        func()
        assert call_count == 2  # Cache expired, re-computed
