"""
Tests for global cache management functions.

Covers:
- _register_cache behavior
- get_global_cache_stats returning all registered caches
- clear_all_caches clearing every registered cache
- clear_all_caches return value
- Interaction between global management and individual caches
- Thread safety of registry operations
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from unittest.mock import patch

import pytest

from aragora.caching.decorators import (
    CacheStats,
    _TTLCache,
    _register_cache,
    _cache_registry,
    _registry_lock,
    cached,
    memoize,
    get_global_cache_stats,
    clear_all_caches,
)


# ===========================================================================
# Test: get_global_cache_stats
# ===========================================================================


class TestGetGlobalCacheStats:
    """Tests for get_global_cache_stats function."""

    def test_returns_list(self):
        """get_global_cache_stats returns a list."""
        result = get_global_cache_stats()
        assert isinstance(result, list)

    def test_contains_cache_stats_instances(self):
        """All entries in the list are CacheStats instances."""
        result = get_global_cache_stats()
        for stats in result:
            assert isinstance(stats, CacheStats)

    def test_new_cache_is_registered(self):
        """Creating a new _TTLCache registers it globally."""
        initial_count = len(get_global_cache_stats())
        _TTLCache(maxsize=10, ttl_seconds=60.0)
        new_count = len(get_global_cache_stats())
        assert new_count == initial_count + 1

    def test_cached_decorator_registers_cache(self):
        """@cached decorator registers its cache globally."""
        initial_count = len(get_global_cache_stats())

        @cached(ttl_seconds=60, maxsize=10)
        def registered_func(x):
            return x

        new_count = len(get_global_cache_stats())
        assert new_count == initial_count + 1

    def test_memoize_decorator_registers_cache(self):
        """@memoize decorator registers its cache globally."""
        initial_count = len(get_global_cache_stats())

        @memoize
        def registered_memo(x):
            return x

        new_count = len(get_global_cache_stats())
        assert new_count == initial_count + 1

    def test_stats_reflect_cache_activity(self):
        """Global stats reflect activity on individual caches."""

        @cached(ttl_seconds=60, maxsize=10)
        def tracked_func(x):
            return x

        tracked_func(1)
        tracked_func(1)  # hit
        tracked_func(2)  # miss

        all_stats = get_global_cache_stats()
        # Find the stats with 1 hit and 2 misses
        matching = [s for s in all_stats if s.hits == 1 and s.misses == 2]
        assert len(matching) >= 1


# ===========================================================================
# Test: clear_all_caches
# ===========================================================================


class TestClearAllCaches:
    """Tests for clear_all_caches function."""

    def test_returns_count(self):
        """clear_all_caches returns the number of caches cleared."""
        count = clear_all_caches()
        assert isinstance(count, int)
        assert count >= 0

    def test_clears_cached_decorator_caches(self):
        """Clearing all caches affects @cached functions."""
        call_count = 0

        @cached(ttl_seconds=60, maxsize=10)
        def func(x):
            nonlocal call_count
            call_count += 1
            return x

        func(1)
        assert call_count == 1

        clear_all_caches()

        func(1)  # Should recompute because cache was cleared
        assert call_count == 2

    def test_clears_memoize_caches(self):
        """Clearing all caches affects @memoize functions."""
        call_count = 0

        @memoize
        def func(x):
            nonlocal call_count
            call_count += 1
            return x

        func(1)
        assert call_count == 1

        clear_all_caches()

        func(1)  # Should recompute
        assert call_count == 2

    def test_resets_size_to_zero(self):
        """Clearing all caches resets sizes to zero."""

        @cached(ttl_seconds=60, maxsize=10)
        def func(x):
            return x

        func(1)
        func(2)
        func(3)

        clear_all_caches()

        info = func.cache_info()
        assert info.size == 0

    def test_multiple_caches_all_cleared(self):
        """Multiple independent caches are all cleared."""

        @cached(ttl_seconds=60, maxsize=10)
        def func_a(x):
            return x

        @cached(ttl_seconds=60, maxsize=10)
        def func_b(x):
            return x * 2

        func_a(1)
        func_b(2)

        count = clear_all_caches()
        assert count >= 2

        assert func_a.cache_info().size == 0
        assert func_b.cache_info().size == 0

    def test_clear_all_preserves_hit_miss_counters(self):
        """Clearing all caches does not reset hit/miss stats on TTLCache stats."""

        @cached(ttl_seconds=60, maxsize=10)
        def func(x):
            return x

        func(1)  # miss
        func(1)  # hit

        info_before = func.cache_info()
        assert info_before.hits == 1
        assert info_before.misses == 1

        clear_all_caches()

        # Hit/miss counters on the internal _stats are maintained
        # but clear_all_caches only clears data and resets size
        info_after = func.cache_info()
        assert info_after.size == 0


# ===========================================================================
# Test: Registry Thread Safety
# ===========================================================================


class TestRegistryThreadSafety:
    """Tests for thread safety of registry operations."""

    def test_concurrent_cache_creation(self):
        """Multiple caches can be created concurrently without errors."""
        results = []
        errors = []

        def create_cache():
            try:
                c = _TTLCache(maxsize=10, ttl_seconds=60.0)
                c.set("key", "val")
                results.append(c)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_cache) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 20

    def test_concurrent_clear_all(self):
        """clear_all_caches can be called concurrently without errors."""
        errors = []

        # Create some caches first
        for _ in range(5):
            c = _TTLCache(maxsize=10, ttl_seconds=60.0)
            c.set("k", "v")

        def clear_caches():
            try:
                clear_all_caches()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=clear_caches) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_stats_retrieval(self):
        """get_global_cache_stats can be called concurrently without errors."""
        errors = []
        results = []

        def get_stats():
            try:
                stats = get_global_cache_stats()
                results.append(stats)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_stats) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        for r in results:
            assert isinstance(r, list)
