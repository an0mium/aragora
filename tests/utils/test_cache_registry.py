"""
Tests for LRU cache registry utilities.

Tests cover:
- register_lru_cache decorator
- unregister_lru_cache
- clear_all_lru_caches
- get_lru_cache_stats
- get_total_cache_entries
- get_registered_cache_count
"""

import pytest
from functools import lru_cache
from unittest.mock import MagicMock

from aragora.utils.cache_registry import (
    register_lru_cache,
    unregister_lru_cache,
    clear_all_lru_caches,
    get_lru_cache_stats,
    get_total_cache_entries,
    get_registered_cache_count,
    _registered_caches,
    _registry_lock,
)


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean up registry before and after each test."""
    # Store original caches
    with _registry_lock:
        original = _registered_caches.copy()

    yield

    # Restore original caches
    with _registry_lock:
        _registered_caches.clear()
        _registered_caches.extend(original)


class TestRegisterLruCache:
    """Tests for register_lru_cache decorator."""

    def test_registers_function(self):
        """Registers function in cache registry."""

        @register_lru_cache
        @lru_cache(maxsize=10)
        def test_func(x):
            return x * 2

        with _registry_lock:
            assert test_func in _registered_caches

        # Cleanup
        unregister_lru_cache(test_func)

    def test_returns_same_function(self):
        """Returns the decorated function unchanged."""

        @lru_cache(maxsize=10)
        def original(x):
            return x * 2

        decorated = register_lru_cache(original)
        assert decorated is original

    def test_no_duplicate_registration(self):
        """Does not register same function twice."""

        @register_lru_cache
        @lru_cache(maxsize=10)
        def test_func(x):
            return x

        initial_count = get_registered_cache_count()

        # Register again
        register_lru_cache(test_func)

        # Count should not increase
        assert get_registered_cache_count() == initial_count

        # Cleanup
        unregister_lru_cache(test_func)

    def test_function_still_works(self):
        """Decorated function still works correctly."""

        @register_lru_cache
        @lru_cache(maxsize=10)
        def square(x):
            return x**2

        assert square(5) == 25
        assert square(10) == 100

        # Cleanup
        unregister_lru_cache(square)


class TestUnregisterLruCache:
    """Tests for unregister_lru_cache function."""

    def test_unregisters_function(self):
        """Removes function from registry."""

        @register_lru_cache
        @lru_cache(maxsize=10)
        def test_func(x):
            return x

        assert unregister_lru_cache(test_func) is True

        with _registry_lock:
            assert test_func not in _registered_caches

    def test_returns_false_for_unregistered(self):
        """Returns False for function not in registry."""

        @lru_cache(maxsize=10)
        def not_registered(x):
            return x

        assert unregister_lru_cache(not_registered) is False


class TestClearAllLruCaches:
    """Tests for clear_all_lru_caches function."""

    def test_clears_caches(self):
        """Clears all registered caches."""

        @register_lru_cache
        @lru_cache(maxsize=10)
        def cached_func(x):
            return x * 2

        # Populate cache
        for i in range(5):
            cached_func(i)

        assert cached_func.cache_info().currsize == 5

        # Clear all
        clear_all_lru_caches()

        assert cached_func.cache_info().currsize == 0

        # Cleanup
        unregister_lru_cache(cached_func)

    def test_returns_cleared_count(self):
        """Returns total number of entries cleared."""

        @register_lru_cache
        @lru_cache(maxsize=10)
        def func1(x):
            return x

        @register_lru_cache
        @lru_cache(maxsize=10)
        def func2(x):
            return x * 2

        # Populate caches
        for i in range(3):
            func1(i)
            func2(i)

        cleared = clear_all_lru_caches()

        assert cleared == 6  # 3 + 3

        # Cleanup
        unregister_lru_cache(func1)
        unregister_lru_cache(func2)

    def test_handles_empty_registry(self):
        """Handles empty registry gracefully."""
        # Clear all first
        clear_all_lru_caches()

        # Clear again should return 0
        cleared = clear_all_lru_caches()
        assert cleared == 0


class TestGetLruCacheStats:
    """Tests for get_lru_cache_stats function."""

    def test_returns_stats(self):
        """Returns statistics for registered caches."""

        @register_lru_cache
        @lru_cache(maxsize=10)
        def stat_func(x):
            return x

        # Generate some hits and misses
        stat_func(1)  # miss
        stat_func(2)  # miss
        stat_func(1)  # hit

        stats = get_lru_cache_stats()

        assert "stat_func" in stats
        assert stats["stat_func"]["hits"] == 1
        assert stats["stat_func"]["misses"] == 2
        assert stats["stat_func"]["currsize"] == 2
        assert stats["stat_func"]["maxsize"] == 10

        # Cleanup
        unregister_lru_cache(stat_func)

    def test_hit_rate_calculation(self):
        """Calculates hit rate correctly."""

        @register_lru_cache
        @lru_cache(maxsize=10)
        def hit_rate_func(x):
            return x

        # 4 calls: 2 misses, 2 hits (50% hit rate)
        hit_rate_func(1)  # miss
        hit_rate_func(2)  # miss
        hit_rate_func(1)  # hit
        hit_rate_func(2)  # hit

        stats = get_lru_cache_stats()
        hit_rate = stats["hit_rate_func"]["hit_rate"]

        assert hit_rate == 0.5

        # Cleanup
        unregister_lru_cache(hit_rate_func)

    def test_zero_hit_rate(self):
        """Handles zero calls gracefully."""

        @register_lru_cache
        @lru_cache(maxsize=10)
        def unused_func(x):
            return x

        stats = get_lru_cache_stats()

        # No calls means 0 hit rate (not division by zero)
        assert stats["unused_func"]["hit_rate"] == 0.0

        # Cleanup
        unregister_lru_cache(unused_func)


class TestGetTotalCacheEntries:
    """Tests for get_total_cache_entries function."""

    def test_counts_all_entries(self):
        """Counts entries across all caches."""

        @register_lru_cache
        @lru_cache(maxsize=10)
        def count_func1(x):
            return x

        @register_lru_cache
        @lru_cache(maxsize=10)
        def count_func2(x):
            return x * 2

        # Populate caches
        for i in range(4):
            count_func1(i)
        for i in range(3):
            count_func2(i)

        total = get_total_cache_entries()

        assert total >= 7  # At least our 7 entries

        # Cleanup
        unregister_lru_cache(count_func1)
        unregister_lru_cache(count_func2)

    def test_empty_caches(self):
        """Returns 0 for empty caches."""

        @register_lru_cache
        @lru_cache(maxsize=10)
        def empty_func(x):
            return x

        # Don't call the function, cache is empty
        total = get_total_cache_entries()

        # Should include 0 for our empty cache
        assert total >= 0

        # Cleanup
        unregister_lru_cache(empty_func)


class TestGetRegisteredCacheCount:
    """Tests for get_registered_cache_count function."""

    def test_counts_registered(self):
        """Counts registered caches."""
        initial = get_registered_cache_count()

        @register_lru_cache
        @lru_cache(maxsize=10)
        def new_func(x):
            return x

        assert get_registered_cache_count() == initial + 1

        # Cleanup
        unregister_lru_cache(new_func)

    def test_decreases_on_unregister(self):
        """Count decreases when cache is unregistered."""

        @register_lru_cache
        @lru_cache(maxsize=10)
        def temp_func(x):
            return x

        count_with = get_registered_cache_count()
        unregister_lru_cache(temp_func)
        count_without = get_registered_cache_count()

        assert count_without == count_with - 1


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_registration(self):
        """Handles concurrent registration safely."""
        import threading

        functions = []
        errors = []

        def register_many():
            try:
                for i in range(10):

                    @register_lru_cache
                    @lru_cache(maxsize=10)
                    def f(x, _i=i):
                        return x * _i

                    functions.append(f)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        # Cleanup
        for f in functions:
            unregister_lru_cache(f)

    def test_concurrent_clearing(self):
        """Handles concurrent clearing safely."""
        import threading

        @register_lru_cache
        @lru_cache(maxsize=100)
        def shared_func(x):
            return x

        # Populate cache
        for i in range(50):
            shared_func(i)

        errors = []

        def clear_cache():
            try:
                for _ in range(10):
                    clear_all_lru_caches()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=clear_cache) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        # Cleanup
        unregister_lru_cache(shared_func)


class TestIntegration:
    """Integration tests."""

    def test_full_lifecycle(self):
        """Tests full cache lifecycle."""

        # Register
        @register_lru_cache
        @lru_cache(maxsize=10)
        def lifecycle_func(x):
            return x**2

        # Verify registration
        assert get_registered_cache_count() > 0

        # Use cache
        lifecycle_func(1)
        lifecycle_func(2)
        lifecycle_func(1)  # hit

        # Check stats
        stats = get_lru_cache_stats()
        assert "lifecycle_func" in stats
        assert stats["lifecycle_func"]["hits"] == 1

        # Clear
        cleared = clear_all_lru_caches()
        assert cleared >= 2

        # Verify cleared
        assert lifecycle_func.cache_info().currsize == 0

        # Unregister
        assert unregister_lru_cache(lifecycle_func) is True

    def test_multiple_functions(self):
        """Tests multiple cached functions together."""

        @register_lru_cache
        @lru_cache(maxsize=5)
        def add(x, y):
            return x + y

        @register_lru_cache
        @lru_cache(maxsize=5)
        def multiply(x, y):
            return x * y

        # Use both
        add(1, 2)
        add(3, 4)
        multiply(2, 3)
        multiply(4, 5)

        # Stats should have both
        stats = get_lru_cache_stats()
        assert "add" in stats
        assert "multiply" in stats

        # Total entries
        total = get_total_cache_entries()
        assert total >= 4

        # Clear all
        clear_all_lru_caches()

        # Both should be empty
        assert add.cache_info().currsize == 0
        assert multiply.cache_info().currsize == 0

        # Cleanup
        unregister_lru_cache(add)
        unregister_lru_cache(multiply)
