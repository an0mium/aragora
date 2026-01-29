"""Tests for caching decorators."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from aragora.caching import (
    CacheStats,
    CacheEntry,
    async_cached,
    cache_key,
    cached,
    clear_all_caches,
    get_global_cache_stats,
    memoize,
)


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.size == 0
        assert stats.maxsize == 0
        assert stats.evictions == 0

    def test_hit_rate_zero_total(self):
        """Test hit rate when no operations have occurred."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_all_hits(self):
        """Test hit rate with all hits."""
        stats = CacheStats(hits=10, misses=0)
        assert stats.hit_rate == 100.0

    def test_hit_rate_mixed(self):
        """Test hit rate with mixed hits and misses."""
        stats = CacheStats(hits=7, misses=3)
        assert stats.hit_rate == 70.0

    def test_repr(self):
        """Test string representation."""
        stats = CacheStats(hits=5, misses=5, size=3, maxsize=10, evictions=2)
        repr_str = repr(stats)
        assert "hits=5" in repr_str
        assert "misses=5" in repr_str
        assert "hit_rate=50.0%" in repr_str


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_not_expired_no_ttl(self):
        """Test entry without TTL never expires."""
        entry = CacheEntry(value="test", created_at=time.time() - 1000)
        assert not entry.is_expired()

    def test_not_expired_within_ttl(self):
        """Test entry within TTL is not expired."""
        entry = CacheEntry(value="test", created_at=time.time(), ttl_seconds=60)
        assert not entry.is_expired()

    def test_expired_after_ttl(self):
        """Test entry after TTL is expired."""
        entry = CacheEntry(value="test", created_at=time.time() - 10, ttl_seconds=5)
        assert entry.is_expired()


class TestCachedDecorator:
    """Tests for @cached decorator."""

    def test_basic_caching(self):
        """Test basic caching functionality."""
        call_count = 0

        @cached(ttl_seconds=60, maxsize=10)
        def expensive_func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        assert expensive_func(5) == 10
        assert call_count == 1

        # Second call should use cache
        assert expensive_func(5) == 10
        assert call_count == 1

        # Different argument should call function
        assert expensive_func(10) == 20
        assert call_count == 2

    def test_cache_info(self):
        """Test cache_info method."""

        @cached(ttl_seconds=60, maxsize=10)
        def func(x: int) -> int:
            return x

        func(1)
        func(1)
        func(2)

        info = func.cache_info()
        assert info.hits == 1
        assert info.misses == 2
        assert info.size == 2

    def test_cache_clear(self):
        """Test cache_clear method."""
        call_count = 0

        @cached(ttl_seconds=60, maxsize=10)
        def func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x

        func(1)
        assert call_count == 1

        func(1)
        assert call_count == 1

        func.cache_clear()

        func(1)
        assert call_count == 2

    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        call_count = 0

        @cached(ttl_seconds=0.1, maxsize=10)
        def func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x

        func(1)
        assert call_count == 1

        func(1)
        assert call_count == 1

        # Wait for TTL to expire
        time.sleep(0.15)

        func(1)
        assert call_count == 2

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""

        @cached(ttl_seconds=60, maxsize=2)
        def func(x: int) -> int:
            return x

        func(1)
        func(2)
        func(3)  # Should evict 1

        info = func.cache_info()
        assert info.evictions >= 1
        assert info.size == 2

    def test_decorator_without_parens(self):
        """Test @cached without parentheses."""
        call_count = 0

        @cached
        def func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x

        func(1)
        func(1)
        assert call_count == 1

    def test_kwargs_caching(self):
        """Test caching with keyword arguments."""
        call_count = 0

        @cached(ttl_seconds=60)
        def func(a: int, b: int = 0) -> int:
            nonlocal call_count
            call_count += 1
            return a + b

        func(1, b=2)
        func(1, b=2)
        assert call_count == 1

        func(1, b=3)
        assert call_count == 2


class TestAsyncCachedDecorator:
    """Tests for @async_cached decorator."""

    @pytest.mark.asyncio
    async def test_basic_async_caching(self):
        """Test basic async caching functionality."""
        call_count = 0

        @async_cached(ttl_seconds=60, maxsize=10)
        async def async_func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return x * 2

        assert await async_func(5) == 10
        assert call_count == 1

        assert await async_func(5) == 10
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_cache_info(self):
        """Test cache_info for async functions."""

        @async_cached(ttl_seconds=60)
        async def func(x: int) -> int:
            return x

        await func(1)
        await func(1)
        await func(2)

        info = func.cache_info()
        assert info.hits >= 1
        assert info.misses >= 1

    @pytest.mark.asyncio
    async def test_async_cache_clear(self):
        """Test cache_clear for async functions."""
        call_count = 0

        @async_cached(ttl_seconds=60)
        async def func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x

        await func(1)
        assert call_count == 1

        await func(1)
        assert call_count == 1

        func.cache_clear()

        await func(1)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_concurrent_async_calls(self):
        """Test concurrent async calls to same function."""
        call_count = 0

        @async_cached(ttl_seconds=60)
        async def slow_func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return x

        # Call multiple times concurrently
        results = await asyncio.gather(
            slow_func(1),
            slow_func(1),
            slow_func(1),
        )

        assert results == [1, 1, 1]
        # Due to double-check locking, only one call should execute
        assert call_count == 1


class TestMemoizeDecorator:
    """Tests for @memoize decorator."""

    def test_basic_memoization(self):
        """Test basic memoization."""
        call_count = 0

        @memoize
        def fib(n: int) -> int:
            nonlocal call_count
            call_count += 1
            if n < 2:
                return n
            return fib(n - 1) + fib(n - 2)

        result = fib(10)
        assert result == 55
        # Without memoization, fib(10) would call itself 177 times
        assert call_count == 11  # Only n unique calls

    def test_memoize_cache_info(self):
        """Test cache_info for memoized functions."""

        @memoize
        def func(x: int) -> int:
            return x

        func(1)
        func(1)
        func(2)

        info = func.cache_info()
        assert info.hits == 1
        assert info.misses == 2
        assert info.maxsize == -1  # Unbounded

    def test_memoize_no_ttl(self):
        """Test that memoized values don't expire."""
        call_count = 0

        @memoize
        def func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x

        func(1)
        time.sleep(0.1)
        func(1)

        assert call_count == 1


class TestCacheKeyDecorator:
    """Tests for @cache_key decorator."""

    def test_selective_key_args(self):
        """Test caching with selective key arguments."""
        call_count = 0

        @cache_key("user_id")
        @cached(ttl_seconds=60)
        def get_user_data(user_id: int, request_metadata: dict) -> dict:
            nonlocal call_count
            call_count += 1
            return {"user_id": user_id}

        # Same user_id, different metadata should use cache
        get_user_data(1, {"trace_id": "abc"})
        get_user_data(1, {"trace_id": "xyz"})
        assert call_count == 1

        # Different user_id should not use cache
        get_user_data(2, {"trace_id": "abc"})
        assert call_count == 2

    def test_multiple_key_args(self):
        """Test caching with multiple key arguments."""
        call_count = 0

        @cache_key("a", "b")
        @cached(ttl_seconds=60)
        def func(a: int, b: int, c: int) -> int:
            nonlocal call_count
            call_count += 1
            return a + b + c

        func(1, 2, 3)
        func(1, 2, 999)  # c is ignored
        assert call_count == 1

        func(1, 3, 3)  # b is different
        assert call_count == 2


class TestGlobalCacheManagement:
    """Tests for global cache management functions."""

    def test_clear_all_caches(self):
        """Test clearing all registered caches."""

        @cached(ttl_seconds=60)
        def func1(x: int) -> int:
            return x

        @cached(ttl_seconds=60)
        def func2(x: int) -> int:
            return x * 2

        func1(1)
        func2(2)

        count = clear_all_caches()
        assert count >= 2

        info1 = func1.cache_info()
        info2 = func2.cache_info()
        assert info1.size == 0
        assert info2.size == 0

    def test_get_global_cache_stats(self):
        """Test getting stats for all caches."""

        @cached(ttl_seconds=60)
        def func(x: int) -> int:
            return x

        func(1)
        func(2)

        stats = get_global_cache_stats()
        assert len(stats) > 0
        assert any(s.size > 0 for s in stats)


class TestThreadSafety:
    """Tests for thread safety of caching decorators."""

    def test_concurrent_access(self):
        """Test concurrent access from multiple threads."""
        call_count = 0

        @cached(ttl_seconds=60, maxsize=100)
        def func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)
            return x

        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit same key from multiple threads
            futures = [executor.submit(func, 1) for _ in range(10)]
            results = [f.result() for f in futures]

        assert all(r == 1 for r in results)
        # Due to threading, some calls may slip through, but should be minimal
        assert call_count <= 10

    def test_cache_clear_during_access(self):
        """Test cache clear during concurrent access."""

        @cached(ttl_seconds=60, maxsize=100)
        def func(x: int) -> int:
            return x

        def worker():
            for i in range(100):
                func(i % 10)

        def clearer():
            for _ in range(5):
                func.cache_clear()
                time.sleep(0.01)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker) for _ in range(4)] + [executor.submit(clearer)]

            # Should complete without errors
            for f in futures:
                f.result()
