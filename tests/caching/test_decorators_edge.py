"""Edge case tests for caching decorators.

Covers: CacheContext, exception handling, unhashable types,
zero TTL, maxsize=1, rapid operations, None return values,
_make_cache_key internals, and _cleanup_expired behavior.
"""

import asyncio
import time

import pytest

from aragora.caching import (
    CacheEntry,
    CacheStats,
    async_cached,
    cache_key,
    cached,
    memoize,
)
from aragora.caching.decorators import (
    CacheContext,
    _make_cache_key,
    _TTLCache,
)


# ---------------------------------------------------------------------------
# CacheContext tests
# ---------------------------------------------------------------------------


class TestCacheContext:
    """Tests for the CacheContext context manager."""

    def setup_method(self):
        """Ensure CacheContext is enabled before each test."""
        CacheContext._enabled = True

    def teardown_method(self):
        """Reset CacheContext after each test."""
        CacheContext._enabled = True

    def test_default_is_enabled(self):
        """CacheContext should be enabled by default."""
        assert CacheContext.is_enabled() is True

    def test_disable_within_context(self):
        """CacheContext(enabled=False) should disable caching inside the block."""
        with CacheContext(enabled=False):
            assert CacheContext.is_enabled() is False
        # Restored after exiting
        assert CacheContext.is_enabled() is True

    def test_nested_contexts_restore_correctly(self):
        """Nested CacheContext blocks should restore state in LIFO order."""
        assert CacheContext.is_enabled() is True

        with CacheContext(enabled=False):
            assert CacheContext.is_enabled() is False

            with CacheContext(enabled=True):
                assert CacheContext.is_enabled() is True

            # Inner context exited, should restore to False
            assert CacheContext.is_enabled() is False

        # Outer context exited, should restore to True
        assert CacheContext.is_enabled() is True

    def test_context_returns_self(self):
        """The __enter__ method should return the CacheContext instance."""
        ctx = CacheContext(enabled=False)
        with ctx as returned:
            assert returned is ctx


# ---------------------------------------------------------------------------
# Exception handling
# ---------------------------------------------------------------------------


class TestExceptionHandling:
    """Tests for cached function exception propagation."""

    def test_cached_function_exception_not_stored(self):
        """Exceptions raised by cached functions should propagate, not be cached."""
        call_count = 0

        @cached(ttl_seconds=60, maxsize=10)
        def failing_func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            if x == 0:
                raise ValueError("cannot be zero")
            return x * 2

        with pytest.raises(ValueError, match="cannot be zero"):
            failing_func(0)
        assert call_count == 1

        # A second call should invoke the function again (error was not cached)
        with pytest.raises(ValueError, match="cannot be zero"):
            failing_func(0)
        assert call_count == 2

        # Cache should be empty for that key
        info = failing_func.cache_info()
        assert info.size == 0

    @pytest.mark.asyncio
    async def test_async_cached_exception_not_stored(self):
        """Exceptions in async cached functions should propagate, not be cached."""
        call_count = 0

        @async_cached(ttl_seconds=60, maxsize=10)
        async def failing_async(x: int) -> int:
            nonlocal call_count
            call_count += 1
            if x < 0:
                raise RuntimeError("negative")
            return x

        with pytest.raises(RuntimeError, match="negative"):
            await failing_async(-1)
        assert call_count == 1

        with pytest.raises(RuntimeError, match="negative"):
            await failing_async(-1)
        assert call_count == 2

    def test_memoize_exception_not_stored(self):
        """Exceptions in memoized functions should propagate, not be cached."""
        call_count = 0

        @memoize
        def bad_func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            raise TypeError("oops")

        with pytest.raises(TypeError, match="oops"):
            bad_func(1)

        with pytest.raises(TypeError, match="oops"):
            bad_func(1)

        assert call_count == 2


# ---------------------------------------------------------------------------
# Unhashable types
# ---------------------------------------------------------------------------


class TestUnhashableTypes:
    """Tests for caching with unhashable argument types."""

    def test_dict_argument(self):
        """Caching should work with dict arguments via pickle/repr fallback."""
        call_count = 0

        @cached(ttl_seconds=60, maxsize=10)
        def process(data: dict) -> str:
            nonlocal call_count
            call_count += 1
            return str(data)

        result1 = process({"a": 1, "b": 2})
        result2 = process({"a": 1, "b": 2})
        assert result1 == result2
        assert call_count == 1

    def test_list_argument(self):
        """Caching should work with list arguments via pickle/repr fallback."""
        call_count = 0

        @cached(ttl_seconds=60, maxsize=10)
        def process(items: list) -> int:
            nonlocal call_count
            call_count += 1
            return sum(items)

        assert process([1, 2, 3]) == 6
        assert process([1, 2, 3]) == 6
        assert call_count == 1

        # Different list should produce a different cache key
        assert process([4, 5, 6]) == 15
        assert call_count == 2

    def test_nested_unhashable_argument(self):
        """Caching should work with nested dicts/lists."""
        call_count = 0

        @cached(ttl_seconds=60, maxsize=10)
        def process(payload: dict) -> str:
            nonlocal call_count
            call_count += 1
            return repr(payload)

        nested = {"key": [1, {"inner": True}]}
        process(nested)
        process(nested)
        assert call_count == 1


# ---------------------------------------------------------------------------
# Edge cases: zero TTL, maxsize=1, None return values
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for boundary conditions and edge cases."""

    def test_zero_ttl_never_caches(self):
        """A TTL of 0 means entries expire immediately."""
        call_count = 0

        @cached(ttl_seconds=0, maxsize=10)
        def func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x

        func(1)
        # Even a tiny elapsed time makes it expire with ttl=0
        time.sleep(0.001)
        func(1)
        assert call_count == 2

    def test_maxsize_one_evicts_previous(self):
        """With maxsize=1, inserting a new key should evict the old one."""
        call_count = 0

        @cached(ttl_seconds=60, maxsize=1)
        def func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x

        func(1)
        assert call_count == 1
        func(1)
        assert call_count == 1  # cached

        func(2)  # evicts key for arg=1
        assert call_count == 2

        func(1)  # must recompute, was evicted
        assert call_count == 3

        info = func.cache_info()
        assert info.evictions >= 1
        assert info.size == 1

    def test_none_return_value_is_cached(self):
        """A function returning None should still have its result cached."""
        call_count = 0

        @cached(ttl_seconds=60, maxsize=10)
        def func(x: int):
            nonlocal call_count
            call_count += 1
            return None

        result1 = func(1)
        result2 = func(1)
        assert result1 is None
        assert result2 is None
        assert call_count == 1

    def test_rapid_set_get_cycles(self):
        """Rapid set/get cycles should not corrupt the cache."""
        cache = _TTLCache(maxsize=5, ttl_seconds=60)
        for i in range(100):
            cache.set(f"key-{i}", i)

        # Only the last 5 entries should remain
        info = cache.cache_info()
        assert info.size == 5
        assert info.evictions == 95

        # The most recent entries should be present
        for i in range(95, 100):
            found, val = cache.get(f"key-{i}")
            assert found is True
            assert val == i

    def test_cache_entry_zero_ttl_expires_immediately(self):
        """A CacheEntry with ttl_seconds=0 should expire right away."""
        entry = CacheEntry(value="x", created_at=time.time(), ttl_seconds=0)
        # Any amount of elapsed time after creation means it is expired
        time.sleep(0.001)
        assert entry.is_expired() is True


# ---------------------------------------------------------------------------
# _make_cache_key tests
# ---------------------------------------------------------------------------


class TestMakeCacheKey:
    """Tests for the _make_cache_key helper."""

    def test_same_args_produce_same_key(self):
        """Identical arguments should produce the same cache key."""
        key1 = _make_cache_key((1, "hello"), {"flag": True})
        key2 = _make_cache_key((1, "hello"), {"flag": True})
        assert key1 == key2

    def test_different_args_produce_different_keys(self):
        """Different arguments should produce different cache keys."""
        key1 = _make_cache_key((1,), {})
        key2 = _make_cache_key((2,), {})
        assert key1 != key2

    def test_kwargs_order_independent(self):
        """Keyword argument ordering should not affect the cache key."""
        key1 = _make_cache_key((), {"a": 1, "b": 2})
        key2 = _make_cache_key((), {"b": 2, "a": 1})
        assert key1 == key2

    def test_key_args_filter(self):
        """When key_args is provided, only those args should matter."""
        param_names = ("user_id", "action", "metadata")

        key1 = _make_cache_key(
            (42, "read", {"trace": "a"}),
            {},
            key_args=("user_id", "action"),
            param_names=param_names,
        )
        key2 = _make_cache_key(
            (42, "read", {"trace": "z"}),
            {},
            key_args=("user_id", "action"),
            param_names=param_names,
        )
        # metadata differs but is not in key_args, so keys should match
        assert key1 == key2

    def test_key_with_unhashable_dict_arg(self):
        """_make_cache_key should handle dict arguments without raising."""
        key = _make_cache_key(({"nested": [1, 2]},), {})
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex digest length

    def test_key_is_sha256_hex(self):
        """Cache keys should be 64-character hex strings (SHA-256)."""
        key = _make_cache_key(("abc",), {"x": 1})
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


# ---------------------------------------------------------------------------
# _cleanup_expired tests
# ---------------------------------------------------------------------------


class TestCleanupExpired:
    """Tests for the _TTLCache._cleanup_expired method."""

    def test_expired_entries_removed_on_set(self):
        """Inserting a new entry should trigger cleanup of expired ones."""
        cache = _TTLCache(maxsize=10, ttl_seconds=0.05)

        cache.set("old1", "value1")
        cache.set("old2", "value2")

        # Wait for entries to expire
        time.sleep(0.1)

        # This set should trigger _cleanup_expired
        cache.set("new", "fresh")

        info = cache.cache_info()
        # Only the new entry should remain
        assert info.size == 1

        found, val = cache.get("new")
        assert found is True
        assert val == "fresh"

        found_old, _ = cache.get("old1")
        assert found_old is False

    def test_get_removes_single_expired_entry(self):
        """Getting an expired entry should remove it and count as a miss."""
        cache = _TTLCache(maxsize=10, ttl_seconds=0.05)
        cache.set("key", "value")

        time.sleep(0.1)

        found, val = cache.get("key")
        assert found is False
        assert val is None

        info = cache.cache_info()
        assert info.size == 0
        assert info.misses >= 1
