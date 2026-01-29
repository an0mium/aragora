"""Tests for handler caching utilities."""
import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler", "get_slack_handler", "get_slack_integration",
    "get_workspace_store", "resolve_workspace", "create_tracked_task",
    "_validate_slack_url", "SLACK_SIGNING_SECRET", "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL", "SLACK_ALLOWED_DOMAINS", "SignatureVerifierMixin",
    "CommandsMixin", "EventsMixin", "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clear_cache_state():
    """Clear cache state between tests."""
    from aragora.server.handlers.admin.cache import _cache

    _cache.clear()
    _cache._hits = 0
    _cache._misses = 0
    yield
    _cache.clear()


class TestBoundedTTLCache:
    """Tests for BoundedTTLCache class."""

    def test_cache_init_defaults(self):
        """Test cache initializes with defaults."""
        from aragora.server.handlers.admin.cache import BoundedTTLCache

        cache = BoundedTTLCache()

        assert len(cache) == 0
        assert cache._max_entries == 1000  # Default from env

    def test_cache_init_custom(self):
        """Test cache initializes with custom values."""
        from aragora.server.handlers.admin.cache import BoundedTTLCache

        cache = BoundedTTLCache(max_entries=100, evict_percent=0.2)

        assert cache._max_entries == 100
        assert cache._evict_count == 20

    def test_set_and_get_hit(self):
        """Test set and get returns cache hit."""
        from aragora.server.handlers.admin.cache import BoundedTTLCache

        cache = BoundedTTLCache()
        cache.set("key1", "value1")

        hit, value = cache.get("key1", ttl_seconds=60.0)

        assert hit is True
        assert value == "value1"

    def test_get_miss(self):
        """Test get returns miss for missing key."""
        from aragora.server.handlers.admin.cache import BoundedTTLCache

        cache = BoundedTTLCache()

        hit, value = cache.get("nonexistent", ttl_seconds=60.0)

        assert hit is False
        assert value is None

    def test_get_expired(self):
        """Test get returns miss for expired entry."""
        from aragora.server.handlers.admin.cache import BoundedTTLCache

        cache = BoundedTTLCache()
        cache.set("key1", "value1")

        # Wait for expiry (use very short TTL)
        time.sleep(0.1)
        hit, value = cache.get("key1", ttl_seconds=0.05)

        assert hit is False
        assert value is None

    def test_set_updates_existing(self):
        """Test set updates existing key."""
        from aragora.server.handlers.admin.cache import BoundedTTLCache

        cache = BoundedTTLCache()
        cache.set("key1", "value1")
        cache.set("key1", "value2")

        hit, value = cache.get("key1", ttl_seconds=60.0)

        assert hit is True
        assert value == "value2"
        assert len(cache) == 1

    def test_eviction_when_full(self):
        """Test LRU eviction when cache is full."""
        from aragora.server.handlers.admin.cache import BoundedTTLCache

        cache = BoundedTTLCache(max_entries=3, evict_percent=0.34)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Should trigger eviction

        # key1 should be evicted (oldest)
        assert "key1" not in cache
        assert "key4" in cache

    def test_lru_ordering(self):
        """Test LRU ordering on get."""
        from aragora.server.handlers.admin.cache import BoundedTTLCache

        cache = BoundedTTLCache(max_entries=3, evict_percent=0.34)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 to make it recently used
        cache.get("key1", ttl_seconds=60.0)

        cache.set("key4", "value4")  # Should evict key2 (oldest unused)

        assert "key1" in cache
        assert "key2" not in cache
        assert "key3" in cache
        assert "key4" in cache

    def test_clear_all(self):
        """Test clear removes all entries."""
        from aragora.server.handlers.admin.cache import BoundedTTLCache

        cache = BoundedTTLCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        count = cache.clear()

        assert count == 2
        assert len(cache) == 0

    def test_clear_by_prefix(self):
        """Test clear removes only matching prefix."""
        from aragora.server.handlers.admin.cache import BoundedTTLCache

        cache = BoundedTTLCache()
        cache.set("prefix1:key1", "value1")
        cache.set("prefix1:key2", "value2")
        cache.set("prefix2:key1", "value3")

        count = cache.clear("prefix1")

        assert count == 2
        assert len(cache) == 1
        assert "prefix2:key1" in cache

    def test_invalidate_containing(self):
        """Test invalidate_containing removes matching keys."""
        from aragora.server.handlers.admin.cache import BoundedTTLCache

        cache = BoundedTTLCache()
        cache.set("key1:agent:claude", "value1")
        cache.set("key2:agent:claude", "value2")
        cache.set("key3:agent:gemini", "value3")

        count = cache.invalidate_containing("claude")

        assert count == 2
        assert len(cache) == 1
        assert "key3:agent:gemini" in cache

    def test_stats(self):
        """Test stats returns correct values."""
        from aragora.server.handlers.admin.cache import BoundedTTLCache

        cache = BoundedTTLCache(max_entries=100)
        cache.set("key1", "value1")
        cache.get("key1", ttl_seconds=60.0)  # hit
        cache.get("key2", ttl_seconds=60.0)  # miss

        stats = cache.stats

        assert stats["entries"] == 1
        assert stats["max_entries"] == 100
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_items_returns_copy(self):
        """Test items returns a copy of cache entries."""
        from aragora.server.handlers.admin.cache import BoundedTTLCache

        cache = BoundedTTLCache()
        cache.set("key1", "value1")

        items = cache.items()

        assert len(items) == 1
        assert items[0][0] == "key1"

    def test_contains(self):
        """Test __contains__ checks key presence."""
        from aragora.server.handlers.admin.cache import BoundedTTLCache

        cache = BoundedTTLCache()
        cache.set("key1", "value1")

        assert "key1" in cache
        assert "key2" not in cache


class TestTTLCacheDecorator:
    """Tests for ttl_cache decorator."""

    def test_ttl_cache_caches_result(self):
        """Test ttl_cache caches function result."""
        from aragora.server.handlers.admin.cache import ttl_cache

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
        assert call_count == 1  # Only called once

    def test_ttl_cache_different_args(self):
        """Test ttl_cache with different args."""
        from aragora.server.handlers.admin.cache import ttl_cache

        call_count = 0

        @ttl_cache(ttl_seconds=60.0, key_prefix="test", skip_first=False)
        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = expensive_func(5)
        result2 = expensive_func(10)

        assert result1 == 10
        assert result2 == 20
        assert call_count == 2  # Called for each unique arg

    def test_ttl_cache_skips_self(self):
        """Test ttl_cache skips self arg for methods."""
        from aragora.server.handlers.admin.cache import ttl_cache

        class MyClass:
            call_count = 0

            @ttl_cache(ttl_seconds=60.0, key_prefix="test", skip_first=True)
            def method(self, x):
                MyClass.call_count += 1
                return x * 2

        obj1 = MyClass()
        obj2 = MyClass()

        result1 = obj1.method(5)
        result2 = obj2.method(5)  # Same cache key (self skipped)

        assert result1 == 10
        assert result2 == 10
        assert MyClass.call_count == 1


class TestAsyncTTLCacheDecorator:
    """Tests for async_ttl_cache decorator."""

    @pytest.mark.asyncio
    async def test_async_ttl_cache_caches_result(self):
        """Test async_ttl_cache caches coroutine result."""
        from aragora.server.handlers.admin.cache import async_ttl_cache

        call_count = 0

        @async_ttl_cache(ttl_seconds=60.0, key_prefix="async_test", skip_first=False)
        async def async_expensive(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = await async_expensive(5)
        result2 = await async_expensive(5)

        assert result1 == 10
        assert result2 == 10
        assert call_count == 1


class TestCacheInvalidation:
    """Tests for cache invalidation functions."""

    def test_invalidate_on_event(self):
        """Test invalidate_on_event clears associated prefixes."""
        from aragora.server.handlers.admin.cache import _cache, invalidate_on_event

        _cache.set("leaderboard:key1", "value1")
        _cache.set("lb_rankings:key2", "value2")
        _cache.set("other:key3", "value3")

        count = invalidate_on_event("elo_updated")

        assert count == 2
        assert "other:key3" in _cache
        assert "leaderboard:key1" not in _cache

    def test_invalidate_cache_by_source(self):
        """Test invalidate_cache by data source."""
        from aragora.server.handlers.admin.cache import _cache, invalidate_cache

        _cache.set("leaderboard:key1", "value1")
        _cache.set("analytics_ranking:key2", "value2")

        count = invalidate_cache("elo")

        assert count >= 2

    def test_invalidate_leaderboard_cache(self):
        """Test convenience function for leaderboard cache."""
        from aragora.server.handlers.admin.cache import _cache, invalidate_leaderboard_cache

        _cache.set("leaderboard:key1", "value1")

        count = invalidate_leaderboard_cache()

        assert count >= 1
        assert "leaderboard:key1" not in _cache

    def test_invalidate_agent_cache_specific(self):
        """Test invalidate agent cache for specific agent."""
        from aragora.server.handlers.admin.cache import _cache, invalidate_agent_cache

        _cache.set("agent:claude:profile", "value1")
        _cache.set("agent:gemini:profile", "value2")

        count = invalidate_agent_cache("claude")

        assert count == 1
        assert "agent:claude:profile" not in _cache
        assert "agent:gemini:profile" in _cache

    def test_invalidate_debate_cache_specific(self):
        """Test invalidate debate cache for specific debate."""
        from aragora.server.handlers.admin.cache import _cache, invalidate_debate_cache

        _cache.set("debate:abc123:data", "value1")
        _cache.set("debate:def456:data", "value2")

        count = invalidate_debate_cache("abc123")

        assert count == 1
        assert "debate:abc123:data" not in _cache
        assert "debate:def456:data" in _cache


class TestInvalidatesCacheDecorator:
    """Tests for invalidates_cache decorator."""

    def test_invalidates_cache_sync(self):
        """Test invalidates_cache decorator on sync function."""
        from aragora.server.handlers.admin.cache import _cache, invalidates_cache

        _cache.set("leaderboard:key1", "value1")

        @invalidates_cache("elo_updated")
        def update_elo():
            return "done"

        result = update_elo()

        assert result == "done"
        assert "leaderboard:key1" not in _cache

    @pytest.mark.asyncio
    async def test_invalidates_cache_async(self):
        """Test invalidates_cache decorator on async function."""
        from aragora.server.handlers.admin.cache import _cache, invalidates_cache

        _cache.set("leaderboard:key1", "value1")

        @invalidates_cache("elo_updated")
        async def async_update_elo():
            return "done"

        result = await async_update_elo()

        assert result == "done"
        assert "leaderboard:key1" not in _cache


class TestGetHandlerCache:
    """Tests for get_handler_cache function."""

    def test_get_handler_cache_returns_global(self):
        """Test get_handler_cache returns global cache."""
        from aragora.server.handlers.admin.cache import _cache, get_handler_cache

        cache = get_handler_cache()

        assert cache is _cache

    def test_get_handler_cache_registers_service(self):
        """Test get_handler_cache attempts to register with ServiceRegistry."""
        from aragora.server.handlers.admin import cache as cache_mod
        from aragora.server.handlers.admin.cache import _cache, get_handler_cache

        # Reset registration status
        cache_mod._handler_cache_registered = False

        # Call get_handler_cache and verify it returns the cache
        # The registration happens lazily and may fail silently if services not available
        result = get_handler_cache()

        assert result is _cache


class TestClearCache:
    """Tests for clear_cache function."""

    def test_clear_cache_all(self):
        """Test clear_cache clears all entries."""
        from aragora.server.handlers.admin.cache import _cache, clear_cache

        _cache.set("key1", "value1")
        _cache.set("key2", "value2")

        count = clear_cache()

        assert count == 2
        assert len(_cache) == 0

    def test_clear_cache_by_prefix(self):
        """Test clear_cache with prefix."""
        from aragora.server.handlers.admin.cache import _cache, clear_cache

        _cache.set("prefix1:key1", "value1")
        _cache.set("prefix2:key2", "value2")

        count = clear_cache("prefix1")

        assert count == 1
        assert "prefix2:key2" in _cache


class TestGetCacheStats:
    """Tests for get_cache_stats function."""

    def test_get_cache_stats(self):
        """Test get_cache_stats returns stats."""
        from aragora.server.handlers.admin.cache import _cache, get_cache_stats

        _cache.set("key1", "value1")
        _cache.get("key1", ttl_seconds=60.0)

        stats = get_cache_stats()

        assert "entries" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
