"""
Tests for Redis cache LRU entry bounding.

Covers max_entries enforcement, LRU eviction, entry tracking,
touch-on-read behavior, and memory stats reporting.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.knowledge.mound.redis_cache import RedisCache


# ===========================================================================
# Helpers
# ===========================================================================


def _make_cache(max_entries: int = 5, prefix: str = "test:km") -> RedisCache:
    """Create a RedisCache with a mock client for testing."""
    cache = RedisCache(
        url="redis://localhost:6379",
        max_entries=max_entries,
        prefix=prefix,
    )
    cache._client = AsyncMock()
    cache._connected = True
    return cache


def _make_node_mock():
    """Create a mock KnowledgeItem."""
    node = MagicMock()
    node.to_dict.return_value = {"id": "n-1", "content": "test"}
    return node


# ===========================================================================
# max_entries configuration
# ===========================================================================


class TestMaxEntriesConfig:
    """Tests for max_entries parameter handling."""

    def test_default_max_entries(self):
        cache = RedisCache(url="redis://localhost")
        assert cache._max_entries == 10_000

    def test_custom_max_entries(self):
        cache = RedisCache(url="redis://localhost", max_entries=500)
        assert cache._max_entries == 500

    def test_tracker_key_uses_prefix(self):
        cache = RedisCache(url="redis://localhost", prefix="myapp:cache")
        assert cache._tracker_key == "myapp:cache:_entry_tracker"


# ===========================================================================
# Entry tracking on set operations
# ===========================================================================


class TestEntryTracking:
    """Tests for LRU entry tracking on cache writes."""

    @pytest.mark.asyncio
    async def test_set_node_tracks_entry(self):
        cache = _make_cache(max_entries=100)
        cache._client.zcard = AsyncMock(return_value=0)
        node = _make_node_mock()

        await cache.set_node("n-1", node)

        # Should track the entry in the ZSET
        cache._client.zadd.assert_called()
        call_args = cache._client.zadd.call_args
        assert cache._tracker_key in str(call_args)

    @pytest.mark.asyncio
    async def test_set_query_tracks_entry(self):
        cache = _make_cache(max_entries=100)
        cache._client.zcard = AsyncMock(return_value=0)
        result = MagicMock()
        result.to_dict.return_value = {"items": [], "total_count": 0, "query": "test"}

        await cache.set_query("q-key", result)

        cache._client.zadd.assert_called()

    @pytest.mark.asyncio
    async def test_set_culture_tracks_entry(self):
        cache = _make_cache(max_entries=100)
        cache._client.zcard = AsyncMock(return_value=0)
        profile = MagicMock()
        profile.workspace_id = "ws-1"
        profile.patterns = {}
        profile.generated_at = MagicMock()
        profile.generated_at.isoformat.return_value = "2025-01-01T00:00:00"
        profile.total_observations = 0
        profile.dominant_traits = {}

        await cache.set_culture("ws-1", profile)

        cache._client.zadd.assert_called()


# ===========================================================================
# LRU eviction
# ===========================================================================


class TestLRUEviction:
    """Tests for LRU eviction when max_entries is reached."""

    @pytest.mark.asyncio
    async def test_no_eviction_under_limit(self):
        cache = _make_cache(max_entries=10)
        cache._client.zcard = AsyncMock(return_value=5)

        evicted = await cache._enforce_max_entries()

        assert evicted == 0
        cache._client.zrange.assert_not_called()

    @pytest.mark.asyncio
    async def test_eviction_at_limit(self):
        cache = _make_cache(max_entries=5)
        cache._client.zcard = AsyncMock(return_value=5)
        cache._client.zrange = AsyncMock(return_value=["key:old1", "key:old2"])

        evicted = await cache._enforce_max_entries()

        assert evicted == 2
        # Should delete the victim keys
        cache._client.delete.assert_called_with("key:old1", "key:old2")
        # Should remove from tracker
        cache._client.zremrangebyrank.assert_called_once()

    @pytest.mark.asyncio
    async def test_eviction_over_limit(self):
        cache = _make_cache(max_entries=3)
        cache._client.zcard = AsyncMock(return_value=5)
        # overage = 5 - 3 + 1 = 3 victims
        cache._client.zrange = AsyncMock(return_value=["k1", "k2", "k3"])

        evicted = await cache._enforce_max_entries()

        assert evicted == 3

    @pytest.mark.asyncio
    async def test_eviction_no_victims_returns_zero(self):
        cache = _make_cache(max_entries=3)
        cache._client.zcard = AsyncMock(return_value=3)
        cache._client.zrange = AsyncMock(return_value=[])

        evicted = await cache._enforce_max_entries()

        assert evicted == 0

    @pytest.mark.asyncio
    async def test_set_node_triggers_eviction(self):
        cache = _make_cache(max_entries=2)
        cache._client.zcard = AsyncMock(return_value=2)
        cache._client.zrange = AsyncMock(return_value=["test:km:node:old"])
        node = _make_node_mock()

        await cache.set_node("n-new", node)

        # Eviction should have been called
        cache._client.zrange.assert_called_once()
        cache._client.delete.assert_called()


# ===========================================================================
# Touch on read (LRU refresh)
# ===========================================================================


class TestTouchOnRead:
    """Tests for LRU access-time refresh on cache reads."""

    @pytest.mark.asyncio
    async def test_get_node_touches_entry(self):
        cache = _make_cache()
        cache._client.get = AsyncMock(return_value='{"id": "n-1"}')
        cache._client.zscore = AsyncMock(return_value=1000.0)

        with patch("aragora.knowledge.mound.redis_cache.json.loads", return_value={"id": "n-1"}):
            with patch("aragora.knowledge.mound.types.KnowledgeItem") as mock_ki:
                mock_ki.from_dict.return_value = MagicMock()
                # Need to patch the import inside the function
                with patch.dict("sys.modules", {"aragora.knowledge.mound.types": mock_ki}):
                    try:
                        await cache.get_node("n-1")
                    except Exception:
                        pass  # Import may fail in test env

        # zscore should have been called to check if entry exists
        cache._client.zscore.assert_called()

    @pytest.mark.asyncio
    async def test_touch_updates_timestamp(self):
        cache = _make_cache()
        cache._client.zscore = AsyncMock(return_value=1000.0)

        await cache._touch_entry("some:key")

        # Should have called zadd with updated timestamp
        cache._client.zadd.assert_called_once()
        call_args = cache._client.zadd.call_args[0]
        assert call_args[0] == cache._tracker_key

    @pytest.mark.asyncio
    async def test_touch_nonexistent_entry_does_nothing(self):
        cache = _make_cache()
        cache._client.zscore = AsyncMock(return_value=None)

        await cache._touch_entry("nonexistent:key")

        cache._client.zadd.assert_not_called()


# ===========================================================================
# Invalidation removes from tracker
# ===========================================================================


class TestInvalidationTracking:
    """Tests that invalidation also removes entries from the LRU tracker."""

    @pytest.mark.asyncio
    async def test_invalidate_node_untracks(self):
        cache = _make_cache()

        await cache.invalidate_node("n-1")

        cache._client.zrem.assert_called_with(cache._tracker_key, "test:km:node:n-1")

    @pytest.mark.asyncio
    async def test_invalidate_nodes_untracks_all(self):
        cache = _make_cache()

        await cache.invalidate_nodes(["n-1", "n-2", "n-3"])

        # Should untrack all keys
        cache._client.zrem.assert_called_with(
            cache._tracker_key,
            "test:km:node:n-1",
            "test:km:node:n-2",
            "test:km:node:n-3",
        )

    @pytest.mark.asyncio
    async def test_invalidate_culture_untracks(self):
        cache = _make_cache()

        await cache.invalidate_culture("ws-1")

        cache._client.zrem.assert_called_with(cache._tracker_key, "test:km:ws-1:culture")

    @pytest.mark.asyncio
    async def test_clear_all_untracks(self):
        cache = _make_cache()
        cache._client.scan = AsyncMock(return_value=(0, ["test:km:node:1", "test:km:node:2"]))

        await cache.clear_all()

        # Should untrack the deleted keys
        cache._client.zrem.assert_called_with(
            cache._tracker_key,
            "test:km:node:1",
            "test:km:node:2",
        )


# ===========================================================================
# Memory stats
# ===========================================================================


class TestMemoryStats:
    """Tests for get_memory_stats and get_entry_count."""

    @pytest.mark.asyncio
    async def test_get_entry_count(self):
        cache = _make_cache(max_entries=100)
        cache._client.zcard = AsyncMock(return_value=42)

        count = await cache.get_entry_count()

        assert count == 42

    @pytest.mark.asyncio
    async def test_get_memory_stats(self):
        cache = _make_cache(max_entries=1000)
        cache._client.zcard = AsyncMock(return_value=250)
        cache._client.info = AsyncMock(
            return_value={
                "used_memory_human": "10.5M",
                "used_memory": 11010048,
            }
        )

        stats = await cache.get_memory_stats()

        assert stats["entry_count"] == 250
        assert stats["max_entries"] == 1000
        assert stats["utilization"] == 0.25
        assert stats["used_memory"] == "10.5M"
        assert stats["used_memory_bytes"] == 11010048

    @pytest.mark.asyncio
    async def test_get_memory_stats_zero_max_entries(self):
        cache = _make_cache(max_entries=0)
        cache._client.zcard = AsyncMock(return_value=0)
        cache._client.info = AsyncMock(return_value={})

        stats = await cache.get_memory_stats()

        assert stats["utilization"] == 0

    @pytest.mark.asyncio
    async def test_get_entry_count_requires_connection(self):
        cache = RedisCache(url="redis://localhost", max_entries=100)

        with pytest.raises(RuntimeError, match="Redis not connected"):
            await cache.get_entry_count()
