"""
Tests for Decision Router Cache Invalidation.

Tests cache invalidation strategies including workspace-scoped,
policy version, agent version, and tag-based invalidation.
"""

import pytest
import time
from aragora.server.middleware.decision_routing import (
    ResponseCache,
    CacheEntry,
    get_decision_middleware,
    reset_decision_middleware,
    invalidate_cache_for_workspace,
    invalidate_cache_for_policy_change,
    invalidate_cache_for_agent_upgrade,
    get_cache_stats,
)


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_matches_tag(self):
        """Should match tags correctly."""
        entry = CacheEntry(
            result="test",
            timestamp=time.time(),
            tags=["governance", "policy-v1"],
        )
        assert entry.matches_tag("governance") is True
        assert entry.matches_tag("policy-v1") is True
        assert entry.matches_tag("unknown") is False

    def test_matches_workspace(self):
        """Should match workspace correctly."""
        entry = CacheEntry(
            result="test",
            timestamp=time.time(),
            workspace_id="ws-123",
        )
        assert entry.matches_workspace("ws-123") is True
        assert entry.matches_workspace("ws-456") is False


class TestResponseCacheInvalidation:
    """Tests for ResponseCache invalidation methods."""

    @pytest.fixture
    def cache(self):
        """Create a fresh cache."""
        return ResponseCache(ttl_seconds=3600.0, max_size=100)

    @pytest.mark.asyncio
    async def test_invalidate_by_workspace(self, cache):
        """Should invalidate all entries for a workspace."""
        # Add entries for different workspaces
        await cache.set("q1", "a1", context={"workspace_id": "ws-A"})
        await cache.set("q2", "a2", context={"workspace_id": "ws-A"})
        await cache.set("q3", "a3", context={"workspace_id": "ws-B"})

        # Verify all are cached
        assert await cache.get("q1", context={"workspace_id": "ws-A"}) == "a1"
        assert await cache.get("q2", context={"workspace_id": "ws-A"}) == "a2"
        assert await cache.get("q3", context={"workspace_id": "ws-B"}) == "a3"

        # Invalidate workspace A
        count = await cache.invalidate_by_workspace("ws-A")
        assert count == 2

        # Workspace A entries should be gone
        assert await cache.get("q1", context={"workspace_id": "ws-A"}) is None
        assert await cache.get("q2", context={"workspace_id": "ws-A"}) is None

        # Workspace B entry should remain
        assert await cache.get("q3", context={"workspace_id": "ws-B"}) == "a3"

    @pytest.mark.asyncio
    async def test_invalidate_by_tag(self, cache):
        """Should invalidate entries by tag."""
        await cache.set("q1", "a1", tags=["governance"])
        await cache.set("q2", "a2", tags=["governance", "policy"])
        await cache.set("q3", "a3", tags=["other"])

        # Invalidate governance tag
        count = await cache.invalidate_by_tag("governance")
        assert count == 2

        # Tagged entries should be gone
        assert await cache.get("q1") is None
        assert await cache.get("q2") is None

        # Untagged entry should remain
        assert await cache.get("q3") == "a3"

    @pytest.mark.asyncio
    async def test_invalidate_by_agent_version(self, cache):
        """Should invalidate entries using specific agent version."""
        await cache.set(
            "q1", "a1",
            agent_versions={"claude": "1.0", "gpt": "4.0"}
        )
        await cache.set(
            "q2", "a2",
            agent_versions={"claude": "2.0", "gpt": "4.0"}
        )
        await cache.set(
            "q3", "a3",
            agent_versions={"gpt": "4.0"}  # No claude
        )

        # Invalidate claude 1.0
        count = await cache.invalidate_by_agent_version("claude", "1.0")
        assert count == 1

        assert await cache.get("q1") is None
        assert await cache.get("q2") == "a2"
        assert await cache.get("q3") == "a3"

    @pytest.mark.asyncio
    async def test_policy_version_invalidation(self, cache):
        """Should invalidate entries when policy version changes."""
        # Set initial policy version
        cache.set_policy_version("v1")

        # Cache with v1
        await cache.set("query", "answer-v1")

        # Entry should be available
        assert await cache.get("query") == "answer-v1"

        # Update policy version
        cache.set_policy_version("v2")

        # Entry should be invalidated (lazy invalidation on get)
        assert await cache.get("query") is None

    @pytest.mark.asyncio
    async def test_cache_stats(self, cache):
        """Should track cache statistics."""
        # Generate some activity
        await cache.set("q1", "a1", context={"workspace_id": "ws-1"})
        await cache.set("q2", "a2", context={"workspace_id": "ws-2"})

        # Some hits
        await cache.get("q1", context={"workspace_id": "ws-1"})
        await cache.get("q1", context={"workspace_id": "ws-1"})

        # Some misses
        await cache.get("missing")
        await cache.get("also-missing")

        stats = await cache.get_stats()

        assert stats["size"] == 2
        assert stats["hits"] == 2
        assert stats["misses"] == 2
        assert stats["hit_rate"] == 0.5
        assert "entries_by_workspace" in stats
        assert stats["entries_by_workspace"]["ws-1"] == 1
        assert stats["entries_by_workspace"]["ws-2"] == 1

    @pytest.mark.asyncio
    async def test_clear_updates_invalidation_count(self, cache):
        """Clear should update invalidation count."""
        await cache.set("q1", "a1")
        await cache.set("q2", "a2")
        await cache.set("q3", "a3")

        initial_stats = await cache.get_stats()
        initial_invalidations = initial_stats["invalidations"]

        cleared = await cache.clear()
        assert cleared == 3

        new_stats = await cache.get_stats()
        assert new_stats["invalidations"] == initial_invalidations + 3


class TestGlobalCacheInvalidation:
    """Tests for global cache invalidation functions."""

    def setup_method(self):
        """Reset middleware before each test."""
        reset_decision_middleware()

    @pytest.mark.asyncio
    async def test_invalidate_cache_for_workspace(self):
        """Should invalidate workspace via global function."""
        middleware = await get_decision_middleware()

        # Populate cache
        await middleware._cache.set(
            "workspace query",
            "workspace answer",
            context={"workspace_id": "test-ws", "channel": "api"},
        )

        # Verify cached
        assert await middleware._cache.get(
            "workspace query",
            context={"workspace_id": "test-ws", "channel": "api"},
        ) == "workspace answer"

        # Invalidate via global function
        count = await invalidate_cache_for_workspace("test-ws")
        assert count == 1

        # Verify invalidated
        assert await middleware._cache.get(
            "workspace query",
            context={"workspace_id": "test-ws", "channel": "api"},
        ) is None

    @pytest.mark.asyncio
    async def test_invalidate_cache_for_policy_change(self):
        """Should mark cache as stale on policy change."""
        middleware = await get_decision_middleware()

        # Set initial policy
        middleware._cache.set_policy_version("policy-v1")

        # Populate cache
        await middleware._cache.set("policy query", "old answer")

        # Change policy
        await invalidate_cache_for_policy_change("policy-v2")

        # Entry should be invalidated on next access
        assert await middleware._cache.get("policy query") is None

    @pytest.mark.asyncio
    async def test_get_cache_stats(self):
        """Should return cache stats via global function."""
        middleware = await get_decision_middleware()

        # Generate activity
        await middleware._cache.set("stats-q1", "stats-a1")
        await middleware._cache.get("stats-q1")
        await middleware._cache.get("missing")

        stats = await get_cache_stats()

        assert "size" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats


class TestCacheEvictionWithMetadata:
    """Tests for cache eviction with metadata preservation."""

    @pytest.mark.asyncio
    async def test_eviction_removes_oldest_entry(self):
        """Should evict oldest entry when at capacity."""
        cache = ResponseCache(ttl_seconds=3600.0, max_size=3)

        # Fill cache
        await cache.set("q1", "a1")
        await cache.set("q2", "a2")
        await cache.set("q3", "a3")

        # Verify all cached
        assert await cache.get("q1") == "a1"
        assert await cache.get("q2") == "a2"
        assert await cache.get("q3") == "a3"

        # Add one more - should evict oldest (q1)
        await cache.set("q4", "a4")

        assert await cache.get("q1") is None  # Evicted
        assert await cache.get("q2") == "a2"
        assert await cache.get("q3") == "a3"
        assert await cache.get("q4") == "a4"

        stats = await cache.get_stats()
        assert stats["evictions"] >= 1

    @pytest.mark.asyncio
    async def test_metadata_preserved_through_set(self):
        """Should preserve metadata when setting cache entries."""
        cache = ResponseCache()

        await cache.set(
            "query",
            "answer",
            context={"workspace_id": "ws-test", "channel": "slack"},
            tags=["debate", "consensus"],
            agent_versions={"claude": "3.0", "gpt": "4.0"},
        )

        # Access the entry directly to verify metadata
        entry = cache._cache[cache._compute_hash(
            "query",
            {"workspace_id": "ws-test", "channel": "slack"}
        )]

        assert entry.workspace_id == "ws-test"
        assert "debate" in entry.tags
        assert "consensus" in entry.tags
        assert entry.agent_versions["claude"] == "3.0"
        assert entry.agent_versions["gpt"] == "4.0"


class TestCacheTTLWithPolicyVersion:
    """Tests for TTL and policy version interaction."""

    @pytest.mark.asyncio
    async def test_ttl_expiry_still_works_with_policy(self):
        """TTL expiry should work even with matching policy version."""
        cache = ResponseCache(ttl_seconds=0.1, max_size=100)  # Very short TTL
        cache.set_policy_version("v1")

        await cache.set("query", "answer")

        # Should be cached initially
        assert await cache.get("query") == "answer"

        # Wait for TTL
        import asyncio
        await asyncio.sleep(0.15)

        # Should be expired despite matching policy
        assert await cache.get("query") is None

    @pytest.mark.asyncio
    async def test_policy_invalidation_immediate(self):
        """Policy version mismatch should invalidate immediately on get."""
        cache = ResponseCache(ttl_seconds=3600.0, max_size=100)
        cache.set_policy_version("v1")

        await cache.set("query", "answer")

        # Verify cached
        assert await cache.get("query") == "answer"

        # Change policy - no waiting needed
        cache.set_policy_version("v2")

        # Immediately invalidated
        assert await cache.get("query") is None
