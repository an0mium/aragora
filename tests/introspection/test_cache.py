"""
Tests for aragora.introspection.cache module.

Tests cover:
- IntrospectionCache initialization
- Cache warming with agents
- Get/invalidate operations
- Properties (is_warm, agent_count)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.introspection.cache import IntrospectionCache
from aragora.introspection.types import IntrospectionSnapshot


class TestIntrospectionCacheInit:
    """Test IntrospectionCache initialization."""

    def test_init_creates_empty_cache(self):
        """Cache should start empty."""
        cache = IntrospectionCache()
        assert cache._cache == {}
        assert cache._loaded_at is None

    def test_init_is_not_warm(self):
        """Fresh cache should not be warm."""
        cache = IntrospectionCache()
        assert not cache.is_warm

    def test_init_agent_count_zero(self):
        """Fresh cache should have zero agents."""
        cache = IntrospectionCache()
        assert cache.agent_count == 0


class TestIntrospectionCacheWarm:
    """Test cache warming functionality."""

    def test_warm_loads_agents(self):
        """Warming cache should load all agents."""
        cache = IntrospectionCache()

        # Create mock agents
        agent1 = MagicMock()
        agent1.name = "claude"
        agent2 = MagicMock()
        agent2.name = "gpt-4"

        # Mock the introspection API at the source module
        with patch("aragora.introspection.api.get_agent_introspection") as mock_get:
            mock_get.side_effect = [
                IntrospectionSnapshot(agent_name="claude"),
                IntrospectionSnapshot(agent_name="gpt-4"),
            ]

            cache.warm(agents=[agent1, agent2])

        assert cache.is_warm
        assert cache.agent_count == 2
        assert mock_get.call_count == 2

    def test_warm_clears_existing_cache(self):
        """Warming should clear existing data."""
        cache = IntrospectionCache()
        cache._cache["old_agent"] = IntrospectionSnapshot(agent_name="old_agent")

        agent = MagicMock()
        agent.name = "new_agent"

        with patch("aragora.introspection.api.get_agent_introspection") as mock_get:
            mock_get.return_value = IntrospectionSnapshot(agent_name="new_agent")
            cache.warm(agents=[agent])

        assert "old_agent" not in cache._cache
        assert "new_agent" in cache._cache

    def test_warm_handles_agents_without_name(self):
        """Warming should handle agents without name attribute."""
        cache = IntrospectionCache()

        # Use a simple class without 'name' attribute
        class AgentLike:
            def __str__(self):
                return "agent_str"

        agent = AgentLike()

        with patch("aragora.introspection.api.get_agent_introspection") as mock_get:
            mock_get.return_value = IntrospectionSnapshot(agent_name="agent_str")
            cache.warm(agents=[agent])

        assert cache.agent_count == 1


class TestIntrospectionCacheGet:
    """Test cache get operations."""

    def test_get_returns_cached_snapshot(self):
        """Get should return cached snapshot."""
        cache = IntrospectionCache()
        snapshot = IntrospectionSnapshot(agent_name="claude", reputation_score=0.85)
        cache._cache["claude"] = snapshot

        result = cache.get("claude")
        assert result == snapshot
        assert result.reputation_score == 0.85

    def test_get_returns_none_for_missing(self):
        """Get should return None for missing agents."""
        cache = IntrospectionCache()
        result = cache.get("unknown_agent")
        assert result is None

    def test_get_all_returns_copy(self):
        """get_all should return a copy of the cache."""
        cache = IntrospectionCache()
        snapshot = IntrospectionSnapshot(agent_name="claude")
        cache._cache["claude"] = snapshot

        all_snapshots = cache.get_all()
        assert all_snapshots == {"claude": snapshot}

        # Verify it's a copy
        all_snapshots["new_agent"] = IntrospectionSnapshot(agent_name="new")
        assert "new_agent" not in cache._cache


class TestIntrospectionCacheInvalidate:
    """Test cache invalidation."""

    def test_invalidate_clears_cache(self):
        """Invalidate should clear all cached data."""
        cache = IntrospectionCache()
        cache._cache["claude"] = IntrospectionSnapshot(agent_name="claude")
        cache._loaded_at = MagicMock()

        cache.invalidate()

        assert cache._cache == {}
        assert cache._loaded_at is None
        assert not cache.is_warm

    def test_invalidate_allows_rewarming(self):
        """After invalidation, cache can be warmed again."""
        cache = IntrospectionCache()
        cache._cache["old"] = IntrospectionSnapshot(agent_name="old")

        cache.invalidate()

        agent = MagicMock()
        agent.name = "new"

        with patch("aragora.introspection.api.get_agent_introspection") as mock_get:
            mock_get.return_value = IntrospectionSnapshot(agent_name="new")
            cache.warm(agents=[agent])

        assert cache.is_warm
        assert cache.get("new") is not None


class TestIntrospectionCacheProperties:
    """Test cache properties."""

    def test_is_warm_true_when_loaded(self):
        """is_warm should be True when cache has data."""
        cache = IntrospectionCache()
        cache._loaded_at = MagicMock()
        cache._cache["agent"] = IntrospectionSnapshot(agent_name="agent")

        assert cache.is_warm

    def test_is_warm_false_when_empty(self):
        """is_warm should be False when cache is empty."""
        cache = IntrospectionCache()
        cache._loaded_at = MagicMock()  # Set loaded_at but no data

        assert not cache.is_warm

    def test_is_warm_false_when_not_loaded(self):
        """is_warm should be False when not loaded."""
        cache = IntrospectionCache()
        cache._cache["agent"] = IntrospectionSnapshot(agent_name="agent")  # Data but no timestamp

        assert not cache.is_warm

    def test_agent_count_reflects_cache_size(self):
        """agent_count should reflect number of cached agents."""
        cache = IntrospectionCache()
        assert cache.agent_count == 0

        cache._cache["a1"] = IntrospectionSnapshot(agent_name="a1")
        assert cache.agent_count == 1

        cache._cache["a2"] = IntrospectionSnapshot(agent_name="a2")
        assert cache.agent_count == 2
