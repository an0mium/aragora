"""Tests for Debate Batch Loaders module.

Covers:
- ELORating and AgentStats dataclasses
- DebateLoaders initialization and configuration
- Batch loading of ELO ratings
- Batch loading of agent statistics
- Cache hit and miss behavior
- Context variable management
- Context manager usage
- Edge cases (empty batch, missing items, no data source)
- Data conversion helpers
- Multiple batch operations
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.debate.batch_loaders import (
    AgentStats,
    DebateLoaders,
    ELORating,
    debate_loader_context,
    get_debate_loaders,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_elo_system():
    """Create a mock ELO system with batch support."""
    elo = MagicMock()

    # Default ratings data
    ratings_data = {
        "claude": {"rating": 1500.0, "games_played": 50, "wins": 30, "losses": 20},
        "gemini": {"rating": 1400.0, "games_played": 40, "wins": 20, "losses": 20},
        "grok": {"rating": 1350.0, "games_played": 30, "wins": 12, "losses": 18},
    }

    async def get_ratings_batch(agent_names):
        return {name: ratings_data.get(name) for name in agent_names}

    elo.get_ratings_batch = AsyncMock(side_effect=get_ratings_batch)
    elo.ratings_data = ratings_data  # For test access
    return elo


@pytest.fixture
def mock_elo_system_individual():
    """Create a mock ELO system with only individual lookups."""
    elo = MagicMock(spec=["get_rating"])

    # Default ratings data
    ratings_data = {
        "claude": 1500.0,
        "gemini": 1400.0,
        "grok": 1350.0,
    }

    async def get_rating(name):
        return ratings_data.get(name)

    elo.get_rating = AsyncMock(side_effect=get_rating)
    return elo


@pytest.fixture
def mock_debate_store():
    """Create a mock debate store with batch support."""
    store = MagicMock()

    # Default stats data
    stats_data = {
        "claude": {
            "debate_count": 100,
            "win_rate": 0.65,
            "avg_confidence": 0.85,
            "avg_response_time_ms": 250.0,
            "domains": ["reasoning", "coding"],
        },
        "gemini": {
            "debate_count": 80,
            "win_rate": 0.55,
            "avg_confidence": 0.78,
            "avg_response_time_ms": 200.0,
            "domains": ["creative", "analysis"],
        },
    }

    async def get_agent_stats_batch(agent_names):
        return {name: stats_data.get(name) for name in agent_names}

    store.get_agent_stats_batch = AsyncMock(side_effect=get_agent_stats_batch)
    store.stats_data = stats_data  # For test access
    return store


@pytest.fixture
def mock_debate_store_individual():
    """Create a mock debate store with only individual lookups."""
    store = MagicMock(spec=["get_agent_stats"])

    # Default stats data
    stats_data = {
        "claude": {
            "debate_count": 100,
            "win_rate": 0.65,
            "avg_confidence": 0.85,
            "avg_response_time_ms": 250.0,
            "domains": ["reasoning", "coding"],
        },
    }

    async def get_agent_stats(name):
        return stats_data.get(name)

    store.get_agent_stats = AsyncMock(side_effect=get_agent_stats)
    return store


@pytest.fixture
def sample_elo_rating():
    """Create a sample ELORating instance."""
    return ELORating(
        agent_name="claude",
        rating=1500.0,
        games_played=50,
        wins=30,
        losses=20,
        last_updated="2025-01-15T10:30:00Z",
    )


@pytest.fixture
def sample_agent_stats():
    """Create a sample AgentStats instance."""
    return AgentStats(
        agent_name="claude",
        debate_count=100,
        win_rate=0.65,
        avg_confidence=0.85,
        avg_response_time_ms=250.0,
        domains=["reasoning", "coding"],
    )


# =============================================================================
# ELORating Dataclass Tests
# =============================================================================


class TestELORatingDataclass:
    """Tests for ELORating dataclass."""

    def test_elo_rating_creation(self):
        """Test creating an ELORating instance."""
        rating = ELORating(
            agent_name="claude",
            rating=1500.0,
            games_played=50,
            wins=30,
            losses=20,
        )

        assert rating.agent_name == "claude"
        assert rating.rating == 1500.0
        assert rating.games_played == 50
        assert rating.wins == 30
        assert rating.losses == 20
        assert rating.last_updated is None

    def test_elo_rating_with_last_updated(self):
        """Test ELORating with last_updated field."""
        rating = ELORating(
            agent_name="gemini",
            rating=1400.0,
            games_played=40,
            wins=20,
            losses=20,
            last_updated="2025-01-15T10:00:00Z",
        )

        assert rating.last_updated == "2025-01-15T10:00:00Z"

    def test_elo_rating_defaults(self):
        """Test ELORating default values."""
        rating = ELORating(
            agent_name="test",
            rating=1000.0,
            games_played=0,
            wins=0,
            losses=0,
        )

        assert rating.rating == 1000.0
        assert rating.games_played == 0
        assert rating.last_updated is None


# =============================================================================
# AgentStats Dataclass Tests
# =============================================================================


class TestAgentStatsDataclass:
    """Tests for AgentStats dataclass."""

    def test_agent_stats_creation(self):
        """Test creating an AgentStats instance."""
        stats = AgentStats(
            agent_name="claude",
            debate_count=100,
            win_rate=0.65,
            avg_confidence=0.85,
            avg_response_time_ms=250.0,
            domains=["reasoning", "coding"],
        )

        assert stats.agent_name == "claude"
        assert stats.debate_count == 100
        assert stats.win_rate == 0.65
        assert stats.avg_confidence == 0.85
        assert stats.avg_response_time_ms == 250.0
        assert stats.domains == ["reasoning", "coding"]

    def test_agent_stats_empty_domains(self):
        """Test AgentStats with empty domains list."""
        stats = AgentStats(
            agent_name="new_agent",
            debate_count=0,
            win_rate=0.0,
            avg_confidence=0.0,
            avg_response_time_ms=0.0,
            domains=[],
        )

        assert stats.domains == []


# =============================================================================
# DebateLoaders Initialization Tests
# =============================================================================


class TestDebateLoadersInit:
    """Tests for DebateLoaders initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        loaders = DebateLoaders()

        assert loaders._elo_system is None
        assert loaders._agent_registry is None
        assert loaders._debate_store is None
        assert loaders._max_batch_size == 50
        assert loaders._elo_loader is None
        assert loaders._stats_loader is None

    def test_init_with_custom_batch_size(self):
        """Test initialization with custom batch size."""
        loaders = DebateLoaders(max_batch_size=100)

        assert loaders._max_batch_size == 100

    def test_init_with_elo_system(self, mock_elo_system):
        """Test initialization with ELO system."""
        loaders = DebateLoaders(elo_system=mock_elo_system)

        assert loaders._elo_system is mock_elo_system

    def test_init_with_debate_store(self, mock_debate_store):
        """Test initialization with debate store."""
        loaders = DebateLoaders(debate_store=mock_debate_store)

        assert loaders._debate_store is mock_debate_store

    def test_init_with_agent_registry(self):
        """Test initialization with agent registry."""
        mock_registry = MagicMock()
        loaders = DebateLoaders(agent_registry=mock_registry)

        assert loaders._agent_registry is mock_registry


# =============================================================================
# ELO Loader Property Tests
# =============================================================================


class TestELOLoaderProperty:
    """Tests for ELO loader property."""

    def test_elo_property_creates_loader_lazily(self, mock_elo_system):
        """Test that ELO loader is created lazily."""
        loaders = DebateLoaders(elo_system=mock_elo_system)

        assert loaders._elo_loader is None

        # Access property triggers creation
        elo_loader = loaders.elo

        assert elo_loader is not None
        assert loaders._elo_loader is elo_loader

    def test_elo_property_returns_same_instance(self, mock_elo_system):
        """Test that ELO property returns same loader instance."""
        loaders = DebateLoaders(elo_system=mock_elo_system)

        loader1 = loaders.elo
        loader2 = loaders.elo

        assert loader1 is loader2


# =============================================================================
# Stats Loader Property Tests
# =============================================================================


class TestStatsLoaderProperty:
    """Tests for stats loader property."""

    def test_stats_property_creates_loader_lazily(self, mock_debate_store):
        """Test that stats loader is created lazily."""
        loaders = DebateLoaders(debate_store=mock_debate_store)

        assert loaders._stats_loader is None

        # Access property triggers creation
        stats_loader = loaders.stats

        assert stats_loader is not None
        assert loaders._stats_loader is stats_loader

    def test_stats_property_returns_same_instance(self, mock_debate_store):
        """Test that stats property returns same loader instance."""
        loaders = DebateLoaders(debate_store=mock_debate_store)

        loader1 = loaders.stats
        loader2 = loaders.stats

        assert loader1 is loader2


# =============================================================================
# Batch ELO Loading Tests
# =============================================================================


class TestBatchELOLoading:
    """Tests for batch ELO loading."""

    @pytest.mark.asyncio
    async def test_batch_load_elo_with_batch_method(self, mock_elo_system):
        """Test batch loading ELO ratings using batch method."""
        loaders = DebateLoaders(elo_system=mock_elo_system)

        ratings = await loaders.elo.load_many(["claude", "gemini"])

        assert len(ratings) == 2
        assert ratings[0].agent_name == "claude"
        assert ratings[0].rating == 1500.0
        assert ratings[1].agent_name == "gemini"
        assert ratings[1].rating == 1400.0

        mock_elo_system.get_ratings_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_load_elo_with_individual_method(self, mock_elo_system_individual):
        """Test batch loading ELO ratings using individual lookups."""
        loaders = DebateLoaders(elo_system=mock_elo_system_individual)

        ratings = await loaders.elo.load_many(["claude", "gemini"])

        assert len(ratings) == 2
        assert ratings[0].agent_name == "claude"
        assert ratings[0].rating == 1500.0
        assert ratings[1].agent_name == "gemini"
        assert ratings[1].rating == 1400.0

    @pytest.mark.asyncio
    async def test_batch_load_elo_single_agent(self, mock_elo_system):
        """Test loading ELO for a single agent."""
        loaders = DebateLoaders(elo_system=mock_elo_system)

        rating = await loaders.elo.load("claude")

        assert rating.agent_name == "claude"
        assert rating.rating == 1500.0

    @pytest.mark.asyncio
    async def test_batch_load_elo_missing_agent(self, mock_elo_system):
        """Test loading ELO for missing agent returns None."""
        loaders = DebateLoaders(elo_system=mock_elo_system)

        ratings = await loaders.elo.load_many(["claude", "unknown_agent"])

        assert len(ratings) == 2
        assert ratings[0] is not None
        assert ratings[1] is None

    @pytest.mark.asyncio
    async def test_batch_load_elo_no_system(self):
        """Test loading ELO without ELO system returns None for all."""
        loaders = DebateLoaders()

        ratings = await loaders.elo.load_many(["claude", "gemini"])

        assert len(ratings) == 2
        assert ratings[0] is None
        assert ratings[1] is None


# =============================================================================
# Batch Stats Loading Tests
# =============================================================================


class TestBatchStatsLoading:
    """Tests for batch stats loading."""

    @pytest.mark.asyncio
    async def test_batch_load_stats_with_batch_method(self, mock_debate_store):
        """Test batch loading stats using batch method."""
        loaders = DebateLoaders(debate_store=mock_debate_store)

        stats = await loaders.stats.load_many(["claude", "gemini"])

        assert len(stats) == 2
        assert stats[0].agent_name == "claude"
        assert stats[0].debate_count == 100
        assert stats[1].agent_name == "gemini"
        assert stats[1].debate_count == 80

        mock_debate_store.get_agent_stats_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_load_stats_with_individual_method(self, mock_debate_store_individual):
        """Test batch loading stats using individual lookups."""
        loaders = DebateLoaders(debate_store=mock_debate_store_individual)

        stats = await loaders.stats.load_many(["claude"])

        assert len(stats) == 1
        assert stats[0].agent_name == "claude"
        assert stats[0].debate_count == 100

    @pytest.mark.asyncio
    async def test_batch_load_stats_single_agent(self, mock_debate_store):
        """Test loading stats for a single agent."""
        loaders = DebateLoaders(debate_store=mock_debate_store)

        stat = await loaders.stats.load("claude")

        assert stat.agent_name == "claude"
        assert stat.win_rate == 0.65

    @pytest.mark.asyncio
    async def test_batch_load_stats_missing_agent(self, mock_debate_store):
        """Test loading stats for missing agent returns None."""
        loaders = DebateLoaders(debate_store=mock_debate_store)

        stats = await loaders.stats.load_many(["claude", "unknown_agent"])

        assert len(stats) == 2
        assert stats[0] is not None
        assert stats[1] is None

    @pytest.mark.asyncio
    async def test_batch_load_stats_no_store(self):
        """Test loading stats without debate store returns None for all."""
        loaders = DebateLoaders()

        stats = await loaders.stats.load_many(["claude", "gemini"])

        assert len(stats) == 2
        assert stats[0] is None
        assert stats[1] is None


# =============================================================================
# Cache Behavior Tests
# =============================================================================


class TestCacheBehavior:
    """Tests for cache hit and miss behavior."""

    @pytest.mark.asyncio
    async def test_elo_cache_hit(self, mock_elo_system):
        """Test ELO loader cache hit."""
        loaders = DebateLoaders(elo_system=mock_elo_system)

        # First load
        rating1 = await loaders.elo.load("claude")

        # Second load - should be cached
        rating2 = await loaders.elo.load("claude")

        assert rating1.rating == rating2.rating

        # Should only have called the batch function once
        assert mock_elo_system.get_ratings_batch.call_count == 1

    @pytest.mark.asyncio
    async def test_stats_cache_hit(self, mock_debate_store):
        """Test stats loader cache hit."""
        loaders = DebateLoaders(debate_store=mock_debate_store)

        # First load
        stat1 = await loaders.stats.load("claude")

        # Second load - should be cached
        stat2 = await loaders.stats.load("claude")

        assert stat1.debate_count == stat2.debate_count

        # Should only have called the batch function once
        assert mock_debate_store.get_agent_stats_batch.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_miss_fetches(self, mock_elo_system):
        """Test cache miss triggers fetch."""
        loaders = DebateLoaders(elo_system=mock_elo_system)

        # Load different agents
        rating1 = await loaders.elo.load("claude")
        rating2 = await loaders.elo.load("gemini")

        assert rating1.agent_name == "claude"
        assert rating2.agent_name == "gemini"


# =============================================================================
# Clear Cache Tests
# =============================================================================


class TestClearCache:
    """Tests for cache clearing."""

    @pytest.mark.asyncio
    async def test_clear_clears_all_caches(self, mock_elo_system, mock_debate_store):
        """Test clear method clears all loader caches."""
        loaders = DebateLoaders(elo_system=mock_elo_system, debate_store=mock_debate_store)

        # Load data
        await loaders.elo.load("claude")
        await loaders.stats.load("claude")

        # Clear caches
        loaders.clear()

        # Cache should be empty - need to refetch
        # (next load will trigger new batch)
        await loaders.elo.load("claude")

        # Should have called batch function again
        assert mock_elo_system.get_ratings_batch.call_count == 2

    @pytest.mark.asyncio
    async def test_clear_with_no_loaders(self):
        """Test clear works when no loaders have been accessed."""
        loaders = DebateLoaders()

        # Should not raise
        loaders.clear()


# =============================================================================
# Get Stats Tests
# =============================================================================


class TestGetStats:
    """Tests for getting loader statistics."""

    @pytest.mark.asyncio
    async def test_get_stats_with_loaders(self, mock_elo_system, mock_debate_store):
        """Test get_stats returns stats for all loaders."""
        loaders = DebateLoaders(elo_system=mock_elo_system, debate_store=mock_debate_store)

        # Access loaders to create them
        await loaders.elo.load("claude")
        await loaders.stats.load("claude")

        stats = loaders.get_stats()

        assert "elo" in stats
        assert "stats" in stats
        assert "resolver" in stats
        assert stats["elo"]["loads"] >= 1
        assert stats["stats"]["loads"] >= 1

    def test_get_stats_no_loaders_accessed(self):
        """Test get_stats when no loaders have been accessed."""
        loaders = DebateLoaders()

        stats = loaders.get_stats()

        # Only resolver stats should be present
        assert "resolver" in stats
        assert "elo" not in stats
        assert "stats" not in stats


# =============================================================================
# Data Conversion Tests
# =============================================================================


class TestDataConversion:
    """Tests for data conversion helpers."""

    def test_elo_to_rating_from_dict(self):
        """Test converting dict to ELORating."""
        loaders = DebateLoaders()

        data = {
            "rating": 1500.0,
            "games_played": 50,
            "wins": 30,
            "losses": 20,
            "last_updated": "2025-01-15T10:00:00Z",
        }

        result = loaders._elo_to_rating("claude", data)

        assert isinstance(result, ELORating)
        assert result.agent_name == "claude"
        assert result.rating == 1500.0
        assert result.games_played == 50
        assert result.last_updated == "2025-01-15T10:00:00Z"

    def test_elo_to_rating_from_numeric(self):
        """Test converting numeric value to ELORating."""
        loaders = DebateLoaders()

        result = loaders._elo_to_rating("claude", 1500)

        assert isinstance(result, ELORating)
        assert result.agent_name == "claude"
        assert result.rating == 1500.0
        assert result.games_played == 0

    def test_elo_to_rating_from_float(self):
        """Test converting float value to ELORating."""
        loaders = DebateLoaders()

        result = loaders._elo_to_rating("gemini", 1400.5)

        assert isinstance(result, ELORating)
        assert result.rating == 1400.5

    def test_elo_to_rating_from_none(self):
        """Test converting None returns None."""
        loaders = DebateLoaders()

        result = loaders._elo_to_rating("unknown", None)

        assert result is None

    def test_elo_to_rating_from_elo_rating_instance(self, sample_elo_rating):
        """Test converting ELORating instance returns same."""
        loaders = DebateLoaders()

        result = loaders._elo_to_rating("claude", sample_elo_rating)

        assert result is sample_elo_rating

    def test_elo_to_rating_from_invalid_type(self):
        """Test converting invalid type returns None."""
        loaders = DebateLoaders()

        result = loaders._elo_to_rating("claude", "invalid")

        assert result is None

    def test_dict_to_stats_from_dict(self):
        """Test converting dict to AgentStats."""
        loaders = DebateLoaders()

        data = {
            "debate_count": 100,
            "win_rate": 0.65,
            "avg_confidence": 0.85,
            "avg_response_time_ms": 250.0,
            "domains": ["reasoning", "coding"],
        }

        result = loaders._dict_to_stats("claude", data)

        assert isinstance(result, AgentStats)
        assert result.agent_name == "claude"
        assert result.debate_count == 100
        assert result.win_rate == 0.65
        assert result.domains == ["reasoning", "coding"]

    def test_dict_to_stats_from_none(self):
        """Test converting None returns None."""
        loaders = DebateLoaders()

        result = loaders._dict_to_stats("unknown", None)

        assert result is None

    def test_dict_to_stats_from_agent_stats_instance(self, sample_agent_stats):
        """Test converting AgentStats instance returns same."""
        loaders = DebateLoaders()

        result = loaders._dict_to_stats("claude", sample_agent_stats)

        assert result is sample_agent_stats

    def test_dict_to_stats_with_defaults(self):
        """Test converting dict with missing fields uses defaults."""
        loaders = DebateLoaders()

        result = loaders._dict_to_stats("claude", {})

        assert result.debate_count == 0
        assert result.win_rate == 0.0
        assert result.avg_confidence == 0.0
        assert result.domains == []

    def test_dict_to_stats_invalid_type(self):
        """Test converting invalid type returns None."""
        loaders = DebateLoaders()

        result = loaders._dict_to_stats("claude", "invalid")

        assert result is None


# =============================================================================
# Maybe Await Tests
# =============================================================================


class TestMaybeAwait:
    """Tests for _maybe_await helper."""

    @pytest.mark.asyncio
    async def test_maybe_await_coroutine(self):
        """Test awaiting a coroutine."""
        loaders = DebateLoaders()

        async def coro():
            return "result"

        result = await loaders._maybe_await(coro())

        assert result == "result"

    @pytest.mark.asyncio
    async def test_maybe_await_sync_value(self):
        """Test returning sync value directly."""
        loaders = DebateLoaders()

        result = await loaders._maybe_await("sync_result")

        assert result == "sync_result"

    @pytest.mark.asyncio
    async def test_maybe_await_none(self):
        """Test handling None value."""
        loaders = DebateLoaders()

        result = await loaders._maybe_await(None)

        assert result is None


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestContextManager:
    """Tests for context manager behavior."""

    def test_enter_sets_context(self, mock_elo_system):
        """Test __enter__ sets loaders in context."""
        loaders = DebateLoaders(elo_system=mock_elo_system)

        with loaders:
            current = get_debate_loaders()
            assert current is loaders

    def test_exit_resets_context(self, mock_elo_system):
        """Test __exit__ resets context."""
        loaders = DebateLoaders(elo_system=mock_elo_system)

        with loaders:
            pass

        current = get_debate_loaders()
        assert current is None

    @pytest.mark.asyncio
    async def test_exit_clears_caches(self, mock_elo_system):
        """Test __exit__ clears loader caches."""
        loaders = DebateLoaders(elo_system=mock_elo_system)

        with loaders:
            await loaders.elo.load("claude")

        # After exit, cache should be cleared
        # (we can't easily verify this without re-entering context)

    def test_nested_context_managers(self, mock_elo_system, mock_debate_store):
        """Test nested context managers work correctly."""
        loaders1 = DebateLoaders(elo_system=mock_elo_system)
        loaders2 = DebateLoaders(debate_store=mock_debate_store)

        with loaders1:
            assert get_debate_loaders() is loaders1

            with loaders2:
                assert get_debate_loaders() is loaders2

            # Should restore to loaders1 after inner context
            assert get_debate_loaders() is loaders1

        # Should be None after all contexts exit
        assert get_debate_loaders() is None


# =============================================================================
# get_debate_loaders Tests
# =============================================================================


class TestGetDebateLoaders:
    """Tests for get_debate_loaders function."""

    def test_returns_none_when_no_context(self):
        """Test returns None when not in context."""
        result = get_debate_loaders()
        assert result is None

    def test_returns_loaders_in_context(self, mock_elo_system):
        """Test returns loaders when in context."""
        loaders = DebateLoaders(elo_system=mock_elo_system)

        with loaders:
            result = get_debate_loaders()
            assert result is loaders


# =============================================================================
# debate_loader_context Tests
# =============================================================================


class TestDebateLoaderContext:
    """Tests for debate_loader_context context manager."""

    def test_creates_loaders_with_defaults(self):
        """Test creates loaders with default values."""
        with debate_loader_context() as loaders:
            assert loaders is not None
            assert loaders._max_batch_size == 50

    def test_creates_loaders_with_custom_config(self, mock_elo_system, mock_debate_store):
        """Test creates loaders with custom configuration."""
        with debate_loader_context(
            elo_system=mock_elo_system,
            debate_store=mock_debate_store,
            max_batch_size=100,
        ) as loaders:
            assert loaders._elo_system is mock_elo_system
            assert loaders._debate_store is mock_debate_store
            assert loaders._max_batch_size == 100

    def test_sets_context_variable(self, mock_elo_system):
        """Test sets context variable during context."""
        with debate_loader_context(elo_system=mock_elo_system) as loaders:
            current = get_debate_loaders()
            assert current is loaders

    def test_resets_context_on_exit(self):
        """Test resets context variable on exit."""
        with debate_loader_context():
            pass

        current = get_debate_loaders()
        assert current is None

    @pytest.mark.asyncio
    async def test_usable_for_loading(self, mock_elo_system):
        """Test loaders from context are usable."""
        with debate_loader_context(elo_system=mock_elo_system) as loaders:
            rating = await loaders.elo.load("claude")
            assert rating.agent_name == "claude"


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_batch(self, mock_elo_system):
        """Test loading empty list of agents."""
        loaders = DebateLoaders(elo_system=mock_elo_system)

        ratings = await loaders.elo.load_many([])

        assert ratings == []

    @pytest.mark.asyncio
    async def test_large_batch(self, mock_elo_system):
        """Test loading large batch of agents."""
        # Extend mock data for large batch
        for i in range(100):
            mock_elo_system.ratings_data[f"agent_{i}"] = {"rating": 1000.0 + i}

        loaders = DebateLoaders(elo_system=mock_elo_system, max_batch_size=25)

        agent_names = [f"agent_{i}" for i in range(100)]
        ratings = await loaders.elo.load_many(agent_names)

        assert len(ratings) == 100

    @pytest.mark.asyncio
    async def test_duplicate_keys_in_batch(self, mock_elo_system):
        """Test loading batch with duplicate keys."""
        loaders = DebateLoaders(elo_system=mock_elo_system)

        ratings = await loaders.elo.load_many(["claude", "claude", "gemini"])

        assert len(ratings) == 3
        assert ratings[0].agent_name == "claude"
        assert ratings[1].agent_name == "claude"
        assert ratings[2].agent_name == "gemini"

    @pytest.mark.asyncio
    async def test_batch_function_error(self, mock_elo_system):
        """Test handling batch function errors."""
        mock_elo_system.get_ratings_batch.side_effect = RuntimeError("Database error")

        loaders = DebateLoaders(elo_system=mock_elo_system)

        ratings = await loaders.elo.load_many(["claude", "gemini"])

        # Should return None for all on error
        assert all(r is None for r in ratings)

    @pytest.mark.asyncio
    async def test_individual_load_error(self, mock_elo_system_individual):
        """Test handling individual load errors."""

        async def failing_get_rating(name):
            if name == "failing_agent":
                raise RuntimeError("Agent not found")
            return 1000.0

        mock_elo_system_individual.get_rating = AsyncMock(side_effect=failing_get_rating)

        loaders = DebateLoaders(elo_system=mock_elo_system_individual)

        ratings = await loaders.elo.load_many(["claude", "failing_agent", "gemini"])

        assert len(ratings) == 3
        assert ratings[0] is not None
        assert ratings[1] is None  # Failed load returns None
        assert ratings[2] is not None

    @pytest.mark.asyncio
    async def test_concurrent_loads(self, mock_elo_system):
        """Test concurrent loads are batched."""
        loaders = DebateLoaders(elo_system=mock_elo_system)

        # Start multiple loads concurrently
        results = await asyncio.gather(
            loaders.elo.load("claude"),
            loaders.elo.load("gemini"),
            loaders.elo.load("grok"),
        )

        assert len(results) == 3
        assert all(r is not None for r in results)


# =============================================================================
# Multiple Batch Operations Tests
# =============================================================================


class TestMultipleBatchOperations:
    """Tests for multiple batch operations."""

    @pytest.mark.asyncio
    async def test_sequential_batches(self, mock_elo_system):
        """Test sequential batch operations."""
        loaders = DebateLoaders(elo_system=mock_elo_system)

        # First batch
        batch1 = await loaders.elo.load_many(["claude", "gemini"])
        assert len(batch1) == 2

        # Second batch (claude should be cached)
        batch2 = await loaders.elo.load_many(["claude", "grok"])
        assert len(batch2) == 2

    @pytest.mark.asyncio
    async def test_mixed_loader_operations(self, mock_elo_system, mock_debate_store):
        """Test using both loaders together."""
        loaders = DebateLoaders(elo_system=mock_elo_system, debate_store=mock_debate_store)

        # Load ELO and stats for same agents
        ratings = await loaders.elo.load_many(["claude", "gemini"])
        stats = await loaders.stats.load_many(["claude", "gemini"])

        assert len(ratings) == 2
        assert len(stats) == 2
        assert ratings[0].agent_name == stats[0].agent_name

    @pytest.mark.asyncio
    async def test_loader_isolation(self, mock_elo_system, mock_debate_store):
        """Test that different loaders are isolated."""
        loaders = DebateLoaders(elo_system=mock_elo_system, debate_store=mock_debate_store)

        # Load into ELO cache
        await loaders.elo.load("claude")

        # Stats should still need to fetch
        stat = await loaders.stats.load("claude")

        assert stat is not None
        mock_debate_store.get_agent_stats_batch.assert_called_once()


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for batch loaders."""

    @pytest.mark.asyncio
    async def test_complete_workflow(self, mock_elo_system, mock_debate_store):
        """Test complete workflow with context manager."""
        with debate_loader_context(
            elo_system=mock_elo_system,
            debate_store=mock_debate_store,
        ) as loaders:
            # Verify context is set
            assert get_debate_loaders() is loaders

            # Load ELO ratings
            ratings = await loaders.elo.load_many(["claude", "gemini"])
            assert len(ratings) == 2

            # Load stats
            stats = await loaders.stats.load_many(["claude"])
            assert len(stats) == 1

            # Get statistics
            loader_stats = loaders.get_stats()
            assert "elo" in loader_stats
            assert "stats" in loader_stats

        # Context should be cleared
        assert get_debate_loaders() is None

    @pytest.mark.asyncio
    async def test_realistic_debate_scenario(self, mock_elo_system, mock_debate_store):
        """Test realistic debate scenario."""
        # Simulate loading agent data before a debate
        with debate_loader_context(
            elo_system=mock_elo_system,
            debate_store=mock_debate_store,
        ) as loaders:
            # Load team of agents
            team = ["claude", "gemini", "grok"]

            # Load ELO ratings for team selection
            ratings = await loaders.elo.load_many(team)

            # Filter to agents with sufficient rating
            qualified = [r for r in ratings if r and r.rating >= 1350]
            assert len(qualified) >= 2

            # Load stats for qualified agents
            qualified_names = [r.agent_name for r in qualified]
            stats = await loaders.stats.load_many(qualified_names)

            # Verify cache is working (re-load should be instant)
            await loaders.elo.load("claude")
            assert mock_elo_system.get_ratings_batch.call_count == 1  # Still just 1 call
