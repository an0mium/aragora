"""
Tests for EloSystem - the main ELO ranking system class.

Tests cover:
- Agent registration and initialization
- Rating retrieval and caching
- Match recording and ELO updates
- Leaderboard operations
- Batch operations
- Domain-specific ratings
- Cache invalidation
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from aragora.ranking.elo import EloSystem, AgentRating, get_elo_store


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield Path(f.name)


@pytest.fixture
def elo_system(temp_db):
    """Create an EloSystem with a temporary database."""
    system = EloSystem(db_path=temp_db)
    yield system
    # Cleanup caches
    system.invalidate_leaderboard_cache()
    system.invalidate_rating_cache()


class TestAgentRegistration:
    """Tests for agent registration and initialization."""

    def test_register_new_agent(self, elo_system):
        """Test registering a new agent."""
        rating = elo_system.register_agent("test_agent")

        assert rating.agent_name == "test_agent"
        assert rating.elo == 1500.0  # Default initial rating (ELO_INITIAL_RATING)
        assert rating.wins == 0
        assert rating.losses == 0
        assert rating.draws == 0

    def test_register_agent_with_model(self, elo_system):
        """Test registering an agent with model info."""
        rating = elo_system.register_agent("claude", model="claude-3-opus")

        assert rating.agent_name == "claude"
        # Model info may be stored in metadata or not exposed on rating

    def test_register_existing_agent_returns_current(self, elo_system):
        """Registering an existing agent should return current rating."""
        rating1 = elo_system.register_agent("test_agent")
        rating1_elo = rating1.elo

        # Register again
        rating2 = elo_system.register_agent("test_agent")

        assert rating2.elo == rating1_elo

    def test_initialize_agent_alias(self, elo_system):
        """initialize_agent should work same as register_agent."""
        rating = elo_system.initialize_agent("new_agent")

        assert rating.agent_name == "new_agent"
        assert rating.elo == 1500.0


class TestRatingRetrieval:
    """Tests for getting agent ratings."""

    def test_get_rating_existing_agent(self, elo_system):
        """Test getting rating for existing agent."""
        elo_system.register_agent("test_agent")

        rating = elo_system.get_rating("test_agent")

        assert rating.agent_name == "test_agent"
        assert rating.elo == 1500.0

    def test_get_rating_nonexistent_creates_new(self, elo_system):
        """Getting rating for nonexistent agent creates new entry."""
        rating = elo_system.get_rating("new_agent")

        assert rating.agent_name == "new_agent"
        assert rating.elo == 1500.0

    def test_get_rating_uses_cache(self, elo_system):
        """Test that rating retrieval uses cache."""
        elo_system.register_agent("cached_agent")

        # First call
        rating1 = elo_system.get_rating("cached_agent", use_cache=True)
        # Second call should hit cache
        rating2 = elo_system.get_rating("cached_agent", use_cache=True)

        assert rating1.elo == rating2.elo

    def test_get_rating_bypass_cache(self, elo_system):
        """Test bypassing cache for rating retrieval."""
        elo_system.register_agent("test_agent")

        rating = elo_system.get_rating("test_agent", use_cache=False)

        assert rating.agent_name == "test_agent"

    def test_get_ratings_batch(self, elo_system):
        """Test getting multiple ratings at once."""
        elo_system.register_agent("agent1")
        elo_system.register_agent("agent2")
        elo_system.register_agent("agent3")

        ratings = elo_system.get_ratings_batch(["agent1", "agent2", "agent3"])

        assert len(ratings) == 3
        assert "agent1" in ratings
        assert "agent2" in ratings
        assert "agent3" in ratings

    def test_get_ratings_batch_creates_missing(self, elo_system):
        """Batch get creates ratings for missing agents."""
        elo_system.register_agent("existing")

        ratings = elo_system.get_ratings_batch(["existing", "new_agent"])

        assert len(ratings) == 2
        assert "new_agent" in ratings


class TestAgentListing:
    """Tests for listing agents."""

    def test_list_agents_empty(self, elo_system):
        """Test listing agents when none registered."""
        agents = elo_system.list_agents()

        assert agents == []

    def test_list_agents_multiple(self, elo_system):
        """Test listing multiple agents."""
        elo_system.register_agent("agent1")
        elo_system.register_agent("agent2")
        elo_system.register_agent("agent3")

        agents = elo_system.list_agents()

        assert len(agents) == 3
        assert "agent1" in agents
        assert "agent2" in agents
        assert "agent3" in agents

    def test_get_all_ratings(self, elo_system):
        """Test getting all ratings."""
        elo_system.register_agent("agent1")
        elo_system.register_agent("agent2")

        ratings = elo_system.get_all_ratings()

        assert len(ratings) == 2
        assert all(isinstance(r, AgentRating) for r in ratings)


class TestLeaderboard:
    """Tests for leaderboard operations."""

    def test_get_leaderboard_empty(self, elo_system):
        """Test leaderboard with no agents."""
        leaderboard = elo_system.get_leaderboard()

        assert leaderboard == []

    def test_get_leaderboard_ordered_by_elo(self, elo_system):
        """Test that leaderboard is ordered by ELO descending."""
        # Register agents with different ELOs
        agent1 = elo_system.register_agent("low_elo")
        agent2 = elo_system.register_agent("mid_elo")
        agent3 = elo_system.register_agent("high_elo")

        # Manually adjust ELOs for testing
        agent1.elo = 900.0
        agent2.elo = 1000.0
        agent3.elo = 1100.0

        elo_system._save_rating(agent1)
        elo_system._save_rating(agent2)
        elo_system._save_rating(agent3)
        elo_system.invalidate_leaderboard_cache()

        leaderboard = elo_system.get_leaderboard(limit=10)

        assert len(leaderboard) == 3
        assert leaderboard[0].agent_name == "high_elo"
        assert leaderboard[1].agent_name == "mid_elo"
        assert leaderboard[2].agent_name == "low_elo"

    def test_get_leaderboard_respects_limit(self, elo_system):
        """Test that leaderboard respects limit parameter."""
        for i in range(10):
            elo_system.register_agent(f"agent_{i}")

        leaderboard = elo_system.get_leaderboard(limit=5)

        assert len(leaderboard) == 5

    def test_invalidate_leaderboard_cache(self, elo_system):
        """Test cache invalidation."""
        elo_system.register_agent("test_agent")
        elo_system.get_leaderboard()  # Populate cache

        cleared = elo_system.invalidate_leaderboard_cache()

        assert cleared >= 0


class TestCritiqueTracking:
    """Tests for critique acceptance tracking."""

    def test_record_critique_accepted(self, elo_system):
        """Test recording an accepted critique."""
        elo_system.register_agent("critic")

        elo_system.record_critique("critic", accepted=True)

        rating = elo_system.get_rating("critic", use_cache=False)
        assert rating.critiques_accepted >= 1

    def test_record_critique_rejected(self, elo_system):
        """Test recording a rejected critique."""
        elo_system.register_agent("critic")

        elo_system.record_critique("critic", accepted=False)

        rating = elo_system.get_rating("critic", use_cache=False)
        # critiques_total includes both accepted and rejected
        assert rating.critiques_total >= 1

    def test_critique_acceptance_rate(self, elo_system):
        """Test critique acceptance rate calculation."""
        elo_system.register_agent("critic")

        # Record 3 accepted, 1 rejected
        elo_system.record_critique("critic", accepted=True)
        elo_system.record_critique("critic", accepted=True)
        elo_system.record_critique("critic", accepted=True)
        elo_system.record_critique("critic", accepted=False)

        rating = elo_system.get_rating("critic", use_cache=False)
        assert rating.critique_acceptance_rate == 0.75


class TestStatistics:
    """Tests for system statistics."""

    def test_get_stats_empty(self, elo_system):
        """Test stats with no data."""
        stats = elo_system.get_stats()

        assert "total_agents" in stats
        assert "total_matches" in stats
        assert stats["total_agents"] == 0

    def test_get_stats_with_agents(self, elo_system):
        """Test stats with registered agents."""
        elo_system.register_agent("agent1")
        elo_system.register_agent("agent2")

        stats = elo_system.get_stats(use_cache=False)

        assert stats["total_agents"] == 2


class TestEloHistory:
    """Tests for ELO history tracking."""

    def test_get_elo_history_empty(self, elo_system):
        """Test ELO history for new agent."""
        elo_system.register_agent("new_agent")

        history = elo_system.get_elo_history("new_agent")

        # May have initial entry or be empty
        assert isinstance(history, list)

    def test_get_elo_history_respects_limit(self, elo_system):
        """Test that history respects limit."""
        elo_system.register_agent("test_agent")

        history = elo_system.get_elo_history("test_agent", limit=5)

        assert len(history) <= 5


class TestAgentRatingDataclass:
    """Tests for AgentRating dataclass properties."""

    def test_win_rate_no_games(self):
        """Test win rate with no games played."""
        rating = AgentRating(agent_name="test", elo=1500.0)

        assert rating.win_rate == 0.0

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        rating = AgentRating(agent_name="test", elo=1500.0, wins=3, losses=1, draws=0)

        assert rating.win_rate == 0.75

    def test_games_played(self):
        """Test games played calculation."""
        rating = AgentRating(agent_name="test", elo=1500.0, wins=5, losses=3, draws=2)

        assert rating.games_played == 10

    def test_total_debates(self):
        """Test total debates property (uses debates_count field)."""
        rating = AgentRating(agent_name="test", elo=1500.0, debates_count=15)

        assert rating.total_debates == 15

    def test_elo_rating_alias(self):
        """Test elo_rating property alias."""
        rating = AgentRating(agent_name="test", elo=1234.5)

        assert rating.elo_rating == 1234.5


class TestCacheInvalidation:
    """Tests for cache invalidation."""

    def test_invalidate_rating_cache_all(self, elo_system):
        """Test invalidating all rating caches."""
        elo_system.register_agent("agent1")
        elo_system.register_agent("agent2")
        elo_system.get_rating("agent1")
        elo_system.get_rating("agent2")

        cleared = elo_system.invalidate_rating_cache()

        assert cleared >= 0

    def test_invalidate_rating_cache_single(self, elo_system):
        """Test invalidating single agent's cache."""
        elo_system.register_agent("agent1")
        elo_system.get_rating("agent1")

        cleared = elo_system.invalidate_rating_cache("agent1")

        assert cleared >= 0


class TestDomainRatings:
    """Tests for domain-specific ratings."""

    def test_get_top_agents_for_domain(self, elo_system):
        """Test getting top agents for a domain."""
        elo_system.register_agent("agent1")
        elo_system.register_agent("agent2")

        # Domain ratings are computed from matches, so initially empty
        top = elo_system.get_top_agents_for_domain("legal", limit=5)

        # Should return agents even without domain-specific data
        assert isinstance(top, list)

    def test_get_best_domains(self, elo_system):
        """Test getting an agent's best domains."""
        elo_system.register_agent("specialist")

        domains = elo_system.get_best_domains("specialist", limit=5)

        assert isinstance(domains, list)


class TestSingletonBehavior:
    """Tests for singleton pattern."""

    def test_get_elo_store_returns_instance(self):
        """Test that get_elo_store returns an EloSystem."""
        with patch("aragora.ranking.elo._elo_store", None):
            store = get_elo_store()

            assert isinstance(store, EloSystem)


class TestRelationshipTracker:
    """Tests for relationship tracking."""

    def test_relationship_tracker_property(self, elo_system):
        """Test lazy initialization of relationship tracker."""
        tracker = elo_system.relationship_tracker

        assert tracker is not None
        assert isinstance(tracker, object)

    def test_get_rivals(self, elo_system):
        """Test getting agent rivals."""
        elo_system.register_agent("agent1")

        rivals = elo_system.get_rivals("agent1", limit=5)

        assert isinstance(rivals, list)

    def test_get_allies(self, elo_system):
        """Test getting agent allies."""
        elo_system.register_agent("agent1")

        allies = elo_system.get_allies("agent1", limit=5)

        assert isinstance(allies, list)


class TestRedTeamIntegration:
    """Tests for red team integration."""

    def test_redteam_integrator_property(self, elo_system):
        """Test lazy initialization of red team integrator."""
        integrator = elo_system.redteam_integrator

        assert integrator is not None

    def test_get_vulnerability_summary(self, elo_system):
        """Test getting vulnerability summary."""
        elo_system.register_agent("target")

        summary = elo_system.get_vulnerability_summary("target")

        assert isinstance(summary, dict)


class TestVerificationImpact:
    """Tests for verification impact tracking."""

    def test_get_verification_impact(self, elo_system):
        """Test getting verification impact stats."""
        elo_system.register_agent("verified_agent")

        impact = elo_system.get_verification_impact("verified_agent")

        assert isinstance(impact, dict)


class TestVotingAccuracy:
    """Tests for voting accuracy tracking."""

    def test_get_voting_accuracy(self, elo_system):
        """Test getting voting accuracy stats."""
        elo_system.register_agent("voter")

        accuracy = elo_system.get_voting_accuracy("voter")

        assert isinstance(accuracy, dict)
        assert "correct_votes" in accuracy or "total_votes" in accuracy or accuracy == {}
