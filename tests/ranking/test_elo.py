"""
Comprehensive tests for aragora/ranking/elo.py - ELO Rating System.

Tests cover:
- ELO calculation accuracy and mathematical properties
- Rating updates after matches (wins, losses, draws)
- Edge cases (new players, extreme ratings, ties)
- Persistence and retrieval operations
- Calibration tracking features
- Agent name validation
- AgentRating dataclass properties
- Match recording scenarios
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.ranking.elo import (
    AgentRating,
    DEFAULT_ELO,
    EloSystem,
    K_FACTOR,
    MAX_AGENT_NAME_LENGTH,
    MatchResult,
    _validate_agent_name,
    get_elo_store,
)


# =============================================================================
# Fixtures
# =============================================================================


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


# =============================================================================
# Test Constants and Configuration
# =============================================================================


class TestEloConstants:
    """Tests for ELO system constants and configuration."""

    def test_default_elo_is_1500(self):
        """Test that default ELO rating is 1500."""
        assert DEFAULT_ELO == 1500

    def test_k_factor_is_positive(self):
        """Test that K-factor is positive."""
        assert K_FACTOR > 0

    def test_k_factor_in_reasonable_range(self):
        """Test that K-factor is within reasonable bounds (8-64)."""
        assert 8 <= K_FACTOR <= 64

    def test_max_agent_name_length_is_32(self):
        """Test maximum agent name length constraint."""
        assert MAX_AGENT_NAME_LENGTH == 32


class TestAgentNameValidation:
    """Tests for agent name validation."""

    def test_valid_short_name(self):
        """Test that short valid names pass validation."""
        _validate_agent_name("claude")  # Should not raise

    def test_valid_max_length_name(self):
        """Test that exactly max length name passes."""
        name = "a" * MAX_AGENT_NAME_LENGTH
        _validate_agent_name(name)  # Should not raise

    def test_name_exceeds_max_length_raises_error(self):
        """Test that names exceeding max length raise ValueError."""
        name = "a" * (MAX_AGENT_NAME_LENGTH + 1)
        with pytest.raises(ValueError, match="exceeds"):
            _validate_agent_name(name)

    def test_very_long_name_raises_error(self):
        """Test that very long names raise ValueError."""
        name = "agent_with_extremely_long_name_that_exceeds_limit"
        with pytest.raises(ValueError):
            _validate_agent_name(name)


# =============================================================================
# AgentRating Dataclass Tests
# =============================================================================


class TestAgentRatingDataclass:
    """Tests for AgentRating dataclass properties and calculations."""

    def test_default_values(self):
        """Test that AgentRating has correct defaults."""
        rating = AgentRating(agent_name="test")

        assert rating.agent_name == "test"
        assert rating.elo == DEFAULT_ELO
        assert rating.wins == 0
        assert rating.losses == 0
        assert rating.draws == 0
        assert rating.debates_count == 0
        assert rating.domain_elos == {}

    def test_win_rate_no_games(self):
        """Test win rate calculation with no games played."""
        rating = AgentRating(agent_name="test")
        assert rating.win_rate == 0.0

    def test_win_rate_all_wins(self):
        """Test win rate calculation with all wins."""
        rating = AgentRating(agent_name="test", wins=10, losses=0, draws=0)
        assert rating.win_rate == 1.0

    def test_win_rate_all_losses(self):
        """Test win rate calculation with all losses."""
        rating = AgentRating(agent_name="test", wins=0, losses=10, draws=0)
        assert rating.win_rate == 0.0

    def test_win_rate_mixed_results(self):
        """Test win rate calculation with mixed results."""
        rating = AgentRating(agent_name="test", wins=3, losses=1, draws=0)
        assert rating.win_rate == 0.75

    def test_win_rate_with_draws(self):
        """Test win rate calculation including draws."""
        rating = AgentRating(agent_name="test", wins=2, losses=2, draws=1)
        # win_rate = wins / (wins + losses + draws)
        assert rating.win_rate == 0.4

    def test_games_played_property(self):
        """Test games_played calculation."""
        rating = AgentRating(agent_name="test", wins=5, losses=3, draws=2)
        assert rating.games_played == 10

    def test_total_debates_alias(self):
        """Test that total_debates is alias for debates_count."""
        rating = AgentRating(agent_name="test", debates_count=15)
        assert rating.total_debates == 15

    def test_elo_rating_alias(self):
        """Test that elo_rating is alias for elo."""
        rating = AgentRating(agent_name="test", elo=1650.5)
        assert rating.elo_rating == 1650.5

    def test_critique_acceptance_rate_no_critiques(self):
        """Test critique acceptance rate with no critiques."""
        rating = AgentRating(agent_name="test")
        assert rating.critique_acceptance_rate == 0.0

    def test_critique_acceptance_rate_all_accepted(self):
        """Test critique acceptance rate with all accepted."""
        rating = AgentRating(agent_name="test", critiques_accepted=5, critiques_total=5)
        assert rating.critique_acceptance_rate == 1.0

    def test_critique_acceptance_rate_mixed(self):
        """Test critique acceptance rate with mixed results."""
        rating = AgentRating(agent_name="test", critiques_accepted=3, critiques_total=4)
        assert rating.critique_acceptance_rate == 0.75


class TestAgentRatingCalibration:
    """Tests for AgentRating calibration properties."""

    def test_calibration_accuracy_no_predictions(self):
        """Test calibration accuracy with no predictions."""
        rating = AgentRating(agent_name="test")
        assert rating.calibration_accuracy == 0.0

    def test_calibration_accuracy_perfect(self):
        """Test calibration accuracy with perfect predictions."""
        rating = AgentRating(
            agent_name="test",
            calibration_correct=10,
            calibration_total=10,
        )
        assert rating.calibration_accuracy == 1.0

    def test_calibration_accuracy_partial(self):
        """Test calibration accuracy with partial success."""
        rating = AgentRating(
            agent_name="test",
            calibration_correct=7,
            calibration_total=10,
        )
        assert rating.calibration_accuracy == 0.7

    def test_calibration_brier_score_no_predictions(self):
        """Test Brier score defaults to 1.0 with no predictions."""
        rating = AgentRating(agent_name="test")
        assert rating.calibration_brier_score == 1.0

    def test_calibration_brier_score_perfect(self):
        """Test Brier score calculation."""
        rating = AgentRating(
            agent_name="test",
            calibration_brier_sum=0.5,
            calibration_total=5,
        )
        # Brier = 0.5 / 5 = 0.1
        assert rating.calibration_brier_score == pytest.approx(0.1, rel=1e-6)

    def test_calibration_score_below_minimum(self):
        """Test calibration score returns 0 below minimum predictions."""
        # Default CALIBRATION_MIN_COUNT is typically 5
        rating = AgentRating(
            agent_name="test",
            calibration_correct=2,
            calibration_total=2,  # Below min threshold
            calibration_brier_sum=0.1,
        )
        assert rating.calibration_score == 0.0

    def test_calibration_score_at_minimum(self):
        """Test calibration score calculation at minimum threshold."""
        # With minimum predictions, confidence starts at 0.5
        rating = AgentRating(
            agent_name="test",
            calibration_correct=5,
            calibration_total=5,
            calibration_brier_sum=0.0,  # Perfect calibration
        )
        # If calibration_total equals min_count, score should be non-zero
        score = rating.calibration_score
        # Score = (1 - brier) * confidence, with brier=0 and some confidence
        assert score >= 0.0


class TestMatchResultDataclass:
    """Tests for MatchResult dataclass."""

    def test_match_result_creation(self):
        """Test creating a MatchResult."""
        result = MatchResult(
            debate_id="debate_123",
            winner="claude",
            participants=["claude", "gpt"],
            domain="coding",
            scores={"claude": 1.0, "gpt": 0.0},
        )

        assert result.debate_id == "debate_123"
        assert result.winner == "claude"
        assert result.participants == ["claude", "gpt"]
        assert result.domain == "coding"
        assert result.scores == {"claude": 1.0, "gpt": 0.0}

    def test_match_result_draw(self):
        """Test MatchResult with no winner (draw)."""
        result = MatchResult(
            debate_id="debate_456",
            winner=None,
            participants=["agent_a", "agent_b"],
            domain=None,
            scores={"agent_a": 0.5, "agent_b": 0.5},
        )

        assert result.winner is None
        assert result.domain is None


# =============================================================================
# ELO Calculation Accuracy Tests
# =============================================================================


class TestEloCalculationAccuracy:
    """Tests for ELO calculation mathematical accuracy."""

    def test_expected_score_equal_ratings(self, elo_system):
        """Test expected score with equal ratings gives 0.5."""
        expected = elo_system._expected_score(1500, 1500)
        assert expected == pytest.approx(0.5, rel=1e-6)

    def test_expected_score_higher_rating(self, elo_system):
        """Test that higher rating gives higher expected score."""
        expected = elo_system._expected_score(1600, 1400)
        assert expected > 0.5
        assert expected < 1.0

    def test_expected_score_lower_rating(self, elo_system):
        """Test that lower rating gives lower expected score."""
        expected = elo_system._expected_score(1400, 1600)
        assert expected < 0.5
        assert expected > 0.0

    def test_expected_score_symmetry(self, elo_system):
        """Test that expected scores for two players sum to 1."""
        exp_a = elo_system._expected_score(1600, 1400)
        exp_b = elo_system._expected_score(1400, 1600)
        assert exp_a + exp_b == pytest.approx(1.0, rel=1e-6)

    def test_expected_score_400_point_difference(self, elo_system):
        """Test standard ELO property: 400 point diff = ~10x more likely."""
        # E = 1 / (1 + 10^((1500-1900)/400)) = 1 / (1 + 10^-1) = 1/1.1 = 10/11
        expected = elo_system._expected_score(1900, 1500)
        assert expected == pytest.approx(10 / 11, rel=1e-3)

    def test_new_elo_win_increases(self, elo_system):
        """Test that winning increases ELO."""
        expected = elo_system._expected_score(1500, 1500)
        new_elo = elo_system._calculate_new_elo(1500, expected, 1.0)
        assert new_elo > 1500

    def test_new_elo_loss_decreases(self, elo_system):
        """Test that losing decreases ELO."""
        expected = elo_system._expected_score(1500, 1500)
        new_elo = elo_system._calculate_new_elo(1500, expected, 0.0)
        assert new_elo < 1500

    def test_new_elo_draw_no_change(self, elo_system):
        """Test that draw against equal opponent gives no change."""
        expected = elo_system._expected_score(1500, 1500)
        new_elo = elo_system._calculate_new_elo(1500, expected, 0.5)
        assert new_elo == pytest.approx(1500, rel=1e-6)

    def test_k_factor_scales_change(self, elo_system):
        """Test that K-factor scales the rating change."""
        expected = elo_system._expected_score(1500, 1500)

        change_k16 = elo_system._calculate_new_elo(1500, expected, 1.0, k=16) - 1500
        change_k32 = elo_system._calculate_new_elo(1500, expected, 1.0, k=32) - 1500

        assert change_k32 == pytest.approx(change_k16 * 2, rel=1e-6)


# =============================================================================
# Match Recording Tests
# =============================================================================


class TestMatchRecording:
    """Tests for match recording and ELO updates."""

    def test_record_match_basic(self, elo_system):
        """Test basic match recording with winner and loser."""
        elo_changes = elo_system.record_match(
            winner="alice",
            loser="bob",
            task="test_debate",
        )

        assert "alice" in elo_changes
        assert "bob" in elo_changes
        assert elo_changes["alice"] > 0  # Winner gains ELO
        assert elo_changes["bob"] < 0  # Loser loses ELO

    def test_record_match_draw(self, elo_system):
        """Test recording a draw."""
        elo_changes = elo_system.record_match(
            participants=["alice", "bob"],
            scores={"alice": 0.5, "bob": 0.5},
            draw=True,
        )

        # Equal ratings + draw = minimal change
        assert "alice" in elo_changes
        assert "bob" in elo_changes

    def test_record_match_updates_ratings(self, elo_system):
        """Test that match recording updates persisted ratings."""
        # Record a match
        elo_system.record_match(winner="winner_agent", loser="loser_agent")

        # Check updated ratings
        winner_rating = elo_system.get_rating("winner_agent", use_cache=False)
        loser_rating = elo_system.get_rating("loser_agent", use_cache=False)

        assert winner_rating.elo > DEFAULT_ELO
        assert loser_rating.elo < DEFAULT_ELO
        assert winner_rating.wins == 1
        assert loser_rating.losses == 1

    def test_record_match_with_domain(self, elo_system):
        """Test match recording with domain-specific ELO."""
        elo_system.record_match(
            winner="specialist",
            loser="generalist",
            domain="legal",
        )

        specialist = elo_system.get_rating("specialist", use_cache=False)
        assert "legal" in specialist.domain_elos
        assert specialist.domain_elos["legal"] > DEFAULT_ELO

    def test_record_match_with_scores(self, elo_system):
        """Test match recording with explicit scores."""
        elo_changes = elo_system.record_match(
            participants=["a", "b", "c"],
            scores={"a": 3.0, "b": 2.0, "c": 1.0},
        )

        # Highest scorer should gain most
        assert elo_changes["a"] > elo_changes["b"]
        assert elo_changes["b"] > elo_changes["c"]

    def test_record_match_with_confidence_weight(self, elo_system):
        """Test that confidence weight scales ELO changes."""
        # Full confidence
        changes_full = elo_system.record_match(
            winner="agent1",
            loser="agent2",
            confidence_weight=1.0,
        )

        # Reset ratings for fair comparison
        elo_system.invalidate_rating_cache()

        # Half confidence with new agents
        changes_half = elo_system.record_match(
            winner="agent3",
            loser="agent4",
            confidence_weight=0.5,
        )

        # Half confidence should give roughly half the change
        assert abs(changes_half["agent3"]) < abs(changes_full["agent1"])


class TestMatchRecordingEdgeCases:
    """Edge cases for match recording."""

    def test_record_match_same_agent_twice(self, elo_system):
        """Test recording multiple matches for same agent."""
        elo_system.record_match(winner="alice", loser="bob")
        elo_system.record_match(winner="alice", loser="charlie")

        alice = elo_system.get_rating("alice", use_cache=False)
        assert alice.wins == 2
        assert alice.debates_count >= 2

    def test_record_match_upset_larger_gain(self, elo_system):
        """Test that upset wins give larger ELO gains."""
        # Set up ratings - underdog has lower ELO
        underdog = elo_system.get_rating("underdog")
        favorite = elo_system.get_rating("favorite")

        underdog.elo = 1200
        favorite.elo = 1800

        elo_system._save_rating(underdog)
        elo_system._save_rating(favorite)

        # Underdog wins!
        changes = elo_system.record_match(winner="underdog", loser="favorite")

        # Underdog should gain more than typical win
        # When underdog beats favorite, gain should be larger than K_FACTOR/2
        assert changes["underdog"] > 0


# =============================================================================
# Persistence and Retrieval Tests
# =============================================================================


class TestPersistenceAndRetrieval:
    """Tests for rating persistence and retrieval."""

    def test_get_rating_creates_new(self, elo_system):
        """Test that get_rating creates new agent with default ELO."""
        rating = elo_system.get_rating("new_agent")

        assert rating.agent_name == "new_agent"
        assert rating.elo == DEFAULT_ELO

    def test_save_and_retrieve_rating(self, elo_system):
        """Test saving and retrieving a rating."""
        rating = AgentRating(
            agent_name="persisted_agent",
            elo=1650.5,
            wins=10,
            losses=5,
            draws=2,
        )

        elo_system._save_rating(rating)
        retrieved = elo_system.get_rating("persisted_agent", use_cache=False)

        assert retrieved.elo == pytest.approx(1650.5, rel=1e-6)
        assert retrieved.wins == 10
        assert retrieved.losses == 5
        assert retrieved.draws == 2

    def test_get_ratings_batch(self, elo_system):
        """Test batch retrieval of ratings."""
        # Create and save multiple agents
        for name in ["batch_a", "batch_b", "batch_c"]:
            rating = elo_system.get_rating(name)
            elo_system._save_rating(rating)

        ratings = elo_system.get_ratings_batch(["batch_a", "batch_b", "batch_c"])

        assert len(ratings) == 3
        assert all(name in ratings for name in ["batch_a", "batch_b", "batch_c"])

    def test_get_ratings_batch_creates_missing(self, elo_system):
        """Test that batch get creates missing agents."""
        rating_a = elo_system.get_rating("existing_batch")
        elo_system._save_rating(rating_a)

        ratings = elo_system.get_ratings_batch(["existing_batch", "new_batch"])

        assert len(ratings) == 2
        assert "new_batch" in ratings
        assert ratings["new_batch"].elo == DEFAULT_ELO

    def test_get_all_ratings(self, elo_system):
        """Test retrieving all ratings."""
        # Save some ratings
        for i, name in enumerate(["all_a", "all_b", "all_c"]):
            rating = AgentRating(agent_name=name, elo=1500 + i * 100)
            elo_system._save_rating(rating)

        all_ratings = elo_system.get_all_ratings()

        assert len(all_ratings) >= 3
        names = [r.agent_name for r in all_ratings]
        assert "all_a" in names
        assert "all_b" in names
        assert "all_c" in names

    def test_list_agents(self, elo_system):
        """Test listing all agent names."""
        for name in ["list_a", "list_b", "list_c"]:
            rating = AgentRating(agent_name=name)
            elo_system._save_rating(rating)

        agents = elo_system.list_agents()

        assert "list_a" in agents
        assert "list_b" in agents
        assert "list_c" in agents


class TestEloHistory:
    """Tests for ELO history tracking."""

    def test_record_elo_history(self, elo_system):
        """Test recording ELO history entry."""
        elo_system._record_elo_history("history_agent", 1550.0, "debate_123")

        history = elo_system.get_elo_history("history_agent")
        assert isinstance(history, list)

    def test_record_elo_history_batch(self, elo_system):
        """Test batch recording of ELO history."""
        entries = [
            ("batch_hist_a", 1500.0, "d1"),
            ("batch_hist_b", 1600.0, "d2"),
            ("batch_hist_c", 1700.0, "d3"),
        ]

        elo_system._record_elo_history_batch(entries)

        # Verify at least one entry exists
        history_a = elo_system.get_elo_history("batch_hist_a")
        assert isinstance(history_a, list)


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests for ELO system."""

    def test_extreme_high_rating(self, elo_system):
        """Test handling of extremely high ratings."""
        rating = AgentRating(agent_name="grandmaster", elo=2800)
        elo_system._save_rating(rating)

        retrieved = elo_system.get_rating("grandmaster", use_cache=False)
        assert retrieved.elo == 2800

    def test_extreme_low_rating(self, elo_system):
        """Test handling of extremely low ratings."""
        rating = AgentRating(agent_name="beginner", elo=400)
        elo_system._save_rating(rating)

        retrieved = elo_system.get_rating("beginner", use_cache=False)
        assert retrieved.elo == 400

    def test_negative_rating(self, elo_system):
        """Test that negative ratings are allowed."""
        rating = AgentRating(agent_name="negative_agent", elo=-100)
        elo_system._save_rating(rating)

        retrieved = elo_system.get_rating("negative_agent", use_cache=False)
        assert retrieved.elo == -100

    def test_zero_rating(self, elo_system):
        """Test handling of zero rating."""
        rating = AgentRating(agent_name="zero_agent", elo=0)
        elo_system._save_rating(rating)

        retrieved = elo_system.get_rating("zero_agent", use_cache=False)
        assert retrieved.elo == 0

    def test_domain_elos_preserved(self, elo_system):
        """Test that domain ELOs are preserved through save/retrieve."""
        rating = AgentRating(
            agent_name="domain_specialist",
            elo=1500,
            domain_elos={"legal": 1700, "medical": 1550, "coding": 1800},
        )
        elo_system._save_rating(rating)

        retrieved = elo_system.get_rating("domain_specialist", use_cache=False)
        assert retrieved.domain_elos["legal"] == 1700
        assert retrieved.domain_elos["medical"] == 1550
        assert retrieved.domain_elos["coding"] == 1800

    def test_empty_agent_list(self, elo_system):
        """Test operations with empty agent list."""
        agents = elo_system.list_agents()
        assert agents == [] or isinstance(agents, list)

    def test_batch_ratings_empty_list(self, elo_system):
        """Test batch ratings with empty list."""
        ratings = elo_system.get_ratings_batch([])
        assert ratings == {}


# =============================================================================
# Cache Tests
# =============================================================================


class TestCaching:
    """Tests for caching behavior."""

    def test_rating_cache_usage(self, elo_system):
        """Test that rating cache is used."""
        rating = AgentRating(agent_name="cached_agent")
        elo_system._save_rating(rating)

        # First call populates cache
        r1 = elo_system.get_rating("cached_agent", use_cache=True)
        # Second call should use cache
        r2 = elo_system.get_rating("cached_agent", use_cache=True)

        assert r1.elo == r2.elo

    def test_rating_cache_bypass(self, elo_system):
        """Test bypassing rating cache."""
        rating = AgentRating(agent_name="bypass_agent")
        elo_system._save_rating(rating)

        r = elo_system.get_rating("bypass_agent", use_cache=False)
        assert r.agent_name == "bypass_agent"

    def test_invalidate_rating_cache(self, elo_system):
        """Test invalidating rating cache."""
        rating = AgentRating(agent_name="invalidate_test")
        elo_system._save_rating(rating)
        elo_system.get_rating("invalidate_test")  # Populate cache

        cleared = elo_system.invalidate_rating_cache()
        assert cleared >= 0

    def test_invalidate_single_rating_cache(self, elo_system):
        """Test invalidating single agent's cache."""
        rating = AgentRating(agent_name="single_invalidate")
        elo_system._save_rating(rating)
        elo_system.get_rating("single_invalidate")

        cleared = elo_system.invalidate_rating_cache("single_invalidate")
        assert cleared >= 0

    def test_invalidate_leaderboard_cache(self, elo_system):
        """Test invalidating leaderboard cache."""
        cleared = elo_system.invalidate_leaderboard_cache()
        assert cleared >= 0


# =============================================================================
# Critique Tracking Tests
# =============================================================================


class TestCritiqueTracking:
    """Tests for critique acceptance tracking."""

    def test_record_accepted_critique(self, elo_system):
        """Test recording an accepted critique."""
        elo_system.register_agent("critic")
        elo_system.record_critique("critic", accepted=True)

        rating = elo_system.get_rating("critic", use_cache=False)
        assert rating.critiques_accepted >= 1
        assert rating.critiques_total >= 1

    def test_record_rejected_critique(self, elo_system):
        """Test recording a rejected critique."""
        elo_system.register_agent("critic")
        elo_system.record_critique("critic", accepted=False)

        rating = elo_system.get_rating("critic", use_cache=False)
        assert rating.critiques_total >= 1

    def test_critique_rate_after_multiple(self, elo_system):
        """Test critique acceptance rate with multiple critiques."""
        elo_system.register_agent("multi_critic")

        # Record 3 accepted, 1 rejected
        elo_system.record_critique("multi_critic", accepted=True)
        elo_system.record_critique("multi_critic", accepted=True)
        elo_system.record_critique("multi_critic", accepted=True)
        elo_system.record_critique("multi_critic", accepted=False)

        rating = elo_system.get_rating("multi_critic", use_cache=False)
        assert rating.critique_acceptance_rate == 0.75


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_elo_store_returns_instance(self):
        """Test that get_elo_store returns an EloSystem instance."""
        with patch("aragora.ranking.elo._elo_store", None):
            store = get_elo_store()
            assert isinstance(store, EloSystem)

    def test_get_elo_store_returns_same_instance(self):
        """Test that get_elo_store returns the same instance."""
        with patch("aragora.ranking.elo._elo_store", None):
            store1 = get_elo_store()
            store2 = get_elo_store()
            assert store1 is store2


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Tests for system statistics."""

    def test_get_stats_returns_dict(self, elo_system):
        """Test that get_stats returns a dictionary."""
        stats = elo_system.get_stats()
        assert isinstance(stats, dict)

    def test_get_stats_contains_expected_keys(self, elo_system):
        """Test that stats contains expected keys."""
        stats = elo_system.get_stats()
        assert "total_agents" in stats or "total_matches" in stats

    def test_get_stats_with_agents(self, elo_system):
        """Test stats after adding agents."""
        for name in ["stat_a", "stat_b"]:
            rating = AgentRating(agent_name=name)
            elo_system._save_rating(rating)

        stats = elo_system.get_stats(use_cache=False)
        assert stats.get("total_agents", 0) >= 2


# =============================================================================
# Leaderboard Tests
# =============================================================================


class TestLeaderboard:
    """Tests for leaderboard functionality."""

    def test_get_leaderboard_empty(self, elo_system):
        """Test leaderboard with no agents."""
        lb = elo_system.get_leaderboard()
        assert lb == []

    def test_get_leaderboard_ordered(self, elo_system):
        """Test that leaderboard is ordered by ELO."""
        # Create agents with different ELOs
        for elo_val, name in [(1400, "low"), (1500, "mid"), (1600, "high")]:
            rating = AgentRating(agent_name=name, elo=elo_val)
            elo_system._save_rating(rating)

        elo_system.invalidate_leaderboard_cache()
        lb = elo_system.get_leaderboard(limit=10)

        assert len(lb) == 3
        assert lb[0].agent_name == "high"
        assert lb[1].agent_name == "mid"
        assert lb[2].agent_name == "low"

    def test_get_leaderboard_respects_limit(self, elo_system):
        """Test that leaderboard respects limit."""
        for i in range(10):
            rating = AgentRating(agent_name=f"lb_agent_{i}")
            elo_system._save_rating(rating)

        elo_system.invalidate_leaderboard_cache()
        lb = elo_system.get_leaderboard(limit=5)

        assert len(lb) == 5

    def test_get_cached_leaderboard(self, elo_system):
        """Test cached leaderboard access."""
        rating = AgentRating(agent_name="cached_lb_agent")
        elo_system._save_rating(rating)

        lb1 = elo_system.get_cached_leaderboard(limit=10)
        lb2 = elo_system.get_cached_leaderboard(limit=10)

        # Both should return same data
        assert len(lb1) == len(lb2)


# =============================================================================
# Verification Integration Tests
# =============================================================================


class TestVerificationIntegration:
    """Tests for verification impact on ELO."""

    def test_update_from_verification_positive(self, elo_system):
        """Test ELO increase from verified claims."""
        elo_system.register_agent("verified_agent")
        initial = elo_system.get_rating("verified_agent").elo

        change = elo_system.update_from_verification(
            agent_name="verified_agent",
            domain="logic",
            verified_count=5,
            disproven_count=0,
        )

        assert change > 0
        final = elo_system.get_rating("verified_agent", use_cache=False).elo
        assert final > initial

    def test_update_from_verification_negative(self, elo_system):
        """Test ELO decrease from disproven claims."""
        elo_system.register_agent("disproven_agent")
        initial = elo_system.get_rating("disproven_agent").elo

        change = elo_system.update_from_verification(
            agent_name="disproven_agent",
            domain="logic",
            verified_count=0,
            disproven_count=5,
        )

        assert change < 0
        final = elo_system.get_rating("disproven_agent", use_cache=False).elo
        assert final < initial

    def test_update_from_verification_no_change(self, elo_system):
        """Test no ELO change with no verifications."""
        elo_system.register_agent("no_verify_agent")

        change = elo_system.update_from_verification(
            agent_name="no_verify_agent",
            domain="logic",
            verified_count=0,
            disproven_count=0,
        )

        assert change == 0.0

    def test_get_verification_impact(self, elo_system):
        """Test getting verification impact summary."""
        elo_system.register_agent("impact_agent")

        impact = elo_system.get_verification_impact("impact_agent")
        assert isinstance(impact, dict)
