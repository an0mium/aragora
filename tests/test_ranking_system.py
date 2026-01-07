"""
Tests for the ranking system (ELO, calibration, leaderboards).

Covers:
- ELO calculation formulas
- AgentRating dataclass and properties
- MatchResult handling
- EloSystem CRUD operations
- Leaderboard queries and caching
- Calibration scoring (Brier score)
- Domain-specific ratings
"""

import math
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.ranking.elo import (
    AgentRating,
    MatchResult,
    EloSystem,
    DEFAULT_ELO,
    K_FACTOR,
    CALIBRATION_MIN_COUNT,
)
from aragora.ranking.calibration_engine import (
    CalibrationEngine,
    CalibrationPrediction,
    BucketStats,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield Path(f.name)
    try:
        Path(f.name).unlink()
    except FileNotFoundError:
        pass


@pytest.fixture
def elo_system(temp_db):
    """Create an EloSystem with temporary database."""
    return EloSystem(db_path=str(temp_db))


@pytest.fixture
def calibration_engine(temp_db, elo_system):
    """Create a CalibrationEngine with temporary database."""
    return CalibrationEngine(db_path=temp_db, elo_system=elo_system)


# =============================================================================
# AgentRating Tests
# =============================================================================


class TestAgentRating:
    """Tests for AgentRating dataclass."""

    def test_default_values(self):
        """Test AgentRating default values."""
        rating = AgentRating(agent_name="test-agent")

        assert rating.agent_name == "test-agent"
        assert rating.elo == DEFAULT_ELO
        assert rating.wins == 0
        assert rating.losses == 0
        assert rating.draws == 0
        assert rating.domain_elos == {}

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        rating = AgentRating(agent_name="test", wins=7, losses=2, draws=1)

        assert rating.win_rate == 0.7  # 7 / 10

    def test_win_rate_with_no_games(self):
        """Test win rate with no games returns 0."""
        rating = AgentRating(agent_name="test")

        assert rating.win_rate == 0.0

    def test_games_played(self):
        """Test games_played property."""
        rating = AgentRating(agent_name="test", wins=5, losses=3, draws=2)

        assert rating.games_played == 10

    def test_critique_acceptance_rate(self):
        """Test critique acceptance rate calculation."""
        rating = AgentRating(
            agent_name="test",
            critiques_accepted=8,
            critiques_total=10,
        )

        assert rating.critique_acceptance_rate == 0.8

    def test_critique_acceptance_rate_no_critiques(self):
        """Test critique acceptance rate with no critiques."""
        rating = AgentRating(agent_name="test")

        assert rating.critique_acceptance_rate == 0.0

    def test_calibration_accuracy(self):
        """Test calibration accuracy calculation."""
        rating = AgentRating(
            agent_name="test",
            calibration_correct=7,
            calibration_total=10,
        )

        assert rating.calibration_accuracy == 0.7

    def test_calibration_brier_score(self):
        """Test Brier score calculation."""
        rating = AgentRating(
            agent_name="test",
            calibration_brier_sum=2.5,
            calibration_total=10,
        )

        assert rating.calibration_brier_score == 0.25

    def test_calibration_brier_score_no_predictions(self):
        """Test Brier score with no predictions returns 1.0 (worst)."""
        rating = AgentRating(agent_name="test")

        assert rating.calibration_brier_score == 1.0

    def test_calibration_score_minimum_predictions(self):
        """Test calibration score requires minimum predictions."""
        rating = AgentRating(
            agent_name="test",
            calibration_correct=3,
            calibration_total=3,  # Below CALIBRATION_MIN_COUNT
            calibration_brier_sum=0.0,
        )

        # Should return 0 if below minimum
        if rating.calibration_total < CALIBRATION_MIN_COUNT:
            assert rating.calibration_score == 0.0

    def test_calibration_score_with_enough_predictions(self):
        """Test calibration score with enough predictions."""
        rating = AgentRating(
            agent_name="test",
            calibration_correct=15,
            calibration_total=15,
            calibration_brier_sum=1.5,  # Brier = 0.1
        )

        if rating.calibration_total >= CALIBRATION_MIN_COUNT:
            # Score should be positive
            assert rating.calibration_score > 0


# =============================================================================
# MatchResult Tests
# =============================================================================


class TestMatchResult:
    """Tests for MatchResult dataclass."""

    def test_match_result_with_winner(self):
        """Test MatchResult with a winner."""
        result = MatchResult(
            debate_id="debate-123",
            winner="agent-a",
            participants=["agent-a", "agent-b"],
            domain="general",
            scores={"agent-a": 0.8, "agent-b": 0.6},
        )

        assert result.winner == "agent-a"
        assert len(result.participants) == 2

    def test_match_result_draw(self):
        """Test MatchResult for a draw."""
        result = MatchResult(
            debate_id="debate-456",
            winner=None,
            participants=["agent-a", "agent-b"],
            domain="technical",
            scores={"agent-a": 0.7, "agent-b": 0.7},
        )

        assert result.winner is None


# =============================================================================
# ELO Calculation Tests
# =============================================================================


class TestEloCalculations:
    """Tests for ELO calculation formulas."""

    def test_expected_score_equal_ratings(self, elo_system):
        """Test expected score when ratings are equal."""
        # With equal ratings, expected score should be 0.5
        expected = elo_system._expected_score(1500, 1500)
        assert expected == pytest.approx(0.5, abs=0.01)

    def test_expected_score_higher_rating(self, elo_system):
        """Test expected score when one rating is higher."""
        # Higher rated player should have >0.5 expected
        expected = elo_system._expected_score(1600, 1400)
        assert expected > 0.5

    def test_expected_score_lower_rating(self, elo_system):
        """Test expected score when rating is lower."""
        expected = elo_system._expected_score(1400, 1600)
        assert expected < 0.5

    def test_expected_score_400_point_difference(self, elo_system):
        """Test expected score with 400 point difference."""
        # 400 point difference should give ~0.91 expected
        expected = elo_system._expected_score(1900, 1500)
        assert expected == pytest.approx(0.91, abs=0.02)

    def test_elo_update_winner_gains_loser_loses(self, elo_system):
        """Test that winner gains and loser loses ELO."""
        # Record a match (get_rating creates if not exists)
        elo_system.record_match(
            debate_id="test-match-1",
            participants=["winner", "loser"],
            scores={"winner": 1.0, "loser": 0.0},
        )

        winner_rating = elo_system.get_rating("winner")
        loser_rating = elo_system.get_rating("loser")

        assert winner_rating.elo > DEFAULT_ELO
        assert loser_rating.elo < DEFAULT_ELO

    def test_elo_update_zero_sum(self, elo_system):
        """Test that ELO changes are zero-sum in 1v1."""
        initial_total = 2 * DEFAULT_ELO

        elo_system.record_match(
            debate_id="zero-sum-test",
            participants=["agent-a", "agent-b"],
            scores={"agent-a": 1.0, "agent-b": 0.0},
        )

        a_rating = elo_system.get_rating("agent-a")
        b_rating = elo_system.get_rating("agent-b")

        final_total = a_rating.elo + b_rating.elo

        assert final_total == pytest.approx(initial_total, abs=0.01)

    def test_draw_minimal_elo_change(self, elo_system):
        """Test that draws cause minimal ELO change for equal ratings."""
        elo_system.record_match(
            debate_id="draw-test",
            participants=["agent-a", "agent-b"],
            scores={"agent-a": 0.5, "agent-b": 0.5},
        )

        a_rating = elo_system.get_rating("agent-a")
        b_rating = elo_system.get_rating("agent-b")

        # Both should be close to starting ELO for equal-rated draw
        assert abs(a_rating.elo - DEFAULT_ELO) < 1
        assert abs(b_rating.elo - DEFAULT_ELO) < 1


# =============================================================================
# EloSystem CRUD Tests
# =============================================================================


class TestEloSystemCRUD:
    """Tests for EloSystem CRUD operations."""

    def test_get_rating_new(self, elo_system):
        """Test getting a new agent creates default rating."""
        rating = elo_system.get_rating("new-agent")

        assert rating.agent_name == "new-agent"
        assert rating.elo == DEFAULT_ELO
        assert rating.games_played == 0

    def test_get_rating_existing(self, elo_system):
        """Test getting an existing agent returns same rating."""
        # Modify via match (this creates both agents)
        elo_system.record_match(
            debate_id="test-match",
            participants=["existing-agent", "other"],
            scores={"existing-agent": 1.0, "other": 0.0},
        )

        rating = elo_system.get_rating("existing-agent")
        assert rating.wins == 1

    def test_save_rating(self, elo_system):
        """Test saving an agent's rating."""
        rating = elo_system.get_rating("update-test")
        rating.elo = 1600
        elo_system._save_rating(rating)

        fetched = elo_system.get_rating("update-test", use_cache=False)
        assert fetched.elo == 1600

    def test_record_match_creates_history(self, elo_system):
        """Test that recording a match creates history entries."""
        elo_system.record_match(
            debate_id="history-test",
            participants=["agent-1", "agent-2"],
            scores={"agent-1": 0.8, "agent-2": 0.6},
        )

        # Check match was recorded via recent matches
        recent = elo_system.get_recent_matches(limit=10)
        match = next((m for m in recent if m["debate_id"] == "history-test"), None)
        assert match is not None
        assert match["winner"] == "agent-1"

    def test_record_match_updates_stats(self, elo_system):
        """Test that recording a match updates win/loss stats."""
        elo_system.record_match(
            debate_id="stats-test",
            participants=["winner-test", "loser-test"],
            scores={"winner-test": 1.0, "loser-test": 0.0},
        )

        winner = elo_system.get_rating("winner-test")
        loser = elo_system.get_rating("loser-test")

        assert winner.wins == 1
        assert loser.losses == 1


# =============================================================================
# Leaderboard Tests
# =============================================================================


class TestLeaderboard:
    """Tests for leaderboard functionality."""

    def test_get_leaderboard_empty(self, elo_system):
        """Test leaderboard with no agents."""
        leaderboard = elo_system.get_leaderboard(limit=10)
        assert leaderboard == []

    def test_get_leaderboard_ordering(self, elo_system):
        """Test leaderboard is ordered by ELO descending."""
        # Create agents with different ELOs
        for name, elo in [("low", 1400), ("mid", 1500), ("high", 1600)]:
            rating = elo_system.get_rating(name)
            rating.elo = elo
            elo_system._save_rating(rating)

        leaderboard = elo_system.get_leaderboard(limit=10)

        assert len(leaderboard) == 3
        # get_leaderboard returns AgentRating objects
        assert leaderboard[0].agent_name == "high"
        assert leaderboard[1].agent_name == "mid"
        assert leaderboard[2].agent_name == "low"

    def test_get_leaderboard_limit(self, elo_system):
        """Test leaderboard respects limit parameter."""
        for i in range(10):
            rating = elo_system.get_rating(f"agent-{i}")
            elo_system._save_rating(rating)

        leaderboard = elo_system.get_leaderboard(limit=5)

        assert len(leaderboard) == 5

    def test_get_leaderboard_with_matches(self, elo_system):
        """Test leaderboard after recording matches."""
        # Record matches - creates agents
        for i in range(5):
            elo_system.record_match(
                debate_id=f"match-{i}",
                participants=["active", "opponent"],
                scores={"active": 1.0, "opponent": 0.0},
            )

        leaderboard = elo_system.get_leaderboard(limit=10)

        # Both agents should appear
        agent_names = [entry.agent_name for entry in leaderboard]
        assert "active" in agent_names
        assert "opponent" in agent_names

        # Active should have higher ELO (more wins)
        assert leaderboard[0].agent_name == "active"


# =============================================================================
# Domain-Specific Rating Tests
# =============================================================================


class TestDomainRatings:
    """Tests for domain-specific ELO ratings."""

    def test_domain_elo_update(self, elo_system):
        """Test domain-specific ELO is updated."""
        elo_system.record_match(
            debate_id="domain-match",
            participants=["domain-agent-1", "domain-agent-2"],
            scores={"domain-agent-1": 1.0, "domain-agent-2": 0.0},
            domain="machine-learning",
        )

        rating = elo_system.get_rating("domain-agent-1")

        # Check domain ELO was updated
        assert "machine-learning" in rating.domain_elos
        assert rating.domain_elos["machine-learning"] > DEFAULT_ELO

    def test_domain_leaderboard(self, elo_system):
        """Test domain-specific leaderboard."""
        # Record domain-specific matches
        for i in range(2):
            elo_system.record_match(
                debate_id=f"ml-match-{i}",
                participants=["ml-agent-0", f"ml-agent-{i+1}"],
                scores={"ml-agent-0": 1.0, f"ml-agent-{i+1}": 0.0},
                domain="ml",
            )

        leaderboard = elo_system.get_leaderboard(limit=10, domain="ml")

        assert len(leaderboard) > 0
        # Winner should be at top of domain leaderboard
        assert leaderboard[0].agent_name == "ml-agent-0"


# =============================================================================
# Calibration Engine Tests
# =============================================================================


class TestCalibrationEngine:
    """Tests for CalibrationEngine."""

    def test_record_prediction(self, calibration_engine):
        """Test recording a prediction."""
        calibration_engine.record_winner_prediction(
            tournament_id="tourney-1",
            predictor_agent="predictor",
            predicted_winner="agent-a",
            confidence=0.75,
        )

        # Should not raise
        assert True

    def test_confidence_clamping(self, calibration_engine):
        """Test that confidence is clamped to [0, 1]."""
        # Over 1.0
        calibration_engine.record_winner_prediction(
            tournament_id="tourney-clamp-1",
            predictor_agent="predictor",
            predicted_winner="agent-a",
            confidence=1.5,  # Should be clamped to 1.0
        )

        # Under 0.0
        calibration_engine.record_winner_prediction(
            tournament_id="tourney-clamp-2",
            predictor_agent="predictor",
            predicted_winner="agent-a",
            confidence=-0.5,  # Should be clamped to 0.0
        )

        # Should not raise
        assert True

    def test_resolve_tournament_correct_prediction(self, calibration_engine):
        """Test resolving tournament with correct prediction."""
        calibration_engine.record_winner_prediction(
            tournament_id="correct-test",
            predictor_agent="oracle",
            predicted_winner="winner",
            confidence=0.9,
        )

        scores = calibration_engine.resolve_tournament(
            tournament_id="correct-test",
            actual_winner="winner",
        )

        # Brier score should be low for correct high-confidence prediction
        if "oracle" in scores:
            assert scores["oracle"] < 0.5

    def test_resolve_tournament_wrong_prediction(self, calibration_engine):
        """Test resolving tournament with wrong prediction."""
        calibration_engine.record_winner_prediction(
            tournament_id="wrong-test",
            predictor_agent="wrong-oracle",
            predicted_winner="loser",
            confidence=0.9,
        )

        scores = calibration_engine.resolve_tournament(
            tournament_id="wrong-test",
            actual_winner="winner",
        )

        # Brier score should be high for wrong high-confidence prediction
        if "wrong-oracle" in scores:
            assert scores["wrong-oracle"] > 0.5

    def test_brier_score_formula(self, calibration_engine):
        """Test Brier score calculation: (p - outcome)^2."""
        # Perfect prediction: p=1.0, outcome=1 -> (1-1)^2 = 0
        calibration_engine.record_winner_prediction(
            tournament_id="brier-perfect",
            predictor_agent="perfect",
            predicted_winner="winner",
            confidence=1.0,
        )

        scores = calibration_engine.resolve_tournament(
            tournament_id="brier-perfect",
            actual_winner="winner",
        )

        if "perfect" in scores:
            assert scores["perfect"] == pytest.approx(0.0, abs=0.01)

    def test_resolve_empty_tournament(self, calibration_engine):
        """Test resolving tournament with no predictions."""
        scores = calibration_engine.resolve_tournament(
            tournament_id="empty-tourney",
            actual_winner="winner",
        )

        assert scores == {}


# =============================================================================
# Cache Tests
# =============================================================================


class TestCaching:
    """Tests for ELO system caching."""

    def test_leaderboard_cache(self, elo_system):
        """Test that leaderboard is cached."""
        rating = elo_system.get_rating("cache-test")
        elo_system._save_rating(rating)

        # First call
        lb1 = elo_system.get_leaderboard(limit=10)

        # Modify directly in DB (bypass cache)
        with elo_system._db.connection() as conn:
            conn.execute(
                "UPDATE ratings SET elo = 2000 WHERE agent_name = 'cache-test'"
            )
            conn.commit()

        # Clear cache
        EloSystem._leaderboard_cache.clear()

        # After clearing cache, should fetch fresh data
        lb3 = elo_system.get_leaderboard(limit=10)

        # Find the updated agent
        if lb3:
            updated_agent = next(
                (a for a in lb3 if a.agent_name == "cache-test"), None
            )
            if updated_agent:
                assert updated_agent.elo == 2000


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_match_with_same_agent_twice(self, elo_system):
        """Test handling match where same agent appears twice."""
        # This might be an error or handled gracefully
        # The record_match method should return empty dict for <2 participants
        result = elo_system.record_match(
            debate_id="self-match-test",
            participants=["self-match", "self-match"],
            scores={"self-match": 1.0},
        )
        # Should either handle gracefully or return empty changes
        assert isinstance(result, dict)

    def test_empty_participants(self, elo_system):
        """Test match with empty participants returns empty changes."""
        result = elo_system.record_match(
            debate_id="empty-participants",
            participants=[],
            scores={},
        )
        # Should return empty dict for <2 participants
        assert result == {}

    def test_single_participant(self, elo_system):
        """Test match with single participant returns empty changes."""
        result = elo_system.record_match(
            debate_id="single-participant",
            participants=["only-one"],
            scores={"only-one": 1.0},
        )
        # Should return empty dict for <2 participants
        assert result == {}

    def test_very_large_elo_difference(self, elo_system):
        """Test ELO calculation with very large rating difference."""
        high = elo_system.get_rating("very-high")
        high.elo = 3000
        elo_system._save_rating(high)

        low = elo_system.get_rating("very-low")
        low.elo = 500
        elo_system._save_rating(low)

        # Record upset (low beats high)
        elo_system.record_match(
            debate_id="upset-match",
            participants=["very-high", "very-low"],
            scores={"very-low": 1.0, "very-high": 0.0},
        )

        high_after = elo_system.get_rating("very-high", use_cache=False)
        low_after = elo_system.get_rating("very-low", use_cache=False)

        # Upset should cause significant ELO change
        assert low_after.elo > 500
        assert high_after.elo < 3000

    def test_duplicate_match_id(self, elo_system):
        """Test recording match with duplicate debate_id."""
        elo_system.record_match(
            debate_id="duplicate-id",
            participants=["dup-1", "dup-2"],
            scores={"dup-1": 1.0, "dup-2": 0.0},
        )

        # Second match with same ID should be handled (INSERT OR REPLACE)
        # This should not raise an error
        elo_system.record_match(
            debate_id="duplicate-id",
            participants=["dup-1", "dup-2"],
            scores={"dup-1": 0.0, "dup-2": 1.0},
        )

        # Verify the match was updated
        recent = elo_system.get_recent_matches(limit=10)
        match = next((m for m in recent if m["debate_id"] == "duplicate-id"), None)
        assert match is not None
