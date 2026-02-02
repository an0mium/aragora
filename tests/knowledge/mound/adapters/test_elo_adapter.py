"""Tests for elo_adapter backward-compatibility shim and ELO functionality."""

import warnings
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from aragora.knowledge.mound.adapters import PerformanceAdapter


# =============================================================================
# Helper functions for creating mock objects
# =============================================================================


def _make_mock_rating(
    agent_name: str = "claude",
    elo: float = 1500.0,
    wins: int = 10,
    losses: int = 5,
    draws: int = 2,
    domain_elos: dict | None = None,
    debates_count: int = 17,
    games_played: int = 17,
    critiques_accepted: int = 8,
    critiques_total: int = 12,
    critique_acceptance_rate: float = 0.667,
    calibration_correct: int = 7,
    calibration_total: int = 10,
    calibration_accuracy: float = 0.7,
    updated_at: str = "2025-01-15T10:00:00+00:00",
) -> MagicMock:
    """Create a mock AgentRating."""
    rating = MagicMock()
    rating.agent_name = agent_name
    rating.elo = elo
    rating.domain_elos = domain_elos or {"security": 1600, "coding": 1450}
    rating.wins = wins
    rating.losses = losses
    rating.draws = draws
    rating.debates_count = debates_count
    rating.win_rate = wins / max(1, wins + losses + draws)
    rating.games_played = games_played
    rating.critiques_accepted = critiques_accepted
    rating.critiques_total = critiques_total
    rating.critique_acceptance_rate = critique_acceptance_rate
    rating.calibration_correct = calibration_correct
    rating.calibration_total = calibration_total
    rating.calibration_accuracy = calibration_accuracy
    rating.updated_at = updated_at
    return rating


def _make_mock_match(
    debate_id: str = "d-123",
    winner: str = "claude",
    participants: list | None = None,
    domain: str = "security",
    scores: dict | None = None,
    created_at: str = "2025-01-15T10:00:00+00:00",
) -> MagicMock:
    """Create a mock MatchResult."""
    match = MagicMock()
    match.debate_id = debate_id
    match.winner = winner
    match.participants = participants or ["claude", "gpt4"]
    match.domain = domain
    match.scores = scores or {"claude": 0.8, "gpt4": 0.6}
    match.created_at = created_at
    return match


def _make_mock_relationship(
    agent_a: str = "claude",
    agent_b: str = "gpt4",
    debates_together: int = 10,
    a_wins_vs_b: int = 6,
    b_wins_vs_a: int = 3,
    draws: int = 1,
) -> MagicMock:
    """Create a mock RelationshipMetrics."""
    metrics = MagicMock()
    metrics.agent_a = agent_a
    metrics.agent_b = agent_b
    metrics.debates_together = debates_together
    metrics.a_wins_vs_b = a_wins_vs_b
    metrics.b_wins_vs_a = b_wins_vs_a
    metrics.draws = draws
    metrics.avg_elo_diff = 50.0
    metrics.synergy_score = 0.7
    return metrics


# =============================================================================
# Deprecation Tests
# =============================================================================


class TestEloAdapterDeprecation:
    """Verify the deprecated elo_adapter module works and warns."""

    def test_import_issues_deprecation_warning(self):
        """Importing elo_adapter should issue a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Force a fresh import by removing from cache
            import sys

            sys.modules.pop("aragora.knowledge.mound.adapters.elo_adapter", None)

            import aragora.knowledge.mound.adapters.elo_adapter  # noqa: F401

            # Check that a DeprecationWarning was issued
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            assert "deprecated" in str(dep_warnings[0].message).lower()

    def test_exports_elo_adapter_class(self):
        """Should re-export EloAdapter (alias for PerformanceAdapter)."""
        from aragora.knowledge.mound.adapters.elo_adapter import EloAdapter
        from aragora.knowledge.mound.adapters.performance_adapter import (
            EloAdapter as OriginalEloAdapter,
        )

        assert EloAdapter is OriginalEloAdapter

    def test_exports_performance_adapter_class(self):
        """Should re-export PerformanceAdapter."""
        from aragora.knowledge.mound.adapters.elo_adapter import PerformanceAdapter
        from aragora.knowledge.mound.adapters.performance_adapter import (
            PerformanceAdapter as OriginalPerformanceAdapter,
        )

        assert PerformanceAdapter is OriginalPerformanceAdapter

    def test_exports_all_types(self):
        """Should re-export all support types."""
        from aragora.knowledge.mound.adapters.elo_adapter import (
            RatingSearchResult,
            KMEloPattern,
            EloAdjustmentRecommendation,
            EloSyncResult,
        )

        # Verify they are accessible
        assert RatingSearchResult is not None
        assert KMEloPattern is not None
        assert EloAdjustmentRecommendation is not None
        assert EloSyncResult is not None

    def test_all_list(self):
        """__all__ should list all re-exported names."""
        from aragora.knowledge.mound.adapters import elo_adapter

        expected = {
            "EloAdapter",
            "PerformanceAdapter",
            "RatingSearchResult",
            "KMEloPattern",
            "EloAdjustmentRecommendation",
            "EloSyncResult",
        }
        assert set(elo_adapter.__all__) == expected


# =============================================================================
# Batch Match Operations Tests
# =============================================================================


class TestBatchMatchOperations:
    """Tests for batch match storage and retrieval."""

    def test_store_multiple_matches(self):
        """Should store multiple matches independently."""
        adapter = PerformanceAdapter(enable_resilience=False)

        matches = [
            _make_mock_match(debate_id="d-1", winner="claude"),
            _make_mock_match(debate_id="d-2", winner="gpt4"),
            _make_mock_match(debate_id="d-3", winner="claude"),
        ]

        match_ids = []
        for match in matches:
            match_id = adapter.store_match(match)
            match_ids.append(match_id)

        assert len(match_ids) == 3
        assert all(mid.startswith("el_match_") for mid in match_ids)
        assert len(adapter._matches) == 3

    def test_batch_matches_with_different_participants(self):
        """Should correctly index matches with different participants."""
        adapter = PerformanceAdapter(enable_resilience=False)

        adapter.store_match(_make_mock_match(debate_id="d-1", participants=["claude", "gpt4"]))
        adapter.store_match(_make_mock_match(debate_id="d-2", participants=["claude", "gemini"]))
        adapter.store_match(_make_mock_match(debate_id="d-3", participants=["gpt4", "gemini"]))

        # Check indices
        assert len(adapter.get_agent_matches("claude")) == 2
        assert len(adapter.get_agent_matches("gpt4")) == 2
        assert len(adapter.get_agent_matches("gemini")) == 2

    def test_batch_matches_preserve_order(self):
        """Should preserve chronological order in retrieval."""
        adapter = PerformanceAdapter(enable_resilience=False)

        for i in range(5):
            match = _make_mock_match(
                debate_id=f"d-{i}",
                created_at=f"2025-01-{15 + i:02d}T10:00:00+00:00",
            )
            adapter.store_match(match)

        matches = adapter.get_agent_matches("claude", limit=10)

        # Should be sorted by created_at descending (newest first)
        for i in range(len(matches) - 1):
            assert matches[i]["created_at"] >= matches[i + 1]["created_at"]

    def test_batch_matches_with_limit(self):
        """Should respect limit when retrieving batch matches."""
        adapter = PerformanceAdapter(enable_resilience=False)

        for i in range(10):
            adapter.store_match(_make_mock_match(debate_id=f"d-{i}"))

        matches = adapter.get_agent_matches("claude", limit=3)

        assert len(matches) == 3


# =============================================================================
# ELO Calculation Edge Cases Tests
# =============================================================================


class TestEloCalculationEdgeCases:
    """Tests for ELO calculation edge cases."""

    def test_draw_result_storage(self):
        """Should correctly store draw results (winner=None)."""
        adapter = PerformanceAdapter(enable_resilience=False)

        # Store ratings for a draw
        rating1 = _make_mock_rating(
            agent_name="claude",
            elo=1500,
            wins=10,
            losses=5,
            draws=3,  # Had 2 draws, now 3
        )
        rating2 = _make_mock_rating(
            agent_name="gpt4",
            elo=1500,
            wins=8,
            losses=7,
            draws=3,
        )

        adapter.store_rating(rating1, debate_id="draw-debate", reason="draw")
        adapter.store_rating(rating2, debate_id="draw-debate", reason="draw")

        # Verify both stored correctly
        history1 = adapter.get_agent_skill_history("claude")
        history2 = adapter.get_agent_skill_history("gpt4")

        assert len(history1) == 1
        assert len(history2) == 1
        assert history1[0]["draws"] == 3
        assert history2[0]["draws"] == 3

    def test_upset_win_lower_rated_beats_higher(self):
        """Should correctly track upset when lower-rated wins."""
        adapter = PerformanceAdapter(enable_resilience=False)

        # Lower rated wins
        lower_winner = _make_mock_rating(
            agent_name="underdog",
            elo=1350,  # Was 1300, gained 50 from upset win
            wins=6,
            losses=10,
        )
        higher_loser = _make_mock_rating(
            agent_name="favorite",
            elo=1650,  # Was 1700, lost 50 to upset
            wins=15,
            losses=6,
        )

        adapter.store_rating(lower_winner, debate_id="upset-1", reason="win_vs_higher")
        adapter.store_rating(higher_loser, debate_id="upset-1", reason="loss_vs_lower")

        # Check the ratings are stored
        underdog = adapter.get_agent_skill_history("underdog")[0]
        favorite = adapter.get_agent_skill_history("favorite")[0]

        assert underdog["elo"] == 1350
        assert underdog["reason"] == "win_vs_higher"
        assert favorite["elo"] == 1650
        assert favorite["reason"] == "loss_vs_lower"

    def test_elo_boundary_conditions_minimum(self):
        """Should handle minimum ELO boundary (typically 100)."""
        adapter = PerformanceAdapter(enable_resilience=False)

        low_elo = _make_mock_rating(
            agent_name="struggling",
            elo=100,  # Minimum
            wins=1,
            losses=50,
        )

        adapter.store_rating(low_elo)

        history = adapter.get_agent_skill_history("struggling")
        assert history[0]["elo"] == 100

    def test_elo_boundary_conditions_high(self):
        """Should handle very high ELO values."""
        adapter = PerformanceAdapter(enable_resilience=False)

        high_elo = _make_mock_rating(
            agent_name="champion",
            elo=2800,  # Very high
            wins=100,
            losses=2,
        )

        adapter.store_rating(high_elo)

        history = adapter.get_agent_skill_history("champion")
        assert history[0]["elo"] == 2800

    def test_zero_games_played_new_player(self):
        """Should handle new player with zero games played."""
        adapter = PerformanceAdapter(enable_resilience=False)

        new_player = _make_mock_rating(
            agent_name="newbie",
            elo=1500,
            wins=0,
            losses=0,
            draws=0,
            debates_count=0,
            games_played=0,
        )

        rating_id = adapter.store_rating(new_player)

        assert rating_id is not None
        history = adapter.get_agent_skill_history("newbie")
        assert history[0]["games_played"] == 0
        assert history[0]["elo"] == 1500  # Default starting ELO


# =============================================================================
# New Player Uncertainty Handling Tests
# =============================================================================


class TestNewPlayerUncertaintyHandling:
    """Tests for handling new players with uncertain ratings."""

    def test_new_player_confidence_tracking(self):
        """Should track confidence for new players based on debate count."""
        adapter = PerformanceAdapter(enable_resilience=False)

        # Store expertise with low debate count
        adapter.store_agent_expertise(
            agent_name="newbie",
            domain="security",
            elo=1550,
            delta=50,
            debate_id="d-1",
        )

        expertise = adapter.get_agent_expertise("newbie", "security")

        # Confidence should be low (1/5 = 0.2)
        assert expertise["confidence"] == 0.2
        assert expertise["debate_count"] == 1

    def test_confidence_increases_with_more_debates(self):
        """Should increase confidence as debate count increases."""
        adapter = PerformanceAdapter(enable_resilience=False)

        # Simulate 5 debates
        for i in range(5):
            adapter.store_agent_expertise(
                agent_name="improving",
                domain="security",
                elo=1500 + i * 30,
                delta=30,
                debate_id=f"d-{i}",
            )

        expertise = adapter.get_agent_expertise("improving", "security")

        # After 5 debates, confidence should be 1.0
        assert expertise["confidence"] == 1.0
        assert expertise["debate_count"] == 5

    def test_new_player_not_returned_with_high_confidence_filter(self):
        """New players should be filtered out when min_confidence is high."""
        adapter = PerformanceAdapter(enable_resilience=False)

        # Add experienced player
        for _ in range(5):
            adapter.store_agent_expertise("veteran", "security", 1700, delta=30)

        # Add new player
        adapter.store_agent_expertise("rookie", "security", 1800, delta=50)

        # Query with high confidence filter
        experts = adapter.get_domain_experts("security", min_confidence=0.9)

        assert len(experts) == 1
        assert experts[0].agent_name == "veteran"

    def test_new_player_included_with_low_confidence_filter(self):
        """New players should be included when min_confidence is low."""
        adapter = PerformanceAdapter(enable_resilience=False)

        adapter.store_agent_expertise("newbie", "security", 1600, delta=50)

        experts = adapter.get_domain_experts("security", min_confidence=0.0)

        assert len(experts) == 1
        assert experts[0].agent_name == "newbie"

    def test_calibration_accuracy_for_new_player(self):
        """Should handle calibration tracking for new players."""
        adapter = PerformanceAdapter(enable_resilience=False)

        # First calibration for new player
        adapter.store_calibration(
            agent_name="newbie",
            debate_id="d-1",
            predicted_winner="claude",
            predicted_confidence=0.5,  # Low confidence for new player
            actual_winner="claude",
            was_correct=True,
            brier_score=0.25,
        )

        history = adapter.get_agent_calibration_history("newbie")

        assert len(history) == 1
        assert history[0]["predicted_confidence"] == 0.5


# =============================================================================
# Error Resilience Tests
# =============================================================================


class TestEloErrorResilience:
    """Tests for error handling and resilience."""

    def test_get_nonexistent_rating(self):
        """Should return None for nonexistent rating."""
        adapter = PerformanceAdapter(enable_resilience=False)

        result = adapter.get_rating("nonexistent-rating-id")

        assert result is None

    def test_get_nonexistent_match(self):
        """Should return None for nonexistent match."""
        adapter = PerformanceAdapter(enable_resilience=False)

        result = adapter.get_match("nonexistent-match-id")

        assert result is None

    def test_get_history_for_unknown_agent(self):
        """Should return empty list for unknown agent history."""
        adapter = PerformanceAdapter(enable_resilience=False)

        history = adapter.get_agent_skill_history("unknown-agent")

        assert history == []

    def test_get_matches_for_unknown_agent(self):
        """Should return empty list for unknown agent matches."""
        adapter = PerformanceAdapter(enable_resilience=False)

        matches = adapter.get_agent_matches("unknown-agent")

        assert matches == []

    def test_get_relationship_unknown_agents(self):
        """Should return None for relationship between unknown agents."""
        adapter = PerformanceAdapter(enable_resilience=False)

        result = adapter.get_relationship("unknown1", "unknown2")

        assert result is None

    def test_store_rating_with_missing_optional_fields(self):
        """Should handle rating with minimal required fields."""
        adapter = PerformanceAdapter(enable_resilience=False)

        minimal_rating = MagicMock()
        minimal_rating.agent_name = "minimal"
        minimal_rating.elo = 1500
        minimal_rating.domain_elos = {}
        minimal_rating.wins = 0
        minimal_rating.losses = 0
        minimal_rating.draws = 0
        minimal_rating.debates_count = 0
        minimal_rating.win_rate = 0.0
        minimal_rating.games_played = 0
        minimal_rating.critiques_accepted = 0
        minimal_rating.critiques_total = 0
        minimal_rating.critique_acceptance_rate = 0.0
        minimal_rating.calibration_correct = 0
        minimal_rating.calibration_total = 0
        minimal_rating.calibration_accuracy = 0.0
        minimal_rating.updated_at = ""

        rating_id = adapter.store_rating(minimal_rating)

        assert rating_id is not None
        stored = adapter.get_rating(rating_id)
        assert stored["agent_name"] == "minimal"

    def test_relationship_below_threshold_returns_none(self):
        """Should not store relationship below debate threshold."""
        adapter = PerformanceAdapter(enable_resilience=False)

        metrics = _make_mock_relationship(debates_together=2)  # Below 5 threshold

        result = adapter.store_relationship(metrics)

        assert result is None
        assert len(adapter._relationships) == 0

    def test_relationship_exactly_at_threshold(self):
        """Should store relationship exactly at debate threshold."""
        adapter = PerformanceAdapter(enable_resilience=False)

        metrics = _make_mock_relationship(debates_together=5)  # Exactly at threshold

        result = adapter.store_relationship(metrics)

        assert result is not None
        assert len(adapter._relationships) == 1


# =============================================================================
# Additional ELO Storage Tests
# =============================================================================


class TestEloStorageAdditional:
    """Additional tests for ELO storage functionality."""

    def test_store_rating_generates_unique_ids(self):
        """Should generate unique IDs for each rating snapshot."""
        adapter = PerformanceAdapter(enable_resilience=False)

        rating = _make_mock_rating()

        id1 = adapter.store_rating(rating)
        id2 = adapter.store_rating(rating)

        assert id1 != id2
        assert len(adapter._ratings) == 2

    def test_store_match_with_three_participants(self):
        """Should handle matches with more than two participants."""
        adapter = PerformanceAdapter(enable_resilience=False)

        match = _make_mock_match(
            debate_id="d-multi",
            participants=["claude", "gpt4", "gemini"],
            scores={"claude": 0.9, "gpt4": 0.7, "gemini": 0.6},
        )

        match_id = adapter.store_match(match)

        stored = adapter.get_match(match_id)
        assert len(stored["participants"]) == 3

    def test_calibration_history_ordering(self):
        """Should return calibrations in chronological order."""
        adapter = PerformanceAdapter(enable_resilience=False)

        for i in range(5):
            adapter.store_calibration(
                agent_name="claude",
                debate_id=f"d-{i}",
                predicted_winner="claude",
                predicted_confidence=0.6 + i * 0.05,
                actual_winner="claude" if i % 2 == 0 else "gpt4",
                was_correct=i % 2 == 0,
                brier_score=0.1 + i * 0.05,
            )

        history = adapter.get_agent_calibration_history("claude")

        assert len(history) == 5
        # Should be ordered by created_at descending
        for i in range(len(history) - 1):
            assert history[i]["created_at"] >= history[i + 1]["created_at"]

    def test_domain_expertise_retrieval(self):
        """Should retrieve agents by domain from ELO ratings."""
        adapter = PerformanceAdapter(enable_resilience=False)

        # Store ratings with different domains
        r1 = _make_mock_rating(agent_name="claude", domain_elos={"security": 1700})
        r2 = _make_mock_rating(agent_name="gpt4", domain_elos={"security": 1650})
        r3 = _make_mock_rating(agent_name="gemini", domain_elos={"coding": 1800})

        adapter.store_rating(r1)
        adapter.store_rating(r2)
        adapter.store_rating(r3)

        security_experts = adapter.get_domain_expertise("security")

        assert len(security_experts) == 2
        # Should be sorted by domain ELO descending
        assert security_experts[0]["agent_name"] == "claude"

    def test_relationship_stores_all_fields(self):
        """Should store all relationship metric fields."""
        adapter = PerformanceAdapter(enable_resilience=False)

        metrics = _make_mock_relationship(
            agent_a="claude",
            agent_b="gpt4",
            debates_together=15,
            a_wins_vs_b=8,
            b_wins_vs_a=5,
            draws=2,
        )

        adapter.store_relationship(metrics)

        rel = adapter.get_relationship("claude", "gpt4")

        assert rel["debates_together"] == 15
        assert rel["a_wins_vs_b"] == 8
        assert rel["b_wins_vs_a"] == 5
        assert rel["draws"] == 2
        assert rel["avg_elo_diff"] == 50.0
        assert rel["synergy_score"] == 0.7

    def test_relationship_lookup_both_orders(self):
        """Should find relationship regardless of agent order."""
        adapter = PerformanceAdapter(enable_resilience=False)

        metrics = _make_mock_relationship(agent_a="alpha", agent_b="beta")
        adapter.store_relationship(metrics)

        # Both orderings should work
        assert adapter.get_relationship("alpha", "beta") is not None
        assert adapter.get_relationship("beta", "alpha") is not None
