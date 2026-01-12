"""
Tests for aragora.ranking.elo_core - Pure ELO calculation functions.

Tests cover:
- expected_score() calculations
- calculate_new_elo() rating updates
- calculate_pairwise_elo_changes() multi-agent scenarios
- apply_elo_changes() rating mutations
- calculate_win_probability() predictions
- elo_diff_for_probability() inverse calculations
"""

import math
import pytest
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

from aragora.ranking.elo_core import (
    DEFAULT_ELO,
    K_FACTOR,
    expected_score,
    calculate_new_elo,
    calculate_pairwise_elo_changes,
    apply_elo_changes,
    calculate_win_probability,
    elo_diff_for_probability,
)


# Mock AgentRating for testing (matches the real AgentRating interface)
@dataclass
class MockAgentRating:
    """Mock AgentRating for testing elo_core functions."""

    name: str
    elo: float = DEFAULT_ELO
    wins: int = 0
    losses: int = 0
    draws: int = 0
    debates_count: int = 0
    critiques_accepted: int = 0
    critiques_total: int = 0
    domain_elos: Dict[str, float] = field(default_factory=dict)
    updated_at: str = ""


class TestExpectedScore:
    """Tests for expected_score function."""

    def test_equal_ratings_returns_half(self):
        """Equal ratings should give 50% expected score."""
        assert expected_score(1500, 1500) == pytest.approx(0.5)

    def test_higher_rating_above_half(self):
        """Higher rated player should have >50% expected score."""
        result = expected_score(1600, 1500)
        assert result > 0.5

    def test_lower_rating_below_half(self):
        """Lower rated player should have <50% expected score."""
        result = expected_score(1400, 1500)
        assert result < 0.5

    def test_400_point_difference(self):
        """400 point difference should give ~91% expected score."""
        result = expected_score(1900, 1500)
        assert result == pytest.approx(0.909, rel=0.01)

    def test_800_point_difference(self):
        """800 point difference should give ~99% expected score."""
        result = expected_score(2300, 1500)
        assert result == pytest.approx(0.99, rel=0.01)

    def test_symmetry(self):
        """Expected scores should sum to 1.0."""
        score_a = expected_score(1600, 1400)
        score_b = expected_score(1400, 1600)
        assert score_a + score_b == pytest.approx(1.0)

    def test_very_large_difference(self):
        """Very large differences should approach but not exceed limits."""
        result = expected_score(3000, 1000)
        assert 0.99 < result < 1.0

    def test_very_small_difference(self):
        """Very small differences should be close to 0.5."""
        result = expected_score(1500, 1490)
        assert 0.49 < result < 0.52


class TestCalculateNewElo:
    """Tests for calculate_new_elo function."""

    def test_win_increases_rating(self):
        """Winning should increase rating."""
        new_elo = calculate_new_elo(1500, 0.5, 1.0)
        assert new_elo > 1500

    def test_loss_decreases_rating(self):
        """Losing should decrease rating."""
        new_elo = calculate_new_elo(1500, 0.5, 0.0)
        assert new_elo < 1500

    def test_draw_with_equal_expected(self):
        """Draw with 50% expected should not change rating."""
        new_elo = calculate_new_elo(1500, 0.5, 0.5)
        assert new_elo == pytest.approx(1500)

    def test_expected_win_but_lost(self):
        """Losing when expected to win should decrease rating more."""
        # High expected (0.9) but lost (0)
        new_elo = calculate_new_elo(1500, 0.9, 0.0)
        expected_change = K_FACTOR * (0.0 - 0.9)
        assert new_elo == pytest.approx(1500 + expected_change)

    def test_expected_loss_but_won(self):
        """Winning when expected to lose should increase rating more."""
        # Low expected (0.1) but won (1)
        new_elo = calculate_new_elo(1500, 0.1, 1.0)
        expected_change = K_FACTOR * (1.0 - 0.1)
        assert new_elo == pytest.approx(1500 + expected_change)

    def test_custom_k_factor(self):
        """Custom K-factor should scale rating changes."""
        custom_k = 64
        new_elo = calculate_new_elo(1500, 0.5, 1.0, k=custom_k)
        expected_change = custom_k * (1.0 - 0.5)
        assert new_elo == pytest.approx(1500 + expected_change)

    def test_zero_k_factor(self):
        """Zero K-factor should result in no change."""
        new_elo = calculate_new_elo(1500, 0.5, 1.0, k=0)
        assert new_elo == 1500


class TestCalculatePairwiseEloChanges:
    """Tests for calculate_pairwise_elo_changes function."""

    def test_two_agents_clear_winner(self):
        """Two agents with clear winner."""
        ratings = {
            "alice": MockAgentRating(name="alice", elo=1500),
            "bob": MockAgentRating(name="bob", elo=1500),
        }
        scores = {"alice": 1.0, "bob": 0.0}

        changes = calculate_pairwise_elo_changes(
            ["alice", "bob"], scores, ratings
        )

        assert "alice" in changes
        assert "bob" in changes
        assert changes["alice"] > 0
        assert changes["bob"] < 0

    def test_two_agents_draw(self):
        """Two agents with equal scores (draw)."""
        ratings = {
            "alice": MockAgentRating(name="alice", elo=1500),
            "bob": MockAgentRating(name="bob", elo=1500),
        }
        scores = {"alice": 0.5, "bob": 0.5}

        changes = calculate_pairwise_elo_changes(
            ["alice", "bob"], scores, ratings
        )

        # With equal ratings and equal scores, changes should be minimal
        assert abs(changes["alice"]) < 0.01
        assert abs(changes["bob"]) < 0.01

    def test_three_agents_pairwise(self):
        """Three agents should compute all pairwise combinations."""
        ratings = {
            "alice": MockAgentRating(name="alice", elo=1500),
            "bob": MockAgentRating(name="bob", elo=1500),
            "carol": MockAgentRating(name="carol", elo=1500),
        }
        scores = {"alice": 3.0, "bob": 2.0, "carol": 1.0}

        changes = calculate_pairwise_elo_changes(
            ["alice", "bob", "carol"], scores, ratings
        )

        # All three should have changes
        assert len(changes) == 3
        # Alice (highest score) should gain, Carol (lowest) should lose
        assert changes["alice"] > changes["bob"] > changes["carol"]

    def test_confidence_weight(self):
        """Confidence weight should scale all changes."""
        ratings = {
            "alice": MockAgentRating(name="alice", elo=1500),
            "bob": MockAgentRating(name="bob", elo=1500),
        }
        scores = {"alice": 1.0, "bob": 0.0}

        full_changes = calculate_pairwise_elo_changes(
            ["alice", "bob"], scores, ratings, confidence_weight=1.0
        )
        half_changes = calculate_pairwise_elo_changes(
            ["alice", "bob"], scores, ratings, confidence_weight=0.5
        )

        assert half_changes["alice"] == pytest.approx(full_changes["alice"] * 0.5)

    def test_k_multipliers_affect_changes(self):
        """K-multipliers should affect individual agent changes."""
        ratings = {
            "alice": MockAgentRating(name="alice", elo=1500),
            "bob": MockAgentRating(name="bob", elo=1500),
        }
        scores = {"alice": 1.0, "bob": 0.0}

        # Alice with 1.5x multiplier, Bob with 1.0x
        changes_with_mult = calculate_pairwise_elo_changes(
            ["alice", "bob"], scores, ratings,
            k_multipliers={"alice": 1.5, "bob": 1.0}
        )
        changes_no_mult = calculate_pairwise_elo_changes(
            ["alice", "bob"], scores, ratings
        )

        # Alice's gain should be 1.5x base
        assert changes_with_mult["alice"] == pytest.approx(changes_no_mult["alice"] * 1.5)
        # Bob's loss should be same (1.0x multiplier)
        assert changes_with_mult["bob"] == pytest.approx(changes_no_mult["bob"])

    def test_empty_participants(self):
        """Empty participants should return empty changes."""
        changes = calculate_pairwise_elo_changes([], {}, {})
        assert changes == {}

    def test_single_participant(self):
        """Single participant should return empty changes (no pairs)."""
        ratings = {"alice": MockAgentRating(name="alice", elo=1500)}
        scores = {"alice": 1.0}

        changes = calculate_pairwise_elo_changes(["alice"], scores, ratings)
        assert changes == {}

    def test_zero_total_scores(self):
        """Zero total scores should result in draw (0.5)."""
        ratings = {
            "alice": MockAgentRating(name="alice", elo=1500),
            "bob": MockAgentRating(name="bob", elo=1500),
        }
        scores = {"alice": 0, "bob": 0}

        changes = calculate_pairwise_elo_changes(
            ["alice", "bob"], scores, ratings
        )

        # Should behave like a draw
        assert abs(changes["alice"]) < 0.01
        assert abs(changes["bob"]) < 0.01


class TestApplyEloChanges:
    """Tests for apply_elo_changes function."""

    def test_applies_changes_to_ratings(self):
        """Changes should be applied to rating objects."""
        ratings = {
            "alice": MockAgentRating(name="alice", elo=1500),
            "bob": MockAgentRating(name="bob", elo=1500),
        }
        changes = {"alice": 15.0, "bob": -15.0}

        updated, history = apply_elo_changes(changes, ratings, winner="alice")

        assert ratings["alice"].elo == pytest.approx(1515.0)
        assert ratings["bob"].elo == pytest.approx(1485.0)

    def test_updates_win_loss_counts(self):
        """Winner should get win, loser should get loss."""
        ratings = {
            "alice": MockAgentRating(name="alice", elo=1500),
            "bob": MockAgentRating(name="bob", elo=1500),
        }
        changes = {"alice": 15.0, "bob": -15.0}

        apply_elo_changes(changes, ratings, winner="alice")

        assert ratings["alice"].wins == 1
        assert ratings["alice"].losses == 0
        assert ratings["bob"].wins == 0
        assert ratings["bob"].losses == 1

    def test_draw_updates_draw_count(self):
        """Draw should update draw counts for all."""
        ratings = {
            "alice": MockAgentRating(name="alice", elo=1500),
            "bob": MockAgentRating(name="bob", elo=1500),
        }
        changes = {"alice": 0.0, "bob": 0.0}

        apply_elo_changes(changes, ratings, winner=None)  # Draw

        assert ratings["alice"].draws == 1
        assert ratings["bob"].draws == 1

    def test_increments_debates_count(self):
        """Debates count should be incremented."""
        ratings = {
            "alice": MockAgentRating(name="alice", elo=1500, debates_count=5),
        }
        changes = {"alice": 10.0}

        apply_elo_changes(changes, ratings, winner="alice")

        assert ratings["alice"].debates_count == 6

    def test_updates_domain_elos(self):
        """Domain ELOs should be updated if domain specified."""
        ratings = {
            "alice": MockAgentRating(name="alice", elo=1500),
        }
        changes = {"alice": 10.0}

        apply_elo_changes(changes, ratings, winner="alice", domain="security")

        assert "security" in ratings["alice"].domain_elos
        assert ratings["alice"].domain_elos["security"] == pytest.approx(DEFAULT_ELO + 10.0)

    def test_returns_ratings_and_history(self):
        """Should return updated ratings and history entries."""
        ratings = {
            "alice": MockAgentRating(name="alice", elo=1500),
        }
        changes = {"alice": 10.0}

        updated, history = apply_elo_changes(
            changes, ratings, winner="alice", debate_id="debate-123"
        )

        assert len(updated) == 1
        assert updated[0].name == "alice"
        assert len(history) == 1
        assert history[0] == ("alice", pytest.approx(1510.0), "debate-123")

    def test_updates_timestamp(self):
        """Updated_at should be set."""
        ratings = {
            "alice": MockAgentRating(name="alice", elo=1500),
        }
        changes = {"alice": 10.0}

        apply_elo_changes(changes, ratings, winner="alice")

        assert ratings["alice"].updated_at != ""


class TestCalculateWinProbability:
    """Tests for calculate_win_probability function."""

    def test_equal_ratings(self):
        """Equal ratings should give 50% win probability."""
        prob = calculate_win_probability(1500, 1500)
        assert prob == pytest.approx(0.5)

    def test_higher_rating_advantage(self):
        """Higher rated player should have higher win probability."""
        prob = calculate_win_probability(1700, 1500)
        assert prob > 0.5

    def test_consistent_with_expected_score(self):
        """Should be consistent with expected_score function."""
        for elo_a in [1400, 1500, 1600]:
            for elo_b in [1400, 1500, 1600]:
                prob = calculate_win_probability(elo_a, elo_b)
                expected = expected_score(elo_a, elo_b)
                assert prob == pytest.approx(expected)


class TestEloDiffForProbability:
    """Tests for elo_diff_for_probability function."""

    def test_50_percent_is_zero_diff(self):
        """50% probability should require 0 ELO difference."""
        diff = elo_diff_for_probability(0.5)
        assert diff == pytest.approx(0.0)

    def test_high_probability_positive_diff(self):
        """High probability should require positive ELO difference."""
        diff = elo_diff_for_probability(0.9)
        assert diff > 0

    def test_low_probability_negative_diff(self):
        """Low probability should require negative ELO difference."""
        diff = elo_diff_for_probability(0.1)
        assert diff < 0

    def test_round_trip_consistency(self):
        """Should be inverse of expected_score."""
        for target_prob in [0.1, 0.3, 0.5, 0.7, 0.9]:
            diff = elo_diff_for_probability(target_prob)
            # If we apply this diff to equal ratings, we should get target_prob
            actual_prob = expected_score(1500 + diff, 1500)
            assert actual_prob == pytest.approx(target_prob, rel=0.01)

    def test_invalid_probability_zero(self):
        """Probability of 0 should raise ValueError."""
        with pytest.raises(ValueError):
            elo_diff_for_probability(0.0)

    def test_invalid_probability_one(self):
        """Probability of 1 should raise ValueError."""
        with pytest.raises(ValueError):
            elo_diff_for_probability(1.0)

    def test_invalid_probability_negative(self):
        """Negative probability should raise ValueError."""
        with pytest.raises(ValueError):
            elo_diff_for_probability(-0.5)

    def test_invalid_probability_above_one(self):
        """Probability > 1 should raise ValueError."""
        with pytest.raises(ValueError):
            elo_diff_for_probability(1.5)


class TestModuleConstants:
    """Tests for module constants."""

    def test_default_elo_value(self):
        """DEFAULT_ELO should be a reasonable starting rating."""
        assert 1000 <= DEFAULT_ELO <= 2000

    def test_k_factor_value(self):
        """K_FACTOR should be a reasonable volatility value."""
        assert 10 <= K_FACTOR <= 64


class TestEdgeCases:
    """Edge case tests."""

    def test_very_high_elos(self):
        """Functions should work with very high ELOs."""
        score = expected_score(3000, 2800)
        assert 0.5 < score < 1.0

    def test_very_low_elos(self):
        """Functions should work with very low ELOs."""
        score = expected_score(800, 600)
        assert 0.5 < score < 1.0

    def test_negative_elos(self):
        """Functions should work with negative ELOs (though unusual)."""
        score = expected_score(-100, -200)
        assert 0.5 < score < 1.0

    def test_float_precision(self):
        """Results should have reasonable float precision."""
        score = expected_score(1500.123456789, 1499.987654321)
        assert isinstance(score, float)
        assert 0.49 < score < 0.51
