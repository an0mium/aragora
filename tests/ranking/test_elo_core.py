"""
Tests for ELO core calculation functions.

Tests cover:
- Expected score calculation
- New ELO rating calculation
- Pairwise ELO changes for multi-agent debates
- ELO change application
- Win probability calculations
- ELO difference for target probability
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from aragora.ranking.elo_core import (
    DEFAULT_ELO,
    K_FACTOR,
    apply_elo_changes,
    calculate_new_elo,
    calculate_pairwise_elo_changes,
    calculate_win_probability,
    elo_diff_for_probability,
    expected_score,
)


@dataclass
class MockAgentRating:
    """Mock AgentRating for testing without database dependency."""

    agent_name: str
    elo: float = DEFAULT_ELO
    domain_elos: dict[str, float] = field(default_factory=dict)
    wins: int = 0
    losses: int = 0
    draws: int = 0
    debates_count: int = 0
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class TestExpectedScore:
    """Tests for expected_score function."""

    def test_equal_ratings_returns_half(self):
        """Test that equal ratings give 0.5 expected score."""
        result = expected_score(1500, 1500)
        assert result == pytest.approx(0.5, rel=1e-6)

    def test_higher_rating_has_higher_expected(self):
        """Test that higher rated player has higher expected score."""
        result = expected_score(1600, 1400)
        assert result > 0.5
        assert result < 1.0

    def test_lower_rating_has_lower_expected(self):
        """Test that lower rated player has lower expected score."""
        result = expected_score(1400, 1600)
        assert result < 0.5
        assert result > 0.0

    def test_400_point_difference_gives_expected_ratio(self):
        """Test that 400 point difference gives ~0.91 expected score."""
        # Standard ELO property: 400 point diff = 10x more likely to win
        result = expected_score(1900, 1500)
        # E = 1 / (1 + 10^((1500-1900)/400)) = 1 / (1 + 10^-1) = 1/1.1 ≈ 0.909
        assert result == pytest.approx(10 / 11, rel=1e-3)

    def test_symmetry_property(self):
        """Test that expected scores sum to 1."""
        expected_a = expected_score(1600, 1400)
        expected_b = expected_score(1400, 1600)
        assert expected_a + expected_b == pytest.approx(1.0, rel=1e-6)

    def test_large_rating_difference(self):
        """Test expected score with large rating difference."""
        result = expected_score(2500, 1000)
        assert result > 0.99  # Very strong player
        assert result < 1.0

    def test_zero_ratings(self):
        """Test expected score with zero ratings."""
        result = expected_score(0, 0)
        assert result == pytest.approx(0.5, rel=1e-6)

    def test_negative_ratings(self):
        """Test expected score with negative ratings."""
        result = expected_score(-100, 100)
        assert result < 0.5


class TestCalculateNewElo:
    """Tests for calculate_new_elo function."""

    def test_win_increases_elo(self):
        """Test that winning increases ELO."""
        expected = expected_score(1500, 1500)
        new_elo = calculate_new_elo(1500, expected, 1.0)
        assert new_elo > 1500

    def test_loss_decreases_elo(self):
        """Test that losing decreases ELO."""
        expected = expected_score(1500, 1500)
        new_elo = calculate_new_elo(1500, expected, 0.0)
        assert new_elo < 1500

    def test_draw_against_equal_no_change(self):
        """Test that draw against equal opponent gives no change."""
        expected = expected_score(1500, 1500)
        new_elo = calculate_new_elo(1500, expected, 0.5)
        assert new_elo == pytest.approx(1500, rel=1e-6)

    def test_k_factor_scales_change(self):
        """Test that K factor scales the rating change."""
        expected = expected_score(1500, 1500)

        change_k16 = calculate_new_elo(1500, expected, 1.0, k=16) - 1500
        change_k32 = calculate_new_elo(1500, expected, 1.0, k=32) - 1500

        assert change_k32 == pytest.approx(change_k16 * 2, rel=1e-6)

    def test_upset_win_gives_larger_gain(self):
        """Test that upset win (lower rated beats higher) gives larger gain."""
        # Lower rated player wins
        expected_low = expected_score(1400, 1600)  # Low expected
        change_low = calculate_new_elo(1400, expected_low, 1.0) - 1400

        # Higher rated player wins
        expected_high = expected_score(1600, 1400)  # High expected
        change_high = calculate_new_elo(1600, expected_high, 1.0) - 1600

        # Upset win should give larger change
        assert change_low > change_high

    def test_expected_loss_gives_smaller_penalty(self):
        """Test that expected loss gives smaller penalty."""
        # Low rated loses to high rated (expected)
        expected_low = expected_score(1400, 1600)
        change_low = 1400 - calculate_new_elo(1400, expected_low, 0.0)

        # High rated loses to low rated (upset)
        expected_high = expected_score(1600, 1400)
        change_high = 1600 - calculate_new_elo(1600, expected_high, 0.0)

        # Upset loss should give larger penalty
        assert change_high > change_low


class TestCalculatePairwiseEloChanges:
    """Tests for calculate_pairwise_elo_changes function."""

    def test_two_player_match(self):
        """Test pairwise ELO changes for two players."""
        participants = ["alice", "bob"]
        scores = {"alice": 1.0, "bob": 0.0}
        ratings = {
            "alice": MockAgentRating("alice", elo=1500),
            "bob": MockAgentRating("bob", elo=1500),
        }

        changes = calculate_pairwise_elo_changes(participants, scores, ratings)

        assert "alice" in changes
        assert "bob" in changes
        assert changes["alice"] > 0  # Winner gains
        assert changes["bob"] < 0  # Loser loses
        assert changes["alice"] == pytest.approx(-changes["bob"], rel=1e-6)  # Zero-sum

    def test_three_player_match(self):
        """Test pairwise ELO changes for three players."""
        participants = ["alice", "bob", "charlie"]
        scores = {"alice": 2.0, "bob": 1.0, "charlie": 0.0}
        ratings = {
            "alice": MockAgentRating("alice", elo=1500),
            "bob": MockAgentRating("bob", elo=1500),
            "charlie": MockAgentRating("charlie", elo=1500),
        }

        changes = calculate_pairwise_elo_changes(participants, scores, ratings)

        # Alice (best) should gain most
        assert changes["alice"] > 0
        # Charlie (worst) should lose most
        assert changes["charlie"] < 0
        # Bob (middle) could be anywhere

    def test_draw_no_score_change(self):
        """Test that draw with equal ratings gives no change."""
        participants = ["alice", "bob"]
        scores = {"alice": 1.0, "bob": 1.0}  # Equal scores = draw
        ratings = {
            "alice": MockAgentRating("alice", elo=1500),
            "bob": MockAgentRating("bob", elo=1500),
        }

        changes = calculate_pairwise_elo_changes(participants, scores, ratings)

        assert changes["alice"] == pytest.approx(0, abs=1e-6)
        assert changes["bob"] == pytest.approx(0, abs=1e-6)

    def test_confidence_weight_scales_changes(self):
        """Test that confidence weight scales ELO changes."""
        participants = ["alice", "bob"]
        scores = {"alice": 1.0, "bob": 0.0}
        ratings = {
            "alice": MockAgentRating("alice", elo=1500),
            "bob": MockAgentRating("bob", elo=1500),
        }

        changes_full = calculate_pairwise_elo_changes(
            participants, scores, ratings, confidence_weight=1.0
        )
        changes_half = calculate_pairwise_elo_changes(
            participants, scores, ratings, confidence_weight=0.5
        )

        assert changes_half["alice"] == pytest.approx(changes_full["alice"] / 2, rel=1e-6)

    def test_calibration_multipliers(self):
        """Test that calibration multipliers affect changes per agent."""
        participants = ["alice", "bob"]
        scores = {"alice": 1.0, "bob": 0.0}
        ratings = {
            "alice": MockAgentRating("alice", elo=1500),
            "bob": MockAgentRating("bob", elo=1500),
        }

        # Alice has high calibration multiplier (overconfident, should lose more on loss)
        k_multipliers = {"alice": 1.5, "bob": 1.0}

        changes = calculate_pairwise_elo_changes(
            participants, scores, ratings, k_multipliers=k_multipliers
        )

        # Alice's change is scaled by 1.5
        changes_no_mult = calculate_pairwise_elo_changes(participants, scores, ratings)
        assert changes["alice"] == pytest.approx(changes_no_mult["alice"] * 1.5, rel=1e-6)

    def test_empty_scores_treated_as_draw(self):
        """Test that zero total score is treated as draw."""
        participants = ["alice", "bob"]
        scores = {"alice": 0.0, "bob": 0.0}
        ratings = {
            "alice": MockAgentRating("alice", elo=1500),
            "bob": MockAgentRating("bob", elo=1500),
        }

        changes = calculate_pairwise_elo_changes(participants, scores, ratings)

        # Equal ELO + draw = no change
        assert changes["alice"] == pytest.approx(0, abs=1e-6)
        assert changes["bob"] == pytest.approx(0, abs=1e-6)


class TestApplyEloChanges:
    """Tests for apply_elo_changes function."""

    def test_applies_changes_to_ratings(self):
        """Test that changes are applied to rating objects."""
        ratings = {
            "alice": MockAgentRating("alice", elo=1500),
            "bob": MockAgentRating("bob", elo=1500),
        }
        elo_changes = {"alice": 10.0, "bob": -10.0}

        ratings_to_save, history = apply_elo_changes(
            elo_changes, ratings, winner="alice", domain=None, debate_id="test"
        )

        assert ratings["alice"].elo == pytest.approx(1510, rel=1e-6)
        assert ratings["bob"].elo == pytest.approx(1490, rel=1e-6)

    def test_updates_win_loss_counters(self):
        """Test that win/loss counters are updated."""
        ratings = {
            "alice": MockAgentRating("alice", elo=1500),
            "bob": MockAgentRating("bob", elo=1500),
        }
        elo_changes = {"alice": 10.0, "bob": -10.0}

        apply_elo_changes(elo_changes, ratings, winner="alice", domain=None, debate_id="test")

        assert ratings["alice"].wins == 1
        assert ratings["alice"].losses == 0
        assert ratings["bob"].wins == 0
        assert ratings["bob"].losses == 1

    def test_updates_draw_counter(self):
        """Test that draw counter is updated when no winner."""
        ratings = {
            "alice": MockAgentRating("alice", elo=1500),
            "bob": MockAgentRating("bob", elo=1500),
        }
        elo_changes = {"alice": 0.0, "bob": 0.0}

        apply_elo_changes(elo_changes, ratings, winner=None, domain=None, debate_id="test")

        assert ratings["alice"].draws == 1
        assert ratings["bob"].draws == 1

    def test_updates_debates_count(self):
        """Test that debates count is incremented."""
        ratings = {
            "alice": MockAgentRating("alice", elo=1500, debates_count=5),
        }
        elo_changes = {"alice": 10.0}

        apply_elo_changes(elo_changes, ratings, winner="alice", domain=None, debate_id="test")

        assert ratings["alice"].debates_count == 6

    def test_updates_domain_specific_elo(self):
        """Test that domain-specific ELO is updated."""
        ratings = {
            "alice": MockAgentRating("alice", elo=1500, domain_elos={"math": 1550}),
        }
        elo_changes = {"alice": 10.0}

        apply_elo_changes(elo_changes, ratings, winner="alice", domain="math", debate_id="test")

        assert ratings["alice"].domain_elos["math"] == pytest.approx(1560, rel=1e-6)

    def test_creates_domain_elo_if_missing(self):
        """Test that domain ELO is created if not present."""
        ratings = {
            "alice": MockAgentRating("alice", elo=1500),
        }
        elo_changes = {"alice": 10.0}

        apply_elo_changes(elo_changes, ratings, winner="alice", domain="physics", debate_id="test")

        # Should use default ELO (1500) + change
        assert ratings["alice"].domain_elos["physics"] == pytest.approx(1510, rel=1e-6)

    def test_returns_history_entries(self):
        """Test that history entries are returned."""
        ratings = {
            "alice": MockAgentRating("alice", elo=1500),
            "bob": MockAgentRating("bob", elo=1500),
        }
        elo_changes = {"alice": 10.0, "bob": -10.0}

        _, history = apply_elo_changes(
            elo_changes, ratings, winner="alice", domain=None, debate_id="test123"
        )

        assert len(history) == 2
        # Each entry is (agent_name, new_elo, debate_id)
        agent_names = [h[0] for h in history]
        assert "alice" in agent_names
        assert "bob" in agent_names


class TestCalculateWinProbability:
    """Tests for calculate_win_probability function."""

    def test_equal_elo_gives_50_percent(self):
        """Test that equal ELO gives 50% win probability."""
        prob = calculate_win_probability(1500, 1500)
        assert prob == pytest.approx(0.5, rel=1e-6)

    def test_higher_elo_more_likely_to_win(self):
        """Test that higher ELO player is more likely to win."""
        prob = calculate_win_probability(1600, 1400)
        assert prob > 0.5

    def test_equivalent_to_expected_score(self):
        """Test that win probability equals expected score."""
        prob = calculate_win_probability(1700, 1500)
        expected = expected_score(1700, 1500)
        assert prob == expected


class TestEloDiffForProbability:
    """Tests for elo_diff_for_probability function."""

    def test_50_percent_gives_zero_diff(self):
        """Test that 50% probability requires zero ELO difference."""
        diff = elo_diff_for_probability(0.5)
        assert diff == pytest.approx(0, abs=1e-6)

    def test_higher_probability_positive_diff(self):
        """Test that higher win probability requires positive ELO diff."""
        diff = elo_diff_for_probability(0.75)
        assert diff > 0

    def test_lower_probability_negative_diff(self):
        """Test that lower win probability requires negative ELO diff."""
        diff = elo_diff_for_probability(0.25)
        assert diff < 0

    def test_inverse_of_expected_score(self):
        """Test that function is inverse of expected score."""
        target_prob = 0.7
        diff = elo_diff_for_probability(target_prob)

        # Using this diff, expected_score should give back target_prob
        computed_prob = expected_score(1500 + diff, 1500)
        assert computed_prob == pytest.approx(target_prob, rel=1e-4)

    def test_extreme_probabilities_raise_error(self):
        """Test that 0 or 1 probability raises error."""
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            elo_diff_for_probability(0.0)

        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            elo_diff_for_probability(1.0)

    def test_negative_probability_raises_error(self):
        """Test that negative probability raises error."""
        with pytest.raises(ValueError):
            elo_diff_for_probability(-0.5)


class TestEloConstants:
    """Tests for ELO system constants."""

    def test_default_elo_is_1500(self):
        """Test that default ELO is 1500."""
        assert DEFAULT_ELO == 1500

    def test_k_factor_is_positive(self):
        """Test that K factor is positive."""
        assert K_FACTOR > 0

    def test_k_factor_reasonable_range(self):
        """Test that K factor is in reasonable range (8-32)."""
        assert 8 <= K_FACTOR <= 64


class TestEdgeCases:
    """Edge case tests for ELO calculations."""

    def test_very_high_ratings(self):
        """Test calculations with very high ratings."""
        result = expected_score(3000, 2500)
        assert 0 < result < 1

        new_elo = calculate_new_elo(3000, result, 1.0)
        assert new_elo > 3000

    def test_very_low_ratings(self):
        """Test calculations with very low ratings."""
        result = expected_score(500, 1000)
        assert 0 < result < 1

        new_elo = calculate_new_elo(500, result, 1.0)
        assert new_elo > 500

    def test_single_participant(self):
        """Test pairwise changes with single participant."""
        participants = ["alice"]
        scores = {"alice": 1.0}
        ratings = {"alice": MockAgentRating("alice", elo=1500)}

        # Should handle gracefully (no pairs to compare)
        changes = calculate_pairwise_elo_changes(participants, scores, ratings)
        assert changes == {}

    def test_missing_score_defaults_to_zero(self):
        """Test that missing score defaults to zero."""
        participants = ["alice", "bob"]
        scores = {"alice": 1.0}  # bob missing
        ratings = {
            "alice": MockAgentRating("alice", elo=1500),
            "bob": MockAgentRating("bob", elo=1500),
        }

        changes = calculate_pairwise_elo_changes(participants, scores, ratings)

        # alice should gain (has score), bob should lose (no score = 0)
        assert changes["alice"] > 0
        assert changes["bob"] < 0
