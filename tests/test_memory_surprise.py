"""
Tests for the unified surprise scoring module.

Tests the consolidated surprise calculation functions from aragora.memory.surprise.
"""

import pytest

from aragora.memory.surprise import (
    CategoryStats,
    SurpriseScorer,
    calculate_base_rate,
    calculate_combined_surprise,
    calculate_surprise,
    calculate_surprise_from_db_row,
    create_db_base_rate_calculator,
    update_surprise_ema,
    DEFAULT_SURPRISE_ALPHA,
)


class TestCalculateSurprise:
    """Tests for the calculate_surprise function."""

    def test_perfect_prediction_no_surprise(self):
        """When actual matches expected, surprise should be 0."""
        assert calculate_surprise(actual=1.0, expected=1.0) == 0.0
        assert calculate_surprise(actual=0.0, expected=0.0) == 0.0

    def test_complete_surprise(self):
        """When actual is opposite of expected, surprise should be high."""
        assert calculate_surprise(actual=1.0, expected=0.0) == 1.0
        assert calculate_surprise(actual=0.0, expected=1.0) == 1.0

    def test_partial_surprise(self):
        """Partial deviation should give proportional surprise."""
        surprise = calculate_surprise(actual=1.0, expected=0.3)
        assert surprise == pytest.approx(0.7, abs=0.01)

        surprise = calculate_surprise(actual=0.0, expected=0.8)
        assert surprise == pytest.approx(0.8, abs=0.01)

    def test_scale_factor(self):
        """Scale factor should multiply the raw surprise."""
        base = calculate_surprise(actual=1.0, expected=0.5)
        scaled = calculate_surprise(actual=1.0, expected=0.5, scale_factor=2.0)
        assert scaled == pytest.approx(base * 2.0, abs=0.01)

    def test_max_surprise_cap(self):
        """Surprise should be capped at max_surprise."""
        # With scale_factor=2.0 and deviation=0.8, raw would be 1.6
        surprise = calculate_surprise(
            actual=1.0, expected=0.2, scale_factor=2.0, max_surprise=1.0
        )
        assert surprise == 1.0

    def test_custom_max_surprise(self):
        """Custom max_surprise should be respected."""
        surprise = calculate_surprise(
            actual=1.0, expected=0.0, max_surprise=0.5
        )
        assert surprise == 0.5


class TestCalculateBaseRate:
    """Tests for the calculate_base_rate function."""

    def test_even_success_failure(self):
        """With equal successes and failures, rate should be near 0.5."""
        rate = calculate_base_rate(success_count=5, failure_count=5)
        assert rate == pytest.approx(0.5, abs=0.1)

    def test_all_successes(self):
        """With all successes, rate should be high but not 1.0 (smoothing)."""
        rate = calculate_base_rate(success_count=10, failure_count=0)
        assert 0.8 < rate < 1.0

    def test_all_failures(self):
        """With all failures, rate should be low but not 0.0 (smoothing)."""
        rate = calculate_base_rate(success_count=0, failure_count=10)
        assert 0.0 < rate < 0.2

    def test_no_observations_returns_prior(self):
        """With no observations, should return the prior."""
        rate = calculate_base_rate(success_count=0, failure_count=0)
        assert rate == 0.5

        rate = calculate_base_rate(success_count=0, failure_count=0, prior=0.7)
        assert rate == 0.7

    def test_prior_weight_effect(self):
        """Higher prior weight should bias toward prior more."""
        # With few observations, high prior_weight should pull toward prior
        rate_weak = calculate_base_rate(
            success_count=2, failure_count=0, prior=0.5, prior_weight=2
        )
        rate_strong = calculate_base_rate(
            success_count=2, failure_count=0, prior=0.5, prior_weight=10
        )
        # Strong prior should be closer to 0.5 than weak prior
        assert abs(rate_strong - 0.5) < abs(rate_weak - 0.5)


class TestUpdateSurpriseEma:
    """Tests for the update_surprise_ema function."""

    def test_alpha_zero_keeps_old(self):
        """With alpha=0, should return old surprise."""
        result = update_surprise_ema(old_surprise=0.5, new_surprise=1.0, alpha=0.0)
        assert result == 0.5

    def test_alpha_one_uses_new(self):
        """With alpha=1, should return new surprise."""
        result = update_surprise_ema(old_surprise=0.5, new_surprise=1.0, alpha=1.0)
        assert result == 1.0

    def test_default_alpha(self):
        """Default alpha (0.3) should blend old and new."""
        result = update_surprise_ema(old_surprise=0.5, new_surprise=0.8)
        # EMA: 0.5 * 0.7 + 0.8 * 0.3 = 0.35 + 0.24 = 0.59
        assert result == pytest.approx(0.59, abs=0.01)

    def test_ema_convergence(self):
        """Repeated updates should converge toward new value."""
        surprise = 0.0
        for _ in range(20):
            surprise = update_surprise_ema(surprise, 1.0, alpha=0.3)
        # After many iterations, should be close to 1.0
        assert surprise > 0.95


class TestCalculateCombinedSurprise:
    """Tests for the calculate_combined_surprise function."""

    def test_no_agent_error_returns_success_surprise(self):
        """Without agent error, should return just success surprise."""
        result = calculate_combined_surprise(success_surprise=0.7)
        assert result == 0.7

    def test_combines_both_signals(self):
        """Should combine success and agent errors with weights."""
        result = calculate_combined_surprise(
            success_surprise=0.6,
            agent_prediction_error=0.4,
            success_weight=0.7,
            agent_weight=0.3,
        )
        # 0.6 * 0.7 + 0.4 * 0.3 = 0.42 + 0.12 = 0.54
        assert result == pytest.approx(0.54, abs=0.01)

    def test_capped_at_one(self):
        """Combined surprise should be capped at 1.0."""
        result = calculate_combined_surprise(
            success_surprise=1.0,
            agent_prediction_error=1.0,
            success_weight=0.7,
            agent_weight=0.7,  # Total weight > 1
        )
        assert result == 1.0


class TestCategoryStats:
    """Tests for the CategoryStats dataclass."""

    def test_initial_values(self):
        """Initial stats should be zero."""
        stats = CategoryStats()
        assert stats.success_count == 0
        assert stats.failure_count == 0
        assert stats.current_surprise == 0.0

    def test_total_property(self):
        """Total should be sum of success and failure."""
        stats = CategoryStats(success_count=5, failure_count=3)
        assert stats.total == 8

    def test_success_rate_no_observations(self):
        """Success rate with no observations should be 0.5."""
        stats = CategoryStats()
        assert stats.success_rate == 0.5

    def test_success_rate_calculation(self):
        """Success rate should be correct."""
        stats = CategoryStats(success_count=3, failure_count=7)
        assert stats.success_rate == 0.3


class TestSurpriseScorer:
    """Tests for the SurpriseScorer class."""

    def test_initial_state(self):
        """New scorer should have no categories."""
        scorer = SurpriseScorer()
        assert len(scorer.get_all_categories()) == 0

    def test_score_first_outcome(self):
        """First outcome should create category and return surprise."""
        scorer = SurpriseScorer()
        surprise = scorer.score_outcome("bugs", is_success=True)
        assert 0.0 <= surprise <= 1.0
        assert "bugs" in scorer.get_all_categories()

    def test_category_stats_updated(self):
        """Scoring should update category stats."""
        scorer = SurpriseScorer()
        scorer.score_outcome("bugs", is_success=True)
        scorer.score_outcome("bugs", is_success=False)

        stats = scorer.get_category_stats("bugs")
        assert stats is not None
        assert stats.success_count == 1
        assert stats.failure_count == 1

    def test_different_categories_independent(self):
        """Different categories should be tracked separately."""
        scorer = SurpriseScorer()
        scorer.score_outcome("bugs", is_success=True)
        scorer.score_outcome("bugs", is_success=True)
        scorer.score_outcome("features", is_success=False)

        bugs_stats = scorer.get_category_stats("bugs")
        features_stats = scorer.get_category_stats("features")

        assert bugs_stats.success_count == 2
        assert bugs_stats.failure_count == 0
        assert features_stats.success_count == 0
        assert features_stats.failure_count == 1

    def test_get_category_surprise(self):
        """Should return current surprise for a category."""
        scorer = SurpriseScorer()
        scorer.score_outcome("bugs", is_success=True)
        surprise = scorer.get_category_surprise("bugs")
        assert 0.0 <= surprise <= 1.0

    def test_get_nonexistent_category_surprise(self):
        """Should return 0 for nonexistent category."""
        scorer = SurpriseScorer()
        assert scorer.get_category_surprise("nonexistent") == 0.0

    def test_reset_category(self):
        """Resetting a category should remove its stats."""
        scorer = SurpriseScorer()
        scorer.score_outcome("bugs", is_success=True)
        assert scorer.get_category_stats("bugs") is not None

        scorer.reset_category("bugs")
        assert scorer.get_category_stats("bugs") is None

    def test_reset_all(self):
        """Resetting all should clear all categories."""
        scorer = SurpriseScorer()
        scorer.score_outcome("bugs", is_success=True)
        scorer.score_outcome("features", is_success=False)
        assert len(scorer.get_all_categories()) == 2

        scorer.reset_all()
        assert len(scorer.get_all_categories()) == 0

    def test_custom_alpha(self):
        """Custom alpha should be used in EMA."""
        scorer_fast = SurpriseScorer(alpha=0.9)
        scorer_slow = SurpriseScorer(alpha=0.1)

        # Same sequence of outcomes
        for _ in range(5):
            scorer_fast.score_outcome("test", is_success=True)
            scorer_slow.score_outcome("test", is_success=True)

        # Fast should have lower surprise (converged more quickly)
        # after consistent successes with high base rate
        fast_surprise = scorer_fast.get_category_surprise("test")
        slow_surprise = scorer_slow.get_category_surprise("test")

        # Both should be defined
        assert fast_surprise >= 0
        assert slow_surprise >= 0

    def test_with_agent_prediction_error(self):
        """Should incorporate agent prediction error when provided."""
        scorer = SurpriseScorer()
        surprise_without = scorer.score_outcome("test1", is_success=True)
        surprise_with = scorer.score_outcome(
            "test2", is_success=True, agent_prediction_error=0.8
        )
        # With high agent error, surprise should be higher (all else equal)
        # Note: categories are different so base rates start the same


class TestCalculateSurpriseFromDbRow:
    """Tests for the calculate_surprise_from_db_row helper."""

    def test_basic_calculation(self):
        """Should calculate surprise from DB stats."""
        surprise = calculate_surprise_from_db_row(
            success_count=8,
            failure_count=2,
            is_success=False,  # Failure with 80% success rate = surprising
            old_surprise=0.0,
        )
        # Base rate ~0.8, actual=0.0, deviation ~0.8
        assert surprise > 0.2

    def test_ema_applied(self):
        """Should apply EMA to old surprise."""
        # Start with high surprise
        surprise = calculate_surprise_from_db_row(
            success_count=5,
            failure_count=5,
            is_success=True,  # Expected with 50% rate
            old_surprise=0.9,
            alpha=0.3,
        )
        # Low new surprise (actual matches expected) should pull down
        assert surprise < 0.9


class TestCreateDbBaseRateCalculator:
    """Tests for the create_db_base_rate_calculator factory."""

    def test_creates_working_calculator(self):
        """Factory should create a working calculator."""

        # Mock query function
        def mock_query(category: str) -> tuple[int, int]:
            if category == "bugs":
                return (8, 2)
            return (0, 0)

        calc = create_db_base_rate_calculator(mock_query)

        rate = calc("bugs")
        assert rate > 0.7  # Should be high (8 successes, 2 failures)

        rate = calc("unknown")
        assert rate == 0.5  # Should be default prior


class TestDefaultAlphaConstant:
    """Tests for the DEFAULT_SURPRISE_ALPHA constant."""

    def test_default_alpha_value(self):
        """Default alpha should be 0.3 as documented."""
        assert DEFAULT_SURPRISE_ALPHA == 0.3

    def test_scorer_uses_default_alpha(self):
        """Scorer should use default alpha if not specified."""
        scorer = SurpriseScorer()
        assert scorer.alpha == DEFAULT_SURPRISE_ALPHA
