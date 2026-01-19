"""Tests for surprise-based memorization module.

Tests the unified surprise scoring used for memory retention decisions.
Based on Titans/MIRAS principles - high surprise = novel = prioritize learning.
"""

import pytest


class TestCalculateSurprise:
    """Test calculate_surprise function."""

    def test_surprise_expected_success(self):
        """Test surprise when actual matches expected."""
        from aragora.memory.surprise import calculate_surprise

        # Expected 80% success, got success -> low surprise
        surprise = calculate_surprise(actual=1.0, expected=0.8)
        assert surprise == pytest.approx(0.2)

    def test_surprise_unexpected_success(self):
        """Test high surprise when unexpected success."""
        from aragora.memory.surprise import calculate_surprise

        # Expected 20% success, got success -> high surprise
        surprise = calculate_surprise(actual=1.0, expected=0.2)
        assert surprise == pytest.approx(0.8)

    def test_surprise_unexpected_failure(self):
        """Test high surprise when unexpected failure."""
        from aragora.memory.surprise import calculate_surprise

        # Expected 90% success, got failure -> high surprise
        surprise = calculate_surprise(actual=0.0, expected=0.9)
        assert surprise == pytest.approx(0.9)

    def test_surprise_expected_failure(self):
        """Test low surprise when expected failure."""
        from aragora.memory.surprise import calculate_surprise

        # Expected 10% success, got failure -> low surprise
        surprise = calculate_surprise(actual=0.0, expected=0.1)
        assert surprise == pytest.approx(0.1)

    def test_surprise_scale_factor(self):
        """Test surprise with scale factor."""
        from aragora.memory.surprise import calculate_surprise

        surprise = calculate_surprise(actual=1.0, expected=0.5, scale_factor=2.0)
        assert surprise == pytest.approx(1.0)  # Capped at max

    def test_surprise_max_limit(self):
        """Test surprise respects max_surprise."""
        from aragora.memory.surprise import calculate_surprise

        surprise = calculate_surprise(
            actual=1.0, expected=0.0, scale_factor=2.0, max_surprise=0.5
        )
        assert surprise == pytest.approx(0.5)


class TestCalculateBaseRate:
    """Test calculate_base_rate function."""

    def test_base_rate_no_observations(self):
        """Test base rate with no observations returns prior."""
        from aragora.memory.surprise import calculate_base_rate

        rate = calculate_base_rate(success_count=0, failure_count=0)
        assert rate == pytest.approx(0.5)  # Default prior

    def test_base_rate_some_observations(self):
        """Test base rate with observations."""
        from aragora.memory.surprise import calculate_base_rate

        # 8 successes, 2 failures = 80% raw rate
        rate = calculate_base_rate(success_count=8, failure_count=2)
        # With Bayesian smoothing, should be close to 0.75
        assert 0.7 < rate < 0.85

    def test_base_rate_custom_prior(self):
        """Test base rate with custom prior."""
        from aragora.memory.surprise import calculate_base_rate

        # No observations but prior of 0.3
        rate = calculate_base_rate(success_count=0, failure_count=0, prior=0.3)
        assert rate == pytest.approx(0.3)

    def test_base_rate_smoothing_effect(self):
        """Test that smoothing pulls extreme rates toward prior."""
        from aragora.memory.surprise import calculate_base_rate

        # 1 success, 0 failures = 100% raw rate
        rate = calculate_base_rate(success_count=1, failure_count=0)
        # But with smoothing, should be less than 1.0
        assert rate < 1.0
        assert rate > 0.5


class TestUpdateSurpriseEMA:
    """Test update_surprise_ema function."""

    def test_ema_basic_update(self):
        """Test basic EMA update."""
        from aragora.memory.surprise import update_surprise_ema

        # Old surprise 0.5, new surprise 0.8, default alpha 0.3
        result = update_surprise_ema(old_surprise=0.5, new_surprise=0.8)
        expected = 0.5 * 0.7 + 0.8 * 0.3  # 0.35 + 0.24 = 0.59
        assert result == pytest.approx(expected)

    def test_ema_high_alpha(self):
        """Test EMA with high alpha favors new value."""
        from aragora.memory.surprise import update_surprise_ema

        result = update_surprise_ema(old_surprise=0.5, new_surprise=0.8, alpha=0.9)
        # Should be close to new value
        assert result > 0.7

    def test_ema_low_alpha(self):
        """Test EMA with low alpha favors old value."""
        from aragora.memory.surprise import update_surprise_ema

        result = update_surprise_ema(old_surprise=0.5, new_surprise=0.8, alpha=0.1)
        # Should be close to old value
        assert result < 0.6


class TestCalculateCombinedSurprise:
    """Test calculate_combined_surprise function."""

    def test_combined_surprise_no_agent_error(self):
        """Test combined surprise without agent error."""
        from aragora.memory.surprise import calculate_combined_surprise

        result = calculate_combined_surprise(
            success_surprise=0.7,
            agent_prediction_error=None,
        )
        assert result == pytest.approx(0.7)

    def test_combined_surprise_with_agent_error(self):
        """Test combined surprise with agent error."""
        from aragora.memory.surprise import calculate_combined_surprise

        result = calculate_combined_surprise(
            success_surprise=0.7,
            agent_prediction_error=0.5,
        )
        # Default weights: 0.7 * 0.7 + 0.3 * 0.5 = 0.49 + 0.15 = 0.64
        expected = 0.7 * 0.7 + 0.3 * 0.5
        assert result == pytest.approx(expected)

    def test_combined_surprise_capped(self):
        """Test combined surprise is capped at 1.0."""
        from aragora.memory.surprise import calculate_combined_surprise

        result = calculate_combined_surprise(
            success_surprise=0.9,
            agent_prediction_error=0.9,
        )
        assert result <= 1.0


class TestCategoryStats:
    """Test CategoryStats dataclass."""

    def test_category_stats_defaults(self):
        """Test CategoryStats default values."""
        from aragora.memory.surprise import CategoryStats

        stats = CategoryStats()
        assert stats.success_count == 0
        assert stats.failure_count == 0
        assert stats.current_surprise == 0.0

    def test_category_stats_total(self):
        """Test CategoryStats total property."""
        from aragora.memory.surprise import CategoryStats

        stats = CategoryStats(success_count=5, failure_count=3)
        assert stats.total == 8

    def test_category_stats_success_rate(self):
        """Test CategoryStats success_rate property."""
        from aragora.memory.surprise import CategoryStats

        stats = CategoryStats(success_count=8, failure_count=2)
        assert stats.success_rate == pytest.approx(0.8)

    def test_category_stats_success_rate_empty(self):
        """Test CategoryStats success_rate with no data."""
        from aragora.memory.surprise import CategoryStats

        stats = CategoryStats()
        assert stats.success_rate == pytest.approx(0.5)  # Default


class TestSurpriseScorer:
    """Test SurpriseScorer class."""

    def test_scorer_initialization(self):
        """Test scorer initialization."""
        from aragora.memory.surprise import SurpriseScorer

        scorer = SurpriseScorer()
        assert scorer.alpha == pytest.approx(0.3)
        assert len(scorer.get_all_categories()) == 0

    def test_scorer_score_outcome_new_category(self):
        """Test scoring outcome for new category."""
        from aragora.memory.surprise import SurpriseScorer

        scorer = SurpriseScorer()
        surprise = scorer.score_outcome("type_errors", is_success=True)

        # First outcome against prior of 0.5, success -> surprise = 0.5
        assert 0 <= surprise <= 1.0

        stats = scorer.get_category_stats("type_errors")
        assert stats is not None
        assert stats.success_count == 1
        assert stats.failure_count == 0

    def test_scorer_multiple_outcomes(self):
        """Test scoring multiple outcomes."""
        from aragora.memory.surprise import SurpriseScorer

        scorer = SurpriseScorer()

        # Score several successes
        for _ in range(5):
            scorer.score_outcome("bugs", is_success=True)

        stats = scorer.get_category_stats("bugs")
        assert stats.success_count == 5
        assert stats.failure_count == 0
        assert stats.success_rate == 1.0

    def test_scorer_with_agent_error(self):
        """Test scoring with agent prediction error."""
        from aragora.memory.surprise import SurpriseScorer

        scorer = SurpriseScorer()
        surprise = scorer.score_outcome(
            "type_errors",
            is_success=True,
            agent_prediction_error=0.3,
        )
        assert 0 <= surprise <= 1.0

    def test_scorer_get_category_surprise(self):
        """Test getting surprise for category."""
        from aragora.memory.surprise import SurpriseScorer

        scorer = SurpriseScorer()
        scorer.score_outcome("test", is_success=True)

        surprise = scorer.get_category_surprise("test")
        assert surprise > 0

        # Unknown category returns 0
        assert scorer.get_category_surprise("unknown") == 0.0

    def test_scorer_reset_category(self):
        """Test resetting a category."""
        from aragora.memory.surprise import SurpriseScorer

        scorer = SurpriseScorer()
        scorer.score_outcome("test", is_success=True)

        assert scorer.get_category_stats("test") is not None
        scorer.reset_category("test")
        assert scorer.get_category_stats("test") is None

    def test_scorer_reset_all(self):
        """Test resetting all categories."""
        from aragora.memory.surprise import SurpriseScorer

        scorer = SurpriseScorer()
        scorer.score_outcome("cat1", is_success=True)
        scorer.score_outcome("cat2", is_success=False)

        assert len(scorer.get_all_categories()) == 2
        scorer.reset_all()
        assert len(scorer.get_all_categories()) == 0


class TestCalculateSurpriseFromDbRow:
    """Test calculate_surprise_from_db_row function."""

    def test_calculate_from_db_row(self):
        """Test surprise calculation from database row."""
        from aragora.memory.surprise import calculate_surprise_from_db_row

        # 8 successes, 2 failures, current success
        surprise = calculate_surprise_from_db_row(
            success_count=8,
            failure_count=2,
            is_success=True,
            old_surprise=0.5,
        )
        assert 0 <= surprise <= 1.0

    def test_calculate_from_db_row_failure(self):
        """Test surprise for failure outcome."""
        from aragora.memory.surprise import calculate_surprise_from_db_row

        # High success rate, but got failure = high surprise
        surprise = calculate_surprise_from_db_row(
            success_count=9,
            failure_count=1,
            is_success=False,
            old_surprise=0.1,
        )
        # Should have increased surprise
        assert surprise > 0.1


class TestCreateDbBaseRateCalculator:
    """Test create_db_base_rate_calculator function."""

    def test_create_calculator(self):
        """Test creating a database base rate calculator."""
        from aragora.memory.surprise import create_db_base_rate_calculator

        # Mock query function
        def mock_query(category: str) -> tuple[int, int]:
            if category == "known":
                return (8, 2)  # 80% success
            return (0, 0)  # Unknown category

        calc = create_db_base_rate_calculator(mock_query)

        # Test known category
        rate = calc("known")
        assert 0.7 < rate < 0.85

        # Test unknown category (returns prior)
        rate = calc("unknown")
        assert rate == pytest.approx(0.5)
