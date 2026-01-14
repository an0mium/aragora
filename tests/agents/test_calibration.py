"""Tests for calibration auto-tuning features."""

import math
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from aragora.agents.calibration import (
    DEFAULT_TEMPERATURE,
    MIN_PREDICTIONS_FOR_TUNING,
    CalibrationBucket,
    CalibrationSummary,
    CalibrationTracker,
    TemperatureParams,
    _logit,
    _sigmoid,
    adjust_agent_confidence,
    temperature_scale,
)


class TestTemperatureScaling:
    """Tests for temperature scaling functions."""

    def test_logit_and_sigmoid_are_inverses(self):
        """Logit and sigmoid should be inverse functions."""
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert abs(_sigmoid(_logit(p)) - p) < 1e-6

    def test_logit_handles_extreme_values(self):
        """Logit should handle values near 0 and 1."""
        # Should not raise errors
        _logit(0.0001)
        _logit(0.9999)
        # Very extreme values get clamped
        assert _logit(0.0) > -20
        assert _logit(1.0) < 20

    def test_sigmoid_handles_extreme_values(self):
        """Sigmoid should handle large positive and negative values."""
        assert abs(_sigmoid(100) - 1.0) < 1e-6
        assert abs(_sigmoid(-100) - 0.0) < 1e-6

    def test_temperature_scale_identity(self):
        """Temperature=1 should leave values unchanged (approximately)."""
        for conf in [0.2, 0.5, 0.8]:
            scaled = temperature_scale(conf, 1.0)
            assert abs(scaled - conf) < 0.01

    def test_temperature_scale_compression(self):
        """Temperature>1 should compress confidence toward 0.5."""
        # High confidence gets reduced
        high_conf = temperature_scale(0.9, 1.5)
        assert high_conf < 0.9

        # Low confidence gets increased
        low_conf = temperature_scale(0.1, 1.5)
        assert low_conf > 0.1

    def test_temperature_scale_expansion(self):
        """Temperature<1 should expand confidence away from 0.5."""
        # High confidence gets increased
        high_conf = temperature_scale(0.9, 0.7)
        assert high_conf > 0.9

        # Low confidence gets decreased
        low_conf = temperature_scale(0.1, 0.7)
        assert low_conf < 0.1

    def test_temperature_scale_clamps_output(self):
        """Output should be clamped to [0.05, 0.95]."""
        # Very extreme values should be clamped
        assert temperature_scale(0.99, 0.5) <= 0.95
        assert temperature_scale(0.01, 0.5) >= 0.05

    def test_temperature_scale_handles_zero_temperature(self):
        """Zero temperature should use default (no change)."""
        conf = 0.7
        result = temperature_scale(conf, 0)
        # Should use DEFAULT_TEMPERATURE=1.0
        assert abs(result - conf) < 0.01


class TestTemperatureParams:
    """Tests for TemperatureParams dataclass."""

    def test_default_values(self):
        """Default params should use default temperature."""
        params = TemperatureParams()
        assert params.temperature == DEFAULT_TEMPERATURE
        assert params.domain_temperatures == {}
        assert params.last_tuned is None
        assert params.predictions_at_tune == 0

    def test_get_temperature_global(self):
        """Should return global temperature when no domain match."""
        params = TemperatureParams(temperature=1.5)
        assert params.get_temperature() == 1.5
        assert params.get_temperature("unknown_domain") == 1.5

    def test_get_temperature_domain_specific(self):
        """Should return domain-specific temperature when available."""
        params = TemperatureParams(
            temperature=1.0,
            domain_temperatures={"security": 0.8, "performance": 1.2},
        )
        assert params.get_temperature("security") == 0.8
        assert params.get_temperature("performance") == 1.2
        assert params.get_temperature("other") == 1.0

    def test_is_stale_no_tuning(self):
        """Should be stale if never tuned."""
        params = TemperatureParams()
        assert params.is_stale(100)

    def test_is_stale_prediction_increase(self):
        """Should be stale if predictions increased by 50%."""
        params = TemperatureParams(
            last_tuned=datetime.now(),
            predictions_at_tune=100,
        )
        # Not stale at 140 predictions (40% increase)
        assert not params.is_stale(140)
        # Stale at 160 predictions (60% increase)
        assert params.is_stale(160)

    def test_is_stale_age(self):
        """Should be stale if older than max_age_hours."""
        params = TemperatureParams(
            last_tuned=datetime.now() - timedelta(hours=25),
            predictions_at_tune=100,
        )
        assert params.is_stale(100, max_age_hours=24)


class TestCalibrationSummaryWithTemperature:
    """Tests for CalibrationSummary with temperature scaling."""

    def test_adjust_confidence_uses_temperature(self):
        """Should use temperature scaling when available."""
        summary = CalibrationSummary(
            agent="test",
            total_predictions=100,
            temperature_params=TemperatureParams(temperature=1.5),
        )
        raw = 0.9
        adjusted = summary.adjust_confidence(raw)
        # With T=1.5, high confidence should be compressed
        assert adjusted < raw

    def test_adjust_confidence_fallback_linear(self):
        """Should fall back to linear adjustment when use_temperature=False."""
        # Create buckets showing overconfidence (accuracy < expected)
        buckets = [
            CalibrationBucket(0.7, 0.8, 50, 30),  # 60% accuracy vs 75% expected
            CalibrationBucket(0.8, 0.9, 50, 35),  # 70% accuracy vs 85% expected
        ]
        summary = CalibrationSummary(
            agent="test",
            total_predictions=100,
            total_correct=65,
            brier_score=0.2,
            ece=0.15,
            buckets=buckets,
        )
        raw = 0.8
        # The linear adjustment is based on get_confidence_adjustment()
        # which uses ECE to determine adjustment magnitude
        adjustment = summary.get_confidence_adjustment()
        adjusted = summary.adjust_confidence(raw, use_temperature=False)
        # With ECE=0.15, should have some adjustment
        expected = max(0.05, min(0.95, raw * adjustment))
        assert abs(adjusted - expected) < 0.01

    def test_adjust_confidence_domain_specific(self):
        """Should use domain-specific temperature when available."""
        summary = CalibrationSummary(
            agent="test",
            total_predictions=100,
            temperature_params=TemperatureParams(
                temperature=1.2,  # Slightly compressed global
                domain_temperatures={"security": 1.5},  # More compressed for security
            ),
        )
        raw = 0.9

        # Global temperature (1.2) - some compression
        global_adjusted = summary.adjust_confidence(raw, domain="general")
        assert global_adjusted < raw  # Compressed

        # Domain-specific temperature (1.5) - more compressed
        domain_adjusted = summary.adjust_confidence(raw, domain="security")
        assert domain_adjusted < global_adjusted  # Even more compressed


class TestCalibrationTrackerAutoTune:
    """Tests for CalibrationTracker auto-tuning methods."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a CalibrationTracker with temp database."""
        db_path = str(tmp_path / "test_calibration.db")
        return CalibrationTracker(db_path=db_path)

    def test_record_and_retrieve_predictions(self, tracker):
        """Basic test that predictions can be recorded and retrieved."""
        tracker.record_prediction("agent1", 0.8, True, "general")
        tracker.record_prediction("agent1", 0.6, False, "general")

        summary = tracker.get_calibration_summary("agent1")
        assert summary.total_predictions == 2
        assert summary.total_correct == 1

    def test_recency_weighted_predictions(self, tracker):
        """Should return predictions with recency weights."""
        # Record predictions
        for i in range(10):
            tracker.record_prediction("agent1", 0.7, i % 2 == 0, "general")

        predictions = tracker.get_recency_weighted_predictions("agent1")
        assert len(predictions) == 10

        # Most recent should have highest weight
        weights = [w for _, _, w in predictions]
        assert weights[0] >= weights[-1]

    def test_compute_optimal_temperature_insufficient_data(self, tracker):
        """Should return default temperature with insufficient data."""
        # Only 5 predictions (below MIN_PREDICTIONS_FOR_TUNING)
        for _ in range(5):
            tracker.record_prediction("agent1", 0.7, True, "general")

        temp = tracker.compute_optimal_temperature("agent1")
        assert temp == DEFAULT_TEMPERATURE

    def test_compute_optimal_temperature_with_data(self, tracker):
        """Should compute optimal temperature with sufficient data."""
        # Add overconfident predictions (high confidence, often wrong)
        for _ in range(30):
            tracker.record_prediction("agent1", 0.9, False, "general")
        for _ in range(10):
            tracker.record_prediction("agent1", 0.9, True, "general")

        temp = tracker.compute_optimal_temperature("agent1")
        # Should suggest higher temperature to compress overconfidence
        assert temp > 1.0

    def test_auto_tune_agent(self, tracker):
        """Should auto-tune agent and store parameters."""
        # Add predictions
        for _ in range(50):
            tracker.record_prediction("agent1", 0.8, False, "general")
        for _ in range(10):
            tracker.record_prediction("agent1", 0.8, True, "general")

        params = tracker.auto_tune_agent("agent1")

        assert params.temperature != DEFAULT_TEMPERATURE
        assert params.last_tuned is not None
        assert params.predictions_at_tune == 60

        # Verify it was stored
        retrieved = tracker.get_temperature_params("agent1")
        assert retrieved.temperature == params.temperature

    def test_auto_tune_domain_specific(self, tracker):
        """Should compute domain-specific temperatures."""
        # Add predictions in different domains
        for _ in range(30):
            tracker.record_prediction("agent1", 0.9, False, "security")
            tracker.record_prediction("agent1", 0.6, True, "performance")

        params = tracker.auto_tune_agent("agent1", tune_domains=True)

        # Should have domain-specific temperatures
        assert "security" in params.domain_temperatures
        assert "performance" in params.domain_temperatures

    def test_auto_tune_respects_staleness(self, tracker):
        """Should not retune if not stale."""
        # Add predictions and tune
        for _ in range(50):
            tracker.record_prediction("agent1", 0.7, True, "general")

        params1 = tracker.auto_tune_agent("agent1")
        first_tune = params1.last_tuned

        # Try to tune again without force - should not change
        params2 = tracker.auto_tune_agent("agent1", force=False)
        assert params2.last_tuned == first_tune

        # Force tune - should update
        params3 = tracker.auto_tune_agent("agent1", force=True)
        assert params3.last_tuned >= first_tune

    def test_save_and_load_temperature_params(self, tracker):
        """Should correctly persist temperature parameters."""
        params = TemperatureParams(
            temperature=1.3,
            domain_temperatures={"security": 0.9, "general": 1.1},
            last_tuned=datetime.now(),
            predictions_at_tune=100,
        )

        tracker.save_temperature_params("agent1", params)
        loaded = tracker.get_temperature_params("agent1")

        assert loaded.temperature == params.temperature
        assert loaded.domain_temperatures == params.domain_temperatures
        assert loaded.predictions_at_tune == params.predictions_at_tune

    def test_delete_agent_removes_temperature_params(self, tracker):
        """Deleting agent should also remove temperature params."""
        tracker.record_prediction("agent1", 0.8, True, "general")
        tracker.save_temperature_params(
            "agent1", TemperatureParams(temperature=1.5)
        )

        tracker.delete_agent_data("agent1")

        # Both predictions and params should be gone
        summary = tracker.get_calibration_summary("agent1")
        assert summary.total_predictions == 0

        params = tracker.get_temperature_params("agent1")
        assert params.temperature == DEFAULT_TEMPERATURE


class TestAdjustAgentConfidence:
    """Tests for the utility function adjust_agent_confidence."""

    def test_none_summary_returns_original(self):
        """Should return original confidence if no summary."""
        result = adjust_agent_confidence(0.8, None)
        assert result == 0.8

    def test_with_domain(self):
        """Should pass domain to summary adjustment."""
        summary = CalibrationSummary(
            agent="test",
            total_predictions=100,
            temperature_params=TemperatureParams(
                temperature=1.2,  # Some compression globally
                domain_temperatures={"security": 1.5},  # More compression for security
            ),
        )

        # Domain-specific should be different
        general = adjust_agent_confidence(0.9, summary, domain="general")
        security = adjust_agent_confidence(0.9, summary, domain="security")

        assert security < general  # T=1.5 compresses more than T=1.2


class TestCalibrationBucket:
    """Tests for CalibrationBucket."""

    def test_accuracy_calculation(self):
        """Should calculate accuracy correctly."""
        bucket = CalibrationBucket(0.7, 0.8, 100, 80)
        assert bucket.accuracy == 0.8

    def test_accuracy_empty_bucket(self):
        """Should return 0 for empty bucket."""
        bucket = CalibrationBucket(0.7, 0.8, 0, 0)
        assert bucket.accuracy == 0.0

    def test_expected_accuracy(self):
        """Should return bucket midpoint."""
        bucket = CalibrationBucket(0.7, 0.8, 10, 7)
        assert bucket.expected_accuracy == 0.75

    def test_calibration_error(self):
        """Should compute absolute error correctly."""
        # Perfect calibration in 0.7-0.8 bucket with 75% accuracy
        bucket = CalibrationBucket(0.7, 0.8, 100, 75)
        assert bucket.calibration_error == 0.0

        # Overconfident - 50% accuracy in 0.7-0.8 bucket
        bucket = CalibrationBucket(0.7, 0.8, 100, 50)
        assert bucket.calibration_error == 0.25


class TestCalibrationIntegration:
    """Integration tests for the full calibration flow."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a CalibrationTracker with temp database."""
        db_path = str(tmp_path / "test_calibration.db")
        return CalibrationTracker(db_path=db_path)

    def test_full_calibration_flow(self, tracker):
        """Test complete flow: record -> tune -> adjust."""
        agent = "claude-api"

        # Simulate an overconfident agent
        # High confidence (0.85-0.95) predictions, only 60% correct
        for i in range(60):
            tracker.record_prediction(agent, 0.9, i < 36, "general")

        # Auto-tune
        params = tracker.auto_tune_agent(agent)

        # Temperature should be > 1 to compress overconfidence
        assert params.temperature > 1.0

        # Get summary with temperature
        summary = tracker.get_calibration_summary(agent)

        # Adjusted confidence should be lower than raw
        raw_conf = 0.9
        adjusted = summary.adjust_confidence(raw_conf)
        assert adjusted < raw_conf

    def test_domain_specific_calibration(self, tracker):
        """Test domain-specific calibration tuning."""
        agent = "test-agent"

        # Security domain: underconfident (70% correct at 50% confidence)
        for i in range(40):
            tracker.record_prediction(agent, 0.5, i < 28, "security")

        # Performance domain: overconfident (40% correct at 80% confidence)
        for i in range(40):
            tracker.record_prediction(agent, 0.8, i < 16, "performance")

        # Auto-tune with domains
        params = tracker.auto_tune_agent(agent, tune_domains=True)

        # Should have different temperatures per domain
        security_temp = params.get_temperature("security")
        performance_temp = params.get_temperature("performance")

        # Security should have lower temp (expand confidence)
        # Performance should have higher temp (compress confidence)
        assert performance_temp > security_temp
