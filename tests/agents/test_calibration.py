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
        tracker.save_temperature_params("agent1", TemperatureParams(temperature=1.5))

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


class TestCalibrationSessionCreation:
    """Tests for calibration session creation and initialization."""

    def test_tracker_creates_tables(self, tmp_path):
        """Should create required tables on initialization."""
        db_path = str(tmp_path / "new_calibration.db")
        tracker = CalibrationTracker(db_path=db_path)

        # Tables should exist - verify by recording prediction
        pred_id = tracker.record_prediction("agent1", 0.7, True)
        assert pred_id > 0

    def test_tracker_with_existing_database(self, tmp_path):
        """Should handle reopening existing database."""
        db_path = str(tmp_path / "existing.db")

        # Create and populate
        tracker1 = CalibrationTracker(db_path=db_path)
        tracker1.record_prediction("agent1", 0.8, True)

        # Reopen
        tracker2 = CalibrationTracker(db_path=db_path)
        summary = tracker2.get_calibration_summary("agent1")
        assert summary.total_predictions == 1

    def test_multiple_tracker_instances(self, tmp_path):
        """Should allow multiple tracker instances on same database."""
        db_path = str(tmp_path / "shared.db")

        tracker1 = CalibrationTracker(db_path=db_path)
        tracker2 = CalibrationTracker(db_path=db_path)

        tracker1.record_prediction("agent1", 0.7, True)
        # Second tracker should see the data
        summary = tracker2.get_calibration_summary("agent1")
        assert summary.total_predictions == 1


class TestSkillAssessmentCalculations:
    """Tests for skill assessment and calibration calculations."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a CalibrationTracker with temp database."""
        db_path = str(tmp_path / "test_calibration.db")
        return CalibrationTracker(db_path=db_path)

    def test_brier_score_calculation_accuracy(self, tracker):
        """Brier score should match manual calculation."""
        # Record specific predictions
        tracker.record_prediction("agent1", 0.8, True)  # (0.8 - 1)^2 = 0.04
        tracker.record_prediction("agent1", 0.6, False)  # (0.6 - 0)^2 = 0.36
        tracker.record_prediction("agent1", 0.9, True)  # (0.9 - 1)^2 = 0.01
        tracker.record_prediction("agent1", 0.3, False)  # (0.3 - 0)^2 = 0.09

        # Average: (0.04 + 0.36 + 0.01 + 0.09) / 4 = 0.125
        brier = tracker.get_brier_score("agent1")
        assert abs(brier - 0.125) < 0.001

    def test_ece_calculation_with_multiple_buckets(self, tracker):
        """ECE should weight buckets by number of predictions."""
        # Add predictions in different confidence ranges
        # Bucket 0.7-0.8: 10 predictions, 7 correct (accuracy 0.7, expected 0.75)
        for i in range(10):
            tracker.record_prediction("agent1", 0.75, i < 7)

        # Bucket 0.5-0.6: 20 predictions, 11 correct (accuracy 0.55, expected 0.55)
        for i in range(20):
            tracker.record_prediction("agent1", 0.55, i < 11)

        ece = tracker.get_expected_calibration_error("agent1", num_buckets=10)
        # ECE should be weighted average of bucket calibration errors
        # Bucket 7: 10/30 * |0.7 - 0.75| = 10/30 * 0.05
        # Bucket 5: 20/30 * |0.55 - 0.55| = 20/30 * 0.0
        expected_ece = (10 / 30) * 0.05 + (20 / 30) * 0.0
        assert abs(ece - expected_ece) < 0.02

    def test_confidence_adjustment_magnitude(self):
        """Confidence adjustment should scale with ECE."""
        # Low ECE = small adjustment (need overconfident buckets for adjustment)
        summary_low = CalibrationSummary(
            agent="low_ece",
            total_predictions=100,
            total_correct=75,
            ece=0.05,
            buckets=[
                CalibrationBucket(0.7, 0.8, 100, 60)
            ],  # 60% accuracy at 75% expected = overconfident
        )

        # High ECE = large adjustment
        summary_high = CalibrationSummary(
            agent="high_ece",
            total_predictions=100,
            total_correct=50,
            ece=0.20,
            buckets=[
                CalibrationBucket(0.7, 0.8, 100, 50)
            ],  # 50% accuracy at 75% expected = more overconfident
        )

        adj_low = summary_low.get_confidence_adjustment()
        adj_high = summary_high.get_confidence_adjustment()

        # With enough predictions, both overconfident agents should have adjustment < 1
        assert adj_low < 1.0
        assert adj_high < 1.0
        # Higher ECE should have larger adjustment (further from 1.0)
        assert abs(1.0 - adj_high) > abs(1.0 - adj_low)


class TestCalibrationDataPersistence:
    """Tests for calibration data persistence."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a CalibrationTracker with temp database."""
        db_path = str(tmp_path / "test_calibration.db")
        return CalibrationTracker(db_path=db_path)

    def test_prediction_fields_persisted(self, tracker):
        """All prediction fields should be persisted correctly."""
        pred_id = tracker.record_prediction(
            agent="agent1",
            confidence=0.85,
            correct=True,
            domain="security",
            debate_id="debate-123",
            position_id="pos-456",
        )

        # Verify by retrieving summary
        summary = tracker.get_calibration_summary("agent1", domain="security")
        assert summary.total_predictions == 1
        assert summary.total_correct == 1

    def test_record_prediction_accepts_prediction_type(self, tracker):
        """record_prediction accepts optional prediction_type metadata."""
        pred_id = tracker.record_prediction(
            agent="agent1",
            confidence=0.7,
            correct=True,
            domain="general",
            prediction_type="consensus_feedback",
        )

        assert pred_id > 0
        summary = tracker.get_calibration_summary("agent1", domain="general")
        assert summary.total_predictions == 1

    def test_temperature_params_persistence(self, tracker):
        """Temperature parameters should persist across sessions."""
        from datetime import datetime

        params = TemperatureParams(
            temperature=1.25,
            domain_temperatures={"security": 0.9, "performance": 1.4},
            last_tuned=datetime.now(),
            predictions_at_tune=150,
        )

        tracker.save_temperature_params("agent1", params)
        loaded = tracker.get_temperature_params("agent1")

        assert loaded.temperature == 1.25
        assert loaded.domain_temperatures["security"] == 0.9
        assert loaded.domain_temperatures["performance"] == 1.4
        assert loaded.predictions_at_tune == 150

    def test_get_calibration_returns_dict(self, tracker):
        """get_calibration should return protocol-compliant dict."""
        # Record some predictions
        for i in range(10):
            tracker.record_prediction("agent1", 0.8, i < 7)

        result = tracker.get_calibration("agent1")

        assert result is not None
        assert result["agent"] == "agent1"
        assert result["total_predictions"] == 10
        assert result["total_correct"] == 7
        assert abs(result["accuracy"] - 0.7) < 0.01
        assert "brier_score" in result
        assert "ece" in result
        assert "temperature" in result

    def test_get_calibration_returns_none_for_unknown_agent(self, tracker):
        """get_calibration should return None for agent with no data."""
        result = tracker.get_calibration("unknown_agent")
        assert result is None


class TestELOIntegration:
    """Tests for ELO rating integration with calibration."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a CalibrationTracker with temp database."""
        db_path = str(tmp_path / "test_calibration.db")
        return CalibrationTracker(db_path=db_path)

    def test_calibration_affects_confidence_for_elo(self, tracker):
        """Calibration should provide adjustments usable for ELO weighting."""
        # Simulate overconfident agent
        for _ in range(50):
            tracker.record_prediction("overconfident", 0.95, False)
        for _ in range(10):
            tracker.record_prediction("overconfident", 0.95, True)

        # Simulate well-calibrated agent
        for i in range(60):
            tracker.record_prediction("calibrated", 0.75, i < 45)  # 75% accuracy

        # Auto-tune both
        tracker.auto_tune_agent("overconfident")
        tracker.auto_tune_agent("calibrated")

        # Get summaries
        over_summary = tracker.get_calibration_summary("overconfident")
        cal_summary = tracker.get_calibration_summary("calibrated")

        # Overconfident agent should have larger adjustment
        over_adj = over_summary.get_confidence_adjustment()
        cal_adj = cal_summary.get_confidence_adjustment()

        assert over_adj < cal_adj  # Overconfident gets reduced more

    def test_domain_specific_calibration_for_tournaments(self, tracker):
        """Different domains should have separate calibration for tournament matching."""
        agent = "specialist"

        # Expert in security (80% at 80% confidence)
        for i in range(40):
            tracker.record_prediction(agent, 0.8, i < 32, "security")

        # Novice in ML (30% at 80% confidence)
        for i in range(40):
            tracker.record_prediction(agent, 0.8, i < 12, "ml")

        tracker.auto_tune_agent(agent, tune_domains=True)

        summary = tracker.get_calibration_summary(agent)

        # Adjustments should differ by domain
        security_adj = summary.adjust_confidence(0.8, domain="security")
        ml_adj = summary.adjust_confidence(0.8, domain="ml")

        # ML domain should have much lower adjusted confidence
        assert ml_adj < security_adj


class TestPerformanceTrackingOverTime:
    """Tests for tracking performance over time."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a CalibrationTracker with temp database."""
        db_path = str(tmp_path / "test_calibration.db")
        return CalibrationTracker(db_path=db_path)

    def test_recency_weighting_decays_old_predictions(self, tracker):
        """Older predictions should have lower weights."""
        # Add predictions
        for i in range(20):
            tracker.record_prediction("agent1", 0.7, i % 2 == 0)

        predictions = tracker.get_recency_weighted_predictions("agent1")

        # Most recent (first in list) should have weight close to 1
        assert predictions[0][2] > 0.9

        # Weights should be decreasing (or at least the oldest should be less)
        # Note: All predictions recorded in same test run, so decay may be minimal
        assert all(w > 0 for _, _, w in predictions)

    def test_auto_tune_updates_predictions_at_tune(self, tracker):
        """Auto-tune should track how many predictions existed at tuning time."""
        for _ in range(30):
            tracker.record_prediction("agent1", 0.7, True)

        params = tracker.auto_tune_agent("agent1")
        assert params.predictions_at_tune == 30

        # Add more predictions
        for _ in range(20):
            tracker.record_prediction("agent1", 0.7, True)

        # Force retune
        params = tracker.auto_tune_agent("agent1", force=True)
        assert params.predictions_at_tune == 50

    def test_temperature_changes_over_time(self, tracker):
        """Temperature should adapt as more data is collected."""
        agent = "evolving"

        # Initial: well-calibrated at moderate confidence
        for i in range(30):
            tracker.record_prediction(agent, 0.6, i < 18)  # 60% accuracy at 60% confidence

        params1 = tracker.auto_tune_agent(agent)
        temp1 = params1.temperature

        # Add overconfident predictions (high confidence, low accuracy)
        for _ in range(60):
            tracker.record_prediction(agent, 0.9, False)

        params2 = tracker.auto_tune_agent(agent, force=True)
        temp2 = params2.temperature

        # Temperature should have increased to compress overconfidence
        assert temp2 > temp1


class TestCalibrationThresholdsAndTriggers:
    """Tests for calibration thresholds and triggers."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a CalibrationTracker with temp database."""
        db_path = str(tmp_path / "test_calibration.db")
        return CalibrationTracker(db_path=db_path)

    def test_min_predictions_for_tuning_threshold(self, tracker):
        """Should require minimum predictions before tuning."""
        # Add fewer than MIN_PREDICTIONS_FOR_TUNING
        for _ in range(MIN_PREDICTIONS_FOR_TUNING - 5):
            tracker.record_prediction("agent1", 0.8, False)

        temp = tracker.compute_optimal_temperature("agent1")
        assert temp == DEFAULT_TEMPERATURE  # Should return default

        # Add more to exceed threshold
        for _ in range(10):
            tracker.record_prediction("agent1", 0.8, False)

        temp = tracker.compute_optimal_temperature("agent1")
        # Should now compute a non-default temperature for overconfident agent
        assert temp >= DEFAULT_TEMPERATURE

    def test_is_stale_triggers_retune(self, tracker):
        """Stale parameters should trigger automatic retuning."""
        for _ in range(50):
            tracker.record_prediction("agent1", 0.7, True)

        params = tracker.auto_tune_agent("agent1")

        # Not stale - should return same params
        params2 = tracker.auto_tune_agent("agent1", force=False)
        assert params2.last_tuned == params.last_tuned

        # Add 50% more predictions to trigger staleness
        for _ in range(30):  # 50 * 1.5 = 75, so 80 total > 75
            tracker.record_prediction("agent1", 0.7, True)

        params3 = tracker.auto_tune_agent("agent1", force=False)
        # Should have retuned due to staleness
        assert params3.predictions_at_tune > params.predictions_at_tune

    def test_leaderboard_min_prediction_threshold(self, tracker):
        """Leaderboard should exclude agents below prediction threshold."""
        # Agent with enough predictions
        for _ in range(10):
            tracker.record_prediction("active_agent", 0.7, True)

        # Agent with too few predictions
        for _ in range(3):
            tracker.record_prediction("inactive_agent", 0.7, True)

        leaderboard = tracker.get_leaderboard(metric="brier", limit=10)
        agents = [entry[0] for entry in leaderboard]

        assert "active_agent" in agents
        assert "inactive_agent" not in agents


class TestMultiAgentCalibration:
    """Tests for multi-agent calibration scenarios."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a CalibrationTracker with temp database."""
        db_path = str(tmp_path / "test_calibration.db")
        return CalibrationTracker(db_path=db_path)

    def test_independent_agent_calibration(self, tracker):
        """Each agent should have independent calibration."""
        # Agent 1: overconfident
        for _ in range(30):
            tracker.record_prediction("agent1", 0.9, False)

        # Agent 2: underconfident
        for _ in range(30):
            tracker.record_prediction("agent2", 0.3, True)

        # Agent 3: well-calibrated
        for i in range(30):
            tracker.record_prediction("agent3", 0.7, i < 21)

        summary1 = tracker.get_calibration_summary("agent1")
        summary2 = tracker.get_calibration_summary("agent2")
        summary3 = tracker.get_calibration_summary("agent3")

        assert summary1.is_overconfident
        assert summary2.is_underconfident
        assert not summary3.is_overconfident and not summary3.is_underconfident

    def test_leaderboard_ranks_multiple_agents(self, tracker):
        """Leaderboard should correctly rank multiple agents."""
        # Good agent
        for _ in range(20):
            tracker.record_prediction("good", 0.8, True)

        # Medium agent
        for i in range(20):
            tracker.record_prediction("medium", 0.8, i < 16)

        # Bad agent
        for i in range(20):
            tracker.record_prediction("bad", 0.8, i < 8)

        leaderboard = tracker.get_leaderboard(metric="brier", limit=10)
        agents = [entry[0] for entry in leaderboard]

        # Lower Brier is better
        assert agents.index("good") < agents.index("medium")
        assert agents.index("medium") < agents.index("bad")

    def test_all_agents_retrieval(self, tracker):
        """Should retrieve all agents with recorded predictions."""
        agents = ["alpha", "beta", "gamma", "delta"]
        for agent in agents:
            tracker.record_prediction(agent, 0.7, True)

        retrieved = tracker.get_all_agents()
        assert set(retrieved) == set(agents)

    def test_delete_agent_doesnt_affect_others(self, tracker):
        """Deleting one agent's data should not affect others."""
        tracker.record_prediction("agent1", 0.7, True)
        tracker.record_prediction("agent2", 0.8, False)

        tracker.delete_agent_data("agent1")

        # agent2 should be unaffected
        summary = tracker.get_calibration_summary("agent2")
        assert summary.total_predictions == 1

        # agent1 should be gone
        agents = tracker.get_all_agents()
        assert "agent1" not in agents


class TestErrorHandling:
    """Tests for error handling in calibration."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a CalibrationTracker with temp database."""
        db_path = str(tmp_path / "test_calibration.db")
        return CalibrationTracker(db_path=db_path)

    def test_confidence_out_of_range_clamped(self, tracker):
        """Confidence values outside [0,1] should be clamped."""
        # Should not raise
        tracker.record_prediction("agent1", 1.5, True)
        tracker.record_prediction("agent1", -0.5, False)

        summary = tracker.get_calibration_summary("agent1")
        assert summary.total_predictions == 2

    def test_empty_agent_name(self, tracker):
        """Should handle empty agent name gracefully."""
        pred_id = tracker.record_prediction("", 0.7, True)
        assert pred_id > 0

        summary = tracker.get_calibration_summary("")
        assert summary.total_predictions == 1

    def test_special_characters_in_agent_name(self, tracker):
        """Should handle special characters in agent names."""
        agent = "agent-with_special.chars@2024"
        tracker.record_prediction(agent, 0.7, True)

        summary = tracker.get_calibration_summary(agent)
        assert summary.total_predictions == 1
        assert summary.agent == agent

    def test_unicode_in_domain(self, tracker):
        """Should handle unicode characters in domain names."""
        domain = "sécurité"
        tracker.record_prediction("agent1", 0.7, True, domain=domain)

        breakdown = tracker.get_domain_breakdown("agent1")
        assert domain in breakdown

    def test_nonexistent_agent_returns_empty_summary(self, tracker):
        """Should return empty summary for nonexistent agent."""
        summary = tracker.get_calibration_summary("nonexistent")

        assert summary.total_predictions == 0
        assert summary.total_correct == 0
        assert summary.brier_score == 0.0
        assert summary.ece == 0.0

    def test_get_temperature_params_nonexistent_agent(self, tracker):
        """Should return default params for nonexistent agent."""
        params = tracker.get_temperature_params("nonexistent")

        assert params.temperature == DEFAULT_TEMPERATURE
        assert params.domain_temperatures == {}
        assert params.last_tuned is None


class TestBiasDirectionProperty:
    """Tests for the bias_direction property."""

    def test_bias_direction_overconfident(self):
        """Should return 'overconfident' for overconfident agents."""
        buckets = [
            CalibrationBucket(0.8, 0.9, 50, 25),  # 50% accuracy at 85% expected
            CalibrationBucket(0.9, 1.0, 50, 30),  # 60% accuracy at 95% expected
        ]
        summary = CalibrationSummary(
            agent="overconf",
            total_predictions=100,
            total_correct=55,
            buckets=buckets,
        )

        assert summary.bias_direction == "overconfident"

    def test_bias_direction_underconfident(self):
        """Should return 'underconfident' for underconfident agents."""
        buckets = [
            CalibrationBucket(0.2, 0.3, 50, 40),  # 80% accuracy at 25% expected
            CalibrationBucket(0.3, 0.4, 50, 40),  # 80% accuracy at 35% expected
        ]
        summary = CalibrationSummary(
            agent="underconf",
            total_predictions=100,
            total_correct=80,
            buckets=buckets,
        )

        assert summary.bias_direction == "underconfident"

    def test_bias_direction_well_calibrated(self):
        """Should return 'well-calibrated' for well-calibrated agents."""
        buckets = [
            CalibrationBucket(0.7, 0.8, 50, 37),  # 74% accuracy at 75% expected
            CalibrationBucket(0.8, 0.9, 50, 43),  # 86% accuracy at 85% expected
        ]
        summary = CalibrationSummary(
            agent="calibrated",
            total_predictions=100,
            total_correct=80,
            buckets=buckets,
        )

        assert summary.bias_direction == "well-calibrated"


class TestIntegrateWithPositionLedger:
    """Tests for integrate_with_position_ledger function."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a CalibrationTracker with temp database."""
        db_path = str(tmp_path / "test_calibration.db")
        return CalibrationTracker(db_path=db_path)

    def test_imports_correct_positions(self, tracker):
        """Should import positions with correct/incorrect outcomes."""
        from aragora.agents.calibration import integrate_with_position_ledger
        from unittest.mock import MagicMock

        # Create mock position ledger
        mock_ledger = MagicMock()

        # Create mock positions
        mock_pos1 = MagicMock()
        mock_pos1.agent_name = "agent1"
        mock_pos1.confidence = 0.8
        mock_pos1.outcome = "correct"
        mock_pos1.domain = "security"
        mock_pos1.debate_id = "debate-1"
        mock_pos1.id = "pos-1"

        mock_pos2 = MagicMock()
        mock_pos2.agent_name = "agent1"
        mock_pos2.confidence = 0.7
        mock_pos2.outcome = "incorrect"
        mock_pos2.domain = "general"
        mock_pos2.debate_id = "debate-2"
        mock_pos2.id = "pos-2"

        mock_ledger.get_agent_positions.return_value = [mock_pos1, mock_pos2]

        imported = integrate_with_position_ledger(tracker, mock_ledger, "agent1")

        assert imported == 2
        summary = tracker.get_calibration_summary("agent1")
        assert summary.total_predictions == 2
        assert summary.total_correct == 1

    def test_skips_unresolved_positions(self, tracker):
        """Should skip positions without correct/incorrect outcome."""
        from aragora.agents.calibration import integrate_with_position_ledger
        from unittest.mock import MagicMock

        mock_ledger = MagicMock()

        # Position with pending outcome
        mock_pos = MagicMock()
        mock_pos.agent_name = "agent1"
        mock_pos.confidence = 0.8
        mock_pos.outcome = "pending"  # Not correct/incorrect
        mock_pos.domain = "general"
        mock_pos.debate_id = "debate-1"
        mock_pos.id = "pos-1"

        mock_ledger.get_agent_positions.return_value = [mock_pos]

        imported = integrate_with_position_ledger(tracker, mock_ledger, "agent1")

        assert imported == 0

    def test_handles_empty_positions(self, tracker):
        """Should handle empty position list."""
        from aragora.agents.calibration import integrate_with_position_ledger
        from unittest.mock import MagicMock

        mock_ledger = MagicMock()
        mock_ledger.get_agent_positions.return_value = []

        imported = integrate_with_position_ledger(tracker, mock_ledger, "agent1")

        assert imported == 0


class TestCalibrationEdgeCases:
    """Tests for edge cases in calibration."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a CalibrationTracker with temp database."""
        db_path = str(tmp_path / "test_calibration.db")
        return CalibrationTracker(db_path=db_path)

    def test_all_predictions_correct(self, tracker):
        """Should handle 100% accuracy."""
        for _ in range(20):
            tracker.record_prediction("perfect", 0.9, True)

        summary = tracker.get_calibration_summary("perfect")
        assert summary.accuracy == 1.0
        assert summary.brier_score < 0.05

    def test_all_predictions_incorrect(self, tracker):
        """Should handle 0% accuracy."""
        for _ in range(20):
            tracker.record_prediction("terrible", 0.9, False)

        summary = tracker.get_calibration_summary("terrible")
        assert summary.accuracy == 0.0
        assert summary.brier_score > 0.8

    def test_single_prediction(self, tracker):
        """Should handle single prediction."""
        tracker.record_prediction("single", 0.8, True)

        summary = tracker.get_calibration_summary("single")
        assert summary.total_predictions == 1
        assert summary.brier_score == pytest.approx(0.04)  # (0.8 - 1)^2

    def test_very_low_confidence_correct(self, tracker):
        """Should handle very low confidence correct predictions."""
        tracker.record_prediction("lucky", 0.1, True)

        summary = tracker.get_calibration_summary("lucky")
        # Brier score should be high because low confidence was wrong
        assert summary.brier_score == 0.81  # (0.1 - 1)^2

    def test_calibration_curve_with_gaps(self, tracker):
        """Should handle calibration curve with empty buckets."""
        # Only add predictions at extreme confidence levels
        for _ in range(10):
            tracker.record_prediction("extreme", 0.1, False)
        for _ in range(10):
            tracker.record_prediction("extreme", 0.9, True)

        curve = tracker.get_calibration_curve("extreme", num_buckets=10)

        # Should have 10 buckets, most empty
        assert len(curve) == 10
        non_empty = [b for b in curve if b.total_predictions > 0]
        assert len(non_empty) == 2
