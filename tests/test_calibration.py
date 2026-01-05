"""
Tests for CalibrationTracker.

Tests prediction calibration tracking including:
- Recording predictions with outcomes
- Brier score calculation
- Expected Calibration Error (ECE)
- Calibration curves and buckets
- Over/under confidence detection
"""

import os
import tempfile
import pytest

from aragora.agents.calibration import (
    CalibrationTracker,
    CalibrationBucket,
    CalibrationSummary,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def tracker(temp_db):
    """Create a CalibrationTracker instance with temp database."""
    return CalibrationTracker(db_path=temp_db)


class TestCalibrationBucket:
    """Test CalibrationBucket dataclass properties."""

    def test_accuracy_with_predictions(self):
        """Test accuracy calculation with predictions."""
        bucket = CalibrationBucket(
            range_start=0.7,
            range_end=0.8,
            total_predictions=10,
            correct_predictions=7,
            brier_sum=0.5,
        )
        assert bucket.accuracy == 0.7

    def test_accuracy_no_predictions(self):
        """Test accuracy is 0 with no predictions."""
        bucket = CalibrationBucket(
            range_start=0.7,
            range_end=0.8,
            total_predictions=0,
            correct_predictions=0,
        )
        assert bucket.accuracy == 0.0

    def test_expected_accuracy(self):
        """Test expected accuracy is bucket midpoint."""
        bucket = CalibrationBucket(
            range_start=0.6,
            range_end=0.8,
            total_predictions=5,
            correct_predictions=3,
        )
        assert bucket.expected_accuracy == 0.7

    def test_calibration_error(self):
        """Test calibration error calculation."""
        # Bucket 0.6-0.8 (expected 0.7), actual accuracy 0.5
        bucket = CalibrationBucket(
            range_start=0.6,
            range_end=0.8,
            total_predictions=10,
            correct_predictions=5,
        )
        assert bucket.calibration_error == pytest.approx(0.2)

    def test_brier_score(self):
        """Test Brier score calculation for bucket."""
        bucket = CalibrationBucket(
            range_start=0.7,
            range_end=0.8,
            total_predictions=10,
            correct_predictions=7,
            brier_sum=2.0,
        )
        assert bucket.brier_score == 0.2


class TestCalibrationSummary:
    """Test CalibrationSummary dataclass properties."""

    def test_accuracy(self):
        """Test overall accuracy calculation."""
        summary = CalibrationSummary(
            agent="test_agent",
            total_predictions=100,
            total_correct=75,
            brier_score=0.15,
            ece=0.1,
        )
        assert summary.accuracy == 0.75

    def test_accuracy_no_predictions(self):
        """Test accuracy is 0 with no predictions."""
        summary = CalibrationSummary(agent="test_agent")
        assert summary.accuracy == 0.0

    def test_is_overconfident(self):
        """Test overconfidence detection."""
        # High confidence buckets with low actual accuracy
        buckets = [
            CalibrationBucket(0.8, 0.9, total_predictions=10, correct_predictions=5),
            CalibrationBucket(0.9, 1.0, total_predictions=10, correct_predictions=5),
        ]
        summary = CalibrationSummary(
            agent="test_agent",
            total_predictions=20,
            total_correct=10,
            buckets=buckets,
        )
        assert summary.is_overconfident is True

    def test_is_underconfident(self):
        """Test underconfidence detection."""
        # Low confidence buckets with high actual accuracy
        buckets = [
            CalibrationBucket(0.2, 0.3, total_predictions=10, correct_predictions=8),
            CalibrationBucket(0.3, 0.4, total_predictions=10, correct_predictions=8),
        ]
        summary = CalibrationSummary(
            agent="test_agent",
            total_predictions=20,
            total_correct=16,
            buckets=buckets,
        )
        assert summary.is_underconfident is True

    def test_well_calibrated(self):
        """Test well-calibrated agent detection."""
        # Buckets where accuracy matches confidence
        buckets = [
            CalibrationBucket(0.7, 0.8, total_predictions=10, correct_predictions=7),
            CalibrationBucket(0.8, 0.9, total_predictions=10, correct_predictions=8),
        ]
        summary = CalibrationSummary(
            agent="test_agent",
            total_predictions=20,
            total_correct=15,
            buckets=buckets,
        )
        assert summary.is_overconfident is False
        assert summary.is_underconfident is False


class TestCalibrationTrackerBasics:
    """Test basic CalibrationTracker operations."""

    def test_record_prediction(self, tracker):
        """Test recording a prediction."""
        pred_id = tracker.record_prediction(
            agent="agent_1",
            confidence=0.8,
            correct=True,
            domain="security",
        )
        assert pred_id > 0

    def test_record_multiple_predictions(self, tracker):
        """Test recording multiple predictions."""
        ids = []
        for i in range(5):
            pred_id = tracker.record_prediction(
                agent="agent_1",
                confidence=0.7,
                correct=(i % 2 == 0),
            )
            ids.append(pred_id)
        assert len(set(ids)) == 5  # All unique IDs

    def test_confidence_clamping(self, tracker):
        """Test that confidence is clamped to [0, 1]."""
        # Values outside [0,1] should be clamped and recorded
        # 1.5 -> 1.0, -0.5 -> 0.0
        tracker.record_prediction(agent="agent_clamp", confidence=1.5, correct=True)
        tracker.record_prediction(agent="agent_clamp", confidence=-0.5, correct=False)

        summary = tracker.get_calibration_summary("agent_clamp")
        # Both predictions should be recorded (clamped but not rejected)
        assert summary.total_predictions == 2


class TestBrierScore:
    """Test Brier score calculation."""

    def test_perfect_predictions(self, tracker):
        """Test Brier score for perfect predictions."""
        # All predictions correct with 100% confidence
        for _ in range(10):
            tracker.record_prediction(agent="perfect", confidence=1.0, correct=True)

        score = tracker.get_brier_score("perfect")
        assert score == 0.0

    def test_worst_predictions(self, tracker):
        """Test Brier score for worst predictions."""
        # All predictions wrong with 100% confidence
        for _ in range(10):
            tracker.record_prediction(agent="worst", confidence=1.0, correct=False)

        score = tracker.get_brier_score("worst")
        assert score == 1.0

    def test_random_baseline(self, tracker):
        """Test Brier score at random baseline (50% confidence)."""
        for _ in range(10):
            tracker.record_prediction(agent="random", confidence=0.5, correct=True)
        for _ in range(10):
            tracker.record_prediction(agent="random", confidence=0.5, correct=False)

        score = tracker.get_brier_score("random")
        assert score == pytest.approx(0.25)

    def test_brier_score_by_domain(self, tracker):
        """Test Brier score filtered by domain."""
        # Good predictions in security domain
        for _ in range(5):
            tracker.record_prediction(
                agent="agent_1", confidence=0.9, correct=True, domain="security"
            )

        # Bad predictions in performance domain
        for _ in range(5):
            tracker.record_prediction(
                agent="agent_1", confidence=0.9, correct=False, domain="performance"
            )

        security_score = tracker.get_brier_score("agent_1", domain="security")
        performance_score = tracker.get_brier_score("agent_1", domain="performance")

        assert security_score < performance_score


class TestExpectedCalibrationError:
    """Test Expected Calibration Error (ECE) calculation."""

    def test_perfect_calibration(self, tracker):
        """Test ECE for perfectly calibrated predictions."""
        # 80% confidence, 80% accuracy
        for _ in range(8):
            tracker.record_prediction(agent="calibrated", confidence=0.8, correct=True)
        for _ in range(2):
            tracker.record_prediction(agent="calibrated", confidence=0.8, correct=False)

        ece = tracker.get_expected_calibration_error("calibrated")
        assert ece < 0.1  # Should be close to 0

    def test_poor_calibration(self, tracker):
        """Test ECE for poorly calibrated predictions."""
        # High confidence but low accuracy
        for _ in range(10):
            tracker.record_prediction(agent="uncalibrated", confidence=0.9, correct=False)

        ece = tracker.get_expected_calibration_error("uncalibrated")
        assert ece > 0.5  # Should be high

    def test_ece_no_predictions(self, tracker):
        """Test ECE returns 0 with no predictions."""
        ece = tracker.get_expected_calibration_error("nonexistent")
        assert ece == 0.0


class TestCalibrationCurve:
    """Test calibration curve generation."""

    def test_curve_buckets(self, tracker):
        """Test calibration curve has correct number of buckets."""
        for conf in [0.1, 0.3, 0.5, 0.7, 0.9]:
            tracker.record_prediction(agent="agent_1", confidence=conf, correct=True)

        curve = tracker.get_calibration_curve("agent_1", num_buckets=10)
        assert len(curve) == 10

    def test_curve_bucket_ranges(self, tracker):
        """Test bucket ranges are correct."""
        tracker.record_prediction(agent="agent_1", confidence=0.5, correct=True)

        curve = tracker.get_calibration_curve("agent_1", num_buckets=5)

        assert curve[0].range_start == 0.0
        assert curve[0].range_end == 0.2
        assert curve[-1].range_start == 0.8
        assert curve[-1].range_end == 1.0

    def test_predictions_in_correct_buckets(self, tracker):
        """Test predictions are assigned to correct buckets."""
        # Add predictions at 0.75 confidence
        for _ in range(5):
            tracker.record_prediction(agent="agent_1", confidence=0.75, correct=True)

        curve = tracker.get_calibration_curve("agent_1", num_buckets=10)

        # Bucket 7 covers 0.7-0.8
        assert curve[7].total_predictions == 5
        assert curve[7].correct_predictions == 5


class TestCalibrationSummaryMethod:
    """Test get_calibration_summary method."""

    def test_summary_aggregation(self, tracker):
        """Test summary aggregates all predictions correctly."""
        for _ in range(6):
            tracker.record_prediction(agent="agent_1", confidence=0.8, correct=True)
        for _ in range(4):
            tracker.record_prediction(agent="agent_1", confidence=0.8, correct=False)

        summary = tracker.get_calibration_summary("agent_1")

        assert summary.agent == "agent_1"
        assert summary.total_predictions == 10
        assert summary.total_correct == 6
        assert summary.accuracy == 0.6

    def test_summary_with_domain_filter(self, tracker):
        """Test summary filtered by domain."""
        tracker.record_prediction(agent="agent_1", confidence=0.8, correct=True, domain="a")
        tracker.record_prediction(agent="agent_1", confidence=0.8, correct=False, domain="b")

        summary_a = tracker.get_calibration_summary("agent_1", domain="a")
        summary_b = tracker.get_calibration_summary("agent_1", domain="b")

        assert summary_a.total_predictions == 1
        assert summary_a.total_correct == 1
        assert summary_b.total_predictions == 1
        assert summary_b.total_correct == 0


class TestDomainBreakdown:
    """Test domain breakdown functionality."""

    def test_domain_breakdown(self, tracker):
        """Test getting breakdown by domain."""
        tracker.record_prediction(agent="agent_1", confidence=0.8, correct=True, domain="security")
        tracker.record_prediction(agent="agent_1", confidence=0.8, correct=True, domain="performance")
        tracker.record_prediction(agent="agent_1", confidence=0.8, correct=False, domain="security")

        breakdown = tracker.get_domain_breakdown("agent_1")

        assert "security" in breakdown
        assert "performance" in breakdown
        assert breakdown["security"].total_predictions == 2
        assert breakdown["performance"].total_predictions == 1


class TestAgentManagement:
    """Test agent listing and data management."""

    def test_get_all_agents(self, tracker):
        """Test listing all agents."""
        tracker.record_prediction(agent="alpha", confidence=0.5, correct=True)
        tracker.record_prediction(agent="beta", confidence=0.5, correct=True)
        tracker.record_prediction(agent="gamma", confidence=0.5, correct=True)

        agents = tracker.get_all_agents()
        assert set(agents) == {"alpha", "beta", "gamma"}

    def test_delete_agent_data(self, tracker):
        """Test deleting agent data."""
        for _ in range(5):
            tracker.record_prediction(agent="deletable", confidence=0.5, correct=True)

        deleted = tracker.delete_agent_data("deletable")
        assert deleted == 5

        agents = tracker.get_all_agents()
        assert "deletable" not in agents


class TestLeaderboard:
    """Test leaderboard functionality."""

    def test_leaderboard_by_brier(self, tracker):
        """Test ranking agents by Brier score."""
        # Good agent (low Brier)
        for _ in range(10):
            tracker.record_prediction(agent="good", confidence=0.9, correct=True)

        # Bad agent (high Brier)
        for _ in range(10):
            tracker.record_prediction(agent="bad", confidence=0.9, correct=False)

        leaderboard = tracker.get_leaderboard(metric="brier", limit=10)

        # Lower Brier is better, so "good" should be first
        assert len(leaderboard) == 2
        assert leaderboard[0][0] == "good"
        assert leaderboard[1][0] == "bad"

    def test_leaderboard_by_accuracy(self, tracker):
        """Test ranking agents by accuracy."""
        # High accuracy agent
        for _ in range(10):
            tracker.record_prediction(agent="accurate", confidence=0.7, correct=True)

        # Low accuracy agent
        for _ in range(10):
            tracker.record_prediction(agent="inaccurate", confidence=0.7, correct=False)

        leaderboard = tracker.get_leaderboard(metric="accuracy", limit=10)

        # Higher accuracy is better
        assert leaderboard[0][0] == "accurate"
        assert leaderboard[1][0] == "inaccurate"

    def test_leaderboard_min_predictions(self, tracker):
        """Test that agents with few predictions are excluded."""
        # Agent with 10 predictions
        for _ in range(10):
            tracker.record_prediction(agent="many", confidence=0.7, correct=True)

        # Agent with 3 predictions (below threshold)
        for _ in range(3):
            tracker.record_prediction(agent="few", confidence=0.7, correct=True)

        leaderboard = tracker.get_leaderboard(limit=10)

        agents = [entry[0] for entry in leaderboard]
        assert "many" in agents
        assert "few" not in agents


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
