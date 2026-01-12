"""
Tests for aragora.debate.outcome_tracker - Consensus outcome tracking.

Tests cover:
- ConsensusOutcome dataclass
- CalibrationBucket dataclass
- OutcomeTracker database operations
- Calibration curve calculations
- Success rate analysis
- Failure pattern detection
- Overconfidence detection
- Calibration adjustment recommendations
"""

import json
import os
import pytest
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

from aragora.debate.outcome_tracker import (
    ConsensusOutcome,
    CalibrationBucket,
    OutcomeTracker,
)


@pytest.fixture
def temp_db_path():
    """Create temporary database path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield Path(path)
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def tracker(temp_db_path):
    """Create OutcomeTracker with temp database."""
    return OutcomeTracker(db_path=temp_db_path)


class TestConsensusOutcome:
    """Tests for ConsensusOutcome dataclass."""

    def test_required_fields(self):
        """Should create with required fields."""
        outcome = ConsensusOutcome(
            debate_id="debate-1",
            consensus_text="Test consensus",
            consensus_confidence=0.85,
            implementation_attempted=True,
            implementation_succeeded=True,
        )

        assert outcome.debate_id == "debate-1"
        assert outcome.consensus_confidence == 0.85
        assert outcome.implementation_succeeded is True

    def test_default_fields(self):
        """Default fields should have sensible values."""
        outcome = ConsensusOutcome(
            debate_id="test",
            consensus_text="Test",
            consensus_confidence=0.5,
            implementation_attempted=False,
            implementation_succeeded=False,
        )

        assert outcome.tests_passed == 0
        assert outcome.tests_failed == 0
        assert outcome.rollback_triggered is False
        assert outcome.time_to_failure is None
        assert outcome.failure_reason is None
        assert outcome.agents_participating == []
        assert outcome.rounds_completed == 0
        assert outcome.trickster_interventions == 0
        assert outcome.evidence_coverage == 0.0

    def test_timestamp_auto_generated(self):
        """Timestamp should be auto-generated."""
        outcome = ConsensusOutcome(
            debate_id="test",
            consensus_text="Test",
            consensus_confidence=0.5,
            implementation_attempted=True,
            implementation_succeeded=True,
        )

        assert outcome.timestamp is not None
        assert len(outcome.timestamp) > 0

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        outcome = ConsensusOutcome(
            debate_id="debate-1",
            consensus_text="Test consensus",
            consensus_confidence=0.85,
            implementation_attempted=True,
            implementation_succeeded=True,
            agents_participating=["claude", "gpt4"],
        )

        d = outcome.to_dict()

        assert d["debate_id"] == "debate-1"
        assert d["consensus_confidence"] == 0.85
        # agents_participating should be JSON-encoded
        assert d["agents_participating"] == '["claude", "gpt4"]'

    def test_from_row(self):
        """from_row should deserialize from database row."""
        # Create a mock row
        class MockRow(dict):
            def __getitem__(self, key):
                return dict.get(self, key)

        row = MockRow({
            "debate_id": "debate-1",
            "consensus_text": "Test",
            "consensus_confidence": 0.85,
            "implementation_attempted": 1,
            "implementation_succeeded": 1,
            "tests_passed": 10,
            "tests_failed": 0,
            "rollback_triggered": 0,
            "time_to_failure": None,
            "failure_reason": None,
            "timestamp": "2024-01-01T00:00:00",
            "agents_participating": '["claude"]',
            "rounds_completed": 3,
            "trickster_interventions": 1,
            "evidence_coverage": 0.8,
        })

        outcome = ConsensusOutcome.from_row(row)

        assert outcome.debate_id == "debate-1"
        assert outcome.agents_participating == ["claude"]


class TestCalibrationBucket:
    """Tests for CalibrationBucket dataclass."""

    def test_success_rate_calculation(self):
        """success_rate should calculate correctly."""
        bucket = CalibrationBucket(
            confidence_min=0.7,
            confidence_max=0.8,
            total_count=10,
            success_count=8,
        )

        assert bucket.success_rate == 0.8

    def test_success_rate_zero_count(self):
        """success_rate should handle zero count."""
        bucket = CalibrationBucket(
            confidence_min=0.7,
            confidence_max=0.8,
            total_count=0,
            success_count=0,
        )

        assert bucket.success_rate == 0.0

    def test_expected_rate(self):
        """expected_rate should be bucket midpoint."""
        bucket = CalibrationBucket(
            confidence_min=0.7,
            confidence_max=0.8,
            total_count=10,
            success_count=8,
        )

        assert bucket.expected_rate == 0.75

    def test_calibration_error_overconfident(self):
        """Positive error means overconfident."""
        bucket = CalibrationBucket(
            confidence_min=0.8,
            confidence_max=0.9,
            total_count=10,
            success_count=5,  # 50% actual vs 85% expected
        )

        # Expected 0.85, actual 0.5, error = 0.85 - 0.5 = 0.35
        assert bucket.calibration_error == pytest.approx(0.35)

    def test_calibration_error_underconfident(self):
        """Negative error means underconfident."""
        bucket = CalibrationBucket(
            confidence_min=0.3,
            confidence_max=0.4,
            total_count=10,
            success_count=8,  # 80% actual vs 35% expected
        )

        # Expected 0.35, actual 0.8, error = 0.35 - 0.8 = -0.45
        assert bucket.calibration_error == pytest.approx(-0.45)

    def test_calibration_error_well_calibrated(self):
        """Zero error means well calibrated."""
        bucket = CalibrationBucket(
            confidence_min=0.7,
            confidence_max=0.8,
            total_count=100,
            success_count=75,  # 75% actual = 75% expected
        )

        assert bucket.calibration_error == pytest.approx(0.0)


class TestOutcomeTrackerInit:
    """Tests for OutcomeTracker initialization."""

    def test_creates_database(self, temp_db_path):
        """Should create database file."""
        tracker = OutcomeTracker(db_path=temp_db_path)

        assert temp_db_path.exists()

    def test_creates_schema(self, tracker, temp_db_path):
        """Should create required tables."""
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]

        assert "outcomes" in tables

    def test_creates_indexes(self, tracker, temp_db_path):
        """Should create required indexes."""
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
            indexes = [row[0] for row in cursor.fetchall()]

        assert "idx_outcomes_confidence" in indexes


class TestRecordOutcome:
    """Tests for record_outcome method."""

    def test_records_outcome(self, tracker):
        """Should record outcome to database."""
        outcome = ConsensusOutcome(
            debate_id="debate-1",
            consensus_text="Test consensus",
            consensus_confidence=0.85,
            implementation_attempted=True,
            implementation_succeeded=True,
        )

        tracker.record_outcome(outcome)

        # Verify stored
        stored = tracker.get_outcome("debate-1")
        assert stored is not None
        assert stored.consensus_confidence == 0.85

    def test_replace_existing(self, tracker):
        """Should replace existing outcome with same ID."""
        outcome1 = ConsensusOutcome(
            debate_id="debate-1",
            consensus_text="First",
            consensus_confidence=0.5,
            implementation_attempted=True,
            implementation_succeeded=False,
        )
        outcome2 = ConsensusOutcome(
            debate_id="debate-1",
            consensus_text="Second",
            consensus_confidence=0.9,
            implementation_attempted=True,
            implementation_succeeded=True,
        )

        tracker.record_outcome(outcome1)
        tracker.record_outcome(outcome2)

        stored = tracker.get_outcome("debate-1")
        assert stored.consensus_text == "Second"
        assert stored.consensus_confidence == 0.9

    def test_records_all_fields(self, tracker):
        """Should record all fields correctly."""
        outcome = ConsensusOutcome(
            debate_id="debate-full",
            consensus_text="Full test",
            consensus_confidence=0.75,
            implementation_attempted=True,
            implementation_succeeded=True,
            tests_passed=42,
            tests_failed=3,
            rollback_triggered=False,
            time_to_failure=None,
            failure_reason=None,
            agents_participating=["claude", "gpt4", "gemini"],
            rounds_completed=5,
            trickster_interventions=2,
            evidence_coverage=0.85,
        )

        tracker.record_outcome(outcome)
        stored = tracker.get_outcome("debate-full")

        assert stored.tests_passed == 42
        assert stored.tests_failed == 3
        assert stored.agents_participating == ["claude", "gpt4", "gemini"]
        assert stored.rounds_completed == 5
        assert stored.trickster_interventions == 2
        assert stored.evidence_coverage == 0.85


class TestGetOutcome:
    """Tests for get_outcome method."""

    def test_returns_none_for_missing(self, tracker):
        """Should return None for non-existent ID."""
        result = tracker.get_outcome("nonexistent")
        assert result is None

    def test_returns_outcome(self, tracker):
        """Should return stored outcome."""
        outcome = ConsensusOutcome(
            debate_id="test-get",
            consensus_text="Test",
            consensus_confidence=0.5,
            implementation_attempted=True,
            implementation_succeeded=True,
        )
        tracker.record_outcome(outcome)

        result = tracker.get_outcome("test-get")

        assert result is not None
        assert result.debate_id == "test-get"


class TestGetRecentOutcomes:
    """Tests for get_recent_outcomes method."""

    def test_returns_empty_list(self, tracker):
        """Should return empty list when no outcomes."""
        results = tracker.get_recent_outcomes()
        assert results == []

    def test_returns_recent_first(self, tracker):
        """Should return most recent first."""
        for i in range(5):
            outcome = ConsensusOutcome(
                debate_id=f"debate-{i}",
                consensus_text=f"Test {i}",
                consensus_confidence=0.5,
                implementation_attempted=True,
                implementation_succeeded=True,
                timestamp=f"2024-01-0{i+1}T00:00:00",
            )
            tracker.record_outcome(outcome)

        results = tracker.get_recent_outcomes(limit=3)

        assert len(results) == 3
        # Most recent (debate-4) should be first
        assert results[0].debate_id == "debate-4"

    def test_respects_limit(self, tracker):
        """Should respect limit parameter."""
        for i in range(10):
            outcome = ConsensusOutcome(
                debate_id=f"debate-{i}",
                consensus_text="Test",
                consensus_confidence=0.5,
                implementation_attempted=True,
                implementation_succeeded=True,
            )
            tracker.record_outcome(outcome)

        results = tracker.get_recent_outcomes(limit=5)

        assert len(results) == 5


class TestSuccessRateByConfidence:
    """Tests for get_success_rate_by_confidence method."""

    def test_empty_returns_empty_dict(self, tracker):
        """Should return empty dict when no data."""
        result = tracker.get_success_rate_by_confidence()
        assert result == {}

    def test_calculates_success_rate(self, tracker):
        """Should calculate success rates by bucket."""
        # Add outcomes with varying confidence
        for conf, success in [
            (0.75, True), (0.72, True), (0.78, False),  # 0.7-0.8 bucket
            (0.85, True), (0.88, True), (0.82, True),   # 0.8-0.9 bucket
        ]:
            tracker.record_outcome(ConsensusOutcome(
                debate_id=f"debate-{conf}-{success}",
                consensus_text="Test",
                consensus_confidence=conf,
                implementation_attempted=True,
                implementation_succeeded=success,
            ))

        result = tracker.get_success_rate_by_confidence()

        # 0.7-0.8: 2 success out of 3
        assert "0.7-0.8" in result
        assert result["0.7-0.8"] == pytest.approx(2/3, rel=0.01)

        # 0.8-0.9: 3 success out of 3
        assert "0.8-0.9" in result
        assert result["0.8-0.9"] == 1.0


class TestCalibrationCurve:
    """Tests for get_calibration_curve method."""

    def test_returns_buckets(self, tracker):
        """Should return list of CalibrationBucket."""
        result = tracker.get_calibration_curve(num_buckets=10)

        assert len(result) == 10
        assert all(isinstance(b, CalibrationBucket) for b in result)

    def test_buckets_cover_full_range(self, tracker):
        """Buckets should cover 0.0 to 1.0."""
        result = tracker.get_calibration_curve(num_buckets=10)

        assert result[0].confidence_min == 0.0
        assert result[-1].confidence_max == 1.0

    def test_bucket_counts(self, tracker):
        """Should count outcomes in correct buckets."""
        # Add outcomes at specific confidence levels
        # Use enumerate for unique debate_ids
        for i, conf in enumerate([0.75, 0.85, 0.86, 0.95]):
            tracker.record_outcome(ConsensusOutcome(
                debate_id=f"debate-bucket-{i}",
                consensus_text="Test",
                consensus_confidence=conf,
                implementation_attempted=True,
                implementation_succeeded=True,
            ))

        result = tracker.get_calibration_curve(num_buckets=10)

        # Check bucket 7 (0.7-0.8) has 1
        # Check bucket 8 (0.8-0.9) has 2
        # Check bucket 9 (0.9-1.0) has 1
        bucket_7 = result[7]  # 0.7-0.8
        bucket_8 = result[8]  # 0.8-0.9
        bucket_9 = result[9]  # 0.9-1.0

        assert bucket_7.total_count == 1
        assert bucket_8.total_count == 2
        assert bucket_9.total_count == 1


class TestFailurePatterns:
    """Tests for get_failure_patterns method."""

    def test_returns_empty_for_no_failures(self, tracker):
        """Should return empty list when no failures."""
        result = tracker.get_failure_patterns()
        assert result == []

    def test_counts_failure_reasons(self, tracker):
        """Should count failure reasons."""
        # Add failures with reasons
        for i, reason in enumerate([
            "Type error",
            "Type error",
            "Type error",
            "Timeout",
            "Timeout",
            "Connection failed",
        ]):
            tracker.record_outcome(ConsensusOutcome(
                debate_id=f"fail-{i}",
                consensus_text="Test",
                consensus_confidence=0.5,
                implementation_attempted=True,
                implementation_succeeded=False,
                failure_reason=reason,
            ))

        result = tracker.get_failure_patterns(limit=10)

        assert len(result) == 3
        assert result[0]["reason"] == "Type error"
        assert result[0]["count"] == 3
        assert result[1]["reason"] == "Timeout"
        assert result[1]["count"] == 2

    def test_respects_limit(self, tracker):
        """Should respect limit parameter."""
        for i in range(10):
            tracker.record_outcome(ConsensusOutcome(
                debate_id=f"fail-{i}",
                consensus_text="Test",
                consensus_confidence=0.5,
                implementation_attempted=True,
                implementation_succeeded=False,
                failure_reason=f"Reason {i}",
            ))

        result = tracker.get_failure_patterns(limit=3)

        assert len(result) == 3


class TestOverallStats:
    """Tests for get_overall_stats method."""

    def test_empty_stats(self, tracker):
        """Should return zeros for empty tracker."""
        stats = tracker.get_overall_stats()

        assert stats["total_outcomes"] == 0
        assert stats["attempted"] == 0
        assert stats["succeeded"] == 0
        assert stats["success_rate"] == 0.0

    def test_calculates_stats(self, tracker):
        """Should calculate correct statistics."""
        # Add mix of outcomes
        tracker.record_outcome(ConsensusOutcome(
            debate_id="d1",
            consensus_text="T",
            consensus_confidence=0.8,
            implementation_attempted=True,
            implementation_succeeded=True,
            tests_passed=10,
            tests_failed=0,
        ))
        tracker.record_outcome(ConsensusOutcome(
            debate_id="d2",
            consensus_text="T",
            consensus_confidence=0.7,
            implementation_attempted=True,
            implementation_succeeded=False,
            tests_passed=5,
            tests_failed=3,
            rollback_triggered=True,
        ))
        tracker.record_outcome(ConsensusOutcome(
            debate_id="d3",
            consensus_text="T",
            consensus_confidence=0.6,
            implementation_attempted=False,
            implementation_succeeded=False,
        ))

        stats = tracker.get_overall_stats()

        assert stats["total_outcomes"] == 3
        assert stats["attempted"] == 2
        assert stats["succeeded"] == 1
        assert stats["success_rate"] == 0.5
        assert stats["rollbacks"] == 1
        assert stats["avg_confidence"] == pytest.approx(0.7, rel=0.01)
        assert stats["total_tests_passed"] == 15
        assert stats["total_tests_failed"] == 3


class TestIsOverconfident:
    """Tests for is_overconfident method."""

    def test_false_with_no_data(self, tracker):
        """Should return False with insufficient data."""
        result = tracker.is_overconfident()
        assert result is False

    def test_false_with_few_samples(self, tracker):
        """Should return False with < 5 samples."""
        for i in range(3):
            tracker.record_outcome(ConsensusOutcome(
                debate_id=f"d{i}",
                consensus_text="T",
                consensus_confidence=0.9,
                implementation_attempted=True,
                implementation_succeeded=False,
            ))

        result = tracker.is_overconfident(threshold=0.7)
        assert result is False  # Not enough samples

    def test_true_when_overconfident(self, tracker):
        """Should return True when system is overconfident."""
        # High confidence, low success rate
        for i in range(10):
            tracker.record_outcome(ConsensusOutcome(
                debate_id=f"d{i}",
                consensus_text="T",
                consensus_confidence=0.9,  # 90% confident
                implementation_attempted=True,
                implementation_succeeded=(i < 3),  # Only 30% success
            ))

        result = tracker.is_overconfident(threshold=0.7)
        assert result is True  # 90% conf vs 30% success

    def test_false_when_well_calibrated(self, tracker):
        """Should return False when well calibrated."""
        # High confidence, high success rate
        for i in range(10):
            tracker.record_outcome(ConsensusOutcome(
                debate_id=f"d{i}",
                consensus_text="T",
                consensus_confidence=0.85,  # 85% confident
                implementation_attempted=True,
                implementation_succeeded=(i < 8),  # 80% success
            ))

        result = tracker.is_overconfident(threshold=0.7)
        assert result is False  # Within 10% margin


class TestCalibrationAdjustment:
    """Tests for get_calibration_adjustment method."""

    def test_returns_one_with_no_data(self, tracker):
        """Should return 1.0 with no data."""
        adjustment = tracker.get_calibration_adjustment()
        assert adjustment == 1.0

    def test_adjustment_for_overconfidence(self, tracker):
        """Should return < 1.0 for overconfidence."""
        # High confidence, low success
        for i in range(10):
            tracker.record_outcome(ConsensusOutcome(
                debate_id=f"d{i}",
                consensus_text="T",
                consensus_confidence=0.85,  # 85% confident
                implementation_attempted=True,
                implementation_succeeded=(i < 5),  # 50% success
            ))

        adjustment = tracker.get_calibration_adjustment()
        assert adjustment < 1.0  # Should recommend increasing sensitivity

    def test_adjustment_for_underconfidence(self, tracker):
        """Should return > 1.0 for underconfidence."""
        # Low confidence, high success
        for i in range(10):
            tracker.record_outcome(ConsensusOutcome(
                debate_id=f"d{i}",
                consensus_text="T",
                consensus_confidence=0.75,  # 75% confident
                implementation_attempted=True,
                implementation_succeeded=True,  # 100% success
            ))

        adjustment = tracker.get_calibration_adjustment()
        assert adjustment > 1.0  # Should recommend decreasing sensitivity

    def test_adjustment_clamped(self, tracker):
        """Adjustment should be clamped to reasonable range."""
        # Extreme overconfidence
        for i in range(10):
            tracker.record_outcome(ConsensusOutcome(
                debate_id=f"d{i}",
                consensus_text="T",
                consensus_confidence=0.95,
                implementation_attempted=True,
                implementation_succeeded=False,
            ))

        adjustment = tracker.get_calibration_adjustment()
        assert 0.5 <= adjustment <= 1.5


class TestEdgeCases:
    """Edge case tests."""

    def test_very_long_consensus_text(self, tracker):
        """Should handle very long consensus text."""
        long_text = "A" * 10000

        outcome = ConsensusOutcome(
            debate_id="long-text",
            consensus_text=long_text,
            consensus_confidence=0.5,
            implementation_attempted=True,
            implementation_succeeded=True,
        )

        tracker.record_outcome(outcome)
        stored = tracker.get_outcome("long-text")

        assert stored.consensus_text == long_text

    def test_special_characters_in_failure_reason(self, tracker):
        """Should handle special characters in failure reason."""
        outcome = ConsensusOutcome(
            debate_id="special-chars",
            consensus_text="Test",
            consensus_confidence=0.5,
            implementation_attempted=True,
            implementation_succeeded=False,
            failure_reason="Error: 'unexpected' token at line 42",
        )

        tracker.record_outcome(outcome)
        stored = tracker.get_outcome("special-chars")

        assert "unexpected" in stored.failure_reason

    def test_unicode_in_text(self, tracker):
        """Should handle unicode characters."""
        outcome = ConsensusOutcome(
            debate_id="unicode",
            consensus_text="Test with emoji: Test",
            consensus_confidence=0.5,
            implementation_attempted=True,
            implementation_succeeded=True,
        )

        tracker.record_outcome(outcome)
        stored = tracker.get_outcome("unicode")

        assert "emoji" in stored.consensus_text
