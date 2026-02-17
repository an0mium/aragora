"""
Tests for aragora.debate.outcome_tracker module.

Tests:
- ConsensusOutcome.to_dict() and from_row()
- CalibrationBucket properties
- OutcomeTracker CRUD operations
- Calibration curve with controlled data
- is_overconfident with overconfident and well-calibrated data
- get_calibration_adjustment
- AsyncOutcomeTracker (async wrapper)
- Failure patterns and overall stats
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aragora.debate.outcome_tracker import (
    AsyncOutcomeTracker,
    CalibrationBucket,
    ConsensusOutcome,
    OutcomeTracker,
)


# ============================================================================
# ConsensusOutcome Tests
# ============================================================================


class TestConsensusOutcome:
    """Tests for ConsensusOutcome dataclass."""

    def test_to_dict_converts_agents_to_json(self):
        """Test that to_dict serializes agents_participating as JSON."""
        outcome = ConsensusOutcome(
            debate_id="test-1",
            consensus_text="Add feature X",
            consensus_confidence=0.85,
            implementation_attempted=True,
            implementation_succeeded=True,
            agents_participating=["agent1", "agent2", "agent3"],
        )

        result = outcome.to_dict()

        assert result["agents_participating"] == json.dumps(["agent1", "agent2", "agent3"])
        assert result["debate_id"] == "test-1"
        assert result["consensus_confidence"] == 0.85

    def test_to_dict_with_empty_agents(self):
        """Test to_dict with no agents."""
        outcome = ConsensusOutcome(
            debate_id="test-2",
            consensus_text="Refactor module",
            consensus_confidence=0.7,
            implementation_attempted=False,
            implementation_succeeded=False,
        )

        result = outcome.to_dict()

        assert result["agents_participating"] == json.dumps([])

    def test_from_row_deserializes_agents(self):
        """Test from_row reconstructs agents_participating from JSON."""
        # Mock sqlite3.Row
        row_data = {
            "debate_id": "test-3",
            "consensus_text": "Fix bug Y",
            "consensus_confidence": 0.92,
            "implementation_attempted": True,
            "implementation_succeeded": True,
            "tests_passed": 10,
            "tests_failed": 0,
            "rollback_triggered": False,
            "time_to_failure": None,
            "failure_reason": None,
            "timestamp": "2026-02-17T10:00:00",
            "agents_participating": json.dumps(["claude", "codex"]),
            "rounds_completed": 3,
            "trickster_interventions": 1,
            "evidence_coverage": 0.8,
        }

        # Create a mock Row object
        mock_row = MagicMock(spec=sqlite3.Row)
        mock_row.keys.return_value = list(row_data.keys())
        mock_row.__getitem__ = lambda self, key: row_data[key]
        mock_row.__iter__ = lambda self: iter(row_data.items())

        # Make dict(row) work
        def items_side_effect():
            return row_data.items()

        type(mock_row).keys = MagicMock(return_value=list(row_data.keys()))
        mock_row.__iter__ = lambda: iter(row_data.items())

        # Create a real dict that behaves like Row
        class FakeRow(dict):
            def keys(self):
                return super().keys()

        fake_row = FakeRow(row_data)

        outcome = ConsensusOutcome.from_row(fake_row)

        assert outcome.debate_id == "test-3"
        assert outcome.agents_participating == ["claude", "codex"]
        assert outcome.consensus_confidence == 0.92
        assert outcome.tests_passed == 10

    def test_from_row_filters_extra_fields(self):
        """Test that from_row filters out database-only fields like 'id'."""
        row_data = {
            "id": 42,  # Database-only field
            "debate_id": "test-4",
            "consensus_text": "Deploy to prod",
            "consensus_confidence": 0.95,
            "implementation_attempted": True,
            "implementation_succeeded": True,
            "tests_passed": 50,
            "tests_failed": 0,
            "rollback_triggered": False,
            "time_to_failure": None,
            "failure_reason": None,
            "timestamp": "2026-02-17T11:00:00",
            "agents_participating": "[]",
            "rounds_completed": 5,
            "trickster_interventions": 0,
            "evidence_coverage": 0.9,
        }

        class FakeRow(dict):
            pass

        fake_row = FakeRow(row_data)

        outcome = ConsensusOutcome.from_row(fake_row)

        assert outcome.debate_id == "test-4"
        assert not hasattr(outcome, "id")

    def test_from_row_handles_missing_agents(self):
        """Test from_row with missing agents_participating field."""
        row_data = {
            "debate_id": "test-5",
            "consensus_text": "Update deps",
            "consensus_confidence": 0.6,
            "implementation_attempted": True,
            "implementation_succeeded": False,
            "tests_passed": 5,
            "tests_failed": 2,
            "rollback_triggered": True,
            "time_to_failure": 120.5,
            "failure_reason": "Tests failed",
            "timestamp": "2026-02-17T12:00:00",
            # agents_participating is missing
            "rounds_completed": 2,
            "trickster_interventions": 0,
            "evidence_coverage": 0.5,
        }

        class FakeRow(dict):
            pass

        fake_row = FakeRow(row_data)

        outcome = ConsensusOutcome.from_row(fake_row)

        assert outcome.agents_participating == []  # Default from JSON parsing


# ============================================================================
# CalibrationBucket Tests
# ============================================================================


class TestCalibrationBucket:
    """Tests for CalibrationBucket dataclass."""

    def test_success_rate_with_data(self):
        """Test success_rate property calculates correctly."""
        bucket = CalibrationBucket(
            confidence_min=0.7,
            confidence_max=0.8,
            total_count=10,
            success_count=8,
        )

        assert bucket.success_rate == 0.8

    def test_success_rate_with_zero_count(self):
        """Test success_rate returns 0 when total_count is 0."""
        bucket = CalibrationBucket(
            confidence_min=0.9,
            confidence_max=1.0,
            total_count=0,
            success_count=0,
        )

        assert bucket.success_rate == 0.0

    def test_expected_rate(self):
        """Test expected_rate is midpoint of bucket."""
        bucket = CalibrationBucket(
            confidence_min=0.5,
            confidence_max=0.6,
            total_count=20,
            success_count=11,
        )

        assert bucket.expected_rate == 0.55

    def test_calibration_error_overconfident(self):
        """Test calibration_error when overconfident (positive error)."""
        bucket = CalibrationBucket(
            confidence_min=0.8,
            confidence_max=0.9,
            total_count=10,
            success_count=7,  # 70% success vs 85% expected
        )

        # expected = 0.85, success_rate = 0.7
        # error = 0.85 - 0.7 = 0.15
        assert bucket.calibration_error == pytest.approx(0.15, abs=0.01)

    def test_calibration_error_underconfident(self):
        """Test calibration_error when underconfident (negative error)."""
        bucket = CalibrationBucket(
            confidence_min=0.6,
            confidence_max=0.7,
            total_count=10,
            success_count=8,  # 80% success vs 65% expected
        )

        # expected = 0.65, success_rate = 0.8
        # error = 0.65 - 0.8 = -0.15
        assert bucket.calibration_error == pytest.approx(-0.15, abs=0.01)

    def test_calibration_error_well_calibrated(self):
        """Test calibration_error when well-calibrated (near zero)."""
        bucket = CalibrationBucket(
            confidence_min=0.7,
            confidence_max=0.8,
            total_count=20,
            success_count=15,  # 75% success vs 75% expected
        )

        assert bucket.calibration_error == pytest.approx(0.0, abs=0.01)


# ============================================================================
# OutcomeTracker Tests
# ============================================================================


class TestOutcomeTracker:
    """Tests for OutcomeTracker class."""

    def test_init_creates_database(self, tmp_path: Path):
        """Test that init creates SQLite database."""
        db_path = tmp_path / "test_outcomes.db"

        tracker = OutcomeTracker(db_path)

        assert tracker.db_path == db_path
        assert db_path.exists()

    def test_record_outcome(self, tmp_path: Path):
        """Test recording an outcome."""
        tracker = OutcomeTracker(tmp_path / "outcomes.db")

        outcome = ConsensusOutcome(
            debate_id="debate-1",
            consensus_text="Implement feature A",
            consensus_confidence=0.85,
            implementation_attempted=True,
            implementation_succeeded=True,
            tests_passed=42,
            agents_participating=["agent1", "agent2"],
        )

        tracker.record_outcome(outcome)

        # Verify it was stored
        retrieved = tracker.get_outcome("debate-1")
        assert retrieved is not None
        assert retrieved.debate_id == "debate-1"
        assert retrieved.consensus_confidence == 0.85
        assert retrieved.tests_passed == 42

    def test_record_outcome_replace_existing(self, tmp_path: Path):
        """Test INSERT OR REPLACE updates existing outcome."""
        tracker = OutcomeTracker(tmp_path / "outcomes.db")

        outcome1 = ConsensusOutcome(
            debate_id="debate-2",
            consensus_text="First version",
            consensus_confidence=0.7,
            implementation_attempted=True,
            implementation_succeeded=False,
        )
        tracker.record_outcome(outcome1)

        outcome2 = ConsensusOutcome(
            debate_id="debate-2",
            consensus_text="Updated version",
            consensus_confidence=0.9,
            implementation_attempted=True,
            implementation_succeeded=True,
            tests_passed=10,
        )
        tracker.record_outcome(outcome2)

        retrieved = tracker.get_outcome("debate-2")
        assert retrieved.consensus_text == "Updated version"
        assert retrieved.consensus_confidence == 0.9
        assert retrieved.tests_passed == 10

    def test_get_outcome_not_found(self, tmp_path: Path):
        """Test get_outcome returns None for missing debate_id."""
        tracker = OutcomeTracker(tmp_path / "outcomes.db")

        result = tracker.get_outcome("nonexistent")

        assert result is None

    def test_get_recent_outcomes_ordered_by_timestamp(self, tmp_path: Path):
        """Test get_recent_outcomes returns outcomes in descending timestamp order."""
        tracker = OutcomeTracker(tmp_path / "outcomes.db")

        # Create outcomes with different timestamps
        for i in range(5):
            outcome = ConsensusOutcome(
                debate_id=f"debate-{i}",
                consensus_text=f"Consensus {i}",
                consensus_confidence=0.8,
                implementation_attempted=True,
                implementation_succeeded=True,
                timestamp=f"2026-02-17T10:0{i}:00",
            )
            tracker.record_outcome(outcome)

        recent = tracker.get_recent_outcomes(limit=3)

        assert len(recent) == 3
        # Most recent first
        assert recent[0].debate_id == "debate-4"
        assert recent[1].debate_id == "debate-3"
        assert recent[2].debate_id == "debate-2"

    def test_get_recent_outcomes_respects_limit(self, tmp_path: Path):
        """Test that limit parameter works."""
        tracker = OutcomeTracker(tmp_path / "outcomes.db")

        for i in range(10):
            outcome = ConsensusOutcome(
                debate_id=f"debate-{i}",
                consensus_text="Test",
                consensus_confidence=0.8,
                implementation_attempted=True,
                implementation_succeeded=True,
            )
            tracker.record_outcome(outcome)

        recent = tracker.get_recent_outcomes(limit=5)

        assert len(recent) == 5

    def test_get_success_rate_by_confidence(self, tmp_path: Path):
        """Test success rate bucketing by confidence."""
        tracker = OutcomeTracker(tmp_path / "outcomes.db")

        # Add outcomes in different confidence buckets
        outcomes = [
            ("d1", 0.75, True),
            ("d2", 0.78, True),
            ("d3", 0.72, False),  # 0.7-0.8 bucket: 2/3 success
            ("d4", 0.85, True),
            ("d5", 0.88, True),
            ("d6", 0.89, True),  # 0.8-0.9 bucket: 3/3 success
        ]

        for debate_id, conf, success in outcomes:
            outcome = ConsensusOutcome(
                debate_id=debate_id,
                consensus_text="Test",
                consensus_confidence=conf,
                implementation_attempted=True,
                implementation_succeeded=success,
            )
            tracker.record_outcome(outcome)

        rates = tracker.get_success_rate_by_confidence()

        assert "0.7-0.8" in rates
        assert "0.8-0.9" in rates
        assert rates["0.7-0.8"] == pytest.approx(2 / 3, abs=0.01)
        assert rates["0.8-0.9"] == 1.0

    def test_get_calibration_curve(self, tmp_path: Path):
        """Test calibration curve generation."""
        tracker = OutcomeTracker(tmp_path / "outcomes.db")

        # Add outcomes in 0.7-0.8 range
        for i in range(10):
            outcome = ConsensusOutcome(
                debate_id=f"debate-{i}",
                consensus_text="Test",
                consensus_confidence=0.75,
                implementation_attempted=True,
                implementation_succeeded=i < 7,  # 70% success rate
            )
            tracker.record_outcome(outcome)

        curve = tracker.get_calibration_curve(num_buckets=10)

        assert len(curve) == 10

        # Find the 0.7-0.8 bucket (bucket 7)
        bucket_7 = curve[7]
        assert bucket_7.confidence_min == pytest.approx(0.7, abs=0.01)
        assert bucket_7.confidence_max == pytest.approx(0.8, abs=0.01)
        assert bucket_7.total_count == 10
        assert bucket_7.success_count == 7
        assert bucket_7.success_rate == 0.7

    def test_get_calibration_curve_empty_buckets(self, tmp_path: Path):
        """Test calibration curve with no data."""
        tracker = OutcomeTracker(tmp_path / "outcomes.db")

        curve = tracker.get_calibration_curve(num_buckets=5)

        assert len(curve) == 5
        for bucket in curve:
            assert bucket.total_count == 0
            assert bucket.success_count == 0

    def test_get_failure_patterns(self, tmp_path: Path):
        """Test failure pattern aggregation."""
        tracker = OutcomeTracker(tmp_path / "outcomes.db")

        failures = [
            ("d1", "Tests failed"),
            ("d2", "Tests failed"),
            ("d3", "Tests failed"),
            ("d4", "Syntax error"),
            ("d5", "Syntax error"),
            ("d6", "Import error"),
        ]

        for debate_id, reason in failures:
            outcome = ConsensusOutcome(
                debate_id=debate_id,
                consensus_text="Test",
                consensus_confidence=0.8,
                implementation_attempted=True,
                implementation_succeeded=False,
                failure_reason=reason,
            )
            tracker.record_outcome(outcome)

        patterns = tracker.get_failure_patterns(limit=2)

        assert len(patterns) == 2
        assert patterns[0]["reason"] == "Tests failed"
        assert patterns[0]["count"] == 3
        assert patterns[1]["reason"] == "Syntax error"
        assert patterns[1]["count"] == 2

    def test_get_failure_patterns_ignores_successes(self, tmp_path: Path):
        """Test that success outcomes are not included in failure patterns."""
        tracker = OutcomeTracker(tmp_path / "outcomes.db")

        outcome_success = ConsensusOutcome(
            debate_id="d1",
            consensus_text="Test",
            consensus_confidence=0.9,
            implementation_attempted=True,
            implementation_succeeded=True,
            failure_reason=None,
        )
        tracker.record_outcome(outcome_success)

        outcome_failure = ConsensusOutcome(
            debate_id="d2",
            consensus_text="Test",
            consensus_confidence=0.8,
            implementation_attempted=True,
            implementation_succeeded=False,
            failure_reason="Error",
        )
        tracker.record_outcome(outcome_failure)

        patterns = tracker.get_failure_patterns()

        assert len(patterns) == 1
        assert patterns[0]["reason"] == "Error"

    def test_get_overall_stats(self, tmp_path: Path):
        """Test overall statistics calculation."""
        tracker = OutcomeTracker(tmp_path / "outcomes.db")

        outcomes = [
            ConsensusOutcome(
                debate_id="d1",
                consensus_text="Test",
                consensus_confidence=0.8,
                implementation_attempted=True,
                implementation_succeeded=True,
                tests_passed=10,
            ),
            ConsensusOutcome(
                debate_id="d2",
                consensus_text="Test",
                consensus_confidence=0.9,
                implementation_attempted=True,
                implementation_succeeded=True,
                tests_passed=20,
            ),
            ConsensusOutcome(
                debate_id="d3",
                consensus_text="Test",
                consensus_confidence=0.7,
                implementation_attempted=True,
                implementation_succeeded=False,
                tests_failed=5,
                rollback_triggered=True,
            ),
        ]

        for outcome in outcomes:
            tracker.record_outcome(outcome)

        stats = tracker.get_overall_stats()

        assert stats["total_outcomes"] == 3
        assert stats["attempted"] == 3
        assert stats["succeeded"] == 2
        assert stats["success_rate"] == pytest.approx(2 / 3, abs=0.01)
        assert stats["rollbacks"] == 1
        assert stats["avg_confidence"] == pytest.approx(0.8, abs=0.01)
        assert stats["total_tests_passed"] == 30
        assert stats["total_tests_failed"] == 5

    def test_get_overall_stats_empty_db(self, tmp_path: Path):
        """Test overall stats with no data."""
        tracker = OutcomeTracker(tmp_path / "outcomes.db")

        stats = tracker.get_overall_stats()

        assert stats["total_outcomes"] == 0
        assert stats["attempted"] == 0
        assert stats["succeeded"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["rollbacks"] == 0
        assert stats["avg_confidence"] == 0.0

    def test_is_overconfident_with_overconfident_data(self, tmp_path: Path):
        """Test is_overconfident detects overconfident system."""
        tracker = OutcomeTracker(tmp_path / "outcomes.db")

        # High confidence (0.85 avg) but low success (60%)
        for i in range(10):
            outcome = ConsensusOutcome(
                debate_id=f"debate-{i}",
                consensus_text="Test",
                consensus_confidence=0.85,
                implementation_attempted=True,
                implementation_succeeded=i < 6,  # 60% success
            )
            tracker.record_outcome(outcome)

        is_overconf = tracker.is_overconfident(threshold=0.7)

        # avg_confidence=0.85, success_rate=0.6
        # 0.85 > 0.6 + 0.1 = True
        assert is_overconf is True

    def test_is_overconfident_with_well_calibrated_data(self, tmp_path: Path):
        """Test is_overconfident with well-calibrated system."""
        tracker = OutcomeTracker(tmp_path / "outcomes.db")

        # Confidence 0.85, success 82%
        for i in range(10):
            outcome = ConsensusOutcome(
                debate_id=f"debate-{i}",
                consensus_text="Test",
                consensus_confidence=0.85,
                implementation_attempted=True,
                implementation_succeeded=i < 8,  # 80% success
            )
            tracker.record_outcome(outcome)

        is_overconf = tracker.is_overconfident(threshold=0.7)

        # avg_confidence=0.85, success_rate=0.8
        # 0.85 > 0.8 + 0.1 = False
        assert is_overconf is False

    def test_is_overconfident_insufficient_samples(self, tmp_path: Path):
        """Test is_overconfident returns False with <5 samples."""
        tracker = OutcomeTracker(tmp_path / "outcomes.db")

        # Only 3 samples
        for i in range(3):
            outcome = ConsensusOutcome(
                debate_id=f"debate-{i}",
                consensus_text="Test",
                consensus_confidence=0.9,
                implementation_attempted=True,
                implementation_succeeded=False,  # All fail but too few samples
            )
            tracker.record_outcome(outcome)

        is_overconf = tracker.is_overconfident(threshold=0.7)

        assert is_overconf is False

    def test_is_overconfident_respects_threshold(self, tmp_path: Path):
        """Test is_overconfident only considers outcomes above threshold."""
        tracker = OutcomeTracker(tmp_path / "outcomes.db")

        # Low confidence outcomes (below threshold) - well calibrated
        for i in range(5):
            outcome = ConsensusOutcome(
                debate_id=f"low-{i}",
                consensus_text="Test",
                consensus_confidence=0.5,
                implementation_attempted=True,
                implementation_succeeded=i < 3,  # 60% success
            )
            tracker.record_outcome(outcome)

        # High confidence outcomes (above threshold) - overconfident
        for i in range(10):
            outcome = ConsensusOutcome(
                debate_id=f"high-{i}",
                consensus_text="Test",
                consensus_confidence=0.9,
                implementation_attempted=True,
                implementation_succeeded=i < 5,  # 50% success
            )
            tracker.record_outcome(outcome)

        is_overconf = tracker.is_overconfident(threshold=0.7)

        # Should only look at high confidence ones
        # avg_confidence=0.9, success_rate=0.5
        # 0.9 > 0.5 + 0.1 = True
        assert is_overconf is True

    def test_get_calibration_adjustment_overconfident(self, tmp_path: Path):
        """Test calibration adjustment for overconfident system."""
        tracker = OutcomeTracker(tmp_path / "outcomes.db")

        # High confidence (0.85) but low success (65%)
        # Error = 0.85 - 0.65 = 0.2
        for i in range(10):
            outcome = ConsensusOutcome(
                debate_id=f"debate-{i}",
                consensus_text="Test",
                consensus_confidence=0.85,
                implementation_attempted=True,
                implementation_succeeded=i < 7,  # 70% success (bucket 0.7-0.8 has 10 items)
            )
            tracker.record_outcome(outcome)

        adjustment = tracker.get_calibration_adjustment()

        # avg error in high-conf buckets should be positive
        # adjustment = 1.0 - error, so < 1.0
        assert adjustment < 1.0
        assert adjustment >= 0.5  # Clamped

    def test_get_calibration_adjustment_underconfident(self, tmp_path: Path):
        """Test calibration adjustment for underconfident system."""
        tracker = OutcomeTracker(tmp_path / "outcomes.db")

        # Confidence 0.75, success 90%
        # Error = 0.75 - 0.9 = -0.15
        for i in range(10):
            outcome = ConsensusOutcome(
                debate_id=f"debate-{i}",
                consensus_text="Test",
                consensus_confidence=0.75,
                implementation_attempted=True,
                implementation_succeeded=i < 9,  # 90% success
            )
            tracker.record_outcome(outcome)

        adjustment = tracker.get_calibration_adjustment()

        # negative error -> adjustment > 1.0
        assert adjustment > 1.0
        assert adjustment <= 1.5  # Clamped

    def test_get_calibration_adjustment_no_data(self, tmp_path: Path):
        """Test calibration adjustment with no high-confidence data."""
        tracker = OutcomeTracker(tmp_path / "outcomes.db")

        # Only low confidence outcomes
        for i in range(5):
            outcome = ConsensusOutcome(
                debate_id=f"debate-{i}",
                consensus_text="Test",
                consensus_confidence=0.5,
                implementation_attempted=True,
                implementation_succeeded=True,
            )
            tracker.record_outcome(outcome)

        adjustment = tracker.get_calibration_adjustment()

        # No high-confidence data -> return 1.0
        assert adjustment == 1.0

    def test_get_calibration_adjustment_clamping(self, tmp_path: Path):
        """Test that calibration adjustment is clamped to [0.5, 1.5]."""
        tracker = OutcomeTracker(tmp_path / "outcomes.db")

        # Extreme overconfidence: 0.95 confidence, 0% success
        # Error = 0.95 - 0.0 = 0.95
        # adjustment = 1.0 - 0.95 = 0.05 -> clamped to 0.5
        for i in range(10):
            outcome = ConsensusOutcome(
                debate_id=f"debate-{i}",
                consensus_text="Test",
                consensus_confidence=0.95,
                implementation_attempted=True,
                implementation_succeeded=False,
            )
            tracker.record_outcome(outcome)

        adjustment = tracker.get_calibration_adjustment()

        assert adjustment == 0.5  # Clamped to minimum


# ============================================================================
# AsyncOutcomeTracker Tests
# ============================================================================


class TestAsyncOutcomeTracker:
    """Tests for AsyncOutcomeTracker async wrapper."""

    @pytest.mark.asyncio
    async def test_record_outcome_async(self, tmp_path: Path):
        """Test async record_outcome."""
        tracker = AsyncOutcomeTracker(tmp_path / "async_outcomes.db")

        outcome = ConsensusOutcome(
            debate_id="async-1",
            consensus_text="Async test",
            consensus_confidence=0.8,
            implementation_attempted=True,
            implementation_succeeded=True,
        )

        await tracker.record_outcome(outcome)

        # Verify it was stored
        retrieved = await tracker.get_outcome("async-1")
        assert retrieved is not None
        assert retrieved.debate_id == "async-1"

    @pytest.mark.asyncio
    async def test_get_outcome_async(self, tmp_path: Path):
        """Test async get_outcome."""
        tracker = AsyncOutcomeTracker(tmp_path / "async_outcomes.db")

        outcome = ConsensusOutcome(
            debate_id="async-2",
            consensus_text="Test",
            consensus_confidence=0.75,
            implementation_attempted=True,
            implementation_succeeded=True,
        )
        await tracker.record_outcome(outcome)

        retrieved = await tracker.get_outcome("async-2")

        assert retrieved is not None
        assert retrieved.consensus_confidence == 0.75

    @pytest.mark.asyncio
    async def test_get_outcome_not_found_async(self, tmp_path: Path):
        """Test async get_outcome returns None for missing debate."""
        tracker = AsyncOutcomeTracker(tmp_path / "async_outcomes.db")

        result = await tracker.get_outcome("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_recent_outcomes_async(self, tmp_path: Path):
        """Test async get_recent_outcomes."""
        tracker = AsyncOutcomeTracker(tmp_path / "async_outcomes.db")

        for i in range(5):
            outcome = ConsensusOutcome(
                debate_id=f"async-{i}",
                consensus_text="Test",
                consensus_confidence=0.8,
                implementation_attempted=True,
                implementation_succeeded=True,
            )
            await tracker.record_outcome(outcome)

        recent = await tracker.get_recent_outcomes(limit=3)

        assert len(recent) == 3

    @pytest.mark.asyncio
    async def test_get_success_rate_by_confidence_async(self, tmp_path: Path):
        """Test async get_success_rate_by_confidence."""
        tracker = AsyncOutcomeTracker(tmp_path / "async_outcomes.db")

        for i in range(3):
            outcome = ConsensusOutcome(
                debate_id=f"async-{i}",
                consensus_text="Test",
                consensus_confidence=0.75,
                implementation_attempted=True,
                implementation_succeeded=i < 2,  # 2/3 success
            )
            await tracker.record_outcome(outcome)

        rates = await tracker.get_success_rate_by_confidence()

        assert "0.7-0.8" in rates
        assert rates["0.7-0.8"] == pytest.approx(2 / 3, abs=0.01)

    @pytest.mark.asyncio
    async def test_get_calibration_curve_async(self, tmp_path: Path):
        """Test async get_calibration_curve."""
        tracker = AsyncOutcomeTracker(tmp_path / "async_outcomes.db")

        for i in range(5):
            outcome = ConsensusOutcome(
                debate_id=f"async-{i}",
                consensus_text="Test",
                consensus_confidence=0.85,
                implementation_attempted=True,
                implementation_succeeded=i < 4,  # 80% success
            )
            await tracker.record_outcome(outcome)

        curve = await tracker.get_calibration_curve(num_buckets=10)

        assert len(curve) == 10

    @pytest.mark.asyncio
    async def test_get_failure_patterns_async(self, tmp_path: Path):
        """Test async get_failure_patterns."""
        tracker = AsyncOutcomeTracker(tmp_path / "async_outcomes.db")

        for i in range(3):
            outcome = ConsensusOutcome(
                debate_id=f"async-{i}",
                consensus_text="Test",
                consensus_confidence=0.8,
                implementation_attempted=True,
                implementation_succeeded=False,
                failure_reason="Async error",
            )
            await tracker.record_outcome(outcome)

        patterns = await tracker.get_failure_patterns()

        assert len(patterns) == 1
        assert patterns[0]["reason"] == "Async error"
        assert patterns[0]["count"] == 3

    @pytest.mark.asyncio
    async def test_get_overall_stats_async(self, tmp_path: Path):
        """Test async get_overall_stats."""
        tracker = AsyncOutcomeTracker(tmp_path / "async_outcomes.db")

        for i in range(4):
            outcome = ConsensusOutcome(
                debate_id=f"async-{i}",
                consensus_text="Test",
                consensus_confidence=0.8,
                implementation_attempted=True,
                implementation_succeeded=i < 3,  # 3/4 success
                tests_passed=10 if i < 3 else 0,
            )
            await tracker.record_outcome(outcome)

        stats = await tracker.get_overall_stats()

        assert stats["total_outcomes"] == 4
        assert stats["attempted"] == 4
        assert stats["succeeded"] == 3
        assert stats["success_rate"] == 0.75
        assert stats["total_tests_passed"] == 30

    @pytest.mark.asyncio
    async def test_is_overconfident_async(self, tmp_path: Path):
        """Test async is_overconfident."""
        tracker = AsyncOutcomeTracker(tmp_path / "async_outcomes.db")

        # Overconfident: 0.9 confidence, 50% success
        for i in range(10):
            outcome = ConsensusOutcome(
                debate_id=f"async-{i}",
                consensus_text="Test",
                consensus_confidence=0.9,
                implementation_attempted=True,
                implementation_succeeded=i < 5,
            )
            await tracker.record_outcome(outcome)

        is_overconf = await tracker.is_overconfident(threshold=0.7)

        assert is_overconf is True

    @pytest.mark.asyncio
    async def test_get_calibration_adjustment_async(self, tmp_path: Path):
        """Test async get_calibration_adjustment."""
        tracker = AsyncOutcomeTracker(tmp_path / "async_outcomes.db")

        for i in range(10):
            outcome = ConsensusOutcome(
                debate_id=f"async-{i}",
                consensus_text="Test",
                consensus_confidence=0.85,
                implementation_attempted=True,
                implementation_succeeded=i < 6,  # 60% success, overconfident
            )
            await tracker.record_outcome(outcome)

        adjustment = await tracker.get_calibration_adjustment()

        assert adjustment < 1.0  # Overconfident -> lower multiplier

    @pytest.mark.asyncio
    async def test_db_path_property(self, tmp_path: Path):
        """Test db_path property."""
        db_path = tmp_path / "async_outcomes.db"
        tracker = AsyncOutcomeTracker(db_path)

        assert tracker.db_path == db_path
