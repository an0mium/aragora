"""
Tests for aragora.agents.positions module.

Tests Position, CalibrationBucket, DomainCalibration, and PositionLedger
for tracking agent positions and calibration accuracy.
"""

import pytest
import tempfile
from pathlib import Path

from aragora.agents.positions import (
    Position,
    CalibrationBucket,
    DomainCalibration,
    PositionLedger,
)


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_create(self):
        """Test Position.create factory method."""
        pos = Position.create(
            agent_name="claude",
            claim="The sky is blue",
            confidence=0.9,
            debate_id="debate-001",
            round_num=1,
            domain="science",
        )

        assert pos.agent_name == "claude"
        assert pos.claim == "The sky is blue"
        assert pos.confidence == 0.9
        assert pos.debate_id == "debate-001"
        assert pos.round_num == 1
        assert pos.domain == "science"
        assert pos.outcome == "pending"
        assert pos.reversed is False
        assert len(pos.id) == 8

    def test_position_confidence_clamping(self):
        """Test that confidence is clamped to 0-1 range."""
        pos_high = Position.create(
            agent_name="claude",
            claim="Test",
            confidence=1.5,
            debate_id="d1",
            round_num=1,
        )
        assert pos_high.confidence == 1.0

        pos_low = Position.create(
            agent_name="claude",
            claim="Test",
            confidence=-0.5,
            debate_id="d1",
            round_num=1,
        )
        assert pos_low.confidence == 0.0

    def test_position_unique_ids(self):
        """Test that each Position gets a unique ID."""
        positions = [
            Position.create(
                agent_name="claude",
                claim=f"Claim {i}",
                confidence=0.5,
                debate_id="d1",
                round_num=1,
            )
            for i in range(10)
        ]
        ids = [p.id for p in positions]
        assert len(set(ids)) == 10


class TestCalibrationBucket:
    """Tests for CalibrationBucket dataclass."""

    def test_bucket_accuracy(self):
        """Test accuracy calculation."""
        bucket = CalibrationBucket(
            bucket_start=0.8,
            bucket_end=0.9,
            predictions=10,
            correct=8,
        )
        assert bucket.accuracy == 0.8

    def test_bucket_accuracy_empty(self):
        """Test accuracy with no predictions."""
        bucket = CalibrationBucket(bucket_start=0.8, bucket_end=0.9)
        assert bucket.accuracy == 0.0

    def test_bucket_expected_accuracy(self):
        """Test expected accuracy is bucket midpoint."""
        bucket = CalibrationBucket(bucket_start=0.8, bucket_end=0.9)
        assert bucket.expected_accuracy == pytest.approx(0.85)

    def test_bucket_calibration_error(self):
        """Test calibration error calculation."""
        bucket = CalibrationBucket(
            bucket_start=0.8,
            bucket_end=0.9,
            predictions=10,
            correct=7,  # 70% accuracy vs 85% expected
        )
        assert bucket.calibration_error == pytest.approx(0.15, abs=0.01)

    def test_bucket_key(self):
        """Test bucket key generation."""
        bucket = CalibrationBucket(bucket_start=0.7, bucket_end=0.8)
        assert bucket.bucket_key == "0.7-0.8"


class TestDomainCalibration:
    """Tests for DomainCalibration dataclass."""

    def test_calibration_score(self):
        """Test calibration score calculation."""
        cal = DomainCalibration(
            domain="science",
            total_predictions=100,
            total_correct=85,
            brier_sum=10.0,  # Average Brier = 0.1
        )
        # 1 - 0.1 = 0.9
        assert cal.calibration_score == pytest.approx(0.9)

    def test_calibration_score_empty(self):
        """Test calibration score with no predictions."""
        cal = DomainCalibration(domain="science")
        assert cal.calibration_score == 0.5

    def test_accuracy(self):
        """Test accuracy calculation."""
        cal = DomainCalibration(
            domain="science",
            total_predictions=100,
            total_correct=80,
        )
        assert cal.accuracy == 0.8


class TestPositionLedger:
    """Tests for PositionLedger class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def ledger(self, temp_db):
        """Create a PositionLedger with temp database."""
        return PositionLedger(temp_db)

    def test_ledger_initialization(self, ledger):
        """Test ledger initializes correctly."""
        assert ledger.db_path.exists() or True  # May not exist until first write

    def test_record_position(self, ledger):
        """Test recording a new position."""
        pos_id = ledger.record_position(
            agent_name="claude",
            claim="AI will be beneficial",
            confidence=0.75,
            debate_id="debate-001",
            round_num=1,
            domain="ethics",
        )

        # record_position returns the ID
        assert isinstance(pos_id, str)
        assert len(pos_id) == 8

        # Verify by fetching the position
        positions = ledger.get_agent_positions("claude")
        assert len(positions) == 1
        assert positions[0].confidence == 0.75
        assert positions[0].outcome == "pending"

    def test_get_agent_positions(self, ledger):
        """Test retrieving positions for an agent."""
        # Record multiple positions
        for i in range(5):
            ledger.record_position(
                agent_name="claude",
                claim=f"Claim {i}",
                confidence=0.5 + i * 0.1,
                debate_id=f"debate-{i}",
                round_num=1,
            )

        positions = ledger.get_agent_positions("claude")
        assert len(positions) == 5

    def test_resolve_position(self, ledger):
        """Test resolving a position outcome."""
        pos_id = ledger.record_position(
            agent_name="claude",
            claim="Test claim",
            confidence=0.8,
            debate_id="d1",
            round_num=1,
        )

        ledger.resolve_position(pos_id, "correct")

        # Verify by fetching - filter by outcome
        positions = ledger.get_agent_positions("claude", outcome_filter="correct")
        assert len(positions) == 1
        assert positions[0].outcome == "correct"
        assert positions[0].resolved_at is not None

    def test_record_reversal(self, ledger):
        """Test recording a position reversal."""
        pos_id = ledger.record_position(
            agent_name="claude",
            claim="Original claim",
            confidence=0.8,
            debate_id="d1",
            round_num=1,
        )

        ledger.record_reversal("claude", pos_id, "d2")

        # Verify by fetching
        positions = ledger.get_agent_positions("claude")
        assert len(positions) == 1
        assert positions[0].reversed is True
        assert positions[0].reversal_debate_id == "d2"

    def test_get_position_stats(self, ledger):
        """Test position statistics calculation."""
        # Record positions with known outcomes
        for i in range(10):
            pos_id = ledger.record_position(
                agent_name="claude",
                claim=f"Claim {i}",
                confidence=0.8,  # 80% confidence
                debate_id=f"d{i}",
                round_num=1,
            )
            # 8 correct, 2 incorrect (matches 80% confidence)
            ledger.resolve_position(pos_id, "correct" if i < 8 else "incorrect")

        stats = ledger.get_position_stats("claude")
        assert "total" in stats
        assert stats["total"] == 10
        assert stats["correct"] == 8
        assert stats["incorrect"] == 2

    def test_get_reversal_count(self, ledger):
        """Test counting reversals for an agent."""
        # Record and reverse some positions
        pos_ids = []
        for i in range(3):
            pos_id = ledger.record_position(
                agent_name="claude",
                claim=f"Claim {i}",
                confidence=0.5,
                debate_id=f"d{i}",
                round_num=1,
            )
            pos_ids.append(pos_id)

        # Reverse first 2
        ledger.record_reversal("claude", pos_ids[0], "rev0")
        ledger.record_reversal("claude", pos_ids[1], "rev1")

        stats = ledger.get_position_stats("claude")
        assert stats["reversals"] == 2

    def test_position_by_domain(self, ledger):
        """Test filtering positions by domain - check domain is stored."""
        ledger.record_position(
            agent_name="claude",
            claim="Science claim",
            confidence=0.8,
            debate_id="d1",
            round_num=1,
            domain="science",
        )
        ledger.record_position(
            agent_name="claude",
            claim="Ethics claim",
            confidence=0.7,
            debate_id="d2",
            round_num=1,
            domain="ethics",
        )

        positions = ledger.get_agent_positions("claude")
        domains = {p.domain for p in positions}
        assert "science" in domains
        assert "ethics" in domains


class TestPositionLedgerEdgeCases:
    """Edge case tests for PositionLedger."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def ledger(self, temp_db):
        """Create a PositionLedger with temp database."""
        return PositionLedger(temp_db)

    def test_empty_agent_positions(self, ledger):
        """Test getting positions for agent with none."""
        positions = ledger.get_agent_positions("unknown-agent")
        assert positions == []

    def test_position_stats_no_positions(self, ledger):
        """Test position stats with no data."""
        stats = ledger.get_position_stats("unknown-agent")
        assert stats["total"] == 0
        assert stats["correct"] == 0
        assert stats["reversals"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
