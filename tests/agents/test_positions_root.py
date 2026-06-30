"""
Tests for agents/positions.py - Position tracking and calibration.

Tests cover:
- Position dataclass creation and validation
- CalibrationBucket calculations
- DomainCalibration scoring
- PositionLedger CRUD operations
- Position resolution and reversal tracking
- Domain detection
- Statistics aggregation
"""

import pytest
import tempfile
import os
from pathlib import Path

from aragora.agents.positions import (
    Position,
    CalibrationBucket,
    DomainCalibration,
    PositionLedger,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def ledger(temp_db):
    """Create a PositionLedger with temporary database."""
    return PositionLedger(db_path=temp_db)


# =============================================================================
# Position Dataclass Tests
# =============================================================================


class TestPosition:
    """Tests for Position dataclass."""

    def test_create_position(self):
        """Should create position with generated ID."""
        pos = Position.create(
            agent_name="claude",
            claim="AI should be regulated",
            confidence=0.8,
            debate_id="debate-001",
            round_num=1,
            domain="policy",
        )

        assert pos.id is not None
        assert len(pos.id) == 8  # UUID[:8]
        assert pos.agent_name == "claude"
        assert pos.claim == "AI should be regulated"
        assert pos.confidence == 0.8
        assert pos.debate_id == "debate-001"
        assert pos.round_num == 1
        assert pos.domain == "policy"
        assert pos.outcome == "pending"
        assert pos.reversed is False

    def test_confidence_clamped_to_range(self):
        """Should clamp confidence to 0.0-1.0 range."""
        pos_high = Position.create(
            agent_name="claude",
            claim="test",
            confidence=1.5,  # Above max
            debate_id="d1",
            round_num=1,
        )
        assert pos_high.confidence == 1.0

        pos_low = Position.create(
            agent_name="claude",
            claim="test",
            confidence=-0.5,  # Below min
            debate_id="d1",
            round_num=1,
        )
        assert pos_low.confidence == 0.0

    def test_create_sets_timestamp(self):
        """Should set created_at timestamp."""
        pos = Position.create(
            agent_name="claude",
            claim="test",
            confidence=0.5,
            debate_id="d1",
            round_num=1,
        )
        assert pos.created_at is not None
        assert "T" in pos.created_at  # ISO format


# =============================================================================
# CalibrationBucket Tests
# =============================================================================


class TestCalibrationBucket:
    """Tests for CalibrationBucket dataclass."""

    def test_accuracy_calculation(self):
        """Should calculate accuracy correctly."""
        bucket = CalibrationBucket(
            bucket_start=0.8,
            bucket_end=0.9,
            predictions=10,
            correct=8,
        )
        assert bucket.accuracy == 0.8

    def test_accuracy_zero_predictions(self):
        """Should return 0.0 accuracy for zero predictions."""
        bucket = CalibrationBucket(
            bucket_start=0.8,
            bucket_end=0.9,
            predictions=0,
            correct=0,
        )
        assert bucket.accuracy == 0.0

    def test_expected_accuracy(self):
        """Should return midpoint as expected accuracy."""
        bucket = CalibrationBucket(
            bucket_start=0.8,
            bucket_end=0.9,
        )
        assert bucket.expected_accuracy == pytest.approx(0.85)

    def test_calibration_error(self):
        """Should calculate calibration error as |accuracy - expected|."""
        bucket = CalibrationBucket(
            bucket_start=0.8,
            bucket_end=0.9,
            predictions=10,
            correct=7,  # 70% accuracy, expected 85%
        )
        assert bucket.calibration_error == pytest.approx(0.15, abs=0.01)

    def test_bucket_key(self):
        """Should format bucket key correctly."""
        bucket = CalibrationBucket(bucket_start=0.8, bucket_end=0.9)
        assert bucket.bucket_key == "0.8-0.9"

    def test_well_calibrated_bucket(self):
        """Should have low error when well calibrated."""
        bucket = CalibrationBucket(
            bucket_start=0.7,
            bucket_end=0.8,
            predictions=100,
            correct=75,  # 75% accuracy, expected 75%
        )
        assert bucket.calibration_error < 0.01


# =============================================================================
# DomainCalibration Tests
# =============================================================================


class TestDomainCalibration:
    """Tests for DomainCalibration dataclass."""

    def test_calibration_score_no_predictions(self):
        """Should return 0.5 for no predictions."""
        dc = DomainCalibration(domain="security")
        assert dc.calibration_score == 0.5

    def test_calibration_score_calculation(self):
        """Should calculate 1 - avg_brier_score."""
        dc = DomainCalibration(
            domain="security",
            total_predictions=10,
            brier_sum=2.0,  # avg brier = 0.2
        )
        assert dc.calibration_score == pytest.approx(0.8)

    def test_accuracy_calculation(self):
        """Should calculate overall accuracy."""
        dc = DomainCalibration(
            domain="security",
            total_predictions=20,
            total_correct=15,
        )
        assert dc.accuracy == 0.75

    def test_accuracy_zero_predictions(self):
        """Should return 0.0 for zero predictions."""
        dc = DomainCalibration(domain="security")
        assert dc.accuracy == 0.0


# =============================================================================
# PositionLedger Tests - Basic Operations
# =============================================================================


class TestPositionLedgerBasic:
    """Tests for basic PositionLedger operations."""

    def test_init_creates_tables(self, ledger):
        """Should create positions table on init."""
        # Verify table exists by recording a position
        pos_id = ledger.record_position(
            agent_name="claude",
            claim="test claim",
            confidence=0.8,
            debate_id="d1",
            round_num=1,
        )
        assert pos_id is not None

    def test_record_position(self, ledger):
        """Should record a new position and return ID."""
        pos_id = ledger.record_position(
            agent_name="gpt4",
            claim="Testing is important",
            confidence=0.9,
            debate_id="debate-123",
            round_num=2,
            domain="testing",
        )

        assert pos_id is not None
        assert len(pos_id) == 8

    def test_record_multiple_positions(self, ledger):
        """Should record multiple positions with unique IDs."""
        ids = []
        for i in range(5):
            pos_id = ledger.record_position(
                agent_name=f"agent-{i}",
                claim=f"claim-{i}",
                confidence=0.5 + i * 0.1,
                debate_id="d1",
                round_num=1,
            )
            ids.append(pos_id)

        # All IDs should be unique
        assert len(set(ids)) == 5


# =============================================================================
# PositionLedger Tests - Resolution
# =============================================================================


class TestPositionLedgerResolution:
    """Tests for position resolution."""

    def test_resolve_position_correct(self, ledger):
        """Should mark position as correct."""
        pos_id = ledger.record_position(
            agent_name="claude",
            claim="test",
            confidence=0.8,
            debate_id="d1",
            round_num=1,
        )

        ledger.resolve_position(pos_id, "correct")

        positions = ledger.get_agent_positions("claude")
        assert len(positions) == 1
        assert positions[0].outcome == "correct"
        assert positions[0].resolved_at is not None

    def test_resolve_position_incorrect(self, ledger):
        """Should mark position as incorrect."""
        pos_id = ledger.record_position(
            agent_name="claude",
            claim="test",
            confidence=0.8,
            debate_id="d1",
            round_num=1,
        )

        ledger.resolve_position(pos_id, "incorrect")

        positions = ledger.get_agent_positions("claude")
        assert positions[0].outcome == "incorrect"

    def test_resolve_position_unresolved(self, ledger):
        """Should mark position as unresolved."""
        pos_id = ledger.record_position(
            agent_name="claude",
            claim="test",
            confidence=0.8,
            debate_id="d1",
            round_num=1,
        )

        ledger.resolve_position(pos_id, "unresolved")

        positions = ledger.get_agent_positions("claude")
        assert positions[0].outcome == "unresolved"


# =============================================================================
# PositionLedger Tests - Reversal
# =============================================================================


class TestPositionLedgerReversal:
    """Tests for position reversal tracking."""

    def test_record_reversal(self, ledger):
        """Should mark position as reversed."""
        pos_id = ledger.record_position(
            agent_name="claude",
            claim="AI is safe",
            confidence=0.7,
            debate_id="d1",
            round_num=1,
        )

        ledger.record_reversal(
            agent_name="claude",
            original_position_id=pos_id,
            new_debate_id="d2",
        )

        positions = ledger.get_agent_positions("claude")
        assert positions[0].reversed is True
        assert positions[0].reversal_debate_id == "d2"

    def test_reversal_wrong_agent_no_effect(self, ledger):
        """Should not reverse if agent name doesn't match."""
        pos_id = ledger.record_position(
            agent_name="claude",
            claim="test",
            confidence=0.7,
            debate_id="d1",
            round_num=1,
        )

        # Try to reverse with wrong agent
        ledger.record_reversal(
            agent_name="gpt4",
            original_position_id=pos_id,
            new_debate_id="d2",
        )

        positions = ledger.get_agent_positions("claude")
        assert positions[0].reversed is False


# =============================================================================
# PositionLedger Tests - Queries
# =============================================================================


class TestPositionLedgerQueries:
    """Tests for position querying."""

    def test_get_agent_positions(self, ledger):
        """Should return positions for specific agent."""
        # Record positions for multiple agents
        ledger.record_position("claude", "c1", 0.8, "d1", 1)
        ledger.record_position("claude", "c2", 0.9, "d1", 2)
        ledger.record_position("gpt4", "g1", 0.7, "d1", 1)

        positions = ledger.get_agent_positions("claude")
        assert len(positions) == 2
        assert all(p.agent_name == "claude" for p in positions)

    def test_get_agent_positions_with_limit(self, ledger):
        """Should respect limit parameter."""
        for i in range(10):
            ledger.record_position("claude", f"c{i}", 0.5, "d1", i)

        positions = ledger.get_agent_positions("claude", limit=3)
        assert len(positions) == 3

    def test_get_agent_positions_with_outcome_filter(self, ledger):
        """Should filter by outcome."""
        pos1 = ledger.record_position("claude", "c1", 0.8, "d1", 1)
        pos2 = ledger.record_position("claude", "c2", 0.9, "d1", 2)

        ledger.resolve_position(pos1, "correct")
        ledger.resolve_position(pos2, "incorrect")

        correct_positions = ledger.get_agent_positions("claude", outcome_filter="correct")
        assert len(correct_positions) == 1
        assert correct_positions[0].outcome == "correct"

    def test_get_positions_for_debate(self, ledger):
        """Should return all positions from a specific debate."""
        ledger.record_position("claude", "c1", 0.8, "debate-A", 1)
        ledger.record_position("gpt4", "g1", 0.7, "debate-A", 1)
        ledger.record_position("claude", "c2", 0.9, "debate-B", 1)

        positions = ledger.get_positions_for_debate("debate-A")
        assert len(positions) == 2
        assert all(p.debate_id == "debate-A" for p in positions)

    def test_positions_ordered_by_round(self, ledger):
        """Should return positions ordered by round."""
        ledger.record_position("claude", "c3", 0.8, "d1", 3)
        ledger.record_position("claude", "c1", 0.8, "d1", 1)
        ledger.record_position("claude", "c2", 0.8, "d1", 2)

        positions = ledger.get_positions_for_debate("d1")
        round_nums = [p.round_num for p in positions]
        assert round_nums == [1, 2, 3]


# =============================================================================
# PositionLedger Tests - Statistics
# =============================================================================


class TestPositionLedgerStats:
    """Tests for position statistics."""

    def test_get_position_stats_empty(self, ledger):
        """Should return zero stats for unknown agent."""
        stats = ledger.get_position_stats("unknown")
        assert stats["total"] == 0
        assert stats["correct"] == 0
        assert stats["incorrect"] == 0

    def test_get_position_stats(self, ledger):
        """Should calculate aggregate statistics."""
        # Record and resolve multiple positions
        pos1 = ledger.record_position("claude", "c1", 0.9, "d1", 1)
        pos2 = ledger.record_position("claude", "c2", 0.8, "d1", 2)
        pos3 = ledger.record_position("claude", "c3", 0.7, "d1", 3)
        pos4 = ledger.record_position("claude", "c4", 0.6, "d1", 4)

        ledger.resolve_position(pos1, "correct")
        ledger.resolve_position(pos2, "correct")
        ledger.resolve_position(pos3, "incorrect")
        # pos4 remains pending

        stats = ledger.get_position_stats("claude")
        assert stats["total"] == 4
        assert stats["correct"] == 2
        assert stats["incorrect"] == 1
        assert stats["pending"] == 1

    def test_stats_include_reversals(self, ledger):
        """Should count reversals in stats."""
        pos1 = ledger.record_position("claude", "c1", 0.8, "d1", 1)
        pos2 = ledger.record_position("claude", "c2", 0.9, "d1", 2)

        ledger.record_reversal("claude", pos1, "d2")

        stats = ledger.get_position_stats("claude")
        assert stats["reversals"] == 1

    def test_avg_confidence_by_outcome(self, ledger):
        """Should calculate average confidence per outcome."""
        pos1 = ledger.record_position("claude", "c1", 0.9, "d1", 1)
        pos2 = ledger.record_position("claude", "c2", 0.8, "d1", 2)
        pos3 = ledger.record_position("claude", "c3", 0.5, "d1", 3)

        ledger.resolve_position(pos1, "correct")
        ledger.resolve_position(pos2, "correct")
        ledger.resolve_position(pos3, "incorrect")

        stats = ledger.get_position_stats("claude")
        assert stats["avg_confidence_when_correct"] == pytest.approx(0.85)
        assert stats["avg_confidence_when_incorrect"] == pytest.approx(0.5)


# =============================================================================
# Domain Detection Tests
# =============================================================================


class TestDomainDetection:
    """Tests for domain detection."""

    def test_detect_security_domain(self, ledger):
        """Should detect security domain."""
        domain = ledger.detect_domain("XSS vulnerability in user input")
        assert domain == "security"

    def test_detect_performance_domain(self, ledger):
        """Should detect performance domain."""
        domain = ledger.detect_domain("Optimize query latency")
        assert domain == "performance"

    def test_detect_testing_domain(self, ledger):
        """Should detect testing domain."""
        domain = ledger.detect_domain("Add unit test coverage")
        assert domain == "testing"

    def test_detect_database_domain(self, ledger):
        """Should detect database domain."""
        domain = ledger.detect_domain("Add index to query")
        assert domain == "database"

    def test_detect_frontend_domain(self, ledger):
        """Should detect frontend domain."""
        domain = ledger.detect_domain("Fix responsive CSS")
        assert domain == "frontend"

    def test_detect_no_domain(self, ledger):
        """Should return None for unrecognized content."""
        domain = ledger.detect_domain("Random unrelated content")
        assert domain is None

    def test_detect_case_insensitive(self, ledger):
        """Should be case insensitive."""
        domain = ledger.detect_domain("SECURITY VULNERABILITY FOUND")
        assert domain == "security"


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_claim(self, ledger):
        """Should handle empty claim."""
        pos_id = ledger.record_position(
            agent_name="claude",
            claim="",
            confidence=0.5,
            debate_id="d1",
            round_num=1,
        )
        assert pos_id is not None

    def test_special_characters_in_claim(self, ledger):
        """Should handle special characters."""
        claim = "User said: \"It's <dangerous>\" & needs 'escaping'"
        pos_id = ledger.record_position(
            agent_name="claude",
            claim=claim,
            confidence=0.5,
            debate_id="d1",
            round_num=1,
        )

        positions = ledger.get_agent_positions("claude")
        assert positions[0].claim == claim

    def test_unicode_in_claim(self, ledger):
        """Should handle unicode characters."""
        claim = "AI 安全性について議論 🤖"
        pos_id = ledger.record_position(
            agent_name="claude",
            claim=claim,
            confidence=0.7,
            debate_id="d1",
            round_num=1,
        )

        positions = ledger.get_agent_positions("claude")
        assert positions[0].claim == claim

    def test_confidence_boundary_values(self, ledger):
        """Should handle boundary confidence values."""
        # Exactly 0.0
        pos1 = ledger.record_position("claude", "c1", 0.0, "d1", 1)
        # Exactly 1.0
        pos2 = ledger.record_position("claude", "c2", 1.0, "d1", 2)

        positions = ledger.get_agent_positions("claude")
        confidences = {p.confidence for p in positions}
        assert 0.0 in confidences
        assert 1.0 in confidences

    def test_high_round_number(self, ledger):
        """Should handle high round numbers."""
        pos_id = ledger.record_position(
            agent_name="claude",
            claim="test",
            confidence=0.5,
            debate_id="d1",
            round_num=999,
        )

        positions = ledger.get_agent_positions("claude")
        assert positions[0].round_num == 999

    def test_resolve_nonexistent_position(self, ledger):
        """Should handle resolving nonexistent position gracefully."""
        # Should not raise
        ledger.resolve_position("nonexistent-id", "correct")
