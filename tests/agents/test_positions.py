"""
Tests for agent position ledger tracking.

Tests cover:
- Position dataclass and factory methods
- CalibrationBucket statistics and properties
- DomainCalibration aggregation
- PositionLedger CRUD operations
- Position resolution and reversal tracking
- Statistics aggregation
- Domain detection
"""

import pytest
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from aragora.agents.positions import (
    Position,
    CalibrationBucket,
    DomainCalibration,
    PositionLedger,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield Path(f.name)


@pytest.fixture
def position_ledger(temp_db):
    """Create a PositionLedger with a temporary database."""
    return PositionLedger(db_path=temp_db)


class TestPositionDataclass:
    """Tests for the Position dataclass."""

    def test_create_basic_position(self):
        """Test creating a basic position with all fields."""
        position = Position(
            id="abc123",
            agent_name="claude",
            claim="The answer is 42",
            confidence=0.85,
            debate_id="debate-001",
            round_num=1,
        )

        assert position.id == "abc123"
        assert position.agent_name == "claude"
        assert position.claim == "The answer is 42"
        assert position.confidence == 0.85
        assert position.debate_id == "debate-001"
        assert position.round_num == 1

    def test_position_defaults(self):
        """Test position default values."""
        position = Position(
            id="test",
            agent_name="agent",
            claim="claim",
            confidence=0.5,
            debate_id="debate",
            round_num=0,
        )

        assert position.outcome == "pending"
        assert position.reversed is False
        assert position.reversal_debate_id is None
        assert position.domain is None
        assert position.resolved_at is None

    def test_create_factory_generates_id(self):
        """Test Position.create generates a unique ID."""
        position = Position.create(
            agent_name="claude",
            claim="Test claim",
            confidence=0.75,
            debate_id="debate-001",
            round_num=1,
        )

        assert len(position.id) == 8
        assert position.agent_name == "claude"
        assert position.claim == "Test claim"

    def test_create_factory_clamps_confidence(self):
        """Test Position.create clamps confidence to [0, 1]."""
        position_high = Position.create(
            agent_name="test",
            claim="claim",
            confidence=1.5,  # Above max
            debate_id="debate",
            round_num=0,
        )
        assert position_high.confidence == 1.0

        position_low = Position.create(
            agent_name="test",
            claim="claim",
            confidence=-0.5,  # Below min
            debate_id="debate",
            round_num=0,
        )
        assert position_low.confidence == 0.0

    def test_create_factory_with_domain(self):
        """Test Position.create with domain."""
        position = Position.create(
            agent_name="claude",
            claim="Security best practice",
            confidence=0.9,
            debate_id="debate-001",
            round_num=1,
            domain="security",
        )

        assert position.domain == "security"

    def test_create_factory_sets_timestamp(self):
        """Test Position.create sets created_at timestamp."""
        before = datetime.now().isoformat()
        position = Position.create(
            agent_name="test",
            claim="claim",
            confidence=0.5,
            debate_id="debate",
            round_num=0,
        )
        after = datetime.now().isoformat()

        assert before <= position.created_at <= after

    def test_from_row_creates_position(self):
        """Test Position.from_row creates position from database row."""
        # Create a mock sqlite row
        mock_row = {
            "id": "row123",
            "agent_name": "claude",
            "claim": "Test claim",
            "confidence": 0.8,
            "debate_id": "debate-001",
            "round_num": 2,
            "outcome": "correct",
            "reversed": 1,
            "reversal_debate_id": "debate-002",
            "domain": "testing",
            "created_at": "2024-01-01T00:00:00",
            "resolved_at": "2024-01-02T00:00:00",
        }

        position = Position.from_row(mock_row)

        assert position.id == "row123"
        assert position.agent_name == "claude"
        assert position.reversed is True
        assert position.outcome == "correct"
        assert position.domain == "testing"


class TestCalibrationBucket:
    """Tests for the CalibrationBucket dataclass."""

    def test_create_bucket(self):
        """Test creating a calibration bucket."""
        bucket = CalibrationBucket(
            bucket_start=0.7,
            bucket_end=0.8,
            predictions=100,
            correct=75,
        )

        assert bucket.bucket_start == 0.7
        assert bucket.bucket_end == 0.8
        assert bucket.predictions == 100
        assert bucket.correct == 75

    def test_bucket_defaults(self):
        """Test calibration bucket default values."""
        bucket = CalibrationBucket(bucket_start=0.5, bucket_end=0.6)

        assert bucket.predictions == 0
        assert bucket.correct == 0
        assert bucket.brier_sum == 0.0

    def test_accuracy_calculation(self):
        """Test accuracy property."""
        bucket = CalibrationBucket(
            bucket_start=0.7,
            bucket_end=0.8,
            predictions=100,
            correct=80,
        )

        assert bucket.accuracy == 0.8

    def test_accuracy_empty_bucket(self):
        """Test accuracy returns 0 for empty bucket."""
        bucket = CalibrationBucket(
            bucket_start=0.7,
            bucket_end=0.8,
            predictions=0,
            correct=0,
        )

        assert bucket.accuracy == 0.0

    def test_expected_accuracy(self):
        """Test expected_accuracy returns bucket midpoint."""
        bucket = CalibrationBucket(
            bucket_start=0.7,
            bucket_end=0.8,
            predictions=10,
            correct=7,
        )

        assert bucket.expected_accuracy == 0.75

    def test_calibration_error_perfect(self):
        """Test calibration error for perfectly calibrated bucket."""
        bucket = CalibrationBucket(
            bucket_start=0.7,
            bucket_end=0.8,
            predictions=100,
            correct=75,  # 75% accuracy matches 75% expected
        )

        assert bucket.calibration_error == 0.0

    def test_calibration_error_overconfident(self):
        """Test calibration error for overconfident bucket."""
        bucket = CalibrationBucket(
            bucket_start=0.8,
            bucket_end=0.9,
            predictions=100,
            correct=50,  # 50% accuracy vs 85% expected
        )

        assert abs(bucket.calibration_error - 0.35) < 0.01

    def test_bucket_key_format(self):
        """Test bucket_key returns correct format."""
        bucket = CalibrationBucket(bucket_start=0.8, bucket_end=0.9)

        assert bucket.bucket_key == "0.8-0.9"


class TestDomainCalibration:
    """Tests for the DomainCalibration dataclass."""

    def test_create_domain_calibration(self):
        """Test creating domain calibration."""
        calibration = DomainCalibration(
            domain="security",
            total_predictions=100,
            total_correct=80,
        )

        assert calibration.domain == "security"
        assert calibration.total_predictions == 100
        assert calibration.total_correct == 80

    def test_domain_calibration_defaults(self):
        """Test domain calibration default values."""
        calibration = DomainCalibration(domain="testing")

        assert calibration.total_predictions == 0
        assert calibration.total_correct == 0
        assert calibration.brier_sum == 0.0
        assert calibration.buckets == {}

    def test_calibration_score_empty(self):
        """Test calibration score returns 0.5 with no predictions."""
        calibration = DomainCalibration(domain="testing")

        assert calibration.calibration_score == 0.5

    def test_calibration_score_with_predictions(self):
        """Test calibration score calculation."""
        calibration = DomainCalibration(
            domain="security",
            total_predictions=100,
            total_correct=80,
            brier_sum=10.0,  # Average Brier = 0.1
        )

        # Score = 1 - (brier_sum / total_predictions) = 1 - 0.1 = 0.9
        assert calibration.calibration_score == 0.9

    def test_accuracy_calculation(self):
        """Test accuracy property."""
        calibration = DomainCalibration(
            domain="security",
            total_predictions=50,
            total_correct=35,
        )

        assert calibration.accuracy == 0.7

    def test_accuracy_empty(self):
        """Test accuracy returns 0 with no predictions."""
        calibration = DomainCalibration(domain="testing")

        assert calibration.accuracy == 0.0


class TestPositionLedgerInit:
    """Tests for PositionLedger initialization."""

    def test_init_with_custom_path(self, temp_db):
        """Test initialization with custom database path."""
        ledger = PositionLedger(db_path=temp_db)

        assert ledger.db_path == temp_db

    def test_init_creates_tables(self, temp_db):
        """Test initialization creates required tables."""
        ledger = PositionLedger(db_path=temp_db)

        # Verify table exists by recording a position
        position_id = ledger.record_position(
            agent_name="test",
            claim="test claim",
            confidence=0.5,
            debate_id="debate",
            round_num=0,
        )

        assert position_id is not None
        assert len(position_id) == 8


class TestPositionLedgerRecording:
    """Tests for position recording operations."""

    def test_record_position(self, position_ledger):
        """Test recording a new position."""
        position_id = position_ledger.record_position(
            agent_name="claude",
            claim="The solution is efficient",
            confidence=0.85,
            debate_id="debate-001",
            round_num=1,
        )

        assert position_id is not None
        assert len(position_id) == 8

    def test_record_position_with_domain(self, position_ledger):
        """Test recording position with domain."""
        position_id = position_ledger.record_position(
            agent_name="claude",
            claim="Security vulnerability detected",
            confidence=0.9,
            debate_id="debate-001",
            round_num=1,
            domain="security",
        )

        positions = position_ledger.get_agent_positions("claude")
        assert len(positions) == 1
        assert positions[0].domain == "security"

    def test_record_multiple_positions(self, position_ledger):
        """Test recording multiple positions."""
        for i in range(5):
            position_ledger.record_position(
                agent_name="claude",
                claim=f"Claim {i}",
                confidence=0.5 + i * 0.1,
                debate_id="debate-001",
                round_num=i,
            )

        positions = position_ledger.get_agent_positions("claude")
        assert len(positions) == 5


class TestPositionLedgerResolution:
    """Tests for position resolution operations."""

    def test_resolve_position_correct(self, position_ledger):
        """Test resolving position as correct."""
        position_id = position_ledger.record_position(
            agent_name="claude",
            claim="Correct claim",
            confidence=0.8,
            debate_id="debate-001",
            round_num=1,
        )

        position_ledger.resolve_position(position_id, "correct")

        positions = position_ledger.get_agent_positions("claude")
        assert positions[0].outcome == "correct"
        assert positions[0].resolved_at is not None

    def test_resolve_position_incorrect(self, position_ledger):
        """Test resolving position as incorrect."""
        position_id = position_ledger.record_position(
            agent_name="claude",
            claim="Incorrect claim",
            confidence=0.8,
            debate_id="debate-001",
            round_num=1,
        )

        position_ledger.resolve_position(position_id, "incorrect")

        positions = position_ledger.get_agent_positions("claude")
        assert positions[0].outcome == "incorrect"

    def test_resolve_position_unresolved(self, position_ledger):
        """Test resolving position as unresolved."""
        position_id = position_ledger.record_position(
            agent_name="claude",
            claim="Ambiguous claim",
            confidence=0.5,
            debate_id="debate-001",
            round_num=1,
        )

        position_ledger.resolve_position(position_id, "unresolved")

        positions = position_ledger.get_agent_positions("claude")
        assert positions[0].outcome == "unresolved"


class TestPositionLedgerReversal:
    """Tests for position reversal tracking."""

    def test_record_reversal(self, position_ledger):
        """Test recording a position reversal."""
        position_id = position_ledger.record_position(
            agent_name="claude",
            claim="Initial claim",
            confidence=0.7,
            debate_id="debate-001",
            round_num=1,
        )

        position_ledger.record_reversal(
            agent_name="claude",
            original_position_id=position_id,
            new_debate_id="debate-002",
        )

        positions = position_ledger.get_agent_positions("claude")
        assert positions[0].reversed is True
        assert positions[0].reversal_debate_id == "debate-002"

    def test_reversal_only_affects_matching_agent(self, position_ledger):
        """Test reversal only affects position with matching agent."""
        position_id = position_ledger.record_position(
            agent_name="claude",
            claim="Claude's claim",
            confidence=0.7,
            debate_id="debate-001",
            round_num=1,
        )

        # Try to reverse with wrong agent name
        position_ledger.record_reversal(
            agent_name="gpt",  # Wrong agent
            original_position_id=position_id,
            new_debate_id="debate-002",
        )

        # Position should not be reversed
        positions = position_ledger.get_agent_positions("claude")
        assert positions[0].reversed is False


class TestPositionLedgerRetrieval:
    """Tests for position retrieval operations."""

    def test_get_agent_positions(self, position_ledger):
        """Test getting positions for an agent."""
        for i in range(3):
            position_ledger.record_position(
                agent_name="claude",
                claim=f"Claim {i}",
                confidence=0.7,
                debate_id="debate-001",
                round_num=i,
            )

        positions = position_ledger.get_agent_positions("claude")

        assert len(positions) == 3
        # Should be ordered by created_at DESC
        assert positions[0].round_num == 2

    def test_get_agent_positions_with_limit(self, position_ledger):
        """Test getting positions with limit."""
        for i in range(10):
            position_ledger.record_position(
                agent_name="claude",
                claim=f"Claim {i}",
                confidence=0.7,
                debate_id="debate-001",
                round_num=i,
            )

        positions = position_ledger.get_agent_positions("claude", limit=5)

        assert len(positions) == 5

    def test_get_agent_positions_with_outcome_filter(self, position_ledger):
        """Test getting positions filtered by outcome."""
        # Create positions with different outcomes
        for i, outcome in enumerate(["correct", "incorrect", "correct", "pending"]):
            position_id = position_ledger.record_position(
                agent_name="claude",
                claim=f"Claim {i}",
                confidence=0.7,
                debate_id="debate-001",
                round_num=i,
            )
            if outcome != "pending":
                position_ledger.resolve_position(position_id, outcome)

        correct_positions = position_ledger.get_agent_positions("claude", outcome_filter="correct")

        assert len(correct_positions) == 2
        assert all(p.outcome == "correct" for p in correct_positions)

    def test_get_positions_for_debate(self, position_ledger):
        """Test getting all positions from a debate."""
        # Add positions from multiple agents
        position_ledger.record_position(
            agent_name="claude",
            claim="Claude's claim",
            confidence=0.8,
            debate_id="debate-001",
            round_num=1,
        )
        position_ledger.record_position(
            agent_name="gpt",
            claim="GPT's claim",
            confidence=0.7,
            debate_id="debate-001",
            round_num=1,
        )
        position_ledger.record_position(
            agent_name="claude",
            claim="Different debate",
            confidence=0.6,
            debate_id="debate-002",
            round_num=1,
        )

        debate_positions = position_ledger.get_positions_for_debate("debate-001")

        assert len(debate_positions) == 2
        agents = [p.agent_name for p in debate_positions]
        assert "claude" in agents
        assert "gpt" in agents

    def test_get_positions_empty_result(self, position_ledger):
        """Test getting positions for non-existent agent."""
        positions = position_ledger.get_agent_positions("nonexistent")

        assert positions == []


class TestPositionLedgerStatistics:
    """Tests for position statistics."""

    def test_get_position_stats_empty(self, position_ledger):
        """Test statistics for agent with no positions."""
        stats = position_ledger.get_position_stats("nonexistent")

        assert stats["total"] == 0
        assert stats["correct"] == 0
        assert stats["incorrect"] == 0

    def test_get_position_stats_basic(self, position_ledger):
        """Test basic position statistics."""
        # Create and resolve positions
        for i in range(10):
            position_id = position_ledger.record_position(
                agent_name="claude",
                claim=f"Claim {i}",
                confidence=0.7,
                debate_id="debate-001",
                round_num=i,
            )
            if i < 6:
                position_ledger.resolve_position(position_id, "correct")
            elif i < 8:
                position_ledger.resolve_position(position_id, "incorrect")
            # Leave 2 as pending

        stats = position_ledger.get_position_stats("claude")

        assert stats["total"] == 10
        assert stats["correct"] == 6
        assert stats["incorrect"] == 2
        assert stats["pending"] == 2

    def test_get_position_stats_with_reversals(self, position_ledger):
        """Test statistics include reversal count."""
        position_id = position_ledger.record_position(
            agent_name="claude",
            claim="Initial claim",
            confidence=0.7,
            debate_id="debate-001",
            round_num=1,
        )
        position_ledger.record_reversal("claude", position_id, "debate-002")

        position_ledger.record_position(
            agent_name="claude",
            claim="Second claim",
            confidence=0.8,
            debate_id="debate-002",
            round_num=1,
        )

        stats = position_ledger.get_position_stats("claude")

        assert stats["reversals"] == 1

    def test_get_position_stats_avg_confidence(self, position_ledger):
        """Test average confidence by outcome."""
        # Create correct positions with high confidence
        for i in range(3):
            position_id = position_ledger.record_position(
                agent_name="claude",
                claim=f"Correct claim {i}",
                confidence=0.9,
                debate_id="debate-001",
                round_num=i,
            )
            position_ledger.resolve_position(position_id, "correct")

        # Create incorrect positions with low confidence
        for i in range(3, 6):
            position_id = position_ledger.record_position(
                agent_name="claude",
                claim=f"Incorrect claim {i}",
                confidence=0.3,
                debate_id="debate-001",
                round_num=i,
            )
            position_ledger.resolve_position(position_id, "incorrect")

        stats = position_ledger.get_position_stats("claude")

        assert abs(stats["avg_confidence_when_correct"] - 0.9) < 0.01
        assert abs(stats["avg_confidence_when_incorrect"] - 0.3) < 0.01


class TestDomainDetection:
    """Tests for domain detection from content."""

    def test_detect_security_domain(self, position_ledger):
        """Test detecting security domain."""
        content = "There is a SQL injection vulnerability in the auth module"
        domain = position_ledger.detect_domain(content)

        assert domain == "security"

    def test_detect_performance_domain(self, position_ledger):
        """Test detecting performance domain."""
        content = "The cache optimization will improve latency by 50%"
        domain = position_ledger.detect_domain(content)

        assert domain == "performance"

    def test_detect_testing_domain(self, position_ledger):
        """Test detecting testing domain."""
        content = "We need better test coverage with proper fixtures and mocks"
        domain = position_ledger.detect_domain(content)

        assert domain == "testing"

    def test_detect_architecture_domain(self, position_ledger):
        """Test detecting architecture domain."""
        content = "The design pattern improves modularity and reduces coupling"
        domain = position_ledger.detect_domain(content)

        assert domain == "architecture"

    def test_detect_error_handling_domain(self, position_ledger):
        """Test detecting error handling domain."""
        content = "We should add a retry mechanism with graceful fallback"
        domain = position_ledger.detect_domain(content)

        assert domain == "error_handling"

    def test_detect_concurrency_domain(self, position_ledger):
        """Test detecting concurrency domain."""
        content = "This could cause a race condition or deadlock"
        domain = position_ledger.detect_domain(content)

        assert domain == "concurrency"

    def test_detect_api_design_domain(self, position_ledger):
        """Test detecting API design domain."""
        content = "The REST endpoint interface should follow standard conventions"
        domain = position_ledger.detect_domain(content)

        assert domain == "api_design"

    def test_detect_database_domain(self, position_ledger):
        """Test detecting database domain."""
        content = "Adding an index on this column will speed up the query"
        domain = position_ledger.detect_domain(content)

        assert domain == "database"

    def test_detect_frontend_domain(self, position_ledger):
        """Test detecting frontend domain."""
        content = "The UI component needs responsive CSS styling"
        domain = position_ledger.detect_domain(content)

        assert domain == "frontend"

    def test_detect_devops_domain(self, position_ledger):
        """Test detecting devops domain."""
        content = "The Docker container deploys to kubernetes in CI/CD"
        domain = position_ledger.detect_domain(content)

        assert domain == "devops"

    def test_detect_no_domain(self, position_ledger):
        """Test detecting no domain for generic content."""
        content = "This is a general statement about nothing specific"
        domain = position_ledger.detect_domain(content)

        assert domain is None

    def test_detect_domain_case_insensitive(self, position_ledger):
        """Test domain detection is case insensitive."""
        content = "SECURITY VULNERABILITY in AUTH system"
        domain = position_ledger.detect_domain(content)

        assert domain == "security"


class TestPositionLedgerEdgeCases:
    """Tests for edge cases in position ledger."""

    def test_empty_claim(self, position_ledger):
        """Test recording position with empty claim."""
        position_id = position_ledger.record_position(
            agent_name="claude",
            claim="",
            confidence=0.5,
            debate_id="debate-001",
            round_num=0,
        )

        assert position_id is not None

    def test_special_characters_in_claim(self, position_ledger):
        """Test recording position with special characters."""
        claim = "Test with 'quotes' and \"double quotes\" and emoji \U0001f600"
        position_id = position_ledger.record_position(
            agent_name="claude",
            claim=claim,
            confidence=0.5,
            debate_id="debate-001",
            round_num=0,
        )

        positions = position_ledger.get_agent_positions("claude")
        assert positions[0].claim == claim

    def test_unicode_agent_name(self, position_ledger):
        """Test recording position with unicode agent name."""
        position_id = position_ledger.record_position(
            agent_name="claude_\u00e9ric",
            claim="Test claim",
            confidence=0.5,
            debate_id="debate-001",
            round_num=0,
        )

        positions = position_ledger.get_agent_positions("claude_\u00e9ric")
        assert len(positions) == 1

    def test_very_long_claim(self, position_ledger):
        """Test recording position with very long claim."""
        long_claim = "A" * 10000
        position_id = position_ledger.record_position(
            agent_name="claude",
            claim=long_claim,
            confidence=0.5,
            debate_id="debate-001",
            round_num=0,
        )

        positions = position_ledger.get_agent_positions("claude")
        assert positions[0].claim == long_claim

    def test_confidence_boundary_values(self, position_ledger):
        """Test recording positions with boundary confidence values."""
        position_ledger.record_position(
            agent_name="claude",
            claim="Zero confidence",
            confidence=0.0,
            debate_id="debate-001",
            round_num=0,
        )
        position_ledger.record_position(
            agent_name="claude",
            claim="Full confidence",
            confidence=1.0,
            debate_id="debate-001",
            round_num=1,
        )

        positions = position_ledger.get_agent_positions("claude")
        confidences = [p.confidence for p in positions]

        assert 0.0 in confidences
        assert 1.0 in confidences


class TestPositionLedgerConcurrency:
    """Tests for concurrent access patterns."""

    def test_multiple_ledger_instances(self, temp_db):
        """Test multiple ledger instances on same database."""
        ledger1 = PositionLedger(db_path=temp_db)
        ledger2 = PositionLedger(db_path=temp_db)

        # Record with first instance
        ledger1.record_position(
            agent_name="claude",
            claim="First ledger claim",
            confidence=0.7,
            debate_id="debate-001",
            round_num=0,
        )

        # Read with second instance
        positions = ledger2.get_agent_positions("claude")

        assert len(positions) == 1
        assert positions[0].claim == "First ledger claim"

    def test_reopening_database(self, temp_db):
        """Test reopening database preserves data."""
        # Create and populate
        ledger1 = PositionLedger(db_path=temp_db)
        ledger1.record_position(
            agent_name="claude",
            claim="Persistent claim",
            confidence=0.8,
            debate_id="debate-001",
            round_num=0,
        )

        # Reopen
        ledger2 = PositionLedger(db_path=temp_db)
        positions = ledger2.get_agent_positions("claude")

        assert len(positions) == 1
        assert positions[0].claim == "Persistent claim"
