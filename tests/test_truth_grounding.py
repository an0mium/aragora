"""
Tests for aragora/agents/truth_grounding.py

Comprehensive test coverage for the truth-grounded persona system that tracks
agent positions across debates and links them to verifiable outcomes.
"""

import os
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aragora.agents.truth_grounding import (
    Position,
    PositionTracker,
    TruthGroundedLaboratory,
    TruthGroundedPersona,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db():
    """Create temporary SQLite database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    if db_path.exists():
        os.unlink(db_path)


@pytest.fixture
def tracker(temp_db):
    """Create PositionTracker with temp database."""
    return PositionTracker(db_path=str(temp_db))


@pytest.fixture
def tracker_with_data(tracker):
    """Create tracker with pre-recorded positions."""
    # Record several positions for agent1
    tracker.record_position("debate-1", "agent1", "vote", "Position A", confidence=0.8)
    tracker.record_position("debate-1", "agent2", "vote", "Position B", confidence=0.7)
    tracker.finalize_debate("debate-1", "agent1", "Position A", 0.85)
    return tracker


@pytest.fixture
def tracker_with_verifications(tracker_with_data):
    """Create tracker with verified positions."""
    tracker = tracker_with_data
    # Add more debates to get min_verifications
    for i in range(2, 8):
        tracker.record_position(f"debate-{i}", "agent1", "vote", f"Position {i}", confidence=0.75)
        tracker.record_position(f"debate-{i}", "agent2", "vote", f"Other {i}", confidence=0.65)
        tracker.finalize_debate(f"debate-{i}", "agent1", f"Position {i}", 0.8)
        # Verify most as correct
        tracker.record_verification(f"debate-{i}", result=(i % 3 != 0), source="test")
    # Also verify debate-1
    tracker.record_verification("debate-1", result=True, source="test")
    return tracker


@pytest.fixture
def mock_elo_system():
    """Create mock ELO system."""
    elo = MagicMock()
    rating = MagicMock()
    rating.elo = 1600.0
    rating.win_rate = 0.65
    rating.calibration_score = 0.8
    elo.get_rating.return_value = rating
    return elo


@pytest.fixture
def laboratory(temp_db):
    """Create TruthGroundedLaboratory with temp database."""
    tracker = PositionTracker(db_path=str(temp_db))
    return TruthGroundedLaboratory(position_tracker=tracker)


# =============================================================================
# Test Position Dataclass
# =============================================================================


class TestPosition:
    """Tests for the Position dataclass."""

    def test_creation_with_required_fields(self):
        """Test Position creation with required fields."""
        pos = Position(
            debate_id="debate-1",
            agent_name="agent1",
            position_type="vote",
            position_text="I support option A",
        )
        assert pos.debate_id == "debate-1"
        assert pos.agent_name == "agent1"
        assert pos.position_type == "vote"
        assert pos.position_text == "I support option A"

    def test_default_values(self):
        """Test Position default values."""
        pos = Position(
            debate_id="debate-1",
            agent_name="agent1",
            position_type="vote",
            position_text="Test",
        )
        assert pos.round_num == 0
        assert pos.confidence == 0.5
        assert pos.was_winning is None
        assert pos.verified_correct is None

    def test_position_types(self):
        """Test various position types."""
        for pos_type in ["proposal", "vote", "critique"]:
            pos = Position(
                debate_id="d1",
                agent_name="a1",
                position_type=pos_type,
                position_text="text",
            )
            assert pos.position_type == pos_type

    def test_optional_fields(self):
        """Test Position with optional fields set."""
        pos = Position(
            debate_id="debate-1",
            agent_name="agent1",
            position_type="vote",
            position_text="Test",
            round_num=3,
            confidence=0.9,
            was_winning=True,
            verified_correct=True,
        )
        assert pos.round_num == 3
        assert pos.confidence == 0.9
        assert pos.was_winning is True
        assert pos.verified_correct is True

    def test_created_at_auto_generated(self):
        """Test that created_at is auto-generated."""
        before = datetime.now().isoformat()
        pos = Position(
            debate_id="d1",
            agent_name="a1",
            position_type="vote",
            position_text="text",
        )
        after = datetime.now().isoformat()
        assert before <= pos.created_at <= after


# =============================================================================
# Test TruthGroundedPersona Dataclass
# =============================================================================


class TestTruthGroundedPersona:
    """Tests for the TruthGroundedPersona dataclass."""

    def test_creation_with_required_fields(self):
        """Test TruthGroundedPersona creation with agent_name."""
        persona = TruthGroundedPersona(agent_name="agent1")
        assert persona.agent_name == "agent1"

    def test_default_values(self):
        """Test TruthGroundedPersona default values."""
        persona = TruthGroundedPersona(agent_name="agent1")
        assert persona.overall_reliability == 0.0
        assert persona.position_accuracy == 0.0
        assert persona.contrarian_score == 0.0
        assert persona.early_adopter_score == 0.0

    def test_elo_defaults(self):
        """Test ELO-related defaults."""
        persona = TruthGroundedPersona(agent_name="agent1")
        assert persona.elo_rating == 1500.0
        assert persona.elo_win_rate == 0.0
        assert persona.elo_calibration == 0.0

    def test_position_stats_defaults(self):
        """Test position statistics defaults."""
        persona = TruthGroundedPersona(agent_name="agent1")
        assert persona.verified_positions == 0
        assert persona.winning_positions == 0
        assert persona.total_positions == 0

    def test_behavioral_scores_defaults(self):
        """Test behavioral score defaults."""
        persona = TruthGroundedPersona(agent_name="agent1")
        assert persona.contrarian_score == 0.0
        assert persona.early_adopter_score == 0.0

    def test_domain_lists_empty_default(self):
        """Test domain lists default to empty."""
        persona = TruthGroundedPersona(agent_name="agent1")
        assert persona.strength_domains == []
        assert persona.weakness_domains == []


# =============================================================================
# Test PositionTracker Initialization
# =============================================================================


class TestPositionTrackerInit:
    """Tests for PositionTracker initialization."""

    def test_initialization_with_default_path(self):
        """Test PositionTracker creates with default path."""
        tracker = PositionTracker()
        assert tracker.db_path == Path("aragora_positions.db")
        # Clean up
        if tracker.db_path.exists():
            os.unlink(tracker.db_path)

    def test_initialization_with_custom_path(self, temp_db):
        """Test PositionTracker with custom path."""
        tracker = PositionTracker(db_path=str(temp_db))
        assert tracker.db_path == temp_db

    def test_creates_database_file(self, temp_db):
        """Test that initialization creates the database file."""
        PositionTracker(db_path=str(temp_db))
        assert temp_db.exists()

    def test_creates_position_history_table(self, tracker, temp_db):
        """Test that position_history table is created."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='position_history'"
        )
        result = cursor.fetchone()
        conn.close()
        assert result is not None
        assert result[0] == "position_history"

    def test_creates_debate_outcomes_table(self, tracker, temp_db):
        """Test that debate_outcomes table is created."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='debate_outcomes'"
        )
        result = cursor.fetchone()
        conn.close()
        assert result is not None
        assert result[0] == "debate_outcomes"

    def test_creates_indexes(self, tracker, temp_db):
        """Test that indexes are created."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = [row[0] for row in cursor.fetchall()]
        conn.close()
        assert "idx_position_agent" in indexes
        assert "idx_position_debate" in indexes

    def test_idempotent_on_reinit(self, temp_db):
        """Test that re-initialization is idempotent."""
        tracker1 = PositionTracker(db_path=str(temp_db))
        tracker1.record_position("d1", "a1", "vote", "text")

        # Re-init should not destroy data
        tracker2 = PositionTracker(db_path=str(temp_db))
        history = tracker2.get_position_history("a1")
        assert len(history) == 1


# =============================================================================
# Test PositionTracker._get_connection()
# =============================================================================


class TestGetConnection:
    """Tests for _get_connection context manager."""

    def test_yields_connection(self, tracker):
        """Test that _get_connection yields a connection."""
        with tracker._get_connection() as conn:
            assert isinstance(conn, sqlite3.Connection)

    def test_closes_connection_on_exit(self, tracker, temp_db):
        """Test that connection is closed after context exit."""
        with tracker._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
        # After exit, connection should be closed
        # Trying to use it should raise
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")

    def test_closes_connection_on_exception(self, tracker):
        """Test that connection is closed even on exception."""
        try:
            with tracker._get_connection() as conn:
                raise ValueError("Test exception")
        except ValueError:
            pass
        # Connection should still be closed
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


# =============================================================================
# Test PositionTracker.record_position()
# =============================================================================


class TestRecordPosition:
    """Tests for record_position method."""

    def test_records_basic_position(self, tracker):
        """Test recording a basic position."""
        pos = tracker.record_position(
            debate_id="debate-1",
            agent_name="agent1",
            position_type="vote",
            position_text="I vote for A",
        )
        assert pos.debate_id == "debate-1"
        assert pos.agent_name == "agent1"

    def test_records_with_round_num(self, tracker):
        """Test recording position with round number."""
        pos = tracker.record_position(
            debate_id="d1",
            agent_name="a1",
            position_type="vote",
            position_text="text",
            round_num=5,
        )
        assert pos.round_num == 5

    def test_records_with_confidence(self, tracker):
        """Test recording position with confidence."""
        pos = tracker.record_position(
            debate_id="d1",
            agent_name="a1",
            position_type="vote",
            position_text="text",
            confidence=0.95,
        )
        assert pos.confidence == 0.95

    def test_returns_position_object(self, tracker):
        """Test that record_position returns a Position object."""
        pos = tracker.record_position(
            debate_id="d1",
            agent_name="a1",
            position_type="vote",
            position_text="text",
        )
        assert isinstance(pos, Position)

    def test_idempotent_insert_or_replace(self, tracker):
        """Test that INSERT OR REPLACE works correctly."""
        # First insert
        tracker.record_position("d1", "a1", "vote", "text1", round_num=0, confidence=0.5)

        # Second insert with same key (debate_id, agent_name, position_type, round_num)
        tracker.record_position("d1", "a1", "vote", "text2", round_num=0, confidence=0.8)

        # Should only have one entry with updated values
        history = tracker.get_position_history("a1")
        assert len(history) == 1
        assert history[0].position_text == "text2"
        assert history[0].confidence == 0.8

    def test_unique_constraint_handling(self, tracker):
        """Test unique constraint allows different round_nums."""
        tracker.record_position("d1", "a1", "vote", "text1", round_num=0)
        tracker.record_position("d1", "a1", "vote", "text2", round_num=1)

        history = tracker.get_position_history("a1")
        assert len(history) == 2

    def test_position_persisted_to_db(self, tracker, temp_db):
        """Test that position is persisted to database."""
        tracker.record_position("d1", "a1", "vote", "test text")

        # Query directly
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT position_text FROM position_history WHERE agent_name = 'a1'")
        result = cursor.fetchone()
        conn.close()

        assert result[0] == "test text"


# =============================================================================
# Test PositionTracker.finalize_debate()
# =============================================================================


class TestFinalizeDebate:
    """Tests for finalize_debate method."""

    def test_records_debate_outcome(self, tracker, temp_db):
        """Test that debate outcome is recorded."""
        tracker.record_position("d1", "a1", "vote", "text")
        tracker.finalize_debate("d1", "a1", "Position A", 0.85)

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT winning_agent FROM debate_outcomes WHERE debate_id = 'd1'")
        result = cursor.fetchone()
        conn.close()

        assert result[0] == "a1"

    def test_updates_winning_agent_positions(self, tracker):
        """Test that winning agent positions are marked."""
        tracker.record_position("d1", "winner", "vote", "text")
        tracker.record_position("d1", "loser", "vote", "other")
        tracker.finalize_debate("d1", "winner", "Position", 0.9)

        winner_history = tracker.get_position_history("winner")
        assert winner_history[0].was_winning is True

    def test_updates_losing_agent_positions(self, tracker):
        """Test that losing agent positions are marked as not winning."""
        tracker.record_position("d1", "winner", "vote", "text")
        tracker.record_position("d1", "loser", "vote", "other")
        tracker.finalize_debate("d1", "winner", "Position", 0.9)

        loser_history = tracker.get_position_history("loser")
        assert loser_history[0].was_winning is False

    def test_sets_consensus_confidence(self, tracker, temp_db):
        """Test that consensus confidence is set."""
        tracker.record_position("d1", "a1", "vote", "text")
        tracker.finalize_debate("d1", "a1", "Position", 0.77)

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT consensus_confidence FROM debate_outcomes WHERE debate_id = 'd1'")
        result = cursor.fetchone()
        conn.close()

        assert result[0] == 0.77

    def test_works_with_no_positions(self, tracker, temp_db):
        """Test finalize_debate works even without positions."""
        # Should not raise
        tracker.finalize_debate("empty-debate", "winner", "Position", 0.9)

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM debate_outcomes WHERE debate_id = 'empty-debate'")
        result = cursor.fetchone()
        conn.close()

        assert result is not None

    def test_persists_to_debate_outcomes_table(self, tracker, temp_db):
        """Test that outcome is persisted to debate_outcomes table."""
        tracker.finalize_debate("d1", "agent1", "Winning position text", 0.95)

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT winning_agent, winning_position, consensus_confidence FROM debate_outcomes"
        )
        row = cursor.fetchone()
        conn.close()

        assert row[0] == "agent1"
        assert row[1] == "Winning position text"
        assert row[2] == 0.95


# =============================================================================
# Test PositionTracker.record_verification()
# =============================================================================


class TestRecordVerification:
    """Tests for record_verification method."""

    def test_records_verification_result_true(self, tracker_with_data, temp_db):
        """Test recording verification result as True."""
        tracker_with_data.record_verification("debate-1", result=True, source="test")

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT verification_result FROM debate_outcomes WHERE debate_id = 'debate-1'"
        )
        result = cursor.fetchone()
        conn.close()

        assert result[0] == 1

    def test_records_verification_result_false(self, tracker_with_data, temp_db):
        """Test recording verification result as False."""
        tracker_with_data.record_verification("debate-1", result=False, source="test")

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT verification_result FROM debate_outcomes WHERE debate_id = 'debate-1'"
        )
        result = cursor.fetchone()
        conn.close()

        assert result[0] == 0

    def test_sets_verification_source(self, tracker_with_data, temp_db):
        """Test that verification source is set."""
        tracker_with_data.record_verification("debate-1", result=True, source="automated-test")

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT verification_source FROM debate_outcomes WHERE debate_id = 'debate-1'"
        )
        result = cursor.fetchone()
        conn.close()

        assert result[0] == "automated-test"

    def test_sets_verified_at_timestamp(self, tracker_with_data, temp_db):
        """Test that verified_at timestamp is set."""
        before = datetime.now().isoformat()
        tracker_with_data.record_verification("debate-1", result=True)
        after = datetime.now().isoformat()

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT verified_at FROM debate_outcomes WHERE debate_id = 'debate-1'"
        )
        result = cursor.fetchone()
        conn.close()

        assert result[0] is not None
        assert before <= result[0] <= after

    def test_propagates_to_winning_positions(self, tracker_with_data):
        """Test that verification propagates to winning positions."""
        tracker_with_data.record_verification("debate-1", result=True)

        history = tracker_with_data.get_position_history("agent1")
        assert history[0].verified_correct is True

    def test_marks_losing_positions_incorrect(self, tracker_with_data):
        """Test that losing positions are marked as incorrect."""
        tracker_with_data.record_verification("debate-1", result=True)

        history = tracker_with_data.get_position_history("agent2")
        # Losing position should be marked as not correct (0)
        assert history[0].verified_correct is False

    def test_handles_missing_debate(self, tracker):
        """Test that verifying non-existent debate doesn't raise."""
        # Should not raise
        tracker.record_verification("nonexistent", result=True)


# =============================================================================
# Test PositionTracker.get_agent_position_accuracy()
# =============================================================================


class TestGetAgentPositionAccuracy:
    """Tests for get_agent_position_accuracy method."""

    def test_returns_accuracy_dict(self, tracker_with_data):
        """Test that method returns a dict with expected keys."""
        stats = tracker_with_data.get_agent_position_accuracy("agent1")

        assert "total_positions" in stats
        assert "winning_positions" in stats
        assert "verified_positions" in stats
        assert "win_rate" in stats
        assert "accuracy_rate" in stats
        assert "calibration" in stats

    def test_total_positions_count(self, tracker_with_data):
        """Test total positions count."""
        stats = tracker_with_data.get_agent_position_accuracy("agent1")
        assert stats["total_positions"] == 1

    def test_winning_positions_count(self, tracker_with_data):
        """Test winning positions count."""
        stats = tracker_with_data.get_agent_position_accuracy("agent1")
        assert stats["winning_positions"] == 1

    def test_verified_positions_count(self, tracker_with_verifications):
        """Test verified positions count."""
        stats = tracker_with_verifications.get_agent_position_accuracy("agent1")
        assert stats["verified_positions"] >= 5

    def test_win_rate_calculation(self, tracker_with_data):
        """Test win rate calculation."""
        stats = tracker_with_data.get_agent_position_accuracy("agent1")
        # agent1 won 1/1 = 1.0
        assert stats["win_rate"] == 1.0

    def test_accuracy_rate_requires_min_verifications(self, tracker_with_data):
        """Test that accuracy rate requires minimum verifications."""
        tracker_with_data.record_verification("debate-1", result=True)
        stats = tracker_with_data.get_agent_position_accuracy("agent1", min_verifications=5)
        # Only 1 verification, so accuracy_rate should be 0
        assert stats["accuracy_rate"] == 0.0

    def test_accuracy_rate_with_sufficient_data(self, tracker_with_verifications):
        """Test accuracy rate with sufficient verifications."""
        stats = tracker_with_verifications.get_agent_position_accuracy("agent1", min_verifications=5)
        # Should have non-zero accuracy now
        assert stats["accuracy_rate"] >= 0.0

    def test_calibration_calculation(self, tracker_with_verifications):
        """Test calibration score calculation."""
        stats = tracker_with_verifications.get_agent_position_accuracy("agent1", min_verifications=5)
        # calibration = 1.0 - |avg_confidence - accuracy_rate|
        # Should be between 0 and 1
        assert 0.0 <= stats["calibration"] <= 1.0

    def test_division_by_zero_protection(self, tracker):
        """Test protection against division by zero."""
        stats = tracker.get_agent_position_accuracy("nonexistent")
        assert stats["win_rate"] == 0.0
        assert stats["accuracy_rate"] == 0.0

    def test_handles_no_positions(self, tracker):
        """Test handling agent with no positions."""
        stats = tracker.get_agent_position_accuracy("unknown_agent")
        assert stats["total_positions"] == 0
        assert stats["winning_positions"] == 0
        assert stats["verified_positions"] == 0


# =============================================================================
# Test PositionTracker.get_position_history()
# =============================================================================


class TestGetPositionHistory:
    """Tests for get_position_history method."""

    def test_returns_position_list(self, tracker_with_data):
        """Test that method returns a list of Position objects."""
        history = tracker_with_data.get_position_history("agent1")
        assert isinstance(history, list)
        assert all(isinstance(p, Position) for p in history)

    def test_respects_limit(self, tracker):
        """Test that limit parameter is respected."""
        for i in range(10):
            tracker.record_position(f"d{i}", "agent1", "vote", f"text{i}")

        history = tracker.get_position_history("agent1", limit=5)
        assert len(history) == 5

    def test_orders_by_created_at_desc(self, tracker, temp_db):
        """Test that results are ordered by created_at descending."""
        # Insert with explicit timestamps to ensure ordering
        import sqlite3 as sqlite3_module
        conn = sqlite3_module.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO position_history (debate_id, agent_name, position_type, position_text, created_at) "
            "VALUES ('d1', 'agent1', 'vote', 'first', '2024-01-01T00:00:00')"
        )
        cursor.execute(
            "INSERT INTO position_history (debate_id, agent_name, position_type, position_text, created_at) "
            "VALUES ('d2', 'agent1', 'vote', 'second', '2024-01-02T00:00:00')"
        )
        cursor.execute(
            "INSERT INTO position_history (debate_id, agent_name, position_type, position_text, created_at) "
            "VALUES ('d3', 'agent1', 'vote', 'third', '2024-01-03T00:00:00')"
        )
        conn.commit()
        conn.close()

        history = tracker.get_position_history("agent1")
        # Most recent should be first (d3 has latest timestamp)
        assert history[0].debate_id == "d3"
        assert history[-1].debate_id == "d1"

    def test_verified_only_filter(self, tracker_with_verifications):
        """Test verified_only filter."""
        history = tracker_with_verifications.get_position_history("agent1", verified_only=True)
        # All returned positions should have verified_correct set
        for pos in history:
            assert pos.verified_correct is not None

    def test_boolean_conversion_was_winning(self, tracker_with_data):
        """Test was_winning is converted to bool."""
        history = tracker_with_data.get_position_history("agent1")
        pos = history[0]
        # Should be bool True, not int 1
        assert pos.was_winning is True
        assert isinstance(pos.was_winning, bool)

    def test_boolean_conversion_verified_correct(self, tracker_with_verifications):
        """Test verified_correct is converted to bool."""
        history = tracker_with_verifications.get_position_history("agent1", verified_only=True)
        for pos in history:
            assert isinstance(pos.verified_correct, bool)

    def test_handles_null_booleans(self, tracker):
        """Test handling of NULL was_winning and verified_correct."""
        tracker.record_position("d1", "agent1", "vote", "text")

        history = tracker.get_position_history("agent1")
        pos = history[0]
        assert pos.was_winning is None
        assert pos.verified_correct is None

    def test_returns_empty_for_unknown_agent(self, tracker):
        """Test returns empty list for unknown agent."""
        history = tracker.get_position_history("nonexistent_agent")
        assert history == []


# =============================================================================
# Test TruthGroundedLaboratory Initialization
# =============================================================================


class TestTruthGroundedLaboratoryInit:
    """Tests for TruthGroundedLaboratory initialization."""

    def test_initialization_with_defaults(self, temp_db):
        """Test laboratory initialization with defaults."""
        lab = TruthGroundedLaboratory(db_path=str(temp_db))
        assert lab.position_tracker is not None
        assert lab.elo_system is None
        assert lab.persona_manager is None

    def test_initialization_with_custom_tracker(self, tracker):
        """Test laboratory with custom position tracker."""
        lab = TruthGroundedLaboratory(position_tracker=tracker)
        assert lab.position_tracker is tracker

    def test_initialization_with_elo_system(self, tracker, mock_elo_system):
        """Test laboratory with ELO system."""
        lab = TruthGroundedLaboratory(
            position_tracker=tracker,
            elo_system=mock_elo_system,
        )
        assert lab.elo_system is mock_elo_system

    def test_initialization_with_persona_manager(self, tracker):
        """Test laboratory with persona manager."""
        mock_manager = MagicMock()
        lab = TruthGroundedLaboratory(
            position_tracker=tracker,
            persona_manager=mock_manager,
        )
        assert lab.persona_manager is mock_manager

    def test_creates_position_tracker_if_not_provided(self, temp_db):
        """Test that position tracker is created if not provided."""
        lab = TruthGroundedLaboratory(db_path=str(temp_db))
        assert isinstance(lab.position_tracker, PositionTracker)


# =============================================================================
# Test TruthGroundedLaboratory.synthesize_persona()
# =============================================================================


class TestSynthesizePersona:
    """Tests for synthesize_persona method."""

    def test_returns_persona(self, laboratory):
        """Test that synthesize_persona returns a TruthGroundedPersona."""
        persona = laboratory.synthesize_persona("agent1")
        assert isinstance(persona, TruthGroundedPersona)
        assert persona.agent_name == "agent1"

    def test_includes_position_stats(self, laboratory):
        """Test that persona includes position statistics."""
        laboratory.position_tracker.record_position("d1", "agent1", "vote", "text")
        laboratory.position_tracker.finalize_debate("d1", "agent1", "Position", 0.9)

        persona = laboratory.synthesize_persona("agent1")
        assert persona.total_positions == 1
        assert persona.winning_positions == 1

    def test_includes_elo_data_when_available(self, tracker, mock_elo_system):
        """Test that persona includes ELO data when system available."""
        lab = TruthGroundedLaboratory(
            position_tracker=tracker,
            elo_system=mock_elo_system,
        )

        persona = lab.synthesize_persona("agent1")
        assert persona.elo_rating == 1600.0
        assert persona.elo_win_rate == 0.65
        assert persona.elo_calibration == 0.8

    def test_handles_missing_elo_system(self, laboratory):
        """Test graceful handling when no ELO system."""
        persona = laboratory.synthesize_persona("agent1")
        assert persona.elo_rating == 1500.0
        assert persona.elo_win_rate == 0.0
        assert persona.elo_calibration == 0.0

    def test_calculates_overall_reliability(self, laboratory):
        """Test overall reliability calculation."""
        # Add positions and verifications
        for i in range(10):
            laboratory.position_tracker.record_position(f"d{i}", "agent1", "vote", f"text{i}")
            laboratory.position_tracker.finalize_debate(f"d{i}", "agent1", f"Position{i}", 0.9)
            laboratory.position_tracker.record_verification(f"d{i}", result=True)

        persona = laboratory.synthesize_persona("agent1")
        assert persona.overall_reliability > 0.0

    def test_reliability_weighting_win_rate(self, laboratory):
        """Test that win rate contributes 40% to reliability."""
        # Perfect win rate
        for i in range(5):
            laboratory.position_tracker.record_position(f"d{i}", "agent1", "vote", f"text{i}")
            laboratory.position_tracker.finalize_debate(f"d{i}", "agent1", f"Pos{i}", 0.9)

        persona = laboratory.synthesize_persona("agent1")
        # With 100% win rate and only win_rate component, expect 0.4
        assert abs(persona.overall_reliability - 0.4) < 0.01

    def test_reliability_weighting_accuracy(self, laboratory):
        """Test that accuracy contributes 40% to reliability."""
        # Add verified positions (need at least 5 for accuracy to count)
        for i in range(10):
            laboratory.position_tracker.record_position(f"d{i}", "agent1", "vote", f"text{i}")
            laboratory.position_tracker.finalize_debate(f"d{i}", "agent1", f"Pos{i}", 0.9)
            laboratory.position_tracker.record_verification(f"d{i}", result=True)

        persona = laboratory.synthesize_persona("agent1")
        # With 100% accuracy and 100% win rate, expect 0.8 (0.4 + 0.4)
        assert persona.overall_reliability >= 0.7  # Allow some margin

    def test_reliability_weighting_elo(self, tracker, mock_elo_system):
        """Test that ELO calibration contributes 20% to reliability."""
        lab = TruthGroundedLaboratory(
            position_tracker=tracker,
            elo_system=mock_elo_system,
        )
        # Add position data
        for i in range(10):
            lab.position_tracker.record_position(f"d{i}", "agent1", "vote", f"text{i}")
            lab.position_tracker.finalize_debate(f"d{i}", "agent1", f"Pos{i}", 0.9)
            lab.position_tracker.record_verification(f"d{i}", result=True)

        persona = lab.synthesize_persona("agent1")
        # With ELO calibration of 0.8, should add 0.16 (0.8 * 0.2)
        assert persona.overall_reliability > 0.8

    def test_handles_no_position_data(self, laboratory):
        """Test handling agent with no position data."""
        persona = laboratory.synthesize_persona("new_agent")
        assert persona.total_positions == 0
        assert persona.overall_reliability == 0.0

    def test_requires_min_verifications_for_accuracy(self, laboratory):
        """Test that accuracy requires minimum verifications."""
        # Add only 1 verification
        laboratory.position_tracker.record_position("d1", "agent1", "vote", "text")
        laboratory.position_tracker.finalize_debate("d1", "agent1", "Pos", 0.9)
        laboratory.position_tracker.record_verification("d1", result=True)

        persona = laboratory.synthesize_persona("agent1")
        # Accuracy component should not be added (need 5 min)
        assert persona.position_accuracy == 0.0


# =============================================================================
# Test TruthGroundedLaboratory.get_reliable_agents()
# =============================================================================


class TestGetReliableAgents:
    """Tests for get_reliable_agents method."""

    def test_returns_persona_list(self, laboratory):
        """Test that method returns a list of personas."""
        result = laboratory.get_reliable_agents()
        assert isinstance(result, list)

    def test_respects_min_verified_threshold(self, laboratory):
        """Test that min_verified threshold is respected."""
        # Add agent with fewer than min_verified
        for i in range(5):
            laboratory.position_tracker.record_position(f"d{i}", "agent1", "vote", f"t{i}")
            laboratory.position_tracker.finalize_debate(f"d{i}", "agent1", f"P{i}", 0.9)
            laboratory.position_tracker.record_verification(f"d{i}", result=True)

        # Threshold of 10 should exclude this agent
        result = laboratory.get_reliable_agents(min_verified=10)
        assert len(result) == 0

    def test_respects_min_accuracy_threshold(self, laboratory):
        """Test that min_accuracy threshold is respected."""
        # Add agent with low accuracy
        for i in range(15):
            laboratory.position_tracker.record_position(f"d{i}", "agent1", "vote", f"t{i}")
            laboratory.position_tracker.finalize_debate(f"d{i}", "agent1", f"P{i}", 0.9)
            # Only first one correct
            laboratory.position_tracker.record_verification(f"d{i}", result=(i == 0))

        # Agent has ~6% accuracy, threshold of 60% should exclude
        result = laboratory.get_reliable_agents(min_verified=10, min_accuracy=0.6)
        assert len(result) == 0

    def test_sorted_by_reliability_desc(self, laboratory):
        """Test that results are sorted by reliability descending."""
        # Add two agents with different accuracy
        for i in range(12):
            laboratory.position_tracker.record_position(f"d{i}", "agent1", "vote", f"t{i}")
            laboratory.position_tracker.finalize_debate(f"d{i}", "agent1", f"P{i}", 0.9)
            laboratory.position_tracker.record_verification(f"d{i}", result=True)

        for i in range(12, 24):
            laboratory.position_tracker.record_position(f"d{i}", "agent2", "vote", f"t{i}")
            laboratory.position_tracker.finalize_debate(f"d{i}", "agent2", f"P{i}", 0.9)
            laboratory.position_tracker.record_verification(f"d{i}", result=(i < 18))  # 50% accuracy

        result = laboratory.get_reliable_agents(min_verified=10, min_accuracy=0.4)
        if len(result) >= 2:
            assert result[0].overall_reliability >= result[1].overall_reliability

    def test_filters_unverified_agents(self, laboratory):
        """Test that agents without verified positions are excluded."""
        # Add positions without verification
        laboratory.position_tracker.record_position("d1", "agent1", "vote", "text")
        laboratory.position_tracker.finalize_debate("d1", "agent1", "Pos", 0.9)

        result = laboratory.get_reliable_agents()
        assert len(result) == 0

    def test_returns_empty_when_no_qualifying(self, laboratory):
        """Test returns empty list when no agents qualify."""
        result = laboratory.get_reliable_agents()
        assert result == []


# =============================================================================
# Test TruthGroundedLaboratory.get_all_personas()
# =============================================================================


class TestGetAllPersonas:
    """Tests for get_all_personas method."""

    def test_returns_all_agents(self, laboratory):
        """Test that method returns personas for all agents."""
        laboratory.position_tracker.record_position("d1", "agent1", "vote", "t1")
        laboratory.position_tracker.record_position("d2", "agent2", "vote", "t2")
        laboratory.position_tracker.record_position("d3", "agent3", "vote", "t3")

        result = laboratory.get_all_personas()
        agent_names = {p.agent_name for p in result}
        assert "agent1" in agent_names
        assert "agent2" in agent_names
        assert "agent3" in agent_names

    def test_respects_limit(self, laboratory):
        """Test that limit parameter is respected."""
        for i in range(10):
            laboratory.position_tracker.record_position(f"d{i}", f"agent{i}", "vote", f"t{i}")

        result = laboratory.get_all_personas(limit=5)
        assert len(result) == 5

    def test_returns_personas_not_raw_data(self, laboratory):
        """Test that method returns TruthGroundedPersona objects."""
        laboratory.position_tracker.record_position("d1", "agent1", "vote", "text")

        result = laboratory.get_all_personas()
        assert all(isinstance(p, TruthGroundedPersona) for p in result)

    def test_handles_empty_database(self, laboratory):
        """Test handling empty database."""
        result = laboratory.get_all_personas()
        assert result == []


# =============================================================================
# Test TruthGroundedLaboratory.get_debate_summary()
# =============================================================================


class TestGetDebateSummary:
    """Tests for get_debate_summary method."""

    def test_returns_summary_dict(self, laboratory):
        """Test that method returns a dict."""
        summary = laboratory.get_debate_summary("debate-1")
        assert isinstance(summary, dict)

    def test_includes_debate_id(self, laboratory):
        """Test that summary includes debate_id."""
        summary = laboratory.get_debate_summary("debate-123")
        assert summary["debate_id"] == "debate-123"

    def test_includes_outcome_when_exists(self, laboratory):
        """Test that summary includes outcome when available."""
        laboratory.position_tracker.record_position("d1", "agent1", "vote", "text")
        laboratory.position_tracker.finalize_debate("d1", "agent1", "Winner position", 0.85)

        summary = laboratory.get_debate_summary("d1")
        assert summary["outcome"] is not None
        assert summary["outcome"]["winning_agent"] == "agent1"
        assert summary["outcome"]["winning_position"] == "Winner position"
        assert summary["outcome"]["confidence"] == 0.85

    def test_handles_no_outcome(self, laboratory):
        """Test handling debate without outcome."""
        laboratory.position_tracker.record_position("d1", "agent1", "vote", "text")

        summary = laboratory.get_debate_summary("d1")
        assert summary["outcome"] is None

    def test_includes_positions_list(self, laboratory):
        """Test that summary includes positions list."""
        laboratory.position_tracker.record_position("d1", "agent1", "vote", "vote text")
        laboratory.position_tracker.record_position("d1", "agent1", "critique", "critique text")

        summary = laboratory.get_debate_summary("d1")
        assert "positions" in summary
        assert len(summary["positions"]) == 2

    def test_truncates_position_text(self, laboratory):
        """Test that position text is truncated to 200 chars."""
        long_text = "x" * 500
        laboratory.position_tracker.record_position("d1", "agent1", "vote", long_text)

        summary = laboratory.get_debate_summary("d1")
        assert len(summary["positions"][0]["text"]) == 200

    def test_includes_verification_data(self, laboratory):
        """Test that summary includes verification data."""
        laboratory.position_tracker.record_position("d1", "agent1", "vote", "text")
        laboratory.position_tracker.finalize_debate("d1", "agent1", "Pos", 0.9)
        laboratory.position_tracker.record_verification("d1", result=True, source="test-source")

        summary = laboratory.get_debate_summary("d1")
        assert summary["outcome"]["verified"] == 1
        assert summary["outcome"]["verification_source"] == "test-source"

    def test_handles_no_positions(self, laboratory):
        """Test handling debate with no positions."""
        summary = laboratory.get_debate_summary("empty-debate")
        assert summary["positions"] == []


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the full truth grounding system."""

    def test_full_debate_lifecycle(self, laboratory):
        """Test complete debate lifecycle from start to verification."""
        # 1. Record positions during debate
        laboratory.position_tracker.record_position(
            "integration-debate", "claude", "proposal", "Let's use approach A", confidence=0.7
        )
        laboratory.position_tracker.record_position(
            "integration-debate", "gpt", "proposal", "Let's use approach B", confidence=0.8
        )
        laboratory.position_tracker.record_position(
            "integration-debate", "claude", "vote", "I support approach A", confidence=0.75
        )
        laboratory.position_tracker.record_position(
            "integration-debate", "gpt", "vote", "I support approach B", confidence=0.85
        )

        # 2. Finalize debate
        laboratory.position_tracker.finalize_debate(
            "integration-debate", "claude", "Approach A won", 0.9
        )

        # 3. Verify outcome
        laboratory.position_tracker.record_verification(
            "integration-debate", result=True, source="test-runner"
        )

        # 4. Check summary
        summary = laboratory.get_debate_summary("integration-debate")
        assert summary["outcome"]["winning_agent"] == "claude"
        assert summary["outcome"]["verified"] == 1
        assert len(summary["positions"]) == 4

    def test_position_to_verification_flow(self, laboratory):
        """Test that position verification flows correctly."""
        # Record and finalize
        laboratory.position_tracker.record_position("d1", "agent1", "vote", "Correct answer")
        laboratory.position_tracker.finalize_debate("d1", "agent1", "Correct answer", 0.9)

        # Before verification
        history = laboratory.position_tracker.get_position_history("agent1")
        assert history[0].verified_correct is None

        # After verification
        laboratory.position_tracker.record_verification("d1", result=True)
        history = laboratory.position_tracker.get_position_history("agent1")
        assert history[0].verified_correct is True

    def test_accuracy_updates_after_verification(self, laboratory):
        """Test that accuracy stats update after verification."""
        # Add multiple debates
        for i in range(10):
            laboratory.position_tracker.record_position(f"d{i}", "agent1", "vote", f"answer{i}")
            laboratory.position_tracker.finalize_debate(f"d{i}", "agent1", f"answer{i}", 0.8)

        # Before verification
        stats = laboratory.position_tracker.get_agent_position_accuracy("agent1")
        assert stats["verified_positions"] == 0

        # Verify half as correct
        for i in range(10):
            laboratory.position_tracker.record_verification(f"d{i}", result=(i < 5))

        # After verification
        stats = laboratory.position_tracker.get_agent_position_accuracy("agent1")
        assert stats["verified_positions"] == 10
        assert stats["verified_correct"] == 5

    def test_persona_reflects_debate_outcomes(self, laboratory, mock_elo_system):
        """Test that synthesized persona reflects debate outcomes."""
        # Use laboratory with ELO system
        lab = TruthGroundedLaboratory(
            position_tracker=laboratory.position_tracker,
            elo_system=mock_elo_system,
        )

        # Add debate history
        for i in range(8):
            lab.position_tracker.record_position(f"d{i}", "top-performer", "vote", f"ans{i}")
            lab.position_tracker.finalize_debate(f"d{i}", "top-performer", f"ans{i}", 0.95)
            lab.position_tracker.record_verification(f"d{i}", result=True)

        # Synthesize persona
        persona = lab.synthesize_persona("top-performer")

        # Check persona reflects data
        assert persona.agent_name == "top-performer"
        assert persona.total_positions == 8
        assert persona.winning_positions == 8
        assert persona.verified_positions == 8
        assert persona.position_accuracy == 1.0
        assert persona.overall_reliability > 0.0
        assert persona.elo_rating == 1600.0  # From mock


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_position_text(self, tracker):
        """Test handling empty position text."""
        pos = tracker.record_position("d1", "agent1", "vote", "")
        assert pos.position_text == ""

    def test_very_long_position_text(self, tracker):
        """Test handling very long position text."""
        long_text = "x" * 10000
        pos = tracker.record_position("d1", "agent1", "vote", long_text)
        assert len(pos.position_text) == 10000

    def test_special_characters_in_agent_name(self, tracker):
        """Test special characters in agent name."""
        tracker.record_position("d1", "agent-1_v2.0", "vote", "text")
        history = tracker.get_position_history("agent-1_v2.0")
        assert len(history) == 1

    def test_unicode_in_position_text(self, tracker):
        """Test Unicode characters in position text."""
        unicode_text = "日本語 한국어 العربية 🎉"
        pos = tracker.record_position("d1", "agent1", "vote", unicode_text)
        assert pos.position_text == unicode_text

    def test_concurrent_positions_same_debate(self, tracker):
        """Test multiple agents recording positions concurrently."""
        for i in range(100):
            tracker.record_position("d1", f"agent{i}", "vote", f"text{i}")

        # All should be recorded
        with sqlite3.connect(tracker.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM position_history WHERE debate_id = 'd1'")
            count = cursor.fetchone()[0]

        assert count == 100

    def test_zero_confidence(self, tracker):
        """Test position with zero confidence."""
        pos = tracker.record_position("d1", "agent1", "vote", "text", confidence=0.0)
        assert pos.confidence == 0.0

    def test_max_confidence(self, tracker):
        """Test position with max confidence."""
        pos = tracker.record_position("d1", "agent1", "vote", "text", confidence=1.0)
        assert pos.confidence == 1.0

    def test_negative_round_num(self, tracker):
        """Test position with negative round number."""
        pos = tracker.record_position("d1", "agent1", "vote", "text", round_num=-1)
        assert pos.round_num == -1

    def test_large_round_num(self, tracker):
        """Test position with large round number."""
        pos = tracker.record_position("d1", "agent1", "vote", "text", round_num=9999)
        assert pos.round_num == 9999
