"""
Tests for Truth-Grounded Persona System.

Tests cover:
- Position dataclass creation and from_row conversion
- PositionTracker recording, finalization, verification, and retrieval
- TruthGroundedPersona dataclass and computed properties
- TruthGroundedLaboratory persona synthesis, reliable agents, and debate summaries
- Database operations and error handling
- Integration with ELO system and PersonaManager
"""

import pytest
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from aragora.agents.truth_grounding import (
    Position,
    PositionTracker,
    TruthGroundedPersona,
    TruthGroundedLaboratory,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield Path(f.name)


@pytest.fixture
def position_tracker(temp_db):
    """Create a PositionTracker with a temporary database."""
    return PositionTracker(db_path=str(temp_db))


@pytest.fixture
def laboratory(temp_db):
    """Create a TruthGroundedLaboratory with a temporary database."""
    tracker = PositionTracker(db_path=str(temp_db))
    return TruthGroundedLaboratory(position_tracker=tracker)


@pytest.fixture
def mock_elo_system():
    """Create a mock ELO system."""
    elo = MagicMock()

    @dataclass
    class MockRating:
        elo: float = 1650.0
        win_rate: float = 0.65
        calibration_score: float = 0.72

    elo.get_rating.return_value = MockRating()
    return elo


@pytest.fixture
def mock_persona_manager():
    """Create a mock PersonaManager."""
    manager = MagicMock()
    manager.get_persona.return_value = MagicMock(
        agent_name="claude",
        traits=["thorough", "analytical"],
        expertise={"security": 0.9},
    )
    return manager


# ============================================================================
# Position Dataclass Tests
# ============================================================================


class TestPositionDataclass:
    """Tests for the Position dataclass."""

    def test_create_basic_position(self):
        """Test creating a basic position."""
        position = Position(
            debate_id="debate_001",
            agent_name="claude",
            position_type="proposal",
            position_text="We should implement rate limiting.",
        )

        assert position.debate_id == "debate_001"
        assert position.agent_name == "claude"
        assert position.position_type == "proposal"
        assert position.position_text == "We should implement rate limiting."

    def test_position_defaults(self):
        """Test position default values."""
        position = Position(
            debate_id="d1",
            agent_name="agent1",
            position_type="vote",
            position_text="Yes",
        )

        assert position.round_num == 0
        assert position.confidence == 0.5
        assert position.was_winning is None
        assert position.verified_correct is None
        assert position.created_at is not None

    def test_position_with_all_fields(self):
        """Test creating a position with all fields."""
        position = Position(
            debate_id="debate_002",
            agent_name="gemini",
            position_type="critique",
            position_text="This approach has scalability issues.",
            round_num=2,
            confidence=0.85,
            was_winning=True,
            verified_correct=True,
            created_at="2024-01-15T10:30:00",
        )

        assert position.round_num == 2
        assert position.confidence == 0.85
        assert position.was_winning is True
        assert position.verified_correct is True
        assert position.created_at == "2024-01-15T10:30:00"

    def test_from_row_basic(self):
        """Test creating Position from database row."""
        row = (
            "debate_001",
            "claude",
            "vote",
            "Agree",
            1,
            0.8,
            1,  # was_winning
            1,  # verified_correct
            "2024-01-15T10:30:00",
        )

        position = Position.from_row(row)

        assert position.debate_id == "debate_001"
        assert position.agent_name == "claude"
        assert position.position_type == "vote"
        assert position.position_text == "Agree"
        assert position.round_num == 1
        assert position.confidence == 0.8
        assert position.was_winning is True
        assert position.verified_correct is True

    def test_from_row_with_none_values(self):
        """Test creating Position from row with NULL values."""
        row = (
            "debate_002",
            "gpt4",
            "proposal",
            "Let's try option B",
            0,
            0.6,
            None,  # was_winning not set
            None,  # verified_correct not set
            "2024-01-16T14:00:00",
        )

        position = Position.from_row(row)

        assert position.was_winning is None
        assert position.verified_correct is None

    def test_from_row_false_values(self):
        """Test creating Position from row with false boolean values."""
        row = (
            "debate_003",
            "mistral",
            "vote",
            "Disagree",
            2,
            0.3,
            0,  # was_winning = False
            0,  # verified_correct = False
            "2024-01-17T09:00:00",
        )

        position = Position.from_row(row)

        assert position.was_winning is False
        assert position.verified_correct is False


# ============================================================================
# PositionTracker Tests
# ============================================================================


class TestPositionTracker:
    """Tests for the PositionTracker class."""

    def test_init_creates_database(self, temp_db):
        """Test that initialization creates the database."""
        tracker = PositionTracker(db_path=str(temp_db))

        assert tracker.db is not None
        assert tracker.db_path == temp_db

    def test_record_position_basic(self, position_tracker):
        """Test recording a basic position."""
        position = position_tracker.record_position(
            debate_id="debate_001",
            agent_name="claude",
            position_type="proposal",
            position_text="Implement caching for better performance.",
        )

        assert position.debate_id == "debate_001"
        assert position.agent_name == "claude"
        assert position.position_type == "proposal"
        assert position.confidence == 0.5  # default

    def test_record_position_with_confidence(self, position_tracker):
        """Test recording a position with custom confidence."""
        position = position_tracker.record_position(
            debate_id="debate_002",
            agent_name="gpt4",
            position_type="vote",
            position_text="I strongly agree.",
            round_num=3,
            confidence=0.92,
        )

        assert position.confidence == 0.92
        assert position.round_num == 3

    def test_record_multiple_positions(self, position_tracker):
        """Test recording multiple positions for the same debate."""
        position_tracker.record_position(
            debate_id="debate_001",
            agent_name="claude",
            position_type="proposal",
            position_text="Proposal from Claude",
            round_num=1,
        )
        position_tracker.record_position(
            debate_id="debate_001",
            agent_name="gemini",
            position_type="proposal",
            position_text="Proposal from Gemini",
            round_num=1,
        )
        position_tracker.record_position(
            debate_id="debate_001",
            agent_name="claude",
            position_type="vote",
            position_text="Final vote",
            round_num=2,
        )

        # Verify all positions are recorded
        history = position_tracker.get_position_history("claude")
        assert len(history) == 2

    def test_finalize_debate(self, position_tracker):
        """Test finalizing a debate marks winning positions."""
        # Record positions
        position_tracker.record_position(
            debate_id="debate_001",
            agent_name="claude",
            position_type="vote",
            position_text="Option A is better",
        )
        position_tracker.record_position(
            debate_id="debate_001",
            agent_name="gemini",
            position_type="vote",
            position_text="Option B is better",
        )

        # Finalize debate with Claude as winner
        position_tracker.finalize_debate(
            debate_id="debate_001",
            winning_agent="claude",
            winning_position="Option A is better",
            consensus_confidence=0.85,
        )

        # Check outcome is recorded
        with position_tracker.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT winning_agent, consensus_confidence FROM debate_outcomes WHERE debate_id = ?",
                ("debate_001",),
            )
            outcome = cursor.fetchone()

        assert outcome[0] == "claude"
        assert outcome[1] == 0.85

    def test_record_verification_correct(self, position_tracker):
        """Test recording a correct verification."""
        # Set up debate
        position_tracker.record_position(
            debate_id="debate_001",
            agent_name="claude",
            position_type="vote",
            position_text="This approach will work",
        )
        position_tracker.finalize_debate(
            debate_id="debate_001",
            winning_agent="claude",
            winning_position="This approach will work",
            consensus_confidence=0.8,
        )

        # Record verification
        position_tracker.record_verification(
            debate_id="debate_001",
            result=True,
            source="manual_review",
        )

        # Verify verification was recorded
        with position_tracker.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT verification_result, verification_source FROM debate_outcomes WHERE debate_id = ?",
                ("debate_001",),
            )
            result = cursor.fetchone()

        assert result[0] == 1  # True
        assert result[1] == "manual_review"

    def test_record_verification_incorrect(self, position_tracker):
        """Test recording an incorrect verification."""
        position_tracker.record_position(
            debate_id="debate_002",
            agent_name="gpt4",
            position_type="vote",
            position_text="This will definitely work",
        )
        position_tracker.finalize_debate(
            debate_id="debate_002",
            winning_agent="gpt4",
            winning_position="This will definitely work",
            consensus_confidence=0.9,
        )

        position_tracker.record_verification(
            debate_id="debate_002",
            result=False,
            source="production_failure",
        )

        with position_tracker.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT verification_result FROM debate_outcomes WHERE debate_id = ?",
                ("debate_002",),
            )
            result = cursor.fetchone()

        assert result[0] == 0  # False

    def test_get_agent_position_accuracy_no_data(self, position_tracker):
        """Test accuracy stats with no data."""
        stats = position_tracker.get_agent_position_accuracy("unknown_agent")

        assert stats["total_positions"] == 0
        assert stats["winning_positions"] == 0
        assert stats["verified_positions"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["accuracy_rate"] == 0.0
        assert stats["calibration"] == 0.0

    def test_get_agent_position_accuracy_with_data(self, position_tracker):
        """Test accuracy stats with data."""
        # Record multiple verified positions
        for i in range(10):
            debate_id = f"debate_{i:03d}"
            position_tracker.record_position(
                debate_id=debate_id,
                agent_name="claude",
                position_type="vote",
                position_text=f"Position {i}",
                confidence=0.7,
            )
            position_tracker.finalize_debate(
                debate_id=debate_id,
                winning_agent="claude",
                winning_position=f"Position {i}",
                consensus_confidence=0.8,
            )
            # 7 out of 10 are correct
            position_tracker.record_verification(
                debate_id=debate_id,
                result=(i < 7),
                source="test",
            )

        stats = position_tracker.get_agent_position_accuracy("claude")

        assert stats["total_positions"] == 10
        assert stats["winning_positions"] == 10
        assert stats["verified_positions"] == 10
        assert stats["verified_correct"] == 7
        assert stats["win_rate"] == 1.0
        assert stats["accuracy_rate"] == 0.7

    def test_get_agent_position_accuracy_min_verifications(self, position_tracker):
        """Test that accuracy requires minimum verifications."""
        # Only 3 verified positions (below default min of 5)
        for i in range(3):
            debate_id = f"debate_{i:03d}"
            position_tracker.record_position(
                debate_id=debate_id,
                agent_name="claude",
                position_type="vote",
                position_text=f"Position {i}",
            )
            position_tracker.finalize_debate(
                debate_id=debate_id,
                winning_agent="claude",
                winning_position=f"Position {i}",
                consensus_confidence=0.8,
            )
            position_tracker.record_verification(
                debate_id=debate_id,
                result=True,
                source="test",
            )

        stats = position_tracker.get_agent_position_accuracy("claude", min_verifications=5)

        assert stats["verified_positions"] == 3
        assert stats["accuracy_rate"] == 0.0  # Below min threshold
        assert stats["calibration"] == 0.0

    def test_get_position_history_empty(self, position_tracker):
        """Test getting empty position history."""
        history = position_tracker.get_position_history("unknown_agent")

        assert history == []

    def test_get_position_history_with_limit(self, position_tracker):
        """Test position history respects limit."""
        for i in range(20):
            position_tracker.record_position(
                debate_id=f"debate_{i:03d}",
                agent_name="claude",
                position_type="proposal",
                position_text=f"Position {i}",
            )

        history = position_tracker.get_position_history("claude", limit=10)

        assert len(history) == 10

    def test_get_position_history_verified_only(self, position_tracker):
        """Test filtering for verified positions only."""
        # Create some verified and unverified positions
        for i in range(5):
            debate_id = f"debate_{i:03d}"
            position_tracker.record_position(
                debate_id=debate_id,
                agent_name="claude",
                position_type="vote",
                position_text=f"Position {i}",
            )
            position_tracker.finalize_debate(
                debate_id=debate_id,
                winning_agent="claude",
                winning_position=f"Position {i}",
                consensus_confidence=0.8,
            )
            # Only verify first 3
            if i < 3:
                position_tracker.record_verification(
                    debate_id=debate_id,
                    result=True,
                    source="test",
                )

        all_history = position_tracker.get_position_history("claude")
        verified_history = position_tracker.get_position_history("claude", verified_only=True)

        assert len(all_history) == 5
        assert len(verified_history) == 3


# ============================================================================
# TruthGroundedPersona Dataclass Tests
# ============================================================================


class TestTruthGroundedPersonaDataclass:
    """Tests for the TruthGroundedPersona dataclass."""

    def test_create_basic_persona(self):
        """Test creating a basic persona."""
        persona = TruthGroundedPersona(agent_name="claude")

        assert persona.agent_name == "claude"
        assert persona.elo_rating == 1500.0
        assert persona.elo_win_rate == 0.0
        assert persona.overall_reliability == 0.0

    def test_persona_defaults(self):
        """Test persona default values."""
        persona = TruthGroundedPersona(agent_name="test_agent")

        assert persona.elo_calibration == 0.0
        assert persona.position_accuracy == 0.0
        assert persona.verified_positions == 0
        assert persona.winning_positions == 0
        assert persona.total_positions == 0
        assert persona.strength_domains == []
        assert persona.weakness_domains == []
        assert persona.contrarian_score == 0.0
        assert persona.early_adopter_score == 0.0

    def test_persona_with_full_data(self):
        """Test creating a fully populated persona."""
        persona = TruthGroundedPersona(
            agent_name="claude",
            elo_rating=1750.0,
            elo_win_rate=0.68,
            elo_calibration=0.82,
            position_accuracy=0.73,
            verified_positions=50,
            winning_positions=35,
            total_positions=75,
            overall_reliability=0.76,
            strength_domains=["security", "architecture"],
            weakness_domains=["UI/UX"],
            contrarian_score=0.15,
            early_adopter_score=0.42,
        )

        assert persona.elo_rating == 1750.0
        assert persona.elo_win_rate == 0.68
        assert persona.position_accuracy == 0.73
        assert persona.verified_positions == 50
        assert persona.strength_domains == ["security", "architecture"]


# ============================================================================
# TruthGroundedLaboratory Tests
# ============================================================================


class TestTruthGroundedLaboratory:
    """Tests for the TruthGroundedLaboratory class."""

    def test_init_with_tracker(self, temp_db):
        """Test initialization with explicit tracker."""
        tracker = PositionTracker(db_path=str(temp_db))
        lab = TruthGroundedLaboratory(position_tracker=tracker)

        assert lab.position_tracker is tracker
        assert lab.elo_system is None
        assert lab.persona_manager is None

    def test_init_creates_default_tracker(self, temp_db):
        """Test initialization creates default tracker."""
        with patch("aragora.agents.truth_grounding.resolve_db_path") as mock_resolve:
            mock_resolve.return_value = str(temp_db)
            lab = TruthGroundedLaboratory(db_path=str(temp_db))

            assert lab.position_tracker is not None

    def test_synthesize_persona_no_data(self, laboratory):
        """Test synthesizing persona with no data."""
        persona = laboratory.synthesize_persona("unknown_agent")

        assert persona.agent_name == "unknown_agent"
        assert persona.elo_rating == 1500.0
        assert persona.total_positions == 0
        assert persona.overall_reliability == 0.0

    def test_synthesize_persona_with_positions(self, laboratory):
        """Test synthesizing persona with position data."""
        tracker = laboratory.position_tracker

        # Create position data
        for i in range(10):
            debate_id = f"debate_{i:03d}"
            tracker.record_position(
                debate_id=debate_id,
                agent_name="claude",
                position_type="vote",
                position_text=f"Position {i}",
                confidence=0.75,
            )
            tracker.finalize_debate(
                debate_id=debate_id,
                winning_agent="claude",
                winning_position=f"Position {i}",
                consensus_confidence=0.8,
            )
            # 8 out of 10 correct
            tracker.record_verification(
                debate_id=debate_id,
                result=(i < 8),
                source="test",
            )

        persona = laboratory.synthesize_persona("claude")

        assert persona.total_positions == 10
        assert persona.winning_positions == 10
        assert persona.verified_positions == 10
        assert persona.position_accuracy == 0.8

    def test_synthesize_persona_with_elo_system(self, temp_db, mock_elo_system):
        """Test synthesizing persona with ELO system."""
        tracker = PositionTracker(db_path=str(temp_db))
        lab = TruthGroundedLaboratory(
            position_tracker=tracker,
            elo_system=mock_elo_system,
        )

        persona = lab.synthesize_persona("claude")

        assert persona.elo_rating == 1650.0
        assert persona.elo_win_rate == 0.65
        assert persona.elo_calibration == 0.72

    def test_synthesize_persona_reliability_calculation(self, temp_db, mock_elo_system):
        """Test reliability calculation combines multiple sources."""
        tracker = PositionTracker(db_path=str(temp_db))
        lab = TruthGroundedLaboratory(
            position_tracker=tracker,
            elo_system=mock_elo_system,
        )

        # Add positions with good accuracy
        for i in range(10):
            debate_id = f"debate_{i:03d}"
            tracker.record_position(
                debate_id=debate_id,
                agent_name="claude",
                position_type="vote",
                position_text=f"Position {i}",
            )
            tracker.finalize_debate(
                debate_id=debate_id,
                winning_agent="claude",
                winning_position=f"Position {i}",
                consensus_confidence=0.8,
            )
            tracker.record_verification(
                debate_id=debate_id,
                result=(i < 9),  # 90% accuracy
                source="test",
            )

        persona = lab.synthesize_persona("claude")

        # Reliability should combine win_rate, accuracy, and calibration
        # 40% win_rate (1.0) + 40% accuracy (0.9) + 20% calibration (0.72)
        # = 0.4 + 0.36 + 0.144 = 0.904
        assert persona.overall_reliability > 0.0

    def test_get_reliable_agents_empty(self, laboratory):
        """Test getting reliable agents with no data."""
        reliable = laboratory.get_reliable_agents()

        assert reliable == []

    def test_get_reliable_agents_filters_by_accuracy(self, laboratory):
        """Test getting reliable agents filters by accuracy."""
        tracker = laboratory.position_tracker

        # Create high accuracy agent
        for i in range(15):
            debate_id = f"high_debate_{i:03d}"
            tracker.record_position(
                debate_id=debate_id,
                agent_name="high_accuracy",
                position_type="vote",
                position_text=f"Position {i}",
            )
            tracker.finalize_debate(
                debate_id=debate_id,
                winning_agent="high_accuracy",
                winning_position=f"Position {i}",
                consensus_confidence=0.8,
            )
            tracker.record_verification(
                debate_id=debate_id,
                result=(i < 12),  # 80% accuracy
                source="test",
            )

        # Create low accuracy agent
        for i in range(15):
            debate_id = f"low_debate_{i:03d}"
            tracker.record_position(
                debate_id=debate_id,
                agent_name="low_accuracy",
                position_type="vote",
                position_text=f"Position {i}",
            )
            tracker.finalize_debate(
                debate_id=debate_id,
                winning_agent="low_accuracy",
                winning_position=f"Position {i}",
                consensus_confidence=0.8,
            )
            tracker.record_verification(
                debate_id=debate_id,
                result=(i < 5),  # 33% accuracy
                source="test",
            )

        reliable = laboratory.get_reliable_agents(min_verified=10, min_accuracy=0.6)

        agent_names = [p.agent_name for p in reliable]
        assert "high_accuracy" in agent_names
        assert "low_accuracy" not in agent_names

    def test_get_all_personas(self, laboratory):
        """Test getting all personas."""
        tracker = laboratory.position_tracker

        # Create data for multiple agents
        for agent in ["claude", "gemini", "gpt4"]:
            tracker.record_position(
                debate_id=f"debate_{agent}",
                agent_name=agent,
                position_type="proposal",
                position_text=f"Position from {agent}",
            )

        personas = laboratory.get_all_personas()

        assert len(personas) == 3
        agent_names = [p.agent_name for p in personas]
        assert "claude" in agent_names
        assert "gemini" in agent_names
        assert "gpt4" in agent_names

    def test_get_all_personas_respects_limit(self, laboratory):
        """Test that get_all_personas respects limit."""
        tracker = laboratory.position_tracker

        for i in range(10):
            tracker.record_position(
                debate_id=f"debate_{i:03d}",
                agent_name=f"agent_{i}",
                position_type="proposal",
                position_text=f"Position {i}",
            )

        personas = laboratory.get_all_personas(limit=5)

        assert len(personas) == 5

    def test_get_debate_summary_no_data(self, laboratory):
        """Test getting summary for non-existent debate."""
        summary = laboratory.get_debate_summary("unknown_debate")

        assert summary["debate_id"] == "unknown_debate"
        assert summary["outcome"] is None
        assert summary["positions"] == []

    def test_get_debate_summary_with_data(self, laboratory):
        """Test getting summary for debate with data."""
        tracker = laboratory.position_tracker

        # Create debate with multiple positions
        tracker.record_position(
            debate_id="debate_001",
            agent_name="claude",
            position_type="proposal",
            position_text="Proposal from Claude - a long text that describes the approach",
            confidence=0.8,
        )
        tracker.record_position(
            debate_id="debate_001",
            agent_name="gemini",
            position_type="critique",
            position_text="Critique from Gemini",
            confidence=0.7,
        )

        tracker.finalize_debate(
            debate_id="debate_001",
            winning_agent="claude",
            winning_position="Proposal from Claude",
            consensus_confidence=0.85,
        )

        tracker.record_verification(
            debate_id="debate_001",
            result=True,
            source="production",
        )

        summary = laboratory.get_debate_summary("debate_001")

        assert summary["debate_id"] == "debate_001"
        assert summary["outcome"]["winning_agent"] == "claude"
        assert summary["outcome"]["confidence"] == 0.85
        assert summary["outcome"]["verified"] == 1
        assert summary["outcome"]["verification_source"] == "production"
        assert len(summary["positions"]) == 2


# ============================================================================
# Module Exports Tests
# ============================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports_importable(self):
        """All items in __all__ are importable."""
        from aragora.agents import truth_grounding

        for name in truth_grounding.__all__:
            assert hasattr(truth_grounding, name), f"{name} not found in module"

    def test_exports_contain_key_items(self):
        """Module exports include key classes."""
        from aragora.agents.truth_grounding import __all__

        assert "Position" in __all__
        assert "PositionTracker" in __all__
        assert "TruthGroundedPersona" in __all__
        assert "TruthGroundedLaboratory" in __all__


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_position_with_empty_text(self, position_tracker):
        """Test recording a position with empty text."""
        position = position_tracker.record_position(
            debate_id="debate_001",
            agent_name="claude",
            position_type="vote",
            position_text="",
        )

        assert position.position_text == ""

    def test_position_with_special_characters(self, position_tracker):
        """Test recording a position with special characters."""
        special_text = "Let's use 'quotes' and \"double quotes\" with <tags> & symbols!"
        position = position_tracker.record_position(
            debate_id="debate_001",
            agent_name="claude",
            position_type="proposal",
            position_text=special_text,
        )

        assert position.position_text == special_text

    def test_position_with_unicode(self, position_tracker):
        """Test recording a position with unicode characters."""
        unicode_text = "Unicode support: \u4e2d\u6587 \u0420\u0443\u0441\u0441\u043a\u0438\u0439 \u65e5\u672c\u8a9e"
        position = position_tracker.record_position(
            debate_id="debate_001",
            agent_name="claude",
            position_type="proposal",
            position_text=unicode_text,
        )

        history = position_tracker.get_position_history("claude")
        assert history[0].position_text == unicode_text

    def test_finalize_nonexistent_debate(self, position_tracker):
        """Test finalizing a debate that doesn't have positions."""
        # Should not raise error
        position_tracker.finalize_debate(
            debate_id="nonexistent_debate",
            winning_agent="claude",
            winning_position="Some position",
            consensus_confidence=0.9,
        )

        # Outcome should still be recorded
        with position_tracker.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT winning_agent FROM debate_outcomes WHERE debate_id = ?",
                ("nonexistent_debate",),
            )
            result = cursor.fetchone()

        assert result is not None
        assert result[0] == "claude"

    def test_record_verification_nonexistent_debate(self, position_tracker):
        """Test recording verification for non-existent debate."""
        # Should not raise error
        position_tracker.record_verification(
            debate_id="nonexistent_debate",
            result=True,
            source="test",
        )

        # No outcome to update, but should not fail

    def test_get_accuracy_with_only_losses(self, position_tracker):
        """Test accuracy when agent never wins."""
        for i in range(10):
            debate_id = f"debate_{i:03d}"
            position_tracker.record_position(
                debate_id=debate_id,
                agent_name="loser",
                position_type="vote",
                position_text=f"Position {i}",
            )
            # Winner is always "winner", not "loser"
            position_tracker.finalize_debate(
                debate_id=debate_id,
                winning_agent="winner",
                winning_position="Winner's position",
                consensus_confidence=0.8,
            )

        stats = position_tracker.get_agent_position_accuracy("loser")

        assert stats["total_positions"] == 10
        assert stats["winning_positions"] == 0
        assert stats["win_rate"] == 0.0

    def test_position_replacement(self, position_tracker):
        """Test that positions can be replaced (INSERT OR REPLACE)."""
        # Record initial position
        position_tracker.record_position(
            debate_id="debate_001",
            agent_name="claude",
            position_type="vote",
            position_text="Initial position",
            round_num=1,
            confidence=0.5,
        )

        # Replace with new position
        position_tracker.record_position(
            debate_id="debate_001",
            agent_name="claude",
            position_type="vote",
            position_text="Updated position",
            round_num=1,
            confidence=0.9,
        )

        history = position_tracker.get_position_history("claude")
        assert len(history) == 1
        assert history[0].position_text == "Updated position"
        assert history[0].confidence == 0.9

    def test_debate_summary_truncates_long_text(self, laboratory):
        """Test that debate summary truncates long position text."""
        tracker = laboratory.position_tracker

        long_text = "A" * 500  # 500 character text
        tracker.record_position(
            debate_id="debate_001",
            agent_name="claude",
            position_type="proposal",
            position_text=long_text,
        )

        summary = laboratory.get_debate_summary("debate_001")

        # Text should be truncated to 200 characters
        assert len(summary["positions"][0]["text"]) == 200

    def test_synthesize_persona_elo_error_propagates(self, laboratory):
        """Test that ELO lookup errors propagate."""
        mock_elo = MagicMock()
        mock_elo.get_rating.side_effect = KeyError("Agent not found")

        lab = TruthGroundedLaboratory(
            position_tracker=laboratory.position_tracker,
            elo_system=mock_elo,
        )

        # Error should propagate since code doesn't catch exceptions
        with pytest.raises(KeyError):
            lab.synthesize_persona("unknown")

    def test_synthesize_persona_no_elo_system(self, laboratory):
        """Test persona synthesis without ELO system uses defaults."""
        # Laboratory without ELO system
        assert laboratory.elo_system is None

        persona = laboratory.synthesize_persona("some_agent")

        assert persona.elo_rating == 1500.0
        assert persona.elo_win_rate == 0.0
        assert persona.elo_calibration == 0.0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_debate_lifecycle(self, laboratory):
        """Test complete debate lifecycle: record, finalize, verify, analyze."""
        tracker = laboratory.position_tracker

        # Phase 1: Record positions
        tracker.record_position(
            debate_id="lifecycle_debate",
            agent_name="claude",
            position_type="proposal",
            position_text="We should use microservices",
            round_num=1,
            confidence=0.85,
        )
        tracker.record_position(
            debate_id="lifecycle_debate",
            agent_name="gemini",
            position_type="proposal",
            position_text="Monolith is better for this case",
            round_num=1,
            confidence=0.75,
        )
        tracker.record_position(
            debate_id="lifecycle_debate",
            agent_name="claude",
            position_type="vote",
            position_text="I maintain my position",
            round_num=2,
            confidence=0.88,
        )
        tracker.record_position(
            debate_id="lifecycle_debate",
            agent_name="gemini",
            position_type="vote",
            position_text="I concede, microservices are better here",
            round_num=2,
            confidence=0.6,
        )

        # Phase 2: Finalize debate
        tracker.finalize_debate(
            debate_id="lifecycle_debate",
            winning_agent="claude",
            winning_position="We should use microservices",
            consensus_confidence=0.82,
        )

        # Phase 3: Verify outcome
        tracker.record_verification(
            debate_id="lifecycle_debate",
            result=True,
            source="6_month_review",
        )

        # Phase 4: Analyze
        summary = laboratory.get_debate_summary("lifecycle_debate")
        claude_persona = laboratory.synthesize_persona("claude")
        gemini_persona = laboratory.synthesize_persona("gemini")

        # Verify summary
        assert summary["outcome"]["winning_agent"] == "claude"
        assert summary["outcome"]["verified"] == 1
        assert len(summary["positions"]) == 4

        # Claude should have winning stats
        assert claude_persona.winning_positions >= 1

    def test_multiple_agents_accuracy_comparison(self, laboratory):
        """Test comparing accuracy across multiple agents."""
        tracker = laboratory.position_tracker

        # Agent 1: 80% accuracy
        for i in range(10):
            debate_id = f"agent1_debate_{i}"
            tracker.record_position(
                debate_id=debate_id,
                agent_name="accurate_agent",
                position_type="vote",
                position_text=f"Position {i}",
            )
            tracker.finalize_debate(
                debate_id=debate_id,
                winning_agent="accurate_agent",
                winning_position=f"Position {i}",
                consensus_confidence=0.8,
            )
            tracker.record_verification(
                debate_id=debate_id,
                result=(i < 8),
                source="test",
            )

        # Agent 2: 50% accuracy
        for i in range(10):
            debate_id = f"agent2_debate_{i}"
            tracker.record_position(
                debate_id=debate_id,
                agent_name="average_agent",
                position_type="vote",
                position_text=f"Position {i}",
            )
            tracker.finalize_debate(
                debate_id=debate_id,
                winning_agent="average_agent",
                winning_position=f"Position {i}",
                consensus_confidence=0.8,
            )
            tracker.record_verification(
                debate_id=debate_id,
                result=(i < 5),
                source="test",
            )

        accurate_persona = laboratory.synthesize_persona("accurate_agent")
        average_persona = laboratory.synthesize_persona("average_agent")

        assert accurate_persona.position_accuracy > average_persona.position_accuracy
        assert accurate_persona.position_accuracy == 0.8
        assert average_persona.position_accuracy == 0.5

    def test_reliable_agents_sorted_by_reliability(self, temp_db, mock_elo_system):
        """Test that reliable agents are sorted by reliability."""
        tracker = PositionTracker(db_path=str(temp_db))
        lab = TruthGroundedLaboratory(
            position_tracker=tracker,
            elo_system=mock_elo_system,
        )

        # Create agents with different accuracy levels
        accuracies = [("best", 10), ("good", 8), ("okay", 7)]

        for agent, correct_count in accuracies:
            for i in range(12):
                debate_id = f"{agent}_debate_{i}"
                tracker.record_position(
                    debate_id=debate_id,
                    agent_name=agent,
                    position_type="vote",
                    position_text=f"Position {i}",
                )
                tracker.finalize_debate(
                    debate_id=debate_id,
                    winning_agent=agent,
                    winning_position=f"Position {i}",
                    consensus_confidence=0.8,
                )
                tracker.record_verification(
                    debate_id=debate_id,
                    result=(i < correct_count),
                    source="test",
                )

        reliable = lab.get_reliable_agents(min_verified=10, min_accuracy=0.5)

        # Should be sorted by overall_reliability descending
        assert len(reliable) == 3
        assert reliable[0].agent_name == "best"
        assert reliable[1].agent_name == "good"
        assert reliable[2].agent_name == "okay"
