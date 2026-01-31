"""
Tests for Grounded Personas - Evidence-based agent identity system.

Tests cover:
- GroundedPersona dataclass properties and defaults
- PersonaSynthesizer prompt generation and data integration
- SignificantMoment tracking and serialization
- MomentDetector for detecting debate moments
- Position tracking and calibration
- Relationship computation
- Identity prompt generation
- Performance data integration
- Error handling and edge cases
"""

import pytest
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

from aragora.agents.grounded import (
    GroundedPersona,
    PersonaSynthesizer,
    SignificantMoment,
    MomentDetector,
)
from aragora.agents.positions import (
    Position,
    CalibrationBucket,
    DomainCalibration,
    PositionLedger,
)
from aragora.agents.relationships import RelationshipTracker, AgentRelationship
from aragora.agents.personas import Persona, PersonaManager
from aragora.ranking.relationships import RelationshipMetrics


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield Path(f.name)


@pytest.fixture
def mock_persona_manager():
    """Create a mock PersonaManager."""
    manager = MagicMock(spec=PersonaManager)
    manager.get_persona.return_value = Persona(
        agent_name="claude",
        traits=["thorough", "analytical"],
        expertise={"security": 0.9, "testing": 0.7},
        description="A thorough and analytical AI assistant",
    )
    return manager


@pytest.fixture
def mock_elo_system():
    """Create a mock ELO system."""
    elo = MagicMock()

    @dataclass
    class MockRating:
        elo: float = 1650.0
        wins: int = 25
        losses: int = 10
        draws: int = 5
        domain_elos: dict = None
        calibration_score: float = 0.72

        def __post_init__(self):
            if self.domain_elos is None:
                self.domain_elos = {"security": 1700.0, "testing": 1620.0}

    elo.get_rating.return_value = MockRating()
    return elo


@pytest.fixture
def mock_position_ledger():
    """Create a mock PositionLedger."""
    ledger = MagicMock(spec=PositionLedger)
    ledger.get_position_stats.return_value = {
        "total": 50,
        "correct": 35,
        "incorrect": 10,
        "unresolved": 3,
        "pending": 2,
        "reversals": 5,
    }
    return ledger


@pytest.fixture
def mock_relationship_tracker():
    """Create a mock RelationshipTracker."""
    tracker = MagicMock(spec=RelationshipTracker)

    # Mock rivals with RelationshipMetrics
    tracker.get_rivals.return_value = [
        RelationshipMetrics(
            agent_a="claude",
            agent_b="gemini",
            rivalry_score=0.75,
            alliance_score=0.2,
            relationship="rival",
            debate_count=15,
        ),
        RelationshipMetrics(
            agent_a="claude",
            agent_b="grok",
            rivalry_score=0.45,
            alliance_score=0.3,
            relationship="rival",
            debate_count=8,
        ),
    ]

    # Mock allies
    tracker.get_allies.return_value = [
        RelationshipMetrics(
            agent_a="claude",
            agent_b="gpt4",
            rivalry_score=0.15,
            alliance_score=0.8,
            relationship="ally",
            debate_count=20,
        ),
    ]

    # Mock influence network
    tracker.get_influence_network.return_value = {
        "influences": [("gpt4", 0.6), ("gemini", 0.4)],
        "influenced_by": [("mistral", 0.5)],
    }

    # Mock get_relationship
    def mock_get_relationship(agent_a, agent_b):
        return AgentRelationship(
            agent_a=min(agent_a, agent_b),
            agent_b=max(agent_a, agent_b),
            debate_count=10,
            agreement_count=3,
            a_wins_over_b=6,
            b_wins_over_a=4,
        )

    tracker.get_relationship.side_effect = mock_get_relationship

    return tracker


# ============================================================================
# GroundedPersona Tests
# ============================================================================


class TestGroundedPersonaDataclass:
    """Tests for the GroundedPersona dataclass."""

    def test_create_basic_grounded_persona(self):
        """Test creating a basic grounded persona."""
        persona = GroundedPersona(agent_name="test_agent")

        assert persona.agent_name == "test_agent"
        assert persona.elo == 1500.0
        assert persona.win_rate == 0.0
        assert persona.games_played == 0

    def test_grounded_persona_defaults(self):
        """Test grounded persona default values."""
        persona = GroundedPersona(agent_name="default_agent")

        assert persona.base_persona is None
        assert persona.domain_elos == {}
        assert persona.positions_taken == 0
        assert persona.positions_correct == 0
        assert persona.positions_incorrect == 0
        assert persona.reversals == 0
        assert persona.overall_calibration == 0.5
        assert persona.domain_calibrations == {}
        assert persona.rivals == []
        assert persona.allies == []
        assert persona.influences == []
        assert persona.influenced_by == []

    def test_reversal_rate_zero_positions(self):
        """Test reversal rate when no positions taken."""
        persona = GroundedPersona(agent_name="no_positions")

        assert persona.reversal_rate == 0.0

    def test_reversal_rate_with_positions(self):
        """Test reversal rate calculation with positions."""
        persona = GroundedPersona(
            agent_name="with_positions",
            positions_taken=20,
            reversals=4,
        )

        assert persona.reversal_rate == 0.2

    def test_position_accuracy_no_resolved(self):
        """Test position accuracy when no positions resolved."""
        persona = GroundedPersona(
            agent_name="no_resolved",
            positions_taken=10,
            positions_correct=0,
            positions_incorrect=0,
        )

        assert persona.position_accuracy == 0.0

    def test_position_accuracy_with_resolved(self):
        """Test position accuracy calculation with resolved positions."""
        persona = GroundedPersona(
            agent_name="with_resolved",
            positions_correct=7,
            positions_incorrect=3,
        )

        assert persona.position_accuracy == 0.7

    def test_grounded_persona_with_full_data(self):
        """Test creating a fully populated grounded persona."""
        base = Persona(agent_name="full_agent", traits=["thorough"])

        persona = GroundedPersona(
            agent_name="full_agent",
            base_persona=base,
            elo=1750.0,
            domain_elos={"security": 1800.0},
            win_rate=0.65,
            games_played=100,
            positions_taken=50,
            positions_correct=35,
            positions_incorrect=10,
            reversals=5,
            overall_calibration=0.78,
        )

        assert persona.base_persona == base
        assert persona.elo == 1750.0
        assert persona.win_rate == 0.65
        assert persona.position_accuracy == pytest.approx(0.777, rel=0.01)
        assert persona.reversal_rate == 0.1


# ============================================================================
# PersonaSynthesizer Tests
# ============================================================================


class TestPersonaSynthesizer:
    """Tests for the PersonaSynthesizer class."""

    def test_init_with_no_components(self):
        """Test initializing synthesizer with no components."""
        synthesizer = PersonaSynthesizer()

        assert synthesizer.persona_manager is None
        assert synthesizer.elo_system is None
        assert synthesizer.position_ledger is None
        assert synthesizer.relationship_tracker is None

    def test_init_with_all_components(
        self,
        mock_persona_manager,
        mock_elo_system,
        mock_position_ledger,
        mock_relationship_tracker,
    ):
        """Test initializing synthesizer with all components."""
        synthesizer = PersonaSynthesizer(
            persona_manager=mock_persona_manager,
            elo_system=mock_elo_system,
            position_ledger=mock_position_ledger,
            relationship_tracker=mock_relationship_tracker,
        )

        assert synthesizer.persona_manager == mock_persona_manager
        assert synthesizer.elo_system == mock_elo_system
        assert synthesizer.position_ledger == mock_position_ledger
        assert synthesizer.relationship_tracker == mock_relationship_tracker

    def test_get_grounded_persona_empty(self):
        """Test getting grounded persona with no components."""
        synthesizer = PersonaSynthesizer()
        persona = synthesizer.get_grounded_persona("test_agent")

        assert persona.agent_name == "test_agent"
        assert persona.base_persona is None
        assert persona.elo == 1500.0

    def test_get_grounded_persona_with_persona_manager(self, mock_persona_manager):
        """Test getting grounded persona with PersonaManager."""
        synthesizer = PersonaSynthesizer(persona_manager=mock_persona_manager)
        persona = synthesizer.get_grounded_persona("claude")

        assert persona.base_persona is not None
        assert persona.base_persona.agent_name == "claude"
        assert "thorough" in persona.base_persona.traits

    def test_get_grounded_persona_with_elo_system(self, mock_elo_system):
        """Test getting grounded persona with ELO system."""
        synthesizer = PersonaSynthesizer(elo_system=mock_elo_system)
        persona = synthesizer.get_grounded_persona("claude")

        assert persona.elo == 1650.0
        assert persona.games_played == 40  # 25 + 10 + 5
        assert persona.win_rate == pytest.approx(0.625, rel=0.01)  # 25/40
        assert persona.overall_calibration == 0.72
        assert "security" in persona.domain_elos

    def test_get_grounded_persona_with_position_ledger(self, mock_position_ledger):
        """Test getting grounded persona with PositionLedger."""
        synthesizer = PersonaSynthesizer(position_ledger=mock_position_ledger)
        persona = synthesizer.get_grounded_persona("claude")

        assert persona.positions_taken == 50
        assert persona.positions_correct == 35
        assert persona.positions_incorrect == 10
        assert persona.reversals == 5

    def test_get_grounded_persona_with_relationship_tracker(self, mock_relationship_tracker):
        """Test getting grounded persona with RelationshipTracker."""
        synthesizer = PersonaSynthesizer(relationship_tracker=mock_relationship_tracker)
        persona = synthesizer.get_grounded_persona("claude")

        assert len(persona.rivals) == 2
        assert len(persona.allies) == 1
        assert len(persona.influences) == 2
        assert len(persona.influenced_by) == 1

    def test_get_grounded_persona_full_integration(
        self,
        mock_persona_manager,
        mock_elo_system,
        mock_position_ledger,
        mock_relationship_tracker,
    ):
        """Test getting grounded persona with all components."""
        synthesizer = PersonaSynthesizer(
            persona_manager=mock_persona_manager,
            elo_system=mock_elo_system,
            position_ledger=mock_position_ledger,
            relationship_tracker=mock_relationship_tracker,
        )
        persona = synthesizer.get_grounded_persona("claude")

        # All data should be populated
        assert persona.base_persona is not None
        assert persona.elo == 1650.0
        assert persona.positions_taken == 50
        assert len(persona.rivals) == 2

    def test_synthesize_identity_prompt_basic(self):
        """Test synthesizing a basic identity prompt."""
        synthesizer = PersonaSynthesizer()
        prompt = synthesizer.synthesize_identity_prompt("test_agent")

        assert "## Your Identity: test_agent" in prompt

    def test_synthesize_identity_prompt_with_performance(self, mock_elo_system):
        """Test identity prompt includes performance section."""
        synthesizer = PersonaSynthesizer(elo_system=mock_elo_system)
        prompt = synthesizer.synthesize_identity_prompt("claude")

        assert "### Your Track Record" in prompt
        assert "ELO Rating" in prompt
        assert "Win Rate" in prompt

    def test_synthesize_identity_prompt_with_calibration(self, mock_elo_system):
        """Test identity prompt includes calibration section."""
        synthesizer = PersonaSynthesizer(elo_system=mock_elo_system)
        prompt = synthesizer.synthesize_identity_prompt("claude")

        assert "### Your Calibration" in prompt
        # 0.72 > 0.7 so should be "well-calibrated"
        assert "well-calibrated" in prompt

    def test_synthesize_identity_prompt_with_positions(self, mock_position_ledger):
        """Test identity prompt includes position history section."""
        synthesizer = PersonaSynthesizer(position_ledger=mock_position_ledger)
        prompt = synthesizer.synthesize_identity_prompt("claude")

        assert "### Your Position History" in prompt
        assert "Positions taken" in prompt

    def test_synthesize_identity_prompt_with_relationships(self, mock_relationship_tracker):
        """Test identity prompt includes relationships section."""
        synthesizer = PersonaSynthesizer(relationship_tracker=mock_relationship_tracker)
        prompt = synthesizer.synthesize_identity_prompt("claude", opponent_names=["gemini"])

        assert "### Your Relationships" in prompt

    def test_synthesize_identity_prompt_custom_sections(self, mock_elo_system):
        """Test identity prompt with custom section selection."""
        synthesizer = PersonaSynthesizer(elo_system=mock_elo_system)
        prompt = synthesizer.synthesize_identity_prompt("claude", include_sections=["performance"])

        assert "### Your Track Record" in prompt
        assert "### Your Calibration" not in prompt

    def test_get_opponent_briefing_no_tracker(self):
        """Test opponent briefing with no tracker returns empty."""
        synthesizer = PersonaSynthesizer()
        briefing = synthesizer.get_opponent_briefing("claude", "gemini")

        assert briefing == ""

    def test_get_opponent_briefing_no_history(self):
        """Test opponent briefing when no debate history."""
        # Create a fresh mock to avoid side_effect from fixture
        tracker = MagicMock(spec=RelationshipTracker)
        tracker.get_relationship.return_value = AgentRelationship(
            agent_a="claude", agent_b="newbie", debate_count=0
        )
        synthesizer = PersonaSynthesizer(relationship_tracker=tracker)
        briefing = synthesizer.get_opponent_briefing("claude", "newbie")

        assert "have not debated" in briefing

    def test_get_opponent_briefing_with_history(self, mock_relationship_tracker):
        """Test opponent briefing with debate history."""
        synthesizer = PersonaSynthesizer(relationship_tracker=mock_relationship_tracker)
        briefing = synthesizer.get_opponent_briefing("claude", "gemini")

        assert "### Briefing: gemini" in briefing
        assert "Previous debates" in briefing


# ============================================================================
# SignificantMoment Tests
# ============================================================================


class TestSignificantMoment:
    """Tests for the SignificantMoment dataclass."""

    def test_create_basic_moment(self):
        """Test creating a basic significant moment."""
        moment = SignificantMoment(
            id="test123",
            moment_type="upset_victory",
            agent_name="claude",
            description="Claude won against higher-rated opponent",
            significance_score=0.8,
        )

        assert moment.id == "test123"
        assert moment.moment_type == "upset_victory"
        assert moment.agent_name == "claude"
        assert moment.significance_score == 0.8

    def test_moment_defaults(self):
        """Test moment default values."""
        moment = SignificantMoment(
            id="test",
            moment_type="position_reversal",
            agent_name="agent",
            description="desc",
            significance_score=0.5,
        )

        assert moment.debate_id is None
        assert moment.other_agents == []
        assert moment.metadata == {}
        assert moment.created_at is not None

    def test_moment_to_dict(self):
        """Test moment serialization to dictionary."""
        moment = SignificantMoment(
            id="test123",
            moment_type="calibration_vindication",
            agent_name="claude",
            description="Prediction was correct",
            significance_score=0.9,
            debate_id="debate456",
            other_agents=["gemini"],
            metadata={"domain": "security"},
        )

        d = moment.to_dict()

        assert d["id"] == "test123"
        assert d["moment_type"] == "calibration_vindication"
        assert d["agent_name"] == "claude"
        assert d["description"] == "Prediction was correct"
        assert d["significance_score"] == 0.9
        assert d["debate_id"] == "debate456"
        assert d["other_agents"] == ["gemini"]
        assert d["metadata"]["domain"] == "security"
        assert "created_at" in d

    def test_all_moment_types(self):
        """Test that all moment types can be created."""
        moment_types = [
            "upset_victory",
            "position_reversal",
            "calibration_vindication",
            "alliance_shift",
            "consensus_breakthrough",
            "streak_achievement",
            "domain_mastery",
        ]

        for mt in moment_types:
            moment = SignificantMoment(
                id=f"test_{mt}",
                moment_type=mt,
                agent_name="agent",
                description=f"Test {mt}",
                significance_score=0.5,
            )
            assert moment.moment_type == mt


# ============================================================================
# MomentDetector Tests
# ============================================================================


class TestMomentDetector:
    """Tests for the MomentDetector class."""

    def test_init_with_no_components(self):
        """Test initializing detector with no components."""
        detector = MomentDetector()

        assert detector.elo_system is None
        assert detector.position_ledger is None
        assert detector.relationship_tracker is None
        assert detector._max_moments_per_agent == 100

    def test_init_with_custom_max_moments(self):
        """Test initializing detector with custom max moments."""
        detector = MomentDetector(max_moments_per_agent=50)

        assert detector._max_moments_per_agent == 50

    def test_detect_upset_victory_no_elo(self):
        """Test upset detection returns None without ELO system."""
        detector = MomentDetector()
        result = detector.detect_upset_victory("claude", "gemini", "debate1")

        assert result is None

    def test_detect_upset_victory_no_upset(self, mock_elo_system):
        """Test no upset detected when winner has higher ELO."""

        # Winner has higher ELO than loser - not an upset
        @dataclass
        class MockRatingWinner:
            elo: float = 1700.0
            wins: int = 20
            losses: int = 5
            draws: int = 0
            domain_elos: dict = None

        @dataclass
        class MockRatingLoser:
            elo: float = 1500.0
            wins: int = 10
            losses: int = 15
            draws: int = 0
            domain_elos: dict = None

        mock_elo_system.get_rating.side_effect = lambda x: (
            MockRatingWinner() if x == "winner" else MockRatingLoser()
        )

        detector = MomentDetector(elo_system=mock_elo_system)
        result = detector.detect_upset_victory("winner", "loser", "debate1")

        assert result is None

    def test_detect_upset_victory_significant(self, mock_elo_system):
        """Test detecting a significant upset victory."""

        @dataclass
        class MockRatingUnderdog:
            elo: float = 1400.0
            wins: int = 5
            losses: int = 10
            draws: int = 0
            domain_elos: dict = None

        @dataclass
        class MockRatingFavorite:
            elo: float = 1650.0
            wins: int = 25
            losses: int = 5
            draws: int = 0
            domain_elos: dict = None

        mock_elo_system.get_rating.side_effect = lambda x: (
            MockRatingUnderdog() if x == "underdog" else MockRatingFavorite()
        )

        detector = MomentDetector(elo_system=mock_elo_system)
        result = detector.detect_upset_victory("underdog", "favorite", "debate1")

        assert result is not None
        assert result.moment_type == "upset_victory"
        assert result.agent_name == "underdog"
        assert "underdog" in result.description
        assert "favorite" in result.description
        assert result.significance_score > 0.5  # 250 ELO diff / 300

    def test_detect_position_reversal_not_reversed(self):
        """Test no moment when position not reversed."""
        position = Position.create(
            agent_name="claude",
            claim="Testing is important",
            confidence=0.9,
            debate_id="debate1",
            round_num=1,
        )
        position.reversed = False

        detector = MomentDetector()
        result = detector.detect_position_reversal("claude", position, position, "debate2")

        assert result is None

    def test_detect_position_reversal_significant(self):
        """Test detecting a significant position reversal."""
        original = Position.create(
            agent_name="claude",
            claim="Microservices are always better than monoliths",
            confidence=0.85,
            debate_id="debate1",
            round_num=1,
        )
        original.reversed = True
        original.outcome = "incorrect"

        new = Position.create(
            agent_name="claude",
            claim="Monoliths can be better in some cases",
            confidence=0.75,
            debate_id="debate2",
            round_num=1,
        )

        detector = MomentDetector()
        result = detector.detect_position_reversal("claude", original, new, "debate2")

        assert result is not None
        assert result.moment_type == "position_reversal"
        assert "reversed" in result.description.lower()
        # 0.85 * 0.8 + 0.2 (incorrect bonus) = 0.88
        assert result.significance_score > 0.8

    def test_detect_calibration_vindication_low_confidence(self):
        """Test no vindication for low confidence predictions."""
        detector = MomentDetector()
        result = detector.detect_calibration_vindication(
            "claude",
            prediction_confidence=0.7,
            was_correct=True,
            domain="security",
            debate_id="debate1",
        )

        assert result is None

    def test_detect_calibration_vindication_incorrect(self):
        """Test no vindication for incorrect predictions."""
        detector = MomentDetector()
        result = detector.detect_calibration_vindication(
            "claude",
            prediction_confidence=0.95,
            was_correct=False,
            domain="security",
            debate_id="debate1",
        )

        assert result is None

    def test_detect_calibration_vindication_significant(self):
        """Test detecting a significant calibration vindication."""
        detector = MomentDetector()
        result = detector.detect_calibration_vindication(
            "claude",
            prediction_confidence=0.92,
            was_correct=True,
            domain="security",
            debate_id="debate1",
        )

        assert result is not None
        assert result.moment_type == "calibration_vindication"
        assert "security" in result.description
        assert "92%" in result.description
        # (0.92 - 0.5) * 2 = 0.84
        assert result.significance_score > 0.8

    def test_detect_streak_achievement_short_streak(self):
        """Test no achievement for short streak."""
        detector = MomentDetector()
        result = detector.detect_streak_achievement(
            "claude", streak_type="win", streak_length=3, debate_id="debate1"
        )

        assert result is None

    def test_detect_streak_achievement_win_streak(self):
        """Test detecting a win streak achievement."""
        detector = MomentDetector()
        result = detector.detect_streak_achievement(
            "claude", streak_type="win", streak_length=7, debate_id="debate1"
        )

        assert result is not None
        assert result.moment_type == "streak_achievement"
        assert "7-debate winning streak" in result.description
        assert result.significance_score == 0.7  # 7/10

    def test_detect_streak_achievement_loss_streak(self):
        """Test detecting a loss streak."""
        detector = MomentDetector()
        result = detector.detect_streak_achievement(
            "claude", streak_type="loss", streak_length=6, debate_id="debate1"
        )

        assert result is not None
        assert "losing streak" in result.description

    def test_detect_domain_mastery_not_first(self):
        """Test no mastery when not rank 1."""
        detector = MomentDetector()
        result = detector.detect_domain_mastery("claude", "security", rank=2, elo=1750)

        assert result is None

    def test_detect_domain_mastery_first_place(self):
        """Test detecting domain mastery at rank 1."""
        detector = MomentDetector()
        result = detector.detect_domain_mastery("claude", "security", rank=1, elo=1800)

        assert result is not None
        assert result.moment_type == "domain_mastery"
        assert "#1" in result.description
        assert "security" in result.description
        assert result.significance_score == 0.9

    def test_detect_consensus_breakthrough_insufficient_agents(self):
        """Test no breakthrough with only one agent."""
        detector = MomentDetector()
        result = detector.detect_consensus_breakthrough(
            agents=["claude"],
            topic="AI ethics",
            confidence=0.9,
            debate_id="debate1",
        )

        assert result is None

    def test_detect_consensus_breakthrough_low_confidence(self):
        """Test no breakthrough with low confidence."""
        detector = MomentDetector()
        result = detector.detect_consensus_breakthrough(
            agents=["claude", "gemini"],
            topic="AI ethics",
            confidence=0.5,
            debate_id="debate1",
        )

        assert result is None

    def test_detect_consensus_breakthrough_significant(self, mock_relationship_tracker):
        """Test detecting a significant consensus breakthrough."""
        detector = MomentDetector(relationship_tracker=mock_relationship_tracker)
        result = detector.detect_consensus_breakthrough(
            agents=["claude", "gemini", "gpt4"],
            topic="AI safety guidelines should prioritize human oversight",
            confidence=0.85,
            debate_id="debate1",
        )

        assert result is not None
        assert result.moment_type == "consensus_breakthrough"
        assert "Consensus" in result.description
        assert "85%" in result.description
        assert result.other_agents == ["gemini", "gpt4"]

    def test_record_moment(self):
        """Test recording a moment."""
        detector = MomentDetector()
        moment = SignificantMoment(
            id="test1",
            moment_type="upset_victory",
            agent_name="claude",
            description="Test upset",
            significance_score=0.8,
            other_agents=["gemini"],
        )

        detector.record_moment(moment)

        # Should be cached for both agents
        assert len(detector._moment_cache.get("claude", [])) == 1
        assert len(detector._moment_cache.get("gemini", [])) == 1

    def test_trim_moments(self):
        """Test that moments are trimmed to max size."""
        detector = MomentDetector(max_moments_per_agent=3)

        # Add more than max moments
        for i in range(5):
            moment = SignificantMoment(
                id=f"test{i}",
                moment_type="upset_victory",
                agent_name="claude",
                description=f"Test moment {i}",
                significance_score=i * 0.2,  # Different significance
            )
            detector.record_moment(moment)

        # Should be trimmed to 3
        assert len(detector._moment_cache["claude"]) == 3

        # Should keep most significant
        scores = [m.significance_score for m in detector._moment_cache["claude"]]
        assert max(scores) == 0.8  # Most significant kept

    def test_get_agent_moments_empty(self):
        """Test getting moments when none recorded."""
        detector = MomentDetector()
        moments = detector.get_agent_moments("claude")

        assert moments == []

    def test_get_agent_moments_filtered_by_type(self):
        """Test getting moments filtered by type."""
        detector = MomentDetector()

        # Add moments of different types
        detector.record_moment(
            SignificantMoment(
                id="1",
                moment_type="upset_victory",
                agent_name="claude",
                description="Upset",
                significance_score=0.8,
            )
        )
        detector.record_moment(
            SignificantMoment(
                id="2",
                moment_type="position_reversal",
                agent_name="claude",
                description="Reversal",
                significance_score=0.6,
            )
        )

        upsets = detector.get_agent_moments("claude", moment_types=["upset_victory"])
        assert len(upsets) == 1
        assert upsets[0].moment_type == "upset_victory"

    def test_format_moment_narrative(self):
        """Test formatting a moment as narrative."""
        detector = MomentDetector()

        # Test different significance levels
        low = SignificantMoment(
            id="1",
            moment_type="upset_victory",
            agent_name="claude",
            description="Low sig event",
            significance_score=0.2,
        )
        assert "Notable" in detector.format_moment_narrative(low)

        high = SignificantMoment(
            id="2",
            moment_type="upset_victory",
            agent_name="claude",
            description="High sig event",
            significance_score=0.85,
        )
        assert "Defining" in detector.format_moment_narrative(high)

    def test_get_narrative_summary_empty(self):
        """Test narrative summary when no moments."""
        detector = MomentDetector()
        summary = detector.get_narrative_summary("claude")

        assert "not yet established" in summary

    def test_get_narrative_summary_with_moments(self):
        """Test narrative summary with moments."""
        detector = MomentDetector()

        detector.record_moment(
            SignificantMoment(
                id="1",
                moment_type="domain_mastery",
                agent_name="claude",
                description="Became #1 in security",
                significance_score=0.9,
            )
        )

        summary = detector.get_narrative_summary("claude")

        assert "### claude's Defining Moments" in summary
        assert "#1 in security" in summary


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in grounded module."""

    def test_persona_manager_error_handling(self):
        """Test graceful handling of PersonaManager errors."""
        manager = MagicMock(spec=PersonaManager)
        manager.get_persona.side_effect = KeyError("Agent not found")

        synthesizer = PersonaSynthesizer(persona_manager=manager)
        persona = synthesizer.get_grounded_persona("missing_agent")

        # Should still return a persona, just without base_persona
        assert persona.agent_name == "missing_agent"
        assert persona.base_persona is None

    def test_elo_system_error_handling(self):
        """Test graceful handling of ELO system errors."""
        elo = MagicMock()
        elo.get_rating.side_effect = ValueError("Invalid agent")

        synthesizer = PersonaSynthesizer(elo_system=elo)
        persona = synthesizer.get_grounded_persona("invalid_agent")

        # Should use defaults
        assert persona.elo == 1500.0
        assert persona.games_played == 0

    def test_position_ledger_error_handling(self):
        """Test graceful handling of PositionLedger errors."""
        ledger = MagicMock(spec=PositionLedger)
        ledger.get_position_stats.side_effect = TypeError("Invalid data")

        synthesizer = PersonaSynthesizer(position_ledger=ledger)
        persona = synthesizer.get_grounded_persona("error_agent")

        # Should use defaults
        assert persona.positions_taken == 0

    def test_relationship_tracker_error_handling(self):
        """Test graceful handling of RelationshipTracker errors."""
        tracker = MagicMock(spec=RelationshipTracker)
        tracker.get_rivals.side_effect = AttributeError("No data")
        tracker.get_allies.side_effect = AttributeError("No data")
        tracker.get_influence_network.side_effect = AttributeError("No data")

        synthesizer = PersonaSynthesizer(relationship_tracker=tracker)
        persona = synthesizer.get_grounded_persona("error_agent")

        # Should use defaults
        assert persona.rivals == []
        assert persona.allies == []

    def test_moment_detector_elo_error(self):
        """Test MomentDetector handles ELO errors gracefully."""
        elo = MagicMock()
        elo.get_rating.side_effect = KeyError("Agent not found")

        detector = MomentDetector(elo_system=elo)
        result = detector.detect_upset_victory("agent1", "agent2", "debate1")

        assert result is None

    def test_consensus_breakthrough_relationship_error(self):
        """Test consensus breakthrough handles relationship errors."""
        tracker = MagicMock(spec=RelationshipTracker)
        tracker.get_relationship.side_effect = ValueError("Error")

        detector = MomentDetector(relationship_tracker=tracker)
        result = detector.detect_consensus_breakthrough(
            agents=["claude", "gemini"],
            topic="test topic",
            confidence=0.9,
            debate_id="debate1",
        )

        # Should still detect breakthrough, just without rivalry bonus
        assert result is not None
        assert result.metadata["rivalry_score"] == 0.0


# ============================================================================
# Integration-style Tests
# ============================================================================


class TestIntegration:
    """Integration-style tests combining multiple components."""

    def test_full_synthesis_workflow(
        self,
        mock_persona_manager,
        mock_elo_system,
        mock_position_ledger,
        mock_relationship_tracker,
    ):
        """Test complete workflow of synthesizing identity prompt."""
        synthesizer = PersonaSynthesizer(
            persona_manager=mock_persona_manager,
            elo_system=mock_elo_system,
            position_ledger=mock_position_ledger,
            relationship_tracker=mock_relationship_tracker,
        )

        # Get persona
        persona = synthesizer.get_grounded_persona("claude")

        # Generate full identity prompt
        prompt = synthesizer.synthesize_identity_prompt("claude", opponent_names=["gemini"])

        # All sections should be present
        assert "## Your Identity: claude" in prompt
        assert "### Your Track Record" in prompt
        assert "### Your Calibration" in prompt
        assert "### Your Relationships" in prompt
        assert "### Your Position History" in prompt

    def test_moment_detection_and_narrative(self, mock_elo_system):
        """Test detecting moments and generating narrative."""

        @dataclass
        class MockUnderdog:
            elo: float = 1350.0
            wins: int = 5
            losses: int = 15
            draws: int = 0
            domain_elos: dict = None

        @dataclass
        class MockChampion:
            elo: float = 1800.0
            wins: int = 50
            losses: int = 10
            draws: int = 0
            domain_elos: dict = None

        mock_elo_system.get_rating.side_effect = lambda x: (
            MockUnderdog() if x == "underdog" else MockChampion()
        )

        detector = MomentDetector(elo_system=mock_elo_system)

        # Detect and record upset
        moment = detector.detect_upset_victory("underdog", "champion", "debate123")
        assert moment is not None
        detector.record_moment(moment)

        # Get narrative
        narrative = detector.get_narrative_summary("underdog")
        assert "Defining Moments" in narrative
        assert "underdog" in narrative
