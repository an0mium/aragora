"""
Tests for the grounded personas module.

Covers:
- Position data class
- CalibrationBucket data class
- DomainCalibration data class
- AgentRelationship data class
- GroundedPersona data class
- SignificantMoment data class
- PositionLedger with SQLite persistence
- RelationshipTracker with SQLite persistence
- PersonaSynthesizer prompt generation
- MomentDetector narrative detection
"""

import pytest
import tempfile
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from aragora.agents.grounded import (
    Position,
    CalibrationBucket,
    DomainCalibration,
    AgentRelationship,
    GroundedPersona,
    SignificantMoment,
    PositionLedger,
    RelationshipTracker,
    PersonaSynthesizer,
    MomentDetector,
)


# ============================================================================
# Position Tests
# ============================================================================


class TestPosition:
    """Tests for Position data class."""

    def test_create_basic(self):
        """Test creating a position with required fields."""
        pos = Position.create(
            agent_name="claude",
            claim="The sky is blue",
            confidence=0.9,
            debate_id="debate-123",
            round_num=1,
        )

        assert pos.agent_name == "claude"
        assert pos.claim == "The sky is blue"
        assert pos.confidence == 0.9
        assert pos.debate_id == "debate-123"
        assert pos.round_num == 1
        assert pos.outcome == "pending"
        assert pos.reversed is False
        assert len(pos.id) == 8  # UUID prefix

    def test_create_with_domain(self):
        """Test creating a position with domain."""
        pos = Position.create(
            agent_name="gemini",
            claim="Use caching for performance",
            confidence=0.85,
            debate_id="debate-456",
            round_num=2,
            domain="performance",
        )

        assert pos.domain == "performance"

    def test_confidence_clamping_high(self):
        """Test confidence is clamped to 1.0."""
        pos = Position.create(
            agent_name="agent",
            claim="test",
            confidence=1.5,
            debate_id="d",
            round_num=1,
        )

        assert pos.confidence == 1.0

    def test_confidence_clamping_low(self):
        """Test confidence is clamped to 0.0."""
        pos = Position.create(
            agent_name="agent",
            claim="test",
            confidence=-0.5,
            debate_id="d",
            round_num=1,
        )

        assert pos.confidence == 0.0

    def test_created_at_timestamp(self):
        """Test created_at is set automatically."""
        pos = Position.create(
            agent_name="agent",
            claim="test",
            confidence=0.5,
            debate_id="d",
            round_num=1,
        )

        # Should be a valid ISO timestamp
        datetime.fromisoformat(pos.created_at)


# ============================================================================
# CalibrationBucket Tests
# ============================================================================


class TestCalibrationBucket:
    """Tests for CalibrationBucket data class."""

    def test_accuracy_calculation(self):
        """Test accuracy property."""
        bucket = CalibrationBucket(
            bucket_start=0.8,
            bucket_end=0.9,
            predictions=10,
            correct=8,
        )

        assert bucket.accuracy == 0.8

    def test_accuracy_zero_predictions(self):
        """Test accuracy with no predictions."""
        bucket = CalibrationBucket(bucket_start=0.7, bucket_end=0.8)

        assert bucket.accuracy == 0.0

    def test_expected_accuracy(self):
        """Test expected accuracy is bucket midpoint."""
        bucket = CalibrationBucket(bucket_start=0.8, bucket_end=0.9)

        assert abs(bucket.expected_accuracy - 0.85) < 0.001

    def test_calibration_error(self):
        """Test calibration error calculation."""
        bucket = CalibrationBucket(
            bucket_start=0.8,
            bucket_end=0.9,
            predictions=10,
            correct=7,  # 70% accuracy, expected 85%
        )

        assert abs(bucket.calibration_error - 0.15) < 0.01

    def test_bucket_key(self):
        """Test bucket key formatting."""
        bucket = CalibrationBucket(bucket_start=0.8, bucket_end=0.9)

        assert bucket.bucket_key == "0.8-0.9"


# ============================================================================
# DomainCalibration Tests
# ============================================================================


class TestDomainCalibration:
    """Tests for DomainCalibration data class."""

    def test_calibration_score_no_predictions(self):
        """Test calibration score with no predictions."""
        dc = DomainCalibration(domain="security")

        assert dc.calibration_score == 0.5

    def test_calibration_score_with_data(self):
        """Test calibration score calculation."""
        dc = DomainCalibration(
            domain="performance",
            total_predictions=10,
            brier_sum=2.0,  # avg brier = 0.2
        )

        # 1 - 0.2 = 0.8
        assert dc.calibration_score == 0.8

    def test_accuracy_property(self):
        """Test accuracy calculation."""
        dc = DomainCalibration(
            domain="testing",
            total_predictions=20,
            total_correct=15,
        )

        assert dc.accuracy == 0.75


# ============================================================================
# AgentRelationship Tests
# ============================================================================


class TestAgentRelationship:
    """Tests for AgentRelationship data class."""

    def test_rivalry_score_few_debates(self):
        """Test rivalry score with too few debates."""
        rel = AgentRelationship(
            agent_a="claude",
            agent_b="gemini",
            debate_count=2,
        )

        assert rel.rivalry_score == 0.0

    def test_rivalry_score_calculation(self):
        """Test rivalry score with sufficient debates."""
        rel = AgentRelationship(
            agent_a="claude",
            agent_b="gemini",
            debate_count=20,
            agreement_count=4,  # 20% agreement = 80% disagreement
            a_wins_over_b=5,
            b_wins_over_a=5,  # Perfectly competitive
        )

        # High rivalry: high disagreement + competitive wins
        assert rel.rivalry_score > 0.5

    def test_alliance_score_few_debates(self):
        """Test alliance score with too few debates."""
        rel = AgentRelationship(
            agent_a="claude",
            agent_b="gemini",
            debate_count=2,
        )

        assert rel.alliance_score == 0.0

    def test_alliance_score_calculation(self):
        """Test alliance score with good agreement."""
        rel = AgentRelationship(
            agent_a="claude",
            agent_b="gemini",
            debate_count=10,
            agreement_count=8,  # 80% agreement
            critique_count_a_to_b=5,
            critique_count_b_to_a=5,
            critique_accepted_a_to_b=4,
            critique_accepted_b_to_a=4,  # 80% acceptance
        )

        # High alliance: high agreement + high critique acceptance
        assert rel.alliance_score > 0.6

    def test_influence_scores(self):
        """Test influence calculation."""
        rel = AgentRelationship(
            agent_a="claude",
            agent_b="gemini",
            debate_count=10,
            position_changes_a_after_b=3,
            position_changes_b_after_a=1,
        )

        assert rel.influence_b_on_a == 0.3  # B influences A
        assert rel.influence_a_on_b == 0.1  # A influences B

    def test_get_influence(self):
        """Test get_influence method."""
        rel = AgentRelationship(
            agent_a="claude",
            agent_b="gemini",
            debate_count=10,
            position_changes_a_after_b=2,
            position_changes_b_after_a=4,
        )

        assert rel.get_influence("claude") == 0.4  # claude influences gemini
        assert rel.get_influence("gemini") == 0.2  # gemini influences claude
        assert rel.get_influence("unknown") == 0.0


# ============================================================================
# GroundedPersona Tests
# ============================================================================


class TestGroundedPersona:
    """Tests for GroundedPersona data class."""

    def test_default_values(self):
        """Test default values."""
        persona = GroundedPersona(agent_name="claude")

        assert persona.agent_name == "claude"
        assert persona.elo == 1500.0
        assert persona.win_rate == 0.0
        assert persona.positions_taken == 0
        assert persona.overall_calibration == 0.5

    def test_reversal_rate_no_positions(self):
        """Test reversal rate with no positions."""
        persona = GroundedPersona(agent_name="claude")

        assert persona.reversal_rate == 0.0

    def test_reversal_rate_with_positions(self):
        """Test reversal rate calculation."""
        persona = GroundedPersona(
            agent_name="claude",
            positions_taken=10,
            reversals=2,
        )

        assert persona.reversal_rate == 0.2

    def test_position_accuracy_no_resolved(self):
        """Test position accuracy with no resolved positions."""
        persona = GroundedPersona(agent_name="claude")

        assert persona.position_accuracy == 0.0

    def test_position_accuracy_calculation(self):
        """Test position accuracy calculation."""
        persona = GroundedPersona(
            agent_name="claude",
            positions_correct=7,
            positions_incorrect=3,
        )

        assert persona.position_accuracy == 0.7


# ============================================================================
# SignificantMoment Tests
# ============================================================================


class TestSignificantMoment:
    """Tests for SignificantMoment data class."""

    def test_creation(self):
        """Test creating a significant moment."""
        moment = SignificantMoment(
            id="abc12345",
            moment_type="upset_victory",
            agent_name="claude",
            description="Claude defeated Gemini despite being 150 ELO lower",
            significance_score=0.75,
            debate_id="debate-123",
            other_agents=["gemini"],
        )

        assert moment.moment_type == "upset_victory"
        assert moment.significance_score == 0.75
        assert "gemini" in moment.other_agents

    def test_to_dict(self):
        """Test serialization to dict."""
        moment = SignificantMoment(
            id="xyz",
            moment_type="position_reversal",
            agent_name="gemini",
            description="Position changed",
            significance_score=0.5,
        )

        d = moment.to_dict()

        assert d["id"] == "xyz"
        assert d["moment_type"] == "position_reversal"
        assert d["agent_name"] == "gemini"
        assert "created_at" in d


# ============================================================================
# PositionLedger Tests
# ============================================================================


class TestPositionLedger:
    """Tests for PositionLedger with SQLite."""

    @pytest.fixture
    def ledger(self, tmp_path):
        """Create a temporary position ledger."""
        db_path = str(tmp_path / "test_positions.db")
        return PositionLedger(db_path=db_path)

    def test_record_position(self, ledger):
        """Test recording a position."""
        pos_id = ledger.record_position(
            agent_name="claude",
            claim="Testing is important",
            confidence=0.9,
            debate_id="debate-1",
            round_num=1,
            domain="testing",
        )

        assert len(pos_id) == 8

    def test_get_agent_positions(self, ledger):
        """Test retrieving positions for an agent."""
        ledger.record_position(
            agent_name="claude",
            claim="Claim 1",
            confidence=0.8,
            debate_id="d1",
            round_num=1,
        )
        ledger.record_position(
            agent_name="claude",
            claim="Claim 2",
            confidence=0.7,
            debate_id="d2",
            round_num=1,
        )

        positions = ledger.get_agent_positions("claude")

        assert len(positions) == 2
        assert all(p.agent_name == "claude" for p in positions)

    def test_resolve_position(self, ledger):
        """Test resolving a position outcome."""
        pos_id = ledger.record_position(
            agent_name="gemini",
            claim="This will be correct",
            confidence=0.85,
            debate_id="d1",
            round_num=1,
        )

        ledger.resolve_position(pos_id, "correct")

        positions = ledger.get_agent_positions("gemini")
        assert positions[0].outcome == "correct"
        assert positions[0].resolved_at is not None

    def test_record_reversal(self, ledger):
        """Test recording a position reversal."""
        pos_id = ledger.record_position(
            agent_name="claude",
            claim="Original position",
            confidence=0.9,
            debate_id="d1",
            round_num=1,
        )

        ledger.record_reversal("claude", pos_id, "d2")

        positions = ledger.get_agent_positions("claude")
        assert positions[0].reversed is True
        assert positions[0].reversal_debate_id == "d2"

    def test_get_position_stats(self, ledger):
        """Test getting aggregate position statistics."""
        # Record some positions
        pos1 = ledger.record_position("claude", "Claim 1", 0.9, "d1", 1)
        pos2 = ledger.record_position("claude", "Claim 2", 0.8, "d2", 1)
        pos3 = ledger.record_position("claude", "Claim 3", 0.7, "d3", 1)

        # Resolve them
        ledger.resolve_position(pos1, "correct")
        ledger.resolve_position(pos2, "incorrect")
        # pos3 stays pending

        stats = ledger.get_position_stats("claude")

        assert stats["total"] == 3
        assert stats["correct"] == 1
        assert stats["incorrect"] == 1
        assert stats["pending"] == 1

    def test_get_positions_for_debate(self, ledger):
        """Test getting all positions from a debate."""
        ledger.record_position("claude", "C1", 0.9, "debate-X", 1)
        ledger.record_position("gemini", "C2", 0.8, "debate-X", 1)
        ledger.record_position("claude", "C3", 0.7, "other-debate", 1)

        positions = ledger.get_positions_for_debate("debate-X")

        assert len(positions) == 2
        assert all(p.debate_id == "debate-X" for p in positions)

    def test_detect_domain_security(self, ledger):
        """Test domain detection for security content."""
        domain = ledger.detect_domain("We should add authentication and protect against XSS")
        assert domain == "security"

    def test_detect_domain_performance(self, ledger):
        """Test domain detection for performance content."""
        domain = ledger.detect_domain("Use caching to improve latency and throughput")
        assert domain == "performance"

    def test_detect_domain_none(self, ledger):
        """Test domain detection with no keywords."""
        domain = ledger.detect_domain("This is a generic statement")
        assert domain is None


# ============================================================================
# RelationshipTracker Tests
# ============================================================================


class TestRelationshipTracker:
    """Tests for RelationshipTracker with SQLite."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a temporary relationship tracker."""
        db_path = str(tmp_path / "test_elo.db")
        return RelationshipTracker(elo_db_path=db_path)

    def test_canonical_pair(self, tracker):
        """Test canonical pair ordering."""
        assert tracker._canonical_pair("claude", "gemini") == ("claude", "gemini")
        assert tracker._canonical_pair("gemini", "claude") == ("claude", "gemini")

    def test_update_from_debate(self, tracker):
        """Test updating relationships from a debate."""
        tracker.update_from_debate(
            debate_id="debate-1",
            participants=["claude", "gemini", "gpt"],
            winner="claude",
            votes={"claude": "A", "gemini": "A", "gpt": "B"},
            critiques=[
                {"agent": "claude", "target": "gemini"},
                {"agent": "gemini", "target": "claude"},
            ],
        )

        # Check claude-gemini relationship
        rel = tracker.get_relationship("claude", "gemini")
        assert rel.debate_count == 1
        assert rel.agreement_count == 1  # Both voted A

    def test_get_relationship_nonexistent(self, tracker):
        """Test getting nonexistent relationship returns empty."""
        rel = tracker.get_relationship("agent1", "agent2")

        assert rel.debate_count == 0
        assert rel.agreement_count == 0

    def test_get_all_relationships(self, tracker):
        """Test getting all relationships for an agent."""
        # Create some relationships
        tracker.update_from_debate("d1", ["claude", "gemini"], None, {}, [])
        tracker.update_from_debate("d2", ["claude", "gpt"], None, {}, [])
        tracker.update_from_debate("d3", ["gemini", "gpt"], None, {}, [])

        rels = tracker.get_all_relationships("claude")

        assert len(rels) == 2  # claude-gemini, claude-gpt

    def test_get_rivals(self, tracker):
        """Test getting top rivals."""
        # Create enough debates for rivalry to register
        for i in range(5):
            tracker.update_from_debate(
                f"d{i}",
                ["claude", "gemini"],
                "claude" if i % 2 == 0 else "gemini",  # Alternating wins
                {"claude": "A", "gemini": "B"},  # Disagreement
                [],
            )

        rivals = tracker.get_rivals("claude")

        # gemini should be a rival (high disagreement, competitive)
        if rivals:
            assert rivals[0][0] == "gemini"

    def test_get_allies(self, tracker):
        """Test getting top allies."""
        # Create debates with high agreement
        for i in range(5):
            tracker.update_from_debate(
                f"d{i}",
                ["claude", "gemini"],
                None,
                {"claude": "A", "gemini": "A"},  # Agreement
                [],
            )

        allies = tracker.get_allies("claude")

        # gemini should be an ally
        if allies:
            assert allies[0][0] == "gemini"

    def test_get_influence_network(self, tracker):
        """Test getting influence network."""
        network = tracker.get_influence_network("claude")

        assert "influences" in network
        assert "influenced_by" in network


# ============================================================================
# PersonaSynthesizer Tests
# ============================================================================


class TestPersonaSynthesizer:
    """Tests for PersonaSynthesizer prompt generation."""

    def test_get_grounded_persona_minimal(self):
        """Test getting persona with no data sources."""
        synthesizer = PersonaSynthesizer()
        persona = synthesizer.get_grounded_persona("claude")

        assert persona.agent_name == "claude"
        assert persona.elo == 1500.0  # Default

    def test_get_grounded_persona_with_position_ledger(self, tmp_path):
        """Test persona with position ledger data."""
        # Create ledger and add positions
        ledger = PositionLedger(db_path=str(tmp_path / "pos.db"))
        pos_id = ledger.record_position("claude", "Test", 0.9, "d1", 1)
        ledger.resolve_position(pos_id, "correct")

        synthesizer = PersonaSynthesizer(position_ledger=ledger)
        persona = synthesizer.get_grounded_persona("claude")

        assert persona.positions_taken == 1
        assert persona.positions_correct == 1

    def test_synthesize_identity_prompt(self):
        """Test identity prompt generation."""
        synthesizer = PersonaSynthesizer()
        prompt = synthesizer.synthesize_identity_prompt("claude")

        assert "claude" in prompt
        assert "Identity" in prompt

    def test_synthesize_identity_prompt_with_sections(self, tmp_path):
        """Test identity prompt with specific sections."""
        ledger = PositionLedger(db_path=str(tmp_path / "pos.db"))
        ledger.record_position("claude", "Test", 0.9, "d1", 1)

        synthesizer = PersonaSynthesizer(position_ledger=ledger)
        prompt = synthesizer.synthesize_identity_prompt(
            "claude",
            include_sections=["positions"],
        )

        assert "Position History" in prompt

    def test_format_performance_section(self):
        """Test performance section formatting."""
        synthesizer = PersonaSynthesizer()
        persona = GroundedPersona(
            agent_name="claude",
            elo=1650,
            win_rate=0.65,
            games_played=20,
        )

        section = synthesizer._format_performance_section(persona)

        assert "1650" in section
        assert "65%" in section

    def test_format_calibration_section(self):
        """Test calibration section formatting."""
        synthesizer = PersonaSynthesizer()
        persona = GroundedPersona(
            agent_name="claude",
            overall_calibration=0.75,
            positions_taken=10,
            positions_correct=8,
            positions_incorrect=2,
        )

        section = synthesizer._format_calibration_section(persona)

        assert "calibrated" in section.lower()

    def test_get_opponent_briefing_no_history(self, tmp_path):
        """Test opponent briefing with no debate history."""
        tracker = RelationshipTracker(elo_db_path=str(tmp_path / "elo.db"))
        synthesizer = PersonaSynthesizer(relationship_tracker=tracker)

        briefing = synthesizer.get_opponent_briefing("claude", "gemini")

        assert "not debated" in briefing.lower()

    def test_get_opponent_briefing_with_history(self, tmp_path):
        """Test opponent briefing with debate history."""
        tracker = RelationshipTracker(elo_db_path=str(tmp_path / "elo.db"))
        for i in range(3):
            tracker.update_from_debate(
                f"d{i}",
                ["claude", "gemini"],
                "claude",
                {},
                [],
            )

        synthesizer = PersonaSynthesizer(relationship_tracker=tracker)
        briefing = synthesizer.get_opponent_briefing("claude", "gemini")

        assert "gemini" in briefing
        assert "3" in briefing  # debate count


# ============================================================================
# MomentDetector Tests
# ============================================================================


class TestMomentDetector:
    """Tests for MomentDetector narrative detection."""

    def test_detect_upset_victory_no_elo(self):
        """Test upset detection without ELO system."""
        detector = MomentDetector()
        moment = detector.detect_upset_victory("claude", "gemini", "d1")

        assert moment is None

    def test_detect_upset_victory_with_elo(self):
        """Test upset detection with ELO system."""
        # Mock ELO system
        elo_system = Mock()
        winner_rating = Mock(elo=1400)
        loser_rating = Mock(elo=1600)
        elo_system.get_rating.side_effect = lambda x: (
            winner_rating if x == "underdog" else loser_rating
        )

        detector = MomentDetector(elo_system=elo_system)
        moment = detector.detect_upset_victory("underdog", "favorite", "d1")

        assert moment is not None
        assert moment.moment_type == "upset_victory"
        assert moment.significance_score > 0.5

    def test_detect_upset_victory_no_upset(self):
        """Test no upset when winner had higher ELO."""
        elo_system = Mock()
        winner_rating = Mock(elo=1600)
        loser_rating = Mock(elo=1500)
        elo_system.get_rating.side_effect = lambda x: (
            winner_rating if x == "favorite" else loser_rating
        )

        detector = MomentDetector(elo_system=elo_system)
        moment = detector.detect_upset_victory("favorite", "underdog", "d1")

        assert moment is None

    def test_detect_position_reversal(self):
        """Test position reversal detection."""
        detector = MomentDetector()
        original = Position(
            id="orig",
            agent_name="claude",
            claim="Original claim",
            confidence=0.9,
            debate_id="d1",
            round_num=1,
            reversed=True,
            outcome="incorrect",
        )
        new = Position.create("claude", "New claim", 0.8, "d2", 1)

        moment = detector.detect_position_reversal("claude", original, new, "d2")

        assert moment is not None
        assert moment.moment_type == "position_reversal"

    def test_detect_position_reversal_not_reversed(self):
        """Test no detection when position not reversed."""
        detector = MomentDetector()
        original = Position(
            id="orig",
            agent_name="claude",
            claim="Original",
            confidence=0.9,
            debate_id="d1",
            round_num=1,
            reversed=False,  # Not reversed
        )

        moment = detector.detect_position_reversal("claude", original, None, "d2")

        assert moment is None

    def test_detect_calibration_vindication_high_confidence(self):
        """Test calibration vindication with high confidence."""
        detector = MomentDetector()
        moment = detector.detect_calibration_vindication(
            agent_name="claude",
            prediction_confidence=0.95,
            was_correct=True,
            domain="security",
            debate_id="d1",
        )

        assert moment is not None
        assert moment.moment_type == "calibration_vindication"

    def test_detect_calibration_vindication_low_confidence(self):
        """Test no vindication with low confidence."""
        detector = MomentDetector()
        moment = detector.detect_calibration_vindication(
            agent_name="claude",
            prediction_confidence=0.7,
            was_correct=True,
            domain="testing",
            debate_id="d1",
        )

        assert moment is None

    def test_detect_calibration_vindication_incorrect(self):
        """Test no vindication when prediction was wrong."""
        detector = MomentDetector()
        moment = detector.detect_calibration_vindication(
            agent_name="claude",
            prediction_confidence=0.95,
            was_correct=False,
            domain="security",
            debate_id="d1",
        )

        assert moment is None

    def test_detect_streak_achievement_win(self):
        """Test win streak detection."""
        detector = MomentDetector()
        moment = detector.detect_streak_achievement("claude", "win", 7, "d1")

        assert moment is not None
        assert moment.moment_type == "streak_achievement"
        assert "winning" in moment.description

    def test_detect_streak_achievement_short(self):
        """Test no streak for short streaks."""
        detector = MomentDetector()
        moment = detector.detect_streak_achievement("claude", "win", 3, "d1")

        assert moment is None

    def test_detect_domain_mastery(self):
        """Test domain mastery detection."""
        detector = MomentDetector()
        moment = detector.detect_domain_mastery("claude", "security", 1, 1750)

        assert moment is not None
        assert moment.moment_type == "domain_mastery"
        assert "security" in moment.description

    def test_detect_domain_mastery_not_first(self):
        """Test no mastery when not #1."""
        detector = MomentDetector()
        moment = detector.detect_domain_mastery("claude", "security", 2, 1700)

        assert moment is None

    def test_detect_consensus_breakthrough(self):
        """Test consensus breakthrough detection."""
        detector = MomentDetector()
        moment = detector.detect_consensus_breakthrough(
            agents=["claude", "gemini"],
            topic="Important decision",
            confidence=0.85,
            debate_id="d1",
        )

        assert moment is not None
        assert moment.moment_type == "consensus_breakthrough"

    def test_detect_consensus_breakthrough_low_confidence(self):
        """Test no breakthrough with low confidence."""
        detector = MomentDetector()
        moment = detector.detect_consensus_breakthrough(
            agents=["claude", "gemini"],
            topic="Topic",
            confidence=0.5,
            debate_id="d1",
        )

        assert moment is None

    def test_record_and_get_moments(self):
        """Test recording and retrieving moments."""
        detector = MomentDetector()

        moment = SignificantMoment(
            id="m1",
            moment_type="upset_victory",
            agent_name="claude",
            description="Test moment",
            significance_score=0.8,
            other_agents=["gemini"],
        )

        detector.record_moment(moment)

        # Should be retrievable for both agents
        claude_moments = detector.get_agent_moments("claude")
        gemini_moments = detector.get_agent_moments("gemini")

        assert len(claude_moments) == 1
        assert len(gemini_moments) == 1

    def test_format_moment_narrative(self):
        """Test moment narrative formatting."""
        detector = MomentDetector()

        moment = SignificantMoment(
            id="m1",
            moment_type="domain_mastery",
            agent_name="claude",
            description="Claude becomes #1 in security",
            significance_score=0.9,
        )

        narrative = detector.format_moment_narrative(moment)

        assert "Defining" in narrative  # High significance
        assert "Claude" in narrative

    def test_get_narrative_summary_no_moments(self):
        """Test narrative summary with no moments."""
        detector = MomentDetector()
        summary = detector.get_narrative_summary("new_agent")

        assert "not yet established" in summary

    def test_get_narrative_summary_with_moments(self):
        """Test narrative summary with moments."""
        detector = MomentDetector()

        moment = SignificantMoment(
            id="m1",
            moment_type="upset_victory",
            agent_name="claude",
            description="Great victory",
            significance_score=0.7,
        )
        detector.record_moment(moment)

        summary = detector.get_narrative_summary("claude")

        assert "Defining Moments" in summary
        assert "Great victory" in summary
