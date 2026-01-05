"""Tests for MomentDetector - significant narrative moment detection."""

import pytest
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

from aragora.agents.grounded import (
    MomentDetector,
    SignificantMoment,
    Position,
)


@dataclass
class MockELORating:
    """Mock ELO rating for testing."""
    elo: float = 1500.0


class TestMomentDetectorUpsetVictory:
    """Tests for upset victory detection."""

    def test_detects_significant_upset(self):
        """Upset detected when winner was 100+ ELO below loser."""
        elo_system = Mock()
        elo_system.get_rating.side_effect = lambda name: {
            "underdog": MockELORating(elo=1400),
            "favorite": MockELORating(elo=1600),
        }[name]

        detector = MomentDetector(elo_system=elo_system)
        moment = detector.detect_upset_victory(
            winner="underdog",
            loser="favorite",
            debate_id="debate-123",
        )

        assert moment is not None
        assert moment.moment_type == "upset_victory"
        assert moment.agent_name == "underdog"
        assert "underdog" in moment.description
        assert "favorite" in moment.description
        assert "200" in moment.description  # ELO difference
        assert moment.significance_score > 0.5
        assert moment.debate_id == "debate-123"

    def test_no_upset_when_favorite_wins(self):
        """No upset when higher-rated agent wins."""
        elo_system = Mock()
        elo_system.get_rating.side_effect = lambda name: {
            "favorite": MockELORating(elo=1600),
            "underdog": MockELORating(elo=1400),
        }[name]

        detector = MomentDetector(elo_system=elo_system)
        moment = detector.detect_upset_victory(
            winner="favorite",
            loser="underdog",
            debate_id="debate-123",
        )

        assert moment is None

    def test_no_upset_small_elo_difference(self):
        """No upset when ELO difference is less than 100."""
        elo_system = Mock()
        elo_system.get_rating.side_effect = lambda name: {
            "player_a": MockELORating(elo=1450),
            "player_b": MockELORating(elo=1500),
        }[name]

        detector = MomentDetector(elo_system=elo_system)
        moment = detector.detect_upset_victory(
            winner="player_a",
            loser="player_b",
            debate_id="debate-123",
        )

        assert moment is None

    def test_significance_scales_with_elo_diff(self):
        """Larger ELO difference = higher significance."""
        elo_system = Mock()

        # Test with 150 ELO diff
        elo_system.get_rating.side_effect = lambda name: {
            "underdog": MockELORating(elo=1350),
            "favorite": MockELORating(elo=1500),
        }[name]

        detector = MomentDetector(elo_system=elo_system)
        moment_150 = detector.detect_upset_victory("underdog", "favorite", "d1")

        # Test with 300 ELO diff (max significance)
        elo_system.get_rating.side_effect = lambda name: {
            "underdog": MockELORating(elo=1200),
            "favorite": MockELORating(elo=1500),
        }[name]

        moment_300 = detector.detect_upset_victory("underdog", "favorite", "d2")

        assert moment_150.significance_score < moment_300.significance_score
        assert moment_300.significance_score == 1.0  # Capped at 1.0

    def test_no_elo_system_returns_none(self):
        """Without ELO system, cannot detect upsets."""
        detector = MomentDetector(elo_system=None)
        moment = detector.detect_upset_victory("a", "b", "debate-123")
        assert moment is None


class TestMomentDetectorCalibrationVindication:
    """Tests for calibration vindication detection."""

    def test_detects_high_confidence_correct(self):
        """Vindication when high-confidence prediction is correct."""
        detector = MomentDetector()
        moment = detector.detect_calibration_vindication(
            agent_name="claude",
            prediction_confidence=0.90,
            was_correct=True,
            domain="ethics",
            debate_id="debate-456",
        )

        assert moment is not None
        assert moment.moment_type == "calibration_vindication"
        assert moment.agent_name == "claude"
        assert "90%" in moment.description
        assert "ethics" in moment.description
        assert moment.metadata["prediction_confidence"] == 0.90

    def test_no_vindication_below_85_confidence(self):
        """No vindication for predictions below 85% confidence."""
        detector = MomentDetector()
        moment = detector.detect_calibration_vindication(
            agent_name="claude",
            prediction_confidence=0.80,
            was_correct=True,
            domain="ethics",
            debate_id="debate-456",
        )

        assert moment is None

    def test_no_vindication_when_wrong(self):
        """No vindication when prediction was incorrect."""
        detector = MomentDetector()
        moment = detector.detect_calibration_vindication(
            agent_name="claude",
            prediction_confidence=0.95,
            was_correct=False,
            domain="ethics",
            debate_id="debate-456",
        )

        assert moment is None

    def test_significance_scales_with_confidence(self):
        """Higher confidence = higher significance."""
        detector = MomentDetector()

        moment_85 = detector.detect_calibration_vindication(
            "agent", 0.85, True, "domain", "d1"
        )
        moment_95 = detector.detect_calibration_vindication(
            "agent", 0.95, True, "domain", "d2"
        )

        assert moment_85.significance_score < moment_95.significance_score


class TestMomentDetectorStreakAchievement:
    """Tests for streak detection."""

    def test_detects_winning_streak(self):
        """Detect 5+ game winning streak."""
        detector = MomentDetector()
        moment = detector.detect_streak_achievement(
            agent_name="gemini",
            streak_type="win",
            streak_length=7,
            debate_id="debate-789",
        )

        assert moment is not None
        assert moment.moment_type == "streak_achievement"
        assert "7-debate winning streak" in moment.description
        assert moment.metadata["streak_type"] == "win"
        assert moment.metadata["streak_length"] == 7

    def test_detects_losing_streak(self):
        """Detect 5+ game losing streak."""
        detector = MomentDetector()
        moment = detector.detect_streak_achievement(
            agent_name="grok",
            streak_type="loss",
            streak_length=6,
            debate_id="debate-789",
        )

        assert moment is not None
        assert "6-debate losing streak" in moment.description

    def test_no_streak_below_5(self):
        """Streaks below 5 are not significant."""
        detector = MomentDetector()
        moment = detector.detect_streak_achievement(
            agent_name="agent",
            streak_type="win",
            streak_length=4,
            debate_id="debate-789",
        )

        assert moment is None

    def test_significance_caps_at_10(self):
        """Significance maxes out at streak of 10."""
        detector = MomentDetector()
        moment_10 = detector.detect_streak_achievement("a", "win", 10, "d1")
        moment_15 = detector.detect_streak_achievement("a", "win", 15, "d2")

        assert moment_10.significance_score == 1.0
        assert moment_15.significance_score == 1.0


class TestMomentDetectorDomainMastery:
    """Tests for domain mastery detection."""

    def test_detects_rank_1(self):
        """Detect becoming #1 in a domain."""
        detector = MomentDetector()
        moment = detector.detect_domain_mastery(
            agent_name="claude",
            domain="philosophy",
            rank=1,
            elo=1850.0,
        )

        assert moment is not None
        assert moment.moment_type == "domain_mastery"
        assert "#1" in moment.description
        assert "philosophy" in moment.description
        assert "1850" in moment.description
        assert moment.significance_score == 0.9

    def test_no_mastery_for_rank_2(self):
        """Only rank 1 counts as mastery."""
        detector = MomentDetector()
        moment = detector.detect_domain_mastery(
            agent_name="claude",
            domain="philosophy",
            rank=2,
            elo=1800.0,
        )

        assert moment is None


class TestMomentDetectorConsensusBreakthrough:
    """Tests for consensus breakthrough detection."""

    def test_detects_high_confidence_consensus(self):
        """Detect consensus with 70%+ confidence."""
        detector = MomentDetector()
        moment = detector.detect_consensus_breakthrough(
            agents=["claude", "gemini"],
            topic="AI safety requires international cooperation",
            confidence=0.85,
            debate_id="debate-101",
        )

        assert moment is not None
        assert moment.moment_type == "consensus_breakthrough"
        assert "Consensus reached" in moment.description
        assert "gemini" in moment.other_agents

    def test_no_breakthrough_low_confidence(self):
        """No breakthrough below 70% confidence."""
        detector = MomentDetector()
        moment = detector.detect_consensus_breakthrough(
            agents=["claude", "gemini"],
            topic="Some topic",
            confidence=0.65,
            debate_id="debate-101",
        )

        assert moment is None

    def test_no_breakthrough_single_agent(self):
        """Need at least 2 agents for consensus."""
        detector = MomentDetector()
        moment = detector.detect_consensus_breakthrough(
            agents=["claude"],
            topic="Some topic",
            confidence=0.90,
            debate_id="debate-101",
        )

        assert moment is None

    def test_rivalry_increases_significance(self):
        """Consensus between rivals is more significant."""
        relationship_tracker = Mock()
        mock_rel = Mock()
        mock_rel.rivalry_score = 0.8  # High rivalry
        relationship_tracker.get_relationship.return_value = mock_rel

        detector = MomentDetector(relationship_tracker=relationship_tracker)
        moment = detector.detect_consensus_breakthrough(
            agents=["claude", "gemini"],
            topic="AI ethics",
            confidence=0.75,
            debate_id="debate-102",
        )

        assert moment is not None
        assert moment.significance_score > 0.6  # Boosted by rivalry


class TestMomentDetectorPositionReversal:
    """Tests for position reversal detection."""

    def test_detects_reversal(self):
        """Detect when position is reversed."""
        detector = MomentDetector()

        original = Position.create(
            agent_name="claude",
            claim="X is always true",
            confidence=0.85,
            debate_id="d1",
            round_num=1,
        )
        original.reversed = True
        original.outcome = "incorrect"

        new = Position.create(
            agent_name="claude",
            claim="X is sometimes false",
            confidence=0.70,
            debate_id="d2",
            round_num=1,
        )

        moment = detector.detect_position_reversal(
            agent_name="claude",
            original_position=original,
            new_position=new,
            debate_id="d2",
        )

        assert moment is not None
        assert moment.moment_type == "position_reversal"
        assert "reversed" in moment.description.lower()
        assert "85%" in moment.description

    def test_no_reversal_if_not_reversed(self):
        """No moment if position wasn't actually reversed."""
        detector = MomentDetector()

        original = Position.create(
            agent_name="claude",
            claim="X is true",
            confidence=0.85,
            debate_id="d1",
            round_num=1,
        )
        original.reversed = False  # Not reversed

        new = Position.create(
            agent_name="claude",
            claim="X is sometimes true",
            confidence=0.70,
            debate_id="d2",
            round_num=1,
        )

        moment = detector.detect_position_reversal("claude", original, new, "d2")
        assert moment is None


class TestMomentDetectorRecordAndRetrieve:
    """Tests for recording and retrieving moments."""

    def test_record_and_get_moments(self):
        """Record moments and retrieve them."""
        detector = MomentDetector()

        moment1 = SignificantMoment(
            id="m1",
            moment_type="upset_victory",
            agent_name="claude",
            description="Claude upset win",
            significance_score=0.8,
        )
        moment2 = SignificantMoment(
            id="m2",
            moment_type="streak_achievement",
            agent_name="claude",
            description="Claude streak",
            significance_score=0.5,
        )

        detector.record_moment(moment1)
        detector.record_moment(moment2)

        moments = detector.get_agent_moments("claude")
        assert len(moments) == 2
        # Sorted by significance (highest first)
        assert moments[0].id == "m1"

    def test_filter_by_moment_type(self):
        """Filter moments by type."""
        detector = MomentDetector()

        detector.record_moment(SignificantMoment(
            id="m1", moment_type="upset_victory", agent_name="a",
            description="d", significance_score=0.8,
        ))
        detector.record_moment(SignificantMoment(
            id="m2", moment_type="streak_achievement", agent_name="a",
            description="d", significance_score=0.5,
        ))

        upsets = detector.get_agent_moments("a", moment_types=["upset_victory"])
        assert len(upsets) == 1
        assert upsets[0].moment_type == "upset_victory"

    def test_moment_recorded_for_other_agents(self):
        """Moments with other_agents are accessible for those agents too."""
        detector = MomentDetector()

        moment = SignificantMoment(
            id="m1",
            moment_type="consensus_breakthrough",
            agent_name="claude",
            description="Consensus",
            significance_score=0.7,
            other_agents=["gemini", "grok"],
        )

        detector.record_moment(moment)

        # Accessible for primary agent
        assert len(detector.get_agent_moments("claude")) == 1
        # Also accessible for other agents
        assert len(detector.get_agent_moments("gemini")) == 1
        assert len(detector.get_agent_moments("grok")) == 1

    def test_limit_moments(self):
        """Limit number of returned moments."""
        detector = MomentDetector()

        for i in range(10):
            detector.record_moment(SignificantMoment(
                id=f"m{i}", moment_type="streak_achievement", agent_name="a",
                description="d", significance_score=0.5 + i * 0.05,
            ))

        moments = detector.get_agent_moments("a", limit=3)
        assert len(moments) == 3


class TestMomentDetectorNarrative:
    """Tests for narrative formatting."""

    def test_format_moment_narrative(self):
        """Format a moment as narrative text."""
        detector = MomentDetector()

        moment = SignificantMoment(
            id="m1",
            moment_type="upset_victory",
            agent_name="claude",
            description="Claude defeated Gemini despite 200 ELO disadvantage",
            significance_score=0.75,
        )

        narrative = detector.format_moment_narrative(moment)
        assert "Major Moment" in narrative  # 0.6-0.8 = "major"
        assert "Claude defeated Gemini" in narrative

    def test_significance_labels(self):
        """Different significance scores get different labels."""
        detector = MomentDetector()

        notable = SignificantMoment(id="1", moment_type="t", agent_name="a",
                                     description="d", significance_score=0.2)
        significant = SignificantMoment(id="2", moment_type="t", agent_name="a",
                                         description="d", significance_score=0.4)
        major = SignificantMoment(id="3", moment_type="t", agent_name="a",
                                   description="d", significance_score=0.7)
        defining = SignificantMoment(id="4", moment_type="t", agent_name="a",
                                      description="d", significance_score=0.9)

        assert "Notable" in detector.format_moment_narrative(notable)
        assert "Significant" in detector.format_moment_narrative(significant)
        assert "Major" in detector.format_moment_narrative(major)
        assert "Defining" in detector.format_moment_narrative(defining)

    def test_narrative_summary_no_moments(self):
        """Summary for agent with no moments."""
        detector = MomentDetector()
        summary = detector.get_narrative_summary("unknown_agent")
        assert "not yet established" in summary

    def test_narrative_summary_with_moments(self):
        """Summary for agent with moments."""
        detector = MomentDetector()
        detector.record_moment(SignificantMoment(
            id="m1", moment_type="upset_victory", agent_name="claude",
            description="Claude had a major upset", significance_score=0.8,
        ))

        summary = detector.get_narrative_summary("claude")
        assert "claude" in summary.lower()
        assert "Defining Moments" in summary
        assert "upset" in summary.lower()
