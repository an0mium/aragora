"""
Tests for Position Tracker - Agent position evolution tracking.

Tests cover:
- Position recording
- Pivot detection
- Convergence scoring
- Stability analysis
- Influence identification
"""

import pytest
from datetime import datetime, timezone


class TestPositionStance:
    """Tests for PositionStance enum."""

    def test_from_confidence_strongly_agree(self):
        """Should classify high confidence as strongly agree."""
        from aragora.reasoning.position_tracker import PositionStance

        stance = PositionStance.from_confidence(0.95, agrees=True)
        assert stance == PositionStance.STRONGLY_AGREE

    def test_from_confidence_disagree(self):
        """Should classify disagreement correctly."""
        from aragora.reasoning.position_tracker import PositionStance

        stance = PositionStance.from_confidence(0.75, agrees=False)
        assert stance == PositionStance.DISAGREE

    def test_from_confidence_neutral(self):
        """Should classify low confidence as neutral."""
        from aragora.reasoning.position_tracker import PositionStance

        stance = PositionStance.from_confidence(0.4, agrees=True)
        assert stance == PositionStance.NEUTRAL

    def test_numeric_values(self):
        """Should have correct numeric values for comparison."""
        from aragora.reasoning.position_tracker import PositionStance

        assert PositionStance.STRONGLY_AGREE.numeric_value == 1.0
        assert PositionStance.NEUTRAL.numeric_value == 0.5
        assert PositionStance.STRONGLY_DISAGREE.numeric_value == 0.0


class TestPositionRecord:
    """Tests for PositionRecord dataclass."""

    def test_create_position_record(self):
        """Should create position record with required fields."""
        from aragora.reasoning.position_tracker import PositionRecord, PositionStance

        record = PositionRecord(
            agent="claude",
            round_number=1,
            stance=PositionStance.AGREE,
            confidence=0.8,
            key_argument="The approach is sound because...",
        )

        assert record.agent == "claude"
        assert record.round_number == 1
        assert record.stance == PositionStance.AGREE
        assert record.confidence == 0.8

    def test_position_record_to_dict(self):
        """Should serialize to dictionary."""
        from aragora.reasoning.position_tracker import PositionRecord, PositionStance

        record = PositionRecord(
            agent="gpt",
            round_number=2,
            stance=PositionStance.LEAN_DISAGREE,
            confidence=0.6,
            key_argument="I have concerns about...",
            influenced_by=["claude"],
        )

        data = record.to_dict()
        assert data["agent"] == "gpt"
        assert data["round"] == 2
        assert data["stance"] == "lean_disagree"
        assert "claude" in data["influenced_by"]


class TestPositionPivot:
    """Tests for PositionPivot dataclass."""

    def test_pivot_reversal_detection(self):
        """Should detect reversal pivot type."""
        from aragora.reasoning.position_tracker import PositionPivot, PositionStance

        pivot = PositionPivot(
            agent="claude",
            from_round=1,
            to_round=2,
            from_stance=PositionStance.AGREE,
            to_stance=PositionStance.DISAGREE,
            from_confidence=0.7,
            to_confidence=0.8,
        )

        assert pivot.pivot_type == "reversal"
        assert pivot.pivot_magnitude > 0.4  # Significant change

    def test_pivot_strengthening_detection(self):
        """Should detect strengthening pivot type."""
        from aragora.reasoning.position_tracker import PositionPivot, PositionStance

        pivot = PositionPivot(
            agent="gpt",
            from_round=1,
            to_round=2,
            from_stance=PositionStance.LEAN_AGREE,
            to_stance=PositionStance.STRONGLY_AGREE,
            from_confidence=0.6,
            to_confidence=0.95,
        )

        assert pivot.pivot_type == "strengthening"

    def test_pivot_to_dict(self):
        """Should serialize to dictionary."""
        from aragora.reasoning.position_tracker import PositionPivot, PositionStance

        pivot = PositionPivot(
            agent="gemini",
            from_round=2,
            to_round=3,
            from_stance=PositionStance.NEUTRAL,
            to_stance=PositionStance.AGREE,
            from_confidence=0.5,
            to_confidence=0.75,
            trigger_agent="claude",
        )

        data = pivot.to_dict()
        assert data["agent"] == "gemini"
        assert data["from_stance"] == "neutral"
        assert data["to_stance"] == "agree"
        assert data["trigger_agent"] == "claude"


class TestPositionEvolution:
    """Tests for PositionEvolution class."""

    def test_record_position_creates_record(self):
        """Should create position record."""
        from aragora.reasoning.position_tracker import PositionEvolution, PositionStance

        evolution = PositionEvolution(
            debate_id="debate-001",
            topic="Should we adopt microservices?",
        )

        pivot = evolution.record_position(
            agent="claude",
            round_number=1,
            stance=PositionStance.AGREE,
            confidence=0.7,
            key_argument="Microservices improve scalability",
        )

        assert pivot is None  # No pivot on first position
        assert "claude" in evolution.positions
        assert len(evolution.positions["claude"]) == 1

    def test_record_position_detects_pivot(self):
        """Should detect pivot when position changes."""
        from aragora.reasoning.position_tracker import PositionEvolution, PositionStance

        evolution = PositionEvolution(
            debate_id="debate-002",
            topic="API design approach",
        )

        # Initial position
        evolution.record_position(
            agent="gpt",
            round_number=1,
            stance=PositionStance.DISAGREE,
            confidence=0.7,
            key_argument="GraphQL has complexity issues",
        )

        # Changed position
        pivot = evolution.record_position(
            agent="gpt",
            round_number=2,
            stance=PositionStance.AGREE,
            confidence=0.8,
            key_argument="After considering the type safety benefits...",
            influenced_by=["claude"],
        )

        assert pivot is not None
        assert pivot.agent == "gpt"
        assert pivot.from_stance == PositionStance.DISAGREE
        assert pivot.to_stance == PositionStance.AGREE
        assert pivot.pivot_type == "reversal"

    def test_get_agent_trajectory(self):
        """Should return position trajectory for agent."""
        from aragora.reasoning.position_tracker import PositionEvolution, PositionStance

        evolution = PositionEvolution(debate_id="debate-003", topic="Testing")

        evolution.record_position("claude", 1, PositionStance.NEUTRAL, 0.5, "Initial")
        evolution.record_position("claude", 2, PositionStance.LEAN_AGREE, 0.6, "Reconsidering")
        evolution.record_position("claude", 3, PositionStance.AGREE, 0.8, "Convinced")

        trajectory = evolution.get_agent_trajectory("claude")
        assert len(trajectory) == 3
        assert trajectory[0].stance == PositionStance.NEUTRAL
        assert trajectory[2].stance == PositionStance.AGREE

    def test_get_round_positions(self):
        """Should return all positions for a round."""
        from aragora.reasoning.position_tracker import PositionEvolution, PositionStance

        evolution = PositionEvolution(debate_id="debate-004", topic="Database choice")

        evolution.record_position("claude", 1, PositionStance.AGREE, 0.7, "PostgreSQL")
        evolution.record_position("gpt", 1, PositionStance.DISAGREE, 0.6, "MongoDB")
        evolution.record_position("gemini", 1, PositionStance.NEUTRAL, 0.5, "Either works")

        round_positions = evolution.get_round_positions(1)
        assert len(round_positions) == 3
        assert round_positions["claude"].stance == PositionStance.AGREE
        assert round_positions["gpt"].stance == PositionStance.DISAGREE

    def test_calculate_convergence_score_full_consensus(self):
        """Should return high score for full consensus."""
        from aragora.reasoning.position_tracker import PositionEvolution, PositionStance

        evolution = PositionEvolution(debate_id="debate-005", topic="Consensus test")

        # All agents end up agreeing
        evolution.record_position("claude", 3, PositionStance.STRONGLY_AGREE, 0.95, "Yes")
        evolution.record_position("gpt", 3, PositionStance.STRONGLY_AGREE, 0.9, "Agreed")
        evolution.record_position("gemini", 3, PositionStance.AGREE, 0.85, "Makes sense")

        score = evolution.calculate_convergence_score()
        assert score >= 0.9  # Should be high

    def test_calculate_convergence_score_no_consensus(self):
        """Should return low score for polarized positions."""
        from aragora.reasoning.position_tracker import PositionEvolution, PositionStance

        evolution = PositionEvolution(debate_id="debate-006", topic="Polarized test")

        # Agents remain polarized
        evolution.record_position("claude", 3, PositionStance.STRONGLY_AGREE, 0.95, "Yes")
        evolution.record_position("gpt", 3, PositionStance.STRONGLY_DISAGREE, 0.95, "No")

        score = evolution.calculate_convergence_score()
        assert score < 0.5  # Should be low

    def test_calculate_stability_scores(self):
        """Should calculate stability for each agent."""
        from aragora.reasoning.position_tracker import PositionEvolution, PositionStance

        evolution = PositionEvolution(debate_id="debate-007", topic="Stability test")

        # Claude is stable
        evolution.record_position("claude", 1, PositionStance.AGREE, 0.7, "A")
        evolution.record_position("claude", 2, PositionStance.AGREE, 0.75, "A")
        evolution.record_position("claude", 3, PositionStance.STRONGLY_AGREE, 0.9, "A")

        # GPT is unstable (flip-flops)
        evolution.record_position("gpt", 1, PositionStance.AGREE, 0.7, "B")
        evolution.record_position("gpt", 2, PositionStance.DISAGREE, 0.6, "B")
        evolution.record_position("gpt", 3, PositionStance.AGREE, 0.8, "B")

        stability = evolution.calculate_stability_scores()
        assert stability["claude"] > stability["gpt"]

    def test_identify_influencers(self):
        """Should identify agents who triggered pivots."""
        from aragora.reasoning.position_tracker import PositionEvolution, PositionStance

        evolution = PositionEvolution(debate_id="debate-008", topic="Influence test")

        # GPT starts neutral
        evolution.record_position("gpt", 1, PositionStance.NEUTRAL, 0.5, "Unsure")

        # GPT pivots, influenced by claude
        evolution.record_position(
            "gpt", 2, PositionStance.AGREE, 0.7, "Convinced",
            influenced_by=["claude"],
        )

        # Gemini starts neutral
        evolution.record_position("gemini", 1, PositionStance.NEUTRAL, 0.5, "Unsure")

        # Gemini pivots, influenced by claude
        evolution.record_position(
            "gemini", 2, PositionStance.AGREE, 0.75, "Also convinced",
            influenced_by=["claude"],
        )

        influencers = evolution.identify_influencers()
        assert "claude" in influencers
        assert influencers["claude"] == 2  # Triggered 2 pivots

    def test_to_dict_serialization(self):
        """Should serialize complete evolution to dict."""
        from aragora.reasoning.position_tracker import PositionEvolution, PositionStance

        evolution = PositionEvolution(debate_id="debate-009", topic="Serialization test")
        evolution.record_position("claude", 1, PositionStance.AGREE, 0.8, "Argument")

        data = evolution.to_dict()
        assert data["debate_id"] == "debate-009"
        assert "positions" in data
        assert "pivots" in data
        assert "summary" in data
        assert "convergence_score" in data["summary"]


class TestPositionTracker:
    """Tests for PositionTracker service."""

    def test_create_evolution(self):
        """Should create new evolution tracker."""
        from aragora.reasoning.position_tracker import PositionTracker

        tracker = PositionTracker()
        evolution = tracker.create_evolution("debate-100", "Test topic")

        assert evolution is not None
        assert evolution.debate_id == "debate-100"
        assert evolution.topic == "Test topic"

    def test_get_evolution(self):
        """Should retrieve existing evolution."""
        from aragora.reasoning.position_tracker import PositionTracker

        tracker = PositionTracker()
        tracker.create_evolution("debate-101", "Test")

        evolution = tracker.get_evolution("debate-101")
        assert evolution is not None
        assert evolution.debate_id == "debate-101"

    def test_get_evolution_not_found(self):
        """Should return None for unknown debate."""
        from aragora.reasoning.position_tracker import PositionTracker

        tracker = PositionTracker()
        evolution = tracker.get_evolution("nonexistent")
        assert evolution is None

    def test_record_from_message(self):
        """Should infer stance from sentiment score."""
        from aragora.reasoning.position_tracker import PositionTracker, PositionStance

        tracker = PositionTracker()
        tracker.create_evolution("debate-102", "Sentiment test")

        # High sentiment -> agree
        tracker.record_from_message(
            debate_id="debate-102",
            agent="claude",
            round_number=1,
            content="This is a great idea!",
            sentiment_score=0.85,
        )

        evolution = tracker.get_evolution("debate-102")
        assert evolution.positions["claude"][0].stance == PositionStance.AGREE

    def test_analyze_debate(self):
        """Should generate comprehensive analysis."""
        from aragora.reasoning.position_tracker import PositionTracker, PositionStance

        tracker = PositionTracker()
        evolution = tracker.create_evolution("debate-103", "Analysis test")

        evolution.record_position("claude", 1, PositionStance.NEUTRAL, 0.5, "Start")
        evolution.record_position("claude", 2, PositionStance.AGREE, 0.8, "End")

        analysis = tracker.analyze_debate("debate-103")

        assert analysis is not None
        assert analysis["debate_id"] == "debate-103"
        assert "position_trajectories" in analysis
        assert "pivot_analysis" in analysis
        assert "convergence" in analysis
        assert "stability" in analysis

    def test_convergence_interpretation(self):
        """Should provide human-readable convergence interpretation."""
        from aragora.reasoning.position_tracker import PositionTracker

        tracker = PositionTracker()

        # Test different scores
        assert "Strong consensus" in tracker._interpret_convergence(0.95)
        assert "Good convergence" in tracker._interpret_convergence(0.75)
        assert "Moderate" in tracker._interpret_convergence(0.55)
        assert "Weak" in tracker._interpret_convergence(0.35)
        assert "No convergence" in tracker._interpret_convergence(0.2)


class TestGlobalPositionTracker:
    """Tests for global position tracker singleton."""

    def test_get_position_tracker(self):
        """Should return same instance."""
        from aragora.reasoning.position_tracker import get_position_tracker

        tracker1 = get_position_tracker()
        tracker2 = get_position_tracker()

        assert tracker1 is tracker2
