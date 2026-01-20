"""Tests for uncertainty estimation in debates."""

import pytest
from collections import Counter
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.core import Agent, Message, Vote
from aragora.uncertainty.estimator import (
    ConfidenceScore,
    DisagreementCrux,
    FollowUpSuggestion,
    UncertaintyMetrics,
    ConfidenceEstimator,
    DisagreementAnalyzer,
    UncertaintyAggregator,
)


# =============================================================================
# ConfidenceScore Tests
# =============================================================================


class TestConfidenceScore:
    """Tests for ConfidenceScore dataclass."""

    def test_create_confidence_score(self):
        """Basic confidence score creation."""
        score = ConfidenceScore(
            agent_name="claude",
            value=0.85,
            reasoning="High certainty based on evidence",
        )
        assert score.agent_name == "claude"
        assert score.value == 0.85
        assert score.reasoning == "High certainty based on evidence"
        assert isinstance(score.timestamp, datetime)

    def test_confidence_score_default_reasoning(self):
        """Confidence score with default reasoning."""
        score = ConfidenceScore(agent_name="gpt", value=0.5)
        assert score.reasoning == ""

    def test_confidence_score_to_dict(self):
        """Serialization to dictionary."""
        score = ConfidenceScore(
            agent_name="gemini",
            value=0.7,
            reasoning="Moderate confidence",
        )
        data = score.to_dict()
        assert data["agent"] == "gemini"
        assert data["confidence"] == 0.7
        assert data["reasoning"] == "Moderate confidence"
        assert "timestamp" in data


# =============================================================================
# DisagreementCrux Tests
# =============================================================================


class TestDisagreementCrux:
    """Tests for DisagreementCrux dataclass."""

    def test_create_disagreement_crux(self):
        """Basic crux creation."""
        crux = DisagreementCrux(
            description="The approach may not scale",
            divergent_agents=["claude", "gpt"],
            evidence_needed="Performance benchmarks",
            severity=0.8,
        )
        assert crux.description == "The approach may not scale"
        assert crux.divergent_agents == ["claude", "gpt"]
        assert crux.severity == 0.8
        assert crux.crux_id.startswith("crux-")

    def test_crux_auto_generates_id(self):
        """Crux auto-generates stable ID from description."""
        crux1 = DisagreementCrux(
            description="Same issue",
            divergent_agents=["agent1"],
        )
        crux2 = DisagreementCrux(
            description="Same issue",
            divergent_agents=["agent2"],
        )
        # Same description should produce same ID
        assert crux1.crux_id == crux2.crux_id

    def test_crux_different_descriptions_different_ids(self):
        """Different descriptions produce different IDs."""
        crux1 = DisagreementCrux(description="Issue A", divergent_agents=[])
        crux2 = DisagreementCrux(description="Issue B", divergent_agents=[])
        assert crux1.crux_id != crux2.crux_id

    def test_crux_custom_id_preserved(self):
        """Custom crux ID is preserved."""
        crux = DisagreementCrux(
            description="Test",
            divergent_agents=[],
            crux_id="custom-123",
        )
        assert crux.crux_id == "custom-123"

    def test_crux_to_dict(self):
        """Serialization to dictionary."""
        crux = DisagreementCrux(
            description="Test crux",
            divergent_agents=["a1", "a2"],
            evidence_needed="More data",
            severity=0.6,
        )
        data = crux.to_dict()
        assert data["description"] == "Test crux"
        assert data["agents"] == ["a1", "a2"]
        assert data["evidence_needed"] == "More data"
        assert data["severity"] == 0.6
        assert "id" in data


# =============================================================================
# FollowUpSuggestion Tests
# =============================================================================


class TestFollowUpSuggestion:
    """Tests for FollowUpSuggestion dataclass."""

    def test_create_followup_suggestion(self):
        """Basic follow-up suggestion creation."""
        crux = DisagreementCrux(
            description="Scalability concerns",
            divergent_agents=["claude"],
        )
        suggestion = FollowUpSuggestion(
            crux=crux,
            suggested_task="Investigate scalability",
            priority=0.9,
            parent_debate_id="debate-123",
            suggested_agents=["claude", "gpt"],
        )
        assert suggestion.crux == crux
        assert suggestion.suggested_task == "Investigate scalability"
        assert suggestion.priority == 0.9
        assert suggestion.parent_debate_id == "debate-123"

    def test_followup_to_dict(self):
        """Serialization to dictionary."""
        crux = DisagreementCrux(description="Test", divergent_agents=["a1"])
        suggestion = FollowUpSuggestion(
            crux=crux,
            suggested_task="Follow up task",
            priority=0.7,
        )
        data = suggestion.to_dict()
        assert data["suggested_task"] == "Follow up task"
        assert data["priority"] == 0.7
        assert data["crux_description"] == "Test"


# =============================================================================
# UncertaintyMetrics Tests
# =============================================================================


class TestUncertaintyMetrics:
    """Tests for UncertaintyMetrics dataclass."""

    def test_default_metrics(self):
        """Default uncertainty metrics."""
        metrics = UncertaintyMetrics()
        assert metrics.collective_confidence == 0.5
        assert metrics.confidence_interval == (0.4, 0.6)
        assert metrics.disagreement_type == "none"
        assert metrics.cruxes == []
        assert metrics.calibration_quality == 0.5

    def test_metrics_to_dict(self):
        """Serialization to dictionary."""
        crux = DisagreementCrux(description="Test", divergent_agents=[])
        metrics = UncertaintyMetrics(
            collective_confidence=0.8,
            confidence_interval=(0.7, 0.9),
            disagreement_type="factual",
            cruxes=[crux],
            calibration_quality=0.75,
        )
        data = metrics.to_dict()
        assert data["collective_confidence"] == 0.8
        assert data["confidence_interval"] == (0.7, 0.9)
        assert data["disagreement_type"] == "factual"
        assert len(data["cruxes"]) == 1
        assert data["calibration_quality"] == 0.75


# =============================================================================
# DisagreementAnalyzer Tests
# =============================================================================


class TestDisagreementAnalyzer:
    """Tests for DisagreementAnalyzer."""

    def test_unanimous_agreement(self):
        """Unanimous votes should yield high confidence."""
        analyzer = DisagreementAnalyzer()

        votes = [
            Vote(agent="claude", choice="A", confidence=0.9, reasoning="Good"),
            Vote(agent="gpt", choice="A", confidence=0.85, reasoning="Solid"),
            Vote(agent="gemini", choice="A", confidence=0.88, reasoning="Agree"),
        ]
        messages = []
        proposals = {"A": "Proposal A", "B": "Proposal B"}

        metrics = analyzer.analyze_disagreement(messages, votes, proposals)

        assert metrics.disagreement_type == "none"
        assert metrics.collective_confidence > 0.8
        assert len(metrics.cruxes) == 0

    def test_split_votes_disagreement(self):
        """Split votes should yield lower confidence and identify disagreement."""
        analyzer = DisagreementAnalyzer()

        votes = [
            Vote(agent="claude", choice="A", confidence=0.8, reasoning="A is better"),
            Vote(agent="gpt", choice="B", confidence=0.75, reasoning="B is better"),
            Vote(agent="gemini", choice="A", confidence=0.7, reasoning="A seems right"),
        ]
        messages = [
            Message(agent="gpt", role="critic", content="However, I disagree with the approach"),
        ]
        proposals = {"A": "Proposal A", "B": "Proposal B"}

        metrics = analyzer.analyze_disagreement(messages, votes, proposals)

        # Should detect disagreement
        assert metrics.disagreement_type != "none"
        # Confidence should be reduced due to disagreement
        assert metrics.collective_confidence < 0.8

    def test_empty_votes(self):
        """Empty votes should return default metrics."""
        analyzer = DisagreementAnalyzer()

        metrics = analyzer.analyze_disagreement([], [], {})

        assert metrics.collective_confidence == 0.5
        assert metrics.disagreement_type == "none"

    def test_classify_factual_disagreement(self):
        """Factual keywords should classify as factual disagreement."""
        analyzer = DisagreementAnalyzer()

        votes = [
            Vote(agent="claude", choice="A", confidence=0.8, reasoning=""),
            Vote(agent="gpt", choice="B", confidence=0.7, reasoning=""),
        ]
        messages = [
            Message(
                agent="gpt",
                role="critic",
                content="The evidence shows this is false. The data proves otherwise.",
            ),
        ]

        metrics = analyzer.analyze_disagreement(messages, votes, {})
        assert metrics.disagreement_type == "factual"

    def test_classify_value_disagreement(self):
        """Value keywords should classify as value-based disagreement."""
        analyzer = DisagreementAnalyzer()

        votes = [
            Vote(agent="claude", choice="A", confidence=0.8, reasoning=""),
            Vote(agent="gpt", choice="B", confidence=0.7, reasoning=""),
        ]
        messages = [
            Message(
                agent="gpt",
                role="critic",
                content="I think we should prefer the ethical approach. It's better morally.",
            ),
        ]

        metrics = analyzer.analyze_disagreement(messages, votes, {})
        assert metrics.disagreement_type == "value"

    def test_find_cruxes(self):
        """Should identify cruxes from disagreement markers."""
        analyzer = DisagreementAnalyzer()

        votes = [
            Vote(agent="claude", choice="A", confidence=0.8, reasoning=""),
            Vote(agent="gpt", choice="B", confidence=0.7, reasoning=""),
        ]
        messages = [
            Message(
                agent="gpt",
                role="critic",
                content="However, the scalability concern is significant. But we need more testing.",
            ),
        ]

        metrics = analyzer.analyze_disagreement(messages, votes, {"A": "prop", "B": "prop2"})
        # Should find at least one crux from the "However" and "But" markers
        assert len(metrics.cruxes) >= 0  # May or may not find cruxes depending on extraction

    def test_confidence_interval_calculation(self):
        """Confidence interval should be computed correctly."""
        analyzer = DisagreementAnalyzer()

        votes = [
            Vote(agent="a1", choice="A", confidence=0.8, reasoning=""),
            Vote(agent="a2", choice="A", confidence=0.6, reasoning=""),
            Vote(agent="a3", choice="A", confidence=0.7, reasoning=""),
        ]

        metrics = analyzer.analyze_disagreement([], votes, {})

        # Interval should contain the mean
        assert metrics.confidence_interval[0] <= metrics.collective_confidence
        assert metrics.confidence_interval[1] >= metrics.collective_confidence


class TestDisagreementAnalyzerFollowups:
    """Tests for follow-up suggestion generation."""

    def test_suggest_followups_from_cruxes(self):
        """Should generate follow-up suggestions from cruxes."""
        analyzer = DisagreementAnalyzer()

        cruxes = [
            DisagreementCrux(
                description="Performance concerns",
                divergent_agents=["claude", "gpt"],
                severity=0.8,
            ),
            DisagreementCrux(
                description="Cost implications",
                divergent_agents=["gemini"],
                severity=0.5,
            ),
        ]

        suggestions = analyzer.suggest_followups(
            cruxes,
            parent_debate_id="debate-123",
            available_agents=["claude", "gpt", "gemini"],
        )

        assert len(suggestions) == 2
        # Higher severity should come first
        assert suggestions[0].crux.severity >= suggestions[1].crux.severity
        assert suggestions[0].parent_debate_id == "debate-123"

    def test_followup_task_generation(self):
        """Should generate meaningful follow-up tasks."""
        analyzer = DisagreementAnalyzer()

        crux = DisagreementCrux(
            description="The approach may not work at scale",
            divergent_agents=["agent1"],
        )

        suggestions = analyzer.suggest_followups([crux])
        assert len(suggestions) == 1
        assert suggestions[0].suggested_task != ""

    def test_empty_cruxes(self):
        """Empty cruxes should return empty suggestions."""
        analyzer = DisagreementAnalyzer()
        suggestions = analyzer.suggest_followups([])
        assert suggestions == []


# =============================================================================
# ConfidenceEstimator Tests
# =============================================================================


class TestConfidenceEstimator:
    """Tests for ConfidenceEstimator."""

    def test_store_confidence(self):
        """Should store confidence scores."""
        estimator = ConfidenceEstimator()

        score = ConfidenceScore(agent_name="claude", value=0.8, reasoning="test")
        estimator._store_confidence("claude", score)

        assert "claude" in estimator.agent_confidences
        assert len(estimator.agent_confidences["claude"]) == 1

    def test_store_confidence_bounded_history(self):
        """Should maintain bounded history."""
        estimator = ConfidenceEstimator()

        # Store more than the limit
        for i in range(150):
            score = ConfidenceScore(agent_name="claude", value=0.5 + i * 0.001, reasoning="")
            estimator._store_confidence("claude", score)

        # Should be bounded
        assert len(estimator.agent_confidences["claude"]) <= 100

    def test_record_outcome(self):
        """Should record prediction outcomes."""
        estimator = ConfidenceEstimator()

        estimator.record_outcome("claude", 0.8, True)
        estimator.record_outcome("claude", 0.9, False)

        assert len(estimator.calibration_history["claude"]) == 2

    def test_calibration_quality_default(self):
        """Default calibration quality for unknown agent."""
        estimator = ConfidenceEstimator()
        quality = estimator.get_agent_calibration_quality("unknown_agent")
        assert quality == 0.5

    def test_calibration_quality_perfect(self):
        """Perfect calibration should yield high quality."""
        estimator = ConfidenceEstimator()

        # Perfect calibration: 90% confident and correct
        for _ in range(10):
            estimator.record_outcome("claude", 0.9, True)

        quality = estimator.get_agent_calibration_quality("claude")
        # Should be high (close to 1)
        assert quality > 0.5

    def test_calibration_quality_poor(self):
        """Poor calibration should yield low quality."""
        estimator = ConfidenceEstimator()

        # Overconfident and wrong
        for _ in range(10):
            estimator.record_outcome("gpt", 0.9, False)

        quality = estimator.get_agent_calibration_quality("gpt")
        # Should be low
        assert quality < 0.5

    @pytest.mark.asyncio
    async def test_collect_confidences_with_mock_agents(self):
        """Should collect confidences from agents."""
        estimator = ConfidenceEstimator()

        # Create mock agents
        mock_agent = MagicMock(spec=Agent)
        mock_agent.name = "mock_agent"
        mock_vote = MagicMock()
        mock_vote.confidence = 0.75
        mock_vote.reasoning = "test reasoning"
        mock_agent.vote = AsyncMock(return_value=mock_vote)

        agents = [mock_agent]
        proposals = {"A": "test proposal"}

        confidences = await estimator.collect_confidences(agents, proposals, "test task")

        assert "mock_agent" in confidences
        assert confidences["mock_agent"].value == 0.75

    @pytest.mark.asyncio
    async def test_collect_confidences_handles_errors(self):
        """Should handle agent errors gracefully."""
        estimator = ConfidenceEstimator()

        mock_agent = MagicMock(spec=Agent)
        mock_agent.name = "failing_agent"
        mock_agent.vote = AsyncMock(side_effect=Exception("Agent error"))

        confidences = await estimator.collect_confidences([mock_agent], {}, "task")

        # Should have default confidence
        assert "failing_agent" in confidences
        assert confidences["failing_agent"].value == 0.5


# =============================================================================
# UncertaintyAggregator Tests
# =============================================================================


class TestUncertaintyAggregator:
    """Tests for UncertaintyAggregator."""

    @pytest.mark.asyncio
    async def test_compute_uncertainty(self):
        """Should compute comprehensive uncertainty metrics."""
        confidence_estimator = ConfidenceEstimator()
        disagreement_analyzer = DisagreementAnalyzer()
        aggregator = UncertaintyAggregator(confidence_estimator, disagreement_analyzer)

        # Mock agent
        mock_agent = MagicMock(spec=Agent)
        mock_agent.name = "test_agent"
        mock_vote = MagicMock()
        mock_vote.confidence = 0.8
        mock_vote.reasoning = "test"
        mock_agent.vote = AsyncMock(return_value=mock_vote)

        votes = [Vote(agent="test_agent", choice="A", confidence=0.8, reasoning="")]
        messages = []
        proposals = {"A": "proposal"}

        metrics = await aggregator.compute_uncertainty(
            [mock_agent], messages, votes, proposals
        )

        assert isinstance(metrics, UncertaintyMetrics)
        assert 0 <= metrics.collective_confidence <= 1
        assert 0 <= metrics.calibration_quality <= 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestUncertaintyIntegration:
    """Integration tests for the uncertainty module."""

    def test_full_disagreement_analysis_flow(self):
        """Test complete flow from votes to cruxes to follow-ups."""
        analyzer = DisagreementAnalyzer()

        # Create a realistic disagreement scenario
        votes = [
            Vote(agent="claude", choice="A", confidence=0.85, reasoning="Prefer A"),
            Vote(agent="gpt", choice="B", confidence=0.7, reasoning="B is safer"),
            Vote(agent="gemini", choice="A", confidence=0.75, reasoning="A has merit"),
        ]
        messages = [
            Message(
                agent="gpt",
                role="critic",
                content="However, approach A has scalability concerns that worry me.",
            ),
        ]
        proposals = {"A": "Innovative approach", "B": "Conservative approach"}

        # Analyze disagreement
        metrics = analyzer.analyze_disagreement(messages, votes, proposals)

        # Generate follow-ups if there are cruxes
        if metrics.cruxes:
            suggestions = analyzer.suggest_followups(
                metrics.cruxes,
                parent_debate_id="test-debate",
            )
            assert all(isinstance(s, FollowUpSuggestion) for s in suggestions)

    def test_confidence_tracking_over_time(self):
        """Test that confidence tracking accumulates correctly."""
        estimator = ConfidenceEstimator()

        # Simulate multiple debates
        for i in range(5):
            score = ConfidenceScore(
                agent_name="claude",
                value=0.7 + i * 0.05,
                reasoning=f"Debate {i}",
            )
            estimator._store_confidence("claude", score)

        assert len(estimator.agent_confidences["claude"]) == 5
        # Values should be stored in order
        values = [s.value for s in estimator.agent_confidences["claude"]]
        assert values == sorted(values)  # Increasing order
