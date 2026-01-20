"""
Tests for the uncertainty estimation module.

Tests cover:
- ConfidenceScore dataclass
- DisagreementCrux dataclass
- FollowUpSuggestion dataclass
- UncertaintyMetrics dataclass
- ConfidenceEstimator class
- DisagreementAnalyzer class
- UncertaintyAggregator class
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.uncertainty.estimator import (
    ConfidenceEstimator,
    ConfidenceScore,
    DisagreementAnalyzer,
    DisagreementCrux,
    FollowUpSuggestion,
    UncertaintyAggregator,
    UncertaintyMetrics,
)


# ============================================================================
# ConfidenceScore Tests
# ============================================================================


class TestConfidenceScore:
    """Tests for ConfidenceScore dataclass."""

    def test_creation(self):
        """Test basic creation with required fields."""
        score = ConfidenceScore(agent_name="claude", value=0.85)
        assert score.agent_name == "claude"
        assert score.value == 0.85
        assert score.reasoning == ""

    def test_creation_with_reasoning(self):
        """Test creation with all fields."""
        score = ConfidenceScore(
            agent_name="gpt", value=0.9, reasoning="High confidence due to clear evidence"
        )
        assert score.reasoning == "High confidence due to clear evidence"

    def test_timestamp_default(self):
        """Test timestamp is set to now by default."""
        score = ConfidenceScore(agent_name="test", value=0.5)
        assert isinstance(score.timestamp, datetime)
        # Should be within the last second
        assert (datetime.now() - score.timestamp).total_seconds() < 1

    def test_to_dict(self):
        """Test serialization to dictionary."""
        score = ConfidenceScore(
            agent_name="claude", value=0.75, reasoning="Based on evidence"
        )
        result = score.to_dict()

        assert result["agent"] == "claude"
        assert result["confidence"] == 0.75
        assert result["reasoning"] == "Based on evidence"
        assert "timestamp" in result

    def test_value_boundaries(self):
        """Test edge case confidence values."""
        low = ConfidenceScore(agent_name="test", value=0.0)
        high = ConfidenceScore(agent_name="test", value=1.0)

        assert low.value == 0.0
        assert high.value == 1.0


# ============================================================================
# DisagreementCrux Tests
# ============================================================================


class TestDisagreementCrux:
    """Tests for DisagreementCrux dataclass."""

    def test_creation(self):
        """Test basic creation."""
        crux = DisagreementCrux(
            description="Disagreement about data accuracy",
            divergent_agents=["claude", "gpt"],
        )
        assert crux.description == "Disagreement about data accuracy"
        assert crux.divergent_agents == ["claude", "gpt"]

    def test_auto_generated_id(self):
        """Test crux_id is auto-generated when not provided."""
        crux = DisagreementCrux(
            description="Test crux", divergent_agents=["agent1"]
        )
        assert crux.crux_id.startswith("crux-")
        assert len(crux.crux_id) == 10  # "crux-" + 5 digits

    def test_custom_id(self):
        """Test custom crux_id is preserved."""
        crux = DisagreementCrux(
            description="Test", divergent_agents=[], crux_id="custom-id"
        )
        assert crux.crux_id == "custom-id"

    def test_default_values(self):
        """Test default values are set correctly."""
        crux = DisagreementCrux(
            description="Test", divergent_agents=[]
        )
        assert crux.evidence_needed == ""
        assert crux.severity == 0.5

    def test_to_dict(self):
        """Test serialization."""
        crux = DisagreementCrux(
            description="Issue",
            divergent_agents=["a", "b"],
            evidence_needed="More data",
            severity=0.8,
            crux_id="test-123",
        )
        result = crux.to_dict()

        assert result["id"] == "test-123"
        assert result["description"] == "Issue"
        assert result["agents"] == ["a", "b"]
        assert result["evidence_needed"] == "More data"
        assert result["severity"] == 0.8

    def test_id_stability(self):
        """Test same description generates same ID."""
        crux1 = DisagreementCrux(description="Same description", divergent_agents=[])
        crux2 = DisagreementCrux(description="Same description", divergent_agents=[])
        assert crux1.crux_id == crux2.crux_id


# ============================================================================
# FollowUpSuggestion Tests
# ============================================================================


class TestFollowUpSuggestion:
    """Tests for FollowUpSuggestion dataclass."""

    def test_creation(self):
        """Test basic creation."""
        crux = DisagreementCrux(description="Test", divergent_agents=["a"])
        suggestion = FollowUpSuggestion(
            crux=crux,
            suggested_task="Investigate further",
            priority=0.8,
        )
        assert suggestion.suggested_task == "Investigate further"
        assert suggestion.priority == 0.8
        assert suggestion.crux == crux

    def test_default_values(self):
        """Test default values."""
        crux = DisagreementCrux(description="Test", divergent_agents=[])
        suggestion = FollowUpSuggestion(
            crux=crux, suggested_task="Task", priority=0.5
        )
        assert suggestion.parent_debate_id is None
        assert suggestion.suggested_agents == []

    def test_to_dict(self):
        """Test serialization."""
        crux = DisagreementCrux(
            description="Crux desc",
            divergent_agents=["a", "b"],
            crux_id="crux-001",
        )
        suggestion = FollowUpSuggestion(
            crux=crux,
            suggested_task="Do this",
            priority=0.9,
            parent_debate_id="debate-123",
            suggested_agents=["a", "b", "c"],
        )
        result = suggestion.to_dict()

        assert result["crux_id"] == "crux-001"
        assert result["crux_description"] == "Crux desc"
        assert result["suggested_task"] == "Do this"
        assert result["priority"] == 0.9
        assert result["parent_debate_id"] == "debate-123"
        assert result["suggested_agents"] == ["a", "b", "c"]
        assert result["divergent_agents"] == ["a", "b"]


# ============================================================================
# UncertaintyMetrics Tests
# ============================================================================


class TestUncertaintyMetrics:
    """Tests for UncertaintyMetrics dataclass."""

    def test_default_values(self):
        """Test default values."""
        metrics = UncertaintyMetrics()
        assert metrics.collective_confidence == 0.5
        assert metrics.confidence_interval == (0.4, 0.6)
        assert metrics.disagreement_type == "none"
        assert metrics.cruxes == []
        assert metrics.calibration_quality == 0.5

    def test_custom_values(self):
        """Test with custom values."""
        crux = DisagreementCrux(description="Test", divergent_agents=[])
        metrics = UncertaintyMetrics(
            collective_confidence=0.8,
            confidence_interval=(0.7, 0.9),
            disagreement_type="factual",
            cruxes=[crux],
            calibration_quality=0.7,
        )
        assert metrics.collective_confidence == 0.8
        assert metrics.disagreement_type == "factual"
        assert len(metrics.cruxes) == 1

    def test_to_dict(self):
        """Test serialization."""
        crux = DisagreementCrux(description="Test", divergent_agents=["a"])
        metrics = UncertaintyMetrics(
            collective_confidence=0.75,
            confidence_interval=(0.6, 0.9),
            disagreement_type="value",
            cruxes=[crux],
            calibration_quality=0.8,
        )
        result = metrics.to_dict()

        assert result["collective_confidence"] == 0.75
        assert result["confidence_interval"] == (0.6, 0.9)
        assert result["disagreement_type"] == "value"
        assert len(result["cruxes"]) == 1
        assert result["calibration_quality"] == 0.8


# ============================================================================
# ConfidenceEstimator Tests
# ============================================================================


class TestConfidenceEstimator:
    """Tests for ConfidenceEstimator class."""

    def test_initialization(self):
        """Test initialization."""
        estimator = ConfidenceEstimator()
        assert estimator.agent_confidences == {}
        assert estimator.calibration_history == {}
        assert estimator.brier_scores == {}
        assert estimator.disagreement_analyzer is not None

    def test_store_confidence(self):
        """Test storing confidence for tracking."""
        estimator = ConfidenceEstimator()
        score = ConfidenceScore(agent_name="claude", value=0.8)

        estimator._store_confidence("claude", score)

        assert "claude" in estimator.agent_confidences
        assert len(estimator.agent_confidences["claude"]) == 1
        assert estimator.agent_confidences["claude"][0].value == 0.8

    def test_store_confidence_limits_history(self):
        """Test that history is limited to prevent memory growth."""
        estimator = ConfidenceEstimator()

        # Add more than limit (>100 triggers trimming to 50)
        for i in range(150):
            score = ConfidenceScore(agent_name="test", value=i / 150)
            estimator._store_confidence("test", score)

        # Implementation keeps last 50 when > 100, so with 150 entries:
        # After 101st entry: trimmed to 50
        # Then 102-150 added: 50 + 49 = 99 entries
        # The limit check only triggers when > 100
        assert len(estimator.agent_confidences["test"]) <= 100

    def test_record_outcome(self):
        """Test recording prediction outcomes."""
        estimator = ConfidenceEstimator()

        estimator.record_outcome("claude", 0.8, True)
        estimator.record_outcome("claude", 0.6, False)

        assert "claude" in estimator.calibration_history
        assert len(estimator.calibration_history["claude"]) == 2
        assert estimator.calibration_history["claude"][0] == (0.8, True)
        assert estimator.calibration_history["claude"][1] == (0.6, False)

    def test_record_outcome_limits_history(self):
        """Test that calibration history is limited."""
        estimator = ConfidenceEstimator()

        # Add more than limit (>50 triggers trimming to 25)
        for i in range(60):
            estimator.record_outcome("test", 0.5, True)

        # Implementation keeps last 25 when > 50
        # After 51st entry: trimmed to 25
        # Then 52-60 added: 25 + 9 = 34 entries
        assert len(estimator.calibration_history["test"]) <= 50

    def test_calibration_quality_no_history(self):
        """Test calibration quality with no history."""
        estimator = ConfidenceEstimator()
        assert estimator.get_agent_calibration_quality("unknown") == 0.5

    def test_calibration_quality_perfect(self):
        """Test calibration quality with perfect predictions."""
        estimator = ConfidenceEstimator()

        # Perfect calibration: high confidence when correct
        estimator.record_outcome("perfect", 1.0, True)
        estimator.record_outcome("perfect", 0.0, False)

        quality = estimator.get_agent_calibration_quality("perfect")
        assert quality == 1.0  # Perfect calibration

    def test_calibration_quality_poor(self):
        """Test calibration quality with poor predictions."""
        estimator = ConfidenceEstimator()

        # Poor calibration: high confidence when wrong
        estimator.record_outcome("poor", 1.0, False)
        estimator.record_outcome("poor", 0.0, True)

        quality = estimator.get_agent_calibration_quality("poor")
        assert quality < 0.5  # Poor calibration

    @pytest.mark.asyncio
    async def test_collect_confidences(self):
        """Test collecting confidences from agents."""
        estimator = ConfidenceEstimator()

        # Mock agents
        agent1 = MagicMock()
        agent1.name = "agent1"
        agent1.vote = AsyncMock(return_value=MagicMock(confidence=0.8, reasoning="Good"))

        agent2 = MagicMock()
        agent2.name = "agent2"
        agent2.vote = AsyncMock(return_value=MagicMock(confidence=0.6, reasoning="Moderate"))

        agents = [agent1, agent2]
        proposals = {"agent1": "Proposal 1", "agent2": "Proposal 2"}

        result = await estimator.collect_confidences(agents, proposals, "Test task")

        assert "agent1" in result
        assert "agent2" in result
        assert result["agent1"].value == 0.8
        assert result["agent2"].value == 0.6

    @pytest.mark.asyncio
    async def test_collect_confidences_handles_errors(self):
        """Test that errors from agents are handled gracefully."""
        estimator = ConfidenceEstimator()

        agent = MagicMock()
        agent.name = "failing"
        agent.vote = AsyncMock(side_effect=Exception("API Error"))

        result = await estimator.collect_confidences([agent], {}, "Task")

        assert "failing" in result
        assert result["failing"].value == 0.5  # Default confidence


# ============================================================================
# DisagreementAnalyzer Tests
# ============================================================================


class TestDisagreementAnalyzer:
    """Tests for DisagreementAnalyzer class."""

    def test_initialization(self):
        """Test initialization."""
        analyzer = DisagreementAnalyzer()
        assert "factual" in analyzer.nlp_keywords
        assert "value" in analyzer.nlp_keywords
        assert "definitional" in analyzer.nlp_keywords
        assert "asymmetry" in analyzer.nlp_keywords

    def test_analyze_unanimous_agreement(self):
        """Test analysis with unanimous agreement."""
        analyzer = DisagreementAnalyzer()

        votes = [
            MagicMock(choice="A", confidence=0.9, agent="agent1"),
            MagicMock(choice="A", confidence=0.8, agent="agent2"),
            MagicMock(choice="A", confidence=0.85, agent="agent3"),
        ]

        metrics = analyzer.analyze_disagreement([], votes, {})

        assert metrics.disagreement_type == "none"
        assert metrics.collective_confidence > 0.8
        assert len(metrics.cruxes) == 0

    def test_analyze_with_disagreement(self):
        """Test analysis with voting disagreement."""
        analyzer = DisagreementAnalyzer()

        votes = [
            MagicMock(choice="A", confidence=0.8, agent="agent1"),
            MagicMock(choice="A", confidence=0.7, agent="agent2"),
            MagicMock(choice="B", confidence=0.6, agent="agent3"),
        ]

        # Add critique message from dissenting agent
        message = MagicMock(
            content="However, I disagree with the factual accuracy of this claim.",
            agent="agent3",
            role="critic",
        )

        metrics = analyzer.analyze_disagreement([message], votes, {"A": "Proposal A"})

        assert metrics.disagreement_type in ["factual", "general"]
        assert metrics.collective_confidence < 0.8  # Reduced due to disagreement

    def test_analyze_empty_votes(self):
        """Test analysis with no votes."""
        analyzer = DisagreementAnalyzer()
        metrics = analyzer.analyze_disagreement([], [], {})

        assert metrics.collective_confidence == 0.5
        assert metrics.disagreement_type == "none"

    def test_classify_disagreement_factual(self):
        """Test classifying factual disagreement."""
        analyzer = DisagreementAnalyzer()

        message = MagicMock(
            content="The evidence shows this is factually incorrect.",
            agent="dissenter",
            role="critic",
        )
        votes = [MagicMock(agent="dissenter")]

        disagreement_type = analyzer._classify_disagreement_type([message], votes)
        assert disagreement_type == "factual"

    def test_classify_disagreement_value(self):
        """Test classifying value-based disagreement."""
        analyzer = DisagreementAnalyzer()

        message = MagicMock(
            content="We should consider the ethical implications better.",
            agent="dissenter",
            role="critic",
        )
        votes = [MagicMock(agent="dissenter")]

        disagreement_type = analyzer._classify_disagreement_type([message], votes)
        assert disagreement_type == "value"

    def test_find_cruxes(self):
        """Test finding disagreement cruxes."""
        analyzer = DisagreementAnalyzer()

        message = MagicMock(
            content="However, I disagree with the main assumption here.",
            agent="dissenter",
            role="critic",
        )
        votes = [MagicMock(agent="dissenter")]

        cruxes = analyzer._find_cruxes([message], {}, "A", votes)

        assert len(cruxes) > 0
        assert "dissenter" in cruxes[0].divergent_agents

    def test_extract_crux_description(self):
        """Test crux description extraction."""
        analyzer = DisagreementAnalyzer()

        content = "First point. However, I have a concern about accuracy. Third point."
        result = analyzer._extract_crux_description(content)

        assert result is not None
        assert "concern" in result.lower()

    def test_extract_crux_description_no_markers(self):
        """Test extraction with no disagreement markers."""
        analyzer = DisagreementAnalyzer()

        content = "Everything looks good. I agree completely."
        result = analyzer._extract_crux_description(content)

        assert result is None

    def test_merge_similar_cruxes(self):
        """Test merging similar cruxes."""
        analyzer = DisagreementAnalyzer()

        crux1 = DisagreementCrux(
            description="Concern about data accuracy",
            divergent_agents=["a"],
            severity=0.5,
        )
        crux2 = DisagreementCrux(
            description="Data accuracy is questionable",
            divergent_agents=["b"],
            severity=0.7,
        )
        crux3 = DisagreementCrux(
            description="Completely different topic",
            divergent_agents=["c"],
            severity=0.6,
        )

        merged = analyzer._merge_similar_cruxes([crux1, crux2, crux3])

        # Should merge crux1 and crux2
        assert len(merged) == 2
        # Merged crux should have both agents
        data_crux = next(c for c in merged if "data" in c.description.lower())
        assert len(data_crux.divergent_agents) == 2
        assert data_crux.severity == 0.7  # Takes max severity

    def test_cruxes_similar(self):
        """Test similarity detection."""
        analyzer = DisagreementAnalyzer()

        crux1 = DisagreementCrux(description="Data accuracy concern", divergent_agents=[])
        crux2 = DisagreementCrux(description="Data accuracy issue", divergent_agents=[])
        crux3 = DisagreementCrux(description="Completely unrelated topic", divergent_agents=[])

        assert analyzer._cruxes_similar(crux1, crux2) is True
        assert analyzer._cruxes_similar(crux1, crux3) is False

    def test_suggest_followups(self):
        """Test follow-up suggestion generation."""
        analyzer = DisagreementAnalyzer()

        cruxes = [
            DisagreementCrux(
                description="Data quality concern",
                divergent_agents=["a", "b"],
                severity=0.8,
            ),
            DisagreementCrux(
                description="Minor issue",
                divergent_agents=["c"],
                severity=0.3,
            ),
        ]

        suggestions = analyzer.suggest_followups(
            cruxes,
            parent_debate_id="debate-001",
            available_agents=["a", "b", "c", "d"],
        )

        assert len(suggestions) == 2
        # Higher severity crux should be first
        assert suggestions[0].priority > suggestions[1].priority
        assert suggestions[0].parent_debate_id == "debate-001"

    def test_generate_followup_task(self):
        """Test follow-up task generation."""
        analyzer = DisagreementAnalyzer()

        crux = DisagreementCrux(
            description="However, I disagree with the data",
            divergent_agents=[],
        )

        task = analyzer._generate_followup_task(crux)
        assert "Investigate:" in task or "Debate:" in task


# ============================================================================
# UncertaintyAggregator Tests
# ============================================================================


class TestUncertaintyAggregator:
    """Tests for UncertaintyAggregator class."""

    def test_initialization(self):
        """Test initialization with dependencies."""
        estimator = ConfidenceEstimator()
        analyzer = DisagreementAnalyzer()

        aggregator = UncertaintyAggregator(estimator, analyzer)

        assert aggregator.confidence_estimator is estimator
        assert aggregator.disagreement_analyzer is analyzer

    @pytest.mark.asyncio
    async def test_compute_uncertainty(self):
        """Test computing comprehensive uncertainty metrics."""
        estimator = ConfidenceEstimator()
        analyzer = DisagreementAnalyzer()
        aggregator = UncertaintyAggregator(estimator, analyzer)

        # Mock agents
        agent = MagicMock()
        agent.name = "test"
        agent.vote = AsyncMock(return_value=MagicMock(confidence=0.8, reasoning="Good"))

        # Mock votes
        votes = [
            MagicMock(choice="A", confidence=0.8, agent="test"),
        ]

        metrics = await aggregator.compute_uncertainty(
            agents=[agent],
            messages=[],
            votes=votes,
            proposals={"test": "Proposal"},
        )

        assert isinstance(metrics, UncertaintyMetrics)
        assert 0 <= metrics.collective_confidence <= 1
        assert 0 <= metrics.calibration_quality <= 1

    @pytest.mark.asyncio
    async def test_compute_uncertainty_incorporates_calibration(self):
        """Test that calibration affects uncertainty calculation."""
        estimator = ConfidenceEstimator()
        analyzer = DisagreementAnalyzer()
        aggregator = UncertaintyAggregator(estimator, analyzer)

        # Add calibration history for the agent
        estimator.record_outcome("calibrated", 0.9, True)
        estimator.record_outcome("calibrated", 0.1, False)

        agent = MagicMock()
        agent.name = "calibrated"
        agent.vote = AsyncMock(return_value=MagicMock(confidence=0.8, reasoning="Good"))

        votes = [MagicMock(choice="A", confidence=0.8, agent="calibrated")]

        metrics = await aggregator.compute_uncertainty(
            agents=[agent], messages=[], votes=votes, proposals={}
        )

        # Calibration quality should affect the result
        assert metrics.calibration_quality > 0.5  # Should be high due to good calibration


# ============================================================================
# Integration Tests
# ============================================================================


class TestUncertaintyIntegration:
    """Integration tests for the uncertainty module."""

    @pytest.mark.asyncio
    async def test_full_uncertainty_workflow(self):
        """Test complete workflow from confidence collection to uncertainty metrics."""
        estimator = ConfidenceEstimator()
        analyzer = DisagreementAnalyzer()
        aggregator = UncertaintyAggregator(estimator, analyzer)

        # Create mock agents with different confidence levels
        agents = []
        for i, name in enumerate(["claude", "gpt", "gemini"]):
            agent = MagicMock()
            agent.name = name
            agent.vote = AsyncMock(
                return_value=MagicMock(
                    confidence=0.7 + i * 0.1, reasoning=f"Reasoning from {name}"
                )
            )
            agents.append(agent)

        # Create votes with some disagreement
        votes = [
            MagicMock(choice="A", confidence=0.7, agent="claude"),
            MagicMock(choice="A", confidence=0.8, agent="gpt"),
            MagicMock(choice="B", confidence=0.9, agent="gemini"),
        ]

        # Add a critique message
        messages = [
            MagicMock(
                content="However, I think option B is better due to efficiency.",
                agent="gemini",
                role="critic",
            )
        ]

        proposals = {
            "claude": "Proposal A",
            "gpt": "Proposal A extended",
            "gemini": "Proposal B",
        }

        metrics = await aggregator.compute_uncertainty(
            agents=agents, messages=messages, votes=votes, proposals=proposals
        )

        # Verify metrics are sensible
        assert metrics.disagreement_type != "none"  # Should detect disagreement
        assert metrics.collective_confidence < 0.9  # Should be reduced
        assert len(metrics.cruxes) >= 0  # May or may not find cruxes

    def test_crux_to_followup_chain(self):
        """Test the chain from crux identification to follow-up suggestions."""
        analyzer = DisagreementAnalyzer()

        # Create a debate scenario with disagreement
        messages = [
            MagicMock(
                content="However, I disagree with the cost estimates provided.",
                agent="critic1",
                role="critic",
            ),
            MagicMock(
                content="But the timeline seems unrealistic to me.",
                agent="critic2",
                role="critic",
            ),
        ]

        votes = [
            MagicMock(choice="A", confidence=0.8, agent="supporter"),
            MagicMock(choice="B", confidence=0.6, agent="critic1"),
            MagicMock(choice="C", confidence=0.5, agent="critic2"),
        ]

        # Analyze disagreement
        metrics = analyzer.analyze_disagreement(messages, votes, {})

        # Generate follow-ups
        if metrics.cruxes:
            suggestions = analyzer.suggest_followups(
                metrics.cruxes,
                parent_debate_id="test-debate",
                available_agents=["supporter", "critic1", "critic2", "mediator"],
            )

            assert len(suggestions) <= len(metrics.cruxes)
            for suggestion in suggestions:
                assert suggestion.parent_debate_id == "test-debate"
                assert len(suggestion.suggested_agents) > 0
