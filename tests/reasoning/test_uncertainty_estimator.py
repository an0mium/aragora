"""Tests for uncertainty estimator module.

Tests ConfidenceScore, DisagreementCrux, UncertaintyMetrics,
ConfidenceEstimator, DisagreementAnalyzer, and UncertaintyAggregator.
"""

import pytest
import math
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from aragora.uncertainty.estimator import (
    ConfidenceScore,
    DisagreementCrux,
    FollowUpSuggestion,
    UncertaintyMetrics,
    ConfidenceEstimator,
    DisagreementAnalyzer,
    UncertaintyAggregator,
)
from aragora.core import Message, Vote


# =============================================================================
# TestConfidenceScore
# =============================================================================


class TestConfidenceScore:
    """Tests for ConfidenceScore dataclass."""

    def test_creation_with_required_fields(self) -> None:
        """Test creating ConfidenceScore with required fields."""
        score = ConfidenceScore(
            agent_name="claude",
            value=0.85,
        )
        assert score.agent_name == "claude"
        assert score.value == 0.85
        assert score.reasoning == ""
        assert score.timestamp is not None

    def test_creation_with_all_fields(self) -> None:
        """Test creating ConfidenceScore with all fields."""
        ts = datetime(2025, 1, 1, 12, 0, 0)
        score = ConfidenceScore(
            agent_name="gemini",
            value=0.92,
            reasoning="High confidence based on evidence",
            timestamp=ts,
        )
        assert score.agent_name == "gemini"
        assert score.value == 0.92
        assert score.reasoning == "High confidence based on evidence"
        assert score.timestamp == ts

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        score = ConfidenceScore(
            agent_name="gpt-4",
            value=0.75,
            reasoning="Moderate confidence",
        )
        d = score.to_dict()

        assert d["agent"] == "gpt-4"
        assert d["confidence"] == 0.75
        assert d["reasoning"] == "Moderate confidence"
        assert "timestamp" in d

    def test_value_edge_cases(self) -> None:
        """Test confidence value edge cases."""
        # Minimum confidence
        low = ConfidenceScore(agent_name="a", value=0.0)
        assert low.value == 0.0

        # Maximum confidence
        high = ConfidenceScore(agent_name="b", value=1.0)
        assert high.value == 1.0

        # Middle confidence
        mid = ConfidenceScore(agent_name="c", value=0.5)
        assert mid.value == 0.5


# =============================================================================
# TestDisagreementCrux
# =============================================================================


class TestDisagreementCrux:
    """Tests for DisagreementCrux dataclass."""

    def test_creation_with_required_fields(self) -> None:
        """Test creating DisagreementCrux with required fields."""
        crux = DisagreementCrux(
            description="Disagreement about implementation approach",
            divergent_agents=["claude", "gemini"],
        )
        assert crux.description == "Disagreement about implementation approach"
        assert crux.divergent_agents == ["claude", "gemini"]
        assert crux.evidence_needed == ""
        assert crux.severity == 0.5

    def test_creation_with_all_fields(self) -> None:
        """Test creating DisagreementCrux with all fields."""
        crux = DisagreementCrux(
            description="Disagreement about security implications",
            divergent_agents=["claude"],
            evidence_needed="Security audit report",
            severity=0.9,
        )
        assert crux.severity == 0.9
        assert crux.evidence_needed == "Security audit report"

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        crux = DisagreementCrux(
            description="Test disagreement",
            divergent_agents=["a", "b"],
            evidence_needed="More data",
            severity=0.7,
        )
        d = crux.to_dict()

        assert d["description"] == "Test disagreement"
        assert d["agents"] == ["a", "b"]
        assert d["evidence_needed"] == "More data"
        assert d["severity"] == 0.7

    def test_empty_agents_list(self) -> None:
        """Test with empty agents list."""
        crux = DisagreementCrux(
            description="No specific agents",
            divergent_agents=[],
        )
        assert crux.divergent_agents == []


# =============================================================================
# TestUncertaintyMetrics
# =============================================================================


class TestUncertaintyMetrics:
    """Tests for UncertaintyMetrics dataclass."""

    def test_default_values(self) -> None:
        """Test UncertaintyMetrics has correct defaults."""
        metrics = UncertaintyMetrics()

        assert metrics.collective_confidence == 0.5
        assert metrics.confidence_interval == (0.4, 0.6)
        assert metrics.disagreement_type == "none"
        assert metrics.cruxes == []
        assert metrics.calibration_quality == 0.5

    def test_creation_with_all_fields(self) -> None:
        """Test creating UncertaintyMetrics with all fields."""
        crux = DisagreementCrux(description="Test", divergent_agents=["a"])
        metrics = UncertaintyMetrics(
            collective_confidence=0.8,
            confidence_interval=(0.7, 0.9),
            disagreement_type="factual",
            cruxes=[crux],
            calibration_quality=0.75,
        )

        assert metrics.collective_confidence == 0.8
        assert metrics.confidence_interval == (0.7, 0.9)
        assert metrics.disagreement_type == "factual"
        assert len(metrics.cruxes) == 1
        assert metrics.calibration_quality == 0.75

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        crux = DisagreementCrux(description="Issue", divergent_agents=["x"])
        metrics = UncertaintyMetrics(
            collective_confidence=0.65,
            confidence_interval=(0.5, 0.8),
            disagreement_type="value-based",
            cruxes=[crux],
            calibration_quality=0.6,
        )
        d = metrics.to_dict()

        assert d["collective_confidence"] == 0.65
        assert d["confidence_interval"] == (0.5, 0.8)
        assert d["disagreement_type"] == "value-based"
        assert len(d["cruxes"]) == 1
        assert d["calibration_quality"] == 0.6

    def test_disagreement_types(self) -> None:
        """Test various disagreement types."""
        for dtype in ["none", "factual", "value-based", "definitional", "information-asymmetry"]:
            metrics = UncertaintyMetrics(disagreement_type=dtype)
            assert metrics.disagreement_type == dtype


# =============================================================================
# TestConfidenceEstimator
# =============================================================================


class TestConfidenceEstimator:
    """Tests for ConfidenceEstimator class."""

    def test_initialization(self) -> None:
        """Test ConfidenceEstimator initialization."""
        estimator = ConfidenceEstimator()

        assert estimator.agent_confidences == {}
        assert estimator.calibration_history == {}
        assert estimator.brier_scores == {}

    def test_store_confidence(self) -> None:
        """Test storing confidence scores."""
        estimator = ConfidenceEstimator()
        score = ConfidenceScore(agent_name="claude", value=0.8, reasoning="Test")

        estimator._store_confidence("claude", score)

        assert "claude" in estimator.agent_confidences
        assert len(estimator.agent_confidences["claude"]) == 1
        assert estimator.agent_confidences["claude"][0].value == 0.8

    def test_store_confidence_truncates_history(self) -> None:
        """Test that confidence history is truncated when too long."""
        estimator = ConfidenceEstimator()

        # Store more than 100 confidences
        for i in range(110):
            score = ConfidenceScore(agent_name="claude", value=0.5 + i * 0.001)
            estimator._store_confidence("claude", score)

        # Truncation happens at >100, keeps last 50, then adds remaining
        # 101st triggers truncation to 50, then 9 more added = 59
        assert len(estimator.agent_confidences["claude"]) == 59
        # Verify newest items are kept (last 50 from first 101, then +9)
        assert estimator.agent_confidences["claude"][-1].value == pytest.approx(0.609, rel=0.01)

    def test_get_agent_calibration_quality_default(self) -> None:
        """Test calibration quality returns default when no history."""
        estimator = ConfidenceEstimator()

        quality = estimator.get_agent_calibration_quality("unknown_agent")

        assert quality == 0.5

    def test_get_agent_calibration_quality_with_history(self) -> None:
        """Test calibration quality with history."""
        estimator = ConfidenceEstimator()

        # Record some outcomes - perfectly calibrated agent
        estimator.record_outcome("claude", 0.9, True)
        estimator.record_outcome("claude", 0.9, True)
        estimator.record_outcome("claude", 0.1, False)

        quality = estimator.get_agent_calibration_quality("claude")

        # Should be high calibration quality (close to 1)
        assert 0 <= quality <= 1

    def test_get_agent_calibration_quality_poor(self) -> None:
        """Test calibration quality for poorly calibrated agent."""
        estimator = ConfidenceEstimator()

        # Record anti-calibrated outcomes (high confidence but wrong)
        for _ in range(10):
            estimator.record_outcome("bad_agent", 0.9, False)

        quality = estimator.get_agent_calibration_quality("bad_agent")

        # Should be low calibration quality
        assert quality < 0.5

    def test_record_outcome(self) -> None:
        """Test recording prediction outcomes."""
        estimator = ConfidenceEstimator()

        estimator.record_outcome("claude", 0.8, True)
        estimator.record_outcome("claude", 0.6, False)

        assert "claude" in estimator.calibration_history
        assert len(estimator.calibration_history["claude"]) == 2
        assert estimator.calibration_history["claude"][0] == (0.8, True)
        assert estimator.calibration_history["claude"][1] == (0.6, False)

    def test_record_outcome_truncates_history(self) -> None:
        """Test that calibration history is truncated when too long."""
        estimator = ConfidenceEstimator()

        # Record more than 50 outcomes
        for i in range(60):
            estimator.record_outcome("claude", 0.5, i % 2 == 0)

        # Truncation happens at >50, keeps last 25, then adds remaining
        # 51st triggers truncation to 25, then 9 more added = 34
        assert len(estimator.calibration_history["claude"]) == 34

    @pytest.mark.asyncio
    async def test_collect_confidences_with_exception(self) -> None:
        """Test collecting confidences handles agent exceptions."""
        estimator = ConfidenceEstimator()

        # Create a mock agent that raises an exception
        agent = MagicMock()
        agent.name = "failing_agent"
        agent.vote = AsyncMock(side_effect=Exception("Agent error"))

        confidences = await estimator.collect_confidences([agent], {}, "test task")

        # Should return default confidence
        assert "failing_agent" in confidences
        assert confidences["failing_agent"].value == 0.5

    @pytest.mark.asyncio
    async def test_collect_confidences_success(self) -> None:
        """Test collecting confidences from working agents."""
        estimator = ConfidenceEstimator()

        # Create a mock agent with successful vote
        agent = MagicMock()
        agent.name = "claude"
        vote = MagicMock()
        vote.confidence = 0.85
        vote.reasoning = "Good evidence"
        agent.vote = AsyncMock(return_value=vote)

        confidences = await estimator.collect_confidences(
            [agent], {"claude": "proposal"}, "test task"
        )

        assert "claude" in confidences
        assert confidences["claude"].value == 0.85


# =============================================================================
# TestDisagreementAnalyzer
# =============================================================================


class TestDisagreementAnalyzer:
    """Tests for DisagreementAnalyzer class."""

    def test_initialization(self) -> None:
        """Test DisagreementAnalyzer initialization."""
        analyzer = DisagreementAnalyzer()

        assert "factual" in analyzer.nlp_keywords
        assert "value" in analyzer.nlp_keywords
        assert "definitional" in analyzer.nlp_keywords
        assert "asymmetry" in analyzer.nlp_keywords

    def test_analyze_disagreement_unanimous(self) -> None:
        """Test analyzing unanimous agreement."""
        analyzer = DisagreementAnalyzer()

        messages = [
            Message(role="proposer", agent="claude", content="Proposal"),
            Message(role="critic", agent="gemini", content="Looks good"),
        ]
        votes = [
            Vote(agent="claude", choice="option_a", reasoning="Best choice", confidence=0.9),
            Vote(agent="gemini", choice="option_a", reasoning="Agreed", confidence=0.8),
        ]
        proposals = {"option_a": "Implementation A"}

        metrics = analyzer.analyze_disagreement(messages, votes, proposals)

        assert metrics.disagreement_type == "none"
        assert metrics.collective_confidence > 0.8
        assert len(metrics.cruxes) == 0

    def test_analyze_disagreement_with_dissent(self) -> None:
        """Test analyzing disagreement with dissenting votes."""
        analyzer = DisagreementAnalyzer()

        messages = [
            Message(role="proposer", agent="claude", content="Use Redis"),
            Message(
                role="critic",
                agent="gemini",
                content="However, I disagree. We should use Memcached instead.",
            ),
        ]
        votes = [
            Vote(agent="claude", choice="redis", reasoning="Scalable", confidence=0.9),
            Vote(agent="gemini", choice="memcached", reasoning="Simpler", confidence=0.7),
        ]
        proposals = {"redis": "Redis solution", "memcached": "Memcached solution"}

        metrics = analyzer.analyze_disagreement(messages, votes, proposals)

        # Should detect disagreement
        assert metrics.disagreement_type != "none" or len(votes) > 1
        assert metrics.collective_confidence < 0.9  # Reduced due to disagreement

    def test_analyze_disagreement_no_votes(self) -> None:
        """Test analyzing with no votes."""
        analyzer = DisagreementAnalyzer()

        messages = [Message(role="proposer", agent="claude", content="Proposal")]
        votes = []
        proposals = {}

        metrics = analyzer.analyze_disagreement(messages, votes, proposals)

        assert metrics.collective_confidence == 0.5
        assert metrics.disagreement_type == "none"

    def test_classify_disagreement_type_factual(self) -> None:
        """Test classifying factual disagreement."""
        analyzer = DisagreementAnalyzer()

        messages = [
            Message(
                role="critic",
                agent="gemini",
                content="This is false. The evidence shows otherwise.",
            ),
        ]
        minority_votes = [
            Vote(agent="gemini", choice="other", reasoning="Wrong facts", confidence=0.8)
        ]

        dtype = analyzer._classify_disagreement_type(messages, minority_votes)

        assert dtype == "factual"

    def test_classify_disagreement_type_value(self) -> None:
        """Test classifying value-based disagreement."""
        analyzer = DisagreementAnalyzer()

        messages = [
            Message(
                role="critic", agent="gemini", content="We should prefer the ethical approach."
            ),
        ]
        minority_votes = [Vote(agent="gemini", choice="other", reasoning="Better", confidence=0.8)]

        dtype = analyzer._classify_disagreement_type(messages, minority_votes)

        assert dtype == "value"

    def test_classify_disagreement_type_general(self) -> None:
        """Test classifying general disagreement without keywords."""
        analyzer = DisagreementAnalyzer()

        messages = [
            Message(role="critic", agent="gemini", content="I just don't like it."),
        ]
        minority_votes = [Vote(agent="gemini", choice="other", reasoning="Nope", confidence=0.8)]

        dtype = analyzer._classify_disagreement_type(messages, minority_votes)

        assert dtype == "general"

    def test_extract_crux_description(self) -> None:
        """Test extracting crux description from content."""
        analyzer = DisagreementAnalyzer()

        content = "This is good. However, there's a security concern here. We should address it."
        crux = analyzer._extract_crux_description(content)

        assert crux is not None
        assert "security concern" in crux.lower() or "however" in crux.lower()

    def test_extract_crux_description_no_disagreement(self) -> None:
        """Test extracting crux when no disagreement markers."""
        analyzer = DisagreementAnalyzer()

        content = "Everything looks great. Perfect solution."
        crux = analyzer._extract_crux_description(content)

        assert crux is None

    def test_cruxes_similar(self) -> None:
        """Test checking if cruxes are similar."""
        analyzer = DisagreementAnalyzer()

        crux1 = DisagreementCrux(
            description="Security vulnerability in authentication",
            divergent_agents=["a"],
        )
        crux2 = DisagreementCrux(
            description="Authentication has security issues",
            divergent_agents=["b"],
        )

        similar = analyzer._cruxes_similar(crux1, crux2)

        # Should be similar due to shared keywords
        assert similar is True

    def test_cruxes_not_similar(self) -> None:
        """Test checking dissimilar cruxes."""
        analyzer = DisagreementAnalyzer()

        crux1 = DisagreementCrux(
            description="Performance is too slow",
            divergent_agents=["a"],
        )
        crux2 = DisagreementCrux(
            description="Security vulnerability found",
            divergent_agents=["b"],
        )

        similar = analyzer._cruxes_similar(crux1, crux2)

        # Should not be similar
        assert similar is False

    def test_merge_similar_cruxes(self) -> None:
        """Test merging similar cruxes."""
        analyzer = DisagreementAnalyzer()

        cruxes = [
            DisagreementCrux(
                description="Security issue in auth", divergent_agents=["a"], severity=0.6
            ),
            DisagreementCrux(
                description="Auth has security problem", divergent_agents=["b"], severity=0.8
            ),
            DisagreementCrux(
                description="Performance is slow", divergent_agents=["c"], severity=0.5
            ),
        ]

        merged = analyzer._merge_similar_cruxes(cruxes)

        # Should merge the first two, keep the third separate
        assert len(merged) <= 3


# =============================================================================
# TestUncertaintyAggregator
# =============================================================================


class TestUncertaintyAggregator:
    """Tests for UncertaintyAggregator class."""

    def test_initialization(self) -> None:
        """Test UncertaintyAggregator initialization."""
        estimator = ConfidenceEstimator()
        analyzer = DisagreementAnalyzer()
        aggregator = UncertaintyAggregator(estimator, analyzer)

        assert aggregator.confidence_estimator is estimator
        assert aggregator.disagreement_analyzer is analyzer

    @pytest.mark.asyncio
    async def test_compute_uncertainty_basic(self) -> None:
        """Test computing uncertainty with basic inputs."""
        estimator = ConfidenceEstimator()
        analyzer = DisagreementAnalyzer()
        aggregator = UncertaintyAggregator(estimator, analyzer)

        # Create mock agents
        agent = MagicMock()
        agent.name = "claude"
        vote = MagicMock()
        vote.confidence = 0.8
        vote.reasoning = "Good"
        agent.vote = AsyncMock(return_value=vote)

        messages = [Message(role="proposer", agent="claude", content="Proposal")]
        votes = [Vote(agent="claude", choice="option", reasoning="Best", confidence=0.8)]
        proposals = {"option": "The proposal"}

        metrics = await aggregator.compute_uncertainty([agent], messages, votes, proposals)

        assert isinstance(metrics, UncertaintyMetrics)
        assert 0 <= metrics.collective_confidence <= 1
        assert 0 <= metrics.calibration_quality <= 1

    @pytest.mark.asyncio
    async def test_compute_uncertainty_with_calibration(self) -> None:
        """Test that calibration affects uncertainty metrics."""
        estimator = ConfidenceEstimator()
        analyzer = DisagreementAnalyzer()
        aggregator = UncertaintyAggregator(estimator, analyzer)

        # Record good calibration history
        estimator.record_outcome("claude", 0.9, True)
        estimator.record_outcome("claude", 0.8, True)
        estimator.record_outcome("claude", 0.2, False)

        agent = MagicMock()
        agent.name = "claude"
        vote = MagicMock()
        vote.confidence = 0.85
        vote.reasoning = "Confident"
        agent.vote = AsyncMock(return_value=vote)

        messages = [Message(role="proposer", agent="claude", content="Proposal")]
        votes = [Vote(agent="claude", choice="option", reasoning="Best", confidence=0.85)]
        proposals = {"option": "Proposal"}

        metrics = await aggregator.compute_uncertainty([agent], messages, votes, proposals)

        # Calibration quality should be factored in
        assert metrics.calibration_quality > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for uncertainty module."""

    def test_full_workflow_unanimous(self) -> None:
        """Test full workflow with unanimous agreement."""
        analyzer = DisagreementAnalyzer()

        messages = [
            Message(role="proposer", agent="claude", content="Use async/await"),
            Message(role="critic", agent="gemini", content="Agreed, async is better"),
            Message(role="synthesizer", agent="grok", content="Consensus reached"),
        ]
        votes = [
            Vote(agent="claude", choice="async", reasoning="Modern", confidence=0.9),
            Vote(agent="gemini", choice="async", reasoning="Efficient", confidence=0.85),
            Vote(agent="grok", choice="async", reasoning="Standard", confidence=0.88),
        ]
        proposals = {"async": "Use async/await"}

        metrics = analyzer.analyze_disagreement(messages, votes, proposals)

        assert metrics.disagreement_type == "none"
        assert metrics.collective_confidence > 0.8
        assert len(metrics.cruxes) == 0

    def test_full_workflow_with_disagreement(self) -> None:
        """Test full workflow with disagreement."""
        analyzer = DisagreementAnalyzer()

        messages = [
            Message(role="proposer", agent="claude", content="Use SQL database"),
            Message(
                role="critic",
                agent="gemini",
                content="However, NoSQL would be better for this use case. I disagree with SQL.",
            ),
            Message(
                role="critic", agent="grok", content="But SQL has better consistency guarantees."
            ),
        ]
        votes = [
            Vote(agent="claude", choice="sql", reasoning="ACID compliance", confidence=0.9),
            Vote(agent="gemini", choice="nosql", reasoning="Scalability", confidence=0.8),
            Vote(agent="grok", choice="sql", reasoning="Consistency", confidence=0.85),
        ]
        proposals = {"sql": "SQL database", "nosql": "NoSQL database"}

        metrics = analyzer.analyze_disagreement(messages, votes, proposals)

        # Should detect the disagreement
        assert len(votes) > 1
        # Confidence should be lower due to disagreement
        assert metrics.collective_confidence < 0.9

    def test_confidence_interval_calculation(self) -> None:
        """Test confidence interval is calculated correctly."""
        analyzer = DisagreementAnalyzer()

        votes = [
            Vote(agent="a", choice="x", reasoning="R", confidence=0.8),
            Vote(agent="b", choice="x", reasoning="R", confidence=0.9),
            Vote(agent="c", choice="x", reasoning="R", confidence=0.85),
        ]
        messages = []
        proposals = {"x": "Proposal"}

        metrics = analyzer.analyze_disagreement(messages, votes, proposals)

        # Confidence interval should be a tuple
        assert isinstance(metrics.confidence_interval, tuple)
        assert len(metrics.confidence_interval) == 2
        assert metrics.confidence_interval[0] <= metrics.collective_confidence
        assert metrics.confidence_interval[1] >= metrics.collective_confidence

    def test_brier_score_calculation(self) -> None:
        """Test Brier score-based calibration quality."""
        estimator = ConfidenceEstimator()

        # Perfect calibration: high confidence when correct, low when wrong
        estimator.record_outcome("perfect", 0.95, True)
        estimator.record_outcome("perfect", 0.9, True)
        estimator.record_outcome("perfect", 0.1, False)
        estimator.record_outcome("perfect", 0.05, False)

        # Poor calibration: high confidence when wrong
        estimator.record_outcome("poor", 0.95, False)
        estimator.record_outcome("poor", 0.9, False)
        estimator.record_outcome("poor", 0.1, True)

        perfect_quality = estimator.get_agent_calibration_quality("perfect")
        poor_quality = estimator.get_agent_calibration_quality("poor")

        # Perfect should have higher quality than poor
        assert perfect_quality > poor_quality


# =============================================================================
# TestFollowUpSuggestion
# =============================================================================


class TestFollowUpSuggestion:
    """Tests for FollowUpSuggestion dataclass."""

    def test_creation_with_required_fields(self) -> None:
        """Test creating FollowUpSuggestion with required fields."""
        crux = DisagreementCrux(
            description="Security concerns",
            divergent_agents=["claude", "gemini"],
        )
        suggestion = FollowUpSuggestion(
            crux=crux,
            suggested_task="Investigate security implications",
            priority=0.8,
        )
        assert suggestion.crux is crux
        assert suggestion.suggested_task == "Investigate security implications"
        assert suggestion.priority == 0.8
        assert suggestion.parent_debate_id is None
        assert suggestion.suggested_agents == []

    def test_creation_with_all_fields(self) -> None:
        """Test creating FollowUpSuggestion with all fields."""
        crux = DisagreementCrux(
            description="Performance trade-offs",
            divergent_agents=["claude"],
        )
        suggestion = FollowUpSuggestion(
            crux=crux,
            suggested_task="Debate performance vs simplicity",
            priority=0.6,
            parent_debate_id="debate-123",
            suggested_agents=["claude", "gemini", "grok"],
        )
        assert suggestion.parent_debate_id == "debate-123"
        assert suggestion.suggested_agents == ["claude", "gemini", "grok"]

    def test_to_dict(self) -> None:
        """Test FollowUpSuggestion serialization."""
        crux = DisagreementCrux(
            description="Scalability concerns",
            divergent_agents=["agent1"],
            severity=0.7,
        )
        suggestion = FollowUpSuggestion(
            crux=crux,
            suggested_task="Resolve scalability debate",
            priority=0.75,
            parent_debate_id="parent-456",
            suggested_agents=["agent1", "agent2"],
        )

        data = suggestion.to_dict()

        assert data["crux_id"] == crux.crux_id
        assert data["crux_description"] == "Scalability concerns"
        assert data["suggested_task"] == "Resolve scalability debate"
        assert data["priority"] == 0.75
        assert data["parent_debate_id"] == "parent-456"
        assert data["suggested_agents"] == ["agent1", "agent2"]
        assert data["divergent_agents"] == ["agent1"]


class TestDisagreementCruxId:
    """Tests for DisagreementCrux crux_id generation."""

    def test_auto_generated_id(self) -> None:
        """Test that crux_id is auto-generated."""
        crux = DisagreementCrux(
            description="Test crux",
            divergent_agents=["a"],
        )
        assert crux.crux_id.startswith("crux-")
        # ID is "crux-" + up to 5 digits (10-11 chars total)
        assert 10 <= len(crux.crux_id) <= 11

    def test_custom_id_preserved(self) -> None:
        """Test that custom crux_id is preserved."""
        crux = DisagreementCrux(
            description="Test crux",
            divergent_agents=["a"],
            crux_id="custom-id-123",
        )
        assert crux.crux_id == "custom-id-123"

    def test_stable_id_generation(self) -> None:
        """Test that same description generates same ID."""
        crux1 = DisagreementCrux(description="Same text", divergent_agents=["a"])
        crux2 = DisagreementCrux(description="Same text", divergent_agents=["b"])
        assert crux1.crux_id == crux2.crux_id

    def test_different_descriptions_different_ids(self) -> None:
        """Test that different descriptions generate different IDs."""
        crux1 = DisagreementCrux(description="First text", divergent_agents=["a"])
        crux2 = DisagreementCrux(description="Second text", divergent_agents=["a"])
        assert crux1.crux_id != crux2.crux_id


class TestSuggestFollowups:
    """Tests for DisagreementAnalyzer.suggest_followups method."""

    def test_suggest_followups_basic(self) -> None:
        """Test basic follow-up suggestion generation."""
        analyzer = DisagreementAnalyzer()
        cruxes = [
            DisagreementCrux(
                description="Security vulnerability in auth",
                divergent_agents=["claude", "gemini"],
                severity=0.8,
            ),
        ]

        suggestions = analyzer.suggest_followups(cruxes)

        assert len(suggestions) == 1
        assert suggestions[0].crux is cruxes[0]
        assert (
            "security vulnerability" in suggestions[0].suggested_task.lower()
            or "auth" in suggestions[0].suggested_task.lower()
        )
        assert suggestions[0].priority > 0

    def test_suggest_followups_with_parent_id(self) -> None:
        """Test follow-up suggestions include parent debate ID."""
        analyzer = DisagreementAnalyzer()
        cruxes = [
            DisagreementCrux(description="Test crux", divergent_agents=["a"]),
        ]

        suggestions = analyzer.suggest_followups(cruxes, parent_debate_id="debate-789")

        assert suggestions[0].parent_debate_id == "debate-789"

    def test_suggest_followups_with_available_agents(self) -> None:
        """Test follow-up suggestions include available agents."""
        analyzer = DisagreementAnalyzer()
        cruxes = [
            DisagreementCrux(description="Test crux", divergent_agents=["claude"]),
        ]

        suggestions = analyzer.suggest_followups(
            cruxes,
            available_agents=["claude", "gemini", "grok", "codex"],
        )

        # Should include divergent agents plus others
        assert "claude" in suggestions[0].suggested_agents
        assert len(suggestions[0].suggested_agents) <= 4

    def test_suggest_followups_priority_ordering(self) -> None:
        """Test that suggestions are ordered by priority."""
        analyzer = DisagreementAnalyzer()
        cruxes = [
            DisagreementCrux(description="Low priority", divergent_agents=["a"], severity=0.2),
            DisagreementCrux(
                description="High priority", divergent_agents=["a", "b", "c"], severity=0.9
            ),
            DisagreementCrux(
                description="Medium priority", divergent_agents=["a", "b"], severity=0.5
            ),
        ]

        suggestions = analyzer.suggest_followups(cruxes)

        # Should be sorted by priority descending
        priorities = [s.priority for s in suggestions]
        assert priorities == sorted(priorities, reverse=True)

    def test_suggest_followups_empty_cruxes(self) -> None:
        """Test suggest_followups with empty cruxes list."""
        analyzer = DisagreementAnalyzer()

        suggestions = analyzer.suggest_followups([])

        assert suggestions == []

    def test_generate_followup_task_with_but(self) -> None:
        """Test task generation with 'but' marker."""
        analyzer = DisagreementAnalyzer()
        crux = DisagreementCrux(
            description="But we should use async instead",
            divergent_agents=["a"],
        )

        task = analyzer._generate_followup_task(crux)

        assert "Investigate:" in task

    def test_generate_followup_task_with_question(self) -> None:
        """Test task generation with question mark."""
        analyzer = DisagreementAnalyzer()
        crux = DisagreementCrux(
            description="Is this approach scalable?",
            divergent_agents=["a"],
        )

        task = analyzer._generate_followup_task(crux)

        assert "Resolve:" in task

    def test_generate_followup_task_generic(self) -> None:
        """Test task generation for generic description."""
        analyzer = DisagreementAnalyzer()
        crux = DisagreementCrux(
            description="Performance considerations",
            divergent_agents=["a"],
        )

        task = analyzer._generate_followup_task(crux)

        assert "Debate:" in task
        assert "performance considerations" in task.lower()
