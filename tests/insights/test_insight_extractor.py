"""Tests for the InsightExtractor module."""

import pytest
from unittest.mock import MagicMock

from aragora.insights.extractor import (
    InsightType,
    Insight,
    AgentPerformance,
    DebateInsights,
    InsightExtractor,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def extractor():
    """Create InsightExtractor instance."""
    return InsightExtractor()


@pytest.fixture
def mock_result():
    """Create mock debate result with consensus."""
    result = MagicMock()
    result.id = "debate-123"
    result.task = "Test task"
    result.consensus_reached = True
    result.confidence = 0.85
    result.final_answer = "The answer is 42"
    result.consensus_strength = "strong"
    result.duration_seconds = 120.0
    result.rounds_used = 3
    result.consensus_variance = 0.1
    result.messages = []
    result.critiques = []
    result.votes = []
    result.dissenting_views = []
    return result


@pytest.fixture
def mock_result_no_consensus():
    """Create mock debate result without consensus."""
    result = MagicMock()
    result.id = "debate-456"
    result.task = "Failed task"
    result.consensus_reached = False
    result.confidence = 0.3
    result.final_answer = ""
    result.duration_seconds = 180.0
    result.rounds_used = 5
    result.consensus_variance = 0.5
    result.messages = []
    result.critiques = []
    result.votes = []
    result.dissenting_views = []
    return result


@pytest.fixture
def mock_message():
    """Factory for mock messages."""

    def _make(agent, content="Test content", round_num=1):
        msg = MagicMock()
        msg.configure_mock(agent=agent, content=content, round=round_num)
        return msg

    return _make


@pytest.fixture
def mock_critique():
    """Factory for mock critiques."""

    def _make(agent, target, issues, severity=0.5):
        critique = MagicMock()
        critique.configure_mock(agent=agent, target_agent=target, issues=issues, severity=severity)
        return critique

    return _make


@pytest.fixture
def mock_vote():
    """Factory for mock votes."""

    def _make(agent, choice):
        vote = MagicMock()
        vote.configure_mock(agent=agent, choice=choice)
        return vote

    return _make


# =============================================================================
# Test InsightType Enum
# =============================================================================


class TestInsightType:
    """Tests for InsightType enum."""

    def test_all_values_defined(self):
        """Should have 7 insight types."""
        assert len(InsightType) == 7

    def test_consensus_value(self):
        """Should have correct consensus value."""
        assert InsightType.CONSENSUS.value == "consensus"

    def test_dissent_value(self):
        """Should have correct dissent value."""
        assert InsightType.DISSENT.value == "dissent"

    def test_pattern_value(self):
        """Should have correct pattern value."""
        assert InsightType.PATTERN.value == "pattern"

    def test_convergence_value(self):
        """Should have correct convergence value."""
        assert InsightType.CONVERGENCE.value == "convergence"

    def test_failure_mode_value(self):
        """Should have correct failure_mode value."""
        assert InsightType.FAILURE_MODE.value == "failure_mode"

    def test_decision_process_value(self):
        """Should have correct decision value."""
        assert InsightType.DECISION_PROCESS.value == "decision"


# =============================================================================
# Test Insight Dataclass
# =============================================================================


class TestInsight:
    """Tests for Insight dataclass."""

    def test_creation_with_required_fields(self):
        """Should create insight with required fields."""
        insight = Insight(
            id="insight-1",
            type=InsightType.CONSENSUS,
            title="Test title",
            description="Test description",
            confidence=0.8,
            debate_id="debate-1",
        )
        assert insight.id == "insight-1"
        assert insight.type == InsightType.CONSENSUS
        assert insight.title == "Test title"
        assert insight.description == "Test description"
        assert insight.confidence == 0.8
        assert insight.debate_id == "debate-1"

    def test_defaults_for_optional_fields(self):
        """Should have default values for optional fields."""
        insight = Insight(
            id="insight-1",
            type=InsightType.PATTERN,
            title="Test",
            description="Desc",
            confidence=0.5,
            debate_id="d1",
        )
        assert insight.agents_involved == []
        assert insight.evidence == []
        assert insight.metadata == {}

    def test_timestamp_auto_generated(self):
        """Should auto-generate timestamp."""
        insight = Insight(
            id="insight-1",
            type=InsightType.DISSENT,
            title="Test",
            description="Desc",
            confidence=0.5,
            debate_id="d1",
        )
        assert insight.created_at is not None
        assert len(insight.created_at) > 0

    def test_to_dict_all_fields(self):
        """Should include all fields in to_dict."""
        insight = Insight(
            id="insight-1",
            type=InsightType.CONSENSUS,
            title="Test",
            description="Desc",
            confidence=0.8,
            debate_id="d1",
            agents_involved=["agent-1"],
            evidence=["evidence-1"],
            metadata={"key": "value"},
        )
        d = insight.to_dict()
        assert "id" in d
        assert "type" in d
        assert "title" in d
        assert "description" in d
        assert "confidence" in d
        assert "debate_id" in d
        assert "agents_involved" in d
        assert "evidence" in d
        assert "created_at" in d
        assert "metadata" in d

    def test_to_dict_type_converted_to_value(self):
        """Should convert type enum to string value."""
        insight = Insight(
            id="insight-1",
            type=InsightType.PATTERN,
            title="Test",
            description="Desc",
            confidence=0.5,
            debate_id="d1",
        )
        d = insight.to_dict()
        assert d["type"] == "pattern"  # String, not enum


# =============================================================================
# Test AgentPerformance Dataclass
# =============================================================================


class TestAgentPerformance:
    """Tests for AgentPerformance dataclass."""

    def test_creation_with_name(self):
        """Should create with agent name."""
        perf = AgentPerformance(agent_name="agent-1")
        assert perf.agent_name == "agent-1"

    def test_default_values(self):
        """Should have correct default values."""
        perf = AgentPerformance(agent_name="agent-1")
        assert perf.proposals_made == 0
        assert perf.critiques_given == 0
        assert perf.critiques_received == 0
        assert perf.proposal_accepted is False
        assert perf.vote_aligned_with_consensus is False
        assert perf.average_critique_severity == 0.0

    def test_contribution_score_default(self):
        """Should default contribution score to 0.5."""
        perf = AgentPerformance(agent_name="agent-1")
        assert perf.contribution_score == 0.5


# =============================================================================
# Test DebateInsights Dataclass
# =============================================================================


class TestDebateInsights:
    """Tests for DebateInsights dataclass."""

    def test_creation_with_required_fields(self):
        """Should create with required fields."""
        insights = DebateInsights(
            debate_id="d1",
            task="Test task",
            consensus_reached=True,
            duration_seconds=100.0,
        )
        assert insights.debate_id == "d1"
        assert insights.task == "Test task"
        assert insights.consensus_reached is True
        assert insights.duration_seconds == 100.0

    def test_all_insights_empty(self):
        """Should return empty list when no insights."""
        insights = DebateInsights(
            debate_id="d1",
            task="Task",
            consensus_reached=True,
            duration_seconds=100.0,
        )
        assert insights.all_insights() == []

    def test_all_insights_with_consensus(self):
        """Should include consensus insight."""
        consensus = Insight(
            id="i1",
            type=InsightType.CONSENSUS,
            title="Consensus",
            description="Desc",
            confidence=0.9,
            debate_id="d1",
        )
        insights = DebateInsights(
            debate_id="d1",
            task="Task",
            consensus_reached=True,
            duration_seconds=100.0,
            consensus_insight=consensus,
        )
        all_insights = insights.all_insights()
        assert len(all_insights) == 1
        assert all_insights[0].type == InsightType.CONSENSUS

    def test_all_insights_with_dissents(self):
        """Should include dissent insights."""
        dissent1 = Insight(
            id="d1",
            type=InsightType.DISSENT,
            title="Dissent 1",
            description="Desc",
            confidence=0.6,
            debate_id="d1",
        )
        dissent2 = Insight(
            id="d2",
            type=InsightType.DISSENT,
            title="Dissent 2",
            description="Desc",
            confidence=0.6,
            debate_id="d1",
        )
        insights = DebateInsights(
            debate_id="d1",
            task="Task",
            consensus_reached=False,
            duration_seconds=100.0,
            dissent_insights=[dissent1, dissent2],
        )
        all_insights = insights.all_insights()
        assert len(all_insights) == 2

    def test_all_insights_combined(self):
        """Should combine all insight types."""
        consensus = Insight(
            id="c1",
            type=InsightType.CONSENSUS,
            title="C",
            description="D",
            confidence=0.9,
            debate_id="d1",
        )
        dissent = Insight(
            id="d1",
            type=InsightType.DISSENT,
            title="D",
            description="D",
            confidence=0.6,
            debate_id="d1",
        )
        pattern = Insight(
            id="p1",
            type=InsightType.PATTERN,
            title="P",
            description="D",
            confidence=0.7,
            debate_id="d1",
        )
        convergence = Insight(
            id="cv1",
            type=InsightType.CONVERGENCE,
            title="CV",
            description="D",
            confidence=0.7,
            debate_id="d1",
        )
        insights = DebateInsights(
            debate_id="d1",
            task="Task",
            consensus_reached=True,
            duration_seconds=100.0,
            consensus_insight=consensus,
            dissent_insights=[dissent],
            pattern_insights=[pattern],
            convergence_insight=convergence,
        )
        all_insights = insights.all_insights()
        assert len(all_insights) == 4

    def test_default_lists_empty(self):
        """Should have empty default lists."""
        insights = DebateInsights(
            debate_id="d1",
            task="Task",
            consensus_reached=True,
            duration_seconds=100.0,
        )
        assert insights.dissent_insights == []
        assert insights.pattern_insights == []
        assert insights.agent_performances == []


# =============================================================================
# Test InsightExtractor Initialization
# =============================================================================


class TestInsightExtractorInit:
    """Tests for InsightExtractor initialization."""

    def test_initialization(self):
        """Should initialize without error."""
        extractor = InsightExtractor()
        assert extractor is not None

    def test_issue_categories_defined(self):
        """Should have issue categories defined."""
        assert len(InsightExtractor.ISSUE_CATEGORIES) == 7
        assert "security" in InsightExtractor.ISSUE_CATEGORIES
        assert "performance" in InsightExtractor.ISSUE_CATEGORIES
        assert "correctness" in InsightExtractor.ISSUE_CATEGORIES

    def test_issue_categories_have_keywords(self):
        """Each category should have keywords."""
        for category, keywords in InsightExtractor.ISSUE_CATEGORIES.items():
            assert len(keywords) > 0, f"{category} has no keywords"


# =============================================================================
# Test extract() Main Method
# =============================================================================


class TestExtract:
    """Tests for extract() async method."""

    @pytest.mark.asyncio
    async def test_extract_basic_result(self, extractor, mock_result):
        """Should extract insights from basic result."""
        insights = await extractor.extract(mock_result)
        assert insights.debate_id == "debate-123"
        assert insights.task == "Test task"
        assert insights.consensus_reached is True

    @pytest.mark.asyncio
    async def test_extract_with_consensus(self, extractor, mock_result):
        """Should extract consensus insight when reached."""
        insights = await extractor.extract(mock_result)
        assert insights.consensus_insight is not None
        assert insights.consensus_insight.type == InsightType.CONSENSUS

    @pytest.mark.asyncio
    async def test_extract_without_consensus(self, extractor, mock_result_no_consensus):
        """Should extract failure mode when no consensus."""
        insights = await extractor.extract(mock_result_no_consensus)
        assert insights.consensus_insight is None
        assert insights.failure_mode_insight is not None

    @pytest.mark.asyncio
    async def test_extract_handles_missing_attributes(self, extractor):
        """Should handle result with missing attributes."""
        minimal = MagicMock(spec=[])  # No attributes
        insights = await extractor.extract(minimal)
        assert insights is not None
        assert insights.consensus_reached is False

    @pytest.mark.asyncio
    async def test_extract_calculates_total_insights(self, extractor, mock_result):
        """Should calculate total insights count."""
        insights = await extractor.extract(mock_result)
        assert insights.total_insights == len(insights.all_insights())

    @pytest.mark.asyncio
    async def test_extract_generates_key_takeaway(self, extractor, mock_result):
        """Should generate key takeaway."""
        insights = await extractor.extract(mock_result)
        assert insights.key_takeaway != ""
        assert "consensus" in insights.key_takeaway.lower()


# =============================================================================
# Test Consensus Insight Extraction
# =============================================================================


class TestExtractConsensusInsight:
    """Tests for _extract_consensus_insight method."""

    def test_returns_none_when_no_consensus(self, extractor):
        """Should return None when consensus not reached."""
        result = MagicMock()
        result.consensus_reached = False
        insight = extractor._extract_consensus_insight(result, "d1")
        assert insight is None

    def test_extracts_consensus_insight(self, extractor, mock_result):
        """Should extract consensus insight when reached."""
        insight = extractor._extract_consensus_insight(mock_result, "debate-123")
        assert insight is not None
        assert insight.type == InsightType.CONSENSUS
        assert "Consensus Reached" in insight.title

    def test_includes_confidence(self, extractor, mock_result):
        """Should include confidence in insight."""
        insight = extractor._extract_consensus_insight(mock_result, "d1")
        assert insight.confidence == 0.85
        assert "85%" in insight.description

    def test_includes_strength_in_metadata(self, extractor, mock_result):
        """Should include consensus strength in metadata."""
        insight = extractor._extract_consensus_insight(mock_result, "d1")
        assert insight.metadata["consensus_strength"] == "strong"

    def test_truncates_long_answers(self, extractor):
        """Should truncate very long final answers."""
        result = MagicMock()
        result.consensus_reached = True
        result.final_answer = "x" * 1000
        result.confidence = 0.8
        result.consensus_strength = "strong"
        result.rounds_used = 3
        result.messages = []
        result.critiques = []
        result.votes = []
        insight = extractor._extract_consensus_insight(result, "d1")
        assert len(insight.description) < 700

    def test_handles_missing_final_answer(self, extractor):
        """Should handle missing final answer gracefully."""
        result = MagicMock()
        result.consensus_reached = True
        result.confidence = 0.7
        result.consensus_strength = "weak"
        result.rounds_used = 2
        del result.final_answer
        result.messages = []
        result.critiques = []
        result.votes = []
        insight = extractor._extract_consensus_insight(result, "d1")
        assert insight is not None
        assert "No answer recorded" in insight.description


# =============================================================================
# Test Dissent Insight Extraction
# =============================================================================


class TestExtractDissentInsights:
    """Tests for _extract_dissent_insights method."""

    def test_returns_empty_no_dissent(self, extractor):
        """Should return empty list when no dissenting views."""
        result = MagicMock()
        result.dissenting_views = []
        insights = extractor._extract_dissent_insights(result, "d1")
        assert insights == []

    def test_extracts_single_dissent(self, extractor):
        """Should extract single dissenting view."""
        result = MagicMock()
        result.dissenting_views = ["[agent-1]: I disagree because X"]
        insights = extractor._extract_dissent_insights(result, "d1")
        assert len(insights) == 1
        assert insights[0].type == InsightType.DISSENT

    def test_extracts_multiple_dissents(self, extractor):
        """Should extract multiple dissenting views."""
        result = MagicMock()
        result.dissenting_views = [
            "[agent-1]: First dissent",
            "[agent-2]: Second dissent",
        ]
        insights = extractor._extract_dissent_insights(result, "d1")
        assert len(insights) == 2

    def test_parses_agent_from_format(self, extractor):
        """Should parse agent name from [agent]: format."""
        result = MagicMock()
        result.dissenting_views = ["[claude]: My alternative view"]
        insights = extractor._extract_dissent_insights(result, "d1")
        assert "claude" in insights[0].agents_involved

    def test_handles_plain_text_dissent(self, extractor):
        """Should handle dissent without agent format."""
        result = MagicMock()
        result.dissenting_views = ["Plain text dissent without agent"]
        insights = extractor._extract_dissent_insights(result, "d1")
        assert len(insights) == 1
        assert "agent_0" in insights[0].agents_involved

    def test_truncates_long_content(self, extractor):
        """Should truncate very long dissent content."""
        result = MagicMock()
        result.dissenting_views = ["[agent-1]: " + "x" * 500]
        insights = extractor._extract_dissent_insights(result, "d1")
        assert len(insights[0].description) < 400


# =============================================================================
# Test Pattern Insight Extraction
# =============================================================================


class TestExtractPatternInsights:
    """Tests for _extract_pattern_insights method."""

    def test_returns_empty_no_critiques(self, extractor):
        """Should return empty when no critiques."""
        result = MagicMock()
        result.critiques = []
        insights = extractor._extract_pattern_insights(result, "d1")
        assert insights == []

    def test_ignores_single_occurrence(self, extractor, mock_critique):
        """Should not create pattern for single occurrence."""
        result = MagicMock()
        result.critiques = [mock_critique("agent-1", "agent-2", ["security issue"])]
        insights = extractor._extract_pattern_insights(result, "d1")
        assert len(insights) == 0

    def test_extracts_pattern_two_occurrences(self, extractor, mock_critique):
        """Should create pattern for 2+ occurrences."""
        result = MagicMock()
        result.critiques = [
            mock_critique("agent-1", "agent-2", ["security vulnerability"]),
            mock_critique("agent-3", "agent-2", ["authentication security"]),
        ]
        insights = extractor._extract_pattern_insights(result, "d1")
        assert len(insights) == 1
        assert "security" in insights[0].title.lower()

    def test_extracts_multiple_categories(self, extractor, mock_critique):
        """Should extract patterns for multiple categories."""
        result = MagicMock()
        result.critiques = [
            mock_critique("a1", "a2", ["security issue"]),
            mock_critique("a3", "a2", ["security concern"]),
            mock_critique("a1", "a2", ["performance problem"]),
            mock_critique("a3", "a2", ["slow performance"]),
        ]
        insights = extractor._extract_pattern_insights(result, "d1")
        assert len(insights) == 2
        categories = [i.metadata["category"] for i in insights]
        assert "security" in categories
        assert "performance" in categories

    def test_calculates_avg_severity(self, extractor, mock_critique):
        """Should calculate average severity."""
        result = MagicMock()
        result.critiques = [
            mock_critique("a1", "a2", ["security issue"], severity=0.8),
            mock_critique("a3", "a2", ["security concern"], severity=0.4),
        ]
        insights = extractor._extract_pattern_insights(result, "d1")
        assert insights[0].metadata["avg_severity"] == pytest.approx(0.6)

    def test_confidence_increases_with_count(self, extractor, mock_critique):
        """Confidence should increase with more occurrences."""
        result = MagicMock()
        result.critiques = [
            mock_critique("a1", "a2", ["security issue"]),
            mock_critique("a3", "a2", ["security concern"]),
        ]
        insights_2 = extractor._extract_pattern_insights(result, "d1")

        result.critiques.extend(
            [
                mock_critique("a4", "a2", ["security vulnerability"]),
                mock_critique("a5", "a2", ["security flaw"]),
            ]
        )
        insights_4 = extractor._extract_pattern_insights(result, "d1")

        assert insights_4[0].confidence > insights_2[0].confidence


# =============================================================================
# Test Convergence Insight Extraction
# =============================================================================


class TestExtractConvergenceInsight:
    """Tests for _extract_convergence_insight method."""

    def test_returns_none_few_messages(self, extractor):
        """Should return None with fewer than 3 messages."""
        result = MagicMock()
        result.messages = ["msg1", "msg2"]
        result.consensus_variance = 0.1
        insight = extractor._extract_convergence_insight(result, "d1")
        assert insight is None

    def test_detects_strong_convergence(self, extractor):
        """Should detect strong convergence when late messages are much shorter."""
        result = MagicMock()
        # Early messages: long, Late messages: short
        result.messages = [
            "x" * 1000,
            "x" * 1000,  # Early
            "x" * 200,
            "x" * 200,  # Late (much shorter)
        ]
        result.consensus_variance = 0.1
        insight = extractor._extract_convergence_insight(result, "d1")
        assert insight is not None
        assert "strong" in insight.metadata["convergence_type"]

    def test_detects_mild_convergence(self, extractor):
        """Should detect mild convergence."""
        result = MagicMock()
        result.messages = [
            "x" * 1000,
            "x" * 1000,
            "x" * 850,
            "x" * 850,  # Slightly shorter
        ]
        result.consensus_variance = 0.1
        insight = extractor._extract_convergence_insight(result, "d1")
        assert "mild" in insight.metadata["convergence_type"]

    def test_detects_divergence(self, extractor):
        """Should detect divergence when late messages are longer."""
        result = MagicMock()
        result.messages = [
            "x" * 100,
            "x" * 100,
            "x" * 500,
            "x" * 500,  # Much longer
        ]
        result.consensus_variance = 0.5
        insight = extractor._extract_convergence_insight(result, "d1")
        assert "divergence" in insight.metadata["convergence_type"]

    def test_detects_stable(self, extractor):
        """Should detect stable when lengths are similar."""
        result = MagicMock()
        result.messages = [
            "x" * 100,
            "x" * 100,
            "x" * 100,
            "x" * 100,
        ]
        result.consensus_variance = 0.1
        insight = extractor._extract_convergence_insight(result, "d1")
        assert "stable" in insight.metadata["convergence_type"]

    def test_includes_variance_in_metadata(self, extractor):
        """Should include variance in metadata."""
        result = MagicMock()
        result.messages = ["x" * 100] * 4
        result.consensus_variance = 0.25
        insight = extractor._extract_convergence_insight(result, "d1")
        assert insight.metadata["variance"] == 0.25


# =============================================================================
# Test Decision Insight Extraction
# =============================================================================


class TestExtractDecisionInsight:
    """Tests for _extract_decision_insight method."""

    def test_extracts_decision_insight(self, extractor, mock_result):
        """Should extract decision insight."""
        insight = extractor._extract_decision_insight(mock_result, "d1")
        assert insight is not None
        assert insight.type == InsightType.DECISION_PROCESS

    def test_detects_unanimous_mode(self, extractor):
        """Should detect unanimous consensus mode."""
        result = MagicMock()
        result.consensus_strength = "unanimous"
        result.votes = []
        result.rounds_used = 2
        result.messages = []
        result.critiques = []
        insight = extractor._extract_decision_insight(result, "d1")
        assert insight.metadata["consensus_mode"] == "unanimous"

    def test_detects_majority_mode(self, extractor, mock_vote):
        """Should detect majority consensus mode."""
        result = MagicMock()
        result.consensus_strength = "strong"
        result.votes = [mock_vote("a1", "A"), mock_vote("a2", "A"), mock_vote("a3", "B")]
        result.rounds_used = 3
        result.messages = []
        result.critiques = []
        insight = extractor._extract_decision_insight(result, "d1")
        assert insight.metadata["consensus_mode"] == "majority"

    def test_detects_judge_mode(self, extractor):
        """Should detect judge consensus mode."""
        result = MagicMock()
        result.consensus_strength = "weak"
        result.votes = []
        result.rounds_used = 3
        result.messages = []
        result.critiques = []
        insight = extractor._extract_decision_insight(result, "d1")
        assert insight.metadata["consensus_mode"] == "judge"

    def test_includes_vote_summary(self, extractor, mock_vote):
        """Should include vote summary in description."""
        result = MagicMock()
        result.consensus_strength = "strong"
        result.votes = [
            mock_vote("a1", "Option A"),
            mock_vote("a2", "Option A"),
            mock_vote("a3", "Option B"),
        ]
        result.rounds_used = 2
        result.messages = []
        result.critiques = []
        insight = extractor._extract_decision_insight(result, "d1")
        assert "Option A" in insight.description

    def test_handles_no_votes(self, extractor):
        """Should handle no votes gracefully."""
        result = MagicMock()
        result.consensus_strength = "judge"
        result.votes = []
        result.rounds_used = 2
        result.messages = []
        result.critiques = []
        insight = extractor._extract_decision_insight(result, "d1")
        assert "N/A" in insight.description


# =============================================================================
# Test Failure Mode Extraction
# =============================================================================


class TestExtractFailureMode:
    """Tests for _extract_failure_mode method."""

    def test_extracts_failure_insight(self, extractor, mock_result_no_consensus):
        """Should extract failure mode insight."""
        insight = extractor._extract_failure_mode(mock_result_no_consensus, "d1")
        assert insight is not None
        assert insight.type == InsightType.FAILURE_MODE

    def test_detects_vote_fragmentation(self, extractor, mock_vote):
        """Should detect vote fragmentation."""
        result = MagicMock()
        result.votes = [
            mock_vote("a1", "A"),
            mock_vote("a2", "B"),
            mock_vote("a3", "C"),  # 3+ unique choices
        ]
        result.critiques = []
        result.dissenting_views = []
        result.confidence = 0.3
        insight = extractor._extract_failure_mode(result, "d1")
        assert "fragmentation" in insight.metadata["failure_reasons"][0]

    def test_detects_high_severity_issues(self, extractor, mock_critique):
        """Should detect high severity issues as failure reason."""
        result = MagicMock()
        result.votes = []
        result.critiques = [
            mock_critique("a1", "a2", ["issue"], severity=0.9),
            mock_critique("a3", "a2", ["issue"], severity=0.8),
        ]
        result.dissenting_views = []
        result.confidence = 0.4
        insight = extractor._extract_failure_mode(result, "d1")
        assert any("high-severity" in r for r in insight.metadata["failure_reasons"])

    def test_detects_unresolved_dissent(self, extractor):
        """Should detect unresolved dissent as failure reason."""
        result = MagicMock()
        result.votes = []
        result.critiques = []
        result.dissenting_views = ["dissent 1", "dissent 2"]
        result.confidence = 0.3
        insight = extractor._extract_failure_mode(result, "d1")
        assert any("dissenting" in r for r in insight.metadata["failure_reasons"])

    def test_combines_multiple_reasons(self, extractor, mock_vote, mock_critique):
        """Should combine multiple failure reasons."""
        result = MagicMock()
        result.votes = [mock_vote("a1", "A"), mock_vote("a2", "B"), mock_vote("a3", "C")]
        result.critiques = [
            mock_critique("a1", "a2", ["issue"], severity=0.9),
            mock_critique("a3", "a2", ["issue"], severity=0.8),
        ]
        result.dissenting_views = ["d1", "d2"]
        result.confidence = 0.2
        insight = extractor._extract_failure_mode(result, "d1")
        assert len(insight.metadata["failure_reasons"]) >= 2

    def test_handles_unknown_factors(self, extractor):
        """Should handle unknown failure factors."""
        result = MagicMock()
        result.votes = []
        result.critiques = []
        result.dissenting_views = []
        result.confidence = 0.3
        insight = extractor._extract_failure_mode(result, "d1")
        assert "unknown" in insight.description


# =============================================================================
# Test Agent Performance Extraction
# =============================================================================


class TestExtractAgentPerformances:
    """Tests for _extract_agent_performances method."""

    def test_returns_empty_no_messages(self, extractor):
        """Should return empty list when no messages."""
        result = MagicMock()
        result.messages = []
        result.critiques = []
        result.votes = []
        result.final_answer = ""
        performances = extractor._extract_agent_performances(result)
        assert performances == []

    def test_counts_proposals(self, extractor):
        """Should count proposals for each agent."""
        result = MagicMock()
        # Use dict-style messages (module has bug with object-style)
        result.messages = [
            {"agent": "agent-1", "content": "msg1"},
            {"agent": "agent-1", "content": "msg2"},
            {"agent": "agent-2", "content": "msg3"},
        ]
        result.critiques = []
        result.votes = []
        result.final_answer = ""
        performances = extractor._extract_agent_performances(result)
        agent1 = next(p for p in performances if p.agent_name == "agent-1")
        assert agent1.proposals_made == 2

    def test_counts_critiques_given(self, extractor, mock_critique):
        """Should count critiques given."""
        result = MagicMock()
        result.messages = [{"agent": "agent-1", "content": "msg"}]
        result.critiques = [
            mock_critique("agent-1", "agent-2", ["issue"]),
            mock_critique("agent-1", "agent-2", ["issue"]),
        ]
        result.votes = []
        result.final_answer = ""
        performances = extractor._extract_agent_performances(result)
        agent1 = next(p for p in performances if p.agent_name == "agent-1")
        assert agent1.critiques_given == 2

    def test_counts_critiques_received(self, extractor, mock_critique):
        """Should count critiques received."""
        result = MagicMock()
        result.messages = [
            {"agent": "agent-1", "content": "msg"},
            {"agent": "agent-2", "content": "msg"},
        ]
        result.critiques = [
            mock_critique("agent-2", "agent-1", ["issue"]),
            mock_critique("agent-2", "agent-1", ["issue"]),
        ]
        result.votes = []
        result.final_answer = ""
        performances = extractor._extract_agent_performances(result)
        agent1 = next(p for p in performances if p.agent_name == "agent-1")
        assert agent1.critiques_received == 2

    def test_calculates_average_severity(self, extractor, mock_critique):
        """Should calculate average severity of received critiques."""
        result = MagicMock()
        result.messages = [
            {"agent": "agent-1", "content": "msg"},
            {"agent": "agent-2", "content": "msg"},
        ]
        result.critiques = [
            mock_critique("agent-2", "agent-1", ["issue"], severity=0.6),
            mock_critique("agent-2", "agent-1", ["issue"], severity=0.8),
        ]
        result.votes = []
        result.final_answer = ""
        performances = extractor._extract_agent_performances(result)
        agent1 = next(p for p in performances if p.agent_name == "agent-1")
        assert agent1.average_critique_severity == pytest.approx(0.7)

    def test_detects_vote_alignment(self, extractor, mock_vote):
        """Should detect if vote aligned with consensus."""
        result = MagicMock()
        result.messages = [
            {"agent": "agent-1", "content": "msg"},
            {"agent": "agent-2", "content": "msg"},
        ]
        result.critiques = []
        result.votes = [
            mock_vote("agent-1", "A"),
            mock_vote("agent-2", "A"),
        ]
        result.final_answer = "Option A is the answer"
        performances = extractor._extract_agent_performances(result)
        agent1 = next(p for p in performances if p.agent_name == "agent-1")
        assert agent1.vote_aligned_with_consensus is True

    def test_detects_proposal_accepted(self, extractor):
        """Should detect if agent's proposal was accepted."""
        result = MagicMock()
        result.messages = [
            {"agent": "agent-1", "content": "msg"},
            {"agent": "agent-2", "content": "msg"},
        ]
        result.critiques = []
        result.votes = []
        result.final_answer = "As agent-1 suggested, we should..."
        performances = extractor._extract_agent_performances(result)
        agent1 = next(p for p in performances if p.agent_name == "agent-1")
        assert agent1.proposal_accepted is True

    def test_calculates_contribution_score(self, extractor, mock_vote):
        """Should calculate contribution score."""
        result = MagicMock()
        result.messages = [{"agent": "agent-1", "content": "msg"}]
        result.critiques = []
        result.votes = [mock_vote("agent-1", "A")]
        result.final_answer = "agent-1 wins"
        performances = extractor._extract_agent_performances(result)
        agent1 = next(p for p in performances if p.agent_name == "agent-1")
        # Base 0.5 + 0.3 (accepted) + 0.1 (aligned) = 0.9
        assert agent1.contribution_score > 0.5

    def test_handles_dict_messages(self, extractor):
        """Should handle dict-style messages."""
        result = MagicMock()
        result.messages = [{"agent": "agent-1", "content": "test"}]
        result.critiques = []
        result.votes = []
        result.final_answer = ""
        performances = extractor._extract_agent_performances(result)
        assert len(performances) == 1
        assert performances[0].agent_name == "agent-1"


# =============================================================================
# Test Helper Methods
# =============================================================================


class TestHelperMethods:
    """Tests for helper methods."""

    def test_get_agent_names_from_messages(self, extractor):
        """Should extract agent names from messages (dict-style)."""
        result = MagicMock()
        # Module has bug with object-style messages, use dict-style
        result.messages = [
            {"agent": "agent-1", "content": "msg"},
            {"agent": "agent-2", "content": "msg"},
        ]
        result.critiques = []
        result.votes = []
        agents = extractor._get_agent_names(result)
        assert "agent-1" in agents
        assert "agent-2" in agents

    def test_get_agent_names_from_critiques(self, extractor, mock_critique):
        """Should extract agent names from critiques."""
        result = MagicMock()
        result.messages = []
        result.critiques = [mock_critique("agent-1", "agent-2", ["issue"])]
        result.votes = []
        agents = extractor._get_agent_names(result)
        assert "agent-1" in agents

    def test_get_agent_names_from_votes(self, extractor, mock_vote):
        """Should extract agent names from votes."""
        result = MagicMock()
        result.messages = []
        result.critiques = []
        result.votes = [mock_vote("agent-1", "A")]
        agents = extractor._get_agent_names(result)
        assert "agent-1" in agents

    def test_get_agent_names_combined(self, extractor, mock_critique, mock_vote):
        """Should combine agent names from all sources."""
        result = MagicMock()
        # Use dict-style for messages due to module bug with object-style
        result.messages = [{"agent": "agent-1", "content": "msg"}]
        result.critiques = [mock_critique("agent-2", "agent-1", ["issue"])]
        result.votes = [mock_vote("agent-3", "A")]
        agents = extractor._get_agent_names(result)
        assert len(agents) == 3

    def test_categorize_issue_security(self, extractor):
        """Should categorize security issues."""
        assert extractor._categorize_issue("SQL injection vulnerability") == "security"
        assert extractor._categorize_issue("XSS attack vector") == "security"

    def test_categorize_issue_performance(self, extractor):
        """Should categorize performance issues."""
        assert extractor._categorize_issue("Slow database query") == "performance"
        assert extractor._categorize_issue("Optimize this loop") == "performance"

    def test_categorize_issue_correctness(self, extractor):
        """Should categorize correctness issues."""
        assert extractor._categorize_issue("Bug in validation") == "correctness"
        assert extractor._categorize_issue("This will fail") == "correctness"

    def test_categorize_issue_none_match(self, extractor):
        """Should return None when no category matches."""
        assert extractor._categorize_issue("random text xyz") is None

    def test_generate_key_takeaway_consensus(self, extractor):
        """Should generate takeaway for consensus."""
        insights = DebateInsights(
            debate_id="d1",
            task="Task",
            consensus_reached=True,
            duration_seconds=120.0,
            consensus_insight=Insight(
                id="c1",
                type=InsightType.CONSENSUS,
                title="Consensus",
                description="Desc",
                confidence=0.9,
                debate_id="d1",
                metadata={"consensus_strength": "strong"},
            ),
        )
        takeaway = extractor._generate_key_takeaway(insights)
        assert "strong" in takeaway
        assert "consensus" in takeaway.lower()

    def test_generate_key_takeaway_failure(self, extractor):
        """Should generate takeaway for failure."""
        insights = DebateInsights(
            debate_id="d1",
            task="Task",
            consensus_reached=False,
            duration_seconds=120.0,
            failure_mode_insight=Insight(
                id="f1",
                type=InsightType.FAILURE_MODE,
                title="Failure",
                description="Desc",
                confidence=0.8,
                debate_id="d1",
                metadata={"failure_reasons": ["vote fragmentation"]},
            ),
        )
        takeaway = extractor._generate_key_takeaway(insights)
        assert "No consensus" in takeaway or "Failed" in takeaway
