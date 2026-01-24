"""
Tests for the InsightExtractor module.

Tests insight extraction from debate results.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

import pytest

from aragora.insights.extractor import (
    Insight,
    InsightType,
    AgentPerformance,
    DebateInsights,
    InsightExtractor,
)


# =============================================================================
# Mock Objects for Testing
# =============================================================================


class MockMessage(dict):
    """Mock debate message that works as both dict and object."""

    def __init__(self, agent: str, content: str):
        super().__init__(agent=agent, content=content)
        self.agent = agent
        self.content = content


@dataclass
class MockCritique:
    """Mock critique with issues."""

    agent: str
    target_agent: str
    severity: float
    issues: List[str] = field(default_factory=list)


@dataclass
class MockVote:
    """Mock vote."""

    agent: str
    choice: str


@dataclass
class MockDebateResult:
    """Mock debate result for testing."""

    id: str = "test-debate-123"
    task: str = "Test task description"
    consensus_reached: bool = True
    confidence: float = 0.85
    consensus_strength: str = "strong"
    final_answer: str = "The consensus is that AI is beneficial"
    duration_seconds: float = 120.0
    rounds_used: int = 3
    consensus_variance: float = 0.15

    messages: List[MockMessage] = field(default_factory=list)
    critiques: List[MockCritique] = field(default_factory=list)
    votes: List[MockVote] = field(default_factory=list)
    dissenting_views: List[str] = field(default_factory=list)


# =============================================================================
# InsightType Tests
# =============================================================================


class TestInsightType:
    """Tests for InsightType enum."""

    def test_all_insight_types_defined(self):
        """Test that all insight types are defined."""
        expected_types = [
            "consensus",
            "dissent",
            "pattern",
            "convergence",
            "agent_perf",
            "failure_mode",
            "decision",
        ]

        for type_value in expected_types:
            assert any(t.value == type_value for t in InsightType)


# =============================================================================
# Insight Tests
# =============================================================================


class TestInsight:
    """Tests for Insight dataclass."""

    def test_create_insight(self):
        """Test creating an insight."""
        insight = Insight(
            id="insight-123",
            type=InsightType.CONSENSUS,
            title="Consensus Reached",
            description="The debate reached consensus on the topic.",
            confidence=0.9,
            debate_id="debate-1",
            agents_involved=["claude", "gpt"],
            evidence=["Evidence 1", "Evidence 2"],
        )

        assert insight.id == "insight-123"
        assert insight.type == InsightType.CONSENSUS
        assert insight.confidence == 0.9
        assert len(insight.agents_involved) == 2

    def test_insight_serialization(self):
        """Test insight to_dict serialization."""
        insight = Insight(
            id="insight-456",
            type=InsightType.PATTERN,
            title="Recurring Issue",
            description="Multiple agents raised similar concerns.",
            confidence=0.75,
            debate_id="debate-2",
        )

        data = insight.to_dict()

        assert data["id"] == "insight-456"
        assert data["title"] == "Recurring Issue"
        assert "created_at" in data


# =============================================================================
# AgentPerformance Tests
# =============================================================================


class TestAgentPerformance:
    """Tests for AgentPerformance dataclass."""

    def test_default_values(self):
        """Test default values for agent performance."""
        perf = AgentPerformance(agent_name="claude")

        assert perf.proposals_made == 0
        assert perf.critiques_given == 0
        assert perf.critiques_received == 0
        assert perf.proposal_accepted is False
        assert perf.vote_aligned_with_consensus is False
        assert perf.contribution_score == 0.5

    def test_performance_with_values(self):
        """Test agent performance with custom values."""
        perf = AgentPerformance(
            agent_name="gpt",
            proposals_made=3,
            critiques_given=5,
            critiques_received=2,
            proposal_accepted=True,
            vote_aligned_with_consensus=True,
            average_critique_severity=0.6,
            contribution_score=0.85,
        )

        assert perf.proposals_made == 3
        assert perf.critiques_given == 5
        assert perf.proposal_accepted is True
        assert perf.contribution_score == 0.85


# =============================================================================
# DebateInsights Tests
# =============================================================================


class TestDebateInsights:
    """Tests for DebateInsights dataclass."""

    def test_all_insights_empty(self):
        """Test all_insights with no insights."""
        insights = DebateInsights(
            debate_id="d1",
            task="Task",
            consensus_reached=True,
            duration_seconds=60.0,
        )

        assert insights.all_insights() == []

    def test_all_insights_with_data(self):
        """Test all_insights returns all insight types."""
        consensus = Insight(
            id="c1",
            type=InsightType.CONSENSUS,
            title="Consensus",
            description="Desc",
            confidence=0.9,
            debate_id="d1",
        )
        dissent = Insight(
            id="d1",
            type=InsightType.DISSENT,
            title="Dissent",
            description="Desc",
            confidence=0.6,
            debate_id="d1",
        )
        pattern = Insight(
            id="p1",
            type=InsightType.PATTERN,
            title="Pattern",
            description="Desc",
            confidence=0.7,
            debate_id="d1",
        )

        insights = DebateInsights(
            debate_id="d1",
            task="Task",
            consensus_reached=True,
            duration_seconds=60.0,
            consensus_insight=consensus,
            dissent_insights=[dissent],
            pattern_insights=[pattern],
        )

        all_insights = insights.all_insights()
        assert len(all_insights) == 3
        assert consensus in all_insights
        assert dissent in all_insights
        assert pattern in all_insights


# =============================================================================
# InsightExtractor Tests
# =============================================================================


class TestInsightExtractor:
    """Tests for InsightExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create an InsightExtractor instance."""
        return InsightExtractor()

    @pytest.fixture
    def basic_result(self):
        """Create a basic mock debate result."""
        return MockDebateResult(
            messages=[
                MockMessage(agent="claude", content="Initial proposal"),
                MockMessage(agent="gpt", content="Counter proposal"),
                MockMessage(agent="claude", content="Revised proposal"),
            ],
            votes=[
                MockVote(agent="claude", choice="A"),
                MockVote(agent="gpt", choice="A"),
            ],
        )

    @pytest.mark.asyncio
    async def test_extract_basic(self, extractor, basic_result):
        """Test basic insight extraction."""
        insights = await extractor.extract(basic_result)

        assert insights.debate_id == "test-debate-123"
        assert insights.consensus_reached is True
        assert insights.duration_seconds == 120.0

    @pytest.mark.asyncio
    async def test_extract_consensus_insight(self, extractor, basic_result):
        """Test consensus insight extraction."""
        insights = await extractor.extract(basic_result)

        assert insights.consensus_insight is not None
        assert insights.consensus_insight.type == InsightType.CONSENSUS
        assert "strong" in insights.consensus_insight.title.lower()

    @pytest.mark.asyncio
    async def test_extract_no_consensus(self, extractor):
        """Test extraction when consensus not reached."""
        result = MockDebateResult(
            consensus_reached=False,
            dissenting_views=[
                "[claude]: I disagree with this approach",
                "[gpt]: The evidence doesn't support this",
            ],
        )

        insights = await extractor.extract(result)

        assert insights.consensus_insight is None
        assert insights.failure_mode_insight is not None
        assert insights.failure_mode_insight.type == InsightType.FAILURE_MODE

    @pytest.mark.asyncio
    async def test_extract_dissent_insights(self, extractor):
        """Test dissent insight extraction."""
        result = MockDebateResult(
            dissenting_views=[
                "[claude]: I disagree because of X",
                "[gpt]: My alternative view is Y",
            ]
        )

        insights = await extractor.extract(result)

        assert len(insights.dissent_insights) == 2
        assert insights.dissent_insights[0].type == InsightType.DISSENT
        assert "claude" in insights.dissent_insights[0].agents_involved

    @pytest.mark.asyncio
    async def test_extract_pattern_insights(self, extractor):
        """Test pattern insight extraction from critiques."""
        result = MockDebateResult(
            critiques=[
                MockCritique(
                    agent="claude",
                    target_agent="gpt",
                    severity=0.7,
                    issues=["This has a security vulnerability"],
                ),
                MockCritique(
                    agent="gpt",
                    target_agent="gemini",
                    severity=0.8,
                    issues=["Security concern with auth"],
                ),
                MockCritique(
                    agent="gemini",
                    target_agent="claude",
                    severity=0.6,
                    issues=["Performance is slow"],
                ),
            ]
        )

        insights = await extractor.extract(result)

        # Should detect security as a recurring pattern (2+ occurrences)
        security_patterns = [p for p in insights.pattern_insights if "security" in p.title.lower()]
        assert len(security_patterns) == 1

    @pytest.mark.asyncio
    async def test_extract_convergence_insight(self, extractor):
        """Test convergence insight extraction."""
        # Create messages with decreasing length to simulate convergence
        messages = [
            MockMessage(agent="claude", content="A" * 500),
            MockMessage(agent="gpt", content="B" * 400),
            MockMessage(agent="claude", content="C" * 300),
            MockMessage(agent="gpt", content="D" * 200),
            MockMessage(agent="claude", content="E" * 100),
            MockMessage(agent="gpt", content="F" * 50),
        ]

        result = MockDebateResult(messages=messages)
        insights = await extractor.extract(result)

        assert insights.convergence_insight is not None
        assert insights.convergence_insight.type == InsightType.CONVERGENCE
        assert "convergence" in insights.convergence_insight.metadata.get("convergence_type", "")

    @pytest.mark.asyncio
    async def test_extract_decision_insight(self, extractor, basic_result):
        """Test decision process insight extraction."""
        insights = await extractor.extract(basic_result)

        assert insights.decision_insight is not None
        assert insights.decision_insight.type == InsightType.DECISION_PROCESS

    @pytest.mark.asyncio
    async def test_extract_agent_performances(self, extractor, basic_result):
        """Test agent performance extraction."""
        insights = await extractor.extract(basic_result)

        assert len(insights.agent_performances) >= 2

        claude_perf = next(
            (p for p in insights.agent_performances if p.agent_name == "claude"),
            None,
        )
        assert claude_perf is not None
        assert claude_perf.proposals_made >= 1

    @pytest.mark.asyncio
    async def test_key_takeaway_consensus(self, extractor, basic_result):
        """Test key takeaway for consensus result."""
        insights = await extractor.extract(basic_result)

        assert "consensus" in insights.key_takeaway.lower()

    @pytest.mark.asyncio
    async def test_key_takeaway_no_consensus(self, extractor):
        """Test key takeaway when no consensus."""
        result = MockDebateResult(
            consensus_reached=False,
            dissenting_views=["[a]: X", "[b]: Y"],
        )

        insights = await extractor.extract(result)

        assert "no consensus" in insights.key_takeaway.lower()

    @pytest.mark.asyncio
    async def test_total_insights_count(self, extractor, basic_result):
        """Test total insights count."""
        insights = await extractor.extract(basic_result)

        assert insights.total_insights == len(insights.all_insights())

    def test_categorize_issue_security(self, extractor):
        """Test issue categorization for security."""
        category = extractor._categorize_issue("SQL injection vulnerability found")
        assert category == "security"

    def test_categorize_issue_performance(self, extractor):
        """Test issue categorization for performance."""
        category = extractor._categorize_issue("This function is slow")
        assert category == "performance"

    def test_categorize_issue_testing(self, extractor):
        """Test issue categorization for testing."""
        category = extractor._categorize_issue("Unit test coverage is insufficient")
        assert category == "testing"

    def test_categorize_issue_unknown(self, extractor):
        """Test issue categorization for unknown issue."""
        category = extractor._categorize_issue("Something unrelated")
        assert category is None

    def test_get_agent_names(self, extractor, basic_result):
        """Test extracting agent names from result."""
        agents = extractor._get_agent_names(basic_result)

        assert "claude" in agents
        assert "gpt" in agents

    @pytest.mark.asyncio
    async def test_extract_with_critiques_performance(self, extractor):
        """Test agent performance with critiques."""
        result = MockDebateResult(
            messages=[
                MockMessage(agent="claude", content="Proposal"),
                MockMessage(agent="gpt", content="Response"),
            ],
            critiques=[
                MockCritique(
                    agent="gpt",
                    target_agent="claude",
                    severity=0.8,
                    issues=["Bug found"],
                ),
            ],
            votes=[
                MockVote(agent="claude", choice="A"),
                MockVote(agent="gpt", choice="A"),
            ],
        )

        insights = await extractor.extract(result)

        claude_perf = next(
            (p for p in insights.agent_performances if p.agent_name == "claude"),
            None,
        )
        gpt_perf = next(
            (p for p in insights.agent_performances if p.agent_name == "gpt"),
            None,
        )

        assert claude_perf is not None
        assert claude_perf.critiques_received >= 1

        assert gpt_perf is not None
        assert gpt_perf.critiques_given >= 1


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def extractor(self):
        return InsightExtractor()

    @pytest.mark.asyncio
    async def test_extract_minimal_result(self, extractor):
        """Test extraction with minimal result data."""

        @dataclass
        class MinimalResult:
            id: str = "minimal"

        result = MinimalResult()
        insights = await extractor.extract(result)

        assert insights.debate_id == "minimal"
        assert insights.consensus_reached is False

    @pytest.mark.asyncio
    async def test_extract_empty_messages(self, extractor):
        """Test extraction with empty messages list."""
        result = MockDebateResult(messages=[])
        insights = await extractor.extract(result)

        # Should still work without errors
        assert insights.convergence_insight is None

    @pytest.mark.asyncio
    async def test_extract_very_long_content(self, extractor):
        """Test extraction handles very long content."""
        long_answer = "A" * 10000
        result = MockDebateResult(final_answer=long_answer)

        insights = await extractor.extract(result)

        # Should truncate properly
        assert insights.consensus_insight is not None
        assert len(insights.consensus_insight.description) < 1000

    @pytest.mark.asyncio
    async def test_extract_unicode_content(self, extractor):
        """Test extraction handles unicode content."""
        result = MockDebateResult(
            task="Test unicode: \u4e2d\u6587\u6d4b\u8bd5",
            final_answer="Answer with emoji: \U0001f680",
            messages=[
                MockMessage(agent="claude", content="Japanese: \u3053\u3093\u306b\u3061\u306f")
            ],
        )

        insights = await extractor.extract(result)

        assert insights.task == "Test unicode: \u4e2d\u6587\u6d4b\u8bd5"

    @pytest.mark.asyncio
    async def test_failure_mode_vote_fragmentation(self, extractor):
        """Test failure mode detection for vote fragmentation."""
        result = MockDebateResult(
            consensus_reached=False,
            votes=[
                MockVote(agent="a", choice="X"),
                MockVote(agent="b", choice="Y"),
                MockVote(agent="c", choice="Z"),
            ],
        )

        insights = await extractor.extract(result)

        assert insights.failure_mode_insight is not None
        reasons = insights.failure_mode_insight.metadata.get("failure_reasons", [])
        assert any("fragmentation" in r for r in reasons)

    @pytest.mark.asyncio
    async def test_failure_mode_high_severity_critiques(self, extractor):
        """Test failure mode detection for high severity critiques."""
        result = MockDebateResult(
            consensus_reached=False,
            critiques=[
                MockCritique(agent="a", target_agent="b", severity=0.9, issues=["Critical"]),
                MockCritique(agent="b", target_agent="c", severity=0.85, issues=["Major"]),
                MockCritique(agent="c", target_agent="a", severity=0.8, issues=["Severe"]),
            ],
        )

        insights = await extractor.extract(result)

        assert insights.failure_mode_insight is not None
        reasons = insights.failure_mode_insight.metadata.get("failure_reasons", [])
        assert any("severity" in r for r in reasons)
