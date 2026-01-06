"""
Tests for aragora.insights.extractor module.

Tests InsightExtractor and supporting dataclasses.
"""

import pytest
from dataclasses import dataclass, field
from unittest.mock import MagicMock

from aragora.insights.extractor import (
    InsightType,
    Insight,
    AgentPerformance,
    DebateInsights,
    InsightExtractor,
)


# ============================================================================
# Test Data Fixtures
# ============================================================================

@dataclass
class MockCritique:
    """Mock critique object."""
    agent: str = "claude"
    target_agent: str = "gpt4"
    issues: list = field(default_factory=list)
    severity: float = 0.5


@dataclass
class MockVote:
    """Mock vote object."""
    agent: str = "claude"
    choice: str = "A"


@dataclass
class MockMessage:
    """Mock message object."""
    agent: str = "claude"
    content: str = "Test message"


@dataclass
class MockDebateResult:
    """Mock debate result for testing."""
    id: str = "debate_001"
    task: str = "Test task"
    consensus_reached: bool = True
    confidence: float = 0.85
    duration_seconds: float = 120.0
    final_answer: str = "The answer is X"
    consensus_strength: str = "strong"
    rounds_used: int = 3
    consensus_variance: float = 0.1
    messages: list = field(default_factory=list)
    critiques: list = field(default_factory=list)
    votes: list = field(default_factory=list)
    dissenting_views: list = field(default_factory=list)


# ============================================================================
# Insight Dataclass Tests
# ============================================================================

class TestInsight:
    """Tests for Insight dataclass."""

    def test_to_dict_all_fields(self):
        """Should serialize all fields."""
        insight = Insight(
            id="test_insight",
            type=InsightType.CONSENSUS,
            title="Test Title",
            description="Test description",
            confidence=0.9,
            debate_id="debate_001",
            agents_involved=["claude", "gpt4"],
            evidence=["evidence1"],
            metadata={"key": "value"},
        )
        d = insight.to_dict()

        assert d["id"] == "test_insight"
        assert d["type"] == "consensus"
        assert d["title"] == "Test Title"
        assert d["confidence"] == 0.9
        assert d["debate_id"] == "debate_001"
        assert d["agents_involved"] == ["claude", "gpt4"]
        assert d["metadata"]["key"] == "value"

    def test_insight_types(self):
        """Should have expected insight types."""
        assert InsightType.CONSENSUS.value == "consensus"
        assert InsightType.DISSENT.value == "dissent"
        assert InsightType.PATTERN.value == "pattern"
        assert InsightType.CONVERGENCE.value == "convergence"
        assert InsightType.AGENT_PERFORMANCE.value == "agent_perf"
        assert InsightType.FAILURE_MODE.value == "failure_mode"
        assert InsightType.DECISION_PROCESS.value == "decision"


class TestAgentPerformance:
    """Tests for AgentPerformance dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        perf = AgentPerformance(agent_name="claude")

        assert perf.agent_name == "claude"
        assert perf.proposals_made == 0
        assert perf.critiques_given == 0
        assert perf.contribution_score == 0.5

    def test_all_fields(self):
        """Should store all performance metrics."""
        perf = AgentPerformance(
            agent_name="gpt4",
            proposals_made=5,
            critiques_given=3,
            critiques_received=2,
            proposal_accepted=True,
            vote_aligned_with_consensus=True,
            average_critique_severity=0.3,
            contribution_score=0.8,
        )

        assert perf.proposals_made == 5
        assert perf.critiques_given == 3
        assert perf.proposal_accepted is True
        assert perf.contribution_score == 0.8


class TestDebateInsights:
    """Tests for DebateInsights dataclass."""

    def test_all_insights_empty(self):
        """Should return empty list when no insights."""
        insights = DebateInsights(
            debate_id="d001",
            task="Test",
            consensus_reached=True,
            duration_seconds=60.0,
        )

        assert insights.all_insights() == []

    def test_all_insights_with_content(self):
        """Should collect all insights types."""
        consensus = Insight(
            id="c1", type=InsightType.CONSENSUS, title="T",
            description="D", confidence=0.9, debate_id="d001",
        )
        dissent = Insight(
            id="d1", type=InsightType.DISSENT, title="T",
            description="D", confidence=0.6, debate_id="d001",
        )
        pattern = Insight(
            id="p1", type=InsightType.PATTERN, title="T",
            description="D", confidence=0.7, debate_id="d001",
        )

        insights = DebateInsights(
            debate_id="d001",
            task="Test",
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


# ============================================================================
# InsightExtractor Tests
# ============================================================================

class TestInsightExtractor:
    """Tests for InsightExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create an InsightExtractor instance."""
        return InsightExtractor()

    @pytest.mark.asyncio
    async def test_extract_basic(self, extractor):
        """Should extract basic insights from result."""
        result = MockDebateResult(
            messages=[MockMessage(agent="claude")],
        )

        insights = await extractor.extract(result)

        assert insights.debate_id == "debate_001"
        assert insights.task == "Test task"
        assert insights.consensus_reached is True

    @pytest.mark.asyncio
    async def test_extract_consensus_insight(self, extractor):
        """Should extract consensus insight when consensus reached."""
        result = MockDebateResult(
            consensus_reached=True,
            final_answer="The solution is X",
            consensus_strength="unanimous",
        )

        insights = await extractor.extract(result)

        assert insights.consensus_insight is not None
        assert insights.consensus_insight.type == InsightType.CONSENSUS
        assert "unanimous" in insights.consensus_insight.title.lower()

    @pytest.mark.asyncio
    async def test_extract_no_consensus_insight_when_not_reached(self, extractor):
        """Should not extract consensus insight when no consensus."""
        result = MockDebateResult(consensus_reached=False)

        insights = await extractor.extract(result)

        assert insights.consensus_insight is None

    @pytest.mark.asyncio
    async def test_extract_dissent_insights(self, extractor):
        """Should extract dissent insights."""
        result = MockDebateResult(
            dissenting_views=[
                "[claude]: I disagree because X",
                "[gpt4]: Alternative approach is Y",
            ],
        )

        insights = await extractor.extract(result)

        assert len(insights.dissent_insights) == 2
        assert all(i.type == InsightType.DISSENT for i in insights.dissent_insights)

    @pytest.mark.asyncio
    async def test_extract_failure_mode_when_no_consensus(self, extractor):
        """Should extract failure mode when consensus not reached."""
        result = MockDebateResult(
            consensus_reached=False,
            votes=[MockVote(agent="a", choice="X"), MockVote(agent="b", choice="Y")],
        )

        insights = await extractor.extract(result)

        assert insights.failure_mode_insight is not None
        assert insights.failure_mode_insight.type == InsightType.FAILURE_MODE

    @pytest.mark.asyncio
    async def test_extract_pattern_insights(self, extractor):
        """Should extract pattern insights from critiques."""
        result = MockDebateResult(
            critiques=[
                MockCritique(agent="a", issues=["security vulnerability found"]),
                MockCritique(agent="b", issues=["authentication issue"]),
                MockCritique(agent="c", issues=["performance problem"]),
            ],
        )

        insights = await extractor.extract(result)

        # Should find security pattern (2 mentions)
        security_patterns = [
            p for p in insights.pattern_insights
            if "security" in p.title.lower()
        ]
        assert len(security_patterns) == 1

    @pytest.mark.asyncio
    async def test_extract_convergence_insight(self, extractor):
        """Should extract convergence insight from messages."""
        # Create messages with decreasing length (convergence)
        result = MockDebateResult(
            messages=[
                MockMessage(content="A" * 100),
                MockMessage(content="B" * 100),
                MockMessage(content="C" * 50),
                MockMessage(content="D" * 50),
            ],
        )

        insights = await extractor.extract(result)

        assert insights.convergence_insight is not None
        assert insights.convergence_insight.type == InsightType.CONVERGENCE

    @pytest.mark.asyncio
    async def test_extract_decision_insight(self, extractor):
        """Should extract decision process insight."""
        result = MockDebateResult(
            votes=[
                MockVote(agent="a", choice="X"),
                MockVote(agent="b", choice="X"),
            ],
            rounds_used=3,
        )

        insights = await extractor.extract(result)

        assert insights.decision_insight is not None
        assert insights.decision_insight.type == InsightType.DECISION_PROCESS

    @pytest.mark.asyncio
    async def test_extract_agent_performances(self, extractor):
        """Should extract agent performance metrics."""
        result = MockDebateResult(
            messages=[
                {"agent": "claude", "content": "X"},
                {"agent": "claude", "content": "Y"},
                {"agent": "gpt4", "content": "Z"},
            ],
            critiques=[
                MockCritique(agent="gpt4", target_agent="claude", severity=0.3),
            ],
        )

        insights = await extractor.extract(result)

        assert len(insights.agent_performances) >= 2

        claude_perf = next(
            (p for p in insights.agent_performances if p.agent_name == "claude"),
            None,
        )
        assert claude_perf is not None
        assert claude_perf.proposals_made == 2

    @pytest.mark.asyncio
    async def test_key_takeaway_consensus(self, extractor):
        """Should generate key takeaway for consensus."""
        result = MockDebateResult(
            consensus_reached=True,
            consensus_strength="strong",
            duration_seconds=90.0,
        )

        insights = await extractor.extract(result)

        assert "consensus" in insights.key_takeaway.lower()

    @pytest.mark.asyncio
    async def test_key_takeaway_no_consensus(self, extractor):
        """Should generate key takeaway for no consensus."""
        result = MockDebateResult(
            consensus_reached=False,
        )

        insights = await extractor.extract(result)

        assert "consensus" in insights.key_takeaway.lower()


class TestInsightExtractorHelpers:
    """Tests for InsightExtractor helper methods."""

    @pytest.fixture
    def extractor(self):
        return InsightExtractor()

    def test_categorize_issue_security(self, extractor):
        """Should categorize security issues."""
        assert extractor._categorize_issue("SQL injection risk") == "security"
        assert extractor._categorize_issue("authentication bypass") == "security"

    def test_categorize_issue_performance(self, extractor):
        """Should categorize performance issues."""
        assert extractor._categorize_issue("slow response time") == "performance"
        assert extractor._categorize_issue("optimize query") == "performance"

    def test_categorize_issue_correctness(self, extractor):
        """Should categorize correctness issues."""
        assert extractor._categorize_issue("bug in calculation") == "correctness"
        assert extractor._categorize_issue("error handling missing") == "correctness"

    def test_categorize_issue_unknown(self, extractor):
        """Should return None for uncategorized issues."""
        assert extractor._categorize_issue("some random issue") is None

    def test_get_agent_names_from_messages(self, extractor):
        """Should extract agent names from messages."""
        result = MockDebateResult(
            messages=[
                {"agent": "claude", "content": "x"},
                {"agent": "gpt4", "content": "y"},
                {"agent": "claude", "content": "z"},
            ],
        )

        names = extractor._get_agent_names(result)

        assert set(names) == {"claude", "gpt4"}

    def test_get_agent_names_from_critiques(self, extractor):
        """Should extract agent names from critiques."""
        result = MockDebateResult(
            critiques=[MockCritique(agent="gemini")],
        )

        names = extractor._get_agent_names(result)

        assert "gemini" in names

    def test_get_agent_names_from_votes(self, extractor):
        """Should extract agent names from votes."""
        result = MockDebateResult(
            votes=[MockVote(agent="mistral")],
        )

        names = extractor._get_agent_names(result)

        assert "mistral" in names


class TestInsightExtractorEdgeCases:
    """Edge case tests for InsightExtractor."""

    @pytest.fixture
    def extractor(self):
        return InsightExtractor()

    @pytest.mark.asyncio
    async def test_extract_with_empty_result(self, extractor):
        """Should handle empty result gracefully."""
        result = MockDebateResult(
            id="empty",
            task="",
            messages=[],
            critiques=[],
            votes=[],
        )

        insights = await extractor.extract(result)

        assert insights.debate_id == "empty"
        assert insights.total_insights >= 0

    @pytest.mark.asyncio
    async def test_extract_with_long_task(self, extractor):
        """Should truncate long tasks."""
        result = MockDebateResult(
            task="A" * 500,
        )

        insights = await extractor.extract(result)

        assert len(insights.task) == 200

    @pytest.mark.asyncio
    async def test_extract_with_dict_messages(self, extractor):
        """Should handle dict-style messages."""
        result = MockDebateResult()
        result.messages = [{"agent": "claude", "content": "test"}]

        insights = await extractor.extract(result)

        # Should not crash
        assert insights.debate_id is not None

    @pytest.mark.asyncio
    async def test_extract_generates_id_from_hash(self, extractor):
        """Should generate ID when not provided."""
        @dataclass
        class ResultWithoutId:
            task: str = "Test"
            consensus_reached: bool = False
            duration_seconds: float = 10.0

        result = ResultWithoutId()
        insights = await extractor.extract(result)

        assert insights.debate_id is not None
        assert len(insights.debate_id) == 16


# ============================================================================
# Additional Edge Case Tests (Phase 2 Coverage Improvements)
# ============================================================================

class TestInsightExtractorAdditionalEdgeCases:
    """Additional edge case tests for InsightExtractor."""

    @pytest.fixture
    def extractor(self):
        return InsightExtractor()

    @pytest.mark.asyncio
    async def test_empty_final_answer_string(self, extractor):
        """Should handle empty final_answer gracefully."""
        result = MockDebateResult(
            consensus_reached=True,
            final_answer="",
            consensus_strength="weak",
            confidence=0.5,
        )

        insights = await extractor.extract(result)

        # Should still create consensus insight
        assert insights.consensus_insight is not None
        # Empty answer should show placeholder
        assert "No answer recorded" in insights.consensus_insight.description

    @pytest.mark.asyncio
    async def test_confidence_zero(self, extractor):
        """Should handle zero confidence."""
        result = MockDebateResult(
            consensus_reached=True,
            confidence=0.0,
            final_answer="Answer",
            consensus_strength="weak",
        )

        insights = await extractor.extract(result)

        assert insights.consensus_insight.confidence == 0.0
        assert "0%" in insights.consensus_insight.description

    @pytest.mark.asyncio
    async def test_confidence_one(self, extractor):
        """Should handle confidence of 1.0."""
        result = MockDebateResult(
            consensus_reached=True,
            confidence=1.0,
            final_answer="Answer",
            consensus_strength="unanimous",
        )

        insights = await extractor.extract(result)

        assert insights.consensus_insight.confidence == 1.0

    @pytest.mark.asyncio
    async def test_confidence_out_of_range_high(self, extractor):
        """Should handle confidence > 1.0 (doesn't crash)."""
        result = MockDebateResult(
            consensus_reached=True,
            confidence=1.5,  # Invalid but shouldn't crash
            final_answer="Answer",
            consensus_strength="unanimous",
        )

        insights = await extractor.extract(result)

        # Should not crash, value passed through
        assert insights.consensus_insight is not None
        assert insights.consensus_insight.confidence == 1.5

    @pytest.mark.asyncio
    async def test_confidence_negative(self, extractor):
        """Should handle negative confidence (shouldn't crash)."""
        result = MockDebateResult(
            consensus_reached=True,
            confidence=-0.5,  # Invalid but shouldn't crash
            final_answer="Answer",
            consensus_strength="weak",
        )

        insights = await extractor.extract(result)

        assert insights.consensus_insight is not None

    @pytest.mark.asyncio
    async def test_average_severity_no_critiques(self, extractor):
        """Should handle average severity with no critiques received."""
        result = MockDebateResult(
            messages=[{"agent": "agent-1", "content": "msg"}],
            critiques=[],  # No critiques
        )

        insights = await extractor.extract(result)

        # Should not divide by zero
        if insights.agent_performances:
            agent1 = insights.agent_performances[0]
            assert agent1.average_critique_severity == 0.0

    @pytest.mark.asyncio
    async def test_average_severity_single_critique(self, extractor):
        """Should calculate average severity with single critique."""
        result = MockDebateResult(
            messages=[
                {"agent": "agent-1", "content": "msg"},
                {"agent": "agent-2", "content": "msg"},
            ],
            critiques=[
                MockCritique(agent="agent-2", target_agent="agent-1", severity=0.7),
            ],
        )

        insights = await extractor.extract(result)

        agent1 = next(
            (p for p in insights.agent_performances if p.agent_name == "agent-1"),
            None,
        )
        assert agent1 is not None
        assert agent1.critiques_received == 1
        assert agent1.average_critique_severity == 0.7

    @pytest.mark.asyncio
    async def test_very_large_debate_messages(self, extractor):
        """Should handle debates with many messages."""
        # Create a large debate with 500 messages
        large_messages = [
            {"agent": f"agent-{i % 5}", "content": f"Message {i}"}
            for i in range(500)
        ]

        result = MockDebateResult(
            messages=large_messages,
            duration_seconds=3600.0,
        )

        insights = await extractor.extract(result)

        # Should not crash and should have convergence insight
        assert insights.debate_id is not None
        assert insights.convergence_insight is not None
        # All 5 agents should have performances
        assert len(insights.agent_performances) == 5

    @pytest.mark.asyncio
    async def test_very_large_debate_critiques(self, extractor):
        """Should handle debates with many critiques."""
        # Create 100 critiques
        large_critiques = [
            MockCritique(
                agent=f"agent-{i % 3}",
                target_agent=f"agent-{(i + 1) % 3}",
                issues=[f"Issue {i}"],
                severity=0.5 + (i % 5) * 0.1,
            )
            for i in range(100)
        ]

        result = MockDebateResult(
            messages=[{"agent": "agent-0", "content": "test"}],
            critiques=large_critiques,
        )

        insights = await extractor.extract(result)

        # Should handle without crashing
        assert insights.debate_id is not None

    @pytest.mark.asyncio
    async def test_mixed_message_formats(self, extractor):
        """Should handle mixed dict and object message formats."""
        obj_msg = MockMessage(agent="claude", content="Object message")
        dict_msg = {"agent": "gpt4", "content": "Dict message"}

        result = MockDebateResult(
            messages=[obj_msg, dict_msg],
        )

        insights = await extractor.extract(result)

        # Should extract agents from both formats
        agent_names = [p.agent_name for p in insights.agent_performances]
        assert "claude" in agent_names or "gpt4" in agent_names

    @pytest.mark.asyncio
    async def test_null_bytes_in_strings(self, extractor):
        """Should handle null bytes in strings without crashing."""
        result = MockDebateResult(
            task="Task with \x00 null byte",
            final_answer="Answer with \x00 null",
            dissenting_views=["[agent]: View with \x00 null"],
        )

        insights = await extractor.extract(result)

        # Should not crash
        assert insights.debate_id is not None

    @pytest.mark.asyncio
    async def test_unicode_in_strings(self, extractor):
        """Should handle unicode characters properly."""
        result = MockDebateResult(
            task="Task with émojis 🎉 and ü nicode",
            final_answer="答え: The answer with 日本語 and αβγ",
            dissenting_views=["[агент]: Россия dissent with Ελληνικά"],
        )

        insights = await extractor.extract(result)

        # Should preserve unicode
        assert "émojis" in insights.task or "emoji" in insights.task.lower()
        assert insights.debate_id is not None

    @pytest.mark.asyncio
    async def test_whitespace_only_final_answer(self, extractor):
        """Should handle whitespace-only final answer."""
        result = MockDebateResult(
            consensus_reached=True,
            final_answer="   \n\t  ",
            confidence=0.5,
            consensus_strength="weak",
        )

        insights = await extractor.extract(result)

        # Should handle as effectively empty
        assert insights.consensus_insight is not None

    @pytest.mark.asyncio
    async def test_none_values_in_result(self, extractor):
        """Should handle None values in result attributes."""
        result = MockDebateResult(
            task=None,
            final_answer=None,
        )
        # Force some attributes to None
        result.messages = None
        result.critiques = None

        # Should handle gracefully (may raise or return partial results)
        try:
            insights = await extractor.extract(result)
            assert insights is not None
        except (TypeError, AttributeError):
            # Acceptable to fail on None values
            pass

    @pytest.mark.asyncio
    async def test_very_long_dissenting_view(self, extractor):
        """Should truncate very long dissenting views."""
        long_view = "[agent]: " + "x" * 10000

        result = MockDebateResult(
            dissenting_views=[long_view],
        )

        insights = await extractor.extract(result)

        # Should truncate
        if insights.dissent_insights:
            assert len(insights.dissent_insights[0].description) < 500

    @pytest.mark.asyncio
    async def test_special_characters_in_agent_name(self, extractor):
        """Should handle special characters in agent names."""
        result = MockDebateResult(
            messages=[
                {"agent": "agent<script>", "content": "msg"},
                {"agent": "agent&name", "content": "msg"},
                {"agent": "agent\"quotes\"", "content": "msg"},
            ],
        )

        insights = await extractor.extract(result)

        # Should not crash, names preserved
        agent_names = [p.agent_name for p in insights.agent_performances]
        assert len(agent_names) == 3

    @pytest.mark.asyncio
    async def test_empty_issues_list_in_critique(self, extractor):
        """Should handle critiques with empty issues list."""
        result = MockDebateResult(
            critiques=[
                MockCritique(agent="a1", target_agent="a2", issues=[]),
                MockCritique(agent="a3", target_agent="a2", issues=[]),
            ],
        )

        insights = await extractor.extract(result)

        # Should not crash, no patterns found
        assert insights.pattern_insights == []

    @pytest.mark.asyncio
    async def test_zero_duration(self, extractor):
        """Should handle zero duration."""
        result = MockDebateResult(
            duration_seconds=0.0,
        )

        insights = await extractor.extract(result)

        assert insights.duration_seconds == 0.0

    @pytest.mark.asyncio
    async def test_negative_duration(self, extractor):
        """Should handle negative duration (shouldn't crash)."""
        result = MockDebateResult(
            duration_seconds=-100.0,
        )

        insights = await extractor.extract(result)

        # Shouldn't crash, passes through
        assert insights.duration_seconds == -100.0

    def test_categorize_issue_case_insensitive(self, extractor):
        """Issue categorization should be case insensitive."""
        assert extractor._categorize_issue("SECURITY vulnerability") == "security"
        assert extractor._categorize_issue("Performance ISSUE") == "performance"
        assert extractor._categorize_issue("BUG in code") == "correctness"

    def test_categorize_issue_partial_match(self, extractor):
        """Issue categorization should match partial keywords."""
        assert extractor._categorize_issue("vulnerabilities found") == "security"  # matches 'vulnerab'
        assert extractor._categorize_issue("readable code") == "clarity"  # matches 'readab'

    @pytest.mark.asyncio
    async def test_duplicate_agents_in_messages(self, extractor):
        """Should handle same agent appearing many times."""
        result = MockDebateResult(
            messages=[
                {"agent": "claude", "content": "msg1"},
                {"agent": "claude", "content": "msg2"},
                {"agent": "claude", "content": "msg3"},
            ],
        )

        insights = await extractor.extract(result)

        # Should have one performance entry with 3 proposals
        assert len(insights.agent_performances) == 1
        assert insights.agent_performances[0].proposals_made == 3
