"""
Safety-Critical Module Tests.

Tests for Byzantine fault-tolerant consensus and cognitive load limiting.
These modules protect debate integrity and prevent agent overload.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.byzantine import (
    ByzantineConsensus,
    ByzantineConsensusConfig,
    ByzantineConsensusResult,
    ByzantineMessage,
    ByzantinePhase,
    ConsensusFailure,
    ViewChangeReason,
    verify_with_byzantine_consensus,
)
from aragora.debate.cognitive_limiter import (
    CHARS_PER_TOKEN,
    STRESS_BUDGETS,
    CognitiveBudget,
    CognitiveLoadLimiter,
    limit_debate_context,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockAgent:
    """Mock agent for testing."""

    name: str
    _response: str = "PREPARE: YES\nREASONING: This proposal looks good."

    def set_response(self, response: str) -> None:
        self._response = response

    async def generate(self, prompt: str) -> str:
        return self._response


@dataclass
class MockMessage:
    """Mock message for testing."""

    content: str
    role: str = "assistant"


@dataclass
class MockCritique:
    """Mock critique for testing."""

    reasoning: str
    severity: float = 0.5
    issues: list = None
    suggestions: list = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.suggestions is None:
            self.suggestions = []


def create_mock_agents(count: int, cooperative: bool = True) -> list[MockAgent]:
    """Create mock agents for testing."""
    agents = []
    for i in range(count):
        agent = MockAgent(name=f"agent_{i}")
        if not cooperative:
            agent.set_response("PREPARE: NO\nREASONING: I disagree.")
        agents.append(agent)
    return agents


# =============================================================================
# Byzantine Phase Enum Tests
# =============================================================================


class TestByzantinePhase:
    """Tests for ByzantinePhase enum."""

    def test_all_phases_exist(self):
        """All PBFT phases should be defined."""
        assert ByzantinePhase.PRE_PREPARE.value == "pre_prepare"
        assert ByzantinePhase.PREPARE.value == "prepare"
        assert ByzantinePhase.COMMIT.value == "commit"
        assert ByzantinePhase.REPLY.value == "reply"
        assert ByzantinePhase.VIEW_CHANGE.value == "view_change"

    def test_phase_count(self):
        """Should have exactly 5 phases."""
        assert len(ByzantinePhase) == 5


class TestViewChangeReason:
    """Tests for ViewChangeReason enum."""

    def test_all_reasons_exist(self):
        """All view change reasons should be defined."""
        assert ViewChangeReason.LEADER_TIMEOUT.value == "leader_timeout"
        assert ViewChangeReason.LEADER_FAILURE.value == "leader_failure"
        assert ViewChangeReason.INVALID_PROPOSAL.value == "invalid_proposal"
        assert ViewChangeReason.CONSENSUS_STALL.value == "consensus_stall"


# =============================================================================
# Byzantine Message Tests
# =============================================================================


class TestByzantineMessage:
    """Tests for ByzantineMessage dataclass."""

    def test_message_creation(self):
        """Message should be created with required fields."""
        msg = ByzantineMessage(
            phase=ByzantinePhase.PRE_PREPARE,
            view=0,
            sequence=1,
            sender="leader",
            proposal_hash="abc123",
            proposal="Test proposal",
        )

        assert msg.phase == ByzantinePhase.PRE_PREPARE
        assert msg.view == 0
        assert msg.sequence == 1
        assert msg.sender == "leader"
        assert msg.proposal_hash == "abc123"
        assert msg.proposal == "Test proposal"
        assert msg.timestamp  # Auto-generated

    def test_message_without_proposal(self):
        """Non-PRE_PREPARE messages don't need proposal."""
        msg = ByzantineMessage(
            phase=ByzantinePhase.PREPARE,
            view=0,
            sequence=1,
            sender="agent_1",
            proposal_hash="abc123",
        )

        assert msg.proposal is None

    def test_compute_hash(self):
        """Hash should be deterministic."""
        msg = ByzantineMessage(
            phase=ByzantinePhase.PRE_PREPARE,
            view=0,
            sequence=1,
            sender="leader",
            proposal_hash="abc123",
        )

        hash1 = msg.compute_hash()
        hash2 = msg.compute_hash()

        assert hash1 == hash2
        assert len(hash1) == 16

    def test_different_messages_different_hashes(self):
        """Different messages should have different hashes."""
        msg1 = ByzantineMessage(
            phase=ByzantinePhase.PRE_PREPARE,
            view=0,
            sequence=1,
            sender="leader",
            proposal_hash="abc123",
        )
        msg2 = ByzantineMessage(
            phase=ByzantinePhase.PRE_PREPARE,
            view=1,  # Different view
            sequence=1,
            sender="leader",
            proposal_hash="abc123",
        )

        assert msg1.compute_hash() != msg2.compute_hash()


# =============================================================================
# Byzantine Consensus Config Tests
# =============================================================================


class TestByzantineConsensusConfig:
    """Tests for ByzantineConsensusConfig."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = ByzantineConsensusConfig()

        assert config.max_faulty_fraction == 0.33
        assert config.phase_timeout_seconds == 30.0
        assert config.view_change_timeout_seconds == 60.0
        assert config.min_agents == 4
        assert config.max_view_changes == 3
        assert config.max_retries_per_phase == 2

    def test_custom_config(self):
        """Custom config values should be respected."""
        config = ByzantineConsensusConfig(
            max_faulty_fraction=0.25,
            phase_timeout_seconds=60.0,
            min_agents=7,
        )

        assert config.max_faulty_fraction == 0.25
        assert config.phase_timeout_seconds == 60.0
        assert config.min_agents == 7


# =============================================================================
# Byzantine Consensus Result Tests
# =============================================================================


class TestByzantineConsensusResult:
    """Tests for ByzantineConsensusResult."""

    def test_successful_result(self):
        """Successful result should have correct fields."""
        result = ByzantineConsensusResult(
            success=True,
            value="Agreed value",
            confidence=0.8,
            commit_count=4,
            total_agents=5,
        )

        assert result.success
        assert result.value == "Agreed value"
        assert result.confidence == 0.8

    def test_agreement_ratio(self):
        """Agreement ratio should be calculated correctly."""
        result = ByzantineConsensusResult(
            success=True,
            commit_count=4,
            total_agents=5,
        )

        assert result.agreement_ratio == 0.8

    def test_agreement_ratio_zero_agents(self):
        """Agreement ratio should be 0 with no agents."""
        result = ByzantineConsensusResult(
            success=False,
            commit_count=0,
            total_agents=0,
        )

        assert result.agreement_ratio == 0.0

    def test_failed_result(self):
        """Failed result should have failure reason."""
        result = ByzantineConsensusResult(
            success=False,
            failure_reason="Consensus failed after 3 view changes",
            total_agents=5,
        )

        assert not result.success
        assert result.failure_reason
        assert result.value is None


# =============================================================================
# Byzantine Consensus Protocol Tests
# =============================================================================


class TestByzantineConsensusProperties:
    """Tests for ByzantineConsensus properties."""

    def test_n_property(self):
        """n should be total agent count."""
        agents = create_mock_agents(5)
        protocol = ByzantineConsensus(agents=agents)

        assert protocol.n == 5

    def test_f_property(self):
        """f should be (n-1) // 3."""
        # n=4: f=(4-1)//3 = 1
        agents4 = create_mock_agents(4)
        assert ByzantineConsensus(agents=agents4).f == 1

        # n=7: f=(7-1)//3 = 2
        agents7 = create_mock_agents(7)
        assert ByzantineConsensus(agents=agents7).f == 2

        # n=10: f=(10-1)//3 = 3
        agents10 = create_mock_agents(10)
        assert ByzantineConsensus(agents=agents10).f == 3

    def test_quorum_size(self):
        """Quorum should be 2f + 1."""
        agents = create_mock_agents(7)
        protocol = ByzantineConsensus(agents=agents)

        # n=7, f=2, quorum = 2*2+1 = 5
        assert protocol.quorum_size == 5

    def test_leader_rotation(self):
        """Leader should rotate with view number."""
        agents = create_mock_agents(4)
        protocol = ByzantineConsensus(agents=agents)

        # View 0: agent 0
        assert protocol.leader.name == "agent_0"

        # Simulate view change
        protocol._current_view = 1
        assert protocol.leader.name == "agent_1"

        protocol._current_view = 4  # Wraps around
        assert protocol.leader.name == "agent_0"


class TestByzantineConsensusVoting:
    """Tests for vote collection and parsing."""

    def test_parse_agreement_explicit_yes(self):
        """Explicit YES should be parsed as agreement."""
        agents = create_mock_agents(4)
        protocol = ByzantineConsensus(agents=agents)

        assert protocol._parse_agreement("PREPARE: YES\nREASONING: Looks good")
        assert protocol._parse_agreement("COMMIT: YES")

    def test_parse_agreement_explicit_no(self):
        """Explicit NO should be parsed as disagreement."""
        agents = create_mock_agents(4)
        protocol = ByzantineConsensus(agents=agents)

        assert not protocol._parse_agreement("PREPARE: NO\nREASONING: Issues found")
        assert not protocol._parse_agreement("COMMIT: NO")

    def test_parse_agreement_case_insensitive(self):
        """Parsing should be case insensitive."""
        agents = create_mock_agents(4)
        protocol = ByzantineConsensus(agents=agents)

        assert protocol._parse_agreement("prepare: yes")
        assert not protocol._parse_agreement("PREPARE: no")

    def test_parse_agreement_fallback_words(self):
        """Should fall back to agreement words."""
        agents = create_mock_agents(4)
        protocol = ByzantineConsensus(agents=agents)

        assert protocol._parse_agreement("I agree with this proposal")
        assert protocol._parse_agreement("I accept the decision")
        assert not protocol._parse_agreement("I disagree and reject this")

    def test_compute_proposal_hash(self):
        """Proposal hash should be deterministic."""
        agents = create_mock_agents(4)
        protocol = ByzantineConsensus(agents=agents)

        hash1 = protocol._compute_proposal_hash("Test proposal")
        hash2 = protocol._compute_proposal_hash("Test proposal")

        assert hash1 == hash2
        assert len(hash1) == 16


class TestByzantineConsensusExecution:
    """Tests for consensus execution."""

    @pytest.mark.asyncio
    async def test_successful_consensus(self):
        """All cooperative agents should reach consensus."""
        agents = create_mock_agents(4, cooperative=True)
        protocol = ByzantineConsensus(agents=agents)

        result = await protocol.propose("Test proposal", task="Test task")

        assert result.success
        assert result.value == "Test proposal"
        assert result.confidence > 0.5
        assert result.commit_count >= protocol.quorum_size

    @pytest.mark.asyncio
    async def test_consensus_with_minority_faulty(self):
        """Consensus should succeed with minority faulty agents."""
        agents = create_mock_agents(4)
        # Make 1 agent (< f+1) faulty
        agents[0].set_response("PREPARE: NO\nREASONING: I disagree.")

        protocol = ByzantineConsensus(agents=agents)
        result = await protocol.propose("Test proposal")

        # Should still succeed with 3/4 agents
        assert result.success
        assert result.commit_count >= 3

    @pytest.mark.asyncio
    async def test_consensus_fails_with_majority_faulty(self):
        """Consensus should fail with majority faulty agents."""
        agents = create_mock_agents(4, cooperative=False)

        protocol = ByzantineConsensus(agents=agents)
        result = await protocol.propose("Test proposal")

        assert not result.success
        assert "view changes" in result.failure_reason.lower()

    @pytest.mark.asyncio
    async def test_sequence_increments(self):
        """Sequence number should increment with each proposal."""
        agents = create_mock_agents(4)
        protocol = ByzantineConsensus(agents=agents)

        result1 = await protocol.propose("Proposal 1")
        result2 = await protocol.propose("Proposal 2")

        assert result2.sequence > result1.sequence

    @pytest.mark.asyncio
    async def test_view_change_on_failure(self):
        """View should change on consensus failure."""
        agents = create_mock_agents(4)
        # Make enough agents faulty to fail prepare phase
        for agent in agents[:3]:
            agent.set_response("PREPARE: NO")

        protocol = ByzantineConsensus(agents=agents)
        initial_view = protocol._current_view

        await protocol.propose("Test proposal")

        # View should have changed
        assert protocol._current_view > initial_view


class TestByzantineConsensusTimeout:
    """Tests for timeout handling."""

    @pytest.mark.asyncio
    async def test_agent_timeout_during_prepare(self):
        """Timeout during prepare should not block consensus."""
        agents = create_mock_agents(4)

        # Make one agent timeout
        async def slow_generate(prompt):
            await asyncio.sleep(100)  # Will timeout
            return "PREPARE: YES"

        agents[0].generate = slow_generate

        config = ByzantineConsensusConfig(phase_timeout_seconds=0.1)
        protocol = ByzantineConsensus(agents=agents, config=config)

        result = await protocol.propose("Test proposal")

        # Should still succeed with remaining agents
        assert result.success
        assert result.commit_count >= 3


class TestVerifyWithByzantineConsensus:
    """Tests for convenience function."""

    @pytest.mark.asyncio
    async def test_verify_function(self):
        """Convenience function should work correctly."""
        agents = create_mock_agents(4)

        result = await verify_with_byzantine_consensus(
            proposal="Test value",
            agents=agents,
            task="Verification task",
            fault_tolerance=0.33,
        )

        assert result.success
        assert isinstance(result, ByzantineConsensusResult)


# =============================================================================
# Cognitive Budget Tests
# =============================================================================


class TestCognitiveBudget:
    """Tests for CognitiveBudget dataclass."""

    def test_default_values(self):
        """Default budget should have sensible values."""
        budget = CognitiveBudget()

        assert budget.max_context_tokens == 6000
        assert budget.max_history_messages == 15
        assert budget.max_critique_chars == 800
        assert budget.max_proposal_chars == 2000
        assert budget.max_patterns_chars == 500
        assert budget.reserve_for_response == 2000

    def test_max_context_chars(self):
        """max_context_chars should be tokens * CHARS_PER_TOKEN."""
        budget = CognitiveBudget(max_context_tokens=1000)

        assert budget.max_context_chars == 1000 * CHARS_PER_TOKEN

    def test_scale_budgets(self):
        """Scale should multiply all budgets except response reserve."""
        budget = CognitiveBudget(
            max_context_tokens=1000,
            max_history_messages=10,
            max_critique_chars=100,
            max_proposal_chars=200,
            max_patterns_chars=50,
            reserve_for_response=500,
        )

        scaled = budget.scale(0.5)

        assert scaled.max_context_tokens == 500
        assert scaled.max_history_messages == 5
        assert scaled.max_critique_chars == 50
        assert scaled.max_proposal_chars == 100
        assert scaled.max_patterns_chars == 25
        assert scaled.reserve_for_response == 500  # Unchanged

    def test_scale_minimum_messages(self):
        """Scaling should maintain minimum 3 messages."""
        budget = CognitiveBudget(max_history_messages=4)

        scaled = budget.scale(0.1)

        assert scaled.max_history_messages == 3  # Minimum


# =============================================================================
# Stress Budget Tests
# =============================================================================


class TestStressBudgets:
    """Tests for stress level budgets."""

    def test_all_levels_exist(self):
        """All stress levels should have budgets."""
        assert "nominal" in STRESS_BUDGETS
        assert "elevated" in STRESS_BUDGETS
        assert "high" in STRESS_BUDGETS
        assert "critical" in STRESS_BUDGETS

    def test_budgets_decrease_with_stress(self):
        """Budgets should decrease as stress increases."""
        nominal = STRESS_BUDGETS["nominal"]
        elevated = STRESS_BUDGETS["elevated"]
        high = STRESS_BUDGETS["high"]
        critical = STRESS_BUDGETS["critical"]

        # Tokens
        assert nominal.max_context_tokens > elevated.max_context_tokens
        assert elevated.max_context_tokens > high.max_context_tokens
        assert high.max_context_tokens > critical.max_context_tokens

        # Messages
        assert nominal.max_history_messages > elevated.max_history_messages
        assert elevated.max_history_messages > high.max_history_messages
        assert high.max_history_messages > critical.max_history_messages

    def test_budget_values(self):
        """Verify specific budget values."""
        assert STRESS_BUDGETS["nominal"].max_context_tokens == 8000
        assert STRESS_BUDGETS["elevated"].max_context_tokens == 6000
        assert STRESS_BUDGETS["high"].max_context_tokens == 4000
        assert STRESS_BUDGETS["critical"].max_context_tokens == 2000


# =============================================================================
# Cognitive Load Limiter Tests
# =============================================================================


class TestCognitiveLoadLimiterInit:
    """Tests for CognitiveLoadLimiter initialization."""

    def test_default_budget(self):
        """Default budget should be elevated."""
        limiter = CognitiveLoadLimiter()

        assert limiter.budget == STRESS_BUDGETS["elevated"]

    def test_custom_budget(self):
        """Custom budget should be used."""
        custom = CognitiveBudget(max_context_tokens=1000)
        limiter = CognitiveLoadLimiter(budget=custom)

        assert limiter.budget.max_context_tokens == 1000

    def test_stats_initialized(self):
        """Stats should be initialized to zero."""
        limiter = CognitiveLoadLimiter()

        assert limiter.stats["messages_truncated"] == 0
        assert limiter.stats["critiques_truncated"] == 0
        assert limiter.stats["total_chars_removed"] == 0


class TestCognitiveLoadLimiterFactories:
    """Tests for factory methods."""

    def test_for_stress_level(self):
        """for_stress_level should create correct limiter."""
        limiter = CognitiveLoadLimiter.for_stress_level("high")

        assert limiter.budget == STRESS_BUDGETS["high"]

    def test_for_unknown_stress_level(self):
        """Unknown level should default to elevated."""
        limiter = CognitiveLoadLimiter.for_stress_level("unknown")

        assert limiter.budget == STRESS_BUDGETS["elevated"]

    def test_from_governor_import_error(self):
        """from_governor should handle import error gracefully."""
        with patch.dict("sys.modules", {"aragora.debate.complexity_governor": None}):
            limiter = CognitiveLoadLimiter.from_governor()

        assert isinstance(limiter, CognitiveLoadLimiter)


class TestCognitiveLoadLimiterTokenEstimation:
    """Tests for token estimation."""

    def test_estimate_tokens(self):
        """Token estimation should use CHARS_PER_TOKEN."""
        limiter = CognitiveLoadLimiter()

        # 20 chars / 4 = 5 tokens
        assert limiter.estimate_tokens("x" * 20) == 5

    def test_estimate_tokens_empty(self):
        """Empty string should have 0 tokens."""
        limiter = CognitiveLoadLimiter()

        assert limiter.estimate_tokens("") == 0


class TestCognitiveLoadLimiterMessages:
    """Tests for message limiting."""

    def test_limit_empty_messages(self):
        """Empty list should return empty list."""
        limiter = CognitiveLoadLimiter()

        result = limiter.limit_messages([])

        assert result == []

    def test_limit_within_budget(self):
        """Messages within budget should not be truncated."""
        limiter = CognitiveLoadLimiter()
        messages = [MockMessage(content=f"Message {i}") for i in range(5)]

        result = limiter.limit_messages(messages)

        assert len(result) == 5

    def test_limit_exceeds_count(self):
        """Messages exceeding count should be truncated."""
        budget = CognitiveBudget(max_history_messages=5)
        limiter = CognitiveLoadLimiter(budget=budget)
        messages = [MockMessage(content=f"Message {i}") for i in range(10)]

        result = limiter.limit_messages(messages)

        # Should keep first + last 4
        assert len(result) == 5
        assert result[0].content == "Message 0"  # First preserved
        assert result[-1].content == "Message 9"  # Last preserved

    def test_limit_preserves_first_message(self):
        """First message (task) should always be preserved."""
        budget = CognitiveBudget(max_history_messages=3)
        limiter = CognitiveLoadLimiter(budget=budget)
        messages = [MockMessage(content=f"Message {i}") for i in range(10)]

        result = limiter.limit_messages(messages)

        assert result[0].content == "Message 0"

    def test_limit_stats_updated(self):
        """Stats should track truncations."""
        budget = CognitiveBudget(max_history_messages=5)
        limiter = CognitiveLoadLimiter(budget=budget)
        messages = [MockMessage(content=f"Message {i}") for i in range(10)]

        limiter.limit_messages(messages)

        assert limiter.stats["messages_truncated"] == 5


class TestCognitiveLoadLimiterCritiques:
    """Tests for critique limiting."""

    def test_limit_empty_critiques(self):
        """Empty list should return empty list."""
        limiter = CognitiveLoadLimiter()

        result = limiter.limit_critiques([])

        assert result == []

    def test_limit_critiques_count(self):
        """Should respect max critique count."""
        limiter = CognitiveLoadLimiter()
        critiques = [MockCritique(reasoning=f"Critique {i}") for i in range(10)]

        result = limiter.limit_critiques(critiques, max_critiques=5)

        assert len(result) == 5

    def test_limit_critiques_by_severity(self):
        """Higher severity critiques should be prioritized."""
        limiter = CognitiveLoadLimiter()
        critiques = [
            MockCritique(reasoning="Low", severity=0.2),
            MockCritique(reasoning="High", severity=0.9),
            MockCritique(reasoning="Medium", severity=0.5),
        ]

        result = limiter.limit_critiques(critiques, max_critiques=2)

        # Should have high and medium
        assert len(result) == 2
        assert result[0].severity == 0.9
        assert result[1].severity == 0.5

    def test_limit_long_critique(self):
        """Long critiques should be summarized."""
        limiter = CognitiveLoadLimiter()
        long_reasoning = "x" * 1000
        critique = MockCritique(
            reasoning=long_reasoning,
            issues=["Issue 1", "Issue 2"],
            suggestions=["Fix 1"],
        )

        result = limiter.limit_critiques([critique], max_chars_per=100)

        # Should be summarized
        assert len(result[0].reasoning) <= 100
        assert limiter.stats["critiques_truncated"] == 1


class TestCognitiveLoadLimiterContext:
    """Tests for full context limiting."""

    def test_limit_context_messages_only(self):
        """Should handle messages-only context."""
        limiter = CognitiveLoadLimiter()
        messages = [MockMessage(content="Test") for _ in range(3)]

        result = limiter.limit_context(messages=messages)

        assert "messages" in result
        assert len(result["messages"]) == 3

    def test_limit_context_all_components(self):
        """Should handle all context components."""
        limiter = CognitiveLoadLimiter()
        messages = [MockMessage(content="Test")]
        critiques = [MockCritique(reasoning="Critique")]
        patterns = "Pattern data"
        extra = "Extra context"

        result = limiter.limit_context(
            messages=messages,
            critiques=critiques,
            patterns=patterns,
            extra_context=extra,
        )

        assert "messages" in result
        assert "critiques" in result
        assert "patterns" in result
        assert "extra_context" in result

    def test_limit_context_truncates_patterns(self):
        """Long patterns should be truncated."""
        budget = CognitiveBudget(max_patterns_chars=10)
        limiter = CognitiveLoadLimiter(budget=budget)

        result = limiter.limit_context(patterns="x" * 100)

        assert len(result["patterns"]) <= 14  # 10 + "..."


class TestCognitiveLoadLimiterStats:
    """Tests for statistics tracking."""

    def test_get_stats(self):
        """get_stats should return all stats."""
        limiter = CognitiveLoadLimiter()

        stats = limiter.get_stats()

        assert "messages_truncated" in stats
        assert "critiques_truncated" in stats
        assert "total_chars_removed" in stats
        assert "budget" in stats

    def test_reset_stats(self):
        """reset_stats should clear counters."""
        limiter = CognitiveLoadLimiter()
        limiter.stats["messages_truncated"] = 10

        limiter.reset_stats()

        assert limiter.stats["messages_truncated"] == 0


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestLimitDebateContext:
    """Tests for limit_debate_context function."""

    def test_limit_debate_context(self):
        """Convenience function should work correctly."""
        messages = [MockMessage(content="Test")]
        critiques = [MockCritique(reasoning="Critique")]

        result = limit_debate_context(
            messages=messages,
            critiques=critiques,
            stress_level="high",
        )

        assert "messages" in result
        assert "critiques" in result

    def test_limit_debate_context_default_level(self):
        """Default stress level should be elevated."""
        messages = [MockMessage(content="Test")]

        result = limit_debate_context(messages=messages)

        assert "messages" in result


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_byzantine_single_agent(self):
        """Single agent should still work (but warn)."""
        agents = create_mock_agents(1)
        protocol = ByzantineConsensus(agents=agents)

        assert protocol.n == 1
        assert protocol.f == 0
        assert protocol.quorum_size == 1

    @pytest.mark.asyncio
    async def test_byzantine_empty_proposal(self):
        """Empty proposal should be handled."""
        agents = create_mock_agents(4)
        protocol = ByzantineConsensus(agents=agents)

        result = await protocol.propose("")

        assert result.success
        assert result.value == ""

    def test_cognitive_limiter_string_messages(self):
        """Should handle plain string messages."""
        limiter = CognitiveLoadLimiter()
        messages = ["Message 1", "Message 2", "Message 3"]

        result = limiter.limit_messages(messages, max_messages=2)

        # Should work with strings
        assert len(result) == 2

    def test_cognitive_limiter_very_long_message(self):
        """Very long message in multi-message list should be truncated."""
        budget = CognitiveBudget(max_context_tokens=100)  # 400 chars
        limiter = CognitiveLoadLimiter(budget=budget)
        # First message (task) is always preserved, second can be truncated
        messages = [
            MockMessage(content="Task description"),
            MockMessage(content="x" * 1000),  # Long message
        ]

        result = limiter.limit_messages(messages, max_chars=400)

        # Second message should be truncated to fit budget
        assert len(result) == 2
        assert result[0].content == "Task description"  # First preserved
        # Second message truncated with marker
        if len(result) > 1 and hasattr(result[1], "content"):
            assert len(result[1].content) < 1000 or "[... truncated ...]" in str(result[1])

    def test_cognitive_limiter_critique_no_issues(self):
        """Critique without issues should still summarize."""
        limiter = CognitiveLoadLimiter()
        critique = MockCritique(
            reasoning="x" * 1000,
            issues=[],
            suggestions=[],
        )

        result = limiter.limit_critiques([critique], max_chars_per=100)

        # Should truncate reasoning - check the reasoning attribute
        assert len(result[0].reasoning) <= 103  # 100 + "..."
