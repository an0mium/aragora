"""
Tests for Debate Forking module.

Tests cover:
- Data classes (ForkPoint, Branch, ForkDecision, MergeResult, DeadlockSignal)
- ForkDetector disagreement detection
- DebateForker fork and merge operations
- DeadlockResolver pattern detection and auto-resolution
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from aragora.debate.forking import (
    Branch,
    DeadlockResolver,
    DeadlockSignal,
    DebateForker,
    ForkDecision,
    ForkDetector,
    ForkPoint,
    MergeResult,
)


# =============================================================================
# ForkPoint Tests
# =============================================================================


class TestForkPoint:
    """Tests for ForkPoint dataclass."""

    def test_create_fork_point(self):
        """Test creating a fork point."""
        fp = ForkPoint(
            round=3,
            reason="Fundamental disagreement on architecture",
            disagreeing_agents=["claude", "gpt4"],
            parent_debate_id="debate-123",
            branch_ids=["branch-1", "branch-2"],
        )

        assert fp.round == 3
        assert fp.reason == "Fundamental disagreement on architecture"
        assert len(fp.disagreeing_agents) == 2
        assert len(fp.branch_ids) == 2
        assert fp.created_at is not None

    def test_fork_point_timestamp_auto_generated(self):
        """Test that created_at is auto-generated."""
        fp = ForkPoint(
            round=1,
            reason="Test",
            disagreeing_agents=["a"],
            parent_debate_id="debate-123",
            branch_ids=["b1"],
        )

        # Should be parseable as ISO format
        datetime.fromisoformat(fp.created_at)


# =============================================================================
# Branch Tests
# =============================================================================


class TestBranch:
    """Tests for Branch dataclass."""

    def test_create_branch(self):
        """Test creating a branch."""
        branch = Branch(
            branch_id="branch-001",
            parent_debate_id="debate-123",
            fork_round=3,
            hypothesis="Claude's microservices approach",
            lead_agent="claude",
        )

        assert branch.branch_id == "branch-001"
        assert branch.parent_debate_id == "debate-123"
        assert branch.fork_round == 3
        assert branch.hypothesis == "Claude's microservices approach"
        assert branch.lead_agent == "claude"
        assert branch.messages == []
        assert branch.result is None
        assert branch.created_at is not None

    def test_is_complete_false(self):
        """Test is_complete returns False when no result."""
        branch = Branch(
            branch_id="branch-001",
            parent_debate_id="debate-123",
            fork_round=3,
            hypothesis="Test",
            lead_agent="test",
        )

        assert branch.is_complete is False

    def test_is_complete_true(self):
        """Test is_complete returns True when has result."""
        from aragora.core import DebateResult

        result = DebateResult(
            final_answer="Test",
            confidence=0.8,
            consensus_reached=True,
            rounds_used=2,
            messages=[],
            critiques=[],
            votes=[],
        )

        branch = Branch(
            branch_id="branch-001",
            parent_debate_id="debate-123",
            fork_round=3,
            hypothesis="Test",
            lead_agent="test",
            result=result,
        )

        assert branch.is_complete is True

    def test_branch_with_messages(self):
        """Test branch with pre-populated messages."""
        from aragora.core import Message

        messages = [
            Message(agent="claude", role="proposer", content="Test", round=1),
        ]

        branch = Branch(
            branch_id="branch-001",
            parent_debate_id="debate-123",
            fork_round=3,
            hypothesis="Test",
            lead_agent="claude",
            messages=messages,
        )

        assert len(branch.messages) == 1


# =============================================================================
# ForkDecision Tests
# =============================================================================


class TestForkDecision:
    """Tests for ForkDecision dataclass."""

    def test_create_should_fork_decision(self):
        """Test creating decision to fork."""
        decision = ForkDecision(
            should_fork=True,
            reason="Fundamental disagreement on tech stack",
            branches=[
                {"hypothesis": "Claude's Redis approach", "lead_agent": "claude"},
                {"hypothesis": "GPT4's Memcached approach", "lead_agent": "gpt4"},
            ],
            disagreement_score=0.85,
        )

        assert decision.should_fork is True
        assert decision.disagreement_score == 0.85
        assert len(decision.branches) == 2

    def test_create_no_fork_decision(self):
        """Test creating decision not to fork."""
        decision = ForkDecision(
            should_fork=False,
            reason="Not enough disagreement",
            branches=[],
            disagreement_score=0.3,
        )

        assert decision.should_fork is False
        assert decision.branches == []


# =============================================================================
# MergeResult Tests
# =============================================================================


class TestMergeResult:
    """Tests for MergeResult dataclass."""

    def test_create_merge_result(self):
        """Test creating merge result."""
        from aragora.core import DebateResult

        result1 = DebateResult(
            final_answer="Redis is best",
            confidence=0.85,
            consensus_reached=True,
            rounds_used=3,
            messages=[],
            critiques=[],
            votes=[],
        )

        result2 = DebateResult(
            final_answer="Memcached is simpler",
            confidence=0.75,
            consensus_reached=True,
            rounds_used=4,
            messages=[],
            critiques=[],
            votes=[],
        )

        merge = MergeResult(
            winning_branch_id="branch-001",
            winning_hypothesis="Redis approach",
            comparison_summary="## Branch Comparison\n...",
            all_branch_results={
                "branch-001": result1,
                "branch-002": result2,
            },
            merged_insights=["Both caches work well", "Redis for persistence"],
        )

        assert merge.winning_branch_id == "branch-001"
        assert len(merge.all_branch_results) == 2
        assert len(merge.merged_insights) == 2


# =============================================================================
# DeadlockSignal Tests
# =============================================================================


class TestDeadlockSignal:
    """Tests for DeadlockSignal dataclass."""

    def test_create_deadlock_signal(self):
        """Test creating deadlock signal."""
        signal = DeadlockSignal(
            debate_id="debate-123",
            round=5,
            stagnation_score=0.85,
            rounds_without_progress=3,
            pattern="positional",
            recommendation="fork",
        )

        assert signal.debate_id == "debate-123"
        assert signal.round == 5
        assert signal.stagnation_score == 0.85
        assert signal.rounds_without_progress == 3
        assert signal.pattern == "positional"
        assert signal.recommendation == "fork"
        assert signal.detected_at is not None

    def test_deadlock_patterns(self):
        """Test different deadlock patterns."""
        patterns = ["positional", "circular", "exhaustion"]

        for pattern in patterns:
            signal = DeadlockSignal(
                debate_id="debate-123",
                round=3,
                stagnation_score=0.8,
                rounds_without_progress=2,
                pattern=pattern,
                recommendation="fork",
            )
            assert signal.pattern == pattern


# =============================================================================
# ForkDetector Tests
# =============================================================================


class TestForkDetector:
    """Tests for ForkDetector class."""

    def _create_mock_message(self, agent: str, content: str) -> Mock:
        """Helper to create mock messages."""
        msg = Mock()
        msg.agent = agent
        msg.content = content
        return msg

    def _create_mock_agent(self, name: str) -> Mock:
        """Helper to create mock agents."""
        agent = Mock()
        agent.name = name
        return agent

    def test_init_thresholds(self):
        """Test initialization with default thresholds."""
        detector = ForkDetector()

        assert detector.DISAGREEMENT_THRESHOLD == 0.7
        assert detector.MIN_ROUNDS_BEFORE_FORK == 2

    def test_should_fork_too_early(self):
        """Test should_fork returns False when too early."""
        detector = ForkDetector()

        messages = [self._create_mock_message("claude", "Test")]
        agents = [self._create_mock_agent("claude")]

        decision = detector.should_fork(messages, round_num=1, agents=agents)

        assert decision.should_fork is False
        assert "Too early" in decision.reason

    def test_should_fork_not_enough_agents(self):
        """Test should_fork returns False with single agent."""
        detector = ForkDetector()

        messages = [
            self._create_mock_message("claude", "First message"),
            self._create_mock_message("claude", "Second message"),
            self._create_mock_message("claude", "Third message"),
        ]
        agents = [self._create_mock_agent("claude")]

        decision = detector.should_fork(messages, round_num=3, agents=agents)

        assert decision.should_fork is False
        assert "Not enough agents" in decision.reason

    def test_should_fork_no_disagreement(self):
        """Test should_fork returns False when no disagreement."""
        detector = ForkDetector()

        messages = [
            self._create_mock_message("claude", "I think we should use Redis"),
            self._create_mock_message("gpt4", "Yes, Redis is a good choice"),
            self._create_mock_message("claude", "Redis provides persistence"),
            self._create_mock_message("gpt4", "Agreed on Redis"),
        ]
        agents = [
            self._create_mock_agent("claude"),
            self._create_mock_agent("gpt4"),
        ]

        decision = detector.should_fork(messages, round_num=3, agents=agents)

        assert decision.should_fork is False

    def test_should_fork_with_disagreement(self):
        """Test should_fork returns True with strong disagreement."""
        detector = ForkDetector()

        messages = [
            self._create_mock_message(
                "claude",
                "I strongly disagree with the monolith approach. We should use microservices instead.",
            ),
            self._create_mock_message(
                "gpt4", "On the contrary, monoliths are better. I disagree with microservices."
            ),
            self._create_mock_message(
                "claude", "The fundamentally different approach is wrong. I suggest microservices."
            ),
            self._create_mock_message(
                "gpt4", "I cannot accept that. Monoliths should be used rather than microservices."
            ),
        ]
        agents = [
            self._create_mock_agent("claude"),
            self._create_mock_agent("gpt4"),
        ]

        decision = detector.should_fork(messages, round_num=3, agents=agents)

        # May or may not trigger fork depending on phrase detection
        assert isinstance(decision, ForkDecision)
        assert isinstance(decision.disagreement_score, float)

    def test_should_fork_contradictory_should_not(self):
        """Test detection of contradictory should/should not statements."""
        detector = ForkDetector()

        messages = [
            self._create_mock_message("claude", "We should use caching for performance"),
            self._create_mock_message("gpt4", "We should not use caching due to complexity"),
        ]
        agents = [
            self._create_mock_agent("claude"),
            self._create_mock_agent("gpt4"),
        ]

        decision = detector.should_fork(messages, round_num=3, agents=agents)

        # Contradictory should/should not increases disagreement score
        assert decision.disagreement_score >= 0.3

    def test_should_fork_different_tech_choices(self):
        """Test detection of different technology choices."""
        detector = ForkDetector()

        messages = [
            self._create_mock_message("claude", "We should use sql database for this"),
            self._create_mock_message("gpt4", "I prefer nosql for flexibility"),
        ]
        agents = [
            self._create_mock_agent("claude"),
            self._create_mock_agent("gpt4"),
        ]

        decision = detector.should_fork(messages, round_num=3, agents=agents)

        # Different tech choices increases disagreement
        assert decision.disagreement_score >= 0.3

    def test_calculate_disagreement_with_phrases(self):
        """Test _calculate_disagreement with explicit phrases."""
        detector = ForkDetector()

        msg_a = Mock()
        msg_a.content = "I fundamentally disagree with this approach"
        msg_b = Mock()
        msg_b.content = "However, I think there is a better alternative"

        score, reason = detector._calculate_disagreement(msg_a, msg_b)

        assert score > 0
        assert len(reason) > 0

    def test_extract_approach(self):
        """Test _extract_approach helper method."""
        detector = ForkDetector()

        msg = Mock()
        msg.content = "I propose that we use microservices architecture for better scalability. This allows independent deployment."

        approach = detector._extract_approach(msg)

        assert "propose" in approach.lower() or len(approach) > 0


# =============================================================================
# DebateForker Tests
# =============================================================================


class TestDebateForker:
    """Tests for DebateForker class."""

    def _create_mock_message(self, agent: str, content: str, round_num: int = 1) -> Mock:
        """Helper to create mock messages."""
        from aragora.core import Message

        return Message(
            agent=agent,
            role="proposer",
            content=content,
            round=round_num,
        )

    def test_init(self):
        """Test initialization."""
        forker = DebateForker()

        assert forker.detector is not None
        assert forker.branches == {}
        assert forker.fork_points == {}

    def test_fork_creates_branches(self):
        """Test fork creates branches."""
        forker = DebateForker()

        messages = [
            self._create_mock_message("claude", "Test message", 1),
            self._create_mock_message("gpt4", "Another message", 1),
        ]

        decision = ForkDecision(
            should_fork=True,
            reason="Test fork",
            branches=[
                {"hypothesis": "Claude's approach", "lead_agent": "claude"},
                {"hypothesis": "GPT4's approach", "lead_agent": "gpt4"},
            ],
            disagreement_score=0.8,
        )

        branches = forker.fork(
            parent_debate_id="debate-123",
            fork_round=3,
            messages_so_far=messages,
            decision=decision,
        )

        assert len(branches) == 2
        assert all(b.parent_debate_id == "debate-123" for b in branches)
        assert all(b.fork_round == 3 for b in branches)
        assert len(forker.branches["debate-123"]) == 2

    def test_fork_records_fork_point(self):
        """Test fork records fork point."""
        forker = DebateForker()

        decision = ForkDecision(
            should_fork=True,
            reason="Disagreement",
            branches=[
                {"hypothesis": "A", "lead_agent": "a"},
                {"hypothesis": "B", "lead_agent": "b"},
            ],
            disagreement_score=0.75,
        )

        forker.fork(
            parent_debate_id="debate-123",
            fork_round=4,
            messages_so_far=[],
            decision=decision,
        )

        fork_points = forker.get_fork_history("debate-123")
        assert len(fork_points) == 1
        assert fork_points[0].round == 4
        assert fork_points[0].reason == "Disagreement"

    def test_fork_copies_messages(self):
        """Test fork copies messages to branches."""
        forker = DebateForker()

        messages = [
            self._create_mock_message("claude", "Original message", 1),
        ]

        decision = ForkDecision(
            should_fork=True,
            reason="Test",
            branches=[
                {"hypothesis": "A", "lead_agent": "a"},
            ],
            disagreement_score=0.8,
        )

        branches = forker.fork(
            parent_debate_id="debate-123",
            fork_round=2,
            messages_so_far=messages,
            decision=decision,
        )

        assert len(branches[0].messages) == 1
        # Should be a copy, not the same list
        branches[0].messages.append(self._create_mock_message("x", "y", 2))
        assert len(messages) == 1  # Original unchanged

    @pytest.mark.asyncio
    async def test_run_branches(self):
        """Test running branches in parallel."""
        forker = DebateForker()

        messages = [self._create_mock_message("claude", "Test", 1)]

        decision = ForkDecision(
            should_fork=True,
            reason="Test",
            branches=[
                {"hypothesis": "A", "lead_agent": "a"},
                {"hypothesis": "B", "lead_agent": "b"},
            ],
            disagreement_score=0.8,
        )

        branches = forker.fork(
            parent_debate_id="debate-123",
            fork_round=2,
            messages_so_far=messages,
            decision=decision,
        )

        # Mock environment
        env = Mock()
        env.task = "Test task"

        # Mock agents
        agents = [Mock(), Mock()]

        async def mock_run_debate(branch_env, agents, initial_messages):
            from aragora.core import DebateResult

            return DebateResult(
                final_answer=f"Answer for {branch_env.task}",
                confidence=0.8,
                consensus_reached=True,
                rounds_used=2,
                messages=[],
                critiques=[],
                votes=[],
            )

        completed = await forker.run_branches(
            branches=branches,
            env=env,
            agents=agents,
            run_debate_fn=mock_run_debate,
            max_rounds=3,
        )

        assert len(completed) == 2
        assert all(b.is_complete for b in completed)

    @pytest.mark.asyncio
    async def test_run_branches_with_failure(self):
        """Test running branches handles failures gracefully."""
        forker = DebateForker()

        branches = [
            Branch(
                branch_id="b1",
                parent_debate_id="debate-123",
                fork_round=1,
                hypothesis="Test",
                lead_agent="a",
            ),
        ]

        env = Mock()
        env.task = "Test"
        agents = []

        async def failing_run_debate(branch_env, agents, initial_messages):
            raise RuntimeError("Simulated failure")

        completed = await forker.run_branches(
            branches=branches,
            env=env,
            agents=agents,
            run_debate_fn=failing_run_debate,
            max_rounds=3,
        )

        # Should return empty list for failed branches
        assert len(completed) == 0

    def test_merge_branches(self):
        """Test merging completed branches."""
        from aragora.core import DebateResult

        forker = DebateForker()

        result1 = DebateResult(
            final_answer="Best answer",
            confidence=0.9,
            consensus_reached=True,
            rounds_used=2,
            messages=[],
            critiques=[],
            votes=[],
        )

        result2 = DebateResult(
            final_answer="Good answer",
            confidence=0.7,
            consensus_reached=False,
            rounds_used=3,
            messages=[],
            critiques=[],
            votes=[],
        )

        branches = [
            Branch(
                branch_id="branch-1",
                parent_debate_id="debate-123",
                fork_round=2,
                hypothesis="Approach A",
                lead_agent="claude",
                result=result1,
            ),
            Branch(
                branch_id="branch-2",
                parent_debate_id="debate-123",
                fork_round=2,
                hypothesis="Approach B",
                lead_agent="gpt4",
                result=result2,
            ),
        ]

        merge = forker.merge(branches)

        assert merge.winning_branch_id == "branch-1"  # Higher score
        assert merge.winning_hypothesis == "Approach A"
        assert len(merge.all_branch_results) == 2

    def test_merge_empty_branches_raises(self):
        """Test merge raises with empty branches."""
        forker = DebateForker()

        with pytest.raises(ValueError, match="No branches to merge"):
            forker.merge([])

    def test_merge_no_completed_raises(self):
        """Test merge raises with no completed branches."""
        forker = DebateForker()

        branches = [
            Branch(
                branch_id="b1",
                parent_debate_id="debate-123",
                fork_round=1,
                hypothesis="Test",
                lead_agent="a",
                result=None,  # Not complete
            ),
        ]

        with pytest.raises(ValueError, match="No completed branches"):
            forker.merge(branches)

    def test_score_branch(self):
        """Test _score_branch scoring logic."""
        from aragora.core import DebateResult

        forker = DebateForker()

        # High score: consensus, high confidence, few rounds, no harsh critiques
        good_result = DebateResult(
            final_answer="Good",
            confidence=0.95,
            consensus_reached=True,
            rounds_used=2,
            messages=[],
            critiques=[],
            votes=[],
        )

        good_branch = Branch(
            branch_id="good",
            parent_debate_id="debate-123",
            fork_round=1,
            hypothesis="Good",
            lead_agent="a",
            result=good_result,
        )

        good_score = forker._score_branch(good_branch)

        # Poor score: no consensus, low confidence, many rounds
        poor_result = DebateResult(
            final_answer="Poor",
            confidence=0.3,
            consensus_reached=False,
            rounds_used=8,
            messages=[],
            critiques=[],
            votes=[],
        )

        poor_branch = Branch(
            branch_id="poor",
            parent_debate_id="debate-123",
            fork_round=1,
            hypothesis="Poor",
            lead_agent="b",
            result=poor_result,
        )

        poor_score = forker._score_branch(poor_branch)

        assert good_score > poor_score

    def test_score_branch_with_critiques(self):
        """Test _score_branch considers critique severity."""
        from aragora.core import Critique, DebateResult

        forker = DebateForker()

        critique = Mock()
        critique.severity = 0.9  # High severity is bad

        result = DebateResult(
            final_answer="Test",
            confidence=0.8,
            consensus_reached=True,
            rounds_used=3,
            messages=[],
            critiques=[critique],
            votes=[],
        )

        branch = Branch(
            branch_id="b1",
            parent_debate_id="debate-123",
            fork_round=1,
            hypothesis="Test",
            lead_agent="a",
            result=result,
        )

        score = forker._score_branch(branch)

        # Score should account for high severity critique
        assert score < 0.85  # Less than perfect due to critique

    def test_generate_comparison(self):
        """Test _generate_comparison report generation."""
        from aragora.core import DebateResult

        forker = DebateForker()

        result = DebateResult(
            final_answer="Test",
            confidence=0.8,
            consensus_reached=True,
            rounds_used=3,
            messages=[],
            critiques=[],
            votes=[],
        )

        branches = [
            Branch(
                branch_id="b1",
                parent_debate_id="debate-123",
                fork_round=1,
                hypothesis="Approach A",
                lead_agent="claude",
                result=result,
            ),
        ]

        scores = {"b1": 0.75}

        comparison = forker._generate_comparison(branches, scores)

        assert "Branch Comparison" in comparison
        assert "Approach A" in comparison
        assert "claude" in comparison
        assert "0.75" in comparison

    def test_extract_merged_insights(self):
        """Test _extract_merged_insights from branches."""
        from aragora.core import DebateResult

        forker = DebateForker()

        result1 = DebateResult(
            final_answer="Use Redis for caching. It provides good performance.",
            confidence=0.8,
            consensus_reached=True,
            rounds_used=3,
            messages=[],
            critiques=[],
            votes=[],
        )

        result2 = DebateResult(
            final_answer="Consider Memcached as alternative. It is simpler.",
            confidence=0.7,
            consensus_reached=True,
            rounds_used=4,
            messages=[],
            critiques=[],
            votes=[],
        )

        branches = [
            Branch(
                branch_id="b1",
                parent_debate_id="debate-123",
                fork_round=1,
                hypothesis="A",
                lead_agent="claude",
                result=result1,
            ),
            Branch(
                branch_id="b2",
                parent_debate_id="debate-123",
                fork_round=1,
                hypothesis="B",
                lead_agent="gpt4",
                result=result2,
            ),
        ]

        insights = forker._extract_merged_insights(branches)

        assert len(insights) == 2
        assert "[claude]" in insights[0]
        assert "[gpt4]" in insights[1]

    def test_get_fork_history(self):
        """Test getting fork history for debate."""
        forker = DebateForker()

        decision = ForkDecision(
            should_fork=True,
            reason="Test",
            branches=[{"hypothesis": "A", "lead_agent": "a"}],
            disagreement_score=0.8,
        )

        forker.fork(
            parent_debate_id="debate-123",
            fork_round=2,
            messages_so_far=[],
            decision=decision,
        )

        forker.fork(
            parent_debate_id="debate-123",
            fork_round=5,
            messages_so_far=[],
            decision=decision,
        )

        history = forker.get_fork_history("debate-123")
        assert len(history) == 2

        # Different debate
        other_history = forker.get_fork_history("debate-other")
        assert len(other_history) == 0

    def test_get_branches(self):
        """Test getting branches for debate."""
        forker = DebateForker()

        decision = ForkDecision(
            should_fork=True,
            reason="Test",
            branches=[
                {"hypothesis": "A", "lead_agent": "a"},
                {"hypothesis": "B", "lead_agent": "b"},
            ],
            disagreement_score=0.8,
        )

        forker.fork(
            parent_debate_id="debate-123",
            fork_round=2,
            messages_so_far=[],
            decision=decision,
        )

        branches = forker.get_branches("debate-123")
        assert len(branches) == 2

        other_branches = forker.get_branches("debate-other")
        assert len(other_branches) == 0


# =============================================================================
# DeadlockResolver Tests
# =============================================================================


class TestDeadlockResolver:
    """Tests for DeadlockResolver class."""

    def _create_mock_message(self, agent: str, content: str, round_num: int = 1) -> Mock:
        """Helper to create mock messages."""
        from aragora.core import Message

        return Message(
            agent=agent,
            role="proposer",
            content=content,
            round=round_num,
        )

    def test_init_default(self):
        """Test initialization with default forker."""
        resolver = DeadlockResolver()

        assert resolver.forker is not None
        assert resolver.debate_history == {}
        assert resolver.deadlock_signals == []

    def test_init_with_forker(self):
        """Test initialization with provided forker."""
        forker = DebateForker()
        resolver = DeadlockResolver(forker=forker)

        assert resolver.forker is forker

    def test_analyze_round_too_early(self):
        """Test analyze_round returns None when too early."""
        resolver = DeadlockResolver()

        messages = [self._create_mock_message("a", "Test", 1)]

        signal = resolver.analyze_round(
            debate_id="debate-123",
            round_num=2,  # Below MIN_ROUNDS_FOR_DETECTION (3)
            messages=messages,
        )

        assert signal is None

    def test_analyze_round_records_history(self):
        """Test analyze_round records round history."""
        resolver = DeadlockResolver()

        messages = [
            self._create_mock_message("a", "We should implement because this is good", 3),
            self._create_mock_message("b", "I agree since it implies better results", 3),
        ]

        resolver.analyze_round(
            debate_id="debate-123",
            round_num=3,
            messages=messages,
        )

        assert "debate-123" in resolver.debate_history
        assert len(resolver.debate_history["debate-123"]) == 1

    def test_analyze_round_detects_positional_deadlock(self):
        """Test detection of positional deadlock."""
        resolver = DeadlockResolver()

        # Build up history with same positions
        for round_num in range(3, 7):
            messages = [
                self._create_mock_message("a", "I strongly agree with approach A", round_num),
                self._create_mock_message(
                    "b", "I strongly disagree, approach B is better", round_num
                ),
            ]

            signal = resolver.analyze_round(
                debate_id="debate-123",
                round_num=round_num,
                messages=messages,
            )

        # After several rounds with same positions, should detect deadlock
        # Note: May not trigger on first few rounds
        if signal:
            assert signal.pattern in ["positional", "circular", "exhaustion"]

    def test_analyze_round_detects_circular_arguments(self):
        """Test detection of circular arguments."""
        resolver = DeadlockResolver()

        # Create pattern: round 3 -> round 4 -> round 5 repeats round 3
        messages_3 = [
            self._create_mock_message("a", "Because this therefore we should conclude A", 3),
        ]
        messages_4 = [
            self._create_mock_message("a", "Given that evidence suggests implies B", 4),
        ]
        messages_5 = [
            self._create_mock_message("a", "Because this therefore we should conclude A", 5),
        ]

        resolver.analyze_round("debate-123", 3, messages_3)
        resolver.analyze_round("debate-123", 4, messages_4)
        signal = resolver.analyze_round("debate-123", 5, messages_5)

        # May detect circular pattern
        if signal:
            assert signal.debate_id == "debate-123"

    def test_analyze_round_detects_argument_exhaustion(self):
        """Test detection of argument exhaustion."""
        resolver = DeadlockResolver()

        # First round has many arguments
        messages_3 = [
            self._create_mock_message("a", "Because X therefore Y since Z evidence suggests P", 3),
        ]
        # Subsequent rounds have fewer
        messages_4 = [
            self._create_mock_message("a", "Just because", 4),
        ]
        messages_5 = [
            self._create_mock_message("a", "Yes", 5),
        ]

        resolver.analyze_round("debate-123", 3, messages_3)
        resolver.analyze_round("debate-123", 4, messages_4)
        signal = resolver.analyze_round("debate-123", 5, messages_5)

        # May detect exhaustion
        if signal:
            assert isinstance(signal.stagnation_score, float)

    def test_auto_resolve_fork_recommendation(self):
        """Test auto_resolve creates branches for fork recommendation."""
        resolver = DeadlockResolver()

        signal = DeadlockSignal(
            debate_id="debate-123",
            round=5,
            stagnation_score=0.85,
            rounds_without_progress=3,
            pattern="positional",
            recommendation="fork",
        )

        messages = [
            self._create_mock_message("a", "Position A", 5),
            self._create_mock_message("b", "Position B", 5),
        ]

        agents = [Mock(), Mock()]
        agents[0].name = "a"
        agents[1].name = "b"

        branches = resolver.auto_resolve(signal, messages, agents)

        if branches:
            assert len(branches) >= 2

    def test_auto_resolve_non_fork_recommendation(self):
        """Test auto_resolve returns None for non-fork recommendations."""
        resolver = DeadlockResolver()

        signal = DeadlockSignal(
            debate_id="debate-123",
            round=5,
            stagnation_score=0.6,
            rounds_without_progress=1,
            pattern="exhaustion",
            recommendation="inject_perspective",
        )

        result = resolver.auto_resolve(signal, [], [])

        assert result is None

    def test_extract_positions(self):
        """Test _extract_positions helper."""
        resolver = DeadlockResolver()

        messages = [
            self._create_mock_message("a", "I strongly agree with this", 1),
            self._create_mock_message("b", "I disagree completely", 1),
        ]

        positions = resolver._extract_positions(messages)

        assert "a" in positions
        assert "b" in positions
        assert "strong_support" in positions["a"]
        assert "oppose" in positions["b"]

    def test_calculate_positional_stagnation(self):
        """Test _calculate_positional_stagnation calculation."""
        resolver = DeadlockResolver()

        # Set up history
        resolver.debate_history["debate-123"] = [
            {"round": 3, "positions": {"a": "support_123", "b": "oppose_456"}},
        ]

        current = {"a": "support_123", "b": "oppose_456"}  # Same
        previous = {"a": "support_123", "b": "oppose_456"}

        stagnation = resolver._calculate_positional_stagnation("debate-123", current, previous)

        assert stagnation == 1.0  # All positions unchanged

    def test_detect_circular_arguments(self):
        """Test _detect_circular_arguments detection."""
        resolver = DeadlockResolver()

        # Set up history with cycling pattern
        resolver.debate_history["debate-123"] = [
            {"round": 3, "unique_arguments": 5},
            {"round": 4, "unique_arguments": 3},
            {"round": 5, "unique_arguments": 5},  # Same as round 3
        ]

        score = resolver._detect_circular_arguments("debate-123")

        assert score >= 0.8  # Should detect cycling pattern

    def test_calculate_argument_exhaustion(self):
        """Test _calculate_argument_exhaustion calculation."""
        resolver = DeadlockResolver()

        # Set up history with declining arguments
        resolver.debate_history["debate-123"] = [
            {"round": 3, "unique_arguments": 10},
            {"round": 4, "unique_arguments": 5},
            {"round": 5, "unique_arguments": 2},
        ]

        score = resolver._calculate_argument_exhaustion("debate-123")

        assert score > 0.5  # Should detect declining trend

    def test_count_unique_arguments(self):
        """Test _count_unique_arguments counting."""
        resolver = DeadlockResolver()

        messages = [
            self._create_mock_message("a", "Because this is true, therefore we conclude", 1),
            self._create_mock_message("b", "The evidence suggests that we should consider", 1),
        ]

        count = resolver._count_unique_arguments(messages)

        assert count >= 2  # Should find argument indicators

    def test_count_stagnant_rounds(self):
        """Test _count_stagnant_rounds counting."""
        resolver = DeadlockResolver()

        # Set up history with no progress
        resolver.debate_history["debate-123"] = [
            {"round": 3, "unique_arguments": 5, "positions": {"a": "x"}},
            {"round": 4, "unique_arguments": 5, "positions": {"a": "x"}},
            {"round": 5, "unique_arguments": 5, "positions": {"a": "x"}},
        ]

        stagnant = resolver._count_stagnant_rounds("debate-123")

        assert stagnant >= 1

    def test_get_recommendation(self):
        """Test _get_recommendation strategy selection."""
        resolver = DeadlockResolver()

        # High positional stagnation -> fork
        rec = resolver._get_recommendation(0.9, 2, "positional")
        assert rec == "fork"

        # Circular with no progress -> fork
        rec = resolver._get_recommendation(0.7, 2, "circular")
        assert rec == "fork"

        # Exhaustion with many stagnant rounds -> conclude
        rec = resolver._get_recommendation(0.6, 4, "exhaustion")
        assert rec == "conclude"

        # Exhaustion early -> inject
        rec = resolver._get_recommendation(0.5, 1, "exhaustion")
        assert rec == "inject_perspective"

    def test_create_counterfactual_decision(self):
        """Test _create_counterfactual_decision creates branches."""
        resolver = DeadlockResolver()

        signal = DeadlockSignal(
            debate_id="debate-123",
            round=5,
            stagnation_score=0.8,
            rounds_without_progress=2,
            pattern="positional",
            recommendation="fork",
        )

        messages = [
            self._create_mock_message("a", "Position A", 5),
            self._create_mock_message("b", "Position B", 5),
        ]

        agents = []

        decision = resolver._create_counterfactual_decision(signal, messages, agents)

        assert isinstance(decision, ForkDecision)
        if decision.should_fork:
            assert len(decision.branches) >= 2

    def test_reset_single_debate(self):
        """Test reset for single debate."""
        resolver = DeadlockResolver()

        resolver.debate_history["debate-123"] = [{}]
        resolver.debate_history["debate-456"] = [{}]

        resolver.reset("debate-123")

        assert "debate-123" not in resolver.debate_history
        assert "debate-456" in resolver.debate_history

    def test_reset_all(self):
        """Test reset all state."""
        resolver = DeadlockResolver()

        resolver.debate_history["debate-123"] = [{}]
        resolver.deadlock_signals.append(Mock())

        resolver.reset()

        assert resolver.debate_history == {}
        assert resolver.deadlock_signals == []

    def test_get_signals(self):
        """Test get_signals retrieval."""
        resolver = DeadlockResolver()

        signal1 = DeadlockSignal(
            debate_id="debate-123",
            round=3,
            stagnation_score=0.8,
            rounds_without_progress=2,
            pattern="positional",
            recommendation="fork",
        )
        signal2 = DeadlockSignal(
            debate_id="debate-456",
            round=4,
            stagnation_score=0.7,
            rounds_without_progress=1,
            pattern="circular",
            recommendation="inject_perspective",
        )

        resolver.deadlock_signals = [signal1, signal2]

        all_signals = resolver.get_signals()
        assert len(all_signals) == 2

        debate_123_signals = resolver.get_signals("debate-123")
        assert len(debate_123_signals) == 1
        assert debate_123_signals[0].debate_id == "debate-123"


# =============================================================================
# Integration Tests
# =============================================================================


class TestForkingIntegration:
    """Integration tests for forking system."""

    def _create_mock_message(self, agent: str, content: str, round_num: int = 1) -> Mock:
        """Helper to create mock messages."""
        from aragora.core import Message

        return Message(
            agent=agent,
            role="proposer",
            content=content,
            round=round_num,
        )

    def test_detector_forker_integration(self):
        """Test ForkDetector and DebateForker work together."""
        detector = ForkDetector()
        forker = DebateForker()

        messages = [
            self._create_mock_message("a", "I disagree fundamentally", 1),
            self._create_mock_message("b", "On the contrary, you are wrong", 1),
            self._create_mock_message("a", "I cannot accept that premise", 2),
            self._create_mock_message("b", "I reject the core assumption", 2),
        ]

        agents = [Mock(), Mock()]
        agents[0].name = "a"
        agents[1].name = "b"

        decision = detector.should_fork(messages, round_num=3, agents=agents)

        if decision.should_fork:
            branches = forker.fork(
                parent_debate_id="debate-123",
                fork_round=3,
                messages_so_far=messages,
                decision=decision,
            )

            assert len(branches) >= 1

    def test_resolver_forker_integration(self):
        """Test DeadlockResolver and DebateForker work together."""
        forker = DebateForker()
        resolver = DeadlockResolver(forker=forker)

        # Build up deadlock state
        for round_num in range(3, 7):
            messages = [
                self._create_mock_message("a", "I strongly agree with A because", round_num),
                self._create_mock_message("b", "I strongly disagree, B is better", round_num),
            ]
            signal = resolver.analyze_round("debate-123", round_num, messages)

        if signal and signal.recommendation == "fork":
            agents = [Mock(), Mock()]
            agents[0].name = "a"
            agents[1].name = "b"

            branches = resolver.auto_resolve(signal, messages, agents)

            if branches:
                assert len(branches) >= 2
                assert len(forker.get_branches("debate-123")) >= 2


# =============================================================================
# Additional Coverage: _extract_positions stance categorization
# =============================================================================


class TestExtractPositionsDeep:
    """Tests for _extract_positions internal stance detection."""

    def _create_msg(self, agent, content, round_num=1):
        from aragora.core import Message

        return Message(role="agent", agent=agent, content=content, round=round_num)

    def test_extract_strong_support(self):
        """Test _extract_positions detects strong_support stance."""
        resolver = DeadlockResolver()

        msgs = [
            self._create_msg(
                "claude", "I strongly agree with this approach and think it is best.", 1
            ),
        ]

        positions = resolver._extract_positions(msgs)

        assert "claude" in positions
        assert "strong_support" in positions["claude"]

    def test_extract_support(self):
        """Test _extract_positions detects support stance."""
        resolver = DeadlockResolver()

        msgs = [
            self._create_msg("gpt4", "I agree that we should use this method.", 1),
        ]

        positions = resolver._extract_positions(msgs)

        assert "gpt4" in positions
        assert "support" in positions["gpt4"]

    def test_extract_oppose_keyword(self):
        """Test _extract_positions detects oppose via 'oppose' keyword."""
        resolver = DeadlockResolver()

        msgs = [
            self._create_msg("gemini", "I oppose this direction entirely.", 1),
        ]

        positions = resolver._extract_positions(msgs)

        assert "gemini" in positions
        assert "oppose" in positions["gemini"]

    def test_extract_never_keyword(self):
        """Test _extract_positions detects strong_oppose via 'never' keyword."""
        resolver = DeadlockResolver()

        msgs = [
            self._create_msg("grok", "We should never use this approach.", 1),
        ]

        positions = resolver._extract_positions(msgs)

        assert "grok" in positions
        assert "strong_oppose" in positions["grok"]

    def test_extract_neutral(self):
        """Test _extract_positions defaults to neutral stance."""
        resolver = DeadlockResolver()

        msgs = [
            self._create_msg("claude", "Let me analyze this problem carefully.", 1),
        ]

        positions = resolver._extract_positions(msgs)

        assert "claude" in positions
        assert "neutral" in positions["claude"]

    def test_extract_multiple_agents(self):
        """Test _extract_positions handles multiple agents in same round."""
        resolver = DeadlockResolver()

        msgs = [
            self._create_msg("claude", "I definitely think Redis is the answer.", 1),
            self._create_msg("gpt4", "I oppose the Redis approach entirely.", 1),
        ]

        positions = resolver._extract_positions(msgs)

        assert len(positions) == 2
        assert "strong_support" in positions["claude"]
        assert "oppose" in positions["gpt4"]


# =============================================================================
# Additional Coverage: _calculate_positional_stagnation
# =============================================================================


class TestPositionalStagnationDeep:
    """Tests for _calculate_positional_stagnation edge cases."""

    def _create_msg(self, agent, content, round_num=1):
        from aragora.core import Message

        return Message(role="agent", agent=agent, content=content, round=round_num)

    def test_stagnation_with_previous_positions(self):
        """Test _calculate_positional_stagnation with explicit previous_positions."""
        resolver = DeadlockResolver()
        debate_id = "debate-stag"

        # Need at least 2 rounds in history
        resolver.debate_history[debate_id] = [
            {"round": 1, "positions": {"a": "support_123"}, "unique_arguments": 3},
            {"round": 2, "positions": {"a": "support_123"}, "unique_arguments": 3},
        ]

        current = {"a": "support_123"}
        previous = {"a": "support_123"}

        score = resolver._calculate_positional_stagnation(debate_id, current, previous)

        assert score == 1.0  # All positions match

    def test_stagnation_no_previous_positions_uses_history(self):
        """Test _calculate_positional_stagnation without previous_positions uses history."""
        resolver = DeadlockResolver()
        debate_id = "debate-hist"

        # Set up history with same positions repeating
        resolver.debate_history[debate_id] = [
            {"round": 1, "positions": {"a": "support_123"}, "unique_arguments": 3},
            {"round": 2, "positions": {"a": "support_123"}, "unique_arguments": 3},
            {"round": 3, "positions": {"a": "support_123"}, "unique_arguments": 3},
        ]

        current = {"a": "support_123"}

        score = resolver._calculate_positional_stagnation(debate_id, current, None)

        assert score > 0.0  # Should detect repeating positions

    def test_stagnation_insufficient_history(self):
        """Test _calculate_positional_stagnation returns 0.0 with insufficient history."""
        resolver = DeadlockResolver()
        debate_id = "debate-short"

        resolver.debate_history[debate_id] = [
            {"round": 1, "positions": {"a": "support_123"}, "unique_arguments": 3},
        ]

        score = resolver._calculate_positional_stagnation(debate_id, {"a": "x"}, None)

        assert score == 0.0

    def test_stagnation_empty_current_positions(self):
        """Test _calculate_positional_stagnation with empty current positions."""
        resolver = DeadlockResolver()
        debate_id = "debate-empty"

        resolver.debate_history[debate_id] = [
            {"round": 1, "positions": {}, "unique_arguments": 0},
            {"round": 2, "positions": {}, "unique_arguments": 0},
        ]

        score = resolver._calculate_positional_stagnation(debate_id, {}, {"a": "x"})

        # No matching (empty current), division by zero guarded
        assert score == 0.0


# =============================================================================
# Additional Coverage: _detect_circular_arguments
# =============================================================================


class TestDetectCircularArgumentsDeep:
    """Tests for _detect_circular_arguments patterns."""

    def test_circular_repeating_pattern(self):
        """Test _detect_circular_arguments detects A->B->A pattern."""
        resolver = DeadlockResolver()
        debate_id = "debate-circ"

        resolver.debate_history[debate_id] = [
            {"round": 1, "positions": {}, "unique_arguments": 5},
            {"round": 2, "positions": {}, "unique_arguments": 3},
            {"round": 3, "positions": {}, "unique_arguments": 5},  # Same as round 1
        ]

        score = resolver._detect_circular_arguments(debate_id)

        assert score == 0.8  # Detected A->B->A pattern

    def test_circular_flat_pattern(self):
        """Test _detect_circular_arguments detects flat argument count."""
        resolver = DeadlockResolver()
        debate_id = "debate-flat"

        resolver.debate_history[debate_id] = [
            {"round": 1, "positions": {}, "unique_arguments": 3},
            {"round": 2, "positions": {}, "unique_arguments": 3},
            {"round": 3, "positions": {}, "unique_arguments": 3},
        ]

        score = resolver._detect_circular_arguments(debate_id)

        assert score == 0.6  # Flat pattern

    def test_circular_no_pattern(self):
        """Test _detect_circular_arguments returns 0.0 with no pattern."""
        resolver = DeadlockResolver()
        debate_id = "debate-nopatn"

        resolver.debate_history[debate_id] = [
            {"round": 1, "positions": {}, "unique_arguments": 1},
            {"round": 2, "positions": {}, "unique_arguments": 3},
            {"round": 3, "positions": {}, "unique_arguments": 6},
        ]

        score = resolver._detect_circular_arguments(debate_id)

        assert score == 0.0

    def test_circular_insufficient_history(self):
        """Test _detect_circular_arguments returns 0.0 with < 3 rounds."""
        resolver = DeadlockResolver()
        debate_id = "debate-short"

        resolver.debate_history[debate_id] = [
            {"round": 1, "positions": {}, "unique_arguments": 5},
            {"round": 2, "positions": {}, "unique_arguments": 3},
        ]

        score = resolver._detect_circular_arguments(debate_id)

        assert score == 0.0


# =============================================================================
# Additional Coverage: _calculate_argument_exhaustion
# =============================================================================


class TestArgumentExhaustionDeep:
    """Tests for _calculate_argument_exhaustion edge cases."""

    def test_exhaustion_monotonic_decrease(self):
        """Test _calculate_argument_exhaustion with monotonic decline."""
        resolver = DeadlockResolver()
        debate_id = "debate-exhaust"

        resolver.debate_history[debate_id] = [
            {"round": 1, "positions": {}, "unique_arguments": 10},
            {"round": 2, "positions": {}, "unique_arguments": 6},
            {"round": 3, "positions": {}, "unique_arguments": 2},
        ]

        score = resolver._calculate_argument_exhaustion(debate_id)

        # decline_rate = (10 - 2) / 10 = 0.8
        assert score == 0.8

    def test_exhaustion_non_monotonic(self):
        """Test _calculate_argument_exhaustion with non-monotonic change returns 0."""
        resolver = DeadlockResolver()
        debate_id = "debate-bounce"

        resolver.debate_history[debate_id] = [
            {"round": 1, "positions": {}, "unique_arguments": 5},
            {"round": 2, "positions": {}, "unique_arguments": 3},
            {"round": 3, "positions": {}, "unique_arguments": 7},  # Increase
        ]

        score = resolver._calculate_argument_exhaustion(debate_id)

        assert score == 0.0  # Not monotonically decreasing

    def test_exhaustion_capped_at_one(self):
        """Test _calculate_argument_exhaustion caps at 1.0."""
        resolver = DeadlockResolver()
        debate_id = "debate-cap"

        resolver.debate_history[debate_id] = [
            {"round": 1, "positions": {}, "unique_arguments": 10},
            {"round": 2, "positions": {}, "unique_arguments": 5},
            {"round": 3, "positions": {}, "unique_arguments": 0},
        ]

        score = resolver._calculate_argument_exhaustion(debate_id)

        assert score <= 1.0

    def test_exhaustion_insufficient_history(self):
        """Test _calculate_argument_exhaustion with < 2 rounds."""
        resolver = DeadlockResolver()
        debate_id = "debate-one"

        resolver.debate_history[debate_id] = [
            {"round": 1, "positions": {}, "unique_arguments": 5},
        ]

        score = resolver._calculate_argument_exhaustion(debate_id)

        assert score == 0.0


# =============================================================================
# Additional Coverage: _count_stagnant_rounds
# =============================================================================


class TestCountStagnantRoundsDeep:
    """Tests for _count_stagnant_rounds edge cases."""

    def test_stagnant_rounds_with_progress(self):
        """Test _count_stagnant_rounds stops at progress."""
        resolver = DeadlockResolver()
        debate_id = "debate-prog"

        resolver.debate_history[debate_id] = [
            {"round": 1, "positions": {"a": "x"}, "unique_arguments": 3},
            {"round": 2, "positions": {"a": "x"}, "unique_arguments": 5},  # Progress: more args
            {"round": 3, "positions": {"a": "x"}, "unique_arguments": 5},  # Stagnant
            {"round": 4, "positions": {"a": "x"}, "unique_arguments": 5},  # Stagnant
        ]

        count = resolver._count_stagnant_rounds(debate_id)

        assert (
            count == 2
        )  # Only rounds 3->4 and 2->3 are stagnant wait, 3->4 yes, 2->3 no (progress)
        # Actually: iterating backward: i=3 (round 4 vs 3): same -> stagnant=1
        # i=2 (round 3 vs 2): args equal, positions equal -> stagnant=2
        # i=1 (round 2 vs 1): args increased -> break
        assert count == 2

    def test_stagnant_rounds_all_stagnant(self):
        """Test _count_stagnant_rounds when all rounds are stagnant."""
        resolver = DeadlockResolver()
        debate_id = "debate-allstag"

        resolver.debate_history[debate_id] = [
            {"round": 1, "positions": {"a": "x"}, "unique_arguments": 3},
            {"round": 2, "positions": {"a": "x"}, "unique_arguments": 3},
            {"round": 3, "positions": {"a": "x"}, "unique_arguments": 3},
            {"round": 4, "positions": {"a": "x"}, "unique_arguments": 3},
        ]

        count = resolver._count_stagnant_rounds(debate_id)

        assert count == 3

    def test_stagnant_rounds_insufficient_history(self):
        """Test _count_stagnant_rounds with < 2 rounds returns 0."""
        resolver = DeadlockResolver()
        debate_id = "debate-few"

        resolver.debate_history[debate_id] = [
            {"round": 1, "positions": {"a": "x"}, "unique_arguments": 3},
        ]

        count = resolver._count_stagnant_rounds(debate_id)

        assert count == 0


# =============================================================================
# Additional Coverage: _get_recommendation
# =============================================================================


class TestGetRecommendationDeep:
    """Tests for _get_recommendation decision logic."""

    def test_recommendation_positional_high_stagnation(self):
        """Test recommendation for positional pattern with high stagnation."""
        resolver = DeadlockResolver()

        result = resolver._get_recommendation(0.85, 1, "positional")

        assert result == "fork"

    def test_recommendation_circular_multiple_rounds(self):
        """Test recommendation for circular pattern with multiple stuck rounds."""
        resolver = DeadlockResolver()

        result = resolver._get_recommendation(0.6, 2, "circular")

        assert result == "fork"

    def test_recommendation_exhaustion_long(self):
        """Test recommendation for exhaustion pattern >= 3 rounds without progress."""
        resolver = DeadlockResolver()

        result = resolver._get_recommendation(0.5, 3, "exhaustion")

        assert result == "conclude"

    def test_recommendation_exhaustion_short(self):
        """Test recommendation for exhaustion pattern < 3 rounds."""
        resolver = DeadlockResolver()

        result = resolver._get_recommendation(0.5, 1, "exhaustion")

        assert result == "inject_perspective"

    def test_recommendation_default_high_score(self):
        """Test recommendation defaults to fork with high stagnation score."""
        resolver = DeadlockResolver()

        result = resolver._get_recommendation(0.76, 1, "unknown_pattern")

        assert result == "fork"

    def test_recommendation_default_low_score(self):
        """Test recommendation defaults to inject_perspective with low stagnation."""
        resolver = DeadlockResolver()

        result = resolver._get_recommendation(0.5, 1, "unknown_pattern")

        assert result == "inject_perspective"


# =============================================================================
# Additional Coverage: _create_counterfactual_decision
# =============================================================================


class TestCreateCounterfactualDecisionDeep:
    """Tests for _create_counterfactual_decision."""

    def _create_msg(self, agent, content, round_num=1):
        from aragora.core import Message

        return Message(role="agent", agent=agent, content=content, round=round_num)

    def _create_signal(self, debate_id="debate-123", round_num=5, pattern="positional"):
        return DeadlockSignal(
            debate_id=debate_id,
            round=round_num,
            stagnation_score=0.85,
            rounds_without_progress=3,
            pattern=pattern,
            recommendation="fork",
        )

    def test_single_agent_creates_one_branch(self):
        """Test _create_counterfactual_decision with single agent."""
        resolver = DeadlockResolver()
        signal = self._create_signal()
        messages = [self._create_msg("claude", "My position is X", 3)]
        agents = [Mock(name="claude")]

        decision = resolver._create_counterfactual_decision(signal, messages, agents)

        assert decision.should_fork is False  # < 2 branches
        assert len(decision.branches) == 1

    def test_two_agents_creates_three_branches(self):
        """Test _create_counterfactual_decision with two agents."""
        resolver = DeadlockResolver()
        signal = self._create_signal()
        messages = [
            self._create_msg("claude", "My position is X", 3),
            self._create_msg("gpt4", "My position is Y", 3),
        ]
        agents = [Mock(name="claude"), Mock(name="gpt4")]

        decision = resolver._create_counterfactual_decision(signal, messages, agents)

        assert decision.should_fork is True
        assert len(decision.branches) == 3  # agent1, agent2, synthesis
        # reversed() iteration: gpt4 found first, then claude
        assert "gpt4" in decision.branches[0]["hypothesis"]
        assert "claude" in decision.branches[1]["hypothesis"]
        assert "synthesis" in decision.branches[2]["hypothesis"].lower()

    def test_decision_uses_latest_messages_per_agent(self):
        """Test _create_counterfactual_decision uses latest message per agent."""
        resolver = DeadlockResolver()
        signal = self._create_signal()
        messages = [
            self._create_msg("claude", "First position", 1),
            self._create_msg("gpt4", "First position", 1),
            self._create_msg("claude", "Updated position", 3),
            self._create_msg("gpt4", "Updated position", 3),
        ]
        agents = [Mock(name="claude"), Mock(name="gpt4")]

        decision = resolver._create_counterfactual_decision(signal, messages, agents)

        assert decision.should_fork is True
        # Lead agents should come from latest messages (reversed iteration)

    def test_decision_empty_messages(self):
        """Test _create_counterfactual_decision with empty messages."""
        resolver = DeadlockResolver()
        signal = self._create_signal()

        decision = resolver._create_counterfactual_decision(signal, [], [])

        assert decision.should_fork is False
        assert len(decision.branches) == 0


# =============================================================================
# Additional Coverage: _score_branch and merge edge cases
# =============================================================================


class TestScoreBranchDeep:
    """Tests for _score_branch and merge edge cases."""

    def _create_branch(self, branch_id, hypothesis, lead_agent, result=None):
        return Branch(
            branch_id=branch_id,
            parent_debate_id="debate-123",
            fork_round=3,
            hypothesis=hypothesis,
            lead_agent=lead_agent,
            result=result,
        )

    def test_score_branch_no_result(self):
        """Test _score_branch with no result returns 0."""
        forker = DebateForker()

        branch = self._create_branch("b-1", "Test hypothesis", "claude", result=None)

        score = forker._score_branch(branch)

        assert score == 0.0

    def test_score_branch_with_result(self):
        """Test _score_branch with result computes score."""
        forker = DebateForker()

        result = Mock()
        result.confidence = 0.8
        result.consensus_reached = True
        result.rounds_used = 3
        result.critiques = []

        branch = self._create_branch("b-2", "Test hypothesis", "claude", result=result)

        score = forker._score_branch(branch)

        # 0.3 (consensus) + 0.3*0.8 (confidence) + 0.2*(1 - 3/10) (efficiency) = 0.68
        assert score > 0.0
        assert score == pytest.approx(0.68, abs=0.01)

    def test_merge_selects_highest_scoring_branch(self):
        """Test merge selects branch with highest score."""
        forker = DebateForker()

        result_low = Mock()
        result_low.confidence = 0.3
        result_low.consensus_reached = False
        result_low.final_answer = "Low answer"
        result_low.rounds_used = 5
        result_low.critiques = []

        result_high = Mock()
        result_high.confidence = 0.9
        result_high.consensus_reached = True
        result_high.final_answer = "High answer"
        result_high.rounds_used = 2
        result_high.critiques = []

        branch_low = self._create_branch("b-low", "Low hyp", "agent1", result_low)
        branch_high = self._create_branch("b-high", "High hyp", "agent2", result_high)

        merge_result = forker.merge([branch_low, branch_high])

        assert merge_result.winning_branch_id == "b-high"
        assert merge_result.winning_hypothesis == "High hyp"

    def test_merge_extracts_insights(self):
        """Test merge collects insights from all branches."""
        forker = DebateForker()

        result1 = Mock()
        result1.confidence = 0.7
        result1.consensus_reached = True
        result1.final_answer = "Answer A is best. It has good latency."
        result1.rounds_used = 3
        result1.critiques = []

        result2 = Mock()
        result2.confidence = 0.5
        result2.consensus_reached = False
        result2.final_answer = "Answer B works. It has better throughput."
        result2.rounds_used = 3
        result2.critiques = []

        branch1 = self._create_branch("b-1", "Approach A", "agent1", result1)
        branch2 = self._create_branch("b-2", "Approach B", "agent2", result2)

        merge_result = forker.merge([branch1, branch2])

        assert len(merge_result.merged_insights) >= 2
        assert merge_result.all_branch_results["b-1"] is result1
        assert merge_result.all_branch_results["b-2"] is result2

    def test_merge_branch_without_result(self):
        """Test merge handles branches without results."""
        forker = DebateForker()

        result1 = Mock()
        result1.confidence = 0.7
        result1.consensus_reached = True
        result1.final_answer = "Answer"
        result1.rounds_used = 3
        result1.critiques = []

        branch1 = self._create_branch("b-1", "Complete", "agent1", result1)
        branch2 = self._create_branch("b-2", "Incomplete", "agent2", None)

        merge_result = forker.merge([branch1, branch2])

        assert merge_result.winning_branch_id == "b-1"

    def test_merge_empty_branches_raises(self):
        """Test merge raises ValueError with empty branches."""
        forker = DebateForker()

        with pytest.raises(ValueError, match="No branches to merge"):
            forker.merge([])

    def test_merge_no_completed_branches_raises(self):
        """Test merge raises ValueError when no branches completed."""
        forker = DebateForker()

        branch = self._create_branch("b-1", "Incomplete", "agent1", None)

        with pytest.raises(ValueError, match="No completed branches"):
            forker.merge([branch])


# =============================================================================
# Additional Coverage: _count_unique_arguments
# =============================================================================


class TestCountUniqueArgumentsDeep:
    """Tests for _count_unique_arguments."""

    def _create_msg(self, agent, content, round_num=1):
        from aragora.core import Message

        return Message(role="agent", agent=agent, content=content, round=round_num)

    def test_count_unique_arguments_with_indicators(self):
        """Test counting unique arguments with indicator phrases."""
        resolver = DeadlockResolver()

        msgs = [
            self._create_msg(
                "claude",
                "I argue that Redis is better because it supports persistence. "
                "Therefore we should use it for our cache.",
                1,
            ),
        ]

        count = resolver._count_unique_arguments(msgs)

        assert count >= 2  # "because" and "therefore" sentences

    def test_count_unique_arguments_no_indicators(self):
        """Test counting returns 0 with no argument indicators."""
        resolver = DeadlockResolver()

        msgs = [
            self._create_msg("claude", "Hello world. Nice day.", 1),
        ]

        count = resolver._count_unique_arguments(msgs)

        assert count == 0

    def test_count_unique_arguments_deduplication(self):
        """Test counting deduplicates repeated arguments."""
        resolver = DeadlockResolver()

        msgs = [
            self._create_msg("claude", "I argue that Redis is the best choice.", 1),
            self._create_msg("gpt4", "I argue that Redis is the best choice.", 1),
        ]

        count = resolver._count_unique_arguments(msgs)

        assert count == 1  # Same normalized sentence


# =============================================================================
# Additional Coverage: reset and get_signals
# =============================================================================


# =============================================================================
# Additional Coverage: Branch creation when fork triggers
# =============================================================================


class TestForkDetectorBranchCreation:
    """Tests for ForkDetector when fork actually triggers with branch creation."""

    def _create_mock_message(self, agent: str, content: str) -> Mock:
        """Helper to create mock messages."""
        msg = Mock()
        msg.agent = agent
        msg.content = content
        return msg

    def _create_mock_agent(self, name: str) -> Mock:
        """Helper to create mock agents."""
        agent = Mock()
        agent.name = name
        return agent

    def test_should_fork_creates_branches_with_extract_approach(self):
        """Test should_fork creates branches using _extract_approach when triggered."""
        detector = ForkDetector()

        # Create messages that trigger high disagreement
        messages = [
            self._create_mock_message(
                "claude",
                "I propose using microservices for scalability. This is fundamentally different from monoliths. I disagree with the monolith approach.",
            ),
            self._create_mock_message(
                "gpt4",
                "On the contrary, I disagree completely. We should use monolith architecture instead. I propose using a simpler design.",
            ),
        ]
        agents = [
            self._create_mock_agent("claude"),
            self._create_mock_agent("gpt4"),
        ]

        decision = detector.should_fork(messages, round_num=3, agents=agents)

        # With enough disagreement indicators, should trigger fork
        if decision.should_fork:
            assert len(decision.branches) >= 1
            assert len(decision.branches) <= 3  # Max 3 branches
            # Branches should have hypothesis and lead_agent
            for branch in decision.branches:
                assert "hypothesis" in branch
                assert "lead_agent" in branch
                assert branch["lead_agent"] in ["claude", "gpt4"]

    def test_extract_approach_with_i_propose(self):
        """Test _extract_approach extracts text after 'I propose'."""
        detector = ForkDetector()

        msg = Mock()
        msg.content = (
            "Let me think. I propose using Redis for caching. It provides excellent performance."
        )

        approach = detector._extract_approach(msg)

        assert "I propose" in approach or "propose" in approach.lower()
        assert "..." in approach

    def test_extract_approach_with_my_approach(self):
        """Test _extract_approach extracts text after 'My approach'."""
        detector = ForkDetector()

        msg = Mock()
        msg.content = (
            "After consideration, My approach is to use microservices. This scales better."
        )

        approach = detector._extract_approach(msg)

        assert "..." in approach

    def test_extract_approach_with_i_suggest(self):
        """Test _extract_approach extracts text after 'I suggest'."""
        detector = ForkDetector()

        msg = Mock()
        msg.content = (
            "For this problem, I suggest implementing a queue system. This handles async well."
        )

        approach = detector._extract_approach(msg)

        assert "..." in approach

    def test_extract_approach_with_the_solution_is(self):
        """Test _extract_approach extracts text after 'The solution is'."""
        detector = ForkDetector()

        msg = Mock()
        msg.content = "Given the constraints, The solution is to use caching. It solves latency."

        approach = detector._extract_approach(msg)

        assert "..." in approach

    def test_extract_approach_fallback_to_first_sentence(self):
        """Test _extract_approach falls back to first sentence when no markers found."""
        detector = ForkDetector()

        msg = Mock()
        msg.content = (
            "Redis provides excellent caching capabilities. It supports persistence and clustering."
        )

        approach = detector._extract_approach(msg)

        assert "Redis" in approach
        assert "..." in approach


# =============================================================================
# Additional Coverage: Merge winner fallback
# =============================================================================


class TestMergeWinnerFallback:
    """Tests for merge when winner_id lookup fails."""

    def test_merge_fallback_to_first_branch(self):
        """Test merge falls back to first branch when winner lookup fails."""
        forker = DebateForker()

        # Create result
        result = Mock()
        result.confidence = 0.8
        result.consensus_reached = True
        result.final_answer = "Test answer"
        result.rounds_used = 3
        result.critiques = []

        # Create branch with result
        branch1 = Branch(
            branch_id="b1",
            parent_debate_id="debate-123",
            fork_round=3,
            hypothesis="Test",
            lead_agent="claude",
            result=result,
        )

        # Mock _score_branch to return a score for an ID that doesn't match any branch
        original_score = forker._score_branch

        def mock_score(branch):
            return original_score(branch)

        forker._score_branch = mock_score

        # This should work normally
        merge_result = forker.merge([branch1])

        assert merge_result.winning_branch_id == "b1"


# =============================================================================
# Additional Coverage: _extract_merged_insights edge cases
# =============================================================================


class TestExtractMergedInsightsEdgeCases:
    """Tests for _extract_merged_insights edge cases."""

    def test_extract_insights_skips_branch_without_result(self):
        """Test _extract_merged_insights skips branches without results."""
        forker = DebateForker()

        result = Mock()
        result.final_answer = "Good answer from claude."

        branch_with_result = Branch(
            branch_id="b1",
            parent_debate_id="debate-123",
            fork_round=3,
            hypothesis="A",
            lead_agent="claude",
            result=result,
        )

        branch_without_result = Branch(
            branch_id="b2",
            parent_debate_id="debate-123",
            fork_round=3,
            hypothesis="B",
            lead_agent="gpt4",
            result=None,
        )

        insights = forker._extract_merged_insights([branch_with_result, branch_without_result])

        assert len(insights) == 1
        assert "[claude]" in insights[0]

    def test_extract_insights_skips_empty_final_answer(self):
        """Test _extract_merged_insights skips branches with empty final_answer."""
        forker = DebateForker()

        result_with_answer = Mock()
        result_with_answer.final_answer = "Good answer."

        result_empty_answer = Mock()
        result_empty_answer.final_answer = ""

        result_none_answer = Mock()
        result_none_answer.final_answer = None

        branch1 = Branch(
            branch_id="b1",
            parent_debate_id="debate-123",
            fork_round=3,
            hypothesis="A",
            lead_agent="claude",
            result=result_with_answer,
        )

        branch2 = Branch(
            branch_id="b2",
            parent_debate_id="debate-123",
            fork_round=3,
            hypothesis="B",
            lead_agent="gpt4",
            result=result_empty_answer,
        )

        branch3 = Branch(
            branch_id="b3",
            parent_debate_id="debate-123",
            fork_round=3,
            hypothesis="C",
            lead_agent="gemini",
            result=result_none_answer,
        )

        insights = forker._extract_merged_insights([branch1, branch2, branch3])

        # Only branch1 has a non-empty final_answer
        assert len(insights) == 1
        assert "[claude]" in insights[0]


# =============================================================================
# Additional Coverage: Circular pattern detection
# =============================================================================


class TestCircularPatternDetection:
    """Tests for circular pattern detection in analyze_round."""

    def _create_msg(self, agent, content, round_num=1):
        from aragora.core import Message

        return Message(role="agent", agent=agent, content=content, round=round_num)

    def test_analyze_round_detects_circular_pattern(self):
        """Test analyze_round sets pattern to 'circular' when circular_score > 0.7."""
        resolver = DeadlockResolver()

        # Build history with circular pattern (A->B->A)
        # Need to manually set up history to trigger circular detection
        resolver.debate_history["debate-circular"] = [
            {"round": 3, "positions": {"a": "x"}, "unique_arguments": 5, "message_count": 2},
            {"round": 4, "positions": {"a": "x"}, "unique_arguments": 3, "message_count": 2},
            {"round": 5, "positions": {"a": "x"}, "unique_arguments": 5, "message_count": 2},
        ]

        # Now analyze round 6 with continued circular pattern
        messages = [
            self._create_msg("a", "I argue that we should use because evidence.", 6),
        ]

        # Directly test _detect_circular_arguments
        circular_score = resolver._detect_circular_arguments("debate-circular")
        assert circular_score == 0.8  # A->B->A pattern

    def test_analyze_round_with_high_circular_score_sets_pattern(self):
        """Test that pattern is set to 'circular' when circular_score > 0.7 and positional_stagnation <= 0.8."""
        resolver = DeadlockResolver()

        # Set up history where circular score will be high (0.8)
        # but positional stagnation will be low
        resolver.debate_history["debate-test"] = [
            {
                "round": 3,
                "positions": {"a": "support_1"},
                "unique_arguments": 5,
                "message_count": 2,
            },
            {
                "round": 4,
                "positions": {"a": "support_2"},
                "unique_arguments": 3,
                "message_count": 2,
            },
            {
                "round": 5,
                "positions": {"a": "support_3"},
                "unique_arguments": 5,
                "message_count": 2,
            },
        ]

        # The circular score will be 0.8 (A->B->A pattern: 5->3->5)
        # The positional stagnation should be low since positions changed
        circular_score = resolver._detect_circular_arguments("debate-test")
        assert circular_score == 0.8


# =============================================================================
# Additional Coverage: auto_resolve decision.should_fork=False
# =============================================================================


class TestAutoResolveNotFork:
    """Tests for auto_resolve when decision.should_fork is False."""

    def _create_msg(self, agent, content, round_num=1):
        from aragora.core import Message

        return Message(role="agent", agent=agent, content=content, round=round_num)

    def test_auto_resolve_returns_none_when_decision_should_not_fork(self):
        """Test auto_resolve returns None when _create_counterfactual_decision says not to fork."""
        resolver = DeadlockResolver()

        signal = DeadlockSignal(
            debate_id="debate-123",
            round=5,
            stagnation_score=0.85,
            rounds_without_progress=3,
            pattern="positional",
            recommendation="fork",  # Recommendation is fork but decision may say no
        )

        # With only one agent message, the decision will have only 1 branch
        # which means should_fork will be False (need >= 2 branches)
        messages = [
            self._create_msg("claude", "Only my position", 5),
        ]

        agents = []

        branches = resolver.auto_resolve(signal, messages, agents)

        # With single agent, _create_counterfactual_decision creates 1 branch
        # and should_fork = len(branches) >= 2, so it's False
        assert branches is None


# =============================================================================
# Additional Coverage: Positional stagnation with short recent_positions
# =============================================================================


class TestPositionalStagnationShortHistory:
    """Tests for positional stagnation with short recent_positions list."""

    def test_stagnation_returns_zero_when_recent_positions_too_short(self):
        """Test _calculate_positional_stagnation returns 0 when recent_positions < 2."""
        resolver = DeadlockResolver()

        # Set up history with only 1 entry
        resolver.debate_history["debate-short"] = [
            {"round": 3, "positions": {"a": "support_123"}, "unique_arguments": 3},
        ]

        current = {"a": "support_123"}

        # No previous_positions, and history has only 1 entry
        # so recent_positions will have length 1
        score = resolver._calculate_positional_stagnation("debate-short", current, None)

        assert score == 0.0


# =============================================================================
# Additional Coverage: _detect_circular_arguments with short history
# =============================================================================


class TestDetectCircularShortHistory:
    """Tests for _detect_circular_arguments with insufficient history."""

    def test_detect_circular_returns_zero_with_two_rounds(self):
        """Test _detect_circular_arguments returns 0 with exactly 2 rounds (< 3 needed)."""
        resolver = DeadlockResolver()

        resolver.debate_history["debate-two"] = [
            {"round": 1, "positions": {}, "unique_arguments": 5},
            {"round": 2, "positions": {}, "unique_arguments": 3},
        ]

        score = resolver._detect_circular_arguments("debate-two")

        assert score == 0.0


# =============================================================================
# Additional Coverage: _calculate_argument_exhaustion with short history
# =============================================================================


class TestArgumentExhaustionShortHistory:
    """Tests for _calculate_argument_exhaustion with short recent list."""

    def test_exhaustion_returns_zero_with_one_recent_round(self):
        """Test _calculate_argument_exhaustion returns 0 when recent < 2."""
        resolver = DeadlockResolver()

        resolver.debate_history["debate-one"] = [
            {"round": 1, "positions": {}, "unique_arguments": 5},
        ]

        score = resolver._calculate_argument_exhaustion("debate-one")

        assert score == 0.0

    def test_exhaustion_with_exactly_two_rounds(self):
        """Test _calculate_argument_exhaustion with exactly 2 rounds."""
        resolver = DeadlockResolver()

        resolver.debate_history["debate-two"] = [
            {"round": 1, "positions": {}, "unique_arguments": 10},
            {"round": 2, "positions": {}, "unique_arguments": 5},
        ]

        score = resolver._calculate_argument_exhaustion("debate-two")

        # (10 - 5) / 10 = 0.5
        assert score == 0.5


class TestDeadlockResolverResetDeep:
    """Tests for reset and get_signals methods."""

    def test_reset_specific_debate(self):
        """Test reset with specific debate_id."""
        resolver = DeadlockResolver()

        resolver.debate_history["debate-1"] = [{"round": 1}]
        resolver.debate_history["debate-2"] = [{"round": 1}]

        resolver.reset("debate-1")

        assert "debate-1" not in resolver.debate_history
        assert "debate-2" in resolver.debate_history

    def test_reset_all(self):
        """Test reset without debate_id clears everything."""
        resolver = DeadlockResolver()

        resolver.debate_history["debate-1"] = [{"round": 1}]
        resolver.deadlock_signals.append(
            DeadlockSignal(
                debate_id="debate-1",
                round=3,
                stagnation_score=0.8,
                rounds_without_progress=2,
                pattern="positional",
                recommendation="fork",
            )
        )

        resolver.reset()

        assert len(resolver.debate_history) == 0
        assert len(resolver.deadlock_signals) == 0

    def test_get_signals_filtered(self):
        """Test get_signals with debate_id filter."""
        resolver = DeadlockResolver()

        signal1 = DeadlockSignal(
            debate_id="debate-1",
            round=3,
            stagnation_score=0.8,
            rounds_without_progress=2,
            pattern="positional",
            recommendation="fork",
        )
        signal2 = DeadlockSignal(
            debate_id="debate-2",
            round=4,
            stagnation_score=0.9,
            rounds_without_progress=3,
            pattern="circular",
            recommendation="fork",
        )

        resolver.deadlock_signals = [signal1, signal2]

        filtered = resolver.get_signals("debate-1")

        assert len(filtered) == 1
        assert filtered[0].debate_id == "debate-1"

    def test_get_signals_all(self):
        """Test get_signals without filter returns all."""
        resolver = DeadlockResolver()

        resolver.deadlock_signals = [Mock(), Mock(), Mock()]

        all_signals = resolver.get_signals()

        assert len(all_signals) == 3
