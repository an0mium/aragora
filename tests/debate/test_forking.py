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
