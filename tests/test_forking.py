"""Tests for debate forking module - branch creation, detection, and merging."""

import asyncio
import pytest
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

from aragora.core import Message, DebateResult, Environment, Critique
from aragora.debate.forking import (
    ForkPoint,
    Branch,
    ForkDecision,
    MergeResult,
    ForkDetector,
    DebateForker,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fork_detector():
    """Create ForkDetector instance."""
    return ForkDetector()


@pytest.fixture
def forker():
    """Create DebateForker instance."""
    return DebateForker()


@pytest.fixture
def sample_message():
    """Create sample Message."""
    return Message(
        role="proposer",
        agent="claude",
        content="I propose we use microservices architecture.",
        round=1,
    )


@pytest.fixture
def sample_result():
    """Create sample DebateResult."""
    return DebateResult(
        id="result-123",
        task="Design the system",
        final_answer="Use microservices with async messaging.",
        confidence=0.85,
        consensus_reached=True,
        rounds_used=3,
    )


@pytest.fixture
def agreeing_messages():
    """Create messages where agents agree."""
    return [
        Message(
            role="proposer", agent="claude", content="I think we should use REST APIs.", round=1
        ),
        Message(
            role="critic", agent="gemini", content="I agree, REST APIs are a good choice.", round=1
        ),
    ]


@pytest.fixture
def disagreeing_messages():
    """Create messages where agents fundamentally disagree."""
    return [
        Message(
            role="proposer",
            agent="claude",
            content="I propose we use microservices for scalability.",
            round=2,
        ),
        Message(
            role="critic",
            agent="gemini",
            content="I disagree. A monolith is better. Wrong approach for this scale.",
            round=2,
        ),
    ]


@pytest.fixture
def tech_disagreement_messages():
    """Create messages with different tech choices."""
    return [
        Message(
            role="proposer",
            agent="claude",
            content="We should use SQL databases for reliability.",
            round=2,
        ),
        Message(role="critic", agent="gemini", content="I prefer NoSQL for flexibility.", round=2),
    ]


@pytest.fixture
def sample_fork_decision():
    """Create sample ForkDecision."""
    return ForkDecision(
        should_fork=True,
        reason="Fundamental disagreement on architecture",
        branches=[
            {"hypothesis": "Microservices approach", "lead_agent": "claude"},
            {"hypothesis": "Monolith approach", "lead_agent": "gemini"},
        ],
        disagreement_score=0.8,
    )


@pytest.fixture
def sample_branches(sample_message):
    """Create sample Branch instances."""
    return [
        Branch(
            branch_id="branch-1",
            parent_debate_id="debate-123",
            fork_round=2,
            hypothesis="Microservices approach",
            lead_agent="claude",
            messages=[sample_message],
        ),
        Branch(
            branch_id="branch-2",
            parent_debate_id="debate-123",
            fork_round=2,
            hypothesis="Monolith approach",
            lead_agent="gemini",
            messages=[sample_message],
        ),
    ]


# =============================================================================
# ForkPoint Dataclass Tests
# =============================================================================


class TestForkPoint:
    """Tests for ForkPoint dataclass."""

    def test_all_fields_initialized(self):
        """ForkPoint should initialize all fields correctly."""
        fork_point = ForkPoint(
            round=3,
            reason="Fundamental disagreement",
            disagreeing_agents=["claude", "gemini"],
            parent_debate_id="debate-123",
            branch_ids=["branch-1", "branch-2"],
        )

        assert fork_point.round == 3
        assert fork_point.reason == "Fundamental disagreement"
        assert fork_point.disagreeing_agents == ["claude", "gemini"]
        assert fork_point.parent_debate_id == "debate-123"
        assert fork_point.branch_ids == ["branch-1", "branch-2"]

    def test_created_at_auto_populated(self):
        """created_at should be auto-populated with timestamp."""
        fork_point = ForkPoint(
            round=1,
            reason="Test",
            disagreeing_agents=[],
            parent_debate_id="test",
            branch_ids=[],
        )

        assert fork_point.created_at is not None
        assert len(fork_point.created_at) > 0
        # Should be ISO format
        assert "T" in fork_point.created_at


# =============================================================================
# Branch Dataclass Tests
# =============================================================================


class TestBranch:
    """Tests for Branch dataclass."""

    def test_all_fields_initialized(self):
        """Branch should initialize all fields correctly."""
        branch = Branch(
            branch_id="branch-abc",
            parent_debate_id="debate-123",
            fork_round=2,
            hypothesis="Test hypothesis",
            lead_agent="claude",
        )

        assert branch.branch_id == "branch-abc"
        assert branch.parent_debate_id == "debate-123"
        assert branch.fork_round == 2
        assert branch.hypothesis == "Test hypothesis"
        assert branch.lead_agent == "claude"
        assert branch.messages == []
        assert branch.result is None

    def test_created_at_auto_populated(self):
        """created_at should be auto-populated."""
        branch = Branch(
            branch_id="test",
            parent_debate_id="test",
            fork_round=1,
            hypothesis="test",
            lead_agent="test",
        )

        assert branch.created_at is not None
        assert "T" in branch.created_at

    def test_is_complete_false_when_no_result(self):
        """is_complete should return False when result is None."""
        branch = Branch(
            branch_id="test",
            parent_debate_id="test",
            fork_round=1,
            hypothesis="test",
            lead_agent="test",
        )

        assert branch.is_complete is False

    def test_is_complete_true_when_result_set(self, sample_result):
        """is_complete should return True when result is set."""
        branch = Branch(
            branch_id="test",
            parent_debate_id="test",
            fork_round=1,
            hypothesis="test",
            lead_agent="test",
            result=sample_result,
        )

        assert branch.is_complete is True


# =============================================================================
# ForkDecision Dataclass Tests
# =============================================================================


class TestForkDecision:
    """Tests for ForkDecision dataclass."""

    def test_all_fields_initialized(self):
        """ForkDecision should initialize all fields correctly."""
        decision = ForkDecision(
            should_fork=True,
            reason="Agents disagree",
            branches=[{"hypothesis": "A", "lead_agent": "x"}],
            disagreement_score=0.75,
        )

        assert decision.should_fork is True
        assert decision.reason == "Agents disagree"
        assert decision.disagreement_score == 0.75

    def test_branches_stores_dict_specs(self):
        """branches should store dict specifications."""
        branches = [
            {"hypothesis": "Approach A", "lead_agent": "claude"},
            {"hypothesis": "Approach B", "lead_agent": "gemini"},
        ]
        decision = ForkDecision(
            should_fork=True,
            reason="Test",
            branches=branches,
            disagreement_score=0.8,
        )

        assert len(decision.branches) == 2
        assert decision.branches[0]["hypothesis"] == "Approach A"
        assert decision.branches[1]["lead_agent"] == "gemini"


# =============================================================================
# MergeResult Dataclass Tests
# =============================================================================


class TestMergeResult:
    """Tests for MergeResult dataclass."""

    def test_all_fields_initialized(self, sample_result):
        """MergeResult should initialize all fields correctly."""
        merge_result = MergeResult(
            winning_branch_id="branch-1",
            winning_hypothesis="Microservices",
            comparison_summary="Branch 1 won with higher confidence",
            all_branch_results={"branch-1": sample_result},
            merged_insights=["Insight 1", "Insight 2"],
        )

        assert merge_result.winning_branch_id == "branch-1"
        assert merge_result.winning_hypothesis == "Microservices"
        assert "Branch 1 won" in merge_result.comparison_summary

    def test_all_branch_results_maps_correctly(self, sample_result):
        """all_branch_results should map branch_id to result."""
        result2 = DebateResult(id="result-2", consensus_reached=False)
        merge_result = MergeResult(
            winning_branch_id="branch-1",
            winning_hypothesis="Test",
            comparison_summary="Test",
            all_branch_results={
                "branch-1": sample_result,
                "branch-2": result2,
            },
            merged_insights=[],
        )

        assert "branch-1" in merge_result.all_branch_results
        assert "branch-2" in merge_result.all_branch_results
        assert merge_result.all_branch_results["branch-1"].id == "result-123"


# =============================================================================
# ForkDetector.should_fork Tests
# =============================================================================


class TestForkDetectorShouldFork:
    """Tests for ForkDetector.should_fork method."""

    def test_no_fork_when_round_below_min(self, fork_detector, agreeing_messages):
        """should_fork returns no fork when round < MIN_ROUNDS_BEFORE_FORK."""
        decision = fork_detector.should_fork(
            messages=agreeing_messages,
            round_num=1,  # Below threshold of 2
            agents=[],
        )

        assert decision.should_fork is False
        assert "Too early" in decision.reason
        assert decision.disagreement_score == 0.0

    def test_no_fork_when_less_than_2_agents(self, fork_detector):
        """should_fork returns no fork when less than 2 agents have messages."""
        messages = [
            Message(role="proposer", agent="claude", content="Only one agent", round=2),
        ]

        decision = fork_detector.should_fork(
            messages=messages,
            round_num=2,
            agents=[],
        )

        assert decision.should_fork is False
        assert "Not enough agents" in decision.reason

    def test_no_fork_when_no_disagreements(self, fork_detector, agreeing_messages):
        """should_fork returns no fork when no disagreements detected."""
        decision = fork_detector.should_fork(
            messages=agreeing_messages,
            round_num=3,
            agents=[],
        )

        assert decision.should_fork is False
        # Either "No fundamental disagreements" or low score
        assert decision.disagreement_score < 0.7

    def test_no_fork_when_score_below_threshold(self, fork_detector):
        """should_fork returns no fork when disagreement score below threshold."""
        # Messages with mild disagreement
        messages = [
            Message(role="proposer", agent="claude", content="I think option A is good.", round=2),
            Message(
                role="critic", agent="gemini", content="However, option B has merits too.", round=2
            ),
        ]

        decision = fork_detector.should_fork(
            messages=messages,
            round_num=3,
            agents=[],
        )

        # Score should be below 0.7 threshold
        if decision.disagreement_score >= 0.7:
            assert decision.should_fork is True
        else:
            assert decision.should_fork is False

    def test_fork_when_score_above_threshold(self, fork_detector, disagreeing_messages):
        """should_fork returns fork when disagreement score above threshold."""
        decision = fork_detector.should_fork(
            messages=disagreeing_messages,
            round_num=3,
            agents=[],
        )

        assert decision.should_fork is True
        assert decision.disagreement_score >= 0.7
        assert len(decision.branches) > 0

    def test_branches_limited_to_max_3(self, fork_detector):
        """should_fork limits branches to max 3."""
        # Create messages from 5 agents all disagreeing
        messages = [
            Message(
                role="proposer",
                agent=f"agent{i}",
                content="I disagree with everyone. Wrong approach.",
                round=2,
            )
            for i in range(5)
        ]

        decision = fork_detector.should_fork(
            messages=messages,
            round_num=3,
            agents=[],
        )

        if decision.should_fork:
            assert len(decision.branches) <= 3

    def test_returns_correct_disagreeing_agents(self, fork_detector, disagreeing_messages):
        """should_fork returns branches with correct lead agents."""
        decision = fork_detector.should_fork(
            messages=disagreeing_messages,
            round_num=3,
            agents=[],
        )

        if decision.should_fork:
            lead_agents = [b["lead_agent"] for b in decision.branches]
            # Should include the disagreeing agents
            assert "claude" in lead_agents or "gemini" in lead_agents

    def test_disagreement_score_included(self, fork_detector, disagreeing_messages):
        """should_fork includes disagreement score in decision."""
        decision = fork_detector.should_fork(
            messages=disagreeing_messages,
            round_num=3,
            agents=[],
        )

        assert decision.disagreement_score >= 0.0
        assert decision.disagreement_score <= 1.0


# =============================================================================
# ForkDetector._detect_disagreements Tests
# =============================================================================


class TestForkDetectorDetectDisagreements:
    """Tests for ForkDetector._detect_disagreements method."""

    def test_empty_for_identical_messages(self, fork_detector):
        """_detect_disagreements returns empty list for identical messages."""
        msg = Message(role="proposer", agent="claude", content="Same content", round=1)
        latest_by_agent = {
            "claude": msg,
            "gemini": Message(role="critic", agent="gemini", content="Same content", round=1),
        }

        disagreements = fork_detector._detect_disagreements(latest_by_agent)

        # Should have low scores filtered out
        high_score = [d for d in disagreements if d["score"] > 0.3]
        assert len(high_score) == 0

    def test_detects_disagreement_phrases(self, fork_detector):
        """_detect_disagreements detects explicit disagreement phrases."""
        latest_by_agent = {
            "claude": Message(role="proposer", agent="claude", content="Use REST", round=1),
            "gemini": Message(
                role="critic", agent="gemini", content="I disagree. GraphQL is better.", round=1
            ),
        }

        disagreements = fork_detector._detect_disagreements(latest_by_agent)

        assert len(disagreements) >= 1
        assert disagreements[0]["score"] > 0

    def test_pairs_all_agents(self, fork_detector):
        """_detect_disagreements pairs all agents correctly."""
        # Use strong disagreement phrases to ensure detection
        latest_by_agent = {
            "claude": Message(
                role="proposer", agent="claude", content="We should use microservices", round=1
            ),
            "gemini": Message(
                role="critic",
                agent="gemini",
                content="I disagree. We should not. Monolith is better.",
                round=1,
            ),
            "gpt": Message(
                role="critic", agent="gpt", content="I also disagree. Incorrect approach.", round=1
            ),
        }

        disagreements = fork_detector._detect_disagreements(latest_by_agent)

        # Should have detected some disagreements
        assert len(disagreements) >= 1

        # Check that agent pairs were compared
        all_agents_in_pairs = set()
        for d in disagreements:
            all_agents_in_pairs.update(d["agents"])

        # At least some agents should appear in disagreements
        assert len(all_agents_in_pairs) >= 2

    def test_score_threshold_filters(self, fork_detector):
        """_detect_disagreements filters disagreements below 0.3 score."""
        latest_by_agent = {
            "claude": Message(
                role="proposer", agent="claude", content="Nice weather today", round=1
            ),
            "gemini": Message(role="critic", agent="gemini", content="Indeed it is", round=1),
        }

        disagreements = fork_detector._detect_disagreements(latest_by_agent)

        # Low scores should be filtered
        for d in disagreements:
            assert d["score"] > 0.3


# =============================================================================
# ForkDetector._calculate_disagreement Tests
# =============================================================================


class TestForkDetectorCalculateDisagreement:
    """Tests for ForkDetector._calculate_disagreement method."""

    def test_zero_score_for_agreeing(self, fork_detector):
        """_calculate_disagreement returns ~0 score for agreeing messages."""
        msg_a = Message(role="proposer", agent="claude", content="Let's use Python", round=1)
        msg_b = Message(role="critic", agent="gemini", content="Python is a good choice", round=1)

        score, reason = fork_detector._calculate_disagreement(msg_a, msg_b)

        assert score < 0.3

    def test_detects_disagreement_phrases(self, fork_detector):
        """_calculate_disagreement adds 0.2 for disagreement phrases."""
        msg_a = Message(role="proposer", agent="claude", content="Use REST", round=1)
        msg_b = Message(
            role="critic", agent="gemini", content="I disagree with that approach", round=1
        )

        score, reason = fork_detector._calculate_disagreement(msg_a, msg_b)

        assert score >= 0.2
        assert "disagree" in reason.lower()

    def test_detects_contradictory_should(self, fork_detector):
        """_calculate_disagreement adds 0.3 for should/should not contradiction."""
        msg_a = Message(role="proposer", agent="claude", content="We should use caching", round=1)
        msg_b = Message(role="critic", agent="gemini", content="We should not use caching", round=1)

        score, reason = fork_detector._calculate_disagreement(msg_a, msg_b)

        assert score >= 0.3
        assert "should" in reason.lower()

    def test_detects_different_tech_choices(self, fork_detector):
        """_calculate_disagreement adds 0.3 for different tech choices."""
        # Use exact tech_terms from the code: microservice vs monolith
        msg_a = Message(
            role="proposer", agent="claude", content="Use microservice architecture", round=1
        )
        msg_b = Message(role="critic", agent="gemini", content="Use a monolith instead", round=1)

        score, reason = fork_detector._calculate_disagreement(msg_a, msg_b)

        assert score >= 0.3
        assert "tech" in reason.lower() or "microservice" in reason.lower()

    def test_caps_score_at_one(self, fork_detector):
        """_calculate_disagreement caps score at 1.0."""
        # Message with multiple disagreement indicators
        msg_a = Message(
            role="proposer",
            agent="claude",
            content="We should use SQL with REST and sync processing",
            round=1,
        )
        msg_b = Message(
            role="critic",
            agent="gemini",
            content="I disagree. We should not use that. NoSQL with GraphQL and async is better. Wrong approach entirely.",
            round=1,
        )

        score, reason = fork_detector._calculate_disagreement(msg_a, msg_b)

        assert score <= 1.0


# =============================================================================
# ForkDetector._extract_approach Tests
# =============================================================================


class TestForkDetectorExtractApproach:
    """Tests for ForkDetector._extract_approach method."""

    def test_extracts_after_i_propose(self, fork_detector):
        """_extract_approach extracts text after 'I propose'."""
        msg = Message(
            role="proposer",
            agent="claude",
            content="After analysis, I propose we use microservices. This will help scale.",
            round=1,
        )

        approach = fork_detector._extract_approach(msg)

        assert "propose" in approach.lower() or "microservices" in approach.lower()

    def test_extracts_after_my_approach(self, fork_detector):
        """_extract_approach extracts text after 'My approach'."""
        msg = Message(
            role="proposer",
            agent="claude",
            content="Given the requirements, my approach is to use event sourcing. It fits well.",
            round=1,
        )

        approach = fork_detector._extract_approach(msg)

        assert "approach" in approach.lower() or "event" in approach.lower()

    def test_falls_back_to_first_100_chars(self, fork_detector):
        """_extract_approach falls back to first 100 chars."""
        msg = Message(
            role="proposer",
            agent="claude",
            content="This is a message without any proposal markers. It just discusses options.",
            round=1,
        )

        approach = fork_detector._extract_approach(msg)

        assert len(approach) <= 103  # 100 chars + "..."
        assert "..." in approach


# =============================================================================
# DebateForker.fork Tests
# =============================================================================


class TestDebateForkerFork:
    """Tests for DebateForker.fork method."""

    def test_creates_branches_with_unique_ids(self, forker, sample_fork_decision, sample_message):
        """fork creates branches with unique IDs."""
        branches = forker.fork(
            parent_debate_id="debate-123",
            fork_round=2,
            messages_so_far=[sample_message],
            decision=sample_fork_decision,
        )

        assert len(branches) == 2
        ids = [b.branch_id for b in branches]
        assert len(set(ids)) == len(ids)  # All unique

    def test_copies_messages_to_each_branch(self, forker, sample_fork_decision):
        """fork copies messages_so_far to each branch."""
        messages = [
            Message(role="proposer", agent="claude", content="Msg 1", round=1),
            Message(role="critic", agent="gemini", content="Msg 2", round=1),
        ]

        branches = forker.fork(
            parent_debate_id="debate-123",
            fork_round=2,
            messages_so_far=messages,
            decision=sample_fork_decision,
        )

        for branch in branches:
            assert len(branch.messages) == 2
            # Should be deep copies, not references
            assert branch.messages is not messages

    def test_creates_fork_point_with_correct_data(
        self, forker, sample_fork_decision, sample_message
    ):
        """fork creates ForkPoint with correct round/reason."""
        forker.fork(
            parent_debate_id="debate-123",
            fork_round=3,
            messages_so_far=[sample_message],
            decision=sample_fork_decision,
        )

        fork_points = forker.get_fork_history("debate-123")
        assert len(fork_points) == 1
        assert fork_points[0].round == 3
        assert fork_points[0].reason == sample_fork_decision.reason

    def test_stores_branches_in_dict(self, forker, sample_fork_decision, sample_message):
        """fork stores branches in self.branches dict."""
        branches = forker.fork(
            parent_debate_id="debate-123",
            fork_round=2,
            messages_so_far=[sample_message],
            decision=sample_fork_decision,
        )

        stored = forker.get_branches("debate-123")
        assert len(stored) == len(branches)

    def test_stores_fork_point_in_dict(self, forker, sample_fork_decision, sample_message):
        """fork stores fork_point in self.fork_points dict."""
        forker.fork(
            parent_debate_id="debate-123",
            fork_round=2,
            messages_so_far=[sample_message],
            decision=sample_fork_decision,
        )

        fork_points = forker.get_fork_history("debate-123")
        assert len(fork_points) == 1
        assert fork_points[0].parent_debate_id == "debate-123"


# =============================================================================
# DebateForker.run_branches Tests
# =============================================================================


class TestDebateForkerRunBranches:
    """Tests for DebateForker.run_branches method."""

    @pytest.mark.asyncio
    async def test_runs_all_branches_parallel(self, forker, sample_branches):
        """run_branches runs all branches in parallel."""
        call_order = []

        async def mock_run_debate(env, agents, initial_messages=None):
            call_order.append(env.task)
            await asyncio.sleep(0.01)
            return DebateResult(consensus_reached=True, confidence=0.8)

        results = await forker.run_branches(
            branches=sample_branches,
            env=Environment(task="test"),
            agents=[],
            run_debate_fn=mock_run_debate,
        )

        assert len(results) == 2
        assert len(call_order) == 2

    @pytest.mark.asyncio
    async def test_populates_result_on_branches(self, forker, sample_branches):
        """run_branches populates result on each branch."""

        async def mock_run_debate(env, agents, initial_messages=None):
            return DebateResult(consensus_reached=True, confidence=0.9, rounds_used=2)

        results = await forker.run_branches(
            branches=sample_branches,
            env=Environment(task="test"),
            agents=[],
            run_debate_fn=mock_run_debate,
        )

        for branch in results:
            assert branch.result is not None
            assert branch.result.consensus_reached is True

    @pytest.mark.asyncio
    async def test_handles_exceptions_in_branches(self, forker, sample_branches):
        """run_branches handles exceptions in individual branches."""
        call_count = [0]

        async def mock_run_debate(env, agents, initial_messages=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Branch failed")
            return DebateResult(consensus_reached=True)

        results = await forker.run_branches(
            branches=sample_branches,
            env=Environment(task="test"),
            agents=[],
            run_debate_fn=mock_run_debate,
        )

        # Should have one successful result
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_returns_only_completed_branches(self, forker, sample_branches):
        """run_branches returns only completed branches."""

        async def mock_run_debate(env, agents, initial_messages=None):
            if "branch-1" in env.task:
                raise Exception("Failed")
            return DebateResult(consensus_reached=True)

        # Modify branches to have identifiable tasks
        sample_branches[0].hypothesis = "branch-1"
        sample_branches[1].hypothesis = "branch-2"

        results = await forker.run_branches(
            branches=sample_branches,
            env=Environment(task=""),
            agents=[],
            run_debate_fn=mock_run_debate,
        )

        # Only successful branch returned
        for branch in results:
            assert branch.result is not None


# =============================================================================
# DebateForker.merge Tests
# =============================================================================


class TestDebateForkerMerge:
    """Tests for DebateForker.merge method."""

    def test_raises_when_no_branches(self, forker):
        """merge raises ValueError when no branches."""
        with pytest.raises(ValueError, match="No branches"):
            forker.merge([])

    def test_raises_when_no_completed_branches(self, forker, sample_branches):
        """merge raises ValueError when no completed branches."""
        # Branches without results
        with pytest.raises(ValueError, match="No completed"):
            forker.merge(sample_branches)

    def test_selects_highest_score_winner(self, forker, sample_result):
        """merge selects branch with highest score as winner."""
        branch1 = Branch(
            branch_id="branch-1",
            parent_debate_id="test",
            fork_round=1,
            hypothesis="Low confidence",
            lead_agent="claude",
            result=DebateResult(consensus_reached=False, confidence=0.3, rounds_used=5),
        )
        branch2 = Branch(
            branch_id="branch-2",
            parent_debate_id="test",
            fork_round=1,
            hypothesis="High confidence",
            lead_agent="gemini",
            result=DebateResult(consensus_reached=True, confidence=0.9, rounds_used=2),
        )

        result = forker.merge([branch1, branch2])

        assert result.winning_branch_id == "branch-2"
        assert result.winning_hypothesis == "High confidence"

    def test_returns_merge_result_with_all_branches(self, forker, sample_result):
        """merge returns MergeResult with all branch results."""
        branch1 = Branch(
            branch_id="b1",
            parent_debate_id="test",
            fork_round=1,
            hypothesis="A",
            lead_agent="claude",
            result=sample_result,
        )
        branch2 = Branch(
            branch_id="b2",
            parent_debate_id="test",
            fork_round=1,
            hypothesis="B",
            lead_agent="gemini",
            result=DebateResult(consensus_reached=False),
        )

        result = forker.merge([branch1, branch2])

        assert "b1" in result.all_branch_results
        assert "b2" in result.all_branch_results
        assert result.comparison_summary is not None


# =============================================================================
# DebateForker._score_branch Tests
# =============================================================================


class TestDebateForkerScoreBranch:
    """Tests for DebateForker._score_branch method."""

    def test_returns_zero_without_result(self, forker):
        """_score_branch returns 0 for branch without result."""
        branch = Branch(
            branch_id="test",
            parent_debate_id="test",
            fork_round=1,
            hypothesis="test",
            lead_agent="test",
        )

        score = forker._score_branch(branch)

        assert score == 0.0

    def test_adds_for_consensus_reached(self, forker):
        """_score_branch adds 0.3 for consensus_reached."""
        branch_with = Branch(
            branch_id="test",
            parent_debate_id="test",
            fork_round=1,
            hypothesis="test",
            lead_agent="test",
            result=DebateResult(consensus_reached=True, confidence=0.0, rounds_used=10),
        )
        branch_without = Branch(
            branch_id="test",
            parent_debate_id="test",
            fork_round=1,
            hypothesis="test",
            lead_agent="test",
            result=DebateResult(consensus_reached=False, confidence=0.0, rounds_used=10),
        )

        score_with = forker._score_branch(branch_with)
        score_without = forker._score_branch(branch_without)

        assert score_with > score_without
        assert score_with - score_without >= 0.25  # Approximately 0.3

    def test_adds_weighted_confidence(self, forker):
        """_score_branch adds weighted confidence score."""
        branch_high = Branch(
            branch_id="test",
            parent_debate_id="test",
            fork_round=1,
            hypothesis="test",
            lead_agent="test",
            result=DebateResult(consensus_reached=False, confidence=1.0, rounds_used=10),
        )
        branch_low = Branch(
            branch_id="test",
            parent_debate_id="test",
            fork_round=1,
            hypothesis="test",
            lead_agent="test",
            result=DebateResult(consensus_reached=False, confidence=0.0, rounds_used=10),
        )

        score_high = forker._score_branch(branch_high)
        score_low = forker._score_branch(branch_low)

        assert score_high > score_low

    def test_penalizes_high_severity_critiques(self, forker):
        """_score_branch penalizes high severity critiques."""
        low_severity = [
            Critique(
                agent="x",
                target_agent="y",
                target_content="test",
                issues=["minor issue"],
                suggestions=["fix it"],
                severity=0.2,
                reasoning="Minor concern",
            )
        ]
        high_severity = [
            Critique(
                agent="x",
                target_agent="y",
                target_content="test",
                issues=["major issue"],
                suggestions=["urgent fix"],
                severity=0.9,
                reasoning="Critical problem",
            )
        ]

        branch_low = Branch(
            branch_id="test",
            parent_debate_id="test",
            fork_round=1,
            hypothesis="test",
            lead_agent="test",
            result=DebateResult(
                consensus_reached=True, confidence=0.8, rounds_used=3, critiques=low_severity
            ),
        )
        branch_high = Branch(
            branch_id="test",
            parent_debate_id="test",
            fork_round=1,
            hypothesis="test",
            lead_agent="test",
            result=DebateResult(
                consensus_reached=True, confidence=0.8, rounds_used=3, critiques=high_severity
            ),
        )

        score_low = forker._score_branch(branch_low)
        score_high = forker._score_branch(branch_high)

        assert score_low > score_high  # Lower severity = higher score


# =============================================================================
# DebateForker Helper Methods Tests
# =============================================================================


class TestDebateForkerHelpers:
    """Tests for DebateForker helper methods."""

    def test_generate_comparison_formats_by_score(self, forker, sample_result):
        """_generate_comparison formats branches by score."""
        branch1 = Branch(
            branch_id="b1",
            parent_debate_id="test",
            fork_round=1,
            hypothesis="Winner",
            lead_agent="claude",
            result=DebateResult(consensus_reached=True, confidence=0.9, rounds_used=2),
        )
        branch2 = Branch(
            branch_id="b2",
            parent_debate_id="test",
            fork_round=1,
            hypothesis="Loser",
            lead_agent="gemini",
            result=DebateResult(consensus_reached=False, confidence=0.3, rounds_used=5),
        )

        scores = {"b1": 0.9, "b2": 0.3}
        comparison = forker._generate_comparison([branch1, branch2], scores)

        assert "Winner" in comparison
        assert "Loser" in comparison
        # Winner should appear before loser (sorted by score)
        assert comparison.index("Winner") < comparison.index("Loser")

    def test_extract_merged_insights_from_final_answer(self, forker):
        """_extract_merged_insights extracts from final_answer."""
        branch = Branch(
            branch_id="test",
            parent_debate_id="test",
            fork_round=1,
            hypothesis="test",
            lead_agent="claude",
            result=DebateResult(
                final_answer="Use microservices for scalability. They help with deployment.",
            ),
        )

        insights = forker._extract_merged_insights([branch])

        assert len(insights) >= 1
        assert "claude" in insights[0].lower()

    def test_get_fork_history_and_branches_return_stored(
        self, forker, sample_fork_decision, sample_message
    ):
        """get_fork_history and get_branches return stored data."""
        forker.fork(
            parent_debate_id="debate-xyz",
            fork_round=2,
            messages_so_far=[sample_message],
            decision=sample_fork_decision,
        )

        history = forker.get_fork_history("debate-xyz")
        branches = forker.get_branches("debate-xyz")

        assert len(history) == 1
        assert len(branches) == 2

        # Non-existent debate returns empty
        assert forker.get_fork_history("nonexistent") == []
        assert forker.get_branches("nonexistent") == []
