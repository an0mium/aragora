"""
Integration tests for Agent-as-a-Judge bias mitigation.

Tests the bias mitigation features in realistic debate scenarios:
- Position shuffling in vote collection
- Self-vote detection in weight calculation
- Verbosity normalization in consensus
- Judge deliberation protocol
- Process evaluation bonuses
"""

import pytest
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.debate.protocol import DebateProtocol
from aragora.debate.phases.vote_collector import VoteCollector, VoteCollectorConfig
from aragora.debate.phases.weight_calculator import WeightCalculator, WeightCalculatorConfig
from aragora.debate.judge_selector import JudgePanel, JudgingStrategy, JudgeVote


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockAgent:
    """Mock agent for testing."""
    name: str
    role: str = "debater"


@dataclass
class MockVote:
    """Mock vote for testing."""
    agent: str
    choice: str
    confidence: float
    reasoning: str = ""
    continue_debate: bool = False


@dataclass
class MockEnvironment:
    """Mock environment for testing."""
    task: str = "Test debate task"


@dataclass
class MockResult:
    """Mock result for testing."""
    votes: List[Any] = field(default_factory=list)
    final_answer: str = ""
    consensus_reached: bool = False
    confidence: float = 0.0
    winner: str = ""
    rounds_used: int = 0
    id: str = "test-debate"


@dataclass
class MockDebateContext:
    """Mock debate context for testing."""
    agents: List[MockAgent] = field(default_factory=list)
    proposals: Dict[str, str] = field(default_factory=dict)
    env: MockEnvironment = field(default_factory=MockEnvironment)
    result: MockResult = field(default_factory=MockResult)
    context_messages: List[Any] = field(default_factory=list)
    winner_agent: Optional[str] = None
    vote_tally: Dict[str, float] = field(default_factory=dict)
    cancellation_token: Optional[Any] = None


# =============================================================================
# Position Shuffling Integration Tests
# =============================================================================


class TestPositionShufflingIntegration:
    """Integration tests for position shuffling in vote collection."""

    @pytest.mark.asyncio
    async def test_vote_collector_with_shuffling_enabled(self):
        """Test VoteCollector uses shuffling when enabled."""
        agents = [MockAgent(name="alice"), MockAgent(name="bob"), MockAgent(name="carol")]
        proposals = {
            "alice": "Alice's proposal",
            "bob": "Bob's proposal",
            "carol": "Carol's proposal",
        }

        # Track which proposal orderings were seen
        orderings_seen = []

        async def mock_vote(agent, props, task):
            orderings_seen.append(list(props.keys()))
            return MockVote(
                agent=agent.name,
                choice=list(props.keys())[0],  # Vote for first proposal
                confidence=0.8,
            )

        config = VoteCollectorConfig(
            vote_with_agent=mock_vote,
            enable_position_shuffling=True,
            position_shuffling_permutations=3,
            position_shuffling_seed=42,
        )

        collector = VoteCollector(config)
        ctx = MockDebateContext(agents=agents, proposals=proposals)

        votes = await collector.collect_votes(ctx)

        # Should have collected votes (may be averaged)
        assert len(votes) > 0
        # Should have seen multiple orderings (3 permutations * 3 agents = 9 calls)
        assert len(orderings_seen) >= 3

    @pytest.mark.asyncio
    async def test_vote_collector_without_shuffling(self):
        """Test VoteCollector works normally without shuffling."""
        agents = [MockAgent(name="alice"), MockAgent(name="bob")]
        proposals = {"alice": "A", "bob": "B"}

        vote_calls = []

        async def mock_vote(agent, props, task):
            vote_calls.append((agent.name, list(props.keys())))
            return MockVote(agent=agent.name, choice="alice", confidence=0.9)

        config = VoteCollectorConfig(
            vote_with_agent=mock_vote,
            enable_position_shuffling=False,  # Disabled
        )

        collector = VoteCollector(config)
        ctx = MockDebateContext(agents=agents, proposals=proposals)

        votes = await collector.collect_votes(ctx)

        # Should get one vote per agent
        assert len(votes) == 2
        # Each agent should have been called once
        assert len(vote_calls) == 2


# =============================================================================
# Self-Vote Detection Integration Tests
# =============================================================================


class TestSelfVoteIntegration:
    """Integration tests for self-vote detection in weight calculation."""

    def test_weight_calculator_detects_self_votes(self):
        """Test WeightCalculator applies self-vote penalty."""
        agents = [MockAgent(name="alice"), MockAgent(name="bob")]
        proposals = {"alice": "Alice's proposal", "bob": "Bob's proposal"}

        votes = [
            MockVote(agent="alice", choice="alice", confidence=0.9),  # Self-vote
            MockVote(agent="bob", choice="alice", confidence=0.8),    # Not self-vote
        ]

        config = WeightCalculatorConfig(
            enable_self_vote_mitigation=True,
            self_vote_mode="downweight",
            self_vote_downweight=0.5,
        )

        calculator = WeightCalculator(config=config)
        weights = calculator.compute_weights_with_context(agents, votes, proposals)

        # Alice's weight should be lower due to self-vote
        assert weights["alice"] < weights["bob"]

    def test_weight_calculator_exclude_mode(self):
        """Test WeightCalculator excludes self-votes when configured."""
        agents = [MockAgent(name="alice")]
        proposals = {"alice": "Alice's proposal"}

        votes = [MockVote(agent="alice", choice="alice", confidence=0.9)]

        config = WeightCalculatorConfig(
            enable_self_vote_mitigation=True,
            self_vote_mode="exclude",
            min_weight=0.0,  # Allow zero weight for exclude mode
        )

        calculator = WeightCalculator(config=config)
        weights = calculator.compute_weights_with_context(agents, votes, proposals)

        # Alice's weight should be zero (excluded)
        assert weights["alice"] == 0.0


# =============================================================================
# Verbosity Normalization Integration Tests
# =============================================================================


class TestVerbosityIntegration:
    """Integration tests for verbosity normalization."""

    def test_weight_calculator_penalizes_verbose_proposals(self):
        """Test that voting for verbose proposals is penalized."""
        agents = [MockAgent(name="alice"), MockAgent(name="bob")]
        proposals = {
            "alice": "Short.",
            "bob": "A" * 5000,  # Very verbose
        }

        votes = [
            MockVote(agent="alice", choice="bob", confidence=0.8),  # Votes for verbose
            MockVote(agent="bob", choice="alice", confidence=0.8),  # Votes for short
        ]

        config = WeightCalculatorConfig(
            enable_verbosity_normalization=True,
            verbosity_target_length=500,
            verbosity_penalty_threshold=2.0,
            verbosity_max_penalty=0.3,
        )

        calculator = WeightCalculator(config=config)
        weights = calculator.compute_weights_with_context(agents, votes, proposals)

        # Alice voted for verbose proposal - should have lower weight
        # Bob voted for short proposal - should have full weight
        assert weights["alice"] < weights["bob"]


# =============================================================================
# Judge Deliberation Integration Tests
# =============================================================================


class TestJudgeDeliberationIntegration:
    """Integration tests for judge deliberation protocol."""

    @pytest.mark.asyncio
    async def test_judge_panel_deliberation(self):
        """Test JudgePanel deliberation process."""
        judges = [
            MockAgent(name="judge1"),
            MockAgent(name="judge2"),
            MockAgent(name="judge3"),
        ]

        proposals = {
            "alice": "Alice's proposal with strong reasoning.",
            "bob": "Bob's proposal that is also good.",
        }

        # Mock generate function that returns assessments
        assessment_count = {"count": 0}

        async def mock_generate(agent, prompt, context):
            assessment_count["count"] += 1
            # Simulate deliberation changing opinions
            if "deliberation" in prompt.lower():
                return "After considering other views, I APPROVE the consensus."
            return f"Initial assessment: I recommend APPROVE. The reasoning is sound."

        panel = JudgePanel(
            judges=judges,
            strategy=JudgingStrategy.MAJORITY,
        )

        result = await panel.deliberate_and_vote(
            proposals=proposals,
            task="Evaluate the proposals",
            context=[],
            generate_fn=mock_generate,
            deliberation_rounds=2,
        )

        # Should have votes from all judges
        assert len(result.votes) == 3
        # Generate should have been called multiple times (assessments + deliberation)
        assert assessment_count["count"] > 3

    def test_judge_panel_voting_strategies(self):
        """Test different voting strategies produce different results."""
        judges = [MockAgent(name=f"judge{i}") for i in range(5)]

        # Test majority
        panel_majority = JudgePanel(judges=judges, strategy=JudgingStrategy.MAJORITY)
        panel_majority.record_vote("judge0", JudgeVote.APPROVE, 0.8, "Good")
        panel_majority.record_vote("judge1", JudgeVote.APPROVE, 0.7, "OK")
        panel_majority.record_vote("judge2", JudgeVote.APPROVE, 0.9, "Great")
        panel_majority.record_vote("judge3", JudgeVote.REJECT, 0.6, "Issues")
        panel_majority.record_vote("judge4", JudgeVote.REJECT, 0.5, "Problems")

        result_majority = panel_majority.get_result()
        assert result_majority.approved is True  # 3/5 approve

        # Test supermajority (should fail with 3/5)
        panel_super = JudgePanel(judges=judges, strategy=JudgingStrategy.SUPERMAJORITY)
        panel_super.record_vote("judge0", JudgeVote.APPROVE, 0.8, "Good")
        panel_super.record_vote("judge1", JudgeVote.APPROVE, 0.7, "OK")
        panel_super.record_vote("judge2", JudgeVote.APPROVE, 0.9, "Great")
        panel_super.record_vote("judge3", JudgeVote.REJECT, 0.6, "Issues")
        panel_super.record_vote("judge4", JudgeVote.REJECT, 0.5, "Problems")

        result_super = panel_super.get_result()
        assert result_super.approved is False  # 3/5 < 2/3


# =============================================================================
# Protocol Configuration Integration Tests
# =============================================================================


class TestProtocolBiasMitigationConfig:
    """Test that DebateProtocol correctly configures bias mitigation."""

    def test_protocol_has_all_bias_mitigation_flags(self):
        """Test DebateProtocol has all required bias mitigation flags."""
        protocol = DebateProtocol()

        # Position shuffling
        assert hasattr(protocol, 'enable_position_shuffling')
        assert hasattr(protocol, 'position_shuffling_permutations')

        # Self-vote
        assert hasattr(protocol, 'enable_self_vote_mitigation')
        assert hasattr(protocol, 'self_vote_mode')
        assert hasattr(protocol, 'self_vote_downweight')

        # Verbosity
        assert hasattr(protocol, 'enable_verbosity_normalization')
        assert hasattr(protocol, 'verbosity_target_length')

        # Judge deliberation
        assert hasattr(protocol, 'enable_judge_deliberation')
        assert hasattr(protocol, 'judge_deliberation_rounds')

        # Process evaluation
        assert hasattr(protocol, 'enable_process_evaluation')

    def test_protocol_bias_mitigation_all_disabled_by_default(self):
        """Test all bias mitigation is disabled by default for backwards compatibility."""
        protocol = DebateProtocol()

        assert protocol.enable_position_shuffling is False
        assert protocol.enable_self_vote_mitigation is False
        assert protocol.enable_verbosity_normalization is False
        assert protocol.enable_judge_deliberation is False
        assert protocol.enable_process_evaluation is False

    def test_protocol_bias_mitigation_can_be_enabled(self):
        """Test bias mitigation features can be enabled."""
        protocol = DebateProtocol(
            enable_position_shuffling=True,
            position_shuffling_permutations=5,
            enable_self_vote_mitigation=True,
            self_vote_mode="exclude",
            enable_verbosity_normalization=True,
            enable_judge_deliberation=True,
            judge_deliberation_rounds=3,
            enable_process_evaluation=True,
        )

        assert protocol.enable_position_shuffling is True
        assert protocol.position_shuffling_permutations == 5
        assert protocol.enable_self_vote_mitigation is True
        assert protocol.self_vote_mode == "exclude"
        assert protocol.enable_verbosity_normalization is True
        assert protocol.enable_judge_deliberation is True
        assert protocol.judge_deliberation_rounds == 3
        assert protocol.enable_process_evaluation is True


# =============================================================================
# Combined Bias Mitigation Integration Tests
# =============================================================================


class TestCombinedBiasMitigation:
    """Test multiple bias mitigation features working together."""

    def test_weight_calculator_combines_all_factors(self):
        """Test WeightCalculator combines self-vote and verbosity penalties."""
        agents = [MockAgent(name="alice"), MockAgent(name="bob")]
        proposals = {
            "alice": "A" * 4000,  # Verbose
            "bob": "Short proposal",
        }

        votes = [
            # Alice votes for herself (verbose proposal) - double penalty
            MockVote(agent="alice", choice="alice", confidence=0.9),
            # Bob votes for alice's verbose proposal - verbosity penalty only
            MockVote(agent="bob", choice="alice", confidence=0.8),
        ]

        config = WeightCalculatorConfig(
            enable_self_vote_mitigation=True,
            self_vote_mode="downweight",
            self_vote_downweight=0.5,
            enable_verbosity_normalization=True,
            verbosity_target_length=500,
            verbosity_penalty_threshold=2.0,
            verbosity_max_penalty=0.3,
        )

        calculator = WeightCalculator(config=config)
        weights = calculator.compute_weights_with_context(agents, votes, proposals)

        # Alice has both penalties (self-vote + verbosity)
        # Bob has only verbosity penalty
        assert weights["alice"] < weights["bob"]
        # Both should be less than 1.0
        assert weights["alice"] < 1.0
        assert weights["bob"] < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
