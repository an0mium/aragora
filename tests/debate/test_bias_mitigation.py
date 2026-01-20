"""
Tests for Agent-as-a-Judge bias mitigation module.

Tests cover:
- Position bias mitigation (shuffling, permutations, averaging)
- Self-enhancement bias detection
- Verbosity bias normalization
- Process-based evaluation rubrics
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from dataclasses import dataclass

from aragora.debate.bias_mitigation import (
    # Position bias
    PositionBiasConfig,
    shuffle_proposals,
    generate_permutations,
    average_permutation_votes,
    # Self-vote bias
    SelfVoteConfig,
    detect_self_vote,
    apply_self_vote_penalty,
    # Verbosity bias
    VerbosityBiasConfig,
    calculate_verbosity_factor,
    get_verbosity_weights,
    # Process evaluation
    ProcessEvaluator,
    EvaluationCriterion,
    ProcessEvaluationResult,
    # Config
    BiasMitigationConfig,
)


# =============================================================================
# Position Bias Tests
# =============================================================================


class TestPositionBiasMitigation:
    """Tests for position bias mitigation utilities."""

    def test_shuffle_proposals_basic(self):
        """Test basic proposal shuffling."""
        proposals = {"alice": "Proposal A", "bob": "Proposal B", "carol": "Proposal C"}

        shuffled = shuffle_proposals(proposals, seed=42)

        # Should have same keys and values
        assert set(shuffled.keys()) == set(proposals.keys())
        assert set(shuffled.values()) == set(proposals.values())

    def test_shuffle_proposals_reproducible(self):
        """Test that same seed produces same shuffle."""
        proposals = {"alice": "Proposal A", "bob": "Proposal B", "carol": "Proposal C"}

        shuffled1 = shuffle_proposals(proposals, seed=42)
        shuffled2 = shuffle_proposals(proposals, seed=42)

        assert list(shuffled1.keys()) == list(shuffled2.keys())

    def test_shuffle_proposals_different_seeds(self):
        """Test that different seeds can produce different shuffles."""
        proposals = {f"agent_{i}": f"Proposal {i}" for i in range(10)}

        shuffled1 = shuffle_proposals(proposals, seed=1)
        shuffled2 = shuffle_proposals(proposals, seed=2)

        # With 10 items, different seeds should likely produce different orderings
        # (not guaranteed but extremely likely)
        assert list(shuffled1.keys()) != list(shuffled2.keys())

    def test_generate_permutations_count(self):
        """Test that correct number of permutations is generated."""
        proposals = {"alice": "A", "bob": "B", "carol": "C"}

        perms = generate_permutations(proposals, num_permutations=5)

        assert len(perms) == 5

    def test_generate_permutations_uniqueness(self):
        """Test that permutations are different."""
        proposals = {f"agent_{i}": f"Proposal {i}" for i in range(10)}

        perms = generate_permutations(proposals, num_permutations=3)

        # Convert to tuples for comparison
        orderings = [tuple(p.keys()) for p in perms]

        # At least some should be different (with high probability)
        assert len(set(orderings)) > 1

    def test_average_permutation_votes_basic(self):
        """Test basic vote averaging across permutations."""
        # Create mock votes
        @dataclass
        class MockVote:
            agent: str
            choice: str
            confidence: float
            reasoning: str = ""
            continue_debate: bool = False

        proposals = {"alice": "A", "bob": "B"}

        # Agent voted for different choices across permutations
        votes_by_agent = {
            "judge1": [
                MockVote(agent="judge1", choice="alice", confidence=0.8),
                MockVote(agent="judge1", choice="alice", confidence=0.9),
                MockVote(agent="judge1", choice="bob", confidence=0.7),
            ],
        }

        averaged = average_permutation_votes(votes_by_agent, proposals)

        # Should have one averaged vote per agent
        assert len(averaged) == 1
        vote = averaged[0]
        assert vote.agent == "judge1"
        # Confidence should be average
        assert 0.7 < vote.confidence < 0.9


# =============================================================================
# Self-Vote Bias Tests
# =============================================================================


class TestSelfVoteBiasDetection:
    """Tests for self-enhancement bias detection."""

    def test_detect_self_vote_true(self):
        """Test detection when agent votes for own proposal."""
        @dataclass
        class MockVote:
            agent: str
            choice: str
            confidence: float = 0.8
            reasoning: str = ""

        vote = MockVote(agent="alice", choice="alice")
        proposals = {"alice": "Alice's proposal", "bob": "Bob's proposal"}

        assert detect_self_vote(vote, proposals) is True

    def test_detect_self_vote_false(self):
        """Test detection when agent votes for different proposal."""
        @dataclass
        class MockVote:
            agent: str
            choice: str
            confidence: float = 0.8
            reasoning: str = ""

        vote = MockVote(agent="alice", choice="bob")
        proposals = {"alice": "Alice's proposal", "bob": "Bob's proposal"}

        assert detect_self_vote(vote, proposals) is False

    def test_apply_self_vote_penalty_exclude(self):
        """Test exclude mode for self-vote penalty."""
        @dataclass
        class MockVote:
            agent: str
            choice: str
            confidence: float = 0.8
            reasoning: str = ""

        votes = [
            MockVote(agent="alice", choice="alice"),  # Self-vote
            MockVote(agent="bob", choice="alice"),    # Not self-vote
        ]
        proposals = {"alice": "A", "bob": "B"}
        weights = {"alice": 1.0, "bob": 1.0}
        config = SelfVoteConfig(enabled=True, mode="exclude")

        adjusted = apply_self_vote_penalty(weights.copy(), votes, proposals, config)

        assert adjusted["alice"] == 0.0  # Excluded
        assert adjusted["bob"] == 1.0    # Unchanged

    def test_apply_self_vote_penalty_downweight(self):
        """Test downweight mode for self-vote penalty."""
        @dataclass
        class MockVote:
            agent: str
            choice: str
            confidence: float = 0.8
            reasoning: str = ""

        votes = [MockVote(agent="alice", choice="alice")]
        proposals = {"alice": "A", "bob": "B"}
        weights = {"alice": 1.0}
        config = SelfVoteConfig(enabled=True, mode="downweight", downweight_factor=0.5)

        adjusted = apply_self_vote_penalty(weights.copy(), votes, proposals, config)

        assert adjusted["alice"] == 0.5  # Downweighted by factor


# =============================================================================
# Verbosity Bias Tests
# =============================================================================


class TestVerbosityBiasNormalization:
    """Tests for verbosity bias normalization."""

    def test_calculate_verbosity_factor_normal_length(self):
        """Test that normal-length proposals get factor of 1.0."""
        config = VerbosityBiasConfig(
            enabled=True,
            target_length=1000,
            penalty_threshold=3.0,
            max_penalty=0.3,
        )

        # At target length
        factor = calculate_verbosity_factor(1000, config)
        assert factor == 1.0

        # Below target
        factor = calculate_verbosity_factor(500, config)
        assert factor == 1.0

    def test_calculate_verbosity_factor_excessive_length(self):
        """Test that excessive length proposals get penalty."""
        config = VerbosityBiasConfig(
            enabled=True,
            target_length=1000,
            penalty_threshold=3.0,
            max_penalty=0.3,
        )

        # 4x target length (exceeds 3.0 threshold)
        factor = calculate_verbosity_factor(4000, config)
        assert factor < 1.0
        assert factor >= 0.7  # Max penalty is 0.3, so min factor is 0.7

    def test_calculate_verbosity_factor_disabled(self):
        """Test that disabled config returns 1.0."""
        config = VerbosityBiasConfig(enabled=False)

        factor = calculate_verbosity_factor(10000, config)
        assert factor == 1.0

    def test_get_verbosity_weights(self):
        """Test getting verbosity weights for multiple proposals."""
        proposals = {
            "short": "Brief.",
            "normal": "A" * 1000,
            "verbose": "A" * 5000,
        }
        config = VerbosityBiasConfig(
            enabled=True,
            target_length=1000,
            penalty_threshold=2.0,
            max_penalty=0.3,
        )

        weights = get_verbosity_weights(proposals, config)

        assert weights["short"] == 1.0
        assert weights["normal"] == 1.0
        assert weights["verbose"] < 1.0


# =============================================================================
# Process Evaluation Tests
# =============================================================================


class TestProcessEvaluation:
    """Tests for process-based evaluation rubrics."""

    @pytest.mark.asyncio
    async def test_process_evaluator_basic(self):
        """Test basic process evaluation."""
        evaluator = ProcessEvaluator()

        proposal = """
        Based on the evidence (EVID-123), I believe we should implement caching.

        This is because:
        1. Performance data shows 500ms latency
        2. Caching can reduce this to 50ms

        However, there are counterarguments:
        - Increased complexity
        - Cache invalidation challenges

        I'm 80% confident in this recommendation, with uncertainty around
        the cache invalidation strategy.
        """

        result = await evaluator.evaluate_proposal(
            agent_name="test_agent",
            proposal=proposal,
            task="How should we improve API performance?",
        )

        assert isinstance(result, ProcessEvaluationResult)
        assert 0.0 <= result.overall_score <= 1.0
        assert len(result.criteria_scores) > 0

    @pytest.mark.asyncio
    async def test_process_evaluator_custom_criteria(self):
        """Test process evaluation with custom criteria."""
        custom_criteria = [
            EvaluationCriterion(
                name="creativity",
                description="Novel approach",
                weight=2.0,
            ),
            EvaluationCriterion(
                name="feasibility",
                description="Can be implemented",
                weight=1.0,
            ),
        ]

        evaluator = ProcessEvaluator(criteria=custom_criteria)

        result = await evaluator.evaluate_proposal(
            agent_name="test_agent",
            proposal="A creative but feasible solution.",
            task="Design a new feature",
        )

        assert len(result.criteria_scores) == 2

    @pytest.mark.asyncio
    async def test_process_evaluator_evidence_bonus(self):
        """Test that evidence citations boost score."""
        evaluator = ProcessEvaluator()

        # Mock evidence pack
        @dataclass
        class MockSnippet:
            id: str
            text: str

        @dataclass
        class MockEvidencePack:
            snippets: list

        evidence = MockEvidencePack(snippets=[
            MockSnippet(id="123", text="Evidence 1"),
            MockSnippet(id="456", text="Evidence 2"),
        ])

        proposal_with_evidence = """
        Based on EVID-123 and EVID-456, we should proceed.
        The evidence clearly shows the benefits.
        """

        proposal_without_evidence = """
        We should proceed with the implementation.
        I think it will be beneficial.
        """

        result_with = await evaluator.evaluate_proposal(
            agent_name="test",
            proposal=proposal_with_evidence,
            task="Should we proceed?",
            evidence_pack=evidence,
        )

        result_without = await evaluator.evaluate_proposal(
            agent_name="test",
            proposal=proposal_without_evidence,
            task="Should we proceed?",
            evidence_pack=evidence,
        )

        # Score with evidence citations should be higher
        assert result_with.overall_score >= result_without.overall_score


# =============================================================================
# Configuration Tests
# =============================================================================


class TestBiasMitigationConfig:
    """Tests for unified bias mitigation configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = BiasMitigationConfig()

        assert config.enable_position_shuffling is False
        assert config.enable_self_vote_mitigation is False
        assert config.enable_verbosity_normalization is False
        assert config.enable_process_evaluation is False

    def test_config_all_enabled(self):
        """Test configuration with all features enabled."""
        config = BiasMitigationConfig(
            enable_position_shuffling=True,
            position_shuffling_permutations=5,
            enable_self_vote_mitigation=True,
            self_vote_mode="exclude",
            enable_verbosity_normalization=True,
            verbosity_target_length=500,
            enable_process_evaluation=True,
        )

        assert config.enable_position_shuffling is True
        assert config.position_shuffling_permutations == 5
        assert config.enable_self_vote_mitigation is True
        assert config.self_vote_mode == "exclude"
        assert config.enable_verbosity_normalization is True
        assert config.verbosity_target_length == 500
        assert config.enable_process_evaluation is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestBiasMitigationIntegration:
    """Integration tests combining multiple bias mitigation features."""

    @pytest.mark.asyncio
    async def test_combined_bias_mitigation(self):
        """Test using multiple bias mitigation features together."""
        # This tests the workflow of:
        # 1. Shuffling proposals
        # 2. Detecting self-votes
        # 3. Applying verbosity penalties

        @dataclass
        class MockVote:
            agent: str
            choice: str
            confidence: float
            reasoning: str = ""

        proposals = {
            "alice": "Short proposal",
            "bob": "A" * 3000,  # Verbose
            "carol": "Medium length proposal with good content",
        }

        # Generate permutations
        perms = generate_permutations(proposals, num_permutations=3)
        assert len(perms) == 3

        # Simulate votes
        votes = [
            MockVote(agent="alice", choice="alice", confidence=0.9),  # Self-vote
            MockVote(agent="bob", choice="alice", confidence=0.8),
            MockVote(agent="carol", choice="bob", confidence=0.7),
        ]

        # Apply self-vote penalty
        weights = {"alice": 1.0, "bob": 1.0, "carol": 1.0}
        config = SelfVoteConfig(enabled=True, mode="downweight", downweight_factor=0.5)
        weights = apply_self_vote_penalty(weights, votes, proposals, config)

        assert weights["alice"] == 0.5  # Penalized for self-vote
        assert weights["bob"] == 1.0
        assert weights["carol"] == 1.0

        # Apply verbosity weights
        verbosity_config = VerbosityBiasConfig(
            enabled=True,
            target_length=1000,
            penalty_threshold=2.0,
            max_penalty=0.3,
        )
        verbosity_weights = get_verbosity_weights(proposals, verbosity_config)

        assert verbosity_weights["alice"] == 1.0  # Short is fine
        assert verbosity_weights["bob"] < 1.0     # Verbose gets penalty
        assert verbosity_weights["carol"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
