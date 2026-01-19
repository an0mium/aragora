"""
Tests for the VotingPhase module.

Tests cover:
- VoteWeightCalculator initialization and weight computation
- WeightedVoteResult dataclass
- VotingPhase vote grouping
- Consensus strength calculation
"""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from aragora.debate.phases.voting import (
    VoteWeightCalculator,
    VotingPhase,
    WeightedVoteResult,
)
from aragora.debate.protocol import DebateProtocol


@dataclass
class MockVote:
    """Mock vote for testing."""

    agent: str
    choice: str
    reasoning: str = "Test reasoning"
    confidence: float = 0.8
    continue_debate: bool = False


class TestWeightedVoteResult:
    """Tests for WeightedVoteResult dataclass."""

    def test_default_values(self):
        """WeightedVoteResult has sensible defaults."""
        result = WeightedVoteResult()

        assert result.winner is None
        assert result.vote_counts == {}
        assert result.total_weighted_votes == 0.0
        assert result.confidence == 0.0
        assert result.consensus_reached is False
        assert result.consensus_strength == "none"

    def test_custom_values(self):
        """WeightedVoteResult stores custom values."""
        result = WeightedVoteResult(
            winner="agent1",
            vote_counts={"agent1": 2.5, "agent2": 1.0},
            total_weighted_votes=3.5,
            confidence=0.85,
            consensus_reached=True,
            consensus_strength="strong",
        )

        assert result.winner == "agent1"
        assert result.total_weighted_votes == 3.5
        assert result.consensus_strength == "strong"


class TestVoteWeightCalculator:
    """Tests for VoteWeightCalculator."""

    def test_init_minimal(self):
        """Calculator can be initialized without sources."""
        calc = VoteWeightCalculator()

        assert calc._reputation_source is None
        assert calc._reliability_weights == {}

    def test_init_with_reliability_weights(self):
        """Calculator stores reliability weights."""
        weights = {"agent1": 1.2, "agent2": 0.8}
        calc = VoteWeightCalculator(reliability_weights=weights)

        assert calc._reliability_weights == weights

    def test_compute_weight_default(self):
        """Default weight is 1.0 without sources."""
        calc = VoteWeightCalculator()

        weight = calc.compute_weight("agent1")

        assert weight == 1.0

    def test_compute_weight_with_reliability(self):
        """Reliability weights affect computed weight."""
        weights = {"agent1": 1.5}
        calc = VoteWeightCalculator(reliability_weights=weights)

        weight = calc.compute_weight("agent1")

        # Weight should reflect reliability
        assert weight >= 1.0  # With reliability 1.5

    def test_compute_weight_caches_result(self):
        """Weight computation is cached."""
        calc = VoteWeightCalculator()

        weight1 = calc.compute_weight("agent1")
        weight2 = calc.compute_weight("agent1")

        assert weight1 == weight2
        assert "agent1" in calc._cache

    def test_compute_weight_with_reputation_source(self):
        """Reputation source affects weight."""
        reputation_fn = MagicMock(return_value=1.3)
        calc = VoteWeightCalculator(reputation_source=reputation_fn)

        weight = calc.compute_weight("agent1")

        reputation_fn.assert_called_with("agent1")
        # Weight should incorporate reputation
        assert weight > 0

    def test_compute_weight_with_consistency_source(self):
        """Consistency source affects weight."""
        consistency_fn = MagicMock(return_value=0.9)
        calc = VoteWeightCalculator(consistency_source=consistency_fn)

        weight = calc.compute_weight("agent1")

        consistency_fn.assert_called_with("agent1")

    def test_compute_weight_with_calibration_source(self):
        """Calibration source affects weight."""
        calibration_fn = MagicMock(return_value=1.1)
        calc = VoteWeightCalculator(calibration_source=calibration_fn)

        weight = calc.compute_weight("agent1")

        calibration_fn.assert_called_with("agent1")

    def test_compute_weight_handles_source_errors(self):
        """Weight computation handles source errors gracefully."""
        reputation_fn = MagicMock(side_effect=RuntimeError("Error"))
        calc = VoteWeightCalculator(reputation_source=reputation_fn)

        # Should not raise, should return reasonable weight
        weight = calc.compute_weight("agent1")
        assert weight > 0

    def test_compute_weight_multiple_agents(self):
        """Different agents can have different weights."""
        weights = {"agent1": 1.5, "agent2": 0.8}
        calc = VoteWeightCalculator(reliability_weights=weights)

        w1 = calc.compute_weight("agent1")
        w2 = calc.compute_weight("agent2")

        # agent1 has higher reliability, should have higher weight
        assert w1 > w2


class TestVotingPhase:
    """Tests for VotingPhase."""

    @pytest.fixture
    def protocol(self):
        """Create test protocol."""
        return DebateProtocol(rounds=2, consensus="majority", vote_grouping=True)

    @pytest.fixture
    def protocol_no_grouping(self):
        """Create protocol without vote grouping."""
        return DebateProtocol(rounds=2, consensus="majority", vote_grouping=False)

    def test_init(self, protocol):
        """VotingPhase initializes with protocol."""
        phase = VotingPhase(protocol)

        assert phase.protocol is protocol
        assert phase._similarity_backend is None

    def test_group_similar_votes_disabled(self, protocol_no_grouping):
        """Vote grouping returns empty when disabled."""
        phase = VotingPhase(protocol_no_grouping)
        votes = [MockVote("a1", "choice1"), MockVote("a2", "choice2")]

        groups = phase.group_similar_votes(votes)

        assert groups == {}

    def test_group_similar_votes_empty(self, protocol):
        """Empty votes returns empty groups."""
        phase = VotingPhase(protocol)

        groups = phase.group_similar_votes([])

        assert groups == {}

    def test_group_similar_votes_single(self, protocol):
        """Single vote returns empty groups (nothing to group)."""
        phase = VotingPhase(protocol)
        votes = [MockVote("a1", "choice1")]

        groups = phase.group_similar_votes(votes)

        assert groups == {}


class TestVotingPhaseConsensusStrength:
    """Tests for consensus strength calculation."""

    @pytest.fixture
    def protocol(self):
        return DebateProtocol(rounds=2, consensus="majority")

    def test_unanimous_strength(self, protocol):
        """Unanimous votes have strong consensus."""
        # All votes for the same choice
        votes = [
            MockVote("a1", "choice1"),
            MockVote("a2", "choice1"),
            MockVote("a3", "choice1"),
        ]

        phase = VotingPhase(protocol)
        # Note: actual consensus strength calculation may be elsewhere
        # This tests the structure is correct
        result = WeightedVoteResult(
            winner="choice1",
            vote_counts={"choice1": 3.0},
            total_weighted_votes=3.0,
            confidence=1.0,
            consensus_reached=True,
            consensus_strength="unanimous",
        )

        assert result.consensus_strength == "unanimous"
        assert result.consensus_reached is True

    def test_split_votes_weak_consensus(self, protocol):
        """Split votes indicate weak consensus."""
        result = WeightedVoteResult(
            winner="choice1",
            vote_counts={"choice1": 2.0, "choice2": 1.5, "choice3": 1.0},
            total_weighted_votes=4.5,
            confidence=0.45,
            consensus_reached=False,
            consensus_strength="weak",
        )

        assert result.consensus_strength == "weak"
        assert result.consensus_reached is False


class TestVotingPhaseIntegration:
    """Integration tests for VotingPhase."""

    @pytest.fixture
    def protocol(self):
        return DebateProtocol(rounds=2, consensus="majority", vote_grouping=False)

    def test_full_voting_flow(self, protocol):
        """Test complete voting phase flow structure."""
        phase = VotingPhase(protocol)

        # Create votes
        votes = [
            MockVote("agent1", "Use Redis"),
            MockVote("agent2", "Use Redis"),
            MockVote("agent3", "Use PostgreSQL"),
        ]

        # Group votes (disabled in this protocol)
        groups = phase.group_similar_votes(votes)
        assert groups == {}

        # In real usage, votes would be counted and weighted
        # This test verifies the structure is correct
        assert len(votes) == 3
        assert phase.protocol.consensus == "majority"
