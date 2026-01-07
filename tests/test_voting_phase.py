"""Tests for voting phase functionality."""

import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass

from aragora.debate.phases.voting import (
    VotingPhase,
    VoteWeightCalculator,
    WeightedVoteResult,
)


# Mock Vote dataclass to avoid full imports
@dataclass
class MockVote:
    agent: str
    choice: str
    reasoning: str = ""
    confidence: float = 1.0


@pytest.fixture
def mock_protocol():
    """Create mock protocol with voting settings."""
    protocol = MagicMock()
    protocol.vote_grouping = True
    protocol.vote_grouping_threshold = 0.8
    protocol.consensus_threshold = 0.5
    return protocol


@pytest.fixture
def voting_phase(mock_protocol):
    """Create voting phase instance."""
    return VotingPhase(protocol=mock_protocol)


class TestVoteWeightCalculator:
    """Tests for VoteWeightCalculator."""

    def test_default_weight_is_one(self):
        """Without any sources, weight should be 1.0."""
        calc = VoteWeightCalculator()
        assert calc.compute_weight("agent1") == 1.0

    def test_reputation_source_applied(self):
        """Reputation source weight is multiplied."""
        calc = VoteWeightCalculator(
            reputation_source=lambda name: 1.5 if name == "trusted" else 0.5
        )
        assert calc.compute_weight("trusted") == 1.5
        assert calc.compute_weight("untrusted") == 0.5

    def test_reliability_weights_applied(self):
        """Pre-computed reliability weights are applied."""
        calc = VoteWeightCalculator(
            reliability_weights={"agent1": 0.8, "agent2": 1.0}
        )
        assert calc.compute_weight("agent1") == 0.8
        assert calc.compute_weight("agent2") == 1.0
        assert calc.compute_weight("unknown") == 1.0  # Default

    def test_consistency_source_maps_to_weight(self):
        """Consistency score (0-1) maps to 0.5-1.0 weight."""
        calc = VoteWeightCalculator(
            consistency_source=lambda name: 1.0 if name == "consistent" else 0.0
        )
        # Score 1.0 -> weight 0.5 + 0.5 = 1.0
        assert calc.compute_weight("consistent") == 1.0
        # Score 0.0 -> weight 0.5 + 0.0 = 0.5
        assert calc.compute_weight("inconsistent") == 0.5

    def test_calibration_source_applied(self):
        """Calibration weight is multiplied."""
        calc = VoteWeightCalculator(
            calibration_source=lambda name: 1.2
        )
        assert calc.compute_weight("agent1") == 1.2

    def test_all_weights_multiply(self):
        """All weight sources multiply together."""
        calc = VoteWeightCalculator(
            reputation_source=lambda _: 1.2,  # 1.2
            reliability_weights={"agent1": 0.9},  # * 0.9
            consistency_source=lambda _: 0.8,  # * (0.5 + 0.4) = 0.9
            calibration_source=lambda _: 1.1,  # * 1.1
        )
        expected = 1.2 * 0.9 * 0.9 * 1.1
        assert abs(calc.compute_weight("agent1") - expected) < 0.001

    def test_weight_caching(self):
        """Weights are cached after first computation."""
        call_count = [0]

        def reputation_source(name):
            call_count[0] += 1
            return 1.5

        calc = VoteWeightCalculator(reputation_source=reputation_source)
        calc.compute_weight("agent1")
        calc.compute_weight("agent1")
        calc.compute_weight("agent1")
        assert call_count[0] == 1  # Only called once

    def test_clear_cache(self):
        """Cache can be cleared."""
        call_count = [0]

        def reputation_source(name):
            call_count[0] += 1
            return 1.5

        calc = VoteWeightCalculator(reputation_source=reputation_source)
        calc.compute_weight("agent1")
        calc.clear_cache()
        calc.compute_weight("agent1")
        assert call_count[0] == 2  # Called twice after cache clear

    def test_compute_weights_batch(self):
        """Batch computation returns dict of weights."""
        calc = VoteWeightCalculator(
            reliability_weights={"a": 0.8, "b": 0.9}
        )
        weights = calc.compute_weights_batch(["a", "b", "c"])
        assert weights == {"a": 0.8, "b": 0.9, "c": 1.0}

    def test_error_in_source_logs_and_continues(self):
        """Errors in weight sources are logged, weight defaults to multiplier of 1."""
        def bad_source(name):
            raise ValueError("Test error")

        calc = VoteWeightCalculator(
            reputation_source=bad_source,
            reliability_weights={"agent1": 0.8}
        )
        # Should still return the reliability weight despite reputation error
        assert calc.compute_weight("agent1") == 0.8


class TestVotingPhaseGroupSimilarVotes:
    """Tests for semantic vote grouping."""

    def test_no_grouping_when_disabled(self, voting_phase):
        """Returns empty when vote_grouping is disabled."""
        voting_phase.protocol.vote_grouping = False
        votes = [MockVote("a", "choice1"), MockVote("b", "choice2")]
        result = voting_phase.group_similar_votes(votes)
        assert result == {}

    def test_no_grouping_for_single_choice(self, voting_phase):
        """No grouping needed with single unique choice."""
        votes = [MockVote("a", "same"), MockVote("b", "same")]
        result = voting_phase.group_similar_votes(votes)
        assert result == {}

    def test_groups_similar_choices(self, voting_phase):
        """Similar choices are grouped together."""
        mock_backend = MagicMock()
        mock_backend.compute_similarity.return_value = 0.9  # Above threshold
        voting_phase._similarity_backend = mock_backend

        votes = [
            MockVote("a", "Vector DB"),
            MockVote("b", "Use vector database"),
        ]
        result = voting_phase.group_similar_votes(votes)

        # One of them should be canonical
        assert len(result) == 1
        canonical = list(result.keys())[0]
        assert len(result[canonical]) == 2

    def test_dissimilar_choices_not_grouped(self, voting_phase):
        """Dissimilar choices remain separate."""
        mock_backend = MagicMock()
        mock_backend.compute_similarity.return_value = 0.3  # Below threshold
        voting_phase._similarity_backend = mock_backend

        votes = [
            MockVote("a", "Option A"),
            MockVote("b", "Option B"),
        ]
        result = voting_phase.group_similar_votes(votes)
        assert result == {}  # No groups with multiple members


class TestVotingPhaseCountWeightedVotes:
    """Tests for weighted vote counting."""

    def test_basic_counting_without_weights(self, voting_phase):
        """Basic vote counting without weight calculator."""
        voting_phase.protocol.vote_grouping = False
        votes = [
            MockVote("a", "choice1"),
            MockVote("b", "choice1"),
            MockVote("c", "choice2"),
        ]
        result = voting_phase.count_weighted_votes(votes)

        assert result.winner == "choice1"
        assert result.vote_counts["choice1"] == 2.0
        assert result.vote_counts["choice2"] == 1.0
        assert result.total_weighted_votes == 3.0
        assert abs(result.confidence - 2/3) < 0.001

    def test_weighted_counting(self, voting_phase):
        """Votes are weighted by calculator."""
        voting_phase.protocol.vote_grouping = False
        votes = [
            MockVote("trusted", "choice1"),
            MockVote("untrusted", "choice2"),
        ]
        calc = VoteWeightCalculator(
            reliability_weights={"trusted": 2.0, "untrusted": 0.5}
        )
        result = voting_phase.count_weighted_votes(votes, weight_calculator=calc)

        assert result.winner == "choice1"
        assert result.vote_counts["choice1"] == 2.0
        assert result.vote_counts["choice2"] == 0.5

    def test_user_votes_included(self, voting_phase):
        """User votes are added with base weight."""
        voting_phase.protocol.vote_grouping = False
        votes = [MockVote("a", "choice1")]
        user_votes = [{"choice": "choice2", "user_id": "user1"}]

        result = voting_phase.count_weighted_votes(
            votes, user_votes=user_votes, user_vote_weight=0.5
        )

        assert result.vote_counts["choice1"] == 1.0
        assert result.vote_counts["choice2"] == 0.5

    def test_user_vote_intensity_multiplier(self, voting_phase):
        """User vote intensity is applied via multiplier function."""
        voting_phase.protocol.vote_grouping = False
        votes = [MockVote("a", "choice1")]
        user_votes = [{"choice": "choice2", "intensity": 10}]

        def intensity_multiplier(intensity, protocol):
            return intensity / 5  # intensity 10 -> 2x multiplier

        result = voting_phase.count_weighted_votes(
            votes,
            user_votes=user_votes,
            user_vote_weight=0.5,
            user_vote_multiplier=intensity_multiplier,
        )

        # 0.5 base * 2.0 multiplier = 1.0
        assert result.vote_counts["choice2"] == 1.0

    def test_empty_votes_returns_empty_result(self, voting_phase):
        """Empty vote list returns empty result."""
        result = voting_phase.count_weighted_votes([])
        assert result.winner is None
        assert result.vote_counts == {}
        assert result.total_weighted_votes == 0.0


class TestVotingPhaseConsensusStrength:
    """Tests for consensus strength calculation."""

    def test_unanimous_strength(self, voting_phase):
        """Single choice is unanimous."""
        result = voting_phase.compute_consensus_strength({"choice1": 3}, 3)
        assert result["strength"] == "unanimous"
        assert result["variance"] == 0.0

    def test_strong_consensus(self, voting_phase):
        """Low variance is strong consensus."""
        # 2.5 vs 2.4 - very close, variance < 1
        result = voting_phase.compute_consensus_strength({"a": 2.5, "b": 2.4}, 4.9)
        assert result["strength"] == "strong"
        assert result["variance"] < 1

    def test_weak_consensus(self, voting_phase):
        """High variance is weak consensus."""
        # 5 vs 1 - big difference
        result = voting_phase.compute_consensus_strength({"a": 5, "b": 1}, 6)
        assert result["strength"] == "weak"
        assert result["variance"] >= 2

    def test_empty_votes_is_none(self, voting_phase):
        """Empty vote counts returns none strength."""
        result = voting_phase.compute_consensus_strength({}, 0)
        assert result["strength"] == "none"


class TestVotingPhaseCheckUnanimous:
    """Tests for unanimous consensus checking."""

    def test_unanimous_when_all_agree(self, voting_phase):
        """Unanimous when all votes match."""
        voting_phase.protocol.vote_grouping = False
        votes = [
            MockVote("a", "choice1"),
            MockVote("b", "choice1"),
            MockVote("c", "choice1"),
        ]
        result = voting_phase.check_unanimous(votes)

        assert result.consensus_reached is True
        assert result.consensus_strength == "unanimous"
        assert result.winner == "choice1"
        assert result.confidence == 1.0

    def test_not_unanimous_with_dissent(self, voting_phase):
        """Not unanimous when any vote differs."""
        voting_phase.protocol.vote_grouping = False
        votes = [
            MockVote("a", "choice1"),
            MockVote("b", "choice1"),
            MockVote("c", "choice2"),
        ]
        result = voting_phase.check_unanimous(votes)

        assert result.consensus_reached is False
        assert result.consensus_strength == "none"

    def test_voting_errors_count_as_dissent(self, voting_phase):
        """Voting errors prevent unanimity."""
        voting_phase.protocol.vote_grouping = False
        votes = [
            MockVote("a", "choice1"),
            MockVote("b", "choice1"),
        ]
        result = voting_phase.check_unanimous(votes, voting_errors=1)

        assert result.consensus_reached is False
        # 2 votes for choice1 out of 3 total (including 1 error)
        assert abs(result.confidence - 2/3) < 0.001


class TestWeightedVoteResult:
    """Tests for WeightedVoteResult dataclass."""

    def test_default_values(self):
        """Default values are sensible."""
        result = WeightedVoteResult()
        assert result.winner is None
        assert result.vote_counts == {}
        assert result.total_weighted_votes == 0.0
        assert result.confidence == 0.0
        assert result.consensus_reached is False
        assert result.consensus_strength == "none"
        assert result.consensus_variance == 0.0
        assert result.choice_mapping == {}
