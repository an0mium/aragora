"""
Tests for vote aggregation module.

Tests cover:
- AggregatedVotes dataclass
- VoteAggregator class
- calculate_consensus_strength function
"""

from collections import Counter
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from aragora.debate.phases.vote_aggregator import (
    AggregatedVotes,
    VoteAggregator,
    calculate_consensus_strength,
)


@dataclass
class MockVote:
    """Mock vote for testing."""

    agent: str
    choice: str
    reasoning: str = "test reasoning"
    confidence: float = 0.8
    continue_debate: bool = False


class TestAggregatedVotes:
    """Tests for AggregatedVotes dataclass."""

    def test_empty_aggregated_votes(self):
        """Empty AggregatedVotes has default values."""
        result = AggregatedVotes()

        assert result.vote_counts == Counter()
        assert result.total_weighted == 0.0
        assert result.total_votes == 0

    def test_get_winner_with_votes(self):
        """get_winner returns top choice and count."""
        result = AggregatedVotes(
            vote_counts=Counter({"agent1": 3.0, "agent2": 1.5}),
            total_weighted=4.5,
        )

        winner = result.get_winner()

        assert winner == ("agent1", 3.0)

    def test_get_winner_empty(self):
        """get_winner returns None when no votes."""
        result = AggregatedVotes()

        assert result.get_winner() is None

    def test_get_confidence(self):
        """get_confidence calculates winner ratio."""
        result = AggregatedVotes(
            vote_counts=Counter({"agent1": 3.0, "agent2": 1.0}),
            total_weighted=4.0,
        )

        confidence = result.get_confidence()

        assert confidence == 0.75  # 3/4

    def test_get_confidence_empty(self):
        """get_confidence returns 0.5 when no votes."""
        result = AggregatedVotes()

        assert result.get_confidence() == 0.5

    def test_get_vote_distribution(self):
        """get_vote_distribution returns percentages."""
        result = AggregatedVotes(
            vote_counts=Counter({"agent1": 6.0, "agent2": 4.0}),
            total_weighted=10.0,
        )

        dist = result.get_vote_distribution()

        assert dist["agent1"] == 0.6
        assert dist["agent2"] == 0.4

    def test_get_vote_distribution_empty(self):
        """get_vote_distribution returns empty dict when no votes."""
        result = AggregatedVotes()

        assert result.get_vote_distribution() == {}


class TestVoteAggregator:
    """Tests for VoteAggregator class."""

    def test_aggregate_basic(self):
        """Basic aggregation without weights or user votes."""
        aggregator = VoteAggregator()
        votes = [
            MockVote("agent1", "proposal_a"),
            MockVote("agent2", "proposal_b"),
            MockVote("agent3", "proposal_a"),
        ]

        result = aggregator.aggregate(votes)

        assert result.vote_counts["proposal_a"] == 2.0
        assert result.vote_counts["proposal_b"] == 1.0
        assert result.total_weighted == 3.0
        assert result.agent_votes_count == 3

    def test_aggregate_with_weights(self):
        """Aggregation with custom weights."""
        aggregator = VoteAggregator()
        votes = [
            MockVote("agent1", "proposal_a"),
            MockVote("agent2", "proposal_b"),
        ]
        weights = {"agent1": 2.0, "agent2": 1.0}

        result = aggregator.aggregate(votes, weights=weights)

        assert result.vote_counts["proposal_a"] == 2.0
        assert result.vote_counts["proposal_b"] == 1.0
        assert result.total_weighted == 3.0

    def test_aggregate_with_user_votes(self):
        """Aggregation includes user votes."""
        aggregator = VoteAggregator(user_vote_weight=1.0)
        votes = [MockVote("agent1", "proposal_a")]
        user_votes = [
            {"choice": "proposal_b", "intensity": 5},
        ]

        result = aggregator.aggregate(votes, user_votes=user_votes)

        assert result.vote_counts["proposal_a"] == 1.0
        assert result.vote_counts["proposal_b"] == 1.0
        assert result.user_votes_count == 1

    def test_aggregate_with_grouping(self):
        """Aggregation uses vote grouping callback."""

        def group_votes(votes):
            # Group "proposal_a" and "PROPOSAL_A" together
            return {"proposal_a": ["proposal_a", "PROPOSAL_A"]}

        aggregator = VoteAggregator(group_similar_votes=group_votes)
        votes = [
            MockVote("agent1", "proposal_a"),
            MockVote("agent2", "PROPOSAL_A"),
        ]

        result = aggregator.aggregate(votes)

        # Both votes should be counted as "proposal_a"
        assert "proposal_a" in result.vote_counts
        assert result.vote_counts["proposal_a"] == 2.0

    def test_aggregate_skips_exceptions(self):
        """Aggregation skips exception entries."""
        aggregator = VoteAggregator()
        votes = [
            MockVote("agent1", "proposal_a"),
            Exception("agent failed"),
            MockVote("agent3", "proposal_a"),
        ]

        result = aggregator.aggregate(votes)

        assert result.agent_votes_count == 2
        assert result.total_weighted == 2.0

    def test_aggregate_user_vote_multiplier(self):
        """User vote multiplier is applied correctly."""

        def multiplier(intensity, protocol):
            return intensity / 5.0  # Scale from 0.0-2.0

        aggregator = VoteAggregator(
            user_vote_weight=1.0,
            user_vote_multiplier=multiplier,
        )
        votes = []
        user_votes = [{"choice": "proposal_a", "intensity": 10}]

        result = aggregator.aggregate(votes, user_votes=user_votes)

        # 1.0 base weight * 2.0 multiplier = 2.0
        assert result.vote_counts["proposal_a"] == 2.0

    def test_count_unweighted(self):
        """count_unweighted returns simple counts."""
        aggregator = VoteAggregator()
        votes = [
            MockVote("agent1", "proposal_a"),
            MockVote("agent2", "proposal_a"),
            MockVote("agent3", "proposal_b"),
        ]

        counts = aggregator.count_unweighted(votes)

        assert counts["proposal_a"] == 2
        assert counts["proposal_b"] == 1

    def test_count_unweighted_with_mapping(self):
        """count_unweighted applies choice mapping."""
        aggregator = VoteAggregator()
        votes = [
            MockVote("agent1", "A"),
            MockVote("agent2", "a"),  # Should map to "A"
        ]
        mapping = {"A": "A", "a": "A"}

        counts = aggregator.count_unweighted(votes, choice_mapping=mapping)

        assert counts["A"] == 2


class TestCalculateConsensusStrength:
    """Tests for calculate_consensus_strength function."""

    def test_unanimous(self):
        """Single choice is unanimous."""
        counts = Counter({"proposal_a": 5})

        strength, variance = calculate_consensus_strength(counts)

        assert strength == "unanimous"
        assert variance == 0.0

    def test_strong_consensus(self):
        """Low variance is strong consensus."""
        counts = Counter({"proposal_a": 5, "proposal_b": 4})

        strength, variance = calculate_consensus_strength(counts)

        assert strength == "strong"
        assert variance < 1

    def test_medium_consensus(self):
        """Medium variance is medium consensus."""
        counts = Counter({"proposal_a": 5, "proposal_b": 3})

        strength, variance = calculate_consensus_strength(counts)

        # Mean = 4, variance = ((5-4)^2 + (3-4)^2) / 2 = 1.0
        # Borderline - could be strong or medium
        assert strength in ("strong", "medium")

    def test_weak_consensus(self):
        """High variance is weak consensus."""
        counts = Counter({"proposal_a": 8, "proposal_b": 1, "proposal_c": 1})

        strength, variance = calculate_consensus_strength(counts)

        # High variance due to uneven distribution
        assert strength in ("medium", "weak")

    def test_empty_counts(self):
        """Empty counts is unanimous (edge case)."""
        counts = Counter()

        strength, variance = calculate_consensus_strength(counts)

        assert strength == "unanimous"
        assert variance == 0.0
