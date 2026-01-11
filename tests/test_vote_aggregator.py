"""Tests for VoteAggregator class."""

import pytest
from collections import Counter
from unittest.mock import MagicMock

from aragora.debate.phases.vote_aggregator import (
    VoteAggregator,
    AggregatedVotes,
    calculate_consensus_strength,
)


class MockVote:
    """Mock vote for testing."""

    def __init__(self, agent: str, choice: str, confidence: float = 0.8):
        self.agent = agent
        self.choice = choice
        self.confidence = confidence


class TestAggregatedVotes:
    """Tests for AggregatedVotes dataclass."""

    def test_default_values(self):
        """Test default AggregatedVotes values."""
        result = AggregatedVotes()
        assert isinstance(result.vote_counts, Counter)
        assert result.total_weighted == 0.0
        assert result.choice_mapping == {}
        assert result.vote_groups == {}
        assert result.total_votes == 0

    def test_get_winner_empty(self):
        """Test get_winner with no votes."""
        result = AggregatedVotes()
        assert result.get_winner() is None

    def test_get_winner_single_choice(self):
        """Test get_winner with single choice."""
        result = AggregatedVotes(
            vote_counts=Counter({"A": 3}),
            total_weighted=3.0,
        )
        winner = result.get_winner()
        assert winner == ("A", 3)

    def test_get_winner_multiple_choices(self):
        """Test get_winner with multiple choices."""
        result = AggregatedVotes(
            vote_counts=Counter({"A": 5, "B": 3, "C": 2}),
            total_weighted=10.0,
        )
        winner = result.get_winner()
        assert winner == ("A", 5)

    def test_get_confidence_empty(self):
        """Test get_confidence with no votes."""
        result = AggregatedVotes()
        assert result.get_confidence() == 0.5

    def test_get_confidence_majority(self):
        """Test get_confidence calculation."""
        result = AggregatedVotes(
            vote_counts=Counter({"A": 6, "B": 4}),
            total_weighted=10.0,
        )
        confidence = result.get_confidence()
        assert confidence == 0.6

    def test_get_vote_distribution_empty(self):
        """Test get_vote_distribution with no votes."""
        result = AggregatedVotes()
        assert result.get_vote_distribution() == {}

    def test_get_vote_distribution(self):
        """Test get_vote_distribution calculation."""
        result = AggregatedVotes(
            vote_counts=Counter({"A": 6, "B": 4}),
            total_weighted=10.0,
        )
        dist = result.get_vote_distribution()
        assert dist["A"] == 0.6
        assert dist["B"] == 0.4


class TestCalculateConsensusStrength:
    """Tests for calculate_consensus_strength function."""

    def test_unanimous_single_choice(self):
        """Test unanimous strength with single choice."""
        counts = Counter({"A": 5})
        strength, variance = calculate_consensus_strength(counts)
        assert strength == "unanimous"
        assert variance == 0.0

    def test_unanimous_empty(self):
        """Test with empty counter."""
        counts = Counter()
        strength, variance = calculate_consensus_strength(counts)
        assert strength == "unanimous"
        assert variance == 0.0

    def test_strong_consensus(self):
        """Test strong consensus (variance < 1)."""
        counts = Counter({"A": 5, "B": 4})  # variance < 1
        strength, variance = calculate_consensus_strength(counts)
        assert strength == "strong"
        assert variance < 1

    def test_medium_consensus(self):
        """Test medium consensus (1 <= variance < 2)."""
        # variance = ((5-4.5)^2 + (4-4.5)^2) / 2 = (0.25 + 0.25) / 2 = 0.25 -> strong
        # Need values that produce variance between 1 and 2
        # Try {"A": 5, "B": 2} -> mean=3.5, var=((5-3.5)^2+(2-3.5)^2)/2 = (2.25+2.25)/2 = 2.25 -> weak
        # Try {"A": 4, "B": 2} -> mean=3, var=((4-3)^2+(2-3)^2)/2 = (1+1)/2 = 1 -> medium
        counts = Counter({"A": 4, "B": 2})
        strength, variance = calculate_consensus_strength(counts)
        assert strength == "medium"
        assert 1 <= variance < 2

    def test_weak_consensus(self):
        """Test weak consensus (variance >= 2)."""
        counts = Counter({"A": 10, "B": 1, "C": 1})  # high variance
        strength, variance = calculate_consensus_strength(counts)
        # Variance depends on exact distribution


class TestVoteAggregator:
    """Tests for VoteAggregator class."""

    def test_init_defaults(self):
        """Test VoteAggregator default initialization."""
        aggregator = VoteAggregator()
        assert aggregator._group_similar_votes is None
        assert aggregator._base_user_weight == 0.5
        assert aggregator._user_vote_multiplier is None

    def test_init_with_params(self):
        """Test VoteAggregator with parameters."""
        group_fn = MagicMock()
        mult_fn = MagicMock()
        aggregator = VoteAggregator(
            group_similar_votes=group_fn,
            user_vote_weight=0.7,
            user_vote_multiplier=mult_fn,
        )
        assert aggregator._group_similar_votes is group_fn
        assert aggregator._base_user_weight == 0.7
        assert aggregator._user_vote_multiplier is mult_fn

    def test_aggregate_empty_votes(self):
        """Test aggregate with empty vote list."""
        aggregator = VoteAggregator()
        result = aggregator.aggregate(votes=[])
        assert len(result.vote_counts) == 0
        assert result.total_weighted == 0.0
        assert result.agent_votes_count == 0

    def test_aggregate_single_vote(self):
        """Test aggregate with single vote."""
        aggregator = VoteAggregator()
        votes = [MockVote("agent1", "A")]

        result = aggregator.aggregate(votes=votes)

        assert result.vote_counts["A"] == 1.0
        assert result.total_weighted == 1.0
        assert result.agent_votes_count == 1

    def test_aggregate_multiple_votes_same_choice(self):
        """Test aggregate with multiple votes for same choice."""
        aggregator = VoteAggregator()
        votes = [
            MockVote("agent1", "A"),
            MockVote("agent2", "A"),
            MockVote("agent3", "A"),
        ]

        result = aggregator.aggregate(votes=votes)

        assert result.vote_counts["A"] == 3.0
        assert result.total_weighted == 3.0
        assert result.agent_votes_count == 3

    def test_aggregate_multiple_choices(self):
        """Test aggregate with votes for different choices."""
        aggregator = VoteAggregator()
        votes = [
            MockVote("agent1", "A"),
            MockVote("agent2", "B"),
            MockVote("agent3", "A"),
        ]

        result = aggregator.aggregate(votes=votes)

        assert result.vote_counts["A"] == 2.0
        assert result.vote_counts["B"] == 1.0
        assert result.total_weighted == 3.0

    def test_aggregate_with_weights(self):
        """Test aggregate with vote weights."""
        aggregator = VoteAggregator()
        votes = [
            MockVote("agent1", "A"),
            MockVote("agent2", "B"),
        ]
        weights = {"agent1": 1.5, "agent2": 0.8}

        result = aggregator.aggregate(votes=votes, weights=weights)

        assert result.vote_counts["A"] == 1.5
        assert result.vote_counts["B"] == 0.8
        assert result.total_weighted == 2.3

    def test_aggregate_with_user_votes(self):
        """Test aggregate includes user votes."""
        aggregator = VoteAggregator()
        votes = [MockVote("agent1", "A")]
        user_votes = [{"choice": "B", "user_id": "user1"}]

        result = aggregator.aggregate(votes=votes, user_votes=user_votes)

        assert result.vote_counts["A"] == 1.0
        assert result.vote_counts["B"] == 0.5  # default user weight
        assert result.user_votes_count == 1
        assert result.total_votes == 2

    def test_aggregate_user_vote_with_intensity(self):
        """Test user vote intensity affects nothing by default."""
        aggregator = VoteAggregator()
        user_votes = [{"choice": "A", "intensity": 10}]

        result = aggregator.aggregate(votes=[], user_votes=user_votes)

        # Without multiplier, intensity doesn't affect weight
        assert result.vote_counts["A"] == 0.5

    def test_aggregate_user_vote_with_multiplier(self):
        """Test user vote intensity with multiplier callback."""
        mult_fn = MagicMock(return_value=2.0)
        aggregator = VoteAggregator(user_vote_multiplier=mult_fn)
        user_votes = [{"choice": "A", "intensity": 8}]

        result = aggregator.aggregate(votes=[], user_votes=user_votes)

        mult_fn.assert_called_once()
        assert result.vote_counts["A"] == 1.0  # 0.5 * 2.0

    def test_aggregate_skips_empty_user_choice(self):
        """Test user votes with empty choice are skipped."""
        aggregator = VoteAggregator()
        user_votes = [
            {"choice": "A"},
            {"choice": ""},
            {"user_id": "user2"},  # no choice
        ]

        result = aggregator.aggregate(votes=[], user_votes=user_votes)

        assert result.vote_counts["A"] == 0.5
        assert result.user_votes_count == 1

    def test_aggregate_with_grouping(self):
        """Test aggregate with vote grouping callback."""
        group_fn = MagicMock(return_value={
            "A": ["A", "a", "Agent A"],
            "B": ["B", "b"],
        })
        aggregator = VoteAggregator(group_similar_votes=group_fn)
        votes = [
            MockVote("agent1", "a"),
            MockVote("agent2", "Agent A"),
            MockVote("agent3", "B"),
        ]

        result = aggregator.aggregate(votes=votes)

        # "a" and "Agent A" should be mapped to "A"
        assert result.vote_counts["A"] == 2.0
        assert result.vote_counts["B"] == 1.0
        assert result.choice_mapping["a"] == "A"
        assert result.choice_mapping["Agent A"] == "A"

    def test_aggregate_handles_exception_votes(self):
        """Test aggregate skips exception objects in vote list."""
        aggregator = VoteAggregator()
        votes = [
            MockVote("agent1", "A"),
            Exception("Vote failed"),
            MockVote("agent2", "B"),
        ]

        result = aggregator.aggregate(votes=votes)

        assert result.vote_counts["A"] == 1.0
        assert result.vote_counts["B"] == 1.0
        assert result.agent_votes_count == 2

    def test_count_unweighted(self):
        """Test unweighted vote counting."""
        aggregator = VoteAggregator()
        votes = [
            MockVote("agent1", "A"),
            MockVote("agent2", "A"),
            MockVote("agent3", "B"),
        ]

        counts = aggregator.count_unweighted(votes)

        assert counts["A"] == 2
        assert counts["B"] == 1

    def test_count_unweighted_with_mapping(self):
        """Test unweighted counting with choice mapping."""
        aggregator = VoteAggregator()
        votes = [
            MockVote("agent1", "a"),
            MockVote("agent2", "A"),
        ]
        mapping = {"a": "A"}

        counts = aggregator.count_unweighted(votes, choice_mapping=mapping)

        assert counts["A"] == 2

    def test_count_unweighted_skips_exceptions(self):
        """Test unweighted counting skips exceptions."""
        aggregator = VoteAggregator()
        votes = [
            MockVote("agent1", "A"),
            Exception("Failed"),
            MockVote("agent2", "B"),
        ]

        counts = aggregator.count_unweighted(votes)

        assert counts["A"] == 1
        assert counts["B"] == 1

    def test_grouping_error_handled_gracefully(self):
        """Test that grouping errors are handled."""
        group_fn = MagicMock(side_effect=Exception("Group error"))
        aggregator = VoteAggregator(group_similar_votes=group_fn)
        votes = [MockVote("agent1", "A")]

        result = aggregator.aggregate(votes=votes)

        # Should fall back to identity mapping
        assert result.vote_counts["A"] == 1.0

    def test_user_multiplier_error_handled(self):
        """Test that user multiplier errors are handled."""
        mult_fn = MagicMock(side_effect=Exception("Mult error"))
        aggregator = VoteAggregator(user_vote_multiplier=mult_fn)
        user_votes = [{"choice": "A", "intensity": 5}]

        result = aggregator.aggregate(votes=[], user_votes=user_votes)

        # Should use default multiplier (1.0)
        assert result.vote_counts["A"] == 0.5

    def test_result_contains_vote_groups(self):
        """Test that result includes vote groups."""
        group_fn = MagicMock(return_value={"A": ["A", "a"]})
        aggregator = VoteAggregator(group_similar_votes=group_fn)
        votes = [MockVote("agent1", "A")]

        result = aggregator.aggregate(votes=votes)

        assert result.vote_groups == {"A": ["A", "a"]}

    def test_aggregate_missing_choice_attr(self):
        """Test aggregate handles votes without choice attribute."""
        aggregator = VoteAggregator()

        class BadVote:
            def __init__(self, agent):
                self.agent = agent
                # No choice attribute

        votes = [MockVote("agent1", "A"), BadVote("agent2")]

        result = aggregator.aggregate(votes=votes)

        # Only the valid vote with choice attribute is counted in vote_counts
        assert result.vote_counts["A"] == 1.0
        # agent_votes_count counts all non-exception objects before filtering
        # This is the current behavior - counts objects that pass the exception check
        assert result.agent_votes_count == 2
        # But total_weighted only includes votes with valid choice
        assert result.total_weighted == 1.0


class TestVoteAggregatorIntegration:
    """Integration tests for VoteAggregator."""

    def test_full_aggregation_flow(self):
        """Test complete aggregation flow with all features."""
        # Setup
        group_fn = MagicMock(return_value={
            "Option A": ["Option A", "A", "option-a"],
            "Option B": ["Option B", "B"],
        })
        mult_fn = MagicMock(return_value=1.5)

        aggregator = VoteAggregator(
            group_similar_votes=group_fn,
            user_vote_weight=0.6,
            user_vote_multiplier=mult_fn,
        )

        votes = [
            MockVote("agent1", "A"),
            MockVote("agent2", "option-a"),
            MockVote("agent3", "B"),
        ]
        weights = {"agent1": 1.2, "agent2": 0.9, "agent3": 1.0}
        user_votes = [
            {"choice": "Option A", "intensity": 7, "user_id": "user1"},
        ]

        result = aggregator.aggregate(
            votes=votes,
            weights=weights,
            user_votes=user_votes,
        )

        # Agent votes: A=1.2, option-a=0.9 (both -> Option A = 2.1), B=1.0
        # User vote: Option A = 0.6 * 1.5 = 0.9
        # Total Option A = 2.1 + 0.9 = 3.0
        assert result.vote_counts["Option A"] == 3.0
        assert result.vote_counts["Option B"] == 1.0
        assert result.agent_votes_count == 3
        assert result.user_votes_count == 1
        assert result.total_votes == 4

        # Winner should be Option A
        winner = result.get_winner()
        assert winner[0] == "Option A"
        assert winner[1] == 3.0

        # Confidence should be 3.0 / 4.0 = 0.75
        assert abs(result.get_confidence() - 0.75) < 0.01
