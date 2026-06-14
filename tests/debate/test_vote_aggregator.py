"""Tests for the vote_aggregator module."""

import pytest
from unittest.mock import Mock, MagicMock
from collections import Counter


# -----------------------------------------------------------------------------
# AggregatedVotes Tests
# -----------------------------------------------------------------------------


class TestAggregatedVotes:
    """Test AggregatedVotes dataclass."""

    def test_default_values(self):
        """AggregatedVotes has correct defaults."""
        from aragora.debate.phases.vote_aggregator import AggregatedVotes

        result = AggregatedVotes()

        assert result.vote_counts == Counter()
        assert result.total_weighted == 0.0
        assert result.choice_mapping == {}
        assert result.vote_groups == {}
        assert result.total_votes == 0
        assert result.user_votes_count == 0
        assert result.agent_votes_count == 0

    def test_get_winner_empty(self):
        """get_winner returns None for empty votes."""
        from aragora.debate.phases.vote_aggregator import AggregatedVotes

        result = AggregatedVotes()
        assert result.get_winner() is None

    def test_get_winner_single_choice(self):
        """get_winner returns the only choice."""
        from aragora.debate.phases.vote_aggregator import AggregatedVotes

        result = AggregatedVotes(
            vote_counts=Counter({"Option A": 5.0}),
            total_weighted=5.0,
        )

        winner = result.get_winner()
        assert winner == ("Option A", 5.0)

    def test_get_winner_multiple_choices(self):
        """get_winner returns choice with most votes."""
        from aragora.debate.phases.vote_aggregator import AggregatedVotes

        result = AggregatedVotes(
            vote_counts=Counter({"Option A": 3.0, "Option B": 5.0, "Option C": 2.0}),
            total_weighted=10.0,
        )

        winner = result.get_winner()
        assert winner == ("Option B", 5.0)

    def test_get_confidence_empty(self):
        """get_confidence returns 0.5 for empty votes."""
        from aragora.debate.phases.vote_aggregator import AggregatedVotes

        result = AggregatedVotes()
        assert result.get_confidence() == 0.5

    def test_get_confidence_calculation(self):
        """get_confidence calculates winner_votes / total."""
        from aragora.debate.phases.vote_aggregator import AggregatedVotes

        result = AggregatedVotes(
            vote_counts=Counter({"Option A": 7.0, "Option B": 3.0}),
            total_weighted=10.0,
        )

        assert result.get_confidence() == 0.7

    def test_get_vote_distribution_empty(self):
        """get_vote_distribution returns empty dict for no votes."""
        from aragora.debate.phases.vote_aggregator import AggregatedVotes

        result = AggregatedVotes()
        assert result.get_vote_distribution() == {}

    def test_get_vote_distribution(self):
        """get_vote_distribution returns percentages."""
        from aragora.debate.phases.vote_aggregator import AggregatedVotes

        result = AggregatedVotes(
            vote_counts=Counter({"A": 3.0, "B": 2.0}),
            total_weighted=5.0,
        )

        dist = result.get_vote_distribution()
        assert dist["A"] == 0.6
        assert dist["B"] == 0.4


# -----------------------------------------------------------------------------
# VoteAggregator Initialization Tests
# -----------------------------------------------------------------------------


class TestVoteAggregatorInit:
    """Test VoteAggregator initialization."""

    def test_default_init(self):
        """Can initialize with defaults."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator

        agg = VoteAggregator()

        assert agg._group_similar_votes is None
        assert agg._base_user_weight == 0.5
        assert agg._user_vote_multiplier is None
        assert agg.protocol is None

    def test_custom_init(self):
        """Can initialize with custom values."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator

        group_fn = Mock()
        multiplier_fn = Mock()
        protocol = Mock()

        agg = VoteAggregator(
            group_similar_votes=group_fn,
            user_vote_weight=0.8,
            user_vote_multiplier=multiplier_fn,
            protocol=protocol,
        )

        assert agg._group_similar_votes is group_fn
        assert agg._base_user_weight == 0.8
        assert agg._user_vote_multiplier is multiplier_fn
        assert agg.protocol is protocol


# -----------------------------------------------------------------------------
# VoteAggregator Vote Grouping Tests
# -----------------------------------------------------------------------------


class TestComputeVoteGroups:
    """Test vote grouping computation."""

    @pytest.fixture
    def mock_vote(self):
        def _make_vote(agent: str, choice: str):
            vote = Mock(spec=["agent", "choice"])
            vote.agent = agent
            vote.choice = choice
            return vote

        return _make_vote

    def test_no_grouping_function(self, mock_vote):
        """Without grouping function, creates identity mapping."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator

        agg = VoteAggregator()
        votes = [mock_vote("a1", "Option A"), mock_vote("a2", "Option B")]

        groups, mapping = agg._compute_vote_groups(votes)

        assert "Option A" in groups
        assert "Option B" in groups
        assert mapping["Option A"] == "Option A"
        assert mapping["Option B"] == "Option B"

    def test_with_grouping_function(self, mock_vote):
        """Grouping function merges similar choices."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator

        def mock_grouping(votes):
            return {"Vector DB": ["Vector DB", "Use vector database"]}

        agg = VoteAggregator(group_similar_votes=mock_grouping)
        votes = [
            mock_vote("a1", "Vector DB"),
            mock_vote("a2", "Use vector database"),
        ]

        groups, mapping = agg._compute_vote_groups(votes)

        assert groups == {"Vector DB": ["Vector DB", "Use vector database"]}
        assert mapping["Vector DB"] == "Vector DB"
        assert mapping["Use vector database"] == "Vector DB"

    def test_grouping_function_error(self, mock_vote):
        """Errors in grouping function fall back to identity."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator

        def failing_grouping(votes):
            raise ValueError("Test error")

        agg = VoteAggregator(group_similar_votes=failing_grouping)
        votes = [mock_vote("a1", "Option A")]

        # Should not raise
        groups, mapping = agg._compute_vote_groups(votes)

        assert "Option A" in groups
        assert mapping["Option A"] == "Option A"

    def test_skips_exception_votes(self, mock_vote):
        """Exception votes are skipped in grouping."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator

        agg = VoteAggregator()
        votes = [
            mock_vote("a1", "Option A"),
            ValueError("Test error"),  # Exception
        ]

        groups, mapping = agg._compute_vote_groups(votes)

        assert "Option A" in groups
        assert len(groups) == 1


# -----------------------------------------------------------------------------
# VoteAggregator Weighted Counting Tests
# -----------------------------------------------------------------------------


class TestCountWeightedVotes:
    """Test weighted vote counting."""

    @pytest.fixture
    def mock_vote(self):
        def _make_vote(agent: str, choice: str):
            vote = Mock(spec=["agent", "choice"])
            vote.agent = agent
            vote.choice = choice
            return vote

        return _make_vote

    def test_unweighted_counting(self, mock_vote):
        """Without weights, all votes count as 1."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator

        agg = VoteAggregator()
        votes = [
            mock_vote("a1", "Option A"),
            mock_vote("a2", "Option A"),
            mock_vote("a3", "Option B"),
        ]
        mapping = {"Option A": "Option A", "Option B": "Option B"}

        counts, total = agg._count_weighted_votes(votes, mapping, {})

        assert counts["Option A"] == 2.0
        assert counts["Option B"] == 1.0
        assert total == 3.0

    def test_weighted_counting(self, mock_vote):
        """Weights are applied to votes."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator

        agg = VoteAggregator()
        votes = [
            mock_vote("expert", "Option A"),
            mock_vote("junior", "Option B"),
        ]
        mapping = {"Option A": "Option A", "Option B": "Option B"}
        weights = {"expert": 2.0, "junior": 0.5}

        counts, total = agg._count_weighted_votes(votes, mapping, weights)

        assert counts["Option A"] == 2.0
        assert counts["Option B"] == 0.5
        assert total == 2.5

    def test_applies_choice_mapping(self, mock_vote):
        """Choice mapping normalizes vote choices."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator

        agg = VoteAggregator()
        votes = [
            mock_vote("a1", "Vector DB"),
            mock_vote("a2", "Use vector database"),
        ]
        mapping = {
            "Vector DB": "Vector DB",
            "Use vector database": "Vector DB",
        }

        counts, total = agg._count_weighted_votes(votes, mapping, {})

        assert counts["Vector DB"] == 2.0
        assert len(counts) == 1

    def test_skips_exceptions(self, mock_vote):
        """Exception votes are skipped."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator

        agg = VoteAggregator()
        votes = [
            mock_vote("a1", "Option A"),
            ValueError("Test error"),
        ]
        mapping = {"Option A": "Option A"}

        counts, total = agg._count_weighted_votes(votes, mapping, {})

        assert counts["Option A"] == 1.0
        assert total == 1.0

    def test_skips_votes_without_choice(self, mock_vote):
        """Votes without choice attribute are skipped."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator

        agg = VoteAggregator()
        invalid_vote = Mock(spec=["agent"])  # No 'choice' attr
        invalid_vote.agent = "a2"
        votes = [mock_vote("a1", "Option A"), invalid_vote]
        mapping = {"Option A": "Option A"}

        counts, total = agg._count_weighted_votes(votes, mapping, {})

        assert counts["Option A"] == 1.0
        assert total == 1.0


# -----------------------------------------------------------------------------
# VoteAggregator User Votes Tests
# -----------------------------------------------------------------------------


class TestAddUserVotes:
    """Test adding user votes."""

    def test_empty_user_votes(self):
        """Empty user votes don't change counts."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator

        agg = VoteAggregator()
        counts = Counter({"A": 2.0})
        total = 2.0

        new_counts, new_total, user_count = agg._add_user_votes(counts, total, {}, [])

        assert new_counts["A"] == 2.0
        assert new_total == 2.0
        assert user_count == 0

    def test_add_single_user_vote(self):
        """Single user vote is added with base weight."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator

        agg = VoteAggregator(user_vote_weight=0.5)
        counts = Counter({"A": 2.0})
        total = 2.0
        user_votes = [{"choice": "B", "user_id": "user1"}]

        new_counts, new_total, user_count = agg._add_user_votes(counts, total, {}, user_votes)

        assert new_counts["A"] == 2.0
        assert new_counts["B"] == 0.5
        assert new_total == 2.5
        assert user_count == 1

    def test_user_vote_with_intensity_multiplier(self):
        """Intensity multiplier is applied to user votes."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator

        def intensity_multiplier(intensity, protocol):
            return intensity / 5  # intensity 10 -> 2.0x

        agg = VoteAggregator(
            user_vote_weight=0.5,
            user_vote_multiplier=intensity_multiplier,
        )
        counts = Counter()
        user_votes = [{"choice": "A", "intensity": 10}]

        new_counts, new_total, user_count = agg._add_user_votes(counts, 0.0, {}, user_votes)

        assert new_counts["A"] == 1.0  # 0.5 * 2.0
        assert new_total == 1.0

    def test_user_vote_applies_choice_mapping(self):
        """Choice mapping normalizes user vote choices."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator

        agg = VoteAggregator(user_vote_weight=1.0)
        counts = Counter({"Vector DB": 2.0})
        mapping = {"use vector database": "Vector DB"}
        user_votes = [{"choice": "use vector database"}]

        new_counts, new_total, _ = agg._add_user_votes(counts, 2.0, mapping, user_votes)

        assert new_counts["Vector DB"] == 3.0

    def test_skips_empty_choice(self):
        """User votes with empty choice are skipped."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator

        agg = VoteAggregator()
        counts = Counter()
        user_votes = [{"choice": "", "user_id": "user1"}]

        new_counts, new_total, user_count = agg._add_user_votes(counts, 0.0, {}, user_votes)

        assert len(new_counts) == 0
        assert user_count == 0

    def test_multiplier_error_fallback(self):
        """Errors in multiplier fall back to 1.0."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator

        def failing_multiplier(intensity, protocol):
            raise ValueError("Test error")

        agg = VoteAggregator(
            user_vote_weight=0.5,
            user_vote_multiplier=failing_multiplier,
        )
        counts = Counter()
        user_votes = [{"choice": "A", "intensity": 10}]

        # Should not raise
        new_counts, new_total, _ = agg._add_user_votes(counts, 0.0, {}, user_votes)

        assert new_counts["A"] == 0.5  # fallback multiplier 1.0


# -----------------------------------------------------------------------------
# VoteAggregator Count Unweighted Tests
# -----------------------------------------------------------------------------


class TestCountUnweighted:
    """Test unweighted counting for unanimous mode."""

    @pytest.fixture
    def mock_vote(self):
        def _make_vote(agent: str, choice: str):
            vote = Mock(spec=["agent", "choice"])
            vote.agent = agent
            vote.choice = choice
            return vote

        return _make_vote

    def test_basic_counting(self, mock_vote):
        """Counts votes without weights."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator

        agg = VoteAggregator()
        votes = [
            mock_vote("a1", "Option A"),
            mock_vote("a2", "Option A"),
            mock_vote("a3", "Option B"),
        ]

        counts = agg.count_unweighted(votes)

        assert counts["Option A"] == 2
        assert counts["Option B"] == 1

    def test_applies_choice_mapping(self, mock_vote):
        """Choice mapping normalizes choices."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator

        agg = VoteAggregator()
        votes = [
            mock_vote("a1", "Vector DB"),
            mock_vote("a2", "vector database"),
        ]
        mapping = {"vector database": "Vector DB"}

        counts = agg.count_unweighted(votes, mapping)

        assert counts["Vector DB"] == 2

    def test_skips_exceptions(self, mock_vote):
        """Exception votes are skipped."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator

        agg = VoteAggregator()
        votes = [
            mock_vote("a1", "Option A"),
            ValueError("Test"),
        ]

        counts = agg.count_unweighted(votes)

        assert counts["Option A"] == 1
        assert len(counts) == 1


# -----------------------------------------------------------------------------
# VoteAggregator Aggregate Tests
# -----------------------------------------------------------------------------


class TestAggregate:
    """Test the main aggregate method."""

    @pytest.fixture
    def mock_vote(self):
        def _make_vote(agent: str, choice: str):
            vote = Mock(spec=["agent", "choice"])
            vote.agent = agent
            vote.choice = choice
            return vote

        return _make_vote

    def test_basic_aggregation(self, mock_vote):
        """Basic aggregation without user votes."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator

        agg = VoteAggregator()
        votes = [
            mock_vote("a1", "Option A"),
            mock_vote("a2", "Option A"),
            mock_vote("a3", "Option B"),
        ]

        result = agg.aggregate(votes)

        assert result.vote_counts["Option A"] == 2.0
        assert result.vote_counts["Option B"] == 1.0
        assert result.total_weighted == 3.0
        assert result.agent_votes_count == 3
        assert result.user_votes_count == 0

    def test_aggregation_with_weights(self, mock_vote):
        """Aggregation with vote weights."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator

        agg = VoteAggregator()
        votes = [
            mock_vote("expert", "Option A"),
            mock_vote("junior", "Option B"),
        ]
        weights = {"expert": 2.0, "junior": 0.5}

        result = agg.aggregate(votes, weights=weights)

        assert result.vote_counts["Option A"] == 2.0
        assert result.vote_counts["Option B"] == 0.5
        assert result.total_weighted == 2.5

    def test_aggregation_with_user_votes(self, mock_vote):
        """Aggregation including user votes."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator

        agg = VoteAggregator(user_vote_weight=0.5)
        votes = [mock_vote("a1", "Option A")]
        user_votes = [
            {"choice": "Option B", "user_id": "user1"},
            {"choice": "Option B", "user_id": "user2"},
        ]

        result = agg.aggregate(votes, user_votes=user_votes)

        assert result.vote_counts["Option A"] == 1.0
        assert result.vote_counts["Option B"] == 1.0  # 2 * 0.5
        assert result.agent_votes_count == 1
        assert result.user_votes_count == 2
        assert result.total_votes == 3

    def test_aggregation_with_grouping(self, mock_vote):
        """Aggregation with vote grouping."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator

        def mock_grouping(votes):
            return {"Vector DB": ["Vector DB", "Use vector database"]}

        agg = VoteAggregator(group_similar_votes=mock_grouping)
        votes = [
            mock_vote("a1", "Vector DB"),
            mock_vote("a2", "Use vector database"),
        ]

        result = agg.aggregate(votes)

        assert result.vote_counts["Vector DB"] == 2.0
        assert result.choice_mapping["Use vector database"] == "Vector DB"


# -----------------------------------------------------------------------------
# calculate_consensus_strength Tests
# -----------------------------------------------------------------------------


class TestCalculateConsensusStrength:
    """Test consensus strength calculation."""

    def test_empty_votes(self):
        """Empty votes returns unanimous with 0 variance."""
        from aragora.debate.phases.vote_aggregator import calculate_consensus_strength

        strength, variance = calculate_consensus_strength(Counter())

        assert strength == "unanimous"
        assert variance == 0.0

    def test_single_choice_unanimous(self):
        """Single choice is unanimous."""
        from aragora.debate.phases.vote_aggregator import calculate_consensus_strength

        strength, variance = calculate_consensus_strength(Counter({"A": 5}))

        assert strength == "unanimous"
        assert variance == 0.0

    def test_strong_consensus(self):
        """Low variance is strong consensus."""
        from aragora.debate.phases.vote_aggregator import calculate_consensus_strength

        # Variance < 1
        strength, variance = calculate_consensus_strength(Counter({"A": 4, "B": 4}))  # variance = 0

        assert strength == "strong"
        assert variance < 1

    def test_medium_consensus(self):
        """Moderate variance is medium consensus."""
        from aragora.debate.phases.vote_aggregator import calculate_consensus_strength

        # Variance 1-2
        strength, variance = calculate_consensus_strength(Counter({"A": 5, "B": 3}))  # variance = 1

        assert strength == "medium"
        assert 1 <= variance < 2

    def test_weak_consensus(self):
        """High variance is weak consensus."""
        from aragora.debate.phases.vote_aggregator import calculate_consensus_strength

        # Variance >= 2
        strength, variance = calculate_consensus_strength(Counter({"A": 8, "B": 2}))  # variance = 9

        assert strength == "weak"
        assert variance >= 2


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestVoteAggregatorIntegration:
    """Integration tests for vote aggregation workflow."""

    @pytest.fixture
    def mock_vote(self):
        def _make_vote(agent: str, choice: str):
            vote = Mock(spec=["agent", "choice"])
            vote.agent = agent
            vote.choice = choice
            return vote

        return _make_vote

    def test_full_workflow(self, mock_vote):
        """Test complete aggregation workflow."""
        from aragora.debate.phases.vote_aggregator import (
            VoteAggregator,
            calculate_consensus_strength,
        )

        # Setup with grouping and weights
        def group_fn(votes):
            return {"Implement caching": ["Implement caching", "Add cache"]}

        agg = VoteAggregator(
            group_similar_votes=group_fn,
            user_vote_weight=0.5,
        )

        votes = [
            mock_vote("expert", "Implement caching"),
            mock_vote("junior", "Add cache"),
            mock_vote("standard", "More tests"),
        ]
        weights = {"expert": 1.5, "junior": 0.8, "standard": 1.0}
        user_votes = [{"choice": "Implement caching", "user_id": "user1"}]

        # Aggregate
        result = agg.aggregate(votes, weights=weights, user_votes=user_votes)

        # Verify
        assert result.vote_counts["Implement caching"] == 2.8  # 1.5 + 0.8 + 0.5
        assert result.vote_counts["More tests"] == 1.0
        assert result.agent_votes_count == 3
        assert result.user_votes_count == 1

        # Check winner
        winner = result.get_winner()
        assert winner[0] == "Implement caching"

        # Check consensus strength
        strength, _ = calculate_consensus_strength(result.vote_counts)
        assert strength in ["strong", "medium", "weak"]

    def test_tie_scenario(self, mock_vote):
        """Test handling of tied votes."""
        from aragora.debate.phases.vote_aggregator import VoteAggregator

        agg = VoteAggregator()
        votes = [
            mock_vote("a1", "Option A"),
            mock_vote("a2", "Option B"),
        ]

        result = agg.aggregate(votes)

        # Both have equal weight
        assert result.vote_counts["Option A"] == 1.0
        assert result.vote_counts["Option B"] == 1.0
        # Winner is one of them (Counter.most_common returns arbitrary for ties)
        winner = result.get_winner()
        assert winner[0] in ["Option A", "Option B"]
        assert winner[1] == 1.0
