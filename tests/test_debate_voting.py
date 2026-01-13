"""
Tests for the Voting Phase system.

Tests vote collection, weighting, semantic grouping,
consensus strength calculation, and vote aggregation.
"""

import pytest
from unittest.mock import MagicMock, patch
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from aragora.debate.phases.voting import (
    VoteWeightCalculator,
    VotingPhase,
    WeightedVoteResult,
)


# =============================================================================
# Mock Classes for Testing
# =============================================================================


@dataclass
class MockVote:
    """Mock Vote class for testing."""

    agent: str
    choice: str
    reasoning: str = ""
    confidence: Optional[float] = None


@dataclass
class MockProtocol:
    """Mock DebateProtocol for testing."""

    vote_grouping: bool = False  # Default to False to avoid lazy backend loading
    vote_grouping_threshold: float = 0.7
    enable_weighted_voting: bool = True


@dataclass
class MockSimilarityBackend:
    """Mock similarity backend for testing."""

    similarities: dict = None

    def __post_init__(self):
        if self.similarities is None:
            self.similarities = {}

    def compute_similarity(self, text1: str, text2: str) -> float:
        key = (text1, text2)
        rev_key = (text2, text1)
        return self.similarities.get(key, self.similarities.get(rev_key, 0.0))


# =============================================================================
# VoteWeightCalculator Tests
# =============================================================================


class TestVoteWeightCalculator:
    """Tests for VoteWeightCalculator class."""

    def test_default_weight_no_sources(self):
        """Test default weight when no sources configured."""
        calculator = VoteWeightCalculator()
        weight = calculator.compute_weight("agent1")

        assert weight == 1.0

    def test_reputation_weight(self):
        """Test weight with reputation source."""
        reputation = lambda name: 1.2 if name == "expert" else 0.8
        calculator = VoteWeightCalculator(reputation_source=reputation)

        assert calculator.compute_weight("expert") == 1.2
        assert calculator.compute_weight("novice") == 0.8

    def test_reliability_weight(self):
        """Test weight with reliability weights."""
        calculator = VoteWeightCalculator(reliability_weights={"reliable": 1.0, "unreliable": 0.5})

        assert calculator.compute_weight("reliable") == 1.0
        assert calculator.compute_weight("unreliable") == 0.5
        assert calculator.compute_weight("unknown") == 1.0

    def test_consistency_weight(self):
        """Test weight with consistency source."""
        # Consistency score 0-1 maps to 0.5-1.0 weight multiplier
        consistency = lambda name: 1.0 if name == "consistent" else 0.0
        calculator = VoteWeightCalculator(consistency_source=consistency)

        assert calculator.compute_weight("consistent") == 1.0  # 0.5 + (1.0 * 0.5)
        assert calculator.compute_weight("inconsistent") == 0.5  # 0.5 + (0.0 * 0.5)

    def test_calibration_weight(self):
        """Test weight with calibration source."""
        calibration = lambda name: 1.3 if name == "calibrated" else 0.7
        calculator = VoteWeightCalculator(calibration_source=calibration)

        assert calculator.compute_weight("calibrated") == 1.3
        assert calculator.compute_weight("uncalibrated") == 0.7

    def test_combined_weights_multiplication(self):
        """Test that weights are multiplied together."""
        calculator = VoteWeightCalculator(
            reputation_source=lambda name: 1.2,
            reliability_weights={"agent1": 0.9},
            calibration_source=lambda name: 1.1,
        )

        weight = calculator.compute_weight("agent1")
        expected = 1.2 * 0.9 * 1.1
        assert weight == pytest.approx(expected, rel=1e-3)

    def test_weight_caching(self):
        """Test that weights are cached."""
        call_count = 0

        def reputation(name):
            nonlocal call_count
            call_count += 1
            return 1.0

        calculator = VoteWeightCalculator(reputation_source=reputation)

        calculator.compute_weight("agent1")
        calculator.compute_weight("agent1")
        calculator.compute_weight("agent1")

        assert call_count == 1  # Only called once, rest from cache

    def test_clear_cache(self):
        """Test clearing the weight cache."""
        call_count = 0

        def reputation(name):
            nonlocal call_count
            call_count += 1
            return 1.0

        calculator = VoteWeightCalculator(reputation_source=reputation)

        calculator.compute_weight("agent1")
        calculator.clear_cache()
        calculator.compute_weight("agent1")

        assert call_count == 2  # Called twice after cache clear

    def test_compute_weights_batch(self):
        """Test batch weight computation."""
        calculator = VoteWeightCalculator(reliability_weights={"a": 0.8, "b": 1.2})

        weights = calculator.compute_weights_batch(["a", "b", "c"])

        assert weights["a"] == 0.8
        assert weights["b"] == 1.2
        assert weights["c"] == 1.0

    def test_error_handling_in_sources(self):
        """Test that errors in weight sources are handled gracefully."""

        def failing_reputation(name):
            raise ValueError("Reputation lookup failed")

        calculator = VoteWeightCalculator(reputation_source=failing_reputation)

        # Should not raise, should return default weight
        weight = calculator.compute_weight("agent1")
        assert weight == 1.0


# =============================================================================
# VotingPhase Initialization Tests
# =============================================================================


class TestVotingPhaseInit:
    """Tests for VotingPhase initialization."""

    def test_basic_initialization(self):
        """Test basic VotingPhase initialization."""
        protocol = MockProtocol()
        voting = VotingPhase(protocol=protocol)

        assert voting.protocol is protocol
        assert voting._similarity_backend is None

    def test_initialization_with_backend(self):
        """Test initialization with similarity backend."""
        protocol = MockProtocol()
        backend = MockSimilarityBackend()
        voting = VotingPhase(protocol=protocol, similarity_backend=backend)

        assert voting._similarity_backend is backend


# =============================================================================
# Vote Grouping Tests
# =============================================================================


class TestVoteGrouping:
    """Tests for semantic vote grouping."""

    def test_group_similar_votes_disabled(self):
        """Test grouping when vote_grouping is disabled."""
        protocol = MockProtocol(vote_grouping=False)
        voting = VotingPhase(protocol=protocol)

        votes = [
            MockVote(agent="a", choice="Option A"),
            MockVote(agent="b", choice="Option A"),
        ]

        groups = voting.group_similar_votes(votes)

        assert groups == {}

    def test_group_similar_votes_empty(self):
        """Test grouping with empty votes."""
        protocol = MockProtocol(vote_grouping=True)
        backend = MockSimilarityBackend()
        voting = VotingPhase(protocol=protocol, similarity_backend=backend)

        groups = voting.group_similar_votes([])

        assert groups == {}

    def test_group_similar_votes_single_vote(self):
        """Test grouping with single vote."""
        protocol = MockProtocol(vote_grouping=True)
        backend = MockSimilarityBackend()
        voting = VotingPhase(protocol=protocol, similarity_backend=backend)

        votes = [MockVote(agent="a", choice="Option A")]
        groups = voting.group_similar_votes(votes)

        assert groups == {}

    def test_group_similar_votes_identical(self):
        """Test grouping identical choices (no merging needed)."""
        protocol = MockProtocol(vote_grouping=True)
        backend = MockSimilarityBackend(similarities={})
        voting = VotingPhase(protocol=protocol, similarity_backend=backend)

        votes = [
            MockVote(agent="a", choice="Vector DB"),
            MockVote(agent="b", choice="Vector DB"),
        ]

        groups = voting.group_similar_votes(votes)

        # Identical choices don't need grouping
        assert groups == {}

    def test_group_similar_votes_merges_similar(self):
        """Test that similar votes are merged."""
        protocol = MockProtocol(vote_grouping=True, vote_grouping_threshold=0.7)
        backend = MockSimilarityBackend(similarities={("Vector DB", "Use vector database"): 0.85})
        voting = VotingPhase(protocol=protocol, similarity_backend=backend)

        votes = [
            MockVote(agent="a", choice="Vector DB"),
            MockVote(agent="b", choice="Use vector database"),
        ]

        groups = voting.group_similar_votes(votes)

        # One group should contain both choices
        assert len(groups) == 1
        canonical = list(groups.keys())[0]
        assert len(groups[canonical]) == 2

    def test_group_similar_votes_keeps_distinct(self):
        """Test that distinct votes are not merged."""
        protocol = MockProtocol(vote_grouping=True, vote_grouping_threshold=0.7)
        backend = MockSimilarityBackend(similarities={("Option A", "Option B"): 0.2})
        voting = VotingPhase(protocol=protocol, similarity_backend=backend)

        votes = [
            MockVote(agent="a", choice="Option A"),
            MockVote(agent="b", choice="Option B"),
        ]

        groups = voting.group_similar_votes(votes)

        # No merging should occur
        assert groups == {}

    def test_apply_vote_grouping_empty_groups(self):
        """Test applying empty grouping."""
        protocol = MockProtocol()
        voting = VotingPhase(protocol=protocol)

        votes = [
            MockVote(agent="a", choice="Option A"),
            MockVote(agent="b", choice="Option B"),
        ]

        result = voting.apply_vote_grouping(votes, {})

        assert len(result) == 2
        assert result[0].choice == "Option A"
        assert result[1].choice == "Option B"

    def test_apply_vote_grouping_normalizes_choices(self):
        """Test that grouping normalizes vote choices."""
        protocol = MockProtocol()
        voting = VotingPhase(protocol=protocol)

        votes = [
            MockVote(agent="a", choice="Vector DB"),
            MockVote(agent="b", choice="Use vector database"),
        ]

        groups = {"Vector DB": ["Vector DB", "Use vector database"]}

        result = voting.apply_vote_grouping(votes, groups)

        # Both should now have the canonical choice
        assert result[0].choice == "Vector DB"
        assert result[1].choice == "Vector DB"


# =============================================================================
# Vote Distribution Tests
# =============================================================================


class TestVoteDistribution:
    """Tests for vote distribution computation."""

    def test_compute_distribution_empty(self):
        """Test distribution with empty votes."""
        protocol = MockProtocol()
        voting = VotingPhase(protocol=protocol)

        dist = voting.compute_vote_distribution([])

        assert dist == {}

    def test_compute_distribution_single_choice(self):
        """Test distribution with all votes for one choice."""
        protocol = MockProtocol()
        voting = VotingPhase(protocol=protocol)

        votes = [
            MockVote(agent="a", choice="Option A"),
            MockVote(agent="b", choice="Option A"),
            MockVote(agent="c", choice="Option A"),
        ]

        dist = voting.compute_vote_distribution(votes)

        assert "Option A" in dist
        assert dist["Option A"]["count"] == 3
        assert dist["Option A"]["percentage"] == 100.0
        assert len(dist["Option A"]["voters"]) == 3

    def test_compute_distribution_multiple_choices(self):
        """Test distribution with multiple choices."""
        protocol = MockProtocol()
        voting = VotingPhase(protocol=protocol)

        votes = [
            MockVote(agent="a", choice="Option A"),
            MockVote(agent="b", choice="Option A"),
            MockVote(agent="c", choice="Option B"),
            MockVote(agent="d", choice="Option C"),
        ]

        dist = voting.compute_vote_distribution(votes)

        assert dist["Option A"]["count"] == 2
        assert dist["Option A"]["percentage"] == 50.0
        assert dist["Option B"]["count"] == 1
        assert dist["Option B"]["percentage"] == 25.0
        assert dist["Option C"]["count"] == 1

    def test_compute_distribution_with_confidence(self):
        """Test distribution includes average confidence."""
        protocol = MockProtocol()
        voting = VotingPhase(protocol=protocol)

        votes = [
            MockVote(agent="a", choice="Option A", confidence=0.8),
            MockVote(agent="b", choice="Option A", confidence=0.6),
        ]

        dist = voting.compute_vote_distribution(votes)

        assert dist["Option A"]["avg_confidence"] == pytest.approx(0.7)

    def test_compute_distribution_missing_confidence(self):
        """Test distribution handles missing confidence."""
        protocol = MockProtocol()
        voting = VotingPhase(protocol=protocol)

        votes = [
            MockVote(agent="a", choice="Option A"),  # No confidence
            MockVote(agent="b", choice="Option A"),
        ]

        dist = voting.compute_vote_distribution(votes)

        assert dist["Option A"]["avg_confidence"] is None


# =============================================================================
# Winner Determination Tests
# =============================================================================


class TestDetermineWinner:
    """Tests for winner determination."""

    def test_determine_winner_clear_majority(self):
        """Test winner with clear majority."""
        protocol = MockProtocol()
        voting = VotingPhase(protocol=protocol)

        votes = [
            MockVote(agent="a", choice="Winner"),
            MockVote(agent="b", choice="Winner"),
            MockVote(agent="c", choice="Loser"),
        ]

        winner = voting.determine_winner(votes)

        assert winner == "Winner"

    def test_determine_winner_require_majority_pass(self):
        """Test winner when majority is required and achieved."""
        protocol = MockProtocol()
        voting = VotingPhase(protocol=protocol)

        votes = [
            MockVote(agent="a", choice="Winner"),
            MockVote(agent="b", choice="Winner"),
            MockVote(agent="c", choice="Winner"),
            MockVote(agent="d", choice="Loser"),
        ]

        winner = voting.determine_winner(votes, require_majority=True)

        assert winner == "Winner"  # 75% > 50%

    def test_determine_winner_require_majority_fail(self):
        """Test no winner when majority required but not achieved."""
        protocol = MockProtocol()
        voting = VotingPhase(protocol=protocol)

        votes = [
            MockVote(agent="a", choice="A"),
            MockVote(agent="b", choice="A"),
            MockVote(agent="c", choice="B"),
            MockVote(agent="d", choice="C"),
        ]

        winner = voting.determine_winner(votes, require_majority=True)

        assert winner is None  # 50% is not > 50%

    def test_determine_winner_min_margin_pass(self):
        """Test winner when margin requirement is met."""
        protocol = MockProtocol()
        voting = VotingPhase(protocol=protocol)

        votes = [
            MockVote(agent="a", choice="A"),
            MockVote(agent="b", choice="A"),
            MockVote(agent="c", choice="A"),
            MockVote(agent="d", choice="B"),
        ]

        winner = voting.determine_winner(votes, min_margin=0.4)

        assert winner == "A"  # 75% - 25% = 50% margin >= 40%

    def test_determine_winner_min_margin_fail(self):
        """Test no winner when margin requirement not met."""
        protocol = MockProtocol()
        voting = VotingPhase(protocol=protocol)

        votes = [
            MockVote(agent="a", choice="A"),
            MockVote(agent="b", choice="A"),
            MockVote(agent="c", choice="B"),
            MockVote(agent="d", choice="C"),
        ]

        winner = voting.determine_winner(votes, min_margin=0.3)

        assert winner is None  # 50% - 25% = 25% < 30%

    def test_determine_winner_empty_votes(self):
        """Test no winner with empty votes."""
        protocol = MockProtocol()
        voting = VotingPhase(protocol=protocol)

        winner = voting.determine_winner([])

        assert winner is None

    def test_determine_winner_tied(self):
        """Test winner selection when tied (first in sort order)."""
        protocol = MockProtocol()
        voting = VotingPhase(protocol=protocol)

        votes = [
            MockVote(agent="a", choice="A"),
            MockVote(agent="b", choice="B"),
        ]

        winner = voting.determine_winner(votes)

        # Either A or B could win, both have equal votes
        assert winner in ["A", "B"]


# =============================================================================
# Weighted Vote Counting Tests
# =============================================================================


class TestCountWeightedVotes:
    """Tests for weighted vote counting."""

    def test_count_votes_no_weighting(self):
        """Test counting votes without weighting."""
        protocol = MockProtocol(vote_grouping=False)
        voting = VotingPhase(protocol=protocol)

        votes = [
            MockVote(agent="a", choice="A"),
            MockVote(agent="b", choice="A"),
            MockVote(agent="c", choice="B"),
        ]

        result = voting.count_weighted_votes(votes)

        assert result.winner == "A"
        assert result.vote_counts["A"] == 2.0
        assert result.vote_counts["B"] == 1.0
        assert result.total_weighted_votes == 3.0

    def test_count_votes_with_weight_calculator(self):
        """Test counting votes with weight calculator."""
        protocol = MockProtocol(vote_grouping=False)
        voting = VotingPhase(protocol=protocol)

        weight_calc = VoteWeightCalculator(reliability_weights={"expert": 2.0, "novice": 0.5})

        votes = [
            MockVote(agent="expert", choice="A"),
            MockVote(agent="novice", choice="B"),
            MockVote(agent="novice", choice="B"),
        ]

        result = voting.count_weighted_votes(votes, weight_calculator=weight_calc)

        # Expert: 2.0 for A, Novices: 0.5 + 0.5 = 1.0 for B
        assert result.winner == "A"
        assert result.vote_counts["A"] == 2.0
        assert result.vote_counts["B"] == 1.0

    def test_count_votes_with_user_votes(self):
        """Test counting votes including user votes."""
        protocol = MockProtocol(vote_grouping=False)
        voting = VotingPhase(protocol=protocol)

        votes = [MockVote(agent="agent1", choice="A")]

        user_votes = [
            {"choice": "B", "user_id": "user1"},
            {"choice": "B", "user_id": "user2"},
        ]

        result = voting.count_weighted_votes(votes, user_votes=user_votes, user_vote_weight=0.5)

        # Agent: 1.0 for A, Users: 0.5 + 0.5 = 1.0 for B
        assert result.vote_counts["A"] == 1.0
        assert result.vote_counts["B"] == 1.0

    def test_count_votes_user_intensity_multiplier(self):
        """Test user vote intensity multiplier."""
        protocol = MockProtocol(vote_grouping=False)
        voting = VotingPhase(protocol=protocol)

        votes = [MockVote(agent="agent1", choice="A")]

        user_votes = [{"choice": "B", "intensity": 10}]

        # Intensity multiplier: doubles weight at intensity 10
        multiplier = lambda intensity, protocol: intensity / 5.0

        result = voting.count_weighted_votes(
            votes,
            user_votes=user_votes,
            user_vote_weight=0.5,
            user_vote_multiplier=multiplier,
        )

        # User: 0.5 * (10/5) = 1.0 for B
        assert result.vote_counts["B"] == 1.0

    def test_count_votes_empty(self):
        """Test counting empty votes."""
        protocol = MockProtocol()
        voting = VotingPhase(protocol=protocol)

        result = voting.count_weighted_votes([])

        assert result.winner is None
        assert result.total_weighted_votes == 0.0

    def test_count_votes_confidence_calculation(self):
        """Test confidence is calculated correctly."""
        protocol = MockProtocol(vote_grouping=False)
        voting = VotingPhase(protocol=protocol)

        votes = [
            MockVote(agent="a", choice="A"),
            MockVote(agent="b", choice="A"),
            MockVote(agent="c", choice="A"),
            MockVote(agent="d", choice="B"),
        ]

        result = voting.count_weighted_votes(votes)

        # Confidence = winner votes / total = 3/4 = 0.75
        assert result.confidence == pytest.approx(0.75)

    def test_count_votes_applies_grouping(self):
        """Test that vote grouping is applied."""
        protocol = MockProtocol(vote_grouping=True, vote_grouping_threshold=0.7)
        backend = MockSimilarityBackend(similarities={("Vector DB", "Use vector database"): 0.85})
        voting = VotingPhase(protocol=protocol, similarity_backend=backend)

        votes = [
            MockVote(agent="a", choice="Vector DB"),
            MockVote(agent="b", choice="Use vector database"),
        ]

        result = voting.count_weighted_votes(votes)

        # Both should be grouped under one choice
        assert len(result.vote_counts) == 1
        assert result.total_weighted_votes == 2.0


# =============================================================================
# Consensus Strength Tests
# =============================================================================


class TestConsensusStrength:
    """Tests for consensus strength calculation."""

    def test_consensus_strength_unanimous(self):
        """Test unanimous consensus strength."""
        protocol = MockProtocol()
        voting = VotingPhase(protocol=protocol)

        vote_counts = {"A": 5.0}

        result = voting.compute_consensus_strength(vote_counts, 5.0)

        assert result["strength"] == "unanimous"
        assert result["variance"] == 0.0

    def test_consensus_strength_strong(self):
        """Test strong consensus (low variance)."""
        protocol = MockProtocol()
        voting = VotingPhase(protocol=protocol)

        vote_counts = {"A": 4.0, "B": 4.0}

        result = voting.compute_consensus_strength(vote_counts, 8.0)

        assert result["strength"] == "strong"
        assert result["variance"] < 1

    def test_consensus_strength_medium(self):
        """Test medium consensus."""
        protocol = MockProtocol()
        voting = VotingPhase(protocol=protocol)

        vote_counts = {"A": 5.0, "B": 3.0}

        result = voting.compute_consensus_strength(vote_counts, 8.0)

        assert result["strength"] in ["strong", "medium"]

    def test_consensus_strength_weak(self):
        """Test weak consensus (high variance)."""
        protocol = MockProtocol()
        voting = VotingPhase(protocol=protocol)

        vote_counts = {"A": 8.0, "B": 2.0}

        result = voting.compute_consensus_strength(vote_counts, 10.0)

        # High variance = weak
        assert result["strength"] in ["medium", "weak"]

    def test_consensus_strength_empty(self):
        """Test consensus strength with empty counts."""
        protocol = MockProtocol()
        voting = VotingPhase(protocol=protocol)

        result = voting.compute_consensus_strength({}, 0.0)

        assert result["strength"] == "none"
        assert result["variance"] == 0.0


# =============================================================================
# Unanimous Check Tests
# =============================================================================


class TestCheckUnanimous:
    """Tests for unanimous consensus checking."""

    def test_check_unanimous_all_agree(self):
        """Test unanimous check when all agree."""
        protocol = MockProtocol(vote_grouping=False)
        voting = VotingPhase(protocol=protocol)

        votes = [
            MockVote(agent="a", choice="X"),
            MockVote(agent="b", choice="X"),
            MockVote(agent="c", choice="X"),
        ]

        result = voting.check_unanimous(votes)

        assert result.consensus_reached is True
        assert result.consensus_strength == "unanimous"
        assert result.winner == "X"
        assert result.confidence == 1.0

    def test_check_unanimous_with_dissent(self):
        """Test unanimous check with dissenting vote."""
        protocol = MockProtocol(vote_grouping=False)
        voting = VotingPhase(protocol=protocol)

        votes = [
            MockVote(agent="a", choice="X"),
            MockVote(agent="b", choice="X"),
            MockVote(agent="c", choice="Y"),
        ]

        result = voting.check_unanimous(votes)

        assert result.consensus_reached is False
        assert result.consensus_strength == "none"

    def test_check_unanimous_with_voting_errors(self):
        """Test unanimous check with voting errors."""
        protocol = MockProtocol(vote_grouping=False)
        voting = VotingPhase(protocol=protocol)

        votes = [
            MockVote(agent="a", choice="X"),
            MockVote(agent="b", choice="X"),
        ]

        result = voting.check_unanimous(votes, voting_errors=1)

        # Errors count as dissent
        assert result.consensus_reached is False
        assert result.confidence == pytest.approx(2 / 3)

    def test_check_unanimous_empty(self):
        """Test unanimous check with empty votes."""
        protocol = MockProtocol()
        voting = VotingPhase(protocol=protocol)

        result = voting.check_unanimous([])

        assert result.consensus_reached is False
        assert result.winner is None

    def test_check_unanimous_applies_grouping(self):
        """Test that unanimous check applies vote grouping."""
        protocol = MockProtocol(vote_grouping=True, vote_grouping_threshold=0.7)
        backend = MockSimilarityBackend(similarities={("Option A", "Choice A"): 0.9})
        voting = VotingPhase(protocol=protocol, similarity_backend=backend)

        votes = [
            MockVote(agent="a", choice="Option A"),
            MockVote(agent="b", choice="Choice A"),
        ]

        result = voting.check_unanimous(votes)

        # Both should be grouped, resulting in unanimity
        assert result.consensus_reached is True


# =============================================================================
# WeightedVoteResult Tests
# =============================================================================


class TestWeightedVoteResult:
    """Tests for WeightedVoteResult dataclass."""

    def test_default_values(self):
        """Test default values of WeightedVoteResult."""
        result = WeightedVoteResult()

        assert result.winner is None
        assert result.vote_counts == {}
        assert result.total_weighted_votes == 0.0
        assert result.confidence == 0.0
        assert result.consensus_reached is False
        assert result.consensus_strength == "none"
        assert result.consensus_variance == 0.0
        assert result.choice_mapping == {}

    def test_populated_result(self):
        """Test populated WeightedVoteResult."""
        result = WeightedVoteResult(
            winner="A",
            vote_counts={"A": 3.0, "B": 1.0},
            total_weighted_votes=4.0,
            confidence=0.75,
            consensus_reached=True,
            consensus_strength="strong",
            choice_mapping={"Option A": "A"},
        )

        assert result.winner == "A"
        assert result.confidence == 0.75
        assert result.consensus_reached is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestVotingIntegration:
    """Integration tests for the voting system."""

    def test_full_voting_workflow(self):
        """Test complete voting workflow."""
        protocol = MockProtocol(vote_grouping_threshold=0.7)
        backend = MockSimilarityBackend(similarities={("Use Python", "Python is best"): 0.8})
        voting = VotingPhase(protocol=protocol, similarity_backend=backend)

        # Create weight calculator
        weight_calc = VoteWeightCalculator(
            reputation_source=lambda name: 1.2 if "senior" in name else 1.0
        )

        # Votes with some groupable choices
        votes = [
            MockVote(agent="senior_dev", choice="Use Python", confidence=0.9),
            MockVote(agent="junior_dev", choice="Python is best", confidence=0.8),
            MockVote(agent="other_dev", choice="Use Java", confidence=0.7),
        ]

        # Count weighted votes
        result = voting.count_weighted_votes(votes, weight_calculator=weight_calc)

        # Python variants should be grouped
        assert result.winner is not None
        assert result.total_weighted_votes > 0
        assert result.confidence > 0

    def test_voting_with_user_participation(self):
        """Test voting with agent and user votes."""
        protocol = MockProtocol(vote_grouping=False)
        voting = VotingPhase(protocol=protocol)

        weight_calc = VoteWeightCalculator(reliability_weights={"agent1": 1.0, "agent2": 1.0})

        agent_votes = [
            MockVote(agent="agent1", choice="A"),
            MockVote(agent="agent2", choice="B"),
        ]

        user_votes = [
            {"choice": "A", "user_id": "user1", "intensity": 8},
            {"choice": "A", "user_id": "user2", "intensity": 6},
        ]

        intensity_multiplier = lambda intensity, protocol: intensity / 10.0

        result = voting.count_weighted_votes(
            agent_votes,
            weight_calculator=weight_calc,
            user_votes=user_votes,
            user_vote_weight=0.5,
            user_vote_multiplier=intensity_multiplier,
        )

        # A should win with combined agent + user votes
        assert result.winner == "A"

    def test_exception_votes_skipped(self):
        """Test that exception votes are skipped."""
        protocol = MockProtocol(vote_grouping=False)
        voting = VotingPhase(protocol=protocol)

        votes = [
            MockVote(agent="a", choice="A"),
            ValueError("Vote error"),  # type: ignore - testing exception handling
            MockVote(agent="c", choice="A"),
        ]

        result = voting.count_weighted_votes(votes)

        # Only valid votes should be counted
        assert result.vote_counts["A"] == 2.0


# =============================================================================
# Edge Cases
# =============================================================================


class TestVotingEdgeCases:
    """Tests for edge cases in voting."""

    def test_none_choice_skipped_in_distribution(self):
        """Test that None choices are skipped in distribution."""
        protocol = MockProtocol()
        voting = VotingPhase(protocol=protocol)

        votes = [
            MockVote(agent="a", choice="A"),
            MockVote(agent="b", choice=""),  # Empty choice
        ]

        dist = voting.compute_vote_distribution(votes)

        assert "A" in dist
        assert "" not in dist or dist.get("", {}).get("count", 0) == 0

    def test_single_voter(self):
        """Test voting with single voter."""
        protocol = MockProtocol(vote_grouping=False)
        voting = VotingPhase(protocol=protocol)

        votes = [MockVote(agent="solo", choice="Only Option")]

        result = voting.count_weighted_votes(votes)

        assert result.winner == "Only Option"
        assert result.confidence == 1.0

    def test_large_number_of_votes(self):
        """Test voting with many votes."""
        protocol = MockProtocol(vote_grouping=False)
        voting = VotingPhase(protocol=protocol)

        # Create 100 votes
        votes = [MockVote(agent=f"agent_{i}", choice="A" if i < 60 else "B") for i in range(100)]

        result = voting.count_weighted_votes(votes)

        assert result.winner == "A"
        assert result.vote_counts["A"] == 60.0
        assert result.vote_counts["B"] == 40.0

    def test_vote_grouping_with_many_variants(self):
        """Test grouping with many similar variants."""
        protocol = MockProtocol(vote_grouping=True, vote_grouping_threshold=0.6)

        # Create backend where all Python variants are similar
        similarities = {
            ("Python", "Use Python"): 0.8,
            ("Python", "Python is great"): 0.7,
            ("Use Python", "Python is great"): 0.75,
        }
        backend = MockSimilarityBackend(similarities=similarities)
        voting = VotingPhase(protocol=protocol, similarity_backend=backend)

        votes = [
            MockVote(agent="a", choice="Python"),
            MockVote(agent="b", choice="Use Python"),
            MockVote(agent="c", choice="Python is great"),
            MockVote(agent="d", choice="Java"),
        ]

        result = voting.count_weighted_votes(votes)

        # All Python variants should be grouped, giving 3 votes to winner
        assert result.total_weighted_votes == 4.0
        # The grouped Python variant should have 3 votes
        assert max(result.vote_counts.values()) == 3.0
