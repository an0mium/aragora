"""Tests for the voting phase module."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from collections import Counter


# -----------------------------------------------------------------------------
# WeightedVoteResult Tests
# -----------------------------------------------------------------------------


class TestWeightedVoteResult:
    """Test WeightedVoteResult dataclass."""

    def test_default_values(self):
        """WeightedVoteResult has correct defaults."""
        from aragora.debate.phases.voting import WeightedVoteResult

        result = WeightedVoteResult()

        assert result.winner is None
        assert result.vote_counts == {}
        assert result.total_weighted_votes == 0.0
        assert result.confidence == 0.0
        assert result.consensus_reached is False
        assert result.consensus_strength == "none"
        assert result.consensus_variance == 0.0
        assert result.choice_mapping == {}

    def test_custom_values(self):
        """WeightedVoteResult accepts custom values."""
        from aragora.debate.phases.voting import WeightedVoteResult

        result = WeightedVoteResult(
            winner="Option A",
            vote_counts={"Option A": 3.0, "Option B": 1.0},
            total_weighted_votes=4.0,
            confidence=0.75,
            consensus_reached=True,
            consensus_strength="strong",
            consensus_variance=0.5,
            choice_mapping={"opt_a": "Option A"},
        )

        assert result.winner == "Option A"
        assert result.vote_counts == {"Option A": 3.0, "Option B": 1.0}
        assert result.total_weighted_votes == 4.0
        assert result.confidence == 0.75
        assert result.consensus_reached is True
        assert result.consensus_strength == "strong"
        assert result.consensus_variance == 0.5


# -----------------------------------------------------------------------------
# VoteWeightCalculator Tests
# -----------------------------------------------------------------------------


class TestVoteWeightCalculator:
    """Test VoteWeightCalculator class."""

    def test_default_weight_is_one(self):
        """Without sources, weight is 1.0."""
        from aragora.debate.phases.voting import VoteWeightCalculator

        calc = VoteWeightCalculator()
        weight = calc.compute_weight("agent-1")

        assert weight == 1.0

    def test_reputation_source(self):
        """Reputation source multiplies weight."""
        from aragora.debate.phases.voting import VoteWeightCalculator

        reputation_fn = Mock(return_value=1.5)
        calc = VoteWeightCalculator(reputation_source=reputation_fn)

        weight = calc.compute_weight("agent-1")

        assert weight == 1.5
        reputation_fn.assert_called_once_with("agent-1")

    def test_reliability_weights(self):
        """Reliability weights multiply weight."""
        from aragora.debate.phases.voting import VoteWeightCalculator

        calc = VoteWeightCalculator(
            reliability_weights={"agent-1": 0.8, "agent-2": 1.2}
        )

        assert calc.compute_weight("agent-1") == 0.8
        assert calc.compute_weight("agent-2") == 1.2
        assert calc.compute_weight("agent-3") == 1.0  # Not in dict

    def test_consistency_source(self):
        """Consistency source maps 0-1 to 0.5-1.0 multiplier."""
        from aragora.debate.phases.voting import VoteWeightCalculator

        # Score 1.0 -> weight 1.0
        consistency_fn = Mock(return_value=1.0)
        calc = VoteWeightCalculator(consistency_source=consistency_fn)
        assert calc.compute_weight("agent-1") == 1.0

        # Score 0.0 -> weight 0.5
        calc.clear_cache()
        consistency_fn.return_value = 0.0
        assert calc.compute_weight("agent-1") == 0.5

        # Score 0.5 -> weight 0.75
        calc.clear_cache()
        consistency_fn.return_value = 0.5
        assert calc.compute_weight("agent-1") == 0.75

    def test_calibration_source(self):
        """Calibration source multiplies weight."""
        from aragora.debate.phases.voting import VoteWeightCalculator

        calibration_fn = Mock(return_value=1.2)
        calc = VoteWeightCalculator(calibration_source=calibration_fn)

        weight = calc.compute_weight("agent-1")

        assert weight == 1.2
        calibration_fn.assert_called_once_with("agent-1")

    def test_combined_weights(self):
        """Multiple sources multiply together."""
        from aragora.debate.phases.voting import VoteWeightCalculator

        calc = VoteWeightCalculator(
            reputation_source=lambda n: 1.5,
            reliability_weights={"agent-1": 0.8},
            consistency_source=lambda n: 1.0,  # -> 1.0 multiplier
            calibration_source=lambda n: 1.2,
        )

        # 1.0 * 1.5 * 0.8 * 1.0 * 1.2 = 1.44
        weight = calc.compute_weight("agent-1")
        assert abs(weight - 1.44) < 0.001

    def test_caching(self):
        """Weights are cached."""
        from aragora.debate.phases.voting import VoteWeightCalculator

        call_count = 0

        def counting_reputation(name):
            nonlocal call_count
            call_count += 1
            return 1.5

        calc = VoteWeightCalculator(reputation_source=counting_reputation)

        calc.compute_weight("agent-1")
        calc.compute_weight("agent-1")
        calc.compute_weight("agent-1")

        assert call_count == 1  # Only called once due to cache

    def test_clear_cache(self):
        """clear_cache empties the cache."""
        from aragora.debate.phases.voting import VoteWeightCalculator

        call_count = 0

        def counting_reputation(name):
            nonlocal call_count
            call_count += 1
            return 1.5

        calc = VoteWeightCalculator(reputation_source=counting_reputation)

        calc.compute_weight("agent-1")
        calc.clear_cache()
        calc.compute_weight("agent-1")

        assert call_count == 2  # Called twice after cache clear

    def test_compute_weights_batch(self):
        """compute_weights_batch returns dict of weights."""
        from aragora.debate.phases.voting import VoteWeightCalculator

        calc = VoteWeightCalculator(
            reliability_weights={"agent-1": 0.8, "agent-2": 1.2}
        )

        weights = calc.compute_weights_batch(["agent-1", "agent-2", "agent-3"])

        assert weights == {"agent-1": 0.8, "agent-2": 1.2, "agent-3": 1.0}

    def test_error_handling_reputation(self):
        """Errors in reputation source are handled gracefully."""
        from aragora.debate.phases.voting import VoteWeightCalculator

        def failing_reputation(name):
            raise ValueError("Test error")

        calc = VoteWeightCalculator(reputation_source=failing_reputation)

        # Should not raise, returns default weight
        weight = calc.compute_weight("agent-1")
        assert weight == 1.0

    def test_error_handling_consistency(self):
        """Errors in consistency source are handled gracefully."""
        from aragora.debate.phases.voting import VoteWeightCalculator

        def failing_consistency(name):
            raise ValueError("Test error")

        calc = VoteWeightCalculator(consistency_source=failing_consistency)

        weight = calc.compute_weight("agent-1")
        assert weight == 1.0

    def test_error_handling_calibration(self):
        """Errors in calibration source are handled gracefully."""
        from aragora.debate.phases.voting import VoteWeightCalculator

        def failing_calibration(name):
            raise ValueError("Test error")

        calc = VoteWeightCalculator(calibration_source=failing_calibration)

        weight = calc.compute_weight("agent-1")
        assert weight == 1.0


# -----------------------------------------------------------------------------
# VotingPhase Tests
# -----------------------------------------------------------------------------


class TestVotingPhaseInit:
    """Test VotingPhase initialization."""

    @pytest.fixture
    def mock_protocol(self):
        """Create mock debate protocol."""
        protocol = Mock()
        protocol.vote_grouping = True
        protocol.vote_grouping_threshold = 0.8
        return protocol

    def test_init_with_protocol(self, mock_protocol):
        """Can initialize with protocol."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)

        assert phase.protocol is mock_protocol
        assert phase._similarity_backend is None

    def test_init_with_similarity_backend(self, mock_protocol):
        """Can initialize with custom similarity backend."""
        from aragora.debate.phases.voting import VotingPhase

        mock_backend = Mock()
        phase = VotingPhase(protocol=mock_protocol, similarity_backend=mock_backend)

        assert phase._similarity_backend is mock_backend


# -----------------------------------------------------------------------------
# VotingPhase Vote Grouping Tests
# -----------------------------------------------------------------------------


class TestVoteGrouping:
    """Test vote grouping functionality."""

    @pytest.fixture
    def mock_protocol(self):
        protocol = Mock()
        protocol.vote_grouping = True
        protocol.vote_grouping_threshold = 0.8
        return protocol

    @pytest.fixture
    def mock_vote(self):
        """Factory for mock votes."""
        def _make_vote(agent: str, choice: str, confidence: float = None):
            vote = Mock(spec=["agent", "choice", "reasoning", "confidence"])
            vote.agent = agent
            vote.choice = choice
            vote.reasoning = "Test reasoning"
            vote.confidence = confidence
            return vote
        return _make_vote

    def test_grouping_disabled(self, mock_vote):
        """Returns empty dict when grouping disabled."""
        from aragora.debate.phases.voting import VotingPhase

        protocol = Mock()
        protocol.vote_grouping = False

        phase = VotingPhase(protocol=protocol)
        votes = [mock_vote("a1", "Option A"), mock_vote("a2", "Option B")]

        result = phase.group_similar_votes(votes)

        assert result == {}

    def test_grouping_empty_votes(self, mock_protocol):
        """Returns empty dict for empty votes."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        result = phase.group_similar_votes([])

        assert result == {}

    def test_grouping_single_choice(self, mock_protocol, mock_vote):
        """Returns empty dict when only one choice."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        votes = [
            mock_vote("a1", "Option A"),
            mock_vote("a2", "Option A"),
        ]

        result = phase.group_similar_votes(votes)

        assert result == {}

    def test_grouping_similar_choices(self, mock_protocol, mock_vote):
        """Groups similar choices together."""
        from aragora.debate.phases.voting import VotingPhase

        mock_backend = Mock()
        mock_backend.compute_similarity.return_value = 0.9  # Above threshold

        phase = VotingPhase(protocol=mock_protocol, similarity_backend=mock_backend)
        votes = [
            mock_vote("a1", "Use Vector DB"),
            mock_vote("a2", "Vector Database"),
        ]

        result = phase.group_similar_votes(votes)

        # One group with both choices
        assert len(result) == 1
        canonical = list(result.keys())[0]
        assert len(result[canonical]) == 2

    def test_grouping_dissimilar_choices(self, mock_protocol, mock_vote):
        """Does not group dissimilar choices."""
        from aragora.debate.phases.voting import VotingPhase

        mock_backend = Mock()
        mock_backend.compute_similarity.return_value = 0.3  # Below threshold

        phase = VotingPhase(protocol=mock_protocol, similarity_backend=mock_backend)
        votes = [
            mock_vote("a1", "Option A"),
            mock_vote("a2", "Option B"),
        ]

        result = phase.group_similar_votes(votes)

        # No groups (no merges)
        assert result == {}


class TestApplyVoteGrouping:
    """Test applying vote grouping to normalize votes."""

    @pytest.fixture
    def mock_protocol(self):
        protocol = Mock()
        protocol.vote_grouping = True
        protocol.vote_grouping_threshold = 0.8
        return protocol

    @pytest.fixture
    def mock_vote(self):
        def _make_vote(agent: str, choice: str, confidence: float = None):
            vote = Mock(spec=["agent", "choice", "reasoning", "confidence"])
            vote.agent = agent
            vote.choice = choice
            vote.reasoning = "Test"
            vote.confidence = confidence
            return vote
        return _make_vote

    def test_no_groups(self, mock_protocol, mock_vote):
        """Returns original votes when no groups."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        votes = [mock_vote("a1", "Option A")]

        result = phase.apply_vote_grouping(votes, {})

        assert result is votes

    def test_applies_grouping(self, mock_protocol, mock_vote):
        """Normalizes votes according to groups."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        votes = [
            mock_vote("a1", "Vector DB"),
            mock_vote("a2", "Vector Database"),
        ]
        groups = {"Vector DB": ["Vector DB", "Vector Database"]}

        result = phase.apply_vote_grouping(votes, groups)

        # Both votes should now have canonical choice
        assert result[0].choice == "Vector DB"
        assert result[1].choice == "Vector DB"


# -----------------------------------------------------------------------------
# VotingPhase Vote Distribution Tests
# -----------------------------------------------------------------------------


class TestComputeVoteDistribution:
    """Test vote distribution computation."""

    @pytest.fixture
    def mock_protocol(self):
        protocol = Mock()
        protocol.vote_grouping = False
        return protocol

    @pytest.fixture
    def mock_vote(self):
        def _make_vote(agent: str, choice: str, confidence: float = None):
            vote = Mock(spec=["agent", "choice", "reasoning", "confidence"])
            vote.agent = agent
            vote.choice = choice
            vote.reasoning = "Test"
            vote.confidence = confidence
            return vote
        return _make_vote

    def test_empty_votes(self, mock_protocol):
        """Returns empty dict for no votes."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        result = phase.compute_vote_distribution([])

        assert result == {}

    def test_single_choice(self, mock_protocol, mock_vote):
        """Computes distribution for single choice."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        votes = [
            mock_vote("a1", "Option A"),
            mock_vote("a2", "Option A"),
            mock_vote("a3", "Option A"),
        ]

        result = phase.compute_vote_distribution(votes)

        assert "Option A" in result
        assert result["Option A"]["count"] == 3
        assert result["Option A"]["percentage"] == 100.0
        assert len(result["Option A"]["voters"]) == 3

    def test_multiple_choices(self, mock_protocol, mock_vote):
        """Computes distribution for multiple choices."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        votes = [
            mock_vote("a1", "Option A"),
            mock_vote("a2", "Option A"),
            mock_vote("a3", "Option B"),
            mock_vote("a4", "Option C"),
        ]

        result = phase.compute_vote_distribution(votes)

        assert result["Option A"]["count"] == 2
        assert result["Option A"]["percentage"] == 50.0
        assert result["Option B"]["count"] == 1
        assert result["Option B"]["percentage"] == 25.0
        assert result["Option C"]["count"] == 1
        assert result["Option C"]["percentage"] == 25.0

    def test_confidence_averaging(self, mock_protocol, mock_vote):
        """Computes average confidence."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        votes = [
            mock_vote("a1", "Option A", confidence=0.8),
            mock_vote("a2", "Option A", confidence=0.6),
        ]

        result = phase.compute_vote_distribution(votes)

        assert result["Option A"]["avg_confidence"] == 0.7


# -----------------------------------------------------------------------------
# VotingPhase Determine Winner Tests
# -----------------------------------------------------------------------------


class TestDetermineWinner:
    """Test winner determination."""

    @pytest.fixture
    def mock_protocol(self):
        protocol = Mock()
        protocol.vote_grouping = False
        return protocol

    @pytest.fixture
    def mock_vote(self):
        def _make_vote(agent: str, choice: str, confidence: float = None):
            vote = Mock(spec=["agent", "choice", "reasoning", "confidence"])
            vote.agent = agent
            vote.choice = choice
            vote.reasoning = "Test"
            vote.confidence = confidence
            return vote
        return _make_vote

    def test_no_votes(self, mock_protocol):
        """Returns None for no votes."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        result = phase.determine_winner([])

        assert result is None

    def test_clear_winner(self, mock_protocol, mock_vote):
        """Returns winner with most votes."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        votes = [
            mock_vote("a1", "Option A"),
            mock_vote("a2", "Option A"),
            mock_vote("a3", "Option B"),
        ]

        result = phase.determine_winner(votes)

        assert result == "Option A"

    def test_require_majority_passes(self, mock_protocol, mock_vote):
        """Returns winner when majority achieved."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        votes = [
            mock_vote("a1", "Option A"),
            mock_vote("a2", "Option A"),
            mock_vote("a3", "Option A"),
            mock_vote("a4", "Option B"),
        ]

        result = phase.determine_winner(votes, require_majority=True)

        assert result == "Option A"  # 75% > 50%

    def test_require_majority_fails(self, mock_protocol, mock_vote):
        """Returns None when majority not achieved."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        votes = [
            mock_vote("a1", "Option A"),
            mock_vote("a2", "Option A"),
            mock_vote("a3", "Option B"),
            mock_vote("a4", "Option C"),
        ]

        result = phase.determine_winner(votes, require_majority=True)

        assert result is None  # 50% not > 50%

    def test_min_margin_passes(self, mock_protocol, mock_vote):
        """Returns winner when margin achieved."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        votes = [
            mock_vote("a1", "Option A"),
            mock_vote("a2", "Option A"),
            mock_vote("a3", "Option A"),
            mock_vote("a4", "Option B"),
        ]

        result = phase.determine_winner(votes, min_margin=0.3)

        assert result == "Option A"  # 75% - 25% = 50% margin

    def test_min_margin_fails(self, mock_protocol, mock_vote):
        """Returns None when margin not achieved."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        votes = [
            mock_vote("a1", "Option A"),
            mock_vote("a2", "Option A"),
            mock_vote("a3", "Option B"),
            mock_vote("a4", "Option B"),
        ]

        result = phase.determine_winner(votes, min_margin=0.3)

        assert result is None  # 50% - 50% = 0% margin


# -----------------------------------------------------------------------------
# VotingPhase Weighted Vote Counting Tests
# -----------------------------------------------------------------------------


class TestCountWeightedVotes:
    """Test weighted vote counting."""

    @pytest.fixture
    def mock_protocol(self):
        protocol = Mock()
        protocol.vote_grouping = False
        return protocol

    @pytest.fixture
    def mock_vote(self):
        def _make_vote(agent: str, choice: str, confidence: float = None):
            vote = Mock(spec=["agent", "choice", "reasoning", "confidence"])
            vote.agent = agent
            vote.choice = choice
            vote.reasoning = "Test"
            vote.confidence = confidence
            return vote
        return _make_vote

    def test_no_votes(self, mock_protocol):
        """Returns empty result for no votes."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        result = phase.count_weighted_votes([])

        assert result.winner is None
        assert result.vote_counts == {}
        assert result.total_weighted_votes == 0.0

    def test_unweighted_votes(self, mock_protocol, mock_vote):
        """Counts votes without weights."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        votes = [
            mock_vote("a1", "Option A"),
            mock_vote("a2", "Option A"),
            mock_vote("a3", "Option B"),
        ]

        result = phase.count_weighted_votes(votes)

        assert result.winner == "Option A"
        assert result.vote_counts["Option A"] == 2.0
        assert result.vote_counts["Option B"] == 1.0
        assert result.total_weighted_votes == 3.0
        assert abs(result.confidence - 2/3) < 0.001

    def test_weighted_votes(self, mock_protocol, mock_vote):
        """Counts votes with weights."""
        from aragora.debate.phases.voting import VotingPhase, VoteWeightCalculator

        phase = VotingPhase(protocol=mock_protocol)
        calculator = VoteWeightCalculator(
            reliability_weights={"a1": 2.0, "a2": 0.5, "a3": 1.0}
        )

        votes = [
            mock_vote("a1", "Option A"),  # 2.0 weight
            mock_vote("a2", "Option B"),  # 0.5 weight
            mock_vote("a3", "Option B"),  # 1.0 weight
        ]

        result = phase.count_weighted_votes(votes, weight_calculator=calculator)

        assert result.winner == "Option A"  # 2.0 > 1.5
        assert result.vote_counts["Option A"] == 2.0
        assert result.vote_counts["Option B"] == 1.5
        assert result.total_weighted_votes == 3.5

    def test_user_votes(self, mock_protocol, mock_vote):
        """Includes user votes."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        votes = [mock_vote("a1", "Option A")]

        user_votes = [
            {"choice": "Option B", "user_id": "user1"},
            {"choice": "Option B", "user_id": "user2"},
        ]

        result = phase.count_weighted_votes(
            votes,
            user_votes=user_votes,
            user_vote_weight=0.5,
        )

        assert result.vote_counts["Option A"] == 1.0
        assert result.vote_counts["Option B"] == 1.0  # 2 * 0.5

    def test_user_vote_multiplier(self, mock_protocol, mock_vote):
        """Applies user vote multiplier."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        votes = [mock_vote("a1", "Option A")]

        user_votes = [{"choice": "Option B", "intensity": 10}]

        def intensity_multiplier(intensity, protocol):
            return intensity / 5  # intensity 10 -> 2.0x

        result = phase.count_weighted_votes(
            votes,
            user_votes=user_votes,
            user_vote_weight=0.5,
            user_vote_multiplier=intensity_multiplier,
        )

        assert result.vote_counts["Option B"] == 1.0  # 0.5 * 2.0

    def test_skips_exception_votes(self, mock_protocol, mock_vote):
        """Skips votes that are exceptions."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        votes = [
            mock_vote("a1", "Option A"),
            ValueError("Test error"),  # Should be skipped
        ]

        result = phase.count_weighted_votes(votes)

        assert result.vote_counts["Option A"] == 1.0
        assert result.total_weighted_votes == 1.0


# -----------------------------------------------------------------------------
# VotingPhase Consensus Strength Tests
# -----------------------------------------------------------------------------


class TestComputeConsensusStrength:
    """Test consensus strength computation."""

    @pytest.fixture
    def mock_protocol(self):
        protocol = Mock()
        protocol.vote_grouping = False
        return protocol

    def test_no_votes(self, mock_protocol):
        """Returns 'none' for no votes."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        result = phase.compute_consensus_strength({}, 0)

        assert result["strength"] == "none"
        assert result["variance"] == 0.0

    def test_unanimous(self, mock_protocol):
        """Returns 'unanimous' for single choice."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        result = phase.compute_consensus_strength({"Option A": 5.0}, 5.0)

        assert result["strength"] == "unanimous"
        assert result["variance"] == 0.0

    def test_strong_consensus(self, mock_protocol):
        """Returns 'strong' for low variance."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        # Variance < 1
        result = phase.compute_consensus_strength({"A": 4.5, "B": 4.0}, 8.5)

        assert result["strength"] == "strong"
        assert result["variance"] < 1

    def test_medium_consensus(self, mock_protocol):
        """Returns 'medium' for moderate variance."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        # Variance 1-2
        result = phase.compute_consensus_strength({"A": 5.0, "B": 3.0}, 8.0)

        assert result["strength"] == "medium"
        assert 1 <= result["variance"] < 2

    def test_weak_consensus(self, mock_protocol):
        """Returns 'weak' for high variance."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        # Variance >= 2
        result = phase.compute_consensus_strength({"A": 8.0, "B": 2.0}, 10.0)

        assert result["strength"] == "weak"
        assert result["variance"] >= 2


# -----------------------------------------------------------------------------
# VotingPhase Check Unanimous Tests
# -----------------------------------------------------------------------------


class TestCheckUnanimous:
    """Test unanimous consensus checking."""

    @pytest.fixture
    def mock_protocol(self):
        protocol = Mock()
        protocol.vote_grouping = False
        return protocol

    @pytest.fixture
    def mock_vote(self):
        def _make_vote(agent: str, choice: str, confidence: float = None):
            vote = Mock(spec=["agent", "choice", "reasoning", "confidence"])
            vote.agent = agent
            vote.choice = choice
            vote.reasoning = "Test"
            vote.confidence = confidence
            return vote
        return _make_vote

    def test_no_votes(self, mock_protocol):
        """Returns empty result for no votes."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        result = phase.check_unanimous([])

        assert result.winner is None
        assert result.consensus_reached is False

    def test_unanimous_all_same(self, mock_protocol, mock_vote):
        """Returns unanimous consensus when all votes same."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        votes = [
            mock_vote("a1", "Option A"),
            mock_vote("a2", "Option A"),
            mock_vote("a3", "Option A"),
        ]

        result = phase.check_unanimous(votes)

        assert result.winner == "Option A"
        assert result.consensus_reached is True
        assert result.consensus_strength == "unanimous"
        assert result.confidence == 1.0

    def test_not_unanimous_with_dissent(self, mock_protocol, mock_vote):
        """Returns not unanimous when votes differ."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        votes = [
            mock_vote("a1", "Option A"),
            mock_vote("a2", "Option A"),
            mock_vote("a3", "Option B"),
        ]

        result = phase.check_unanimous(votes)

        assert result.winner == "Option A"
        assert result.consensus_reached is False
        assert result.consensus_strength == "none"
        assert result.confidence < 1.0

    def test_voting_errors_count_as_dissent(self, mock_protocol, mock_vote):
        """Voting errors prevent unanimity."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        votes = [
            mock_vote("a1", "Option A"),
            mock_vote("a2", "Option A"),
        ]

        result = phase.check_unanimous(votes, voting_errors=1)

        assert result.consensus_reached is False
        assert result.confidence < 1.0  # 2/3 not unanimous

    def test_skips_exception_votes(self, mock_protocol, mock_vote):
        """Exception votes are skipped."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        votes = [
            mock_vote("a1", "Option A"),
            ValueError("Test error"),
        ]

        result = phase.check_unanimous(votes)

        assert result.vote_counts["Option A"] == 1
        # Exception is skipped, only 1 valid vote
        assert result.total_weighted_votes == 1.0


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestVotingIntegration:
    """Integration tests for voting workflow."""

    @pytest.fixture
    def mock_protocol(self):
        protocol = Mock()
        protocol.vote_grouping = False
        return protocol

    @pytest.fixture
    def mock_vote(self):
        def _make_vote(agent: str, choice: str, confidence: float = None):
            vote = Mock(spec=["agent", "choice", "reasoning", "confidence"])
            vote.agent = agent
            vote.choice = choice
            vote.reasoning = "Test reasoning"
            vote.confidence = confidence
            return vote
        return _make_vote

    def test_full_voting_workflow(self, mock_protocol, mock_vote):
        """Test complete voting workflow."""
        from aragora.debate.phases.voting import VotingPhase, VoteWeightCalculator

        # Setup
        phase = VotingPhase(protocol=mock_protocol)
        calculator = VoteWeightCalculator(
            reliability_weights={"expert": 1.5, "junior": 0.8, "standard": 1.0}
        )

        votes = [
            mock_vote("expert", "Implement caching"),
            mock_vote("junior", "Implement caching"),
            mock_vote("standard", "Add more tests"),
        ]

        # 1. Compute distribution
        distribution = phase.compute_vote_distribution(votes)
        assert len(distribution) == 2

        # 2. Determine unweighted winner
        winner = phase.determine_winner(votes)
        assert winner == "Implement caching"

        # 3. Count weighted votes
        result = phase.count_weighted_votes(votes, weight_calculator=calculator)
        assert result.winner == "Implement caching"
        assert result.vote_counts["Implement caching"] == 2.3  # 1.5 + 0.8
        assert result.vote_counts["Add more tests"] == 1.0

    def test_tie_resolution(self, mock_protocol, mock_vote):
        """Test handling of tied votes."""
        from aragora.debate.phases.voting import VotingPhase

        phase = VotingPhase(protocol=mock_protocol)
        votes = [
            mock_vote("a1", "Option A"),
            mock_vote("a2", "Option B"),
        ]

        # With min_margin, should return None for tie
        winner = phase.determine_winner(votes, min_margin=0.1)
        assert winner is None

        # Without min_margin, returns one of them
        winner = phase.determine_winner(votes)
        assert winner in ["Option A", "Option B"]
