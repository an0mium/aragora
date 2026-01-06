"""
Tests for debate protocol configuration.

Tests DebateProtocol dataclass defaults, validation, and
user_vote_multiplier conviction-weighted voting calculation.
"""

import pytest
from dataclasses import fields

from aragora.debate.protocol import DebateProtocol, user_vote_multiplier


# =============================================================================
# DebateProtocol Tests
# =============================================================================

class TestDebateProtocol:
    """Tests for DebateProtocol configuration."""

    def test_default_values(self):
        """Default values should be sensible."""
        protocol = DebateProtocol()

        # Core defaults
        assert protocol.topology == "round-robin"
        assert protocol.rounds == 5
        assert protocol.consensus == "majority"
        assert protocol.consensus_threshold == 0.6

        # Role and voting defaults
        assert protocol.allow_abstain is True
        assert protocol.require_reasoning is True
        assert protocol.proposer_count == 1
        assert protocol.critic_count == -1  # All non-proposers

        # Early stopping defaults
        assert protocol.early_stopping is True
        assert protocol.early_stop_threshold == 0.66
        assert protocol.min_rounds_before_early_stop == 2

        # Convergence defaults
        assert protocol.convergence_detection is True
        assert protocol.convergence_threshold == 0.85
        assert protocol.divergence_threshold == 0.40

        # Role rotation defaults
        assert protocol.role_rotation is True
        assert protocol.role_rotation_config is None

    def test_custom_configuration(self):
        """Should accept custom configuration values."""
        protocol = DebateProtocol(
            topology="star",
            rounds=10,
            consensus="unanimous",
            consensus_threshold=0.9,
            agreement_intensity=2,
            early_stopping=False,
            role_rotation=False,
        )

        assert protocol.topology == "star"
        assert protocol.rounds == 10
        assert protocol.consensus == "unanimous"
        assert protocol.consensus_threshold == 0.9
        assert protocol.agreement_intensity == 2
        assert protocol.early_stopping is False
        assert protocol.role_rotation is False

    def test_topology_options(self):
        """Should support all valid topology options."""
        valid_topologies = [
            "all-to-all",
            "sparse",
            "round-robin",
            "ring",
            "star",
            "random-graph",
        ]

        for topology in valid_topologies:
            protocol = DebateProtocol(topology=topology)
            assert protocol.topology == topology

    def test_consensus_options(self):
        """Should support all valid consensus modes."""
        valid_consensus = ["majority", "unanimous", "judge", "none"]

        for consensus in valid_consensus:
            protocol = DebateProtocol(consensus=consensus)
            assert protocol.consensus == consensus

    def test_judge_selection_options(self):
        """Should support all valid judge selection modes."""
        valid_selections = ["random", "voted", "last", "elo_ranked"]

        for selection in valid_selections:
            protocol = DebateProtocol(judge_selection=selection)
            assert protocol.judge_selection == selection

    def test_user_vote_intensity_parameters(self):
        """User vote intensity parameters should have valid defaults."""
        protocol = DebateProtocol()

        assert protocol.user_vote_intensity_scale == 10
        assert protocol.user_vote_intensity_neutral == 5
        assert protocol.user_vote_intensity_min_multiplier == 0.5
        assert protocol.user_vote_intensity_max_multiplier == 2.0
        assert protocol.user_vote_weight == 0.5

    def test_timeout_defaults(self):
        """Timeout parameters should have sensible defaults."""
        protocol = DebateProtocol()

        assert protocol.timeout_seconds == 0  # No timeout by default
        assert protocol.round_timeout_seconds == 120  # 2 minutes per round


# =============================================================================
# User Vote Multiplier Tests
# =============================================================================

class TestUserVoteMultiplier:
    """Tests for conviction-weighted vote multiplier calculation."""

    @pytest.fixture
    def default_protocol(self):
        """Default protocol for testing."""
        return DebateProtocol()

    def test_neutral_intensity_returns_one(self, default_protocol):
        """Neutral intensity (5) should return multiplier of 1.0."""
        multiplier = user_vote_multiplier(5, default_protocol)
        assert multiplier == 1.0

    def test_minimum_intensity_returns_min_multiplier(self, default_protocol):
        """Minimum intensity (1) should return min_multiplier (0.5)."""
        multiplier = user_vote_multiplier(1, default_protocol)
        assert multiplier == pytest.approx(0.5, rel=0.01)

    def test_maximum_intensity_returns_max_multiplier(self, default_protocol):
        """Maximum intensity (10) should return max_multiplier (2.0)."""
        multiplier = user_vote_multiplier(10, default_protocol)
        assert multiplier == pytest.approx(2.0, rel=0.01)

    def test_below_neutral_interpolation(self, default_protocol):
        """Values below neutral should interpolate between min and 1.0."""
        # intensity=3 is halfway between 1 and 5
        # Should be halfway between 0.5 and 1.0 = 0.75
        multiplier = user_vote_multiplier(3, default_protocol)
        assert multiplier == pytest.approx(0.75, rel=0.01)

    def test_above_neutral_interpolation(self, default_protocol):
        """Values above neutral should interpolate between 1.0 and max."""
        # intensity=7.5 is halfway between 5 and 10
        # Should be halfway between 1.0 and 2.0 = 1.5
        # But intensity is int, so use 8 (60% between 5-10)
        multiplier = user_vote_multiplier(8, default_protocol)
        # (8-5)/(10-5) = 0.6, so 1.0 + 0.6 * (2.0 - 1.0) = 1.6
        assert multiplier == pytest.approx(1.6, rel=0.01)

    def test_intensity_clamped_to_range(self, default_protocol):
        """Intensity should be clamped to valid range."""
        # Below minimum
        multiplier_low = user_vote_multiplier(0, default_protocol)
        assert multiplier_low == user_vote_multiplier(1, default_protocol)

        # Above maximum
        multiplier_high = user_vote_multiplier(15, default_protocol)
        assert multiplier_high == user_vote_multiplier(10, default_protocol)

    def test_custom_protocol_parameters(self):
        """Should respect custom protocol parameters."""
        protocol = DebateProtocol(
            user_vote_intensity_scale=5,
            user_vote_intensity_neutral=3,
            user_vote_intensity_min_multiplier=0.1,
            user_vote_intensity_max_multiplier=3.0,
        )

        # Neutral should still return 1.0
        assert user_vote_multiplier(3, protocol) == 1.0

        # Min should return custom min_multiplier
        assert user_vote_multiplier(1, protocol) == pytest.approx(0.1, rel=0.01)

        # Max should return custom max_multiplier
        assert user_vote_multiplier(5, protocol) == pytest.approx(3.0, rel=0.01)

    def test_monotonic_increasing(self, default_protocol):
        """Multiplier should increase monotonically with intensity."""
        prev_multiplier = 0
        for intensity in range(1, 11):
            multiplier = user_vote_multiplier(intensity, default_protocol)
            assert multiplier >= prev_multiplier
            prev_multiplier = multiplier
