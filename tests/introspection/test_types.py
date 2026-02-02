"""
Tests for aragora.introspection.types module.

Tests cover:
- IntrospectionSnapshot dataclass
- Computed properties (proposal_acceptance_rate, critique_effectiveness, calibration_label)
- to_prompt_section() formatting
- to_dict() serialization
"""

from __future__ import annotations

import pytest

from aragora.introspection.types import IntrospectionSnapshot


class TestIntrospectionSnapshotDefaults:
    """Test IntrospectionSnapshot default values."""

    def test_defaults(self):
        """Snapshot should have sensible defaults."""
        snapshot = IntrospectionSnapshot(agent_name="test")

        assert snapshot.agent_name == "test"
        assert snapshot.reputation_score == 0.0
        assert snapshot.vote_weight == 1.0
        assert snapshot.proposals_made == 0
        assert snapshot.proposals_accepted == 0
        assert snapshot.critiques_given == 0
        assert snapshot.critiques_valuable == 0
        assert snapshot.calibration_score == 0.5
        assert snapshot.debate_count == 0
        assert snapshot.top_expertise == []
        assert snapshot.traits == []


class TestProposalAcceptanceRate:
    """Test proposal_acceptance_rate property."""

    def test_zero_proposals(self):
        """Rate should be 0.0 when no proposals made."""
        snapshot = IntrospectionSnapshot(agent_name="test", proposals_made=0)
        assert snapshot.proposal_acceptance_rate == 0.0

    def test_all_accepted(self):
        """Rate should be 1.0 when all proposals accepted."""
        snapshot = IntrospectionSnapshot(
            agent_name="test", proposals_made=10, proposals_accepted=10
        )
        assert snapshot.proposal_acceptance_rate == 1.0

    def test_partial_acceptance(self):
        """Rate should reflect partial acceptance."""
        snapshot = IntrospectionSnapshot(agent_name="test", proposals_made=10, proposals_accepted=7)
        assert snapshot.proposal_acceptance_rate == 0.7


class TestCritiqueEffectiveness:
    """Test critique_effectiveness property."""

    def test_zero_critiques(self):
        """Effectiveness should be 0.0 when no critiques given."""
        snapshot = IntrospectionSnapshot(agent_name="test", critiques_given=0)
        assert snapshot.critique_effectiveness == 0.0

    def test_all_valuable(self):
        """Effectiveness should be 1.0 when all critiques valuable."""
        snapshot = IntrospectionSnapshot(agent_name="test", critiques_given=5, critiques_valuable=5)
        assert snapshot.critique_effectiveness == 1.0

    def test_partial_valuable(self):
        """Effectiveness should reflect partial value."""
        snapshot = IntrospectionSnapshot(
            agent_name="test", critiques_given=10, critiques_valuable=6
        )
        assert snapshot.critique_effectiveness == 0.6


class TestCalibrationLabel:
    """Test calibration_label property."""

    def test_excellent(self):
        """Score >= 0.7 should be 'excellent'."""
        snapshot = IntrospectionSnapshot(agent_name="test", calibration_score=0.8)
        assert snapshot.calibration_label == "excellent"

        snapshot = IntrospectionSnapshot(agent_name="test", calibration_score=0.7)
        assert snapshot.calibration_label == "excellent"

    def test_good(self):
        """Score >= 0.5 and < 0.7 should be 'good'."""
        snapshot = IntrospectionSnapshot(agent_name="test", calibration_score=0.6)
        assert snapshot.calibration_label == "good"

        snapshot = IntrospectionSnapshot(agent_name="test", calibration_score=0.5)
        assert snapshot.calibration_label == "good"

    def test_fair(self):
        """Score >= 0.3 and < 0.5 should be 'fair'."""
        snapshot = IntrospectionSnapshot(agent_name="test", calibration_score=0.4)
        assert snapshot.calibration_label == "fair"

        snapshot = IntrospectionSnapshot(agent_name="test", calibration_score=0.3)
        assert snapshot.calibration_label == "fair"

    def test_developing(self):
        """Score < 0.3 should be 'developing'."""
        snapshot = IntrospectionSnapshot(agent_name="test", calibration_score=0.2)
        assert snapshot.calibration_label == "developing"

        snapshot = IntrospectionSnapshot(agent_name="test", calibration_score=0.0)
        assert snapshot.calibration_label == "developing"


class TestToPromptSection:
    """Test to_prompt_section() method."""

    def test_basic_format(self):
        """Output should include header and basic stats."""
        snapshot = IntrospectionSnapshot(
            agent_name="claude", reputation_score=0.85, vote_weight=1.2
        )

        result = snapshot.to_prompt_section()

        assert "## YOUR TRACK RECORD" in result
        assert "Reputation: 85%" in result
        assert "Vote weight: 1.2x" in result

    def test_includes_proposals_when_present(self):
        """Output should include proposals when made."""
        snapshot = IntrospectionSnapshot(
            agent_name="claude",
            proposals_made=10,
            proposals_accepted=7,
        )

        result = snapshot.to_prompt_section()

        assert "Proposals: 7/10 accepted (70%)" in result

    def test_excludes_proposals_when_zero(self):
        """Output should exclude proposals section when none made."""
        snapshot = IntrospectionSnapshot(agent_name="claude")

        result = snapshot.to_prompt_section()

        assert "Proposals:" not in result

    def test_includes_critiques_when_present(self):
        """Output should include critiques when given."""
        snapshot = IntrospectionSnapshot(
            agent_name="claude",
            critiques_given=10,
            critiques_valuable=8,
            calibration_score=0.75,
        )

        result = snapshot.to_prompt_section()

        assert "Critiques: 80% valuable" in result
        assert "Calibration: excellent" in result

    def test_includes_expertise(self):
        """Output should include top expertise."""
        snapshot = IntrospectionSnapshot(
            agent_name="claude", top_expertise=["security", "testing", "api_design"]
        )

        result = snapshot.to_prompt_section()

        assert "Expertise: security, testing, api_design" in result

    def test_includes_traits(self):
        """Output should include style traits."""
        snapshot = IntrospectionSnapshot(
            agent_name="claude", traits=["analytical", "thorough", "cautious"]
        )

        result = snapshot.to_prompt_section()

        assert "Style: analytical, thorough, cautious" in result

    def test_respects_max_chars(self):
        """Output should not exceed max_chars."""
        snapshot = IntrospectionSnapshot(
            agent_name="claude",
            reputation_score=0.9,
            proposals_made=100,
            proposals_accepted=90,
            critiques_given=50,
            critiques_valuable=45,
            top_expertise=["a", "b", "c"],
            traits=["x", "y", "z"],
        )

        result = snapshot.to_prompt_section(max_chars=100)

        assert len(result) <= 100


class TestToDict:
    """Test to_dict() method."""

    def test_serializes_all_fields(self):
        """to_dict should include all fields."""
        snapshot = IntrospectionSnapshot(
            agent_name="claude",
            reputation_score=0.85,
            vote_weight=1.3,
            proposals_made=10,
            proposals_accepted=8,
            critiques_given=5,
            critiques_valuable=4,
            calibration_score=0.7,
            debate_count=15,
            top_expertise=["security"],
            traits=["analytical"],
        )

        result = snapshot.to_dict()

        assert result["agent_name"] == "claude"
        assert result["reputation_score"] == 0.85
        assert result["vote_weight"] == 1.3
        assert result["proposals_made"] == 10
        assert result["proposals_accepted"] == 8
        assert result["proposal_acceptance_rate"] == 0.8
        assert result["critiques_given"] == 5
        assert result["critiques_valuable"] == 4

    def test_includes_computed_properties(self):
        """to_dict should include computed properties."""
        snapshot = IntrospectionSnapshot(
            agent_name="claude",
            proposals_made=10,
            proposals_accepted=7,
            critiques_given=5,
            critiques_valuable=4,
        )

        result = snapshot.to_dict()

        assert result["proposal_acceptance_rate"] == 0.7
        # critique_effectiveness should also be included
