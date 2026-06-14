"""
Tests for the introspection module core components.

Tests for:
- IntrospectionSnapshot dataclass (types.py)
- IntrospectionCache class (cache.py)
- get_agent_introspection() function (api.py)
- format_introspection_section() function (api.py)
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from aragora.introspection.types import IntrospectionSnapshot
from aragora.introspection.cache import IntrospectionCache
from aragora.introspection.api import get_agent_introspection, format_introspection_section


# ============================================================================
# IntrospectionSnapshot Tests
# ============================================================================


class TestIntrospectionSnapshot:
    """Tests for IntrospectionSnapshot dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        snapshot = IntrospectionSnapshot(agent_name="test-agent")

        assert snapshot.agent_name == "test-agent"
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

    def test_custom_values(self):
        """Test creating snapshot with custom values."""
        snapshot = IntrospectionSnapshot(
            agent_name="claude",
            reputation_score=0.85,
            vote_weight=1.4,
            proposals_made=10,
            proposals_accepted=8,
            critiques_given=20,
            critiques_valuable=15,
            calibration_score=0.75,
            debate_count=50,
            top_expertise=["ethics", "reasoning"],
            traits=["analytical", "thorough"],
        )

        assert snapshot.agent_name == "claude"
        assert snapshot.reputation_score == 0.85
        assert snapshot.vote_weight == 1.4
        assert snapshot.proposals_made == 10
        assert snapshot.proposals_accepted == 8


class TestProposalAcceptanceRate:
    """Tests for proposal_acceptance_rate property."""

    def test_zero_proposals(self):
        """Test rate when no proposals made."""
        snapshot = IntrospectionSnapshot(agent_name="test")
        assert snapshot.proposal_acceptance_rate == 0.0

    def test_all_accepted(self):
        """Test rate when all proposals accepted."""
        snapshot = IntrospectionSnapshot(
            agent_name="test",
            proposals_made=10,
            proposals_accepted=10,
        )
        assert snapshot.proposal_acceptance_rate == 1.0

    def test_partial_acceptance(self):
        """Test rate with partial acceptance."""
        snapshot = IntrospectionSnapshot(
            agent_name="test",
            proposals_made=10,
            proposals_accepted=6,
        )
        assert snapshot.proposal_acceptance_rate == 0.6

    def test_none_accepted(self):
        """Test rate when no proposals accepted."""
        snapshot = IntrospectionSnapshot(
            agent_name="test",
            proposals_made=5,
            proposals_accepted=0,
        )
        assert snapshot.proposal_acceptance_rate == 0.0


class TestCritiqueEffectiveness:
    """Tests for critique_effectiveness property."""

    def test_zero_critiques(self):
        """Test effectiveness when no critiques given."""
        snapshot = IntrospectionSnapshot(agent_name="test")
        assert snapshot.critique_effectiveness == 0.0

    def test_all_valuable(self):
        """Test effectiveness when all critiques valuable."""
        snapshot = IntrospectionSnapshot(
            agent_name="test",
            critiques_given=15,
            critiques_valuable=15,
        )
        assert snapshot.critique_effectiveness == 1.0

    def test_partial_valuable(self):
        """Test effectiveness with partial valuable critiques."""
        snapshot = IntrospectionSnapshot(
            agent_name="test",
            critiques_given=20,
            critiques_valuable=15,
        )
        assert snapshot.critique_effectiveness == 0.75


class TestCalibrationLabel:
    """Tests for calibration_label property."""

    def test_excellent_calibration(self):
        """Test label for excellent calibration (>= 0.7)."""
        snapshot = IntrospectionSnapshot(agent_name="test", calibration_score=0.75)
        assert snapshot.calibration_label == "excellent"

    def test_good_calibration(self):
        """Test label for good calibration (>= 0.5)."""
        snapshot = IntrospectionSnapshot(agent_name="test", calibration_score=0.55)
        assert snapshot.calibration_label == "good"

    def test_fair_calibration(self):
        """Test label for fair calibration (>= 0.3)."""
        snapshot = IntrospectionSnapshot(agent_name="test", calibration_score=0.35)
        assert snapshot.calibration_label == "fair"

    def test_developing_calibration(self):
        """Test label for developing calibration (< 0.3)."""
        snapshot = IntrospectionSnapshot(agent_name="test", calibration_score=0.2)
        assert snapshot.calibration_label == "developing"

    def test_boundary_excellent(self):
        """Test boundary at 0.7."""
        snapshot = IntrospectionSnapshot(agent_name="test", calibration_score=0.7)
        assert snapshot.calibration_label == "excellent"

    def test_boundary_good(self):
        """Test boundary at 0.5."""
        snapshot = IntrospectionSnapshot(agent_name="test", calibration_score=0.5)
        assert snapshot.calibration_label == "good"

    def test_boundary_fair(self):
        """Test boundary at 0.3."""
        snapshot = IntrospectionSnapshot(agent_name="test", calibration_score=0.3)
        assert snapshot.calibration_label == "fair"


class TestToPromptSection:
    """Tests for to_prompt_section() method."""

    def test_basic_format(self):
        """Test basic prompt section format."""
        snapshot = IntrospectionSnapshot(
            agent_name="claude",
            reputation_score=0.75,
            vote_weight=1.2,
        )

        result = snapshot.to_prompt_section()

        assert "## YOUR TRACK RECORD" in result
        assert "Reputation: 75%" in result
        assert "Vote weight: 1.2x" in result

    def test_with_proposals(self):
        """Test format includes proposals when present."""
        snapshot = IntrospectionSnapshot(
            agent_name="claude",
            reputation_score=0.75,
            vote_weight=1.2,
            proposals_made=10,
            proposals_accepted=8,
        )

        result = snapshot.to_prompt_section()

        assert "Proposals: 8/10" in result
        assert "80%" in result

    def test_with_critiques(self):
        """Test format includes critiques when present."""
        snapshot = IntrospectionSnapshot(
            agent_name="claude",
            reputation_score=0.75,
            vote_weight=1.2,
            critiques_given=20,
            critiques_valuable=15,
            calibration_score=0.75,
        )

        result = snapshot.to_prompt_section()

        assert "Critiques: 75% valuable" in result
        assert "Calibration: excellent" in result

    def test_with_expertise(self):
        """Test format includes expertise when present."""
        snapshot = IntrospectionSnapshot(
            agent_name="claude",
            reputation_score=0.75,
            vote_weight=1.2,
            top_expertise=["ethics", "reasoning", "code"],
        )

        result = snapshot.to_prompt_section()

        assert "Expertise:" in result
        assert "ethics" in result

    def test_with_traits(self):
        """Test format includes traits when present."""
        snapshot = IntrospectionSnapshot(
            agent_name="claude",
            reputation_score=0.75,
            vote_weight=1.2,
            traits=["analytical", "thorough", "precise"],
        )

        result = snapshot.to_prompt_section()

        assert "Style:" in result
        assert "analytical" in result

    def test_max_chars_limit(self):
        """Test that output respects max_chars limit."""
        snapshot = IntrospectionSnapshot(
            agent_name="claude",
            reputation_score=0.75,
            vote_weight=1.2,
            proposals_made=10,
            proposals_accepted=8,
            critiques_given=20,
            critiques_valuable=15,
            calibration_score=0.75,
            top_expertise=["ethics", "reasoning", "code"],
            traits=["analytical", "thorough", "precise"],
        )

        result = snapshot.to_prompt_section(max_chars=100)

        assert len(result) <= 100

    def test_truncation_preserves_header(self):
        """Test that truncation preserves header."""
        snapshot = IntrospectionSnapshot(
            agent_name="claude",
            reputation_score=0.75,
            vote_weight=1.2,
            proposals_made=10,
            proposals_accepted=8,
            top_expertise=["ethics", "reasoning", "code"],
            traits=["analytical", "thorough", "precise"],
        )

        result = snapshot.to_prompt_section(max_chars=100)

        assert "## YOUR TRACK RECORD" in result

    def test_expertise_limited_to_three(self):
        """Test that only top 3 expertise domains shown."""
        snapshot = IntrospectionSnapshot(
            agent_name="claude",
            reputation_score=0.5,
            vote_weight=1.0,
            top_expertise=["ethics", "reasoning", "code", "security", "testing"],
        )

        result = snapshot.to_prompt_section()

        # Only first 3 should appear
        assert "ethics" in result
        assert "reasoning" in result
        assert "code" in result
        # 4th and 5th should not appear in the expertise line
        assert result.count("security") == 0 or "Expertise:" not in result

    def test_traits_limited_to_three(self):
        """Test that only top 3 traits shown."""
        snapshot = IntrospectionSnapshot(
            agent_name="claude",
            reputation_score=0.5,
            vote_weight=1.0,
            traits=["analytical", "thorough", "precise", "verbose", "careful"],
        )

        result = snapshot.to_prompt_section()

        # Only first 3 should appear in Style line
        assert "analytical" in result


class TestToDict:
    """Tests for to_dict() method."""

    def test_all_fields_present(self):
        """Test that all fields are in dictionary."""
        snapshot = IntrospectionSnapshot(
            agent_name="claude",
            reputation_score=0.75,
            vote_weight=1.2,
            proposals_made=10,
            proposals_accepted=8,
            critiques_given=20,
            critiques_valuable=15,
            calibration_score=0.75,
            debate_count=50,
            top_expertise=["ethics"],
            traits=["analytical"],
        )

        result = snapshot.to_dict()

        assert result["agent_name"] == "claude"
        assert result["reputation_score"] == 0.75
        assert result["vote_weight"] == 1.2
        assert result["proposals_made"] == 10
        assert result["proposals_accepted"] == 8
        assert result["proposal_acceptance_rate"] == 0.8
        assert result["critiques_given"] == 20
        assert result["critiques_valuable"] == 15
        assert result["critique_effectiveness"] == 0.75
        assert result["calibration_score"] == 0.75
        assert result["calibration_label"] == "excellent"
        assert result["debate_count"] == 50
        assert result["top_expertise"] == ["ethics"]
        assert result["traits"] == ["analytical"]

    def test_computed_properties_included(self):
        """Test that computed properties are in dictionary."""
        snapshot = IntrospectionSnapshot(
            agent_name="test",
            proposals_made=10,
            proposals_accepted=5,
            critiques_given=10,
            critiques_valuable=8,
            calibration_score=0.6,
        )

        result = snapshot.to_dict()

        assert "proposal_acceptance_rate" in result
        assert result["proposal_acceptance_rate"] == 0.5
        assert "critique_effectiveness" in result
        assert result["critique_effectiveness"] == 0.8
        assert "calibration_label" in result
        assert result["calibration_label"] == "good"


# ============================================================================
# IntrospectionCache Tests
# ============================================================================


class TestIntrospectionCacheInit:
    """Tests for IntrospectionCache initialization."""

    def test_empty_on_creation(self):
        """Test cache is empty on creation."""
        cache = IntrospectionCache()

        assert cache.is_warm is False
        assert cache.agent_count == 0

    def test_get_returns_none_when_empty(self):
        """Test get returns None when cache is empty."""
        cache = IntrospectionCache()

        assert cache.get("claude") is None

    def test_get_all_returns_empty_dict(self):
        """Test get_all returns empty dict when cache is empty."""
        cache = IntrospectionCache()

        assert cache.get_all() == {}


class TestIntrospectionCacheWarm:
    """Tests for IntrospectionCache.warm() method."""

    def test_warm_populates_cache(self):
        """Test that warm() populates the cache."""
        cache = IntrospectionCache()

        agent1 = Mock()
        agent1.name = "claude"
        agent2 = Mock()
        agent2.name = "gpt-4"

        cache.warm(agents=[agent1, agent2])

        assert cache.is_warm is True
        assert cache.agent_count == 2

    def test_warm_creates_snapshots(self):
        """Test that warm() creates snapshots for each agent."""
        cache = IntrospectionCache()

        agent = Mock()
        agent.name = "claude"

        cache.warm(agents=[agent])

        snapshot = cache.get("claude")
        assert snapshot is not None
        assert isinstance(snapshot, IntrospectionSnapshot)
        assert snapshot.agent_name == "claude"

    def test_warm_with_memory(self):
        """Test warm() with CritiqueStore."""
        cache = IntrospectionCache()

        agent = Mock()
        agent.name = "claude"

        # Mock reputation
        mock_memory = Mock()
        mock_reputation = Mock()
        mock_reputation.score = 0.85
        mock_reputation.vote_weight = 1.3
        mock_reputation.proposals_made = 10
        mock_reputation.proposals_accepted = 8
        mock_reputation.critiques_given = 20
        mock_reputation.critiques_valuable = 15
        mock_reputation.calibration_score = 0.7
        mock_memory.get_reputation.return_value = mock_reputation

        cache.warm(agents=[agent], memory=mock_memory)

        snapshot = cache.get("claude")
        assert snapshot.reputation_score == 0.85
        assert snapshot.vote_weight == 1.3

    def test_warm_with_persona_manager(self):
        """Test warm() with PersonaManager."""
        cache = IntrospectionCache()

        agent = Mock()
        agent.name = "claude"

        # Mock persona - use spec to properly simulate hasattr behavior
        mock_persona_manager = Mock()
        mock_persona = Mock(spec=["expertise", "traits"])
        mock_persona.expertise = {"ethics": 0.9, "code": 0.8, "security": 0.7}
        mock_persona.traits = ["analytical", "thorough"]
        mock_persona_manager.get_persona.return_value = mock_persona

        cache.warm(agents=[agent], persona_manager=mock_persona_manager)

        snapshot = cache.get("claude")
        assert len(snapshot.top_expertise) > 0
        assert snapshot.traits == ["analytical", "thorough"]

    def test_warm_clears_previous_cache(self):
        """Test that warm() clears previous cache contents."""
        cache = IntrospectionCache()

        agent1 = Mock()
        agent1.name = "old-agent"
        cache.warm(agents=[agent1])

        agent2 = Mock()
        agent2.name = "new-agent"
        cache.warm(agents=[agent2])

        assert cache.get("old-agent") is None
        assert cache.get("new-agent") is not None
        assert cache.agent_count == 1

    def test_warm_handles_agent_without_name(self):
        """Test warm() handles agents without name attribute."""
        cache = IntrospectionCache()

        agent = "string-agent"  # No name attribute

        cache.warm(agents=[agent])

        snapshot = cache.get("string-agent")
        assert snapshot is not None
        assert snapshot.agent_name == "string-agent"


class TestIntrospectionCacheGet:
    """Tests for IntrospectionCache.get() method."""

    def test_get_existing_agent(self):
        """Test get returns snapshot for existing agent."""
        cache = IntrospectionCache()

        agent = Mock()
        agent.name = "claude"
        cache.warm(agents=[agent])

        snapshot = cache.get("claude")

        assert snapshot is not None
        assert snapshot.agent_name == "claude"

    def test_get_nonexistent_agent(self):
        """Test get returns None for non-existent agent."""
        cache = IntrospectionCache()

        agent = Mock()
        agent.name = "claude"
        cache.warm(agents=[agent])

        assert cache.get("gpt-4") is None


class TestIntrospectionCacheInvalidate:
    """Tests for IntrospectionCache.invalidate() method."""

    def test_invalidate_clears_cache(self):
        """Test invalidate clears the cache."""
        cache = IntrospectionCache()

        agent = Mock()
        agent.name = "claude"
        cache.warm(agents=[agent])

        cache.invalidate()

        assert cache.is_warm is False
        assert cache.agent_count == 0
        assert cache.get("claude") is None


class TestIntrospectionCacheGetAll:
    """Tests for IntrospectionCache.get_all() method."""

    def test_get_all_returns_copy(self):
        """Test get_all returns a copy of the cache."""
        cache = IntrospectionCache()

        agent = Mock()
        agent.name = "claude"
        cache.warm(agents=[agent])

        all_snapshots = cache.get_all()

        # Modify the returned dict
        all_snapshots["new-agent"] = IntrospectionSnapshot(agent_name="new")

        # Original cache should be unchanged
        assert cache.get("new-agent") is None
        assert cache.agent_count == 1


# ============================================================================
# get_agent_introspection Tests
# ============================================================================


class TestGetAgentIntrospection:
    """Tests for get_agent_introspection() function."""

    def test_basic_snapshot(self):
        """Test basic snapshot creation."""
        snapshot = get_agent_introspection("claude")

        assert snapshot.agent_name == "claude"
        assert isinstance(snapshot, IntrospectionSnapshot)

    def test_with_memory(self):
        """Test with CritiqueStore providing reputation."""
        mock_memory = Mock()
        mock_reputation = Mock()
        mock_reputation.score = 0.8
        mock_reputation.vote_weight = 1.2
        mock_reputation.proposals_made = 5
        mock_reputation.proposals_accepted = 4
        mock_reputation.critiques_given = 10
        mock_reputation.critiques_valuable = 8
        mock_reputation.calibration_score = 0.65
        mock_memory.get_reputation.return_value = mock_reputation

        snapshot = get_agent_introspection("claude", memory=mock_memory)

        assert snapshot.reputation_score == 0.8
        assert snapshot.vote_weight == 1.2
        assert snapshot.proposals_made == 5
        assert snapshot.proposals_accepted == 4

    def test_with_persona_manager(self):
        """Test with PersonaManager providing traits."""
        mock_persona_manager = Mock()
        # Use spec to control which attributes exist
        mock_persona = Mock(spec=["expertise", "traits"])
        mock_persona.expertise = {"ethics": 0.9, "code": 0.8}
        mock_persona.traits = ["analytical", "precise"]
        mock_persona_manager.get_persona.return_value = mock_persona

        snapshot = get_agent_introspection("claude", persona_manager=mock_persona_manager)

        assert len(snapshot.top_expertise) > 0
        assert "analytical" in snapshot.traits

    def test_memory_error_graceful_degradation(self):
        """Test graceful degradation when memory raises error."""
        mock_memory = Mock()
        mock_memory.get_reputation.side_effect = Exception("DB error")

        # Should not raise, should return default values
        snapshot = get_agent_introspection("claude", memory=mock_memory)

        assert snapshot.agent_name == "claude"
        assert snapshot.reputation_score == 0.0

    def test_persona_error_graceful_degradation(self):
        """Test graceful degradation when persona_manager raises error."""
        mock_persona_manager = Mock()
        mock_persona_manager.get_persona.side_effect = Exception("Persona error")

        # Should not raise, should return default values
        snapshot = get_agent_introspection("claude", persona_manager=mock_persona_manager)

        assert snapshot.agent_name == "claude"
        assert snapshot.top_expertise == []
        assert snapshot.traits == []

    def test_memory_returns_none(self):
        """Test handling when memory returns None reputation."""
        mock_memory = Mock()
        mock_memory.get_reputation.return_value = None

        snapshot = get_agent_introspection("claude", memory=mock_memory)

        assert snapshot.reputation_score == 0.0

    def test_persona_returns_none(self):
        """Test handling when persona_manager returns None."""
        mock_persona_manager = Mock()
        mock_persona_manager.get_persona.return_value = None

        snapshot = get_agent_introspection("claude", persona_manager=mock_persona_manager)

        assert snapshot.top_expertise == []
        assert snapshot.traits == []

    def test_persona_with_top_expertise_property(self):
        """Test persona with top_expertise property."""
        mock_persona_manager = Mock()
        mock_persona = Mock()
        mock_persona.top_expertise = [("ethics", 0.9), ("code", 0.8), ("security", 0.7)]
        del mock_persona.expertise  # Remove expertise attr
        mock_persona.traits = []
        mock_persona_manager.get_persona.return_value = mock_persona

        snapshot = get_agent_introspection("claude", persona_manager=mock_persona_manager)

        assert snapshot.top_expertise == ["ethics", "code", "security"]


# ============================================================================
# format_introspection_section Tests
# ============================================================================


class TestFormatIntrospectionSection:
    """Tests for format_introspection_section() function."""

    def test_basic_format(self):
        """Test basic formatting."""
        snapshot = IntrospectionSnapshot(
            agent_name="claude",
            reputation_score=0.75,
            vote_weight=1.2,
        )

        result = format_introspection_section(snapshot)

        assert "## YOUR TRACK RECORD" in result
        assert "Reputation: 75%" in result

    def test_custom_max_chars(self):
        """Test with custom max_chars."""
        snapshot = IntrospectionSnapshot(
            agent_name="claude",
            reputation_score=0.75,
            vote_weight=1.2,
            proposals_made=10,
            proposals_accepted=8,
            top_expertise=["ethics", "code"],
            traits=["analytical"],
        )

        result = format_introspection_section(snapshot, max_chars=100)

        assert len(result) <= 100

    def test_delegates_to_snapshot(self):
        """Test that it delegates to snapshot.to_prompt_section()."""
        snapshot = IntrospectionSnapshot(
            agent_name="claude",
            reputation_score=0.5,
            vote_weight=1.0,
        )

        direct_result = snapshot.to_prompt_section(max_chars=500)
        function_result = format_introspection_section(snapshot, max_chars=500)

        assert direct_result == function_result
