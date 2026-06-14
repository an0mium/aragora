"""
Tests for Agent Introspection API.

Tests the introspection module that provides self-awareness data
injection for agents during debates.
"""

import pytest
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

from aragora.introspection import (
    IntrospectionSnapshot,
    IntrospectionCache,
    get_agent_introspection,
    format_introspection_section,
)


class TestIntrospectionSnapshot:
    """Test IntrospectionSnapshot data class."""

    def test_default_values(self):
        """Test default snapshot values."""
        snapshot = IntrospectionSnapshot(agent_name="test_agent")

        assert snapshot.agent_name == "test_agent"
        assert snapshot.reputation_score == 0.0
        assert snapshot.vote_weight == 1.0
        assert snapshot.proposals_made == 0
        assert snapshot.proposals_accepted == 0
        assert snapshot.calibration_score == 0.5

    def test_proposal_acceptance_rate(self):
        """Test proposal acceptance rate calculation."""
        snapshot = IntrospectionSnapshot(
            agent_name="test",
            proposals_made=10,
            proposals_accepted=7,
        )

        assert snapshot.proposal_acceptance_rate == 0.7

    def test_proposal_acceptance_rate_zero_proposals(self):
        """Test acceptance rate with no proposals."""
        snapshot = IntrospectionSnapshot(agent_name="test", proposals_made=0)

        assert snapshot.proposal_acceptance_rate == 0.0

    def test_critique_effectiveness(self):
        """Test critique effectiveness calculation."""
        snapshot = IntrospectionSnapshot(
            agent_name="test",
            critiques_given=20,
            critiques_valuable=15,
        )

        assert snapshot.critique_effectiveness == 0.75

    def test_critique_effectiveness_zero_critiques(self):
        """Test effectiveness with no critiques."""
        snapshot = IntrospectionSnapshot(agent_name="test", critiques_given=0)

        assert snapshot.critique_effectiveness == 0.0

    def test_calibration_labels(self):
        """Test calibration label generation."""
        excellent = IntrospectionSnapshot(agent_name="a", calibration_score=0.8)
        good = IntrospectionSnapshot(agent_name="b", calibration_score=0.6)
        fair = IntrospectionSnapshot(agent_name="c", calibration_score=0.4)
        developing = IntrospectionSnapshot(agent_name="d", calibration_score=0.2)

        assert excellent.calibration_label == "excellent"
        assert good.calibration_label == "good"
        assert fair.calibration_label == "fair"
        assert developing.calibration_label == "developing"

    def test_to_prompt_section_under_limit(self):
        """Verify prompt section stays under 600 chars by default."""
        snapshot = IntrospectionSnapshot(
            agent_name="claude",
            reputation_score=0.72,
            vote_weight=1.2,
            proposals_made=11,
            proposals_accepted=8,
            critiques_given=20,
            critiques_valuable=13,
            calibration_score=0.68,
            debate_count=15,
            top_expertise=["security", "performance", "architecture"],
            traits=["thorough", "pragmatic", "innovative"],
        )

        section = snapshot.to_prompt_section()

        assert len(section) <= 600
        assert "## YOUR TRACK RECORD" in section
        assert "72%" in section  # Reputation percentage
        assert "1.2x" in section  # Vote weight

    def test_to_prompt_section_custom_limit(self):
        """Test custom character limit."""
        snapshot = IntrospectionSnapshot(
            agent_name="test",
            reputation_score=0.72,
            vote_weight=1.2,
            proposals_made=10,
            proposals_accepted=7,
        )

        section = snapshot.to_prompt_section(max_chars=200)

        assert len(section) <= 200
        assert "## YOUR TRACK RECORD" in section

    def test_to_prompt_section_minimal_data(self):
        """Test prompt section with minimal data."""
        snapshot = IntrospectionSnapshot(agent_name="new_agent")

        section = snapshot.to_prompt_section()

        assert "## YOUR TRACK RECORD" in section
        assert "Reputation: 0%" in section

    def test_to_prompt_section_includes_expertise(self):
        """Test that expertise is included when available."""
        snapshot = IntrospectionSnapshot(
            agent_name="test",
            top_expertise=["security", "performance"],
        )

        section = snapshot.to_prompt_section()

        assert "Expertise:" in section
        assert "security" in section

    def test_to_prompt_section_includes_traits(self):
        """Test that traits are included when available."""
        snapshot = IntrospectionSnapshot(
            agent_name="test",
            traits=["thorough", "pragmatic"],
        )

        section = snapshot.to_prompt_section()

        assert "Style:" in section
        assert "thorough" in section

    def test_to_dict(self):
        """Test serialization to dictionary."""
        snapshot = IntrospectionSnapshot(
            agent_name="test",
            reputation_score=0.72,
            proposals_made=10,
            proposals_accepted=7,
        )

        d = snapshot.to_dict()

        assert d["agent_name"] == "test"
        assert d["reputation_score"] == 0.72
        assert d["proposal_acceptance_rate"] == 0.7
        assert "calibration_label" in d


class TestIntrospectionCache:
    """Test caching layer."""

    def test_initial_state(self):
        """Test cache initial state."""
        cache = IntrospectionCache()

        assert not cache.is_warm
        assert cache.agent_count == 0
        assert cache.get("nonexistent") is None

    def test_warm_loads_agents(self):
        """Test that warm() loads data for all agents."""
        # Create mock agents
        mock_agent1 = Mock()
        mock_agent1.name = "claude"
        mock_agent2 = Mock()
        mock_agent2.name = "gemini"

        cache = IntrospectionCache()
        cache.warm(agents=[mock_agent1, mock_agent2])

        assert cache.is_warm
        assert cache.agent_count == 2
        assert cache.get("claude") is not None
        assert cache.get("gemini") is not None

    def test_warm_with_memory(self):
        """Test warm() with CritiqueStore integration."""
        # Create mock agent
        mock_agent = Mock()
        mock_agent.name = "claude"

        # Create mock memory with reputation
        mock_memory = Mock()
        mock_rep = Mock()
        mock_rep.score = 0.72
        mock_rep.vote_weight = 1.2
        mock_rep.proposals_made = 10
        mock_rep.proposals_accepted = 7
        mock_rep.critiques_given = 20
        mock_rep.critiques_valuable = 15
        mock_rep.calibration_score = 0.68
        mock_memory.get_reputation.return_value = mock_rep

        cache = IntrospectionCache()
        cache.warm(agents=[mock_agent], memory=mock_memory)

        snapshot = cache.get("claude")
        assert snapshot is not None
        assert snapshot.reputation_score == 0.72
        assert snapshot.vote_weight == 1.2
        assert snapshot.proposals_made == 10

    def test_warm_with_persona_manager(self):
        """Test warm() with PersonaManager integration."""
        mock_agent = Mock()
        mock_agent.name = "claude"

        # Create mock persona manager
        mock_persona_manager = Mock()
        mock_persona = Mock()
        mock_persona.top_expertise = [("security", 0.9), ("performance", 0.8)]
        mock_persona.traits = ["thorough", "pragmatic"]
        mock_persona_manager.get_persona.return_value = mock_persona

        cache = IntrospectionCache()
        cache.warm(agents=[mock_agent], persona_manager=mock_persona_manager)

        snapshot = cache.get("claude")
        assert snapshot is not None
        assert "security" in snapshot.top_expertise
        assert "thorough" in snapshot.traits

    def test_get_returns_cached_object(self):
        """Test that get() returns the same cached object."""
        mock_agent = Mock()
        mock_agent.name = "test"

        cache = IntrospectionCache()
        cache.warm(agents=[mock_agent])

        result1 = cache.get("test")
        result2 = cache.get("test")

        assert result1 is result2

    def test_invalidate(self):
        """Test cache invalidation."""
        mock_agent = Mock()
        mock_agent.name = "test"

        cache = IntrospectionCache()
        cache.warm(agents=[mock_agent])

        assert cache.is_warm
        assert cache.agent_count == 1

        cache.invalidate()

        assert not cache.is_warm
        assert cache.agent_count == 0
        assert cache.get("test") is None

    def test_get_all(self):
        """Test getting all cached snapshots."""
        mock_agent1 = Mock()
        mock_agent1.name = "claude"
        mock_agent2 = Mock()
        mock_agent2.name = "gemini"

        cache = IntrospectionCache()
        cache.warm(agents=[mock_agent1, mock_agent2])

        all_snapshots = cache.get_all()

        assert len(all_snapshots) == 2
        assert "claude" in all_snapshots
        assert "gemini" in all_snapshots


class TestGetAgentIntrospection:
    """Test core introspection API function."""

    def test_basic_call(self):
        """Test basic function call."""
        snapshot = get_agent_introspection("test_agent")

        assert snapshot.agent_name == "test_agent"
        assert snapshot.reputation_score == 0.0

    def test_with_memory(self):
        """Test with CritiqueStore data."""
        mock_memory = Mock()
        mock_rep = Mock()
        mock_rep.score = 0.72
        mock_rep.vote_weight = 1.2
        mock_rep.proposals_made = 10
        mock_rep.proposals_accepted = 7
        mock_rep.critiques_given = 20
        mock_rep.critiques_valuable = 15
        mock_rep.calibration_score = 0.68
        mock_memory.get_reputation.return_value = mock_rep

        snapshot = get_agent_introspection("claude", memory=mock_memory)

        assert snapshot.reputation_score == 0.72
        assert snapshot.proposals_made == 10
        assert snapshot.calibration_score == 0.68

    def test_with_persona_manager(self):
        """Test with PersonaManager data."""
        mock_persona_manager = Mock()
        mock_persona = Mock()
        mock_persona.top_expertise = [("security", 0.9), ("performance", 0.8)]
        mock_persona.traits = ["thorough", "pragmatic"]
        mock_persona_manager.get_persona.return_value = mock_persona

        snapshot = get_agent_introspection("claude", persona_manager=mock_persona_manager)

        assert "security" in snapshot.top_expertise
        assert "thorough" in snapshot.traits

    def test_graceful_degradation_memory_error(self):
        """Test graceful handling of memory errors."""
        mock_memory = Mock()
        mock_memory.get_reputation.side_effect = Exception("DB error")

        # Should not raise, returns default values
        snapshot = get_agent_introspection("claude", memory=mock_memory)

        assert snapshot.agent_name == "claude"
        assert snapshot.reputation_score == 0.0

    def test_graceful_degradation_persona_error(self):
        """Test graceful handling of persona errors."""
        mock_persona_manager = Mock()
        mock_persona_manager.get_persona.side_effect = Exception("DB error")

        # Should not raise, returns default values
        snapshot = get_agent_introspection("claude", persona_manager=mock_persona_manager)

        assert snapshot.agent_name == "claude"
        assert len(snapshot.top_expertise) == 0


class TestFormatIntrospectionSection:
    """Test convenience formatting function."""

    def test_basic_format(self):
        """Test basic formatting."""
        snapshot = IntrospectionSnapshot(
            agent_name="test",
            reputation_score=0.72,
        )

        section = format_introspection_section(snapshot)

        assert "## YOUR TRACK RECORD" in section
        assert "72%" in section

    def test_custom_max_chars(self):
        """Test custom character limit."""
        snapshot = IntrospectionSnapshot(
            agent_name="test",
            reputation_score=0.72,
            top_expertise=["security", "performance", "architecture"],
            traits=["thorough", "pragmatic", "innovative"],
        )

        section = format_introspection_section(snapshot, max_chars=200)

        assert len(section) <= 200


class TestPromptFormatConsistency:
    """Test that prompt formatting matches existing patterns."""

    def test_header_format_matches_patterns(self):
        """Test that header uses ## format like SUCCESSFUL PATTERNS."""
        snapshot = IntrospectionSnapshot(agent_name="test")

        section = snapshot.to_prompt_section()

        # Should start with ## header like other sections
        assert section.startswith("## ")
        assert "YOUR TRACK RECORD" in section

    def test_no_empty_lines_at_boundaries(self):
        """Test that output doesn't have leading/trailing empty lines."""
        snapshot = IntrospectionSnapshot(
            agent_name="test",
            reputation_score=0.72,
        )

        section = snapshot.to_prompt_section()

        assert not section.startswith("\n")
        assert not section.endswith("\n\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
