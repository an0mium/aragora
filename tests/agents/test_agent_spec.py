"""
Tests for AgentSpec parsing and validation.

Tests the unified agent specification format and deprecated string parsing.
"""

import pytest
import warnings

from aragora.agents.spec import AgentSpec, parse_agents, VALID_ROLES


class TestAgentSpecCreation:
    """Test AgentSpec creation with explicit fields."""

    def test_create_with_provider_only(self):
        """Create spec with just provider."""
        spec = AgentSpec(provider="anthropic-api")
        assert spec.provider == "anthropic-api"
        assert spec.model is None
        assert spec.persona is None
        assert spec.role is None

    def test_create_with_all_fields(self):
        """Create spec with all fields."""
        spec = AgentSpec(
            provider="anthropic-api",
            model="claude-opus-4-5-20251101",
            persona="philosopher",
            role="proposer",
        )
        assert spec.provider == "anthropic-api"
        assert spec.model == "claude-opus-4-5-20251101"
        assert spec.persona == "philosopher"
        assert spec.role == "proposer"

    def test_valid_roles(self):
        """Verify valid roles are defined."""
        expected_roles = {"proposer", "critic", "synthesizer", "judge"}
        assert VALID_ROLES == expected_roles


class TestAgentSpecParsing:
    """Test deprecated string parsing (maintained for backward compatibility)."""

    def test_parse_pipe_delimited(self):
        """Parse pipe-delimited spec string."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            spec = AgentSpec.parse("anthropic-api|claude-opus|philosopher|proposer")

            # Should emit deprecation warning
            assert len(w) >= 1
            assert "deprecated" in str(w[0].message).lower()

        assert spec.provider == "anthropic-api"
        assert spec.persona == "philosopher"
        assert spec.role == "proposer"

    def test_parse_provider_only(self):
        """Parse simple provider string."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            spec = AgentSpec.parse("openai-api", _warn=False)

        assert spec.provider == "openai-api"
        assert spec.model is None
        assert spec.persona is None
        assert spec.role is None

    def test_parse_with_persona(self):
        """Parse provider with persona."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            spec = AgentSpec.parse("anthropic-api||skeptic", _warn=False)

        assert spec.provider == "anthropic-api"
        assert spec.persona == "skeptic"

    def test_parse_empty_fields(self):
        """Parse with empty middle fields."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            spec = AgentSpec.parse("gemini|||judge", _warn=False)

        assert spec.provider == "gemini"
        assert spec.role == "judge"


class TestAgentSpecParseList:
    """Test parsing comma-separated list of specs."""

    def test_parse_list_simple(self):
        """Parse simple comma-separated providers."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            specs = AgentSpec.parse_list("anthropic-api,openai-api,gemini", _warn=False)

        assert len(specs) == 3
        assert specs[0].provider == "anthropic-api"
        assert specs[1].provider == "openai-api"
        assert specs[2].provider == "gemini"

    def test_parse_list_with_specs(self):
        """Parse list with detailed specs."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            specs = AgentSpec.parse_list(
                "anthropic-api||philosopher|proposer,openai-api||skeptic|critic",
                _warn=False,
            )

        assert len(specs) == 2
        assert specs[0].persona == "philosopher"
        assert specs[0].role == "proposer"
        assert specs[1].persona == "skeptic"
        assert specs[1].role == "critic"


class TestAgentSpecToString:
    """Test string serialization."""

    def test_to_string_full(self):
        """Serialize spec with all fields to string."""
        spec = AgentSpec(
            provider="anthropic-api",
            model="claude-opus",
            persona="philosopher",
            role="proposer",
        )
        result = spec.to_string()
        assert "anthropic-api" in result
        assert "philosopher" in result
        assert "proposer" in result

    def test_to_string_minimal(self):
        """Serialize minimal spec to string."""
        spec = AgentSpec(provider="openai-api")
        result = spec.to_string()
        assert "openai-api" in result


class TestAgentSpecCreateTeam:
    """Test team creation from dicts."""

    def test_create_team_from_dicts(self):
        """Create team from list of dicts."""
        team = AgentSpec.create_team([
            {"provider": "anthropic-api", "persona": "philosopher", "role": "proposer"},
            {"provider": "openai-api", "persona": "skeptic", "role": "critic"},
            {"provider": "gemini", "role": "synthesizer"},
        ])

        assert len(team) == 3
        assert team[0].provider == "anthropic-api"
        assert team[0].persona == "philosopher"
        assert team[1].provider == "openai-api"
        assert team[1].role == "critic"
        assert team[2].provider == "gemini"

    def test_create_team_empty(self):
        """Create empty team."""
        team = AgentSpec.create_team([])
        assert len(team) == 0


class TestParseAgentsFunction:
    """Test the parse_agents convenience function."""

    def test_parse_agents_string(self):
        """Parse agents from string."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            agents = parse_agents("anthropic-api,openai-api")

        assert len(agents) == 2
        assert agents[0].provider == "anthropic-api"
        assert agents[1].provider == "openai-api"


class TestAgentSpecEquality:
    """Test spec comparison."""

    def test_specs_equal(self):
        """Equal specs should be equal."""
        spec1 = AgentSpec(provider="anthropic-api", persona="philosopher")
        spec2 = AgentSpec(provider="anthropic-api", persona="philosopher")
        assert spec1 == spec2

    def test_specs_not_equal(self):
        """Different specs should not be equal."""
        spec1 = AgentSpec(provider="anthropic-api", persona="philosopher")
        spec2 = AgentSpec(provider="anthropic-api", persona="skeptic")
        assert spec1 != spec2
