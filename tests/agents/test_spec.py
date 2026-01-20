"""
Tests for the unified AgentSpec class.

Tests both new pipe-delimited format and legacy colon format parsing,
ensuring backward compatibility and correct role/persona handling.
"""

import pytest

from aragora.agents.spec import AgentSpec, VALID_ROLES, parse_agents


class TestAgentSpecPipeFormat:
    """Tests for new pipe-delimited format: provider|model|persona|role."""

    def test_full_spec(self):
        """Parse full pipe-delimited spec with all fields."""
        spec = AgentSpec.parse("anthropic-api|claude-opus|philosopher|critic")
        assert spec.provider == "anthropic-api"
        assert spec.model == "claude-opus"
        assert spec.persona == "philosopher"
        assert spec.role == "critic"

    def test_provider_only(self):
        """Parse spec with only provider (empty other fields)."""
        spec = AgentSpec.parse("gemini|||")
        assert spec.provider == "gemini"
        assert spec.model is None
        assert spec.persona is None
        assert spec.role is None  # None = assign automatically based on position

    def test_provider_and_role(self):
        """Parse spec with provider and role only."""
        spec = AgentSpec.parse("anthropic-api|||judge")
        assert spec.provider == "anthropic-api"
        assert spec.model is None
        assert spec.persona is None
        assert spec.role == "judge"

    def test_provider_model_role(self):
        """Parse spec with provider, model, and role."""
        spec = AgentSpec.parse("openai-api|gpt-4o||synthesizer")
        assert spec.provider == "openai-api"
        assert spec.model == "gpt-4o"
        assert spec.persona is None
        assert spec.role == "synthesizer"

    def test_provider_persona_role(self):
        """Parse spec with provider, persona, and role (no model)."""
        spec = AgentSpec.parse("qwen||qwen|critic")
        assert spec.provider == "qwen"
        assert spec.model is None
        assert spec.persona == "qwen"
        assert spec.role == "critic"

    def test_empty_role_is_none(self):
        """Empty role field returns None for automatic assignment."""
        spec = AgentSpec.parse("anthropic-api|claude-opus|philosopher|")
        assert spec.role is None  # None = assign automatically based on position


class TestAgentSpecLegacyFormat:
    """Tests for legacy colon format: provider:persona."""

    def test_provider_persona(self):
        """Parse legacy format with provider:persona."""
        spec = AgentSpec.parse("anthropic-api:claude")
        assert spec.provider == "anthropic-api"
        assert spec.persona == "claude"
        assert spec.role is None  # None = assign automatically based on position

    def test_provider_only(self):
        """Parse legacy format with just provider."""
        spec = AgentSpec.parse("gemini")
        assert spec.provider == "gemini"
        assert spec.persona is None
        assert spec.role is None  # None = assign automatically based on position

    def test_valid_role_parsed_as_role(self):
        """Legacy format treats second part as role if it's a valid role."""
        # Valid roles (proposer, critic, synthesizer, judge) are recognized as roles
        spec = AgentSpec.parse("anthropic-api:critic")
        assert spec.role == "critic"  # Recognized as explicit role
        assert spec.persona is None  # Not a persona

    def test_non_role_parsed_as_persona(self):
        """Legacy format treats second part as persona if not a valid role."""
        # Non-role strings like "philosopher" are treated as personas
        spec = AgentSpec.parse("anthropic-api:philosopher")
        assert spec.persona == "philosopher"  # Treated as persona
        assert spec.role is None  # None = assign automatically based on position

    def test_complex_persona_name(self):
        """Legacy format handles complex persona names."""
        spec = AgentSpec.parse("anthropic-api:security_engineer")
        assert spec.persona == "security_engineer"
        assert spec.role is None  # None = assign automatically based on position

    @pytest.mark.parametrize("role", ["proposer", "critic", "synthesizer", "judge"])
    def test_all_valid_roles_recognized(self, role):
        """All valid roles are recognized in legacy colon format."""
        spec = AgentSpec.parse(f"anthropic-api:{role}")
        assert spec.role == role
        assert spec.persona is None

    def test_role_case_insensitive(self):
        """Role parsing is case-insensitive."""
        spec = AgentSpec.parse("anthropic-api:CRITIC")
        assert spec.role == "critic"
        assert spec.persona is None


class TestAgentSpecValidation:
    """Tests for AgentSpec validation."""

    def test_invalid_provider_raises(self):
        """Invalid provider raises ValueError."""
        with pytest.raises(ValueError, match="Invalid agent provider"):
            AgentSpec.parse("invalid-provider|||")

    def test_invalid_role_raises(self):
        """Invalid role raises ValueError."""
        with pytest.raises(ValueError, match="Invalid agent role"):
            AgentSpec.parse("anthropic-api|||invalid_role")

    def test_valid_roles(self):
        """All valid roles are accepted."""
        for role in VALID_ROLES:
            spec = AgentSpec.parse(f"anthropic-api|||{role}")
            assert spec.role == role

    def test_empty_spec_raises(self):
        """Empty spec string raises ValueError."""
        with pytest.raises(ValueError, match="Empty agent spec"):
            AgentSpec.parse("")

    def test_whitespace_only_raises(self):
        """Whitespace-only spec string raises ValueError."""
        with pytest.raises(ValueError, match="Empty agent spec"):
            AgentSpec.parse("   ")

    def test_provider_case_insensitive(self):
        """Provider names are normalized to lowercase."""
        spec = AgentSpec.parse("ANTHROPIC-API|||proposer")
        assert spec.provider == "anthropic-api"


class TestAgentSpecParseList:
    """Tests for parsing comma-separated lists of specs."""

    def test_parse_multiple_specs(self):
        """Parse comma-separated list of specs."""
        specs = AgentSpec.parse_list("anthropic-api|||proposer,qwen||qwen|critic")
        assert len(specs) == 2
        assert specs[0].provider == "anthropic-api"
        assert specs[0].role == "proposer"
        assert specs[1].provider == "qwen"
        assert specs[1].persona == "qwen"
        assert specs[1].role == "critic"

    def test_parse_mixed_formats(self):
        """Parse list mixing new and legacy formats."""
        specs = AgentSpec.parse_list("anthropic-api:claude,qwen||qwen|critic,gemini")
        assert len(specs) == 3
        # Legacy format with persona
        assert specs[0].provider == "anthropic-api"
        assert specs[0].persona == "claude"
        assert specs[0].role is None  # Auto-assign based on position
        # New format with explicit role
        assert specs[1].provider == "qwen"
        assert specs[1].role == "critic"
        # Simple legacy - no role specified
        assert specs[2].provider == "gemini"
        assert specs[2].role is None  # Auto-assign based on position

    def test_parse_empty_string(self):
        """Empty string returns empty list."""
        specs = AgentSpec.parse_list("")
        assert specs == []

    def test_parse_with_whitespace(self):
        """Whitespace around specs is trimmed."""
        specs = AgentSpec.parse_list("  anthropic-api  ,  gemini  ")
        assert len(specs) == 2
        assert specs[0].provider == "anthropic-api"
        assert specs[1].provider == "gemini"

    def test_parse_agents_convenience_function(self):
        """parse_agents convenience function works."""
        specs = parse_agents("anthropic-api,gemini")
        assert len(specs) == 2


class TestAgentSpecSerialization:
    """Tests for serialization methods."""

    def test_to_string(self):
        """to_string produces pipe-delimited format."""
        spec = AgentSpec(
            provider="anthropic-api",
            model="claude-opus",
            persona="philosopher",
            role="critic",
        )
        assert spec.to_string() == "anthropic-api|claude-opus|philosopher|critic"

    def test_to_string_with_empty_fields(self):
        """to_string handles empty fields correctly."""
        spec = AgentSpec(provider="gemini", role="proposer")
        assert spec.to_string() == "gemini|||proposer"

    def test_to_legacy_string_with_persona(self):
        """to_legacy_string produces colon format with persona."""
        spec = AgentSpec(provider="anthropic-api", persona="claude", role="proposer")
        assert spec.to_legacy_string() == "anthropic-api:claude"

    def test_to_legacy_string_without_persona(self):
        """to_legacy_string produces provider-only format."""
        spec = AgentSpec(provider="gemini", role="proposer")
        assert spec.to_legacy_string() == "gemini"

    def test_roundtrip_pipe_format(self):
        """Parse and serialize roundtrip preserves data."""
        original = "anthropic-api|claude-opus|philosopher|critic"
        spec = AgentSpec.parse(original)
        assert spec.to_string() == original


class TestAgentSpecMutation:
    """Tests for creating new specs with modified fields."""

    def test_with_role(self):
        """with_role creates new spec with different role."""
        original = AgentSpec(provider="anthropic-api", persona="claude", role="proposer")
        modified = original.with_role("critic")

        # Original unchanged
        assert original.role == "proposer"
        # New spec has new role
        assert modified.role == "critic"
        assert modified.provider == "anthropic-api"
        assert modified.persona == "claude"

    def test_with_persona(self):
        """with_persona creates new spec with different persona."""
        original = AgentSpec(provider="anthropic-api", role="proposer")
        modified = original.with_persona("philosopher")

        # Original unchanged
        assert original.persona is None
        # New spec has new persona
        assert modified.persona == "philosopher"
        assert modified.provider == "anthropic-api"


class TestAgentSpecName:
    """Tests for auto-generated agent names."""

    def test_name_generation_full(self):
        """Name includes provider, persona, and role."""
        spec = AgentSpec(provider="anthropic-api", persona="claude", role="critic")
        assert spec.name == "anthropic-api_claude_critic"

    def test_name_generation_no_persona(self):
        """Name without persona."""
        spec = AgentSpec(provider="gemini", role="proposer")
        assert spec.name == "gemini_proposer"

    def test_custom_name(self):
        """Custom name overrides auto-generation."""
        spec = AgentSpec(provider="anthropic-api", role="proposer", name="my_agent")
        assert spec.name == "my_agent"


class TestAgentSpecRepr:
    """Tests for string representation."""

    def test_repr_full(self):
        """__repr__ includes all non-None fields."""
        spec = AgentSpec(
            provider="anthropic-api",
            model="claude-opus",
            persona="philosopher",
            role="critic",
        )
        r = repr(spec)
        assert "provider='anthropic-api'" in r
        assert "model='claude-opus'" in r
        assert "persona='philosopher'" in r
        assert "role='critic'" in r

    def test_repr_minimal(self):
        """__repr__ excludes None fields."""
        spec = AgentSpec(provider="gemini", role="proposer")
        r = repr(spec)
        assert "provider='gemini'" in r
        assert "role='proposer'" in r
        assert "model" not in r
        assert "persona" not in r


class TestAgentSpecCreateTeam:
    """Tests for create_team() factory method (preferred over parse_list)."""

    def test_create_team_from_dicts(self):
        """Create team from list of dicts with explicit fields."""
        team = AgentSpec.create_team(
            [
                {"provider": "anthropic-api", "persona": "philosopher"},
                {"provider": "openai-api", "persona": "skeptic"},
                {"provider": "gemini"},
            ]
        )

        assert len(team) == 3
        # Default role rotation
        assert team[0].role == "proposer"
        assert team[1].role == "critic"
        assert team[2].role == "synthesizer"

    def test_create_team_explicit_roles(self):
        """Explicit roles override default rotation."""
        team = AgentSpec.create_team(
            [
                {"provider": "anthropic-api", "role": "judge"},
                {"provider": "openai-api", "role": "judge"},
            ]
        )

        assert team[0].role == "judge"
        assert team[1].role == "judge"

    def test_create_team_mixed_explicit_default(self):
        """Mix of explicit roles and default rotation."""
        team = AgentSpec.create_team(
            [
                {"provider": "anthropic-api", "role": "judge"},  # Explicit
                {"provider": "openai-api"},  # Should get critic (position 1)
                {"provider": "gemini"},  # Should get synthesizer (position 2)
            ]
        )

        assert team[0].role == "judge"
        assert team[1].role == "critic"
        assert team[2].role == "synthesizer"

    def test_create_team_with_model(self):
        """Create team with model specified."""
        team = AgentSpec.create_team(
            [
                {"provider": "anthropic-api", "model": "claude-opus-4-5", "role": "proposer"},
            ]
        )

        assert team[0].model == "claude-opus-4-5"

    def test_create_team_with_custom_name(self):
        """Create team with custom agent names."""
        team = AgentSpec.create_team(
            [
                {"provider": "anthropic-api", "name": "lead_researcher"},
            ]
        )

        assert team[0].name == "lead_researcher"

    def test_create_team_from_agentspec_instances(self):
        """Create team from existing AgentSpec instances."""
        spec1 = AgentSpec(provider="anthropic-api", role="proposer")
        spec2 = AgentSpec(provider="gemini", role="critic")

        team = AgentSpec.create_team([spec1, spec2])

        assert team[0] is spec1  # Same instance
        assert team[1] is spec2

    def test_create_team_mixed_dict_and_spec(self):
        """Create team from mix of dicts and AgentSpec instances."""
        spec1 = AgentSpec(provider="anthropic-api", role="judge")
        team = AgentSpec.create_team(
            [
                spec1,
                {"provider": "openai-api"},
            ]
        )

        assert team[0] is spec1
        assert team[1].provider == "openai-api"

    def test_create_team_disable_role_rotation(self):
        """Disable default role rotation."""
        team = AgentSpec.create_team(
            [
                {"provider": "anthropic-api"},
                {"provider": "openai-api"},
            ],
            default_role_rotation=False,
        )

        # All should be proposer (default) when rotation disabled
        assert team[0].role == "proposer"
        assert team[1].role == "proposer"

    def test_create_team_missing_provider_raises(self):
        """Missing provider field raises ValueError."""
        with pytest.raises(ValueError, match="missing required 'provider' field"):
            AgentSpec.create_team(
                [
                    {"persona": "philosopher"},  # No provider
                ]
            )

    def test_create_team_empty_list(self):
        """Empty list returns empty team."""
        team = AgentSpec.create_team([])
        assert team == []

    def test_create_team_role_rotation_cycles(self):
        """Role rotation cycles through all 4 roles."""
        team = AgentSpec.create_team(
            [
                {"provider": "anthropic-api"},
                {"provider": "openai-api"},
                {"provider": "gemini"},
                {"provider": "grok"},
                {"provider": "deepseek"},  # Wraps to proposer
            ]
        )

        assert team[0].role == "proposer"
        assert team[1].role == "critic"
        assert team[2].role == "synthesizer"
        assert team[3].role == "judge"
        assert team[4].role == "proposer"  # Cycles back


class TestQuestionClassifierIntegration:
    """Tests for integration with question_classifier output format."""

    def test_parse_classifier_output(self):
        """Parse output from question_classifier.get_agent_string()."""
        # Example output from question_classifier
        agent_string = "anthropic-api||claude|proposer,qwen||qwen|critic,gemini|||synthesizer"
        specs = AgentSpec.parse_list(agent_string)

        assert len(specs) == 3

        # First agent
        assert specs[0].provider == "anthropic-api"
        assert specs[0].persona == "claude"
        assert specs[0].role == "proposer"

        # Second agent
        assert specs[1].provider == "qwen"
        assert specs[1].persona == "qwen"
        assert specs[1].role == "critic"

        # Third agent
        assert specs[2].provider == "gemini"
        assert specs[2].role == "synthesizer"
