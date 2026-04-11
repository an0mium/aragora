"""Tests for aragora.agents.personas.helpers module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.agents.personas.core import Persona, PersonaManager
from aragora.agents.personas.helpers import (
    apply_persona_to_agent,
    get_or_create_persona,
    get_persona_prompt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_persona() -> Persona:
    return Persona(
        agent_name="tester",
        description="A test persona",
        traits=["thorough", "pragmatic"],
        expertise={"security": 0.9, "testing": 0.8, "performance": 0.7},
        temperature=0.6,
        top_p=0.95,
        frequency_penalty=0.1,
    )


@pytest.fixture()
def empty_persona() -> Persona:
    return Persona(agent_name="empty")


@pytest.fixture()
def mock_manager(sample_persona: Persona) -> MagicMock:
    mgr = MagicMock(spec=PersonaManager)
    mgr.get_persona.return_value = sample_persona
    mgr.create_persona.return_value = sample_persona
    return mgr


# ---------------------------------------------------------------------------
# get_or_create_persona
# ---------------------------------------------------------------------------


class TestGetOrCreatePersona:
    """Tests for get_or_create_persona."""

    def test_returns_existing_persona(
        self, mock_manager: MagicMock, sample_persona: Persona
    ) -> None:
        result = get_or_create_persona(mock_manager, "tester")

        assert result is sample_persona
        mock_manager.get_persona.assert_called_once_with("tester")
        mock_manager.create_persona.assert_not_called()

    def test_creates_from_default_when_not_found(self, mock_manager: MagicMock) -> None:
        mock_manager.get_persona.return_value = None

        result = get_or_create_persona(mock_manager, "claude_critic")

        mock_manager.create_persona.assert_called_once()
        call_kwargs = mock_manager.create_persona.call_args
        assert call_kwargs[1]["agent_name"] == "claude_critic"
        assert "description" in call_kwargs[1]
        assert "traits" in call_kwargs[1]
        assert "expertise" in call_kwargs[1]

    def test_creates_empty_persona_for_unknown_agent(self, mock_manager: MagicMock) -> None:
        mock_manager.get_persona.return_value = None

        get_or_create_persona(mock_manager, "totally_unknown_agent")

        mock_manager.create_persona.assert_called_once_with(agent_name="totally_unknown_agent")

    def test_base_name_extraction_uses_first_segment(self, mock_manager: MagicMock) -> None:
        """Agent name 'grok_specialist' should match default persona 'grok'."""
        mock_manager.get_persona.return_value = None

        get_or_create_persona(mock_manager, "grok_specialist")

        call_kwargs = mock_manager.create_persona.call_args[1]
        assert call_kwargs["agent_name"] == "grok_specialist"
        # Should have pulled traits from the 'grok' default persona
        assert "contrarian" in call_kwargs["traits"]


# ---------------------------------------------------------------------------
# apply_persona_to_agent
# ---------------------------------------------------------------------------


class TestApplyPersonaToAgent:
    """Tests for apply_persona_to_agent."""

    def test_applies_default_persona_sets_system_prompt(self, sample_persona: Persona) -> None:
        agent = MagicMock()
        agent.system_prompt = "Existing prompt."

        with patch(
            "aragora.agents.personas.helpers.DEFAULT_PERSONAS",
            {"tester": sample_persona},
        ):
            result = apply_persona_to_agent(agent, "tester")

        assert result is True
        assert "Existing prompt." in agent.system_prompt
        # Persona context is prepended
        assert agent.system_prompt.startswith("Your role:")

    def test_applies_generation_params_via_method(self, sample_persona: Persona) -> None:
        agent = MagicMock()
        agent.system_prompt = ""

        with patch(
            "aragora.agents.personas.helpers.DEFAULT_PERSONAS",
            {"tester": sample_persona},
        ):
            apply_persona_to_agent(agent, "tester")

        agent.set_generation_params.assert_called_once_with(
            temperature=0.6,
            top_p=0.95,
            frequency_penalty=0.1,
        )

    def test_applies_generation_params_via_attributes(self, sample_persona: Persona) -> None:
        agent = MagicMock(spec=[])  # no methods
        agent.system_prompt = ""
        agent.temperature = 0.7
        agent.top_p = 1.0
        agent.frequency_penalty = 0.0

        with patch(
            "aragora.agents.personas.helpers.DEFAULT_PERSONAS",
            {"tester": sample_persona},
        ):
            apply_persona_to_agent(agent, "tester")

        assert agent.temperature == 0.6
        assert agent.top_p == 0.95
        assert agent.frequency_penalty == 0.1

    def test_returns_false_for_unknown_persona(self) -> None:
        agent = MagicMock()

        with patch("aragora.agents.personas.helpers.DEFAULT_PERSONAS", {}):
            result = apply_persona_to_agent(agent, "nonexistent")

        assert result is False

    def test_falls_back_to_manager_persona(
        self, mock_manager: MagicMock, sample_persona: Persona
    ) -> None:
        agent = MagicMock()
        agent.system_prompt = ""

        with patch("aragora.agents.personas.helpers.DEFAULT_PERSONAS", {}):
            result = apply_persona_to_agent(agent, "tester", manager=mock_manager)

        assert result is True
        mock_manager.get_persona.assert_called_once_with("tester")

    def test_builds_fallback_prompt_when_to_prompt_context_empty(self) -> None:
        persona = Persona(
            agent_name="fallback",
            description="A fallback agent",
            traits=["direct", "pragmatic"],
            expertise={"security": 0.9, "testing": 0.8},
        )
        # Ensure to_prompt_context returns content (it will because description + traits exist)
        # Instead test the branch where to_prompt_context returns ""
        persona_no_fields = Persona(agent_name="bare")

        agent = MagicMock()
        agent.system_prompt = ""

        with patch(
            "aragora.agents.personas.helpers.DEFAULT_PERSONAS",
            {"bare": persona_no_fields},
        ):
            result = apply_persona_to_agent(agent, "bare")

        # Persona has no description/traits/expertise so prompt is empty
        assert result is True

    def test_no_system_prompt_attribute_skips_prompt_assignment(
        self, sample_persona: Persona
    ) -> None:
        agent = MagicMock(spec=[])  # agent without system_prompt

        with patch(
            "aragora.agents.personas.helpers.DEFAULT_PERSONAS",
            {"tester": sample_persona},
        ):
            result = apply_persona_to_agent(agent, "tester")

        assert result is True
        assert not hasattr(agent, "system_prompt")


# ---------------------------------------------------------------------------
# get_persona_prompt
# ---------------------------------------------------------------------------


class TestGetPersonaPrompt:
    """Tests for get_persona_prompt."""

    def test_returns_prompt_for_default_persona(self) -> None:
        prompt = get_persona_prompt("claude")

        assert "Your role:" in prompt
        assert "thorough" in prompt

    def test_returns_empty_string_for_unknown(self) -> None:
        prompt = get_persona_prompt("nonexistent_agent_xyz")

        assert prompt == ""

    def test_uses_manager_when_not_in_defaults(
        self, mock_manager: MagicMock, sample_persona: Persona
    ) -> None:
        with patch("aragora.agents.personas.helpers.DEFAULT_PERSONAS", {}):
            prompt = get_persona_prompt("tester", manager=mock_manager)

        assert "Your role:" in prompt
        mock_manager.get_persona.assert_called_once_with("tester")

    def test_builds_fallback_prompt_from_traits_and_expertise(self) -> None:
        persona = Persona(
            agent_name="custom",
            traits=["direct", "innovative"],
            expertise={"security": 0.9},
        )
        mgr = MagicMock(spec=PersonaManager)
        mgr.get_persona.return_value = persona

        with patch("aragora.agents.personas.helpers.DEFAULT_PERSONAS", {}):
            prompt = get_persona_prompt("custom", manager=mgr)

        # to_prompt_context produces content because traits + expertise are set
        assert "direct" in prompt
        assert "security" in prompt

    def test_fallback_prompt_without_expertise(self) -> None:
        """Persona with traits but no expertise still generates a prompt."""
        persona = Persona(
            agent_name="minimal",
            description="Minimal agent",
            traits=["thorough"],
        )
        mgr = MagicMock(spec=PersonaManager)
        mgr.get_persona.return_value = persona

        with patch("aragora.agents.personas.helpers.DEFAULT_PERSONAS", {}):
            prompt = get_persona_prompt("minimal", manager=mgr)

        assert "Minimal agent" in prompt
        assert "thorough" in prompt


# ---------------------------------------------------------------------------
# Persona dataclass edge-case coverage (exercised via helpers)
# ---------------------------------------------------------------------------


class TestPersonaEdgeCases:
    """Edge cases for Persona used through the helpers."""

    def test_top_expertise_returns_top_3(self) -> None:
        persona = Persona(
            agent_name="broad",
            expertise={
                "security": 0.9,
                "testing": 0.8,
                "performance": 0.7,
                "api_design": 0.6,
            },
        )
        top = persona.top_expertise
        assert len(top) == 3
        assert top[0] == ("security", 0.9)

    def test_trait_string_defaults_to_balanced(self) -> None:
        persona = Persona(agent_name="plain")
        assert persona.trait_string == "balanced"

    def test_generation_params_dict(self, sample_persona: Persona) -> None:
        params = sample_persona.generation_params
        assert params == {
            "temperature": 0.6,
            "top_p": 0.95,
            "frequency_penalty": 0.1,
        }

    def test_to_prompt_context_empty_persona(self) -> None:
        persona = Persona(agent_name="blank")
        assert persona.to_prompt_context() == ""

    def test_serialization_roundtrip(self, sample_persona: Persona) -> None:
        """Persona can be serialized to dict and reconstructed."""
        import dataclasses

        data = dataclasses.asdict(sample_persona)
        reconstructed = Persona(**data)

        assert reconstructed.agent_name == sample_persona.agent_name
        assert reconstructed.traits == sample_persona.traits
        assert reconstructed.expertise == sample_persona.expertise
        assert reconstructed.temperature == sample_persona.temperature
        assert reconstructed.top_p == sample_persona.top_p
        assert reconstructed.frequency_penalty == sample_persona.frequency_penalty
