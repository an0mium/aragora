"""Tests for introspection injection into agent prompts (Sprint 15C)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.prompt_builder import PromptBuilder


def _make_prompt_builder(enable_introspection: bool = True, **kwargs) -> PromptBuilder:
    """Create a minimal PromptBuilder for testing."""
    protocol = MagicMock()
    protocol.rounds = 3
    protocol.agreement_intensity = None
    protocol.asymmetric_stances = False
    protocol.enforce_language = False
    protocol.response_language = "English"
    protocol.enable_privacy_anonymization = False

    env = MagicMock()
    env.task = "Should we adopt microservices?"
    env.context = ""

    return PromptBuilder(
        protocol=protocol,
        env=env,
        enable_introspection=enable_introspection,
        **kwargs,
    )


def _make_agent(name: str = "claude") -> MagicMock:
    agent = MagicMock()
    agent.name = name
    agent.role = "analyst"
    return agent


class TestIntrospectionInjection:
    """Test that introspection context is injected into prompts."""

    def test_introspection_injected_when_enabled(self):
        """Introspection section appears in prompt when enabled."""
        builder = _make_prompt_builder(enable_introspection=True)
        agent = _make_agent("claude")

        with patch(
            "aragora.debate.prompt_builder.PromptBuilder._get_introspection_context",
            return_value="## YOUR TRACK RECORD\nReputation: 85% | Vote weight: 1.3x",
        ):
            prompt = builder.build_proposal_prompt(agent)

        assert "YOUR TRACK RECORD" in prompt
        assert "Reputation: 85%" in prompt

    def test_introspection_not_injected_when_disabled(self):
        """Introspection section absent when disabled."""
        builder = _make_prompt_builder(enable_introspection=False)
        agent = _make_agent("claude")

        prompt = builder.build_proposal_prompt(agent)

        assert "YOUR TRACK RECORD" not in prompt

    def test_graceful_with_no_reputation_data(self):
        """Returns empty string when no reputation data exists."""
        builder = _make_prompt_builder(enable_introspection=True)

        # No memory or persona_manager set, so introspection returns defaults
        result = builder._get_introspection_context("unknown_agent")

        # Should return a section even with defaults (reputation: 0%)
        assert isinstance(result, str)
        assert "YOUR TRACK RECORD" in result

    def test_within_char_limit(self):
        """Introspection context stays within 600 char limit."""
        builder = _make_prompt_builder(enable_introspection=True)

        result = builder._get_introspection_context("claude")

        assert len(result) <= 600

    def test_introspection_calls_correct_api(self):
        """_get_introspection_context calls the introspection API correctly."""
        memory = MagicMock()
        persona_manager = MagicMock()
        builder = _make_prompt_builder(
            enable_introspection=True,
            memory=memory,
            persona_manager=persona_manager,
        )

        with patch("aragora.introspection.api.get_agent_introspection") as mock_get:
            from aragora.introspection.types import IntrospectionSnapshot

            mock_get.return_value = IntrospectionSnapshot(
                agent_name="claude",
                reputation_score=0.9,
                vote_weight=1.4,
            )
            result = builder._get_introspection_context("claude")

            mock_get.assert_called_once_with(
                "claude",
                memory=memory,
                persona_manager=persona_manager,
            )
            assert "YOUR TRACK RECORD" in result

    def test_introspection_returns_empty_on_import_error(self):
        """Returns empty string when introspection module not available."""
        builder = _make_prompt_builder(enable_introspection=True)

        with patch(
            "builtins.__import__",
            side_effect=ImportError("no module"),
        ):
            result = builder._get_introspection_context("claude")

        assert result == ""

    def test_arena_config_has_enable_introspection(self):
        """ArenaConfig has enable_introspection defaulting to True."""
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig()
        assert config.enable_introspection is True

        config2 = ArenaConfig(enable_introspection=False)
        assert config2.enable_introspection is False
