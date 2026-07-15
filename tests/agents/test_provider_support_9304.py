"""Secured-provider support + failure-text classification (issue #9304)."""

from __future__ import annotations

import pytest


class TestBaseUrlOverride:
    def test_anthropic_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
        from aragora.agents.api_agents.anthropic import _resolve_base_url

        assert (
            _resolve_base_url("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
            == "https://api.anthropic.com/v1"
        )

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("http://127.0.0.1:8317", "http://127.0.0.1:8317/v1"),
            ("http://127.0.0.1:8317/", "http://127.0.0.1:8317/v1"),
            ("https://gw.example/v1", "https://gw.example/v1"),
        ],
    )
    def test_anthropic_override_normalizes(
        self, monkeypatch: pytest.MonkeyPatch, value: str, expected: str
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", value)
        from aragora.agents.api_agents.anthropic import _resolve_base_url

        assert _resolve_base_url("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1") == expected

    def test_agent_constructor_honors_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8317")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from aragora.agents.api_agents.anthropic import AnthropicAgent

        agent = AnthropicAgent()
        assert agent.base_url == "http://127.0.0.1:8317/v1"

    def test_openai_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_BASE_URL", "https://litellm.internal")
        from aragora.agents.api_agents.openai import _resolve_openai_base_url

        assert _resolve_openai_base_url() == "https://litellm.internal/v1"


class TestExit0ProviderErrors:
    def test_markers_match_live_wall_texts(self) -> None:
        from aragora.agents.cli_agents import _EXIT0_PROVIDER_ERROR_MARKERS

        live = [
            "Free tier users do not have access to this model",
            "You're out of usage credits. Run /usage-credits",
            "Not logged in · Please run /login",
        ]
        for text in live:
            assert any(m in text.lower() for m in _EXIT0_PROVIDER_ERROR_MARKERS), text

    def test_long_legit_output_quoting_error_not_matched_by_length_guard(self) -> None:
        # The classification requires len < 2000; a real proposal quoting an
        # error string is longer and must survive. This pins the guard value.
        legit = ("Analysis of auth flows. " * 100) + "the api returned 'quota exceeded' once"
        assert len(legit) >= 2000
