"""Issue #8101 sub-bug 3: RLM backend must not hard-require an OpenAI key.

``AragoraRLM`` defaults to the ``openai`` backend. When no OpenAI credential
is configured the bridge must route to a configured provider (openrouter,
anthropic) or skip TRUE RLM init with a logged note — never let the debate's
context_initializer phase die on ``openai.AuthenticationError``.

All credential lookups are mocked; no live secrets or API calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import pytest


@dataclass
class _Presence:
    name: str
    source: str
    critical: bool = True
    managed: bool = True


def _presence_factory(configured: dict[str, str]):
    """Build a get_secret_presence stand-in: configured maps name -> source."""

    def _presence(name: str, strict=None):  # noqa: ANN001 - test double
        return _Presence(name=name, source=configured.get(name, "missing"))

    return _presence


@pytest.fixture(autouse=True)
def _clear_rlm_env(monkeypatch: pytest.MonkeyPatch):
    for var in (
        "ARAGORA_RLM_BACKEND",
        "ARAGORA_RLM_PROVIDER",
        "ARAGORA_RLM_MODEL",
        "ARAGORA_RLM_MODEL_NAME",
        "ARAGORA_RLM_FALLBACK_BACKEND",
        "ARAGORA_RLM_FALLBACK_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_rlm(configured: dict[str, str]):
    from aragora.rlm.bridge import AragoraRLM

    with (
        patch(
            "aragora.rlm.bridge.get_secret_presence",
            side_effect=_presence_factory(configured),
        ),
        patch.object(AragoraRLM, "_init_official_rlm") as init_mock,
    ):
        rlm = AragoraRLM(enable_caching=False)
    return rlm, init_mock


class TestBackendSelectionWithoutOpenAIKey:
    def test_routes_to_openrouter_when_openai_missing(self):
        rlm, _ = _make_rlm({"OPENROUTER_API_KEY": "env"})
        assert rlm.backend_config.backend == "openrouter"

    def test_routes_to_anthropic_when_only_anthropic_configured(self):
        rlm, _ = _make_rlm({"ANTHROPIC_API_KEY": "aws"})
        assert rlm.backend_config.backend == "anthropic"
        # Default openai model name must not leak into the anthropic backend.
        assert "gpt" not in rlm.backend_config.model_name

    def test_keeps_openai_when_openai_configured(self):
        rlm, _ = _make_rlm({"OPENAI_API_KEY": "env"})
        assert rlm.backend_config.backend == "openai"

    def test_explicit_env_backend_is_respected(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ARAGORA_RLM_BACKEND", "openai")
        rlm, _ = _make_rlm({"OPENROUTER_API_KEY": "env"})
        assert rlm.backend_config.backend == "openai"

    def test_blocked_by_strict_mode_counts_as_unconfigured(self):
        rlm, _ = _make_rlm(
            {"OPENAI_API_KEY": "blocked_by_strict_mode", "OPENROUTER_API_KEY": "aws"}
        )
        assert rlm.backend_config.backend == "openrouter"


class TestCredentialGate:
    def test_no_credential_skips_true_rlm_init(self):
        """With no provider configured at all, TRUE RLM init must be skipped
        (logged note) instead of constructing an OpenAI-keyed RLM that will
        401 inside the context_initializer phase."""
        from aragora.rlm import bridge as bridge_mod
        from aragora.rlm.bridge import AragoraRLM

        with (
            patch(
                "aragora.rlm.bridge.get_secret_presence",
                side_effect=_presence_factory({}),
            ),
            patch.object(bridge_mod, "HAS_OFFICIAL_RLM", True),
            patch.object(AragoraRLM, "_init_official_rlm") as init_mock,
        ):
            rlm = AragoraRLM(enable_caching=False)

        init_mock.assert_not_called()
        assert rlm._official_rlm is None

    def test_configured_credential_allows_true_rlm_init(self):
        from aragora.rlm import bridge as bridge_mod
        from aragora.rlm.bridge import AragoraRLM

        with (
            patch(
                "aragora.rlm.bridge.get_secret_presence",
                side_effect=_presence_factory({"OPENAI_API_KEY": "aws"}),
            ),
            patch.object(bridge_mod, "HAS_OFFICIAL_RLM", True),
            patch.object(AragoraRLM, "_init_official_rlm") as init_mock,
        ):
            AragoraRLM(enable_caching=False)

        init_mock.assert_called_once()

    def test_unknown_backend_is_not_blocked(self):
        """Backends without a known key mapping (e.g. litellm) must not be
        gated — we cannot prove they lack credentials."""
        from aragora.rlm.bridge import AragoraRLM, RLMBackendConfig

        with patch(
            "aragora.rlm.bridge.get_secret_presence",
            side_effect=_presence_factory({}),
        ):
            with patch.object(AragoraRLM, "_init_official_rlm"):
                rlm = AragoraRLM(
                    backend_config=RLMBackendConfig(backend="litellm"),
                    enable_caching=False,
                )
        assert rlm._backend_has_usable_credential() is True
