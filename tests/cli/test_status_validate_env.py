from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from typing import Any

import pytest

from aragora.cli import parser as cli_parser
from aragora.cli.commands import status as status_mod
from aragora.config.provider_readiness import PROVIDER_CREDENTIAL_SPECS


def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for spec in PROVIDER_CREDENTIAL_SPECS:
        for env_var in spec.env_vars:
            monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(
        "aragora.config.provider_readiness._hydrate_from_secret_loaders",
        lambda _names: ((), ()),
    )


def _patch_backend_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_redis(*_args: Any, **_kwargs: Any) -> tuple[bool, str]:
        return False, "not configured in test"

    async def fake_database(*_args: Any, **_kwargs: Any) -> tuple[bool, str]:
        return False, "not configured in test"

    monkeypatch.setattr("aragora.server.startup.validate_redis_connectivity", fake_redis)
    monkeypatch.setattr("aragora.server.startup.validate_database_connectivity", fake_database)


def _validate_args(**overrides: Any) -> argparse.Namespace:
    values = {
        "verbose": False,
        "json": True,
        "strict": False,
        "smoke": True,
        "agents": "openai",
        "smoke_timeout": 1.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_validate_env_parser_accepts_smoke_agents() -> None:
    parser = cli_parser.build_parser()

    args = parser.parse_args(
        ["validate-env", "--smoke", "--agents", "openai", "--smoke-timeout", "3"]
    )

    assert args.smoke is True
    assert args.agents == "openai"
    assert args.smoke_timeout == 3.0


def test_validate_env_fails_when_configured_provider_rejects_key(monkeypatch, capsys) -> None:
    """validate-env should not report readiness for an expired configured provider."""
    _clear_provider_env(monkeypatch)
    _patch_backend_checks(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def fake_validate_provider_key(provider: str) -> SimpleNamespace:
        assert provider == "gemini"
        return SimpleNamespace(
            remote_status="invalid",
            is_valid=False,
            message="Provider rejected the API key",
        )

    monkeypatch.setattr(
        "aragora.cli.api_keys.validate_provider_key",
        fake_validate_provider_key,
    )

    with pytest.raises(SystemExit) as exc_info:
        status_mod.cmd_validate_env(_validate_args(smoke=False))

    assert exc_info.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["checks"]["ai_providers"]["status"] == "error"
    assert payload["checks"]["ai_providers"]["configured"] == ["gemini"]
    assert payload["checks"]["ai_providers"]["validation"] == [
        {
            "provider": "gemini",
            "remote_status": "invalid",
            "is_valid": False,
            "message": "Provider rejected the API key",
        }
    ]
    assert payload["errors"] == ["gemini: Provider rejected the API key"]


def test_validate_env_smoke_requires_agents(monkeypatch, capsys) -> None:
    _clear_provider_env(monkeypatch)
    _patch_backend_checks(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        status_mod.cmd_validate_env(_validate_args(agents=""))

    assert exc.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["checks"]["ai_provider_smoke"]["status"] == "error"
    assert "No AI provider smoke agents selected" in payload["errors"]


def test_validate_env_smoke_fails_when_selected_provider_missing(monkeypatch, capsys) -> None:
    _clear_provider_env(monkeypatch)
    _patch_backend_checks(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        status_mod.cmd_validate_env(_validate_args(agents="anthropic-api"))

    assert exc.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    smoke = payload["checks"]["ai_provider_smoke"]
    assert smoke["status"] == "error"
    assert smoke["agents"][0]["agent"] == "anthropic-api"
    assert "no configured credential" in smoke["agents"][0]["message"]


def test_validate_env_smoke_fails_on_invalid_provider_call(monkeypatch, capsys) -> None:
    _clear_provider_env(monkeypatch)
    _patch_backend_checks(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FailingAgent:
        async def generate(self, _prompt: str) -> str:
            raise RuntimeError("invalid api key")

    monkeypatch.setattr(
        "aragora.agents.base.create_agent", lambda *_args, **_kwargs: FailingAgent()
    )

    with pytest.raises(SystemExit) as exc:
        status_mod.cmd_validate_env(_validate_args(agents="openai"))

    assert exc.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    smoke = payload["checks"]["ai_provider_smoke"]
    assert smoke["status"] == "error"
    assert "invalid api key" in smoke["agents"][0]["message"]


def test_validate_env_smoke_passes_on_tiny_ok_response(monkeypatch, capsys) -> None:
    _clear_provider_env(monkeypatch)
    _patch_backend_checks(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_validate_provider_key(provider: str) -> SimpleNamespace:
        assert provider == "openai"
        return SimpleNamespace(remote_status="valid", is_valid=True, message="ok")

    class OkAgent:
        async def generate(self, _prompt: str) -> str:
            return "ok"

    monkeypatch.setattr(
        "aragora.cli.api_keys.validate_provider_key",
        fake_validate_provider_key,
    )
    monkeypatch.setattr("aragora.agents.base.create_agent", lambda *_args, **_kwargs: OkAgent())

    with pytest.raises(SystemExit) as exc:
        status_mod.cmd_validate_env(_validate_args(agents="openai"))

    assert exc.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    smoke = payload["checks"]["ai_provider_smoke"]
    assert smoke["status"] == "ok"
    assert smoke["agents"][0]["response_preview"] == "ok"
