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


def test_status_handles_ssrf_blocked_localhost(monkeypatch, capsys) -> None:
    """`aragora status` must not crash when the health probe raises SSRFBlockedError.

    The default server URL is http://localhost:8080, and safe_get() rejects
    localhost with SSRFBlockedError (a subclass of SSRFValidationError -> Exception,
    NOT of OSError/ConnectionError/RuntimeError). The command should render a
    friendly 'not reachable' line and still print the remaining sections.
    """
    from aragora.security.safe_http import SSRFBlockedError

    def fake_safe_get(*_args: Any, **_kwargs: Any) -> Any:
        raise SSRFBlockedError("Localhost hostname detected", url="http://localhost:8080")

    monkeypatch.setattr("aragora.security.safe_http.safe_get", fake_safe_get)

    args = argparse.Namespace(server="http://localhost:8080")

    # Must not raise (previously aborted with an uncaught traceback).
    status_mod.cmd_status(args)

    out = capsys.readouterr().out
    assert "Server not reachable at http://localhost:8080" in out
    # Sections after the server health probe must still render.
    assert "Databases:" in out


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


def _patch_provider_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "aragora.cli.api_keys.validate_provider_key",
        lambda _provider: SimpleNamespace(remote_status="valid", is_valid=True, message="ok"),
    )


def test_validate_env_unconfigured_backends_are_not_reported_connected(monkeypatch, capsys) -> None:
    """Regression: an unconfigured (skipped) Redis/PostgreSQL must NOT be reported
    as a live connection.

    The startup validators return ``(True, "<backend> not configured (skipping
    connectivity check)")`` for optional, unconfigured backends. Previously the
    CLI mapped that ``True`` to ``connected: True`` / ``status: "ok"``, falsely
    telling operators and machine consumers the datastore was reachable.
    """
    _clear_provider_env(monkeypatch)
    _patch_provider_valid(monkeypatch)

    async def skipped_redis(*_a: Any, **_k: Any) -> tuple[bool, str]:
        return True, "Redis not configured (skipping connectivity check)"

    async def skipped_db(*_a: Any, **_k: Any) -> tuple[bool, str]:
        return True, "PostgreSQL not configured (skipping connectivity check)"

    monkeypatch.setattr("aragora.server.startup.validate_redis_connectivity", skipped_redis)
    monkeypatch.setattr("aragora.server.startup.validate_database_connectivity", skipped_db)

    with pytest.raises(SystemExit):
        status_mod.cmd_validate_env(_validate_args(smoke=False))

    payload = json.loads(capsys.readouterr().out)
    redis_check = payload["checks"]["redis"]
    postgres_check = payload["checks"]["postgresql"]
    assert redis_check["connected"] is False
    assert redis_check["status"] == "skip"
    assert postgres_check["connected"] is False
    assert postgres_check["status"] == "skip"


def test_validate_env_live_backend_connection_is_reported_connected(monkeypatch, capsys) -> None:
    """A genuine live connection must still be reported as connected (no regression)."""
    _clear_provider_env(monkeypatch)
    _patch_provider_valid(monkeypatch)

    async def live_redis(*_a: Any, **_k: Any) -> tuple[bool, str]:
        return True, "Redis connected (version 7.0)"

    async def live_db(*_a: Any, **_k: Any) -> tuple[bool, str]:
        return True, "PostgreSQL connected (version 16)"

    monkeypatch.setattr("aragora.server.startup.validate_redis_connectivity", live_redis)
    monkeypatch.setattr("aragora.server.startup.validate_database_connectivity", live_db)

    with pytest.raises(SystemExit):
        status_mod.cmd_validate_env(_validate_args(smoke=False))

    payload = json.loads(capsys.readouterr().out)
    assert payload["checks"]["redis"] == {
        "status": "ok",
        "connected": True,
        "message": "Redis connected (version 7.0)",
    }
    assert payload["checks"]["postgresql"] == {
        "status": "ok",
        "connected": True,
        "message": "PostgreSQL connected (version 16)",
    }


def test_validate_env_required_unconfigured_redis_fails(monkeypatch, capsys) -> None:
    """When Redis is required (distributed/multi-instance) but unconfigured,
    validate-env must FAIL — not report a benign skip / 'ready'."""
    _clear_provider_env(monkeypatch)
    _patch_provider_valid(monkeypatch)
    monkeypatch.setattr("aragora.control_plane.leader.is_distributed_state_required", lambda: True)

    async def skipped_redis(*_a: Any, **_k: Any) -> tuple[bool, str]:
        return True, "Redis not configured (skipping connectivity check)"

    async def live_db(*_a: Any, **_k: Any) -> tuple[bool, str]:
        return True, "PostgreSQL connected (version 16)"

    monkeypatch.setattr("aragora.server.startup.validate_redis_connectivity", skipped_redis)
    monkeypatch.setattr("aragora.server.startup.validate_database_connectivity", live_db)

    with pytest.raises(SystemExit) as exc:
        status_mod.cmd_validate_env(_validate_args(smoke=False))

    assert exc.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["checks"]["redis"]["status"] == "error"
    assert payload["checks"]["redis"]["connected"] is False
    assert payload["valid"] is False
    assert any("Redis" in err for err in payload["errors"])


class TestConnectivitySkipped:
    def test_not_configured_message_is_skipped(self) -> None:
        assert status_mod._connectivity_skipped(
            "Redis not configured (skipping connectivity check)"
        )

    def test_skipping_connectivity_message_is_skipped(self) -> None:
        assert status_mod._connectivity_skipped("skipping connectivity check")

    def test_live_connection_message_is_not_skipped(self) -> None:
        assert not status_mod._connectivity_skipped("Redis connected (version 7.0)")

    def test_empty_message_is_not_skipped(self) -> None:
        assert not status_mod._connectivity_skipped("")


class TestConnectivityCheckResult:
    def test_skipped_backend_is_not_connected(self) -> None:
        result = status_mod._connectivity_check_result(
            True,
            "Redis not configured (skipping connectivity check)",
            required=False,
            optional_status="warning",
        )
        assert result["connected"] is False
        assert result["status"] == "skip"

    def test_required_unconfigured_backend_is_error(self) -> None:
        # A required backend that is not even configured must fail (not be a
        # benign skip), so validate-env never reports ready while a required
        # datastore is absent.
        result = status_mod._connectivity_check_result(
            True,
            "PostgreSQL not configured (skipping connectivity check)",
            required=True,
            optional_status="info",
        )
        assert result["connected"] is False
        assert result["status"] == "error"

    def test_live_connection_is_connected(self) -> None:
        result = status_mod._connectivity_check_result(
            True, "Redis connected (version 7.0)", required=False, optional_status="warning"
        )
        assert result["connected"] is True
        assert result["status"] == "ok"

    def test_failure_when_required_is_error(self) -> None:
        result = status_mod._connectivity_check_result(
            False, "Redis connection failed", required=True, optional_status="warning"
        )
        assert result["connected"] is False
        assert result["status"] == "error"

    def test_failure_when_optional_uses_optional_status(self) -> None:
        result = status_mod._connectivity_check_result(
            False, "PostgreSQL unreachable", required=False, optional_status="info"
        )
        assert result["connected"] is False
        assert result["status"] == "info"
