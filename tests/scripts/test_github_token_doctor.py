from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_module(script_name: str) -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


doctor = _load_module("github_token_doctor.py")


def test_probe_gh_user_reports_rate_limit_without_printing_token(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"cmd": cmd, "env": dict(kwargs.get("env") or {})})
        if cmd == ["gh", "auth", "token", "--user", "scarmani"]:
            return subprocess.CompletedProcess(cmd, 0, "secret-token\n", "")
        if cmd == ["gh", "api", "rate_limit"]:
            assert kwargs["env"]["GH_TOKEN"] == "secret-token"
            return subprocess.CompletedProcess(
                cmd,
                0,
                json.dumps(
                    {
                        "resources": {
                            "core": {"remaining": 4999, "limit": 5000},
                            "graphql": {"remaining": 4988, "limit": 5000, "reset": 123},
                        }
                    }
                ),
                "",
            )
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(doctor.subprocess, "run", fake_run)

    result = doctor.probe_gh_user("scarmani")

    assert result.to_dict() == {
        "source": "gh-user:scarmani",
        "available": True,
        "core_remaining": 4999,
        "core_limit": 5000,
        "graphql_remaining": 4988,
        "graphql_limit": 5000,
        "graphql_reset": 123,
        "error": "",
    }
    rendered = doctor._format_capacity(result)
    assert "secret-token" not in rendered
    assert calls[0]["env"] == {}


def test_probe_gh_user_reports_unavailable_token(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        doctor.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1, "", "not logged in"),
    )

    result = doctor.probe_gh_user("missing")

    assert result.source == "gh-user:missing"
    assert result.available is False
    assert result.error == "gh token unavailable"


def test_rate_limit_timeout_reports_source_unavailable(monkeypatch: Any) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(doctor.subprocess, "run", fake_run)

    result = doctor._rate_limit_for_token("secret-token", timeout=0.01)

    assert result.available is False
    assert result.error == "gh api rate_limit timed out"


def test_probe_app_token_uses_app_minter(monkeypatch: Any) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert cmd == ["gh", "api", "rate_limit"]
        assert kwargs["env"]["GH_TOKEN"] == "app-token"
        return subprocess.CompletedProcess(
            cmd,
            0,
            json.dumps(
                {
                    "resources": {
                        "core": {"remaining": 14999, "limit": 15000},
                        "graphql": {"remaining": 5000, "limit": 5000, "reset": 456},
                    }
                }
            ),
            "",
        )

    monkeypatch.setattr(doctor.subprocess, "run", fake_run)
    monkeypatch.setattr(
        "aragora.swarm.github_app_auth.get_github_app_installation_token",
        lambda: "app-token",
    )

    result = doctor.probe_app_token()

    assert result.source == "github-app"
    assert result.available is True
    assert result.core_limit == 15000
    assert result.graphql_remaining == 5000
