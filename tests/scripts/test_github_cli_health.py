from __future__ import annotations

import subprocess
from pathlib import Path

import scripts.github_cli_health as mod


def test_run_uses_github_cli_env_for_gh(monkeypatch) -> None:
    monkeypatch.setattr(mod, "github_cli_env", lambda env: {"GH_TOKEN": "app-token"})

    captured: dict[str, object] = {}

    def fake_subprocess_run(*args, **kwargs):
        captured["env"] = kwargs.get("env")
        return subprocess.CompletedProcess(
            args=kwargs.get("args", args[0]), returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_subprocess_run)

    mod._run(["gh", "auth", "status"], cwd=Path("."), timeout_seconds=5)

    assert captured["env"] == {"GH_TOKEN": "app-token"}


def test_check_github_cli_health_ready(monkeypatch) -> None:
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/bin/gh")

    def fake_run(
        args: list[str], *, cwd: Path, timeout_seconds: int
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(mod, "_run", fake_run)

    health = mod.check_github_cli_health(Path("."))

    assert health.ready is True
    assert health.mode == "ready"
    assert health.auth_ok is True
    assert health.api_ok is True


def test_check_github_cli_health_ready_with_app_env_auth_even_if_auth_status_is_stale(
    monkeypatch,
) -> None:
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/bin/gh")

    calls: list[list[str]] = []

    def fake_run(
        args: list[str], *, cwd: Path, timeout_seconds: int
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[:3] == ["gh", "api", "rate_limit"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")
        return subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout="",
            stderr="the token in default is invalid",
        )

    monkeypatch.setattr(mod, "_run", fake_run)

    health = mod.check_github_cli_health(Path("."))

    assert health.ready is True
    assert health.mode == "ready"
    assert health.auth_ok is True
    assert health.api_ok is True
    assert calls == [["gh", "api", "rate_limit"]]


def test_check_github_cli_health_detects_connectivity_failure(monkeypatch) -> None:
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/bin/gh")

    def fake_run(
        args: list[str], *, cwd: Path, timeout_seconds: int
    ) -> subprocess.CompletedProcess[str]:
        if args[:3] == ["gh", "api", "rate_limit"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=1,
                stdout="",
                stderr="error connecting to api.github.com",
            )
        if args[:3] == ["gh", "auth", "status"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")
        raise AssertionError(f"unexpected args: {args}")

    monkeypatch.setattr(mod, "_run", fake_run)

    health = mod.check_github_cli_health(Path("."))

    assert health.ready is False
    assert health.mode == "connectivity_failed"
    assert health.auth_ok is True
    assert health.api_ok is False
    assert mod.is_github_connectivity_error(health.error) is True


def test_github_connectivity_error_detects_dns_lookup_failures() -> None:
    assert mod.is_github_connectivity_error(
        'Get "https://api.github.com/rate_limit": dial tcp: lookup api.github.com: no such host'
    )
    assert mod.is_github_connectivity_error(
        'Post "https://api.github.com/graphql": dial tcp: lookup github.com: no such host'
    )


def test_github_connectivity_error_detects_bounded_probe_timeouts() -> None:
    assert mod.is_github_connectivity_error("command timed out after 20s: gh api rate_limit")
    assert mod.is_github_connectivity_error(
        "Get https://api.github.com/rate_limit: net/http: request canceled"
    )


def test_check_github_cli_health_classifies_api_timeout_as_connectivity(
    monkeypatch,
) -> None:
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/bin/gh")

    def fake_run(
        args: list[str], *, cwd: Path, timeout_seconds: int
    ) -> subprocess.CompletedProcess[str]:
        if args[:3] == ["gh", "api", "rate_limit"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=124,
                stdout="",
                stderr="command timed out after 20s: gh api rate_limit",
            )
        if args[:3] == ["gh", "auth", "status"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")
        raise AssertionError(f"unexpected args: {args}")

    monkeypatch.setattr(mod, "_run", fake_run)

    health = mod.check_github_cli_health(Path("."))

    assert health.ready is False
    assert health.mode == "connectivity_failed"
    assert health.auth_ok is True
    assert health.api_ok is False


def test_check_github_cli_health_prefers_connectivity_error_when_auth_status_is_stale(
    monkeypatch,
) -> None:
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/bin/gh")

    def fake_run(
        args: list[str], *, cwd: Path, timeout_seconds: int
    ) -> subprocess.CompletedProcess[str]:
        if args[:3] == ["gh", "api", "rate_limit"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=1,
                stdout="",
                stderr="error connecting to api.github.com",
            )
        if args[:3] == ["gh", "auth", "status"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=1,
                stdout="",
                stderr="the token in default is invalid",
            )
        raise AssertionError(f"unexpected args: {args}")

    monkeypatch.setattr(mod, "_run", fake_run)

    health = mod.check_github_cli_health(Path("."))

    assert health.ready is False
    assert health.mode == "connectivity_failed"
    assert health.auth_ok is False
    assert health.api_ok is False
    assert health.error == "error connecting to api.github.com"


def test_check_github_cli_health_detects_auth_failure(monkeypatch) -> None:
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/bin/gh")

    def fake_run(
        args: list[str], *, cwd: Path, timeout_seconds: int
    ) -> subprocess.CompletedProcess[str]:
        if args[:3] == ["gh", "api", "rate_limit"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=1,
                stdout="",
                stderr="HTTP 401: Requires authentication",
            )
        if args[:3] == ["gh", "auth", "status"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=1,
                stdout="",
                stderr="You are not logged into any GitHub hosts. Run gh auth login.",
            )
        return subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout="",
            stderr="unexpected args",
        )

    monkeypatch.setattr(mod, "_run", fake_run)

    health = mod.check_github_cli_health(Path("."))

    assert health.ready is False
    assert health.mode == "auth_failed"
    assert health.auth_ok is False
    assert health.api_ok is False


def test_main_json_reports_unavailable_state(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda repo_root, timeout_seconds=mod.DEFAULT_TIMEOUT_SECONDS: mod.GitHubCLIHealth(
            ready=False,
            auth_ok=False,
            api_ok=False,
            mode="connectivity_failed",
            error="error connecting to api.github.com",
            repo=str(Path(repo_root).resolve()),
        ),
    )

    exit_code = mod.main(["--json"])

    assert exit_code == 1
    out = capsys.readouterr().out
    assert '"mode": "connectivity_failed"' in out
    assert '"ready": false' in out
