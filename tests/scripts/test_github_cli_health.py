from __future__ import annotations

import subprocess
from pathlib import Path

import scripts.github_cli_health as mod


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


def test_check_github_cli_health_detects_connectivity_failure(monkeypatch) -> None:
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/bin/gh")

    def fake_run(
        args: list[str], *, cwd: Path, timeout_seconds: int
    ) -> subprocess.CompletedProcess[str]:
        if args[:3] == ["gh", "auth", "status"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")
        return subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout="",
            stderr="error connecting to api.github.com",
        )

    monkeypatch.setattr(mod, "_run", fake_run)

    health = mod.check_github_cli_health(Path("."))

    assert health.ready is False
    assert health.mode == "connectivity_failed"
    assert health.auth_ok is True
    assert health.api_ok is False
    assert mod.is_github_connectivity_error(health.error) is True


def test_check_github_cli_health_detects_auth_failure(monkeypatch) -> None:
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/bin/gh")

    def fake_run(
        args: list[str], *, cwd: Path, timeout_seconds: int
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout="",
            stderr="You are not logged into any GitHub hosts. Run gh auth login.",
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
