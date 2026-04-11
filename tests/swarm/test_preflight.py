from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from aragora.swarm import preflight as mod
from aragora.swarm.worker_contract import checksum_contract_payload
from aragora.swarm.worker_process import WorkerProcess


def _worker(*, branch: str, checksum: str | None = None) -> WorkerProcess:
    contract = {
        "runner_type": "codex-cli",
        "agent": "codex",
        "model": "gpt-5.4",
        "profile": "default",
        "permissions": {"allow_full_auto": True},
        "execution_mode": "autonomous",
        "git_auth_mode": "https",
        "gh_api_auth_mode": "none",
        "budget": {"max_wall_time_seconds": 900.0},
        "env_checksum": "env123",
        "contract_version": "1",
    }
    return WorkerProcess(
        work_order_id="preflight-1",
        agent="codex",
        worktree_path="/tmp/preflight",
        branch=branch,
        commit_shas=["deadbeef"],
        worker_contract=contract,
        worker_contract_checksum=checksum or checksum_contract_payload(contract),
    )


def test_run_preflight_returns_structured_result(monkeypatch, tmp_path: Path) -> None:
    branch = "preflight/20260411-test"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    commands: list[list[str]] = []
    cleanup_commands: list[list[str]] = []

    async def fake_run_worker(**_: object) -> WorkerProcess:
        return _worker(branch=branch)

    def fake_run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        commands.append(list(cmd))

    def fake_subprocess_run(cmd: list[str], **_: object) -> SimpleNamespace:
        cleanup_commands.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_branch_name", lambda: branch)
    monkeypatch.setattr(mod, "_run_worker", fake_run_worker)
    monkeypatch.setattr(mod, "_run", fake_run)
    monkeypatch.setattr(mod.subprocess, "run", fake_subprocess_run)

    result = mod.run_preflight(
        repo_root=repo_root,
        agent="codex",
        base_ref="origin/main",
        skip_publication=False,
    )

    expected_worktree = repo_root / ".worktrees" / "preflight-preflight-20260411-test"
    assert commands[0] == [
        "git",
        "worktree",
        "add",
        "-b",
        branch,
        str(expected_worktree),
        "origin/main",
    ]
    assert commands[1] == ["git", "push", "origin", "HEAD"]
    assert "--base" in commands[2]
    assert commands[2][commands[2].index("--base") + 1] == "origin/main"
    assert commands[3][:4] == ["gh", "pr", "close", "--repo"]
    assert result.published is True
    assert result.pull_request_created is True
    assert result.pull_request_closed is True
    assert result.cleanup_worktree_removed is True
    assert result.cleanup_branch_removed is True
    assert result.worker["worker_contract_checksum"] == checksum_contract_payload(
        result.worker["worker_contract"]
    )
    assert cleanup_commands[0] == ["git", "worktree", "remove", "--force", str(expected_worktree)]
    assert cleanup_commands[1] == ["git", "branch", "-D", branch]


def test_run_preflight_fails_closed_on_contract_checksum_mismatch(
    monkeypatch, tmp_path: Path
) -> None:
    branch = "preflight/20260411-bad-checksum"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    commands: list[list[str]] = []
    cleanup_commands: list[list[str]] = []

    async def fake_run_worker(**_: object) -> WorkerProcess:
        return _worker(branch=branch, checksum="bad-checksum")

    def fake_run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        commands.append(list(cmd))

    def fake_subprocess_run(cmd: list[str], **_: object) -> SimpleNamespace:
        cleanup_commands.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_branch_name", lambda: branch)
    monkeypatch.setattr(mod, "_run_worker", fake_run_worker)
    monkeypatch.setattr(mod, "_run", fake_run)
    monkeypatch.setattr(mod.subprocess, "run", fake_subprocess_run)

    with pytest.raises(RuntimeError, match="checksum"):
        mod.run_preflight(
            repo_root=repo_root,
            agent="codex",
            base_ref="main",
            skip_publication=False,
        )

    assert commands == [
        [
            "git",
            "worktree",
            "add",
            "-b",
            branch,
            str(repo_root / ".worktrees" / "preflight-preflight-20260411-bad-checksum"),
            "main",
        ]
    ]
    assert cleanup_commands[0][:4] == ["git", "worktree", "remove", "--force"]
    assert cleanup_commands[1] == ["git", "branch", "-D", branch]
