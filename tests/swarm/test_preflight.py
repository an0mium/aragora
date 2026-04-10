from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from aragora.swarm import preflight


def test_work_order_and_worktree_path_keep_scope_bounded(tmp_path: Path) -> None:
    order = preflight._work_order("codex")

    assert order["target_agent"] == "codex"
    assert order["file_scope"] == ["scratch/preflight_worker_check.txt"]
    assert order["metadata"] == {"admin_approved": True}
    assert str(order["work_order_id"]).startswith("preflight-")
    assert preflight._worktree_path(tmp_path, "preflight/demo") == (
        tmp_path / ".worktrees" / "preflight-preflight-demo"
    )


def test_run_raises_with_process_detail_or_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(*args, **kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args[0], 1, stdout="", stderr="")

    monkeypatch.setattr(preflight.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Command failed: git status"):
        preflight._run(["git", "status"], cwd=tmp_path)


def test_run_worker_requires_commit_and_sets_full_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeLauncher:
        def __init__(self, *, config: object) -> None:
            captured["config"] = config

        async def launch_and_wait(
            self,
            work_order: dict[str, object],
            *,
            worktree_path: str,
            branch: str,
            timeout: float,
        ) -> SimpleNamespace:
            captured["work_order"] = work_order
            captured["worktree_path"] = worktree_path
            captured["branch"] = branch
            captured["timeout"] = timeout
            return SimpleNamespace(commit_shas=[])

    monkeypatch.setattr(preflight, "WorkerLauncher", FakeLauncher)

    with pytest.raises(RuntimeError, match="did not produce a commit"):
        asyncio.run(
            preflight._run_worker(
                repo_root=Path("/tmp/repo"),
                worktree_path=Path("/tmp/repo/.worktrees/preflight"),
                branch="preflight/demo",
                agent="codex",
            )
        )

    config = captured["config"]
    assert getattr(config, "allow_claude_dangerously_skip_permissions") is True
    assert getattr(config, "allow_codex_full_auto") is True
    assert captured["work_order"] == preflight._work_order("codex")
    assert captured["worktree_path"] == "/tmp/repo/.worktrees/preflight"
    assert captured["branch"] == "preflight/demo"
    assert captured["timeout"] == 900.0


def test_main_skip_publication_runs_worker_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree_path = preflight._worktree_path(tmp_path, "preflight/demo")
    run_calls: list[tuple[list[str], Path, dict[str, str] | None]] = []
    cleanup_calls: list[tuple[list[str], str]] = []
    worker_calls: list[tuple[Path, Path, str, str]] = []

    async def fake_run_worker(
        *, repo_root: Path, worktree_path: Path, branch: str, agent: str
    ) -> None:
        worker_calls.append((repo_root, worktree_path, branch, agent))

    def fake_run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        run_calls.append((cmd, cwd, env))

    def fake_cleanup(cmd: list[str], *, cwd: str, **kwargs) -> subprocess.CompletedProcess[str]:
        cleanup_calls.append((cmd, cwd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(preflight, "_branch_name", lambda: "preflight/demo")
    monkeypatch.setattr(preflight, "_run_worker", fake_run_worker)
    monkeypatch.setattr(preflight, "_run", fake_run)
    monkeypatch.setattr(preflight.subprocess, "run", fake_cleanup)
    monkeypatch.setattr(
        sys, "argv", ["preflight.py", "--repo-root", str(tmp_path), "--skip-publication"]
    )

    assert preflight.main() == 0
    assert run_calls == [
        (
            ["git", "worktree", "add", "-b", "preflight/demo", str(worktree_path), "main"],
            tmp_path,
            None,
        )
    ]
    assert worker_calls == [(tmp_path, worktree_path, "preflight/demo", "claude")]
    assert cleanup_calls == [
        (["git", "worktree", "remove", "--force", str(worktree_path)], str(tmp_path)),
        (["git", "branch", "-D", "preflight/demo"], str(tmp_path)),
    ]


def test_main_publication_path_pushes_and_manages_pr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree_path = preflight._worktree_path(tmp_path, "preflight/demo")
    run_calls: list[tuple[list[str], Path, dict[str, str] | None]] = []
    pr_calls: list[tuple[str, Path, str]] = []

    async def fake_run_worker(
        *, repo_root: Path, worktree_path: Path, branch: str, agent: str
    ) -> None:
        return None

    def fake_run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
        run_calls.append((cmd, cwd, env))

    def fake_create_pr(repo_root: Path, branch: str) -> None:
        pr_calls.append(("create", repo_root, branch))

    def fake_close_pr(repo_root: Path, branch: str) -> None:
        pr_calls.append(("close", repo_root, branch))

    monkeypatch.setattr(preflight, "_branch_name", lambda: "preflight/demo")
    monkeypatch.setattr(preflight, "_run_worker", fake_run_worker)
    monkeypatch.setattr(preflight, "_run", fake_run)
    monkeypatch.setattr(preflight, "_create_pr", fake_create_pr)
    monkeypatch.setattr(preflight, "_close_pr", fake_close_pr)
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda cmd, *, cwd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
    )
    monkeypatch.setattr(
        preflight,
        "git_safe_env",
        lambda: {"GIT_AUTHOR_NAME": "Codex", "GIT_AUTHOR_EMAIL": "codex@example.com"},
    )
    monkeypatch.setattr(
        sys, "argv", ["preflight.py", "--repo-root", str(tmp_path), "--agent", "codex"]
    )

    assert preflight.main() == 0
    assert run_calls == [
        (
            ["git", "worktree", "add", "-b", "preflight/demo", str(worktree_path), "main"],
            tmp_path,
            None,
        ),
        (
            ["git", "push", "origin", "HEAD"],
            worktree_path,
            {"GIT_AUTHOR_NAME": "Codex", "GIT_AUTHOR_EMAIL": "codex@example.com"},
        ),
    ]
    assert pr_calls == [
        ("create", tmp_path, "preflight/demo"),
        ("close", tmp_path, "preflight/demo"),
    ]
