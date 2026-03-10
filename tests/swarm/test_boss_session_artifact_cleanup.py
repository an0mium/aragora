"""Focused regression tests for boss-side session artifact cleanup.

Issue #892 observed that a pure live Boss-loop run left ``.codex_session_meta.json``
behind in the boss worktree after completion. These tests cover the detached
collection path used by Boss-supervised runs and prove that session artifacts
do not remain as future worktree dirt once results are collected.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from aragora.swarm.worker_launcher import WorkerLauncher


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "git", "init", "-b", "main")
    _run(repo, "git", "config", "user.email", "test@example.com")
    _run(repo, "git", "config", "user.name", "Test User")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _run(repo, "git", "add", "README.md")
    _run(repo, "git", "commit", "-m", "initial")
    _run(repo, "git", "remote", "add", "origin", str(repo))
    _run(repo, "git", "update-ref", "refs/remotes/origin/main", "HEAD")
    return repo


def _run(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(args), cwd=cwd, text=True, capture_output=True, check=True)


def _write_terminal_session_meta(repo: Path, *, exit_code: int = 0) -> None:
    payload = {
        "pid": 12345,
        "session_id": "boss-session",
        "agent": "codex",
        "branch": "main",
        "worktree_path": str(repo),
        "started_at": "2026-03-09T23:16:00Z",
        "ended_at": "2026-03-09T23:20:00Z",
        "exit_code": exit_code,
    }
    (repo / ".codex_session_meta.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (repo / ".codex_session.log").write_text("boss loop log\n", encoding="utf-8")
    (repo / ".codex_session_active").write_text("pid=12345\n", encoding="utf-8")


class TestBossDetachedSessionArtifactCleanup:
    def test_detached_collection_removes_session_artifacts(self, repo: Path) -> None:
        initial_head = _run(repo, "git", "rev-parse", "HEAD").stdout.strip()
        _write_terminal_session_meta(repo)
        (repo / "real_work.py").write_text("print('ok')\n", encoding="utf-8")

        result = asyncio.run(
            WorkerLauncher.collect_detached_result(
                work_order_id="boss-cleanup",
                agent="codex",
                worktree_path=str(repo),
                branch="main",
                initial_head=initial_head,
                auto_commit=False,
            )
        )

        assert result is not None
        assert result.changed_paths == ["real_work.py"]
        assert not (repo / ".codex_session_meta.json").exists()
        assert not (repo / ".codex_session.log").exists()
        assert not (repo / ".codex_session_active").exists()

        status_lines = _run(repo, "git", "status", "--porcelain").stdout.splitlines()
        assert any("real_work.py" in line for line in status_lines)
        assert all(".codex_session_" not in line for line in status_lines)

    def test_artifact_only_detached_collection_leaves_clean_worktree(self, repo: Path) -> None:
        initial_head = _run(repo, "git", "rev-parse", "HEAD").stdout.strip()
        _write_terminal_session_meta(repo)

        result = asyncio.run(
            WorkerLauncher.collect_detached_result(
                work_order_id="boss-artifacts-only",
                agent="codex",
                worktree_path=str(repo),
                branch="main",
                initial_head=initial_head,
                auto_commit=False,
            )
        )

        assert result is not None
        assert result.changed_paths == []
        assert result.commit_shas == []
        assert not (repo / ".codex_session_meta.json").exists()
        assert not (repo / ".codex_session.log").exists()
        assert not (repo / ".codex_session_active").exists()
        assert _run(repo, "git", "status", "--porcelain").stdout.strip() == ""
