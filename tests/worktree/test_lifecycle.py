"""Tests for shared worktree lifecycle service."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock

from aragora.worktree.lifecycle import WorktreeLifecycleService


def test_discover_managed_dirs_defaults(tmp_path: Path) -> None:
    base = tmp_path / ".worktrees"
    (base / "codex-auto").mkdir(parents=True)
    (base / "codex-auto-ci").mkdir(parents=True)
    (base / "codex-auto-debate").mkdir(parents=True)

    service = WorktreeLifecycleService(repo_root=tmp_path)
    found = service.discover_managed_dirs()

    assert ".worktrees/codex-auto" in found
    assert ".worktrees/codex-auto-ci" in found
    assert ".worktrees/codex-auto-debate" in found


def test_maintain_managed_dirs_skips_active_and_missing(tmp_path: Path) -> None:
    active_dir = tmp_path / ".worktrees" / "codex-auto-active"
    ok_dir = tmp_path / ".worktrees" / "codex-auto-ok"
    active_dir.mkdir(parents=True)
    ok_dir.mkdir(parents=True)
    (active_dir / ".codex_session_active").write_text("1\n", encoding="utf-8")

    service = WorktreeLifecycleService(repo_root=tmp_path)
    service.run_autopilot_action = MagicMock(
        return_value=argparse.Namespace(returncode=0, stdout='{"ok": true}', stderr="")
    )

    summary = service.maintain_managed_dirs(
        managed_dirs=[
            ".worktrees/codex-auto-active",
            ".worktrees/codex-auto-ok",
            ".worktrees/codex-auto-missing",
        ],
        reconcile_only=True,
    )

    assert summary["ok"] is True
    assert summary["directories_total"] == 3
    assert summary["processed"] == 1
    assert summary["skipped_active"] == 1
    assert summary["skipped_missing"] == 1
    assert summary["failures"] == 0
    assert any(r.get("action") == "reconcile" for r in summary["results"] if r["status"] == "ok")


def test_maintain_managed_dirs_tracks_failures(tmp_path: Path) -> None:
    managed = tmp_path / ".worktrees" / "codex-auto"
    managed.mkdir(parents=True)

    service = WorktreeLifecycleService(repo_root=tmp_path)
    service.run_autopilot_action = MagicMock(
        return_value=argparse.Namespace(returncode=2, stdout='{"ok": false}', stderr="conflict")
    )

    summary = service.maintain_managed_dirs(
        managed_dirs=[".worktrees/codex-auto"],
        reconcile_only=False,
    )

    assert summary["ok"] is False
    assert summary["processed"] == 1
    assert summary["failures"] == 1
    row = summary["results"][0]
    assert row["status"] == "failed"
    assert row["action"] == "maintain"


def test_create_worktree_uses_git_runner() -> None:
    service = WorktreeLifecycleService(repo_root=Path("/tmp/repo"))
    git_runner = MagicMock(return_value=argparse.Namespace(returncode=0, stdout="ok", stderr=""))

    result = service.create_worktree(
        worktree_path=Path("/tmp/repo/.worktrees/dev-test"),
        ref="main",
        branch="dev/test",
        git_runner=git_runner,
    )

    assert result.success is True
    git_runner.assert_called_once()
    call = git_runner.call_args
    assert call.args[:4] == ("worktree", "add", "-b", "dev/test")
    assert call.kwargs["check"] is False


def test_remove_worktree_force_uses_git_runner() -> None:
    service = WorktreeLifecycleService(repo_root=Path("/tmp/repo"))
    git_runner = MagicMock(return_value=argparse.Namespace(returncode=0, stdout="", stderr=""))

    result = service.remove_worktree(
        worktree_path=Path("/tmp/repo/.worktrees/dev-test"),
        force=True,
        git_runner=git_runner,
    )

    assert result.success is True
    call = git_runner.call_args
    assert "--force" in call.args
