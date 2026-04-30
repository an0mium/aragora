from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import scripts.audit_branch_backlog_parallel as mod

UTC = timezone.utc


def _branch_row(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "head_sha": "abc1234",
        "committed_at": datetime.now(UTC).isoformat(),
        "ahead_count": "1",
        "subject": "fix automation classifier",
    }


def test_is_dirty_protects_existing_worktree_when_status_fails(
    tmp_path: Path, monkeypatch: Any
) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()

    def fake_run_git(
        args: list[str],
        cwd: Path,
        *,
        timeout: int = 60,
    ) -> subprocess.CompletedProcess[str]:
        assert args == ["status", "--porcelain"]
        assert cwd == worktree
        assert timeout == 15
        return subprocess.CompletedProcess(
            args=args, returncode=128, stdout="", stderr="bad gitdir"
        )

    monkeypatch.setattr(mod, "run_git", fake_run_git)

    assert mod._is_dirty(worktree) is True


def test_is_dirty_ignores_missing_stale_worktree_path(tmp_path: Path, monkeypatch: Any) -> None:
    missing = tmp_path / "missing-worktree"

    def fail_if_called(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise AssertionError("missing worktree paths should not run git status")

    monkeypatch.setattr(mod, "run_git", fail_if_called)

    assert mod._is_dirty(missing) is False


def test_has_active_session_recognizes_canonical_codex_marker(tmp_path: Path) -> None:
    (tmp_path / ".codex_session_active").write_text("active\n", encoding="utf-8")

    assert mod._has_active_session(tmp_path) is True


def test_has_active_session_preserves_parallel_legacy_marker(tmp_path: Path) -> None:
    (tmp_path / ".codex-session-active").write_text("active\n", encoding="utf-8")

    assert mod._has_active_session(tmp_path) is True


def test_classify_one_protects_status_failed_worktree(tmp_path: Path, monkeypatch: Any) -> None:
    branch = "codex/status-failed"
    worktree = tmp_path / "worktree"
    worktree.mkdir()

    monkeypatch.setattr(mod, "_is_dirty", lambda path: path == worktree)

    record = mod._classify_one(
        branch_row=_branch_row(branch),
        open_pr_branches={},
        receipted_branches=set(),
        outbox_branches=set(),
        worktrees={branch: [worktree]},
        merged=set(),
        remotes=set(),
        patch_equivalents={},
        handoff_patch_ids=set(),
        branch_patch_ids_map={},
        recent_threshold=datetime.now(UTC) - timedelta(hours=72),
    )

    assert record["category"] == mod.CATEGORY_PROTECTED_DIRTY_WT
    assert record["dirty"] is True
    assert record["worktree_paths"] == [str(worktree)]


def test_classify_one_protects_canonical_active_session_marker(tmp_path: Path) -> None:
    branch = "codex/active"
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / ".codex_session_active").write_text("active\n", encoding="utf-8")

    record = mod._classify_one(
        branch_row=_branch_row(branch),
        open_pr_branches={},
        receipted_branches=set(),
        outbox_branches=set(),
        worktrees={branch: [worktree]},
        merged=set(),
        remotes=set(),
        patch_equivalents={},
        handoff_patch_ids=set(),
        branch_patch_ids_map={},
        recent_threshold=datetime.now(UTC) - timedelta(hours=72),
    )

    assert record["category"] == mod.CATEGORY_PROTECTED_ACTIVE_WT
    assert record["active_session"] is True
    assert record["worktree_paths"] == [str(worktree)]
