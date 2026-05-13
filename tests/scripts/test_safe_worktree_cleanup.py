"""Unit tests for scripts/safe_worktree_cleanup.py."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


@pytest.fixture(autouse=True)
def _setup_path():
    sys.path.insert(0, str(SCRIPTS_DIR))
    yield
    sys.path.remove(str(SCRIPTS_DIR))


def test_inspect_reports_open_pr_blocker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import safe_worktree_cleanup as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    worktree = tmp_path / "wt"
    worktree.mkdir()

    monkeypatch.setattr(mod.autopilot, "_repo_root_from", lambda _path: repo_root)
    monkeypatch.setattr(
        mod.autopilot,
        "_get_worktree_entries",
        lambda _repo: [mod.autopilot.WorktreeEntry(path=worktree, branch="codex/test")],
    )
    monkeypatch.setattr(mod.autopilot, "_has_active_session", lambda _path: False)
    monkeypatch.setattr(mod, "_worktree_is_dirty", lambda _path: False)
    monkeypatch.setattr(mod, "_unique_commits_ahead_of_main", lambda _repo, _branch: (0, False))
    monkeypatch.setattr(
        mod,
        "_lookup_open_prs",
        lambda _repo, _branch: (
            [{"number": 1361, "title": "Open PR", "url": "https://example.com/pr/1361"}],
            False,
        ),
    )

    inspection = mod.inspect_worktree(repo_root, worktree)

    assert inspection.tracked_worktree is True
    assert inspection.branch == "codex/test"
    assert inspection.dirty is False
    assert inspection.unique_commits_ahead == 0
    assert inspection.blockers == ["open_pr"]


def test_remove_refuses_blocked_worktree_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import safe_worktree_cleanup as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    worktree = tmp_path / "wt"
    worktree.mkdir()

    inspection = mod.WorktreeInspection(
        path=str(worktree),
        exists=True,
        tracked_worktree=True,
        branch="codex/test",
        active_session=False,
        lock_files=[],
        dirty=False,
        unique_commits_ahead=0,
        ahead_lookup_failed=False,
        patch_equivalent_to_origin_main=False,
        patch_equivalence_lookup_failed=False,
        open_prs=[{"number": 1361, "title": "Open PR", "url": "https://example.com/pr/1361"}],
        pr_lookup_failed=False,
        blockers=["open_pr"],
    )
    monkeypatch.setattr(
        mod,
        "inspect_worktree",
        lambda _repo, _path, branch_override=None: inspection,
    )
    monkeypatch.setattr(mod.autopilot, "_repo_root_from", lambda _path: repo_root)

    args = argparse.Namespace(
        repo=".",
        path=str(worktree),
        branch=None,
        delete_branch=False,
        purge_path=False,
        force=False,
        json=True,
    )

    rc = mod.cmd_remove(args)
    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["status"] == "blocked"
    assert payload["blockers"] == ["open_pr"]


def test_inspect_accepts_branch_override_for_orphaned_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import safe_worktree_cleanup as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    orphan_path = tmp_path / "manual-orphan"
    orphan_path.mkdir()

    monkeypatch.setattr(mod.autopilot, "_get_worktree_entries", lambda _repo: [])
    monkeypatch.setattr(mod.autopilot, "_has_active_session", lambda _path: False)
    monkeypatch.setattr(mod, "_worktree_is_dirty", lambda _path: False)
    monkeypatch.setattr(mod, "_unique_commits_ahead_of_main", lambda _repo, _branch: (0, False))
    monkeypatch.setattr(
        mod,
        "_lookup_open_prs",
        lambda _repo, branch: (
            [{"number": 2000, "title": branch, "url": "https://example.com/pr/2000"}],
            False,
        ),
    )

    inspection = mod.inspect_worktree(
        repo_root,
        orphan_path,
        branch_override="codex/orphaned-branch",
    )

    assert inspection.tracked_worktree is False
    assert inspection.branch == "codex/orphaned-branch"
    assert inspection.blockers == ["open_pr"]


def test_branch_detection_requires_local_git_metadata(tmp_path: Path) -> None:
    import safe_worktree_cleanup as mod

    orphan_path = tmp_path / "orphan"
    orphan_path.mkdir()

    assert mod._branch_for_path(orphan_path, None) is None


def test_remove_purges_residual_path_after_failed_git_remove(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import safe_worktree_cleanup as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    worktree = tmp_path / "wt"
    residual = worktree / "aragora" / "live" / ".next"
    residual.mkdir(parents=True)

    inspection = mod.WorktreeInspection(
        path=str(worktree),
        exists=True,
        tracked_worktree=True,
        branch="codex/test",
        active_session=False,
        lock_files=[],
        dirty=False,
        unique_commits_ahead=0,
        ahead_lookup_failed=False,
        patch_equivalent_to_origin_main=False,
        patch_equivalence_lookup_failed=False,
        open_prs=[],
        pr_lookup_failed=False,
        blockers=[],
    )

    monkeypatch.setattr(
        mod.autopilot,
        "_run_git",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["git", "worktree", "remove"],
            returncode=255,
            stdout="",
            stderr="Directory not empty",
        ),
    )
    monkeypatch.setattr(mod.autopilot, "_branch_exists", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(mod.autopilot, "_repo_root_from", lambda _path: repo_root)

    result = mod.remove_worktree(
        repo_root,
        inspection,
        delete_branch=False,
        purge_path=True,
        force=False,
    )

    assert result["status"] == "purged_after_failed_remove"
    assert result["path_purged"] is True
    assert worktree.exists() is False


def test_remove_deletes_branch_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import safe_worktree_cleanup as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    worktree = tmp_path / "wt"
    worktree.mkdir()

    inspection = mod.WorktreeInspection(
        path=str(worktree),
        exists=True,
        tracked_worktree=False,
        branch="codex/test",
        active_session=False,
        lock_files=[],
        dirty=False,
        unique_commits_ahead=0,
        ahead_lookup_failed=False,
        patch_equivalent_to_origin_main=False,
        patch_equivalence_lookup_failed=False,
        open_prs=[],
        pr_lookup_failed=False,
        blockers=[],
    )
    monkeypatch.setattr(mod.autopilot, "_branch_exists", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        mod.autopilot,
        "_run_git",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["git", "branch", "-D", "codex/test"],
            returncode=0,
            stdout="Deleted branch codex/test\n",
            stderr="",
        ),
    )

    result = mod.remove_worktree(
        repo_root,
        inspection,
        delete_branch=True,
        purge_path=True,
        force=False,
    )

    assert result["branch_deleted"] is True
    assert result["status"] == "purged"


def test_inspect_blocks_dirty_and_ahead_worktrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import safe_worktree_cleanup as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    worktree = tmp_path / "wt"
    worktree.mkdir()

    monkeypatch.setattr(mod.autopilot, "_repo_root_from", lambda _path: repo_root)
    monkeypatch.setattr(
        mod.autopilot,
        "_get_worktree_entries",
        lambda _repo: [mod.autopilot.WorktreeEntry(path=worktree, branch="codex/test")],
    )
    monkeypatch.setattr(mod.autopilot, "_has_active_session", lambda _path: False)
    monkeypatch.setattr(mod, "_worktree_is_dirty", lambda _path: True)
    monkeypatch.setattr(mod, "_unique_commits_ahead_of_main", lambda _repo, _branch: (2, False))
    monkeypatch.setattr(mod, "_lookup_open_prs", lambda _repo, _branch: ([], False))

    inspection = mod.inspect_worktree(repo_root, worktree)

    assert inspection.dirty is True
    assert inspection.unique_commits_ahead == 2
    assert inspection.blockers == ["dirty_worktree", "branch_ahead_of_origin_main"]


def test_inspect_allows_pr_lookup_failure_for_branch_with_no_unique_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import safe_worktree_cleanup as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    worktree = tmp_path / "wt"
    worktree.mkdir()

    monkeypatch.setattr(mod.autopilot, "_repo_root_from", lambda _path: repo_root)
    monkeypatch.setattr(
        mod.autopilot,
        "_get_worktree_entries",
        lambda _repo: [mod.autopilot.WorktreeEntry(path=worktree, branch="codex/merged")],
    )
    monkeypatch.setattr(mod.autopilot, "_has_active_session", lambda _path: False)
    monkeypatch.setattr(mod, "_worktree_is_dirty", lambda _path: False)
    monkeypatch.setattr(mod, "_unique_commits_ahead_of_main", lambda _repo, _branch: (0, False))
    monkeypatch.setattr(mod, "_lookup_open_prs", lambda _repo, _branch: ([], True))

    inspection = mod.inspect_worktree(repo_root, worktree)

    assert inspection.pr_lookup_failed is True
    assert inspection.unique_commits_ahead == 0
    assert inspection.blockers == []


def test_inspect_allows_patch_equivalent_branch_when_pr_lookup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import safe_worktree_cleanup as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    worktree = tmp_path / "wt"
    worktree.mkdir()

    monkeypatch.setattr(mod.autopilot, "_repo_root_from", lambda _path: repo_root)
    monkeypatch.setattr(
        mod.autopilot,
        "_get_worktree_entries",
        lambda _repo: [mod.autopilot.WorktreeEntry(path=worktree, branch="codex/replayed")],
    )
    monkeypatch.setattr(mod.autopilot, "_has_active_session", lambda _path: False)
    monkeypatch.setattr(mod, "_worktree_is_dirty", lambda _path: False)
    monkeypatch.setattr(mod, "_unique_commits_ahead_of_main", lambda _repo, _branch: (4, False))
    monkeypatch.setattr(mod, "_patch_equivalent_to_main", lambda _repo, _branch: (True, False))
    monkeypatch.setattr(mod, "_lookup_open_prs", lambda _repo, _branch: ([], True))

    inspection = mod.inspect_worktree(repo_root, worktree)

    assert inspection.pr_lookup_failed is True
    assert inspection.unique_commits_ahead == 4
    assert inspection.patch_equivalent_to_origin_main is True
    assert inspection.blockers == []


def test_inspect_blocks_lock_files_and_history_lookup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import safe_worktree_cleanup as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".codex_session_active").write_text("active\n")

    monkeypatch.setattr(mod.autopilot, "_repo_root_from", lambda _path: repo_root)
    monkeypatch.setattr(
        mod.autopilot,
        "_get_worktree_entries",
        lambda _repo: [mod.autopilot.WorktreeEntry(path=worktree, branch="codex/test")],
    )
    monkeypatch.setattr(mod.autopilot, "_has_active_session", lambda _path: False)
    monkeypatch.setattr(mod, "_worktree_is_dirty", lambda _path: False)
    monkeypatch.setattr(mod, "_unique_commits_ahead_of_main", lambda _repo, _branch: (0, True))
    monkeypatch.setattr(mod, "_lookup_open_prs", lambda _repo, _branch: ([], False))

    inspection = mod.inspect_worktree(repo_root, worktree)

    assert inspection.lock_files == [".codex_session_active"]
    assert inspection.ahead_lookup_failed is True
    assert inspection.blockers == ["session_lock_present", "ahead_lookup_failed"]
