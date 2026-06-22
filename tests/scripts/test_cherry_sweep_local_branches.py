"""Tests for scripts/cherry_sweep_local_branches.py.

These tests use a real, tiny ``git`` repository in ``tmp_path`` so the
``git cherry`` semantics that the script depends on are exercised
end-to-end, not stubbed. The repo layout:

- ``main`` carries commit A.
- ``origin/main`` (configured as a remote pointing at the same repo)
  also has commit A. We then add B onto ``main`` so ``origin/main``
  lags.
- ``feature-equiv`` branches off A, applies B', a copy of B's diff
  (created via ``git format-patch`` + ``git am``). Because the *patch*
  is identical, ``git cherry origin/main feature-equiv`` reports ``-``.
- ``feature-unique`` branches off A and adds a commit with a *different*
  diff, so ``git cherry`` reports ``+``.

The fourth test mocks the lane registry to verify that claimed lane
branches are skipped even when they are otherwise patch-equivalent.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Generator
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


@pytest.fixture(autouse=True)
def _setup_path() -> Generator[None, None, None]:
    sys.path.insert(0, str(SCRIPTS_DIR))
    yield
    sys.path.remove(str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Tmp repo fixture
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=check,
        capture_output=True,
        text=True,
        env={
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "HOME": str(repo.parent),
            "PATH": __import__("os").environ.get("PATH", ""),
        },
    )
    return proc


def _make_repo(tmp_path: Path) -> Path:
    """Build a working repo + a bare "origin" + branches required by tests.

    Returns the path to the working repo.
    """
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    work.mkdir()
    _git(work.parent, "init", "--bare", str(origin))
    _git(work, "init")
    _git(work, "checkout", "-b", "main")
    (work / "README").write_text("hello\n")
    _git(work, "add", "README")
    _git(work, "commit", "-m", "A: initial")
    _git(work, "remote", "add", "origin", str(origin))
    _git(work, "push", "-u", "origin", "main")

    # Commit B on main and PUSH so origin/main carries B's patch. The
    # `git cherry origin/main <branch>` check looks for patch-equivalent
    # commits *on origin/main*, so origin must have the equivalent
    # patch already.
    (work / "fileB.txt").write_text("B-content\n")
    _git(work, "add", "fileB.txt")
    _git(work, "commit", "-m", "B: add fileB")
    _git(work, "push", "origin", "main")

    # feature-equiv: branch from A (the parent of B), then cherry-pick
    # B so the *patch* matches the one on origin/main exactly.
    a_sha = _git(work, "rev-parse", "origin/main^").stdout.strip()
    b_sha = _git(work, "rev-parse", "origin/main").stdout.strip()
    _git(work, "checkout", "-b", "feature-equiv", a_sha)
    _git(work, "cherry-pick", b_sha)

    # feature-unique: branch from A, add a *different* commit.
    _git(work, "checkout", "-b", "feature-unique", a_sha)
    (work / "fileU.txt").write_text("U-content\n")
    _git(work, "add", "fileU.txt")
    _git(work, "commit", "-m", "U: add fileU (unique)")

    # claimed-equiv: another patch-equivalent branch, used by the
    # lane-registry-skip test.
    _git(work, "checkout", "-b", "claimed-equiv", a_sha)
    _git(work, "cherry-pick", b_sha)

    # Always end on main so worktree HEAD is main.
    _git(work, "checkout", "main")
    return work


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    return _make_repo(tmp_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_patch_equivalent_branch_is_candidate(tmp_repo: Path) -> None:
    """A branch whose every commit is patch-equivalent on origin/main is a candidate."""
    import cherry_sweep_local_branches as mod

    summary = mod.sweep(
        repo=tmp_repo,
        base="origin/main",
        apply=False,
        include_candidates=True,
    )
    assert "feature-equiv" in summary["candidates"]
    assert "feature-unique" not in summary["candidates"]
    assert summary["dry_run"] is True
    assert summary["applied"] is False
    assert summary["deleted"] == 0
    # main + feature-equiv + feature-unique + claimed-equiv = 4
    assert summary["scanned"] == 4
    assert summary["skipped_main"] == 1


def test_branch_with_unique_commit_is_preserved(tmp_repo: Path) -> None:
    """A branch with at least one unique commit is preserved, even in --apply."""
    import cherry_sweep_local_branches as mod

    summary = mod.sweep(
        repo=tmp_repo,
        base="origin/main",
        apply=True,
        include_candidates=True,
        include_preserved=True,
    )
    assert "feature-unique" not in summary["candidates"]
    assert "feature-unique" in summary["preserved"]
    assert summary["preserved_with_unique"] >= 1
    # Ensure feature-unique still exists after --apply.
    proc = subprocess.run(
        ["git", "branch", "--list", "feature-unique"],
        cwd=str(tmp_repo),
        capture_output=True,
        text=True,
    )
    assert "feature-unique" in proc.stdout


def test_main_branch_is_never_deleted(tmp_repo: Path) -> None:
    """``main`` must always be protected, regardless of cherry output."""
    import cherry_sweep_local_branches as mod

    summary = mod.sweep(
        repo=tmp_repo,
        base="origin/main",
        apply=True,
        include_candidates=True,
    )
    assert "main" not in summary.get("candidates", [])
    assert "main" not in summary.get("deleted_branches", [])
    assert summary["skipped_main"] == 1
    # main must still exist locally.
    proc = subprocess.run(
        ["git", "branch", "--list", "main"],
        cwd=str(tmp_repo),
        capture_output=True,
        text=True,
    )
    assert "main" in proc.stdout


def test_claimed_lane_branch_is_skipped(tmp_repo: Path, tmp_path: Path) -> None:
    """A patch-equivalent branch claimed in lanes.json must be skipped."""
    import cherry_sweep_local_branches as mod

    registry = tmp_path / "lanes.json"
    registry.write_text(
        json.dumps(
            [
                {
                    "lane_id": "demo-lane",
                    "owner_session": "session-1",
                    "branch": "claimed-equiv",
                    "status": "active",
                    "updated_at": "2026-05-18T19:00:00Z",
                }
            ]
        )
    )

    summary = mod.sweep(
        repo=tmp_repo,
        base="origin/main",
        apply=True,
        include_candidates=True,
        registry_path=registry,
    )
    assert summary["skipped_claim"] == 1
    assert "claimed-equiv" not in summary.get("candidates", [])
    assert "claimed-equiv" not in summary.get("deleted_branches", [])
    # Branch must still exist.
    proc = subprocess.run(
        ["git", "branch", "--list", "claimed-equiv"],
        cwd=str(tmp_repo),
        capture_output=True,
        text=True,
    )
    assert "claimed-equiv" in proc.stdout


def test_apply_actually_deletes_candidates(tmp_repo: Path) -> None:
    """--apply must run `git branch -D` on candidates and reflect deletion."""
    import cherry_sweep_local_branches as mod

    summary = mod.sweep(
        repo=tmp_repo,
        base="origin/main",
        apply=True,
        include_candidates=True,
    )
    assert "feature-equiv" in summary.get("candidates", [])
    assert "feature-equiv" in summary.get("deleted_branches", [])
    assert summary["deleted"] >= 1
    proc = subprocess.run(
        ["git", "branch", "--list", "feature-equiv"],
        cwd=str(tmp_repo),
        capture_output=True,
        text=True,
    )
    assert "feature-equiv" not in proc.stdout


def test_limit_caps_deletions(tmp_repo: Path) -> None:
    """``--limit`` caps the number of branches deleted in a single run."""
    import cherry_sweep_local_branches as mod

    # claimed-equiv and feature-equiv are both candidates without a
    # registry; limit=1 should only delete one of them.
    summary = mod.sweep(
        repo=tmp_repo,
        base="origin/main",
        apply=True,
        limit=1,
        include_candidates=True,
    )
    assert summary["candidate_count"] >= 2
    assert summary["deleted"] == 1


def test_tracked_remote_still_present_is_skipped() -> None:
    """A branch with a tracked, still-present remote must be skipped."""
    import cherry_sweep_local_branches as mod

    assert (
        mod.has_tracked_remote_still_present(
            "refs/remotes/origin/foo",
            "[ahead 1, behind 2]",
        )
        is True
    )
    # Gone upstream → deletion allowed.
    assert mod.has_tracked_remote_still_present("refs/remotes/origin/foo", "[gone]") is False
    # No upstream at all → deletion allowed.
    assert mod.has_tracked_remote_still_present("", "") is False


def test_worktree_bound_branch_is_skipped(tmp_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A branch checked out in a worktree is preserved even if patch-equivalent."""
    import cherry_sweep_local_branches as mod

    # Fake that feature-equiv is checked out by a worktree.
    monkeypatch.setattr(
        mod,
        "worktree_bound_branches",
        lambda _repo: {"main", "feature-equiv"},
    )

    summary = mod.sweep(
        repo=tmp_repo,
        base="origin/main",
        apply=True,
        include_candidates=True,
        worktree_lister=lambda _r: {"main", "feature-equiv"},
    )
    assert "feature-equiv" not in summary.get("candidates", [])
    assert summary["skipped_worktree"] >= 1
