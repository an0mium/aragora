"""Tests for ``scripts/guard_amend_pushed.sh`` (v13 lane P72)."""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "guard_amend_pushed.sh"


def _run(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)


def _init_repo_with_remote(tmp_path: Path) -> tuple[Path, Path]:
    """Create a bare ``origin`` and a working clone of it.

    Returns ``(clone_path, remote_path)``.
    """
    remote = tmp_path / "remote.git"
    _run(["git", "init", "--bare", "-b", "main", str(remote)], cwd=tmp_path)

    clone = tmp_path / "clone"
    _run(["git", "clone", str(remote), str(clone)], cwd=tmp_path)
    _run(["git", "config", "user.name", "Test User"], cwd=clone)
    _run(["git", "config", "user.email", "test@example.com"], cwd=clone)
    (clone / "README.md").write_text("hello\n", encoding="utf-8")
    _run(["git", "add", "README.md"], cwd=clone)
    _run(["git", "commit", "-m", "init"], cwd=clone)
    _run(["git", "push", "-u", "origin", "main"], cwd=clone)
    return clone, remote


def test_guard_blocks_when_head_equals_remote(tmp_path: Path) -> None:
    """Dangerous path: local HEAD == origin/main tip -> exit 1."""
    clone, _ = _init_repo_with_remote(tmp_path)

    proc = _run(["bash", str(SCRIPT)], cwd=clone)

    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    assert "AMEND-BLOCKED" in proc.stderr
    assert "origin/main" in proc.stderr
    assert "Use a new commit instead." in proc.stderr


def test_guard_allows_when_head_ahead_of_remote(tmp_path: Path) -> None:
    """Happy path: local HEAD is ahead of origin -> exit 0."""
    clone, _ = _init_repo_with_remote(tmp_path)
    (clone / "extra.txt").write_text("ahead\n", encoding="utf-8")
    _run(["git", "add", "extra.txt"], cwd=clone)
    _run(["git", "commit", "-m", "feat: ahead of origin"], cwd=clone)

    proc = _run(["bash", str(SCRIPT)], cwd=clone)

    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "AMEND-BLOCKED" not in proc.stderr
    assert "amend is safe" in proc.stdout


def test_guard_allows_when_branch_absent_remotely(tmp_path: Path) -> None:
    """Branch does not exist on the remote -> exit 0."""
    clone, _ = _init_repo_with_remote(tmp_path)
    _run(["git", "switch", "-c", "feature/never-pushed"], cwd=clone)
    (clone / "feature.txt").write_text("feature\n", encoding="utf-8")
    _run(["git", "add", "feature.txt"], cwd=clone)
    _run(["git", "commit", "-m", "feat: local-only branch"], cwd=clone)

    proc = _run(["bash", str(SCRIPT)], cwd=clone)

    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "not found remotely" in proc.stdout


def test_guard_respects_explicit_branch_flag(tmp_path: Path) -> None:
    """``--branch`` overrides the auto-detected branch."""
    clone, _ = _init_repo_with_remote(tmp_path)
    # main is already pushed; switch to a fresh local branch but ask about main.
    _run(["git", "switch", "-c", "scratch"], cwd=clone)
    (clone / "scratch.txt").write_text("scratch\n", encoding="utf-8")
    _run(["git", "add", "scratch.txt"], cwd=clone)
    _run(["git", "commit", "-m", "chore: scratch commit"], cwd=clone)
    # HEAD on scratch is local-only; asking about main with --branch should also
    # be safe because scratch's HEAD differs from origin/main's tip.
    proc = _run(
        ["bash", str(SCRIPT), "--branch", "main"],
        cwd=clone,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "amend is safe" in proc.stdout


def test_guard_help_flag_exits_zero(tmp_path: Path) -> None:
    """``--help`` prints usage and exits 0."""
    clone, _ = _init_repo_with_remote(tmp_path)
    proc = _run(["bash", str(SCRIPT), "--help"], cwd=clone)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "Usage:" in proc.stdout
    assert "R19" in proc.stdout
