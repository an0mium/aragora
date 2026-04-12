from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


@pytest.fixture(autouse=True)
def _setup_path():
    sys.path.insert(0, str(SCRIPTS_DIR))
    yield
    sys.path.remove(str(SCRIPTS_DIR))


def test_cleanup_worktree_uses_safe_cleanup_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nomic_eval as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "scripts").mkdir()
    (repo_root / "scripts" / "safe_worktree_cleanup.py").write_text("# stub\n")
    worktree_dir = tmp_path / "wt"

    calls: list[list[str]] = []

    def _fake_run(cmd, cwd=None, env=None):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout='{"status":"removed"}', stderr=""
        )

    monkeypatch.setattr(mod, "_run", _fake_run)

    mod._cleanup_worktree(repo_root, worktree_dir, "codex/test")

    assert calls == [
        [
            sys.executable,
            str(repo_root / "scripts" / "safe_worktree_cleanup.py"),
            "--repo",
            str(repo_root),
            "remove",
            str(worktree_dir),
            "--branch",
            "codex/test",
            "--delete-branch",
            "--purge-path",
            "--json",
        ]
    ]


def test_cleanup_worktree_raises_when_safe_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nomic_eval as mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "scripts").mkdir()
    (repo_root / "scripts" / "safe_worktree_cleanup.py").write_text("# stub\n")

    monkeypatch.setattr(
        mod,
        "_run",
        lambda cmd, cwd=None, env=None: subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout='{"status":"blocked"}',
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="blocked"):
        mod._cleanup_worktree(repo_root, tmp_path / "wt", "codex/test")


def test_nomic_eval_help_runs_from_repo_root() -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "scripts/nomic_eval.py", "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "usage: nomic_eval.py" in result.stdout
