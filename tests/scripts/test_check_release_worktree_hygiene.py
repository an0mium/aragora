from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_release_worktree_hygiene.sh"


def test_check_release_worktree_hygiene_help_exits_before_git_checks() -> None:
    proc = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "Usage:" in proc.stdout
    assert "origin/--help" not in proc.stdout
    assert "origin/--help" not in proc.stderr
