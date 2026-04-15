from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_studio_health_uses_operator_status_and_git_worktree_count() -> None:
    script = (REPO_ROOT / "scripts" / "studio-health.sh").read_text(encoding="utf-8")

    assert "git worktree list --porcelain" in script
    assert "from aragora.cli.commands.swarm_status import load_operator_status" in script
    assert "echo '--- Operator Truth ---'" in script
    assert "find .worktrees -maxdepth 2 -type d" not in script
