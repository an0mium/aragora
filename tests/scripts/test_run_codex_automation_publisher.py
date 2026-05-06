from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_publisher_wrapper_passes_shared_outbox_to_branch_publisher() -> None:
    script = (REPO_ROOT / "scripts" / "run_codex_automation_publisher.sh").read_text(
        encoding="utf-8"
    )

    assert "repo_root_available()" in script
    assert '--repo "${REPO_ROOT}"' in script
    assert '--outbox-dir "${HANDOFF_OUTBOX_DIR}"' in script


def test_publisher_wrapper_accepts_direct_dot_aragora_state_root() -> None:
    script = (REPO_ROOT / "scripts" / "run_codex_automation_publisher.sh").read_text(
        encoding="utf-8"
    )

    assert 'AUTOMATION_STATE_ROOT##*/}" == ".aragora"' in script
    assert 'AUTOMATION_STATE_ROOT="$(cd "${AUTOMATION_STATE_ROOT}/.." && pwd)"' in script


def test_launchd_installer_prefers_canonical_git_worktree_root() -> None:
    script = (REPO_ROOT / "scripts" / "install_codex_automation_publisher_launchd.sh").read_text(
        encoding="utf-8"
    )

    assert "ARAGORA_AUTOMATION_PUBLISHER_REPO_ROOT" in script
    assert 'git -C "${SCRIPT_REPO_ROOT}" worktree list --porcelain' in script
    assert 'REPO_ROOT="${CANONICAL_REPO_ROOT}"' in script
    assert 'LOG_PATH="${REPO_ROOT}/.aragora/overnight/codex-automation-publisher.log"' in script
