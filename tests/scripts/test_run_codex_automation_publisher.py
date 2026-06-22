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


def test_publisher_wrapper_sets_unattended_guardrail_defaults() -> None:
    script = (REPO_ROOT / "scripts" / "run_codex_automation_publisher.sh").read_text(
        encoding="utf-8"
    )

    assert 'ARAGORA_AUTOMATION_MIN_FREE_GIB="${ARAGORA_AUTOMATION_MIN_FREE_GIB:-50}"' in script
    assert (
        'ARAGORA_AUTOMATION_CODEX_RSS_MAX_GIB="${ARAGORA_AUTOMATION_CODEX_RSS_MAX_GIB:-25}"'
        in script
    )
    assert (
        'ARAGORA_AUTOMATION_SPEND_DAILY_CAP_USD="${ARAGORA_AUTOMATION_SPEND_DAILY_CAP_USD:-200}"'
        in script
    )
    assert (
        'ARAGORA_AUTOMATION_SPEND_WEEKLY_CAP_USD="${ARAGORA_AUTOMATION_SPEND_WEEKLY_CAP_USD:-500}"'
        in script
    )


def test_publisher_wrapper_accepts_direct_dot_aragora_state_root() -> None:
    script = (REPO_ROOT / "scripts" / "run_codex_automation_publisher.sh").read_text(
        encoding="utf-8"
    )

    assert 'AUTOMATION_STATE_ROOT##*/}" == ".aragora"' in script
    assert 'AUTOMATION_STATE_ROOT="$(cd "${AUTOMATION_STATE_ROOT}/.." && pwd)"' in script


def test_publisher_wrapper_handles_missing_gh_as_unavailable() -> None:
    script = (REPO_ROOT / "scripts" / "run_codex_automation_publisher.sh").read_text(
        encoding="utf-8"
    )

    assert "command -v gh" not in script
    assert "gh CLI not found" not in script
    assert "GitHub unavailable; leaving automations in handoff-only mode" in script


def test_launchd_installer_prefers_canonical_git_worktree_root() -> None:
    script = (REPO_ROOT / "scripts" / "install_codex_automation_publisher_launchd.sh").read_text(
        encoding="utf-8"
    )

    assert "ARAGORA_AUTOMATION_PUBLISHER_REPO_ROOT" in script
    assert 'git -C "${SCRIPT_REPO_ROOT}" worktree list --porcelain' in script
    assert 'REPO_ROOT="${CANONICAL_REPO_ROOT}"' in script
    assert 'LOG_PATH="${REPO_ROOT}/.aragora/overnight/codex-automation-publisher.log"' in script
