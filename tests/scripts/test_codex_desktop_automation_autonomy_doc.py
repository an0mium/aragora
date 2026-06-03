"""Contract checks for the Codex Desktop automation autonomy brief."""

from __future__ import annotations

from pathlib import Path


def test_backlog_startup_probe_requests_compact_examples() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    brief = repo_root / "docs" / "briefs" / "codex-desktop-automation-autonomy.md"
    text = brief.read_text(encoding="utf-8")

    command_line = next(
        line for line in text.splitlines() if "scripts/audit_codex_branch_backlog.py" in line
    )

    assert "--json --summary-only --examples 3 --outbox-dir" in command_line
    assert ".aragora/automation-outbox" in command_line
    assert ".aragora/automation-receipts" in command_line
