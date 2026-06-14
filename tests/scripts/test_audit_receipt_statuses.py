from __future__ import annotations

from pathlib import Path

import scripts.audit_codex_branch_backlog as mod


def test_terminal_handoff_keys_accepts_completed_cleanup_status_variants(tmp_path: Path) -> None:
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    terminal_statuses = [
        "checkout_retired_branch_preserved",
        "checkout_unregistered_branch_preserved",
        "completed_with_anchor_residue",
        "removed_checkout_only",
        "removed_local_branch",
        "retired_checkout_only",
        "retired_local_branch",
        "retired_local_patch_equivalent",
    ]
    for index, status in enumerate(terminal_statuses):
        (receipts / f"receipt-{index}.json").write_text(
            f'{{"idempotency_key": "key-{index}", "status": "{status}"}}',
            encoding="utf-8",
        )
    (receipts / "blocked.json").write_text(
        '{"idempotency_key": "blocked", "status": "blocked"}',
        encoding="utf-8",
    )
    (receipts / "missing-status.json").write_text(
        '{"idempotency_key": "missing"}',
        encoding="utf-8",
    )

    assert mod.terminal_handoff_keys(receipts) == {
        f"key-{index}" for index in range(len(terminal_statuses))
    }
