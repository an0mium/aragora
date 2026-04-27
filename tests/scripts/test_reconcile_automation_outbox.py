from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import scripts.reconcile_automation_outbox as mod


def test_terminal_receipt_keys_falls_back_to_receipt_filename(tmp_path: Path) -> None:
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    key = "open-pr-codex-example-abc123"
    (receipt_dir / f"{key}.json").write_text(
        json.dumps({"status": "published"}),
        encoding="utf-8",
    )

    assert mod._terminal_receipt_keys(receipt_dir) == {key}


def test_dry_run_does_not_write_report_by_default(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    monkeypatch.setattr(mod, "open_pr_heads", lambda *_args: {})

    rc = mod.main(["--repo", str(tmp_path), "--base", "origin/main"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "report: not written in dry-run" in out
    assert not (tmp_path / ".aragora" / "cleanup-state").exists()


def test_dry_run_can_write_report_when_requested(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    monkeypatch.setattr(mod, "open_pr_heads", lambda *_args: {})

    rc = mod.main(["--repo", str(tmp_path), "--base", "origin/main", "--write-report"])

    out = capsys.readouterr().out
    reports = list((tmp_path / ".aragora" / "cleanup-state").glob("*.json"))
    assert rc == 0
    assert "report:" in out
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    assert payload["applied"] is False


def test_branch_from_payload_tolerates_list_local_evidence() -> None:
    payload = {
        "branch": "codex/openrouter-kimi-fallback-haiku",
        "local_evidence": [
            "older handoffs sometimes stored local evidence as bullet text",
        ],
    }

    assert mod._branch_from_payload(payload) == "codex/openrouter-kimi-fallback-haiku"


def test_branch_from_payload_prefers_structured_local_evidence() -> None:
    payload = {
        "branch": "codex/stale-top-level",
        "local_evidence": {"branch": "codex/structured"},
    }

    assert mod._branch_from_payload(payload) == "codex/structured"


def test_reconcile_existing_receipt_uses_structured_requested_action_branch(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    outbox_dir = tmp_path / ".aragora" / "automation-outbox"
    receipt_dir = tmp_path / ".aragora" / "automation-receipts"
    outbox_dir.mkdir(parents=True)
    receipt_dir.mkdir(parents=True)
    key = "open-pr-codex-structured-action-abc123"
    (outbox_dir / "structured-action.json").write_text(
        json.dumps(
            {
                "task": "Publish structured-action branch",
                "requires_github": True,
                "requested_action": {
                    "type": "open_pr",
                    "branch": "codex/structured-action",
                },
                "repo": "synaptent/aragora",
                "local_evidence": {},
                "validation": ["pytest tests/example.py -q"],
                "idempotency_key": key,
                "created_at": "2026-04-27T10:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (receipt_dir / f"{key}.json").write_text(
        json.dumps({"idempotency_key": key, "status": "published"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "open_pr_heads", lambda *_args: {})

    assert mod.main(["--repo", str(tmp_path), "--write-report"]) == 0

    reports = sorted((tmp_path / ".aragora" / "cleanup-state").glob("*.json"))
    payload = json.loads(reports[-1].read_text(encoding="utf-8"))
    assert payload["counts"]["satisfied_by_existing_receipt"] == 1
    assert payload["counts"]["skipped_unparseable"] == 0
    assert payload["actions"][0]["branch"] == "codex/structured-action"
