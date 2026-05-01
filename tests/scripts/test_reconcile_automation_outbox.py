from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import scripts.reconcile_automation_outbox as mod


def _unhealthy_github() -> SimpleNamespace:
    return SimpleNamespace(
        ready=False,
        mode="connectivity_failed",
        error="error connecting to api.github.com",
    )


def _ready_github() -> SimpleNamespace:
    return SimpleNamespace(ready=True, mode="ready", error="")


def _write_outbox_handoff(outbox_dir: Path, *, branch: str, key: str) -> Path:
    outbox_dir.mkdir(parents=True, exist_ok=True)
    path = outbox_dir / f"{key}.json"
    path.write_text(
        json.dumps(
            {
                "task": f"Publish {branch}",
                "requires_github": True,
                "requested_action": {"type": "open_pr", "branch": branch},
                "repo": "synaptent/aragora",
                "idempotency_key": key,
            }
        ),
        encoding="utf-8",
    )
    return path


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


def test_explicit_dry_run_flag_keeps_read_only_default(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    monkeypatch.setattr(mod, "open_pr_heads", lambda *_args: {})

    rc = mod.main(["--repo", str(tmp_path), "--dry-run"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "mode: DRY-RUN" in out
    assert "DRY-RUN" in out
    assert not (tmp_path / ".aragora" / "cleanup-state").exists()


def test_json_output_reports_reconciliation_without_human_preamble(
    tmp_path: Path,
    capsys: Any,
) -> None:
    outbox_dir = tmp_path / ".aragora" / "automation-outbox"
    receipt_dir = tmp_path / ".aragora" / "automation-receipts"
    key = "open-pr-codex-json-output-abc123"
    _write_outbox_handoff(outbox_dir, branch="codex/json-output", key=key)
    receipt_dir.mkdir(parents=True)
    (receipt_dir / f"{key}.json").write_text(
        json.dumps({"idempotency_key": key, "status": "published"}),
        encoding="utf-8",
    )

    rc = mod.main(["--repo", str(tmp_path), "--json"])

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert rc == 0
    assert not out.startswith("outbox_dir:")
    assert payload["dry_run"] is True
    assert payload["applied"] is False
    assert payload["report"] is None
    assert payload["outbox_count"] == 1
    assert payload["terminal_receipt_count"] == 1
    assert payload["counts"]["satisfied_by_existing_receipt"] == 1
    assert payload["archived"] == 1
    assert payload["kept"] == 0
    assert payload["actions"][0]["branch"] == "codex/json-output"


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


def test_branch_from_payload_uses_list_local_evidence_mapping() -> None:
    payload = {
        "requested_action": "open_pr",
        "local_evidence": [
            "older handoffs sometimes stored local evidence as bullet text",
            {"branch": "codex/list-evidence"},
        ],
    }

    assert mod._branch_from_payload(payload) == "codex/list-evidence"


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


def test_apply_preserves_missing_branch_when_open_pr_state_unavailable(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    outbox_dir = tmp_path / ".aragora" / "automation-outbox"
    receipt_dir = tmp_path / ".aragora" / "automation-receipts"
    key = "open-pr-codex-missing-abc123"
    handoff = _write_outbox_handoff(outbox_dir, branch="codex/missing", key=key)
    receipt_dir.mkdir(parents=True)

    monkeypatch.setattr(mod, "check_github_cli_health", lambda _root: _unhealthy_github())
    monkeypatch.setattr(
        mod,
        "open_pr_heads",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("open PR fetch should be skipped when GitHub is unhealthy")
        ),
    )
    monkeypatch.setattr(
        mod,
        "run_git",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["git"], returncode=128, stdout="", stderr="missing ref"
        ),
    )

    assert mod.main(["--repo", str(tmp_path), "--apply"]) == 0

    reports = sorted((tmp_path / ".aragora" / "cleanup-state").glob("*.json"))
    payload = json.loads(reports[-1].read_text(encoding="utf-8"))
    assert payload["counts"]["blocked_missing_branch_open_pr_unknown"] == 1
    assert payload["counts"]["missing_branch"] == 0
    assert payload["actions"][0]["decision"] == "keep"
    assert "open PR state is unavailable" in payload["actions"][0]["reason"]
    assert handoff.exists()
    assert not (receipt_dir / f"{key}.json").exists()
    assert not list((tmp_path / ".aragora" / "automation-outbox-archive").glob("*.json"))


def test_missing_branch_archives_when_open_pr_state_is_available(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    outbox_dir = tmp_path / ".aragora" / "automation-outbox"
    key = "open-pr-codex-missing-abc123"
    _write_outbox_handoff(outbox_dir, branch="codex/missing", key=key)

    monkeypatch.setattr(mod, "check_github_cli_health", lambda _root: _ready_github())
    monkeypatch.setattr(mod, "open_pr_heads", lambda *_args: {})
    monkeypatch.setattr(
        mod,
        "run_git",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["git"], returncode=128, stdout="", stderr="missing ref"
        ),
    )

    assert mod.main(["--repo", str(tmp_path), "--write-report"]) == 0

    reports = sorted((tmp_path / ".aragora" / "cleanup-state").glob("*.json"))
    payload = json.loads(reports[-1].read_text(encoding="utf-8"))
    assert payload["counts"]["missing_branch"] == 1
    assert payload["counts"]["blocked_missing_branch_open_pr_unknown"] == 0
    assert payload["actions"][0]["decision"] == "archive"
    assert payload["actions"][0]["reason"] == "branch no longer exists"
