from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import scripts.handoff_state as handoff_state
import scripts.reconcile_automation_outbox as reconcile


HEAD = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def _write_outbox(repo: Path, *, key: str = "open-pr-codex-example-aaaaaaaa") -> Path:
    outbox_dir = repo / ".aragora" / "automation-outbox"
    outbox_dir.mkdir(parents=True, exist_ok=True)
    path = outbox_dir / f"{key}.json"
    path.write_text(
        json.dumps(
            {
                "branch": "codex/example",
                "desired_head_sha": HEAD,
                "head_sha": HEAD,
                "idempotency_key": key,
                "repo": "synaptent/aragora",
                "requested_action": {
                    "type": "open_or_update_pr",
                    "branch": "codex/example",
                    "desired_head_sha": HEAD,
                },
                "task": "Open PR for codex/example",
            }
        ),
        encoding="utf-8",
    )
    return path


def _classifier_payload(outbox_file: str) -> dict[str, Any]:
    return {
        "counts": {handoff_state.HandoffState.REPRESENTED_BY_EXACT_OPEN_PR.value: 1},
        "github": {"mode": "ready", "error": None},
        "items": [
            {
                "branch": "codex/example",
                "desired_head_sha": HEAD,
                "evidence": {
                    "github": {
                        "exact_open_pr": {
                            "draft": True,
                            "head": "codex/example",
                            "head_sha": HEAD,
                            "html_url": "https://github.com/synaptent/aragora/pull/8589",
                            "number": 8589,
                            "state": "open",
                        }
                    }
                },
                "idempotency_key": "open-pr-codex-example-aaaaaaaa",
                "next_mutation_candidate": "write_representation_receipt_then_archive",
                "outbox_file": outbox_file,
                "reason": "branch has exact-head open PR #8589",
                "safe_to_mutate": False,
                "state": handoff_state.HandoffState.REPRESENTED_BY_EXACT_OPEN_PR.value,
            }
        ],
        "outbox_count": 1,
    }


def test_exact_open_pr_representation_dry_run_reports_archive_without_writes(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    outbox_file = _write_outbox(tmp_path)

    def fake_classify_handoffs(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["outbox_file"] == outbox_file.name
        assert kwargs["github_repo"] == "synaptent/aragora"
        return _classifier_payload(outbox_file.name)

    monkeypatch.setattr(reconcile, "classify_handoffs", fake_classify_handoffs)

    rc = reconcile.main(
        [
            "--repo",
            str(tmp_path),
            "--state-root",
            str(tmp_path),
            "--outbox-file",
            outbox_file.name,
            "--dry-run",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["dry_run"] is True
    assert payload["archived"] == 1
    assert payload["counts"]["satisfied_by_exact_open_pr_representation"] == 1
    assert payload["actions"][0]["decision"] == "archive"
    assert payload["actions"][0]["synthetic_receipt"] is True
    assert payload["actions"][0]["representation_pr"]["number"] == 8589
    assert outbox_file.exists()
    assert not (tmp_path / ".aragora" / "automation-receipts" / f"{outbox_file.stem}.json").exists()


def test_exact_open_pr_representation_apply_writes_receipt_and_archives(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    outbox_file = _write_outbox(tmp_path)

    def fake_classify_handoffs(**kwargs: Any) -> dict[str, Any]:
        return _classifier_payload(outbox_file.name)

    monkeypatch.setattr(reconcile, "classify_handoffs", fake_classify_handoffs)

    rc = reconcile.main(
        [
            "--repo",
            str(tmp_path),
            "--state-root",
            str(tmp_path),
            "--outbox-file",
            outbox_file.name,
            "--apply",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["dry_run"] is False
    assert payload["archived"] == 1
    assert not outbox_file.exists()
    assert (tmp_path / ".aragora" / "automation-outbox-archive" / outbox_file.name).exists()
    receipt_path = tmp_path / ".aragora" / "automation-receipts" / f"{outbox_file.stem}.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "already_satisfied"
    assert receipt["reason"] == "exact_open_pr_representation"
    assert receipt["existing_pr_url"] == "https://github.com/synaptent/aragora/pull/8589"
    assert receipt["synthetic"] is True
