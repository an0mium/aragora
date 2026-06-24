from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

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
                "safe_to_mutate": True,
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

    def fake_reverify(**kwargs: Any) -> tuple[dict[str, Any] | None, str | None]:
        return {
            "number": 8589,
            "url": "https://github.com/synaptent/aragora/pull/8589",
            "head": "codex/example",
            "head_sha": HEAD,
            "draft": True,
            "state": "open",
            "apply_reverified": True,
        }, None

    monkeypatch.setattr(reconcile, "_verify_exact_open_pr_representation", fake_reverify)

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


def test_exact_open_pr_representation_requires_safe_to_mutate(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    outbox_file = _write_outbox(tmp_path)

    def fake_classify_handoffs(**kwargs: Any) -> dict[str, Any]:
        payload = _classifier_payload(outbox_file.name)
        payload["items"][0]["safe_to_mutate"] = False
        return payload

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

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["archived"] == 0
    assert payload["counts"]["blocked_exact_open_pr_representation"] == 1
    assert payload["actions"][0]["decision"] == "keep"
    assert "safe_to_mutate" in payload["actions"][0]["reason"]
    assert outbox_file.exists()


def test_exact_open_pr_representation_apply_reverifies_before_mutating(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    outbox_file = _write_outbox(tmp_path)
    calls: list[dict[str, Any]] = []

    def fake_classify_handoffs(**kwargs: Any) -> dict[str, Any]:
        return _classifier_payload(outbox_file.name)

    def fake_reverify(**kwargs: Any) -> tuple[dict[str, Any] | None, str | None]:
        calls.append(dict(kwargs))
        representation = dict(kwargs["representation"])
        representation["apply_reverified"] = True
        return representation, None

    monkeypatch.setattr(reconcile, "classify_handoffs", fake_classify_handoffs)
    monkeypatch.setattr(reconcile, "_verify_exact_open_pr_representation", fake_reverify)

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

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["archived"] == 1
    assert calls
    assert calls[0]["branch"] == "codex/example"
    assert calls[0]["desired_head"] == HEAD
    assert not outbox_file.exists()


def test_exact_open_pr_representation_apply_blocks_when_reverify_fails(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    outbox_file = _write_outbox(tmp_path)

    def fake_classify_handoffs(**kwargs: Any) -> dict[str, Any]:
        return _classifier_payload(outbox_file.name)

    def fake_reverify(**kwargs: Any) -> tuple[dict[str, Any] | None, str | None]:
        return None, "exact-open-pr apply reverify failed (PR head changed)"

    monkeypatch.setattr(reconcile, "classify_handoffs", fake_classify_handoffs)
    monkeypatch.setattr(reconcile, "_verify_exact_open_pr_representation", fake_reverify)

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

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["archived"] == 0
    assert payload["counts"]["blocked_exact_open_pr_representation"] == 1
    assert payload["actions"][0]["decision"] == "keep"
    assert "PR head changed" in payload["actions"][0]["reason"]
    assert outbox_file.exists()
    assert not (tmp_path / ".aragora" / "automation-receipts" / f"{outbox_file.stem}.json").exists()


def test_exact_open_pr_representation_receipt_failure_keeps_live_outbox(
    tmp_path: Path, monkeypatch: Any
) -> None:
    outbox_file = _write_outbox(tmp_path)

    def fake_classify_handoffs(**kwargs: Any) -> dict[str, Any]:
        return _classifier_payload(outbox_file.name)

    def fake_reverify(**kwargs: Any) -> tuple[dict[str, Any] | None, str | None]:
        representation = dict(kwargs["representation"])
        representation["apply_reverified"] = True
        return representation, None

    def fail_receipt(**kwargs: Any) -> Path:
        raise OSError("receipt failed")

    monkeypatch.setattr(reconcile, "classify_handoffs", fake_classify_handoffs)
    monkeypatch.setattr(reconcile, "_verify_exact_open_pr_representation", fake_reverify)
    monkeypatch.setattr(reconcile, "_write_synthetic_receipt", fail_receipt)

    with pytest.raises(OSError):
        reconcile.main(
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

    assert outbox_file.exists()
    assert not (tmp_path / ".aragora" / "automation-receipts" / f"{outbox_file.stem}.json").exists()


def test_update_pr_idempotency_key_is_pr_publication_request() -> None:
    assert reconcile._is_pr_publication_request(
        {"idempotency_key": "update-pr-codex-example-abc123"}
    )
