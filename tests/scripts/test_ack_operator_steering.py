"""Tests for ``scripts/ack_operator_steering.py``."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_module(script_name: str) -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ack = _load_module("ack_operator_steering.py")
steering = _load_module("send_operator_steering.py")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_message(inbox: Path, filename: str = "message.json") -> dict[str, Any]:
    message = steering.build_message(
        to_session=inbox.name,
        body="PR is terminal; release the lane.",
        pr_hint=8610,
        sent_at_utc="2026-07-06T14:00:00.000Z",
    )
    _write_json(inbox / filename, message)
    return message


def _write_receipt(
    inbox: Path,
    *,
    message_filename: str = "message.json",
    message_sha256: str,
    outcome: str = "superseded",
    filename: str = "receipt.json",
) -> Path:
    path = inbox / "_read_receipts" / filename
    _write_json(
        path,
        {
            "schema_version": "aragora-operator-steering-read-receipt/1.0",
            "owner_session": inbox.name,
            "read_by_session": inbox.name,
            "read_at_utc": "2026-07-06T14:01:00.000Z",
            "message_filename": message_filename,
            "message_sha256": message_sha256,
            "outcome": outcome,
        },
    )
    return path


def _run_json(args: list[str], capsys: Any) -> tuple[int, dict[str, Any]]:
    rc = ack.main([*args, "--json"])
    payload = json.loads(capsys.readouterr().out)
    return rc, payload


def test_apply_moves_message_and_writes_ack_sidecar(tmp_path: Path, capsys: Any) -> None:
    root = tmp_path / "operator-steering"
    inbox = root / "owner-session"
    message = _write_message(inbox)
    receipt = _write_receipt(inbox, message_sha256=message["message_sha256"])

    rc, payload = _run_json(
        ["--to", inbox.name, "--steering-inbox-root", str(root), "--apply"],
        capsys,
    )

    assert rc == 0
    assert payload["ok"] is True
    assert payload["applied"] is True
    assert payload["acked_count"] == 1
    assert payload["refused_count"] == 0
    assert not (inbox / "message.json").exists()
    assert (inbox / "_acked" / "message.json").exists()
    sidecar = inbox / "_acked" / "message.json.ack.json"
    assert sidecar.exists()
    ack_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert ack_payload["schema_version"] == "aragora-operator-steering-ack/1.0"
    assert ack_payload["message_sha256"] == message["message_sha256"]
    assert ack_payload["receipt_filename"] == receipt.name


def test_dry_run_mutates_nothing(tmp_path: Path, capsys: Any) -> None:
    root = tmp_path / "operator-steering"
    inbox = root / "owner-session"
    message = _write_message(inbox)
    _write_receipt(inbox, message_sha256=message["message_sha256"])

    rc, payload = _run_json(["--to", inbox.name, "--steering-inbox-root", str(root)], capsys)

    assert rc == 0
    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert payload["messages"][0]["status"] == "ready"
    assert (inbox / "message.json").exists()
    assert not (inbox / "_acked").exists()


def test_missing_receipt_refuses_without_mutation(tmp_path: Path, capsys: Any) -> None:
    root = tmp_path / "operator-steering"
    inbox = root / "owner-session"
    _write_message(inbox)

    rc, payload = _run_json(
        ["--to", inbox.name, "--steering-inbox-root", str(root), "--apply"],
        capsys,
    )

    assert rc == 1
    assert payload["ok"] is False
    assert payload["refused_count"] == 1
    assert payload["messages"][0]["reason"] == "missing_receipt"
    assert (inbox / "message.json").exists()


def test_sha_mismatch_refuses_without_mutation(tmp_path: Path, capsys: Any) -> None:
    root = tmp_path / "operator-steering"
    inbox = root / "owner-session"
    _write_message(inbox)
    _write_receipt(inbox, message_sha256="wrong")

    rc, payload = _run_json(
        ["--to", inbox.name, "--steering-inbox-root", str(root), "--apply"],
        capsys,
    )

    assert rc == 1
    assert payload["messages"][0]["reason"] == "sha_mismatch"
    assert (inbox / "message.json").exists()


def test_non_terminal_outcome_refuses_without_mutation(tmp_path: Path, capsys: Any) -> None:
    root = tmp_path / "operator-steering"
    inbox = root / "owner-session"
    message = _write_message(inbox)
    _write_receipt(inbox, message_sha256=message["message_sha256"], outcome="held")

    rc, payload = _run_json(
        ["--to", inbox.name, "--steering-inbox-root", str(root), "--apply"],
        capsys,
    )

    assert rc == 1
    assert payload["messages"][0]["reason"] == "non_terminal_outcome"
    assert (inbox / "message.json").exists()


def test_apply_is_idempotent_for_already_acked_message(tmp_path: Path, capsys: Any) -> None:
    root = tmp_path / "operator-steering"
    inbox = root / "owner-session"
    message = _write_message(inbox)
    _write_receipt(inbox, message_sha256=message["message_sha256"])

    rc, payload = _run_json(
        ["--to", inbox.name, "--steering-inbox-root", str(root), "--apply"],
        capsys,
    )
    assert rc == 0
    assert payload["acked_count"] == 1

    rc, payload = _run_json(
        [
            "--to",
            inbox.name,
            "--steering-inbox-root",
            str(root),
            "--message",
            "message.json",
            "--apply",
        ],
        capsys,
    )

    assert rc == 0
    assert payload["ok"] is True
    assert payload["acked_count"] == 0
    assert payload["already_acked_count"] == 1
    assert payload["messages"][0]["status"] == "already_acked"


def test_resolves_mailbox_by_pr_from_registry(tmp_path: Path, capsys: Any) -> None:
    root = tmp_path / "operator-steering"
    registry = tmp_path / "lanes.json"
    inbox = root / "owner-session"
    message = _write_message(inbox)
    _write_receipt(inbox, message_sha256=message["message_sha256"])
    _write_json(
        registry,
        [
            {
                "lane_id": "lane-1",
                "owner_session": inbox.name,
                "pr_number": 8610,
                "branch": "codex/example",
            }
        ],
    )

    rc, payload = _run_json(
        [
            "--pr",
            "8610",
            "--steering-inbox-root",
            str(root),
            "--registry-path",
            str(registry),
        ],
        capsys,
    )

    assert rc == 0
    assert payload["resolved_via"] == "pr"
    assert payload["pr_number"] == 8610
    assert payload["message_count"] == 1
    assert payload["messages"][0]["status"] == "ready"
