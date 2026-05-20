"""Tests for ``scripts/read_operator_steering.py``.

Fixture-driven; all steering mailbox reads/writes happen under ``tmp_path``.
The reader must leave original messages in place and write append-only read
receipts into a sidecar directory.
"""

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


sos = _load_module("send_operator_steering.py")
ros = _load_module("read_operator_steering.py")


def _write_message(root: Path, recipient: str, body: str = "body") -> Path:
    message = sos.build_message(
        to_session=recipient,
        body=body,
        priority="blocking",
        lane_id_hint="P-read-fixture",
        pr_hint=7373,
        sent_at_utc="2026-05-19T22:00:00.000Z",
    )
    return sos.write_message(message, steering_inbox_root=root)


def _receipt_files(root: Path, recipient: str) -> list[Path]:
    receipt_dir = root / recipient / "_read_receipts"
    return sorted(receipt_dir.glob("*.json")) if receipt_dir.is_dir() else []


def test_reads_only_selected_owner_and_writes_bound_receipt(tmp_path: Path, capsys: Any) -> None:
    steering_root = tmp_path / "steering"
    selected = _write_message(steering_root, "codex-selected", "selected body")
    other = _write_message(steering_root, "codex-other", "other body")

    rc = ros.main(
        [
            "--to",
            "codex-selected",
            "--read-by-session",
            "reader-session",
            "--outcome",
            "obeyed",
            "--outcome-note",
            "validated and followed",
            "--steering-inbox-root",
            str(steering_root),
            "--json",
        ]
    )

    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["owner_session"] == "codex-selected"
    assert out["message_count"] == 1
    assert out["receipt_count"] == 1
    assert out["messages"][0]["filename"] == selected.name
    assert out["messages"][0]["sha256_valid"] is True
    assert selected.exists()
    assert other.exists()

    receipts = _receipt_files(steering_root, "codex-selected")
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    selected_payload = json.loads(selected.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == "aragora-operator-steering-read-receipt/1.0"
    assert receipt["owner_session"] == "codex-selected"
    assert receipt["read_by_session"] == "reader-session"
    assert receipt["message_filename"] == selected.name
    assert receipt["message_sha256"] == selected_payload["message_sha256"]
    assert receipt["message_sent_at_utc"] == "2026-05-19T22:00:00.000Z"
    assert receipt["priority"] == "blocking"
    assert receipt["lane_id_hint"] == "P-read-fixture"
    assert receipt["pr_hint"] == 7373
    assert receipt["subject"] == "selected body"
    assert receipt["outcome"] == "obeyed"
    assert receipt["outcome_note"] == "validated and followed"
    assert _receipt_files(steering_root, "codex-other") == []


def test_no_receipt_lists_messages_without_writing(tmp_path: Path, capsys: Any) -> None:
    steering_root = tmp_path / "steering"
    msg = _write_message(steering_root, "codex-dry", "dry body")

    rc = ros.main(
        [
            "--to",
            "codex-dry",
            "--no-receipt",
            "--steering-inbox-root",
            str(steering_root),
            "--json",
        ]
    )

    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["message_count"] == 1
    assert out["receipt_count"] == 0
    assert out["messages"][0]["filename"] == msg.name
    assert _receipt_files(steering_root, "codex-dry") == []
    assert msg.exists()


def test_rejects_unsafe_session_before_reading(tmp_path: Path, capsys: Any) -> None:
    rc = ros.main(
        [
            "--to",
            "../escape",
            "--steering-inbox-root",
            str(tmp_path / "steering"),
            "--json",
        ]
    )

    assert rc == 2
    assert "path separators" in capsys.readouterr().err
    assert list(tmp_path.rglob("*.json")) == []


def test_resolves_owner_by_lane_id(tmp_path: Path, capsys: Any) -> None:
    steering_root = tmp_path / "steering"
    registry = tmp_path / "lanes.json"
    registry.write_text(
        json.dumps(
            [
                {
                    "lane_id": "P-read-fixture",
                    "owner_session": "codex-lane-owner",
                    "status": "active",
                    "branch": "codex/read-fixture",
                    "pr_number": 7373,
                }
            ]
        ),
        encoding="utf-8",
    )
    _write_message(steering_root, "codex-lane-owner", "lane body")

    rc = ros.main(
        [
            "--lane-id",
            "P-read-fixture",
            "--registry-path",
            str(registry),
            "--steering-inbox-root",
            str(steering_root),
            "--json",
        ]
    )

    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["owner_session"] == "codex-lane-owner"
    assert out["resolved_via"] == "lane-id"
    assert out["message_count"] == 1
