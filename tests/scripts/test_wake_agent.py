"""Tests for ``scripts/wake_agent.py`` safe steering transport selection."""

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


wake = _load_module("wake_agent.py")


def _parse(args: list[str]) -> Any:
    return wake.build_parser().parse_args(args)


def test_choose_transport_prefers_explicit_tmux() -> None:
    transport, details = wake.choose_transport(
        lane={"contact_method": "tmux:aragora:3"},
        fallback="mailbox-only",
    )

    assert transport == "tmux"
    assert details["target"] == "aragora:3"


def test_choose_transport_uses_codex_thread_id_without_ui_injection() -> None:
    transport, details = wake.choose_transport(
        lane={"codex_thread_id": "019e-thread"},
        fallback="mailbox-only",
    )

    assert transport == "codex-exec-resume"
    assert details["thread_id"] == "019e-thread"


def test_codex_app_server_contact_falls_back_to_mailbox_until_adapter_exists() -> None:
    transport, details = wake.choose_transport(
        lane={
            "contact_method": "codex-app-server:/tmp/codex.sock",
            "contact_payload": {"socket": "/tmp/codex.sock", "thread_id": "t1"},
        },
        fallback="mailbox-only",
    )

    assert transport == "mailbox"
    assert "not yet enabled" in details["fallback_reason"]


def test_dry_run_does_not_write_mailbox_but_writes_receipt(tmp_path: Path) -> None:
    args = _parse(
        [
            "--to",
            "codex-owner",
            "--prompt",
            "check your mailbox",
            "--steering-inbox-root",
            str(tmp_path / "steering"),
            "--receipt-root",
            str(tmp_path / "receipts"),
            "--json",
        ]
    )

    result = wake.run(args)

    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["status"] == "dry-run"
    assert not (tmp_path / "steering").exists()
    receipt_path = Path(result["receipt_path"])
    assert receipt_path.exists()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["prompt_sha256"] == result["prompt_sha256"]


def test_apply_mailbox_writes_message_and_receipt(tmp_path: Path) -> None:
    args = _parse(
        [
            "--to",
            "codex-owner",
            "--prompt",
            "real mailbox delivery",
            "--apply",
            "--priority",
            "blocking",
            "--steering-inbox-root",
            str(tmp_path / "steering"),
            "--receipt-root",
            str(tmp_path / "receipts"),
            "--json",
        ]
    )

    result = wake.run(args)

    assert result["ok"] is True
    assert result["status"] == "mailbox-written"
    message_files = list((tmp_path / "steering" / "codex-owner").glob("*.json"))
    assert len(message_files) == 1
    message = json.loads(message_files[0].read_text(encoding="utf-8"))
    assert message["body"] == "real mailbox delivery"
    assert message["priority"] == "blocking"
    assert Path(result["receipt_path"]).exists()
