from __future__ import annotations

import argparse
import json

from aragora.cli.commands.inbox_wedge import (
    cmd_inbox_wedge_create,
    cmd_inbox_wedge_list,
    cmd_inbox_wedge_export,
    cmd_inbox_wedge_report,
    cmd_inbox_wedge_review,
    cmd_inbox_wedge_show,
)
from aragora.cli.parser import build_parser
from aragora.gauntlet.signing import HMACSigner, ReceiptSigner
from aragora.inbox.trust_wedge import (
    ActionIntent,
    InboxTrustWedgeService,
    InboxTrustWedgeStore,
    TriageDecision,
)
from aragora.services.email_actions import EmailActionsService


def test_parser_registers_inbox_wedge_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "inbox-wedge",
            "create",
            "--user-id",
            "user-1",
            "--message-id",
            "msg-1",
            "--action",
            "archive",
            "--confidence",
            "0.91",
            "--rationale",
            "archive low-signal message",
            "--content-text",
            "subject: test\nbody: hello",
        ]
    )

    assert args.command == "inbox-wedge"
    assert args.inbox_wedge_command == "create"
    assert callable(args.func)


def test_create_command_emits_receipt_json(monkeypatch, tmp_path, capsys):
    store = InboxTrustWedgeStore(db_path=str(tmp_path / "wedge.db"))
    service = InboxTrustWedgeService(
        store=store,
        signer=ReceiptSigner(HMACSigner(secret_key=b"\x02" * 32, key_id="cli-test-key")),
        email_actions_service=EmailActionsService(),
    )
    monkeypatch.setattr(
        "aragora.cli.commands.inbox_wedge.get_inbox_trust_wedge_service",
        lambda: service,
    )

    args = argparse.Namespace(
        provider="gmail",
        user_id="user-1",
        message_id="msg-1",
        action="archive",
        confidence=0.91,
        rationale="archive low-signal message",
        provider_route="openrouter-fallback",
        debate_id="debate-1",
        dissent_summary="",
        label_id=None,
        content_hash=None,
        content_text="subject: test\nbody: hello",
        content_file=None,
        expires_in_hours=24.0,
        auto_approve=False,
    )

    try:
        cmd_inbox_wedge_create(args)
        payload = json.loads(capsys.readouterr().out)
    finally:
        store.close()

    assert payload["receipt"]["state"] == "created"
    assert payload["intent"]["provider"] == "gmail"
    assert payload["decision"]["final_action"] == "archive"


def _create_receipt(
    service: InboxTrustWedgeService,
    *,
    message_id: str,
    action: str,
    confidence: float,
    provider_route: str,
    auto_approve: bool = False,
):
    intent = ActionIntent.create(
        provider="gmail",
        user_id="user-1",
        message_id=message_id,
        action=action,
        content_hash=ActionIntent.compute_content_hash(message_id, action),
        synthesized_rationale=f"{action} rationale",
        confidence=confidence,
        provider_route=provider_route,
        debate_id=f"debate-{message_id}",
    )
    decision = TriageDecision.create(
        final_action=action,
        confidence=confidence,
        dissent_summary="",
        latency_seconds=1.5,
        cost_usd=0.02,
    )
    return service.create_receipt(intent, decision, auto_approve=auto_approve)


def test_parser_registers_report_and_export_commands():
    parser = build_parser()

    report_args = parser.parse_args(["inbox-wedge", "report"])
    export_args = parser.parse_args(["inbox-wedge", "export", "/tmp/receipts.jsonl"])
    review_args = parser.parse_args(
        ["inbox-wedge", "review", "receipt-1", "--choice", "skip", "--json"]
    )

    assert report_args.command == "inbox-wedge"
    assert report_args.inbox_wedge_command == "report"
    assert callable(report_args.func)
    assert export_args.inbox_wedge_command == "export"
    assert callable(export_args.func)
    assert report_args.json is False
    assert review_args.json is True


def test_report_command_summarizes_receipts(monkeypatch, tmp_path, capsys):
    store = InboxTrustWedgeStore(db_path=str(tmp_path / "report.db"))
    service = InboxTrustWedgeService(
        store=store,
        signer=ReceiptSigner(HMACSigner(secret_key=b"\x03" * 32, key_id="report-test-key")),
        email_actions_service=EmailActionsService(),
    )
    _create_receipt(
        service,
        message_id="msg-created",
        action="archive",
        confidence=0.9,
        provider_route="openrouter",
    )
    _create_receipt(
        service,
        message_id="msg-approved",
        action="star",
        confidence=0.95,
        provider_route="direct",
        auto_approve=True,
    )
    monkeypatch.setattr(
        "aragora.cli.commands.inbox_wedge.get_inbox_trust_wedge_store",
        lambda: store,
    )

    args = argparse.Namespace(state=None, limit=50, json=True)
    try:
        cmd_inbox_wedge_report(args)
        payload = json.loads(capsys.readouterr().out)
    finally:
        store.close()

    assert payload["total_receipts"] == 2
    assert payload["states"]["created"] == 1
    assert payload["states"]["approved"] == 1
    assert payload["actions"]["archive"] == 1
    assert payload["actions"]["star"] == 1
    assert payload["provider_routes"]["openrouter"] == 1
    assert payload["provider_routes"]["direct"] == 1
    assert payload["auto_approved_count"] == 1
    assert payload["average_confidence"] == 0.925


def test_list_command_renders_terminal_summary_by_default(monkeypatch, tmp_path, capsys):
    store = InboxTrustWedgeStore(db_path=str(tmp_path / "list.db"))
    service = InboxTrustWedgeService(
        store=store,
        signer=ReceiptSigner(HMACSigner(secret_key=b"\x05" * 32, key_id="list-test-key")),
        email_actions_service=EmailActionsService(),
    )
    envelope = _create_receipt(
        service,
        message_id="msg-listed",
        action="archive",
        confidence=0.9,
        provider_route="direct",
    )
    monkeypatch.setattr(
        "aragora.cli.commands.inbox_wedge.get_inbox_trust_wedge_store",
        lambda: store,
    )

    args = argparse.Namespace(state=None, limit=10, json=False)
    try:
        cmd_inbox_wedge_list(args)
        output = capsys.readouterr().out
    finally:
        store.close()

    assert "STATE" in output
    assert envelope.receipt.receipt_id[:12] in output
    assert "archive" in output
    assert "direct" in output
    assert "1 receipt(s) shown." in output


def test_show_command_renders_terminal_summary_by_default(monkeypatch, tmp_path, capsys):
    store = InboxTrustWedgeStore(db_path=str(tmp_path / "show.db"))
    service = InboxTrustWedgeService(
        store=store,
        signer=ReceiptSigner(HMACSigner(secret_key=b"\x06" * 32, key_id="show-test-key")),
        email_actions_service=EmailActionsService(),
    )
    envelope = _create_receipt(
        service,
        message_id="msg-show",
        action="ignore",
        confidence=0.4,
        provider_route="direct",
    )
    monkeypatch.setattr(
        "aragora.cli.commands.inbox_wedge.get_inbox_trust_wedge_store",
        lambda: store,
    )

    args = argparse.Namespace(receipt_id=envelope.receipt.receipt_id, json=False)
    try:
        cmd_inbox_wedge_show(args)
        output = capsys.readouterr().out
    finally:
        store.close()

    assert "Inbox Trust Wedge Receipt" in output
    assert envelope.receipt.receipt_id in output
    assert "Action:" in output
    assert "ignore" in output
    assert "Message ID:" in output
    assert "msg-show" in output


def test_report_command_renders_terminal_summary_by_default(monkeypatch, tmp_path, capsys):
    store = InboxTrustWedgeStore(db_path=str(tmp_path / "report-human.db"))
    service = InboxTrustWedgeService(
        store=store,
        signer=ReceiptSigner(HMACSigner(secret_key=b"\x07" * 32, key_id="report-human-key")),
        email_actions_service=EmailActionsService(),
    )
    _create_receipt(
        service,
        message_id="msg-report-human",
        action="archive",
        confidence=0.9,
        provider_route="direct",
    )
    monkeypatch.setattr(
        "aragora.cli.commands.inbox_wedge.get_inbox_trust_wedge_store",
        lambda: store,
    )

    args = argparse.Namespace(state=None, limit=10, json=False)
    try:
        cmd_inbox_wedge_report(args)
        output = capsys.readouterr().out
    finally:
        store.close()

    assert "Inbox Trust Wedge Report" in output
    assert "Total Receipts:" in output
    assert "Actions:" in output
    assert "archive" in output


def test_review_command_renders_terminal_summary_by_default(monkeypatch, tmp_path, capsys):
    store = InboxTrustWedgeStore(db_path=str(tmp_path / "review.db"))
    service = InboxTrustWedgeService(
        store=store,
        signer=ReceiptSigner(HMACSigner(secret_key=b"\x08" * 32, key_id="review-test-key")),
        email_actions_service=EmailActionsService(),
    )
    envelope = _create_receipt(
        service,
        message_id="msg-review",
        action="archive",
        confidence=0.9,
        provider_route="direct",
    )
    monkeypatch.setattr(
        "aragora.cli.commands.inbox_wedge.get_inbox_trust_wedge_service",
        lambda: service,
    )

    args = argparse.Namespace(
        receipt_id=envelope.receipt.receipt_id,
        choice="skip",
        action=None,
        rationale=None,
        label_id=None,
        json=False,
    )
    try:
        cmd_inbox_wedge_review(args)
        output = capsys.readouterr().out
    finally:
        store.close()

    assert "Inbox Trust Wedge Receipt" in output
    assert "Selected:" in output
    assert "skip" in output
    assert "Effect:" in output
    assert "no state change" in output


def test_review_command_can_emit_json(monkeypatch, tmp_path, capsys):
    store = InboxTrustWedgeStore(db_path=str(tmp_path / "review-json.db"))
    service = InboxTrustWedgeService(
        store=store,
        signer=ReceiptSigner(HMACSigner(secret_key=b"\x09" * 32, key_id="review-json-test-key")),
        email_actions_service=EmailActionsService(),
    )
    envelope = _create_receipt(
        service,
        message_id="msg-review-json",
        action="archive",
        confidence=0.9,
        provider_route="direct",
    )
    monkeypatch.setattr(
        "aragora.cli.commands.inbox_wedge.get_inbox_trust_wedge_service",
        lambda: service,
    )

    args = argparse.Namespace(
        receipt_id=envelope.receipt.receipt_id,
        choice="skip",
        action=None,
        rationale=None,
        label_id=None,
        json=True,
    )
    try:
        cmd_inbox_wedge_review(args)
        payload = json.loads(capsys.readouterr().out)
    finally:
        store.close()

    assert payload["receipt"]["receipt_id"] == envelope.receipt.receipt_id
    assert payload["review_choice"] is None


def test_export_command_writes_jsonl(monkeypatch, tmp_path, capsys):
    store = InboxTrustWedgeStore(db_path=str(tmp_path / "export.db"))
    service = InboxTrustWedgeService(
        store=store,
        signer=ReceiptSigner(HMACSigner(secret_key=b"\x04" * 32, key_id="export-test-key")),
        email_actions_service=EmailActionsService(),
    )
    _create_receipt(
        service,
        message_id="msg-1",
        action="archive",
        confidence=0.88,
        provider_route="openrouter",
    )
    _create_receipt(
        service,
        message_id="msg-2",
        action="ignore",
        confidence=0.87,
        provider_route="openrouter",
    )
    monkeypatch.setattr(
        "aragora.cli.commands.inbox_wedge.get_inbox_trust_wedge_store",
        lambda: store,
    )
    output_path = tmp_path / "receipts.jsonl"

    args = argparse.Namespace(
        output_path=str(output_path),
        state=None,
        limit=50,
        format="jsonl",
    )
    try:
        cmd_inbox_wedge_export(args)
        payload = json.loads(capsys.readouterr().out)
    finally:
        store.close()

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert payload["count"] == 2
    assert payload["format"] == "jsonl"
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert "receipt" in first
    assert "intent" in first
