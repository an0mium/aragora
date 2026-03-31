"""Tests for receipt list/show CLI convergence on the durable store."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aragora.cli.commands.receipt import (
    add_receipt_parser,
    _format_receipt_created_at,
    cmd_receipt_list,
    cmd_receipt_show,
)


@dataclass
class _StoredReceiptStub:
    receipt_id: str
    gauntlet_id: str
    verdict: str
    confidence: float
    created_at: float
    data: dict = field(default_factory=dict)

    def to_full_dict(self) -> dict:
        payload = dict(self.data)
        payload.setdefault("receipt_id", self.receipt_id)
        payload.setdefault("gauntlet_id", self.gauntlet_id)
        payload.setdefault("verdict", self.verdict)
        payload.setdefault("confidence", self.confidence)
        return payload


def _build_receipt_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_receipt_parser(subparsers)
    return parser


def _make_decision_receipt_payload() -> dict:
    from aragora.gauntlet.receipt_models import DecisionReceipt

    receipt = DecisionReceipt(
        receipt_id="rcpt-format-123",
        gauntlet_id="gauntlet-format-123",
        timestamp="2026-03-30T18:47:29+00:00",
        input_summary="Review PR #42",
        input_hash="abc123def456",
        risk_summary={"critical": 1, "high": 0, "medium": 0, "low": 0},
        attacks_attempted=5,
        attacks_successful=0,
        probes_run=7,
        vulnerabilities_found=1,
        verdict="CONDITIONAL",
        confidence=0.86,
        robustness_score=0.74,
        verdict_reasoning="Escalate for manual review.",
        vulnerability_details=[
            {
                "id": "F-001",
                "title": "SQL Injection",
                "severity": "critical",
                "category": "security",
                "description": "User input reaches query builder unsanitized.",
                "mitigation": "Use parameterized queries.",
            }
        ],
    )
    return receipt.to_dict()


def test_receipt_list_reads_durable_store_by_default(capsys: pytest.CaptureFixture[str]) -> None:
    stored = _StoredReceiptStub(
        receipt_id="rcpt-quickstart-123",
        gauntlet_id="rcpt-quickstart-123",
        verdict="PASS",
        confidence=1.0,
        created_at=1711300000.0,
        data={"risk_summary": {"total": 2}},
    )
    durable_store = MagicMock()
    durable_store.list.return_value = [stored]

    with patch("aragora.cli.commands.receipt._load_storage_receipt_list", return_value=[stored]):
        with patch("aragora.cli.commands.receipt._load_legacy_receipt_list") as legacy_loader:
            cmd_receipt_list(argparse.Namespace(limit=5, verdict=None, kind=None, org_id=None))

    output = capsys.readouterr().out
    assert "rcpt-quickst.." in output
    assert "decision" in output
    assert "PASS" in output
    assert "2" in output
    legacy_loader.assert_not_called()


def test_receipt_list_falls_back_to_legacy_when_durable_empty(
    capsys: pytest.CaptureFixture[str],
) -> None:
    legacy_row = SimpleNamespace(
        gauntlet_id="gauntlet-legacy-123",
        verdict="FAIL",
        confidence=0.25,
        total_findings=4,
        created_at=datetime(2026, 3, 24, 12, 0, tzinfo=timezone.utc),
    )

    with patch("aragora.cli.commands.receipt._load_storage_receipt_list", return_value=[]):
        with patch(
            "aragora.cli.commands.receipt._load_legacy_receipt_list",
            return_value=[legacy_row],
        ):
            cmd_receipt_list(argparse.Namespace(limit=5, verdict="fail", kind=None, org_id=None))

    output = capsys.readouterr().out
    assert "gauntlet-leg.." in output
    assert "other" in output
    assert "FAIL" in output
    assert "4" in output


def test_receipt_list_normalizes_trust_wedge_receipts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    stored = _StoredReceiptStub(
        receipt_id="rcpt-triage-123",
        gauntlet_id="rcpt-triage-123",
        verdict="UNKNOWN",
        confidence=0.0,
        created_at=1711300000.0,
        data={
            "state": "CREATED",
            "triage_decision": {
                "confidence": 0.73,
                "blocked_by_policy": True,
            },
        },
    )

    with patch("aragora.cli.commands.receipt._load_storage_receipt_list", return_value=[stored]):
        cmd_receipt_list(argparse.Namespace(limit=5, verdict=None, kind=None, org_id=None))

    output = capsys.readouterr().out
    assert "inbox" in output
    assert "BLOCKED" in output
    assert "73%" in output


def test_receipt_list_filters_by_kind(capsys: pytest.CaptureFixture[str]) -> None:
    inbox = _StoredReceiptStub(
        receipt_id="rcpt-inbox-123",
        gauntlet_id="rcpt-inbox-123",
        verdict="CONDITIONAL",
        confidence=0.95,
        created_at=1711300000.0,
        data={"action_intent": {}, "triage_decision": {}},
    )
    decision = _StoredReceiptStub(
        receipt_id="rcpt-decision-456",
        gauntlet_id="rcpt-decision-456",
        verdict="PASS",
        confidence=0.85,
        created_at=1711300001.0,
        data={"consensus_proof": {}, "agent_responses": []},
    )

    with patch(
        "aragora.cli.commands.receipt._load_storage_receipt_list",
        return_value=[inbox, decision],
    ):
        cmd_receipt_list(argparse.Namespace(limit=5, verdict=None, kind="inbox", org_id=None))

    output = capsys.readouterr().out
    assert "rcpt-inbox-123" in output
    assert "inbox" in output
    assert "rcpt-decisio.." not in output


def test_receipt_created_at_formats_epoch_and_iso_consistently() -> None:
    iso_timestamp = "2026-03-30T18:47:29.647269+00:00"
    epoch_timestamp = datetime.fromisoformat(iso_timestamp).timestamp()

    assert _format_receipt_created_at(epoch_timestamp) == _format_receipt_created_at(iso_timestamp)


def test_receipt_show_reads_durable_store_by_receipt_id(
    capsys: pytest.CaptureFixture[str],
) -> None:
    stored = _StoredReceiptStub(
        receipt_id="rcpt-live-123",
        gauntlet_id="rcpt-live-123",
        verdict="PASS",
        confidence=1.0,
        created_at=1711300000.0,
        data={"summary": "Stored in durable receipt store"},
    )

    with patch(
        "aragora.cli.commands.receipt._load_storage_receipt", return_value=stored.to_full_dict()
    ):
        with patch("aragora.cli.commands.receipt._load_legacy_receipt") as legacy_loader:
            cmd_receipt_show(argparse.Namespace(id="rcpt-live-123", format="json", org_id=None))

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["receipt_id"] == "rcpt-live-123"
    assert payload["summary"] == "Stored in durable receipt store"
    legacy_loader.assert_not_called()


def test_receipt_show_normalizes_trust_wedge_receipts_for_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    stored = {
        "receipt_id": "rcpt-triage-456",
        "gauntlet_id": "rcpt-triage-456",
        "verdict": "UNKNOWN",
        "confidence": 0.0,
        "state": "CREATED",
        "triage_decision": {
            "confidence": 0.61,
            "blocked_by_policy": True,
        },
    }

    with patch("aragora.cli.commands.receipt._load_storage_receipt", return_value=stored):
        cmd_receipt_show(argparse.Namespace(id="rcpt-triage-456", format="json", org_id=None))

    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "BLOCKED"
    assert payload["confidence"] == pytest.approx(0.61)


def test_receipt_show_format_json_outputs_expected_receipt_payload(
    capsys: pytest.CaptureFixture[str],
) -> None:
    stored = _make_decision_receipt_payload()

    with patch("aragora.cli.commands.receipt._load_storage_receipt", return_value=stored):
        cmd_receipt_show(argparse.Namespace(id="rcpt-format-123", format="json", org_id=None))

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert output.lstrip().startswith("{")
    assert payload["receipt_id"] == "rcpt-format-123"
    assert payload["gauntlet_id"] == "gauntlet-format-123"
    assert payload["verdict"] == "CONDITIONAL"
    assert payload["vulnerability_details"][0]["title"] == "SQL Injection"


def test_receipt_show_format_markdown_flag_renders_markdown_receipt(
    capsys: pytest.CaptureFixture[str],
) -> None:
    stored = _make_decision_receipt_payload()
    parser = _build_receipt_cli_parser()
    args = parser.parse_args(["receipt", "show", "rcpt-format-123", "--format", "markdown"])

    with patch("aragora.cli.commands.receipt._load_storage_receipt", return_value=stored):
        args.func(args)

    output = capsys.readouterr().out
    assert output.splitlines()[0] == "# Decision Receipt"
    assert "**Receipt ID:** `rcpt-format-123`" in output
    assert "**Gauntlet ID:** `gauntlet-format-123`" in output
    assert "## Verdict: [~] CONDITIONAL" in output
    assert "**Confidence:** 86.0%" in output
    assert "## Critical Findings" in output
    assert "### [F-001] SQL Injection" in output
    assert "**Mitigation:** Use parameterized queries." in output
    assert "## Integrity Verification" in output


def test_receipt_show_renders_inbox_receipt_details(
    capsys: pytest.CaptureFixture[str],
) -> None:
    stored = {
        "receipt_id": "rcpt-inbox-789",
        "gauntlet_id": "rcpt-inbox-789",
        "verdict": "CONDITIONAL",
        "confidence": 0.95,
        "state": "created",
        "action_intent": {
            "provider": "gmail",
            "message_id": "msg-123",
            "action": "archive",
            "provider_route": "direct",
            "synthesized_rationale": "Archive the newsletter.",
        },
        "triage_decision": {
            "final_action": "archive",
            "provider_route": "direct",
            "receipt_state": "created",
            "blocked_by_policy": False,
        },
    }

    with patch("aragora.cli.commands.receipt._load_storage_receipt", return_value=stored):
        cmd_receipt_show(argparse.Namespace(id="rcpt-inbox-789", format=None, org_id=None))

    output = capsys.readouterr().out
    assert "Type:          inbox" in output
    assert "Action:        archive" in output
    assert "Provider:      gmail" in output
    assert "Message ID:    msg-123" in output
    assert "Receipt State: created" in output
    assert "Rationale:     Archive the newsletter." in output


def test_receipt_show_falls_back_to_legacy_when_durable_missing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    legacy_data = {
        "receipt_id": "legacy-rcpt-456",
        "gauntlet_id": "gauntlet-live-456",
        "verdict": "CONDITIONAL",
        "confidence": 0.6,
    }

    with patch("aragora.cli.commands.receipt._load_storage_receipt", return_value=None):
        with patch(
            "aragora.cli.commands.receipt._load_legacy_receipt",
            return_value=legacy_data,
        ):
            cmd_receipt_show(argparse.Namespace(id="gauntlet-live-456", format="json", org_id=None))

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["receipt_id"] == "legacy-rcpt-456"
    assert payload["gauntlet_id"] == "gauntlet-live-456"
