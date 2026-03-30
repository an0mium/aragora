"""Tests for receipt list/show CLI convergence on the durable store."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from aragora.cli.commands.receipt import cmd_receipt_list, cmd_receipt_show


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
            cmd_receipt_list(argparse.Namespace(limit=5, verdict=None, org_id=None))

    output = capsys.readouterr().out
    assert "rcpt-quickst.." in output
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
            cmd_receipt_list(argparse.Namespace(limit=5, verdict="fail", org_id=None))

    output = capsys.readouterr().out
    assert "gauntlet-leg.." in output
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
        cmd_receipt_list(argparse.Namespace(limit=5, verdict=None, org_id=None))

    output = capsys.readouterr().out
    assert "BLOCKED" in output
    assert "73%" in output


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
