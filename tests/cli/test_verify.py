"""Tests for ``aragora verify`` CLI command.

Validates that the verify command correctly:
- Detects valid receipts and returns exit code 0
- Detects tampered receipts and returns exit code 1
- Handles missing files gracefully
- Produces valid JSON output with --format json
- Handles receipts missing schema_version gracefully
"""

from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path
from typing import Any

import pytest

from aragora.cli.commands.verify import (
    _is_valid_iso_timestamp,
    _is_valid_verdict,
    _recompute_checksum,
    _verify_receipt,
    cmd_verify,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_receipt_data(
    *,
    receipt_id: str = "rcpt_test123",
    verdict: str = "approved",
    confidence: float = 0.85,
    schema_version: str = "1.0",
    timestamp: str = "2026-02-11T10:00:00+00:00",
    findings: list[dict[str, Any]] | None = None,
    critical_count: int = 0,
    audit_trail_id: str | None = None,
    include_checksum: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal valid receipt dict with a correct checksum."""
    data: dict[str, Any] = {
        "receipt_id": receipt_id,
        "gauntlet_id": "gauntlet_test456",
        "timestamp": timestamp,
        "input_summary": "Test receipt",
        "input_type": "spec",
        "schema_version": schema_version,
        "verdict": verdict,
        "confidence": confidence,
        "risk_level": "LOW",
        "risk_score": 0.15,
        "robustness_score": 0.85,
        "coverage_score": 0.9,
        "verification_coverage": 0.0,
        "findings": findings or [],
        "critical_count": critical_count,
        "high_count": 0,
        "medium_count": 0,
        "low_count": 0,
        "mitigations": [],
        "dissenting_views": [],
        "unresolved_tensions": [],
        "verified_claims": [],
        "unverified_claims": [],
        "agents_involved": ["agent-a", "agent-b"],
        "rounds_completed": 3,
        "duration_seconds": 12.5,
        "audit_trail_id": audit_trail_id,
        "cost_usd": 0.0,
        "tokens_used": 0,
        "budget_limit_usd": None,
    }
    if extra:
        data.update(extra)
    if include_checksum:
        data["checksum"] = _recompute_checksum(data)
    return data


def _write_receipt(tmp_path: Path, data: dict[str, Any], filename: str = "receipt.json") -> Path:
    """Write receipt data to a temp JSON file and return the path."""
    path = tmp_path / filename
    path.write_text(json.dumps(data, indent=2))
    return path


class _FakeArgs:
    """Minimal argparse.Namespace stand-in for cmd_verify."""

    def __init__(self, receipt_path: str, output_format: str = "text", verbose: bool = False):
        self.receipt_path = receipt_path
        self.output_format = output_format
        self.verbose = verbose


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for the internal helper functions."""

    def test_is_valid_verdict_canonical(self):
        assert _is_valid_verdict("approved")
        assert _is_valid_verdict("approved_with_conditions")
        assert _is_valid_verdict("needs_review")
        assert _is_valid_verdict("rejected")

    def test_is_valid_verdict_case_insensitive(self):
        assert _is_valid_verdict("APPROVED")
        assert _is_valid_verdict("Rejected")

    def test_is_valid_verdict_invalid(self):
        assert not _is_valid_verdict("maybe")
        assert not _is_valid_verdict("")
        assert not _is_valid_verdict("unknown_verdict")

    def test_is_valid_iso_timestamp_valid(self):
        assert _is_valid_iso_timestamp("2026-02-11T10:00:00+00:00")
        assert _is_valid_iso_timestamp("2026-02-11T10:00:00")
        assert _is_valid_iso_timestamp("2026-02-11")

    def test_is_valid_iso_timestamp_invalid(self):
        assert not _is_valid_iso_timestamp("not-a-date")
        assert not _is_valid_iso_timestamp("")
        assert not _is_valid_iso_timestamp("2026/02/11")

    def test_recompute_checksum_deterministic(self):
        data = _make_receipt_data()
        c1 = _recompute_checksum(data)
        c2 = _recompute_checksum(data)
        assert c1 == c2
        assert len(c1) == 16  # SHA-256 truncated to 16 hex chars


# ---------------------------------------------------------------------------
# Integration tests for _verify_receipt
# ---------------------------------------------------------------------------


class TestVerifyReceipt:
    """Tests for the _verify_receipt function."""

    def test_valid_receipt(self):
        data = _make_receipt_data()
        result = _verify_receipt(data)
        assert result["valid"] is True
        assert all(c["passed"] for c in result["checks"])

    def test_tampered_verdict(self):
        """Changing the verdict after checksum computation should fail."""
        data = _make_receipt_data(verdict="approved")
        # Tamper: change verdict without recomputing checksum
        data["verdict"] = "rejected"
        result = _verify_receipt(data)
        assert result["valid"] is False
        checksum_check = next(c for c in result["checks"] if c["name"] == "checksum")
        assert checksum_check["passed"] is False

    def test_tampered_confidence(self):
        """Changing confidence after checksum computation should fail."""
        data = _make_receipt_data(confidence=0.95)
        data["confidence"] = 0.1
        result = _verify_receipt(data)
        assert result["valid"] is False

    def test_missing_schema_version(self):
        data = _make_receipt_data()
        del data["schema_version"]
        result = _verify_receipt(data)
        assert result["valid"] is False
        sv_check = next(c for c in result["checks"] if c["name"] == "schema_version")
        assert sv_check["passed"] is False

    def test_invalid_verdict_value(self):
        data = _make_receipt_data(verdict="banana")
        result = _verify_receipt(data)
        assert result["valid"] is False
        verdict_check = next(c for c in result["checks"] if c["name"] == "verdict")
        assert verdict_check["passed"] is False

    def test_missing_checksum(self):
        data = _make_receipt_data(include_checksum=False)
        result = _verify_receipt(data)
        assert result["valid"] is False
        checksum_check = next(c for c in result["checks"] if c["name"] == "checksum")
        assert checksum_check["passed"] is False

    def test_invalid_timestamp(self):
        data = _make_receipt_data(timestamp="not-a-date")
        result = _verify_receipt(data)
        assert result["valid"] is False
        ts_check = next(c for c in result["checks"] if c["name"] == "timestamp")
        assert ts_check["passed"] is False

    def test_verbose_shows_recomputed(self):
        data = _make_receipt_data()
        result = _verify_receipt(data, verbose=True)
        checksum_check = next(c for c in result["checks"] if c["name"] == "checksum")
        assert "recomputed=" in checksum_check["detail"]


# ---------------------------------------------------------------------------
# CLI cmd_verify tests
# ---------------------------------------------------------------------------


class TestCmdVerify:
    """End-to-end tests for cmd_verify through argparse namespace."""

    def test_verify_valid_receipt(self, tmp_path: Path):
        """A valid receipt should return exit code 0."""
        data = _make_receipt_data()
        path = _write_receipt(tmp_path, data)
        args = _FakeArgs(receipt_path=str(path))
        rc = cmd_verify(args)
        assert rc == 0

    def test_verify_invalid_receipt(self, tmp_path: Path):
        """A tampered receipt should return exit code 1."""
        data = _make_receipt_data()
        data["verdict"] = "rejected"  # tamper without recomputing checksum
        path = _write_receipt(tmp_path, data)
        args = _FakeArgs(receipt_path=str(path))
        rc = cmd_verify(args)
        assert rc == 1

    def test_verify_missing_file(self, tmp_path: Path, capsys):
        """A missing file should return exit code 1 with error message."""
        args = _FakeArgs(receipt_path=str(tmp_path / "nonexistent.json"))
        rc = cmd_verify(args)
        assert rc == 1
        captured = capsys.readouterr()
        assert "File not found" in captured.err or "not found" in captured.err.lower()

    def test_verify_json_output(self, tmp_path: Path, capsys):
        """--format json should produce valid JSON output."""
        data = _make_receipt_data()
        path = _write_receipt(tmp_path, data)
        args = _FakeArgs(receipt_path=str(path), output_format="json")
        rc = cmd_verify(args)
        assert rc == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["valid"] is True
        assert isinstance(output["checks"], list)
        assert output["receipt_id"] == "rcpt_test123"

    def test_verify_json_output_invalid(self, tmp_path: Path, capsys):
        """--format json with invalid receipt should produce valid JSON with valid=false."""
        data = _make_receipt_data()
        data["verdict"] = "rejected"  # tamper
        path = _write_receipt(tmp_path, data)
        args = _FakeArgs(receipt_path=str(path), output_format="json")
        rc = cmd_verify(args)
        assert rc == 1
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["valid"] is False

    def test_verify_invalid_schema(self, tmp_path: Path, capsys):
        """Receipt missing schema_version should be handled gracefully."""
        data = _make_receipt_data()
        del data["schema_version"]
        path = _write_receipt(tmp_path, data)
        args = _FakeArgs(receipt_path=str(path), output_format="json")
        rc = cmd_verify(args)
        assert rc == 1
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["valid"] is False
        sv_check = next(c for c in output["checks"] if c["name"] == "schema_version")
        assert sv_check["passed"] is False

    def test_verify_missing_file_json_output(self, tmp_path: Path, capsys):
        """Missing file with --format json should produce valid JSON error."""
        args = _FakeArgs(
            receipt_path=str(tmp_path / "gone.json"),
            output_format="json",
        )
        rc = cmd_verify(args)
        assert rc == 1
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["valid"] is False
        assert "error" in output

    def test_verify_malformed_json(self, tmp_path: Path, capsys):
        """A file with invalid JSON should return exit code 1."""
        path = tmp_path / "bad.json"
        path.write_text("{ not valid json !!!")
        args = _FakeArgs(receipt_path=str(path))
        rc = cmd_verify(args)
        assert rc == 1

    def test_verify_verbose(self, tmp_path: Path, capsys):
        """--verbose should show additional details in text output."""
        data = _make_receipt_data()
        path = _write_receipt(tmp_path, data)
        args = _FakeArgs(receipt_path=str(path), verbose=True)
        rc = cmd_verify(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "PASS" in captured.out
        assert "VALID" in captured.out

    def test_verify_non_dict_json(self, tmp_path: Path, capsys):
        """A JSON file containing a list (not dict) should fail gracefully."""
        path = tmp_path / "list.json"
        path.write_text("[1, 2, 3]")
        args = _FakeArgs(receipt_path=str(path))
        rc = cmd_verify(args)
        assert rc == 1

    def test_verify_receipt_with_findings(self, tmp_path: Path):
        """A receipt with findings should still verify if checksum is valid."""
        findings = [
            {
                "id": "f1",
                "severity": "MEDIUM",
                "category": "test",
                "title": "Test finding",
                "description": "A test finding",
                "mitigation": None,
                "source": "agent-a",
                "verified": False,
            }
        ]
        data = _make_receipt_data(findings=findings)
        path = _write_receipt(tmp_path, data)
        args = _FakeArgs(receipt_path=str(path))
        rc = cmd_verify(args)
        assert rc == 0
