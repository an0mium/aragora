"""
End-to-end smoke tests for the golden path: review -> receipt -> SARIF export.

Validates that the core review pipeline works without making real API calls:
1. Demo review produces output and exit code 0
2. Demo review with --sarif flag produces valid SARIF 2.1.0 JSON
3. Verdict enum has all expected values
4. DecisionReceipt round-trips through to_dict and to_markdown
5. MCP tools include verify_plan and get_receipt
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# 1. test_review_demo_produces_receipt
# ---------------------------------------------------------------------------


def test_review_demo_produces_receipt(capsys: pytest.CaptureFixture[str]) -> None:
    """Run ``aragora review --demo`` programmatically and verify exit code 0."""
    from aragora.cli.review import cmd_review

    args = argparse.Namespace(
        demo=True,
        output_format="json",
        output_dir=None,
        sarif=None,
        gauntlet=False,
        ci=False,
        share=False,
    )

    exit_code = cmd_review(args)

    assert exit_code == 0, f"cmd_review --demo returned {exit_code}, expected 0"

    captured = capsys.readouterr()
    assert len(captured.out) > 0, "Demo mode produced no stdout output"

    # The JSON output should be parseable
    output = json.loads(captured.out)
    assert "unanimous_critiques" in output
    assert "agreement_score" in output


# ---------------------------------------------------------------------------
# 2. test_review_demo_sarif_output
# ---------------------------------------------------------------------------


def test_review_demo_sarif_output(capsys: pytest.CaptureFixture[str]) -> None:
    """Run review with --demo --sarif and verify valid SARIF 2.1.0 JSON."""
    from aragora.cli.review import cmd_review

    with tempfile.TemporaryDirectory() as tmpdir:
        sarif_path = str(Path(tmpdir) / "review-results.sarif")

        args = argparse.Namespace(
            demo=True,
            output_format="json",
            output_dir=None,
            sarif=sarif_path,
            gauntlet=False,
            ci=False,
            share=False,
        )

        exit_code = cmd_review(args)
        assert exit_code == 0

        # Read and validate SARIF output
        sarif_file = Path(sarif_path)
        assert sarif_file.exists(), f"SARIF file not created at {sarif_path}"

        sarif_data = json.loads(sarif_file.read_text())

        # Verify SARIF 2.1.0 structure
        assert sarif_data["version"] == "2.1.0"
        assert "$schema" in sarif_data
        assert "sarif-schema-2.1.0" in sarif_data["$schema"]

        # Verify runs array
        assert "runs" in sarif_data
        assert isinstance(sarif_data["runs"], list)
        assert len(sarif_data["runs"]) >= 1

        run = sarif_data["runs"][0]

        # Verify tool driver
        assert "tool" in run
        assert "driver" in run["tool"]
        assert "name" in run["tool"]["driver"]

        # Verify results array exists
        assert "results" in run
        assert isinstance(run["results"], list)
        # Demo mode produces findings, so results should be non-empty
        assert len(run["results"]) > 0


# ---------------------------------------------------------------------------
# 3. test_verdict_enum_consistency
# ---------------------------------------------------------------------------


def test_verdict_enum_consistency() -> None:
    """Verify Verdict enum has all 4 expected values and they are strings."""
    from aragora.core_types import Verdict

    expected_values = {"approved", "approved_with_conditions", "needs_review", "rejected"}
    actual_values = {v.value for v in Verdict}

    assert actual_values == expected_values, (
        f"Verdict enum values mismatch: expected {expected_values}, got {actual_values}"
    )

    # All values should be strings (Verdict inherits from str)
    for member in Verdict:
        assert isinstance(member.value, str), f"Verdict.{member.name}.value is not a str"
        # Since Verdict(str, Enum), the member itself should be a str
        assert isinstance(member, str), f"Verdict.{member.name} does not inherit from str"

    # Verify default verdict on DecisionReceipt matches Verdict.NEEDS_REVIEW.value
    from aragora.export.decision_receipt import DecisionReceipt

    receipt = DecisionReceipt(receipt_id="test-id", gauntlet_id="test-gauntlet")
    assert receipt.verdict == Verdict.NEEDS_REVIEW.value.upper(), (
        f"Default verdict is '{receipt.verdict}', expected '{Verdict.NEEDS_REVIEW.value.upper()}'"
    )


# ---------------------------------------------------------------------------
# 4. test_receipt_roundtrip
# ---------------------------------------------------------------------------


def test_receipt_roundtrip() -> None:
    """Create a DecisionReceipt, call to_dict(), and verify fields; also test to_markdown()."""
    from aragora.export.decision_receipt import DecisionReceipt, ReceiptFinding

    receipt = DecisionReceipt(
        receipt_id="smoke-test-001",
        gauntlet_id="gauntlet-smoke-001",
        input_summary="Smoke test input",
        verdict="approved",
        confidence=0.95,
        risk_level="LOW",
        findings=[
            ReceiptFinding(
                id="f1",
                severity="MEDIUM",
                category="test",
                title="Test finding",
                description="A test finding for the smoke test",
            )
        ],
        agents_involved=["agent-a", "agent-b"],
        rounds_completed=3,
        duration_seconds=1.5,
    )

    # Verify to_dict() round-trip
    d = receipt.to_dict()
    assert "schema_version" in d, "to_dict() missing 'schema_version'"
    assert "verdict" in d, "to_dict() missing 'verdict'"
    assert "receipt_id" in d, "to_dict() missing 'receipt_id'"
    assert d["receipt_id"] == "smoke-test-001"
    assert d["verdict"] == "approved"
    assert d["schema_version"] == "1.0"
    assert len(d["findings"]) == 1
    assert d["findings"][0]["title"] == "Test finding"

    # Verify to_markdown() produces non-empty output
    md = receipt.to_markdown()
    assert isinstance(md, str)
    assert len(md) > 0, "to_markdown() returned empty string"
    assert "Decision Receipt" in md
    assert "smoke-test-001" in md


# ---------------------------------------------------------------------------
# 5. test_mcp_tools_registered
# ---------------------------------------------------------------------------


def test_mcp_tools_registered() -> None:
    """Verify that verify_plan and get_receipt are registered in MCP TOOLS_METADATA."""
    from aragora.mcp.tools import TOOLS_METADATA

    tool_names = [t["name"] for t in TOOLS_METADATA]

    assert "verify_plan" in tool_names, (
        f"'verify_plan' not found in TOOLS_METADATA. Available: {tool_names}"
    )
    assert "get_receipt" in tool_names, (
        f"'get_receipt' not found in TOOLS_METADATA. Available: {tool_names}"
    )
