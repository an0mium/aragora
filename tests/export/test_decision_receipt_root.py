"""
Tests for DecisionReceipt - audit-ready compliance artifacts.

Tests cover:
- Receipt creation and serialization
- Checksum/integrity verification
- Export formats (JSON, Markdown, HTML)
- Loading and round-trip serialization
- Edge cases (empty findings, missing fields)
"""

import json
import pytest
from pathlib import Path
from datetime import datetime

from aragora.export.decision_receipt import (
    DecisionReceipt,
    ReceiptFinding,
    ReceiptDissent,
    ReceiptVerification,
    DecisionReceiptGenerator,
)


class TestReceiptFinding:
    """Tests for ReceiptFinding dataclass."""

    def test_creation_basic(self):
        """Test basic finding creation."""
        finding = ReceiptFinding(
            id="finding-001",
            severity="HIGH",
            category="security",
            title="SQL Injection",
            description="User input not sanitized",
        )
        assert finding.id == "finding-001"
        assert finding.severity == "HIGH"
        assert finding.category == "security"
        assert finding.title == "SQL Injection"
        assert finding.verified is False  # default

    def test_creation_with_all_fields(self):
        """Test finding with all optional fields."""
        finding = ReceiptFinding(
            id="finding-002",
            severity="CRITICAL",
            category="auth",
            title="Auth Bypass",
            description="Authentication can be bypassed",
            mitigation="Implement proper auth checks",
            source="red-team-agent",
            verified=True,
        )
        assert finding.mitigation == "Implement proper auth checks"
        assert finding.source == "red-team-agent"
        assert finding.verified is True


class TestReceiptDissent:
    """Tests for ReceiptDissent dataclass."""

    def test_creation(self):
        """Test dissent record creation."""
        dissent = ReceiptDissent(
            agent="critic-agent",
            type="partial_disagree",
            severity=0.7,
            reasons=["Insufficient evidence", "Contradicts prior analysis"],
            alternative="Consider phased rollout instead",
        )
        assert dissent.agent == "critic-agent"
        assert dissent.type == "partial_disagree"
        assert dissent.severity == 0.7
        assert len(dissent.reasons) == 2
        assert dissent.alternative is not None


class TestReceiptVerification:
    """Tests for ReceiptVerification dataclass."""

    def test_creation_verified(self):
        """Test verified claim record."""
        verification = ReceiptVerification(
            claim="The system handles 1000 concurrent requests",
            verified=True,
            method="load_testing",
            proof_hash="abc123def456",
        )
        assert verification.claim.startswith("The system")
        assert verification.verified is True
        assert verification.method == "load_testing"
        assert verification.proof_hash == "abc123def456"

    def test_creation_unverified(self):
        """Test unverified claim record."""
        verification = ReceiptVerification(
            claim="AI responses are always accurate",
            verified=False,
            method="manual_review",
        )
        assert verification.verified is False
        assert verification.proof_hash is None


class TestDecisionReceipt:
    """Tests for DecisionReceipt main class."""

    def test_creation_minimal(self):
        """Test minimal receipt creation."""
        receipt = DecisionReceipt(
            receipt_id="receipt-001",
            gauntlet_id="gauntlet-20260111-abc123",
        )
        assert receipt.receipt_id == "receipt-001"
        assert receipt.gauntlet_id == "gauntlet-20260111-abc123"
        assert receipt.verdict == "NEEDS_REVIEW"  # default
        assert receipt.confidence == 0.0
        assert receipt.checksum != ""  # auto-computed

    def test_creation_full(self):
        """Test full receipt creation with all fields."""
        findings = [
            ReceiptFinding(
                id="f1",
                severity="CRITICAL",
                category="security",
                title="Critical Vulnerability",
                description="A critical security issue",
            ),
            ReceiptFinding(
                id="f2",
                severity="HIGH",
                category="performance",
                title="Performance Issue",
                description="Slow response times",
            ),
        ]

        dissents = [
            ReceiptDissent(
                agent="agent-1",
                type="full_disagree",
                severity=0.9,
                reasons=["Insufficient testing"],
            )
        ]

        verifications = [
            ReceiptVerification(
                claim="System is secure",
                verified=True,
                method="formal_verification",
                proof_hash="hash123",
            )
        ]

        receipt = DecisionReceipt(
            receipt_id="receipt-full",
            gauntlet_id="gauntlet-full",
            input_summary="Test the rate limiter implementation",
            input_type="spec",
            verdict="APPROVED_WITH_CONDITIONS",
            confidence=0.85,
            risk_level="MEDIUM",
            risk_score=0.45,
            robustness_score=0.80,
            coverage_score=0.75,
            verification_coverage=0.60,
            findings=findings,
            critical_count=1,
            high_count=1,
            medium_count=0,
            low_count=0,
            mitigations=["Implement rate limiting", "Add input validation"],
            dissenting_views=dissents,
            unresolved_tensions=["Trade-off between security and usability"],
            verified_claims=verifications,
            unverified_claims=["All edge cases handled"],
            agents_involved=["auditor", "red-team", "verifier"],
            rounds_completed=5,
            duration_seconds=120.5,
        )

        assert receipt.verdict == "APPROVED_WITH_CONDITIONS"
        assert receipt.confidence == 0.85
        assert len(receipt.findings) == 2
        assert receipt.critical_count == 1
        assert len(receipt.agents_involved) == 3

    def test_checksum_auto_computed(self):
        """Test that checksum is automatically computed on creation."""
        receipt = DecisionReceipt(
            receipt_id="test-checksum",
            gauntlet_id="gauntlet-123",
            verdict="APPROVED",
            confidence=0.95,
        )
        assert len(receipt.checksum) == 16  # Truncated SHA-256

    def test_checksum_consistency(self):
        """Test that same inputs produce same checksum."""
        receipt1 = DecisionReceipt(
            receipt_id="test-001",
            gauntlet_id="gauntlet-xyz",
            verdict="APPROVED",
            confidence=0.9,
            timestamp="2026-01-11T12:00:00",
        )
        receipt2 = DecisionReceipt(
            receipt_id="test-001",
            gauntlet_id="gauntlet-xyz",
            verdict="APPROVED",
            confidence=0.9,
            timestamp="2026-01-11T12:00:00",
        )
        assert receipt1.checksum == receipt2.checksum

    def test_verify_integrity_valid(self):
        """Test integrity verification passes for unchanged receipt."""
        receipt = DecisionReceipt(
            receipt_id="integrity-test",
            gauntlet_id="gauntlet-int",
            verdict="REJECTED",
            confidence=0.99,
        )
        assert receipt.verify_integrity() is True

    def test_verify_integrity_tampered(self):
        """Test integrity verification fails for tampered receipt."""
        receipt = DecisionReceipt(
            receipt_id="tamper-test",
            gauntlet_id="gauntlet-tam",
            verdict="APPROVED",
            confidence=0.9,
        )
        original_checksum = receipt.checksum

        # Tamper with the receipt
        receipt.verdict = "REJECTED"

        # Checksum still has old value, so verification fails
        assert receipt.checksum == original_checksum
        assert receipt.verify_integrity() is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        findings = [
            ReceiptFinding(
                id="f1",
                severity="HIGH",
                category="test",
                title="Test Finding",
                description="Description",
            )
        ]
        receipt = DecisionReceipt(
            receipt_id="dict-test",
            gauntlet_id="gauntlet-dict",
            verdict="APPROVED",
            confidence=0.9,
            findings=findings,
        )

        data = receipt.to_dict()
        assert data["receipt_id"] == "dict-test"
        assert data["verdict"] == "APPROVED"
        assert len(data["findings"]) == 1
        assert data["findings"][0]["severity"] == "HIGH"

    def test_to_json(self):
        """Test JSON serialization."""
        receipt = DecisionReceipt(
            receipt_id="json-test",
            gauntlet_id="gauntlet-json",
            verdict="NEEDS_REVIEW",
            confidence=0.5,
        )

        json_str = receipt.to_json()
        data = json.loads(json_str)
        assert data["receipt_id"] == "json-test"
        assert data["verdict"] == "NEEDS_REVIEW"

    def test_to_markdown(self):
        """Test Markdown export."""
        findings = [
            ReceiptFinding(
                id="f1",
                severity="CRITICAL",
                category="security",
                title="SQL Injection",
                description="User input not sanitized in query builder",
                mitigation="Use parameterized queries",
            )
        ]
        receipt = DecisionReceipt(
            receipt_id="md-test",
            gauntlet_id="gauntlet-md",
            verdict="REJECTED",
            confidence=0.95,
            risk_level="CRITICAL",
            findings=findings,
            critical_count=1,
            agents_involved=["auditor", "red-team"],
            duration_seconds=60.0,
        )

        md = receipt.to_markdown()
        assert "# Decision Receipt" in md
        assert "REJECTED" in md
        assert "SQL Injection" in md
        assert "Critical Issues" in md
        assert "parameterized queries" in md
        assert "Checksum" in md

    def test_to_html(self):
        """Test HTML export."""
        receipt = DecisionReceipt(
            receipt_id="html-test",
            gauntlet_id="gauntlet-html",
            verdict="APPROVED",
            confidence=0.85,
            robustness_score=0.90,
            coverage_score=0.80,
            verification_coverage=0.70,
        )

        html = receipt.to_html()
        assert "<!DOCTYPE html>" in html
        assert "APPROVED" in html
        assert "html-test" in html
        assert "Robustness" in html
        assert "#28a745" in html  # green color for APPROVED

    def test_from_json_roundtrip(self):
        """Test JSON roundtrip serialization."""
        original = DecisionReceipt(
            receipt_id="roundtrip-test",
            gauntlet_id="gauntlet-rt",
            input_summary="Test input",
            verdict="APPROVED_WITH_CONDITIONS",
            confidence=0.88,
            findings=[
                ReceiptFinding(
                    id="f1",
                    severity="MEDIUM",
                    category="code",
                    title="Test",
                    description="Desc",
                )
            ],
            dissenting_views=[
                ReceiptDissent(
                    agent="agent-1",
                    type="minor",
                    severity=0.3,
                    reasons=["Reason 1"],
                )
            ],
            verified_claims=[
                ReceiptVerification(
                    claim="Claim 1",
                    verified=True,
                    method="test",
                )
            ],
        )

        json_str = original.to_json()
        loaded = DecisionReceipt.from_json(json_str)

        assert loaded.receipt_id == original.receipt_id
        assert loaded.verdict == original.verdict
        assert loaded.confidence == original.confidence
        assert len(loaded.findings) == len(original.findings)
        assert loaded.findings[0].severity == "MEDIUM"
        assert len(loaded.dissenting_views) == 1
        assert len(loaded.verified_claims) == 1

    def test_save_and_load(self, tmp_path):
        """Test saving and loading receipt files."""
        receipt = DecisionReceipt(
            receipt_id="save-test",
            gauntlet_id="gauntlet-save",
            verdict="APPROVED",
            confidence=0.9,
        )

        # Save as JSON
        json_path = receipt.save(tmp_path / "receipt", format="json")
        assert json_path.suffix == ".json"
        assert json_path.exists()

        # Load and verify
        loaded = DecisionReceipt.load(json_path)
        assert loaded.receipt_id == receipt.receipt_id
        assert loaded.verify_integrity()

    def test_save_markdown(self, tmp_path):
        """Test saving as Markdown."""
        receipt = DecisionReceipt(
            receipt_id="md-save",
            gauntlet_id="gauntlet-mds",
            verdict="REJECTED",
            confidence=0.99,
        )

        md_path = receipt.save(tmp_path / "receipt", format="md")
        assert md_path.suffix == ".md"
        content = md_path.read_text()
        assert "REJECTED" in content

    def test_save_html(self, tmp_path):
        """Test saving as HTML."""
        receipt = DecisionReceipt(
            receipt_id="html-save",
            gauntlet_id="gauntlet-htmls",
            verdict="APPROVED",
            confidence=0.85,
        )

        html_path = receipt.save(tmp_path / "receipt", format="html")
        assert html_path.suffix == ".html"
        content = html_path.read_text()
        assert "<!DOCTYPE html>" in content

    def test_save_invalid_format(self, tmp_path):
        """Test that invalid format raises error."""
        receipt = DecisionReceipt(
            receipt_id="err-test",
            gauntlet_id="gauntlet-err",
        )

        with pytest.raises(ValueError, match="Unknown format"):
            receipt.save(tmp_path / "receipt", format="xml")


class TestDecisionReceiptEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_findings(self):
        """Test receipt with no findings."""
        receipt = DecisionReceipt(
            receipt_id="empty-findings",
            gauntlet_id="gauntlet-ef",
            verdict="APPROVED",
            confidence=1.0,
            findings=[],
        )
        assert len(receipt.findings) == 0
        md = receipt.to_markdown()
        assert "Total** | **0**" in md

    def test_long_input_summary(self):
        """Test receipt with very long input summary."""
        long_summary = "Test " * 500  # 2500 chars
        receipt = DecisionReceipt(
            receipt_id="long-input",
            gauntlet_id="gauntlet-li",
            input_summary=long_summary,
        )
        md = receipt.to_markdown()
        assert "..." in md  # Should be truncated

    def test_special_characters_in_content(self):
        """Test that special characters are handled in export."""
        receipt = DecisionReceipt(
            receipt_id="special-chars",
            gauntlet_id="gauntlet-sc",
            input_summary="Test <script>alert('xss')</script>",
            findings=[
                ReceiptFinding(
                    id="f1",
                    severity="HIGH",
                    category="xss",
                    title="XSS & SQL' Injection",
                    description='Contains "quotes" and <html>',
                )
            ],
        )
        # Should not raise during serialization
        json_str = receipt.to_json()
        html = receipt.to_html()
        # Special chars in findings should appear in HTML
        assert "XSS &amp;" in html or "XSS &" in html  # Either escaped or raw
        assert 'Contains "quotes"' in html or "Contains &quot;quotes&quot;" in html

    def test_zero_confidence(self):
        """Test receipt with zero confidence."""
        receipt = DecisionReceipt(
            receipt_id="zero-conf",
            gauntlet_id="gauntlet-zc",
            verdict="NEEDS_REVIEW",
            confidence=0.0,
        )
        assert receipt.confidence == 0.0
        md = receipt.to_markdown()
        assert "0%" in md

    def test_all_severity_counts(self):
        """Test receipt with findings at all severity levels."""
        findings = [
            ReceiptFinding(
                id="c1", severity="CRITICAL", category="sec", title="C1", description="d"
            ),
            ReceiptFinding(
                id="c2", severity="CRITICAL", category="sec", title="C2", description="d"
            ),
            ReceiptFinding(id="h1", severity="HIGH", category="perf", title="H1", description="d"),
            ReceiptFinding(
                id="m1", severity="MEDIUM", category="code", title="M1", description="d"
            ),
            ReceiptFinding(
                id="m2", severity="MEDIUM", category="code", title="M2", description="d"
            ),
            ReceiptFinding(
                id="m3", severity="MEDIUM", category="code", title="M3", description="d"
            ),
            ReceiptFinding(id="l1", severity="LOW", category="style", title="L1", description="d"),
        ]
        receipt = DecisionReceipt(
            receipt_id="all-sev",
            gauntlet_id="gauntlet-as",
            findings=findings,
            critical_count=2,
            high_count=1,
            medium_count=3,
            low_count=1,
        )

        data = receipt.to_dict()
        assert data["critical_count"] == 2
        assert data["high_count"] == 1
        assert data["medium_count"] == 3
        assert data["low_count"] == 1

    def test_many_agents(self):
        """Test receipt with many agents involved."""
        agents = [f"agent-{i}" for i in range(20)]
        receipt = DecisionReceipt(
            receipt_id="many-agents",
            gauntlet_id="gauntlet-ma",
            agents_involved=agents,
        )
        assert len(receipt.agents_involved) == 20
        md = receipt.to_markdown()
        assert "agent-0" in md
        assert "agent-19" in md

    def test_very_long_duration(self):
        """Test receipt with long duration."""
        receipt = DecisionReceipt(
            receipt_id="long-duration",
            gauntlet_id="gauntlet-ld",
            duration_seconds=86400.0,  # 24 hours
        )
        md = receipt.to_markdown()
        assert "86400.0s" in md

    def test_custom_timestamp(self):
        """Test receipt with custom timestamp."""
        custom_ts = "2025-06-15T14:30:00Z"
        receipt = DecisionReceipt(
            receipt_id="custom-ts",
            gauntlet_id="gauntlet-cts",
            timestamp=custom_ts,
        )
        assert receipt.timestamp == custom_ts
        # Checksum should be consistent
        assert receipt.verify_integrity()


class TestDecisionReceiptIntegrity:
    """Tests focused on checksum and integrity features."""

    def test_different_receipts_different_checksums(self):
        """Test that different receipts have different checksums."""
        receipt1 = DecisionReceipt(
            receipt_id="r1",
            gauntlet_id="g1",
            verdict="APPROVED",
            confidence=0.9,
            timestamp="2026-01-11T00:00:00",
        )
        receipt2 = DecisionReceipt(
            receipt_id="r2",
            gauntlet_id="g2",
            verdict="REJECTED",
            confidence=0.99,
            timestamp="2026-01-11T00:00:00",
        )
        assert receipt1.checksum != receipt2.checksum

    def test_checksum_includes_critical_fields(self):
        """Test that checksum changes when critical fields change."""
        base = {
            "receipt_id": "test",
            "gauntlet_id": "gauntlet",
            "timestamp": "2026-01-11T12:00:00",
        }

        # Different verdicts should have different checksums
        r1 = DecisionReceipt(**base, verdict="APPROVED", confidence=0.9)
        r2 = DecisionReceipt(**base, verdict="REJECTED", confidence=0.9)
        assert r1.checksum != r2.checksum

        # Different confidence should have different checksums
        r3 = DecisionReceipt(**base, verdict="APPROVED", confidence=0.5)
        assert r1.checksum != r3.checksum

    def test_checksum_not_affected_by_noncritical_fields(self):
        """Test that some fields don't affect the checksum."""
        # Note: The current implementation only includes specific fields in checksum
        base = {
            "receipt_id": "test",
            "gauntlet_id": "gauntlet",
            "verdict": "APPROVED",
            "confidence": 0.9,
            "timestamp": "2026-01-11T12:00:00",
        }

        r1 = DecisionReceipt(**base, input_summary="Summary 1")
        r2 = DecisionReceipt(**base, input_summary="Summary 2")

        # input_summary is not in the checksum calculation
        # so checksums should be the same
        assert r1.checksum == r2.checksum

    def test_integrity_preserved_through_json_roundtrip(self):
        """Test that integrity is preserved through JSON serialization."""
        original = DecisionReceipt(
            receipt_id="roundtrip-integrity",
            gauntlet_id="gauntlet-ri",
            verdict="APPROVED",
            confidence=0.95,
        )
        original_checksum = original.checksum

        # Roundtrip through JSON
        json_str = original.to_json()
        loaded = DecisionReceipt.from_json(json_str)

        # Loaded checksum should match original
        assert loaded.checksum == original_checksum
        assert loaded.verify_integrity()
