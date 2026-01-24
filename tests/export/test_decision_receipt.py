"""
Tests for Decision Receipt generation and export.

Tests the audit-ready compliance artifacts for Gauntlet stress-tests.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from aragora.export.decision_receipt import (
    DecisionReceipt,
    ReceiptDissent,
    ReceiptFinding,
    ReceiptVerification,
)


class TestReceiptFinding:
    """Tests for ReceiptFinding dataclass."""

    def test_create_basic(self):
        """Test basic finding creation."""
        finding = ReceiptFinding(
            id="finding-001",
            severity="HIGH",
            category="security",
            title="SQL Injection Risk",
            description="Potential SQL injection in user input handling",
        )

        assert finding.id == "finding-001"
        assert finding.severity == "HIGH"
        assert finding.category == "security"
        assert finding.mitigation is None
        assert finding.verified is False

    def test_create_with_mitigation(self):
        """Test finding with mitigation."""
        finding = ReceiptFinding(
            id="finding-002",
            severity="CRITICAL",
            category="security",
            title="Auth Bypass",
            description="Authentication can be bypassed",
            mitigation="Implement proper token validation",
            verified=True,
        )

        assert finding.mitigation == "Implement proper token validation"
        assert finding.verified is True


class TestReceiptDissent:
    """Tests for ReceiptDissent dataclass."""

    def test_create_basic(self):
        """Test basic dissent creation."""
        dissent = ReceiptDissent(
            agent="claude",
            type="partial_disagree",
            severity=0.6,
            reasons=["Insufficient evidence", "Missing edge cases"],
        )

        assert dissent.agent == "claude"
        assert dissent.type == "partial_disagree"
        assert len(dissent.reasons) == 2
        assert dissent.alternative is None

    def test_create_with_alternative(self):
        """Test dissent with alternative view."""
        dissent = ReceiptDissent(
            agent="gpt-4",
            type="full_disagree",
            severity=0.9,
            reasons=["Approach is flawed"],
            alternative="Consider using a different architecture",
        )

        assert dissent.alternative is not None


class TestReceiptVerification:
    """Tests for ReceiptVerification dataclass."""

    def test_create_verified(self):
        """Test verified claim creation."""
        verification = ReceiptVerification(
            claim="All inputs are sanitized",
            verified=True,
            method="static_analysis",
            proof_hash="abc123def456",
        )

        assert verification.verified is True
        assert verification.proof_hash is not None

    def test_create_refuted(self):
        """Test refuted claim creation."""
        verification = ReceiptVerification(
            claim="No null pointer dereferences",
            verified=False,
            method="formal_verification",
        )

        assert verification.verified is False
        assert verification.proof_hash is None


class TestDecisionReceipt:
    """Tests for DecisionReceipt dataclass."""

    @pytest.fixture
    def basic_receipt(self) -> DecisionReceipt:
        """Create a basic receipt for testing."""
        return DecisionReceipt(
            receipt_id="test-receipt-123",
            gauntlet_id="gauntlet-456",
            input_summary="Test API specification",
            verdict="APPROVED",
            confidence=0.85,
            risk_level="MEDIUM",
            risk_score=0.35,
            robustness_score=0.90,
            coverage_score=0.88,
            verification_coverage=0.75,
            agents_involved=["claude", "gpt-4", "gemini"],
            rounds_completed=3,
            duration_seconds=45.5,
        )

    @pytest.fixture
    def detailed_receipt(self, basic_receipt: DecisionReceipt) -> DecisionReceipt:
        """Create a detailed receipt with findings."""
        basic_receipt.findings = [
            ReceiptFinding(
                id="f1",
                severity="CRITICAL",
                category="security",
                title="SQL Injection",
                description="User input not sanitized",
                mitigation="Use prepared statements",
            ),
            ReceiptFinding(
                id="f2",
                severity="HIGH",
                category="performance",
                title="N+1 Query",
                description="Inefficient database access pattern",
            ),
            ReceiptFinding(
                id="f3",
                severity="MEDIUM",
                category="code_quality",
                title="Missing Validation",
                description="Input validation incomplete",
            ),
        ]
        basic_receipt.critical_count = 1
        basic_receipt.high_count = 1
        basic_receipt.medium_count = 1

        basic_receipt.dissenting_views = [
            ReceiptDissent(
                agent="gpt-4",
                type="partial_disagree",
                severity=0.5,
                reasons=["Edge cases not considered"],
            )
        ]

        basic_receipt.verified_claims = [
            ReceiptVerification(
                claim="All endpoints authenticated",
                verified=True,
                method="code_review",
                proof_hash="abc123",
            )
        ]

        basic_receipt.unverified_claims = [
            "Rate limiting is sufficient",
        ]

        basic_receipt.mitigations = [
            "Implement input sanitization",
            "Add prepared statements",
        ]

        return basic_receipt

    def test_create_basic(self, basic_receipt: DecisionReceipt):
        """Test basic receipt creation."""
        assert basic_receipt.receipt_id == "test-receipt-123"
        assert basic_receipt.verdict == "APPROVED"
        assert basic_receipt.confidence == 0.85

    def test_checksum_computed_on_init(self, basic_receipt: DecisionReceipt):
        """Test that checksum is computed on initialization."""
        assert basic_receipt.checksum != ""
        assert len(basic_receipt.checksum) == 16

    def test_checksum_consistency(self):
        """Test that same data produces same checksum."""
        # Use explicit timestamp to ensure consistency
        fixed_timestamp = "2024-01-15T10:30:00"
        receipt1 = DecisionReceipt(
            receipt_id="test-123",
            gauntlet_id="gauntlet-456",
            verdict="APPROVED",
            confidence=0.85,
            timestamp=fixed_timestamp,
        )
        # Create with same data
        receipt1_copy = DecisionReceipt(
            receipt_id="test-123",
            gauntlet_id="gauntlet-456",
            verdict="APPROVED",
            confidence=0.85,
            timestamp=fixed_timestamp,
        )

        assert receipt1.checksum == receipt1_copy.checksum

    def test_verify_integrity_valid(self, basic_receipt: DecisionReceipt):
        """Test integrity verification for valid receipt."""
        assert basic_receipt.verify_integrity() is True

    def test_verify_integrity_tampered(self, basic_receipt: DecisionReceipt):
        """Test integrity verification detects tampering."""
        # Tamper with the receipt
        basic_receipt.verdict = "REJECTED"

        # Integrity check should fail
        assert basic_receipt.verify_integrity() is False

    def test_to_dict(self, detailed_receipt: DecisionReceipt):
        """Test conversion to dictionary."""
        result = detailed_receipt.to_dict()

        assert isinstance(result, dict)
        assert result["receipt_id"] == "test-receipt-123"
        assert result["verdict"] == "APPROVED"
        assert len(result["findings"]) == 3
        assert len(result["dissenting_views"]) == 1
        assert len(result["verified_claims"]) == 1

    def test_to_json(self, basic_receipt: DecisionReceipt):
        """Test JSON serialization."""
        json_str = basic_receipt.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["receipt_id"] == "test-receipt-123"

    def test_to_json_with_indent(self, basic_receipt: DecisionReceipt):
        """Test JSON serialization with custom indent."""
        json_str = basic_receipt.to_json(indent=4)

        # Should have proper indentation
        assert "    " in json_str  # 4-space indent

    def test_to_markdown(self, detailed_receipt: DecisionReceipt):
        """Test Markdown export."""
        md = detailed_receipt.to_markdown()

        assert "# Decision Receipt" in md
        assert "## Verdict" in md
        assert "APPROVED" in md
        assert "85%" in md
        assert "## Findings Summary" in md
        assert "Critical" in md
        assert "SQL Injection" in md  # Critical finding
        assert "N+1 Query" in md  # High finding

    def test_to_markdown_with_dissent(self, detailed_receipt: DecisionReceipt):
        """Test Markdown includes dissenting views."""
        md = detailed_receipt.to_markdown()

        assert "## Dissenting Views" in md
        assert "gpt-4" in md
        assert "partial_disagree" in md

    def test_to_markdown_with_verification(self, detailed_receipt: DecisionReceipt):
        """Test Markdown includes verification results."""
        md = detailed_receipt.to_markdown()

        assert "## Verification Results" in md
        assert "75%" in md  # verification_coverage
        assert "[VERIFIED]" in md

    def test_to_html(self, detailed_receipt: DecisionReceipt):
        """Test HTML export."""
        html = detailed_receipt.to_html()

        assert "<!DOCTYPE html>" in html
        assert "<title>Decision Receipt" in html
        assert "test-receipt-123" in html
        assert "APPROVED" in html

    def test_to_html_verdict_colors(self):
        """Test HTML uses correct colors for different verdicts."""
        verdicts = [
            ("APPROVED", "#28a745"),
            ("APPROVED_WITH_CONDITIONS", "#ffc107"),
            ("NEEDS_REVIEW", "#fd7e14"),
            ("REJECTED", "#dc3545"),
        ]

        for verdict, expected_color in verdicts:
            receipt = DecisionReceipt(
                receipt_id="test",
                gauntlet_id="gauntlet",
                verdict=verdict,
                confidence=0.5,
            )
            html = receipt.to_html()
            assert expected_color in html

    def test_save_json(self, basic_receipt: DecisionReceipt):
        """Test saving as JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "receipt"
            result_path = basic_receipt.save(path, format="json")

            assert result_path.suffix == ".json"
            assert result_path.exists()

            content = json.loads(result_path.read_text())
            assert content["receipt_id"] == "test-receipt-123"

    def test_save_markdown(self, basic_receipt: DecisionReceipt):
        """Test saving as Markdown file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "receipt"
            result_path = basic_receipt.save(path, format="md")

            assert result_path.suffix == ".md"
            assert result_path.exists()
            assert "# Decision Receipt" in result_path.read_text()

    def test_save_html(self, basic_receipt: DecisionReceipt):
        """Test saving as HTML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "receipt"
            result_path = basic_receipt.save(path, format="html")

            assert result_path.suffix == ".html"
            assert result_path.exists()
            assert "<!DOCTYPE html>" in result_path.read_text()

    def test_save_unknown_format(self, basic_receipt: DecisionReceipt):
        """Test saving with unknown format raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "receipt"

            with pytest.raises(ValueError, match="Unknown format"):
                basic_receipt.save(path, format="unknown")

    def test_from_json(self, detailed_receipt: DecisionReceipt):
        """Test loading from JSON."""
        json_str = detailed_receipt.to_json()
        loaded = DecisionReceipt.from_json(json_str)

        assert loaded.receipt_id == detailed_receipt.receipt_id
        assert loaded.verdict == detailed_receipt.verdict
        assert len(loaded.findings) == len(detailed_receipt.findings)
        assert len(loaded.dissenting_views) == len(detailed_receipt.dissenting_views)

    def test_load_from_file(self, basic_receipt: DecisionReceipt):
        """Test loading from saved file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "receipt"
            saved_path = basic_receipt.save(path, format="json")

            loaded = DecisionReceipt.load(saved_path)
            assert loaded.receipt_id == basic_receipt.receipt_id

    def test_roundtrip_json(self, detailed_receipt: DecisionReceipt):
        """Test JSON serialization roundtrip preserves data."""
        json_str = detailed_receipt.to_json()
        loaded = DecisionReceipt.from_json(json_str)

        assert loaded.receipt_id == detailed_receipt.receipt_id
        assert loaded.verdict == detailed_receipt.verdict
        assert loaded.confidence == detailed_receipt.confidence
        assert loaded.risk_level == detailed_receipt.risk_level
        assert len(loaded.findings) == len(detailed_receipt.findings)
        assert loaded.findings[0].title == detailed_receipt.findings[0].title


class TestDecisionReceiptEdgeCases:
    """Edge case tests for DecisionReceipt."""

    def test_empty_receipt(self):
        """Test receipt with minimal data."""
        receipt = DecisionReceipt(
            receipt_id="minimal",
            gauntlet_id="gauntlet",
        )

        assert receipt.verdict == "NEEDS_REVIEW"
        assert receipt.confidence == 0.0
        assert receipt.findings == []

    def test_long_input_summary_markdown(self):
        """Test Markdown truncates long input summaries."""
        receipt = DecisionReceipt(
            receipt_id="test",
            gauntlet_id="gauntlet",
            input_summary="A" * 2000,
        )

        md = receipt.to_markdown()
        assert "..." in md  # Should be truncated

    def test_many_findings(self):
        """Test handling many findings."""
        findings = [
            ReceiptFinding(
                id=f"f{i}",
                severity="LOW",
                category="test",
                title=f"Finding {i}",
                description=f"Description {i}",
            )
            for i in range(100)
        ]

        receipt = DecisionReceipt(
            receipt_id="test",
            gauntlet_id="gauntlet",
            findings=findings,
            low_count=100,
        )

        # Should handle without error
        json_str = receipt.to_json()
        md = receipt.to_markdown()
        html = receipt.to_html()

        assert len(json.loads(json_str)["findings"]) == 100

    def test_special_characters_in_content(self):
        """Test handling of special characters."""
        receipt = DecisionReceipt(
            receipt_id="test",
            gauntlet_id="gauntlet",
            input_summary='Input with "quotes" and <brackets> & ampersands',
            findings=[
                ReceiptFinding(
                    id="f1",
                    severity="LOW",
                    category="test",
                    title="Title with 'single' quotes",
                    description="Description with `backticks` and *asterisks*",
                )
            ],
        )

        # All exports should handle without error
        json_str = receipt.to_json()
        md = receipt.to_markdown()
        html = receipt.to_html()

        assert "quotes" in json_str
        assert "quotes" in md
        assert "quotes" in html

    def test_unicode_content(self):
        """Test handling of unicode content."""
        receipt = DecisionReceipt(
            receipt_id="test",
            gauntlet_id="gauntlet",
            input_summary="Testing with emojis: Cloud adoption",
            agents_involved=["Claude", "GPT-4"],
        )

        json_str = receipt.to_json()
        md = receipt.to_markdown()

        assert "emojis" in json_str
        assert "emojis" in md
