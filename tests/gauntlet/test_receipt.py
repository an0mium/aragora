"""
Tests for Decision Receipt.

Tests the receipt module including:
- ProvenanceRecord dataclass
- ConsensusProof dataclass
- DecisionReceipt class
- Receipt creation from various result types
- Export formats (JSON, Markdown, HTML, SARIF, CSV)
- Integrity verification
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest

from aragora.gauntlet.receipt import (
    ConsensusProof,
    DecisionReceipt,
    ProvenanceRecord,
)


# =============================================================================
# ProvenanceRecord Tests
# =============================================================================


class TestProvenanceRecord:
    """Test ProvenanceRecord dataclass."""

    def test_basic_creation(self):
        """Test basic ProvenanceRecord creation."""
        record = ProvenanceRecord(
            timestamp="2024-01-15T10:30:00",
            event_type="attack",
            agent="claude",
            description="SQL injection attempt",
            evidence_hash="abc123",
        )
        assert record.timestamp == "2024-01-15T10:30:00"
        assert record.event_type == "attack"
        assert record.agent == "claude"
        assert record.description == "SQL injection attempt"
        assert record.evidence_hash == "abc123"

    def test_default_values(self):
        """Test default values."""
        record = ProvenanceRecord(
            timestamp="2024-01-15T10:30:00",
            event_type="probe",
        )
        assert record.agent is None
        assert record.description == ""
        assert record.evidence_hash == ""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        record = ProvenanceRecord(
            timestamp="2024-01-15T10:30:00",
            event_type="verdict",
            description="Final verdict",
        )
        data = record.to_dict()

        assert data["timestamp"] == "2024-01-15T10:30:00"
        assert data["event_type"] == "verdict"
        assert data["description"] == "Final verdict"
        assert "agent" in data
        assert "evidence_hash" in data


# =============================================================================
# ConsensusProof Tests
# =============================================================================


class TestConsensusProof:
    """Test ConsensusProof dataclass."""

    def test_basic_creation(self):
        """Test basic ConsensusProof creation."""
        proof = ConsensusProof(
            reached=True,
            confidence=0.85,
            supporting_agents=["claude", "gpt-4"],
            dissenting_agents=["gemini"],
            method="majority",
        )
        assert proof.reached is True
        assert proof.confidence == 0.85
        assert "claude" in proof.supporting_agents
        assert "gemini" in proof.dissenting_agents
        assert proof.method == "majority"

    def test_default_values(self):
        """Test default values."""
        proof = ConsensusProof(
            reached=False,
            confidence=0.5,
        )
        assert proof.supporting_agents == []
        assert proof.dissenting_agents == []
        assert proof.method == "majority"
        assert proof.evidence_hash == ""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        proof = ConsensusProof(
            reached=True,
            confidence=0.9,
            supporting_agents=["agent-1", "agent-2"],
            method="unanimous",
        )
        data = proof.to_dict()

        assert data["reached"] is True
        assert data["confidence"] == 0.9
        assert data["supporting_agents"] == ["agent-1", "agent-2"]
        assert data["method"] == "unanimous"


# =============================================================================
# DecisionReceipt Tests
# =============================================================================


class TestDecisionReceiptCreation:
    """Test DecisionReceipt creation."""

    @pytest.fixture
    def basic_receipt(self):
        """Create a basic receipt for testing."""
        return DecisionReceipt(
            receipt_id="test-receipt-123",
            gauntlet_id="gauntlet-456",
            timestamp="2024-01-15T10:30:00Z",
            input_summary="Test input content",
            input_hash="abc123def456",
            risk_summary={"critical": 1, "high": 2, "medium": 3, "low": 4},
            attacks_attempted=10,
            attacks_successful=3,
            probes_run=15,
            vulnerabilities_found=6,
            verdict="CONDITIONAL",
            confidence=0.75,
            robustness_score=0.65,
        )

    def test_basic_creation(self, basic_receipt):
        """Test basic DecisionReceipt creation."""
        assert basic_receipt.receipt_id == "test-receipt-123"
        assert basic_receipt.gauntlet_id == "gauntlet-456"
        assert basic_receipt.verdict == "CONDITIONAL"
        assert basic_receipt.confidence == 0.75

    def test_auto_hash_generation(self, basic_receipt):
        """Test automatic artifact hash generation."""
        assert basic_receipt.artifact_hash  # Should be non-empty
        assert len(basic_receipt.artifact_hash) == 64  # SHA-256

    def test_deterministic_hash(self):
        """Test hash is deterministic for same content."""
        receipt1 = DecisionReceipt(
            receipt_id="test-123",
            gauntlet_id="gauntlet-456",
            timestamp="2024-01-15T10:30:00Z",
            input_summary="Same content",
            input_hash="same-hash",
            risk_summary={"critical": 0},
            attacks_attempted=1,
            attacks_successful=0,
            probes_run=1,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.85,
        )
        receipt2 = DecisionReceipt(
            receipt_id="test-123",
            gauntlet_id="gauntlet-456",
            timestamp="2024-01-15T10:30:00Z",
            input_summary="Same content",
            input_hash="same-hash",
            risk_summary={"critical": 0},
            attacks_attempted=1,
            attacks_successful=0,
            probes_run=1,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.85,
        )
        assert receipt1.artifact_hash == receipt2.artifact_hash


class TestDecisionReceiptIntegrity:
    """Test receipt integrity verification."""

    def test_verify_integrity_valid(self):
        """Test integrity verification succeeds for unmodified receipt."""
        receipt = DecisionReceipt(
            receipt_id="test-123",
            gauntlet_id="gauntlet-456",
            timestamp="2024-01-15T10:30:00Z",
            input_summary="Test",
            input_hash="hash123",
            risk_summary={"critical": 0},
            attacks_attempted=1,
            attacks_successful=0,
            probes_run=1,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.85,
        )
        assert receipt.verify_integrity() is True

    def test_verify_integrity_tampered(self):
        """Test integrity verification fails for modified receipt."""
        receipt = DecisionReceipt(
            receipt_id="test-123",
            gauntlet_id="gauntlet-456",
            timestamp="2024-01-15T10:30:00Z",
            input_summary="Test",
            input_hash="hash123",
            risk_summary={"critical": 0},
            attacks_attempted=1,
            attacks_successful=0,
            probes_run=1,
            vulnerabilities_found=0,
            verdict="PASS",
            confidence=0.9,
            robustness_score=0.85,
        )
        original_hash = receipt.artifact_hash

        # Tamper with receipt
        receipt.verdict = "FAIL"

        # Original hash no longer matches
        assert receipt.artifact_hash == original_hash  # Hash unchanged
        assert receipt.verify_integrity() is False  # But verification fails


class TestDecisionReceiptSerialization:
    """Test receipt serialization."""

    @pytest.fixture
    def full_receipt(self):
        """Create a receipt with all fields populated."""
        consensus = ConsensusProof(
            reached=True,
            confidence=0.85,
            supporting_agents=["claude", "gpt-4"],
            method="majority",
        )
        provenance = [
            ProvenanceRecord(
                timestamp="2024-01-15T10:00:00Z",
                event_type="attack",
                agent="claude",
                description="Security probe",
            ),
            ProvenanceRecord(
                timestamp="2024-01-15T10:30:00Z",
                event_type="verdict",
                description="Final verdict",
            ),
        ]
        return DecisionReceipt(
            receipt_id="test-123",
            gauntlet_id="gauntlet-456",
            timestamp="2024-01-15T10:30:00Z",
            input_summary="Test input for validation",
            input_hash="abc123",
            risk_summary={"critical": 0, "high": 1, "medium": 2, "low": 3},
            attacks_attempted=10,
            attacks_successful=2,
            probes_run=15,
            vulnerabilities_found=3,
            vulnerability_details=[
                {
                    "id": "vuln-001",
                    "title": "SQL Injection",
                    "severity": "HIGH",
                    "description": "Possible SQL injection in query",
                    "mitigation": "Use parameterized queries",
                }
            ],
            verdict="CONDITIONAL",
            confidence=0.85,
            robustness_score=0.75,
            verdict_reasoning="Some issues found but manageable",
            dissenting_views=["Agent X disagreed on severity"],
            consensus_proof=consensus,
            provenance_chain=provenance,
            config_used={"template": "security"},
        )

    def test_to_dict(self, full_receipt):
        """Test serialization to dictionary."""
        data = full_receipt.to_dict()

        assert data["receipt_id"] == "test-123"
        assert data["gauntlet_id"] == "gauntlet-456"
        assert data["verdict"] == "CONDITIONAL"
        assert data["confidence"] == 0.85
        assert "risk_summary" in data
        assert "consensus_proof" in data
        assert "provenance_chain" in data
        assert len(data["provenance_chain"]) == 2

    def test_from_dict(self, full_receipt):
        """Test deserialization from dictionary."""
        data = full_receipt.to_dict()
        restored = DecisionReceipt.from_dict(data)

        assert restored.receipt_id == full_receipt.receipt_id
        assert restored.verdict == full_receipt.verdict
        assert restored.confidence == full_receipt.confidence
        assert restored.consensus_proof is not None
        assert len(restored.provenance_chain) == 2

    def test_to_json(self, full_receipt):
        """Test JSON export."""
        json_str = full_receipt.to_json()
        data = json.loads(json_str)

        assert data["receipt_id"] == "test-123"
        assert data["verdict"] == "CONDITIONAL"

    def test_round_trip_serialization(self, full_receipt):
        """Test round-trip serialization preserves data."""
        json_str = full_receipt.to_json()
        data = json.loads(json_str)
        restored = DecisionReceipt.from_dict(data)

        assert restored.receipt_id == full_receipt.receipt_id
        assert restored.verdict == full_receipt.verdict
        assert restored.artifact_hash == full_receipt.artifact_hash


class TestDecisionReceiptMarkdown:
    """Test Markdown export."""

    @pytest.fixture
    def receipt_with_findings(self):
        """Create receipt with vulnerability details."""
        return DecisionReceipt(
            receipt_id="test-123",
            gauntlet_id="gauntlet-456",
            timestamp="2024-01-15T10:30:00Z",
            input_summary="API endpoint validation",
            input_hash="abc123",
            risk_summary={"critical": 1, "high": 2, "medium": 1, "low": 0},
            attacks_attempted=5,
            attacks_successful=2,
            probes_run=10,
            vulnerabilities_found=4,
            vulnerability_details=[
                {
                    "id": "vuln-001",
                    "title": "Authentication Bypass",
                    "severity": "CRITICAL",
                    "category": "security",
                    "description": "Token validation can be bypassed",
                    "mitigation": "Implement proper JWT validation",
                }
            ],
            verdict="FAIL",
            confidence=0.9,
            robustness_score=0.3,
            verdict_reasoning="Critical security issues found",
        )

    def test_to_markdown_contains_verdict(self, receipt_with_findings):
        """Test Markdown contains verdict section."""
        md = receipt_with_findings.to_markdown()

        assert "Decision Receipt" in md
        assert "FAIL" in md
        assert "Confidence:" in md
        assert "90.0%" in md or "90%" in md or "0.9" in md

    def test_to_markdown_contains_risk_summary(self, receipt_with_findings):
        """Test Markdown contains risk summary."""
        md = receipt_with_findings.to_markdown()

        assert "Risk Summary" in md
        assert "Critical" in md
        assert "High" in md

    def test_to_markdown_contains_findings(self, receipt_with_findings):
        """Test Markdown contains findings."""
        md = receipt_with_findings.to_markdown()

        assert "Authentication Bypass" in md
        assert "CRITICAL" in md

    def test_to_markdown_contains_integrity(self, receipt_with_findings):
        """Test Markdown contains integrity section."""
        md = receipt_with_findings.to_markdown()

        assert "Integrity" in md
        assert receipt_with_findings.input_hash in md


class TestDecisionReceiptHTML:
    """Test HTML export."""

    @pytest.fixture
    def basic_receipt(self):
        """Create a basic receipt."""
        return DecisionReceipt(
            receipt_id="test-123",
            gauntlet_id="gauntlet-456",
            timestamp="2024-01-15T10:30:00Z",
            input_summary="Test",
            input_hash="abc123",
            risk_summary={"critical": 0, "high": 1},
            attacks_attempted=5,
            attacks_successful=1,
            probes_run=10,
            vulnerabilities_found=1,
            verdict="PASS",
            confidence=0.85,
            robustness_score=0.8,
        )

    def test_to_html_structure(self, basic_receipt):
        """Test HTML has proper structure."""
        html = basic_receipt.to_html()

        assert "<!DOCTYPE html>" in html
        assert "<html>" in html
        assert "</html>" in html
        assert "<head>" in html
        assert "<body>" in html

    def test_to_html_contains_data(self, basic_receipt):
        """Test HTML contains receipt data."""
        html = basic_receipt.to_html()

        assert "test-123" in html
        assert "PASS" in html


class TestDecisionReceiptSARIF:
    """Test SARIF export."""

    @pytest.fixture
    def receipt_with_findings(self):
        """Create receipt with vulnerability details for SARIF."""
        return DecisionReceipt(
            receipt_id="test-123",
            gauntlet_id="gauntlet-456",
            timestamp="2024-01-15T10:30:00Z",
            input_summary="Code review",
            input_hash="abc123",
            risk_summary={"critical": 1, "high": 1},
            attacks_attempted=5,
            attacks_successful=2,
            probes_run=10,
            vulnerabilities_found=2,
            vulnerability_details=[
                {
                    "id": "vuln-001",
                    "title": "Buffer Overflow",
                    "severity": "CRITICAL",
                    "severity_level": "CRITICAL",
                    "category": "memory_safety",
                    "description": "Unbounded buffer write",
                },
                {
                    "id": "vuln-002",
                    "title": "XSS Vulnerability",
                    "severity": "HIGH",
                    "severity_level": "HIGH",
                    "category": "web_security",
                    "description": "Unescaped user input",
                },
            ],
            verdict="FAIL",
            confidence=0.95,
            robustness_score=0.3,
        )

    def test_to_sarif_structure(self, receipt_with_findings):
        """Test SARIF has proper structure."""
        sarif = receipt_with_findings.to_sarif()

        assert "$schema" in sarif
        assert "version" in sarif
        assert sarif["version"] == "2.1.0"
        assert "runs" in sarif
        assert len(sarif["runs"]) == 1

    def test_to_sarif_tool_info(self, receipt_with_findings):
        """Test SARIF contains tool information."""
        sarif = receipt_with_findings.to_sarif()
        run = sarif["runs"][0]

        assert "tool" in run
        assert "driver" in run["tool"]
        assert run["tool"]["driver"]["name"] == "Aragora Gauntlet"

    def test_to_sarif_results(self, receipt_with_findings):
        """Test SARIF contains results."""
        sarif = receipt_with_findings.to_sarif()
        run = sarif["runs"][0]

        assert "results" in run
        assert len(run["results"]) == 2

    def test_to_sarif_json(self, receipt_with_findings):
        """Test SARIF JSON export is valid."""
        sarif_json = receipt_with_findings.to_sarif_json()
        data = json.loads(sarif_json)

        assert "version" in data
        assert data["version"] == "2.1.0"


class TestDecisionReceiptCSV:
    """Test CSV export."""

    @pytest.fixture
    def receipt_with_findings(self):
        """Create receipt with findings for CSV."""
        return DecisionReceipt(
            receipt_id="test-123",
            gauntlet_id="gauntlet-456",
            timestamp="2024-01-15T10:30:00Z",
            input_summary="Test",
            input_hash="abc123",
            risk_summary={"critical": 1},
            attacks_attempted=1,
            attacks_successful=1,
            probes_run=1,
            vulnerabilities_found=1,
            vulnerability_details=[
                {
                    "id": "vuln-001",
                    "category": "security",
                    "severity": "CRITICAL",
                    "title": "Test Finding",
                    "description": "Test description",
                    "mitigation": "Fix it",
                    "verified": True,
                    "source": "red_team",
                }
            ],
            verdict="FAIL",
            confidence=0.9,
            robustness_score=0.2,
        )

    def test_to_csv_structure(self, receipt_with_findings):
        """Test CSV has proper structure."""
        csv = receipt_with_findings.to_csv()
        lines = csv.strip().split("\n")

        # Header + 1 data row
        assert len(lines) == 2

        # Check header
        header = lines[0]
        assert "Finding ID" in header
        assert "Severity" in header
        assert "Category" in header

    def test_to_csv_content(self, receipt_with_findings):
        """Test CSV contains data."""
        csv = receipt_with_findings.to_csv()

        assert "vuln-001" in csv
        assert "security" in csv
        assert "Test Finding" in csv


# =============================================================================
# Integration Tests
# =============================================================================


class TestDecisionReceiptIntegration:
    """Integration tests for DecisionReceipt."""

    def test_full_workflow(self):
        """Test complete receipt workflow."""
        # Create receipt with all fields
        consensus = ConsensusProof(
            reached=True,
            confidence=0.9,
            supporting_agents=["claude", "gpt-4", "gemini"],
            method="majority",
        )

        provenance = [
            ProvenanceRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type="attack",
                agent="claude",
                description="Security probe executed",
                evidence_hash="abc123",
            ),
            ProvenanceRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type="probe",
                agent="gpt-4",
                description="Hallucination check",
                evidence_hash="def456",
            ),
            ProvenanceRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type="verdict",
                description="Final verdict rendered",
            ),
        ]

        receipt = DecisionReceipt(
            receipt_id="integration-test-001",
            gauntlet_id="gauntlet-integration",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_summary="Full integration test input",
            input_hash=hashlib.sha256(b"test input").hexdigest(),
            risk_summary={"critical": 0, "high": 1, "medium": 2, "low": 3},
            attacks_attempted=20,
            attacks_successful=3,
            probes_run=30,
            vulnerabilities_found=6,
            vulnerability_details=[
                {
                    "id": "vuln-001",
                    "title": "Rate Limiting Bypass",
                    "severity": "HIGH",
                    "category": "security",
                    "description": "Rate limits can be bypassed via header manipulation",
                    "mitigation": "Validate all rate limit headers server-side",
                }
            ],
            verdict="CONDITIONAL",
            confidence=0.85,
            robustness_score=0.7,
            verdict_reasoning="Minor security issues found but manageable",
            dissenting_views=["Agent gemini suggested stricter validation"],
            consensus_proof=consensus,
            provenance_chain=provenance,
            config_used={"template": "api_security", "agents": 3},
        )

        # Verify integrity
        assert receipt.verify_integrity() is True

        # Export to all formats
        json_export = receipt.to_json()
        md_export = receipt.to_markdown()
        html_export = receipt.to_html()
        sarif_export = receipt.to_sarif()
        csv_export = receipt.to_csv()

        # Verify exports are non-empty
        assert len(json_export) > 0
        assert len(md_export) > 0
        assert len(html_export) > 0
        assert len(sarif_export) > 0
        assert len(csv_export) > 0

        # Verify round-trip
        data = json.loads(json_export)
        restored = DecisionReceipt.from_dict(data)

        assert restored.receipt_id == receipt.receipt_id
        assert restored.verdict == receipt.verdict
        assert restored.artifact_hash == receipt.artifact_hash
        assert restored.verify_integrity() is True
