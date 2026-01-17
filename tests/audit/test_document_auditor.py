"""
Tests for document auditor orchestrator.
"""

import pytest
from datetime import datetime

from aragora.audit.document_auditor import (
    AuditStatus,
    AuditType,
    FindingSeverity,
    FindingStatus,
    AuditFinding,
)


class TestAuditStatus:
    """Tests for AuditStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert AuditStatus.PENDING.value == "pending"
        assert AuditStatus.RUNNING.value == "running"
        assert AuditStatus.PAUSED.value == "paused"
        assert AuditStatus.COMPLETED.value == "completed"
        assert AuditStatus.FAILED.value == "failed"
        assert AuditStatus.CANCELLED.value == "cancelled"


class TestAuditType:
    """Tests for AuditType enum."""

    def test_audit_type_values(self):
        """Test all audit type values exist."""
        assert AuditType.SECURITY.value == "security"
        assert AuditType.COMPLIANCE.value == "compliance"
        assert AuditType.CONSISTENCY.value == "consistency"
        assert AuditType.QUALITY.value == "quality"
        assert AuditType.ALL.value == "all"


class TestFindingSeverity:
    """Tests for FindingSeverity enum."""

    def test_severity_values(self):
        """Test all severity values exist."""
        assert FindingSeverity.CRITICAL.value == "critical"
        assert FindingSeverity.HIGH.value == "high"
        assert FindingSeverity.MEDIUM.value == "medium"
        assert FindingSeverity.LOW.value == "low"
        assert FindingSeverity.INFO.value == "info"


class TestFindingStatus:
    """Tests for FindingStatus enum."""

    def test_finding_status_values(self):
        """Test all finding status values exist."""
        assert FindingStatus.OPEN.value == "open"
        assert FindingStatus.ACKNOWLEDGED.value == "acknowledged"
        assert FindingStatus.RESOLVED.value == "resolved"
        assert FindingStatus.FALSE_POSITIVE.value == "false_positive"
        assert FindingStatus.WONT_FIX.value == "wont_fix"


class TestAuditFinding:
    """Tests for AuditFinding dataclass."""

    def test_create_finding(self):
        """Test creating a basic finding."""
        finding = AuditFinding(
            title="Test Finding",
            description="This is a test",
            severity=FindingSeverity.HIGH,
            audit_type=AuditType.SECURITY,
        )

        assert finding.title == "Test Finding"
        assert finding.severity == FindingSeverity.HIGH
        assert finding.audit_type == AuditType.SECURITY
        assert finding.id is not None  # Auto-generated

    def test_finding_defaults(self):
        """Test finding default values."""
        finding = AuditFinding()

        assert finding.severity == FindingSeverity.MEDIUM
        assert finding.audit_type == AuditType.QUALITY
        assert finding.status == FindingStatus.OPEN
        assert finding.confidence == 0.8
        assert finding.confirmed_by == []
        assert finding.disputed_by == []

    def test_finding_to_dict(self):
        """Test finding serialization."""
        finding = AuditFinding(
            id="finding-123",
            session_id="session-456",
            document_id="doc-789",
            title="SQL Injection Risk",
            severity=FindingSeverity.CRITICAL,
            audit_type=AuditType.SECURITY,
            category="injection",
            evidence_text="SELECT * FROM users WHERE id = $input",
            evidence_location="config.py:42",
        )

        data = finding.to_dict()

        assert data["id"] == "finding-123"
        assert data["session_id"] == "session-456"
        assert data["severity"] == "critical"
        assert data["audit_type"] == "security"
        assert data["category"] == "injection"
        assert "created_at" in data

    def test_finding_from_dict(self):
        """Test finding deserialization."""
        data = {
            "id": "f-abc",
            "title": "Compliance Issue",
            "severity": "high",
            "audit_type": "compliance",
            "category": "gdpr",
            "document_id": "doc-1",
            "confidence": 0.9,
        }

        finding = AuditFinding.from_dict(data)

        assert finding.id == "f-abc"
        assert finding.title == "Compliance Issue"
        assert finding.severity == FindingSeverity.HIGH
        assert finding.audit_type == AuditType.COMPLIANCE
        assert finding.confidence == 0.9

    def test_finding_round_trip(self):
        """Test serialization round-trip."""
        original = AuditFinding(
            title="Round Trip Test",
            severity=FindingSeverity.MEDIUM,
            audit_type=AuditType.CONSISTENCY,
            category="references",
            tags=["important", "review"],
        )

        data = original.to_dict()
        restored = AuditFinding.from_dict(data)

        assert restored.title == original.title
        assert restored.severity == original.severity
        assert restored.audit_type == original.audit_type
        assert restored.category == original.category
        assert restored.tags == original.tags

    def test_finding_with_agent_attribution(self):
        """Test finding with agent attribution."""
        finding = AuditFinding(
            title="Multi-Agent Finding",
            found_by="claude-agent",
            confirmed_by=["gpt-agent", "gemini-agent"],
            disputed_by=[],
        )

        assert finding.found_by == "claude-agent"
        assert len(finding.confirmed_by) == 2
        assert "gpt-agent" in finding.confirmed_by

    def test_finding_timestamps(self):
        """Test that timestamps are set correctly."""
        finding = AuditFinding()

        assert isinstance(finding.created_at, datetime)
        assert isinstance(finding.updated_at, datetime)

        data = finding.to_dict()
        # Should be ISO format strings
        assert "T" in data["created_at"]

    def test_finding_with_all_fields(self):
        """Test finding with all fields populated."""
        finding = AuditFinding(
            id="complete-finding",
            session_id="session-1",
            document_id="doc-1",
            chunk_id="chunk-5",
            audit_type=AuditType.SECURITY,
            category="credentials",
            severity=FindingSeverity.CRITICAL,
            confidence=0.95,
            title="Exposed API Key",
            description="An API key was found in plain text",
            evidence_text="api_key = 'sk-1234567890abcdef'",
            evidence_location="config.py:15",
            recommendation="Move to environment variable or secrets manager",
            affected_scope="file",
            found_by="security-scanner",
            confirmed_by=["claude", "gpt"],
            disputed_by=[],
            status=FindingStatus.OPEN,
            tags=["security", "credentials", "high-priority"],
        )

        data = finding.to_dict()

        # Verify all fields are present
        assert data["id"] == "complete-finding"
        assert data["chunk_id"] == "chunk-5"
        assert data["confidence"] == 0.95
        assert data["recommendation"] == "Move to environment variable or secrets manager"
        assert data["affected_scope"] == "file"
        assert len(data["confirmed_by"]) == 2
        assert len(data["tags"]) == 3


class TestAuditTypesImport:
    """Tests to verify audit types can be imported."""

    def test_import_security_auditor(self):
        """Test SecurityAuditor can be imported."""
        from aragora.audit.audit_types import SecurityAuditor
        assert SecurityAuditor is not None

    def test_import_compliance_auditor(self):
        """Test ComplianceAuditor can be imported."""
        from aragora.audit.audit_types import ComplianceAuditor
        assert ComplianceAuditor is not None

    def test_import_consistency_auditor(self):
        """Test ConsistencyAuditor can be imported."""
        from aragora.audit.audit_types import ConsistencyAuditor
        assert ConsistencyAuditor is not None

    def test_import_quality_auditor(self):
        """Test QualityAuditor can be imported."""
        from aragora.audit.audit_types import QualityAuditor
        assert QualityAuditor is not None


class TestReportsImport:
    """Tests to verify reports module can be imported."""

    def test_import_defect_report(self):
        """Test DefectReport can be imported."""
        from aragora.audit.reports import DefectReport
        assert DefectReport is not None

    def test_import_report_config(self):
        """Test ReportConfig can be imported."""
        from aragora.audit.reports import ReportConfig
        assert ReportConfig is not None

    def test_import_report_format(self):
        """Test ReportFormat can be imported."""
        from aragora.audit.reports import ReportFormat
        assert ReportFormat is not None

    def test_import_generate_report(self):
        """Test generate_report function can be imported."""
        from aragora.audit.reports import generate_report
        assert callable(generate_report)


class TestFindingReportIntegration:
    """Integration tests between findings and reports."""

    def test_findings_to_report(self):
        """Test generating report from AuditFindings."""
        from aragora.audit.reports import DefectReport, ReportFormat

        findings = [
            AuditFinding(
                id="f1",
                title="Security Issue",
                severity=FindingSeverity.HIGH,
                audit_type=AuditType.SECURITY,
                category="credentials",
                document_id="doc-1",
            ),
            AuditFinding(
                id="f2",
                title="Compliance Gap",
                severity=FindingSeverity.MEDIUM,
                audit_type=AuditType.COMPLIANCE,
                category="gdpr",
                document_id="doc-2",
            ),
        ]

        report = DefectReport(findings, session_id="test-session")

        # Test Markdown output
        md = report.to_markdown()
        assert "Security Issue" in md
        assert "Compliance Gap" in md
        assert "[HIGH]" in md

        # Test JSON output
        json_out = report.to_json()
        assert "Security Issue" in json_out
        assert "test-session" in json_out

    def test_report_severity_grouping(self):
        """Test that report groups findings by severity correctly."""
        from aragora.audit.reports import DefectReport, ReportConfig

        findings = [
            AuditFinding(title="Low 1", severity=FindingSeverity.LOW),
            AuditFinding(title="Critical 1", severity=FindingSeverity.CRITICAL),
            AuditFinding(title="High 1", severity=FindingSeverity.HIGH),
            AuditFinding(title="Critical 2", severity=FindingSeverity.CRITICAL),
        ]

        config = ReportConfig(group_by="severity")
        report = DefectReport(findings, config=config)

        assert report.severity_stats.critical == 2
        assert report.severity_stats.high == 1
        assert report.severity_stats.low == 1
