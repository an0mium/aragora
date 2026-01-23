"""
Tests for Codebase Audit Handler.

Tests cover enums, dataclasses, and basic handler creation.
"""

import pytest
from datetime import datetime, timezone

from aragora.server.handlers.features.codebase_audit import (
    CodebaseAuditHandler,
    ScanType,
    ScanStatus,
    FindingSeverity,
    FindingStatus,
    Finding,
    ScanResult,
)


class TestScanTypeEnum:
    """Tests for ScanType enum."""

    def test_all_scan_types_defined(self):
        """Test that all scan types are available."""
        expected = ["comprehensive", "sast", "bugs", "secrets", "dependencies", "metrics"]
        for scan_type in expected:
            assert ScanType(scan_type) is not None

    def test_scan_type_values(self):
        """Test scan type enum values."""
        assert ScanType.COMPREHENSIVE.value == "comprehensive"
        assert ScanType.SAST.value == "sast"
        assert ScanType.BUGS.value == "bugs"


class TestScanStatusEnum:
    """Tests for ScanStatus enum."""

    def test_all_statuses_defined(self):
        """Test that all scan statuses are available."""
        expected = ["pending", "running", "completed", "failed", "cancelled"]
        for status in expected:
            assert ScanStatus(status) is not None

    def test_status_values(self):
        """Test scan status enum values."""
        assert ScanStatus.PENDING.value == "pending"
        assert ScanStatus.RUNNING.value == "running"
        assert ScanStatus.COMPLETED.value == "completed"


class TestFindingSeverityEnum:
    """Tests for FindingSeverity enum."""

    def test_all_severities_defined(self):
        """Test that all severity levels are available."""
        expected = ["critical", "high", "medium", "low", "info"]
        for severity in expected:
            assert FindingSeverity(severity) is not None

    def test_severity_values(self):
        """Test severity enum values."""
        assert FindingSeverity.CRITICAL.value == "critical"
        assert FindingSeverity.HIGH.value == "high"
        assert FindingSeverity.INFO.value == "info"


class TestFindingStatusEnum:
    """Tests for FindingStatus enum."""

    def test_all_statuses_defined(self):
        """Test that all finding statuses are available."""
        expected = ["open", "dismissed", "fixed", "false_positive", "accepted_risk"]
        for status in expected:
            assert FindingStatus(status) is not None


class TestFinding:
    """Tests for Finding dataclass."""

    def test_finding_creation(self):
        """Test creating a finding."""
        finding = Finding(
            id="find_123",
            scan_id="scan_456",
            scan_type=ScanType.SAST,
            severity=FindingSeverity.HIGH,
            title="SQL Injection Vulnerability",
            description="User input directly concatenated into SQL query",
            file_path="src/database/queries.py",
            line_number=42,
            code_snippet='cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")',
            rule_id="python.sql-injection",
            cwe_id="CWE-89",
            owasp_category="A03:2021 - Injection",
            remediation="Use parameterized queries",
            confidence=0.95,
        )

        assert finding.id == "find_123"
        assert finding.scan_type == ScanType.SAST
        assert finding.severity == FindingSeverity.HIGH
        assert finding.line_number == 42

    def test_finding_defaults(self):
        """Test finding with default values."""
        finding = Finding(
            id="find_789",
            scan_id="scan_abc",
            scan_type=ScanType.BUGS,
            severity=FindingSeverity.MEDIUM,
            title="Potential Bug",
            description="Found a potential issue",
            file_path="src/app.py",
        )

        assert finding.status == FindingStatus.OPEN
        assert finding.confidence == 0.8
        assert finding.line_number is None
        assert finding.dismissed_by is None

    def test_finding_to_dict(self):
        """Test finding serialization."""
        finding = Finding(
            id="find_test",
            scan_id="scan_test",
            scan_type=ScanType.SECRETS,
            severity=FindingSeverity.CRITICAL,
            title="Hardcoded Secret",
            description="API key found in source",
            file_path="config/settings.py",
            line_number=10,
        )

        data = finding.to_dict()
        assert data["id"] == "find_test"
        assert data["scan_type"] == "secrets"
        assert data["severity"] == "critical"
        assert data["status"] == "open"


class TestScanResult:
    """Tests for ScanResult dataclass."""

    def test_scan_result_creation(self):
        """Test creating a scan result."""
        result = ScanResult(
            id="scan_123",
            tenant_id="tenant_456",
            scan_type=ScanType.COMPREHENSIVE,
            status=ScanStatus.RUNNING,
            target_path="/path/to/code",
            started_at=datetime.now(timezone.utc),
            files_scanned=100,
            findings_count=5,
        )

        assert result.id == "scan_123"
        assert result.tenant_id == "tenant_456"
        assert result.scan_type == ScanType.COMPREHENSIVE
        assert result.status == ScanStatus.RUNNING

    def test_scan_result_defaults(self):
        """Test scan result with default values."""
        result = ScanResult(
            id="scan_789",
            tenant_id="tenant_abc",
            scan_type=ScanType.SAST,
            status=ScanStatus.PENDING,
            target_path=".",
            started_at=datetime.now(timezone.utc),
        )

        assert result.completed_at is None
        assert result.files_scanned == 0
        assert result.findings_count == 0
        assert result.findings == []
        assert result.metrics == {}

    def test_scan_result_to_dict(self):
        """Test scan result serialization."""
        started = datetime.now(timezone.utc)
        result = ScanResult(
            id="scan_test",
            tenant_id="tenant_test",
            scan_type=ScanType.DEPENDENCIES,
            status=ScanStatus.COMPLETED,
            target_path="/project",
            started_at=started,
            files_scanned=50,
            findings_count=3,
            duration_seconds=120.5,
        )

        data = result.to_dict()
        assert data["id"] == "scan_test"
        assert data["scan_type"] == "dependencies"
        assert data["status"] == "completed"
        assert data["files_scanned"] == 50
        assert data["duration_seconds"] == 120.5


class TestCodebaseAuditHandler:
    """Tests for CodebaseAuditHandler class."""

    def test_handler_creation(self):
        """Test creating handler instance."""
        handler = CodebaseAuditHandler(server_context={})
        assert handler is not None

    def test_handler_routes(self):
        """Test that handler has route definitions."""
        assert hasattr(CodebaseAuditHandler, "ROUTES")
        routes = CodebaseAuditHandler.ROUTES
        assert "/api/v1/codebase/scan" in routes
        assert "/api/v1/codebase/sast" in routes
        assert "/api/v1/codebase/findings" in routes
        assert "/api/v1/codebase/dashboard" in routes
