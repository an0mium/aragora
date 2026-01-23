"""
Tests for Codebase Audit Handler.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.features.codebase_audit import (
    CodebaseAuditHandler,
    get_codebase_audit_handler,
    handle_codebase_audit,
    ScanType,
    ScanStatus,
    FindingSeverity,
    FindingStatus,
    Finding,
    ScanResult,
    _get_mock_sast_findings,
    _get_mock_bug_findings,
    _get_mock_secrets_findings,
    _get_mock_dependency_findings,
    _get_mock_metrics,
)


class TestScanType:
    """Tests for ScanType enum."""

    def test_scan_type_values(self):
        """Test scan type enum values."""
        assert ScanType.COMPREHENSIVE.value == "comprehensive"
        assert ScanType.SAST.value == "sast"
        assert ScanType.BUGS.value == "bugs"
        assert ScanType.SECRETS.value == "secrets"
        assert ScanType.DEPENDENCIES.value == "dependencies"
        assert ScanType.METRICS.value == "metrics"


class TestScanStatus:
    """Tests for ScanStatus enum."""

    def test_scan_status_values(self):
        """Test scan status enum values."""
        assert ScanStatus.PENDING.value == "pending"
        assert ScanStatus.RUNNING.value == "running"
        assert ScanStatus.COMPLETED.value == "completed"
        assert ScanStatus.FAILED.value == "failed"


class TestFindingSeverity:
    """Tests for FindingSeverity enum."""

    def test_severity_values(self):
        """Test severity enum values."""
        assert FindingSeverity.CRITICAL.value == "critical"
        assert FindingSeverity.HIGH.value == "high"
        assert FindingSeverity.MEDIUM.value == "medium"
        assert FindingSeverity.LOW.value == "low"
        assert FindingSeverity.INFO.value == "info"


class TestFinding:
    """Tests for Finding dataclass."""

    def test_finding_creation(self):
        """Test finding creation."""
        finding = Finding(
            id="finding_123",
            scan_id="scan_456",
            scan_type=ScanType.SAST,
            severity=FindingSeverity.HIGH,
            title="SQL Injection",
            description="User input in SQL query",
            file_path="src/db.py",
            line_number=42,
        )

        assert finding.id == "finding_123"
        assert finding.scan_type == ScanType.SAST
        assert finding.severity == FindingSeverity.HIGH
        assert finding.status == FindingStatus.OPEN

    def test_finding_to_dict(self):
        """Test finding serialization."""
        finding = Finding(
            id="finding_123",
            scan_id="scan_456",
            scan_type=ScanType.SAST,
            severity=FindingSeverity.HIGH,
            title="SQL Injection",
            description="User input in SQL query",
            file_path="src/db.py",
            line_number=42,
            cwe_id="CWE-89",
            owasp_category="A03:2021",
        )

        data = finding.to_dict()

        assert data["id"] == "finding_123"
        assert data["scan_type"] == "sast"
        assert data["severity"] == "high"
        assert data["cwe_id"] == "CWE-89"
        assert data["status"] == "open"


class TestScanResult:
    """Tests for ScanResult dataclass."""

    def test_scan_result_creation(self):
        """Test scan result creation."""
        result = ScanResult(
            id="scan_123",
            tenant_id="tenant_456",
            scan_type=ScanType.COMPREHENSIVE,
            status=ScanStatus.COMPLETED,
            target_path="/code",
            started_at=datetime.now(timezone.utc),
            files_scanned=100,
            findings_count=5,
        )

        assert result.id == "scan_123"
        assert result.scan_type == ScanType.COMPREHENSIVE
        assert result.status == ScanStatus.COMPLETED

    def test_scan_result_to_dict(self):
        """Test scan result serialization."""
        now = datetime.now(timezone.utc)
        result = ScanResult(
            id="scan_123",
            tenant_id="tenant_456",
            scan_type=ScanType.SAST,
            status=ScanStatus.COMPLETED,
            target_path="/code",
            started_at=now,
            completed_at=now,
            files_scanned=100,
        )

        data = result.to_dict()

        assert data["id"] == "scan_123"
        assert data["scan_type"] == "sast"
        assert data["status"] == "completed"


class TestCodebaseAuditHandler:
    """Tests for CodebaseAuditHandler."""

    def test_handler_routes(self):
        """Test handler has expected routes."""
        handler = CodebaseAuditHandler()

        expected_routes = [
            "/api/v1/codebase/scan",
            "/api/v1/codebase/scans",
            "/api/v1/codebase/sast",
            "/api/v1/codebase/bugs",
            "/api/v1/codebase/secrets",
            "/api/v1/codebase/dependencies",
            "/api/v1/codebase/metrics",
            "/api/v1/codebase/findings",
            "/api/v1/codebase/dashboard",
        ]

        for route in expected_routes:
            assert any(route in r for r in handler.ROUTES), f"Missing route: {route}"

    def test_get_handler_instance(self):
        """Test getting handler instance."""
        handler1 = get_codebase_audit_handler()
        handler2 = get_codebase_audit_handler()

        assert handler1 is handler2


@pytest.fixture
def mock_scanners():
    """Mock scanner functions to return mock data (avoids resource-heavy real scans)."""
    handler_module = "aragora.server.handlers.features.codebase_audit"

    async def mock_sast(*args, **kwargs):
        return _get_mock_sast_findings(args[1] if len(args) > 1 else "test_scan")

    async def mock_bugs(*args, **kwargs):
        return _get_mock_bug_findings(args[1] if len(args) > 1 else "test_scan")

    async def mock_secrets(*args, **kwargs):
        return _get_mock_secrets_findings(args[1] if len(args) > 1 else "test_scan")

    async def mock_deps(*args, **kwargs):
        return _get_mock_dependency_findings(args[1] if len(args) > 1 else "test_scan")

    async def mock_metrics(*args, **kwargs):
        return _get_mock_metrics()

    with (
        patch(f"{handler_module}.run_sast_scan", side_effect=mock_sast),
        patch(f"{handler_module}.run_bug_scan", side_effect=mock_bugs),
        patch(f"{handler_module}.run_secrets_scan", side_effect=mock_secrets),
        patch(f"{handler_module}.run_dependency_scan", side_effect=mock_deps),
        patch(f"{handler_module}.run_metrics_analysis", side_effect=mock_metrics),
    ):
        yield


class TestComprehensiveScan:
    """Tests for comprehensive scan."""

    @pytest.mark.asyncio
    async def test_comprehensive_scan_default_path(self, mock_scanners):
        """Test comprehensive scan with default path."""
        handler = CodebaseAuditHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(
            return_value={
                "scan_types": ["sast", "bugs"],
            }
        )

        result = await handler.handle(request, "/api/v1/codebase/scan", "POST")

        assert result is not None
        assert result.status_code == 200
        assert b"scan" in result.body
        assert b"findings" in result.body

    @pytest.mark.asyncio
    async def test_comprehensive_scan_with_path(self, mock_scanners):
        """Test comprehensive scan with custom path."""
        handler = CodebaseAuditHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(
            return_value={
                "target_path": "/custom/path",
                "scan_types": ["sast"],
                "languages": ["python"],
            }
        )

        result = await handler.handle(request, "/api/v1/codebase/scan", "POST")

        assert result is not None
        assert result.status_code == 200


class TestIndividualScans:
    """Tests for individual scan types."""

    @pytest.mark.asyncio
    async def test_sast_scan(self, mock_scanners):
        """Test SAST-only scan."""
        handler = CodebaseAuditHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(return_value={"target_path": "."})

        result = await handler.handle(request, "/api/v1/codebase/sast", "POST")

        assert result is not None
        assert result.status_code == 200
        assert b"findings" in result.body

    @pytest.mark.asyncio
    async def test_bug_scan(self, mock_scanners):
        """Test bug detection scan."""
        handler = CodebaseAuditHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(return_value={"target_path": "."})

        result = await handler.handle(request, "/api/v1/codebase/bugs", "POST")

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_secrets_scan(self, mock_scanners):
        """Test secrets scan."""
        handler = CodebaseAuditHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(return_value={"target_path": "."})

        result = await handler.handle(request, "/api/v1/codebase/secrets", "POST")

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_dependency_scan(self, mock_scanners):
        """Test dependency scan."""
        handler = CodebaseAuditHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(return_value={"target_path": "."})

        result = await handler.handle(request, "/api/v1/codebase/dependencies", "POST")

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_metrics_analysis(self, mock_scanners):
        """Test metrics analysis."""
        handler = CodebaseAuditHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(return_value={"target_path": "."})

        result = await handler.handle(request, "/api/v1/codebase/metrics", "POST")

        assert result is not None
        assert result.status_code == 200
        assert b"metrics" in result.body


class TestListScans:
    """Tests for listing scans."""

    @pytest.mark.asyncio
    async def test_list_scans_empty(self):
        """Test listing scans when empty."""
        handler = CodebaseAuditHandler()

        request = MagicMock()
        request.tenant_id = "new_tenant"
        request.query = {}

        result = await handler.handle(request, "/api/v1/codebase/scans", "GET")

        assert result is not None
        assert result.status_code == 200
        assert b"scans" in result.body

    @pytest.mark.asyncio
    async def test_list_scans_with_filters(self):
        """Test listing scans with filters."""
        handler = CodebaseAuditHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.query = {"type": "sast", "status": "completed", "limit": "10"}

        result = await handler.handle(request, "/api/v1/codebase/scans", "GET")

        assert result is not None
        assert result.status_code == 200


class TestGetScan:
    """Tests for getting a specific scan."""

    @pytest.mark.asyncio
    async def test_get_scan_not_found(self):
        """Test get scan returns 404 when not found."""
        handler = CodebaseAuditHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"

        result = await handler.handle(request, "/api/v1/codebase/scan/nonexistent", "GET")

        assert result is not None
        assert result.status_code == 404


class TestListFindings:
    """Tests for listing findings."""

    @pytest.mark.asyncio
    async def test_list_findings_empty(self):
        """Test listing findings when empty."""
        handler = CodebaseAuditHandler()

        request = MagicMock()
        request.tenant_id = "empty_tenant"
        request.query = {}

        result = await handler.handle(request, "/api/v1/codebase/findings", "GET")

        assert result is not None
        assert result.status_code == 200
        assert b"findings" in result.body

    @pytest.mark.asyncio
    async def test_list_findings_with_filters(self):
        """Test listing findings with filters."""
        handler = CodebaseAuditHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.query = {
            "severity": "high",
            "type": "sast",
            "status": "open",
            "limit": "20",
        }

        result = await handler.handle(request, "/api/v1/codebase/findings", "GET")

        assert result is not None
        assert result.status_code == 200


class TestDismissFinding:
    """Tests for dismissing findings."""

    @pytest.mark.asyncio
    async def test_dismiss_finding_not_found(self):
        """Test dismissing non-existent finding."""
        handler = CodebaseAuditHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.user_id = "user_123"
        request.json = AsyncMock(
            return_value={
                "reason": "False positive",
                "status": "false_positive",
            }
        )

        result = await handler.handle(
            request, "/api/v1/codebase/findings/nonexistent/dismiss", "POST"
        )

        assert result is not None
        assert result.status_code == 404


class TestCreateIssue:
    """Tests for creating GitHub issues."""

    @pytest.mark.asyncio
    async def test_create_issue_not_found(self):
        """Test creating issue for non-existent finding."""
        handler = CodebaseAuditHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(return_value={"repo": "org/repo"})

        result = await handler.handle(
            request, "/api/v1/codebase/findings/nonexistent/create-issue", "POST"
        )

        assert result is not None
        assert result.status_code == 404


class TestDashboard:
    """Tests for dashboard endpoint."""

    @pytest.mark.asyncio
    async def test_dashboard_empty(self):
        """Test dashboard with no data."""
        handler = CodebaseAuditHandler()

        request = MagicMock()
        request.tenant_id = "dashboard_tenant"

        result = await handler.handle(request, "/api/v1/codebase/dashboard", "GET")

        assert result is not None
        assert result.status_code == 200
        assert b"summary" in result.body
        assert b"total_findings" in result.body


class TestDemoEndpoint:
    """Tests for demo endpoint."""

    @pytest.mark.asyncio
    async def test_demo_endpoint(self):
        """Test demo endpoint returns mock data."""
        handler = CodebaseAuditHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"

        result = await handler.handle(request, "/api/v1/codebase/demo", "GET")

        assert result is not None
        assert result.status_code == 200
        assert b"is_demo" in result.body
        assert b"summary" in result.body
        assert b"findings" in result.body


class TestMockData:
    """Tests for mock data generators."""

    def test_mock_sast_findings(self):
        """Test mock SAST findings generation."""
        findings = _get_mock_sast_findings("test_scan")

        assert len(findings) > 0
        assert all(f.scan_type == ScanType.SAST for f in findings)
        assert any(f.severity == FindingSeverity.CRITICAL for f in findings)

    def test_mock_bug_findings(self):
        """Test mock bug findings generation."""
        findings = _get_mock_bug_findings("test_scan")

        assert len(findings) > 0
        assert all(f.scan_type == ScanType.BUGS for f in findings)

    def test_mock_secrets_findings(self):
        """Test mock secrets findings generation."""
        findings = _get_mock_secrets_findings("test_scan")

        assert len(findings) > 0
        assert all(f.scan_type == ScanType.SECRETS for f in findings)
        assert all(f.severity == FindingSeverity.CRITICAL for f in findings)

    def test_mock_dependency_findings(self):
        """Test mock dependency findings generation."""
        findings = _get_mock_dependency_findings("test_scan")

        assert len(findings) > 0
        assert all(f.scan_type == ScanType.DEPENDENCIES for f in findings)

    def test_mock_metrics(self):
        """Test mock metrics generation."""
        metrics = _get_mock_metrics()

        assert "total_lines" in metrics
        assert "average_complexity" in metrics
        assert "maintainability_index" in metrics
        assert "hotspots" in metrics


class TestHandleCodebaseAudit:
    """Tests for handle_codebase_audit entry point."""

    @pytest.mark.asyncio
    async def test_entry_point(self):
        """Test entry point function."""
        request = MagicMock()
        request.tenant_id = "test"
        request.query = {}

        result = await handle_codebase_audit(request, "/api/v1/codebase/scans", "GET")

        assert result is not None


class TestNotFoundRoute:
    """Tests for not found route."""

    @pytest.mark.asyncio
    async def test_unknown_route(self):
        """Test handling unknown route."""
        handler = CodebaseAuditHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"

        result = await handler.handle(request, "/api/v1/codebase/unknown/path", "GET")

        assert result is not None
        assert result.status_code == 404


class TestImports:
    """Test that imports work correctly."""

    def test_import_from_package(self):
        """Test imports from features package."""
        from aragora.server.handlers.features import (
            CodebaseAuditHandler,
            handle_codebase_audit,
            get_codebase_audit_handler,
            ScanType,
            ScanStatus,
            FindingSeverity,
            FindingStatus,
            Finding,
            ScanResult,
        )

        assert CodebaseAuditHandler is not None
        assert handle_codebase_audit is not None
        assert ScanType is not None
        assert Finding is not None
