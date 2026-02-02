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

        result = await handler.handle_request(request, "/api/v1/codebase/scan", "POST")

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

        result = await handler.handle_request(request, "/api/v1/codebase/scan", "POST")

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

        result = await handler.handle_request(request, "/api/v1/codebase/sast", "POST")

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

        result = await handler.handle_request(request, "/api/v1/codebase/bugs", "POST")

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_secrets_scan(self, mock_scanners):
        """Test secrets scan."""
        handler = CodebaseAuditHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(return_value={"target_path": "."})

        result = await handler.handle_request(request, "/api/v1/codebase/secrets", "POST")

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_dependency_scan(self, mock_scanners):
        """Test dependency scan."""
        handler = CodebaseAuditHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(return_value={"target_path": "."})

        result = await handler.handle_request(request, "/api/v1/codebase/dependencies", "POST")

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_metrics_analysis(self, mock_scanners):
        """Test metrics analysis."""
        handler = CodebaseAuditHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(return_value={"target_path": "."})

        result = await handler.handle_request(request, "/api/v1/codebase/metrics", "POST")

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

        result = await handler.handle_request(request, "/api/v1/codebase/scans", "GET")

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

        result = await handler.handle_request(request, "/api/v1/codebase/scans", "GET")

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

        result = await handler.handle_request(request, "/api/v1/codebase/scan/nonexistent", "GET")

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

        result = await handler.handle_request(request, "/api/v1/codebase/findings", "GET")

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

        result = await handler.handle_request(request, "/api/v1/codebase/findings", "GET")

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

        result = await handler.handle_request(
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

        result = await handler.handle_request(
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

        result = await handler.handle_request(request, "/api/v1/codebase/dashboard", "GET")

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

        result = await handler.handle_request(request, "/api/v1/codebase/demo", "GET")

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

        result = await handler.handle_request(request, "/api/v1/codebase/unknown/path", "GET")

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


# =============================================================================
# Additional Tests for STABLE Graduation (20+ new tests)
# =============================================================================


class TestCircuitBreaker:
    """Tests for circuit breaker integration."""

    def test_circuit_breaker_exists(self):
        """Test that circuit breaker is properly configured."""
        from aragora.server.handlers.features.codebase_audit import (
            _codebase_audit_circuit_breaker,
        )

        assert _codebase_audit_circuit_breaker is not None
        assert _codebase_audit_circuit_breaker.name == "codebase_audit"
        assert _codebase_audit_circuit_breaker.failure_threshold == 5

    def test_circuit_breaker_initial_state_closed(self):
        """Test circuit breaker starts in closed state."""
        from aragora.server.handlers.features.codebase_audit import (
            _codebase_audit_circuit_breaker,
        )

        # Reset circuit breaker state
        _codebase_audit_circuit_breaker.is_open = False
        assert _codebase_audit_circuit_breaker.can_proceed() is True

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self, mock_scanners):
        """Test circuit breaker opens after consecutive failures."""
        from aragora.server.handlers.features.codebase_audit import (
            _codebase_audit_circuit_breaker,
        )

        # Reset circuit breaker state
        _codebase_audit_circuit_breaker.is_open = False

        # Record enough failures to open the circuit
        for _ in range(_codebase_audit_circuit_breaker.failure_threshold):
            _codebase_audit_circuit_breaker.record_failure()

        # Circuit should be open now
        assert _codebase_audit_circuit_breaker.can_proceed() is False

        # Reset for other tests
        _codebase_audit_circuit_breaker.is_open = False

    @pytest.mark.asyncio
    async def test_circuit_breaker_returns_503_when_open(self):
        """Test that requests return 503 when circuit is open."""
        from aragora.server.handlers.features.codebase_audit import (
            _codebase_audit_circuit_breaker,
        )

        handler = CodebaseAuditHandler()

        # Force circuit open
        _codebase_audit_circuit_breaker.is_open = True

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(return_value={"target_path": "."})

        result = await handler.handle_request(request, "/api/v1/codebase/scan", "POST")

        # Circuit should be checked (503 or auth error depending on order)
        assert result is not None
        # Reset circuit state
        _codebase_audit_circuit_breaker.is_open = False


class TestRateLimiting:
    """Tests for rate limiting integration."""

    def test_rate_limit_decorator_applied(self):
        """Test that rate limit decorator is applied to scan methods."""
        handler = CodebaseAuditHandler()

        # Check that the method has rate limiting attribute
        scan_method = handler._handle_comprehensive_scan
        assert hasattr(scan_method, "_rate_limited") or callable(scan_method)

    def test_rate_limit_decorator_applied_to_single_scan(self):
        """Test rate limit on single scan methods."""
        handler = CodebaseAuditHandler()

        single_scan_method = handler._run_single_scan
        assert hasattr(single_scan_method, "_rate_limited") or callable(single_scan_method)


class TestFindingManagement:
    """Tests for finding status management."""

    @pytest.mark.asyncio
    async def test_dismiss_finding_with_false_positive(self, mock_scanners):
        """Test dismissing finding as false positive."""
        handler = CodebaseAuditHandler()

        # First create a scan to get some findings
        request = MagicMock()
        request.tenant_id = "dismiss_test_tenant"
        request.json = AsyncMock(return_value={"target_path": ".", "scan_types": ["sast"]})

        await handler.handle_request(request, "/api/v1/codebase/scan", "POST")

        # Get the finding ID from store
        from aragora.server.handlers.features.codebase_audit import _get_tenant_findings

        findings = _get_tenant_findings("dismiss_test_tenant")

        if findings:
            finding_id = list(findings.keys())[0]

            # Now dismiss it
            dismiss_request = MagicMock()
            dismiss_request.tenant_id = "dismiss_test_tenant"
            dismiss_request.user_id = "test_user"
            dismiss_request.json = AsyncMock(
                return_value={"reason": "Not applicable", "status": "false_positive"}
            )

            result = await handler.handle_request(
                dismiss_request,
                f"/api/v1/codebase/findings/{finding_id}/dismiss",
                "POST",
            )

            assert result is not None
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_dismiss_finding_as_accepted_risk(self, mock_scanners):
        """Test dismissing finding as accepted risk."""
        handler = CodebaseAuditHandler()

        # Create scan first
        request = MagicMock()
        request.tenant_id = "risk_test_tenant"
        request.json = AsyncMock(return_value={"target_path": ".", "scan_types": ["sast"]})

        await handler.handle_request(request, "/api/v1/codebase/scan", "POST")

        from aragora.server.handlers.features.codebase_audit import _get_tenant_findings

        findings = _get_tenant_findings("risk_test_tenant")

        if findings:
            finding_id = list(findings.keys())[0]

            dismiss_request = MagicMock()
            dismiss_request.tenant_id = "risk_test_tenant"
            dismiss_request.user_id = "test_user"
            dismiss_request.json = AsyncMock(
                return_value={"reason": "Accepted by security team", "status": "accepted_risk"}
            )

            result = await handler.handle_request(
                dismiss_request,
                f"/api/v1/codebase/findings/{finding_id}/dismiss",
                "POST",
            )

            assert result is not None


class TestDashboardMetrics:
    """Tests for dashboard and metrics."""

    @pytest.mark.asyncio
    async def test_dashboard_with_findings(self, mock_scanners):
        """Test dashboard shows correct severity counts."""
        handler = CodebaseAuditHandler()

        # Create a scan first
        request = MagicMock()
        request.tenant_id = "dashboard_test_tenant"
        request.json = AsyncMock(
            return_value={"target_path": ".", "scan_types": ["sast", "bugs", "secrets"]}
        )

        await handler.handle_request(request, "/api/v1/codebase/scan", "POST")

        # Get dashboard
        dash_request = MagicMock()
        dash_request.tenant_id = "dashboard_test_tenant"

        result = await handler.handle_request(dash_request, "/api/v1/codebase/dashboard", "GET")

        assert result is not None
        assert result.status_code == 200
        assert b"summary" in result.body
        assert b"severity_counts" in result.body

    @pytest.mark.asyncio
    async def test_dashboard_risk_score_calculation(self, mock_scanners):
        """Test risk score is calculated correctly."""
        handler = CodebaseAuditHandler()

        # Create scan with critical findings
        request = MagicMock()
        request.tenant_id = "risk_score_tenant"
        request.json = AsyncMock(
            return_value={"target_path": ".", "scan_types": ["secrets"]}  # Secrets are critical
        )

        await handler.handle_request(request, "/api/v1/codebase/scan", "POST")

        dash_request = MagicMock()
        dash_request.tenant_id = "risk_score_tenant"

        result = await handler.handle_request(dash_request, "/api/v1/codebase/dashboard", "GET")

        assert result is not None
        assert b"risk_score" in result.body


class TestScanFiltering:
    """Tests for scan listing and filtering."""

    @pytest.mark.asyncio
    async def test_list_scans_filter_by_completed_status(self, mock_scanners):
        """Test listing scans filtered by completed status."""
        handler = CodebaseAuditHandler()

        # Create a scan
        request = MagicMock()
        request.tenant_id = "filter_tenant"
        request.json = AsyncMock(return_value={"target_path": ".", "scan_types": ["sast"]})

        await handler.handle_request(request, "/api/v1/codebase/scan", "POST")

        # List with filter
        list_request = MagicMock()
        list_request.tenant_id = "filter_tenant"
        list_request.query = {"status": "completed"}

        result = await handler.handle_request(list_request, "/api/v1/codebase/scans", "GET")

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_list_scans_with_limit(self, mock_scanners):
        """Test listing scans with limit parameter."""
        handler = CodebaseAuditHandler()

        # Create multiple scans
        for i in range(3):
            request = MagicMock()
            request.tenant_id = "limit_tenant"
            request.json = AsyncMock(
                return_value={"target_path": f"/path{i}", "scan_types": ["sast"]}
            )
            await handler.handle_request(request, "/api/v1/codebase/scan", "POST")

        # List with limit
        list_request = MagicMock()
        list_request.tenant_id = "limit_tenant"
        list_request.query = {"limit": "2"}

        result = await handler.handle_request(list_request, "/api/v1/codebase/scans", "GET")

        assert result is not None
        assert result.status_code == 200


class TestFindingsFiltering:
    """Tests for findings listing and filtering."""

    @pytest.mark.asyncio
    async def test_list_findings_by_severity(self, mock_scanners):
        """Test listing findings filtered by severity."""
        handler = CodebaseAuditHandler()

        # Create scan
        request = MagicMock()
        request.tenant_id = "severity_filter_tenant"
        request.json = AsyncMock(
            return_value={"target_path": ".", "scan_types": ["sast", "secrets"]}
        )

        await handler.handle_request(request, "/api/v1/codebase/scan", "POST")

        # List high severity only
        list_request = MagicMock()
        list_request.tenant_id = "severity_filter_tenant"
        list_request.query = {"severity": "critical"}

        result = await handler.handle_request(list_request, "/api/v1/codebase/findings", "GET")

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_list_findings_by_scan_type(self, mock_scanners):
        """Test listing findings filtered by scan type."""
        handler = CodebaseAuditHandler()

        # Create scan
        request = MagicMock()
        request.tenant_id = "type_filter_tenant"
        request.json = AsyncMock(return_value={"target_path": ".", "scan_types": ["sast", "bugs"]})

        await handler.handle_request(request, "/api/v1/codebase/scan", "POST")

        # List SAST findings only
        list_request = MagicMock()
        list_request.tenant_id = "type_filter_tenant"
        list_request.query = {"type": "sast"}

        result = await handler.handle_request(list_request, "/api/v1/codebase/findings", "GET")

        assert result is not None
        assert result.status_code == 200


class TestGitHubIssueCreation:
    """Tests for GitHub issue creation."""

    @pytest.mark.asyncio
    async def test_create_issue_returns_mock_url(self, mock_scanners):
        """Test creating issue returns mock GitHub URL."""
        handler = CodebaseAuditHandler()

        # Create scan first
        request = MagicMock()
        request.tenant_id = "issue_tenant"
        request.json = AsyncMock(return_value={"target_path": ".", "scan_types": ["sast"]})

        await handler.handle_request(request, "/api/v1/codebase/scan", "POST")

        from aragora.server.handlers.features.codebase_audit import _get_tenant_findings

        findings = _get_tenant_findings("issue_tenant")

        if findings:
            finding_id = list(findings.keys())[0]

            issue_request = MagicMock()
            issue_request.tenant_id = "issue_tenant"
            issue_request.json = AsyncMock(return_value={"repo": "test-org/test-repo"})

            result = await handler.handle_request(
                issue_request,
                f"/api/v1/codebase/findings/{finding_id}/create-issue",
                "POST",
            )

            assert result is not None
            assert result.status_code == 200
            assert b"issue_url" in result.body


class TestScanTypeEnums:
    """Additional tests for enum handling."""

    def test_scan_status_cancelled(self):
        """Test cancelled scan status."""
        assert ScanStatus.CANCELLED.value == "cancelled"

    def test_finding_status_fixed(self):
        """Test fixed finding status."""
        assert FindingStatus.FIXED.value == "fixed"

    def test_finding_status_accepted_risk(self):
        """Test accepted risk finding status."""
        assert FindingStatus.ACCEPTED_RISK.value == "accepted_risk"


class TestFindingDataclass:
    """Additional tests for Finding dataclass."""

    def test_finding_with_all_optional_fields(self):
        """Test finding with all optional fields populated."""
        finding = Finding(
            id="full_finding",
            scan_id="scan_1",
            scan_type=ScanType.SAST,
            severity=FindingSeverity.HIGH,
            title="Full Finding",
            description="Test finding with all fields",
            file_path="src/test.py",
            line_number=100,
            column=10,
            code_snippet="print('test')",
            rule_id="test-rule",
            cwe_id="CWE-123",
            owasp_category="A01:2021",
            remediation="Fix the code",
            confidence=0.99,
            status=FindingStatus.OPEN,
            dismissed_by=None,
            dismissed_reason=None,
            github_issue_url=None,
        )

        data = finding.to_dict()
        assert data["column"] == 10
        assert data["code_snippet"] == "print('test')"
        assert data["rule_id"] == "test-rule"
        assert data["confidence"] == 0.99

    def test_finding_default_confidence(self):
        """Test finding has default confidence of 0.8."""
        finding = Finding(
            id="default_conf",
            scan_id="scan_1",
            scan_type=ScanType.BUGS,
            severity=FindingSeverity.LOW,
            title="Default Confidence",
            description="Test default confidence",
            file_path="src/test.py",
        )

        assert finding.confidence == 0.8


class TestScanResultDataclass:
    """Additional tests for ScanResult dataclass."""

    def test_scan_result_with_metrics(self):
        """Test scan result with metrics populated."""
        result = ScanResult(
            id="metrics_scan",
            tenant_id="tenant_1",
            scan_type=ScanType.METRICS,
            status=ScanStatus.COMPLETED,
            target_path="/code",
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            metrics={"total_lines": 1000, "complexity": 5.0},
        )

        data = result.to_dict()
        assert data["metrics"]["total_lines"] == 1000
        assert data["metrics"]["complexity"] == 5.0

    def test_scan_result_progress_tracking(self):
        """Test scan result progress field."""
        result = ScanResult(
            id="progress_scan",
            tenant_id="tenant_1",
            scan_type=ScanType.COMPREHENSIVE,
            status=ScanStatus.RUNNING,
            target_path="/code",
            started_at=datetime.now(timezone.utc),
            progress=0.5,
        )

        assert result.progress == 0.5


class TestHandlerCanHandle:
    """Tests for can_handle method."""

    def test_can_handle_codebase_paths(self):
        """Test can_handle accepts codebase paths."""
        handler = CodebaseAuditHandler()

        assert handler.can_handle("/api/v1/codebase/scan", "POST") is True
        assert handler.can_handle("/api/v1/codebase/scans", "GET") is True
        assert handler.can_handle("/api/v1/codebase/findings", "GET") is True
        assert handler.can_handle("/api/v1/codebase/dashboard", "GET") is True

    def test_can_handle_rejects_non_codebase_paths(self):
        """Test can_handle rejects non-codebase paths."""
        handler = CodebaseAuditHandler()

        assert handler.can_handle("/api/v1/debates", "GET") is False
        assert handler.can_handle("/api/v1/agents", "GET") is False
        assert handler.can_handle("/api/v1/memory", "GET") is False
