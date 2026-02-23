"""Comprehensive tests for the Codebase Audit API handler.

Covers all routes and behaviour of CodebaseAuditHandler:
- can_handle() routing
- POST /api/v1/codebase/scan          (comprehensive scan)
- GET  /api/v1/codebase/scan/{id}     (get scan by id)
- GET  /api/v1/codebase/scans         (list scans)
- POST /api/v1/codebase/sast          (SAST scan)
- POST /api/v1/codebase/bugs          (bug detection)
- POST /api/v1/codebase/secrets       (secrets scan)
- POST /api/v1/codebase/dependencies  (dependency scan)
- POST /api/v1/codebase/metrics       (metrics analysis)
- GET  /api/v1/codebase/findings      (list findings)
- POST /api/v1/codebase/findings/{id}/dismiss     (dismiss finding)
- POST /api/v1/codebase/findings/{id}/create-issue (create GitHub issue)
- GET  /api/v1/codebase/dashboard     (dashboard data)
- GET  /api/v1/codebase/demo          (demo data)
- RBAC / auth enforcement
- Circuit breaker integration
- Validation of inputs (paths, scan types, severities, etc.)
- 404 fallback for unknown routes / methods
- Error handling
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.codebase_audit.handler import (
    CodebaseAuditHandler,
    _codebase_audit_circuit_breaker,
    get_codebase_audit_handler,
    handle_codebase_audit,
)
from aragora.server.handlers.features.codebase_audit.rules import (
    Finding,
    FindingSeverity,
    FindingStatus,
    ScanResult,
    ScanStatus,
    ScanType,
)
from aragora.server.handlers.features.codebase_audit.scanning import (
    _finding_store,
    _get_tenant_findings,
    _get_tenant_scans,
    _scan_store,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Mock Request
# ---------------------------------------------------------------------------


@dataclass
class MockRequest:
    """Mock async HTTP request for CodebaseAuditHandler."""

    method: str = "GET"
    path: str = "/"
    query: dict[str, str] = field(default_factory=dict)
    _body: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    tenant_id: str = "test-tenant"
    user_id: str = "test-user"

    async def json(self) -> dict[str, Any]:
        return self._body or {}

    async def body(self) -> bytes:
        return json.dumps(self._body or {}).encode()


def _make_request(
    method: str = "GET",
    path: str = "/",
    query: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    tenant_id: str = "test-tenant",
) -> MockRequest:
    return MockRequest(
        method=method,
        path=path,
        query=query or {},
        _body=body or {},
        tenant_id=tenant_id,
    )


# ---------------------------------------------------------------------------
# Helpers to seed store data
# ---------------------------------------------------------------------------


def _seed_finding(
    tenant_id: str = "test-tenant",
    finding_id: str = "find_abc123",
    severity: FindingSeverity = FindingSeverity.HIGH,
    scan_type: ScanType = ScanType.SAST,
    status: FindingStatus = FindingStatus.OPEN,
    title: str = "Test Finding",
) -> Finding:
    """Create and store a finding in the in-memory store."""
    finding = Finding(
        id=finding_id,
        scan_id="scan_test001",
        scan_type=scan_type,
        severity=severity,
        title=title,
        description="A test finding",
        file_path="src/test.py",
        line_number=10,
        status=status,
    )
    store = _get_tenant_findings(tenant_id)
    store[finding_id] = finding
    return finding


def _seed_scan(
    tenant_id: str = "test-tenant",
    scan_id: str = "scan_test001",
    scan_type: ScanType = ScanType.SAST,
    status: ScanStatus = ScanStatus.COMPLETED,
) -> ScanResult:
    """Create and store a scan result in the in-memory store."""
    scan = ScanResult(
        id=scan_id,
        tenant_id=tenant_id,
        scan_type=scan_type,
        status=status,
        target_path=".",
        started_at=datetime.now(timezone.utc),
    )
    store = _get_tenant_scans(tenant_id)
    store[scan_id] = scan
    return scan


# ---------------------------------------------------------------------------
# Mock scanner functions
# ---------------------------------------------------------------------------


def _mock_findings(scan_id: str = "scan_test") -> list[Finding]:
    """Return a small list of mock findings for testing."""
    return [
        Finding(
            id="sast_mock1",
            scan_id=scan_id,
            scan_type=ScanType.SAST,
            severity=FindingSeverity.HIGH,
            title="Mock SQL Injection",
            description="Mock finding",
            file_path="src/db.py",
            line_number=42,
        ),
        Finding(
            id="sast_mock2",
            scan_id=scan_id,
            scan_type=ScanType.SAST,
            severity=FindingSeverity.MEDIUM,
            title="Mock XSS",
            description="Mock finding",
            file_path="src/ui.py",
            line_number=10,
        ),
    ]


def _mock_metrics_result() -> dict[str, Any]:
    return {
        "total_lines": 1000,
        "code_lines": 800,
        "comment_lines": 100,
        "blank_lines": 100,
        "files_analyzed": 10,
        "average_complexity": 3.5,
        "max_complexity": 15,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a CodebaseAuditHandler instance."""
    return CodebaseAuditHandler(server_context={})


@pytest.fixture(autouse=True)
def _clear_stores():
    """Clear in-memory stores between tests."""
    _scan_store.clear()
    _finding_store.clear()
    yield
    _scan_store.clear()
    _finding_store.clear()


@pytest.fixture(autouse=True)
def _reset_circuit_breaker():
    """Reset the circuit breaker between tests."""
    _codebase_audit_circuit_breaker.reset()
    yield
    _codebase_audit_circuit_breaker.reset()


@pytest.fixture
def _open_circuit_breaker():
    """Open the circuit breaker by recording enough failures."""
    for _ in range(10):
        _codebase_audit_circuit_breaker.record_failure()


# ---------------------------------------------------------------------------
# Helper to patch scanner functions on the package namespace
# ---------------------------------------------------------------------------

_SCAN_MODULE = "aragora.server.handlers.features.codebase_audit"


def _patch_scanner(name: str, return_value: Any = None):
    """Return a patch context manager for a scanner function."""
    if return_value is None:
        return_value = []
    return patch(
        f"{_SCAN_MODULE}.{name}",
        new_callable=AsyncMock,
        return_value=return_value,
    )


# ===========================================================================
# 1. can_handle
# ===========================================================================


class TestCanHandle:
    """Tests for CodebaseAuditHandler.can_handle()."""

    def test_handles_codebase_prefix(self, handler):
        assert handler.can_handle("/api/v1/codebase", "GET") is True

    def test_handles_scan_path(self, handler):
        assert handler.can_handle("/api/v1/codebase/scan", "POST") is True

    def test_handles_findings_path(self, handler):
        assert handler.can_handle("/api/v1/codebase/findings", "GET") is True

    def test_handles_dashboard_path(self, handler):
        assert handler.can_handle("/api/v1/codebase/dashboard", "GET") is True

    def test_rejects_non_codebase_path(self, handler):
        assert handler.can_handle("/api/v1/debates", "GET") is False

    def test_rejects_root_path(self, handler):
        assert handler.can_handle("/api/v1/code", "GET") is False

    def test_handles_nested_codebase_path(self, handler):
        assert handler.can_handle("/api/v1/codebase/scan/some_id", "GET") is True


# ===========================================================================
# 2. 404 Fallback
# ===========================================================================


class TestNotFound:
    """Tests for unknown route / method combinations returning 404."""

    @pytest.mark.asyncio
    async def test_unknown_path_returns_404(self, handler):
        req = _make_request("GET", "/api/v1/codebase/nonexistent")
        result = await handler.handle(req, "/api/v1/codebase/nonexistent", "GET")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_scan_post_with_get_returns_404(self, handler):
        req = _make_request("GET", "/api/v1/codebase/scan")
        result = await handler.handle(req, "/api/v1/codebase/scan", "GET")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_findings_post_without_action_returns_404(self, handler):
        req = _make_request("POST", "/api/v1/codebase/findings")
        result = await handler.handle(req, "/api/v1/codebase/findings", "POST")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_scans_post_returns_404(self, handler):
        req = _make_request("POST", "/api/v1/codebase/scans")
        result = await handler.handle(req, "/api/v1/codebase/scans", "POST")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_dashboard_post_returns_404(self, handler):
        req = _make_request("POST", "/api/v1/codebase/dashboard")
        result = await handler.handle(req, "/api/v1/codebase/dashboard", "POST")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_scan_id_path_with_post_returns_404(self, handler):
        req = _make_request("POST", "/api/v1/codebase/scan/some_id")
        result = await handler.handle(req, "/api/v1/codebase/scan/some_id", "POST")
        assert _status(result) == 404


# ===========================================================================
# 3. Comprehensive Scan (POST /api/v1/codebase/scan)
# ===========================================================================


class TestComprehensiveScan:
    """Tests for POST /api/v1/codebase/scan."""

    PATH = "/api/v1/codebase/scan"

    @pytest.mark.asyncio
    async def test_basic_scan_success(self, handler):
        mock_findings = _mock_findings()
        with (
            _patch_scanner("run_sast_scan", mock_findings),
            _patch_scanner("run_bug_scan", []),
            _patch_scanner("run_secrets_scan", []),
            _patch_scanner("run_dependency_scan", []),
        ):
            req = _make_request("POST", self.PATH, body={"target_path": ".", "scan_types": ["sast"]})
            result = await handler.handle(req, self.PATH, "POST")

        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["summary"]["total_findings"] == 2
        assert "sast" in body["data"]["summary"]["scan_types_run"]

    @pytest.mark.asyncio
    async def test_scan_with_all_types(self, handler):
        with (
            _patch_scanner("run_sast_scan", _mock_findings()),
            _patch_scanner("run_bug_scan", []),
            _patch_scanner("run_secrets_scan", []),
            _patch_scanner("run_dependency_scan", []),
            _patch_scanner("run_metrics_analysis", _mock_metrics_result()),
        ):
            req = _make_request(
                "POST",
                self.PATH,
                body={
                    "target_path": ".",
                    "scan_types": ["sast", "bugs", "secrets", "dependencies", "metrics"],
                },
            )
            result = await handler.handle(req, self.PATH, "POST")

        assert _status(result) == 200
        body = _body(result)
        assert len(body["data"]["summary"]["scan_types_run"]) == 5

    @pytest.mark.asyncio
    async def test_scan_default_types(self, handler):
        """When no scan_types specified, defaults to sast/bugs/secrets/dependencies."""
        with (
            _patch_scanner("run_sast_scan", []),
            _patch_scanner("run_bug_scan", []),
            _patch_scanner("run_secrets_scan", []),
            _patch_scanner("run_dependency_scan", []),
        ):
            req = _make_request("POST", self.PATH, body={"target_path": "."})
            result = await handler.handle(req, self.PATH, "POST")

        assert _status(result) == 200
        body = _body(result)
        types_run = body["data"]["summary"]["scan_types_run"]
        assert set(types_run) == {"sast", "bugs", "secrets", "dependencies"}

    @pytest.mark.asyncio
    async def test_scan_stores_findings(self, handler):
        mock_findings = _mock_findings()
        with (
            _patch_scanner("run_sast_scan", mock_findings),
            _patch_scanner("run_bug_scan", []),
            _patch_scanner("run_secrets_scan", []),
            _patch_scanner("run_dependency_scan", []),
        ):
            req = _make_request("POST", self.PATH, body={"target_path": ".", "scan_types": ["sast"]})
            await handler.handle(req, self.PATH, "POST")

        findings = _get_tenant_findings("test-tenant")
        assert len(findings) == 2

    @pytest.mark.asyncio
    async def test_scan_stores_scan_result(self, handler):
        with (
            _patch_scanner("run_sast_scan", []),
            _patch_scanner("run_bug_scan", []),
            _patch_scanner("run_secrets_scan", []),
            _patch_scanner("run_dependency_scan", []),
        ):
            req = _make_request("POST", self.PATH, body={"target_path": ".", "scan_types": ["sast"]})
            await handler.handle(req, self.PATH, "POST")

        scans = _get_tenant_scans("test-tenant")
        assert len(scans) == 1

    @pytest.mark.asyncio
    async def test_scan_severity_counts(self, handler):
        findings = [
            Finding(
                id="f1", scan_id="s", scan_type=ScanType.SAST,
                severity=FindingSeverity.CRITICAL, title="crit",
                description="d", file_path="f.py",
            ),
            Finding(
                id="f2", scan_id="s", scan_type=ScanType.SAST,
                severity=FindingSeverity.HIGH, title="high",
                description="d", file_path="f.py",
            ),
        ]
        with (
            _patch_scanner("run_sast_scan", findings),
            _patch_scanner("run_bug_scan", []),
            _patch_scanner("run_secrets_scan", []),
            _patch_scanner("run_dependency_scan", []),
        ):
            req = _make_request("POST", self.PATH, body={"target_path": ".", "scan_types": ["sast"]})
            result = await handler.handle(req, self.PATH, "POST")

        body = _body(result)
        counts = body["data"]["summary"]["severity_counts"]
        assert counts["critical"] == 1
        assert counts["high"] == 1

    @pytest.mark.asyncio
    async def test_scan_includes_metrics_when_requested(self, handler):
        metrics = _mock_metrics_result()
        with (
            _patch_scanner("run_sast_scan", []),
            _patch_scanner("run_metrics_analysis", metrics),
        ):
            req = _make_request(
                "POST", self.PATH, body={"target_path": ".", "scan_types": ["sast", "metrics"]}
            )
            result = await handler.handle(req, self.PATH, "POST")

        body = _body(result)
        assert body["data"]["metrics"]["total_lines"] == 1000

    @pytest.mark.asyncio
    async def test_scan_handles_scanner_exception(self, handler):
        """If a scanner raises, comprehensive scan still completes (gather return_exceptions)."""
        with (
            _patch_scanner("run_sast_scan") as mock_sast,
            _patch_scanner("run_bug_scan", []),
            _patch_scanner("run_secrets_scan", []),
            _patch_scanner("run_dependency_scan", []),
        ):
            mock_sast.side_effect = RuntimeError("boom")
            req = _make_request("POST", self.PATH, body={"target_path": ".", "scan_types": ["sast", "bugs"]})
            result = await handler.handle(req, self.PATH, "POST")

        # Should still succeed -- exceptions are captured by gather
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_scan_invalid_path_traversal(self, handler):
        req = _make_request("POST", self.PATH, body={"target_path": "../../../etc/passwd"})
        result = await handler.handle(req, self.PATH, "POST")
        assert _status(result) == 400
        assert "Invalid target_path" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_scan_invalid_path_null_byte(self, handler):
        req = _make_request("POST", self.PATH, body={"target_path": "src/\x00evil"})
        result = await handler.handle(req, self.PATH, "POST")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_scan_invalid_path_shell_metachar(self, handler):
        req = _make_request("POST", self.PATH, body={"target_path": "src; rm -rf /"})
        result = await handler.handle(req, self.PATH, "POST")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_scan_invalid_path_home_expansion(self, handler):
        req = _make_request("POST", self.PATH, body={"target_path": "~/secrets"})
        result = await handler.handle(req, self.PATH, "POST")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_scan_invalid_scan_types(self, handler):
        req = _make_request("POST", self.PATH, body={"target_path": ".", "scan_types": ["invalid_type"]})
        result = await handler.handle(req, self.PATH, "POST")
        assert _status(result) == 400
        assert "Invalid scan_types" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_scan_circuit_breaker_open(self, handler, _open_circuit_breaker):
        req = _make_request("POST", self.PATH, body={"target_path": "."})
        result = await handler.handle(req, self.PATH, "POST")
        assert _status(result) == 503
        assert "temporarily unavailable" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_scan_empty_body_uses_defaults(self, handler):
        with (
            _patch_scanner("run_sast_scan", []),
            _patch_scanner("run_bug_scan", []),
            _patch_scanner("run_secrets_scan", []),
            _patch_scanner("run_dependency_scan", []),
        ):
            req = _make_request("POST", self.PATH, body={})
            result = await handler.handle(req, self.PATH, "POST")

        assert _status(result) == 200


# ===========================================================================
# 4. Individual Scan Types
# ===========================================================================


class TestSASTScan:
    """Tests for POST /api/v1/codebase/sast."""

    PATH = "/api/v1/codebase/sast"

    @pytest.mark.asyncio
    async def test_sast_scan_success(self, handler):
        with _patch_scanner("run_sast_scan", _mock_findings()):
            req = _make_request("POST", self.PATH, body={"target_path": "."})
            result = await handler.handle(req, self.PATH, "POST")

        assert _status(result) == 200
        body = _body(result)
        assert len(body["data"]["findings"]) == 2

    @pytest.mark.asyncio
    async def test_sast_scan_with_languages(self, handler):
        with _patch_scanner("run_sast_scan", []) as mock_scan:
            req = _make_request(
                "POST", self.PATH, body={"target_path": ".", "languages": ["python"]}
            )
            result = await handler.handle(req, self.PATH, "POST")

        assert _status(result) == 200
        # Verify languages were passed
        mock_scan.assert_awaited_once()
        call_args = mock_scan.call_args
        assert call_args[0][3] == ["python"]  # 4th positional arg = languages

    @pytest.mark.asyncio
    async def test_sast_scan_invalid_path(self, handler):
        req = _make_request("POST", self.PATH, body={"target_path": "../etc"})
        result = await handler.handle(req, self.PATH, "POST")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_sast_scan_circuit_breaker_open(self, handler, _open_circuit_breaker):
        req = _make_request("POST", self.PATH, body={"target_path": "."})
        result = await handler.handle(req, self.PATH, "POST")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_sast_scan_stores_findings(self, handler):
        with _patch_scanner("run_sast_scan", _mock_findings()):
            req = _make_request("POST", self.PATH, body={"target_path": "."})
            await handler.handle(req, self.PATH, "POST")

        findings = _get_tenant_findings("test-tenant")
        assert len(findings) == 2


class TestBugScan:
    """Tests for POST /api/v1/codebase/bugs."""

    PATH = "/api/v1/codebase/bugs"

    @pytest.mark.asyncio
    async def test_bug_scan_success(self, handler):
        findings = [
            Finding(
                id="bug1", scan_id="s", scan_type=ScanType.BUGS,
                severity=FindingSeverity.MEDIUM, title="Bug",
                description="d", file_path="f.py",
            )
        ]
        with _patch_scanner("run_bug_scan", findings):
            req = _make_request("POST", self.PATH, body={"target_path": "."})
            result = await handler.handle(req, self.PATH, "POST")

        assert _status(result) == 200
        body = _body(result)
        assert len(body["data"]["findings"]) == 1

    @pytest.mark.asyncio
    async def test_bug_scan_invalid_path(self, handler):
        req = _make_request("POST", self.PATH, body={"target_path": "../../x"})
        result = await handler.handle(req, self.PATH, "POST")
        assert _status(result) == 400


class TestSecretsScan:
    """Tests for POST /api/v1/codebase/secrets."""

    PATH = "/api/v1/codebase/secrets"

    @pytest.mark.asyncio
    async def test_secrets_scan_success(self, handler):
        with _patch_scanner("run_secrets_scan", []):
            req = _make_request("POST", self.PATH, body={"target_path": "."})
            result = await handler.handle(req, self.PATH, "POST")

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_secrets_scan_invalid_path(self, handler):
        req = _make_request("POST", self.PATH, body={"target_path": "src|cat /etc/passwd"})
        result = await handler.handle(req, self.PATH, "POST")
        assert _status(result) == 400


class TestDependencyScan:
    """Tests for POST /api/v1/codebase/dependencies."""

    PATH = "/api/v1/codebase/dependencies"

    @pytest.mark.asyncio
    async def test_dependency_scan_success(self, handler):
        with _patch_scanner("run_dependency_scan", []):
            req = _make_request("POST", self.PATH, body={"target_path": "."})
            result = await handler.handle(req, self.PATH, "POST")

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_dependency_scan_circuit_breaker_open(self, handler, _open_circuit_breaker):
        req = _make_request("POST", self.PATH, body={"target_path": "."})
        result = await handler.handle(req, self.PATH, "POST")
        assert _status(result) == 503


class TestMetricsAnalysis:
    """Tests for POST /api/v1/codebase/metrics."""

    PATH = "/api/v1/codebase/metrics"

    @pytest.mark.asyncio
    async def test_metrics_success(self, handler):
        with _patch_scanner("run_metrics_analysis", _mock_metrics_result()):
            req = _make_request("POST", self.PATH, body={"target_path": "."})
            result = await handler.handle(req, self.PATH, "POST")

        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["metrics"]["total_lines"] == 1000

    @pytest.mark.asyncio
    async def test_metrics_invalid_path(self, handler):
        req = _make_request("POST", self.PATH, body={"target_path": "../../../etc"})
        result = await handler.handle(req, self.PATH, "POST")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_metrics_stores_scan(self, handler):
        with _patch_scanner("run_metrics_analysis", _mock_metrics_result()):
            req = _make_request("POST", self.PATH, body={"target_path": "."})
            await handler.handle(req, self.PATH, "POST")

        scans = _get_tenant_scans("test-tenant")
        assert len(scans) == 1
        scan = list(scans.values())[0]
        assert scan.scan_type == ScanType.METRICS
        assert scan.status == ScanStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_metrics_default_target_path(self, handler):
        with _patch_scanner("run_metrics_analysis", _mock_metrics_result()):
            req = _make_request("POST", self.PATH, body={})
            result = await handler.handle(req, self.PATH, "POST")

        assert _status(result) == 200


# ===========================================================================
# 5. List Scans (GET /api/v1/codebase/scans)
# ===========================================================================


class TestListScans:
    """Tests for GET /api/v1/codebase/scans."""

    PATH = "/api/v1/codebase/scans"

    @pytest.mark.asyncio
    async def test_list_scans_empty(self, handler):
        req = _make_request("GET", self.PATH)
        result = await handler.handle(req, self.PATH, "GET")

        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["total"] == 0
        assert body["data"]["scans"] == []

    @pytest.mark.asyncio
    async def test_list_scans_returns_scans(self, handler):
        _seed_scan(scan_id="scan_001", scan_type=ScanType.SAST)
        _seed_scan(scan_id="scan_002", scan_type=ScanType.BUGS)

        req = _make_request("GET", self.PATH)
        result = await handler.handle(req, self.PATH, "GET")

        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["total"] == 2

    @pytest.mark.asyncio
    async def test_list_scans_filter_by_type(self, handler):
        _seed_scan(scan_id="scan_001", scan_type=ScanType.SAST)
        _seed_scan(scan_id="scan_002", scan_type=ScanType.BUGS)

        req = _make_request("GET", self.PATH, query={"type": "sast"})
        result = await handler.handle(req, self.PATH, "GET")

        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["total"] == 1
        assert body["data"]["scans"][0]["scan_type"] == "sast"

    @pytest.mark.asyncio
    async def test_list_scans_filter_by_status(self, handler):
        _seed_scan(scan_id="scan_001", status=ScanStatus.COMPLETED)
        _seed_scan(scan_id="scan_002", status=ScanStatus.RUNNING)

        req = _make_request("GET", self.PATH, query={"status": "running"})
        result = await handler.handle(req, self.PATH, "GET")

        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["total"] == 1

    @pytest.mark.asyncio
    async def test_list_scans_invalid_type_filter(self, handler):
        req = _make_request("GET", self.PATH, query={"type": "not_real"})
        result = await handler.handle(req, self.PATH, "GET")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_list_scans_invalid_status_filter(self, handler):
        req = _make_request("GET", self.PATH, query={"status": "bogus"})
        result = await handler.handle(req, self.PATH, "GET")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_list_scans_with_limit(self, handler):
        for i in range(5):
            _seed_scan(scan_id=f"scan_{i:03d}")

        req = _make_request("GET", self.PATH, query={"limit": "2"})
        result = await handler.handle(req, self.PATH, "GET")

        assert _status(result) == 200
        body = _body(result)
        assert len(body["data"]["scans"]) == 2
        assert body["data"]["total"] == 5

    @pytest.mark.asyncio
    async def test_list_scans_invalid_limit(self, handler):
        req = _make_request("GET", self.PATH, query={"limit": "abc"})
        result = await handler.handle(req, self.PATH, "GET")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_list_scans_negative_limit(self, handler):
        req = _make_request("GET", self.PATH, query={"limit": "-1"})
        result = await handler.handle(req, self.PATH, "GET")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_list_scans_sorted_newest_first(self, handler):
        s1 = _seed_scan(scan_id="scan_001")
        s1.started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        s2 = _seed_scan(scan_id="scan_002")
        s2.started_at = datetime(2026, 2, 1, tzinfo=timezone.utc)

        req = _make_request("GET", self.PATH)
        result = await handler.handle(req, self.PATH, "GET")

        body = _body(result)
        assert body["data"]["scans"][0]["id"] == "scan_002"


# ===========================================================================
# 6. Get Scan (GET /api/v1/codebase/scan/{scan_id})
# ===========================================================================


class TestGetScan:
    """Tests for GET /api/v1/codebase/scan/{scan_id}."""

    @pytest.mark.asyncio
    async def test_get_scan_success(self, handler):
        _seed_scan(scan_id="scan_abc123")
        path = "/api/v1/codebase/scan/scan_abc123"
        req = _make_request("GET", path)
        result = await handler.handle(req, path, "GET")

        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["scan"]["id"] == "scan_abc123"

    @pytest.mark.asyncio
    async def test_get_scan_not_found(self, handler):
        path = "/api/v1/codebase/scan/scan_nonexistent"
        req = _make_request("GET", path)
        result = await handler.handle(req, path, "GET")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_scan_invalid_id_format(self, handler):
        path = "/api/v1/codebase/scan/!!!invalid!!!"
        req = _make_request("GET", path)
        result = await handler.handle(req, path, "GET")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_get_scan_with_findings(self, handler):
        scan = _seed_scan(scan_id="scan_withfindings")
        finding = _seed_finding(finding_id="find_01")
        scan.findings = [finding]

        path = "/api/v1/codebase/scan/scan_withfindings"
        req = _make_request("GET", path)
        result = await handler.handle(req, path, "GET")

        body = _body(result)
        assert len(body["data"]["findings"]) == 1

    @pytest.mark.asyncio
    async def test_get_scan_cross_tenant_isolation(self, handler):
        """Scans from another tenant should not be visible."""
        _seed_scan(tenant_id="other-tenant", scan_id="scan_other")
        path = "/api/v1/codebase/scan/scan_other"
        req = _make_request("GET", path, tenant_id="test-tenant")
        result = await handler.handle(req, path, "GET")
        assert _status(result) == 404


# ===========================================================================
# 7. List Findings (GET /api/v1/codebase/findings)
# ===========================================================================


class TestListFindings:
    """Tests for GET /api/v1/codebase/findings."""

    PATH = "/api/v1/codebase/findings"

    @pytest.mark.asyncio
    async def test_list_findings_empty(self, handler):
        req = _make_request("GET", self.PATH)
        result = await handler.handle(req, self.PATH, "GET")

        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["total"] == 0

    @pytest.mark.asyncio
    async def test_list_findings_returns_findings(self, handler):
        _seed_finding(finding_id="f1")
        _seed_finding(finding_id="f2")

        req = _make_request("GET", self.PATH)
        result = await handler.handle(req, self.PATH, "GET")

        body = _body(result)
        assert body["data"]["total"] == 2

    @pytest.mark.asyncio
    async def test_list_findings_filter_by_severity(self, handler):
        _seed_finding(finding_id="f1", severity=FindingSeverity.CRITICAL)
        _seed_finding(finding_id="f2", severity=FindingSeverity.LOW)

        req = _make_request("GET", self.PATH, query={"severity": "critical"})
        result = await handler.handle(req, self.PATH, "GET")

        body = _body(result)
        assert body["data"]["total"] == 1
        assert body["data"]["findings"][0]["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_list_findings_filter_by_type(self, handler):
        _seed_finding(finding_id="f1", scan_type=ScanType.SAST)
        _seed_finding(finding_id="f2", scan_type=ScanType.BUGS)

        req = _make_request("GET", self.PATH, query={"type": "bugs"})
        result = await handler.handle(req, self.PATH, "GET")

        body = _body(result)
        assert body["data"]["total"] == 1

    @pytest.mark.asyncio
    async def test_list_findings_filter_by_status(self, handler):
        _seed_finding(finding_id="f1", status=FindingStatus.OPEN)
        _seed_finding(finding_id="f2", status=FindingStatus.DISMISSED)

        req = _make_request("GET", self.PATH, query={"status": "dismissed"})
        result = await handler.handle(req, self.PATH, "GET")

        body = _body(result)
        assert body["data"]["total"] == 1

    @pytest.mark.asyncio
    async def test_list_findings_default_status_is_open(self, handler):
        _seed_finding(finding_id="f1", status=FindingStatus.OPEN)
        _seed_finding(finding_id="f2", status=FindingStatus.DISMISSED)

        req = _make_request("GET", self.PATH)
        result = await handler.handle(req, self.PATH, "GET")

        body = _body(result)
        # Default status filter is "open"
        assert body["data"]["total"] == 1
        assert body["data"]["findings"][0]["status"] == "open"

    @pytest.mark.asyncio
    async def test_list_findings_invalid_severity(self, handler):
        req = _make_request("GET", self.PATH, query={"severity": "ultra_critical"})
        result = await handler.handle(req, self.PATH, "GET")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_list_findings_invalid_type(self, handler):
        req = _make_request("GET", self.PATH, query={"type": "notreal"})
        result = await handler.handle(req, self.PATH, "GET")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_list_findings_invalid_status(self, handler):
        req = _make_request("GET", self.PATH, query={"status": "bogus"})
        result = await handler.handle(req, self.PATH, "GET")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_list_findings_with_limit(self, handler):
        for i in range(5):
            _seed_finding(finding_id=f"f_{i}")

        req = _make_request("GET", self.PATH, query={"limit": "2"})
        result = await handler.handle(req, self.PATH, "GET")

        body = _body(result)
        assert len(body["data"]["findings"]) == 2
        assert body["data"]["total"] == 5

    @pytest.mark.asyncio
    async def test_list_findings_sorted_by_severity(self, handler):
        _seed_finding(finding_id="f_low", severity=FindingSeverity.LOW)
        _seed_finding(finding_id="f_crit", severity=FindingSeverity.CRITICAL)

        req = _make_request("GET", self.PATH)
        result = await handler.handle(req, self.PATH, "GET")

        body = _body(result)
        if body["data"]["total"] >= 2:
            first_severity = body["data"]["findings"][0]["severity"]
            assert first_severity == "critical"

    @pytest.mark.asyncio
    async def test_list_findings_invalid_limit(self, handler):
        req = _make_request("GET", self.PATH, query={"limit": "xyz"})
        result = await handler.handle(req, self.PATH, "GET")
        assert _status(result) == 400


# ===========================================================================
# 8. Dismiss Finding (POST /api/v1/codebase/findings/{id}/dismiss)
# ===========================================================================


class TestDismissFinding:
    """Tests for POST /api/v1/codebase/findings/{id}/dismiss."""

    @pytest.mark.asyncio
    async def test_dismiss_success(self, handler):
        _seed_finding(finding_id="find_001")
        path = "/api/v1/codebase/findings/find_001/dismiss"
        req = _make_request("POST", path, body={"reason": "false alarm", "status": "dismissed"})
        result = await handler.handle(req, path, "POST")

        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["status"] == "dismissed"
        assert body["data"]["finding"]["status"] == "dismissed"

    @pytest.mark.asyncio
    async def test_dismiss_as_false_positive(self, handler):
        _seed_finding(finding_id="find_002")
        path = "/api/v1/codebase/findings/find_002/dismiss"
        req = _make_request("POST", path, body={"reason": "not applicable", "status": "false_positive"})
        result = await handler.handle(req, path, "POST")

        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["finding"]["status"] == "false_positive"

    @pytest.mark.asyncio
    async def test_dismiss_as_accepted_risk(self, handler):
        _seed_finding(finding_id="find_003")
        path = "/api/v1/codebase/findings/find_003/dismiss"
        req = _make_request("POST", path, body={"status": "accepted_risk"})
        result = await handler.handle(req, path, "POST")

        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["finding"]["status"] == "accepted_risk"

    @pytest.mark.asyncio
    async def test_dismiss_as_fixed(self, handler):
        _seed_finding(finding_id="find_004")
        path = "/api/v1/codebase/findings/find_004/dismiss"
        req = _make_request("POST", path, body={"status": "fixed"})
        result = await handler.handle(req, path, "POST")

        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["finding"]["status"] == "fixed"

    @pytest.mark.asyncio
    async def test_dismiss_default_status(self, handler):
        _seed_finding(finding_id="find_005")
        path = "/api/v1/codebase/findings/find_005/dismiss"
        req = _make_request("POST", path, body={"reason": "no reason"})
        result = await handler.handle(req, path, "POST")

        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["finding"]["status"] == "dismissed"

    @pytest.mark.asyncio
    async def test_dismiss_not_found(self, handler):
        path = "/api/v1/codebase/findings/find_nonexistent/dismiss"
        req = _make_request("POST", path, body={"reason": "n/a"})
        result = await handler.handle(req, path, "POST")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_dismiss_invalid_finding_id(self, handler):
        path = "/api/v1/codebase/findings/!!!bad!!!/dismiss"
        req = _make_request("POST", path, body={"reason": "n/a"})
        result = await handler.handle(req, path, "POST")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_dismiss_invalid_status(self, handler):
        _seed_finding(finding_id="find_006")
        path = "/api/v1/codebase/findings/find_006/dismiss"
        req = _make_request("POST", path, body={"status": "invalid_status_value"})
        result = await handler.handle(req, path, "POST")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_dismiss_reason_too_long(self, handler):
        _seed_finding(finding_id="find_007")
        path = "/api/v1/codebase/findings/find_007/dismiss"
        req = _make_request("POST", path, body={"reason": "x" * 2001})
        result = await handler.handle(req, path, "POST")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_dismiss_updates_dismissed_by(self, handler):
        _seed_finding(finding_id="find_008")
        path = "/api/v1/codebase/findings/find_008/dismiss"
        req = _make_request("POST", path, body={"reason": "dismiss"})
        await handler.handle(req, path, "POST")

        findings = _get_tenant_findings("test-tenant")
        assert findings["find_008"].dismissed_by is not None


# ===========================================================================
# 9. Create Issue (POST /api/v1/codebase/findings/{id}/create-issue)
# ===========================================================================


class TestCreateIssue:
    """Tests for POST /api/v1/codebase/findings/{id}/create-issue."""

    @pytest.mark.asyncio
    async def test_create_issue_success(self, handler):
        _seed_finding(finding_id="find_issue1")
        path = "/api/v1/codebase/findings/find_issue1/create-issue"
        req = _make_request("POST", path, body={"repo": "owner/repo"})
        result = await handler.handle(req, path, "POST")

        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["status"] == "created"
        assert "github.com" in body["data"]["issue_url"]
        assert "owner/repo" in body["data"]["issue_url"]

    @pytest.mark.asyncio
    async def test_create_issue_title_includes_severity(self, handler):
        _seed_finding(finding_id="find_issue2", severity=FindingSeverity.CRITICAL, title="Critical Bug")
        path = "/api/v1/codebase/findings/find_issue2/create-issue"
        req = _make_request("POST", path, body={"repo": "org/project"})
        result = await handler.handle(req, path, "POST")

        body = _body(result)
        assert "CRITICAL" in body["data"]["title"]
        assert "Critical Bug" in body["data"]["title"]

    @pytest.mark.asyncio
    async def test_create_issue_not_found(self, handler):
        path = "/api/v1/codebase/findings/find_missing/create-issue"
        req = _make_request("POST", path, body={"repo": "owner/repo"})
        result = await handler.handle(req, path, "POST")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_create_issue_invalid_finding_id(self, handler):
        path = "/api/v1/codebase/findings/!!!/create-issue"
        req = _make_request("POST", path, body={"repo": "owner/repo"})
        result = await handler.handle(req, path, "POST")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_issue_missing_repo(self, handler):
        _seed_finding(finding_id="find_issue3")
        path = "/api/v1/codebase/findings/find_issue3/create-issue"
        req = _make_request("POST", path, body={})
        result = await handler.handle(req, path, "POST")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_issue_invalid_repo_format(self, handler):
        _seed_finding(finding_id="find_issue4")
        path = "/api/v1/codebase/findings/find_issue4/create-issue"
        req = _make_request("POST", path, body={"repo": "not-a-valid-repo"})
        result = await handler.handle(req, path, "POST")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_issue_repo_with_empty_parts(self, handler):
        _seed_finding(finding_id="find_issue5")
        path = "/api/v1/codebase/findings/find_issue5/create-issue"
        req = _make_request("POST", path, body={"repo": "/repo"})
        result = await handler.handle(req, path, "POST")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_issue_repo_with_too_many_slashes(self, handler):
        _seed_finding(finding_id="find_issue6")
        path = "/api/v1/codebase/findings/find_issue6/create-issue"
        req = _make_request("POST", path, body={"repo": "a/b/c"})
        result = await handler.handle(req, path, "POST")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_issue_stores_github_url(self, handler):
        _seed_finding(finding_id="find_issue7")
        path = "/api/v1/codebase/findings/find_issue7/create-issue"
        req = _make_request("POST", path, body={"repo": "owner/repo"})
        await handler.handle(req, path, "POST")

        findings = _get_tenant_findings("test-tenant")
        assert findings["find_issue7"].github_issue_url is not None
        assert "github.com" in findings["find_issue7"].github_issue_url


# ===========================================================================
# 10. Dashboard (GET /api/v1/codebase/dashboard)
# ===========================================================================


class TestDashboard:
    """Tests for GET /api/v1/codebase/dashboard."""

    PATH = "/api/v1/codebase/dashboard"

    @pytest.mark.asyncio
    async def test_dashboard_empty(self, handler):
        req = _make_request("GET", self.PATH)
        result = await handler.handle(req, self.PATH, "GET")

        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["summary"]["total_findings"] == 0
        assert body["data"]["summary"]["total_scans"] == 0

    @pytest.mark.asyncio
    async def test_dashboard_with_findings(self, handler):
        _seed_finding(finding_id="f1", severity=FindingSeverity.CRITICAL)
        _seed_finding(finding_id="f2", severity=FindingSeverity.HIGH)
        _seed_scan(scan_id="scan_001")

        req = _make_request("GET", self.PATH)
        result = await handler.handle(req, self.PATH, "GET")

        body = _body(result)
        assert body["data"]["summary"]["total_findings"] == 2
        assert body["data"]["summary"]["total_scans"] == 1

    @pytest.mark.asyncio
    async def test_dashboard_severity_counts(self, handler):
        _seed_finding(finding_id="f1", severity=FindingSeverity.CRITICAL)
        _seed_finding(finding_id="f2", severity=FindingSeverity.CRITICAL)
        _seed_finding(finding_id="f3", severity=FindingSeverity.LOW)

        req = _make_request("GET", self.PATH)
        result = await handler.handle(req, self.PATH, "GET")

        body = _body(result)
        counts = body["data"]["summary"]["severity_counts"]
        assert counts["critical"] == 2
        assert counts["low"] == 1

    @pytest.mark.asyncio
    async def test_dashboard_ignores_dismissed_findings(self, handler):
        _seed_finding(finding_id="f1", status=FindingStatus.OPEN)
        _seed_finding(finding_id="f2", status=FindingStatus.DISMISSED)

        req = _make_request("GET", self.PATH)
        result = await handler.handle(req, self.PATH, "GET")

        body = _body(result)
        assert body["data"]["summary"]["total_findings"] == 1

    @pytest.mark.asyncio
    async def test_dashboard_risk_score(self, handler):
        _seed_finding(finding_id="f1", severity=FindingSeverity.CRITICAL)

        req = _make_request("GET", self.PATH)
        result = await handler.handle(req, self.PATH, "GET")

        body = _body(result)
        assert body["data"]["summary"]["risk_score"] > 0


# ===========================================================================
# 11. Demo (GET /api/v1/codebase/demo)
# ===========================================================================


class TestDemo:
    """Tests for GET /api/v1/codebase/demo."""

    PATH = "/api/v1/codebase/demo"

    @pytest.mark.asyncio
    async def test_demo_returns_data(self, handler):
        req = _make_request("GET", self.PATH)
        result = await handler.handle(req, self.PATH, "GET")

        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["is_demo"] is True

    @pytest.mark.asyncio
    async def test_demo_has_findings(self, handler):
        req = _make_request("GET", self.PATH)
        result = await handler.handle(req, self.PATH, "GET")

        body = _body(result)
        assert len(body["data"]["findings"]) > 0

    @pytest.mark.asyncio
    async def test_demo_has_metrics(self, handler):
        req = _make_request("GET", self.PATH)
        result = await handler.handle(req, self.PATH, "GET")

        body = _body(result)
        assert "total_lines" in body["data"]["metrics"]

    @pytest.mark.asyncio
    async def test_demo_has_severity_counts(self, handler):
        req = _make_request("GET", self.PATH)
        result = await handler.handle(req, self.PATH, "GET")

        body = _body(result)
        counts = body["data"]["summary"]["severity_counts"]
        assert "critical" in counts
        assert "high" in counts


# ===========================================================================
# 12. Tenant Isolation
# ===========================================================================


class TestTenantIsolation:
    """Tests for multi-tenant data isolation."""

    @pytest.mark.asyncio
    async def test_scans_isolated_by_tenant(self, handler):
        _seed_scan(tenant_id="tenant-a", scan_id="scan_a")
        _seed_scan(tenant_id="tenant-b", scan_id="scan_b")

        req = _make_request("GET", "/api/v1/codebase/scans", tenant_id="tenant-a")
        result = await handler.handle(req, "/api/v1/codebase/scans", "GET")

        body = _body(result)
        assert body["data"]["total"] == 1
        assert body["data"]["scans"][0]["id"] == "scan_a"

    @pytest.mark.asyncio
    async def test_findings_isolated_by_tenant(self, handler):
        _seed_finding(tenant_id="tenant-a", finding_id="f_a")
        _seed_finding(tenant_id="tenant-b", finding_id="f_b")

        req = _make_request("GET", "/api/v1/codebase/findings", tenant_id="tenant-b")
        result = await handler.handle(req, "/api/v1/codebase/findings", "GET")

        body = _body(result)
        assert body["data"]["total"] == 1

    def test_default_tenant_id(self, handler):
        """When request has no tenant_id attribute, default to 'default'."""

        class BareRequest:
            """Request object with no tenant_id attribute."""
            pass

        req = BareRequest()
        tid = handler._get_tenant_id(req)
        assert tid == "default"


# ===========================================================================
# 13. RBAC / Auth enforcement
# ===========================================================================


class TestRBACEnforcement:
    """Tests for RBAC permission enforcement."""

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_unauthenticated_returns_401(self, handler):
        """Without auth bypass, unauthenticated requests return 401."""
        req = _make_request("GET", "/api/v1/codebase/scans")
        result = await handler.handle(req, "/api/v1/codebase/scans", "GET")
        assert _status(result) == 401

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_unauthenticated_post_returns_401(self, handler):
        req = _make_request("POST", "/api/v1/codebase/scan", body={"target_path": "."})
        result = await handler.handle(req, "/api/v1/codebase/scan", "POST")
        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_authenticated_get_succeeds(self, handler):
        """With auth bypass (autouse fixture), GET requests succeed."""
        req = _make_request("GET", "/api/v1/codebase/scans")
        result = await handler.handle(req, "/api/v1/codebase/scans", "GET")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_post_requires_write_permission(self, handler):
        """POST routes should require codebase_audit:write permission."""
        with (
            _patch_scanner("run_sast_scan", []),
            _patch_scanner("run_bug_scan", []),
            _patch_scanner("run_secrets_scan", []),
            _patch_scanner("run_dependency_scan", []),
        ):
            req = _make_request("POST", "/api/v1/codebase/scan", body={"target_path": "."})
            result = await handler.handle(req, "/api/v1/codebase/scan", "POST")
        assert _status(result) == 200


# ===========================================================================
# 14. Query param sanitization
# ===========================================================================


class TestQueryParamSanitization:
    """Tests for _get_query_params sanitization."""

    def test_unknown_params_filtered(self, handler):
        class FakeReq:
            query = {"type": "sast", "evil_param": "drop tables"}

        result = handler._get_query_params(FakeReq())
        assert "type" in result
        assert "evil_param" not in result

    def test_control_characters_removed(self, handler):
        class FakeReq:
            query = {"type": "sast\x00\x01\x02"}

        result = handler._get_query_params(FakeReq())
        assert result["type"] == "sast"

    def test_params_from_args_attribute(self, handler):
        class FakeReq:
            args = {"severity": "high"}

        result = handler._get_query_params(FakeReq())
        assert result["severity"] == "high"

    def test_empty_params(self, handler):
        class FakeReq:
            pass

        result = handler._get_query_params(FakeReq())
        assert result == {}


# ===========================================================================
# 15. _get_json_body edge cases
# ===========================================================================


class TestGetJsonBody:
    """Tests for _get_json_body utility method."""

    @pytest.mark.asyncio
    async def test_json_as_coroutine(self, handler):
        req = _make_request(body={"key": "value"})
        body = await handler._get_json_body(req)
        assert body == {"key": "value"}

    @pytest.mark.asyncio
    async def test_json_as_dict_attribute(self, handler):
        class FakeReq:
            json = {"key": "direct"}

        body = await handler._get_json_body(FakeReq())
        assert body == {"key": "direct"}

    @pytest.mark.asyncio
    async def test_json_not_dict_returns_empty(self, handler):
        class FakeReq:
            json = "not a dict"

        body = await handler._get_json_body(FakeReq())
        assert body == {}

    @pytest.mark.asyncio
    async def test_no_json_attribute_returns_empty(self, handler):
        class FakeReq:
            pass

        body = await handler._get_json_body(FakeReq())
        assert body == {}

    @pytest.mark.asyncio
    async def test_callable_json_returning_non_dict(self, handler):
        class FakeReq:
            async def json(self):
                return ["a", "list"]

        body = await handler._get_json_body(FakeReq())
        assert body == {}


# ===========================================================================
# 16. Handler Registration / Singleton
# ===========================================================================


class TestHandlerRegistration:
    """Tests for get_codebase_audit_handler and handle_codebase_audit."""

    def test_get_handler_returns_instance(self):
        # Reset singleton
        import aragora.server.handlers.features.codebase_audit.handler as hmod
        hmod._handler_instance = None
        h = get_codebase_audit_handler()
        assert isinstance(h, CodebaseAuditHandler)

    def test_get_handler_is_singleton(self):
        import aragora.server.handlers.features.codebase_audit.handler as hmod
        hmod._handler_instance = None
        h1 = get_codebase_audit_handler()
        h2 = get_codebase_audit_handler()
        assert h1 is h2

    @pytest.mark.asyncio
    async def test_handle_codebase_audit_delegates(self):
        import aragora.server.handlers.features.codebase_audit.handler as hmod
        hmod._handler_instance = None

        req = _make_request("GET", "/api/v1/codebase/demo")
        result = await handle_codebase_audit(req, "/api/v1/codebase/demo", "GET")
        assert _status(result) == 200


# ===========================================================================
# 17. Circuit Breaker Integration
# ===========================================================================


class TestCircuitBreaker:
    """Tests for circuit breaker integration in scan endpoints."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self, handler):
        """After enough failures the CB should open and block scans."""
        for _ in range(10):
            _codebase_audit_circuit_breaker.record_failure()

        req = _make_request("POST", "/api/v1/codebase/sast", body={"target_path": "."})
        result = await handler.handle(req, "/api/v1/codebase/sast", "POST")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_on_success(self, handler):
        """Successful scans record success in the CB."""
        with _patch_scanner("run_sast_scan", []):
            req = _make_request("POST", "/api/v1/codebase/sast", body={"target_path": "."})
            result = await handler.handle(req, "/api/v1/codebase/sast", "POST")

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_comprehensive_scan_also_blocked_by_circuit_breaker(self, handler, _open_circuit_breaker):
        req = _make_request("POST", "/api/v1/codebase/scan", body={"target_path": "."})
        result = await handler.handle(req, "/api/v1/codebase/scan", "POST")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_bugs_scan_blocked_by_circuit_breaker(self, handler, _open_circuit_breaker):
        req = _make_request("POST", "/api/v1/codebase/bugs", body={"target_path": "."})
        result = await handler.handle(req, "/api/v1/codebase/bugs", "POST")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_secrets_scan_blocked_by_circuit_breaker(self, handler, _open_circuit_breaker):
        req = _make_request("POST", "/api/v1/codebase/secrets", body={"target_path": "."})
        result = await handler.handle(req, "/api/v1/codebase/secrets", "POST")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_dependencies_scan_blocked_by_circuit_breaker(self, handler, _open_circuit_breaker):
        req = _make_request("POST", "/api/v1/codebase/dependencies", body={"target_path": "."})
        result = await handler.handle(req, "/api/v1/codebase/dependencies", "POST")
        assert _status(result) == 503


# ===========================================================================
# 18. Error Handling in Scans
# ===========================================================================


class TestScanErrorHandling:
    """Tests for error handling within scan operations."""

    @pytest.mark.asyncio
    async def test_single_scan_error_returns_500(self, handler):
        with _patch_scanner("run_bug_scan") as mock_scan:
            mock_scan.side_effect = ValueError("bad scan")
            req = _make_request("POST", "/api/v1/codebase/bugs", body={"target_path": "."})
            result = await handler.handle(req, "/api/v1/codebase/bugs", "POST")

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_metrics_error_returns_500(self, handler):
        with _patch_scanner("run_metrics_analysis") as mock_metrics:
            mock_metrics.side_effect = OSError("disk error")
            req = _make_request("POST", "/api/v1/codebase/metrics", body={"target_path": "."})
            result = await handler.handle(req, "/api/v1/codebase/metrics", "POST")

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_comprehensive_scan_with_partial_failure(self, handler):
        """One scanner failing should not prevent others from succeeding."""
        good_findings = [
            Finding(
                id="good1", scan_id="s", scan_type=ScanType.BUGS,
                severity=FindingSeverity.LOW, title="OK",
                description="d", file_path="f.py",
            )
        ]
        with (
            _patch_scanner("run_sast_scan") as mock_sast,
            _patch_scanner("run_bug_scan", good_findings),
            _patch_scanner("run_secrets_scan", []),
            _patch_scanner("run_dependency_scan", []),
        ):
            mock_sast.side_effect = RuntimeError("scanner crashed")
            req = _make_request(
                "POST",
                "/api/v1/codebase/scan",
                body={"target_path": ".", "scan_types": ["sast", "bugs"]},
            )
            result = await handler.handle(req, "/api/v1/codebase/scan", "POST")

        assert _status(result) == 200
        body = _body(result)
        # Only bug findings should be present (sast failed)
        assert body["data"]["summary"]["total_findings"] == 1


# ===========================================================================
# 19. Validation Edge Cases
# ===========================================================================


class TestValidationEdgeCases:
    """Edge case tests for input validation."""

    @pytest.mark.asyncio
    async def test_scan_path_pipe_injection(self, handler):
        req = _make_request("POST", "/api/v1/codebase/sast", body={"target_path": "src | cat /etc/shadow"})
        result = await handler.handle(req, "/api/v1/codebase/sast", "POST")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_scan_path_backtick_injection(self, handler):
        req = _make_request("POST", "/api/v1/codebase/sast", body={"target_path": "src/`whoami`"})
        result = await handler.handle(req, "/api/v1/codebase/sast", "POST")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_scan_path_dollar_injection(self, handler):
        req = _make_request("POST", "/api/v1/codebase/sast", body={"target_path": "src/$HOME"})
        result = await handler.handle(req, "/api/v1/codebase/sast", "POST")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_scan_path_newline_injection(self, handler):
        req = _make_request("POST", "/api/v1/codebase/sast", body={"target_path": "src\nrm -rf /"})
        result = await handler.handle(req, "/api/v1/codebase/sast", "POST")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_scan_id_too_long(self, handler):
        long_id = "scan_" + "a" * 100
        path = f"/api/v1/codebase/scan/{long_id}"
        req = _make_request("GET", path)
        result = await handler.handle(req, path, "GET")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_finding_id_too_long(self, handler):
        long_id = "find_" + "a" * 100
        path = f"/api/v1/codebase/findings/{long_id}/dismiss"
        req = _make_request("POST", path, body={"reason": "n/a"})
        result = await handler.handle(req, path, "POST")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_github_repo_with_special_chars(self, handler):
        _seed_finding(finding_id="find_val1")
        path = "/api/v1/codebase/findings/find_val1/create-issue"
        req = _make_request("POST", path, body={"repo": "own er/re po"})
        result = await handler.handle(req, path, "POST")
        assert _status(result) == 400


# ===========================================================================
# 20. Risk Score Calculation
# ===========================================================================


class TestRiskScore:
    """Tests for the risk score calculation method."""

    def test_risk_score_no_findings(self, handler):
        assert handler._calculate_risk_score([]) == 0.0

    def test_risk_score_with_critical_finding(self, handler):
        findings = [
            Finding(
                id="f1", scan_id="s", scan_type=ScanType.SAST,
                severity=FindingSeverity.CRITICAL, title="crit",
                description="d", file_path="f.py",
                confidence=1.0,
            )
        ]
        score = handler._calculate_risk_score(findings)
        assert score == 10.0  # critical weight=10 * confidence=1.0

    def test_risk_score_capped_at_100(self, handler):
        findings = [
            Finding(
                id=f"f{i}", scan_id="s", scan_type=ScanType.SAST,
                severity=FindingSeverity.CRITICAL, title=f"crit{i}",
                description="d", file_path="f.py",
                confidence=1.0,
            )
            for i in range(20)
        ]
        score = handler._calculate_risk_score(findings)
        assert score == 100.0

    def test_risk_score_weighted_by_confidence(self, handler):
        findings = [
            Finding(
                id="f1", scan_id="s", scan_type=ScanType.SAST,
                severity=FindingSeverity.HIGH, title="high",
                description="d", file_path="f.py",
                confidence=0.5,
            )
        ]
        score = handler._calculate_risk_score(findings)
        assert score == 2.5  # high weight=5 * confidence=0.5


# ===========================================================================
# 21. Path routing edge cases
# ===========================================================================


class TestPathRoutingEdgeCases:
    """Tests for edge cases in path parameter extraction."""

    @pytest.mark.asyncio
    async def test_findings_path_with_unknown_action(self, handler):
        """Unknown action under findings/{id}/ returns 404."""
        path = "/api/v1/codebase/findings/some_id/unknown_action"
        req = _make_request("POST", path, body={})
        result = await handler.handle(req, path, "POST")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_scan_path_with_extra_segments(self, handler):
        """Extra segments after scan/{id} return 404."""
        path = "/api/v1/codebase/scan/some_id/extra"
        req = _make_request("GET", path)
        result = await handler.handle(req, path, "GET")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_findings_dismiss_with_get_returns_404(self, handler):
        """GET on dismiss endpoint returns 404."""
        _seed_finding(finding_id="find_x")
        path = "/api/v1/codebase/findings/find_x/dismiss"
        req = _make_request("GET", path)
        result = await handler.handle(req, path, "GET")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_findings_create_issue_with_get_returns_404(self, handler):
        _seed_finding(finding_id="find_y")
        path = "/api/v1/codebase/findings/find_y/create-issue"
        req = _make_request("GET", path)
        result = await handler.handle(req, path, "GET")
        assert _status(result) == 404


# ===========================================================================
# 22. Scan result data integrity
# ===========================================================================


class TestScanResultIntegrity:
    """Tests for scan result data structure and integrity."""

    @pytest.mark.asyncio
    async def test_scan_result_has_duration(self, handler):
        with _patch_scanner("run_sast_scan", []):
            req = _make_request("POST", "/api/v1/codebase/sast", body={"target_path": "."})
            result = await handler.handle(req, "/api/v1/codebase/sast", "POST")

        body = _body(result)
        scan = body["data"]["scan"]
        assert "duration_seconds" in scan
        assert scan["duration_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_scan_result_has_completed_at(self, handler):
        with _patch_scanner("run_sast_scan", []):
            req = _make_request("POST", "/api/v1/codebase/sast", body={"target_path": "."})
            result = await handler.handle(req, "/api/v1/codebase/sast", "POST")

        body = _body(result)
        scan = body["data"]["scan"]
        assert scan["status"] == "completed"
        assert scan["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_scan_result_has_correct_type(self, handler):
        with _patch_scanner("run_secrets_scan", []):
            req = _make_request("POST", "/api/v1/codebase/secrets", body={"target_path": "."})
            result = await handler.handle(req, "/api/v1/codebase/secrets", "POST")

        body = _body(result)
        assert body["data"]["scan"]["scan_type"] == "secrets"

    @pytest.mark.asyncio
    async def test_scan_result_has_findings_count(self, handler):
        with _patch_scanner("run_sast_scan", _mock_findings()):
            req = _make_request("POST", "/api/v1/codebase/sast", body={"target_path": "."})
            result = await handler.handle(req, "/api/v1/codebase/sast", "POST")

        body = _body(result)
        assert body["data"]["scan"]["findings_count"] == 2

    @pytest.mark.asyncio
    async def test_comprehensive_scan_result_has_severity_counts(self, handler):
        findings = [
            Finding(
                id="f1", scan_id="s", scan_type=ScanType.SAST,
                severity=FindingSeverity.HIGH, title="h",
                description="d", file_path="f.py",
            ),
        ]
        with (
            _patch_scanner("run_sast_scan", findings),
            _patch_scanner("run_bug_scan", []),
            _patch_scanner("run_secrets_scan", []),
            _patch_scanner("run_dependency_scan", []),
        ):
            req = _make_request(
                "POST", "/api/v1/codebase/scan",
                body={"target_path": ".", "scan_types": ["sast"]},
            )
            result = await handler.handle(req, "/api/v1/codebase/scan", "POST")

        body = _body(result)
        scan = body["data"]["scan"]
        assert "severity_counts" in scan
        assert scan["severity_counts"]["high"] == 1

    @pytest.mark.asyncio
    async def test_metrics_scan_has_metrics_in_result(self, handler):
        with _patch_scanner("run_metrics_analysis", _mock_metrics_result()):
            req = _make_request("POST", "/api/v1/codebase/metrics", body={"target_path": "."})
            result = await handler.handle(req, "/api/v1/codebase/metrics", "POST")

        body = _body(result)
        assert body["data"]["scan"]["metrics"]["total_lines"] == 1000
