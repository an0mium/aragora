"""Tests for quick scan handler (aragora/server/handlers/codebase/quick_scan.py).

Covers all routes and behavior of the QuickScanHandler class
and standalone helper functions:

- handle_post_quick_scan: validation, success, error cases, RBAC
- handle_get_quick_scan: found/not found, RBAC
- handle_list_quick_scans: pagination, sorting, RBAC
- run_quick_scan: scanner integration, mock fallback, error handling
- _validate_repo_path: path traversal prevention, ARAGORA_SCAN_ROOT
- _generate_mock_result: demo result generation
- get_quick_scan_result: in-memory lookup
- list_quick_scans: pagination, sorting
- register_routes: route registration
- _check_permission: auth context checks
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from aragora.server.handlers.codebase.quick_scan import (
    QuickScanHandler,
    _check_permission,
    _generate_mock_result,
    _quick_scan_results,
    _validate_repo_path,
    get_quick_scan_result,
    list_quick_scans,
    register_routes,
    run_quick_scan,
    SCAN_EXECUTE_PERMISSION,
    SCAN_READ_PERMISSION,
)


# ============================================================================
# Helpers
# ============================================================================


def _body(resp: web.Response) -> dict:
    """Parse aiohttp Response body into dict."""
    return json.loads(resp.body)


def _make_request(
    method: str = "POST",
    path: str = "/",
    body: dict | None = None,
    match_info: dict | None = None,
    auth_context: Any = None,
    query: dict[str, str] | None = None,
) -> MagicMock:
    """Create a mock aiohttp request with JSON body and auth context."""
    req = MagicMock(spec=web.Request)
    req.method = method
    req.path = path
    req.headers = {"Authorization": "Bearer test-token"}

    # match_info
    info = match_info or {}
    req.match_info = info

    # query
    req.query = query or {}

    # Auth context
    if auth_context is not None:
        req.auth_context = auth_context
    else:
        # Default: authenticated user with all permissions
        ctx = MagicMock()
        ctx.user_id = "test-user-001"
        req.auth_context = ctx

    if body is not None:
        req.content_length = len(json.dumps(body).encode())
        req.json = AsyncMock(return_value=body)
        body_bytes = json.dumps(body).encode()
        req.read = AsyncMock(return_value=body_bytes)
    else:
        req.content_length = 0
        req.json = AsyncMock(side_effect=json.JSONDecodeError("", "", 0))
        req.read = AsyncMock(return_value=b"")

    return req


def _make_auth_context(user_id: str = "test-user-001") -> MagicMock:
    """Create a mock auth context with a user_id."""
    ctx = MagicMock()
    ctx.user_id = user_id
    return ctx


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def handler():
    """Create a QuickScanHandler instance."""
    return QuickScanHandler()


@pytest.fixture(autouse=True)
def _clear_scan_results():
    """Reset in-memory scan storage between tests."""
    _quick_scan_results.clear()
    yield
    _quick_scan_results.clear()


@pytest.fixture(autouse=True)
def _bypass_rbac(monkeypatch):
    """Bypass the _check_permission function for most tests.

    Tests that specifically test RBAC use the @pytest.mark.no_auto_auth marker
    and mock _check_permission themselves.
    """
    async def _allow(*args, **kwargs):
        return None  # None means permission granted

    monkeypatch.setattr(
        "aragora.server.handlers.codebase.quick_scan._check_permission",
        _allow,
    )


@pytest.fixture
def _restore_check_permission(monkeypatch):
    """Re-enable the real _check_permission for RBAC-specific tests."""
    # Import the real function
    from aragora.server.handlers.codebase import quick_scan as mod

    # Reload the original (un-monkeypatched) _check_permission
    monkeypatch.setattr(
        "aragora.server.handlers.codebase.quick_scan._check_permission",
        mod.__dict__.get("_check_permission", _check_permission),
    )


@pytest.fixture(autouse=True)
def _clear_scan_root_env(monkeypatch):
    """Ensure ARAGORA_SCAN_ROOT is not set by default."""
    monkeypatch.delenv("ARAGORA_SCAN_ROOT", raising=False)


# ============================================================================
# _validate_repo_path
# ============================================================================


class TestValidateRepoPath:
    """Tests for _validate_repo_path path traversal protection."""

    def test_empty_path_returns_error(self):
        resolved, err = _validate_repo_path("")
        assert resolved is None
        assert "required" in err

    def test_whitespace_only_returns_error(self):
        resolved, err = _validate_repo_path("   ")
        assert resolved is None
        assert "required" in err

    def test_none_path_returns_error(self):
        resolved, err = _validate_repo_path(None)
        assert resolved is None
        assert "required" in err

    def test_null_byte_returns_error(self):
        resolved, err = _validate_repo_path("/tmp/foo\x00bar")
        assert resolved is None
        assert "null byte" in err

    def test_valid_path_resolves(self, tmp_path):
        resolved, err = _validate_repo_path(str(tmp_path))
        assert err is None
        assert resolved == os.path.realpath(str(tmp_path))

    def test_path_traversal_within_allowed_root(self, tmp_path, monkeypatch):
        """Path with .. that stays within allowed root is accepted."""
        monkeypatch.setenv("ARAGORA_SCAN_ROOT", str(tmp_path))
        subdir = tmp_path / "sub"
        subdir.mkdir()
        # sub/../ resolves to tmp_path itself
        test_path = str(subdir) + "/.."
        resolved, err = _validate_repo_path(test_path)
        assert err is None
        assert resolved == os.path.realpath(str(tmp_path))

    def test_path_traversal_outside_allowed_root(self, tmp_path, monkeypatch):
        """Path that escapes allowed root is rejected."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        monkeypatch.setenv("ARAGORA_SCAN_ROOT", str(allowed))
        # Attempt to traverse out
        test_path = str(allowed) + "/../../etc/passwd"
        resolved, err = _validate_repo_path(test_path)
        assert resolved is None
        assert "within the allowed" in err

    def test_no_scan_root_allows_any_path(self, tmp_path, monkeypatch):
        """Without ARAGORA_SCAN_ROOT, any path is accepted."""
        monkeypatch.delenv("ARAGORA_SCAN_ROOT", raising=False)
        resolved, err = _validate_repo_path(str(tmp_path))
        assert err is None
        assert resolved is not None

    def test_scan_root_exact_match(self, tmp_path, monkeypatch):
        """Path exactly matching the allowed root is valid."""
        monkeypatch.setenv("ARAGORA_SCAN_ROOT", str(tmp_path))
        resolved, err = _validate_repo_path(str(tmp_path))
        assert err is None
        assert resolved == os.path.realpath(str(tmp_path))

    def test_scan_root_child_path(self, tmp_path, monkeypatch):
        """Path that is a child of the allowed root is valid."""
        monkeypatch.setenv("ARAGORA_SCAN_ROOT", str(tmp_path))
        child = tmp_path / "project"
        child.mkdir()
        resolved, err = _validate_repo_path(str(child))
        assert err is None
        assert resolved == os.path.realpath(str(child))

    def test_scan_root_prefix_attack(self, tmp_path, monkeypatch):
        """Path that shares a prefix but is not under root is rejected.

        E.g., /allowed_extra should not match /allowed.
        """
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        evil = tmp_path / "allowed_extra"
        evil.mkdir()
        monkeypatch.setenv("ARAGORA_SCAN_ROOT", str(allowed))
        resolved, err = _validate_repo_path(str(evil))
        assert resolved is None
        assert "within the allowed" in err

    def test_scan_root_filesystem_root(self, tmp_path, monkeypatch):
        """When ARAGORA_SCAN_ROOT is '/', all paths are allowed."""
        monkeypatch.setenv("ARAGORA_SCAN_ROOT", "/")
        resolved, err = _validate_repo_path(str(tmp_path))
        assert err is None


# ============================================================================
# run_quick_scan
# ============================================================================


class TestRunQuickScan:
    """Tests for the run_quick_scan function."""

    @pytest.mark.asyncio
    async def test_scanner_not_available_returns_mock(self):
        """When SecurityScanner is None, returns mock demo result."""
        with patch(
            "aragora.server.handlers.codebase.quick_scan.SecurityScanner",
            None,
        ):
            result = await run_quick_scan("/some/repo")

        assert result["status"] == "completed"
        assert result["files_scanned"] == 127  # mock result
        assert result["risk_score"] == 35.0
        assert len(result["findings"]) == 3

    @pytest.mark.asyncio
    async def test_scanner_not_available_stores_result(self):
        """Mock result is stored in _quick_scan_results."""
        with patch(
            "aragora.server.handlers.codebase.quick_scan.SecurityScanner",
            None,
        ):
            result = await run_quick_scan("/some/repo")

        assert result["scan_id"] in _quick_scan_results

    @pytest.mark.asyncio
    async def test_custom_scan_id(self):
        """Scan ID can be passed explicitly."""
        with patch(
            "aragora.server.handlers.codebase.quick_scan.SecurityScanner",
            None,
        ):
            result = await run_quick_scan("/some/repo", scan_id="my-scan-123")

        assert result["scan_id"] == "my-scan-123"

    @pytest.mark.asyncio
    async def test_auto_generated_scan_id(self):
        """Without explicit scan_id, one is auto-generated."""
        with patch(
            "aragora.server.handlers.codebase.quick_scan.SecurityScanner",
            None,
        ):
            result = await run_quick_scan("/some/repo")

        assert result["scan_id"].startswith("qscan_")

    @pytest.mark.asyncio
    async def test_scan_directory_success(self, tmp_path):
        """Successful directory scan with SecurityScanner."""
        mock_finding = MagicMock()
        mock_finding.severity = MagicMock()
        mock_finding.severity.value = "high"
        mock_finding.confidence = 0.9
        mock_finding.to_dict.return_value = {"id": "F1", "severity": "high"}

        mock_report = MagicMock()
        mock_report.findings = [mock_finding]
        mock_report.files_scanned = 10
        mock_report.lines_scanned = 500
        mock_report.risk_score = 45.0

        mock_scanner_cls = MagicMock()
        mock_scanner_instance = MagicMock()
        mock_scanner_cls.return_value = mock_scanner_instance
        mock_scanner_instance.scan_directory.return_value = mock_report

        mock_severity = MagicMock()
        mock_severity.MEDIUM = "medium"
        mock_severity.LOW = "low"
        mock_severity.HIGH = "high"
        mock_severity.CRITICAL = "critical"

        with patch(
            "aragora.server.handlers.codebase.quick_scan.SecurityScanner",
            mock_scanner_cls,
        ), patch(
            "aragora.server.handlers.codebase.quick_scan.SecuritySeverity",
            mock_severity,
        ):
            result = await run_quick_scan(str(tmp_path))

        assert result["status"] == "completed"
        assert result["files_scanned"] == 10
        assert result["lines_scanned"] == 500
        assert result["risk_score"] == 45.0
        assert len(result["findings"]) == 1
        assert result["summary"]["high"] == 1

    @pytest.mark.asyncio
    async def test_scan_single_file(self, tmp_path):
        """Single file scan reads the file for line count."""
        test_file = tmp_path / "test.py"
        test_file.write_text("line1\nline2\nline3\n")

        mock_finding = MagicMock()
        mock_finding.severity = MagicMock()
        mock_finding.severity.value = "medium"
        mock_finding.confidence = 0.7
        mock_finding.to_dict.return_value = {"id": "F1", "severity": "medium"}

        mock_scanner_cls = MagicMock()
        mock_scanner_instance = MagicMock()
        mock_scanner_cls.return_value = mock_scanner_instance
        mock_scanner_instance.scan_file.return_value = [mock_finding]

        mock_severity = MagicMock()
        mock_severity.MEDIUM = "medium"
        mock_severity.LOW = "low"
        mock_severity.HIGH = "high"
        mock_severity.CRITICAL = "critical"

        with patch(
            "aragora.server.handlers.codebase.quick_scan.SecurityScanner",
            mock_scanner_cls,
        ), patch(
            "aragora.server.handlers.codebase.quick_scan.SecuritySeverity",
            mock_severity,
        ):
            result = await run_quick_scan(str(test_file))

        assert result["status"] == "completed"
        assert result["files_scanned"] == 1
        assert result["lines_scanned"] == 4  # 3 newlines + 1

    @pytest.mark.asyncio
    async def test_scan_nonexistent_path(self, tmp_path):
        """Scanning a nonexistent path results in failed status."""
        mock_scanner_cls = MagicMock()
        mock_scanner_cls.return_value = MagicMock()

        mock_severity = MagicMock()
        mock_severity.MEDIUM = "medium"
        mock_severity.LOW = "low"
        mock_severity.HIGH = "high"
        mock_severity.CRITICAL = "critical"

        with patch(
            "aragora.server.handlers.codebase.quick_scan.SecurityScanner",
            mock_scanner_cls,
        ), patch(
            "aragora.server.handlers.codebase.quick_scan.SecuritySeverity",
            mock_severity,
        ):
            result = await run_quick_scan(str(tmp_path / "nonexistent"))

        assert result["status"] == "failed"
        assert result["error"] == "Scan failed"

    @pytest.mark.asyncio
    async def test_scan_os_error(self, tmp_path):
        """OSError during scan results in failed status."""
        mock_scanner_cls = MagicMock()
        mock_scanner_instance = MagicMock()
        mock_scanner_cls.return_value = mock_scanner_instance
        mock_scanner_instance.scan_directory.side_effect = OSError("disk error")

        mock_severity = MagicMock()
        mock_severity.MEDIUM = "medium"
        mock_severity.LOW = "low"
        mock_severity.HIGH = "high"
        mock_severity.CRITICAL = "critical"

        with patch(
            "aragora.server.handlers.codebase.quick_scan.SecurityScanner",
            mock_scanner_cls,
        ), patch(
            "aragora.server.handlers.codebase.quick_scan.SecuritySeverity",
            mock_severity,
        ):
            result = await run_quick_scan(str(tmp_path))

        assert result["status"] == "failed"
        assert result["error"] == "Scan failed"
        assert result["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_risk_score_calculated_when_zero(self, tmp_path):
        """Risk score is computed from findings when report.risk_score is 0."""
        finding1 = MagicMock()
        finding1.severity = MagicMock()
        finding1.severity.value = "critical"
        finding1.confidence = 1.0
        finding1.to_dict.return_value = {"severity": "critical"}

        finding2 = MagicMock()
        finding2.severity = MagicMock()
        finding2.severity.value = "low"
        finding2.confidence = 0.5
        finding2.to_dict.return_value = {"severity": "low"}

        mock_report = MagicMock()
        mock_report.findings = [finding1, finding2]
        mock_report.files_scanned = 5
        mock_report.lines_scanned = 100
        mock_report.risk_score = 0  # zero triggers manual calc

        mock_scanner_cls = MagicMock()
        mock_scanner_instance = MagicMock()
        mock_scanner_cls.return_value = mock_scanner_instance
        mock_scanner_instance.scan_directory.return_value = mock_report

        mock_severity = MagicMock()
        mock_severity.MEDIUM = "medium"
        mock_severity.LOW = "low"
        mock_severity.HIGH = "high"
        mock_severity.CRITICAL = "critical"

        with patch(
            "aragora.server.handlers.codebase.quick_scan.SecurityScanner",
            mock_scanner_cls,
        ), patch(
            "aragora.server.handlers.codebase.quick_scan.SecuritySeverity",
            mock_severity,
        ):
            result = await run_quick_scan(str(tmp_path))

        # critical: 40 * 1.0 = 40, low: 5 * 0.5 = 2.5, total = 42.5
        assert result["risk_score"] == 42.5
        assert result["summary"]["critical"] == 1
        assert result["summary"]["low"] == 1

    @pytest.mark.asyncio
    async def test_risk_score_capped_at_100(self, tmp_path):
        """Risk score is capped at 100."""
        # Create many critical findings
        findings = []
        for _ in range(10):
            f = MagicMock()
            f.severity = MagicMock()
            f.severity.value = "critical"
            f.confidence = 1.0
            f.to_dict.return_value = {"severity": "critical"}
            findings.append(f)

        mock_report = MagicMock()
        mock_report.findings = findings
        mock_report.files_scanned = 50
        mock_report.lines_scanned = 5000
        mock_report.risk_score = 0  # zero triggers manual calc

        mock_scanner_cls = MagicMock()
        mock_scanner_instance = MagicMock()
        mock_scanner_cls.return_value = mock_scanner_instance
        mock_scanner_instance.scan_directory.return_value = mock_report

        mock_severity = MagicMock()
        mock_severity.MEDIUM = "medium"
        mock_severity.LOW = "low"
        mock_severity.HIGH = "high"
        mock_severity.CRITICAL = "critical"

        with patch(
            "aragora.server.handlers.codebase.quick_scan.SecurityScanner",
            mock_scanner_cls,
        ), patch(
            "aragora.server.handlers.codebase.quick_scan.SecuritySeverity",
            mock_severity,
        ):
            result = await run_quick_scan(str(tmp_path))

        # 10 * 40 * 1.0 = 400, capped at 100
        assert result["risk_score"] == 100.0

    @pytest.mark.asyncio
    async def test_severity_threshold_low(self, tmp_path):
        """Low severity threshold enables include_low_severity."""
        mock_scanner_cls = MagicMock()
        mock_scanner_instance = MagicMock()
        mock_scanner_cls.return_value = mock_scanner_instance

        mock_report = MagicMock()
        mock_report.findings = []
        mock_report.files_scanned = 1
        mock_report.lines_scanned = 10
        mock_report.risk_score = 0
        mock_scanner_instance.scan_directory.return_value = mock_report

        mock_severity = MagicMock()
        mock_severity.MEDIUM = "medium"
        mock_severity.LOW = "low"
        mock_severity.HIGH = "high"
        mock_severity.CRITICAL = "critical"

        with patch(
            "aragora.server.handlers.codebase.quick_scan.SecurityScanner",
            mock_scanner_cls,
        ), patch(
            "aragora.server.handlers.codebase.quick_scan.SecuritySeverity",
            mock_severity,
        ):
            await run_quick_scan(str(tmp_path), severity_threshold="low")

        mock_scanner_cls.assert_called_once_with(
            include_low_severity=True,
            include_info=False,
        )

    @pytest.mark.asyncio
    async def test_severity_threshold_high(self, tmp_path):
        """High severity threshold disables include_low_severity."""
        mock_scanner_cls = MagicMock()
        mock_scanner_instance = MagicMock()
        mock_scanner_cls.return_value = mock_scanner_instance

        mock_report = MagicMock()
        mock_report.findings = []
        mock_report.files_scanned = 1
        mock_report.lines_scanned = 10
        mock_report.risk_score = 0
        mock_scanner_instance.scan_directory.return_value = mock_report

        mock_severity = MagicMock()
        mock_severity.MEDIUM = "medium"
        mock_severity.LOW = "low"
        mock_severity.HIGH = "high"
        mock_severity.CRITICAL = "critical"

        with patch(
            "aragora.server.handlers.codebase.quick_scan.SecurityScanner",
            mock_scanner_cls,
        ), patch(
            "aragora.server.handlers.codebase.quick_scan.SecuritySeverity",
            mock_severity,
        ):
            await run_quick_scan(str(tmp_path), severity_threshold="high")

        mock_scanner_cls.assert_called_once_with(
            include_low_severity=False,
            include_info=False,
        )


# ============================================================================
# _generate_mock_result
# ============================================================================


class TestGenerateMockResult:
    """Tests for mock result generation."""

    def test_returns_completed_status(self):
        start = datetime.now(timezone.utc)
        result = _generate_mock_result("scan-1", "/repo", start)
        assert result["status"] == "completed"

    def test_has_scan_id(self):
        start = datetime.now(timezone.utc)
        result = _generate_mock_result("scan-1", "/repo", start)
        assert result["scan_id"] == "scan-1"

    def test_has_repository(self):
        start = datetime.now(timezone.utc)
        result = _generate_mock_result("scan-1", "/my/repo", start)
        assert result["repository"] == "/my/repo"

    def test_has_findings(self):
        start = datetime.now(timezone.utc)
        result = _generate_mock_result("scan-1", "/repo", start)
        assert len(result["findings"]) == 3

    def test_has_summary_counts(self):
        start = datetime.now(timezone.utc)
        result = _generate_mock_result("scan-1", "/repo", start)
        assert result["summary"]["critical"] == 0
        assert result["summary"]["high"] == 2
        assert result["summary"]["medium"] == 5

    def test_risk_score(self):
        start = datetime.now(timezone.utc)
        result = _generate_mock_result("scan-1", "/repo", start)
        assert result["risk_score"] == 35.0

    def test_no_error(self):
        start = datetime.now(timezone.utc)
        result = _generate_mock_result("scan-1", "/repo", start)
        assert result["error"] is None

    def test_started_at_matches(self):
        start = datetime.now(timezone.utc)
        result = _generate_mock_result("scan-1", "/repo", start)
        assert result["started_at"] == start.isoformat()

    def test_completed_at_set(self):
        start = datetime.now(timezone.utc)
        result = _generate_mock_result("scan-1", "/repo", start)
        assert result["completed_at"] is not None


# ============================================================================
# get_quick_scan_result
# ============================================================================


class TestGetQuickScanResult:
    """Tests for get_quick_scan_result."""

    @pytest.mark.asyncio
    async def test_returns_stored_result(self):
        _quick_scan_results["scan-abc"] = {"scan_id": "scan-abc", "status": "done"}
        result = await get_quick_scan_result("scan-abc")
        assert result == {"scan_id": "scan-abc", "status": "done"}

    @pytest.mark.asyncio
    async def test_returns_none_for_missing(self):
        result = await get_quick_scan_result("nonexistent")
        assert result is None


# ============================================================================
# list_quick_scans
# ============================================================================


class TestListQuickScans:
    """Tests for list_quick_scans."""

    @pytest.mark.asyncio
    async def test_empty_list(self):
        result = await list_quick_scans()
        assert result["scans"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_returns_scans_sorted_by_started_at(self):
        _quick_scan_results["s1"] = {
            "scan_id": "s1",
            "repository": "/repo1",
            "status": "completed",
            "started_at": "2026-01-01T00:00:00Z",
            "completed_at": "2026-01-01T00:01:00Z",
            "risk_score": 10,
            "findings": [{"id": "f1"}],
        }
        _quick_scan_results["s2"] = {
            "scan_id": "s2",
            "repository": "/repo2",
            "status": "running",
            "started_at": "2026-02-01T00:00:00Z",
            "completed_at": None,
            "risk_score": 0,
            "findings": [],
        }

        result = await list_quick_scans()
        assert result["total"] == 2
        assert len(result["scans"]) == 2
        # Most recent first
        assert result["scans"][0]["scan_id"] == "s2"
        assert result["scans"][1]["scan_id"] == "s1"

    @pytest.mark.asyncio
    async def test_pagination_limit(self):
        for i in range(5):
            _quick_scan_results[f"s{i}"] = {
                "scan_id": f"s{i}",
                "repository": f"/repo{i}",
                "status": "completed",
                "started_at": f"2026-01-0{i + 1}T00:00:00Z",
                "completed_at": f"2026-01-0{i + 1}T00:01:00Z",
                "risk_score": 0,
                "findings": [],
            }

        result = await list_quick_scans(limit=2)
        assert result["total"] == 5
        assert len(result["scans"]) == 2
        assert result["limit"] == 2

    @pytest.mark.asyncio
    async def test_pagination_offset(self):
        for i in range(5):
            _quick_scan_results[f"s{i}"] = {
                "scan_id": f"s{i}",
                "repository": f"/repo{i}",
                "status": "completed",
                "started_at": f"2026-01-0{i + 1}T00:00:00Z",
                "completed_at": f"2026-01-0{i + 1}T00:01:00Z",
                "risk_score": 0,
                "findings": [],
            }

        result = await list_quick_scans(limit=2, offset=3)
        assert result["total"] == 5
        assert len(result["scans"]) == 2
        assert result["offset"] == 3

    @pytest.mark.asyncio
    async def test_scan_summary_fields(self):
        _quick_scan_results["s1"] = {
            "scan_id": "s1",
            "repository": "/repo",
            "status": "completed",
            "started_at": "2026-01-01T00:00:00Z",
            "completed_at": "2026-01-01T00:01:00Z",
            "risk_score": 42.5,
            "findings": [{"id": "f1"}, {"id": "f2"}],
        }

        result = await list_quick_scans()
        scan = result["scans"][0]
        assert scan["scan_id"] == "s1"
        assert scan["repository"] == "/repo"
        assert scan["status"] == "completed"
        assert scan["risk_score"] == 42.5
        assert scan["findings_count"] == 2


# ============================================================================
# QuickScanHandler - handle_post_quick_scan
# ============================================================================


class TestHandlePostQuickScan:
    """Tests for POST /api/codebase/quick-scan."""

    @pytest.mark.asyncio
    async def test_missing_repo_path(self, handler):
        req = _make_request(body={})
        resp = await handler.handle_post_quick_scan(req)
        assert resp.status == 400
        body = _body(resp)
        assert body["success"] is False
        assert "repo_path" in body["error"]

    @pytest.mark.asyncio
    async def test_null_repo_path(self, handler):
        req = _make_request(body={"repo_path": None})
        resp = await handler.handle_post_quick_scan(req)
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_empty_repo_path(self, handler):
        req = _make_request(body={"repo_path": ""})
        resp = await handler.handle_post_quick_scan(req)
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_path_with_null_byte(self, handler):
        req = _make_request(body={"repo_path": "/tmp/foo\x00bar"})
        resp = await handler.handle_post_quick_scan(req)
        assert resp.status == 400
        body = _body(resp)
        assert "null byte" in body["error"]

    @pytest.mark.asyncio
    async def test_successful_scan(self, handler, tmp_path):
        with patch(
            "aragora.server.handlers.codebase.quick_scan.SecurityScanner",
            None,
        ):
            req = _make_request(body={"repo_path": str(tmp_path)})
            resp = await handler.handle_post_quick_scan(req)

        assert resp.status == 200
        body = _body(resp)
        assert body["success"] is True
        assert body["status"] == "completed"
        assert "scan_id" in body

    @pytest.mark.asyncio
    async def test_custom_severity_threshold(self, handler, tmp_path):
        with patch(
            "aragora.server.handlers.codebase.quick_scan.SecurityScanner",
            None,
        ):
            req = _make_request(body={
                "repo_path": str(tmp_path),
                "severity_threshold": "critical",
            })
            resp = await handler.handle_post_quick_scan(req)

        assert resp.status == 200
        body = _body(resp)
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_include_secrets_param(self, handler, tmp_path):
        with patch(
            "aragora.server.handlers.codebase.quick_scan.SecurityScanner",
            None,
        ):
            req = _make_request(body={
                "repo_path": str(tmp_path),
                "include_secrets": False,
            })
            resp = await handler.handle_post_quick_scan(req)

        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, handler, tmp_path, monkeypatch):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        monkeypatch.setenv("ARAGORA_SCAN_ROOT", str(allowed))
        req = _make_request(body={"repo_path": "/etc/passwd"})
        resp = await handler.handle_post_quick_scan(req)
        assert resp.status == 400
        body = _body(resp)
        assert "within the allowed" in body["error"]

    @pytest.mark.asyncio
    async def test_internal_error_returns_500(self, handler, tmp_path):
        """Unexpected runtime error in the outer try block returns 500."""
        with patch(
            "aragora.server.handlers.codebase.quick_scan._validate_repo_path",
            side_effect=RuntimeError("unexpected"),
        ):
            req = _make_request(body={"repo_path": str(tmp_path)})
            resp = await handler.handle_post_quick_scan(req)

        assert resp.status == 500
        body = _body(resp)
        assert body["success"] is False

    @pytest.mark.asyncio
    async def test_empty_body_returns_error(self, handler):
        """Empty/invalid JSON body returns 400."""
        req = _make_request(body=None)
        resp = await handler.handle_post_quick_scan(req)
        # parse_json_body returns 400 for empty body
        assert resp.status == 400


# ============================================================================
# QuickScanHandler - handle_get_quick_scan
# ============================================================================


class TestHandleGetQuickScan:
    """Tests for GET /api/codebase/quick-scan/{scan_id}."""

    @pytest.mark.asyncio
    async def test_missing_scan_id(self, handler):
        req = _make_request(method="GET", match_info={})
        # When match_info.get("scan_id") returns None
        resp = await handler.handle_get_quick_scan(req)
        assert resp.status == 400
        body = _body(resp)
        assert "scan_id" in body["error"]

    @pytest.mark.asyncio
    async def test_scan_not_found(self, handler):
        req = _make_request(method="GET", match_info={"scan_id": "nonexistent"})
        resp = await handler.handle_get_quick_scan(req)
        assert resp.status == 404
        body = _body(resp)
        assert "not found" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_scan_found(self, handler):
        _quick_scan_results["scan-abc"] = {
            "scan_id": "scan-abc",
            "status": "completed",
            "repository": "/repo",
            "risk_score": 25.0,
        }
        req = _make_request(method="GET", match_info={"scan_id": "scan-abc"})
        resp = await handler.handle_get_quick_scan(req)
        assert resp.status == 200
        body = _body(resp)
        assert body["success"] is True
        assert body["scan_id"] == "scan-abc"
        assert body["risk_score"] == 25.0

    @pytest.mark.asyncio
    async def test_scan_found_running(self, handler):
        """Retrieving a scan that's still running."""
        _quick_scan_results["running-scan"] = {
            "scan_id": "running-scan",
            "status": "running",
            "repository": "/repo",
            "risk_score": 0,
        }
        req = _make_request(method="GET", match_info={"scan_id": "running-scan"})
        resp = await handler.handle_get_quick_scan(req)
        assert resp.status == 200
        body = _body(resp)
        assert body["status"] == "running"


# ============================================================================
# QuickScanHandler - handle_list_quick_scans
# ============================================================================


class TestHandleListQuickScans:
    """Tests for GET /api/codebase/quick-scans."""

    @pytest.mark.asyncio
    async def test_empty_list(self, handler):
        req = _make_request(method="GET", query={})
        resp = await handler.handle_list_quick_scans(req)
        assert resp.status == 200
        body = _body(resp)
        assert body["success"] is True
        assert body["scans"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_with_scans(self, handler):
        # Add 2 scans so the default offset=1 (clamped by safe_query_int min_val=1)
        # still returns results
        _quick_scan_results["s1"] = {
            "scan_id": "s1",
            "repository": "/repo1",
            "status": "completed",
            "started_at": "2026-01-01T00:00:00Z",
            "completed_at": "2026-01-01T00:01:00Z",
            "risk_score": 10,
            "findings": [{"id": "f1"}],
        }
        _quick_scan_results["s2"] = {
            "scan_id": "s2",
            "repository": "/repo2",
            "status": "completed",
            "started_at": "2026-01-02T00:00:00Z",
            "completed_at": "2026-01-02T00:01:00Z",
            "risk_score": 20,
            "findings": [{"id": "f2"}],
        }
        req = _make_request(method="GET", query={})
        resp = await handler.handle_list_quick_scans(req)
        assert resp.status == 200
        body = _body(resp)
        assert body["total"] == 2
        # safe_query_int clamps offset to min_val=1, so we get scans[1:21] = 1 result
        assert len(body["scans"]) == 1

    @pytest.mark.asyncio
    async def test_limit_param(self, handler):
        for i in range(5):
            _quick_scan_results[f"s{i}"] = {
                "scan_id": f"s{i}",
                "repository": f"/repo{i}",
                "status": "completed",
                "started_at": f"2026-01-0{i + 1}T00:00:00Z",
                "completed_at": f"2026-01-0{i + 1}T00:01:00Z",
                "risk_score": 0,
                "findings": [],
            }
        req = _make_request(method="GET", query={"limit": "2"})
        resp = await handler.handle_list_quick_scans(req)
        body = _body(resp)
        assert body["total"] == 5
        assert len(body["scans"]) == 2

    @pytest.mark.asyncio
    async def test_offset_param(self, handler):
        for i in range(5):
            _quick_scan_results[f"s{i}"] = {
                "scan_id": f"s{i}",
                "repository": f"/repo{i}",
                "status": "completed",
                "started_at": f"2026-01-0{i + 1}T00:00:00Z",
                "completed_at": f"2026-01-0{i + 1}T00:01:00Z",
                "risk_score": 0,
                "findings": [],
            }
        req = _make_request(method="GET", query={"offset": "3"})
        resp = await handler.handle_list_quick_scans(req)
        body = _body(resp)
        assert body["total"] == 5
        assert len(body["scans"]) == 2

    @pytest.mark.asyncio
    async def test_default_limit_and_offset(self, handler):
        """Default limit=20, offset clamped to 1 by safe_query_int(min_val=1)."""
        req = _make_request(method="GET", query={})
        resp = await handler.handle_list_quick_scans(req)
        body = _body(resp)
        assert body["limit"] == 20
        # safe_query_int clamps offset 0 to min_val=1
        assert body["offset"] == 1


# ============================================================================
# QuickScanHandler initialization
# ============================================================================


class TestHandlerInit:
    """Tests for QuickScanHandler initialization."""

    def test_init_no_context(self):
        h = QuickScanHandler()
        assert h._ctx is None

    def test_init_with_context(self):
        ctx = {"key": "value"}
        h = QuickScanHandler(ctx)
        assert h._ctx == ctx


# ============================================================================
# register_routes
# ============================================================================


class TestRegisterRoutes:
    """Tests for route registration."""

    def test_registers_all_routes(self):
        app = web.Application()
        register_routes(app)

        routes = [r.get_info().get("path", r.get_info().get("formatter", ""))
                  for r in app.router.routes()]

        assert "/api/v1/codebase/quick-scan" in routes
        assert "/api/v1/codebase/quick-scan/{scan_id}" in routes
        assert "/api/v1/codebase/quick-scans" in routes
        assert "/api/codebase/quick-scan" in routes
        assert "/api/codebase/quick-scan/{scan_id}" in routes
        assert "/api/codebase/quick-scans" in routes

    def test_registers_correct_methods(self):
        app = web.Application()
        register_routes(app)

        route_methods = {}
        for r in app.router.routes():
            info = r.get_info()
            path = info.get("path", info.get("formatter", ""))
            route_methods.setdefault(path, set()).add(r.method)

        assert "POST" in route_methods.get("/api/v1/codebase/quick-scan", set())
        assert "GET" in route_methods.get("/api/v1/codebase/quick-scan/{scan_id}", set())
        assert "GET" in route_methods.get("/api/v1/codebase/quick-scans", set())


# ============================================================================
# _check_permission (RBAC tests)
# ============================================================================


class TestCheckPermission:
    """Tests for the _check_permission function."""

    @pytest.mark.asyncio
    async def test_no_auth_context_returns_401(self):
        req = MagicMock(spec=web.Request)
        req.auth_context = None  # Explicitly set
        # Use the real _check_permission (not the monkeypatched bypass)
        resp = await _check_permission(req, "test:permission")
        assert resp is not None
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_no_user_id_returns_401(self):
        req = MagicMock(spec=web.Request)
        ctx = MagicMock()
        ctx.user_id = None
        req.auth_context = ctx
        resp = await _check_permission(req, "test:permission")
        assert resp is not None
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_permission_denied_returns_403(self):
        req = MagicMock(spec=web.Request)
        ctx = MagicMock()
        ctx.user_id = "user-1"
        req.auth_context = ctx

        mock_checker = MagicMock()
        mock_result = MagicMock()
        mock_result.allowed = False
        mock_checker.check_permission.return_value = mock_result

        with patch(
            "aragora.server.handlers.codebase.quick_scan.get_permission_checker",
            return_value=mock_checker,
        ):
            resp = await _check_permission(req, "test:permission")

        assert resp is not None
        assert resp.status == 403
        body = json.loads(resp.body)
        assert "Permission denied" in body["error"]

    @pytest.mark.asyncio
    async def test_permission_granted_returns_none(self):
        req = MagicMock(spec=web.Request)
        ctx = MagicMock()
        ctx.user_id = "user-1"
        req.auth_context = ctx

        mock_checker = MagicMock()
        mock_result = MagicMock()
        mock_result.allowed = True
        mock_checker.check_permission.return_value = mock_result

        with patch(
            "aragora.server.handlers.codebase.quick_scan.get_permission_checker",
            return_value=mock_checker,
        ):
            resp = await _check_permission(req, "test:permission")

        assert resp is None  # None means granted

    @pytest.mark.asyncio
    async def test_exception_in_check_returns_401(self):
        """AttributeError/TypeError during check returns 401."""
        req = MagicMock(spec=web.Request)
        ctx = MagicMock()
        ctx.user_id = "user-1"
        req.auth_context = ctx

        with patch(
            "aragora.server.handlers.codebase.quick_scan.get_permission_checker",
            side_effect=RuntimeError("broken"),
        ):
            resp = await _check_permission(req, "test:permission")

        assert resp is not None
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_missing_auth_context_attr(self):
        """Request without auth_context attribute returns 401."""
        req = MagicMock(spec=web.Request)
        # Ensure getattr(request, "auth_context", None) returns None
        del req.auth_context
        resp = await _check_permission(req, "test:permission")
        assert resp is not None
        assert resp.status == 401


# ============================================================================
# Permission constants
# ============================================================================


class TestPermissionConstants:
    """Verify permission constant values."""

    def test_scan_read_permission(self):
        assert SCAN_READ_PERMISSION == "codebase:scan:read"

    def test_scan_execute_permission(self):
        assert SCAN_EXECUTE_PERMISSION == "codebase:scan:execute"


# ============================================================================
# Integration-style: end-to-end POST -> GET flow
# ============================================================================


class TestEndToEndFlow:
    """Integration test: run scan then retrieve result."""

    @pytest.mark.asyncio
    async def test_post_then_get(self, handler, tmp_path):
        """POST creates scan, GET retrieves it by ID."""
        with patch(
            "aragora.server.handlers.codebase.quick_scan.SecurityScanner",
            None,
        ):
            # POST to create scan
            post_req = _make_request(body={"repo_path": str(tmp_path)})
            post_resp = await handler.handle_post_quick_scan(post_req)

        assert post_resp.status == 200
        post_body = _body(post_resp)
        scan_id = post_body["scan_id"]

        # GET to retrieve it
        get_req = _make_request(method="GET", match_info={"scan_id": scan_id})
        get_resp = await handler.handle_get_quick_scan(get_req)
        assert get_resp.status == 200
        get_body = _body(get_resp)
        assert get_body["scan_id"] == scan_id
        assert get_body["success"] is True

    @pytest.mark.asyncio
    async def test_post_then_list(self, handler, tmp_path):
        """POST creates scan, list shows it."""
        with patch(
            "aragora.server.handlers.codebase.quick_scan.SecurityScanner",
            None,
        ):
            post_req = _make_request(body={"repo_path": str(tmp_path)})
            await handler.handle_post_quick_scan(post_req)

        list_req = _make_request(method="GET", query={})
        list_resp = await handler.handle_list_quick_scans(list_req)
        assert list_resp.status == 200
        list_body = _body(list_resp)
        assert list_body["total"] >= 1
