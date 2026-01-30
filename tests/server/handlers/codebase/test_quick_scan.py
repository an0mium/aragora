"""
Tests for Quick Security Scan API Handler.

Tests cover:
- QuickScanHandler routing (can_handle)
- POST /api/codebase/quick-scan - Run quick security scan
- GET /api/codebase/quick-scan/{scan_id} - Get scan result
- GET /api/codebase/quick-scans - List quick scans
- RBAC permission checks (codebase:scan:read, codebase:scan:execute)
- Error handling (missing fields, path not found, internal errors)
- Mock result generation
"""

import json
import pytest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from aragora.server.handlers.codebase.quick_scan import (
    QuickScanHandler,
    run_quick_scan,
    get_quick_scan_result,
    list_quick_scans,
    _quick_scan_results,
    _generate_mock_result,
    SCAN_READ_PERMISSION,
    SCAN_EXECUTE_PERMISSION,
)


# =============================================================================
# Helpers
# =============================================================================


def _parse_result(result):
    """Parse aiohttp Response into (body_dict, status_code)."""
    if hasattr(result, "body"):
        body = json.loads(result.body) if result.body else {}
    elif hasattr(result, "text"):
        body = json.loads(result.text) if result.text else {}
    else:
        body = {}
    return body, getattr(result, "status", 200)


def _make_mock_request(
    *,
    method: str = "GET",
    match_info: dict = None,
    query: dict = None,
    json_body: dict = None,
    auth_context=None,
):
    """Build a mock aiohttp request."""
    request = MagicMock()
    request.method = method
    request.match_info = match_info or {}
    request.query = query or {}

    if json_body is not None:

        async def _json():
            return json_body

        request.json = _json
    else:

        async def _json():
            return {}

        request.json = _json

    # Set auth context if provided
    if auth_context is not None:
        request.auth_context = auth_context
    else:
        # Default: no auth
        request.auth_context = None

    return request


def _make_auth_context(user_id: str = "user_123", roles: set = None):
    """Create a mock authorization context."""
    ctx = MagicMock()
    ctx.user_id = user_id
    ctx.roles = roles or {"member"}
    return ctx


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clear_scan_results():
    """Clear the in-memory scan results between tests."""
    _quick_scan_results.clear()
    yield
    _quick_scan_results.clear()


@pytest.fixture
def handler():
    """Create a QuickScanHandler instance."""
    return QuickScanHandler()


@pytest.fixture
def mock_permission_checker():
    """Create a mock permission checker that allows all permissions."""
    checker = MagicMock()
    decision = MagicMock()
    decision.allowed = True
    checker.check_permission.return_value = decision
    return checker


@pytest.fixture
def mock_permission_checker_denied():
    """Create a mock permission checker that denies all permissions."""
    checker = MagicMock()
    decision = MagicMock()
    decision.allowed = False
    checker.check_permission.return_value = decision
    return checker


# =============================================================================
# Tests: Permission Constants
# =============================================================================


class TestPermissionConstants:
    """Test permission constant definitions."""

    def test_scan_read_permission(self):
        assert SCAN_READ_PERMISSION == "codebase:scan:read"

    def test_scan_execute_permission(self):
        assert SCAN_EXECUTE_PERMISSION == "codebase:scan:execute"


# =============================================================================
# Tests: run_quick_scan
# =============================================================================


class TestRunQuickScan:
    """Tests for the run_quick_scan function."""

    @pytest.mark.asyncio
    async def test_run_quick_scan_mock_fallback(self):
        """When scanner import fails, returns mock result."""
        # Without the actual scanner, should return mock result
        with patch(
            "aragora.server.handlers.codebase.quick_scan.SecurityScanner",
            side_effect=ImportError("No scanner"),
        ):
            result = await run_quick_scan("/some/repo/path")

        assert result["status"] == "completed"
        assert result["scan_id"].startswith("qscan_")
        assert result["repository"] == "/some/repo/path"
        assert "findings" in result
        assert "summary" in result

    @pytest.mark.asyncio
    async def test_run_quick_scan_path_not_found(self):
        """When path does not exist, returns error."""
        result = await run_quick_scan("/nonexistent/path/that/does/not/exist")

        assert result["status"] == "failed"
        assert result["error"] is not None
        assert "does not exist" in result["error"]

    @pytest.mark.asyncio
    async def test_run_quick_scan_stores_result(self):
        """Scan result is stored in memory."""
        with patch(
            "aragora.server.handlers.codebase.quick_scan.SecurityScanner",
            side_effect=ImportError("No scanner"),
        ):
            result = await run_quick_scan("/some/path")

        scan_id = result["scan_id"]
        assert scan_id in _quick_scan_results

    @pytest.mark.asyncio
    async def test_run_quick_scan_custom_scan_id(self):
        """Can provide a custom scan ID."""
        with patch(
            "aragora.server.handlers.codebase.quick_scan.SecurityScanner",
            side_effect=ImportError("No scanner"),
        ):
            result = await run_quick_scan("/path", scan_id="custom_scan_123")

        assert result["scan_id"] == "custom_scan_123"


# =============================================================================
# Tests: get_quick_scan_result
# =============================================================================


class TestGetQuickScanResult:
    """Tests for get_quick_scan_result function."""

    @pytest.mark.asyncio
    async def test_get_existing_result(self):
        """Can retrieve an existing scan result."""
        _quick_scan_results["test_scan_1"] = {
            "scan_id": "test_scan_1",
            "status": "completed",
            "repository": "/test/path",
        }

        result = await get_quick_scan_result("test_scan_1")

        assert result is not None
        assert result["scan_id"] == "test_scan_1"
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_get_nonexistent_result(self):
        """Returns None for nonexistent scan."""
        result = await get_quick_scan_result("nonexistent_scan")
        assert result is None


# =============================================================================
# Tests: list_quick_scans
# =============================================================================


class TestListQuickScans:
    """Tests for list_quick_scans function."""

    @pytest.mark.asyncio
    async def test_list_empty(self):
        """Returns empty list when no scans exist."""
        result = await list_quick_scans()

        assert result["scans"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_list_scans(self):
        """Returns list of scan summaries."""
        _quick_scan_results["scan_1"] = {
            "scan_id": "scan_1",
            "repository": "/path/1",
            "status": "completed",
            "started_at": "2025-01-01T00:00:00Z",
            "completed_at": "2025-01-01T00:01:00Z",
            "risk_score": 25.0,
            "findings": [{"id": "1"}, {"id": "2"}],
        }
        _quick_scan_results["scan_2"] = {
            "scan_id": "scan_2",
            "repository": "/path/2",
            "status": "running",
            "started_at": "2025-01-01T00:02:00Z",
            "completed_at": None,
            "risk_score": 0,
            "findings": [],
        }

        result = await list_quick_scans()

        assert result["total"] == 2
        assert len(result["scans"]) == 2

    @pytest.mark.asyncio
    async def test_list_scans_with_pagination(self):
        """Supports limit and offset."""
        for i in range(5):
            _quick_scan_results[f"scan_{i}"] = {
                "scan_id": f"scan_{i}",
                "repository": f"/path/{i}",
                "status": "completed",
                "started_at": f"2025-01-0{i + 1}T00:00:00Z",
                "completed_at": f"2025-01-0{i + 1}T00:01:00Z",
                "risk_score": i * 10,
                "findings": [],
            }

        result = await list_quick_scans(limit=2, offset=1)

        assert result["total"] == 5
        assert len(result["scans"]) == 2
        assert result["limit"] == 2
        assert result["offset"] == 1


# =============================================================================
# Tests: _generate_mock_result
# =============================================================================


class TestGenerateMockResult:
    """Tests for mock result generation."""

    def test_mock_result_structure(self):
        """Mock result has expected structure."""
        start_time = datetime.now(timezone.utc)
        result = _generate_mock_result("test_id", "/test/repo", start_time)

        assert result["scan_id"] == "test_id"
        assert result["repository"] == "/test/repo"
        assert result["status"] == "completed"
        assert result["files_scanned"] > 0
        assert result["lines_scanned"] > 0
        assert "summary" in result
        assert "findings" in result
        assert len(result["findings"]) > 0

    def test_mock_result_findings(self):
        """Mock result includes realistic findings."""
        start_time = datetime.now(timezone.utc)
        result = _generate_mock_result("test_id", "/test/repo", start_time)

        finding = result["findings"][0]
        assert "id" in finding
        assert "title" in finding
        assert "severity" in finding
        assert "file_path" in finding


# =============================================================================
# Tests: QuickScanHandler POST /api/codebase/quick-scan
# =============================================================================


class TestHandlePostQuickScan:
    """Tests for POST quick-scan endpoint."""

    @pytest.mark.asyncio
    async def test_post_unauthenticated(self, handler):
        """Returns 401 when not authenticated."""
        request = _make_mock_request(
            method="POST",
            json_body={"repo_path": "/test/path"},
        )

        result = await handler.handle_post_quick_scan(request)
        body, status = _parse_result(result)

        assert status == 401

    @pytest.mark.asyncio
    async def test_post_no_user_id(self, handler, mock_permission_checker):
        """Returns 401 when auth context has no user_id."""
        auth_ctx = MagicMock()
        auth_ctx.user_id = None

        request = _make_mock_request(
            method="POST",
            json_body={"repo_path": "/test/path"},
            auth_context=auth_ctx,
        )

        with patch(
            "aragora.server.handlers.codebase.quick_scan.get_permission_checker",
            return_value=mock_permission_checker,
        ):
            result = await handler.handle_post_quick_scan(request)

        body, status = _parse_result(result)
        assert status == 401

    @pytest.mark.asyncio
    async def test_post_permission_denied(self, handler, mock_permission_checker_denied):
        """Returns 403 when permission denied."""
        auth_ctx = _make_auth_context()
        request = _make_mock_request(
            method="POST",
            json_body={"repo_path": "/test/path"},
            auth_context=auth_ctx,
        )

        with patch(
            "aragora.server.handlers.codebase.quick_scan.get_permission_checker",
            return_value=mock_permission_checker_denied,
        ):
            result = await handler.handle_post_quick_scan(request)

        body, status = _parse_result(result)
        assert status == 403

    @pytest.mark.asyncio
    async def test_post_missing_repo_path(self, handler, mock_permission_checker):
        """Returns 400 when repo_path is missing."""
        auth_ctx = _make_auth_context()
        request = _make_mock_request(
            method="POST",
            json_body={},
            auth_context=auth_ctx,
        )

        with patch(
            "aragora.server.handlers.codebase.quick_scan.get_permission_checker",
            return_value=mock_permission_checker,
        ):
            result = await handler.handle_post_quick_scan(request)

        body, status = _parse_result(result)
        assert status == 400
        assert "repo_path" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_post_success(self, handler, mock_permission_checker):
        """Successfully runs a quick scan."""
        auth_ctx = _make_auth_context()
        request = _make_mock_request(
            method="POST",
            json_body={"repo_path": "/test/path"},
            auth_context=auth_ctx,
        )

        with (
            patch(
                "aragora.server.handlers.codebase.quick_scan.get_permission_checker",
                return_value=mock_permission_checker,
            ),
            patch(
                "aragora.server.handlers.codebase.quick_scan.run_quick_scan",
                new_callable=AsyncMock,
                return_value={
                    "scan_id": "test_scan",
                    "status": "completed",
                    "repository": "/test/path",
                    "findings": [],
                    "summary": {"critical": 0, "high": 0, "medium": 0, "low": 0},
                },
            ),
        ):
            result = await handler.handle_post_quick_scan(request)

        body, status = _parse_result(result)
        assert status == 200
        assert body["success"] is True
        assert body["scan_id"] == "test_scan"

    @pytest.mark.asyncio
    async def test_post_with_options(self, handler, mock_permission_checker):
        """Passes severity_threshold and include_secrets options."""
        auth_ctx = _make_auth_context()
        request = _make_mock_request(
            method="POST",
            json_body={
                "repo_path": "/test/path",
                "severity_threshold": "high",
                "include_secrets": False,
            },
            auth_context=auth_ctx,
        )

        mock_run = AsyncMock(
            return_value={
                "scan_id": "test_scan",
                "status": "completed",
                "repository": "/test/path",
                "findings": [],
            }
        )

        with (
            patch(
                "aragora.server.handlers.codebase.quick_scan.get_permission_checker",
                return_value=mock_permission_checker,
            ),
            patch(
                "aragora.server.handlers.codebase.quick_scan.run_quick_scan",
                mock_run,
            ),
        ):
            await handler.handle_post_quick_scan(request)

        mock_run.assert_called_once_with(
            repo_path="/test/path",
            severity_threshold="high",
            include_secrets=False,
        )


# =============================================================================
# Tests: QuickScanHandler GET /api/codebase/quick-scan/{scan_id}
# =============================================================================


class TestHandleGetQuickScan:
    """Tests for GET quick-scan by ID endpoint."""

    @pytest.mark.asyncio
    async def test_get_unauthenticated(self, handler):
        """Returns 401 when not authenticated."""
        request = _make_mock_request(
            method="GET",
            match_info={"scan_id": "test_scan"},
        )

        result = await handler.handle_get_quick_scan(request)
        body, status = _parse_result(result)

        assert status == 401

    @pytest.mark.asyncio
    async def test_get_missing_scan_id(self, handler, mock_permission_checker):
        """Returns 400 when scan_id is missing."""
        auth_ctx = _make_auth_context()
        request = _make_mock_request(
            method="GET",
            match_info={},
            auth_context=auth_ctx,
        )

        with patch(
            "aragora.server.handlers.codebase.quick_scan.get_permission_checker",
            return_value=mock_permission_checker,
        ):
            result = await handler.handle_get_quick_scan(request)

        body, status = _parse_result(result)
        assert status == 400

    @pytest.mark.asyncio
    async def test_get_not_found(self, handler, mock_permission_checker):
        """Returns 404 when scan not found."""
        auth_ctx = _make_auth_context()
        request = _make_mock_request(
            method="GET",
            match_info={"scan_id": "nonexistent"},
            auth_context=auth_ctx,
        )

        with patch(
            "aragora.server.handlers.codebase.quick_scan.get_permission_checker",
            return_value=mock_permission_checker,
        ):
            result = await handler.handle_get_quick_scan(request)

        body, status = _parse_result(result)
        assert status == 404

    @pytest.mark.asyncio
    async def test_get_success(self, handler, mock_permission_checker):
        """Successfully retrieves a scan result."""
        _quick_scan_results["test_scan"] = {
            "scan_id": "test_scan",
            "status": "completed",
            "repository": "/test/path",
            "findings": [],
        }

        auth_ctx = _make_auth_context()
        request = _make_mock_request(
            method="GET",
            match_info={"scan_id": "test_scan"},
            auth_context=auth_ctx,
        )

        with patch(
            "aragora.server.handlers.codebase.quick_scan.get_permission_checker",
            return_value=mock_permission_checker,
        ):
            result = await handler.handle_get_quick_scan(request)

        body, status = _parse_result(result)
        assert status == 200
        assert body["success"] is True
        assert body["scan_id"] == "test_scan"


# =============================================================================
# Tests: QuickScanHandler GET /api/codebase/quick-scans
# =============================================================================


class TestHandleListQuickScans:
    """Tests for GET quick-scans list endpoint."""

    @pytest.mark.asyncio
    async def test_list_unauthenticated(self, handler):
        """Returns 401 when not authenticated."""
        request = _make_mock_request(method="GET")

        result = await handler.handle_list_quick_scans(request)
        body, status = _parse_result(result)

        assert status == 401

    @pytest.mark.asyncio
    async def test_list_success(self, handler, mock_permission_checker):
        """Successfully lists scans."""
        auth_ctx = _make_auth_context()
        request = _make_mock_request(
            method="GET",
            query={},
            auth_context=auth_ctx,
        )

        with patch(
            "aragora.server.handlers.codebase.quick_scan.get_permission_checker",
            return_value=mock_permission_checker,
        ):
            result = await handler.handle_list_quick_scans(request)

        body, status = _parse_result(result)
        assert status == 200
        assert body["success"] is True
        assert "scans" in body
        assert "total" in body

    @pytest.mark.asyncio
    async def test_list_with_pagination(self, handler, mock_permission_checker):
        """Supports limit and offset query params."""
        auth_ctx = _make_auth_context()
        request = _make_mock_request(
            method="GET",
            query={"limit": "10", "offset": "5"},
            auth_context=auth_ctx,
        )

        with patch(
            "aragora.server.handlers.codebase.quick_scan.get_permission_checker",
            return_value=mock_permission_checker,
        ):
            result = await handler.handle_list_quick_scans(request)

        body, status = _parse_result(result)
        assert status == 200
        assert body["limit"] == 10
        assert body["offset"] == 5


# =============================================================================
# Tests: Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_post_internal_error(self, handler, mock_permission_checker):
        """Returns 500 on internal error."""
        auth_ctx = _make_auth_context()
        request = _make_mock_request(
            method="POST",
            json_body={"repo_path": "/test/path"},
            auth_context=auth_ctx,
        )

        with (
            patch(
                "aragora.server.handlers.codebase.quick_scan.get_permission_checker",
                return_value=mock_permission_checker,
            ),
            patch(
                "aragora.server.handlers.codebase.quick_scan.run_quick_scan",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Internal error"),
            ),
        ):
            result = await handler.handle_post_quick_scan(request)

        body, status = _parse_result(result)
        assert status == 500

    @pytest.mark.asyncio
    async def test_permission_check_exception(self, handler):
        """Returns 401 when permission check raises exception."""
        auth_ctx = _make_auth_context()
        request = _make_mock_request(
            method="POST",
            json_body={"repo_path": "/test/path"},
            auth_context=auth_ctx,
        )

        with patch(
            "aragora.server.handlers.codebase.quick_scan.get_permission_checker",
            side_effect=RuntimeError("Checker unavailable"),
        ):
            result = await handler.handle_post_quick_scan(request)

        body, status = _parse_result(result)
        assert status == 401


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
