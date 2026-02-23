"""Tests for code review handler (aragora/server/handlers/code_review.py).

Covers all routes and behavior:
- POST /api/v1/code-review/review         - Review code snippet
- POST /api/v1/code-review/diff           - Review diff/patch
- POST /api/v1/code-review/pr             - Review GitHub PR
- GET  /api/v1/code-review/results/{id}   - Get review result by ID
- GET  /api/v1/code-review/history        - Get review history
- POST /api/v1/code-review/security-scan  - Quick security scan

Also tests:
- CodeReviewHandler class routing (_ROUTE_MAP and DYNAMIC_ROUTES)
- Circuit breaker integration
- Rate limiting bypass
- Input validation
- Error handling
- In-memory review result storage
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.code_review import (
    CodeReviewHandler,
    _review_results,
    _review_results_lock,
    get_code_review_circuit_breaker,
    handle_get_review_history,
    handle_get_review_result,
    handle_quick_security_scan,
    handle_review_code,
    handle_review_diff,
    handle_review_pr,
    reset_code_review_circuit_breaker,
    store_review_result,
)
from aragora.server.handlers.base import HandlerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: HandlerResult) -> dict:
    """Extract the JSON body from a HandlerResult."""
    if isinstance(result, HandlerResult):
        if isinstance(result.body, bytes):
            return json.loads(result.body.decode("utf-8"))
        return result.body
    if isinstance(result, dict):
        return result.get("body", result)
    return {}


def _status(result: HandlerResult) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, HandlerResult):
        return result.status_code
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return 200


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------


class MockFinding:
    """Mock code review finding."""

    def __init__(self, category="security", severity="high", message="SQL injection"):
        self.category = category
        self.severity = severity
        self.message = message

    def to_dict(self):
        return {
            "category": self.category,
            "severity": self.severity,
            "message": self.message,
        }


class MockReviewResult:
    """Mock review result from CodeReviewOrchestrator."""

    def __init__(self, result_id="review_001", findings=None):
        self.id = result_id
        self.findings = findings or []

    def to_dict(self):
        return {
            "id": self.id,
            "findings": [f.to_dict() for f in self.findings],
            "summary": "Review completed",
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_circuit_breaker():
    """Reset the module-level code review circuit breaker between tests."""
    reset_code_review_circuit_breaker()
    yield
    reset_code_review_circuit_breaker()


@pytest.fixture(autouse=True)
def _clear_review_results():
    """Clear in-memory review results between tests."""
    with _review_results_lock:
        _review_results.clear()
    yield
    with _review_results_lock:
        _review_results.clear()


@pytest.fixture(autouse=True)
def _disable_rate_limiting():
    """Disable rate limiting entirely during tests."""
    import sys

    _rl_mod = sys.modules["aragora.server.handlers.utils.rate_limit"]
    original = _rl_mod.RATE_LIMITING_DISABLED
    _rl_mod.RATE_LIMITING_DISABLED = True
    yield
    _rl_mod.RATE_LIMITING_DISABLED = original


@pytest.fixture(autouse=True)
def _reset_reviewer_singleton():
    """Reset the module-level code reviewer singleton between tests."""
    import aragora.server.handlers.code_review as mod

    mod._code_reviewer = None
    yield
    mod._code_reviewer = None


@pytest.fixture
def mock_reviewer():
    """Create a fully-mocked CodeReviewOrchestrator."""
    reviewer = AsyncMock()
    reviewer.review_code = AsyncMock(return_value=MockReviewResult("review_001"))
    reviewer.review_diff = AsyncMock(return_value=MockReviewResult("review_002"))
    reviewer.review_pr = AsyncMock(return_value=MockReviewResult("review_003"))
    return reviewer


@pytest.fixture(autouse=True)
def _patch_code_reviewer(mock_reviewer):
    """Patch get_code_reviewer to return the mock reviewer."""
    with patch(
        "aragora.server.handlers.code_review.get_code_reviewer",
        return_value=mock_reviewer,
    ):
        yield


@pytest.fixture
def handler():
    """Create CodeReviewHandler with empty context."""
    return CodeReviewHandler(ctx={})


# ============================================================================
# CodeReviewHandler Routing (can_handle / _ROUTE_MAP / DYNAMIC_ROUTES)
# ============================================================================


class TestCodeReviewHandlerRouting:
    """Verify that the CodeReviewHandler class has correct route definitions."""

    def test_handler_extends_base_handler(self, handler):
        """Handler extends BaseHandler."""
        from aragora.server.handlers.base import BaseHandler

        assert isinstance(handler, BaseHandler)

    def test_handler_init_with_context(self):
        """Handler stores provided context."""
        ctx = {"some_key": "some_value"}
        h = CodeReviewHandler(ctx=ctx)
        assert h.ctx == ctx

    def test_handler_init_without_context(self):
        """Handler defaults to empty dict when no context given."""
        h = CodeReviewHandler()
        assert h.ctx == {}

    def test_route_map_has_review_endpoint(self, handler):
        """_ROUTE_MAP includes POST /api/v1/code-review/review."""
        assert "POST /api/v1/code-review/review" in handler._ROUTE_MAP

    def test_route_map_has_diff_endpoint(self, handler):
        """_ROUTE_MAP includes POST /api/v1/code-review/diff."""
        assert "POST /api/v1/code-review/diff" in handler._ROUTE_MAP

    def test_route_map_has_pr_endpoint(self, handler):
        """_ROUTE_MAP includes POST /api/v1/code-review/pr."""
        assert "POST /api/v1/code-review/pr" in handler._ROUTE_MAP

    def test_route_map_has_history_endpoint(self, handler):
        """_ROUTE_MAP includes GET /api/v1/code-review/history."""
        assert "GET /api/v1/code-review/history" in handler._ROUTE_MAP

    def test_route_map_has_security_scan_endpoint(self, handler):
        """_ROUTE_MAP includes POST /api/v1/code-review/security-scan."""
        assert "POST /api/v1/code-review/security-scan" in handler._ROUTE_MAP

    def test_route_map_count(self, handler):
        """_ROUTE_MAP has exactly 5 entries."""
        assert len(handler._ROUTE_MAP) == 5

    def test_dynamic_routes_has_results_endpoint(self, handler):
        """DYNAMIC_ROUTES includes GET results/{result_id}."""
        key = "GET /api/v1/code-review/results/{result_id}"
        assert key in handler.DYNAMIC_ROUTES

    def test_dynamic_routes_count(self, handler):
        """DYNAMIC_ROUTES has exactly 1 entry."""
        assert len(handler.DYNAMIC_ROUTES) == 1

    def test_routes_list(self, handler):
        """ROUTES list has expected paths."""
        assert "/api/v1/code-review/review" in handler.ROUTES
        assert "/api/v1/code-review/diff" in handler.ROUTES
        assert "/api/v1/code-review/pr" in handler.ROUTES
        assert "/api/v1/code-review/history" in handler.ROUTES
        assert "/api/v1/code-review/security-scan" in handler.ROUTES

    def test_routes_list_count(self, handler):
        """ROUTES list has exactly 5 entries."""
        assert len(handler.ROUTES) == 5

    def test_route_map_functions_are_callable(self, handler):
        """All _ROUTE_MAP values are callable."""
        for route_key, func in handler._ROUTE_MAP.items():
            assert callable(func), f"Route {route_key} is not callable"

    def test_dynamic_routes_functions_are_callable(self, handler):
        """All DYNAMIC_ROUTES values are callable."""
        for route_key, func in handler.DYNAMIC_ROUTES.items():
            assert callable(func), f"Route {route_key} is not callable"


# ============================================================================
# POST /api/v1/code-review/review  (handle_review_code)
# ============================================================================


class TestReviewCode:
    """Tests for handle_review_code."""

    @pytest.mark.asyncio
    async def test_review_code_success(self, mock_reviewer):
        data = {"code": "def hello():\n    print('hi')"}
        result = await handle_review_code(data)
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["data"]["message"] == "Code review completed"
        assert "result" in body["data"]
        assert "result_id" in body["data"]

    @pytest.mark.asyncio
    async def test_review_code_missing_code(self):
        data = {"language": "python"}
        result = await handle_review_code(data)
        assert _status(result) == 400
        assert "code" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_review_code_empty_code(self):
        data = {"code": ""}
        result = await handle_review_code(data)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_review_code_with_language(self, mock_reviewer):
        data = {"code": "print('hi')", "language": "python"}
        result = await handle_review_code(data)
        assert _status(result) == 200
        mock_reviewer.review_code.assert_awaited_once()
        call_kwargs = mock_reviewer.review_code.call_args.kwargs
        assert call_kwargs["language"] == "python"

    @pytest.mark.asyncio
    async def test_review_code_with_file_path(self, mock_reviewer):
        data = {"code": "print('hi')", "file_path": "src/main.py"}
        result = await handle_review_code(data)
        assert _status(result) == 200
        call_kwargs = mock_reviewer.review_code.call_args.kwargs
        assert call_kwargs["file_path"] == "src/main.py"

    @pytest.mark.asyncio
    async def test_review_code_with_review_types(self, mock_reviewer):
        data = {"code": "print('hi')", "review_types": ["security", "performance"]}
        result = await handle_review_code(data)
        assert _status(result) == 200
        call_kwargs = mock_reviewer.review_code.call_args.kwargs
        assert call_kwargs["review_types"] == ["security", "performance"]

    @pytest.mark.asyncio
    async def test_review_code_with_context(self, mock_reviewer):
        data = {"code": "print('hi')", "context": "This is a utility function"}
        result = await handle_review_code(data)
        assert _status(result) == 200
        call_kwargs = mock_reviewer.review_code.call_args.kwargs
        assert call_kwargs["context"] == "This is a utility function"

    @pytest.mark.asyncio
    async def test_review_code_stores_result(self, mock_reviewer):
        data = {"code": "print('hi')"}
        result = await handle_review_code(data)
        assert _status(result) == 200
        result_id = _body(result)["data"]["result_id"]
        assert result_id in _review_results

    @pytest.mark.asyncio
    async def test_review_code_records_circuit_breaker_success(self, mock_reviewer):
        cb = get_code_review_circuit_breaker()
        data = {"code": "print('hi')"}
        await handle_review_code(data)
        # CB should still be closed (able to proceed) after success
        assert cb.can_proceed()

    @pytest.mark.asyncio
    async def test_review_code_connection_error(self, mock_reviewer):
        mock_reviewer.review_code.side_effect = ConnectionError("timeout")
        data = {"code": "print('hi')"}
        result = await handle_review_code(data)
        assert _status(result) == 500
        assert "failed" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_review_code_timeout_error(self, mock_reviewer):
        mock_reviewer.review_code.side_effect = TimeoutError("timed out")
        data = {"code": "print('hi')"}
        result = await handle_review_code(data)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_review_code_runtime_error(self, mock_reviewer):
        mock_reviewer.review_code.side_effect = RuntimeError("internal")
        data = {"code": "print('hi')"}
        result = await handle_review_code(data)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_review_code_value_error(self, mock_reviewer):
        mock_reviewer.review_code.side_effect = ValueError("bad value")
        data = {"code": "print('hi')"}
        result = await handle_review_code(data)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_review_code_os_error(self, mock_reviewer):
        mock_reviewer.review_code.side_effect = OSError("io error")
        data = {"code": "print('hi')"}
        result = await handle_review_code(data)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_review_code_records_circuit_breaker_failure(self, mock_reviewer):
        mock_reviewer.review_code.side_effect = ConnectionError("timeout")
        cb = get_code_review_circuit_breaker()
        data = {"code": "print('hi')"}
        await handle_review_code(data)
        # After one failure CB should still allow calls (threshold=5)
        assert cb.can_proceed()

    @pytest.mark.asyncio
    async def test_review_code_no_data_key(self):
        data = {}
        result = await handle_review_code(data)
        assert _status(result) == 400


# ============================================================================
# POST /api/v1/code-review/diff  (handle_review_diff)
# ============================================================================


class TestReviewDiff:
    """Tests for handle_review_diff."""

    @pytest.mark.asyncio
    async def test_review_diff_success(self, mock_reviewer):
        data = {"diff": "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new"}
        result = await handle_review_diff(data)
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["data"]["message"] == "Diff review completed"
        assert "result" in body["data"]
        assert "result_id" in body["data"]

    @pytest.mark.asyncio
    async def test_review_diff_missing_diff(self):
        data = {"base_branch": "main"}
        result = await handle_review_diff(data)
        assert _status(result) == 400
        assert "diff" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_review_diff_empty_diff(self):
        data = {"diff": ""}
        result = await handle_review_diff(data)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_review_diff_with_branches(self, mock_reviewer):
        data = {"diff": "+ new line", "base_branch": "main", "head_branch": "feature"}
        result = await handle_review_diff(data)
        assert _status(result) == 200
        call_kwargs = mock_reviewer.review_diff.call_args.kwargs
        assert call_kwargs["base_branch"] == "main"
        assert call_kwargs["head_branch"] == "feature"

    @pytest.mark.asyncio
    async def test_review_diff_with_review_types(self, mock_reviewer):
        data = {"diff": "+ new line", "review_types": ["maintainability"]}
        result = await handle_review_diff(data)
        assert _status(result) == 200
        call_kwargs = mock_reviewer.review_diff.call_args.kwargs
        assert call_kwargs["review_types"] == ["maintainability"]

    @pytest.mark.asyncio
    async def test_review_diff_with_context(self, mock_reviewer):
        data = {"diff": "+ new line", "context": "Bug fix for issue #42"}
        result = await handle_review_diff(data)
        assert _status(result) == 200
        call_kwargs = mock_reviewer.review_diff.call_args.kwargs
        assert call_kwargs["context"] == "Bug fix for issue #42"

    @pytest.mark.asyncio
    async def test_review_diff_stores_result(self, mock_reviewer):
        data = {"diff": "+ new line"}
        result = await handle_review_diff(data)
        assert _status(result) == 200
        result_id = _body(result)["data"]["result_id"]
        assert result_id in _review_results

    @pytest.mark.asyncio
    async def test_review_diff_connection_error(self, mock_reviewer):
        mock_reviewer.review_diff.side_effect = ConnectionError("timeout")
        data = {"diff": "+ new line"}
        result = await handle_review_diff(data)
        assert _status(result) == 500
        assert "failed" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_review_diff_timeout_error(self, mock_reviewer):
        mock_reviewer.review_diff.side_effect = TimeoutError("timed out")
        data = {"diff": "+ new line"}
        result = await handle_review_diff(data)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_review_diff_no_data(self):
        result = await handle_review_diff({})
        assert _status(result) == 400


# ============================================================================
# POST /api/v1/code-review/pr  (handle_review_pr)
# ============================================================================


class TestReviewPR:
    """Tests for handle_review_pr."""

    @pytest.mark.asyncio
    async def test_review_pr_success(self, mock_reviewer):
        data = {"pr_url": "https://github.com/owner/repo/pull/123"}
        result = await handle_review_pr(data)
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["data"]["message"] == "PR review completed"
        assert "result" in body["data"]
        assert "result_id" in body["data"]

    @pytest.mark.asyncio
    async def test_review_pr_missing_url(self):
        data = {"review_types": ["security"]}
        result = await handle_review_pr(data)
        assert _status(result) == 400
        assert "pr_url" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_review_pr_empty_url(self):
        data = {"pr_url": ""}
        result = await handle_review_pr(data)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_review_pr_invalid_url_no_github(self):
        data = {"pr_url": "https://gitlab.com/owner/repo/merge_requests/5"}
        result = await handle_review_pr(data)
        assert _status(result) == 400
        assert "invalid" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_review_pr_invalid_url_no_pull(self):
        data = {"pr_url": "https://github.com/owner/repo/issues/5"}
        result = await handle_review_pr(data)
        assert _status(result) == 400
        assert "invalid" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_review_pr_invalid_url_format_message(self):
        data = {"pr_url": "https://example.com/something"}
        result = await handle_review_pr(data)
        assert _status(result) == 400
        body = _body(result)
        assert "github.com/owner/repo/pull/123" in body["error"]

    @pytest.mark.asyncio
    async def test_review_pr_with_review_types(self, mock_reviewer):
        data = {
            "pr_url": "https://github.com/owner/repo/pull/42",
            "review_types": ["security", "test_coverage"],
        }
        result = await handle_review_pr(data)
        assert _status(result) == 200
        call_kwargs = mock_reviewer.review_pr.call_args.kwargs
        assert call_kwargs["review_types"] == ["security", "test_coverage"]

    @pytest.mark.asyncio
    async def test_review_pr_with_post_comments_true(self, mock_reviewer):
        data = {
            "pr_url": "https://github.com/owner/repo/pull/42",
            "post_comments": True,
        }
        result = await handle_review_pr(data)
        assert _status(result) == 200
        call_kwargs = mock_reviewer.review_pr.call_args.kwargs
        assert call_kwargs["post_comments"] is True

    @pytest.mark.asyncio
    async def test_review_pr_post_comments_defaults_false(self, mock_reviewer):
        data = {"pr_url": "https://github.com/owner/repo/pull/42"}
        result = await handle_review_pr(data)
        assert _status(result) == 200
        call_kwargs = mock_reviewer.review_pr.call_args.kwargs
        assert call_kwargs["post_comments"] is False

    @pytest.mark.asyncio
    async def test_review_pr_stores_result(self, mock_reviewer):
        data = {"pr_url": "https://github.com/owner/repo/pull/42"}
        result = await handle_review_pr(data)
        assert _status(result) == 200
        result_id = _body(result)["data"]["result_id"]
        assert result_id in _review_results

    @pytest.mark.asyncio
    async def test_review_pr_connection_error(self, mock_reviewer):
        mock_reviewer.review_pr.side_effect = ConnectionError("network error")
        data = {"pr_url": "https://github.com/owner/repo/pull/42"}
        result = await handle_review_pr(data)
        assert _status(result) == 500
        assert "failed" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_review_pr_timeout_error(self, mock_reviewer):
        mock_reviewer.review_pr.side_effect = TimeoutError("timed out")
        data = {"pr_url": "https://github.com/owner/repo/pull/42"}
        result = await handle_review_pr(data)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_review_pr_no_data(self):
        result = await handle_review_pr({})
        assert _status(result) == 400


# ============================================================================
# GET /api/v1/code-review/results/{result_id}  (handle_get_review_result)
# ============================================================================


class TestGetReviewResult:
    """Tests for handle_get_review_result."""

    @pytest.mark.asyncio
    async def test_get_result_found(self):
        # Pre-populate the store
        with _review_results_lock:
            _review_results["review_001"] = {
                "id": "review_001",
                "findings": [],
                "stored_at": "2026-01-01T00:00:00",
            }

        result = await handle_get_review_result({}, result_id="review_001")
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["data"]["result"]["id"] == "review_001"

    @pytest.mark.asyncio
    async def test_get_result_not_found(self):
        result = await handle_get_review_result({}, result_id="nonexistent_id")
        assert _status(result) == 404
        assert "not found" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_get_result_empty_id(self):
        result = await handle_get_review_result({}, result_id="")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_result_returns_stored_data(self):
        stored_data = {
            "id": "review_xyz",
            "findings": [{"severity": "high", "message": "SQL injection"}],
            "stored_at": "2026-02-01T12:00:00",
        }
        with _review_results_lock:
            _review_results["review_xyz"] = stored_data

        result = await handle_get_review_result({}, result_id="review_xyz")
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["result"]["findings"][0]["severity"] == "high"


# ============================================================================
# GET /api/v1/code-review/history  (handle_get_review_history)
# ============================================================================


class TestGetReviewHistory:
    """Tests for handle_get_review_history."""

    @pytest.mark.asyncio
    async def test_get_history_empty(self):
        result = await handle_get_review_history({})
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["reviews"] == []
        assert body["data"]["total"] == 0

    @pytest.mark.asyncio
    async def test_get_history_with_results(self):
        with _review_results_lock:
            _review_results["r1"] = {"id": "r1", "stored_at": "2026-01-01T00:00:00"}
            _review_results["r2"] = {"id": "r2", "stored_at": "2026-01-02T00:00:00"}

        result = await handle_get_review_history({})
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["total"] == 2
        assert len(body["data"]["reviews"]) == 2

    @pytest.mark.asyncio
    async def test_get_history_sorted_descending(self):
        with _review_results_lock:
            _review_results["r1"] = {"id": "r1", "stored_at": "2026-01-01T00:00:00"}
            _review_results["r2"] = {"id": "r2", "stored_at": "2026-01-03T00:00:00"}
            _review_results["r3"] = {"id": "r3", "stored_at": "2026-01-02T00:00:00"}

        result = await handle_get_review_history({})
        body = _body(result)
        reviews = body["data"]["reviews"]
        # Most recent first
        assert reviews[0]["id"] == "r2"
        assert reviews[1]["id"] == "r3"
        assert reviews[2]["id"] == "r1"

    @pytest.mark.asyncio
    async def test_get_history_with_limit(self):
        with _review_results_lock:
            for i in range(10):
                _review_results[f"r{i}"] = {
                    "id": f"r{i}",
                    "stored_at": f"2026-01-{i+1:02d}T00:00:00",
                }

        result = await handle_get_review_history({"limit": 3})
        body = _body(result)
        assert len(body["data"]["reviews"]) == 3
        assert body["data"]["total"] == 10
        assert body["data"]["limit"] == 3

    @pytest.mark.asyncio
    async def test_get_history_with_offset(self):
        with _review_results_lock:
            for i in range(10):
                _review_results[f"r{i}"] = {
                    "id": f"r{i}",
                    "stored_at": f"2026-01-{i+1:02d}T00:00:00",
                }

        result = await handle_get_review_history({"limit": 3, "offset": 5})
        body = _body(result)
        assert len(body["data"]["reviews"]) == 3
        assert body["data"]["offset"] == 5

    @pytest.mark.asyncio
    async def test_get_history_default_limit(self):
        result = await handle_get_review_history({})
        body = _body(result)
        assert body["data"]["limit"] == 50

    @pytest.mark.asyncio
    async def test_get_history_default_offset(self):
        result = await handle_get_review_history({})
        body = _body(result)
        assert body["data"]["offset"] == 0

    @pytest.mark.asyncio
    async def test_get_history_limit_clamped_min(self):
        result = await handle_get_review_history({"limit": -10})
        body = _body(result)
        assert body["data"]["limit"] >= 1

    @pytest.mark.asyncio
    async def test_get_history_limit_clamped_max(self):
        result = await handle_get_review_history({"limit": 9999})
        body = _body(result)
        assert body["data"]["limit"] <= 500

    @pytest.mark.asyncio
    async def test_get_history_offset_clamped_min(self):
        result = await handle_get_review_history({"offset": -5})
        body = _body(result)
        assert body["data"]["offset"] >= 0

    @pytest.mark.asyncio
    async def test_get_history_offset_beyond_total(self):
        with _review_results_lock:
            _review_results["r1"] = {"id": "r1", "stored_at": "2026-01-01T00:00:00"}

        result = await handle_get_review_history({"offset": 100})
        body = _body(result)
        assert body["data"]["reviews"] == []
        assert body["data"]["total"] == 1


# ============================================================================
# POST /api/v1/code-review/security-scan  (handle_quick_security_scan)
# ============================================================================


class TestQuickSecurityScan:
    """Tests for handle_quick_security_scan."""

    @pytest.mark.asyncio
    async def test_security_scan_success(self, mock_reviewer):
        findings = [
            MockFinding(category="security", severity="critical", message="SQL injection"),
            MockFinding(category="security", severity="high", message="XSS"),
            MockFinding(category="performance", severity="medium", message="N+1 query"),
        ]
        mock_reviewer.review_code = AsyncMock(
            return_value=MockReviewResult("sec_001", findings=findings)
        )

        data = {"code": "user_input = input(); db.execute(user_input)"}
        result = await handle_quick_security_scan(data)
        assert _status(result) == 200
        body = _body(result)
        # Only security findings returned, not performance
        assert body["data"]["total"] == 2
        assert len(body["data"]["findings"]) == 2

    @pytest.mark.asyncio
    async def test_security_scan_severity_summary(self, mock_reviewer):
        findings = [
            MockFinding(category="security", severity="critical"),
            MockFinding(category="security", severity="high"),
            MockFinding(category="security", severity="high"),
            MockFinding(category="security", severity="medium"),
            MockFinding(category="security", severity="low"),
        ]
        mock_reviewer.review_code = AsyncMock(
            return_value=MockReviewResult("sec_002", findings=findings)
        )

        data = {"code": "import os; os.system(user_input)"}
        result = await handle_quick_security_scan(data)
        assert _status(result) == 200
        body = _body(result)
        summary = body["data"]["severity_summary"]
        assert summary["critical"] == 1
        assert summary["high"] == 2
        assert summary["medium"] == 1
        assert summary["low"] == 1

    @pytest.mark.asyncio
    async def test_security_scan_no_security_findings(self, mock_reviewer):
        findings = [
            MockFinding(category="performance", severity="medium", message="slow loop"),
        ]
        mock_reviewer.review_code = AsyncMock(
            return_value=MockReviewResult("sec_003", findings=findings)
        )

        data = {"code": "for i in range(100): pass"}
        result = await handle_quick_security_scan(data)
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["total"] == 0
        assert body["data"]["findings"] == []

    @pytest.mark.asyncio
    async def test_security_scan_missing_code(self):
        data = {"language": "python"}
        result = await handle_quick_security_scan(data)
        assert _status(result) == 400
        assert "code" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_security_scan_empty_code(self):
        data = {"code": ""}
        result = await handle_quick_security_scan(data)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_security_scan_with_language(self, mock_reviewer):
        mock_reviewer.review_code = AsyncMock(
            return_value=MockReviewResult("sec_004", findings=[])
        )
        data = {"code": "print('hi')", "language": "python"}
        result = await handle_quick_security_scan(data)
        assert _status(result) == 200
        call_kwargs = mock_reviewer.review_code.call_args.kwargs
        assert call_kwargs["language"] == "python"
        assert call_kwargs["review_types"] == ["security"]

    @pytest.mark.asyncio
    async def test_security_scan_forces_security_review_type(self, mock_reviewer):
        mock_reviewer.review_code = AsyncMock(
            return_value=MockReviewResult("sec_005", findings=[])
        )
        data = {"code": "print('hi')"}
        await handle_quick_security_scan(data)
        call_kwargs = mock_reviewer.review_code.call_args.kwargs
        assert call_kwargs["review_types"] == ["security"]

    @pytest.mark.asyncio
    async def test_security_scan_connection_error(self, mock_reviewer):
        mock_reviewer.review_code.side_effect = ConnectionError("timeout")
        data = {"code": "print('hi')"}
        result = await handle_quick_security_scan(data)
        assert _status(result) == 500
        assert "failed" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_security_scan_runtime_error(self, mock_reviewer):
        mock_reviewer.review_code.side_effect = RuntimeError("internal")
        data = {"code": "print('hi')"}
        result = await handle_quick_security_scan(data)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_security_scan_no_data(self):
        result = await handle_quick_security_scan({})
        assert _status(result) == 400


# ============================================================================
# Circuit Breaker Integration
# ============================================================================


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker behavior across all write endpoints."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_blocks_review_code(self):
        cb = get_code_review_circuit_breaker()
        # Trip the circuit breaker by recording 5 failures
        for _ in range(5):
            cb.record_failure()

        data = {"code": "print('hi')"}
        result = await handle_review_code(data)
        assert _status(result) == 503
        assert "temporarily unavailable" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_blocks_review_diff(self):
        cb = get_code_review_circuit_breaker()
        for _ in range(5):
            cb.record_failure()

        data = {"diff": "+ new line"}
        result = await handle_review_diff(data)
        assert _status(result) == 503
        assert "temporarily unavailable" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_blocks_review_pr(self):
        cb = get_code_review_circuit_breaker()
        for _ in range(5):
            cb.record_failure()

        data = {"pr_url": "https://github.com/owner/repo/pull/42"}
        result = await handle_review_pr(data)
        assert _status(result) == 503
        assert "temporarily unavailable" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_allows_requests(self, mock_reviewer):
        cb = get_code_review_circuit_breaker()
        assert cb.can_proceed()

        data = {"code": "print('hi')"}
        result = await handle_review_code(data)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_circuit_breaker_reset_allows_requests_again(self, mock_reviewer):
        cb = get_code_review_circuit_breaker()
        for _ in range(5):
            cb.record_failure()
        assert not cb.can_proceed()

        reset_code_review_circuit_breaker()
        assert cb.can_proceed()

        data = {"code": "print('hi')"}
        result = await handle_review_code(data)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_increments(self, mock_reviewer):
        mock_reviewer.review_code.side_effect = ConnectionError("timeout")
        cb = get_code_review_circuit_breaker()

        data = {"code": "print('hi')"}
        # Each failure should increment the counter
        for _ in range(4):
            await handle_review_code(data)
        # After 4 failures, still under threshold of 5
        assert cb.can_proceed()

        # 5th failure should trip the breaker
        await handle_review_code(data)
        assert not cb.can_proceed()

    def test_get_code_review_circuit_breaker_returns_singleton(self):
        cb1 = get_code_review_circuit_breaker()
        cb2 = get_code_review_circuit_breaker()
        assert cb1 is cb2

    def test_circuit_breaker_has_correct_name(self):
        cb = get_code_review_circuit_breaker()
        assert cb.name == "code_review_handler"


# ============================================================================
# store_review_result helper
# ============================================================================


class TestStoreReviewResult:
    """Tests for the store_review_result helper function."""

    def test_store_result_with_to_dict(self):
        review = MockReviewResult("review_100")
        result_id = store_review_result(review)
        assert result_id == "review_100"
        assert "review_100" in _review_results
        assert "stored_at" in _review_results["review_100"]

    def test_store_result_with_plain_dict(self):
        data = {"id": "review_dict", "summary": "Looks good"}
        result_id = store_review_result(data)
        assert result_id == "review_dict"
        assert "review_dict" in _review_results

    def test_store_result_without_id_generates_one(self):
        data = {"summary": "No id provided"}
        result_id = store_review_result(data)
        assert result_id.startswith("review_")
        assert result_id in _review_results

    def test_store_result_adds_stored_at(self):
        data = {"id": "ts_test"}
        store_review_result(data)
        assert "stored_at" in _review_results["ts_test"]

    def test_store_multiple_results(self):
        for i in range(5):
            store_review_result({"id": f"multi_{i}"})
        assert len(_review_results) == 5


# ============================================================================
# Error handling edge cases
# ============================================================================


class TestErrorHandlingEdgeCases:
    """Tests for error handling across all endpoints."""

    @pytest.mark.asyncio
    async def test_review_code_with_all_optional_params(self, mock_reviewer):
        data = {
            "code": "def f(): pass",
            "language": "python",
            "file_path": "src/app.py",
            "review_types": ["security", "performance", "maintainability", "test_coverage"],
            "context": "Main application entry point",
        }
        result = await handle_review_code(data)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_review_diff_with_all_optional_params(self, mock_reviewer):
        data = {
            "diff": "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new",
            "base_branch": "main",
            "head_branch": "feature/new",
            "review_types": ["security"],
            "context": "Fixing critical bug",
        }
        result = await handle_review_diff(data)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_review_pr_with_all_optional_params(self, mock_reviewer):
        data = {
            "pr_url": "https://github.com/owner/repo/pull/999",
            "review_types": ["security", "test_coverage"],
            "post_comments": True,
        }
        result = await handle_review_pr(data)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_review_pr_validates_github_subdomain(self):
        # URL has github.com but is not a PR
        data = {"pr_url": "https://github.com/owner/repo/tree/main"}
        result = await handle_review_pr(data)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_review_pr_accepts_github_enterprise_style(self, mock_reviewer):
        # As long as it contains github.com and /pull/ it should pass
        data = {"pr_url": "https://github.com/my-org/my-repo/pull/1"}
        result = await handle_review_pr(data)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_get_review_history_with_string_limit(self):
        """Limit is cast to int from data dict."""
        with _review_results_lock:
            _review_results["r1"] = {"id": "r1", "stored_at": "2026-01-01T00:00:00"}

        result = await handle_get_review_history({"limit": "2"})
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["limit"] == 2

    @pytest.mark.asyncio
    async def test_get_review_history_with_string_offset(self):
        """Offset is cast to int from data dict."""
        result = await handle_get_review_history({"offset": "5"})
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["offset"] == 5
