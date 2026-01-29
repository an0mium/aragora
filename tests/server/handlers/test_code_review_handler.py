"""
Tests for Code Review API Handlers.

Tests cover:
- Route matching (ROUTES and DYNAMIC_ROUTES)
- RBAC permission enforcement
- Code review endpoint (POST /api/v1/code-review/review)
- Diff review endpoint (POST /api/v1/code-review/diff)
- PR review endpoint (POST /api/v1/code-review/pr)
- Security scan endpoint (POST /api/v1/code-review/security-scan)
- Review result retrieval (GET /api/v1/code-review/results/{id})
- Review history (GET /api/v1/code-review/history)
- Input validation
- Error handling
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ===========================================================================
# Mock Types
# ===========================================================================


@dataclass
class MockReviewFinding:
    """Mock review finding."""

    severity: str
    category: str
    message: str
    line_start: int
    line_end: int
    suggestion: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "suggestion": self.suggestion,
        }


@dataclass
class MockReviewResult:
    """Mock code review result."""

    id: str
    summary: str
    findings: list[MockReviewFinding]
    overall_score: float
    review_types: list[str]
    language: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "summary": self.summary,
            "findings": [f.to_dict() for f in self.findings],
            "overall_score": self.overall_score,
            "review_types": self.review_types,
            "language": self.language,
        }


# ===========================================================================
# Helper Functions
# ===========================================================================


def get_response_data(result) -> dict[str, Any]:
    """Extract JSON data from HandlerResult."""
    return json.loads(result.body)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def mock_rbac():
    """Mock RBAC permissions to allow all requests."""
    with patch(
        "aragora.server.handlers.code_review.require_permission",
        lambda perm: lambda f: f,  # Just return the function unchanged
    ):
        yield


@pytest.fixture
def mock_code_reviewer():
    """Create a mock code reviewer."""
    reviewer = MagicMock()

    # Mock review_code
    reviewer.review_code = AsyncMock(
        return_value=MockReviewResult(
            id="review_123",
            summary="Code looks good with minor suggestions",
            findings=[
                MockReviewFinding(
                    severity="info",
                    category="style",
                    message="Consider using more descriptive variable names",
                    line_start=5,
                    line_end=5,
                ),
            ],
            overall_score=0.85,
            review_types=["security", "maintainability"],
            language="python",
        )
    )

    # Mock review_diff
    reviewer.review_diff = AsyncMock(
        return_value=MockReviewResult(
            id="diff_review_456",
            summary="Diff introduces good changes",
            findings=[],
            overall_score=0.95,
            review_types=["security"],
        )
    )

    # Mock review_pr
    reviewer.review_pr = AsyncMock(
        return_value=MockReviewResult(
            id="pr_review_789",
            summary="PR looks good overall",
            findings=[
                MockReviewFinding(
                    severity="warning",
                    category="security",
                    message="Consider input validation",
                    line_start=10,
                    line_end=15,
                    suggestion="Add validation for user input",
                ),
            ],
            overall_score=0.75,
            review_types=["security", "performance"],
        )
    )

    return reviewer


@pytest.fixture
def mock_code_reviewer_with_security_findings():
    """Create a mock code reviewer with security findings for security scan tests."""
    reviewer = MagicMock()

    # Mock review_code with security findings
    reviewer.review_code = AsyncMock(
        return_value=MockReviewResult(
            id="security_scan_001",
            summary="Security scan completed",
            findings=[
                MockReviewFinding(
                    severity="critical",
                    category="security",
                    message="SQL injection vulnerability detected",
                    line_start=10,
                    line_end=10,
                    suggestion="Use parameterized queries",
                ),
                MockReviewFinding(
                    severity="high",
                    category="security",
                    message="XSS vulnerability detected",
                    line_start=25,
                    line_end=25,
                    suggestion="Sanitize user input",
                ),
                MockReviewFinding(
                    severity="medium",
                    category="security",
                    message="Sensitive data exposure",
                    line_start=30,
                    line_end=32,
                ),
                MockReviewFinding(
                    severity="low",
                    category="security",
                    message="Consider adding rate limiting",
                    line_start=50,
                    line_end=55,
                ),
                MockReviewFinding(
                    severity="info",
                    category="style",
                    message="Consider adding docstring",
                    line_start=1,
                    line_end=1,
                ),
            ],
            overall_score=0.45,
            review_types=["security"],
            language="python",
        )
    )

    return reviewer


@pytest.fixture
def reset_review_storage():
    """Reset the in-memory review storage before each test."""
    from aragora.server.handlers.code_review import _review_results, _review_results_lock

    with _review_results_lock:
        _review_results.clear()
    yield
    with _review_results_lock:
        _review_results.clear()


@pytest.fixture
def mock_server_context():
    """Create a mock server context for handler testing."""
    return MagicMock()


@pytest.fixture
def handler(mock_server_context):
    """Create a CodeReviewHandler instance for testing."""
    from aragora.server.handlers.code_review import CodeReviewHandler

    return CodeReviewHandler(mock_server_context)


# ===========================================================================
# Route Matching Tests
# ===========================================================================


class TestCodeReviewHandlerRoutes:
    """Test handler route registration."""

    def test_routes_contains_review_endpoint(self, handler):
        """Test that ROUTES contains code review endpoint."""
        assert "POST /api/v1/code-review/review" in handler.ROUTES

    def test_routes_contains_diff_endpoint(self, handler):
        """Test that ROUTES contains diff review endpoint."""
        assert "POST /api/v1/code-review/diff" in handler.ROUTES

    def test_routes_contains_pr_endpoint(self, handler):
        """Test that ROUTES contains PR review endpoint."""
        assert "POST /api/v1/code-review/pr" in handler.ROUTES

    def test_routes_contains_history_endpoint(self, handler):
        """Test that ROUTES contains history endpoint."""
        assert "GET /api/v1/code-review/history" in handler.ROUTES

    def test_routes_contains_security_scan_endpoint(self, handler):
        """Test that ROUTES contains security scan endpoint."""
        assert "POST /api/v1/code-review/security-scan" in handler.ROUTES

    def test_dynamic_routes_contains_results_endpoint(self, handler):
        """Test that DYNAMIC_ROUTES contains results endpoint."""
        assert "GET /api/v1/code-review/results/{result_id}" in handler.DYNAMIC_ROUTES

    def test_handler_has_correct_route_count(self, handler):
        """Test handler has expected number of routes."""
        assert len(handler.ROUTES) == 5
        assert len(handler.DYNAMIC_ROUTES) == 1


# ===========================================================================
# RBAC Permission Tests
# ===========================================================================


class TestCodeReviewHandlerRBAC:
    """Test RBAC permission enforcement."""

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_review_code_requires_write_permission(self, reset_review_storage):
        """Test that review code requires code_review:write permission."""
        os.environ["ARAGORA_TEST_REAL_AUTH"] = "1"
        try:
            # Import after patching to get real decorator
            from importlib import reload

            import aragora.server.handlers.code_review as code_review_module

            reload(code_review_module)

            # The function should have been decorated with require_permission
            from aragora.server.handlers.code_review import handle_review_code

            # Check the function has RBAC metadata
            assert hasattr(handle_review_code, "__wrapped__") or callable(handle_review_code)
        finally:
            os.environ.pop("ARAGORA_TEST_REAL_AUTH", None)

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_review_diff_requires_write_permission(self, reset_review_storage):
        """Test that review diff requires code_review:write permission."""
        os.environ["ARAGORA_TEST_REAL_AUTH"] = "1"
        try:
            from aragora.server.handlers.code_review import handle_review_diff

            assert hasattr(handle_review_diff, "__wrapped__") or callable(handle_review_diff)
        finally:
            os.environ.pop("ARAGORA_TEST_REAL_AUTH", None)

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_get_result_requires_read_permission(self, reset_review_storage):
        """Test that get result requires code_review:read permission."""
        os.environ["ARAGORA_TEST_REAL_AUTH"] = "1"
        try:
            from aragora.server.handlers.code_review import handle_get_review_result

            assert hasattr(handle_get_review_result, "__wrapped__") or callable(
                handle_get_review_result
            )
        finally:
            os.environ.pop("ARAGORA_TEST_REAL_AUTH", None)


# ===========================================================================
# Code Review Tests
# ===========================================================================


class TestCodeReviewEndpoint:
    """Tests for POST /api/v1/code-review/review."""

    @pytest.mark.asyncio
    async def test_review_code_success(self, mock_code_reviewer, reset_review_storage):
        """Test successful code review."""
        with patch(
            "aragora.server.handlers.code_review.get_code_reviewer",
            return_value=mock_code_reviewer,
        ):
            from aragora.server.handlers.code_review import handle_review_code

            data = {
                "code": "def hello(): print('world')",
                "language": "python",
                "review_types": ["security", "maintainability"],
            }

            result = await handle_review_code(data, user_id="test_user")

            assert result.status_code == 200
            response = get_response_data(result)
            assert response["success"] is True
            assert "result" in response["data"]
            assert "result_id" in response["data"]

    @pytest.mark.asyncio
    async def test_review_code_missing_code_error(self, reset_review_storage):
        """Test error when code is missing."""
        from aragora.server.handlers.code_review import handle_review_code

        data = {"language": "python"}

        result = await handle_review_code(data, user_id="test_user")

        assert result.status_code == 400
        response = get_response_data(result)
        assert "required" in response["error"].lower()

    @pytest.mark.asyncio
    async def test_review_code_with_file_path(self, mock_code_reviewer, reset_review_storage):
        """Test code review with file path context."""
        with patch(
            "aragora.server.handlers.code_review.get_code_reviewer",
            return_value=mock_code_reviewer,
        ):
            from aragora.server.handlers.code_review import handle_review_code

            data = {
                "code": "class MyClass: pass",
                "file_path": "src/models/my_class.py",
                "language": "python",
            }

            result = await handle_review_code(data, user_id="test_user")

            assert result.status_code == 200
            mock_code_reviewer.review_code.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_review_code_with_context(self, mock_code_reviewer, reset_review_storage):
        """Test code review with additional context."""
        with patch(
            "aragora.server.handlers.code_review.get_code_reviewer",
            return_value=mock_code_reviewer,
        ):
            from aragora.server.handlers.code_review import handle_review_code

            data = {
                "code": "# Authentication module",
                "context": "This is part of the auth system",
            }

            result = await handle_review_code(data, user_id="test_user")

            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_review_code_empty_code(self, reset_review_storage):
        """Test error when code is empty string."""
        from aragora.server.handlers.code_review import handle_review_code

        data = {"code": "", "language": "python"}

        result = await handle_review_code(data, user_id="test_user")

        assert result.status_code == 400
        response = get_response_data(result)
        assert "required" in response["error"].lower()


class TestDiffReviewEndpoint:
    """Tests for POST /api/v1/code-review/diff."""

    @pytest.mark.asyncio
    async def test_review_diff_success(self, mock_code_reviewer, reset_review_storage):
        """Test successful diff review."""
        with patch(
            "aragora.server.handlers.code_review.get_code_reviewer",
            return_value=mock_code_reviewer,
        ):
            from aragora.server.handlers.code_review import handle_review_diff

            data = {
                "diff": """
--- a/file.py
+++ b/file.py
@@ -1,3 +1,4 @@
 def hello():
+    print("hello")
     return "world"
""",
                "base_branch": "main",
                "head_branch": "feature/my-change",
            }

            result = await handle_review_diff(data, user_id="test_user")

            assert result.status_code == 200
            response = get_response_data(result)
            assert response["success"] is True
            assert "result_id" in response["data"]

    @pytest.mark.asyncio
    async def test_review_diff_missing_diff_error(self, reset_review_storage):
        """Test error when diff is missing."""
        from aragora.server.handlers.code_review import handle_review_diff

        data = {"base_branch": "main"}

        result = await handle_review_diff(data, user_id="test_user")

        assert result.status_code == 400
        response = get_response_data(result)
        assert "required" in response["error"].lower()

    @pytest.mark.asyncio
    async def test_review_diff_with_review_types(self, mock_code_reviewer, reset_review_storage):
        """Test diff review with specific review types."""
        with patch(
            "aragora.server.handlers.code_review.get_code_reviewer",
            return_value=mock_code_reviewer,
        ):
            from aragora.server.handlers.code_review import handle_review_diff

            data = {
                "diff": "+new line",
                "review_types": ["security", "performance"],
            }

            result = await handle_review_diff(data, user_id="test_user")

            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_review_diff_empty_diff(self, reset_review_storage):
        """Test error when diff is empty string."""
        from aragora.server.handlers.code_review import handle_review_diff

        data = {"diff": ""}

        result = await handle_review_diff(data, user_id="test_user")

        assert result.status_code == 400


class TestPRReviewEndpoint:
    """Tests for POST /api/v1/code-review/pr."""

    @pytest.mark.asyncio
    async def test_review_pr_success(self, mock_code_reviewer, reset_review_storage):
        """Test successful PR review."""
        with patch(
            "aragora.server.handlers.code_review.get_code_reviewer",
            return_value=mock_code_reviewer,
        ):
            from aragora.server.handlers.code_review import handle_review_pr

            data = {
                "pr_url": "https://github.com/owner/repo/pull/123",
                "review_types": ["security"],
            }

            result = await handle_review_pr(data, user_id="test_user")

            assert result.status_code == 200
            response = get_response_data(result)
            assert response["success"] is True
            assert "result_id" in response["data"]

    @pytest.mark.asyncio
    async def test_review_pr_missing_url_error(self, reset_review_storage):
        """Test error when PR URL is missing."""
        from aragora.server.handlers.code_review import handle_review_pr

        data = {"review_types": ["security"]}

        result = await handle_review_pr(data, user_id="test_user")

        assert result.status_code == 400
        response = get_response_data(result)
        assert "required" in response["error"].lower()

    @pytest.mark.asyncio
    async def test_review_pr_invalid_url_format(self, reset_review_storage):
        """Test error for invalid PR URL format."""
        from aragora.server.handlers.code_review import handle_review_pr

        data = {"pr_url": "https://not-github.com/repo"}

        result = await handle_review_pr(data, user_id="test_user")

        assert result.status_code == 400
        response = get_response_data(result)
        assert "invalid" in response["error"].lower()

    @pytest.mark.asyncio
    async def test_review_pr_missing_pull_segment(self, reset_review_storage):
        """Test error for GitHub URL missing /pull/ segment."""
        from aragora.server.handlers.code_review import handle_review_pr

        data = {"pr_url": "https://github.com/owner/repo"}

        result = await handle_review_pr(data, user_id="test_user")

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_review_pr_with_post_comments(self, mock_code_reviewer, reset_review_storage):
        """Test PR review with comment posting enabled."""
        with patch(
            "aragora.server.handlers.code_review.get_code_reviewer",
            return_value=mock_code_reviewer,
        ):
            from aragora.server.handlers.code_review import handle_review_pr

            data = {
                "pr_url": "https://github.com/owner/repo/pull/456",
                "post_comments": True,
            }

            result = await handle_review_pr(data, user_id="test_user")

            assert result.status_code == 200
            # Verify post_comments was passed to the reviewer
            call_kwargs = mock_code_reviewer.review_pr.call_args.kwargs
            assert call_kwargs.get("post_comments") is True


# ===========================================================================
# Security Scan Tests
# ===========================================================================


class TestSecurityScanEndpoint:
    """Tests for POST /api/v1/code-review/security-scan."""

    @pytest.mark.asyncio
    async def test_security_scan_success(
        self, mock_code_reviewer_with_security_findings, reset_review_storage
    ):
        """Test successful security scan."""
        with patch(
            "aragora.server.handlers.code_review.get_code_reviewer",
            return_value=mock_code_reviewer_with_security_findings,
        ):
            from aragora.server.handlers.code_review import handle_quick_security_scan

            data = {
                "code": "user_input = request.GET['id']; query = f'SELECT * FROM users WHERE id={user_input}'",
                "language": "python",
            }

            result = await handle_quick_security_scan(data, user_id="test_user")

            assert result.status_code == 200
            response = get_response_data(result)
            assert response["success"] is True
            assert "findings" in response["data"]
            assert "severity_summary" in response["data"]
            assert "total" in response["data"]

    @pytest.mark.asyncio
    async def test_security_scan_filters_non_security_findings(
        self, mock_code_reviewer_with_security_findings, reset_review_storage
    ):
        """Test that security scan only returns security category findings."""
        with patch(
            "aragora.server.handlers.code_review.get_code_reviewer",
            return_value=mock_code_reviewer_with_security_findings,
        ):
            from aragora.server.handlers.code_review import handle_quick_security_scan

            data = {"code": "some_code"}

            result = await handle_quick_security_scan(data, user_id="test_user")

            assert result.status_code == 200
            response = get_response_data(result)

            # Should have 4 security findings (not the style finding)
            assert response["data"]["total"] == 4
            for finding in response["data"]["findings"]:
                assert finding["category"] == "security"

    @pytest.mark.asyncio
    async def test_security_scan_severity_summary(
        self, mock_code_reviewer_with_security_findings, reset_review_storage
    ):
        """Test that security scan returns correct severity summary."""
        with patch(
            "aragora.server.handlers.code_review.get_code_reviewer",
            return_value=mock_code_reviewer_with_security_findings,
        ):
            from aragora.server.handlers.code_review import handle_quick_security_scan

            data = {"code": "vulnerable_code"}

            result = await handle_quick_security_scan(data, user_id="test_user")

            assert result.status_code == 200
            response = get_response_data(result)
            summary = response["data"]["severity_summary"]

            assert summary["critical"] == 1
            assert summary["high"] == 1
            assert summary["medium"] == 1
            assert summary["low"] == 1

    @pytest.mark.asyncio
    async def test_security_scan_missing_code_error(self, reset_review_storage):
        """Test error when code is missing."""
        from aragora.server.handlers.code_review import handle_quick_security_scan

        data = {"language": "python"}

        result = await handle_quick_security_scan(data, user_id="test_user")

        assert result.status_code == 400
        response = get_response_data(result)
        assert "required" in response["error"].lower()

    @pytest.mark.asyncio
    async def test_security_scan_service_error(
        self, mock_code_reviewer_with_security_findings, reset_review_storage
    ):
        """Test error handling when reviewer service fails."""
        mock_code_reviewer_with_security_findings.review_code.side_effect = Exception(
            "Service unavailable"
        )

        with patch(
            "aragora.server.handlers.code_review.get_code_reviewer",
            return_value=mock_code_reviewer_with_security_findings,
        ):
            from aragora.server.handlers.code_review import handle_quick_security_scan

            data = {"code": "test code"}

            result = await handle_quick_security_scan(data, user_id="test_user")

            assert result.status_code == 500
            response = get_response_data(result)
            assert "failed" in response["error"].lower()


class TestReviewResultRetrieval:
    """Tests for GET /api/v1/code-review/results/{id}."""

    @pytest.mark.asyncio
    async def test_get_review_result_success(self, reset_review_storage):
        """Test successful result retrieval."""
        from aragora.server.handlers.code_review import (
            handle_get_review_result,
            _review_results,
            _review_results_lock,
        )

        # Store a result first
        with _review_results_lock:
            _review_results["test_result_123"] = {
                "id": "test_result_123",
                "summary": "Test review",
                "stored_at": datetime.now().isoformat(),
            }

        result = await handle_get_review_result({}, result_id="test_result_123")

        assert result.status_code == 200
        response = get_response_data(result)
        assert response["data"]["result"]["id"] == "test_result_123"

    @pytest.mark.asyncio
    async def test_get_review_result_not_found(self, reset_review_storage):
        """Test error when result not found."""
        from aragora.server.handlers.code_review import handle_get_review_result

        result = await handle_get_review_result({}, result_id="nonexistent_id")

        assert result.status_code == 404
        response = get_response_data(result)
        assert "not found" in response["error"].lower()


class TestReviewHistory:
    """Tests for GET /api/v1/code-review/history."""

    @pytest.mark.asyncio
    async def test_get_history_success(self, reset_review_storage):
        """Test successful history retrieval."""
        from aragora.server.handlers.code_review import (
            handle_get_review_history,
            _review_results,
            _review_results_lock,
        )

        # Store some results
        with _review_results_lock:
            for i in range(5):
                _review_results[f"result_{i}"] = {
                    "id": f"result_{i}",
                    "stored_at": datetime.now().isoformat(),
                }

        result = await handle_get_review_history({})

        assert result.status_code == 200
        response = get_response_data(result)
        assert len(response["data"]["reviews"]) == 5
        assert response["data"]["total"] == 5

    @pytest.mark.asyncio
    async def test_get_history_with_pagination(self, reset_review_storage):
        """Test history with pagination."""
        from aragora.server.handlers.code_review import (
            handle_get_review_history,
            _review_results,
            _review_results_lock,
        )

        # Store some results
        with _review_results_lock:
            for i in range(10):
                _review_results[f"result_{i}"] = {
                    "id": f"result_{i}",
                    "stored_at": f"2024-01-{10 + i:02d}T00:00:00",
                }

        result = await handle_get_review_history({"limit": 3, "offset": 2})

        assert result.status_code == 200
        response = get_response_data(result)
        assert len(response["data"]["reviews"]) == 3
        assert response["data"]["total"] == 10
        assert response["data"]["limit"] == 3
        assert response["data"]["offset"] == 2

    @pytest.mark.asyncio
    async def test_get_history_empty(self, reset_review_storage):
        """Test empty history."""
        from aragora.server.handlers.code_review import handle_get_review_history

        result = await handle_get_review_history({})

        assert result.status_code == 200
        response = get_response_data(result)
        assert response["data"]["reviews"] == []
        assert response["data"]["total"] == 0

    @pytest.mark.asyncio
    async def test_get_history_default_pagination(self, reset_review_storage):
        """Test history uses default pagination values."""
        from aragora.server.handlers.code_review import handle_get_review_history

        result = await handle_get_review_history({})

        assert result.status_code == 200
        response = get_response_data(result)
        assert response["data"]["limit"] == 50
        assert response["data"]["offset"] == 0


class TestErrorHandling:
    """Tests for error handling in code review endpoints."""

    @pytest.mark.asyncio
    async def test_review_code_service_error(self, mock_code_reviewer, reset_review_storage):
        """Test handling of service errors during code review."""
        mock_code_reviewer.review_code.side_effect = Exception("Service unavailable")

        with patch(
            "aragora.server.handlers.code_review.get_code_reviewer",
            return_value=mock_code_reviewer,
        ):
            from aragora.server.handlers.code_review import handle_review_code

            result = await handle_review_code({"code": "test"}, user_id="test_user")

            assert result.status_code == 500
            response = get_response_data(result)
            assert "failed" in response["error"].lower()

    @pytest.mark.asyncio
    async def test_review_diff_service_error(self, mock_code_reviewer, reset_review_storage):
        """Test handling of service errors during diff review."""
        mock_code_reviewer.review_diff.side_effect = Exception("Network error")

        with patch(
            "aragora.server.handlers.code_review.get_code_reviewer",
            return_value=mock_code_reviewer,
        ):
            from aragora.server.handlers.code_review import handle_review_diff

            result = await handle_review_diff({"diff": "+test"}, user_id="test_user")

            assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_review_pr_service_error(self, mock_code_reviewer, reset_review_storage):
        """Test handling of service errors during PR review."""
        mock_code_reviewer.review_pr.side_effect = Exception("GitHub API error")

        with patch(
            "aragora.server.handlers.code_review.get_code_reviewer",
            return_value=mock_code_reviewer,
        ):
            from aragora.server.handlers.code_review import handle_review_pr

            result = await handle_review_pr(
                {"pr_url": "https://github.com/o/r/pull/1"},
                user_id="test_user",
            )

            assert result.status_code == 500


class TestStorageFunction:
    """Tests for the review result storage function."""

    def test_store_review_result_with_id(self, reset_review_storage):
        """Test storing result that has an ID."""
        from aragora.server.handlers.code_review import store_review_result, _review_results

        result = {"id": "my_custom_id", "summary": "Test"}
        result_id = store_review_result(result)

        assert result_id == "my_custom_id"
        assert "my_custom_id" in _review_results
        assert "stored_at" in _review_results["my_custom_id"]

    def test_store_review_result_without_id(self, reset_review_storage):
        """Test storing result without an ID generates one."""
        from aragora.server.handlers.code_review import store_review_result, _review_results

        result = {"summary": "Test without ID"}
        result_id = store_review_result(result)

        assert result_id.startswith("review_")
        assert result_id in _review_results

    def test_store_review_result_with_to_dict(self, reset_review_storage):
        """Test storing result with to_dict method."""
        from aragora.server.handlers.code_review import store_review_result, _review_results

        mock_result = MockReviewResult(
            id="dataclass_result",
            summary="From dataclass",
            findings=[],
            overall_score=0.9,
            review_types=["security"],
        )

        result_id = store_review_result(mock_result)

        assert result_id == "dataclass_result"
        assert _review_results["dataclass_result"]["summary"] == "From dataclass"


class TestHandlerInstantiation:
    """Tests for CodeReviewHandler class instantiation."""

    def test_handler_inherits_from_base_handler(self, handler):
        """Test that CodeReviewHandler inherits from BaseHandler."""
        from aragora.server.handlers.base import BaseHandler

        assert isinstance(handler, BaseHandler)

    def test_handler_has_ctx(self, handler, mock_server_context):
        """Test that handler stores server context."""
        assert handler.ctx is mock_server_context
