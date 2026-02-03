"""
Comprehensive tests for aragora.server.handlers.code_review.

Tests cover:
- Code review endpoint (review code snippet)
- Diff review endpoint (review unified diff)
- PR review endpoint (review GitHub PR)
- Review results retrieval
- Review history listing
- Security scan endpoint
- Circuit breaker behavior
- Rate limiting
- Input validation
- Error handling

Target: 35+ tests covering the 6 endpoints in code_review.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import json

import pytest


# ===========================================================================
# Rate Limit Bypass for Testing
# ===========================================================================


def _always_allowed(key: str) -> bool:
    """Always allow requests for testing."""
    return True


@pytest.fixture(autouse=True)
def disable_rate_limits():
    """Disable rate limits for all tests in this module."""
    import sys

    rl_module = sys.modules.get("aragora.server.handlers.utils.rate_limit")
    if rl_module is None:
        yield
        return

    original_is_allowed = {}
    for name, limiter in getattr(rl_module, "_limiters", {}).items():
        original_is_allowed[name] = limiter.is_allowed
        limiter.is_allowed = _always_allowed

    yield

    for name, original in original_is_allowed.items():
        if name in getattr(rl_module, "_limiters", {}):
            rl_module._limiters[name].is_allowed = original


# ===========================================================================
# Mock Classes
# ===========================================================================


@dataclass
class MockFinding:
    """Mock code review finding."""

    id: str
    category: str
    severity: str
    message: str
    line: int | None = None
    file_path: str | None = None
    suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "severity": self.severity,
            "message": self.message,
            "line": self.line,
            "file_path": self.file_path,
            "suggestion": self.suggestion,
        }


@dataclass
class MockReviewResult:
    """Mock code review result."""

    id: str
    status: str = "completed"
    findings: list[MockFinding] = field(default_factory=list)
    summary: str = "Review completed"
    score: float = 0.8
    reviewed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
            "score": self.score,
            "reviewed_at": self.reviewed_at,
        }


class MockCodeReviewer:
    """Mock code review orchestrator."""

    def __init__(self):
        self.review_code_called = False
        self.review_diff_called = False
        self.review_pr_called = False
        self._next_result: MockReviewResult | None = None
        self._should_fail = False

    def set_next_result(self, result: MockReviewResult) -> None:
        self._next_result = result

    def set_should_fail(self, should_fail: bool) -> None:
        self._should_fail = should_fail

    async def review_code(
        self,
        code: str,
        language: str | None = None,
        file_path: str | None = None,
        review_types: list[str] | None = None,
        context: str | None = None,
    ) -> MockReviewResult:
        self.review_code_called = True
        if self._should_fail:
            raise RuntimeError("Review failed")
        if self._next_result:
            return self._next_result
        return MockReviewResult(
            id=f"review_{datetime.now().timestamp()}",
            findings=[
                MockFinding(
                    id="f1",
                    category="security",
                    severity="medium",
                    message="Potential SQL injection",
                    line=10,
                ),
            ],
        )

    async def review_diff(
        self,
        diff: str,
        base_branch: str | None = None,
        head_branch: str | None = None,
        review_types: list[str] | None = None,
        context: str | None = None,
    ) -> MockReviewResult:
        self.review_diff_called = True
        if self._should_fail:
            raise RuntimeError("Diff review failed")
        if self._next_result:
            return self._next_result
        return MockReviewResult(
            id=f"diff_review_{datetime.now().timestamp()}",
            findings=[
                MockFinding(
                    id="d1",
                    category="maintainability",
                    severity="low",
                    message="Consider adding docstring",
                ),
            ],
        )

    async def review_pr(
        self,
        pr_url: str,
        review_types: list[str] | None = None,
        post_comments: bool = False,
    ) -> MockReviewResult:
        self.review_pr_called = True
        if self._should_fail:
            raise RuntimeError("PR review failed")
        if self._next_result:
            return self._next_result
        return MockReviewResult(
            id=f"pr_review_{datetime.now().timestamp()}",
            findings=[
                MockFinding(
                    id="p1",
                    category="performance",
                    severity="high",
                    message="Inefficient loop detected",
                ),
            ],
        )


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_reviewer():
    """Create mock code reviewer."""
    return MockCodeReviewer()


@pytest.fixture
def reset_circuit_breaker():
    """Reset circuit breaker before each test."""
    from aragora.server.handlers.code_review import reset_code_review_circuit_breaker

    reset_code_review_circuit_breaker()
    yield
    reset_code_review_circuit_breaker()


@pytest.fixture
def clear_review_results():
    """Clear stored review results before each test."""
    from aragora.server.handlers.code_review import _review_results, _review_results_lock

    with _review_results_lock:
        _review_results.clear()
    yield
    with _review_results_lock:
        _review_results.clear()


# ===========================================================================
# Code Review Endpoint Tests
# ===========================================================================


class TestReviewCode:
    """Tests for POST /api/v1/code-review/review."""

    @pytest.mark.asyncio
    async def test_review_code_success(self, mock_reviewer, reset_circuit_breaker):
        """Test successful code review."""
        with patch(
            "aragora.server.handlers.code_review.get_code_reviewer",
            return_value=mock_reviewer,
        ):
            from aragora.server.handlers.code_review import handle_review_code

            result = await handle_review_code.__wrapped__.__wrapped__(
                data={"code": "def foo(): pass", "language": "python"},
                user_id="test-user",
            )

            assert result is not None
            body = result.to_dict()["body"]
            data = body.get("data", body)
            assert "result" in data
            assert "result_id" in data
            assert mock_reviewer.review_code_called

    @pytest.mark.asyncio
    async def test_review_code_missing_code(self, reset_circuit_breaker):
        """Test error when code is missing."""
        from aragora.server.handlers.code_review import handle_review_code

        result = await handle_review_code.__wrapped__.__wrapped__(
            data={},
            user_id="test-user",
        )

        assert result.status_code == 400
        body = result.to_dict()["body"]
        assert "code is required" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_review_code_with_context(self, mock_reviewer, reset_circuit_breaker):
        """Test code review with additional context."""
        with patch(
            "aragora.server.handlers.code_review.get_code_reviewer",
            return_value=mock_reviewer,
        ):
            from aragora.server.handlers.code_review import handle_review_code

            result = await handle_review_code.__wrapped__.__wrapped__(
                data={
                    "code": "def process(data): return data",
                    "language": "python",
                    "file_path": "src/process.py",
                    "context": "This processes user input",
                    "review_types": ["security", "performance"],
                },
                user_id="test-user",
            )

            assert result is not None
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_review_code_service_failure(self, mock_reviewer, reset_circuit_breaker):
        """Test handling of service failure."""
        mock_reviewer.set_should_fail(True)
        with patch(
            "aragora.server.handlers.code_review.get_code_reviewer",
            return_value=mock_reviewer,
        ):
            from aragora.server.handlers.code_review import handle_review_code

            result = await handle_review_code.__wrapped__.__wrapped__(
                data={"code": "def foo(): pass"},
                user_id="test-user",
            )

            assert result.status_code == 500


# ===========================================================================
# Diff Review Endpoint Tests
# ===========================================================================


class TestReviewDiff:
    """Tests for POST /api/v1/code-review/diff."""

    @pytest.mark.asyncio
    async def test_review_diff_success(self, mock_reviewer, reset_circuit_breaker):
        """Test successful diff review."""
        with patch(
            "aragora.server.handlers.code_review.get_code_reviewer",
            return_value=mock_reviewer,
        ):
            from aragora.server.handlers.code_review import handle_review_diff

            diff = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,4 @@
 def foo():
+    print("hello")
     pass"""

            result = await handle_review_diff.__wrapped__.__wrapped__(
                data={"diff": diff},
                user_id="test-user",
            )

            assert result is not None
            body = result.to_dict()["body"]
            data = body.get("data", body)
            assert "result" in data
            assert mock_reviewer.review_diff_called

    @pytest.mark.asyncio
    async def test_review_diff_missing_diff(self, reset_circuit_breaker):
        """Test error when diff is missing."""
        from aragora.server.handlers.code_review import handle_review_diff

        result = await handle_review_diff.__wrapped__.__wrapped__(
            data={},
            user_id="test-user",
        )

        assert result.status_code == 400
        body = result.to_dict()["body"]
        assert "diff is required" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_review_diff_with_branches(self, mock_reviewer, reset_circuit_breaker):
        """Test diff review with branch info."""
        with patch(
            "aragora.server.handlers.code_review.get_code_reviewer",
            return_value=mock_reviewer,
        ):
            from aragora.server.handlers.code_review import handle_review_diff

            result = await handle_review_diff.__wrapped__.__wrapped__(
                data={
                    "diff": "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new",
                    "base_branch": "main",
                    "head_branch": "feature/new",
                },
                user_id="test-user",
            )

            assert result.status_code == 200


# ===========================================================================
# PR Review Endpoint Tests
# ===========================================================================


class TestReviewPR:
    """Tests for POST /api/v1/code-review/pr."""

    @pytest.mark.asyncio
    async def test_review_pr_success(self, mock_reviewer, reset_circuit_breaker):
        """Test successful PR review."""
        with patch(
            "aragora.server.handlers.code_review.get_code_reviewer",
            return_value=mock_reviewer,
        ):
            from aragora.server.handlers.code_review import handle_review_pr

            result = await handle_review_pr.__wrapped__.__wrapped__(
                data={"pr_url": "https://github.com/owner/repo/pull/123"},
                user_id="test-user",
            )

            assert result is not None
            body = result.to_dict()["body"]
            data = body.get("data", body)
            assert "result" in data
            assert mock_reviewer.review_pr_called

    @pytest.mark.asyncio
    async def test_review_pr_missing_url(self, reset_circuit_breaker):
        """Test error when pr_url is missing."""
        from aragora.server.handlers.code_review import handle_review_pr

        result = await handle_review_pr.__wrapped__.__wrapped__(
            data={},
            user_id="test-user",
        )

        assert result.status_code == 400
        body = result.to_dict()["body"]
        assert "pr_url is required" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_review_pr_invalid_url(self, reset_circuit_breaker):
        """Test error for invalid PR URL."""
        from aragora.server.handlers.code_review import handle_review_pr

        result = await handle_review_pr.__wrapped__.__wrapped__(
            data={"pr_url": "https://example.com/not-a-pr"},
            user_id="test-user",
        )

        assert result.status_code == 400
        body = result.to_dict()["body"]
        assert "Invalid PR URL" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_review_pr_with_comments(self, mock_reviewer, reset_circuit_breaker):
        """Test PR review with comment posting enabled."""
        with patch(
            "aragora.server.handlers.code_review.get_code_reviewer",
            return_value=mock_reviewer,
        ):
            from aragora.server.handlers.code_review import handle_review_pr

            result = await handle_review_pr.__wrapped__.__wrapped__(
                data={
                    "pr_url": "https://github.com/owner/repo/pull/456",
                    "post_comments": True,
                },
                user_id="test-user",
            )

            assert result.status_code == 200


# ===========================================================================
# Review Results Endpoint Tests
# ===========================================================================


class TestGetReviewResult:
    """Tests for GET /api/v1/code-review/results/{result_id}."""

    @pytest.mark.asyncio
    async def test_get_result_success(self, clear_review_results):
        """Test successful result retrieval."""
        from aragora.server.handlers.code_review import (
            handle_get_review_result,
            store_review_result,
        )

        # Store a result
        result = MockReviewResult(id="test-result-123")
        result_id = store_review_result(result)

        # Retrieve it
        response = await handle_get_review_result.__wrapped__(
            data={},
            result_id=result_id,
            user_id="test-user",
        )

        assert response.status_code == 200
        body = response.to_dict()["body"]
        data = body.get("data", body)
        assert "result" in data
        assert data["result"]["id"] == "test-result-123"

    @pytest.mark.asyncio
    async def test_get_result_not_found(self, clear_review_results):
        """Test error when result not found."""
        from aragora.server.handlers.code_review import handle_get_review_result

        response = await handle_get_review_result.__wrapped__(
            data={},
            result_id="nonexistent-id",
            user_id="test-user",
        )

        assert response.status_code == 404


# ===========================================================================
# Review History Endpoint Tests
# ===========================================================================


class TestGetReviewHistory:
    """Tests for GET /api/v1/code-review/history."""

    @pytest.mark.asyncio
    async def test_get_history_empty(self, clear_review_results):
        """Test empty history."""
        from aragora.server.handlers.code_review import handle_get_review_history

        response = await handle_get_review_history.__wrapped__(
            data={},
            user_id="test-user",
        )

        assert response.status_code == 200
        body = response.to_dict()["body"]
        data = body.get("data", body)
        assert data["reviews"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_get_history_with_results(self, clear_review_results):
        """Test history with results."""
        from aragora.server.handlers.code_review import (
            handle_get_review_history,
            store_review_result,
        )

        # Store some results
        for i in range(5):
            store_review_result(MockReviewResult(id=f"result-{i}"))

        response = await handle_get_review_history.__wrapped__(
            data={},
            user_id="test-user",
        )

        assert response.status_code == 200
        body = response.to_dict()["body"]
        data = body.get("data", body)
        assert len(data["reviews"]) == 5
        assert data["total"] == 5

    @pytest.mark.asyncio
    async def test_get_history_pagination(self, clear_review_results):
        """Test history pagination."""
        from aragora.server.handlers.code_review import (
            handle_get_review_history,
            store_review_result,
        )

        # Store 10 results
        for i in range(10):
            store_review_result(MockReviewResult(id=f"result-{i}"))

        response = await handle_get_review_history.__wrapped__(
            data={"limit": 3, "offset": 2},
            user_id="test-user",
        )

        assert response.status_code == 200
        body = response.to_dict()["body"]
        data = body.get("data", body)
        assert len(data["reviews"]) == 3
        assert data["total"] == 10
        assert data["limit"] == 3
        assert data["offset"] == 2


# ===========================================================================
# Security Scan Endpoint Tests
# ===========================================================================


class TestSecurityScan:
    """Tests for POST /api/v1/code-review/security-scan."""

    @pytest.mark.asyncio
    async def test_security_scan_success(self, mock_reviewer):
        """Test successful security scan."""
        # Set up result with security findings
        mock_reviewer.set_next_result(
            MockReviewResult(
                id="sec-scan-1",
                findings=[
                    MockFinding(
                        id="s1",
                        category="security",
                        severity="high",
                        message="SQL injection vulnerability",
                    ),
                    MockFinding(
                        id="s2",
                        category="security",
                        severity="critical",
                        message="Hardcoded credentials",
                    ),
                    MockFinding(
                        id="m1",
                        category="maintainability",
                        severity="low",
                        message="Missing docstring",
                    ),
                ],
            )
        )

        with patch(
            "aragora.server.handlers.code_review.get_code_reviewer",
            return_value=mock_reviewer,
        ):
            from aragora.server.handlers.code_review import handle_quick_security_scan

            result = await handle_quick_security_scan.__wrapped__(
                data={"code": "password = 'secret123'"},
                user_id="test-user",
            )

            assert result.status_code == 200
            body = result.to_dict()["body"]
            data = body.get("data", body)
            # Should only include security findings
            assert data["total"] == 2
            assert data["severity_summary"]["critical"] == 1
            assert data["severity_summary"]["high"] == 1

    @pytest.mark.asyncio
    async def test_security_scan_missing_code(self):
        """Test error when code is missing."""
        from aragora.server.handlers.code_review import handle_quick_security_scan

        result = await handle_quick_security_scan.__wrapped__(
            data={},
            user_id="test-user",
        )

        assert result.status_code == 400


# ===========================================================================
# Circuit Breaker Tests
# ===========================================================================


class TestCircuitBreaker:
    """Tests for circuit breaker behavior."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self, mock_reviewer, reset_circuit_breaker):
        """Test circuit breaker opens after consecutive failures."""
        mock_reviewer.set_should_fail(True)

        with patch(
            "aragora.server.handlers.code_review.get_code_reviewer",
            return_value=mock_reviewer,
        ):
            from aragora.server.handlers.code_review import (
                get_code_review_circuit_breaker,
                handle_review_code,
            )

            cb = get_code_review_circuit_breaker()

            # Make enough failing requests to open circuit
            for _ in range(6):
                await handle_review_code.__wrapped__.__wrapped__(
                    data={"code": "def foo(): pass"},
                    user_id="test-user",
                )

            # Circuit should be open now
            assert not cb.can_proceed()

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_when_open(self, reset_circuit_breaker):
        """Test requests are blocked when circuit is open."""
        from aragora.server.handlers.code_review import (
            get_code_review_circuit_breaker,
            handle_review_code,
        )

        cb = get_code_review_circuit_breaker()

        # Force circuit open
        for _ in range(10):
            cb.record_failure()

        result = await handle_review_code.__wrapped__.__wrapped__(
            data={"code": "def foo(): pass"},
            user_id="test-user",
        )

        assert result.status_code == 503
        body = result.to_dict()["body"]
        assert "temporarily unavailable" in body.get("error", "").lower()


# ===========================================================================
# Helper Function Tests
# ===========================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_store_review_result(self, clear_review_results):
        """Test result storage."""
        from aragora.server.handlers.code_review import (
            _review_results,
            store_review_result,
        )

        result = MockReviewResult(id="test-123")
        result_id = store_review_result(result)

        assert result_id == "test-123"
        assert "test-123" in _review_results

    def test_get_code_reviewer_singleton(self):
        """Test code reviewer is singleton."""
        with patch("aragora.agents.code_reviewer.CodeReviewOrchestrator") as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            # Reset the singleton
            import aragora.server.handlers.code_review as cr_module

            cr_module._code_reviewer = None

            from aragora.server.handlers.code_review import get_code_reviewer

            first = get_code_reviewer()
            second = get_code_reviewer()

            # Should return same instance
            assert first is second
            # Constructor should only be called once
            assert mock_class.call_count == 1

            # Clean up
            cr_module._code_reviewer = None


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestCodeReviewHandlerClass:
    """Tests for CodeReviewHandler class."""

    def test_routes_defined(self):
        """Test that all routes are defined."""
        from aragora.server.handlers.code_review import CodeReviewHandler

        handler = CodeReviewHandler({})

        assert "POST /api/v1/code-review/review" in handler.ROUTES
        assert "POST /api/v1/code-review/diff" in handler.ROUTES
        assert "POST /api/v1/code-review/pr" in handler.ROUTES
        assert "GET /api/v1/code-review/history" in handler.ROUTES
        assert "POST /api/v1/code-review/security-scan" in handler.ROUTES

    def test_dynamic_routes_defined(self):
        """Test that dynamic routes are defined."""
        from aragora.server.handlers.code_review import CodeReviewHandler

        handler = CodeReviewHandler({})

        assert "GET /api/v1/code-review/results/{result_id}" in handler.DYNAMIC_ROUTES
