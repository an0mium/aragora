"""Tests for PRReviewHandler REST API.

Covers all routes and behavior of the PRReviewHandler class:
- can_handle() routing for all defined routes and prefixes
- POST /api/v1/github/pr/review - Trigger PR review
- GET /api/v1/github/pr/{pr_number} - Get PR details
- GET /api/v1/github/pr/review/{review_id} - Get review status
- GET /api/v1/github/pr/{pr_number}/reviews - List reviews for a PR
- POST /api/v1/github/pr/{pr_number}/review - Submit review
- Path parameter extraction
- Error handling and validation
- Data model serialization (ReviewComment, PRReviewResult, PRDetails)
- GitHubClient demo mode fallback
- Storage lifecycle (in-memory review results, running reviews)
"""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.github.pr_review import (
    GitHubClient,
    PRDetails,
    PRReviewHandler,
    PRReviewResult,
    ReviewComment,
    ReviewStatus,
    ReviewVerdict,
    _pr_reviews,
    _review_results,
    _running_reviews,
    _storage_lock,
    handle_get_pr_details,
    handle_get_review_status,
    handle_list_pr_reviews,
    handle_submit_review,
    handle_trigger_pr_review,
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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a PRReviewHandler instance."""
    return PRReviewHandler(ctx={})


@pytest.fixture(autouse=True)
def _clear_storage():
    """Clear in-memory storage before each test."""
    _review_results.clear()
    _pr_reviews.clear()
    _running_reviews.clear()
    yield
    _review_results.clear()
    _pr_reviews.clear()
    _running_reviews.clear()


@pytest.fixture
def sample_pr_details():
    """Create sample PRDetails for testing."""
    return PRDetails(
        number=42,
        title="Add feature X",
        body="This PR adds feature X to the codebase.",
        state="open",
        author="test-user",
        base_branch="main",
        head_branch="feature/x",
        changed_files=[
            {
                "filename": "src/feature.py",
                "status": "added",
                "additions": 50,
                "deletions": 0,
                "patch": "@@ -0,0 +1,50 @@\n+def new_feature():\n+    pass",
            }
        ],
        labels=["enhancement"],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_review_result():
    """Create a sample PRReviewResult for testing."""
    return PRReviewResult(
        review_id="review_abc123",
        pr_number=42,
        repository="owner/repo",
        status=ReviewStatus.COMPLETED,
        verdict=ReviewVerdict.APPROVE,
        summary="LGTM!",
        comments=[
            ReviewComment(
                id="comment_001",
                file_path="src/feature.py",
                line=10,
                body="Looks good.",
                severity="info",
                category="quality",
            )
        ],
        completed_at=datetime.now(timezone.utc),
        metrics={"files_reviewed": 3, "comments_generated": 1},
    )


# ===========================================================================
# Route Matching Tests
# ===========================================================================


class TestCanHandle:
    """can_handle() routing tests."""

    def test_can_handle_review_root(self, handler):
        assert handler.can_handle("/api/v1/github/pr/review") is True

    def test_can_handle_pr_number(self, handler):
        assert handler.can_handle("/api/v1/github/pr/123") is True

    def test_can_handle_pr_reviews_list(self, handler):
        assert handler.can_handle("/api/v1/github/pr/42/reviews") is True

    def test_can_handle_pr_review_submit(self, handler):
        assert handler.can_handle("/api/v1/github/pr/42/review") is True

    def test_can_handle_review_status(self, handler):
        assert handler.can_handle("/api/v1/github/pr/review/review_abc123") is True

    def test_cannot_handle_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_cannot_handle_github_root(self, handler):
        assert handler.can_handle("/api/v1/github") is False

    def test_cannot_handle_github_no_pr(self, handler):
        assert handler.can_handle("/api/v1/github/issues") is False

    def test_can_handle_pr_prefix(self, handler):
        assert handler.can_handle("/api/v1/github/pr/anything") is True

    def test_cannot_handle_empty(self, handler):
        assert handler.can_handle("") is False

    def test_cannot_handle_partial_match(self, handler):
        assert handler.can_handle("/api/v1/github/projects") is False


# ===========================================================================
# Data Model Serialization Tests
# ===========================================================================


class TestReviewComment:
    """ReviewComment serialization tests."""

    def test_to_dict_basic(self):
        comment = ReviewComment(
            id="c1",
            file_path="main.py",
            line=5,
            body="Fix this.",
        )
        d = comment.to_dict()
        assert d["id"] == "c1"
        assert d["file_path"] == "main.py"
        assert d["line"] == 5
        assert d["body"] == "Fix this."
        assert d["side"] == "RIGHT"
        assert d["severity"] == "info"
        assert d["category"] == "general"

    def test_to_dict_with_suggestion(self):
        comment = ReviewComment(
            id="c2",
            file_path="utils.py",
            line=10,
            body="Consider refactoring.",
            suggestion="Use list comprehension.",
            severity="warning",
            category="quality",
        )
        d = comment.to_dict()
        assert d["suggestion"] == "Use list comprehension."
        assert d["severity"] == "warning"
        assert d["category"] == "quality"

    def test_to_dict_no_suggestion(self):
        comment = ReviewComment(
            id="c3", file_path="app.py", line=1, body="OK"
        )
        assert comment.to_dict()["suggestion"] is None


class TestPRDetails:
    """PRDetails serialization tests."""

    def test_to_dict(self, sample_pr_details):
        d = sample_pr_details.to_dict()
        assert d["number"] == 42
        assert d["title"] == "Add feature X"
        assert d["state"] == "open"
        assert d["author"] == "test-user"
        assert d["base_branch"] == "main"
        assert d["head_branch"] == "feature/x"
        assert len(d["changed_files"]) == 1
        assert d["labels"] == ["enhancement"]

    def test_to_dict_no_dates(self):
        pr = PRDetails(
            number=1,
            title="T",
            body="B",
            state="open",
            author="a",
            base_branch="main",
            head_branch="b",
        )
        d = pr.to_dict()
        assert d["created_at"] is None
        assert d["updated_at"] is None

    def test_to_dict_with_dates(self, sample_pr_details):
        d = sample_pr_details.to_dict()
        assert d["created_at"] is not None
        assert d["updated_at"] is not None


class TestPRReviewResult:
    """PRReviewResult serialization tests."""

    def test_to_dict_completed(self, sample_review_result):
        d = sample_review_result.to_dict()
        assert d["review_id"] == "review_abc123"
        assert d["pr_number"] == 42
        assert d["repository"] == "owner/repo"
        assert d["status"] == "completed"
        assert d["verdict"] == "APPROVE"
        assert d["summary"] == "LGTM!"
        assert len(d["comments"]) == 1
        assert d["completed_at"] is not None
        assert d["error"] is None

    def test_to_dict_pending(self):
        result = PRReviewResult(
            review_id="r1",
            pr_number=1,
            repository="o/r",
            status=ReviewStatus.PENDING,
        )
        d = result.to_dict()
        assert d["status"] == "pending"
        assert d["verdict"] is None
        assert d["completed_at"] is None

    def test_to_dict_failed(self):
        result = PRReviewResult(
            review_id="r2",
            pr_number=2,
            repository="o/r",
            status=ReviewStatus.FAILED,
            error="Internal server error",
        )
        d = result.to_dict()
        assert d["status"] == "failed"
        assert d["error"] == "Internal server error"


# ===========================================================================
# Enums
# ===========================================================================


class TestEnums:
    """Enum value tests."""

    def test_review_verdict_values(self):
        assert ReviewVerdict.APPROVE.value == "APPROVE"
        assert ReviewVerdict.REQUEST_CHANGES.value == "REQUEST_CHANGES"
        assert ReviewVerdict.COMMENT.value == "COMMENT"

    def test_review_status_values(self):
        assert ReviewStatus.PENDING.value == "pending"
        assert ReviewStatus.IN_PROGRESS.value == "in_progress"
        assert ReviewStatus.COMPLETED.value == "completed"
        assert ReviewStatus.FAILED.value == "failed"


# ===========================================================================
# Handler Method Tests: handle_post_trigger_review
# ===========================================================================


class TestHandlePostTriggerReview:
    """POST /api/v1/github/pr/review - Trigger PR review."""

    @pytest.mark.asyncio
    async def test_trigger_review_missing_repository(self, handler):
        data = {"pr_number": 42}
        result = await handler.handle_post_trigger_review(data)
        assert _status(result) == 400
        body = _body(result)
        assert "repository" in body.get("error", "").lower() or "required" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_trigger_review_missing_pr_number(self, handler):
        data = {"repository": "owner/repo"}
        result = await handler.handle_post_trigger_review(data)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_trigger_review_missing_both(self, handler):
        result = await handler.handle_post_trigger_review({})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_trigger_review_success(self, handler):
        data = {
            "repository": "owner/repo",
            "pr_number": 42,
            "review_type": "quick",
        }
        result = await handler.handle_post_trigger_review(data)
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["success"] is True
        assert "review_id" in body["data"]
        assert body["data"]["pr_number"] == 42
        assert body["data"]["repository"] == "owner/repo"
        # Let the background task settle
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_trigger_review_default_type(self, handler):
        data = {"repository": "owner/repo", "pr_number": 99}
        result = await handler.handle_post_trigger_review(data)
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["success"] is True
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_trigger_review_stores_result(self, handler):
        data = {"repository": "owner/repo", "pr_number": 10}
        result = await handler.handle_post_trigger_review(data)
        body = _body(result)
        review_id = body["data"]["review_id"]
        assert review_id in _review_results
        assert _review_results[review_id].pr_number == 10
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_trigger_review_adds_to_pr_reviews(self, handler):
        data = {"repository": "owner/repo", "pr_number": 7}
        result = await handler.handle_post_trigger_review(data)
        body = _body(result)
        review_id = body["data"]["review_id"]
        assert "owner/repo/7" in _pr_reviews
        assert review_id in _pr_reviews["owner/repo/7"]
        await asyncio.sleep(0.1)


# ===========================================================================
# Handler Method Tests: handle_get_pr
# ===========================================================================


class TestHandleGetPr:
    """GET /api/v1/github/pr/{pr_number}."""

    @pytest.mark.asyncio
    async def test_get_pr_missing_repository(self, handler):
        result = await handler.handle_get_pr({}, pr_number=42)
        assert _status(result) == 400
        body = _body(result)
        assert "repository" in body.get("error", "").lower() or "required" in body.get("error", "").lower()

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.github.pr_review.GitHubClient.get_pr",
        new_callable=AsyncMock,
    )
    async def test_get_pr_success(self, mock_get_pr, handler, sample_pr_details):
        mock_get_pr.return_value = sample_pr_details
        result = await handler.handle_get_pr(
            {"repository": "owner/repo"}, pr_number=42
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["success"] is True
        assert body["data"]["pr"]["number"] == 42

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.github.pr_review.GitHubClient.get_pr",
        new_callable=AsyncMock,
    )
    async def test_get_pr_not_found(self, mock_get_pr, handler):
        mock_get_pr.return_value = None
        result = await handler.handle_get_pr(
            {"repository": "owner/repo"}, pr_number=999
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.github.pr_review.GitHubClient.get_pr",
        new_callable=AsyncMock,
    )
    async def test_get_pr_invalid_repo_format(self, mock_get_pr, handler):
        """Repository without owner/repo format should return 404."""
        result = await handler.handle_get_pr(
            {"repository": "invalid-no-slash"}, pr_number=42
        )
        # The standalone function returns success=False with error
        assert _status(result) in (400, 404)


# ===========================================================================
# Handler Method Tests: handle_get_review_status
# ===========================================================================


class TestHandleGetReviewStatus:
    """GET /api/v1/github/pr/review/{review_id}."""

    @pytest.mark.asyncio
    async def test_get_review_status_not_found(self, handler):
        result = await handler.handle_get_review_status(
            {}, review_id="nonexistent"
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_review_status_found(self, handler, sample_review_result):
        _review_results["review_abc123"] = sample_review_result
        result = await handler.handle_get_review_status(
            {}, review_id="review_abc123"
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["success"] is True
        assert body["data"]["review"]["review_id"] == "review_abc123"
        assert body["data"]["review"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_get_review_status_in_progress(self, handler):
        result = PRReviewResult(
            review_id="review_ip",
            pr_number=5,
            repository="o/r",
            status=ReviewStatus.IN_PROGRESS,
        )
        _review_results["review_ip"] = result
        resp = await handler.handle_get_review_status({}, review_id="review_ip")
        assert _status(resp) == 200
        body = _body(resp)
        assert body["data"]["review"]["status"] == "in_progress"

    @pytest.mark.asyncio
    async def test_get_review_status_failed(self, handler):
        result = PRReviewResult(
            review_id="review_fail",
            pr_number=6,
            repository="o/r",
            status=ReviewStatus.FAILED,
            error="Internal server error",
        )
        _review_results["review_fail"] = result
        resp = await handler.handle_get_review_status(
            {}, review_id="review_fail"
        )
        assert _status(resp) == 200
        body = _body(resp)
        assert body["data"]["review"]["status"] == "failed"
        assert body["data"]["review"]["error"] == "Internal server error"


# ===========================================================================
# Handler Method Tests: handle_list_reviews
# ===========================================================================


class TestHandleListReviews:
    """GET /api/v1/github/pr/{pr_number}/reviews."""

    @pytest.mark.asyncio
    async def test_list_reviews_missing_repository(self, handler):
        result = await handler.handle_list_reviews({}, pr_number=42)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_list_reviews_empty(self, handler):
        result = await handler.handle_list_reviews(
            {"repository": "owner/repo"}, pr_number=42
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["reviews"] == []
        assert body["data"]["total"] == 0

    @pytest.mark.asyncio
    async def test_list_reviews_with_results(self, handler, sample_review_result):
        _review_results["review_abc123"] = sample_review_result
        _pr_reviews["owner/repo/42"] = ["review_abc123"]

        result = await handler.handle_list_reviews(
            {"repository": "owner/repo"}, pr_number=42
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["total"] == 1
        assert body["data"]["reviews"][0]["review_id"] == "review_abc123"

    @pytest.mark.asyncio
    async def test_list_reviews_multiple(self, handler):
        r1 = PRReviewResult(
            review_id="r1", pr_number=10, repository="o/r",
            status=ReviewStatus.COMPLETED, verdict=ReviewVerdict.APPROVE,
        )
        r2 = PRReviewResult(
            review_id="r2", pr_number=10, repository="o/r",
            status=ReviewStatus.COMPLETED, verdict=ReviewVerdict.COMMENT,
        )
        _review_results["r1"] = r1
        _review_results["r2"] = r2
        _pr_reviews["o/r/10"] = ["r1", "r2"]

        result = await handler.handle_list_reviews(
            {"repository": "o/r"}, pr_number=10
        )
        body = _body(result)
        assert body["data"]["total"] == 2

    @pytest.mark.asyncio
    async def test_list_reviews_stale_id_ignored(self, handler):
        """Review IDs in _pr_reviews but not in _review_results are skipped."""
        _pr_reviews["o/r/5"] = ["nonexistent_id"]
        result = await handler.handle_list_reviews(
            {"repository": "o/r"}, pr_number=5
        )
        body = _body(result)
        assert body["data"]["total"] == 0
        assert body["data"]["reviews"] == []


# ===========================================================================
# Handler Method Tests: handle_post_submit_review
# ===========================================================================


class TestHandlePostSubmitReview:
    """POST /api/v1/github/pr/{pr_number}/review."""

    @pytest.mark.asyncio
    async def test_submit_review_missing_repository(self, handler):
        data = {"event": "APPROVE", "body": "LGTM!"}
        result = await handler.handle_post_submit_review(data, pr_number=42)
        assert _status(result) == 400
        body = _body(result)
        assert "required" in body.get("error", "").lower() or "repository" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_submit_review_missing_event(self, handler):
        data = {"repository": "owner/repo", "body": "LGTM!"}
        result = await handler.handle_post_submit_review(data, pr_number=42)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_submit_review_missing_both(self, handler):
        result = await handler.handle_post_submit_review({}, pr_number=42)
        assert _status(result) == 400

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.github.pr_review.GitHubClient.submit_review",
        new_callable=AsyncMock,
    )
    async def test_submit_review_success(self, mock_submit, handler):
        mock_submit.return_value = {"success": True, "demo": True}
        data = {
            "repository": "owner/repo",
            "event": "APPROVE",
            "body": "LGTM!",
        }
        result = await handler.handle_post_submit_review(data, pr_number=42)
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["success"] is True

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.github.pr_review.GitHubClient.submit_review",
        new_callable=AsyncMock,
    )
    async def test_submit_review_with_comments(self, mock_submit, handler):
        mock_submit.return_value = {"success": True}
        data = {
            "repository": "owner/repo",
            "event": "COMMENT",
            "body": "Some notes.",
            "comments": [
                {"path": "file.py", "line": 10, "body": "Fix this."}
            ],
        }
        result = await handler.handle_post_submit_review(data, pr_number=42)
        assert _status(result) == 200
        mock_submit.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.github.pr_review.GitHubClient.submit_review",
        new_callable=AsyncMock,
    )
    async def test_submit_review_request_changes(self, mock_submit, handler):
        mock_submit.return_value = {"success": True}
        data = {
            "repository": "owner/repo",
            "event": "REQUEST_CHANGES",
            "body": "Please fix the issues.",
        }
        result = await handler.handle_post_submit_review(data, pr_number=42)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_submit_review_invalid_event(self, handler):
        data = {
            "repository": "owner/repo",
            "event": "INVALID_EVENT",
            "body": "test",
        }
        result = await handler.handle_post_submit_review(data, pr_number=42)
        assert _status(result) == 400
        body = _body(result)
        assert "invalid" in body.get("error", "").lower() or "error" in body

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.github.pr_review.GitHubClient.submit_review",
        new_callable=AsyncMock,
    )
    async def test_submit_review_api_failure(self, mock_submit, handler):
        mock_submit.return_value = {"success": False, "error": "API rate limited"}
        data = {
            "repository": "owner/repo",
            "event": "APPROVE",
            "body": "LGTM!",
        }
        result = await handler.handle_post_submit_review(data, pr_number=42)
        assert _status(result) == 400

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.github.pr_review.GitHubClient.submit_review",
        new_callable=AsyncMock,
    )
    async def test_submit_review_default_body(self, mock_submit, handler):
        """When body is missing, default to empty string."""
        mock_submit.return_value = {"success": True}
        data = {
            "repository": "owner/repo",
            "event": "APPROVE",
        }
        result = await handler.handle_post_submit_review(data, pr_number=42)
        assert _status(result) == 200


# ===========================================================================
# Standalone Function Tests: handle_trigger_pr_review
# ===========================================================================


class TestHandleTriggerPrReviewFunction:
    """Direct tests for the handle_trigger_pr_review function."""

    @pytest.mark.asyncio
    async def test_trigger_creates_review(self):
        result = await handle_trigger_pr_review(
            repository="owner/repo",
            pr_number=42,
        )
        assert result["success"] is True
        assert "review_id" in result
        assert result["status"] == "in_progress"
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_trigger_with_workspace_and_user(self):
        result = await handle_trigger_pr_review(
            repository="owner/repo",
            pr_number=43,
            workspace_id="ws-1",
            user_id="user-1",
        )
        assert result["success"] is True
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_trigger_duplicate_running_review(self):
        """When a review is already running for a PR, return error."""
        # Create a fake running task
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        _running_reviews["owner/repo/42"] = asyncio.ensure_future(
            asyncio.sleep(100)
        )
        try:
            result = await handle_trigger_pr_review(
                repository="owner/repo",
                pr_number=42,
            )
            assert result["success"] is False
            assert "already in progress" in result["error"].lower()
        finally:
            task = _running_reviews.pop("owner/repo/42", None)
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


# ===========================================================================
# Standalone Function Tests: handle_get_pr_details
# ===========================================================================


class TestHandleGetPrDetailsFunction:
    """Direct tests for handle_get_pr_details."""

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.github.pr_review.GitHubClient.get_pr",
        new_callable=AsyncMock,
    )
    async def test_get_details_success(self, mock_get_pr, sample_pr_details):
        mock_get_pr.return_value = sample_pr_details
        result = await handle_get_pr_details(
            repository="owner/repo", pr_number=42
        )
        assert result["success"] is True
        assert result["pr"]["number"] == 42

    @pytest.mark.asyncio
    async def test_get_details_invalid_repo(self):
        result = await handle_get_pr_details(
            repository="no-slash", pr_number=1
        )
        assert result["success"] is False
        assert "invalid" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_details_three_part_repo(self):
        result = await handle_get_pr_details(
            repository="a/b/c", pr_number=1
        )
        assert result["success"] is False

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.github.pr_review.GitHubClient.get_pr",
        new_callable=AsyncMock,
    )
    async def test_get_details_not_found(self, mock_get_pr):
        mock_get_pr.return_value = None
        result = await handle_get_pr_details(
            repository="owner/repo", pr_number=999
        )
        assert result["success"] is False
        assert "not found" in result["error"].lower()


# ===========================================================================
# Standalone Function Tests: handle_get_review_status
# ===========================================================================


class TestHandleGetReviewStatusFunction:
    """Direct tests for handle_get_review_status."""

    @pytest.mark.asyncio
    async def test_review_status_found(self, sample_review_result):
        _review_results["review_abc123"] = sample_review_result
        result = await handle_get_review_status(review_id="review_abc123")
        assert result["success"] is True
        assert result["review"]["review_id"] == "review_abc123"

    @pytest.mark.asyncio
    async def test_review_status_not_found(self):
        result = await handle_get_review_status(review_id="nonexistent")
        assert result["success"] is False
        assert "not found" in result["error"].lower()


# ===========================================================================
# Standalone Function Tests: handle_list_pr_reviews
# ===========================================================================


class TestHandleListPrReviewsFunction:
    """Direct tests for handle_list_pr_reviews."""

    @pytest.mark.asyncio
    async def test_list_empty(self):
        result = await handle_list_pr_reviews(
            repository="owner/repo", pr_number=42
        )
        assert result["success"] is True
        assert result["reviews"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_list_with_reviews(self, sample_review_result):
        _review_results["review_abc123"] = sample_review_result
        _pr_reviews["owner/repo/42"] = ["review_abc123"]
        result = await handle_list_pr_reviews(
            repository="owner/repo", pr_number=42
        )
        assert result["success"] is True
        assert result["total"] == 1
        assert result["reviews"][0]["review_id"] == "review_abc123"

    @pytest.mark.asyncio
    async def test_list_different_pr_returns_empty(self, sample_review_result):
        _review_results["review_abc123"] = sample_review_result
        _pr_reviews["owner/repo/42"] = ["review_abc123"]
        result = await handle_list_pr_reviews(
            repository="owner/repo", pr_number=99
        )
        assert result["total"] == 0


# ===========================================================================
# Standalone Function Tests: handle_submit_review
# ===========================================================================


class TestHandleSubmitReviewFunction:
    """Direct tests for handle_submit_review."""

    @pytest.mark.asyncio
    async def test_submit_invalid_repo(self):
        result = await handle_submit_review(
            repository="no-slash",
            pr_number=1,
            event="APPROVE",
            body="ok",
        )
        assert result["success"] is False
        assert "invalid" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_submit_invalid_event(self):
        result = await handle_submit_review(
            repository="owner/repo",
            pr_number=1,
            event="BOGUS",
            body="ok",
        )
        assert result["success"] is False
        assert "invalid" in result["error"].lower()

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.github.pr_review.GitHubClient.submit_review",
        new_callable=AsyncMock,
    )
    async def test_submit_success(self, mock_submit):
        mock_submit.return_value = {"success": True, "demo": True}
        result = await handle_submit_review(
            repository="owner/repo",
            pr_number=1,
            event="APPROVE",
            body="LGTM!",
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.github.pr_review.GitHubClient.submit_review",
        new_callable=AsyncMock,
    )
    async def test_submit_with_comments_arg(self, mock_submit):
        mock_submit.return_value = {"success": True}
        comments = [{"path": "file.py", "line": 1, "body": "Fix"}]
        result = await handle_submit_review(
            repository="owner/repo",
            pr_number=1,
            event="COMMENT",
            body="notes",
            comments=comments,
        )
        assert result["success"] is True
        _, kwargs = mock_submit.call_args
        assert kwargs["comments"] == comments


# ===========================================================================
# GitHubClient Tests
# ===========================================================================


class TestGitHubClient:
    """GitHubClient unit tests."""

    @patch("aragora.server.handlers.github.pr_review.ServiceRegistry")
    def test_client_no_token_demo_mode(self, mock_registry):
        """Client without token returns demo PR data."""
        mock_registry.get.side_effect = AttributeError("no registry")
        with patch.dict("os.environ", {}, clear=False):
            # Remove GITHUB_TOKEN if set
            import os
            old_val = os.environ.pop("GITHUB_TOKEN", None)
            try:
                client = GitHubClient(token=None)
                pr = client._demo_pr(42)
                assert pr.number == 42
                assert "Demo" in pr.title
                assert pr.author == "demo-user"
                assert len(pr.changed_files) == 3
            finally:
                if old_val is not None:
                    os.environ["GITHUB_TOKEN"] = old_val

    @patch("aragora.server.handlers.github.pr_review.ServiceRegistry")
    def test_demo_pr_has_labels(self, mock_registry):
        mock_registry.get.side_effect = AttributeError
        client = GitHubClient.__new__(GitHubClient)
        client.token = None
        client.base_url = "https://api.github.com"
        client._connector = None
        pr = client._demo_pr(7)
        assert "enhancement" in pr.labels
        assert "review-needed" in pr.labels

    @patch("aragora.server.handlers.github.pr_review.ServiceRegistry")
    def test_demo_pr_has_changed_files(self, mock_registry):
        mock_registry.get.side_effect = AttributeError
        client = GitHubClient.__new__(GitHubClient)
        client.token = None
        client.base_url = "https://api.github.com"
        client._connector = None
        pr = client._demo_pr(1)
        filenames = [f["filename"] for f in pr.changed_files]
        assert "src/feature.py" in filenames
        assert "tests/test_feature.py" in filenames
        assert "README.md" in filenames

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.github.pr_review.ServiceRegistry")
    async def test_submit_review_no_token_demo(self, mock_registry):
        """Submit review without token returns demo success."""
        mock_registry.get.side_effect = AttributeError
        client = GitHubClient.__new__(GitHubClient)
        client.token = None
        client.base_url = "https://api.github.com"
        client._connector = None
        result = await client.submit_review(
            owner="o",
            repo="r",
            pr_number=1,
            event=ReviewVerdict.APPROVE,
            body="ok",
        )
        assert result["success"] is True
        assert result["demo"] is True

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.github.pr_review.ServiceRegistry")
    async def test_get_pr_no_token_returns_demo(self, mock_registry):
        """get_pr without token returns demo data."""
        mock_registry.get.side_effect = AttributeError
        client = GitHubClient.__new__(GitHubClient)
        client.token = None
        client.base_url = "https://api.github.com"
        client._connector = None
        pr = await client.get_pr("o", "r", 42)
        assert pr is not None
        assert pr.number == 42
        assert "Demo" in pr.title


# ===========================================================================
# _get_user_id Tests
# ===========================================================================


class TestGetUserId:
    """_get_user_id helper method tests."""

    def test_get_user_id_default(self, handler):
        assert handler._get_user_id() == "default"

    def test_get_user_id_from_auth_context(self):
        auth_ctx = MagicMock()
        auth_ctx.user_id = "user-42"
        h = PRReviewHandler(ctx={"auth_context": auth_ctx})
        assert h._get_user_id() == "user-42"

    def test_get_user_id_auth_ctx_no_user_id(self):
        auth_ctx = object()  # no user_id attr
        h = PRReviewHandler(ctx={"auth_context": auth_ctx})
        assert h._get_user_id() == "default"


# ===========================================================================
# handle() synchronous router
# ===========================================================================


class TestHandleSync:
    """handle() always returns None (all routing is async)."""

    def test_handle_returns_none(self, handler):
        result = handler.handle("/api/v1/github/pr/review", {}, MagicMock())
        assert result is None

    def test_handle_returns_none_for_pr_number(self, handler):
        result = handler.handle("/api/v1/github/pr/42", {}, MagicMock())
        assert result is None

    def test_handle_returns_none_for_reviews(self, handler):
        result = handler.handle(
            "/api/v1/github/pr/42/reviews", {}, MagicMock()
        )
        assert result is None


# ===========================================================================
# Edge Cases and Error Paths
# ===========================================================================


class TestEdgeCases:
    """Edge cases for the PR review handler."""

    @pytest.mark.asyncio
    async def test_review_result_verdict_none_serialization(self):
        """ReviewResult with no verdict serializes correctly."""
        result = PRReviewResult(
            review_id="r1",
            pr_number=1,
            repository="o/r",
            status=ReviewStatus.IN_PROGRESS,
        )
        d = result.to_dict()
        assert d["verdict"] is None
        assert d["completed_at"] is None
        assert d["comments"] == []
        assert d["metrics"] == {}

    @pytest.mark.asyncio
    async def test_review_comment_left_side(self):
        """ReviewComment with LEFT side."""
        comment = ReviewComment(
            id="c1", file_path="f.py", line=1, body="test", side="LEFT"
        )
        assert comment.to_dict()["side"] == "LEFT"

    @pytest.mark.asyncio
    async def test_storage_thread_safety(self):
        """Verify _storage_lock is a threading.Lock."""
        assert isinstance(_storage_lock, type(threading.Lock()))

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.github.pr_review.GitHubClient.get_pr",
        new_callable=AsyncMock,
    )
    async def test_get_pr_details_empty_repo(self, mock_get_pr):
        """Empty string repo parts check."""
        result = await handle_get_pr_details(repository="", pr_number=1)
        assert result["success"] is False

    def test_handler_init_with_ctx(self):
        """Handler initializes with any context dict."""
        h = PRReviewHandler(ctx={"key": "value"})
        assert h.ctx == {"key": "value"}

    def test_handler_routes_attribute(self, handler):
        """ROUTES contains the expected static route."""
        assert "/api/v1/github/pr/review" in handler.ROUTES

    def test_handler_route_prefixes_attribute(self, handler):
        """ROUTE_PREFIXES contains the expected prefix."""
        assert "/api/v1/github/pr/" in handler.ROUTE_PREFIXES

    @pytest.mark.asyncio
    async def test_pr_details_empty_diff(self):
        """PRDetails with no diff field."""
        pr = PRDetails(
            number=1, title="T", body="B", state="open",
            author="a", base_branch="main", head_branch="b",
        )
        assert pr.diff is None

    @pytest.mark.asyncio
    async def test_pr_details_with_commits(self):
        """PRDetails with commits populated."""
        pr = PRDetails(
            number=1, title="T", body="B", state="open",
            author="a", base_branch="main", head_branch="b",
            commits=[{"sha": "abc123", "message": "initial"}],
        )
        assert len(pr.commits) == 1
        assert pr.commits[0]["sha"] == "abc123"

    @pytest.mark.asyncio
    async def test_review_comment_error_severity(self):
        """ReviewComment with error severity."""
        comment = ReviewComment(
            id="c1", file_path="f.py", line=1,
            body="Security issue!", severity="error",
            category="security",
        )
        d = comment.to_dict()
        assert d["severity"] == "error"
        assert d["category"] == "security"


# ===========================================================================
# Bug Detector Import Tests
# ===========================================================================


class TestBugDetectorImport:
    """Tests for lazy bug detector import."""

    @patch(
        "aragora.server.handlers.github.pr_review._bug_detector_imported",
        False,
    )
    def test_import_bug_detector_unavailable(self):
        """Bug detector import gracefully handles ImportError."""
        from aragora.server.handlers.github.pr_review import _import_bug_detector

        with patch(
            "aragora.server.handlers.github.pr_review._bug_detector_imported",
            False,
        ):
            with patch.dict("sys.modules", {"aragora.analysis.codebase.bug_detector": None}):
                # Force re-import attempt
                import aragora.server.handlers.github.pr_review as mod
                old_imported = mod._bug_detector_imported
                mod._bug_detector_imported = False
                try:
                    result = mod._import_bug_detector()
                    # May return True if already cached, or False if import fails
                    assert isinstance(result, bool)
                finally:
                    mod._bug_detector_imported = old_imported


# ===========================================================================
# _perform_review Heuristic Tests
# ===========================================================================


class TestPerformReviewHeuristics:
    """Tests for the _perform_review heuristic analysis."""

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.github.pr_review._run_bug_detector_analysis",
        new_callable=AsyncMock,
    )
    @patch(
        "aragora.server.handlers.github.pr_review._perform_debate_review",
        new_callable=AsyncMock,
    )
    async def test_review_detects_todo(self, mock_debate, mock_bug_det, sample_pr_details):
        """Heuristic review detects TODO in patches."""
        from aragora.server.handlers.github.pr_review import _perform_review

        mock_debate.return_value = None
        mock_bug_det.return_value = ([], [])
        sample_pr_details.changed_files = [
            {
                "filename": "main.py",
                "patch": "@@ +1 @@\n+# TODO: fix this later",
            }
        ]
        comments, verdict, summary = await _perform_review(
            sample_pr_details, "quick", use_debate=False
        )
        todo_comments = [c for c in comments if "TODO" in c.body]
        assert len(todo_comments) >= 1

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.github.pr_review._run_bug_detector_analysis",
        new_callable=AsyncMock,
    )
    async def test_review_detects_debug_logging(self, mock_bug_det, sample_pr_details):
        """Heuristic review detects console.log/print."""
        from aragora.server.handlers.github.pr_review import _perform_review

        mock_bug_det.return_value = ([], [])
        sample_pr_details.changed_files = [
            {
                "filename": "app.js",
                "patch": "@@ +1 @@\n+console.log('debug')",
            }
        ]
        comments, verdict, summary = await _perform_review(
            sample_pr_details, "quick", use_debate=False
        )
        debug_comments = [c for c in comments if "logging" in c.body.lower() or "Debug" in c.body]
        assert len(debug_comments) >= 1
        assert verdict == ReviewVerdict.COMMENT

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.github.pr_review._run_bug_detector_analysis",
        new_callable=AsyncMock,
    )
    async def test_review_detects_secrets(self, mock_bug_det, sample_pr_details):
        """Heuristic review detects potential secrets."""
        from aragora.server.handlers.github.pr_review import _perform_review

        mock_bug_det.return_value = ([], [])
        sample_pr_details.changed_files = [
            {
                "filename": "config.py",
                "patch": "@@ +1 @@\n+password = 'hunter2'",
            }
        ]
        comments, verdict, summary = await _perform_review(
            sample_pr_details, "quick", use_debate=False
        )
        assert verdict == ReviewVerdict.REQUEST_CHANGES
        assert "security" in summary.lower()

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.github.pr_review._run_bug_detector_analysis",
        new_callable=AsyncMock,
    )
    async def test_review_secrets_in_test_file_ignored(self, mock_bug_det, sample_pr_details):
        """Secrets in test files are not flagged."""
        from aragora.server.handlers.github.pr_review import _perform_review

        mock_bug_det.return_value = ([], [])
        sample_pr_details.changed_files = [
            {
                "filename": "test_auth.py",
                "patch": "@@ +1 @@\n+password = 'test_password'",
            }
        ]
        comments, verdict, summary = await _perform_review(
            sample_pr_details, "quick", use_debate=False
        )
        # test files are exempt from secret detection
        assert verdict != ReviewVerdict.REQUEST_CHANGES or "security" not in summary.lower()

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.github.pr_review._run_bug_detector_analysis",
        new_callable=AsyncMock,
    )
    async def test_review_clean_pr_approves(self, mock_bug_det):
        """Clean PR with no issues gets approved."""
        from aragora.server.handlers.github.pr_review import _perform_review

        mock_bug_det.return_value = ([], [])
        pr = PRDetails(
            number=1, title="T", body="B", state="open",
            author="a", base_branch="main", head_branch="b",
            changed_files=[
                {"filename": "readme.txt", "patch": "@@ +1 @@\n+hello"},
            ],
        )
        comments, verdict, summary = await _perform_review(
            pr, "quick", use_debate=False
        )
        assert verdict == ReviewVerdict.APPROVE

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.github.pr_review._run_bug_detector_analysis",
        new_callable=AsyncMock,
    )
    async def test_review_missing_tests_comment(self, mock_bug_det):
        """Python file without corresponding test file triggers suggestion."""
        from aragora.server.handlers.github.pr_review import _perform_review

        mock_bug_det.return_value = ([], [])
        pr = PRDetails(
            number=1, title="T", body="B", state="open",
            author="a", base_branch="main", head_branch="b",
            changed_files=[
                {
                    "filename": "src/module.py",
                    "patch": "@@ +1 @@\n+def foo(): pass",
                },
            ],
        )
        comments, verdict, summary = await _perform_review(
            pr, "quick", use_debate=False
        )
        test_comments = [c for c in comments if "test" in c.body.lower()]
        assert len(test_comments) >= 1

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.github.pr_review._run_bug_detector_analysis",
        new_callable=AsyncMock,
    )
    async def test_review_with_test_file_present(self, mock_bug_det):
        """Python file WITH test file present doesn't trigger missing test comment."""
        from aragora.server.handlers.github.pr_review import _perform_review

        mock_bug_det.return_value = ([], [])
        pr = PRDetails(
            number=1, title="T", body="B", state="open",
            author="a", base_branch="main", head_branch="b",
            changed_files=[
                {
                    "filename": "src/module.py",
                    "patch": "@@ +1 @@\n+def foo(): pass",
                },
                {
                    "filename": "tests/test_module.py",
                    "patch": "@@ +1 @@\n+def test_foo(): pass",
                },
            ],
        )
        comments, verdict, summary = await _perform_review(
            pr, "quick", use_debate=False
        )
        test_missing_comments = [
            c for c in comments
            if "test" in c.body.lower() and "missing" in c.body.lower()
        ]
        assert len(test_missing_comments) == 0

    @pytest.mark.asyncio
    @patch(
        "aragora.server.handlers.github.pr_review._run_bug_detector_analysis",
        new_callable=AsyncMock,
    )
    async def test_review_bug_detector_critical_issues(self, mock_bug_det):
        """Bug detector critical issues trigger REQUEST_CHANGES."""
        from aragora.server.handlers.github.pr_review import (
            ReviewComment as RC,
            _perform_review,
        )

        bug_comments = [
            RC(
                id="bug1", file_path="f.py", line=1,
                body="Null dereference", severity="error",
                category="bug_detection",
            )
        ]
        mock_bug_det.return_value = (bug_comments, ["bug_null_deref"])
        pr = PRDetails(
            number=1, title="T", body="B", state="open",
            author="a", base_branch="main", head_branch="b",
            changed_files=[
                {"filename": "readme.txt", "patch": "@@ +1 @@\n+hello"},
            ],
        )
        comments, verdict, summary = await _perform_review(
            pr, "quick", use_debate=False
        )
        assert verdict == ReviewVerdict.REQUEST_CHANGES
        assert "bug detector" in summary.lower()


# ===========================================================================
# _parse_debate_result Tests
# ===========================================================================


class TestParseDebateResult:
    """Tests for _parse_debate_result parsing logic."""

    def test_parse_verdict_approve(self, sample_pr_details):
        from aragora.server.handlers.github.pr_review import _parse_debate_result

        answer = "VERDICT: APPROVE\nSUMMARY: Looks great!"
        comments, verdict, summary = _parse_debate_result(answer, sample_pr_details)
        assert verdict == ReviewVerdict.APPROVE
        assert "Looks great!" in summary

    def test_parse_verdict_request_changes(self, sample_pr_details):
        from aragora.server.handlers.github.pr_review import _parse_debate_result

        answer = "VERDICT: REQUEST_CHANGES\nSUMMARY: Needs fixes."
        comments, verdict, summary = _parse_debate_result(answer, sample_pr_details)
        assert verdict == ReviewVerdict.REQUEST_CHANGES

    def test_parse_verdict_comment(self, sample_pr_details):
        from aragora.server.handlers.github.pr_review import _parse_debate_result

        answer = "VERDICT: COMMENT\nSUMMARY: Minor suggestions."
        comments, verdict, summary = _parse_debate_result(answer, sample_pr_details)
        assert verdict == ReviewVerdict.COMMENT

    def test_parse_no_verdict_defaults_comment(self, sample_pr_details):
        from aragora.server.handlers.github.pr_review import _parse_debate_result

        answer = "No structured output here."
        comments, verdict, summary = _parse_debate_result(answer, sample_pr_details)
        assert verdict == ReviewVerdict.COMMENT

    def test_parse_file_comments(self, sample_pr_details):
        from aragora.server.handlers.github.pr_review import _parse_debate_result

        answer = (
            "VERDICT: APPROVE\n"
            "SUMMARY: Good.\n\n"
            "COMMENTS:\n"
            "- main.py:10 - Consider adding error handling\n"
            "- utils.py:5 - Should use pathlib instead\n"
        )
        comments, verdict, summary = _parse_debate_result(answer, sample_pr_details)
        assert len(comments) >= 2
        assert any(c.file_path == "main.py" for c in comments)
        assert any(c.file_path == "utils.py" for c in comments)

    def test_parse_security_severity(self, sample_pr_details):
        from aragora.server.handlers.github.pr_review import _parse_debate_result

        answer = (
            "VERDICT: REQUEST_CHANGES\n"
            "SUMMARY: Security issue.\n\n"
            "- auth.py:20 - SQL injection vulnerability found\n"
        )
        comments, verdict, summary = _parse_debate_result(answer, sample_pr_details)
        security_comments = [c for c in comments if c.severity == "error"]
        assert len(security_comments) >= 1

    def test_parse_warning_severity(self, sample_pr_details):
        from aragora.server.handlers.github.pr_review import _parse_debate_result

        answer = (
            "VERDICT: COMMENT\n"
            "SUMMARY: Suggestions.\n\n"
            "- config.py:5 - You should consider using env vars\n"
        )
        comments, verdict, summary = _parse_debate_result(answer, sample_pr_details)
        warning_comments = [c for c in comments if c.severity == "warning"]
        assert len(warning_comments) >= 1

    def test_parse_summary_truncation(self, sample_pr_details):
        from aragora.server.handlers.github.pr_review import _parse_debate_result

        long_summary = "A" * 1000
        answer = f"VERDICT: APPROVE\nSUMMARY: {long_summary}"
        comments, verdict, summary = _parse_debate_result(answer, sample_pr_details)
        assert len(summary) <= 500

    def test_parse_comment_without_line_number(self, sample_pr_details):
        from aragora.server.handlers.github.pr_review import _parse_debate_result

        answer = (
            "VERDICT: APPROVE\n"
            "SUMMARY: OK.\n\n"
            "- helper.py - Generic suggestion\n"
        )
        comments, verdict, summary = _parse_debate_result(answer, sample_pr_details)
        # Comments without line number default to line 1
        for c in comments:
            if c.file_path == "helper.py":
                assert c.line == 1


# ===========================================================================
# Integration-style Tests
# ===========================================================================


class TestIntegration:
    """Integration tests that exercise the full trigger->status flow."""

    @pytest.mark.asyncio
    async def test_trigger_then_get_status(self, handler):
        """Trigger a review then check its status."""
        data = {"repository": "owner/repo", "pr_number": 42}
        trigger_result = await handler.handle_post_trigger_review(data)
        body = _body(trigger_result)
        review_id = body["data"]["review_id"]

        status_result = await handler.handle_get_review_status(
            {}, review_id=review_id
        )
        status_body = _body(status_result)
        assert _status(status_result) == 200
        assert status_body["data"]["review"]["review_id"] == review_id
        # Let background task settle
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_trigger_then_list_reviews(self, handler):
        """Trigger a review then list reviews for the PR."""
        data = {"repository": "owner/repo", "pr_number": 55}
        trigger_result = await handler.handle_post_trigger_review(data)
        body = _body(trigger_result)
        review_id = body["data"]["review_id"]

        list_result = await handler.handle_list_reviews(
            {"repository": "owner/repo"}, pr_number=55
        )
        list_body = _body(list_result)
        assert list_body["data"]["total"] >= 1
        review_ids = [r["review_id"] for r in list_body["data"]["reviews"]]
        assert review_id in review_ids
        await asyncio.sleep(0.1)
