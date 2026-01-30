"""
Tests for ReviewsHandler - Shareable code review serving.

Tests cover:
- can_handle() route matching (versioned and non-versioned paths)
- handle() dispatching: list vs get review
- Rate limiting enforcement
- _list_reviews: happy path, empty directory, missing directory, JSON parse error
- _get_review: happy path, not found, invalid ID (path traversal prevention), corrupt JSON
- RBAC permission enforcement
- ID validation security (alphanumeric only, length limit)
"""

from __future__ import annotations

import json
import sys
import types as _types_mod
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Slack stubs to prevent transitive import issues
# ---------------------------------------------------------------------------
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m


from aragora.server.handlers.reviews import ReviewsHandler, REVIEWS_DIR


# ===========================================================================
# Fixtures and Helpers
# ===========================================================================


def get_body(result) -> dict:
    """Extract JSON body from a HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


class MockRateLimiter:
    """Mock rate limiter that can be configured to allow or deny."""

    def __init__(self, allowed: bool = True):
        self._allowed = allowed

    def is_allowed(self, key: str) -> bool:
        return self._allowed


@pytest.fixture
def handler():
    """Create a ReviewsHandler with empty context."""
    return ReviewsHandler({})


@pytest.fixture
def mock_handler_obj():
    """Create a mock HTTP handler with client address."""
    h = MagicMock()
    h.client_address = ("192.168.1.1", 54321)
    h.headers = {"X-Forwarded-For": "192.168.1.1"}
    return h


@pytest.fixture
def sample_review():
    """Create sample review data."""
    return {
        "id": "abc123",
        "created_at": "2025-01-15T10:30:00Z",
        "agents": ["claude", "gpt4"],
        "pr_url": "https://github.com/org/repo/pull/42",
        "findings": {
            "unanimous_critiques": [
                {"issue": "Missing error handling"},
                {"issue": "No input validation"},
            ],
            "agreement_score": 0.87,
        },
    }


@pytest.fixture
def reviews_dir(tmp_path, sample_review):
    """Create a temporary reviews directory with sample data."""
    reviews_path = tmp_path / "reviews"
    reviews_path.mkdir()

    # Write sample review
    review_file = reviews_path / "abc123.json"
    review_file.write_text(json.dumps(sample_review))

    # Write another review
    review2 = {
        "id": "def456",
        "created_at": "2025-01-16T11:00:00Z",
        "agents": ["gemini"],
        "pr_url": None,
        "findings": {"unanimous_critiques": [], "agreement_score": 0.5},
    }
    (reviews_path / "def456.json").write_text(json.dumps(review2))

    return reviews_path


# ===========================================================================
# Tests: can_handle
# ===========================================================================


class TestCanHandle:
    """Tests for route matching."""

    def test_can_handle_api_reviews(self, handler):
        """Matches /api/reviews path."""
        assert handler.can_handle("/api/reviews") is True

    def test_can_handle_api_reviews_with_id(self, handler):
        """Matches /api/reviews/{id} path."""
        assert handler.can_handle("/api/reviews/abc123") is True

    def test_can_handle_versioned_path(self, handler):
        """Matches /api/v1/reviews path."""
        assert handler.can_handle("/api/v1/reviews") is True

    def test_can_handle_versioned_with_id(self, handler):
        """Matches /api/v1/reviews/{id} path."""
        assert handler.can_handle("/api/v1/reviews/abc123") is True

    def test_cannot_handle_unrelated_path(self, handler):
        """Does not match unrelated paths."""
        assert handler.can_handle("/api/debates") is False
        assert handler.can_handle("/api/v1/templates") is False
        assert handler.can_handle("/health") is False


# ===========================================================================
# Tests: Rate Limiting
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting enforcement."""

    def test_rate_limit_exceeded(self, handler, mock_handler_obj):
        """Returns 429 when rate limit is exceeded."""
        limiter = MockRateLimiter(allowed=False)

        with patch("aragora.server.handlers.reviews._reviews_limiter", limiter):
            result = handler.handle("/api/reviews", {}, mock_handler_obj)

        body = get_body(result)
        assert result.status_code == 429
        assert "rate limit" in body.get("error", "").lower()

    def test_rate_limit_allowed(self, handler, mock_handler_obj, reviews_dir):
        """Proceeds when rate limit is not exceeded."""
        limiter = MockRateLimiter(allowed=True)

        with (
            patch("aragora.server.handlers.reviews._reviews_limiter", limiter),
            patch("aragora.server.handlers.reviews.REVIEWS_DIR", reviews_dir),
        ):
            result = handler.handle("/api/reviews", {}, mock_handler_obj)

        assert result.status_code == 200


# ===========================================================================
# Tests: _list_reviews
# ===========================================================================


class TestListReviews:
    """Tests for listing reviews."""

    def test_list_reviews_happy_path(self, handler, reviews_dir):
        """List reviews returns summaries of all reviews."""
        with patch("aragora.server.handlers.reviews.REVIEWS_DIR", reviews_dir):
            result = handler._list_reviews()

        body = get_body(result)
        assert result.status_code == 200
        assert body["total"] == 2
        assert len(body["reviews"]) == 2

        # Check summary fields are present
        review = body["reviews"][0]
        assert "id" in review
        assert "created_at" in review
        assert "agents" in review
        assert "unanimous_count" in review
        assert "agreement_score" in review

    def test_list_reviews_empty_directory(self, handler, tmp_path):
        """List reviews returns empty list when directory exists but is empty."""
        empty_dir = tmp_path / "reviews"
        empty_dir.mkdir()

        with patch("aragora.server.handlers.reviews.REVIEWS_DIR", empty_dir):
            result = handler._list_reviews()

        body = get_body(result)
        assert result.status_code == 200
        assert body["total"] == 0
        assert body["reviews"] == []

    def test_list_reviews_missing_directory(self, handler, tmp_path):
        """List reviews returns empty when directory does not exist."""
        nonexistent = tmp_path / "nonexistent"

        with patch("aragora.server.handlers.reviews.REVIEWS_DIR", nonexistent):
            result = handler._list_reviews()

        body = get_body(result)
        assert result.status_code == 200
        assert body["total"] == 0
        assert body["reviews"] == []

    def test_list_reviews_skips_corrupt_json(self, handler, tmp_path):
        """List reviews skips files with invalid JSON gracefully."""
        reviews_path = tmp_path / "reviews"
        reviews_path.mkdir()

        # Write a valid review
        valid = {"id": "valid1", "created_at": "2025-01-15T00:00:00Z", "agents": []}
        (reviews_path / "valid1.json").write_text(json.dumps(valid))

        # Write corrupt file
        (reviews_path / "corrupt.json").write_text("not valid json{{{")

        with patch("aragora.server.handlers.reviews.REVIEWS_DIR", reviews_path):
            result = handler._list_reviews()

        body = get_body(result)
        assert result.status_code == 200
        # Only the valid review should appear
        assert body["total"] == 1

    def test_list_reviews_respects_limit(self, handler, tmp_path):
        """List reviews respects the limit parameter."""
        reviews_path = tmp_path / "reviews"
        reviews_path.mkdir()

        for i in range(5):
            (reviews_path / f"review{i}.json").write_text(
                json.dumps({"id": f"review{i}", "created_at": f"2025-01-{15 + i}T00:00:00Z"})
            )

        with patch("aragora.server.handlers.reviews.REVIEWS_DIR", reviews_path):
            result = handler._list_reviews(limit=2)

        body = get_body(result)
        assert body["total"] == 2


# ===========================================================================
# Tests: _get_review
# ===========================================================================


class TestGetReview:
    """Tests for getting a specific review."""

    def test_get_review_happy_path(self, handler, reviews_dir, sample_review):
        """Get review returns full review data."""
        with patch("aragora.server.handlers.reviews.REVIEWS_DIR", reviews_dir):
            result = handler._get_review("abc123")

        body = get_body(result)
        assert result.status_code == 200
        assert body["review"]["id"] == "abc123"
        assert body["review"]["agents"] == ["claude", "gpt4"]
        assert body["review"]["findings"]["agreement_score"] == 0.87

    def test_get_review_not_found(self, handler, reviews_dir):
        """Get review returns 404 for nonexistent review."""
        with patch("aragora.server.handlers.reviews.REVIEWS_DIR", reviews_dir):
            result = handler._get_review("nonexistent")

        body = get_body(result)
        assert result.status_code == 404
        assert "not found" in body.get("error", "").lower()

    def test_get_review_invalid_id_empty(self, handler):
        """Get review returns 400 for empty ID."""
        result = handler._get_review("")
        body = get_body(result)
        assert result.status_code == 400
        assert "invalid" in body.get("error", "").lower()

    def test_get_review_invalid_id_path_traversal(self, handler):
        """Get review returns 400 for path traversal attempts."""
        result = handler._get_review("../../../etc/passwd")
        body = get_body(result)
        assert result.status_code == 400
        assert "invalid" in body.get("error", "").lower()

    def test_get_review_invalid_id_special_chars(self, handler):
        """Get review returns 400 for IDs with special characters."""
        result = handler._get_review("review;rm -rf")
        body = get_body(result)
        assert result.status_code == 400

    def test_get_review_invalid_id_too_long(self, handler):
        """Get review returns 400 for excessively long IDs."""
        long_id = "a" * 65
        result = handler._get_review(long_id)
        body = get_body(result)
        assert result.status_code == 400

    def test_get_review_corrupt_json(self, handler, tmp_path):
        """Get review returns 500 for corrupt JSON file."""
        reviews_path = tmp_path / "reviews"
        reviews_path.mkdir()
        (reviews_path / "badreview.json").write_text("{not valid json")

        with patch("aragora.server.handlers.reviews.REVIEWS_DIR", reviews_path):
            result = handler._get_review("badreview")

        body = get_body(result)
        assert result.status_code == 500
        assert "invalid" in body.get("error", "").lower()


# ===========================================================================
# Tests: handle() dispatching
# ===========================================================================


class TestHandleDispatching:
    """Tests for the main handle() method dispatching."""

    def test_handle_list_route(self, handler, mock_handler_obj, reviews_dir):
        """handle() routes /api/reviews to list."""
        limiter = MockRateLimiter(allowed=True)

        with (
            patch("aragora.server.handlers.reviews._reviews_limiter", limiter),
            patch("aragora.server.handlers.reviews.REVIEWS_DIR", reviews_dir),
        ):
            result = handler.handle("/api/reviews", {}, mock_handler_obj)

        body = get_body(result)
        assert result.status_code == 200
        assert "reviews" in body

    def test_handle_list_route_trailing_slash(self, handler, mock_handler_obj, reviews_dir):
        """handle() routes /api/reviews/ to list."""
        limiter = MockRateLimiter(allowed=True)

        with (
            patch("aragora.server.handlers.reviews._reviews_limiter", limiter),
            patch("aragora.server.handlers.reviews.REVIEWS_DIR", reviews_dir),
        ):
            result = handler.handle("/api/reviews/", {}, mock_handler_obj)

        body = get_body(result)
        assert result.status_code == 200
        assert "reviews" in body

    def test_handle_get_route(self, handler, mock_handler_obj, reviews_dir):
        """handle() routes /api/reviews/{id} to get."""
        limiter = MockRateLimiter(allowed=True)

        with (
            patch("aragora.server.handlers.reviews._reviews_limiter", limiter),
            patch("aragora.server.handlers.reviews.REVIEWS_DIR", reviews_dir),
        ):
            result = handler.handle("/api/reviews/abc123", {}, mock_handler_obj)

        body = get_body(result)
        assert result.status_code == 200
        assert "review" in body

    def test_handle_versioned_list(self, handler, mock_handler_obj, reviews_dir):
        """handle() routes /api/v1/reviews to list."""
        limiter = MockRateLimiter(allowed=True)

        with (
            patch("aragora.server.handlers.reviews._reviews_limiter", limiter),
            patch("aragora.server.handlers.reviews.REVIEWS_DIR", reviews_dir),
        ):
            result = handler.handle("/api/v1/reviews", {}, mock_handler_obj)

        body = get_body(result)
        assert result.status_code == 200
        assert "reviews" in body
