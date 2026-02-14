"""
Tests for aragora.server.handlers.reviews - Reviews HTTP Handlers.

Tests cover:
- ReviewsHandler: instantiation, ROUTES, can_handle
- GET /api/reviews: list reviews, empty directory, no directory
- GET /api/reviews/{id}: found, not found, invalid ID, corrupt file
- handle routing: rate limiting, auth, permission checks
- Version prefix stripping (/api/v1/reviews)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.reviews import ReviewsHandler
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helpers
# ===========================================================================


def _parse_body(result: HandlerResult) -> dict[str, Any]:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body)


def _make_mock_handler(
    method: str = "GET",
    body: bytes = b"",
    content_type: str = "application/json",
) -> MagicMock:
    """Create a mock HTTP handler object."""
    handler = MagicMock()
    handler.command = method
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": content_type,
        "Host": "localhost:8080",
        "Authorization": "Bearer test-token",
    }
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = body
    return handler


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the rate limiter between tests."""
    from aragora.server.handlers.reviews import _reviews_limiter

    _reviews_limiter._buckets.clear()


@pytest.fixture
def handler():
    """Create a ReviewsHandler."""
    return ReviewsHandler(ctx={})


@pytest.fixture
def tmp_reviews_dir(tmp_path):
    """Create a temporary reviews directory with sample data."""
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()

    # Create sample review files
    review_1 = {
        "id": "abc123",
        "created_at": "2026-02-14T10:00:00Z",
        "agents": ["claude", "gpt-4"],
        "pr_url": "https://github.com/org/repo/pull/42",
        "findings": {
            "unanimous_critiques": [
                {"message": "Missing error handling"},
            ],
            "agreement_score": 0.85,
        },
    }
    review_2 = {
        "id": "def456",
        "created_at": "2026-02-14T11:00:00Z",
        "agents": ["claude"],
        "pr_url": None,
        "findings": {
            "unanimous_critiques": [],
            "agreement_score": 1.0,
        },
    }

    (reviews_dir / "abc123.json").write_text(json.dumps(review_1))
    (reviews_dir / "def456.json").write_text(json.dumps(review_2))

    return reviews_dir


# ===========================================================================
# Test Instantiation and Basics
# ===========================================================================


class TestReviewsHandlerBasics:
    """Basic instantiation and attribute tests."""

    def test_instantiation(self, handler):
        assert handler is not None
        assert isinstance(handler, ReviewsHandler)

    def test_routes(self, handler):
        assert "/api/reviews" in handler.ROUTES
        assert "/api/v1/reviews" in handler.ROUTES

    def test_can_handle_reviews(self, handler):
        assert handler.can_handle("/api/reviews") is True

    def test_can_handle_review_by_id(self, handler):
        assert handler.can_handle("/api/reviews/abc123") is True

    def test_can_handle_versioned(self, handler):
        assert handler.can_handle("/api/v1/reviews") is True

    def test_can_handle_versioned_with_id(self, handler):
        assert handler.can_handle("/api/v1/reviews/abc123") is True

    def test_cannot_handle_other_path(self, handler):
        assert handler.can_handle("/api/debates") is False

    def test_default_context(self):
        h = ReviewsHandler()
        assert h.ctx == {}


# ===========================================================================
# Test _list_reviews
# ===========================================================================


class TestListReviews:
    """Tests for the list reviews endpoint."""

    def test_list_reviews_success(self, handler, tmp_reviews_dir):
        with patch("aragora.server.handlers.reviews.REVIEWS_DIR", tmp_reviews_dir):
            result = handler._list_reviews()
            assert result.status_code == 200
            data = _parse_body(result)
            assert "reviews" in data
            assert data["total"] == 2
            # Check review summary structure
            review = data["reviews"][0]
            assert "id" in review
            assert "created_at" in review
            assert "agents" in review
            assert "agreement_score" in review

    def test_list_reviews_no_directory(self, handler, tmp_path):
        nonexistent = tmp_path / "nonexistent"
        with patch("aragora.server.handlers.reviews.REVIEWS_DIR", nonexistent):
            result = handler._list_reviews()
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["reviews"] == []
            assert data["total"] == 0

    def test_list_reviews_empty_directory(self, handler, tmp_path):
        empty_dir = tmp_path / "empty_reviews"
        empty_dir.mkdir()
        with patch("aragora.server.handlers.reviews.REVIEWS_DIR", empty_dir):
            result = handler._list_reviews()
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["reviews"] == []
            assert data["total"] == 0

    def test_list_reviews_skips_corrupt_file(self, handler, tmp_path):
        reviews_dir = tmp_path / "reviews_corrupt"
        reviews_dir.mkdir()
        (reviews_dir / "good.json").write_text(
            json.dumps({
                "id": "good1",
                "created_at": "2026-01-01",
                "agents": [],
                "findings": {"unanimous_critiques": [], "agreement_score": 0.5},
            })
        )
        (reviews_dir / "bad.json").write_text("not valid json {{{")

        with patch("aragora.server.handlers.reviews.REVIEWS_DIR", reviews_dir):
            result = handler._list_reviews()
            assert result.status_code == 200
            data = _parse_body(result)
            # Only the good file should be returned
            assert data["total"] == 1

    def test_list_reviews_with_limit(self, handler, tmp_reviews_dir):
        with patch("aragora.server.handlers.reviews.REVIEWS_DIR", tmp_reviews_dir):
            result = handler._list_reviews(limit=1)
            assert result.status_code == 200
            data = _parse_body(result)
            assert len(data["reviews"]) == 1


# ===========================================================================
# Test _get_review
# ===========================================================================


class TestGetReview:
    """Tests for getting a specific review."""

    def test_get_review_success(self, handler, tmp_reviews_dir):
        with patch("aragora.server.handlers.reviews.REVIEWS_DIR", tmp_reviews_dir):
            result = handler._get_review("abc123")
            assert result.status_code == 200
            data = _parse_body(result)
            assert "review" in data
            assert data["review"]["id"] == "abc123"

    def test_get_review_not_found(self, handler, tmp_reviews_dir):
        with patch("aragora.server.handlers.reviews.REVIEWS_DIR", tmp_reviews_dir):
            result = handler._get_review("nonexistent")
            assert result.status_code == 404

    def test_get_review_invalid_id_empty(self, handler):
        result = handler._get_review("")
        assert result.status_code == 400

    def test_get_review_invalid_id_special_chars(self, handler):
        result = handler._get_review("../../../etc/passwd")
        assert result.status_code == 400

    def test_get_review_invalid_id_too_long(self, handler):
        result = handler._get_review("a" * 65)
        assert result.status_code == 400

    def test_get_review_corrupt_file(self, handler, tmp_path):
        reviews_dir = tmp_path / "reviews_bad"
        reviews_dir.mkdir()
        (reviews_dir / "abc123.json").write_text("not json")

        with patch("aragora.server.handlers.reviews.REVIEWS_DIR", reviews_dir):
            result = handler._get_review("abc123")
            assert result.status_code == 500


# ===========================================================================
# Test handle() Routing (GET)
# ===========================================================================


class TestHandleRouting:
    """Tests for the top-level handle() method routing."""

    def test_handle_list_reviews(self, handler, tmp_path):
        mock_handler = _make_mock_handler()
        empty_dir = tmp_path / "reviews"
        empty_dir.mkdir()

        with patch("aragora.server.handlers.reviews.REVIEWS_DIR", empty_dir):
            with patch.object(handler, "require_auth_or_error", return_value=(MagicMock(), None)):
                with patch.object(handler, "require_permission_or_error", return_value=(MagicMock(), None)):
                    result = handler.handle("/api/reviews", {}, mock_handler)
                    assert result is not None
                    assert result.status_code == 200

    def test_handle_get_review_by_id(self, handler, tmp_reviews_dir):
        mock_handler = _make_mock_handler()

        with patch("aragora.server.handlers.reviews.REVIEWS_DIR", tmp_reviews_dir):
            with patch.object(handler, "require_auth_or_error", return_value=(MagicMock(), None)):
                with patch.object(handler, "require_permission_or_error", return_value=(MagicMock(), None)):
                    result = handler.handle("/api/reviews/abc123", {}, mock_handler)
                    assert result is not None
                    assert result.status_code == 200

    def test_handle_auth_failure(self, handler):
        mock_handler = _make_mock_handler()
        err = HandlerResult(status_code=401, content_type="application/json", body=b'{"error":"Unauthorized"}')

        with patch.object(handler, "require_auth_or_error", return_value=(None, err)):
            result = handler.handle("/api/reviews", {}, mock_handler)
            assert result.status_code == 401

    def test_handle_permission_failure(self, handler):
        mock_handler = _make_mock_handler()
        perm_err = HandlerResult(status_code=403, content_type="application/json", body=b'{"error":"Forbidden"}')

        with patch.object(handler, "require_auth_or_error", return_value=(MagicMock(), None)):
            with patch.object(handler, "require_permission_or_error", return_value=(None, perm_err)):
                result = handler.handle("/api/reviews", {}, mock_handler)
                assert result.status_code == 403

    def test_handle_rate_limited(self, handler):
        from aragora.server.handlers.reviews import _reviews_limiter

        mock_handler = _make_mock_handler()
        with patch.object(_reviews_limiter, "is_allowed", return_value=False):
            result = handler.handle("/api/reviews", {}, mock_handler)
            assert result.status_code == 429

    def test_handle_versioned_path(self, handler, tmp_path):
        mock_handler = _make_mock_handler()
        empty_dir = tmp_path / "reviews"
        empty_dir.mkdir()

        with patch("aragora.server.handlers.reviews.REVIEWS_DIR", empty_dir):
            with patch.object(handler, "require_auth_or_error", return_value=(MagicMock(), None)):
                with patch.object(handler, "require_permission_or_error", return_value=(MagicMock(), None)):
                    result = handler.handle("/api/v1/reviews", {}, mock_handler)
                    assert result is not None
                    assert result.status_code == 200
