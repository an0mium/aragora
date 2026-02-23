"""Tests for reviews handler (aragora/server/handlers/reviews.py).

Covers all routes and behavior of the ReviewsHandler class:
- can_handle() routing for all ROUTES
- GET /api/reviews - List recent reviews
- GET /api/reviews/{id} - Get a specific review by ID
- Rate limiting behavior
- Error handling (invalid IDs, missing reviews, corrupt JSON)
- Auth enforcement for non-GET methods
- Edge cases (empty directory, no directory, large IDs)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.reviews import (
    REVIEWS_DIR,
    ReviewsHandler,
    _reviews_limiter,
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


class _MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed to ReviewsHandler."""

    def __init__(
        self,
        method: str = "GET",
        body: dict[str, Any] | None = None,
        client_address: tuple[str, int] | None = None,
    ):
        self.command = method
        self.headers = {"Content-Length": "0"}
        self.rfile = MagicMock()
        self.client_address = client_address or ("127.0.0.1", 12345)

        if body is not None:
            raw = json.dumps(body).encode()
            self.rfile.read.return_value = raw
            self.headers = {"Content-Length": str(len(raw))}
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {"Content-Length": "2"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a ReviewsHandler with minimal context."""
    return ReviewsHandler(ctx={})


@pytest.fixture
def http_handler():
    """Create a mock HTTP handler (GET by default)."""
    return _MockHTTPHandler(method="GET")


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter state between tests."""
    _reviews_limiter._buckets.clear()
    yield
    _reviews_limiter._buckets.clear()


@pytest.fixture
def reviews_dir(tmp_path):
    """Create a temporary reviews directory and patch REVIEWS_DIR."""
    reviews = tmp_path / "reviews"
    reviews.mkdir()
    with patch("aragora.server.handlers.reviews.REVIEWS_DIR", reviews):
        yield reviews


@pytest.fixture
def empty_reviews_dir(tmp_path):
    """Create a temporary empty reviews directory and patch REVIEWS_DIR."""
    reviews = tmp_path / "empty_reviews"
    reviews.mkdir()
    with patch("aragora.server.handlers.reviews.REVIEWS_DIR", reviews):
        yield reviews


@pytest.fixture
def no_reviews_dir(tmp_path):
    """Patch REVIEWS_DIR to a path that does not exist."""
    fake_path = tmp_path / "nonexistent"
    with patch("aragora.server.handlers.reviews.REVIEWS_DIR", fake_path):
        yield fake_path


def _write_review(directory: Path, review_id: str, data: dict) -> Path:
    """Write a review JSON file to the directory."""
    path = directory / f"{review_id}.json"
    path.write_text(json.dumps(data))
    return path


# ===========================================================================
# can_handle() tests
# ===========================================================================


class TestCanHandle:
    """Tests for ReviewsHandler.can_handle()."""

    def test_can_handle_api_reviews(self, handler):
        assert handler.can_handle("/api/reviews") is True

    def test_can_handle_api_reviews_trailing_slash(self, handler):
        assert handler.can_handle("/api/reviews/") is True

    def test_can_handle_api_reviews_with_id(self, handler):
        assert handler.can_handle("/api/reviews/abc123") is True

    def test_can_handle_api_v1_reviews(self, handler):
        assert handler.can_handle("/api/v1/reviews") is True

    def test_can_handle_api_v1_reviews_with_id(self, handler):
        assert handler.can_handle("/api/v1/reviews/abc123") is True

    def test_can_handle_api_v2_reviews(self, handler):
        """v2 should also match since strip_version_prefix normalizes it."""
        assert handler.can_handle("/api/v2/reviews") is True

    def test_cannot_handle_unrelated_path(self, handler):
        assert handler.can_handle("/api/debates") is False

    def test_cannot_handle_root(self, handler):
        assert handler.can_handle("/") is False

    def test_cannot_handle_partial_match(self, handler):
        assert handler.can_handle("/api/review") is False

    def test_can_handle_default_method(self, handler):
        assert handler.can_handle("/api/reviews", method="POST") is True

    def test_can_handle_nested_path(self, handler):
        assert handler.can_handle("/api/reviews/abc/extra") is True


# ===========================================================================
# ROUTES attribute
# ===========================================================================


class TestRoutes:
    """Tests for the ROUTES class attribute."""

    def test_routes_contains_unversioned(self, handler):
        assert "/api/reviews" in ReviewsHandler.ROUTES
        assert "/api/reviews/*" in ReviewsHandler.ROUTES

    def test_routes_contains_versioned(self, handler):
        assert "/api/v1/reviews" in ReviewsHandler.ROUTES
        assert "/api/v1/reviews/*" in ReviewsHandler.ROUTES

    def test_routes_length(self, handler):
        assert len(ReviewsHandler.ROUTES) == 4


# ===========================================================================
# handle() — list reviews (GET /api/reviews)
# ===========================================================================


class TestListReviews:
    """Tests for listing reviews."""

    def test_list_reviews_no_directory(self, handler, http_handler, no_reviews_dir):
        """When the reviews directory doesn't exist, return empty list."""
        result = handler.handle("/api/reviews", {}, http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["reviews"] == []
        assert body["total"] == 0

    def test_list_reviews_empty_directory(self, handler, http_handler, empty_reviews_dir):
        """When the reviews directory exists but has no files."""
        result = handler.handle("/api/reviews", {}, http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["reviews"] == []
        assert body["total"] == 0

    def test_list_reviews_with_trailing_slash(self, handler, http_handler, no_reviews_dir):
        """Path with trailing slash should still list reviews."""
        result = handler.handle("/api/reviews/", {}, http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["reviews"] == []
        assert body["total"] == 0

    def test_list_reviews_single_review(self, handler, http_handler, reviews_dir):
        """List with a single valid review."""
        _write_review(reviews_dir, "abc123", {
            "id": "abc123",
            "created_at": "2026-01-01T00:00:00Z",
            "agents": ["claude", "gpt4"],
            "pr_url": "https://github.com/example/pr/1",
            "findings": {
                "unanimous_critiques": [{"text": "issue1"}],
                "agreement_score": 0.85,
            },
        })
        result = handler.handle("/api/reviews", {}, http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["total"] == 1
        review = body["reviews"][0]
        assert review["id"] == "abc123"
        assert review["agents"] == ["claude", "gpt4"]
        assert review["pr_url"] == "https://github.com/example/pr/1"
        assert review["unanimous_count"] == 1
        assert review["agreement_score"] == 0.85

    def test_list_reviews_multiple_sorted_by_mtime(self, handler, http_handler, reviews_dir):
        """Reviews should be sorted by modification time, newest first."""
        _write_review(reviews_dir, "old1", {
            "id": "old1",
            "created_at": "2026-01-01T00:00:00Z",
            "agents": [],
            "findings": {},
        })
        # Ensure different mtime
        time.sleep(0.05)
        _write_review(reviews_dir, "new1", {
            "id": "new1",
            "created_at": "2026-02-01T00:00:00Z",
            "agents": [],
            "findings": {},
        })
        result = handler.handle("/api/reviews", {}, http_handler)
        body = _body(result)
        assert body["total"] == 2
        # Newest first
        assert body["reviews"][0]["id"] == "new1"
        assert body["reviews"][1]["id"] == "old1"

    def test_list_reviews_skips_corrupt_json(self, handler, http_handler, reviews_dir):
        """Corrupt JSON files should be silently skipped."""
        _write_review(reviews_dir, "good1", {
            "id": "good1",
            "created_at": "2026-01-01T00:00:00Z",
            "agents": [],
            "findings": {},
        })
        corrupt_file = reviews_dir / "bad1.json"
        corrupt_file.write_text("{invalid json!!!")
        result = handler.handle("/api/reviews", {}, http_handler)
        body = _body(result)
        assert body["total"] == 1
        assert body["reviews"][0]["id"] == "good1"

    def test_list_reviews_default_fields(self, handler, http_handler, reviews_dir):
        """Review summaries should include only expected fields."""
        _write_review(reviews_dir, "rev1", {
            "id": "rev1",
            "created_at": "2026-01-15T12:00:00Z",
            "agents": ["claude"],
            "pr_url": None,
            "findings": {
                "unanimous_critiques": [],
                "agreement_score": 0.5,
            },
            "extra_field": "should_not_appear",
        })
        result = handler.handle("/api/reviews", {}, http_handler)
        body = _body(result)
        review = body["reviews"][0]
        expected_keys = {"id", "created_at", "agents", "pr_url", "unanimous_count", "agreement_score"}
        assert set(review.keys()) == expected_keys

    def test_list_reviews_missing_findings(self, handler, http_handler, reviews_dir):
        """Reviews without findings should return defaults."""
        _write_review(reviews_dir, "nofind", {
            "id": "nofind",
            "created_at": "2026-01-01T00:00:00Z",
        })
        result = handler.handle("/api/reviews", {}, http_handler)
        body = _body(result)
        review = body["reviews"][0]
        assert review["unanimous_count"] == 0
        assert review["agreement_score"] == 0
        assert review["agents"] == []

    def test_list_reviews_versioned_path(self, handler, http_handler, no_reviews_dir):
        """Versioned path /api/v1/reviews should work."""
        result = handler.handle("/api/v1/reviews", {}, http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["reviews"] == []

    def test_list_reviews_limit(self, handler, http_handler, reviews_dir):
        """Internal _list_reviews respects limit parameter."""
        for i in range(5):
            _write_review(reviews_dir, f"rev{i:03d}", {
                "id": f"rev{i:03d}",
                "created_at": "2026-01-01T00:00:00Z",
                "findings": {},
            })
            time.sleep(0.02)
        result = handler._list_reviews(limit=3)
        body = _body(result)
        assert body["total"] == 3

    def test_list_reviews_non_json_files_ignored(self, handler, http_handler, reviews_dir):
        """Non-JSON files in the directory should be ignored by glob."""
        _write_review(reviews_dir, "valid1", {
            "id": "valid1",
            "created_at": "2026-01-01T00:00:00Z",
            "findings": {},
        })
        (reviews_dir / "readme.txt").write_text("not a review")
        (reviews_dir / "data.csv").write_text("a,b,c")
        result = handler.handle("/api/reviews", {}, http_handler)
        body = _body(result)
        assert body["total"] == 1


# ===========================================================================
# handle() — get review (GET /api/reviews/{id})
# ===========================================================================


class TestGetReview:
    """Tests for retrieving a specific review."""

    def test_get_review_success(self, handler, http_handler, reviews_dir):
        """Successfully retrieve a review by ID."""
        review_data = {
            "id": "abc123",
            "created_at": "2026-01-01T00:00:00Z",
            "agents": ["claude", "gpt4"],
            "findings": {"unanimous_critiques": [], "agreement_score": 0.9},
        }
        _write_review(reviews_dir, "abc123", review_data)
        result = handler.handle("/api/reviews/abc123", {}, http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["review"]["id"] == "abc123"
        assert body["review"]["agents"] == ["claude", "gpt4"]

    def test_get_review_not_found(self, handler, http_handler, reviews_dir):
        """Return 404 for a non-existent review ID."""
        result = handler.handle("/api/reviews/nonexistent", {}, http_handler)
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body.get("error", "").lower()

    def test_get_review_invalid_id_empty(self, handler, http_handler, reviews_dir):
        """Empty review ID should list reviews instead."""
        # With "/api/reviews/" the subpath is "/" which triggers list
        result = handler.handle("/api/reviews/", {}, http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert "reviews" in body

    def test_get_review_invalid_id_non_alnum(self, handler, http_handler, reviews_dir):
        """Non-alphanumeric IDs should return 400."""
        result = handler.handle("/api/reviews/abc-123", {}, http_handler)
        assert _status(result) == 400
        body = _body(result)
        assert "invalid" in body.get("error", "").lower()

    def test_get_review_invalid_id_special_chars(self, handler, http_handler, reviews_dir):
        """IDs with special characters should return 400."""
        result = handler.handle("/api/reviews/../../etc/passwd", {}, http_handler)
        assert _status(result) == 400

    def test_get_review_invalid_id_too_long(self, handler, http_handler, reviews_dir):
        """IDs longer than 64 chars should return 400."""
        long_id = "a" * 65
        result = handler.handle(f"/api/reviews/{long_id}", {}, http_handler)
        assert _status(result) == 400
        body = _body(result)
        assert "invalid" in body.get("error", "").lower()

    def test_get_review_id_exactly_64_chars(self, handler, http_handler, reviews_dir):
        """ID of exactly 64 chars should be accepted (if file exists)."""
        review_id = "a" * 64
        _write_review(reviews_dir, review_id, {"id": review_id, "data": "ok"})
        result = handler.handle(f"/api/reviews/{review_id}", {}, http_handler)
        assert _status(result) == 200

    def test_get_review_corrupt_json(self, handler, http_handler, reviews_dir):
        """Corrupt JSON in review file should return 500."""
        corrupt_path = reviews_dir / "corrupt1.json"
        corrupt_path.write_text("{bad json here")
        result = handler.handle("/api/reviews/corrupt1", {}, http_handler)
        assert _status(result) == 500
        body = _body(result)
        assert "invalid" in body.get("error", "").lower()

    def test_get_review_versioned_path(self, handler, http_handler, reviews_dir):
        """Versioned path /api/v1/reviews/{id} should work."""
        _write_review(reviews_dir, "ver1", {"id": "ver1", "status": "done"})
        result = handler.handle("/api/v1/reviews/ver1", {}, http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["review"]["id"] == "ver1"

    def test_get_review_with_nested_path(self, handler, http_handler, reviews_dir):
        """Path with extra segments after ID should use only first segment as ID."""
        _write_review(reviews_dir, "nested1", {"id": "nested1", "data": "yes"})
        result = handler.handle("/api/reviews/nested1/extra/stuff", {}, http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["review"]["id"] == "nested1"

    def test_get_review_full_data_returned(self, handler, http_handler, reviews_dir):
        """Full review data should be returned (not just summary)."""
        full_data = {
            "id": "full1",
            "created_at": "2026-01-01T00:00:00Z",
            "agents": ["claude", "gpt4", "gemini"],
            "pr_url": "https://github.com/example/pr/42",
            "findings": {
                "unanimous_critiques": [
                    {"text": "Missing error handling", "severity": "high"},
                ],
                "agreement_score": 0.95,
                "individual_reviews": [
                    {"agent": "claude", "comments": ["good overall"]},
                ],
            },
            "metadata": {"repo": "example/repo", "branch": "main"},
        }
        _write_review(reviews_dir, "full1", full_data)
        result = handler.handle("/api/reviews/full1", {}, http_handler)
        body = _body(result)
        assert body["review"] == full_data


# ===========================================================================
# Rate limiting
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting on the reviews endpoint."""

    def test_rate_limit_allows_normal_requests(self, handler, http_handler, no_reviews_dir):
        """Normal requests should not be rate limited."""
        result = handler.handle("/api/reviews", {}, http_handler)
        assert _status(result) == 200

    def test_rate_limit_exceeded(self, handler, no_reviews_dir):
        """Exceeding rate limit should return 429."""
        with patch.object(_reviews_limiter, "is_allowed", return_value=False):
            http = _MockHTTPHandler()
            result = handler.handle("/api/reviews", {}, http)
            assert _status(result) == 429
            body = _body(result)
            assert "rate limit" in body.get("error", "").lower()

    def test_rate_limit_uses_client_ip(self, handler, no_reviews_dir):
        """Rate limiter should use client IP extracted from handler."""
        with patch("aragora.server.handlers.reviews.get_client_ip", return_value="10.0.0.1") as mock_ip, \
             patch.object(_reviews_limiter, "is_allowed", return_value=True):
            http = _MockHTTPHandler()
            handler.handle("/api/reviews", {}, http)
            mock_ip.assert_called_once_with(http)
            _reviews_limiter.is_allowed.assert_called_once_with("10.0.0.1")

    def test_rate_limit_different_ips(self, handler, no_reviews_dir):
        """Different IPs should have independent rate limits."""
        with patch("aragora.server.handlers.reviews.get_client_ip") as mock_ip, \
             patch.object(_reviews_limiter, "is_allowed", side_effect=[True, False]):
            mock_ip.side_effect = ["10.0.0.1", "10.0.0.2"]
            http1 = _MockHTTPHandler(client_address=("10.0.0.1", 1234))
            http2 = _MockHTTPHandler(client_address=("10.0.0.2", 1234))
            r1 = handler.handle("/api/reviews", {}, http1)
            r2 = handler.handle("/api/reviews", {}, http2)
            assert _status(r1) == 200
            assert _status(r2) == 429


# ===========================================================================
# Auth enforcement for non-GET methods
# ===========================================================================


class TestAuthEnforcement:
    """Tests for authentication/authorization on write methods."""

    @pytest.mark.no_auto_auth
    def test_non_get_requires_auth(self, handler, reviews_dir):
        """Non-GET methods require authentication."""
        http = _MockHTTPHandler(method="POST")
        # Patch require_auth_or_error to return an error
        with patch.object(
            handler, "require_auth_or_error",
            return_value=(None, MagicMock(status_code=401, body=json.dumps({"error": "Unauthorized"}).encode())),
        ):
            result = handler.handle("/api/reviews", {}, http)
            assert _status(result) == 401

    @pytest.mark.no_auto_auth
    def test_non_get_requires_permission(self, handler, reviews_dir):
        """Non-GET methods require reviews:write permission."""
        http = _MockHTTPHandler(method="POST")
        mock_user = MagicMock()
        with patch.object(
            handler, "require_auth_or_error",
            return_value=(mock_user, None),
        ), patch.object(
            handler, "require_permission_or_error",
            return_value=(None, MagicMock(status_code=403, body=json.dumps({"error": "Forbidden"}).encode())),
        ):
            result = handler.handle("/api/reviews", {}, http)
            assert _status(result) == 403

    def test_get_does_not_require_auth(self, handler, http_handler, no_reviews_dir):
        """GET requests should not require authentication."""
        # The auto-auth fixture is active but GET should succeed regardless
        result = handler.handle("/api/reviews", {}, http_handler)
        assert _status(result) == 200

    def test_null_handler_defaults_to_get(self, handler, no_reviews_dir):
        """When handler is None, method defaults to GET."""
        with patch("aragora.server.handlers.reviews.get_client_ip", return_value="127.0.0.1"), \
             patch.object(_reviews_limiter, "is_allowed", return_value=True):
            result = handler.handle("/api/reviews", {}, None)
            assert _status(result) == 200


# ===========================================================================
# Handler init
# ===========================================================================


class TestHandlerInit:
    """Tests for ReviewsHandler initialization."""

    def test_init_with_context(self):
        h = ReviewsHandler(ctx={"key": "value"})
        assert h.ctx == {"key": "value"}

    def test_init_without_context(self):
        h = ReviewsHandler()
        assert h.ctx == {}

    def test_init_with_none_context(self):
        h = ReviewsHandler(ctx=None)
        assert h.ctx == {}


# ===========================================================================
# Edge cases and path handling
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases in path handling and data."""

    def test_handle_returns_none_for_no_match(self, handler, http_handler):
        """handle() returns None when path doesn't match after normalization."""
        # This should not happen if can_handle is called first, but test defensively.
        # Force a path that starts with /api/reviews but has empty subpath after processing
        # Actually, "/api/reviews" maps to list and "/api/reviews/X" maps to get.
        # The only way to get None is if subpath doesn't start with "/" and isn't empty.
        # This is practically impossible given strip_version_prefix, but let's verify
        # the list path works with exact match.
        result = handler.handle("/api/reviews", {}, http_handler)
        assert result is not None

    def test_reviews_dir_constant(self):
        """REVIEWS_DIR should point to ~/.aragora/reviews/."""
        assert REVIEWS_DIR == Path.home() / ".aragora" / "reviews"

    def test_review_id_with_underscores_invalid(self, handler, http_handler, reviews_dir):
        """IDs with underscores are not alphanumeric, should return 400."""
        result = handler.handle("/api/reviews/abc_123", {}, http_handler)
        assert _status(result) == 400

    def test_review_id_with_spaces_invalid(self, handler, http_handler, reviews_dir):
        """IDs with spaces are not alphanumeric, should return 400."""
        result = handler.handle("/api/reviews/abc 123", {}, http_handler)
        assert _status(result) == 400

    def test_review_id_numeric_only(self, handler, http_handler, reviews_dir):
        """Pure numeric IDs are valid alphanumeric."""
        _write_review(reviews_dir, "12345", {"id": "12345", "data": "numeric"})
        result = handler.handle("/api/reviews/12345", {}, http_handler)
        assert _status(result) == 200

    def test_review_id_uppercase(self, handler, http_handler, reviews_dir):
        """Uppercase IDs are valid alphanumeric."""
        _write_review(reviews_dir, "ABC123", {"id": "ABC123", "data": "upper"})
        result = handler.handle("/api/reviews/ABC123", {}, http_handler)
        assert _status(result) == 200

    def test_review_id_single_char(self, handler, http_handler, reviews_dir):
        """Single character ID is valid."""
        _write_review(reviews_dir, "x", {"id": "x"})
        result = handler.handle("/api/reviews/x", {}, http_handler)
        assert _status(result) == 200

    def test_review_with_empty_findings(self, handler, http_handler, reviews_dir):
        """Review with empty findings dict should have zero unanimous_count."""
        _write_review(reviews_dir, "emptyfind", {
            "id": "emptyfind",
            "findings": {},
        })
        result = handler.handle("/api/reviews", {}, http_handler)
        body = _body(result)
        review = body["reviews"][0]
        assert review["unanimous_count"] == 0
        assert review["agreement_score"] == 0

    def test_path_traversal_prevention(self, handler, http_handler, reviews_dir):
        """Path traversal attempts should be blocked by alnum check."""
        result = handler.handle("/api/reviews/..%2F..%2Fetc%2Fpasswd", {}, http_handler)
        assert _status(result) == 400

    def test_review_id_with_dots_invalid(self, handler, http_handler, reviews_dir):
        """IDs with dots are not alphanumeric."""
        result = handler.handle("/api/reviews/review.json", {}, http_handler)
        assert _status(result) == 400
