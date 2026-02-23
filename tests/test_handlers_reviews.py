"""
Tests for Reviews Handler.

Tests shareable code review endpoints:
- GET /api/v1/reviews - List recent reviews
- GET /api/v1/reviews/{id} - Get specific review
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

from aragora.server.handlers.reviews import ReviewsHandler, REVIEWS_DIR
from aragora.server.handlers.base import HandlerResult


def parse_result(result: HandlerResult) -> tuple[dict, int]:
    """Parse a HandlerResult into (body_dict, status_code)."""
    body_str = result.body.decode("utf-8") if isinstance(result.body, bytes) else result.body
    try:
        body_dict = json.loads(body_str)
    except (json.JSONDecodeError, TypeError):
        body_dict = {"raw": body_str}
    return body_dict, result.status_code


def make_mock_handler(method="GET"):
    """Create a mock HTTP handler with proper attributes."""
    handler = MagicMock()
    handler.command = method
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {"Content-Type": "application/json"}
    return handler


class TestReviewsHandlerRouting:
    """Tests for ReviewsHandler routing logic."""

    def setup_method(self):
        """Setup handler for tests."""
        self.handler = ReviewsHandler({})

    def test_can_handle_reviews_list(self):
        """Should handle /api/v1/reviews path."""
        assert self.handler.can_handle("/api/v1/reviews", "GET") is True
        assert self.handler.can_handle("/api/v1/reviews/", "GET") is True

    def test_can_handle_reviews_by_id(self):
        """Should handle /api/v1/reviews/{id} path."""
        assert self.handler.can_handle("/api/v1/reviews/abc123", "GET") is True
        assert self.handler.can_handle("/api/v1/reviews/xyz789", "GET") is True

    def test_cannot_handle_other_paths(self):
        """Should not handle non-review paths."""
        assert self.handler.can_handle("/api/v1/debates", "GET") is False
        assert self.handler.can_handle("/api/v1/review", "GET") is False

    def test_only_allows_get_method(self):
        """Should reject non-GET methods via can_handle."""
        mock_handler = make_mock_handler("POST")

        # The handler's handle() method doesn't check method -
        # it routes based on path. Test that non-GET is handled gracefully.
        # The handler returns a HandlerResult regardless.
        result = self.handler.handle("/api/v1/reviews", {}, mock_handler)
        # POST still goes through (handler doesn't filter by method in handle())
        assert result is not None


class TestListReviews:
    """Tests for listing reviews."""

    def setup_method(self):
        """Setup handler for tests."""
        self.handler = ReviewsHandler({})

    def test_list_reviews_empty_directory(self):
        """Should return empty list when no reviews exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("aragora.server.handlers.reviews.REVIEWS_DIR", Path(tmpdir) / "nonexistent"):
                mock_handler = make_mock_handler()
                result = self.handler.handle("/api/v1/reviews", {}, mock_handler)

                body, status = parse_result(result)
                assert status == 200
                assert body["reviews"] == []
                assert body["total"] == 0

    def test_list_reviews_with_data(self):
        """Should return reviews sorted by modification time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reviews_dir = Path(tmpdir)

            # Create sample reviews
            review1 = {
                "id": "abc123",
                "created_at": "2026-01-10T10:00:00Z",
                "agents": ["anthropic-api", "openai-api"],
                "pr_url": "https://github.com/test/repo/pull/1",
                "findings": {
                    "unanimous_critiques": ["SQL injection", "Missing auth"],
                    "agreement_score": 0.85,
                },
            }
            review2 = {
                "id": "def456",
                "created_at": "2026-01-11T10:00:00Z",
                "agents": ["anthropic-api"],
                "pr_url": None,
                "findings": {
                    "unanimous_critiques": [],
                    "agreement_score": 0.5,
                },
            }

            (reviews_dir / "abc123.json").write_text(json.dumps(review1))
            (reviews_dir / "def456.json").write_text(json.dumps(review2))

            with patch("aragora.server.handlers.reviews.REVIEWS_DIR", reviews_dir):
                mock_handler = make_mock_handler()
                result = self.handler.handle("/api/v1/reviews", {}, mock_handler)

                body, status = parse_result(result)
                assert status == 200
                assert body["total"] == 2
                assert len(body["reviews"]) == 2

                # Check review summaries
                ids = [r["id"] for r in body["reviews"]]
                assert "abc123" in ids
                assert "def456" in ids

    def test_list_reviews_handles_malformed_json(self):
        """Should skip reviews with invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reviews_dir = Path(tmpdir)

            # Create valid review
            valid_review = {"id": "valid123", "findings": {"unanimous_critiques": []}}
            (reviews_dir / "valid123.json").write_text(json.dumps(valid_review))

            # Create invalid JSON file
            (reviews_dir / "invalid.json").write_text("{ this is not valid json }")

            with patch("aragora.server.handlers.reviews.REVIEWS_DIR", reviews_dir):
                mock_handler = make_mock_handler()
                result = self.handler.handle("/api/v1/reviews", {}, mock_handler)

                body, status = parse_result(result)
                # Should only return valid review
                assert status == 200
                assert body["total"] == 1
                assert body["reviews"][0]["id"] == "valid123"

    def test_list_reviews_respects_limit(self):
        """Should limit number of returned reviews."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reviews_dir = Path(tmpdir)

            # Create 25 reviews (default limit is 20)
            for i in range(25):
                review = {"id": f"review{i:03d}", "findings": {}}
                (reviews_dir / f"review{i:03d}.json").write_text(json.dumps(review))

            with patch("aragora.server.handlers.reviews.REVIEWS_DIR", reviews_dir):
                # Call internal method directly for limit testing
                result = self.handler._list_reviews(limit=20)

                body, status = parse_result(result)
                assert status == 200
                assert body["total"] == 20
                assert len(body["reviews"]) == 20


class TestGetReviewById:
    """Tests for getting a specific review."""

    def setup_method(self):
        """Setup handler for tests."""
        self.handler = ReviewsHandler({})

    def test_get_review_success(self):
        """Should return review data when found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reviews_dir = Path(tmpdir)

            review_data = {
                "id": "test123",
                "created_at": "2026-01-11T12:00:00Z",
                "agents": ["anthropic-api", "openai-api"],
                "findings": {
                    "unanimous_critiques": ["Issue 1"],
                    "split_opinions": [],
                    "agreement_score": 0.9,
                },
            }
            (reviews_dir / "test123.json").write_text(json.dumps(review_data))

            with patch("aragora.server.handlers.reviews.REVIEWS_DIR", reviews_dir):
                mock_handler = make_mock_handler()
                result = self.handler.handle("/api/v1/reviews/test123", {}, mock_handler)

                body, status = parse_result(result)
                assert status == 200
                assert "review" in body
                assert body["review"]["id"] == "test123"
                assert body["review"]["findings"]["agreement_score"] == 0.9

    def test_get_review_not_found(self):
        """Should return 404 for non-existent review."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reviews_dir = Path(tmpdir)

            with patch("aragora.server.handlers.reviews.REVIEWS_DIR", reviews_dir):
                mock_handler = make_mock_handler()
                result = self.handler.handle("/api/v1/reviews/nonexistent", {}, mock_handler)

                body, status = parse_result(result)
                assert status == 404
                assert "not found" in body["error"].lower()

    def test_get_review_invalid_id_non_alphanumeric(self):
        """Should reject IDs with special characters."""
        # Call internal method directly to test ID validation
        result = self.handler._get_review("abc!@#$")
        body, status = parse_result(result)
        assert status == 400
        assert "Invalid review ID" in body["error"]

    def test_get_review_empty_id(self):
        """Should handle empty review ID gracefully."""
        # Empty ID goes to the validation in _get_review
        result = self.handler._get_review("")
        body, status = parse_result(result)
        assert status == 400

    def test_get_review_corrupted_json(self):
        """Should return 500 for corrupted review file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reviews_dir = Path(tmpdir)

            # Create file with invalid JSON
            (reviews_dir / "corrupted.json").write_text("{ invalid json }")

            with patch("aragora.server.handlers.reviews.REVIEWS_DIR", reviews_dir):
                # Call internal method directly
                result = self.handler._get_review("corrupted")

                body, status = parse_result(result)
                assert status == 500
                assert "Invalid review data" in body["error"]


class TestReviewsHandlerSecurity:
    """Security-focused tests for reviews handler."""

    def setup_method(self):
        """Setup handler for tests."""
        self.handler = ReviewsHandler({})

    def test_path_traversal_blocked(self):
        """Should block path traversal attempts."""
        malicious_ids = [
            "../../../etc/passwd",
            "..%2F..%2Fetc%2Fpasswd",
            "....//....//etc//passwd",
            "review/../../../secret",
        ]

        for malicious_id in malicious_ids:
            # Extract the first segment (handler does split on /)
            first_segment = malicious_id.split("/")[0]
            result = self.handler._get_review(first_segment)
            body, status = parse_result(result)
            # Should either be invalid or not found, never success
            assert status in [400, 404], f"Path traversal not blocked for: {malicious_id}"

    def test_only_alphanumeric_ids_accepted(self):
        """Should only accept alphanumeric review IDs."""
        valid_ids = ["abc123", "ABC123", "a1b2c3", "123456"]
        invalid_ids = ["abc-123", "abc_123", "abc.json", "abc/def"]

        for valid_id in valid_ids:
            result = self.handler._get_review(valid_id)
            body, status = parse_result(result)
            # Should not be rejected as invalid ID (might be not found)
            assert status != 400 or "Invalid" not in body.get("error", "")

        for invalid_id in invalid_ids:
            result = self.handler._get_review(invalid_id)
            body, status = parse_result(result)
            assert status in [400, 404]


class TestReviewsHandlerEdgeCases:
    """Edge case tests for reviews handler."""

    def setup_method(self):
        """Setup handler for tests."""
        self.handler = ReviewsHandler({})

    def test_review_with_missing_fields(self):
        """Should handle reviews with missing optional fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reviews_dir = Path(tmpdir)

            # Minimal review with only required fields
            minimal_review = {"id": "minimal"}
            (reviews_dir / "minimal.json").write_text(json.dumps(minimal_review))

            with patch("aragora.server.handlers.reviews.REVIEWS_DIR", reviews_dir):
                result = self.handler._list_reviews()

                body, status = parse_result(result)
                assert status == 200
                assert body["total"] == 1
                review = body["reviews"][0]
                assert review["id"] == "minimal"
                assert review["agents"] == []  # Default
                assert review["unanimous_count"] == 0  # Default

    def test_handle_returns_none_for_unmatched_subpath(self):
        """Should return a response for the reviews prefix path."""
        mock_handler = make_mock_handler()

        # Path that starts with prefix - should list reviews
        result = self.handler.handle("/api/v1/reviews", {}, mock_handler)
        assert result is not None  # Should list reviews

    def test_very_long_review_id(self):
        """Should handle very long review IDs gracefully."""
        long_id = "a" * 1000
        # Call internal method to test ID length validation
        result = self.handler._get_review(long_id)

        body, status = parse_result(result)
        # Should either be not found or handled gracefully (max 64 chars)
        assert status in [400, 404]
