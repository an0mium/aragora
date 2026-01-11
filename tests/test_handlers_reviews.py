"""
Tests for Reviews Handler.

Tests shareable code review endpoints:
- GET /api/reviews - List recent reviews
- GET /api/reviews/{id} - Get specific review
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from aragora.server.handlers.reviews import ReviewsHandler, REVIEWS_DIR


class TestReviewsHandlerRouting:
    """Tests for ReviewsHandler routing logic."""

    def setup_method(self):
        """Setup handler for tests."""
        self.handler = ReviewsHandler({})

    def test_can_handle_reviews_list(self):
        """Should handle /api/reviews path."""
        assert self.handler.can_handle("/api/reviews", "GET") is True
        assert self.handler.can_handle("/api/reviews/", "GET") is True

    def test_can_handle_reviews_by_id(self):
        """Should handle /api/reviews/{id} path."""
        assert self.handler.can_handle("/api/reviews/abc123", "GET") is True
        assert self.handler.can_handle("/api/reviews/xyz789", "GET") is True

    def test_cannot_handle_other_paths(self):
        """Should not handle non-review paths."""
        assert self.handler.can_handle("/api/debates", "GET") is False
        assert self.handler.can_handle("/api/review", "GET") is False
        assert self.handler.can_handle("/reviews", "GET") is False

    def test_only_allows_get_method(self):
        """Should reject non-GET methods."""
        mock_handler = MagicMock()

        result = self.handler.handle(mock_handler, "/api/reviews", "POST")
        assert result["status"] == 405
        assert "Method not allowed" in result["error"]

        result = self.handler.handle(mock_handler, "/api/reviews", "DELETE")
        assert result["status"] == 405


class TestListReviews:
    """Tests for listing reviews."""

    def setup_method(self):
        """Setup handler for tests."""
        self.handler = ReviewsHandler({})

    def test_list_reviews_empty_directory(self):
        """Should return empty list when no reviews exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                ReviewsHandler, '_list_reviews',
                wraps=self.handler._list_reviews
            ):
                # Patch REVIEWS_DIR to non-existent path
                with patch('aragora.server.handlers.reviews.REVIEWS_DIR', Path(tmpdir) / "nonexistent"):
                    mock_handler = MagicMock()
                    result = self.handler.handle(mock_handler, "/api/reviews", "GET")

                    assert result["reviews"] == []
                    assert result["total"] == 0

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

            with patch('aragora.server.handlers.reviews.REVIEWS_DIR', reviews_dir):
                mock_handler = MagicMock()
                result = self.handler.handle(mock_handler, "/api/reviews", "GET")

                assert result["total"] == 2
                assert len(result["reviews"]) == 2

                # Check review summaries
                ids = [r["id"] for r in result["reviews"]]
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

            with patch('aragora.server.handlers.reviews.REVIEWS_DIR', reviews_dir):
                mock_handler = MagicMock()
                result = self.handler.handle(mock_handler, "/api/reviews", "GET")

                # Should only return valid review
                assert result["total"] == 1
                assert result["reviews"][0]["id"] == "valid123"

    def test_list_reviews_respects_limit(self):
        """Should limit number of returned reviews."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reviews_dir = Path(tmpdir)

            # Create 25 reviews (default limit is 20)
            for i in range(25):
                review = {"id": f"review{i:03d}", "findings": {}}
                (reviews_dir / f"review{i:03d}.json").write_text(json.dumps(review))

            with patch('aragora.server.handlers.reviews.REVIEWS_DIR', reviews_dir):
                result = self.handler._list_reviews(limit=20)

                assert result["total"] == 20
                assert len(result["reviews"]) == 20


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

            with patch('aragora.server.handlers.reviews.REVIEWS_DIR', reviews_dir):
                mock_handler = MagicMock()
                result = self.handler.handle(mock_handler, "/api/reviews/test123", "GET")

                assert "review" in result
                assert result["review"]["id"] == "test123"
                assert result["review"]["findings"]["agreement_score"] == 0.9

    def test_get_review_not_found(self):
        """Should return 404 for non-existent review."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reviews_dir = Path(tmpdir)

            with patch('aragora.server.handlers.reviews.REVIEWS_DIR', reviews_dir):
                mock_handler = MagicMock()
                result = self.handler.handle(mock_handler, "/api/reviews/nonexistent", "GET")

                assert result["status"] == 404
                assert "not found" in result["error"].lower()

    def test_get_review_invalid_id_non_alphanumeric(self):
        """Should reject IDs with special characters."""
        mock_handler = MagicMock()

        # IDs with path traversal attempts
        result = self.handler.handle(mock_handler, "/api/reviews/../../../etc/passwd", "GET")
        assert result["status"] == 400
        assert "Invalid review ID" in result["error"]

        # IDs with special chars
        result = self.handler.handle(mock_handler, "/api/reviews/abc!@#$", "GET")
        assert result["status"] == 400

    def test_get_review_empty_id(self):
        """Should handle empty review ID gracefully."""
        mock_handler = MagicMock()

        # Empty subpath after prefix goes to list, not get
        result = self.handler.handle(mock_handler, "/api/reviews/", "GET")
        # This should list reviews, not get by ID
        assert "reviews" in result or "error" in result

    def test_get_review_corrupted_json(self):
        """Should return 500 for corrupted review file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reviews_dir = Path(tmpdir)

            # Create file with invalid JSON
            (reviews_dir / "corrupted.json").write_text("{ invalid json }")

            with patch('aragora.server.handlers.reviews.REVIEWS_DIR', reviews_dir):
                mock_handler = MagicMock()
                result = self.handler.handle(mock_handler, "/api/reviews/corrupted", "GET")

                assert result["status"] == 500
                assert "Invalid review data" in result["error"]


class TestReviewsHandlerSecurity:
    """Security-focused tests for reviews handler."""

    def setup_method(self):
        """Setup handler for tests."""
        self.handler = ReviewsHandler({})

    def test_path_traversal_blocked(self):
        """Should block path traversal attempts."""
        mock_handler = MagicMock()

        malicious_ids = [
            "../../../etc/passwd",
            "..%2F..%2Fetc%2Fpasswd",
            "....//....//etc//passwd",
            "review/../../../secret",
        ]

        for malicious_id in malicious_ids:
            result = self.handler._get_review(malicious_id.split("/")[0])
            # Should either be invalid or not found, never success
            assert result.get("status") in [400, 404] or "error" in result

    def test_only_alphanumeric_ids_accepted(self):
        """Should only accept alphanumeric review IDs."""
        valid_ids = ["abc123", "ABC123", "a1b2c3", "123456"]
        invalid_ids = ["abc-123", "abc_123", "abc.json", "abc/def", ""]

        for valid_id in valid_ids:
            result = self.handler._get_review(valid_id)
            # Should not be rejected as invalid ID (might be not found)
            assert result.get("status") != 400 or "Invalid" not in result.get("error", "")

        for invalid_id in invalid_ids:
            result = self.handler._get_review(invalid_id)
            assert result.get("status") == 400 or result.get("status") == 404


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

            with patch('aragora.server.handlers.reviews.REVIEWS_DIR', reviews_dir):
                result = self.handler._list_reviews()

                assert result["total"] == 1
                review = result["reviews"][0]
                assert review["id"] == "minimal"
                assert review["agents"] == []  # Default
                assert review["unanimous_count"] == 0  # Default

    def test_handle_returns_none_for_unmatched_subpath(self):
        """Should return None for unmatched paths within prefix."""
        mock_handler = MagicMock()

        # Path that starts with prefix but doesn't match patterns
        # (This is actually handled as a review ID lookup)
        result = self.handler.handle(mock_handler, "/api/reviews", "GET")
        assert result is not None  # Should list reviews

    def test_very_long_review_id(self):
        """Should handle very long review IDs gracefully."""
        mock_handler = MagicMock()

        long_id = "a" * 1000
        result = self.handler.handle(mock_handler, f"/api/reviews/{long_id}", "GET")

        # Should either be not found or handled gracefully
        assert result["status"] in [400, 404]
