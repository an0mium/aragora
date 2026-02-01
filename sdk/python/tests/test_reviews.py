"""Tests for Reviews namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient

# =========================================================================
# List Reviews Operations
# =========================================================================


class TestReviewsList:
    """Tests for list reviews operations."""

    def test_list_reviews_default(self) -> None:
        """List reviews with default parameters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "reviews": [
                    {
                        "id": "rev_123",
                        "debate_id": "deb_456",
                        "status": "pending",
                        "created_at": "2025-01-15T10:00:00Z",
                    }
                ],
                "total": 1,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.reviews.list()

            mock_request.assert_called_once_with(
                "GET", "/api/v1/reviews", params={"limit": 50, "offset": 0}
            )
            assert result["total"] == 1
            assert result["reviews"][0]["status"] == "pending"
            client.close()

    def test_list_reviews_with_pagination(self) -> None:
        """List reviews with pagination parameters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"reviews": [], "total": 100}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.reviews.list(limit=25, offset=50)

            mock_request.assert_called_once_with(
                "GET", "/api/v1/reviews", params={"limit": 25, "offset": 50}
            )
            client.close()

    def test_list_reviews_with_status_filter(self) -> None:
        """List reviews with status filter."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "reviews": [{"id": "rev_1", "status": "approved"}],
                "total": 1,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.reviews.list(status="approved")

            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["status"] == "approved"
            assert result["reviews"][0]["status"] == "approved"
            client.close()

    def test_list_reviews_with_all_filters(self) -> None:
        """List reviews with all filters applied."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"reviews": [], "total": 0}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.reviews.list(limit=10, offset=5, status="needs_revision")

            call_args = mock_request.call_args
            params = call_args[1]["params"]
            assert params["limit"] == 10
            assert params["offset"] == 5
            assert params["status"] == "needs_revision"
            client.close()


# =========================================================================
# Get Review Operations
# =========================================================================


class TestReviewsGet:
    """Tests for get review operations."""

    def test_get_review(self) -> None:
        """Get a specific review by ID."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "id": "rev_123",
                "debate_id": "deb_456",
                "reviewer": "user_789",
                "status": "approved",
                "rating": 4.5,
                "comments": "Good analysis but could use more detail",
                "created_at": "2025-01-15T10:00:00Z",
                "updated_at": "2025-01-16T14:30:00Z",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.reviews.get("rev_123")

            mock_request.assert_called_once_with("GET", "/api/v1/reviews/rev_123")
            assert result["id"] == "rev_123"
            assert result["status"] == "approved"
            assert result["rating"] == 4.5
            client.close()

    def test_get_review_with_metadata(self) -> None:
        """Get a review that has metadata."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "id": "rev_456",
                "status": "rejected",
                "metadata": {"reason_code": "quality", "tags": ["incomplete"]},
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.reviews.get("rev_456")

            assert result["status"] == "rejected"
            assert result["metadata"]["reason_code"] == "quality"
            client.close()


# =========================================================================
# Async Tests
# =========================================================================


class TestAsyncReviews:
    """Tests for async Reviews API."""

    @pytest.mark.asyncio
    async def test_async_list_reviews(self) -> None:
        """List reviews asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"reviews": [], "total": 0}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.reviews.list()

                mock_request.assert_called_once_with(
                    "GET", "/api/v1/reviews", params={"limit": 50, "offset": 0}
                )
                assert "reviews" in result

    @pytest.mark.asyncio
    async def test_async_list_reviews_with_status(self) -> None:
        """List reviews with status filter asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "reviews": [{"id": "rev_1", "status": "pending"}],
                "total": 1,
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.reviews.list(status="pending")

                call_args = mock_request.call_args
                params = call_args[1]["params"]
                assert params["status"] == "pending"
                assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_async_get_review(self) -> None:
        """Get a review asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "id": "rev_789",
                "status": "approved",
                "rating": 5.0,
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.reviews.get("rev_789")

                mock_request.assert_called_once_with("GET", "/api/v1/reviews/rev_789")
                assert result["id"] == "rev_789"
                assert result["rating"] == 5.0
