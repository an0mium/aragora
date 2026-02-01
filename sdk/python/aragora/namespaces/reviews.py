"""
Reviews Namespace API

Provides access to debate and decision reviews:
- List recent reviews with filtering
- Get specific reviews by ID
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

ReviewStatus = Literal["pending", "approved", "rejected", "needs_revision"]


class ReviewsAPI:
    """
    Synchronous Reviews API.

    Provides access to debate and decision reviews.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> reviews = client.reviews.list(status="pending")
        >>> for review in reviews["reviews"]:
        ...     print(review["id"], review["status"])
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def list(
        self,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
    ) -> dict[str, Any]:
        """
        List recent reviews.

        Args:
            limit: Maximum number of reviews to return (default: 50)
            offset: Pagination offset (default: 0)
            status: Filter by review status (pending, approved, rejected, needs_revision)

        Returns:
            Dictionary with 'reviews' list and 'total' count
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        return self._client.request("GET", "/api/v1/reviews", params=params)

    def get(self, review_id: str) -> dict[str, Any]:
        """
        Get a specific review by ID.

        Args:
            review_id: The review ID

        Returns:
            Review details including id, debate_id, reviewer, status,
            rating, comments, and timestamps
        """
        return self._client.request("GET", f"/api/v1/reviews/{review_id}")


class AsyncReviewsAPI:
    """
    Asynchronous Reviews API.

    Provides access to debate and decision reviews.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     reviews = await client.reviews.list(status="pending")
        ...     for review in reviews["reviews"]:
        ...         print(review["id"], review["status"])
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list(
        self,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
    ) -> dict[str, Any]:
        """
        List recent reviews.

        Args:
            limit: Maximum number of reviews to return (default: 50)
            offset: Pagination offset (default: 0)
            status: Filter by review status (pending, approved, rejected, needs_revision)

        Returns:
            Dictionary with 'reviews' list and 'total' count
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        return await self._client.request("GET", "/api/v1/reviews", params=params)

    async def get(self, review_id: str) -> dict[str, Any]:
        """
        Get a specific review by ID.

        Args:
            review_id: The review ID

        Returns:
            Review details including id, debate_id, reviewer, status,
            rating, comments, and timestamps
        """
        return await self._client.request("GET", f"/api/v1/reviews/{review_id}")
