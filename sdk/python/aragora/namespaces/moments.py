"""
Moments Namespace API

Provides access to debate moment detection and highlights.
Moments are key events or turning points in debates.

Features:
- Get moment summaries for debates
- View moment timelines
- Find trending moments
- Filter moments by type
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


MomentType = Literal["breakthrough", "conflict", "consensus", "insight", "question", "evidence"]


class MomentsAPI:
    """
    Synchronous Moments API.

    Provides methods for accessing debate moments and highlights:
    - Get moment summaries for debates
    - View moment timelines
    - Find trending moments
    - Filter moments by type

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai", api_key="...")
        >>> summary = client.moments.get_summary("debate-123")
        >>> timeline = client.moments.get_timeline("debate-123")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_summary(self, debate_id: str | None = None) -> dict[str, Any]:
        """
        Get moment summary for debates.

        Args:
            debate_id: Optional debate ID to filter by

        Returns:
            Dict with:
            - debate_id: The debate ID
            - total_moments: Number of moments
            - by_type: Count by moment type
            - highlights: Top moments
            - key_turning_points: Critical moments
        """
        params = {"debate_id": debate_id} if debate_id else None
        return self._client.request("GET", "/api/v1/moments/summary", params=params)

    def get_timeline(self, debate_id: str | None = None) -> dict[str, Any]:
        """
        Get moment timeline for a debate.

        Args:
            debate_id: Optional debate ID to filter by

        Returns:
            Dict with:
            - debate_id: The debate ID
            - moments: List of moments in chronological order
            - duration_seconds: Total debate duration
        """
        params = {"debate_id": debate_id} if debate_id else None
        return self._client.request("GET", "/api/v1/moments/timeline", params=params)

    def get_trending(
        self,
        period: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Get trending moments across debates.

        Args:
            period: Time period (e.g., "24h", "7d", "30d")
            limit: Maximum number of moments to return

        Returns:
            Dict with:
            - period: The time period
            - moments: Trending moments
            - top_debates: Debates with most moments
        """
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        if limit:
            params["limit"] = limit
        return self._client.request(
            "GET", "/api/v1/moments/trending", params=params if params else None
        )

    def get_by_type(
        self,
        moment_type: MomentType,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """
        Get moments filtered by type.

        Args:
            moment_type: Type of moment to filter by
            limit: Maximum number of moments to return
            offset: Number of moments to skip

        Returns:
            Dict with:
            - moments: List of moments of the specified type
            - total: Total count
        """
        params: dict[str, Any] = {}
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        return self._client.request(
            "GET",
            f"/api/v1/moments/by-type/{moment_type}",
            params=params if params else None,
        )


class AsyncMomentsAPI:
    """
    Asynchronous Moments API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     summary = await client.moments.get_summary("debate-123")
        ...     timeline = await client.moments.get_timeline("debate-123")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_summary(self, debate_id: str | None = None) -> dict[str, Any]:
        """Get moment summary for debates."""
        params = {"debate_id": debate_id} if debate_id else None
        return await self._client.request("GET", "/api/v1/moments/summary", params=params)

    async def get_timeline(self, debate_id: str | None = None) -> dict[str, Any]:
        """Get moment timeline for a debate."""
        params = {"debate_id": debate_id} if debate_id else None
        return await self._client.request("GET", "/api/v1/moments/timeline", params=params)

    async def get_trending(
        self,
        period: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Get trending moments across debates."""
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        if limit:
            params["limit"] = limit
        return await self._client.request(
            "GET", "/api/v1/moments/trending", params=params if params else None
        )

    async def get_by_type(
        self,
        moment_type: MomentType,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """Get moments filtered by type."""
        params: dict[str, Any] = {}
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        return await self._client.request(
            "GET",
            f"/api/v1/moments/by-type/{moment_type}",
            params=params if params else None,
        )
