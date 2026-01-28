"""
Pulse Namespace API

Provides access to trending topics and real-time signals from various sources.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class PulseAPI:
    """Synchronous Pulse API for trending topics and signals."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_trending(
        self,
        source: Literal["all", "hackernews", "reddit", "twitter"] | None = None,
        category: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Get trending topics.

        Args:
            source: Filter by source platform.
            category: Filter by category/domain.
            limit: Maximum number of topics to return.

        Returns:
            Trending topics with scores and metadata.
        """
        params: dict[str, Any] = {"limit": limit}
        if source:
            params["source"] = source
        if category:
            params["category"] = category
        return self._client.request("GET", "/api/v1/pulse/trending", params=params)

    def get_topic(self, topic_id: str) -> dict[str, Any]:
        """Get details for a specific trending topic.

        Args:
            topic_id: The topic ID.

        Returns:
            Topic details with sources and history.
        """
        return self._client.request("GET", f"/api/v1/pulse/topics/{topic_id}")

    def search(
        self,
        query: str,
        source: str | None = None,
        min_score: float | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Search pulse topics.

        Args:
            query: Search query.
            source: Filter by source.
            min_score: Minimum relevance score.
            limit: Maximum results.

        Returns:
            Search results with matching topics.
        """
        params: dict[str, Any] = {"q": query, "limit": limit}
        if source:
            params["source"] = source
        if min_score is not None:
            params["min_score"] = min_score
        return self._client.request("GET", "/api/v1/pulse/search", params=params)

    def get_stats(self) -> dict[str, Any]:
        """Get pulse system statistics.

        Returns:
            Statistics about topics, sources, and ingestion.
        """
        return self._client.request("GET", "/api/v1/pulse/stats")

    def get_sources(self) -> list[dict[str, Any]]:
        """List configured pulse sources.

        Returns:
            List of source configurations.
        """
        response = self._client.request("GET", "/api/v1/pulse/sources")
        return response.get("sources", [])

    def get_categories(self) -> list[dict[str, Any]]:
        """List available topic categories.

        Returns:
            List of categories with topic counts.
        """
        response = self._client.request("GET", "/api/v1/pulse/categories")
        return response.get("categories", [])

    def suggest_debate_topic(
        self,
        source: str | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        """Get a suggested debate topic from trending signals.

        Args:
            source: Prefer topics from this source.
            category: Prefer topics in this category.

        Returns:
            Suggested topic with context and reasoning.
        """
        params: dict[str, Any] = {}
        if source:
            params["source"] = source
        if category:
            params["category"] = category
        return self._client.request("GET", "/api/v1/pulse/suggest", params=params)

    def get_history(
        self,
        topic_id: str,
        days: int = 7,
    ) -> list[dict[str, Any]]:
        """Get trending history for a topic.

        Args:
            topic_id: The topic ID.
            days: Number of days of history.

        Returns:
            Historical data points.
        """
        params = {"days": days}
        response = self._client.request(
            "GET", f"/api/v1/pulse/topics/{topic_id}/history", params=params
        )
        return response.get("history", [])


class AsyncPulseAPI:
    """Asynchronous Pulse API for trending topics and signals."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_trending(
        self,
        source: Literal["all", "hackernews", "reddit", "twitter"] | None = None,
        category: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Get trending topics."""
        params: dict[str, Any] = {"limit": limit}
        if source:
            params["source"] = source
        if category:
            params["category"] = category
        return await self._client.request("GET", "/api/v1/pulse/trending", params=params)

    async def get_topic(self, topic_id: str) -> dict[str, Any]:
        """Get details for a specific trending topic."""
        return await self._client.request("GET", f"/api/v1/pulse/topics/{topic_id}")

    async def search(
        self,
        query: str,
        source: str | None = None,
        min_score: float | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Search pulse topics."""
        params: dict[str, Any] = {"q": query, "limit": limit}
        if source:
            params["source"] = source
        if min_score is not None:
            params["min_score"] = min_score
        return await self._client.request("GET", "/api/v1/pulse/search", params=params)

    async def get_stats(self) -> dict[str, Any]:
        """Get pulse system statistics."""
        return await self._client.request("GET", "/api/v1/pulse/stats")

    async def get_sources(self) -> list[dict[str, Any]]:
        """List configured pulse sources."""
        response = await self._client.request("GET", "/api/v1/pulse/sources")
        return response.get("sources", [])

    async def get_categories(self) -> list[dict[str, Any]]:
        """List available topic categories."""
        response = await self._client.request("GET", "/api/v1/pulse/categories")
        return response.get("categories", [])

    async def suggest_debate_topic(
        self,
        source: str | None = None,
        category: str | None = None,
    ) -> dict[str, Any]:
        """Get a suggested debate topic from trending signals."""
        params: dict[str, Any] = {}
        if source:
            params["source"] = source
        if category:
            params["category"] = category
        return await self._client.request("GET", "/api/v1/pulse/suggest", params=params)

    async def get_history(
        self,
        topic_id: str,
        days: int = 7,
    ) -> list[dict[str, Any]]:
        """Get trending history for a topic."""
        params = {"days": days}
        response = await self._client.request(
            "GET", f"/api/v1/pulse/topics/{topic_id}/history", params=params
        )
        return response.get("history", [])
