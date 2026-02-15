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

    def get_scheduler_analytics(self) -> dict[str, Any]:
        """Get scheduler runtime metrics and store analytics.

        Returns combined scheduler metrics (polls, debates created/failed,
        uptime) and store analytics (by platform, by category, daily counts).

        Returns:
            Scheduler metrics and store analytics.
        """
        return self._client.request("GET", "/api/v1/pulse/scheduler/analytics")


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

    async def get_scheduler_analytics(self) -> dict[str, Any]:
        """Get scheduler runtime metrics and store analytics."""
        return await self._client.request("GET", "/api/v1/pulse/scheduler/analytics")

