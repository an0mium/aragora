"""
Leaderboard Namespace API

Provides access to agent rankings, ELO scores, and performance comparisons.

Features:
- View global and domain-specific leaderboards
- Compare agent performance
- Track ELO changes over time
- Analyze head-to-head matchups
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

PerformancePeriod = Literal["7d", "30d", "90d", "all"]
MoverPeriod = Literal["24h", "7d", "30d"]
MoverDirection = Literal["up", "down", "both"]


class LeaderboardAPI:
    """
    Synchronous Leaderboard API.

    Provides methods for viewing agent rankings and performance:
    - View global and domain-specific leaderboards
    - Compare agent performance
    - Track ELO changes over time
    - Analyze head-to-head matchups

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai", api_key="...")
        >>> rankings = client.leaderboard.get_rankings()
        >>> comparison = client.leaderboard.compare_agents("claude", "gpt4")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_rankings(
        self,
        limit: int | None = None,
        offset: int | None = None,
        min_debates: int | None = None,
    ) -> dict[str, Any]:
        """
        Get global agent rankings.

        Args:
            limit: Maximum entries to return
            offset: Number to skip
            min_debates: Minimum debates to qualify

        Returns:
            Dict with rankings and total count
        """
        params: dict[str, Any] = {}
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        if min_debates:
            params["min_debates"] = min_debates
        return self._client.request("GET", "/api/leaderboard", params=params if params else None)

    def get_view(self) -> dict[str, Any]:
        """
        Get full leaderboard view with global and domain rankings.

        Returns:
            Dict with:
            - global: Global rankings
            - by_domain: Domain-specific rankings
            - recent_movers: Agents with recent rank changes
            - updated_at: Last update time
        """
        return self._client.request("GET", "/api/leaderboard-view")


class AsyncLeaderboardAPI:
    """
    Asynchronous Leaderboard API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     rankings = await client.leaderboard.get_rankings()
        ...     perf = await client.leaderboard.get_agent_performance("claude")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_rankings(
        self,
        limit: int | None = None,
        offset: int | None = None,
        min_debates: int | None = None,
    ) -> dict[str, Any]:
        """Get global agent rankings."""
        params: dict[str, Any] = {}
        if limit:
            params["limit"] = limit
        if offset:
            params["offset"] = offset
        if min_debates:
            params["min_debates"] = min_debates
        return await self._client.request(
            "GET", "/api/leaderboard", params=params if params else None
        )

    async def get_view(self) -> dict[str, Any]:
        """Get full leaderboard view."""
        return await self._client.request("GET", "/api/leaderboard-view")
