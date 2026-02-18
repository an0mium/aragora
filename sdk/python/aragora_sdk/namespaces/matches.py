"""
Matches Namespace API

Provides access to agent match history and rankings:
- List recent matches between agents
- Get match details
- View match statistics
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class MatchesAPI:
    """
    Synchronous Matches API for agent matches.

    Provides access to agent-vs-agent match history from debates
    and tournaments, including ELO rating changes.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> recent = client.matches.list_recent(limit=10)
        >>> for match in recent["matches"]:
        ...     print(f"{match['agent_a']} vs {match['agent_b']}: {match['winner']}")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def list_recent(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List recent matches.

        Args:
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            Dict with recent matches including agents, outcomes,
            and ELO changes.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return self._client.request("GET", "/api/v1/matches/recent", params=params)

    def get(self, match_id: str) -> dict[str, Any]:
        """
        Get details for a specific match.

        Args:
            match_id: Match identifier.

        Returns:
            Dict with full match details including agents, rounds,
            scores, and outcome.
        """
        return self._client.request("GET", f"/api/v1/matches/{match_id}")

    def get_stats(
        self,
        agent: str | None = None,
        period: str | None = None,
    ) -> dict[str, Any]:
        """
        Get match statistics.

        Args:
            agent: Optional agent to filter stats for.
            period: Time period (e.g., '7d', '30d', '90d').

        Returns:
            Dict with match statistics including win/loss records,
            average performance, and trends.
        """
        params: dict[str, Any] = {}
        if agent:
            params["agent"] = agent
        if period:
            params["period"] = period
        return self._client.request(
            "GET", "/api/v1/matches/stats", params=params or None
        )


class AsyncMatchesAPI:
    """
    Asynchronous Matches API for agent matches.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     recent = await client.matches.list_recent(limit=10)
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list_recent(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List recent matches."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return await self._client.request("GET", "/api/v1/matches/recent", params=params)

    async def get(self, match_id: str) -> dict[str, Any]:
        """Get details for a specific match."""
        return await self._client.request("GET", f"/api/v1/matches/{match_id}")

    async def get_stats(
        self,
        agent: str | None = None,
        period: str | None = None,
    ) -> dict[str, Any]:
        """Get match statistics."""
        params: dict[str, Any] = {}
        if agent:
            params["agent"] = agent
        if period:
            params["period"] = period
        return await self._client.request(
            "GET", "/api/v1/matches/stats", params=params or None
        )
