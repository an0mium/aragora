"""
Ranking Namespace API

Provides access to agent ELO rankings and performance statistics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class RankingAPI:
    """Synchronous Ranking API for agent performance rankings."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def list(
        self,
        limit: int = 50,
        offset: int = 0,
        domain: str | None = None,
        min_debates: int | None = None,
        sort_by: Literal["elo", "wins", "win_rate", "recent_activity"] = "elo",
        order: Literal["asc", "desc"] = "desc",
    ) -> list[dict[str, Any]]:
        """List agent rankings.

        Args:
            limit: Maximum number of rankings to return.
            offset: Number of rankings to skip.
            domain: Filter by domain (e.g., 'technology', 'finance').
            min_debates: Minimum number of debates to qualify.
            sort_by: Field to sort by.
            order: Sort order.

        Returns:
            List of agent ranking entries.
        """
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "sort_by": sort_by,
            "order": order,
        }
        if domain:
            params["domain"] = domain
        if min_debates is not None:
            params["min_debates"] = min_debates

        response = self._client.request("GET", "/api/v1/rankings", params=params)
        return response.get("rankings", [])

    def get(self, agent_name: str) -> dict[str, Any]:
        """Get a specific agent's ranking.

        Args:
            agent_name: The agent name.

        Returns:
            Agent ranking entry.
        """
        response = self._client.request("GET", f"/api/v1/rankings/{agent_name}")
        return response.get("ranking", response)

    def get_stats(self) -> dict[str, Any]:
        """Get aggregate ranking statistics.

        Returns:
            Statistics including total agents, average ELO, etc.
        """
        return self._client.request("GET", "/api/v1/ranking/stats")

    def list_by_domain(
        self,
        domain: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get rankings for a specific domain.

        Args:
            domain: The domain to filter by.
            limit: Maximum number of rankings.
            offset: Number to skip.

        Returns:
            List of rankings in the domain.
        """
        return self.list(domain=domain, limit=limit, offset=offset)

    def get_top(self, n: int = 10) -> list[dict[str, Any]]:
        """Get the top N agents by ELO.

        Args:
            n: Number of top agents to return.

        Returns:
            List of top agent rankings.
        """
        return self.list(limit=n, sort_by="elo", order="desc")

    def get_recently_active(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recently active agents' rankings.

        Args:
            limit: Maximum number of rankings.

        Returns:
            List of recently active agent rankings.
        """
        return self.list(limit=limit, sort_by="recent_activity", order="desc")

    def get_leaderboard(self, domain: str | None = None) -> dict[str, Any]:
        """Get the full leaderboard with statistics.

        Args:
            domain: Optional domain filter.

        Returns:
            Leaderboard data with rankings and stats.
        """
        params: dict[str, Any] = {}
        if domain:
            params["domain"] = domain
        return self._client.request("GET", "/api/v1/leaderboard", params=params)

    def get_history(
        self,
        agent_name: str,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """Get ELO history for an agent.

        Args:
            agent_name: The agent name.
            days: Number of days of history.

        Returns:
            List of historical ELO data points.
        """
        params = {"days": days}
        response = self._client.request(
            "GET", f"/api/v1/rankings/{agent_name}/history", params=params
        )
        return response.get("history", [])


class AsyncRankingAPI:
    """Asynchronous Ranking API for agent performance rankings."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list(
        self,
        limit: int = 50,
        offset: int = 0,
        domain: str | None = None,
        min_debates: int | None = None,
        sort_by: Literal["elo", "wins", "win_rate", "recent_activity"] = "elo",
        order: Literal["asc", "desc"] = "desc",
    ) -> list[dict[str, Any]]:
        """List agent rankings."""
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "sort_by": sort_by,
            "order": order,
        }
        if domain:
            params["domain"] = domain
        if min_debates is not None:
            params["min_debates"] = min_debates

        response = await self._client.request("GET", "/api/v1/rankings", params=params)
        return response.get("rankings", [])

    async def get(self, agent_name: str) -> dict[str, Any]:
        """Get a specific agent's ranking."""
        response = await self._client.request("GET", f"/api/v1/rankings/{agent_name}")
        return response.get("ranking", response)

    async def get_stats(self) -> dict[str, Any]:
        """Get aggregate ranking statistics."""
        return await self._client.request("GET", "/api/v1/ranking/stats")

    async def list_by_domain(
        self,
        domain: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get rankings for a specific domain."""
        return await self.list(domain=domain, limit=limit, offset=offset)

    async def get_top(self, n: int = 10) -> list[dict[str, Any]]:
        """Get the top N agents by ELO."""
        return await self.list(limit=n, sort_by="elo", order="desc")

    async def get_recently_active(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recently active agents' rankings."""
        return await self.list(limit=limit, sort_by="recent_activity", order="desc")

    async def get_leaderboard(self, domain: str | None = None) -> dict[str, Any]:
        """Get the full leaderboard with statistics."""
        params: dict[str, Any] = {}
        if domain:
            params["domain"] = domain
        return await self._client.request("GET", "/api/v1/leaderboard", params=params)

    async def get_history(
        self,
        agent_name: str,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """Get ELO history for an agent."""
        params = {"days": days}
        response = await self._client.request(
            "GET", f"/api/v1/rankings/{agent_name}/history", params=params
        )
        return response.get("history", [])
