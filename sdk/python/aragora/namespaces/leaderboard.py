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

    def get_domain_rankings(
        self,
        domain: str,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Get rankings for a specific domain.

        Args:
            domain: Domain name
            limit: Maximum entries

        Returns:
            Dict with domain leaderboard
        """
        params = {"limit": limit} if limit else None
        return self._client.request("GET", f"/api/leaderboard/domain/{domain}", params=params)

    def get_agent_performance(
        self,
        agent_name: str,
        period: PerformancePeriod | None = None,
    ) -> dict[str, Any]:
        """
        Get detailed performance metrics for an agent.

        Args:
            agent_name: The agent name
            period: Time period for metrics

        Returns:
            Dict with:
            - agent_name: Agent name
            - elo: Current ELO
            - elo_history: ELO over time
            - debates_by_domain: Domain breakdown
            - win_rate_by_domain: Win rates
            - average_proposal_quality: Proposal score
            - critique_effectiveness: Critique score
            - consensus_contribution: Consensus score
        """
        params = {"period": period} if period else None
        return self._client.request("GET", f"/api/leaderboard/agent/{agent_name}", params=params)

    def compare_agents(self, agent_a: str, agent_b: str) -> dict[str, Any]:
        """
        Compare two agents head-to-head.

        Args:
            agent_a: First agent name
            agent_b: Second agent name

        Returns:
            Dict with:
            - agent_a/agent_b: Agent names
            - agent_a_wins/agent_b_wins: Win counts
            - draws: Draw count
            - total_matchups: Total matches
            - domains: Domains they've competed in
        """
        return self._client.request(
            "GET",
            "/api/leaderboard/compare",
            params={"agent_a": agent_a, "agent_b": agent_b},
        )

    def get_elo_history(
        self,
        agent_name: str,
        period: PerformancePeriod | None = None,
    ) -> dict[str, Any]:
        """
        Get ELO history for an agent.

        Args:
            agent_name: The agent name
            period: Time period

        Returns:
            Dict with history of ELO changes
        """
        params = {"period": period} if period else None
        return self._client.request(
            "GET", f"/api/leaderboard/agent/{agent_name}/elo-history", params=params
        )

    def get_movers(
        self,
        period: MoverPeriod | None = None,
        direction: MoverDirection | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Get agents that have moved the most in rankings recently.

        Args:
            period: Time period
            direction: Movement direction filter
            limit: Maximum entries

        Returns:
            Dict with movers list
        """
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        if direction:
            params["direction"] = direction
        if limit:
            params["limit"] = limit
        return self._client.request(
            "GET", "/api/leaderboard/movers", params=params if params else None
        )

    def get_domains(self) -> dict[str, Any]:
        """
        Get list of domains with active leaderboards.

        Returns:
            Dict with domains and their stats
        """
        return self._client.request("GET", "/api/leaderboard/domains")


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

    async def get_domain_rankings(
        self,
        domain: str,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Get rankings for a specific domain."""
        params = {"limit": limit} if limit else None
        return await self._client.request("GET", f"/api/leaderboard/domain/{domain}", params=params)

    async def get_agent_performance(
        self,
        agent_name: str,
        period: PerformancePeriod | None = None,
    ) -> dict[str, Any]:
        """Get detailed performance metrics for an agent."""
        params = {"period": period} if period else None
        return await self._client.request(
            "GET", f"/api/leaderboard/agent/{agent_name}", params=params
        )

    async def compare_agents(self, agent_a: str, agent_b: str) -> dict[str, Any]:
        """Compare two agents head-to-head."""
        return await self._client.request(
            "GET",
            "/api/leaderboard/compare",
            params={"agent_a": agent_a, "agent_b": agent_b},
        )

    async def get_elo_history(
        self,
        agent_name: str,
        period: PerformancePeriod | None = None,
    ) -> dict[str, Any]:
        """Get ELO history for an agent."""
        params = {"period": period} if period else None
        return await self._client.request(
            "GET", f"/api/leaderboard/agent/{agent_name}/elo-history", params=params
        )

    async def get_movers(
        self,
        period: MoverPeriod | None = None,
        direction: MoverDirection | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Get agents that have moved the most in rankings recently."""
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        if direction:
            params["direction"] = direction
        if limit:
            params["limit"] = limit
        return await self._client.request(
            "GET", "/api/leaderboard/movers", params=params if params else None
        )

    async def get_domains(self) -> dict[str, Any]:
        """Get list of domains with active leaderboards."""
        return await self._client.request("GET", "/api/leaderboard/domains")
