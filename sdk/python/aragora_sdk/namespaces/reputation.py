"""
Reputation Namespace API

Provides access to agent reputation data:
- List all agent reputations
- Get individual agent reputation details
- View reputation history and trends
- Get reputation by domain expertise
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class ReputationAPI:
    """
    Synchronous Reputation API for agent reputation scores.

    Reputation scores reflect an agent's track record based on debate
    performance, calibration quality, and peer assessments.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> all_reps = client.reputation.list_all()
        >>> for agent in all_reps["agents"]:
        ...     print(f"{agent['name']}: {agent['reputation_score']:.2f}")
        >>> claude_rep = client.reputation.get("claude")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def list_all(self) -> dict[str, Any]:
        """
        List all agent reputations.

        Returns:
            Dict with all agents and their reputation scores,
            rankings, and summary statistics.
        """
        return self._client.request("GET", "/api/v1/reputation/all")

    def get(self, agent: str) -> dict[str, Any]:
        """
        Get reputation details for a specific agent.

        Args:
            agent: Agent name or identifier.

        Returns:
            Dict with agent reputation details including:
            - reputation_score: Overall reputation score
            - domain_scores: Per-domain reputation scores
            - trend: Recent reputation trend (improving, stable, declining)
            - debate_count: Number of debates participated in
        """
        return self._client.request("GET", f"/api/v1/reputation/{agent}")

    def get_history(
        self,
        agent: str | None = None,
        period: str | None = None,
    ) -> dict[str, Any]:
        """
        Get reputation history over time.

        Args:
            agent: Optional agent to filter history for.
            period: Time period (e.g., '7d', '30d', '90d').

        Returns:
            Dict with historical reputation data points showing
            reputation evolution over time.
        """
        params: dict[str, Any] = {}
        if agent:
            params["agent"] = agent
        if period:
            params["period"] = period
        return self._client.request("GET", "/api/v1/reputation/history", params=params or None)

    def get_by_domain(self, domain: str) -> dict[str, Any]:
        """
        Get agent reputations filtered by domain expertise.

        Args:
            domain: Domain to filter by (e.g., 'security', 'finance',
                'healthcare', 'engineering').

        Returns:
            Dict with agents ranked by reputation in the specified domain.
        """
        return self._client.request("GET", "/api/v1/reputation/domain", params={"domain": domain})


class AsyncReputationAPI:
    """
    Asynchronous Reputation API for agent reputation scores.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     all_reps = await client.reputation.list_all()
        ...     claude_rep = await client.reputation.get("claude")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list_all(self) -> dict[str, Any]:
        """List all agent reputations."""
        return await self._client.request("GET", "/api/v1/reputation/all")

    async def get(self, agent: str) -> dict[str, Any]:
        """Get reputation details for a specific agent."""
        return await self._client.request("GET", f"/api/v1/reputation/{agent}")

    async def get_history(
        self,
        agent: str | None = None,
        period: str | None = None,
    ) -> dict[str, Any]:
        """Get reputation history over time."""
        params: dict[str, Any] = {}
        if agent:
            params["agent"] = agent
        if period:
            params["period"] = period
        return await self._client.request(
            "GET", "/api/v1/reputation/history", params=params or None
        )

    async def get_by_domain(self, domain: str) -> dict[str, Any]:
        """Get agent reputations filtered by domain expertise."""
        return await self._client.request(
            "GET", "/api/v1/reputation/domain", params={"domain": domain}
        )
