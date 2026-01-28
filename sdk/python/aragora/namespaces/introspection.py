"""
Introspection Namespace API

Provides access to system introspection and agent information.
Useful for debugging, monitoring, and understanding system state.

Features:
- Get full system introspection
- View agent leaderboards
- List available agents
- Check agent availability
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class IntrospectionAPI:
    """
    Synchronous Introspection API.

    Provides methods for system introspection:
    - Get full system state
    - View agent leaderboards
    - List available agents
    - Get specific agent details

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai", api_key="...")
        >>> system = client.introspection.get_all()
        >>> agents = client.introspection.list_agents()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_all(self) -> dict[str, Any]:
        """
        Get full system introspection.

        Returns:
            Dict with:
            - version: System version
            - agents: List of agent info
            - features: Feature flags
            - capabilities: System capabilities
            - limits: System limits
            - uptime_seconds: Server uptime
        """
        return self._client.request("GET", "/api/v1/introspection/all")

    def get_leaderboard(self, limit: int | None = None) -> dict[str, Any]:
        """
        Get agent leaderboard.

        Args:
            limit: Maximum number of entries

        Returns:
            Dict with leaderboard entries containing:
            - agent: Agent name
            - elo_rating: ELO score
            - wins/losses/draws: Record
            - total_debates: Total debates
            - win_rate: Win percentage
            - rank: Current rank
        """
        params = {"limit": limit} if limit else None
        return self._client.request("GET", "/api/v1/introspection/leaderboard", params=params)

    def list_agents(self) -> dict[str, Any]:
        """
        List available agents.

        Returns:
            Dict with agents list containing:
            - name: Agent name
            - type: Agent type
            - provider: Model provider
            - model: Model name
            - available: Availability status
            - capabilities: Agent capabilities
        """
        return self._client.request("GET", "/api/v1/introspection/agents")

    def get_agent(self, name: str) -> dict[str, Any]:
        """
        Get specific agent details.

        Args:
            name: Agent name

        Returns:
            Dict with full agent information
        """
        return self._client.request("GET", f"/api/v1/introspection/agents/{name}")

    def check_availability(self) -> dict[str, Any]:
        """
        Check agent availability.

        Returns:
            Dict with:
            - available: List of available agents
            - unavailable: List of unavailable agents
        """
        return self._client.request("GET", "/api/v1/introspection/agents/availability")


class AsyncIntrospectionAPI:
    """
    Asynchronous Introspection API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     system = await client.introspection.get_all()
        ...     agents = await client.introspection.list_agents()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_all(self) -> dict[str, Any]:
        """Get full system introspection."""
        return await self._client.request("GET", "/api/v1/introspection/all")

    async def get_leaderboard(self, limit: int | None = None) -> dict[str, Any]:
        """Get agent leaderboard."""
        params = {"limit": limit} if limit else None
        return await self._client.request("GET", "/api/v1/introspection/leaderboard", params=params)

    async def list_agents(self) -> dict[str, Any]:
        """List available agents."""
        return await self._client.request("GET", "/api/v1/introspection/agents")

    async def get_agent(self, name: str) -> dict[str, Any]:
        """Get specific agent details."""
        return await self._client.request("GET", f"/api/v1/introspection/agents/{name}")

    async def check_availability(self) -> dict[str, Any]:
        """Check agent availability."""
        return await self._client.request("GET", "/api/v1/introspection/agents/availability")
