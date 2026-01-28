"""
Deliberations Namespace API.

Provides visibility into active vetted decisionmaking sessions across the system.
Deliberations are multi-agent debates that produce consensus decisions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

DeliberationStatus = Literal["initializing", "active", "consensus_forming", "complete", "failed"]


class DeliberationsAPI:
    """
    Synchronous Deliberations API.

    Provides visibility into multi-agent vetted decisionmaking sessions.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> # Get active deliberations
        >>> active = client.deliberations.list_active()
        >>> print(f"{active['count']} active deliberations")
        >>> # Get statistics
        >>> stats = client.deliberations.get_stats()
        >>> print(f"Completed today: {stats['completed_today']}")
        >>> # Get specific deliberation
        >>> delib = client.deliberations.get("delib-123")
        >>> print(f"Round {delib['current_round']}/{delib['total_rounds']}")
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    def list_active(self) -> dict[str, Any]:
        """
        List all active deliberation sessions.

        Returns:
            Active deliberations with count and timestamp.
        """
        return self._client.request("GET", "/api/v2/deliberations/active")

    def get_stats(self) -> dict[str, Any]:
        """
        Get deliberation statistics.

        Returns:
            Stats including active count, completed today, averages.
        """
        return self._client.request("GET", "/api/v2/deliberations/stats")

    def get(self, deliberation_id: str) -> dict[str, Any]:
        """
        Get a specific deliberation by ID.

        Args:
            deliberation_id: Deliberation identifier.

        Returns:
            Deliberation details with status, agents, rounds.
        """
        return self._client.request("GET", f"/api/v2/deliberations/{deliberation_id}")

    def get_stream_config(self) -> dict[str, Any]:
        """
        Get WebSocket stream configuration for real-time updates.

        Returns:
            Stream configuration with type, path, and events.
        """
        return self._client.request("GET", "/api/v2/deliberations/stream/config")


class AsyncDeliberationsAPI:
    """Asynchronous Deliberations API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def list_active(self) -> dict[str, Any]:
        """List all active deliberation sessions."""
        return await self._client.request("GET", "/api/v2/deliberations/active")

    async def get_stats(self) -> dict[str, Any]:
        """Get deliberation statistics."""
        return await self._client.request("GET", "/api/v2/deliberations/stats")

    async def get(self, deliberation_id: str) -> dict[str, Any]:
        """Get a specific deliberation by ID."""
        return await self._client.request("GET", f"/api/v2/deliberations/{deliberation_id}")

    async def get_stream_config(self) -> dict[str, Any]:
        """Get WebSocket stream configuration for real-time updates."""
        return await self._client.request("GET", "/api/v2/deliberations/stream/config")
