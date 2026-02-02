"""
Reputation Namespace API

Provides access to agent reputation data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class ReputationAPI:
    """Synchronous Reputation API for agent reputation scores."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def list_all(self) -> dict[str, Any]:
        """List all agent reputations.

        Returns:
            All agents with reputation scores.
        """
        return self._client.request("GET", "/api/v1/reputation/all")


class AsyncReputationAPI:
    """Asynchronous Reputation API for agent reputation scores."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list_all(self) -> dict[str, Any]:
        """List all agent reputations.

        Returns:
            All agents with reputation scores.
        """
        return await self._client.request("GET", "/api/v1/reputation/all")
