"""
Channels Namespace API

Provides methods for channel health monitoring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class ChannelsAPI:
    """Synchronous Channels API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_health(self) -> dict[str, Any]:
        """Get overall channel health status.

        Returns:
            Dict with health status for all channels.
        """
        return self._client.request("GET", "/api/v1/channels/health")

    def get_channel_health(self, channel_id: str) -> dict[str, Any]:
        """Get health status for a specific channel.

        Args:
            channel_id: Channel identifier.

        Returns:
            Dict with channel health details.
        """
        return self._client.request("GET", f"/api/v1/channels/{channel_id}/health")


class AsyncChannelsAPI:
    """Asynchronous Channels API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_health(self) -> dict[str, Any]:
        """Get overall channel health status."""
        return await self._client.request("GET", "/api/v1/channels/health")

    async def get_channel_health(self, channel_id: str) -> dict[str, Any]:
        """Get health status for a specific channel."""
        return await self._client.request(
            "GET", f"/api/v1/channels/{channel_id}/health"
        )
