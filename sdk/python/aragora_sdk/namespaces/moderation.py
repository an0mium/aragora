"""
Moderation Namespace API

Provides methods for content moderation:
- Get and update moderation configuration
- View moderation statistics
- Manage moderation queue (approve/reject items)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class ModerationAPI:
    """
    Synchronous Moderation API.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> config = client.moderation.get_config()
        >>> stats = client.moderation.get_stats()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_config(self) -> dict[str, Any]:
        """Get current moderation configuration."""
        return self._client.request("GET", "/api/v1/moderation/config")

    def update_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Update moderation configuration."""
        return self._client.request("PUT", "/api/v1/moderation/config", json=config)

    def get_stats(self) -> dict[str, Any]:
        """Get moderation statistics."""
        return self._client.request("GET", "/api/v1/moderation/stats")

    def get_queue(self) -> dict[str, Any]:
        """Get items in the moderation queue."""
        return self._client.request("GET", "/api/v1/moderation/queue")

    def approve_item(self, item_id: str) -> dict[str, Any]:
        """Approve a moderation queue item."""
        return self._client.request("POST", f"/api/v1/moderation/queue/{item_id}/approve")

    def reject_item(self, item_id: str) -> dict[str, Any]:
        """Reject a moderation queue item."""
        return self._client.request("POST", f"/api/v1/moderation/queue/{item_id}/reject")


class AsyncModerationAPI:
    """
    Asynchronous Moderation API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     config = await client.moderation.get_config()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_config(self) -> dict[str, Any]:
        """Get current moderation configuration."""
        return await self._client.request("GET", "/api/v1/moderation/config")

    async def update_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Update moderation configuration."""
        return await self._client.request("PUT", "/api/v1/moderation/config", json=config)

    async def get_stats(self) -> dict[str, Any]:
        """Get moderation statistics."""
        return await self._client.request("GET", "/api/v1/moderation/stats")

    async def get_queue(self) -> dict[str, Any]:
        """Get items in the moderation queue."""
        return await self._client.request("GET", "/api/v1/moderation/queue")

    async def approve_item(self, item_id: str) -> dict[str, Any]:
        """Approve a moderation queue item."""
        return await self._client.request("POST", f"/api/v1/moderation/queue/{item_id}/approve")

    async def reject_item(self, item_id: str) -> dict[str, Any]:
        """Reject a moderation queue item."""
        return await self._client.request("POST", f"/api/v1/moderation/queue/{item_id}/reject")
