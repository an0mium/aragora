"""
Webhooks Namespace API

Provides methods for webhook management:
- Create and manage webhooks
- Track deliveries
- Test webhook endpoints
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

_List = list  # Preserve builtin list for type annotations

class WebhooksAPI:
    """
    Synchronous Webhooks API.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> webhook = client.webhooks.create(
        ...     url="https://example.com/webhook",
        ...     events=["debate.completed", "receipt.created"]
        ... )
        >>> print(webhook["webhook_id"])
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def list(
        self,
        active_only: bool = True,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List webhooks.

        Args:
            active_only: Only return active webhooks
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of webhooks
        """
        params: dict[str, Any] = {
            "active_only": active_only,
            "limit": limit,
            "offset": offset,
        }
        return self._client.request("GET", "/api/v1/webhooks", params=params)

    def create(
        self,
        url: str,
        events: _List[str],
        secret: str | None = None,
        description: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Create a webhook.

        Args:
            url: Webhook endpoint URL
            events: List of event types to subscribe to
            secret: Webhook signing secret (auto-generated if not provided)
            description: Webhook description
            headers: Custom headers to include in requests

        Returns:
            Created webhook with secret
        """
        data: dict[str, Any] = {"url": url, "events": events}
        if secret:
            data["secret"] = secret
        if description:
            data["description"] = description
        if headers:
            data["headers"] = headers

        return self._client.request("POST", "/api/v1/webhooks", json=data)

    def get_events(self) -> dict[str, Any]:
        """
        Get available webhook event types.

        Returns:
            List of event types with descriptions
        """
        return self._client.request("GET", "/api/v1/webhooks/events")

class AsyncWebhooksAPI:
    """
    Asynchronous Webhooks API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     webhook = await client.webhooks.create(
        ...         url="https://example.com/webhook",
        ...         events=["debate.completed"]
        ...     )
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list(
        self,
        active_only: bool = True,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List webhooks."""
        params: dict[str, Any] = {
            "active_only": active_only,
            "limit": limit,
            "offset": offset,
        }
        return await self._client.request("GET", "/api/v1/webhooks", params=params)

    async def create(
        self,
        url: str,
        events: _List[str],
        secret: str | None = None,
        description: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Create a webhook."""
        data: dict[str, Any] = {"url": url, "events": events}
        if secret:
            data["secret"] = secret
        if description:
            data["description"] = description
        if headers:
            data["headers"] = headers

        return await self._client.request("POST", "/api/v1/webhooks", json=data)

    async def get_events(self) -> dict[str, Any]:
        """Get available webhook event types."""
        return await self._client.request("GET", "/api/v1/webhooks/events")
