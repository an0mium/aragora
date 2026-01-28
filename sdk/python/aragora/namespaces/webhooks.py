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

    def get(self, webhook_id: str) -> dict[str, Any]:
        """
        Get a webhook by ID.

        Args:
            webhook_id: Webhook ID

        Returns:
            Webhook details
        """
        return self._client.request("GET", f"/api/v1/webhooks/{webhook_id}")

    def create(
        self,
        url: str,
        events: list[str],
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

    def update(
        self,
        webhook_id: str,
        url: str | None = None,
        events: list[str] | None = None,
        active: bool | None = None,
        description: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Update a webhook.

        Args:
            webhook_id: Webhook ID
            url: New endpoint URL
            events: New event list
            active: Active status
            description: New description
            headers: New custom headers

        Returns:
            Updated webhook
        """
        data: dict[str, Any] = {}
        if url is not None:
            data["url"] = url
        if events is not None:
            data["events"] = events
        if active is not None:
            data["active"] = active
        if description is not None:
            data["description"] = description
        if headers is not None:
            data["headers"] = headers

        return self._client.request("PUT", f"/api/v1/webhooks/{webhook_id}", json=data)

    def delete(self, webhook_id: str) -> dict[str, Any]:
        """
        Delete a webhook.

        Args:
            webhook_id: Webhook ID

        Returns:
            Deletion result
        """
        return self._client.request("DELETE", f"/api/v1/webhooks/{webhook_id}")

    def rotate_secret(self, webhook_id: str) -> dict[str, Any]:
        """
        Rotate webhook signing secret.

        Args:
            webhook_id: Webhook ID

        Returns:
            New webhook secret
        """
        return self._client.request("POST", f"/api/v1/webhooks/{webhook_id}/rotate-secret")

    def get_deliveries(
        self,
        webhook_id: str,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Get webhook delivery history.

        Args:
            webhook_id: Webhook ID
            status: Filter by status (success, failed, pending)
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of delivery attempts
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        return self._client.request(
            "GET", f"/api/v1/webhooks/{webhook_id}/deliveries", params=params
        )

    def get_delivery(self, webhook_id: str, delivery_id: str) -> dict[str, Any]:
        """
        Get a specific delivery.

        Args:
            webhook_id: Webhook ID
            delivery_id: Delivery ID

        Returns:
            Delivery details including request/response
        """
        return self._client.request(
            "GET", f"/api/v1/webhooks/{webhook_id}/deliveries/{delivery_id}"
        )

    def retry_delivery(self, webhook_id: str, delivery_id: str) -> dict[str, Any]:
        """
        Retry a failed delivery.

        Args:
            webhook_id: Webhook ID
            delivery_id: Delivery ID

        Returns:
            Retry result
        """
        return self._client.request(
            "POST", f"/api/v1/webhooks/{webhook_id}/deliveries/{delivery_id}/retry"
        )

    def test(self, webhook_id: str, event_type: str | None = None) -> dict[str, Any]:
        """
        Send a test event to webhook.

        Args:
            webhook_id: Webhook ID
            event_type: Event type to simulate (uses first subscribed event if not specified)

        Returns:
            Test result with response details
        """
        data: dict[str, Any] = {}
        if event_type:
            data["event_type"] = event_type

        return self._client.request("POST", f"/api/v1/webhooks/{webhook_id}/test", json=data)

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

    async def get(self, webhook_id: str) -> dict[str, Any]:
        """Get a webhook by ID."""
        return await self._client.request("GET", f"/api/v1/webhooks/{webhook_id}")

    async def create(
        self,
        url: str,
        events: list[str],
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

    async def update(
        self,
        webhook_id: str,
        url: str | None = None,
        events: list[str] | None = None,
        active: bool | None = None,
        description: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Update a webhook."""
        data: dict[str, Any] = {}
        if url is not None:
            data["url"] = url
        if events is not None:
            data["events"] = events
        if active is not None:
            data["active"] = active
        if description is not None:
            data["description"] = description
        if headers is not None:
            data["headers"] = headers

        return await self._client.request("PUT", f"/api/v1/webhooks/{webhook_id}", json=data)

    async def delete(self, webhook_id: str) -> dict[str, Any]:
        """Delete a webhook."""
        return await self._client.request("DELETE", f"/api/v1/webhooks/{webhook_id}")

    async def rotate_secret(self, webhook_id: str) -> dict[str, Any]:
        """Rotate webhook signing secret."""
        return await self._client.request("POST", f"/api/v1/webhooks/{webhook_id}/rotate-secret")

    async def get_deliveries(
        self,
        webhook_id: str,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Get webhook delivery history."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        return await self._client.request(
            "GET", f"/api/v1/webhooks/{webhook_id}/deliveries", params=params
        )

    async def get_delivery(self, webhook_id: str, delivery_id: str) -> dict[str, Any]:
        """Get a specific delivery."""
        return await self._client.request(
            "GET", f"/api/v1/webhooks/{webhook_id}/deliveries/{delivery_id}"
        )

    async def retry_delivery(self, webhook_id: str, delivery_id: str) -> dict[str, Any]:
        """Retry a failed delivery."""
        return await self._client.request(
            "POST", f"/api/v1/webhooks/{webhook_id}/deliveries/{delivery_id}/retry"
        )

    async def test(self, webhook_id: str, event_type: str | None = None) -> dict[str, Any]:
        """Send a test event to webhook."""
        data: dict[str, Any] = {}
        if event_type:
            data["event_type"] = event_type

        return await self._client.request("POST", f"/api/v1/webhooks/{webhook_id}/test", json=data)

    async def get_events(self) -> dict[str, Any]:
        """Get available webhook event types."""
        return await self._client.request("GET", "/api/v1/webhooks/events")
