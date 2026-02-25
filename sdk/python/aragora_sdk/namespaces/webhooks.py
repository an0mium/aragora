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

    def get_event_categories(self) -> dict[str, Any]:
        """Get webhook event categories."""
        return self._client.request("GET", "/api/v1/webhooks/events/categories")

    def test_slo(self, webhook_id: str | None = None) -> dict[str, Any]:
        """Test webhook SLO delivery."""
        data: dict[str, Any] = {}
        if webhook_id:
            data["webhook_id"] = webhook_id
        return self._client.request("POST", "/api/v1/webhooks/slo/test", json=data)

    def bulk_create(self, webhooks: _List[dict[str, Any]]) -> dict[str, Any]:
        """Create multiple webhooks in bulk."""
        return self._client.request("POST", "/api/v1/webhooks/bulk", json={"webhooks": webhooks})

    def pause_all(self) -> dict[str, Any]:
        """Pause all webhooks."""
        return self._client.request("POST", "/api/v1/webhooks/pause-all")

    def resume_all(self) -> dict[str, Any]:
        """Resume all webhooks."""
        return self._client.request("POST", "/api/v1/webhooks/resume-all")

    def dispatch(self, **kwargs: Any) -> dict[str, Any]:
        """
        Manually dispatch a webhook event.

        Args:
            **kwargs: Event parameters (event_type, payload, webhook_id, etc.)

        Returns:
            Dict with dispatch result and delivery status.
        """
        return self._client.request("POST", "/api/webhooks/dispatch", json=kwargs)

    def list_platforms(self) -> dict[str, Any]:
        """
        List supported webhook platform integrations.

        Returns:
            Dict with supported platforms and their capabilities.
        """
        return self._client.request("GET", "/api/webhooks/platforms")


    # =========================================================================
    # Webhook CRUD
    # =========================================================================

    def get(self, webhook_id: str) -> dict[str, Any]:
        """Get a specific webhook by ID."""
        return self._client.request("GET", f"/api/v1/webhooks/{webhook_id}")

    def update(self, webhook_id: str, **updates: Any) -> dict[str, Any]:
        """Update an existing webhook."""
        return self._client.request("PATCH", f"/api/v1/webhooks/{webhook_id}", json=updates)

    def delete(self, webhook_id: str) -> dict[str, Any]:
        """Delete a webhook."""
        return self._client.request("DELETE", f"/api/v1/webhooks/{webhook_id}")

    def test(self, webhook_id: str) -> dict[str, Any]:
        """Test a webhook by sending a test event."""
        return self._client.request("POST", f"/api/v1/webhooks/{webhook_id}/test")

    # =========================================================================
    # Delivery Management
    # =========================================================================

    def list_deliveries(self, webhook_id: str, status: str | None = None, limit: int = 20, offset: int = 0) -> dict[str, Any]:
        """List webhook deliveries."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        return self._client.request("GET", f"/api/v1/webhooks/{webhook_id}/deliveries", params=params)

    def get_delivery(self, webhook_id: str, delivery_id: str) -> dict[str, Any]:
        """Get delivery details."""
        return self._client.request("GET", f"/api/v1/webhooks/{webhook_id}/deliveries/{delivery_id}")

    def retry_delivery(self, webhook_id: str, delivery_id: str) -> dict[str, Any]:
        """Retry a failed delivery."""
        return self._client.request("POST", f"/api/v1/webhooks/{webhook_id}/deliveries/{delivery_id}/retry")

    def get_delivery_stats(self, webhook_id: str, days: int | None = None) -> dict[str, Any]:
        """Get delivery stats for a webhook."""
        params: dict[str, Any] = {}
        if days is not None:
            params["days"] = days
        return self._client.request("GET", f"/api/v1/webhooks/{webhook_id}/stats", params=params)

    def subscribe_events(self, webhook_id: str, events: _List[str]) -> dict[str, Any]:
        """Subscribe to events for a webhook."""
        return self._client.request("POST", f"/api/v1/webhooks/{webhook_id}/events", json={"events": events})

    def unsubscribe_events(self, webhook_id: str, events: _List[str]) -> dict[str, Any]:
        """Unsubscribe from events for a webhook."""
        return self._client.request("DELETE", f"/api/v1/webhooks/{webhook_id}/events", json={"events": events})

    def get_retry_policy(self, webhook_id: str) -> dict[str, Any]:
        """Get webhook retry policy."""
        return self._client.request("GET", f"/api/v1/webhooks/{webhook_id}/retry-policy")

    def update_retry_policy(self, webhook_id: str, **policy: Any) -> dict[str, Any]:
        """Update webhook retry policy."""
        return self._client.request("PUT", f"/api/v1/webhooks/{webhook_id}/retry-policy", json=policy)

    def rotate_secret(self, webhook_id: str) -> dict[str, Any]:
        """Rotate webhook secret."""
        return self._client.request("POST", f"/api/v1/webhooks/{webhook_id}/rotate-secret")

    def get_signing_info(self, webhook_id: str) -> dict[str, Any]:
        """Get signing key info."""
        return self._client.request("GET", f"/api/v1/webhooks/{webhook_id}/signing")

    def list_dead_letter(self, limit: int | None = None) -> dict[str, Any]:
        """List deliveries in the dead-letter queue."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        return self._client.request("GET", "/api/v1/webhooks/dead-letter", params=params)

    def get_dead_letter(self, dead_letter_id: str) -> dict[str, Any]:
        """Get a specific dead-letter delivery by ID."""
        return self._client.request("GET", f"/api/v1/webhooks/dead-letter/{dead_letter_id}")

    def retry_dead_letter(self, dead_letter_id: str) -> dict[str, Any]:
        """Retry a dead-letter delivery."""
        return self._client.request("POST", f"/api/v1/webhooks/dead-letter/{dead_letter_id}/retry")


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

    async def get_event_categories(self) -> dict[str, Any]:
        """Get webhook event categories."""
        return await self._client.request("GET", "/api/v1/webhooks/events/categories")

    async def test_slo(self, webhook_id: str | None = None) -> dict[str, Any]:
        """Test webhook SLO delivery."""
        data: dict[str, Any] = {}
        if webhook_id:
            data["webhook_id"] = webhook_id
        return await self._client.request("POST", "/api/v1/webhooks/slo/test", json=data)

    async def bulk_create(self, webhooks: _List[dict[str, Any]]) -> dict[str, Any]:
        """Create multiple webhooks in bulk."""
        return await self._client.request(
            "POST", "/api/v1/webhooks/bulk", json={"webhooks": webhooks}
        )

    async def pause_all(self) -> dict[str, Any]:
        """Pause all webhooks."""
        return await self._client.request("POST", "/api/v1/webhooks/pause-all")

    async def resume_all(self) -> dict[str, Any]:
        """Resume all webhooks."""
        return await self._client.request("POST", "/api/v1/webhooks/resume-all")

    async def dispatch(self, **kwargs: Any) -> dict[str, Any]:
        """Manually dispatch a webhook event."""
        return await self._client.request("POST", "/api/webhooks/dispatch", json=kwargs)

    async def list_platforms(self) -> dict[str, Any]:
        """List supported webhook platform integrations."""
        return await self._client.request("GET", "/api/webhooks/platforms")

    # =========================================================================
    # Webhook CRUD
    # =========================================================================

    async def get(self, webhook_id: str) -> dict[str, Any]:
        """Get a specific webhook by ID."""
        return await self._client.request("GET", f"/api/v1/webhooks/{webhook_id}")

    async def update(self, webhook_id: str, **updates: Any) -> dict[str, Any]:
        """Update an existing webhook."""
        return await self._client.request("PATCH", f"/api/v1/webhooks/{webhook_id}", json=updates)

    async def delete(self, webhook_id: str) -> dict[str, Any]:
        """Delete a webhook."""
        return await self._client.request("DELETE", f"/api/v1/webhooks/{webhook_id}")

    async def test(self, webhook_id: str) -> dict[str, Any]:
        """Test a webhook by sending a test event."""
        return await self._client.request("POST", f"/api/v1/webhooks/{webhook_id}/test")

    # =========================================================================
    # Delivery Management
    # =========================================================================

    async def list_deliveries(self, webhook_id: str, status: str | None = None, limit: int = 20, offset: int = 0) -> dict[str, Any]:
        """List webhook deliveries."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        return await self._client.request("GET", f"/api/v1/webhooks/{webhook_id}/deliveries", params=params)

    async def get_delivery(self, webhook_id: str, delivery_id: str) -> dict[str, Any]:
        """Get delivery details."""
        return await self._client.request("GET", f"/api/v1/webhooks/{webhook_id}/deliveries/{delivery_id}")

    async def retry_delivery(self, webhook_id: str, delivery_id: str) -> dict[str, Any]:
        """Retry a failed delivery."""
        return await self._client.request("POST", f"/api/v1/webhooks/{webhook_id}/deliveries/{delivery_id}/retry")

    async def get_delivery_stats(self, webhook_id: str, days: int | None = None) -> dict[str, Any]:
        """Get delivery stats for a webhook."""
        params: dict[str, Any] = {}
        if days is not None:
            params["days"] = days
        return await self._client.request("GET", f"/api/v1/webhooks/{webhook_id}/stats", params=params)

    async def subscribe_events(self, webhook_id: str, events: _List[str]) -> dict[str, Any]:
        """Subscribe to events for a webhook."""
        return await self._client.request("POST", f"/api/v1/webhooks/{webhook_id}/events", json={"events": events})

    async def unsubscribe_events(self, webhook_id: str, events: _List[str]) -> dict[str, Any]:
        """Unsubscribe from events for a webhook."""
        return await self._client.request("DELETE", f"/api/v1/webhooks/{webhook_id}/events", json={"events": events})

    async def get_retry_policy(self, webhook_id: str) -> dict[str, Any]:
        """Get webhook retry policy."""
        return await self._client.request("GET", f"/api/v1/webhooks/{webhook_id}/retry-policy")

    async def update_retry_policy(self, webhook_id: str, **policy: Any) -> dict[str, Any]:
        """Update webhook retry policy."""
        return await self._client.request("PUT", f"/api/v1/webhooks/{webhook_id}/retry-policy", json=policy)

    async def rotate_secret(self, webhook_id: str) -> dict[str, Any]:
        """Rotate webhook secret."""
        return await self._client.request("POST", f"/api/v1/webhooks/{webhook_id}/rotate-secret")

    async def get_signing_info(self, webhook_id: str) -> dict[str, Any]:
        """Get signing key info."""
        return await self._client.request("GET", f"/api/v1/webhooks/{webhook_id}/signing")

    async def list_dead_letter(self, limit: int | None = None) -> dict[str, Any]:
        """List deliveries in the dead-letter queue."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        return await self._client.request("GET", "/api/v1/webhooks/dead-letter", params=params)

    async def get_dead_letter(self, dead_letter_id: str) -> dict[str, Any]:
        """Get a specific dead-letter delivery by ID."""
        return await self._client.request("GET", f"/api/v1/webhooks/dead-letter/{dead_letter_id}")

    async def retry_dead_letter(self, dead_letter_id: str) -> dict[str, Any]:
        """Retry a dead-letter delivery."""
        return await self._client.request("POST", f"/api/v1/webhooks/dead-letter/{dead_letter_id}/retry")


