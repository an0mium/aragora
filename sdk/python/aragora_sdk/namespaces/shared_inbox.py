"""
Shared Inbox Namespace API

Provides methods for shared inbox management:
- Message handling
- Assignment and routing
- SLA tracking
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

class SharedInboxAPI:
    """Synchronous Shared Inbox API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def list_shared(self, **kwargs: Any) -> dict[str, Any]:
        """
        List shared inbox items.

        Returns:
            Dict with shared inbox items
        """
        return self._client.request("GET", "/api/v1/inbox/shared", params=kwargs)

    def list_routing_rules(self) -> dict[str, Any]:
        """
        List inbox routing rules.

        Returns:
            Dict with routing rules
        """
        return self._client.request("GET", "/api/v1/inbox/routing/rules")

    def list_mentions(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List inbox mentions for the current user.

        Args:
            limit: Maximum mentions to return.
            offset: Pagination offset.

        Returns:
            Dict with mentions array and count.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return self._client.request("GET", "/api/v1/inbox/mentions", params=params)


class AsyncSharedInboxAPI:
    """Asynchronous Shared Inbox API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list_shared(self, **kwargs: Any) -> dict[str, Any]:
        """List shared inbox items."""
        return await self._client.request("GET", "/api/v1/inbox/shared", params=kwargs)

    async def list_routing_rules(self) -> dict[str, Any]:
        """List inbox routing rules."""
        return await self._client.request("GET", "/api/v1/inbox/routing/rules")

    async def list_mentions(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List inbox mentions for the current user."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        return await self._client.request("GET", "/api/v1/inbox/mentions", params=params)

