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

    def list_messages(self, status: str | None = None, limit: int = 20) -> dict[str, Any]:
        """List inbox messages."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return self._client.request("GET", "/api/v1/shared-inbox/messages", params=params)

    def get_message(self, message_id: str) -> dict[str, Any]:
        """Get message by ID."""
        return self._client.request("GET", f"/api/v1/shared-inbox/messages/{message_id}")

    def assign(self, message_id: str, assignee_id: str) -> dict[str, Any]:
        """Assign message to a user."""
        return self._client.request(
            "POST",
            f"/api/v1/shared-inbox/messages/{message_id}/assign",
            json={
                "assignee_id": assignee_id,
            },
        )

    def resolve(self, message_id: str, resolution: str | None = None) -> dict[str, Any]:
        """Resolve a message."""
        data: dict[str, Any] = {}
        if resolution:
            data["resolution"] = resolution
        return self._client.request(
            "POST", f"/api/v1/shared-inbox/messages/{message_id}/resolve", json=data
        )

    def reply(self, message_id: str, content: str) -> dict[str, Any]:
        """Reply to a message."""
        return self._client.request(
            "POST",
            f"/api/v1/shared-inbox/messages/{message_id}/reply",
            json={
                "content": content,
            },
        )

    def get_sla_status(self, message_id: str) -> dict[str, Any]:
        """Get SLA status for a message."""
        return self._client.request("GET", f"/api/v1/shared-inbox/messages/{message_id}/sla")

    def list_rules(self, limit: int = 20) -> dict[str, Any]:
        """List routing rules."""
        return self._client.request("GET", "/api/v1/shared-inbox/rules", params={"limit": limit})

    def create_rule(
        self, name: str, conditions: dict[str, Any], actions: dict[str, Any]
    ) -> dict[str, Any]:
        """Create a routing rule."""
        return self._client.request(
            "POST",
            "/api/v1/shared-inbox/rules",
            json={
                "name": name,
                "conditions": conditions,
                "actions": actions,
            },
        )

    def get_stats(self) -> dict[str, Any]:
        """Get inbox statistics."""
        return self._client.request("GET", "/api/v1/shared-inbox/stats")


class AsyncSharedInboxAPI:
    """Asynchronous Shared Inbox API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list_messages(self, status: str | None = None, limit: int = 20) -> dict[str, Any]:
        """List inbox messages."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return await self._client.request("GET", "/api/v1/shared-inbox/messages", params=params)

    async def get_message(self, message_id: str) -> dict[str, Any]:
        """Get message by ID."""
        return await self._client.request("GET", f"/api/v1/shared-inbox/messages/{message_id}")

    async def assign(self, message_id: str, assignee_id: str) -> dict[str, Any]:
        """Assign message to a user."""
        return await self._client.request(
            "POST",
            f"/api/v1/shared-inbox/messages/{message_id}/assign",
            json={
                "assignee_id": assignee_id,
            },
        )

    async def resolve(self, message_id: str, resolution: str | None = None) -> dict[str, Any]:
        """Resolve a message."""
        data: dict[str, Any] = {}
        if resolution:
            data["resolution"] = resolution
        return await self._client.request(
            "POST", f"/api/v1/shared-inbox/messages/{message_id}/resolve", json=data
        )

    async def reply(self, message_id: str, content: str) -> dict[str, Any]:
        """Reply to a message."""
        return await self._client.request(
            "POST",
            f"/api/v1/shared-inbox/messages/{message_id}/reply",
            json={
                "content": content,
            },
        )

    async def get_sla_status(self, message_id: str) -> dict[str, Any]:
        """Get SLA status for a message."""
        return await self._client.request("GET", f"/api/v1/shared-inbox/messages/{message_id}/sla")

    async def list_rules(self, limit: int = 20) -> dict[str, Any]:
        """List routing rules."""
        return await self._client.request(
            "GET", "/api/v1/shared-inbox/rules", params={"limit": limit}
        )

    async def create_rule(
        self, name: str, conditions: dict[str, Any], actions: dict[str, Any]
    ) -> dict[str, Any]:
        """Create a routing rule."""
        return await self._client.request(
            "POST",
            "/api/v1/shared-inbox/rules",
            json={
                "name": name,
                "conditions": conditions,
                "actions": actions,
            },
        )

    async def get_stats(self) -> dict[str, Any]:
        """Get inbox statistics."""
        return await self._client.request("GET", "/api/v1/shared-inbox/stats")
