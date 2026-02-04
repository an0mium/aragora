"""
Support Namespace API

Provides methods for support ticket management:
- Create and manage tickets
- Escalation workflows
- Knowledge base search
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class SupportAPI:
    """Synchronous Support API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def create_ticket(
        self, subject: str, description: str, priority: str = "normal"
    ) -> dict[str, Any]:
        """Create a support ticket."""
        return self._client.request(
            "POST",
            "/api/v1/support/tickets",
            json={
                "subject": subject,
                "description": description,
                "priority": priority,
            },
        )

    def list_tickets(
        self, status: str | None = None, priority: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        """List support tickets."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if priority:
            params["priority"] = priority
        return self._client.request("GET", "/api/v1/support/tickets", params=params)

    def get_ticket(self, ticket_id: str) -> dict[str, Any]:
        """Get ticket by ID."""
        return self._client.request("GET", f"/api/v1/support/tickets/{ticket_id}")

    def update_ticket(self, ticket_id: str, **updates: Any) -> dict[str, Any]:
        """Update a ticket."""
        return self._client.request("PATCH", f"/api/v1/support/tickets/{ticket_id}", json=updates)

    def add_comment(self, ticket_id: str, content: str) -> dict[str, Any]:
        """Add comment to ticket."""
        return self._client.request(
            "POST",
            f"/api/v1/support/tickets/{ticket_id}/comments",
            json={
                "content": content,
            },
        )

    def escalate(self, ticket_id: str, reason: str) -> dict[str, Any]:
        """Escalate a ticket."""
        return self._client.request(
            "POST",
            f"/api/v1/support/tickets/{ticket_id}/escalate",
            json={
                "reason": reason,
            },
        )

    def resolve(self, ticket_id: str, resolution: str) -> dict[str, Any]:
        """Resolve a ticket."""
        return self._client.request(
            "POST",
            f"/api/v1/support/tickets/{ticket_id}/resolve",
            json={
                "resolution": resolution,
            },
        )

    def search_kb(self, query: str, limit: int = 10) -> dict[str, Any]:
        """Search knowledge base."""
        return self._client.request(
            "GET",
            "/api/v1/support/kb/search",
            params={
                "query": query,
                "limit": limit,
            },
        )

    def get_stats(self) -> dict[str, Any]:
        """Get support statistics."""
        return self._client.request("GET", "/api/v1/support/stats")


class AsyncSupportAPI:
    """Asynchronous Support API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def create_ticket(
        self, subject: str, description: str, priority: str = "normal"
    ) -> dict[str, Any]:
        """Create a support ticket."""
        return await self._client.request(
            "POST",
            "/api/v1/support/tickets",
            json={
                "subject": subject,
                "description": description,
                "priority": priority,
            },
        )

    async def list_tickets(
        self, status: str | None = None, priority: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        """List support tickets."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if priority:
            params["priority"] = priority
        return await self._client.request("GET", "/api/v1/support/tickets", params=params)

    async def get_ticket(self, ticket_id: str) -> dict[str, Any]:
        """Get ticket by ID."""
        return await self._client.request("GET", f"/api/v1/support/tickets/{ticket_id}")

    async def update_ticket(self, ticket_id: str, **updates: Any) -> dict[str, Any]:
        """Update a ticket."""
        return await self._client.request(
            "PATCH", f"/api/v1/support/tickets/{ticket_id}", json=updates
        )

    async def add_comment(self, ticket_id: str, content: str) -> dict[str, Any]:
        """Add comment to ticket."""
        return await self._client.request(
            "POST",
            f"/api/v1/support/tickets/{ticket_id}/comments",
            json={
                "content": content,
            },
        )

    async def escalate(self, ticket_id: str, reason: str) -> dict[str, Any]:
        """Escalate a ticket."""
        return await self._client.request(
            "POST",
            f"/api/v1/support/tickets/{ticket_id}/escalate",
            json={
                "reason": reason,
            },
        )

    async def resolve(self, ticket_id: str, resolution: str) -> dict[str, Any]:
        """Resolve a ticket."""
        return await self._client.request(
            "POST",
            f"/api/v1/support/tickets/{ticket_id}/resolve",
            json={
                "resolution": resolution,
            },
        )

    async def search_kb(self, query: str, limit: int = 10) -> dict[str, Any]:
        """Search knowledge base."""
        return await self._client.request(
            "GET",
            "/api/v1/support/kb/search",
            params={
                "query": query,
                "limit": limit,
            },
        )

    async def get_stats(self) -> dict[str, Any]:
        """Get support statistics."""
        return await self._client.request("GET", "/api/v1/support/stats")
