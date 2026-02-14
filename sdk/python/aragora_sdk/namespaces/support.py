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

