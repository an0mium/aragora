"""
Support Namespace API

Provides methods for unified customer support platform management:
- Platform connections (Zendesk, Freshdesk, Intercom, Help Scout)
- Cross-platform ticket management
- AI-powered ticket triage and auto-response
- Support metrics overview
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class SupportAPI:
    """
    Synchronous Support API.

    Unified interface for customer support platforms including Zendesk,
    Freshdesk, Intercom, and Help Scout. Supports cross-platform ticket
    management, AI-powered triage, and metrics.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> platforms = client.support.list_platforms()
        >>> tickets = client.support.list_tickets(status="open")
        >>> triage = client.support.triage(ticket_id="ticket-123")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # =========================================================================
    # Platform Management
    # =========================================================================

    def list_platforms(self) -> dict[str, Any]:
        """
        List connected support platforms.

        Returns:
            Dict with connected platforms and their configurations.
        """
        return self._client.request("GET", "/api/v1/support/platforms")

    def connect(self, **kwargs: Any) -> dict[str, Any]:
        """
        Connect a support platform.

        Args:
            **kwargs: Connection parameters including:
                - platform: Platform type (zendesk, freshdesk, intercom, helpscout)
                - api_key: Platform API key
                - domain: Platform domain (e.g., 'company.zendesk.com')

        Returns:
            Dict with connection status and platform details.
        """
        return self._client.request("POST", "/api/v1/support/connect", json=kwargs)

    def disconnect(self, platform: str) -> dict[str, Any]:
        """
        Disconnect a support platform.

        Args:
            platform: Platform identifier to disconnect.

        Returns:
            Dict with disconnection confirmation.
        """
        return self._client.request("DELETE", f"/api/v1/support/{platform}")

    # =========================================================================
    # Ticket Management
    # =========================================================================

    def list_tickets(
        self,
        status: str | None = None,
        priority: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """
        List support tickets across all connected platforms.

        Args:
            status: Filter by ticket status (open, pending, resolved, closed).
            priority: Filter by priority (low, normal, high, urgent).
            limit: Maximum number of tickets to return.

        Returns:
            Dict with tickets list and pagination info.
        """
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if priority:
            params["priority"] = priority
        return self._client.request("GET", "/api/v1/support/tickets", params=params)

    def list_platform_tickets(
        self,
        platform: str,
        status: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """
        List tickets for a specific platform.

        Args:
            platform: Platform identifier (zendesk, freshdesk, etc.).
            status: Filter by ticket status.
            limit: Maximum number of tickets to return.

        Returns:
            Dict with platform-specific tickets.
        """
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return self._client.request(
            "GET", f"/api/v1/support/{platform}/tickets", params=params
        )

    def create_ticket(
        self,
        platform: str,
        subject: str,
        description: str,
        priority: str = "normal",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Create a support ticket on a specific platform.

        Args:
            platform: Platform identifier.
            subject: Ticket subject.
            description: Ticket description.
            priority: Ticket priority (low, normal, high, urgent).
            **kwargs: Additional platform-specific fields.

        Returns:
            Dict with created ticket details.
        """
        data: dict[str, Any] = {
            "subject": subject,
            "description": description,
            "priority": priority,
            **kwargs,
        }
        return self._client.request(
            "POST", f"/api/v1/support/{platform}/tickets", json=data
        )

    def update_ticket(
        self, platform: str, ticket_id: str, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Update a support ticket.

        Args:
            platform: Platform identifier.
            ticket_id: Ticket identifier.
            **kwargs: Fields to update (status, priority, assignee, etc.).

        Returns:
            Dict with updated ticket details.
        """
        return self._client.request(
            "PUT", f"/api/v1/support/{platform}/tickets/{ticket_id}", json=kwargs
        )

    def reply_to_ticket(
        self, platform: str, ticket_id: str, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Reply to a support ticket.

        Args:
            platform: Platform identifier.
            ticket_id: Ticket identifier.
            **kwargs: Reply content including:
                - body: Reply text
                - internal: Whether this is an internal note

        Returns:
            Dict with reply confirmation.
        """
        return self._client.request(
            "POST", f"/api/v1/support/{platform}/tickets/{ticket_id}/reply", json=kwargs
        )

    # =========================================================================
    # AI Features
    # =========================================================================

    def triage(self, **kwargs: Any) -> dict[str, Any]:
        """
        AI-powered ticket triage.

        Analyzes ticket content and suggests priority, category,
        and assignment.

        Args:
            **kwargs: Triage parameters including:
                - ticket_id: Ticket to triage
                - content: Ticket content for ad-hoc triage

        Returns:
            Dict with triage suggestions including priority, category,
            suggested assignee, and confidence scores.
        """
        return self._client.request("POST", "/api/v1/support/triage", json=kwargs)

    def auto_respond(self, **kwargs: Any) -> dict[str, Any]:
        """
        Generate AI-powered response suggestions for a ticket.

        Args:
            **kwargs: Auto-respond parameters including:
                - ticket_id: Ticket to generate responses for
                - tone: Response tone (professional, friendly, formal)
                - max_suggestions: Number of suggestions to generate

        Returns:
            Dict with response suggestions and confidence scores.
        """
        return self._client.request("POST", "/api/v1/support/auto-respond", json=kwargs)

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """
        Search across support tickets and knowledge base.

        Args:
            query: Search query string.
            **kwargs: Additional search filters.

        Returns:
            Dict with search results from tickets and knowledge base.
        """
        params: dict[str, Any] = {"query": query, **kwargs}
        return self._client.request("GET", "/api/v1/support/search", params=params)

    # =========================================================================
    # Metrics
    # =========================================================================

    def get_metrics(self) -> dict[str, Any]:
        """
        Get support metrics overview.

        Returns:
            Dict with support metrics including response times,
            resolution rates, and ticket volume trends.
        """
        return self._client.request("GET", "/api/v1/support/metrics")


class AsyncSupportAPI:
    """
    Asynchronous Support API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     platforms = await client.support.list_platforms()
        ...     tickets = await client.support.list_tickets(status="open")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # =========================================================================
    # Platform Management
    # =========================================================================

    async def list_platforms(self) -> dict[str, Any]:
        """List connected support platforms."""
        return await self._client.request("GET", "/api/v1/support/platforms")

    async def connect(self, **kwargs: Any) -> dict[str, Any]:
        """Connect a support platform."""
        return await self._client.request("POST", "/api/v1/support/connect", json=kwargs)

    async def disconnect(self, platform: str) -> dict[str, Any]:
        """Disconnect a support platform."""
        return await self._client.request("DELETE", f"/api/v1/support/{platform}")

    # =========================================================================
    # Ticket Management
    # =========================================================================

    async def list_tickets(
        self,
        status: str | None = None,
        priority: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """List support tickets across all connected platforms."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if priority:
            params["priority"] = priority
        return await self._client.request("GET", "/api/v1/support/tickets", params=params)

    async def list_platform_tickets(
        self,
        platform: str,
        status: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """List tickets for a specific platform."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return await self._client.request(
            "GET", f"/api/v1/support/{platform}/tickets", params=params
        )

    async def create_ticket(
        self,
        platform: str,
        subject: str,
        description: str,
        priority: str = "normal",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a support ticket on a specific platform."""
        data: dict[str, Any] = {
            "subject": subject,
            "description": description,
            "priority": priority,
            **kwargs,
        }
        return await self._client.request(
            "POST", f"/api/v1/support/{platform}/tickets", json=data
        )

    async def update_ticket(
        self, platform: str, ticket_id: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Update a support ticket."""
        return await self._client.request(
            "PUT", f"/api/v1/support/{platform}/tickets/{ticket_id}", json=kwargs
        )

    async def reply_to_ticket(
        self, platform: str, ticket_id: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Reply to a support ticket."""
        return await self._client.request(
            "POST",
            f"/api/v1/support/{platform}/tickets/{ticket_id}/reply",
            json=kwargs,
        )

    # =========================================================================
    # AI Features
    # =========================================================================

    async def triage(self, **kwargs: Any) -> dict[str, Any]:
        """AI-powered ticket triage."""
        return await self._client.request("POST", "/api/v1/support/triage", json=kwargs)

    async def auto_respond(self, **kwargs: Any) -> dict[str, Any]:
        """Generate AI-powered response suggestions for a ticket."""
        return await self._client.request(
            "POST", "/api/v1/support/auto-respond", json=kwargs
        )

    async def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Search across support tickets and knowledge base."""
        params: dict[str, Any] = {"query": query, **kwargs}
        return await self._client.request("GET", "/api/v1/support/search", params=params)

    # =========================================================================
    # Metrics
    # =========================================================================

    async def get_metrics(self) -> dict[str, Any]:
        """Get support metrics overview."""
        return await self._client.request("GET", "/api/v1/support/metrics")
