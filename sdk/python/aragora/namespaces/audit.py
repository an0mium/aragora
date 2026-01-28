"""
Audit Namespace API

Provides methods for audit trail management:
- List and filter audit events
- Export audit data for compliance
- Generate compliance reports
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

ExportFormat = Literal["json", "csv", "pdf"]


class AuditAPI:
    """
    Synchronous Audit API.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> events = client.audit.list_events(event_type="debate.created")
        >>> for event in events["events"]:
        ...     print(event["timestamp"], event["actor_id"], event["action"])
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def list_events(
        self,
        event_type: str | None = None,
        actor_id: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List audit events with filtering.

        Args:
            event_type: Filter by event type (e.g., "debate.created")
            actor_id: Filter by actor (user/agent) ID
            resource_type: Filter by resource type
            resource_id: Filter by specific resource
            from_date: Start date (ISO format)
            to_date: End date (ISO format)
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of audit events with pagination
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if event_type:
            params["event_type"] = event_type
        if actor_id:
            params["actor_id"] = actor_id
        if resource_type:
            params["resource_type"] = resource_type
        if resource_id:
            params["resource_id"] = resource_id
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date

        return self._client.request("GET", "/api/v1/audit/events", params=params)

    def get_event(self, event_id: str) -> dict[str, Any]:
        """
        Get a specific audit event.

        Args:
            event_id: Event ID

        Returns:
            Audit event details
        """
        return self._client.request("GET", f"/api/v1/audit/events/{event_id}")

    def export(
        self,
        format: ExportFormat = "json",
        from_date: str | None = None,
        to_date: str | None = None,
        event_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Export audit events.

        Args:
            format: Export format (json, csv, pdf)
            from_date: Start date (ISO format)
            to_date: End date (ISO format)
            event_types: Filter by event types

        Returns:
            Export result with download URL or data
        """
        data: dict[str, Any] = {"format": format}
        if from_date:
            data["from_date"] = from_date
        if to_date:
            data["to_date"] = to_date
        if event_types:
            data["event_types"] = event_types

        return self._client.request("POST", "/api/v1/audit/export", json=data)

    def get_compliance_report(
        self,
        period: str = "monthly",
        framework: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate a compliance report.

        Args:
            period: Report period (daily, weekly, monthly, quarterly)
            framework: Compliance framework (soc2, gdpr, hipaa)

        Returns:
            Compliance report with metrics and findings
        """
        params: dict[str, Any] = {"period": period}
        if framework:
            params["framework"] = framework

        return self._client.request("GET", "/api/v1/audit/compliance/report", params=params)

    def get_actor_activity(
        self,
        actor_id: str,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict[str, Any]:
        """
        Get activity summary for an actor.

        Args:
            actor_id: Actor (user/agent) ID
            from_date: Start date
            to_date: End date

        Returns:
            Activity summary with event counts
        """
        params: dict[str, Any] = {"actor_id": actor_id}
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date

        return self._client.request("GET", "/api/v1/audit/actors/activity", params=params)

    def get_resource_history(
        self,
        resource_type: str,
        resource_id: str,
    ) -> dict[str, Any]:
        """
        Get audit history for a specific resource.

        Args:
            resource_type: Resource type (debate, agent, etc.)
            resource_id: Resource ID

        Returns:
            Resource audit history
        """
        return self._client.request(
            "GET", f"/api/v1/audit/resources/{resource_type}/{resource_id}/history"
        )

    def get_stats(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict[str, Any]:
        """
        Get audit statistics.

        Args:
            from_date: Start date
            to_date: End date

        Returns:
            Statistics including event counts by type, top actors, etc.
        """
        params: dict[str, Any] = {}
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date

        return self._client.request("GET", "/api/v1/audit/stats", params=params)


class AsyncAuditAPI:
    """
    Asynchronous Audit API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     events = await client.audit.list_events(event_type="debate.created")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list_events(
        self,
        event_type: str | None = None,
        actor_id: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List audit events with filtering."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if event_type:
            params["event_type"] = event_type
        if actor_id:
            params["actor_id"] = actor_id
        if resource_type:
            params["resource_type"] = resource_type
        if resource_id:
            params["resource_id"] = resource_id
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date

        return await self._client.request("GET", "/api/v1/audit/events", params=params)

    async def get_event(self, event_id: str) -> dict[str, Any]:
        """Get a specific audit event."""
        return await self._client.request("GET", f"/api/v1/audit/events/{event_id}")

    async def export(
        self,
        format: ExportFormat = "json",
        from_date: str | None = None,
        to_date: str | None = None,
        event_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Export audit events."""
        data: dict[str, Any] = {"format": format}
        if from_date:
            data["from_date"] = from_date
        if to_date:
            data["to_date"] = to_date
        if event_types:
            data["event_types"] = event_types

        return await self._client.request("POST", "/api/v1/audit/export", json=data)

    async def get_compliance_report(
        self,
        period: str = "monthly",
        framework: str | None = None,
    ) -> dict[str, Any]:
        """Generate a compliance report."""
        params: dict[str, Any] = {"period": period}
        if framework:
            params["framework"] = framework

        return await self._client.request("GET", "/api/v1/audit/compliance/report", params=params)

    async def get_actor_activity(
        self,
        actor_id: str,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict[str, Any]:
        """Get activity summary for an actor."""
        params: dict[str, Any] = {"actor_id": actor_id}
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date

        return await self._client.request("GET", "/api/v1/audit/actors/activity", params=params)

    async def get_resource_history(
        self,
        resource_type: str,
        resource_id: str,
    ) -> dict[str, Any]:
        """Get audit history for a specific resource."""
        return await self._client.request(
            "GET", f"/api/v1/audit/resources/{resource_type}/{resource_id}/history"
        )

    async def get_stats(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict[str, Any]:
        """Get audit statistics."""
        params: dict[str, Any] = {}
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date

        return await self._client.request("GET", "/api/v1/audit/stats", params=params)
