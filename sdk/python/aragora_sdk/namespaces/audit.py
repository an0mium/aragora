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

    # OpenAPI-aligned session endpoints
    def list_entries(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List audit entries."""
        return self._client.request(
            "GET", "/api/v1/audit/entries", params={"limit": limit, "offset": offset}
        )

    def get_report(self) -> dict[str, Any]:
        """Get audit report."""
        return self._client.request("GET", "/api/v1/audit/report")

    def verify(self) -> dict[str, Any]:
        """Verify audit integrity."""
        return self._client.request("GET", "/api/v1/audit/verify")

    def list_sessions(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List audit sessions."""
        return self._client.request(
            "GET", "/api/v1/audit/sessions", params={"limit": limit, "offset": offset}
        )

    def create_session(self, name: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Create an audit session."""
        data: dict[str, Any] = {"name": name}
        if config:
            data["config"] = config
        return self._client.request("POST", "/api/v1/audit/sessions", json=data)

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Get an audit session by ID."""
        return self._client.request("GET", f"/api/v1/audit/sessions/{session_id}")

    def delete_session(self, session_id: str) -> dict[str, Any]:
        """Delete an audit session."""
        return self._client.request("DELETE", f"/api/v1/audit/sessions/{session_id}")

    def get_session_events(self, session_id: str) -> dict[str, Any]:
        """Get events for an audit session."""
        return self._client.request("GET", f"/api/v1/audit/sessions/{session_id}/events")

    def get_session_findings(self, session_id: str) -> dict[str, Any]:
        """Get findings for an audit session."""
        return self._client.request("GET", f"/api/v1/audit/sessions/{session_id}/findings")

    def get_session_report(self, session_id: str) -> dict[str, Any]:
        """Get report for an audit session."""
        return self._client.request("GET", f"/api/v1/audit/sessions/{session_id}/report")

    def start_session(self, session_id: str) -> dict[str, Any]:
        """Start an audit session."""
        return self._client.request("POST", f"/api/v1/audit/sessions/{session_id}/start")

    def pause_session(self, session_id: str) -> dict[str, Any]:
        """Pause an audit session."""
        return self._client.request("POST", f"/api/v1/audit/sessions/{session_id}/pause")

    def resume_session(self, session_id: str) -> dict[str, Any]:
        """Resume an audit session."""
        return self._client.request("POST", f"/api/v1/audit/sessions/{session_id}/resume")

    def cancel_session(self, session_id: str) -> dict[str, Any]:
        """Cancel an audit session."""
        return self._client.request("POST", f"/api/v1/audit/sessions/{session_id}/cancel")

    def intervene_session(self, session_id: str, action: str) -> dict[str, Any]:
        """Intervene in an audit session."""
        return self._client.request(
            "POST", f"/api/v1/audit/sessions/{session_id}/intervene", json={"action": action}
        )

    def end_session(self, session_id: str) -> dict[str, Any]:
        """
        End an audit session.

        Args:
            session_id: Session ID

        Returns:
            Dict with session status and summary
        """
        return self._client.request("POST", f"/api/v1/audit/sessions/{session_id}/end")

    def export_session(
        self,
        session_id: str,
        format: ExportFormat = "json",
    ) -> dict[str, Any]:
        """
        Export an audit session.

        Args:
            session_id: Session ID
            format: Export format (json, csv, pdf)

        Returns:
            Dict with export URL or data
        """
        return self._client.request(
            "POST",
            f"/api/v1/audit/sessions/{session_id}/export",
            json={"format": format},
        )

    # ===========================================================================
    # Finding Management
    # ===========================================================================

    def list_findings(
        self,
        session_id: str | None = None,
        status: str | None = None,
        priority: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List audit findings.

        Args:
            session_id: Filter by session ID
            status: Filter by status (open, in_progress, resolved, dismissed)
            priority: Filter by priority (critical, high, medium, low)
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of findings with pagination
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if session_id:
            params["session_id"] = session_id
        if status:
            params["status"] = status
        if priority:
            params["priority"] = priority

        return self._client.request("GET", "/api/v1/audit/findings", params=params)

    def get_finding(self, finding_id: str) -> dict[str, Any]:
        """
        Get a specific audit finding.

        Args:
            finding_id: Finding ID

        Returns:
            Finding details
        """
        return self._client.request("GET", f"/api/v1/audit/findings/{finding_id}")

    def assign_finding(
        self,
        finding_id: str,
        assignee_id: str,
    ) -> dict[str, Any]:
        """
        Assign a finding to a user.

        Args:
            finding_id: Finding ID
            assignee_id: User ID to assign to

        Returns:
            Dict with success status
        """
        return self._client.request(
            "POST",
            f"/api/v1/audit/findings/{finding_id}/assign",
            json={"assignee_id": assignee_id},
        )

    def unassign_finding(self, finding_id: str) -> dict[str, Any]:
        """
        Unassign a finding.

        Args:
            finding_id: Finding ID

        Returns:
            Dict with success status
        """
        return self._client.request(
            "POST",
            f"/api/v1/audit/findings/{finding_id}/unassign",
        )

    def update_finding_status(
        self,
        finding_id: str,
        status: str,
        resolution_notes: str | None = None,
    ) -> dict[str, Any]:
        """
        Update finding status.

        Args:
            finding_id: Finding ID
            status: New status (open, in_progress, resolved, dismissed)
            resolution_notes: Optional notes about resolution

        Returns:
            Updated finding
        """
        data: dict[str, Any] = {"status": status}
        if resolution_notes:
            data["resolution_notes"] = resolution_notes

        return self._client.request(
            "PATCH",
            f"/api/v1/audit/findings/{finding_id}/status",
            json=data,
        )

    def update_finding_priority(
        self,
        finding_id: str,
        priority: str,
    ) -> dict[str, Any]:
        """
        Update finding priority.

        Args:
            finding_id: Finding ID
            priority: New priority (critical, high, medium, low)

        Returns:
            Updated finding
        """
        return self._client.request(
            "PATCH",
            f"/api/v1/audit/findings/{finding_id}/priority",
            json={"priority": priority},
        )

    def add_finding_comment(
        self,
        finding_id: str,
        content: str,
    ) -> dict[str, Any]:
        """
        Add a comment to a finding.

        Args:
            finding_id: Finding ID
            content: Comment text

        Returns:
            Created comment
        """
        return self._client.request(
            "POST",
            f"/api/v1/audit/findings/{finding_id}/comments",
            json={"content": content},
        )

    def list_finding_comments(self, finding_id: str) -> dict[str, Any]:
        """
        List comments on a finding.

        Args:
            finding_id: Finding ID

        Returns:
            List of comments
        """
        return self._client.request(
            "GET",
            f"/api/v1/audit/findings/{finding_id}/comments",
        )


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

    async def list_entries(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List audit entries."""
        return await self._client.request(
            "GET", "/api/v1/audit/entries", params={"limit": limit, "offset": offset}
        )

    async def get_report(self) -> dict[str, Any]:
        """Get audit report."""
        return await self._client.request("GET", "/api/v1/audit/report")

    async def verify(self) -> dict[str, Any]:
        """Verify audit integrity."""
        return await self._client.request("GET", "/api/v1/audit/verify")

    async def list_sessions(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List audit sessions."""
        return await self._client.request(
            "GET", "/api/v1/audit/sessions", params={"limit": limit, "offset": offset}
        )

    async def create_session(
        self, name: str, config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Create an audit session."""
        data: dict[str, Any] = {"name": name}
        if config:
            data["config"] = config
        return await self._client.request("POST", "/api/v1/audit/sessions", json=data)

    async def get_session(self, session_id: str) -> dict[str, Any]:
        """Get an audit session by ID."""
        return await self._client.request("GET", f"/api/v1/audit/sessions/{session_id}")

    async def delete_session(self, session_id: str) -> dict[str, Any]:
        """Delete an audit session."""
        return await self._client.request("DELETE", f"/api/v1/audit/sessions/{session_id}")

    async def get_session_events(self, session_id: str) -> dict[str, Any]:
        """Get events for an audit session."""
        return await self._client.request("GET", f"/api/v1/audit/sessions/{session_id}/events")

    async def get_session_findings(self, session_id: str) -> dict[str, Any]:
        """Get findings for an audit session."""
        return await self._client.request("GET", f"/api/v1/audit/sessions/{session_id}/findings")

    async def get_session_report(self, session_id: str) -> dict[str, Any]:
        """Get report for an audit session."""
        return await self._client.request("GET", f"/api/v1/audit/sessions/{session_id}/report")

    async def start_session(self, session_id: str) -> dict[str, Any]:
        """Start an audit session."""
        return await self._client.request("POST", f"/api/v1/audit/sessions/{session_id}/start")

    async def pause_session(self, session_id: str) -> dict[str, Any]:
        """Pause an audit session."""
        return await self._client.request("POST", f"/api/v1/audit/sessions/{session_id}/pause")

    async def resume_session(self, session_id: str) -> dict[str, Any]:
        """Resume an audit session."""
        return await self._client.request("POST", f"/api/v1/audit/sessions/{session_id}/resume")

    async def cancel_session(self, session_id: str) -> dict[str, Any]:
        """Cancel an audit session."""
        return await self._client.request("POST", f"/api/v1/audit/sessions/{session_id}/cancel")

    async def intervene_session(self, session_id: str, action: str) -> dict[str, Any]:
        """Intervene in an audit session."""
        return await self._client.request(
            "POST", f"/api/v1/audit/sessions/{session_id}/intervene", json={"action": action}
        )

    async def end_session(self, session_id: str) -> dict[str, Any]:
        """End an audit session."""
        return await self._client.request("POST", f"/api/v1/audit/sessions/{session_id}/end")

    async def export_session(
        self,
        session_id: str,
        format: ExportFormat = "json",
    ) -> dict[str, Any]:
        """Export an audit session."""
        return await self._client.request(
            "POST",
            f"/api/v1/audit/sessions/{session_id}/export",
            json={"format": format},
        )

    # ===========================================================================
    # Finding Management
    # ===========================================================================

    async def list_findings(
        self,
        session_id: str | None = None,
        status: str | None = None,
        priority: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List audit findings."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if session_id:
            params["session_id"] = session_id
        if status:
            params["status"] = status
        if priority:
            params["priority"] = priority

        return await self._client.request("GET", "/api/v1/audit/findings", params=params)

    async def get_finding(self, finding_id: str) -> dict[str, Any]:
        """Get a specific audit finding."""
        return await self._client.request("GET", f"/api/v1/audit/findings/{finding_id}")

    async def assign_finding(
        self,
        finding_id: str,
        assignee_id: str,
    ) -> dict[str, Any]:
        """Assign a finding to a user."""
        return await self._client.request(
            "POST",
            f"/api/v1/audit/findings/{finding_id}/assign",
            json={"assignee_id": assignee_id},
        )

    async def unassign_finding(self, finding_id: str) -> dict[str, Any]:
        """Unassign a finding."""
        return await self._client.request(
            "POST",
            f"/api/v1/audit/findings/{finding_id}/unassign",
        )

    async def update_finding_status(
        self,
        finding_id: str,
        status: str,
        resolution_notes: str | None = None,
    ) -> dict[str, Any]:
        """Update finding status."""
        data: dict[str, Any] = {"status": status}
        if resolution_notes:
            data["resolution_notes"] = resolution_notes

        return await self._client.request(
            "PATCH",
            f"/api/v1/audit/findings/{finding_id}/status",
            json=data,
        )

    async def update_finding_priority(
        self,
        finding_id: str,
        priority: str,
    ) -> dict[str, Any]:
        """Update finding priority."""
        return await self._client.request(
            "PATCH",
            f"/api/v1/audit/findings/{finding_id}/priority",
            json={"priority": priority},
        )

    async def add_finding_comment(
        self,
        finding_id: str,
        content: str,
    ) -> dict[str, Any]:
        """Add a comment to a finding."""
        return await self._client.request(
            "POST",
            f"/api/v1/audit/findings/{finding_id}/comments",
            json={"content": content},
        )

    async def list_finding_comments(self, finding_id: str) -> dict[str, Any]:
        """List comments on a finding."""
        return await self._client.request(
            "GET",
            f"/api/v1/audit/findings/{finding_id}/comments",
        )
