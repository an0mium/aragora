"""
Audit Namespace API

Provides methods for audit trail management:
- List and query audit entries
- Session lifecycle management (create, start, pause, resume, cancel, delete)
- Finding management (assign, status, priority, comments, linking)
- Compliance reporting and verification
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class AuditAPI:
    """
    Synchronous Audit API.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> entries = client.audit.list_entries()
        >>> report = client.audit.get_report()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # =========================================================================
    # Entries, Report & Verification
    # =========================================================================

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

    # =========================================================================
    # Session Management
    # =========================================================================

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
        """Get details for a specific audit session.

        Args:
            session_id: The audit session identifier.

        Returns:
            Dict with session details including status, progress, and config.
        """
        return self._client.request(
            "GET", f"/api/v1/audit/sessions/{session_id}"
        )

    def delete_session(self, session_id: str) -> dict[str, Any]:
        """Delete an audit session.

        The session must not be in a running state. Pause or cancel it first.

        Args:
            session_id: The audit session identifier.

        Returns:
            Dict with deletion confirmation.
        """
        return self._client.request(
            "DELETE", f"/api/v1/audit/sessions/{session_id}"
        )

    def start_session(self, session_id: str) -> dict[str, Any]:
        """Start a pending or paused audit session.

        Args:
            session_id: The audit session identifier.

        Returns:
            Dict with updated session details.
        """
        return self._client.request(
            "POST", f"/api/v1/audit/sessions/{session_id}/start"
        )

    def pause_session(self, session_id: str) -> dict[str, Any]:
        """Pause a running audit session.

        Args:
            session_id: The audit session identifier.

        Returns:
            Dict with updated session details.
        """
        return self._client.request(
            "POST", f"/api/v1/audit/sessions/{session_id}/pause"
        )

    def resume_session(self, session_id: str) -> dict[str, Any]:
        """Resume a paused audit session.

        Args:
            session_id: The audit session identifier.

        Returns:
            Dict with updated session details.
        """
        return self._client.request(
            "POST", f"/api/v1/audit/sessions/{session_id}/resume"
        )

    def cancel_session(
        self, session_id: str, reason: str | None = None
    ) -> dict[str, Any]:
        """Cancel an audit session.

        Args:
            session_id: The audit session identifier.
            reason: Optional cancellation reason.

        Returns:
            Dict with updated session details.
        """
        data: dict[str, Any] = {}
        if reason:
            data["reason"] = reason
        return self._client.request(
            "POST", f"/api/v1/audit/sessions/{session_id}/cancel", json=data
        )

    def get_session_findings(
        self,
        session_id: str,
        severity: str | None = None,
        audit_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Get findings for an audit session.

        Args:
            session_id: The audit session identifier.
            severity: Filter by severity (critical, high, medium, low, info).
            audit_type: Filter by audit type.
            status: Filter by finding status.
            limit: Maximum findings to return.
            offset: Pagination offset.

        Returns:
            Dict with findings array and pagination info.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if severity:
            params["severity"] = severity
        if audit_type:
            params["audit_type"] = audit_type
        if status:
            params["status"] = status
        return self._client.request(
            "GET", f"/api/v1/audit/sessions/{session_id}/findings", params=params
        )

    def get_session_events(self, session_id: str) -> dict[str, Any]:
        """Get the event stream URL for an audit session.

        The actual event stream uses SSE (Server-Sent Events). This method
        returns the connection information for establishing the stream.

        Args:
            session_id: The audit session identifier.

        Returns:
            Dict with SSE event stream connection info.
        """
        return self._client.request(
            "GET", f"/api/v1/audit/sessions/{session_id}/events"
        )

    def intervene_session(
        self,
        session_id: str,
        action: str,
        finding_id: str | None = None,
        reason: str | None = None,
        context: str | None = None,
    ) -> dict[str, Any]:
        """Submit a human intervention for an audit session.

        Args:
            session_id: The audit session identifier.
            action: Intervention action (approve_finding, reject_finding,
                    add_context, override_decision).
            finding_id: Finding to act on (for finding-related actions).
            reason: Reason for the intervention.
            context: Additional context to inject.

        Returns:
            Dict with intervention confirmation.
        """
        data: dict[str, Any] = {"action": action}
        if finding_id:
            data["finding_id"] = finding_id
        if reason:
            data["reason"] = reason
        if context:
            data["context"] = context
        return self._client.request(
            "POST", f"/api/v1/audit/sessions/{session_id}/intervene", json=data
        )

    def export_session_report(
        self,
        session_id: str,
        format: str = "markdown",
        template: str | None = None,
        min_severity: str | None = None,
    ) -> dict[str, Any]:
        """Export an audit session report.

        Args:
            session_id: The audit session identifier.
            format: Report format (json, markdown, html, pdf).
            template: Report template (executive_summary, detailed_findings,
                      compliance_attestation, security_assessment).
            min_severity: Minimum severity to include (critical, high, medium,
                          low, info).

        Returns:
            Dict with report content and metadata.
        """
        params: dict[str, Any] = {"format": format}
        if template:
            params["template"] = template
        if min_severity:
            params["min_severity"] = min_severity
        return self._client.request(
            "GET", f"/api/v1/audit/sessions/{session_id}/report", params=params
        )

    # =========================================================================
    # Finding Workflow Management
    # =========================================================================

    def update_finding_status(
        self, finding_id: str, status: str, reason: str | None = None
    ) -> dict[str, Any]:
        """Update the workflow status of a finding.

        Args:
            finding_id: The finding identifier.
            status: New status value (e.g. open, in_progress, resolved, closed).
            reason: Optional reason for the status change.

        Returns:
            Dict with updated finding details.
        """
        data: dict[str, Any] = {"status": status}
        if reason:
            data["reason"] = reason
        return self._client.request(
            "PATCH", f"/api/v1/audit/findings/{finding_id}/status", json=data
        )

    def assign_finding(
        self, finding_id: str, assignee: str, note: str | None = None
    ) -> dict[str, Any]:
        """Assign a finding to a user.

        Args:
            finding_id: The finding identifier.
            assignee: User ID or email to assign to.
            note: Optional note for the assignee.

        Returns:
            Dict with assignment confirmation.
        """
        data: dict[str, Any] = {"assignee": assignee}
        if note:
            data["note"] = note
        return self._client.request(
            "PATCH", f"/api/v1/audit/findings/{finding_id}/assign", json=data
        )

    def unassign_finding(self, finding_id: str) -> dict[str, Any]:
        """Remove the assignment from a finding.

        Args:
            finding_id: The finding identifier.

        Returns:
            Dict with confirmation.
        """
        return self._client.request(
            "POST", f"/api/v1/audit/findings/{finding_id}/unassign"
        )

    def add_finding_comment(
        self, finding_id: str, text: str
    ) -> dict[str, Any]:
        """Add a comment to a finding.

        Args:
            finding_id: The finding identifier.
            text: Comment text.

        Returns:
            Dict with the created comment.
        """
        return self._client.request(
            "POST",
            f"/api/v1/audit/findings/{finding_id}/comments",
            json={"text": text},
        )

    def get_finding_history(self, finding_id: str) -> dict[str, Any]:
        """Get the workflow history of a finding.

        Args:
            finding_id: The finding identifier.

        Returns:
            Dict with history entries (status changes, assignments, comments).
        """
        return self._client.request(
            "GET", f"/api/v1/audit/findings/{finding_id}/history"
        )

    def set_finding_priority(
        self, finding_id: str, priority: str
    ) -> dict[str, Any]:
        """Set the priority of a finding.

        Args:
            finding_id: The finding identifier.
            priority: Priority level (e.g. critical, high, medium, low).

        Returns:
            Dict with updated finding details.
        """
        return self._client.request(
            "PATCH",
            f"/api/v1/audit/findings/{finding_id}/priority",
            json={"priority": priority},
        )

    def set_finding_due_date(
        self, finding_id: str, due_date: str
    ) -> dict[str, Any]:
        """Set the due date for a finding.

        Args:
            finding_id: The finding identifier.
            due_date: Due date in ISO 8601 format.

        Returns:
            Dict with updated finding details.
        """
        return self._client.request(
            "PATCH",
            f"/api/v1/audit/findings/{finding_id}/due-date",
            json={"due_date": due_date},
        )

    def link_finding(
        self, finding_id: str, target_finding_id: str, link_type: str = "related"
    ) -> dict[str, Any]:
        """Link a finding to another finding.

        Args:
            finding_id: The source finding identifier.
            target_finding_id: The target finding to link to.
            link_type: Relationship type (related, blocks, caused_by).

        Returns:
            Dict with link confirmation.
        """
        return self._client.request(
            "POST",
            f"/api/v1/audit/findings/{finding_id}/link",
            json={"target_finding_id": target_finding_id, "link_type": link_type},
        )

    def mark_finding_duplicate(
        self, finding_id: str, duplicate_of: str
    ) -> dict[str, Any]:
        """Mark a finding as a duplicate of another finding.

        Args:
            finding_id: The finding to mark as duplicate.
            duplicate_of: The canonical finding identifier.

        Returns:
            Dict with confirmation.
        """
        return self._client.request(
            "POST",
            f"/api/v1/audit/findings/{finding_id}/duplicate",
            json={"duplicate_of": duplicate_of},
        )

    # =========================================================================
    # Security Debate
    # =========================================================================

    def list_security_debates(self, **kwargs: Any) -> dict[str, Any]:
        """
        List security audit debates.

        GET /api/v1/audit/security/debate

        Returns:
            Dict with security debate entries
        """
        return self._client.request("GET", "/api/v1/audit/security/debate", params=kwargs or None)



class AsyncAuditAPI:
    """
    Asynchronous Audit API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     entries = await client.audit.list_entries()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # =========================================================================
    # Entries, Report & Verification
    # =========================================================================

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

    # =========================================================================
    # Session Management
    # =========================================================================

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
        """Get details for a specific audit session.

        Args:
            session_id: The audit session identifier.

        Returns:
            Dict with session details including status, progress, and config.
        """
        return await self._client.request(
            "GET", f"/api/v1/audit/sessions/{session_id}"
        )

    async def delete_session(self, session_id: str) -> dict[str, Any]:
        """Delete an audit session.

        The session must not be in a running state. Pause or cancel it first.

        Args:
            session_id: The audit session identifier.

        Returns:
            Dict with deletion confirmation.
        """
        return await self._client.request(
            "DELETE", f"/api/v1/audit/sessions/{session_id}"
        )

    async def start_session(self, session_id: str) -> dict[str, Any]:
        """Start a pending or paused audit session.

        Args:
            session_id: The audit session identifier.

        Returns:
            Dict with updated session details.
        """
        return await self._client.request(
            "POST", f"/api/v1/audit/sessions/{session_id}/start"
        )

    async def pause_session(self, session_id: str) -> dict[str, Any]:
        """Pause a running audit session.

        Args:
            session_id: The audit session identifier.

        Returns:
            Dict with updated session details.
        """
        return await self._client.request(
            "POST", f"/api/v1/audit/sessions/{session_id}/pause"
        )

    async def resume_session(self, session_id: str) -> dict[str, Any]:
        """Resume a paused audit session.

        Args:
            session_id: The audit session identifier.

        Returns:
            Dict with updated session details.
        """
        return await self._client.request(
            "POST", f"/api/v1/audit/sessions/{session_id}/resume"
        )

    async def cancel_session(
        self, session_id: str, reason: str | None = None
    ) -> dict[str, Any]:
        """Cancel an audit session.

        Args:
            session_id: The audit session identifier.
            reason: Optional cancellation reason.

        Returns:
            Dict with updated session details.
        """
        data: dict[str, Any] = {}
        if reason:
            data["reason"] = reason
        return await self._client.request(
            "POST", f"/api/v1/audit/sessions/{session_id}/cancel", json=data
        )

    async def get_session_findings(
        self,
        session_id: str,
        severity: str | None = None,
        audit_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Get findings for an audit session.

        Args:
            session_id: The audit session identifier.
            severity: Filter by severity (critical, high, medium, low, info).
            audit_type: Filter by audit type.
            status: Filter by finding status.
            limit: Maximum findings to return.
            offset: Pagination offset.

        Returns:
            Dict with findings array and pagination info.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if severity:
            params["severity"] = severity
        if audit_type:
            params["audit_type"] = audit_type
        if status:
            params["status"] = status
        return await self._client.request(
            "GET", f"/api/v1/audit/sessions/{session_id}/findings", params=params
        )

    async def get_session_events(self, session_id: str) -> dict[str, Any]:
        """Get the event stream URL for an audit session.

        The actual event stream uses SSE (Server-Sent Events). This method
        returns the connection information for establishing the stream.

        Args:
            session_id: The audit session identifier.

        Returns:
            Dict with SSE event stream connection info.
        """
        return await self._client.request(
            "GET", f"/api/v1/audit/sessions/{session_id}/events"
        )

    async def intervene_session(
        self,
        session_id: str,
        action: str,
        finding_id: str | None = None,
        reason: str | None = None,
        context: str | None = None,
    ) -> dict[str, Any]:
        """Submit a human intervention for an audit session.

        Args:
            session_id: The audit session identifier.
            action: Intervention action (approve_finding, reject_finding,
                    add_context, override_decision).
            finding_id: Finding to act on (for finding-related actions).
            reason: Reason for the intervention.
            context: Additional context to inject.

        Returns:
            Dict with intervention confirmation.
        """
        data: dict[str, Any] = {"action": action}
        if finding_id:
            data["finding_id"] = finding_id
        if reason:
            data["reason"] = reason
        if context:
            data["context"] = context
        return await self._client.request(
            "POST", f"/api/v1/audit/sessions/{session_id}/intervene", json=data
        )

    async def export_session_report(
        self,
        session_id: str,
        format: str = "markdown",
        template: str | None = None,
        min_severity: str | None = None,
    ) -> dict[str, Any]:
        """Export an audit session report.

        Args:
            session_id: The audit session identifier.
            format: Report format (json, markdown, html, pdf).
            template: Report template (executive_summary, detailed_findings,
                      compliance_attestation, security_assessment).
            min_severity: Minimum severity to include (critical, high, medium,
                          low, info).

        Returns:
            Dict with report content and metadata.
        """
        params: dict[str, Any] = {"format": format}
        if template:
            params["template"] = template
        if min_severity:
            params["min_severity"] = min_severity
        return await self._client.request(
            "GET", f"/api/v1/audit/sessions/{session_id}/report", params=params
        )

    # =========================================================================
    # Finding Workflow Management
    # =========================================================================

    async def update_finding_status(
        self, finding_id: str, status: str, reason: str | None = None
    ) -> dict[str, Any]:
        """Update the workflow status of a finding.

        Args:
            finding_id: The finding identifier.
            status: New status value (e.g. open, in_progress, resolved, closed).
            reason: Optional reason for the status change.

        Returns:
            Dict with updated finding details.
        """
        data: dict[str, Any] = {"status": status}
        if reason:
            data["reason"] = reason
        return await self._client.request(
            "PATCH", f"/api/v1/audit/findings/{finding_id}/status", json=data
        )

    async def assign_finding(
        self, finding_id: str, assignee: str, note: str | None = None
    ) -> dict[str, Any]:
        """Assign a finding to a user.

        Args:
            finding_id: The finding identifier.
            assignee: User ID or email to assign to.
            note: Optional note for the assignee.

        Returns:
            Dict with assignment confirmation.
        """
        data: dict[str, Any] = {"assignee": assignee}
        if note:
            data["note"] = note
        return await self._client.request(
            "PATCH", f"/api/v1/audit/findings/{finding_id}/assign", json=data
        )

    async def unassign_finding(self, finding_id: str) -> dict[str, Any]:
        """Remove the assignment from a finding.

        Args:
            finding_id: The finding identifier.

        Returns:
            Dict with confirmation.
        """
        return await self._client.request(
            "POST", f"/api/v1/audit/findings/{finding_id}/unassign"
        )

    async def add_finding_comment(
        self, finding_id: str, text: str
    ) -> dict[str, Any]:
        """Add a comment to a finding.

        Args:
            finding_id: The finding identifier.
            text: Comment text.

        Returns:
            Dict with the created comment.
        """
        return await self._client.request(
            "POST",
            f"/api/v1/audit/findings/{finding_id}/comments",
            json={"text": text},
        )

    async def get_finding_history(self, finding_id: str) -> dict[str, Any]:
        """Get the workflow history of a finding.

        Args:
            finding_id: The finding identifier.

        Returns:
            Dict with history entries (status changes, assignments, comments).
        """
        return await self._client.request(
            "GET", f"/api/v1/audit/findings/{finding_id}/history"
        )

    async def set_finding_priority(
        self, finding_id: str, priority: str
    ) -> dict[str, Any]:
        """Set the priority of a finding.

        Args:
            finding_id: The finding identifier.
            priority: Priority level (e.g. critical, high, medium, low).

        Returns:
            Dict with updated finding details.
        """
        return await self._client.request(
            "PATCH",
            f"/api/v1/audit/findings/{finding_id}/priority",
            json={"priority": priority},
        )

    async def set_finding_due_date(
        self, finding_id: str, due_date: str
    ) -> dict[str, Any]:
        """Set the due date for a finding.

        Args:
            finding_id: The finding identifier.
            due_date: Due date in ISO 8601 format.

        Returns:
            Dict with updated finding details.
        """
        return await self._client.request(
            "PATCH",
            f"/api/v1/audit/findings/{finding_id}/due-date",
            json={"due_date": due_date},
        )

    async def link_finding(
        self, finding_id: str, target_finding_id: str, link_type: str = "related"
    ) -> dict[str, Any]:
        """Link a finding to another finding.

        Args:
            finding_id: The source finding identifier.
            target_finding_id: The target finding to link to.
            link_type: Relationship type (related, blocks, caused_by).

        Returns:
            Dict with link confirmation.
        """
        return await self._client.request(
            "POST",
            f"/api/v1/audit/findings/{finding_id}/link",
            json={"target_finding_id": target_finding_id, "link_type": link_type},
        )

    async def mark_finding_duplicate(
        self, finding_id: str, duplicate_of: str
    ) -> dict[str, Any]:
        """Mark a finding as a duplicate of another finding.

        Args:
            finding_id: The finding to mark as duplicate.
            duplicate_of: The canonical finding identifier.

        Returns:
            Dict with confirmation.
        """
        return await self._client.request(
            "POST",
            f"/api/v1/audit/findings/{finding_id}/duplicate",
            json={"duplicate_of": duplicate_of},
        )

    # =========================================================================
    # Security Debate
    # =========================================================================

    async def list_security_debates(self, **kwargs: Any) -> dict[str, Any]:
        """List security audit debates. GET /api/v1/audit/security/debate"""
        return await self._client.request("GET", "/api/v1/audit/security/debate", params=kwargs or None)

