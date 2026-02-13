"""
Audit Namespace API

Provides methods for audit trail management:
- List and query audit entries
- Session lifecycle management
- Finding management (assign, status, priority, comments)
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

    # =========================================================================
    # Finding Management
    # =========================================================================

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

    # =========================================================================
    # Finding Management
    # =========================================================================

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
