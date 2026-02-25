"""
Audit Trail Namespace API

Provides methods for audit trail access and verification:
- List and retrieve audit entries
- Export in multiple formats (json, csv, md)
- Verify integrity checksums
- Access v1 decision receipts
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

AuditExportFormat = Literal["json", "csv", "md"]


class AuditTrailAPI:
    """
    Synchronous Audit Trail API.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> entries = client.audit_trail.list_entries()
        >>> report = client.audit_trail.get_report()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # -- Audit trails ----------------------------------------------------------

    def list_trails(
        self,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """List recent audit trails.

        Args:
            limit: Maximum number of trails to return
            offset: Number of trails to skip

        Returns:
            Dict with audit trails and pagination info
        """
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return self._client.request("GET", "/api/v1/audit-trails", params=params or None)

    # -- Audit entries ---------------------------------------------------------

    def list_entries(self) -> dict[str, Any]:
        """List audit entries."""
        return self._client.request("GET", "/api/v1/audit/entries")

    def get_report(self) -> dict[str, Any]:
        """Get audit report."""
        return self._client.request("GET", "/api/v1/audit/report")

    def verify(self) -> dict[str, Any]:
        """Verify audit trail integrity."""
        return self._client.request("GET", "/api/v1/audit/verify")

    def list_types(self) -> dict[str, Any]:
        """List audit event types."""
        return self._client.request("GET", "/api/v1/audit/types")

    def list_presets(self) -> dict[str, Any]:
        """List audit query presets."""
        return self._client.request("GET", "/api/v1/audit/presets")

    def get_workflow_states(self) -> dict[str, Any]:
        """Get audit workflow states."""
        return self._client.request("GET", "/api/v1/audit/workflow/states")

    # -- Sessions --------------------------------------------------------------

    def list_sessions(self) -> dict[str, Any]:
        """List audit sessions."""
        return self._client.request("GET", "/api/v1/audit/sessions")

    def create_session(self, **kwargs: Any) -> dict[str, Any]:
        """Create an audit session."""
        return self._client.request("POST", "/api/v1/audit/sessions", json=kwargs)

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Get an audit session by ID."""
        return self._client.request("GET", f"/api/v1/audit/sessions/{session_id}")

    def delete_session(self, session_id: str) -> dict[str, Any]:
        """Delete an audit session."""
        return self._client.request("DELETE", f"/api/v1/audit/sessions/{session_id}")

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

    def intervene_session(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        """Intervene in an audit session."""
        return self._client.request(
            "POST", f"/api/v1/audit/sessions/{session_id}/intervene", json=kwargs
        )

    def get_session_events(self, session_id: str) -> dict[str, Any]:
        """Get events for an audit session."""
        return self._client.request("GET", f"/api/v1/audit/sessions/{session_id}/events")

    def get_session_findings(self, session_id: str) -> dict[str, Any]:
        """Get findings for an audit session."""
        return self._client.request("GET", f"/api/v1/audit/sessions/{session_id}/findings")

    def get_session_report(self, session_id: str) -> dict[str, Any]:
        """Get report for an audit session."""
        return self._client.request("GET", f"/api/v1/audit/sessions/{session_id}/report")

    # -- Security debates ------------------------------------------------------

    def create_security_debate(self, **kwargs: Any) -> dict[str, Any]:
        """Create a security audit debate."""
        return self._client.request("POST", "/api/v1/audit/security/debate", json=kwargs)

    def get_security_debate(self, debate_id: str) -> dict[str, Any]:
        """Get a security audit debate."""
        return self._client.request("GET", f"/api/v1/audit/security/debate/{debate_id}")


class AsyncAuditTrailAPI:
    """
    Asynchronous Audit Trail API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     entries = await client.audit_trail.list_entries()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list_trails(
        self,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """List recent audit trails.

        Args:
            limit: Maximum number of trails to return
            offset: Number of trails to skip

        Returns:
            Dict with audit trails and pagination info
        """
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return await self._client.request("GET", "/api/v1/audit-trails", params=params or None)

    async def list_entries(self) -> dict[str, Any]:
        """List audit entries."""
        return await self._client.request("GET", "/api/v1/audit/entries")

    async def get_report(self) -> dict[str, Any]:
        """Get audit report."""
        return await self._client.request("GET", "/api/v1/audit/report")

    async def verify(self) -> dict[str, Any]:
        """Verify audit trail integrity."""
        return await self._client.request("GET", "/api/v1/audit/verify")

    async def list_types(self) -> dict[str, Any]:
        """List audit event types."""
        return await self._client.request("GET", "/api/v1/audit/types")

    async def list_presets(self) -> dict[str, Any]:
        """List audit query presets."""
        return await self._client.request("GET", "/api/v1/audit/presets")

    async def get_workflow_states(self) -> dict[str, Any]:
        """Get audit workflow states."""
        return await self._client.request("GET", "/api/v1/audit/workflow/states")

    async def list_sessions(self) -> dict[str, Any]:
        """List audit sessions."""
        return await self._client.request("GET", "/api/v1/audit/sessions")

    async def create_session(self, **kwargs: Any) -> dict[str, Any]:
        """Create an audit session."""
        return await self._client.request("POST", "/api/v1/audit/sessions", json=kwargs)

    async def get_session(self, session_id: str) -> dict[str, Any]:
        """Get an audit session by ID."""
        return await self._client.request("GET", f"/api/v1/audit/sessions/{session_id}")

    async def delete_session(self, session_id: str) -> dict[str, Any]:
        """Delete an audit session."""
        return await self._client.request("DELETE", f"/api/v1/audit/sessions/{session_id}")

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

    async def intervene_session(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        """Intervene in an audit session."""
        return await self._client.request(
            "POST", f"/api/v1/audit/sessions/{session_id}/intervene", json=kwargs
        )

    async def get_session_events(self, session_id: str) -> dict[str, Any]:
        """Get events for an audit session."""
        return await self._client.request("GET", f"/api/v1/audit/sessions/{session_id}/events")

    async def get_session_findings(self, session_id: str) -> dict[str, Any]:
        """Get findings for an audit session."""
        return await self._client.request("GET", f"/api/v1/audit/sessions/{session_id}/findings")

    async def get_session_report(self, session_id: str) -> dict[str, Any]:
        """Get report for an audit session."""
        return await self._client.request("GET", f"/api/v1/audit/sessions/{session_id}/report")

    async def create_security_debate(self, **kwargs: Any) -> dict[str, Any]:
        """Create a security audit debate."""
        return await self._client.request("POST", "/api/v1/audit/security/debate", json=kwargs)

    async def get_security_debate(self, debate_id: str) -> dict[str, Any]:
        """Get a security audit debate."""
        return await self._client.request("GET", f"/api/v1/audit/security/debate/{debate_id}")
