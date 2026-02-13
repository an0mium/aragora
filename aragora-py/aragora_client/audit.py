"""Audit API for the Aragora SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from aragora_client.client import AragoraClient


class AuditEvent(BaseModel):
    """Audit event model."""

    id: str
    timestamp: str
    actor_id: str | None = None
    actor_type: str | None = None
    action: str
    resource_type: str
    resource_id: str | None = None
    details: dict[str, Any] = {}
    ip_address: str | None = None
    user_agent: str | None = None
    tenant_id: str | None = None


class AuditStats(BaseModel):
    """Audit statistics."""

    total_events: int
    events_by_action: dict[str, int] = {}
    events_by_resource_type: dict[str, int] = {}
    top_actors: list[dict[str, Any]] = []
    period_start: str | None = None
    period_end: str | None = None


class AuditExportRequest(BaseModel):
    """Audit export request."""

    start_date: str
    end_date: str
    format: Literal["json", "csv", "pdf"] = "json"
    filters: dict[str, str] | None = None


class AuditExportResponse(BaseModel):
    """Audit export response."""

    export_id: str
    download_url: str | None = None
    status: str = "pending"


class AuditIntegrityResult(BaseModel):
    """Audit integrity verification result."""

    verified: bool
    entries_checked: int
    tampered_entries: int
    details: list[dict[str, Any]] | None = None


class AuditRetentionPolicy(BaseModel):
    """Audit retention policy."""

    id: str
    name: str
    retention_days: int
    resource_types: list[str] = []
    actions: list[str] = []
    is_active: bool = True


class CreateRetentionPolicyRequest(BaseModel):
    """Create retention policy request."""

    name: str
    retention_days: int
    resource_types: list[str] | None = None
    actions: list[str] | None = None


class AuditAPI:
    """API for audit operations."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    async def list_events(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        actor_id: str | None = None,
        resource_type: str | None = None,
        action: str | None = None,
        tenant_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[AuditEvent], int]:
        """List audit events."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if actor_id:
            params["actor_id"] = actor_id
        if resource_type:
            params["resource_type"] = resource_type
        if action:
            params["action"] = action
        if tenant_id:
            params["tenant_id"] = tenant_id
        data = await self._client._get("/api/v1/audit/events", params=params)
        events = [AuditEvent.model_validate(e) for e in data.get("events", [])]
        return events, data.get("total", len(events))

    async def get_event(self, event_id: str) -> AuditEvent:
        """Get a specific audit event."""
        data = await self._client._get(f"/api/v1/audit/events/{event_id}")
        return AuditEvent.model_validate(data)

    async def get_stats(
        self,
        *,
        period: str | None = None,
        tenant_id: str | None = None,
    ) -> AuditStats:
        """Get audit statistics."""
        params: dict[str, Any] = {}
        if period:
            params["period"] = period
        if tenant_id:
            params["tenant_id"] = tenant_id
        data = await self._client._get("/api/v1/audit/stats", params=params)
        return AuditStats.model_validate(data)

    async def export(
        self,
        start_date: str,
        end_date: str,
        *,
        format: Literal["json", "csv", "pdf"] = "json",
        filters: dict[str, str] | None = None,
    ) -> AuditExportResponse:
        """Export audit logs."""
        request = AuditExportRequest(
            start_date=start_date,
            end_date=end_date,
            format=format,
            filters=filters,
        )
        data = await self._client._post("/api/v1/audit/export", request.model_dump())
        return AuditExportResponse.model_validate(data)

    async def get_export_status(self, export_id: str) -> AuditExportResponse:
        """Get export status."""
        data = await self._client._get(f"/api/v1/audit/export/{export_id}")
        return AuditExportResponse.model_validate(data)

    async def verify_integrity(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> AuditIntegrityResult:
        """Verify audit log integrity."""
        params: dict[str, Any] = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        data = await self._client._get("/api/v1/audit/verify", params=params)
        return AuditIntegrityResult.model_validate(data)

    async def search(
        self,
        query: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AuditEvent]:
        """Search audit events."""
        params: dict[str, Any] = {"q": query, "limit": limit, "offset": offset}
        data = await self._client._get("/api/v1/audit/search", params=params)
        return [AuditEvent.model_validate(e) for e in data.get("events", [])]

    # Retention policies
    async def list_retention_policies(self) -> list[AuditRetentionPolicy]:
        """List audit retention policies."""
        data = await self._client._get("/api/v1/audit/retention")
        return [
            AuditRetentionPolicy.model_validate(p) for p in data.get("policies", [])
        ]

    async def create_retention_policy(
        self,
        name: str,
        retention_days: int,
        *,
        resource_types: list[str] | None = None,
        actions: list[str] | None = None,
    ) -> AuditRetentionPolicy:
        """Create a retention policy."""
        request = CreateRetentionPolicyRequest(
            name=name,
            retention_days=retention_days,
            resource_types=resource_types,
            actions=actions,
        )
        data = await self._client._post("/api/v1/audit/retention", request.model_dump())
        return AuditRetentionPolicy.model_validate(data)

    async def delete_retention_policy(self, policy_id: str) -> None:
        """Delete a retention policy."""
        await self._client._delete(f"/api/v1/audit/retention/{policy_id}")

    async def get_actor_history(
        self,
        actor_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AuditEvent]:
        """Get activity history for an actor."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        data = await self._client._get(
            f"/api/v1/audit/actors/{actor_id}/history", params=params
        )
        return [AuditEvent.model_validate(e) for e in data.get("events", [])]

    async def get_resource_history(
        self,
        resource_type: str,
        resource_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AuditEvent]:
        """Get activity history for a resource."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        data = await self._client._get(
            f"/api/v1/audit/resources/{resource_type}/{resource_id}/history",
            params=params,
        )
        return [AuditEvent.model_validate(e) for e in data.get("events", [])]
