"""
Autonomous Namespace API

Provides methods for autonomous agent operations:
- Manage autonomous approvals
- Configure triggers
- Monitor autonomous executions
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class AutonomousAPI:
    """Synchronous Autonomous API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def list_approvals(self, status: str | None = None, limit: int = 20) -> dict[str, Any]:
        """List autonomous approvals."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return self._client.request("GET", "/api/v1/autonomous/approvals", params=params)

    def get_approval(self, approval_id: str) -> dict[str, Any]:
        """Get approval by ID."""
        return self._client.request("GET", f"/api/v1/autonomous/approvals/{approval_id}")

    def approve(self, approval_id: str, notes: str | None = None) -> dict[str, Any]:
        """Approve an autonomous action."""
        data: dict[str, Any] = {}
        if notes:
            data["notes"] = notes
        return self._client.request(
            "POST", f"/api/v1/autonomous/approvals/{approval_id}/approve", json=data
        )

    def reject(self, approval_id: str, reason: str | None = None) -> dict[str, Any]:
        """Reject an autonomous action."""
        data: dict[str, Any] = {}
        if reason:
            data["reason"] = reason
        return self._client.request(
            "POST", f"/api/v1/autonomous/approvals/{approval_id}/reject", json=data
        )

    def list_triggers(self, limit: int = 20) -> dict[str, Any]:
        """List configured triggers."""
        return self._client.request("GET", "/api/v1/autonomous/triggers", params={"limit": limit})

    def create_trigger(
        self, name: str, condition: dict[str, Any], action: dict[str, Any]
    ) -> dict[str, Any]:
        """Create a new trigger."""
        return self._client.request(
            "POST",
            "/api/v1/autonomous/triggers",
            json={
                "name": name,
                "condition": condition,
                "action": action,
            },
        )

    def get_trigger(self, trigger_id: str) -> dict[str, Any]:
        """Get trigger by ID."""
        return self._client.request("GET", f"/api/v1/autonomous/triggers/{trigger_id}")

    def update_trigger(self, trigger_id: str, **kwargs: Any) -> dict[str, Any]:
        """Update a trigger."""
        return self._client.request(
            "PATCH", f"/api/v1/autonomous/triggers/{trigger_id}", json=kwargs
        )

    def delete_trigger(self, trigger_id: str) -> dict[str, Any]:
        """Delete a trigger."""
        return self._client.request("DELETE", f"/api/v1/autonomous/triggers/{trigger_id}")

    def get_stats(self) -> dict[str, Any]:
        """Get autonomous system statistics."""
        return self._client.request("GET", "/api/v1/autonomous/stats")


class AsyncAutonomousAPI:
    """Asynchronous Autonomous API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list_approvals(self, status: str | None = None, limit: int = 20) -> dict[str, Any]:
        """List autonomous approvals."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        return await self._client.request("GET", "/api/v1/autonomous/approvals", params=params)

    async def get_approval(self, approval_id: str) -> dict[str, Any]:
        """Get approval by ID."""
        return await self._client.request("GET", f"/api/v1/autonomous/approvals/{approval_id}")

    async def approve(self, approval_id: str, notes: str | None = None) -> dict[str, Any]:
        """Approve an autonomous action."""
        data: dict[str, Any] = {}
        if notes:
            data["notes"] = notes
        return await self._client.request(
            "POST", f"/api/v1/autonomous/approvals/{approval_id}/approve", json=data
        )

    async def reject(self, approval_id: str, reason: str | None = None) -> dict[str, Any]:
        """Reject an autonomous action."""
        data: dict[str, Any] = {}
        if reason:
            data["reason"] = reason
        return await self._client.request(
            "POST", f"/api/v1/autonomous/approvals/{approval_id}/reject", json=data
        )

    async def list_triggers(self, limit: int = 20) -> dict[str, Any]:
        """List configured triggers."""
        return await self._client.request(
            "GET", "/api/v1/autonomous/triggers", params={"limit": limit}
        )

    async def create_trigger(
        self, name: str, condition: dict[str, Any], action: dict[str, Any]
    ) -> dict[str, Any]:
        """Create a new trigger."""
        return await self._client.request(
            "POST",
            "/api/v1/autonomous/triggers",
            json={
                "name": name,
                "condition": condition,
                "action": action,
            },
        )

    async def get_trigger(self, trigger_id: str) -> dict[str, Any]:
        """Get trigger by ID."""
        return await self._client.request("GET", f"/api/v1/autonomous/triggers/{trigger_id}")

    async def update_trigger(self, trigger_id: str, **kwargs: Any) -> dict[str, Any]:
        """Update a trigger."""
        return await self._client.request(
            "PATCH", f"/api/v1/autonomous/triggers/{trigger_id}", json=kwargs
        )

    async def delete_trigger(self, trigger_id: str) -> dict[str, Any]:
        """Delete a trigger."""
        return await self._client.request("DELETE", f"/api/v1/autonomous/triggers/{trigger_id}")

    async def get_stats(self) -> dict[str, Any]:
        """Get autonomous system statistics."""
        return await self._client.request("GET", "/api/v1/autonomous/stats")
