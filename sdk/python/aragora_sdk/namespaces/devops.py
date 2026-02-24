"""
DevOps Namespace API

Provides methods for DevOps integration:
- CI/CD pipelines
- Deployment management
- Infrastructure monitoring
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class DevopsAPI:
    """
    Synchronous DevOps API.

    Provides methods for incident management via PagerDuty:
    - Create and manage incidents
    - Query on-call schedules
    - Service health status

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> incidents = client.devops.list_incidents()
        >>> oncall = client.devops.get_oncall()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_status(self) -> dict[str, Any]:
        """Get PagerDuty connection status."""
        return self._client.request("GET", "/api/v1/devops/status")

    def list_incidents(
        self,
        status: str | None = None,
        urgency: str | None = None,
        service_ids: str | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List incidents with optional filtering.

        Args:
            status: Filter by status (triggered, acknowledged, resolved)
            urgency: Filter by urgency (high, low)
            service_ids: Comma-separated service IDs
            limit: Maximum results (default 25)
            offset: Pagination offset

        Returns:
            List of incidents
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if urgency:
            params["urgency"] = urgency
        if service_ids:
            params["service_ids"] = service_ids
        return self._client.request("GET", "/api/v1/incidents", params=params)

    def get_incident(self, incident_id: str) -> dict[str, Any]:
        """Get incident details."""
        return self._client.request("GET", f"/api/v1/incidents/{incident_id}")

    def create_incident(
        self,
        title: str,
        service_id: str,
        urgency: str = "high",
        description: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a new incident.

        Args:
            title: Incident title
            service_id: PagerDuty service ID
            urgency: Urgency level (high, low)
            description: Optional description

        Returns:
            Created incident details
        """
        data: dict[str, Any] = {
            "title": title,
            "service_id": service_id,
            "urgency": urgency,
        }
        if description:
            data["description"] = description
        return self._client.request("POST", "/api/v1/incidents", json=data)

    def acknowledge_incident(self, incident_id: str) -> dict[str, Any]:
        """Acknowledge an incident."""
        return self._client.request("POST", f"/api/v1/incidents/{incident_id}/acknowledge")

    def resolve_incident(self, incident_id: str, resolution: str | None = None) -> dict[str, Any]:
        """Resolve an incident."""
        data: dict[str, Any] = {}
        if resolution:
            data["resolution"] = resolution
        return self._client.request(
            "POST", f"/api/v1/incidents/{incident_id}/resolve", json=data or None
        )

    def reassign_incident(self, incident_id: str, user_ids: list[str]) -> dict[str, Any]:
        """Reassign an incident to different users."""
        return self._client.request(
            "POST",
            f"/api/v1/incidents/{incident_id}/reassign",
            json={"user_ids": user_ids},
        )

    def add_note(self, incident_id: str, content: str) -> dict[str, Any]:
        """Add a note to an incident."""
        return self._client.request(
            "POST", f"/api/v1/incidents/{incident_id}/notes", json={"content": content}
        )

    def list_notes(self, incident_id: str) -> dict[str, Any]:
        """List notes for an incident."""
        return self._client.request("GET", f"/api/v1/incidents/{incident_id}/notes")

    def get_oncall(self) -> dict[str, Any]:
        """Get current on-call schedule."""
        return self._client.request("GET", "/api/v1/oncall")

    def get_oncall_for_service(self, service_id: str) -> dict[str, Any]:
        """
        Get on-call schedule for a specific service.

        Args:
            service_id: PagerDuty service ID

        Returns:
            On-call information for the service
        """
        return self._client.request("GET", f"/api/v1/oncall/services/{service_id}")

    def list_services(self) -> dict[str, Any]:
        """List PagerDuty services."""
        return self._client.request("GET", "/api/v1/services")

    def get_service(self, service_id: str) -> dict[str, Any]:
        """Get PagerDuty service details."""
        return self._client.request("GET", f"/api/v1/services/{service_id}")


class AsyncDevopsAPI:
    """Asynchronous DevOps API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_status(self) -> dict[str, Any]:
        """Get PagerDuty connection status."""
        return await self._client.request("GET", "/api/v1/devops/status")

    async def list_incidents(
        self,
        status: str | None = None,
        urgency: str | None = None,
        service_ids: str | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List incidents with optional filtering."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if urgency:
            params["urgency"] = urgency
        if service_ids:
            params["service_ids"] = service_ids
        return await self._client.request("GET", "/api/v1/incidents", params=params)

    async def get_incident(self, incident_id: str) -> dict[str, Any]:
        """Get incident details."""
        return await self._client.request("GET", f"/api/v1/incidents/{incident_id}")

    async def create_incident(
        self,
        title: str,
        service_id: str,
        urgency: str = "high",
        description: str | None = None,
    ) -> dict[str, Any]:
        """Create a new incident."""
        data: dict[str, Any] = {
            "title": title,
            "service_id": service_id,
            "urgency": urgency,
        }
        if description:
            data["description"] = description
        return await self._client.request("POST", "/api/v1/incidents", json=data)

    async def acknowledge_incident(self, incident_id: str) -> dict[str, Any]:
        """Acknowledge an incident."""
        return await self._client.request("POST", f"/api/v1/incidents/{incident_id}/acknowledge")

    async def resolve_incident(
        self, incident_id: str, resolution: str | None = None
    ) -> dict[str, Any]:
        """Resolve an incident."""
        data: dict[str, Any] = {}
        if resolution:
            data["resolution"] = resolution
        return await self._client.request(
            "POST", f"/api/v1/incidents/{incident_id}/resolve", json=data or None
        )

    async def reassign_incident(self, incident_id: str, user_ids: list[str]) -> dict[str, Any]:
        """Reassign an incident to different users."""
        return await self._client.request(
            "POST",
            f"/api/v1/incidents/{incident_id}/reassign",
            json={"user_ids": user_ids},
        )

    async def add_note(self, incident_id: str, content: str) -> dict[str, Any]:
        """Add a note to an incident."""
        return await self._client.request(
            "POST", f"/api/v1/incidents/{incident_id}/notes", json={"content": content}
        )

    async def list_notes(self, incident_id: str) -> dict[str, Any]:
        """List notes for an incident."""
        return await self._client.request("GET", f"/api/v1/incidents/{incident_id}/notes")

    async def get_oncall(self) -> dict[str, Any]:
        """Get current on-call schedule."""
        return await self._client.request("GET", "/api/v1/oncall")

    async def get_oncall_for_service(self, service_id: str) -> dict[str, Any]:
        """Get on-call schedule for a specific service."""
        return await self._client.request("GET", f"/api/v1/oncall/services/{service_id}")

    async def list_services(self) -> dict[str, Any]:
        """List PagerDuty services."""
        return await self._client.request("GET", "/api/v1/services")

    async def get_service(self, service_id: str) -> dict[str, Any]:
        """Get PagerDuty service details."""
        return await self._client.request("GET", f"/api/v1/services/{service_id}")
