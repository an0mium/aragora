"""
Workspace Settings Namespace API

Provides methods for workspace configuration:
- Workspace preferences
- Member settings
- Integration settings
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class WorkspaceSettingsAPI:
    """Synchronous Workspace Settings API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def get(self, workspace_id: str) -> dict[str, Any]:
        """Get workspace settings."""
        return self._client.request("GET", f"/api/v1/workspaces/{workspace_id}/settings")

    def update(self, workspace_id: str, **settings: Any) -> dict[str, Any]:
        """Update workspace settings."""
        return self._client.request(
            "PATCH", f"/api/v1/workspaces/{workspace_id}/settings", json=settings
        )

    def get_integrations(self, workspace_id: str) -> dict[str, Any]:
        """Get integration settings."""
        return self._client.request(
            "GET", f"/api/v1/workspaces/{workspace_id}/settings/integrations"
        )

    def update_integration(
        self, workspace_id: str, integration: str, **settings: Any
    ) -> dict[str, Any]:
        """Update integration settings."""
        return self._client.request(
            "PATCH",
            f"/api/v1/workspaces/{workspace_id}/settings/integrations/{integration}",
            json=settings,
        )

    def get_notifications(self, workspace_id: str) -> dict[str, Any]:
        """Get notification settings."""
        return self._client.request(
            "GET", f"/api/v1/workspaces/{workspace_id}/settings/notifications"
        )

    def update_notifications(self, workspace_id: str, **settings: Any) -> dict[str, Any]:
        """Update notification settings."""
        return self._client.request(
            "PATCH", f"/api/v1/workspaces/{workspace_id}/settings/notifications", json=settings
        )

    def get_security(self, workspace_id: str) -> dict[str, Any]:
        """Get security settings."""
        return self._client.request("GET", f"/api/v1/workspaces/{workspace_id}/settings/security")

    def update_security(self, workspace_id: str, **settings: Any) -> dict[str, Any]:
        """Update security settings."""
        return self._client.request(
            "PATCH", f"/api/v1/workspaces/{workspace_id}/settings/security", json=settings
        )


class AsyncWorkspaceSettingsAPI:
    """Asynchronous Workspace Settings API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get(self, workspace_id: str) -> dict[str, Any]:
        """Get workspace settings."""
        return await self._client.request("GET", f"/api/v1/workspaces/{workspace_id}/settings")

    async def update(self, workspace_id: str, **settings: Any) -> dict[str, Any]:
        """Update workspace settings."""
        return await self._client.request(
            "PATCH", f"/api/v1/workspaces/{workspace_id}/settings", json=settings
        )

    async def get_integrations(self, workspace_id: str) -> dict[str, Any]:
        """Get integration settings."""
        return await self._client.request(
            "GET", f"/api/v1/workspaces/{workspace_id}/settings/integrations"
        )

    async def update_integration(
        self, workspace_id: str, integration: str, **settings: Any
    ) -> dict[str, Any]:
        """Update integration settings."""
        return await self._client.request(
            "PATCH",
            f"/api/v1/workspaces/{workspace_id}/settings/integrations/{integration}",
            json=settings,
        )

    async def get_notifications(self, workspace_id: str) -> dict[str, Any]:
        """Get notification settings."""
        return await self._client.request(
            "GET", f"/api/v1/workspaces/{workspace_id}/settings/notifications"
        )

    async def update_notifications(self, workspace_id: str, **settings: Any) -> dict[str, Any]:
        """Update notification settings."""
        return await self._client.request(
            "PATCH", f"/api/v1/workspaces/{workspace_id}/settings/notifications", json=settings
        )

    async def get_security(self, workspace_id: str) -> dict[str, Any]:
        """Get security settings."""
        return await self._client.request(
            "GET", f"/api/v1/workspaces/{workspace_id}/settings/security"
        )

    async def update_security(self, workspace_id: str, **settings: Any) -> dict[str, Any]:
        """Update security settings."""
        return await self._client.request(
            "PATCH", f"/api/v1/workspaces/{workspace_id}/settings/security", json=settings
        )
