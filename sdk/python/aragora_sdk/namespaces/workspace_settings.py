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

    def list_workspaces(self) -> dict[str, Any]:
        """List all workspaces."""
        return self._client.request("GET", "/api/v1/workspaces")

    def create_workspace(self, **kwargs: Any) -> dict[str, Any]:
        """Create a new workspace."""
        return self._client.request("POST", "/api/v1/workspaces", json=kwargs)

    def get_profiles(self) -> dict[str, Any]:
        """Get workspace profiles."""
        return self._client.request("GET", "/api/v1/workspaces/profiles")

    def get_workspace(self, workspace_id: str) -> dict[str, Any]:
        """Get a workspace by ID."""
        return self._client.request("GET", f"/api/v1/workspaces/{workspace_id}")

    def delete_workspace(self, workspace_id: str) -> dict[str, Any]:
        """Delete a workspace."""
        return self._client.request("DELETE", f"/api/v1/workspaces/{workspace_id}")

    def list_invites(self, workspace_id: str) -> dict[str, Any]:
        """List invites for a workspace."""
        return self._client.request("GET", f"/api/v1/workspaces/{workspace_id}/invites")

    def create_invite(self, workspace_id: str, **kwargs: Any) -> dict[str, Any]:
        """Create an invite for a workspace."""
        return self._client.request(
            "POST", f"/api/v1/workspaces/{workspace_id}/invites", json=kwargs
        )

    def delete_invite(self, workspace_id: str, invite_id: str) -> dict[str, Any]:
        """Delete an invite."""
        return self._client.request(
            "DELETE", f"/api/v1/workspaces/{workspace_id}/invites/{invite_id}"
        )

    def resend_invite(self, workspace_id: str, invite_id: str) -> dict[str, Any]:
        """Resend an invite."""
        return self._client.request(
            "POST", f"/api/v1/workspaces/{workspace_id}/invites/{invite_id}/resend"
        )

    def list_members(self, workspace_id: str) -> dict[str, Any]:
        """List members of a workspace."""
        return self._client.request("GET", f"/api/v1/workspaces/{workspace_id}/members")

    def add_member(self, workspace_id: str, **kwargs: Any) -> dict[str, Any]:
        """Add a member to a workspace."""
        return self._client.request(
            "POST", f"/api/v1/workspaces/{workspace_id}/members", json=kwargs
        )

    def remove_member(self, workspace_id: str, user_id: str) -> dict[str, Any]:
        """Remove a member from a workspace."""
        return self._client.request(
            "DELETE", f"/api/v1/workspaces/{workspace_id}/members/{user_id}"
        )

    def update_member_role(self, workspace_id: str, user_id: str, role: str) -> dict[str, Any]:
        """Update a member's role."""
        return self._client.request(
            "PUT", f"/api/v1/workspaces/{workspace_id}/members/{user_id}/role", json={"role": role}
        )

    def list_roles(self, workspace_id: str) -> dict[str, Any]:
        """List available roles for a workspace."""
        return self._client.request("GET", f"/api/v1/workspaces/{workspace_id}/roles")


class AsyncWorkspaceSettingsAPI:
    """Asynchronous Workspace Settings API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list_workspaces(self) -> dict[str, Any]:
        """List all workspaces."""
        return await self._client.request("GET", "/api/v1/workspaces")

    async def create_workspace(self, **kwargs: Any) -> dict[str, Any]:
        """Create a new workspace."""
        return await self._client.request("POST", "/api/v1/workspaces", json=kwargs)

    async def get_profiles(self) -> dict[str, Any]:
        """Get workspace profiles."""
        return await self._client.request("GET", "/api/v1/workspaces/profiles")

    async def get_workspace(self, workspace_id: str) -> dict[str, Any]:
        """Get a workspace by ID."""
        return await self._client.request("GET", f"/api/v1/workspaces/{workspace_id}")

    async def delete_workspace(self, workspace_id: str) -> dict[str, Any]:
        """Delete a workspace."""
        return await self._client.request("DELETE", f"/api/v1/workspaces/{workspace_id}")

    async def list_invites(self, workspace_id: str) -> dict[str, Any]:
        """List invites for a workspace."""
        return await self._client.request("GET", f"/api/v1/workspaces/{workspace_id}/invites")

    async def create_invite(self, workspace_id: str, **kwargs: Any) -> dict[str, Any]:
        """Create an invite for a workspace."""
        return await self._client.request(
            "POST", f"/api/v1/workspaces/{workspace_id}/invites", json=kwargs
        )

    async def delete_invite(self, workspace_id: str, invite_id: str) -> dict[str, Any]:
        """Delete an invite."""
        return await self._client.request(
            "DELETE", f"/api/v1/workspaces/{workspace_id}/invites/{invite_id}"
        )

    async def resend_invite(self, workspace_id: str, invite_id: str) -> dict[str, Any]:
        """Resend an invite."""
        return await self._client.request(
            "POST", f"/api/v1/workspaces/{workspace_id}/invites/{invite_id}/resend"
        )

    async def list_members(self, workspace_id: str) -> dict[str, Any]:
        """List members of a workspace."""
        return await self._client.request("GET", f"/api/v1/workspaces/{workspace_id}/members")

    async def add_member(self, workspace_id: str, **kwargs: Any) -> dict[str, Any]:
        """Add a member to a workspace."""
        return await self._client.request(
            "POST", f"/api/v1/workspaces/{workspace_id}/members", json=kwargs
        )

    async def remove_member(self, workspace_id: str, user_id: str) -> dict[str, Any]:
        """Remove a member from a workspace."""
        return await self._client.request(
            "DELETE", f"/api/v1/workspaces/{workspace_id}/members/{user_id}"
        )

    async def update_member_role(
        self, workspace_id: str, user_id: str, role: str
    ) -> dict[str, Any]:
        """Update a member's role."""
        return await self._client.request(
            "PUT", f"/api/v1/workspaces/{workspace_id}/members/{user_id}/role", json={"role": role}
        )

    async def list_roles(self, workspace_id: str) -> dict[str, Any]:
        """List available roles for a workspace."""
        return await self._client.request("GET", f"/api/v1/workspaces/{workspace_id}/roles")
