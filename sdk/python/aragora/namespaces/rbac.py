"""
RBAC Namespace API

Provides methods for Role-Based Access Control:
- Permission management
- Role management
- Role assignments
- Permission checking
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class RBACAPI:
    """
    Synchronous RBAC API.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> roles = client.rbac.list_roles()
        >>> can_access = client.rbac.check_permission("user_123", "debates:create")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def list_permissions(
        self,
        resource_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List available permissions.

        Args:
            resource_type: Filter by resource type
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of permissions with descriptions
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if resource_type:
            params["resource_type"] = resource_type

        return self._client.request("GET", "/api/v1/rbac/permissions", params=params)

    def get_permission(self, permission_key: str) -> dict[str, Any]:
        """
        Get a permission by key.

        Args:
            permission_key: Permission key (e.g., "debates:create")

        Returns:
            Permission details
        """
        return self._client.request("GET", f"/api/v1/rbac/permissions/{permission_key}")

    def list_roles(
        self,
        include_system: bool = True,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List available roles.

        Args:
            include_system: Include system roles (admin, viewer, etc.)
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of roles with permissions
        """
        params: dict[str, Any] = {
            "include_system": include_system,
            "limit": limit,
            "offset": offset,
        }
        return self._client.request("GET", "/api/v1/rbac/roles", params=params)

    def get_role(self, role_id: str) -> dict[str, Any]:
        """
        Get a role by ID.

        Args:
            role_id: Role ID

        Returns:
            Role details with permissions
        """
        return self._client.request("GET", f"/api/v1/rbac/roles/{role_id}")

    def create_role(
        self,
        name: str,
        permissions: list[str],
        description: str | None = None,
        parent_role: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a custom role.

        Args:
            name: Role name
            permissions: List of permission keys
            description: Role description
            parent_role: Parent role to inherit from

        Returns:
            Created role
        """
        data: dict[str, Any] = {"name": name, "permissions": permissions}
        if description:
            data["description"] = description
        if parent_role:
            data["parent_role"] = parent_role

        return self._client.request("POST", "/api/v1/rbac/roles", json=data)

    def update_role(
        self,
        role_id: str,
        permissions: list[str] | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """
        Update a role.

        Args:
            role_id: Role ID
            permissions: New permissions list
            description: New description

        Returns:
            Updated role
        """
        data: dict[str, Any] = {}
        if permissions is not None:
            data["permissions"] = permissions
        if description is not None:
            data["description"] = description

        return self._client.request("PUT", f"/api/v1/rbac/roles/{role_id}", json=data)

    def delete_role(self, role_id: str) -> dict[str, Any]:
        """
        Delete a custom role.

        Args:
            role_id: Role ID

        Returns:
            Deletion result
        """
        return self._client.request("DELETE", f"/api/v1/rbac/roles/{role_id}")

    def assign_role(
        self,
        user_id: str,
        role_id: str,
        scope: str | None = None,
    ) -> dict[str, Any]:
        """
        Assign a role to a user.

        Args:
            user_id: User ID
            role_id: Role ID to assign
            scope: Optional scope (org, workspace, etc.)

        Returns:
            Role assignment
        """
        data: dict[str, Any] = {"user_id": user_id, "role_id": role_id}
        if scope:
            data["scope"] = scope

        return self._client.request("POST", "/api/v1/rbac/assignments", json=data)

    def revoke_role(self, user_id: str, role_id: str) -> dict[str, Any]:
        """
        Revoke a role from a user.

        Args:
            user_id: User ID
            role_id: Role ID to revoke

        Returns:
            Revocation result
        """
        return self._client.request(
            "DELETE",
            "/api/v1/rbac/assignments",
            params={"user_id": user_id, "role_id": role_id},
        )

    def get_user_roles(self, user_id: str) -> dict[str, Any]:
        """
        Get roles assigned to a user.

        Args:
            user_id: User ID

        Returns:
            User's role assignments
        """
        return self._client.request("GET", f"/api/v1/rbac/users/{user_id}/roles")

    def check_permission(
        self,
        user_id: str,
        permission: str,
        resource_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Check if a user has a specific permission.

        Args:
            user_id: User ID
            permission: Permission key to check
            resource_id: Optional specific resource

        Returns:
            Permission check result with allowed status
        """
        data: dict[str, Any] = {"user_id": user_id, "permission": permission}
        if resource_id:
            data["resource_id"] = resource_id

        return self._client.request("POST", "/api/v1/rbac/check", json=data)

    def get_effective_permissions(self, user_id: str) -> dict[str, Any]:
        """
        Get all effective permissions for a user.

        Args:
            user_id: User ID

        Returns:
            List of all permissions the user has
        """
        return self._client.request("GET", f"/api/v1/rbac/users/{user_id}/permissions")


class AsyncRBACAPI:
    """
    Asynchronous RBAC API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     roles = await client.rbac.list_roles()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list_permissions(
        self,
        resource_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List available permissions."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if resource_type:
            params["resource_type"] = resource_type

        return await self._client.request("GET", "/api/v1/rbac/permissions", params=params)

    async def get_permission(self, permission_key: str) -> dict[str, Any]:
        """Get a permission by key."""
        return await self._client.request("GET", f"/api/v1/rbac/permissions/{permission_key}")

    async def list_roles(
        self,
        include_system: bool = True,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List available roles."""
        params: dict[str, Any] = {
            "include_system": include_system,
            "limit": limit,
            "offset": offset,
        }
        return await self._client.request("GET", "/api/v1/rbac/roles", params=params)

    async def get_role(self, role_id: str) -> dict[str, Any]:
        """Get a role by ID."""
        return await self._client.request("GET", f"/api/v1/rbac/roles/{role_id}")

    async def create_role(
        self,
        name: str,
        permissions: list[str],
        description: str | None = None,
        parent_role: str | None = None,
    ) -> dict[str, Any]:
        """Create a custom role."""
        data: dict[str, Any] = {"name": name, "permissions": permissions}
        if description:
            data["description"] = description
        if parent_role:
            data["parent_role"] = parent_role

        return await self._client.request("POST", "/api/v1/rbac/roles", json=data)

    async def update_role(
        self,
        role_id: str,
        permissions: list[str] | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Update a role."""
        data: dict[str, Any] = {}
        if permissions is not None:
            data["permissions"] = permissions
        if description is not None:
            data["description"] = description

        return await self._client.request("PUT", f"/api/v1/rbac/roles/{role_id}", json=data)

    async def delete_role(self, role_id: str) -> dict[str, Any]:
        """Delete a custom role."""
        return await self._client.request("DELETE", f"/api/v1/rbac/roles/{role_id}")

    async def assign_role(
        self,
        user_id: str,
        role_id: str,
        scope: str | None = None,
    ) -> dict[str, Any]:
        """Assign a role to a user."""
        data: dict[str, Any] = {"user_id": user_id, "role_id": role_id}
        if scope:
            data["scope"] = scope

        return await self._client.request("POST", "/api/v1/rbac/assignments", json=data)

    async def revoke_role(self, user_id: str, role_id: str) -> dict[str, Any]:
        """Revoke a role from a user."""
        return await self._client.request(
            "DELETE",
            "/api/v1/rbac/assignments",
            params={"user_id": user_id, "role_id": role_id},
        )

    async def get_user_roles(self, user_id: str) -> dict[str, Any]:
        """Get roles assigned to a user."""
        return await self._client.request("GET", f"/api/v1/rbac/users/{user_id}/roles")

    async def check_permission(
        self,
        user_id: str,
        permission: str,
        resource_id: str | None = None,
    ) -> dict[str, Any]:
        """Check if a user has a specific permission."""
        data: dict[str, Any] = {"user_id": user_id, "permission": permission}
        if resource_id:
            data["resource_id"] = resource_id

        return await self._client.request("POST", "/api/v1/rbac/check", json=data)

    async def get_effective_permissions(self, user_id: str) -> dict[str, Any]:
        """Get all effective permissions for a user."""
        return await self._client.request("GET", f"/api/v1/rbac/users/{user_id}/permissions")
