"""RBACAPI resource for the Aragora client."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..client import AragoraClient


@dataclass
class Permission:
    """RBAC permission definition."""

    id: str
    name: str
    description: str
    resource: str
    action: str
    conditions: Optional[dict[str, Any]] = None


@dataclass
class Role:
    """RBAC role definition."""

    id: str
    name: str
    description: str
    permissions: list[str]
    is_system: bool = False
    inherits_from: list[str] = field(default_factory=list)
    tenant_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class RoleAssignment:
    """Role assignment to a user."""

    id: str
    user_id: str
    role_id: str
    role_name: str
    tenant_id: Optional[str] = None
    assigned_at: str = ""
    assigned_by: Optional[str] = None


@dataclass
class PermissionCheck:
    """Result of a permission check."""

    allowed: bool
    permission: str
    resource: Optional[str] = None
    reason: Optional[str] = None


class RBACAPI:
    """API interface for Role-Based Access Control (RBAC)."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    # -------------------------------------------------------------------------
    # Roles
    # -------------------------------------------------------------------------

    def list_roles(
        self, tenant_id: Optional[str] = None, limit: int = 50, offset: int = 0
    ) -> tuple[list[Role], int]:
        """
        List all roles.

        Args:
            tenant_id: Optional tenant ID to filter by.
            limit: Maximum number of roles to return.
            offset: Offset for pagination.

        Returns:
            Tuple of (list of Role objects, total count).
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if tenant_id:
            params["tenant_id"] = tenant_id
        response = self._client._get("/api/v1/rbac/roles", params=params)
        roles = [Role(**r) for r in response.get("roles", [])]
        return roles, response.get("total", len(roles))

    async def list_roles_async(
        self, tenant_id: Optional[str] = None, limit: int = 50, offset: int = 0
    ) -> tuple[list[Role], int]:
        """Async version of list_roles()."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if tenant_id:
            params["tenant_id"] = tenant_id
        response = await self._client._get_async("/api/v1/rbac/roles", params=params)
        roles = [Role(**r) for r in response.get("roles", [])]
        return roles, response.get("total", len(roles))

    def get_role(self, role_id: str) -> Role:
        """
        Get a specific role.

        Args:
            role_id: The role ID.

        Returns:
            Role object.
        """
        response = self._client._get(f"/api/v1/rbac/roles/{role_id}")
        return Role(**response)

    async def get_role_async(self, role_id: str) -> Role:
        """Async version of get_role()."""
        response = await self._client._get_async(f"/api/v1/rbac/roles/{role_id}")
        return Role(**response)

    def create_role(
        self,
        name: str,
        description: str,
        permissions: list[str],
        inherits_from: Optional[list[str]] = None,
        tenant_id: Optional[str] = None,
    ) -> Role:
        """
        Create a new role.

        Args:
            name: Role name.
            description: Role description.
            permissions: List of permission IDs.
            inherits_from: Optional list of role IDs to inherit from.
            tenant_id: Optional tenant ID for tenant-scoped role.

        Returns:
            Created Role object.
        """
        body: dict[str, Any] = {
            "name": name,
            "description": description,
            "permissions": permissions,
        }
        if inherits_from:
            body["inherits_from"] = inherits_from
        if tenant_id:
            body["tenant_id"] = tenant_id
        response = self._client._post("/api/v1/rbac/roles", data=body)
        return Role(**response)

    async def create_role_async(
        self,
        name: str,
        description: str,
        permissions: list[str],
        inherits_from: Optional[list[str]] = None,
        tenant_id: Optional[str] = None,
    ) -> Role:
        """Async version of create_role()."""
        body: dict[str, Any] = {
            "name": name,
            "description": description,
            "permissions": permissions,
        }
        if inherits_from:
            body["inherits_from"] = inherits_from
        if tenant_id:
            body["tenant_id"] = tenant_id
        response = await self._client._post_async("/api/v1/rbac/roles", data=body)
        return Role(**response)

    def update_role(
        self,
        role_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        permissions: Optional[list[str]] = None,
        inherits_from: Optional[list[str]] = None,
    ) -> Role:
        """
        Update a role.

        Args:
            role_id: The role ID.
            name: Optional new name.
            description: Optional new description.
            permissions: Optional new permissions list.
            inherits_from: Optional new inheritance list.

        Returns:
            Updated Role object.
        """
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if permissions is not None:
            body["permissions"] = permissions
        if inherits_from is not None:
            body["inherits_from"] = inherits_from
        response = self._client._patch(f"/api/v1/rbac/roles/{role_id}", data=body)
        return Role(**response)

    async def update_role_async(
        self,
        role_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        permissions: Optional[list[str]] = None,
        inherits_from: Optional[list[str]] = None,
    ) -> Role:
        """Async version of update_role()."""
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if description is not None:
            body["description"] = description
        if permissions is not None:
            body["permissions"] = permissions
        if inherits_from is not None:
            body["inherits_from"] = inherits_from
        response = await self._client._patch_async(f"/api/v1/rbac/roles/{role_id}", data=body)
        return Role(**response)

    def delete_role(self, role_id: str) -> None:
        """
        Delete a role.

        Args:
            role_id: The role ID to delete.
        """
        self._client._delete(f"/api/v1/rbac/roles/{role_id}")

    async def delete_role_async(self, role_id: str) -> None:
        """Async version of delete_role()."""
        await self._client._delete_async(f"/api/v1/rbac/roles/{role_id}")

    # -------------------------------------------------------------------------
    # Permissions
    # -------------------------------------------------------------------------

    def list_permissions(self) -> list[Permission]:
        """
        List all available permissions.

        Returns:
            List of Permission objects.
        """
        response = self._client._get("/api/v1/rbac/permissions")
        return [Permission(**p) for p in response.get("permissions", [])]

    async def list_permissions_async(self) -> list[Permission]:
        """Async version of list_permissions()."""
        response = await self._client._get_async("/api/v1/rbac/permissions")
        return [Permission(**p) for p in response.get("permissions", [])]

    def check_permission(
        self,
        permission: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        resource_id: Optional[str] = None,
    ) -> PermissionCheck:
        """
        Check if a user has a specific permission.

        Args:
            permission: The permission to check.
            user_id: Optional user ID (defaults to current user).
            resource: Optional resource type.
            resource_id: Optional specific resource ID.

        Returns:
            PermissionCheck result.
        """
        body: dict[str, Any] = {"permission": permission}
        if user_id:
            body["user_id"] = user_id
        if resource:
            body["resource"] = resource
        if resource_id:
            body["resource_id"] = resource_id
        response = self._client._post("/api/v1/rbac/check", data=body)
        return PermissionCheck(**response)

    async def check_permission_async(
        self,
        permission: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        resource_id: Optional[str] = None,
    ) -> PermissionCheck:
        """Async version of check_permission()."""
        body: dict[str, Any] = {"permission": permission}
        if user_id:
            body["user_id"] = user_id
        if resource:
            body["resource"] = resource
        if resource_id:
            body["resource_id"] = resource_id
        response = await self._client._post_async("/api/v1/rbac/check", data=body)
        return PermissionCheck(**response)

    # -------------------------------------------------------------------------
    # Role Assignments
    # -------------------------------------------------------------------------

    def list_assignments(
        self,
        user_id: Optional[str] = None,
        role_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[RoleAssignment], int]:
        """
        List role assignments.

        Args:
            user_id: Optional user ID to filter by.
            role_id: Optional role ID to filter by.
            tenant_id: Optional tenant ID to filter by.
            limit: Maximum number of assignments to return.
            offset: Offset for pagination.

        Returns:
            Tuple of (list of RoleAssignment objects, total count).
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if user_id:
            params["user_id"] = user_id
        if role_id:
            params["role_id"] = role_id
        if tenant_id:
            params["tenant_id"] = tenant_id
        response = self._client._get("/api/v1/rbac/assignments", params=params)
        assignments = [RoleAssignment(**a) for a in response.get("assignments", [])]
        return assignments, response.get("total", len(assignments))

    async def list_assignments_async(
        self,
        user_id: Optional[str] = None,
        role_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[RoleAssignment], int]:
        """Async version of list_assignments()."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if user_id:
            params["user_id"] = user_id
        if role_id:
            params["role_id"] = role_id
        if tenant_id:
            params["tenant_id"] = tenant_id
        response = await self._client._get_async("/api/v1/rbac/assignments", params=params)
        assignments = [RoleAssignment(**a) for a in response.get("assignments", [])]
        return assignments, response.get("total", len(assignments))

    def assign_role(
        self, user_id: str, role_id: str, tenant_id: Optional[str] = None
    ) -> RoleAssignment:
        """
        Assign a role to a user.

        Args:
            user_id: The user ID.
            role_id: The role ID to assign.
            tenant_id: Optional tenant ID for tenant-scoped assignment.

        Returns:
            Created RoleAssignment object.
        """
        body: dict[str, Any] = {"user_id": user_id, "role_id": role_id}
        if tenant_id:
            body["tenant_id"] = tenant_id
        response = self._client._post("/api/v1/rbac/assignments", data=body)
        return RoleAssignment(**response)

    async def assign_role_async(
        self, user_id: str, role_id: str, tenant_id: Optional[str] = None
    ) -> RoleAssignment:
        """Async version of assign_role()."""
        body: dict[str, Any] = {"user_id": user_id, "role_id": role_id}
        if tenant_id:
            body["tenant_id"] = tenant_id
        response = await self._client._post_async("/api/v1/rbac/assignments", data=body)
        return RoleAssignment(**response)

    def revoke_role(self, user_id: str, role_id: str, tenant_id: Optional[str] = None) -> None:
        """
        Revoke a role from a user.

        Args:
            user_id: The user ID.
            role_id: The role ID to revoke.
            tenant_id: Optional tenant ID for tenant-scoped revocation.
        """
        body: dict[str, Any] = {"user_id": user_id, "role_id": role_id}
        if tenant_id:
            body["tenant_id"] = tenant_id
        self._client._post("/api/v1/rbac/revoke", data=body)

    async def revoke_role_async(
        self, user_id: str, role_id: str, tenant_id: Optional[str] = None
    ) -> None:
        """Async version of revoke_role()."""
        body: dict[str, Any] = {"user_id": user_id, "role_id": role_id}
        if tenant_id:
            body["tenant_id"] = tenant_id
        await self._client._post_async("/api/v1/rbac/revoke", data=body)

    def bulk_assign(self, assignments: list[dict[str, str]]) -> list[RoleAssignment]:
        """
        Bulk assign roles to users.

        Args:
            assignments: List of dicts with user_id, role_id, and optional tenant_id.

        Returns:
            List of created RoleAssignment objects.
        """
        response = self._client._post(
            "/api/v1/rbac/assignments/bulk", json={"assignments": assignments}
        )
        return [RoleAssignment(**a) for a in response.get("assignments", [])]

    async def bulk_assign_async(self, assignments: list[dict[str, str]]) -> list[RoleAssignment]:
        """Async version of bulk_assign()."""
        response = await self._client._post_async(
            "/api/v1/rbac/assignments/bulk", json={"assignments": assignments}
        )
        return [RoleAssignment(**a) for a in response.get("assignments", [])]

    # -------------------------------------------------------------------------
    # User Permissions
    # -------------------------------------------------------------------------

    def get_user_permissions(
        self, user_id: Optional[str] = None, tenant_id: Optional[str] = None
    ) -> list[str]:
        """
        Get all effective permissions for a user.

        Args:
            user_id: Optional user ID (defaults to current user).
            tenant_id: Optional tenant ID to get tenant-specific permissions.

        Returns:
            List of permission strings.
        """
        params: dict[str, Any] = {}
        if user_id:
            params["user_id"] = user_id
        if tenant_id:
            params["tenant_id"] = tenant_id
        response = self._client._get("/api/v1/rbac/user-permissions", params=params)
        return response.get("permissions", [])

    async def get_user_permissions_async(
        self, user_id: Optional[str] = None, tenant_id: Optional[str] = None
    ) -> list[str]:
        """Async version of get_user_permissions()."""
        params: dict[str, Any] = {}
        if user_id:
            params["user_id"] = user_id
        if tenant_id:
            params["tenant_id"] = tenant_id
        response = await self._client._get_async("/api/v1/rbac/user-permissions", params=params)
        return response.get("permissions", [])

    def get_user_roles(
        self, user_id: Optional[str] = None, tenant_id: Optional[str] = None
    ) -> list[Role]:
        """
        Get all roles assigned to a user.

        Args:
            user_id: Optional user ID (defaults to current user).
            tenant_id: Optional tenant ID to get tenant-specific roles.

        Returns:
            List of Role objects.
        """
        params: dict[str, Any] = {}
        if user_id:
            params["user_id"] = user_id
        if tenant_id:
            params["tenant_id"] = tenant_id
        response = self._client._get("/api/v1/rbac/user-roles", params=params)
        return [Role(**r) for r in response.get("roles", [])]

    async def get_user_roles_async(
        self, user_id: Optional[str] = None, tenant_id: Optional[str] = None
    ) -> list[Role]:
        """Async version of get_user_roles()."""
        params: dict[str, Any] = {}
        if user_id:
            params["user_id"] = user_id
        if tenant_id:
            params["tenant_id"] = tenant_id
        response = await self._client._get_async("/api/v1/rbac/user-roles", params=params)
        return [Role(**r) for r in response.get("roles", [])]
