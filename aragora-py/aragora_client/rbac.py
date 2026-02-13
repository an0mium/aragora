"""RBAC API for the Aragora SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from aragora_client.client import AragoraClient


class Permission(BaseModel):
    """Permission model."""

    id: str
    name: str
    description: str | None = None
    resource_type: str | None = None


class Role(BaseModel):
    """Role model."""

    id: str
    name: str
    description: str | None = None
    permissions: list[str] = []
    is_system: bool = False
    created_at: str | None = None
    updated_at: str | None = None


class CreateRoleRequest(BaseModel):
    """Create role request."""

    name: str
    description: str | None = None
    permissions: list[str] = []


class UpdateRoleRequest(BaseModel):
    """Update role request."""

    name: str | None = None
    description: str | None = None
    permissions: list[str] | None = None


class RoleAssignment(BaseModel):
    """Role assignment."""

    user_id: str
    role_id: str
    tenant_id: str | None = None
    assigned_at: str | None = None
    assigned_by: str | None = None


class AssignRoleRequest(BaseModel):
    """Assign role request."""

    user_id: str
    role_id: str
    tenant_id: str | None = None


class BulkAssignRequest(BaseModel):
    """Bulk assign roles request."""

    assignments: list[AssignRoleRequest]


class BulkAssignResponse(BaseModel):
    """Bulk assign roles response."""

    success_count: int
    failure_count: int
    failures: list[dict[str, Any]] = []


class PermissionCheck(BaseModel):
    """Permission check result."""

    allowed: bool
    reason: str | None = None


class RBACAPI:
    """API for role-based access control operations."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # Roles
    async def list_roles(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Role]:
        """List all roles."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        data = await self._client._get("/api/v1/rbac/roles", params=params)
        return [Role.model_validate(r) for r in data.get("roles", [])]

    async def get_role(self, role_id: str) -> Role:
        """Get a role by ID."""
        data = await self._client._get(f"/api/v1/rbac/roles/{role_id}")
        return Role.model_validate(data)

    async def create_role(
        self,
        name: str,
        *,
        description: str | None = None,
        permissions: list[str] | None = None,
    ) -> Role:
        """Create a new role."""
        request = CreateRoleRequest(
            name=name,
            description=description,
            permissions=permissions or [],
        )
        data = await self._client._post("/api/v1/rbac/roles", request.model_dump())
        return Role.model_validate(data)

    async def update_role(
        self,
        role_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        permissions: list[str] | None = None,
    ) -> Role:
        """Update a role."""
        request = UpdateRoleRequest(
            name=name,
            description=description,
            permissions=permissions,
        )
        data = await self._client._patch(
            f"/api/v1/rbac/roles/{role_id}", request.model_dump(exclude_none=True)
        )
        return Role.model_validate(data)

    async def delete_role(self, role_id: str) -> None:
        """Delete a role."""
        await self._client._delete(f"/api/v1/rbac/roles/{role_id}")

    # Permissions
    async def list_permissions(self) -> list[Permission]:
        """List all available permissions."""
        data = await self._client._get("/api/v1/rbac/permissions")
        return [Permission.model_validate(p) for p in data.get("permissions", [])]

    # Assignments
    async def assign_role(
        self,
        user_id: str,
        role_id: str,
        *,
        tenant_id: str | None = None,
    ) -> None:
        """Assign a role to a user."""
        await self._client._post(
            "/api/v1/rbac/assignments",
            {"user_id": user_id, "role_id": role_id, "tenant_id": tenant_id},
        )

    async def revoke_role(
        self,
        user_id: str,
        role_id: str,
        *,
        tenant_id: str | None = None,
    ) -> None:
        """Revoke a role from a user."""
        await self._client._delete_with_body(
            "/api/v1/rbac/assignments",
            {"user_id": user_id, "role_id": role_id, "tenant_id": tenant_id},
        )

    async def get_user_roles(
        self,
        user_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Role]:
        """Get all roles assigned to a user."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        data = await self._client._get(
            f"/api/v1/rbac/users/{user_id}/roles", params=params
        )
        return [Role.model_validate(r) for r in data.get("roles", [])]

    async def check_permission(
        self,
        user_id: str,
        permission: str,
        *,
        resource: str | None = None,
    ) -> PermissionCheck:
        """Check if a user has a specific permission."""
        params: dict[str, Any] = {"user_id": user_id, "permission": permission}
        if resource:
            params["resource"] = resource
        data = await self._client._get("/api/v1/rbac/check", params=params)
        return PermissionCheck.model_validate(data)

    async def list_role_assignments(
        self,
        role_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[RoleAssignment]:
        """List all users assigned to a role."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        data = await self._client._get(
            f"/api/v1/rbac/roles/{role_id}/assignments", params=params
        )
        return [RoleAssignment.model_validate(a) for a in data.get("assignments", [])]

    async def bulk_assign_roles(
        self,
        assignments: list[dict[str, Any]],
    ) -> BulkAssignResponse:
        """Bulk assign roles to multiple users."""
        data = await self._client._post(
            "/api/v1/rbac/assignments/bulk", {"assignments": assignments}
        )
        return BulkAssignResponse.model_validate(data)

    async def get_effective_permissions(
        self,
        user_id: str,
        *,
        tenant_id: str | None = None,
    ) -> list[str]:
        """Get all effective permissions for a user."""
        params: dict[str, Any] = {}
        if tenant_id:
            params["tenant_id"] = tenant_id
        data = await self._client._get(
            f"/api/v1/rbac/users/{user_id}/permissions", params=params
        )
        return data.get("permissions", [])
