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

    def list_profiles(self) -> dict[str, Any]:
        """List available RBAC profiles (lite, standard, enterprise)."""
        return self._client.request("GET", "/api/v1/workspaces/profiles")

    # ========== Audit Trail ==========

    def query_audit(
        self,
        action: str | None = None,
        user_id: str | None = None,
        since: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Query audit log entries."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if action:
            params["action"] = action
        if user_id:
            params["user_id"] = user_id
        if since:
            params["since"] = since
        return self._client.request("GET", "/api/v1/audit/entries", params=params)

    def get_audit_report(
        self, framework: str | None = None, since: str | None = None
    ) -> dict[str, Any]:
        """Generate compliance audit report."""
        params: dict[str, Any] = {}
        if framework:
            params["framework"] = framework
        if since:
            params["since"] = since
        return self._client.request("GET", "/api/v1/audit/report", params=params)

    def verify_audit_integrity(self) -> dict[str, Any]:
        """Verify audit log integrity."""
        return self._client.request("GET", "/api/v1/audit/verify")

    def get_denied_access(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """Get denied access attempts."""
        return self._client.request(
            "GET", "/api/v1/audit/denied", params={"limit": limit, "offset": offset}
        )

    # ========== API Keys ==========

    def generate_api_key(
        self,
        name: str,
        permissions: list[str] | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        """Generate a new API key."""
        payload: dict[str, Any] = {"name": name}
        if permissions:
            payload["permissions"] = permissions
        if expires_at:
            payload["expires_at"] = expires_at
        return self._client.request("POST", "/api/auth/api-key", json=payload)

    def list_api_keys(self) -> dict[str, Any]:
        """List API keys for current user."""
        return self._client.request("GET", "/api/keys")

    def revoke_api_key(self, key_id: str) -> dict[str, Any]:
        """Revoke an API key."""
        return self._client.request("DELETE", f"/api/keys/{key_id}")

    # ========== Sessions ==========

    def list_sessions(self) -> dict[str, Any]:
        """List active sessions for current user."""
        return self._client.request("GET", "/api/auth/sessions")

    def revoke_session(self, session_id: str) -> dict[str, Any]:
        """Revoke a specific session."""
        return self._client.request("DELETE", f"/api/auth/sessions/{session_id}")

    def logout_all(self) -> dict[str, Any]:
        """Logout from all devices."""
        return self._client.request("POST", "/api/auth/logout-all", json={})

    # ========== MFA ==========

    def setup_mfa(self) -> dict[str, Any]:
        """Setup MFA - generate secret and QR code."""
        return self._client.request("POST", "/api/auth/mfa/setup", json={})

    def enable_mfa(self, code: str) -> dict[str, Any]:
        """Enable MFA by verifying setup code."""
        return self._client.request("POST", "/api/auth/mfa/enable", json={"code": code})

    def disable_mfa(self, code: str) -> dict[str, Any]:
        """Disable MFA."""
        return self._client.request("POST", "/api/auth/mfa/disable", json={"code": code})

    def verify_mfa(self, code: str) -> dict[str, Any]:
        """Verify MFA code during login."""
        return self._client.request("POST", "/api/auth/mfa/verify", json={"code": code})

    def regenerate_backup_codes(self, code: str) -> dict[str, Any]:
        """Regenerate MFA backup codes."""
        return self._client.request("POST", "/api/auth/mfa/backup-codes", json={"code": code})

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

    async def list_profiles(self) -> dict[str, Any]:
        """List available RBAC profiles (lite, standard, enterprise)."""
        return await self._client.request("GET", "/api/v1/workspaces/profiles")

    # ========== Audit Trail ==========

    async def query_audit(
        self,
        action: str | None = None,
        user_id: str | None = None,
        since: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Query audit log entries."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if action:
            params["action"] = action
        if user_id:
            params["user_id"] = user_id
        if since:
            params["since"] = since
        return await self._client.request("GET", "/api/v1/audit/entries", params=params)

    async def get_audit_report(
        self, framework: str | None = None, since: str | None = None
    ) -> dict[str, Any]:
        """Generate compliance audit report."""
        params: dict[str, Any] = {}
        if framework:
            params["framework"] = framework
        if since:
            params["since"] = since
        return await self._client.request("GET", "/api/v1/audit/report", params=params)

    async def verify_audit_integrity(self) -> dict[str, Any]:
        """Verify audit log integrity."""
        return await self._client.request("GET", "/api/v1/audit/verify")

    async def get_denied_access(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """Get denied access attempts."""
        return await self._client.request(
            "GET", "/api/v1/audit/denied", params={"limit": limit, "offset": offset}
        )

    # ========== API Keys ==========

    async def generate_api_key(
        self,
        name: str,
        permissions: list[str] | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        """Generate a new API key."""
        payload: dict[str, Any] = {"name": name}
        if permissions:
            payload["permissions"] = permissions
        if expires_at:
            payload["expires_at"] = expires_at
        return await self._client.request("POST", "/api/auth/api-key", json=payload)

    async def list_api_keys(self) -> dict[str, Any]:
        """List API keys for current user."""
        return await self._client.request("GET", "/api/keys")

    async def revoke_api_key(self, key_id: str) -> dict[str, Any]:
        """Revoke an API key."""
        return await self._client.request("DELETE", f"/api/keys/{key_id}")

    # ========== Sessions ==========

    async def list_sessions(self) -> dict[str, Any]:
        """List active sessions for current user."""
        return await self._client.request("GET", "/api/auth/sessions")

    async def revoke_session(self, session_id: str) -> dict[str, Any]:
        """Revoke a specific session."""
        return await self._client.request("DELETE", f"/api/auth/sessions/{session_id}")

    async def logout_all(self) -> dict[str, Any]:
        """Logout from all devices."""
        return await self._client.request("POST", "/api/auth/logout-all", json={})

    # ========== MFA ==========

    async def setup_mfa(self) -> dict[str, Any]:
        """Setup MFA - generate secret and QR code."""
        return await self._client.request("POST", "/api/auth/mfa/setup", json={})

    async def enable_mfa(self, code: str) -> dict[str, Any]:
        """Enable MFA by verifying setup code."""
        return await self._client.request("POST", "/api/auth/mfa/enable", json={"code": code})

    async def disable_mfa(self, code: str) -> dict[str, Any]:
        """Disable MFA."""
        return await self._client.request("POST", "/api/auth/mfa/disable", json={"code": code})

    async def verify_mfa(self, code: str) -> dict[str, Any]:
        """Verify MFA code during login."""
        return await self._client.request("POST", "/api/auth/mfa/verify", json={"code": code})

    async def regenerate_backup_codes(self, code: str) -> dict[str, Any]:
        """Regenerate MFA backup codes."""
        return await self._client.request("POST", "/api/auth/mfa/backup-codes", json={"code": code})
