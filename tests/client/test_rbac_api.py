"""
Tests for RBAC API resource.

Tests cover:
- RBACAPI role management (list, get, create, update, delete)
- RBACAPI permission management (list, check)
- RBACAPI role assignments (list, assign, revoke, bulk)
- RBACAPI user permissions and roles
- Dataclass models (Permission, Role, RoleAssignment, PermissionCheck)
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.client.client import AragoraClient
from aragora.client.resources.rbac import (
    Permission,
    PermissionCheck,
    RBACAPI,
    Role,
    RoleAssignment,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_client() -> AragoraClient:
    """Create a mock AragoraClient."""
    client = MagicMock(spec=AragoraClient)
    return client


@pytest.fixture
def rbac_api(mock_client: AragoraClient) -> RBACAPI:
    """Create a RBACAPI with mock client."""
    return RBACAPI(mock_client)


@pytest.fixture
def sample_timestamp() -> str:
    """Sample ISO timestamp for tests."""
    return datetime.now(timezone.utc).isoformat()


# ============================================================================
# Permission Dataclass Tests
# ============================================================================


class TestPermissionDataclass:
    """Tests for Permission dataclass."""

    def test_permission_basic(self):
        """Test Permission with required fields."""
        perm = Permission(
            id="perm-1",
            name="Read Debates",
            description="Can read debates",
            resource="debates",
            action="read",
        )
        assert perm.id == "perm-1"
        assert perm.name == "Read Debates"
        assert perm.resource == "debates"
        assert perm.action == "read"
        assert perm.conditions is None

    def test_permission_with_conditions(self):
        """Test Permission with conditions."""
        perm = Permission(
            id="perm-2",
            name="Edit Own",
            description="Can edit own resources",
            resource="documents",
            action="update",
            conditions={"owner": "${user.id}"},
        )
        assert perm.conditions == {"owner": "${user.id}"}


# ============================================================================
# Role Dataclass Tests
# ============================================================================


class TestRoleDataclass:
    """Tests for Role dataclass."""

    def test_role_minimal(self):
        """Test Role with required fields."""
        role = Role(
            id="role-1",
            name="Admin",
            description="Administrator role",
            permissions=["perm-1", "perm-2"],
        )
        assert role.id == "role-1"
        assert role.name == "Admin"
        assert role.is_system is False
        assert role.inherits_from == []
        assert role.tenant_id is None

    def test_role_full(self):
        """Test Role with all fields."""
        role = Role(
            id="role-2",
            name="Super Admin",
            description="Full access",
            permissions=["*"],
            is_system=True,
            inherits_from=["role-1"],
            tenant_id="tenant-123",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-02T00:00:00Z",
        )
        assert role.is_system is True
        assert role.inherits_from == ["role-1"]
        assert role.tenant_id == "tenant-123"


# ============================================================================
# RoleAssignment Dataclass Tests
# ============================================================================


class TestRoleAssignmentDataclass:
    """Tests for RoleAssignment dataclass."""

    def test_assignment_minimal(self):
        """Test RoleAssignment with required fields."""
        assignment = RoleAssignment(
            id="assign-1",
            user_id="user-123",
            role_id="role-1",
            role_name="Admin",
        )
        assert assignment.id == "assign-1"
        assert assignment.user_id == "user-123"
        assert assignment.role_id == "role-1"
        assert assignment.role_name == "Admin"
        assert assignment.tenant_id is None
        assert assignment.assigned_at == ""
        assert assignment.assigned_by is None

    def test_assignment_full(self):
        """Test RoleAssignment with all fields."""
        assignment = RoleAssignment(
            id="assign-2",
            user_id="user-456",
            role_id="role-2",
            role_name="Manager",
            tenant_id="tenant-789",
            assigned_at="2024-01-01T00:00:00Z",
            assigned_by="admin-user",
        )
        assert assignment.tenant_id == "tenant-789"
        assert assignment.assigned_by == "admin-user"


# ============================================================================
# PermissionCheck Dataclass Tests
# ============================================================================


class TestPermissionCheckDataclass:
    """Tests for PermissionCheck dataclass."""

    def test_permission_check_allowed(self):
        """Test PermissionCheck when allowed."""
        check = PermissionCheck(
            allowed=True,
            permission="debates:read",
        )
        assert check.allowed is True
        assert check.permission == "debates:read"
        assert check.resource is None
        assert check.reason is None

    def test_permission_check_denied(self):
        """Test PermissionCheck when denied."""
        check = PermissionCheck(
            allowed=False,
            permission="debates:delete",
            resource="debates",
            reason="Insufficient privileges",
        )
        assert check.allowed is False
        assert check.reason == "Insufficient privileges"


# ============================================================================
# RBACAPI Role Methods Tests
# ============================================================================


class TestRBACAPIListRoles:
    """Tests for RBACAPI.list_roles() method."""

    def test_list_roles_basic(
        self, rbac_api: RBACAPI, mock_client: MagicMock, sample_timestamp: str
    ):
        """Test list_roles() basic call."""
        mock_client._get.return_value = {
            "roles": [
                {
                    "id": "role-1",
                    "name": "Admin",
                    "description": "Administrator",
                    "permissions": ["*"],
                },
                {
                    "id": "role-2",
                    "name": "User",
                    "description": "Regular user",
                    "permissions": ["read"],
                },
            ],
            "total": 2,
        }

        roles, total = rbac_api.list_roles()

        assert len(roles) == 2
        assert total == 2
        assert roles[0].id == "role-1"
        assert roles[0].name == "Admin"
        mock_client._get.assert_called_once()

    def test_list_roles_with_tenant(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test list_roles() with tenant filter."""
        mock_client._get.return_value = {"roles": [], "total": 0}

        rbac_api.list_roles(tenant_id="tenant-123", limit=10, offset=5)

        call_args = mock_client._get.call_args
        params = call_args[1]["params"]
        assert params["tenant_id"] == "tenant-123"
        assert params["limit"] == 10
        assert params["offset"] == 5


class TestRBACAPIListRolesAsync:
    """Tests for RBACAPI.list_roles_async() method."""

    @pytest.mark.asyncio
    async def test_list_roles_async(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test list_roles_async() call."""
        mock_client._get_async = AsyncMock(
            return_value={
                "roles": [
                    {"id": "r-async", "name": "Async Role", "description": "", "permissions": []}
                ],
                "total": 1,
            }
        )

        roles, total = await rbac_api.list_roles_async()

        assert len(roles) == 1
        assert roles[0].id == "r-async"


class TestRBACAPIGetRole:
    """Tests for RBACAPI.get_role() method."""

    def test_get_role(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test get_role() retrieves a role."""
        mock_client._get.return_value = {
            "id": "role-admin",
            "name": "Admin",
            "description": "Full access",
            "permissions": ["*"],
            "is_system": True,
        }

        role = rbac_api.get_role("role-admin")

        assert role.id == "role-admin"
        assert role.is_system is True
        mock_client._get.assert_called_once_with("/api/v1/rbac/roles/role-admin")


class TestRBACAPIGetRoleAsync:
    """Tests for RBACAPI.get_role_async() method."""

    @pytest.mark.asyncio
    async def test_get_role_async(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test get_role_async() call."""
        mock_client._get_async = AsyncMock(
            return_value={
                "id": "r-async",
                "name": "Async Role",
                "description": "",
                "permissions": [],
            }
        )

        role = await rbac_api.get_role_async("r-async")

        assert role.id == "r-async"


class TestRBACAPICreateRole:
    """Tests for RBACAPI.create_role() method."""

    def test_create_role_basic(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test create_role() basic call."""
        mock_client._post.return_value = {
            "id": "role-new",
            "name": "New Role",
            "description": "A new role",
            "permissions": ["debates:read", "debates:write"],
        }

        role = rbac_api.create_role(
            name="New Role",
            description="A new role",
            permissions=["debates:read", "debates:write"],
        )

        assert role.id == "role-new"
        assert role.name == "New Role"
        call_args = mock_client._post.call_args
        assert call_args[0][0] == "/api/v1/rbac/roles"
        body = call_args[1]["data"]
        assert body["name"] == "New Role"
        assert body["permissions"] == ["debates:read", "debates:write"]

    def test_create_role_with_inheritance(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test create_role() with inheritance."""
        mock_client._post.return_value = {
            "id": "role-child",
            "name": "Child Role",
            "description": "Inherits from parent",
            "permissions": [],
            "inherits_from": ["role-parent"],
            "tenant_id": "tenant-1",
        }

        role = rbac_api.create_role(
            name="Child Role",
            description="Inherits from parent",
            permissions=[],
            inherits_from=["role-parent"],
            tenant_id="tenant-1",
        )

        assert role.inherits_from == ["role-parent"]
        assert role.tenant_id == "tenant-1"


class TestRBACAPICreateRoleAsync:
    """Tests for RBACAPI.create_role_async() method."""

    @pytest.mark.asyncio
    async def test_create_role_async(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test create_role_async() call."""
        mock_client._post_async = AsyncMock(
            return_value={
                "id": "r-async-new",
                "name": "Async New",
                "description": "",
                "permissions": [],
            }
        )

        role = await rbac_api.create_role_async(
            name="Async New",
            description="",
            permissions=[],
        )

        assert role.id == "r-async-new"


class TestRBACAPIUpdateRole:
    """Tests for RBACAPI.update_role() method."""

    def test_update_role_name(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test update_role() with name change."""
        mock_client._patch.return_value = {
            "id": "role-1",
            "name": "Updated Name",
            "description": "Original desc",
            "permissions": ["read"],
        }

        role = rbac_api.update_role("role-1", name="Updated Name")

        assert role.name == "Updated Name"
        call_args = mock_client._patch.call_args
        assert call_args[0][0] == "/api/v1/rbac/roles/role-1"
        body = call_args[1]["data"]
        assert body == {"name": "Updated Name"}

    def test_update_role_permissions(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test update_role() with permissions change."""
        mock_client._patch.return_value = {
            "id": "role-2",
            "name": "Role 2",
            "description": "Desc",
            "permissions": ["*"],
        }

        role = rbac_api.update_role("role-2", permissions=["*"])

        call_args = mock_client._patch.call_args
        body = call_args[1]["data"]
        assert body["permissions"] == ["*"]

    def test_update_role_multiple_fields(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test update_role() with multiple fields."""
        mock_client._patch.return_value = {
            "id": "role-3",
            "name": "New Name",
            "description": "New Desc",
            "permissions": ["a", "b"],
            "inherits_from": ["role-1"],
        }

        role = rbac_api.update_role(
            "role-3",
            name="New Name",
            description="New Desc",
            permissions=["a", "b"],
            inherits_from=["role-1"],
        )

        call_args = mock_client._patch.call_args
        body = call_args[1]["data"]
        assert body["name"] == "New Name"
        assert body["description"] == "New Desc"
        assert body["permissions"] == ["a", "b"]
        assert body["inherits_from"] == ["role-1"]


class TestRBACAPIUpdateRoleAsync:
    """Tests for RBACAPI.update_role_async() method."""

    @pytest.mark.asyncio
    async def test_update_role_async(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test update_role_async() call."""
        mock_client._patch_async = AsyncMock(
            return_value={
                "id": "r-async",
                "name": "Updated Async",
                "description": "",
                "permissions": [],
            }
        )

        role = await rbac_api.update_role_async("r-async", name="Updated Async")

        assert role.name == "Updated Async"


class TestRBACAPIDeleteRole:
    """Tests for RBACAPI.delete_role() method."""

    def test_delete_role(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test delete_role() call."""
        mock_client._delete.return_value = None

        rbac_api.delete_role("role-to-delete")

        mock_client._delete.assert_called_once_with("/api/v1/rbac/roles/role-to-delete")


class TestRBACAPIDeleteRoleAsync:
    """Tests for RBACAPI.delete_role_async() method."""

    @pytest.mark.asyncio
    async def test_delete_role_async(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test delete_role_async() call."""
        mock_client._delete_async = AsyncMock(return_value=None)

        await rbac_api.delete_role_async("role-async-delete")

        mock_client._delete_async.assert_called_once()


# ============================================================================
# RBACAPI Permission Methods Tests
# ============================================================================


class TestRBACAPIListPermissions:
    """Tests for RBACAPI.list_permissions() method."""

    def test_list_permissions(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test list_permissions() call."""
        mock_client._get.return_value = {
            "permissions": [
                {
                    "id": "perm-1",
                    "name": "Read Debates",
                    "description": "Can read debates",
                    "resource": "debates",
                    "action": "read",
                },
                {
                    "id": "perm-2",
                    "name": "Write Debates",
                    "description": "Can write debates",
                    "resource": "debates",
                    "action": "write",
                },
            ]
        }

        permissions = rbac_api.list_permissions()

        assert len(permissions) == 2
        assert permissions[0].id == "perm-1"
        assert permissions[0].resource == "debates"
        mock_client._get.assert_called_once_with("/api/v1/rbac/permissions")


class TestRBACAPIListPermissionsAsync:
    """Tests for RBACAPI.list_permissions_async() method."""

    @pytest.mark.asyncio
    async def test_list_permissions_async(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test list_permissions_async() call."""
        mock_client._get_async = AsyncMock(
            return_value={
                "permissions": [
                    {
                        "id": "p-async",
                        "name": "Async Perm",
                        "description": "",
                        "resource": "test",
                        "action": "read",
                    }
                ]
            }
        )

        permissions = await rbac_api.list_permissions_async()

        assert len(permissions) == 1


class TestRBACAPICheckPermission:
    """Tests for RBACAPI.check_permission() method."""

    def test_check_permission_allowed(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test check_permission() when allowed."""
        mock_client._post.return_value = {
            "allowed": True,
            "permission": "debates:read",
        }

        result = rbac_api.check_permission("debates:read")

        assert result.allowed is True
        assert result.permission == "debates:read"
        call_args = mock_client._post.call_args
        assert call_args[0][0] == "/api/v1/rbac/check"

    def test_check_permission_denied(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test check_permission() when denied."""
        mock_client._post.return_value = {
            "allowed": False,
            "permission": "debates:delete",
            "reason": "Role does not have delete permission",
        }

        result = rbac_api.check_permission("debates:delete")

        assert result.allowed is False
        assert result.reason == "Role does not have delete permission"

    def test_check_permission_with_context(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test check_permission() with full context."""
        mock_client._post.return_value = {
            "allowed": True,
            "permission": "documents:write",
            "resource": "documents",
        }

        result = rbac_api.check_permission(
            permission="documents:write",
            user_id="user-123",
            resource="documents",
            resource_id="doc-456",
        )

        call_args = mock_client._post.call_args
        body = call_args[1]["data"]
        assert body["permission"] == "documents:write"
        assert body["user_id"] == "user-123"
        assert body["resource"] == "documents"
        assert body["resource_id"] == "doc-456"


class TestRBACAPICheckPermissionAsync:
    """Tests for RBACAPI.check_permission_async() method."""

    @pytest.mark.asyncio
    async def test_check_permission_async(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test check_permission_async() call."""
        mock_client._post_async = AsyncMock(
            return_value={
                "allowed": True,
                "permission": "async:perm",
            }
        )

        result = await rbac_api.check_permission_async("async:perm")

        assert result.allowed is True


# ============================================================================
# RBACAPI Assignment Methods Tests
# ============================================================================


class TestRBACAPIListAssignments:
    """Tests for RBACAPI.list_assignments() method."""

    def test_list_assignments_basic(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test list_assignments() basic call."""
        mock_client._get.return_value = {
            "assignments": [
                {
                    "id": "assign-1",
                    "user_id": "user-1",
                    "role_id": "role-admin",
                    "role_name": "Admin",
                },
                {
                    "id": "assign-2",
                    "user_id": "user-2",
                    "role_id": "role-user",
                    "role_name": "User",
                },
            ],
            "total": 2,
        }

        assignments, total = rbac_api.list_assignments()

        assert len(assignments) == 2
        assert total == 2
        assert assignments[0].user_id == "user-1"

    def test_list_assignments_with_filters(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test list_assignments() with filters."""
        mock_client._get.return_value = {"assignments": [], "total": 0}

        rbac_api.list_assignments(
            user_id="user-123",
            role_id="role-456",
            tenant_id="tenant-789",
            limit=25,
            offset=10,
        )

        call_args = mock_client._get.call_args
        params = call_args[1]["params"]
        assert params["user_id"] == "user-123"
        assert params["role_id"] == "role-456"
        assert params["tenant_id"] == "tenant-789"
        assert params["limit"] == 25
        assert params["offset"] == 10


class TestRBACAPIListAssignmentsAsync:
    """Tests for RBACAPI.list_assignments_async() method."""

    @pytest.mark.asyncio
    async def test_list_assignments_async(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test list_assignments_async() call."""
        mock_client._get_async = AsyncMock(
            return_value={
                "assignments": [
                    {
                        "id": "a-async",
                        "user_id": "u-async",
                        "role_id": "r-async",
                        "role_name": "Async Role",
                    }
                ],
                "total": 1,
            }
        )

        assignments, total = await rbac_api.list_assignments_async()

        assert len(assignments) == 1


class TestRBACAPIAssignRole:
    """Tests for RBACAPI.assign_role() method."""

    def test_assign_role_basic(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test assign_role() basic call."""
        mock_client._post.return_value = {
            "id": "assign-new",
            "user_id": "user-abc",
            "role_id": "role-xyz",
            "role_name": "Manager",
            "assigned_at": "2024-01-01T00:00:00Z",
        }

        result = rbac_api.assign_role("user-abc", "role-xyz")

        assert result.id == "assign-new"
        assert result.user_id == "user-abc"
        assert result.role_id == "role-xyz"
        call_args = mock_client._post.call_args
        assert call_args[0][0] == "/api/v1/rbac/assignments"
        body = call_args[1]["data"]
        assert body["user_id"] == "user-abc"
        assert body["role_id"] == "role-xyz"

    def test_assign_role_with_tenant(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test assign_role() with tenant."""
        mock_client._post.return_value = {
            "id": "assign-tenant",
            "user_id": "user-1",
            "role_id": "role-1",
            "role_name": "Tenant Admin",
            "tenant_id": "tenant-123",
        }

        result = rbac_api.assign_role("user-1", "role-1", tenant_id="tenant-123")

        assert result.tenant_id == "tenant-123"
        call_args = mock_client._post.call_args
        body = call_args[1]["data"]
        assert body["tenant_id"] == "tenant-123"


class TestRBACAPIAssignRoleAsync:
    """Tests for RBACAPI.assign_role_async() method."""

    @pytest.mark.asyncio
    async def test_assign_role_async(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test assign_role_async() call."""
        mock_client._post_async = AsyncMock(
            return_value={
                "id": "a-async",
                "user_id": "u-async",
                "role_id": "r-async",
                "role_name": "Async",
            }
        )

        result = await rbac_api.assign_role_async("u-async", "r-async")

        assert result.id == "a-async"


class TestRBACAPIRevokeRole:
    """Tests for RBACAPI.revoke_role() method."""

    def test_revoke_role_basic(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test revoke_role() basic call."""
        mock_client._post.return_value = None

        rbac_api.revoke_role("user-123", "role-456")

        call_args = mock_client._post.call_args
        assert call_args[0][0] == "/api/v1/rbac/revoke"
        body = call_args[1]["data"]
        assert body["user_id"] == "user-123"
        assert body["role_id"] == "role-456"

    def test_revoke_role_with_tenant(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test revoke_role() with tenant."""
        mock_client._post.return_value = None

        rbac_api.revoke_role("user-1", "role-1", tenant_id="tenant-789")

        call_args = mock_client._post.call_args
        body = call_args[1]["data"]
        assert body["tenant_id"] == "tenant-789"


class TestRBACAPIRevokeRoleAsync:
    """Tests for RBACAPI.revoke_role_async() method."""

    @pytest.mark.asyncio
    async def test_revoke_role_async(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test revoke_role_async() call."""
        mock_client._post_async = AsyncMock(return_value=None)

        await rbac_api.revoke_role_async("u-async", "r-async")

        mock_client._post_async.assert_called_once()


class TestRBACAPIBulkAssign:
    """Tests for RBACAPI.bulk_assign() method."""

    def test_bulk_assign(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test bulk_assign() call."""
        mock_client._post.return_value = {
            "assignments": [
                {"id": "a1", "user_id": "u1", "role_id": "r1", "role_name": "Role 1"},
                {"id": "a2", "user_id": "u2", "role_id": "r2", "role_name": "Role 2"},
            ]
        }

        assignments_input = [
            {"user_id": "u1", "role_id": "r1"},
            {"user_id": "u2", "role_id": "r2"},
        ]
        result = rbac_api.bulk_assign(assignments_input)

        assert len(result) == 2
        assert result[0].user_id == "u1"
        assert result[1].user_id == "u2"
        call_args = mock_client._post.call_args
        assert call_args[0][0] == "/api/v1/rbac/assignments/bulk"


class TestRBACAPIBulkAssignAsync:
    """Tests for RBACAPI.bulk_assign_async() method."""

    @pytest.mark.asyncio
    async def test_bulk_assign_async(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test bulk_assign_async() call."""
        mock_client._post_async = AsyncMock(
            return_value={
                "assignments": [
                    {
                        "id": "a-async",
                        "user_id": "u-async",
                        "role_id": "r-async",
                        "role_name": "Async",
                    }
                ]
            }
        )

        result = await rbac_api.bulk_assign_async([{"user_id": "u-async", "role_id": "r-async"}])

        assert len(result) == 1


# ============================================================================
# RBACAPI User Permission/Role Methods Tests
# ============================================================================


class TestRBACAPIGetUserPermissions:
    """Tests for RBACAPI.get_user_permissions() method."""

    def test_get_user_permissions_default(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test get_user_permissions() for current user."""
        mock_client._get.return_value = {
            "permissions": ["debates:read", "debates:write", "documents:read"]
        }

        permissions = rbac_api.get_user_permissions()

        assert len(permissions) == 3
        assert "debates:read" in permissions
        mock_client._get.assert_called_once_with("/api/v1/rbac/user-permissions", params={})

    def test_get_user_permissions_specific_user(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test get_user_permissions() for specific user."""
        mock_client._get.return_value = {"permissions": ["admin:*"]}

        permissions = rbac_api.get_user_permissions(user_id="admin-user")

        call_args = mock_client._get.call_args
        params = call_args[1]["params"]
        assert params["user_id"] == "admin-user"

    def test_get_user_permissions_with_tenant(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test get_user_permissions() with tenant."""
        mock_client._get.return_value = {"permissions": []}

        rbac_api.get_user_permissions(user_id="user-1", tenant_id="tenant-1")

        call_args = mock_client._get.call_args
        params = call_args[1]["params"]
        assert params["user_id"] == "user-1"
        assert params["tenant_id"] == "tenant-1"


class TestRBACAPIGetUserPermissionsAsync:
    """Tests for RBACAPI.get_user_permissions_async() method."""

    @pytest.mark.asyncio
    async def test_get_user_permissions_async(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test get_user_permissions_async() call."""
        mock_client._get_async = AsyncMock(return_value={"permissions": ["async:perm"]})

        permissions = await rbac_api.get_user_permissions_async()

        assert "async:perm" in permissions


class TestRBACAPIGetUserRoles:
    """Tests for RBACAPI.get_user_roles() method."""

    def test_get_user_roles_default(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test get_user_roles() for current user."""
        mock_client._get.return_value = {
            "roles": [
                {"id": "role-1", "name": "Admin", "description": "", "permissions": ["*"]},
                {"id": "role-2", "name": "Manager", "description": "", "permissions": ["read"]},
            ]
        }

        roles = rbac_api.get_user_roles()

        assert len(roles) == 2
        assert roles[0].name == "Admin"
        mock_client._get.assert_called_once_with("/api/v1/rbac/user-roles", params={})

    def test_get_user_roles_specific_user(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test get_user_roles() for specific user."""
        mock_client._get.return_value = {"roles": []}

        rbac_api.get_user_roles(user_id="user-123", tenant_id="tenant-456")

        call_args = mock_client._get.call_args
        params = call_args[1]["params"]
        assert params["user_id"] == "user-123"
        assert params["tenant_id"] == "tenant-456"


class TestRBACAPIGetUserRolesAsync:
    """Tests for RBACAPI.get_user_roles_async() method."""

    @pytest.mark.asyncio
    async def test_get_user_roles_async(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test get_user_roles_async() call."""
        mock_client._get_async = AsyncMock(
            return_value={
                "roles": [
                    {"id": "r-async", "name": "Async Role", "description": "", "permissions": []}
                ]
            }
        )

        roles = await rbac_api.get_user_roles_async()

        assert len(roles) == 1


# ============================================================================
# Integration-like Tests
# ============================================================================


class TestRBACAPIIntegration:
    """Integration-like tests for RBACAPI."""

    def test_role_lifecycle(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test full role lifecycle: create -> update -> delete."""
        # Create role
        mock_client._post.return_value = {
            "id": "role-new",
            "name": "New Role",
            "description": "A new role",
            "permissions": ["read"],
        }
        role = rbac_api.create_role("New Role", "A new role", ["read"])
        assert role.id == "role-new"

        # Update role
        mock_client._patch.return_value = {
            "id": "role-new",
            "name": "Updated Role",
            "description": "Updated description",
            "permissions": ["read", "write"],
        }
        updated = rbac_api.update_role(
            "role-new", name="Updated Role", permissions=["read", "write"]
        )
        assert updated.name == "Updated Role"

        # Delete role
        mock_client._delete.return_value = None
        rbac_api.delete_role("role-new")
        mock_client._delete.assert_called_once()

    def test_assignment_workflow(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test assignment workflow: assign -> list -> revoke."""
        # Assign role
        mock_client._post.return_value = {
            "id": "assign-1",
            "user_id": "user-1",
            "role_id": "role-admin",
            "role_name": "Admin",
        }
        assignment = rbac_api.assign_role("user-1", "role-admin")
        assert assignment.role_name == "Admin"

        # List assignments
        mock_client._get.return_value = {
            "assignments": [
                {
                    "id": "assign-1",
                    "user_id": "user-1",
                    "role_id": "role-admin",
                    "role_name": "Admin",
                }
            ],
            "total": 1,
        }
        assignments, total = rbac_api.list_assignments(user_id="user-1")
        assert len(assignments) == 1

        # Revoke role
        mock_client._post.return_value = None
        rbac_api.revoke_role("user-1", "role-admin")

    def test_permission_check_workflow(self, rbac_api: RBACAPI, mock_client: MagicMock):
        """Test permission check workflow."""
        # List all permissions
        mock_client._get.return_value = {
            "permissions": [
                {
                    "id": "perm-debates-read",
                    "name": "Read Debates",
                    "description": "",
                    "resource": "debates",
                    "action": "read",
                },
            ]
        }
        all_perms = rbac_api.list_permissions()
        assert len(all_perms) == 1

        # Check specific permission
        mock_client._post.return_value = {
            "allowed": True,
            "permission": "debates:read",
        }
        check = rbac_api.check_permission("debates:read")
        assert check.allowed is True

        # Get user's effective permissions
        mock_client._get.return_value = {"permissions": ["debates:read", "debates:write"]}
        user_perms = rbac_api.get_user_permissions()
        assert "debates:read" in user_perms
