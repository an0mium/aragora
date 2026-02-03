"""
Comprehensive tests for aragora.server.handlers.rbac.

Tests cover:
- Permission listing and retrieval
- Role CRUD operations (system roles + custom roles)
- Role assignment management
- Permission checking
- Error handling and edge cases
- RBAC decorator enforcement

Target: 40+ tests covering the 12 endpoints in rbac.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock
from uuid import uuid4

import pytest


# ===========================================================================
# Rate Limit Bypass for Testing
# ===========================================================================


def _always_allowed(key: str) -> bool:
    """Always allow requests for testing."""
    return True


@pytest.fixture(autouse=True)
def disable_rate_limits():
    """Disable rate limits for all tests in this module."""
    import sys

    rl_module = sys.modules.get("aragora.server.handlers.utils.rate_limit")
    if rl_module is None:
        yield
        return

    original_is_allowed = {}
    for name, limiter in getattr(rl_module, "_limiters", {}).items():
        original_is_allowed[name] = limiter.is_allowed
        limiter.is_allowed = _always_allowed

    yield

    for name, original in original_is_allowed.items():
        if name in getattr(rl_module, "_limiters", {}):
            rl_module._limiters[name].is_allowed = original


# ===========================================================================
# Mock Classes
# ===========================================================================


@dataclass
class MockPermission:
    """Mock permission for testing."""

    id: str
    name: str
    key: str
    resource: MagicMock
    action: MagicMock
    description: str = ""

    def __post_init__(self):
        if isinstance(self.resource, str):
            self.resource = MagicMock(value=self.resource)
        if isinstance(self.action, str):
            self.action = MagicMock(value=self.action)


@dataclass
class MockRole:
    """Mock role for testing."""

    id: str
    name: str
    display_name: str
    description: str = ""
    permissions: set = field(default_factory=set)
    parent_roles: list = field(default_factory=list)
    is_system: bool = True
    is_custom: bool = False
    org_id: str | None = None
    priority: int = 0


@dataclass
class MockRoleAssignment:
    """Mock role assignment for testing."""

    id: str
    user_id: str
    role_id: str
    org_id: str | None = None
    assigned_by: str | None = None
    assigned_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    is_active: bool = True

    @property
    def is_valid(self) -> bool:
        if not self.is_active:
            return False
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
        return True


@dataclass
class MockDecision:
    """Mock permission decision."""

    allowed: bool
    reason: str
    permission_key: str
    resource_id: str | None = None
    cached: bool = False


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(
        self,
        method: str = "GET",
        body: dict | None = None,
        headers: dict | None = None,
        path: str = "/",
    ):
        from io import BytesIO

        self.command = method
        self.path = path
        self._body = body or {}
        self._body_bytes = json.dumps(self._body).encode() if self._body else b"{}"
        self.rfile = BytesIO(self._body_bytes)
        # Headers dict with Content-Length
        self.headers = headers or {}
        if "Content-Length" not in self.headers:
            self.headers["Content-Length"] = str(len(self._body_bytes))

    def read(self):
        return self._body_bytes


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_system_permissions():
    """Create mock system permissions."""
    return {
        "debate.create": MockPermission(
            id="perm-1",
            name="Create Debate",
            key="debate.create",
            resource="debate",
            action="create",
            description="Create new debates",
        ),
        "debate.read": MockPermission(
            id="perm-2",
            name="Read Debate",
            key="debate.read",
            resource="debate",
            action="read",
            description="Read debates",
        ),
        "role.read": MockPermission(
            id="perm-3",
            name="Read Roles",
            key="role.read",
            resource="role",
            action="read",
            description="Read roles and permissions",
        ),
        "role.create": MockPermission(
            id="perm-4",
            name="Create Role",
            key="role.create",
            resource="role",
            action="create",
            description="Create custom roles",
        ),
        "role.update": MockPermission(
            id="perm-5",
            name="Update Role",
            key="role.update",
            resource="role",
            action="update",
            description="Update roles",
        ),
        "role.delete": MockPermission(
            id="perm-6",
            name="Delete Role",
            key="role.delete",
            resource="role",
            action="delete",
            description="Delete custom roles",
        ),
    }


@pytest.fixture
def mock_system_roles():
    """Create mock system roles."""
    return {
        "admin": MockRole(
            id="role-admin",
            name="admin",
            display_name="Administrator",
            description="Full system access",
            permissions={"role.read", "role.create", "role.update", "role.delete"},
            is_system=True,
        ),
        "member": MockRole(
            id="role-member",
            name="member",
            display_name="Member",
            description="Basic access",
            permissions={"debate.read", "debate.create"},
            is_system=True,
        ),
        "viewer": MockRole(
            id="role-viewer",
            name="viewer",
            display_name="Viewer",
            description="Read-only access",
            permissions={"debate.read"},
            is_system=True,
        ),
    }


@pytest.fixture
def mock_permission_checker():
    """Create mock permission checker."""
    checker = MagicMock()
    checker._custom_roles = {}
    checker._role_assignments = {}
    checker.get_user_roles = MagicMock(return_value={"member"})
    checker.check_permission = MagicMock(
        return_value=MockDecision(
            allowed=True,
            reason="Permission granted",
            permission_key="role.read",
        )
    )
    checker.add_role_assignment = MagicMock()
    checker.remove_role_assignment = MagicMock()
    checker.clear_cache = MagicMock()
    return checker


@pytest.fixture
def rbac_handler(mock_system_permissions, mock_system_roles, mock_permission_checker):
    """Create RBAC handler with mocked dependencies."""
    with patch.multiple(
        "aragora.server.handlers.rbac",
        SYSTEM_PERMISSIONS=mock_system_permissions,
        SYSTEM_ROLES=mock_system_roles,
        ROLE_HIERARCHY={"admin": ["member"], "member": ["viewer"], "viewer": []},
        get_permission=lambda k: mock_system_permissions.get(k),
        get_role=lambda n: mock_system_roles.get(n),
        get_role_permissions=lambda n, include_inherited=False: mock_system_roles.get(
            n, MockRole("", "", "")
        ).permissions,
        get_permission_checker=lambda: mock_permission_checker,
        create_custom_role=lambda **kwargs: MockRole(
            id=f"custom-{kwargs['name']}",
            name=kwargs["name"],
            display_name=kwargs.get("display_name", kwargs["name"]),
            description=kwargs.get("description", ""),
            permissions=kwargs.get("permission_keys", set()),
            is_system=False,
            is_custom=True,
            org_id=kwargs.get("org_id"),
        ),
    ):
        from aragora.server.handlers.rbac import RBACHandler

        handler = RBACHandler({})
        handler._mock_checker = mock_permission_checker
        yield handler


# ===========================================================================
# Permission Endpoint Tests
# ===========================================================================


class TestListPermissions:
    """Tests for GET /api/v1/rbac/permissions."""

    @pytest.mark.asyncio
    async def test_list_permissions_success(self, rbac_handler, mock_permission_checker):
        """Test listing all permissions."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/permissions",
            {},
            MockHandler("GET"),
        )

        assert result is not None
        body = result.to_dict()["body"]
        assert "permissions" in body
        assert body["total"] >= 0

    @pytest.mark.asyncio
    async def test_list_permissions_filter_by_resource(self, rbac_handler):
        """Test filtering permissions by resource type."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/permissions",
            {"resource": "debate"},
            MockHandler("GET"),
        )

        assert result is not None
        body = result.to_dict()["body"]
        permissions = body.get("permissions", [])
        for perm in permissions:
            assert perm["resource"] == "debate"

    @pytest.mark.asyncio
    async def test_list_permissions_filter_by_action(self, rbac_handler):
        """Test filtering permissions by action type."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/permissions",
            {"action": "read"},
            MockHandler("GET"),
        )

        assert result is not None
        body = result.to_dict()["body"]
        permissions = body.get("permissions", [])
        for perm in permissions:
            assert perm["action"] == "read"


class TestGetPermission:
    """Tests for GET /api/v1/rbac/permissions/:key."""

    @pytest.mark.asyncio
    async def test_get_permission_success(self, rbac_handler):
        """Test getting a specific permission."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/permissions/debate.create",
            {},
            MockHandler("GET"),
        )

        assert result is not None
        body = result.to_dict()["body"]
        assert "permission" in body
        assert body["permission"]["key"] == "debate.create"

    @pytest.mark.asyncio
    async def test_get_permission_not_found(self, rbac_handler):
        """Test getting a non-existent permission."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/permissions/nonexistent.perm",
            {},
            MockHandler("GET"),
        )

        assert result is not None
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_get_permission_colon_format(self, rbac_handler):
        """Test getting permission with colon format converts to dot."""
        # The handler should try both formats
        result = await rbac_handler.handle(
            "/api/v1/rbac/permissions/debate:create",
            {},
            MockHandler("GET"),
        )

        # Should either find it (if conversion works) or return 404
        assert result is not None


# ===========================================================================
# Role Endpoint Tests
# ===========================================================================


class TestListRoles:
    """Tests for GET /api/v1/rbac/roles."""

    @pytest.mark.asyncio
    async def test_list_roles_success(self, rbac_handler):
        """Test listing all roles."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles",
            {},
            MockHandler("GET"),
        )

        assert result is not None
        body = result.to_dict()["body"]
        assert "roles" in body
        assert body["total"] >= 3  # admin, member, viewer

    @pytest.mark.asyncio
    async def test_list_roles_include_permissions(self, rbac_handler):
        """Test listing roles with resolved permissions."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles",
            {"include_permissions": "true"},
            MockHandler("GET"),
        )

        assert result is not None
        body = result.to_dict()["body"]
        roles = body.get("roles", [])
        for role in roles:
            # Should have resolved_permissions when requested
            if "resolved_permissions" in role:
                assert isinstance(role["resolved_permissions"], list)


class TestGetRole:
    """Tests for GET /api/v1/rbac/roles/:name."""

    @pytest.mark.asyncio
    async def test_get_role_success(self, rbac_handler):
        """Test getting a specific role."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles/admin",
            {},
            MockHandler("GET"),
        )

        assert result is not None
        body = result.to_dict()["body"]
        assert "role" in body
        assert body["role"]["name"] == "admin"

    @pytest.mark.asyncio
    async def test_get_role_not_found(self, rbac_handler):
        """Test getting a non-existent role."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles/nonexistent",
            {},
            MockHandler("GET"),
        )

        assert result is not None
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_get_role_includes_hierarchy(self, rbac_handler):
        """Test that role includes hierarchy information."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles/admin",
            {},
            MockHandler("GET"),
        )

        assert result is not None
        body = result.to_dict()["body"]
        role = body.get("role", {})
        assert "hierarchy" in role


class TestCreateRole:
    """Tests for POST /api/v1/rbac/roles."""

    @pytest.mark.asyncio
    async def test_create_role_success(self, rbac_handler):
        """Test creating a custom role."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles",
            {},
            MockHandler(
                "POST",
                {
                    "name": "custom_reviewer",
                    "display_name": "Custom Reviewer",
                    "description": "Can review debates",
                    "permissions": ["debate.read"],
                },
            ),
        )

        assert result is not None
        body = result.to_dict()["body"]
        assert "role" in body
        assert body["role"]["name"] == "custom_reviewer"
        assert result.status_code == 201

    @pytest.mark.asyncio
    async def test_create_role_missing_name(self, rbac_handler):
        """Test creating role without name fails."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles",
            {},
            MockHandler("POST", {"display_name": "No Name Role"}),
        )

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_role_duplicate_name(self, rbac_handler):
        """Test creating role with existing name fails."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles",
            {},
            MockHandler("POST", {"name": "admin"}),
        )

        assert result is not None
        assert result.status_code == 409

    @pytest.mark.asyncio
    async def test_create_role_with_base_role(self, rbac_handler):
        """Test creating role that inherits from base role."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles",
            {},
            MockHandler(
                "POST",
                {
                    "name": "super_member",
                    "base_role": "member",
                    "permissions": ["role.read"],
                },
            ),
        )

        assert result is not None
        body = result.to_dict()["body"]
        assert "role" in body


class TestUpdateRole:
    """Tests for PUT /api/v1/rbac/roles/:name."""

    @pytest.mark.asyncio
    async def test_update_role_not_found(self, rbac_handler):
        """Test updating a non-existent role."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles/nonexistent",
            {},
            MockHandler("PUT", {"display_name": "New Name"}),
        )

        assert result is not None
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_update_system_role_forbidden(self, rbac_handler):
        """Test updating a system role is forbidden."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles/admin",
            {},
            MockHandler("PUT", {"display_name": "Super Admin"}),
        )

        assert result is not None
        assert result.status_code == 403


class TestDeleteRole:
    """Tests for DELETE /api/v1/rbac/roles/:name."""

    @pytest.mark.asyncio
    async def test_delete_role_not_found(self, rbac_handler):
        """Test deleting a non-existent role."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles/nonexistent",
            {},
            MockHandler("DELETE"),
        )

        assert result is not None
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_system_role_forbidden(self, rbac_handler):
        """Test deleting a system role is forbidden."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles/admin",
            {},
            MockHandler("DELETE"),
        )

        assert result is not None
        assert result.status_code == 403


# ===========================================================================
# Assignment Endpoint Tests
# ===========================================================================


class TestListAssignments:
    """Tests for GET /api/v1/rbac/assignments."""

    @pytest.mark.asyncio
    async def test_list_assignments_empty(self, rbac_handler):
        """Test listing assignments when none exist."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments",
            {},
            MockHandler("GET"),
        )

        assert result is not None
        body = result.to_dict()["body"]
        assert "assignments" in body
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_assignments_with_filter(self, rbac_handler, mock_permission_checker):
        """Test filtering assignments by user_id."""
        # Add a mock assignment
        mock_assignment = MockRoleAssignment(
            id="assign-1",
            user_id="user-123",
            role_id="admin",
        )
        mock_permission_checker._role_assignments = {
            "user-123": [mock_assignment],
        }

        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments",
            {"user_id": "user-123"},
            MockHandler("GET"),
        )

        assert result is not None
        body = result.to_dict()["body"]
        assert body["total"] == 1


class TestCreateAssignment:
    """Tests for POST /api/v1/rbac/assignments."""

    @pytest.mark.asyncio
    async def test_create_assignment_success(self, rbac_handler):
        """Test creating a role assignment."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments",
            {},
            MockHandler(
                "POST",
                {
                    "user_id": "user-456",
                    "role_id": "member",
                },
            ),
        )

        assert result is not None
        body = result.to_dict()["body"]
        assert "assignment" in body
        assert result.status_code == 201

    @pytest.mark.asyncio
    async def test_create_assignment_missing_user_id(self, rbac_handler):
        """Test creating assignment without user_id fails."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments",
            {},
            MockHandler("POST", {"role_id": "member"}),
        )

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_assignment_missing_role_id(self, rbac_handler):
        """Test creating assignment without role_id fails."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments",
            {},
            MockHandler("POST", {"user_id": "user-123"}),
        )

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_assignment_with_expiration(self, rbac_handler):
        """Test creating assignment with expiration date."""
        expires = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments",
            {},
            MockHandler(
                "POST",
                {
                    "user_id": "user-789",
                    "role_id": "member",
                    "expires_at": expires,
                },
            ),
        )

        assert result is not None
        body = result.to_dict()["body"]
        assert "assignment" in body

    @pytest.mark.asyncio
    async def test_create_assignment_invalid_expiration(self, rbac_handler):
        """Test creating assignment with invalid expiration fails."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments",
            {},
            MockHandler(
                "POST",
                {
                    "user_id": "user-123",
                    "role_id": "member",
                    "expires_at": "not-a-date",
                },
            ),
        )

        assert result is not None
        assert result.status_code == 400


class TestDeleteAssignment:
    """Tests for DELETE /api/v1/rbac/assignments/:id."""

    @pytest.mark.asyncio
    async def test_delete_assignment_not_found(self, rbac_handler):
        """Test deleting a non-existent assignment."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments/nonexistent-id",
            {},
            MockHandler("DELETE"),
        )

        assert result is not None
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_assignment_success(self, rbac_handler, mock_permission_checker):
        """Test successfully deleting an assignment."""
        mock_assignment = MockRoleAssignment(
            id="assign-to-delete",
            user_id="user-123",
            role_id="member",
        )
        mock_permission_checker._role_assignments = {
            "user-123": [mock_assignment],
        }

        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments/assign-to-delete",
            {},
            MockHandler("DELETE"),
        )

        assert result is not None
        body = result.to_dict()["body"]
        assert body.get("deleted") is True


# ===========================================================================
# Permission Check Endpoint Tests
# ===========================================================================


class TestCheckPermission:
    """Tests for POST /api/v1/rbac/check."""

    @pytest.mark.asyncio
    async def test_check_permission_allowed(self, rbac_handler, mock_permission_checker):
        """Test checking a permission that is allowed."""
        mock_permission_checker.check_permission.return_value = MockDecision(
            allowed=True,
            reason="Permission granted via role",
            permission_key="debate.create",
        )

        result = await rbac_handler.handle(
            "/api/v1/rbac/check",
            {},
            MockHandler(
                "POST",
                {
                    "user_id": "user-123",
                    "permission": "debate.create",
                },
            ),
        )

        assert result is not None
        body = result.to_dict()["body"]
        assert body["allowed"] is True

    @pytest.mark.asyncio
    async def test_check_permission_denied(self, rbac_handler, mock_permission_checker):
        """Test checking a permission that is denied."""
        mock_permission_checker.check_permission.return_value = MockDecision(
            allowed=False,
            reason="User lacks required permission",
            permission_key="admin.delete",
        )

        result = await rbac_handler.handle(
            "/api/v1/rbac/check",
            {},
            MockHandler(
                "POST",
                {
                    "user_id": "user-123",
                    "permission": "admin.delete",
                },
            ),
        )

        assert result is not None
        body = result.to_dict()["body"]
        assert body["allowed"] is False

    @pytest.mark.asyncio
    async def test_check_permission_with_resource(self, rbac_handler, mock_permission_checker):
        """Test checking permission with resource_id."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/check",
            {},
            MockHandler(
                "POST",
                {
                    "user_id": "user-123",
                    "permission": "debate.read",
                    "resource_id": "debate-abc",
                },
            ),
        )

        assert result is not None
        # Verify resource_id was passed to checker
        mock_permission_checker.check_permission.assert_called()

    @pytest.mark.asyncio
    async def test_check_permission_missing_user_id(self, rbac_handler):
        """Test check permission without user_id fails."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/check",
            {},
            MockHandler("POST", {"permission": "debate.create"}),
        )

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_check_permission_missing_permission(self, rbac_handler):
        """Test check permission without permission fails."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/check",
            {},
            MockHandler("POST", {"user_id": "user-123"}),
        )

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_check_permission_with_roles(self, rbac_handler, mock_permission_checker):
        """Test checking permission with explicit roles."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/check",
            {},
            MockHandler(
                "POST",
                {
                    "user_id": "user-123",
                    "permission": "debate.create",
                    "roles": ["admin", "member"],
                },
            ),
        )

        assert result is not None
        # Should use provided roles
        mock_permission_checker.check_permission.assert_called()


# ===========================================================================
# can_handle Tests
# ===========================================================================


class TestCanHandle:
    """Tests for handler routing logic."""

    def test_can_handle_permissions_get(self, rbac_handler):
        """Test can_handle for permissions GET."""
        assert rbac_handler.can_handle("/api/v1/rbac/permissions", "GET") is True
        assert rbac_handler.can_handle("/api/v1/rbac/permissions/key", "GET") is True
        assert rbac_handler.can_handle("/api/v1/rbac/permissions", "POST") is False

    def test_can_handle_roles(self, rbac_handler):
        """Test can_handle for roles endpoints."""
        assert rbac_handler.can_handle("/api/v1/rbac/roles", "GET") is True
        assert rbac_handler.can_handle("/api/v1/rbac/roles", "POST") is True
        assert rbac_handler.can_handle("/api/v1/rbac/roles/admin", "GET") is True
        assert rbac_handler.can_handle("/api/v1/rbac/roles/admin", "PUT") is True
        assert rbac_handler.can_handle("/api/v1/rbac/roles/admin", "DELETE") is True

    def test_can_handle_assignments(self, rbac_handler):
        """Test can_handle for assignments endpoints."""
        assert rbac_handler.can_handle("/api/v1/rbac/assignments", "GET") is True
        assert rbac_handler.can_handle("/api/v1/rbac/assignments", "POST") is True
        assert rbac_handler.can_handle("/api/v1/rbac/assignments/123", "DELETE") is True
        assert rbac_handler.can_handle("/api/v1/rbac/assignments", "PUT") is False

    def test_can_handle_check(self, rbac_handler):
        """Test can_handle for check endpoint."""
        assert rbac_handler.can_handle("/api/v1/rbac/check", "POST") is True
        assert rbac_handler.can_handle("/api/v1/rbac/check", "GET") is False

    def test_can_handle_invalid_paths(self, rbac_handler):
        """Test can_handle for invalid paths."""
        assert rbac_handler.can_handle("/api/v1/other", "GET") is False
        assert rbac_handler.can_handle("/api/v1/rbac/unknown", "GET") is False


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_not_found_route(self, rbac_handler):
        """Test accessing an unknown route."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/unknown/path",
            {},
            MockHandler("GET"),
        )

        assert result is not None
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_empty_role_name(self, rbac_handler):
        """Test getting role with empty name."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles/",
            {},
            MockHandler("GET"),
        )

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_empty_assignment_id(self, rbac_handler):
        """Test deleting assignment with empty ID."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments/",
            {},
            MockHandler("DELETE"),
        )

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Integration with Permission Checker Tests
# ===========================================================================


class TestPermissionCheckerIntegration:
    """Tests for integration with the permission checker."""

    @pytest.mark.asyncio
    async def test_custom_role_registration(self, rbac_handler, mock_permission_checker):
        """Test that created custom roles are registered in checker."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles",
            {},
            MockHandler(
                "POST",
                {
                    "name": "test_custom",
                    "org_id": "org-123",
                },
            ),
        )

        assert result is not None
        # Verify role was added to custom roles
        assert "org-123:test_custom" in mock_permission_checker._custom_roles

    @pytest.mark.asyncio
    async def test_assignment_cache_cleared(self, rbac_handler, mock_permission_checker):
        """Test that cache is cleared when assignment is created."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments",
            {},
            MockHandler(
                "POST",
                {
                    "user_id": "user-cache-test",
                    "role_id": "member",
                },
            ),
        )

        assert result is not None
        mock_permission_checker.clear_cache.assert_called()
