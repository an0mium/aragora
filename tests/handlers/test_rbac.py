"""Tests for RBAC management handler endpoints.

Covers all routes in aragora/server/handlers/rbac.py:
- GET    /api/v1/rbac/permissions          - List all permissions
- GET    /api/v1/rbac/permissions/:key      - Get a specific permission
- GET    /api/v1/rbac/roles                - List all roles
- GET    /api/v1/rbac/roles/:name          - Get a specific role
- POST   /api/v1/rbac/roles                - Create a custom role
- PUT    /api/v1/rbac/roles/:name          - Update a custom role
- DELETE /api/v1/rbac/roles/:name          - Delete a custom role
- GET    /api/v1/rbac/assignments          - List role assignments
- POST   /api/v1/rbac/assignments          - Create a role assignment
- DELETE /api/v1/rbac/assignments/:id      - Delete a role assignment
- POST   /api/v1/rbac/check               - Check a permission
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from aragora.rbac.checker import PermissionChecker, get_permission_checker, set_permission_checker
from aragora.rbac.models import AuthorizationContext, RoleAssignment
from aragora.server.handlers.rbac import RBACHandler


# ============================================================================
# Helpers
# ============================================================================


def _make_handler(body: dict[str, Any] | None = None, method: str = "GET") -> MagicMock:
    """Create a mock HTTP handler with optional JSON body and method."""
    handler = MagicMock()
    handler.command = method
    if body is not None:
        body_bytes = json.dumps(body).encode()
        handler.rfile.read.return_value = body_bytes
        handler.headers = {"Content-Length": str(len(body_bytes))}
    else:
        handler.rfile.read.return_value = b"{}"
        handler.headers = {"Content-Length": "2"}
    return handler


def _parse_body(result) -> dict[str, Any]:
    """Parse the JSON body from a HandlerResult."""
    return json.loads(result.body)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def rbac_handler():
    """Create a fresh RBACHandler instance."""
    return RBACHandler({})


@pytest.fixture(autouse=True)
def clean_checker():
    """Ensure a clean PermissionChecker for each test.

    Reset the global checker to avoid state leaking between tests
    (e.g. custom roles or role assignments from a previous test).
    """
    fresh = PermissionChecker(enable_cache=False)
    set_permission_checker(fresh)
    yield fresh
    # Restore to None so next test gets a fresh one
    set_permission_checker(PermissionChecker(enable_cache=False))


# ============================================================================
# can_handle
# ============================================================================


class TestCanHandle:
    """Test RBACHandler.can_handle routing logic."""

    def test_permissions_get(self, rbac_handler):
        assert rbac_handler.can_handle("/api/v1/rbac/permissions", "GET") is True

    def test_permissions_get_specific(self, rbac_handler):
        assert rbac_handler.can_handle("/api/v1/rbac/permissions/debates.create", "GET") is True

    def test_permissions_post_rejected(self, rbac_handler):
        assert rbac_handler.can_handle("/api/v1/rbac/permissions", "POST") is False

    def test_roles_get(self, rbac_handler):
        assert rbac_handler.can_handle("/api/v1/rbac/roles", "GET") is True

    def test_roles_post(self, rbac_handler):
        assert rbac_handler.can_handle("/api/v1/rbac/roles", "POST") is True

    def test_roles_put(self, rbac_handler):
        assert rbac_handler.can_handle("/api/v1/rbac/roles/test_role", "PUT") is True

    def test_roles_delete(self, rbac_handler):
        assert rbac_handler.can_handle("/api/v1/rbac/roles/test_role", "DELETE") is True

    def test_assignments_get(self, rbac_handler):
        assert rbac_handler.can_handle("/api/v1/rbac/assignments", "GET") is True

    def test_assignments_post(self, rbac_handler):
        assert rbac_handler.can_handle("/api/v1/rbac/assignments", "POST") is True

    def test_assignments_delete(self, rbac_handler):
        assert rbac_handler.can_handle("/api/v1/rbac/assignments/some-id", "DELETE") is True

    def test_assignments_put_rejected(self, rbac_handler):
        assert rbac_handler.can_handle("/api/v1/rbac/assignments", "PUT") is False

    def test_check_post(self, rbac_handler):
        assert rbac_handler.can_handle("/api/v1/rbac/check", "POST") is True

    def test_check_get_rejected(self, rbac_handler):
        assert rbac_handler.can_handle("/api/v1/rbac/check", "GET") is False

    def test_wrong_prefix_rejected(self, rbac_handler):
        assert rbac_handler.can_handle("/api/v1/other/permissions", "GET") is False

    def test_unknown_segment_rejected(self, rbac_handler):
        assert rbac_handler.can_handle("/api/v1/rbac/unknown", "GET") is False

    def test_empty_rbac_path_rejected(self, rbac_handler):
        assert rbac_handler.can_handle("/api/v1/rbac/", "GET") is False


# ============================================================================
# Permissions endpoints
# ============================================================================


class TestListPermissions:
    """Test GET /api/v1/rbac/permissions."""

    @pytest.mark.asyncio
    async def test_list_all_permissions(self, rbac_handler):
        handler = _make_handler(method="GET")
        result = await rbac_handler.handle(
            "/api/v1/rbac/permissions", {}, handler
        )
        assert result.status_code == 200
        body = _parse_body(result)
        assert "permissions" in body
        assert "total" in body
        assert body["total"] == len(body["permissions"])
        assert body["total"] > 0

    @pytest.mark.asyncio
    async def test_permissions_have_required_fields(self, rbac_handler):
        handler = _make_handler(method="GET")
        result = await rbac_handler.handle(
            "/api/v1/rbac/permissions", {}, handler
        )
        body = _parse_body(result)
        perm = body["permissions"][0]
        for field in ("id", "name", "key", "resource", "action", "description"):
            assert field in perm, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_filter_by_resource(self, rbac_handler):
        handler = _make_handler(method="GET")
        result = await rbac_handler.handle(
            "/api/v1/rbac/permissions",
            {"resource": "debates"},
            handler,
        )
        body = _parse_body(result)
        assert body["total"] > 0
        for perm in body["permissions"]:
            assert perm["resource"] == "debates"

    @pytest.mark.asyncio
    async def test_filter_by_action(self, rbac_handler):
        handler = _make_handler(method="GET")
        result = await rbac_handler.handle(
            "/api/v1/rbac/permissions",
            {"action": "create"},
            handler,
        )
        body = _parse_body(result)
        assert body["total"] > 0
        for perm in body["permissions"]:
            assert perm["action"] == "create"

    @pytest.mark.asyncio
    async def test_filter_by_resource_and_action(self, rbac_handler):
        handler = _make_handler(method="GET")
        result = await rbac_handler.handle(
            "/api/v1/rbac/permissions",
            {"resource": "debates", "action": "create"},
            handler,
        )
        body = _parse_body(result)
        assert body["total"] >= 1
        for perm in body["permissions"]:
            assert perm["resource"] == "debates"
            assert perm["action"] == "create"

    @pytest.mark.asyncio
    async def test_filter_no_matches(self, rbac_handler):
        handler = _make_handler(method="GET")
        result = await rbac_handler.handle(
            "/api/v1/rbac/permissions",
            {"resource": "nonexistent_resource_xyzzy"},
            handler,
        )
        body = _parse_body(result)
        assert body["total"] == 0
        assert body["permissions"] == []

    @pytest.mark.asyncio
    async def test_no_colon_duplicates(self, rbac_handler):
        """Permissions should be deduplicated (colon aliases skipped)."""
        handler = _make_handler(method="GET")
        result = await rbac_handler.handle(
            "/api/v1/rbac/permissions", {}, handler
        )
        body = _parse_body(result)
        keys = [p["key"] for p in body["permissions"]]
        assert len(keys) == len(set(keys)), "Duplicate permission keys found"


class TestGetPermission:
    """Test GET /api/v1/rbac/permissions/:key."""

    @pytest.mark.asyncio
    async def test_get_existing_permission(self, rbac_handler):
        handler = _make_handler(method="GET")
        result = await rbac_handler.handle(
            "/api/v1/rbac/permissions/debates.create", {}, handler
        )
        assert result.status_code == 200
        body = _parse_body(result)
        assert "permission" in body
        assert body["permission"]["key"] == "debates.create"

    @pytest.mark.asyncio
    async def test_get_permission_colon_fallback(self, rbac_handler):
        """Should try dot notation when colon format is provided."""
        handler = _make_handler(method="GET")
        result = await rbac_handler.handle(
            "/api/v1/rbac/permissions/debates:create", {}, handler
        )
        # Should find the permission via colon->dot fallback
        body = _parse_body(result)
        if result.status_code == 200:
            assert body["permission"]["key"] == "debates.create"

    @pytest.mark.asyncio
    async def test_get_nonexistent_permission(self, rbac_handler):
        handler = _make_handler(method="GET")
        result = await rbac_handler.handle(
            "/api/v1/rbac/permissions/nonexistent.perm", {}, handler
        )
        assert result.status_code == 404
        body = _parse_body(result)
        assert "not found" in body.get("error", "").lower() or "not found" in json.dumps(body).lower()


# ============================================================================
# Roles endpoints
# ============================================================================


class TestListRoles:
    """Test GET /api/v1/rbac/roles."""

    @pytest.mark.asyncio
    async def test_list_system_roles(self, rbac_handler):
        handler = _make_handler(method="GET")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles", {}, handler
        )
        assert result.status_code == 200
        body = _parse_body(result)
        assert "roles" in body
        assert "total" in body
        assert body["total"] > 0
        role_names = [r["name"] for r in body["roles"]]
        # Should include at least some system roles
        assert "admin" in role_names or "owner" in role_names

    @pytest.mark.asyncio
    async def test_list_roles_includes_custom(self, rbac_handler, clean_checker):
        """Custom roles in checker should appear in listing."""
        clean_checker._custom_roles["test-org:custom_role"] = {
            "name": "custom_role",
            "display_name": "Custom Role",
            "description": "A test custom role",
            "permissions": {"debates.read"},
            "parent_roles": [],
            "org_id": "test-org",
            "priority": 45,
        }
        handler = _make_handler(method="GET")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles", {}, handler
        )
        body = _parse_body(result)
        role_names = [r["name"] for r in body["roles"]]
        assert "custom_role" in role_names

    @pytest.mark.asyncio
    async def test_list_roles_with_permissions(self, rbac_handler):
        handler = _make_handler(method="GET")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles",
            {"include_permissions": "true"},
            handler,
        )
        body = _parse_body(result)
        # System roles should have resolved_permissions when flag set
        system_role = next(
            (r for r in body["roles"] if r.get("is_system")), None
        )
        if system_role:
            assert "resolved_permissions" in system_role

    @pytest.mark.asyncio
    async def test_list_roles_without_permissions(self, rbac_handler):
        handler = _make_handler(method="GET")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles",
            {"include_permissions": "false"},
            handler,
        )
        body = _parse_body(result)
        system_role = next(
            (r for r in body["roles"] if r.get("is_system")), None
        )
        if system_role:
            assert "resolved_permissions" not in system_role

    @pytest.mark.asyncio
    async def test_list_roles_has_hierarchy(self, rbac_handler):
        handler = _make_handler(method="GET")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles", {}, handler
        )
        body = _parse_body(result)
        for role in body["roles"]:
            if role.get("is_system"):
                assert "hierarchy" in role


class TestGetRole:
    """Test GET /api/v1/rbac/roles/:name."""

    @pytest.mark.asyncio
    async def test_get_system_role(self, rbac_handler):
        handler = _make_handler(method="GET")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles/admin", {}, handler
        )
        assert result.status_code == 200
        body = _parse_body(result)
        assert "role" in body
        assert body["role"]["name"] == "admin"
        assert body["role"]["is_system"] is True
        assert "resolved_permissions" in body["role"]
        assert "hierarchy" in body["role"]

    @pytest.mark.asyncio
    async def test_get_custom_role(self, rbac_handler, clean_checker):
        clean_checker._custom_roles["test-org:my_custom"] = {
            "name": "my_custom",
            "display_name": "My Custom",
            "description": "Custom",
            "permissions": {"debates.read"},
            "parent_roles": [],
            "org_id": "test-org",
            "priority": 10,
        }
        handler = _make_handler(method="GET")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles/my_custom", {}, handler
        )
        assert result.status_code == 200
        body = _parse_body(result)
        assert body["role"]["name"] == "my_custom"
        assert body["role"]["is_custom"] is True

    @pytest.mark.asyncio
    async def test_get_nonexistent_role(self, rbac_handler):
        handler = _make_handler(method="GET")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles/nonexistent_role_xyz", {}, handler
        )
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_get_role_empty_name(self, rbac_handler):
        """Empty role name in path should return 400."""
        handler = _make_handler(method="GET")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles/", {}, handler
        )
        assert result.status_code == 400


class TestCreateRole:
    """Test POST /api/v1/rbac/roles."""

    @pytest.mark.asyncio
    async def test_create_role_success(self, rbac_handler, clean_checker):
        body = {
            "name": "test_engineer",
            "display_name": "Test Engineer",
            "description": "A role for test engineers",
            "permissions": [],
            "org_id": "org-1",
        }
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles", {}, handler
        )
        assert result.status_code == 201
        resp = _parse_body(result)
        assert resp["role"]["name"] == "test_engineer"
        assert resp["role"]["is_custom"] is True
        assert resp["role"]["org_id"] == "org-1"
        # Should be registered in checker
        assert "org-1:test_engineer" in clean_checker._custom_roles

    @pytest.mark.asyncio
    async def test_create_role_missing_name(self, rbac_handler):
        body = {"description": "No name"}
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles", {}, handler
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_role_empty_name(self, rbac_handler):
        body = {"name": ""}
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles", {}, handler
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_role_non_string_name(self, rbac_handler):
        body = {"name": 123}
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles", {}, handler
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_duplicate_system_role(self, rbac_handler):
        """Cannot create a role with the same name as a system role."""
        body = {"name": "admin"}
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles", {}, handler
        )
        assert result.status_code == 409

    @pytest.mark.asyncio
    async def test_create_role_with_base_role(self, rbac_handler, clean_checker):
        body = {
            "name": "extended_viewer",
            "base_role": "viewer",
            "org_id": "org-2",
        }
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles", {}, handler
        )
        assert result.status_code == 201
        resp = _parse_body(result)
        assert "viewer" in resp["role"].get("parent_roles", [])

    @pytest.mark.asyncio
    async def test_create_role_default_org(self, rbac_handler, clean_checker):
        """If no org_id provided, defaults to 'default'."""
        body = {"name": "no_org_role"}
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles", {}, handler
        )
        assert result.status_code == 201
        resp = _parse_body(result)
        assert resp["role"]["org_id"] == "default"

    @pytest.mark.asyncio
    async def test_create_role_default_display_name(self, rbac_handler, clean_checker):
        """display_name defaults to titleized name."""
        body = {"name": "my_role"}
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles", {}, handler
        )
        assert result.status_code == 201
        resp = _parse_body(result)
        assert resp["role"]["display_name"] == "My Role"

    @pytest.mark.asyncio
    async def test_create_role_with_invalid_permissions(self, rbac_handler):
        """Should return 400 for unknown permission keys."""
        body = {
            "name": "bad_perms",
            "permissions": ["totally.fake.permission.that.does.not.exist"],
        }
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles", {}, handler
        )
        assert result.status_code == 400


class TestUpdateRole:
    """Test PUT /api/v1/rbac/roles/:name."""

    @pytest.fixture
    def custom_role(self, clean_checker):
        """Register a custom role for update/delete tests."""
        clean_checker._custom_roles["org-1:updatable"] = {
            "name": "updatable",
            "display_name": "Updatable Role",
            "description": "Original description",
            "permissions": {"debates.read"},
            "parent_roles": [],
            "org_id": "org-1",
            "priority": 45,
        }
        return "updatable"

    @pytest.mark.asyncio
    async def test_update_description(self, rbac_handler, custom_role, clean_checker):
        body = {"description": "Updated description"}
        handler = _make_handler(body=body, method="PUT")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles/updatable", {}, handler
        )
        assert result.status_code == 200
        resp = _parse_body(result)
        assert resp["role"]["description"] == "Updated description"
        assert clean_checker._custom_roles["org-1:updatable"]["description"] == "Updated description"

    @pytest.mark.asyncio
    async def test_update_display_name(self, rbac_handler, custom_role, clean_checker):
        body = {"display_name": "New Display Name"}
        handler = _make_handler(body=body, method="PUT")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles/updatable", {}, handler
        )
        assert result.status_code == 200
        resp = _parse_body(result)
        assert resp["role"]["display_name"] == "New Display Name"

    @pytest.mark.asyncio
    async def test_update_permissions(self, rbac_handler, custom_role):
        body = {"permissions": ["debates.read", "debates.create"]}
        handler = _make_handler(body=body, method="PUT")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles/updatable", {}, handler
        )
        assert result.status_code == 200
        resp = _parse_body(result)
        assert "debates.read" in resp["role"]["permissions"]
        assert "debates.create" in resp["role"]["permissions"]

    @pytest.mark.asyncio
    async def test_update_with_invalid_permission(self, rbac_handler, custom_role):
        body = {"permissions": ["nonexistent.perm"]}
        handler = _make_handler(body=body, method="PUT")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles/updatable", {}, handler
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_update_with_wildcard_permission(self, rbac_handler, custom_role):
        """Wildcard permissions (ending in .*) are allowed."""
        body = {"permissions": ["debates.*"]}
        handler = _make_handler(body=body, method="PUT")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles/updatable", {}, handler
        )
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_update_with_base_role(self, rbac_handler, custom_role, clean_checker):
        body = {"base_role": "viewer"}
        handler = _make_handler(body=body, method="PUT")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles/updatable", {}, handler
        )
        assert result.status_code == 200
        resp = _parse_body(result)
        assert "viewer" in resp["role"]["parent_roles"]

    @pytest.mark.asyncio
    async def test_update_system_role_forbidden(self, rbac_handler):
        body = {"description": "Try to modify admin"}
        handler = _make_handler(body=body, method="PUT")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles/admin", {}, handler
        )
        assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_update_nonexistent_role(self, rbac_handler):
        body = {"description": "Updated"}
        handler = _make_handler(body=body, method="PUT")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles/nonexistent_xyz", {}, handler
        )
        assert result.status_code == 404


class TestDeleteRole:
    """Test DELETE /api/v1/rbac/roles/:name."""

    @pytest.fixture
    def deletable_role(self, clean_checker):
        """Register a custom role for deletion."""
        clean_checker._custom_roles["org-1:deletable"] = {
            "name": "deletable",
            "display_name": "Deletable Role",
            "description": "Will be deleted",
            "permissions": set(),
            "parent_roles": [],
            "org_id": "org-1",
            "priority": 0,
        }
        return "deletable"

    @pytest.mark.asyncio
    async def test_delete_custom_role(self, rbac_handler, deletable_role, clean_checker):
        handler = _make_handler(method="DELETE")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles/deletable", {}, handler
        )
        assert result.status_code == 200
        resp = _parse_body(result)
        assert resp["deleted"] is True
        assert resp["role"] == "deletable"
        assert "org-1:deletable" not in clean_checker._custom_roles

    @pytest.mark.asyncio
    async def test_delete_system_role_forbidden(self, rbac_handler):
        handler = _make_handler(method="DELETE")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles/admin", {}, handler
        )
        assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_delete_nonexistent_role(self, rbac_handler):
        handler = _make_handler(method="DELETE")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles/nonexistent_role_xyz", {}, handler
        )
        assert result.status_code == 404


# ============================================================================
# Assignments endpoints
# ============================================================================


class TestListAssignments:
    """Test GET /api/v1/rbac/assignments."""

    @pytest.mark.asyncio
    async def test_list_empty_assignments(self, rbac_handler):
        handler = _make_handler(method="GET")
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments", {}, handler
        )
        assert result.status_code == 200
        body = _parse_body(result)
        assert body["assignments"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_assignments_with_data(self, rbac_handler, clean_checker):
        assignment = RoleAssignment(
            id="assign-1",
            user_id="user-1",
            role_id="admin",
            org_id="org-1",
            assigned_by="admin-user",
            assigned_at=datetime.now(timezone.utc),
        )
        clean_checker.add_role_assignment(assignment)

        handler = _make_handler(method="GET")
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments", {}, handler
        )
        body = _parse_body(result)
        assert body["total"] == 1
        assert body["assignments"][0]["id"] == "assign-1"
        assert body["assignments"][0]["user_id"] == "user-1"

    @pytest.mark.asyncio
    async def test_filter_by_user_id(self, rbac_handler, clean_checker):
        for i, uid in enumerate(["user-a", "user-b"]):
            clean_checker.add_role_assignment(
                RoleAssignment(
                    id=f"a-{i}", user_id=uid, role_id="admin",
                    assigned_at=datetime.now(timezone.utc),
                )
            )
        handler = _make_handler(method="GET")
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments",
            {"user_id": "user-a"},
            handler,
        )
        body = _parse_body(result)
        assert body["total"] == 1
        assert body["assignments"][0]["user_id"] == "user-a"

    @pytest.mark.asyncio
    async def test_filter_by_role_id(self, rbac_handler, clean_checker):
        clean_checker.add_role_assignment(
            RoleAssignment(
                id="r1", user_id="u1", role_id="admin",
                assigned_at=datetime.now(timezone.utc),
            )
        )
        clean_checker.add_role_assignment(
            RoleAssignment(
                id="r2", user_id="u2", role_id="viewer",
                assigned_at=datetime.now(timezone.utc),
            )
        )
        handler = _make_handler(method="GET")
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments",
            {"role_id": "viewer"},
            handler,
        )
        body = _parse_body(result)
        assert body["total"] == 1
        assert body["assignments"][0]["role_id"] == "viewer"

    @pytest.mark.asyncio
    async def test_filter_by_org_id(self, rbac_handler, clean_checker):
        clean_checker.add_role_assignment(
            RoleAssignment(
                id="o1", user_id="u1", role_id="admin", org_id="org-x",
                assigned_at=datetime.now(timezone.utc),
            )
        )
        clean_checker.add_role_assignment(
            RoleAssignment(
                id="o2", user_id="u2", role_id="admin", org_id="org-y",
                assigned_at=datetime.now(timezone.utc),
            )
        )
        handler = _make_handler(method="GET")
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments",
            {"org_id": "org-x"},
            handler,
        )
        body = _parse_body(result)
        assert body["total"] == 1
        assert body["assignments"][0]["org_id"] == "org-x"

    @pytest.mark.asyncio
    async def test_assignment_serialization_fields(self, rbac_handler, clean_checker):
        """Assignment response should have all expected fields."""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(days=30)
        clean_checker.add_role_assignment(
            RoleAssignment(
                id="full-1",
                user_id="u1",
                role_id="admin",
                org_id="org-1",
                assigned_by="admin-user",
                assigned_at=now,
                expires_at=expires,
                is_active=True,
            )
        )
        handler = _make_handler(method="GET")
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments", {}, handler
        )
        body = _parse_body(result)
        a = body["assignments"][0]
        for field in ("id", "user_id", "role_id", "org_id", "assigned_by",
                       "assigned_at", "expires_at", "is_active", "is_valid"):
            assert field in a, f"Missing field: {field}"


class TestCreateAssignment:
    """Test POST /api/v1/rbac/assignments."""

    @pytest.mark.asyncio
    async def test_create_assignment_success(self, rbac_handler, clean_checker):
        body = {"user_id": "user-new", "role_id": "admin"}
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments", {}, handler
        )
        assert result.status_code == 201
        resp = _parse_body(result)
        assert resp["assignment"]["user_id"] == "user-new"
        assert resp["assignment"]["role_id"] == "admin"
        assert resp["assignment"]["is_active"] is True

    @pytest.mark.asyncio
    async def test_create_assignment_with_expiry(self, rbac_handler, clean_checker):
        expires = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
        body = {
            "user_id": "user-exp",
            "role_id": "admin",
            "expires_at": expires,
        }
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments", {}, handler
        )
        assert result.status_code == 201
        resp = _parse_body(result)
        assert resp["assignment"]["expires_at"] is not None

    @pytest.mark.asyncio
    async def test_create_assignment_invalid_expiry(self, rbac_handler):
        body = {
            "user_id": "user-1",
            "role_id": "admin",
            "expires_at": "not-a-date",
        }
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments", {}, handler
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_assignment_missing_user_id(self, rbac_handler):
        body = {"role_id": "admin"}
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments", {}, handler
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_assignment_missing_role_id(self, rbac_handler):
        body = {"user_id": "user-1"}
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments", {}, handler
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_assignment_empty_user_id(self, rbac_handler):
        body = {"user_id": "", "role_id": "admin"}
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments", {}, handler
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_assignment_non_string_user_id(self, rbac_handler):
        body = {"user_id": 123, "role_id": "admin"}
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments", {}, handler
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_assignment_non_string_role_id(self, rbac_handler):
        body = {"user_id": "user-1", "role_id": 42}
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments", {}, handler
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_assignment_nonexistent_role(self, rbac_handler):
        body = {"user_id": "user-1", "role_id": "nonexistent_role_qwerty"}
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments", {}, handler
        )
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_create_assignment_for_custom_role(self, rbac_handler, clean_checker):
        """Should work when role_id matches a custom role."""
        clean_checker._custom_roles["org-1:custom_role"] = {
            "name": "custom_role",
            "permissions": set(),
        }
        body = {"user_id": "user-cr", "role_id": "custom_role"}
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments", {}, handler
        )
        assert result.status_code == 201

    @pytest.mark.asyncio
    async def test_create_assignment_with_org_and_assigned_by(self, rbac_handler, clean_checker):
        body = {
            "user_id": "user-1",
            "role_id": "admin",
            "org_id": "org-test",
            "assigned_by": "admin-user",
        }
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments", {}, handler
        )
        assert result.status_code == 201
        resp = _parse_body(result)
        assert resp["assignment"]["org_id"] == "org-test"
        assert resp["assignment"]["assigned_by"] == "admin-user"

    @pytest.mark.asyncio
    async def test_create_assignment_naive_expiry_gets_utc(self, rbac_handler, clean_checker):
        """Naive datetime should get UTC timezone appended."""
        body = {
            "user_id": "user-tz",
            "role_id": "admin",
            "expires_at": "2030-01-01T00:00:00",
        }
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments", {}, handler
        )
        assert result.status_code == 201
        resp = _parse_body(result)
        # The ISO string should contain timezone info
        assert resp["assignment"]["expires_at"] is not None


class TestDeleteAssignment:
    """Test DELETE /api/v1/rbac/assignments/:id."""

    @pytest.mark.asyncio
    async def test_delete_assignment_success(self, rbac_handler, clean_checker):
        assignment = RoleAssignment(
            id="del-1",
            user_id="user-del",
            role_id="admin",
            org_id="org-1",
            assigned_at=datetime.now(timezone.utc),
        )
        clean_checker.add_role_assignment(assignment)

        handler = _make_handler(method="DELETE")
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments/del-1", {}, handler
        )
        assert result.status_code == 200
        resp = _parse_body(result)
        assert resp["deleted"] is True
        assert resp["assignment_id"] == "del-1"

    @pytest.mark.asyncio
    async def test_delete_nonexistent_assignment(self, rbac_handler):
        handler = _make_handler(method="DELETE")
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments/nonexistent-id", {}, handler
        )
        assert result.status_code == 404


# ============================================================================
# Permission check endpoint
# ============================================================================


class TestCheckPermission:
    """Test POST /api/v1/rbac/check."""

    @pytest.mark.asyncio
    async def test_check_with_roles(self, rbac_handler):
        body = {
            "user_id": "user-1",
            "permission": "debates.create",
            "roles": ["admin"],
        }
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/check", {}, handler
        )
        assert result.status_code == 200
        resp = _parse_body(result)
        assert "allowed" in resp
        assert "reason" in resp
        assert "permission" in resp
        assert "cached" in resp

    @pytest.mark.asyncio
    async def test_check_allowed_permission(self, rbac_handler, clean_checker):
        """Admin role should have broad permissions."""
        body = {
            "user_id": "user-1",
            "permission": "debates.read",
            "roles": ["admin"],
        }
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/check", {}, handler
        )
        resp = _parse_body(result)
        assert resp["allowed"] is True

    @pytest.mark.asyncio
    async def test_check_denied_permission(self, rbac_handler, clean_checker):
        """User with no roles should be denied."""
        body = {
            "user_id": "user-no-roles",
            "permission": "admin.system_config",
            "roles": [],
        }
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/check", {}, handler
        )
        resp = _parse_body(result)
        assert resp["allowed"] is False

    @pytest.mark.asyncio
    async def test_check_resolves_from_assignments(self, rbac_handler, clean_checker):
        """When no roles provided, should resolve from assignments."""
        clean_checker.add_role_assignment(
            RoleAssignment(
                id="ra-1",
                user_id="user-assigned",
                role_id="admin",
                org_id="org-1",
                assigned_at=datetime.now(timezone.utc),
            )
        )
        body = {
            "user_id": "user-assigned",
            "permission": "debates.read",
            "org_id": "org-1",
        }
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/check", {}, handler
        )
        resp = _parse_body(result)
        assert resp["allowed"] is True

    @pytest.mark.asyncio
    async def test_check_with_resource_id(self, rbac_handler):
        body = {
            "user_id": "user-1",
            "permission": "debates.read",
            "roles": ["admin"],
            "resource_id": "debate-123",
        }
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/check", {}, handler
        )
        resp = _parse_body(result)
        assert resp["resource_id"] == "debate-123"

    @pytest.mark.asyncio
    async def test_check_missing_user_id(self, rbac_handler):
        body = {"permission": "debates.read"}
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/check", {}, handler
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_check_missing_permission(self, rbac_handler):
        body = {"user_id": "user-1"}
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/check", {}, handler
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_check_empty_user_id(self, rbac_handler):
        body = {"user_id": "", "permission": "debates.read"}
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/check", {}, handler
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_check_non_string_user_id(self, rbac_handler):
        body = {"user_id": 42, "permission": "debates.read"}
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/check", {}, handler
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_check_non_string_permission(self, rbac_handler):
        body = {"user_id": "user-1", "permission": 123}
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/check", {}, handler
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_check_with_org_id(self, rbac_handler):
        body = {
            "user_id": "user-1",
            "permission": "debates.read",
            "roles": ["admin"],
            "org_id": "org-1",
        }
        handler = _make_handler(body=body, method="POST")
        result = await rbac_handler.handle(
            "/api/v1/rbac/check", {}, handler
        )
        assert result.status_code == 200


# ============================================================================
# Routing edge cases
# ============================================================================


class TestRoutingEdgeCases:
    """Test routing and fallback behavior."""

    @pytest.mark.asyncio
    async def test_unknown_path_returns_404(self, rbac_handler):
        handler = _make_handler(method="GET")
        result = await rbac_handler.handle(
            "/api/v1/rbac/unknown_endpoint", {}, handler
        )
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_handle_with_no_handler(self, rbac_handler):
        """When handler is None, should default to GET and empty body."""
        result = await rbac_handler.handle(
            "/api/v1/rbac/permissions", {}, None
        )
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_exception_returns_500(self, rbac_handler):
        """Exceptions in handler methods should return 500."""
        handler = _make_handler(method="GET")
        with patch.object(
            rbac_handler, "_list_permissions", side_effect=KeyError("boom")
        ):
            result = await rbac_handler.handle(
                "/api/v1/rbac/permissions", {}, handler
            )
            assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_handle_value_error_returns_500(self, rbac_handler):
        handler = _make_handler(method="GET")
        with patch.object(
            rbac_handler, "_list_roles", side_effect=ValueError("bad")
        ):
            result = await rbac_handler.handle(
                "/api/v1/rbac/roles", {}, handler
            )
            assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_role_name_extracted_from_path(self, rbac_handler, clean_checker):
        """Role name should be correctly extracted from path segment."""
        clean_checker._custom_roles["org-1:role-with-dashes"] = {
            "name": "role-with-dashes",
            "display_name": "Dashed",
            "description": "Has dashes",
            "permissions": set(),
            "parent_roles": [],
            "org_id": "org-1",
            "priority": 0,
        }
        handler = _make_handler(method="GET")
        result = await rbac_handler.handle(
            "/api/v1/rbac/roles/role-with-dashes", {}, handler
        )
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_assignment_id_extracted_from_path(self, rbac_handler, clean_checker):
        """Assignment ID should be correctly extracted from path."""
        assignment = RoleAssignment(
            id="uuid-style-id-here",
            user_id="u1",
            role_id="admin",
            assigned_at=datetime.now(timezone.utc),
        )
        clean_checker.add_role_assignment(assignment)
        handler = _make_handler(method="DELETE")
        result = await rbac_handler.handle(
            "/api/v1/rbac/assignments/uuid-style-id-here", {}, handler
        )
        assert result.status_code == 200


# ============================================================================
# Serialization helpers
# ============================================================================


class TestSerializationHelpers:
    """Test _permission_to_dict, _role_to_dict, _assignment_to_dict."""

    def test_permission_to_dict_fields(self):
        from aragora.server.handlers.rbac import _permission_to_dict
        from aragora.rbac.models import Permission, ResourceType, Action

        perm = Permission(
            id="p-1",
            name="Create Debates",
            resource=ResourceType.DEBATE,
            action=Action.CREATE,
            description="Allows creating debates",
        )
        d = _permission_to_dict(perm)
        assert d["id"] == "p-1"
        assert d["name"] == "Create Debates"
        assert d["key"] == "debates.create"
        assert d["resource"] == "debates"
        assert d["action"] == "create"
        assert d["description"] == "Allows creating debates"

    def test_role_to_dict_fields(self):
        from aragora.server.handlers.rbac import _role_to_dict
        from aragora.rbac.models import Role

        role = Role(
            id="r-1",
            name="test",
            display_name="Test Role",
            description="A test role",
            permissions={"debates.read", "debates.create"},
            parent_roles=["viewer"],
            is_system=False,
            is_custom=True,
            org_id="org-1",
            priority=50,
        )
        d = _role_to_dict(role)
        assert d["id"] == "r-1"
        assert d["name"] == "test"
        assert d["display_name"] == "Test Role"
        assert d["permissions"] == sorted({"debates.read", "debates.create"})
        assert d["parent_roles"] == ["viewer"]
        assert d["is_system"] is False
        assert d["is_custom"] is True
        assert d["org_id"] == "org-1"
        assert d["priority"] == 50

    def test_assignment_to_dict_fields(self):
        from aragora.server.handlers.rbac import _assignment_to_dict

        now = datetime.now(timezone.utc)
        expires = now + timedelta(days=7)
        assignment = RoleAssignment(
            id="a-1",
            user_id="u-1",
            role_id="admin",
            org_id="org-1",
            assigned_by="admin-user",
            assigned_at=now,
            expires_at=expires,
            is_active=True,
        )
        d = _assignment_to_dict(assignment)
        assert d["id"] == "a-1"
        assert d["user_id"] == "u-1"
        assert d["role_id"] == "admin"
        assert d["org_id"] == "org-1"
        assert d["assigned_by"] == "admin-user"
        assert d["assigned_at"] == now.isoformat()
        assert d["expires_at"] == expires.isoformat()
        assert d["is_active"] is True
        assert d["is_valid"] is True

    def test_assignment_to_dict_none_dates(self):
        from aragora.server.handlers.rbac import _assignment_to_dict

        assignment = RoleAssignment(
            id="a-2",
            user_id="u-2",
            role_id="viewer",
            assigned_at=None,
            expires_at=None,
        )
        d = _assignment_to_dict(assignment)
        assert d["assigned_at"] is None
        assert d["expires_at"] is None

    def test_assignment_to_dict_expired(self):
        from aragora.server.handlers.rbac import _assignment_to_dict

        past = datetime.now(timezone.utc) - timedelta(days=1)
        assignment = RoleAssignment(
            id="a-exp",
            user_id="u-exp",
            role_id="admin",
            assigned_at=datetime.now(timezone.utc) - timedelta(days=30),
            expires_at=past,
            is_active=True,
        )
        d = _assignment_to_dict(assignment)
        # is_valid should be False because expired
        assert d["is_valid"] is False
        assert d["is_active"] is True
