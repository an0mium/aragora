"""Tests for GovernanceOperationsMixin (aragora/server/handlers/knowledge_base/mound/governance.py).

Covers all routes and behavior of the governance mixin:
- POST /api/v1/knowledge/mound/governance/roles - Create a role
- GET  /api/v1/knowledge/mound/governance/roles - List roles (routed)
- POST /api/v1/knowledge/mound/governance/roles/assign - Assign role to user
- POST /api/v1/knowledge/mound/governance/roles/revoke - Revoke role from user
- GET  /api/v1/knowledge/mound/governance/permissions/:user_id - Get user permissions
- POST /api/v1/knowledge/mound/governance/permissions/check - Check specific permission
- GET  /api/v1/knowledge/mound/governance/audit - Query audit trail
- GET  /api/v1/knowledge/mound/governance/audit/user/:user_id - Get user activity
- GET  /api/v1/knowledge/mound/governance/stats - Get governance stats
- Error cases: missing mound, invalid permissions, missing fields, server errors
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.knowledge.mound.ops.governance import AuditAction, Permission
from aragora.server.handlers.knowledge_base.mound.governance import (
    GovernanceOperationsMixin,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return -1
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


def _run(coro):
    """Run an async coroutine synchronously for testing."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Mock dataclasses for mound return values
# ---------------------------------------------------------------------------


@dataclass
class MockRole:
    id: str = "role-001"
    name: str = "Custom Editor"
    description: str = "Can read and edit items"
    permissions: set = field(default_factory=lambda: {Permission.READ, Permission.UPDATE})
    workspace_id: str | None = None
    created_at: str = "2025-01-15T10:00:00"
    created_by: str | None = "admin-user"
    is_builtin: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "permissions": [p.value if hasattr(p, "value") else p for p in self.permissions],
            "workspace_id": self.workspace_id,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "is_builtin": self.is_builtin,
        }


@dataclass
class MockRoleAssignment:
    id: str = "assign-001"
    user_id: str = "user-123"
    role_id: str = "role-001"
    workspace_id: str | None = None
    assigned_at: str = "2025-01-15T10:00:00"
    assigned_by: str | None = "admin-user"
    expires_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "role_id": self.role_id,
            "workspace_id": self.workspace_id,
            "assigned_at": self.assigned_at,
            "assigned_by": self.assigned_by,
            "expires_at": self.expires_at,
        }


@dataclass
class MockAuditEntry:
    id: str = "audit-001"
    actor_id: str = "user-123"
    action: str = "item.create"
    resource_id: str = "node-001"
    workspace_id: str | None = None
    timestamp: str = "2025-01-15T10:00:00"
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "actor_id": self.actor_id,
            "action": self.action,
            "resource_id": self.resource_id,
            "workspace_id": self.workspace_id,
            "timestamp": self.timestamp,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Concrete test class combining the mixin with stubs
# ---------------------------------------------------------------------------


class GovernanceTestHandler(GovernanceOperationsMixin):
    """Concrete handler for testing the governance mixin."""

    def __init__(self, mound=None):
        self._mound = mound

    def _get_mound(self):
        return self._mound


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound with governance methods."""
    mound = MagicMock()
    mound.create_role = AsyncMock(return_value=MockRole())
    mound.assign_role = AsyncMock(return_value=MockRoleAssignment())
    mound.revoke_role = AsyncMock(return_value=True)
    mound.get_user_permissions = AsyncMock(return_value={Permission.READ, Permission.CREATE})
    mound.check_permission = AsyncMock(return_value=True)
    mound.query_audit = AsyncMock(
        return_value=[
            MockAuditEntry(id="audit-001", action="item.create"),
            MockAuditEntry(id="audit-002", action="role.assign"),
        ]
    )
    mound.get_user_activity = AsyncMock(
        return_value={
            "user_id": "user-123",
            "total_actions": 42,
            "actions_by_type": {"item.create": 20, "item.read": 15, "share.grant": 7},
            "period_days": 30,
        }
    )
    mound.get_governance_stats = MagicMock(
        return_value={
            "total_roles": 5,
            "total_assignments": 12,
            "total_audit_entries": 150,
            "active_users": 8,
        }
    )
    return mound


@pytest.fixture
def handler(mock_mound):
    """Create a GovernanceTestHandler with a mocked mound."""
    return GovernanceTestHandler(mound=mock_mound)


@pytest.fixture
def handler_no_mound():
    """Create a GovernanceTestHandler with no mound (None)."""
    return GovernanceTestHandler(mound=None)


# ============================================================================
# Tests: create_role
# ============================================================================


class TestCreateRole:
    """Test create_role (POST /api/knowledge/mound/governance/roles)."""

    def test_create_role_success(self, handler, mock_mound):
        """Successfully creating a role returns success with role data."""
        result = _run(
            handler.create_role(
                name="Custom Editor",
                permissions=["read", "update"],
                description="Can read and edit items",
                workspace_id="ws-1",
                created_by="admin-user",
            )
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert "role" in body
        assert body["role"]["name"] == "Custom Editor"

    def test_create_role_mound_called_with_correct_args(self, handler, mock_mound):
        """Mound.create_role is called with Permission enums."""
        _run(
            handler.create_role(
                name="Test Role",
                permissions=["read", "create"],
                description="desc",
                workspace_id="ws-1",
                created_by="admin",
            )
        )
        mock_mound.create_role.assert_called_once()
        call_kwargs = mock_mound.create_role.call_args.kwargs
        assert call_kwargs["name"] == "Test Role"
        assert call_kwargs["description"] == "desc"
        assert call_kwargs["workspace_id"] == "ws-1"
        assert call_kwargs["created_by"] == "admin"
        # permissions should be a set of Permission enums
        assert isinstance(call_kwargs["permissions"], set)
        assert Permission.READ in call_kwargs["permissions"]
        assert Permission.CREATE in call_kwargs["permissions"]

    def test_create_role_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = _run(handler_no_mound.create_role(name="Test", permissions=["read"]))
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    def test_create_role_empty_name_returns_400(self, handler):
        """Empty name returns 400."""
        result = _run(handler.create_role(name="", permissions=["read"]))
        assert _status(result) == 400
        body = _body(result)
        assert "name" in body["error"].lower()

    def test_create_role_empty_permissions_returns_400(self, handler):
        """Empty permissions list returns 400."""
        result = _run(handler.create_role(name="Test", permissions=[]))
        assert _status(result) == 400
        body = _body(result)
        assert "permissions" in body["error"].lower()

    def test_create_role_invalid_permission_returns_400(self, handler):
        """Invalid permission string returns 400 with valid permissions list."""
        result = _run(handler.create_role(name="Test", permissions=["read", "invalid_perm"]))
        assert _status(result) == 400
        body = _body(result)
        assert "invalid permission" in body["error"].lower()

    def test_create_role_mound_error_returns_500(self, handler, mock_mound):
        """Server error from mound returns 500."""
        mock_mound.create_role = AsyncMock(side_effect=OSError("db fail"))
        result = _run(handler.create_role(name="Test", permissions=["read"]))
        assert _status(result) == 500

    def test_create_role_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.create_role = AsyncMock(side_effect=ValueError("bad data"))
        result = _run(handler.create_role(name="Test", permissions=["read"]))
        assert _status(result) == 500

    def test_create_role_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.create_role = AsyncMock(side_effect=KeyError("missing"))
        result = _run(handler.create_role(name="Test", permissions=["read"]))
        assert _status(result) == 500

    def test_create_role_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.create_role = AsyncMock(side_effect=TypeError("wrong type"))
        result = _run(handler.create_role(name="Test", permissions=["read"]))
        assert _status(result) == 500

    def test_create_role_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        mock_mound.create_role = AsyncMock(side_effect=AttributeError("missing attr"))
        result = _run(handler.create_role(name="Test", permissions=["read"]))
        assert _status(result) == 500

    def test_create_role_with_default_optional_params(self, handler, mock_mound):
        """Optional params default correctly."""
        result = _run(handler.create_role(name="Test", permissions=["read"]))
        assert _status(result) == 200
        call_kwargs = mock_mound.create_role.call_args.kwargs
        assert call_kwargs["description"] == ""
        assert call_kwargs["workspace_id"] is None
        assert call_kwargs["created_by"] is None

    def test_create_role_multiple_permissions(self, handler, mock_mound):
        """Multiple valid permissions are all converted to enums."""
        result = _run(
            handler.create_role(
                name="Full Editor",
                permissions=["read", "create", "update", "delete"],
            )
        )
        assert _status(result) == 200
        call_kwargs = mock_mound.create_role.call_args.kwargs
        assert len(call_kwargs["permissions"]) == 4
        assert Permission.READ in call_kwargs["permissions"]
        assert Permission.DELETE in call_kwargs["permissions"]

    def test_create_role_single_permission(self, handler, mock_mound):
        """Single permission is wrapped in a set."""
        result = _run(handler.create_role(name="Viewer", permissions=["read"]))
        assert _status(result) == 200
        call_kwargs = mock_mound.create_role.call_args.kwargs
        assert call_kwargs["permissions"] == {Permission.READ}

    def test_create_role_admin_permission(self, handler, mock_mound):
        """Admin permission value is accepted."""
        result = _run(handler.create_role(name="Admin", permissions=["admin"]))
        assert _status(result) == 200
        call_kwargs = mock_mound.create_role.call_args.kwargs
        assert Permission.ADMIN in call_kwargs["permissions"]


# ============================================================================
# Tests: assign_role
# ============================================================================


class TestAssignRole:
    """Test assign_role (POST /api/knowledge/mound/governance/roles/assign)."""

    def test_assign_role_success(self, handler, mock_mound):
        """Successfully assigning a role returns assignment data."""
        result = _run(
            handler.assign_role(
                user_id="user-123",
                role_id="role-001",
                workspace_id="ws-1",
                assigned_by="admin",
            )
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert "assignment" in body
        assert body["assignment"]["user_id"] == "user-123"
        assert body["assignment"]["role_id"] == "role-001"

    def test_assign_role_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = _run(handler_no_mound.assign_role(user_id="user-1", role_id="role-1"))
        assert _status(result) == 503

    def test_assign_role_missing_user_id_returns_400(self, handler):
        """Empty user_id returns 400."""
        result = _run(handler.assign_role(user_id="", role_id="role-1"))
        assert _status(result) == 400
        body = _body(result)
        assert "user_id" in body["error"].lower()

    def test_assign_role_missing_role_id_returns_400(self, handler):
        """Empty role_id returns 400."""
        result = _run(handler.assign_role(user_id="user-1", role_id=""))
        assert _status(result) == 400

    def test_assign_role_both_missing_returns_400(self, handler):
        """Both empty returns 400."""
        result = _run(handler.assign_role(user_id="", role_id=""))
        assert _status(result) == 400

    def test_assign_role_value_error_returns_404(self, handler, mock_mound):
        """ValueError (e.g., role not found) returns 404."""
        mock_mound.assign_role = AsyncMock(side_effect=ValueError("Role not found"))
        result = _run(handler.assign_role(user_id="user-1", role_id="nonexistent"))
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body["error"].lower()

    def test_assign_role_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.assign_role = AsyncMock(side_effect=KeyError("missing"))
        result = _run(handler.assign_role(user_id="user-1", role_id="role-1"))
        assert _status(result) == 500

    def test_assign_role_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.assign_role = AsyncMock(side_effect=OSError("disk full"))
        result = _run(handler.assign_role(user_id="user-1", role_id="role-1"))
        assert _status(result) == 500

    def test_assign_role_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.assign_role = AsyncMock(side_effect=TypeError("wrong type"))
        result = _run(handler.assign_role(user_id="user-1", role_id="role-1"))
        assert _status(result) == 500

    def test_assign_role_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        mock_mound.assign_role = AsyncMock(side_effect=AttributeError("attr"))
        result = _run(handler.assign_role(user_id="user-1", role_id="role-1"))
        assert _status(result) == 500

    def test_assign_role_default_optional_params(self, handler, mock_mound):
        """Optional params default correctly."""
        _run(handler.assign_role(user_id="user-1", role_id="role-1"))
        call_kwargs = mock_mound.assign_role.call_args.kwargs
        assert call_kwargs["workspace_id"] is None
        assert call_kwargs["assigned_by"] is None

    def test_assign_role_with_workspace(self, handler, mock_mound):
        """workspace_id is forwarded to mound."""
        _run(
            handler.assign_role(
                user_id="user-1",
                role_id="role-1",
                workspace_id="ws-test",
            )
        )
        call_kwargs = mock_mound.assign_role.call_args.kwargs
        assert call_kwargs["workspace_id"] == "ws-test"

    def test_assign_role_with_assigned_by(self, handler, mock_mound):
        """assigned_by is forwarded to mound."""
        _run(
            handler.assign_role(
                user_id="user-1",
                role_id="role-1",
                assigned_by="admin-user",
            )
        )
        call_kwargs = mock_mound.assign_role.call_args.kwargs
        assert call_kwargs["assigned_by"] == "admin-user"


# ============================================================================
# Tests: revoke_role
# ============================================================================


class TestRevokeRole:
    """Test revoke_role (POST /api/knowledge/mound/governance/roles/revoke)."""

    def test_revoke_role_success(self, handler, mock_mound):
        """Successful revoke returns success message."""
        result = _run(
            handler.revoke_role(user_id="user-123", role_id="role-001", workspace_id="ws-1")
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert "revoked" in body["message"].lower()
        assert "role-001" in body["message"]
        assert "user-123" in body["message"]

    def test_revoke_role_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = _run(handler_no_mound.revoke_role(user_id="user-1", role_id="role-1"))
        assert _status(result) == 503

    def test_revoke_role_missing_user_id_returns_400(self, handler):
        """Empty user_id returns 400."""
        result = _run(handler.revoke_role(user_id="", role_id="role-1"))
        assert _status(result) == 400

    def test_revoke_role_missing_role_id_returns_400(self, handler):
        """Empty role_id returns 400."""
        result = _run(handler.revoke_role(user_id="user-1", role_id=""))
        assert _status(result) == 400

    def test_revoke_role_not_found_returns_404(self, handler, mock_mound):
        """When mound.revoke_role returns False, returns 404."""
        mock_mound.revoke_role = AsyncMock(return_value=False)
        result = _run(handler.revoke_role(user_id="user-1", role_id="role-1"))
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body["error"].lower()

    def test_revoke_role_server_error_returns_500(self, handler, mock_mound):
        """Server error returns 500."""
        mock_mound.revoke_role = AsyncMock(side_effect=OSError("db error"))
        result = _run(handler.revoke_role(user_id="user-1", role_id="role-1"))
        assert _status(result) == 500

    def test_revoke_role_value_error_returns_500(self, handler, mock_mound):
        """ValueError returns 500."""
        mock_mound.revoke_role = AsyncMock(side_effect=ValueError("bad"))
        result = _run(handler.revoke_role(user_id="user-1", role_id="role-1"))
        assert _status(result) == 500

    def test_revoke_role_key_error_returns_500(self, handler, mock_mound):
        """KeyError returns 500."""
        mock_mound.revoke_role = AsyncMock(side_effect=KeyError("missing"))
        result = _run(handler.revoke_role(user_id="user-1", role_id="role-1"))
        assert _status(result) == 500

    def test_revoke_role_type_error_returns_500(self, handler, mock_mound):
        """TypeError returns 500."""
        mock_mound.revoke_role = AsyncMock(side_effect=TypeError("bad type"))
        result = _run(handler.revoke_role(user_id="user-1", role_id="role-1"))
        assert _status(result) == 500

    def test_revoke_role_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError returns 500."""
        mock_mound.revoke_role = AsyncMock(side_effect=AttributeError("attr"))
        result = _run(handler.revoke_role(user_id="user-1", role_id="role-1"))
        assert _status(result) == 500

    def test_revoke_role_default_workspace(self, handler, mock_mound):
        """Default workspace_id is None."""
        _run(handler.revoke_role(user_id="user-1", role_id="role-1"))
        call_kwargs = mock_mound.revoke_role.call_args.kwargs
        assert call_kwargs["workspace_id"] is None

    def test_revoke_role_with_workspace(self, handler, mock_mound):
        """workspace_id is forwarded to mound."""
        _run(handler.revoke_role(user_id="user-1", role_id="role-1", workspace_id="ws-prod"))
        call_kwargs = mock_mound.revoke_role.call_args.kwargs
        assert call_kwargs["workspace_id"] == "ws-prod"


# ============================================================================
# Tests: get_user_permissions
# ============================================================================


class TestGetUserPermissions:
    """Test get_user_permissions (GET /api/knowledge/mound/governance/permissions/:user_id)."""

    def test_get_permissions_success(self, handler, mock_mound):
        """Successfully getting permissions returns user_id and permission list."""
        result = _run(handler.get_user_permissions(user_id="user-123", workspace_id="ws-1"))
        assert _status(result) == 200
        body = _body(result)
        assert body["user_id"] == "user-123"
        assert body["workspace_id"] == "ws-1"
        assert isinstance(body["permissions"], list)
        assert len(body["permissions"]) == 2

    def test_get_permissions_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = _run(handler_no_mound.get_user_permissions(user_id="user-1"))
        assert _status(result) == 503

    def test_get_permissions_empty_user_id_returns_400(self, handler):
        """Empty user_id returns 400."""
        result = _run(handler.get_user_permissions(user_id=""))
        assert _status(result) == 400
        body = _body(result)
        assert "user_id" in body["error"].lower()

    def test_get_permissions_default_workspace_none(self, handler, mock_mound):
        """Default workspace_id is None."""
        result = _run(handler.get_user_permissions(user_id="user-1"))
        body = _body(result)
        assert body["workspace_id"] is None
        call_kwargs = mock_mound.get_user_permissions.call_args.kwargs
        assert call_kwargs["workspace_id"] is None

    def test_get_permissions_server_error_returns_500(self, handler, mock_mound):
        """Server error returns 500."""
        mock_mound.get_user_permissions = AsyncMock(side_effect=OSError("db fail"))
        result = _run(handler.get_user_permissions(user_id="user-1"))
        assert _status(result) == 500

    def test_get_permissions_value_error_returns_500(self, handler, mock_mound):
        """ValueError returns 500."""
        mock_mound.get_user_permissions = AsyncMock(side_effect=ValueError("bad"))
        result = _run(handler.get_user_permissions(user_id="user-1"))
        assert _status(result) == 500

    def test_get_permissions_key_error_returns_500(self, handler, mock_mound):
        """KeyError returns 500."""
        mock_mound.get_user_permissions = AsyncMock(side_effect=KeyError("missing"))
        result = _run(handler.get_user_permissions(user_id="user-1"))
        assert _status(result) == 500

    def test_get_permissions_permissions_are_string_values(self, handler, mock_mound):
        """Permissions are returned as string values (enum .value)."""
        result = _run(handler.get_user_permissions(user_id="user-1"))
        body = _body(result)
        for perm in body["permissions"]:
            assert isinstance(perm, str)

    def test_get_permissions_empty_set(self, handler, mock_mound):
        """Empty permission set returns empty list."""
        mock_mound.get_user_permissions = AsyncMock(return_value=set())
        result = _run(handler.get_user_permissions(user_id="user-1"))
        body = _body(result)
        assert body["permissions"] == []

    def test_get_permissions_with_workspace(self, handler, mock_mound):
        """workspace_id is forwarded to mound."""
        _run(handler.get_user_permissions(user_id="user-1", workspace_id="ws-test"))
        call_kwargs = mock_mound.get_user_permissions.call_args.kwargs
        assert call_kwargs["workspace_id"] == "ws-test"

    def test_get_permissions_contains_known_values(self, handler, mock_mound):
        """Returned permissions contain known Permission enum values."""
        result = _run(handler.get_user_permissions(user_id="user-1"))
        body = _body(result)
        assert "read" in body["permissions"]
        assert "create" in body["permissions"]


# ============================================================================
# Tests: check_permission
# ============================================================================


class TestCheckPermission:
    """Test check_permission (POST /api/knowledge/mound/governance/permissions/check)."""

    def test_check_permission_has_permission(self, handler, mock_mound):
        """User with permission returns has_permission=True."""
        result = _run(
            handler.check_permission(
                user_id="user-123",
                permission="read",
                workspace_id="ws-1",
            )
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["user_id"] == "user-123"
        assert body["permission"] == "read"
        assert body["workspace_id"] == "ws-1"
        assert body["has_permission"] is True

    def test_check_permission_denied(self, handler, mock_mound):
        """User without permission returns has_permission=False."""
        mock_mound.check_permission = AsyncMock(return_value=False)
        result = _run(handler.check_permission(user_id="user-1", permission="delete"))
        assert _status(result) == 200
        body = _body(result)
        assert body["has_permission"] is False

    def test_check_permission_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = _run(handler_no_mound.check_permission(user_id="u1", permission="read"))
        assert _status(result) == 503

    def test_check_permission_missing_user_id_returns_400(self, handler):
        """Empty user_id returns 400."""
        result = _run(handler.check_permission(user_id="", permission="read"))
        assert _status(result) == 400

    def test_check_permission_missing_permission_returns_400(self, handler):
        """Empty permission returns 400."""
        result = _run(handler.check_permission(user_id="user-1", permission=""))
        assert _status(result) == 400

    def test_check_permission_invalid_permission_returns_400(self, handler):
        """Invalid permission string returns 400 with valid options."""
        result = _run(handler.check_permission(user_id="user-1", permission="not_a_real_perm"))
        assert _status(result) == 400
        body = _body(result)
        assert "invalid permission" in body["error"].lower()
        assert "valid permissions" in body["error"].lower()

    def test_check_permission_server_error_returns_500(self, handler, mock_mound):
        """Server error returns 500."""
        mock_mound.check_permission = AsyncMock(side_effect=OSError("db fail"))
        result = _run(handler.check_permission(user_id="user-1", permission="read"))
        assert _status(result) == 500

    def test_check_permission_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.check_permission = AsyncMock(side_effect=ValueError("bad"))
        result = _run(handler.check_permission(user_id="user-1", permission="read"))
        assert _status(result) == 500

    def test_check_permission_default_workspace_none(self, handler, mock_mound):
        """Default workspace_id is None in response."""
        result = _run(handler.check_permission(user_id="user-1", permission="read"))
        body = _body(result)
        assert body["workspace_id"] is None

    def test_check_permission_mound_called_with_enum(self, handler, mock_mound):
        """Permission string is converted to enum before passing to mound."""
        _run(handler.check_permission(user_id="user-1", permission="read"))
        call_kwargs = mock_mound.check_permission.call_args.kwargs
        assert call_kwargs["permission"] == Permission.READ

    def test_check_permission_with_workspace(self, handler, mock_mound):
        """workspace_id is forwarded to mound."""
        _run(handler.check_permission(user_id="user-1", permission="read", workspace_id="ws-prod"))
        call_kwargs = mock_mound.check_permission.call_args.kwargs
        assert call_kwargs["workspace_id"] == "ws-prod"

    def test_check_permission_all_valid_permissions(self, handler, mock_mound):
        """Every valid permission string is accepted."""
        for perm in Permission:
            mock_mound.check_permission = AsyncMock(return_value=True)
            result = _run(handler.check_permission(user_id="user-1", permission=perm.value))
            assert _status(result) == 200, f"Permission {perm.value} should be valid"


# ============================================================================
# Tests: query_audit_trail
# ============================================================================


class TestQueryAuditTrail:
    """Test query_audit_trail (GET /api/knowledge/mound/governance/audit)."""

    def test_query_audit_success(self, handler, mock_mound):
        """Successfully querying audit trail returns entries."""
        result = _run(handler.query_audit_trail())
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 2
        assert len(body["entries"]) == 2
        assert body["entries"][0]["id"] == "audit-001"

    def test_query_audit_with_filters(self, handler, mock_mound):
        """Query params are included in response filters."""
        result = _run(
            handler.query_audit_trail(
                actor_id="user-123",
                action="item.create",
                workspace_id="ws-1",
                limit=50,
            )
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["filters"]["actor_id"] == "user-123"
        assert body["filters"]["action"] == "item.create"
        assert body["filters"]["workspace_id"] == "ws-1"

    def test_query_audit_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = _run(handler_no_mound.query_audit_trail())
        assert _status(result) == 503

    def test_query_audit_invalid_action_returns_400(self, handler):
        """Invalid action string returns 400."""
        result = _run(handler.query_audit_trail(action="not_real_action"))
        assert _status(result) == 400
        body = _body(result)
        assert "invalid action" in body["error"].lower()

    def test_query_audit_action_enum_forwarded(self, handler, mock_mound):
        """Action string is converted to AuditAction enum and forwarded."""
        _run(handler.query_audit_trail(action="item.create"))
        call_kwargs = mock_mound.query_audit.call_args.kwargs
        assert call_kwargs["action"] == AuditAction.ITEM_CREATE

    def test_query_audit_no_action_passes_none(self, handler, mock_mound):
        """No action filter passes None to mound."""
        _run(handler.query_audit_trail())
        call_kwargs = mock_mound.query_audit.call_args.kwargs
        assert call_kwargs["action"] is None

    def test_query_audit_default_limit(self, handler, mock_mound):
        """Default limit is 100."""
        _run(handler.query_audit_trail())
        call_kwargs = mock_mound.query_audit.call_args.kwargs
        assert call_kwargs["limit"] == 100

    def test_query_audit_custom_limit(self, handler, mock_mound):
        """Custom limit is forwarded."""
        _run(handler.query_audit_trail(limit=25))
        call_kwargs = mock_mound.query_audit.call_args.kwargs
        assert call_kwargs["limit"] == 25

    def test_query_audit_server_error_returns_500(self, handler, mock_mound):
        """Server error returns 500."""
        mock_mound.query_audit = AsyncMock(side_effect=OSError("db fail"))
        result = _run(handler.query_audit_trail())
        assert _status(result) == 500

    def test_query_audit_value_error_returns_500(self, handler, mock_mound):
        """ValueError returns 500."""
        mock_mound.query_audit = AsyncMock(side_effect=ValueError("bad"))
        result = _run(handler.query_audit_trail())
        assert _status(result) == 500

    def test_query_audit_key_error_returns_500(self, handler, mock_mound):
        """KeyError returns 500."""
        mock_mound.query_audit = AsyncMock(side_effect=KeyError("missing"))
        result = _run(handler.query_audit_trail())
        assert _status(result) == 500

    def test_query_audit_empty_results(self, handler, mock_mound):
        """Empty audit trail returns count=0 and empty entries."""
        mock_mound.query_audit = AsyncMock(return_value=[])
        result = _run(handler.query_audit_trail())
        body = _body(result)
        assert body["count"] == 0
        assert body["entries"] == []

    def test_query_audit_count_matches_entries(self, handler, mock_mound):
        """Count field matches the number of entries."""
        result = _run(handler.query_audit_trail())
        body = _body(result)
        assert body["count"] == len(body["entries"])

    def test_query_audit_filters_reflect_none_defaults(self, handler, mock_mound):
        """When no filters passed, filters object shows None values."""
        result = _run(handler.query_audit_trail())
        body = _body(result)
        assert body["filters"]["actor_id"] is None
        assert body["filters"]["action"] is None
        assert body["filters"]["workspace_id"] is None

    def test_query_audit_with_actor_id_only(self, handler, mock_mound):
        """Actor ID filter is forwarded to mound."""
        _run(handler.query_audit_trail(actor_id="user-456"))
        call_kwargs = mock_mound.query_audit.call_args.kwargs
        assert call_kwargs["actor_id"] == "user-456"

    def test_query_audit_with_workspace_only(self, handler, mock_mound):
        """Workspace filter is forwarded to mound."""
        _run(handler.query_audit_trail(workspace_id="ws-prod"))
        call_kwargs = mock_mound.query_audit.call_args.kwargs
        assert call_kwargs["workspace_id"] == "ws-prod"

    def test_query_audit_valid_action_values(self, handler, mock_mound):
        """All valid AuditAction values are accepted."""
        for action in AuditAction:
            mock_mound.query_audit = AsyncMock(return_value=[])
            result = _run(handler.query_audit_trail(action=action.value))
            assert _status(result) == 200, f"Action {action.value} should be valid"


# ============================================================================
# Tests: get_user_activity
# ============================================================================


class TestGetUserActivity:
    """Test get_user_activity (GET /api/knowledge/mound/governance/audit/user/:user_id)."""

    def test_get_activity_success(self, handler, mock_mound):
        """Successfully getting user activity returns activity data."""
        result = _run(handler.get_user_activity(user_id="user-123", days=30))
        assert _status(result) == 200
        body = _body(result)
        assert body["user_id"] == "user-123"
        assert body["total_actions"] == 42

    def test_get_activity_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = _run(handler_no_mound.get_user_activity(user_id="user-1"))
        assert _status(result) == 503

    def test_get_activity_empty_user_id_returns_400(self, handler):
        """Empty user_id returns 400."""
        result = _run(handler.get_user_activity(user_id=""))
        assert _status(result) == 400

    def test_get_activity_default_days(self, handler, mock_mound):
        """Default days is 30."""
        _run(handler.get_user_activity(user_id="user-1"))
        call_kwargs = mock_mound.get_user_activity.call_args.kwargs
        assert call_kwargs["days"] == 30

    def test_get_activity_custom_days(self, handler, mock_mound):
        """Custom days value is forwarded."""
        _run(handler.get_user_activity(user_id="user-1", days=7))
        call_kwargs = mock_mound.get_user_activity.call_args.kwargs
        assert call_kwargs["days"] == 7

    def test_get_activity_server_error_returns_500(self, handler, mock_mound):
        """Server error returns 500."""
        mock_mound.get_user_activity = AsyncMock(side_effect=OSError("db fail"))
        result = _run(handler.get_user_activity(user_id="user-1"))
        assert _status(result) == 500

    def test_get_activity_value_error_returns_500(self, handler, mock_mound):
        """ValueError returns 500."""
        mock_mound.get_user_activity = AsyncMock(side_effect=ValueError("bad"))
        result = _run(handler.get_user_activity(user_id="user-1"))
        assert _status(result) == 500

    def test_get_activity_key_error_returns_500(self, handler, mock_mound):
        """KeyError returns 500."""
        mock_mound.get_user_activity = AsyncMock(side_effect=KeyError("missing"))
        result = _run(handler.get_user_activity(user_id="user-1"))
        assert _status(result) == 500

    def test_get_activity_type_error_returns_500(self, handler, mock_mound):
        """TypeError returns 500."""
        mock_mound.get_user_activity = AsyncMock(side_effect=TypeError("bad type"))
        result = _run(handler.get_user_activity(user_id="user-1"))
        assert _status(result) == 500

    def test_get_activity_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError returns 500."""
        mock_mound.get_user_activity = AsyncMock(side_effect=AttributeError("attr"))
        result = _run(handler.get_user_activity(user_id="user-1"))
        assert _status(result) == 500

    def test_get_activity_response_passed_through(self, handler, mock_mound):
        """Activity dict from mound is passed through directly as json_response."""
        custom_activity = {"user_id": "u1", "total_actions": 100, "custom_field": "yes"}
        mock_mound.get_user_activity = AsyncMock(return_value=custom_activity)
        result = _run(handler.get_user_activity(user_id="u1"))
        body = _body(result)
        assert body["custom_field"] == "yes"
        assert body["total_actions"] == 100

    def test_get_activity_user_id_forwarded(self, handler, mock_mound):
        """user_id is forwarded to mound."""
        _run(handler.get_user_activity(user_id="specific-user"))
        call_kwargs = mock_mound.get_user_activity.call_args.kwargs
        assert call_kwargs["user_id"] == "specific-user"


# ============================================================================
# Tests: get_governance_stats
# ============================================================================


class TestGetGovernanceStats:
    """Test get_governance_stats (GET /api/knowledge/mound/governance/stats)."""

    def test_get_stats_success(self, handler, mock_mound):
        """Successfully getting stats returns stats data."""
        result = _run(handler.get_governance_stats())
        assert _status(result) == 200
        body = _body(result)
        assert body["total_roles"] == 5
        assert body["total_assignments"] == 12
        assert body["total_audit_entries"] == 150
        assert body["active_users"] == 8

    def test_get_stats_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = _run(handler_no_mound.get_governance_stats())
        assert _status(result) == 503

    def test_get_stats_server_error_returns_500(self, handler, mock_mound):
        """Server error returns 500."""
        mock_mound.get_governance_stats = MagicMock(side_effect=OSError("db fail"))
        result = _run(handler.get_governance_stats())
        assert _status(result) == 500

    def test_get_stats_value_error_returns_500(self, handler, mock_mound):
        """ValueError returns 500."""
        mock_mound.get_governance_stats = MagicMock(side_effect=ValueError("bad"))
        result = _run(handler.get_governance_stats())
        assert _status(result) == 500

    def test_get_stats_key_error_returns_500(self, handler, mock_mound):
        """KeyError returns 500."""
        mock_mound.get_governance_stats = MagicMock(side_effect=KeyError("missing"))
        result = _run(handler.get_governance_stats())
        assert _status(result) == 500

    def test_get_stats_type_error_returns_500(self, handler, mock_mound):
        """TypeError returns 500."""
        mock_mound.get_governance_stats = MagicMock(side_effect=TypeError("wrong type"))
        result = _run(handler.get_governance_stats())
        assert _status(result) == 500

    def test_get_stats_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError returns 500."""
        mock_mound.get_governance_stats = MagicMock(side_effect=AttributeError("missing attr"))
        result = _run(handler.get_governance_stats())
        assert _status(result) == 500

    def test_get_stats_is_sync(self, handler, mock_mound):
        """get_governance_stats calls mound.get_governance_stats synchronously (not async)."""
        result = _run(handler.get_governance_stats())
        mock_mound.get_governance_stats.assert_called_once()
        assert _status(result) == 200

    def test_get_stats_empty_response(self, handler, mock_mound):
        """Stats response with empty/zero values works correctly."""
        mock_mound.get_governance_stats = MagicMock(
            return_value={
                "total_roles": 0,
                "total_assignments": 0,
                "total_audit_entries": 0,
                "active_users": 0,
            }
        )
        result = _run(handler.get_governance_stats())
        body = _body(result)
        assert body["total_roles"] == 0
        assert body["active_users"] == 0

    def test_get_stats_custom_fields(self, handler, mock_mound):
        """Custom fields in stats are passed through."""
        mock_mound.get_governance_stats = MagicMock(return_value={"custom": "value", "count": 42})
        result = _run(handler.get_governance_stats())
        body = _body(result)
        assert body["custom"] == "value"
        assert body["count"] == 42


# ============================================================================
# Tests: routing integration (how routing.py dispatches to mixin methods)
# ============================================================================


class TestGovernanceRouting:
    """Test routing dispatch for governance endpoints.

    These tests verify the routing wrappers in routing.py correctly parse
    request bodies/params and call the mixin methods.
    """

    def test_route_governance_roles_post_dispatches_to_create(self, handler, mock_mound):
        """POST /governance/roles dispatches to _handle_create_role."""
        mock_http = MagicMock()
        mock_http.command = "POST"
        mock_http.request.body = json.dumps(
            {
                "name": "Test Role",
                "permissions": ["read"],
            }
        ).encode()

        from aragora.server.handlers.knowledge_base.mound.routing import RoutingMixin

        # Add the _handle_create_role wrapper from RoutingMixin to our test handler
        handler._handle_create_role = lambda h: RoutingMixin._handle_create_role(handler, h)

        result = RoutingMixin._route_governance_roles(handler, "/governance/roles", {}, mock_http)
        assert result is not None
        assert _status(result) == 200
        mock_mound.create_role.assert_called_once()

    def test_route_governance_roles_get_returns_none(self, handler):
        """GET /governance/roles returns None (not implemented yet)."""
        mock_handler = MagicMock()
        mock_handler.command = "GET"

        from aragora.server.handlers.knowledge_base.mound.routing import RoutingMixin

        result = RoutingMixin._route_governance_roles(
            handler, "/governance/roles", {}, mock_handler
        )
        assert result is None

    def test_handle_governance_stats_via_routing(self, handler, mock_mound):
        """_handle_governance_stats calls get_governance_stats via _run_async."""
        from aragora.server.handlers.knowledge_base.mound.routing import RoutingMixin

        result = RoutingMixin._handle_governance_stats(handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["total_roles"] == 5


# ============================================================================
# Tests: edge cases and combined scenarios
# ============================================================================


class TestGovernanceEdgeCases:
    """Test edge cases across governance operations."""

    def test_create_role_none_name_returns_400(self, handler):
        """None name (falsy) returns 400."""
        result = _run(handler.create_role(name=None, permissions=["read"]))
        assert _status(result) == 400

    def test_create_role_none_permissions_returns_400(self, handler):
        """None permissions (falsy) returns 400."""
        result = _run(handler.create_role(name="Test", permissions=None))
        assert _status(result) == 400

    def test_assign_role_none_user_id_returns_400(self, handler):
        """None user_id (falsy) returns 400."""
        result = _run(handler.assign_role(user_id=None, role_id="role-1"))
        assert _status(result) == 400

    def test_assign_role_none_role_id_returns_400(self, handler):
        """None role_id (falsy) returns 400."""
        result = _run(handler.assign_role(user_id="user-1", role_id=None))
        assert _status(result) == 400

    def test_revoke_role_none_user_id_returns_400(self, handler):
        """None user_id (falsy) returns 400."""
        result = _run(handler.revoke_role(user_id=None, role_id="role-1"))
        assert _status(result) == 400

    def test_revoke_role_none_role_id_returns_400(self, handler):
        """None role_id (falsy) returns 400."""
        result = _run(handler.revoke_role(user_id="user-1", role_id=None))
        assert _status(result) == 400

    def test_check_permission_none_user_id_returns_400(self, handler):
        """None user_id returns 400."""
        result = _run(handler.check_permission(user_id=None, permission="read"))
        assert _status(result) == 400

    def test_check_permission_none_permission_returns_400(self, handler):
        """None permission returns 400."""
        result = _run(handler.check_permission(user_id="user-1", permission=None))
        assert _status(result) == 400

    def test_get_user_permissions_none_user_id_returns_400(self, handler):
        """None user_id returns 400."""
        result = _run(handler.get_user_permissions(user_id=None))
        assert _status(result) == 400

    def test_get_user_activity_none_user_id_returns_400(self, handler):
        """None user_id returns 400."""
        result = _run(handler.get_user_activity(user_id=None))
        assert _status(result) == 400

    def test_role_to_dict_in_create_response(self, handler, mock_mound):
        """Role.to_dict() is called and its result included in the response."""
        custom_role = MockRole(
            id="role-custom",
            name="Special Role",
            description="Special desc",
            workspace_id="ws-special",
        )
        mock_mound.create_role = AsyncMock(return_value=custom_role)
        result = _run(handler.create_role(name="Special Role", permissions=["read"]))
        body = _body(result)
        assert body["role"]["id"] == "role-custom"
        assert body["role"]["name"] == "Special Role"
        assert body["role"]["workspace_id"] == "ws-special"

    def test_assignment_to_dict_in_assign_response(self, handler, mock_mound):
        """RoleAssignment.to_dict() is called and result included in response."""
        custom_assignment = MockRoleAssignment(
            id="assign-custom",
            user_id="user-special",
            role_id="role-special",
            workspace_id="ws-special",
            assigned_by="admin-special",
        )
        mock_mound.assign_role = AsyncMock(return_value=custom_assignment)
        result = _run(
            handler.assign_role(
                user_id="user-special",
                role_id="role-special",
                workspace_id="ws-special",
                assigned_by="admin-special",
            )
        )
        body = _body(result)
        assert body["assignment"]["id"] == "assign-custom"
        assert body["assignment"]["assigned_by"] == "admin-special"

    def test_query_audit_entries_to_dict(self, handler, mock_mound):
        """Audit entries are converted via to_dict."""
        entries = [
            MockAuditEntry(id="a1", actor_id="u1", action="role.create"),
            MockAuditEntry(id="a2", actor_id="u2", action="share.grant"),
        ]
        mock_mound.query_audit = AsyncMock(return_value=entries)
        result = _run(handler.query_audit_trail())
        body = _body(result)
        assert body["entries"][0]["actor_id"] == "u1"
        assert body["entries"][1]["action"] == "share.grant"

    def test_revoke_role_message_format(self, handler, mock_mound):
        """Revoke message includes the role_id and user_id."""
        result = _run(handler.revoke_role(user_id="alice", role_id="editor-role"))
        body = _body(result)
        assert "editor-role" in body["message"]
        assert "alice" in body["message"]

    def test_create_role_response_has_success_true(self, handler, mock_mound):
        """Create role response always has success=True."""
        result = _run(handler.create_role(name="R", permissions=["read"]))
        body = _body(result)
        assert body["success"] is True

    def test_assign_role_response_has_success_true(self, handler, mock_mound):
        """Assign role response always has success=True."""
        result = _run(handler.assign_role(user_id="u1", role_id="r1"))
        body = _body(result)
        assert body["success"] is True

    def test_revoke_role_response_has_success_true(self, handler, mock_mound):
        """Revoke role response has success=True when revoke succeeds."""
        result = _run(handler.revoke_role(user_id="u1", role_id="r1"))
        body = _body(result)
        assert body["success"] is True
