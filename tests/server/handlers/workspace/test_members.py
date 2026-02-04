"""
Tests for aragora.server.handlers.workspace.members module.

Tests cover:
1. WorkspaceMembersMixin handler methods
2. Add/remove member operations
3. Role management (list, update)
4. RBAC profile listing
5. Authentication and permission checks
6. Error handling and validation
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Mock Data Classes
# =============================================================================


@dataclass
class MockAuthContext:
    """Mock authorization context."""

    user_id: str = "user_123"
    is_authenticated: bool = True
    tenant_id: str = "tenant_123"


@dataclass
class MockWorkspace:
    """Mock workspace object."""

    id: str = "ws_test123"
    name: str = "Test Workspace"
    owner_id: str = "user_123"
    members: list[str] = field(default_factory=lambda: ["user_123", "user_456"])
    rbac_profile: str = "lite"
    member_roles: dict[str, str] = field(
        default_factory=lambda: {"user_123": "owner", "user_456": "member"}
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "owner_id": self.owner_id,
            "members": self.members,
            "rbac_profile": self.rbac_profile,
            "member_roles": self.member_roles,
        }


@dataclass
class MockProfileConfig:
    """Mock RBAC profile configuration."""

    name: str = "Lite"
    description: str = "Simple RBAC for small teams"
    roles: list[str] = field(default_factory=lambda: ["owner", "admin", "member"])
    default_role: str = "member"
    features: set[str] = field(default_factory=lambda: {"basic_sharing", "audit_log"})


@dataclass
class MockRole:
    """Mock role definition."""

    name: str = "Member"
    description: str = "Basic workspace access"


# =============================================================================
# Mock Module Helper
# =============================================================================


def create_mock_module():
    """Create a mock of the workspace_module with all required attributes."""
    mock_mod = MagicMock()

    # Permission constants
    mock_mod.PERM_WORKSPACE_SHARE = "workspace:share"
    mock_mod.PERM_WORKSPACE_READ = "workspace:read"
    mock_mod.PERM_WORKSPACE_ADMIN = "workspace:admin"

    # Enums and classes
    mock_mod.WorkspacePermission = MagicMock(side_effect=lambda x: x)
    mock_mod.AuditAction = MagicMock()
    mock_mod.AuditAction.ADD_MEMBER = "add_member"
    mock_mod.AuditAction.REMOVE_MEMBER = "remove_member"
    mock_mod.AuditAction.MODIFY_PERMISSIONS = "modify_permissions"
    mock_mod.Actor = MagicMock()
    mock_mod.Resource = MagicMock()
    mock_mod.AuditOutcome = MagicMock()
    mock_mod.AuditOutcome.SUCCESS = "success"
    mock_mod.AccessDeniedException = type("AccessDeniedException", (Exception,), {})

    # Profile-related
    mock_mod.PROFILES_AVAILABLE = True

    # Mock profile enum members with .value attribute (like Python enums)
    class MockProfile:
        def __init__(self, name: str):
            self.value = name
            self.name = name.upper()

        def __repr__(self):
            return f"MockProfile({self.value})"

    # Create profiles list - this is iterable and each item has .value
    profiles_list = [
        MockProfile("lite"),
        MockProfile("standard"),
        MockProfile("enterprise"),
    ]

    # RBACProfile needs to be iterable (for `for profile in m.RBACProfile:`)
    mock_mod.RBACProfile = profiles_list
    mock_mod.get_profile_config = MagicMock(return_value=MockProfileConfig())
    mock_mod.get_profile_roles = MagicMock(
        return_value={
            "owner": MockRole(name="Owner", description="Full control"),
            "admin": MockRole(name="Admin", description="Administrative access"),
            "member": MockRole(name="Member", description="Basic access"),
        }
    )
    mock_mod.get_lite_role_summary = MagicMock(
        return_value={
            "owner": "Full workspace control",
            "admin": "Manage members and content",
            "member": "Read and contribute",
        }
    )
    mock_mod.get_available_roles_for_assignment = MagicMock(return_value=["admin", "member"])

    # Response helpers
    mock_mod.extract_user_from_request = MagicMock(return_value=MockAuthContext())
    mock_mod.error_response = MagicMock(
        side_effect=lambda msg, status: {"error": msg, "status": status}
    )
    mock_mod.json_response = MagicMock(
        side_effect=lambda data, status=200: {"data": data, "status": status}
    )

    return mock_mod


# =============================================================================
# Mock Handler Class
# =============================================================================


class MockWorkspaceHandler:
    """Mock handler class that includes the members mixin."""

    def __init__(self):
        self._user_store = MagicMock()
        self._isolation_manager = MagicMock()
        self._audit_log = MagicMock()
        self._audit_log.log = AsyncMock()
        self._rbac_error = None

    def _get_user_store(self):
        return self._user_store

    def _get_isolation_manager(self):
        return self._isolation_manager

    def _get_audit_log(self):
        return self._audit_log

    def _run_async(self, coro):
        """Run async coroutine synchronously."""
        if asyncio.iscoroutine(coro):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
        return coro

    def _check_rbac_permission(self, handler, perm, auth_ctx):
        return self._rbac_error

    def read_json_body(self, handler):
        return handler._json_body


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_workspace_handler():
    """Create a mock workspace handler with mixin methods."""
    handler = MockWorkspaceHandler()
    handler._isolation_manager.add_member = AsyncMock()
    handler._isolation_manager.remove_member = AsyncMock()
    handler._isolation_manager.get_workspace = AsyncMock(return_value=MockWorkspace())
    return handler


# =============================================================================
# Test Add Member
# =============================================================================


class TestAddMember:
    """Tests for _handle_add_member."""

    def test_add_member_success(self, mock_workspace_handler):
        """Add member returns success message."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()
        http_handler._json_body = {"user_id": "user_789", "permissions": ["read", "write"]}

        with patch("aragora.server.handlers.workspace.members._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.members import WorkspaceMembersMixin

            class TestHandler(WorkspaceMembersMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_add_member(http_handler, "ws_123")

        assert result["status"] == 201
        assert "user_789" in result["data"]["message"]

    def test_add_member_missing_user_id(self, mock_workspace_handler):
        """Add member returns error when user_id missing."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()
        http_handler._json_body = {"permissions": ["read"]}

        with patch("aragora.server.handlers.workspace.members._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.members import WorkspaceMembersMixin

            class TestHandler(WorkspaceMembersMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_add_member(http_handler, "ws_123")

        assert result["status"] == 400
        assert "user_id is required" in result["error"]

    def test_add_member_invalid_json(self, mock_workspace_handler):
        """Add member returns error when JSON invalid."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()
        http_handler._json_body = None

        with patch("aragora.server.handlers.workspace.members._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.members import WorkspaceMembersMixin

            class TestHandler(WorkspaceMembersMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_add_member(http_handler, "ws_123")

        assert result["status"] == 400
        assert "Invalid JSON" in result["error"]

    def test_add_member_not_authenticated(self, mock_workspace_handler):
        """Add member returns 401 when not authenticated."""
        mock_mod = create_mock_module()
        mock_mod.extract_user_from_request.return_value = MockAuthContext(is_authenticated=False)
        http_handler = MagicMock()
        http_handler._json_body = {"user_id": "user_789"}

        with patch("aragora.server.handlers.workspace.members._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.members import WorkspaceMembersMixin

            class TestHandler(WorkspaceMembersMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_add_member(http_handler, "ws_123")

        assert result["status"] == 401

    def test_add_member_access_denied(self, mock_workspace_handler):
        """Add member returns 403 when access denied."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()
        http_handler._json_body = {"user_id": "user_789"}

        with patch("aragora.server.handlers.workspace.members._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.members import WorkspaceMembersMixin

            class TestHandler(WorkspaceMembersMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            handler._isolation_manager.add_member = AsyncMock(
                side_effect=mock_mod.AccessDeniedException("Cannot add member")
            )
            result = handler._handle_add_member(http_handler, "ws_123")

        assert result["status"] == 403

    def test_add_member_default_permissions(self, mock_workspace_handler):
        """Add member uses default permissions when not specified."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()
        http_handler._json_body = {"user_id": "user_789"}  # No permissions specified

        with patch("aragora.server.handlers.workspace.members._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.members import WorkspaceMembersMixin

            class TestHandler(WorkspaceMembersMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            handler._handle_add_member(http_handler, "ws_123")

        # Default should be ["read"]
        handler._isolation_manager.add_member.assert_called_once()


# =============================================================================
# Test Remove Member
# =============================================================================


class TestRemoveMember:
    """Tests for _handle_remove_member."""

    def test_remove_member_success(self, mock_workspace_handler):
        """Remove member returns success message."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()

        with patch("aragora.server.handlers.workspace.members._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.members import WorkspaceMembersMixin

            class TestHandler(WorkspaceMembersMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_remove_member(http_handler, "ws_123", "user_456")

        assert result["status"] == 200
        assert "user_456" in result["data"]["message"]
        assert "removed" in result["data"]["message"]

    def test_remove_member_not_authenticated(self, mock_workspace_handler):
        """Remove member returns 401 when not authenticated."""
        mock_mod = create_mock_module()
        mock_mod.extract_user_from_request.return_value = MockAuthContext(is_authenticated=False)
        http_handler = MagicMock()

        with patch("aragora.server.handlers.workspace.members._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.members import WorkspaceMembersMixin

            class TestHandler(WorkspaceMembersMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_remove_member(http_handler, "ws_123", "user_456")

        assert result["status"] == 401

    def test_remove_member_access_denied(self, mock_workspace_handler):
        """Remove member returns 403 when access denied."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()

        with patch("aragora.server.handlers.workspace.members._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.members import WorkspaceMembersMixin

            class TestHandler(WorkspaceMembersMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            handler._isolation_manager.remove_member = AsyncMock(
                side_effect=mock_mod.AccessDeniedException("Cannot remove member")
            )
            result = handler._handle_remove_member(http_handler, "ws_123", "user_456")

        assert result["status"] == 403


# =============================================================================
# Test List Profiles
# =============================================================================


class TestListProfiles:
    """Tests for _handle_list_profiles."""

    def test_list_profiles_success(self, mock_workspace_handler):
        """List profiles returns available profiles."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()

        with patch("aragora.server.handlers.workspace.members._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.members import WorkspaceMembersMixin

            class TestHandler(WorkspaceMembersMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_list_profiles(http_handler)

        assert result["status"] == 200
        assert "profiles" in result["data"]
        assert "lite_roles_detail" in result["data"]
        assert "recommended" in result["data"]

    def test_list_profiles_not_authenticated(self, mock_workspace_handler):
        """List profiles returns 401 when not authenticated."""
        mock_mod = create_mock_module()
        mock_mod.extract_user_from_request.return_value = MockAuthContext(is_authenticated=False)
        http_handler = MagicMock()

        with patch("aragora.server.handlers.workspace.members._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.members import WorkspaceMembersMixin

            class TestHandler(WorkspaceMembersMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_list_profiles(http_handler)

        assert result["status"] == 401

    def test_list_profiles_not_available(self, mock_workspace_handler):
        """List profiles returns 503 when profiles not available."""
        mock_mod = create_mock_module()
        mock_mod.PROFILES_AVAILABLE = False
        http_handler = MagicMock()

        with patch("aragora.server.handlers.workspace.members._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.members import WorkspaceMembersMixin

            class TestHandler(WorkspaceMembersMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_list_profiles(http_handler)

        assert result["status"] == 503


# =============================================================================
# Test Get Workspace Roles
# =============================================================================


class TestGetWorkspaceRoles:
    """Tests for _handle_get_workspace_roles."""

    def test_get_workspace_roles_success(self, mock_workspace_handler):
        """Get workspace roles returns role list."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()

        with patch("aragora.server.handlers.workspace.members._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.members import WorkspaceMembersMixin

            class TestHandler(WorkspaceMembersMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_get_workspace_roles(http_handler, "ws_123")

        assert result["status"] == 200
        assert "roles" in result["data"]
        assert "workspace_id" in result["data"]
        assert "profile" in result["data"]
        assert "your_role" in result["data"]

    def test_get_workspace_roles_not_authenticated(self, mock_workspace_handler):
        """Get workspace roles returns 401 when not authenticated."""
        mock_mod = create_mock_module()
        mock_mod.extract_user_from_request.return_value = MockAuthContext(is_authenticated=False)
        http_handler = MagicMock()

        with patch("aragora.server.handlers.workspace.members._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.members import WorkspaceMembersMixin

            class TestHandler(WorkspaceMembersMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_get_workspace_roles(http_handler, "ws_123")

        assert result["status"] == 401

    def test_get_workspace_roles_access_denied(self, mock_workspace_handler):
        """Get workspace roles returns 403 when access denied."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()

        with patch("aragora.server.handlers.workspace.members._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.members import WorkspaceMembersMixin

            class TestHandler(WorkspaceMembersMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            handler._isolation_manager.get_workspace = AsyncMock(
                side_effect=mock_mod.AccessDeniedException("Access denied")
            )
            result = handler._handle_get_workspace_roles(http_handler, "ws_123")

        assert result["status"] == 403


# =============================================================================
# Test Update Member Role
# =============================================================================


class TestUpdateMemberRole:
    """Tests for _handle_update_member_role."""

    def test_update_member_role_success(self, mock_workspace_handler):
        """Update member role returns success."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()
        http_handler._json_body = {"role": "admin"}

        with patch("aragora.server.handlers.workspace.members._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.members import WorkspaceMembersMixin

            class TestHandler(WorkspaceMembersMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_update_member_role(http_handler, "ws_123", "user_456")

        assert result["status"] == 200
        assert "admin" in result["data"]["message"]

    def test_update_member_role_missing_role(self, mock_workspace_handler):
        """Update member role returns error when role missing."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()
        http_handler._json_body = {}

        with patch("aragora.server.handlers.workspace.members._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.members import WorkspaceMembersMixin

            class TestHandler(WorkspaceMembersMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_update_member_role(http_handler, "ws_123", "user_456")

        assert result["status"] == 400
        assert "role is required" in result["error"]

    def test_update_member_role_invalid_json(self, mock_workspace_handler):
        """Update member role returns error when JSON invalid."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()
        http_handler._json_body = None

        with patch("aragora.server.handlers.workspace.members._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.members import WorkspaceMembersMixin

            class TestHandler(WorkspaceMembersMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_update_member_role(http_handler, "ws_123", "user_456")

        assert result["status"] == 400

    def test_update_member_role_invalid_role(self, mock_workspace_handler):
        """Update member role returns error for invalid role."""
        mock_mod = create_mock_module()
        mock_mod.get_profile_config.return_value = MockProfileConfig()
        http_handler = MagicMock()
        http_handler._json_body = {"role": "superadmin"}  # Not a valid role

        with patch("aragora.server.handlers.workspace.members._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.members import WorkspaceMembersMixin

            class TestHandler(WorkspaceMembersMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_update_member_role(http_handler, "ws_123", "user_456")

        assert result["status"] == 400
        assert "not available" in result["error"]

    def test_update_member_role_cannot_assign(self, mock_workspace_handler):
        """Update member role returns 403 when user cannot assign role."""
        mock_mod = create_mock_module()
        mock_mod.get_available_roles_for_assignment.return_value = [
            "member"
        ]  # Can only assign member
        http_handler = MagicMock()
        http_handler._json_body = {"role": "admin"}  # Trying to assign admin

        with patch("aragora.server.handlers.workspace.members._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.members import WorkspaceMembersMixin

            class TestHandler(WorkspaceMembersMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_update_member_role(http_handler, "ws_123", "user_456")

        assert result["status"] == 403
        assert "cannot assign" in result["error"]

    def test_update_member_role_last_owner_protection(self, mock_workspace_handler):
        """Update member role prevents removing last owner."""
        mock_mod = create_mock_module()
        mock_mod.get_available_roles_for_assignment.return_value = ["owner", "admin", "member"]
        http_handler = MagicMock()
        http_handler._json_body = {"role": "member"}  # Demoting from owner

        # Only one owner in workspace
        workspace = MockWorkspace()
        workspace.member_roles = {"user_456": "owner"}  # user_456 is the only owner

        with patch("aragora.server.handlers.workspace.members._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.members import WorkspaceMembersMixin

            class TestHandler(WorkspaceMembersMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            handler._isolation_manager.get_workspace = AsyncMock(return_value=workspace)
            result = handler._handle_update_member_role(http_handler, "ws_123", "user_456")

        assert result["status"] == 400
        assert "last owner" in result["error"]

    def test_update_member_role_not_authenticated(self, mock_workspace_handler):
        """Update member role returns 401 when not authenticated."""
        mock_mod = create_mock_module()
        mock_mod.extract_user_from_request.return_value = MockAuthContext(is_authenticated=False)
        http_handler = MagicMock()
        http_handler._json_body = {"role": "admin"}

        with patch("aragora.server.handlers.workspace.members._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.members import WorkspaceMembersMixin

            class TestHandler(WorkspaceMembersMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_update_member_role(http_handler, "ws_123", "user_456")

        assert result["status"] == 401

    def test_update_member_role_profiles_not_available(self, mock_workspace_handler):
        """Update member role returns 503 when profiles not available."""
        mock_mod = create_mock_module()
        mock_mod.PROFILES_AVAILABLE = False
        http_handler = MagicMock()
        http_handler._json_body = {"role": "admin"}

        with patch("aragora.server.handlers.workspace.members._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.members import WorkspaceMembersMixin

            class TestHandler(WorkspaceMembersMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_update_member_role(http_handler, "ws_123", "user_456")

        assert result["status"] == 503


__all__ = [
    "TestAddMember",
    "TestRemoveMember",
    "TestListProfiles",
    "TestGetWorkspaceRoles",
    "TestUpdateMemberRole",
]
