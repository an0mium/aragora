"""
Tests for aragora.server.handlers.workspace.crud module.

Tests cover:
1. WorkspaceCrudMixin CRUD operations
2. Create workspace with validation
3. List workspaces with organization filtering
4. Get workspace by ID
5. Delete workspace
6. Cross-tenant security checks
7. RBAC permission enforcement
8. Audit logging
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Mock Classes
# =============================================================================


@dataclass
class MockWorkspace:
    """Mock workspace for testing."""

    id: str = "ws_test123"
    name: str = "Test Workspace"
    organization_id: str = "org_test123"
    created_by: str = "user_test123"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    members: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "organization_id": self.organization_id,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "members": self.members,
        }


@dataclass
class MockAuthContext:
    """Mock authorization context."""

    user_id: str = "user_test123"
    org_id: str = "org_test123"
    is_authenticated: bool = True
    tenant_id: str = "tenant_test123"


class MockAccessDeniedException(Exception):
    """Mock access denied exception."""

    pass


# =============================================================================
# Mock Module Helper
# =============================================================================


def create_mock_module():
    """Create a mock of workspace_module with all required attributes."""
    mock_mod = MagicMock()
    mock_mod.PERM_WORKSPACE_READ = "workspace:read"
    mock_mod.PERM_WORKSPACE_WRITE = "workspace:write"
    mock_mod.PERM_WORKSPACE_DELETE = "workspace:delete"
    mock_mod.AuditAction = MagicMock()
    mock_mod.AuditAction.CREATE_WORKSPACE = "create_workspace"
    mock_mod.AuditAction.DELETE_WORKSPACE = "delete_workspace"
    mock_mod.Actor = MagicMock()
    mock_mod.Resource = MagicMock()
    mock_mod.AuditOutcome = MagicMock()
    mock_mod.AuditOutcome.SUCCESS = "success"
    mock_mod.AccessDeniedException = MockAccessDeniedException
    mock_mod.extract_user_from_request = MagicMock(return_value=MockAuthContext())
    mock_mod.error_response = MagicMock(
        side_effect=lambda msg, status: {"error": msg, "status": status}
    )
    mock_mod.json_response = MagicMock(
        side_effect=lambda data, status=200: {"data": data, "status": status}
    )
    return mock_mod


# =============================================================================
# Mock Handler Base Class
# =============================================================================


class MockWorkspaceHandler:
    """Mock handler class that hosts the CRUD mixin."""

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
def mock_http_handler():
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler._json_body = {}
    return handler


@pytest.fixture
def mock_workspace():
    """Create a mock workspace."""
    return MockWorkspace()


# =============================================================================
# Test Create Workspace
# =============================================================================


class TestCreateWorkspace:
    """Tests for _handle_create_workspace."""

    def test_create_workspace_success(self, mock_http_handler, mock_workspace):
        """Create workspace returns created workspace."""
        mock_mod = create_mock_module()
        mock_http_handler._json_body = {"name": "New Workspace"}

        with patch("aragora.server.handlers.workspace.crud._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.crud import WorkspaceCrudMixin

            class TestHandler(WorkspaceCrudMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            handler._isolation_manager.create_workspace = AsyncMock(return_value=mock_workspace)
            result = handler._handle_create_workspace(mock_http_handler)

        assert result["status"] == 201
        assert "workspace" in result["data"]
        assert result["data"]["message"] == "Workspace created successfully"

    def test_create_workspace_not_authenticated(self, mock_http_handler):
        """Create workspace returns 401 when not authenticated."""
        mock_mod = create_mock_module()
        mock_mod.extract_user_from_request.return_value = MockAuthContext(is_authenticated=False)
        mock_http_handler._json_body = {"name": "New Workspace"}

        with patch("aragora.server.handlers.workspace.crud._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.crud import WorkspaceCrudMixin

            class TestHandler(WorkspaceCrudMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_create_workspace(mock_http_handler)

        assert result["status"] == 401
        assert "Not authenticated" in result["error"]

    def test_create_workspace_missing_name(self, mock_http_handler):
        """Create workspace returns 400 when name missing."""
        mock_mod = create_mock_module()
        mock_http_handler._json_body = {}

        with patch("aragora.server.handlers.workspace.crud._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.crud import WorkspaceCrudMixin

            class TestHandler(WorkspaceCrudMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_create_workspace(mock_http_handler)

        assert result["status"] == 400
        assert "name is required" in result["error"]

    def test_create_workspace_invalid_json(self, mock_http_handler):
        """Create workspace returns 400 when JSON invalid."""
        mock_mod = create_mock_module()
        mock_http_handler._json_body = None

        with patch("aragora.server.handlers.workspace.crud._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.crud import WorkspaceCrudMixin

            class TestHandler(WorkspaceCrudMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_create_workspace(mock_http_handler)

        assert result["status"] == 400
        assert "Invalid JSON" in result["error"]

    def test_create_workspace_cross_tenant_rejected(self, mock_http_handler, mock_workspace):
        """Create workspace rejects cross-tenant creation attempt."""
        mock_mod = create_mock_module()
        mock_http_handler._json_body = {
            "name": "New Workspace",
            "organization_id": "different_org_456",  # Different from user's org
        }

        with patch("aragora.server.handlers.workspace.crud._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.crud import WorkspaceCrudMixin

            class TestHandler(WorkspaceCrudMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_create_workspace(mock_http_handler)

        assert result["status"] == 403
        assert "Cannot create workspace in another organization" in result["error"]

    def test_create_workspace_rbac_denied(self, mock_http_handler):
        """Create workspace returns error when RBAC denied."""
        mock_mod = create_mock_module()
        mock_http_handler._json_body = {"name": "New Workspace"}

        with patch("aragora.server.handlers.workspace.crud._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.crud import WorkspaceCrudMixin

            class TestHandler(WorkspaceCrudMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            handler._rbac_error = {"error": "Permission denied", "status": 403}
            result = handler._handle_create_workspace(mock_http_handler)

        assert result["status"] == 403

    def test_create_workspace_missing_org_id(self, mock_http_handler):
        """Create workspace returns 400 when org_id missing from auth context."""
        mock_mod = create_mock_module()
        mock_mod.extract_user_from_request.return_value = MockAuthContext(org_id=None)
        mock_http_handler._json_body = {"name": "New Workspace"}

        with patch("aragora.server.handlers.workspace.crud._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.crud import WorkspaceCrudMixin

            class TestHandler(WorkspaceCrudMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_create_workspace(mock_http_handler)

        assert result["status"] == 400
        assert "organization_id is required" in result["error"]

    def test_create_workspace_with_initial_members(self, mock_http_handler, mock_workspace):
        """Create workspace accepts initial members list."""
        mock_mod = create_mock_module()
        mock_http_handler._json_body = {
            "name": "New Workspace",
            "members": ["user_1", "user_2"],
        }

        with patch("aragora.server.handlers.workspace.crud._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.crud import WorkspaceCrudMixin

            class TestHandler(WorkspaceCrudMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            handler._isolation_manager.create_workspace = AsyncMock(return_value=mock_workspace)
            handler._handle_create_workspace(mock_http_handler)

        # Verify initial_members was passed to manager
        handler._isolation_manager.create_workspace.assert_called_once()
        call_kwargs = handler._isolation_manager.create_workspace.call_args.kwargs
        assert call_kwargs["initial_members"] == ["user_1", "user_2"]


# =============================================================================
# Test List Workspaces
# =============================================================================


class TestListWorkspaces:
    """Tests for _handle_list_workspaces."""

    def test_list_workspaces_success(self, mock_http_handler, mock_workspace):
        """List workspaces returns workspace list."""
        mock_mod = create_mock_module()

        with patch("aragora.server.handlers.workspace.crud._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.crud import WorkspaceCrudMixin

            class TestHandler(WorkspaceCrudMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            handler._isolation_manager.list_workspaces = AsyncMock(return_value=[mock_workspace])
            result = handler._handle_list_workspaces(mock_http_handler, {})

        assert result["status"] == 200
        assert "workspaces" in result["data"]
        assert result["data"]["total"] == 1

    def test_list_workspaces_not_authenticated(self, mock_http_handler):
        """List workspaces returns 401 when not authenticated."""
        mock_mod = create_mock_module()
        mock_mod.extract_user_from_request.return_value = MockAuthContext(is_authenticated=False)

        with patch("aragora.server.handlers.workspace.crud._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.crud import WorkspaceCrudMixin

            class TestHandler(WorkspaceCrudMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_list_workspaces(mock_http_handler, {})

        assert result["status"] == 401

    def test_list_workspaces_cross_tenant_rejected(self, mock_http_handler):
        """List workspaces rejects cross-tenant access attempt."""
        mock_mod = create_mock_module()

        with patch("aragora.server.handlers.workspace.crud._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.crud import WorkspaceCrudMixin

            class TestHandler(WorkspaceCrudMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_list_workspaces(
                mock_http_handler, {"organization_id": "different_org_456"}
            )

        assert result["status"] == 403
        assert "Cannot list workspaces from another organization" in result["error"]

    def test_list_workspaces_rbac_denied(self, mock_http_handler):
        """List workspaces returns error when RBAC denied."""
        mock_mod = create_mock_module()

        with patch("aragora.server.handlers.workspace.crud._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.crud import WorkspaceCrudMixin

            class TestHandler(WorkspaceCrudMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            handler._rbac_error = {"error": "Permission denied", "status": 403}
            result = handler._handle_list_workspaces(mock_http_handler, {})

        assert result["status"] == 403


# =============================================================================
# Test Get Workspace
# =============================================================================


class TestGetWorkspace:
    """Tests for _handle_get_workspace."""

    def test_get_workspace_success(self, mock_http_handler, mock_workspace):
        """Get workspace returns workspace details."""
        mock_mod = create_mock_module()

        with patch("aragora.server.handlers.workspace.crud._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.crud import WorkspaceCrudMixin

            class TestHandler(WorkspaceCrudMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            handler._isolation_manager.get_workspace = AsyncMock(return_value=mock_workspace)
            result = handler._handle_get_workspace(mock_http_handler, "ws_123")

        assert result["status"] == 200
        assert "workspace" in result["data"]

    def test_get_workspace_not_authenticated(self, mock_http_handler):
        """Get workspace returns 401 when not authenticated."""
        mock_mod = create_mock_module()
        mock_mod.extract_user_from_request.return_value = MockAuthContext(is_authenticated=False)

        with patch("aragora.server.handlers.workspace.crud._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.crud import WorkspaceCrudMixin

            class TestHandler(WorkspaceCrudMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_get_workspace(mock_http_handler, "ws_123")

        assert result["status"] == 401

    def test_get_workspace_access_denied(self, mock_http_handler):
        """Get workspace returns 403 when access denied."""
        mock_mod = create_mock_module()

        with patch("aragora.server.handlers.workspace.crud._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.crud import WorkspaceCrudMixin

            class TestHandler(WorkspaceCrudMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            handler._isolation_manager.get_workspace = AsyncMock(
                side_effect=MockAccessDeniedException("Not allowed")
            )
            result = handler._handle_get_workspace(mock_http_handler, "ws_123")

        assert result["status"] == 403

    def test_get_workspace_rbac_denied(self, mock_http_handler):
        """Get workspace returns error when RBAC denied."""
        mock_mod = create_mock_module()

        with patch("aragora.server.handlers.workspace.crud._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.crud import WorkspaceCrudMixin

            class TestHandler(WorkspaceCrudMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            handler._rbac_error = {"error": "Permission denied", "status": 403}
            result = handler._handle_get_workspace(mock_http_handler, "ws_123")

        assert result["status"] == 403


# =============================================================================
# Test Delete Workspace
# =============================================================================


class TestDeleteWorkspace:
    """Tests for _handle_delete_workspace."""

    def test_delete_workspace_success(self, mock_http_handler):
        """Delete workspace returns success message."""
        mock_mod = create_mock_module()
        mock_http_handler._json_body = {}

        with patch("aragora.server.handlers.workspace.crud._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.crud import WorkspaceCrudMixin

            class TestHandler(WorkspaceCrudMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            handler._isolation_manager.delete_workspace = AsyncMock()
            result = handler._handle_delete_workspace(mock_http_handler, "ws_123")

        assert result["status"] == 200
        assert "deleted successfully" in result["data"]["message"]

    def test_delete_workspace_not_authenticated(self, mock_http_handler):
        """Delete workspace returns 401 when not authenticated."""
        mock_mod = create_mock_module()
        mock_mod.extract_user_from_request.return_value = MockAuthContext(is_authenticated=False)

        with patch("aragora.server.handlers.workspace.crud._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.crud import WorkspaceCrudMixin

            class TestHandler(WorkspaceCrudMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_delete_workspace(mock_http_handler, "ws_123")

        assert result["status"] == 401

    def test_delete_workspace_access_denied(self, mock_http_handler):
        """Delete workspace returns 403 when access denied."""
        mock_mod = create_mock_module()
        mock_http_handler._json_body = {}

        with patch("aragora.server.handlers.workspace.crud._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.crud import WorkspaceCrudMixin

            class TestHandler(WorkspaceCrudMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            handler._isolation_manager.delete_workspace = AsyncMock(
                side_effect=MockAccessDeniedException("Not allowed")
            )
            result = handler._handle_delete_workspace(mock_http_handler, "ws_123")

        assert result["status"] == 403

    def test_delete_workspace_force_flag(self, mock_http_handler):
        """Delete workspace passes force flag to manager."""
        mock_mod = create_mock_module()
        mock_http_handler._json_body = {"force": True}

        with patch("aragora.server.handlers.workspace.crud._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.crud import WorkspaceCrudMixin

            class TestHandler(WorkspaceCrudMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            handler._isolation_manager.delete_workspace = AsyncMock()
            handler._handle_delete_workspace(mock_http_handler, "ws_123")

        # Verify force was passed to manager
        handler._isolation_manager.delete_workspace.assert_called_once()
        call_kwargs = handler._isolation_manager.delete_workspace.call_args.kwargs
        assert call_kwargs["force"] is True

    def test_delete_workspace_rbac_denied(self, mock_http_handler):
        """Delete workspace returns error when RBAC denied."""
        mock_mod = create_mock_module()
        mock_http_handler._json_body = {}

        with patch("aragora.server.handlers.workspace.crud._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.crud import WorkspaceCrudMixin

            class TestHandler(WorkspaceCrudMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            handler._rbac_error = {"error": "Permission denied", "status": 403}
            result = handler._handle_delete_workspace(mock_http_handler, "ws_123")

        assert result["status"] == 403


# =============================================================================
# Test Audit Logging
# =============================================================================


class TestAuditLogging:
    """Tests for audit log integration."""

    def test_create_workspace_audit_logged(self, mock_http_handler, mock_workspace):
        """Create workspace logs audit event."""
        mock_mod = create_mock_module()
        mock_http_handler._json_body = {"name": "New Workspace"}

        with patch("aragora.server.handlers.workspace.crud._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.crud import WorkspaceCrudMixin

            class TestHandler(WorkspaceCrudMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            handler._isolation_manager.create_workspace = AsyncMock(return_value=mock_workspace)
            handler._handle_create_workspace(mock_http_handler)

        # Verify audit log was called
        handler._audit_log.log.assert_called_once()

    def test_delete_workspace_audit_logged(self, mock_http_handler):
        """Delete workspace logs audit event."""
        mock_mod = create_mock_module()
        mock_http_handler._json_body = {}

        with patch("aragora.server.handlers.workspace.crud._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.crud import WorkspaceCrudMixin

            class TestHandler(WorkspaceCrudMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            handler._isolation_manager.delete_workspace = AsyncMock()
            handler._handle_delete_workspace(mock_http_handler, "ws_123")

        # Verify audit log was called
        handler._audit_log.log.assert_called_once()


__all__ = [
    "TestCreateWorkspace",
    "TestListWorkspaces",
    "TestGetWorkspace",
    "TestDeleteWorkspace",
    "TestAuditLogging",
]
