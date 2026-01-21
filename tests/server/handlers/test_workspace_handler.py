"""
Tests for the WorkspaceHandler module.

Tests cover:
- Handler routing for all workspace endpoints
- Workspace CRUD routes
- Retention policy routes
- Classification routes
- Audit log routes
- can_handle method
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from aragora.server.handlers.workspace import WorkspaceHandler


@pytest.fixture
def mock_server_context():
    """Create mock server context for handler initialization."""
    return {"storage": None, "elo_system": None, "nomic_dir": None, "user_store": None}


class TestWorkspaceHandlerRouting:
    """Tests for handler routing."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkspaceHandler(mock_server_context)

    def test_can_handle_workspaces(self, handler):
        """Handler can handle workspaces base path."""
        assert handler.can_handle("/api/workspaces")

    def test_can_handle_workspace_by_id(self, handler):
        """Handler can handle workspace by ID."""
        assert handler.can_handle("/api/workspaces/ws_123")

    def test_can_handle_workspace_members(self, handler):
        """Handler can handle workspace members."""
        assert handler.can_handle("/api/workspaces/ws_123/members")

    def test_can_handle_retention_policies(self, handler):
        """Handler can handle retention policies."""
        assert handler.can_handle("/api/retention/policies")

    def test_can_handle_retention_expiring(self, handler):
        """Handler can handle retention expiring."""
        assert handler.can_handle("/api/retention/expiring")

    def test_can_handle_classify(self, handler):
        """Handler can handle classify endpoint."""
        assert handler.can_handle("/api/classify")

    def test_can_handle_classify_policy(self, handler):
        """Handler can handle classify policy by level."""
        assert handler.can_handle("/api/classify/policy/high")

    def test_can_handle_audit_entries(self, handler):
        """Handler can handle audit entries."""
        assert handler.can_handle("/api/audit/entries")

    def test_can_handle_audit_report(self, handler):
        """Handler can handle audit report."""
        assert handler.can_handle("/api/audit/report")

    def test_can_handle_audit_verify(self, handler):
        """Handler can handle audit verify."""
        assert handler.can_handle("/api/audit/verify")

    def test_can_handle_audit_actor(self, handler):
        """Handler can handle audit actor history."""
        assert handler.can_handle("/api/audit/actor/user123/history")

    def test_can_handle_audit_resource(self, handler):
        """Handler can handle audit resource history."""
        assert handler.can_handle("/api/audit/resource/doc123/history")

    def test_can_handle_audit_denied(self, handler):
        """Handler can handle audit denied attempts."""
        assert handler.can_handle("/api/audit/denied")

    def test_cannot_handle_other_paths(self, handler):
        """Handler cannot handle unrelated paths."""
        assert not handler.can_handle("/api/debates")
        assert not handler.can_handle("/api/other")


class TestWorkspaceHandlerRoutesAttribute:
    """Tests for ROUTES class attribute."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkspaceHandler(mock_server_context)

    def test_routes_contains_workspaces(self, handler):
        """ROUTES contains workspaces."""
        assert "/api/workspaces" in handler.ROUTES

    def test_routes_contains_retention(self, handler):
        """ROUTES contains retention endpoints."""
        assert "/api/retention/policies" in handler.ROUTES
        assert "/api/retention/expiring" in handler.ROUTES

    def test_routes_contains_classify(self, handler):
        """ROUTES contains classify."""
        assert "/api/classify" in handler.ROUTES

    def test_routes_contains_audit_endpoints(self, handler):
        """ROUTES contains audit endpoints."""
        assert "/api/audit/entries" in handler.ROUTES
        assert "/api/audit/report" in handler.ROUTES
        assert "/api/audit/verify" in handler.ROUTES


class TestWorkspaceHandlerManagerInitialization:
    """Tests for lazy manager initialization."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkspaceHandler(mock_server_context)

    def test_isolation_manager_starts_none(self, handler):
        """Isolation manager starts as None."""
        assert handler._isolation_manager is None

    def test_retention_manager_starts_none(self, handler):
        """Retention manager starts as None."""
        assert handler._retention_manager is None

    def test_classifier_starts_none(self, handler):
        """Classifier starts as None."""
        assert handler._classifier is None

    def test_audit_log_starts_none(self, handler):
        """Audit log starts as None."""
        assert handler._audit_log is None


class TestWorkspaceHandlerWorkspaceRouting:
    """Tests for workspace route dispatch."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkspaceHandler(mock_server_context)

    def test_route_workspace_list_get(self, handler):
        """GET /api/workspaces routes to list handler."""
        mock_http = MagicMock()
        mock_http.command = "GET"

        result = handler._route_workspace("/api/workspaces", {}, mock_http, "GET")

        # Should return a result (auth error expected without user_store)
        assert result is not None

    def test_route_workspace_create_post(self, handler):
        """POST /api/workspaces routes to create handler."""
        mock_http = MagicMock()
        mock_http.command = "POST"

        result = handler._route_workspace("/api/workspaces", {}, mock_http, "POST")

        assert result is not None

    def test_route_workspace_get_by_id(self, handler):
        """GET /api/workspaces/{id} routes to get handler."""
        mock_http = MagicMock()
        mock_http.command = "GET"

        result = handler._route_workspace("/api/workspaces/ws_123", {}, mock_http, "GET")

        assert result is not None

    def test_route_workspace_delete(self, handler):
        """DELETE /api/workspaces/{id} routes to delete handler."""
        mock_http = MagicMock()
        mock_http.command = "DELETE"

        result = handler._route_workspace("/api/workspaces/ws_123", {}, mock_http, "DELETE")

        assert result is not None

    def test_route_workspace_add_member(self, handler):
        """POST /api/workspaces/{id}/members routes to add member."""
        mock_http = MagicMock()
        mock_http.command = "POST"

        result = handler._route_workspace("/api/workspaces/ws_123/members", {}, mock_http, "POST")

        assert result is not None

    def test_route_workspace_remove_member(self, handler):
        """DELETE /api/workspaces/{id}/members/{user_id} routes to remove member."""
        mock_http = MagicMock()
        mock_http.command = "DELETE"

        result = handler._route_workspace(
            "/api/workspaces/ws_123/members/user456", {}, mock_http, "DELETE"
        )

        assert result is not None

    def test_route_workspace_unknown_returns_404(self, handler):
        """Unknown workspace path returns 404."""
        mock_http = MagicMock()
        mock_http.command = "GET"

        result = handler._route_workspace("/api/workspaces/ws_123/unknown", {}, mock_http, "GET")

        assert result is not None
        assert result.status_code == 404


class TestWorkspaceHandlerRetentionRouting:
    """Tests for retention route dispatch."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkspaceHandler(mock_server_context)

    def test_route_retention_list_policies(self, handler):
        """GET /api/retention/policies routes to list."""
        mock_http = MagicMock()

        result = handler._route_retention("/api/retention/policies", {}, mock_http, "GET")

        assert result is not None

    def test_route_retention_create_policy(self, handler):
        """POST /api/retention/policies routes to create."""
        mock_http = MagicMock()

        result = handler._route_retention("/api/retention/policies", {}, mock_http, "POST")

        assert result is not None

    def test_route_retention_get_policy(self, handler):
        """GET /api/retention/policies/{id} routes to get."""
        mock_http = MagicMock()

        result = handler._route_retention("/api/retention/policies/pol_123", {}, mock_http, "GET")

        assert result is not None

    def test_route_retention_update_policy(self, handler):
        """PUT /api/retention/policies/{id} routes to update."""
        mock_http = MagicMock()

        result = handler._route_retention("/api/retention/policies/pol_123", {}, mock_http, "PUT")

        assert result is not None

    def test_route_retention_delete_policy(self, handler):
        """DELETE /api/retention/policies/{id} routes to delete."""
        mock_http = MagicMock()

        result = handler._route_retention(
            "/api/retention/policies/pol_123", {}, mock_http, "DELETE"
        )

        assert result is not None

    def test_route_retention_execute_policy(self, handler):
        """POST /api/retention/policies/{id}/execute routes to execute."""
        mock_http = MagicMock()

        result = handler._route_retention(
            "/api/retention/policies/pol_123/execute", {}, mock_http, "POST"
        )

        assert result is not None

    def test_route_retention_expiring(self, handler):
        """GET /api/retention/expiring routes to expiring handler."""
        mock_http = MagicMock()

        result = handler._route_retention("/api/retention/expiring", {}, mock_http, "GET")

        assert result is not None


class TestWorkspaceHandlerClassifyRouting:
    """Tests for classification route dispatch."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkspaceHandler(mock_server_context)

    def test_route_classify_content(self, handler):
        """POST /api/classify routes to classify handler."""
        mock_http = MagicMock()

        result = handler._route_classify("/api/classify", {}, mock_http, "POST")

        assert result is not None

    def test_route_classify_get_level_policy(self, handler):
        """GET /api/classify/policy/{level} routes to level policy handler."""
        mock_http = MagicMock()

        result = handler._route_classify("/api/classify/policy/high", {}, mock_http, "GET")

        assert result is not None

    def test_route_classify_unknown_returns_404(self, handler):
        """Unknown classify path returns 404."""
        mock_http = MagicMock()

        result = handler._route_classify("/api/classify/unknown", {}, mock_http, "GET")

        assert result is not None
        assert result.status_code == 404


class TestWorkspaceHandlerAuditRouting:
    """Tests for audit route dispatch."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkspaceHandler(mock_server_context)

    def test_route_audit_entries(self, handler):
        """GET /api/audit/entries routes to query handler."""
        mock_http = MagicMock()

        result = handler._route_audit("/api/audit/entries", {}, mock_http, "GET")

        assert result is not None

    def test_route_audit_report(self, handler):
        """GET /api/audit/report routes to report handler."""
        mock_http = MagicMock()

        result = handler._route_audit("/api/audit/report", {}, mock_http, "GET")

        assert result is not None

    def test_route_audit_verify(self, handler):
        """GET /api/audit/verify routes to verify handler."""
        mock_http = MagicMock()

        result = handler._route_audit("/api/audit/verify", {}, mock_http, "GET")

        assert result is not None

    def test_route_audit_actor_history(self, handler):
        """GET /api/audit/actor/{id}/history routes to actor handler."""
        mock_http = MagicMock()

        result = handler._route_audit("/api/audit/actor/user123/history", {}, mock_http, "GET")

        assert result is not None

    def test_route_audit_resource_history(self, handler):
        """GET /api/audit/resource/{id}/history routes to resource handler."""
        mock_http = MagicMock()

        result = handler._route_audit("/api/audit/resource/doc123/history", {}, mock_http, "GET")

        assert result is not None

    def test_route_audit_denied(self, handler):
        """GET /api/audit/denied routes to denied handler."""
        mock_http = MagicMock()

        result = handler._route_audit("/api/audit/denied", {}, mock_http, "GET")

        assert result is not None


class TestWorkspaceHandlerHttpMethods:
    """Tests for HTTP method routing."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkspaceHandler(mock_server_context)

    def test_handle_post_delegates_to_handle(self, handler):
        """handle_post delegates to handle with POST method."""
        mock_http = MagicMock()
        mock_http.command = "POST"

        result = handler.handle_post("/api/workspaces", {}, mock_http)

        assert result is not None

    def test_handle_delete_delegates_to_handle(self, handler):
        """handle_delete delegates to handle with DELETE method."""
        mock_http = MagicMock()
        mock_http.command = "DELETE"

        result = handler.handle_delete("/api/workspaces/ws_123", {}, mock_http)

        assert result is not None

    def test_handle_put_delegates_to_handle(self, handler):
        """handle_put delegates to handle with PUT method."""
        mock_http = MagicMock()
        mock_http.command = "PUT"

        result = handler.handle_put("/api/retention/policies/pol_123", {}, mock_http)

        assert result is not None


class TestWorkspaceHandlerMainHandle:
    """Tests for main handle method."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkspaceHandler(mock_server_context)

    def test_handle_routes_workspaces(self, handler):
        """Handle routes workspace paths correctly."""
        mock_http = MagicMock()
        mock_http.command = "GET"

        result = handler.handle("/api/workspaces", {}, mock_http)

        assert result is not None

    def test_handle_routes_retention(self, handler):
        """Handle routes retention paths correctly."""
        mock_http = MagicMock()
        mock_http.command = "GET"

        result = handler.handle("/api/retention/policies", {}, mock_http)

        assert result is not None

    def test_handle_routes_classify(self, handler):
        """Handle routes classify paths correctly."""
        mock_http = MagicMock()
        mock_http.command = "POST"

        result = handler.handle("/api/classify", {}, mock_http)

        assert result is not None

    def test_handle_routes_audit(self, handler):
        """Handle routes audit paths correctly."""
        mock_http = MagicMock()
        mock_http.command = "GET"

        result = handler.handle("/api/audit/entries", {}, mock_http)

        assert result is not None

    def test_handle_returns_none_for_unknown(self, handler):
        """Handle returns None for unknown paths."""
        mock_http = MagicMock()
        mock_http.command = "GET"

        result = handler.handle("/api/other", {}, mock_http)

        assert result is None


# ===========================================================================
# RBAC Tests
# ===========================================================================


from dataclasses import dataclass
from unittest.mock import patch


@dataclass
class MockPermissionDecision:
    """Mock RBAC permission decision."""

    allowed: bool = True
    reason: str = "Allowed by test"


@dataclass
class MockAuthContext:
    """Mock auth context for testing."""

    is_authenticated: bool = True
    user_id: str = "user-123"
    org_id: str = "org-123"


@dataclass
class MockUser:
    """Mock user for testing."""

    id: str = "user-123"
    role: str = "admin"
    org_id: str = "org-123"


def mock_check_permission_allowed(*args, **kwargs):
    """Mock check_permission that always allows."""
    return MockPermissionDecision(allowed=True)


def mock_check_permission_denied(*args, **kwargs):
    """Mock check_permission that always denies."""
    return MockPermissionDecision(allowed=False, reason="Permission denied by test")


class TestWorkspaceRBAC:
    """Tests for RBAC permission checks in WorkspaceHandler."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkspaceHandler(mock_server_context)

    def test_rbac_helper_methods_exist(self, handler):
        """Handler should have RBAC helper methods."""
        assert hasattr(handler, "_check_rbac_permission")
        assert hasattr(handler, "_get_auth_context")

    @patch("aragora.server.handlers.workspace.RBAC_AVAILABLE", False)
    def test_permission_check_without_rbac(self, handler):
        """Permission check should pass when RBAC not available."""
        mock_http = MagicMock()
        auth_ctx = MockAuthContext()

        result = handler._check_rbac_permission(mock_http, "workspaces.create", auth_ctx)
        assert result is None  # None means allowed

    @patch("aragora.server.handlers.workspace.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.workspace.check_permission", mock_check_permission_allowed)
    def test_permission_check_allowed(self, handler):
        """Permission check should pass when RBAC allows."""
        mock_http = MagicMock()
        auth_ctx = MockAuthContext()
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = MockUser()
        handler.ctx["user_store"] = mock_user_store

        result = handler._check_rbac_permission(mock_http, "workspaces.create", auth_ctx)
        assert result is None  # None means allowed

    @patch("aragora.server.handlers.workspace.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.workspace.check_permission", mock_check_permission_denied)
    def test_permission_check_denied(self, handler):
        """Permission check should return error when RBAC denies."""
        mock_http = MagicMock()
        auth_ctx = MockAuthContext()
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = MockUser(role="viewer")
        handler.ctx["user_store"] = mock_user_store

        result = handler._check_rbac_permission(mock_http, "workspaces.create", auth_ctx)
        assert result is not None
        assert result.status_code == 403

    @patch("aragora.server.handlers.workspace.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.workspace.check_permission", mock_check_permission_denied)
    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_create_workspace_rbac_denied(self, mock_extract, handler):
        """Create workspace should deny when RBAC denies."""
        mock_extract.return_value = MockAuthContext()
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = MockUser()
        handler.ctx["user_store"] = mock_user_store
        mock_http = MagicMock()

        result = handler._handle_create_workspace(mock_http)
        assert result.status_code == 403

    @patch("aragora.server.handlers.workspace.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.workspace.check_permission", mock_check_permission_denied)
    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_delete_workspace_rbac_denied(self, mock_extract, handler):
        """Delete workspace should deny when RBAC denies."""
        mock_extract.return_value = MockAuthContext()
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = MockUser()
        handler.ctx["user_store"] = mock_user_store
        mock_http = MagicMock()

        result = handler._handle_delete_workspace(mock_http, "ws-123")
        assert result.status_code == 403

    @patch("aragora.server.handlers.workspace.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.workspace.check_permission", mock_check_permission_denied)
    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_add_member_rbac_denied(self, mock_extract, handler):
        """Add member should deny when RBAC denies."""
        mock_extract.return_value = MockAuthContext()
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = MockUser()
        handler.ctx["user_store"] = mock_user_store
        mock_http = MagicMock()

        result = handler._handle_add_member(mock_http, "ws-123")
        assert result.status_code == 403

    @patch("aragora.server.handlers.workspace.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.workspace.check_permission", mock_check_permission_denied)
    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_remove_member_rbac_denied(self, mock_extract, handler):
        """Remove member should deny when RBAC denies."""
        mock_extract.return_value = MockAuthContext()
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = MockUser()
        handler.ctx["user_store"] = mock_user_store
        mock_http = MagicMock()

        result = handler._handle_remove_member(mock_http, "ws-123", "user-456")
        assert result.status_code == 403

    @patch("aragora.server.handlers.workspace.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.workspace.check_permission", mock_check_permission_denied)
    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_create_policy_rbac_denied(self, mock_extract, handler):
        """Create retention policy should deny when RBAC denies."""
        mock_extract.return_value = MockAuthContext()
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = MockUser()
        handler.ctx["user_store"] = mock_user_store
        mock_http = MagicMock()

        result = handler._handle_create_policy(mock_http)
        assert result.status_code == 403

    @patch("aragora.server.handlers.workspace.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.workspace.check_permission", mock_check_permission_denied)
    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_update_policy_rbac_denied(self, mock_extract, handler):
        """Update retention policy should deny when RBAC denies."""
        mock_extract.return_value = MockAuthContext()
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = MockUser()
        handler.ctx["user_store"] = mock_user_store
        mock_http = MagicMock()

        result = handler._handle_update_policy(mock_http, "pol-123")
        assert result.status_code == 403

    @patch("aragora.server.handlers.workspace.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.workspace.check_permission", mock_check_permission_denied)
    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_delete_policy_rbac_denied(self, mock_extract, handler):
        """Delete retention policy should deny when RBAC denies."""
        mock_extract.return_value = MockAuthContext()
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = MockUser()
        handler.ctx["user_store"] = mock_user_store
        mock_http = MagicMock()

        result = handler._handle_delete_policy(mock_http, "pol-123")
        assert result.status_code == 403

    @patch("aragora.server.handlers.workspace.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.workspace.check_permission", mock_check_permission_denied)
    @patch("aragora.server.handlers.workspace.extract_user_from_request")
    def test_execute_policy_rbac_denied(self, mock_extract, handler):
        """Execute retention policy should deny when RBAC denies."""
        mock_extract.return_value = MockAuthContext()
        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id.return_value = MockUser()
        handler.ctx["user_store"] = mock_user_store
        mock_http = MagicMock()

        result = handler._handle_execute_policy(mock_http, "pol-123", {})
        assert result.status_code == 403
