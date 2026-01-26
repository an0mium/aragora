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
        assert handler.can_handle("/api/v1/workspaces")

    def test_can_handle_workspace_by_id(self, handler):
        """Handler can handle workspace by ID."""
        assert handler.can_handle("/api/v1/workspaces/ws_123")

    def test_can_handle_workspace_members(self, handler):
        """Handler can handle workspace members."""
        assert handler.can_handle("/api/v1/workspaces/ws_123/members")

    def test_can_handle_retention_policies(self, handler):
        """Handler can handle retention policies."""
        assert handler.can_handle("/api/v1/retention/policies")

    def test_can_handle_retention_expiring(self, handler):
        """Handler can handle retention expiring."""
        assert handler.can_handle("/api/v1/retention/expiring")

    def test_can_handle_classify(self, handler):
        """Handler can handle classify endpoint."""
        assert handler.can_handle("/api/v1/classify")

    def test_can_handle_classify_policy(self, handler):
        """Handler can handle classify policy by level."""
        assert handler.can_handle("/api/v1/classify/policy/high")

    def test_can_handle_audit_entries(self, handler):
        """Handler can handle audit entries."""
        assert handler.can_handle("/api/v1/audit/entries")

    def test_can_handle_audit_report(self, handler):
        """Handler can handle audit report."""
        assert handler.can_handle("/api/v1/audit/report")

    def test_can_handle_audit_verify(self, handler):
        """Handler can handle audit verify."""
        assert handler.can_handle("/api/v1/audit/verify")

    def test_can_handle_audit_actor(self, handler):
        """Handler can handle audit actor history."""
        assert handler.can_handle("/api/v1/audit/actor/user123/history")

    def test_can_handle_audit_resource(self, handler):
        """Handler can handle audit resource history."""
        assert handler.can_handle("/api/v1/audit/resource/doc123/history")

    def test_can_handle_audit_denied(self, handler):
        """Handler can handle audit denied attempts."""
        assert handler.can_handle("/api/v1/audit/denied")

    def test_cannot_handle_other_paths(self, handler):
        """Handler cannot handle unrelated paths."""
        assert not handler.can_handle("/api/v1/debates")
        assert not handler.can_handle("/api/v1/other")


class TestWorkspaceHandlerRoutesAttribute:
    """Tests for ROUTES class attribute."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkspaceHandler(mock_server_context)

    def test_routes_contains_workspaces(self, handler):
        """ROUTES contains workspaces."""
        assert "/api/v1/workspaces" in handler.ROUTES

    def test_routes_contains_retention(self, handler):
        """ROUTES contains retention endpoints."""
        assert "/api/v1/retention/policies" in handler.ROUTES
        assert "/api/v1/retention/expiring" in handler.ROUTES

    def test_routes_contains_classify(self, handler):
        """ROUTES contains classify."""
        assert "/api/v1/classify" in handler.ROUTES

    def test_routes_contains_audit_endpoints(self, handler):
        """ROUTES contains audit endpoints."""
        assert "/api/v1/audit/entries" in handler.ROUTES
        assert "/api/v1/audit/report" in handler.ROUTES
        assert "/api/v1/audit/verify" in handler.ROUTES


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
        """GET /api/v1/workspaces routes to list handler."""
        mock_http = MagicMock()
        mock_http.command = "GET"

        result = handler._route_workspace("/api/v1/workspaces", {}, mock_http, "GET")

        # Should return a result (auth error expected without user_store)
        assert result is not None

    def test_route_workspace_create_post(self, handler):
        """POST /api/v1/workspaces routes to create handler."""
        mock_http = MagicMock()
        mock_http.command = "POST"

        result = handler._route_workspace("/api/v1/workspaces", {}, mock_http, "POST")

        assert result is not None

    def test_route_workspace_get_by_id(self, handler):
        """GET /api/v1/workspaces/{id} routes to get handler."""
        mock_http = MagicMock()
        mock_http.command = "GET"

        result = handler._route_workspace("/api/v1/workspaces/ws_123", {}, mock_http, "GET")

        assert result is not None

    def test_route_workspace_delete(self, handler):
        """DELETE /api/v1/workspaces/{id} routes to delete handler."""
        mock_http = MagicMock()
        mock_http.command = "DELETE"

        result = handler._route_workspace("/api/v1/workspaces/ws_123", {}, mock_http, "DELETE")

        assert result is not None

    def test_route_workspace_add_member(self, handler):
        """POST /api/v1/workspaces/{id}/members routes to add member."""
        mock_http = MagicMock()
        mock_http.command = "POST"

        result = handler._route_workspace(
            "/api/v1/workspaces/ws_123/members", {}, mock_http, "POST"
        )

        assert result is not None

    def test_route_workspace_remove_member(self, handler):
        """DELETE /api/v1/workspaces/{id}/members/{user_id} routes to remove member."""
        mock_http = MagicMock()
        mock_http.command = "DELETE"

        result = handler._route_workspace(
            "/api/v1/workspaces/ws_123/members/user456", {}, mock_http, "DELETE"
        )

        assert result is not None

    def test_route_workspace_unknown_returns_404(self, handler):
        """Unknown workspace path returns 404."""
        mock_http = MagicMock()
        mock_http.command = "GET"

        result = handler._route_workspace("/api/v1/workspaces/ws_123/unknown", {}, mock_http, "GET")

        assert result is not None
        assert result.status_code == 404


class TestWorkspaceHandlerRetentionRouting:
    """Tests for retention route dispatch."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkspaceHandler(mock_server_context)

    def test_route_retention_list_policies(self, handler):
        """GET /api/v1/retention/policies routes to list."""
        mock_http = MagicMock()

        result = handler._route_retention("/api/v1/retention/policies", {}, mock_http, "GET")

        assert result is not None

    def test_route_retention_create_policy(self, handler):
        """POST /api/v1/retention/policies routes to create."""
        mock_http = MagicMock()

        result = handler._route_retention("/api/v1/retention/policies", {}, mock_http, "POST")

        assert result is not None

    def test_route_retention_get_policy(self, handler):
        """GET /api/v1/retention/policies/{id} routes to get."""
        mock_http = MagicMock()

        result = handler._route_retention(
            "/api/v1/retention/policies/pol_123", {}, mock_http, "GET"
        )

        assert result is not None

    def test_route_retention_update_policy(self, handler):
        """PUT /api/v1/retention/policies/{id} routes to update."""
        mock_http = MagicMock()

        result = handler._route_retention(
            "/api/v1/retention/policies/pol_123", {}, mock_http, "PUT"
        )

        assert result is not None

    def test_route_retention_delete_policy(self, handler):
        """DELETE /api/v1/retention/policies/{id} routes to delete."""
        mock_http = MagicMock()

        result = handler._route_retention(
            "/api/v1/retention/policies/pol_123", {}, mock_http, "DELETE"
        )

        assert result is not None

    def test_route_retention_execute_policy(self, handler):
        """POST /api/v1/retention/policies/{id}/execute routes to execute."""
        mock_http = MagicMock()

        result = handler._route_retention(
            "/api/v1/retention/policies/pol_123/execute", {}, mock_http, "POST"
        )

        assert result is not None

    def test_route_retention_expiring(self, handler):
        """GET /api/v1/retention/expiring routes to expiring handler."""
        mock_http = MagicMock()

        result = handler._route_retention("/api/v1/retention/expiring", {}, mock_http, "GET")

        assert result is not None


class TestWorkspaceHandlerClassifyRouting:
    """Tests for classification route dispatch."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkspaceHandler(mock_server_context)

    def test_route_classify_content(self, handler):
        """POST /api/v1/classify routes to classify handler."""
        mock_http = MagicMock()

        result = handler._route_classify("/api/v1/classify", {}, mock_http, "POST")

        assert result is not None

    def test_route_classify_get_level_policy(self, handler):
        """GET /api/v1/classify/policy/{level} routes to level policy handler."""
        mock_http = MagicMock()

        result = handler._route_classify("/api/v1/classify/policy/high", {}, mock_http, "GET")

        assert result is not None

    def test_route_classify_unknown_returns_404(self, handler):
        """Unknown classify path returns 404."""
        mock_http = MagicMock()

        result = handler._route_classify("/api/v1/classify/unknown", {}, mock_http, "GET")

        assert result is not None
        assert result.status_code == 404


class TestWorkspaceHandlerAuditRouting:
    """Tests for audit route dispatch."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkspaceHandler(mock_server_context)

    def test_route_audit_entries(self, handler):
        """GET /api/v1/audit/entries routes to query handler."""
        mock_http = MagicMock()

        result = handler._route_audit("/api/v1/audit/entries", {}, mock_http, "GET")

        assert result is not None

    def test_route_audit_report(self, handler):
        """GET /api/v1/audit/report routes to report handler."""
        mock_http = MagicMock()

        result = handler._route_audit("/api/v1/audit/report", {}, mock_http, "GET")

        assert result is not None

    def test_route_audit_verify(self, handler):
        """GET /api/v1/audit/verify routes to verify handler."""
        mock_http = MagicMock()

        result = handler._route_audit("/api/v1/audit/verify", {}, mock_http, "GET")

        assert result is not None

    def test_route_audit_actor_history(self, handler):
        """GET /api/v1/audit/actor/{id}/history routes to actor handler."""
        mock_http = MagicMock()

        result = handler._route_audit("/api/v1/audit/actor/user123/history", {}, mock_http, "GET")

        assert result is not None

    def test_route_audit_resource_history(self, handler):
        """GET /api/v1/audit/resource/{id}/history routes to resource handler."""
        mock_http = MagicMock()

        result = handler._route_audit("/api/v1/audit/resource/doc123/history", {}, mock_http, "GET")

        assert result is not None

    def test_route_audit_denied(self, handler):
        """GET /api/v1/audit/denied routes to denied handler."""
        mock_http = MagicMock()

        result = handler._route_audit("/api/v1/audit/denied", {}, mock_http, "GET")

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

        result = handler.handle_post("/api/v1/workspaces", {}, mock_http)

        assert result is not None

    def test_handle_delete_delegates_to_handle(self, handler):
        """handle_delete delegates to handle with DELETE method."""
        mock_http = MagicMock()
        mock_http.command = "DELETE"

        result = handler.handle_delete("/api/v1/workspaces/ws_123", {}, mock_http)

        assert result is not None

    def test_handle_put_delegates_to_handle(self, handler):
        """handle_put delegates to handle with PUT method."""
        mock_http = MagicMock()
        mock_http.command = "PUT"

        result = handler.handle_put("/api/v1/retention/policies/pol_123", {}, mock_http)

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

        result = handler.handle("/api/v1/workspaces", {}, mock_http)

        assert result is not None

    def test_handle_routes_retention(self, handler):
        """Handle routes retention paths correctly."""
        mock_http = MagicMock()
        mock_http.command = "GET"

        result = handler.handle("/api/v1/retention/policies", {}, mock_http)

        assert result is not None

    def test_handle_routes_classify(self, handler):
        """Handle routes classify paths correctly."""
        mock_http = MagicMock()
        mock_http.command = "POST"

        result = handler.handle("/api/v1/classify", {}, mock_http)

        assert result is not None

    def test_handle_routes_audit(self, handler):
        """Handle routes audit paths correctly."""
        mock_http = MagicMock()
        mock_http.command = "GET"

        result = handler.handle("/api/v1/audit/entries", {}, mock_http)

        assert result is not None

    def test_handle_returns_none_for_unknown(self, handler):
        """Handle returns None for unknown paths."""
        mock_http = MagicMock()
        mock_http.command = "GET"

        result = handler.handle("/api/v1/other", {}, mock_http)

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


# ===========================================================================
# Additional Tests based on patterns from test_admin_health_handler.py
# and test_admin_billing_handler.py
# ===========================================================================


class TestWorkspaceHandlerResourceType:
    """Tests for handler resource type configuration."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkspaceHandler(mock_server_context)

    def test_resource_type_is_workspace(self, handler):
        """Handler resource type is set to 'workspace'."""
        assert handler.RESOURCE_TYPE == "workspace"


class TestWorkspaceHandlerContextMethods:
    """Tests for context accessor methods."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkspaceHandler(mock_server_context)

    def test_get_user_store_returns_from_context(self, handler):
        """_get_user_store returns user_store from context."""
        mock_store = MagicMock()
        handler.ctx["user_store"] = mock_store

        result = handler._get_user_store()

        assert result is mock_store

    def test_get_user_store_returns_none_when_missing(self, handler):
        """_get_user_store returns None when not in context."""
        result = handler._get_user_store()

        assert result is None

    def test_get_isolation_manager_returns_same_instance(self, handler):
        """_get_isolation_manager returns same instance on repeated calls."""
        first = handler._get_isolation_manager()
        second = handler._get_isolation_manager()

        assert first is second
        assert first is not None

    def test_get_retention_manager_returns_same_instance(self, handler):
        """_get_retention_manager returns same instance on repeated calls."""
        first = handler._get_retention_manager()
        second = handler._get_retention_manager()

        assert first is second
        assert first is not None

    def test_get_classifier_returns_same_instance(self, handler):
        """_get_classifier returns same instance on repeated calls."""
        first = handler._get_classifier()
        second = handler._get_classifier()

        assert first is second
        assert first is not None

    @patch("aragora.server.handlers.workspace.PrivacyAuditLog")
    def test_get_audit_log_returns_same_instance(self, mock_audit_log_class, handler):
        """_get_audit_log returns same instance on repeated calls."""
        mock_audit_instance = MagicMock()
        mock_audit_log_class.return_value = mock_audit_instance

        first = handler._get_audit_log()
        second = handler._get_audit_log()

        assert first is second
        assert first is not None


class TestWorkspaceHandlerResponseFormat:
    """Tests for response format validation."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkspaceHandler(mock_server_context)

    def test_workspace_route_response_is_json(self, handler):
        """Workspace route returns JSON response."""
        mock_http = MagicMock()
        mock_http.command = "GET"

        result = handler._route_workspace("/api/v1/workspaces", {}, mock_http, "GET")

        assert result is not None
        assert result.content_type == "application/json"

    def test_retention_route_response_is_json(self, handler):
        """Retention route returns JSON response."""
        mock_http = MagicMock()
        mock_http.command = "GET"

        result = handler._route_retention("/api/v1/retention/policies", {}, mock_http, "GET")

        assert result is not None
        assert result.content_type == "application/json"

    def test_classify_route_response_is_json(self, handler):
        """Classify route returns JSON response."""
        mock_http = MagicMock()
        mock_http.command = "POST"

        result = handler._route_classify("/api/v1/classify", {}, mock_http, "POST")

        assert result is not None
        assert result.content_type == "application/json"

    def test_audit_route_response_is_json(self, handler):
        """Audit route returns JSON response."""
        mock_http = MagicMock()
        mock_http.command = "GET"

        result = handler._route_audit("/api/v1/audit/entries", {}, mock_http, "GET")

        assert result is not None
        assert result.content_type == "application/json"

    def test_404_response_is_json(self, handler):
        """404 not found response is JSON."""
        mock_http = MagicMock()
        mock_http.command = "GET"

        result = handler._route_workspace("/api/v1/workspaces/ws_123/invalid", {}, mock_http, "GET")

        assert result is not None
        assert result.status_code == 404
        assert result.content_type == "application/json"


class TestWorkspaceHandlerRoutesCount:
    """Tests for minimum routes count."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkspaceHandler(mock_server_context)

    def test_routes_count_minimum(self, handler):
        """ROUTES has expected minimum number of endpoints."""
        # At least 11 routes based on handler inspection
        assert len(handler.ROUTES) >= 11


class TestWorkspaceHandlerImport:
    """Tests for importing WorkspaceHandler."""

    def test_can_import_handler(self):
        """WorkspaceHandler can be imported."""
        from aragora.server.handlers.workspace import WorkspaceHandler

        assert WorkspaceHandler is not None

    def test_handler_in_all(self):
        """WorkspaceHandler is in __all__."""
        from aragora.server.handlers import workspace

        assert "WorkspaceHandler" in workspace.__all__


class TestWorkspaceHandlerMethodExtraction:
    """Tests for method extraction from handler.command."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkspaceHandler(mock_server_context)

    def test_extracts_method_from_command_attribute(self, handler):
        """Handler extracts method from http.command attribute."""
        mock_http = MagicMock()
        mock_http.command = "GET"

        # Don't pass method, let it be extracted from command
        result = handler.handle("/api/v1/workspaces", {}, mock_http)

        # Should work with extracted method
        assert result is not None


class TestWorkspaceHandlerProfilesEndpoint:
    """Tests for workspace profiles endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkspaceHandler(mock_server_context)

    def test_can_handle_profiles(self, handler):
        """Handler can handle profiles endpoint."""
        assert handler.can_handle("/api/v1/workspaces/profiles")

    def test_routes_contains_profiles(self, handler):
        """ROUTES contains workspaces profiles endpoint."""
        assert "/api/v1/workspaces/profiles" in handler.ROUTES

    def test_profiles_route_returns_result(self, handler):
        """Profiles route returns a result."""
        mock_http = MagicMock()
        mock_http.command = "GET"

        result = handler._route_workspace("/api/v1/workspaces/profiles", {}, mock_http, "GET")

        assert result is not None


class TestWorkspaceHandlerAuditEndpoints:
    """Tests for all audit endpoints in ROUTES."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkspaceHandler(mock_server_context)

    def test_routes_contains_audit_actor(self, handler):
        """ROUTES contains audit actor endpoint."""
        assert "/api/v1/audit/actor" in handler.ROUTES

    def test_routes_contains_audit_resource(self, handler):
        """ROUTES contains audit resource endpoint."""
        assert "/api/v1/audit/resource" in handler.ROUTES

    def test_routes_contains_audit_denied(self, handler):
        """ROUTES contains audit denied endpoint."""
        assert "/api/v1/audit/denied" in handler.ROUTES


class TestWorkspaceHandlerCanHandleExtended:
    """Extended tests for can_handle matching patterns."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkspaceHandler(mock_server_context)

    def test_can_handle_workspaces_with_uuid(self, handler):
        """Handler can handle workspaces with UUID-style ID."""
        assert handler.can_handle("/api/v1/workspaces/550e8400-e29b-41d4-a716-446655440000")

    def test_can_handle_retention_policy_execute(self, handler):
        """Handler can handle retention policy execute endpoint."""
        assert handler.can_handle("/api/v1/retention/policies/pol-123/execute")

    def test_can_handle_workspace_member_role(self, handler):
        """Handler can handle workspace member role endpoint."""
        assert handler.can_handle("/api/v1/workspaces/ws-123/members/user-456/role")

    def test_cannot_handle_debates(self, handler):
        """Handler cannot handle debates path."""
        assert not handler.can_handle("/api/v1/debates")

    def test_cannot_handle_health(self, handler):
        """Handler cannot handle health path."""
        assert not handler.can_handle("/api/v1/health")

    def test_cannot_handle_billing(self, handler):
        """Handler cannot handle billing path."""
        assert not handler.can_handle("/api/v1/billing")

    def test_cannot_handle_root(self, handler):
        """Handler cannot handle root path."""
        assert not handler.can_handle("/")
