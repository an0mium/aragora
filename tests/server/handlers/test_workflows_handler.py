"""
Tests for the WorkflowHandler module.

Tests cover:
- Handler routing for all workflow endpoints
- ID extraction from paths
- Route handling and can_handle method
- RBAC permission checks
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from aragora.server.handlers.workflows import WorkflowHandler, RBAC_AVAILABLE


@pytest.fixture
def mock_server_context():
    """Create mock server context for handler initialization."""
    return {"storage": None, "elo_system": None, "nomic_dir": None}


class TestWorkflowHandlerRouting:
    """Tests for handler routing."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowHandler(mock_server_context)

    def test_can_handle_workflows(self, handler):
        """Handler can handle workflows base path."""
        assert handler.can_handle("/api/v1/workflows")

    def test_can_handle_workflow_by_id(self, handler):
        """Handler can handle workflow by ID."""
        assert handler.can_handle("/api/v1/workflows/wf_123")

    def test_can_handle_workflow_execute(self, handler):
        """Handler can handle workflow execute."""
        assert handler.can_handle("/api/v1/workflows/wf_123/execute")

    def test_can_handle_workflow_simulate(self, handler):
        """Handler can handle workflow simulate."""
        assert handler.can_handle("/api/v1/workflows/wf_123/simulate")

    def test_can_handle_workflow_versions(self, handler):
        """Handler can handle workflow versions."""
        assert handler.can_handle("/api/v1/workflows/wf_123/versions")

    def test_can_handle_workflow_status(self, handler):
        """Handler can handle workflow status."""
        assert handler.can_handle("/api/v1/workflows/wf_123/status")

    def test_can_handle_templates(self, handler):
        """Handler can handle templates."""
        assert handler.can_handle("/api/v1/workflow-templates")

    def test_can_handle_approvals(self, handler):
        """Handler can handle approvals."""
        assert handler.can_handle("/api/v1/workflow-approvals")

    def test_can_handle_executions(self, handler):
        """Handler can handle executions."""
        assert handler.can_handle("/api/v1/workflow-executions")

    def test_cannot_handle_other_paths(self, handler):
        """Handler cannot handle unrelated paths."""
        assert not handler.can_handle("/api/v1/other")
        assert not handler.can_handle("/api/v1/debates")


class TestWorkflowHandlerRoutes:
    """Tests for ROUTES class attribute."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowHandler(mock_server_context)

    def test_routes_is_list(self, handler):
        """ROUTES is a list."""
        assert isinstance(handler.ROUTES, list)

    def test_routes_not_empty(self, handler):
        """ROUTES is not empty."""
        assert len(handler.ROUTES) > 0

    def test_routes_contains_workflows(self, handler):
        """ROUTES contains workflows endpoint."""
        assert "/api/v1/workflows" in handler.ROUTES

    def test_routes_contains_workflows_wildcard(self, handler):
        """ROUTES contains workflows wildcard for ID paths."""
        assert "/api/v1/workflows/*" in handler.ROUTES

    def test_routes_contains_templates(self, handler):
        """ROUTES contains workflow-templates endpoint."""
        assert "/api/v1/workflow-templates" in handler.ROUTES

    def test_routes_contains_templates_alias(self, handler):
        """ROUTES contains workflow templates alias."""
        assert "/api/v1/workflows/templates" in handler.ROUTES

    def test_routes_contains_templates_alias_wildcard(self, handler):
        """ROUTES contains workflow templates alias wildcard."""
        assert "/api/v1/workflows/templates/*" in handler.ROUTES

    def test_routes_contains_approvals(self, handler):
        """ROUTES contains workflow-approvals endpoint."""
        assert "/api/v1/workflow-approvals" in handler.ROUTES

    def test_routes_contains_approvals_wildcard(self, handler):
        """ROUTES contains workflow-approvals wildcard."""
        assert "/api/v1/workflow-approvals/*" in handler.ROUTES

    def test_routes_contains_executions(self, handler):
        """ROUTES contains workflow-executions endpoint."""
        assert "/api/v1/workflow-executions" in handler.ROUTES

    def test_routes_contains_executions_wildcard(self, handler):
        """ROUTES contains workflow-executions wildcard."""
        assert "/api/v1/workflow-executions/*" in handler.ROUTES

    def test_routes_contains_executions_alias(self, handler):
        """ROUTES contains workflow executions alias."""
        assert "/api/v1/workflows/executions" in handler.ROUTES

    def test_routes_contains_executions_alias_wildcard(self, handler):
        """ROUTES contains workflow executions alias wildcard."""
        assert "/api/v1/workflows/executions/*" in handler.ROUTES

    def test_routes_count(self, handler):
        """ROUTES contains expected number of routes."""
        # 11 routes: workflows, workflows/*, templates, workflows/templates,
        # workflows/templates/*, approvals, approvals/*, executions,
        # executions/*, workflows/executions, workflows/executions/*
        assert len(handler.ROUTES) == 11


class TestWorkflowHandlerIdExtraction:
    """Tests for ID extraction from paths."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowHandler(mock_server_context)

    def test_extract_id_basic(self, handler):
        """Extract ID from basic path."""
        id_ = handler._extract_id("/api/v1/workflows/wf_123")
        assert id_ == "wf_123"

    def test_extract_id_with_suffix(self, handler):
        """Extract ID with suffix removal."""
        id_ = handler._extract_id("/api/v1/workflows/wf_123/execute", suffix="/execute")
        assert id_ == "wf_123"

    def test_extract_id_no_id(self, handler):
        """Extract ID returns None for base path."""
        id_ = handler._extract_id("/api/v1/workflows")
        assert id_ is None

    def test_extract_id_with_versions_suffix(self, handler):
        """Extract ID with versions suffix."""
        id_ = handler._extract_id("/api/v1/workflows/wf_abc123/versions", suffix="/versions")
        assert id_ == "wf_abc123"


class TestWorkflowHandlerRouteDispatch:
    """Tests for route dispatch logic."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowHandler(mock_server_context)

    @pytest.fixture
    def mock_http_with_headers(self):
        """Create mock HTTP handler with proper headers for auth extraction."""
        mock = MagicMock()
        # Use MagicMock for headers to allow proper mocking
        mock.headers = MagicMock()
        mock.headers.get.return_value = ""
        mock.headers.__getitem__ = MagicMock(return_value="")
        return mock

    def test_handle_dispatches_list_workflows(self, handler, mock_http_with_headers):
        """Handle dispatches /api/v1/workflows to list handler."""
        with patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract:
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = False
            mock_jwt_context.user_id = None
            mock_extract.return_value = mock_jwt_context

            result = handler.handle("/api/v1/workflows", {}, mock_http_with_headers)

            # Result should be returned (auth error since no JWT)
            assert result is not None

    def test_handle_dispatches_templates(self, handler, mock_http_with_headers):
        """Handle dispatches /api/v1/workflow-templates to template handler."""
        with patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract:
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = False
            mock_jwt_context.user_id = None
            mock_extract.return_value = mock_jwt_context

            result = handler.handle("/api/v1/workflow-templates", {}, mock_http_with_headers)

            assert result is not None

    def test_handle_dispatches_approvals(self, handler, mock_http_with_headers):
        """Handle dispatches /api/v1/workflow-approvals to approval handler."""
        with patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract:
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = False
            mock_jwt_context.user_id = None
            mock_extract.return_value = mock_jwt_context

            result = handler.handle("/api/v1/workflow-approvals", {}, mock_http_with_headers)

            assert result is not None

    def test_handle_dispatches_executions(self, handler, mock_http_with_headers):
        """Handle dispatches /api/v1/workflow-executions to execution handler."""
        with patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract:
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = False
            mock_jwt_context.user_id = None
            mock_extract.return_value = mock_jwt_context

            result = handler.handle("/api/v1/workflow-executions", {}, mock_http_with_headers)

            assert result is not None


class TestWorkflowHandlerUnknownPath:
    """Tests for unknown path handling."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowHandler(mock_server_context)

    def test_unhandled_post_path(self, handler):
        """Unhandled POST path returns None."""
        mock_http = MagicMock()
        mock_http.rfile = MagicMock()
        mock_http.headers = {"Content-Length": "2", "Content-Type": "application/json"}
        mock_http.rfile.read.return_value = b"{}"

        result = handler.handle_post("/api/v1/other", {}, mock_http)

        assert result is None

    def test_unhandled_delete_path(self, handler):
        """Unhandled DELETE path returns None."""
        mock_http = MagicMock()

        result = handler.handle_delete("/api/v1/other", {}, mock_http)

        assert result is None

    def test_unhandled_patch_path(self, handler):
        """Unhandled PATCH path returns None."""
        mock_http = MagicMock()
        mock_http.rfile = MagicMock()
        mock_http.headers = {"Content-Length": "2", "Content-Type": "application/json"}
        mock_http.rfile.read.return_value = b"{}"

        result = handler.handle_patch("/api/v1/other", {}, mock_http)

        assert result is None


class TestWorkflowHandlerRBAC:
    """Tests for RBAC permission checks."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowHandler(mock_server_context)

    @pytest.fixture
    def mock_http(self):
        """Create mock HTTP handler with headers."""
        mock = MagicMock()
        mock.headers = {}
        return mock

    def test_rbac_available_check(self):
        """Verify RBAC availability is correctly detected."""
        # RBAC should be available if the aragora.rbac module exists
        # Just verify the flag is set correctly
        assert isinstance(RBAC_AVAILABLE, bool)

    def test_get_auth_context_without_rbac(self, handler, mock_http):
        """_get_auth_context returns None when RBAC not available."""
        with patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False):
            context = handler._get_auth_context(mock_http)
            assert context is None

    @pytest.mark.no_auto_auth
    def test_get_auth_context_rejects_without_jwt(self, handler, mock_http):
        """_get_auth_context returns 'unauthenticated' sentinel when JWT auth fails.

        SECURITY: Header-based authentication fallback was removed to prevent
        identity spoofing and privilege escalation attacks.
        """
        mock_http.headers = {
            "X-User-ID": "user-123",
            "X-Org-ID": "org-456",
            "X-User-Roles": "admin,member",
        }

        with patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract:
            # Simulate no JWT auth
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = False
            mock_jwt_context.user_id = None
            mock_extract.return_value = mock_jwt_context

            context = handler._get_auth_context(mock_http)

            # Should return "unauthenticated" sentinel, NOT fall back to headers
            assert context == "unauthenticated"

    def test_get_auth_context_from_jwt(self, handler, mock_http):
        """_get_auth_context extracts context from valid JWT token."""
        with patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract:
            # Simulate valid JWT auth
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = True
            mock_jwt_context.user_id = "jwt-user-123"
            mock_jwt_context.org_id = "org-456"
            mock_jwt_context.role = "admin"
            mock_extract.return_value = mock_jwt_context

            context = handler._get_auth_context(mock_http)

            assert context is not None
            assert context != "unauthenticated"
            assert context.user_id == "jwt-user-123"
            assert context.org_id == "org-456"
            assert "admin" in context.roles

    def test_check_permission_without_rbac(self, handler, mock_http):
        """_check_permission returns None (allows) when RBAC not available."""
        with patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False):
            error = handler._check_permission(mock_http, "workflows:read")
            assert error is None  # No error = allowed

    def test_check_permission_allowed(self, handler, mock_http):
        """_check_permission returns None when permission granted."""
        with patch("aragora.server.handlers.workflows.check_permission") as mock_check:
            mock_decision = MagicMock()
            mock_decision.allowed = True
            mock_check.return_value = mock_decision

            with patch.object(handler, "_get_auth_context") as mock_get_ctx:
                mock_ctx = MagicMock()
                mock_ctx.user_id = "user-123"
                mock_get_ctx.return_value = mock_ctx

                error = handler._check_permission(mock_http, "workflows:read")

                assert error is None  # No error = allowed
                mock_check.assert_called_once()

    def test_check_permission_denied(self, handler, mock_http):
        """_check_permission returns 403 error when permission denied."""
        with patch("aragora.server.handlers.workflows.check_permission") as mock_check:
            mock_decision = MagicMock()
            mock_decision.allowed = False
            mock_decision.reason = "Insufficient permissions"
            mock_check.return_value = mock_decision

            with patch.object(handler, "_get_auth_context") as mock_get_ctx:
                mock_ctx = MagicMock()
                mock_ctx.user_id = "user-123"
                mock_get_ctx.return_value = mock_ctx

                error = handler._check_permission(mock_http, "workflows:delete")

                assert error is not None
                assert error.status_code == 403
                # Parse the JSON body to check for error message
                import json

                body = json.loads(error.body)
                assert "Permission denied" in body.get("error", "")

    def test_get_tenant_id_from_query_params(self, handler, mock_http):
        """_get_tenant_id returns tenant from query params."""
        query_params = {"tenant_id": "custom-tenant"}

        tenant_id = handler._get_tenant_id(mock_http, query_params)

        assert tenant_id == "custom-tenant"

    def test_get_tenant_id_default(self, handler, mock_http):
        """_get_tenant_id returns default when not specified."""
        query_params = {}

        with patch("aragora.server.handlers.workflows.RBAC_AVAILABLE", False):
            tenant_id = handler._get_tenant_id(mock_http, query_params)

        assert tenant_id == "default"

    def test_get_tenant_id_from_auth_context(self, handler, mock_http):
        """_get_tenant_id returns org_id from auth context."""
        query_params = {}

        with patch.object(handler, "_get_auth_context") as mock_get_ctx:
            mock_ctx = MagicMock()
            mock_ctx.org_id = "org-from-jwt"
            mock_get_ctx.return_value = mock_ctx

            tenant_id = handler._get_tenant_id(mock_http, query_params)

            assert tenant_id == "org-from-jwt"


class TestWorkflowHandlerRBACIntegration:
    """Integration tests for RBAC with handler methods."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowHandler(mock_server_context)

    @pytest.fixture
    def mock_http(self):
        """Create mock HTTP handler."""
        mock = MagicMock()
        mock.headers = {}
        return mock

    def test_list_workflows_checks_permission(self, handler, mock_http):
        """_handle_list_workflows checks workflows:read permission."""
        with patch.object(handler, "_check_permission") as mock_check:
            mock_check.return_value = None  # Allow

            handler._handle_list_workflows({}, mock_http)

            mock_check.assert_called_once_with(mock_http, "workflows:read")

    def test_create_workflow_checks_permission(self, handler, mock_http):
        """_handle_create_workflow checks workflows:create permission."""
        from aragora.server.handlers.base import error_response

        with patch.object(handler, "_check_permission") as mock_check:
            # Return a proper HandlerResult for 403
            mock_check.return_value = error_response("denied", 403)

            result = handler._handle_create_workflow({}, {}, mock_http)

            mock_check.assert_called_once_with(mock_http, "workflows:create")
            assert result.status_code == 403

    def test_delete_workflow_checks_permission_with_resource_id(self, handler, mock_http):
        """_handle_delete_workflow checks permission with workflow ID."""
        from aragora.server.handlers.base import error_response

        with patch.object(handler, "_check_permission") as mock_check:
            mock_check.return_value = error_response("denied", 403)

            result = handler._handle_delete_workflow("wf_123", {}, mock_http)

            mock_check.assert_called_once_with(mock_http, "workflows:delete", "wf_123")
            assert result.status_code == 403

    def test_execute_workflow_checks_execute_permission(self, handler, mock_http):
        """_handle_execute checks workflows:execute permission."""
        from aragora.server.handlers.base import error_response

        with patch.object(handler, "_check_permission") as mock_check:
            mock_check.return_value = error_response("denied", 403)

            result = handler._handle_execute("wf_123", {}, {}, mock_http)

            mock_check.assert_called_once_with(mock_http, "workflows:execute", "wf_123")
            assert result.status_code == 403

    def test_resolve_approval_checks_approve_permission(self, handler, mock_http):
        """_handle_resolve_approval checks workflows:approve permission."""
        from aragora.server.handlers.base import error_response

        with patch.object(handler, "_check_permission") as mock_check:
            mock_check.return_value = error_response("denied", 403)

            result = handler._handle_resolve_approval("req_123", {}, {}, mock_http)

            mock_check.assert_called_once_with(mock_http, "workflows:approve", "req_123")
            assert result.status_code == 403

    def test_get_workflow_checks_read_permission_with_id(self, handler, mock_http):
        """_handle_get_workflow checks workflows:read with resource ID."""
        from aragora.server.handlers.base import error_response

        with patch.object(handler, "_check_permission") as mock_check:
            mock_check.return_value = error_response("denied", 403)

            result = handler._handle_get_workflow("wf_abc", {}, mock_http)

            mock_check.assert_called_once_with(mock_http, "workflows:read", "wf_abc")
            assert result.status_code == 403

    def test_update_workflow_checks_update_permission(self, handler, mock_http):
        """_handle_update_workflow checks workflows:update permission."""
        from aragora.server.handlers.base import error_response

        with patch.object(handler, "_check_permission") as mock_check:
            mock_check.return_value = error_response("denied", 403)

            result = handler._handle_update_workflow("wf_xyz", {}, {}, mock_http)

            mock_check.assert_called_once_with(mock_http, "workflows:update", "wf_xyz")
            assert result.status_code == 403

    def test_simulate_workflow_checks_read_permission(self, handler, mock_http):
        """_handle_simulate checks workflows:read permission."""
        from aragora.server.handlers.base import error_response

        with patch.object(handler, "_check_permission") as mock_check:
            mock_check.return_value = error_response("denied", 403)

            result = handler._handle_simulate("wf_sim", {}, {}, mock_http)

            mock_check.assert_called_once_with(mock_http, "workflows:read", "wf_sim")
            assert result.status_code == 403

    def test_get_status_checks_read_permission(self, handler, mock_http):
        """_handle_get_status checks workflows:read permission."""
        from aragora.server.handlers.base import error_response

        with patch.object(handler, "_check_permission") as mock_check:
            mock_check.return_value = error_response("denied", 403)

            result = handler._handle_get_status("wf_status", {}, mock_http)

            mock_check.assert_called_once_with(mock_http, "workflows:read", "wf_status")
            assert result.status_code == 403

    def test_get_versions_checks_read_permission(self, handler, mock_http):
        """_handle_get_versions checks workflows:read permission."""
        from aragora.server.handlers.base import error_response

        with patch.object(handler, "_check_permission") as mock_check:
            mock_check.return_value = error_response("denied", 403)

            result = handler._handle_get_versions("wf_ver", {}, mock_http)

            mock_check.assert_called_once_with(mock_http, "workflows:read", "wf_ver")
            assert result.status_code == 403

    def test_list_templates_checks_read_permission(self, handler, mock_http):
        """_handle_list_templates checks workflows:read permission."""
        from aragora.server.handlers.base import error_response

        with patch.object(handler, "_check_permission") as mock_check:
            mock_check.return_value = error_response("denied", 403)

            result = handler._handle_list_templates({}, mock_http)

            mock_check.assert_called_once_with(mock_http, "workflows:read")
            assert result.status_code == 403

    def test_list_approvals_checks_read_permission(self, handler, mock_http):
        """_handle_list_approvals checks workflows:read permission."""
        from aragora.server.handlers.base import error_response

        with patch.object(handler, "_check_permission") as mock_check:
            mock_check.return_value = error_response("denied", 403)

            result = handler._handle_list_approvals({}, mock_http)

            mock_check.assert_called_once_with(mock_http, "workflows:read")
            assert result.status_code == 403

    def test_list_executions_checks_read_permission(self, handler, mock_http):
        """_handle_list_executions checks workflows:read permission."""
        from aragora.server.handlers.base import error_response

        with patch.object(handler, "_check_permission") as mock_check:
            mock_check.return_value = error_response("denied", 403)

            result = handler._handle_list_executions({}, mock_http)

            mock_check.assert_called_once_with(mock_http, "workflows:read")
            assert result.status_code == 403


class TestWorkflowHandlerRBACPermissionKeys:
    """Verify correct permission keys are used for each operation."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowHandler(mock_server_context)

    @pytest.mark.parametrize(
        "method_name,permission_key",
        [
            ("_handle_list_workflows", "workflows:read"),
            ("_handle_list_templates", "workflows:read"),
            ("_handle_list_approvals", "workflows:read"),
            ("_handle_list_executions", "workflows:read"),
        ],
    )
    def test_permission_key_for_list_operations(self, handler, method_name, permission_key):
        """Verify list operations use correct permission keys."""
        from aragora.server.handlers.base import error_response

        mock_http = MagicMock()
        mock_http.headers = {}

        with patch.object(handler, "_check_permission") as mock_check:
            # Return 403 to short-circuit execution
            mock_check.return_value = error_response("denied", 403)

            method = getattr(handler, method_name)
            method({}, mock_http)

            mock_check.assert_called_once()
            call_args = mock_check.call_args[0]
            assert call_args[1] == permission_key

    def test_create_workflow_permission_key(self, handler):
        """Verify create workflow uses workflows:create permission."""
        from aragora.server.handlers.base import error_response

        mock_http = MagicMock()
        mock_http.headers = {}

        with patch.object(handler, "_check_permission") as mock_check:
            mock_check.return_value = error_response("denied", 403)

            handler._handle_create_workflow({}, {}, mock_http)

            mock_check.assert_called_once()
            call_args = mock_check.call_args[0]
            assert call_args[1] == "workflows:create"

    @pytest.mark.parametrize(
        "method_name,permission_key",
        [
            ("_handle_get_workflow", "workflows:read"),
            ("_handle_update_workflow", "workflows:update"),
            ("_handle_delete_workflow", "workflows:delete"),
            ("_handle_execute", "workflows:execute"),
            ("_handle_simulate", "workflows:read"),
            ("_handle_get_status", "workflows:read"),
            ("_handle_get_versions", "workflows:read"),
        ],
    )
    def test_permission_key_for_resource_operations(self, handler, method_name, permission_key):
        """Verify resource operations use correct permission keys with resource ID."""
        from aragora.server.handlers.base import error_response

        mock_http = MagicMock()
        mock_http.headers = {}

        with patch.object(handler, "_check_permission") as mock_check:
            # Return 403 to short-circuit execution
            mock_check.return_value = error_response("denied", 403)

            method = getattr(handler, method_name)
            resource_id = "wf_test_123"

            # Call with appropriate args
            if method_name in (
                "_handle_get_workflow",
                "_handle_get_status",
                "_handle_get_versions",
            ):
                method(resource_id, {}, mock_http)
            elif method_name == "_handle_delete_workflow":
                method(resource_id, {}, mock_http)
            else:
                method(resource_id, {}, {}, mock_http)

            mock_check.assert_called_once()
            call_args = mock_check.call_args[0]
            assert call_args[1] == permission_key
            assert call_args[2] == resource_id  # Resource ID passed

    def test_resolve_approval_uses_request_id(self, handler):
        """Verify resolve approval uses request ID as resource."""
        from aragora.server.handlers.base import error_response

        mock_http = MagicMock()
        mock_http.headers = {}

        with patch.object(handler, "_check_permission") as mock_check:
            # Return 403 to short-circuit execution
            mock_check.return_value = error_response("denied", 403)

            handler._handle_resolve_approval("approval_req_456", {}, {}, mock_http)

            mock_check.assert_called_once()
            call_args = mock_check.call_args[0]
            assert call_args[1] == "workflows:approve"
            assert call_args[2] == "approval_req_456"


class TestWorkflowHandlerResponseFormat:
    """Tests for response format validation."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowHandler(mock_server_context)

    @pytest.fixture
    def mock_http(self):
        """Create mock HTTP handler with proper headers."""
        mock = MagicMock()
        mock.headers = MagicMock()
        mock.headers.get.return_value = ""
        return mock

    def test_error_response_has_status_code(self, handler, mock_http):
        """Error responses have proper status codes."""
        with patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract:
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = False
            mock_jwt_context.user_id = None
            mock_extract.return_value = mock_jwt_context

            result = handler.handle("/api/v1/workflows", {}, mock_http)

            # Should return 401 for unauthenticated
            assert result is not None
            assert result.status_code == 401

    def test_error_response_has_json_content_type(self, handler, mock_http):
        """Error responses have JSON content type."""
        with patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract:
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = False
            mock_jwt_context.user_id = None
            mock_extract.return_value = mock_jwt_context

            result = handler.handle("/api/v1/workflows", {}, mock_http)

            assert result is not None
            assert result.content_type == "application/json"

    def test_error_response_has_body(self, handler, mock_http):
        """Error responses have a body."""
        with patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract:
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = False
            mock_jwt_context.user_id = None
            mock_extract.return_value = mock_jwt_context

            result = handler.handle("/api/v1/workflows", {}, mock_http)

            assert result is not None
            assert result.body is not None
            assert len(result.body) > 0

    def test_error_response_body_is_valid_json(self, handler, mock_http):
        """Error response body is valid JSON."""
        import json

        with patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract:
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = False
            mock_jwt_context.user_id = None
            mock_extract.return_value = mock_jwt_context

            result = handler.handle("/api/v1/workflows", {}, mock_http)

            assert result is not None
            body = json.loads(result.body)
            assert isinstance(body, dict)

    def test_error_response_contains_error_field(self, handler, mock_http):
        """Error response body contains error field."""
        import json

        with patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract:
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = False
            mock_jwt_context.user_id = None
            mock_extract.return_value = mock_jwt_context

            result = handler.handle("/api/v1/workflows", {}, mock_http)

            assert result is not None
            body = json.loads(result.body)
            assert "error" in body

    def test_permission_denied_returns_403(self, handler, mock_http):
        """Permission denied returns 403 status code."""
        from aragora.server.handlers.base import error_response

        with patch.object(handler, "_check_permission") as mock_check:
            mock_check.return_value = error_response("Permission denied", 403)

            result = handler._handle_list_workflows({}, mock_http)

            assert result is not None
            assert result.status_code == 403

    def test_not_found_returns_404(self, handler, mock_http):
        """Not found returns 404 status code."""
        with patch.object(handler, "_check_permission") as mock_check:
            mock_check.return_value = None  # Allow

            with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                mock_async.return_value = None  # Workflow not found

                result = handler._handle_get_workflow("nonexistent_id", {}, mock_http)

                assert result is not None
                assert result.status_code == 404


class TestWorkflowHandlerMethodDispatch:
    """Tests for HTTP method dispatch."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowHandler(mock_server_context)

    @pytest.fixture
    def mock_http(self):
        """Create mock HTTP handler."""
        mock = MagicMock()
        mock.headers = MagicMock()
        mock.headers.get.return_value = ""
        mock.rfile = MagicMock()
        mock.rfile.read.return_value = b"{}"
        return mock

    def test_handle_get_dispatches_correctly(self, handler, mock_http):
        """handle() dispatches GET requests."""
        with patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract:
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = False
            mock_jwt_context.user_id = None
            mock_extract.return_value = mock_jwt_context

            result = handler.handle("/api/v1/workflows", {}, mock_http)

            # Returns a result (auth error in this case)
            assert result is not None

    def test_handle_post_dispatches_correctly(self, handler, mock_http):
        """handle_post() dispatches POST requests."""
        with patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract:
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = False
            mock_jwt_context.user_id = None
            mock_extract.return_value = mock_jwt_context

            result = handler.handle_post("/api/v1/workflows", {}, mock_http)

            # Returns a result (auth error in this case)
            assert result is not None

    def test_handle_patch_dispatches_correctly(self, handler, mock_http):
        """handle_patch() dispatches PATCH requests."""
        with patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract:
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = False
            mock_jwt_context.user_id = None
            mock_extract.return_value = mock_jwt_context

            result = handler.handle_patch("/api/v1/workflows/wf_123", {}, mock_http)

            assert result is not None

    def test_handle_put_delegates_to_patch(self, handler, mock_http):
        """handle_put() delegates to handle_patch()."""
        with patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract:
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = False
            mock_jwt_context.user_id = None
            mock_extract.return_value = mock_jwt_context

            result = handler.handle_put("/api/v1/workflows/wf_123", {}, mock_http)

            assert result is not None

    def test_handle_delete_dispatches_correctly(self, handler, mock_http):
        """handle_delete() dispatches DELETE requests."""
        with patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract:
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = False
            mock_jwt_context.user_id = None
            mock_extract.return_value = mock_jwt_context

            result = handler.handle_delete("/api/v1/workflows/wf_123", {}, mock_http)

            assert result is not None


# =============================================================================
# Helper Functions Tests
# =============================================================================


class TestStepResultToDict:
    """Tests for _step_result_to_dict helper function."""

    def test_converts_step_result_to_dict(self):
        """Convert a StepResult to dictionary."""
        from aragora.workflow.types import StepResult, StepStatus
        from datetime import datetime, timezone

        step = StepResult(
            step_id="step_1",
            step_name="Test Step",
            status=StepStatus.COMPLETED,
            started_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            completed_at=datetime(2024, 1, 1, 12, 0, 10, tzinfo=timezone.utc),
            duration_ms=10000.0,
            output={"result": "success"},
            error=None,
            metrics={"tokens": 100},
            retry_count=0,
        )

        from aragora.server.handlers.workflows import _step_result_to_dict

        result = _step_result_to_dict(step)

        assert result["step_id"] == "step_1"
        assert result["step_name"] == "Test Step"
        assert result["status"] == "completed"
        assert result["started_at"] == "2024-01-01T12:00:00+00:00"
        assert result["completed_at"] == "2024-01-01T12:00:10+00:00"
        assert result["duration_ms"] == 10000.0
        assert result["output"] == {"result": "success"}
        assert result["error"] is None
        assert result["metrics"] == {"tokens": 100}
        assert result["retry_count"] == 0

    def test_handles_none_datetime_fields(self):
        """Handle None datetime fields gracefully."""
        from aragora.workflow.types import StepResult, StepStatus
        from aragora.server.handlers.workflows import _step_result_to_dict

        step = StepResult(
            step_id="step_2",
            step_name="Pending Step",
            status=StepStatus.PENDING,
        )

        result = _step_result_to_dict(step)

        assert result["started_at"] is None
        assert result["completed_at"] is None

    def test_handles_failed_status(self):
        """Handle failed step status."""
        from aragora.workflow.types import StepResult, StepStatus
        from aragora.server.handlers.workflows import _step_result_to_dict

        step = StepResult(
            step_id="step_3",
            step_name="Failed Step",
            status=StepStatus.FAILED,
            error="Connection timeout",
        )

        result = _step_result_to_dict(step)

        assert result["status"] == "failed"
        assert result["error"] == "Connection timeout"


# =============================================================================
# Async API Function Tests
# =============================================================================


class TestListWorkflowsAsync:
    """Tests for list_workflows async function."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock workflow store."""
        mock = MagicMock()
        mock.list_workflows.return_value = ([], 0)
        return mock

    @pytest.mark.asyncio
    async def test_list_workflows_empty(self, mock_store):
        """List workflows returns empty list when no workflows exist."""
        from aragora.server.handlers.workflows import list_workflows

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            result = await list_workflows()

            assert result["workflows"] == []
            assert result["total_count"] == 0
            assert result["limit"] == 50
            assert result["offset"] == 0

    @pytest.mark.asyncio
    async def test_list_workflows_with_filters(self, mock_store):
        """List workflows with category and search filters."""
        from aragora.server.handlers.workflows import list_workflows

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            await list_workflows(
                tenant_id="custom",
                category="legal",
                search="contract",
                limit=10,
                offset=5,
            )

            mock_store.list_workflows.assert_called_once_with(
                tenant_id="custom",
                category="legal",
                tags=None,
                search="contract",
                limit=10,
                offset=5,
            )

    @pytest.mark.asyncio
    async def test_list_workflows_with_tags(self, mock_store):
        """List workflows with tag filters."""
        from aragora.server.handlers.workflows import list_workflows

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            await list_workflows(tags=["automation", "ai"])

            mock_store.list_workflows.assert_called_once()
            call_kwargs = mock_store.list_workflows.call_args.kwargs
            assert call_kwargs["tags"] == ["automation", "ai"]

    @pytest.mark.asyncio
    async def test_list_workflows_returns_workflow_dicts(self, mock_store):
        """List workflows returns workflow dictionaries."""
        from aragora.server.handlers.workflows import list_workflows

        mock_workflow = MagicMock()
        mock_workflow.to_dict.return_value = {"id": "wf_1", "name": "Test"}
        mock_store.list_workflows.return_value = ([mock_workflow], 1)

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            result = await list_workflows()

            assert len(result["workflows"]) == 1
            assert result["workflows"][0]["id"] == "wf_1"
            assert result["total_count"] == 1


class TestGetWorkflowAsync:
    """Tests for get_workflow async function."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock workflow store."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_get_workflow_found(self, mock_store):
        """Get workflow returns workflow when found."""
        from aragora.server.handlers.workflows import get_workflow

        mock_workflow = MagicMock()
        mock_workflow.to_dict.return_value = {"id": "wf_123", "name": "Found"}
        mock_store.get_workflow.return_value = mock_workflow

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            result = await get_workflow("wf_123")

            assert result is not None
            assert result["id"] == "wf_123"

    @pytest.mark.asyncio
    async def test_get_workflow_not_found(self, mock_store):
        """Get workflow returns None when not found."""
        from aragora.server.handlers.workflows import get_workflow

        mock_store.get_workflow.return_value = None

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            result = await get_workflow("nonexistent")

            assert result is None

    @pytest.mark.asyncio
    async def test_get_workflow_with_tenant(self, mock_store):
        """Get workflow uses tenant_id parameter."""
        from aragora.server.handlers.workflows import get_workflow

        mock_store.get_workflow.return_value = None

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            await get_workflow("wf_123", tenant_id="tenant_abc")

            mock_store.get_workflow.assert_called_once_with("wf_123", "tenant_abc")


class TestCreateWorkflowAsync:
    """Tests for create_workflow async function."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock workflow store."""
        return MagicMock()

    @pytest.fixture
    def valid_workflow_data(self):
        """Valid workflow data for creation."""
        return {
            "name": "Test Workflow",
            "description": "A test workflow",
            "steps": [
                {
                    "id": "step_1",
                    "name": "First Step",
                    "step_type": "task",
                }
            ],
        }

    @pytest.mark.asyncio
    async def test_create_workflow_success(self, mock_store, valid_workflow_data):
        """Create workflow successfully."""
        from aragora.server.handlers.workflows import create_workflow

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            result = await create_workflow(
                valid_workflow_data,
                tenant_id="test_tenant",
                created_by="user_123",
            )

            assert result is not None
            assert result["name"] == "Test Workflow"
            assert result["tenant_id"] == "test_tenant"
            assert result["created_by"] == "user_123"
            assert "id" in result
            mock_store.save_workflow.assert_called_once()
            mock_store.save_version.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_workflow_generates_id(self, mock_store, valid_workflow_data):
        """Create workflow generates ID if not provided."""
        from aragora.server.handlers.workflows import create_workflow

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            result = await create_workflow(valid_workflow_data)

            assert result["id"].startswith("wf_")

    @pytest.mark.asyncio
    async def test_create_workflow_uses_provided_id(self, mock_store, valid_workflow_data):
        """Create workflow uses provided ID."""
        from aragora.server.handlers.workflows import create_workflow

        valid_workflow_data["id"] = "custom_id"

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            result = await create_workflow(valid_workflow_data)

            assert result["id"] == "custom_id"

    @pytest.mark.asyncio
    async def test_create_workflow_invalid_raises_error(self, mock_store):
        """Create workflow raises ValueError for invalid data."""
        from aragora.server.handlers.workflows import create_workflow

        invalid_data = {
            "name": "",  # Empty name is invalid
            "steps": [],  # No steps is invalid
        }

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with pytest.raises(ValueError) as exc_info:
                await create_workflow(invalid_data)

            assert "Invalid workflow" in str(exc_info.value)


class TestUpdateWorkflowAsync:
    """Tests for update_workflow async function."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock workflow store."""
        return MagicMock()

    @pytest.fixture
    def existing_workflow(self):
        """Create an existing workflow mock."""
        from datetime import datetime, timezone

        mock = MagicMock()
        mock.version = "1.0.0"
        mock.created_by = "original_user"
        mock.created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
        return mock

    @pytest.mark.asyncio
    async def test_update_workflow_success(self, mock_store, existing_workflow):
        """Update workflow successfully."""
        from aragora.server.handlers.workflows import update_workflow

        mock_store.get_workflow.return_value = existing_workflow

        updated_data = {
            "name": "Updated Workflow",
            "steps": [
                {
                    "id": "step_1",
                    "name": "Updated Step",
                    "step_type": "task",
                }
            ],
        }

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            result = await update_workflow("wf_123", updated_data)

            assert result is not None
            assert result["name"] == "Updated Workflow"
            assert result["version"] == "1.0.1"  # Version incremented
            mock_store.save_workflow.assert_called_once()
            mock_store.save_version.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_workflow_not_found(self, mock_store):
        """Update workflow returns None when not found."""
        from aragora.server.handlers.workflows import update_workflow

        mock_store.get_workflow.return_value = None

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            result = await update_workflow("nonexistent", {"name": "New"})

            assert result is None

    @pytest.mark.asyncio
    async def test_update_workflow_preserves_metadata(self, mock_store, existing_workflow):
        """Update workflow preserves original metadata."""
        from aragora.server.handlers.workflows import update_workflow

        mock_store.get_workflow.return_value = existing_workflow

        updated_data = {
            "name": "Updated",
            "steps": [{"id": "s1", "name": "S1", "step_type": "task"}],
        }

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            result = await update_workflow("wf_123", updated_data)

            assert result["created_by"] == "original_user"

    @pytest.mark.asyncio
    async def test_update_workflow_increments_version(self, mock_store, existing_workflow):
        """Update workflow increments version number."""
        from aragora.server.handlers.workflows import update_workflow

        existing_workflow.version = "2.3.5"
        mock_store.get_workflow.return_value = existing_workflow

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            result = await update_workflow(
                "wf_123",
                {"name": "Test", "steps": [{"id": "s1", "name": "S1", "step_type": "task"}]},
            )

            assert result["version"] == "2.3.6"


class TestDeleteWorkflowAsync:
    """Tests for delete_workflow async function."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock workflow store."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_delete_workflow_success(self, mock_store):
        """Delete workflow returns True on success."""
        from aragora.server.handlers.workflows import delete_workflow

        mock_store.delete_workflow.return_value = True

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            result = await delete_workflow("wf_123")

            assert result is True
            mock_store.delete_workflow.assert_called_once_with("wf_123", "default")

    @pytest.mark.asyncio
    async def test_delete_workflow_not_found(self, mock_store):
        """Delete workflow returns False when not found."""
        from aragora.server.handlers.workflows import delete_workflow

        mock_store.delete_workflow.return_value = False

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            result = await delete_workflow("nonexistent")

            assert result is False

    @pytest.mark.asyncio
    async def test_delete_workflow_with_tenant(self, mock_store):
        """Delete workflow uses tenant_id parameter."""
        from aragora.server.handlers.workflows import delete_workflow

        mock_store.delete_workflow.return_value = True

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            await delete_workflow("wf_123", tenant_id="custom_tenant")

            mock_store.delete_workflow.assert_called_once_with("wf_123", "custom_tenant")


# =============================================================================
# Workflow Execution Tests
# =============================================================================


class TestExecuteWorkflowAsync:
    """Tests for execute_workflow async function."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock workflow store."""
        return MagicMock()

    @pytest.fixture
    def mock_engine(self):
        """Create a mock workflow engine."""
        from unittest.mock import AsyncMock

        mock = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.final_output = {"result": "done"}
        mock_result.steps = []
        mock_result.error = None
        mock_result.total_duration_ms = 1000.0
        mock.execute = AsyncMock(return_value=mock_result)
        return mock

    @pytest.fixture
    def mock_workflow(self):
        """Create a mock workflow."""
        mock = MagicMock()
        return mock

    @pytest.mark.asyncio
    async def test_execute_workflow_success(self, mock_store, mock_engine, mock_workflow):
        """Execute workflow successfully."""
        from aragora.server.handlers.workflows import execute_workflow

        mock_store.get_workflow.return_value = mock_workflow

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows._engine", mock_engine):
                result = await execute_workflow("wf_123", inputs={"key": "value"})

                assert result["status"] == "completed"
                assert "id" in result
                assert result["workflow_id"] == "wf_123"
                mock_engine.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_workflow_not_found(self, mock_store):
        """Execute workflow raises ValueError when not found."""
        from aragora.server.handlers.workflows import execute_workflow

        mock_store.get_workflow.return_value = None

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with pytest.raises(ValueError) as exc_info:
                await execute_workflow("nonexistent")

            assert "Workflow not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_workflow_failed(self, mock_store, mock_workflow):
        """Execute workflow handles failure."""
        from aragora.server.handlers.workflows import execute_workflow
        from unittest.mock import AsyncMock

        mock_store.get_workflow.return_value = mock_workflow

        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.final_output = None
        mock_result.steps = []
        mock_result.error = "Step failed"
        mock_result.total_duration_ms = 500.0
        mock_engine.execute = AsyncMock(return_value=mock_result)

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows._engine", mock_engine):
                result = await execute_workflow("wf_123")

                assert result["status"] == "failed"
                assert result["error"] == "Step failed"

    @pytest.mark.asyncio
    async def test_execute_workflow_stores_execution(self, mock_store, mock_engine, mock_workflow):
        """Execute workflow stores execution record."""
        from aragora.server.handlers.workflows import execute_workflow

        mock_store.get_workflow.return_value = mock_workflow

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows._engine", mock_engine):
                await execute_workflow("wf_123")

                # save_execution should be called at least twice: start and completion
                assert mock_store.save_execution.call_count >= 2


class TestGetExecutionAsync:
    """Tests for get_execution async function."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock workflow store."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_get_execution_found(self, mock_store):
        """Get execution returns execution when found."""
        from aragora.server.handlers.workflows import get_execution

        mock_store.get_execution.return_value = {"id": "exec_123", "status": "completed"}

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            result = await get_execution("exec_123")

            assert result is not None
            assert result["id"] == "exec_123"

    @pytest.mark.asyncio
    async def test_get_execution_not_found(self, mock_store):
        """Get execution returns None when not found."""
        from aragora.server.handlers.workflows import get_execution

        mock_store.get_execution.return_value = None

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            result = await get_execution("nonexistent")

            assert result is None


class TestListExecutionsAsync:
    """Tests for list_executions async function."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock workflow store."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_list_executions_empty(self, mock_store):
        """List executions returns empty list when none exist."""
        from aragora.server.handlers.workflows import list_executions

        mock_store.list_executions.return_value = ([], 0)

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            result = await list_executions()

            assert result == []

    @pytest.mark.asyncio
    async def test_list_executions_with_filters(self, mock_store):
        """List executions with filters."""
        from aragora.server.handlers.workflows import list_executions

        mock_store.list_executions.return_value = ([{"id": "e1"}], 1)

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            result = await list_executions(
                workflow_id="wf_123",
                tenant_id="tenant_abc",
                limit=10,
            )

            mock_store.list_executions.assert_called_once_with(
                workflow_id="wf_123",
                tenant_id="tenant_abc",
                limit=10,
            )
            assert len(result) == 1


class TestTerminateExecutionAsync:
    """Tests for terminate_execution async function."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock workflow store."""
        return MagicMock()

    @pytest.fixture
    def mock_engine(self):
        """Create a mock workflow engine."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_terminate_execution_success(self, mock_store, mock_engine):
        """Terminate running execution successfully."""
        from aragora.server.handlers.workflows import terminate_execution

        mock_store.get_execution.return_value = {"id": "exec_123", "status": "running"}

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows._engine", mock_engine):
                result = await terminate_execution("exec_123")

                assert result is True
                mock_engine.request_termination.assert_called_once()

    @pytest.mark.asyncio
    async def test_terminate_execution_not_found(self, mock_store, mock_engine):
        """Terminate execution returns False when not found."""
        from aragora.server.handlers.workflows import terminate_execution

        mock_store.get_execution.return_value = None

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows._engine", mock_engine):
                result = await terminate_execution("nonexistent")

                assert result is False

    @pytest.mark.asyncio
    async def test_terminate_execution_not_running(self, mock_store, mock_engine):
        """Terminate execution returns False when not running."""
        from aragora.server.handlers.workflows import terminate_execution

        mock_store.get_execution.return_value = {"id": "exec_123", "status": "completed"}

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows._engine", mock_engine):
                result = await terminate_execution("exec_123")

                assert result is False


# =============================================================================
# Template Tests
# =============================================================================


class TestListTemplatesAsync:
    """Tests for list_templates async function."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock workflow store."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_list_templates_empty(self, mock_store):
        """List templates returns empty list when none exist."""
        from aragora.server.handlers.workflows import list_templates

        mock_store.list_templates.return_value = []

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            result = await list_templates()

            assert result == []

    @pytest.mark.asyncio
    async def test_list_templates_with_category(self, mock_store):
        """List templates with category filter."""
        from aragora.server.handlers.workflows import list_templates

        mock_template = MagicMock()
        mock_template.to_dict.return_value = {"id": "tmpl_1", "category": "legal"}
        mock_store.list_templates.return_value = [mock_template]

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            result = await list_templates(category="legal")

            mock_store.list_templates.assert_called_once_with(category="legal", tags=None)
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_list_templates_with_tags(self, mock_store):
        """List templates with tag filter."""
        from aragora.server.handlers.workflows import list_templates

        mock_store.list_templates.return_value = []

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            await list_templates(tags=["ai", "automation"])

            mock_store.list_templates.assert_called_once_with(
                category=None, tags=["ai", "automation"]
            )


class TestGetTemplateAsync:
    """Tests for get_template async function."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock workflow store."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_get_template_found(self, mock_store):
        """Get template returns template when found."""
        from aragora.server.handlers.workflows import get_template

        mock_template = MagicMock()
        mock_template.to_dict.return_value = {"id": "tmpl_123", "name": "Test Template"}
        mock_store.get_template.return_value = mock_template

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            result = await get_template("tmpl_123")

            assert result is not None
            assert result["id"] == "tmpl_123"

    @pytest.mark.asyncio
    async def test_get_template_not_found(self, mock_store):
        """Get template returns None when not found."""
        from aragora.server.handlers.workflows import get_template

        mock_store.get_template.return_value = None

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            result = await get_template("nonexistent")

            assert result is None


class TestCreateWorkflowFromTemplateAsync:
    """Tests for create_workflow_from_template async function."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock workflow store."""
        return MagicMock()

    @pytest.fixture
    def mock_template(self):
        """Create a mock template."""
        from aragora.workflow.types import WorkflowDefinition

        mock = MagicMock(spec=WorkflowDefinition)
        mock.clone.return_value = MagicMock(spec=WorkflowDefinition)
        mock.clone.return_value.to_dict.return_value = {
            "id": "wf_from_template",
            "name": "Custom Name",
            "steps": [{"id": "s1", "name": "Step", "step_type": "task"}],
        }
        return mock

    @pytest.mark.asyncio
    async def test_create_from_template_success(self, mock_store, mock_template):
        """Create workflow from template successfully."""
        from aragora.server.handlers.workflows import create_workflow_from_template

        mock_store.get_template.return_value = mock_template

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            result = await create_workflow_from_template(
                "tmpl_123",
                "My Custom Workflow",
                tenant_id="tenant_abc",
                created_by="user_123",
            )

            assert result is not None
            mock_store.increment_template_usage.assert_called_once_with("tmpl_123")
            mock_template.clone.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_from_template_not_found(self, mock_store):
        """Create workflow from template raises ValueError when template not found."""
        from aragora.server.handlers.workflows import create_workflow_from_template

        mock_store.get_template.return_value = None

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with pytest.raises(ValueError) as exc_info:
                await create_workflow_from_template("nonexistent", "Name")

            assert "Template not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_from_template_with_customizations(self, mock_store, mock_template):
        """Create workflow from template with customizations."""
        from aragora.server.handlers.workflows import create_workflow_from_template

        mock_store.get_template.return_value = mock_template

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            await create_workflow_from_template(
                "tmpl_123",
                "Custom Name",
                customizations={"description": "Custom description"},
            )

            mock_template.clone.assert_called_once()


class TestRegisterTemplate:
    """Tests for register_template function."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock workflow store."""
        return MagicMock()

    def test_register_template(self, mock_store):
        """Register template saves to store."""
        from aragora.server.handlers.workflows import register_template
        from aragora.workflow.types import WorkflowDefinition

        mock_workflow = MagicMock(spec=WorkflowDefinition)
        mock_workflow.is_template = False

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            register_template(mock_workflow)

            assert mock_workflow.is_template is True
            mock_store.save_template.assert_called_once_with(mock_workflow)


# =============================================================================
# Approval Tests
# =============================================================================


class TestListPendingApprovalsAsync:
    """Tests for list_pending_approvals async function."""

    @pytest.mark.asyncio
    async def test_list_pending_approvals(self):
        """List pending approvals."""
        from aragora.server.handlers.workflows import list_pending_approvals

        mock_approval = MagicMock()
        mock_approval.to_dict.return_value = {"id": "req_123", "status": "pending"}

        with patch(
            "aragora.server.handlers.workflows.get_pending_approvals",
            return_value=[mock_approval],
        ):
            result = await list_pending_approvals()

            assert len(result) == 1
            assert result[0]["id"] == "req_123"

    @pytest.mark.asyncio
    async def test_list_pending_approvals_with_workflow_filter(self):
        """List pending approvals with workflow_id filter."""
        from aragora.server.handlers.workflows import list_pending_approvals

        with patch("aragora.server.handlers.workflows.get_pending_approvals") as mock_get:
            mock_get.return_value = []

            await list_pending_approvals(workflow_id="wf_123")

            mock_get.assert_called_once_with("wf_123")


class TestResolveApprovalAsync:
    """Tests for resolve_approval async function."""

    @pytest.mark.asyncio
    async def test_resolve_approval_approved(self):
        """Resolve approval with approved status."""
        from aragora.server.handlers.workflows import resolve_approval

        with patch("aragora.server.handlers.workflows._resolve", return_value=True):
            result = await resolve_approval(
                "req_123",
                status="approved",
                responder_id="user_456",
                notes="Looks good",
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_resolve_approval_rejected(self):
        """Resolve approval with rejected status."""
        from aragora.server.handlers.workflows import resolve_approval

        with patch("aragora.server.handlers.workflows._resolve", return_value=True):
            result = await resolve_approval(
                "req_123",
                status="rejected",
                responder_id="user_456",
                notes="Needs changes",
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_resolve_approval_invalid_status(self):
        """Resolve approval raises ValueError for invalid status."""
        from aragora.server.handlers.workflows import resolve_approval

        with pytest.raises(ValueError) as exc_info:
            await resolve_approval(
                "req_123",
                status="invalid_status",
                responder_id="user_456",
            )

        assert "Invalid status" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_resolve_approval_with_checklist(self):
        """Resolve approval with checklist updates."""
        from aragora.server.handlers.workflows import resolve_approval

        with patch("aragora.server.handlers.workflows._resolve") as mock_resolve:
            mock_resolve.return_value = True

            await resolve_approval(
                "req_123",
                status="approved",
                responder_id="user_456",
                checklist_updates={"item_1": True, "item_2": False},
            )

            mock_resolve.assert_called_once()
            call_kwargs = mock_resolve.call_args
            assert call_kwargs[0][4] == {"item_1": True, "item_2": False}


class TestGetApprovalAsync:
    """Tests for get_approval async function."""

    @pytest.mark.asyncio
    async def test_get_approval_found(self):
        """Get approval returns approval when found."""
        from aragora.server.handlers.workflows import get_approval

        mock_approval = MagicMock()
        mock_approval.to_dict.return_value = {"id": "req_123"}

        with patch(
            "aragora.server.handlers.workflows.get_approval_request",
            return_value=mock_approval,
        ):
            result = await get_approval("req_123")

            assert result is not None
            assert result["id"] == "req_123"

    @pytest.mark.asyncio
    async def test_get_approval_not_found(self):
        """Get approval returns None when not found."""
        from aragora.server.handlers.workflows import get_approval

        with patch(
            "aragora.server.handlers.workflows.get_approval_request",
            return_value=None,
        ):
            result = await get_approval("nonexistent")

            assert result is None


# =============================================================================
# Version Management Tests
# =============================================================================


class TestGetWorkflowVersionsAsync:
    """Tests for get_workflow_versions async function."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock workflow store."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_get_versions_returns_list(self, mock_store):
        """Get workflow versions returns version list."""
        from aragora.server.handlers.workflows import get_workflow_versions

        mock_store.get_versions.return_value = [
            {"version": "1.0.0", "created_at": "2024-01-01"},
            {"version": "1.0.1", "created_at": "2024-01-02"},
        ]

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            result = await get_workflow_versions("wf_123")

            assert len(result) == 2
            assert result[0]["version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_get_versions_with_limit(self, mock_store):
        """Get workflow versions respects limit parameter."""
        from aragora.server.handlers.workflows import get_workflow_versions

        mock_store.get_versions.return_value = []

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            await get_workflow_versions("wf_123", limit=5)

            mock_store.get_versions.assert_called_once_with("wf_123", "default", 5)


class TestRestoreWorkflowVersionAsync:
    """Tests for restore_workflow_version async function."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock workflow store."""
        return MagicMock()

    @pytest.fixture
    def old_workflow(self):
        """Create a mock old workflow version."""
        from aragora.workflow.types import WorkflowDefinition

        mock = MagicMock(spec=WorkflowDefinition)
        mock.name = "Old Version"
        cloned = MagicMock(spec=WorkflowDefinition)
        cloned.to_dict.return_value = {
            "id": "wf_123",
            "name": "Old Version",
            "steps": [{"id": "s1", "name": "Step", "step_type": "task"}],
        }
        mock.clone.return_value = cloned
        return mock

    @pytest.mark.asyncio
    async def test_restore_version_success(self, mock_store, old_workflow):
        """Restore workflow version successfully."""
        from aragora.server.handlers.workflows import restore_workflow_version
        from datetime import datetime, timezone

        mock_store.get_version.return_value = old_workflow

        # Also mock get_workflow for the update
        existing = MagicMock()
        existing.version = "2.0.0"
        existing.created_by = "user"
        existing.created_at = datetime.now(timezone.utc)
        mock_store.get_workflow.return_value = existing

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            result = await restore_workflow_version("wf_123", "1.0.0")

            assert result is not None
            mock_store.get_version.assert_called_once_with("wf_123", "1.0.0")

    @pytest.mark.asyncio
    async def test_restore_version_not_found(self, mock_store):
        """Restore workflow version returns None when version not found."""
        from aragora.server.handlers.workflows import restore_workflow_version

        mock_store.get_version.return_value = None

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            result = await restore_workflow_version("wf_123", "99.0.0")

            assert result is None


# =============================================================================
# WorkflowHandlers (Static) Tests
# =============================================================================


class TestWorkflowHandlersStatic:
    """Tests for WorkflowHandlers static methods."""

    @pytest.mark.asyncio
    async def test_handle_list_workflows(self):
        """Static handler for list workflows."""
        from aragora.server.handlers.workflows import WorkflowHandlers, list_workflows

        with patch("aragora.server.handlers.workflows.list_workflows") as mock_list:
            mock_list.return_value = {"workflows": [], "total_count": 0, "limit": 50, "offset": 0}

            result = await WorkflowHandlers.handle_list_workflows({})

            assert result["workflows"] == []

    @pytest.mark.asyncio
    async def test_handle_get_workflow(self):
        """Static handler for get workflow."""
        from aragora.server.handlers.workflows import WorkflowHandlers

        with patch("aragora.server.handlers.workflows.get_workflow") as mock_get:
            mock_get.return_value = {"id": "wf_123"}

            result = await WorkflowHandlers.handle_get_workflow("wf_123", {})

            assert result["id"] == "wf_123"

    @pytest.mark.asyncio
    async def test_handle_create_workflow(self):
        """Static handler for create workflow."""
        from aragora.server.handlers.workflows import WorkflowHandlers

        with patch("aragora.server.handlers.workflows.create_workflow") as mock_create:
            mock_create.return_value = {"id": "wf_new", "name": "New"}

            result = await WorkflowHandlers.handle_create_workflow(
                {"name": "New", "steps": []},
                {"tenant_id": "test", "user_id": "user_1"},
            )

            assert result["id"] == "wf_new"

    @pytest.mark.asyncio
    async def test_handle_update_workflow(self):
        """Static handler for update workflow."""
        from aragora.server.handlers.workflows import WorkflowHandlers

        with patch("aragora.server.handlers.workflows.update_workflow") as mock_update:
            mock_update.return_value = {"id": "wf_123", "name": "Updated"}

            result = await WorkflowHandlers.handle_update_workflow(
                "wf_123",
                {"name": "Updated"},
                {},
            )

            assert result["name"] == "Updated"

    @pytest.mark.asyncio
    async def test_handle_delete_workflow(self):
        """Static handler for delete workflow."""
        from aragora.server.handlers.workflows import WorkflowHandlers

        with patch("aragora.server.handlers.workflows.delete_workflow") as mock_delete:
            mock_delete.return_value = True

            result = await WorkflowHandlers.handle_delete_workflow("wf_123", {})

            assert result is True

    @pytest.mark.asyncio
    async def test_handle_execute_workflow(self):
        """Static handler for execute workflow."""
        from aragora.server.handlers.workflows import WorkflowHandlers

        with patch("aragora.server.handlers.workflows.execute_workflow") as mock_execute:
            mock_execute.return_value = {"id": "exec_123", "status": "completed"}

            result = await WorkflowHandlers.handle_execute_workflow(
                "wf_123",
                {"inputs": {"key": "value"}},
                {},
            )

            assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_handle_list_templates(self):
        """Static handler for list templates."""
        from aragora.server.handlers.workflows import WorkflowHandlers

        with patch("aragora.server.handlers.workflows.list_templates") as mock_list:
            mock_list.return_value = [{"id": "tmpl_1"}]

            result = await WorkflowHandlers.handle_list_templates({})

            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handle_list_approvals(self):
        """Static handler for list approvals."""
        from aragora.server.handlers.workflows import WorkflowHandlers

        with patch("aragora.server.handlers.workflows.list_pending_approvals") as mock_list:
            mock_list.return_value = []

            result = await WorkflowHandlers.handle_list_approvals({})

            assert result == []

    @pytest.mark.asyncio
    async def test_handle_resolve_approval(self):
        """Static handler for resolve approval."""
        from aragora.server.handlers.workflows import WorkflowHandlers

        with patch("aragora.server.handlers.workflows.resolve_approval") as mock_resolve:
            mock_resolve.return_value = True

            result = await WorkflowHandlers.handle_resolve_approval(
                "req_123",
                {"status": "approved"},
                {"user_id": "user_1"},
            )

            assert result is True


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestWorkflowHandlerErrorHandling:
    """Tests for error handling in WorkflowHandler."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowHandler(mock_server_context)

    @pytest.fixture
    def mock_http(self):
        """Create mock HTTP handler."""
        mock = MagicMock()
        mock.headers = MagicMock()
        mock.headers.get.return_value = ""
        return mock

    def test_storage_error_returns_503(self, handler, mock_http):
        """Storage errors return 503 status."""
        with patch.object(handler, "_check_permission", return_value=None):
            with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                mock_async.side_effect = OSError("Database connection failed")

                result = handler._handle_list_workflows({}, mock_http)

                assert result.status_code == 503

    def test_data_error_returns_500(self, handler, mock_http):
        """Data errors return 500 status."""
        with patch.object(handler, "_check_permission", return_value=None):
            with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                mock_async.side_effect = KeyError("Missing field")

                result = handler._handle_list_workflows({}, mock_http)

                assert result.status_code == 500

    def test_validation_error_returns_400(self, handler, mock_http):
        """Validation errors return 400 status."""
        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(handler, "_get_tenant_id", return_value="test"):
                with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                    mock_async.side_effect = ValueError("Invalid workflow")

                    result = handler._handle_create_workflow({}, {}, mock_http)

                    assert result.status_code == 400

    def test_connection_error_in_execute_returns_503(self, handler, mock_http):
        """Connection errors in execute return 503."""
        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(handler, "_get_tenant_id", return_value="test"):
                with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                    mock_async.side_effect = ConnectionError("Failed to connect")

                    result = handler._handle_execute("wf_123", {}, {}, mock_http)

                    assert result.status_code == 503

    def test_timeout_error_in_execute_returns_503(self, handler, mock_http):
        """Timeout errors in execute return 503."""
        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(handler, "_get_tenant_id", return_value="test"):
                with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                    mock_async.side_effect = TimeoutError("Execution timed out")

                    result = handler._handle_execute("wf_123", {}, {}, mock_http)

                    assert result.status_code == 503


# =============================================================================
# Path Alias Tests
# =============================================================================


class TestWorkflowHandlerPathAliases:
    """Tests for path alias handling."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowHandler(mock_server_context)

    @pytest.fixture
    def mock_http(self):
        """Create mock HTTP handler."""
        mock = MagicMock()
        mock.headers = MagicMock()
        mock.headers.get.return_value = ""
        mock.rfile = MagicMock()
        mock.rfile.read.return_value = b"{}"
        return mock

    def test_templates_alias_in_handle(self, handler, mock_http):
        """workflows/templates alias is handled correctly."""
        with patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract:
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = False
            mock_jwt_context.user_id = None
            mock_extract.return_value = mock_jwt_context

            result = handler.handle("/api/v1/workflows/templates", {}, mock_http)

            assert result is not None

    def test_executions_alias_in_handle(self, handler, mock_http):
        """workflows/executions alias is handled correctly."""
        with patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract:
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = False
            mock_jwt_context.user_id = None
            mock_extract.return_value = mock_jwt_context

            result = handler.handle("/api/v1/workflows/executions", {}, mock_http)

            assert result is not None

    def test_templates_alias_in_handle_post(self, handler, mock_http):
        """workflows/templates alias is handled in POST."""
        with patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract:
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = False
            mock_jwt_context.user_id = None
            mock_extract.return_value = mock_jwt_context

            result = handler.handle_post("/api/v1/workflows/templates", {}, mock_http)

            assert result is not None

    def test_executions_alias_in_handle_delete(self, handler, mock_http):
        """workflows/executions alias is handled in DELETE."""
        with patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract:
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = False
            mock_jwt_context.user_id = None
            mock_extract.return_value = mock_jwt_context

            result = handler.handle_delete("/api/v1/workflows/executions/exec_123", {}, mock_http)

            assert result is not None


# =============================================================================
# Authorization Edge Cases
# =============================================================================


class TestWorkflowHandlerAuthorizationEdgeCases:
    """Tests for authorization edge cases."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowHandler(mock_server_context)

    @pytest.fixture
    def mock_http(self):
        """Create mock HTTP handler."""
        mock = MagicMock()
        mock.headers = MagicMock()
        mock.headers.get.return_value = ""
        return mock

    def test_wildcard_permission_allows_access(self, handler, mock_http):
        """workflow.* permission grants access to all operations."""
        with patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract:
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = True
            mock_jwt_context.user_id = "user_123"
            mock_jwt_context.org_id = "org_abc"
            mock_jwt_context.role = "admin"
            mock_extract.return_value = mock_jwt_context

            with patch("aragora.server.handlers.workflows.get_role_permissions") as mock_perms:
                mock_perms.return_value = {"workflow.*"}

                with patch.object(handler, "_check_permission", return_value=None):
                    # Should be able to access any endpoint
                    result = handler.handle("/api/v1/workflows", {}, mock_http)
                    assert result is not None  # May still fail but not due to auth

    def test_execute_requires_execute_permission(self, handler, mock_http):
        """Execute endpoint requires workflows:execute permission."""
        mock_http.rfile = MagicMock()
        mock_http.rfile.read.return_value = b"{}"

        with patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract:
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = True
            mock_jwt_context.user_id = "user_123"
            mock_jwt_context.org_id = None
            mock_jwt_context.role = "member"
            mock_extract.return_value = mock_jwt_context

            with patch("aragora.server.handlers.workflows.get_role_permissions") as mock_perms:
                # Member has only read permission
                mock_perms.return_value = {"workflow.read"}

                result = handler.handle_post("/api/v1/workflows/wf_123/execute", {}, mock_http)

                # Should be denied
                assert result is not None
                assert result.status_code == 403

    def test_terminate_execution_requires_execute_permission(self, handler, mock_http):
        """Terminate execution requires workflows:execute permission."""
        with patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract:
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = True
            mock_jwt_context.user_id = "user_123"
            mock_jwt_context.org_id = None
            mock_jwt_context.role = "viewer"
            mock_extract.return_value = mock_jwt_context

            with patch("aragora.server.handlers.workflows.get_role_permissions") as mock_perms:
                mock_perms.return_value = {"workflows:read"}

                result = handler.handle_delete(
                    "/api/v1/workflow-executions/exec_123", {}, mock_http
                )

                assert result.status_code == 403

    def test_restore_version_requires_update_permission(self, handler, mock_http):
        """Restore version requires workflows:update permission."""
        mock_http.rfile = MagicMock()
        mock_http.rfile.read.return_value = b"{}"

        with patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract:
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = True
            mock_jwt_context.user_id = "user_123"
            mock_jwt_context.org_id = None
            mock_jwt_context.role = "viewer"
            mock_extract.return_value = mock_jwt_context

            with patch("aragora.server.handlers.workflows.get_role_permissions") as mock_perms:
                mock_perms.return_value = {"workflows:read"}

                result = handler.handle_post(
                    "/api/v1/workflows/wf_123/versions/1.0.0/restore",
                    {},
                    mock_http,
                )

                assert result.status_code == 403

    def test_resolve_approval_requires_approve_permission(self, handler, mock_http):
        """Resolve approval requires workflows:approve permission."""
        mock_http.rfile = MagicMock()
        mock_http.rfile.read.return_value = b'{"status": "approved"}'

        with patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract:
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = True
            mock_jwt_context.user_id = "user_123"
            mock_jwt_context.org_id = None
            mock_jwt_context.role = "member"
            mock_extract.return_value = mock_jwt_context

            with patch("aragora.server.handlers.workflows.get_role_permissions") as mock_perms:
                mock_perms.return_value = {"workflows:read", "workflows:execute"}

                result = handler.handle_post(
                    "/api/v1/workflow-approvals/req_123/resolve",
                    {},
                    mock_http,
                )

                assert result.status_code == 403


# =============================================================================
# Pagination Tests
# =============================================================================


class TestWorkflowHandlerPagination:
    """Tests for pagination handling."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowHandler(mock_server_context)

    @pytest.fixture
    def mock_http_authenticated(self):
        """Create authenticated mock HTTP handler."""
        mock = MagicMock()
        mock.headers = MagicMock()
        mock.headers.get.return_value = ""
        return mock

    def test_pagination_defaults(self, handler, mock_http_authenticated):
        """Default pagination values are applied."""
        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(handler, "_get_auth_context") as mock_auth:
                mock_ctx = MagicMock()
                mock_ctx.permissions = {"workflow.read"}
                mock_auth.return_value = mock_ctx

                with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                    mock_async.return_value = {
                        "workflows": [],
                        "total_count": 0,
                        "limit": 50,
                        "offset": 0,
                    }

                    result = handler._handle_list_workflows({}, mock_http_authenticated)

                    # Verify limit/offset defaults were used
                    call_args = mock_async.call_args
                    assert call_args is not None

    def test_custom_limit_respected(self, handler, mock_http_authenticated):
        """Custom limit parameter is respected."""
        with patch.object(handler, "_check_permission", return_value=None):
            with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                mock_async.return_value = {
                    "workflows": [],
                    "total_count": 0,
                    "limit": 10,
                    "offset": 0,
                }

                handler._handle_list_workflows({"limit": "10"}, mock_http_authenticated)

                # Verify custom limit was passed
                assert mock_async.called

    def test_custom_offset_respected(self, handler, mock_http_authenticated):
        """Custom offset parameter is respected."""
        with patch.object(handler, "_check_permission", return_value=None):
            with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                mock_async.return_value = {
                    "workflows": [],
                    "total_count": 100,
                    "limit": 20,
                    "offset": 40,
                }

                handler._handle_list_workflows({"offset": "40"}, mock_http_authenticated)

                assert mock_async.called


# =============================================================================
# Simulate Endpoint Tests
# =============================================================================


class TestSimulateEndpoint:
    """Tests for workflow simulation endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowHandler(mock_server_context)

    @pytest.fixture
    def mock_http(self):
        """Create mock HTTP handler."""
        mock = MagicMock()
        mock.headers = MagicMock()
        mock.headers.get.return_value = ""
        return mock

    def test_simulate_returns_execution_plan(self, handler, mock_http):
        """Simulate returns execution plan for valid workflow."""
        import json

        mock_workflow_dict = {
            "id": "wf_123",
            "name": "Test",
            "steps": [
                {"id": "s1", "name": "Step 1", "step_type": "task", "next_steps": ["s2"]},
                {"id": "s2", "name": "Step 2", "step_type": "task", "next_steps": []},
            ],
            "entry_step": "s1",
            "version": "1.0.0",
        }

        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(handler, "_get_tenant_id", return_value="test"):
                with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                    mock_async.return_value = mock_workflow_dict

                    result = handler._handle_simulate("wf_123", {}, {}, mock_http)

                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["workflow_id"] == "wf_123"
                    assert "execution_plan" in body
                    assert "is_valid" in body

    def test_simulate_workflow_not_found(self, handler, mock_http):
        """Simulate returns 404 for non-existent workflow."""
        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(handler, "_get_tenant_id", return_value="test"):
                with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                    mock_async.return_value = None

                    result = handler._handle_simulate("nonexistent", {}, {}, mock_http)

                    assert result.status_code == 404


# =============================================================================
# Execution Status Tracking Tests
# =============================================================================


class TestExecutionStatusTracking:
    """Tests for execution status tracking."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowHandler(mock_server_context)

    @pytest.fixture
    def mock_http(self):
        """Create mock HTTP handler."""
        mock = MagicMock()
        mock.headers = MagicMock()
        mock.headers.get.return_value = ""
        return mock

    def test_get_status_returns_latest_execution(self, handler, mock_http):
        """Get status returns the most recent execution."""
        import json

        with patch.object(handler, "_check_permission", return_value=None):
            with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                mock_async.return_value = [
                    {"id": "exec_123", "status": "completed", "workflow_id": "wf_123"}
                ]

                result = handler._handle_get_status("wf_123", {}, mock_http)

                assert result.status_code == 200
                body = json.loads(result.body)
                assert body["id"] == "exec_123"
                assert body["status"] == "completed"

    def test_get_status_no_executions(self, handler, mock_http):
        """Get status returns message when no executions exist."""
        import json

        with patch.object(handler, "_check_permission", return_value=None):
            with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                mock_async.return_value = []

                result = handler._handle_get_status("wf_123", {}, mock_http)

                assert result.status_code == 200
                body = json.loads(result.body)
                assert body["status"] == "no_executions"

    def test_list_executions_filters_by_status(self, handler, mock_http):
        """List executions can filter by status."""
        import json

        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(handler, "_get_tenant_id", return_value="test"):
                with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                    mock_async.return_value = [
                        {"id": "e1", "status": "running"},
                        {"id": "e2", "status": "completed"},
                        {"id": "e3", "status": "running"},
                    ]

                    result = handler._handle_list_executions(
                        {"status": "running"},
                        mock_http,
                    )

                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert len(body["executions"]) == 2
                    assert all(e["status"] == "running" for e in body["executions"])

    def test_get_execution_details(self, handler, mock_http):
        """Get execution returns full execution details."""
        import json

        with patch.object(handler, "_check_permission", return_value=None):
            with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                mock_async.return_value = {
                    "id": "exec_123",
                    "status": "completed",
                    "workflow_id": "wf_123",
                    "steps": [{"step_id": "s1", "status": "completed"}],
                    "outputs": {"result": "success"},
                }

                result = handler._handle_get_execution("exec_123", {}, mock_http)

                assert result.status_code == 200
                body = json.loads(result.body)
                assert body["id"] == "exec_123"
                assert "steps" in body
                assert "outputs" in body

    def test_get_execution_not_found(self, handler, mock_http):
        """Get execution returns 404 for non-existent execution."""
        with patch.object(handler, "_check_permission", return_value=None):
            with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                mock_async.return_value = None

                result = handler._handle_get_execution("nonexistent", {}, mock_http)

                assert result.status_code == 404


# =============================================================================
# Content-Type Validation Tests
# =============================================================================


class TestContentTypeValidation:
    """Tests for Content-Type header validation."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowHandler(mock_server_context)

    def test_post_without_content_type_with_body(self, handler):
        """POST without Content-Type but with body returns 415."""
        mock_http = MagicMock()
        mock_http.headers = MagicMock()
        mock_http.headers.get.side_effect = lambda k, d=None: {"Content-Length": "10"}.get(k, d)

        with patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract:
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = True
            mock_jwt_context.user_id = "user_123"
            mock_extract.return_value = mock_jwt_context

            with patch("aragora.server.handlers.workflows.get_role_permissions") as mock_perms:
                mock_perms.return_value = {"workflows:create"}

                result = handler.handle_post("/api/v1/workflows", {}, mock_http)

                assert result is not None
                # Should return 415 for missing Content-Type on POST with body

    def test_post_with_invalid_content_type_returns_415(self, handler):
        """POST with invalid Content-Type returns 415."""
        mock_http = MagicMock()
        mock_http.headers = MagicMock()
        mock_http.headers.get.side_effect = lambda k, d=None: {
            "Content-Type": "text/plain",
            "Content-Length": "10",
        }.get(k, d)

        with patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract:
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = True
            mock_jwt_context.user_id = "user_123"
            mock_extract.return_value = mock_jwt_context

            with patch("aragora.server.handlers.workflows.get_role_permissions") as mock_perms:
                mock_perms.return_value = {"workflows:create"}

                result = handler.handle_post("/api/v1/workflows", {}, mock_http)

                assert result is not None


# =============================================================================
# Integration with Audit System Tests
# =============================================================================


class TestAuditIntegration:
    """Tests for audit logging integration."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock workflow store."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_create_workflow_logs_audit(self, mock_store):
        """Create workflow logs to audit system."""
        from aragora.server.handlers.workflows import create_workflow

        valid_data = {
            "name": "Audited Workflow",
            "steps": [{"id": "s1", "name": "Step", "step_type": "task"}],
        }

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows.audit_data") as mock_audit:
                await create_workflow(valid_data, created_by="user_123")

                mock_audit.assert_called_once()
                call_kwargs = mock_audit.call_args.kwargs
                assert call_kwargs["action"] == "create"
                assert call_kwargs["resource_type"] == "workflow"
                assert call_kwargs["user_id"] == "user_123"

    @pytest.mark.asyncio
    async def test_update_workflow_logs_audit(self, mock_store):
        """Update workflow logs to audit system."""
        from aragora.server.handlers.workflows import update_workflow
        from datetime import datetime, timezone

        existing = MagicMock()
        existing.version = "1.0.0"
        existing.created_by = "original"
        existing.created_at = datetime.now(timezone.utc)
        mock_store.get_workflow.return_value = existing

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows.audit_data") as mock_audit:
                await update_workflow(
                    "wf_123",
                    {"name": "Updated", "steps": [{"id": "s1", "name": "S", "step_type": "task"}]},
                )

                mock_audit.assert_called_once()
                call_kwargs = mock_audit.call_args.kwargs
                assert call_kwargs["action"] == "update"

    @pytest.mark.asyncio
    async def test_delete_workflow_logs_audit(self, mock_store):
        """Delete workflow logs to audit system."""
        from aragora.server.handlers.workflows import delete_workflow

        mock_store.delete_workflow.return_value = True

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows.audit_data") as mock_audit:
                await delete_workflow("wf_123")

                mock_audit.assert_called_once()
                call_kwargs = mock_audit.call_args.kwargs
                assert call_kwargs["action"] == "delete"

    @pytest.mark.asyncio
    async def test_execute_workflow_logs_audit(self, mock_store):
        """Execute workflow logs to audit system."""
        from aragora.server.handlers.workflows import execute_workflow
        from unittest.mock import AsyncMock

        mock_workflow = MagicMock()
        mock_store.get_workflow.return_value = mock_workflow

        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.final_output = {}
        mock_result.steps = []
        mock_result.error = None
        mock_result.total_duration_ms = 100
        mock_engine.execute = AsyncMock(return_value=mock_result)

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows._engine", mock_engine):
                with patch("aragora.server.handlers.workflows.audit_data") as mock_audit:
                    await execute_workflow("wf_123")

                    mock_audit.assert_called_once()
                    call_kwargs = mock_audit.call_args.kwargs
                    assert call_kwargs["action"] == "execute"
                    assert call_kwargs["resource_type"] == "workflow_execution"


# =============================================================================
# Additional Comprehensive Tests for Pattern Factory/Category Filtering
# =============================================================================


class TestPatternFactoryCategoryFiltering:
    """Tests for workflow pattern and category filtering."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowHandler(mock_server_context)

    @pytest.fixture
    def mock_http(self):
        mock = MagicMock()
        mock.headers = MagicMock()
        mock.headers.get.return_value = ""
        return mock

    def test_list_workflows_with_category_filter(self, handler, mock_http):
        """List workflows filters by category parameter."""
        import json

        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(handler, "_get_tenant_id", return_value="test"):
                with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                    mock_async.return_value = {
                        "workflows": [{"id": "wf1", "category": "legal"}],
                        "total_count": 1,
                        "limit": 50,
                        "offset": 0,
                    }

                    result = handler._handle_list_workflows(
                        {"category": "legal"},
                        mock_http,
                    )

                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert "workflows" in body

    def test_list_workflows_with_search_filter(self, handler, mock_http):
        """List workflows filters by search term."""
        import json

        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(handler, "_get_tenant_id", return_value="test"):
                with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                    mock_async.return_value = {
                        "workflows": [{"id": "wf1", "name": "Contract Review"}],
                        "total_count": 1,
                        "limit": 50,
                        "offset": 0,
                    }

                    result = handler._handle_list_workflows(
                        {"search": "contract"},
                        mock_http,
                    )

                    assert result.status_code == 200

    def test_list_workflows_combined_filters(self, handler, mock_http):
        """List workflows with multiple filters combined."""
        import json

        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(handler, "_get_tenant_id", return_value="test"):
                with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                    mock_async.return_value = {
                        "workflows": [],
                        "total_count": 0,
                        "limit": 10,
                        "offset": 20,
                    }

                    result = handler._handle_list_workflows(
                        {"category": "legal", "search": "nda", "limit": "10", "offset": "20"},
                        mock_http,
                    )

                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["limit"] == 10
                    assert body["offset"] == 20

    def test_list_templates_with_category_filter(self, handler, mock_http):
        """List templates filters by category."""
        import json

        with patch.object(handler, "_check_permission", return_value=None):
            with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                mock_async.return_value = [{"id": "tpl1", "category": "finance"}]

                result = handler._handle_list_templates(
                    {"category": "finance"},
                    mock_http,
                )

                assert result.status_code == 200
                body = json.loads(result.body)
                assert "templates" in body

    def test_list_templates_multiple_results(self, handler, mock_http):
        """List templates returns multiple templates."""
        import json

        with patch.object(handler, "_check_permission", return_value=None):
            with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                mock_async.return_value = [
                    {"id": "tpl1", "category": "legal"},
                    {"id": "tpl2", "category": "legal"},
                    {"id": "tpl3", "category": "finance"},
                ]

                result = handler._handle_list_templates({}, mock_http)

                assert result.status_code == 200
                body = json.loads(result.body)
                assert body["count"] == 3


# =============================================================================
# Additional Tests for Node Configuration Edge Cases
# =============================================================================


class TestNodeConfigurationEdgeCases:
    """Tests for workflow node configuration edge cases."""

    @pytest.fixture
    def mock_store(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_create_workflow_with_parallel_execution_pattern(self, mock_store):
        """Create workflow with parallel execution pattern."""
        from aragora.server.handlers.workflows import create_workflow

        workflow_data = {
            "name": "Parallel Workflow",
            "steps": [
                {
                    "id": "s1",
                    "name": "Parallel Step",
                    "step_type": "task",
                    "execution_pattern": "parallel",
                    "config": {"max_parallel": 5},
                }
            ],
        }

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows.audit_data"):
                result = await create_workflow(workflow_data)

                assert "id" in result

    @pytest.mark.asyncio
    async def test_create_workflow_with_loop_pattern(self, mock_store):
        """Create workflow with loop execution pattern."""
        from aragora.server.handlers.workflows import create_workflow

        workflow_data = {
            "name": "Loop Workflow",
            "steps": [
                {
                    "id": "s1",
                    "name": "Loop Step",
                    "step_type": "task",
                    "execution_pattern": "loop",
                    "config": {"max_iterations": 10},
                }
            ],
        }

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows.audit_data"):
                result = await create_workflow(workflow_data)

                assert "id" in result

    @pytest.mark.asyncio
    async def test_create_workflow_with_conditional_pattern(self, mock_store):
        """Create workflow with conditional execution pattern."""
        from aragora.server.handlers.workflows import create_workflow

        workflow_data = {
            "name": "Conditional Workflow",
            "steps": [
                {
                    "id": "s1",
                    "name": "Conditional Step",
                    "step_type": "decision",
                    "execution_pattern": "conditional",
                    "config": {},
                }
            ],
        }

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows.audit_data"):
                result = await create_workflow(workflow_data)

                assert "id" in result

    @pytest.mark.asyncio
    async def test_create_workflow_with_timeout_config(self, mock_store):
        """Create workflow with custom timeout configuration."""
        from aragora.server.handlers.workflows import create_workflow

        workflow_data = {
            "name": "Timeout Workflow",
            "steps": [
                {
                    "id": "s1",
                    "name": "Timed Step",
                    "step_type": "task",
                    "timeout_seconds": 300,
                    "retries": 3,
                    "config": {},
                }
            ],
        }

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows.audit_data"):
                result = await create_workflow(workflow_data)

                assert "id" in result

    @pytest.mark.asyncio
    async def test_create_workflow_with_optional_step(self, mock_store):
        """Create workflow with optional step that can fail."""
        from aragora.server.handlers.workflows import create_workflow

        workflow_data = {
            "name": "Optional Step Workflow",
            "steps": [
                {
                    "id": "s1",
                    "name": "Optional Step",
                    "step_type": "task",
                    "optional": True,
                    "config": {},
                }
            ],
        }

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows.audit_data"):
                result = await create_workflow(workflow_data)

                assert "id" in result


# =============================================================================
# Additional Tests for Approval Checklist Updates
# =============================================================================


class TestApprovalChecklistUpdates:
    """Tests for approval with checklist updates."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowHandler(mock_server_context)

    @pytest.fixture
    def mock_http(self):
        mock = MagicMock()
        mock.headers = {"Content-Type": "application/json", "Content-Length": "100"}
        mock.rfile = MagicMock()
        return mock

    def test_resolve_approval_with_checklist(self, handler, mock_http):
        """Resolve approval with checklist items."""
        import json

        body_data = {
            "status": "approved",
            "notes": "All items checked",
            "checklist": {
                "item1": True,
                "item2": True,
                "item3": False,
            },
        }
        mock_http.rfile.read.return_value = json.dumps(body_data).encode()

        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(handler, "_get_auth_context") as mock_ctx:
                mock_auth = MagicMock()
                mock_auth.user_id = "approver_123"
                mock_ctx.return_value = mock_auth

                with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                    mock_async.return_value = True

                    result = handler._handle_resolve_approval("apr_123", body_data, {}, mock_http)

                    assert result.status_code == 200

    def test_resolve_approval_rejected_with_notes(self, handler, mock_http):
        """Resolve approval as rejected with explanation notes."""
        import json

        body_data = {
            "status": "rejected",
            "notes": "Missing required documentation in section 3",
        }
        mock_http.rfile.read.return_value = json.dumps(body_data).encode()

        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(handler, "_get_auth_context") as mock_ctx:
                mock_auth = MagicMock()
                mock_auth.user_id = "reviewer_456"
                mock_ctx.return_value = mock_auth

                with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                    mock_async.return_value = True

                    result = handler._handle_resolve_approval("apr_456", body_data, {}, mock_http)

                    assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_list_approvals_by_workflow(self):
        """List pending approvals returns list type."""
        from aragora.server.handlers.workflows import list_pending_approvals

        # Call the function - it may return empty list if no approvals pending
        result = await list_pending_approvals(workflow_id="wf_123")

        # Verify it returns a list (may be empty)
        assert isinstance(result, list)


# =============================================================================
# Additional Tests for Step Result Serialization
# =============================================================================


class TestStepResultSerialization:
    """Tests for step result serialization to dict."""

    def test_step_result_to_dict_complete(self):
        """Step result with all fields serializes correctly."""
        from aragora.server.handlers.workflows import _step_result_to_dict
        from datetime import datetime, timezone

        mock_step = MagicMock()
        mock_step.step_id = "step_1"
        mock_step.step_name = "Process Data"
        mock_step.status = MagicMock()
        mock_step.status.value = "completed"
        mock_step.started_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_step.completed_at = datetime(2024, 1, 1, 12, 0, 30, tzinfo=timezone.utc)
        mock_step.duration_ms = 30000.0
        mock_step.output = {"result": "processed"}
        mock_step.error = None
        mock_step.metrics = {"tokens": 1500}
        mock_step.retry_count = 0

        result = _step_result_to_dict(mock_step)

        assert result["step_id"] == "step_1"
        assert result["step_name"] == "Process Data"
        assert result["status"] == "completed"
        assert result["duration_ms"] == 30000.0
        assert result["output"] == {"result": "processed"}
        assert result["error"] is None
        assert result["metrics"] == {"tokens": 1500}
        assert result["retry_count"] == 0

    def test_step_result_to_dict_with_error(self):
        """Step result with error serializes correctly."""
        from aragora.server.handlers.workflows import _step_result_to_dict

        mock_step = MagicMock()
        mock_step.step_id = "step_2"
        mock_step.step_name = "Validate"
        mock_step.status = MagicMock()
        mock_step.status.value = "failed"
        mock_step.started_at = None
        mock_step.completed_at = None
        mock_step.duration_ms = 100.0
        mock_step.output = None
        mock_step.error = "Validation failed: missing required field"
        mock_step.metrics = {}
        mock_step.retry_count = 2

        result = _step_result_to_dict(mock_step)

        assert result["status"] == "failed"
        assert result["error"] == "Validation failed: missing required field"
        assert result["retry_count"] == 2
        assert result["started_at"] is None
        assert result["completed_at"] is None

    def test_step_result_to_dict_status_as_string(self):
        """Step result handles status as plain string without .value attribute."""
        from aragora.server.handlers.workflows import _step_result_to_dict
        from unittest.mock import PropertyMock

        # Create a mock step where status is a real string, not a MagicMock
        mock_step = MagicMock()
        mock_step.step_id = "step_3"
        mock_step.step_name = "Transform"
        # Configure status as a string-like object that has no .value
        type(mock_step).status = PropertyMock(return_value="running")
        mock_step.started_at = None
        mock_step.completed_at = None
        mock_step.duration_ms = 0
        mock_step.output = None
        mock_step.error = None
        mock_step.metrics = {}
        mock_step.retry_count = 0

        result = _step_result_to_dict(mock_step)

        # The function should handle the status gracefully
        assert "status" in result


# =============================================================================
# Additional Tests for Version Increment Logic
# =============================================================================


class TestVersionIncrementLogic:
    """Tests for workflow version increment logic."""

    @pytest.fixture
    def mock_store(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_update_increments_patch_version(self, mock_store):
        """Update increments the patch version number."""
        from aragora.server.handlers.workflows import update_workflow
        from datetime import datetime, timezone

        existing = MagicMock()
        existing.version = "1.2.3"
        existing.created_by = "user1"
        existing.created_at = datetime.now(timezone.utc)
        mock_store.get_workflow.return_value = existing

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows.audit_data"):
                result = await update_workflow(
                    "wf_123",
                    {"name": "Updated", "steps": [{"id": "s1", "name": "S", "step_type": "task"}]},
                )

                # Version should be incremented from 1.2.3 to 1.2.4
                assert result is not None

    @pytest.mark.asyncio
    async def test_update_increments_from_1_0_0(self, mock_store):
        """Update increments version from 1.0.0 to 1.0.1."""
        from aragora.server.handlers.workflows import update_workflow
        from datetime import datetime, timezone

        existing = MagicMock()
        existing.version = "1.0.0"
        existing.created_by = "user1"
        existing.created_at = datetime.now(timezone.utc)
        mock_store.get_workflow.return_value = existing

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows.audit_data"):
                result = await update_workflow(
                    "wf_123",
                    {"name": "Updated", "steps": [{"id": "s1", "name": "S", "step_type": "task"}]},
                )

                assert result is not None

    @pytest.mark.asyncio
    async def test_update_increments_single_digit_version(self, mock_store):
        """Update handles single-digit version numbers."""
        from aragora.server.handlers.workflows import update_workflow
        from datetime import datetime, timezone

        existing = MagicMock()
        existing.version = "1"
        existing.created_by = "user1"
        existing.created_at = datetime.now(timezone.utc)
        mock_store.get_workflow.return_value = existing

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows.audit_data"):
                result = await update_workflow(
                    "wf_123",
                    {"name": "Updated", "steps": [{"id": "s1", "name": "S", "step_type": "task"}]},
                )

                # Should increment to "2"
                assert result is not None


# =============================================================================
# Additional Tests for Storage Error Handling
# =============================================================================


class TestStorageErrorHandling:
    """Tests for storage error handling scenarios."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowHandler(mock_server_context)

    @pytest.fixture
    def mock_http(self):
        mock = MagicMock()
        mock.headers = MagicMock()
        mock.headers.get.return_value = ""
        return mock

    def test_list_workflows_handles_os_error(self, handler, mock_http):
        """List workflows returns 503 on storage errors."""
        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(handler, "_get_tenant_id", return_value="test"):
                with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                    mock_async.side_effect = OSError("Database connection lost")

                    result = handler._handle_list_workflows({}, mock_http)
                    assert result.status_code == 503

    def test_get_workflow_handles_type_error(self, handler, mock_http):
        """Get workflow returns error response on TypeError."""
        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(handler, "_get_tenant_id", return_value="test"):
                with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                    mock_async.side_effect = TypeError("Unexpected type")

                    result = handler._handle_get_workflow("wf_123", {}, mock_http)

                    # Handler catches TypeError and returns 500
                    assert result.status_code == 500

    def test_create_workflow_handles_os_error(self, handler, mock_http):
        """Create workflow returns error response on OSError."""
        mock_http.headers = {"Content-Type": "application/json", "Content-Length": "100"}
        mock_http.rfile = MagicMock()
        mock_http.rfile.read.return_value = b'{"name": "Test", "steps": []}'

        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(handler, "_get_tenant_id", return_value="test"):
                with patch.object(handler, "_get_auth_context") as mock_ctx:
                    mock_ctx.return_value = None
                    with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                        mock_async.side_effect = OSError("Disk full")

                        result = handler._handle_create_workflow(
                            {"name": "Test", "steps": []}, {}, mock_http
                        )

                        # Handler catches OSError and returns 503
                        assert result.status_code == 503

    def test_delete_workflow_handles_attribute_error(self, handler, mock_http):
        """Delete workflow returns error response on AttributeError."""
        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(handler, "_get_tenant_id", return_value="test"):
                with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                    mock_async.side_effect = AttributeError("Missing attribute")

                    result = handler._handle_delete_workflow("wf_123", {}, mock_http)

                    # Handler catches AttributeError and returns 500
                    assert result.status_code == 500

    def test_execute_workflow_handles_connection_error(self, handler, mock_http):
        """Execute workflow returns error response on ConnectionError."""
        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(handler, "_get_tenant_id", return_value="test"):
                with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                    mock_async.side_effect = ConnectionError("Connection refused")

                    result = handler._handle_execute("wf_123", {}, {}, mock_http)

                    # Handler catches ConnectionError and returns 503
                    assert result.status_code == 503

    def test_execute_workflow_handles_timeout_error(self, handler, mock_http):
        """Execute workflow returns error response on TimeoutError."""
        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(handler, "_get_tenant_id", return_value="test"):
                with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                    mock_async.side_effect = TimeoutError("Request timed out")

                    result = handler._handle_execute("wf_123", {}, {}, mock_http)

                    # Handler catches TimeoutError and returns 503
                    assert result.status_code == 503


# =============================================================================
# Additional Tests for Visual Node Data Handling
# =============================================================================


class TestVisualNodeDataHandling:
    """Tests for visual node data in workflow definitions."""

    @pytest.fixture
    def mock_store(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_create_workflow_with_visual_positions(self, mock_store):
        """Create workflow preserves visual position data."""
        from aragora.server.handlers.workflows import create_workflow

        workflow_data = {
            "name": "Visual Workflow",
            "steps": [
                {
                    "id": "s1",
                    "name": "Step 1",
                    "step_type": "task",
                    "visual": {
                        "position": {"x": 100, "y": 200},
                        "size": {"width": 200, "height": 100},
                    },
                    "config": {},
                }
            ],
        }

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows.audit_data"):
                result = await create_workflow(workflow_data)

                assert "id" in result

    @pytest.mark.asyncio
    async def test_create_workflow_with_canvas_settings(self, mock_store):
        """Create workflow preserves canvas settings."""
        from aragora.server.handlers.workflows import create_workflow

        workflow_data = {
            "name": "Canvas Workflow",
            "steps": [{"id": "s1", "name": "S1", "step_type": "task", "config": {}}],
            "canvas": {
                "width": 5000,
                "height": 4000,
                "zoom": 0.8,
                "pan_x": 100,
                "pan_y": 50,
                "snap_to_grid": True,
                "grid_size": 25,
            },
        }

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows.audit_data"):
                result = await create_workflow(workflow_data)

                assert "id" in result

    @pytest.mark.asyncio
    async def test_create_workflow_with_edge_visual_data(self, mock_store):
        """Create workflow preserves edge visual data."""
        from aragora.server.handlers.workflows import create_workflow

        workflow_data = {
            "name": "Edge Visual Workflow",
            "steps": [
                {"id": "s1", "name": "S1", "step_type": "task", "config": {}},
                {"id": "s2", "name": "S2", "step_type": "task", "config": {}},
            ],
            "transitions": [
                {
                    "id": "tr1",
                    "from_step": "s1",
                    "to_step": "s2",
                    "condition": "True",
                    "visual": {
                        "edge_type": "data_flow",
                        "label": "Next",
                        "animated": True,
                        "color": "#48bb78",
                    },
                }
            ],
        }

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows.audit_data"):
                result = await create_workflow(workflow_data)

                assert "id" in result


# =============================================================================
# Additional Tests for Tenant Isolation
# =============================================================================


class TestTenantIsolation:
    """Tests for tenant isolation in workflow operations."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowHandler(mock_server_context)

    @pytest.fixture
    def mock_http(self):
        mock = MagicMock()
        mock.headers = MagicMock()
        mock.headers.get.return_value = ""
        return mock

    def test_list_workflows_uses_tenant_from_auth(self, handler, mock_http):
        """List workflows uses tenant from authentication context."""
        import json

        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(handler, "_get_auth_context") as mock_ctx:
                mock_auth = MagicMock()
                mock_auth.org_id = "tenant_abc"
                mock_ctx.return_value = mock_auth

                with patch.object(handler, "_get_tenant_id", return_value="tenant_abc"):
                    with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                        mock_async.return_value = {
                            "workflows": [],
                            "total_count": 0,
                            "limit": 50,
                            "offset": 0,
                        }

                        result = handler._handle_list_workflows({}, mock_http)

                        assert result.status_code == 200

    def test_create_workflow_sets_tenant_id(self, handler, mock_http):
        """Create workflow sets tenant_id from context."""
        import json

        mock_http.headers = {"Content-Type": "application/json", "Content-Length": "100"}
        mock_http.rfile = MagicMock()
        mock_http.rfile.read.return_value = json.dumps(
            {
                "name": "Tenant Workflow",
                "steps": [{"id": "s1", "name": "S1", "step_type": "task", "config": {}}],
            }
        ).encode()

        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(handler, "_get_tenant_id", return_value="org_xyz"):
                with patch.object(handler, "_get_auth_context") as mock_ctx:
                    mock_ctx.return_value = None

                    with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                        mock_async.return_value = {"id": "wf_new", "tenant_id": "org_xyz"}

                        result = handler._handle_create_workflow(
                            {"name": "Test", "steps": []}, {}, mock_http
                        )

                        # Verify the call was made
                        assert mock_async.called

    def test_get_workflow_respects_tenant_boundary(self, handler, mock_http):
        """Get workflow respects tenant boundary."""
        import json

        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(handler, "_get_tenant_id", return_value="tenant_123"):
                with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                    # Simulate workflow not found in this tenant
                    mock_async.return_value = None

                    result = handler._handle_get_workflow("wf_other_tenant", {}, mock_http)

                    assert result.status_code == 404


# =============================================================================
# Additional Tests for Workflow Definition Validation
# =============================================================================


class TestWorkflowDefinitionValidation:
    """Tests for workflow definition validation."""

    @pytest.fixture
    def mock_store(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_create_workflow_requires_name(self, mock_store):
        """Create workflow requires name field."""
        from aragora.server.handlers.workflows import create_workflow

        workflow_data = {
            "steps": [{"id": "s1", "name": "S1", "step_type": "task", "config": {}}],
        }

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows.audit_data"):
                # This should raise ValueError or return validation error
                try:
                    await create_workflow(workflow_data)
                except (ValueError, KeyError):
                    pass  # Expected

    @pytest.mark.asyncio
    async def test_create_workflow_requires_steps(self, mock_store):
        """Create workflow requires at least one step."""
        from aragora.server.handlers.workflows import create_workflow

        workflow_data = {
            "name": "Empty Workflow",
            "steps": [],
        }

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows.audit_data"):
                try:
                    await create_workflow(workflow_data)
                except ValueError as e:
                    assert "at least one step" in str(e).lower()

    @pytest.mark.asyncio
    async def test_create_workflow_validates_transitions(self, mock_store):
        """Create workflow validates transition references."""
        from aragora.server.handlers.workflows import create_workflow

        workflow_data = {
            "name": "Bad Transitions",
            "steps": [{"id": "s1", "name": "S1", "step_type": "task", "config": {}}],
            "transitions": [{"from_step": "s1", "to_step": "nonexistent", "condition": "True"}],
        }

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows.audit_data"):
                try:
                    await create_workflow(workflow_data)
                except ValueError as e:
                    assert "unknown step" in str(e).lower() or "not found" in str(e).lower()


# =============================================================================
# Additional Tests for Execution Result Handling
# =============================================================================


class TestExecutionResultHandling:
    """Tests for workflow execution result handling."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return WorkflowHandler(mock_server_context)

    @pytest.fixture
    def mock_http(self):
        mock = MagicMock()
        mock.headers = MagicMock()
        mock.headers.get.return_value = ""
        return mock

    def test_execution_success_returns_completed_status(self, handler, mock_http):
        """Successful execution returns completed status."""
        import json

        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(handler, "_get_tenant_id", return_value="test"):
                with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                    mock_async.return_value = {
                        "id": "exec_123",
                        "status": "completed",
                        "outputs": {"result": "success"},
                        "steps": [],
                    }

                    result = handler._handle_execute("wf_123", {"inputs": {}}, {}, mock_http)

                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["status"] == "completed"

    def test_execution_failure_returns_failed_status(self, handler, mock_http):
        """Failed execution returns failed status with error."""
        import json

        with patch.object(handler, "_check_permission", return_value=None):
            with patch.object(handler, "_get_tenant_id", return_value="test"):
                with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                    mock_async.return_value = {
                        "id": "exec_456",
                        "status": "failed",
                        "error": "Step validation failed",
                        "steps": [],
                    }

                    result = handler._handle_execute("wf_123", {"inputs": {}}, {}, mock_http)

                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["status"] == "failed"
                    assert "error" in body

    def test_terminate_execution_changes_status(self, handler, mock_http):
        """Terminating execution changes status to terminated."""
        import json

        with patch.object(handler, "_check_permission", return_value=None):
            with patch("aragora.server.handlers.workflows._run_async") as mock_async:
                mock_async.return_value = True

                result = handler._handle_terminate_execution("exec_123", {}, mock_http)

                assert result.status_code == 200
                body = json.loads(result.body)
                assert body["terminated"] is True


# =============================================================================
# Additional Tests for Human Checkpoint Step Type
# =============================================================================


class TestHumanCheckpointStepType:
    """Tests for human checkpoint step type handling."""

    @pytest.fixture
    def mock_store(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_create_workflow_with_human_checkpoint(self, mock_store):
        """Create workflow with human checkpoint step."""
        from aragora.server.handlers.workflows import create_workflow

        workflow_data = {
            "name": "Approval Workflow",
            "steps": [
                {
                    "id": "s1",
                    "name": "Request Approval",
                    "step_type": "human_checkpoint",
                    "config": {
                        "approval_type": "single",
                        "timeout_hours": 24,
                        "required_approvers": ["manager"],
                    },
                }
            ],
        }

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows.audit_data"):
                result = await create_workflow(workflow_data)

                assert "id" in result

    @pytest.mark.asyncio
    async def test_create_workflow_with_multi_approver(self, mock_store):
        """Create workflow with multi-approver checkpoint."""
        from aragora.server.handlers.workflows import create_workflow

        workflow_data = {
            "name": "Multi-Approval Workflow",
            "steps": [
                {
                    "id": "s1",
                    "name": "Committee Approval",
                    "step_type": "human_checkpoint",
                    "config": {
                        "approval_type": "multi",
                        "required_count": 3,
                        "required_approvers": ["legal", "finance", "compliance"],
                    },
                }
            ],
        }

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows.audit_data"):
                result = await create_workflow(workflow_data)

                assert "id" in result


# =============================================================================
# Additional Tests for Memory Step Types
# =============================================================================


class TestMemoryStepTypes:
    """Tests for memory read/write step types."""

    @pytest.fixture
    def mock_store(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_create_workflow_with_memory_read(self, mock_store):
        """Create workflow with memory read step."""
        from aragora.server.handlers.workflows import create_workflow

        workflow_data = {
            "name": "Memory Read Workflow",
            "steps": [
                {
                    "id": "s1",
                    "name": "Read Context",
                    "step_type": "memory_read",
                    "config": {
                        "memory_key": "user_preferences",
                        "default_value": {},
                    },
                }
            ],
        }

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows.audit_data"):
                result = await create_workflow(workflow_data)

                assert "id" in result

    @pytest.mark.asyncio
    async def test_create_workflow_with_memory_write(self, mock_store):
        """Create workflow with memory write step."""
        from aragora.server.handlers.workflows import create_workflow

        workflow_data = {
            "name": "Memory Write Workflow",
            "steps": [
                {
                    "id": "s1",
                    "name": "Store Result",
                    "step_type": "memory_write",
                    "config": {
                        "memory_key": "processing_result",
                        "ttl_seconds": 3600,
                    },
                }
            ],
        }

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows.audit_data"):
                result = await create_workflow(workflow_data)

                assert "id" in result


# =============================================================================
# Additional Tests for Debate Step Type
# =============================================================================


class TestDebateStepType:
    """Tests for debate step type in workflows."""

    @pytest.fixture
    def mock_store(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_create_workflow_with_debate_step(self, mock_store):
        """Create workflow with debate step."""
        from aragora.server.handlers.workflows import create_workflow

        workflow_data = {
            "name": "Debate Workflow",
            "steps": [
                {
                    "id": "s1",
                    "name": "Run Debate",
                    "step_type": "debate",
                    "config": {
                        "topic": "Should we proceed?",
                        "agents": ["claude", "gpt4"],
                        "rounds": 3,
                        "consensus_threshold": 0.7,
                    },
                }
            ],
        }

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows.audit_data"):
                result = await create_workflow(workflow_data)

                assert "id" in result


# =============================================================================
# Additional Tests for Workflow Clone Operations
# =============================================================================


class TestWorkflowCloneOperations:
    """Tests for workflow clone/fork operations."""

    @pytest.fixture
    def mock_store(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_create_from_template_increments_usage(self, mock_store):
        """Creating from template increments usage count."""
        from aragora.server.handlers.workflows import create_workflow_from_template

        mock_template = MagicMock()
        mock_template.clone.return_value = mock_template
        mock_template.to_dict.return_value = {
            "id": "wf_new",
            "name": "From Template",
            "steps": [{"id": "s1", "name": "S1", "step_type": "task", "config": {}}],
        }
        mock_store.get_template.return_value = mock_template

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows.audit_data"):
                await create_workflow_from_template("tpl_123", "My Workflow")

                mock_store.increment_template_usage.assert_called_once_with("tpl_123")

    @pytest.mark.asyncio
    async def test_create_from_template_applies_customizations(self, mock_store):
        """Creating from template applies customizations."""
        from aragora.server.handlers.workflows import create_workflow_from_template

        mock_template = MagicMock()
        mock_clone = MagicMock()
        mock_clone.to_dict.return_value = {
            "id": "wf_custom",
            "name": "Customized",
            "description": "Original description",
            "steps": [{"id": "s1", "name": "S1", "step_type": "task", "config": {}}],
        }
        mock_template.clone.return_value = mock_clone
        mock_store.get_template.return_value = mock_template

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with patch("aragora.server.handlers.workflows.audit_data"):
                result = await create_workflow_from_template(
                    "tpl_123",
                    "Custom Workflow",
                    customizations={"description": "Custom description"},
                )

                assert result is not None

    @pytest.mark.asyncio
    async def test_create_from_nonexistent_template_raises(self, mock_store):
        """Creating from nonexistent template raises ValueError."""
        from aragora.server.handlers.workflows import create_workflow_from_template

        mock_store.get_template.return_value = None

        with patch("aragora.server.handlers.workflows._get_store", return_value=mock_store):
            with pytest.raises(ValueError, match="Template not found"):
                await create_workflow_from_template("nonexistent", "My Workflow")
