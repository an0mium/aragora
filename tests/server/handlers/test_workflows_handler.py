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

    def test_routes_count(self, handler):
        """ROUTES contains expected number of routes."""
        # 7 routes: workflows, workflows/*, templates, approvals, approvals/*,
        # executions, executions/*
        assert len(handler.ROUTES) == 7


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
