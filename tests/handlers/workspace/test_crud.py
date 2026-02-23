"""Tests for workspace CRUD handler (WorkspaceCrudMixin).

Tests the CRUD endpoints:
- POST   /api/v1/workspaces                 - Create workspace
- GET    /api/v1/workspaces                  - List workspaces
- GET    /api/v1/workspaces/{workspace_id}   - Get workspace details
- DELETE /api/v1/workspaces/{workspace_id}   - Delete workspace
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status(result) -> int:
    """Extract status code from HandlerResult or infer from dict."""
    if isinstance(result, dict):
        return 200
    return result.status_code


def _body(result) -> dict[str, Any]:
    """Extract JSON body from HandlerResult or return dict as-is."""
    if isinstance(result, dict):
        return result
    try:
        return json.loads(result.body.decode("utf-8"))
    except (json.JSONDecodeError, AttributeError, UnicodeDecodeError):
        return {}


def _error(result) -> str:
    """Extract error message from HandlerResult."""
    body = _body(result)
    return body.get("error", "")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_rate_limiters():
    """Reset rate limiter state between tests."""
    try:
        from aragora.server.middleware.rate_limit.registry import (
            reset_rate_limiters as _reset,
        )

        _reset()
    except ImportError:
        pass
    yield
    try:
        from aragora.server.middleware.rate_limit.registry import (
            reset_rate_limiters as _reset,
        )

        _reset()
    except ImportError:
        pass


@pytest.fixture
def mock_workspace_module():
    """Patch the workspace_module imported via _mod().

    Returns a MagicMock that provides all the symbols the CRUD mixin
    accesses through ``_mod()``.
    """
    m = MagicMock()

    # Auth context returned by extract_user_from_request
    auth_ctx = MagicMock()
    auth_ctx.is_authenticated = True
    auth_ctx.user_id = "test-user-001"
    auth_ctx.org_id = "test-org-001"
    m.extract_user_from_request.return_value = auth_ctx

    # Permission constants
    m.PERM_WORKSPACE_READ = "workspace:read"
    m.PERM_WORKSPACE_WRITE = "workspace:write"
    m.PERM_WORKSPACE_DELETE = "workspace:delete"

    # Audit types
    m.AuditAction = MagicMock()
    m.AuditAction.CREATE_WORKSPACE = "create_workspace"
    m.AuditAction.DELETE_WORKSPACE = "delete_workspace"
    m.AuditOutcome = MagicMock()
    m.AuditOutcome.SUCCESS = "success"
    m.Actor = MagicMock()
    m.Resource = MagicMock()

    # AccessDeniedException
    m.AccessDeniedException = type("AccessDeniedException", (Exception,), {})

    # Response helpers -- delegate to real implementations
    from aragora.server.handlers.base import json_response, error_response

    m.json_response = json_response
    m.error_response = error_response

    with patch(
        "aragora.server.handlers.workspace.crud._mod",
        return_value=m,
    ):
        yield m


@pytest.fixture
def handler(mock_workspace_module):
    """Create a WorkspaceCrudMixin instance with mocked dependencies."""
    from aragora.server.handlers.workspace.crud import WorkspaceCrudMixin

    class _TestHandler(WorkspaceCrudMixin):
        """Concrete handler combining mixin with mock infrastructure."""

        def __init__(self):
            self._mock_isolation_manager = MagicMock()
            self._mock_isolation_manager.create_workspace = AsyncMock()
            self._mock_isolation_manager.list_workspaces = AsyncMock(return_value=[])
            self._mock_isolation_manager.get_workspace = AsyncMock()
            self._mock_isolation_manager.delete_workspace = AsyncMock()
            self._mock_user_store = MagicMock()
            self._mock_audit_log = MagicMock()
            self._mock_audit_log.log = AsyncMock()

            # Set up default workspace mock for create
            ws_mock = MagicMock()
            ws_mock.id = "ws-new-001"
            ws_mock.to_dict.return_value = {
                "id": "ws-new-001",
                "name": "Test Workspace",
                "organization_id": "test-org-001",
                "created_by": "test-user-001",
            }
            self._mock_isolation_manager.create_workspace.return_value = ws_mock
            self._default_workspace = ws_mock

        def _get_user_store(self):
            return self._mock_user_store

        def _get_isolation_manager(self):
            return self._mock_isolation_manager

        def _get_audit_log(self):
            return self._mock_audit_log

        def _run_async(self, coro):
            """Run coroutine synchronously for tests."""
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        return pool.submit(asyncio.run, coro).result()
                return loop.run_until_complete(coro)
            except RuntimeError:
                return asyncio.run(coro)

        def _check_rbac_permission(self, handler, perm, auth_ctx):
            return None  # Always allow in tests

        def read_json_body(self, handler):
            return getattr(handler, "_json_body", None)

    return _TestHandler()


@pytest.fixture
def make_handler_request():
    """Factory for creating mock HTTP handler objects."""

    def _make(
        path: str = "/api/v1/workspaces",
        method: str = "GET",
        body: dict[str, Any] | None = None,
        query: str = "",
    ):
        h = MagicMock()
        full_path = f"{path}?{query}" if query else path
        h.path = full_path
        h.command = method
        h.headers = {"Content-Length": "0"}
        if body is not None:
            h._json_body = body
        else:
            h._json_body = None
        return h

    return _make


# ===========================================================================
# POST /api/v1/workspaces - Create Workspace
# ===========================================================================


class TestCreateWorkspace:
    """Test the _handle_create_workspace endpoint."""

    def test_create_workspace_success(self, handler, make_handler_request):
        req = make_handler_request(method="POST", body={"name": "My Workspace"})
        result = handler._handle_create_workspace(req)
        assert _status(result) == 201
        body = _body(result)
        assert "workspace" in body
        assert body["workspace"]["id"] == "ws-new-001"
        assert body["message"] == "Workspace created successfully"

    def test_create_workspace_passes_name(self, handler, make_handler_request):
        req = make_handler_request(method="POST", body={"name": "Engineering"})
        handler._handle_create_workspace(req)

        call_kwargs = handler._mock_isolation_manager.create_workspace.call_args.kwargs
        assert call_kwargs["name"] == "Engineering"

    def test_create_workspace_passes_org_id_from_auth(self, handler, make_handler_request):
        """Organization ID should come from auth context, not from body."""
        req = make_handler_request(method="POST", body={"name": "WS"})
        handler._handle_create_workspace(req)

        call_kwargs = handler._mock_isolation_manager.create_workspace.call_args.kwargs
        assert call_kwargs["organization_id"] == "test-org-001"

    def test_create_workspace_passes_created_by(self, handler, make_handler_request):
        req = make_handler_request(method="POST", body={"name": "WS"})
        handler._handle_create_workspace(req)

        call_kwargs = handler._mock_isolation_manager.create_workspace.call_args.kwargs
        assert call_kwargs["created_by"] == "test-user-001"

    def test_create_workspace_passes_initial_members(self, handler, make_handler_request):
        req = make_handler_request(
            method="POST",
            body={"name": "WS", "members": ["user-a", "user-b"]},
        )
        handler._handle_create_workspace(req)

        call_kwargs = handler._mock_isolation_manager.create_workspace.call_args.kwargs
        assert call_kwargs["initial_members"] == ["user-a", "user-b"]

    def test_create_workspace_default_members_empty(self, handler, make_handler_request):
        req = make_handler_request(method="POST", body={"name": "WS"})
        handler._handle_create_workspace(req)

        call_kwargs = handler._mock_isolation_manager.create_workspace.call_args.kwargs
        assert call_kwargs["initial_members"] == []

    def test_create_workspace_missing_name(self, handler, make_handler_request):
        req = make_handler_request(method="POST", body={})
        result = handler._handle_create_workspace(req)
        assert _status(result) == 400
        assert "name" in _error(result).lower()

    def test_create_workspace_empty_name(self, handler, make_handler_request):
        req = make_handler_request(method="POST", body={"name": ""})
        result = handler._handle_create_workspace(req)
        assert _status(result) == 400
        assert "name" in _error(result).lower()

    def test_create_workspace_null_body(self, handler, make_handler_request):
        req = make_handler_request(method="POST", body=None)
        req._json_body = None
        result = handler._handle_create_workspace(req)
        assert _status(result) == 400
        assert "json" in _error(result).lower()

    def test_create_workspace_not_authenticated(
        self, handler, make_handler_request, mock_workspace_module
    ):
        mock_workspace_module.extract_user_from_request.return_value.is_authenticated = False

        req = make_handler_request(method="POST", body={"name": "WS"})
        result = handler._handle_create_workspace(req)
        assert _status(result) == 401

    def test_create_workspace_rbac_denied(
        self, handler, make_handler_request, mock_workspace_module
    ):
        from aragora.server.handlers.base import error_response

        handler._check_rbac_permission = lambda h, p, a: error_response("Forbidden", 403)

        req = make_handler_request(method="POST", body={"name": "WS"})
        result = handler._handle_create_workspace(req)
        assert _status(result) == 403

    def test_create_workspace_no_org_id(
        self, handler, make_handler_request, mock_workspace_module
    ):
        """When auth context has no org_id, should return 400."""
        mock_workspace_module.extract_user_from_request.return_value.org_id = None

        req = make_handler_request(method="POST", body={"name": "WS"})
        result = handler._handle_create_workspace(req)
        assert _status(result) == 400
        assert "organization_id" in _error(result).lower()

    def test_create_workspace_empty_org_id(
        self, handler, make_handler_request, mock_workspace_module
    ):
        """When auth context has empty org_id, should return 400."""
        mock_workspace_module.extract_user_from_request.return_value.org_id = ""

        req = make_handler_request(method="POST", body={"name": "WS"})
        result = handler._handle_create_workspace(req)
        assert _status(result) == 400
        assert "organization_id" in _error(result).lower()

    def test_create_workspace_cross_tenant_blocked(
        self, handler, make_handler_request
    ):
        """Attempting to create a workspace in another org should be rejected."""
        req = make_handler_request(
            method="POST",
            body={"name": "WS", "organization_id": "other-org-999"},
        )
        result = handler._handle_create_workspace(req)
        assert _status(result) == 403
        assert "another organization" in _error(result).lower()

    def test_create_workspace_same_org_id_allowed(
        self, handler, make_handler_request
    ):
        """Specifying the same org_id as the user's own should be allowed."""
        req = make_handler_request(
            method="POST",
            body={"name": "WS", "organization_id": "test-org-001"},
        )
        result = handler._handle_create_workspace(req)
        assert _status(result) == 201

    def test_create_workspace_no_org_id_in_body_is_ok(
        self, handler, make_handler_request
    ):
        """Not specifying organization_id in body is fine (uses auth context)."""
        req = make_handler_request(
            method="POST",
            body={"name": "WS"},
        )
        result = handler._handle_create_workspace(req)
        assert _status(result) == 201

    def test_create_workspace_audit_log_called(self, handler, make_handler_request):
        req = make_handler_request(method="POST", body={"name": "WS"})
        handler._handle_create_workspace(req)
        handler._mock_audit_log.log.assert_called_once()

    def test_create_workspace_audit_log_details(
        self, handler, make_handler_request, mock_workspace_module
    ):
        req = make_handler_request(method="POST", body={"name": "Engineering"})
        handler._handle_create_workspace(req)

        call_kwargs = handler._mock_audit_log.log.call_args.kwargs
        assert call_kwargs["details"]["name"] == "Engineering"
        assert call_kwargs["details"]["org_id"] == "test-org-001"

    def test_create_workspace_audit_action(
        self, handler, make_handler_request, mock_workspace_module
    ):
        req = make_handler_request(method="POST", body={"name": "WS"})
        handler._handle_create_workspace(req)

        call_kwargs = handler._mock_audit_log.log.call_args.kwargs
        assert call_kwargs["action"] == mock_workspace_module.AuditAction.CREATE_WORKSPACE

    def test_create_workspace_audit_outcome_success(
        self, handler, make_handler_request, mock_workspace_module
    ):
        req = make_handler_request(method="POST", body={"name": "WS"})
        handler._handle_create_workspace(req)

        call_kwargs = handler._mock_audit_log.log.call_args.kwargs
        assert call_kwargs["outcome"] == mock_workspace_module.AuditOutcome.SUCCESS

    def test_create_workspace_audit_actor_is_current_user(
        self, handler, make_handler_request, mock_workspace_module
    ):
        req = make_handler_request(method="POST", body={"name": "WS"})
        handler._handle_create_workspace(req)

        mock_workspace_module.Actor.assert_called_with(id="test-user-001", type="user")

    def test_create_workspace_audit_resource_is_workspace(
        self, handler, make_handler_request, mock_workspace_module
    ):
        req = make_handler_request(method="POST", body={"name": "WS"})
        handler._handle_create_workspace(req)

        mock_workspace_module.Resource.assert_called_with(
            id="ws-new-001", type="workspace", workspace_id="ws-new-001"
        )

    def test_create_workspace_emits_handler_event(self, handler, make_handler_request):
        """Verify that creating a workspace emits a CREATED event."""
        with patch(
            "aragora.server.handlers.workspace.crud.emit_handler_event"
        ) as mock_emit:
            req = make_handler_request(method="POST", body={"name": "WS"})
            handler._handle_create_workspace(req)

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[0][0] == "workspace"
            # Second positional arg is the event type (CREATED)
            assert call_args[0][2] == {"workspace_id": "ws-new-001"}
            assert call_args[1]["user_id"] == "test-user-001"

    def test_create_workspace_rbac_checks_write_permission(
        self, handler, make_handler_request
    ):
        """Verify create workspace uses PERM_WORKSPACE_WRITE."""
        captured_perms = []

        def capture_rbac(h, perm, auth_ctx):
            captured_perms.append(perm)
            return None

        handler._check_rbac_permission = capture_rbac

        req = make_handler_request(method="POST", body={"name": "WS"})
        handler._handle_create_workspace(req)
        assert "workspace:write" in captured_perms

    def test_create_workspace_returns_workspace_dict(self, handler, make_handler_request):
        """Response should contain the workspace's serialized form."""
        req = make_handler_request(method="POST", body={"name": "WS"})
        result = handler._handle_create_workspace(req)
        body = _body(result)
        assert body["workspace"]["name"] == "Test Workspace"
        assert body["workspace"]["organization_id"] == "test-org-001"

    def test_create_workspace_runtime_error(self, handler, make_handler_request):
        """RuntimeError in isolation manager should return 500."""
        handler._mock_isolation_manager.create_workspace = AsyncMock(
            side_effect=RuntimeError("db fail")
        )

        req = make_handler_request(method="POST", body={"name": "WS"})
        result = handler._handle_create_workspace(req)
        assert _status(result) == 500

    def test_create_workspace_value_error(self, handler, make_handler_request):
        """ValueError in isolation manager should return 400."""
        handler._mock_isolation_manager.create_workspace = AsyncMock(
            side_effect=ValueError("invalid workspace config")
        )

        req = make_handler_request(method="POST", body={"name": "WS"})
        result = handler._handle_create_workspace(req)
        assert _status(result) == 400


# ===========================================================================
# GET /api/v1/workspaces - List Workspaces
# ===========================================================================


class TestListWorkspaces:
    """Test the _handle_list_workspaces endpoint."""

    def test_list_workspaces_empty(self, handler, make_handler_request):
        req = make_handler_request()
        result = handler._handle_list_workspaces(req, {})
        assert _status(result) == 200
        body = _body(result)
        assert body["workspaces"] == []
        assert body["total"] == 0

    def test_list_workspaces_single(self, handler, make_handler_request):
        ws = MagicMock()
        ws.to_dict.return_value = {
            "id": "ws-1",
            "name": "Workspace 1",
            "organization_id": "test-org-001",
        }
        handler._mock_isolation_manager.list_workspaces = AsyncMock(
            return_value=[ws]
        )

        req = make_handler_request()
        result = handler._handle_list_workspaces(req, {})
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["workspaces"][0]["id"] == "ws-1"
        assert body["workspaces"][0]["name"] == "Workspace 1"

    def test_list_workspaces_multiple(self, handler, make_handler_request):
        workspaces = []
        for i in range(5):
            ws = MagicMock()
            ws.to_dict.return_value = {"id": f"ws-{i}", "name": f"WS {i}"}
            workspaces.append(ws)

        handler._mock_isolation_manager.list_workspaces = AsyncMock(
            return_value=workspaces
        )

        req = make_handler_request()
        result = handler._handle_list_workspaces(req, {})
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 5
        assert len(body["workspaces"]) == 5

    def test_list_workspaces_uses_auth_org_id(self, handler, make_handler_request):
        """Should list workspaces using the user's own org_id."""
        req = make_handler_request()
        handler._handle_list_workspaces(req, {})

        call_kwargs = handler._mock_isolation_manager.list_workspaces.call_args.kwargs
        assert call_kwargs["organization_id"] == "test-org-001"
        assert call_kwargs["actor"] == "test-user-001"

    def test_list_workspaces_not_authenticated(
        self, handler, make_handler_request, mock_workspace_module
    ):
        mock_workspace_module.extract_user_from_request.return_value.is_authenticated = False

        req = make_handler_request()
        result = handler._handle_list_workspaces(req, {})
        assert _status(result) == 401

    def test_list_workspaces_rbac_denied(
        self, handler, make_handler_request, mock_workspace_module
    ):
        from aragora.server.handlers.base import error_response

        handler._check_rbac_permission = lambda h, p, a: error_response("Forbidden", 403)

        req = make_handler_request()
        result = handler._handle_list_workspaces(req, {})
        assert _status(result) == 403

    def test_list_workspaces_cross_tenant_blocked(
        self, handler, make_handler_request
    ):
        """Attempting to list workspaces for another org should be rejected."""
        req = make_handler_request()
        result = handler._handle_list_workspaces(
            req, {"organization_id": "other-org-999"}
        )
        assert _status(result) == 403
        assert "another organization" in _error(result).lower()

    def test_list_workspaces_same_org_id_allowed(
        self, handler, make_handler_request
    ):
        """Specifying the user's own org_id should succeed."""
        req = make_handler_request()
        result = handler._handle_list_workspaces(
            req, {"organization_id": "test-org-001"}
        )
        assert _status(result) == 200

    def test_list_workspaces_no_org_filter_is_ok(
        self, handler, make_handler_request
    ):
        """Not providing organization_id in query is fine."""
        req = make_handler_request()
        result = handler._handle_list_workspaces(req, {})
        assert _status(result) == 200

    def test_list_workspaces_rbac_checks_read_permission(
        self, handler, make_handler_request
    ):
        """Verify list workspaces uses PERM_WORKSPACE_READ."""
        captured_perms = []

        def capture_rbac(h, perm, auth_ctx):
            captured_perms.append(perm)
            return None

        handler._check_rbac_permission = capture_rbac

        req = make_handler_request()
        handler._handle_list_workspaces(req, {})
        assert "workspace:read" in captured_perms

    def test_list_workspaces_runtime_error(self, handler, make_handler_request):
        handler._mock_isolation_manager.list_workspaces = AsyncMock(
            side_effect=RuntimeError("db fail")
        )

        req = make_handler_request()
        result = handler._handle_list_workspaces(req, {})
        assert _status(result) == 500

    def test_list_workspaces_os_error(self, handler, make_handler_request):
        handler._mock_isolation_manager.list_workspaces = AsyncMock(
            side_effect=OSError("io fail")
        )

        req = make_handler_request()
        result = handler._handle_list_workspaces(req, {})
        assert _status(result) == 500

    def test_list_workspaces_value_error(self, handler, make_handler_request):
        handler._mock_isolation_manager.list_workspaces = AsyncMock(
            side_effect=ValueError("bad params")
        )

        req = make_handler_request()
        result = handler._handle_list_workspaces(req, {})
        assert _status(result) == 400

    def test_list_workspaces_type_error(self, handler, make_handler_request):
        handler._mock_isolation_manager.list_workspaces = AsyncMock(
            side_effect=TypeError("bad type")
        )

        req = make_handler_request()
        result = handler._handle_list_workspaces(req, {})
        assert _status(result) == 400

    def test_list_workspaces_none_org_id_in_query_allowed(
        self, handler, make_handler_request
    ):
        """None value for organization_id should be treated as no filter."""
        req = make_handler_request()
        result = handler._handle_list_workspaces(
            req, {"organization_id": None}
        )
        # None is falsy, so the cross-tenant check should not trigger
        assert _status(result) == 200


# ===========================================================================
# GET /api/v1/workspaces/{workspace_id} - Get Workspace
# ===========================================================================


class TestGetWorkspace:
    """Test the _handle_get_workspace endpoint."""

    def _setup_workspace(self, handler, workspace_id="ws-1"):
        ws = MagicMock()
        ws.to_dict.return_value = {
            "id": workspace_id,
            "name": "Test Workspace",
            "organization_id": "test-org-001",
        }
        handler._mock_isolation_manager.get_workspace = AsyncMock(return_value=ws)
        return ws

    def test_get_workspace_success(self, handler, make_handler_request):
        self._setup_workspace(handler)

        req = make_handler_request(path="/api/v1/workspaces/ws-1")
        result = handler._handle_get_workspace(req, "ws-1")
        assert _status(result) == 200
        body = _body(result)
        assert "workspace" in body
        assert body["workspace"]["id"] == "ws-1"
        assert body["workspace"]["name"] == "Test Workspace"

    def test_get_workspace_passes_workspace_id(self, handler, make_handler_request):
        self._setup_workspace(handler)

        req = make_handler_request(path="/api/v1/workspaces/ws-target")
        handler._handle_get_workspace(req, "ws-target")

        call_kwargs = handler._mock_isolation_manager.get_workspace.call_args.kwargs
        assert call_kwargs["workspace_id"] == "ws-target"

    def test_get_workspace_passes_actor(self, handler, make_handler_request):
        self._setup_workspace(handler)

        req = make_handler_request(path="/api/v1/workspaces/ws-1")
        handler._handle_get_workspace(req, "ws-1")

        call_kwargs = handler._mock_isolation_manager.get_workspace.call_args.kwargs
        assert call_kwargs["actor"] == "test-user-001"

    def test_get_workspace_not_authenticated(
        self, handler, make_handler_request, mock_workspace_module
    ):
        mock_workspace_module.extract_user_from_request.return_value.is_authenticated = False

        req = make_handler_request(path="/api/v1/workspaces/ws-1")
        result = handler._handle_get_workspace(req, "ws-1")
        assert _status(result) == 401

    def test_get_workspace_rbac_denied(
        self, handler, make_handler_request, mock_workspace_module
    ):
        from aragora.server.handlers.base import error_response

        handler._check_rbac_permission = lambda h, p, a: error_response("Forbidden", 403)

        req = make_handler_request(path="/api/v1/workspaces/ws-1")
        result = handler._handle_get_workspace(req, "ws-1")
        assert _status(result) == 403

    def test_get_workspace_access_denied_exception(
        self, handler, make_handler_request, mock_workspace_module
    ):
        exc_class = mock_workspace_module.AccessDeniedException
        handler._mock_isolation_manager.get_workspace = AsyncMock(
            side_effect=exc_class("denied")
        )

        req = make_handler_request(path="/api/v1/workspaces/ws-1")
        result = handler._handle_get_workspace(req, "ws-1")
        assert _status(result) == 403

    def test_get_workspace_rbac_checks_read_permission(
        self, handler, make_handler_request
    ):
        """Verify get workspace uses PERM_WORKSPACE_READ."""
        self._setup_workspace(handler)
        captured_perms = []

        def capture_rbac(h, perm, auth_ctx):
            captured_perms.append(perm)
            return None

        handler._check_rbac_permission = capture_rbac

        req = make_handler_request(path="/api/v1/workspaces/ws-1")
        handler._handle_get_workspace(req, "ws-1")
        assert "workspace:read" in captured_perms

    def test_get_workspace_runtime_error(self, handler, make_handler_request):
        handler._mock_isolation_manager.get_workspace = AsyncMock(
            side_effect=RuntimeError("db fail")
        )

        req = make_handler_request(path="/api/v1/workspaces/ws-1")
        result = handler._handle_get_workspace(req, "ws-1")
        assert _status(result) == 500

    def test_get_workspace_value_error(self, handler, make_handler_request):
        handler._mock_isolation_manager.get_workspace = AsyncMock(
            side_effect=ValueError("invalid id")
        )

        req = make_handler_request(path="/api/v1/workspaces/ws-1")
        result = handler._handle_get_workspace(req, "ws-1")
        assert _status(result) == 400

    def test_get_workspace_returns_full_dict(self, handler, make_handler_request):
        """Response should contain the full workspace dict."""
        ws = MagicMock()
        ws.to_dict.return_value = {
            "id": "ws-detailed",
            "name": "Detailed Workspace",
            "organization_id": "org-001",
            "created_by": "user-001",
            "members": ["user-001", "user-002"],
        }
        handler._mock_isolation_manager.get_workspace = AsyncMock(return_value=ws)

        req = make_handler_request(path="/api/v1/workspaces/ws-detailed")
        result = handler._handle_get_workspace(req, "ws-detailed")
        body = _body(result)
        assert body["workspace"]["created_by"] == "user-001"
        assert body["workspace"]["members"] == ["user-001", "user-002"]


# ===========================================================================
# DELETE /api/v1/workspaces/{workspace_id} - Delete Workspace
# ===========================================================================


class TestDeleteWorkspace:
    """Test the _handle_delete_workspace endpoint."""

    def test_delete_workspace_success(self, handler, make_handler_request):
        req = make_handler_request(
            path="/api/v1/workspaces/ws-1",
            method="DELETE",
        )
        result = handler._handle_delete_workspace(req, "ws-1")
        assert _status(result) == 200
        body = _body(result)
        assert body["message"] == "Workspace deleted successfully"

    def test_delete_workspace_passes_workspace_id(self, handler, make_handler_request):
        req = make_handler_request(method="DELETE")
        handler._handle_delete_workspace(req, "ws-target")

        call_kwargs = handler._mock_isolation_manager.delete_workspace.call_args.kwargs
        assert call_kwargs["workspace_id"] == "ws-target"

    def test_delete_workspace_passes_deleted_by(self, handler, make_handler_request):
        req = make_handler_request(method="DELETE")
        handler._handle_delete_workspace(req, "ws-1")

        call_kwargs = handler._mock_isolation_manager.delete_workspace.call_args.kwargs
        assert call_kwargs["deleted_by"] == "test-user-001"

    def test_delete_workspace_default_force_false(self, handler, make_handler_request):
        """When no body or force not specified, force should be False."""
        req = make_handler_request(method="DELETE")
        handler._handle_delete_workspace(req, "ws-1")

        call_kwargs = handler._mock_isolation_manager.delete_workspace.call_args.kwargs
        assert call_kwargs["force"] is False

    def test_delete_workspace_force_true(self, handler, make_handler_request):
        req = make_handler_request(method="DELETE", body={"force": True})
        handler._handle_delete_workspace(req, "ws-1")

        call_kwargs = handler._mock_isolation_manager.delete_workspace.call_args.kwargs
        assert call_kwargs["force"] is True

    def test_delete_workspace_force_false_explicit(self, handler, make_handler_request):
        req = make_handler_request(method="DELETE", body={"force": False})
        handler._handle_delete_workspace(req, "ws-1")

        call_kwargs = handler._mock_isolation_manager.delete_workspace.call_args.kwargs
        assert call_kwargs["force"] is False

    def test_delete_workspace_null_body_uses_default(self, handler, make_handler_request):
        """When body is None, read_json_body returns None, and `or {}` gives empty dict."""
        req = make_handler_request(method="DELETE", body=None)
        req._json_body = None
        result = handler._handle_delete_workspace(req, "ws-1")
        assert _status(result) == 200

        call_kwargs = handler._mock_isolation_manager.delete_workspace.call_args.kwargs
        assert call_kwargs["force"] is False

    def test_delete_workspace_not_authenticated(
        self, handler, make_handler_request, mock_workspace_module
    ):
        mock_workspace_module.extract_user_from_request.return_value.is_authenticated = False

        req = make_handler_request(method="DELETE")
        result = handler._handle_delete_workspace(req, "ws-1")
        assert _status(result) == 401

    def test_delete_workspace_rbac_denied(
        self, handler, make_handler_request, mock_workspace_module
    ):
        from aragora.server.handlers.base import error_response

        handler._check_rbac_permission = lambda h, p, a: error_response("Forbidden", 403)

        req = make_handler_request(method="DELETE")
        result = handler._handle_delete_workspace(req, "ws-1")
        assert _status(result) == 403

    def test_delete_workspace_access_denied_exception(
        self, handler, make_handler_request, mock_workspace_module
    ):
        exc_class = mock_workspace_module.AccessDeniedException
        handler._mock_isolation_manager.delete_workspace = AsyncMock(
            side_effect=exc_class("denied")
        )

        req = make_handler_request(method="DELETE")
        result = handler._handle_delete_workspace(req, "ws-1")
        assert _status(result) == 403

    def test_delete_workspace_audit_log_called(self, handler, make_handler_request):
        req = make_handler_request(method="DELETE")
        handler._handle_delete_workspace(req, "ws-1")
        handler._mock_audit_log.log.assert_called_once()

    def test_delete_workspace_audit_action(
        self, handler, make_handler_request, mock_workspace_module
    ):
        req = make_handler_request(method="DELETE")
        handler._handle_delete_workspace(req, "ws-1")

        call_kwargs = handler._mock_audit_log.log.call_args.kwargs
        assert call_kwargs["action"] == mock_workspace_module.AuditAction.DELETE_WORKSPACE

    def test_delete_workspace_audit_outcome_success(
        self, handler, make_handler_request, mock_workspace_module
    ):
        req = make_handler_request(method="DELETE")
        handler._handle_delete_workspace(req, "ws-1")

        call_kwargs = handler._mock_audit_log.log.call_args.kwargs
        assert call_kwargs["outcome"] == mock_workspace_module.AuditOutcome.SUCCESS

    def test_delete_workspace_audit_actor_is_current_user(
        self, handler, make_handler_request, mock_workspace_module
    ):
        req = make_handler_request(method="DELETE")
        handler._handle_delete_workspace(req, "ws-1")

        mock_workspace_module.Actor.assert_called_with(id="test-user-001", type="user")

    def test_delete_workspace_audit_resource_is_workspace(
        self, handler, make_handler_request, mock_workspace_module
    ):
        req = make_handler_request(method="DELETE")
        handler._handle_delete_workspace(req, "ws-target")

        mock_workspace_module.Resource.assert_called_with(
            id="ws-target", type="workspace", workspace_id="ws-target"
        )

    def test_delete_workspace_emits_handler_event(self, handler, make_handler_request):
        """Verify that deleting a workspace emits a DELETED event."""
        with patch(
            "aragora.server.handlers.workspace.crud.emit_handler_event"
        ) as mock_emit:
            req = make_handler_request(method="DELETE")
            handler._handle_delete_workspace(req, "ws-del-001")

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[0][0] == "workspace"
            assert call_args[0][2] == {"workspace_id": "ws-del-001"}
            assert call_args[1]["user_id"] == "test-user-001"

    def test_delete_workspace_rbac_checks_delete_permission(
        self, handler, make_handler_request
    ):
        """Verify delete workspace uses PERM_WORKSPACE_DELETE."""
        captured_perms = []

        def capture_rbac(h, perm, auth_ctx):
            captured_perms.append(perm)
            return None

        handler._check_rbac_permission = capture_rbac

        req = make_handler_request(method="DELETE")
        handler._handle_delete_workspace(req, "ws-1")
        assert "workspace:delete" in captured_perms

    def test_delete_workspace_runtime_error(self, handler, make_handler_request):
        handler._mock_isolation_manager.delete_workspace = AsyncMock(
            side_effect=RuntimeError("db fail")
        )

        req = make_handler_request(method="DELETE")
        result = handler._handle_delete_workspace(req, "ws-1")
        assert _status(result) == 500

    def test_delete_workspace_value_error(self, handler, make_handler_request):
        handler._mock_isolation_manager.delete_workspace = AsyncMock(
            side_effect=ValueError("invalid workspace")
        )

        req = make_handler_request(method="DELETE")
        result = handler._handle_delete_workspace(req, "ws-1")
        assert _status(result) == 400

    def test_delete_workspace_os_error(self, handler, make_handler_request):
        handler._mock_isolation_manager.delete_workspace = AsyncMock(
            side_effect=OSError("filesystem error")
        )

        req = make_handler_request(method="DELETE")
        result = handler._handle_delete_workspace(req, "ws-1")
        assert _status(result) == 500


# ===========================================================================
# Module exports
# ===========================================================================


class TestModuleExports:
    """Test module-level exports."""

    def test_all_exports(self):
        from aragora.server.handlers.workspace import crud

        assert "WorkspaceCrudMixin" in crud.__all__

    def test_all_count(self):
        from aragora.server.handlers.workspace import crud

        assert len(crud.__all__) == 1


# ===========================================================================
# Security edge cases
# ===========================================================================


class TestSecurityEdgeCases:
    """Test security edge cases and input validation."""

    def test_create_workspace_sql_injection_in_name(self, handler, make_handler_request):
        """SQL injection in name should be passed through safely (mock)."""
        req = make_handler_request(
            method="POST",
            body={"name": "'; DROP TABLE workspaces; --"},
        )
        result = handler._handle_create_workspace(req)
        assert _status(result) == 201

    def test_create_workspace_xss_in_name(self, handler, make_handler_request):
        """XSS in name should not cause issues."""
        req = make_handler_request(
            method="POST",
            body={"name": "<script>alert(1)</script>"},
        )
        result = handler._handle_create_workspace(req)
        assert _status(result) in (201, 400, 500)

    def test_create_workspace_very_long_name(self, handler, make_handler_request):
        """Very long name should be handled safely."""
        req = make_handler_request(
            method="POST",
            body={"name": "x" * 10000},
        )
        result = handler._handle_create_workspace(req)
        # Should succeed (mock) or fail gracefully
        assert _status(result) in (201, 400, 500)

    def test_get_workspace_path_traversal(self, handler, make_handler_request):
        """Path traversal in workspace_id should be handled."""
        ws = MagicMock()
        ws.to_dict.return_value = {"id": "../../etc/passwd"}
        handler._mock_isolation_manager.get_workspace = AsyncMock(return_value=ws)

        req = make_handler_request(path="/api/v1/workspaces/../../etc/passwd")
        result = handler._handle_get_workspace(req, "../../etc/passwd")
        assert _status(result) in (200, 400, 403, 500)

    def test_delete_workspace_unicode_id(self, handler, make_handler_request):
        """Unicode workspace_id should be handled safely."""
        req = make_handler_request(method="DELETE")
        result = handler._handle_delete_workspace(req, "ws-\u00e9\u00e8\u00ea")
        assert _status(result) in (200, 400, 500)

    def test_create_workspace_name_is_none(self, handler, make_handler_request):
        """None as name should be rejected."""
        req = make_handler_request(method="POST", body={"name": None})
        result = handler._handle_create_workspace(req)
        assert _status(result) == 400
        assert "name" in _error(result).lower()

    def test_create_workspace_name_is_whitespace(self, handler, make_handler_request):
        """Whitespace-only name should still pass (mixin does not strip)."""
        req = make_handler_request(method="POST", body={"name": "   "})
        result = handler._handle_create_workspace(req)
        # "   " is truthy, so it passes the `if not name` check
        assert _status(result) == 201

    def test_list_workspaces_empty_org_id_in_query(
        self, handler, make_handler_request
    ):
        """Empty string org_id in query params should not trigger cross-tenant check."""
        req = make_handler_request()
        result = handler._handle_list_workspaces(req, {"organization_id": ""})
        # Empty string is falsy, so the check `if requested_org_id` is False
        assert _status(result) == 200

    def test_create_workspace_org_id_empty_in_body(
        self, handler, make_handler_request
    ):
        """Empty string organization_id in body should not trigger cross-tenant check."""
        req = make_handler_request(
            method="POST",
            body={"name": "WS", "organization_id": ""},
        )
        result = handler._handle_create_workspace(req)
        # Empty string is falsy, so `if requested_org_id and ...` is False
        assert _status(result) == 201


# ===========================================================================
# Cross-cutting behavior
# ===========================================================================


class TestCrossCuttingBehavior:
    """Test behavior that spans multiple endpoints."""

    def test_all_write_endpoints_check_auth(
        self, handler, make_handler_request, mock_workspace_module
    ):
        """All write endpoints should reject unauthenticated requests."""
        mock_workspace_module.extract_user_from_request.return_value.is_authenticated = False

        # Create workspace
        req1 = make_handler_request(method="POST", body={"name": "WS"})
        assert _status(handler._handle_create_workspace(req1)) == 401

        # Delete workspace
        req2 = make_handler_request(method="DELETE")
        assert _status(handler._handle_delete_workspace(req2, "ws-1")) == 401

    def test_all_read_endpoints_check_auth(
        self, handler, make_handler_request, mock_workspace_module
    ):
        """All read endpoints should reject unauthenticated requests."""
        mock_workspace_module.extract_user_from_request.return_value.is_authenticated = False

        # List workspaces
        req1 = make_handler_request()
        assert _status(handler._handle_list_workspaces(req1, {})) == 401

        # Get workspace
        req2 = make_handler_request(path="/api/v1/workspaces/ws-1")
        assert _status(handler._handle_get_workspace(req2, "ws-1")) == 401

    def test_all_endpoints_respect_rbac(
        self, handler, make_handler_request, mock_workspace_module
    ):
        """All endpoints should respect RBAC permission denial."""
        from aragora.server.handlers.base import error_response

        handler._check_rbac_permission = lambda h, p, a: error_response("Forbidden", 403)

        # Create
        req1 = make_handler_request(method="POST", body={"name": "WS"})
        assert _status(handler._handle_create_workspace(req1)) == 403

        # List
        req2 = make_handler_request()
        assert _status(handler._handle_list_workspaces(req2, {})) == 403

        # Get
        req3 = make_handler_request(path="/api/v1/workspaces/ws-1")
        assert _status(handler._handle_get_workspace(req3, "ws-1")) == 403

        # Delete
        req4 = make_handler_request(method="DELETE")
        assert _status(handler._handle_delete_workspace(req4, "ws-1")) == 403

    def test_create_and_delete_both_emit_events(self, handler, make_handler_request):
        """Both create and delete should emit handler events."""
        with patch(
            "aragora.server.handlers.workspace.crud.emit_handler_event"
        ) as mock_emit:
            req1 = make_handler_request(method="POST", body={"name": "WS"})
            handler._handle_create_workspace(req1)
            assert mock_emit.call_count == 1

            req2 = make_handler_request(method="DELETE")
            handler._handle_delete_workspace(req2, "ws-1")
            assert mock_emit.call_count == 2

    def test_create_and_delete_both_audit_log(self, handler, make_handler_request):
        """Both create and delete should write to the audit log."""
        req1 = make_handler_request(method="POST", body={"name": "WS"})
        handler._handle_create_workspace(req1)
        assert handler._mock_audit_log.log.call_count == 1

        req2 = make_handler_request(method="DELETE")
        handler._handle_delete_workspace(req2, "ws-1")
        assert handler._mock_audit_log.log.call_count == 2

    def test_list_and_get_do_not_audit_log(self, handler, make_handler_request):
        """List and get (read operations) should not write to audit log."""
        ws = MagicMock()
        ws.to_dict.return_value = {"id": "ws-1"}
        handler._mock_isolation_manager.get_workspace = AsyncMock(return_value=ws)

        req1 = make_handler_request()
        handler._handle_list_workspaces(req1, {})

        req2 = make_handler_request(path="/api/v1/workspaces/ws-1")
        handler._handle_get_workspace(req2, "ws-1")

        handler._mock_audit_log.log.assert_not_called()

    def test_cross_tenant_prevention_on_create(
        self, handler, make_handler_request
    ):
        """Verify cross-tenant prevention is applied on workspace creation."""
        req = make_handler_request(
            method="POST",
            body={"name": "WS", "organization_id": "malicious-org"},
        )
        result = handler._handle_create_workspace(req)
        assert _status(result) == 403

    def test_cross_tenant_prevention_on_list(
        self, handler, make_handler_request
    ):
        """Verify cross-tenant prevention is applied on workspace listing."""
        req = make_handler_request()
        result = handler._handle_list_workspaces(
            req, {"organization_id": "malicious-org"}
        )
        assert _status(result) == 403
