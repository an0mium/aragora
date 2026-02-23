"""Tests for typed handler base classes (aragora/server/handlers/typed_handlers.py).

Covers all classes and methods in typed_handlers.py:
- TypedHandler: Base handler with type annotations, read_json_body, error_response,
  handle/handle_post/handle_delete/handle_patch/handle_put, dependency injection,
  require_auth_or_error, get_current_user, require_admin_or_error,
  require_permission_or_error
- AuthenticatedHandler: _ensure_authenticated, _ensure_admin, current_user property
- PermissionHandler: _ensure_permission, _check_custom_permission, REQUIRED_PERMISSIONS
- AdminHandler: AUDIT_ACTIONS, _log_admin_action
- AsyncTypedHandler: async handle/handle_post/handle_delete/handle_patch/handle_put
- ResourceHandler: CRUD routing, _extract_resource_id, _get_resource_permissions,
  _list_resources, _get_resource, _create_resource, _update_resource, _patch_resource,
  _delete_resource
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.billing.auth.context import UserAuthContext
from aragora.server.handlers.typed_handlers import (
    TypedHandler,
    AuthenticatedHandler,
    PermissionHandler,
    AdminHandler,
    AsyncTypedHandler,
    ResourceHandler,
    MaybeAsyncHandlerResult,
)
from aragora.server.handlers.utils.responses import HandlerResult, error_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


def _make_mock_http_handler(
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    path: str = "/api/v1/test",
    command: str = "GET",
) -> MagicMock:
    """Create a mock HTTP request handler."""
    handler = MagicMock()
    handler.path = path
    handler.command = command

    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        handler.headers = {"Content-Length": str(len(body_bytes)), **(headers or {})}
        handler.rfile = MagicMock()
        handler.rfile.read.return_value = body_bytes
    else:
        handler.headers = {"Content-Length": "0", **(headers or {})}
        handler.rfile = MagicMock()
        handler.rfile.read.return_value = b""

    return handler


def _make_auth_user(
    user_id: str = "test-user-001",
    email: str = "test@example.com",
    role: str = "admin",
    authenticated: bool = True,
    roles: list[str] | None = None,
    permissions: list[str] | None = None,
) -> UserAuthContext:
    """Create a UserAuthContext with optional extra attributes.

    Note: ``is_admin`` is a read-only property on ``UserAuthContext`` derived
    from ``role``.  To get ``is_admin == True`` pass ``role="admin"`` (or
    ``role="owner"``).
    """
    ctx = UserAuthContext(
        authenticated=authenticated,
        user_id=user_id,
        email=email,
        role=role,
        token_type="access",
        client_ip="127.0.0.1",
    )
    if roles is not None:
        object.__setattr__(ctx, "roles", roles)
    if permissions is not None:
        object.__setattr__(ctx, "permissions", permissions)
    return ctx


# ---------------------------------------------------------------------------
# Module-level mock user context for TypedHandler auth bypass
# ---------------------------------------------------------------------------

_MOCK_USER_CTX = UserAuthContext(
    authenticated=True,
    user_id="test-user-001",
    email="test@example.com",
    org_id="test-org-001",
    role="admin",
    token_type="access",
    client_ip="127.0.0.1",
)
object.__setattr__(_MOCK_USER_CTX, "roles", ["admin", "owner"])
object.__setattr__(
    _MOCK_USER_CTX,
    "permissions",
    ["*", "admin", "knowledge.read", "knowledge.write", "knowledge.delete"],
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_typed_handler_auth(request, monkeypatch):
    """Patch TypedHandler auth methods so CRUD tests pass without real auth.

    TypedHandler overrides require_auth_or_error / get_current_user / require_admin_or_error
    from BaseHandler, so the conftest patches on BaseHandler do NOT take effect.
    We must patch them on TypedHandler directly.
    Tests marked with @pytest.mark.no_auto_auth opt out.
    """
    if "no_auto_auth" in [m.name for m in request.node.iter_markers()]:
        yield
        return

    monkeypatch.setattr(
        TypedHandler,
        "require_auth_or_error",
        lambda self, handler: (_MOCK_USER_CTX, None),
    )
    monkeypatch.setattr(
        TypedHandler,
        "get_current_user",
        lambda self, handler: _MOCK_USER_CTX,
    )
    monkeypatch.setattr(
        TypedHandler,
        "require_admin_or_error",
        lambda self, handler: (_MOCK_USER_CTX, None),
    )
    monkeypatch.setattr(
        TypedHandler,
        "require_permission_or_error",
        lambda self, handler, permission: (_MOCK_USER_CTX, None),
    )
    yield


@pytest.fixture
def mock_http_handler():
    """Create a basic mock HTTP handler."""
    return _make_mock_http_handler()


@pytest.fixture
def typed_handler():
    """Create a TypedHandler instance."""
    return TypedHandler(server_context={})


@pytest.fixture
def auth_handler():
    """Create an AuthenticatedHandler instance."""
    return AuthenticatedHandler(server_context={})


@pytest.fixture
def perm_handler():
    """Create a PermissionHandler instance."""
    return PermissionHandler(server_context={})


@pytest.fixture
def admin_handler():
    """Create an AdminHandler instance."""
    return AdminHandler(server_context={})


@pytest.fixture
def async_handler():
    """Create an AsyncTypedHandler instance."""
    return AsyncTypedHandler(server_context={})


@pytest.fixture
def resource_handler():
    """Create a ResourceHandler instance."""
    return ResourceHandler(server_context={})


# ============================================================================
# TypedHandler Tests
# ============================================================================


class TestTypedHandlerInit:
    """Test TypedHandler initialization."""

    def test_init_with_empty_context(self):
        handler = TypedHandler(server_context={})
        assert handler.ctx == {}

    def test_init_with_context_values(self):
        ctx = {"storage": MagicMock(), "user_store": MagicMock()}
        handler = TypedHandler(server_context=ctx)
        assert handler.ctx is ctx

    def test_default_factories_are_none(self):
        handler = TypedHandler(server_context={})
        assert handler._user_store_factory is None
        assert handler._storage_factory is None


class TestTypedHandlerReadJsonBody:
    """Test TypedHandler.read_json_body."""

    def test_read_valid_json(self, typed_handler):
        data = {"key": "value", "number": 42}
        http = _make_mock_http_handler(body=data)
        result = typed_handler.read_json_body(http)
        assert result == data

    def test_read_empty_body(self, typed_handler):
        http = _make_mock_http_handler()
        result = typed_handler.read_json_body(http)
        assert result is None

    def test_read_exceeds_max_size(self, typed_handler):
        data = {"key": "x" * 1000}
        http = _make_mock_http_handler(body=data)
        result = typed_handler.read_json_body(http, max_size=10)
        assert result is None

    def test_read_invalid_json(self, typed_handler):
        http = MagicMock()
        http.headers = {"Content-Length": "5"}
        http.rfile = MagicMock()
        http.rfile.read.return_value = b"notjn"
        result = typed_handler.read_json_body(http)
        assert result is None

    def test_read_value_error_in_content_length(self, typed_handler):
        http = MagicMock()
        http.headers = {"Content-Length": "invalid"}
        result = typed_handler.read_json_body(http)
        assert result is None

    def test_read_os_error(self, typed_handler):
        http = MagicMock()
        http.headers = {"Content-Length": "10"}
        http.rfile = MagicMock()
        http.rfile.read.side_effect = OSError("disk error")
        result = typed_handler.read_json_body(http)
        assert result is None

    def test_read_no_content_length(self, typed_handler):
        http = MagicMock()
        http.headers = {}
        result = typed_handler.read_json_body(http)
        assert result is None

    def test_read_nested_json(self, typed_handler):
        data = {"nested": {"deeply": {"value": [1, 2, 3]}}}
        http = _make_mock_http_handler(body=data)
        result = typed_handler.read_json_body(http)
        assert result == data

    def test_max_size_zero_does_not_reject(self, typed_handler):
        """max_size=0 is falsy, so no size check is done."""
        data = {"key": "value"}
        http = _make_mock_http_handler(body=data)
        result = typed_handler.read_json_body(http, max_size=0)
        assert result == data


class TestTypedHandlerErrorResponse:
    """Test TypedHandler.error_response."""

    def test_error_response_default_status(self, typed_handler):
        result = typed_handler.error_response("Something went wrong")
        assert _status(result) == 400
        body = _body(result)
        assert "error" in body or "message" in body

    def test_error_response_custom_status(self, typed_handler):
        result = typed_handler.error_response("Not Found", status=404)
        assert _status(result) == 404

    def test_error_response_500(self, typed_handler):
        result = typed_handler.error_response("Internal error", status=500)
        assert _status(result) == 500

    def test_error_response_returns_handler_result(self, typed_handler):
        result = typed_handler.error_response("err")
        assert isinstance(result, HandlerResult)


class TestTypedHandlerDefaultMethods:
    """Test TypedHandler default handle methods return None."""

    def test_handle_returns_none(self, typed_handler, mock_http_handler):
        result = typed_handler.handle("/api/v1/test", {}, mock_http_handler)
        assert result is None

    def test_handle_post_returns_none(self, typed_handler, mock_http_handler):
        result = typed_handler.handle_post("/api/v1/test", {}, mock_http_handler)
        assert result is None

    def test_handle_delete_returns_none(self, typed_handler, mock_http_handler):
        result = typed_handler.handle_delete("/api/v1/test", {}, mock_http_handler)
        assert result is None

    def test_handle_patch_returns_none(self, typed_handler, mock_http_handler):
        result = typed_handler.handle_patch("/api/v1/test", {}, mock_http_handler)
        assert result is None

    def test_handle_put_returns_none(self, typed_handler, mock_http_handler):
        result = typed_handler.handle_put("/api/v1/test", {}, mock_http_handler)
        assert result is None


class TestTypedHandlerDependencyAccess:
    """Test TypedHandler dependency injection and access."""

    def test_get_user_store_from_context(self):
        mock_store = MagicMock()
        handler = TypedHandler(server_context={"user_store": mock_store})
        assert handler.get_user_store() is mock_store

    def test_get_user_store_none_when_missing(self, typed_handler):
        assert typed_handler.get_user_store() is None

    def test_get_user_store_from_factory(self, typed_handler):
        mock_store = MagicMock()
        typed_handler._user_store_factory = lambda: mock_store
        assert typed_handler.get_user_store() is mock_store

    def test_get_storage_from_context(self):
        mock_storage = MagicMock()
        handler = TypedHandler(server_context={"storage": mock_storage})
        assert handler.get_storage() is mock_storage

    def test_get_storage_none_when_missing(self, typed_handler):
        assert typed_handler.get_storage() is None

    def test_get_storage_from_factory(self, typed_handler):
        mock_storage = MagicMock()
        typed_handler._storage_factory = lambda: mock_storage
        assert typed_handler.get_storage() is mock_storage

    def test_factory_takes_precedence_over_context(self):
        ctx_store = MagicMock(name="ctx_store")
        factory_store = MagicMock(name="factory_store")
        handler = TypedHandler(server_context={"user_store": ctx_store})
        handler._user_store_factory = lambda: factory_store
        assert handler.get_user_store() is factory_store


class TestTypedHandlerWithDependencies:
    """Test TypedHandler.with_dependencies factory method."""

    def test_with_no_dependencies(self):
        handler = TypedHandler.with_dependencies(server_context={})
        assert isinstance(handler, TypedHandler)
        assert handler.get_user_store() is None
        assert handler.get_storage() is None

    def test_with_user_store(self):
        mock_store = MagicMock()
        handler = TypedHandler.with_dependencies(server_context={}, user_store=mock_store)
        assert handler.get_user_store() is mock_store

    def test_with_storage(self):
        mock_storage = MagicMock()
        handler = TypedHandler.with_dependencies(server_context={}, storage=mock_storage)
        assert handler.get_storage() is mock_storage

    def test_with_both_dependencies(self):
        mock_store = MagicMock()
        mock_storage = MagicMock()
        handler = TypedHandler.with_dependencies(
            server_context={}, user_store=mock_store, storage=mock_storage
        )
        assert handler.get_user_store() is mock_store
        assert handler.get_storage() is mock_storage

    def test_with_dependencies_preserves_context(self):
        ctx = {"custom_key": "custom_value"}
        handler = TypedHandler.with_dependencies(server_context=ctx)
        assert handler.ctx.get("custom_key") == "custom_value"


class TestTypedHandlerRequireAuth:
    """Test TypedHandler authentication helper methods."""

    @pytest.mark.no_auto_auth
    def test_require_auth_or_error_unauthenticated(self, typed_handler):
        """When get_current_user returns None, expect 401."""
        http = _make_mock_http_handler()
        with patch.object(typed_handler, "get_current_user", return_value=None):
            user, err = typed_handler.require_auth_or_error(http)
            assert user is None
            assert err is not None
            assert _status(err) == 401

    def test_require_auth_or_error_authenticated(self, typed_handler, mock_http_handler):
        """With conftest auto-auth, authentication succeeds."""
        user, err = typed_handler.require_auth_or_error(mock_http_handler)
        assert err is None
        assert user is not None
        assert user.user_id is not None

    @pytest.mark.no_auto_auth
    def test_get_current_user_unauthenticated(self, typed_handler):
        http = _make_mock_http_handler()
        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=UserAuthContext(authenticated=False),
        ):
            result = typed_handler.get_current_user(http)
            assert result is None

    @pytest.mark.no_auto_auth
    def test_get_current_user_authenticated(self, typed_handler):
        auth_user = _make_auth_user()
        http = _make_mock_http_handler()
        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_user,
        ):
            result = typed_handler.get_current_user(http)
            assert result is not None
            assert result.user_id == "test-user-001"

    @pytest.mark.no_auto_auth
    def test_get_current_user_uses_handler_user_store(self, typed_handler):
        """get_current_user should check handler.user_store."""
        mock_store = MagicMock()
        http = _make_mock_http_handler()
        http.user_store = mock_store
        auth_user = _make_auth_user()
        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_user,
        ) as mock_extract:
            typed_handler.get_current_user(http)
            mock_extract.assert_called_once_with(http, mock_store)

    @pytest.mark.no_auto_auth
    def test_get_current_user_uses_class_user_store(self):
        """get_current_user should check cls.user_store."""
        mock_store = MagicMock()

        class CustomHandler(TypedHandler):
            user_store = mock_store

        handler = CustomHandler(server_context={})
        http = _make_mock_http_handler()
        # Remove handler-level user_store so it falls through to class level
        del http.user_store
        auth_user = _make_auth_user()
        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_user,
        ) as mock_extract:
            handler.get_current_user(http)
            mock_extract.assert_called_once_with(http, mock_store)

    @pytest.mark.no_auto_auth
    def test_get_current_user_uses_factory(self):
        mock_store = MagicMock()
        handler = TypedHandler(server_context={})
        handler._user_store_factory = lambda: mock_store
        http = _make_mock_http_handler()
        # Ensure handler has no user_store attr and class has no user_store attr
        if hasattr(http, "user_store"):
            del http.user_store
        auth_user = _make_auth_user()
        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=auth_user,
        ) as mock_extract:
            handler.get_current_user(http)
            mock_extract.assert_called_once_with(http, mock_store)


class TestTypedHandlerRequireAdmin:
    """Test TypedHandler.require_admin_or_error."""

    def test_require_admin_authenticated_admin(self, typed_handler, mock_http_handler):
        """With conftest auto-auth (admin), expect success."""
        user, err = typed_handler.require_admin_or_error(mock_http_handler)
        assert err is None
        assert user is not None

    @pytest.mark.no_auto_auth
    def test_require_admin_unauthenticated(self, typed_handler):
        http = _make_mock_http_handler()
        with patch.object(
            typed_handler,
            "require_auth_or_error",
            return_value=(None, error_response("Authentication required", 401)),
        ):
            user, err = typed_handler.require_admin_or_error(http)
            assert user is None
            assert _status(err) == 401

    @pytest.mark.no_auto_auth
    def test_require_admin_non_admin_user(self, typed_handler):
        non_admin = _make_auth_user(role="member", roles=["member"], permissions=[])
        http = _make_mock_http_handler()
        with patch.object(typed_handler, "require_auth_or_error", return_value=(non_admin, None)):
            user, err = typed_handler.require_admin_or_error(http)
            assert user is None
            assert _status(err) == 403

    @pytest.mark.no_auto_auth
    def test_require_admin_with_admin_role(self, typed_handler):
        admin = _make_auth_user(roles=["admin"], permissions=[])
        http = _make_mock_http_handler()
        with patch.object(typed_handler, "require_auth_or_error", return_value=(admin, None)):
            user, err = typed_handler.require_admin_or_error(http)
            assert err is None
            assert user is not None

    @pytest.mark.no_auto_auth
    def test_require_admin_with_admin_permission(self, typed_handler):
        admin = _make_auth_user(roles=[], permissions=["admin"])
        http = _make_mock_http_handler()
        with patch.object(typed_handler, "require_auth_or_error", return_value=(admin, None)):
            user, err = typed_handler.require_admin_or_error(http)
            assert err is None

    @pytest.mark.no_auto_auth
    def test_require_admin_with_is_admin_attr(self, typed_handler):
        """Test that getattr(user, 'is_admin', False) path works."""
        # Create a mock user where .is_admin returns True but roles/permissions are empty
        admin = MagicMock()
        admin.roles = []
        admin.permissions = []
        admin.is_admin = True
        http = _make_mock_http_handler()
        with patch.object(typed_handler, "require_auth_or_error", return_value=(admin, None)):
            user, err = typed_handler.require_admin_or_error(http)
            assert err is None

    @pytest.mark.no_auto_auth
    def test_require_admin_none_roles(self, typed_handler):
        """Test when roles attribute is None."""
        user_ctx = _make_auth_user(role="member", roles=None, permissions=None)
        # Ensure roles returns None-ish value: set as None directly
        object.__setattr__(user_ctx, "roles", None)
        object.__setattr__(user_ctx, "permissions", None)
        http = _make_mock_http_handler()
        with patch.object(typed_handler, "require_auth_or_error", return_value=(user_ctx, None)):
            user, err = typed_handler.require_admin_or_error(http)
            assert user is None
            assert _status(err) == 403


class TestTypedHandlerRequirePermission:
    """Test TypedHandler.require_permission_or_error."""

    @pytest.mark.no_auto_auth
    def test_unauthenticated(self, typed_handler):
        http = _make_mock_http_handler()
        with patch.object(
            typed_handler,
            "require_auth_or_error",
            return_value=(None, error_response("Auth required", 401)),
        ):
            user, err = typed_handler.require_permission_or_error(http, "knowledge.read")
            assert user is None
            assert _status(err) == 401

    @pytest.mark.no_auto_auth
    def test_admin_role_grants_all(self, typed_handler):
        admin = _make_auth_user(roles=["admin"], permissions=[])
        http = _make_mock_http_handler()
        with patch.object(typed_handler, "require_auth_or_error", return_value=(admin, None)):
            user, err = typed_handler.require_permission_or_error(http, "anything.write")
            assert err is None
            assert user is admin

    @pytest.mark.no_auto_auth
    def test_owner_role_grants_all(self, typed_handler):
        owner = _make_auth_user(roles=["owner"], permissions=[])
        http = _make_mock_http_handler()
        with patch.object(typed_handler, "require_auth_or_error", return_value=(owner, None)):
            user, err = typed_handler.require_permission_or_error(http, "anything.delete")
            assert err is None

    @pytest.mark.no_auto_auth
    def test_admin_in_permissions(self, typed_handler):
        ctx = _make_auth_user(roles=[], permissions=["admin"])
        http = _make_mock_http_handler()
        with patch.object(typed_handler, "require_auth_or_error", return_value=(ctx, None)):
            user, err = typed_handler.require_permission_or_error(http, "knowledge.read")
            assert err is None

    @pytest.mark.no_auto_auth
    def test_owner_role_attribute(self, typed_handler):
        ctx = _make_auth_user(role="owner", roles=[], permissions=[])
        http = _make_mock_http_handler()
        with patch.object(typed_handler, "require_auth_or_error", return_value=(ctx, None)):
            user, err = typed_handler.require_permission_or_error(http, "knowledge.read")
            assert err is None

    @pytest.mark.no_auto_auth
    def test_admin_role_attribute(self, typed_handler):
        ctx = _make_auth_user(role="admin", roles=[], permissions=[])
        http = _make_mock_http_handler()
        with patch.object(typed_handler, "require_auth_or_error", return_value=(ctx, None)):
            user, err = typed_handler.require_permission_or_error(http, "knowledge.read")
            assert err is None

    @pytest.mark.no_auto_auth
    def test_specific_permission_granted(self, typed_handler):
        ctx = _make_auth_user(role="member", roles=[], permissions=["knowledge.read"])
        http = _make_mock_http_handler()
        with patch.object(typed_handler, "require_auth_or_error", return_value=(ctx, None)):
            user, err = typed_handler.require_permission_or_error(http, "knowledge.read")
            assert err is None

    @pytest.mark.no_auto_auth
    def test_permission_denied(self, typed_handler):
        ctx = _make_auth_user(role="member", roles=[], permissions=[])
        http = _make_mock_http_handler()
        with patch.object(typed_handler, "require_auth_or_error", return_value=(ctx, None)):
            with patch(
                "aragora.server.handlers.utils.decorators.has_permission",
                return_value=False,
            ):
                user, err = typed_handler.require_permission_or_error(http, "knowledge.write")
                assert user is None
                assert _status(err) == 403

    @pytest.mark.no_auto_auth
    def test_permission_via_has_permission_function(self, typed_handler):
        ctx = _make_auth_user(role="editor", roles=[], permissions=[])
        http = _make_mock_http_handler()
        with patch.object(typed_handler, "require_auth_or_error", return_value=(ctx, None)):
            with patch(
                "aragora.server.handlers.utils.decorators.has_permission",
                return_value=True,
            ):
                user, err = typed_handler.require_permission_or_error(http, "knowledge.write")
                assert err is None

    @pytest.mark.no_auto_auth
    def test_permission_denied_completely(self, typed_handler):
        ctx = _make_auth_user(role="viewer", roles=[], permissions=[])
        http = _make_mock_http_handler()
        with patch.object(typed_handler, "require_auth_or_error", return_value=(ctx, None)):
            with patch(
                "aragora.server.handlers.utils.decorators.has_permission",
                return_value=False,
            ):
                user, err = typed_handler.require_permission_or_error(http, "knowledge.delete")
                assert user is None
                assert _status(err) == 403


# ============================================================================
# AuthenticatedHandler Tests
# ============================================================================


class TestAuthenticatedHandler:
    """Test AuthenticatedHandler methods."""

    def test_current_user_initially_none(self, auth_handler):
        assert auth_handler.current_user is None

    def test_ensure_authenticated_success(self, auth_handler, mock_http_handler):
        """With conftest auto-auth, authentication succeeds."""
        user, err = auth_handler._ensure_authenticated(mock_http_handler)
        assert err is None
        assert user is not None
        assert auth_handler.current_user is not None

    @pytest.mark.no_auto_auth
    def test_ensure_authenticated_failure(self, auth_handler):
        http = _make_mock_http_handler()
        with patch.object(
            auth_handler,
            "require_auth_or_error",
            return_value=(None, error_response("Authentication required", 401)),
        ):
            user, err = auth_handler._ensure_authenticated(http)
            assert user is None
            assert _status(err) == 401
            assert auth_handler.current_user is None

    def test_ensure_admin_success(self, auth_handler, mock_http_handler):
        """With conftest auto-auth (admin), expect success."""
        user, err = auth_handler._ensure_admin(mock_http_handler)
        assert err is None
        assert user is not None
        assert auth_handler.current_user is not None

    @pytest.mark.no_auto_auth
    def test_ensure_admin_failure(self, auth_handler):
        http = _make_mock_http_handler()
        with patch.object(
            auth_handler,
            "require_admin_or_error",
            return_value=(None, error_response("Admin access required", 403)),
        ):
            user, err = auth_handler._ensure_admin(http)
            assert user is None
            assert _status(err) == 403
            assert auth_handler.current_user is None

    @pytest.mark.no_auto_auth
    def test_ensure_admin_unauthenticated(self, auth_handler):
        http = _make_mock_http_handler()
        with patch.object(
            auth_handler,
            "require_admin_or_error",
            return_value=(None, error_response("Authentication required", 401)),
        ):
            user, err = auth_handler._ensure_admin(http)
            assert user is None
            assert _status(err) == 401

    def test_current_user_cached_after_auth(self, auth_handler, mock_http_handler):
        """After _ensure_authenticated, current_user should be set."""
        auth_handler._ensure_authenticated(mock_http_handler)
        assert auth_handler.current_user is not None
        user = auth_handler.current_user
        # Second call should also work
        auth_handler._ensure_authenticated(mock_http_handler)
        assert auth_handler.current_user is not None


# ============================================================================
# PermissionHandler Tests
# ============================================================================


class TestPermissionHandler:
    """Test PermissionHandler methods."""

    def test_default_permissions(self, perm_handler):
        """Default REQUIRED_PERMISSIONS should have None for all methods."""
        assert perm_handler.REQUIRED_PERMISSIONS["GET"] is None
        assert perm_handler.REQUIRED_PERMISSIONS["POST"] is None
        assert perm_handler.REQUIRED_PERMISSIONS["PUT"] is None
        assert perm_handler.REQUIRED_PERMISSIONS["PATCH"] is None
        assert perm_handler.REQUIRED_PERMISSIONS["DELETE"] is None

    def test_ensure_permission_no_specific_permission(self, perm_handler, mock_http_handler):
        """When permission is None (default), just auth is required."""
        user, err = perm_handler._ensure_permission(mock_http_handler, "GET")
        assert err is None
        assert user is not None

    def test_ensure_permission_with_method(self, mock_http_handler):
        """When a permission is defined for a method, it should be checked."""

        class CustomPermHandler(PermissionHandler):
            REQUIRED_PERMISSIONS = {
                "GET": "knowledge:read",
                "POST": "knowledge:write",
            }

        handler = CustomPermHandler(server_context={})
        # With conftest auto-auth (admin), permission check should succeed
        user, err = handler._ensure_permission(mock_http_handler, "GET")
        assert err is None

    def test_ensure_permission_extracts_method_from_handler(self, mock_http_handler):
        """When method is None, should extract from handler.command."""
        mock_http_handler.command = "POST"
        handler = PermissionHandler(server_context={})
        user, err = handler._ensure_permission(mock_http_handler, method=None)
        assert err is None

    def test_ensure_permission_unknown_method(self, perm_handler, mock_http_handler):
        """Unknown method defaults to None permission (just auth)."""
        user, err = perm_handler._ensure_permission(mock_http_handler, "OPTIONS")
        assert err is None

    @pytest.mark.no_auto_auth
    def test_ensure_permission_unauthenticated(self, perm_handler):
        http = _make_mock_http_handler()
        with patch.object(
            perm_handler,
            "_ensure_authenticated",
            return_value=(None, error_response("Authentication required", 401)),
        ):
            user, err = perm_handler._ensure_permission(http, "GET")
            assert user is None
            assert _status(err) == 401

    def test_check_custom_permission_success(self, perm_handler, mock_http_handler):
        """With conftest auto-auth, custom permission check should succeed."""
        user, err = perm_handler._check_custom_permission(mock_http_handler, "debates:fork")
        assert err is None
        assert user is not None

    @pytest.mark.no_auto_auth
    def test_check_custom_permission_unauthenticated(self, perm_handler):
        http = _make_mock_http_handler()
        with patch.object(
            perm_handler,
            "_ensure_authenticated",
            return_value=(None, error_response("Auth required", 401)),
        ):
            user, err = perm_handler._check_custom_permission(http, "debates:fork")
            assert user is None
            assert _status(err) == 401

    @pytest.mark.no_auto_auth
    def test_check_custom_permission_denied(self, perm_handler):
        user_ctx = _make_auth_user(role="viewer", roles=[], permissions=[])
        http = _make_mock_http_handler()
        with patch.object(
            perm_handler,
            "_ensure_authenticated",
            return_value=(user_ctx, None),
        ):
            with patch.object(
                perm_handler,
                "require_permission_or_error",
                return_value=(None, error_response("Permission denied", 403)),
            ):
                user, err = perm_handler._check_custom_permission(http, "admin:manage")
                assert user is None
                assert _status(err) == 403


# ============================================================================
# AdminHandler Tests
# ============================================================================


class TestAdminHandler:
    """Test AdminHandler methods."""

    def test_audit_actions_default_true(self, admin_handler):
        assert admin_handler.AUDIT_ACTIONS is True

    def test_log_admin_action_with_user(self, admin_handler, mock_http_handler):
        """Log admin action when current_user is set."""
        admin_handler._ensure_authenticated(mock_http_handler)
        # Should not raise
        admin_handler._log_admin_action(
            action="update_config",
            resource_id="config-001",
            details={"key": "value"},
        )

    def test_log_admin_action_no_user(self, admin_handler):
        """Log admin action when no current_user is set."""
        admin_handler._log_admin_action(
            action="update_config",
            resource_id="config-001",
        )

    def test_log_admin_action_audit_disabled(self):
        """When AUDIT_ACTIONS is False, logging is skipped."""

        class NoAuditHandler(AdminHandler):
            AUDIT_ACTIONS = False

        handler = NoAuditHandler(server_context={})
        # Should return immediately without logging
        handler._log_admin_action(action="delete_user", resource_id="user-001")

    def test_log_admin_action_with_audit_module(self, admin_handler, mock_http_handler):
        admin_handler._ensure_authenticated(mock_http_handler)
        with patch(
            "aragora.server.handlers.typed_handlers.audit_admin",
            create=True,
        ) as mock_audit:
            # The import is inside a try/except, so we need to patch at the import site
            admin_handler._log_admin_action(
                action="delete_user",
                resource_id="user-001",
                details={"reason": "test"},
            )
            # Even if audit_admin fails to import, should not raise

    def test_log_admin_action_audit_import_error(self, admin_handler, mock_http_handler):
        """When audit module is not available, should not raise."""
        admin_handler._ensure_authenticated(mock_http_handler)
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args, **kwargs: (
                __import__(name, *args, **kwargs)
                if name != "aragora.audit.unified"
                else (_ for _ in ()).throw(ImportError("not available"))
            ),
        ):
            # Should not raise
            admin_handler._log_admin_action(action="test", resource_id="r1")

    def test_log_admin_action_none_details(self, admin_handler, mock_http_handler):
        """Details=None should be handled."""
        admin_handler._ensure_authenticated(mock_http_handler)
        admin_handler._log_admin_action(action="test")

    def test_log_admin_action_none_resource_id(self, admin_handler, mock_http_handler):
        """resource_id=None should be handled."""
        admin_handler._ensure_authenticated(mock_http_handler)
        admin_handler._log_admin_action(action="test", resource_id=None)


# ============================================================================
# AsyncTypedHandler Tests
# ============================================================================


class TestAsyncTypedHandler:
    """Test AsyncTypedHandler methods."""

    @pytest.mark.asyncio
    async def test_handle_returns_none(self, async_handler, mock_http_handler):
        result = await async_handler.handle("/api/v1/test", {}, mock_http_handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_post_returns_none(self, async_handler, mock_http_handler):
        result = await async_handler.handle_post("/api/v1/test", {}, mock_http_handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_delete_returns_none(self, async_handler, mock_http_handler):
        result = await async_handler.handle_delete("/api/v1/test", {}, mock_http_handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_patch_returns_none(self, async_handler, mock_http_handler):
        result = await async_handler.handle_patch("/api/v1/test", {}, mock_http_handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_put_returns_none(self, async_handler, mock_http_handler):
        result = await async_handler.handle_put("/api/v1/test", {}, mock_http_handler)
        assert result is None

    def test_async_handler_is_typed_handler(self, async_handler):
        assert isinstance(async_handler, TypedHandler)


# ============================================================================
# ResourceHandler Tests
# ============================================================================


class TestResourceHandlerInit:
    """Test ResourceHandler initialization and configuration."""

    def test_default_resource_name(self, resource_handler):
        assert resource_handler.RESOURCE_NAME == "resource"

    def test_default_resource_id_param(self, resource_handler):
        assert resource_handler.RESOURCE_ID_PARAM == "id"

    def test_default_routes(self, resource_handler):
        assert resource_handler.ROUTES == []

    def test_permissions_generated_from_resource_name(self):
        class DocHandler(ResourceHandler):
            RESOURCE_NAME = "document"

        handler = DocHandler(server_context={})
        perms = handler.REQUIRED_PERMISSIONS
        assert perms["GET"] == "document:read"
        assert perms["POST"] == "document:create"
        assert perms["PUT"] == "document:update"
        assert perms["PATCH"] == "document:update"
        assert perms["DELETE"] == "document:delete"

    def test_get_resource_permissions_classmethod(self):
        class ItemHandler(ResourceHandler):
            RESOURCE_NAME = "item"

        perms = ItemHandler._get_resource_permissions()
        assert perms["GET"] == "item:read"
        assert perms["POST"] == "item:create"


class TestResourceHandlerExtractId:
    """Test ResourceHandler._extract_resource_id."""

    def test_extract_id_from_path(self, resource_handler):
        result = resource_handler._extract_resource_id("/api/v1/resources/abc-123")
        assert result == "abc-123"

    def test_extract_id_trailing_slash(self, resource_handler):
        result = resource_handler._extract_resource_id("/api/v1/resources/abc-123/")
        assert result == "abc-123"

    def test_no_id_collection_endpoint(self, resource_handler):
        """When path ends with the plural resource name, return None."""
        # resource_handler.RESOURCE_NAME is "resource", so "resources" is the plural
        result = resource_handler._extract_resource_id("/api/v1/resources")
        # The method checks: last == f"{self.RESOURCE_NAME}s" -> "resources" == "resources" -> True -> None
        assert result is None

    def test_no_id_resource_name_singular(self, resource_handler):
        result = resource_handler._extract_resource_id("/api/v1/resource")
        assert result is None

    def test_no_id_list(self, resource_handler):
        result = resource_handler._extract_resource_id("/api/v1/items/list")
        assert result is None

    def test_no_id_search(self, resource_handler):
        result = resource_handler._extract_resource_id("/api/v1/items/search")
        assert result is None

    def test_no_id_empty_last(self, resource_handler):
        result = resource_handler._extract_resource_id("/")
        assert result is None

    def test_extract_uuid_style_id(self, resource_handler):
        result = resource_handler._extract_resource_id(
            "/api/v1/things/550e8400-e29b-41d4-a716-446655440000"
        )
        assert result == "550e8400-e29b-41d4-a716-446655440000"

    def test_extract_numeric_id(self, resource_handler):
        result = resource_handler._extract_resource_id("/api/v1/things/42")
        assert result == "42"

    def test_short_path(self, resource_handler):
        result = resource_handler._extract_resource_id("/a")
        assert result == "a"

    def test_custom_resource_name(self):
        class DocHandler(ResourceHandler):
            RESOURCE_NAME = "document"

        handler = DocHandler(server_context={})
        # "documents" is the plural -> returns None
        assert handler._extract_resource_id("/api/v1/documents") is None
        # An actual ID is returned
        assert handler._extract_resource_id("/api/v1/documents/doc-1") == "doc-1"


class TestResourceHandlerCRUD:
    """Test ResourceHandler CRUD routing and default implementations."""

    def test_handle_get_list(self, resource_handler, mock_http_handler):
        """GET on collection endpoint returns 501 (not implemented)."""
        mock_http_handler.path = "/api/v1/resources"
        result = resource_handler.handle("/api/v1/resources", {}, mock_http_handler)
        assert _status(result) == 501
        assert (
            "list" in _body(result).get("error", "").lower()
            or "not implemented" in _body(result).get("error", "").lower()
        )

    def test_handle_get_single(self, resource_handler, mock_http_handler):
        """GET on resource with ID returns 501 (not implemented)."""
        mock_http_handler.path = "/api/v1/resources/r-001"
        result = resource_handler.handle("/api/v1/resources/r-001", {}, mock_http_handler)
        assert _status(result) == 501
        assert "not implemented" in _body(result).get("error", "").lower()

    def test_handle_post_create(self, resource_handler, mock_http_handler):
        """POST returns 501 (not implemented)."""
        result = resource_handler.handle_post("/api/v1/resources", {}, mock_http_handler)
        assert _status(result) == 501

    def test_handle_put_requires_id(self, resource_handler, mock_http_handler):
        """PUT on collection (no ID) returns 400."""
        mock_http_handler.path = "/api/v1/resources"
        result = resource_handler.handle_put("/api/v1/resources", {}, mock_http_handler)
        assert _status(result) == 400
        assert (
            "id required" in _body(result).get("error", "").lower()
            or "required" in _body(result).get("error", "").lower()
        )

    def test_handle_put_with_id(self, resource_handler, mock_http_handler):
        """PUT with resource ID returns 501 (not implemented)."""
        mock_http_handler.path = "/api/v1/resources/r-001"
        result = resource_handler.handle_put("/api/v1/resources/r-001", {}, mock_http_handler)
        assert _status(result) == 501

    def test_handle_patch_requires_id(self, resource_handler, mock_http_handler):
        """PATCH on collection (no ID) returns 400."""
        mock_http_handler.path = "/api/v1/resources"
        result = resource_handler.handle_patch("/api/v1/resources", {}, mock_http_handler)
        assert _status(result) == 400

    def test_handle_patch_with_id(self, resource_handler, mock_http_handler):
        """PATCH with resource ID returns 501 (default delegates to _update_resource)."""
        mock_http_handler.path = "/api/v1/resources/r-001"
        result = resource_handler.handle_patch("/api/v1/resources/r-001", {}, mock_http_handler)
        assert _status(result) == 501

    def test_handle_delete_requires_id(self, resource_handler, mock_http_handler):
        """DELETE on collection (no ID) returns 400."""
        mock_http_handler.path = "/api/v1/resources"
        result = resource_handler.handle_delete("/api/v1/resources", {}, mock_http_handler)
        assert _status(result) == 400

    def test_handle_delete_with_id(self, resource_handler, mock_http_handler):
        """DELETE with resource ID returns 501 (not implemented)."""
        mock_http_handler.path = "/api/v1/resources/r-001"
        result = resource_handler.handle_delete("/api/v1/resources/r-001", {}, mock_http_handler)
        assert _status(result) == 501

    def test_patch_defaults_to_update(self, resource_handler, mock_http_handler):
        """_patch_resource by default calls _update_resource."""
        mock_http_handler.path = "/api/v1/things/t-001"
        result = resource_handler._patch_resource("t-001", mock_http_handler)
        assert _status(result) == 501
        # Should contain "update" since it delegates to _update_resource
        body = _body(result)
        assert "update" in body.get("error", "").lower()


class TestResourceHandlerCustomImpl:
    """Test ResourceHandler with custom implementations."""

    def test_custom_list_resources(self, mock_http_handler):
        class ItemHandler(ResourceHandler):
            RESOURCE_NAME = "item"
            ROUTES = ["/api/v1/items"]

            def _list_resources(self, query_params, handler):
                return self.json_response({"items": [{"id": "1"}], "total": 1})

        handler = ItemHandler(server_context={})
        mock_http_handler.path = "/api/v1/items"
        result = handler.handle("/api/v1/items", {}, mock_http_handler)
        assert _status(result) == 200
        assert _body(result)["total"] == 1

    def test_custom_get_resource(self, mock_http_handler):
        class ItemHandler(ResourceHandler):
            RESOURCE_NAME = "item"

            def _get_resource(self, resource_id, handler):
                return self.json_response({"id": resource_id, "name": "Test"})

        handler = ItemHandler(server_context={})
        mock_http_handler.path = "/api/v1/items/item-1"
        result = handler.handle("/api/v1/items/item-1", {}, mock_http_handler)
        assert _status(result) == 200
        assert _body(result)["id"] == "item-1"

    def test_custom_create_resource(self, mock_http_handler):
        class ItemHandler(ResourceHandler):
            RESOURCE_NAME = "item"

            def _create_resource(self, handler):
                return self.json_response({"id": "new-item"}, status=201)

        handler = ItemHandler(server_context={})
        result = handler.handle_post("/api/v1/items", {}, mock_http_handler)
        assert _status(result) == 201
        assert _body(result)["id"] == "new-item"

    def test_custom_update_resource(self, mock_http_handler):
        class ItemHandler(ResourceHandler):
            RESOURCE_NAME = "item"

            def _update_resource(self, resource_id, handler):
                return self.json_response({"id": resource_id, "updated": True})

        handler = ItemHandler(server_context={})
        mock_http_handler.path = "/api/v1/items/item-1"
        result = handler.handle_put("/api/v1/items/item-1", {}, mock_http_handler)
        assert _status(result) == 200
        assert _body(result)["updated"] is True

    def test_custom_patch_resource(self, mock_http_handler):
        class ItemHandler(ResourceHandler):
            RESOURCE_NAME = "item"

            def _patch_resource(self, resource_id, handler):
                return self.json_response({"id": resource_id, "patched": True})

        handler = ItemHandler(server_context={})
        mock_http_handler.path = "/api/v1/items/item-1"
        result = handler.handle_patch("/api/v1/items/item-1", {}, mock_http_handler)
        assert _status(result) == 200
        assert _body(result)["patched"] is True

    def test_custom_delete_resource(self, mock_http_handler):
        class ItemHandler(ResourceHandler):
            RESOURCE_NAME = "item"

            def _delete_resource(self, resource_id, handler):
                return self.json_response({"deleted": True}, status=204)

        handler = ItemHandler(server_context={})
        mock_http_handler.path = "/api/v1/items/item-1"
        result = handler.handle_delete("/api/v1/items/item-1", {}, mock_http_handler)
        assert _status(result) == 204


# ============================================================================
# ResourceHandler Auth Integration Tests
# ============================================================================


class TestResourceHandlerAuth:
    """Test ResourceHandler RBAC integration."""

    @pytest.mark.no_auto_auth
    def test_handle_get_requires_auth(self):
        handler = ResourceHandler(server_context={})
        http = _make_mock_http_handler(path="/api/v1/resources/r-001")
        with patch.object(
            handler,
            "_ensure_permission",
            return_value=(None, error_response("Authentication required", 401)),
        ):
            result = handler.handle("/api/v1/resources/r-001", {}, http)
            assert _status(result) == 401

    @pytest.mark.no_auto_auth
    def test_handle_post_requires_auth(self):
        handler = ResourceHandler(server_context={})
        http = _make_mock_http_handler()
        with patch.object(
            handler,
            "_ensure_permission",
            return_value=(None, error_response("Authentication required", 401)),
        ):
            result = handler.handle_post("/api/v1/resources", {}, http)
            assert _status(result) == 401

    @pytest.mark.no_auto_auth
    def test_handle_put_requires_auth(self):
        handler = ResourceHandler(server_context={})
        http = _make_mock_http_handler(path="/api/v1/resources/r-001")
        with patch.object(
            handler,
            "_ensure_permission",
            return_value=(None, error_response("Authentication required", 401)),
        ):
            result = handler.handle_put("/api/v1/resources/r-001", {}, http)
            assert _status(result) == 401

    @pytest.mark.no_auto_auth
    def test_handle_delete_requires_auth(self):
        handler = ResourceHandler(server_context={})
        http = _make_mock_http_handler(path="/api/v1/resources/r-001")
        with patch.object(
            handler,
            "_ensure_permission",
            return_value=(None, error_response("Authentication required", 401)),
        ):
            result = handler.handle_delete("/api/v1/resources/r-001", {}, http)
            assert _status(result) == 401


# ============================================================================
# Type Alias Test
# ============================================================================


class TestMaybeAsyncHandlerResult:
    """Test MaybeAsyncHandlerResult type alias."""

    def test_type_alias_exists(self):
        # MaybeAsyncHandlerResult should be importable
        assert MaybeAsyncHandlerResult is not None


# ============================================================================
# Inheritance Tests
# ============================================================================


class TestInheritanceHierarchy:
    """Test the class hierarchy is correct."""

    def test_typed_handler_is_base_handler(self):
        from aragora.server.handlers.base import BaseHandler

        assert issubclass(TypedHandler, BaseHandler)

    def test_authenticated_handler_is_typed_handler(self):
        assert issubclass(AuthenticatedHandler, TypedHandler)

    def test_permission_handler_is_authenticated_handler(self):
        assert issubclass(PermissionHandler, AuthenticatedHandler)

    def test_admin_handler_is_authenticated_handler(self):
        assert issubclass(AdminHandler, AuthenticatedHandler)

    def test_async_handler_is_typed_handler(self):
        assert issubclass(AsyncTypedHandler, TypedHandler)

    def test_resource_handler_is_permission_handler(self):
        assert issubclass(ResourceHandler, PermissionHandler)


# ============================================================================
# Integration / Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and integration scenarios."""

    def test_json_response_method(self, typed_handler):
        result = typed_handler.json_response({"hello": "world"})
        assert _status(result) == 200
        assert _body(result)["hello"] == "world"

    def test_json_response_custom_status(self, typed_handler):
        result = typed_handler.json_response({"created": True}, status=201)
        assert _status(result) == 201

    def test_error_response_wraps_correctly(self, typed_handler):
        result = typed_handler.error_response("bad request", 400)
        body = _body(result)
        assert (
            "bad request" in body.get("error", "").lower()
            or "bad request" in json.dumps(body).lower()
        )

    def test_handler_with_full_context(self):
        """Test handler with a more realistic context."""
        mock_storage = MagicMock()
        mock_user_store = MagicMock()
        ctx = {
            "storage": mock_storage,
            "user_store": mock_user_store,
        }
        handler = TypedHandler(server_context=ctx)
        assert handler.get_storage() is mock_storage
        assert handler.get_user_store() is mock_user_store

    def test_subclass_can_override_handle(self, mock_http_handler):
        """Test that subclasses can override handle method."""

        class MyHandler(TypedHandler):
            def handle(self, path, query_params, handler):
                return self.json_response({"custom": True})

        h = MyHandler(server_context={})
        result = h.handle("/test", {}, mock_http_handler)
        assert _status(result) == 200
        assert _body(result)["custom"] is True

    def test_multiple_handler_instances_independent(self):
        """Test that multiple handler instances don't share state."""
        h1 = AuthenticatedHandler(server_context={"key": "val1"})
        h2 = AuthenticatedHandler(server_context={"key": "val2"})
        assert h1.ctx["key"] == "val1"
        assert h2.ctx["key"] == "val2"
        assert h1._current_user is None
        assert h2._current_user is None

    def test_resource_handler_custom_id_extraction(self, mock_http_handler):
        """Test overriding _extract_resource_id."""

        class CustomHandler(ResourceHandler):
            RESOURCE_NAME = "widget"

            def _extract_resource_id(self, path):
                # Custom: always extract from 4th segment
                parts = path.strip("/").split("/")
                return parts[3] if len(parts) > 3 else None

            def _get_resource(self, resource_id, handler):
                return self.json_response({"widget_id": resource_id})

        handler = CustomHandler(server_context={})
        mock_http_handler.path = "/api/v1/widgets/w-42"
        result = handler.handle("/api/v1/widgets/w-42", {}, mock_http_handler)
        assert _status(result) == 200
        assert _body(result)["widget_id"] == "w-42"

    def test_with_dependencies_returns_correct_subclass(self):
        """with_dependencies on a subclass should return an instance of that subclass."""

        class MyTypedHandler(TypedHandler):
            pass

        handler = MyTypedHandler.with_dependencies(server_context={})
        assert isinstance(handler, MyTypedHandler)

    def test_read_json_body_unicode(self, typed_handler):
        """Test reading JSON body with unicode characters."""
        data = {"message": "Hello \u00e9\u00e8\u00ea \u00fc\u00f6\u00e4"}
        http = _make_mock_http_handler(body=data)
        result = typed_handler.read_json_body(http)
        assert result == data

    def test_read_json_body_empty_object(self, typed_handler):
        """Test reading empty JSON object."""
        data = {}
        http = _make_mock_http_handler(body=data)
        result = typed_handler.read_json_body(http)
        assert result == {}

    def test_read_json_body_array(self, typed_handler):
        """Test reading JSON array (non-dict)."""
        body_bytes = json.dumps([1, 2, 3]).encode("utf-8")
        http = MagicMock()
        http.headers = {"Content-Length": str(len(body_bytes))}
        http.rfile = MagicMock()
        http.rfile.read.return_value = body_bytes
        result = typed_handler.read_json_body(http)
        assert result == [1, 2, 3]


class TestResourceHandlerPermissionGeneration:
    """Test ResourceHandler auto-generates correct permissions."""

    def test_default_resource_permissions(self):
        handler = ResourceHandler(server_context={})
        perms = handler.REQUIRED_PERMISSIONS
        assert perms["GET"] == "resource:read"
        assert perms["POST"] == "resource:create"
        assert perms["PUT"] == "resource:update"
        assert perms["PATCH"] == "resource:update"
        assert perms["DELETE"] == "resource:delete"

    def test_custom_resource_name_permissions(self):
        class TaskHandler(ResourceHandler):
            RESOURCE_NAME = "task"

        handler = TaskHandler(server_context={})
        assert handler.REQUIRED_PERMISSIONS["GET"] == "task:read"
        assert handler.REQUIRED_PERMISSIONS["DELETE"] == "task:delete"

    def test_permissions_overwritten_on_init(self):
        """Verify that __init__ sets permissions from resource name."""

        class V1Handler(ResourceHandler):
            RESOURCE_NAME = "v1_thing"

        handler = V1Handler(server_context={})
        assert handler.REQUIRED_PERMISSIONS["POST"] == "v1_thing:create"


class TestAdminHandlerLogging:
    """Test AdminHandler audit logging details."""

    def test_log_with_all_params(self, admin_handler, mock_http_handler, caplog):
        admin_handler._ensure_authenticated(mock_http_handler)
        with caplog.at_level(logging.INFO, logger="aragora.server.handlers.base"):
            admin_handler._log_admin_action(
                action="update_config",
                resource_id="cfg-001",
                details={"setting": "max_agents", "value": 10},
            )
        # Check that admin action was logged
        assert any("Admin action" in r.message for r in caplog.records)

    def test_log_unknown_user(self, admin_handler, caplog):
        """When current_user is None, user_id should be 'unknown'."""
        admin_handler._current_user = None
        with caplog.at_level(logging.INFO, logger="aragora.server.handlers.base"):
            admin_handler._log_admin_action(action="test_action")
        assert any("unknown" in r.message for r in caplog.records)


class TestAsyncTypedHandlerInheritance:
    """Test AsyncTypedHandler inherits TypedHandler functionality."""

    def test_read_json_body(self, async_handler):
        data = {"async": True}
        http = _make_mock_http_handler(body=data)
        result = async_handler.read_json_body(http)
        assert result == data

    def test_error_response(self, async_handler):
        result = async_handler.error_response("async error", 500)
        assert _status(result) == 500

    def test_get_storage(self):
        mock_storage = MagicMock()
        handler = AsyncTypedHandler(server_context={"storage": mock_storage})
        assert handler.get_storage() is mock_storage

    def test_get_user_store(self):
        mock_store = MagicMock()
        handler = AsyncTypedHandler(server_context={"user_store": mock_store})
        assert handler.get_user_store() is mock_store


class TestResourceHandlerPathEdgeCases:
    """Test ResourceHandler with various path patterns."""

    def test_deeply_nested_path(self, resource_handler):
        result = resource_handler._extract_resource_id("/a/b/c/d/e/f/g/item-99")
        assert result == "item-99"

    def test_empty_path(self, resource_handler):
        result = resource_handler._extract_resource_id("")
        assert result is None

    def test_single_slash(self, resource_handler):
        result = resource_handler._extract_resource_id("/")
        assert result is None

    def test_path_with_only_resource_name(self):
        class FooHandler(ResourceHandler):
            RESOURCE_NAME = "foo"

        handler = FooHandler(server_context={})
        assert handler._extract_resource_id("/foo") is None
        assert handler._extract_resource_id("/foos") is None
        assert handler._extract_resource_id("/foos/bar") == "bar"
