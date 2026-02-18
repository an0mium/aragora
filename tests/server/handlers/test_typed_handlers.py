"""
Tests for aragora.server.handlers.typed_handlers module.

Tests cover:
- TypedHandler base class functionality
- AuthenticatedHandler authentication enforcement
- PermissionHandler permission checking
- AdminHandler admin privilege requirements
- AsyncTypedHandler async method signatures
- ResourceHandler RESTful patterns
- Error response generation
- JSON body parsing
- Dependency injection for testing
- Handler method dispatch
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.typed_handlers import (
    AdminHandler,
    AsyncTypedHandler,
    AuthenticatedHandler,
    MaybeAsyncHandlerResult,
    PermissionHandler,
    ResourceHandler,
    TypedHandler,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Test Fixtures
# ===========================================================================


@dataclass
class MockAuthContext:
    """Mock authentication context for testing."""

    authenticated: bool = True
    user_id: str = "user-123"
    email: str = "test@example.com"
    org_id: str | None = "org-123"
    role: str = "member"
    roles: list[str] | None = None
    permissions: list[str] | None = None

    def __post_init__(self):
        if self.roles is None:
            self.roles = []
        if self.permissions is None:
            self.permissions = []

    @property
    def is_authenticated(self) -> bool:
        """Alias for authenticated."""
        return self.authenticated

    @property
    def is_admin(self) -> bool:
        """Check if user is admin or owner."""
        return self.role in ("owner", "admin") or "admin" in (self.roles or [])


def make_mock_handler(
    body: dict | None = None,
    method: str = "GET",
    headers: dict | None = None,
    path: str = "/api/v1/test",
) -> MagicMock:
    """Create a mock HTTP handler for testing."""
    handler = MagicMock()
    handler.command = method
    handler.headers = headers or {}
    handler.client_address = ("127.0.0.1", 12345)
    handler.path = path

    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        handler.headers["Content-Length"] = str(len(body_bytes))
        handler.rfile = BytesIO(body_bytes)
    else:
        handler.rfile = BytesIO(b"")
        handler.headers["Content-Length"] = "0"

    return handler


@pytest.fixture
def server_context() -> dict[str, Any]:
    """Create a minimal server context for testing."""
    return {
        "storage": MagicMock(),
        "user_store": MagicMock(),
    }


@pytest.fixture
def mock_authenticated_user() -> MockAuthContext:
    """Create an authenticated user context."""
    return MockAuthContext(authenticated=True, user_id="user-123")


@pytest.fixture
def mock_admin_user() -> MockAuthContext:
    """Create an admin user context."""
    return MockAuthContext(
        authenticated=True,
        user_id="admin-123",
        role="admin",
        roles=["admin"],
        permissions=["admin"],
    )


@pytest.fixture
def mock_unauthenticated_user() -> MockAuthContext:
    """Create an unauthenticated user context."""
    return MockAuthContext(authenticated=False, user_id=None)


# ===========================================================================
# TypedHandler Tests
# ===========================================================================


class TestTypedHandlerInit:
    """Tests for TypedHandler initialization."""

    def test_typed_handler_init(self, server_context):
        """TypedHandler initializes with server context."""
        handler = TypedHandler(server_context)
        assert handler.ctx == server_context

    def test_typed_handler_empty_context(self):
        """TypedHandler accepts empty context."""
        handler = TypedHandler({})
        assert handler.ctx == {}

    def test_typed_handler_ctx_attribute(self, server_context):
        """TypedHandler exposes ctx attribute."""
        handler = TypedHandler(server_context)
        assert handler.ctx is server_context


class TestTypedHandlerMethods:
    """Tests for TypedHandler HTTP method handlers."""

    def test_handle_returns_none_by_default(self, server_context):
        """Default handle method returns None."""
        handler = TypedHandler(server_context)
        mock_http = make_mock_handler()
        result = handler.handle("/test", {}, mock_http)
        assert result is None

    def test_handle_post_returns_none_by_default(self, server_context):
        """Default handle_post method returns None."""
        handler = TypedHandler(server_context)
        mock_http = make_mock_handler(method="POST")
        result = handler.handle_post("/test", {}, mock_http)
        assert result is None

    def test_handle_put_returns_none_by_default(self, server_context):
        """Default handle_put method returns None."""
        handler = TypedHandler(server_context)
        mock_http = make_mock_handler(method="PUT")
        result = handler.handle_put("/test", {}, mock_http)
        assert result is None

    def test_handle_patch_returns_none_by_default(self, server_context):
        """Default handle_patch method returns None."""
        handler = TypedHandler(server_context)
        mock_http = make_mock_handler(method="PATCH")
        result = handler.handle_patch("/test", {}, mock_http)
        assert result is None

    def test_handle_delete_returns_none_by_default(self, server_context):
        """Default handle_delete method returns None."""
        handler = TypedHandler(server_context)
        mock_http = make_mock_handler(method="DELETE")
        result = handler.handle_delete("/test", {}, mock_http)
        assert result is None


class TestTypedHandlerJsonParsing:
    """Tests for TypedHandler JSON body parsing."""

    def test_read_json_body_success(self, server_context):
        """read_json_body parses valid JSON."""
        handler = TypedHandler(server_context)
        mock_http = make_mock_handler(body={"key": "value"})

        result = handler.read_json_body(mock_http)
        assert result == {"key": "value"}

    def test_read_json_body_nested_structure(self, server_context):
        """read_json_body handles nested JSON structures."""
        handler = TypedHandler(server_context)
        body = {"nested": {"deep": {"value": 42}}, "list": [1, 2, 3]}
        mock_http = make_mock_handler(body=body)

        result = handler.read_json_body(mock_http)
        assert result == body
        assert result["nested"]["deep"]["value"] == 42

    def test_read_json_body_empty_content_length_zero(self, server_context):
        """read_json_body returns None for Content-Length: 0."""
        handler = TypedHandler(server_context)
        mock_http = make_mock_handler()
        mock_http.headers["Content-Length"] = "0"

        result = handler.read_json_body(mock_http)
        assert result is None

    def test_read_json_body_invalid_json(self, server_context):
        """read_json_body returns None for invalid JSON."""
        handler = TypedHandler(server_context)
        mock_http = MagicMock()
        body_bytes = b"not valid json {"
        mock_http.headers = {"Content-Length": str(len(body_bytes))}
        mock_http.rfile = BytesIO(body_bytes)

        result = handler.read_json_body(mock_http)
        assert result is None

    def test_read_json_body_exceeds_max_size(self, server_context):
        """read_json_body returns None when body exceeds max_size."""
        handler = TypedHandler(server_context)
        mock_http = MagicMock()
        mock_http.headers = {"Content-Length": "10000"}

        result = handler.read_json_body(mock_http, max_size=100)
        assert result is None

    def test_read_json_body_missing_content_length(self, server_context):
        """read_json_body handles missing Content-Length header."""
        handler = TypedHandler(server_context)
        mock_http = MagicMock()
        mock_http.headers = {}

        result = handler.read_json_body(mock_http)
        assert result is None

    def test_read_json_body_unicode_content(self, server_context):
        """read_json_body handles Unicode content correctly."""
        handler = TypedHandler(server_context)
        body = {"message": "Hello, World!"}
        mock_http = make_mock_handler(body=body)

        result = handler.read_json_body(mock_http)
        assert result["message"] == "Hello, World!"

    def test_read_json_body_special_characters(self, server_context):
        """read_json_body handles special characters."""
        handler = TypedHandler(server_context)
        body = {"text": 'Line 1\nLine 2\tTabbed "quoted"'}
        mock_http = make_mock_handler(body=body)

        result = handler.read_json_body(mock_http)
        assert result["text"] == 'Line 1\nLine 2\tTabbed "quoted"'


class TestTypedHandlerErrorResponse:
    """Tests for TypedHandler error response generation."""

    def test_error_response_basic(self, server_context):
        """error_response creates error with message."""
        handler = TypedHandler(server_context)
        result = handler.error_response("Test error", 400)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert body["error"] == "Test error"

    def test_error_response_not_found(self, server_context):
        """error_response handles 404 status."""
        handler = TypedHandler(server_context)
        result = handler.error_response("Not found", 404)

        assert result.status_code == 404
        body = json.loads(result.body)
        assert body["error"] == "Not found"

    def test_error_response_server_error(self, server_context):
        """error_response handles 500 status."""
        handler = TypedHandler(server_context)
        result = handler.error_response("Internal server error", 500)

        assert result.status_code == 500
        body = json.loads(result.body)
        assert body["error"] == "Internal server error"

    def test_error_response_default_status(self, server_context):
        """error_response defaults to 400 status."""
        handler = TypedHandler(server_context)
        result = handler.error_response("Bad request")

        assert result.status_code == 400


class TestTypedHandlerDependencyInjection:
    """Tests for TypedHandler dependency injection."""

    def test_get_user_store_from_context(self, server_context):
        """get_user_store returns store from context."""
        mock_store = MagicMock()
        server_context["user_store"] = mock_store

        handler = TypedHandler(server_context)
        assert handler.get_user_store() is mock_store

    def test_get_storage_from_context(self, server_context):
        """get_storage returns storage from context."""
        mock_storage = MagicMock()
        server_context["storage"] = mock_storage

        handler = TypedHandler(server_context)
        assert handler.get_storage() is mock_storage

    def test_get_user_store_missing(self):
        """get_user_store returns None when not in context."""
        handler = TypedHandler({})
        assert handler.get_user_store() is None

    def test_get_storage_missing(self):
        """get_storage returns None when not in context."""
        handler = TypedHandler({})
        assert handler.get_storage() is None

    def test_with_dependencies_injects_user_store(self, server_context):
        """with_dependencies allows injecting user_store."""
        mock_store = MagicMock()
        handler = TypedHandler.with_dependencies(server_context, user_store=mock_store)

        assert handler.get_user_store() is mock_store

    def test_with_dependencies_injects_storage(self, server_context):
        """with_dependencies allows injecting storage."""
        mock_storage = MagicMock()
        handler = TypedHandler.with_dependencies(server_context, storage=mock_storage)

        assert handler.get_storage() is mock_storage

    def test_with_dependencies_both(self, server_context):
        """with_dependencies injects both user_store and storage."""
        mock_store = MagicMock()
        mock_storage = MagicMock()

        handler = TypedHandler.with_dependencies(
            server_context, user_store=mock_store, storage=mock_storage
        )

        assert handler.get_user_store() is mock_store
        assert handler.get_storage() is mock_storage


class TestTypedHandlerAuthentication:
    """Tests for TypedHandler authentication helpers."""

    def test_require_auth_or_error_authenticated(self, server_context, mock_authenticated_user):
        """require_auth_or_error returns user when authenticated."""
        handler = TypedHandler(server_context)
        mock_http = make_mock_handler(headers={"Authorization": "Bearer token123"})

        with patch.object(handler, "get_current_user", return_value=mock_authenticated_user):
            user, err = handler.require_auth_or_error(mock_http)

        assert user is mock_authenticated_user
        assert err is None

    def test_require_auth_or_error_unauthenticated(self, server_context):
        """require_auth_or_error returns 401 error when not authenticated."""
        handler = TypedHandler(server_context)
        mock_http = make_mock_handler()

        with patch.object(handler, "get_current_user", return_value=None):
            user, err = handler.require_auth_or_error(mock_http)

        assert user is None
        assert err is not None
        assert err.status_code == 401
        body = json.loads(err.body)
        assert "Authentication required" in body["error"]

    def test_require_admin_or_error_admin_user(self, server_context, mock_admin_user):
        """require_admin_or_error returns user when admin."""
        handler = TypedHandler(server_context)
        mock_http = make_mock_handler(headers={"Authorization": "Bearer token123"})

        with patch.object(handler, "get_current_user", return_value=mock_admin_user):
            user, err = handler.require_admin_or_error(mock_http)

        assert user is mock_admin_user
        assert err is None

    def test_require_admin_or_error_non_admin(self, server_context, mock_authenticated_user):
        """require_admin_or_error returns 403 error when not admin."""
        handler = TypedHandler(server_context)
        mock_http = make_mock_handler(headers={"Authorization": "Bearer token123"})

        with patch.object(handler, "get_current_user", return_value=mock_authenticated_user):
            user, err = handler.require_admin_or_error(mock_http)

        assert user is None
        assert err is not None
        assert err.status_code == 403
        body = json.loads(err.body)
        assert "Admin access required" in body["error"]

    def test_require_admin_or_error_unauthenticated(self, server_context):
        """require_admin_or_error returns 401 error when not authenticated."""
        handler = TypedHandler(server_context)
        mock_http = make_mock_handler()

        with patch.object(handler, "get_current_user", return_value=None):
            user, err = handler.require_admin_or_error(mock_http)

        assert user is None
        assert err is not None
        assert err.status_code == 401


class TestTypedHandlerPermissions:
    """Tests for TypedHandler permission checking."""

    def test_require_permission_with_admin_role(self, server_context, mock_admin_user):
        """require_permission_or_error grants all permissions to admin."""
        handler = TypedHandler(server_context)
        mock_http = make_mock_handler(headers={"Authorization": "Bearer token123"})

        with patch.object(handler, "get_current_user", return_value=mock_admin_user):
            user, err = handler.require_permission_or_error(mock_http, "any:permission")

        assert user is mock_admin_user
        assert err is None

    def test_require_permission_with_owner_role(self, server_context):
        """require_permission_or_error grants all permissions to owner."""
        owner_user = MockAuthContext(
            authenticated=True,
            user_id="owner-123",
            role="owner",
            roles=["owner"],
        )
        handler = TypedHandler(server_context)
        mock_http = make_mock_handler(headers={"Authorization": "Bearer token123"})

        with patch.object(handler, "get_current_user", return_value=owner_user):
            user, err = handler.require_permission_or_error(mock_http, "any:permission")

        assert user is owner_user
        assert err is None

    def test_require_permission_with_specific_permission(self, server_context):
        """require_permission_or_error grants access with specific permission."""
        user_with_perm = MockAuthContext(
            authenticated=True,
            user_id="user-123",
            role="member",
            permissions=["debates:read"],
        )
        handler = TypedHandler(server_context)
        mock_http = make_mock_handler(headers={"Authorization": "Bearer token123"})

        with patch.object(handler, "get_current_user", return_value=user_with_perm):
            user, err = handler.require_permission_or_error(mock_http, "debates:read")

        assert user is user_with_perm
        assert err is None

    def test_require_permission_denied(self, server_context, mock_authenticated_user):
        """require_permission_or_error returns 403 when permission missing."""
        handler = TypedHandler(server_context)
        mock_http = make_mock_handler(headers={"Authorization": "Bearer token123"})

        with patch.object(handler, "get_current_user", return_value=mock_authenticated_user):
            # has_permission is imported inside the method, mock it at import location
            with patch.dict(
                "sys.modules",
                {
                    "aragora.server.handlers.utils.decorators": MagicMock(
                        has_permission=lambda r, p: False
                    )
                },
            ):
                user, err = handler.require_permission_or_error(mock_http, "admin:write")

        assert user is None
        assert err is not None
        assert err.status_code == 403
        body = json.loads(err.body)
        assert "Permission denied" in body["error"]

    def test_require_permission_unauthenticated(self, server_context):
        """require_permission_or_error returns 401 when not authenticated."""
        handler = TypedHandler(server_context)
        mock_http = make_mock_handler()

        with patch.object(handler, "get_current_user", return_value=None):
            user, err = handler.require_permission_or_error(mock_http, "debates:read")

        assert user is None
        assert err is not None
        assert err.status_code == 401


# ===========================================================================
# AuthenticatedHandler Tests
# ===========================================================================


class TestAuthenticatedHandler:
    """Tests for AuthenticatedHandler class."""

    def test_authenticated_handler_init(self, server_context):
        """AuthenticatedHandler initializes properly."""
        handler = AuthenticatedHandler(server_context)
        assert handler.ctx == server_context
        assert handler.current_user is None

    def test_ensure_authenticated_success(self, server_context, mock_authenticated_user):
        """_ensure_authenticated returns user when authenticated."""
        handler = AuthenticatedHandler(server_context)
        mock_http = make_mock_handler(headers={"Authorization": "Bearer token123"})

        with patch.object(handler, "get_current_user", return_value=mock_authenticated_user):
            user, err = handler._ensure_authenticated(mock_http)

        assert user is mock_authenticated_user
        assert err is None
        assert handler.current_user is mock_authenticated_user

    def test_ensure_authenticated_failure(self, server_context):
        """_ensure_authenticated returns error when not authenticated."""
        handler = AuthenticatedHandler(server_context)
        mock_http = make_mock_handler()

        with patch.object(handler, "get_current_user", return_value=None):
            user, err = handler._ensure_authenticated(mock_http)

        assert user is None
        assert err is not None
        assert err.status_code == 401
        assert handler.current_user is None

    def test_ensure_admin_success(self, server_context, mock_admin_user):
        """_ensure_admin returns user when admin."""
        handler = AuthenticatedHandler(server_context)
        mock_http = make_mock_handler(headers={"Authorization": "Bearer token123"})

        with patch.object(handler, "get_current_user", return_value=mock_admin_user):
            user, err = handler._ensure_admin(mock_http)

        assert user is mock_admin_user
        assert err is None
        assert handler.current_user is mock_admin_user

    def test_ensure_admin_failure_not_admin(self, server_context, mock_authenticated_user):
        """_ensure_admin returns 403 when not admin."""
        handler = AuthenticatedHandler(server_context)
        mock_http = make_mock_handler(headers={"Authorization": "Bearer token123"})

        with patch.object(handler, "get_current_user", return_value=mock_authenticated_user):
            user, err = handler._ensure_admin(mock_http)

        assert user is None
        assert err is not None
        assert err.status_code == 403
        assert handler.current_user is None

    def test_current_user_property(self, server_context, mock_authenticated_user):
        """current_user property returns cached user after authentication."""
        handler = AuthenticatedHandler(server_context)
        mock_http = make_mock_handler(headers={"Authorization": "Bearer token123"})

        with patch.object(handler, "get_current_user", return_value=mock_authenticated_user):
            handler._ensure_authenticated(mock_http)

        assert handler.current_user is mock_authenticated_user


# ===========================================================================
# PermissionHandler Tests
# ===========================================================================


class TestPermissionHandler:
    """Tests for PermissionHandler class."""

    def test_permission_handler_init(self, server_context):
        """PermissionHandler initializes with default permissions."""
        handler = PermissionHandler(server_context)
        assert handler.REQUIRED_PERMISSIONS is not None
        assert "GET" in handler.REQUIRED_PERMISSIONS

    def test_permission_handler_default_permissions_are_none(self, server_context):
        """Default REQUIRED_PERMISSIONS values are None (no permission required)."""
        handler = PermissionHandler(server_context)
        assert handler.REQUIRED_PERMISSIONS["GET"] is None
        assert handler.REQUIRED_PERMISSIONS["POST"] is None
        assert handler.REQUIRED_PERMISSIONS["PUT"] is None
        assert handler.REQUIRED_PERMISSIONS["PATCH"] is None
        assert handler.REQUIRED_PERMISSIONS["DELETE"] is None

    def test_ensure_permission_with_no_required_permission(
        self, server_context, mock_authenticated_user
    ):
        """_ensure_permission passes when no permission required for method."""
        handler = PermissionHandler(server_context)
        mock_http = make_mock_handler(method="GET")

        with patch.object(handler, "get_current_user", return_value=mock_authenticated_user):
            user, err = handler._ensure_permission(mock_http, "GET")

        assert user is mock_authenticated_user
        assert err is None

    def test_ensure_permission_with_required_permission(self, server_context):
        """_ensure_permission checks required permission for method."""

        class CustomPermissionHandler(PermissionHandler):
            REQUIRED_PERMISSIONS = {
                "GET": "resource:read",
                "POST": "resource:write",
            }

        user_with_perm = MockAuthContext(
            authenticated=True,
            user_id="user-123",
            permissions=["resource:read"],
        )
        handler = CustomPermissionHandler(server_context)
        mock_http = make_mock_handler(method="GET")

        with patch.object(handler, "get_current_user", return_value=user_with_perm):
            user, err = handler._ensure_permission(mock_http, "GET")

        assert user is user_with_perm
        assert err is None

    def test_ensure_permission_denied(self, server_context, mock_authenticated_user):
        """_ensure_permission returns 403 when permission denied."""

        class CustomPermissionHandler(PermissionHandler):
            REQUIRED_PERMISSIONS = {
                "GET": "admin:read",
            }

        handler = CustomPermissionHandler(server_context)
        mock_http = make_mock_handler(method="GET")

        with patch.object(handler, "get_current_user", return_value=mock_authenticated_user):
            # Mock the has_permission import to return False
            with patch.dict(
                "sys.modules",
                {
                    "aragora.server.handlers.utils.decorators": MagicMock(
                        has_permission=lambda r, p: False
                    )
                },
            ):
                user, err = handler._ensure_permission(mock_http, "GET")

        assert user is None
        assert err is not None
        assert err.status_code == 403

    def test_check_custom_permission_success(self, server_context):
        """_check_custom_permission checks arbitrary permissions."""
        user_with_perm = MockAuthContext(
            authenticated=True,
            user_id="user-123",
            permissions=["debates:fork"],
        )
        handler = PermissionHandler(server_context)
        mock_http = make_mock_handler()

        with patch.object(handler, "get_current_user", return_value=user_with_perm):
            user, err = handler._check_custom_permission(mock_http, "debates:fork")

        assert user is user_with_perm
        assert err is None

    def test_check_custom_permission_denied(self, server_context, mock_authenticated_user):
        """_check_custom_permission returns 403 when permission missing."""
        handler = PermissionHandler(server_context)
        mock_http = make_mock_handler()

        with patch.object(handler, "get_current_user", return_value=mock_authenticated_user):
            # Mock the has_permission import to return False
            with patch.dict(
                "sys.modules",
                {
                    "aragora.server.handlers.utils.decorators": MagicMock(
                        has_permission=lambda r, p: False
                    )
                },
            ):
                user, err = handler._check_custom_permission(mock_http, "admin:special")

        assert user is None
        assert err is not None
        assert err.status_code == 403


# ===========================================================================
# AdminHandler Tests
# ===========================================================================


class TestAdminHandler:
    """Tests for AdminHandler class."""

    def test_admin_handler_init(self, server_context):
        """AdminHandler initializes properly."""
        handler = AdminHandler(server_context)
        assert handler.ctx == server_context
        assert handler.AUDIT_ACTIONS is True

    def test_log_admin_action_with_user(self, server_context, mock_admin_user):
        """_log_admin_action logs action with user info."""
        handler = AdminHandler(server_context)
        handler._current_user = mock_admin_user

        # Should not raise
        handler._log_admin_action("update_config", "config-123", {"key": "value"})

    def test_log_admin_action_without_user(self, server_context):
        """_log_admin_action handles missing user."""
        handler = AdminHandler(server_context)
        handler._current_user = None

        # Should not raise
        handler._log_admin_action("delete_user", "user-456")

    def test_log_admin_action_disabled(self, server_context, mock_admin_user):
        """_log_admin_action does nothing when AUDIT_ACTIONS is False."""

        class NoAuditHandler(AdminHandler):
            AUDIT_ACTIONS = False

        handler = NoAuditHandler(server_context)
        handler._current_user = mock_admin_user

        # Should not raise and should not log
        handler._log_admin_action("sensitive_action")


# ===========================================================================
# AsyncTypedHandler Tests
# ===========================================================================


class TestAsyncTypedHandler:
    """Tests for AsyncTypedHandler class."""

    def test_async_handler_init(self, server_context):
        """AsyncTypedHandler initializes properly."""
        handler = AsyncTypedHandler(server_context)
        assert handler.ctx == server_context

    @pytest.mark.asyncio
    async def test_async_handle_returns_none(self, server_context):
        """Async handle method returns None by default."""
        handler = AsyncTypedHandler(server_context)
        mock_http = make_mock_handler()

        result = await handler.handle("/test", {}, mock_http)
        assert result is None

    @pytest.mark.asyncio
    async def test_async_handle_post_returns_none(self, server_context):
        """Async handle_post method returns None by default."""
        handler = AsyncTypedHandler(server_context)
        mock_http = make_mock_handler(method="POST")

        result = await handler.handle_post("/test", {}, mock_http)
        assert result is None

    @pytest.mark.asyncio
    async def test_async_handle_put_returns_none(self, server_context):
        """Async handle_put method returns None by default."""
        handler = AsyncTypedHandler(server_context)
        mock_http = make_mock_handler(method="PUT")

        result = await handler.handle_put("/test", {}, mock_http)
        assert result is None

    @pytest.mark.asyncio
    async def test_async_handle_patch_returns_none(self, server_context):
        """Async handle_patch method returns None by default."""
        handler = AsyncTypedHandler(server_context)
        mock_http = make_mock_handler(method="PATCH")

        result = await handler.handle_patch("/test", {}, mock_http)
        assert result is None

    @pytest.mark.asyncio
    async def test_async_handle_delete_returns_none(self, server_context):
        """Async handle_delete method returns None by default."""
        handler = AsyncTypedHandler(server_context)
        mock_http = make_mock_handler(method="DELETE")

        result = await handler.handle_delete("/test", {}, mock_http)
        assert result is None

    @pytest.mark.asyncio
    async def test_async_handler_can_be_subclassed(self, server_context):
        """AsyncTypedHandler can be subclassed with custom async methods."""

        class CustomAsyncHandler(AsyncTypedHandler):
            async def handle(self, path, query_params, handler):
                await asyncio.sleep(0)  # Simulate async operation
                return HandlerResult(
                    status_code=200,
                    content_type="application/json",
                    body=b'{"async": true}',
                )

        handler = CustomAsyncHandler(server_context)
        mock_http = make_mock_handler()

        result = await handler.handle("/test", {}, mock_http)
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["async"] is True


# ===========================================================================
# ResourceHandler Tests
# ===========================================================================


class TestResourceHandler:
    """Tests for ResourceHandler class."""

    def test_resource_handler_init(self, server_context):
        """ResourceHandler initializes with resource permissions."""
        handler = ResourceHandler(server_context)
        assert handler.RESOURCE_NAME == "resource"
        assert handler.RESOURCE_ID_PARAM == "id"

    def test_resource_handler_generates_permissions(self, server_context):
        """ResourceHandler generates permissions from resource name."""

        class DocumentHandler(ResourceHandler):
            RESOURCE_NAME = "document"

        handler = DocumentHandler(server_context)
        perms = handler._get_resource_permissions()

        assert perms["GET"] == "document:read"
        assert perms["POST"] == "document:create"
        assert perms["PUT"] == "document:update"
        assert perms["PATCH"] == "document:update"
        assert perms["DELETE"] == "document:delete"

    def test_extract_resource_id_from_path(self, server_context):
        """_extract_resource_id extracts ID from path."""

        class ItemHandler(ResourceHandler):
            RESOURCE_NAME = "item"

        handler = ItemHandler(server_context)

        # /api/v1/items/item-123
        resource_id = handler._extract_resource_id("/api/v1/items/item-123")
        assert resource_id == "item-123"

    def test_extract_resource_id_collection_endpoint(self, server_context):
        """_extract_resource_id returns None for collection endpoint."""

        class ItemHandler(ResourceHandler):
            RESOURCE_NAME = "item"

        handler = ItemHandler(server_context)

        # /api/v1/items (collection)
        resource_id = handler._extract_resource_id("/api/v1/items")
        assert resource_id is None

    def test_extract_resource_id_plural_collection(self, server_context):
        """_extract_resource_id returns None for plural collection."""

        class UserHandler(ResourceHandler):
            RESOURCE_NAME = "user"

        handler = UserHandler(server_context)

        # /api/v1/users
        resource_id = handler._extract_resource_id("/api/v1/users")
        assert resource_id is None

    def test_extract_resource_id_list_endpoint(self, server_context):
        """_extract_resource_id returns None for 'list' endpoint."""
        handler = ResourceHandler(server_context)

        resource_id = handler._extract_resource_id("/api/v1/resources/list")
        assert resource_id is None

    def test_extract_resource_id_search_endpoint(self, server_context):
        """_extract_resource_id returns None for 'search' endpoint."""
        handler = ResourceHandler(server_context)

        resource_id = handler._extract_resource_id("/api/v1/resources/search")
        assert resource_id is None

    def test_handle_get_collection(self, server_context, mock_authenticated_user):
        """handle routes GET without ID to _list_resources."""

        class CustomResourceHandler(ResourceHandler):
            RESOURCE_NAME = "item"
            list_called = False

            def _list_resources(self, query_params, handler):
                self.list_called = True
                return HandlerResult(
                    status_code=200,
                    content_type="application/json",
                    body=b'{"items": []}',
                )

        handler = CustomResourceHandler(server_context)
        mock_http = make_mock_handler(method="GET", path="/api/v1/items")

        with patch.object(
            handler, "_ensure_permission", return_value=(mock_authenticated_user, None)
        ):
            result = handler.handle("/api/v1/items", {}, mock_http)

        assert handler.list_called
        assert result.status_code == 200

    def test_handle_get_single_resource(self, server_context, mock_authenticated_user):
        """handle routes GET with ID to _get_resource."""

        class CustomResourceHandler(ResourceHandler):
            RESOURCE_NAME = "item"
            get_called = False
            get_id = None

            def _get_resource(self, resource_id, handler):
                self.get_called = True
                self.get_id = resource_id
                return HandlerResult(
                    status_code=200,
                    content_type="application/json",
                    body=b'{"id": "item-123"}',
                )

        handler = CustomResourceHandler(server_context)
        mock_http = make_mock_handler(method="GET", path="/api/v1/items/item-123")

        with patch.object(
            handler, "_ensure_permission", return_value=(mock_authenticated_user, None)
        ):
            result = handler.handle("/api/v1/items/item-123", {}, mock_http)

        assert handler.get_called
        assert handler.get_id == "item-123"
        assert result.status_code == 200

    def test_handle_post_creates_resource(self, server_context, mock_authenticated_user):
        """handle_post routes to _create_resource."""

        class CustomResourceHandler(ResourceHandler):
            RESOURCE_NAME = "item"
            create_called = False

            def _create_resource(self, handler):
                self.create_called = True
                return HandlerResult(
                    status_code=201,
                    content_type="application/json",
                    body=b'{"id": "new-item"}',
                )

        handler = CustomResourceHandler(server_context)
        mock_http = make_mock_handler(method="POST", body={"name": "New Item"})

        with patch.object(
            handler, "_ensure_permission", return_value=(mock_authenticated_user, None)
        ):
            result = handler.handle_post("/api/v1/items", {}, mock_http)

        assert handler.create_called
        assert result.status_code == 201

    def test_handle_put_updates_resource(self, server_context, mock_authenticated_user):
        """handle_put routes to _update_resource."""

        class CustomResourceHandler(ResourceHandler):
            RESOURCE_NAME = "item"
            update_called = False
            update_id = None

            def _update_resource(self, resource_id, handler):
                self.update_called = True
                self.update_id = resource_id
                return HandlerResult(
                    status_code=200,
                    content_type="application/json",
                    body=b'{"updated": true}',
                )

        handler = CustomResourceHandler(server_context)
        mock_http = make_mock_handler(method="PUT", body={"name": "Updated"})

        with patch.object(
            handler, "_ensure_permission", return_value=(mock_authenticated_user, None)
        ):
            result = handler.handle_put("/api/v1/items/item-123", {}, mock_http)

        assert handler.update_called
        assert handler.update_id == "item-123"
        assert result.status_code == 200

    def test_handle_put_requires_id(self, server_context, mock_authenticated_user):
        """handle_put returns 400 when ID is missing."""

        class ItemHandler(ResourceHandler):
            RESOURCE_NAME = "item"

        handler = ItemHandler(server_context)
        mock_http = make_mock_handler(method="PUT")

        with patch.object(
            handler, "_ensure_permission", return_value=(mock_authenticated_user, None)
        ):
            result = handler.handle_put("/api/v1/items", {}, mock_http)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "ID required" in body["error"]

    def test_handle_patch_routes_to_patch_resource(self, server_context, mock_authenticated_user):
        """handle_patch routes to _patch_resource."""

        class CustomResourceHandler(ResourceHandler):
            RESOURCE_NAME = "item"
            patch_called = False

            def _patch_resource(self, resource_id, handler):
                self.patch_called = True
                return HandlerResult(
                    status_code=200,
                    content_type="application/json",
                    body=b'{"patched": true}',
                )

        handler = CustomResourceHandler(server_context)
        mock_http = make_mock_handler(method="PATCH")

        with patch.object(
            handler, "_ensure_permission", return_value=(mock_authenticated_user, None)
        ):
            result = handler.handle_patch("/api/v1/items/item-123", {}, mock_http)

        assert handler.patch_called
        assert result.status_code == 200

    def test_handle_patch_defaults_to_update(self, server_context, mock_authenticated_user):
        """handle_patch defaults to _update_resource if _patch_resource not overridden."""

        class CustomResourceHandler(ResourceHandler):
            RESOURCE_NAME = "item"
            update_called = False

            def _update_resource(self, resource_id, handler):
                self.update_called = True
                return HandlerResult(
                    status_code=200,
                    content_type="application/json",
                    body=b'{"updated": true}',
                )

        handler = CustomResourceHandler(server_context)
        mock_http = make_mock_handler(method="PATCH")

        with patch.object(
            handler, "_ensure_permission", return_value=(mock_authenticated_user, None)
        ):
            result = handler.handle_patch("/api/v1/items/item-123", {}, mock_http)

        assert handler.update_called

    def test_handle_delete_deletes_resource(self, server_context, mock_authenticated_user):
        """handle_delete routes to _delete_resource."""

        class CustomResourceHandler(ResourceHandler):
            RESOURCE_NAME = "item"
            delete_called = False
            delete_id = None

            def _delete_resource(self, resource_id, handler):
                self.delete_called = True
                self.delete_id = resource_id
                return HandlerResult(
                    status_code=204,
                    content_type="application/json",
                    body=b"",
                )

        handler = CustomResourceHandler(server_context)
        mock_http = make_mock_handler(method="DELETE")

        with patch.object(
            handler, "_ensure_permission", return_value=(mock_authenticated_user, None)
        ):
            result = handler.handle_delete("/api/v1/items/item-123", {}, mock_http)

        assert handler.delete_called
        assert handler.delete_id == "item-123"
        assert result.status_code == 204

    def test_handle_delete_requires_id(self, server_context, mock_authenticated_user):
        """handle_delete returns 400 when ID is missing."""

        class ItemHandler(ResourceHandler):
            RESOURCE_NAME = "item"

        handler = ItemHandler(server_context)
        mock_http = make_mock_handler(method="DELETE")

        with patch.object(
            handler, "_ensure_permission", return_value=(mock_authenticated_user, None)
        ):
            result = handler.handle_delete("/api/v1/items", {}, mock_http)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "ID required" in body["error"]

    def test_default_resource_methods_return_501(self, server_context, mock_authenticated_user):
        """Default resource methods return 501 Not Implemented."""
        handler = ResourceHandler(server_context)
        mock_http = make_mock_handler()

        # Test _list_resources
        result = handler._list_resources({}, mock_http)
        assert result.status_code == 501

        # Test _get_resource
        result = handler._get_resource("id-123", mock_http)
        assert result.status_code == 501

        # Test _create_resource
        result = handler._create_resource(mock_http)
        assert result.status_code == 501

        # Test _update_resource
        result = handler._update_resource("id-123", mock_http)
        assert result.status_code == 501

        # Test _delete_resource
        result = handler._delete_resource("id-123", mock_http)
        assert result.status_code == 501


# ===========================================================================
# Edge Case Tests
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_string_handling(self, server_context):
        """Handler handles empty string body."""
        handler = TypedHandler(server_context)
        mock_http = MagicMock()
        mock_http.headers = {"Content-Length": "2"}
        mock_http.rfile = BytesIO(b'""')  # Empty JSON string

        result = handler.read_json_body(mock_http)
        assert result == ""

    def test_whitespace_only_json(self, server_context):
        """Handler handles whitespace-only body as invalid JSON."""
        handler = TypedHandler(server_context)
        mock_http = MagicMock()
        body = b"   "
        mock_http.headers = {"Content-Length": str(len(body))}
        mock_http.rfile = BytesIO(body)

        result = handler.read_json_body(mock_http)
        assert result is None

    def test_very_long_string_in_json(self, server_context):
        """Handler handles very long strings in JSON."""
        handler = TypedHandler(server_context)
        long_string = "x" * 10000
        body = {"text": long_string}
        mock_http = make_mock_handler(body=body)

        result = handler.read_json_body(mock_http)
        assert result["text"] == long_string
        assert len(result["text"]) == 10000

    def test_special_unicode_characters(self, server_context):
        """Handler handles special Unicode characters."""
        handler = TypedHandler(server_context)
        body = {"emoji": "Hello! \U0001f600", "chinese": "Chinese characters"}
        mock_http = make_mock_handler(body=body)

        result = handler.read_json_body(mock_http)
        assert "Hello!" in result["emoji"]

    def test_none_value_in_json(self, server_context):
        """Handler handles null/None values in JSON."""
        handler = TypedHandler(server_context)
        body = {"value": None, "nested": {"also_null": None}}
        mock_http = make_mock_handler(body=body)

        result = handler.read_json_body(mock_http)
        assert result["value"] is None
        assert result["nested"]["also_null"] is None

    def test_boolean_values_in_json(self, server_context):
        """Handler handles boolean values correctly."""
        handler = TypedHandler(server_context)
        body = {"true_val": True, "false_val": False}
        mock_http = make_mock_handler(body=body)

        result = handler.read_json_body(mock_http)
        assert result["true_val"] is True
        assert result["false_val"] is False

    def test_numeric_values_in_json(self, server_context):
        """Handler handles various numeric values."""
        handler = TypedHandler(server_context)
        body = {
            "int": 42,
            "float": 3.14159,
            "negative": -100,
            "zero": 0,
            "large": 9999999999999,
        }
        mock_http = make_mock_handler(body=body)

        result = handler.read_json_body(mock_http)
        assert result["int"] == 42
        assert result["float"] == 3.14159
        assert result["negative"] == -100
        assert result["zero"] == 0
        assert result["large"] == 9999999999999

    def test_list_values_in_json(self, server_context):
        """Handler handles list values correctly."""
        handler = TypedHandler(server_context)
        body = {
            "empty_list": [],
            "number_list": [1, 2, 3],
            "mixed_list": [1, "two", None, True],
        }
        mock_http = make_mock_handler(body=body)

        result = handler.read_json_body(mock_http)
        assert result["empty_list"] == []
        assert result["number_list"] == [1, 2, 3]
        assert result["mixed_list"] == [1, "two", None, True]

    def test_deeply_nested_json(self, server_context):
        """Handler handles deeply nested JSON structures."""
        handler = TypedHandler(server_context)
        body = {"l1": {"l2": {"l3": {"l4": {"l5": "deep"}}}}}
        mock_http = make_mock_handler(body=body)

        result = handler.read_json_body(mock_http)
        assert result["l1"]["l2"]["l3"]["l4"]["l5"] == "deep"

    def test_resource_handler_with_trailing_slash(self, server_context):
        """ResourceHandler handles paths with trailing slashes."""
        handler = ResourceHandler(server_context)

        # Path with trailing slash
        resource_id = handler._extract_resource_id("/api/v1/resources/res-123/")
        # After rstrip("/"), should extract "res-123"
        assert resource_id == "res-123"

    def test_resource_handler_with_empty_path_segment(self, server_context):
        """ResourceHandler handles paths with empty segments."""
        handler = ResourceHandler(server_context)

        # Path: /api/v1/resources// (double slash)
        resource_id = handler._extract_resource_id("/api/v1/resources//")
        # After processing, last non-empty segment logic applies
        assert resource_id is None or resource_id == ""


# ===========================================================================
# Handler Integration Tests
# ===========================================================================


class TestHandlerIntegration:
    """Integration tests for typed handlers."""

    def test_custom_handler_subclass(self, server_context, mock_authenticated_user):
        """Custom handler subclass works correctly."""

        class MyCustomHandler(TypedHandler):
            def handle(self, path, query_params, handler):
                return HandlerResult(
                    status_code=200,
                    content_type="application/json",
                    body=json.dumps({"path": path, "params": query_params}).encode(),
                )

        handler = MyCustomHandler(server_context)
        mock_http = make_mock_handler()

        result = handler.handle("/test/path", {"foo": "bar"}, mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["path"] == "/test/path"
        assert body["params"] == {"foo": "bar"}

    def test_authenticated_handler_subclass(self, server_context, mock_authenticated_user):
        """AuthenticatedHandler subclass enforces authentication."""

        class ProtectedHandler(AuthenticatedHandler):
            def handle(self, path, query_params, handler):
                user, err = self._ensure_authenticated(handler)
                if err:
                    return err
                return HandlerResult(
                    status_code=200,
                    content_type="application/json",
                    body=json.dumps({"user_id": user.user_id}).encode(),
                )

        handler = ProtectedHandler(server_context)
        mock_http = make_mock_handler(headers={"Authorization": "Bearer token123"})

        with patch.object(handler, "get_current_user", return_value=mock_authenticated_user):
            result = handler.handle("/protected", {}, mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["user_id"] == "user-123"

    def test_permission_handler_subclass(self, server_context):
        """PermissionHandler subclass enforces permissions."""

        class SecuredHandler(PermissionHandler):
            REQUIRED_PERMISSIONS = {
                "GET": "data:read",
                "POST": "data:write",
            }

            def handle(self, path, query_params, handler):
                user, err = self._ensure_permission(handler, "GET")
                if err:
                    return err
                return HandlerResult(
                    status_code=200,
                    content_type="application/json",
                    body=b'{"access": "granted"}',
                )

        user_with_perm = MockAuthContext(
            authenticated=True,
            user_id="user-123",
            permissions=["data:read"],
        )
        handler = SecuredHandler(server_context)
        mock_http = make_mock_handler()

        with patch.object(handler, "get_current_user", return_value=user_with_perm):
            result = handler.handle("/data", {}, mock_http)

        assert result.status_code == 200

    def test_resource_handler_full_crud_cycle(self, server_context, mock_authenticated_user):
        """ResourceHandler supports full CRUD cycle."""
        items = {}

        class ItemHandler(ResourceHandler):
            RESOURCE_NAME = "item"

            def _list_resources(self, query_params, handler):
                return HandlerResult(
                    status_code=200,
                    content_type="application/json",
                    body=json.dumps({"items": list(items.values())}).encode(),
                )

            def _create_resource(self, handler):
                body = self.read_json_body(handler)
                if not body:
                    return self.error_response("Invalid body")
                item_id = f"item-{len(items) + 1}"
                items[item_id] = {"id": item_id, **body}
                return HandlerResult(
                    status_code=201,
                    content_type="application/json",
                    body=json.dumps(items[item_id]).encode(),
                )

            def _get_resource(self, resource_id, handler):
                if resource_id not in items:
                    return self.error_response("Not found", 404)
                return HandlerResult(
                    status_code=200,
                    content_type="application/json",
                    body=json.dumps(items[resource_id]).encode(),
                )

            def _delete_resource(self, resource_id, handler):
                if resource_id in items:
                    del items[resource_id]
                return HandlerResult(
                    status_code=204,
                    content_type="application/json",
                    body=b"",
                )

        handler = ItemHandler(server_context)

        # Create
        mock_post = make_mock_handler(method="POST", body={"name": "Test Item"})
        with patch.object(
            handler, "_ensure_permission", return_value=(mock_authenticated_user, None)
        ):
            result = handler.handle_post("/api/items", {}, mock_post)
        assert result.status_code == 201
        created = json.loads(result.body)
        item_id = created["id"]

        # Read
        with patch.object(
            handler, "_ensure_permission", return_value=(mock_authenticated_user, None)
        ):
            result = handler.handle(f"/api/items/{item_id}", {}, make_mock_handler())
        assert result.status_code == 200
        assert json.loads(result.body)["name"] == "Test Item"

        # List
        with patch.object(
            handler, "_ensure_permission", return_value=(mock_authenticated_user, None)
        ):
            result = handler.handle("/api/items", {}, make_mock_handler())
        assert result.status_code == 200
        assert len(json.loads(result.body)["items"]) == 1

        # Delete
        with patch.object(
            handler, "_ensure_permission", return_value=(mock_authenticated_user, None)
        ):
            result = handler.handle_delete(f"/api/items/{item_id}", {}, make_mock_handler())
        assert result.status_code == 204

        # Verify deleted
        with patch.object(
            handler, "_ensure_permission", return_value=(mock_authenticated_user, None)
        ):
            result = handler.handle(f"/api/items/{item_id}", {}, make_mock_handler())
        assert result.status_code == 404


# ===========================================================================
# Type Alias Tests
# ===========================================================================


class TestTypeAliases:
    """Tests for type aliases."""

    def test_maybe_async_handler_result_accepts_handler_result(self):
        """MaybeAsyncHandlerResult accepts HandlerResult."""

        def sync_handler() -> MaybeAsyncHandlerResult:
            return HandlerResult(
                status_code=200,
                content_type="application/json",
                body=b"{}",
            )

        result = sync_handler()
        assert isinstance(result, HandlerResult)

    def test_maybe_async_handler_result_accepts_none(self):
        """MaybeAsyncHandlerResult accepts None."""

        def none_handler() -> MaybeAsyncHandlerResult:
            return None

        result = none_handler()
        assert result is None
