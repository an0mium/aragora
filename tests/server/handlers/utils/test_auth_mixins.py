"""Tests for auth_mixins module."""

from __future__ import annotations

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.utils.auth_mixins import (
    AuthenticatedHandlerMixin,
    SecureEndpointMixin,
    require_permission,
    require_any_permission,
    require_all_permissions,
)


# =============================================================================
# Mock Classes for Testing
# =============================================================================


class MockAuthorizationContext:
    """Mock authorization context for tests."""

    def __init__(
        self,
        user_id: str = "test-user",
        org_id: Optional[str] = "test-org",
        roles: Optional[set] = None,
        permissions: Optional[set] = None,
    ):
        self.user_id = user_id
        self.org_id = org_id
        self.roles = roles or {"member"}
        self.permissions = permissions or {"read"}


class MockForbiddenError(Exception):
    """Mock ForbiddenError for tests."""

    pass


class MockUnauthorizedError(Exception):
    """Mock UnauthorizedError for tests."""

    pass


class MockSecureHandler(SecureEndpointMixin):
    """Mock handler class that implements SecureEndpointMixin."""

    def __init__(
        self,
        auth_context: Optional[MockAuthorizationContext] = None,
        raise_unauthorized: bool = False,
        raise_forbidden: bool = False,
        raise_unexpected: bool = False,
        permission_granted: bool = True,
    ):
        self._auth_context = auth_context
        self._raise_unauthorized = raise_unauthorized
        self._raise_forbidden = raise_forbidden
        self._raise_unexpected = raise_unexpected
        self._permission_granted = permission_granted

    async def get_auth_context(
        self, request: Any, require_auth: bool = False
    ) -> MockAuthorizationContext:
        if self._raise_unauthorized:
            from aragora.server.handlers.utils.auth_mixins import _UnauthorizedError

            raise _UnauthorizedError("Unauthorized")
        if self._raise_unexpected:
            raise RuntimeError("Unexpected error")
        if self._auth_context is None:
            self._auth_context = MockAuthorizationContext()
        return self._auth_context

    def check_permission(
        self, auth_context: Any, permission: str, resource_id: Optional[str] = None
    ) -> None:
        if self._raise_forbidden:
            from aragora.server.handlers.utils.auth_mixins import _ForbiddenError

            raise _ForbiddenError(f"Permission denied: {permission}")
        if not self._permission_granted:
            from aragora.server.handlers.utils.auth_mixins import _ForbiddenError

            raise _ForbiddenError(f"Permission denied: {permission}")


# =============================================================================
# Test SecureEndpointMixin
# =============================================================================


class TestSecureEndpointMixinRequireAuthOrError:
    """Tests for SecureEndpointMixin.require_auth_or_error."""

    @pytest.mark.asyncio
    async def test_returns_auth_context_on_success(self):
        """Should return (context, None) when authentication succeeds."""
        handler = MockSecureHandler(auth_context=MockAuthorizationContext(user_id="user-123"))
        request = MagicMock()

        auth_ctx, error = await handler.require_auth_or_error(request)

        assert error is None
        assert auth_ctx is not None
        assert auth_ctx.user_id == "user-123"

    @pytest.mark.asyncio
    async def test_returns_401_on_unauthorized(self):
        """Should return (None, 401 response) when unauthorized."""
        handler = MockSecureHandler(raise_unauthorized=True)
        request = MagicMock()

        auth_ctx, error = await handler.require_auth_or_error(request)

        assert auth_ctx is None
        assert error is not None
        assert error.status_code == 401

    @pytest.mark.asyncio
    async def test_returns_500_on_unexpected_error(self):
        """Should return (None, 500 response) on unexpected errors."""
        handler = MockSecureHandler(raise_unexpected=True)
        request = MagicMock()

        auth_ctx, error = await handler.require_auth_or_error(request)

        assert auth_ctx is None
        assert error is not None
        assert error.status_code == 500


class TestSecureEndpointMixinRequirePermissionOrError:
    """Tests for SecureEndpointMixin.require_permission_or_error."""

    @pytest.mark.asyncio
    async def test_returns_auth_context_on_success(self):
        """Should return (context, None) when permission is granted."""
        handler = MockSecureHandler(permission_granted=True)
        request = MagicMock()

        auth_ctx, error = await handler.require_permission_or_error(request, "resource:read")

        assert error is None
        assert auth_ctx is not None

    @pytest.mark.asyncio
    async def test_returns_401_on_unauthorized(self):
        """Should return (None, 401 response) when unauthorized."""
        handler = MockSecureHandler(raise_unauthorized=True)
        request = MagicMock()

        auth_ctx, error = await handler.require_permission_or_error(request, "resource:read")

        assert auth_ctx is None
        assert error is not None
        assert error.status_code == 401

    @pytest.mark.asyncio
    async def test_returns_403_on_forbidden(self):
        """Should return (None, 403 response) when permission denied."""
        handler = MockSecureHandler(raise_forbidden=True)
        request = MagicMock()

        auth_ctx, error = await handler.require_permission_or_error(request, "admin:manage")

        assert auth_ctx is None
        assert error is not None
        assert error.status_code == 403

    @pytest.mark.asyncio
    async def test_returns_500_on_unexpected_error(self):
        """Should return (None, 500 response) on unexpected errors."""
        handler = MockSecureHandler(raise_unexpected=True)
        request = MagicMock()

        auth_ctx, error = await handler.require_permission_or_error(request, "resource:read")

        assert auth_ctx is None
        assert error is not None
        assert error.status_code == 500

    @pytest.mark.asyncio
    async def test_passes_resource_id_to_check(self):
        """Should pass resource_id to check_permission."""
        handler = MockSecureHandler()
        handler.check_permission = MagicMock()
        request = MagicMock()

        await handler.require_permission_or_error(request, "resource:read", resource_id="res-123")

        handler.check_permission.assert_called_once()
        call_args = handler.check_permission.call_args
        assert call_args[0][1] == "resource:read"
        assert call_args[0][2] == "res-123"


class TestSecureEndpointMixinRequireAnyPermissionOrError:
    """Tests for SecureEndpointMixin.require_any_permission_or_error."""

    @pytest.mark.asyncio
    async def test_returns_context_if_any_permission_granted(self):
        """Should succeed if any permission is granted."""
        handler = MockSecureHandler(permission_granted=True)
        request = MagicMock()

        auth_ctx, error = await handler.require_any_permission_or_error(
            request, ["admin:read", "admin:write"]
        )

        assert error is None
        assert auth_ctx is not None

    @pytest.mark.asyncio
    async def test_returns_400_for_empty_permissions(self):
        """Should return 400 if no permissions specified."""
        handler = MockSecureHandler()
        request = MagicMock()

        auth_ctx, error = await handler.require_any_permission_or_error(request, [])

        assert auth_ctx is None
        assert error is not None
        assert error.status_code == 400

    @pytest.mark.asyncio
    async def test_returns_403_when_all_permissions_denied(self):
        """Should return 403 when all permissions are denied."""
        handler = MockSecureHandler(permission_granted=False)
        request = MagicMock()

        auth_ctx, error = await handler.require_any_permission_or_error(
            request, ["admin:read", "admin:write"]
        )

        assert auth_ctx is None
        assert error is not None
        assert error.status_code == 403

    @pytest.mark.asyncio
    async def test_returns_401_on_unauthorized(self):
        """Should return 401 when unauthorized."""
        handler = MockSecureHandler(raise_unauthorized=True)
        request = MagicMock()

        auth_ctx, error = await handler.require_any_permission_or_error(request, ["resource:read"])

        assert auth_ctx is None
        assert error is not None
        assert error.status_code == 401


class TestSecureEndpointMixinRequireAllPermissionsOrError:
    """Tests for SecureEndpointMixin.require_all_permissions_or_error."""

    @pytest.mark.asyncio
    async def test_returns_context_when_all_granted(self):
        """Should succeed when all permissions are granted."""
        handler = MockSecureHandler(permission_granted=True)
        request = MagicMock()

        auth_ctx, error = await handler.require_all_permissions_or_error(
            request, ["resource:read", "resource:write"]
        )

        assert error is None
        assert auth_ctx is not None

    @pytest.mark.asyncio
    async def test_returns_400_for_empty_permissions(self):
        """Should return 400 if no permissions specified."""
        handler = MockSecureHandler()
        request = MagicMock()

        auth_ctx, error = await handler.require_all_permissions_or_error(request, [])

        assert auth_ctx is None
        assert error is not None
        assert error.status_code == 400

    @pytest.mark.asyncio
    async def test_returns_403_when_any_permission_denied(self):
        """Should return 403 when any permission is denied."""
        handler = MockSecureHandler(raise_forbidden=True)
        request = MagicMock()

        auth_ctx, error = await handler.require_all_permissions_or_error(
            request, ["resource:read", "resource:write"]
        )

        assert auth_ctx is None
        assert error is not None
        assert error.status_code == 403


class TestSecureEndpointMixinRequireAdminOrError:
    """Tests for SecureEndpointMixin.require_admin_or_error."""

    @pytest.mark.asyncio
    async def test_delegates_to_require_permission_or_error(self):
        """Should delegate to require_permission_or_error with admin:*."""
        handler = MockSecureHandler(permission_granted=True)
        request = MagicMock()

        auth_ctx, error = await handler.require_admin_or_error(request)

        assert error is None
        assert auth_ctx is not None


# =============================================================================
# Test AuthenticatedHandlerMixin
# =============================================================================


class TestAuthenticatedHandlerMixin:
    """Tests for AuthenticatedHandlerMixin."""

    def test_current_user_initially_none(self):
        """Should have current_user as None initially."""

        class TestHandler(AuthenticatedHandlerMixin):
            pass

        handler = TestHandler()
        assert handler.current_user is None

    def test_user_id_returns_none_when_no_auth(self):
        """Should return None for user_id when not authenticated."""

        class TestHandler(AuthenticatedHandlerMixin):
            pass

        handler = TestHandler()
        assert handler.user_id is None

    def test_org_id_returns_none_when_no_auth(self):
        """Should return None for org_id when not authenticated."""

        class TestHandler(AuthenticatedHandlerMixin):
            pass

        handler = TestHandler()
        assert handler.org_id is None

    def test_set_auth_context_stores_context(self):
        """Should store auth context when set."""

        class TestHandler(AuthenticatedHandlerMixin):
            pass

        handler = TestHandler()
        ctx = MockAuthorizationContext(user_id="user-123", org_id="org-456")
        handler.set_auth_context(ctx)

        assert handler.current_user == ctx
        assert handler.user_id == "user-123"
        assert handler.org_id == "org-456"

    def test_clear_auth_context_removes_context(self):
        """Should clear auth context when cleared."""

        class TestHandler(AuthenticatedHandlerMixin):
            pass

        handler = TestHandler()
        handler.set_auth_context(MockAuthorizationContext())
        assert handler.current_user is not None

        handler.clear_auth_context()
        assert handler.current_user is None


# =============================================================================
# Test Decorators
# =============================================================================


class TestRequirePermissionDecorator:
    """Tests for @require_permission decorator."""

    @pytest.mark.asyncio
    async def test_calls_function_when_authorized(self):
        """Should call decorated function when permission granted."""
        called = []

        class MockHandler(SecureEndpointMixin):
            async def get_auth_context(self, request, require_auth=False):
                return MockAuthorizationContext()

            def check_permission(self, ctx, perm, res_id=None):
                pass  # Grant permission

            def set_auth_context(self, ctx):
                pass

            @require_permission("resource:read")
            async def test_method(self, request):
                called.append(True)
                return {"success": True}

        handler = MockHandler()
        result = await handler.test_method(MagicMock())

        assert called == [True]
        assert result == {"success": True}


class TestRequireAnyPermissionDecorator:
    """Tests for @require_any_permission decorator."""

    @pytest.mark.asyncio
    async def test_calls_function_when_any_permission_granted(self):
        """Should call decorated function when any permission granted."""
        called = []

        class MockHandler(SecureEndpointMixin, AuthenticatedHandlerMixin):
            async def get_auth_context(self, request, require_auth=False):
                return MockAuthorizationContext()

            def check_permission(self, ctx, perm, res_id=None):
                pass  # Grant permission

            async def require_any_permission_or_error(self, request, permissions, resource_id=None):
                return MockAuthorizationContext(), None

            @require_any_permission("read", "write")
            async def test_method(self, request):
                called.append(True)
                return {"success": True}

        handler = MockHandler()
        result = await handler.test_method(MagicMock())

        assert called == [True]
        assert result == {"success": True}


class TestRequireAllPermissionsDecorator:
    """Tests for @require_all_permissions decorator."""

    @pytest.mark.asyncio
    async def test_calls_function_when_all_permissions_granted(self):
        """Should call decorated function when all permissions granted."""
        called = []

        class MockHandler(SecureEndpointMixin, AuthenticatedHandlerMixin):
            async def get_auth_context(self, request, require_auth=False):
                return MockAuthorizationContext()

            def check_permission(self, ctx, perm, res_id=None):
                pass  # Grant permission

            async def require_all_permissions_or_error(
                self, request, permissions, resource_id=None
            ):
                return MockAuthorizationContext(), None

            @require_all_permissions("read", "write")
            async def test_method(self, request):
                called.append(True)
                return {"success": True}

        handler = MockHandler()
        result = await handler.test_method(MagicMock())

        assert called == [True]
        assert result == {"success": True}
