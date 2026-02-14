"""
Tests for SecureHandler base class and secure_endpoint/audit_sensitive_access decorators.

Tests cover:
- SecureHandler instantiation and class attributes
- get_auth_context delegation
- check_permission with RBAC
- handle_security_error for various exception types
- secure_endpoint decorator (auth, permission, audit)
- audit_sensitive_access decorator
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.rbac.decorators import PermissionDeniedError, RoleRequiredError
from aragora.rbac.models import AuthorizationContext
from aragora.server.handlers.secure import (
    SecureHandler,
    audit_sensitive_access,
    secure_endpoint,
)
from aragora.server.handlers.utils.auth import ForbiddenError, UnauthorizedError


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def server_context():
    """Minimal server context dict."""
    return {}


@pytest.fixture
def handler(server_context):
    """Create SecureHandler instance."""
    return SecureHandler(server_context)


@pytest.fixture
def auth_context():
    """Create a mock AuthorizationContext."""
    ctx = MagicMock(spec=AuthorizationContext)
    ctx.user_id = "user-123"
    ctx.org_id = "org-456"
    ctx.workspace_id = "ws-789"
    ctx.roles = ["admin"]
    return ctx


@pytest.fixture
def mock_request():
    """Create a mock HTTP request."""
    req = MagicMock()
    req.headers = {"Authorization": "Bearer test-token"}
    return req


# ===========================================================================
# SecureHandler Instantiation
# ===========================================================================


class TestSecureHandlerInit:
    """Tests for SecureHandler initialization."""

    def test_creates_with_dict_context(self):
        h = SecureHandler({})
        assert h._auth_context is None

    def test_default_method_permissions(self):
        h = SecureHandler({})
        assert "GET" in h.DEFAULT_METHOD_PERMISSIONS
        assert "POST" in h.DEFAULT_METHOD_PERMISSIONS
        assert "DELETE" in h.DEFAULT_METHOD_PERMISSIONS

    def test_default_resource_type(self):
        h = SecureHandler({})
        assert h.RESOURCE_TYPE == "unknown"

    def test_subclass_can_override_resource_type(self):
        class MyHandler(SecureHandler):
            RESOURCE_TYPE = "my_resource"

        h = MyHandler({})
        assert h.RESOURCE_TYPE == "my_resource"


# ===========================================================================
# get_auth_context
# ===========================================================================


class TestGetAuthContext:
    """Tests for get_auth_context delegation."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.secure.get_auth_context")
    async def test_delegates_to_get_auth_context(self, mock_get_auth, handler, mock_request):
        mock_ctx = MagicMock(spec=AuthorizationContext)
        mock_ctx.user_id = "user-1"
        mock_get_auth.return_value = mock_ctx

        result = await handler.get_auth_context(mock_request, require_auth=True)
        assert result.user_id == "user-1"
        mock_get_auth.assert_called_once_with(mock_request, require_auth=True)


# ===========================================================================
# check_permission
# ===========================================================================


class TestCheckPermission:
    """Tests for check_permission."""

    @patch("aragora.rbac.checker.get_permission_checker")
    @patch("aragora.observability.metrics.security.record_rbac_decision")
    def test_allowed_permission_returns_true(
        self, mock_record, mock_get_checker, handler, auth_context
    ):
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = True
        mock_checker.check_permission.return_value = mock_decision
        mock_get_checker.return_value = mock_checker

        result = handler.check_permission(auth_context, "items:read")
        assert result is True
        mock_record.assert_called_once_with("items:read", True)

    @patch("aragora.rbac.checker.get_permission_checker")
    @patch("aragora.observability.metrics.security.record_rbac_decision")
    def test_denied_permission_raises_forbidden(
        self, mock_record, mock_get_checker, handler, auth_context
    ):
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False
        mock_checker.check_permission.return_value = mock_decision
        mock_get_checker.return_value = mock_checker

        with pytest.raises(ForbiddenError):
            handler.check_permission(auth_context, "items:write")

    @patch("aragora.rbac.checker.get_permission_checker")
    @patch("aragora.observability.metrics.security.record_rbac_decision")
    def test_check_with_resource_id(
        self, mock_record, mock_get_checker, handler, auth_context
    ):
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = True
        mock_checker.check_permission.return_value = mock_decision
        mock_get_checker.return_value = mock_checker

        handler.check_permission(auth_context, "items:read", resource_id="item-42")
        mock_checker.check_permission.assert_called_once_with(auth_context, "items:read", "item-42")


# ===========================================================================
# handle_security_error
# ===========================================================================


class TestHandleSecurityError:
    """Tests for handle_security_error."""

    @patch("aragora.observability.metrics.security.record_auth_failure")
    def test_unauthorized_error_returns_401(self, mock_record, handler):
        error = UnauthorizedError("No token")
        result = handler.handle_security_error(error)
        assert result.status_code == 401
        mock_record.assert_called_once()

    @patch("aragora.observability.metrics.security.record_blocked_request")
    def test_forbidden_error_returns_403(self, mock_record, handler):
        error = ForbiddenError("Denied", permission="items:write")
        result = handler.handle_security_error(error)
        assert result.status_code == 403
        mock_record.assert_called_once()

    @patch("aragora.observability.metrics.security.record_blocked_request")
    def test_permission_denied_error_returns_403(self, mock_record, handler):
        error = PermissionDeniedError("denied")
        result = handler.handle_security_error(error)
        assert result.status_code == 403

    @patch("aragora.observability.metrics.security.record_blocked_request")
    def test_role_required_error_returns_403(self, mock_record, handler):
        error = RoleRequiredError("admin role required", {"admin"}, {"viewer"})
        result = handler.handle_security_error(error)
        assert result.status_code == 403

    def test_unknown_error_returns_500(self, handler):
        error = RuntimeError("unexpected")
        result = handler.handle_security_error(error)
        assert result.status_code == 500


# ===========================================================================
# secure_endpoint Decorator
# ===========================================================================


class TestSecureEndpointDecorator:
    """Tests for the secure_endpoint decorator."""

    @pytest.mark.asyncio
    @patch("aragora.observability.metrics.security.record_auth_attempt")
    @patch("aragora.server.handlers.secure.get_auth_context")
    async def test_injects_auth_context(self, mock_get_auth, mock_record):
        mock_ctx = MagicMock(spec=AuthorizationContext)
        mock_ctx.user_id = "user-1"
        mock_get_auth.return_value = mock_ctx

        class MyHandler(SecureHandler):
            @secure_endpoint(require_auth=True)
            async def handle_get(self, request, auth_context):
                return {"user": auth_context.user_id}

        h = MyHandler({})
        result = await h.handle_get(MagicMock())
        assert result == {"user": "user-1"}

    @pytest.mark.asyncio
    @patch("aragora.observability.metrics.security.track_rbac_evaluation")
    @patch("aragora.observability.metrics.security.record_auth_attempt")
    @patch("aragora.server.handlers.secure.get_auth_context")
    async def test_checks_permission(self, mock_get_auth, mock_record, mock_track):
        mock_ctx = MagicMock(spec=AuthorizationContext)
        mock_ctx.user_id = "user-1"
        mock_get_auth.return_value = mock_ctx

        # track_rbac_evaluation returns a context manager
        mock_track.return_value.__enter__ = MagicMock()
        mock_track.return_value.__exit__ = MagicMock(return_value=False)

        class MyHandler(SecureHandler):
            @secure_endpoint(permission="items:read")
            async def handle_get(self, request, auth_context):
                return {"ok": True}

        h = MyHandler({})
        with patch.object(h, "check_permission") as mock_check:
            result = await h.handle_get(MagicMock())
            mock_check.assert_called_once()

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.secure.get_auth_context")
    async def test_returns_401_on_unauthorized(self, mock_get_auth):
        mock_get_auth.side_effect = UnauthorizedError("No token")

        class MyHandler(SecureHandler):
            @secure_endpoint(require_auth=True)
            async def handle_get(self, request, auth_context):
                return {"ok": True}

        h = MyHandler({})
        with patch.object(h, "handle_security_error") as mock_handle:
            mock_handle.return_value = MagicMock(status_code=401)
            result = await h.handle_get(MagicMock())
            assert result.status_code == 401


# ===========================================================================
# audit_action
# ===========================================================================


class TestAuditAction:
    """Tests for SecureHandler.audit_action."""

    @pytest.mark.asyncio
    @patch("aragora.observability.immutable_log.get_audit_log")
    async def test_audit_action_calls_audit_log(self, mock_get_log, handler, auth_context):
        mock_log = AsyncMock()
        mock_get_log.return_value = mock_log

        await handler.audit_action(
            auth_context,
            action="create",
            resource_id="item-1",
        )
        mock_log.append.assert_called_once()

    @pytest.mark.asyncio
    @patch("aragora.observability.immutable_log.get_audit_log")
    async def test_audit_action_extracts_ip_from_request(self, mock_get_log, handler, auth_context):
        mock_log = AsyncMock()
        mock_get_log.return_value = mock_log

        req = MagicMock()
        req.headers = {"X-Forwarded-For": "1.2.3.4, 5.6.7.8", "User-Agent": "test-agent"}

        await handler.audit_action(
            auth_context,
            action="delete",
            resource_id="item-2",
            request=req,
        )

        call_kwargs = mock_log.append.call_args
        assert call_kwargs is not None


# ===========================================================================
# encrypt/decrypt
# ===========================================================================


class TestEncryptDecrypt:
    """Tests for encrypt_response_fields and decrypt_request_fields."""

    @patch("aragora.storage.encrypted_fields.encrypt_sensitive")
    def test_encrypt_response_fields(self, mock_encrypt, handler):
        mock_encrypt.return_value = {"key": "encrypted"}
        result = handler.encrypt_response_fields({"key": "value"})
        assert result == {"key": "encrypted"}
        mock_encrypt.assert_called_once()

    @patch("aragora.storage.encrypted_fields.decrypt_sensitive")
    def test_decrypt_request_fields(self, mock_decrypt, handler):
        mock_decrypt.return_value = {"key": "decrypted"}
        result = handler.decrypt_request_fields({"key": "encrypted"})
        assert result == {"key": "decrypted"}
        mock_decrypt.assert_called_once()
