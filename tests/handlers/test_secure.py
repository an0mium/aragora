"""
Tests for SecureHandler base class, secure_endpoint decorator, and audit_sensitive_access decorator.

Tests cover:
- SecureHandler initialization and class attributes
- get_auth_context delegation
- check_permission with RBAC checker integration
- audit_action logging with request metadata extraction
- encrypt_response_fields / decrypt_request_fields
- handle_security_error for all error types
- secure_endpoint decorator: auth, permission, audit, error handling
- audit_sensitive_access decorator: metrics and audit trail
- Edge cases: anonymous users, missing request headers, None resource IDs
"""

from __future__ import annotations

import json
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
from aragora.server.handlers.utils.responses import HandlerResult


# ============================================================================
# Patch target constants (all lazy imports in secure.py)
# ============================================================================

_METRICS_MOD = "aragora.observability.metrics.security"
_AUDIT_LOG_MOD = "aragora.observability.immutable_log"
_SECURITY_AUDIT_MOD = "aragora.observability.security_audit"
_ENCRYPTED_MOD = "aragora.storage.encrypted_fields"
_RBAC_CHECKER_MOD = "aragora.rbac.checker"


# ============================================================================
# Helpers
# ============================================================================


def parse_body(result: HandlerResult) -> dict:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


def _make_auth_context(
    user_id: str = "user-1",
    user_email: str = "user@example.com",
    org_id: str = "org-1",
    workspace_id: str = "ws-1",
    roles: set[str] | None = None,
    permissions: set[str] | None = None,
) -> AuthorizationContext:
    """Create an AuthorizationContext for testing."""
    return AuthorizationContext(
        user_id=user_id,
        user_email=user_email,
        org_id=org_id,
        workspace_id=workspace_id,
        roles=roles or {"member"},
        permissions=permissions or {"read"},
    )


def _make_request(
    headers: dict[str, str] | None = None,
    remote: str = "10.0.0.1",
) -> MagicMock:
    """Create a mock request object."""
    req = MagicMock()
    req.headers = headers or {}
    req.remote = remote
    return req


def _patch_metrics(**overrides):
    """Context manager that patches all security metrics functions."""
    defaults = {
        "record_auth_attempt": MagicMock(),
        "record_auth_failure": MagicMock(),
        "record_blocked_request": MagicMock(),
        "record_rbac_decision": MagicMock(),
        "track_rbac_evaluation": MagicMock(
            return_value=MagicMock(
                __enter__=MagicMock(),
                __exit__=MagicMock(return_value=False),
            )
        ),
    }
    defaults.update(overrides)

    # Return a combined context manager patching all metrics at the source module
    import contextlib

    @contextlib.contextmanager
    def _cm():
        with (
            patch(f"{_METRICS_MOD}.record_auth_attempt", defaults["record_auth_attempt"]),
            patch(f"{_METRICS_MOD}.record_auth_failure", defaults["record_auth_failure"]),
            patch(f"{_METRICS_MOD}.record_blocked_request", defaults["record_blocked_request"]),
            patch(f"{_METRICS_MOD}.record_rbac_decision", defaults["record_rbac_decision"]),
            patch(f"{_METRICS_MOD}.track_rbac_evaluation", defaults["track_rbac_evaluation"]),
        ):
            yield defaults

    return _cm()


# ============================================================================
# Concrete subclass for testing (SecureHandler is abstract-ish)
# ============================================================================


class _TestSecureHandler(SecureHandler):
    """Concrete handler subclass for testing SecureHandler behaviour."""

    RESOURCE_TYPE = "test_resource"

    DEFAULT_METHOD_PERMISSIONS = {
        "GET": "test_resource.read",
        "POST": "test_resource.create",
        "PUT": "test_resource.update",
        "PATCH": "test_resource.update",
        "DELETE": "test_resource.delete",
    }


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def server_context():
    """Minimal server context."""
    return {"storage": MagicMock(), "user_store": MagicMock()}


@pytest.fixture
def handler(server_context):
    """Create a test SecureHandler instance."""
    return _TestSecureHandler(server_context)


@pytest.fixture
def auth_ctx():
    """Standard test AuthorizationContext."""
    return _make_auth_context()


@pytest.fixture
def admin_ctx():
    """Admin AuthorizationContext."""
    return _make_auth_context(
        user_id="admin-1",
        roles={"admin", "owner"},
        permissions={"*"},
    )


# ============================================================================
# SecureHandler Initialization
# ============================================================================


class TestSecureHandlerInit:
    """Tests for SecureHandler construction and class attributes."""

    def test_init_with_dict_context(self, server_context):
        """Handler initializes from a plain dict server context."""
        h = _TestSecureHandler(server_context)
        assert h._auth_context is None

    def test_default_method_permissions_on_base(self):
        """Base SecureHandler has None-valued default permissions."""
        for method in ("GET", "POST", "PUT", "PATCH", "DELETE"):
            assert SecureHandler.DEFAULT_METHOD_PERMISSIONS[method] is None

    def test_subclass_overrides_method_permissions(self, handler):
        """Subclass overrides default method permissions."""
        assert handler.DEFAULT_METHOD_PERMISSIONS["GET"] == "test_resource.read"
        assert handler.DEFAULT_METHOD_PERMISSIONS["POST"] == "test_resource.create"
        assert handler.DEFAULT_METHOD_PERMISSIONS["DELETE"] == "test_resource.delete"

    def test_resource_type_default(self):
        """Base SecureHandler has 'unknown' resource type."""
        assert SecureHandler.RESOURCE_TYPE == "unknown"

    def test_resource_type_subclass(self, handler):
        """Subclass overrides resource type."""
        assert handler.RESOURCE_TYPE == "test_resource"

    def test_auth_context_initially_none(self, handler):
        """Handler starts with no auth context."""
        assert handler._auth_context is None

    def test_init_with_empty_context(self):
        """Handler works with empty server context dict."""
        h = _TestSecureHandler({})
        assert h._auth_context is None

    def test_default_method_permissions_has_all_http_methods(self):
        """DEFAULT_METHOD_PERMISSIONS covers all standard HTTP methods."""
        for method in ("GET", "POST", "PUT", "PATCH", "DELETE"):
            assert method in SecureHandler.DEFAULT_METHOD_PERMISSIONS


# ============================================================================
# get_auth_context
# ============================================================================


class TestGetAuthContext:
    """Tests for SecureHandler.get_auth_context.

    These tests use no_auto_auth to prevent the conftest from patching
    SecureHandler.get_auth_context, so we can test delegation to the
    underlying utility function.
    """

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_delegates_to_utils_auth(self, handler):
        """get_auth_context delegates to the utility function."""
        mock_ctx = _make_auth_context()
        request = _make_request()

        with patch(
            "aragora.server.handlers.secure.get_auth_context",
            new_callable=AsyncMock,
            return_value=mock_ctx,
        ) as mock_fn:
            result = await handler.get_auth_context(request, require_auth=True)

        assert result is mock_ctx
        mock_fn.assert_awaited_once_with(request, require_auth=True)

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_require_auth_default_is_true(self, handler):
        """get_auth_context defaults to require_auth=True."""
        mock_ctx = _make_auth_context()
        request = _make_request()

        with patch(
            "aragora.server.handlers.secure.get_auth_context",
            new_callable=AsyncMock,
            return_value=mock_ctx,
        ) as mock_fn:
            result = await handler.get_auth_context(request)

        mock_fn.assert_awaited_once_with(request, require_auth=True)

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_require_auth_false(self, handler):
        """get_auth_context passes require_auth=False when specified."""
        mock_ctx = _make_auth_context(user_id="anonymous")
        request = _make_request()

        with patch(
            "aragora.server.handlers.secure.get_auth_context",
            new_callable=AsyncMock,
            return_value=mock_ctx,
        ) as mock_fn:
            result = await handler.get_auth_context(request, require_auth=False)

        mock_fn.assert_awaited_once_with(request, require_auth=False)

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_unauthorized_propagates(self, handler):
        """UnauthorizedError from get_auth_context propagates."""
        request = _make_request()

        with patch(
            "aragora.server.handlers.secure.get_auth_context",
            new_callable=AsyncMock,
            side_effect=UnauthorizedError("bad token"),
        ):
            with pytest.raises(UnauthorizedError, match="bad token"):
                await handler.get_auth_context(request, require_auth=True)

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_returns_result_from_underlying(self, handler):
        """Return value is exactly what the underlying function returns."""
        expected = _make_auth_context(user_id="specific-user")
        request = _make_request()

        with patch(
            "aragora.server.handlers.secure.get_auth_context",
            new_callable=AsyncMock,
            return_value=expected,
        ):
            result = await handler.get_auth_context(request, require_auth=False)

        assert result.user_id == "specific-user"


# ============================================================================
# check_permission
# ============================================================================


class TestCheckPermission:
    """Tests for SecureHandler.check_permission."""

    def test_permission_granted(self, handler, auth_ctx):
        """Returns True when permission is granted."""
        mock_decision = MagicMock()
        mock_decision.allowed = True

        mock_checker = MagicMock()
        mock_checker.check_permission.return_value = mock_decision

        with patch(
            f"{_RBAC_CHECKER_MOD}.get_permission_checker",
            return_value=mock_checker,
        ), patch(f"{_METRICS_MOD}.record_rbac_decision") as mock_record:
            result = handler.check_permission(auth_ctx, "items.read")

        assert result is True
        mock_checker.check_permission.assert_called_once_with(auth_ctx, "items.read", None)
        mock_record.assert_called_once_with("items.read", True)

    def test_permission_denied_raises_forbidden(self, handler, auth_ctx):
        """Raises ForbiddenError when permission is denied."""
        mock_decision = MagicMock()
        mock_decision.allowed = False

        mock_checker = MagicMock()
        mock_checker.check_permission.return_value = mock_decision

        with patch(
            f"{_RBAC_CHECKER_MOD}.get_permission_checker",
            return_value=mock_checker,
        ), patch(f"{_METRICS_MOD}.record_rbac_decision"):
            with pytest.raises(ForbiddenError) as exc_info:
                handler.check_permission(auth_ctx, "items.delete")

        assert exc_info.value.permission == "items.delete"

    def test_permission_with_resource_id(self, handler, auth_ctx):
        """Passes resource_id to checker."""
        mock_decision = MagicMock()
        mock_decision.allowed = True

        mock_checker = MagicMock()
        mock_checker.check_permission.return_value = mock_decision

        with patch(
            f"{_RBAC_CHECKER_MOD}.get_permission_checker",
            return_value=mock_checker,
        ), patch(f"{_METRICS_MOD}.record_rbac_decision"):
            handler.check_permission(auth_ctx, "items.read", resource_id="res-123")

        mock_checker.check_permission.assert_called_once_with(auth_ctx, "items.read", "res-123")

    def test_records_rbac_decision_on_deny(self, handler, auth_ctx):
        """Records RBAC decision metric even when denied."""
        mock_decision = MagicMock()
        mock_decision.allowed = False

        mock_checker = MagicMock()
        mock_checker.check_permission.return_value = mock_decision

        with patch(
            f"{_RBAC_CHECKER_MOD}.get_permission_checker",
            return_value=mock_checker,
        ), patch(f"{_METRICS_MOD}.record_rbac_decision") as mock_record:
            with pytest.raises(ForbiddenError):
                handler.check_permission(auth_ctx, "secret.read")

        mock_record.assert_called_once_with("secret.read", False)

    def test_permission_granted_with_none_resource_id(self, handler, auth_ctx):
        """None resource_id is properly passed to checker."""
        mock_decision = MagicMock()
        mock_decision.allowed = True

        mock_checker = MagicMock()
        mock_checker.check_permission.return_value = mock_decision

        with patch(
            f"{_RBAC_CHECKER_MOD}.get_permission_checker",
            return_value=mock_checker,
        ), patch(f"{_METRICS_MOD}.record_rbac_decision"):
            handler.check_permission(auth_ctx, "items.list", resource_id=None)

        mock_checker.check_permission.assert_called_once_with(auth_ctx, "items.list", None)


# ============================================================================
# audit_action
# ============================================================================


class TestAuditAction:
    """Tests for SecureHandler.audit_action."""

    @pytest.mark.asyncio
    async def test_audit_basic(self, handler, auth_ctx):
        """Basic audit action logs to immutable log."""
        mock_log = AsyncMock()

        with patch(f"{_AUDIT_LOG_MOD}.get_audit_log", return_value=mock_log):
            await handler.audit_action(
                auth_ctx,
                action="create",
                resource_id="item-42",
            )

        mock_log.append.assert_awaited_once()
        call_kwargs = mock_log.append.call_args[1]
        assert call_kwargs["event_type"] == "test_resource.create"
        assert call_kwargs["actor"] == "user-1"
        assert call_kwargs["actor_type"] == "user"
        assert call_kwargs["resource_type"] == "test_resource"
        assert call_kwargs["resource_id"] == "item-42"
        assert call_kwargs["action"] == "create"
        assert call_kwargs["workspace_id"] == "ws-1"

    @pytest.mark.asyncio
    async def test_audit_custom_resource_type(self, handler, auth_ctx):
        """Override resource_type in audit_action."""
        mock_log = AsyncMock()

        with patch(f"{_AUDIT_LOG_MOD}.get_audit_log", return_value=mock_log):
            await handler.audit_action(
                auth_ctx,
                action="delete",
                resource_id="doc-1",
                resource_type="document",
            )

        call_kwargs = mock_log.append.call_args[1]
        assert call_kwargs["event_type"] == "document.delete"
        assert call_kwargs["resource_type"] == "document"

    @pytest.mark.asyncio
    async def test_audit_with_details(self, handler, auth_ctx):
        """Passes details dict to audit log."""
        mock_log = AsyncMock()

        with patch(f"{_AUDIT_LOG_MOD}.get_audit_log", return_value=mock_log):
            await handler.audit_action(
                auth_ctx,
                action="update",
                resource_id="item-5",
                details={"changed_fields": ["name", "price"]},
            )

        call_kwargs = mock_log.append.call_args[1]
        assert call_kwargs["details"] == {"changed_fields": ["name", "price"]}

    @pytest.mark.asyncio
    async def test_audit_extracts_ip_from_x_forwarded_for(self, handler, auth_ctx):
        """Extracts client IP from X-Forwarded-For header."""
        mock_log = AsyncMock()
        request = _make_request(
            headers={
                "X-Forwarded-For": "203.0.113.50, 70.41.3.18",
                "User-Agent": "TestBot/1.0",
            }
        )

        with patch(f"{_AUDIT_LOG_MOD}.get_audit_log", return_value=mock_log):
            await handler.audit_action(
                auth_ctx,
                action="read",
                resource_id="item-1",
                request=request,
            )

        call_kwargs = mock_log.append.call_args[1]
        assert call_kwargs["ip_address"] == "203.0.113.50"
        assert call_kwargs["user_agent"] == "TestBot/1.0"

    @pytest.mark.asyncio
    async def test_audit_falls_back_to_remote_attr(self, handler, auth_ctx):
        """Falls back to request.remote when X-Forwarded-For is empty."""
        mock_log = AsyncMock()
        request = _make_request(
            headers={"X-Forwarded-For": "", "User-Agent": "CLI/2.0"},
            remote="192.168.1.100",
        )

        with patch(f"{_AUDIT_LOG_MOD}.get_audit_log", return_value=mock_log):
            await handler.audit_action(
                auth_ctx,
                action="read",
                resource_id="item-2",
                request=request,
            )

        call_kwargs = mock_log.append.call_args[1]
        assert call_kwargs["ip_address"] == "192.168.1.100"

    @pytest.mark.asyncio
    async def test_audit_no_request(self, handler, auth_ctx):
        """Without request, ip_address and user_agent are None."""
        mock_log = AsyncMock()

        with patch(f"{_AUDIT_LOG_MOD}.get_audit_log", return_value=mock_log):
            await handler.audit_action(
                auth_ctx,
                action="list",
                resource_id="all",
            )

        call_kwargs = mock_log.append.call_args[1]
        assert call_kwargs["ip_address"] is None
        assert call_kwargs["user_agent"] is None

    @pytest.mark.asyncio
    async def test_audit_request_without_headers(self, handler, auth_ctx):
        """Request without headers attr does not crash."""
        mock_log = AsyncMock()
        request = MagicMock(spec=[])  # no attributes at all

        with patch(f"{_AUDIT_LOG_MOD}.get_audit_log", return_value=mock_log):
            await handler.audit_action(
                auth_ctx,
                action="read",
                resource_id="item-3",
                request=request,
            )

        call_kwargs = mock_log.append.call_args[1]
        assert call_kwargs["ip_address"] is None

    @pytest.mark.asyncio
    async def test_audit_empty_details_default(self, handler, auth_ctx):
        """Default details is empty dict."""
        mock_log = AsyncMock()

        with patch(f"{_AUDIT_LOG_MOD}.get_audit_log", return_value=mock_log):
            await handler.audit_action(
                auth_ctx,
                action="read",
                resource_id="item-4",
            )

        call_kwargs = mock_log.append.call_args[1]
        assert call_kwargs["details"] == {}

    @pytest.mark.asyncio
    async def test_audit_user_agent_extracted(self, handler, auth_ctx):
        """User-Agent header is extracted and logged."""
        mock_log = AsyncMock()
        request = _make_request(
            headers={
                "X-Forwarded-For": "1.2.3.4",
                "User-Agent": "Mozilla/5.0 TestBrowser",
            },
        )

        with patch(f"{_AUDIT_LOG_MOD}.get_audit_log", return_value=mock_log):
            await handler.audit_action(
                auth_ctx,
                action="access",
                resource_id="res-1",
                request=request,
            )

        call_kwargs = mock_log.append.call_args[1]
        assert call_kwargs["user_agent"] == "Mozilla/5.0 TestBrowser"


# ============================================================================
# encrypt_response_fields / decrypt_request_fields
# ============================================================================


class TestEncryptDecrypt:
    """Tests for encrypt and decrypt field helpers."""

    def test_encrypt_response_fields(self, handler):
        """Delegates to storage.encrypted_fields.encrypt_sensitive."""
        data = {"ssn": "123-45-6789", "name": "Alice"}

        with patch(
            f"{_ENCRYPTED_MOD}.encrypt_sensitive",
            return_value={"ssn": "ENC(xxx)", "name": "Alice"},
        ) as mock_enc:
            result = handler.encrypt_response_fields(data)

        mock_enc.assert_called_once_with(data)
        assert result["ssn"] == "ENC(xxx)"
        assert result["name"] == "Alice"

    def test_decrypt_request_fields(self, handler):
        """Delegates to storage.encrypted_fields.decrypt_sensitive."""
        data = {"ssn": "ENC(xxx)", "name": "Alice"}

        with patch(
            f"{_ENCRYPTED_MOD}.decrypt_sensitive",
            return_value={"ssn": "123-45-6789", "name": "Alice"},
        ) as mock_dec:
            result = handler.decrypt_request_fields(data)

        mock_dec.assert_called_once_with(data)
        assert result["ssn"] == "123-45-6789"

    def test_encrypt_with_explicit_fields(self, handler):
        """Fields parameter is accepted (though underlying impl auto-detects)."""
        data = {"token": "secret"}

        with patch(
            f"{_ENCRYPTED_MOD}.encrypt_sensitive",
            return_value={"token": "ENC(y)"},
        ):
            result = handler.encrypt_response_fields(data, fields=["token"])

        assert result["token"] == "ENC(y)"

    def test_decrypt_with_explicit_fields(self, handler):
        """Fields parameter is accepted (though underlying impl auto-detects)."""
        data = {"token": "ENC(y)"}

        with patch(
            f"{_ENCRYPTED_MOD}.decrypt_sensitive",
            return_value={"token": "secret"},
        ):
            result = handler.decrypt_request_fields(data, fields=["token"])

        assert result["token"] == "secret"

    def test_encrypt_empty_dict(self, handler):
        """Encrypting empty dict works."""
        with patch(f"{_ENCRYPTED_MOD}.encrypt_sensitive", return_value={}):
            result = handler.encrypt_response_fields({})

        assert result == {}

    def test_decrypt_empty_dict(self, handler):
        """Decrypting empty dict works."""
        with patch(f"{_ENCRYPTED_MOD}.decrypt_sensitive", return_value={}):
            result = handler.decrypt_request_fields({})

        assert result == {}


# ============================================================================
# handle_security_error
# ============================================================================


class TestHandleSecurityError:
    """Tests for SecureHandler.handle_security_error."""

    def test_unauthorized_error(self, handler):
        """UnauthorizedError returns 401."""
        err = UnauthorizedError("Token expired")

        with _patch_metrics() as mocks:
            result = handler.handle_security_error(err)

        assert result.status_code == 401
        body = parse_body(result)
        assert "Authentication required" in body.get("error", "")
        mocks["record_auth_failure"].assert_called_once_with("jwt", "invalid_token")

    def test_forbidden_error_with_permission(self, handler):
        """ForbiddenError returns 403 with permission detail."""
        err = ForbiddenError("Access denied", permission="items.delete")

        with _patch_metrics() as mocks:
            result = handler.handle_security_error(err)

        assert result.status_code == 403
        body = parse_body(result)
        assert "items.delete" in body.get("error", "")
        mocks["record_blocked_request"].assert_called_once_with("permission_denied", "user")

    def test_forbidden_error_no_permission(self, handler):
        """ForbiddenError without permission shows generic message."""
        err = ForbiddenError("nope", permission=None)

        with _patch_metrics():
            result = handler.handle_security_error(err)

        assert result.status_code == 403
        body = parse_body(result)
        assert "insufficient permissions" in body.get("error", "")

    def test_permission_denied_error(self, handler):
        """PermissionDeniedError returns 403."""
        err = PermissionDeniedError("No access", decision=None)

        with _patch_metrics() as mocks:
            result = handler.handle_security_error(err)

        assert result.status_code == 403
        body = parse_body(result)
        assert "Permission denied" in body.get("error", "")
        mocks["record_blocked_request"].assert_called_once_with("rbac_denied", "user")

    def test_role_required_error(self, handler):
        """RoleRequiredError returns 403 with role info."""
        err = RoleRequiredError(
            "Missing role",
            required_roles={"admin"},
            actual_roles={"member"},
        )

        with _patch_metrics() as mocks:
            result = handler.handle_security_error(err)

        assert result.status_code == 403
        body = parse_body(result)
        assert "admin" in body.get("error", "")
        mocks["record_blocked_request"].assert_called_once_with("role_required", "user")

    def test_unexpected_security_error(self, handler):
        """Unknown exception returns 500."""
        err = RuntimeError("something weird")

        with _patch_metrics():
            result = handler.handle_security_error(err)

        assert result.status_code == 500
        body = parse_body(result)
        assert "Security error" in body.get("error", "")

    def test_handle_security_error_with_request(self, handler):
        """Request parameter is accepted (for future context extraction)."""
        err = UnauthorizedError("no token")
        request = _make_request()

        with _patch_metrics():
            result = handler.handle_security_error(err, request=request)

        assert result.status_code == 401

    def test_generic_exception_returns_500(self, handler):
        """A plain Exception returns 500."""
        err = Exception("something")

        with _patch_metrics():
            result = handler.handle_security_error(err)

        assert result.status_code == 500


# ============================================================================
# secure_endpoint decorator
# ============================================================================


class TestSecureEndpointDecorator:
    """Tests for the @secure_endpoint decorator."""

    @pytest.mark.asyncio
    async def test_basic_authenticated_call(self, handler, auth_ctx):
        """Decorated method receives auth_context and returns result."""

        @secure_endpoint()
        async def handle_get(self, request, auth_context, **kwargs):
            return HandlerResult(
                status_code=200,
                content_type="application/json",
                body=json.dumps({"user": auth_context.user_id}).encode(),
            )

        request = _make_request()

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), _patch_metrics():
            result = await handle_get(handler, request)

        assert result.status_code == 200
        body = parse_body(result)
        assert body["user"] == "user-1"

    @pytest.mark.asyncio
    async def test_permission_check(self, handler, auth_ctx):
        """Permission is checked when specified."""

        @secure_endpoint(permission="items.read")
        async def handle_get(self, request, auth_context, **kwargs):
            return HandlerResult(
                status_code=200,
                content_type="application/json",
                body=b"{}",
            )

        request = _make_request()

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), patch.object(
            handler, "check_permission", return_value=True
        ) as mock_perm, _patch_metrics():
            result = await handle_get(handler, request)

        assert result.status_code == 200
        mock_perm.assert_called_once_with(auth_ctx, "items.read", None)

    @pytest.mark.asyncio
    async def test_permission_with_resource_id_param(self, handler, auth_ctx):
        """Resource ID is extracted from kwargs when resource_id_param is set."""

        @secure_endpoint(permission="items.read", resource_id_param="item_id")
        async def handle_get(self, request, auth_context, **kwargs):
            return HandlerResult(
                status_code=200,
                content_type="application/json",
                body=b"{}",
            )

        request = _make_request()

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), patch.object(
            handler, "check_permission", return_value=True
        ) as mock_perm, _patch_metrics():
            result = await handle_get(handler, request, item_id="abc-123")

        mock_perm.assert_called_once_with(auth_ctx, "items.read", "abc-123")

    @pytest.mark.asyncio
    async def test_resource_id_param_missing(self, handler, auth_ctx):
        """Missing resource_id_param in kwargs passes None."""

        @secure_endpoint(permission="items.read", resource_id_param="item_id")
        async def handle_get(self, request, auth_context, **kwargs):
            return HandlerResult(
                status_code=200,
                content_type="application/json",
                body=b"{}",
            )

        request = _make_request()

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), patch.object(
            handler, "check_permission", return_value=True
        ) as mock_perm, _patch_metrics():
            result = await handle_get(handler, request)

        mock_perm.assert_called_once_with(auth_ctx, "items.read", None)

    @pytest.mark.asyncio
    async def test_audit_enabled(self, handler, auth_ctx):
        """Audit logging happens when audit=True."""

        @secure_endpoint(audit=True, resource_id_param="item_id")
        async def handle_post(self, request, auth_context, **kwargs):
            return HandlerResult(
                status_code=201,
                content_type="application/json",
                body=b'{"created":true}',
            )

        request = _make_request()

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), patch.object(
            handler, "audit_action", new_callable=AsyncMock
        ) as mock_audit, _patch_metrics():
            result = await handle_post(handler, request, item_id="item-99")

        assert result.status_code == 201
        mock_audit.assert_awaited_once()
        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["action"] == "post"  # derived from handle_post -> "post"
        assert call_kwargs["resource_id"] == "item-99"

    @pytest.mark.asyncio
    async def test_audit_custom_action_name(self, handler, auth_ctx):
        """Custom audit_action name overrides auto-derived name."""

        @secure_endpoint(audit=True, audit_action="provision")
        async def handle_post(self, request, auth_context, **kwargs):
            return HandlerResult(
                status_code=201,
                content_type="application/json",
                body=b"{}",
            )

        request = _make_request()

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), patch.object(
            handler, "audit_action", new_callable=AsyncMock
        ) as mock_audit, _patch_metrics():
            await handle_post(handler, request)

        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["action"] == "provision"

    @pytest.mark.asyncio
    async def test_audit_default_resource_id(self, handler, auth_ctx):
        """Without resource_id_param, defaults to 'unknown'."""

        @secure_endpoint(audit=True)
        async def handle_post(self, request, auth_context, **kwargs):
            return HandlerResult(
                status_code=200,
                content_type="application/json",
                body=b"{}",
            )

        request = _make_request()

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), patch.object(
            handler, "audit_action", new_callable=AsyncMock
        ) as mock_audit, _patch_metrics():
            await handle_post(handler, request)

        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["resource_id"] == "unknown"

    @pytest.mark.asyncio
    async def test_no_audit_when_disabled(self, handler, auth_ctx):
        """No audit_action call when audit=False (default)."""

        @secure_endpoint()
        async def handle_get(self, request, auth_context, **kwargs):
            return HandlerResult(
                status_code=200,
                content_type="application/json",
                body=b"{}",
            )

        request = _make_request()

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), patch.object(
            handler, "audit_action", new_callable=AsyncMock
        ) as mock_audit, _patch_metrics():
            await handle_get(handler, request)

        mock_audit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unauthorized_returns_401(self, handler):
        """UnauthorizedError in auth step returns 401."""

        @secure_endpoint(require_auth=True)
        async def handle_get(self, request, auth_context, **kwargs):
            return HandlerResult(
                status_code=200,
                content_type="application/json",
                body=b"{}",
            )

        request = _make_request()

        with patch.object(
            handler,
            "get_auth_context",
            new_callable=AsyncMock,
            side_effect=UnauthorizedError("No token"),
        ), _patch_metrics():
            result = await handle_get(handler, request)

        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_forbidden_returns_403(self, handler, auth_ctx):
        """ForbiddenError in permission step returns 403."""

        @secure_endpoint(permission="secret.access")
        async def handle_get(self, request, auth_context, **kwargs):
            return HandlerResult(
                status_code=200,
                content_type="application/json",
                body=b"{}",
            )

        request = _make_request()

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), patch.object(
            handler,
            "check_permission",
            side_effect=ForbiddenError("denied", permission="secret.access"),
        ), _patch_metrics():
            result = await handle_get(handler, request)

        assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_permission_denied_error_returns_403(self, handler, auth_ctx):
        """PermissionDeniedError returns 403."""

        @secure_endpoint(permission="admin.only")
        async def handle_get(self, request, auth_context, **kwargs):
            return HandlerResult(
                status_code=200,
                content_type="application/json",
                body=b"{}",
            )

        request = _make_request()

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), patch.object(
            handler,
            "check_permission",
            side_effect=PermissionDeniedError("nope", decision=None),
        ), _patch_metrics():
            result = await handle_get(handler, request)

        assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_role_required_error_returns_403(self, handler, auth_ctx):
        """RoleRequiredError returns 403."""

        @secure_endpoint(permission="admin.access")
        async def handle_get(self, request, auth_context, **kwargs):
            return HandlerResult(
                status_code=200,
                content_type="application/json",
                body=b"{}",
            )

        request = _make_request()

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), patch.object(
            handler,
            "check_permission",
            side_effect=RoleRequiredError(
                "Need admin", required_roles={"admin"}, actual_roles={"member"}
            ),
        ), _patch_metrics():
            result = await handle_get(handler, request)

        assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_value_error_propagates(self, handler, auth_ctx):
        """ValueError inside handler propagates (not caught as security error)."""

        @secure_endpoint()
        async def handle_get(self, request, auth_context, **kwargs):
            raise ValueError("bad input")

        request = _make_request()

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), _patch_metrics():
            with pytest.raises(ValueError, match="bad input"):
                await handle_get(handler, request)

    @pytest.mark.asyncio
    async def test_type_error_propagates(self, handler, auth_ctx):
        """TypeError inside handler propagates."""

        @secure_endpoint()
        async def handle_get(self, request, auth_context, **kwargs):
            raise TypeError("wrong type")

        request = _make_request()

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), _patch_metrics():
            with pytest.raises(TypeError, match="wrong type"):
                await handle_get(handler, request)

    @pytest.mark.asyncio
    async def test_key_error_propagates(self, handler, auth_ctx):
        """KeyError inside handler propagates."""

        @secure_endpoint()
        async def handle_get(self, request, auth_context, **kwargs):
            raise KeyError("missing_key")

        request = _make_request()

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), _patch_metrics():
            with pytest.raises(KeyError):
                await handle_get(handler, request)

    @pytest.mark.asyncio
    async def test_records_auth_attempt_authenticated(self, handler, auth_ctx):
        """Records auth attempt with success=True for authenticated user."""

        @secure_endpoint()
        async def handle_get(self, request, auth_context, **kwargs):
            return HandlerResult(
                status_code=200,
                content_type="application/json",
                body=b"{}",
            )

        request = _make_request()

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), _patch_metrics() as mocks:
            await handle_get(handler, request)

        mocks["record_auth_attempt"].assert_called_once_with("jwt", success=True)

    @pytest.mark.asyncio
    async def test_records_auth_attempt_anonymous(self, handler):
        """Records auth attempt with success=False for anonymous user."""
        anon_ctx = _make_auth_context(user_id="anonymous")

        @secure_endpoint(require_auth=False)
        async def handle_get(self, request, auth_context, **kwargs):
            return HandlerResult(
                status_code=200,
                content_type="application/json",
                body=b"{}",
            )

        request = _make_request()

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=anon_ctx
        ), _patch_metrics() as mocks:
            await handle_get(handler, request)

        mocks["record_auth_attempt"].assert_called_once_with("jwt", success=False)

    @pytest.mark.asyncio
    async def test_no_permission_check_when_none(self, handler, auth_ctx):
        """No permission check when permission is not specified."""

        @secure_endpoint(permission=None)
        async def handle_get(self, request, auth_context, **kwargs):
            return HandlerResult(
                status_code=200,
                content_type="application/json",
                body=b"{}",
            )

        request = _make_request()

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), patch.object(
            handler, "check_permission"
        ) as mock_check, _patch_metrics():
            await handle_get(handler, request)

        mock_check.assert_not_called()

    @pytest.mark.asyncio
    async def test_functools_wraps_preserves_name(self):
        """Decorated function preserves __name__."""

        @secure_endpoint()
        async def handle_my_fancy_endpoint(self, request, auth_context, **kwargs):
            pass

        assert handle_my_fancy_endpoint.__name__ == "handle_my_fancy_endpoint"

    @pytest.mark.asyncio
    async def test_runtime_error_propagates(self, handler, auth_ctx):
        """RuntimeError inside handler propagates."""

        @secure_endpoint()
        async def handle_get(self, request, auth_context, **kwargs):
            raise RuntimeError("server crashed")

        request = _make_request()

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), _patch_metrics():
            with pytest.raises(RuntimeError, match="server crashed"):
                await handle_get(handler, request)

    @pytest.mark.asyncio
    async def test_os_error_propagates(self, handler, auth_ctx):
        """OSError inside handler propagates."""

        @secure_endpoint()
        async def handle_get(self, request, auth_context, **kwargs):
            raise OSError("disk full")

        request = _make_request()

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), _patch_metrics():
            with pytest.raises(OSError, match="disk full"):
                await handle_get(handler, request)

    @pytest.mark.asyncio
    async def test_connection_error_propagates(self, handler, auth_ctx):
        """ConnectionError inside handler propagates."""

        @secure_endpoint()
        async def handle_get(self, request, auth_context, **kwargs):
            raise ConnectionError("db gone")

        request = _make_request()

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), _patch_metrics():
            with pytest.raises(ConnectionError, match="db gone"):
                await handle_get(handler, request)

    @pytest.mark.asyncio
    async def test_timeout_error_propagates(self, handler, auth_ctx):
        """TimeoutError inside handler propagates."""

        @secure_endpoint()
        async def handle_get(self, request, auth_context, **kwargs):
            raise TimeoutError("too slow")

        request = _make_request()

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), _patch_metrics():
            with pytest.raises(TimeoutError, match="too slow"):
                await handle_get(handler, request)

    @pytest.mark.asyncio
    async def test_attribute_error_propagates(self, handler, auth_ctx):
        """AttributeError inside handler propagates."""

        @secure_endpoint()
        async def handle_get(self, request, auth_context, **kwargs):
            raise AttributeError("no such attr")

        request = _make_request()

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), _patch_metrics():
            with pytest.raises(AttributeError, match="no such attr"):
                await handle_get(handler, request)

    @pytest.mark.asyncio
    async def test_audit_includes_duration(self, handler, auth_ctx):
        """Audit log includes duration_ms in details."""

        @secure_endpoint(audit=True)
        async def handle_post(self, request, auth_context, **kwargs):
            return HandlerResult(
                status_code=201,
                content_type="application/json",
                body=b"{}",
            )

        request = _make_request()

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), patch.object(
            handler, "audit_action", new_callable=AsyncMock
        ) as mock_audit, _patch_metrics():
            await handle_post(handler, request)

        call_kwargs = mock_audit.call_args[1]
        assert "duration_ms" in call_kwargs["details"]
        assert isinstance(call_kwargs["details"]["duration_ms"], float)

    @pytest.mark.asyncio
    async def test_audit_receives_request(self, handler, auth_ctx):
        """Audit action receives the original request."""

        @secure_endpoint(audit=True)
        async def handle_post(self, request, auth_context, **kwargs):
            return HandlerResult(
                status_code=200,
                content_type="application/json",
                body=b"{}",
            )

        request = _make_request()

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), patch.object(
            handler, "audit_action", new_callable=AsyncMock
        ) as mock_audit, _patch_metrics():
            await handle_post(handler, request)

        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["request"] is request

    @pytest.mark.asyncio
    async def test_resource_id_param_numeric(self, handler, auth_ctx):
        """Numeric resource_id is converted to string."""

        @secure_endpoint(permission="items.read", resource_id_param="item_id")
        async def handle_get(self, request, auth_context, **kwargs):
            return HandlerResult(
                status_code=200,
                content_type="application/json",
                body=b"{}",
            )

        request = _make_request()

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), patch.object(
            handler, "check_permission", return_value=True
        ) as mock_perm, _patch_metrics():
            await handle_get(handler, request, item_id=42)

        mock_perm.assert_called_once_with(auth_ctx, "items.read", "42")


# ============================================================================
# audit_sensitive_access decorator
# ============================================================================


class TestAuditSensitiveAccessDecorator:
    """Tests for the @audit_sensitive_access decorator."""

    @pytest.mark.asyncio
    async def test_records_metric(self, handler, auth_ctx):
        """Records secret_access metric."""

        @audit_sensitive_access("api_key", "read")
        async def get_api_key(self, request, auth_context):
            return {"key": "sk-xxx"}

        request = _make_request()

        with patch(
            f"{_METRICS_MOD}.record_secret_access",
        ) as mock_record, patch(
            f"{_SECURITY_AUDIT_MOD}.audit_secret_access",
            new_callable=AsyncMock,
        ):
            result = await get_api_key(handler, request, auth_ctx)

        assert result["key"] == "sk-xxx"
        mock_record.assert_called_once_with("api_key", "read")

    @pytest.mark.asyncio
    async def test_logs_audit_trail(self, handler, auth_ctx):
        """Logs to audit_secret_access."""

        @audit_sensitive_access("token", "rotate")
        async def rotate_token(self, request, auth_context):
            return {"new_token": "abc"}

        request = _make_request()

        with patch(
            f"{_METRICS_MOD}.record_secret_access",
        ), patch(
            f"{_SECURITY_AUDIT_MOD}.audit_secret_access",
            new_callable=AsyncMock,
        ) as mock_audit:
            await rotate_token(handler, request, auth_ctx)

        mock_audit.assert_awaited_once_with(
            actor="user-1",
            secret_type="token",
            store="test_resource",
            operation="rotate",
            workspace_id="ws-1",
        )

    @pytest.mark.asyncio
    async def test_preserves_function_name(self):
        """Decorated function preserves __name__."""

        @audit_sensitive_access("token", "read")
        async def my_sensitive_function(self, request, auth_context):
            pass

        assert my_sensitive_function.__name__ == "my_sensitive_function"

    @pytest.mark.asyncio
    async def test_passes_through_kwargs(self, handler, auth_ctx):
        """Extra kwargs are passed to the wrapped function."""

        @audit_sensitive_access("key", "list")
        async def list_keys(self, request, auth_context, prefix=None):
            return {"prefix": prefix}

        request = _make_request()

        with patch(
            f"{_METRICS_MOD}.record_secret_access",
        ), patch(
            f"{_SECURITY_AUDIT_MOD}.audit_secret_access",
            new_callable=AsyncMock,
        ):
            result = await list_keys(handler, request, auth_ctx, prefix="sk-")

        assert result["prefix"] == "sk-"

    @pytest.mark.asyncio
    async def test_default_action(self, handler, auth_ctx):
        """Default action is 'access'."""

        @audit_sensitive_access("credential")
        async def get_credential(self, request, auth_context):
            return {}

        request = _make_request()

        with patch(
            f"{_METRICS_MOD}.record_secret_access",
        ) as mock_record, patch(
            f"{_SECURITY_AUDIT_MOD}.audit_secret_access",
            new_callable=AsyncMock,
        ) as mock_audit:
            await get_credential(handler, request, auth_ctx)

        mock_record.assert_called_once_with("credential", "access")
        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["operation"] == "access"

    @pytest.mark.asyncio
    async def test_uses_handler_resource_type(self, handler, auth_ctx):
        """audit_sensitive_access uses self.RESOURCE_TYPE as store."""

        @audit_sensitive_access("secret", "view")
        async def view_secret(self, request, auth_context):
            return {"seen": True}

        request = _make_request()

        with patch(
            f"{_METRICS_MOD}.record_secret_access",
        ), patch(
            f"{_SECURITY_AUDIT_MOD}.audit_secret_access",
            new_callable=AsyncMock,
        ) as mock_audit:
            await view_secret(handler, request, auth_ctx)

        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["store"] == "test_resource"


# ============================================================================
# Integration-style tests: secure_endpoint + check_permission full flow
# ============================================================================


class TestSecureEndpointIntegration:
    """Integration tests combining secure_endpoint with real check_permission calls."""

    @pytest.mark.asyncio
    async def test_full_flow_success(self, handler, auth_ctx):
        """Full secure endpoint flow: auth -> permission -> handler -> audit."""
        call_log = []

        @secure_endpoint(permission="items.write", audit=True, resource_id_param="id")
        async def handle_put(self, request, auth_context, **kwargs):
            call_log.append("handler_called")
            return HandlerResult(
                status_code=200,
                content_type="application/json",
                body=json.dumps({"updated": True}).encode(),
            )

        request = _make_request()

        mock_decision = MagicMock()
        mock_decision.allowed = True

        mock_checker = MagicMock()
        mock_checker.check_permission.return_value = mock_decision

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), _patch_metrics(), patch(
            f"{_RBAC_CHECKER_MOD}.get_permission_checker",
            return_value=mock_checker,
        ), patch.object(
            handler, "audit_action", new_callable=AsyncMock
        ):
            result = await handle_put(handler, request, id="item-100")

        assert result.status_code == 200
        assert "handler_called" in call_log

    @pytest.mark.asyncio
    async def test_full_flow_denied(self, handler, auth_ctx):
        """Full flow where permission is denied returns 403 without calling handler."""
        handler_called = []

        @secure_endpoint(permission="items.admin")
        async def handle_delete(self, request, auth_context, **kwargs):
            handler_called.append(True)
            return HandlerResult(
                status_code=200,
                content_type="application/json",
                body=b"{}",
            )

        request = _make_request()

        mock_decision = MagicMock()
        mock_decision.allowed = False

        mock_checker = MagicMock()
        mock_checker.check_permission.return_value = mock_decision

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), _patch_metrics(), patch(
            f"{_RBAC_CHECKER_MOD}.get_permission_checker",
            return_value=mock_checker,
        ):
            result = await handle_delete(handler, request)

        assert result.status_code == 403
        assert handler_called == []  # handler body never reached

    @pytest.mark.asyncio
    async def test_auth_failure_skips_permission_and_handler(self, handler):
        """When auth fails, neither permission check nor handler body runs."""
        handler_called = []
        perm_checked = []

        @secure_endpoint(permission="items.read")
        async def handle_get(self, request, auth_context, **kwargs):
            handler_called.append(True)
            return HandlerResult(
                status_code=200,
                content_type="application/json",
                body=b"{}",
            )

        request = _make_request()

        with patch.object(
            handler,
            "get_auth_context",
            new_callable=AsyncMock,
            side_effect=UnauthorizedError("expired"),
        ), _patch_metrics():
            result = await handle_get(handler, request)

        assert result.status_code == 401
        assert handler_called == []


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    """Edge cases and corner cases."""

    def test_handler_inherits_from_base_handler(self, handler):
        """SecureHandler is a subclass of BaseHandler."""
        from aragora.server.handlers.base import BaseHandler

        assert isinstance(handler, BaseHandler)

    def test_all_exports(self):
        """__all__ contains expected exports."""
        from aragora.server.handlers import secure

        assert "SecureHandler" in secure.__all__
        assert "secure_endpoint" in secure.__all__
        assert "audit_sensitive_access" in secure.__all__

    def test_all_exports_length(self):
        """__all__ has exactly 3 exports."""
        from aragora.server.handlers import secure

        assert len(secure.__all__) == 3

    def test_forbidden_error_has_permission_attr(self):
        """ForbiddenError stores permission attribute."""
        err = ForbiddenError("no", permission="x.y")
        assert err.permission == "x.y"

    def test_forbidden_error_default_permission(self):
        """ForbiddenError default permission is None."""
        err = ForbiddenError("no")
        assert err.permission is None

    def test_unauthorized_error_has_message(self):
        """UnauthorizedError stores message attribute."""
        err = UnauthorizedError("expired")
        assert err.message == "expired"

    def test_unauthorized_error_default_message(self):
        """UnauthorizedError has default message."""
        err = UnauthorizedError()
        assert err.message == "Authentication required"

    def test_forbidden_error_message(self):
        """ForbiddenError default message is 'Access denied'."""
        err = ForbiddenError()
        assert err.message == "Access denied"

    @pytest.mark.asyncio
    async def test_audit_action_with_no_x_forwarded_for_header(self, handler, auth_ctx):
        """Request with headers but no X-Forwarded-For falls back to remote."""
        mock_log = AsyncMock()
        request = MagicMock()
        request.headers = {}

        with patch(f"{_AUDIT_LOG_MOD}.get_audit_log", return_value=mock_log):
            await handler.audit_action(
                auth_ctx,
                action="test",
                resource_id="r-1",
                request=request,
            )

        call_kwargs = mock_log.append.call_args[1]
        # MagicMock auto-creates .remote, so ip_address should be the mock remote
        assert call_kwargs["ip_address"] is not None

    def test_check_permission_logs_warning_on_deny(self, handler, auth_ctx):
        """Permission denial is logged as a warning."""
        mock_decision = MagicMock()
        mock_decision.allowed = False

        mock_checker = MagicMock()
        mock_checker.check_permission.return_value = mock_decision

        with patch(
            f"{_RBAC_CHECKER_MOD}.get_permission_checker",
            return_value=mock_checker,
        ), patch(
            f"{_METRICS_MOD}.record_rbac_decision",
        ), patch(
            "aragora.server.handlers.secure.logger",
        ) as mock_logger:
            with pytest.raises(ForbiddenError):
                handler.check_permission(auth_ctx, "sensitive.data")

        mock_logger.warning.assert_called_once()
        assert "sensitive.data" in mock_logger.warning.call_args[0][1]

    def test_handle_security_error_logs_unknown(self, handler):
        """Unknown error type is logged at error level."""
        err = Exception("mysterious")

        with _patch_metrics(), patch(
            "aragora.server.handlers.secure.logger",
        ) as mock_logger:
            result = handler.handle_security_error(err)

        assert result.status_code == 500
        mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_secure_endpoint_passes_extra_args(self, handler, auth_ctx):
        """Extra keyword args pass through the decorator."""
        received_kwargs = {}

        @secure_endpoint()
        async def handle_get(self, request, auth_context, **kwargs):
            received_kwargs.update(kwargs)
            return HandlerResult(
                status_code=200,
                content_type="application/json",
                body=b"{}",
            )

        request = _make_request()

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), _patch_metrics():
            await handle_get(handler, request, extra_param="hello")

        assert received_kwargs["extra_param"] == "hello"

    @pytest.mark.asyncio
    async def test_secure_endpoint_audit_action_derived_from_method_name(
        self, handler, auth_ctx
    ):
        """Audit action is derived by stripping 'handle_' prefix."""

        @secure_endpoint(audit=True)
        async def handle_create_widget(self, request, auth_context, **kwargs):
            return HandlerResult(
                status_code=201,
                content_type="application/json",
                body=b"{}",
            )

        request = _make_request()

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), patch.object(
            handler, "audit_action", new_callable=AsyncMock
        ) as mock_audit, _patch_metrics():
            await handle_create_widget(handler, request)

        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["action"] == "create_widget"

    @pytest.mark.asyncio
    async def test_secure_endpoint_no_handle_prefix(self, handler, auth_ctx):
        """When method name has no 'handle_' prefix, full name is used."""

        @secure_endpoint(audit=True)
        async def process_data(self, request, auth_context, **kwargs):
            return HandlerResult(
                status_code=200,
                content_type="application/json",
                body=b"{}",
            )

        request = _make_request()

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_ctx
        ), patch.object(
            handler, "audit_action", new_callable=AsyncMock
        ) as mock_audit, _patch_metrics():
            await process_data(handler, request)

        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["action"] == "process_data"

    def test_permission_denied_error_stores_decision(self):
        """PermissionDeniedError stores the decision object."""
        decision = MagicMock()
        decision.permission_key = "foo.bar"
        decision.resource_id = "res-1"
        err = PermissionDeniedError("denied", decision=decision)
        assert err.permission_key == "foo.bar"
        assert err.resource_id == "res-1"

    def test_role_required_error_stores_roles(self):
        """RoleRequiredError stores required and actual roles."""
        err = RoleRequiredError(
            "Missing roles",
            required_roles={"admin", "superuser"},
            actual_roles={"viewer"},
        )
        assert err.required_roles == {"admin", "superuser"}
        assert err.actual_roles == {"viewer"}

    @pytest.mark.asyncio
    async def test_audit_action_request_none(self, handler, auth_ctx):
        """audit_action with request=None doesn't extract IP."""
        mock_log = AsyncMock()

        with patch(f"{_AUDIT_LOG_MOD}.get_audit_log", return_value=mock_log):
            await handler.audit_action(
                auth_ctx,
                action="noop",
                resource_id="x",
                request=None,
            )

        call_kwargs = mock_log.append.call_args[1]
        assert call_kwargs["ip_address"] is None
        assert call_kwargs["user_agent"] is None

    def test_secure_handler_is_importable(self):
        """SecureHandler can be imported from its module."""
        from aragora.server.handlers.secure import SecureHandler as SH

        assert SH is SecureHandler
