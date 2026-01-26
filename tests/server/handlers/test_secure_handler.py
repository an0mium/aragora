"""
Tests for SecureHandler and security decorators.

Covers:
- SecureHandler class methods
- @secure_endpoint decorator
- @audit_sensitive_access decorator
- Security error handling (401, 403 responses)
"""

from __future__ import annotations

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from aragora.server.handlers.secure import (
    SecureHandler,
    secure_endpoint,
    audit_sensitive_access,
    UnauthorizedError,
    ForbiddenError,
)
from aragora.server.handlers.base import HandlerResult, json_response, error_response
from aragora.rbac.models import AuthorizationContext
from aragora.rbac.decorators import PermissionDeniedError, RoleRequiredError


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def server_context():
    """Create a mock server context."""
    return {"config": {"debug": True}}


@pytest.fixture
def auth_context():
    """Create a test authorization context."""
    return AuthorizationContext(
        user_id="user-123",
        org_id="org-456",
        roles={"member", "analyst"},
        permissions={"debates.read", "debates.create"},
    )


@pytest.fixture
def anonymous_context():
    """Create an anonymous authorization context."""
    return AuthorizationContext(
        user_id="anonymous",
        org_id=None,
        roles=set(),
        permissions=set(),
    )


@pytest.fixture
def mock_request():
    """Create a mock HTTP request."""
    request = MagicMock()
    request.headers = {
        "Authorization": "Bearer test-token",
        "X-Workspace-ID": "ws-789",
        "X-Forwarded-For": "192.168.1.1, 10.0.0.1",
        "User-Agent": "TestClient/1.0",
    }
    request.remote = "127.0.0.1"
    request.method = "GET"
    return request


@pytest.fixture
def secure_handler(server_context):
    """Create a SecureHandler instance."""
    return SecureHandler(server_context)


# -----------------------------------------------------------------------------
# SecureHandler Class Tests
# -----------------------------------------------------------------------------


class TestSecureHandlerInitialization:
    """Tests for SecureHandler initialization."""

    def test_init_with_server_context(self, server_context):
        """SecureHandler initializes with server context."""
        handler = SecureHandler(server_context)
        assert handler.ctx == server_context
        assert handler._auth_context is None

    def test_default_method_permissions(self, secure_handler):
        """SecureHandler has default method permissions."""
        assert secure_handler.DEFAULT_METHOD_PERMISSIONS["GET"] is None
        assert secure_handler.DEFAULT_METHOD_PERMISSIONS["POST"] is None
        assert secure_handler.DEFAULT_METHOD_PERMISSIONS["DELETE"] is None

    def test_default_resource_type(self, secure_handler):
        """SecureHandler has default resource type."""
        assert secure_handler.RESOURCE_TYPE == "unknown"


class TestSecureHandlerGetAuthContext:
    """Tests for SecureHandler.get_auth_context() method."""

    @pytest.mark.asyncio
    async def test_get_auth_context_authenticated(self, secure_handler, mock_request, auth_context):
        """get_auth_context returns context for authenticated user."""
        # Patch at the module level where it's imported in secure.py
        with patch(
            "aragora.server.handlers.secure.get_auth_context",
            new_callable=AsyncMock,
            return_value=auth_context,
        ) as mock_get_auth:
            result = await secure_handler.get_auth_context(mock_request)
            assert result.user_id == "user-123"
            assert result.org_id == "org-456"
            mock_get_auth.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_auth_context_anonymous_allowed(
        self, secure_handler, mock_request, anonymous_context
    ):
        """get_auth_context returns anonymous context when auth not required."""
        with patch(
            "aragora.server.handlers.secure.get_auth_context",
            new_callable=AsyncMock,
            return_value=anonymous_context,
        ):
            result = await secure_handler.get_auth_context(mock_request, require_auth=False)
            assert result.user_id == "anonymous"

    @pytest.mark.asyncio
    async def test_get_auth_context_require_auth_raises(self, secure_handler, mock_request):
        """get_auth_context raises UnauthorizedError when required and not authenticated."""
        with patch(
            "aragora.server.handlers.secure.get_auth_context",
            new_callable=AsyncMock,
            side_effect=UnauthorizedError("Token required"),
        ):
            with pytest.raises(UnauthorizedError) as exc_info:
                await secure_handler.get_auth_context(mock_request, require_auth=True)
            assert "Token required" in str(exc_info.value)


class TestSecureHandlerCheckPermission:
    """Tests for SecureHandler.check_permission() method."""

    def test_check_permission_granted(self, secure_handler, auth_context):
        """check_permission returns True when permission granted."""
        mock_decision = MagicMock()
        mock_decision.allowed = True

        mock_checker = MagicMock()
        mock_checker.check_permission.return_value = mock_decision

        with patch(
            "aragora.rbac.checker.get_permission_checker",
            return_value=mock_checker,
        ):
            with patch("aragora.observability.metrics.security.record_rbac_decision"):
                result = secure_handler.check_permission(auth_context, "debates.read")
                assert result is True
                mock_checker.check_permission.assert_called_once_with(
                    auth_context, "debates.read", None
                )

    def test_check_permission_denied_raises(self, secure_handler, auth_context):
        """check_permission raises ForbiddenError when permission denied."""
        mock_decision = MagicMock()
        mock_decision.allowed = False

        mock_checker = MagicMock()
        mock_checker.check_permission.return_value = mock_decision

        with patch(
            "aragora.rbac.checker.get_permission_checker",
            return_value=mock_checker,
        ):
            with patch("aragora.observability.metrics.security.record_rbac_decision"):
                with pytest.raises(ForbiddenError) as exc_info:
                    secure_handler.check_permission(auth_context, "admin.delete")
                assert "admin.delete" in str(exc_info.value)

    def test_check_permission_with_resource_id(self, secure_handler, auth_context):
        """check_permission passes resource_id to checker."""
        mock_decision = MagicMock()
        mock_decision.allowed = True

        mock_checker = MagicMock()
        mock_checker.check_permission.return_value = mock_decision

        with patch(
            "aragora.rbac.checker.get_permission_checker",
            return_value=mock_checker,
        ):
            with patch("aragora.observability.metrics.security.record_rbac_decision"):
                secure_handler.check_permission(auth_context, "debates.read", "debate-123")
                mock_checker.check_permission.assert_called_once_with(
                    auth_context, "debates.read", "debate-123"
                )


class TestSecureHandlerAuditAction:
    """Tests for SecureHandler.audit_action() method."""

    @pytest.mark.asyncio
    async def test_audit_action_basic(self, secure_handler, auth_context, mock_request):
        """audit_action logs to audit trail."""
        mock_audit_log = AsyncMock()

        with patch(
            "aragora.observability.immutable_log.get_audit_log",
            return_value=mock_audit_log,
        ):
            await secure_handler.audit_action(
                auth_context,
                action="create",
                resource_id="debate-123",
                request=mock_request,
            )

            mock_audit_log.append.assert_awaited_once()
            call_kwargs = mock_audit_log.append.call_args.kwargs
            assert call_kwargs["actor"] == "user-123"
            assert call_kwargs["action"] == "create"
            assert call_kwargs["resource_id"] == "debate-123"
            assert call_kwargs["ip_address"] == "192.168.1.1"
            assert call_kwargs["user_agent"] == "TestClient/1.0"

    @pytest.mark.asyncio
    async def test_audit_action_custom_resource_type(self, secure_handler, auth_context):
        """audit_action uses custom resource type."""
        mock_audit_log = AsyncMock()

        with patch(
            "aragora.observability.immutable_log.get_audit_log",
            return_value=mock_audit_log,
        ):
            await secure_handler.audit_action(
                auth_context,
                action="update",
                resource_id="config-456",
                resource_type="settings",
            )

            call_kwargs = mock_audit_log.append.call_args.kwargs
            assert call_kwargs["resource_type"] == "settings"
            assert call_kwargs["event_type"] == "settings.update"

    @pytest.mark.asyncio
    async def test_audit_action_with_details(self, secure_handler, auth_context):
        """audit_action includes additional details."""
        mock_audit_log = AsyncMock()

        with patch(
            "aragora.observability.immutable_log.get_audit_log",
            return_value=mock_audit_log,
        ):
            await secure_handler.audit_action(
                auth_context,
                action="delete",
                resource_id="item-789",
                details={"reason": "cleanup", "count": 5},
            )

            call_kwargs = mock_audit_log.append.call_args.kwargs
            assert call_kwargs["details"]["reason"] == "cleanup"
            assert call_kwargs["details"]["count"] == 5


class TestSecureHandlerEncryption:
    """Tests for SecureHandler encryption methods."""

    def test_encrypt_response_fields(self, secure_handler):
        """encrypt_response_fields encrypts sensitive data."""
        with patch(
            "aragora.storage.encrypted_fields.encrypt_sensitive",
            return_value={"api_key": "ENCRYPTED", "name": "test"},
        ) as mock_encrypt:
            result = secure_handler.encrypt_response_fields(
                {"api_key": "secret-key", "name": "test"}
            )
            assert result["api_key"] == "ENCRYPTED"
            assert result["name"] == "test"
            mock_encrypt.assert_called_once()

    def test_decrypt_request_fields(self, secure_handler):
        """decrypt_request_fields decrypts sensitive data."""
        with patch(
            "aragora.storage.encrypted_fields.decrypt_sensitive",
            return_value={"api_key": "decrypted-key", "name": "test"},
        ) as mock_decrypt:
            result = secure_handler.decrypt_request_fields({"api_key": "ENCRYPTED", "name": "test"})
            assert result["api_key"] == "decrypted-key"
            mock_decrypt.assert_called_once()


class TestSecureHandlerErrorHandling:
    """Tests for SecureHandler.handle_security_error() method."""

    def test_handle_unauthorized_error(self, secure_handler):
        """handle_security_error returns 401 for UnauthorizedError."""
        with patch("aragora.observability.metrics.security.record_auth_failure"):
            error = UnauthorizedError("Token invalid")
            result = secure_handler.handle_security_error(error)
            assert result.status_code == 401

    def test_handle_forbidden_error(self, secure_handler):
        """handle_security_error returns 403 for ForbiddenError."""
        with patch("aragora.observability.metrics.security.record_blocked_request"):
            error = ForbiddenError("No access", permission="admin.delete")
            result = secure_handler.handle_security_error(error)
            assert result.status_code == 403

    def test_handle_permission_denied_error(self, secure_handler):
        """handle_security_error returns 403 for PermissionDeniedError."""
        with patch("aragora.observability.metrics.security.record_blocked_request"):
            error = PermissionDeniedError("Permission denied: users.delete")
            result = secure_handler.handle_security_error(error)
            assert result.status_code == 403

    def test_handle_role_required_error(self, secure_handler):
        """handle_security_error returns 403 for RoleRequiredError."""
        with patch("aragora.observability.metrics.security.record_blocked_request"):
            error = RoleRequiredError("Role required: admin", {"admin"}, {"member"})
            result = secure_handler.handle_security_error(error)
            assert result.status_code == 403

    def test_handle_unknown_security_error(self, secure_handler):
        """handle_security_error returns 500 for unknown errors."""
        error = Exception("Unexpected error")
        result = secure_handler.handle_security_error(error)
        assert result.status_code == 500


# -----------------------------------------------------------------------------
# @secure_endpoint Decorator Tests
# -----------------------------------------------------------------------------


class TestSecureEndpointDecorator:
    """Tests for @secure_endpoint decorator."""

    @pytest.mark.asyncio
    async def test_secure_endpoint_basic_auth(self, server_context, mock_request, auth_context):
        """@secure_endpoint extracts authentication."""

        class TestHandler(SecureHandler):
            @secure_endpoint()
            async def handle_get(self, request, auth_context):
                return json_response({"user_id": auth_context.user_id})

        handler = TestHandler(server_context)

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch("aragora.observability.metrics.security.record_auth_attempt"):
                result = await handler.handle_get(mock_request)
                assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_secure_endpoint_permission_check(
        self, server_context, mock_request, auth_context
    ):
        """@secure_endpoint checks permission when specified."""

        class TestHandler(SecureHandler):
            @secure_endpoint(permission="debates.create")
            async def handle_post(self, request, auth_context):
                return json_response({"created": True})

        handler = TestHandler(server_context)

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True) as mock_check:
                with patch("aragora.observability.metrics.security.record_auth_attempt"):
                    with patch("aragora.observability.metrics.security.track_rbac_evaluation"):
                        await handler.handle_post(mock_request)
                        mock_check.assert_called_once_with(auth_context, "debates.create", None)

    @pytest.mark.asyncio
    async def test_secure_endpoint_permission_with_resource_id(
        self, server_context, mock_request, auth_context
    ):
        """@secure_endpoint passes resource_id from kwargs."""

        class TestHandler(SecureHandler):
            @secure_endpoint(permission="debates.read", resource_id_param="debate_id")
            async def handle_get(self, request, auth_context, debate_id=None):
                return json_response({"id": debate_id})

        handler = TestHandler(server_context)

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True) as mock_check:
                with patch("aragora.observability.metrics.security.record_auth_attempt"):
                    with patch("aragora.observability.metrics.security.track_rbac_evaluation"):
                        await handler.handle_get(mock_request, debate_id="debate-123")
                        mock_check.assert_called_once_with(
                            auth_context, "debates.read", "debate-123"
                        )

    @pytest.mark.asyncio
    async def test_secure_endpoint_audit_logging(self, server_context, mock_request, auth_context):
        """@secure_endpoint logs to audit trail when audit=True."""

        class TestHandler(SecureHandler):
            @secure_endpoint(permission="items.create", audit=True, resource_id_param="item_id")
            async def handle_post(self, request, auth_context, item_id=None):
                return json_response({"created": True})

        handler = TestHandler(server_context)

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch.object(handler, "audit_action", new_callable=AsyncMock) as mock_audit:
                    with patch("aragora.observability.metrics.security.record_auth_attempt"):
                        with patch("aragora.observability.metrics.security.track_rbac_evaluation"):
                            await handler.handle_post(mock_request, item_id="item-456")
                            mock_audit.assert_awaited_once()
                            call_kwargs = mock_audit.call_args.kwargs
                            assert call_kwargs["action"] == "post"
                            assert call_kwargs["resource_id"] == "item-456"

    @pytest.mark.asyncio
    async def test_secure_endpoint_custom_audit_action(
        self, server_context, mock_request, auth_context
    ):
        """@secure_endpoint uses custom audit_action name."""

        class TestHandler(SecureHandler):
            @secure_endpoint(
                permission="items.delete",
                audit=True,
                audit_action="purge",
                resource_id_param="item_id",
            )
            async def handle_delete(self, request, auth_context, item_id=None):
                return json_response({"deleted": True})

        handler = TestHandler(server_context)

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch.object(handler, "audit_action", new_callable=AsyncMock) as mock_audit:
                    with patch("aragora.observability.metrics.security.record_auth_attempt"):
                        with patch("aragora.observability.metrics.security.track_rbac_evaluation"):
                            await handler.handle_delete(mock_request, item_id="item-789")
                            call_kwargs = mock_audit.call_args.kwargs
                            assert call_kwargs["action"] == "purge"

    @pytest.mark.asyncio
    async def test_secure_endpoint_no_auth_required(
        self, server_context, mock_request, anonymous_context
    ):
        """@secure_endpoint allows anonymous access when require_auth=False."""

        class TestHandler(SecureHandler):
            @secure_endpoint(require_auth=False)
            async def handle_get(self, request, auth_context):
                return json_response({"public": True, "user": auth_context.user_id})

        handler = TestHandler(server_context)

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=anonymous_context
        ):
            with patch("aragora.observability.metrics.security.record_auth_attempt"):
                result = await handler.handle_get(mock_request)
                assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_secure_endpoint_unauthorized_error(self, server_context, mock_request):
        """@secure_endpoint handles UnauthorizedError."""

        class TestHandler(SecureHandler):
            @secure_endpoint()
            async def handle_get(self, request, auth_context):
                return json_response({})

        handler = TestHandler(server_context)

        with patch.object(
            handler,
            "get_auth_context",
            new_callable=AsyncMock,
            side_effect=UnauthorizedError("Invalid token"),
        ):
            with patch.object(
                handler, "handle_security_error", return_value=error_response("Auth required", 401)
            ) as mock_error:
                result = await handler.handle_get(mock_request)
                mock_error.assert_called_once()
                assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_secure_endpoint_forbidden_error(
        self, server_context, mock_request, auth_context
    ):
        """@secure_endpoint handles ForbiddenError."""

        class TestHandler(SecureHandler):
            @secure_endpoint(permission="admin.manage")
            async def handle_post(self, request, auth_context):
                return json_response({})

        handler = TestHandler(server_context)

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(
                handler, "check_permission", side_effect=ForbiddenError("No admin access")
            ):
                with patch.object(
                    handler,
                    "handle_security_error",
                    return_value=error_response("Forbidden", 403),
                ) as mock_error:
                    with patch("aragora.observability.metrics.security.record_auth_attempt"):
                        with patch("aragora.observability.metrics.security.track_rbac_evaluation"):
                            result = await handler.handle_post(mock_request)
                            mock_error.assert_called_once()
                            assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_secure_endpoint_permission_denied_error(
        self, server_context, mock_request, auth_context
    ):
        """@secure_endpoint handles PermissionDeniedError."""

        class TestHandler(SecureHandler):
            @secure_endpoint(permission="users.delete")
            async def handle_delete(self, request, auth_context):
                return json_response({})

        handler = TestHandler(server_context)

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(
                handler,
                "check_permission",
                side_effect=PermissionDeniedError("users.delete"),
            ):
                with patch.object(
                    handler,
                    "handle_security_error",
                    return_value=error_response("Permission denied", 403),
                ):
                    with patch("aragora.observability.metrics.security.record_auth_attempt"):
                        with patch("aragora.observability.metrics.security.track_rbac_evaluation"):
                            result = await handler.handle_delete(mock_request)
                            assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_secure_endpoint_role_required_error(
        self, server_context, mock_request, auth_context
    ):
        """@secure_endpoint handles RoleRequiredError."""

        class TestHandler(SecureHandler):
            @secure_endpoint(permission="system.configure")
            async def handle_post(self, request, auth_context):
                return json_response({})

        handler = TestHandler(server_context)

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(
                handler,
                "check_permission",
                side_effect=RoleRequiredError("Role required", {"owner"}, {"member"}),
            ):
                with patch.object(
                    handler,
                    "handle_security_error",
                    return_value=error_response("Role required", 403),
                ):
                    with patch("aragora.observability.metrics.security.record_auth_attempt"):
                        with patch("aragora.observability.metrics.security.track_rbac_evaluation"):
                            result = await handler.handle_post(mock_request)
                            assert result.status_code == 403


# -----------------------------------------------------------------------------
# @audit_sensitive_access Decorator Tests
# -----------------------------------------------------------------------------


class TestAuditSensitiveAccessDecorator:
    """Tests for @audit_sensitive_access decorator."""

    @pytest.mark.asyncio
    async def test_audit_sensitive_access_records_metric(
        self, server_context, mock_request, auth_context
    ):
        """@audit_sensitive_access records secret access metric."""

        class TestHandler(SecureHandler):
            RESOURCE_TYPE = "api_keys"

            @audit_sensitive_access("api_key", "read")
            async def get_api_key(self, request, auth_context):
                return json_response({"key": "xxx"})

        handler = TestHandler(server_context)

        with patch("aragora.observability.metrics.security.record_secret_access") as mock_metric:
            with patch(
                "aragora.observability.security_audit.audit_secret_access", new_callable=AsyncMock
            ):
                await handler.get_api_key(mock_request, auth_context)
                mock_metric.assert_called_once_with("api_key", "read")

    @pytest.mark.asyncio
    async def test_audit_sensitive_access_logs_audit(
        self, server_context, mock_request, auth_context
    ):
        """@audit_sensitive_access logs to security audit."""

        class TestHandler(SecureHandler):
            RESOURCE_TYPE = "credentials"

            @audit_sensitive_access("oauth_token", "access")
            async def get_token(self, request, auth_context):
                return json_response({"token": "xxx"})

        handler = TestHandler(server_context)

        with patch("aragora.observability.metrics.security.record_secret_access"):
            with patch(
                "aragora.observability.security_audit.audit_secret_access", new_callable=AsyncMock
            ) as mock_audit:
                await handler.get_token(mock_request, auth_context)
                mock_audit.assert_awaited_once()
                call_kwargs = mock_audit.call_args.kwargs
                assert call_kwargs["actor"] == "user-123"
                assert call_kwargs["secret_type"] == "oauth_token"
                assert call_kwargs["operation"] == "access"
                assert call_kwargs["store"] == "credentials"

    @pytest.mark.asyncio
    async def test_audit_sensitive_access_with_org(
        self, server_context, mock_request, auth_context
    ):
        """@audit_sensitive_access logs actor and store correctly."""

        class TestHandler(SecureHandler):
            RESOURCE_TYPE = "secrets"

            @audit_sensitive_access("password", "decrypt")
            async def decrypt_password(self, request, auth_context):
                return json_response({"decrypted": True})

        handler = TestHandler(server_context)

        with patch("aragora.observability.metrics.security.record_secret_access"):
            with patch(
                "aragora.observability.security_audit.audit_secret_access", new_callable=AsyncMock
            ) as mock_audit:
                await handler.decrypt_password(mock_request, auth_context)
                call_kwargs = mock_audit.call_args.kwargs
                assert call_kwargs["actor"] == "user-123"
                assert call_kwargs["store"] == "secrets"


# -----------------------------------------------------------------------------
# Exception Class Tests
# -----------------------------------------------------------------------------


class TestSecurityExceptions:
    """Tests for security exception classes."""

    def test_unauthorized_error_default_message(self):
        """UnauthorizedError has default message."""
        error = UnauthorizedError()
        assert error.message == "Authentication required"

    def test_unauthorized_error_custom_message(self):
        """UnauthorizedError accepts custom message."""
        error = UnauthorizedError("Token expired")
        assert error.message == "Token expired"
        assert str(error) == "Token expired"

    def test_forbidden_error_default_message(self):
        """ForbiddenError has default message."""
        error = ForbiddenError()
        assert error.message == "Access denied"
        assert error.permission is None

    def test_forbidden_error_with_permission(self):
        """ForbiddenError stores permission info."""
        error = ForbiddenError("No access to debates", permission="debates.delete")
        assert error.message == "No access to debates"
        assert error.permission == "debates.delete"


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestSecureHandlerIntegration:
    """Integration tests for SecureHandler patterns."""

    @pytest.mark.asyncio
    async def test_full_secure_endpoint_flow(self, server_context, mock_request, auth_context):
        """Full flow through secure endpoint with auth, permission, and audit."""

        class DebateHandler(SecureHandler):
            RESOURCE_TYPE = "debate"

            @secure_endpoint(
                permission="debates.create",
                audit=True,
                resource_id_param="debate_id",
            )
            async def handle_post(self, request, auth_context, debate_id=None):
                return json_response(
                    {
                        "id": debate_id or "new-debate",
                        "created_by": auth_context.user_id,
                    }
                )

        handler = DebateHandler(server_context)

        # Mock all dependencies
        mock_decision = MagicMock()
        mock_decision.allowed = True
        mock_checker = MagicMock()
        mock_checker.check_permission.return_value = mock_decision

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch(
                "aragora.rbac.checker.get_permission_checker",
                return_value=mock_checker,
            ):
                with patch("aragora.observability.metrics.security.record_rbac_decision"):
                    with patch("aragora.observability.metrics.security.record_auth_attempt"):
                        with patch("aragora.observability.metrics.security.track_rbac_evaluation"):
                            with patch.object(
                                handler, "audit_action", new_callable=AsyncMock
                            ) as mock_audit:
                                result = await handler.handle_post(
                                    mock_request, debate_id="debate-new"
                                )

                                assert result.status_code == 200
                                mock_checker.check_permission.assert_called_once()
                                mock_audit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_secure_handler_subclass_customization(
        self, server_context, mock_request, auth_context
    ):
        """Subclass can customize RESOURCE_TYPE and DEFAULT_METHOD_PERMISSIONS."""

        class WorkflowHandler(SecureHandler):
            RESOURCE_TYPE = "workflow"
            DEFAULT_METHOD_PERMISSIONS = {
                "GET": "workflows.read",
                "POST": "workflows.create",
                "PUT": "workflows.update",
                "DELETE": "workflows.delete",
            }

            @secure_endpoint(permission="workflows.read")
            async def handle_get(self, request, auth_context):
                return json_response({"workflows": []})

        handler = WorkflowHandler(server_context)
        assert handler.RESOURCE_TYPE == "workflow"
        assert handler.DEFAULT_METHOD_PERMISSIONS["GET"] == "workflows.read"

        with patch.object(
            handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(handler, "check_permission", return_value=True):
                with patch("aragora.observability.metrics.security.record_auth_attempt"):
                    with patch("aragora.observability.metrics.security.track_rbac_evaluation"):
                        result = await handler.handle_get(mock_request)
                        assert result.status_code == 200
