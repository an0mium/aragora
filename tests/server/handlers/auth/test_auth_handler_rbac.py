"""
Tests for AuthHandler RBAC enforcement.

Tests cover:
- Permission checks on all protected endpoints
- Role-based access control
- JWT vs header-based authentication
- Graceful error handling for permission denials
"""

import pytest
from unittest.mock import MagicMock, patch

from aragora.server.handlers.auth.handler import AuthHandler
from aragora.rbac import AuthorizationContext, AuthorizationDecision


class TestAuthHandlerRBAC:
    """Tests for RBAC enforcement in AuthHandler."""

    @pytest.fixture
    def handler(self, mock_server_context):
        """Create handler instance."""
        return AuthHandler(server_context=mock_server_context)

    @pytest.fixture
    def mock_http_handler(self):
        """Create mock HTTP handler with configurable headers."""

        def _make_handler(
            user_id="user-1",
            roles="member",
            method="GET",
        ):
            handler = MagicMock()
            handler.command = method
            handler.headers = {
                "X-User-ID": user_id,
                "X-User-Roles": roles,
            }
            return handler

        return _make_handler

    def test_check_permission_requires_jwt(self, handler, mock_http_handler):
        """_check_permission requires JWT - headers alone are rejected."""
        http_handler = mock_http_handler(user_id="test-user", roles="admin")

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request"
        ) as mock_extract:
            mock_extract.return_value = MagicMock(is_authenticated=False, user_id=None)
            result = handler._check_permission(http_handler, "authentication.read")

        # JWT required - headers alone should return 401
        assert result is not None
        assert result.status_code == 401

    def test_check_permission_allows_authorized(self, handler, mock_http_handler):
        """_check_permission allows authorized users with valid JWT."""
        http_handler = mock_http_handler(user_id="admin", roles="admin")

        with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
            mock_check.return_value = AuthorizationDecision(
                allowed=True, reason="Admin role", permission_key="authentication.read"
            )
            with patch(
                "aragora.server.handlers.auth.handler.extract_user_from_request"
            ) as mock_extract:
                # Valid JWT authentication
                mock_extract.return_value = MagicMock(
                    is_authenticated=True,
                    user_id="admin",
                    role="admin",
                    org_id="org-1",
                    client_ip="127.0.0.1",
                )
                result = handler._check_permission(http_handler, "authentication.read")

        assert result is None  # No error means allowed

    @pytest.mark.no_auto_auth
    def test_check_permission_denies_unauthenticated(self, handler, mock_http_handler):
        """_check_permission returns 401 for unauthenticated users."""
        http_handler = mock_http_handler(user_id="viewer", roles="viewer")

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request"
        ) as mock_extract:
            # No valid JWT
            mock_extract.return_value = MagicMock(is_authenticated=False, user_id=None)
            result = handler._check_permission(http_handler, "authentication.read")

        assert result is not None
        assert result.status_code == 401  # Unauthenticated

    def test_check_permission_denies_unauthorized(self, handler, mock_http_handler):
        """_check_permission returns 403 for authenticated but unauthorized users."""
        http_handler = mock_http_handler(user_id="viewer", roles="viewer")

        with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
            mock_check.return_value = AuthorizationDecision(
                allowed=False,
                reason="Insufficient permissions",
                permission_key="api_key.create",
            )
            with patch(
                "aragora.server.handlers.auth.handler.extract_user_from_request"
            ) as mock_extract:
                # Valid JWT but insufficient permissions
                mock_extract.return_value = MagicMock(
                    is_authenticated=True,
                    user_id="viewer",
                    role="viewer",
                    org_id="org-1",
                    client_ip="127.0.0.1",
                )
                result = handler._check_permission(http_handler, "api_key.create")

        assert result is not None
        assert result.status_code == 403  # Forbidden


class TestAuthHandlerPermissionKeys:
    """Tests to verify correct permission keys are used for each endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        """Create handler instance."""
        return AuthHandler(server_context=mock_server_context)

    @pytest.fixture
    def mock_http_handler(self):
        """Create minimal mock HTTP handler."""
        handler = MagicMock()
        handler.command = "POST"
        handler.headers = {"X-User-ID": "test", "X-User-Roles": "admin"}
        return handler

    @pytest.mark.parametrize(
        "method_name,permission_key",
        [
            ("_handle_logout", "authentication.revoke"),
            ("_handle_logout_all", "authentication.revoke"),
            ("_handle_get_me", "authentication.read"),
            ("_handle_update_me", "authentication.read"),
            ("_handle_change_password", "authentication.read"),
            ("_handle_revoke_token", "session.revoke"),
            ("_handle_generate_api_key", "api_key.create"),
            ("_handle_revoke_api_key", "api_key.revoke"),
            ("_handle_mfa_setup", "authentication.create"),
            ("_handle_mfa_enable", "authentication.update"),
            ("_handle_mfa_disable", "authentication.update"),
            ("_handle_mfa_backup_codes", "authentication.read"),
            ("_handle_list_sessions", "session.list_active"),
            ("_handle_revoke_session", "session.revoke"),
        ],
    )
    def test_permission_key_used(self, handler, mock_http_handler, method_name, permission_key):
        """Verify correct permission key is checked for each protected endpoint."""
        with patch.object(handler, "_check_permission") as mock_check:
            mock_check.return_value = MagicMock(status_code=403, body=b"denied")

            method = getattr(handler, method_name)

            # Call with appropriate args based on method signature
            if method_name == "_handle_revoke_session":
                method(mock_http_handler, "session-123")
            else:
                method(mock_http_handler)

            # Verify permission was checked with correct key
            assert mock_check.called
            actual_args = mock_check.call_args[0]
            assert actual_args[1] == permission_key


class TestAuthHandlerRBACIntegration:
    """Integration tests for RBAC enforcement in AuthHandler endpoints."""

    @pytest.fixture
    def handler(self, mock_server_context):
        """Create handler instance."""
        return AuthHandler(server_context=mock_server_context)

    @pytest.fixture
    def mock_http_handler(self):
        """Create mock HTTP handler."""
        handler = MagicMock()
        handler.command = "GET"
        handler.headers = {}
        return handler

    @pytest.mark.no_auto_auth
    def test_logout_requires_authentication(self, handler, mock_http_handler):
        """Logout endpoint returns 401 without authentication."""
        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request"
        ) as mock_extract:
            mock_extract.return_value = MagicMock(is_authenticated=False, user_id=None)
            result = handler._handle_logout(mock_http_handler)

        assert result.status_code == 401

    @pytest.mark.no_auto_auth
    def test_get_me_requires_authentication(self, handler, mock_http_handler):
        """Get me endpoint returns 401 without authentication."""
        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request"
        ) as mock_extract:
            mock_extract.return_value = MagicMock(is_authenticated=False, user_id=None)
            result = handler._handle_get_me(mock_http_handler)

        assert result.status_code == 401

    @pytest.mark.no_auto_auth
    def test_generate_api_key_requires_authentication(self, handler, mock_http_handler):
        """Generate API key endpoint returns 401 without authentication."""
        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request"
        ) as mock_extract:
            mock_extract.return_value = MagicMock(is_authenticated=False, user_id=None)
            result = handler._handle_generate_api_key(mock_http_handler)

        assert result.status_code == 401

    @pytest.mark.no_auto_auth
    def test_mfa_setup_requires_authentication(self, handler, mock_http_handler):
        """MFA setup endpoint returns 401 without authentication."""
        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request"
        ) as mock_extract:
            mock_extract.return_value = MagicMock(is_authenticated=False, user_id=None)
            result = handler._handle_mfa_setup(mock_http_handler)

        assert result.status_code == 401

    @pytest.mark.no_auto_auth
    def test_list_sessions_requires_authentication(self, handler, mock_http_handler):
        """List sessions endpoint returns 401 without authentication."""
        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request"
        ) as mock_extract:
            mock_extract.return_value = MagicMock(is_authenticated=False, user_id=None)
            result = handler._handle_list_sessions(mock_http_handler)

        assert result.status_code == 401
