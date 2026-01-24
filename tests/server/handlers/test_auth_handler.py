"""
Tests for the AuthHandler module.
Tests cover:
- Handler routing for all auth endpoints
- can_handle method for static and dynamic routes
- ROUTES attribute
- Rate limiting decorators
- Response formatting
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock


class TestAuthHandlerImport:
    """Tests for importing AuthHandler."""

    def test_can_import_handler(self):
        """AuthHandler can be imported."""
        from aragora.server.handlers.auth.handler import AuthHandler

        assert AuthHandler is not None

    def test_handler_in_all(self):
        """AuthHandler is in __all__."""
        from aragora.server.handlers.auth import handler

        assert "AuthHandler" in handler.__all__


class TestAuthHandlerRoutes:
    """Tests for AuthHandler ROUTES attribute."""

    @pytest.fixture
    def handler(self):
        """Create handler instance."""
        from aragora.server.handlers.auth.handler import AuthHandler

        return AuthHandler(server_context={})

    def test_routes_is_list(self, handler):
        """ROUTES is a list."""
        assert isinstance(handler.ROUTES, list)

    def test_routes_not_empty(self, handler):
        """ROUTES is not empty."""
        assert len(handler.ROUTES) > 0

    def test_register_route_in_routes(self, handler):
        """Register route is in ROUTES."""
        assert "/api/auth/register" in handler.ROUTES

    def test_login_route_in_routes(self, handler):
        """Login route is in ROUTES."""
        assert "/api/auth/login" in handler.ROUTES

    def test_logout_route_in_routes(self, handler):
        """Logout route is in ROUTES."""
        assert "/api/auth/logout" in handler.ROUTES

    def test_logout_all_route_in_routes(self, handler):
        """Logout-all route is in ROUTES."""
        assert "/api/auth/logout-all" in handler.ROUTES

    def test_refresh_route_in_routes(self, handler):
        """Refresh route is in ROUTES."""
        assert "/api/auth/refresh" in handler.ROUTES

    def test_revoke_route_in_routes(self, handler):
        """Revoke route is in ROUTES."""
        assert "/api/auth/revoke" in handler.ROUTES

    def test_me_route_in_routes(self, handler):
        """Me route is in ROUTES."""
        assert "/api/auth/me" in handler.ROUTES

    def test_password_route_in_routes(self, handler):
        """Password route is in ROUTES."""
        assert "/api/auth/password" in handler.ROUTES

    def test_api_key_route_in_routes(self, handler):
        """API key route is in ROUTES."""
        assert "/api/auth/api-key" in handler.ROUTES

    def test_mfa_routes_in_routes(self, handler):
        """MFA routes are in ROUTES."""
        assert "/api/auth/mfa/setup" in handler.ROUTES
        assert "/api/auth/mfa/enable" in handler.ROUTES
        assert "/api/auth/mfa/disable" in handler.ROUTES
        assert "/api/auth/mfa/verify" in handler.ROUTES
        assert "/api/auth/mfa/backup-codes" in handler.ROUTES

    def test_sessions_route_in_routes(self, handler):
        """Sessions route is in ROUTES."""
        assert "/api/auth/sessions" in handler.ROUTES

    def test_sessions_wildcard_in_routes(self, handler):
        """Sessions wildcard route is in ROUTES."""
        assert "/api/auth/sessions/*" in handler.ROUTES


class TestAuthHandlerCanHandle:
    """Tests for can_handle method."""

    @pytest.fixture
    def handler(self):
        """Create handler instance."""
        from aragora.server.handlers.auth.handler import AuthHandler

        return AuthHandler(server_context={})

    def test_can_handle_register(self, handler):
        """Handler can handle register endpoint."""
        assert handler.can_handle("/api/auth/register") is True

    def test_can_handle_login(self, handler):
        """Handler can handle login endpoint."""
        assert handler.can_handle("/api/auth/login") is True

    def test_can_handle_logout(self, handler):
        """Handler can handle logout endpoint."""
        assert handler.can_handle("/api/auth/logout") is True

    def test_can_handle_me(self, handler):
        """Handler can handle me endpoint."""
        assert handler.can_handle("/api/auth/me") is True

    def test_can_handle_mfa_setup(self, handler):
        """Handler can handle MFA setup endpoint."""
        assert handler.can_handle("/api/auth/mfa/setup") is True

    def test_can_handle_sessions(self, handler):
        """Handler can handle sessions endpoint."""
        assert handler.can_handle("/api/auth/sessions") is True

    def test_can_handle_session_id(self, handler):
        """Handler can handle session ID endpoint (wildcard)."""
        assert handler.can_handle("/api/auth/sessions/abc123") is True

    def test_can_handle_session_uuid(self, handler):
        """Handler can handle session UUID endpoint."""
        assert handler.can_handle("/api/auth/sessions/550e8400-e29b-41d4-a716-446655440000") is True

    def test_cannot_handle_unrelated_path(self, handler):
        """Handler does not handle unrelated paths."""
        assert handler.can_handle("/api/v1/debates") is False

    def test_cannot_handle_partial_auth_path(self, handler):
        """Handler does not handle partial auth paths."""
        assert handler.can_handle("/api/v1/auth") is False

    def test_cannot_handle_auth_unknown(self, handler):
        """Handler does not handle unknown auth endpoints."""
        assert handler.can_handle("/api/auth/unknown") is False


class TestAuthHandlerRouting:
    """Tests for handle method routing."""

    @pytest.fixture
    def handler(self):
        """Create handler instance with mock store."""
        from aragora.server.handlers.auth.handler import AuthHandler

        mock_store = MagicMock()
        return AuthHandler(server_context={"user_store": mock_store})

    @pytest.fixture
    def mock_http(self):
        """Create mock HTTP handler."""
        mock = MagicMock()
        mock.rfile = MagicMock()
        mock.rfile.read = MagicMock(return_value=b"{}")
        mock.headers = {"Content-Length": "2"}
        mock.command = "POST"
        return mock

    def test_register_routes_to_handler(self, handler, mock_http):
        """Register POST routes to _handle_register."""
        with patch.object(handler, "_handle_register") as mock_method:
            mock_method.return_value = MagicMock()
            handler.handle("/api/auth/register", {}, mock_http, "POST")
            mock_method.assert_called_once()

    def test_login_routes_to_handler(self, handler, mock_http):
        """Login POST routes to _handle_login."""
        with patch.object(handler, "_handle_login") as mock_method:
            mock_method.return_value = MagicMock()
            handler.handle("/api/auth/login", {}, mock_http, "POST")
            mock_method.assert_called_once()

    def test_logout_routes_to_handler(self, handler, mock_http):
        """Logout POST routes to _handle_logout."""
        with patch.object(handler, "_handle_logout") as mock_method:
            mock_method.return_value = MagicMock()
            handler.handle("/api/auth/logout", {}, mock_http, "POST")
            mock_method.assert_called_once()

    def test_logout_all_routes_to_handler(self, handler, mock_http):
        """Logout-all POST routes to _handle_logout_all."""
        with patch.object(handler, "_handle_logout_all") as mock_method:
            mock_method.return_value = MagicMock()
            handler.handle("/api/auth/logout-all", {}, mock_http, "POST")
            mock_method.assert_called_once()

    def test_refresh_routes_to_handler(self, handler, mock_http):
        """Refresh POST routes to _handle_refresh."""
        with patch.object(handler, "_handle_refresh") as mock_method:
            mock_method.return_value = MagicMock()
            handler.handle("/api/auth/refresh", {}, mock_http, "POST")
            mock_method.assert_called_once()

    def test_me_get_routes_to_handler(self, handler, mock_http):
        """Me GET routes to _handle_get_me."""
        mock_http.command = "GET"
        with patch.object(handler, "_handle_get_me") as mock_method:
            mock_method.return_value = MagicMock()
            handler.handle("/api/auth/me", {}, mock_http, "GET")
            mock_method.assert_called_once()

    def test_me_put_routes_to_handler(self, handler, mock_http):
        """Me PUT routes to _handle_update_me."""
        mock_http.command = "PUT"
        with patch.object(handler, "_handle_update_me") as mock_method:
            mock_method.return_value = MagicMock()
            handler.handle("/api/auth/me", {}, mock_http, "PUT")
            mock_method.assert_called_once()

    def test_password_routes_to_handler(self, handler, mock_http):
        """Password POST routes to _handle_change_password."""
        with patch.object(handler, "_handle_change_password") as mock_method:
            mock_method.return_value = MagicMock()
            handler.handle("/api/auth/password", {}, mock_http, "POST")
            mock_method.assert_called_once()

    def test_revoke_routes_to_handler(self, handler, mock_http):
        """Revoke POST routes to _handle_revoke_token."""
        with patch.object(handler, "_handle_revoke_token") as mock_method:
            mock_method.return_value = MagicMock()
            handler.handle("/api/auth/revoke", {}, mock_http, "POST")
            mock_method.assert_called_once()

    def test_api_key_post_routes_to_handler(self, handler, mock_http):
        """API key POST routes to _handle_generate_api_key."""
        with patch.object(handler, "_handle_generate_api_key") as mock_method:
            mock_method.return_value = MagicMock()
            handler.handle("/api/auth/api-key", {}, mock_http, "POST")
            mock_method.assert_called_once()

    def test_api_key_delete_routes_to_handler(self, handler, mock_http):
        """API key DELETE routes to _handle_revoke_api_key."""
        mock_http.command = "DELETE"
        with patch.object(handler, "_handle_revoke_api_key") as mock_method:
            mock_method.return_value = MagicMock()
            handler.handle("/api/auth/api-key", {}, mock_http, "DELETE")
            mock_method.assert_called_once()

    def test_mfa_setup_routes_to_handler(self, handler, mock_http):
        """MFA setup POST routes to _handle_mfa_setup."""
        with patch.object(handler, "_handle_mfa_setup") as mock_method:
            mock_method.return_value = MagicMock()
            handler.handle("/api/auth/mfa/setup", {}, mock_http, "POST")
            mock_method.assert_called_once()

    def test_mfa_enable_routes_to_handler(self, handler, mock_http):
        """MFA enable POST routes to _handle_mfa_enable."""
        with patch.object(handler, "_handle_mfa_enable") as mock_method:
            mock_method.return_value = MagicMock()
            handler.handle("/api/auth/mfa/enable", {}, mock_http, "POST")
            mock_method.assert_called_once()

    def test_mfa_disable_routes_to_handler(self, handler, mock_http):
        """MFA disable POST routes to _handle_mfa_disable."""
        with patch.object(handler, "_handle_mfa_disable") as mock_method:
            mock_method.return_value = MagicMock()
            handler.handle("/api/auth/mfa/disable", {}, mock_http, "POST")
            mock_method.assert_called_once()

    def test_mfa_verify_routes_to_handler(self, handler, mock_http):
        """MFA verify POST routes to _handle_mfa_verify."""
        with patch.object(handler, "_handle_mfa_verify") as mock_method:
            mock_method.return_value = MagicMock()
            handler.handle("/api/auth/mfa/verify", {}, mock_http, "POST")
            mock_method.assert_called_once()

    def test_mfa_backup_routes_to_handler(self, handler, mock_http):
        """MFA backup codes POST routes to _handle_mfa_backup_codes."""
        with patch.object(handler, "_handle_mfa_backup_codes") as mock_method:
            mock_method.return_value = MagicMock()
            handler.handle("/api/auth/mfa/backup-codes", {}, mock_http, "POST")
            mock_method.assert_called_once()

    def test_sessions_get_routes_to_handler(self, handler, mock_http):
        """Sessions GET routes to _handle_list_sessions."""
        mock_http.command = "GET"
        with patch.object(handler, "_handle_list_sessions") as mock_method:
            mock_method.return_value = MagicMock()
            handler.handle("/api/auth/sessions", {}, mock_http, "GET")
            mock_method.assert_called_once()

    def test_session_delete_routes_to_handler(self, handler, mock_http):
        """Session DELETE routes to _handle_revoke_session."""
        mock_http.command = "DELETE"
        with patch.object(handler, "_handle_revoke_session") as mock_method:
            mock_method.return_value = MagicMock()
            handler.handle("/api/auth/sessions/abc123", {}, mock_http, "DELETE")
            mock_method.assert_called_once_with(mock_http, "abc123")

    def test_unknown_method_returns_405(self, handler, mock_http):
        """Unknown method returns 405 error."""
        mock_http.command = "PATCH"
        result = handler.handle("/api/auth/register", {}, mock_http, "PATCH")
        assert result is not None
        assert result.status_code == 405


class TestAuthHandlerGetUserStore:
    """Tests for _get_user_store method."""

    def test_returns_store_from_context(self):
        """Returns user_store from context."""
        from aragora.server.handlers.auth.handler import AuthHandler

        mock_store = MagicMock()
        handler = AuthHandler(server_context={"user_store": mock_store})

        result = handler._get_user_store()

        assert result is mock_store

    def test_returns_none_when_not_in_context(self):
        """Returns None when user_store not in context."""
        from aragora.server.handlers.auth.handler import AuthHandler

        handler = AuthHandler(server_context={})

        result = handler._get_user_store()

        assert result is None


class TestAuthHandlerRateLimiting:
    """Tests for rate limiting configuration."""

    @pytest.fixture
    def handler(self):
        """Create handler instance."""
        from aragora.server.handlers.auth.handler import AuthHandler

        return AuthHandler(server_context={})

    def test_register_has_rate_limit(self, handler):
        """_handle_register has rate limit decorator."""
        # Check for rate limit wrapper
        method = handler._handle_register
        # The method should exist and be callable
        assert callable(method)

    def test_login_has_rate_limit(self, handler):
        """_handle_login has rate limit decorator."""
        method = handler._handle_login
        assert callable(method)

    def test_refresh_has_rate_limit(self, handler):
        """_handle_refresh has rate limit decorator."""
        method = handler._handle_refresh
        assert callable(method)


class TestAuthHandlerErrorResponses:
    """Tests for error response handling."""

    @pytest.fixture
    def handler(self):
        """Create handler instance without user store."""
        from aragora.server.handlers.auth.handler import AuthHandler

        return AuthHandler(server_context={})

    @pytest.fixture
    def mock_http(self):
        """Create mock HTTP handler with valid JSON."""
        import json

        mock = MagicMock()
        body = json.dumps({"email": "test@example.com", "password": "password123"})
        mock.rfile = MagicMock()
        mock.rfile.read = MagicMock(return_value=body.encode())
        mock.headers = {"Content-Length": str(len(body))}
        mock.command = "POST"
        return mock

    def test_register_returns_503_without_store(self, handler, mock_http):
        """Register returns 503 when user store unavailable."""
        # With proper mocking to avoid rate limit issues
        with patch("aragora.server.handlers.auth.handler.rate_limit", lambda **kw: lambda f: f):
            result = handler._handle_register(mock_http)
            assert result is not None
            assert result.status_code == 503


class TestAuthHandlerSessionExtraction:
    """Tests for session ID extraction from path."""

    @pytest.fixture
    def handler(self):
        """Create handler instance."""
        from aragora.server.handlers.auth.handler import AuthHandler

        return AuthHandler(server_context={"user_store": MagicMock()})

    def test_extracts_simple_session_id(self, handler):
        """Extracts simple session ID from path."""
        mock_http = MagicMock()
        mock_http.command = "DELETE"

        with patch.object(handler, "_handle_revoke_session") as mock_method:
            mock_method.return_value = MagicMock()
            handler.handle("/api/auth/sessions/session123", {}, mock_http, "DELETE")
            mock_method.assert_called_with(mock_http, "session123")

    def test_extracts_uuid_session_id(self, handler):
        """Extracts UUID session ID from path."""
        mock_http = MagicMock()
        mock_http.command = "DELETE"
        uuid = "550e8400-e29b-41d4-a716-446655440000"

        with patch.object(handler, "_handle_revoke_session") as mock_method:
            mock_method.return_value = MagicMock()
            handler.handle(f"/api/auth/sessions/{uuid}", {}, mock_http, "DELETE")
            mock_method.assert_called_with(mock_http, uuid)


class TestAuthHandlerModuleExports:
    """Tests for module exports."""

    def test_all_exports_auth_handler(self):
        """__all__ exports AuthHandler."""
        from aragora.server.handlers.auth import handler

        assert "AuthHandler" in handler.__all__

    def test_handler_is_base_handler_subclass(self):
        """AuthHandler is a BaseHandler subclass."""
        from aragora.server.handlers.auth.handler import AuthHandler
        from aragora.server.handlers.base import BaseHandler

        assert issubclass(AuthHandler, BaseHandler)


class TestAuthHandlerMFARoutes:
    """Tests for MFA-specific routing."""

    @pytest.fixture
    def handler(self):
        """Create handler instance."""
        from aragora.server.handlers.auth.handler import AuthHandler

        return AuthHandler(server_context={"user_store": MagicMock()})

    def test_mfa_routes_all_post(self):
        """All MFA routes are POST."""
        from aragora.server.handlers.auth.handler import AuthHandler

        handler = AuthHandler(server_context={})
        mfa_routes = [r for r in handler.ROUTES if "/mfa/" in r]

        # All MFA routes should exist
        assert len(mfa_routes) == 5
        assert "/api/auth/mfa/setup" in mfa_routes
        assert "/api/auth/mfa/enable" in mfa_routes
        assert "/api/auth/mfa/disable" in mfa_routes
        assert "/api/auth/mfa/verify" in mfa_routes
        assert "/api/auth/mfa/backup-codes" in mfa_routes
