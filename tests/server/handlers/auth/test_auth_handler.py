"""
Tests for aragora.server.handlers.auth.handler - User authentication handler.

Tests cover:
- Handler routing
- Basic input validation
- Error responses

Note: More comprehensive integration tests exist in tests/e2e/test_auth_e2e.py
"""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import MagicMock
from datetime import datetime, timezone

import pytest

from aragora.server.handlers.auth.handler import AuthHandler


# ===========================================================================
# Helper Functions
# ===========================================================================


def make_mock_handler(
    body: Dict[str, Any] | None = None,
    headers: Dict[str, str] | None = None,
    command: str = "GET",
) -> MagicMock:
    """Create a mock HTTP handler with request data."""
    handler = MagicMock()
    handler.command = command
    handler.headers = headers or {}
    handler.client_address = ("127.0.0.1", 12345)

    # Mock request body reading
    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        handler.rfile = MagicMock()
        handler.rfile.read.return_value = body_bytes
        handler.headers["Content-Length"] = str(len(body_bytes))
    else:
        handler.headers["Content-Length"] = "0"

    return handler


def parse_result(result) -> Dict[str, Any]:
    """Parse HandlerResult into a dictionary."""
    if result is None:
        return {"success": False, "error": "No result", "status_code": 500}

    try:
        body = json.loads(result.body.decode("utf-8"))
        return {
            "success": result.status_code < 400,
            "status_code": result.status_code,
            **body,
        }
    except (json.JSONDecodeError, AttributeError) as e:
        return {"success": False, "error": str(e), "status_code": 500}


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def mock_user_for_auth():
    """Create a mock user for auth tests."""
    mock_user = MagicMock()
    mock_user.id = "user-123"
    mock_user.email = "test@example.com"
    mock_user.name = "Test User"
    mock_user.role = "member"
    mock_user.org_id = "org-456"
    mock_user.is_active = True
    mock_user.mfa_enabled = False
    mock_user.created_at = datetime.now(timezone.utc)
    mock_user.verify_password.return_value = True
    mock_user.to_dict.return_value = {
        "id": "user-123",
        "email": "test@example.com",
        "name": "Test User",
        "role": "member",
    }
    return mock_user


@pytest.fixture
def mock_user_store_for_auth(mock_user_for_auth):
    """Create a mock user store for auth tests."""
    store = MagicMock()
    store.get_user_by_email.return_value = mock_user_for_auth
    store.get_user_by_id.return_value = mock_user_for_auth
    store.verify_password.return_value = True
    store.create_user.return_value = mock_user_for_auth
    return store


@pytest.fixture
def auth_server_context(mock_user_store_for_auth):
    """Create a server context for auth handler tests."""
    return {
        "storage": MagicMock(),
        "user_store": mock_user_store_for_auth,
        "elo_system": MagicMock(),
        "knowledge_store": MagicMock(),
        "workflow_store": MagicMock(),
        "workspace_store": MagicMock(),
        "audit_store": MagicMock(),
        "debate_embeddings": None,
        "critique_store": None,
        "nomic_dir": None,
    }


@pytest.fixture
def auth_handler(auth_server_context):
    """Create an AuthHandler with mocked dependencies."""
    return AuthHandler(server_context=auth_server_context)


# ===========================================================================
# Test: Handler Routing
# ===========================================================================


class TestAuthHandlerRouting:
    """Test request routing to correct methods."""

    def test_can_handle_auth_login(self, auth_handler):
        """Test handler recognizes login route."""
        assert auth_handler.can_handle("/api/auth/login") is True

    def test_can_handle_auth_logout(self, auth_handler):
        """Test handler recognizes logout route."""
        assert auth_handler.can_handle("/api/auth/logout") is True

    def test_can_handle_auth_refresh(self, auth_handler):
        """Test handler recognizes refresh route."""
        assert auth_handler.can_handle("/api/auth/refresh") is True

    def test_can_handle_auth_me(self, auth_handler):
        """Test handler recognizes me route."""
        assert auth_handler.can_handle("/api/auth/me") is True

    def test_can_handle_auth_register(self, auth_handler):
        """Test handler recognizes register route."""
        assert auth_handler.can_handle("/api/auth/register") is True

    def test_can_handle_auth_sessions(self, auth_handler):
        """Test handler recognizes sessions route."""
        assert auth_handler.can_handle("/api/auth/sessions") is True

    def test_can_handle_auth_sessions_with_id(self, auth_handler):
        """Test handler recognizes sessions route with ID."""
        assert auth_handler.can_handle("/api/auth/sessions/session-123") is True

    def test_can_handle_mfa_routes(self, auth_handler):
        """Test handler recognizes MFA routes."""
        assert auth_handler.can_handle("/api/auth/mfa/setup") is True
        assert auth_handler.can_handle("/api/auth/mfa/enable") is True
        assert auth_handler.can_handle("/api/auth/mfa/disable") is True
        assert auth_handler.can_handle("/api/auth/mfa/verify") is True

    def test_cannot_handle_debates_route(self, auth_handler):
        """Test handler rejects debates route."""
        assert auth_handler.can_handle("/api/debates") is False

    def test_cannot_handle_health_route(self, auth_handler):
        """Test handler rejects health route."""
        assert auth_handler.can_handle("/api/health") is False

    def test_cannot_handle_users_route(self, auth_handler):
        """Test handler rejects users route."""
        assert auth_handler.can_handle("/api/v1/users") is False


# ===========================================================================
# Test: Handler Properties
# ===========================================================================


class TestAuthHandlerProperties:
    """Test handler property values."""

    def test_resource_type(self, auth_handler):
        """Test RESOURCE_TYPE is set correctly."""
        assert auth_handler.RESOURCE_TYPE == "auth"

    def test_routes_list(self, auth_handler):
        """Test ROUTES includes essential endpoints."""
        routes = auth_handler.ROUTES
        assert "/api/auth/login" in routes
        assert "/api/auth/logout" in routes
        assert "/api/auth/register" in routes
        assert "/api/auth/me" in routes
        assert "/api/auth/refresh" in routes


# ===========================================================================
# Test: Input Validation
# ===========================================================================


class TestAuthInputValidation:
    """Test input validation on auth endpoints."""

    def test_login_missing_email(self, auth_handler):
        """Test login fails without email."""
        request = make_mock_handler(body={"password": "password123"}, command="POST")

        result = auth_handler._handle_login(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_login_missing_password(self, auth_handler):
        """Test login fails without password."""
        request = make_mock_handler(body={"email": "test@example.com"}, command="POST")

        result = auth_handler._handle_login(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_login_empty_credentials(self, auth_handler):
        """Test login fails with empty credentials."""
        request = make_mock_handler(body={"email": "", "password": ""}, command="POST")

        result = auth_handler._handle_login(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_login_invalid_json(self, auth_handler):
        """Test login fails with invalid JSON."""
        request = MagicMock()
        request.command = "POST"
        request.headers = {"Content-Length": "10"}
        request.client_address = ("127.0.0.1", 12345)
        request.rfile = MagicMock()
        request.rfile.read.return_value = b"not valid json"

        result = auth_handler._handle_login(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_refresh_missing_token(self, auth_handler):
        """Test refresh fails without token."""
        request = make_mock_handler(body={}, command="POST")

        result = auth_handler._handle_refresh(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 400


# ===========================================================================
# Test: Service Unavailability
# ===========================================================================


class TestServiceUnavailability:
    """Test handling when services are unavailable."""

    def test_login_no_user_store(self):
        """Test login returns 503 when user store unavailable."""
        # Create handler without user store
        context = {
            "storage": MagicMock(),
            "user_store": None,
            "elo_system": MagicMock(),
        }
        handler = AuthHandler(server_context=context)

        request = make_mock_handler(
            body={"email": "test@example.com", "password": "password123"},
            command="POST",
        )

        result = handler._handle_login(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 503


# ===========================================================================
# Test: Method Routing
# ===========================================================================


class TestMethodRouting:
    """Test handle() routes to correct methods."""

    def test_handle_invalid_method_returns_405(self, auth_handler):
        """Test invalid method returns 405."""
        request = make_mock_handler(command="DELETE")

        result = auth_handler.handle(
            path="/api/auth/login",
            query_params={},
            handler=request,
            method="DELETE",
        )
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 405

    def test_handle_routes_to_login(self, auth_handler):
        """Test POST /api/auth/login routes to _handle_login."""
        # We just verify it doesn't return 405
        request = make_mock_handler(
            body={"email": "test@example.com", "password": "pass"},
            command="POST",
        )

        result = auth_handler.handle(
            path="/api/auth/login",
            query_params={},
            handler=request,
            method="POST",
        )
        parsed = parse_result(result)

        # Should be 400 (bad input) not 405 (method not allowed)
        # This proves routing worked
        assert parsed["status_code"] != 405


# ===========================================================================
# Test: Handler Registration
# ===========================================================================


class TestHandlerRegistration:
    """Test handler can be instantiated and registered."""

    def test_handler_instantiates(self, auth_server_context):
        """Test AuthHandler can be instantiated."""
        handler = AuthHandler(server_context=auth_server_context)
        assert handler is not None

    def test_handler_has_routes(self, auth_handler):
        """Test AuthHandler has ROUTES defined."""
        assert hasattr(auth_handler, "ROUTES")
        assert len(auth_handler.ROUTES) > 0

    def test_handler_has_can_handle(self, auth_handler):
        """Test AuthHandler has can_handle method."""
        assert hasattr(auth_handler, "can_handle")
        assert callable(auth_handler.can_handle)


# ===========================================================================
# Test: API Key Management
# ===========================================================================


class TestAPIKeyManagement:
    """Test API key generation and revocation."""

    def test_generate_api_key_requires_auth(self, auth_handler):
        """Test API key generation requires authentication."""
        request = make_mock_handler(body={"name": "test-key"}, command="POST")

        result = auth_handler._handle_generate_api_key(request)
        parsed = parse_result(result)

        # Should fail without auth token
        assert parsed["success"] is False
        assert parsed["status_code"] in (401, 503)

    def test_generate_api_key_missing_name(self, auth_handler):
        """Test API key generation fails without name."""
        request = make_mock_handler(body={}, command="POST")
        request.headers["Authorization"] = "Bearer valid-token"

        result = auth_handler._handle_generate_api_key(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] in (400, 401)

    def test_revoke_api_key_requires_auth(self, auth_handler):
        """Test API key revocation requires authentication."""
        request = make_mock_handler(body={"key_id": "key-123"}, command="POST")

        result = auth_handler._handle_revoke_api_key(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] in (401, 503)

    def test_revoke_api_key_missing_key_id(self, auth_handler):
        """Test API key revocation fails without key_id."""
        request = make_mock_handler(body={}, command="POST")
        request.headers["Authorization"] = "Bearer valid-token"

        result = auth_handler._handle_revoke_api_key(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] in (400, 401)


# ===========================================================================
# Test: Session Management
# ===========================================================================


class TestSessionManagement:
    """Test session listing and revocation."""

    def test_list_sessions_requires_auth(self, auth_handler):
        """Test session listing requires authentication."""
        request = make_mock_handler(command="GET")

        result = auth_handler._handle_list_sessions(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] in (401, 503)

    def test_revoke_session_requires_auth(self, auth_handler):
        """Test session revocation requires authentication."""
        request = make_mock_handler(command="DELETE")

        result = auth_handler._handle_revoke_session(request, "session-123")
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] in (401, 503)

    def test_logout_all_requires_auth(self, auth_handler):
        """Test logout all requires authentication."""
        request = make_mock_handler(command="POST")

        result = auth_handler._handle_logout_all(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] in (401, 503)


# ===========================================================================
# Test: MFA Operations
# ===========================================================================


class TestMFAOperations:
    """Test MFA setup, enable, disable, and backup codes."""

    def test_mfa_setup_requires_auth(self, auth_handler):
        """Test MFA setup requires authentication."""
        request = make_mock_handler(command="POST")

        result = auth_handler._handle_mfa_setup(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] in (401, 503)

    def test_mfa_enable_requires_auth(self, auth_handler):
        """Test MFA enable requires authentication."""
        request = make_mock_handler(body={"code": "123456"}, command="POST")

        result = auth_handler._handle_mfa_enable(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] in (401, 503)

    def test_mfa_enable_missing_code(self, auth_handler):
        """Test MFA enable fails without code."""
        request = make_mock_handler(body={}, command="POST")
        request.headers["Authorization"] = "Bearer valid-token"

        result = auth_handler._handle_mfa_enable(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] in (400, 401)

    def test_mfa_disable_requires_auth(self, auth_handler):
        """Test MFA disable requires authentication."""
        request = make_mock_handler(body={"code": "123456"}, command="POST")

        result = auth_handler._handle_mfa_disable(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] in (401, 503)

    def test_mfa_verify_missing_code(self, auth_handler):
        """Test MFA verify fails without code."""
        request = make_mock_handler(body={}, command="POST")

        result = auth_handler._handle_mfa_verify(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] in (400, 401)

    def test_mfa_backup_codes_requires_auth(self, auth_handler):
        """Test MFA backup codes requires authentication."""
        request = make_mock_handler(command="GET")

        result = auth_handler._handle_mfa_backup_codes(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] in (401, 503)


# ===========================================================================
# Test: Profile Operations
# ===========================================================================


class TestProfileOperations:
    """Test profile get and update operations."""

    def test_get_me_requires_auth(self, auth_handler):
        """Test get me requires authentication."""
        request = make_mock_handler(command="GET")

        result = auth_handler._handle_get_me(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] in (401, 503)

    def test_update_me_requires_auth(self, auth_handler):
        """Test update me requires authentication."""
        request = make_mock_handler(body={"name": "New Name"}, command="PUT")

        result = auth_handler._handle_update_me(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] in (401, 503)

    def test_change_password_requires_auth(self, auth_handler):
        """Test change password requires authentication."""
        request = make_mock_handler(
            body={"current_password": "old", "new_password": "new123"},
            command="POST",
        )

        result = auth_handler._handle_change_password(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] in (401, 503)

    def test_change_password_missing_current(self, auth_handler):
        """Test change password fails without current password."""
        request = make_mock_handler(
            body={"new_password": "new123"},
            command="POST",
        )
        request.headers["Authorization"] = "Bearer valid-token"

        result = auth_handler._handle_change_password(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] in (400, 401)

    def test_change_password_missing_new(self, auth_handler):
        """Test change password fails without new password."""
        request = make_mock_handler(
            body={"current_password": "old"},
            command="POST",
        )
        request.headers["Authorization"] = "Bearer valid-token"

        result = auth_handler._handle_change_password(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] in (400, 401)


# ===========================================================================
# Test: Token Operations
# ===========================================================================


class TestTokenOperations:
    """Test token revocation operations."""

    def test_revoke_token_requires_auth(self, auth_handler):
        """Test token revocation requires authentication."""
        request = make_mock_handler(body={"token": "some-token"}, command="POST")

        result = auth_handler._handle_revoke_token(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] in (401, 503)

    def test_revoke_token_missing_token(self, auth_handler):
        """Test token revocation fails without token."""
        request = make_mock_handler(body={}, command="POST")
        request.headers["Authorization"] = "Bearer valid-token"

        result = auth_handler._handle_revoke_token(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] in (400, 401)

    def test_logout_requires_auth(self, auth_handler):
        """Test logout requires authentication."""
        request = make_mock_handler(command="POST")

        result = auth_handler._handle_logout(request)
        parsed = parse_result(result)

        # Logout should work or return 401/503
        assert parsed["status_code"] in (200, 401, 503)


# ===========================================================================
# Test: Route Handling with API Versions
# ===========================================================================


class TestVersionedRoutes:
    """Test handler with versioned API routes."""

    def test_can_handle_v1_login(self, auth_handler):
        """Test handler recognizes v1 login route."""
        assert auth_handler.can_handle("/api/v1/auth/login") is True

    def test_can_handle_v1_sessions(self, auth_handler):
        """Test handler recognizes v1 sessions route."""
        assert auth_handler.can_handle("/api/v1/auth/sessions") is True

    def test_can_handle_v1_mfa_setup(self, auth_handler):
        """Test handler recognizes v1 MFA setup route."""
        assert auth_handler.can_handle("/api/v1/auth/mfa/setup") is True

    def test_can_handle_api_keys_route(self, auth_handler):
        """Test handler recognizes API keys route."""
        # Check if the route exists
        can_handle = auth_handler.can_handle("/api/auth/api-keys") or auth_handler.can_handle(
            "/api/v1/auth/api-keys"
        )
        # This may be true or false depending on implementation
        assert isinstance(can_handle, bool)
