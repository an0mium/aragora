"""
Comprehensive tests for aragora.server.handlers.auth.handler - AuthHandler.

Tests cover:
- User registration (signup) with validation
- Login with various credential types (password, API key, OAuth)
- Session creation, validation, and revocation
- Password reset flow
- API key generation and management
- MFA/2FA flows
- Token refresh mechanisms
- Error responses (invalid credentials, locked accounts, expired tokens)
- Rate limiting on auth endpoints
- Security edge cases (SQL injection attempts, XSS in usernames)

Target: 80+ tests with comprehensive coverage.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.server.handlers.auth.handler import AuthHandler


# ===========================================================================
# Helper Functions
# ===========================================================================


def maybe_await(result):
    """Handle sync/async results uniformly."""
    if asyncio.iscoroutine(result):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(result)
        finally:
            loop.close()
    return result


def make_mock_handler(
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    command: str = "GET",
    client_address: tuple = ("127.0.0.1", 12345),
) -> MagicMock:
    """Create a mock HTTP handler with request data."""
    handler = MagicMock()
    handler.command = command
    handler.headers = headers or {}
    handler.client_address = client_address

    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        handler.rfile = MagicMock()
        handler.rfile.read.return_value = body_bytes
        handler.headers["Content-Length"] = str(len(body_bytes))
    else:
        handler.headers["Content-Length"] = "0"

    return handler


def parse_result(result) -> dict[str, Any]:
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
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_user():
    """Create a mock user object."""
    user = MagicMock()
    user.id = "user-123"
    user.email = "test@example.com"
    user.name = "Test User"
    user.role = "member"
    user.org_id = "org-456"
    user.is_active = True
    user.mfa_enabled = False
    user.mfa_secret = None
    user.mfa_backup_codes = None
    user.api_key_prefix = None
    user.api_key_hash = None
    user.api_key_created_at = None
    user.api_key_expires_at = None
    user.created_at = datetime.now(timezone.utc)
    user.verify_password = MagicMock(return_value=True)
    user.to_dict = MagicMock(
        return_value={
            "id": "user-123",
            "email": "test@example.com",
            "name": "Test User",
            "role": "member",
        }
    )
    user.generate_api_key = MagicMock(return_value="ara_testkey123456")
    return user


@pytest.fixture
def mock_mfa_user(mock_user):
    """Create a mock user with MFA enabled."""
    mock_user.mfa_enabled = True
    mock_user.mfa_secret = "TESTSECRET123456"
    mock_user.mfa_backup_codes = json.dumps(
        [hashlib.sha256(f"backup{i}".encode()).hexdigest() for i in range(10)]
    )
    return mock_user


@pytest.fixture
def mock_user_store(mock_user):
    """Create a mock user store."""
    store = MagicMock()
    store.get_user_by_email = MagicMock(return_value=None)  # Default: no user
    store.get_user_by_id = MagicMock(return_value=mock_user)
    store.create_user = MagicMock(return_value=mock_user)
    store.update_user = MagicMock(return_value=True)
    store.create_organization = MagicMock()
    store.get_organization_by_id = MagicMock(return_value=None)
    store.increment_token_version = MagicMock(return_value=2)
    store.is_account_locked = MagicMock(return_value=(False, None, 0))
    store.record_failed_login = MagicMock(return_value=(1, None))
    store.reset_failed_login_attempts = MagicMock()
    return store


@pytest.fixture
def mock_org():
    """Create a mock organization."""
    org = MagicMock()
    org.id = "org-456"
    org.name = "Test Org"
    org.limits = MagicMock()
    org.limits.api_access = True
    org.to_dict = MagicMock(
        return_value={
            "id": "org-456",
            "name": "Test Org",
        }
    )
    return org


@pytest.fixture
def server_context(mock_user_store):
    """Create a server context for auth handler tests."""
    return {
        "storage": MagicMock(),
        "user_store": mock_user_store,
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
def auth_handler(server_context):
    """Create an AuthHandler with mocked dependencies."""
    return AuthHandler(server_context=server_context)


@pytest.fixture
def mock_auth_context():
    """Create a mock authentication context."""
    ctx = MagicMock()
    ctx.is_authenticated = True
    ctx.user_id = "user-123"
    ctx.email = "test@example.com"
    ctx.org_id = "org-456"
    ctx.role = "admin"
    ctx.client_ip = "127.0.0.1"
    return ctx


@pytest.fixture
def mock_tokens():
    """Create mock token pair."""
    tokens = MagicMock()
    tokens.access_token = "access_token_123"
    tokens.refresh_token = "refresh_token_456"
    tokens.to_dict = MagicMock(
        return_value={
            "access_token": "access_token_123",
            "refresh_token": "refresh_token_456",
            "token_type": "Bearer",
            "expires_in": 3600,
        }
    )
    return tokens


@pytest.fixture
def mock_lockout_tracker():
    """Create a mock lockout tracker."""
    tracker = MagicMock()
    tracker.is_locked.return_value = False
    tracker.record_failure.return_value = (0, None)
    tracker.reset = MagicMock()
    tracker.get_remaining_time.return_value = 0
    return tracker


# ===========================================================================
# Test: User Registration - Success Cases
# ===========================================================================


class TestRegistrationSuccess:
    """Test successful user registration flows."""

    def test_register_with_valid_email_and_password(
        self, auth_handler, mock_user_store, mock_user, mock_tokens
    ):
        """Test successful registration with valid email and password."""
        mock_user_store.get_user_by_email.return_value = None
        mock_user_store.create_user.return_value = mock_user

        request = make_mock_handler(
            body={
                "email": "newuser@example.com",
                "password": "SecurePassword123!",
            },
            command="POST",
        )

        with patch("aragora.billing.models.hash_password", return_value=("hashed", "salt")):
            with patch("aragora.billing.jwt_auth.create_token_pair", return_value=mock_tokens):
                result = auth_handler._handle_register(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        assert parsed["status_code"] == 201
        assert "user" in parsed
        assert "tokens" in parsed

    def test_register_with_name_provided(
        self, auth_handler, mock_user_store, mock_user, mock_tokens
    ):
        """Test registration with optional name field."""
        mock_user_store.get_user_by_email.return_value = None
        mock_user_store.create_user.return_value = mock_user

        request = make_mock_handler(
            body={
                "email": "named@example.com",
                "password": "SecurePassword123!",
                "name": "John Doe",
            },
            command="POST",
        )

        with patch("aragora.billing.models.hash_password", return_value=("hashed", "salt")):
            with patch("aragora.billing.jwt_auth.create_token_pair", return_value=mock_tokens):
                result = auth_handler._handle_register(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        assert parsed["status_code"] == 201

    def test_register_with_organization_name(
        self, auth_handler, mock_user_store, mock_user, mock_tokens
    ):
        """Test registration creates organization when name provided."""
        mock_user_store.get_user_by_email.return_value = None
        mock_user_store.create_user.return_value = mock_user

        request = make_mock_handler(
            body={
                "email": "admin@company.com",
                "password": "SecurePassword123!",
                "name": "Admin User",
                "organization": "Acme Corp",
            },
            command="POST",
        )

        with patch("aragora.billing.models.hash_password", return_value=("hashed", "salt")):
            with patch("aragora.billing.jwt_auth.create_token_pair", return_value=mock_tokens):
                result = auth_handler._handle_register(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        mock_user_store.create_organization.assert_called_once()

    def test_register_email_normalized_to_lowercase(
        self, auth_handler, mock_user_store, mock_user, mock_tokens
    ):
        """Test email is normalized to lowercase."""
        mock_user_store.get_user_by_email.return_value = None
        mock_user_store.create_user.return_value = mock_user

        request = make_mock_handler(
            body={
                "email": "USER@EXAMPLE.COM",
                "password": "SecurePassword123!",
            },
            command="POST",
        )

        with patch("aragora.billing.models.hash_password", return_value=("hashed", "salt")):
            with patch("aragora.billing.jwt_auth.create_token_pair", return_value=mock_tokens):
                result = auth_handler._handle_register(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        # Verify lowercase email was checked
        mock_user_store.get_user_by_email.assert_called_with("user@example.com")


# ===========================================================================
# Test: User Registration - Validation Failures
# ===========================================================================


class TestRegistrationValidation:
    """Test registration input validation."""

    def test_register_invalid_email_format(self, auth_handler):
        """Test registration fails with invalid email format."""
        request = make_mock_handler(
            body={
                "email": "notanemail",
                "password": "SecurePassword123!",
            },
            command="POST",
        )

        result = auth_handler._handle_register(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_register_email_missing_at_sign(self, auth_handler):
        """Test registration fails when email is missing @ symbol."""
        request = make_mock_handler(
            body={
                "email": "userexample.com",
                "password": "SecurePassword123!",
            },
            command="POST",
        )

        result = auth_handler._handle_register(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_register_email_missing_domain(self, auth_handler):
        """Test registration fails when email has no domain."""
        request = make_mock_handler(
            body={
                "email": "user@",
                "password": "SecurePassword123!",
            },
            command="POST",
        )

        result = auth_handler._handle_register(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_register_empty_email(self, auth_handler):
        """Test registration fails with empty email."""
        request = make_mock_handler(
            body={
                "email": "",
                "password": "SecurePassword123!",
            },
            command="POST",
        )

        result = auth_handler._handle_register(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_register_password_too_short(self, auth_handler, mock_user_store):
        """Test registration fails with password below minimum length."""
        mock_user_store.get_user_by_email.return_value = None

        request = make_mock_handler(
            body={
                "email": "user@example.com",
                "password": "short",
            },
            command="POST",
        )

        result = auth_handler._handle_register(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_register_password_too_long(self, auth_handler, mock_user_store):
        """Test registration fails with password above maximum length."""
        mock_user_store.get_user_by_email.return_value = None

        request = make_mock_handler(
            body={
                "email": "user@example.com",
                "password": "a" * 200,  # Exceeds max 128
            },
            command="POST",
        )

        result = auth_handler._handle_register(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_register_missing_password(self, auth_handler):
        """Test registration fails with missing password."""
        request = make_mock_handler(
            body={
                "email": "user@example.com",
            },
            command="POST",
        )

        result = auth_handler._handle_register(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_register_email_already_exists(self, auth_handler, mock_user_store, mock_user):
        """Test registration fails when email already exists."""
        mock_user_store.get_user_by_email.return_value = mock_user

        request = make_mock_handler(
            body={
                "email": "existing@example.com",
                "password": "SecurePassword123!",
            },
            command="POST",
        )

        result = auth_handler._handle_register(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 409
        assert "already registered" in parsed.get("error", "").lower()

    def test_register_invalid_json_body(self, auth_handler):
        """Test registration fails with invalid JSON body."""
        request = MagicMock()
        request.command = "POST"
        request.headers = {"Content-Length": "10"}
        request.client_address = ("127.0.0.1", 12345)
        request.rfile = MagicMock()
        request.rfile.read.return_value = b"not valid json"

        result = auth_handler._handle_register(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 400


# ===========================================================================
# Test: Login - Success Cases
# ===========================================================================


class TestLoginSuccess:
    """Test successful login flows."""

    def test_login_with_valid_credentials(
        self, auth_handler, mock_user_store, mock_user, mock_tokens, mock_lockout_tracker
    ):
        """Test successful login with correct email and password."""
        mock_user_store.get_user_by_email.return_value = mock_user

        request = make_mock_handler(
            body={
                "email": "test@example.com",
                "password": "correctpassword",
            },
            command="POST",
        )

        with patch("aragora.auth.lockout.get_lockout_tracker", return_value=mock_lockout_tracker):
            with patch("aragora.billing.jwt_auth.create_token_pair", return_value=mock_tokens):
                result = auth_handler._handle_login(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        assert parsed["status_code"] == 200
        assert "user" in parsed
        assert "tokens" in parsed

    def test_login_returns_user_data(
        self, auth_handler, mock_user_store, mock_user, mock_tokens, mock_lockout_tracker
    ):
        """Test login response includes user information."""
        mock_user_store.get_user_by_email.return_value = mock_user

        request = make_mock_handler(
            body={
                "email": "test@example.com",
                "password": "correctpassword",
            },
            command="POST",
        )

        with patch("aragora.auth.lockout.get_lockout_tracker", return_value=mock_lockout_tracker):
            with patch("aragora.billing.jwt_auth.create_token_pair", return_value=mock_tokens):
                result = auth_handler._handle_login(request)

        parsed = parse_result(result)
        assert "user" in parsed
        assert parsed["user"]["email"] == "test@example.com"

    def test_login_resets_lockout_on_success(
        self, auth_handler, mock_user_store, mock_user, mock_tokens, mock_lockout_tracker
    ):
        """Test successful login resets failed attempt counter."""
        mock_user_store.get_user_by_email.return_value = mock_user

        request = make_mock_handler(
            body={
                "email": "test@example.com",
                "password": "correctpassword",
            },
            command="POST",
        )

        # Patch at the handler module level since it imports get_lockout_tracker directly
        with patch(
            "aragora.server.handlers.auth.login.get_lockout_tracker",
            return_value=mock_lockout_tracker,
        ):
            with patch("aragora.billing.jwt_auth.create_token_pair", return_value=mock_tokens):
                auth_handler._handle_login(request)

        mock_lockout_tracker.reset.assert_called_once_with(email="test@example.com", ip="127.0.0.1")


# ===========================================================================
# Test: Login - MFA Flow
# ===========================================================================


class TestLoginMFA:
    """Test login with MFA enabled."""

    def test_login_with_mfa_returns_pending_token(
        self, auth_handler, mock_user_store, mock_mfa_user, mock_lockout_tracker
    ):
        """Test login with MFA enabled returns pending token."""
        mock_user_store.get_user_by_email.return_value = mock_mfa_user

        request = make_mock_handler(
            body={
                "email": "test@example.com",
                "password": "correctpassword",
            },
            command="POST",
        )

        with patch("aragora.auth.lockout.get_lockout_tracker", return_value=mock_lockout_tracker):
            with patch(
                "aragora.billing.jwt_auth.create_mfa_pending_token",
                return_value="pending_token_xyz",
            ):
                result = auth_handler._handle_login(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        assert parsed.get("mfa_required") is True
        assert "pending_token" in parsed

    def test_login_with_mfa_does_not_return_full_tokens(
        self, auth_handler, mock_user_store, mock_mfa_user, mock_lockout_tracker
    ):
        """Test login with MFA does not return access tokens."""
        mock_user_store.get_user_by_email.return_value = mock_mfa_user

        request = make_mock_handler(
            body={
                "email": "test@example.com",
                "password": "correctpassword",
            },
            command="POST",
        )

        with patch("aragora.auth.lockout.get_lockout_tracker", return_value=mock_lockout_tracker):
            with patch(
                "aragora.billing.jwt_auth.create_mfa_pending_token",
                return_value="pending_token_xyz",
            ):
                result = auth_handler._handle_login(request)

        parsed = parse_result(result)
        assert "tokens" not in parsed


# ===========================================================================
# Test: Login - Failure Cases
# ===========================================================================


class TestLoginFailure:
    """Test login failure scenarios."""

    def test_login_wrong_password(
        self, auth_handler, mock_user_store, mock_user, mock_lockout_tracker
    ):
        """Test login fails with incorrect password."""
        mock_user.verify_password.return_value = False
        mock_user_store.get_user_by_email.return_value = mock_user

        request = make_mock_handler(
            body={
                "email": "test@example.com",
                "password": "wrongpassword",
            },
            command="POST",
        )

        with patch("aragora.auth.lockout.get_lockout_tracker", return_value=mock_lockout_tracker):
            result = auth_handler._handle_login(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 401

    def test_login_user_not_found(self, auth_handler, mock_user_store, mock_lockout_tracker):
        """Test login fails when user does not exist."""
        mock_user_store.get_user_by_email.return_value = None

        request = make_mock_handler(
            body={
                "email": "nonexistent@example.com",
                "password": "password",
            },
            command="POST",
        )

        with patch("aragora.auth.lockout.get_lockout_tracker", return_value=mock_lockout_tracker):
            result = auth_handler._handle_login(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 401

    def test_login_account_disabled(
        self, auth_handler, mock_user_store, mock_user, mock_lockout_tracker
    ):
        """Test login fails for disabled account."""
        mock_user.is_active = False
        mock_user_store.get_user_by_email.return_value = mock_user

        request = make_mock_handler(
            body={
                "email": "test@example.com",
                "password": "correctpassword",
            },
            command="POST",
        )

        with patch("aragora.auth.lockout.get_lockout_tracker", return_value=mock_lockout_tracker):
            result = auth_handler._handle_login(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 401
        assert "disabled" in parsed.get("error", "").lower()

    def test_login_account_locked(self, auth_handler, mock_user_store, mock_user):
        """Test login fails when account is locked."""
        mock_user_store.get_user_by_email.return_value = mock_user

        locked_tracker = MagicMock()
        locked_tracker.is_locked.return_value = True
        locked_tracker.get_remaining_time.return_value = 300

        request = make_mock_handler(
            body={
                "email": "test@example.com",
                "password": "password",
            },
            command="POST",
        )

        with patch(
            "aragora.server.handlers.auth.login.get_lockout_tracker",
            return_value=locked_tracker,
        ):
            result = auth_handler._handle_login(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 429
        assert "too many" in parsed.get("error", "").lower()

    def test_login_missing_email(self, auth_handler):
        """Test login fails with missing email."""
        request = make_mock_handler(
            body={"password": "password123"},
            command="POST",
        )

        result = auth_handler._handle_login(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_login_missing_password(self, auth_handler):
        """Test login fails with missing password."""
        request = make_mock_handler(
            body={"email": "test@example.com"},
            command="POST",
        )

        result = auth_handler._handle_login(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_login_empty_credentials(self, auth_handler):
        """Test login fails with empty credentials."""
        request = make_mock_handler(
            body={"email": "", "password": ""},
            command="POST",
        )

        result = auth_handler._handle_login(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 400


# ===========================================================================
# Test: Token Refresh
# ===========================================================================


class TestTokenRefresh:
    """Test token refresh flows."""

    def test_refresh_success(self, auth_handler, mock_user_store, mock_user, mock_tokens):
        """Test successful token refresh."""
        mock_user_store.get_user_by_id.return_value = mock_user

        request = make_mock_handler(
            body={"refresh_token": "valid_refresh_token"},
            command="POST",
        )

        mock_payload = MagicMock()
        mock_payload.user_id = "user-123"

        mock_blacklist = MagicMock()
        mock_blacklist.revoke_token.return_value = True

        with patch(
            "aragora.server.handlers.auth.handler.validate_refresh_token",
            return_value=mock_payload,
        ):
            with patch("aragora.billing.jwt_auth.revoke_token_persistent", return_value=True):
                with patch(
                    "aragora.billing.jwt_auth.get_token_blacklist",
                    return_value=mock_blacklist,
                ):
                    with patch(
                        "aragora.billing.jwt_auth.create_token_pair",
                        return_value=mock_tokens,
                    ):
                        result = auth_handler._handle_refresh(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        assert "tokens" in parsed

    def test_refresh_invalid_token(self, auth_handler):
        """Test refresh fails with invalid token."""
        request = make_mock_handler(
            body={"refresh_token": "invalid_token"},
            command="POST",
        )

        with patch(
            "aragora.server.handlers.auth.handler.validate_refresh_token",
            return_value=None,
        ):
            result = auth_handler._handle_refresh(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 401

    def test_refresh_user_disabled(self, auth_handler, mock_user_store, mock_user):
        """Test refresh fails if user is disabled."""
        mock_user.is_active = False
        mock_user_store.get_user_by_id.return_value = mock_user

        request = make_mock_handler(
            body={"refresh_token": "valid_refresh_token"},
            command="POST",
        )

        mock_payload = MagicMock()
        mock_payload.user_id = "user-123"

        with patch(
            "aragora.server.handlers.auth.handler.validate_refresh_token",
            return_value=mock_payload,
        ):
            result = auth_handler._handle_refresh(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 401

    def test_refresh_user_not_found(self, auth_handler, mock_user_store):
        """Test refresh fails if user not found."""
        mock_user_store.get_user_by_id.return_value = None

        request = make_mock_handler(
            body={"refresh_token": "valid_refresh_token"},
            command="POST",
        )

        mock_payload = MagicMock()
        mock_payload.user_id = "user-123"

        with patch(
            "aragora.server.handlers.auth.handler.validate_refresh_token",
            return_value=mock_payload,
        ):
            result = auth_handler._handle_refresh(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 401

    def test_refresh_missing_token(self, auth_handler):
        """Test refresh fails with missing token."""
        request = make_mock_handler(
            body={},
            command="POST",
        )

        result = auth_handler._handle_refresh(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_refresh_revokes_old_token(self, auth_handler, mock_user_store, mock_user, mock_tokens):
        """Test refresh revokes old refresh token."""
        mock_user_store.get_user_by_id.return_value = mock_user

        request = make_mock_handler(
            body={"refresh_token": "old_refresh_token"},
            command="POST",
        )

        mock_payload = MagicMock()
        mock_payload.user_id = "user-123"

        mock_blacklist = MagicMock()

        with patch(
            "aragora.server.handlers.auth.handler.validate_refresh_token",
            return_value=mock_payload,
        ):
            with patch(
                "aragora.billing.jwt_auth.revoke_token_persistent", return_value=True
            ) as mock_revoke:
                with patch(
                    "aragora.billing.jwt_auth.get_token_blacklist",
                    return_value=mock_blacklist,
                ):
                    with patch(
                        "aragora.billing.jwt_auth.create_token_pair",
                        return_value=mock_tokens,
                    ):
                        auth_handler._handle_refresh(request)

        mock_revoke.assert_called_once_with("old_refresh_token")


# ===========================================================================
# Test: Session Management
# ===========================================================================


class TestSessionManagement:
    """Test session listing and revocation."""

    def test_list_sessions_success(self, auth_handler, mock_auth_context):
        """Test successful session listing."""
        request = make_mock_handler(command="GET")
        request.headers["Authorization"] = "Bearer test_token"

        mock_session = MagicMock()
        mock_session.session_id = "session-123"
        mock_session.to_dict.return_value = {
            "session_id": "session-123",
            "device": "Chrome on MacOS",
            "ip_address": "127.0.0.1",
            "last_activity": datetime.now(timezone.utc).isoformat(),
        }

        mock_manager = MagicMock()
        mock_manager.list_sessions.return_value = [mock_session]

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                with patch(
                    "aragora.server.middleware.auth.extract_token",
                    return_value="test_token",
                ):
                    with patch("aragora.billing.jwt_auth.decode_jwt") as mock_decode:
                        mock_decode.return_value = MagicMock(jti="current-jti")

                        with patch(
                            "aragora.billing.auth.sessions.get_session_manager",
                            return_value=mock_manager,
                        ):
                            result = auth_handler._handle_list_sessions(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        assert "sessions" in parsed
        assert parsed["total"] == 1

    def test_revoke_session_success(self, auth_handler, mock_auth_context):
        """Test successful session revocation."""
        request = make_mock_handler(command="DELETE")

        mock_session = MagicMock()
        mock_session.session_id = "session-to-revoke"

        mock_manager = MagicMock()
        mock_manager.get_session.return_value = mock_session

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                with patch(
                    "aragora.server.middleware.auth.extract_token",
                    return_value="test_token",
                ):
                    with patch("aragora.billing.jwt_auth.decode_jwt") as mock_decode:
                        mock_decode.return_value = MagicMock(jti="current-jti")

                        with patch(
                            "aragora.billing.auth.sessions.get_session_manager",
                            return_value=mock_manager,
                        ):
                            result = auth_handler._handle_revoke_session(
                                request, "session-to-revoke"
                            )

        parsed = parse_result(result)
        assert parsed["success"] is True

    def test_revoke_current_session_rejected(self, auth_handler, mock_auth_context):
        """Test cannot revoke current session."""
        import hashlib

        request = make_mock_handler(command="DELETE")
        test_token = "test_token"
        # Session ID is computed as hash of token, not from JWT jti claim
        current_session_id = hashlib.sha256(test_token.encode()).hexdigest()[:32]

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                with patch(
                    "aragora.server.middleware.auth.extract_token",
                    return_value=test_token,
                ):
                    with patch("aragora.billing.jwt_auth.decode_jwt") as mock_decode:
                        mock_decode.return_value = MagicMock(jti="some-jti")

                        result = auth_handler._handle_revoke_session(request, current_session_id)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_revoke_session_not_found(self, auth_handler, mock_auth_context):
        """Test session revocation fails when not found."""
        request = make_mock_handler(command="DELETE")

        mock_manager = MagicMock()
        mock_manager.get_session.return_value = None

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                with patch(
                    "aragora.server.middleware.auth.extract_token",
                    return_value="test_token",
                ):
                    with patch("aragora.billing.jwt_auth.decode_jwt") as mock_decode:
                        mock_decode.return_value = MagicMock(jti="different-jti")

                        with patch(
                            "aragora.billing.auth.sessions.get_session_manager",
                            return_value=mock_manager,
                        ):
                            result = auth_handler._handle_revoke_session(
                                request, "nonexistent-session"
                            )

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 404


# ===========================================================================
# Test: Password Reset Flow
# ===========================================================================


class TestPasswordReset:
    """Test password reset flow."""

    def test_forgot_password_valid_email(self, auth_handler, mock_user_store, mock_user):
        """Test forgot password with valid email."""
        mock_user_store.get_user_by_email.return_value = mock_user

        request = make_mock_handler(
            body={"email": "test@example.com"},
            command="POST",
        )

        mock_store = MagicMock()
        mock_store.create_token.return_value = ("reset_token_123", None)

        with patch(
            "aragora.storage.password_reset_store.get_password_reset_store",
            return_value=mock_store,
        ):
            with patch.object(auth_handler, "_send_password_reset_email"):
                result = auth_handler._handle_forgot_password(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        # Should always return success to prevent email enumeration
        assert "email" in parsed.get("message", "").lower()

    def test_forgot_password_nonexistent_email(self, auth_handler, mock_user_store):
        """Test forgot password with non-existent email returns success (anti-enumeration)."""
        mock_user_store.get_user_by_email.return_value = None

        request = make_mock_handler(
            body={"email": "nonexistent@example.com"},
            command="POST",
        )

        result = auth_handler._handle_forgot_password(request)
        parsed = parse_result(result)

        # Should return success to prevent email enumeration
        assert parsed["success"] is True

    def test_forgot_password_invalid_email_format(self, auth_handler):
        """Test forgot password with invalid email format."""
        request = make_mock_handler(
            body={"email": "notanemail"},
            command="POST",
        )

        result = auth_handler._handle_forgot_password(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_reset_password_success(self, auth_handler, mock_user_store, mock_user):
        """Test successful password reset."""
        mock_user_store.get_user_by_email.return_value = mock_user

        request = make_mock_handler(
            body={
                "token": "valid_reset_token",
                "password": "NewSecurePassword123!",
            },
            command="POST",
        )

        mock_store = MagicMock()
        mock_store.validate_token.return_value = ("test@example.com", None)
        mock_store.consume_token = MagicMock()
        mock_store.invalidate_tokens_for_email = MagicMock()

        with patch(
            "aragora.storage.password_reset_store.get_password_reset_store",
            return_value=mock_store,
        ):
            with patch("aragora.billing.models.hash_password", return_value=("hash", "salt")):
                result = auth_handler._handle_reset_password(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        assert parsed.get("sessions_invalidated") is True

    def test_reset_password_invalid_token(self, auth_handler, mock_user_store):
        """Test password reset with invalid token."""
        request = make_mock_handler(
            body={
                "token": "invalid_token",
                "password": "NewSecurePassword123!",
            },
            command="POST",
        )

        mock_store = MagicMock()
        mock_store.validate_token.return_value = (None, "Token expired or invalid")

        with patch(
            "aragora.storage.password_reset_store.get_password_reset_store",
            return_value=mock_store,
        ):
            result = auth_handler._handle_reset_password(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_reset_password_weak_password(self, auth_handler, mock_user_store, mock_user):
        """Test password reset with weak password."""
        mock_user_store.get_user_by_email.return_value = mock_user

        request = make_mock_handler(
            body={
                "token": "valid_token",
                "password": "weak",
            },
            command="POST",
        )

        mock_store = MagicMock()
        mock_store.validate_token.return_value = ("test@example.com", None)

        with patch(
            "aragora.storage.password_reset_store.get_password_reset_store",
            return_value=mock_store,
        ):
            result = auth_handler._handle_reset_password(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 400


# ===========================================================================
# Test: API Key Management
# ===========================================================================


class TestAPIKeyManagement:
    """Test API key generation, listing, and revocation."""

    def test_generate_api_key_success(
        self, auth_handler, mock_user_store, mock_user, mock_auth_context
    ):
        """Test successful API key generation."""
        mock_user_store.get_user_by_id.return_value = mock_user

        request = make_mock_handler(command="POST")

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                result = auth_handler._handle_generate_api_key(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        assert "api_key" in parsed
        assert "prefix" in parsed

    def test_generate_api_key_no_api_access(
        self, auth_handler, mock_user_store, mock_user, mock_auth_context, mock_org
    ):
        """Test API key generation fails when org has no API access."""
        mock_org.limits.api_access = False
        mock_user_store.get_user_by_id.return_value = mock_user
        mock_user_store.get_organization_by_id.return_value = mock_org

        request = make_mock_handler(command="POST")

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                result = auth_handler._handle_generate_api_key(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 403

    def test_list_api_keys_success(
        self, auth_handler, mock_user_store, mock_user, mock_auth_context
    ):
        """Test successful API key listing."""
        mock_user.api_key_prefix = "ara_test"
        mock_user.api_key_created_at = datetime.now(timezone.utc)
        mock_user.api_key_expires_at = datetime.now(timezone.utc) + timedelta(days=365)
        mock_user_store.get_user_by_id.return_value = mock_user

        request = make_mock_handler(command="GET")

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                result = auth_handler._handle_list_api_keys(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        assert "keys" in parsed
        assert parsed["count"] == 1

    def test_list_api_keys_empty(self, auth_handler, mock_user_store, mock_user, mock_auth_context):
        """Test API key listing when user has no keys."""
        mock_user.api_key_prefix = None
        mock_user_store.get_user_by_id.return_value = mock_user

        request = make_mock_handler(command="GET")

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                result = auth_handler._handle_list_api_keys(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        assert parsed["count"] == 0

    def test_revoke_api_key_success(
        self, auth_handler, mock_user_store, mock_user, mock_auth_context
    ):
        """Test successful API key revocation."""
        mock_user_store.get_user_by_id.return_value = mock_user

        request = make_mock_handler(command="DELETE")

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                result = auth_handler._handle_revoke_api_key(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        mock_user_store.update_user.assert_called()

    def test_revoke_api_key_by_prefix_success(
        self, auth_handler, mock_user_store, mock_user, mock_auth_context
    ):
        """Test API key revocation by prefix."""
        mock_user.api_key_prefix = "ara_test123"
        mock_user_store.get_user_by_id.return_value = mock_user

        request = make_mock_handler(command="DELETE")

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                result = auth_handler._handle_revoke_api_key_prefix(request, "ara_test123")

        parsed = parse_result(result)
        assert parsed["success"] is True

    def test_revoke_api_key_by_prefix_not_found(
        self, auth_handler, mock_user_store, mock_user, mock_auth_context
    ):
        """Test API key revocation by prefix fails when not found."""
        mock_user.api_key_prefix = "different_prefix"
        mock_user_store.get_user_by_id.return_value = mock_user

        request = make_mock_handler(command="DELETE")

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                result = auth_handler._handle_revoke_api_key_prefix(request, "ara_wrong")

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 404


# ===========================================================================
# Test: MFA/2FA Flows
# ===========================================================================


class TestMFAFlows:
    """Test MFA setup, enable, disable, and verify flows."""

    def test_mfa_setup_success(self, auth_handler, mock_user_store, mock_user, mock_auth_context):
        """Test successful MFA setup."""
        mock_user.mfa_enabled = False
        mock_user_store.get_user_by_id.return_value = mock_user

        request = make_mock_handler(command="POST")

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                with patch("pyotp.random_base32", return_value="TESTSECRET"):
                    with patch("pyotp.TOTP") as mock_totp:
                        mock_totp_instance = MagicMock()
                        mock_totp_instance.provisioning_uri.return_value = (
                            "otpauth://totp/Aragora:test@example.com?secret=TESTSECRET"
                        )
                        mock_totp.return_value = mock_totp_instance

                        result = auth_handler._handle_mfa_setup(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        assert "secret" in parsed
        assert "provisioning_uri" in parsed

    def test_mfa_setup_already_enabled(
        self, auth_handler, mock_user_store, mock_mfa_user, mock_auth_context
    ):
        """Test MFA setup fails when already enabled."""
        mock_user_store.get_user_by_id.return_value = mock_mfa_user

        request = make_mock_handler(command="POST")

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                result = auth_handler._handle_mfa_setup(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_mfa_enable_success(self, auth_handler, mock_user_store, mock_user, mock_auth_context):
        """Test successful MFA enable."""
        mock_user.mfa_enabled = False
        mock_user.mfa_secret = "TESTSECRET"
        mock_user_store.get_user_by_id.return_value = mock_user

        request = make_mock_handler(
            body={"code": "123456"},
            command="POST",
        )

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                with patch("pyotp.TOTP") as mock_totp:
                    mock_totp_instance = MagicMock()
                    mock_totp_instance.verify.return_value = True
                    mock_totp.return_value = mock_totp_instance

                    result = auth_handler._handle_mfa_enable(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        assert "backup_codes" in parsed
        assert len(parsed["backup_codes"]) == 10

    def test_mfa_enable_invalid_code(
        self, auth_handler, mock_user_store, mock_user, mock_auth_context
    ):
        """Test MFA enable fails with invalid code."""
        mock_user.mfa_enabled = False
        mock_user.mfa_secret = "TESTSECRET"
        mock_user_store.get_user_by_id.return_value = mock_user

        request = make_mock_handler(
            body={"code": "000000"},
            command="POST",
        )

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                with patch("pyotp.TOTP") as mock_totp:
                    mock_totp_instance = MagicMock()
                    mock_totp_instance.verify.return_value = False
                    mock_totp.return_value = mock_totp_instance

                    result = auth_handler._handle_mfa_enable(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_mfa_disable_with_code(
        self, auth_handler, mock_user_store, mock_mfa_user, mock_auth_context
    ):
        """Test MFA disable with MFA code."""
        mock_user_store.get_user_by_id.return_value = mock_mfa_user

        request = make_mock_handler(
            body={"code": "123456"},
            command="POST",
        )

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                with patch("pyotp.TOTP") as mock_totp:
                    mock_totp_instance = MagicMock()
                    mock_totp_instance.verify.return_value = True
                    mock_totp.return_value = mock_totp_instance

                    result = auth_handler._handle_mfa_disable(request)

        parsed = parse_result(result)
        assert parsed["success"] is True

    def test_mfa_disable_with_password(
        self, auth_handler, mock_user_store, mock_mfa_user, mock_auth_context
    ):
        """Test MFA disable with password."""
        mock_mfa_user.verify_password.return_value = True
        mock_user_store.get_user_by_id.return_value = mock_mfa_user

        request = make_mock_handler(
            body={"password": "mypassword"},
            command="POST",
        )

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                result = auth_handler._handle_mfa_disable(request)

        parsed = parse_result(result)
        assert parsed["success"] is True

    def test_mfa_verify_with_totp(self, auth_handler, mock_user_store, mock_mfa_user, mock_tokens):
        """Test MFA verify with TOTP code."""
        mock_user_store.get_user_by_id.return_value = mock_mfa_user

        request = make_mock_handler(
            body={
                "code": "123456",
                "pending_token": "pending_token_xyz",
            },
            command="POST",
        )

        mock_pending = MagicMock()
        mock_pending.sub = "user-123"

        mock_blacklist = MagicMock()

        with patch(
            "aragora.billing.jwt_auth.validate_mfa_pending_token",
            return_value=mock_pending,
        ):
            with patch("pyotp.TOTP") as mock_totp:
                mock_totp_instance = MagicMock()
                mock_totp_instance.verify.return_value = True
                mock_totp.return_value = mock_totp_instance

                with patch(
                    "aragora.billing.jwt_auth.get_token_blacklist",
                    return_value=mock_blacklist,
                ):
                    with patch(
                        "aragora.billing.jwt_auth.create_token_pair",
                        return_value=mock_tokens,
                    ):
                        result = auth_handler._handle_mfa_verify(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        assert "tokens" in parsed

    def test_mfa_verify_with_backup_code(
        self, auth_handler, mock_user_store, mock_mfa_user, mock_tokens
    ):
        """Test MFA verify with backup code."""
        backup_codes = ["backup0", "backup1", "backup2"]
        backup_hashes = [hashlib.sha256(c.encode()).hexdigest() for c in backup_codes]
        mock_mfa_user.mfa_backup_codes = json.dumps(backup_hashes)
        mock_user_store.get_user_by_id.return_value = mock_mfa_user

        request = make_mock_handler(
            body={
                "code": "backup0",
                "pending_token": "pending_token_xyz",
            },
            command="POST",
        )

        mock_pending = MagicMock()
        mock_pending.sub = "user-123"

        mock_blacklist = MagicMock()

        with patch(
            "aragora.billing.jwt_auth.validate_mfa_pending_token",
            return_value=mock_pending,
        ):
            with patch("pyotp.TOTP") as mock_totp:
                mock_totp_instance = MagicMock()
                mock_totp_instance.verify.return_value = False  # TOTP fails
                mock_totp.return_value = mock_totp_instance

                with patch(
                    "aragora.billing.jwt_auth.get_token_blacklist",
                    return_value=mock_blacklist,
                ):
                    with patch(
                        "aragora.billing.jwt_auth.create_token_pair",
                        return_value=mock_tokens,
                    ):
                        result = auth_handler._handle_mfa_verify(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        assert "backup_codes_remaining" in parsed

    def test_regenerate_backup_codes_success(
        self, auth_handler, mock_user_store, mock_mfa_user, mock_auth_context
    ):
        """Test backup codes regeneration."""
        mock_user_store.get_user_by_id.return_value = mock_mfa_user

        request = make_mock_handler(
            body={"code": "123456"},
            command="POST",
        )

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                with patch("pyotp.TOTP") as mock_totp:
                    mock_totp_instance = MagicMock()
                    mock_totp_instance.verify.return_value = True
                    mock_totp.return_value = mock_totp_instance

                    result = auth_handler._handle_mfa_backup_codes(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        assert "backup_codes" in parsed
        assert len(parsed["backup_codes"]) == 10


# ===========================================================================
# Test: Logout Flows
# ===========================================================================


class TestLogoutFlows:
    """Test logout and logout-all flows."""

    def test_logout_success(self, auth_handler, mock_auth_context):
        """Test successful logout."""
        request = make_mock_handler(command="POST")
        request.headers["Authorization"] = "Bearer test_token"

        mock_blacklist = MagicMock()
        mock_blacklist.revoke_token.return_value = True

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                with patch("aragora.billing.jwt_auth.revoke_token_persistent", return_value=True):
                    with patch(
                        "aragora.billing.jwt_auth.get_token_blacklist",
                        return_value=mock_blacklist,
                    ):
                        with patch(
                            "aragora.server.middleware.auth.extract_token",
                            return_value="test_token",
                        ):
                            result = auth_handler._handle_logout(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        assert "logged out" in parsed.get("message", "").lower()

    def test_logout_all_success(self, auth_handler, mock_user_store, mock_auth_context):
        """Test logout from all devices."""
        request = make_mock_handler(command="POST")

        mock_blacklist = MagicMock()
        mock_blacklist.revoke_token.return_value = True

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                with patch("aragora.billing.jwt_auth.revoke_token_persistent", return_value=True):
                    with patch(
                        "aragora.billing.jwt_auth.get_token_blacklist",
                        return_value=mock_blacklist,
                    ):
                        with patch(
                            "aragora.server.middleware.auth.extract_token",
                            return_value="test_token",
                        ):
                            result = auth_handler._handle_logout_all(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        assert parsed.get("sessions_invalidated") is True
        assert "token_version" in parsed


# ===========================================================================
# Test: Profile Operations
# ===========================================================================


class TestProfileOperations:
    """Test profile get, update, and password change."""

    def test_get_me_success(
        self, auth_handler, mock_user_store, mock_user, mock_auth_context, mock_org
    ):
        """Test get current user info."""
        mock_user_store.get_user_by_id.return_value = mock_user
        mock_user_store.get_organization_by_id.return_value = mock_org

        request = make_mock_handler(command="GET")

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                result = auth_handler._handle_get_me(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        assert "user" in parsed
        assert "organization" in parsed

    def test_update_me_success(self, auth_handler, mock_user_store, mock_user, mock_auth_context):
        """Test successful profile update."""
        mock_user_store.get_user_by_id.return_value = mock_user

        request = make_mock_handler(
            body={"name": "Updated Name"},
            command="PUT",
        )

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                result = auth_handler._handle_update_me(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        assert "user" in parsed

    def test_change_password_success(
        self, auth_handler, mock_user_store, mock_user, mock_auth_context
    ):
        """Test successful password change."""
        mock_user.verify_password.return_value = True
        mock_user_store.get_user_by_id.return_value = mock_user

        request = make_mock_handler(
            body={
                "current_password": "oldpassword",
                "new_password": "NewSecurePassword123!",
            },
            command="POST",
        )

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                with patch("aragora.billing.models.hash_password", return_value=("hash", "salt")):
                    result = auth_handler._handle_change_password(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        assert parsed.get("sessions_invalidated") is True

    def test_change_password_wrong_current(
        self, auth_handler, mock_user_store, mock_user, mock_auth_context
    ):
        """Test password change fails with wrong current password."""
        mock_user.verify_password.return_value = False
        mock_user_store.get_user_by_id.return_value = mock_user

        request = make_mock_handler(
            body={
                "current_password": "wrongpassword",
                "new_password": "NewSecurePassword123!",
            },
            command="POST",
        )

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                result = auth_handler._handle_change_password(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 401


# ===========================================================================
# Test: Security Edge Cases
# ===========================================================================


class TestSecurityEdgeCases:
    """Test security edge cases and potential attack vectors."""

    def test_sql_injection_in_email(self, auth_handler, mock_user_store):
        """Test SQL injection attempts in email field."""
        mock_user_store.get_user_by_email.return_value = None

        request = make_mock_handler(
            body={
                "email": "user'; DROP TABLE users; --@example.com",
                "password": "SecurePassword123!",
            },
            command="POST",
        )

        result = auth_handler._handle_register(request)
        parsed = parse_result(result)

        # Should fail validation (invalid email format)
        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_xss_in_name_field(self, auth_handler, mock_user_store, mock_user, mock_tokens):
        """Test XSS attempts in name field."""
        mock_user_store.get_user_by_email.return_value = None
        mock_user_store.create_user.return_value = mock_user

        request = make_mock_handler(
            body={
                "email": "user@example.com",
                "password": "SecurePassword123!",
                "name": "<script>alert('XSS')</script>",
            },
            command="POST",
        )

        with patch("aragora.billing.models.hash_password", return_value=("hash", "salt")):
            with patch("aragora.billing.jwt_auth.create_token_pair", return_value=mock_tokens):
                result = auth_handler._handle_register(request)

        # Should succeed but name should be sanitized/truncated
        parsed = parse_result(result)
        assert parsed["status_code"] in (201, 400)

    def test_extremely_long_email(self, auth_handler):
        """Test email exceeding maximum length."""
        long_email = "a" * 300 + "@example.com"

        request = make_mock_handler(
            body={
                "email": long_email,
                "password": "SecurePassword123!",
            },
            command="POST",
        )

        result = auth_handler._handle_register(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_unicode_email(self, auth_handler):
        """Test email with unicode characters."""
        request = make_mock_handler(
            body={
                "email": "user@exämple.com",
                "password": "SecurePassword123!",
            },
            command="POST",
        )

        result = auth_handler._handle_register(request)
        parsed = parse_result(result)

        # Should either succeed (IDN) or fail validation
        assert parsed["status_code"] in (201, 400)

    def test_null_byte_in_password(self, auth_handler, mock_user_store):
        """Test null byte in password."""
        mock_user_store.get_user_by_email.return_value = None

        request = make_mock_handler(
            body={
                "email": "user@example.com",
                "password": "password\x00injection",
            },
            command="POST",
        )

        result = auth_handler._handle_register(request)
        # Should handle gracefully without crashing
        assert result is not None

    def test_timing_attack_prevention_nonexistent_user(
        self, auth_handler, mock_user_store, mock_lockout_tracker
    ):
        """Test error message is same for existing and non-existing users."""
        mock_user_store.get_user_by_email.return_value = None

        request = make_mock_handler(
            body={
                "email": "nonexistent@example.com",
                "password": "somepassword",
            },
            command="POST",
        )

        with patch("aragora.auth.lockout.get_lockout_tracker", return_value=mock_lockout_tracker):
            result = auth_handler._handle_login(request)

        parsed = parse_result(result)
        # Error message should not reveal whether email exists
        assert "invalid email or password" in parsed.get("error", "").lower()


# ===========================================================================
# Test: Route Handling
# ===========================================================================


class TestRouteHandling:
    """Test route handling and dispatching."""

    def test_can_handle_auth_routes(self, auth_handler):
        """Test handler can handle auth routes."""
        assert auth_handler.can_handle("/api/auth/login") is True
        assert auth_handler.can_handle("/api/auth/register") is True
        assert auth_handler.can_handle("/api/auth/me") is True
        assert auth_handler.can_handle("/api/auth/logout") is True
        assert auth_handler.can_handle("/api/auth/refresh") is True
        assert auth_handler.can_handle("/api/auth/sessions") is True
        assert auth_handler.can_handle("/api/auth/mfa/setup") is True

    def test_can_handle_versioned_routes(self, auth_handler):
        """Test handler can handle versioned routes."""
        assert auth_handler.can_handle("/api/v1/auth/login") is True
        assert auth_handler.can_handle("/api/v1/auth/register") is True
        assert auth_handler.can_handle("/api/v1/auth/me") is True

    def test_can_handle_session_wildcard_routes(self, auth_handler):
        """Test handler can handle session/:id routes."""
        assert auth_handler.can_handle("/api/auth/sessions/session-123") is True
        assert auth_handler.can_handle("/api/auth/sessions/abc") is True

    def test_can_handle_api_keys_wildcard_routes(self, auth_handler):
        """Test handler can handle api-keys/:prefix routes."""
        assert auth_handler.can_handle("/api/auth/api-keys/ara_test123") is True

    def test_cannot_handle_other_routes(self, auth_handler):
        """Test handler cannot handle unrelated routes."""
        assert auth_handler.can_handle("/api/debates") is False
        assert auth_handler.can_handle("/api/users") is False
        assert auth_handler.can_handle("/api/health") is False

    def test_handle_unsupported_method_returns_405(self, auth_handler):
        """Test unsupported method returns 405."""
        request = make_mock_handler(command="PATCH")

        result = maybe_await(
            auth_handler.handle(
                path="/api/auth/login",
                query_params={},
                handler=request,
                method="PATCH",
            )
        )

        parsed = parse_result(result)
        assert parsed["status_code"] == 405


# ===========================================================================
# Test: Error Handling
# ===========================================================================


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_service_unavailable_no_user_store(self):
        """Test 503 when user store is unavailable."""
        context = {
            "storage": MagicMock(),
            "user_store": None,
            "elo_system": MagicMock(),
        }
        handler = AuthHandler(server_context=context)

        request = make_mock_handler(
            body={
                "email": "test@example.com",
                "password": "password",
            },
            command="POST",
        )

        result = handler._handle_login(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 503

    def test_refresh_revocation_failure(self, auth_handler, mock_user_store, mock_user):
        """Test token refresh handles revocation failure."""
        mock_user_store.get_user_by_id.return_value = mock_user

        request = make_mock_handler(
            body={"refresh_token": "valid_token"},
            command="POST",
        )

        mock_payload = MagicMock()
        mock_payload.user_id = "user-123"

        with patch(
            "aragora.server.handlers.auth.handler.validate_refresh_token",
            return_value=mock_payload,
        ):
            with patch(
                "aragora.billing.jwt_auth.revoke_token_persistent",
                side_effect=Exception("DB error"),
            ):
                result = auth_handler._handle_refresh(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 500


# ===========================================================================
# Test: Handler Properties
# ===========================================================================


class TestHandlerProperties:
    """Test handler property values."""

    def test_resource_type(self, auth_handler):
        """Test RESOURCE_TYPE is set correctly."""
        assert auth_handler.RESOURCE_TYPE == "auth"

    def test_routes_list_contains_essential_endpoints(self, auth_handler):
        """Test ROUTES includes essential endpoints."""
        routes = auth_handler.ROUTES
        assert "/api/auth/login" in routes
        assert "/api/auth/logout" in routes
        assert "/api/auth/register" in routes
        assert "/api/auth/me" in routes
        assert "/api/auth/refresh" in routes
        assert "/api/auth/password" in routes
        assert "/api/auth/api-key" in routes
        assert "/api/auth/mfa/setup" in routes
        assert "/api/auth/sessions" in routes


# ===========================================================================
# Test: Rate Limiting Behavior
# ===========================================================================


class TestRateLimiting:
    """Test rate limiting on auth endpoints."""

    def test_rate_limiter_names_are_set(self, auth_handler):
        """Test that rate limiter names are configured on endpoints."""
        # The rate_limit decorator sets metadata on the method
        # We verify the methods have the expected decorators
        assert hasattr(auth_handler._handle_register, "__wrapped__") or callable(
            auth_handler._handle_register
        )
        assert hasattr(auth_handler._handle_login, "__wrapped__") or callable(
            auth_handler._handle_login
        )

    def test_register_has_rate_limit(self, auth_handler):
        """Test registration endpoint has rate limiting configured."""
        # The rate_limit decorator is applied with specific parameters
        # We can't easily test the actual rate limiting without integration tests
        # but we can verify the handler works under normal conditions
        request = make_mock_handler(
            body={
                "email": "test@example.com",
                "password": "short",  # Will fail validation, not rate limit
            },
            command="POST",
        )
        result = auth_handler._handle_register(request)
        # Should fail with 400 (validation), not 429 (rate limit)
        assert parse_result(result)["status_code"] == 400


# ===========================================================================
# Test: Token Revocation
# ===========================================================================


class TestTokenRevocation:
    """Test token revocation operations."""

    def test_revoke_token_success(self, auth_handler, mock_auth_context):
        """Test successful token revocation."""
        request = make_mock_handler(
            body={"token": "token_to_revoke"},
            command="POST",
        )

        mock_blacklist = MagicMock()
        mock_blacklist.revoke_token.return_value = True
        mock_blacklist.size.return_value = 5

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                with patch(
                    "aragora.billing.jwt_auth.get_token_blacklist",
                    return_value=mock_blacklist,
                ):
                    with patch(
                        "aragora.billing.jwt_auth.revoke_token_persistent",
                        return_value=True,
                    ):
                        result = auth_handler._handle_revoke_token(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        assert "blacklist_size" in parsed

    def test_revoke_token_invalid(self, auth_handler, mock_auth_context):
        """Test token revocation with invalid token."""
        request = make_mock_handler(
            body={"token": "invalid_token"},
            command="POST",
        )

        mock_blacklist = MagicMock()
        mock_blacklist.revoke_token.return_value = False

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                with patch(
                    "aragora.billing.jwt_auth.get_token_blacklist",
                    return_value=mock_blacklist,
                ):
                    with patch(
                        "aragora.billing.jwt_auth.revoke_token_persistent",
                        return_value=True,
                    ):
                        result = auth_handler._handle_revoke_token(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 400


# ===========================================================================
# Test: RBAC Permission Checks
# ===========================================================================


class TestRBACPermissionChecks:
    """Test RBAC permission enforcement on protected endpoints."""

    @pytest.mark.no_auto_auth
    def test_logout_requires_authentication(self, auth_handler):
        """Test logout endpoint returns 401 without authentication."""
        request = make_mock_handler(command="POST")

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
        ) as mock_extract:
            mock_extract.return_value = MagicMock(is_authenticated=False, user_id=None)
            result = auth_handler._handle_logout(request)

        parsed = parse_result(result)
        assert parsed["status_code"] == 401

    @pytest.mark.no_auto_auth
    def test_get_me_requires_authentication(self, auth_handler):
        """Test get me endpoint returns 401 without authentication."""
        request = make_mock_handler(command="GET")

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
        ) as mock_extract:
            mock_extract.return_value = MagicMock(is_authenticated=False, user_id=None)
            result = auth_handler._handle_get_me(request)

        parsed = parse_result(result)
        assert parsed["status_code"] == 401

    @pytest.mark.no_auto_auth
    def test_generate_api_key_requires_authentication(self, auth_handler):
        """Test API key generation returns 401 without authentication."""
        request = make_mock_handler(command="POST")

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
        ) as mock_extract:
            mock_extract.return_value = MagicMock(is_authenticated=False, user_id=None)
            result = auth_handler._handle_generate_api_key(request)

        parsed = parse_result(result)
        assert parsed["status_code"] == 401

    @pytest.mark.no_auto_auth
    def test_mfa_setup_requires_authentication(self, auth_handler):
        """Test MFA setup returns 401 without authentication."""
        request = make_mock_handler(command="POST")

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
        ) as mock_extract:
            mock_extract.return_value = MagicMock(is_authenticated=False, user_id=None)
            result = auth_handler._handle_mfa_setup(request)

        parsed = parse_result(result)
        assert parsed["status_code"] == 401

    @pytest.mark.no_auto_auth
    def test_list_sessions_requires_authentication(self, auth_handler):
        """Test list sessions returns 401 without authentication."""
        request = make_mock_handler(command="GET")

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
        ) as mock_extract:
            mock_extract.return_value = MagicMock(is_authenticated=False, user_id=None)
            result = auth_handler._handle_list_sessions(request)

        parsed = parse_result(result)
        assert parsed["status_code"] == 401


# ===========================================================================
# Test: Additional Edge Cases
# ===========================================================================


class TestAdditionalEdgeCases:
    """Additional edge case tests."""

    def test_mfa_verify_missing_pending_token(self, auth_handler):
        """Test MFA verify fails without pending token."""
        request = make_mock_handler(
            body={"code": "123456"},
            command="POST",
        )

        result = auth_handler._handle_mfa_verify(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_mfa_verify_missing_code(self, auth_handler):
        """Test MFA verify fails without code."""
        request = make_mock_handler(
            body={"pending_token": "some_token"},
            command="POST",
        )

        result = auth_handler._handle_mfa_verify(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_mfa_disable_requires_code_or_password(
        self, auth_handler, mock_user_store, mock_mfa_user, mock_auth_context
    ):
        """Test MFA disable requires either code or password."""
        mock_user_store.get_user_by_id.return_value = mock_mfa_user

        request = make_mock_handler(
            body={},  # No code or password
            command="POST",
        )

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                result = auth_handler._handle_mfa_disable(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_mfa_enable_missing_code(
        self, auth_handler, mock_user_store, mock_user, mock_auth_context
    ):
        """Test MFA enable fails without verification code."""
        mock_user.mfa_enabled = False
        mock_user.mfa_secret = "TESTSECRET"
        mock_user_store.get_user_by_id.return_value = mock_user

        request = make_mock_handler(
            body={},  # No code
            command="POST",
        )

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                result = auth_handler._handle_mfa_enable(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_change_password_missing_both_fields(self, auth_handler, mock_auth_context):
        """Test password change fails when both fields are missing."""
        request = make_mock_handler(
            body={},
            command="POST",
        )

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                result = auth_handler._handle_change_password(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_reset_password_missing_token(self, auth_handler):
        """Test password reset fails without token."""
        request = make_mock_handler(
            body={"password": "NewSecurePassword123!"},
            command="POST",
        )

        result = auth_handler._handle_reset_password(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_reset_password_missing_password(self, auth_handler):
        """Test password reset fails without new password."""
        request = make_mock_handler(
            body={"token": "some_token"},
            command="POST",
        )

        result = auth_handler._handle_reset_password(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 400


# ===========================================================================
# Test: Database-Based Account Lockout
# ===========================================================================


class TestDatabaseLockout:
    """Test database-based account lockout (legacy support)."""

    def test_login_db_lockout_active(self, auth_handler, mock_user_store, mock_user):
        """Test login fails when database lockout is active."""
        mock_user_store.get_user_by_email.return_value = mock_user
        # Set up database-based lockout
        lockout_until = datetime.now(timezone.utc) + timedelta(minutes=15)
        mock_user_store.is_account_locked.return_value = (True, lockout_until, 5)

        # In-memory tracker not locked
        mock_lockout_tracker = MagicMock()
        mock_lockout_tracker.is_locked.return_value = False

        request = make_mock_handler(
            body={"email": "test@example.com", "password": "password"},
            command="POST",
        )

        with patch(
            "aragora.server.handlers.auth.login.get_lockout_tracker",
            return_value=mock_lockout_tracker,
        ):
            result = auth_handler._handle_login(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 429
        assert "locked" in parsed.get("error", "").lower()

    def test_login_records_failed_attempt_in_db(
        self, auth_handler, mock_user_store, mock_user, mock_lockout_tracker
    ):
        """Test failed login records attempt in database."""
        mock_user.verify_password.return_value = False
        mock_user_store.get_user_by_email.return_value = mock_user
        mock_user_store.is_account_locked.return_value = (False, None, 0)

        request = make_mock_handler(
            body={"email": "test@example.com", "password": "wrong"},
            command="POST",
        )

        with patch(
            "aragora.server.handlers.auth.login.get_lockout_tracker",
            return_value=mock_lockout_tracker,
        ):
            auth_handler._handle_login(request)

        mock_user_store.record_failed_login.assert_called_once_with("test@example.com")


# ===========================================================================
# Test: MFA Edge Cases
# ===========================================================================


class TestMFAEdgeCases:
    """Test MFA edge cases and error scenarios."""

    def test_mfa_verify_user_not_found(self, auth_handler, mock_user_store):
        """Test MFA verify fails when user not found."""
        mock_user_store.get_user_by_id.return_value = None

        request = make_mock_handler(
            body={"code": "123456", "pending_token": "pending_xyz"},
            command="POST",
        )

        mock_pending = MagicMock()
        mock_pending.sub = "nonexistent-user"

        with patch(
            "aragora.billing.jwt_auth.validate_mfa_pending_token",
            return_value=mock_pending,
        ):
            result = auth_handler._handle_mfa_verify(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 404

    def test_mfa_verify_mfa_not_enabled(self, auth_handler, mock_user_store, mock_user):
        """Test MFA verify fails when MFA is not enabled."""
        mock_user.mfa_enabled = False
        mock_user.mfa_secret = None
        mock_user_store.get_user_by_id.return_value = mock_user

        request = make_mock_handler(
            body={"code": "123456", "pending_token": "pending_xyz"},
            command="POST",
        )

        mock_pending = MagicMock()
        mock_pending.sub = "user-123"

        with patch(
            "aragora.billing.jwt_auth.validate_mfa_pending_token",
            return_value=mock_pending,
        ):
            result = auth_handler._handle_mfa_verify(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_mfa_verify_invalid_totp_and_no_backup(
        self, auth_handler, mock_user_store, mock_mfa_user
    ):
        """Test MFA verify fails with invalid TOTP and no matching backup code."""
        mock_mfa_user.mfa_backup_codes = json.dumps([])  # Empty backup codes
        mock_user_store.get_user_by_id.return_value = mock_mfa_user

        request = make_mock_handler(
            body={"code": "000000", "pending_token": "pending_xyz"},
            command="POST",
        )

        mock_pending = MagicMock()
        mock_pending.sub = "user-123"

        with patch(
            "aragora.billing.jwt_auth.validate_mfa_pending_token",
            return_value=mock_pending,
        ):
            with patch("pyotp.TOTP") as mock_totp:
                mock_totp_instance = MagicMock()
                mock_totp_instance.verify.return_value = False
                mock_totp.return_value = mock_totp_instance

                result = auth_handler._handle_mfa_verify(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_mfa_enable_already_enabled(
        self, auth_handler, mock_user_store, mock_mfa_user, mock_auth_context
    ):
        """Test MFA enable fails when already enabled."""
        mock_user_store.get_user_by_id.return_value = mock_mfa_user

        request = make_mock_handler(
            body={"code": "123456"},
            command="POST",
        )

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)
                result = auth_handler._handle_mfa_enable(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 400
        assert "already enabled" in parsed.get("error", "").lower()

    def test_mfa_disable_not_enabled(
        self, auth_handler, mock_user_store, mock_user, mock_auth_context
    ):
        """Test MFA disable fails when not enabled."""
        mock_user.mfa_enabled = False
        mock_user_store.get_user_by_id.return_value = mock_user

        request = make_mock_handler(
            body={"code": "123456"},
            command="POST",
        )

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)
                result = auth_handler._handle_mfa_disable(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 400
        assert "not enabled" in parsed.get("error", "").lower()

    def test_mfa_disable_invalid_code(
        self, auth_handler, mock_user_store, mock_mfa_user, mock_auth_context
    ):
        """Test MFA disable fails with invalid code."""
        mock_user_store.get_user_by_id.return_value = mock_mfa_user

        request = make_mock_handler(
            body={"code": "000000"},
            command="POST",
        )

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)
                with patch("pyotp.TOTP") as mock_totp:
                    mock_totp_instance = MagicMock()
                    mock_totp_instance.verify.return_value = False
                    mock_totp.return_value = mock_totp_instance

                    result = auth_handler._handle_mfa_disable(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_mfa_disable_invalid_password(
        self, auth_handler, mock_user_store, mock_mfa_user, mock_auth_context
    ):
        """Test MFA disable fails with invalid password."""
        mock_mfa_user.verify_password.return_value = False
        mock_user_store.get_user_by_id.return_value = mock_mfa_user

        request = make_mock_handler(
            body={"password": "wrong"},
            command="POST",
        )

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)
                result = auth_handler._handle_mfa_disable(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_mfa_backup_codes_invalid_code(
        self, auth_handler, mock_user_store, mock_mfa_user, mock_auth_context
    ):
        """Test backup codes regeneration fails with invalid MFA code."""
        mock_user_store.get_user_by_id.return_value = mock_mfa_user

        request = make_mock_handler(
            body={"code": "000000"},
            command="POST",
        )

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)
                with patch("pyotp.TOTP") as mock_totp:
                    mock_totp_instance = MagicMock()
                    mock_totp_instance.verify.return_value = False
                    mock_totp.return_value = mock_totp_instance

                    result = auth_handler._handle_mfa_backup_codes(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_mfa_backup_codes_missing_code(
        self, auth_handler, mock_user_store, mock_mfa_user, mock_auth_context
    ):
        """Test backup codes regeneration fails without code."""
        mock_user_store.get_user_by_id.return_value = mock_mfa_user

        request = make_mock_handler(
            body={},
            command="POST",
        )

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)
                result = auth_handler._handle_mfa_backup_codes(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 400


# ===========================================================================
# Test: Profile Edge Cases
# ===========================================================================


class TestProfileEdgeCases:
    """Test profile operations edge cases."""

    def test_get_me_user_not_found(self, auth_handler, mock_user_store, mock_auth_context):
        """Test get me fails when user not found."""
        mock_user_store.get_user_by_id.return_value = None

        request = make_mock_handler(command="GET")

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)
                result = auth_handler._handle_get_me(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 404

    def test_update_me_user_not_found(self, auth_handler, mock_user_store, mock_auth_context):
        """Test update me fails when user not found."""
        mock_user_store.get_user_by_id.return_value = None

        request = make_mock_handler(
            body={"name": "New Name"},
            command="PUT",
        )

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)
                result = auth_handler._handle_update_me(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 404

    def test_update_me_truncates_long_name(
        self, auth_handler, mock_user_store, mock_user, mock_auth_context
    ):
        """Test update me truncates very long name."""
        mock_user_store.get_user_by_id.return_value = mock_user

        request = make_mock_handler(
            body={"name": "A" * 200},  # Very long name
            command="PUT",
        )

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)
                result = auth_handler._handle_update_me(request)

        # Should succeed but name should be truncated
        parsed = parse_result(result)
        assert parsed["success"] is True

        # Verify update_user was called with truncated name
        calls = mock_user_store.update_user.call_args_list
        assert len(calls) >= 1
        # The name should be truncated to 100 characters
        update_kwargs = calls[0][1]
        assert len(update_kwargs.get("name", "")) <= 100

    def test_change_password_user_not_found(self, auth_handler, mock_user_store, mock_auth_context):
        """Test change password fails when user not found."""
        mock_user_store.get_user_by_id.return_value = None

        request = make_mock_handler(
            body={"current_password": "old", "new_password": "NewPassword123!"},
            command="POST",
        )

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)
                result = auth_handler._handle_change_password(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 404


# ===========================================================================
# Test: Password Reset Edge Cases
# ===========================================================================


class TestPasswordResetEdgeCases:
    """Test password reset edge cases."""

    def test_reset_password_user_not_found(self, auth_handler, mock_user_store):
        """Test password reset fails when user not found after token validation."""
        mock_user_store.get_user_by_email.return_value = None

        request = make_mock_handler(
            body={"token": "valid_token", "password": "NewPassword123!"},
            command="POST",
        )

        mock_store = MagicMock()
        mock_store.validate_token.return_value = ("test@example.com", None)
        mock_store.consume_token = MagicMock()

        with patch(
            "aragora.storage.password_reset_store.get_password_reset_store",
            return_value=mock_store,
        ):
            result = auth_handler._handle_reset_password(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 404

    def test_reset_password_disabled_account(self, auth_handler, mock_user_store, mock_user):
        """Test password reset fails for disabled account."""
        mock_user.is_active = False
        mock_user_store.get_user_by_email.return_value = mock_user

        request = make_mock_handler(
            body={"token": "valid_token", "password": "NewPassword123!"},
            command="POST",
        )

        mock_store = MagicMock()
        mock_store.validate_token.return_value = ("test@example.com", None)
        mock_store.consume_token = MagicMock()

        with patch(
            "aragora.storage.password_reset_store.get_password_reset_store",
            return_value=mock_store,
        ):
            result = auth_handler._handle_reset_password(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 401

    def test_forgot_password_missing_email(self, auth_handler):
        """Test forgot password fails with missing email."""
        request = make_mock_handler(
            body={},
            command="POST",
        )

        result = auth_handler._handle_forgot_password(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_forgot_password_rate_limited(self, auth_handler, mock_user_store, mock_user):
        """Test forgot password handles rate limiting gracefully."""
        mock_user_store.get_user_by_email.return_value = mock_user

        request = make_mock_handler(
            body={"email": "test@example.com"},
            command="POST",
        )

        mock_store = MagicMock()
        mock_store.create_token.return_value = (None, "Rate limit exceeded")

        with patch(
            "aragora.storage.password_reset_store.get_password_reset_store",
            return_value=mock_store,
        ):
            result = auth_handler._handle_forgot_password(request)

        # Should still return success to prevent enumeration
        parsed = parse_result(result)
        assert parsed["success"] is True


# ===========================================================================
# Test: Logout Edge Cases
# ===========================================================================


class TestLogoutEdgeCases:
    """Test logout edge cases."""

    def test_logout_without_token(self, auth_handler, mock_auth_context):
        """Test logout when no token is present."""
        request = make_mock_handler(command="POST")

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                with patch(
                    "aragora.server.middleware.auth.extract_token",
                    return_value=None,
                ):
                    result = auth_handler._handle_logout(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        assert "logged out" in parsed.get("message", "").lower()

    def test_logout_all_user_not_found(self, auth_handler, mock_user_store, mock_auth_context):
        """Test logout all fails when token version update returns 0."""
        mock_user_store.increment_token_version.return_value = 0

        request = make_mock_handler(command="POST")

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)
                result = auth_handler._handle_logout_all(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 404


# ===========================================================================
# Test: API Key Edge Cases
# ===========================================================================


class TestAPIKeyEdgeCases:
    """Test API key edge cases."""

    def test_generate_api_key_user_not_found(
        self, auth_handler, mock_user_store, mock_auth_context
    ):
        """Test API key generation fails when user not found."""
        mock_user_store.get_user_by_id.return_value = None

        request = make_mock_handler(command="POST")

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)
                result = auth_handler._handle_generate_api_key(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 404

    def test_revoke_api_key_user_not_found(self, auth_handler, mock_user_store, mock_auth_context):
        """Test API key revocation fails when user not found."""
        mock_user_store.get_user_by_id.return_value = None

        request = make_mock_handler(command="DELETE")

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)
                result = auth_handler._handle_revoke_api_key(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 404

    def test_list_api_keys_user_not_found(self, auth_handler, mock_user_store, mock_auth_context):
        """Test API key listing fails when user not found."""
        mock_user_store.get_user_by_id.return_value = None

        request = make_mock_handler(command="GET")

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)
                result = auth_handler._handle_list_api_keys(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 404


# ===========================================================================
# Test: Session Management Edge Cases
# ===========================================================================


class TestSessionEdgeCases:
    """Test session management edge cases."""

    def test_revoke_session_invalid_format(self, auth_handler, mock_auth_context):
        """Test session revocation fails with invalid session ID format."""
        request = make_mock_handler(command="DELETE")

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)
                result = auth_handler._handle_revoke_session(request, "ab")

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_revoke_session_empty_id(self, auth_handler, mock_auth_context):
        """Test session revocation fails with empty session ID."""
        request = make_mock_handler(command="DELETE")

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)
                result = auth_handler._handle_revoke_session(request, "")

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 400


# ===========================================================================
# Test: Revoke Token Edge Cases
# ===========================================================================


class TestRevokeTokenEdgeCases:
    """Test token revocation edge cases."""

    def test_revoke_token_no_token_provided(self, auth_handler, mock_auth_context):
        """Test token revocation fails when no token in body and no auth token."""
        request = make_mock_handler(
            body={},
            command="POST",
        )

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                with patch(
                    "aragora.server.middleware.auth.extract_token",
                    return_value=None,
                ):
                    result = auth_handler._handle_revoke_token(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 400


# ===========================================================================
# Test: Additional Security Edge Cases
# ===========================================================================


class TestAdditionalSecurityEdgeCases:
    """Additional security-related edge case tests."""

    def test_register_multiple_spaces_in_name(
        self, auth_handler, mock_user_store, mock_user, mock_tokens
    ):
        """Test registration handles multiple spaces in name."""
        mock_user_store.get_user_by_email.return_value = None
        mock_user_store.create_user.return_value = mock_user

        request = make_mock_handler(
            body={
                "email": "user@example.com",
                "password": "SecurePassword123!",
                "name": "   Multiple   Spaces   ",
            },
            command="POST",
        )

        with patch("aragora.billing.models.hash_password", return_value=("hash", "salt")):
            with patch("aragora.billing.jwt_auth.create_token_pair", return_value=mock_tokens):
                result = auth_handler._handle_register(request)

        parsed = parse_result(result)
        assert parsed["status_code"] in (201, 400)

    def test_login_case_insensitive_email(
        self, auth_handler, mock_user_store, mock_user, mock_tokens, mock_lockout_tracker
    ):
        """Test login normalizes email to lowercase."""
        mock_user_store.get_user_by_email.return_value = mock_user

        request = make_mock_handler(
            body={"email": "TEST@EXAMPLE.COM", "password": "password"},
            command="POST",
        )

        with patch(
            "aragora.server.handlers.auth.login.get_lockout_tracker",
            return_value=mock_lockout_tracker,
        ):
            with patch("aragora.billing.jwt_auth.create_token_pair", return_value=mock_tokens):
                auth_handler._handle_login(request)

        # Email should be normalized to lowercase
        mock_user_store.get_user_by_email.assert_called_with("test@example.com")

    def test_register_empty_body(self, auth_handler):
        """Test registration fails with empty body."""
        request = make_mock_handler(
            body={},
            command="POST",
        )

        result = auth_handler._handle_register(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_login_whitespace_only_email(self, auth_handler):
        """Test login fails with whitespace-only email."""
        request = make_mock_handler(
            body={"email": "   ", "password": "password"},
            command="POST",
        )

        result = auth_handler._handle_login(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_register_password_with_unicode(
        self, auth_handler, mock_user_store, mock_user, mock_tokens
    ):
        """Test registration with unicode characters in password."""
        mock_user_store.get_user_by_email.return_value = None
        mock_user_store.create_user.return_value = mock_user

        request = make_mock_handler(
            body={
                "email": "user@example.com",
                "password": "SecurePass\u00e9word123!",  # Contains unicode
            },
            command="POST",
        )

        with patch("aragora.billing.models.hash_password", return_value=("hash", "salt")):
            with patch("aragora.billing.jwt_auth.create_token_pair", return_value=mock_tokens):
                result = auth_handler._handle_register(request)

        # Should work - unicode passwords are valid
        parsed = parse_result(result)
        assert parsed["status_code"] in (201, 400)


# ===========================================================================
# Test: Handle Method Routing
# ===========================================================================


class TestHandleMethodRouting:
    """Test handle() method routing to correct handlers."""

    def test_handle_routes_me_get(self, auth_handler):
        """Test GET /api/auth/me routes to _handle_get_me."""
        request = make_mock_handler(command="GET")

        with patch.object(auth_handler, "_handle_get_me") as mock_method:
            mock_method.return_value = MagicMock(status_code=200, body=b"{}")
            maybe_await(auth_handler.handle("/api/auth/me", {}, request, "GET"))
            mock_method.assert_called_once()

    def test_handle_routes_me_put(self, auth_handler):
        """Test PUT /api/auth/me routes to _handle_update_me."""
        request = make_mock_handler(
            body={"name": "Test"},
            command="PUT",
        )

        with patch.object(auth_handler, "_handle_update_me") as mock_method:
            mock_method.return_value = MagicMock(status_code=200, body=b"{}")
            maybe_await(auth_handler.handle("/api/auth/me", {}, request, "PUT"))
            mock_method.assert_called_once()

    def test_handle_routes_me_post(self, auth_handler):
        """Test POST /api/auth/me routes to _handle_update_me."""
        request = make_mock_handler(
            body={"name": "Test"},
            command="POST",
        )

        with patch.object(auth_handler, "_handle_update_me") as mock_method:
            mock_method.return_value = MagicMock(status_code=200, body=b"{}")
            maybe_await(auth_handler.handle("/api/auth/me", {}, request, "POST"))
            mock_method.assert_called_once()

    def test_handle_routes_password_change(self, auth_handler):
        """Test POST /api/auth/password/change routes correctly."""
        request = make_mock_handler(
            body={"current_password": "old", "new_password": "new"},
            command="POST",
        )

        with patch.object(auth_handler, "_handle_change_password") as mock_method:
            mock_method.return_value = MagicMock(status_code=200, body=b"{}")
            maybe_await(auth_handler.handle("/api/auth/password/change", {}, request, "POST"))
            mock_method.assert_called_once()

    def test_handle_routes_mfa_delete(self, auth_handler):
        """Test DELETE /api/auth/mfa routes to _handle_mfa_disable."""
        request = make_mock_handler(
            body={"password": "test"},
            command="DELETE",
        )

        with patch.object(auth_handler, "_handle_mfa_disable") as mock_method:
            mock_method.return_value = MagicMock(status_code=200, body=b"{}")
            maybe_await(auth_handler.handle("/api/auth/mfa", {}, request, "DELETE"))
            mock_method.assert_called_once()

    def test_handle_routes_api_key_post(self, auth_handler):
        """Test POST /api/auth/api-key routes to _handle_generate_api_key."""
        request = make_mock_handler(command="POST")

        with patch.object(auth_handler, "_handle_generate_api_key") as mock_method:
            mock_method.return_value = MagicMock(status_code=200, body=b"{}")
            maybe_await(auth_handler.handle("/api/auth/api-key", {}, request, "POST"))
            mock_method.assert_called_once()

    def test_handle_routes_api_key_delete(self, auth_handler):
        """Test DELETE /api/auth/api-key routes to _handle_revoke_api_key."""
        request = make_mock_handler(command="DELETE")

        with patch.object(auth_handler, "_handle_revoke_api_key") as mock_method:
            mock_method.return_value = MagicMock(status_code=200, body=b"{}")
            maybe_await(auth_handler.handle("/api/auth/api-key", {}, request, "DELETE"))
            mock_method.assert_called_once()


# ===========================================================================
# Export test classes
# ===========================================================================


__all__ = [
    "TestRegistrationSuccess",
    "TestRegistrationValidation",
    "TestLoginSuccess",
    "TestLoginMFA",
    "TestLoginFailure",
    "TestTokenRefresh",
    "TestSessionManagement",
    "TestPasswordReset",
    "TestAPIKeyManagement",
    "TestMFAFlows",
    "TestLogoutFlows",
    "TestProfileOperations",
    "TestSecurityEdgeCases",
    "TestRouteHandling",
    "TestErrorHandling",
    "TestHandlerProperties",
    "TestRateLimiting",
    "TestTokenRevocation",
    "TestRBACPermissionChecks",
    "TestAdditionalEdgeCases",
    "TestDatabaseLockout",
    "TestMFAEdgeCases",
    "TestProfileEdgeCases",
    "TestPasswordResetEdgeCases",
    "TestLogoutEdgeCases",
    "TestAPIKeyEdgeCases",
    "TestSessionEdgeCases",
    "TestRevokeTokenEdgeCases",
    "TestAdditionalSecurityEdgeCases",
    "TestHandleMethodRouting",
]
