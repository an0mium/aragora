"""
Tests for Login and Registration handlers.

Phase 5: Auth Handler Test Coverage - Login handler tests.

Tests for:
- handle_register - Create new user account
- handle_login - Authenticate user and get tokens
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from tests.server.handlers.conftest import (
    assert_error_response,
    assert_success_response,
    parse_handler_response,
)

# Patch paths for functions imported inside handlers
PATCH_CREATE_TOKEN_PAIR = "aragora.billing.jwt_auth.create_token_pair"
PATCH_HASH_PASSWORD = "aragora.billing.models.hash_password"
PATCH_CREATE_MFA_PENDING = "aragora.billing.jwt_auth.create_mfa_pending_token"
PATCH_GET_LOCKOUT_TRACKER = "aragora.server.handlers.auth.login.get_lockout_tracker"
PATCH_GET_CLIENT_IP = "aragora.server.handlers.auth.login.get_client_ip"


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_auth_handler():
    """Create a mock AuthHandler instance."""
    handler_instance = MagicMock()
    handler_instance._check_permission.return_value = None
    return handler_instance


@pytest.fixture
def mock_user_store():
    """Create a mock user store."""
    store = MagicMock()
    store.get_user_by_email.return_value = None
    store.get_user_by_id.return_value = None
    store.update_user.return_value = None
    # Mock is_account_locked to return (is_locked, lockout_until, failed_attempts)
    store.is_account_locked.return_value = (False, None, 0)
    store.record_failed_login.return_value = (1, None)  # (attempts, lockout_until)
    store.reset_failed_login_attempts.return_value = None
    return store


@pytest.fixture
def mock_user():
    """Create a mock authenticated user."""
    user = MagicMock()
    user.id = "user-001"
    user.email = "test@example.com"
    user.name = "Test User"
    user.org_id = "org-001"
    user.role = "user"
    user.is_active = True
    user.mfa_enabled = False
    user.mfa_secret = None

    def verify_password(password: str) -> bool:
        return password == "correct-password"

    user.verify_password = verify_password
    user.to_dict.return_value = {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "role": user.role,
    }

    return user


@pytest.fixture
def mock_user_with_mfa(mock_user):
    """Create a mock user with MFA enabled."""
    mock_user.mfa_enabled = True
    mock_user.mfa_secret = "JBSWY3DPEHPK3PXP"
    return mock_user


@pytest.fixture
def mock_lockout_tracker():
    """Create a mock lockout tracker."""
    tracker = MagicMock()
    tracker.is_locked.return_value = False
    tracker.record_failure.return_value = (1, None)  # (attempts, lockout_seconds)
    tracker.reset.return_value = None
    tracker.get_remaining_time.return_value = 0
    return tracker


@pytest.fixture
def mock_tokens():
    """Create mock token pair."""
    tokens = MagicMock()
    tokens.to_dict.return_value = {
        "access_token": "jwt-access-token",
        "refresh_token": "jwt-refresh-token",
    }
    return tokens


@pytest.fixture
def mock_http_handler():
    """Factory to create mock HTTP handler."""

    def _create(method: str = "POST", body: dict = None):
        mock = MagicMock()
        mock.command = method

        if body is not None:
            body_bytes = json.dumps(body).encode()
        else:
            body_bytes = b"{}"

        mock.rfile = MagicMock()
        mock.rfile.read = MagicMock(return_value=body_bytes)
        mock.headers = {"Content-Length": str(len(body_bytes))}
        mock.client_address = ("127.0.0.1", 12345)

        return mock

    return _create


# ============================================================================
# Test: User Registration (handle_register)
# ============================================================================


class TestUserRegistration:
    """Tests for handle_register."""

    def test_register_creates_user_successfully(
        self, mock_auth_handler, mock_user_store, mock_http_handler, mock_tokens
    ):
        """Test successful user registration."""
        from aragora.server.handlers.auth.login import handle_register

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_auth_handler.read_json_body.return_value = {
            "email": "new@example.com",
            "password": "SecurePass123!",
            "name": "New User",
        }

        # No existing user
        mock_user_store.get_user_by_email.return_value = None

        # Mock created user
        created_user = MagicMock()
        created_user.id = "new-user-001"
        created_user.email = "new@example.com"
        created_user.org_id = None
        created_user.role = "user"
        created_user.to_dict.return_value = {"id": "new-user-001", "email": "new@example.com"}
        mock_user_store.create_user.return_value = created_user

        http = mock_http_handler(
            method="POST",
            body={
                "email": "new@example.com",
                "password": "SecurePass123!",
                "name": "New User",
            },
        )

        with (
            patch(PATCH_CREATE_TOKEN_PAIR, return_value=mock_tokens),
            patch(PATCH_HASH_PASSWORD, return_value=("hash", "salt")),
        ):
            result = handle_register(mock_auth_handler, http)

        assert result.status_code == 201
        body = parse_handler_response(result)
        assert "user" in body
        assert "tokens" in body

    def test_register_validates_email_format(
        self, mock_auth_handler, mock_user_store, mock_http_handler
    ):
        """Test that registration validates email format."""
        from aragora.server.handlers.auth.login import handle_register

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_auth_handler.read_json_body.return_value = {
            "email": "invalid-email",
            "password": "SecurePass123!",
        }

        http = mock_http_handler(
            method="POST",
            body={
                "email": "invalid-email",
                "password": "SecurePass123!",
            },
        )

        result = handle_register(mock_auth_handler, http)

        assert_error_response(result, 400)

    def test_register_validates_password_strength(
        self, mock_auth_handler, mock_user_store, mock_http_handler
    ):
        """Test that registration validates password strength."""
        from aragora.server.handlers.auth.login import handle_register

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_auth_handler.read_json_body.return_value = {
            "email": "new@example.com",
            "password": "weak",
        }

        http = mock_http_handler(
            method="POST",
            body={
                "email": "new@example.com",
                "password": "weak",
            },
        )

        result = handle_register(mock_auth_handler, http)

        assert_error_response(result, 400)

    def test_register_rejects_duplicate_email(
        self, mock_auth_handler, mock_user_store, mock_user, mock_http_handler
    ):
        """Test that registration fails for existing email."""
        from aragora.server.handlers.auth.login import handle_register

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_auth_handler.read_json_body.return_value = {
            "email": "test@example.com",
            "password": "SecurePass123!",
        }

        # Existing user with same email
        mock_user_store.get_user_by_email.return_value = mock_user

        http = mock_http_handler(
            method="POST",
            body={
                "email": "test@example.com",
                "password": "SecurePass123!",
            },
        )

        result = handle_register(mock_auth_handler, http)

        assert_error_response(result, 409, "already registered")

    def test_register_email_case_insensitive(
        self, mock_auth_handler, mock_user_store, mock_http_handler, mock_tokens
    ):
        """Test that email is normalized to lowercase."""
        from aragora.server.handlers.auth.login import handle_register

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_auth_handler.read_json_body.return_value = {
            "email": "TEST@EXAMPLE.COM",
            "password": "SecurePass123!",
        }
        mock_user_store.get_user_by_email.return_value = None

        created_user = MagicMock()
        created_user.id = "new-user-001"
        created_user.email = "test@example.com"
        created_user.org_id = None
        created_user.role = "user"
        created_user.to_dict.return_value = {"id": "new-user-001"}
        mock_user_store.create_user.return_value = created_user

        http = mock_http_handler(
            method="POST",
            body={
                "email": "TEST@EXAMPLE.COM",
                "password": "SecurePass123!",
            },
        )

        with (
            patch(PATCH_CREATE_TOKEN_PAIR, return_value=mock_tokens),
            patch(PATCH_HASH_PASSWORD, return_value=("hash", "salt")),
        ):
            result = handle_register(mock_auth_handler, http)

        # Email should be lowercase in call to get_user_by_email
        mock_user_store.get_user_by_email.assert_called_with("test@example.com")

    def test_register_creates_organization(
        self, mock_auth_handler, mock_user_store, mock_http_handler, mock_tokens
    ):
        """Test that organization is created when name provided."""
        from aragora.server.handlers.auth.login import handle_register

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_auth_handler.read_json_body.return_value = {
            "email": "new@example.com",
            "password": "SecurePass123!",
            "organization": "My Company",
        }
        mock_user_store.get_user_by_email.return_value = None

        created_user = MagicMock()
        created_user.id = "new-user-001"
        created_user.email = "new@example.com"
        created_user.org_id = "org-001"
        created_user.role = "owner"
        created_user.to_dict.return_value = {"id": "new-user-001"}
        mock_user_store.create_user.return_value = created_user
        mock_user_store.get_user_by_id.return_value = created_user

        http = mock_http_handler(
            method="POST",
            body={
                "email": "new@example.com",
                "password": "SecurePass123!",
                "organization": "My Company",
            },
        )

        with (
            patch(PATCH_CREATE_TOKEN_PAIR, return_value=mock_tokens),
            patch(PATCH_HASH_PASSWORD, return_value=("hash", "salt")),
        ):
            result = handle_register(mock_auth_handler, http)

        # Organization should be created
        mock_user_store.create_organization.assert_called_once_with(
            name="My Company",
            owner_id="new-user-001",
        )

    def test_register_returns_tokens(
        self, mock_auth_handler, mock_user_store, mock_http_handler, mock_tokens
    ):
        """Test that registration returns JWT tokens."""
        from aragora.server.handlers.auth.login import handle_register

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_auth_handler.read_json_body.return_value = {
            "email": "new@example.com",
            "password": "SecurePass123!",
        }
        mock_user_store.get_user_by_email.return_value = None

        created_user = MagicMock()
        created_user.id = "new-user-001"
        created_user.email = "new@example.com"
        created_user.org_id = None
        created_user.role = "user"
        created_user.to_dict.return_value = {"id": "new-user-001"}
        mock_user_store.create_user.return_value = created_user

        http = mock_http_handler(
            method="POST",
            body={
                "email": "new@example.com",
                "password": "SecurePass123!",
            },
        )

        with (
            patch(PATCH_CREATE_TOKEN_PAIR, return_value=mock_tokens),
            patch(PATCH_HASH_PASSWORD, return_value=("hash", "salt")),
        ):
            result = handle_register(mock_auth_handler, http)

        body = parse_handler_response(result)
        assert "access_token" in body["tokens"]
        assert "refresh_token" in body["tokens"]

    def test_register_user_service_unavailable(self, mock_auth_handler, mock_http_handler):
        """Test registration when user store unavailable."""
        from aragora.server.handlers.auth.login import handle_register

        mock_auth_handler._get_user_store.return_value = None
        mock_auth_handler.read_json_body.return_value = {
            "email": "new@example.com",
            "password": "SecurePass123!",
        }

        http = mock_http_handler(
            method="POST",
            body={
                "email": "new@example.com",
                "password": "SecurePass123!",
            },
        )

        result = handle_register(mock_auth_handler, http)

        assert_error_response(result, 503, "unavailable")


# ============================================================================
# Test: User Login (handle_login)
# ============================================================================


class TestUserLogin:
    """Tests for handle_login."""

    def test_login_with_valid_credentials_returns_tokens(
        self,
        mock_auth_handler,
        mock_user_store,
        mock_user,
        mock_lockout_tracker,
        mock_tokens,
        mock_http_handler,
    ):
        """Test successful login returns tokens."""
        from aragora.server.handlers.auth.login import handle_login

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_auth_handler.read_json_body.return_value = {
            "email": "test@example.com",
            "password": "correct-password",
        }
        mock_user_store.get_user_by_email.return_value = mock_user

        http = mock_http_handler(
            method="POST",
            body={
                "email": "test@example.com",
                "password": "correct-password",
            },
        )

        with (
            patch(PATCH_GET_LOCKOUT_TRACKER, return_value=mock_lockout_tracker),
            patch(PATCH_CREATE_TOKEN_PAIR, return_value=mock_tokens),
            patch(PATCH_GET_CLIENT_IP, return_value="127.0.0.1"),
        ):
            result = handle_login(mock_auth_handler, http)

        assert result.status_code == 200
        body = parse_handler_response(result)
        assert "tokens" in body
        assert "user" in body

    def test_login_with_invalid_password_returns_401(
        self, mock_auth_handler, mock_user_store, mock_user, mock_lockout_tracker, mock_http_handler
    ):
        """Test login fails with wrong password."""
        from aragora.server.handlers.auth.login import handle_login

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_auth_handler.read_json_body.return_value = {
            "email": "test@example.com",
            "password": "wrong-password",
        }
        mock_user_store.get_user_by_email.return_value = mock_user

        http = mock_http_handler(
            method="POST",
            body={
                "email": "test@example.com",
                "password": "wrong-password",
            },
        )

        with (
            patch(PATCH_GET_LOCKOUT_TRACKER, return_value=mock_lockout_tracker),
            patch(PATCH_GET_CLIENT_IP, return_value="127.0.0.1"),
        ):
            result = handle_login(mock_auth_handler, http)

        assert_error_response(result, 401, "Invalid")

    def test_login_with_unknown_email_returns_401(
        self, mock_auth_handler, mock_user_store, mock_lockout_tracker, mock_http_handler
    ):
        """Test login fails with unknown email."""
        from aragora.server.handlers.auth.login import handle_login

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_auth_handler.read_json_body.return_value = {
            "email": "unknown@example.com",
            "password": "some-password",
        }
        mock_user_store.get_user_by_email.return_value = None

        http = mock_http_handler(
            method="POST",
            body={
                "email": "unknown@example.com",
                "password": "some-password",
            },
        )

        with (
            patch(PATCH_GET_LOCKOUT_TRACKER, return_value=mock_lockout_tracker),
            patch(PATCH_GET_CLIENT_IP, return_value="127.0.0.1"),
        ):
            result = handle_login(mock_auth_handler, http)

        # Same error to prevent email enumeration
        assert_error_response(result, 401, "Invalid")

    def test_login_increments_failed_attempts(
        self, mock_auth_handler, mock_user_store, mock_user, mock_lockout_tracker, mock_http_handler
    ):
        """Test that failed login increments attempt counter."""
        from aragora.server.handlers.auth.login import handle_login

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_auth_handler.read_json_body.return_value = {
            "email": "test@example.com",
            "password": "wrong-password",
        }
        mock_user_store.get_user_by_email.return_value = mock_user

        http = mock_http_handler(
            method="POST",
            body={
                "email": "test@example.com",
                "password": "wrong-password",
            },
        )

        with (
            patch(PATCH_GET_LOCKOUT_TRACKER, return_value=mock_lockout_tracker),
            patch(PATCH_GET_CLIENT_IP, return_value="127.0.0.1"),
        ):
            result = handle_login(mock_auth_handler, http)

        mock_lockout_tracker.record_failure.assert_called()

    def test_login_locks_account_after_max_failures(
        self, mock_auth_handler, mock_user_store, mock_user, mock_lockout_tracker, mock_http_handler
    ):
        """Test that account is locked after too many failures."""
        from aragora.server.handlers.auth.login import handle_login

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_auth_handler.read_json_body.return_value = {
            "email": "test@example.com",
            "password": "wrong-password",
        }
        mock_user_store.get_user_by_email.return_value = mock_user

        # Simulate lockout after this failure
        mock_lockout_tracker.record_failure.return_value = (5, 900)  # 5 attempts, 15 min lockout

        http = mock_http_handler(
            method="POST",
            body={
                "email": "test@example.com",
                "password": "wrong-password",
            },
        )

        with (
            patch(PATCH_GET_LOCKOUT_TRACKER, return_value=mock_lockout_tracker),
            patch(PATCH_GET_CLIENT_IP, return_value="127.0.0.1"),
        ):
            result = handle_login(mock_auth_handler, http)

        assert_error_response(result, 429, "locked")

    def test_login_blocked_when_account_locked(
        self, mock_auth_handler, mock_user_store, mock_lockout_tracker, mock_http_handler
    ):
        """Test that locked account cannot login."""
        from aragora.server.handlers.auth.login import handle_login

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_auth_handler.read_json_body.return_value = {
            "email": "test@example.com",
            "password": "correct-password",
        }

        # Account is already locked
        mock_lockout_tracker.is_locked.return_value = True
        mock_lockout_tracker.get_remaining_time.return_value = 600  # 10 minutes

        http = mock_http_handler(
            method="POST",
            body={
                "email": "test@example.com",
                "password": "correct-password",
            },
        )

        with (
            patch(PATCH_GET_LOCKOUT_TRACKER, return_value=mock_lockout_tracker),
            patch(PATCH_GET_CLIENT_IP, return_value="127.0.0.1"),
        ):
            result = handle_login(mock_auth_handler, http)

        assert_error_response(result, 429, "Too many")

    def test_login_resets_attempts_on_success(
        self,
        mock_auth_handler,
        mock_user_store,
        mock_user,
        mock_lockout_tracker,
        mock_tokens,
        mock_http_handler,
    ):
        """Test that successful login resets attempt counter."""
        from aragora.server.handlers.auth.login import handle_login

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_auth_handler.read_json_body.return_value = {
            "email": "test@example.com",
            "password": "correct-password",
        }
        mock_user_store.get_user_by_email.return_value = mock_user

        http = mock_http_handler(
            method="POST",
            body={
                "email": "test@example.com",
                "password": "correct-password",
            },
        )

        with (
            patch(PATCH_GET_LOCKOUT_TRACKER, return_value=mock_lockout_tracker),
            patch(PATCH_CREATE_TOKEN_PAIR, return_value=mock_tokens),
            patch(PATCH_GET_CLIENT_IP, return_value="127.0.0.1"),
        ):
            result = handle_login(mock_auth_handler, http)

        mock_lockout_tracker.reset.assert_called()

    def test_login_returns_mfa_pending_if_enabled(
        self,
        mock_auth_handler,
        mock_user_store,
        mock_user_with_mfa,
        mock_lockout_tracker,
        mock_http_handler,
    ):
        """Test that MFA-enabled user gets pending token."""
        from aragora.server.handlers.auth.login import handle_login

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_auth_handler.read_json_body.return_value = {
            "email": "test@example.com",
            "password": "correct-password",
        }
        mock_user_store.get_user_by_email.return_value = mock_user_with_mfa

        http = mock_http_handler(
            method="POST",
            body={
                "email": "test@example.com",
                "password": "correct-password",
            },
        )

        with (
            patch(PATCH_GET_LOCKOUT_TRACKER, return_value=mock_lockout_tracker),
            patch(PATCH_CREATE_MFA_PENDING, return_value="mfa-pending-token"),
            patch(PATCH_GET_CLIENT_IP, return_value="127.0.0.1"),
        ):
            result = handle_login(mock_auth_handler, http)

        assert result.status_code == 200
        body = parse_handler_response(result)
        assert body.get("mfa_required") is True
        assert "pending_token" in body

    def test_login_updates_last_login(
        self,
        mock_auth_handler,
        mock_user_store,
        mock_user,
        mock_lockout_tracker,
        mock_tokens,
        mock_http_handler,
    ):
        """Test that login updates last_login_at timestamp."""
        from aragora.server.handlers.auth.login import handle_login

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_auth_handler.read_json_body.return_value = {
            "email": "test@example.com",
            "password": "correct-password",
        }
        mock_user_store.get_user_by_email.return_value = mock_user

        http = mock_http_handler(
            method="POST",
            body={
                "email": "test@example.com",
                "password": "correct-password",
            },
        )

        with (
            patch(PATCH_GET_LOCKOUT_TRACKER, return_value=mock_lockout_tracker),
            patch(PATCH_CREATE_TOKEN_PAIR, return_value=mock_tokens),
            patch(PATCH_GET_CLIENT_IP, return_value="127.0.0.1"),
        ):
            result = handle_login(mock_auth_handler, http)

        # Should update last_login_at
        mock_user_store.update_user.assert_called()
        call_kwargs = mock_user_store.update_user.call_args.kwargs
        assert "last_login_at" in call_kwargs

    def test_login_disabled_account(
        self, mock_auth_handler, mock_user_store, mock_user, mock_lockout_tracker, mock_http_handler
    ):
        """Test that disabled account cannot login."""
        from aragora.server.handlers.auth.login import handle_login

        mock_user.is_active = False

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_auth_handler.read_json_body.return_value = {
            "email": "test@example.com",
            "password": "correct-password",
        }
        mock_user_store.get_user_by_email.return_value = mock_user

        http = mock_http_handler(
            method="POST",
            body={
                "email": "test@example.com",
                "password": "correct-password",
            },
        )

        with (
            patch(PATCH_GET_LOCKOUT_TRACKER, return_value=mock_lockout_tracker),
            patch(PATCH_GET_CLIENT_IP, return_value="127.0.0.1"),
        ):
            result = handle_login(mock_auth_handler, http)

        assert_error_response(result, 401, "disabled")

    def test_login_requires_email_and_password(self, mock_auth_handler, mock_http_handler):
        """Test that login requires both email and password."""
        from aragora.server.handlers.auth.login import handle_login

        mock_auth_handler.read_json_body.return_value = {"email": "test@example.com"}

        http = mock_http_handler(method="POST", body={"email": "test@example.com"})

        result = handle_login(mock_auth_handler, http)

        assert_error_response(result, 400, "required")

    def test_login_service_unavailable(
        self, mock_auth_handler, mock_lockout_tracker, mock_http_handler
    ):
        """Test login when user store unavailable."""
        from aragora.server.handlers.auth.login import handle_login

        mock_auth_handler._get_user_store.return_value = None
        mock_auth_handler.read_json_body.return_value = {
            "email": "test@example.com",
            "password": "password",
        }

        http = mock_http_handler(
            method="POST",
            body={
                "email": "test@example.com",
                "password": "password",
            },
        )

        with (
            patch(PATCH_GET_LOCKOUT_TRACKER, return_value=mock_lockout_tracker),
            patch(PATCH_GET_CLIENT_IP, return_value="127.0.0.1"),
        ):
            result = handle_login(mock_auth_handler, http)

        assert_error_response(result, 503, "unavailable")

    def test_login_email_case_insensitive(
        self,
        mock_auth_handler,
        mock_user_store,
        mock_user,
        mock_lockout_tracker,
        mock_tokens,
        mock_http_handler,
    ):
        """Test that email is normalized to lowercase for lookup."""
        from aragora.server.handlers.auth.login import handle_login

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_auth_handler.read_json_body.return_value = {
            "email": "TEST@EXAMPLE.COM",
            "password": "correct-password",
        }
        mock_user_store.get_user_by_email.return_value = mock_user

        http = mock_http_handler(
            method="POST",
            body={
                "email": "TEST@EXAMPLE.COM",
                "password": "correct-password",
            },
        )

        with (
            patch(PATCH_GET_LOCKOUT_TRACKER, return_value=mock_lockout_tracker),
            patch(PATCH_CREATE_TOKEN_PAIR, return_value=mock_tokens),
            patch(PATCH_GET_CLIENT_IP, return_value="127.0.0.1"),
        ):
            result = handle_login(mock_auth_handler, http)

        # Should lookup lowercase email
        mock_user_store.get_user_by_email.assert_called_with("test@example.com")


# ============================================================================
# Test: Security Properties
# ============================================================================


class TestLoginSecurityProperties:
    """Tests for login security properties."""

    def test_same_error_for_unknown_email_and_wrong_password(
        self, mock_auth_handler, mock_user_store, mock_user, mock_lockout_tracker, mock_http_handler
    ):
        """Test that same error is returned for unknown email and wrong password."""
        from aragora.server.handlers.auth.login import handle_login

        mock_auth_handler._get_user_store.return_value = mock_user_store

        http = mock_http_handler(
            method="POST",
            body={
                "email": "unknown@example.com",
                "password": "password",
            },
        )

        # Unknown email
        mock_user_store.get_user_by_email.return_value = None
        mock_auth_handler.read_json_body.return_value = {
            "email": "unknown@example.com",
            "password": "password",
        }

        with (
            patch(PATCH_GET_LOCKOUT_TRACKER, return_value=mock_lockout_tracker),
            patch(PATCH_GET_CLIENT_IP, return_value="127.0.0.1"),
        ):
            result_unknown = handle_login(mock_auth_handler, http)

        # Wrong password
        mock_user_store.get_user_by_email.return_value = mock_user
        mock_auth_handler.read_json_body.return_value = {
            "email": "test@example.com",
            "password": "wrong-password",
        }

        with (
            patch(PATCH_GET_LOCKOUT_TRACKER, return_value=mock_lockout_tracker),
            patch(PATCH_GET_CLIENT_IP, return_value="127.0.0.1"),
        ):
            result_wrong = handle_login(mock_auth_handler, http)

        # Both should be 401 with similar message to prevent enumeration
        assert result_unknown.status_code == 401
        assert result_wrong.status_code == 401

    def test_lockout_tracks_by_ip(
        self, mock_auth_handler, mock_user_store, mock_lockout_tracker, mock_http_handler
    ):
        """Test that lockout tracks by IP address."""
        from aragora.server.handlers.auth.login import handle_login

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_email.return_value = None
        mock_auth_handler.read_json_body.return_value = {
            "email": "unknown@example.com",
            "password": "password",
        }

        http = mock_http_handler(
            method="POST",
            body={
                "email": "unknown@example.com",
                "password": "password",
            },
        )

        with (
            patch(PATCH_GET_LOCKOUT_TRACKER, return_value=mock_lockout_tracker),
            patch(PATCH_GET_CLIENT_IP, return_value="192.168.1.1"),
        ):
            result = handle_login(mock_auth_handler, http)

        # Should record failure with IP
        mock_lockout_tracker.record_failure.assert_called_with(
            email="unknown@example.com",
            ip="192.168.1.1",
        )


__all__ = [
    "TestUserRegistration",
    "TestUserLogin",
    "TestLoginSecurityProperties",
]
