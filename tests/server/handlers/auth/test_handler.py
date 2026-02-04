"""
Tests for aragora.server.handlers.auth.handler - User authentication handler.

Comprehensive tests for AuthHandler covering:
- Registration success/failure flows
- Login success/failure flows (with and without MFA)
- Token refresh flows
- Logout and logout-all flows
- Profile get/update and password change
- MFA setup/enable/disable/verify flows
- API key generation/revocation/listing
- Session listing and revocation
- Backup codes regeneration
- Route handling and dispatching
- Edge cases and error handling

Target: 80%+ coverage
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.server.handlers.auth.handler import AuthHandler


def maybe_await(result):
    """Handle sync/async results uniformly."""
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return result


# ===========================================================================
# Helper Functions
# ===========================================================================


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
# Test: Registration Flow
# ===========================================================================


class TestRegistration:
    """Test user registration flows."""

    def test_register_success_basic(self, auth_handler, mock_user_store, mock_user, mock_tokens):
        """Test successful registration with basic fields."""
        mock_user_store.get_user_by_email.return_value = None  # Email not taken
        mock_user_store.create_user.return_value = mock_user

        request = make_mock_handler(
            body={
                "email": "newuser@example.com",
                "password": "SecurePassword123!",
                "name": "New User",
            },
            command="POST",
        )

        with patch(
            "aragora.billing.models.hash_password",
            return_value=("hashed", "salt"),
        ):
            with patch(
                "aragora.billing.jwt_auth.create_token_pair",
                return_value=mock_tokens,
            ):
                result = auth_handler._handle_register(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        assert parsed["status_code"] == 201
        assert "user" in parsed
        assert "tokens" in parsed

    def test_register_success_with_organization(
        self, auth_handler, mock_user_store, mock_user, mock_tokens
    ):
        """Test successful registration with organization name."""
        mock_user_store.get_user_by_email.return_value = None
        mock_user_store.create_user.return_value = mock_user

        request = make_mock_handler(
            body={
                "email": "admin@company.com",
                "password": "SecurePassword123!",
                "name": "Company Admin",
                "organization": "My Company Inc",
            },
            command="POST",
        )

        with patch(
            "aragora.billing.models.hash_password",
            return_value=("hashed", "salt"),
        ):
            with patch(
                "aragora.billing.jwt_auth.create_token_pair",
                return_value=mock_tokens,
            ):
                result = auth_handler._handle_register(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        assert parsed["status_code"] == 201
        mock_user_store.create_organization.assert_called_once()

    def test_register_email_already_exists(self, auth_handler, mock_user_store, mock_user):
        """Test registration fails when email exists."""
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

    def test_register_invalid_email(self, auth_handler):
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

    def test_register_weak_password(self, auth_handler, mock_user_store):
        """Test registration fails with weak password."""
        mock_user_store.get_user_by_email.return_value = None

        request = make_mock_handler(
            body={
                "email": "user@example.com",
                "password": "short",  # Too short
            },
            command="POST",
        )

        result = auth_handler._handle_register(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 400

    def test_register_missing_email(self, auth_handler):
        """Test registration fails with missing email."""
        request = make_mock_handler(
            body={
                "password": "SecurePassword123!",
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


# ===========================================================================
# Test: Login Flow
# ===========================================================================


class TestLogin:
    """Test login flows."""

    def test_login_success_no_mfa(
        self, auth_handler, mock_user_store, mock_user, mock_tokens, mock_lockout_tracker
    ):
        """Test successful login without MFA."""
        mock_user_store.get_user_by_email.return_value = mock_user

        request = make_mock_handler(
            body={
                "email": "test@example.com",
                "password": "correctpassword",
            },
            command="POST",
        )

        with patch(
            "aragora.auth.lockout.get_lockout_tracker",
            return_value=mock_lockout_tracker,
        ):
            with patch(
                "aragora.billing.jwt_auth.create_token_pair",
                return_value=mock_tokens,
            ):
                result = auth_handler._handle_login(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        assert parsed["status_code"] == 200
        assert "user" in parsed
        assert "tokens" in parsed

    def test_login_success_with_mfa_pending(
        self, auth_handler, mock_user_store, mock_mfa_user, mock_lockout_tracker
    ):
        """Test login returns pending token when MFA is enabled."""
        mock_user_store.get_user_by_email.return_value = mock_mfa_user

        request = make_mock_handler(
            body={
                "email": "test@example.com",
                "password": "correctpassword",
            },
            command="POST",
        )

        with patch(
            "aragora.auth.lockout.get_lockout_tracker",
            return_value=mock_lockout_tracker,
        ):
            with patch(
                "aragora.billing.jwt_auth.create_mfa_pending_token",
                return_value="pending_token_xyz",
            ):
                result = auth_handler._handle_login(request)

        parsed = parse_result(result)
        assert parsed["success"] is True
        assert parsed.get("mfa_required") is True
        assert "pending_token" in parsed

    def test_login_wrong_password(
        self, auth_handler, mock_user_store, mock_user, mock_lockout_tracker
    ):
        """Test login fails with wrong password."""
        mock_user.verify_password.return_value = False
        mock_user_store.get_user_by_email.return_value = mock_user

        request = make_mock_handler(
            body={
                "email": "test@example.com",
                "password": "wrongpassword",
            },
            command="POST",
        )

        with patch(
            "aragora.auth.lockout.get_lockout_tracker",
            return_value=mock_lockout_tracker,
        ):
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

        with patch(
            "aragora.auth.lockout.get_lockout_tracker",
            return_value=mock_lockout_tracker,
        ):
            result = auth_handler._handle_login(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 401
        assert "disabled" in parsed.get("error", "").lower()

    def test_login_account_locked(self, auth_handler, mock_user_store, mock_user):
        """Test login fails when account is locked."""
        mock_user_store.get_user_by_email.return_value = mock_user

        request = make_mock_handler(
            body={
                "email": "test@example.com",
                "password": "password",
            },
            command="POST",
        )

        # Create a fresh mock tracker that's locked
        locked_tracker = MagicMock()
        locked_tracker.is_locked.return_value = True
        locked_tracker.get_remaining_time.return_value = 300

        with patch(
            "aragora.server.handlers.auth.handler.get_lockout_tracker",
            return_value=locked_tracker,
        ):
            result = auth_handler._handle_login(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 429
        assert "too many" in parsed.get("error", "").lower()

    def test_login_user_not_found(self, auth_handler, mock_user_store, mock_lockout_tracker):
        """Test login fails when user not found."""
        mock_user_store.get_user_by_email.return_value = None

        request = make_mock_handler(
            body={
                "email": "nonexistent@example.com",
                "password": "password",
            },
            command="POST",
        )

        with patch(
            "aragora.auth.lockout.get_lockout_tracker",
            return_value=mock_lockout_tracker,
        ):
            result = auth_handler._handle_login(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 401

    def test_login_missing_credentials(self, auth_handler):
        """Test login fails with missing credentials."""
        request = make_mock_handler(
            body={},
            command="POST",
        )

        result = auth_handler._handle_login(request)
        parsed = parse_result(result)

        assert parsed["success"] is False
        assert parsed["status_code"] == 400


# ===========================================================================
# Test: Token Refresh Flow
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

        # Patch at handler module level since it imports directly
        with patch(
            "aragora.server.handlers.auth.handler.validate_refresh_token",
            return_value=mock_payload,
        ):
            with patch(
                "aragora.billing.jwt_auth.revoke_token_persistent",
                return_value=True,
            ):
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
            "aragora.billing.jwt_auth.validate_refresh_token",
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
            "aragora.billing.jwt_auth.validate_refresh_token",
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
            "aragora.billing.jwt_auth.validate_refresh_token",
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


# ===========================================================================
# Test: Logout Flow
# ===========================================================================


class TestLogout:
    """Test logout flows."""

    def test_logout_success(self, auth_handler, mock_user_store, mock_auth_context):
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

                with patch(
                    "aragora.billing.jwt_auth.revoke_token_persistent",
                    return_value=True,
                ):
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
        """Test successful logout from all devices."""
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

                with patch(
                    "aragora.billing.jwt_auth.revoke_token_persistent",
                    return_value=True,
                ):
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


# ===========================================================================
# Test: Profile Operations
# ===========================================================================


class TestProfileOperations:
    """Test profile operations."""

    def test_get_me_success(
        self, auth_handler, mock_user_store, mock_user, mock_auth_context, mock_org
    ):
        """Test successful get current user info."""
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

                with patch(
                    "aragora.billing.models.hash_password",
                    return_value=("newhash", "newsalt"),
                ):
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

    def test_change_password_weak_new_password(
        self, auth_handler, mock_user_store, mock_user, mock_auth_context
    ):
        """Test password change fails with weak new password."""
        mock_user.verify_password.return_value = True
        mock_user_store.get_user_by_id.return_value = mock_user

        request = make_mock_handler(
            body={
                "current_password": "oldpassword",
                "new_password": "weak",  # Too short
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
        assert parsed["status_code"] == 400


# ===========================================================================
# Test: MFA Operations
# ===========================================================================


class TestMFAOperations:
    """Test MFA operations."""

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
        assert "already enabled" in parsed.get("error", "").lower()

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

    def test_mfa_enable_no_secret_setup(
        self, auth_handler, mock_user_store, mock_user, mock_auth_context
    ):
        """Test MFA enable fails when no secret has been set up."""
        mock_user.mfa_enabled = False
        mock_user.mfa_secret = None
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

                result = auth_handler._handle_mfa_enable(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 400
        assert "not set up" in parsed.get("error", "").lower()

    def test_mfa_disable_success_with_code(
        self, auth_handler, mock_user_store, mock_mfa_user, mock_auth_context
    ):
        """Test successful MFA disable with MFA code."""
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

    def test_mfa_disable_success_with_password(
        self, auth_handler, mock_user_store, mock_mfa_user, mock_auth_context
    ):
        """Test successful MFA disable with password."""
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

    def test_mfa_verify_success_with_totp(
        self, auth_handler, mock_user_store, mock_mfa_user, mock_tokens
    ):
        """Test successful MFA verification with TOTP code."""
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

    def test_mfa_verify_success_with_backup_code(
        self, auth_handler, mock_user_store, mock_mfa_user, mock_tokens
    ):
        """Test successful MFA verification with backup code."""
        # Set up backup codes properly
        backup_codes = ["backup0", "backup1", "backup2"]
        backup_hashes = [hashlib.sha256(c.encode()).hexdigest() for c in backup_codes]
        mock_mfa_user.mfa_backup_codes = json.dumps(backup_hashes)
        mock_user_store.get_user_by_id.return_value = mock_mfa_user

        request = make_mock_handler(
            body={
                "code": "backup0",  # Use first backup code
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

    def test_mfa_verify_invalid_pending_token(self, auth_handler, mock_user_store):
        """Test MFA verify fails with invalid pending token."""
        request = make_mock_handler(
            body={
                "code": "123456",
                "pending_token": "invalid_pending",
            },
            command="POST",
        )

        with patch(
            "aragora.billing.jwt_auth.validate_mfa_pending_token",
            return_value=None,
        ):
            result = auth_handler._handle_mfa_verify(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 401

    def test_regenerate_backup_codes_success(
        self, auth_handler, mock_user_store, mock_mfa_user, mock_auth_context
    ):
        """Test successful backup codes regeneration."""
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

    def test_regenerate_backup_codes_mfa_not_enabled(
        self, auth_handler, mock_user_store, mock_user, mock_auth_context
    ):
        """Test backup codes regeneration fails when MFA not enabled."""
        mock_user.mfa_enabled = False
        mock_user.mfa_secret = None
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

                result = auth_handler._handle_mfa_backup_codes(request)

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 400


# ===========================================================================
# Test: API Key Management
# ===========================================================================


class TestAPIKeyManagement:
    """Test API key management operations."""

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

    def test_list_api_keys_none(self, auth_handler, mock_user_store, mock_user, mock_auth_context):
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

    def test_revoke_api_key_by_prefix_success(
        self, auth_handler, mock_user_store, mock_user, mock_auth_context
    ):
        """Test successful API key revocation by prefix."""
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
# Test: Session Management
# ===========================================================================


class TestSessionManagement:
    """Test session management operations."""

    def test_list_sessions_success(self, auth_handler, mock_user_store, mock_auth_context):
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

    def test_revoke_session_success(self, auth_handler, mock_user_store, mock_auth_context):
        """Test successful session revocation."""
        request = make_mock_handler(command="DELETE")
        request.headers["Authorization"] = "Bearer test_token"

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
        assert parsed.get("session_id") == "session-to-revoke"

    def test_revoke_session_not_found(self, auth_handler, mock_user_store, mock_auth_context):
        """Test session revocation fails when session not found."""
        request = make_mock_handler(command="DELETE")
        request.headers["Authorization"] = "Bearer test_token"

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
                        mock_decode.return_value = MagicMock(jti="current-jti")

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

    def test_revoke_current_session_rejected(
        self, auth_handler, mock_user_store, mock_auth_context
    ):
        """Test cannot revoke current session."""
        request = make_mock_handler(command="DELETE")
        request.headers["Authorization"] = "Bearer test_token"

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
                        mock_decode.return_value = MagicMock(jti="current-session")

                        result = auth_handler._handle_revoke_session(request, "current-session")

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 400
        assert "current session" in parsed.get("error", "").lower()

    def test_invalid_session_id_format(self, auth_handler, mock_user_store, mock_auth_context):
        """Test session revocation rejects invalid session ID format."""
        request = make_mock_handler(command="DELETE")

        with patch(
            "aragora.server.handlers.auth.handler.extract_user_from_request",
            return_value=mock_auth_context,
        ):
            with patch("aragora.server.handlers.auth.handler.check_permission") as mock_check:
                mock_check.return_value = MagicMock(allowed=True)

                result = auth_handler._handle_revoke_session(request, "ab")  # Too short

        parsed = parse_result(result)
        assert parsed["success"] is False
        assert parsed["status_code"] == 400


# ===========================================================================
# Test: Token Revocation
# ===========================================================================


class TestTokenRevocation:
    """Test token revocation operations."""

    def test_revoke_token_success(self, auth_handler, mock_user_store, mock_auth_context):
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

    def test_revoke_current_token_when_no_token_in_body(
        self, auth_handler, mock_user_store, mock_auth_context
    ):
        """Test revoke uses current token when none in body."""
        request = make_mock_handler(
            body={},
            command="POST",
        )
        request.headers["Authorization"] = "Bearer current_token"

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
                    "aragora.server.middleware.auth.extract_token",
                    return_value="current_token",
                ):
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

    def test_can_handle_versioned_routes(self, auth_handler):
        """Test handler can handle versioned routes."""
        assert auth_handler.can_handle("/api/v1/auth/login") is True
        assert auth_handler.can_handle("/api/v1/auth/register") is True
        assert auth_handler.can_handle("/api/v1/auth/me") is True

    def test_cannot_handle_other_routes(self, auth_handler):
        """Test handler cannot handle unrelated routes."""
        assert auth_handler.can_handle("/api/debates") is False
        assert auth_handler.can_handle("/api/users") is False
        assert auth_handler.can_handle("/api/other") is False

    def test_handle_routes_to_register(self, auth_handler, mock_user_store):
        """Test POST /api/auth/register routes correctly."""
        mock_user_store.get_user_by_email.return_value = None

        request = make_mock_handler(
            body={
                "email": "test@example.com",
                "password": "SecurePassword123!",
            },
            command="POST",
        )

        with patch.object(auth_handler, "_handle_register") as mock_register:
            mock_register.return_value = MagicMock(status_code=201, body=b'{"success": true}')

            result = maybe_await(
                auth_handler.handle(
                    path="/api/auth/register",
                    query_params={},
                    handler=request,
                    method="POST",
                )
            )

            mock_register.assert_called_once()

    def test_handle_routes_to_login(self, auth_handler):
        """Test POST /api/auth/login routes correctly."""
        request = make_mock_handler(
            body={
                "email": "test@example.com",
                "password": "password",
            },
            command="POST",
        )

        with patch.object(auth_handler, "_handle_login") as mock_login:
            mock_login.return_value = MagicMock(status_code=200, body=b'{"success": true}')

            result = maybe_await(
                auth_handler.handle(
                    path="/api/auth/login",
                    query_params={},
                    handler=request,
                    method="POST",
                )
            )

            mock_login.assert_called_once()

    def test_handle_routes_v1_paths(self, auth_handler):
        """Test v1 versioned paths are handled correctly."""
        request = make_mock_handler(command="GET")

        with patch.object(auth_handler, "_handle_get_me") as mock_me:
            mock_me.return_value = MagicMock(status_code=200, body=b'{"user": {}}')

            result = maybe_await(
                auth_handler.handle(
                    path="/api/v1/auth/me",
                    query_params={},
                    handler=request,
                    method="GET",
                )
            )

            mock_me.assert_called_once()

    def test_handle_password_reset_enabled_by_default(self, auth_handler):
        """Test password reset endpoints are enabled by default (not 501)."""
        request = make_mock_handler(
            body={"email": "test@example.com"},
            command="POST",
        )

        result = maybe_await(
            auth_handler.handle(
                path="/api/auth/password/forgot",
                query_params={},
                handler=request,
                method="POST",
            )
        )

        parsed = parse_result(result)
        # Password reset is enabled by default - should return 200 (success response)
        # even for non-existent emails to prevent email enumeration
        assert parsed["status_code"] == 200

    def test_handle_password_reset_disabled_when_flag_false(self):
        """Test password reset endpoints return 501 when explicitly disabled."""
        handler = AuthHandler(ctx={"enable_password_reset_routes": False})
        request = make_mock_handler(
            body={"email": "test@example.com"},
            command="POST",
        )

        result = maybe_await(
            handler.handle(
                path="/api/auth/password/forgot",
                query_params={},
                handler=request,
                method="POST",
            )
        )

        parsed = parse_result(result)
        assert parsed["status_code"] == 501

    def test_handle_api_keys_routes(self, auth_handler, mock_auth_context):
        """Test /api/auth/api-keys routes correctly."""
        request = make_mock_handler(command="GET")

        with patch.object(auth_handler, "_handle_list_api_keys") as mock_list:
            mock_list.return_value = MagicMock(status_code=200, body=b'{"keys": []}')

            result = maybe_await(
                auth_handler.handle(
                    path="/api/auth/api-keys",
                    query_params={},
                    handler=request,
                    method="GET",
                )
            )

            mock_list.assert_called_once()

    def test_handle_api_keys_delete_with_prefix(self, auth_handler, mock_auth_context):
        """Test DELETE /api/auth/api-keys/:prefix routes correctly."""
        request = make_mock_handler(command="DELETE")

        with patch.object(auth_handler, "_handle_revoke_api_key_prefix") as mock_revoke:
            mock_revoke.return_value = MagicMock(status_code=200, body=b'{"message": "revoked"}')

            result = maybe_await(
                auth_handler.handle(
                    path="/api/auth/api-keys/ara_test123",
                    query_params={},
                    handler=request,
                    method="DELETE",
                )
            )

            mock_revoke.assert_called_once_with(request, "ara_test123")


# ===========================================================================
# Test: Error Handling
# ===========================================================================


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_refresh_revocation_failure_rollback(self, auth_handler, mock_user_store, mock_user):
        """Test token refresh handles revocation failure."""
        mock_user_store.get_user_by_id.return_value = mock_user

        request = make_mock_handler(
            body={"refresh_token": "valid_token"},
            command="POST",
        )

        mock_payload = MagicMock()
        mock_payload.user_id = "user-123"

        # Patch at handler module level since it imports directly at top
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


__all__ = [
    "TestRegistration",
    "TestLogin",
    "TestTokenRefresh",
    "TestLogout",
    "TestProfileOperations",
    "TestMFAOperations",
    "TestAPIKeyManagement",
    "TestSessionManagement",
    "TestTokenRevocation",
    "TestRouteHandling",
    "TestErrorHandling",
]
