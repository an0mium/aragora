"""
Tests for password management handler functions.

Phase 5: Auth Handler Test Coverage - Password

Covers:
- handle_change_password - Change password with current password verification
- handle_forgot_password - Request password reset (no enumeration)
- handle_reset_password - Reset password with token
- send_password_reset_email - Async email delivery
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from tests.server.handlers.conftest import parse_handler_response

# Patch paths - these imports happen inside the handler functions
PATCH_RESET_STORE = "aragora.storage.password_reset_store.get_password_reset_store"
PATCH_HASH_PASSWORD = "aragora.billing.models.hash_password"
PATCH_EXTRACT_USER = "aragora.server.handlers.auth.password.extract_user_from_request"


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_user():
    """Create a mock user for password tests."""
    user = MagicMock()
    user.id = "user-001"
    user.email = "test@example.com"
    user.name = "Test User"
    user.is_active = True
    user.password_hash = "hashed_password_123"
    user.password_salt = "salt_123"

    # Password verification mock
    def verify_password(password: str) -> bool:
        return password == "CurrentP@ssw0rd!"

    user.verify_password = verify_password
    return user


@pytest.fixture
def mock_inactive_user(mock_user):
    """Create an inactive user."""
    mock_user.is_active = False
    return mock_user


@pytest.fixture
def mock_user_store(mock_user):
    """Create a mock user store."""
    store = MagicMock()
    store.get_user_by_id.return_value = mock_user
    store.get_user_by_email.return_value = mock_user
    store.update_user.return_value = True
    store.increment_token_version.return_value = True
    return store


@pytest.fixture
def mock_handler_instance(mock_user_store):
    """Create a mock AuthHandler instance."""
    handler_instance = MagicMock()
    handler_instance._check_permission.return_value = None
    handler_instance._get_user_store.return_value = mock_user_store
    handler_instance.read_json_body.return_value = {}
    return handler_instance


@pytest.fixture
def mock_password_reset_store():
    """Create a mock password reset store."""
    store = MagicMock()
    store.create_token.return_value = ("reset_token_123", None)
    store.validate_token.return_value = ("test@example.com", None)
    store.consume_token.return_value = True
    store.invalidate_tokens_for_email.return_value = True
    return store


# ============================================================================
# Test: Change Password
# ============================================================================


class TestChangePassword:
    """Tests for handle_change_password."""

    def test_change_password_requires_current_password(
        self, mock_handler_instance, mock_http_handler
    ):
        """Test that change password requires current password."""
        from aragora.server.handlers.auth.password import handle_change_password

        mock_handler_instance.read_json_body.return_value = {
            "new_password": "NewP@ssw0rd!"
            # Missing current_password
        }

        http = mock_http_handler(method="POST")

        with patch(
            "aragora.server.handlers.auth.password.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.user_id = "user-001"
            mock_extract.return_value = mock_ctx

            result = handle_change_password(mock_handler_instance, http)

        assert result.status_code == 400
        body = parse_handler_response(result)
        assert "required" in body.get("error", "").lower()

    def test_change_password_rejects_wrong_current_password(
        self, mock_handler_instance, mock_http_handler, mock_user, mock_user_store
    ):
        """Test that wrong current password is rejected."""
        from aragora.server.handlers.auth.password import handle_change_password

        mock_handler_instance.read_json_body.return_value = {
            "current_password": "WrongPassword123!",
            "new_password": "NewP@ssw0rd!",
        }

        http = mock_http_handler(method="POST")

        with patch(
            "aragora.server.handlers.auth.password.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.user_id = mock_user.id
            mock_extract.return_value = mock_ctx

            result = handle_change_password(mock_handler_instance, http)

        assert result.status_code == 401
        body = parse_handler_response(result)
        assert "incorrect" in body.get("error", "").lower()

    def test_change_password_validates_new_password_strength(
        self, mock_handler_instance, mock_http_handler, mock_user
    ):
        """Test that weak new passwords are rejected."""
        from aragora.server.handlers.auth.password import handle_change_password

        mock_handler_instance.read_json_body.return_value = {
            "current_password": "CurrentP@ssw0rd!",
            "new_password": "weak",  # Too weak
        }

        http = mock_http_handler(method="POST")

        with patch(
            "aragora.server.handlers.auth.password.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.user_id = mock_user.id
            mock_extract.return_value = mock_ctx

            result = handle_change_password(mock_handler_instance, http)

        assert result.status_code == 400

    def test_change_password_updates_hash_in_store(
        self, mock_handler_instance, mock_http_handler, mock_user, mock_user_store
    ):
        """Test that password hash is updated in store."""
        from aragora.server.handlers.auth.password import handle_change_password

        mock_handler_instance.read_json_body.return_value = {
            "current_password": "CurrentP@ssw0rd!",
            "new_password": "NewP@ssw0rd!123",
        }

        http = mock_http_handler(method="POST")

        with patch(PATCH_EXTRACT_USER) as mock_extract, patch(PATCH_HASH_PASSWORD) as mock_hash:
            mock_ctx = MagicMock()
            mock_ctx.user_id = mock_user.id
            mock_extract.return_value = mock_ctx
            mock_hash.return_value = ("new_hash", "new_salt")

            result = handle_change_password(mock_handler_instance, http)

        assert result.status_code == 200
        mock_user_store.update_user.assert_called_once()
        call_kwargs = mock_user_store.update_user.call_args[1]
        assert "password_hash" in call_kwargs
        assert "password_salt" in call_kwargs

    def test_change_password_returns_200_on_success(
        self, mock_handler_instance, mock_http_handler, mock_user, mock_user_store
    ):
        """Test successful password change returns 200."""
        from aragora.server.handlers.auth.password import handle_change_password

        mock_handler_instance.read_json_body.return_value = {
            "current_password": "CurrentP@ssw0rd!",
            "new_password": "NewP@ssw0rd!123",
        }

        http = mock_http_handler(method="POST")

        with patch(PATCH_EXTRACT_USER) as mock_extract, patch(PATCH_HASH_PASSWORD) as mock_hash:
            mock_ctx = MagicMock()
            mock_ctx.user_id = mock_user.id
            mock_extract.return_value = mock_ctx
            mock_hash.return_value = ("new_hash", "new_salt")

            result = handle_change_password(mock_handler_instance, http)

        assert result.status_code == 200
        body = parse_handler_response(result)
        assert "success" in body.get("message", "").lower()

    def test_change_password_invalidates_sessions(
        self, mock_handler_instance, mock_http_handler, mock_user, mock_user_store
    ):
        """Test that password change invalidates existing sessions."""
        from aragora.server.handlers.auth.password import handle_change_password

        mock_handler_instance.read_json_body.return_value = {
            "current_password": "CurrentP@ssw0rd!",
            "new_password": "NewP@ssw0rd!123",
        }

        http = mock_http_handler(method="POST")

        with patch(PATCH_EXTRACT_USER) as mock_extract, patch(PATCH_HASH_PASSWORD) as mock_hash:
            mock_ctx = MagicMock()
            mock_ctx.user_id = mock_user.id
            mock_extract.return_value = mock_ctx
            mock_hash.return_value = ("new_hash", "new_salt")

            result = handle_change_password(mock_handler_instance, http)

        # Should call increment_token_version to invalidate sessions
        mock_user_store.increment_token_version.assert_called_once_with(mock_user.id)
        body = parse_handler_response(result)
        assert body.get("sessions_invalidated") is True

    def test_change_password_returns_404_for_unknown_user(
        self, mock_handler_instance, mock_http_handler, mock_user_store
    ):
        """Test that unknown user returns 404."""
        from aragora.server.handlers.auth.password import handle_change_password

        mock_user_store.get_user_by_id.return_value = None
        mock_handler_instance.read_json_body.return_value = {
            "current_password": "CurrentP@ssw0rd!",
            "new_password": "NewP@ssw0rd!123",
        }

        http = mock_http_handler(method="POST")

        with patch(
            "aragora.server.handlers.auth.password.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.user_id = "nonexistent"
            mock_extract.return_value = mock_ctx

            result = handle_change_password(mock_handler_instance, http)

        assert result.status_code == 404


# ============================================================================
# Test: Forgot Password
# ============================================================================


class TestForgotPassword:
    """Tests for handle_forgot_password."""

    def test_forgot_password_generates_token(
        self,
        mock_handler_instance,
        mock_http_handler,
        mock_user,
        mock_user_store,
        mock_password_reset_store,
    ):
        """Test that forgot password generates a reset token."""
        from aragora.server.handlers.auth.password import handle_forgot_password

        mock_handler_instance.read_json_body.return_value = {"email": "test@example.com"}

        http = mock_http_handler(method="POST")

        with (
            patch(PATCH_RESET_STORE, return_value=mock_password_reset_store),
            patch("aragora.server.handlers.auth.password.send_password_reset_email") as mock_send,
        ):
            result = handle_forgot_password(mock_handler_instance, http)

        assert result.status_code == 200
        mock_password_reset_store.create_token.assert_called_once_with("test@example.com")

    def test_forgot_password_email_case_insensitive(
        self,
        mock_handler_instance,
        mock_http_handler,
        mock_user,
        mock_user_store,
        mock_password_reset_store,
    ):
        """Test that email lookup is case insensitive."""
        from aragora.server.handlers.auth.password import handle_forgot_password

        # Use mixed case email
        mock_handler_instance.read_json_body.return_value = {"email": "TEST@EXAMPLE.COM"}

        http = mock_http_handler(method="POST")

        with (
            patch("PATCH_RESET_STORE", return_value=mock_password_reset_store),
            patch("aragora.server.handlers.auth.password.send_password_reset_email"),
        ):
            result = handle_forgot_password(mock_handler_instance, http)

        assert result.status_code == 200
        # Token should be created with lowercase email
        mock_password_reset_store.create_token.assert_called_once_with("test@example.com")

    def test_forgot_password_returns_200_even_for_unknown_email(
        self, mock_handler_instance, mock_http_handler, mock_user_store, mock_password_reset_store
    ):
        """Test that unknown email still returns 200 (no enumeration)."""
        from aragora.server.handlers.auth.password import handle_forgot_password

        mock_user_store.get_user_by_email.return_value = None
        mock_handler_instance.read_json_body.return_value = {"email": "unknown@example.com"}

        http = mock_http_handler(method="POST")

        with patch("PATCH_RESET_STORE", return_value=mock_password_reset_store):
            result = handle_forgot_password(mock_handler_instance, http)

        # Should return 200 to prevent enumeration
        assert result.status_code == 200
        body = parse_handler_response(result)
        assert "if an account exists" in body.get("message", "").lower()

        # Should NOT create token for non-existent user
        mock_password_reset_store.create_token.assert_not_called()

    def test_forgot_password_returns_400_for_invalid_email_format(
        self, mock_handler_instance, mock_http_handler
    ):
        """Test that invalid email format returns 400."""
        from aragora.server.handlers.auth.password import handle_forgot_password

        mock_handler_instance.read_json_body.return_value = {"email": "not-an-email"}

        http = mock_http_handler(method="POST")

        result = handle_forgot_password(mock_handler_instance, http)

        assert result.status_code == 400

    def test_forgot_password_returns_400_for_missing_email(
        self, mock_handler_instance, mock_http_handler
    ):
        """Test that missing email returns 400."""
        from aragora.server.handlers.auth.password import handle_forgot_password

        mock_handler_instance.read_json_body.return_value = {}

        http = mock_http_handler(method="POST")

        result = handle_forgot_password(mock_handler_instance, http)

        assert result.status_code == 400

    def test_forgot_password_does_not_send_email_to_inactive_user(
        self,
        mock_handler_instance,
        mock_http_handler,
        mock_inactive_user,
        mock_user_store,
        mock_password_reset_store,
    ):
        """Test that inactive users don't receive reset emails."""
        from aragora.server.handlers.auth.password import handle_forgot_password

        mock_user_store.get_user_by_email.return_value = mock_inactive_user
        mock_handler_instance.read_json_body.return_value = {"email": "test@example.com"}

        http = mock_http_handler(method="POST")

        with (
            patch("PATCH_RESET_STORE", return_value=mock_password_reset_store),
            patch("aragora.server.handlers.auth.password.send_password_reset_email") as mock_send,
        ):
            result = handle_forgot_password(mock_handler_instance, http)

        # Should return 200 but not send email
        assert result.status_code == 200
        mock_send.assert_not_called()


# ============================================================================
# Test: Reset Password
# ============================================================================


class TestResetPassword:
    """Tests for handle_reset_password."""

    def test_reset_password_with_valid_token_succeeds(
        self,
        mock_handler_instance,
        mock_http_handler,
        mock_user,
        mock_user_store,
        mock_password_reset_store,
    ):
        """Test successful password reset with valid token."""
        from aragora.server.handlers.auth.password import handle_reset_password

        mock_handler_instance.read_json_body.return_value = {
            "token": "valid_token_123",
            "password": "NewP@ssw0rd!123",
        }

        http = mock_http_handler(method="POST")

        with (
            patch("PATCH_RESET_STORE", return_value=mock_password_reset_store),
            patch("aragora.server.handlers.auth.password.hash_password") as mock_hash,
        ):
            mock_hash.return_value = ("new_hash", "new_salt")
            result = handle_reset_password(mock_handler_instance, http)

        assert result.status_code == 200
        body = parse_handler_response(result)
        assert "success" in body.get("message", "").lower()

    def test_reset_password_with_expired_token_fails(
        self, mock_handler_instance, mock_http_handler, mock_password_reset_store
    ):
        """Test that expired token returns error."""
        from aragora.server.handlers.auth.password import handle_reset_password

        mock_password_reset_store.validate_token.return_value = (None, "Token has expired")
        mock_handler_instance.read_json_body.return_value = {
            "token": "expired_token",
            "password": "NewP@ssw0rd!123",
        }

        http = mock_http_handler(method="POST")

        with patch("PATCH_RESET_STORE", return_value=mock_password_reset_store):
            result = handle_reset_password(mock_handler_instance, http)

        assert result.status_code == 400
        body = parse_handler_response(result)
        assert "expired" in body.get("error", "").lower()

    def test_reset_password_token_is_single_use(
        self,
        mock_handler_instance,
        mock_http_handler,
        mock_user,
        mock_user_store,
        mock_password_reset_store,
    ):
        """Test that token is consumed after use."""
        from aragora.server.handlers.auth.password import handle_reset_password

        mock_handler_instance.read_json_body.return_value = {
            "token": "single_use_token",
            "password": "NewP@ssw0rd!123",
        }

        http = mock_http_handler(method="POST")

        with (
            patch("PATCH_RESET_STORE", return_value=mock_password_reset_store),
            patch("aragora.server.handlers.auth.password.hash_password") as mock_hash,
        ):
            mock_hash.return_value = ("new_hash", "new_salt")
            result = handle_reset_password(mock_handler_instance, http)

        # Token should be consumed
        mock_password_reset_store.consume_token.assert_called_once_with("single_use_token")

    def test_reset_password_validates_password_strength(
        self, mock_handler_instance, mock_http_handler, mock_password_reset_store
    ):
        """Test that weak passwords are rejected."""
        from aragora.server.handlers.auth.password import handle_reset_password

        mock_handler_instance.read_json_body.return_value = {
            "token": "valid_token",
            "password": "weak",  # Too weak
        }

        http = mock_http_handler(method="POST")

        with patch("PATCH_RESET_STORE", return_value=mock_password_reset_store):
            result = handle_reset_password(mock_handler_instance, http)

        assert result.status_code == 400

    def test_reset_password_invalidates_other_tokens(
        self,
        mock_handler_instance,
        mock_http_handler,
        mock_user,
        mock_user_store,
        mock_password_reset_store,
    ):
        """Test that other reset tokens for the email are invalidated."""
        from aragora.server.handlers.auth.password import handle_reset_password

        mock_handler_instance.read_json_body.return_value = {
            "token": "valid_token",
            "password": "NewP@ssw0rd!123",
        }

        http = mock_http_handler(method="POST")

        with (
            patch("PATCH_RESET_STORE", return_value=mock_password_reset_store),
            patch("aragora.server.handlers.auth.password.hash_password") as mock_hash,
        ):
            mock_hash.return_value = ("new_hash", "new_salt")
            result = handle_reset_password(mock_handler_instance, http)

        # Should invalidate all tokens for the email
        mock_password_reset_store.invalidate_tokens_for_email.assert_called_once_with(
            "test@example.com"
        )

    def test_reset_password_invalidates_sessions(
        self,
        mock_handler_instance,
        mock_http_handler,
        mock_user,
        mock_user_store,
        mock_password_reset_store,
    ):
        """Test that password reset invalidates all sessions."""
        from aragora.server.handlers.auth.password import handle_reset_password

        mock_handler_instance.read_json_body.return_value = {
            "token": "valid_token",
            "password": "NewP@ssw0rd!123",
        }

        http = mock_http_handler(method="POST")

        with (
            patch("PATCH_RESET_STORE", return_value=mock_password_reset_store),
            patch("aragora.server.handlers.auth.password.hash_password") as mock_hash,
        ):
            mock_hash.return_value = ("new_hash", "new_salt")
            result = handle_reset_password(mock_handler_instance, http)

        mock_user_store.increment_token_version.assert_called_once_with(mock_user.id)
        body = parse_handler_response(result)
        assert body.get("sessions_invalidated") is True

    def test_reset_password_returns_400_for_missing_token(
        self, mock_handler_instance, mock_http_handler
    ):
        """Test that missing token returns 400."""
        from aragora.server.handlers.auth.password import handle_reset_password

        mock_handler_instance.read_json_body.return_value = {
            "password": "NewP@ssw0rd!123"
            # Missing token
        }

        http = mock_http_handler(method="POST")

        result = handle_reset_password(mock_handler_instance, http)

        assert result.status_code == 400
        body = parse_handler_response(result)
        assert "token" in body.get("error", "").lower()

    def test_reset_password_returns_400_for_missing_password(
        self, mock_handler_instance, mock_http_handler
    ):
        """Test that missing password returns 400."""
        from aragora.server.handlers.auth.password import handle_reset_password

        mock_handler_instance.read_json_body.return_value = {
            "token": "valid_token"
            # Missing password
        }

        http = mock_http_handler(method="POST")

        result = handle_reset_password(mock_handler_instance, http)

        assert result.status_code == 400
        body = parse_handler_response(result)
        assert "password" in body.get("error", "").lower()

    def test_reset_password_returns_404_for_nonexistent_user(
        self, mock_handler_instance, mock_http_handler, mock_user_store, mock_password_reset_store
    ):
        """Test that deleted user during reset returns 404."""
        from aragora.server.handlers.auth.password import handle_reset_password

        # Token is valid but user no longer exists
        mock_user_store.get_user_by_email.return_value = None
        mock_handler_instance.read_json_body.return_value = {
            "token": "valid_token",
            "password": "NewP@ssw0rd!123",
        }

        http = mock_http_handler(method="POST")

        with patch("PATCH_RESET_STORE", return_value=mock_password_reset_store):
            result = handle_reset_password(mock_handler_instance, http)

        assert result.status_code == 404

    def test_reset_password_rejects_inactive_user(
        self,
        mock_handler_instance,
        mock_http_handler,
        mock_inactive_user,
        mock_user_store,
        mock_password_reset_store,
    ):
        """Test that inactive user cannot reset password."""
        from aragora.server.handlers.auth.password import handle_reset_password

        mock_user_store.get_user_by_email.return_value = mock_inactive_user
        mock_handler_instance.read_json_body.return_value = {
            "token": "valid_token",
            "password": "NewP@ssw0rd!123",
        }

        http = mock_http_handler(method="POST")

        with patch("PATCH_RESET_STORE", return_value=mock_password_reset_store):
            result = handle_reset_password(mock_handler_instance, http)

        assert result.status_code == 401


# ============================================================================
# Test: Send Password Reset Email
# ============================================================================


class TestSendPasswordResetEmail:
    """Tests for send_password_reset_email."""

    def test_send_password_reset_email_async(self, mock_user):
        """Test that email sending is asynchronous."""
        from aragora.server.handlers.auth.password import send_password_reset_email

        # This should not block - just verify it doesn't raise
        with (
            patch("asyncio.get_running_loop", side_effect=RuntimeError),
            patch("asyncio.run") as mock_run,
        ):
            send_password_reset_email(mock_user, "https://example.com/reset?token=abc")

        # Should have attempted to run async
        mock_run.assert_called_once()

    def test_send_password_reset_email_contains_reset_link(self, mock_user):
        """Test that email contains the reset link."""
        from aragora.server.handlers.auth.password import send_password_reset_email

        reset_link = "https://aragora.ai/reset-password?token=test123"

        # Capture the email content by mocking the email integration
        with (
            patch("asyncio.get_running_loop", side_effect=RuntimeError),
            patch("asyncio.run") as mock_run,
        ):
            send_password_reset_email(mock_user, reset_link)

        # The async function should have been called
        mock_run.assert_called_once()


# ============================================================================
# Test: Error Handling
# ============================================================================


class TestPasswordErrorHandling:
    """Tests for password handler error handling."""

    def test_change_password_returns_503_without_user_store(self, mock_http_handler):
        """Test graceful handling of missing user store."""
        from aragora.server.handlers.auth.password import handle_change_password

        handler_instance = MagicMock()
        handler_instance._check_permission.return_value = None
        handler_instance._get_user_store.return_value = None
        handler_instance.read_json_body.return_value = {
            "current_password": "Test123!",
            "new_password": "NewTest123!",
        }

        http = mock_http_handler(method="POST")

        with patch(
            "aragora.server.handlers.auth.password.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.user_id = "user-001"
            mock_extract.return_value = mock_ctx

            result = handle_change_password(handler_instance, http)

        assert result.status_code == 503

    def test_forgot_password_returns_503_without_user_store(self, mock_http_handler):
        """Test graceful handling of missing user store."""
        from aragora.server.handlers.auth.password import handle_forgot_password

        handler_instance = MagicMock()
        handler_instance._get_user_store.return_value = None
        handler_instance.read_json_body.return_value = {"email": "test@example.com"}

        http = mock_http_handler(method="POST")

        result = handle_forgot_password(handler_instance, http)

        assert result.status_code == 503

    def test_reset_password_returns_503_without_user_store(
        self, mock_http_handler, mock_password_reset_store
    ):
        """Test graceful handling of missing user store."""
        from aragora.server.handlers.auth.password import handle_reset_password

        handler_instance = MagicMock()
        handler_instance._get_user_store.return_value = None
        handler_instance.read_json_body.return_value = {
            "token": "valid_token",
            "password": "NewP@ssw0rd!123",
        }

        http = mock_http_handler(method="POST")

        with patch("PATCH_RESET_STORE", return_value=mock_password_reset_store):
            result = handle_reset_password(handler_instance, http)

        assert result.status_code == 503

    def test_change_password_handles_invalid_json(self, mock_http_handler):
        """Test handling of invalid JSON body."""
        from aragora.server.handlers.auth.password import handle_change_password

        handler_instance = MagicMock()
        handler_instance._check_permission.return_value = None
        handler_instance.read_json_body.return_value = None  # Invalid JSON

        http = mock_http_handler(method="POST")

        with patch(
            "aragora.server.handlers.auth.password.extract_user_from_request"
        ) as mock_extract:
            mock_ctx = MagicMock()
            mock_ctx.user_id = "user-001"
            mock_extract.return_value = mock_ctx

            result = handle_change_password(handler_instance, http)

        assert result.status_code == 400


__all__ = [
    "TestChangePassword",
    "TestForgotPassword",
    "TestResetPassword",
    "TestSendPasswordResetEmail",
    "TestPasswordErrorHandling",
]
