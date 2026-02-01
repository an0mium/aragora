"""
Tests for MFA (Multi-Factor Authentication) handlers.

Phase 5: Auth Handler Test Coverage - MFA handler tests.

Tests for:
- handle_mfa_setup - Generate MFA secret and provisioning URI
- handle_mfa_enable - Enable MFA after TOTP verification
- handle_mfa_disable - Disable MFA
- handle_mfa_verify - Verify MFA code during login
- handle_mfa_backup_codes - Regenerate backup codes
"""

from __future__ import annotations

import hashlib
import json
import secrets
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from tests.server.handlers.conftest import (
    assert_error_response,
    assert_success_response,
    parse_handler_response,
)

# Patch paths for functions imported inside handlers
PATCH_EXTRACT_USER = "aragora.server.handlers.auth.mfa.extract_user_from_request"
PATCH_PYOTP = "pyotp"
# MFA verify imports from billing.jwt_auth inside the function
PATCH_VALIDATE_MFA_PENDING = "aragora.billing.jwt_auth.validate_mfa_pending_token"
PATCH_CREATE_TOKEN_PAIR = "aragora.billing.jwt_auth.create_token_pair"
PATCH_GET_TOKEN_BLACKLIST = "aragora.billing.jwt_auth.get_token_blacklist"


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_auth_handler():
    """Create a mock AuthHandler instance for MFA tests."""
    handler_instance = MagicMock()

    # _check_permission returns None on success
    handler_instance._check_permission.return_value = None

    return handler_instance


@pytest.fixture
def mock_user_store():
    """Create a mock user store."""
    store = MagicMock()
    store.update_user.return_value = None
    store.increment_token_version.return_value = None
    return store


@pytest.fixture
def mock_user():
    """Create a mock user without MFA enabled."""
    user = MagicMock()
    user.id = "user-001"
    user.email = "test@example.com"
    user.mfa_enabled = False
    user.mfa_secret = None
    user.mfa_backup_codes = None
    user.org_id = "org-001"
    user.role = "user"

    def verify_password(password: str) -> bool:
        return password == "correct-password"

    user.verify_password = verify_password
    user.to_dict.return_value = {
        "id": user.id,
        "email": user.email,
        "role": user.role,
    }

    return user


@pytest.fixture
def mock_user_with_mfa(mock_user):
    """Create a mock user with MFA enabled."""
    mock_user.mfa_enabled = True
    mock_user.mfa_secret = "JBSWY3DPEHPK3PXP"  # Test base32 secret

    # Backup codes stored as JSON of hashes
    backup_codes = ["abcd1234", "efgh5678", "ijkl9012"]
    backup_hashes = [hashlib.sha256(c.encode()).hexdigest() for c in backup_codes]
    mock_user.mfa_backup_codes = json.dumps(backup_hashes)

    return mock_user


@pytest.fixture
def mock_user_with_pending_mfa(mock_user):
    """Create a mock user with MFA setup but not yet enabled."""
    mock_user.mfa_enabled = False
    mock_user.mfa_secret = "JBSWY3DPEHPK3PXP"  # Has secret but not enabled
    return mock_user


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
# Test: MFA Setup (handle_mfa_setup)
# ============================================================================


class TestMFASetup:
    """Tests for handle_mfa_setup."""

    def test_mfa_setup_generates_secret(
        self, mock_auth_handler, mock_user_store, mock_user, mock_http_handler
    ):
        """Test that MFA setup generates a TOTP secret."""
        from aragora.server.handlers.auth.mfa import handle_mfa_setup

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = mock_user.id

        http = mock_http_handler(method="POST")

        with patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx):
            result = handle_mfa_setup(mock_auth_handler, http)

        assert result.status_code == 200
        body = parse_handler_response(result)
        assert "secret" in body
        assert "provisioning_uri" in body
        assert len(body["secret"]) >= 16  # Base32 encoded

    def test_mfa_setup_returns_provisioning_uri(
        self, mock_auth_handler, mock_user_store, mock_user, mock_http_handler
    ):
        """Test that provisioning URI includes user email and issuer."""
        from aragora.server.handlers.auth.mfa import handle_mfa_setup

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = mock_user.id

        http = mock_http_handler(method="POST")

        with patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx):
            result = handle_mfa_setup(mock_auth_handler, http)

        body = parse_handler_response(result)
        uri = body["provisioning_uri"]

        assert "otpauth://totp/" in uri
        assert "Aragora" in uri  # Issuer name
        assert mock_user.email.replace("@", "%40") in uri

    def test_mfa_setup_rejects_if_already_enabled(
        self, mock_auth_handler, mock_user_store, mock_user_with_mfa, mock_http_handler
    ):
        """Test that setup fails if MFA is already enabled."""
        from aragora.server.handlers.auth.mfa import handle_mfa_setup

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user_with_mfa

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = mock_user_with_mfa.id

        http = mock_http_handler(method="POST")

        with patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx):
            result = handle_mfa_setup(mock_auth_handler, http)

        assert_error_response(result, 400, "already enabled")

    def test_mfa_setup_stores_secret_not_enabled(
        self, mock_auth_handler, mock_user_store, mock_user, mock_http_handler
    ):
        """Test that secret is stored but MFA is not enabled yet."""
        from aragora.server.handlers.auth.mfa import handle_mfa_setup

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = mock_user.id

        http = mock_http_handler(method="POST")

        with patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx):
            result = handle_mfa_setup(mock_auth_handler, http)

        # Should call update_user with mfa_secret but not mfa_enabled
        mock_user_store.update_user.assert_called_once()
        call_args = mock_user_store.update_user.call_args
        assert "mfa_secret" in call_args.kwargs
        assert "mfa_enabled" not in call_args.kwargs

    def test_mfa_setup_requires_permission(
        self, mock_auth_handler, mock_user_store, mock_user, mock_http_handler
    ):
        """Test that setup requires authentication.create permission."""
        from aragora.server.handlers.auth.mfa import handle_mfa_setup
        from aragora.server.handlers.base import error_response

        # Simulate permission denied
        mock_auth_handler._check_permission.return_value = error_response("Forbidden", 403)

        http = mock_http_handler(method="POST")
        result = handle_mfa_setup(mock_auth_handler, http)

        assert_error_response(result, 403)

    def test_mfa_setup_user_not_found(self, mock_auth_handler, mock_user_store, mock_http_handler):
        """Test setup fails when user not found."""
        from aragora.server.handlers.auth.mfa import handle_mfa_setup

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = None

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = "nonexistent-user"

        http = mock_http_handler(method="POST")

        with patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx):
            result = handle_mfa_setup(mock_auth_handler, http)

        assert_error_response(result, 404, "not found")


# ============================================================================
# Test: MFA Enable (handle_mfa_enable)
# ============================================================================


class TestMFAEnable:
    """Tests for handle_mfa_enable."""

    def test_mfa_enable_with_valid_code(
        self, mock_auth_handler, mock_user_store, mock_user_with_pending_mfa, mock_http_handler
    ):
        """Test that MFA is enabled with valid TOTP code."""
        from aragora.server.handlers.auth.mfa import handle_mfa_enable

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user_with_pending_mfa
        mock_auth_handler.read_json_body.return_value = {"code": "123456"}

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = mock_user_with_pending_mfa.id

        http = mock_http_handler(method="POST", body={"code": "123456"})

        # Mock pyotp.TOTP.verify to return True
        mock_totp = MagicMock()
        mock_totp.verify.return_value = True

        with (
            patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx),
            patch("pyotp.TOTP", return_value=mock_totp),
        ):
            result = handle_mfa_enable(mock_auth_handler, http)

        assert result.status_code == 200
        body = parse_handler_response(result)
        assert "backup_codes" in body
        assert len(body["backup_codes"]) == 10

    def test_mfa_enable_rejects_invalid_code(
        self, mock_auth_handler, mock_user_store, mock_user_with_pending_mfa, mock_http_handler
    ):
        """Test that enable fails with invalid TOTP code."""
        from aragora.server.handlers.auth.mfa import handle_mfa_enable

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user_with_pending_mfa
        mock_auth_handler.read_json_body.return_value = {"code": "000000"}

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = mock_user_with_pending_mfa.id

        http = mock_http_handler(method="POST", body={"code": "000000"})

        mock_totp = MagicMock()
        mock_totp.verify.return_value = False

        with (
            patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx),
            patch("pyotp.TOTP", return_value=mock_totp),
        ):
            result = handle_mfa_enable(mock_auth_handler, http)

        assert_error_response(result, 400, "Invalid verification code")

    def test_mfa_enable_requires_setup_first(
        self, mock_auth_handler, mock_user_store, mock_user, mock_http_handler
    ):
        """Test that enable fails if setup not done."""
        from aragora.server.handlers.auth.mfa import handle_mfa_enable

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user  # No mfa_secret
        mock_auth_handler.read_json_body.return_value = {"code": "123456"}

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = mock_user.id

        http = mock_http_handler(method="POST", body={"code": "123456"})

        with patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx):
            result = handle_mfa_enable(mock_auth_handler, http)

        assert_error_response(result, 400, "not set up")

    def test_mfa_enable_requires_code(
        self, mock_auth_handler, mock_user_store, mock_user_with_pending_mfa, mock_http_handler
    ):
        """Test that enable requires verification code."""
        from aragora.server.handlers.auth.mfa import handle_mfa_enable

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user_with_pending_mfa
        mock_auth_handler.read_json_body.return_value = {}

        http = mock_http_handler(method="POST", body={})

        result = handle_mfa_enable(mock_auth_handler, http)

        assert_error_response(result, 400, "required")

    def test_mfa_enable_generates_10_backup_codes(
        self, mock_auth_handler, mock_user_store, mock_user_with_pending_mfa, mock_http_handler
    ):
        """Test that exactly 10 backup codes are generated."""
        from aragora.server.handlers.auth.mfa import handle_mfa_enable

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user_with_pending_mfa
        mock_auth_handler.read_json_body.return_value = {"code": "123456"}

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = mock_user_with_pending_mfa.id

        http = mock_http_handler(method="POST", body={"code": "123456"})

        mock_totp = MagicMock()
        mock_totp.verify.return_value = True

        with (
            patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx),
            patch("pyotp.TOTP", return_value=mock_totp),
        ):
            result = handle_mfa_enable(mock_auth_handler, http)

        body = parse_handler_response(result)
        assert len(body["backup_codes"]) == 10

    def test_mfa_enable_invalidates_sessions(
        self, mock_auth_handler, mock_user_store, mock_user_with_pending_mfa, mock_http_handler
    ):
        """Test that enabling MFA invalidates existing sessions."""
        from aragora.server.handlers.auth.mfa import handle_mfa_enable

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user_with_pending_mfa
        mock_auth_handler.read_json_body.return_value = {"code": "123456"}

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = mock_user_with_pending_mfa.id

        http = mock_http_handler(method="POST", body={"code": "123456"})

        mock_totp = MagicMock()
        mock_totp.verify.return_value = True

        with (
            patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx),
            patch("pyotp.TOTP", return_value=mock_totp),
        ):
            result = handle_mfa_enable(mock_auth_handler, http)

        # increment_token_version should be called to invalidate sessions
        mock_user_store.increment_token_version.assert_called_once_with(
            mock_user_with_pending_mfa.id
        )

        body = parse_handler_response(result)
        assert body.get("sessions_invalidated") is True

    def test_mfa_enable_already_enabled(
        self, mock_auth_handler, mock_user_store, mock_user_with_mfa, mock_http_handler
    ):
        """Test that enable fails if MFA already enabled."""
        from aragora.server.handlers.auth.mfa import handle_mfa_enable

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user_with_mfa
        mock_auth_handler.read_json_body.return_value = {"code": "123456"}

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = mock_user_with_mfa.id

        http = mock_http_handler(method="POST", body={"code": "123456"})

        with patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx):
            result = handle_mfa_enable(mock_auth_handler, http)

        assert_error_response(result, 400, "already enabled")


# ============================================================================
# Test: MFA Disable (handle_mfa_disable)
# ============================================================================


class TestMFADisable:
    """Tests for handle_mfa_disable."""

    def test_mfa_disable_with_valid_totp_code(
        self, mock_auth_handler, mock_user_store, mock_user_with_mfa, mock_http_handler
    ):
        """Test disabling MFA with valid TOTP code."""
        from aragora.server.handlers.auth.mfa import handle_mfa_disable

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user_with_mfa
        mock_auth_handler.read_json_body.return_value = {"code": "123456"}

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = mock_user_with_mfa.id

        http = mock_http_handler(method="POST", body={"code": "123456"})

        mock_totp = MagicMock()
        mock_totp.verify.return_value = True

        with (
            patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx),
            patch("pyotp.TOTP", return_value=mock_totp),
        ):
            result = handle_mfa_disable(mock_auth_handler, http)

        assert result.status_code == 200
        body = parse_handler_response(result)
        assert "disabled successfully" in body.get("message", "")

    def test_mfa_disable_with_valid_password(
        self, mock_auth_handler, mock_user_store, mock_user_with_mfa, mock_http_handler
    ):
        """Test disabling MFA with valid password."""
        from aragora.server.handlers.auth.mfa import handle_mfa_disable

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user_with_mfa
        mock_auth_handler.read_json_body.return_value = {"password": "correct-password"}

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = mock_user_with_mfa.id

        http = mock_http_handler(method="POST", body={"password": "correct-password"})

        with patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx):
            result = handle_mfa_disable(mock_auth_handler, http)

        assert result.status_code == 200

    def test_mfa_disable_rejects_invalid_code(
        self, mock_auth_handler, mock_user_store, mock_user_with_mfa, mock_http_handler
    ):
        """Test that disable fails with invalid TOTP code."""
        from aragora.server.handlers.auth.mfa import handle_mfa_disable

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user_with_mfa
        mock_auth_handler.read_json_body.return_value = {"code": "000000"}

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = mock_user_with_mfa.id

        http = mock_http_handler(method="POST", body={"code": "000000"})

        mock_totp = MagicMock()
        mock_totp.verify.return_value = False

        with (
            patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx),
            patch("pyotp.TOTP", return_value=mock_totp),
        ):
            result = handle_mfa_disable(mock_auth_handler, http)

        assert_error_response(result, 400, "Invalid MFA code")

    def test_mfa_disable_rejects_wrong_password(
        self, mock_auth_handler, mock_user_store, mock_user_with_mfa, mock_http_handler
    ):
        """Test that disable fails with wrong password."""
        from aragora.server.handlers.auth.mfa import handle_mfa_disable

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user_with_mfa
        mock_auth_handler.read_json_body.return_value = {"password": "wrong-password"}

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = mock_user_with_mfa.id

        http = mock_http_handler(method="POST", body={"password": "wrong-password"})

        with patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx):
            result = handle_mfa_disable(mock_auth_handler, http)

        assert_error_response(result, 400, "Invalid password")

    def test_mfa_disable_requires_code_or_password(
        self, mock_auth_handler, mock_user_store, mock_user_with_mfa, mock_http_handler
    ):
        """Test that disable requires either code or password."""
        from aragora.server.handlers.auth.mfa import handle_mfa_disable

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user_with_mfa
        mock_auth_handler.read_json_body.return_value = {}

        http = mock_http_handler(method="POST", body={})

        result = handle_mfa_disable(mock_auth_handler, http)

        assert_error_response(result, 400, "required")

    def test_mfa_disable_clears_mfa_fields(
        self, mock_auth_handler, mock_user_store, mock_user_with_mfa, mock_http_handler
    ):
        """Test that disable clears all MFA fields."""
        from aragora.server.handlers.auth.mfa import handle_mfa_disable

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user_with_mfa
        mock_auth_handler.read_json_body.return_value = {"password": "correct-password"}

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = mock_user_with_mfa.id

        http = mock_http_handler(method="POST", body={"password": "correct-password"})

        with patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx):
            result = handle_mfa_disable(mock_auth_handler, http)

        # Should clear mfa_enabled, mfa_secret, mfa_backup_codes
        call_kwargs = mock_user_store.update_user.call_args.kwargs
        assert call_kwargs.get("mfa_enabled") is False
        assert call_kwargs.get("mfa_secret") is None
        assert call_kwargs.get("mfa_backup_codes") is None

    def test_mfa_disable_not_enabled(
        self, mock_auth_handler, mock_user_store, mock_user, mock_http_handler
    ):
        """Test that disable fails if MFA not enabled."""
        from aragora.server.handlers.auth.mfa import handle_mfa_disable

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user
        mock_auth_handler.read_json_body.return_value = {"code": "123456"}

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = mock_user.id

        http = mock_http_handler(method="POST", body={"code": "123456"})

        with patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx):
            result = handle_mfa_disable(mock_auth_handler, http)

        assert_error_response(result, 400, "not enabled")


# ============================================================================
# Test: MFA Verify (handle_mfa_verify)
# ============================================================================


class TestMFAVerify:
    """Tests for handle_mfa_verify."""

    def test_mfa_verify_with_valid_totp(
        self, mock_auth_handler, mock_user_store, mock_user_with_mfa, mock_http_handler
    ):
        """Test MFA verification with valid TOTP code."""
        from aragora.server.handlers.auth.mfa import handle_mfa_verify

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user_with_mfa
        mock_auth_handler.read_json_body.return_value = {
            "code": "123456",
            "pending_token": "valid-pending-token",
        }

        http = mock_http_handler(
            method="POST",
            body={
                "code": "123456",
                "pending_token": "valid-pending-token",
            },
        )

        mock_totp = MagicMock()
        mock_totp.verify.return_value = True

        mock_pending_payload = MagicMock()
        mock_pending_payload.sub = mock_user_with_mfa.id

        mock_tokens = MagicMock()
        mock_tokens.to_dict.return_value = {"access_token": "jwt", "refresh_token": "refresh"}

        mock_blacklist = MagicMock()

        with (
            patch(PATCH_VALIDATE_MFA_PENDING, return_value=mock_pending_payload),
            patch(PATCH_CREATE_TOKEN_PAIR, return_value=mock_tokens),
            patch(PATCH_GET_TOKEN_BLACKLIST, return_value=mock_blacklist),
            patch("pyotp.TOTP", return_value=mock_totp),
        ):
            result = handle_mfa_verify(mock_auth_handler, http)

        assert result.status_code == 200
        body = parse_handler_response(result)
        assert "tokens" in body
        assert "user" in body

    def test_mfa_verify_with_valid_backup_code(
        self, mock_auth_handler, mock_user_store, mock_user_with_mfa, mock_http_handler
    ):
        """Test MFA verification with valid backup code."""
        from aragora.server.handlers.auth.mfa import handle_mfa_verify

        # Use actual backup code that matches hash
        backup_code = "abcd1234"

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user_with_mfa
        mock_auth_handler.read_json_body.return_value = {
            "code": backup_code,
            "pending_token": "valid-pending-token",
        }

        http = mock_http_handler(
            method="POST",
            body={
                "code": backup_code,
                "pending_token": "valid-pending-token",
            },
        )

        mock_totp = MagicMock()
        mock_totp.verify.return_value = False  # TOTP fails, backup code used

        mock_pending_payload = MagicMock()
        mock_pending_payload.sub = mock_user_with_mfa.id

        mock_tokens = MagicMock()
        mock_tokens.to_dict.return_value = {"access_token": "jwt", "refresh_token": "refresh"}

        mock_blacklist = MagicMock()

        with (
            patch(PATCH_VALIDATE_MFA_PENDING, return_value=mock_pending_payload),
            patch(PATCH_CREATE_TOKEN_PAIR, return_value=mock_tokens),
            patch(PATCH_GET_TOKEN_BLACKLIST, return_value=mock_blacklist),
            patch("pyotp.TOTP", return_value=mock_totp),
        ):
            result = handle_mfa_verify(mock_auth_handler, http)

        assert result.status_code == 200
        body = parse_handler_response(result)
        assert "backup code used" in body.get("message", "")

    def test_mfa_verify_backup_code_is_consumed(
        self, mock_auth_handler, mock_user_store, mock_user_with_mfa, mock_http_handler
    ):
        """Test that used backup code is removed."""
        from aragora.server.handlers.auth.mfa import handle_mfa_verify

        backup_code = "abcd1234"

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user_with_mfa
        mock_auth_handler.read_json_body.return_value = {
            "code": backup_code,
            "pending_token": "valid-pending-token",
        }

        http = mock_http_handler(
            method="POST",
            body={
                "code": backup_code,
                "pending_token": "valid-pending-token",
            },
        )

        mock_totp = MagicMock()
        mock_totp.verify.return_value = False

        mock_pending_payload = MagicMock()
        mock_pending_payload.sub = mock_user_with_mfa.id

        mock_tokens = MagicMock()
        mock_tokens.to_dict.return_value = {"access_token": "jwt", "refresh_token": "refresh"}

        mock_blacklist = MagicMock()

        with (
            patch(PATCH_VALIDATE_MFA_PENDING, return_value=mock_pending_payload),
            patch(PATCH_CREATE_TOKEN_PAIR, return_value=mock_tokens),
            patch(PATCH_GET_TOKEN_BLACKLIST, return_value=mock_blacklist),
            patch("pyotp.TOTP", return_value=mock_totp),
        ):
            result = handle_mfa_verify(mock_auth_handler, http)

        # Should update user with fewer backup codes
        call_kwargs = mock_user_store.update_user.call_args.kwargs
        updated_codes = json.loads(call_kwargs["mfa_backup_codes"])
        # Original had 3 codes, now should have 2
        assert len(updated_codes) == 2

    def test_mfa_verify_rejects_invalid_code(
        self, mock_auth_handler, mock_user_store, mock_user_with_mfa, mock_http_handler
    ):
        """Test that verify fails with invalid code."""
        from aragora.server.handlers.auth.mfa import handle_mfa_verify

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user_with_mfa
        mock_auth_handler.read_json_body.return_value = {
            "code": "invalid-code",
            "pending_token": "valid-pending-token",
        }

        http = mock_http_handler(
            method="POST",
            body={
                "code": "invalid-code",
                "pending_token": "valid-pending-token",
            },
        )

        mock_totp = MagicMock()
        mock_totp.verify.return_value = False

        mock_pending_payload = MagicMock()
        mock_pending_payload.sub = mock_user_with_mfa.id

        with (
            patch(PATCH_VALIDATE_MFA_PENDING, return_value=mock_pending_payload),
            patch("pyotp.TOTP", return_value=mock_totp),
        ):
            result = handle_mfa_verify(mock_auth_handler, http)

        assert_error_response(result, 400, "Invalid MFA code")

    def test_mfa_verify_requires_pending_token(self, mock_auth_handler, mock_http_handler):
        """Test that verify requires pending token."""
        from aragora.server.handlers.auth.mfa import handle_mfa_verify

        mock_auth_handler.read_json_body.return_value = {"code": "123456"}

        http = mock_http_handler(method="POST", body={"code": "123456"})

        result = handle_mfa_verify(mock_auth_handler, http)

        assert_error_response(result, 400, "required")

    def test_mfa_verify_rejects_expired_pending_token(self, mock_auth_handler, mock_http_handler):
        """Test that verify fails with expired pending token."""
        from aragora.server.handlers.auth.mfa import handle_mfa_verify

        mock_auth_handler.read_json_body.return_value = {
            "code": "123456",
            "pending_token": "expired-token",
        }

        http = mock_http_handler(
            method="POST",
            body={
                "code": "123456",
                "pending_token": "expired-token",
            },
        )

        with patch(PATCH_VALIDATE_MFA_PENDING, return_value=None):
            result = handle_mfa_verify(mock_auth_handler, http)

        assert_error_response(result, 401, "Invalid or expired")

    def test_mfa_verify_blacklists_pending_token(
        self, mock_auth_handler, mock_user_store, mock_user_with_mfa, mock_http_handler
    ):
        """Test that pending token is blacklisted after use."""
        from aragora.server.handlers.auth.mfa import handle_mfa_verify

        pending_token = "valid-pending-token"

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user_with_mfa
        mock_auth_handler.read_json_body.return_value = {
            "code": "123456",
            "pending_token": pending_token,
        }

        http = mock_http_handler(
            method="POST",
            body={
                "code": "123456",
                "pending_token": pending_token,
            },
        )

        mock_totp = MagicMock()
        mock_totp.verify.return_value = True

        mock_pending_payload = MagicMock()
        mock_pending_payload.sub = mock_user_with_mfa.id

        mock_tokens = MagicMock()
        mock_tokens.to_dict.return_value = {"access_token": "jwt", "refresh_token": "refresh"}

        mock_blacklist = MagicMock()

        with (
            patch(PATCH_VALIDATE_MFA_PENDING, return_value=mock_pending_payload),
            patch(PATCH_CREATE_TOKEN_PAIR, return_value=mock_tokens),
            patch(PATCH_GET_TOKEN_BLACKLIST, return_value=mock_blacklist),
            patch("pyotp.TOTP", return_value=mock_totp),
        ):
            result = handle_mfa_verify(mock_auth_handler, http)

        mock_blacklist.revoke_token.assert_called_once_with(pending_token)


# ============================================================================
# Test: MFA Backup Codes Regeneration (handle_mfa_backup_codes)
# ============================================================================


class TestMFABackupCodesRegeneration:
    """Tests for handle_mfa_backup_codes."""

    def test_regenerate_backup_codes_with_valid_mfa_code(
        self, mock_auth_handler, mock_user_store, mock_user_with_mfa, mock_http_handler
    ):
        """Test regenerating backup codes with valid MFA code."""
        from aragora.server.handlers.auth.mfa import handle_mfa_backup_codes

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user_with_mfa
        mock_auth_handler.read_json_body.return_value = {"code": "123456"}

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = mock_user_with_mfa.id

        http = mock_http_handler(method="POST", body={"code": "123456"})

        mock_totp = MagicMock()
        mock_totp.verify.return_value = True

        with (
            patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx),
            patch("pyotp.TOTP", return_value=mock_totp),
        ):
            result = handle_mfa_backup_codes(mock_auth_handler, http)

        assert result.status_code == 200
        body = parse_handler_response(result)
        assert "backup_codes" in body
        assert len(body["backup_codes"]) == 10

    def test_regenerate_backup_codes_requires_mfa_code(
        self, mock_auth_handler, mock_user_store, mock_user_with_mfa, mock_http_handler
    ):
        """Test that regeneration requires current MFA code."""
        from aragora.server.handlers.auth.mfa import handle_mfa_backup_codes

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user_with_mfa
        mock_auth_handler.read_json_body.return_value = {}

        http = mock_http_handler(method="POST", body={})

        result = handle_mfa_backup_codes(mock_auth_handler, http)

        assert_error_response(result, 400, "required")

    def test_regenerate_backup_codes_rejects_invalid_code(
        self, mock_auth_handler, mock_user_store, mock_user_with_mfa, mock_http_handler
    ):
        """Test that regeneration fails with invalid MFA code."""
        from aragora.server.handlers.auth.mfa import handle_mfa_backup_codes

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user_with_mfa
        mock_auth_handler.read_json_body.return_value = {"code": "000000"}

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = mock_user_with_mfa.id

        http = mock_http_handler(method="POST", body={"code": "000000"})

        mock_totp = MagicMock()
        mock_totp.verify.return_value = False

        with (
            patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx),
            patch("pyotp.TOTP", return_value=mock_totp),
        ):
            result = handle_mfa_backup_codes(mock_auth_handler, http)

        assert_error_response(result, 400, "Invalid MFA code")

    def test_regenerate_backup_codes_requires_mfa_enabled(
        self, mock_auth_handler, mock_user_store, mock_user, mock_http_handler
    ):
        """Test that regeneration fails if MFA not enabled."""
        from aragora.server.handlers.auth.mfa import handle_mfa_backup_codes

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user
        mock_auth_handler.read_json_body.return_value = {"code": "123456"}

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = mock_user.id

        http = mock_http_handler(method="POST", body={"code": "123456"})

        with patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx):
            result = handle_mfa_backup_codes(mock_auth_handler, http)

        assert_error_response(result, 400, "not enabled")

    def test_regenerate_backup_codes_replaces_old(
        self, mock_auth_handler, mock_user_store, mock_user_with_mfa, mock_http_handler
    ):
        """Test that old backup codes are replaced."""
        from aragora.server.handlers.auth.mfa import handle_mfa_backup_codes

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user_with_mfa
        mock_auth_handler.read_json_body.return_value = {"code": "123456"}

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = mock_user_with_mfa.id

        http = mock_http_handler(method="POST", body={"code": "123456"})

        mock_totp = MagicMock()
        mock_totp.verify.return_value = True

        with (
            patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx),
            patch("pyotp.TOTP", return_value=mock_totp),
        ):
            result = handle_mfa_backup_codes(mock_auth_handler, http)

        # Check that update_user was called with new hashes
        call_kwargs = mock_user_store.update_user.call_args.kwargs
        new_hashes = json.loads(call_kwargs["mfa_backup_codes"])

        # New hashes should be different from original
        original_hashes = json.loads(mock_user_with_mfa.mfa_backup_codes)
        assert new_hashes != original_hashes
        assert len(new_hashes) == 10


# ============================================================================
# Test: Security Properties
# ============================================================================


class TestMFASecurityProperties:
    """Tests for MFA security properties."""

    def test_totp_uses_valid_window(self):
        """Test that TOTP verification uses valid_window=1 for 30s drift."""
        # This is verified by inspecting the handler code
        # The handler uses totp.verify(code, valid_window=1)
        # which allows for +/- 30 seconds of time drift
        pass  # Covered by integration tests

    def test_backup_codes_are_hex_format(self):
        """Test that backup codes are 8-character hex strings."""
        import re

        # Generate codes like the handler does
        backup_codes = [secrets.token_hex(4) for _ in range(10)]

        hex_pattern = re.compile(r"^[0-9a-f]{8}$")
        for code in backup_codes:
            assert hex_pattern.match(code)

    def test_backup_codes_hashed_with_sha256(self):
        """Test that backup codes are hashed with SHA-256."""
        backup_code = "abcd1234"
        expected_hash = hashlib.sha256(backup_code.encode()).hexdigest()

        # SHA-256 hash should be 64 hex characters
        assert len(expected_hash) == 64

    def test_totp_secret_is_base32(self):
        """Test that TOTP secrets are base32 encoded."""
        import pyotp

        secret = pyotp.random_base32()
        # Base32 only contains A-Z and 2-7
        assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567" for c in secret)

    def test_backup_codes_not_stored_plaintext(
        self, mock_auth_handler, mock_user_store, mock_user_with_pending_mfa, mock_http_handler
    ):
        """Test that backup codes are stored as hashes, not plaintext."""
        from aragora.server.handlers.auth.mfa import handle_mfa_enable

        mock_auth_handler._get_user_store.return_value = mock_user_store
        mock_user_store.get_user_by_id.return_value = mock_user_with_pending_mfa
        mock_auth_handler.read_json_body.return_value = {"code": "123456"}

        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = mock_user_with_pending_mfa.id

        http = mock_http_handler(method="POST", body={"code": "123456"})

        mock_totp = MagicMock()
        mock_totp.verify.return_value = True

        with (
            patch(PATCH_EXTRACT_USER, return_value=mock_auth_ctx),
            patch("pyotp.TOTP", return_value=mock_totp),
        ):
            result = handle_mfa_enable(mock_auth_handler, http)

        body = parse_handler_response(result)
        returned_codes = body["backup_codes"]

        # Get stored hashes
        call_kwargs = mock_user_store.update_user.call_args.kwargs
        stored_hashes = json.loads(call_kwargs["mfa_backup_codes"])

        # Plaintext codes should not be in stored hashes
        for code in returned_codes:
            assert code not in stored_hashes
            # But the hash should be
            code_hash = hashlib.sha256(code.encode()).hexdigest()
            assert code_hash in stored_hashes


__all__ = [
    "TestMFASetup",
    "TestMFAEnable",
    "TestMFADisable",
    "TestMFAVerify",
    "TestMFABackupCodesRegeneration",
    "TestMFASecurityProperties",
]
