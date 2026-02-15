"""
Tests for OIDC security fix 1A: Remove OIDC Token Validation Fallback in production.

These tests verify that:
1. Startup validation rejects fallback settings in production
2. Fallback is only allowed in development with explicit opt-in
3. Invalid tokens raise errors in production (never fallback)
4. Security warnings and audit events are properly emitted

Security Requirements:
- ARAGORA_ALLOW_DEV_AUTH_FALLBACK must NOT be set when ARAGORA_ENV=production
- Default behavior is secure (production mode, no fallback)
- Fallback requires: ARAGORA_ENV=development AND ARAGORA_ALLOW_DEV_AUTH_FALLBACK=1
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.auth.oidc import (
    OIDCConfig,
    OIDCProvider,
    _allow_dev_auth_fallback,
    _is_production_mode,
    validate_oidc_security_settings,
)
from aragora.auth.sso import (
    SSOAuthenticationError,
    SSOConfigurationError,
    SSOProviderType,
)


def make_oidc_config(**kwargs) -> OIDCConfig:
    """Helper to create OIDCConfig with valid defaults."""
    defaults = {
        "provider_type": SSOProviderType.OIDC,
        "client_id": "test-client",
        "client_secret": "test-secret",
        "issuer_url": "https://login.example.com",
        "callback_url": "https://app.example.com/callback",
        "entity_id": "test-entity",
    }
    defaults.update(kwargs)
    return OIDCConfig(**defaults)


def create_mock_http_pool():
    """Create a mock HTTP pool for testing."""
    mock_pool = MagicMock()
    mock_client = AsyncMock()

    @asynccontextmanager
    async def mock_get_session(name):
        yield mock_client

    mock_pool.get_session = mock_get_session
    return mock_pool, mock_client


# ============================================================================
# Test: Startup rejects fallback in production
# ============================================================================


class TestStartupRejectsFallbackInProduction:
    """Test that startup validation rejects fallback settings in production."""

    def test_startup_rejects_fallback_in_production(self):
        """
        Test: When ARAGORA_ENV=production and ARAGORA_ALLOW_DEV_AUTH_FALLBACK is set,
        validate_oidc_security_settings() should raise SSOConfigurationError.
        """
        with patch.dict(
            os.environ,
            {
                "ARAGORA_ENV": "production",
                "ARAGORA_ALLOW_DEV_AUTH_FALLBACK": "1",
            },
            clear=True,
        ):
            with pytest.raises(SSOConfigurationError) as exc_info:
                validate_oidc_security_settings()

            error_msg = str(exc_info.value)
            assert "SECURITY VIOLATION" in error_msg
            assert "ARAGORA_ALLOW_DEV_AUTH_FALLBACK" in error_msg
            assert exc_info.value.details["error_code"] == "INSECURE_AUTH_FALLBACK_IN_PRODUCTION"

    def test_startup_rejects_fallback_in_production_any_value(self):
        """
        Test: Any value of ARAGORA_ALLOW_DEV_AUTH_FALLBACK in production should be rejected,
        not just '1' or 'true'.
        """
        test_values = ["1", "true", "yes", "false", "0", "no", "anything", ""]

        for value in test_values:
            with patch.dict(
                os.environ,
                {
                    "ARAGORA_ENV": "production",
                    "ARAGORA_ALLOW_DEV_AUTH_FALLBACK": value,
                },
                clear=True,
            ):
                with pytest.raises(SSOConfigurationError) as exc_info:
                    validate_oidc_security_settings()

                assert "SECURITY VIOLATION" in str(exc_info.value), f"Failed for value: {value!r}"

    def test_oidc_provider_init_rejects_fallback_in_production(self):
        """
        Test: OIDCProvider.__init__() should call validate_oidc_security_settings()
        and reject fallback settings in production.
        """
        config = make_oidc_config()

        with patch.dict(
            os.environ,
            {
                "ARAGORA_ENV": "production",
                "ARAGORA_ALLOW_DEV_AUTH_FALLBACK": "1",
            },
            clear=True,
        ):
            with pytest.raises(SSOConfigurationError) as exc_info:
                OIDCProvider(config)

            assert "SECURITY VIOLATION" in str(exc_info.value)


# ============================================================================
# Test: Startup allows fallback in development
# ============================================================================


class TestStartupAllowsFallbackInDevelopment:
    """Test that startup validation allows fallback in development mode."""

    def test_startup_allows_fallback_in_development(self):
        """
        Test: When ARAGORA_ENV=development and ARAGORA_ALLOW_DEV_AUTH_FALLBACK=1,
        validate_oidc_security_settings() should succeed (not raise).
        """
        with patch.dict(
            os.environ,
            {
                "ARAGORA_ENV": "development",
                "ARAGORA_ALLOW_DEV_AUTH_FALLBACK": "1",
            },
            clear=True,
        ):
            # Should not raise
            validate_oidc_security_settings()

    def test_startup_allows_no_fallback_in_development(self):
        """
        Test: When ARAGORA_ENV=development without fallback flag,
        validate_oidc_security_settings() should succeed.
        """
        with patch.dict(
            os.environ,
            {
                "ARAGORA_ENV": "development",
            },
            clear=True,
        ):
            # Should not raise
            validate_oidc_security_settings()

    def test_startup_allows_no_fallback_in_production(self):
        """
        Test: When ARAGORA_ENV=production without fallback flag,
        validate_oidc_security_settings() should succeed.
        """
        with patch.dict(
            os.environ,
            {
                "ARAGORA_ENV": "production",
            },
            clear=True,
        ):
            # Should not raise - no ARAGORA_ALLOW_DEV_AUTH_FALLBACK set
            validate_oidc_security_settings()

    def test_oidc_provider_init_succeeds_in_development_with_fallback(self):
        """
        Test: OIDCProvider.__init__() should succeed in development with fallback enabled.
        """
        config = make_oidc_config()

        with patch.dict(
            os.environ,
            {
                "ARAGORA_ENV": "development",
                "ARAGORA_ALLOW_DEV_AUTH_FALLBACK": "1",
            },
            clear=True,
        ):
            # Should not raise
            provider = OIDCProvider(config)
            assert provider is not None


# ============================================================================
# Test: Fallback blocked in production even if env not set
# ============================================================================


class TestFallbackBlockedInProductionDefault:
    """Test that default behavior is secure (production mode, no fallback)."""

    def test_fallback_blocked_in_production_even_if_env_not_set(self):
        """
        Test: When ARAGORA_ENV is not set (defaults to production),
        the fallback should be blocked.
        """
        # Clear all relevant env vars
        with patch.dict(os.environ, {}, clear=True):
            # Default should be production mode
            assert _is_production_mode() is True
            # Fallback should not be allowed
            assert _allow_dev_auth_fallback() is False

    def test_is_production_mode_defaults_to_true(self):
        """
        Test: _is_production_mode() should return True when ARAGORA_ENV is not set.
        This ensures secure-by-default behavior.
        """
        with patch.dict(os.environ, {}, clear=True):
            assert _is_production_mode() is True

    def test_allow_dev_auth_fallback_requires_dev_mode(self):
        """
        Test: _allow_dev_auth_fallback() should return False in production mode,
        even if ARAGORA_ALLOW_DEV_AUTH_FALLBACK is somehow set.
        """
        with patch.dict(
            os.environ,
            {
                "ARAGORA_ENV": "production",
                "ARAGORA_ALLOW_DEV_AUTH_FALLBACK": "1",
            },
            clear=True,
        ):
            # Should return False because we're in production
            assert _allow_dev_auth_fallback() is False


# ============================================================================
# Test: Invalid token raises error in production
# ============================================================================


class TestInvalidTokenRaisesErrorInProduction:
    """Test that invalid tokens raise SSOAuthenticationError in production."""

    @pytest.fixture
    def provider(self):
        """Create provider for testing."""
        config = make_oidc_config(
            jwks_uri="https://login.example.com/.well-known/jwks.json",
            userinfo_endpoint="https://login.example.com/userinfo",
        )
        # Create provider in development mode to avoid startup validation error
        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}, clear=True):
            return OIDCProvider(config)

    @pytest.mark.asyncio
    async def test_invalid_token_raises_error_in_production(self, provider):
        """
        Test: When ID token validation fails in production mode,
        SSOAuthenticationError should be raised (not fallback to userinfo).
        """
        import jwt.exceptions

        # Mock tokens with invalid ID token
        tokens = {
            "access_token": "valid-access-token",
            "id_token": "invalid.id.token",
            "expires_in": 3600,
        }

        # Mock JWKS client to raise validation error
        mock_signing_key = MagicMock()
        mock_signing_key.key = "test-key"

        mock_jwks_client = MagicMock()
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key

        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}, clear=True):
            with patch.object(provider, "_jwks_client", mock_jwks_client):
                with patch(
                    "jwt.decode",
                    side_effect=jwt.exceptions.InvalidSignatureError(
                        "Signature verification failed"
                    ),
                ):
                    with pytest.raises(SSOAuthenticationError) as exc_info:
                        await provider._get_user_info(tokens)

                    error_msg = str(exc_info.value)
                    assert "validation failed" in error_msg.lower()
                    # Should NOT mention fallback instructions in production
                    assert "ARAGORA_ALLOW_DEV_AUTH_FALLBACK" not in error_msg

    @pytest.mark.asyncio
    async def test_expired_token_raises_error_in_production(self, provider):
        """
        Test: When ID token is expired in production mode,
        SSOAuthenticationError should be raised.
        """
        import jwt.exceptions

        tokens = {
            "access_token": "valid-access-token",
            "id_token": "expired.id.token",
            "expires_in": 3600,
        }

        mock_signing_key = MagicMock()
        mock_signing_key.key = "test-key"

        mock_jwks_client = MagicMock()
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key

        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}, clear=True):
            with patch.object(provider, "_jwks_client", mock_jwks_client):
                with patch(
                    "jwt.decode",
                    side_effect=jwt.exceptions.ExpiredSignatureError("Token expired"),
                ):
                    with pytest.raises(SSOAuthenticationError) as exc_info:
                        await provider._get_user_info(tokens)

                    assert "validation failed" in str(exc_info.value).lower()


# ============================================================================
# Test: Fallback warning logged in development
# ============================================================================


class TestFallbackWarningLoggedInDevelopment:
    """Test that security warnings are logged when fallback is used in development."""

    def test_fallback_warning_logged_in_development(self, caplog):
        """
        Test: When ARAGORA_ENV=development and ARAGORA_ALLOW_DEV_AUTH_FALLBACK=1,
        validate_oidc_security_settings() should log a warning.
        """
        with patch.dict(
            os.environ,
            {
                "ARAGORA_ENV": "development",
                "ARAGORA_ALLOW_DEV_AUTH_FALLBACK": "1",
            },
            clear=True,
        ):
            with caplog.at_level(logging.WARNING):
                validate_oidc_security_settings()

            # Check that warning was logged
            warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
            assert any("SECURITY WARNING" in msg for msg in warning_messages)
            assert any("ARAGORA_ALLOW_DEV_AUTH_FALLBACK" in msg for msg in warning_messages)

    @pytest.mark.asyncio
    async def test_fallback_warning_logged_on_token_validation_failure(self, caplog):
        """
        Test: When ID token validation fails in development with fallback enabled,
        a warning should be logged.
        """
        import jwt.exceptions

        config = make_oidc_config(
            jwks_uri="https://login.example.com/.well-known/jwks.json",
            userinfo_endpoint="https://login.example.com/userinfo",
        )

        with patch.dict(
            os.environ,
            {
                "ARAGORA_ENV": "development",
                "ARAGORA_ALLOW_DEV_AUTH_FALLBACK": "1",
            },
            clear=True,
        ):
            provider = OIDCProvider(config)

            tokens = {
                "access_token": "valid-access-token",
                "id_token": "invalid.id.token",
                "expires_in": 3600,
            }

            mock_signing_key = MagicMock()
            mock_signing_key.key = "test-key"

            mock_jwks_client = MagicMock()
            mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key

            # Mock userinfo endpoint to return user data
            mock_pool, mock_client = create_mock_http_pool()
            mock_userinfo_response = MagicMock()
            mock_userinfo_response.json.return_value = {
                "sub": "user-123",
                "email": "test@example.com",
            }
            mock_userinfo_response.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_userinfo_response)

            # The production code calls jwt.decode twice in the fallback path:
            # 1st call: with signature verification → raises InvalidSignatureError
            # 2nd call: without signature verification → should succeed with claims
            call_count = 0
            original_error = jwt.exceptions.InvalidSignatureError("Signature verification failed")

            def jwt_decode_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise original_error
                # Fallback decode (verify_signature=False): return valid claims
                return {"sub": "user-123", "email": "test@example.com"}

            with patch.object(provider, "_jwks_client", mock_jwks_client):
                with patch("jwt.decode", side_effect=jwt_decode_side_effect):
                    with patch(
                        "aragora.server.http_client_pool.get_http_pool", return_value=mock_pool
                    ):
                        with caplog.at_level(logging.WARNING):
                            user = await provider._get_user_info(tokens)

                            # Should have fallen back to userinfo
                            assert user.email == "test@example.com"

                            # Check warning was logged
                            warning_messages = [
                                r.message for r in caplog.records if r.levelno >= logging.WARNING
                            ]
                            assert any("INSECURE" in msg for msg in warning_messages)
                            assert any(
                                "do not use in production" in msg.lower()
                                for msg in warning_messages
                            )


# ============================================================================
# Test: Audit event emitted on fallback
# ============================================================================


class TestAuditEventEmittedOnFallback:
    """Test that security audit events are emitted when fallback is used."""

    @pytest.mark.asyncio
    async def test_audit_event_emitted_on_fallback(self):
        """
        Test: When fallback is used in development, a security audit event
        should be emitted via audit_security_event().
        """
        import jwt.exceptions

        config = make_oidc_config(
            jwks_uri="https://login.example.com/.well-known/jwks.json",
            userinfo_endpoint="https://login.example.com/userinfo",
        )

        with patch.dict(
            os.environ,
            {
                "ARAGORA_ENV": "development",
                "ARAGORA_ALLOW_DEV_AUTH_FALLBACK": "1",
            },
            clear=True,
        ):
            provider = OIDCProvider(config)

            tokens = {
                "access_token": "valid-access-token",
                "id_token": "invalid.id.token",
                "expires_in": 3600,
            }

            mock_signing_key = MagicMock()
            mock_signing_key.key = "test-key"

            mock_jwks_client = MagicMock()
            mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key

            # Mock userinfo endpoint
            mock_pool, mock_client = create_mock_http_pool()
            mock_userinfo_response = MagicMock()
            mock_userinfo_response.json.return_value = {
                "sub": "user-123",
                "email": "test@example.com",
            }
            mock_userinfo_response.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_userinfo_response)

            # Mock the audit function
            mock_audit = MagicMock()

            # jwt.decode is called twice: 1st with verification (fails), 2nd without (succeeds)
            call_count = 0

            def jwt_decode_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise jwt.exceptions.InvalidSignatureError("Signature verification failed")
                return {"sub": "user-123", "email": "test@example.com"}

            with patch.object(provider, "_jwks_client", mock_jwks_client):
                with patch("jwt.decode", side_effect=jwt_decode_side_effect):
                    with patch(
                        "aragora.server.http_client_pool.get_http_pool", return_value=mock_pool
                    ):
                        with patch(
                            "aragora.server.middleware.audit_logger.audit_security_event",
                            mock_audit,
                        ):
                            await provider._get_user_info(tokens)

                            # Verify audit event was called
                            mock_audit.assert_called_once()
                            call_kwargs = mock_audit.call_args[1]

                            assert call_kwargs["event_type"] == "oidc_token_validation_fallback"
                            assert call_kwargs["actor"] == "system"
                            assert "error" in call_kwargs["details"]
                            assert "InvalidSignatureError" in call_kwargs["details"]["error_type"]
                            assert call_kwargs["details"]["fallback"] == "userinfo_endpoint"

    @pytest.mark.asyncio
    async def test_audit_failure_does_not_break_fallback(self):
        """
        Test: If audit_security_event raises ImportError (audit system not available),
        the fallback should still work.
        """
        import jwt.exceptions

        config = make_oidc_config(
            jwks_uri="https://login.example.com/.well-known/jwks.json",
            userinfo_endpoint="https://login.example.com/userinfo",
        )

        with patch.dict(
            os.environ,
            {
                "ARAGORA_ENV": "development",
                "ARAGORA_ALLOW_DEV_AUTH_FALLBACK": "1",
            },
            clear=True,
        ):
            provider = OIDCProvider(config)

            tokens = {
                "access_token": "valid-access-token",
                "id_token": "invalid.id.token",
                "expires_in": 3600,
            }

            mock_signing_key = MagicMock()
            mock_signing_key.key = "test-key"

            mock_jwks_client = MagicMock()
            mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key

            # Mock userinfo endpoint
            mock_pool, mock_client = create_mock_http_pool()
            mock_userinfo_response = MagicMock()
            mock_userinfo_response.json.return_value = {
                "sub": "user-123",
                "email": "test@example.com",
            }
            mock_userinfo_response.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_userinfo_response)

            # jwt.decode is called twice: 1st with verification (fails), 2nd without (succeeds)
            call_count = 0

            def jwt_decode_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise jwt.exceptions.InvalidSignatureError("Signature verification failed")
                return {"sub": "user-123", "email": "test@example.com"}

            with patch.object(provider, "_jwks_client", mock_jwks_client):
                with patch("jwt.decode", side_effect=jwt_decode_side_effect):
                    with patch(
                        "aragora.server.http_client_pool.get_http_pool", return_value=mock_pool
                    ):
                        # Make audit import fail
                        with patch.dict(
                            "sys.modules", {"aragora.server.middleware.audit_logger": None}
                        ):
                            # Should not raise, should fall back successfully
                            user = await provider._get_user_info(tokens)
                            assert user.email == "test@example.com"


# ============================================================================
# Additional Security Edge Cases
# ============================================================================


class TestSecurityEdgeCases:
    """Additional security edge case tests."""

    def test_production_mode_detection_case_insensitive(self):
        """Test that production mode detection is case-insensitive."""
        production_values = ["production", "PRODUCTION", "Production", "PROD"]
        for value in production_values:
            with patch.dict(os.environ, {"ARAGORA_ENV": value}, clear=True):
                # Only "production" (lowercase) should be considered production
                # Others default to production behavior
                result = _is_production_mode()
                if value.lower() == "production":
                    assert result is True, f"Failed for value: {value}"
                else:
                    # Non-matching values are treated as production (secure default)
                    assert result is True, f"Failed for value: {value}"

    def test_development_mode_variants(self):
        """Test that development mode accepts various dev-like values."""
        dev_values = ["development", "dev", "local", "test"]
        for value in dev_values:
            with patch.dict(os.environ, {"ARAGORA_ENV": value}, clear=True):
                assert _is_production_mode() is False, f"Failed for value: {value}"

    def test_fallback_requires_both_conditions(self):
        """Test that fallback requires BOTH dev mode AND explicit flag."""
        # Dev mode without flag
        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}, clear=True):
            assert _allow_dev_auth_fallback() is False

        # Flag without dev mode
        with patch.dict(
            os.environ,
            {
                "ARAGORA_ENV": "production",
                "ARAGORA_ALLOW_DEV_AUTH_FALLBACK": "1",
            },
            clear=True,
        ):
            assert _allow_dev_auth_fallback() is False

        # Both conditions met
        with patch.dict(
            os.environ,
            {
                "ARAGORA_ENV": "development",
                "ARAGORA_ALLOW_DEV_AUTH_FALLBACK": "1",
            },
            clear=True,
        ):
            assert _allow_dev_auth_fallback() is True

    def test_fallback_flag_values(self):
        """Test which values enable the fallback flag."""
        enabled_values = ["1", "true", "yes"]
        disabled_values = ["0", "false", "no", "", "anything_else"]

        for value in enabled_values:
            with patch.dict(
                os.environ,
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_DEV_AUTH_FALLBACK": value,
                },
                clear=True,
            ):
                assert _allow_dev_auth_fallback() is True, f"Failed for enabled value: {value}"

        for value in disabled_values:
            with patch.dict(
                os.environ,
                {
                    "ARAGORA_ENV": "development",
                    "ARAGORA_ALLOW_DEV_AUTH_FALLBACK": value,
                },
                clear=True,
            ):
                assert _allow_dev_auth_fallback() is False, f"Failed for disabled value: {value}"

    def test_critical_log_on_production_violation(self, caplog):
        """Test that a CRITICAL log is emitted when production is violated."""
        with patch.dict(
            os.environ,
            {
                "ARAGORA_ENV": "production",
                "ARAGORA_ALLOW_DEV_AUTH_FALLBACK": "1",
            },
            clear=True,
        ):
            with caplog.at_level(logging.CRITICAL):
                with pytest.raises(SSOConfigurationError):
                    validate_oidc_security_settings()

                critical_messages = [
                    r.message for r in caplog.records if r.levelno >= logging.CRITICAL
                ]
                assert any("SECURITY VIOLATION" in msg for msg in critical_messages)
