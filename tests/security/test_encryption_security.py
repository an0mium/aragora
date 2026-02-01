"""
Security tests for encryption key requirements.

Tests that verify:
1. Production mode requires ARAGORA_ENCRYPTION_KEY to be set
2. Development mode allows ephemeral keys with warnings
3. validate_encryption_security_settings() works correctly

SECURITY: These tests are critical for ensuring data is not encrypted
with ephemeral keys in production, which would cause data loss on restart.
"""

from __future__ import annotations

import os
import logging
from unittest.mock import patch, MagicMock
import pytest

# Check if cryptography is available
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# Skip all tests if cryptography is not available
pytestmark = pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography package not installed")


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def reset_encryption_service():
    """Reset the global encryption service singleton before and after each test."""
    import aragora.security.encryption as enc_module

    # Save original state
    original_service = enc_module._encryption_service

    # Reset before test
    enc_module._encryption_service = None

    yield

    # Reset after test
    enc_module._encryption_service = original_service


@pytest.fixture
def valid_encryption_key():
    """Generate a valid 32-byte hex encryption key."""
    import secrets

    return secrets.token_hex(32)


# ===========================================================================
# _is_production_mode() Tests
# ===========================================================================


class TestIsProductionMode:
    """Tests for the _is_production_mode() helper function."""

    def test_production_env_detected(self):
        """ARAGORA_ENV=production is detected as production mode."""
        from aragora.security.encryption import _is_production_mode

        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}):
            assert _is_production_mode() is True

    def test_prod_shorthand_detected(self):
        """ARAGORA_ENV=prod is detected as production mode."""
        from aragora.security.encryption import _is_production_mode

        with patch.dict(os.environ, {"ARAGORA_ENV": "prod"}):
            assert _is_production_mode() is True

    def test_staging_env_detected(self):
        """ARAGORA_ENV=staging is detected as production mode."""
        from aragora.security.encryption import _is_production_mode

        with patch.dict(os.environ, {"ARAGORA_ENV": "staging"}):
            assert _is_production_mode() is True

    def test_stage_shorthand_detected(self):
        """ARAGORA_ENV=stage is detected as production mode."""
        from aragora.security.encryption import _is_production_mode

        with patch.dict(os.environ, {"ARAGORA_ENV": "stage"}):
            assert _is_production_mode() is True

    def test_development_not_production(self):
        """ARAGORA_ENV=development is not production mode."""
        from aragora.security.encryption import _is_production_mode

        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}):
            assert _is_production_mode() is False

    def test_empty_env_not_production(self):
        """Empty ARAGORA_ENV is not production mode."""
        from aragora.security.encryption import _is_production_mode

        with patch.dict(os.environ, {"ARAGORA_ENV": ""}):
            assert _is_production_mode() is False

    def test_missing_env_not_production(self):
        """Missing ARAGORA_ENV is not production mode."""
        from aragora.security.encryption import _is_production_mode

        env_backup = os.environ.get("ARAGORA_ENV")
        if "ARAGORA_ENV" in os.environ:
            del os.environ["ARAGORA_ENV"]

        try:
            assert _is_production_mode() is False
        finally:
            if env_backup is not None:
                os.environ["ARAGORA_ENV"] = env_backup

    def test_case_insensitive_detection(self):
        """Production mode detection is case insensitive."""
        from aragora.security.encryption import _is_production_mode

        with patch.dict(os.environ, {"ARAGORA_ENV": "PRODUCTION"}):
            assert _is_production_mode() is True

        with patch.dict(os.environ, {"ARAGORA_ENV": "Production"}):
            assert _is_production_mode() is True


# ===========================================================================
# Production Encryption Key Requirement Tests
# ===========================================================================


class TestProductionRequiresEncryptionKey:
    """Tests that production mode requires ARAGORA_ENCRYPTION_KEY."""

    def test_production_requires_encryption_key(self, reset_encryption_service):
        """SECURITY: ARAGORA_ENV=production without key should raise EncryptionError."""
        from aragora.security.encryption import get_encryption_service, EncryptionError

        with patch.dict(
            os.environ,
            {"ARAGORA_ENV": "production", "ARAGORA_ENCRYPTION_KEY": ""},
            clear=False,
        ):
            # Mock get_secret to return None (no key configured)
            with patch("aragora.config.secrets.get_secret", return_value=None) as mock_get_secret:
                with pytest.raises(EncryptionError) as exc_info:
                    get_encryption_service()

                # Verify error message contains helpful information
                error_message = str(exc_info.value)
                assert "ARAGORA_ENCRYPTION_KEY" in error_message
                assert "production" in error_message
                assert "secrets.token_hex(32)" in error_message

    def test_production_accepts_valid_key(self, reset_encryption_service, valid_encryption_key):
        """ARAGORA_ENV=production with valid key should work."""
        from aragora.security.encryption import get_encryption_service

        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}):
            with patch(
                "aragora.config.secrets.get_secret",
                return_value=valid_encryption_key,
            ):
                service = get_encryption_service()

                assert service is not None
                assert service.get_active_key_id() == "master"

                # Verify encryption/decryption works
                plaintext = "test secret data"
                encrypted = service.encrypt(plaintext)
                decrypted = service.decrypt_string(encrypted)
                assert decrypted == plaintext

    def test_staging_requires_encryption_key(self, reset_encryption_service):
        """SECURITY: ARAGORA_ENV=staging without key should raise EncryptionError."""
        from aragora.security.encryption import get_encryption_service, EncryptionError

        with patch.dict(os.environ, {"ARAGORA_ENV": "staging"}):
            with patch("aragora.config.secrets.get_secret", return_value=None):
                with pytest.raises(EncryptionError) as exc_info:
                    get_encryption_service()

                assert "staging" in str(exc_info.value)


# ===========================================================================
# Development Ephemeral Key Tests
# ===========================================================================


class TestDevelopmentAllowsEphemeralKey:
    """Tests that development mode allows ephemeral keys with warnings."""

    def test_development_allows_ephemeral_key(self, reset_encryption_service):
        """ARAGORA_ENV=development without key should work but warn."""
        from aragora.security.encryption import get_encryption_service

        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}):
            with patch("aragora.config.secrets.get_secret", return_value=None):
                service = get_encryption_service()

                assert service is not None
                assert service.get_active_key_id() == "default"

                # Verify encryption still works
                plaintext = "test data"
                encrypted = service.encrypt(plaintext)
                decrypted = service.decrypt_string(encrypted)
                assert decrypted == plaintext

    def test_ephemeral_key_warning_logged(self, reset_encryption_service, caplog):
        """Verify prominent warning is logged when using ephemeral key."""
        from aragora.security.encryption import get_encryption_service

        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}):
            with patch("aragora.config.secrets.get_secret", return_value=None):
                with caplog.at_level(logging.WARNING):
                    get_encryption_service()

                # Check warning message contains key information
                warning_messages = [
                    record.message for record in caplog.records if record.levelno >= logging.WARNING
                ]
                combined_warnings = " ".join(warning_messages)

                assert "SECURITY WARNING" in combined_warnings
                assert "ephemeral" in combined_warnings.lower()
                assert "LOST" in combined_warnings

    def test_no_env_set_allows_ephemeral_key(self, reset_encryption_service):
        """No ARAGORA_ENV set should allow ephemeral key (defaults to non-production)."""
        from aragora.security.encryption import get_encryption_service

        env_backup = os.environ.get("ARAGORA_ENV")
        if "ARAGORA_ENV" in os.environ:
            del os.environ["ARAGORA_ENV"]

        try:
            with patch("aragora.config.secrets.get_secret", return_value=None):
                service = get_encryption_service()
                assert service is not None
        finally:
            if env_backup is not None:
                os.environ["ARAGORA_ENV"] = env_backup


# ===========================================================================
# validate_encryption_security_settings() Tests
# ===========================================================================


class TestValidateEncryptionSecuritySettings:
    """Tests for the validate_encryption_security_settings() function."""

    def test_validation_function_checks_production(self):
        """validate_encryption_security_settings() raises in production without key."""
        from aragora.security.encryption import (
            validate_encryption_security_settings,
            EncryptionError,
        )

        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}):
            with patch("aragora.config.secrets.get_secret", return_value=None):
                with pytest.raises(EncryptionError) as exc_info:
                    validate_encryption_security_settings()

                assert "ARAGORA_ENCRYPTION_KEY" in str(exc_info.value)

    def test_validation_passes_in_production_with_key(self, valid_encryption_key):
        """validate_encryption_security_settings() passes in production with key."""
        from aragora.security.encryption import validate_encryption_security_settings

        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}):
            with patch(
                "aragora.config.secrets.get_secret",
                return_value=valid_encryption_key,
            ):
                result = validate_encryption_security_settings()

                assert result["is_production"] is True
                assert result["has_encryption_key"] is True
                assert result["is_valid"] is True
                assert len(result["errors"]) == 0

    def test_validation_warns_in_development_without_key(self):
        """validate_encryption_security_settings() warns in development without key."""
        from aragora.security.encryption import validate_encryption_security_settings

        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}):
            with patch("aragora.config.secrets.get_secret", return_value=None):
                result = validate_encryption_security_settings()

                assert result["is_production"] is False
                assert result["has_encryption_key"] is False
                assert result["is_valid"] is True  # Still valid in dev
                assert len(result["warnings"]) > 0
                assert "SECURITY WARNING" in result["warnings"][0]

    def test_validation_no_warnings_with_key_in_development(self, valid_encryption_key):
        """validate_encryption_security_settings() no warnings if key set in dev."""
        from aragora.security.encryption import validate_encryption_security_settings

        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}):
            with patch(
                "aragora.config.secrets.get_secret",
                return_value=valid_encryption_key,
            ):
                result = validate_encryption_security_settings()

                assert result["is_production"] is False
                assert result["has_encryption_key"] is True
                assert result["is_valid"] is True
                assert len(result["warnings"]) == 0
                assert len(result["errors"]) == 0


# ===========================================================================
# Error Message Quality Tests
# ===========================================================================


class TestErrorMessageQuality:
    """Tests that error messages are helpful and actionable."""

    def test_error_includes_key_generation_command(self, reset_encryption_service):
        """Error message includes command to generate key."""
        from aragora.security.encryption import get_encryption_service, EncryptionError

        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}):
            with patch("aragora.config.secrets.get_secret", return_value=None):
                with pytest.raises(EncryptionError) as exc_info:
                    get_encryption_service()

                error_message = str(exc_info.value)
                # Should include the Python command to generate a key
                assert "python -c" in error_message
                assert "secrets.token_hex(32)" in error_message

    def test_error_explains_data_loss_risk(self, reset_encryption_service):
        """Error message explains the data loss risk."""
        from aragora.security.encryption import get_encryption_service, EncryptionError

        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}):
            with patch("aragora.config.secrets.get_secret", return_value=None):
                with pytest.raises(EncryptionError) as exc_info:
                    get_encryption_service()

                error_message = str(exc_info.value)
                assert "unrecoverable" in error_message.lower()


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestEncryptionSecurityIntegration:
    """Integration tests for encryption security controls."""

    def test_full_encryption_workflow_production(
        self, reset_encryption_service, valid_encryption_key
    ):
        """Full encryption workflow works in production with valid key."""
        from aragora.security.encryption import (
            get_encryption_service,
            validate_encryption_security_settings,
        )

        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}):
            with patch(
                "aragora.config.secrets.get_secret",
                return_value=valid_encryption_key,
            ):
                # Validate settings first
                result = validate_encryption_security_settings()
                assert result["is_valid"] is True

                # Get service and encrypt data
                service = get_encryption_service()
                sensitive_data = "credit_card:4111-1111-1111-1111"

                encrypted = service.encrypt(sensitive_data)
                decrypted = service.decrypt_string(encrypted)

                assert decrypted == sensitive_data
                # Ciphertext should not contain plaintext
                assert b"4111" not in encrypted.ciphertext

    def test_security_defaults_are_safe(self, reset_encryption_service):
        """Default security settings fail closed in production."""
        from aragora.security.encryption import get_encryption_service, EncryptionError

        # Production with no key should fail
        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}):
            with patch("aragora.config.secrets.get_secret", return_value=None):
                with pytest.raises(EncryptionError):
                    get_encryption_service()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
