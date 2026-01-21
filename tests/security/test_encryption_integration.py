"""
Integration tests for encryption service and field-level encryption.

Tests the full encryption lifecycle including:
- Key management and rotation
- Field-level encryption/decryption
- Store integration (IntegrationStore, WebhookConfigStore, etc.)
- Error handling and edge cases
"""

import os
import pytest
from unittest.mock import patch, MagicMock

# Set encryption key before imports
os.environ["ARAGORA_ENCRYPTION_KEY"] = "a" * 64  # 32-byte hex key


class TestEncryptionServiceLifecycle:
    """Test encryption service initialization and key management."""

    def test_service_initializes_with_env_key(self):
        """Service should initialize with ARAGORA_ENCRYPTION_KEY."""
        # Reset singleton
        import aragora.security.encryption as enc_module
        enc_module._encryption_service = None

        from aragora.security.encryption import get_encryption_service

        service = get_encryption_service()
        assert service is not None
        assert service.get_active_key_id() == "master"

    def test_service_generates_ephemeral_key_without_env(self):
        """Service should generate ephemeral key when no env key set."""
        import aragora.security.encryption as enc_module
        enc_module._encryption_service = None

        # Temporarily remove env key
        old_key = os.environ.pop("ARAGORA_ENCRYPTION_KEY", None)
        try:
            from aragora.security.encryption import get_encryption_service

            service = get_encryption_service()
            assert service is not None
            # Should have generated a default key
            assert service.get_active_key_id() == "default"
        finally:
            if old_key:
                os.environ["ARAGORA_ENCRYPTION_KEY"] = old_key
            enc_module._encryption_service = None

    def test_key_rotation(self):
        """Key rotation should create new version while keeping old for decryption."""
        import aragora.security.encryption as enc_module
        enc_module._encryption_service = None

        from aragora.security.encryption import EncryptionService

        service = EncryptionService()
        key1 = service.generate_key("test_key")
        assert key1.version == 1

        # Encrypt with key1
        encrypted = service.encrypt("secret data")
        assert encrypted.key_version == 1

        # Rotate key
        key2 = service.rotate_key("test_key")
        assert key2.version == 2

        # Should still decrypt old data
        decrypted = service.decrypt(encrypted)
        assert decrypted == b"secret data"

        # New encryptions should use key2
        encrypted2 = service.encrypt("new secret")
        assert encrypted2.key_version == 2

    def test_encryption_with_associated_data(self):
        """AAD should prevent ciphertext from being used with wrong context."""
        import aragora.security.encryption as enc_module
        enc_module._encryption_service = None

        from aragora.security.encryption import EncryptionService

        service = EncryptionService()
        service.generate_key("test")

        # Encrypt with AAD
        encrypted = service.encrypt("secret", associated_data="user_123")

        # Decrypt with correct AAD
        decrypted = service.decrypt(encrypted, associated_data="user_123")
        assert decrypted == b"secret"

        # Decrypt with wrong AAD should fail
        with pytest.raises(Exception):  # cryptography raises InvalidTag
            service.decrypt(encrypted, associated_data="user_456")


class TestFieldLevelEncryption:
    """Test field-level encryption for storage records."""

    def test_encrypt_fields_marks_encrypted(self):
        """Encrypted fields should be marked with _encrypted flag."""
        import aragora.security.encryption as enc_module
        enc_module._encryption_service = None

        from aragora.security.encryption import EncryptionService

        service = EncryptionService()
        service.generate_key("test")

        record = {
            "name": "My Integration",
            "api_key": "sk-secret-key-123",
            "enabled": True,
        }

        encrypted = service.encrypt_fields(record, ["api_key"])

        assert encrypted["name"] == "My Integration"
        assert encrypted["enabled"] is True
        assert isinstance(encrypted["api_key"], dict)
        assert encrypted["api_key"]["_encrypted"] is True
        assert "_value" in encrypted["api_key"]

    def test_decrypt_fields_restores_values(self):
        """Decrypted fields should restore original values."""
        import aragora.security.encryption as enc_module
        enc_module._encryption_service = None

        from aragora.security.encryption import EncryptionService

        service = EncryptionService()
        service.generate_key("test")

        original = {
            "name": "Test",
            "api_key": "sk-secret-123",
            "password": "hunter2",
        }

        encrypted = service.encrypt_fields(original.copy(), ["api_key", "password"])
        decrypted = service.decrypt_fields(encrypted, ["api_key", "password"])

        assert decrypted == original

    def test_encrypt_fields_with_aad(self):
        """Field encryption with AAD should bind to record ID."""
        import aragora.security.encryption as enc_module
        enc_module._encryption_service = None

        from aragora.security.encryption import EncryptionService

        service = EncryptionService()
        service.generate_key("test")

        record = {"api_key": "secret"}
        encrypted = service.encrypt_fields(record, ["api_key"], associated_data="rec_123")

        # Should decrypt with correct AAD
        decrypted = service.decrypt_fields(encrypted, ["api_key"], associated_data="rec_123")
        assert decrypted["api_key"] == "secret"

        # Should fail with wrong AAD
        with pytest.raises(Exception):
            service.decrypt_fields(encrypted, ["api_key"], associated_data="rec_456")


class TestEncryptedFieldsUtility:
    """Test the encrypted_fields utility module."""

    def test_sensitive_fields_constant(self):
        """SENSITIVE_FIELDS should contain expected field names."""
        from aragora.storage.encrypted_fields import SENSITIVE_FIELDS

        expected = {
            "access_token", "refresh_token", "api_key", "secret",
            "password", "auth_token", "bot_token", "webhook_url",
        }
        for field in expected:
            assert field in SENSITIVE_FIELDS

    def test_encrypt_sensitive_auto_detects_fields(self):
        """encrypt_sensitive should auto-detect sensitive fields."""
        # Reset encryption service
        import aragora.security.encryption as enc_module
        enc_module._encryption_service = None

        from aragora.storage.encrypted_fields import encrypt_sensitive

        data = {
            "name": "test",
            "api_key": "sk-123",
            "access_token": "token-abc",
            "normal_field": "visible",
        }

        result = encrypt_sensitive(data)

        # Non-sensitive fields unchanged
        assert result["name"] == "test"
        assert result["normal_field"] == "visible"

        # Sensitive fields encrypted
        assert isinstance(result["api_key"], dict)
        assert result["api_key"]["_encrypted"] is True
        assert isinstance(result["access_token"], dict)
        assert result["access_token"]["_encrypted"] is True

    def test_decrypt_sensitive_restores_values(self):
        """decrypt_sensitive should restore original values."""
        import aragora.security.encryption as enc_module
        enc_module._encryption_service = None

        from aragora.storage.encrypted_fields import encrypt_sensitive, decrypt_sensitive

        original = {
            "name": "test",
            "api_key": "sk-secret-key",
            "password": "hunter2",
        }

        encrypted = encrypt_sensitive(original.copy())
        decrypted = decrypt_sensitive(encrypted)

        assert decrypted == original

    def test_handles_none_values(self):
        """None values should not be encrypted."""
        import aragora.security.encryption as enc_module
        enc_module._encryption_service = None

        from aragora.storage.encrypted_fields import encrypt_sensitive

        data = {"api_key": None, "name": "test"}
        result = encrypt_sensitive(data)

        assert result["api_key"] is None
        assert result["name"] == "test"

    def test_handles_empty_data(self):
        """Empty data should return empty."""
        from aragora.storage.encrypted_fields import encrypt_sensitive, decrypt_sensitive

        assert encrypt_sensitive({}) == {}
        assert encrypt_sensitive(None) is None
        assert decrypt_sensitive({}) == {}
        assert decrypt_sensitive(None) is None

    def test_is_field_encrypted_helper(self):
        """is_field_encrypted should detect encrypted fields."""
        import aragora.security.encryption as enc_module
        enc_module._encryption_service = None

        from aragora.storage.encrypted_fields import (
            encrypt_sensitive,
            is_field_encrypted,
        )

        data = {"api_key": "secret", "name": "test"}
        encrypted = encrypt_sensitive(data)

        assert is_field_encrypted(encrypted, "api_key") is True
        assert is_field_encrypted(encrypted, "name") is False
        assert is_field_encrypted(encrypted, "nonexistent") is False

    def test_get_encrypted_field_names_helper(self):
        """get_encrypted_field_names should list encrypted fields."""
        import aragora.security.encryption as enc_module
        enc_module._encryption_service = None

        from aragora.storage.encrypted_fields import (
            encrypt_sensitive,
            get_encrypted_field_names,
        )

        data = {"api_key": "secret", "password": "pass", "name": "test"}
        encrypted = encrypt_sensitive(data)

        names = get_encrypted_field_names(encrypted)
        assert "api_key" in names
        assert "password" in names
        assert "name" not in names


class TestEncryptionEdgeCases:
    """Test edge cases and error handling."""

    def test_unicode_data(self):
        """Unicode data should encrypt/decrypt correctly."""
        import aragora.security.encryption as enc_module
        enc_module._encryption_service = None

        from aragora.storage.encrypted_fields import encrypt_sensitive, decrypt_sensitive

        data = {"api_key": "密钥123🔐"}
        encrypted = encrypt_sensitive(data)
        decrypted = decrypt_sensitive(encrypted)

        assert decrypted["api_key"] == "密钥123🔐"

    def test_special_characters(self):
        """Special characters should encrypt/decrypt correctly."""
        import aragora.security.encryption as enc_module
        enc_module._encryption_service = None

        from aragora.storage.encrypted_fields import encrypt_sensitive, decrypt_sensitive

        data = {"api_key": "!@#$%^&*()_+-=[]{}|;':\",./<>?"}
        encrypted = encrypt_sensitive(data)
        decrypted = decrypt_sensitive(encrypted)

        assert decrypted["api_key"] == "!@#$%^&*()_+-=[]{}|;':\",./<>?"

    def test_large_data(self):
        """Large data should encrypt/decrypt correctly."""
        import aragora.security.encryption as enc_module
        enc_module._encryption_service = None

        from aragora.storage.encrypted_fields import encrypt_sensitive, decrypt_sensitive

        # 1MB of data
        large_secret = "x" * (1024 * 1024)
        data = {"api_key": large_secret}

        encrypted = encrypt_sensitive(data)
        decrypted = decrypt_sensitive(encrypted)

        assert decrypted["api_key"] == large_secret

    def test_mixed_encrypted_unencrypted(self):
        """Mix of encrypted and unencrypted fields should work."""
        import aragora.security.encryption as enc_module
        enc_module._encryption_service = None

        from aragora.storage.encrypted_fields import encrypt_sensitive, decrypt_sensitive

        # Partially encrypted data (simulating legacy + new)
        data = {"api_key": "secret"}
        encrypted = encrypt_sensitive(data)
        encrypted["plaintext_field"] = "visible"
        encrypted["password"] = "unencrypted_legacy"

        decrypted = decrypt_sensitive(encrypted)

        assert decrypted["api_key"] == "secret"
        assert decrypted["plaintext_field"] == "visible"
        assert decrypted["password"] == "unencrypted_legacy"  # Not encrypted, returned as-is


class TestEncryptionAvailability:
    """Test encryption availability checks."""

    def test_is_encryption_available(self):
        """is_encryption_available should return True when cryptography installed."""
        from aragora.storage.encrypted_fields import is_encryption_available

        assert is_encryption_available() is True

    def test_is_encryption_configured(self):
        """is_encryption_configured should check for env key."""
        from aragora.storage.encrypted_fields import is_encryption_configured

        # We set the key at module load
        assert is_encryption_configured() is True

    def test_graceful_degradation_without_crypto(self):
        """Should degrade gracefully if cryptography unavailable."""
        from aragora.storage.encrypted_fields import encrypt_sensitive

        # Mock CRYPTO_AVAILABLE = False
        with patch("aragora.storage.encrypted_fields.is_encryption_available", return_value=False):
            data = {"api_key": "secret"}
            result = encrypt_sensitive(data)

            # Should return data unchanged
            assert result["api_key"] == "secret"
