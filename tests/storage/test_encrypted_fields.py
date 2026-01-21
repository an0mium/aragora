"""
Tests for encrypted_fields utility module.

Tests encryption/decryption of sensitive storage fields.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

# Set encryption key before importing modules
os.environ["ARAGORA_ENCRYPTION_KEY"] = "a" * 64  # 32-byte hex key


class TestEncryptSensitive:
    """Test encrypt_sensitive function."""

    def test_encrypts_sensitive_fields(self):
        """Sensitive fields are encrypted."""
        from aragora.storage.encrypted_fields import encrypt_sensitive, SENSITIVE_FIELDS

        data = {
            "name": "test_integration",
            "api_key": "sk-test-12345",
            "access_token": "token123",
            "other_field": "not_sensitive",
        }

        result = encrypt_sensitive(data)

        # Non-sensitive fields unchanged
        assert result["name"] == "test_integration"
        assert result["other_field"] == "not_sensitive"

        # Sensitive fields encrypted
        assert isinstance(result["api_key"], dict)
        assert result["api_key"]["_encrypted"] is True
        assert "_value" in result["api_key"]

        assert isinstance(result["access_token"], dict)
        assert result["access_token"]["_encrypted"] is True

    def test_handles_empty_data(self):
        """Empty dict returns empty dict."""
        from aragora.storage.encrypted_fields import encrypt_sensitive

        assert encrypt_sensitive({}) == {}
        assert encrypt_sensitive(None) is None

    def test_handles_no_sensitive_fields(self):
        """Data with no sensitive fields is unchanged."""
        from aragora.storage.encrypted_fields import encrypt_sensitive

        data = {"name": "test", "value": 123}
        result = encrypt_sensitive(data)
        assert result == data

    def test_handles_none_values(self):
        """None values in sensitive fields are not encrypted."""
        from aragora.storage.encrypted_fields import encrypt_sensitive

        data = {"api_key": None, "name": "test"}
        result = encrypt_sensitive(data)
        assert result["api_key"] is None
        assert result["name"] == "test"

    def test_additional_fields_parameter(self):
        """Additional fields can be specified for encryption."""
        from aragora.storage.encrypted_fields import encrypt_sensitive

        data = {"custom_secret": "my_secret", "normal": "value"}
        result = encrypt_sensitive(data, additional_fields={"custom_secret"})

        assert isinstance(result["custom_secret"], dict)
        assert result["custom_secret"]["_encrypted"] is True
        assert result["normal"] == "value"

    def test_record_id_for_aad(self):
        """Record ID is used as associated authenticated data."""
        from aragora.storage.encrypted_fields import encrypt_sensitive, decrypt_sensitive

        data = {"api_key": "secret123"}

        # Encrypt with record ID
        encrypted = encrypt_sensitive(data, record_id="user_123")

        # Decrypt with same record ID
        decrypted = decrypt_sensitive(encrypted, record_id="user_123")
        assert decrypted["api_key"] == "secret123"


class TestDecryptSensitive:
    """Test decrypt_sensitive function."""

    def test_decrypts_encrypted_fields(self):
        """Encrypted fields are decrypted."""
        from aragora.storage.encrypted_fields import encrypt_sensitive, decrypt_sensitive

        original = {
            "name": "test",
            "api_key": "secret_key_123",
            "access_token": "token_abc",
        }

        encrypted = encrypt_sensitive(original)
        decrypted = decrypt_sensitive(encrypted)

        assert decrypted["name"] == "test"
        assert decrypted["api_key"] == "secret_key_123"
        assert decrypted["access_token"] == "token_abc"

    def test_handles_unencrypted_data(self):
        """Unencrypted data is returned unchanged."""
        from aragora.storage.encrypted_fields import decrypt_sensitive

        data = {"api_key": "plaintext_key", "name": "test"}
        result = decrypt_sensitive(data)
        assert result == data

    def test_handles_empty_data(self):
        """Empty dict returns empty dict."""
        from aragora.storage.encrypted_fields import decrypt_sensitive

        assert decrypt_sensitive({}) == {}
        assert decrypt_sensitive(None) is None

    def test_handles_mixed_encrypted_unencrypted(self):
        """Mix of encrypted and unencrypted fields works."""
        from aragora.storage.encrypted_fields import encrypt_sensitive, decrypt_sensitive

        # Start with data
        data = {"api_key": "secret", "password": "pass123"}

        # Encrypt only api_key
        encrypted = encrypt_sensitive({"api_key": "secret"})
        encrypted["password"] = "pass123"  # Add unencrypted field

        # Decrypt should handle both
        decrypted = decrypt_sensitive(encrypted)
        assert decrypted["api_key"] == "secret"
        assert decrypted["password"] == "pass123"


class TestRoundTrip:
    """Test encryption/decryption round-trip."""

    def test_round_trip_preserves_data(self):
        """Encrypt then decrypt returns original data."""
        from aragora.storage.encrypted_fields import encrypt_sensitive, decrypt_sensitive

        original = {
            "api_key": "sk-test-api-key-12345",
            "secret": "webhook_secret_abc",
            "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            "name": "My Integration",
            "enabled": True,
            "count": 42,
        }

        encrypted = encrypt_sensitive(original)
        decrypted = decrypt_sensitive(encrypted)

        assert decrypted == original

    def test_round_trip_with_special_characters(self):
        """Special characters in secrets are preserved."""
        from aragora.storage.encrypted_fields import encrypt_sensitive, decrypt_sensitive

        original = {
            "api_key": "sk-test!@#$%^&*()_+-=[]{}|;':\",./<>?",
            "password": "p@$$w0rd!",
        }

        encrypted = encrypt_sensitive(original)
        decrypted = decrypt_sensitive(encrypted)

        assert decrypted == original

    def test_round_trip_with_unicode(self):
        """Unicode in secrets is preserved."""
        from aragora.storage.encrypted_fields import encrypt_sensitive, decrypt_sensitive

        original = {
            "api_key": "密钥123",
            "secret": "🔐secret🔑",
        }

        encrypted = encrypt_sensitive(original)
        decrypted = decrypt_sensitive(encrypted)

        assert decrypted == original


class TestHelperFunctions:
    """Test helper functions."""

    def test_is_field_encrypted(self):
        """is_field_encrypted correctly identifies encrypted fields."""
        from aragora.storage.encrypted_fields import (
            encrypt_sensitive,
            is_field_encrypted,
        )

        data = {"api_key": "secret", "name": "test"}
        encrypted = encrypt_sensitive(data)

        assert is_field_encrypted(encrypted, "api_key") is True
        assert is_field_encrypted(encrypted, "name") is False
        assert is_field_encrypted(encrypted, "nonexistent") is False

    def test_get_encrypted_field_names(self):
        """get_encrypted_field_names returns list of encrypted fields."""
        from aragora.storage.encrypted_fields import (
            encrypt_sensitive,
            get_encrypted_field_names,
        )

        data = {"api_key": "secret", "access_token": "token", "name": "test"}
        encrypted = encrypt_sensitive(data)

        encrypted_names = get_encrypted_field_names(encrypted)
        assert "api_key" in encrypted_names
        assert "access_token" in encrypted_names
        assert "name" not in encrypted_names


class TestEncryptionAvailability:
    """Test encryption availability checks."""

    def test_is_encryption_available(self):
        """is_encryption_available returns True when cryptography is available."""
        from aragora.storage.encrypted_fields import is_encryption_available

        # Should be True in test environment with cryptography installed
        assert is_encryption_available() is True

    def test_is_encryption_configured(self):
        """is_encryption_configured checks for environment variable."""
        from aragora.storage.encrypted_fields import is_encryption_configured

        # We set ARAGORA_ENCRYPTION_KEY at module load
        assert is_encryption_configured() is True


class TestSensitiveFields:
    """Test SENSITIVE_FIELDS constant."""

    def test_contains_expected_fields(self):
        """SENSITIVE_FIELDS contains all expected field names."""
        from aragora.storage.encrypted_fields import SENSITIVE_FIELDS

        expected = {
            "access_token",
            "refresh_token",
            "api_key",
            "secret",
            "password",
            "auth_token",
            "bot_token",
            "webhook_url",
        }

        for field in expected:
            assert field in SENSITIVE_FIELDS, f"Missing sensitive field: {field}"

    def test_is_frozenset(self):
        """SENSITIVE_FIELDS is immutable frozenset."""
        from aragora.storage.encrypted_fields import SENSITIVE_FIELDS

        assert isinstance(SENSITIVE_FIELDS, frozenset)
