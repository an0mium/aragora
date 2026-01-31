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


# ===========================================================================
# Test Error Handling
# ===========================================================================


class TestEncryptionErrors:
    """Test error handling during encryption/decryption."""

    def test_encryption_error_class(self):
        """EncryptionError is properly defined."""
        from aragora.storage.encrypted_fields import EncryptionError

        error = EncryptionError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_decryption_error_class(self):
        """DecryptionError is properly defined."""
        from aragora.storage.encrypted_fields import DecryptionError

        error = DecryptionError("Test decryption error")
        assert str(error) == "Test decryption error"
        assert isinstance(error, Exception)

    def test_encrypt_with_invalid_service(self):
        """Encryption with failing service raises EncryptionError."""
        from aragora.storage.encrypted_fields import EncryptionError

        with patch("aragora.storage.encrypted_fields._get_encryption_service") as mock_service:
            mock_svc = MagicMock()
            mock_svc.encrypt_fields.side_effect = RuntimeError("Service failure")
            mock_service.return_value = mock_svc

            from aragora.storage.encrypted_fields import encrypt_sensitive

            with pytest.raises(EncryptionError, match="Failed to encrypt"):
                encrypt_sensitive({"api_key": "secret"})

    def test_decrypt_with_invalid_service(self):
        """Decryption with failing service raises DecryptionError."""
        from aragora.storage.encrypted_fields import DecryptionError

        with patch("aragora.storage.encrypted_fields._get_encryption_service") as mock_service:
            mock_svc = MagicMock()
            mock_svc.decrypt_fields.side_effect = RuntimeError("Service failure")
            mock_service.return_value = mock_svc

            from aragora.storage.encrypted_fields import decrypt_sensitive

            encrypted_data = {"api_key": {"_encrypted": True, "_value": "corrupted_data"}}
            with pytest.raises(DecryptionError, match="Failed to decrypt"):
                decrypt_sensitive(encrypted_data)

    def test_decrypt_wrong_record_id(self):
        """Decryption with wrong record ID fails."""
        from aragora.storage.encrypted_fields import (
            encrypt_sensitive,
            decrypt_sensitive,
            DecryptionError,
        )

        data = {"api_key": "secret123"}

        # Encrypt with one record ID
        encrypted = encrypt_sensitive(data, record_id="user_123")

        # Try to decrypt with different record ID - should fail due to AAD mismatch
        with pytest.raises(DecryptionError):
            decrypt_sensitive(encrypted, record_id="user_456")


# ===========================================================================
# Test Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases for encryption/decryption."""

    def test_empty_string_value(self):
        """Empty string values are encrypted."""
        from aragora.storage.encrypted_fields import encrypt_sensitive, decrypt_sensitive

        data = {"api_key": ""}
        encrypted = encrypt_sensitive(data)
        decrypted = decrypt_sensitive(encrypted)

        assert decrypted["api_key"] == ""

    def test_very_long_value(self):
        """Very long values can be encrypted and decrypted."""
        from aragora.storage.encrypted_fields import encrypt_sensitive, decrypt_sensitive

        long_key = "a" * 10000  # 10KB key
        data = {"api_key": long_key}

        encrypted = encrypt_sensitive(data)
        decrypted = decrypt_sensitive(encrypted)

        assert decrypted["api_key"] == long_key

    def test_binary_like_data(self):
        """Binary-like string data is handled correctly."""
        from aragora.storage.encrypted_fields import encrypt_sensitive, decrypt_sensitive

        binary_str = "\x00\x01\x02\x03\xff\xfe\xfd"
        data = {"api_key": binary_str}

        encrypted = encrypt_sensitive(data)
        decrypted = decrypt_sensitive(encrypted)

        assert decrypted["api_key"] == binary_str

    def test_json_in_value(self):
        """JSON strings in values are parsed after decryption.

        Note: The encryption service uses JSON serialization internally,
        so valid JSON strings are parsed back to their Python equivalents.
        """
        from aragora.storage.encrypted_fields import encrypt_sensitive, decrypt_sensitive

        json_value = '{"nested": {"key": "value"}, "array": [1, 2, 3]}'
        data = {"api_key": json_value}

        encrypted = encrypt_sensitive(data)
        decrypted = decrypt_sensitive(encrypted)

        # JSON strings are parsed back to their Python equivalents
        expected = {"nested": {"key": "value"}, "array": [1, 2, 3]}
        assert decrypted["api_key"] == expected

    def test_numeric_string_value(self):
        """Numeric string values may be parsed to integers.

        Note: The encryption service uses JSON serialization internally,
        which can convert numeric strings to integers.
        """
        from aragora.storage.encrypted_fields import encrypt_sensitive, decrypt_sensitive

        data = {"api_key": "12345678901234567890"}

        encrypted = encrypt_sensitive(data)
        decrypted = decrypt_sensitive(encrypted)

        # Numeric strings are parsed back to integers by JSON serialization
        assert decrypted["api_key"] == 12345678901234567890

    def test_multiple_sensitive_fields(self):
        """Multiple sensitive fields are all encrypted."""
        from aragora.storage.encrypted_fields import (
            encrypt_sensitive,
            decrypt_sensitive,
            is_field_encrypted,
        )

        data = {
            "api_key": "key1",
            "access_token": "token1",
            "refresh_token": "refresh1",
            "secret": "secret1",
            "password": "pass1",
            "regular_field": "not_encrypted",
        }

        encrypted = encrypt_sensitive(data)

        # Check all sensitive fields are encrypted
        assert is_field_encrypted(encrypted, "api_key")
        assert is_field_encrypted(encrypted, "access_token")
        assert is_field_encrypted(encrypted, "refresh_token")
        assert is_field_encrypted(encrypted, "secret")
        assert is_field_encrypted(encrypted, "password")

        # Regular field should not be encrypted
        assert not is_field_encrypted(encrypted, "regular_field")
        assert encrypted["regular_field"] == "not_encrypted"

        # Decrypt and verify
        decrypted = decrypt_sensitive(encrypted)
        assert decrypted["api_key"] == "key1"
        assert decrypted["access_token"] == "token1"
        assert decrypted["refresh_token"] == "refresh1"
        assert decrypted["secret"] == "secret1"
        assert decrypted["password"] == "pass1"

    def test_nested_dict_not_sensitive(self):
        """Nested dictionaries in non-sensitive fields are preserved."""
        from aragora.storage.encrypted_fields import encrypt_sensitive, decrypt_sensitive

        data = {
            "config": {"nested": {"deep": "value"}},
            "api_key": "secret",
        }

        encrypted = encrypt_sensitive(data)
        decrypted = decrypt_sensitive(encrypted)

        assert decrypted["config"] == {"nested": {"deep": "value"}}
        assert decrypted["api_key"] == "secret"

    def test_list_values_in_non_sensitive(self):
        """List values in non-sensitive fields are preserved."""
        from aragora.storage.encrypted_fields import encrypt_sensitive, decrypt_sensitive

        data = {
            "tags": ["tag1", "tag2", "tag3"],
            "api_key": "secret",
        }

        encrypted = encrypt_sensitive(data)
        decrypted = decrypt_sensitive(encrypted)

        assert decrypted["tags"] == ["tag1", "tag2", "tag3"]

    def test_boolean_values_preserved(self):
        """Boolean values in non-sensitive fields are preserved."""
        from aragora.storage.encrypted_fields import encrypt_sensitive, decrypt_sensitive

        data = {
            "enabled": True,
            "disabled": False,
            "api_key": "secret",
        }

        encrypted = encrypt_sensitive(data)
        decrypted = decrypt_sensitive(encrypted)

        assert decrypted["enabled"] is True
        assert decrypted["disabled"] is False

    def test_integer_values_preserved(self):
        """Integer values in non-sensitive fields are preserved."""
        from aragora.storage.encrypted_fields import encrypt_sensitive, decrypt_sensitive

        data = {
            "count": 42,
            "negative": -10,
            "api_key": "secret",
        }

        encrypted = encrypt_sensitive(data)
        decrypted = decrypt_sensitive(encrypted)

        assert decrypted["count"] == 42
        assert decrypted["negative"] == -10


# ===========================================================================
# Test Partial Encryption/Decryption
# ===========================================================================


class TestPartialOperations:
    """Test partial encryption/decryption scenarios."""

    def test_decrypt_already_decrypted(self):
        """Decrypting already-decrypted data returns it unchanged."""
        from aragora.storage.encrypted_fields import decrypt_sensitive

        data = {"api_key": "already_plain_text", "name": "test"}
        result = decrypt_sensitive(data)

        # Should be unchanged
        assert result == data

    def test_encrypt_already_encrypted_field(self):
        """Encrypting data with already-encrypted fields handles them correctly."""
        from aragora.storage.encrypted_fields import encrypt_sensitive

        # Data where api_key looks like encrypted format (but isn't really)
        data = {
            "api_key": {"_encrypted": True, "_value": "some_value"},
            "name": "test",
        }

        # Since api_key is a dict with _encrypted=True, it's already "encrypted"
        # The function should skip None values, but this dict is not None
        result = encrypt_sensitive(data)

        # The structure should remain unchanged since it's already encrypted-looking
        assert result["api_key"]["_encrypted"] is True

    def test_mixed_encrypted_unencrypted_decrypt(self):
        """Decrypting mixed data handles both types correctly."""
        from aragora.storage.encrypted_fields import encrypt_sensitive, decrypt_sensitive

        # Encrypt only one field
        data1 = {"api_key": "secret1"}
        encrypted1 = encrypt_sensitive(data1)

        # Add more fields that aren't encrypted
        encrypted1["password"] = "plain_password"
        encrypted1["normal_field"] = "value"

        decrypted = decrypt_sensitive(encrypted1)

        assert decrypted["api_key"] == "secret1"
        assert decrypted["password"] == "plain_password"
        assert decrypted["normal_field"] == "value"


# ===========================================================================
# Test Encryption Service Integration
# ===========================================================================


class TestEncryptionServiceIntegration:
    """Test integration with encryption service."""

    def test_get_encryption_service_lazy_import(self):
        """_get_encryption_service performs lazy import."""
        from aragora.storage.encrypted_fields import _get_encryption_service

        # Should not raise
        service = _get_encryption_service()
        assert service is not None

    def test_encryption_uses_aad(self):
        """Encryption properly uses AAD (record_id)."""
        from aragora.storage.encrypted_fields import encrypt_sensitive, decrypt_sensitive

        data = {"api_key": "test_secret"}

        # Encrypt with AAD
        encrypted = encrypt_sensitive(data, record_id="unique_id_123")

        # Decrypt with same AAD should work
        decrypted = decrypt_sensitive(encrypted, record_id="unique_id_123")
        assert decrypted["api_key"] == "test_secret"


# ===========================================================================
# Test All Sensitive Fields
# ===========================================================================


class TestAllSensitiveFields:
    """Test all defined sensitive fields."""

    def test_all_sensitive_fields_encrypt(self):
        """Verify all SENSITIVE_FIELDS are properly encrypted."""
        from aragora.storage.encrypted_fields import (
            SENSITIVE_FIELDS,
            encrypt_sensitive,
            is_field_encrypted,
        )

        # Create data with all sensitive fields
        data = {field: f"value_{field}" for field in SENSITIVE_FIELDS}
        data["non_sensitive"] = "regular_value"

        encrypted = encrypt_sensitive(data)

        # All sensitive fields should be encrypted
        for field in SENSITIVE_FIELDS:
            assert is_field_encrypted(encrypted, field), f"{field} should be encrypted"

        # Non-sensitive should not be encrypted
        assert not is_field_encrypted(encrypted, "non_sensitive")

    def test_platform_specific_fields(self):
        """Test platform-specific credential fields."""
        from aragora.storage.encrypted_fields import (
            encrypt_sensitive,
            decrypt_sensitive,
        )

        data = {
            "slack_signing_secret": "slack_secret",
            "discord_token": "discord_token",
            "telegram_token": "telegram_token",
            "github_token": "github_token",
            "sendgrid_api_key": "sendgrid_key",
            "twilio_auth_token": "twilio_token",
        }

        encrypted = encrypt_sensitive(data)
        decrypted = decrypt_sensitive(encrypted)

        for key, value in data.items():
            assert decrypted[key] == value, f"{key} should be correctly decrypted"


# ===========================================================================
# Test Encryption Without Configuration
# ===========================================================================


class TestEncryptionWithoutConfig:
    """Test behavior when encryption is not configured."""

    def test_encrypt_returns_data_unchanged_without_crypto(self):
        """encrypt_sensitive returns data unchanged when crypto unavailable."""
        from aragora.storage.encrypted_fields import encrypt_sensitive

        with patch(
            "aragora.storage.encrypted_fields.is_encryption_available",
            return_value=False,
        ):
            data = {"api_key": "secret"}
            result = encrypt_sensitive(data)

            # Should be unchanged since encryption not available
            assert result == data

    def test_decrypt_returns_data_unchanged_without_crypto(self):
        """decrypt_sensitive returns data unchanged when crypto unavailable."""
        from aragora.storage.encrypted_fields import decrypt_sensitive

        with patch(
            "aragora.storage.encrypted_fields.is_encryption_available",
            return_value=False,
        ):
            data = {"api_key": {"_encrypted": True, "_value": "encrypted"}}
            result = decrypt_sensitive(data)

            # Should be unchanged since decryption not available
            assert result == data

    def test_is_encryption_configured_false_when_no_key(self):
        """is_encryption_configured returns False when key not set."""
        from aragora.storage.encrypted_fields import is_encryption_configured

        with patch.dict(os.environ, {}, clear=True):
            # Remove the key we set at module load
            os.environ.pop("ARAGORA_ENCRYPTION_KEY", None)
            assert is_encryption_configured() is False


# ===========================================================================
# Test Thread Safety
# ===========================================================================


class TestThreadSafety:
    """Test thread safety of encryption operations."""

    def test_concurrent_encryption(self):
        """Concurrent encryption operations work correctly."""
        from aragora.storage.encrypted_fields import encrypt_sensitive, decrypt_sensitive
        import threading

        results = []
        errors = []

        def encrypt_worker(thread_id):
            try:
                for i in range(10):
                    data = {"api_key": f"secret_{thread_id}_{i}"}
                    encrypted = encrypt_sensitive(data, record_id=f"record_{thread_id}_{i}")
                    decrypted = decrypt_sensitive(encrypted, record_id=f"record_{thread_id}_{i}")
                    assert decrypted["api_key"] == f"secret_{thread_id}_{i}"
                    results.append(True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=encrypt_worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors occurred: {errors}"
        assert len(results) == 50
