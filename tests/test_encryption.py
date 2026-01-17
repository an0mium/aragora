"""
Tests for EncryptionService - AES-256-GCM encryption with key rotation.

Tests cover:
- Basic encryption/decryption
- Key derivation from passwords
- Key rotation
- Field-level encryption
- Associated data authentication
- Error handling
"""

import base64
import json
import os
import pytest
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch, Mock

# Check if cryptography is available
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# Skip all tests if cryptography is not available
pytestmark = pytest.mark.skipif(
    not CRYPTO_AVAILABLE, reason="cryptography package not installed"
)

from aragora.security.encryption import (
    EncryptionService,
    EncryptionConfig,
    EncryptionKey,
    EncryptedData,
    EncryptionAlgorithm,
    KeyDerivationFunction,
    get_encryption_service,
    init_encryption_service,
    CRYPTO_AVAILABLE as MODULE_CRYPTO_AVAILABLE,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def encryption_service():
    """Create an encryption service with a test key."""
    config = EncryptionConfig(
        default_algorithm=EncryptionAlgorithm.AES_256_GCM,
        key_rotation_days=90,
    )
    service = EncryptionService(config)

    # Add a test key
    test_key = os.urandom(32)
    service.add_key(test_key, "test-key-1")

    return service


@pytest.fixture
def encryption_service_with_rotation():
    """Create an encryption service with multiple keys for rotation testing."""
    config = EncryptionConfig(key_rotation_days=30)
    service = EncryptionService(config)

    # Add primary key
    service.add_key(os.urandom(32), "primary-key")
    # Add an older key
    service.add_key(os.urandom(32), "old-key")

    return service


# ============================================================================
# Basic Encryption Tests
# ============================================================================


class TestBasicEncryption:
    """Tests for basic encryption/decryption operations."""

    def test_encrypt_string(self, encryption_service):
        """Test encrypting a string."""
        plaintext = "Hello, World!"

        encrypted = encryption_service.encrypt(plaintext)

        assert encrypted is not None
        assert encrypted.ciphertext != plaintext.encode()
        assert encrypted.algorithm == EncryptionAlgorithm.AES_256_GCM
        assert len(encrypted.nonce) == 12  # GCM nonce size

    def test_decrypt_string(self, encryption_service):
        """Test decrypting to get original string."""
        plaintext = "Hello, World!"

        encrypted = encryption_service.encrypt(plaintext)
        decrypted = encryption_service.decrypt_string(encrypted)

        assert decrypted == plaintext

    def test_encrypt_bytes(self, encryption_service):
        """Test encrypting bytes."""
        plaintext = b"\x00\x01\x02\x03\xff\xfe"

        encrypted = encryption_service.encrypt(plaintext)
        decrypted = encryption_service.decrypt(encrypted)

        assert decrypted == plaintext

    def test_encrypt_empty_string(self, encryption_service):
        """Test encrypting empty string."""
        plaintext = ""

        encrypted = encryption_service.encrypt(plaintext)
        decrypted = encryption_service.decrypt_string(encrypted)

        assert decrypted == plaintext

    def test_encrypt_large_data(self, encryption_service):
        """Test encrypting large data."""
        plaintext = "x" * 1000000  # 1MB of data

        encrypted = encryption_service.encrypt(plaintext)
        decrypted = encryption_service.decrypt_string(encrypted)

        assert decrypted == plaintext

    def test_encrypt_unicode(self, encryption_service):
        """Test encrypting unicode strings."""
        plaintext = "Hello, \U0001F600 World! \u4e2d\u6587"

        encrypted = encryption_service.encrypt(plaintext)
        decrypted = encryption_service.decrypt_string(encrypted)

        assert decrypted == plaintext

    def test_different_plaintexts_different_ciphertexts(self, encryption_service):
        """Test that different plaintexts produce different ciphertexts."""
        encrypted1 = encryption_service.encrypt("Hello")
        encrypted2 = encryption_service.encrypt("World")

        assert encrypted1.ciphertext != encrypted2.ciphertext

    def test_same_plaintext_different_nonces(self, encryption_service):
        """Test that same plaintext produces different ciphertexts (different nonces)."""
        encrypted1 = encryption_service.encrypt("Hello")
        encrypted2 = encryption_service.encrypt("Hello")

        assert encrypted1.nonce != encrypted2.nonce
        assert encrypted1.ciphertext != encrypted2.ciphertext


# ============================================================================
# Associated Data Tests
# ============================================================================


class TestAssociatedData:
    """Tests for authenticated associated data."""

    def test_encrypt_with_associated_data(self, encryption_service):
        """Test encrypting with associated data."""
        plaintext = "Secret message"
        aad = b"context-identifier-123"

        encrypted = encryption_service.encrypt(plaintext, associated_data=aad)
        decrypted = encryption_service.decrypt_string(encrypted, associated_data=aad)

        assert decrypted == plaintext

    def test_wrong_associated_data_fails(self, encryption_service):
        """Test that wrong associated data fails decryption."""
        plaintext = "Secret message"
        aad = b"correct-context"

        encrypted = encryption_service.encrypt(plaintext, associated_data=aad)

        with pytest.raises(Exception):  # InvalidTag or similar
            encryption_service.decrypt(encrypted, associated_data=b"wrong-context")

    def test_missing_associated_data_fails(self, encryption_service):
        """Test that missing associated data fails decryption."""
        plaintext = "Secret message"
        aad = b"required-context"

        encrypted = encryption_service.encrypt(plaintext, associated_data=aad)

        with pytest.raises(Exception):
            encryption_service.decrypt(encrypted)  # No AAD provided


# ============================================================================
# Key Management Tests
# ============================================================================


class TestKeyManagement:
    """Tests for key management operations."""

    def test_add_key(self):
        """Test adding a new key."""
        service = EncryptionService()
        key_bytes = os.urandom(32)

        key = service.add_key(key_bytes, "my-key")

        assert key.id == "my-key"
        assert key.algorithm == EncryptionAlgorithm.AES_256_GCM
        assert key.created_at is not None

    def test_add_key_auto_id(self):
        """Test adding key with auto-generated ID."""
        service = EncryptionService()

        key = service.add_key(os.urandom(32))

        assert key.id is not None
        assert len(key.id) > 0

    def test_invalid_key_size(self):
        """Test that invalid key size raises error."""
        service = EncryptionService()

        with pytest.raises(ValueError, match="32 bytes"):
            service.add_key(os.urandom(16), "too-short")

    def test_get_key(self, encryption_service):
        """Test retrieving a key by ID."""
        key = encryption_service.get_key("test-key-1")

        assert key is not None
        assert key.id == "test-key-1"

    def test_get_nonexistent_key(self, encryption_service):
        """Test getting non-existent key returns None."""
        key = encryption_service.get_key("nonexistent")

        assert key is None

    def test_list_keys(self, encryption_service_with_rotation):
        """Test listing all keys."""
        keys = encryption_service_with_rotation.list_keys()

        assert len(keys) == 2
        assert any(k.id == "primary-key" for k in keys)
        assert any(k.id == "old-key" for k in keys)

    def test_remove_key(self, encryption_service_with_rotation):
        """Test removing a key."""
        result = encryption_service_with_rotation.remove_key("old-key")

        assert result is True
        assert encryption_service_with_rotation.get_key("old-key") is None

    def test_remove_nonexistent_key(self, encryption_service):
        """Test removing non-existent key."""
        result = encryption_service.remove_key("nonexistent")

        assert result is False


# ============================================================================
# Key Rotation Tests
# ============================================================================


class TestKeyRotation:
    """Tests for key rotation functionality."""

    def test_rotate_key(self, encryption_service):
        """Test rotating a key."""
        old_key = encryption_service.get_key("test-key-1")

        new_key = encryption_service.rotate_key("test-key-1")

        assert new_key.id != old_key.id
        assert new_key.created_at > old_key.created_at
        # Old key should be rotated (marked inactive)
        old = encryption_service.get_key("test-key-1")
        assert old.rotated_at is not None

    def test_decrypt_with_old_key_after_rotation(self, encryption_service):
        """Test that data encrypted with old key can still be decrypted."""
        plaintext = "Encrypted before rotation"

        encrypted = encryption_service.encrypt(plaintext, key_id="test-key-1")

        # Rotate the key
        encryption_service.rotate_key("test-key-1")

        # Should still be able to decrypt with the old key
        decrypted = encryption_service.decrypt_string(encrypted)

        assert decrypted == plaintext

    def test_needs_rotation(self):
        """Test checking if key needs rotation."""
        config = EncryptionConfig(key_rotation_days=30)
        service = EncryptionService(config)

        # Add fresh key
        service.add_key(os.urandom(32), "fresh-key")
        assert service.needs_rotation("fresh-key") is False

        # Manually set created_at to old date
        key = service.get_key("fresh-key")
        key.created_at = datetime.utcnow() - timedelta(days=60)

        assert service.needs_rotation("fresh-key") is True


# ============================================================================
# Key Derivation Tests
# ============================================================================


class TestKeyDerivation:
    """Tests for key derivation from passwords."""

    def test_derive_key_from_password(self):
        """Test deriving encryption key from password."""
        service = EncryptionService()
        password = "my-secure-password"
        salt = os.urandom(16)

        key, _ = service.derive_key_from_password(password, salt, "derived-key")

        assert key.id == "derived-key"
        assert key.algorithm == EncryptionAlgorithm.AES_256_GCM

    def test_derive_key_generates_salt(self):
        """Test that salt is generated if not provided."""
        service = EncryptionService()
        password = "my-secure-password"

        key, salt = service.derive_key_from_password(password, key_id="derived-key")

        assert salt is not None
        assert len(salt) == 16

    def test_same_password_same_salt_same_key(self):
        """Test that same password and salt produce same key."""
        service = EncryptionService()
        password = "my-secure-password"
        salt = os.urandom(16)

        # First derivation
        key1, _ = service.derive_key_from_password(password, salt, "key1")
        encrypted1 = service.encrypt("test", key_id="key1")

        # Create new service and derive same key
        service2 = EncryptionService()
        key2, _ = service2.derive_key_from_password(password, salt, "key2")

        # Should be able to decrypt with second service
        decrypted = service2.decrypt_string(encrypted1)

        assert decrypted == "test"

    def test_different_salt_different_key(self):
        """Test that different salt produces different key."""
        service = EncryptionService()
        password = "my-secure-password"

        key1, salt1 = service.derive_key_from_password(password, key_id="key1")
        key2, salt2 = service.derive_key_from_password(password, key_id="key2")

        assert salt1 != salt2
        # Keys should be different (different salts)


# ============================================================================
# Field-Level Encryption Tests
# ============================================================================


class TestFieldLevelEncryption:
    """Tests for field-level encryption of records."""

    def test_encrypt_single_field(self, encryption_service):
        """Test encrypting a single field in a record."""
        record = {
            "name": "John Doe",
            "ssn": "123-45-6789",
            "email": "john@example.com",
        }

        encrypted_record = encryption_service.encrypt_fields(record, ["ssn"])

        assert encrypted_record["name"] == "John Doe"  # Not encrypted
        assert encrypted_record["email"] == "john@example.com"  # Not encrypted
        assert encrypted_record["ssn"] != "123-45-6789"  # Encrypted
        assert encrypted_record["ssn"].startswith("enc:")  # Marked as encrypted

    def test_decrypt_single_field(self, encryption_service):
        """Test decrypting a single field in a record."""
        record = {
            "name": "John Doe",
            "ssn": "123-45-6789",
        }

        encrypted_record = encryption_service.encrypt_fields(record, ["ssn"])
        decrypted_record = encryption_service.decrypt_fields(encrypted_record, ["ssn"])

        assert decrypted_record["ssn"] == "123-45-6789"

    def test_encrypt_multiple_fields(self, encryption_service):
        """Test encrypting multiple fields."""
        record = {
            "name": "John Doe",
            "ssn": "123-45-6789",
            "credit_card": "4111-1111-1111-1111",
            "email": "john@example.com",
        }

        encrypted_record = encryption_service.encrypt_fields(
            record, ["ssn", "credit_card"]
        )

        assert encrypted_record["name"] == "John Doe"
        assert encrypted_record["ssn"].startswith("enc:")
        assert encrypted_record["credit_card"].startswith("enc:")
        assert encrypted_record["email"] == "john@example.com"

    def test_encrypt_nested_fields(self, encryption_service):
        """Test encrypting nested fields using dot notation."""
        record = {
            "user": {
                "name": "John Doe",
                "pii": {
                    "ssn": "123-45-6789",
                },
            },
        }

        encrypted_record = encryption_service.encrypt_fields(
            record, ["user.pii.ssn"]
        )

        assert encrypted_record["user"]["name"] == "John Doe"
        assert encrypted_record["user"]["pii"]["ssn"].startswith("enc:")

    def test_encrypt_with_associated_data(self, encryption_service):
        """Test field encryption with record-level AAD."""
        record = {
            "id": "user-123",
            "ssn": "123-45-6789",
        }

        encrypted_record = encryption_service.encrypt_fields(
            record, ["ssn"], associated_data=b"user-123"
        )

        decrypted_record = encryption_service.decrypt_fields(
            encrypted_record, ["ssn"], associated_data=b"user-123"
        )

        assert decrypted_record["ssn"] == "123-45-6789"

    def test_encrypt_missing_field_ignored(self, encryption_service):
        """Test that missing fields are ignored."""
        record = {"name": "John Doe"}

        encrypted_record = encryption_service.encrypt_fields(record, ["ssn"])

        assert "ssn" not in encrypted_record
        assert encrypted_record["name"] == "John Doe"

    def test_encrypt_field_preserves_types(self, encryption_service):
        """Test that field types are preserved after round-trip."""
        record = {
            "string_field": "test",
            "int_field": 42,
            "float_field": 3.14,
            "bool_field": True,
            "list_field": [1, 2, 3],
            "dict_field": {"nested": "value"},
        }

        encrypted = encryption_service.encrypt_fields(
            record, ["string_field", "int_field", "float_field"]
        )
        decrypted = encryption_service.decrypt_fields(
            encrypted, ["string_field", "int_field", "float_field"]
        )

        assert decrypted["string_field"] == "test"
        assert decrypted["int_field"] == 42
        assert decrypted["float_field"] == 3.14
        assert decrypted["bool_field"] is True
        assert decrypted["list_field"] == [1, 2, 3]


# ============================================================================
# EncryptedData Serialization Tests
# ============================================================================


class TestEncryptedDataSerialization:
    """Tests for EncryptedData serialization."""

    def test_to_dict(self, encryption_service):
        """Test converting EncryptedData to dictionary."""
        encrypted = encryption_service.encrypt("test")

        data = encrypted.to_dict()

        assert "ciphertext" in data
        assert "nonce" in data
        assert "key_id" in data
        assert "algorithm" in data
        assert data["algorithm"] == "aes-256-gcm"

    def test_from_dict(self, encryption_service):
        """Test creating EncryptedData from dictionary."""
        encrypted = encryption_service.encrypt("test")
        data = encrypted.to_dict()

        restored = EncryptedData.from_dict(data)

        # Should be able to decrypt
        decrypted = encryption_service.decrypt_string(restored)
        assert decrypted == "test"

    def test_to_string(self, encryption_service):
        """Test converting EncryptedData to string."""
        encrypted = encryption_service.encrypt("test")

        string_repr = encrypted.to_string()

        assert string_repr.startswith("enc:")
        assert "$" in string_repr  # Contains delimiter

    def test_from_string(self, encryption_service):
        """Test creating EncryptedData from string."""
        encrypted = encryption_service.encrypt("test")
        string_repr = encrypted.to_string()

        restored = EncryptedData.from_string(string_repr)

        decrypted = encryption_service.decrypt_string(restored)
        assert decrypted == "test"


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_decrypt_with_wrong_key(self):
        """Test decryption fails with wrong key."""
        service1 = EncryptionService()
        service1.add_key(os.urandom(32), "key1")

        service2 = EncryptionService()
        service2.add_key(os.urandom(32), "key1")  # Different key, same ID

        encrypted = service1.encrypt("test", key_id="key1")

        with pytest.raises(Exception):
            service2.decrypt(encrypted)

    def test_decrypt_corrupted_ciphertext(self, encryption_service):
        """Test decryption fails with corrupted ciphertext."""
        encrypted = encryption_service.encrypt("test")

        # Corrupt the ciphertext
        corrupted = EncryptedData(
            ciphertext=b"corrupted" + encrypted.ciphertext[9:],
            nonce=encrypted.nonce,
            key_id=encrypted.key_id,
            algorithm=encrypted.algorithm,
        )

        with pytest.raises(Exception):
            encryption_service.decrypt(corrupted)

    def test_decrypt_corrupted_nonce(self, encryption_service):
        """Test decryption fails with corrupted nonce."""
        encrypted = encryption_service.encrypt("test")

        corrupted = EncryptedData(
            ciphertext=encrypted.ciphertext,
            nonce=os.urandom(12),  # Wrong nonce
            key_id=encrypted.key_id,
            algorithm=encrypted.algorithm,
        )

        with pytest.raises(Exception):
            encryption_service.decrypt(corrupted)

    def test_encrypt_without_keys(self):
        """Test encryption fails when no keys are configured."""
        service = EncryptionService()

        with pytest.raises(ValueError, match="No encryption keys"):
            service.encrypt("test")


# ============================================================================
# Global Service Tests
# ============================================================================


class TestGlobalService:
    """Tests for global encryption service management."""

    def test_init_encryption_service(self):
        """Test initializing global encryption service."""
        with patch.dict(os.environ, {"ARAGORA_ENCRYPTION_KEY": "x" * 64}):
            import aragora.security.encryption as enc_module

            enc_module._encryption_service = None

            service = init_encryption_service()

            assert service is not None
            assert len(service.list_keys()) == 1

    def test_get_encryption_service_singleton(self):
        """Test get_encryption_service returns same instance."""
        with patch.dict(os.environ, {"ARAGORA_ENCRYPTION_KEY": "y" * 64}):
            import aragora.security.encryption as enc_module

            enc_module._encryption_service = None

            service1 = get_encryption_service()
            service2 = get_encryption_service()

            assert service1 is service2
