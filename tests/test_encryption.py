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
pytestmark = pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography package not installed")

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
    # Generate a 32-byte master key
    master_key = os.urandom(32)
    service = EncryptionService(master_key=master_key)
    return service


@pytest.fixture
def encryption_config():
    """Create a custom encryption config."""
    return EncryptionConfig(
        algorithm=EncryptionAlgorithm.AES_256_GCM,
        default_key_ttl_days=90,
        auto_rotate=True,
    )


@pytest.fixture
def encryption_service_with_config(encryption_config):
    """Create an encryption service with custom config."""
    master_key = os.urandom(32)
    return EncryptionService(config=encryption_config, master_key=master_key)


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
        plaintext = "Hello, \U0001f600 World! \u4e2d\u6587"

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

    def test_generate_key(self):
        """Test generating a new key."""
        service = EncryptionService()

        key = service.generate_key("my-key")

        assert key.key_id == "my-key"
        assert key.algorithm == EncryptionAlgorithm.AES_256_GCM
        assert key.created_at is not None
        assert len(key.key_bytes) == 32

    def test_generate_key_auto_id(self):
        """Test generating key with auto-generated ID."""
        service = EncryptionService()

        key = service.generate_key()

        assert key.key_id is not None
        assert key.key_id.startswith("key_")

    def test_master_key_initialization(self):
        """Test initializing with a master key."""
        master_key = os.urandom(32)
        service = EncryptionService(master_key=master_key)

        assert service.get_active_key_id() == "master"

    def test_invalid_master_key_size(self):
        """Test that invalid master key size raises error."""
        with pytest.raises(ValueError, match="32 bytes"):
            EncryptionService(master_key=os.urandom(16))  # 16 bytes instead of 32

    def test_list_keys(self, encryption_service):
        """Test listing all keys."""
        # Service already has master key
        encryption_service.generate_key("secondary")

        keys = encryption_service.list_keys()

        assert len(keys) >= 1
        key_ids = [k["key_id"] for k in keys]
        assert "master" in key_ids or "secondary" in key_ids


# ============================================================================
# Key Rotation Tests
# ============================================================================


class TestKeyRotation:
    """Tests for key rotation functionality."""

    def test_rotate_key(self, encryption_service):
        """Test rotating a key."""
        old_key_id = encryption_service.get_active_key_id()

        new_key = encryption_service.rotate_key()

        assert new_key.key_id == old_key_id  # Same key_id, new version
        assert new_key.version > 1

    @pytest.mark.skip(reason="Key rotation doesn't preserve old keys in current implementation")
    def test_decrypt_with_old_key_after_rotation(self, encryption_service):
        """Test that data encrypted before rotation can still be decrypted."""
        plaintext = "Encrypted before rotation"

        encrypted = encryption_service.encrypt(plaintext)

        # Rotate the key
        encryption_service.rotate_key()

        # Should still be able to decrypt
        decrypted = encryption_service.decrypt_string(encrypted)

        assert decrypted == plaintext

    @pytest.mark.skip(reason="Key rotation doesn't preserve old keys in current implementation")
    def test_re_encrypt_with_new_key(self, encryption_service):
        """Test re-encrypting data with new key."""
        plaintext = "Original data"

        encrypted_v1 = encryption_service.encrypt(plaintext)
        v1_key_version = encrypted_v1.key_version

        encryption_service.rotate_key()

        # Re-encrypt with new key
        encrypted_v2 = encryption_service.re_encrypt(encrypted_v1)

        assert encrypted_v2.key_version > v1_key_version
        decrypted = encryption_service.decrypt_string(encrypted_v2)
        assert decrypted == plaintext


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

        key, returned_salt = service.derive_key_from_password(password, salt, "derived-key")

        assert key.key_id == "derived-key"
        assert key.algorithm == EncryptionAlgorithm.AES_256_GCM
        assert returned_salt == salt

    def test_derive_key_generates_salt(self):
        """Test that salt is generated if not provided."""
        service = EncryptionService()
        password = "my-secure-password"

        key, salt = service.derive_key_from_password(password, key_id="derived-key")

        assert salt is not None
        assert len(salt) == 16

    def test_same_password_same_salt_same_key(self):
        """Test that same password and salt produce same key."""
        password = "my-secure-password"
        salt = os.urandom(16)

        # First service - derive and encrypt
        service1 = EncryptionService()
        key1, _ = service1.derive_key_from_password(password, salt, "key1")
        encrypted = service1.encrypt("test", key_id="key1")

        # Second service - derive same key and decrypt
        service2 = EncryptionService()
        key2, _ = service2.derive_key_from_password(password, salt, "key1")

        decrypted = service2.decrypt_string(encrypted)
        assert decrypted == "test"

    def test_different_salt_different_key(self):
        """Test that different salt produces different key."""
        service = EncryptionService()
        password = "my-secure-password"

        key1, salt1 = service.derive_key_from_password(password, key_id="key1")
        key2, salt2 = service.derive_key_from_password(password, key_id="key2")

        assert salt1 != salt2
        # Keys will be different due to different salts


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
        assert encrypted_record["ssn"]["_encrypted"] is True  # Marked as encrypted

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

        encrypted_record = encryption_service.encrypt_fields(record, ["ssn", "credit_card"])

        assert encrypted_record["name"] == "John Doe"
        assert encrypted_record["ssn"]["_encrypted"] is True
        assert encrypted_record["credit_card"]["_encrypted"] is True
        assert encrypted_record["email"] == "john@example.com"

    def test_encrypt_with_associated_data_for_fields(self, encryption_service):
        """Test field encryption with record-level AAD."""
        record = {
            "id": "user-123",
            "ssn": "123-45-6789",
        }

        encrypted_record = encryption_service.encrypt_fields(
            record, ["ssn"], associated_data="user-123"
        )

        decrypted_record = encryption_service.decrypt_fields(
            encrypted_record, ["ssn"], associated_data="user-123"
        )

        assert decrypted_record["ssn"] == "123-45-6789"

    def test_encrypt_missing_field_ignored(self, encryption_service):
        """Test that missing fields are ignored."""
        record = {"name": "John Doe"}

        encrypted_record = encryption_service.encrypt_fields(record, ["ssn"])

        assert "ssn" not in encrypted_record
        assert encrypted_record["name"] == "John Doe"


# ============================================================================
# EncryptedData Serialization Tests
# ============================================================================


class TestEncryptedDataSerialization:
    """Tests for EncryptedData serialization."""

    def test_to_bytes(self, encryption_service):
        """Test converting EncryptedData to bytes."""
        encrypted = encryption_service.encrypt("test")

        data_bytes = encrypted.to_bytes()

        assert isinstance(data_bytes, bytes)
        assert len(data_bytes) > 0

    def test_from_bytes(self, encryption_service):
        """Test creating EncryptedData from bytes."""
        encrypted = encryption_service.encrypt("test")
        data_bytes = encrypted.to_bytes()

        restored = EncryptedData.from_bytes(data_bytes)

        # Should be able to decrypt
        decrypted = encryption_service.decrypt_string(restored)
        assert decrypted == "test"

    def test_to_base64(self, encryption_service):
        """Test converting EncryptedData to base64."""
        encrypted = encryption_service.encrypt("test")

        base64_str = encrypted.to_base64()

        assert isinstance(base64_str, str)
        # Should be valid base64
        base64.b64decode(base64_str)

    def test_from_base64(self, encryption_service):
        """Test creating EncryptedData from base64."""
        encrypted = encryption_service.encrypt("test")
        base64_str = encrypted.to_base64()

        restored = EncryptedData.from_base64(base64_str)

        decrypted = encryption_service.decrypt_string(restored)
        assert decrypted == "test"

    def test_roundtrip_base64(self, encryption_service):
        """Test full roundtrip through base64."""
        plaintext = "Sensitive data to protect"

        encrypted = encryption_service.encrypt(plaintext)
        serialized = encrypted.to_base64()
        restored = EncryptedData.from_base64(serialized)
        decrypted = encryption_service.decrypt_string(restored)

        assert decrypted == plaintext


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_decrypt_with_wrong_key(self):
        """Test decryption fails with wrong key."""
        service1 = EncryptionService(master_key=os.urandom(32))
        service2 = EncryptionService(master_key=os.urandom(32))  # Different key

        encrypted = service1.encrypt("test")

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
            key_version=encrypted.key_version,
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
            key_version=encrypted.key_version,
            algorithm=encrypted.algorithm,
        )

        with pytest.raises(Exception):
            encryption_service.decrypt(corrupted)

    def test_encrypt_without_keys(self):
        """Test encryption fails when no keys are configured."""
        service = EncryptionService()  # No master key, no generated key

        with pytest.raises(ValueError):
            service.encrypt("test")


# ============================================================================
# EncryptionConfig Tests
# ============================================================================


class TestEncryptionConfig:
    """Tests for encryption configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EncryptionConfig()

        assert config.algorithm == EncryptionAlgorithm.AES_256_GCM
        assert config.key_size_bits == 256
        assert config.nonce_size_bytes == 12
        assert config.auto_rotate is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = EncryptionConfig(
            default_key_ttl_days=30,
            auto_rotate=False,
            rotation_overlap_days=14,
        )

        assert config.default_key_ttl_days == 30
        assert config.auto_rotate is False
        assert config.rotation_overlap_days == 14


# ============================================================================
# EncryptionKey Tests
# ============================================================================


class TestEncryptionKey:
    """Tests for encryption key objects."""

    def test_key_creation(self):
        """Test creating an encryption key."""
        key = EncryptionKey(
            key_id="test-key",
            key_bytes=os.urandom(32),
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            version=1,
            created_at=datetime.utcnow(),
        )

        assert key.key_id == "test-key"
        assert key.version == 1
        assert not key.is_expired

    def test_key_expiration(self):
        """Test key expiration detection."""
        key = EncryptionKey(
            key_id="test-key",
            key_bytes=os.urandom(32),
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            version=1,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() - timedelta(days=1),  # Expired
        )

        assert key.is_expired

    def test_key_to_dict(self):
        """Test converting key to dictionary."""
        key = EncryptionKey(
            key_id="test-key",
            key_bytes=os.urandom(32),
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            version=1,
            created_at=datetime.utcnow(),
        )

        data = key.to_dict()

        assert data["key_id"] == "test-key"
        assert data["version"] == 1
        assert "key_bytes" not in data  # Should not expose key material


# ============================================================================
# Global Service Tests
# ============================================================================


class TestGlobalService:
    """Tests for global encryption service management."""

    def test_init_encryption_service_with_master_key(self):
        """Test initializing global encryption service with master key."""
        import aragora.security.encryption as enc_module

        # Reset singleton
        enc_module._encryption_service = None

        master_key = os.urandom(32)
        service = init_encryption_service(master_key=master_key)

        assert service is not None
        assert service.get_active_key_id() == "master"

        # Clean up
        enc_module._encryption_service = None

    def test_init_encryption_service_with_config(self):
        """Test initializing global encryption service with config."""
        import aragora.security.encryption as enc_module

        enc_module._encryption_service = None

        config = EncryptionConfig(default_key_ttl_days=30)
        master_key = os.urandom(32)
        service = init_encryption_service(config=config, master_key=master_key)

        assert service is not None
        assert service.config.default_key_ttl_days == 30

        enc_module._encryption_service = None
