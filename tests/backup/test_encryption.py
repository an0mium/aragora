"""
Tests for backup encryption module.

Tests:
- Key generation
- File encryption/decryption
- Bytes encryption/decryption
- Key management
- Error handling
"""

import pytest
import tempfile
from pathlib import Path


class TestEncryptionKey:
    """Tests for EncryptionKey dataclass."""

    def test_key_must_be_32_bytes(self):
        """Key must be exactly 32 bytes."""
        from aragora.backup.encryption import EncryptionKey, KEY_SIZE

        # Valid key
        key = EncryptionKey(key_id="test", key=b"a" * KEY_SIZE)
        assert len(key.key) == KEY_SIZE

        # Invalid key - too short
        with pytest.raises(ValueError, match="32 bytes"):
            EncryptionKey(key_id="test", key=b"short")

        # Invalid key - too long
        with pytest.raises(ValueError, match="32 bytes"):
            EncryptionKey(key_id="test", key=b"a" * 64)

    def test_key_metadata(self):
        """Key can have optional metadata."""
        from aragora.backup.encryption import EncryptionKey, KEY_SIZE

        key = EncryptionKey(
            key_id="test-key",
            key=b"a" * KEY_SIZE,
            created_at="2024-01-01T00:00:00Z",
            description="Test key",
        )

        assert key.key_id == "test-key"
        assert key.created_at == "2024-01-01T00:00:00Z"
        assert key.description == "Test key"


class TestBackupEncryption:
    """Tests for BackupEncryption class."""

    @pytest.fixture
    def encryption_key(self):
        """Create a test encryption key."""
        from aragora.backup.encryption import BackupEncryption

        return BackupEncryption.generate_key("test-key-001", "Test encryption key")

    @pytest.fixture
    def encryptor(self, encryption_key):
        """Create a BackupEncryption instance."""
        from aragora.backup.encryption import BackupEncryption

        return BackupEncryption(encryption_key)

    def test_generate_key(self):
        """Should generate a valid random key."""
        from aragora.backup.encryption import BackupEncryption, KEY_SIZE

        key1 = BackupEncryption.generate_key("key-1")
        key2 = BackupEncryption.generate_key("key-2")

        assert len(key1.key) == KEY_SIZE
        assert len(key2.key) == KEY_SIZE
        assert key1.key != key2.key  # Keys should be unique
        assert key1.key_id == "key-1"
        assert key2.key_id == "key-2"

    def test_encrypt_decrypt_bytes(self, encryptor):
        """Should encrypt and decrypt bytes correctly."""
        original = b"Hello, World! This is test data for encryption."

        encrypted, metadata = encryptor.encrypt_bytes(original)

        assert encrypted != original
        assert len(encrypted) > len(original)  # Encrypted with header
        assert metadata.version == 1
        assert metadata.original_size == len(original)

        decrypted = encryptor.decrypt_bytes(encrypted)

        assert decrypted == original

    def test_encrypt_decrypt_empty_bytes(self, encryptor):
        """Should handle empty bytes."""
        original = b""

        encrypted, metadata = encryptor.encrypt_bytes(original)
        decrypted = encryptor.decrypt_bytes(encrypted)

        assert decrypted == original

    def test_encrypt_decrypt_large_bytes(self, encryptor):
        """Should handle large data."""
        original = b"x" * (1024 * 1024)  # 1MB

        encrypted, metadata = encryptor.encrypt_bytes(original)
        decrypted = encryptor.decrypt_bytes(encrypted)

        assert decrypted == original

    def test_decrypt_with_wrong_key_fails(self, encryption_key):
        """Decryption should fail with wrong key."""
        from aragora.backup.encryption import BackupEncryption

        encryptor1 = BackupEncryption(encryption_key)
        key2 = BackupEncryption.generate_key("different-key")
        encryptor2 = BackupEncryption(key2)

        original = b"Secret data"
        encrypted, _ = encryptor1.encrypt_bytes(original)

        with pytest.raises(ValueError, match="Key ID mismatch"):
            encryptor2.decrypt_bytes(encrypted)

    def test_decrypt_tampered_data_fails(self, encryptor):
        """Decryption should fail if data is tampered."""
        pytest.importorskip("cryptography")
        from cryptography.exceptions import InvalidTag

        original = b"Secret data"
        encrypted, _ = encryptor.encrypt_bytes(original)

        # Tamper with the ciphertext (last byte)
        tampered = bytearray(encrypted)
        tampered[-1] ^= 0xFF
        tampered = bytes(tampered)

        with pytest.raises(InvalidTag):
            encryptor.decrypt_bytes(tampered)

    def test_decrypt_invalid_magic_fails(self, encryptor):
        """Decryption should fail with invalid magic."""
        invalid_data = b"XXXX" + b"\x00" * 100

        with pytest.raises(ValueError, match="Invalid backup"):
            encryptor.decrypt_bytes(invalid_data)

    def test_decrypt_short_data_fails(self, encryptor):
        """Decryption should fail with data too short."""
        short_data = b"ABC"

        with pytest.raises(ValueError, match="too short"):
            encryptor.decrypt_bytes(short_data)


class TestBackupEncryptionFiles:
    """Tests for file encryption/decryption."""

    @pytest.fixture
    def encryption_key(self):
        """Create a test encryption key."""
        from aragora.backup.encryption import BackupEncryption

        return BackupEncryption.generate_key("test-file-key")

    @pytest.fixture
    def encryptor(self, encryption_key):
        """Create a BackupEncryption instance."""
        from aragora.backup.encryption import BackupEncryption

        return BackupEncryption(encryption_key)

    def test_encrypt_decrypt_file(self, encryptor):
        """Should encrypt and decrypt files correctly."""
        pytest.importorskip("cryptography")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create source file
            source = tmpdir / "source.txt"
            original_content = b"File content to encrypt" * 100
            source.write_bytes(original_content)

            # Encrypt
            encrypted = tmpdir / "encrypted.enc"
            meta = encryptor.encrypt_file(source, encrypted)

            assert encrypted.exists()
            assert meta.original_size == len(original_content)
            assert encrypted.read_bytes()[:4] == b"ABKP"  # Magic header

            # Decrypt
            decrypted = tmpdir / "decrypted.txt"
            encryptor.decrypt_file(encrypted, decrypted)

            assert decrypted.read_bytes() == original_content

    def test_encrypt_nonexistent_file_fails(self, encryptor):
        """Should raise error for nonexistent source file."""
        pytest.importorskip("cryptography")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source = tmpdir / "nonexistent.txt"
            encrypted = tmpdir / "encrypted.enc"

            with pytest.raises(FileNotFoundError):
                encryptor.encrypt_file(source, encrypted)


class TestKeyManager:
    """Tests for KeyManager class."""

    def test_generate_and_retrieve_key(self):
        """Should generate and retrieve keys."""
        from aragora.backup.encryption import KeyManager

        manager = KeyManager()

        key = manager.generate_key("my-key", "My test key")

        assert key.key_id == "my-key"
        assert key.description == "My test key"
        assert len(key.key) == 32

        retrieved = manager.get_key("my-key")
        assert retrieved == key

    def test_list_keys(self):
        """Should list all key IDs."""
        from aragora.backup.encryption import KeyManager

        manager = KeyManager()
        manager.generate_key("key-1")
        manager.generate_key("key-2")
        manager.generate_key("key-3")

        keys = manager.list_keys()

        assert set(keys) == {"key-1", "key-2", "key-3"}

    def test_delete_key(self):
        """Should delete keys."""
        from aragora.backup.encryption import KeyManager

        manager = KeyManager()
        manager.generate_key("key-to-delete")

        assert manager.get_key("key-to-delete") is not None
        assert manager.delete_key("key-to-delete") is True
        assert manager.get_key("key-to-delete") is None

    def test_delete_nonexistent_key(self):
        """Should return False for nonexistent key."""
        from aragora.backup.encryption import KeyManager

        manager = KeyManager()

        assert manager.delete_key("nonexistent") is False

    def test_rotate_key(self):
        """Should generate new key for rotation."""
        from aragora.backup.encryption import KeyManager

        manager = KeyManager()
        old_key = manager.generate_key("old-key")
        new_key = manager.rotate_key("old-key", "new-key")

        assert new_key.key_id == "new-key"
        assert new_key.key != old_key.key
        assert "Rotation of old-key" in (new_key.description or "")

        # Both keys should exist
        assert manager.get_key("old-key") is not None
        assert manager.get_key("new-key") is not None

    def test_rotate_nonexistent_key_fails(self):
        """Should fail to rotate nonexistent key."""
        from aragora.backup.encryption import KeyManager

        manager = KeyManager()

        with pytest.raises(ValueError, match="Key not found"):
            manager.rotate_key("nonexistent", "new-key")

    def test_master_key_validation(self):
        """Master key must be 32 bytes."""
        from aragora.backup.encryption import KeyManager

        manager = KeyManager()

        with pytest.raises(ValueError, match="32 bytes"):
            manager.set_master_key(b"short")


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_encrypt_decrypt_backup(self):
        """Should encrypt and decrypt using module functions."""
        pytest.importorskip("cryptography")
        from aragora.backup.encryption import (
            BackupEncryption,
            encrypt_backup,
            decrypt_backup,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create source
            source = tmpdir / "backup.db"
            original = b"Database content" * 1000
            source.write_bytes(original)

            # Generate key
            key = BackupEncryption.generate_key("backup-key")

            # Encrypt
            encrypted = tmpdir / "backup.enc"
            meta = encrypt_backup(source, encrypted, key)

            assert meta.original_size == len(original)

            # Decrypt
            restored = tmpdir / "restored.db"
            decrypt_backup(encrypted, restored, key)

            assert restored.read_bytes() == original


class TestEncryptionConstants:
    """Tests for encryption constants."""

    def test_constants_defined(self):
        """All constants should be defined."""
        from aragora.backup.encryption import (
            ENCRYPTION_VERSION,
            KEY_SIZE,
            NONCE_SIZE,
            TAG_SIZE,
            HEADER_MAGIC,
            HEADER_SIZE,
        )

        assert ENCRYPTION_VERSION == 1
        assert KEY_SIZE == 32
        assert NONCE_SIZE == 12
        assert TAG_SIZE == 16
        assert HEADER_MAGIC == b"ABKP"
        assert HEADER_SIZE == 4 + 1 + NONCE_SIZE + 32
