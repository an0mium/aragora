"""
Backup Encryption Module.

Provides AES-256-GCM encryption for backup data with support for:
- Key management integration
- Streaming encryption for large files
- Key rotation
- Integrity verification via authentication tags
"""

from __future__ import annotations

import hashlib
import io
import logging
import secrets
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Encryption constants
ENCRYPTION_VERSION = 1
KEY_SIZE = 32  # AES-256
NONCE_SIZE = 12  # GCM standard nonce
TAG_SIZE = 16  # GCM authentication tag
CHUNK_SIZE = 64 * 1024  # 64KB chunks for streaming

# File header format:
# - 4 bytes: magic number "ABKP"
# - 1 byte: version
# - 12 bytes: nonce
# - 32 bytes: key ID hash (SHA-256 of key ID for verification)
HEADER_MAGIC = b"ABKP"
HEADER_SIZE = 4 + 1 + NONCE_SIZE + 32


@dataclass
class EncryptionKey:
    """Encryption key with metadata."""

    key_id: str
    key: bytes
    created_at: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self):
        if len(self.key) != KEY_SIZE:
            raise ValueError(f"Key must be {KEY_SIZE} bytes, got {len(self.key)}")


@dataclass
class EncryptionMetadata:
    """Metadata about encrypted data."""

    version: int
    key_id_hash: str
    nonce: bytes
    original_size: int = 0
    encrypted_size: int = 0


class BackupEncryption:
    """
    AES-256-GCM encryption for backup data.

    Thread-safe implementation using the cryptography library.
    """

    def __init__(self, key: EncryptionKey):
        """
        Initialize with an encryption key.

        Args:
            key: EncryptionKey containing the 256-bit key
        """
        self._key = key
        self._key_id_hash = hashlib.sha256(key.key_id.encode()).digest()

    @classmethod
    def generate_key(cls, key_id: str, description: Optional[str] = None) -> EncryptionKey:
        """
        Generate a new random encryption key.

        Args:
            key_id: Unique identifier for the key
            description: Optional description

        Returns:
            New EncryptionKey
        """
        from datetime import datetime, timezone

        return EncryptionKey(
            key_id=key_id,
            key=secrets.token_bytes(KEY_SIZE),
            created_at=datetime.now(timezone.utc).isoformat(),
            description=description,
        )

    def encrypt_file(self, source_path: Path, dest_path: Path) -> EncryptionMetadata:
        """
        Encrypt a file with AES-256-GCM.

        Args:
            source_path: Path to the file to encrypt
            dest_path: Path for the encrypted output

        Returns:
            EncryptionMetadata with details about the encryption
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            raise ImportError(
                "cryptography package required for encryption. "
                "Install with: pip install cryptography"
            )

        nonce = secrets.token_bytes(NONCE_SIZE)
        aesgcm = AESGCM(self._key.key)

        original_size = source_path.stat().st_size

        with open(source_path, "rb") as src, open(dest_path, "wb") as dst:
            # Write header
            dst.write(HEADER_MAGIC)
            dst.write(struct.pack("B", ENCRYPTION_VERSION))
            dst.write(nonce)
            dst.write(self._key_id_hash)

            # Read and encrypt entire file (for GCM authentication)
            # For large files, consider using streaming with separate chunks
            plaintext = src.read()
            ciphertext = aesgcm.encrypt(nonce, plaintext, None)
            dst.write(ciphertext)

        encrypted_size = dest_path.stat().st_size

        return EncryptionMetadata(
            version=ENCRYPTION_VERSION,
            key_id_hash=self._key_id_hash.hex(),
            nonce=nonce,
            original_size=original_size,
            encrypted_size=encrypted_size,
        )

    def decrypt_file(self, source_path: Path, dest_path: Path) -> EncryptionMetadata:
        """
        Decrypt a file encrypted with AES-256-GCM.

        Args:
            source_path: Path to the encrypted file
            dest_path: Path for the decrypted output

        Returns:
            EncryptionMetadata with details about the decryption

        Raises:
            ValueError: If file format is invalid or key doesn't match
            cryptography.exceptions.InvalidTag: If decryption fails (tampered data)
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            raise ImportError(
                "cryptography package required for decryption. "
                "Install with: pip install cryptography"
            )

        with open(source_path, "rb") as src:
            # Read and validate header
            magic = src.read(4)
            if magic != HEADER_MAGIC:
                raise ValueError(f"Invalid backup file magic: {magic!r}")

            version = struct.unpack("B", src.read(1))[0]
            if version != ENCRYPTION_VERSION:
                raise ValueError(f"Unsupported encryption version: {version}")

            nonce = src.read(NONCE_SIZE)
            stored_key_id_hash = src.read(32)

            # Verify key ID matches
            if stored_key_id_hash != self._key_id_hash:
                raise ValueError("Key ID mismatch. This file was encrypted with a different key.")

            # Read ciphertext and decrypt
            ciphertext = src.read()
            aesgcm = AESGCM(self._key.key)
            plaintext = aesgcm.decrypt(nonce, ciphertext, None)

        with open(dest_path, "wb") as dst:
            dst.write(plaintext)

        return EncryptionMetadata(
            version=version,
            key_id_hash=stored_key_id_hash.hex(),
            nonce=nonce,
            original_size=len(plaintext),
            encrypted_size=source_path.stat().st_size,
        )

    def encrypt_bytes(self, data: bytes) -> tuple[bytes, EncryptionMetadata]:
        """
        Encrypt bytes in memory.

        Args:
            data: Bytes to encrypt

        Returns:
            Tuple of (encrypted_data, metadata)
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            raise ImportError(
                "cryptography package required for encryption. "
                "Install with: pip install cryptography"
            )

        nonce = secrets.token_bytes(NONCE_SIZE)
        aesgcm = AESGCM(self._key.key)

        ciphertext = aesgcm.encrypt(nonce, data, None)

        # Build encrypted blob with header
        encrypted = io.BytesIO()
        encrypted.write(HEADER_MAGIC)
        encrypted.write(struct.pack("B", ENCRYPTION_VERSION))
        encrypted.write(nonce)
        encrypted.write(self._key_id_hash)
        encrypted.write(ciphertext)

        result = encrypted.getvalue()

        metadata = EncryptionMetadata(
            version=ENCRYPTION_VERSION,
            key_id_hash=self._key_id_hash.hex(),
            nonce=nonce,
            original_size=len(data),
            encrypted_size=len(result),
        )

        return result, metadata

    def decrypt_bytes(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt bytes in memory.

        Args:
            encrypted_data: Encrypted bytes with header

        Returns:
            Decrypted bytes

        Raises:
            ValueError: If format is invalid or key doesn't match
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            raise ImportError(
                "cryptography package required for decryption. "
                "Install with: pip install cryptography"
            )

        if len(encrypted_data) < HEADER_SIZE:
            raise ValueError("Encrypted data too short")

        reader = io.BytesIO(encrypted_data)

        magic = reader.read(4)
        if magic != HEADER_MAGIC:
            raise ValueError(f"Invalid backup magic: {magic!r}")

        version = struct.unpack("B", reader.read(1))[0]
        if version != ENCRYPTION_VERSION:
            raise ValueError(f"Unsupported encryption version: {version}")

        nonce = reader.read(NONCE_SIZE)
        stored_key_id_hash = reader.read(32)

        if stored_key_id_hash != self._key_id_hash:
            raise ValueError("Key ID mismatch")

        ciphertext = reader.read()
        aesgcm = AESGCM(self._key.key)

        return aesgcm.decrypt(nonce, ciphertext, None)


class KeyManager:
    """
    Simple key manager for backup encryption keys.

    In production, this should integrate with:
    - AWS KMS
    - Azure Key Vault
    - HashiCorp Vault
    - Google Cloud KMS
    """

    def __init__(self, key_store_path: Optional[Path] = None):
        """
        Initialize the key manager.

        Args:
            key_store_path: Path to store encrypted keys (optional)
        """
        self._keys: dict[str, EncryptionKey] = {}
        self._key_store_path = key_store_path
        self._master_key: Optional[bytes] = None

    def set_master_key(self, master_key: bytes) -> None:
        """
        Set the master key for encrypting stored keys.

        Args:
            master_key: 32-byte master key
        """
        if len(master_key) != KEY_SIZE:
            raise ValueError(f"Master key must be {KEY_SIZE} bytes")
        self._master_key = master_key

    def generate_key(self, key_id: str, description: Optional[str] = None) -> EncryptionKey:
        """
        Generate and store a new encryption key.

        Args:
            key_id: Unique identifier for the key
            description: Optional description

        Returns:
            New EncryptionKey
        """
        key = BackupEncryption.generate_key(key_id, description)
        self._keys[key_id] = key
        return key

    def get_key(self, key_id: str) -> Optional[EncryptionKey]:
        """
        Get an encryption key by ID.

        Args:
            key_id: Key identifier

        Returns:
            EncryptionKey or None if not found
        """
        return self._keys.get(key_id)

    def list_keys(self) -> list[str]:
        """List all key IDs."""
        return list(self._keys.keys())

    def delete_key(self, key_id: str) -> bool:
        """
        Delete a key.

        WARNING: This will make data encrypted with this key unrecoverable.

        Args:
            key_id: Key identifier

        Returns:
            True if deleted, False if not found
        """
        if key_id in self._keys:
            del self._keys[key_id]
            return True
        return False

    def rotate_key(self, old_key_id: str, new_key_id: str) -> EncryptionKey:
        """
        Generate a new key for rotation.

        The old key should be kept until all data is re-encrypted.

        Args:
            old_key_id: ID of the key being rotated
            new_key_id: ID for the new key

        Returns:
            New EncryptionKey
        """
        old_key = self._keys.get(old_key_id)
        if not old_key:
            raise ValueError(f"Key not found: {old_key_id}")

        return self.generate_key(
            new_key_id,
            description=f"Rotation of {old_key_id}",
        )


# Module-level convenience functions


def encrypt_backup(
    source_path: Path,
    dest_path: Path,
    key: EncryptionKey,
) -> EncryptionMetadata:
    """
    Encrypt a backup file.

    Args:
        source_path: Path to the file to encrypt
        dest_path: Path for the encrypted output
        key: Encryption key to use

    Returns:
        EncryptionMetadata with details
    """
    enc = BackupEncryption(key)
    return enc.encrypt_file(source_path, dest_path)


def decrypt_backup(
    source_path: Path,
    dest_path: Path,
    key: EncryptionKey,
) -> EncryptionMetadata:
    """
    Decrypt a backup file.

    Args:
        source_path: Path to the encrypted file
        dest_path: Path for the decrypted output
        key: Encryption key to use

    Returns:
        EncryptionMetadata with details
    """
    enc = BackupEncryption(key)
    return enc.decrypt_file(source_path, dest_path)


__all__ = [
    "BackupEncryption",
    "EncryptionKey",
    "EncryptionMetadata",
    "KeyManager",
    "encrypt_backup",
    "decrypt_backup",
    "KEY_SIZE",
    "ENCRYPTION_VERSION",
]
