"""
Application-Level Encryption.

Provides AES-256-GCM encryption for sensitive data at the application layer,
independent of database or storage-level encryption.

Features:
- AES-256-GCM authenticated encryption
- Key derivation from passwords/secrets
- Key rotation support with version tracking
- Envelope encryption for large data
- Field-level encryption for sensitive attributes

Usage:
    from aragora.security.encryption import EncryptionService, get_encryption_service

    # Using the service
    service = get_encryption_service()

    # Encrypt sensitive data
    encrypted = service.encrypt("sensitive data")
    decrypted = service.decrypt(encrypted)

    # Encrypt with associated data (for integrity)
    encrypted = service.encrypt("data", associated_data="user_123")

    # Field-level encryption
    encrypted_record = service.encrypt_fields(
        {"name": "John", "ssn": "123-45-6789"},
        sensitive_fields=["ssn"]
    )
"""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets
import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

# Try to import cryptography library
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography library not available - encryption disabled")


# Encryption enforcement configuration
# When True, encryption failures raise exceptions instead of returning plaintext
ENCRYPTION_REQUIRED = os.environ.get("ARAGORA_ENCRYPTION_REQUIRED", "").lower() in (
    "true",
    "1",
    "yes",
)


class EncryptionError(Exception):
    """Raised when encryption/decryption fails and ENCRYPTION_REQUIRED is True."""

    def __init__(self, operation: str, reason: str, store: str = ""):
        self.operation = operation
        self.reason = reason
        self.store = store
        super().__init__(
            f"Encryption {operation} failed for {store or 'unknown'}: {reason}. "
            f"Set ARAGORA_ENCRYPTION_REQUIRED=false to allow plaintext fallback."
        )


def is_encryption_required() -> bool:
    """Check if encryption is required (fail-fast mode).

    When True, stores must raise EncryptionError instead of falling back
    to plaintext storage.
    """
    # SECURITY: Auto-require encryption in production mode
    if ENCRYPTION_REQUIRED or os.environ.get("ARAGORA_ENV") == "production":
        return True
    return False


class EncryptionAlgorithm(str, Enum):
    """Supported encryption algorithms."""

    AES_256_GCM = "aes-256-gcm"


class KeyDerivationFunction(str, Enum):
    """Key derivation functions."""

    PBKDF2_SHA256 = "pbkdf2-sha256"
    HKDF_SHA256 = "hkdf-sha256"


@dataclass
class EncryptionKey:
    """An encryption key with metadata."""

    key_id: str
    key_bytes: bytes
    algorithm: EncryptionAlgorithm
    version: int
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True

    @property
    def is_expired(self) -> bool:
        """Check if the key is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (without key material)."""
        return {
            "key_id": self.key_id,
            "algorithm": self.algorithm.value,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
            "is_expired": self.is_expired,
        }


@dataclass
class EncryptedData:
    """Container for encrypted data with metadata."""

    ciphertext: bytes
    nonce: bytes
    key_id: str
    key_version: int
    algorithm: EncryptionAlgorithm
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_bytes(self) -> bytes:
        """Serialize to bytes for storage."""
        # Format: version (1) + key_id_len (1) + key_id + key_version (4) +
        #         nonce_len (1) + nonce + ciphertext
        key_id_bytes = self.key_id.encode("utf-8")

        return b"".join(
            [
                struct.pack("B", 1),  # Format version
                struct.pack("B", len(key_id_bytes)),
                key_id_bytes,
                struct.pack(">I", self.key_version),
                struct.pack("B", len(self.nonce)),
                self.nonce,
                self.ciphertext,
            ]
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "EncryptedData":
        """Deserialize from bytes."""
        offset = 0

        # Format version
        version = struct.unpack_from("B", data, offset)[0]
        offset += 1

        if version != 1:
            raise ValueError(f"Unsupported encryption format version: {version}")

        # Key ID
        key_id_len = struct.unpack_from("B", data, offset)[0]
        offset += 1
        key_id = data[offset : offset + key_id_len].decode("utf-8")
        offset += key_id_len

        # Key version
        key_version = struct.unpack_from(">I", data, offset)[0]
        offset += 4

        # Nonce
        nonce_len = struct.unpack_from("B", data, offset)[0]
        offset += 1
        nonce = data[offset : offset + nonce_len]
        offset += nonce_len

        # Ciphertext
        ciphertext = data[offset:]

        return cls(
            ciphertext=ciphertext,
            nonce=nonce,
            key_id=key_id,
            key_version=key_version,
            algorithm=EncryptionAlgorithm.AES_256_GCM,
        )

    def to_base64(self) -> str:
        """Serialize to base64 string."""
        return base64.b64encode(self.to_bytes()).decode("ascii")

    @classmethod
    def from_base64(cls, data: str) -> "EncryptedData":
        """Deserialize from base64 string."""
        return cls.from_bytes(base64.b64decode(data))


@dataclass
class EncryptionConfig:
    """Configuration for encryption service."""

    # Key settings
    key_size_bits: int = 256
    default_key_ttl_days: int = 90

    # Algorithm settings
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    nonce_size_bytes: int = 12  # 96 bits for GCM

    # Key derivation
    kdf: KeyDerivationFunction = KeyDerivationFunction.PBKDF2_SHA256
    kdf_iterations: int = 100000
    kdf_salt_size: int = 16

    # Rotation
    auto_rotate: bool = True
    rotation_overlap_days: int = 7  # Keep old keys active for overlap


class EncryptionService:
    """
    Application-level encryption service.

    Provides authenticated encryption (AES-256-GCM) for sensitive data
    with key rotation and version tracking.
    """

    def __init__(
        self,
        config: Optional[EncryptionConfig] = None,
        master_key: Optional[bytes] = None,
    ):
        """
        Initialize encryption service.

        Args:
            config: Encryption configuration
            master_key: Master encryption key (32 bytes for AES-256)
        """
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library required for encryption")

        self.config = config or EncryptionConfig()
        self._keys: dict[str, EncryptionKey] = {}
        self._active_key_id: Optional[str] = None

        # Initialize with master key if provided
        if master_key:
            self._init_with_master_key(master_key)

    def _init_with_master_key(self, master_key: bytes) -> None:
        """Initialize with a master key."""
        if len(master_key) != 32:
            raise ValueError("Master key must be 32 bytes (256 bits)")

        key = EncryptionKey(
            key_id="master",
            key_bytes=master_key,
            algorithm=self.config.algorithm,
            version=1,
            created_at=datetime.utcnow(),
            is_active=True,
        )
        self._keys["master"] = key
        self._active_key_id = "master"

    def generate_key(
        self,
        key_id: Optional[str] = None,
        ttl_days: Optional[int] = None,
    ) -> EncryptionKey:
        """
        Generate a new encryption key.

        Args:
            key_id: Optional key identifier (auto-generated if not provided)
            ttl_days: Key lifetime in days (uses default if not provided)

        Returns:
            The generated EncryptionKey
        """
        if key_id is None:
            key_id = f"key_{secrets.token_hex(8)}"

        key_bytes = secrets.token_bytes(self.config.key_size_bits // 8)

        # Determine version (increment if key_id exists)
        version = 1
        if key_id in self._keys:
            version = self._keys[key_id].version + 1

        ttl = ttl_days or self.config.default_key_ttl_days
        expires_at = datetime.utcnow() + timedelta(days=ttl) if ttl > 0 else None

        key = EncryptionKey(
            key_id=key_id,
            key_bytes=key_bytes,
            algorithm=self.config.algorithm,
            version=version,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            is_active=True,
        )

        self._keys[key_id] = key
        self._active_key_id = key_id

        logger.info(f"Generated encryption key: {key_id} (version {version})")

        return key

    def derive_key_from_password(
        self,
        password: str,
        salt: Optional[bytes] = None,
        key_id: str = "derived",
    ) -> tuple[EncryptionKey, bytes]:
        """
        Derive an encryption key from a password.

        Args:
            password: The password to derive from
            salt: Optional salt (generated if not provided)
            key_id: Key identifier

        Returns:
            Tuple of (EncryptionKey, salt)
        """
        if salt is None:
            salt = secrets.token_bytes(self.config.kdf_salt_size)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=self.config.key_size_bits // 8,
            salt=salt,
            iterations=self.config.kdf_iterations,
            backend=default_backend(),
        )

        key_bytes = kdf.derive(password.encode("utf-8"))

        key = EncryptionKey(
            key_id=key_id,
            key_bytes=key_bytes,
            algorithm=self.config.algorithm,
            version=1,
            created_at=datetime.utcnow(),
            is_active=True,
        )

        self._keys[key_id] = key
        self._active_key_id = key_id

        return key, salt

    def encrypt(
        self,
        plaintext: Union[str, bytes],
        associated_data: Optional[Union[str, bytes]] = None,
        key_id: Optional[str] = None,
    ) -> EncryptedData:
        """
        Encrypt data using AES-256-GCM.

        Args:
            plaintext: Data to encrypt (string or bytes)
            associated_data: Additional authenticated data (not encrypted, but authenticated)
            key_id: Key to use (uses active key if not specified)

        Returns:
            EncryptedData container
        """
        key = self._get_key(key_id)

        if isinstance(plaintext, str):
            plaintext = plaintext.encode("utf-8")

        if isinstance(associated_data, str):
            associated_data = associated_data.encode("utf-8")

        # Generate nonce
        nonce = secrets.token_bytes(self.config.nonce_size_bytes)

        # Encrypt
        aesgcm = AESGCM(key.key_bytes)
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)

        return EncryptedData(
            ciphertext=ciphertext,
            nonce=nonce,
            key_id=key.key_id,
            key_version=key.version,
            algorithm=key.algorithm,
        )

    def decrypt(
        self,
        encrypted: Union[EncryptedData, str, bytes],
        associated_data: Optional[Union[str, bytes]] = None,
    ) -> bytes:
        """
        Decrypt data.

        Args:
            encrypted: EncryptedData, base64 string, or raw bytes
            associated_data: Additional authenticated data (must match encryption)

        Returns:
            Decrypted bytes
        """
        # Parse encrypted data if needed
        if isinstance(encrypted, str):
            encrypted = EncryptedData.from_base64(encrypted)
        elif isinstance(encrypted, bytes):
            encrypted = EncryptedData.from_bytes(encrypted)

        # Get the key with the correct version for decryption
        key = self._get_key(encrypted.key_id, version=encrypted.key_version)

        if key.version != encrypted.key_version:
            # This should not happen after the fix, but log if it does
            logger.warning(
                f"Key version mismatch: expected {encrypted.key_version}, have {key.version}"
            )

        if isinstance(associated_data, str):
            associated_data = associated_data.encode("utf-8")

        # Decrypt
        aesgcm = AESGCM(key.key_bytes)
        plaintext = aesgcm.decrypt(encrypted.nonce, encrypted.ciphertext, associated_data)

        return plaintext

    def decrypt_string(
        self,
        encrypted: Union[EncryptedData, str, bytes],
        associated_data: Optional[Union[str, bytes]] = None,
    ) -> str:
        """Decrypt and return as string."""
        return self.decrypt(encrypted, associated_data).decode("utf-8")

    def encrypt_fields(
        self,
        record: dict[str, Any],
        sensitive_fields: list[str],
        associated_data: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Encrypt specific fields in a record.

        Args:
            record: Dictionary with data
            sensitive_fields: List of field names to encrypt
            associated_data: AAD for integrity (e.g., record ID)

        Returns:
            Record with specified fields encrypted (as base64 strings)
        """
        result = record.copy()

        for field_name in sensitive_fields:
            if field_name in result and result[field_name] is not None:
                value = result[field_name]
                if not isinstance(value, (str, bytes)):
                    value = json.dumps(value)

                encrypted = self.encrypt(value, associated_data)
                result[field_name] = {
                    "_encrypted": True,
                    "_value": encrypted.to_base64(),
                }

        return result

    def decrypt_fields(
        self,
        record: dict[str, Any],
        sensitive_fields: list[str],
        associated_data: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Decrypt specific fields in a record.

        Args:
            record: Dictionary with encrypted fields
            sensitive_fields: List of field names to decrypt
            associated_data: AAD used during encryption

        Returns:
            Record with specified fields decrypted
        """
        result = record.copy()

        for field_name in sensitive_fields:
            if field_name in result:
                value = result[field_name]
                if isinstance(value, dict) and value.get("_encrypted"):
                    encrypted_value = value["_value"]
                    decrypted = self.decrypt_string(encrypted_value, associated_data)

                    # Try to parse as JSON
                    try:
                        result[field_name] = json.loads(decrypted)
                    except json.JSONDecodeError:
                        result[field_name] = decrypted

        return result

    def rotate_key(self, key_id: Optional[str] = None) -> EncryptionKey:
        """
        Rotate an encryption key.

        Creates a new version of the key. Old versions remain available
        for decryption during the overlap period.

        Args:
            key_id: Key to rotate (uses active key if not specified)

        Returns:
            The new EncryptionKey
        """
        key_id = key_id or self._active_key_id
        if key_id is None:
            raise ValueError("No key to rotate")

        old_key = self._keys.get(key_id)
        if old_key:
            # Keep old key for overlap period
            if self.config.rotation_overlap_days > 0:
                old_key.expires_at = datetime.utcnow() + timedelta(
                    days=self.config.rotation_overlap_days
                )
            old_key.is_active = False

            # Store old version with versioned key_id
            versioned_id = f"{key_id}_v{old_key.version}"
            self._keys[versioned_id] = old_key

        # Generate new key
        return self.generate_key(key_id)

    def re_encrypt(
        self,
        encrypted: Union[EncryptedData, str, bytes],
        associated_data: Optional[Union[str, bytes]] = None,
        new_key_id: Optional[str] = None,
    ) -> EncryptedData:
        """
        Re-encrypt data with a new key.

        Useful for key rotation - decrypt with old key, encrypt with new.

        Args:
            encrypted: Currently encrypted data
            associated_data: AAD (must be same for decrypt and encrypt)
            new_key_id: Key to encrypt with (uses active key if not specified)

        Returns:
            Newly encrypted data
        """
        plaintext = self.decrypt(encrypted, associated_data)
        return self.encrypt(plaintext, associated_data, new_key_id)

    def _get_key(
        self, key_id: Optional[str] = None, version: Optional[int] = None
    ) -> EncryptionKey:
        """Get a key by ID or the active key."""
        if key_id is None:
            key_id = self._active_key_id

        if key_id is None:
            raise ValueError("No encryption key available")

        # If a specific version is requested, try the versioned key ID first
        if version is not None:
            versioned_id = f"{key_id}_v{version}"
            versioned_key = self._keys.get(versioned_id)
            if versioned_key is not None:
                if versioned_key.is_expired:
                    raise ValueError(f"Key expired: {versioned_id}")
                return versioned_key

        key = self._keys.get(key_id)
        if key is None:
            raise ValueError(f"Key not found: {key_id}")

        if key.is_expired:
            raise ValueError(f"Key expired: {key_id}")

        return key

    def list_keys(self) -> list[dict[str, Any]]:
        """List all keys (without key material)."""
        return [key.to_dict() for key in self._keys.values()]

    def get_active_key_id(self) -> Optional[str]:
        """Get the active key ID."""
        return self._active_key_id


# Singleton service instance
_encryption_service: Optional[EncryptionService] = None


def get_encryption_service() -> EncryptionService:
    """Get the global encryption service instance."""
    global _encryption_service

    if _encryption_service is None:
        # Try to initialize from secrets manager or environment
        from aragora.config.secrets import get_secret

        master_key_hex = get_secret("ARAGORA_ENCRYPTION_KEY")
        if master_key_hex:
            master_key = bytes.fromhex(master_key_hex)
            _encryption_service = EncryptionService(master_key=master_key)
        else:
            # Create service and generate a new key
            _encryption_service = EncryptionService()
            _encryption_service.generate_key("default")
            logger.warning(
                "No ARAGORA_ENCRYPTION_KEY set - generated ephemeral key. "
                "Data will not be recoverable after restart."
            )

    return _encryption_service


def init_encryption_service(
    master_key: Optional[bytes] = None,
    config: Optional[EncryptionConfig] = None,
) -> EncryptionService:
    """Initialize the global encryption service."""
    global _encryption_service
    _encryption_service = EncryptionService(config=config, master_key=master_key)
    return _encryption_service


__all__ = [
    "EncryptionService",
    "EncryptionConfig",
    "EncryptionKey",
    "EncryptedData",
    "EncryptionAlgorithm",
    "KeyDerivationFunction",
    "EncryptionError",
    "get_encryption_service",
    "init_encryption_service",
    "is_encryption_required",
    "CRYPTO_AVAILABLE",
    "ENCRYPTION_REQUIRED",
]
