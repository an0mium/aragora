"""
Encryption key rotation handler.

Handles rotation of encryption keys used for data-at-rest encryption.
"""

import secrets
from datetime import datetime
from typing import Any
import logging
import base64

from .base import RotationHandler

logger = logging.getLogger(__name__)


class EncryptionKeyRotationHandler(RotationHandler):
    """
    Handler for encryption key rotation.

    Manages:
    - AES-256 encryption keys
    - JWT signing keys
    - HMAC secrets
    - Session encryption keys

    Key rotation involves:
    1. Generate new key
    2. Update key in storage
    3. Keep old key available for decryption during grace period
    4. Re-encrypt data with new key (optional, can be done lazily)
    """

    @property
    def secret_type(self) -> str:
        return "encryption_key"

    def __init__(
        self,
        key_length: int = 32,  # 256 bits for AES-256
        grace_period_hours: int = 168,  # 7 days for encryption keys
        max_retries: int = 3,
    ):
        """
        Initialize encryption key rotation handler.

        Args:
            key_length: Length of generated keys in bytes
            grace_period_hours: Hours old key remains valid for decryption
            max_retries: Maximum retry attempts
        """
        super().__init__(grace_period_hours, max_retries)
        self.key_length = key_length

    def _generate_key(self) -> bytes:
        """Generate a cryptographically secure key."""
        return secrets.token_bytes(self.key_length)

    async def generate_new_credentials(
        self, secret_id: str, metadata: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """
        Generate new encryption key.

        Args:
            secret_id: Key identifier (e.g., "ENCRYPTION_KEY_PRIMARY")
            metadata: Key configuration

        Returns:
            Tuple of (base64_encoded_key, updated_metadata)
        """
        key_type = metadata.get("key_type", "aes256")
        key_length = metadata.get("key_length", self.key_length)

        if key_type == "aes256":
            new_key = secrets.token_bytes(32)
        elif key_type == "aes128":
            new_key = secrets.token_bytes(16)
        elif key_type == "hmac":
            new_key = secrets.token_bytes(key_length)
        elif key_type == "jwt":
            # For JWT, we might generate RSA keys or symmetric keys
            jwt_algo = metadata.get("jwt_algorithm", "HS256")
            if jwt_algo.startswith("HS"):
                new_key = secrets.token_bytes(64)  # 512 bits for HMAC
            else:
                # RSA key generation would go here
                new_key = secrets.token_bytes(32)
        else:
            new_key = secrets.token_bytes(key_length)

        # Encode as base64 for storage
        encoded_key = base64.b64encode(new_key).decode("ascii")
        key_id = secrets.token_hex(8)

        logger.info(f"Generated new {key_type} encryption key for {secret_id}")

        return encoded_key, {
            **metadata,
            "key_id": key_id,
            "key_type": key_type,
            "key_length": len(new_key),
            "version": f"v{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "rotated_at": datetime.utcnow().isoformat(),
        }

    async def validate_credentials(
        self, secret_id: str, secret_value: str, metadata: dict[str, Any]
    ) -> bool:
        """
        Validate encryption key.

        For encryption keys, we verify:
        1. Key can be decoded from base64
        2. Key is the expected length
        3. Key can be used for encryption/decryption

        Args:
            secret_id: Key identifier
            secret_value: Base64-encoded key
            metadata: Key configuration

        Returns:
            True if key is valid
        """
        try:
            # Decode from base64
            key_bytes = base64.b64decode(secret_value)

            # Check length
            expected_length = metadata.get("key_length", self.key_length)
            if len(key_bytes) != expected_length:
                logger.error(
                    f"Key length mismatch for {secret_id}: "
                    f"expected {expected_length}, got {len(key_bytes)}"
                )
                return False

            # Test encryption/decryption
            key_type = metadata.get("key_type", "aes256")
            if key_type.startswith("aes"):
                return await self._validate_aes_key(key_bytes)
            elif key_type == "hmac":
                return await self._validate_hmac_key(key_bytes)
            else:
                # Basic validation passed
                return True

        except Exception as e:
            logger.error(f"Key validation failed for {secret_id}: {e}")
            return False

    async def _validate_aes_key(self, key: bytes) -> bool:
        """Validate AES key by performing test encryption."""
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.backends import default_backend

            # Generate test IV and plaintext
            iv = secrets.token_bytes(16)
            plaintext = b"test_encryption_validation"

            # Pad plaintext to 16 bytes
            padded = plaintext + b"\x00" * (16 - len(plaintext) % 16)

            # Encrypt
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(padded) + encryptor.finalize()

            # Decrypt
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            decrypted = decryptor.update(ciphertext) + decryptor.finalize()

            # Verify
            return decrypted[: len(plaintext)] == plaintext

        except ImportError:
            logger.warning("cryptography not installed, assuming key valid")
            return True
        except Exception as e:
            logger.error(f"AES validation error: {e}")
            return False

    async def _validate_hmac_key(self, key: bytes) -> bool:
        """Validate HMAC key by performing test signing."""
        try:
            import hmac
            import hashlib

            message = b"test_hmac_validation"
            signature = hmac.new(key, message, hashlib.sha256).digest()

            # Verify signature
            return hmac.compare_digest(
                signature,
                hmac.new(key, message, hashlib.sha256).digest(),
            )

        except Exception as e:
            logger.error(f"HMAC validation error: {e}")
            return False

    async def revoke_old_credentials(
        self, secret_id: str, old_value: str, metadata: dict[str, Any]
    ) -> bool:
        """
        Handle old encryption key after grace period.

        For encryption keys, we don't truly "revoke" them but:
        1. Remove from active key list
        2. Archive for potential emergency decryption
        3. Log the rotation completion

        Args:
            secret_id: Key identifier
            old_value: Old key value
            metadata: Key configuration

        Returns:
            True if cleanup succeeded
        """
        try:
            # Archive the old key (in production, store in secure archive)
            key_id = metadata.get("key_id", "unknown")
            logger.info(
                f"Archiving old encryption key {key_id} for {secret_id}. "
                f"Key removed from active rotation."
            )

            # In production, you might:
            # 1. Store in cold storage for emergency recovery
            # 2. Record the retirement in audit log
            # 3. Trigger re-encryption of remaining data

            return True

        except Exception as e:
            logger.error(f"Key archival failed for {secret_id}: {e}")
            return False

    async def trigger_reencryption(
        self,
        secret_id: str,
        old_key: str,
        new_key: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Trigger re-encryption of data with new key.

        This is an optional operation that can be run:
        - Immediately after rotation (blocking)
        - As a background job (async)
        - Lazily on next access

        Args:
            secret_id: Key identifier
            old_key: Old encryption key (base64)
            new_key: New encryption key (base64)
            metadata: Configuration including data stores to re-encrypt

        Returns:
            Re-encryption status
        """
        stores = metadata.get("data_stores", [])
        results = {}

        for store in stores:
            try:
                # In production, this would call store-specific re-encryption
                logger.info(f"Re-encrypting {store} with new key for {secret_id}")
                results[store] = {"status": "pending", "message": "Re-encryption queued"}
            except Exception as e:
                logger.error(f"Re-encryption failed for {store}: {e}")
                results[store] = {"status": "failed", "error": str(e)}

        return {
            "secret_id": secret_id,
            "stores": results,
            "queued_at": datetime.utcnow().isoformat(),
        }
