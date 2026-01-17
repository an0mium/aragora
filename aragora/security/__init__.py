"""
Aragora Security Module.

Provides enterprise-grade security features including:
- Application-level encryption (AES-256-GCM)
- Key management and rotation
- Field-level encryption for sensitive data

Usage:
    from aragora.security import EncryptionService, get_encryption_service

    # Get the service
    service = get_encryption_service()

    # Encrypt data
    encrypted = service.encrypt("sensitive data")
    decrypted = service.decrypt_string(encrypted)

    # Field-level encryption
    record = {"name": "John", "ssn": "123-45-6789"}
    encrypted_record = service.encrypt_fields(record, ["ssn"])
"""

from .encryption import (
    EncryptionService,
    EncryptionConfig,
    EncryptionKey,
    EncryptedData,
    EncryptionAlgorithm,
    KeyDerivationFunction,
    get_encryption_service,
    init_encryption_service,
    CRYPTO_AVAILABLE,
)

__all__ = [
    "EncryptionService",
    "EncryptionConfig",
    "EncryptionKey",
    "EncryptedData",
    "EncryptionAlgorithm",
    "KeyDerivationFunction",
    "get_encryption_service",
    "init_encryption_service",
    "CRYPTO_AVAILABLE",
]
