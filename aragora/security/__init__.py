"""
Aragora Security Module.

Provides enterprise-grade security features including:
- Application-level encryption (AES-256-GCM)
- Key management and rotation
- Field-level encryption for sensitive data
- Cloud KMS integration (AWS, Azure, GCP)

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

Cloud KMS Usage:
    from aragora.security import get_kms_provider

    # Auto-detect cloud provider
    kms = get_kms_provider()

    # Get encryption key from KMS
    key = await kms.get_encryption_key("aragora-master-key")
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

from .kms_provider import (
    KmsProvider,
    KmsKeyMetadata,
    AwsKmsProvider,
    AzureKeyVaultProvider,
    GcpKmsProvider,
    LocalKmsProvider,
    get_kms_provider,
    init_kms_provider,
    reset_kms_provider,
    detect_cloud_provider,
)

__all__ = [
    # Encryption
    "EncryptionService",
    "EncryptionConfig",
    "EncryptionKey",
    "EncryptedData",
    "EncryptionAlgorithm",
    "KeyDerivationFunction",
    "get_encryption_service",
    "init_encryption_service",
    "CRYPTO_AVAILABLE",
    # KMS Providers
    "KmsProvider",
    "KmsKeyMetadata",
    "AwsKmsProvider",
    "AzureKeyVaultProvider",
    "GcpKmsProvider",
    "LocalKmsProvider",
    "get_kms_provider",
    "init_kms_provider",
    "reset_kms_provider",
    "detect_cloud_provider",
]
