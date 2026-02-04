"""
OpenClaw Credential Vault - Backward-compatible re-export stub.

This module has been decomposed into the ``aragora.gateway.openclaw.credentials``
package. All public names are re-exported here for backward compatibility so that
existing imports continue to work unchanged.

Usage (unchanged):
    from aragora.gateway.openclaw.credential_vault import (
        CredentialVault,
        StoredCredential,
        RotationPolicy,
        get_credential_vault,
    )
"""

from aragora.gateway.openclaw.credentials import (  # noqa: F401
    # Main class
    CredentialVault,
    # Dataclasses
    StoredCredential,
    CredentialMetadata,
    RotationPolicy,
    # Enums
    CredentialType,
    CredentialFramework,
    CredentialAuditEvent,
    # Exceptions
    CredentialVaultError,
    CredentialNotFoundError,
    CredentialAccessDeniedError,
    CredentialExpiredError,
    CredentialRateLimitedError,
    TenantIsolationError,
    EncryptionError,
    # Utilities
    CredentialRateLimiter,
    # Factory functions
    get_credential_vault,
    reset_credential_vault,
    init_credential_vault,
    # Protocol (for type hints)
    KMSProviderProtocol,
    AuditLoggerProtocol,
    AuthorizationContextProtocol,
    # Crypto flag
    CRYPTO_AVAILABLE,
)

__all__ = [
    # Main class
    "CredentialVault",
    # Dataclasses
    "StoredCredential",
    "CredentialMetadata",
    "RotationPolicy",
    # Enums
    "CredentialType",
    "CredentialFramework",
    "CredentialAuditEvent",
    # Exceptions
    "CredentialVaultError",
    "CredentialNotFoundError",
    "CredentialAccessDeniedError",
    "CredentialExpiredError",
    "CredentialRateLimitedError",
    "TenantIsolationError",
    "EncryptionError",
    # Utilities
    "CredentialRateLimiter",
    # Factory functions
    "get_credential_vault",
    "reset_credential_vault",
    "init_credential_vault",
    # Protocol (for type hints)
    "KMSProviderProtocol",
    "AuditLoggerProtocol",
    "AuthorizationContextProtocol",
    # Crypto flag
    "CRYPTO_AVAILABLE",
]
