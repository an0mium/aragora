"""
OpenClaw Credential Vault - Secure credential storage for external frameworks.

Provides enterprise-grade secure credential management for external AI framework
integrations (OpenClaw, LangChain, CrewAI, etc.):
- AES-256-GCM encryption at rest
- Multiple KMS backend support (local, AWS, Azure, GCP, HashiCorp Vault)
- Per-tenant credential isolation with tenant-scoped encryption keys
- Automatic credential rotation with configurable policies
- RBAC-based access control with full audit logging
- Rate limiting for credential retrieval

Security Model:
1. Credentials encrypted at rest using AES-256-GCM
2. Each tenant has isolated encryption keys
3. KMS backends manage master keys securely
4. All credential access logged with HMAC-signed audit events
5. Rate limiting prevents credential harvesting attacks
6. Expiration tracking ensures timely rotation

Usage:
    from aragora.gateway.openclaw.credentials import (
        CredentialVault,
        StoredCredential,
        RotationPolicy,
        get_credential_vault,
    )

    # Get the global vault instance
    vault = get_credential_vault()

    # Store a credential
    cred_id = await vault.store_credential(
        tenant_id="acme-corp",
        framework="openai",
        credential_type="api_key",
        value="sk-...",
        rotation_policy=RotationPolicy(interval_days=90),
        auth_context=ctx,
    )

    # Retrieve credential (RBAC enforced)
    credential = await vault.get_credential(cred_id, auth_context=ctx)

    # List credentials for a tenant
    creds = await vault.list_credentials(
        tenant_id="acme-corp",
        framework="openai",
        auth_context=ctx,
    )

    # Rotate credential
    await vault.rotate_credential(cred_id, new_value="sk-new...", auth_context=ctx)
"""

from __future__ import annotations

import logging
import threading

from .enums import CredentialAuditEvent, CredentialFramework, CredentialType
from .exceptions import (
    CredentialAccessDeniedError,
    CredentialExpiredError,
    CredentialNotFoundError,
    CredentialRateLimitedError,
    CredentialVaultError,
    EncryptionError,
    TenantIsolationError,
)
from .models import CredentialMetadata, RotationPolicy, StoredCredential
from .protocols import (
    AuditLoggerProtocol,
    AuthorizationContextProtocol,
    KMSProviderProtocol,
)
from .rate_limiter import CredentialRateLimiter
from .vault import CRYPTO_AVAILABLE, CredentialVault

logger = logging.getLogger(__name__)

# =============================================================================
# Singleton and Factory
# =============================================================================

_credential_vault: CredentialVault | None = None
_vault_lock = threading.Lock()


def get_credential_vault(
    kms_provider: KMSProviderProtocol | None = None,
    audit_logger: AuditLoggerProtocol | None = None,
) -> CredentialVault:
    """
    Get or create the global credential vault instance.

    Args:
        kms_provider: Optional KMS provider (uses auto-detected if None)
        audit_logger: Optional audit logger

    Returns:
        CredentialVault singleton instance
    """
    global _credential_vault

    if _credential_vault is None:
        with _vault_lock:
            if _credential_vault is None:
                _credential_vault = CredentialVault(
                    kms_provider=kms_provider,
                    audit_logger=audit_logger,
                )
                logger.info("Initialized OpenClaw credential vault")

    return _credential_vault


def reset_credential_vault() -> None:
    """Reset the global credential vault (for testing)."""
    global _credential_vault
    with _vault_lock:
        _credential_vault = None


def init_credential_vault(
    kms_provider: KMSProviderProtocol | None = None,
    audit_logger: AuditLoggerProtocol | None = None,
    rate_limiter: CredentialRateLimiter | None = None,
) -> CredentialVault:
    """
    Initialize a new credential vault with explicit configuration.

    Args:
        kms_provider: KMS provider for key management
        audit_logger: Audit logger
        rate_limiter: Rate limiter configuration

    Returns:
        Configured CredentialVault instance
    """
    global _credential_vault
    with _vault_lock:
        _credential_vault = CredentialVault(
            kms_provider=kms_provider,
            audit_logger=audit_logger,
            rate_limiter=rate_limiter,
        )
    return _credential_vault


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
