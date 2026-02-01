"""
Credential Vault - Secure credential storage and runtime injection.

Provides secure credential management for external agents:
- Credentials are never exposed to external agents directly
- Runtime injection through secure environment or config
- Automatic rotation support
- Scoped credentials per tenant/agent/capability

Security Model:
1. Credentials stored encrypted at rest
2. Retrieved only when needed for execution
3. Injected into sandbox environment (not agent code)
4. Cleared immediately after execution
5. Full audit trail of all credential access
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CredentialScope(str, Enum):
    """Scope for credential access."""

    GLOBAL = "global"  # Available to all agents
    TENANT = "tenant"  # Available to agents in a specific tenant
    AGENT = "agent"  # Available to a specific agent only
    EXECUTION = "execution"  # Single-use for one execution


@dataclass
class CredentialEntry:
    """A credential entry in the vault."""

    credential_id: str
    name: str
    scope: CredentialScope
    encrypted_value: bytes
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    last_accessed_at: datetime | None = None
    access_count: int = 0

    # Scoping
    tenant_id: str | None = None
    agent_names: list[str] | None = None

    # Metadata
    description: str = ""
    tags: list[str] = field(default_factory=list)
    rotation_policy: str | None = None  # "30d", "90d", "manual"

    @property
    def is_expired(self) -> bool:
        """Check if credential is expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at


class CredentialVault:
    """
    Secure vault for managing credentials used by external agents.

    Credentials are:
    - Encrypted at rest using AES-256-GCM
    - Never exposed to external agent code
    - Injected at runtime through secure channels
    - Automatically rotated based on policy
    - Fully audited for compliance

    Usage:
        vault = CredentialVault(encryption_key=key)

        # Store credential
        vault.store(
            name="OPENAI_API_KEY",
            value="sk-...",
            scope=CredentialScope.TENANT,
            tenant_id="acme-corp",
        )

        # Get credentials for execution
        creds = await vault.get_credentials_for_execution(
            agent_name="openclaw",
            tenant_id="acme-corp",
            required_credentials=["OPENAI_API_KEY"],
        )
    """

    def __init__(
        self,
        encryption_key: bytes | None = None,
        storage_backend: Any | None = None,
        audit_logger: Any | None = None,
    ):
        # Use environment key or generate ephemeral key
        self._encryption_key = encryption_key or self._get_or_create_key()
        self._storage = storage_backend  # Future: Redis, Vault, AWS Secrets Manager
        self._audit_logger = audit_logger
        self._credentials: dict[str, CredentialEntry] = {}

    def _get_or_create_key(self) -> bytes:
        """Get encryption key from environment or create ephemeral key."""
        env_key = os.environ.get("ARAGORA_CREDENTIAL_VAULT_KEY")
        if env_key:
            # Use PBKDF2 with salt for proper key derivation
            salt = os.environ.get(
                "ARAGORA_CREDENTIAL_VAULT_SALT", "aragora-vault-default-salt"
            ).encode()
            return hashlib.pbkdf2_hmac(
                "sha256",
                env_key.encode(),
                salt,
                iterations=600_000,  # OWASP 2023 recommendation
            )
        # Ephemeral key for development
        logger.warning(
            "Using ephemeral credential vault key - credentials will not persist across restarts"
        )
        return secrets.token_bytes(32)

    def _encrypt(self, value: str) -> bytes:
        """Encrypt a credential value using AES-256-GCM."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            # SECURITY: Never store credentials unencrypted - fail securely
            raise RuntimeError(
                "cryptography library is required for credential vault. "
                "Install with: pip install cryptography"
            )

        nonce = secrets.token_bytes(12)
        aesgcm = AESGCM(self._encryption_key)
        ciphertext = aesgcm.encrypt(nonce, value.encode("utf-8"), None)
        return nonce + ciphertext

    def _decrypt(self, encrypted: bytes) -> str:
        """Decrypt a credential value."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            # SECURITY: Never allow decryption without proper crypto library
            raise RuntimeError(
                "cryptography library is required for credential vault. "
                "Install with: pip install cryptography"
            )

        nonce = encrypted[:12]
        ciphertext = encrypted[12:]
        aesgcm = AESGCM(self._encryption_key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode("utf-8")

    def _decrypt_with_key(self, encrypted: bytes, key: bytes) -> str:
        """Decrypt a credential value using a specific key."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            raise RuntimeError(
                "cryptography library is required for credential vault. "
                "Install with: pip install cryptography"
            )

        nonce = encrypted[:12]
        ciphertext = encrypted[12:]
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode("utf-8")

    def _try_decrypt_with_migration(self, encrypted_data: bytes, key: bytes) -> str | None:
        """Try to decrypt data, falling back to legacy SHA-256 key if needed."""
        try:
            return self._decrypt_with_key(encrypted_data, key)
        except Exception:
            # Try legacy SHA-256 key derivation
            env_key = os.environ.get("ARAGORA_CREDENTIAL_VAULT_KEY")
            if env_key:
                legacy_key = hashlib.sha256(env_key.encode()).digest()
                try:
                    decrypted = self._decrypt_with_key(encrypted_data, legacy_key)
                    logger.warning(
                        "Credential decrypted with legacy SHA-256 key - "
                        "will be re-encrypted with PBKDF2 on next write"
                    )
                    return decrypted
                except Exception:
                    pass
            return None

    def store(
        self,
        name: str,
        value: str,
        scope: CredentialScope = CredentialScope.GLOBAL,
        tenant_id: str | None = None,
        agent_names: list[str] | None = None,
        expires_in: timedelta | None = None,
        description: str = "",
        rotation_policy: str | None = None,
    ) -> str:
        """
        Store a credential in the vault.

        Args:
            name: Credential name (e.g., "OPENAI_API_KEY")
            value: The secret value
            scope: Access scope
            tenant_id: Tenant restriction (for TENANT scope)
            agent_names: Agent restrictions
            expires_in: Optional expiration
            description: Human-readable description
            rotation_policy: Auto-rotation policy

        Returns:
            Credential ID
        """
        credential_id = f"{scope.value}:{name}:{secrets.token_hex(8)}"

        expires_at = None
        if expires_in:
            expires_at = datetime.now(timezone.utc) + expires_in

        entry = CredentialEntry(
            credential_id=credential_id,
            name=name,
            scope=scope,
            encrypted_value=self._encrypt(value),
            tenant_id=tenant_id,
            agent_names=agent_names,
            expires_at=expires_at,
            description=description,
            rotation_policy=rotation_policy,
        )

        self._credentials[credential_id] = entry
        logger.info(f"Stored credential: {name} (scope={scope.value})")

        return credential_id

    def _get_credential_value(
        self,
        entry: CredentialEntry,
        agent_name: str | None = None,
        tenant_id: str | None = None,
    ) -> str | None:
        """Get decrypted credential value with access checks."""
        # Check expiration
        if entry.is_expired:
            logger.warning(f"Credential {entry.name} has expired")
            return None

        # Check tenant scope
        if entry.scope == CredentialScope.TENANT:
            if entry.tenant_id and entry.tenant_id != tenant_id:
                logger.warning(f"Credential {entry.name} not accessible for tenant {tenant_id}")
                return None

        # Check agent scope
        if entry.agent_names:
            if agent_name not in entry.agent_names:
                logger.warning(f"Credential {entry.name} not accessible for agent {agent_name}")
                return None

        # Update access tracking
        entry.last_accessed_at = datetime.now(timezone.utc)
        entry.access_count += 1

        return self._decrypt(entry.encrypted_value)

    async def get_credentials_for_execution(
        self,
        agent_name: str,
        tenant_id: str | None = None,
        required_credentials: list[str] | None = None,
    ) -> dict[str, str]:
        """
        Get credentials for an agent execution.

        Retrieves all applicable credentials based on scope and restrictions.

        Args:
            agent_name: Name of the agent requesting credentials
            tenant_id: Tenant context
            required_credentials: Specific credentials requested

        Returns:
            Dictionary of credential name to value
        """
        result: dict[str, str] = {}

        for entry in self._credentials.values():
            # Skip if not in required list (if specified)
            if required_credentials and entry.name not in required_credentials:
                continue

            # Try to get the value (respects scope/access rules)
            value = self._get_credential_value(entry, agent_name, tenant_id)
            if value:
                result[entry.name] = value

        # Log audit event
        if self._audit_logger:
            await self._audit_logger.log_credential_access(
                agent_name=agent_name,
                tenant_id=tenant_id,
                credentials_accessed=list(result.keys()),
            )

        return result

    def rotate(self, credential_id: str, new_value: str) -> bool:
        """
        Rotate a credential to a new value.

        Args:
            credential_id: ID of credential to rotate
            new_value: New secret value

        Returns:
            True if rotation successful
        """
        entry = self._credentials.get(credential_id)
        if not entry:
            return False

        entry.encrypted_value = self._encrypt(new_value)
        entry.access_count = 0  # Reset after rotation
        logger.info(f"Rotated credential: {entry.name}")

        return True

    def revoke(self, credential_id: str) -> bool:
        """
        Revoke (delete) a credential.

        Args:
            credential_id: ID of credential to revoke

        Returns:
            True if revocation successful
        """
        if credential_id in self._credentials:
            entry = self._credentials.pop(credential_id)
            logger.info(f"Revoked credential: {entry.name}")
            return True
        return False

    def list_credentials(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
        """
        List credentials (metadata only, not values).

        Args:
            tenant_id: Filter by tenant

        Returns:
            List of credential metadata
        """
        result = []
        for entry in self._credentials.values():
            if tenant_id and entry.tenant_id and entry.tenant_id != tenant_id:
                continue

            result.append(
                {
                    "credential_id": entry.credential_id,
                    "name": entry.name,
                    "scope": entry.scope.value,
                    "tenant_id": entry.tenant_id,
                    "agent_names": entry.agent_names,
                    "created_at": entry.created_at.isoformat(),
                    "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
                    "is_expired": entry.is_expired,
                    "access_count": entry.access_count,
                    "description": entry.description,
                }
            )

        return result

    async def cleanup_expired(self) -> int:
        """Remove expired credentials. Returns count of removed."""
        expired = [cid for cid, entry in self._credentials.items() if entry.is_expired]
        for cid in expired:
            self._credentials.pop(cid)
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired credentials")
        return len(expired)
