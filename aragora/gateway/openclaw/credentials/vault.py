"""
OpenClaw Credential Vault - Main CredentialVault class.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import secrets
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

from .enums import CredentialAuditEvent, CredentialType
from .exceptions import (
    CredentialAccessDeniedError,
    CredentialExpiredError,
    CredentialNotFoundError,
    CredentialRateLimitedError,
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

logger = logging.getLogger(__name__)

# Try to import cryptography library
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography library not available - encryption disabled")


class CredentialVault:
    """
    Secure vault for storing and managing external framework credentials.

    Provides:
    - AES-256-GCM encryption at rest
    - Per-tenant credential isolation
    - Multiple KMS backend support
    - Automatic rotation with configurable policies
    - RBAC-based access control
    - Full audit logging
    - Rate limiting

    Usage:
        vault = CredentialVault(
            kms_provider=get_kms_provider(),
            audit_logger=get_auditor(),
        )

        # Store credential
        cred_id = await vault.store_credential(
            tenant_id="acme",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="sk-...",
            auth_context=ctx,
        )

        # Retrieve credential
        value = await vault.get_credential_value(cred_id, auth_context=ctx)
    """

    # Required permissions
    PERMISSION_CREATE = "credentials:create"
    PERMISSION_READ = "credentials:read"
    PERMISSION_UPDATE = "credentials:update"
    PERMISSION_DELETE = "credentials:delete"
    PERMISSION_ROTATE = "credentials:rotate"
    PERMISSION_LIST = "credentials:list"
    PERMISSION_ADMIN = "credentials:admin"

    def __init__(
        self,
        kms_provider: KMSProviderProtocol | None = None,
        audit_logger: AuditLoggerProtocol | None = None,
        rate_limiter: CredentialRateLimiter | None = None,
        storage_backend: Any | None = None,
        tenant_key_prefix: str = "tenant",
    ):
        """
        Initialize the credential vault.

        Args:
            kms_provider: KMS provider for encryption key management
            audit_logger: Audit logger for tracking all access
            rate_limiter: Rate limiter for access control
            storage_backend: Optional storage backend (default: in-memory)
            tenant_key_prefix: Prefix for tenant-scoped encryption keys
        """
        self._kms_provider = kms_provider
        self._audit_logger = audit_logger
        self._rate_limiter = rate_limiter or CredentialRateLimiter()
        self._storage = storage_backend
        self._tenant_key_prefix = tenant_key_prefix

        # In-memory storage (replaced by storage_backend in production)
        self._credentials: dict[str, StoredCredential] = {}
        self._tenant_keys: dict[str, bytes] = {}
        self._lock = threading.Lock()

        # Ephemeral master key for development (use KMS in production)
        self._master_key: bytes | None = None

    async def _get_tenant_key(self, tenant_id: str) -> bytes:
        """
        Get or generate the encryption key for a tenant.

        In production, this should use the KMS provider to manage
        per-tenant keys securely.
        """
        key_id = f"{self._tenant_key_prefix}:{tenant_id}"

        # Check cache
        if key_id in self._tenant_keys:
            return self._tenant_keys[key_id]

        # Get from KMS
        if self._kms_provider:
            key = await self._kms_provider.get_encryption_key(key_id)
        else:
            # Development fallback: derive from master key
            if self._master_key is None:
                env_key = os.environ.get("ARAGORA_CREDENTIAL_VAULT_KEY")
                if env_key:
                    self._master_key = bytes.fromhex(env_key)
                else:
                    self._master_key = secrets.token_bytes(32)
                    logger.warning(
                        "Using ephemeral credential vault key - "
                        "credentials will not persist across restarts"
                    )

            # Derive tenant-specific key
            key = hashlib.pbkdf2_hmac(
                "sha256",
                self._master_key,
                tenant_id.encode("utf-8"),
                100000,
                dklen=32,
            )

        self._tenant_keys[key_id] = key
        return key

    def _encrypt(self, value: str, key: bytes) -> bytes:
        """Encrypt a credential value using AES-256-GCM."""
        # Look up via the backward-compatible stub so that patching
        # ``aragora.gateway.openclaw.credential_vault.CRYPTO_AVAILABLE``
        # in tests propagates correctly.
        import aragora.gateway.openclaw.credential_vault as _compat_mod

        if not _compat_mod.CRYPTO_AVAILABLE:
            # SECURITY: Never store credentials without proper encryption
            raise EncryptionError(
                "cryptography library is required for credential vault. "
                "Install with: pip install cryptography"
            )

        nonce = secrets.token_bytes(12)
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, value.encode("utf-8"), None)
        return nonce + ciphertext

    def _decrypt(self, encrypted: bytes, key: bytes) -> str:
        """Decrypt a credential value."""
        # Look up via the backward-compatible stub so that patching
        # ``aragora.gateway.openclaw.credential_vault.CRYPTO_AVAILABLE``
        # in tests propagates correctly.
        import aragora.gateway.openclaw.credential_vault as _compat_mod

        if not _compat_mod.CRYPTO_AVAILABLE:
            # SECURITY: Never allow decryption without proper crypto library
            raise EncryptionError(
                "cryptography library is required for credential vault. "
                "Install with: pip install cryptography"
            )

        nonce = encrypted[:12]
        ciphertext = encrypted[12:]
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode("utf-8")

    def _check_permission(
        self,
        auth_context: AuthorizationContextProtocol | None,
        permission: str,
    ) -> bool:
        """Check if auth context has required permission."""
        if auth_context is None:
            # No context = internal call, allow
            return True

        # Admin permission grants all
        if auth_context.has_permission(self.PERMISSION_ADMIN):
            return True

        return auth_context.has_permission(permission)

    def _check_tenant_access(
        self,
        auth_context: AuthorizationContextProtocol | None,
        tenant_id: str,
    ) -> bool:
        """Check if auth context has access to tenant."""
        if auth_context is None:
            return True

        # Admin role has cross-tenant access
        if auth_context.has_role("admin") or auth_context.has_role("owner"):
            return True

        # Check org_id matches tenant_id
        return auth_context.org_id == tenant_id

    async def _log_audit(
        self,
        event: CredentialAuditEvent,
        actor_id: str,
        tenant_id: str | None = None,
        credential_id: str | None = None,
        details: dict[str, Any] | None = None,
        severity: str = "info",
    ) -> None:
        """Log an audit event."""
        if self._audit_logger:
            await self._audit_logger.log_event(
                event_type=event.value,
                actor_id=actor_id,
                tenant_id=tenant_id,
                resource_id=credential_id,
                details=details,
                severity=severity,
            )
        else:
            logger.info(
                "CredentialVault audit: %s actor=%s tenant=%s credential=%s",
                event.value,
                actor_id,
                tenant_id,
                credential_id,
            )

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    async def store_credential(
        self,
        tenant_id: str,
        framework: str,
        credential_type: CredentialType,
        value: str,
        auth_context: AuthorizationContextProtocol | None = None,
        credential_id: str | None = None,
        rotation_policy: RotationPolicy | None = None,
        expires_in: timedelta | None = None,
        allowed_agents: list[str] | None = None,
        allowed_capabilities: list[str] | None = None,
        description: str = "",
        tags: list[str] | None = None,
    ) -> str:
        """
        Store a new credential in the vault.

        Args:
            tenant_id: Tenant that owns the credential
            framework: External framework (openai, anthropic, etc.)
            credential_type: Type of credential
            value: The secret credential value
            auth_context: Authorization context for RBAC
            credential_id: Optional custom credential ID
            rotation_policy: Rotation policy (default: standard)
            expires_in: Expiration time from now
            allowed_agents: Agents allowed to use credential
            allowed_capabilities: Capabilities allowed to use credential
            description: Human-readable description
            tags: Custom tags

        Returns:
            Credential ID

        Raises:
            CredentialAccessDeniedError: If permission denied
            TenantIsolationError: If tenant access denied
        """
        # Permission check
        if not self._check_permission(auth_context, self.PERMISSION_CREATE):
            user_id = auth_context.user_id if auth_context else "unknown"
            await self._log_audit(
                CredentialAuditEvent.CREDENTIAL_ACCESS_DENIED,
                user_id,
                tenant_id,
                details={"action": "create", "reason": "permission_denied"},
                severity="warning",
            )
            raise CredentialAccessDeniedError(
                "Permission denied: credentials:create required",
                user_id=user_id,
                reason="permission_denied",
            )

        # Tenant access check
        if not self._check_tenant_access(auth_context, tenant_id):
            user_id = auth_context.user_id if auth_context else "unknown"
            await self._log_audit(
                CredentialAuditEvent.CREDENTIAL_ACCESS_DENIED,
                user_id,
                tenant_id,
                details={"action": "create", "reason": "tenant_isolation"},
                severity="warning",
            )
            raise TenantIsolationError(f"Access denied to tenant: {tenant_id}")

        # Generate credential ID
        if credential_id is None:
            credential_id = f"cred_{secrets.token_hex(16)}"

        # Get tenant encryption key
        key = await self._get_tenant_key(tenant_id)

        # Encrypt the value
        try:
            encrypted_value = self._encrypt(value, key)
        except (ValueError, RuntimeError, OSError) as e:
            raise EncryptionError(f"Failed to encrypt credential: {e}") from e

        # Calculate expiration
        expires_at = None
        if expires_in:
            expires_at = datetime.now(timezone.utc) + expires_in

        # Create metadata
        user_id = auth_context.user_id if auth_context else "system"
        metadata = CredentialMetadata(
            created_at=datetime.now(timezone.utc),
            created_by=user_id,
            expires_at=expires_at,
            description=description,
            tags=tags or [],
        )

        # Create stored credential
        credential = StoredCredential(
            credential_id=credential_id,
            tenant_id=tenant_id,
            framework=framework,
            credential_type=credential_type,
            encrypted_value=encrypted_value,
            encryption_key_id=f"{self._tenant_key_prefix}:{tenant_id}",
            metadata=metadata,
            rotation_policy=rotation_policy or RotationPolicy.standard(),
            allowed_agents=allowed_agents or [],
            allowed_capabilities=allowed_capabilities or [],
        )

        # Store
        with self._lock:
            self._credentials[credential_id] = credential

        # Audit log
        await self._log_audit(
            CredentialAuditEvent.CREDENTIAL_CREATED,
            user_id,
            tenant_id,
            credential_id,
            {
                "framework": framework,
                "credential_type": credential_type.value,
                "has_expiration": expires_at is not None,
            },
        )

        return credential_id

    async def get_credential(
        self,
        credential_id: str,
        auth_context: AuthorizationContextProtocol | None = None,
    ) -> StoredCredential:
        """
        Get credential metadata (without decrypted value).

        Args:
            credential_id: Credential ID
            auth_context: Authorization context

        Returns:
            StoredCredential (encrypted value only)

        Raises:
            CredentialNotFoundError: If credential not found
            CredentialAccessDeniedError: If permission denied
        """
        # Get credential
        with self._lock:
            credential = self._credentials.get(credential_id)

        if credential is None:
            raise CredentialNotFoundError(f"Credential not found: {credential_id}")

        # Permission check
        if not self._check_permission(auth_context, self.PERMISSION_READ):
            user_id = auth_context.user_id if auth_context else "unknown"
            await self._log_audit(
                CredentialAuditEvent.CREDENTIAL_ACCESS_DENIED,
                user_id,
                credential.tenant_id,
                credential_id,
                {"action": "read", "reason": "permission_denied"},
                severity="warning",
            )
            raise CredentialAccessDeniedError(
                "Permission denied: credentials:read required",
                credential_id=credential_id,
                user_id=user_id,
            )

        # Tenant access check
        if not self._check_tenant_access(auth_context, credential.tenant_id):
            user_id = auth_context.user_id if auth_context else "unknown"
            await self._log_audit(
                CredentialAuditEvent.CREDENTIAL_ACCESS_DENIED,
                user_id,
                credential.tenant_id,
                credential_id,
                {"action": "read", "reason": "tenant_isolation"},
                severity="warning",
            )
            raise TenantIsolationError(
                f"Cross-tenant access denied for credential: {credential_id}"
            )

        return credential

    async def get_credential_value(
        self,
        credential_id: str,
        auth_context: AuthorizationContextProtocol | None = None,
        agent_name: str | None = None,
        capability: str | None = None,
    ) -> str:
        """
        Get decrypted credential value.

        This is the main method for retrieving credentials for use.
        Rate limited and fully audited.

        Args:
            credential_id: Credential ID
            auth_context: Authorization context
            agent_name: Agent requesting the credential
            capability: Capability requesting the credential

        Returns:
            Decrypted credential value

        Raises:
            CredentialNotFoundError: If credential not found
            CredentialAccessDeniedError: If permission denied
            CredentialExpiredError: If credential expired
            CredentialRateLimitedError: If rate limited
        """
        user_id = auth_context.user_id if auth_context else "system"

        # Rate limit check
        allowed, retry_after = self._rate_limiter.check_rate_limit(
            user_id, auth_context.org_id if auth_context else None
        )
        if not allowed:
            await self._log_audit(
                CredentialAuditEvent.CREDENTIAL_RATE_LIMITED,
                user_id,
                details={"credential_id": credential_id, "retry_after": retry_after},
                severity="warning",
            )
            raise CredentialRateLimitedError(
                f"Rate limit exceeded. Retry after {retry_after} seconds.",
                retry_after,
            )

        # Get credential (validates permissions and tenant access)
        credential = await self.get_credential(credential_id, auth_context)

        # Check expiration
        if credential.is_expired:
            await self._log_audit(
                CredentialAuditEvent.CREDENTIAL_EXPIRED,
                user_id,
                credential.tenant_id,
                credential_id,
                severity="warning",
            )
            raise CredentialExpiredError(f"Credential expired: {credential_id}")

        # Check agent allowlist
        if credential.allowed_agents:
            if agent_name and agent_name not in credential.allowed_agents:
                await self._log_audit(
                    CredentialAuditEvent.CREDENTIAL_ACCESS_DENIED,
                    user_id,
                    credential.tenant_id,
                    credential_id,
                    {"reason": "agent_not_allowed", "agent": agent_name},
                    severity="warning",
                )
                raise CredentialAccessDeniedError(
                    f"Agent '{agent_name}' not allowed for credential",
                    credential_id=credential_id,
                    user_id=user_id,
                    reason="agent_not_allowed",
                )

        # Check capability allowlist
        if credential.allowed_capabilities:
            if capability and capability not in credential.allowed_capabilities:
                await self._log_audit(
                    CredentialAuditEvent.CREDENTIAL_ACCESS_DENIED,
                    user_id,
                    credential.tenant_id,
                    credential_id,
                    {"reason": "capability_not_allowed", "capability": capability},
                    severity="warning",
                )
                raise CredentialAccessDeniedError(
                    f"Capability '{capability}' not allowed for credential",
                    credential_id=credential_id,
                    user_id=user_id,
                    reason="capability_not_allowed",
                )

        # Get decryption key
        key = await self._get_tenant_key(credential.tenant_id)

        # Decrypt
        try:
            value = self._decrypt(credential.encrypted_value, key)
        except (ValueError, RuntimeError, OSError) as e:
            raise EncryptionError(f"Failed to decrypt credential: {e}") from e

        # Update access tracking
        with self._lock:
            credential.metadata.access_count += 1
            credential.metadata.last_accessed_at = datetime.now(timezone.utc)
            credential.metadata.last_accessed_by = user_id

        # Audit log
        await self._log_audit(
            CredentialAuditEvent.CREDENTIAL_ACCESSED,
            user_id,
            credential.tenant_id,
            credential_id,
            {
                "framework": credential.framework,
                "agent": agent_name,
                "capability": capability,
                "access_count": credential.metadata.access_count,
            },
        )

        return value

    async def update_credential(
        self,
        credential_id: str,
        auth_context: AuthorizationContextProtocol | None = None,
        rotation_policy: RotationPolicy | None = None,
        expires_in: timedelta | None = None,
        allowed_agents: list[str] | None = None,
        allowed_capabilities: list[str] | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> StoredCredential:
        """
        Update credential metadata (not the value - use rotate_credential for that).

        Args:
            credential_id: Credential ID
            auth_context: Authorization context
            rotation_policy: New rotation policy
            expires_in: New expiration time from now
            allowed_agents: New agent allowlist
            allowed_capabilities: New capability allowlist
            description: New description
            tags: New tags

        Returns:
            Updated StoredCredential

        Raises:
            CredentialNotFoundError: If credential not found
            CredentialAccessDeniedError: If permission denied
        """
        # Permission check
        if not self._check_permission(auth_context, self.PERMISSION_UPDATE):
            user_id = auth_context.user_id if auth_context else "unknown"
            raise CredentialAccessDeniedError(
                "Permission denied: credentials:update required",
                credential_id=credential_id,
                user_id=user_id,
            )

        # Get existing credential
        credential = await self.get_credential(credential_id, auth_context)
        user_id = auth_context.user_id if auth_context else "system"

        # Update fields
        with self._lock:
            if rotation_policy is not None:
                credential.rotation_policy = rotation_policy
            if expires_in is not None:
                credential.metadata.expires_at = datetime.now(timezone.utc) + expires_in
            if allowed_agents is not None:
                credential.allowed_agents = allowed_agents
            if allowed_capabilities is not None:
                credential.allowed_capabilities = allowed_capabilities
            if description is not None:
                credential.metadata.description = description
            if tags is not None:
                credential.metadata.tags = tags

        # Audit log
        await self._log_audit(
            CredentialAuditEvent.CREDENTIAL_UPDATED,
            user_id,
            credential.tenant_id,
            credential_id,
            {
                "updated_fields": [
                    k
                    for k, v in [
                        ("rotation_policy", rotation_policy),
                        ("expires_in", expires_in),
                        ("allowed_agents", allowed_agents),
                        ("allowed_capabilities", allowed_capabilities),
                        ("description", description),
                        ("tags", tags),
                    ]
                    if v is not None
                ]
            },
        )

        return credential

    async def delete_credential(
        self,
        credential_id: str,
        auth_context: AuthorizationContextProtocol | None = None,
    ) -> bool:
        """
        Delete a credential from the vault.

        Args:
            credential_id: Credential ID
            auth_context: Authorization context

        Returns:
            True if deleted

        Raises:
            CredentialNotFoundError: If credential not found
            CredentialAccessDeniedError: If permission denied
        """
        # Permission check
        if not self._check_permission(auth_context, self.PERMISSION_DELETE):
            user_id = auth_context.user_id if auth_context else "unknown"
            raise CredentialAccessDeniedError(
                "Permission denied: credentials:delete required",
                credential_id=credential_id,
                user_id=user_id,
            )

        # Get credential to verify access
        credential = await self.get_credential(credential_id, auth_context)
        user_id = auth_context.user_id if auth_context else "system"

        # Delete
        with self._lock:
            del self._credentials[credential_id]

        # Audit log
        await self._log_audit(
            CredentialAuditEvent.CREDENTIAL_DELETED,
            user_id,
            credential.tenant_id,
            credential_id,
            {"framework": credential.framework},
        )

        return True

    # =========================================================================
    # Rotation Operations
    # =========================================================================

    async def rotate_credential(
        self,
        credential_id: str,
        new_value: str,
        auth_context: AuthorizationContextProtocol | None = None,
    ) -> StoredCredential:
        """
        Rotate a credential to a new value.

        Args:
            credential_id: Credential ID
            new_value: New credential value
            auth_context: Authorization context

        Returns:
            Updated StoredCredential

        Raises:
            CredentialNotFoundError: If credential not found
            CredentialAccessDeniedError: If permission denied
        """
        # Permission check
        if not self._check_permission(auth_context, self.PERMISSION_ROTATE):
            user_id = auth_context.user_id if auth_context else "unknown"
            raise CredentialAccessDeniedError(
                "Permission denied: credentials:rotate required",
                credential_id=credential_id,
                user_id=user_id,
            )

        # Get existing credential
        credential = await self.get_credential(credential_id, auth_context)
        user_id = auth_context.user_id if auth_context else "system"

        # Get encryption key
        key = await self._get_tenant_key(credential.tenant_id)

        # Encrypt new value
        try:
            encrypted_value = self._encrypt(new_value, key)
        except (ValueError, RuntimeError, OSError) as e:
            await self._log_audit(
                CredentialAuditEvent.ROTATION_FAILED,
                user_id,
                credential.tenant_id,
                credential_id,
                {"error": str(e)},
                severity="error",
            )
            raise EncryptionError(f"Failed to encrypt new credential: {e}") from e

        # Update credential
        with self._lock:
            credential.encrypted_value = encrypted_value
            credential.metadata.rotated_at = datetime.now(timezone.utc)
            credential.metadata.rotated_by = user_id
            credential.metadata.version += 1
            credential.metadata.access_count = 0  # Reset after rotation

        # Audit log
        await self._log_audit(
            CredentialAuditEvent.CREDENTIAL_ROTATED,
            user_id,
            credential.tenant_id,
            credential_id,
            {
                "framework": credential.framework,
                "version": credential.metadata.version,
            },
        )

        return credential

    async def get_credentials_needing_rotation(
        self,
        tenant_id: str | None = None,
        auth_context: AuthorizationContextProtocol | None = None,
    ) -> list[StoredCredential]:
        """
        Get credentials that need rotation based on their policies.

        Args:
            tenant_id: Filter by tenant (None = all accessible)
            auth_context: Authorization context

        Returns:
            List of credentials needing rotation
        """
        if not self._check_permission(auth_context, self.PERMISSION_LIST):
            return []

        results = []
        with self._lock:
            for credential in self._credentials.values():
                # Filter by tenant if specified
                if tenant_id and credential.tenant_id != tenant_id:
                    continue

                # Check tenant access
                if not self._check_tenant_access(auth_context, credential.tenant_id):
                    continue

                # Check if rotation needed
                if credential.needs_rotation:
                    results.append(credential)

        return results

    # =========================================================================
    # Listing and Search
    # =========================================================================

    async def list_credentials(
        self,
        tenant_id: str | None = None,
        framework: str | None = None,
        credential_type: CredentialType | None = None,
        auth_context: AuthorizationContextProtocol | None = None,
        include_expired: bool = False,
    ) -> list[StoredCredential]:
        """
        List credentials matching the criteria.

        Args:
            tenant_id: Filter by tenant
            framework: Filter by framework
            credential_type: Filter by credential type
            auth_context: Authorization context
            include_expired: Include expired credentials

        Returns:
            List of matching credentials (metadata only)
        """
        if not self._check_permission(auth_context, self.PERMISSION_LIST):
            user_id = auth_context.user_id if auth_context else "unknown"
            raise CredentialAccessDeniedError(
                "Permission denied: credentials:list required",
                user_id=user_id,
            )

        results = []
        with self._lock:
            for credential in self._credentials.values():
                # Filter by tenant
                if tenant_id and credential.tenant_id != tenant_id:
                    continue

                # Check tenant access
                if not self._check_tenant_access(auth_context, credential.tenant_id):
                    continue

                # Filter by framework
                if framework and credential.framework != framework:
                    continue

                # Filter by type
                if credential_type and credential.credential_type != credential_type:
                    continue

                # Filter expired
                if not include_expired and credential.is_expired:
                    continue

                results.append(credential)

        return results

    async def get_credentials_for_execution(
        self,
        tenant_id: str,
        agent_name: str | None = None,
        frameworks: list[str] | None = None,
        auth_context: AuthorizationContextProtocol | None = None,
    ) -> dict[str, str]:
        """
        Get credentials for agent execution.

        Convenience method that returns decrypted credentials as a dict
        suitable for injecting into environment or config.

        Args:
            tenant_id: Tenant context
            agent_name: Agent requesting credentials
            frameworks: List of frameworks to get credentials for
            auth_context: Authorization context

        Returns:
            Dict of framework -> credential value
        """
        result: dict[str, str] = {}

        # List accessible credentials
        credentials = await self.list_credentials(
            tenant_id=tenant_id,
            auth_context=auth_context,
        )

        for credential in credentials:
            # Filter by framework if specified
            if frameworks and credential.framework not in frameworks:
                continue

            # Check agent allowlist
            if credential.allowed_agents:
                if agent_name and agent_name not in credential.allowed_agents:
                    continue

            # Get decrypted value
            try:
                value = await self.get_credential_value(
                    credential.credential_id,
                    auth_context=auth_context,
                    agent_name=agent_name,
                )
                # Use framework as key (e.g., "openai" -> "sk-...")
                result[credential.framework] = value
            except (CredentialAccessDeniedError, CredentialExpiredError):
                # Skip inaccessible credentials
                continue

        return result

    # =========================================================================
    # Maintenance
    # =========================================================================

    async def cleanup_expired(
        self,
        auth_context: AuthorizationContextProtocol | None = None,
    ) -> int:
        """
        Remove expired credentials.

        Args:
            auth_context: Authorization context (requires admin)

        Returns:
            Number of credentials removed
        """
        if not self._check_permission(auth_context, self.PERMISSION_ADMIN):
            return 0

        user_id = auth_context.user_id if auth_context else "system"
        removed = 0

        with self._lock:
            expired_ids = [
                cred_id for cred_id, cred in self._credentials.items() if cred.is_expired
            ]

            for cred_id in expired_ids:
                credential = self._credentials.pop(cred_id)
                removed += 1

                # Log each removal
                asyncio.create_task(
                    self._log_audit(
                        CredentialAuditEvent.CREDENTIAL_DELETED,
                        user_id,
                        credential.tenant_id,
                        cred_id,
                        {"reason": "expired", "framework": credential.framework},
                    )
                )

        if removed:
            logger.info("Cleaned up %d expired credentials", removed)

        return removed

    def get_stats(self) -> dict[str, Any]:
        """Get vault statistics."""
        with self._lock:
            total = len(self._credentials)
            expired = sum(1 for c in self._credentials.values() if c.is_expired)
            needs_rotation = sum(1 for c in self._credentials.values() if c.needs_rotation)

            by_framework: dict[str, int] = {}
            by_tenant: dict[str, int] = {}

            for cred in self._credentials.values():
                by_framework[cred.framework] = by_framework.get(cred.framework, 0) + 1
                by_tenant[cred.tenant_id] = by_tenant.get(cred.tenant_id, 0) + 1

        return {
            "total_credentials": total,
            "expired_credentials": expired,
            "needs_rotation": needs_rotation,
            "by_framework": by_framework,
            "by_tenant": by_tenant,
            "encryption_available": CRYPTO_AVAILABLE,
        }
