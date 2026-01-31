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
    from aragora.gateway.openclaw.credential_vault import (
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

import asyncio
import base64
import hashlib
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Try to import cryptography library
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography library not available - encryption disabled")


# =============================================================================
# Dataclasses and Enums
# =============================================================================


class CredentialType(str, Enum):
    """Types of credentials that can be stored."""

    API_KEY = "api_key"
    OAUTH_TOKEN = "oauth_token"
    OAUTH_SECRET = "oauth_secret"
    OAUTH_REFRESH_TOKEN = "oauth_refresh_token"
    SERVICE_ACCOUNT = "service_account"
    CERTIFICATE = "certificate"
    PASSWORD = "password"
    BEARER_TOKEN = "bearer_token"
    WEBHOOK_SECRET = "webhook_secret"
    ENCRYPTION_KEY = "encryption_key"


class CredentialFramework(str, Enum):
    """External frameworks that credentials may be used with."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    AWS = "aws"
    HUGGINGFACE = "huggingface"
    LANGCHAIN = "langchain"
    CREWAI = "crewai"
    AUTOGEN = "autogen"
    OPENCLAW = "openclaw"
    CUSTOM = "custom"


@dataclass
class CredentialMetadata:
    """
    Metadata tracking for a stored credential.

    Attributes:
        created_at: When the credential was first stored
        created_by: User ID who created the credential
        rotated_at: When the credential was last rotated (None if never)
        rotated_by: User ID who performed the last rotation
        expires_at: When the credential expires (None if never)
        access_count: Number of times the credential has been retrieved
        last_accessed_at: When the credential was last retrieved
        last_accessed_by: User ID who last accessed the credential
        version: Version number (incremented on rotation)
        tags: Custom tags for categorization
        description: Human-readable description
    """

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str | None = None
    rotated_at: datetime | None = None
    rotated_by: str | None = None
    expires_at: datetime | None = None
    access_count: int = 0
    last_accessed_at: datetime | None = None
    last_accessed_by: str | None = None
    version: int = 1
    tags: list[str] = field(default_factory=list)
    description: str = ""

    @property
    def is_expired(self) -> bool:
        """Check if the credential has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def days_until_expiry(self) -> int | None:
        """Get days until credential expires, or None if no expiry."""
        if self.expires_at is None:
            return None
        delta = self.expires_at - datetime.now(timezone.utc)
        return max(0, delta.days)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "rotated_at": self.rotated_at.isoformat() if self.rotated_at else None,
            "rotated_by": self.rotated_by,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "access_count": self.access_count,
            "last_accessed_at": (
                self.last_accessed_at.isoformat() if self.last_accessed_at else None
            ),
            "last_accessed_by": self.last_accessed_by,
            "version": self.version,
            "tags": self.tags,
            "description": self.description,
            "is_expired": self.is_expired,
            "days_until_expiry": self.days_until_expiry,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CredentialMetadata":
        """Create from dictionary."""

        def parse_dt(val: str | None) -> datetime | None:
            if val is None:
                return None
            return datetime.fromisoformat(val.replace("Z", "+00:00"))

        return cls(
            created_at=parse_dt(data.get("created_at")) or datetime.now(timezone.utc),
            created_by=data.get("created_by"),
            rotated_at=parse_dt(data.get("rotated_at")),
            rotated_by=data.get("rotated_by"),
            expires_at=parse_dt(data.get("expires_at")),
            access_count=data.get("access_count", 0),
            last_accessed_at=parse_dt(data.get("last_accessed_at")),
            last_accessed_by=data.get("last_accessed_by"),
            version=data.get("version", 1),
            tags=data.get("tags", []),
            description=data.get("description", ""),
        )


@dataclass
class RotationPolicy:
    """
    Policy for automatic credential rotation.

    Attributes:
        interval_days: Days between rotations (0 = manual only)
        notify_before_days: Days before expiry to send alert
        auto_rotate: Whether to rotate automatically when due
        on_access_count: Rotate after N accesses (0 = disabled)
        on_compromise: Rotate immediately if compromise suspected
        require_approval: Require manual approval for rotation
        notify_channels: Channels to notify (email, slack, webhook)
    """

    interval_days: int = 90
    notify_before_days: int = 14
    auto_rotate: bool = False
    on_access_count: int = 0
    on_compromise: bool = True
    require_approval: bool = False
    notify_channels: list[str] = field(default_factory=list)

    def is_rotation_due(self, metadata: CredentialMetadata) -> bool:
        """Check if credential rotation is due based on policy."""
        if self.interval_days <= 0:
            return False

        # Check time-based rotation
        last_rotation = metadata.rotated_at or metadata.created_at
        days_since = (datetime.now(timezone.utc) - last_rotation).days
        if days_since >= self.interval_days:
            return True

        # Check access-count based rotation
        if self.on_access_count > 0 and metadata.access_count >= self.on_access_count:
            return True

        return False

    def needs_expiry_alert(self, metadata: CredentialMetadata) -> bool:
        """Check if expiry alert should be sent."""
        if self.notify_before_days <= 0:
            return False
        days_left = metadata.days_until_expiry
        if days_left is None:
            return False
        return days_left <= self.notify_before_days

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "interval_days": self.interval_days,
            "notify_before_days": self.notify_before_days,
            "auto_rotate": self.auto_rotate,
            "on_access_count": self.on_access_count,
            "on_compromise": self.on_compromise,
            "require_approval": self.require_approval,
            "notify_channels": self.notify_channels,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RotationPolicy":
        """Create from dictionary."""
        return cls(
            interval_days=data.get("interval_days", 90),
            notify_before_days=data.get("notify_before_days", 14),
            auto_rotate=data.get("auto_rotate", False),
            on_access_count=data.get("on_access_count", 0),
            on_compromise=data.get("on_compromise", True),
            require_approval=data.get("require_approval", False),
            notify_channels=data.get("notify_channels", []),
        )

    @classmethod
    def strict(cls) -> "RotationPolicy":
        """Create a strict rotation policy for high-security credentials."""
        return cls(
            interval_days=30,
            notify_before_days=7,
            auto_rotate=False,
            on_access_count=1000,
            on_compromise=True,
            require_approval=True,
        )

    @classmethod
    def standard(cls) -> "RotationPolicy":
        """Create a standard rotation policy."""
        return cls(
            interval_days=90,
            notify_before_days=14,
            auto_rotate=True,
            on_access_count=0,
            on_compromise=True,
            require_approval=False,
        )

    @classmethod
    def relaxed(cls) -> "RotationPolicy":
        """Create a relaxed rotation policy for development."""
        return cls(
            interval_days=365,
            notify_before_days=30,
            auto_rotate=False,
            on_access_count=0,
            on_compromise=True,
            require_approval=False,
        )


@dataclass
class StoredCredential:
    """
    A credential stored in the vault.

    Attributes:
        credential_id: Unique identifier for the credential
        tenant_id: Tenant that owns this credential
        framework: External framework the credential is for
        credential_type: Type of credential (api_key, oauth_token, etc.)
        encrypted_value: AES-256-GCM encrypted credential value
        encryption_key_id: ID of the key used for encryption
        metadata: Credential metadata (created_at, access_count, etc.)
        rotation_policy: Automatic rotation policy
        allowed_agents: Agent names allowed to use this credential (empty = all)
        allowed_capabilities: Capabilities that can use this credential
    """

    credential_id: str
    tenant_id: str
    framework: str
    credential_type: CredentialType
    encrypted_value: bytes
    encryption_key_id: str
    metadata: CredentialMetadata = field(default_factory=CredentialMetadata)
    rotation_policy: RotationPolicy = field(default_factory=RotationPolicy)
    allowed_agents: list[str] = field(default_factory=list)
    allowed_capabilities: list[str] = field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        """Check if credential has expired."""
        return self.metadata.is_expired

    @property
    def needs_rotation(self) -> bool:
        """Check if credential needs rotation based on policy."""
        return self.rotation_policy.is_rotation_due(self.metadata)

    def to_dict(self, include_encrypted: bool = False) -> dict[str, Any]:
        """Convert to dictionary for serialization (excludes encrypted value by default)."""
        result = {
            "credential_id": self.credential_id,
            "tenant_id": self.tenant_id,
            "framework": self.framework,
            "credential_type": self.credential_type.value,
            "encryption_key_id": self.encryption_key_id,
            "metadata": self.metadata.to_dict(),
            "rotation_policy": self.rotation_policy.to_dict(),
            "allowed_agents": self.allowed_agents,
            "allowed_capabilities": self.allowed_capabilities,
            "is_expired": self.is_expired,
            "needs_rotation": self.needs_rotation,
        }
        if include_encrypted:
            result["encrypted_value"] = base64.b64encode(self.encrypted_value).decode()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StoredCredential":
        """Create from dictionary."""
        encrypted = data.get("encrypted_value", b"")
        if isinstance(encrypted, str):
            encrypted = base64.b64decode(encrypted)

        return cls(
            credential_id=data["credential_id"],
            tenant_id=data["tenant_id"],
            framework=data["framework"],
            credential_type=CredentialType(data["credential_type"]),
            encrypted_value=encrypted,
            encryption_key_id=data.get("encryption_key_id", "default"),
            metadata=CredentialMetadata.from_dict(data.get("metadata", {})),
            rotation_policy=RotationPolicy.from_dict(data.get("rotation_policy", {})),
            allowed_agents=data.get("allowed_agents", []),
            allowed_capabilities=data.get("allowed_capabilities", []),
        )


# =============================================================================
# Audit Events
# =============================================================================


class CredentialAuditEvent(str, Enum):
    """Audit events for credential operations."""

    CREDENTIAL_CREATED = "credential_created"
    CREDENTIAL_ACCESSED = "credential_accessed"
    CREDENTIAL_UPDATED = "credential_updated"
    CREDENTIAL_ROTATED = "credential_rotated"
    CREDENTIAL_DELETED = "credential_deleted"
    CREDENTIAL_EXPIRED = "credential_expired"
    CREDENTIAL_ACCESS_DENIED = "credential_access_denied"
    CREDENTIAL_RATE_LIMITED = "credential_rate_limited"
    ROTATION_SCHEDULED = "rotation_scheduled"
    ROTATION_COMPLETED = "rotation_completed"
    ROTATION_FAILED = "rotation_failed"
    EXPIRY_ALERT_SENT = "expiry_alert_sent"


# =============================================================================
# Protocol Definitions
# =============================================================================


class KMSProviderProtocol(Protocol):
    """Protocol for KMS providers."""

    async def get_encryption_key(self, key_id: str) -> bytes:
        """Get or generate an encryption key."""
        ...

    async def decrypt_data_key(self, encrypted_key: bytes, key_id: str) -> bytes:
        """Decrypt a data key using the master key."""
        ...

    async def encrypt_data_key(self, plaintext_key: bytes, key_id: str) -> bytes:
        """Encrypt a data key using the master key."""
        ...


class AuditLoggerProtocol(Protocol):
    """Protocol for audit logging."""

    async def log_event(
        self,
        event_type: str,
        actor_id: str,
        tenant_id: str | None = None,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
        severity: str = "info",
    ) -> None:
        """Log an audit event."""
        ...


class AuthorizationContextProtocol(Protocol):
    """Protocol for authorization context."""

    user_id: str
    org_id: str | None
    roles: set[str]
    permissions: set[str]

    def has_permission(self, permission_key: str) -> bool:
        """Check if context has a permission."""
        ...

    def has_role(self, role_name: str) -> bool:
        """Check if context has a specific role."""
        ...


# =============================================================================
# Exceptions
# =============================================================================


class CredentialVaultError(Exception):
    """Base exception for credential vault errors."""

    pass


class CredentialNotFoundError(CredentialVaultError):
    """Credential not found in vault."""

    pass


class CredentialAccessDeniedError(CredentialVaultError):
    """Access to credential denied."""

    def __init__(
        self,
        message: str,
        credential_id: str | None = None,
        user_id: str | None = None,
        reason: str = "permission_denied",
    ):
        super().__init__(message)
        self.credential_id = credential_id
        self.user_id = user_id
        self.reason = reason


class CredentialExpiredError(CredentialVaultError):
    """Credential has expired."""

    pass


class CredentialRateLimitedError(CredentialVaultError):
    """Too many credential access attempts."""

    def __init__(self, message: str, retry_after_seconds: int = 60):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class TenantIsolationError(CredentialVaultError):
    """Cross-tenant credential access attempted."""

    pass


class EncryptionError(CredentialVaultError):
    """Encryption or decryption failed."""

    pass


# =============================================================================
# Rate Limiter
# =============================================================================


class CredentialRateLimiter:
    """
    Rate limiter for credential access to prevent harvesting attacks.

    Uses a sliding window algorithm with per-user and per-tenant limits.
    """

    def __init__(
        self,
        max_per_minute: int = 30,
        max_per_hour: int = 200,
        lockout_duration_seconds: int = 300,
    ):
        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        self.lockout_duration = lockout_duration_seconds

        # Access tracking: key -> list of timestamps
        self._accesses: dict[str, list[float]] = {}
        # Lockout tracking: key -> lockout_until timestamp
        self._lockouts: dict[str, float] = {}
        self._lock = threading.Lock()

    def _get_key(self, user_id: str, tenant_id: str | None = None) -> str:
        """Generate rate limit key."""
        return f"{tenant_id or 'global'}:{user_id}"

    def check_rate_limit(self, user_id: str, tenant_id: str | None = None) -> tuple[bool, int]:
        """
        Check if request is within rate limits.

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        key = self._get_key(user_id, tenant_id)
        now = time.time()

        with self._lock:
            # Check lockout
            if key in self._lockouts:
                if now < self._lockouts[key]:
                    return False, int(self._lockouts[key] - now)
                else:
                    del self._lockouts[key]

            # Get access history
            accesses = self._accesses.get(key, [])

            # Clean old entries
            minute_ago = now - 60
            hour_ago = now - 3600
            accesses = [t for t in accesses if t > hour_ago]

            # Count recent accesses
            minute_count = sum(1 for t in accesses if t > minute_ago)
            hour_count = len(accesses)

            # Check limits
            if minute_count >= self.max_per_minute:
                self._lockouts[key] = now + self.lockout_duration
                return False, self.lockout_duration

            if hour_count >= self.max_per_hour:
                retry_after = int(accesses[0] + 3600 - now) + 1
                return False, retry_after

            # Record access
            accesses.append(now)
            self._accesses[key] = accesses

            return True, 0

    def clear_user(self, user_id: str, tenant_id: str | None = None) -> None:
        """Clear rate limit state for a user."""
        key = self._get_key(user_id, tenant_id)
        with self._lock:
            self._accesses.pop(key, None)
            self._lockouts.pop(key, None)


# =============================================================================
# Credential Vault
# =============================================================================


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
        if not CRYPTO_AVAILABLE:
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
        if not CRYPTO_AVAILABLE:
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
        except Exception as e:
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
        except Exception as e:
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
        except Exception as e:
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
]
