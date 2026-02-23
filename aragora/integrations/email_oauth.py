"""
Email OAuth Token Storage.

Provides persistent storage for email provider OAuth tokens across SMTP providers
that support OAuth2 (Gmail via OAuth, Microsoft 365, etc.).

Supports multi-tenant deployments with tenant-isolated credentials and
automatic token refresh before expiry.

Backends:
- InMemoryEmailCredentialStore: Fast, single-instance only (for testing)
- SQLiteEmailCredentialStore: Persisted, single-instance (default)
- RedisEmailCredentialStore: Distributed, multi-instance (with SQLite fallback)
- PostgresEmailCredentialStore: Production multi-instance

Usage:
    from aragora.integrations.email_oauth import get_email_credential_store

    store = get_email_credential_store()
    await store.save(credential)
    cred = await store.get(tenant_id, provider, email_address)

    # Check if token needs refresh
    if cred.needs_refresh():
        new_token = await refresh_oauth_token(cred)
        cred.access_token = new_token
        await store.save(cred)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aragora.config import resolve_db_path

if TYPE_CHECKING:
    from asyncpg import Pool

logger = logging.getLogger(__name__)

# Import encryption (optional - graceful degradation if not available)
get_encryption_service: Any = None
is_encryption_required: Any = None
EncryptionError: Any = Exception
CRYPTO_AVAILABLE = False

try:
    from aragora.security.encryption import (
        CRYPTO_AVAILABLE,
        EncryptionError,
        get_encryption_service,
        is_encryption_required,
    )
except ImportError:
    CRYPTO_AVAILABLE = False

    def _fb_get_encryption_service() -> Any:
        raise RuntimeError("Encryption not available")

    def _fb_is_encryption_required() -> bool:
        """Fallback when security module unavailable - still check env vars."""
        if os.environ.get("ARAGORA_ENCRYPTION_REQUIRED", "").lower() in ("true", "1", "yes"):
            return True
        if os.environ.get("ARAGORA_ENV") == "production":
            return True
        return False

    class _FBEncryptionError(Exception):
        """Fallback exception when security module unavailable."""

        def __init__(self, operation: str, reason: str, store: str = ""):
            self.operation = operation
            self.reason = reason
            self.store = store
            super().__init__(
                f"Encryption {operation} failed in {store}: {reason}. "
                f"Set ARAGORA_ENCRYPTION_REQUIRED=false to allow plaintext fallback."
            )

    get_encryption_service = _fb_get_encryption_service
    is_encryption_required = _fb_is_encryption_required
    EncryptionError = _FBEncryptionError


# Token fields to encrypt
_TOKEN_FIELDS = ["access_token", "refresh_token", "client_secret"]

# Exported for migration scripts
ENCRYPTED_FIELDS = _TOKEN_FIELDS

# Default token refresh margin (refresh if expires within this time)
DEFAULT_REFRESH_MARGIN_SECONDS = 300  # 5 minutes


def _encrypt_token(token: str, context: str = "") -> str:
    """
    Encrypt a token value for storage.

    Uses context as Associated Authenticated Data (AAD) to bind the ciphertext
    to a specific credential, preventing cross-credential token attacks.

    Raises:
        EncryptionError: If encryption fails and ARAGORA_ENCRYPTION_REQUIRED is True.
    """
    if not token:
        return token

    if not CRYPTO_AVAILABLE:
        if is_encryption_required():
            raise EncryptionError(
                "encrypt",
                "cryptography library not available",
                "email_credential_store",
            )
        return token

    try:
        service = get_encryption_service()
        # AAD binds token to this specific credential
        encrypted = service.encrypt(token, associated_data=context if context else None)
        return encrypted.to_base64()
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        if is_encryption_required():
            raise EncryptionError(
                "encrypt",
                str(e),
                "email_credential_store",
            ) from e
        logger.warning("Token encryption failed, storing unencrypted: %s", e)
        return token


def _decrypt_token(encrypted_token: str, context: str = "") -> str:
    """
    Decrypt a token value, handling legacy unencrypted tokens.

    AAD must match what was used during encryption.
    """
    if not CRYPTO_AVAILABLE or not encrypted_token:
        return encrypted_token

    # Check if it looks like an encrypted value (base64-encoded EncryptedData)
    # EncryptedData always starts with version byte 0x01 which encodes to "A" in base64
    if not encrypted_token.startswith("A"):
        return encrypted_token  # Legacy unencrypted token

    try:
        service = get_encryption_service()
        return service.decrypt_string(encrypted_token, associated_data=context if context else None)
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        # Could be a legacy plain token that happens to start with "A"
        logger.debug("Token decryption failed for context %s, returning as-is: %s", context, e)
        return encrypted_token


@dataclass
class EmailCredential:
    """
    Per-tenant email OAuth credential.

    SECURITY: tenant_id provides isolation for multi-tenant deployments.
    Admin users can only manage email connections within their own tenant.
    """

    # Identity
    tenant_id: str
    provider: str  # "gmail", "microsoft", "smtp"
    email_address: str

    # OAuth tokens
    access_token: str = ""
    refresh_token: str = ""
    token_expiry: datetime | None = None

    # OAuth client config (for providers that need per-tenant apps)
    client_id: str = ""
    client_secret: str = ""

    # Provider-specific metadata
    provider_user_id: str = ""  # Provider's internal user ID
    scopes: list[str] = field(default_factory=list)

    # Status
    is_active: bool = True
    last_used: datetime | None = None
    failure_count: int = 0
    last_error: str = ""

    # Timestamps
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def credential_id(self) -> str:
        """Unique identifier for this credential."""
        return f"{self.tenant_id}:{self.provider}:{self.email_address}"

    def needs_refresh(self, margin_seconds: int = DEFAULT_REFRESH_MARGIN_SECONDS) -> bool:
        """Check if the access token needs refresh.

        Args:
            margin_seconds: Refresh if token expires within this many seconds

        Returns:
            True if token is expired or will expire soon
        """
        if not self.access_token:
            return True
        if not self.token_expiry:
            return False  # No expiry = assume valid
        now = datetime.now(timezone.utc)
        expiry = self.token_expiry
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        return (expiry - now).total_seconds() < margin_seconds

    def is_expired(self) -> bool:
        """Check if the access token is expired."""
        return self.needs_refresh(margin_seconds=0)

    def to_dict(self, include_secrets: bool = False) -> dict[str, Any]:
        """Serialize to dictionary (for API responses)."""
        result = {
            "tenant_id": self.tenant_id,
            "provider": self.provider,
            "email_address": self.email_address,
            "provider_user_id": self.provider_user_id,
            "scopes": self.scopes,
            "is_active": self.is_active,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "failure_count": self.failure_count,
            "needs_refresh": self.needs_refresh(),
            "is_expired": self.is_expired(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if include_secrets:
            result["access_token"] = self.access_token
            result["refresh_token"] = self.refresh_token
            result["token_expiry"] = self.token_expiry.isoformat() if self.token_expiry else None
            result["client_id"] = self.client_id
            result["client_secret"] = self.client_secret
        return result

    def to_json(self) -> str:
        """Serialize to JSON for storage."""
        data = {
            "tenant_id": self.tenant_id,
            "provider": self.provider,
            "email_address": self.email_address,
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_expiry": self.token_expiry.isoformat() if self.token_expiry else None,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "provider_user_id": self.provider_user_id,
            "scopes": self.scopes,
            "is_active": self.is_active,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "failure_count": self.failure_count,
            "last_error": self.last_error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> EmailCredential:
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return cls(
            tenant_id=data["tenant_id"],
            provider=data["provider"],
            email_address=data["email_address"],
            access_token=data.get("access_token", ""),
            refresh_token=data.get("refresh_token", ""),
            token_expiry=(
                datetime.fromisoformat(data["token_expiry"]) if data.get("token_expiry") else None
            ),
            client_id=data.get("client_id", ""),
            client_secret=data.get("client_secret", ""),
            provider_user_id=data.get("provider_user_id", ""),
            scopes=data.get("scopes", []),
            is_active=data.get("is_active", True),
            last_used=(
                datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None
            ),
            failure_count=data.get("failure_count", 0),
            last_error=data.get("last_error", ""),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
        )

    @classmethod
    def from_row(cls, row: tuple) -> EmailCredential:
        """Create from database row."""
        tenant_id = row[0]
        provider = row[1]
        email_address = row[2]
        context = f"{tenant_id}:{provider}:{email_address}"
        return cls(
            tenant_id=tenant_id,
            provider=provider,
            email_address=email_address,
            access_token=_decrypt_token(row[3] or "", context),
            refresh_token=_decrypt_token(row[4] or "", context),
            token_expiry=(datetime.fromisoformat(row[5]) if row[5] else None),
            client_id=row[6] or "",
            client_secret=_decrypt_token(row[7] or "", context),
            provider_user_id=row[8] or "",
            scopes=json.loads(row[9]) if row[9] else [],
            is_active=bool(row[10]) if row[10] is not None else True,
            last_used=(datetime.fromisoformat(row[11]) if row[11] else None),
            failure_count=row[12] or 0,
            last_error=row[13] or "",
            created_at=row[14] or time.time(),
            updated_at=row[15] or time.time(),
        )


class EmailCredentialStoreBackend(ABC):
    """Abstract base for email credential storage backends."""

    @abstractmethod
    async def get(
        self, tenant_id: str, provider: str, email_address: str
    ) -> EmailCredential | None:
        """Get credential for a specific tenant/provider/email combination."""
        pass

    @abstractmethod
    async def save(self, credential: EmailCredential) -> None:
        """Save credential."""
        pass

    @abstractmethod
    async def delete(self, tenant_id: str, provider: str, email_address: str) -> bool:
        """Delete credential. Returns True if deleted."""
        pass

    @abstractmethod
    async def list_for_tenant(self, tenant_id: str) -> list[EmailCredential]:
        """List all credentials for a tenant."""
        pass

    @abstractmethod
    async def list_expiring(self, within_seconds: int = 3600) -> list[EmailCredential]:
        """List credentials that will expire soon (for proactive refresh)."""
        pass

    @abstractmethod
    async def update_last_used(self, tenant_id: str, provider: str, email_address: str) -> None:
        """Update the last_used timestamp."""
        pass

    @abstractmethod
    async def record_failure(
        self, tenant_id: str, provider: str, email_address: str, error: str
    ) -> None:
        """Record a failure for the credential."""
        pass

    @abstractmethod
    async def reset_failures(self, tenant_id: str, provider: str, email_address: str) -> None:
        """Reset failure count after successful operation."""
        pass

    async def close(self) -> None:
        """Close connections (optional to implement)."""
        pass


class InMemoryEmailCredentialStore(EmailCredentialStoreBackend):
    """
    Thread-safe in-memory email credential store.

    Fast but not shared across restarts. Suitable for development/testing.
    """

    def __init__(self) -> None:
        self._credentials: dict[str, EmailCredential] = {}
        self._lock = asyncio.Lock()

    def _key(self, tenant_id: str, provider: str, email_address: str) -> str:
        return f"{tenant_id}:{provider}:{email_address}"

    async def get(
        self, tenant_id: str, provider: str, email_address: str
    ) -> EmailCredential | None:
        async with self._lock:
            return self._credentials.get(self._key(tenant_id, provider, email_address))

    async def save(self, credential: EmailCredential) -> None:
        credential.updated_at = time.time()
        async with self._lock:
            self._credentials[credential.credential_id] = credential

    async def delete(self, tenant_id: str, provider: str, email_address: str) -> bool:
        key = self._key(tenant_id, provider, email_address)
        async with self._lock:
            if key in self._credentials:
                del self._credentials[key]
                return True
            return False

    async def list_for_tenant(self, tenant_id: str) -> list[EmailCredential]:
        async with self._lock:
            return [c for c in self._credentials.values() if c.tenant_id == tenant_id]

    async def list_expiring(self, within_seconds: int = 3600) -> list[EmailCredential]:
        async with self._lock:
            return [
                c
                for c in self._credentials.values()
                if c.is_active and c.needs_refresh(margin_seconds=within_seconds)
            ]

    async def update_last_used(self, tenant_id: str, provider: str, email_address: str) -> None:
        key = self._key(tenant_id, provider, email_address)
        async with self._lock:
            if key in self._credentials:
                self._credentials[key].last_used = datetime.now(timezone.utc)
                self._credentials[key].updated_at = time.time()

    async def record_failure(
        self, tenant_id: str, provider: str, email_address: str, error: str
    ) -> None:
        key = self._key(tenant_id, provider, email_address)
        async with self._lock:
            if key in self._credentials:
                self._credentials[key].failure_count += 1
                self._credentials[key].last_error = error
                self._credentials[key].updated_at = time.time()

    async def reset_failures(self, tenant_id: str, provider: str, email_address: str) -> None:
        key = self._key(tenant_id, provider, email_address)
        async with self._lock:
            if key in self._credentials:
                self._credentials[key].failure_count = 0
                self._credentials[key].last_error = ""
                self._credentials[key].updated_at = time.time()

    def clear(self) -> None:
        """Clear all entries (for testing)."""
        self._credentials.clear()


class SQLiteEmailCredentialStore(EmailCredentialStoreBackend):
    """
    SQLite-backed email credential store.

    Persisted to disk, survives restarts. Suitable for single-instance
    production deployments.
    """

    def __init__(self, db_path: Path | str):
        # SECURITY: Check production guards for SQLite usage
        try:
            from aragora.storage.production_guards import (
                StorageMode,
                require_distributed_store,
            )

            require_distributed_store(
                "email_credential_store",
                StorageMode.SQLITE,
                "Email credential store using SQLite - use PostgreSQL for multi-instance",
            )
        except ImportError:
            pass  # Guards not available, allow SQLite

        self.db_path = Path(resolve_db_path(db_path))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._init_schema()
        logger.info("SQLiteEmailCredentialStore initialized: %s", self.db_path)

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection (thread-safe)."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS email_credentials (
                tenant_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                email_address TEXT NOT NULL,
                access_token TEXT,
                refresh_token TEXT,
                token_expiry TEXT,
                client_id TEXT,
                client_secret TEXT,
                provider_user_id TEXT,
                scopes TEXT,
                is_active INTEGER DEFAULT 1,
                last_used TEXT,
                failure_count INTEGER DEFAULT 0,
                last_error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (tenant_id, provider, email_address)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_email_creds_tenant ON email_credentials(tenant_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_email_creds_expiry ON email_credentials(token_expiry)"
        )
        conn.commit()
        conn.close()

    async def get(
        self, tenant_id: str, provider: str, email_address: str
    ) -> EmailCredential | None:
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                """SELECT tenant_id, provider, email_address, access_token, refresh_token,
                          token_expiry, client_id, client_secret, provider_user_id, scopes,
                          is_active, last_used, failure_count, last_error, created_at, updated_at
                   FROM email_credentials
                   WHERE tenant_id = ? AND provider = ? AND email_address = ?""",
                (tenant_id, provider, email_address),
            )
            row = cursor.fetchone()
            if row:
                return EmailCredential.from_row(row)
            return None

    async def save(self, credential: EmailCredential) -> None:
        credential.updated_at = time.time()
        context = credential.credential_id
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT OR REPLACE INTO email_credentials
                   (tenant_id, provider, email_address, access_token, refresh_token,
                    token_expiry, client_id, client_secret, provider_user_id, scopes,
                    is_active, last_used, failure_count, last_error, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    credential.tenant_id,
                    credential.provider,
                    credential.email_address,
                    _encrypt_token(credential.access_token, context),
                    _encrypt_token(credential.refresh_token, context),
                    credential.token_expiry.isoformat() if credential.token_expiry else None,
                    credential.client_id,
                    _encrypt_token(credential.client_secret, context),
                    credential.provider_user_id,
                    json.dumps(credential.scopes),
                    1 if credential.is_active else 0,
                    credential.last_used.isoformat() if credential.last_used else None,
                    credential.failure_count,
                    credential.last_error,
                    credential.created_at,
                    credential.updated_at,
                ),
            )
            conn.commit()
        logger.debug("Saved email credential for %s", credential.credential_id)

    async def delete(self, tenant_id: str, provider: str, email_address: str) -> bool:
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                "DELETE FROM email_credentials WHERE tenant_id = ? AND provider = ? AND email_address = ?",
                (tenant_id, provider, email_address),
            )
            conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.debug("Deleted email credential for %s:%s:%s", tenant_id, provider, email_address)
            return deleted

    async def list_for_tenant(self, tenant_id: str) -> list[EmailCredential]:
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                """SELECT tenant_id, provider, email_address, access_token, refresh_token,
                          token_expiry, client_id, client_secret, provider_user_id, scopes,
                          is_active, last_used, failure_count, last_error, created_at, updated_at
                   FROM email_credentials WHERE tenant_id = ?""",
                (tenant_id,),
            )
            return [EmailCredential.from_row(row) for row in cursor.fetchall()]

    async def list_expiring(self, within_seconds: int = 3600) -> list[EmailCredential]:
        cutoff = datetime.now(timezone.utc) + timedelta(seconds=within_seconds)
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                """SELECT tenant_id, provider, email_address, access_token, refresh_token,
                          token_expiry, client_id, client_secret, provider_user_id, scopes,
                          is_active, last_used, failure_count, last_error, created_at, updated_at
                   FROM email_credentials
                   WHERE is_active = 1 AND token_expiry IS NOT NULL AND token_expiry < ?""",
                (cutoff.isoformat(),),
            )
            return [EmailCredential.from_row(row) for row in cursor.fetchall()]

    async def update_last_used(self, tenant_id: str, provider: str, email_address: str) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """UPDATE email_credentials
                   SET last_used = ?, updated_at = ?
                   WHERE tenant_id = ? AND provider = ? AND email_address = ?""",
                (now.isoformat(), time.time(), tenant_id, provider, email_address),
            )
            conn.commit()

    async def record_failure(
        self, tenant_id: str, provider: str, email_address: str, error: str
    ) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """UPDATE email_credentials
                   SET failure_count = failure_count + 1, last_error = ?, updated_at = ?
                   WHERE tenant_id = ? AND provider = ? AND email_address = ?""",
                (error, time.time(), tenant_id, provider, email_address),
            )
            conn.commit()

    async def reset_failures(self, tenant_id: str, provider: str, email_address: str) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """UPDATE email_credentials
                   SET failure_count = 0, last_error = '', updated_at = ?
                   WHERE tenant_id = ? AND provider = ? AND email_address = ?""",
                (time.time(), tenant_id, provider, email_address),
            )
            conn.commit()

    async def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


class RedisEmailCredentialStore(EmailCredentialStoreBackend):
    """
    Redis-backed email credential store with SQLite fallback.

    Uses Redis for fast distributed access, with SQLite as durable storage.
    """

    REDIS_PREFIX = "aragora:email:creds"
    REDIS_TTL = 86400 * 7  # 7 days

    def __init__(self, db_path: Path | str, redis_url: str | None = None):
        self._sqlite = SQLiteEmailCredentialStore(db_path)
        self._redis: Any | None = None
        self._redis_url = redis_url or os.environ.get("ARAGORA_REDIS_URL", "redis://localhost:6379")
        self._redis_checked = False
        logger.info("RedisEmailCredentialStore initialized with SQLite fallback")

    def _get_redis(self) -> Any | None:
        """Get Redis client (lazy initialization)."""
        if self._redis_checked:
            return self._redis

        try:
            import redis

            self._redis = redis.from_url(self._redis_url, encoding="utf-8", decode_responses=True)
            self._redis.ping()
            self._redis_checked = True
            logger.info("Redis connected for email credential store")
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.debug("Redis connection failed, using SQLite only: %s: %s", type(e).__name__, e)
            self._redis = None
            self._redis_checked = True
        except Exception as e:  # noqa: BLE001 - redis library custom exceptions may not be importable
            logger.debug("Redis not available, using SQLite only: %s: %s", type(e).__name__, e)
            self._redis = None
            self._redis_checked = True

        return self._redis

    def _cache_key(self, tenant_id: str, provider: str, email_address: str) -> str:
        return f"{self.REDIS_PREFIX}:{tenant_id}:{provider}:{email_address}"

    async def get(
        self, tenant_id: str, provider: str, email_address: str
    ) -> EmailCredential | None:
        redis = self._get_redis()

        if redis is not None:
            try:
                data = redis.get(self._cache_key(tenant_id, provider, email_address))
                if data:
                    return EmailCredential.from_json(data)
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug(
                    "Redis get connection error, falling back to SQLite: %s: %s", type(e).__name__, e
                )
            except (ValueError, TypeError) as e:
                logger.debug(
                    "Redis get deserialization error, falling back to SQLite: %s: %s", type(e).__name__, e
                )

        credential = await self._sqlite.get(tenant_id, provider, email_address)

        # Populate Redis cache if found
        if credential and redis:
            try:
                redis.setex(
                    self._cache_key(tenant_id, provider, email_address),
                    self.REDIS_TTL,
                    credential.to_json(),
                )
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug("Redis cache population failed: %s", e)

        return credential

    async def save(self, credential: EmailCredential) -> None:
        # Always save to SQLite (durable)
        await self._sqlite.save(credential)

        # Update Redis cache
        redis = self._get_redis()
        if redis:
            try:
                redis.setex(
                    self._cache_key(
                        credential.tenant_id, credential.provider, credential.email_address
                    ),
                    self.REDIS_TTL,
                    credential.to_json(),
                )
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug("Redis cache update connection error: %s: %s", type(e).__name__, e)

    async def delete(self, tenant_id: str, provider: str, email_address: str) -> bool:
        redis = self._get_redis()
        if redis:
            try:
                redis.delete(self._cache_key(tenant_id, provider, email_address))
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug("Redis cache deletion failed: %s", e)

        return await self._sqlite.delete(tenant_id, provider, email_address)

    async def list_for_tenant(self, tenant_id: str) -> list[EmailCredential]:
        return await self._sqlite.list_for_tenant(tenant_id)

    async def list_expiring(self, within_seconds: int = 3600) -> list[EmailCredential]:
        return await self._sqlite.list_expiring(within_seconds)

    async def update_last_used(self, tenant_id: str, provider: str, email_address: str) -> None:
        await self._sqlite.update_last_used(tenant_id, provider, email_address)
        # Invalidate cache to force refresh
        redis = self._get_redis()
        if redis:
            try:
                redis.delete(self._cache_key(tenant_id, provider, email_address))
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug("update last used encountered an error: %s", e)

    async def record_failure(
        self, tenant_id: str, provider: str, email_address: str, error: str
    ) -> None:
        await self._sqlite.record_failure(tenant_id, provider, email_address, error)
        # Invalidate cache
        redis = self._get_redis()
        if redis:
            try:
                redis.delete(self._cache_key(tenant_id, provider, email_address))
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug("record failure encountered an error: %s", e)

    async def reset_failures(self, tenant_id: str, provider: str, email_address: str) -> None:
        await self._sqlite.reset_failures(tenant_id, provider, email_address)
        # Invalidate cache
        redis = self._get_redis()
        if redis:
            try:
                redis.delete(self._cache_key(tenant_id, provider, email_address))
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug("reset failures encountered an error: %s", e)

    async def close(self) -> None:
        await self._sqlite.close()
        if self._redis:
            self._redis.close()


class PostgresEmailCredentialStore(EmailCredentialStoreBackend):
    """
    PostgreSQL-backed email credential store.

    Async implementation for production multi-instance deployments
    with horizontal scaling and concurrent writes.
    """

    SCHEMA_NAME = "email_credentials"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS email_credentials (
            tenant_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            email_address TEXT NOT NULL,
            access_token TEXT,
            refresh_token TEXT,
            token_expiry TIMESTAMPTZ,
            client_id TEXT,
            client_secret TEXT,
            provider_user_id TEXT,
            scopes JSONB DEFAULT '[]',
            is_active BOOLEAN DEFAULT TRUE,
            last_used TIMESTAMPTZ,
            failure_count INTEGER DEFAULT 0,
            last_error TEXT,
            created_at DOUBLE PRECISION NOT NULL,
            updated_at DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (tenant_id, provider, email_address)
        );
        CREATE INDEX IF NOT EXISTS idx_email_creds_tenant ON email_credentials(tenant_id);
        CREATE INDEX IF NOT EXISTS idx_email_creds_expiry ON email_credentials(token_expiry)
            WHERE is_active = TRUE;
    """

    def __init__(self, pool: Pool):
        self._pool = pool
        self._initialized = False
        logger.info("PostgresEmailCredentialStore initialized")

    async def initialize(self) -> None:
        """Initialize database schema."""
        if self._initialized:
            return

        async with self._pool.acquire() as conn:
            await conn.execute(self.INITIAL_SCHEMA)

        self._initialized = True
        logger.debug("[%s] Schema initialized", self.SCHEMA_NAME)

    def _row_to_credential(self, row: Any) -> EmailCredential:
        """Convert database row to EmailCredential."""
        tenant_id = row["tenant_id"]
        provider = row["provider"]
        email_address = row["email_address"]
        context = f"{tenant_id}:{provider}:{email_address}"
        return EmailCredential(
            tenant_id=tenant_id,
            provider=provider,
            email_address=email_address,
            access_token=_decrypt_token(row["access_token"] or "", context),
            refresh_token=_decrypt_token(row["refresh_token"] or "", context),
            token_expiry=row["token_expiry"],
            client_id=row["client_id"] or "",
            client_secret=_decrypt_token(row["client_secret"] or "", context),
            provider_user_id=row["provider_user_id"] or "",
            scopes=row["scopes"] if row["scopes"] else [],
            is_active=row["is_active"] if row["is_active"] is not None else True,
            last_used=row["last_used"],
            failure_count=row["failure_count"] or 0,
            last_error=row["last_error"] or "",
            created_at=row["created_at"] or time.time(),
            updated_at=row["updated_at"] or time.time(),
        )

    async def get(
        self, tenant_id: str, provider: str, email_address: str
    ) -> EmailCredential | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT tenant_id, provider, email_address, access_token, refresh_token,
                          token_expiry, client_id, client_secret, provider_user_id, scopes,
                          is_active, last_used, failure_count, last_error, created_at, updated_at
                   FROM email_credentials
                   WHERE tenant_id = $1 AND provider = $2 AND email_address = $3""",
                tenant_id,
                provider,
                email_address,
            )
            if row:
                return self._row_to_credential(row)
            return None

    async def save(self, credential: EmailCredential) -> None:
        credential.updated_at = time.time()
        context = credential.credential_id
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO email_credentials
                   (tenant_id, provider, email_address, access_token, refresh_token,
                    token_expiry, client_id, client_secret, provider_user_id, scopes,
                    is_active, last_used, failure_count, last_error, created_at, updated_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                   ON CONFLICT (tenant_id, provider, email_address) DO UPDATE SET
                    access_token = EXCLUDED.access_token,
                    refresh_token = EXCLUDED.refresh_token,
                    token_expiry = EXCLUDED.token_expiry,
                    client_id = EXCLUDED.client_id,
                    client_secret = EXCLUDED.client_secret,
                    provider_user_id = EXCLUDED.provider_user_id,
                    scopes = EXCLUDED.scopes,
                    is_active = EXCLUDED.is_active,
                    last_used = EXCLUDED.last_used,
                    failure_count = EXCLUDED.failure_count,
                    last_error = EXCLUDED.last_error,
                    updated_at = EXCLUDED.updated_at""",
                credential.tenant_id,
                credential.provider,
                credential.email_address,
                _encrypt_token(credential.access_token, context),
                _encrypt_token(credential.refresh_token, context),
                credential.token_expiry,
                credential.client_id,
                _encrypt_token(credential.client_secret, context),
                credential.provider_user_id,
                json.dumps(credential.scopes),
                credential.is_active,
                credential.last_used,
                credential.failure_count,
                credential.last_error,
                credential.created_at,
                credential.updated_at,
            )
        logger.debug("Saved email credential for %s", credential.credential_id)

    async def delete(self, tenant_id: str, provider: str, email_address: str) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """DELETE FROM email_credentials
                   WHERE tenant_id = $1 AND provider = $2 AND email_address = $3""",
                tenant_id,
                provider,
                email_address,
            )
            deleted = result != "DELETE 0"
            if deleted:
                logger.debug("Deleted email credential for %s:%s:%s", tenant_id, provider, email_address)
            return deleted

    async def list_for_tenant(self, tenant_id: str) -> list[EmailCredential]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT tenant_id, provider, email_address, access_token, refresh_token,
                          token_expiry, client_id, client_secret, provider_user_id, scopes,
                          is_active, last_used, failure_count, last_error, created_at, updated_at
                   FROM email_credentials WHERE tenant_id = $1""",
                tenant_id,
            )
            return [self._row_to_credential(row) for row in rows]

    async def list_expiring(self, within_seconds: int = 3600) -> list[EmailCredential]:
        cutoff = datetime.now(timezone.utc) + timedelta(seconds=within_seconds)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT tenant_id, provider, email_address, access_token, refresh_token,
                          token_expiry, client_id, client_secret, provider_user_id, scopes,
                          is_active, last_used, failure_count, last_error, created_at, updated_at
                   FROM email_credentials
                   WHERE is_active = TRUE AND token_expiry IS NOT NULL AND token_expiry < $1""",
                cutoff,
            )
            return [self._row_to_credential(row) for row in rows]

    async def update_last_used(self, tenant_id: str, provider: str, email_address: str) -> None:
        now = datetime.now(timezone.utc)
        async with self._pool.acquire() as conn:
            await conn.execute(
                """UPDATE email_credentials
                   SET last_used = $1, updated_at = $2
                   WHERE tenant_id = $3 AND provider = $4 AND email_address = $5""",
                now,
                time.time(),
                tenant_id,
                provider,
                email_address,
            )

    async def record_failure(
        self, tenant_id: str, provider: str, email_address: str, error: str
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """UPDATE email_credentials
                   SET failure_count = failure_count + 1, last_error = $1, updated_at = $2
                   WHERE tenant_id = $3 AND provider = $4 AND email_address = $5""",
                error,
                time.time(),
                tenant_id,
                provider,
                email_address,
            )

    async def reset_failures(self, tenant_id: str, provider: str, email_address: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """UPDATE email_credentials
                   SET failure_count = 0, last_error = '', updated_at = $1
                   WHERE tenant_id = $2 AND provider = $3 AND email_address = $4""",
                time.time(),
                tenant_id,
                provider,
                email_address,
            )

    async def close(self) -> None:
        """Close is a no-op for pool-based stores (pool managed externally)."""
        pass


# =============================================================================
# Global Store Factory
# =============================================================================

_email_credential_store: EmailCredentialStoreBackend | None = None
_email_credential_store_lock = threading.Lock()


def get_email_credential_store() -> EmailCredentialStoreBackend:
    """
    Get or create the email credential store.

    Uses environment variables to configure:
    - ARAGORA_DB_BACKEND: "sqlite", "postgres", or "supabase"
    - ARAGORA_EMAIL_CRED_STORE_BACKEND: "memory", "sqlite", "postgres", "redis"
    - ARAGORA_DATA_DIR: Directory for SQLite database
    - ARAGORA_REDIS_URL: Redis connection URL (for redis backend)
    - SUPABASE_URL + SUPABASE_DB_PASSWORD or SUPABASE_POSTGRES_DSN
    - ARAGORA_POSTGRES_DSN or DATABASE_URL

    Returns:
        Configured EmailCredentialStoreBackend instance
    """
    global _email_credential_store

    # Fast path: already initialized
    if _email_credential_store is not None:
        return _email_credential_store

    # Thread-safe initialization
    with _email_credential_store_lock:
        # Double-check after acquiring lock
        if _email_credential_store is not None:
            return _email_credential_store

        # Check store-specific backend first, then global database backend
        backend_type = os.environ.get("ARAGORA_EMAIL_CRED_STORE_BACKEND")
        if not backend_type:
            backend_type = os.environ.get("ARAGORA_DB_BACKEND", "auto")
        backend_type = backend_type.lower()

        # Preserve legacy data directory when configured
        data_dir = None
        try:
            from aragora.persistence.db_config import get_default_data_dir

            data_dir = get_default_data_dir()
        except ImportError:
            data_dir = None

        if backend_type == "redis":
            base_dir = data_dir or Path(".")
            db_path = base_dir / "email_credentials.db"
            logger.info("Using Redis email credential store with SQLite fallback")
            _email_credential_store = RedisEmailCredentialStore(db_path)
            return _email_credential_store

        from aragora.storage.connection_factory import create_persistent_store

        _email_credential_store = create_persistent_store(
            store_name="email_creds",
            sqlite_class=SQLiteEmailCredentialStore,
            postgres_class=PostgresEmailCredentialStore,
            db_filename="email_credentials.db",
            memory_class=InMemoryEmailCredentialStore,
            data_dir=str(data_dir) if data_dir else None,
        )

        return _email_credential_store


def set_email_credential_store(store: EmailCredentialStoreBackend) -> None:
    """Set custom email credential store (for testing)."""
    global _email_credential_store
    _email_credential_store = store
    logger.debug("Email credential store backend set: %s", type(store).__name__)


def reset_email_credential_store() -> None:
    """Reset the global email credential store (for testing)."""
    global _email_credential_store
    _email_credential_store = None


__all__ = [
    "EmailCredential",
    "EmailCredentialStoreBackend",
    "InMemoryEmailCredentialStore",
    "SQLiteEmailCredentialStore",
    "RedisEmailCredentialStore",
    "PostgresEmailCredentialStore",
    "get_email_credential_store",
    "set_email_credential_store",
    "reset_email_credential_store",
    "ENCRYPTED_FIELDS",
    "DEFAULT_REFRESH_MARGIN_SECONDS",
]
