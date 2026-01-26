"""
Integration Configuration Store.

Provides persistent storage for chat platform integration configurations.
Survives server restarts and supports multi-instance deployments via Redis.

Backends:
- InMemoryIntegrationStore: Fast, single-instance only (for testing)
- SQLiteIntegrationStore: Persisted, single-instance (default for production)
- RedisIntegrationStore: Distributed, multi-instance (optional with fallback)

Usage:
    from aragora.storage.integration_store import get_integration_store

    store = get_integration_store()
    await store.save(config)
    config = await store.get("slack", user_id="user123")
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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, cast

if TYPE_CHECKING:
    from asyncpg import Pool

logger = logging.getLogger(__name__)

# Import encryption (optional - graceful degradation if not available)
try:
    from aragora.security.encryption import (
        get_encryption_service,
        is_encryption_required,
        EncryptionError,
        CRYPTO_AVAILABLE,
    )
except ImportError:
    CRYPTO_AVAILABLE = False

    def get_encryption_service():  # type: ignore[misc,no-redef]
        raise RuntimeError("Encryption not available")

    def is_encryption_required() -> bool:  # type: ignore[misc,no-redef]
        """Fallback when security module unavailable - still check env vars."""
        import os

        if os.environ.get("ARAGORA_ENCRYPTION_REQUIRED", "").lower() in ("true", "1", "yes"):
            return True
        if os.environ.get("ARAGORA_ENV") == "production":
            return True
        return False

    class EncryptionError(Exception):  # type: ignore[no-redef]
        """Fallback exception when security module unavailable."""

        def __init__(self, operation: str, reason: str, store: str = ""):
            self.operation = operation
            self.reason = reason
            self.store = store
            super().__init__(
                f"Encryption {operation} failed in {store}: {reason}. "
                f"Set ARAGORA_ENCRYPTION_REQUIRED=false to allow plaintext fallback."
            )


def _record_user_mapping_operation(operation: str, platform: str, found: bool) -> None:
    """Record user mapping operation metric if available."""
    try:
        from aragora.observability.metrics import record_user_mapping_operation  # type: ignore[attr-defined]

        record_user_mapping_operation(operation, platform, found)
    except ImportError:
        pass


def _record_user_mapping_cache_hit(platform: str) -> None:
    """Record user mapping cache hit metric if available."""
    try:
        from aragora.observability.metrics import record_user_mapping_cache_hit  # type: ignore[attr-defined]

        record_user_mapping_cache_hit(platform)
    except ImportError:
        pass


def _record_user_mapping_cache_miss(platform: str) -> None:
    """Record user mapping cache miss metric if available."""
    try:
        from aragora.observability.metrics import record_user_mapping_cache_miss  # type: ignore[attr-defined]

        record_user_mapping_cache_miss(platform)
    except ImportError:
        pass


# =============================================================================
# Integration Types and Models
# =============================================================================

IntegrationType = Literal["slack", "discord", "telegram", "email", "teams", "whatsapp", "matrix"]

VALID_INTEGRATION_TYPES: set[str] = {
    "slack",
    "discord",
    "telegram",
    "email",
    "teams",
    "whatsapp",
    "matrix",
}

# Sensitive keys that should be encrypted or masked
SENSITIVE_KEYS = frozenset(
    [
        "access_token",
        "refresh_token",
        "api_key",
        "bot_token",
        "webhook_url",
        "secret",
        "password",
        "auth_token",
        "sendgrid_api_key",
        "ses_secret_access_key",
        "twilio_auth_token",
        "smtp_password",
    ]
)


def _encrypt_settings(
    settings: Dict[str, Any],
    user_id: str = "default",
    integration_type: str = "",
) -> Dict[str, Any]:
    """
    Encrypt sensitive keys in settings dict before storage.

    Uses Associated Authenticated Data (AAD) to bind ciphertext to user/integration
    context, preventing cross-user/integration attacks.

    Raises:
        EncryptionError: If encryption fails and ARAGORA_ENCRYPTION_REQUIRED is True.
    """
    if not settings:
        return settings

    # Find keys that need encryption and have values
    keys_to_encrypt = [k for k in SENSITIVE_KEYS if k in settings and settings[k]]
    if not keys_to_encrypt:
        return settings

    if not CRYPTO_AVAILABLE:
        if is_encryption_required():
            raise EncryptionError(
                "encrypt",
                "cryptography library not available",
                "integration_store",
            )
        return settings

    try:
        service = get_encryption_service()
        # AAD binds ciphertext to this specific user + integration
        aad = f"{user_id}:{integration_type}"
        encrypted = service.encrypt_fields(settings, keys_to_encrypt, aad)
        logger.debug(f"Encrypted {len(keys_to_encrypt)} sensitive fields for {integration_type}")
        return encrypted
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        if is_encryption_required():
            raise EncryptionError(
                "encrypt",
                str(e),
                "integration_store",
            ) from e
        logger.warning(f"Encryption unavailable, storing unencrypted: {e}")
        return settings
    except RuntimeError as e:
        if is_encryption_required():
            raise EncryptionError(
                "encrypt",
                str(e),
                "integration_store",
            ) from e
        logger.warning(f"Encryption service error, storing unencrypted: {e}")
        return settings


def _decrypt_settings(
    settings: Dict[str, Any],
    user_id: str = "default",
    integration_type: str = "",
) -> Dict[str, Any]:
    """
    Decrypt sensitive keys, handling legacy unencrypted data.

    AAD must match what was used during encryption.
    """
    if not CRYPTO_AVAILABLE or not settings:
        return settings

    # Check for encryption markers - if none present, it's legacy data
    encrypted_keys = [
        k
        for k in SENSITIVE_KEYS
        if k in settings and isinstance(settings.get(k), dict) and settings[k].get("_encrypted")
    ]
    if not encrypted_keys:
        return settings  # Legacy unencrypted data - return as-is

    try:
        service = get_encryption_service()
        aad = f"{user_id}:{integration_type}"
        decrypted = service.decrypt_fields(settings, encrypted_keys, aad)
        logger.debug(f"Decrypted {len(encrypted_keys)} fields for {integration_type}")
        return decrypted
    except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as e:
        logger.warning(f"Decryption failed for {integration_type}: {e}")
        return settings


@dataclass
class UserIdMapping:
    """Cross-platform user identity mapping."""

    email: str
    platform: str  # "slack", "discord", "teams", etc.
    platform_user_id: str
    display_name: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    user_id: str = "default"  # Owner/tenant

    def to_dict(self) -> dict:
        """Convert to dict."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "UserIdMapping":
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def from_row(cls, row: tuple) -> "UserIdMapping":
        """Create from database row."""
        return cls(
            email=row[0],
            platform=row[1],
            platform_user_id=row[2],
            display_name=row[3],
            created_at=row[4],
            updated_at=row[5],
            user_id=row[6],
        )


@dataclass
class IntegrationConfig:
    """Configuration for a chat platform integration."""

    type: str
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Notification settings
    notify_on_consensus: bool = True
    notify_on_debate_end: bool = True
    notify_on_error: bool = False
    notify_on_leaderboard: bool = False

    # Provider-specific settings (stored as dict)
    settings: Dict[str, Any] = field(default_factory=dict)

    # Delivery tracking
    messages_sent: int = 0
    errors_24h: int = 0
    last_activity: Optional[float] = None
    last_error: Optional[str] = None

    # Owner (for multi-tenant)
    user_id: Optional[str] = None
    workspace_id: Optional[str] = None

    def to_dict(self, include_secrets: bool = False) -> dict:
        """Convert to dict, optionally excluding secrets."""
        result = asdict(self)
        if not include_secrets:
            settings = result.get("settings", {})
            for key in SENSITIVE_KEYS:
                if key in settings and settings[key]:
                    settings[key] = "••••••••"
            result["settings"] = settings
        return result

    def to_json(self) -> str:
        """Serialize to JSON for storage."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "IntegrationConfig":
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def from_row(cls, row: tuple) -> "IntegrationConfig":
        """Create from database row (settings decryption done at store level)."""
        return cls(
            type=row[0],
            enabled=bool(row[1]),
            created_at=row[2],
            updated_at=row[3],
            notify_on_consensus=bool(row[4]),
            notify_on_debate_end=bool(row[5]),
            notify_on_error=bool(row[6]),
            notify_on_leaderboard=bool(row[7]),
            settings=json.loads(row[8]) if row[8] else {},
            messages_sent=row[9] or 0,
            errors_24h=row[10] or 0,
            last_activity=row[11],
            last_error=row[12],
            user_id=row[13],
            workspace_id=row[14],
        )

    @property
    def status(self) -> str:
        """Get integration status."""
        if not self.enabled:
            return "disconnected"
        if self.errors_24h > 5:
            return "degraded"
        if self.last_activity:
            return "connected"
        return "not_configured"


def _make_key(integration_type: str, user_id: str = "default") -> str:
    """Generate storage key for integration."""
    return f"{user_id}:{integration_type}"


# =============================================================================
# Abstract Backend
# =============================================================================


class IntegrationStoreBackend(ABC):
    """Abstract base for integration storage backends."""

    @abstractmethod
    async def get(
        self, integration_type: str, user_id: str = "default"
    ) -> Optional[IntegrationConfig]:
        """Get integration configuration."""
        pass

    @abstractmethod
    async def save(self, config: IntegrationConfig) -> None:
        """Save integration configuration."""
        pass

    @abstractmethod
    async def delete(self, integration_type: str, user_id: str = "default") -> bool:
        """Delete integration configuration. Returns True if deleted."""
        pass

    @abstractmethod
    async def list_for_user(self, user_id: str = "default") -> List[IntegrationConfig]:
        """List all integrations for a user."""
        pass

    @abstractmethod
    async def list_all(self) -> List[IntegrationConfig]:
        """List all integrations (admin use)."""
        pass

    # User ID mapping methods
    @abstractmethod
    async def get_user_mapping(
        self, email: str, platform: str, user_id: str = "default"
    ) -> Optional[UserIdMapping]:
        """Get user ID mapping for a platform."""
        pass

    @abstractmethod
    async def save_user_mapping(self, mapping: UserIdMapping) -> None:
        """Save user ID mapping."""
        pass

    @abstractmethod
    async def delete_user_mapping(
        self, email: str, platform: str, user_id: str = "default"
    ) -> bool:
        """Delete user ID mapping."""
        pass

    @abstractmethod
    async def list_user_mappings(
        self, platform: Optional[str] = None, user_id: str = "default"
    ) -> List[UserIdMapping]:
        """List user ID mappings, optionally filtered by platform."""
        pass

    async def close(self) -> None:
        """Close connections (optional to implement)."""
        pass


# =============================================================================
# In-Memory Backend (for testing)
# =============================================================================


class InMemoryIntegrationStore(IntegrationStoreBackend):
    """
    Thread-safe in-memory integration store.

    Fast but not shared across restarts. Suitable for development/testing.
    """

    def __init__(self) -> None:
        self._store: Dict[str, IntegrationConfig] = {}
        self._mappings: Dict[str, UserIdMapping] = {}
        self._lock = threading.RLock()

    async def get(
        self, integration_type: str, user_id: str = "default"
    ) -> Optional[IntegrationConfig]:
        key = _make_key(integration_type, user_id)
        with self._lock:
            return self._store.get(key)

    async def save(self, config: IntegrationConfig) -> None:
        key = _make_key(config.type, config.user_id or "default")
        config.updated_at = time.time()
        with self._lock:
            self._store[key] = config

    async def delete(self, integration_type: str, user_id: str = "default") -> bool:
        key = _make_key(integration_type, user_id)
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    async def list_for_user(self, user_id: str = "default") -> List[IntegrationConfig]:
        prefix = f"{user_id}:"
        with self._lock:
            return [v for k, v in self._store.items() if k.startswith(prefix)]

    async def list_all(self) -> List[IntegrationConfig]:
        with self._lock:
            return list(self._store.values())

    async def get_user_mapping(
        self, email: str, platform: str, user_id: str = "default"
    ) -> Optional[UserIdMapping]:
        key = f"{user_id}:{platform}:{email}"
        with self._lock:
            return self._mappings.get(key)

    async def save_user_mapping(self, mapping: UserIdMapping) -> None:
        key = f"{mapping.user_id}:{mapping.platform}:{mapping.email}"
        mapping.updated_at = time.time()
        with self._lock:
            self._mappings[key] = mapping

    async def delete_user_mapping(
        self, email: str, platform: str, user_id: str = "default"
    ) -> bool:
        key = f"{user_id}:{platform}:{email}"
        with self._lock:
            if key in self._mappings:
                del self._mappings[key]
                return True
            return False

    async def list_user_mappings(
        self, platform: Optional[str] = None, user_id: str = "default"
    ) -> List[UserIdMapping]:
        with self._lock:
            result = []
            for key, mapping in self._mappings.items():
                if mapping.user_id == user_id:
                    if platform is None or mapping.platform == platform:
                        result.append(mapping)
            return result

    def clear(self) -> None:
        """Clear all entries (for testing)."""
        with self._lock:
            self._store.clear()
            self._mappings.clear()


# =============================================================================
# SQLite Backend
# =============================================================================


class SQLiteIntegrationStore(IntegrationStoreBackend):
    """
    SQLite-backed integration store.

    Persisted to disk, survives restarts. Suitable for single-instance
    production deployments.

    Raises:
        DistributedStateError: In production if PostgreSQL is not available
    """

    def __init__(self, db_path: Path | str):
        # SECURITY: Check production guards for SQLite usage
        try:
            from aragora.storage.production_guards import (
                require_distributed_store,
                StorageMode,
            )

            require_distributed_store(
                "integration_store",
                StorageMode.SQLITE,
                "Integration store using SQLite - use PostgreSQL for multi-instance deployments",
            )
        except ImportError:
            pass  # Guards not available, allow SQLite

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()
        logger.info(f"SQLiteIntegrationStore initialized: {self.db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return cast(sqlite3.Connection, self._local.conn)

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS integrations (
                integration_type TEXT NOT NULL,
                user_id TEXT NOT NULL DEFAULT 'default',
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                notify_on_consensus INTEGER DEFAULT 1,
                notify_on_debate_end INTEGER DEFAULT 1,
                notify_on_error INTEGER DEFAULT 0,
                notify_on_leaderboard INTEGER DEFAULT 0,
                settings_json TEXT,
                messages_sent INTEGER DEFAULT 0,
                errors_24h INTEGER DEFAULT 0,
                last_activity REAL,
                last_error TEXT,
                workspace_id TEXT,
                PRIMARY KEY (user_id, integration_type)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_integrations_user ON integrations(user_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_integrations_type ON integrations(integration_type)"
        )

        # User ID mappings table (cross-platform identity resolution)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_id_mappings (
                email TEXT NOT NULL,
                platform TEXT NOT NULL,
                platform_user_id TEXT NOT NULL,
                display_name TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                user_id TEXT NOT NULL DEFAULT 'default',
                PRIMARY KEY (user_id, platform, email)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mappings_email ON user_id_mappings(email)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mappings_platform ON user_id_mappings(platform)"
        )
        conn.commit()
        conn.close()

    async def get(
        self, integration_type: str, user_id: str = "default"
    ) -> Optional[IntegrationConfig]:
        conn = self._get_conn()
        cursor = conn.execute(
            """SELECT integration_type, enabled, created_at, updated_at,
                      notify_on_consensus, notify_on_debate_end, notify_on_error,
                      notify_on_leaderboard, settings_json, messages_sent,
                      errors_24h, last_activity, last_error, user_id, workspace_id
               FROM integrations WHERE user_id = ? AND integration_type = ?""",
            (user_id, integration_type),
        )
        row = cursor.fetchone()
        if row:
            config = IntegrationConfig.from_row(row)
            # Decrypt settings with AAD for integrity verification
            config.settings = _decrypt_settings(config.settings, user_id, integration_type)
            return config
        return None

    async def save(self, config: IntegrationConfig) -> None:
        conn = self._get_conn()
        config.updated_at = time.time()
        user_id = config.user_id or "default"
        # Encrypt settings with AAD binding to user + integration type
        encrypted_settings = _encrypt_settings(config.settings, user_id, config.type)
        conn.execute(
            """INSERT OR REPLACE INTO integrations
               (integration_type, user_id, enabled, created_at, updated_at,
                notify_on_consensus, notify_on_debate_end, notify_on_error,
                notify_on_leaderboard, settings_json, messages_sent, errors_24h,
                last_activity, last_error, workspace_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                config.type,
                user_id,
                int(config.enabled),
                config.created_at,
                config.updated_at,
                int(config.notify_on_consensus),
                int(config.notify_on_debate_end),
                int(config.notify_on_error),
                int(config.notify_on_leaderboard),
                json.dumps(encrypted_settings),
                config.messages_sent,
                config.errors_24h,
                config.last_activity,
                config.last_error,
                config.workspace_id,
            ),
        )
        conn.commit()
        logger.debug(f"Saved integration: {config.type} for user {user_id}")

    async def delete(self, integration_type: str, user_id: str = "default") -> bool:
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM integrations WHERE user_id = ? AND integration_type = ?",
            (user_id, integration_type),
        )
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.debug(f"Deleted integration: {integration_type} for user {user_id}")
        return deleted

    async def list_for_user(self, user_id: str = "default") -> List[IntegrationConfig]:
        conn = self._get_conn()
        cursor = conn.execute(
            """SELECT integration_type, enabled, created_at, updated_at,
                      notify_on_consensus, notify_on_debate_end, notify_on_error,
                      notify_on_leaderboard, settings_json, messages_sent,
                      errors_24h, last_activity, last_error, user_id, workspace_id
               FROM integrations WHERE user_id = ?""",
            (user_id,),
        )
        configs = []
        for row in cursor.fetchall():
            config = IntegrationConfig.from_row(row)
            config.settings = _decrypt_settings(config.settings, user_id, config.type)
            configs.append(config)
        return configs

    async def list_all(self) -> List[IntegrationConfig]:
        conn = self._get_conn()
        cursor = conn.execute(
            """SELECT integration_type, enabled, created_at, updated_at,
                      notify_on_consensus, notify_on_debate_end, notify_on_error,
                      notify_on_leaderboard, settings_json, messages_sent,
                      errors_24h, last_activity, last_error, user_id, workspace_id
               FROM integrations"""
        )
        configs = []
        for row in cursor.fetchall():
            config = IntegrationConfig.from_row(row)
            config.settings = _decrypt_settings(
                config.settings, config.user_id or "default", config.type
            )
            configs.append(config)
        return configs

    async def get_user_mapping(
        self, email: str, platform: str, user_id: str = "default"
    ) -> Optional[UserIdMapping]:
        conn = self._get_conn()
        cursor = conn.execute(
            """SELECT email, platform, platform_user_id, display_name,
                      created_at, updated_at, user_id
               FROM user_id_mappings
               WHERE user_id = ? AND platform = ? AND email = ?""",
            (user_id, platform, email),
        )
        row = cursor.fetchone()
        if row:
            return UserIdMapping.from_row(row)
        return None

    async def save_user_mapping(self, mapping: UserIdMapping) -> None:
        conn = self._get_conn()
        mapping.updated_at = time.time()
        conn.execute(
            """INSERT OR REPLACE INTO user_id_mappings
               (email, platform, platform_user_id, display_name,
                created_at, updated_at, user_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                mapping.email,
                mapping.platform,
                mapping.platform_user_id,
                mapping.display_name,
                mapping.created_at,
                mapping.updated_at,
                mapping.user_id,
            ),
        )
        conn.commit()
        _record_user_mapping_operation("save", mapping.platform, True)
        logger.debug(f"Saved user mapping: {mapping.email} -> {mapping.platform}")

    async def delete_user_mapping(
        self, email: str, platform: str, user_id: str = "default"
    ) -> bool:
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM user_id_mappings WHERE user_id = ? AND platform = ? AND email = ?",
            (user_id, platform, email),
        )
        conn.commit()
        deleted = cursor.rowcount > 0
        _record_user_mapping_operation("delete", platform, deleted)
        return deleted

    async def list_user_mappings(
        self, platform: Optional[str] = None, user_id: str = "default"
    ) -> List[UserIdMapping]:
        conn = self._get_conn()
        if platform:
            cursor = conn.execute(
                """SELECT email, platform, platform_user_id, display_name,
                          created_at, updated_at, user_id
                   FROM user_id_mappings
                   WHERE user_id = ? AND platform = ?""",
                (user_id, platform),
            )
        else:
            cursor = conn.execute(
                """SELECT email, platform, platform_user_id, display_name,
                          created_at, updated_at, user_id
                   FROM user_id_mappings
                   WHERE user_id = ?""",
                (user_id,),
            )
        return [UserIdMapping.from_row(row) for row in cursor.fetchall()]

    async def close(self) -> None:
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn


# =============================================================================
# Redis Backend (with SQLite fallback)
# =============================================================================


class RedisIntegrationStore(IntegrationStoreBackend):
    """
    Redis-backed integration store with SQLite fallback.

    Uses Redis for fast distributed access, with SQLite as durable storage.
    This enables multi-instance deployments while ensuring persistence.
    """

    REDIS_PREFIX = "aragora:integrations"
    REDIS_TTL = 86400  # 24 hours

    def __init__(self, db_path: Path | str, redis_url: Optional[str] = None):
        self._sqlite = SQLiteIntegrationStore(db_path)
        self._redis: Optional[Any] = None
        self._redis_url = redis_url or os.environ.get("ARAGORA_REDIS_URL", "redis://localhost:6379")
        self._redis_checked = False
        logger.info("RedisIntegrationStore initialized with SQLite fallback")

    def _get_redis(self) -> Optional[Any]:
        """Get Redis client (lazy initialization)."""
        if self._redis_checked:
            return self._redis

        try:
            import redis

            self._redis = redis.from_url(self._redis_url, encoding="utf-8", decode_responses=True)
            # Test connection
            self._redis.ping()
            self._redis_checked = True
            logger.info("Redis connected for integration store")
        except ImportError as e:
            logger.debug(f"Redis package not installed: {e}")
            self._redis = None
            self._redis_checked = True
        except Exception as e:
            # Catch all Redis connection errors (redis.exceptions.ConnectionError, etc.)
            logger.debug(f"Redis not available, using SQLite only: {e}")
            self._redis = None
            self._redis_checked = True

        return self._redis

    def _redis_key(self, integration_type: str, user_id: str) -> str:
        return f"{self.REDIS_PREFIX}:{user_id}:{integration_type}"

    async def get(
        self, integration_type: str, user_id: str = "default"
    ) -> Optional[IntegrationConfig]:
        redis = self._get_redis()

        # Try Redis first
        if redis is not None:
            try:
                key = self._redis_key(integration_type, user_id)
                data = redis.get(key)
                if data:
                    return IntegrationConfig.from_json(data)
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug(f"Redis get failed (connection error), falling back to SQLite: {e}")
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.debug(f"Redis get failed (data error), falling back to SQLite: {e}")

        # Fall back to SQLite
        config = await self._sqlite.get(integration_type, user_id)

        # Populate Redis cache if found
        if config and redis:
            try:
                key = self._redis_key(integration_type, user_id)
                redis.setex(key, self.REDIS_TTL, config.to_json())
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug(f"Redis cache population failed (connection issue): {e}")
            except (TypeError, ValueError) as e:
                logger.debug(f"Redis cache population failed (serialization): {e}")

        return config

    async def save(self, config: IntegrationConfig) -> None:
        user_id = config.user_id or "default"

        # Always save to SQLite (durable)
        await self._sqlite.save(config)

        # Update Redis cache
        redis = self._get_redis()
        if redis:
            try:
                key = self._redis_key(config.type, user_id)
                redis.setex(key, self.REDIS_TTL, config.to_json())
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug(f"Redis cache update failed (connection issue): {e}")
            except (TypeError, ValueError) as e:
                logger.debug(f"Redis cache update failed (serialization): {e}")

    async def delete(self, integration_type: str, user_id: str = "default") -> bool:
        # Delete from both stores
        redis = self._get_redis()
        if redis:
            try:
                key = self._redis_key(integration_type, user_id)
                redis.delete(key)
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug(f"Redis cache delete failed (connection issue): {e}")
            except (TypeError, ValueError) as e:
                logger.debug(f"Redis cache delete failed (key error): {e}")

        return await self._sqlite.delete(integration_type, user_id)

    async def list_for_user(self, user_id: str = "default") -> List[IntegrationConfig]:
        # Always use SQLite for list operations (authoritative)
        return await self._sqlite.list_for_user(user_id)

    async def list_all(self) -> List[IntegrationConfig]:
        return await self._sqlite.list_all()

    def _mapping_redis_key(self, email: str, platform: str, user_id: str) -> str:
        return f"{self.REDIS_PREFIX}:mapping:{user_id}:{platform}:{email}"

    async def get_user_mapping(
        self, email: str, platform: str, user_id: str = "default"
    ) -> Optional[UserIdMapping]:
        redis = self._get_redis()

        # Try Redis first
        if redis is not None:
            try:
                key = self._mapping_redis_key(email, platform, user_id)
                data = redis.get(key)
                if data:
                    _record_user_mapping_cache_hit(platform)
                    _record_user_mapping_operation("get", platform, True)
                    return UserIdMapping.from_json(data)
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug(f"Redis mapping get failed (connection error): {e}")
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.debug(f"Redis mapping get failed (data error): {e}")

        # Fall back to SQLite (cache miss)
        _record_user_mapping_cache_miss(platform)
        mapping = await self._sqlite.get_user_mapping(email, platform, user_id)
        _record_user_mapping_operation("get", platform, mapping is not None)

        # Populate Redis cache if found
        if mapping and redis:
            try:
                key = self._mapping_redis_key(email, platform, user_id)
                redis.setex(key, self.REDIS_TTL, mapping.to_json())
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug(f"Redis mapping cache population failed (connection issue): {e}")
            except (TypeError, ValueError) as e:
                logger.debug(f"Redis mapping cache population failed (serialization): {e}")

        return mapping

    async def save_user_mapping(self, mapping: UserIdMapping) -> None:
        # Always save to SQLite (durable)
        await self._sqlite.save_user_mapping(mapping)

        # Update Redis cache
        redis = self._get_redis()
        if redis:
            try:
                key = self._mapping_redis_key(mapping.email, mapping.platform, mapping.user_id)
                redis.setex(key, self.REDIS_TTL, mapping.to_json())
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug(f"Redis mapping cache update failed (connection issue): {e}")
            except (TypeError, ValueError) as e:
                logger.debug(f"Redis mapping cache update failed (serialization): {e}")

    async def delete_user_mapping(
        self, email: str, platform: str, user_id: str = "default"
    ) -> bool:
        redis = self._get_redis()
        if redis:
            try:
                key = self._mapping_redis_key(email, platform, user_id)
                redis.delete(key)
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug(f"Redis mapping delete failed (connection issue): {e}")
            except (TypeError, ValueError) as e:
                logger.debug(f"Redis mapping delete failed (key error): {e}")

        return await self._sqlite.delete_user_mapping(email, platform, user_id)

    async def list_user_mappings(
        self, platform: Optional[str] = None, user_id: str = "default"
    ) -> List[UserIdMapping]:
        # Always use SQLite for list operations (authoritative)
        return await self._sqlite.list_user_mappings(platform, user_id)

    async def close(self) -> None:
        await self._sqlite.close()
        if self._redis:
            self._redis.close()


# =============================================================================
# PostgreSQL Backend
# =============================================================================


class PostgresIntegrationStore(IntegrationStoreBackend):
    """
    PostgreSQL-backed integration store.

    Async implementation for production multi-instance deployments
    with horizontal scaling and concurrent writes.
    """

    SCHEMA_NAME = "integrations"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS integrations (
            integration_type TEXT NOT NULL,
            user_id TEXT NOT NULL DEFAULT 'default',
            enabled BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            notify_on_consensus BOOLEAN DEFAULT TRUE,
            notify_on_debate_end BOOLEAN DEFAULT TRUE,
            notify_on_error BOOLEAN DEFAULT FALSE,
            notify_on_leaderboard BOOLEAN DEFAULT FALSE,
            settings_json JSONB,
            messages_sent INTEGER DEFAULT 0,
            errors_24h INTEGER DEFAULT 0,
            last_activity TIMESTAMPTZ,
            last_error TEXT,
            workspace_id TEXT,
            PRIMARY KEY (user_id, integration_type)
        );
        CREATE INDEX IF NOT EXISTS idx_integrations_user ON integrations(user_id);
        CREATE INDEX IF NOT EXISTS idx_integrations_type ON integrations(integration_type);

        CREATE TABLE IF NOT EXISTS user_id_mappings (
            email TEXT NOT NULL,
            platform TEXT NOT NULL,
            platform_user_id TEXT NOT NULL,
            display_name TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            user_id TEXT NOT NULL DEFAULT 'default',
            PRIMARY KEY (user_id, platform, email)
        );
        CREATE INDEX IF NOT EXISTS idx_mappings_email ON user_id_mappings(email);
        CREATE INDEX IF NOT EXISTS idx_mappings_platform ON user_id_mappings(platform);
    """

    def __init__(self, pool: "Pool"):
        self._pool = pool
        self._initialized = False
        logger.info("PostgresIntegrationStore initialized")

    async def initialize(self) -> None:
        """Initialize database schema."""
        if self._initialized:
            return

        async with self._pool.acquire() as conn:
            await conn.execute(self.INITIAL_SCHEMA)

        self._initialized = True
        logger.debug(f"[{self.SCHEMA_NAME}] Schema initialized")

    def _row_to_config(self, row: Any) -> IntegrationConfig:
        """Convert database row to IntegrationConfig (settings decryption done at store level)."""
        return IntegrationConfig(
            type=row["integration_type"],
            enabled=bool(row["enabled"]),
            created_at=row["created_at"] or time.time(),
            updated_at=row["updated_at"] or time.time(),
            notify_on_consensus=bool(row["notify_on_consensus"]),
            notify_on_debate_end=bool(row["notify_on_debate_end"]),
            notify_on_error=bool(row["notify_on_error"]),
            notify_on_leaderboard=bool(row["notify_on_leaderboard"]),
            settings=json.loads(row["settings_json"]) if row["settings_json"] else {},
            messages_sent=row["messages_sent"] or 0,
            errors_24h=row["errors_24h"] or 0,
            last_activity=row["last_activity"],
            last_error=row["last_error"],
            user_id=row["user_id"],
            workspace_id=row["workspace_id"],
        )

    def _row_to_mapping(self, row: Any) -> UserIdMapping:
        """Convert database row to UserIdMapping."""
        return UserIdMapping(
            email=row["email"],
            platform=row["platform"],
            platform_user_id=row["platform_user_id"],
            display_name=row["display_name"],
            created_at=row["created_at"] or time.time(),
            updated_at=row["updated_at"] or time.time(),
            user_id=row["user_id"],
        )

    async def get(
        self, integration_type: str, user_id: str = "default"
    ) -> Optional[IntegrationConfig]:
        """Get integration configuration (async)."""
        return await self.get_async(integration_type, user_id)

    async def get_async(
        self, integration_type: str, user_id: str = "default"
    ) -> Optional[IntegrationConfig]:
        """Get integration configuration asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT integration_type, enabled,
                          EXTRACT(EPOCH FROM created_at) as created_at,
                          EXTRACT(EPOCH FROM updated_at) as updated_at,
                          notify_on_consensus, notify_on_debate_end, notify_on_error,
                          notify_on_leaderboard, settings_json, messages_sent,
                          errors_24h,
                          EXTRACT(EPOCH FROM last_activity) as last_activity,
                          last_error, user_id, workspace_id
                   FROM integrations WHERE user_id = $1 AND integration_type = $2""",
                user_id,
                integration_type,
            )
            if row:
                config = self._row_to_config(row)
                # Decrypt settings with AAD for integrity verification
                config.settings = _decrypt_settings(config.settings, user_id, integration_type)
                return config
            return None

    def get_sync(
        self, integration_type: str, user_id: str = "default"
    ) -> Optional[IntegrationConfig]:
        """Get integration configuration (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(
            self.get_async(integration_type, user_id)
        )

    async def save(self, config: IntegrationConfig) -> None:
        """Save integration configuration (async)."""
        await self.save_async(config)

    async def save_async(self, config: IntegrationConfig) -> None:
        """Save integration configuration asynchronously."""
        config.updated_at = time.time()
        user_id = config.user_id or "default"
        # Encrypt settings with AAD binding to user + integration type
        encrypted_settings = _encrypt_settings(config.settings, user_id, config.type)

        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO integrations
                   (integration_type, user_id, enabled, created_at, updated_at,
                    notify_on_consensus, notify_on_debate_end, notify_on_error,
                    notify_on_leaderboard, settings_json, messages_sent, errors_24h,
                    last_activity, last_error, workspace_id)
                   VALUES ($1, $2, $3, to_timestamp($4), to_timestamp($5),
                           $6, $7, $8, $9, $10, $11, $12,
                           CASE WHEN $13::float IS NOT NULL THEN to_timestamp($13) ELSE NULL END,
                           $14, $15)
                   ON CONFLICT (user_id, integration_type) DO UPDATE SET
                    enabled = EXCLUDED.enabled,
                    updated_at = EXCLUDED.updated_at,
                    notify_on_consensus = EXCLUDED.notify_on_consensus,
                    notify_on_debate_end = EXCLUDED.notify_on_debate_end,
                    notify_on_error = EXCLUDED.notify_on_error,
                    notify_on_leaderboard = EXCLUDED.notify_on_leaderboard,
                    settings_json = EXCLUDED.settings_json,
                    messages_sent = EXCLUDED.messages_sent,
                    errors_24h = EXCLUDED.errors_24h,
                    last_activity = EXCLUDED.last_activity,
                    last_error = EXCLUDED.last_error,
                    workspace_id = EXCLUDED.workspace_id""",
                config.type,
                user_id,
                config.enabled,
                config.created_at,
                config.updated_at,
                config.notify_on_consensus,
                config.notify_on_debate_end,
                config.notify_on_error,
                config.notify_on_leaderboard,
                json.dumps(encrypted_settings),
                config.messages_sent,
                config.errors_24h,
                config.last_activity,
                config.last_error,
                config.workspace_id,
            )
        logger.debug(f"Saved integration: {config.type} for user {user_id}")

    def save_sync(self, config: IntegrationConfig) -> None:
        """Save integration configuration (sync wrapper for async)."""
        asyncio.get_event_loop().run_until_complete(self.save_async(config))

    async def delete(self, integration_type: str, user_id: str = "default") -> bool:
        """Delete integration configuration (async)."""
        return await self.delete_async(integration_type, user_id)

    async def delete_async(self, integration_type: str, user_id: str = "default") -> bool:
        """Delete integration configuration asynchronously."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM integrations WHERE user_id = $1 AND integration_type = $2",
                user_id,
                integration_type,
            )
            deleted = result != "DELETE 0"
            if deleted:
                logger.debug(f"Deleted integration: {integration_type} for user {user_id}")
            return deleted

    def delete_sync(self, integration_type: str, user_id: str = "default") -> bool:
        """Delete integration configuration (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(
            self.delete_async(integration_type, user_id)
        )

    async def list_for_user(self, user_id: str = "default") -> List[IntegrationConfig]:
        """List all integrations for a user (async)."""
        return await self.list_for_user_async(user_id)

    async def list_for_user_async(self, user_id: str = "default") -> List[IntegrationConfig]:
        """List all integrations for a user asynchronously."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT integration_type, enabled,
                          EXTRACT(EPOCH FROM created_at) as created_at,
                          EXTRACT(EPOCH FROM updated_at) as updated_at,
                          notify_on_consensus, notify_on_debate_end, notify_on_error,
                          notify_on_leaderboard, settings_json, messages_sent,
                          errors_24h,
                          EXTRACT(EPOCH FROM last_activity) as last_activity,
                          last_error, user_id, workspace_id
                   FROM integrations WHERE user_id = $1""",
                user_id,
            )
            configs = []
            for row in rows:
                config = self._row_to_config(row)
                config.settings = _decrypt_settings(config.settings, user_id, config.type)
                configs.append(config)
            return configs

    def list_for_user_sync(self, user_id: str = "default") -> List[IntegrationConfig]:
        """List all integrations for a user (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(self.list_for_user_async(user_id))

    async def list_all(self) -> List[IntegrationConfig]:
        """List all integrations (async)."""
        return await self.list_all_async()

    async def list_all_async(self) -> List[IntegrationConfig]:
        """List all integrations asynchronously."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT integration_type, enabled,
                          EXTRACT(EPOCH FROM created_at) as created_at,
                          EXTRACT(EPOCH FROM updated_at) as updated_at,
                          notify_on_consensus, notify_on_debate_end, notify_on_error,
                          notify_on_leaderboard, settings_json, messages_sent,
                          errors_24h,
                          EXTRACT(EPOCH FROM last_activity) as last_activity,
                          last_error, user_id, workspace_id
                   FROM integrations"""
            )
            configs = []
            for row in rows:
                config = self._row_to_config(row)
                config.settings = _decrypt_settings(
                    config.settings, config.user_id or "default", config.type
                )
                configs.append(config)
            return configs

    def list_all_sync(self) -> List[IntegrationConfig]:
        """List all integrations (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(self.list_all_async())

    async def get_user_mapping(
        self, email: str, platform: str, user_id: str = "default"
    ) -> Optional[UserIdMapping]:
        """Get user ID mapping for a platform (async)."""
        return await self.get_user_mapping_async(email, platform, user_id)

    async def get_user_mapping_async(
        self, email: str, platform: str, user_id: str = "default"
    ) -> Optional[UserIdMapping]:
        """Get user ID mapping for a platform asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT email, platform, platform_user_id, display_name,
                          EXTRACT(EPOCH FROM created_at) as created_at,
                          EXTRACT(EPOCH FROM updated_at) as updated_at,
                          user_id
                   FROM user_id_mappings
                   WHERE user_id = $1 AND platform = $2 AND email = $3""",
                user_id,
                platform,
                email,
            )
            if row:
                _record_user_mapping_operation("get", platform, True)
                return self._row_to_mapping(row)
            _record_user_mapping_operation("get", platform, False)
            return None

    def get_user_mapping_sync(
        self, email: str, platform: str, user_id: str = "default"
    ) -> Optional[UserIdMapping]:
        """Get user ID mapping (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(
            self.get_user_mapping_async(email, platform, user_id)
        )

    async def save_user_mapping(self, mapping: UserIdMapping) -> None:
        """Save user ID mapping (async)."""
        await self.save_user_mapping_async(mapping)

    async def save_user_mapping_async(self, mapping: UserIdMapping) -> None:
        """Save user ID mapping asynchronously."""
        mapping.updated_at = time.time()

        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO user_id_mappings
                   (email, platform, platform_user_id, display_name,
                    created_at, updated_at, user_id)
                   VALUES ($1, $2, $3, $4, to_timestamp($5), to_timestamp($6), $7)
                   ON CONFLICT (user_id, platform, email) DO UPDATE SET
                    platform_user_id = EXCLUDED.platform_user_id,
                    display_name = EXCLUDED.display_name,
                    updated_at = EXCLUDED.updated_at""",
                mapping.email,
                mapping.platform,
                mapping.platform_user_id,
                mapping.display_name,
                mapping.created_at,
                mapping.updated_at,
                mapping.user_id,
            )
        _record_user_mapping_operation("save", mapping.platform, True)
        logger.debug(f"Saved user mapping: {mapping.email} -> {mapping.platform}")

    def save_user_mapping_sync(self, mapping: UserIdMapping) -> None:
        """Save user ID mapping (sync wrapper for async)."""
        asyncio.get_event_loop().run_until_complete(self.save_user_mapping_async(mapping))

    async def delete_user_mapping(
        self, email: str, platform: str, user_id: str = "default"
    ) -> bool:
        """Delete user ID mapping (async)."""
        return await self.delete_user_mapping_async(email, platform, user_id)

    async def delete_user_mapping_async(
        self, email: str, platform: str, user_id: str = "default"
    ) -> bool:
        """Delete user ID mapping asynchronously."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM user_id_mappings WHERE user_id = $1 AND platform = $2 AND email = $3",
                user_id,
                platform,
                email,
            )
            deleted = result != "DELETE 0"
            _record_user_mapping_operation("delete", platform, deleted)
            return deleted

    def delete_user_mapping_sync(self, email: str, platform: str, user_id: str = "default") -> bool:
        """Delete user ID mapping (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(
            self.delete_user_mapping_async(email, platform, user_id)
        )

    async def list_user_mappings(
        self, platform: Optional[str] = None, user_id: str = "default"
    ) -> List[UserIdMapping]:
        """List user ID mappings (async)."""
        return await self.list_user_mappings_async(platform, user_id)

    async def list_user_mappings_async(
        self, platform: Optional[str] = None, user_id: str = "default"
    ) -> List[UserIdMapping]:
        """List user ID mappings asynchronously."""
        async with self._pool.acquire() as conn:
            if platform:
                rows = await conn.fetch(
                    """SELECT email, platform, platform_user_id, display_name,
                              EXTRACT(EPOCH FROM created_at) as created_at,
                              EXTRACT(EPOCH FROM updated_at) as updated_at,
                              user_id
                       FROM user_id_mappings
                       WHERE user_id = $1 AND platform = $2""",
                    user_id,
                    platform,
                )
            else:
                rows = await conn.fetch(
                    """SELECT email, platform, platform_user_id, display_name,
                              EXTRACT(EPOCH FROM created_at) as created_at,
                              EXTRACT(EPOCH FROM updated_at) as updated_at,
                              user_id
                       FROM user_id_mappings
                       WHERE user_id = $1""",
                    user_id,
                )
            return [self._row_to_mapping(row) for row in rows]

    def list_user_mappings_sync(
        self, platform: Optional[str] = None, user_id: str = "default"
    ) -> List[UserIdMapping]:
        """List user ID mappings (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(
            self.list_user_mappings_async(platform, user_id)
        )

    async def close(self) -> None:
        """Close is a no-op for pool-based stores (pool managed externally)."""
        pass


# =============================================================================
# Global Store Factory
# =============================================================================

_integration_store: Optional[IntegrationStoreBackend] = None


def get_integration_store() -> IntegrationStoreBackend:
    """
    Get or create the integration store.

    Backend selection (in preference order):
    1. Supabase PostgreSQL (if SUPABASE_URL + SUPABASE_DB_PASSWORD configured)
    2. Self-hosted PostgreSQL (if DATABASE_URL or ARAGORA_POSTGRES_DSN configured)
    3. Redis (if ARAGORA_INTEGRATION_STORE_BACKEND=redis and ARAGORA_REDIS_URL configured)
    4. SQLite (fallback, with production warning)

    Override via:
    - ARAGORA_INTEGRATION_STORE_BACKEND: "memory", "sqlite", "postgres", "supabase", or "redis"
    - ARAGORA_DB_BACKEND: Global override

    Returns:
        Configured IntegrationStoreBackend instance
    """
    global _integration_store
    if _integration_store is not None:
        return _integration_store

    # Check store-specific backend first
    backend_type = os.environ.get("ARAGORA_INTEGRATION_STORE_BACKEND", "").lower()

    # Handle Redis explicitly (not part of standard persistent store preference)
    if backend_type == "redis":
        # Get data directory for SQLite fallback
        try:
            from aragora.config.legacy import DATA_DIR

            data_dir = DATA_DIR
        except ImportError:
            env_dir = os.environ.get("ARAGORA_DATA_DIR") or os.environ.get("ARAGORA_NOMIC_DIR")
            data_dir = Path(env_dir or ".nomic")
        db_path = data_dir / "integrations.db"
        logger.info("Using Redis integration store with SQLite fallback")
        _integration_store = RedisIntegrationStore(db_path)
        return _integration_store

    # Use unified connection factory for persistent storage
    from aragora.storage.connection_factory import create_persistent_store

    _integration_store = create_persistent_store(
        store_name="integration",
        sqlite_class=SQLiteIntegrationStore,
        postgres_class=PostgresIntegrationStore,
        db_filename="integrations.db",
        memory_class=InMemoryIntegrationStore,
    )

    return _integration_store


def set_integration_store(store: IntegrationStoreBackend) -> None:
    """
    Set custom integration store.

    Useful for testing or custom deployments.
    """
    global _integration_store
    _integration_store = store
    logger.debug(f"Integration store backend set: {type(store).__name__}")


def reset_integration_store() -> None:
    """Reset the global integration store (for testing)."""
    global _integration_store
    _integration_store = None


__all__ = [
    "IntegrationConfig",
    "IntegrationType",
    "VALID_INTEGRATION_TYPES",
    "UserIdMapping",
    "IntegrationStoreBackend",
    "InMemoryIntegrationStore",
    "SQLiteIntegrationStore",
    "RedisIntegrationStore",
    "PostgresIntegrationStore",
    "get_integration_store",
    "set_integration_store",
    "reset_integration_store",
]
