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

import json
import logging
import os
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, cast

logger = logging.getLogger(__name__)

# =============================================================================
# Integration Types and Models
# =============================================================================

IntegrationType = Literal[
    "slack", "discord", "telegram", "email", "teams", "whatsapp", "matrix"
]

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
SENSITIVE_KEYS = frozenset([
    "access_token",
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
])


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
        """Create from database row."""
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

    def clear(self) -> None:
        """Clear all entries (for testing)."""
        with self._lock:
            self._store.clear()


# =============================================================================
# SQLite Backend
# =============================================================================


class SQLiteIntegrationStore(IntegrationStoreBackend):
    """
    SQLite-backed integration store.

    Persisted to disk, survives restarts. Suitable for single-instance
    production deployments.
    """

    def __init__(self, db_path: Path | str):
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
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_integrations_user ON integrations(user_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_integrations_type ON integrations(integration_type)"
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
            return IntegrationConfig.from_row(row)
        return None

    async def save(self, config: IntegrationConfig) -> None:
        conn = self._get_conn()
        config.updated_at = time.time()
        user_id = config.user_id or "default"
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
                json.dumps(config.settings),
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
        return [IntegrationConfig.from_row(row) for row in cursor.fetchall()]

    async def list_all(self) -> List[IntegrationConfig]:
        conn = self._get_conn()
        cursor = conn.execute(
            """SELECT integration_type, enabled, created_at, updated_at,
                      notify_on_consensus, notify_on_debate_end, notify_on_error,
                      notify_on_leaderboard, settings_json, messages_sent,
                      errors_24h, last_activity, last_error, user_id, workspace_id
               FROM integrations"""
        )
        return [IntegrationConfig.from_row(row) for row in cursor.fetchall()]

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
        self._redis_url = redis_url or os.environ.get(
            "ARAGORA_REDIS_URL", "redis://localhost:6379"
        )
        self._redis_checked = False
        logger.info(f"RedisIntegrationStore initialized with SQLite fallback")

    def _get_redis(self) -> Optional[Any]:
        """Get Redis client (lazy initialization)."""
        if self._redis_checked:
            return self._redis

        try:
            import redis

            self._redis = redis.from_url(
                self._redis_url, encoding="utf-8", decode_responses=True
            )
            # Test connection
            self._redis.ping()
            self._redis_checked = True
            logger.info(f"Redis connected for integration store")
        except Exception as e:
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
            except Exception as e:
                logger.debug(f"Redis get failed, falling back to SQLite: {e}")

        # Fall back to SQLite
        config = await self._sqlite.get(integration_type, user_id)

        # Populate Redis cache if found
        if config and redis:
            try:
                key = self._redis_key(integration_type, user_id)
                redis.setex(key, self.REDIS_TTL, config.to_json())
            except Exception:
                pass  # Best effort cache population

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
            except Exception as e:
                logger.debug(f"Redis cache update failed: {e}")

    async def delete(self, integration_type: str, user_id: str = "default") -> bool:
        # Delete from both stores
        redis = self._get_redis()
        if redis:
            try:
                key = self._redis_key(integration_type, user_id)
                redis.delete(key)
            except Exception:
                pass

        return await self._sqlite.delete(integration_type, user_id)

    async def list_for_user(self, user_id: str = "default") -> List[IntegrationConfig]:
        # Always use SQLite for list operations (authoritative)
        return await self._sqlite.list_for_user(user_id)

    async def list_all(self) -> List[IntegrationConfig]:
        return await self._sqlite.list_all()

    async def close(self) -> None:
        await self._sqlite.close()
        if self._redis:
            self._redis.close()


# =============================================================================
# Global Store Factory
# =============================================================================

_integration_store: Optional[IntegrationStoreBackend] = None


def get_integration_store() -> IntegrationStoreBackend:
    """
    Get or create the integration store.

    Uses environment variables to configure:
    - ARAGORA_INTEGRATION_STORE_BACKEND: "memory", "sqlite", or "redis" (default: sqlite)
    - ARAGORA_DATA_DIR: Directory for SQLite database
    - ARAGORA_REDIS_URL: Redis connection URL (for redis backend)

    Returns:
        Configured IntegrationStoreBackend instance
    """
    global _integration_store
    if _integration_store is not None:
        return _integration_store

    backend_type = os.environ.get("ARAGORA_INTEGRATION_STORE_BACKEND", "sqlite").lower()

    # Get data directory
    try:
        from aragora.config.legacy import DATA_DIR

        data_dir = DATA_DIR
    except ImportError:
        env_dir = os.environ.get("ARAGORA_DATA_DIR") or os.environ.get("ARAGORA_NOMIC_DIR")
        data_dir = Path(env_dir or ".nomic")

    db_path = data_dir / "integrations.db"

    if backend_type == "memory":
        logger.info("Using in-memory integration store (not persistent)")
        _integration_store = InMemoryIntegrationStore()
    elif backend_type == "redis":
        logger.info("Using Redis integration store with SQLite fallback")
        _integration_store = RedisIntegrationStore(db_path)
    else:  # Default: sqlite
        logger.info(f"Using SQLite integration store: {db_path}")
        _integration_store = SQLiteIntegrationStore(db_path)

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
    "IntegrationStoreBackend",
    "InMemoryIntegrationStore",
    "SQLiteIntegrationStore",
    "RedisIntegrationStore",
    "get_integration_store",
    "set_integration_store",
    "reset_integration_store",
]
