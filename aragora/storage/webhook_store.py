"""
Webhook Idempotency Storage.

Provides persistent storage for tracking processed webhook events to prevent
duplicate processing. Survives server restarts unlike in-memory dict.

Backends:
- InMemoryWebhookStore: Fast, single-instance only (for testing)
- SQLiteWebhookStore: Persisted, single-instance (default for production)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional, cast

if TYPE_CHECKING:
    from asyncpg import Pool

logger = logging.getLogger(__name__)


class WebhookStoreBackend(ABC):
    """Abstract base for webhook idempotency storage."""

    @abstractmethod
    def is_processed(self, event_id: str) -> bool:
        """
        Check if webhook event was already processed.

        Args:
            event_id: Stripe webhook event ID

        Returns:
            True if event was already processed
        """
        pass

    @abstractmethod
    def mark_processed(self, event_id: str, result: str = "success") -> None:
        """
        Mark webhook event as processed.

        Args:
            event_id: Stripe webhook event ID
            result: Processing result (success, error, etc.)
        """
        pass

    @abstractmethod
    def cleanup_expired(self) -> int:
        """
        Remove entries older than TTL.

        Returns:
            Number of entries removed
        """
        pass

    def size(self) -> int:
        """Get current store size (optional)."""
        return -1  # Not supported by default


class InMemoryWebhookStore(WebhookStoreBackend):
    """
    Thread-safe in-memory webhook store.

    Fast but not shared across restarts. Suitable for development/testing.
    """

    def __init__(self, ttl_seconds: int = 86400, cleanup_interval: int = 3600):
        """
        Initialize in-memory store.

        Args:
            ttl_seconds: Time-to-live for entries (default 24 hours)
            cleanup_interval: Seconds between automatic cleanups
        """
        self._store: dict[str, tuple[float, str]] = {}  # event_id -> (timestamp, result)
        self._lock = threading.RLock()  # RLock to allow recursive locking
        self._ttl_seconds = ttl_seconds
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()

    def is_processed(self, event_id: str) -> bool:
        """Check if event was already processed."""
        with self._lock:
            if event_id not in self._store:
                return False
            timestamp, _ = self._store[event_id]
            # Check if entry is expired
            if time.time() - timestamp > self._ttl_seconds:
                del self._store[event_id]
                return False
            return True

    def mark_processed(self, event_id: str, result: str = "success") -> None:
        """Mark event as processed."""
        with self._lock:
            self._store[event_id] = (time.time(), result)
            self._maybe_cleanup()

    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        now = time.time()
        with self._lock:
            expired = [k for k, (ts, _) in self._store.items() if now - ts > self._ttl_seconds]
            for k in expired:
                del self._store[k]
            self._last_cleanup = now
            if expired:
                logger.debug(f"InMemoryWebhookStore cleanup: removed {len(expired)}")
            return len(expired)

    def _maybe_cleanup(self) -> None:
        """Run cleanup if enough time has passed."""
        now = time.time()
        if now - self._last_cleanup > self._cleanup_interval:
            self.cleanup_expired()

    def size(self) -> int:
        """Get current store size."""
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        """Clear all entries (for testing)."""
        with self._lock:
            self._store.clear()


class SQLiteWebhookStore(WebhookStoreBackend):
    """
    SQLite-backed webhook store.

    Persisted to disk, survives restarts. Suitable for single-instance
    production deployments.
    """

    def __init__(
        self,
        db_path: Path | str,
        ttl_seconds: int = 86400,
        cleanup_interval: int = 3600,
    ):
        """
        Initialize SQLite store.

        Args:
            db_path: Path to SQLite database file
            ttl_seconds: Time-to-live for entries (default 24 hours)
            cleanup_interval: Seconds between automatic cleanups
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._ttl_seconds = ttl_seconds
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()
        self._init_schema()
        logger.info(f"SQLiteWebhookStore initialized: {self.db_path}")

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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS webhook_events (
                event_id TEXT PRIMARY KEY,
                processed_at REAL NOT NULL,
                result TEXT NOT NULL
            )
        """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_webhook_processed_at ON webhook_events(processed_at)"
        )
        conn.commit()
        conn.close()

    def is_processed(self, event_id: str) -> bool:
        """Check if event was already processed."""
        conn = self._get_conn()
        cutoff = time.time() - self._ttl_seconds
        cursor = conn.execute(
            "SELECT 1 FROM webhook_events WHERE event_id = ? AND processed_at > ?",
            (event_id, cutoff),
        )
        return cursor.fetchone() is not None

    def mark_processed(self, event_id: str, result: str = "success") -> None:
        """Mark event as processed."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO webhook_events (event_id, processed_at, result)
               VALUES (?, ?, ?)""",
            (event_id, time.time(), result),
        )
        conn.commit()
        self._maybe_cleanup()

    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        conn = self._get_conn()
        cutoff = time.time() - self._ttl_seconds
        cursor = conn.execute(
            "DELETE FROM webhook_events WHERE processed_at < ?",
            (cutoff,),
        )
        conn.commit()
        removed = cursor.rowcount
        self._last_cleanup = time.time()
        if removed > 0:
            logger.debug(f"SQLiteWebhookStore cleanup: removed {removed}")
        return removed

    def _maybe_cleanup(self) -> None:
        """Run cleanup if enough time has passed."""
        now = time.time()
        if now - self._last_cleanup > self._cleanup_interval:
            self.cleanup_expired()

    def size(self) -> int:
        """Get current store size."""
        conn = self._get_conn()
        cutoff = time.time() - self._ttl_seconds
        cursor = conn.execute(
            "SELECT COUNT(*) FROM webhook_events WHERE processed_at > ?",
            (cutoff,),
        )
        row = cursor.fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn


class PostgresWebhookStore(WebhookStoreBackend):
    """
    PostgreSQL-backed webhook idempotency store.

    Async implementation for production multi-instance deployments
    with horizontal scaling and concurrent writes.
    """

    SCHEMA_NAME = "webhook_events"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS webhook_events (
            event_id TEXT PRIMARY KEY,
            processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            result TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_webhook_processed_at ON webhook_events(processed_at);
    """

    def __init__(
        self,
        pool: "Pool",
        ttl_seconds: int = 86400,
        cleanup_interval: int = 3600,
    ):
        """
        Initialize PostgreSQL store.

        Args:
            pool: asyncpg connection pool
            ttl_seconds: Time-to-live for entries (default 24 hours)
            cleanup_interval: Seconds between automatic cleanups
        """
        self._pool = pool
        self._ttl_seconds = ttl_seconds
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()
        self._initialized = False
        logger.info("PostgresWebhookStore initialized")

    async def initialize(self) -> None:
        """Initialize database schema."""
        if self._initialized:
            return

        async with self._pool.acquire() as conn:
            await conn.execute(self.INITIAL_SCHEMA)

        self._initialized = True
        logger.debug(f"[{self.SCHEMA_NAME}] Schema initialized")

    def is_processed(self, event_id: str) -> bool:
        """Check if event was already processed (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.is_processed_async(event_id))

    async def is_processed_async(self, event_id: str) -> bool:
        """Check if event was already processed asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT 1 FROM webhook_events
                   WHERE event_id = $1
                   AND processed_at > NOW() - INTERVAL '1 second' * $2""",
                event_id, self._ttl_seconds,
            )
            return row is not None

    def mark_processed(self, event_id: str, result: str = "success") -> None:
        """Mark event as processed (sync wrapper)."""
        asyncio.get_event_loop().run_until_complete(
            self.mark_processed_async(event_id, result)
        )

    async def mark_processed_async(self, event_id: str, result: str = "success") -> None:
        """Mark event as processed asynchronously."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO webhook_events (event_id, processed_at, result)
                   VALUES ($1, NOW(), $2)
                   ON CONFLICT (event_id) DO UPDATE SET
                   processed_at = NOW(), result = $2""",
                event_id, result,
            )
        self._maybe_cleanup()

    def cleanup_expired(self) -> int:
        """Remove expired entries (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.cleanup_expired_async())

    async def cleanup_expired_async(self) -> int:
        """Remove expired entries asynchronously."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """DELETE FROM webhook_events
                   WHERE processed_at < NOW() - INTERVAL '1 second' * $1""",
                self._ttl_seconds,
            )
            self._last_cleanup = time.time()
            # Parse "DELETE N" to get count
            parts = result.split()
            removed = int(parts[1]) if len(parts) > 1 else 0
            if removed > 0:
                logger.debug(f"PostgresWebhookStore cleanup: removed {removed}")
            return removed

    def _maybe_cleanup(self) -> None:
        """Run cleanup if enough time has passed."""
        now = time.time()
        if now - self._last_cleanup > self._cleanup_interval:
            # Run cleanup in background to not block the caller
            try:
                asyncio.get_event_loop().run_until_complete(self.cleanup_expired_async())
            except Exception as e:
                logger.debug(f"Background cleanup failed: {e}")

    def size(self) -> int:
        """Get current store size (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.size_async())

    async def size_async(self) -> int:
        """Get current store size asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT COUNT(*) FROM webhook_events
                   WHERE processed_at > NOW() - INTERVAL '1 second' * $1""",
                self._ttl_seconds,
            )
            return row[0] if row else 0

    def close(self) -> None:
        """Close is a no-op for pool-based stores (pool managed externally)."""
        pass


# Global webhook store instance
_webhook_store: Optional[WebhookStoreBackend] = None


def get_webhook_store() -> WebhookStoreBackend:
    """
    Get or create the webhook idempotency store.

    Uses environment variables to configure:
    - ARAGORA_DB_BACKEND: "sqlite" or "postgres" (selects database backend)
    - ARAGORA_WEBHOOK_STORE_BACKEND: "memory", "sqlite", or "postgres" (overrides)
    - ARAGORA_DATA_DIR: Directory for SQLite database
    - ARAGORA_POSTGRES_DSN or DATABASE_URL: PostgreSQL connection string

    Returns:
        Configured WebhookStoreBackend instance
    """
    global _webhook_store
    if _webhook_store is not None:
        return _webhook_store

    # Check store-specific backend first, then global database backend
    backend_type = os.environ.get("ARAGORA_WEBHOOK_STORE_BACKEND")
    if not backend_type:
        # Fall back to global database backend setting
        backend_type = os.environ.get("ARAGORA_DB_BACKEND", "sqlite").lower()
    backend_type = backend_type.lower()

    # Import DATA_DIR from config (handles environment variable)
    try:
        from aragora.config.legacy import DATA_DIR

        data_dir = DATA_DIR
    except ImportError:
        env_dir = os.environ.get("ARAGORA_DATA_DIR") or os.environ.get("ARAGORA_NOMIC_DIR")
        data_dir = Path(env_dir or ".nomic")

    if backend_type == "memory":
        logger.info("Using in-memory webhook store (not persistent)")
        _webhook_store = InMemoryWebhookStore()
    elif backend_type in ("postgres", "postgresql"):
        logger.info("Using PostgreSQL webhook store")
        try:
            from aragora.storage.postgres_store import get_postgres_pool

            pool = asyncio.get_event_loop().run_until_complete(get_postgres_pool())
            store = PostgresWebhookStore(pool)
            asyncio.get_event_loop().run_until_complete(store.initialize())
            _webhook_store = store
        except Exception as e:
            logger.warning(f"PostgreSQL not available, falling back to SQLite: {e}")
            _webhook_store = SQLiteWebhookStore(data_dir / "webhook_events.db")
    else:  # Default: sqlite
        logger.info(f"Using SQLite webhook store: {data_dir / 'webhook_events.db'}")
        _webhook_store = SQLiteWebhookStore(data_dir / "webhook_events.db")

    return _webhook_store


def set_webhook_store(store: WebhookStoreBackend) -> None:
    """
    Set custom webhook store.

    Useful for testing or custom deployments.

    Args:
        store: WebhookStoreBackend instance to use
    """
    global _webhook_store
    _webhook_store = store
    logger.debug(f"Webhook store backend set: {type(store).__name__}")


def reset_webhook_store() -> None:
    """Reset the global webhook store (for testing)."""
    global _webhook_store
    _webhook_store = None


__all__ = [
    "WebhookStoreBackend",
    "InMemoryWebhookStore",
    "SQLiteWebhookStore",
    "PostgresWebhookStore",
    "get_webhook_store",
    "set_webhook_store",
    "reset_webhook_store",
]
