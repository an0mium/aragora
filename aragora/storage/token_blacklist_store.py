"""
Token Blacklist Storage Backends.

Provides pluggable backends for persisting revoked JWT tokens:
- InMemoryBlacklist: Fast, single-instance only (default for dev)
- SQLiteBlacklist: Persisted, single-instance (default for production)
- RedisBlacklist: Shared across instances (optional, for multi-instance deployments)
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class BlacklistBackend(ABC):
    """Abstract base for token blacklist storage."""

    @abstractmethod
    def add(self, token_jti: str, expires_at: float) -> None:
        """
        Add token to blacklist.

        Args:
            token_jti: Token's unique identifier (hash of token)
            expires_at: Unix timestamp when token naturally expires
        """
        pass

    @abstractmethod
    def contains(self, token_jti: str) -> bool:
        """
        Check if token is blacklisted.

        Args:
            token_jti: Token's unique identifier

        Returns:
            True if token is in blacklist
        """
        pass

    @abstractmethod
    def cleanup_expired(self) -> int:
        """
        Remove expired entries from blacklist.

        Returns:
            Number of entries removed
        """
        pass

    def size(self) -> int:
        """Get current blacklist size (optional)."""
        return -1  # Not supported by default


MAX_BLACKLIST_SIZE = 100000  # Prevent unbounded memory growth


class InMemoryBlacklist(BlacklistBackend):
    """
    Thread-safe in-memory token blacklist.

    Fast but not shared across instances. Suitable for development
    or single-instance deployments where restart clears all tokens.
    """

    def __init__(self, cleanup_interval: int = 300):
        """
        Initialize in-memory blacklist.

        Args:
            cleanup_interval: Seconds between automatic cleanups
        """
        self._blacklist: dict[str, float] = {}  # token_jti -> expires_at
        self._lock = threading.Lock()
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()

    def add(self, token_jti: str, expires_at: float) -> None:
        """Add token to blacklist (with size limit enforcement)."""
        with self._lock:
            # Enforce max size - evict oldest expired first, then oldest entries
            if len(self._blacklist) >= MAX_BLACKLIST_SIZE:
                now = time.time()
                # First try to remove expired entries
                expired = [k for k, v in self._blacklist.items() if v < now]
                if expired:
                    remove_count = max(1, len(expired) // 2)
                    for k in expired[:remove_count]:
                        del self._blacklist[k]
                    logger.debug(f"InMemoryBlacklist evicted {remove_count} expired entries")
                else:
                    # No expired entries - remove oldest 10% by expiration time
                    sorted_items = sorted(self._blacklist.items(), key=lambda x: x[1])
                    remove_count = max(1, len(sorted_items) // 10)
                    for k, _ in sorted_items[:remove_count]:
                        del self._blacklist[k]
                    logger.debug(f"InMemoryBlacklist evicted {remove_count} oldest entries")

            self._blacklist[token_jti] = expires_at
            self._maybe_cleanup()

    def contains(self, token_jti: str) -> bool:
        """Check if token is blacklisted."""
        with self._lock:
            return token_jti in self._blacklist

    def cleanup_expired(self) -> int:
        """Remove expired tokens."""
        now = time.time()
        with self._lock:
            expired = [k for k, v in self._blacklist.items() if v < now]
            for k in expired:
                del self._blacklist[k]
            self._last_cleanup = now
            if expired:
                logger.debug(f"InMemoryBlacklist cleanup: removed {len(expired)}")
            return len(expired)

    def _maybe_cleanup(self) -> None:
        """Run cleanup if enough time has passed."""
        now = time.time()
        if now - self._last_cleanup > self._cleanup_interval:
            self.cleanup_expired()

    def size(self) -> int:
        """Get current blacklist size."""
        with self._lock:
            return len(self._blacklist)

    def clear(self) -> None:
        """Clear all entries (for testing)."""
        with self._lock:
            self._blacklist.clear()


class SQLiteBlacklist(BlacklistBackend):
    """
    SQLite-backed token blacklist.

    Persisted to disk, survives restarts. Suitable for single-instance
    production deployments.
    """

    def __init__(self, db_path: Path | str, cleanup_interval: int = 300):
        """
        Initialize SQLite blacklist.

        Args:
            db_path: Path to SQLite database file
            cleanup_interval: Seconds between automatic cleanups
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()
        self._init_schema()
        logger.info(f"SQLiteBlacklist initialized: {self.db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS token_blacklist (
                jti TEXT PRIMARY KEY,
                expires_at REAL NOT NULL,
                revoked_at REAL NOT NULL
            )
        """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_blacklist_expires ON token_blacklist(expires_at)"
        )
        conn.commit()
        conn.close()

    def add(self, token_jti: str, expires_at: float) -> None:
        """Add token to blacklist."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO token_blacklist (jti, expires_at, revoked_at)
               VALUES (?, ?, ?)""",
            (token_jti, expires_at, time.time()),
        )
        conn.commit()
        self._maybe_cleanup()

    def contains(self, token_jti: str) -> bool:
        """Check if token is blacklisted."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT 1 FROM token_blacklist WHERE jti = ? AND expires_at > ?",
            (token_jti, time.time()),
        )
        return cursor.fetchone() is not None

    def cleanup_expired(self) -> int:
        """Remove expired tokens."""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM token_blacklist WHERE expires_at < ?",
            (time.time(),),
        )
        conn.commit()
        removed = cursor.rowcount
        self._last_cleanup = time.time()
        if removed > 0:
            logger.debug(f"SQLiteBlacklist cleanup: removed {removed}")
        return removed

    def _maybe_cleanup(self) -> None:
        """Run cleanup if enough time has passed."""
        now = time.time()
        if now - self._last_cleanup > self._cleanup_interval:
            self.cleanup_expired()

    def size(self) -> int:
        """Get current blacklist size."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT COUNT(*) FROM token_blacklist WHERE expires_at > ?",
            (time.time(),),
        )
        row = cursor.fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn


# Optional Redis backend for multi-instance deployments
try:
    import redis

    class RedisBlacklist(BlacklistBackend):
        """
        Redis-backed token blacklist.

        Shared across multiple server instances. Suitable for distributed
        production deployments. Requires redis-py package.
        """

        def __init__(self, redis_url: str, key_prefix: str = "aragora:blacklist:"):
            """
            Initialize Redis blacklist.

            Args:
                redis_url: Redis connection URL (e.g., redis://localhost:6379/0)
                key_prefix: Prefix for blacklist keys
            """
            self._client = redis.from_url(redis_url)
            self._prefix = key_prefix
            # Test connection
            self._client.ping()
            logger.info(f"RedisBlacklist initialized: {redis_url}")

        def add(self, token_jti: str, expires_at: float) -> None:
            """Add token to blacklist with auto-expiration."""
            ttl = max(1, int(expires_at - time.time()))
            self._client.setex(f"{self._prefix}{token_jti}", ttl, "1")

        def contains(self, token_jti: str) -> bool:
            """Check if token is blacklisted."""
            return self._client.exists(f"{self._prefix}{token_jti}") > 0

        def cleanup_expired(self) -> int:
            """Redis handles TTL automatically, no cleanup needed."""
            return 0

        def size(self) -> int:
            """Get approximate blacklist size."""
            # This is expensive for large keyspaces; use with caution
            keys = self._client.keys(f"{self._prefix}*")
            return len(keys)

    HAS_REDIS = True

except ImportError:
    RedisBlacklist = None  # type: ignore[misc,assignment]
    HAS_REDIS = False


# Global blacklist backend instance
_blacklist_backend: Optional[BlacklistBackend] = None


def get_blacklist_backend() -> BlacklistBackend:
    """
    Get or create the token blacklist backend.

    Uses environment variables to configure:
    - ARAGORA_BLACKLIST_BACKEND: "memory", "sqlite" (default), or "redis"
    - ARAGORA_DATA_DIR: Directory for SQLite database (from config)
    - ARAGORA_REDIS_URL: Redis URL for redis backend

    Returns:
        Configured BlacklistBackend instance
    """
    global _blacklist_backend
    if _blacklist_backend is not None:
        return _blacklist_backend

    backend_type = os.environ.get("ARAGORA_BLACKLIST_BACKEND", "sqlite").lower()

    # Import DATA_DIR from config (handles environment variable)
    try:
        from aragora.config.legacy import DATA_DIR

        data_dir = DATA_DIR
    except ImportError:
        data_dir = Path(os.environ.get("ARAGORA_DATA_DIR", ".nomic"))

    if backend_type == "memory":
        logger.info("Using in-memory token blacklist (not persistent)")
        _blacklist_backend = InMemoryBlacklist()

    elif backend_type == "redis":
        redis_url = os.environ.get("ARAGORA_REDIS_URL", "redis://localhost:6379/0")
        if not HAS_REDIS:
            logger.warning(
                "Redis requested but redis-py not installed. "
                "Falling back to SQLite. Install with: pip install redis"
            )
            _blacklist_backend = SQLiteBlacklist(data_dir / "token_blacklist.db")
        else:
            try:
                _blacklist_backend = RedisBlacklist(redis_url)
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Falling back to SQLite.")
                _blacklist_backend = SQLiteBlacklist(data_dir / "token_blacklist.db")

    else:  # Default: sqlite
        _blacklist_backend = SQLiteBlacklist(data_dir / "token_blacklist.db")

    return _blacklist_backend


def set_blacklist_backend(backend: BlacklistBackend) -> None:
    """
    Set custom blacklist backend.

    Useful for testing or custom deployments.

    Args:
        backend: BlacklistBackend instance to use
    """
    global _blacklist_backend
    _blacklist_backend = backend
    logger.info(f"Token blacklist backend set: {type(backend).__name__}")


__all__ = [
    "BlacklistBackend",
    "InMemoryBlacklist",
    "SQLiteBlacklist",
    "get_blacklist_backend",
    "set_blacklist_backend",
    "HAS_REDIS",
]

if HAS_REDIS:
    __all__.append("RedisBlacklist")
