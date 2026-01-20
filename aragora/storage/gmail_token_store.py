"""
Gmail Token Storage.

Provides persistent storage for Gmail OAuth tokens and user state.
Survives server restarts and supports multi-instance deployments.

Backends:
- InMemoryGmailTokenStore: Fast, single-instance only (for testing)
- SQLiteGmailTokenStore: Persisted, single-instance (default)
- RedisGmailTokenStore: Distributed, multi-instance (with SQLite fallback)

Usage:
    from aragora.storage.gmail_token_store import get_gmail_token_store

    store = get_gmail_token_store()
    await store.save(user_state)
    state = await store.get(user_id)
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

logger = logging.getLogger(__name__)


@dataclass
class GmailUserState:
    """Per-user Gmail state."""

    user_id: str
    email_address: str = ""
    access_token: str = ""
    refresh_token: str = ""
    token_expiry: Optional[datetime] = None
    history_id: str = ""
    last_sync: Optional[datetime] = None
    indexed_count: int = 0
    total_count: int = 0
    connected_at: Optional[datetime] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self, include_tokens: bool = False) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "user_id": self.user_id,
            "email_address": self.email_address,
            "history_id": self.history_id,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "indexed_count": self.indexed_count,
            "total_count": self.total_count,
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
            "is_connected": bool(self.refresh_token),
        }
        if include_tokens:
            result["access_token"] = self.access_token
            result["refresh_token"] = self.refresh_token
            result["token_expiry"] = (
                self.token_expiry.isoformat() if self.token_expiry else None
            )
        return result

    def to_json(self) -> str:
        """Serialize to JSON for storage."""
        data = {
            "user_id": self.user_id,
            "email_address": self.email_address,
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_expiry": self.token_expiry.isoformat() if self.token_expiry else None,
            "history_id": self.history_id,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "indexed_count": self.indexed_count,
            "total_count": self.total_count,
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> "GmailUserState":
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return cls(
            user_id=data["user_id"],
            email_address=data.get("email_address", ""),
            access_token=data.get("access_token", ""),
            refresh_token=data.get("refresh_token", ""),
            token_expiry=(
                datetime.fromisoformat(data["token_expiry"])
                if data.get("token_expiry")
                else None
            ),
            history_id=data.get("history_id", ""),
            last_sync=(
                datetime.fromisoformat(data["last_sync"])
                if data.get("last_sync")
                else None
            ),
            indexed_count=data.get("indexed_count", 0),
            total_count=data.get("total_count", 0),
            connected_at=(
                datetime.fromisoformat(data["connected_at"])
                if data.get("connected_at")
                else None
            ),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
        )

    @classmethod
    def from_row(cls, row: tuple) -> "GmailUserState":
        """Create from database row."""
        return cls(
            user_id=row[0],
            email_address=row[1] or "",
            access_token=row[2] or "",
            refresh_token=row[3] or "",
            token_expiry=(
                datetime.fromisoformat(row[4]) if row[4] else None
            ),
            history_id=row[5] or "",
            last_sync=(
                datetime.fromisoformat(row[6]) if row[6] else None
            ),
            indexed_count=row[7] or 0,
            total_count=row[8] or 0,
            connected_at=(
                datetime.fromisoformat(row[9]) if row[9] else None
            ),
            created_at=row[10] or time.time(),
            updated_at=row[11] or time.time(),
        )


@dataclass
class SyncJobState:
    """Sync job state."""

    user_id: str
    status: str  # running, completed, failed, cancelled
    progress: int = 0
    messages_synced: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    failed_at: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GmailTokenStoreBackend(ABC):
    """Abstract base for Gmail token storage backends."""

    @abstractmethod
    async def get(self, user_id: str) -> Optional[GmailUserState]:
        """Get Gmail state for a user."""
        pass

    @abstractmethod
    async def save(self, state: GmailUserState) -> None:
        """Save Gmail state for a user."""
        pass

    @abstractmethod
    async def delete(self, user_id: str) -> bool:
        """Delete Gmail state for a user. Returns True if deleted."""
        pass

    @abstractmethod
    async def list_all(self) -> List[GmailUserState]:
        """List all Gmail states (admin use)."""
        pass

    @abstractmethod
    async def get_sync_job(self, user_id: str) -> Optional[SyncJobState]:
        """Get sync job state for a user."""
        pass

    @abstractmethod
    async def save_sync_job(self, job: SyncJobState) -> None:
        """Save sync job state."""
        pass

    @abstractmethod
    async def delete_sync_job(self, user_id: str) -> bool:
        """Delete sync job for a user."""
        pass

    async def close(self) -> None:
        """Close connections (optional to implement)."""
        pass


class InMemoryGmailTokenStore(GmailTokenStoreBackend):
    """
    Thread-safe in-memory Gmail token store.

    Fast but not shared across restarts. Suitable for development/testing.
    """

    def __init__(self) -> None:
        self._tokens: Dict[str, GmailUserState] = {}
        self._jobs: Dict[str, SyncJobState] = {}
        self._lock = threading.RLock()

    async def get(self, user_id: str) -> Optional[GmailUserState]:
        with self._lock:
            return self._tokens.get(user_id)

    async def save(self, state: GmailUserState) -> None:
        state.updated_at = time.time()
        with self._lock:
            self._tokens[state.user_id] = state

    async def delete(self, user_id: str) -> bool:
        with self._lock:
            if user_id in self._tokens:
                del self._tokens[user_id]
                return True
            return False

    async def list_all(self) -> List[GmailUserState]:
        with self._lock:
            return list(self._tokens.values())

    async def get_sync_job(self, user_id: str) -> Optional[SyncJobState]:
        with self._lock:
            return self._jobs.get(user_id)

    async def save_sync_job(self, job: SyncJobState) -> None:
        with self._lock:
            self._jobs[job.user_id] = job

    async def delete_sync_job(self, user_id: str) -> bool:
        with self._lock:
            if user_id in self._jobs:
                del self._jobs[user_id]
                return True
            return False

    def clear(self) -> None:
        """Clear all entries (for testing)."""
        with self._lock:
            self._tokens.clear()
            self._jobs.clear()


class SQLiteGmailTokenStore(GmailTokenStoreBackend):
    """
    SQLite-backed Gmail token store.

    Persisted to disk, survives restarts. Suitable for single-instance
    production deployments.
    """

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()
        logger.info(f"SQLiteGmailTokenStore initialized: {self.db_path}")

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
            CREATE TABLE IF NOT EXISTS gmail_tokens (
                user_id TEXT PRIMARY KEY,
                email_address TEXT,
                access_token TEXT,
                refresh_token TEXT,
                token_expiry TEXT,
                history_id TEXT,
                last_sync TEXT,
                indexed_count INTEGER DEFAULT 0,
                total_count INTEGER DEFAULT 0,
                connected_at TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gmail_sync_jobs (
                user_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                progress INTEGER DEFAULT 0,
                messages_synced INTEGER DEFAULT 0,
                started_at TEXT,
                completed_at TEXT,
                failed_at TEXT,
                error TEXT
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_gmail_tokens_email ON gmail_tokens(email_address)"
        )
        conn.commit()
        conn.close()

    async def get(self, user_id: str) -> Optional[GmailUserState]:
        conn = self._get_conn()
        cursor = conn.execute(
            """SELECT user_id, email_address, access_token, refresh_token,
                      token_expiry, history_id, last_sync, indexed_count,
                      total_count, connected_at, created_at, updated_at
               FROM gmail_tokens WHERE user_id = ?""",
            (user_id,),
        )
        row = cursor.fetchone()
        if row:
            return GmailUserState.from_row(row)
        return None

    async def save(self, state: GmailUserState) -> None:
        conn = self._get_conn()
        state.updated_at = time.time()
        conn.execute(
            """INSERT OR REPLACE INTO gmail_tokens
               (user_id, email_address, access_token, refresh_token,
                token_expiry, history_id, last_sync, indexed_count,
                total_count, connected_at, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                state.user_id,
                state.email_address,
                state.access_token,
                state.refresh_token,
                state.token_expiry.isoformat() if state.token_expiry else None,
                state.history_id,
                state.last_sync.isoformat() if state.last_sync else None,
                state.indexed_count,
                state.total_count,
                state.connected_at.isoformat() if state.connected_at else None,
                state.created_at,
                state.updated_at,
            ),
        )
        conn.commit()
        logger.debug(f"Saved Gmail state for user {state.user_id}")

    async def delete(self, user_id: str) -> bool:
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM gmail_tokens WHERE user_id = ?",
            (user_id,),
        )
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.debug(f"Deleted Gmail state for user {user_id}")
        return deleted

    async def list_all(self) -> List[GmailUserState]:
        conn = self._get_conn()
        cursor = conn.execute(
            """SELECT user_id, email_address, access_token, refresh_token,
                      token_expiry, history_id, last_sync, indexed_count,
                      total_count, connected_at, created_at, updated_at
               FROM gmail_tokens"""
        )
        return [GmailUserState.from_row(row) for row in cursor.fetchall()]

    async def get_sync_job(self, user_id: str) -> Optional[SyncJobState]:
        conn = self._get_conn()
        cursor = conn.execute(
            """SELECT user_id, status, progress, messages_synced,
                      started_at, completed_at, failed_at, error
               FROM gmail_sync_jobs WHERE user_id = ?""",
            (user_id,),
        )
        row = cursor.fetchone()
        if row:
            return SyncJobState(
                user_id=row[0],
                status=row[1],
                progress=row[2] or 0,
                messages_synced=row[3] or 0,
                started_at=row[4],
                completed_at=row[5],
                failed_at=row[6],
                error=row[7],
            )
        return None

    async def save_sync_job(self, job: SyncJobState) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO gmail_sync_jobs
               (user_id, status, progress, messages_synced,
                started_at, completed_at, failed_at, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job.user_id,
                job.status,
                job.progress,
                job.messages_synced,
                job.started_at,
                job.completed_at,
                job.failed_at,
                job.error,
            ),
        )
        conn.commit()

    async def delete_sync_job(self, user_id: str) -> bool:
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM gmail_sync_jobs WHERE user_id = ?",
            (user_id,),
        )
        conn.commit()
        return cursor.rowcount > 0

    async def close(self) -> None:
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn


class RedisGmailTokenStore(GmailTokenStoreBackend):
    """
    Redis-backed Gmail token store with SQLite fallback.

    Uses Redis for fast distributed access, with SQLite as durable storage.
    """

    REDIS_PREFIX = "aragora:gmail:tokens"
    REDIS_JOBS_PREFIX = "aragora:gmail:jobs"
    REDIS_TTL = 86400 * 30  # 30 days

    def __init__(self, db_path: Path | str, redis_url: Optional[str] = None):
        self._sqlite = SQLiteGmailTokenStore(db_path)
        self._redis: Optional[Any] = None
        self._redis_url = redis_url or os.environ.get(
            "ARAGORA_REDIS_URL", "redis://localhost:6379"
        )
        self._redis_checked = False
        logger.info("RedisGmailTokenStore initialized with SQLite fallback")

    def _get_redis(self) -> Optional[Any]:
        """Get Redis client (lazy initialization)."""
        if self._redis_checked:
            return self._redis

        try:
            import redis

            self._redis = redis.from_url(
                self._redis_url, encoding="utf-8", decode_responses=True
            )
            self._redis.ping()
            self._redis_checked = True
            logger.info("Redis connected for Gmail token store")
        except Exception as e:
            logger.debug(f"Redis not available, using SQLite only: {e}")
            self._redis = None
            self._redis_checked = True

        return self._redis

    def _token_key(self, user_id: str) -> str:
        return f"{self.REDIS_PREFIX}:{user_id}"

    def _job_key(self, user_id: str) -> str:
        return f"{self.REDIS_JOBS_PREFIX}:{user_id}"

    async def get(self, user_id: str) -> Optional[GmailUserState]:
        redis = self._get_redis()

        if redis is not None:
            try:
                data = redis.get(self._token_key(user_id))
                if data:
                    return GmailUserState.from_json(data)
            except Exception as e:
                logger.debug(f"Redis get failed, falling back to SQLite: {e}")

        state = await self._sqlite.get(user_id)

        # Populate Redis cache if found
        if state and redis:
            try:
                redis.setex(
                    self._token_key(user_id), self.REDIS_TTL, state.to_json()
                )
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug(f"Redis cache population failed (expected): {e}")

        return state

    async def save(self, state: GmailUserState) -> None:
        # Always save to SQLite (durable)
        await self._sqlite.save(state)

        # Update Redis cache
        redis = self._get_redis()
        if redis:
            try:
                redis.setex(
                    self._token_key(state.user_id), self.REDIS_TTL, state.to_json()
                )
            except Exception as e:
                logger.debug(f"Redis cache update failed: {e}")

    async def delete(self, user_id: str) -> bool:
        redis = self._get_redis()
        if redis:
            try:
                redis.delete(self._token_key(user_id))
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug(f"Redis cache deletion failed (expected): {e}")

        return await self._sqlite.delete(user_id)

    async def list_all(self) -> List[GmailUserState]:
        return await self._sqlite.list_all()

    async def get_sync_job(self, user_id: str) -> Optional[SyncJobState]:
        redis = self._get_redis()

        if redis is not None:
            try:
                data = redis.get(self._job_key(user_id))
                if data:
                    job_data = json.loads(data)
                    return SyncJobState(**job_data)
            except (ConnectionError, TimeoutError, OSError, json.JSONDecodeError, TypeError) as e:
                logger.debug(f"Redis sync job get failed, falling back to SQLite: {e}")

        return await self._sqlite.get_sync_job(user_id)

    async def save_sync_job(self, job: SyncJobState) -> None:
        # Save to SQLite
        await self._sqlite.save_sync_job(job)

        # Update Redis cache
        redis = self._get_redis()
        if redis:
            try:
                redis.setex(
                    self._job_key(job.user_id),
                    3600,  # 1 hour TTL for jobs
                    json.dumps(job.to_dict()),
                )
            except (ConnectionError, TimeoutError, OSError, TypeError) as e:
                logger.debug(f"Redis sync job cache update failed: {e}")

    async def delete_sync_job(self, user_id: str) -> bool:
        redis = self._get_redis()
        if redis:
            try:
                redis.delete(self._job_key(user_id))
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug(f"Redis sync job deletion failed (expected): {e}")

        return await self._sqlite.delete_sync_job(user_id)

    async def close(self) -> None:
        await self._sqlite.close()
        if self._redis:
            self._redis.close()


# =============================================================================
# Global Store Factory
# =============================================================================

_gmail_token_store: Optional[GmailTokenStoreBackend] = None


def get_gmail_token_store() -> GmailTokenStoreBackend:
    """
    Get or create the Gmail token store.

    Uses environment variables to configure:
    - ARAGORA_GMAIL_STORE_BACKEND: "memory", "sqlite", or "redis" (default: sqlite)
    - ARAGORA_DATA_DIR: Directory for SQLite database
    - ARAGORA_REDIS_URL: Redis connection URL (for redis backend)

    Returns:
        Configured GmailTokenStoreBackend instance
    """
    global _gmail_token_store
    if _gmail_token_store is not None:
        return _gmail_token_store

    backend_type = os.environ.get("ARAGORA_GMAIL_STORE_BACKEND", "sqlite").lower()

    # Get data directory
    try:
        from aragora.config.legacy import DATA_DIR

        data_dir = DATA_DIR
    except ImportError:
        env_dir = os.environ.get("ARAGORA_DATA_DIR") or os.environ.get("ARAGORA_NOMIC_DIR")
        data_dir = Path(env_dir or ".nomic")

    db_path = data_dir / "gmail_tokens.db"

    if backend_type == "memory":
        logger.info("Using in-memory Gmail token store (not persistent)")
        _gmail_token_store = InMemoryGmailTokenStore()
    elif backend_type == "redis":
        logger.info("Using Redis Gmail token store with SQLite fallback")
        _gmail_token_store = RedisGmailTokenStore(db_path)
    else:  # Default: sqlite
        logger.info(f"Using SQLite Gmail token store: {db_path}")
        _gmail_token_store = SQLiteGmailTokenStore(db_path)

    return _gmail_token_store


def set_gmail_token_store(store: GmailTokenStoreBackend) -> None:
    """Set custom Gmail token store."""
    global _gmail_token_store
    _gmail_token_store = store
    logger.debug(f"Gmail token store backend set: {type(store).__name__}")


def reset_gmail_token_store() -> None:
    """Reset the global Gmail token store (for testing)."""
    global _gmail_token_store
    _gmail_token_store = None


__all__ = [
    "GmailUserState",
    "SyncJobState",
    "GmailTokenStoreBackend",
    "InMemoryGmailTokenStore",
    "SQLiteGmailTokenStore",
    "RedisGmailTokenStore",
    "get_gmail_token_store",
    "set_gmail_token_store",
    "reset_gmail_token_store",
]
