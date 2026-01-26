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

import asyncio
import json
import logging
import os
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

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


# Token fields to encrypt
_TOKEN_FIELDS = ["access_token", "refresh_token"]

# Exported for migration scripts
ENCRYPTED_FIELDS = _TOKEN_FIELDS


def _encrypt_token(token: str, user_id: str = "") -> str:
    """
    Encrypt a token value for storage.

    Uses user_id as Associated Authenticated Data (AAD) to bind the ciphertext
    to a specific user, preventing cross-user token attacks.

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
                "gmail_token_store",
            )
        return token

    try:
        service = get_encryption_service()
        # AAD binds token to this specific user
        encrypted = service.encrypt(token, associated_data=user_id if user_id else None)
        return encrypted.to_base64()
    except Exception as e:
        if is_encryption_required():
            raise EncryptionError(
                "encrypt",
                str(e),
                "gmail_token_store",
            ) from e
        logger.warning(f"Token encryption failed, storing unencrypted: {e}")
        return token


def _decrypt_token(encrypted_token: str, user_id: str = "") -> str:
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
        return service.decrypt_string(encrypted_token, associated_data=user_id if user_id else None)
    except Exception as e:
        # Could be a legacy plain token that happens to start with "A"
        logger.debug(f"Token decryption failed for user {user_id}, returning as-is: {e}")
        return encrypted_token


@dataclass
class GmailUserState:
    """Per-user Gmail state.

    SECURITY: org_id provides tenant isolation for multi-tenant deployments.
    Admin users can only manage Gmail connections within their own org.
    """

    user_id: str
    org_id: str = ""  # Organization ID for tenant isolation
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
            "org_id": self.org_id,
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
            result["token_expiry"] = self.token_expiry.isoformat() if self.token_expiry else None
        return result

    def to_json(self) -> str:
        """Serialize to JSON for storage."""
        data = {
            "user_id": self.user_id,
            "org_id": self.org_id,
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
            org_id=data.get("org_id", ""),
            email_address=data.get("email_address", ""),
            access_token=data.get("access_token", ""),
            refresh_token=data.get("refresh_token", ""),
            token_expiry=(
                datetime.fromisoformat(data["token_expiry"]) if data.get("token_expiry") else None
            ),
            history_id=data.get("history_id", ""),
            last_sync=(
                datetime.fromisoformat(data["last_sync"]) if data.get("last_sync") else None
            ),
            indexed_count=data.get("indexed_count", 0),
            total_count=data.get("total_count", 0),
            connected_at=(
                datetime.fromisoformat(data["connected_at"]) if data.get("connected_at") else None
            ),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
        )

    @classmethod
    def from_row(cls, row: tuple) -> "GmailUserState":
        """Create from database row."""
        user_id = row[0]
        return cls(
            user_id=user_id,
            email_address=row[1] or "",
            access_token=_decrypt_token(row[2] or "", user_id),
            refresh_token=_decrypt_token(row[3] or "", user_id),
            token_expiry=(datetime.fromisoformat(row[4]) if row[4] else None),
            history_id=row[5] or "",
            last_sync=(datetime.fromisoformat(row[6]) if row[6] else None),
            indexed_count=row[7] or 0,
            total_count=row[8] or 0,
            connected_at=(datetime.fromisoformat(row[9]) if row[9] else None),
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
    Uses asyncio.Lock for async-safe operations in multi-worker environments.
    """

    def __init__(self) -> None:
        self._tokens: Dict[str, GmailUserState] = {}
        self._jobs: Dict[str, SyncJobState] = {}
        self._lock = asyncio.Lock()

    async def get(self, user_id: str) -> Optional[GmailUserState]:
        async with self._lock:
            return self._tokens.get(user_id)

    async def save(self, state: GmailUserState) -> None:
        state.updated_at = time.time()
        async with self._lock:
            self._tokens[state.user_id] = state

    async def delete(self, user_id: str) -> bool:
        async with self._lock:
            if user_id in self._tokens:
                del self._tokens[user_id]
                return True
            return False

    async def list_all(self) -> List[GmailUserState]:
        async with self._lock:
            return list(self._tokens.values())

    async def get_sync_job(self, user_id: str) -> Optional[SyncJobState]:
        async with self._lock:
            return self._jobs.get(user_id)

    async def save_sync_job(self, job: SyncJobState) -> None:
        async with self._lock:
            self._jobs[job.user_id] = job

    async def delete_sync_job(self, user_id: str) -> bool:
        async with self._lock:
            if user_id in self._jobs:
                del self._jobs[user_id]
                return True
            return False

    def clear(self) -> None:
        """Clear all entries (for testing). Not async-safe, use only in test setup."""
        self._tokens.clear()
        self._jobs.clear()


class SQLiteGmailTokenStore(GmailTokenStoreBackend):
    """
    SQLite-backed Gmail token store.

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
                "gmail_token_store",
                StorageMode.SQLITE,
                "Gmail token store using SQLite - use PostgreSQL for multi-instance deployments",
            )
        except ImportError:
            pass  # Guards not available, allow SQLite

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
                _encrypt_token(state.access_token, state.user_id),
                _encrypt_token(state.refresh_token, state.user_id),
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
        cursor = conn.execute("""SELECT user_id, email_address, access_token, refresh_token,
                      token_expiry, history_id, last_sync, indexed_count,
                      total_count, connected_at, created_at, updated_at
               FROM gmail_tokens""")
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
        self._redis_url = redis_url or os.environ.get("ARAGORA_REDIS_URL", "redis://localhost:6379")
        self._redis_checked = False
        logger.info("RedisGmailTokenStore initialized with SQLite fallback")

    def _get_redis(self) -> Optional[Any]:
        """Get Redis client (lazy initialization)."""
        if self._redis_checked:
            return self._redis

        try:
            import redis

            self._redis = redis.from_url(self._redis_url, encoding="utf-8", decode_responses=True)
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
                redis.setex(self._token_key(user_id), self.REDIS_TTL, state.to_json())
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
                redis.setex(self._token_key(state.user_id), self.REDIS_TTL, state.to_json())
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


class PostgresGmailTokenStore(GmailTokenStoreBackend):
    """
    PostgreSQL-backed Gmail token store.

    Async implementation for production multi-instance deployments
    with horizontal scaling and concurrent writes.
    """

    SCHEMA_NAME = "gmail_tokens"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS gmail_tokens (
            user_id TEXT PRIMARY KEY,
            email_address TEXT,
            access_token TEXT,
            refresh_token TEXT,
            token_expiry TIMESTAMPTZ,
            history_id TEXT,
            last_sync TIMESTAMPTZ,
            indexed_count INTEGER DEFAULT 0,
            total_count INTEGER DEFAULT 0,
            connected_at TIMESTAMPTZ,
            created_at DOUBLE PRECISION NOT NULL,
            updated_at DOUBLE PRECISION NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_gmail_tokens_email ON gmail_tokens(email_address);

        CREATE TABLE IF NOT EXISTS gmail_sync_jobs (
            user_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            progress INTEGER DEFAULT 0,
            messages_synced INTEGER DEFAULT 0,
            started_at TEXT,
            completed_at TEXT,
            failed_at TEXT,
            error TEXT
        );
    """

    def __init__(self, pool: "Pool"):
        self._pool = pool
        self._initialized = False
        logger.info("PostgresGmailTokenStore initialized")

    async def initialize(self) -> None:
        """Initialize database schema."""
        if self._initialized:
            return

        async with self._pool.acquire() as conn:
            await conn.execute(self.INITIAL_SCHEMA)

        self._initialized = True
        logger.debug(f"[{self.SCHEMA_NAME}] Schema initialized")

    async def get(self, user_id: str) -> Optional[GmailUserState]:
        """Get Gmail state for a user."""
        return await self.get_async(user_id)

    def get_sync(self, user_id: str) -> Optional[GmailUserState]:
        """Get Gmail state for a user (sync wrapper for async)."""
        from aragora.utils.async_utils import run_async

        return run_async(self.get_async(user_id))

    async def get_async(self, user_id: str) -> Optional[GmailUserState]:
        """Get Gmail state for a user asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT user_id, email_address, access_token, refresh_token,
                          token_expiry, history_id, last_sync, indexed_count,
                          total_count, connected_at, created_at, updated_at
                   FROM gmail_tokens WHERE user_id = $1""",
                user_id,
            )
            if row:
                return self._row_to_state(row)
            return None

    def _row_to_state(self, row: Any) -> GmailUserState:
        """Convert database row to GmailUserState."""
        user_id = row["user_id"]
        return GmailUserState(
            user_id=user_id,
            email_address=row["email_address"] or "",
            access_token=_decrypt_token(row["access_token"] or "", user_id),
            refresh_token=_decrypt_token(row["refresh_token"] or "", user_id),
            token_expiry=row["token_expiry"],  # Already a datetime from asyncpg
            history_id=row["history_id"] or "",
            last_sync=row["last_sync"],  # Already a datetime from asyncpg
            indexed_count=row["indexed_count"] or 0,
            total_count=row["total_count"] or 0,
            connected_at=row["connected_at"],  # Already a datetime from asyncpg
            created_at=row["created_at"] or time.time(),
            updated_at=row["updated_at"] or time.time(),
        )

    async def save(self, state: GmailUserState) -> None:
        """Save Gmail state for a user."""
        await self.save_async(state)

    def save_sync(self, state: GmailUserState) -> None:
        """Save Gmail state for a user (sync wrapper for async)."""
        from aragora.utils.async_utils import run_async

        run_async(self.save_async(state))

    async def save_async(self, state: GmailUserState) -> None:
        """Save Gmail state for a user asynchronously."""
        state.updated_at = time.time()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO gmail_tokens
                   (user_id, email_address, access_token, refresh_token,
                    token_expiry, history_id, last_sync, indexed_count,
                    total_count, connected_at, created_at, updated_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                   ON CONFLICT (user_id) DO UPDATE SET
                    email_address = EXCLUDED.email_address,
                    access_token = EXCLUDED.access_token,
                    refresh_token = EXCLUDED.refresh_token,
                    token_expiry = EXCLUDED.token_expiry,
                    history_id = EXCLUDED.history_id,
                    last_sync = EXCLUDED.last_sync,
                    indexed_count = EXCLUDED.indexed_count,
                    total_count = EXCLUDED.total_count,
                    connected_at = EXCLUDED.connected_at,
                    updated_at = EXCLUDED.updated_at""",
                state.user_id,
                state.email_address,
                _encrypt_token(state.access_token, state.user_id),
                _encrypt_token(state.refresh_token, state.user_id),
                state.token_expiry,
                state.history_id,
                state.last_sync,
                state.indexed_count,
                state.total_count,
                state.connected_at,
                state.created_at,
                state.updated_at,
            )
        logger.debug(f"Saved Gmail state for user {state.user_id}")

    async def delete(self, user_id: str) -> bool:
        """Delete Gmail state for a user."""
        return await self.delete_async(user_id)

    def delete_sync(self, user_id: str) -> bool:
        """Delete Gmail state for a user (sync wrapper for async)."""
        from aragora.utils.async_utils import run_async

        return run_async(self.delete_async(user_id))

    async def delete_async(self, user_id: str) -> bool:
        """Delete Gmail state for a user asynchronously."""
        async with self._pool.acquire() as conn:
            result = await conn.execute("DELETE FROM gmail_tokens WHERE user_id = $1", user_id)
            deleted = result != "DELETE 0"
            if deleted:
                logger.debug(f"Deleted Gmail state for user {user_id}")
            return deleted

    async def list_all(self) -> List[GmailUserState]:
        """List all Gmail states (admin use)."""
        return await self.list_all_async()

    def list_all_sync(self) -> List[GmailUserState]:
        """List all Gmail states (sync wrapper for async)."""
        from aragora.utils.async_utils import run_async

        return run_async(self.list_all_async())

    async def list_all_async(self) -> List[GmailUserState]:
        """List all Gmail states asynchronously."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""SELECT user_id, email_address, access_token, refresh_token,
                          token_expiry, history_id, last_sync, indexed_count,
                          total_count, connected_at, created_at, updated_at
                   FROM gmail_tokens""")
            return [self._row_to_state(row) for row in rows]

    async def get_sync_job(self, user_id: str) -> Optional[SyncJobState]:
        """Get sync job state for a user."""
        return await self.get_sync_job_async(user_id)

    def get_sync_job_sync(self, user_id: str) -> Optional[SyncJobState]:
        """Get sync job state for a user (sync wrapper for async)."""
        from aragora.utils.async_utils import run_async

        return run_async(self.get_sync_job_async(user_id))

    async def get_sync_job_async(self, user_id: str) -> Optional[SyncJobState]:
        """Get sync job state for a user asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT user_id, status, progress, messages_synced,
                          started_at, completed_at, failed_at, error
                   FROM gmail_sync_jobs WHERE user_id = $1""",
                user_id,
            )
            if row:
                return SyncJobState(
                    user_id=row["user_id"],
                    status=row["status"],
                    progress=row["progress"] or 0,
                    messages_synced=row["messages_synced"] or 0,
                    started_at=row["started_at"],
                    completed_at=row["completed_at"],
                    failed_at=row["failed_at"],
                    error=row["error"],
                )
            return None

    async def save_sync_job(self, job: SyncJobState) -> None:
        """Save sync job state."""
        await self.save_sync_job_async(job)

    def save_sync_job_sync(self, job: SyncJobState) -> None:
        """Save sync job state (sync wrapper for async)."""
        from aragora.utils.async_utils import run_async

        run_async(self.save_sync_job_async(job))

    async def save_sync_job_async(self, job: SyncJobState) -> None:
        """Save sync job state asynchronously."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO gmail_sync_jobs
                   (user_id, status, progress, messages_synced,
                    started_at, completed_at, failed_at, error)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                   ON CONFLICT (user_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    progress = EXCLUDED.progress,
                    messages_synced = EXCLUDED.messages_synced,
                    started_at = EXCLUDED.started_at,
                    completed_at = EXCLUDED.completed_at,
                    failed_at = EXCLUDED.failed_at,
                    error = EXCLUDED.error""",
                job.user_id,
                job.status,
                job.progress,
                job.messages_synced,
                job.started_at,
                job.completed_at,
                job.failed_at,
                job.error,
            )

    async def delete_sync_job(self, user_id: str) -> bool:
        """Delete sync job for a user."""
        return await self.delete_sync_job_async(user_id)

    def delete_sync_job_sync(self, user_id: str) -> bool:
        """Delete sync job for a user (sync wrapper for async)."""
        from aragora.utils.async_utils import run_async

        return run_async(self.delete_sync_job_async(user_id))

    async def delete_sync_job_async(self, user_id: str) -> bool:
        """Delete sync job for a user asynchronously."""
        async with self._pool.acquire() as conn:
            result = await conn.execute("DELETE FROM gmail_sync_jobs WHERE user_id = $1", user_id)
            return result != "DELETE 0"

    async def close(self) -> None:
        """Close is a no-op for pool-based stores (pool managed externally)."""
        pass


# =============================================================================
# Global Store Factory
# =============================================================================

_gmail_token_store: Optional[GmailTokenStoreBackend] = None
_gmail_store_init_lock = threading.Lock()


def get_gmail_token_store() -> GmailTokenStoreBackend:
    """
    Get or create the Gmail token store.

    Uses environment variables to configure:
    - ARAGORA_DB_BACKEND: "sqlite", "postgres", or "supabase"
    - ARAGORA_GMAIL_STORE_BACKEND: "memory", "sqlite", "postgres", "supabase", or "redis"
    - ARAGORA_DATA_DIR: Directory for SQLite database
    - ARAGORA_REDIS_URL: Redis connection URL (for redis backend)
    - SUPABASE_URL + SUPABASE_DB_PASSWORD or SUPABASE_POSTGRES_DSN
    - ARAGORA_POSTGRES_DSN or DATABASE_URL

    Returns:
        Configured GmailTokenStoreBackend instance
    """
    global _gmail_token_store

    # Fast path: already initialized
    if _gmail_token_store is not None:
        return _gmail_token_store

    # Thread-safe initialization
    with _gmail_store_init_lock:
        # Double-check after acquiring lock
        if _gmail_token_store is not None:
            return _gmail_token_store

        # Check store-specific backend first, then global database backend
        backend_type = os.environ.get("ARAGORA_GMAIL_STORE_BACKEND")
        if not backend_type:
            backend_type = os.environ.get("ARAGORA_DB_BACKEND", "auto")
        backend_type = backend_type.lower()

        # Preserve legacy data directory when configured
        data_dir = None
        try:
            from aragora.config.legacy import DATA_DIR

            data_dir = DATA_DIR
        except ImportError:
            data_dir = None

        if backend_type == "redis":
            env_dir = os.environ.get("ARAGORA_DATA_DIR") or os.environ.get("ARAGORA_NOMIC_DIR")
            base_dir = Path(data_dir or env_dir or ".nomic")
            db_path = base_dir / "gmail_tokens.db"
            logger.info("Using Redis Gmail token store with SQLite fallback")
            _gmail_token_store = RedisGmailTokenStore(db_path)
            return _gmail_token_store

        from aragora.storage.connection_factory import create_persistent_store

        _gmail_token_store = create_persistent_store(
            store_name="gmail",
            sqlite_class=SQLiteGmailTokenStore,
            postgres_class=PostgresGmailTokenStore,
            db_filename="gmail_tokens.db",
            memory_class=InMemoryGmailTokenStore,
            data_dir=str(data_dir) if data_dir else None,
        )

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
    "PostgresGmailTokenStore",
    "get_gmail_token_store",
    "set_gmail_token_store",
    "reset_gmail_token_store",
]
