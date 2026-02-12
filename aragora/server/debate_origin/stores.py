"""Debate origin persistence stores.

SQLite and PostgreSQL backends for durable debate origin tracking.

Note: SQLite operations use a thread pool executor to avoid blocking the
async event loop. This is critical for performance when handling multiple
concurrent chat platform debates.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .models import DebateOrigin

if TYPE_CHECKING:
    from asyncpg import Pool

logger = logging.getLogger(__name__)

# TTL for origin records (24 hours)
ORIGIN_TTL_SECONDS = int(os.environ.get("DEBATE_ORIGIN_TTL", 86400))

# Thread pool for SQLite operations (avoid blocking event loop)
# Using a small pool since SQLite serializes writes anyway
_sqlite_executor: ThreadPoolExecutor | None = None


def _get_sqlite_executor() -> ThreadPoolExecutor:
    """Get or create the thread pool for SQLite operations."""
    global _sqlite_executor
    if _sqlite_executor is None:
        _sqlite_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sqlite_origin_")
    return _sqlite_executor


class SQLiteOriginStore:
    """SQLite-backed debate origin store for durability without Redis."""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            from aragora.persistence.db_config import get_nomic_dir

            db_path = str(get_nomic_dir() / "debate_origins.db")
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS debate_origins (
                debate_id TEXT PRIMARY KEY,
                platform TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                metadata_json TEXT,
                thread_id TEXT,
                message_id TEXT,
                result_sent INTEGER DEFAULT 0,
                result_sent_at REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_origins_created ON debate_origins(created_at)")
        conn.commit()
        conn.close()

    def save(self, origin: DebateOrigin) -> None:
        """Save a debate origin to SQLite."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT OR REPLACE INTO debate_origins
               (debate_id, platform, channel_id, user_id, created_at,
                metadata_json, thread_id, message_id, result_sent, result_sent_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                origin.debate_id,
                origin.platform,
                origin.channel_id,
                origin.user_id,
                origin.created_at,
                json.dumps(origin.metadata),
                origin.thread_id,
                origin.message_id,
                1 if origin.result_sent else 0,
                origin.result_sent_at,
            ),
        )
        conn.commit()
        conn.close()

    def get(self, debate_id: str) -> DebateOrigin | None:
        """Get a debate origin by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT * FROM debate_origins WHERE debate_id = ?", (debate_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return DebateOrigin(
                debate_id=row[0],
                platform=row[1],
                channel_id=row[2],
                user_id=row[3],
                created_at=row[4],
                metadata=json.loads(row[5]) if row[5] else {},
                thread_id=row[6],
                message_id=row[7],
                result_sent=bool(row[8]),
                result_sent_at=row[9],
            )
        return None

    def cleanup_expired(self, ttl_seconds: int = ORIGIN_TTL_SECONDS) -> int:
        """Remove expired origin records."""
        cutoff = time.time() - ttl_seconds
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("DELETE FROM debate_origins WHERE created_at < ?", (cutoff,))
        count = cursor.rowcount
        conn.commit()
        conn.close()
        return count

    # Async methods using thread pool to avoid blocking event loop

    async def save_async(self, origin: DebateOrigin) -> None:
        """Async version of save that doesn't block event loop."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(_get_sqlite_executor(), self.save, origin)

    async def get_async(self, debate_id: str) -> DebateOrigin | None:
        """Async version of get that doesn't block event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_get_sqlite_executor(), self.get, debate_id)

    async def cleanup_expired_async(self, ttl_seconds: int = ORIGIN_TTL_SECONDS) -> int:
        """Async version of cleanup_expired that doesn't block event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _get_sqlite_executor(),
            partial(self.cleanup_expired, ttl_seconds),
        )


class PostgresOriginStore:
    """PostgreSQL-backed debate origin store for multi-instance deployments."""

    SCHEMA_NAME = "debate_origins"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS debate_origins (
            debate_id TEXT PRIMARY KEY,
            platform TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            created_at DOUBLE PRECISION NOT NULL,
            metadata_json TEXT,
            thread_id TEXT,
            message_id TEXT,
            result_sent BOOLEAN DEFAULT FALSE,
            result_sent_at DOUBLE PRECISION,
            expires_at DOUBLE PRECISION
        );
        CREATE INDEX IF NOT EXISTS idx_origins_created ON debate_origins(created_at);
        CREATE INDEX IF NOT EXISTS idx_origins_expires ON debate_origins(expires_at);
    """

    def __init__(self, pool: Pool):
        self._pool = pool
        self._initialized = False
        logger.info("PostgresOriginStore initialized")

    async def initialize(self) -> None:
        """Initialize database schema."""
        if self._initialized:
            return

        async with self._pool.acquire() as conn:
            await conn.execute(self.INITIAL_SCHEMA)

        self._initialized = True
        logger.debug(f"[{self.SCHEMA_NAME}] Schema initialized")

    async def save(self, origin: DebateOrigin) -> None:
        """Save a debate origin to PostgreSQL."""
        expires_at = origin.created_at + ORIGIN_TTL_SECONDS
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO debate_origins
                   (debate_id, platform, channel_id, user_id, created_at,
                    metadata_json, thread_id, message_id, result_sent, result_sent_at, expires_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                   ON CONFLICT (debate_id) DO UPDATE SET
                    platform = EXCLUDED.platform,
                    channel_id = EXCLUDED.channel_id,
                    user_id = EXCLUDED.user_id,
                    metadata_json = EXCLUDED.metadata_json,
                    thread_id = EXCLUDED.thread_id,
                    message_id = EXCLUDED.message_id,
                    result_sent = EXCLUDED.result_sent,
                    result_sent_at = EXCLUDED.result_sent_at,
                    expires_at = EXCLUDED.expires_at""",
                origin.debate_id,
                origin.platform,
                origin.channel_id,
                origin.user_id,
                origin.created_at,
                json.dumps(origin.metadata),
                origin.thread_id,
                origin.message_id,
                origin.result_sent,
                origin.result_sent_at,
                expires_at,
            )

    async def get(self, debate_id: str) -> DebateOrigin | None:
        """Get a debate origin by ID."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM debate_origins WHERE debate_id = $1", debate_id
            )
            if row:
                return DebateOrigin(
                    debate_id=row["debate_id"],
                    platform=row["platform"],
                    channel_id=row["channel_id"],
                    user_id=row["user_id"],
                    created_at=row["created_at"],
                    metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
                    thread_id=row["thread_id"],
                    message_id=row["message_id"],
                    result_sent=row["result_sent"],
                    result_sent_at=row["result_sent_at"],
                )
            return None

    async def cleanup_expired(self, ttl_seconds: int = ORIGIN_TTL_SECONDS) -> int:
        """Remove expired origin records."""
        cutoff = time.time() - ttl_seconds
        async with self._pool.acquire() as conn:
            result = await conn.execute("DELETE FROM debate_origins WHERE created_at < $1", cutoff)
            count = int(result.split()[-1]) if result else 0
            return count


# Lazy-loaded stores
_sqlite_store: SQLiteOriginStore | None = None
_postgres_store: PostgresOriginStore | None = None


def _get_sqlite_store() -> SQLiteOriginStore:
    """Get or create the SQLite origin store."""
    global _sqlite_store
    if _sqlite_store is None:
        _sqlite_store = SQLiteOriginStore()
    return _sqlite_store


async def _get_postgres_store() -> PostgresOriginStore | None:
    """Get or create the PostgreSQL origin store if configured."""
    global _postgres_store
    if _postgres_store is not None:
        return _postgres_store

    # Check if PostgreSQL is configured
    backend = os.environ.get("ARAGORA_DB_BACKEND", "sqlite").lower()
    if backend not in ("postgres", "postgresql"):
        return None

    try:
        from aragora.storage.postgres_store import get_postgres_pool

        pool = await get_postgres_pool()
        _postgres_store = PostgresOriginStore(pool)
        await _postgres_store.initialize()
        logger.info("PostgreSQL origin store initialized")
        return _postgres_store
    except Exception as e:
        logger.warning(f"PostgreSQL origin store not available: {e}")
        return None


def _get_postgres_store_sync() -> PostgresOriginStore | None:
    """Synchronous wrapper for getting PostgreSQL store."""
    try:
        asyncio.get_running_loop()
        # Can't use run_until_complete in async context
        return _postgres_store
    except RuntimeError:
        # No running event loop, try to run async getter
        try:
            return asyncio.run(_get_postgres_store())
        except RuntimeError:
            return None
