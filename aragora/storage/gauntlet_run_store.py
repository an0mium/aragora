"""
Gauntlet Run Storage Backends.

Persistent storage for in-flight gauntlet runs, results, and history.

Backends:
- InMemoryGauntletRunStore: For testing
- SQLiteGauntletRunStore: For single-instance deployments
- RedisGauntletRunStore: For multi-instance (with SQLite fallback)

Usage:
    from aragora.storage.gauntlet_run_store import (
        get_gauntlet_run_store,
        set_gauntlet_run_store,
    )

    # Use default store (configured via environment)
    store = get_gauntlet_run_store()
    await store.save(run_data_dict)
    data = await store.get("run-123")
"""

from __future__ import annotations

import json
import logging
import os
import threading
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aragora.storage.generic_store import (
    GenericInMemoryStore,
    GenericPostgresStore,
    GenericSQLiteStore,
    GenericStoreBackend,
)

if TYPE_CHECKING:
    pass

# redis.exceptions.ConnectionError does NOT inherit from builtins.ConnectionError
# (it inherits from redis.RedisError -> Exception), so we need to catch it explicitly.
try:
    from redis.exceptions import RedisError as _RedisError  # noqa: F401
except ImportError:
    _RedisError = OSError  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


def _batch_deserialize_json(rows: list[tuple[str, ...]], idx: int = 0) -> list[dict[str, Any]]:
    """Batch deserialize JSON from query results.

    Pre-allocates the result list for better memory efficiency.
    Skips json.loads for already-parsed JSONB results (asyncpg returns dicts).

    Args:
        rows: List of tuples from cursor.fetchall()
        idx: Index of the JSON column in each row (default 0)

    Returns:
        List of deserialized dictionaries
    """
    if not rows:
        return []

    results: list[dict[str, Any]] = [{}] * len(rows)
    for i, row in enumerate(rows):
        data = row[idx]
        results[i] = json.loads(data) if isinstance(data, str) else data
    return results


# Global singleton
_gauntlet_run_store: GauntletRunStoreBackend | None = None
_store_lock = threading.RLock()


@dataclass
class GauntletRunItem:
    """
    Gauntlet run data for persistence.

    This is a storage-friendly representation of an in-flight gauntlet run.
    """

    run_id: str
    template_id: str
    status: str = "pending"  # pending, running, completed, failed, cancelled
    config_data: dict[str, Any] = field(default_factory=dict)
    result_data: dict[str, Any] | None = None

    # Timing
    started_at: str | None = None
    completed_at: str | None = None

    # Metadata
    triggered_by: str | None = None
    workspace_id: str | None = None
    tags: list[str] = field(default_factory=list)

    # Timestamps
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        """Set default timestamps if not provided."""
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "template_id": self.template_id,
            "status": self.status,
            "config_data": self.config_data,
            "result_data": self.result_data,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "triggered_by": self.triggered_by,
            "workspace_id": self.workspace_id,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GauntletRunItem:
        """Create from dictionary."""
        return cls(
            run_id=data.get("run_id", ""),
            template_id=data.get("template_id", ""),
            status=data.get("status", "pending"),
            config_data=data.get("config_data", {}),
            result_data=data.get("result_data"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            triggered_by=data.get("triggered_by"),
            workspace_id=data.get("workspace_id"),
            tags=data.get("tags", []),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> GauntletRunItem:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


class GauntletRunStoreBackend(GenericStoreBackend):
    """Abstract base class for gauntlet run storage backends."""

    @abstractmethod
    async def list_by_status(self, status: str) -> list[dict[str, Any]]:
        """List runs by status."""
        ...

    @abstractmethod
    async def list_by_template(self, template_id: str) -> list[dict[str, Any]]:
        """List runs by template."""
        ...

    @abstractmethod
    async def list_active(self) -> list[dict[str, Any]]:
        """List active (pending/running) runs."""
        ...

    @abstractmethod
    async def update_status(
        self, run_id: str, status: str, result_data: dict[str, Any] | None = None
    ) -> bool:
        """Update run status and optionally set result."""
        ...


class InMemoryGauntletRunStore(GenericInMemoryStore, GauntletRunStoreBackend):
    """
    In-memory gauntlet run store for testing.

    Data is lost on restart.
    """

    PRIMARY_KEY = "run_id"

    async def list_by_status(self, status: str) -> list[dict[str, Any]]:
        return self._filter_by("status", status)

    async def list_by_template(self, template_id: str) -> list[dict[str, Any]]:
        return self._filter_by("template_id", template_id)

    async def list_active(self) -> list[dict[str, Any]]:
        with self._lock:
            return [r for r in self._data.values() if r.get("status") in ("pending", "running")]

    async def update_status(
        self, run_id: str, status: str, result_data: dict[str, Any] | None = None
    ) -> bool:
        with self._lock:
            if run_id not in self._data:
                return False
            self._data[run_id]["status"] = status
            self._data[run_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
            if status == "running" and not self._data[run_id].get("started_at"):
                self._data[run_id]["started_at"] = datetime.now(timezone.utc).isoformat()
            if status in ("completed", "failed", "cancelled"):
                self._data[run_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
            if result_data is not None:
                self._data[run_id]["result_data"] = result_data
            return True


class SQLiteGauntletRunStore(GenericSQLiteStore, GauntletRunStoreBackend):
    """
    SQLite-backed gauntlet run store.

    Suitable for single-instance deployments.
    """

    TABLE_NAME = "gauntlet_runs"
    PRIMARY_KEY = "run_id"
    SCHEMA_SQL = """
        CREATE TABLE IF NOT EXISTS gauntlet_runs (
            run_id TEXT PRIMARY KEY,
            template_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            triggered_by TEXT,
            workspace_id TEXT,
            started_at TEXT,
            completed_at TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            data_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_gauntlet_run_status ON gauntlet_runs(status);
        CREATE INDEX IF NOT EXISTS idx_gauntlet_run_template ON gauntlet_runs(template_id);
        CREATE INDEX IF NOT EXISTS idx_gauntlet_run_workspace ON gauntlet_runs(workspace_id);
    """
    INDEX_COLUMNS = {
        "template_id",
        "status",
        "triggered_by",
        "workspace_id",
        "started_at",
        "completed_at",
    }

    async def list_all(self, limit: int = 0, offset: int = 0) -> list[dict[str, Any]]:  # type: ignore[override]
        """List all runs with optional pagination."""
        if limit > 0:
            return self._query_with_sql(
                "1=1", order_by=f"created_at DESC LIMIT {limit} OFFSET {offset}"
            )
        return self._query_with_sql("1=1")

    async def list_by_status(
        self, status: str, limit: int = 0, offset: int = 0
    ) -> list[dict[str, Any]]:
        """List runs by status with optional pagination."""
        if limit > 0:
            return self._query_with_sql(
                "status = ?",
                (status,),
                order_by=f"created_at DESC LIMIT {limit} OFFSET {offset}",
            )
        return self._query_by_column("status", status)

    async def list_by_template(
        self, template_id: str, limit: int = 0, offset: int = 0
    ) -> list[dict[str, Any]]:
        """List runs by template with optional pagination."""
        if limit > 0:
            return self._query_with_sql(
                "template_id = ?",
                (template_id,),
                order_by=f"created_at DESC LIMIT {limit} OFFSET {offset}",
            )
        return self._query_by_column("template_id", template_id)

    async def list_active(self, limit: int = 0) -> list[dict[str, Any]]:
        """List active (pending/running) runs with optional limit."""
        if limit > 0:
            return self._query_with_sql(
                "status IN ('pending', 'running')",
                order_by=f"created_at DESC LIMIT {limit}",
            )
        return self._query_with_sql("status IN ('pending', 'running')")

    async def get_queue_analytics(self) -> dict[str, Any]:
        """Get queue analytics using window functions."""
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT
                        run_id,
                        template_id,
                        status,
                        created_at,
                        ROW_NUMBER() OVER (
                            PARTITION BY status
                            ORDER BY created_at ASC
                        ) as position_in_status,
                        COUNT(*) OVER (PARTITION BY status) as status_count,
                        COUNT(*) OVER () as total_count
                    FROM gauntlet_runs
                    WHERE status IN ('pending', 'running')
                    ORDER BY
                        CASE status WHEN 'running' THEN 0 ELSE 1 END,
                        created_at ASC
                    """)
                rows = cursor.fetchall()

                if not rows:
                    return {
                        "total_active": 0,
                        "pending_count": 0,
                        "running_count": 0,
                        "queue": [],
                    }

                queue = []
                status_counts: dict[str, int] = {}
                for row in rows:
                    run_id, template_id, status, created_at, position, status_count, total = row
                    queue.append(
                        {
                            "run_id": run_id,
                            "template_id": template_id,
                            "status": status,
                            "created_at": created_at,
                            "position_in_status": position,
                        }
                    )
                    status_counts[status] = status_count

                return {
                    "total_active": rows[0][6] if rows else 0,
                    "pending_count": status_counts.get("pending", 0),
                    "running_count": status_counts.get("running", 0),
                    "queue": queue,
                }
            finally:
                conn.close()

    async def update_status(
        self, run_id: str, status: str, result_data: dict[str, Any] | None = None
    ) -> bool:
        updates: dict[str, Any] = {
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if status == "running":
            # Only set started_at if not already set - need to check current data
            existing = await self.get(run_id)
            if not existing:
                return False
            if not existing.get("started_at"):
                updates["started_at"] = datetime.now(timezone.utc).isoformat()
        if status in ("completed", "failed", "cancelled"):
            updates["completed_at"] = datetime.now(timezone.utc).isoformat()
        if result_data is not None:
            updates["result_data"] = result_data

        return self._update_json_field(
            run_id,
            updates,
            extra_column_updates={
                k: v for k, v in updates.items() if k in ("status", "started_at", "completed_at")
            },
        )


class RedisGauntletRunStore(GauntletRunStoreBackend):
    """
    Redis-backed gauntlet run store with SQLite fallback.

    For multi-instance deployments with optional horizontal scaling.
    Falls back to SQLite if Redis is unavailable.
    """

    REDIS_PREFIX = "aragora:gauntlet_run:"
    REDIS_INDEX_STATUS = "aragora:gauntlet_run:idx:status:"
    REDIS_INDEX_TEMPLATE = "aragora:gauntlet_run:idx:template:"

    def __init__(
        self,
        fallback_db_path: Path | None = None,
        redis_url: str | None = None,
    ) -> None:
        self._redis_url = redis_url or os.getenv("ARAGORA_REDIS_URL", "")
        self._redis_client: Any = None
        self._fallback = SQLiteGauntletRunStore(fallback_db_path)
        self._using_fallback = False
        self._lock = threading.RLock()
        self._connect_redis()

    def _connect_redis(self) -> None:
        """Attempt to connect to Redis."""
        if not self._redis_url:
            logger.info("No Redis URL configured, using SQLite fallback")
            self._using_fallback = True
            return

        try:
            import redis

            self._redis_client = redis.from_url(self._redis_url)
            self._redis_client.ping()
            logger.info("Connected to Redis for gauntlet run storage")
            self._using_fallback = False
        except (ImportError, _RedisError, ConnectionError, TimeoutError, OSError, ValueError) as e:
            logger.warning(f"Redis connection failed, using SQLite fallback: {e}")
            self._using_fallback = True
            self._redis_client = None

    async def get(self, run_id: str) -> dict[str, Any] | None:
        if self._using_fallback:
            return await self._fallback.get(run_id)
        try:
            data = self._redis_client.get(f"{self.REDIS_PREFIX}{run_id}")
            if data:
                return json.loads(data)
            return None
        except (_RedisError, ConnectionError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as e:
            logger.warning(f"Redis get failed, using fallback: {e}")
            return await self._fallback.get(run_id)

    async def save(self, data: dict[str, Any]) -> None:
        run_id = data.get("run_id")
        if not run_id:
            raise ValueError("run_id is required")

        await self._fallback.save(data)

        if self._using_fallback:
            return

        try:
            data_json = json.dumps(data)
            pipe = self._redis_client.pipeline()
            pipe.set(f"{self.REDIS_PREFIX}{run_id}", data_json)
            status = data.get("status", "pending")
            pipe.sadd(f"{self.REDIS_INDEX_STATUS}{status}", run_id)
            template_id = data.get("template_id")
            if template_id:
                pipe.sadd(f"{self.REDIS_INDEX_TEMPLATE}{template_id}", run_id)
            pipe.execute()
        except (_RedisError, ConnectionError, TimeoutError, OSError, ValueError) as e:
            logger.warning(f"Redis save failed (SQLite fallback used): {e}")

    async def delete(self, run_id: str) -> bool:
        result = await self._fallback.delete(run_id)

        if self._using_fallback:
            return result

        try:
            data = self._redis_client.get(f"{self.REDIS_PREFIX}{run_id}")
            if data:
                run_data = json.loads(data)
                pipe = self._redis_client.pipeline()
                if run_data.get("status"):
                    pipe.srem(
                        f"{self.REDIS_INDEX_STATUS}{run_data['status']}",
                        run_id,
                    )
                if run_data.get("template_id"):
                    pipe.srem(
                        f"{self.REDIS_INDEX_TEMPLATE}{run_data['template_id']}",
                        run_id,
                    )
                pipe.delete(f"{self.REDIS_PREFIX}{run_id}")
                pipe.execute()
                return True
            return result
        except (_RedisError, ConnectionError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as e:
            logger.warning(f"Redis delete failed: {e}")
            return result

    async def list_all(self) -> list[dict[str, Any]]:
        if self._using_fallback:
            return await self._fallback.list_all()

        try:
            results: list[dict[str, Any]] = []
            cursor = "0"
            while cursor != 0:
                cursor, keys = self._redis_client.scan(
                    cursor=cursor,
                    match=f"{self.REDIS_PREFIX}*",
                    count=100,
                )
                if keys:
                    data_keys = [k for k in keys if b":idx:" not in k and b"idx:" not in k]
                    if data_keys:
                        values = self._redis_client.mget(data_keys)
                        valid_values = [v for v in values if v]
                        if valid_values:
                            results.extend(_batch_deserialize_json([(v,) for v in valid_values]))
            return results
        except (_RedisError, ConnectionError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as e:
            logger.warning(f"Redis list_all failed, using fallback: {e}")
            return await self._fallback.list_all()

    async def list_by_status(self, status: str) -> list[dict[str, Any]]:
        if self._using_fallback:
            return await self._fallback.list_by_status(status)

        try:
            run_ids = self._redis_client.smembers(f"{self.REDIS_INDEX_STATUS}{status}")
            if not run_ids:
                return []
            keys = [f"{self.REDIS_PREFIX}{rid.decode()}" for rid in run_ids]
            values = self._redis_client.mget(keys)
            valid_values = [v for v in values if v]
            return _batch_deserialize_json([(v,) for v in valid_values])
        except (_RedisError, ConnectionError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as e:
            logger.warning(f"Redis list_by_status failed, using fallback: {e}")
            return await self._fallback.list_by_status(status)

    async def list_by_template(self, template_id: str) -> list[dict[str, Any]]:
        if self._using_fallback:
            return await self._fallback.list_by_template(template_id)

        try:
            run_ids = self._redis_client.smembers(f"{self.REDIS_INDEX_TEMPLATE}{template_id}")
            if not run_ids:
                return []
            keys = [f"{self.REDIS_PREFIX}{rid.decode()}" for rid in run_ids]
            values = self._redis_client.mget(keys)
            valid_values = [v for v in values if v]
            return _batch_deserialize_json([(v,) for v in valid_values])
        except (_RedisError, ConnectionError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as e:
            logger.warning(f"Redis list_by_template failed, using fallback: {e}")
            return await self._fallback.list_by_template(template_id)

    async def list_active(self) -> list[dict[str, Any]]:
        if self._using_fallback:
            return await self._fallback.list_active()

        try:
            pending = await self.list_by_status("pending")
            running = await self.list_by_status("running")
            return pending + running
        except (_RedisError, ConnectionError, TimeoutError, OSError, ValueError) as e:
            logger.warning(f"Redis list_active failed, using fallback: {e}")
            return await self._fallback.list_active()

    async def update_status(
        self, run_id: str, status: str, result_data: dict[str, Any] | None = None
    ) -> bool:
        result = await self._fallback.update_status(run_id, status, result_data)

        if self._using_fallback:
            return result

        try:
            data_bytes = self._redis_client.get(f"{self.REDIS_PREFIX}{run_id}")
            if not data_bytes:
                return result

            data = json.loads(data_bytes)
            old_status = data.get("status")

            data["status"] = status
            data["updated_at"] = datetime.now(timezone.utc).isoformat()

            if status == "running" and not data.get("started_at"):
                data["started_at"] = datetime.now(timezone.utc).isoformat()
            if status in ("completed", "failed", "cancelled"):
                data["completed_at"] = datetime.now(timezone.utc).isoformat()
            if result_data is not None:
                data["result_data"] = result_data

            pipe = self._redis_client.pipeline()
            pipe.set(f"{self.REDIS_PREFIX}{run_id}", json.dumps(data))
            if old_status and old_status != status:
                pipe.srem(f"{self.REDIS_INDEX_STATUS}{old_status}", run_id)
            pipe.sadd(f"{self.REDIS_INDEX_STATUS}{status}", run_id)
            pipe.execute()
            return True
        except (_RedisError, ConnectionError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as e:
            logger.warning(f"Redis update_status failed: {e}")
            return result

    async def close(self) -> None:
        await self._fallback.close()
        if self._redis_client:
            try:
                self._redis_client.close()
            except (ConnectionError, OSError) as e:
                logger.debug(f"Redis close failed (connection already closed): {e}")
            except (_RedisError, ConnectionError, TimeoutError, OSError) as e:
                logger.debug(f"Redis close failed: {e}")


class PostgresGauntletRunStore(GenericPostgresStore, GauntletRunStoreBackend):
    """
    PostgreSQL-backed gauntlet run store.

    Async implementation for production multi-instance deployments
    with horizontal scaling and concurrent writes.
    """

    TABLE_NAME = "gauntlet_runs"
    PRIMARY_KEY = "run_id"
    SCHEMA_SQL = """
        CREATE TABLE IF NOT EXISTS gauntlet_runs (
            run_id TEXT PRIMARY KEY,
            template_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            triggered_by TEXT,
            workspace_id TEXT,
            started_at TIMESTAMPTZ,
            completed_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            data_json JSONB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_gauntlet_run_status ON gauntlet_runs(status);
        CREATE INDEX IF NOT EXISTS idx_gauntlet_run_template ON gauntlet_runs(template_id);
        CREATE INDEX IF NOT EXISTS idx_gauntlet_run_workspace ON gauntlet_runs(workspace_id);
    """
    INDEX_COLUMNS = {
        "template_id",
        "status",
        "triggered_by",
        "workspace_id",
        "started_at",
        "completed_at",
    }

    async def list_by_status(self, status: str) -> list[dict[str, Any]]:
        return await self._query_by_column("status", status)

    async def list_by_template(self, template_id: str) -> list[dict[str, Any]]:
        return await self._query_by_column("template_id", template_id)

    async def list_active(self) -> list[dict[str, Any]]:
        return await self._query_with_sql("status IN ('pending', 'running')")

    async def update_status(
        self, run_id: str, status: str, result_data: dict[str, Any] | None = None
    ) -> bool:
        updates: dict[str, Any] = {
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if status == "running":
            existing = await self.get(run_id)
            if not existing:
                return False
            if not existing.get("started_at"):
                updates["started_at"] = datetime.now(timezone.utc).isoformat()
        if status in ("completed", "failed", "cancelled"):
            updates["completed_at"] = datetime.now(timezone.utc).isoformat()
        if result_data is not None:
            updates["result_data"] = result_data

        return await self._update_json_field(
            run_id,
            updates,
            extra_column_updates={
                k: v for k, v in updates.items() if k in ("status", "started_at", "completed_at")
            },
        )

    # Sync wrappers - use run_async for sync access:
    #   from aragora.utils.async_utils import run_async
    #   store = PostgresGauntletRunStore(...)
    #   result = run_async(store.get(run_id))


def get_gauntlet_run_store() -> GauntletRunStoreBackend:
    """
    Get the global gauntlet run store instance.

    Backend is selected based on environment variables:
    - ARAGORA_GAUNTLET_STORE_BACKEND: "memory", "sqlite", "postgres", "supabase", or "redis"
    - ARAGORA_DB_BACKEND: fallback if ARAGORA_GAUNTLET_STORE_BACKEND not set

    Uses unified Supabase -> PostgreSQL -> SQLite preference order.
    """
    global _gauntlet_run_store

    with _store_lock:
        if _gauntlet_run_store is not None:
            return _gauntlet_run_store

        backend = os.getenv("ARAGORA_GAUNTLET_STORE_BACKEND")
        if not backend:
            backend = os.getenv("ARAGORA_DB_BACKEND", "auto")
        backend = backend.lower()

        if backend == "redis":
            _gauntlet_run_store = RedisGauntletRunStore()
            logger.info("Using Redis gauntlet run store")
            return _gauntlet_run_store

        from aragora.storage.connection_factory import create_persistent_store

        _gauntlet_run_store = create_persistent_store(
            store_name="gauntlet",
            sqlite_class=SQLiteGauntletRunStore,
            postgres_class=PostgresGauntletRunStore,
            db_filename="gauntlet_runs.db",
            memory_class=InMemoryGauntletRunStore,
        )

        return _gauntlet_run_store


def set_gauntlet_run_store(store: GauntletRunStoreBackend) -> None:
    """Set a custom gauntlet run store instance."""
    global _gauntlet_run_store

    with _store_lock:
        _gauntlet_run_store = store


def reset_gauntlet_run_store() -> None:
    """Reset the global gauntlet run store (for testing)."""
    global _gauntlet_run_store

    with _store_lock:
        _gauntlet_run_store = None


__all__ = [
    "GauntletRunItem",
    "GauntletRunStoreBackend",
    "InMemoryGauntletRunStore",
    "SQLiteGauntletRunStore",
    "RedisGauntletRunStore",
    "PostgresGauntletRunStore",
    "get_gauntlet_run_store",
    "set_gauntlet_run_store",
    "reset_gauntlet_run_store",
]
