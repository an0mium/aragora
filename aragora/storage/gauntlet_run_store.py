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
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Global singleton
_gauntlet_run_store: Optional["GauntletRunStoreBackend"] = None
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
    result_data: Optional[dict[str, Any]] = None

    # Timing
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Metadata
    triggered_by: Optional[str] = None
    workspace_id: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    # Timestamps
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        """Set default timestamps if not provided."""
        now = datetime.utcnow().isoformat()
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
    def from_dict(cls, data: dict[str, Any]) -> "GauntletRunItem":
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
    def from_json(cls, json_str: str) -> "GauntletRunItem":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


class GauntletRunStoreBackend(ABC):
    """Abstract base class for gauntlet run storage backends."""

    @abstractmethod
    async def get(self, run_id: str) -> Optional[dict[str, Any]]:
        """Get run data by ID."""
        pass

    @abstractmethod
    async def save(self, data: dict[str, Any]) -> None:
        """Save run data."""
        pass

    @abstractmethod
    async def delete(self, run_id: str) -> bool:
        """Delete run data."""
        pass

    @abstractmethod
    async def list_all(self) -> list[dict[str, Any]]:
        """List all runs."""
        pass

    @abstractmethod
    async def list_by_status(self, status: str) -> list[dict[str, Any]]:
        """List runs by status."""
        pass

    @abstractmethod
    async def list_by_template(self, template_id: str) -> list[dict[str, Any]]:
        """List runs by template."""
        pass

    @abstractmethod
    async def list_active(self) -> list[dict[str, Any]]:
        """List active (pending/running) runs."""
        pass

    @abstractmethod
    async def update_status(
        self, run_id: str, status: str, result_data: Optional[dict[str, Any]] = None
    ) -> bool:
        """Update run status and optionally set result."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close any resources."""
        pass


class InMemoryGauntletRunStore(GauntletRunStoreBackend):
    """
    In-memory gauntlet run store for testing.

    Data is lost on restart.
    """

    def __init__(self) -> None:
        """Initialize in-memory store."""
        self._data: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    async def get(self, run_id: str) -> Optional[dict[str, Any]]:
        """Get run data by ID."""
        with self._lock:
            return self._data.get(run_id)

    async def save(self, data: dict[str, Any]) -> None:
        """Save run data."""
        run_id = data.get("run_id")
        if not run_id:
            raise ValueError("run_id is required")
        with self._lock:
            self._data[run_id] = data

    async def delete(self, run_id: str) -> bool:
        """Delete run data."""
        with self._lock:
            if run_id in self._data:
                del self._data[run_id]
                return True
            return False

    async def list_all(self) -> list[dict[str, Any]]:
        """List all runs."""
        with self._lock:
            return list(self._data.values())

    async def list_by_status(self, status: str) -> list[dict[str, Any]]:
        """List runs by status."""
        with self._lock:
            return [r for r in self._data.values() if r.get("status") == status]

    async def list_by_template(self, template_id: str) -> list[dict[str, Any]]:
        """List runs by template."""
        with self._lock:
            return [r for r in self._data.values() if r.get("template_id") == template_id]

    async def list_active(self) -> list[dict[str, Any]]:
        """List active (pending/running) runs."""
        with self._lock:
            return [
                r
                for r in self._data.values()
                if r.get("status") in ("pending", "running")
            ]

    async def update_status(
        self, run_id: str, status: str, result_data: Optional[dict[str, Any]] = None
    ) -> bool:
        """Update run status and optionally set result."""
        with self._lock:
            if run_id not in self._data:
                return False
            self._data[run_id]["status"] = status
            self._data[run_id]["updated_at"] = datetime.utcnow().isoformat()
            if status == "running" and not self._data[run_id].get("started_at"):
                self._data[run_id]["started_at"] = datetime.utcnow().isoformat()
            if status in ("completed", "failed", "cancelled"):
                self._data[run_id]["completed_at"] = datetime.utcnow().isoformat()
            if result_data is not None:
                self._data[run_id]["result_data"] = result_data
            return True

    async def close(self) -> None:
        """No-op for in-memory store."""
        pass


class SQLiteGauntletRunStore(GauntletRunStoreBackend):
    """
    SQLite-backed gauntlet run store.

    Suitable for single-instance deployments.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """
        Initialize SQLite store.

        Args:
            db_path: Path to SQLite database. Defaults to
                     $ARAGORA_DATA_DIR/gauntlet_runs.db
        """
        if db_path is None:
            data_dir = os.getenv("ARAGORA_DATA_DIR", "/tmp/aragora")
            db_path = Path(data_dir) / "gauntlet_runs.db"

        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
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
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_gauntlet_run_status
                    ON gauntlet_runs(status)
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_gauntlet_run_template
                    ON gauntlet_runs(template_id)
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_gauntlet_run_workspace
                    ON gauntlet_runs(workspace_id)
                    """
                )
                conn.commit()
            finally:
                conn.close()

    async def get(self, run_id: str) -> Optional[dict[str, Any]]:
        """Get run data by ID."""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT data_json FROM gauntlet_runs WHERE run_id = ?",
                    (run_id,),
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
                return None
            finally:
                conn.close()

    async def save(self, data: dict[str, Any]) -> None:
        """Save run data."""
        run_id = data.get("run_id")
        if not run_id:
            raise ValueError("run_id is required")

        now = time.time()
        data_json = json.dumps(data)

        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO gauntlet_runs
                    (run_id, template_id, status, triggered_by, workspace_id,
                     started_at, completed_at, created_at, updated_at, data_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        data.get("template_id", ""),
                        data.get("status", "pending"),
                        data.get("triggered_by"),
                        data.get("workspace_id"),
                        data.get("started_at"),
                        data.get("completed_at"),
                        now,
                        now,
                        data_json,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    async def delete(self, run_id: str) -> bool:
        """Delete run data."""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM gauntlet_runs WHERE run_id = ?",
                    (run_id,),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    async def list_all(self) -> list[dict[str, Any]]:
        """List all runs."""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT data_json FROM gauntlet_runs ORDER BY created_at DESC")
                return [json.loads(row[0]) for row in cursor.fetchall()]
            finally:
                conn.close()

    async def list_by_status(self, status: str) -> list[dict[str, Any]]:
        """List runs by status."""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT data_json FROM gauntlet_runs WHERE status = ? ORDER BY created_at DESC",
                    (status,),
                )
                return [json.loads(row[0]) for row in cursor.fetchall()]
            finally:
                conn.close()

    async def list_by_template(self, template_id: str) -> list[dict[str, Any]]:
        """List runs by template."""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT data_json FROM gauntlet_runs WHERE template_id = ? ORDER BY created_at DESC",
                    (template_id,),
                )
                return [json.loads(row[0]) for row in cursor.fetchall()]
            finally:
                conn.close()

    async def list_active(self) -> list[dict[str, Any]]:
        """List active (pending/running) runs."""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT data_json FROM gauntlet_runs
                    WHERE status IN ('pending', 'running')
                    ORDER BY created_at DESC
                    """
                )
                return [json.loads(row[0]) for row in cursor.fetchall()]
            finally:
                conn.close()

    async def update_status(
        self, run_id: str, status: str, result_data: Optional[dict[str, Any]] = None
    ) -> bool:
        """Update run status and optionally set result."""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                # Get current data
                cursor.execute(
                    "SELECT data_json FROM gauntlet_runs WHERE run_id = ?",
                    (run_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return False

                data = json.loads(row[0])
                data["status"] = status
                data["updated_at"] = datetime.utcnow().isoformat()

                if status == "running" and not data.get("started_at"):
                    data["started_at"] = datetime.utcnow().isoformat()
                if status in ("completed", "failed", "cancelled"):
                    data["completed_at"] = datetime.utcnow().isoformat()
                if result_data is not None:
                    data["result_data"] = result_data

                cursor.execute(
                    """
                    UPDATE gauntlet_runs
                    SET status = ?, started_at = ?, completed_at = ?,
                        updated_at = ?, data_json = ?
                    WHERE run_id = ?
                    """,
                    (
                        status,
                        data.get("started_at"),
                        data.get("completed_at"),
                        time.time(),
                        json.dumps(data),
                        run_id,
                    ),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    async def close(self) -> None:
        """No-op for SQLite (connections are per-operation)."""
        pass


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
        fallback_db_path: Optional[Path] = None,
        redis_url: Optional[str] = None,
    ) -> None:
        """
        Initialize Redis store with SQLite fallback.

        Args:
            fallback_db_path: Path for SQLite fallback database
            redis_url: Redis connection URL (defaults to ARAGORA_REDIS_URL env var)
        """
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
        except Exception as e:
            logger.warning(f"Redis connection failed, using SQLite fallback: {e}")
            self._using_fallback = True
            self._redis_client = None

    async def get(self, run_id: str) -> Optional[dict[str, Any]]:
        """Get run data by ID."""
        if self._using_fallback:
            return await self._fallback.get(run_id)

        try:
            data = self._redis_client.get(f"{self.REDIS_PREFIX}{run_id}")
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.warning(f"Redis get failed, using fallback: {e}")
            return await self._fallback.get(run_id)

    async def save(self, data: dict[str, Any]) -> None:
        """Save run data."""
        run_id = data.get("run_id")
        if not run_id:
            raise ValueError("run_id is required")

        # Always save to SQLite fallback for durability
        await self._fallback.save(data)

        if self._using_fallback:
            return

        try:
            data_json = json.dumps(data)
            pipe = self._redis_client.pipeline()

            # Save main data
            pipe.set(f"{self.REDIS_PREFIX}{run_id}", data_json)

            # Update status index
            status = data.get("status", "pending")
            pipe.sadd(f"{self.REDIS_INDEX_STATUS}{status}", run_id)

            # Update template index
            template_id = data.get("template_id")
            if template_id:
                pipe.sadd(f"{self.REDIS_INDEX_TEMPLATE}{template_id}", run_id)

            pipe.execute()
        except Exception as e:
            logger.warning(f"Redis save failed (SQLite fallback used): {e}")

    async def delete(self, run_id: str) -> bool:
        """Delete run data."""
        result = await self._fallback.delete(run_id)

        if self._using_fallback:
            return result

        try:
            # Get current data to clean up indexes
            data = self._redis_client.get(f"{self.REDIS_PREFIX}{run_id}")
            if data:
                run_data = json.loads(data)
                pipe = self._redis_client.pipeline()

                # Remove from status index
                if run_data.get("status"):
                    pipe.srem(
                        f"{self.REDIS_INDEX_STATUS}{run_data['status']}",
                        run_id,
                    )

                # Remove from template index
                if run_data.get("template_id"):
                    pipe.srem(
                        f"{self.REDIS_INDEX_TEMPLATE}{run_data['template_id']}",
                        run_id,
                    )

                # Delete main data
                pipe.delete(f"{self.REDIS_PREFIX}{run_id}")
                pipe.execute()
                return True
            return result
        except Exception as e:
            logger.warning(f"Redis delete failed: {e}")
            return result

    async def list_all(self) -> list[dict[str, Any]]:
        """List all runs."""
        if self._using_fallback:
            return await self._fallback.list_all()

        try:
            results = []
            cursor = "0"
            while cursor != 0:
                cursor, keys = self._redis_client.scan(
                    cursor=cursor,
                    match=f"{self.REDIS_PREFIX}*",
                    count=100,
                )
                if keys:
                    data_keys = [
                        k for k in keys if b":idx:" not in k and b"idx:" not in k
                    ]
                    if data_keys:
                        values = self._redis_client.mget(data_keys)
                        for v in values:
                            if v:
                                results.append(json.loads(v))
            return results
        except Exception as e:
            logger.warning(f"Redis list_all failed, using fallback: {e}")
            return await self._fallback.list_all()

    async def list_by_status(self, status: str) -> list[dict[str, Any]]:
        """List runs by status."""
        if self._using_fallback:
            return await self._fallback.list_by_status(status)

        try:
            run_ids = self._redis_client.smembers(f"{self.REDIS_INDEX_STATUS}{status}")
            if not run_ids:
                return []

            keys = [f"{self.REDIS_PREFIX}{rid.decode()}" for rid in run_ids]
            values = self._redis_client.mget(keys)
            return [json.loads(v) for v in values if v]
        except Exception as e:
            logger.warning(f"Redis list_by_status failed, using fallback: {e}")
            return await self._fallback.list_by_status(status)

    async def list_by_template(self, template_id: str) -> list[dict[str, Any]]:
        """List runs by template."""
        if self._using_fallback:
            return await self._fallback.list_by_template(template_id)

        try:
            run_ids = self._redis_client.smembers(
                f"{self.REDIS_INDEX_TEMPLATE}{template_id}"
            )
            if not run_ids:
                return []

            keys = [f"{self.REDIS_PREFIX}{rid.decode()}" for rid in run_ids]
            values = self._redis_client.mget(keys)
            return [json.loads(v) for v in values if v]
        except Exception as e:
            logger.warning(f"Redis list_by_template failed, using fallback: {e}")
            return await self._fallback.list_by_template(template_id)

    async def list_active(self) -> list[dict[str, Any]]:
        """List active (pending/running) runs."""
        if self._using_fallback:
            return await self._fallback.list_active()

        try:
            pending = await self.list_by_status("pending")
            running = await self.list_by_status("running")
            return pending + running
        except Exception as e:
            logger.warning(f"Redis list_active failed, using fallback: {e}")
            return await self._fallback.list_active()

    async def update_status(
        self, run_id: str, status: str, result_data: Optional[dict[str, Any]] = None
    ) -> bool:
        """Update run status and optionally set result."""
        # Update SQLite fallback
        result = await self._fallback.update_status(run_id, status, result_data)

        if self._using_fallback:
            return result

        try:
            # Get current data
            data_bytes = self._redis_client.get(f"{self.REDIS_PREFIX}{run_id}")
            if not data_bytes:
                return result

            data = json.loads(data_bytes)
            old_status = data.get("status")

            data["status"] = status
            data["updated_at"] = datetime.utcnow().isoformat()

            if status == "running" and not data.get("started_at"):
                data["started_at"] = datetime.utcnow().isoformat()
            if status in ("completed", "failed", "cancelled"):
                data["completed_at"] = datetime.utcnow().isoformat()
            if result_data is not None:
                data["result_data"] = result_data

            pipe = self._redis_client.pipeline()

            # Update main data
            pipe.set(f"{self.REDIS_PREFIX}{run_id}", json.dumps(data))

            # Update status indexes
            if old_status and old_status != status:
                pipe.srem(f"{self.REDIS_INDEX_STATUS}{old_status}", run_id)
            pipe.sadd(f"{self.REDIS_INDEX_STATUS}{status}", run_id)

            pipe.execute()
            return True
        except Exception as e:
            logger.warning(f"Redis update_status failed: {e}")
            return result

    async def close(self) -> None:
        """Close connections."""
        await self._fallback.close()
        if self._redis_client:
            try:
                self._redis_client.close()
            except (ConnectionError, OSError) as e:
                logger.debug(f"Redis close failed (connection already closed): {e}")
            except Exception as e:
                logger.debug(f"Redis close failed: {e}")


def get_gauntlet_run_store() -> GauntletRunStoreBackend:
    """
    Get the global gauntlet run store instance.

    Backend is selected based on ARAGORA_GAUNTLET_STORE_BACKEND env var:
    - "memory": InMemoryGauntletRunStore (for testing)
    - "sqlite": SQLiteGauntletRunStore (default, single-instance)
    - "redis": RedisGauntletRunStore (multi-instance)
    """
    global _gauntlet_run_store

    with _store_lock:
        if _gauntlet_run_store is not None:
            return _gauntlet_run_store

        backend = os.getenv("ARAGORA_GAUNTLET_STORE_BACKEND", "sqlite").lower()

        if backend == "memory":
            _gauntlet_run_store = InMemoryGauntletRunStore()
            logger.info("Using in-memory gauntlet run store")
        elif backend == "redis":
            _gauntlet_run_store = RedisGauntletRunStore()
            logger.info("Using Redis gauntlet run store")
        else:  # Default to SQLite
            _gauntlet_run_store = SQLiteGauntletRunStore()
            logger.info("Using SQLite gauntlet run store")

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
    "get_gauntlet_run_store",
    "set_gauntlet_run_store",
    "reset_gauntlet_run_store",
]
