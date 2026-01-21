"""
Approval Request Storage Backends.

Persistent storage for human approval requests in workflows.

Backends:
- InMemoryApprovalRequestStore: For testing
- SQLiteApprovalRequestStore: For single-instance deployments
- RedisApprovalRequestStore: For multi-instance (with SQLite fallback)

Usage:
    from aragora.storage.approval_request_store import (
        get_approval_request_store,
        set_approval_request_store,
    )

    # Use default store (configured via environment)
    store = get_approval_request_store()
    await store.save(request_data_dict)
    data = await store.get("request-123")
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
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from asyncpg import Pool

logger = logging.getLogger(__name__)

# Global singleton
_approval_request_store: Optional["ApprovalRequestStoreBackend"] = None
_store_lock = threading.RLock()


@dataclass
class ApprovalRequestItem:
    """
    Approval request data for persistence.

    This is a storage-friendly representation of a human approval request.
    """

    request_id: str
    workflow_id: str
    step_id: str
    title: str
    status: str = "pending"  # pending, approved, rejected, expired

    # Request details
    description: Optional[str] = None
    request_data: dict[str, Any] = field(default_factory=dict)
    response_data: Optional[dict[str, Any]] = None

    # Requester/responder
    requester_id: Optional[str] = None
    responder_id: Optional[str] = None

    # Timing
    expires_at: Optional[str] = None
    responded_at: Optional[str] = None

    # Metadata
    workspace_id: Optional[str] = None
    priority: int = 3  # 1=highest, 5=lowest
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
            "request_id": self.request_id,
            "workflow_id": self.workflow_id,
            "step_id": self.step_id,
            "title": self.title,
            "status": self.status,
            "description": self.description,
            "request_data": self.request_data,
            "response_data": self.response_data,
            "requester_id": self.requester_id,
            "responder_id": self.responder_id,
            "expires_at": self.expires_at,
            "responded_at": self.responded_at,
            "workspace_id": self.workspace_id,
            "priority": self.priority,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ApprovalRequestItem":
        """Create from dictionary."""
        return cls(
            request_id=data.get("request_id", ""),
            workflow_id=data.get("workflow_id", ""),
            step_id=data.get("step_id", ""),
            title=data.get("title", ""),
            status=data.get("status", "pending"),
            description=data.get("description"),
            request_data=data.get("request_data", {}),
            response_data=data.get("response_data"),
            requester_id=data.get("requester_id"),
            responder_id=data.get("responder_id"),
            expires_at=data.get("expires_at"),
            responded_at=data.get("responded_at"),
            workspace_id=data.get("workspace_id"),
            priority=data.get("priority", 3),
            tags=data.get("tags", []),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "ApprovalRequestItem":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


class ApprovalRequestStoreBackend(ABC):
    """Abstract base class for approval request storage backends."""

    @abstractmethod
    async def get(self, request_id: str) -> Optional[dict[str, Any]]:
        """Get request data by ID."""
        pass

    @abstractmethod
    async def save(self, data: dict[str, Any]) -> None:
        """Save request data."""
        pass

    @abstractmethod
    async def delete(self, request_id: str) -> bool:
        """Delete request data."""
        pass

    @abstractmethod
    async def list_all(self) -> list[dict[str, Any]]:
        """List all requests."""
        pass

    @abstractmethod
    async def list_by_status(self, status: str) -> list[dict[str, Any]]:
        """List requests by status."""
        pass

    @abstractmethod
    async def list_by_workflow(self, workflow_id: str) -> list[dict[str, Any]]:
        """List requests by workflow."""
        pass

    @abstractmethod
    async def list_pending(self) -> list[dict[str, Any]]:
        """List pending requests."""
        pass

    @abstractmethod
    async def list_expired(self) -> list[dict[str, Any]]:
        """List expired requests (past expires_at but still pending)."""
        pass

    @abstractmethod
    async def respond(
        self,
        request_id: str,
        status: str,
        responder_id: str,
        response_data: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Record a response to an approval request."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close any resources."""
        pass


class InMemoryApprovalRequestStore(ApprovalRequestStoreBackend):
    """
    In-memory approval request store for testing.

    Data is lost on restart.
    """

    def __init__(self) -> None:
        """Initialize in-memory store."""
        self._data: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    async def get(self, request_id: str) -> Optional[dict[str, Any]]:
        """Get request data by ID."""
        with self._lock:
            return self._data.get(request_id)

    async def save(self, data: dict[str, Any]) -> None:
        """Save request data."""
        request_id = data.get("request_id")
        if not request_id:
            raise ValueError("request_id is required")
        with self._lock:
            self._data[request_id] = data

    async def delete(self, request_id: str) -> bool:
        """Delete request data."""
        with self._lock:
            if request_id in self._data:
                del self._data[request_id]
                return True
            return False

    async def list_all(self) -> list[dict[str, Any]]:
        """List all requests."""
        with self._lock:
            return list(self._data.values())

    async def list_by_status(self, status: str) -> list[dict[str, Any]]:
        """List requests by status."""
        with self._lock:
            return [r for r in self._data.values() if r.get("status") == status]

    async def list_by_workflow(self, workflow_id: str) -> list[dict[str, Any]]:
        """List requests by workflow."""
        with self._lock:
            return [r for r in self._data.values() if r.get("workflow_id") == workflow_id]

    async def list_pending(self) -> list[dict[str, Any]]:
        """List pending requests."""
        with self._lock:
            return [r for r in self._data.values() if r.get("status") == "pending"]

    async def list_expired(self) -> list[dict[str, Any]]:
        """List expired requests."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            return [
                r
                for r in self._data.values()
                if (
                    r.get("status") == "pending"
                    and r.get("expires_at")
                    and r.get("expires_at") < now
                )
            ]

    async def respond(
        self,
        request_id: str,
        status: str,
        responder_id: str,
        response_data: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Record a response to an approval request."""
        with self._lock:
            if request_id not in self._data:
                return False
            self._data[request_id]["status"] = status
            self._data[request_id]["responder_id"] = responder_id
            self._data[request_id]["responded_at"] = datetime.now(timezone.utc).isoformat()
            self._data[request_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
            if response_data is not None:
                self._data[request_id]["response_data"] = response_data
            return True

    async def close(self) -> None:
        """No-op for in-memory store."""
        pass


class SQLiteApprovalRequestStore(ApprovalRequestStoreBackend):
    """
    SQLite-backed approval request store.

    Suitable for single-instance deployments.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """
        Initialize SQLite store.

        Args:
            db_path: Path to SQLite database. Defaults to
                     $ARAGORA_DATA_DIR/approval_requests.db
        """
        if db_path is None:
            data_dir = os.getenv("ARAGORA_DATA_DIR", "/tmp/aragora")
            db_path = Path(data_dir) / "approval_requests.db"

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
                    CREATE TABLE IF NOT EXISTS approval_requests (
                        request_id TEXT PRIMARY KEY,
                        workflow_id TEXT NOT NULL,
                        step_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        requester_id TEXT,
                        responder_id TEXT,
                        priority INTEGER DEFAULT 3,
                        workspace_id TEXT,
                        expires_at TEXT,
                        responded_at TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        data_json TEXT NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_approval_status
                    ON approval_requests(status)
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_approval_workflow
                    ON approval_requests(workflow_id)
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_approval_expires
                    ON approval_requests(expires_at)
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_approval_workspace
                    ON approval_requests(workspace_id)
                    """
                )
                conn.commit()
            finally:
                conn.close()

    async def get(self, request_id: str) -> Optional[dict[str, Any]]:
        """Get request data by ID."""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT data_json FROM approval_requests WHERE request_id = ?",
                    (request_id,),
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
                return None
            finally:
                conn.close()

    async def save(self, data: dict[str, Any]) -> None:
        """Save request data."""
        request_id = data.get("request_id")
        if not request_id:
            raise ValueError("request_id is required")

        now = time.time()
        data_json = json.dumps(data)

        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO approval_requests
                    (request_id, workflow_id, step_id, title, status,
                     requester_id, responder_id, priority, workspace_id,
                     expires_at, responded_at, created_at, updated_at, data_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request_id,
                        data.get("workflow_id", ""),
                        data.get("step_id", ""),
                        data.get("title", ""),
                        data.get("status", "pending"),
                        data.get("requester_id"),
                        data.get("responder_id"),
                        data.get("priority", 3),
                        data.get("workspace_id"),
                        data.get("expires_at"),
                        data.get("responded_at"),
                        now,
                        now,
                        data_json,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    async def delete(self, request_id: str) -> bool:
        """Delete request data."""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM approval_requests WHERE request_id = ?",
                    (request_id,),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    async def list_all(self) -> list[dict[str, Any]]:
        """List all requests."""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT data_json FROM approval_requests ORDER BY created_at DESC")
                return [json.loads(row[0]) for row in cursor.fetchall()]
            finally:
                conn.close()

    async def list_by_status(self, status: str) -> list[dict[str, Any]]:
        """List requests by status."""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT data_json FROM approval_requests
                    WHERE status = ?
                    ORDER BY priority ASC, created_at DESC
                    """,
                    (status,),
                )
                return [json.loads(row[0]) for row in cursor.fetchall()]
            finally:
                conn.close()

    async def list_by_workflow(self, workflow_id: str) -> list[dict[str, Any]]:
        """List requests by workflow."""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT data_json FROM approval_requests
                    WHERE workflow_id = ?
                    ORDER BY created_at DESC
                    """,
                    (workflow_id,),
                )
                return [json.loads(row[0]) for row in cursor.fetchall()]
            finally:
                conn.close()

    async def list_pending(self) -> list[dict[str, Any]]:
        """List pending requests."""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT data_json FROM approval_requests
                    WHERE status = 'pending'
                    ORDER BY priority ASC, created_at DESC
                    """
                )
                return [json.loads(row[0]) for row in cursor.fetchall()]
            finally:
                conn.close()

    async def list_expired(self) -> list[dict[str, Any]]:
        """List expired requests."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT data_json FROM approval_requests
                    WHERE status = 'pending'
                      AND expires_at IS NOT NULL
                      AND expires_at < ?
                    ORDER BY expires_at ASC
                    """,
                    (now,),
                )
                return [json.loads(row[0]) for row in cursor.fetchall()]
            finally:
                conn.close()

    async def respond(
        self,
        request_id: str,
        status: str,
        responder_id: str,
        response_data: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Record a response to an approval request."""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                # Get current data
                cursor.execute(
                    "SELECT data_json FROM approval_requests WHERE request_id = ?",
                    (request_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return False

                data = json.loads(row[0])
                data["status"] = status
                data["responder_id"] = responder_id
                data["responded_at"] = datetime.now(timezone.utc).isoformat()
                data["updated_at"] = datetime.now(timezone.utc).isoformat()
                if response_data is not None:
                    data["response_data"] = response_data

                cursor.execute(
                    """
                    UPDATE approval_requests
                    SET status = ?, responder_id = ?, responded_at = ?,
                        updated_at = ?, data_json = ?
                    WHERE request_id = ?
                    """,
                    (
                        status,
                        responder_id,
                        data["responded_at"],
                        time.time(),
                        json.dumps(data),
                        request_id,
                    ),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    async def close(self) -> None:
        """No-op for SQLite (connections are per-operation)."""
        pass


class RedisApprovalRequestStore(ApprovalRequestStoreBackend):
    """
    Redis-backed approval request store with SQLite fallback.

    For multi-instance deployments with optional horizontal scaling.
    Falls back to SQLite if Redis is unavailable.
    """

    REDIS_PREFIX = "aragora:approval_request:"
    REDIS_INDEX_STATUS = "aragora:approval_request:idx:status:"
    REDIS_INDEX_WORKFLOW = "aragora:approval_request:idx:workflow:"

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
        self._fallback = SQLiteApprovalRequestStore(fallback_db_path)
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
            logger.info("Connected to Redis for approval request storage")
            self._using_fallback = False
        except Exception as e:
            logger.warning(f"Redis connection failed, using SQLite fallback: {e}")
            self._using_fallback = True
            self._redis_client = None

    async def get(self, request_id: str) -> Optional[dict[str, Any]]:
        """Get request data by ID."""
        if self._using_fallback:
            return await self._fallback.get(request_id)

        try:
            data = self._redis_client.get(f"{self.REDIS_PREFIX}{request_id}")
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.warning(f"Redis get failed, using fallback: {e}")
            return await self._fallback.get(request_id)

    async def save(self, data: dict[str, Any]) -> None:
        """Save request data."""
        request_id = data.get("request_id")
        if not request_id:
            raise ValueError("request_id is required")

        # Always save to SQLite fallback for durability
        await self._fallback.save(data)

        if self._using_fallback:
            return

        try:
            data_json = json.dumps(data)
            pipe = self._redis_client.pipeline()

            # Save main data
            pipe.set(f"{self.REDIS_PREFIX}{request_id}", data_json)

            # Update status index
            status = data.get("status", "pending")
            pipe.sadd(f"{self.REDIS_INDEX_STATUS}{status}", request_id)

            # Update workflow index
            workflow_id = data.get("workflow_id")
            if workflow_id:
                pipe.sadd(f"{self.REDIS_INDEX_WORKFLOW}{workflow_id}", request_id)

            pipe.execute()
        except Exception as e:
            logger.warning(f"Redis save failed (SQLite fallback used): {e}")

    async def delete(self, request_id: str) -> bool:
        """Delete request data."""
        result = await self._fallback.delete(request_id)

        if self._using_fallback:
            return result

        try:
            # Get current data to clean up indexes
            data = self._redis_client.get(f"{self.REDIS_PREFIX}{request_id}")
            if data:
                request_data = json.loads(data)
                pipe = self._redis_client.pipeline()

                # Remove from status index
                if request_data.get("status"):
                    pipe.srem(
                        f"{self.REDIS_INDEX_STATUS}{request_data['status']}",
                        request_id,
                    )

                # Remove from workflow index
                if request_data.get("workflow_id"):
                    pipe.srem(
                        f"{self.REDIS_INDEX_WORKFLOW}{request_data['workflow_id']}",
                        request_id,
                    )

                # Delete main data
                pipe.delete(f"{self.REDIS_PREFIX}{request_id}")
                pipe.execute()
                return True
            return result
        except Exception as e:
            logger.warning(f"Redis delete failed: {e}")
            return result

    async def list_all(self) -> list[dict[str, Any]]:
        """List all requests."""
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
                    data_keys = [k for k in keys if b":idx:" not in k and b"idx:" not in k]
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
        """List requests by status."""
        if self._using_fallback:
            return await self._fallback.list_by_status(status)

        try:
            request_ids = self._redis_client.smembers(f"{self.REDIS_INDEX_STATUS}{status}")
            if not request_ids:
                return []

            keys = [f"{self.REDIS_PREFIX}{rid.decode()}" for rid in request_ids]
            values = self._redis_client.mget(keys)
            return [json.loads(v) for v in values if v]
        except Exception as e:
            logger.warning(f"Redis list_by_status failed, using fallback: {e}")
            return await self._fallback.list_by_status(status)

    async def list_by_workflow(self, workflow_id: str) -> list[dict[str, Any]]:
        """List requests by workflow."""
        if self._using_fallback:
            return await self._fallback.list_by_workflow(workflow_id)

        try:
            request_ids = self._redis_client.smembers(f"{self.REDIS_INDEX_WORKFLOW}{workflow_id}")
            if not request_ids:
                return []

            keys = [f"{self.REDIS_PREFIX}{rid.decode()}" for rid in request_ids]
            values = self._redis_client.mget(keys)
            return [json.loads(v) for v in values if v]
        except Exception as e:
            logger.warning(f"Redis list_by_workflow failed, using fallback: {e}")
            return await self._fallback.list_by_workflow(workflow_id)

    async def list_pending(self) -> list[dict[str, Any]]:
        """List pending requests."""
        return await self.list_by_status("pending")

    async def list_expired(self) -> list[dict[str, Any]]:
        """List expired requests."""
        # Use SQLite fallback for date comparison queries
        return await self._fallback.list_expired()

    async def respond(
        self,
        request_id: str,
        status: str,
        responder_id: str,
        response_data: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Record a response to an approval request."""
        # Update SQLite fallback
        result = await self._fallback.respond(request_id, status, responder_id, response_data)

        if self._using_fallback:
            return result

        try:
            # Get current data
            data_bytes = self._redis_client.get(f"{self.REDIS_PREFIX}{request_id}")
            if not data_bytes:
                return result

            data = json.loads(data_bytes)
            old_status = data.get("status")

            data["status"] = status
            data["responder_id"] = responder_id
            data["responded_at"] = datetime.now(timezone.utc).isoformat()
            data["updated_at"] = datetime.now(timezone.utc).isoformat()
            if response_data is not None:
                data["response_data"] = response_data

            pipe = self._redis_client.pipeline()

            # Update main data
            pipe.set(f"{self.REDIS_PREFIX}{request_id}", json.dumps(data))

            # Update status indexes
            if old_status and old_status != status:
                pipe.srem(f"{self.REDIS_INDEX_STATUS}{old_status}", request_id)
            pipe.sadd(f"{self.REDIS_INDEX_STATUS}{status}", request_id)

            pipe.execute()
            return True
        except Exception as e:
            logger.warning(f"Redis respond failed: {e}")
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


class PostgresApprovalRequestStore(ApprovalRequestStoreBackend):
    """
    PostgreSQL-backed approval request store.

    Async implementation for production multi-instance deployments
    with horizontal scaling and concurrent writes.
    """

    SCHEMA_NAME = "approval_requests"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS approval_requests (
            request_id TEXT PRIMARY KEY,
            workflow_id TEXT NOT NULL,
            step_id TEXT NOT NULL,
            title TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            requester_id TEXT,
            responder_id TEXT,
            priority INTEGER DEFAULT 3,
            workspace_id TEXT,
            expires_at TEXT,
            responded_at TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            data_json JSONB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_approval_status ON approval_requests(status);
        CREATE INDEX IF NOT EXISTS idx_approval_workflow ON approval_requests(workflow_id);
        CREATE INDEX IF NOT EXISTS idx_approval_expires ON approval_requests(expires_at);
        CREATE INDEX IF NOT EXISTS idx_approval_workspace ON approval_requests(workspace_id);
    """

    def __init__(self, pool: "Pool"):
        self._pool = pool
        self._initialized = False
        logger.info("PostgresApprovalRequestStore initialized")

    async def initialize(self) -> None:
        """Initialize database schema."""
        if self._initialized:
            return

        async with self._pool.acquire() as conn:
            await conn.execute(self.INITIAL_SCHEMA)

        self._initialized = True
        logger.debug(f"[{self.SCHEMA_NAME}] Schema initialized")

    async def get(self, request_id: str) -> Optional[dict[str, Any]]:
        """Get request data by ID."""
        return await self.get_async(request_id)

    async def get_async(self, request_id: str) -> Optional[dict[str, Any]]:
        """Get request data by ID asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data_json FROM approval_requests WHERE request_id = $1",
                request_id,
            )
            if row:
                data = row["data_json"]
                return json.loads(data) if isinstance(data, str) else data
            return None

    async def save(self, data: dict[str, Any]) -> None:
        """Save request data."""
        await self.save_async(data)

    async def save_async(self, data: dict[str, Any]) -> None:
        """Save request data asynchronously."""
        request_id = data.get("request_id")
        if not request_id:
            raise ValueError("request_id is required")

        now = time.time()
        data_json = json.dumps(data)

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO approval_requests
                (request_id, workflow_id, step_id, title, status,
                 requester_id, responder_id, priority, workspace_id,
                 expires_at, responded_at, created_at, updated_at, data_json)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                        to_timestamp($12), to_timestamp($13), $14)
                ON CONFLICT (request_id) DO UPDATE SET
                    workflow_id = EXCLUDED.workflow_id,
                    step_id = EXCLUDED.step_id,
                    title = EXCLUDED.title,
                    status = EXCLUDED.status,
                    requester_id = EXCLUDED.requester_id,
                    responder_id = EXCLUDED.responder_id,
                    priority = EXCLUDED.priority,
                    workspace_id = EXCLUDED.workspace_id,
                    expires_at = EXCLUDED.expires_at,
                    responded_at = EXCLUDED.responded_at,
                    updated_at = to_timestamp($13),
                    data_json = EXCLUDED.data_json
                """,
                request_id,
                data.get("workflow_id", ""),
                data.get("step_id", ""),
                data.get("title", ""),
                data.get("status", "pending"),
                data.get("requester_id"),
                data.get("responder_id"),
                data.get("priority", 3),
                data.get("workspace_id"),
                data.get("expires_at"),
                data.get("responded_at"),
                now,
                now,
                data_json,
            )

    async def delete(self, request_id: str) -> bool:
        """Delete request data."""
        return await self.delete_async(request_id)

    async def delete_async(self, request_id: str) -> bool:
        """Delete request data asynchronously."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM approval_requests WHERE request_id = $1",
                request_id,
            )
            return result != "DELETE 0"

    async def list_all(self) -> list[dict[str, Any]]:
        """List all requests."""
        return await self.list_all_async()

    async def list_all_async(self) -> list[dict[str, Any]]:
        """List all requests asynchronously."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT data_json FROM approval_requests
                   ORDER BY EXTRACT(EPOCH FROM created_at) DESC"""
            )
            results = []
            for row in rows:
                data = row["data_json"]
                results.append(json.loads(data) if isinstance(data, str) else data)
            return results

    async def list_by_status(self, status: str) -> list[dict[str, Any]]:
        """List requests by status."""
        return await self.list_by_status_async(status)

    async def list_by_status_async(self, status: str) -> list[dict[str, Any]]:
        """List requests by status asynchronously."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM approval_requests
                WHERE status = $1
                ORDER BY priority ASC, EXTRACT(EPOCH FROM created_at) DESC
                """,
                status,
            )
            results = []
            for row in rows:
                data = row["data_json"]
                results.append(json.loads(data) if isinstance(data, str) else data)
            return results

    async def list_by_workflow(self, workflow_id: str) -> list[dict[str, Any]]:
        """List requests by workflow."""
        return await self.list_by_workflow_async(workflow_id)

    async def list_by_workflow_async(self, workflow_id: str) -> list[dict[str, Any]]:
        """List requests by workflow asynchronously."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM approval_requests
                WHERE workflow_id = $1
                ORDER BY EXTRACT(EPOCH FROM created_at) DESC
                """,
                workflow_id,
            )
            results = []
            for row in rows:
                data = row["data_json"]
                results.append(json.loads(data) if isinstance(data, str) else data)
            return results

    async def list_pending(self) -> list[dict[str, Any]]:
        """List pending requests."""
        return await self.list_pending_async()

    async def list_pending_async(self) -> list[dict[str, Any]]:
        """List pending requests asynchronously."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM approval_requests
                WHERE status = 'pending'
                ORDER BY priority ASC, EXTRACT(EPOCH FROM created_at) DESC
                """
            )
            results = []
            for row in rows:
                data = row["data_json"]
                results.append(json.loads(data) if isinstance(data, str) else data)
            return results

    async def list_expired(self) -> list[dict[str, Any]]:
        """List expired requests."""
        return await self.list_expired_async()

    async def list_expired_async(self) -> list[dict[str, Any]]:
        """List expired requests asynchronously."""
        now = datetime.now(timezone.utc).isoformat()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM approval_requests
                WHERE status = 'pending'
                  AND expires_at IS NOT NULL
                  AND expires_at < $1
                ORDER BY expires_at ASC
                """,
                now,
            )
            results = []
            for row in rows:
                data = row["data_json"]
                results.append(json.loads(data) if isinstance(data, str) else data)
            return results

    async def respond(
        self,
        request_id: str,
        status: str,
        responder_id: str,
        response_data: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Record a response to an approval request."""
        return await self.respond_async(request_id, status, responder_id, response_data)

    async def respond_async(
        self,
        request_id: str,
        status: str,
        responder_id: str,
        response_data: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Record a response to an approval request asynchronously."""
        async with self._pool.acquire() as conn:
            # Get current data
            row = await conn.fetchrow(
                "SELECT data_json FROM approval_requests WHERE request_id = $1",
                request_id,
            )
            if not row:
                return False

            raw_data = row["data_json"]
            data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
            data["status"] = status
            data["responder_id"] = responder_id
            data["responded_at"] = datetime.now(timezone.utc).isoformat()
            data["updated_at"] = datetime.now(timezone.utc).isoformat()
            if response_data is not None:
                data["response_data"] = response_data

            result = await conn.execute(
                """
                UPDATE approval_requests
                SET status = $1, responder_id = $2, responded_at = $3,
                    updated_at = to_timestamp($4), data_json = $5
                WHERE request_id = $6
                """,
                status,
                responder_id,
                data["responded_at"],
                time.time(),
                json.dumps(data),
                request_id,
            )
            return result != "UPDATE 0"

    async def close(self) -> None:
        """Close is a no-op for pool-based stores (pool managed externally)."""
        pass


def get_approval_request_store() -> ApprovalRequestStoreBackend:
    """
    Get the global approval request store instance.

    Backend is selected based on ARAGORA_APPROVAL_STORE_BACKEND env var:
    - "memory": InMemoryApprovalRequestStore (for testing)
    - "sqlite": SQLiteApprovalRequestStore (default, single-instance)
    - "postgres" or "postgresql": PostgresApprovalRequestStore (multi-instance)
    - "redis": RedisApprovalRequestStore (multi-instance)

    Also checks ARAGORA_DB_BACKEND for global database backend selection.
    """
    global _approval_request_store

    with _store_lock:
        if _approval_request_store is not None:
            return _approval_request_store

        # Check store-specific backend first, then global database backend
        backend = os.getenv("ARAGORA_APPROVAL_STORE_BACKEND")
        if not backend:
            backend = os.getenv("ARAGORA_DB_BACKEND", "sqlite")
        backend = backend.lower()

        if backend == "memory":
            _approval_request_store = InMemoryApprovalRequestStore()
            logger.info("Using in-memory approval request store")
        elif backend == "postgres" or backend == "postgresql":
            logger.info("Using PostgreSQL approval request store")
            try:
                from aragora.storage.postgres_store import get_postgres_pool

                # Initialize PostgreSQL store with connection pool
                pool = asyncio.get_event_loop().run_until_complete(get_postgres_pool())
                store = PostgresApprovalRequestStore(pool)
                asyncio.get_event_loop().run_until_complete(store.initialize())
                _approval_request_store = store
            except Exception as e:
                logger.warning(f"PostgreSQL not available, falling back to SQLite: {e}")
                _approval_request_store = SQLiteApprovalRequestStore()
        elif backend == "redis":
            _approval_request_store = RedisApprovalRequestStore()
            logger.info("Using Redis approval request store")
        else:  # Default to SQLite
            _approval_request_store = SQLiteApprovalRequestStore()
            logger.info("Using SQLite approval request store")

        return _approval_request_store


def set_approval_request_store(store: ApprovalRequestStoreBackend) -> None:
    """Set a custom approval request store instance."""
    global _approval_request_store

    with _store_lock:
        _approval_request_store = store


def reset_approval_request_store() -> None:
    """Reset the global approval request store (for testing)."""
    global _approval_request_store

    with _store_lock:
        _approval_request_store = None


__all__ = [
    "ApprovalRequestItem",
    "ApprovalRequestStoreBackend",
    "InMemoryApprovalRequestStore",
    "SQLiteApprovalRequestStore",
    "RedisApprovalRequestStore",
    "PostgresApprovalRequestStore",
    "get_approval_request_store",
    "set_approval_request_store",
    "reset_approval_request_store",
]
