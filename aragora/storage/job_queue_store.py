"""
Persistent Job Queue Store.

Provides durable storage for job queue entries, ensuring jobs survive
server restarts and enabling distributed job processing.

Backends:
- SQLiteJobStore: Persistent, single-instance (default)
- RedisJobStore: Distributed, multi-instance (with SQLite fallback)

Usage:
    from aragora.storage.job_queue_store import get_job_store

    store = get_job_store()
    await store.enqueue(job)
    job = await store.dequeue(worker_id="worker-1")
    await store.complete(job.id)
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
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Status of a job in the queue."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class QueuedJob:
    """A job in the durable queue."""

    id: str
    job_type: str  # "gauntlet", "workflow", "debate", etc.
    payload: Dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    attempts: int = 0
    max_attempts: int = 3
    worker_id: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    workspace_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "job_type": self.job_type,
            "payload": self.payload,
            "status": self.status.value if isinstance(self.status, JobStatus) else self.status,
            "priority": self.priority,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "worker_id": self.worker_id,
            "error": self.error,
            "result": self.result,
            "user_id": self.user_id,
            "workspace_id": self.workspace_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueuedJob":
        """Create from dictionary."""
        status = data.get("status", "pending")
        if isinstance(status, str):
            status = JobStatus(status)
        return cls(
            id=data["id"],
            job_type=data["job_type"],
            payload=data.get("payload", {}),
            status=status,
            priority=data.get("priority", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            attempts=data.get("attempts", 0),
            max_attempts=data.get("max_attempts", 3),
            worker_id=data.get("worker_id"),
            error=data.get("error"),
            result=data.get("result"),
            user_id=data.get("user_id"),
            workspace_id=data.get("workspace_id"),
        )

    @classmethod
    def from_row(cls, row: tuple) -> "QueuedJob":
        """Create from database row."""
        return cls(
            id=row[0],
            job_type=row[1],
            payload=json.loads(row[2]) if row[2] else {},
            status=JobStatus(row[3]),
            priority=row[4] or 0,
            created_at=row[5] or time.time(),
            updated_at=row[6] or time.time(),
            started_at=row[7],
            completed_at=row[8],
            attempts=row[9] or 0,
            max_attempts=row[10] or 3,
            worker_id=row[11],
            error=row[12],
            result=json.loads(row[13]) if row[13] else None,
            user_id=row[14],
            workspace_id=row[15],
        )


class JobStoreBackend(ABC):
    """Abstract base for job queue storage backends."""

    @abstractmethod
    async def enqueue(self, job: QueuedJob) -> None:
        """Add a job to the queue."""
        pass

    @abstractmethod
    async def dequeue(
        self,
        worker_id: str,
        job_types: Optional[List[str]] = None,
    ) -> Optional[QueuedJob]:
        """
        Get the next available job and mark it as processing.

        Args:
            worker_id: ID of the worker claiming the job
            job_types: Optional list of job types to filter by

        Returns:
            The claimed job, or None if no jobs available
        """
        pass

    @abstractmethod
    async def get(self, job_id: str) -> Optional[QueuedJob]:
        """Get a job by ID."""
        pass

    @abstractmethod
    async def update(self, job: QueuedJob) -> None:
        """Update a job's state."""
        pass

    @abstractmethod
    async def complete(
        self,
        job_id: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark a job as completed."""
        pass

    @abstractmethod
    async def fail(
        self,
        job_id: str,
        error: str,
        should_retry: bool = True,
    ) -> None:
        """Mark a job as failed, optionally scheduling a retry."""
        pass

    @abstractmethod
    async def cancel(self, job_id: str) -> bool:
        """Cancel a pending job. Returns True if cancelled."""
        pass

    @abstractmethod
    async def recover_stale_jobs(
        self,
        stale_threshold_seconds: float = 300.0,
    ) -> int:
        """
        Recover jobs that have been processing too long.

        Returns the number of jobs recovered.
        """
        pass

    @abstractmethod
    async def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        job_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[QueuedJob]:
        """List jobs with optional filtering."""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        pass

    async def close(self) -> None:
        """Close connections."""
        pass


class SQLiteJobStore(JobStoreBackend):
    """
    SQLite-backed job store.

    Provides durable job queue storage that survives restarts.
    """

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()
        logger.info(f"SQLiteJobStore initialized: {self.db_path}")

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
            CREATE TABLE IF NOT EXISTS job_queue (
                id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                payload_json TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                priority INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                started_at REAL,
                completed_at REAL,
                attempts INTEGER DEFAULT 0,
                max_attempts INTEGER DEFAULT 3,
                worker_id TEXT,
                error TEXT,
                result_json TEXT,
                user_id TEXT,
                workspace_id TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_job_status ON job_queue(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_job_type ON job_queue(job_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_job_priority ON job_queue(priority DESC)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_job_pending ON job_queue(status, priority DESC, created_at)"
        )
        conn.commit()
        conn.close()

    async def enqueue(self, job: QueuedJob) -> None:
        """Add a job to the queue."""
        conn = self._get_conn()
        job.updated_at = time.time()
        conn.execute(
            """INSERT OR REPLACE INTO job_queue
               (id, job_type, payload_json, status, priority, created_at, updated_at,
                started_at, completed_at, attempts, max_attempts, worker_id, error,
                result_json, user_id, workspace_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job.id,
                job.job_type,
                json.dumps(job.payload),
                job.status.value,
                job.priority,
                job.created_at,
                job.updated_at,
                job.started_at,
                job.completed_at,
                job.attempts,
                job.max_attempts,
                job.worker_id,
                job.error,
                json.dumps(job.result) if job.result else None,
                job.user_id,
                job.workspace_id,
            ),
        )
        conn.commit()
        logger.debug(f"Enqueued job {job.id} ({job.job_type})")

    async def dequeue(
        self,
        worker_id: str,
        job_types: Optional[List[str]] = None,
    ) -> Optional[QueuedJob]:
        """Get the next available job and mark it as processing."""
        conn = self._get_conn()

        # Build query with optional type filter
        if job_types:
            placeholders = ",".join("?" * len(job_types))
            query = f"""
                SELECT id, job_type, payload_json, status, priority, created_at,
                       updated_at, started_at, completed_at, attempts, max_attempts,
                       worker_id, error, result_json, user_id, workspace_id
                FROM job_queue
                WHERE status = 'pending' AND job_type IN ({placeholders})
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
            """
            cursor = conn.execute(query, job_types)
        else:
            cursor = conn.execute(
                """SELECT id, job_type, payload_json, status, priority, created_at,
                          updated_at, started_at, completed_at, attempts, max_attempts,
                          worker_id, error, result_json, user_id, workspace_id
                   FROM job_queue
                   WHERE status = 'pending'
                   ORDER BY priority DESC, created_at ASC
                   LIMIT 1"""
            )

        row = cursor.fetchone()
        if not row:
            return None

        job = QueuedJob.from_row(row)

        # Atomically claim the job
        now = time.time()
        result = conn.execute(
            """UPDATE job_queue
               SET status = 'processing', worker_id = ?, started_at = ?,
                   updated_at = ?, attempts = attempts + 1
               WHERE id = ? AND status = 'pending'""",
            (worker_id, now, now, job.id),
        )
        conn.commit()

        if result.rowcount == 0:
            # Another worker claimed it
            return await self.dequeue(worker_id, job_types)

        job.status = JobStatus.PROCESSING
        job.worker_id = worker_id
        job.started_at = now
        job.updated_at = now
        job.attempts += 1

        logger.debug(f"Worker {worker_id} claimed job {job.id}")
        return job

    async def get(self, job_id: str) -> Optional[QueuedJob]:
        """Get a job by ID."""
        conn = self._get_conn()
        cursor = conn.execute(
            """SELECT id, job_type, payload_json, status, priority, created_at,
                      updated_at, started_at, completed_at, attempts, max_attempts,
                      worker_id, error, result_json, user_id, workspace_id
               FROM job_queue WHERE id = ?""",
            (job_id,),
        )
        row = cursor.fetchone()
        return QueuedJob.from_row(row) if row else None

    async def update(self, job: QueuedJob) -> None:
        """Update a job's state."""
        conn = self._get_conn()
        job.updated_at = time.time()
        conn.execute(
            """UPDATE job_queue
               SET status = ?, priority = ?, updated_at = ?, started_at = ?,
                   completed_at = ?, attempts = ?, max_attempts = ?, worker_id = ?,
                   error = ?, result_json = ?
               WHERE id = ?""",
            (
                job.status.value,
                job.priority,
                job.updated_at,
                job.started_at,
                job.completed_at,
                job.attempts,
                job.max_attempts,
                job.worker_id,
                job.error,
                json.dumps(job.result) if job.result else None,
                job.id,
            ),
        )
        conn.commit()

    async def complete(
        self,
        job_id: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark a job as completed."""
        conn = self._get_conn()
        now = time.time()
        conn.execute(
            """UPDATE job_queue
               SET status = 'completed', completed_at = ?, updated_at = ?,
                   result_json = ?
               WHERE id = ?""",
            (now, now, json.dumps(result) if result else None, job_id),
        )
        conn.commit()
        logger.debug(f"Job {job_id} completed")

    async def fail(
        self,
        job_id: str,
        error: str,
        should_retry: bool = True,
    ) -> None:
        """Mark a job as failed, optionally scheduling a retry."""
        job = await self.get(job_id)
        if not job:
            return

        now = time.time()
        if should_retry and job.attempts < job.max_attempts:
            # Schedule retry
            new_status = JobStatus.RETRYING
            conn = self._get_conn()
            conn.execute(
                """UPDATE job_queue
                   SET status = 'pending', error = ?, updated_at = ?, worker_id = NULL
                   WHERE id = ?""",
                (error, now, job_id),
            )
            conn.commit()
            logger.info(f"Job {job_id} scheduled for retry ({job.attempts}/{job.max_attempts})")
        else:
            # Final failure
            conn = self._get_conn()
            conn.execute(
                """UPDATE job_queue
                   SET status = 'failed', error = ?, completed_at = ?, updated_at = ?
                   WHERE id = ?""",
                (error, now, now, job_id),
            )
            conn.commit()
            logger.warning(f"Job {job_id} failed permanently: {error}")

    async def cancel(self, job_id: str) -> bool:
        """Cancel a pending job."""
        conn = self._get_conn()
        now = time.time()
        result = conn.execute(
            """UPDATE job_queue
               SET status = 'cancelled', completed_at = ?, updated_at = ?
               WHERE id = ? AND status = 'pending'""",
            (now, now, job_id),
        )
        conn.commit()
        cancelled = result.rowcount > 0
        if cancelled:
            logger.info(f"Job {job_id} cancelled")
        return cancelled

    async def recover_stale_jobs(
        self,
        stale_threshold_seconds: float = 300.0,
    ) -> int:
        """Recover jobs that have been processing too long."""
        conn = self._get_conn()
        cutoff = time.time() - stale_threshold_seconds
        result = conn.execute(
            """UPDATE job_queue
               SET status = 'pending', worker_id = NULL, updated_at = ?
               WHERE status = 'processing' AND started_at < ?""",
            (time.time(), cutoff),
        )
        conn.commit()
        recovered = result.rowcount
        if recovered > 0:
            logger.info(f"Recovered {recovered} stale jobs")
        return recovered

    async def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        job_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[QueuedJob]:
        """List jobs with optional filtering."""
        conn = self._get_conn()
        conditions = []
        params: List[Any] = []

        if status:
            conditions.append("status = ?")
            params.append(status.value)
        if job_type:
            conditions.append("job_type = ?")
            params.append(job_type)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        cursor = conn.execute(
            f"""SELECT id, job_type, payload_json, status, priority, created_at,
                       updated_at, started_at, completed_at, attempts, max_attempts,
                       worker_id, error, result_json, user_id, workspace_id
                FROM job_queue {where_clause}
                ORDER BY created_at DESC
                LIMIT ?""",
            params,
        )
        return [QueuedJob.from_row(row) for row in cursor.fetchall()]

    async def get_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        conn = self._get_conn()
        cursor = conn.execute(
            """SELECT status, COUNT(*) FROM job_queue GROUP BY status"""
        )
        stats = {row[0]: row[1] for row in cursor.fetchall()}
        stats["total"] = sum(stats.values())
        return stats

    async def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn


# Global store singleton
_job_store: Optional[JobStoreBackend] = None


def get_job_store() -> JobStoreBackend:
    """
    Get or create the job queue store.

    Uses environment variables to configure:
    - ARAGORA_JOB_STORE_BACKEND: "sqlite" (default) or "memory"
    - ARAGORA_DATA_DIR: Directory for SQLite database

    Returns:
        Configured JobStoreBackend instance
    """
    global _job_store
    if _job_store is not None:
        return _job_store

    backend_type = os.environ.get("ARAGORA_JOB_STORE_BACKEND", "sqlite").lower()

    # Get data directory
    try:
        from aragora.config.legacy import DATA_DIR

        data_dir = DATA_DIR
    except ImportError:
        env_dir = os.environ.get("ARAGORA_DATA_DIR") or os.environ.get("ARAGORA_NOMIC_DIR")
        data_dir = Path(env_dir or ".nomic")

    db_path = data_dir / "job_queue.db"

    if backend_type == "sqlite":
        logger.info(f"Using SQLite job store: {db_path}")
        _job_store = SQLiteJobStore(db_path)
    else:
        # Default to SQLite
        logger.info(f"Using SQLite job store (default): {db_path}")
        _job_store = SQLiteJobStore(db_path)

    return _job_store


def set_job_store(store: JobStoreBackend) -> None:
    """Set custom job store (for testing)."""
    global _job_store
    _job_store = store


def reset_job_store() -> None:
    """Reset the global job store (for testing)."""
    global _job_store
    _job_store = None


__all__ = [
    "JobStatus",
    "QueuedJob",
    "JobStoreBackend",
    "SQLiteJobStore",
    "get_job_store",
    "set_job_store",
    "reset_job_store",
]
