"""
Persistent Task Queue Storage.

Provides SQLite-backed persistence for the TaskQueue, enabling recovery
of queued and running tasks after server restarts.

Features:
- Persist tasks on enqueue
- Update task status in database
- Recover pending/ready tasks on startup
- Track failed tasks for retry
- Multi-tenant isolation

Usage:
    from aragora.workflow.queue.persistent_queue import PersistentTaskQueue

    queue = PersistentTaskQueue(db_path="./tasks.db")
    await queue.start()

    # Tasks are automatically persisted
    await queue.enqueue(task)

    # On restart, recover pending tasks
    recovered = await queue.recover_tasks()
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from aragora.workflow.queue.queue import TaskQueue, TaskQueueConfig
from aragora.workflow.queue.task import (
    TaskStatus,
    TaskPriority,
    TaskResult,
    WorkflowTask,
)

logger = logging.getLogger(__name__)


def _record_metrics(operation: str, success: bool, latency: float) -> None:
    """Record task queue metrics if available."""
    try:
        from aragora.observability.metrics import record_task_queue_operation

        record_task_queue_operation(operation, success, latency)
    except ImportError:
        pass


def _record_recovery(original_status: str, count: int = 1) -> None:
    """Record recovery metrics if available."""
    try:
        from aragora.observability.metrics import record_task_queue_recovery

        record_task_queue_recovery(original_status, count)
    except ImportError:
        pass


def _record_cleanup(count: int) -> None:
    """Record cleanup metrics if available."""
    try:
        from aragora.observability.metrics import record_task_queue_cleanup

        record_task_queue_cleanup(count)
    except ImportError:
        pass


# Schema for task persistence
TASK_QUEUE_SCHEMA = """
-- Queued tasks table
CREATE TABLE IF NOT EXISTS task_queue (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    step_config JSON DEFAULT '{}',
    depends_on JSON DEFAULT '[]',
    priority INTEGER DEFAULT 50,
    timeout_seconds REAL DEFAULT 300.0,
    max_retries INTEGER DEFAULT 2,
    status TEXT NOT NULL DEFAULT 'pending',
    retry_count INTEGER DEFAULT 0,
    result_json JSON,
    created_at TEXT NOT NULL,
    queued_at TEXT,
    started_at TEXT,
    completed_at TEXT,
    executor_id TEXT,
    tenant_id TEXT DEFAULT 'default',
    metadata JSON DEFAULT '{}'
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_task_queue_workflow ON task_queue(workflow_id);
CREATE INDEX IF NOT EXISTS idx_task_queue_status ON task_queue(status);
CREATE INDEX IF NOT EXISTS idx_task_queue_tenant ON task_queue(tenant_id);
CREATE INDEX IF NOT EXISTS idx_task_queue_priority ON task_queue(priority, created_at);
"""


class PersistentTaskQueue(TaskQueue):
    """
    TaskQueue with SQLite-backed persistence.

    Extends the base TaskQueue to persist tasks to disk,
    enabling recovery after server restarts.
    """

    def __init__(
        self,
        config: Optional[TaskQueueConfig] = None,
        db_path: Optional[Path | str] = None,
    ):
        """
        Initialize persistent task queue.

        Args:
            config: Queue configuration
            db_path: Path to SQLite database
        """
        super().__init__(config)

        # Set up database path
        if db_path:
            self._db_path = Path(db_path)
        else:
            env_dir = os.environ.get("ARAGORA_DATA_DIR") or os.environ.get("ARAGORA_NOMIC_DIR")
            data_dir = Path(env_dir or ".nomic")
            self._db_path = data_dir / "task_queue.db"

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

        logger.info(f"PersistentTaskQueue initialized: {self._db_path}")

        # Auto-recovery configuration
        self._auto_recover = True
        self._recovery_tenant_id: Optional[str] = None

    async def start(self, auto_recover: bool = True, tenant_id: Optional[str] = None) -> None:
        """
        Start the queue processor with automatic task recovery.

        On startup, this will:
        1. Recover any pending/ready/running tasks from the database
        2. Reset running tasks to ready (they were interrupted)
        3. Start the background processor

        Args:
            auto_recover: Whether to automatically recover persisted tasks (default True)
            tenant_id: Optional tenant filter for recovery

        Usage:
            queue = PersistentTaskQueue()
            await queue.start()  # Recovers tasks automatically

            # Or with tenant filter
            await queue.start(tenant_id="tenant_123")

            # Or skip recovery (e.g., for fresh start)
            await queue.start(auto_recover=False)
        """
        # Start the base queue first
        await super().start()

        # Automatically recover persisted tasks
        if auto_recover:
            try:
                recovered_count = await self.recover_tasks(tenant_id)
                if recovered_count > 0:
                    logger.info(
                        f"Auto-recovered {recovered_count} tasks on queue start",
                        extra={"tenant_id": tenant_id, "recovered_count": recovered_count},
                    )
            except Exception as e:
                logger.error(f"Failed to auto-recover tasks on start: {e}")
                # Don't fail start if recovery fails - queue is still functional

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return cast(sqlite3.Connection, self._local.conn)

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.executescript(TASK_QUEUE_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    async def enqueue(self, task: WorkflowTask) -> str:
        """
        Add a task to the queue with persistence.

        Args:
            task: Task to enqueue

        Returns:
            Task ID
        """
        import time as _time

        start = _time.perf_counter()
        success = True
        try:
            # Persist to database first
            self._persist_task(task)

            # Then enqueue in memory
            return await super().enqueue(task)
        except Exception:
            success = False
            raise
        finally:
            latency = _time.perf_counter() - start
            _record_metrics("enqueue", success, latency)

    async def enqueue_many(self, tasks: List[WorkflowTask]) -> List[str]:
        """Enqueue multiple tasks with persistence."""
        # Persist all tasks
        for task in tasks:
            self._persist_task(task)

        # Then enqueue in memory
        return await super().enqueue_many(tasks)

    def _persist_task(self, task: WorkflowTask) -> None:
        """Persist a task to the database."""
        conn = self._get_conn()
        try:
            result_json = json.dumps(task.result.__dict__) if task.result else None

            conn.execute(
                """
                INSERT OR REPLACE INTO task_queue (
                    id, workflow_id, step_id, step_config, depends_on,
                    priority, timeout_seconds, max_retries, status,
                    retry_count, result_json, created_at, queued_at,
                    started_at, completed_at, executor_id, tenant_id, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    task.id,
                    task.workflow_id,
                    task.step_id,
                    json.dumps(task.step_config),
                    json.dumps(task.depends_on),
                    task.priority.value,
                    task.timeout_seconds,
                    task.max_retries,
                    task.status.value,
                    task.retry_count,
                    result_json,
                    task.created_at.isoformat(),
                    task.queued_at.isoformat() if task.queued_at else None,
                    task.started_at.isoformat() if task.started_at else None,
                    task.completed_at.isoformat() if task.completed_at else None,
                    task.executor_id,
                    task.tenant_id,
                    json.dumps(task.metadata),
                ),
            )
            conn.commit()
            logger.debug(f"Persisted task {task.id}")

        except Exception as e:
            logger.error(f"Failed to persist task {task.id}: {e}")

    def _update_task_status(self, task: WorkflowTask) -> None:
        """Update task status in database."""
        conn = self._get_conn()
        try:
            result_json = json.dumps(task.result.__dict__) if task.result else None

            conn.execute(
                """
                UPDATE task_queue SET
                    status = ?,
                    retry_count = ?,
                    result_json = ?,
                    queued_at = ?,
                    started_at = ?,
                    completed_at = ?,
                    executor_id = ?
                WHERE id = ?
            """,
                (
                    task.status.value,
                    task.retry_count,
                    result_json,
                    task.queued_at.isoformat() if task.queued_at else None,
                    task.started_at.isoformat() if task.started_at else None,
                    task.completed_at.isoformat() if task.completed_at else None,
                    task.executor_id,
                    task.id,
                ),
            )
            conn.commit()

        except Exception as e:
            logger.error(f"Failed to update task status {task.id}: {e}")

    async def _execute_task(self, task: WorkflowTask) -> None:
        """Execute task with status persistence."""
        # Update status to running in DB
        task.mark_running(executor_id="queue")
        self._update_task_status(task)

        # Execute via parent
        await super()._execute_task(task)

        # Update final status in DB
        self._update_task_status(task)

    async def recover_tasks(self, tenant_id: Optional[str] = None) -> int:
        """
        Recover pending and ready tasks from database.

        Call this on startup to resume work that was interrupted.

        Args:
            tenant_id: Optional tenant filter

        Returns:
            Number of tasks recovered
        """
        conn = self._get_conn()
        try:
            # Find tasks that need recovery
            # - PENDING: Waiting for dependencies
            # - READY: Was ready to run
            # - RUNNING: Was running when server stopped (reset to READY)
            # - RETRY: Scheduled for retry
            recoverable_statuses = ("pending", "ready", "running", "retry")

            query = """
                SELECT * FROM task_queue
                WHERE status IN (?, ?, ?, ?)
            """
            params: List[Any] = list(recoverable_statuses)

            if tenant_id:
                query += " AND tenant_id = ?"
                params.append(tenant_id)

            query += " ORDER BY priority, created_at"

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            recovered = 0
            recovery_counts: Dict[str, int] = {}
            for row in rows:
                task = self._row_to_task(row)
                original_status = task.status.value

                # Reset running tasks to ready
                if task.status == TaskStatus.RUNNING:
                    task.status = TaskStatus.READY
                    task.started_at = None
                    task.executor_id = None
                    self._update_task_status(task)

                # Re-enqueue in memory (bypass _persist_task since already in DB)
                await super().enqueue(task)
                recovered += 1
                recovery_counts[original_status] = recovery_counts.get(original_status, 0) + 1

            # Record recovery metrics
            for status, count in recovery_counts.items():
                _record_recovery(status, count)

            logger.info(f"Recovered {recovered} tasks from persistent storage")
            return recovered

        except Exception as e:
            logger.error(f"Failed to recover tasks: {e}")
            return 0

    def _row_to_task(self, row: sqlite3.Row) -> WorkflowTask:
        """Convert database row to WorkflowTask."""
        result_data = json.loads(row["result_json"]) if row["result_json"] else None
        result = TaskResult(**result_data) if result_data else None

        return WorkflowTask(
            id=row["id"],
            workflow_id=row["workflow_id"],
            step_id=row["step_id"],
            step_config=json.loads(row["step_config"] or "{}"),
            depends_on=json.loads(row["depends_on"] or "[]"),
            priority=TaskPriority(row["priority"]),
            timeout_seconds=row["timeout_seconds"],
            max_retries=row["max_retries"],
            status=TaskStatus(row["status"]),
            retry_count=row["retry_count"] or 0,
            result=result,
            created_at=(
                datetime.fromisoformat(row["created_at"])
                if row["created_at"]
                else datetime.now(timezone.utc)
            ),
            queued_at=(datetime.fromisoformat(row["queued_at"]) if row["queued_at"] else None),
            started_at=(datetime.fromisoformat(row["started_at"]) if row["started_at"] else None),
            completed_at=(
                datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
            ),
            executor_id=row["executor_id"],
            tenant_id=row["tenant_id"] or "default",
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def get_task_from_db(self, task_id: str) -> Optional[WorkflowTask]:
        """Get a task directly from the database."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "SELECT * FROM task_queue WHERE id = ?",
                (task_id,),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_task(row)
            return None
        except Exception as e:
            logger.error(f"Failed to get task {task_id} from DB: {e}")
            return None

    def list_tasks_from_db(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[str] = None,
        tenant_id: str = "default",
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[List[WorkflowTask], int]:
        """
        List tasks from database with filtering.

        Args:
            workflow_id: Filter by workflow
            status: Filter by status
            tenant_id: Tenant ID
            limit: Max results
            offset: Pagination offset

        Returns:
            Tuple of (tasks, total_count)
        """
        conn = self._get_conn()
        try:
            query = "SELECT * FROM task_queue WHERE tenant_id = ?"
            count_query = "SELECT COUNT(*) FROM task_queue WHERE tenant_id = ?"
            params: List[Any] = [tenant_id]

            if workflow_id:
                query += " AND workflow_id = ?"
                count_query += " AND workflow_id = ?"
                params.append(workflow_id)

            if status:
                query += " AND status = ?"
                count_query += " AND status = ?"
                params.append(status)

            # Get count
            cursor = conn.execute(count_query, params)
            total = cursor.fetchone()[0]

            # Get tasks
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor = conn.execute(query, params)
            tasks = [self._row_to_task(row) for row in cursor.fetchall()]

            return tasks, total

        except Exception as e:
            logger.error(f"Failed to list tasks from DB: {e}")
            return [], 0

    def delete_completed_tasks(
        self,
        older_than_hours: int = 24,
        tenant_id: Optional[str] = None,
    ) -> int:
        """
        Clean up completed tasks older than specified hours.

        Args:
            older_than_hours: Delete tasks older than this
            tenant_id: Optional tenant filter

        Returns:
            Number of tasks deleted
        """
        conn = self._get_conn()
        try:
            datetime.now(timezone.utc)

            query = """
                DELETE FROM task_queue
                WHERE status IN ('completed', 'failed', 'cancelled', 'timeout')
                AND completed_at < datetime('now', ?)
            """
            params: List[Any] = [f"-{older_than_hours} hours"]

            if tenant_id:
                query += " AND tenant_id = ?"
                params.append(tenant_id)

            cursor = conn.execute(query, params)
            deleted = cursor.rowcount
            conn.commit()

            if deleted > 0:
                logger.info(f"Cleaned up {deleted} completed tasks")
                _record_cleanup(deleted)

            return deleted

        except Exception as e:
            logger.error(f"Failed to delete completed tasks: {e}")
            return 0

    async def close(self) -> None:
        """Close database connections."""
        await self.stop(drain=True)
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn


# Global queue instance
_persistent_queue: Optional[PersistentTaskQueue] = None


def get_persistent_task_queue(
    config: Optional[TaskQueueConfig] = None,
    db_path: Optional[Path | str] = None,
) -> PersistentTaskQueue:
    """
    Get or create the global persistent task queue.

    Args:
        config: Queue configuration (only used on first call)
        db_path: Database path (only used on first call)

    Returns:
        PersistentTaskQueue instance
    """
    global _persistent_queue
    if _persistent_queue is None:
        _persistent_queue = PersistentTaskQueue(config, db_path)
    return _persistent_queue


def reset_persistent_task_queue() -> None:
    """Reset the global queue (for testing)."""
    global _persistent_queue
    _persistent_queue = None


__all__ = [
    "PersistentTaskQueue",
    "get_persistent_task_queue",
    "reset_persistent_task_queue",
]
