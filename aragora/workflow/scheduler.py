"""
Workflow Scheduler with Cron Integration.

Provides scheduled execution of workflows using cron expressions with:
- SQLite persistence for schedule entries
- Background tick loop that checks for due schedules
- Standard 5-field cron expression support
- Graceful start/stop lifecycle

Usage:
    scheduler = WorkflowScheduler()
    schedule_id = scheduler.add_schedule("my-workflow", "*/5 * * * *")
    await scheduler.start()
    # ...
    await scheduler.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from aragora.config import resolve_db_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cron expression matching
# ---------------------------------------------------------------------------

_FIELD_RANGES: list[tuple[int, int]] = [
    (0, 59),  # minute
    (0, 23),  # hour
    (1, 31),  # day of month
    (1, 12),  # month
    (0, 6),  # day of week (0=Sunday)
]


def _parse_cron_field(field_str: str, min_val: int, max_val: int) -> set[int]:
    """Parse a single cron field into a set of matching integer values."""
    values: set[int] = set()
    for part in field_str.split(","):
        part = part.strip()
        if part == "*":
            values.update(range(min_val, max_val + 1))
        elif part.startswith("*/"):
            step = int(part[2:])
            if step <= 0:
                raise ValueError(f"Invalid step: {part}")
            values.update(range(min_val, max_val + 1, step))
        elif "-" in part:
            lo_str, hi_str = part.split("-", 1)
            lo, hi = int(lo_str), int(hi_str)
            if lo < min_val or hi > max_val or lo > hi:
                raise ValueError(f"Range out of bounds: {part}")
            values.update(range(lo, hi + 1))
        else:
            val = int(part)
            if val < min_val or val > max_val:
                raise ValueError(f"Value out of range: {val}")
            values.add(val)
    return values


def parse_cron_expression(expr: str) -> list[set[int]]:
    """Parse a 5-field cron expression into a list of value sets.

    Fields: minute hour day_of_month month day_of_week

    Returns:
        List of 5 sets, one per field, containing the matching values.

    Raises:
        ValueError: If the expression is malformed.
    """
    parts = expr.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Cron expression must have 5 fields (got {len(parts)}): {expr}")
    return [_parse_cron_field(parts[i], lo, hi) for i, (lo, hi) in enumerate(_FIELD_RANGES)]


def cron_matches(parsed: list[set[int]], dt: datetime) -> bool:
    """Check whether *dt* matches the parsed cron schedule."""
    minute = dt.minute
    hour = dt.hour
    day = dt.day
    month = dt.month
    dow = dt.weekday()  # Monday=0 .. Sunday=6
    # Convert Python weekday to cron convention (Sunday=0)
    cron_dow = (dow + 1) % 7

    return (
        minute in parsed[0]
        and hour in parsed[1]
        and day in parsed[2]
        and month in parsed[3]
        and cron_dow in parsed[4]
    )


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ScheduleEntry:
    """A persisted workflow schedule."""

    id: str
    workflow_id: str
    cron_expr: str
    inputs: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    created_at: str = ""
    updated_at: str = ""
    last_run_at: str | None = None
    next_run_at: str | None = None
    run_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "cron_expr": self.cron_expr,
            "inputs": self.inputs,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_run_at": self.last_run_at,
            "next_run_at": self.next_run_at,
            "run_count": self.run_count,
        }


# ---------------------------------------------------------------------------
# Persistence layer
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS workflow_schedules (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    cron_expr TEXT NOT NULL,
    inputs TEXT NOT NULL DEFAULT '{}',
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_run_at TEXT,
    next_run_at TEXT,
    run_count INTEGER NOT NULL DEFAULT 0
)
"""


class _ScheduleStore:
    """Thin SQLite wrapper for schedule persistence."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute(_CREATE_TABLE_SQL)
            self._conn.commit()
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- CRUD ---------------------------------------------------------------

    def insert(self, entry: ScheduleEntry) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO workflow_schedules
               (id, workflow_id, cron_expr, inputs, enabled,
                created_at, updated_at, last_run_at, next_run_at, run_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.id,
                entry.workflow_id,
                entry.cron_expr,
                json.dumps(entry.inputs),
                int(entry.enabled),
                entry.created_at,
                entry.updated_at,
                entry.last_run_at,
                entry.next_run_at,
                entry.run_count,
            ),
        )
        conn.commit()

    def update(self, entry: ScheduleEntry) -> None:
        conn = self._get_conn()
        conn.execute(
            """UPDATE workflow_schedules
               SET workflow_id=?, cron_expr=?, inputs=?, enabled=?,
                   updated_at=?, last_run_at=?, next_run_at=?, run_count=?
               WHERE id=?""",
            (
                entry.workflow_id,
                entry.cron_expr,
                json.dumps(entry.inputs),
                int(entry.enabled),
                entry.updated_at,
                entry.last_run_at,
                entry.next_run_at,
                entry.run_count,
                entry.id,
            ),
        )
        conn.commit()

    def delete(self, schedule_id: str) -> bool:
        conn = self._get_conn()
        cur = conn.execute("DELETE FROM workflow_schedules WHERE id=?", (schedule_id,))
        conn.commit()
        return cur.rowcount > 0

    def get(self, schedule_id: str) -> ScheduleEntry | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM workflow_schedules WHERE id=?", (schedule_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    def list_all(self) -> list[ScheduleEntry]:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM workflow_schedules ORDER BY created_at").fetchall()
        return [self._row_to_entry(r) for r in rows]

    def list_enabled(self) -> list[ScheduleEntry]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM workflow_schedules WHERE enabled=1 ORDER BY created_at"
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> ScheduleEntry:
        return ScheduleEntry(
            id=row["id"],
            workflow_id=row["workflow_id"],
            cron_expr=row["cron_expr"],
            inputs=json.loads(row["inputs"]) if row["inputs"] else {},
            enabled=bool(row["enabled"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_run_at=row["last_run_at"],
            next_run_at=row["next_run_at"],
            run_count=row["run_count"],
        )


# ---------------------------------------------------------------------------
# WorkflowScheduler
# ---------------------------------------------------------------------------


class WorkflowScheduler:
    """Background scheduler that triggers workflow execution on cron schedules.

    Usage:
        scheduler = WorkflowScheduler()
        sid = scheduler.add_schedule("daily-report", "0 9 * * 1-5")
        await scheduler.start()
        ...
        await scheduler.stop()
    """

    def __init__(
        self,
        db_path: str | None = None,
        tick_interval: float = 30.0,
    ) -> None:
        if db_path is None:
            db_path = resolve_db_path("workflow_schedules.db")
        self._store = _ScheduleStore(db_path)
        self._tick_interval = tick_interval
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._running = False

    # -- Schedule CRUD -------------------------------------------------------

    def add_schedule(
        self,
        workflow_id: str,
        cron_expr: str,
        inputs: dict[str, Any] | None = None,
        enabled: bool = True,
    ) -> str:
        """Create a new schedule. Returns the schedule id."""
        # Validate cron expression eagerly
        parse_cron_expression(cron_expr)

        now = datetime.now(timezone.utc).isoformat()
        entry = ScheduleEntry(
            id=f"sched_{uuid.uuid4().hex[:12]}",
            workflow_id=workflow_id,
            cron_expr=cron_expr,
            inputs=inputs or {},
            enabled=enabled,
            created_at=now,
            updated_at=now,
        )
        self._store.insert(entry)
        logger.info(
            "Added workflow schedule %s for workflow %s (%s)",
            entry.id,
            workflow_id,
            cron_expr,
        )
        return entry.id

    def remove_schedule(self, schedule_id: str) -> bool:
        """Remove a schedule. Returns True if it existed."""
        removed = self._store.delete(schedule_id)
        if removed:
            logger.info("Removed workflow schedule %s", schedule_id)
        return removed

    def get_schedule(self, schedule_id: str) -> ScheduleEntry | None:
        return self._store.get(schedule_id)

    def list_schedules(self) -> list[ScheduleEntry]:
        return self._store.list_all()

    def update_schedule(
        self,
        schedule_id: str,
        *,
        cron_expr: str | None = None,
        enabled: bool | None = None,
        inputs: dict[str, Any] | None = None,
    ) -> ScheduleEntry | None:
        """Update fields on an existing schedule. Returns the updated entry."""
        entry = self._store.get(schedule_id)
        if entry is None:
            return None

        if cron_expr is not None:
            parse_cron_expression(cron_expr)
            entry.cron_expr = cron_expr
        if enabled is not None:
            entry.enabled = enabled
        if inputs is not None:
            entry.inputs = inputs

        entry.updated_at = datetime.now(timezone.utc).isoformat()
        self._store.update(entry)
        logger.info("Updated workflow schedule %s", schedule_id)
        return entry

    # -- Lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Start the background tick loop."""
        if self._running:
            logger.warning("WorkflowScheduler already running")
            return
        self._running = True
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._loop(), name="workflow_scheduler")
        logger.info("WorkflowScheduler started (tick_interval=%.1fs)", self._tick_interval)

    async def stop(self) -> None:
        """Stop the background tick loop gracefully."""
        if not self._running:
            return
        self._running = False
        if self._stop_event is not None:
            self._stop_event.set()
        if self._task is not None and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=10.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
        self._task = None
        self._store.close()
        logger.info("WorkflowScheduler stopped")

    # -- Internal ------------------------------------------------------------

    async def _loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                await self._tick()
            except (OSError, sqlite3.Error) as exc:
                logger.error("WorkflowScheduler tick error: %s", exc)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._tick_interval)
                break  # stop event set
            except asyncio.TimeoutError:
                pass  # continue next tick

    async def _tick(self) -> None:
        """Check all enabled schedules and enqueue any that are due."""
        now = datetime.now(timezone.utc)
        entries = self._store.list_enabled()
        for entry in entries:
            try:
                parsed = parse_cron_expression(entry.cron_expr)
            except ValueError:
                logger.warning(
                    "Skipping schedule %s with invalid cron: %s",
                    entry.id,
                    entry.cron_expr,
                )
                continue

            if not cron_matches(parsed, now):
                continue

            # Prevent double-fire within the same minute
            if entry.last_run_at:
                try:
                    last = datetime.fromisoformat(entry.last_run_at)
                    if (
                        last.year == now.year
                        and last.month == now.month
                        and last.day == now.day
                        and last.hour == now.hour
                        and last.minute == now.minute
                    ):
                        continue
                except (ValueError, TypeError):
                    pass

            logger.info(
                "Schedule %s due — triggering workflow %s",
                entry.id,
                entry.workflow_id,
            )
            await self._enqueue_execution(entry)

            entry.last_run_at = now.isoformat()
            entry.run_count += 1
            entry.updated_at = now.isoformat()
            self._store.update(entry)

    async def _enqueue_execution(self, entry: ScheduleEntry) -> None:
        """Trigger a workflow execution for the given schedule entry."""
        try:
            from aragora.workflow.engine import get_workflow_engine
            from aragora.workflow.persistent_store import get_workflow_store

            engine = get_workflow_engine()
            store = get_workflow_store()

            # Look up the workflow definition from the persistent store
            execution = store.get_execution(entry.workflow_id)
            if execution and "definition" in execution:
                from aragora.workflow.types import WorkflowDefinition

                definition = WorkflowDefinition.from_dict(execution["definition"])
                exec_id = f"sched_{entry.id}_{int(time.time())}"
                asyncio.create_task(
                    engine.execute(definition, entry.inputs, exec_id),
                    name=f"sched_exec_{exec_id}",
                )
            else:
                logger.warning("No workflow definition found for %s", entry.workflow_id)
        except ImportError:
            logger.warning("Workflow engine not available for scheduled execution")
        except (OSError, RuntimeError, ValueError) as exc:
            logger.error("Failed to enqueue workflow %s: %s", entry.workflow_id, exc)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_scheduler: WorkflowScheduler | None = None


def get_workflow_scheduler(
    db_path: str | None = None,
    tick_interval: float = 30.0,
) -> WorkflowScheduler:
    """Return the module-level WorkflowScheduler singleton."""
    global _scheduler
    if _scheduler is None:
        _scheduler = WorkflowScheduler(db_path=db_path, tick_interval=tick_interval)
    return _scheduler


def reset_workflow_scheduler() -> None:
    """Reset the singleton (for tests)."""
    global _scheduler
    _scheduler = None
