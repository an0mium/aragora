"""
Internal Event Dead Letter Queue.

Captures events that fail internal handler processing after all retries,
enabling reprocessing and debugging. Unlike webhook_delivery.py which handles
external HTTP delivery failures, this module handles internal cross-subscriber
handler failures.

Usage:
    from aragora.events.dead_letter_queue import (
        EventDLQ,
        FailedEvent,
        get_event_dlq,
    )

    dlq = get_event_dlq()

    # Automatic capture (via CrossSubscriberManager integration)
    # Failed events are automatically sent to DLQ after retry exhaustion

    # Query failed events
    failed = dlq.get_failed_events(limit=100)
    for event in failed:
        print(f"{event.handler_name}: {event.error_message}")

    # Retry a specific event
    success = await dlq.retry_event(event.id, handler_fn)

    # Get DLQ stats
    stats = dlq.get_stats()
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Default retention period (7 days)
DEFAULT_RETENTION_DAYS = int(os.environ.get("ARAGORA_DLQ_RETENTION_DAYS", "7"))

# Default max DLQ size
DEFAULT_MAX_SIZE = int(os.environ.get("ARAGORA_DLQ_MAX_SIZE", "10000"))


class FailedEventStatus(str, Enum):
    """Status of a failed event in the DLQ."""

    PENDING = "pending"  # Awaiting manual review or retry
    RETRYING = "retrying"  # Currently being retried
    RECOVERED = "recovered"  # Successfully reprocessed
    DISCARDED = "discarded"  # Manually discarded


@dataclass
class FailedEvent:
    """A failed event captured in the dead letter queue."""

    id: str
    event_type: str
    event_data: dict[str, Any]
    handler_name: str
    error_message: str
    error_type: str
    retry_count: int
    status: FailedEventStatus
    created_at: float
    updated_at: float
    original_timestamp: float
    trace_id: str | None = None
    correlation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "event_data": self.event_data,
            "handler_name": self.handler_name,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "retry_count": self.retry_count,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "original_timestamp": self.original_timestamp,
            "trace_id": self.trace_id,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FailedEvent":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            event_type=data["event_type"],
            event_data=data.get("event_data", {}),
            handler_name=data["handler_name"],
            error_message=data["error_message"],
            error_type=data.get("error_type", "Exception"),
            retry_count=data.get("retry_count", 0),
            status=FailedEventStatus(data.get("status", "pending")),
            created_at=data["created_at"],
            updated_at=data.get("updated_at", data["created_at"]),
            original_timestamp=data.get("original_timestamp", data["created_at"]),
            trace_id=data.get("trace_id"),
            correlation_id=data.get("correlation_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DLQStats:
    """Statistics for the dead letter queue."""

    total_events: int = 0
    pending_events: int = 0
    recovered_events: int = 0
    discarded_events: int = 0
    events_by_handler: dict[str, int] = field(default_factory=dict)
    events_by_type: dict[str, int] = field(default_factory=dict)
    oldest_event_age_seconds: float | None = None
    newest_event_age_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_events": self.total_events,
            "pending_events": self.pending_events,
            "recovered_events": self.recovered_events,
            "discarded_events": self.discarded_events,
            "events_by_handler": self.events_by_handler,
            "events_by_type": self.events_by_type,
            "oldest_event_age_seconds": self.oldest_event_age_seconds,
            "newest_event_age_seconds": self.newest_event_age_seconds,
        }


class EventDLQPersistence:
    """SQLite-backed persistence for the event DLQ."""

    def __init__(self, db_path: str | None = None):
        """Initialize persistence layer.

        Args:
            db_path: Path to SQLite database. Defaults to
                     $ARAGORA_DATA_DIR/.nomic/event_dlq.db
        """
        if db_path is None:
            data_dir = os.environ.get("ARAGORA_DATA_DIR", ".")
            db_dir = Path(data_dir) / ".nomic"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "event_dlq.db")

        self._db_path = db_path
        self._local = threading.local()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS failed_events (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                event_data TEXT NOT NULL,
                handler_name TEXT NOT NULL,
                error_message TEXT NOT NULL,
                error_type TEXT NOT NULL,
                retry_count INTEGER DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                original_timestamp REAL NOT NULL,
                trace_id TEXT,
                correlation_id TEXT,
                metadata TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_failed_events_status
                ON failed_events(status);
            CREATE INDEX IF NOT EXISTS idx_failed_events_handler
                ON failed_events(handler_name);
            CREATE INDEX IF NOT EXISTS idx_failed_events_created
                ON failed_events(created_at);
            CREATE INDEX IF NOT EXISTS idx_failed_events_type
                ON failed_events(event_type);
            """
        )
        conn.commit()

    def save(self, event: FailedEvent) -> None:
        """Save or update a failed event."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO failed_events (
                id, event_type, event_data, handler_name, error_message,
                error_type, retry_count, status, created_at, updated_at,
                original_timestamp, trace_id, correlation_id, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.event_type,
                json.dumps(event.event_data, default=str),
                event.handler_name,
                event.error_message,
                event.error_type,
                event.retry_count,
                event.status.value,
                event.created_at,
                event.updated_at,
                event.original_timestamp,
                event.trace_id,
                event.correlation_id,
                json.dumps(event.metadata, default=str),
            ),
        )
        conn.commit()

    def get(self, event_id: str) -> FailedEvent | None:
        """Get a failed event by ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM failed_events WHERE id = ?", (event_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_event(row)

    def get_pending(self, limit: int = 100, handler_name: str | None = None) -> list[FailedEvent]:
        """Get pending failed events."""
        conn = self._get_conn()
        if handler_name:
            rows = conn.execute(
                """
                SELECT * FROM failed_events
                WHERE status = 'pending' AND handler_name = ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (handler_name, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM failed_events
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def get_by_handler(self, handler_name: str, limit: int = 100) -> list[FailedEvent]:
        """Get failed events for a specific handler."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT * FROM failed_events
            WHERE handler_name = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (handler_name, limit),
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def update_status(self, event_id: str, status: FailedEventStatus) -> bool:
        """Update the status of a failed event."""
        conn = self._get_conn()
        result = conn.execute(
            """
            UPDATE failed_events
            SET status = ?, updated_at = ?
            WHERE id = ?
            """,
            (status.value, time.time(), event_id),
        )
        conn.commit()
        return result.rowcount > 0

    def delete(self, event_id: str) -> bool:
        """Delete a failed event."""
        conn = self._get_conn()
        result = conn.execute("DELETE FROM failed_events WHERE id = ?", (event_id,))
        conn.commit()
        return result.rowcount > 0

    def cleanup_old(self, retention_days: int = DEFAULT_RETENTION_DAYS) -> int:
        """Remove events older than retention period."""
        conn = self._get_conn()
        cutoff = time.time() - (retention_days * 24 * 60 * 60)
        result = conn.execute(
            """
            DELETE FROM failed_events
            WHERE created_at < ? AND status IN ('recovered', 'discarded')
            """,
            (cutoff,),
        )
        conn.commit()
        return result.rowcount

    def get_stats(self) -> DLQStats:
        """Get DLQ statistics."""
        conn = self._get_conn()
        stats = DLQStats()

        # Total and status counts
        for row in conn.execute(
            "SELECT status, COUNT(*) as cnt FROM failed_events GROUP BY status"
        ).fetchall():
            if row["status"] == "pending":
                stats.pending_events = row["cnt"]
            elif row["status"] == "recovered":
                stats.recovered_events = row["cnt"]
            elif row["status"] == "discarded":
                stats.discarded_events = row["cnt"]
            stats.total_events += row["cnt"]

        # By handler
        for row in conn.execute(
            """
            SELECT handler_name, COUNT(*) as cnt
            FROM failed_events WHERE status = 'pending'
            GROUP BY handler_name
            """
        ).fetchall():
            stats.events_by_handler[row["handler_name"]] = row["cnt"]

        # By type
        for row in conn.execute(
            """
            SELECT event_type, COUNT(*) as cnt
            FROM failed_events WHERE status = 'pending'
            GROUP BY event_type
            """
        ).fetchall():
            stats.events_by_type[row["event_type"]] = row["cnt"]

        # Age stats
        now = time.time()
        row = conn.execute(
            "SELECT MIN(created_at), MAX(created_at) FROM failed_events WHERE status = 'pending'"
        ).fetchone()
        if row[0] is not None:
            stats.oldest_event_age_seconds = now - row[0]
            stats.newest_event_age_seconds = now - row[1]

        return stats

    def count_pending(self) -> int:
        """Count pending events."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM failed_events WHERE status = 'pending'").fetchone()
        return row[0] if row else 0

    def _row_to_event(self, row: sqlite3.Row) -> FailedEvent:
        """Convert database row to FailedEvent."""
        return FailedEvent(
            id=row["id"],
            event_type=row["event_type"],
            event_data=json.loads(row["event_data"]) if row["event_data"] else {},
            handler_name=row["handler_name"],
            error_message=row["error_message"],
            error_type=row["error_type"],
            retry_count=row["retry_count"],
            status=FailedEventStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            original_timestamp=row["original_timestamp"],
            trace_id=row["trace_id"],
            correlation_id=row["correlation_id"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )


class EventDLQ:
    """Dead letter queue for failed internal events.

    Captures events that fail after all retry attempts, enabling:
    - Debugging (what failed, why, when)
    - Manual reprocessing
    - Monitoring DLQ depth as health metric
    - Automatic cleanup of old recovered events
    """

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_SIZE,
        persistence: EventDLQPersistence | None = None,
    ):
        """Initialize the event DLQ.

        Args:
            max_size: Maximum number of pending events before oldest are evicted
            persistence: Optional persistence layer (defaults to SQLite)
        """
        self._max_size = max_size
        self._persistence = persistence or EventDLQPersistence()
        self._lock = threading.RLock()

        # In-memory cache for fast stats
        self._cache_pending_count = 0
        self._cache_updated_at = 0.0

    def capture(
        self,
        event_type: str,
        event_data: dict[str, Any],
        handler_name: str,
        error: Exception,
        retry_count: int = 0,
        original_timestamp: float | None = None,
        trace_id: str | None = None,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FailedEvent:
        """Capture a failed event in the DLQ.

        Args:
            event_type: Type of the event (e.g., StreamEventType value)
            event_data: Original event payload
            handler_name: Name of the handler that failed
            error: The exception that caused failure
            retry_count: Number of retries attempted
            original_timestamp: When the event was originally created
            trace_id: Distributed trace ID
            correlation_id: Correlation ID for related events
            metadata: Additional context

        Returns:
            The captured FailedEvent
        """
        now = time.time()
        failed_event = FailedEvent(
            id=str(uuid.uuid4()),
            event_type=event_type,
            event_data=event_data,
            handler_name=handler_name,
            error_message=str(error),
            error_type=type(error).__name__,
            retry_count=retry_count,
            status=FailedEventStatus.PENDING,
            created_at=now,
            updated_at=now,
            original_timestamp=original_timestamp or now,
            trace_id=trace_id,
            correlation_id=correlation_id,
            metadata=metadata or {},
        )

        with self._lock:
            # Check if we need to evict old events
            pending = self._persistence.count_pending()
            if pending >= self._max_size:
                # Evict oldest pending events
                self._evict_oldest(pending - self._max_size + 1)

            self._persistence.save(failed_event)
            self._cache_pending_count += 1
            self._cache_updated_at = now

        logger.warning(
            "Event captured in DLQ: handler=%s type=%s error=%s",
            handler_name,
            event_type,
            type(error).__name__,
        )

        return failed_event

    def get_event(self, event_id: str) -> FailedEvent | None:
        """Get a specific failed event by ID."""
        return self._persistence.get(event_id)

    def get_pending_events(
        self, limit: int = 100, handler_name: str | None = None
    ) -> list[FailedEvent]:
        """Get pending failed events.

        Args:
            limit: Maximum events to return
            handler_name: Optional filter by handler

        Returns:
            List of pending failed events
        """
        return self._persistence.get_pending(limit, handler_name)

    def get_events_by_handler(self, handler_name: str, limit: int = 100) -> list[FailedEvent]:
        """Get failed events for a specific handler."""
        return self._persistence.get_by_handler(handler_name, limit)

    def retry_event(
        self,
        event_id: str,
        handler: Callable[[dict[str, Any]], None],
    ) -> bool:
        """Retry processing a failed event.

        Args:
            event_id: ID of the event to retry
            handler: The handler function to execute

        Returns:
            True if retry succeeded, False otherwise
        """
        event = self._persistence.get(event_id)
        if event is None:
            logger.warning("DLQ retry: event not found: %s", event_id)
            return False

        if event.status != FailedEventStatus.PENDING:
            logger.warning(
                "DLQ retry: event not pending: %s (status=%s)",
                event_id,
                event.status.value,
            )
            return False

        # Mark as retrying
        self._persistence.update_status(event_id, FailedEventStatus.RETRYING)

        try:
            handler(event.event_data)
            self._persistence.update_status(event_id, FailedEventStatus.RECOVERED)
            with self._lock:
                self._cache_pending_count = max(0, self._cache_pending_count - 1)
            logger.info("DLQ retry succeeded: %s", event_id)
            return True
        except Exception as e:
            # Update with new error, increment retry count
            event.retry_count += 1
            event.error_message = str(e)
            event.error_type = type(e).__name__
            event.status = FailedEventStatus.PENDING
            event.updated_at = time.time()
            self._persistence.save(event)
            logger.warning("DLQ retry failed: %s - %s", event_id, e)
            return False

    def discard_event(self, event_id: str) -> bool:
        """Mark an event as discarded (will not be retried).

        Args:
            event_id: ID of the event to discard

        Returns:
            True if event was found and discarded
        """
        success = self._persistence.update_status(event_id, FailedEventStatus.DISCARDED)
        if success:
            with self._lock:
                self._cache_pending_count = max(0, self._cache_pending_count - 1)
            logger.info("DLQ event discarded: %s", event_id)
        return success

    def get_stats(self) -> DLQStats:
        """Get DLQ statistics."""
        return self._persistence.get_stats()

    def cleanup(self, retention_days: int = DEFAULT_RETENTION_DAYS) -> int:
        """Remove old recovered/discarded events.

        Args:
            retention_days: Keep events for this many days

        Returns:
            Number of events removed
        """
        removed = self._persistence.cleanup_old(retention_days)
        if removed > 0:
            logger.info("DLQ cleanup: removed %d old events", removed)
        return removed

    @property
    def pending_count(self) -> int:
        """Get approximate count of pending events (cached)."""
        # Refresh cache if stale (older than 60s)
        if time.time() - self._cache_updated_at > 60:
            with self._lock:
                self._cache_pending_count = self._persistence.count_pending()
                self._cache_updated_at = time.time()
        return self._cache_pending_count

    def _evict_oldest(self, count: int) -> None:
        """Evict oldest pending events (FIFO)."""
        oldest = self._persistence.get_pending(count)
        for event in oldest:
            self._persistence.update_status(event.id, FailedEventStatus.DISCARDED)
            logger.debug("DLQ evicted oldest event: %s", event.id)


# ---------------------------------------------------------------------------
# Global instance
# ---------------------------------------------------------------------------

_dlq: EventDLQ | None = None


def get_event_dlq() -> EventDLQ:
    """Get or create the global event DLQ."""
    global _dlq
    if _dlq is None:
        _dlq = EventDLQ()
    return _dlq


def reset_event_dlq() -> None:
    """Reset the global event DLQ (for testing)."""
    global _dlq
    _dlq = None


__all__ = [
    "EventDLQ",
    "EventDLQPersistence",
    "DLQStats",
    "FailedEvent",
    "FailedEventStatus",
    "get_event_dlq",
    "reset_event_dlq",
]
