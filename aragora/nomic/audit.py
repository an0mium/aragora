"""
Nomic Loop Audit Logging.

DEPRECATION NOTICE:
    This module provides Nomic-specific audit logging. For new code, consider
    using the unified audit system in aragora.audit instead:

        from aragora.audit import AuditLog, AuditEvent, AuditCategory

    The unified system supports all audit categories including SYSTEM events
    for Nomic loop tracking, with compliance export (SOC 2, GDPR, HIPAA).

    This module is maintained for backward compatibility with existing
    Nomic loop code that depends on AuditEventType and AuditLogger.

Comprehensive audit trail for all Nomic loop operations.
Stores events in SQLite for queryability and durability.

Features:
- Complete event history with timestamps
- Gate decision tracking
- Phase transition logging
- Artifact hashes for verification
- Query API for analysis
"""

import hashlib
import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""

    # Lifecycle events
    CYCLE_START = "cycle_start"
    CYCLE_END = "cycle_end"
    CYCLE_ABORT = "cycle_abort"

    # Phase events
    PHASE_START = "phase_start"
    PHASE_END = "phase_end"
    PHASE_ERROR = "phase_error"

    # Gate events
    GATE_CHECK = "gate_check"
    GATE_APPROVED = "gate_approved"
    GATE_REJECTED = "gate_rejected"
    GATE_SKIPPED = "gate_skipped"

    # State machine events
    STATE_TRANSITION = "state_transition"
    CHECKPOINT_SAVED = "checkpoint_saved"
    CHECKPOINT_LOADED = "checkpoint_loaded"

    # Safety events
    CONSTITUTION_CHECK = "constitution_check"
    PROTECTED_FILE_ACCESS = "protected_file_access"
    ROLLBACK = "rollback"

    # Custom events
    CUSTOM = "custom"


@dataclass
class AuditEvent:
    """An audit event record."""

    event_type: AuditEventType
    cycle_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    phase: Optional[str] = None
    actor: str = "system"  # system, human, agent_name
    artifact_hash: Optional[str] = None
    success: bool = True
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "event_type": self.event_type.value,
            "cycle_id": self.cycle_id,
            "timestamp": self.timestamp.isoformat(),
            "phase": self.phase,
            "actor": self.actor,
            "artifact_hash": self.artifact_hash,
            "success": self.success,
            "message": self.message,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEvent":
        """Deserialize from dictionary."""
        return cls(
            event_type=AuditEventType(data.get("event_type", "custom")),
            cycle_id=data.get("cycle_id", ""),
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if data.get("timestamp")
                else datetime.utcnow()
            ),
            phase=data.get("phase"),
            actor=data.get("actor", "system"),
            artifact_hash=data.get("artifact_hash"),
            success=data.get("success", True),
            message=data.get("message", ""),
            metadata=data.get("metadata", {}),
        )


class AuditLogger:
    """
    SQLite-backed audit logger for Nomic loop operations.

    Thread-safe and supports concurrent writes.
    """

    # SQL schema for audit table
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS audit_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT NOT NULL,
        cycle_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        phase TEXT,
        actor TEXT NOT NULL DEFAULT 'system',
        artifact_hash TEXT,
        success INTEGER NOT NULL DEFAULT 1,
        message TEXT,
        metadata TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_audit_cycle_id ON audit_events(cycle_id);
    CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_events(event_type);
    CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp);
    CREATE INDEX IF NOT EXISTS idx_audit_phase ON audit_events(phase);
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        enabled: bool = True,
        max_events: int = 100000,
    ):
        """
        Initialize the audit logger.

        Args:
            db_path: Path to SQLite database. Defaults to .nomic/audit.db
            enabled: Whether logging is enabled
            max_events: Maximum events to retain (older events pruned)
        """
        self.enabled = enabled
        self.max_events = max_events

        if db_path is None:
            db_path = Path(".nomic/audit.db")

        self.db_path = db_path
        self._local = threading.local()
        self._lock = threading.Lock()

        # Ensure directory exists
        if self.enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), timeout=30)
            self._local.conn.row_factory = sqlite3.Row
        conn: sqlite3.Connection = self._local.conn
        return conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        try:
            conn = self._get_connection()
            conn.executescript(self.SCHEMA)
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize audit database: {e}")

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database transactions."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Audit transaction failed: {e}")
            raise

    def log(self, event: AuditEvent) -> bool:
        """
        Log an audit event.

        Args:
            event: The event to log

        Returns:
            True if logged successfully
        """
        if not self.enabled:
            return False

        try:
            with self._transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO audit_events
                    (event_type, cycle_id, timestamp, phase, actor, artifact_hash,
                     success, message, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_type.value,
                        event.cycle_id,
                        event.timestamp.isoformat(),
                        event.phase,
                        event.actor,
                        event.artifact_hash,
                        1 if event.success else 0,
                        event.message,
                        json.dumps(event.metadata),
                    ),
                )

            # Prune old events if needed
            self._prune_if_needed()
            return True

        except sqlite3.Error as e:
            logger.error(f"Failed to log audit event: {e}")
            return False

    def _prune_if_needed(self) -> None:
        """Prune old events if count exceeds max_events."""
        try:
            conn = self._get_connection()
            cursor = conn.execute("SELECT COUNT(*) FROM audit_events")
            count = cursor.fetchone()[0]

            if count > self.max_events:
                # Delete oldest 10% of events
                delete_count = int(self.max_events * 0.1)
                conn.execute(
                    """
                    DELETE FROM audit_events
                    WHERE id IN (
                        SELECT id FROM audit_events
                        ORDER BY timestamp ASC
                        LIMIT ?
                    )
                    """,
                    (delete_count,),
                )
                conn.commit()
                logger.info(f"Pruned {delete_count} old audit events")

        except sqlite3.Error as e:
            logger.warning(f"Failed to prune audit events: {e}")

    def log_cycle_start(self, cycle_id: str, config: Optional[Dict] = None) -> bool:
        """Log cycle start event."""
        return self.log(
            AuditEvent(
                event_type=AuditEventType.CYCLE_START,
                cycle_id=cycle_id,
                message="Nomic cycle started",
                metadata={"config": config or {}},
            )
        )

    def log_cycle_end(self, cycle_id: str, success: bool, duration_seconds: float) -> bool:
        """Log cycle end event."""
        return self.log(
            AuditEvent(
                event_type=AuditEventType.CYCLE_END,
                cycle_id=cycle_id,
                success=success,
                message=f"Nomic cycle {'completed' if success else 'failed'}",
                metadata={"duration_seconds": duration_seconds},
            )
        )

    def log_phase_start(self, cycle_id: str, phase: str) -> bool:
        """Log phase start event."""
        return self.log(
            AuditEvent(
                event_type=AuditEventType.PHASE_START,
                cycle_id=cycle_id,
                phase=phase,
                message=f"Phase {phase} started",
            )
        )

    def log_phase_end(
        self,
        cycle_id: str,
        phase: str,
        success: bool,
        duration_seconds: float,
        result_summary: Optional[str] = None,
    ) -> bool:
        """Log phase end event."""
        return self.log(
            AuditEvent(
                event_type=AuditEventType.PHASE_END,
                cycle_id=cycle_id,
                phase=phase,
                success=success,
                message=f"Phase {phase} {'completed' if success else 'failed'}",
                metadata={
                    "duration_seconds": duration_seconds,
                    "result_summary": result_summary or "",
                },
            )
        )

    def log_gate_decision(
        self,
        cycle_id: str,
        gate_type: str,
        status: str,
        approver: str,
        artifact_hash: str,
        reason: str,
    ) -> bool:
        """Log gate decision event."""
        event_type = {
            "approved": AuditEventType.GATE_APPROVED,
            "rejected": AuditEventType.GATE_REJECTED,
            "skipped": AuditEventType.GATE_SKIPPED,
        }.get(status.lower(), AuditEventType.GATE_CHECK)

        return self.log(
            AuditEvent(
                event_type=event_type,
                cycle_id=cycle_id,
                phase=gate_type,
                actor=approver,
                artifact_hash=artifact_hash,
                success=status.lower() in ("approved", "skipped"),
                message=reason,
                metadata={"gate_type": gate_type, "status": status},
            )
        )

    def log_state_transition(
        self, cycle_id: str, from_state: str, to_state: str, trigger: str
    ) -> bool:
        """Log state machine transition."""
        return self.log(
            AuditEvent(
                event_type=AuditEventType.STATE_TRANSITION,
                cycle_id=cycle_id,
                message=f"Transition: {from_state} -> {to_state}",
                metadata={
                    "from_state": from_state,
                    "to_state": to_state,
                    "trigger": trigger,
                },
            )
        )

    def log_rollback(self, cycle_id: str, reason: str, files_affected: List[str]) -> bool:
        """Log rollback event."""
        return self.log(
            AuditEvent(
                event_type=AuditEventType.ROLLBACK,
                cycle_id=cycle_id,
                success=False,
                message=f"Rollback triggered: {reason}",
                metadata={
                    "reason": reason,
                    "files_affected": files_affected,
                },
            )
        )

    def get_events(
        self,
        cycle_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        phase: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditEvent]:
        """
        Query audit events.

        Args:
            cycle_id: Filter by cycle ID
            event_type: Filter by event type
            phase: Filter by phase
            limit: Maximum events to return
            offset: Offset for pagination

        Returns:
            List of matching audit events
        """
        if not self.enabled:
            return []

        try:
            query = "SELECT * FROM audit_events WHERE 1=1"
            params: List[Any] = []

            if cycle_id:
                query += " AND cycle_id = ?"
                params.append(cycle_id)

            if event_type:
                query += " AND event_type = ?"
                params.append(event_type.value)

            if phase:
                query += " AND phase = ?"
                params.append(phase)

            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            conn = self._get_connection()
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            events = []
            for row in rows:
                events.append(
                    AuditEvent(
                        event_type=AuditEventType(row["event_type"]),
                        cycle_id=row["cycle_id"],
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        phase=row["phase"],
                        actor=row["actor"],
                        artifact_hash=row["artifact_hash"],
                        success=bool(row["success"]),
                        message=row["message"] or "",
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    )
                )
            return events

        except sqlite3.Error as e:
            logger.error(f"Failed to query audit events: {e}")
            return []

    def get_cycle_summary(self, cycle_id: str) -> Dict[str, Any]:
        """
        Get summary of a specific cycle.

        Args:
            cycle_id: The cycle to summarize

        Returns:
            Dictionary with cycle summary
        """
        if not self.enabled:
            return {}

        events = self.get_events(cycle_id=cycle_id, limit=1000)

        if not events:
            return {"cycle_id": cycle_id, "found": False}

        # Calculate summary
        phases = set()
        gates = []
        errors = []
        duration = 0.0

        for event in events:
            if event.phase:
                phases.add(event.phase)

            if event.event_type in (
                AuditEventType.GATE_APPROVED,
                AuditEventType.GATE_REJECTED,
                AuditEventType.GATE_SKIPPED,
            ):
                gates.append(
                    {
                        "type": event.metadata.get("gate_type", "unknown"),
                        "status": event.metadata.get("status", "unknown"),
                        "approver": event.actor,
                    }
                )

            if event.event_type == AuditEventType.PHASE_ERROR or not event.success:
                errors.append(
                    {
                        "phase": event.phase,
                        "message": event.message,
                    }
                )

            if event.event_type == AuditEventType.CYCLE_END:
                duration = event.metadata.get("duration_seconds", 0.0)

        # Determine overall status
        cycle_end = next((e for e in events if e.event_type == AuditEventType.CYCLE_END), None)

        return {
            "cycle_id": cycle_id,
            "found": True,
            "success": cycle_end.success if cycle_end else None,
            "duration_seconds": duration,
            "phases_executed": list(phases),
            "gate_decisions": gates,
            "errors": errors,
            "total_events": len(events),
        }

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger(
    db_path: Optional[Path] = None,
    enabled: bool = True,
) -> AuditLogger:
    """
    Get or create the global audit logger.

    Args:
        db_path: Optional path to database
        enabled: Whether logging is enabled

    Returns:
        The global AuditLogger instance
    """
    global _audit_logger

    if _audit_logger is None:
        _audit_logger = AuditLogger(db_path=db_path, enabled=enabled)

    return _audit_logger


def hash_artifact(content: str) -> str:
    """Create SHA-256 hash of content for audit tracking."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


__all__ = [
    "AuditLogger",
    "AuditEvent",
    "AuditEventType",
    "get_audit_logger",
    "hash_artifact",
]
