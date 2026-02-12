"""
Incident Persistence Store.

SQLite-backed storage for status page incidents, enabling:
- CRUD operations for incidents
- Incident update timelines
- Auto-creation from SLO violations
- Query by status, severity, and time range

Usage:
    from aragora.observability.incident_store import get_incident_store

    store = get_incident_store()

    # Create an incident
    incident_id = store.create_incident(
        title="API latency elevated",
        severity="major",
        components=["api"],
    )

    # Add an update
    store.add_update(incident_id, "investigating", "Identified spike in p99 latency")

    # Resolve
    store.resolve_incident(incident_id, "Scaled up API pods, latency normalized")

    # Query
    active = store.get_active_incidents()
    recent = store.get_recent_incidents(days=7)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections.abc import Iterator

from aragora.config import resolve_db_path

logger = logging.getLogger(__name__)


@dataclass
class IncidentUpdate:
    """A timestamped update to an incident."""

    id: str
    status: str
    message: str
    timestamp: str


@dataclass
class IncidentRecord:
    """A persisted incident."""

    id: str
    title: str
    status: str  # investigating, identified, monitoring, resolved
    severity: str  # minor, major, critical
    components: list[str]
    created_at: str
    updated_at: str
    resolved_at: str | None = None
    updates: list[IncidentUpdate] = field(default_factory=list)
    source: str | None = None  # manual, slo_violation, etc.

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "severity": self.severity,
            "components": self.components,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "resolved_at": self.resolved_at,
            "updates": [
                {
                    "id": u.id,
                    "status": u.status,
                    "message": u.message,
                    "timestamp": u.timestamp,
                }
                for u in self.updates
            ],
            "source": self.source,
        }


class IncidentStore:
    """SQLite-backed incident store for the status page."""

    _CREATE_INCIDENTS = """
        CREATE TABLE IF NOT EXISTS incidents (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'investigating',
            severity TEXT NOT NULL DEFAULT 'minor',
            components TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            resolved_at TEXT,
            source TEXT
        )
    """

    _CREATE_UPDATES = """
        CREATE TABLE IF NOT EXISTS incident_updates (
            id TEXT PRIMARY KEY,
            incident_id TEXT NOT NULL,
            status TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (incident_id) REFERENCES incidents(id)
        )
    """

    _CREATE_INDICES = [
        "CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status)",
        "CREATE INDEX IF NOT EXISTS idx_incidents_severity ON incidents(severity)",
        "CREATE INDEX IF NOT EXISTS idx_incidents_created ON incidents(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_updates_incident ON incident_updates(incident_id)",
    ]

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or self._default_db_path()
        self._lock = threading.Lock()
        self._ensure_schema()

    @staticmethod
    def _default_db_path() -> str:
        db_path = resolve_db_path("incidents.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        return str(db_path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(self._CREATE_INCIDENTS)
            conn.execute(self._CREATE_UPDATES)
            for idx in self._CREATE_INDICES:
                conn.execute(idx)

    def create_incident(
        self,
        title: str,
        severity: str = "minor",
        components: list[str] | None = None,
        source: str | None = None,
        initial_message: str | None = None,
    ) -> str:
        """Create a new incident.

        Returns the incident ID.
        """
        incident_id = str(uuid.uuid4())[:12]
        now = datetime.now(timezone.utc).isoformat()
        comps_json = json.dumps(components or [])

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO incidents (id, title, status, severity, components,
                                       created_at, updated_at, source)
                VALUES (?, ?, 'investigating', ?, ?, ?, ?, ?)
                """,
                (incident_id, title, severity, comps_json, now, now, source),
            )

            if initial_message:
                update_id = str(uuid.uuid4())[:12]
                conn.execute(
                    """
                    INSERT INTO incident_updates (id, incident_id, status, message, timestamp)
                    VALUES (?, ?, 'investigating', ?, ?)
                    """,
                    (update_id, incident_id, initial_message, now),
                )

        logger.info(f"Incident created: {incident_id} - {title} ({severity})")
        return incident_id

    def add_update(
        self,
        incident_id: str,
        status: str,
        message: str,
    ) -> str:
        """Add an update to an incident.

        Args:
            incident_id: Incident to update.
            status: New status (investigating, identified, monitoring, resolved).
            message: Update message.

        Returns the update ID.
        """
        update_id = str(uuid.uuid4())[:12]
        now = datetime.now(timezone.utc).isoformat()

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO incident_updates (id, incident_id, status, message, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (update_id, incident_id, status, message, now),
            )
            resolved_at = now if status == "resolved" else None
            if resolved_at:
                conn.execute(
                    "UPDATE incidents SET status = ?, updated_at = ?, resolved_at = ? WHERE id = ?",
                    (status, now, resolved_at, incident_id),
                )
            else:
                conn.execute(
                    "UPDATE incidents SET status = ?, updated_at = ? WHERE id = ?",
                    (status, now, incident_id),
                )

        return update_id

    def resolve_incident(self, incident_id: str, message: str = "Incident resolved") -> str:
        """Resolve an incident.

        Returns the update ID.
        """
        return self.add_update(incident_id, "resolved", message)

    def get_incident(self, incident_id: str) -> IncidentRecord | None:
        """Get a single incident with its updates."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()

            if not row:
                return None

            updates = conn.execute(
                "SELECT * FROM incident_updates WHERE incident_id = ? ORDER BY timestamp ASC",
                (incident_id,),
            ).fetchall()

        return self._row_to_record(row, updates)

    def get_active_incidents(self) -> list[IncidentRecord]:
        """Get all active (non-resolved) incidents."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM incidents WHERE status != 'resolved' ORDER BY created_at DESC"
            ).fetchall()

            results = []
            for row in rows:
                updates = conn.execute(
                    "SELECT * FROM incident_updates WHERE incident_id = ? ORDER BY timestamp ASC",
                    (row["id"],),
                ).fetchall()
                results.append(self._row_to_record(row, updates))

        return results

    def get_recent_incidents(self, days: int = 7) -> list[IncidentRecord]:
        """Get recently resolved incidents."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM incidents
                WHERE status = 'resolved' AND resolved_at >= ?
                ORDER BY resolved_at DESC
                """,
                (cutoff,),
            ).fetchall()

            results = []
            for row in rows:
                updates = conn.execute(
                    "SELECT * FROM incident_updates WHERE incident_id = ? ORDER BY timestamp ASC",
                    (row["id"],),
                ).fetchall()
                results.append(self._row_to_record(row, updates))

        return results

    def _row_to_record(self, row: sqlite3.Row, updates: list[sqlite3.Row]) -> IncidentRecord:
        try:
            components = json.loads(row["components"])
        except (json.JSONDecodeError, TypeError):
            components = []

        return IncidentRecord(
            id=row["id"],
            title=row["title"],
            status=row["status"],
            severity=row["severity"],
            components=components,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            resolved_at=row["resolved_at"],
            source=row["source"],
            updates=[
                IncidentUpdate(
                    id=u["id"],
                    status=u["status"],
                    message=u["message"],
                    timestamp=u["timestamp"],
                )
                for u in updates
            ],
        )

    def create_from_slo_violation(
        self,
        slo_name: str,
        severity: str,
        message: str,
        components: list[str] | None = None,
    ) -> str:
        """Auto-create an incident from an SLO violation.

        Maps SLO names to components and creates an incident.
        """
        slo_to_components = {
            "availability": ["api"],
            "latency_p99": ["api"],
            "debate_success": ["debates"],
            "knowledge_retrieval": ["knowledge"],
            "websocket_latency": ["websocket"],
            "auth_latency": ["auth"],
        }

        mapped_components = components or slo_to_components.get(slo_name, ["api"])
        title = f"SLO Violation: {slo_name} ({severity})"

        return self.create_incident(
            title=title,
            severity=severity,
            components=mapped_components,
            source="slo_violation",
            initial_message=message,
        )


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_global_store: IncidentStore | None = None
_store_lock = threading.Lock()


def get_incident_store(db_path: str | None = None) -> IncidentStore:
    """Get or create the global incident store."""
    global _global_store
    with _store_lock:
        if _global_store is None:
            _global_store = IncidentStore(db_path=db_path)
        return _global_store


def reset_incident_store() -> None:
    """Reset the global store (for testing)."""
    global _global_store
    with _store_lock:
        _global_store = None


__all__ = [
    "IncidentStore",
    "IncidentRecord",
    "IncidentUpdate",
    "get_incident_store",
    "reset_incident_store",
]
