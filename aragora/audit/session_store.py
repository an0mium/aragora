"""SQLite-backed persistent store for local-mode audit sessions.

Local/air-gapped ``aragora audit`` runs span multiple CLI invocations
(create -> start -> status -> findings -> export -> report). Without a
durable store, each invocation starts a fresh Python process with an empty
in-memory session dict, so a session created by ``create`` is invisible to a
later ``start``/``status`` call.

This store persists full ``AuditSession`` state (including nested findings,
enums, and datetimes) to a single SQLite database under ``ARAGORA_DATA_DIR`` or
``~/.aragora``. The path is deliberately independent of the current working
directory because local audit workflows span separate CLI invocations.

Server/API mode does not use this store; it keeps its own server-side storage.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from aragora.audit.document_auditor import AuditSession

logger = logging.getLogger(__name__)

_DB_FILENAME = "audit_sessions.db"


def get_default_data_dir() -> Path:
    """Resolve the stable local audit data directory.

    Re-exported here (rather than imported at call sites) so tests can
    monkeypatch ``aragora.audit.session_store.get_default_data_dir`` directly,
    per the repo testing convention of patching the function, not a constant.
    """
    override = os.environ.get("ARAGORA_DATA_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".aragora"


def _resolve_db_path() -> Path:
    """Resolve the audit-session database path under the data directory."""
    return get_default_data_dir() / _DB_FILENAME


class AuditSessionStore:
    """Durable single-user store for audit sessions.

    Stores one row per session keyed by session id. The full session is
    serialized to JSON so findings and other nested state round-trip faithfully.
    """

    def __init__(self, db_path: str | os.PathLike[str] | None = None) -> None:
        self._db_path = Path(db_path) if db_path is not None else _resolve_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        # WAL is friendlier for the occasional concurrent CLI invocation.
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:  # pragma: no cover - filesystem dependent
            pass
        return conn

    def _init_db(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_sessions (
                    id TEXT PRIMARY KEY,
                    org_id TEXT,
                    status TEXT,
                    created_at TEXT,
                    data TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_sessions_org ON audit_sessions(org_id)"
            )
            conn.commit()

    def save(self, session: "AuditSession") -> None:
        """Insert or update a session (upsert by id)."""
        payload = json.dumps(session.to_storage_dict())
        created_at = session.created_at.isoformat() if session.created_at else None
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO audit_sessions (id, org_id, status, created_at, data)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    org_id=excluded.org_id,
                    status=excluded.status,
                    created_at=excluded.created_at,
                    data=excluded.data
                """,
                (session.id, session.org_id, session.status.value, created_at, payload),
            )
            conn.commit()

    def get(self, session_id: str) -> "AuditSession | None":
        """Load a single session by id, or None if absent."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT data FROM audit_sessions WHERE id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        return self._deserialize(row["data"])

    def list(
        self,
        org_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list["AuditSession"]:
        """List stored sessions, newest first, optionally filtered by org/status."""
        query = "SELECT data FROM audit_sessions"
        clauses: list[str] = []
        params: list[object] = []
        if org_id:
            clauses.append("org_id = ?")
            params.append(org_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with closing(self._connect()) as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        sessions = []
        for row in rows:
            session = self._deserialize(row["data"])
            if session is not None:
                sessions.append(session)
        return sessions

    def delete(self, session_id: str) -> bool:
        with closing(self._connect()) as conn:
            cur = conn.execute("DELETE FROM audit_sessions WHERE id = ?", (session_id,))
            conn.commit()
            return cur.rowcount > 0

    @staticmethod
    def _deserialize(raw: str) -> "AuditSession | None":
        from aragora.audit.document_auditor import AuditSession

        try:
            data = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Failed to decode stored audit session payload")
            return None
        try:
            return AuditSession.from_storage_dict(data)
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Failed to rebuild audit session from store: %s", exc)
            return None


__all__ = ["AuditSessionStore", "get_default_data_dir"]
