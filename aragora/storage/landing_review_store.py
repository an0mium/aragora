"""SQLite-backed storage for landing telemetry and wrong-answer review reports."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from aragora.persistence.db_config import DatabaseType, get_db_path
from aragora.storage.base_store import SQLiteStore

logger = logging.getLogger(__name__)

_DEFAULT_EVENT_LIMIT = 5_000
_DEFAULT_FEEDBACK_LIMIT = 1_000


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _cutoff_iso(window_seconds: float) -> str:
    """Return the lower timestamp bound for a rolling UTC window."""
    return (datetime.now(timezone.utc) - timedelta(seconds=window_seconds)).isoformat()


class LandingReviewStore(SQLiteStore):
    """Persist bounded landing telemetry and wrong-answer reports."""

    SCHEMA_NAME = "landing_review_store"
    SCHEMA_VERSION = 1
    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS landing_events (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            client_tag TEXT NOT NULL,
            data_json TEXT NOT NULL,
            timestamp TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_landing_events_timestamp
            ON landing_events(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_landing_events_type_timestamp
            ON landing_events(event_type, timestamp DESC);

        CREATE TABLE IF NOT EXISTS landing_feedback_reports (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            id TEXT NOT NULL UNIQUE,
            timestamp TEXT NOT NULL,
            client_tag TEXT NOT NULL,
            question TEXT,
            interpreted_question TEXT,
            final_answer_preview TEXT,
            result_warning TEXT,
            result_mode TEXT NOT NULL DEFAULT 'preview',
            debate_id TEXT,
            verdict TEXT,
            participant_count INTEGER,
            rewritten INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_landing_feedback_timestamp
            ON landing_feedback_reports(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_landing_feedback_client_tag
            ON landing_feedback_reports(client_tag, timestamp DESC);
    """

    def record_event(
        self,
        *,
        event_type: str,
        client_tag: str,
        data: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> None:
        """Persist a single landing telemetry event."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO landing_events (event_type, client_tag, data_json, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (
                    event_type,
                    client_tag,
                    json.dumps(data or {}, sort_keys=True),
                    timestamp or _utc_now_iso(),
                ),
            )
            self._trim_table(conn, "landing_events", keep_limit=_DEFAULT_EVENT_LIMIT)

    def list_recent_events(self, *, window_seconds: float) -> list[dict[str, Any]]:
        """Return recent landing telemetry events within the requested window."""
        rows = self.fetch_all(
            """
            SELECT event_type, client_tag, data_json, timestamp
            FROM landing_events
            WHERE timestamp >= ?
            ORDER BY sequence ASC
            """,
            (_cutoff_iso(window_seconds),),
        )
        return [
            {
                "event_type": row[0],
                "client_tag": row[1],
                "data": self._load_json_object(row[2]),
                "timestamp": row[3],
            }
            for row in rows
        ]

    def record_feedback(self, report: dict[str, Any]) -> None:
        """Persist a bounded wrong-answer report."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO landing_feedback_reports (
                    id,
                    timestamp,
                    client_tag,
                    question,
                    interpreted_question,
                    final_answer_preview,
                    result_warning,
                    result_mode,
                    debate_id,
                    verdict,
                    participant_count,
                    rewritten
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report["id"],
                    report.get("timestamp") or _utc_now_iso(),
                    report.get("client_tag") or "unknown",
                    report.get("question"),
                    report.get("interpreted_question"),
                    report.get("final_answer_preview"),
                    report.get("result_warning"),
                    report.get("result_mode") or "preview",
                    report.get("debate_id"),
                    report.get("verdict"),
                    report.get("participant_count"),
                    1 if report.get("rewritten") is True else 0,
                ),
            )
            self._trim_table(
                conn,
                "landing_feedback_reports",
                keep_limit=_DEFAULT_FEEDBACK_LIMIT,
            )

    def list_recent_feedback(
        self,
        *,
        window_seconds: float,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Return recent landing wrong-answer reports for review."""
        rows = self.fetch_all(
            """
            SELECT
                id,
                timestamp,
                client_tag,
                question,
                interpreted_question,
                final_answer_preview,
                result_warning,
                result_mode,
                debate_id,
                verdict,
                participant_count,
                rewritten
            FROM landing_feedback_reports
            WHERE timestamp >= ?
            ORDER BY sequence DESC
            LIMIT ?
            """,
            (_cutoff_iso(window_seconds), limit),
        )
        return [
            {
                "id": row[0],
                "timestamp": row[1],
                "client_tag": row[2],
                "question": row[3],
                "interpreted_question": row[4],
                "final_answer_preview": row[5],
                "result_warning": row[6],
                "result_mode": row[7],
                "debate_id": row[8],
                "verdict": row[9],
                "participant_count": row[10],
                "rewritten": bool(row[11]),
            }
            for row in rows
        ]

    def count_events(self) -> int:
        """Return the persisted landing telemetry event count."""
        row = self.fetch_one("SELECT COUNT(*) FROM landing_events")
        return int(row[0]) if row else 0

    def count_feedback(self) -> int:
        """Return the persisted landing feedback report count."""
        row = self.fetch_one("SELECT COUNT(*) FROM landing_feedback_reports")
        return int(row[0]) if row else 0

    def clear(self) -> None:
        """Clear all persisted landing review data. Used by tests."""
        with self.connection() as conn:
            conn.execute("DELETE FROM landing_events")
            conn.execute("DELETE FROM landing_feedback_reports")

    @staticmethod
    def _trim_table(conn: Any, table: str, *, keep_limit: int) -> None:
        """Keep only the most recent rows in a bounded table."""
        conn.execute(
            f"""
            DELETE FROM {table}
            WHERE sequence NOT IN (
                SELECT sequence
                FROM {table}
                ORDER BY sequence DESC
                LIMIT ?
            )
            """,
            (keep_limit,),
        )

    @staticmethod
    def _load_json_object(raw: Any) -> dict[str, Any]:
        """Best-effort JSON decoding for persisted event payloads."""
        if not isinstance(raw, str):
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Corrupt landing event payload encountered")
            return {}
        return value if isinstance(value, dict) else {}


_store: LandingReviewStore | None = None


def _resolve_store_path() -> Path:
    """Resolve the landing review database path."""
    override = os.environ.get("ARAGORA_LANDING_REVIEW_DB_PATH")
    if override:
        return Path(override)
    return get_db_path(DatabaseType.SUGGESTION_FEEDBACK)


def get_landing_review_store() -> LandingReviewStore:
    """Return the singleton landing review store, creating it if needed."""
    global _store  # noqa: PLW0603
    resolved_path = _resolve_store_path()
    if _store is None or _store.db_path != resolved_path:
        if _store is not None:
            _store.close()
        _store = LandingReviewStore(resolved_path)
    return _store


def reset_landing_review_store() -> None:
    """Reset the singleton landing review store. Used by tests."""
    global _store  # noqa: PLW0603
    if _store is not None:
        _store.close()
    _store = None
