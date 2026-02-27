"""SQLite-backed store for persisting playground debate results.

Enables shareable debate links by saving results with a unique ID
and supporting retrieval via GET /api/v1/playground/debate/{id}.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from aragora.storage.base_store import SQLiteStore

logger = logging.getLogger(__name__)

_DEFAULT_TTL_DAYS = 30


class DebateResultStore(SQLiteStore):
    """Persist playground debate results for shareable links."""

    SCHEMA_NAME = "debate_results"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS debate_results (
            id TEXT PRIMARY KEY,
            topic TEXT NOT NULL,
            result_json TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'playground',
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_debate_results_created
            ON debate_results(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_debate_results_expires
            ON debate_results(expires_at);
    """

    def save(
        self,
        debate_id: str,
        topic: str,
        result: dict[str, Any],
        *,
        source: str = "playground",
        ttl_days: int = _DEFAULT_TTL_DAYS,
    ) -> None:
        """Save a debate result."""
        now = time.time()
        expires_at = now + (ttl_days * 86400)
        result_json = json.dumps(result, default=str)

        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO debate_results
                    (id, topic, result_json, source, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (debate_id, topic, result_json, source, now, expires_at),
            )

    def get(self, debate_id: str) -> dict[str, Any] | None:
        """Retrieve a debate result by ID, returning None if expired or missing."""
        now = time.time()
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT result_json FROM debate_results
                WHERE id = ? AND expires_at > ?
                """,
                (debate_id, now),
            ).fetchone()

        if row is None:
            return None

        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            logger.warning("Corrupt debate result for %s", debate_id)
            return None

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent debate metadata (not full results)."""
        now = time.time()
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, topic, source, created_at FROM debate_results
                WHERE expires_at > ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (now, limit),
            ).fetchall()

        return [
            {
                "id": r[0],
                "topic": r[1],
                "source": r[2],
                "created_at": r[3],
            }
            for r in rows
        ]

    def cleanup_expired(self) -> int:
        """Delete expired entries. Returns count of deleted rows."""
        now = time.time()
        with self.connection() as conn:
            cursor = conn.execute(
                "DELETE FROM debate_results WHERE expires_at <= ?",
                (now,),
            )
            return cursor.rowcount


# Module-level singleton
_store: DebateResultStore | None = None


def get_debate_store() -> DebateResultStore:
    """Get or create the singleton DebateResultStore."""
    global _store  # noqa: PLW0603
    if _store is None:
        _store = DebateResultStore("debate_results.db")
    return _store
