"""
Follow-Up Persistence Store.

SQLite-backed storage for email follow-up tracking.
Provides persistent storage for follow-up items with efficient querying.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from aragora.storage.base_store import SQLiteStore

logger = logging.getLogger(__name__)


class FollowUpStore(SQLiteStore):
    """
    Persistent storage for email follow-ups.

    Stores follow-up items with their status, timestamps,
    and metadata for efficient querying and analytics.
    """

    SCHEMA_NAME = "followup_store"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS followups (
            id TEXT PRIMARY KEY,
            email_id TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            user_id TEXT NOT NULL DEFAULT 'default',
            subject TEXT,
            recipient TEXT,
            sent_at TEXT NOT NULL,
            expected_by TEXT,
            status TEXT NOT NULL DEFAULT 'awaiting',
            priority TEXT NOT NULL DEFAULT 'normal',
            notes TEXT,
            reminder_count INTEGER DEFAULT 0,
            last_reminder TEXT,
            resolved_at TEXT,
            resolution_notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_followups_user_status
            ON followups(user_id, status);
        CREATE INDEX IF NOT EXISTS idx_followups_thread
            ON followups(thread_id);
        CREATE INDEX IF NOT EXISTS idx_followups_recipient
            ON followups(recipient);
        CREATE INDEX IF NOT EXISTS idx_followups_expected_by
            ON followups(expected_by);
    """

    def save_followup(self, followup: dict[str, Any]) -> None:
        """Save or update a follow-up item."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO followups
                (id, email_id, thread_id, user_id, subject, recipient,
                 sent_at, expected_by, status, priority, notes,
                 reminder_count, last_reminder, resolved_at, resolution_notes,
                 updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    followup["id"],
                    followup["email_id"],
                    followup["thread_id"],
                    followup.get("user_id", "default"),
                    followup.get("subject", ""),
                    followup.get("recipient", ""),
                    followup["sent_at"],
                    followup.get("expected_by"),
                    followup.get("status", "awaiting"),
                    followup.get("priority", "normal"),
                    followup.get("notes", ""),
                    followup.get("reminder_count", 0),
                    followup.get("last_reminder"),
                    followup.get("resolved_at"),
                    followup.get("resolution_notes"),
                    datetime.now().isoformat(),
                ),
            )

    def get_followup(self, followup_id: str) -> dict[str, Any] | None:
        """Get a single follow-up by ID."""
        row = self.fetch_one(
            "SELECT * FROM followups WHERE id = ?",
            (followup_id,),
        )
        return dict(row) if row else None

    def get_pending_followups(
        self,
        user_id: str = "default",
        include_resolved: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get pending follow-ups for a user."""
        if include_resolved:
            rows = self.fetch_all(
                """
                SELECT * FROM followups
                WHERE user_id = ?
                ORDER BY
                    CASE status
                        WHEN 'overdue' THEN 1
                        WHEN 'awaiting' THEN 2
                        ELSE 3
                    END,
                    expected_by ASC
                LIMIT ?
                """,
                (user_id, limit),
            )
        else:
            rows = self.fetch_all(
                """
                SELECT * FROM followups
                WHERE user_id = ? AND status IN ('awaiting', 'overdue')
                ORDER BY
                    CASE status
                        WHEN 'overdue' THEN 1
                        WHEN 'awaiting' THEN 2
                        ELSE 3
                    END,
                    expected_by ASC
                LIMIT ?
                """,
                (user_id, limit),
            )
        return [dict(row) for row in rows]

    def get_followups_by_thread(self, thread_id: str) -> list[dict[str, Any]]:
        """Get all follow-ups for a thread."""
        rows = self.fetch_all(
            "SELECT * FROM followups WHERE thread_id = ?",
            (thread_id,),
        )
        return [dict(row) for row in rows]

    def update_status(
        self,
        followup_id: str,
        status: str,
        resolved_at: str | None = None,
        notes: str | None = None,
    ) -> bool:
        """Update follow-up status."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                UPDATE followups
                SET status = ?, resolved_at = ?, resolution_notes = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, resolved_at, notes, datetime.now().isoformat(), followup_id),
            )
            return cursor.rowcount > 0

    def increment_reminder(self, followup_id: str) -> bool:
        """Increment reminder count and update last reminder time."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                UPDATE followups
                SET reminder_count = reminder_count + 1,
                    last_reminder = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (datetime.now().isoformat(), datetime.now().isoformat(), followup_id),
            )
            return cursor.rowcount > 0

    def delete_followup(self, followup_id: str) -> bool:
        """Delete a follow-up."""
        with self.connection() as conn:
            cursor = conn.execute(
                "DELETE FROM followups WHERE id = ?",
                (followup_id,),
            )
            return cursor.rowcount > 0

    def get_stats(self, user_id: str = "default") -> dict[str, Any]:
        """Get follow-up statistics for a user."""
        with self.connection() as conn:
            # Total pending
            total = conn.execute(
                "SELECT COUNT(*) FROM followups WHERE user_id = ? AND status IN ('awaiting', 'overdue')",
                (user_id,),
            ).fetchone()[0]

            # Overdue count
            overdue = conn.execute(
                "SELECT COUNT(*) FROM followups WHERE user_id = ? AND status = 'overdue'",
                (user_id,),
            ).fetchone()[0]

            # Resolved this week
            week_ago = (
                datetime.now().replace(hour=0, minute=0, second=0)
                - __import__("datetime").timedelta(days=7)
            ).isoformat()
            resolved = conn.execute(
                """
                SELECT COUNT(*) FROM followups
                WHERE user_id = ? AND status IN ('received', 'resolved')
                AND resolved_at >= ?
                """,
                (user_id, week_ago),
            ).fetchone()[0]

            # Average wait days for pending
            avg_wait = (
                conn.execute(
                    """
                SELECT AVG(julianday('now') - julianday(sent_at))
                FROM followups
                WHERE user_id = ? AND status IN ('awaiting', 'overdue')
                """,
                    (user_id,),
                ).fetchone()[0]
                or 0
            )

        return {
            "total_pending": total,
            "overdue_count": overdue,
            "resolved_this_week": resolved,
            "avg_wait_days": round(avg_wait, 1),
        }

    def mark_overdue(self, user_id: str = "default") -> int:
        """Mark overdue follow-ups (called periodically)."""
        now = datetime.now().isoformat()
        with self.connection() as conn:
            cursor = conn.execute(
                """
                UPDATE followups
                SET status = 'overdue', updated_at = ?
                WHERE user_id = ? AND status = 'awaiting'
                AND expected_by < ?
                """,
                (now, user_id, now),
            )
            return cursor.rowcount


# Singleton instance factory
_store_instance: FollowUpStore | None = None


def get_followup_store(db_path: str | None = None) -> FollowUpStore:
    """Get or create the follow-up store singleton."""
    global _store_instance
    if _store_instance is None:
        from aragora.config.legacy import get_db_path

        if db_path is None:
            db_path = str(get_db_path("followups.db"))
        _store_instance = FollowUpStore(db_path)
    return _store_instance
