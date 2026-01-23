"""
Snooze Persistence Store.

SQLite-backed storage for email snooze tracking.
Stores snoozed emails with their wake times for scheduled processing.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from aragora.storage.base_store import SQLiteStore

logger = logging.getLogger(__name__)


class SnoozeStore(SQLiteStore):
    """
    Persistent storage for email snoozes.

    Stores snoozed emails with metadata for scheduled
    wake-up processing and user querying.
    """

    SCHEMA_NAME = "snooze_store"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS snoozes (
            id TEXT PRIMARY KEY,
            email_id TEXT NOT NULL UNIQUE,
            user_id TEXT NOT NULL DEFAULT 'default',
            thread_id TEXT,
            subject TEXT,
            sender TEXT,
            snooze_until TEXT NOT NULL,
            label TEXT DEFAULT 'Snoozed',
            reason TEXT,
            snoozed_at TEXT NOT NULL,
            woken_at TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_snoozes_user_status
            ON snoozes(user_id, status);
        CREATE INDEX IF NOT EXISTS idx_snoozes_wake_time
            ON snoozes(snooze_until) WHERE status = 'active';
        CREATE INDEX IF NOT EXISTS idx_snoozes_email
            ON snoozes(email_id);
    """

    def save_snooze(self, snooze: dict[str, Any]) -> None:
        """Save or update a snooze."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO snoozes
                (id, email_id, user_id, thread_id, subject, sender,
                 snooze_until, label, reason, snoozed_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snooze.get("id", f"snz_{snooze['email_id']}_{datetime.now().timestamp()}"),
                    snooze["email_id"],
                    snooze.get("user_id", "default"),
                    snooze.get("thread_id"),
                    snooze.get("subject", ""),
                    snooze.get("sender", ""),
                    snooze["snooze_until"],
                    snooze.get("label", "Snoozed"),
                    snooze.get("reason", ""),
                    snooze.get("snoozed_at", datetime.now().isoformat()),
                    snooze.get("status", "active"),
                ),
            )

    def get_snooze(self, email_id: str) -> dict[str, Any] | None:
        """Get snooze by email ID."""
        row = self.fetch_one(
            "SELECT * FROM snoozes WHERE email_id = ? AND status = 'active'",
            (email_id,),
        )
        return dict(row) if row else None

    def get_active_snoozes(
        self,
        user_id: str = "default",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get all active snoozes for a user."""
        rows = self.fetch_all(
            """
            SELECT * FROM snoozes
            WHERE user_id = ? AND status = 'active'
            ORDER BY snooze_until ASC
            LIMIT ?
            """,
            (user_id, limit),
        )
        return [dict(row) for row in rows]

    def get_due_snoozes(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """Get snoozes that are due (past their wake time)."""
        now = datetime.now().isoformat()
        if user_id:
            rows = self.fetch_all(
                """
                SELECT * FROM snoozes
                WHERE user_id = ? AND status = 'active' AND snooze_until <= ?
                ORDER BY snooze_until ASC
                """,
                (user_id, now),
            )
        else:
            rows = self.fetch_all(
                """
                SELECT * FROM snoozes
                WHERE status = 'active' AND snooze_until <= ?
                ORDER BY snooze_until ASC
                """,
                (now,),
            )
        return [dict(row) for row in rows]

    def cancel_snooze(self, email_id: str) -> bool:
        """Cancel a snooze."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                UPDATE snoozes
                SET status = 'cancelled', woken_at = ?
                WHERE email_id = ? AND status = 'active'
                """,
                (datetime.now().isoformat(), email_id),
            )
            return cursor.rowcount > 0

    def wake_snooze(self, email_id: str) -> bool:
        """Mark a snooze as woken (processed)."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                UPDATE snoozes
                SET status = 'woken', woken_at = ?
                WHERE email_id = ? AND status = 'active'
                """,
                (datetime.now().isoformat(), email_id),
            )
            return cursor.rowcount > 0

    def process_due_snoozes(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """
        Get and mark all due snoozes as woken.

        Returns the list of snoozes that were processed.
        """
        due = self.get_due_snoozes(user_id)
        now = datetime.now().isoformat()

        with self.connection() as conn:
            for snooze in due:
                conn.execute(
                    """
                    UPDATE snoozes
                    SET status = 'woken', woken_at = ?
                    WHERE id = ?
                    """,
                    (now, snooze["id"]),
                )

        return due

    def get_stats(self, user_id: str = "default") -> dict[str, Any]:
        """Get snooze statistics for a user."""
        now = datetime.now().isoformat()
        with self.connection() as conn:
            # Active snoozes
            active = conn.execute(
                "SELECT COUNT(*) FROM snoozes WHERE user_id = ? AND status = 'active'",
                (user_id,),
            ).fetchone()[0]

            # Due now
            due = conn.execute(
                """
                SELECT COUNT(*) FROM snoozes
                WHERE user_id = ? AND status = 'active' AND snooze_until <= ?
                """,
                (user_id, now),
            ).fetchone()[0]

            # Woken today
            today = datetime.now().replace(hour=0, minute=0, second=0).isoformat()
            woken_today = conn.execute(
                """
                SELECT COUNT(*) FROM snoozes
                WHERE user_id = ? AND status = 'woken' AND woken_at >= ?
                """,
                (user_id, today),
            ).fetchone()[0]

        return {
            "active_snoozes": active,
            "due_now": due,
            "woken_today": woken_today,
        }


# Singleton instance factory
_store_instance: SnoozeStore | None = None


def get_snooze_store(db_path: str | None = None) -> SnoozeStore:
    """Get or create the snooze store singleton."""
    global _store_instance
    if _store_instance is None:
        from aragora.config import get_data_dir

        if db_path is None:
            db_path = str(get_data_dir() / "snoozes.db")
        _store_instance = SnoozeStore(db_path)
    return _store_instance
