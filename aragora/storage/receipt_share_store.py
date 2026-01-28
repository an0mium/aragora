"""
Receipt Share Store.

SQLite-backed storage for receipt shareable links.
Supports time-limited tokens with optional access limits.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Global singleton with thread-safe initialization
_store: Optional["ReceiptShareStore"] = None
_store_lock = threading.Lock()


def get_receipt_share_store() -> "ReceiptShareStore":
    """Get the global receipt share store instance."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                from aragora.config.legacy import DATA_DIR

                db_path = DATA_DIR / "receipt_shares.db"
                _store = ReceiptShareStore(db_path)
    return _store


class ReceiptShareStore:
    """SQLite-backed store for receipt share tokens."""

    def __init__(self, db_path: Path | str):
        """
        Initialize the receipt share store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS receipt_shares (
                token TEXT PRIMARY KEY,
                receipt_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                max_accesses INTEGER,
                access_count INTEGER DEFAULT 0,
                created_by TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_receipt_shares_receipt_id
            ON receipt_shares(receipt_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_receipt_shares_expires_at
            ON receipt_shares(expires_at)
            """
        )
        conn.commit()

    def save(
        self,
        token: str,
        receipt_id: str,
        expires_at: float,
        max_accesses: Optional[int] = None,
        created_by: Optional[str] = None,
    ) -> None:
        """
        Save a new share token.

        Args:
            token: Unique share token
            receipt_id: Receipt ID to share
            expires_at: Unix timestamp when link expires
            max_accesses: Maximum number of accesses (None = unlimited)
            created_by: User ID who created the link
        """
        conn = self._get_connection()
        conn.execute(
            """
            INSERT OR REPLACE INTO receipt_shares
            (token, receipt_id, created_at, expires_at, max_accesses, access_count, created_by)
            VALUES (?, ?, ?, ?, ?, 0, ?)
            """,
            (
                token,
                receipt_id,
                datetime.now(timezone.utc).timestamp(),
                expires_at,
                max_accesses,
                created_by,
            ),
        )
        conn.commit()
        logger.debug(f"Saved share token for receipt {receipt_id}")

    def get_by_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Get share info by token.

        Args:
            token: Share token

        Returns:
            Share info dict or None if not found
        """
        conn = self._get_connection()
        row = conn.execute(
            """
            SELECT token, receipt_id, created_at, expires_at, max_accesses, access_count, created_by
            FROM receipt_shares
            WHERE token = ?
            """,
            (token,),
        ).fetchone()

        if not row:
            return None

        return {
            "token": row["token"],
            "receipt_id": row["receipt_id"],
            "created_at": row["created_at"],
            "expires_at": row["expires_at"],
            "max_accesses": row["max_accesses"],
            "access_count": row["access_count"],
            "created_by": row["created_by"],
        }

    def get_by_receipt(self, receipt_id: str) -> list[Dict[str, Any]]:
        """
        Get all share tokens for a receipt.

        Args:
            receipt_id: Receipt ID

        Returns:
            List of share info dicts
        """
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT token, receipt_id, created_at, expires_at, max_accesses, access_count, created_by
            FROM receipt_shares
            WHERE receipt_id = ?
            ORDER BY created_at DESC
            """,
            (receipt_id,),
        ).fetchall()

        return [
            {
                "token": row["token"],
                "receipt_id": row["receipt_id"],
                "created_at": row["created_at"],
                "expires_at": row["expires_at"],
                "max_accesses": row["max_accesses"],
                "access_count": row["access_count"],
                "created_by": row["created_by"],
            }
            for row in rows
        ]

    def increment_access(self, token: str) -> bool:
        """
        Increment access count for a token.

        Args:
            token: Share token

        Returns:
            True if successful, False if token not found
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            UPDATE receipt_shares
            SET access_count = access_count + 1
            WHERE token = ?
            """,
            (token,),
        )
        conn.commit()
        return cursor.rowcount > 0

    def delete(self, token: str) -> bool:
        """
        Delete a share token.

        Args:
            token: Share token

        Returns:
            True if deleted, False if not found
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            DELETE FROM receipt_shares
            WHERE token = ?
            """,
            (token,),
        )
        conn.commit()
        return cursor.rowcount > 0

    def delete_by_receipt(self, receipt_id: str) -> int:
        """
        Delete all share tokens for a receipt.

        Args:
            receipt_id: Receipt ID

        Returns:
            Number of tokens deleted
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            DELETE FROM receipt_shares
            WHERE receipt_id = ?
            """,
            (receipt_id,),
        )
        conn.commit()
        return cursor.rowcount

    def cleanup_expired(self) -> int:
        """
        Delete expired share tokens.

        Returns:
            Number of tokens deleted
        """
        conn = self._get_connection()
        now = datetime.now(timezone.utc).timestamp()
        cursor = conn.execute(
            """
            DELETE FROM receipt_shares
            WHERE expires_at < ?
            """,
            (now,),
        )
        conn.commit()
        count = cursor.rowcount
        if count > 0:
            logger.info(f"Cleaned up {count} expired receipt share tokens")
        return count


__all__ = ["ReceiptShareStore", "get_receipt_share_store"]
