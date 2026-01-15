"""
SQLite-backed Share Link Storage.

Provides persistent storage for debate sharing settings with:
- TTL-based expiration and cleanup
- Thread-safe concurrent access
- View count tracking
- Token-based lookups

Replaces the in-memory ShareStore for production use.
"""

from __future__ import annotations

import logging
import secrets
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from aragora.storage.base_store import SQLiteStore

if TYPE_CHECKING:
    from aragora.server.handlers.social.sharing import ShareSettings

logger = logging.getLogger(__name__)


class ShareLinkStore(SQLiteStore):
    """
    SQLite-backed store for debate sharing settings.

    Provides persistent storage with automatic TTL cleanup,
    replacing the in-memory ShareStore for production deployments.

    Features:
    - Atomic save/update operations
    - Token-based lookups for public access
    - Automatic expired link cleanup
    - View count tracking

    Usage:
        store = ShareLinkStore("data/share_links.db")
        store.save(settings)
        settings = store.get_by_token("abc123")
    """

    SCHEMA_NAME = "share_links"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS share_links (
            token TEXT PRIMARY KEY,
            debate_id TEXT NOT NULL UNIQUE,
            visibility TEXT NOT NULL DEFAULT 'private',
            owner_id TEXT,
            org_id TEXT,
            created_at REAL NOT NULL,
            expires_at REAL,
            allow_comments INTEGER DEFAULT 0,
            allow_forking INTEGER DEFAULT 0,
            view_count INTEGER DEFAULT 0,
            last_viewed_at REAL
        );

        CREATE INDEX IF NOT EXISTS idx_share_links_debate
        ON share_links(debate_id);

        CREATE INDEX IF NOT EXISTS idx_share_links_expires
        ON share_links(expires_at)
        WHERE expires_at IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_share_links_owner
        ON share_links(owner_id);

        CREATE INDEX IF NOT EXISTS idx_share_links_org
        ON share_links(org_id, visibility);
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        cleanup_interval: int = 300,
        **kwargs,
    ):
        """
        Initialize the share link store.

        Args:
            db_path: Path to SQLite database file
            cleanup_interval: Seconds between automatic TTL cleanups (default: 5 min)
        """
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()
        super().__init__(db_path, **kwargs)
        logger.info(f"ShareLinkStore initialized: {db_path}")

    def _post_init(self) -> None:
        """Run cleanup on startup."""
        self.cleanup_expired()

    def save(self, settings: "ShareSettings") -> None:
        """
        Save or update sharing settings.

        Uses UPSERT semantics - creates new record or updates existing
        based on debate_id uniqueness constraint.

        Args:
            settings: ShareSettings object to persist
        """
        with self.connection() as conn:
            # Generate token if needed and not already set
            token = settings.share_token
            if token is None:
                token = self._generate_token()

            conn.execute(
                """
                INSERT INTO share_links (
                    token, debate_id, visibility, owner_id, org_id,
                    created_at, expires_at, allow_comments, allow_forking,
                    view_count, last_viewed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(debate_id) DO UPDATE SET
                    token = COALESCE(excluded.token, token),
                    visibility = excluded.visibility,
                    owner_id = COALESCE(excluded.owner_id, owner_id),
                    org_id = COALESCE(excluded.org_id, org_id),
                    expires_at = excluded.expires_at,
                    allow_comments = excluded.allow_comments,
                    allow_forking = excluded.allow_forking
                """,
                (
                    token,
                    settings.debate_id,
                    (
                        settings.visibility.value
                        if hasattr(settings.visibility, "value")
                        else settings.visibility
                    ),
                    settings.owner_id,
                    settings.org_id,
                    settings.created_at,
                    settings.expires_at,
                    int(settings.allow_comments),
                    int(settings.allow_forking),
                    settings.view_count,
                    None,
                ),
            )

        self._maybe_cleanup()
        logger.debug(f"Saved share settings for debate {settings.debate_id}")

    def get(self, debate_id: str) -> Optional["ShareSettings"]:
        """
        Get sharing settings by debate ID.

        Args:
            debate_id: The debate identifier

        Returns:
            ShareSettings if found and not expired, None otherwise
        """
        row = self.fetch_one(
            """
            SELECT token, debate_id, visibility, owner_id, org_id,
                   created_at, expires_at, allow_comments, allow_forking,
                   view_count, last_viewed_at
            FROM share_links
            WHERE debate_id = ?
            """,
            (debate_id,),
        )

        if not row:
            return None

        return self._row_to_settings(row)

    def get_by_token(self, token: str) -> Optional["ShareSettings"]:
        """
        Get sharing settings by share token.

        Args:
            token: The share token

        Returns:
            ShareSettings if found and not expired, None otherwise
        """
        row = self.fetch_one(
            """
            SELECT token, debate_id, visibility, owner_id, org_id,
                   created_at, expires_at, allow_comments, allow_forking,
                   view_count, last_viewed_at
            FROM share_links
            WHERE token = ?
            """,
            (token,),
        )

        if not row:
            return None

        return self._row_to_settings(row)

    def delete(self, debate_id: str) -> bool:
        """
        Delete sharing settings for a debate.

        Args:
            debate_id: The debate identifier

        Returns:
            True if a record was deleted
        """
        with self.connection() as conn:
            cursor = conn.execute(
                "DELETE FROM share_links WHERE debate_id = ?",
                (debate_id,),
            )
            deleted = cursor.rowcount > 0

        if deleted:
            logger.info(f"Deleted share settings for debate {debate_id}")

        return deleted

    def revoke_token(self, debate_id: str) -> bool:
        """
        Revoke (nullify) the share token for a debate.

        The record remains but the token is cleared, invalidating
        any existing share links.

        Args:
            debate_id: The debate identifier

        Returns:
            True if a token was revoked
        """
        with self.connection() as conn:
            cursor = conn.execute(
                """
                UPDATE share_links
                SET token = NULL
                WHERE debate_id = ? AND token IS NOT NULL
                """,
                (debate_id,),
            )
            revoked = cursor.rowcount > 0

        if revoked:
            logger.info(f"Revoked share token for debate {debate_id}")

        return revoked

    def increment_view_count(self, debate_id: str) -> None:
        """
        Atomically increment the view count for a shared debate.

        Also updates last_viewed_at timestamp.

        Args:
            debate_id: The debate identifier
        """
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE share_links
                SET view_count = view_count + 1,
                    last_viewed_at = ?
                WHERE debate_id = ?
                """,
                (time.time(), debate_id),
            )

    def cleanup_expired(self) -> int:
        """
        Remove expired share links from the database.

        Returns:
            Number of expired records deleted
        """
        with self.connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM share_links
                WHERE expires_at IS NOT NULL AND expires_at < ?
                """,
                (time.time(),),
            )
            removed = cursor.rowcount

        self._last_cleanup = time.time()

        if removed > 0:
            logger.info(f"ShareLinkStore cleanup: removed {removed} expired links")

        return removed

    def _maybe_cleanup(self) -> None:
        """Run cleanup if cleanup_interval has passed."""
        now = time.time()
        if now - self._last_cleanup > self._cleanup_interval:
            self.cleanup_expired()

    def _generate_token(self) -> str:
        """Generate a secure share token."""
        return secrets.token_urlsafe(16)

    def _row_to_settings(self, row: tuple) -> "ShareSettings":
        """Convert a database row to ShareSettings object."""
        # Import here to avoid circular dependency
        from aragora.server.handlers.social.sharing import DebateVisibility, ShareSettings

        return ShareSettings(
            debate_id=row[1],
            visibility=DebateVisibility(row[2]),
            share_token=row[0],
            owner_id=row[3],
            org_id=row[4],
            created_at=row[5],
            expires_at=row[6],
            allow_comments=bool(row[7]),
            allow_forking=bool(row[8]),
            view_count=row[9] or 0,
        )

    def get_stats(self) -> dict:
        """
        Get statistics about share links.

        Returns:
            Dict with counts by visibility, expired count, etc.
        """
        stats = {}

        # Total count
        row = self.fetch_one("SELECT COUNT(*) FROM share_links")
        stats["total"] = row[0] if row else 0

        # By visibility
        rows = self.fetch_all("SELECT visibility, COUNT(*) FROM share_links GROUP BY visibility")
        stats["by_visibility"] = {row[0]: row[1] for row in rows}

        # With active tokens
        row = self.fetch_one("SELECT COUNT(*) FROM share_links WHERE token IS NOT NULL")
        stats["with_tokens"] = row[0] if row else 0

        # Expired (but not yet cleaned up)
        row = self.fetch_one(
            """
            SELECT COUNT(*) FROM share_links
            WHERE expires_at IS NOT NULL AND expires_at < ?
            """,
            (time.time(),),
        )
        stats["expired"] = row[0] if row else 0

        # Total views
        row = self.fetch_one("SELECT SUM(view_count) FROM share_links")
        stats["total_views"] = row[0] or 0 if row else 0

        return stats


__all__ = ["ShareLinkStore"]
