"""
SQLite-backed debate storage with permalink generation.

Provides persistent storage for debate artifacts with human-readable
URL slugs for sharing (e.g., rate-limiter-2026-01-01).
"""

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from aragora.storage.base_store import SQLiteStore

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aragora.export.artifact import DebateArtifact


def _validate_sql_identifier(name: str, max_length: int = 64) -> bool:
    """Validate SQL identifier to prevent injection.

    Only allows alphanumeric characters and underscores.
    Must start with a letter or underscore.
    Limited to max_length characters (default 64, SQLite limit is 255).

    Args:
        name: Identifier to validate
        max_length: Maximum allowed length (default 64)

    Returns:
        True if valid identifier, False otherwise
    """
    if not name or len(name) > max_length:
        return False
    return bool(re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name))


# Import from centralized location (defined here for backwards compatibility)
from aragora.utils.sql_helpers import _escape_like_pattern
from aragora.storage.schema import DB_TIMEOUT


@dataclass
class DebateMetadata:
    """Summary metadata for a stored debate."""

    slug: str
    debate_id: str
    task: str
    agents: list[str]
    consensus_reached: bool
    confidence: float
    created_at: datetime
    view_count: int = 0
    is_public: bool = False  # If True, artifacts accessible without auth


class DebateStorage(SQLiteStore):
    """
    Debate persistence with shareable permalinks.

    Stores complete debate artifacts in SQLite with auto-generated
    URL-friendly slugs based on the task description.

    Usage:
        storage = DebateStorage("aragora_debates.db")
        slug = storage.save(artifact)
        # -> "rate-limiter-2026-01-01"

        debate = storage.get_by_slug("rate-limiter-2026-01-01")
    """

    SCHEMA_NAME = "debate_storage"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS debates (
            id TEXT PRIMARY KEY,
            slug TEXT UNIQUE NOT NULL,
            task TEXT NOT NULL,
            agents TEXT NOT NULL,
            artifact_json TEXT NOT NULL,
            consensus_reached BOOLEAN,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            view_count INTEGER DEFAULT 0,
            audio_path TEXT,
            audio_generated_at TIMESTAMP,
            audio_duration_seconds INTEGER,
            org_id TEXT,
            is_public INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_slug ON debates(slug);
        CREATE INDEX IF NOT EXISTS idx_created ON debates(created_at);
        CREATE INDEX IF NOT EXISTS idx_task ON debates(task);
        CREATE INDEX IF NOT EXISTS idx_debates_org ON debates(org_id, created_at);
    """

    # Words to exclude from slug generation
    STOP_WORDS = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "for",
        "to",
        "of",
        "and",
        "or",
        "in",
        "on",
        "at",
        "by",
        "with",
        "that",
        "this",
        "it",
        "as",
        "from",
        "how",
        "what",
        "design",
        "implement",
        "create",
        "build",
        "make",
    }

    def __init__(self, db_path: str = "aragora_debates.db"):
        super().__init__(db_path)

    # Known table names for this storage class (defense-in-depth)
    _KNOWN_TABLES = frozenset({"debates"})

    def _safe_add_column(
        self, conn: sqlite3.Connection, table: str, column: str, col_type: str
    ) -> bool:
        """
        Safely add a column if it doesn't exist.

        Uses multiple layers of defense against SQL injection:
        1. Table name must be in _KNOWN_TABLES whitelist
        2. Table and column names must match identifier regex
        3. Column type must be in type whitelist

        Args:
            conn: Database connection
            table: Table name (must be in _KNOWN_TABLES)
            column: Column name to add
            col_type: SQLite column type (must be in whitelist)

        Returns:
            True if column was added, False if it already existed or validation failed
        """
        # Defense layer 1: Table must be in known tables whitelist
        if table not in self._KNOWN_TABLES:
            logger.warning("Table not in whitelist: %s (allowed: %s)", table, self._KNOWN_TABLES)
            return False

        # Defense layer 2: Validate identifier patterns
        if not _validate_sql_identifier(table) or not _validate_sql_identifier(column):
            logger.warning("Invalid SQL identifier: table=%s, column=%s", table, column)
            return False

        # Defense layer 3: Validate col_type against whitelist
        valid_types = frozenset({"TEXT", "INTEGER", "REAL", "BLOB", "TIMESTAMP"})
        if col_type not in valid_types:
            logger.warning("Invalid column type: %s (allowed: %s)", col_type, valid_types)
            return False

        # Safe to execute - all inputs validated
        cursor = conn.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            return True
        return False

    def generate_slug(self, task: str) -> str:
        """
        Generate URL-friendly slug from task description.

        Takes key words from the task, combines with date, and handles
        collisions by appending a counter.

        Examples:
            "Design a rate limiter" -> "rate-limiter-2026-01-01"
            "Design a rate limiter" (second) -> "rate-limiter-2026-01-01-2"
        """
        # Extract words, remove punctuation
        words = re.sub(r"[^\w\s]", "", task.lower()).split()

        # Filter stop words and take first 4 meaningful words
        key_words = [w for w in words if w not in self.STOP_WORDS][:4]
        base = "-".join(key_words) if key_words else "debate"

        # Add date
        date = datetime.now().strftime("%Y-%m-%d")
        slug = f"{base}-{date}"

        # Handle collisions using GLOB for precise matching
        # Matches: slug itself OR slug-N pattern (where N is digits)
        with self.connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM debates WHERE slug = ? OR slug GLOB ?",
                (slug, f"{slug}-[0-9]*"),
            )
            row = cursor.fetchone()
            count = row[0] if row else 0

        return f"{slug}-{count + 1}" if count > 0 else slug

    def save(self, artifact: "DebateArtifact") -> str:
        """
        Save artifact and return permalink slug.

        Args:
            artifact: DebateArtifact to store

        Returns:
            Generated slug for the debate
        """
        slug = self.generate_slug(artifact.task)

        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO debates (
                    id, slug, task, agents, artifact_json,
                    consensus_reached, confidence
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    artifact.artifact_id,
                    slug,
                    artifact.task,
                    json.dumps(artifact.agents),
                    artifact.to_json(),
                    artifact.consensus_proof.reached if artifact.consensus_proof else False,
                    artifact.consensus_proof.confidence if artifact.consensus_proof else 0,
                ),
            )
            conn.commit()

        return slug

    def update_audio(
        self,
        debate_id: str,
        audio_path: str,
        duration_seconds: Optional[int] = None,
    ) -> bool:
        """
        Update audio information for a debate.

        Args:
            debate_id: Debate identifier
            audio_path: Path to the audio file
            duration_seconds: Audio duration in seconds

        Returns:
            True if updated, False if debate not found
        """
        with self.connection() as conn:
            cursor = conn.execute(
                """
                UPDATE debates
                SET audio_path = ?,
                    audio_generated_at = ?,
                    audio_duration_seconds = ?
                WHERE id = ?
                """,
                (audio_path, datetime.now().isoformat(), duration_seconds, debate_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_audio_info(self, debate_id: str) -> Optional[dict]:
        """
        Get audio information for a debate.

        Args:
            debate_id: Debate identifier

        Returns:
            Dict with audio_path, audio_generated_at, audio_duration_seconds
            or None if no audio exists
        """
        with self.connection() as conn:
            cursor = conn.execute(
                """
                SELECT audio_path, audio_generated_at, audio_duration_seconds
                FROM debates
                WHERE id = ?
                """,
                (debate_id,),
            )
            row = cursor.fetchone()

        if not row or not row[0]:
            return None

        return {
            "audio_path": row[0],
            "audio_generated_at": row[1],
            "audio_duration_seconds": row[2],
        }

    def save_dict(self, debate_data: dict, org_id: Optional[str] = None) -> str:
        """
        Save debate data directly (without DebateArtifact).

        Useful for saving streaming debates before full artifact is built.

        Args:
            debate_data: Debate data dict
            org_id: Organization ID for multi-tenancy scoping (optional)

        Returns:
            Generated slug for the debate
        """
        slug = self.generate_slug(debate_data.get("task", "debate"))
        debate_id = debate_data.get("id", slug)

        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO debates (
                    id, slug, task, agents, artifact_json,
                    consensus_reached, confidence, org_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    debate_id,
                    slug,
                    debate_data.get("task", ""),
                    json.dumps(debate_data.get("agents", [])),
                    json.dumps(debate_data),
                    debate_data.get("consensus_reached", False),
                    debate_data.get("confidence", 0),
                    org_id,
                ),
            )
            conn.commit()

        return slug

    def get_by_slug(
        self, slug: str, org_id: Optional[str] = None, verify_ownership: bool = False
    ) -> Optional[dict]:
        """
        Get debate by slug, incrementing view count.

        Args:
            slug: Debate slug
            org_id: Organization ID for ownership verification
            verify_ownership: If True and org_id provided, only return if debate
                              belongs to this org. If False, returns any matching debate.

        Returns:
            Debate artifact dict or None if not found (or ownership check fails)
        """
        # Validate slug to prevent abuse (DoS via extremely long slugs)
        if not slug or len(slug) > 500:
            return None

        with self.connection() as conn:
            if verify_ownership and org_id:
                cursor = conn.execute(
                    "SELECT artifact_json FROM debates WHERE slug = ? AND org_id = ?",
                    (slug, org_id),
                )
            else:
                cursor = conn.execute("SELECT artifact_json FROM debates WHERE slug = ?", (slug,))
            row = cursor.fetchone()

            if row:
                conn.execute(
                    "UPDATE debates SET view_count = view_count + 1 WHERE slug = ?", (slug,)
                )
                conn.commit()

        return json.loads(row[0]) if row else None

    def get_by_id(
        self, debate_id: str, org_id: Optional[str] = None, verify_ownership: bool = False
    ) -> Optional[dict]:
        """
        Get debate by ID.

        Args:
            debate_id: Debate ID
            org_id: Organization ID for ownership verification
            verify_ownership: If True and org_id provided, only return if debate
                              belongs to this org.

        Returns:
            Debate artifact dict or None
        """
        with self.connection() as conn:
            if verify_ownership and org_id:
                cursor = conn.execute(
                    "SELECT artifact_json FROM debates WHERE id = ? AND org_id = ?",
                    (debate_id, org_id),
                )
            else:
                cursor = conn.execute(
                    "SELECT artifact_json FROM debates WHERE id = ?", (debate_id,)
                )
            row = cursor.fetchone()
        return json.loads(row[0]) if row else None

    def list_recent(self, limit: int = 20, org_id: Optional[str] = None) -> list[DebateMetadata]:
        """
        List recent debates, optionally filtered by organization.

        Args:
            limit: Maximum number of debates to return
            org_id: If provided, only return debates for this organization.
                    If None, returns all debates (for backwards compatibility).

        Returns:
            List of DebateMetadata ordered by creation date (newest first)
        """
        with self.connection() as conn:
            if org_id:
                cursor = conn.execute(
                    """
                    SELECT slug, id, task, agents, consensus_reached,
                           confidence, created_at, view_count, is_public
                    FROM debates
                    WHERE org_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """,
                    (org_id, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT slug, id, task, agents, consensus_reached,
                           confidence, created_at, view_count, is_public
                    FROM debates
                    ORDER BY created_at DESC
                    LIMIT ?
                """,
                    (limit,),
                )

            results = []
            for row in cursor.fetchall():
                try:
                    created = datetime.fromisoformat(row[6])
                except (ValueError, TypeError):
                    created = datetime.now()

                results.append(
                    DebateMetadata(
                        slug=row[0],
                        debate_id=row[1],
                        task=row[2],
                        agents=json.loads(row[3]) if row[3] else [],
                        consensus_reached=bool(row[4]),
                        confidence=row[5] or 0,
                        created_at=created,
                        view_count=row[7] or 0,
                        is_public=bool(row[8]) if len(row) > 8 else False,
                    )
                )

        return results

    def search(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0,
        org_id: Optional[str] = None,
    ) -> tuple[list[DebateMetadata], int]:
        """
        Search debates by task/slug using efficient SQL LIKE queries.

        Args:
            query: Search term to match against task and slug
            limit: Maximum number of results to return
            offset: Number of results to skip (for pagination)
            org_id: If provided, only search within this organization's debates

        Returns:
            Tuple of (matching debates, total count)
        """
        # Escape LIKE special characters for safe SQL
        safe_query = _escape_like_pattern(query)
        like_pattern = f"%{safe_query}%"

        with self.connection() as conn:
            # Get total count first
            if org_id:
                count_cursor = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM debates
                    WHERE org_id = ?
                      AND (task LIKE ? ESCAPE '\\' OR slug LIKE ? ESCAPE '\\')
                """,
                    (org_id, like_pattern, like_pattern),
                )
            else:
                count_cursor = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM debates
                    WHERE task LIKE ? ESCAPE '\\'
                       OR slug LIKE ? ESCAPE '\\'
                """,
                    (like_pattern, like_pattern),
                )

            total = count_cursor.fetchone()[0]

            # Get paginated results
            if org_id:
                cursor = conn.execute(
                    """
                    SELECT slug, id, task, agents, consensus_reached,
                           confidence, created_at, view_count, is_public
                    FROM debates
                    WHERE org_id = ?
                      AND (task LIKE ? ESCAPE '\\' OR slug LIKE ? ESCAPE '\\')
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """,
                    (org_id, like_pattern, like_pattern, limit, offset),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT slug, id, task, agents, consensus_reached,
                           confidence, created_at, view_count, is_public
                    FROM debates
                    WHERE task LIKE ? ESCAPE '\\'
                       OR slug LIKE ? ESCAPE '\\'
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """,
                    (like_pattern, like_pattern, limit, offset),
                )

            results = []
            for row in cursor.fetchall():
                try:
                    created = datetime.fromisoformat(row[6])
                except (ValueError, TypeError):
                    created = datetime.now()

                results.append(
                    DebateMetadata(
                        slug=row[0],
                        debate_id=row[1],
                        task=row[2],
                        agents=json.loads(row[3]) if row[3] else [],
                        consensus_reached=bool(row[4]),
                        confidence=row[5] or 0,
                        created_at=created,
                        view_count=row[7] or 0,
                        is_public=bool(row[8]) if len(row) > 8 else False,
                    )
                )

        return results, total

    def delete(
        self, slug: str, org_id: Optional[str] = None, require_ownership: bool = False
    ) -> bool:
        """
        Delete a debate by slug.

        Args:
            slug: Debate slug
            org_id: Organization ID for ownership verification
            require_ownership: If True, only delete if debate belongs to org_id

        Returns:
            True if deleted, False if not found or ownership check failed
        """
        with self.connection() as conn:
            if require_ownership and org_id:
                cursor = conn.execute(
                    "DELETE FROM debates WHERE slug = ? AND org_id = ?", (slug, org_id)
                )
            else:
                cursor = conn.execute("DELETE FROM debates WHERE slug = ?", (slug,))
            deleted = cursor.rowcount > 0
            conn.commit()
        return deleted

    def is_public(self, debate_id: str) -> bool:
        """
        Check if a debate is publicly accessible.

        Args:
            debate_id: Debate ID to check

        Returns:
            True if debate exists and is_public=True, False otherwise
        """
        with self.connection() as conn:
            cursor = conn.execute("SELECT is_public FROM debates WHERE id = ?", (debate_id,))
            row = cursor.fetchone()
        return bool(row and row[0])

    def set_public(self, debate_id: str, is_public: bool, org_id: Optional[str] = None) -> bool:
        """
        Set debate public/private status.

        Args:
            debate_id: Debate ID
            is_public: True for public access, False for auth required
            org_id: If provided, only update if debate belongs to this org

        Returns:
            True if updated, False if not found or ownership check failed
        """
        with self.connection() as conn:
            if org_id:
                cursor = conn.execute(
                    "UPDATE debates SET is_public = ? WHERE id = ? AND org_id = ?",
                    (is_public, debate_id, org_id),
                )
            else:
                cursor = conn.execute(
                    "UPDATE debates SET is_public = ? WHERE id = ?", (is_public, debate_id)
                )
            updated = cursor.rowcount > 0
            conn.commit()
        return updated

    def get(self, debate_id: str) -> Optional[dict]:
        """
        Get debate by ID (alias for get_by_id for interface compatibility).

        Args:
            debate_id: Debate ID

        Returns:
            Debate artifact dict or None
        """
        return self.get_by_id(debate_id)

    def get_debate(self, debate_id: str) -> Optional[dict]:
        """
        Get debate by ID (handler-compatible alias).

        Args:
            debate_id: Debate ID

        Returns:
            Debate artifact dict or None
        """
        return self.get_by_id(debate_id)

    def get_debate_by_slug(self, slug: str) -> Optional[dict]:
        """
        Get debate by slug (handler-compatible alias).

        Args:
            slug: URL-friendly debate slug

        Returns:
            Debate artifact dict or None
        """
        return self.get_by_slug(slug)

    def list_debates(self, limit: int = 20, org_id: Optional[str] = None) -> list[DebateMetadata]:
        """
        List debates (handler-compatible alias for list_recent).

        Args:
            limit: Maximum number of debates to return
            org_id: Filter by organization

        Returns:
            List of DebateMetadata objects
        """
        return self.list_recent(limit=limit, org_id=org_id)


# Global storage instance
_debate_storage: Optional[DebateStorage] = None


def get_debates_db() -> Optional[DebateStorage]:
    """
    Get the global DebateStorage instance.

    Returns a singleton DebateStorage backed by SQLite. The storage
    provides a `get(debate_id)` method for fetching debates by ID.

    Returns:
        DebateStorage instance, or None if initialization fails
    """
    global _debate_storage
    if _debate_storage is None:
        try:
            from aragora.config.legacy import get_db_path

            db_path = get_db_path("aragora_debates.db")
            _debate_storage = DebateStorage(str(db_path))
            logger.info(f"Initialized DebateStorage: {db_path}")
        except Exception as e:
            logger.warning(f"Failed to initialize DebateStorage: {e}")
            return None
    return _debate_storage
