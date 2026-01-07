"""
SQLite-backed debate storage with permalink generation.

Provides persistent storage for debate artifacts with human-readable
URL slugs for sharing (e.g., rate-limiter-2026-01-01).
"""

import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Generator
import logging

from aragora.storage.schema import get_wal_connection, DB_TIMEOUT

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aragora.export.artifact import DebateArtifact


def _validate_sql_identifier(name: str) -> bool:
    """Validate SQL identifier to prevent injection.

    Only allows alphanumeric characters and underscores.
    Must start with a letter or underscore.
    """
    return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name))


def _escape_like_pattern(pattern: str) -> str:
    """Escape special characters for SQL LIKE patterns.

    LIKE metacharacters:
    - % matches any sequence of characters
    - _ matches any single character
    """
    # Escape backslash first (escape character itself)
    pattern = pattern.replace("\\", "\\\\")
    # Escape LIKE metacharacters
    pattern = pattern.replace("%", "\\%")
    pattern = pattern.replace("_", "\\_")
    return pattern


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


class DebateStorage:
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

    # Words to exclude from slug generation
    STOP_WORDS = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
        'for', 'to', 'of', 'and', 'or', 'in', 'on', 'at', 'by',
        'with', 'that', 'this', 'it', 'as', 'from', 'how', 'what',
        'design', 'implement', 'create', 'build', 'make',
    }

    def __init__(self, db_path: str = "aragora_debates.db"):
        self.db_path = Path(db_path)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection as a context manager.

        Uses WAL mode for better concurrency. Creates a new connection
        per operation for thread safety - SQLite connections cannot be
        safely shared across threads.
        """
        conn = get_wal_connection(self.db_path, timeout=DB_TIMEOUT)
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS debates (
                    id TEXT PRIMARY KEY,
                    slug TEXT UNIQUE NOT NULL,
                    task TEXT NOT NULL,
                    agents TEXT NOT NULL,
                    artifact_json TEXT NOT NULL,
                    consensus_reached BOOLEAN,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    view_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_slug ON debates(slug)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON debates(created_at)")

            # Add audio columns (migration for existing databases)
            self._safe_add_column(conn, "debates", "audio_path", "TEXT")
            self._safe_add_column(conn, "debates", "audio_generated_at", "TIMESTAMP")
            self._safe_add_column(conn, "debates", "audio_duration_seconds", "INTEGER")

            conn.commit()

    def _safe_add_column(
        self, conn: sqlite3.Connection, table: str, column: str, col_type: str
    ) -> bool:
        """
        Safely add a column if it doesn't exist.

        Args:
            conn: Database connection
            table: Table name
            column: Column name to add
            col_type: SQLite column type (must be in whitelist)

        Returns:
            True if column was added, False if it already existed or validation failed
        """
        # Validate identifiers to prevent SQL injection
        if not _validate_sql_identifier(table) or not _validate_sql_identifier(column):
            logger.warning("Invalid SQL identifier: table=%s, column=%s", table, column)
            return False

        # Validate col_type against whitelist
        valid_types = {"TEXT", "INTEGER", "REAL", "BLOB", "TIMESTAMP"}
        if col_type not in valid_types:
            logger.warning("Invalid column type: %s (allowed: %s)", col_type, valid_types)
            return False

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
        words = re.sub(r'[^\w\s]', '', task.lower()).split()

        # Filter stop words and take first 4 meaningful words
        key_words = [w for w in words if w not in self.STOP_WORDS][:4]
        base = '-'.join(key_words) if key_words else 'debate'

        # Add date
        date = datetime.now().strftime('%Y-%m-%d')
        slug = f"{base}-{date}"

        # Handle collisions using GLOB for precise matching
        # Matches: slug itself OR slug-N pattern (where N is digits)
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM debates WHERE slug = ? OR slug GLOB ?",
                (slug, f"{slug}-[0-9]*")
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

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO debates (
                    id, slug, task, agents, artifact_json,
                    consensus_reached, confidence
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                artifact.artifact_id,
                slug,
                artifact.task,
                json.dumps(artifact.agents),
                artifact.to_json(),
                artifact.consensus_proof.reached if artifact.consensus_proof else False,
                artifact.consensus_proof.confidence if artifact.consensus_proof else 0,
            ))
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
        with self._get_connection() as conn:
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
        with self._get_connection() as conn:
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

    def save_dict(self, debate_data: dict) -> str:
        """
        Save debate data directly (without DebateArtifact).

        Useful for saving streaming debates before full artifact is built.
        """
        slug = self.generate_slug(debate_data.get("task", "debate"))
        debate_id = debate_data.get("id", slug)

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO debates (
                    id, slug, task, agents, artifact_json,
                    consensus_reached, confidence
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                debate_id,
                slug,
                debate_data.get("task", ""),
                json.dumps(debate_data.get("agents", [])),
                json.dumps(debate_data),
                debate_data.get("consensus_reached", False),
                debate_data.get("confidence", 0),
            ))
            conn.commit()

        return slug

    def get_by_slug(self, slug: str) -> Optional[dict]:
        """
        Get debate by slug, incrementing view count.

        Returns:
            Debate artifact dict or None if not found
        """
        # Validate slug to prevent abuse (DoS via extremely long slugs)
        if not slug or len(slug) > 500:
            return None

        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT artifact_json FROM debates WHERE slug = ?",
                (slug,)
            )
            row = cursor.fetchone()

            if row:
                conn.execute(
                    "UPDATE debates SET view_count = view_count + 1 WHERE slug = ?",
                    (slug,)
                )
                conn.commit()

        return json.loads(row[0]) if row else None

    def get_by_id(self, debate_id: str) -> Optional[dict]:
        """Get debate by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT artifact_json FROM debates WHERE id = ?",
                (debate_id,)
            )
            row = cursor.fetchone()
        return json.loads(row[0]) if row else None

    def list_recent(self, limit: int = 20) -> list[DebateMetadata]:
        """
        List recent debates.

        Returns:
            List of DebateMetadata ordered by creation date (newest first)
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT slug, id, task, agents, consensus_reached,
                       confidence, created_at, view_count
                FROM debates
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))

            results = []
            for row in cursor.fetchall():
                try:
                    created = datetime.fromisoformat(row[6])
                except (ValueError, TypeError):
                    created = datetime.now()

                results.append(DebateMetadata(
                    slug=row[0],
                    debate_id=row[1],
                    task=row[2],
                    agents=json.loads(row[3]) if row[3] else [],
                    consensus_reached=bool(row[4]),
                    confidence=row[5] or 0,
                    created_at=created,
                    view_count=row[7] or 0,
                ))

        return results

    def delete(self, slug: str) -> bool:
        """Delete a debate by slug."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM debates WHERE slug = ?",
                (slug,)
            )
            deleted = cursor.rowcount > 0
            conn.commit()
        return deleted
