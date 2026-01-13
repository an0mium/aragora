"""
Debate Repository for persisting debate artifacts.

Provides data access for debates with slug-based permalinks and search.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseRepository, EntityNotFoundError

logger = logging.getLogger(__name__)


@dataclass
class DebateEntity:
    """
    Debate entity representing a stored debate.

    This is the domain object - independent of database schema.
    """

    id: str
    slug: str
    task: str
    agents: List[str]
    artifact_json: str
    consensus_reached: bool = False
    confidence: float = 0.0
    created_at: Optional[datetime] = None
    view_count: int = 0
    audio_path: Optional[str] = None
    audio_generated_at: Optional[datetime] = None
    audio_duration_seconds: Optional[int] = None

    def to_artifact(self) -> Dict[str, Any]:
        """Parse artifact_json and return the full artifact dict."""
        return json.loads(self.artifact_json)

    @classmethod
    def from_artifact(
        cls,
        artifact: Dict[str, Any],
        slug: str,
        debate_id: Optional[str] = None,
    ) -> "DebateEntity":
        """
        Create entity from artifact dict.

        Args:
            artifact: Full debate artifact dictionary.
            slug: URL slug for the debate.
            debate_id: Optional ID (auto-generated if not provided).

        Returns:
            DebateEntity instance.
        """
        return cls(
            id=debate_id or artifact.get("debate_id", slug),
            slug=slug,
            task=artifact.get("task", ""),
            agents=artifact.get("agents", []),
            artifact_json=json.dumps(artifact),
            consensus_reached=artifact.get("consensus_reached", False),
            confidence=artifact.get("confidence", 0.0),
        )


@dataclass
class DebateMetadata:
    """Summary metadata for a stored debate."""

    slug: str
    debate_id: str
    task: str
    agents: List[str]
    consensus_reached: bool
    confidence: float
    created_at: datetime
    view_count: int = 0


class DebateRepository(BaseRepository[DebateEntity]):
    """
    Repository for debate persistence.

    Provides CRUD operations plus search and slug management.

    Usage:
        repo = DebateRepository()
        entity = DebateEntity(...)
        repo.save(entity)

        # Get by slug
        debate = repo.get_by_slug("rate-limiter-2024-01-01")

        # Search
        debates = repo.search("rate limiter", limit=10)

        # List recent
        recent = repo.list_recent(limit=20)
    """

    # Words to exclude from slug generation
    STOP_WORDS = frozenset(
        {
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
    )

    def __init__(
        self,
        db_path: str | Path = "aragora_debates.db",
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize the debate repository.

        Args:
            db_path: Path to the SQLite database file.
            timeout: Connection timeout in seconds.
        """
        super().__init__(db_path, timeout)

    @property
    def _table_name(self) -> str:
        return "debates"

    @property
    def _entity_name(self) -> str:
        return "Debate"

    def _ensure_schema(self) -> None:
        """Create database schema if needed."""
        with self._connection() as conn:
            conn.execute(
                """
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
                    audio_duration_seconds INTEGER
                )
            """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_slug ON debates(slug)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON debates(created_at)")
            conn.commit()

    def _to_entity(self, row: Any) -> DebateEntity:
        """Convert database row to entity."""
        created_at = row["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        audio_generated_at = row["audio_generated_at"]
        if isinstance(audio_generated_at, str):
            audio_generated_at = datetime.fromisoformat(audio_generated_at)

        return DebateEntity(
            id=row["id"],
            slug=row["slug"],
            task=row["task"],
            agents=json.loads(row["agents"]),
            artifact_json=row["artifact_json"],
            consensus_reached=bool(row["consensus_reached"]),
            confidence=row["confidence"] or 0.0,
            created_at=created_at,
            view_count=row["view_count"] or 0,
            audio_path=row["audio_path"],
            audio_generated_at=audio_generated_at,
            audio_duration_seconds=row["audio_duration_seconds"],
        )

    def _from_entity(self, entity: DebateEntity) -> Dict[str, Any]:
        """Convert entity to database columns."""
        return {
            "id": entity.id,
            "slug": entity.slug,
            "task": entity.task,
            "agents": json.dumps(entity.agents),
            "artifact_json": entity.artifact_json,
            "consensus_reached": entity.consensus_reached,
            "confidence": entity.confidence,
            "created_at": (
                entity.created_at.isoformat() if entity.created_at else datetime.now().isoformat()
            ),
            "view_count": entity.view_count,
            "audio_path": entity.audio_path,
            "audio_generated_at": (
                entity.audio_generated_at.isoformat() if entity.audio_generated_at else None
            ),
            "audio_duration_seconds": entity.audio_duration_seconds,
        }

    def get_by_slug(self, slug: str) -> Optional[DebateEntity]:
        """
        Get a debate by its URL slug.

        Args:
            slug: URL-friendly slug (e.g., "rate-limiter-2024-01-01").

        Returns:
            DebateEntity or None if not found.
        """
        row = self._fetch_one(
            "SELECT * FROM debates WHERE slug = ?",
            (slug,),
        )
        return self._to_entity(row) if row else None

    def get_by_slug_or_raise(self, slug: str) -> DebateEntity:
        """
        Get a debate by slug or raise EntityNotFoundError.

        Args:
            slug: URL-friendly slug.

        Returns:
            DebateEntity.

        Raises:
            EntityNotFoundError: If debate not found.
        """
        entity = self.get_by_slug(slug)
        if entity is None:
            raise EntityNotFoundError("Debate", slug)
        return entity

    def increment_view_count(self, slug: str) -> int:
        """
        Increment the view count for a debate.

        Args:
            slug: Debate slug.

        Returns:
            New view count.
        """
        with self._transaction() as conn:
            conn.execute(
                "UPDATE debates SET view_count = view_count + 1 WHERE slug = ?",
                (slug,),
            )
            cursor = conn.execute(
                "SELECT view_count FROM debates WHERE slug = ?",
                (slug,),
            )
            row = cursor.fetchone()
            return row[0] if row else 0

    def list_recent(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> List[DebateMetadata]:
        """
        List recent debates ordered by creation date.

        Args:
            limit: Maximum number of debates to return.
            offset: Number of debates to skip.

        Returns:
            List of DebateMetadata summaries.
        """
        rows = self._fetch_all(
            """
            SELECT slug, id, task, agents, consensus_reached, confidence,
                   created_at, view_count
            FROM debates
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )

        results = []
        for row in rows:
            created_at = row["created_at"]
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)

            results.append(
                DebateMetadata(
                    slug=row["slug"],
                    debate_id=row["id"],
                    task=row["task"],
                    agents=json.loads(row["agents"]),
                    consensus_reached=bool(row["consensus_reached"]),
                    confidence=row["confidence"] or 0.0,
                    created_at=created_at,
                    view_count=row["view_count"] or 0,
                )
            )
        return results

    def search(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0,
    ) -> List[DebateMetadata]:
        """
        Search debates by task description.

        Args:
            query: Search query (searches task field).
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            List of matching DebateMetadata.
        """
        # Escape LIKE special characters
        escaped_query = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped_query}%"

        rows = self._fetch_all(
            """
            SELECT slug, id, task, agents, consensus_reached, confidence,
                   created_at, view_count
            FROM debates
            WHERE task LIKE ? ESCAPE '\\'
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (pattern, limit, offset),
        )

        results = []
        for row in rows:
            created_at = row["created_at"]
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)

            results.append(
                DebateMetadata(
                    slug=row["slug"],
                    debate_id=row["id"],
                    task=row["task"],
                    agents=json.loads(row["agents"]),
                    consensus_reached=bool(row["consensus_reached"]),
                    confidence=row["confidence"] or 0.0,
                    created_at=created_at,
                    view_count=row["view_count"] or 0,
                )
            )
        return results

    def generate_slug(self, task: str) -> str:
        """
        Generate URL-friendly slug from task description.

        Takes key words from the task, combines with date, and handles
        collisions by appending a counter.

        Examples:
            "Design a rate limiter" -> "rate-limiter-2024-01-01"
            "Design a rate limiter" (second) -> "rate-limiter-2024-01-01-2"

        Args:
            task: Task description.

        Returns:
            Unique URL slug.
        """
        # Extract words, remove punctuation
        words = re.sub(r"[^\w\s]", "", task.lower()).split()

        # Filter stop words and take first 4 meaningful words
        key_words = [w for w in words if w not in self.STOP_WORDS][:4]
        base = "-".join(key_words) if key_words else "debate"

        # Add date
        date = datetime.now().strftime("%Y-%m-%d")
        slug = f"{base}-{date}"

        # Handle collisions
        existing = self.get_by_slug(slug)
        if not existing:
            return slug

        # Find next available number
        counter = 2
        while True:
            numbered_slug = f"{slug}-{counter}"
            if not self.get_by_slug(numbered_slug):
                return numbered_slug
            counter += 1

    def save_with_slug(
        self,
        artifact: Dict[str, Any],
        slug: Optional[str] = None,
    ) -> str:
        """
        Save a debate artifact with auto-generated slug.

        Args:
            artifact: Full debate artifact dictionary.
            slug: Optional slug (auto-generated if not provided).

        Returns:
            The slug for the saved debate.
        """
        if slug is None:
            slug = self.generate_slug(artifact.get("task", "debate"))

        entity = DebateEntity.from_artifact(artifact, slug)
        self.save(entity)
        return slug

    def update_audio_info(
        self,
        slug: str,
        audio_path: str,
        duration_seconds: int,
    ) -> bool:
        """
        Update audio information for a debate.

        Args:
            slug: Debate slug.
            audio_path: Path to audio file.
            duration_seconds: Audio duration in seconds.

        Returns:
            True if updated, False if debate not found.
        """
        with self._transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE debates
                SET audio_path = ?,
                    audio_generated_at = ?,
                    audio_duration_seconds = ?
                WHERE slug = ?
                """,
                (
                    audio_path,
                    datetime.now().isoformat(),
                    duration_seconds,
                    slug,
                ),
            )
            return cursor.rowcount > 0

    def delete_by_slug(self, slug: str) -> bool:
        """
        Delete a debate by slug.

        Args:
            slug: Debate slug.

        Returns:
            True if deleted, False if not found.
        """
        with self._transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM debates WHERE slug = ?",
                (slug,),
            )
            return cursor.rowcount > 0
