"""
SQLite-backed debate storage with permalink generation.

Provides persistent storage for debate artifacts with human-readable
URL slugs for sharing (e.g., rate-limiter-2026-01-01).
"""

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.export.artifact import DebateArtifact


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

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
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
        conn.commit()
        conn.close()

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

        # Handle collisions
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT COUNT(*) FROM debates WHERE slug LIKE ?",
            (f"{slug}%",)
        )
        count = cursor.fetchone()[0]
        conn.close()

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

        conn = sqlite3.connect(self.db_path)
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
        conn.close()

        return slug

    def save_dict(self, debate_data: dict) -> str:
        """
        Save debate data directly (without DebateArtifact).

        Useful for saving streaming debates before full artifact is built.
        """
        slug = self.generate_slug(debate_data.get("task", "debate"))
        debate_id = debate_data.get("id", slug)

        conn = sqlite3.connect(self.db_path)
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
        conn.close()

        return slug

    def get_by_slug(self, slug: str) -> Optional[dict]:
        """
        Get debate by slug, incrementing view count.

        Returns:
            Debate artifact dict or None if not found
        """
        conn = sqlite3.connect(self.db_path)
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

        conn.close()
        return json.loads(row[0]) if row else None

    def get_by_id(self, debate_id: str) -> Optional[dict]:
        """Get debate by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT artifact_json FROM debates WHERE id = ?",
            (debate_id,)
        )
        row = cursor.fetchone()
        conn.close()
        return json.loads(row[0]) if row else None

    def list_recent(self, limit: int = 20) -> list[DebateMetadata]:
        """
        List recent debates.

        Returns:
            List of DebateMetadata ordered by creation date (newest first)
        """
        conn = sqlite3.connect(self.db_path)
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

        conn.close()
        return results

    def delete(self, slug: str) -> bool:
        """Delete a debate by slug."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "DELETE FROM debates WHERE slug = ?",
            (slug,)
        )
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted
