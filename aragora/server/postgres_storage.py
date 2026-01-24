"""
PostgreSQL-backed debate storage with permalink generation.

Provides persistent storage for debate artifacts with human-readable
URL slugs for sharing (e.g., rate-limiter-2026-01-01).
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from aragora.storage.postgres_store import PostgresStore

logger = logging.getLogger(__name__)

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
    is_public: bool = False


class PostgresDebateStorage(PostgresStore):
    """PostgreSQL-backed debate persistence with shareable permalinks.

    Stores complete debate artifacts with auto-generated URL-friendly slugs.
    """

    SCHEMA_NAME = "debate_storage"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS debates (
            id TEXT PRIMARY KEY,
            slug TEXT UNIQUE NOT NULL,
            task TEXT NOT NULL,
            agents JSONB NOT NULL,
            artifact_json JSONB NOT NULL,
            consensus_reached BOOLEAN DEFAULT FALSE,
            confidence REAL DEFAULT 0.0,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            view_count INTEGER DEFAULT 0,
            audio_path TEXT,
            audio_generated_at TIMESTAMPTZ,
            audio_duration_seconds INTEGER,
            org_id TEXT,
            is_public BOOLEAN DEFAULT FALSE,
            search_vector TSVECTOR
        );

        CREATE INDEX IF NOT EXISTS idx_debates_slug ON debates(slug);
        CREATE INDEX IF NOT EXISTS idx_debates_created ON debates(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_debates_task ON debates(task);
        CREATE INDEX IF NOT EXISTS idx_debates_org ON debates(org_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_debates_public ON debates(is_public);
        CREATE INDEX IF NOT EXISTS idx_debates_search ON debates USING GIN(search_vector);

        -- Function to update search vector
        CREATE OR REPLACE FUNCTION update_debate_search_vector()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.search_vector :=
                setweight(to_tsvector('english', COALESCE(NEW.task, '')), 'A') ||
                setweight(to_tsvector('english', COALESCE(NEW.slug, '')), 'B');
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

        -- Trigger to auto-update search vector
        DROP TRIGGER IF EXISTS debate_search_vector_trigger ON debates;
        CREATE TRIGGER debate_search_vector_trigger
            BEFORE INSERT OR UPDATE ON debates
            FOR EACH ROW
            EXECUTE FUNCTION update_debate_search_vector();
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

    # =========================================================================
    # Sync wrappers for compatibility
    # =========================================================================

    def generate_slug(self, task: str) -> str:
        """Generate URL-friendly slug (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.generate_slug_async(task))

    def save(self, artifact: "DebateArtifact") -> str:
        """Save artifact and return permalink slug (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.save_async(artifact))

    def save_dict(self, debate_data: dict, org_id: Optional[str] = None) -> str:
        """Save debate data directly (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.save_dict_async(debate_data, org_id)
        )

    def store(self, debate_data: dict, org_id: Optional[str] = None) -> str:
        """Store debate metadata (sync wrapper)."""
        slug = self.save_dict(debate_data, org_id=org_id)
        return debate_data.get("id", slug)

    def get_by_slug(
        self,
        slug: str,
        org_id: Optional[str] = None,
        verify_ownership: bool = False,
    ) -> Optional[dict]:
        """Get debate by slug (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.get_by_slug_async(slug, org_id, verify_ownership)
        )

    def get_by_id(
        self,
        debate_id: str,
        org_id: Optional[str] = None,
        verify_ownership: bool = False,
    ) -> Optional[dict]:
        """Get debate by ID (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.get_by_id_async(debate_id, org_id, verify_ownership)
        )

    def list_recent(self, limit: int = 20, org_id: Optional[str] = None) -> list[DebateMetadata]:
        """List recent debates (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.list_recent_async(limit, org_id))

    def search(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0,
        org_id: Optional[str] = None,
    ) -> tuple[list[DebateMetadata], int]:
        """Search debates (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.search_async(query, limit, offset, org_id)
        )

    def delete(
        self,
        slug: str,
        org_id: Optional[str] = None,
        require_ownership: bool = False,
    ) -> bool:
        """Delete debate by slug (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.delete_async(slug, org_id, require_ownership)
        )

    def update_audio(
        self,
        debate_id: str,
        audio_path: str,
        duration_seconds: Optional[int] = None,
    ) -> bool:
        """Update audio information (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.update_audio_async(debate_id, audio_path, duration_seconds)
        )

    def get_audio_info(self, debate_id: str) -> Optional[dict]:
        """Get audio information (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.get_audio_info_async(debate_id))

    def is_public(self, debate_id: str) -> bool:
        """Check if debate is public (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.is_public_async(debate_id))

    def set_public(self, debate_id: str, is_public: bool, org_id: Optional[str] = None) -> bool:
        """Set debate public status (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.set_public_async(debate_id, is_public, org_id)
        )

    # Alias methods for interface compatibility
    def get(self, debate_id: str) -> Optional[dict]:
        """Get debate by ID (alias)."""
        return self.get_by_id(debate_id)

    def get_debate(self, debate_id: str) -> Optional[dict]:
        """Get debate by ID (handler-compatible alias)."""
        return self.get_by_id(debate_id)

    def get_debate_by_slug(self, slug: str) -> Optional[dict]:
        """Get debate by slug (handler-compatible alias)."""
        return self.get_by_slug(slug)

    def list_debates(self, limit: int = 20, org_id: Optional[str] = None) -> list[DebateMetadata]:
        """List debates (handler-compatible alias)."""
        return self.list_recent(limit=limit, org_id=org_id)

    # =========================================================================
    # Async implementations
    # =========================================================================

    async def generate_slug_async(self, task: str) -> str:
        """Generate URL-friendly slug from task description."""
        # Extract words, remove punctuation
        words = re.sub(r"[^\w\s]", "", task.lower()).split()

        # Filter stop words and take first 4 meaningful words
        key_words = [w for w in words if w not in self.STOP_WORDS][:4]
        base = "-".join(key_words) if key_words else "debate"

        # Add date
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        slug = f"{base}-{date}"

        # Handle collisions
        async with self.connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT COUNT(*) as count FROM debates
                WHERE slug = $1 OR slug ~ $2
                """,
                slug,
                f"^{re.escape(slug)}-[0-9]+$",
            )
            count = row["count"] if row else 0

        return f"{slug}-{count + 1}" if count > 0 else slug

    async def save_async(self, artifact: "DebateArtifact") -> str:
        """Save artifact and return permalink slug."""
        slug = await self.generate_slug_async(artifact.task)

        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO debates (
                    id, slug, task, agents, artifact_json,
                    consensus_reached, confidence
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                artifact.artifact_id,
                slug,
                artifact.task,
                json.dumps(artifact.agents),
                artifact.to_json(),
                artifact.consensus_proof.reached if artifact.consensus_proof else False,
                artifact.consensus_proof.confidence if artifact.consensus_proof else 0,
            )

        return slug

    async def save_dict_async(self, debate_data: dict, org_id: Optional[str] = None) -> str:
        """Save debate data directly (without DebateArtifact)."""
        slug = await self.generate_slug_async(debate_data.get("task", "debate"))
        debate_id = debate_data.get("id", slug)

        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO debates (
                    id, slug, task, agents, artifact_json,
                    consensus_reached, confidence, org_id
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                debate_id,
                slug,
                debate_data.get("task", ""),
                json.dumps(debate_data.get("agents", [])),
                json.dumps(debate_data),
                debate_data.get("consensus_reached", False),
                debate_data.get("confidence", 0),
                org_id,
            )

        return slug

    async def get_by_slug_async(
        self,
        slug: str,
        org_id: Optional[str] = None,
        verify_ownership: bool = False,
    ) -> Optional[dict]:
        """Get debate by slug, incrementing view count."""
        if not slug or len(slug) > 500:
            return None

        async with self.connection() as conn:
            if verify_ownership and org_id:
                row = await conn.fetchrow(
                    "SELECT artifact_json FROM debates WHERE slug = $1 AND org_id = $2",
                    slug,
                    org_id,
                )
            else:
                row = await conn.fetchrow("SELECT artifact_json FROM debates WHERE slug = $1", slug)

            if row:
                await conn.execute(
                    "UPDATE debates SET view_count = view_count + 1 WHERE slug = $1",
                    slug,
                )
                artifact_json = row["artifact_json"]
                if isinstance(artifact_json, str):
                    return json.loads(artifact_json)
                return artifact_json

        return None

    async def get_by_id_async(
        self,
        debate_id: str,
        org_id: Optional[str] = None,
        verify_ownership: bool = False,
    ) -> Optional[dict]:
        """Get debate by ID."""
        async with self.connection() as conn:
            if verify_ownership and org_id:
                row = await conn.fetchrow(
                    "SELECT artifact_json FROM debates WHERE id = $1 AND org_id = $2",
                    debate_id,
                    org_id,
                )
            else:
                row = await conn.fetchrow(
                    "SELECT artifact_json FROM debates WHERE id = $1", debate_id
                )

            if row:
                artifact_json = row["artifact_json"]
                if isinstance(artifact_json, str):
                    return json.loads(artifact_json)
                return artifact_json

        return None

    async def list_recent_async(
        self, limit: int = 20, org_id: Optional[str] = None
    ) -> list[DebateMetadata]:
        """List recent debates."""
        async with self.connection() as conn:
            if org_id:
                rows = await conn.fetch(
                    """
                    SELECT slug, id, task, agents, consensus_reached,
                           confidence, created_at, view_count, is_public
                    FROM debates
                    WHERE org_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    org_id,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT slug, id, task, agents, consensus_reached,
                           confidence, created_at, view_count, is_public
                    FROM debates
                    ORDER BY created_at DESC
                    LIMIT $1
                    """,
                    limit,
                )

            return [self._row_to_metadata(row) for row in rows]

    async def search_async(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0,
        org_id: Optional[str] = None,
    ) -> tuple[list[DebateMetadata], int]:
        """Search debates by task/slug using full-text search."""
        async with self.connection() as conn:
            params: list[Any] = []
            param_num = 1

            # Build count query
            count_sql = "SELECT COUNT(*) FROM debates WHERE 1=1"
            if org_id:
                count_sql += f" AND org_id = ${param_num}"
                params.append(org_id)
                param_num += 1

            if query:
                count_sql += f" AND search_vector @@ plainto_tsquery('english', ${param_num})"
                params.append(query)
                param_num += 1

            row = await conn.fetchrow(count_sql, *params)
            total = row[0] if row else 0

            # Build search query
            params: list[Any] = []
            param_num = 1
            search_sql = """
                SELECT slug, id, task, agents, consensus_reached,
                       confidence, created_at, view_count, is_public
                FROM debates WHERE 1=1
            """

            if org_id:
                search_sql += f" AND org_id = ${param_num}"
                params.append(org_id)
                param_num += 1

            if query:
                search_sql += f" AND search_vector @@ plainto_tsquery('english', ${param_num})"
                params.append(query)
                param_num += 1

            search_sql += f" ORDER BY created_at DESC LIMIT ${param_num} OFFSET ${param_num + 1}"
            params.extend([limit, offset])

            rows = await conn.fetch(search_sql, *params)
            results = [self._row_to_metadata(row) for row in rows]

            return results, total

    async def delete_async(
        self,
        slug: str,
        org_id: Optional[str] = None,
        require_ownership: bool = False,
    ) -> bool:
        """Delete debate by slug."""
        async with self.connection() as conn:
            if require_ownership and org_id:
                result = await conn.execute(
                    "DELETE FROM debates WHERE slug = $1 AND org_id = $2",
                    slug,
                    org_id,
                )
            else:
                result = await conn.execute("DELETE FROM debates WHERE slug = $1", slug)
            return result != "DELETE 0"

    async def update_audio_async(
        self,
        debate_id: str,
        audio_path: str,
        duration_seconds: Optional[int] = None,
    ) -> bool:
        """Update audio information for a debate."""
        async with self.connection() as conn:
            result = await conn.execute(
                """
                UPDATE debates
                SET audio_path = $1,
                    audio_generated_at = $2,
                    audio_duration_seconds = $3
                WHERE id = $4
                """,
                audio_path,
                datetime.now(timezone.utc),
                duration_seconds,
                debate_id,
            )
            return result != "UPDATE 0"

    async def get_audio_info_async(self, debate_id: str) -> Optional[dict]:
        """Get audio information for a debate."""
        async with self.connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT audio_path, audio_generated_at, audio_duration_seconds
                FROM debates
                WHERE id = $1
                """,
                debate_id,
            )

            if not row or not row["audio_path"]:
                return None

            return {
                "audio_path": row["audio_path"],
                "audio_generated_at": row["audio_generated_at"].isoformat()
                if row["audio_generated_at"]
                else None,
                "audio_duration_seconds": row["audio_duration_seconds"],
            }

    async def is_public_async(self, debate_id: str) -> bool:
        """Check if a debate is publicly accessible."""
        async with self.connection() as conn:
            row = await conn.fetchrow("SELECT is_public FROM debates WHERE id = $1", debate_id)
            return bool(row and row["is_public"])

    async def set_public_async(
        self, debate_id: str, is_public: bool, org_id: Optional[str] = None
    ) -> bool:
        """Set debate public/private status."""
        async with self.connection() as conn:
            if org_id:
                result = await conn.execute(
                    "UPDATE debates SET is_public = $1 WHERE id = $2 AND org_id = $3",
                    is_public,
                    debate_id,
                    org_id,
                )
            else:
                result = await conn.execute(
                    "UPDATE debates SET is_public = $1 WHERE id = $2",
                    is_public,
                    debate_id,
                )
            return result != "UPDATE 0"

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _row_to_metadata(self, row: Any) -> DebateMetadata:
        """Convert database row to DebateMetadata."""
        agents = row["agents"]
        if isinstance(agents, str):
            agents = json.loads(agents)
        elif agents is None:
            agents = []

        created_at = row["created_at"]
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at)
            except ValueError:
                created_at = datetime.now(timezone.utc)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        return DebateMetadata(
            slug=row["slug"],
            debate_id=row["id"],
            task=row["task"],
            agents=agents,
            consensus_reached=bool(row["consensus_reached"]),
            confidence=row["confidence"] or 0,
            created_at=created_at,
            view_count=row["view_count"] or 0,
            is_public=bool(row["is_public"]),
        )

    def close(self) -> None:
        """No-op for pool-based store (pool managed externally)."""
        pass
