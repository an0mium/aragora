"""
PostgreSQL backend for Knowledge Mound.

Provides production-grade storage with connection pooling, async operations,
and full SQL query capabilities for the Knowledge Mound.

Requires: asyncpg, psycopg2 (for sync operations)
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, cast

from aragora.knowledge.mound.types import (
    AccessGrant,
    AccessGrantType,
    KnowledgeItem,
    KnowledgeLink,
    KnowledgeSource,
    ConfidenceLevel,
    MoundStats,
    QueryFilters,
    RelationshipType,
    VisibilityLevel,
)

logger = logging.getLogger(__name__)

# SQL Schema for PostgreSQL
POSTGRES_SCHEMA = """
-- Knowledge nodes (core metadata)
CREATE TABLE IF NOT EXISTS knowledge_nodes (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    node_type TEXT NOT NULL CHECK (node_type IN ('fact', 'claim', 'memory', 'evidence', 'consensus', 'entity', 'culture')),
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    tier TEXT DEFAULT 'slow' CHECK (tier IN ('fast', 'medium', 'slow', 'glacial')),
    surprise_score REAL DEFAULT 0.0,
    update_count INTEGER DEFAULT 1,
    consolidation_score REAL DEFAULT 0.0,
    validation_status TEXT DEFAULT 'unverified',
    consensus_proof_id TEXT,
    last_validated_at TIMESTAMP,
    staleness_score REAL DEFAULT 0.0,
    revalidation_requested BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    -- Visibility and access control (Phase 2)
    visibility TEXT DEFAULT 'workspace' CHECK (visibility IN ('private', 'workspace', 'organization', 'public', 'system')),
    visibility_set_by TEXT,
    is_discoverable BOOLEAN DEFAULT TRUE
);

-- Provenance tracking
CREATE TABLE IF NOT EXISTS provenance_chains (
    id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    agent_id TEXT,
    debate_id TEXT,
    document_id TEXT,
    user_id TEXT,
    parent_provenance_id TEXT REFERENCES provenance_chains(id),
    transformation_type TEXT,
    transformations JSONB DEFAULT '[]',
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Graph relationships
CREATE TABLE IF NOT EXISTS knowledge_relationships (
    id TEXT PRIMARY KEY,
    from_node_id TEXT NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    to_node_id TEXT NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL CHECK (relationship_type IN ('supports', 'contradicts', 'derived_from', 'related_to', 'supersedes', 'cites', 'elaborates')),
    strength REAL DEFAULT 1.0,
    created_by TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Topics for categorization
CREATE TABLE IF NOT EXISTS node_topics (
    node_id TEXT NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    topic TEXT NOT NULL,
    PRIMARY KEY (node_id, topic)
);

-- Culture patterns (organizational learning)
CREATE TABLE IF NOT EXISTS culture_patterns (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    pattern_type TEXT NOT NULL,
    pattern_key TEXT NOT NULL,
    pattern_value JSONB NOT NULL,
    observation_count INTEGER DEFAULT 1,
    confidence REAL DEFAULT 0.5,
    first_observed_at TIMESTAMP NOT NULL,
    last_observed_at TIMESTAMP NOT NULL,
    contributing_debates TEXT[],
    metadata JSONB DEFAULT '{}'
);

-- Staleness tracking
CREATE TABLE IF NOT EXISTS staleness_checks (
    id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    checked_at TIMESTAMP NOT NULL,
    staleness_score REAL NOT NULL,
    trigger TEXT NOT NULL,
    revalidation_requested BOOLEAN DEFAULT FALSE,
    revalidation_completed_at TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Archived nodes
CREATE TABLE IF NOT EXISTS archived_nodes (
    id TEXT PRIMARY KEY,
    original_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    node_data JSONB NOT NULL,
    archived_at TIMESTAMP NOT NULL DEFAULT NOW(),
    archived_by TEXT
);

-- Access grants for fine-grained sharing (Phase 2)
CREATE TABLE IF NOT EXISTS access_grants (
    id TEXT PRIMARY KEY,
    item_id TEXT NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    grantee_type TEXT NOT NULL CHECK (grantee_type IN ('user', 'role', 'workspace', 'organization')),
    grantee_id TEXT NOT NULL,
    permissions TEXT[] DEFAULT '{read}',
    granted_by TEXT,
    granted_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    UNIQUE(item_id, grantee_type, grantee_id)
);

-- Federation nodes (Phase 5) - Track federated KM instances
CREATE TABLE IF NOT EXISTS federation_nodes (
    id TEXT PRIMARY KEY,
    node_name TEXT NOT NULL UNIQUE,
    endpoint_url TEXT NOT NULL,
    region TEXT,
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'syncing', 'offline', 'maintenance')),
    last_heartbeat TIMESTAMP NOT NULL DEFAULT NOW(),
    sync_version BIGINT DEFAULT 0,
    capabilities TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Federation sync state (Phase 5) - Track synchronization between nodes
CREATE TABLE IF NOT EXISTS federation_sync_state (
    id TEXT PRIMARY KEY,
    source_node_id TEXT NOT NULL REFERENCES federation_nodes(id) ON DELETE CASCADE,
    target_node_id TEXT NOT NULL REFERENCES federation_nodes(id) ON DELETE CASCADE,
    last_sync_at TIMESTAMP,
    last_sync_version BIGINT DEFAULT 0,
    sync_status TEXT DEFAULT 'pending' CHECK (sync_status IN ('pending', 'syncing', 'completed', 'failed')),
    items_synced INTEGER DEFAULT 0,
    items_failed INTEGER DEFAULT 0,
    error_message TEXT,
    next_sync_at TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE(source_node_id, target_node_id)
);

-- Federation item ownership (Phase 5) - Track which node owns each item
CREATE TABLE IF NOT EXISTS federation_item_ownership (
    item_id TEXT NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    owner_node_id TEXT NOT NULL REFERENCES federation_nodes(id) ON DELETE CASCADE,
    sync_version BIGINT NOT NULL DEFAULT 1,
    is_authoritative BOOLEAN DEFAULT TRUE,
    replicated_to TEXT[] DEFAULT '{}',
    replicated_at TIMESTAMP,
    PRIMARY KEY (item_id)
);

-- Distributed locks (Phase 5) - Coordinate access across nodes
CREATE TABLE IF NOT EXISTS distributed_locks (
    lock_key TEXT PRIMARY KEY,
    owner_node_id TEXT NOT NULL REFERENCES federation_nodes(id) ON DELETE CASCADE,
    owner_process_id TEXT NOT NULL,
    lock_type TEXT DEFAULT 'exclusive' CHECK (lock_type IN ('exclusive', 'shared')),
    acquired_at TIMESTAMP NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP NOT NULL,
    heartbeat_at TIMESTAMP NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Federation events (Phase 5) - Event log for federation operations
CREATE TABLE IF NOT EXISTS federation_events (
    id BIGSERIAL PRIMARY KEY,
    event_type TEXT NOT NULL CHECK (event_type IN ('sync', 'replicate', 'conflict', 'lock', 'unlock', 'join', 'leave')),
    source_node_id TEXT REFERENCES federation_nodes(id),
    target_node_id TEXT REFERENCES federation_nodes(id),
    item_id TEXT,
    event_data JSONB DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_nodes_workspace ON knowledge_nodes(workspace_id);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON knowledge_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_nodes_tier ON knowledge_nodes(tier);
CREATE INDEX IF NOT EXISTS idx_nodes_staleness ON knowledge_nodes(staleness_score DESC);
CREATE INDEX IF NOT EXISTS idx_nodes_content_hash ON knowledge_nodes(content_hash);
CREATE INDEX IF NOT EXISTS idx_nodes_validation ON knowledge_nodes(validation_status);
CREATE INDEX IF NOT EXISTS idx_nodes_updated ON knowledge_nodes(updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_provenance_node ON provenance_chains(node_id);
CREATE INDEX IF NOT EXISTS idx_provenance_source ON provenance_chains(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_provenance_debate ON provenance_chains(debate_id) WHERE debate_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_rel_from ON knowledge_relationships(from_node_id);
CREATE INDEX IF NOT EXISTS idx_rel_to ON knowledge_relationships(to_node_id);
CREATE INDEX IF NOT EXISTS idx_rel_type ON knowledge_relationships(relationship_type);

CREATE INDEX IF NOT EXISTS idx_topics_topic ON node_topics(topic);

CREATE UNIQUE INDEX IF NOT EXISTS idx_culture_unique ON culture_patterns(workspace_id, pattern_type, pattern_key);
CREATE INDEX IF NOT EXISTS idx_culture_workspace ON culture_patterns(workspace_id);

CREATE INDEX IF NOT EXISTS idx_staleness_node ON staleness_checks(node_id);
CREATE INDEX IF NOT EXISTS idx_staleness_pending ON staleness_checks(revalidation_requested) WHERE revalidation_requested = TRUE;

CREATE INDEX IF NOT EXISTS idx_grants_item ON access_grants(item_id);
CREATE INDEX IF NOT EXISTS idx_grants_grantee ON access_grants(grantee_id);
CREATE INDEX IF NOT EXISTS idx_grants_expires ON access_grants(expires_at) WHERE expires_at IS NOT NULL;

-- Federation indexes (Phase 5)
CREATE INDEX IF NOT EXISTS idx_fed_nodes_status ON federation_nodes(status);
CREATE INDEX IF NOT EXISTS idx_fed_nodes_heartbeat ON federation_nodes(last_heartbeat DESC);

CREATE INDEX IF NOT EXISTS idx_fed_sync_source ON federation_sync_state(source_node_id);
CREATE INDEX IF NOT EXISTS idx_fed_sync_target ON federation_sync_state(target_node_id);
CREATE INDEX IF NOT EXISTS idx_fed_sync_status ON federation_sync_state(sync_status);
CREATE INDEX IF NOT EXISTS idx_fed_sync_next ON federation_sync_state(next_sync_at) WHERE next_sync_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_fed_ownership_owner ON federation_item_ownership(owner_node_id);

CREATE INDEX IF NOT EXISTS idx_dist_locks_owner ON distributed_locks(owner_node_id);
CREATE INDEX IF NOT EXISTS idx_dist_locks_expires ON distributed_locks(expires_at);

CREATE INDEX IF NOT EXISTS idx_fed_events_type ON federation_events(event_type);
CREATE INDEX IF NOT EXISTS idx_fed_events_source ON federation_events(source_node_id);
CREATE INDEX IF NOT EXISTS idx_fed_events_created ON federation_events(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_nodes_visibility ON knowledge_nodes(visibility);
CREATE INDEX IF NOT EXISTS idx_nodes_public ON knowledge_nodes(workspace_id) WHERE visibility = 'public' OR visibility = 'system';

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_nodes_content_fts ON knowledge_nodes USING gin(to_tsvector('english', content));
"""


class PostgresStore:
    """
    PostgreSQL backend for Knowledge Mound with connection pooling.

    Provides async operations for high-performance concurrent access.
    """

    def __init__(
        self,
        url: str,
        pool_size: int = 10,
        max_overflow: int = 5,
    ):
        """
        Initialize PostgreSQL store.

        Args:
            url: PostgreSQL connection URL
            pool_size: Connection pool size
            max_overflow: Max additional connections above pool_size
        """
        self._url = url
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._pool: Optional[Any] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize connection pool and schema."""
        if self._initialized:
            return

        try:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                self._url,
                min_size=2,
                max_size=self._pool_size + self._max_overflow,
            )

            # Create schema
            async with self._pool.acquire() as conn:
                await conn.execute(POSTGRES_SCHEMA)

            self._initialized = True
            logger.info("PostgreSQL store initialized")

        except ImportError:
            raise ImportError(
                "asyncpg required for PostgreSQL backend. Install with: pip install asyncpg"
            )
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected PostgreSQL initialization error: {e}")
            raise

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[Any]:
        """Get a connection from the pool."""
        if not self._pool:
            raise RuntimeError("PostgresStore not initialized")
        async with self._pool.acquire() as conn:
            yield conn

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
        self._initialized = False

    # =========================================================================
    # Node Operations
    # =========================================================================

    async def save_node_async(self, node_data: Dict[str, Any]) -> str:
        """Save a knowledge node."""
        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO knowledge_nodes (
                    id, workspace_id, node_type, content, content_hash,
                    confidence, tier, surprise_score, update_count,
                    consolidation_score, validation_status, consensus_proof_id,
                    staleness_score, created_at, updated_at, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    confidence = EXCLUDED.confidence,
                    update_count = knowledge_nodes.update_count + 1,
                    updated_at = EXCLUDED.updated_at,
                    metadata = EXCLUDED.metadata
                """,
                node_data["id"],
                node_data["workspace_id"],
                node_data.get("node_type", "fact"),
                node_data["content"],
                node_data.get("content_hash", ""),
                node_data.get("confidence", 0.5),
                node_data.get("tier", "slow"),
                node_data.get("surprise_score", 0.0),
                node_data.get("update_count", 1),
                node_data.get("consolidation_score", 0.0),
                node_data.get("validation_status", "unverified"),
                node_data.get("consensus_proof_id"),
                node_data.get("staleness_score", 0.0),
                datetime.fromisoformat(node_data.get("created_at", datetime.now().isoformat())),
                datetime.fromisoformat(node_data.get("updated_at", datetime.now().isoformat())),
                json.dumps(node_data.get("metadata", {})),
            )

            # Save provenance if present
            if node_data.get("source_type"):
                await conn.execute(
                    """
                    INSERT INTO provenance_chains (
                        id, node_id, source_type, source_id, agent_id,
                        debate_id, document_id, user_id, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    f"prov_{node_data['id']}",
                    node_data["id"],
                    node_data["source_type"],
                    node_data.get("debate_id") or node_data.get("document_id") or "",
                    node_data.get("agent_id"),
                    node_data.get("debate_id"),
                    node_data.get("document_id"),
                    node_data.get("user_id"),
                    datetime.now(),
                )

            # Save topics
            if node_data.get("topics"):
                await conn.execute(
                    "DELETE FROM node_topics WHERE node_id = $1",
                    node_data["id"],
                )
                for topic in node_data["topics"]:
                    await conn.execute(
                        "INSERT INTO node_topics (node_id, topic) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                        node_data["id"],
                        topic,
                    )

        return cast(str, node_data["id"])

    async def get_node_async(self, node_id: str) -> Optional[KnowledgeItem]:
        """Get a knowledge node by ID."""
        async with self.connection() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM knowledge_nodes WHERE id = $1",
                node_id,
            )
            if not row:
                return None

            # Get topics
            topic_rows = await conn.fetch(
                "SELECT topic FROM node_topics WHERE node_id = $1",
                node_id,
            )
            topics = [r["topic"] for r in topic_rows]

            return KnowledgeItem(
                id=row["id"],
                content=row["content"],
                source=KnowledgeSource.FACT,
                source_id=row["id"],
                confidence=self._validation_to_confidence(row["validation_status"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                metadata={
                    "node_type": row["node_type"],
                    "tier": row["tier"],
                    "topics": topics,
                    **(json.loads(row["metadata"]) if row["metadata"] else {}),
                },
                importance=row["confidence"],
            )

    async def update_node_async(self, node_id: str, updates: Dict[str, Any]) -> None:
        """Update a knowledge node."""
        # Build dynamic update query
        set_clauses = []
        params = [node_id]
        param_idx = 2

        for key, value in updates.items():
            if key == "update_count" and value == "update_count + 1":
                set_clauses.append("update_count = update_count + 1")
            else:
                set_clauses.append(f"{key} = ${param_idx}")
                params.append(value)
                param_idx += 1

        if not set_clauses:
            return

        query = f"UPDATE knowledge_nodes SET {', '.join(set_clauses)} WHERE id = $1"

        async with self.connection() as conn:
            await conn.execute(query, *params)

    async def delete_node_async(self, node_id: str) -> bool:
        """Delete a knowledge node."""
        async with self.connection() as conn:
            result = await conn.execute(
                "DELETE FROM knowledge_nodes WHERE id = $1",
                node_id,
            )
            return str(result) == "DELETE 1"

    async def find_by_content_hash_async(
        self, content_hash: str, workspace_id: str
    ) -> Optional[str]:
        """Find node by content hash."""
        async with self.connection() as conn:
            row = await conn.fetchrow(
                "SELECT id FROM knowledge_nodes WHERE content_hash = $1 AND workspace_id = $2",
                content_hash,
                workspace_id,
            )
            return row["id"] if row else None

    # =========================================================================
    # Relationship Operations
    # =========================================================================

    async def save_relationship_async(self, from_id: str, to_id: str, rel_type: str) -> str:
        """Save a relationship between nodes."""
        import uuid

        rel_id = f"kr_{uuid.uuid4().hex[:16]}"

        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO knowledge_relationships (id, from_node_id, to_node_id, relationship_type, created_at)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT DO NOTHING
                """,
                rel_id,
                from_id,
                to_id,
                rel_type,
                datetime.now(),
            )

        return rel_id

    async def get_relationships_async(
        self,
        node_id: str,
        types: Optional[List[RelationshipType]] = None,
    ) -> List[KnowledgeLink]:
        """Get relationships for a node."""
        async with self.connection() as conn:
            if types:
                type_values = [t.value for t in types]
                rows = await conn.fetch(
                    """
                    SELECT * FROM knowledge_relationships
                    WHERE (from_node_id = $1 OR to_node_id = $1)
                    AND relationship_type = ANY($2)
                    """,
                    node_id,
                    type_values,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM knowledge_relationships
                    WHERE from_node_id = $1 OR to_node_id = $1
                    """,
                    node_id,
                )

            return [
                KnowledgeLink(
                    id=row["id"],
                    source_id=row["from_node_id"],
                    target_id=row["to_node_id"],
                    relationship=RelationshipType(row["relationship_type"]),
                    confidence=row["strength"],
                    created_at=row["created_at"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                )
                for row in rows
            ]

    # =========================================================================
    # Query Operations
    # =========================================================================

    async def query_async(
        self,
        query: str,
        filters: Optional[QueryFilters],
        limit: int,
        workspace_id: str,
    ) -> List[KnowledgeItem]:
        """Query nodes with full-text search."""
        async with self.connection() as conn:
            # Use PostgreSQL full-text search
            rows = await conn.fetch(
                """
                SELECT *, ts_rank(to_tsvector('english', content), plainto_tsquery('english', $1)) as rank
                FROM knowledge_nodes
                WHERE workspace_id = $2
                AND to_tsvector('english', content) @@ plainto_tsquery('english', $1)
                ORDER BY rank DESC
                LIMIT $3
                """,
                query,
                workspace_id,
                limit,
            )

            return [
                KnowledgeItem(
                    id=row["id"],
                    content=row["content"],
                    source=KnowledgeSource.FACT,
                    source_id=row["id"],
                    confidence=self._validation_to_confidence(row["validation_status"]),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    importance=row["confidence"],
                )
                for row in rows
            ]

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_stats_async(self, workspace_id: str) -> MoundStats:
        """Get statistics about the Knowledge Mound."""
        async with self.connection() as conn:
            total = await conn.fetchval(
                "SELECT COUNT(*) FROM knowledge_nodes WHERE workspace_id = $1",
                workspace_id,
            )

            by_type = {}
            for row in await conn.fetch(
                "SELECT node_type, COUNT(*) as count FROM knowledge_nodes WHERE workspace_id = $1 GROUP BY node_type",
                workspace_id,
            ):
                by_type[row["node_type"]] = row["count"]

            by_tier = {}
            for row in await conn.fetch(
                "SELECT tier, COUNT(*) as count FROM knowledge_nodes WHERE workspace_id = $1 GROUP BY tier",
                workspace_id,
            ):
                by_tier[row["tier"]] = row["count"]

            by_validation = {}
            for row in await conn.fetch(
                "SELECT validation_status, COUNT(*) as count FROM knowledge_nodes WHERE workspace_id = $1 GROUP BY validation_status",
                workspace_id,
            ):
                by_validation[row["validation_status"]] = row["count"]

            avg_confidence = (
                await conn.fetchval(
                    "SELECT AVG(confidence) FROM knowledge_nodes WHERE workspace_id = $1",
                    workspace_id,
                )
                or 0.0
            )

            rel_count = await conn.fetchval("SELECT COUNT(*) FROM knowledge_relationships")

            rel_by_type = {}
            for row in await conn.fetch(
                "SELECT relationship_type, COUNT(*) as count FROM knowledge_relationships GROUP BY relationship_type"
            ):
                rel_by_type[row["relationship_type"]] = row["count"]

            stale_count = await conn.fetchval(
                "SELECT COUNT(*) FROM knowledge_nodes WHERE workspace_id = $1 AND staleness_score > 0.5",
                workspace_id,
            )

            return MoundStats(
                total_nodes=total or 0,
                nodes_by_type=by_type,
                nodes_by_tier=by_tier,
                nodes_by_validation=by_validation,
                total_relationships=rel_count or 0,
                relationships_by_type=rel_by_type,
                average_confidence=round(avg_confidence, 3),
                stale_nodes_count=stale_count or 0,
                workspace_id=workspace_id,
            )

    # =========================================================================
    # Culture Patterns
    # =========================================================================

    async def save_culture_pattern_async(self, pattern: Dict[str, Any]) -> str:
        """Save a culture pattern."""
        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO culture_patterns (
                    id, workspace_id, pattern_type, pattern_key, pattern_value,
                    observation_count, confidence, first_observed_at, last_observed_at,
                    contributing_debates, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (workspace_id, pattern_type, pattern_key) DO UPDATE SET
                    observation_count = culture_patterns.observation_count + 1,
                    confidence = EXCLUDED.confidence,
                    last_observed_at = EXCLUDED.last_observed_at,
                    contributing_debates = array_cat(culture_patterns.contributing_debates, EXCLUDED.contributing_debates)
                """,
                pattern["id"],
                pattern["workspace_id"],
                pattern["pattern_type"],
                pattern["pattern_key"],
                json.dumps(pattern["pattern_value"]),
                pattern.get("observation_count", 1),
                pattern.get("confidence", 0.5),
                pattern.get("first_observed_at", datetime.now()),
                pattern.get("last_observed_at", datetime.now()),
                pattern.get("contributing_debates", []),
                json.dumps(pattern.get("metadata", {})),
            )

        return pattern["id"]

    async def get_culture_patterns_async(
        self, workspace_id: str, pattern_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get culture patterns for a workspace."""
        async with self.connection() as conn:
            if pattern_type:
                rows = await conn.fetch(
                    """
                    SELECT * FROM culture_patterns
                    WHERE workspace_id = $1 AND pattern_type = $2
                    ORDER BY confidence DESC
                    """,
                    workspace_id,
                    pattern_type,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM culture_patterns
                    WHERE workspace_id = $1
                    ORDER BY pattern_type, confidence DESC
                    """,
                    workspace_id,
                )

            return [
                {
                    "id": row["id"],
                    "workspace_id": row["workspace_id"],
                    "pattern_type": row["pattern_type"],
                    "pattern_key": row["pattern_key"],
                    "pattern_value": json.loads(row["pattern_value"]),
                    "observation_count": row["observation_count"],
                    "confidence": row["confidence"],
                    "first_observed_at": row["first_observed_at"],
                    "last_observed_at": row["last_observed_at"],
                    "contributing_debates": row["contributing_debates"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                }
                for row in rows
            ]

    # =========================================================================
    # Access Grants (Phase 2)
    # =========================================================================

    async def save_access_grant_async(self, grant: AccessGrant) -> str:
        """Save an access grant."""
        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO access_grants (
                    id, item_id, grantee_type, grantee_id, permissions,
                    granted_by, granted_at, expires_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (item_id, grantee_type, grantee_id) DO UPDATE SET
                    permissions = EXCLUDED.permissions,
                    granted_by = EXCLUDED.granted_by,
                    granted_at = EXCLUDED.granted_at,
                    expires_at = EXCLUDED.expires_at
                """,
                grant.id,
                grant.item_id,
                grant.grantee_type.value,
                grant.grantee_id,
                grant.permissions,
                grant.granted_by,
                grant.granted_at,
                grant.expires_at,
            )
        return grant.id

    async def get_access_grants_async(self, item_id: str) -> List[AccessGrant]:
        """Get all access grants for an item."""
        async with self.connection() as conn:
            rows = await conn.fetch(
                "SELECT * FROM access_grants WHERE item_id = $1",
                item_id,
            )
            return [
                AccessGrant(
                    id=row["id"],
                    item_id=row["item_id"],
                    grantee_type=AccessGrantType(row["grantee_type"]),
                    grantee_id=row["grantee_id"],
                    permissions=list(row["permissions"]) if row["permissions"] else ["read"],
                    granted_by=row["granted_by"],
                    granted_at=row["granted_at"],
                    expires_at=row["expires_at"],
                )
                for row in rows
            ]

    async def get_grants_for_grantee_async(
        self, grantee_id: str, grantee_type: Optional[AccessGrantType] = None
    ) -> List[AccessGrant]:
        """Get all grants for a specific grantee."""
        async with self.connection() as conn:
            if grantee_type:
                rows = await conn.fetch(
                    """
                    SELECT * FROM access_grants
                    WHERE grantee_id = $1 AND grantee_type = $2
                    AND (expires_at IS NULL OR expires_at > NOW())
                    """,
                    grantee_id,
                    grantee_type.value,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM access_grants
                    WHERE grantee_id = $1
                    AND (expires_at IS NULL OR expires_at > NOW())
                    """,
                    grantee_id,
                )
            return [
                AccessGrant(
                    id=row["id"],
                    item_id=row["item_id"],
                    grantee_type=AccessGrantType(row["grantee_type"]),
                    grantee_id=row["grantee_id"],
                    permissions=list(row["permissions"]) if row["permissions"] else ["read"],
                    granted_by=row["granted_by"],
                    granted_at=row["granted_at"],
                    expires_at=row["expires_at"],
                )
                for row in rows
            ]

    async def delete_access_grant_async(self, item_id: str, grantee_id: str) -> bool:
        """Delete an access grant."""
        async with self.connection() as conn:
            result = await conn.execute(
                "DELETE FROM access_grants WHERE item_id = $1 AND grantee_id = $2",
                item_id,
                grantee_id,
            )
            return "DELETE 1" in result

    async def query_with_visibility_async(
        self,
        query: str,
        workspace_id: str,
        actor_id: str,
        actor_workspace_id: str,
        actor_org_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[KnowledgeItem]:
        """Query nodes with visibility filtering."""
        async with self.connection() as conn:
            # Build visibility filter SQL
            # Items are visible if:
            # 1. visibility is 'public' or 'system'
            # 2. visibility is 'workspace' and workspace_id matches
            # 3. visibility is 'organization' and org matches (via metadata)
            # 4. visibility is 'private' and there's an access grant
            rows = await conn.fetch(
                """
                SELECT DISTINCT n.*, ts_rank(to_tsvector('english', n.content), plainto_tsquery('english', $1)) as rank
                FROM knowledge_nodes n
                LEFT JOIN access_grants g ON n.id = g.item_id
                WHERE (
                    -- Public or system visibility
                    n.visibility IN ('public', 'system')
                    -- Workspace visibility
                    OR (n.visibility = 'workspace' AND n.workspace_id = $2)
                    -- Organization visibility (check metadata for org_id)
                    OR (n.visibility = 'organization' AND n.metadata->>'org_id' = $4)
                    -- Private with explicit grant
                    OR (n.visibility = 'private' AND g.grantee_id = $3 AND (g.expires_at IS NULL OR g.expires_at > NOW()))
                )
                AND n.is_discoverable = TRUE
                AND to_tsvector('english', n.content) @@ plainto_tsquery('english', $1)
                ORDER BY rank DESC
                LIMIT $5
                """,
                query,
                workspace_id,
                actor_id,
                actor_org_id or "",
                limit,
            )

            return [
                KnowledgeItem(
                    id=row["id"],
                    content=row["content"],
                    source=KnowledgeSource.FACT,
                    source_id=row["id"],
                    confidence=self._validation_to_confidence(row["validation_status"]),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    metadata={
                        "node_type": row["node_type"],
                        "tier": row["tier"],
                        "visibility": row["visibility"],
                        **(json.loads(row["metadata"]) if row["metadata"] else {}),
                    },
                    importance=row["confidence"],
                )
                for row in rows
            ]

    async def update_visibility_async(
        self,
        node_id: str,
        visibility: VisibilityLevel,
        set_by: Optional[str] = None,
    ) -> None:
        """Update the visibility of a knowledge node."""
        async with self.connection() as conn:
            await conn.execute(
                """
                UPDATE knowledge_nodes
                SET visibility = $2, visibility_set_by = $3, updated_at = NOW()
                WHERE id = $1
                """,
                node_id,
                visibility.value,
                set_by,
            )

    # =========================================================================
    # Helpers
    # =========================================================================

    def _validation_to_confidence(self, status: str) -> ConfidenceLevel:
        """Map validation status to confidence level."""
        mapping = {
            "verified": ConfidenceLevel.VERIFIED,
            "majority_agreed": ConfidenceLevel.HIGH,
            "byzantine_agreed": ConfidenceLevel.VERIFIED,
            "formally_proven": ConfidenceLevel.VERIFIED,
            "unverified": ConfidenceLevel.MEDIUM,
            "contradicted": ConfidenceLevel.LOW,
        }
        return mapping.get(status.lower(), ConfidenceLevel.MEDIUM)
