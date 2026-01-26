"""
PostgreSQL Fact Store - PostgreSQL-based persistence for the Knowledge Base.

Provides async storage, retrieval, and search for facts using PostgreSQL
with tsvector-based full-text search for production deployments.
"""

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from aragora.knowledge.types import (
    Fact,
    FactFilters,
    FactRelation,
    FactRelationType,
    ValidationStatus,
)
from aragora.storage.postgres_store import PostgresStore

logger = logging.getLogger(__name__)


class PostgresFactStore(PostgresStore):
    """PostgreSQL-backed fact persistence store.

    Stores facts extracted from documents along with their
    validation status, evidence links, and relationships.
    Uses PostgreSQL tsvector for full-text search.
    """

    SCHEMA_NAME = "fact_store"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        -- Main facts table with tsvector for FTS
        CREATE TABLE IF NOT EXISTS facts (
            id TEXT PRIMARY KEY,
            statement TEXT NOT NULL,
            statement_hash TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,
            evidence_ids JSONB DEFAULT '[]',
            consensus_proof_id TEXT,
            source_documents JSONB DEFAULT '[]',
            workspace_id TEXT NOT NULL,
            validation_status TEXT DEFAULT 'unverified',
            topics JSONB DEFAULT '[]',
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            superseded_by TEXT,
            -- tsvector column for full-text search
            search_vector TSVECTOR
        );

        -- Fact relations table
        CREATE TABLE IF NOT EXISTS fact_relations (
            id TEXT PRIMARY KEY,
            source_fact_id TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
            target_fact_id TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
            relation_type TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,
            created_by TEXT,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        -- Indexes for facts
        CREATE INDEX IF NOT EXISTS idx_facts_workspace ON facts(workspace_id);
        CREATE INDEX IF NOT EXISTS idx_facts_status ON facts(validation_status);
        CREATE INDEX IF NOT EXISTS idx_facts_confidence ON facts(confidence);
        CREATE INDEX IF NOT EXISTS idx_facts_hash ON facts(statement_hash);
        CREATE INDEX IF NOT EXISTS idx_facts_created ON facts(created_at);
        CREATE INDEX IF NOT EXISTS idx_facts_hash_workspace ON facts(statement_hash, workspace_id);

        -- GIN index for full-text search
        CREATE INDEX IF NOT EXISTS idx_facts_search ON facts USING GIN(search_vector);

        -- Indexes for relations
        CREATE INDEX IF NOT EXISTS idx_relations_source ON fact_relations(source_fact_id);
        CREATE INDEX IF NOT EXISTS idx_relations_target ON fact_relations(target_fact_id);
        CREATE INDEX IF NOT EXISTS idx_relations_type ON fact_relations(relation_type);

        -- Function to update search vector
        CREATE OR REPLACE FUNCTION update_fact_search_vector()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.search_vector :=
                setweight(to_tsvector('english', COALESCE(NEW.statement, '')), 'A') ||
                setweight(to_tsvector('english', COALESCE(
                    (SELECT string_agg(topic, ' ') FROM jsonb_array_elements_text(NEW.topics) AS topic),
                    ''
                )), 'B');
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

        -- Trigger to auto-update search vector
        DROP TRIGGER IF EXISTS fact_search_vector_trigger ON facts;
        CREATE TRIGGER fact_search_vector_trigger
            BEFORE INSERT OR UPDATE ON facts
            FOR EACH ROW
            EXECUTE FUNCTION update_fact_search_vector();
    """

    def _compute_statement_hash(self, statement: str) -> str:
        """Compute hash for statement deduplication."""
        normalized = " ".join(statement.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    # =========================================================================
    # Sync wrappers for compatibility
    # =========================================================================

    def add_fact(
        self,
        statement: str,
        workspace_id: str,
        evidence_ids: Optional[list[str]] = None,
        source_documents: Optional[list[str]] = None,
        confidence: float = 0.5,
        topics: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        validation_status: ValidationStatus = ValidationStatus.UNVERIFIED,
        deduplicate: bool = True,
    ) -> Fact:
        """Add a fact to the store (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.add_fact_async(
                statement=statement,
                workspace_id=workspace_id,
                evidence_ids=evidence_ids,
                source_documents=source_documents,
                confidence=confidence,
                topics=topics,
                metadata=metadata,
                validation_status=validation_status,
                deduplicate=deduplicate,
            )
        )

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        """Get a fact by ID (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.get_fact_async(fact_id))

    def update_fact(
        self,
        fact_id: str,
        confidence: Optional[float] = None,
        validation_status: Optional[ValidationStatus] = None,
        consensus_proof_id: Optional[str] = None,
        evidence_ids: Optional[list[str]] = None,
        topics: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        superseded_by: Optional[str] = None,
    ) -> Optional[Fact]:
        """Update a fact (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.update_fact_async(
                fact_id=fact_id,
                confidence=confidence,
                validation_status=validation_status,
                consensus_proof_id=consensus_proof_id,
                evidence_ids=evidence_ids,
                topics=topics,
                metadata=metadata,
                superseded_by=superseded_by,
            )
        )

    def query_facts(
        self,
        query: str,
        filters: Optional[FactFilters] = None,
    ) -> list[Fact]:
        """Search facts using full-text search (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.query_facts_async(query, filters))

    def list_facts(self, filters: Optional[FactFilters] = None) -> list[Fact]:
        """List facts with optional filtering (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.list_facts_async(filters))

    def get_contradictions(self, fact_id: str) -> list[Fact]:
        """Get facts that contradict a given fact (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.get_contradictions_async(fact_id))

    def add_relation(
        self,
        source_fact_id: str,
        target_fact_id: str,
        relation_type: FactRelationType,
        confidence: float = 0.5,
        created_by: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> FactRelation:
        """Add a relation between facts (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.add_relation_async(
                source_fact_id=source_fact_id,
                target_fact_id=target_fact_id,
                relation_type=relation_type,
                confidence=confidence,
                created_by=created_by,
                metadata=metadata,
            )
        )

    def get_relations(
        self,
        fact_id: str,
        relation_type: Optional[FactRelationType] = None,
        as_source: bool = True,
        as_target: bool = True,
    ) -> list[FactRelation]:
        """Get relations for a fact (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.get_relations_async(fact_id, relation_type, as_source, as_target)
        )

    def delete_fact(self, fact_id: str) -> bool:
        """Delete a fact and its relations (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.delete_fact_async(fact_id))

    def get_statistics(self, workspace_id: Optional[str] = None) -> dict[str, Any]:
        """Get store statistics (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.get_statistics_async(workspace_id))

    # =========================================================================
    # Async implementations
    # =========================================================================

    async def add_fact_async(
        self,
        statement: str,
        workspace_id: str,
        evidence_ids: Optional[list[str]] = None,
        source_documents: Optional[list[str]] = None,
        confidence: float = 0.5,
        topics: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        validation_status: ValidationStatus = ValidationStatus.UNVERIFIED,
        deduplicate: bool = True,
    ) -> Fact:
        """Add a fact to the store asynchronously.

        Args:
            statement: The factual claim
            workspace_id: Workspace this fact belongs to
            evidence_ids: Links to evidence entries
            source_documents: Document IDs this was extracted from
            confidence: Initial confidence score
            topics: Topics for categorization
            metadata: Additional structured data
            validation_status: Initial validation status
            deduplicate: If True, return existing fact with same statement

        Returns:
            The created or existing Fact
        """
        evidence_ids = evidence_ids or []
        source_documents = source_documents or []
        topics = topics or []
        metadata = metadata or {}

        statement_hash = self._compute_statement_hash(statement)

        async with self.connection() as conn:
            # Check for existing fact with same statement in workspace
            if deduplicate:
                row = await conn.fetchrow(
                    """
                    SELECT id, statement, statement_hash, confidence, evidence_ids,
                           consensus_proof_id, source_documents, workspace_id,
                           validation_status, topics, metadata, created_at,
                           updated_at, superseded_by
                    FROM facts
                    WHERE statement_hash = $1 AND workspace_id = $2
                    """,
                    statement_hash,
                    workspace_id,
                )
                if row:
                    logger.debug(f"Fact deduplicated: {row['id']}")
                    return self._row_to_fact(row)

            # Create new fact
            fact_id = f"fact_{uuid.uuid4().hex[:12]}"
            now = datetime.now(timezone.utc)

            await conn.execute(
                """
                INSERT INTO facts (
                    id, statement, statement_hash, confidence,
                    evidence_ids, source_documents,
                    workspace_id, validation_status, topics,
                    metadata, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $11)
                """,
                fact_id,
                statement,
                statement_hash,
                confidence,
                json.dumps(evidence_ids),
                json.dumps(source_documents),
                workspace_id,
                validation_status.value,
                json.dumps(topics),
                json.dumps(metadata),
                now,
            )

        return Fact(
            id=fact_id,
            statement=statement,
            confidence=confidence,
            evidence_ids=evidence_ids,
            source_documents=source_documents,
            workspace_id=workspace_id,
            validation_status=validation_status,
            topics=topics,
            metadata=metadata,
            created_at=now,
            updated_at=now,
        )

    async def get_fact_async(self, fact_id: str) -> Optional[Fact]:
        """Get a fact by ID asynchronously.

        Args:
            fact_id: Fact ID

        Returns:
            Fact or None if not found
        """
        async with self.connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, statement, statement_hash, confidence, evidence_ids,
                       consensus_proof_id, source_documents, workspace_id,
                       validation_status, topics, metadata, created_at,
                       updated_at, superseded_by
                FROM facts WHERE id = $1
                """,
                fact_id,
            )
            if not row:
                return None
            return self._row_to_fact(row)

    async def update_fact_async(
        self,
        fact_id: str,
        confidence: Optional[float] = None,
        validation_status: Optional[ValidationStatus] = None,
        consensus_proof_id: Optional[str] = None,
        evidence_ids: Optional[list[str]] = None,
        topics: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        superseded_by: Optional[str] = None,
    ) -> Optional[Fact]:
        """Update a fact asynchronously.

        Args:
            fact_id: Fact to update
            confidence: New confidence score
            validation_status: New validation status
            consensus_proof_id: Consensus proof ID
            evidence_ids: Updated evidence links
            topics: Updated topics
            metadata: Updated metadata
            superseded_by: ID of superseding fact

        Returns:
            Updated Fact or None if not found
        """
        updates: list[str] = []
        params: list[Any] = []
        param_num = 1

        if confidence is not None:
            updates.append(f"confidence = ${param_num}")
            params.append(confidence)
            param_num += 1

        if validation_status is not None:
            updates.append(f"validation_status = ${param_num}")
            params.append(validation_status.value)
            param_num += 1

        if consensus_proof_id is not None:
            updates.append(f"consensus_proof_id = ${param_num}")
            params.append(consensus_proof_id)
            param_num += 1

        if evidence_ids is not None:
            updates.append(f"evidence_ids = ${param_num}")
            params.append(json.dumps(evidence_ids))
            param_num += 1

        if topics is not None:
            updates.append(f"topics = ${param_num}")
            params.append(json.dumps(topics))
            param_num += 1

        if metadata is not None:
            updates.append(f"metadata = ${param_num}")
            params.append(json.dumps(metadata))
            param_num += 1

        if superseded_by is not None:
            updates.append(f"superseded_by = ${param_num}")
            params.append(superseded_by)
            param_num += 1

        if not updates:
            return await self.get_fact_async(fact_id)

        updates.append(f"updated_at = ${param_num}")
        params.append(datetime.now(timezone.utc))
        param_num += 1

        params.append(fact_id)

        async with self.connection() as conn:
            result = await conn.execute(
                f"UPDATE facts SET {', '.join(updates)} WHERE id = ${param_num}",
                *params,
            )

            if result == "UPDATE 0":
                return None

        return await self.get_fact_async(fact_id)

    async def query_facts_async(
        self,
        query: str,
        filters: Optional[FactFilters] = None,
    ) -> list[Fact]:
        """Search facts using full-text search asynchronously.

        Args:
            query: Search query
            filters: Optional filters to apply

        Returns:
            List of matching facts
        """
        filters = filters or FactFilters()

        # Sanitize query for tsquery
        sanitized = self._sanitize_tsquery(query)
        if not sanitized:
            return await self.list_facts_async(filters)

        async with self.connection() as conn:
            # Build query with plainto_tsquery for safe parsing
            sql = """
                SELECT id, statement, statement_hash, confidence, evidence_ids,
                       consensus_proof_id, source_documents, workspace_id,
                       validation_status, topics, metadata, created_at,
                       updated_at, superseded_by,
                       ts_rank(search_vector, plainto_tsquery('english', $1)) as rank
                FROM facts
                WHERE search_vector @@ plainto_tsquery('english', $1)
            """
            params: list[Any] = [sanitized]
            param_num = 2

            # Apply filters
            if filters.workspace_id:
                sql += f" AND workspace_id = ${param_num}"
                params.append(filters.workspace_id)
                param_num += 1

            if filters.min_confidence > 0:
                sql += f" AND confidence >= ${param_num}"
                params.append(filters.min_confidence)
                param_num += 1

            if filters.validation_status:
                sql += f" AND validation_status = ${param_num}"
                params.append(filters.validation_status.value)
                param_num += 1

            if not filters.include_superseded:
                sql += " AND superseded_by IS NULL"

            if filters.source_documents:
                for doc_id in filters.source_documents:
                    sql += f" AND source_documents @> ${param_num}::jsonb"
                    params.append(json.dumps([doc_id]))
                    param_num += 1

            if filters.created_after:
                sql += f" AND created_at >= ${param_num}"
                params.append(filters.created_after)
                param_num += 1

            if filters.created_before:
                sql += f" AND created_at <= ${param_num}"
                params.append(filters.created_before)
                param_num += 1

            sql += " ORDER BY rank DESC"
            sql += f" LIMIT ${param_num} OFFSET ${param_num + 1}"
            params.extend([filters.limit, filters.offset])

            rows = await conn.fetch(sql, *params)
            return [self._row_to_fact(row) for row in rows]

    async def list_facts_async(self, filters: Optional[FactFilters] = None) -> list[Fact]:
        """List facts with optional filtering asynchronously.

        Args:
            filters: Optional filters

        Returns:
            List of facts
        """
        filters = filters or FactFilters()

        async with self.connection() as conn:
            sql = """
                SELECT id, statement, statement_hash, confidence, evidence_ids,
                       consensus_proof_id, source_documents, workspace_id,
                       validation_status, topics, metadata, created_at,
                       updated_at, superseded_by
                FROM facts WHERE 1=1
            """
            params: list[Any] = []
            param_num = 1

            if filters.workspace_id:
                sql += f" AND workspace_id = ${param_num}"
                params.append(filters.workspace_id)
                param_num += 1

            if filters.min_confidence > 0:
                sql += f" AND confidence >= ${param_num}"
                params.append(filters.min_confidence)
                param_num += 1

            if filters.validation_status:
                sql += f" AND validation_status = ${param_num}"
                params.append(filters.validation_status.value)
                param_num += 1

            if not filters.include_superseded:
                sql += " AND superseded_by IS NULL"

            if filters.topics:
                for topic in filters.topics:
                    sql += f" AND topics @> ${param_num}::jsonb"
                    params.append(json.dumps([topic]))
                    param_num += 1

            if filters.source_documents:
                for doc_id in filters.source_documents:
                    sql += f" AND source_documents @> ${param_num}::jsonb"
                    params.append(json.dumps([doc_id]))
                    param_num += 1

            if filters.created_after:
                sql += f" AND created_at >= ${param_num}"
                params.append(filters.created_after)
                param_num += 1

            if filters.created_before:
                sql += f" AND created_at <= ${param_num}"
                params.append(filters.created_before)
                param_num += 1

            sql += " ORDER BY confidence DESC, created_at DESC"
            sql += f" LIMIT ${param_num} OFFSET ${param_num + 1}"
            params.extend([filters.limit, filters.offset])

            rows = await conn.fetch(sql, *params)
            return [self._row_to_fact(row) for row in rows]

    async def get_contradictions_async(self, fact_id: str) -> list[Fact]:
        """Get facts that contradict a given fact asynchronously.

        Args:
            fact_id: Fact to find contradictions for

        Returns:
            List of contradicting facts
        """
        async with self.connection() as conn:
            rows = await conn.fetch(
                """
                SELECT f.id, f.statement, f.statement_hash, f.confidence,
                       f.evidence_ids, f.consensus_proof_id, f.source_documents,
                       f.workspace_id, f.validation_status, f.topics, f.metadata,
                       f.created_at, f.updated_at, f.superseded_by
                FROM facts f
                JOIN fact_relations r ON (
                    (r.source_fact_id = $1 AND r.target_fact_id = f.id)
                    OR (r.target_fact_id = $1 AND r.source_fact_id = f.id)
                )
                WHERE r.relation_type = $2
                AND f.superseded_by IS NULL
                """,
                fact_id,
                FactRelationType.CONTRADICTS.value,
            )
            return [self._row_to_fact(row) for row in rows]

    async def add_relation_async(
        self,
        source_fact_id: str,
        target_fact_id: str,
        relation_type: FactRelationType,
        confidence: float = 0.5,
        created_by: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> FactRelation:
        """Add a relation between facts asynchronously.

        Args:
            source_fact_id: Source fact ID
            target_fact_id: Target fact ID
            relation_type: Type of relation
            confidence: Confidence in relation
            created_by: Who created this relation
            metadata: Additional data

        Returns:
            The created relation
        """
        relation_id = f"rel_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        async with self.connection() as conn:
            await conn.execute(
                """
                INSERT INTO fact_relations (
                    id, source_fact_id, target_fact_id, relation_type,
                    confidence, created_by, metadata, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                relation_id,
                source_fact_id,
                target_fact_id,
                relation_type.value,
                confidence,
                created_by,
                json.dumps(metadata or {}),
                now,
            )

        return FactRelation(
            id=relation_id,
            source_fact_id=source_fact_id,
            target_fact_id=target_fact_id,
            relation_type=relation_type,
            confidence=confidence,
            created_by=created_by,
            metadata=metadata or {},
            created_at=now,
        )

    async def get_relations_async(
        self,
        fact_id: str,
        relation_type: Optional[FactRelationType] = None,
        as_source: bool = True,
        as_target: bool = True,
    ) -> list[FactRelation]:
        """Get relations for a fact asynchronously.

        Args:
            fact_id: Fact ID
            relation_type: Optional filter by type
            as_source: Include relations where fact is source
            as_target: Include relations where fact is target

        Returns:
            List of relations
        """
        if not as_source and not as_target:
            return []

        async with self.connection() as conn:
            conditions = []
            params: list[Any] = []
            param_num = 1

            if as_source and as_target:
                conditions.append(
                    f"(source_fact_id = ${param_num} OR target_fact_id = ${param_num})"
                )
                params.append(fact_id)
                param_num += 1
            elif as_source:
                conditions.append(f"source_fact_id = ${param_num}")
                params.append(fact_id)
                param_num += 1
            else:
                conditions.append(f"target_fact_id = ${param_num}")
                params.append(fact_id)
                param_num += 1

            sql = f"SELECT * FROM fact_relations WHERE {conditions[0]}"

            if relation_type:
                sql += f" AND relation_type = ${param_num}"
                params.append(relation_type.value)

            rows = await conn.fetch(sql, *params)

            relations = []
            for row in rows:
                relations.append(
                    FactRelation(
                        id=row["id"],
                        source_fact_id=row["source_fact_id"],
                        target_fact_id=row["target_fact_id"],
                        relation_type=FactRelationType(row["relation_type"]),
                        confidence=row["confidence"],
                        created_by=row["created_by"] or "",
                        metadata=(
                            json.loads(row["metadata"])
                            if isinstance(row["metadata"], str)
                            else (row["metadata"] or {})
                        ),
                        created_at=row["created_at"],
                    )
                )
            return relations

    async def delete_fact_async(self, fact_id: str) -> bool:
        """Delete a fact and its relations asynchronously.

        Args:
            fact_id: Fact to delete

        Returns:
            True if deleted
        """
        async with self.transaction() as conn:
            # Relations are deleted via ON DELETE CASCADE
            result = await conn.execute("DELETE FROM facts WHERE id = $1", fact_id)
            return result != "DELETE 0"

    async def get_statistics_async(self, workspace_id: Optional[str] = None) -> dict[str, Any]:
        """Get store statistics asynchronously.

        Args:
            workspace_id: Optional workspace filter

        Returns:
            Statistics dictionary
        """
        async with self.connection() as conn:
            params: list[Any] = []
            where = ""
            param_num = 1

            if workspace_id:
                where = f" WHERE workspace_id = ${param_num}"
                params = [workspace_id]
                param_num += 1

            # Total facts
            row = await conn.fetchrow(f"SELECT COUNT(*) as count FROM facts{where}", *params)
            total = row["count"] if row else 0

            # By status
            rows = await conn.fetch(
                f"""
                SELECT validation_status, COUNT(*) as count
                FROM facts{where}
                GROUP BY validation_status
                """,
                *params,
            )
            by_status = {row["validation_status"]: row["count"] for row in rows}

            # Average confidence
            row = await conn.fetchrow(
                f"SELECT AVG(confidence) as avg FROM facts{where}",
                *params,
            )
            avg_confidence = row["avg"] if row and row["avg"] else 0.0

            # Verified facts
            verified_statuses = (
                ValidationStatus.MAJORITY_AGREED.value,
                ValidationStatus.BYZANTINE_AGREED.value,
                ValidationStatus.FORMALLY_PROVEN.value,
            )
            verified_where = f"validation_status = ANY(${param_num})"
            verified_params = list(params) + [list(verified_statuses)]
            if workspace_id:
                verified_where = f"workspace_id = $1 AND {verified_where}"

            row = await conn.fetchrow(
                f"SELECT COUNT(*) as count FROM facts WHERE {verified_where}",
                *verified_params,
            )
            verified = row["count"] if row else 0

            # Relations count
            row = await conn.fetchrow("SELECT COUNT(*) as count FROM fact_relations")
            relations = row["count"] if row else 0

            return {
                "total_facts": total,
                "verified_facts": verified,
                "by_status": by_status,
                "average_confidence": float(avg_confidence),
                "total_relations": relations,
            }

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _sanitize_tsquery(self, query: str) -> str:
        """Sanitize a query string for use with plainto_tsquery.

        Args:
            query: Raw search query

        Returns:
            Sanitized query string
        """
        # plainto_tsquery handles most sanitization, but we strip
        # completely empty queries
        sanitized = query.strip()
        if not sanitized:
            return ""
        return sanitized

    def _row_to_fact(self, row: Any) -> Fact:
        """Convert database row to Fact object."""
        # Handle JSONB columns - asyncpg returns them as dicts/lists directly
        evidence_ids = row["evidence_ids"]
        if isinstance(evidence_ids, str):
            evidence_ids = json.loads(evidence_ids)
        elif evidence_ids is None:
            evidence_ids = []

        source_documents = row["source_documents"]
        if isinstance(source_documents, str):
            source_documents = json.loads(source_documents)
        elif source_documents is None:
            source_documents = []

        topics = row["topics"]
        if isinstance(topics, str):
            topics = json.loads(topics)
        elif topics is None:
            topics = []

        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        elif metadata is None:
            metadata = {}

        return Fact(
            id=row["id"],
            statement=row["statement"],
            confidence=row["confidence"],
            evidence_ids=evidence_ids,
            consensus_proof_id=row["consensus_proof_id"],
            source_documents=source_documents,
            workspace_id=row["workspace_id"],
            validation_status=ValidationStatus(row["validation_status"]),
            topics=topics,
            metadata=metadata,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            superseded_by=row["superseded_by"],
        )

    def close(self) -> None:
        """No-op for pool-based store (pool managed externally)."""
        pass
