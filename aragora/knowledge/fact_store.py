"""
Fact Store - SQLite-based persistence for the Knowledge Base.

Provides storage, retrieval, and search for facts extracted from
documents and verified through multi-agent consensus.
"""

import hashlib
import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from aragora.knowledge.types import (
    Fact,
    FactFilters,
    FactRelation,
    FactRelationType,
    ValidationStatus,
)
from aragora.storage.base_store import SQLiteStore

logger = logging.getLogger(__name__)

# FTS query limits
MAX_FTS_QUERY_LENGTH = 500
MAX_FTS_TERMS = 20
FTS_SPECIAL_CHARS = set('"*(){}[]^:?-+~')


def sanitize_fts_query(
    query: str,
    max_length: int = MAX_FTS_QUERY_LENGTH,
    max_terms: int = MAX_FTS_TERMS,
) -> str:
    """Sanitize and limit FTS query complexity.

    Args:
        query: Raw search query
        max_length: Maximum query length
        max_terms: Maximum number of terms

    Returns:
        Sanitized query safe for FTS5
    """
    if not query or not query.strip():
        return ""

    query = query[:max_length]

    sanitized = []
    for char in query:
        if char in FTS_SPECIAL_CHARS:
            if char == "*":
                sanitized.append(char)
        else:
            sanitized.append(char)
    query = "".join(sanitized)

    terms = query.split()
    if len(terms) > max_terms:
        terms = terms[:max_terms]

    return " ".join(terms)


class FactStore(SQLiteStore):
    """SQLite-based fact persistence store.

    Stores facts extracted from documents along with their
    validation status, evidence links, and relationships.
    """

    SCHEMA_NAME = "fact_store"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        -- Main facts table
        CREATE TABLE IF NOT EXISTS facts (
            id TEXT PRIMARY KEY,
            statement TEXT NOT NULL,
            statement_hash TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,
            evidence_ids_json TEXT,
            consensus_proof_id TEXT,
            source_documents_json TEXT,
            workspace_id TEXT NOT NULL,
            validation_status TEXT DEFAULT 'unverified',
            topics_json TEXT,
            metadata_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            superseded_by TEXT
        );

        -- Fact relations table
        CREATE TABLE IF NOT EXISTS fact_relations (
            id TEXT PRIMARY KEY,
            source_fact_id TEXT NOT NULL,
            target_fact_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,
            created_by TEXT,
            metadata_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_fact_id) REFERENCES facts(id),
            FOREIGN KEY (target_fact_id) REFERENCES facts(id)
        );

        -- Full-text search index
        CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
        USING fts5(
            fact_id,
            statement,
            topics,
            content=''
        );

        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_facts_workspace ON facts(workspace_id);
        CREATE INDEX IF NOT EXISTS idx_facts_status ON facts(validation_status);
        CREATE INDEX IF NOT EXISTS idx_facts_confidence ON facts(confidence);
        CREATE INDEX IF NOT EXISTS idx_facts_hash ON facts(statement_hash);
        CREATE INDEX IF NOT EXISTS idx_facts_created ON facts(created_at);
        CREATE INDEX IF NOT EXISTS idx_relations_source ON fact_relations(source_fact_id);
        CREATE INDEX IF NOT EXISTS idx_relations_target ON fact_relations(target_fact_id);
        CREATE INDEX IF NOT EXISTS idx_relations_type ON fact_relations(relation_type);
    """

    DEFAULT_DB_PATH = Path.home() / ".aragora" / "knowledge.db"

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the fact store.

        Args:
            db_path: Path to SQLite database (default: ~/.aragora/knowledge.db)
        """
        super().__init__(db_path=db_path or self.DEFAULT_DB_PATH)

    def _compute_statement_hash(self, statement: str) -> str:
        """Compute hash for statement deduplication."""
        normalized = " ".join(statement.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

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
        """Add a fact to the store.

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

        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Check for existing fact with same statement in workspace
            if deduplicate:
                cursor.execute(
                    """
                    SELECT * FROM facts
                    WHERE statement_hash = ? AND workspace_id = ?
                    """,
                    (statement_hash, workspace_id),
                )
                existing = cursor.fetchone()
                if existing:
                    logger.debug(f"Fact deduplicated: {existing['id']}")
                    return self._row_to_fact(existing)

            # Create new fact
            fact_id = f"fact_{uuid.uuid4().hex[:12]}"
            now = datetime.now()

            cursor.execute(
                """
                INSERT INTO facts (
                    id, statement, statement_hash, confidence,
                    evidence_ids_json, source_documents_json,
                    workspace_id, validation_status, topics_json,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
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
                    now.isoformat(),
                    now.isoformat(),
                ),
            )

            # Update FTS index
            topics_str = ",".join(topics)
            cursor.execute(
                """
                INSERT INTO facts_fts (fact_id, statement, topics)
                VALUES (?, ?, ?)
                """,
                (fact_id, statement, topics_str),
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

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        """Get a fact by ID.

        Args:
            fact_id: Fact ID

        Returns:
            Fact or None if not found
        """
        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM facts WHERE id = ?", (fact_id,))
            row = cursor.fetchone()

            if not row:
                return None

            return self._row_to_fact(row)

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
        """Update a fact.

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

        if confidence is not None:
            updates.append("confidence = ?")
            params.append(confidence)

        if validation_status is not None:
            updates.append("validation_status = ?")
            params.append(validation_status.value)

        if consensus_proof_id is not None:
            updates.append("consensus_proof_id = ?")
            params.append(consensus_proof_id)

        if evidence_ids is not None:
            updates.append("evidence_ids_json = ?")
            params.append(json.dumps(evidence_ids))

        if topics is not None:
            updates.append("topics_json = ?")
            params.append(json.dumps(topics))

        if metadata is not None:
            updates.append("metadata_json = ?")
            params.append(json.dumps(metadata))

        if superseded_by is not None:
            updates.append("superseded_by = ?")
            params.append(superseded_by)

        if not updates:
            return self.get_fact(fact_id)

        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(fact_id)

        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE facts SET {', '.join(updates)} WHERE id = ?",
                params,
            )

            if cursor.rowcount == 0:
                return None

            # Update FTS if topics changed
            if topics is not None:
                cursor.execute(
                    "DELETE FROM facts_fts WHERE fact_id = ?",
                    (fact_id,),
                )
                # Get statement for FTS re-index
                cursor.execute("SELECT statement FROM facts WHERE id = ?", (fact_id,))
                row = cursor.fetchone()
                if row:
                    cursor.execute(
                        "INSERT INTO facts_fts (fact_id, statement, topics) VALUES (?, ?, ?)",
                        (fact_id, row[0], ",".join(topics)),
                    )

        return self.get_fact(fact_id)

    def query_facts(
        self,
        query: str,
        filters: Optional[FactFilters] = None,
    ) -> list[Fact]:
        """Search facts using full-text search.

        Args:
            query: Search query
            filters: Optional filters to apply

        Returns:
            List of matching facts
        """
        filters = filters or FactFilters()
        sanitized = sanitize_fts_query(query)

        if not sanitized:
            return self.list_facts(filters)

        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            sql = """
                SELECT f.*, bm25(facts_fts) as fts_rank
                FROM facts_fts
                JOIN facts f ON facts_fts.fact_id = f.id
                WHERE facts_fts MATCH ?
            """
            params: list[Any] = [sanitized]

            # Apply filters
            if filters.workspace_id:
                sql += " AND f.workspace_id = ?"
                params.append(filters.workspace_id)

            if filters.min_confidence > 0:
                sql += " AND f.confidence >= ?"
                params.append(filters.min_confidence)

            if filters.validation_status:
                sql += " AND f.validation_status = ?"
                params.append(filters.validation_status.value)

            if not filters.include_superseded:
                sql += " AND f.superseded_by IS NULL"

            if filters.source_documents:
                # JSON array containment check
                for doc_id in filters.source_documents:
                    sql += " AND f.source_documents_json LIKE ?"
                    params.append(f'%"{doc_id}"%')

            if filters.created_after:
                sql += " AND f.created_at >= ?"
                params.append(filters.created_after.isoformat())

            if filters.created_before:
                sql += " AND f.created_at <= ?"
                params.append(filters.created_before.isoformat())

            sql += " ORDER BY fts_rank"
            sql += f" LIMIT ? OFFSET ?"
            params.extend([filters.limit, filters.offset])

            cursor.execute(sql, params)
            return [self._row_to_fact(row) for row in cursor.fetchall()]

    def list_facts(self, filters: Optional[FactFilters] = None) -> list[Fact]:
        """List facts with optional filtering.

        Args:
            filters: Optional filters

        Returns:
            List of facts
        """
        filters = filters or FactFilters()

        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            sql = "SELECT * FROM facts WHERE 1=1"
            params: list[Any] = []

            if filters.workspace_id:
                sql += " AND workspace_id = ?"
                params.append(filters.workspace_id)

            if filters.min_confidence > 0:
                sql += " AND confidence >= ?"
                params.append(filters.min_confidence)

            if filters.validation_status:
                sql += " AND validation_status = ?"
                params.append(filters.validation_status.value)

            if not filters.include_superseded:
                sql += " AND superseded_by IS NULL"

            if filters.topics:
                for topic in filters.topics:
                    sql += " AND topics_json LIKE ?"
                    params.append(f'%"{topic}"%')

            if filters.source_documents:
                for doc_id in filters.source_documents:
                    sql += " AND source_documents_json LIKE ?"
                    params.append(f'%"{doc_id}"%')

            if filters.created_after:
                sql += " AND created_at >= ?"
                params.append(filters.created_after.isoformat())

            if filters.created_before:
                sql += " AND created_at <= ?"
                params.append(filters.created_before.isoformat())

            sql += " ORDER BY confidence DESC, created_at DESC"
            sql += f" LIMIT ? OFFSET ?"
            params.extend([filters.limit, filters.offset])

            cursor.execute(sql, params)
            return [self._row_to_fact(row) for row in cursor.fetchall()]

    def get_contradictions(self, fact_id: str) -> list[Fact]:
        """Get facts that contradict a given fact.

        Args:
            fact_id: Fact to find contradictions for

        Returns:
            List of contradicting facts
        """
        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get facts linked by contradiction relation
            cursor.execute(
                """
                SELECT f.* FROM facts f
                JOIN fact_relations r ON (
                    (r.source_fact_id = ? AND r.target_fact_id = f.id)
                    OR (r.target_fact_id = ? AND r.source_fact_id = f.id)
                )
                WHERE r.relation_type = ?
                AND f.superseded_by IS NULL
                """,
                (fact_id, fact_id, FactRelationType.CONTRADICTS.value),
            )

            return [self._row_to_fact(row) for row in cursor.fetchall()]

    def add_relation(
        self,
        source_fact_id: str,
        target_fact_id: str,
        relation_type: FactRelationType,
        confidence: float = 0.5,
        created_by: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> FactRelation:
        """Add a relation between facts.

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
        now = datetime.now()

        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO fact_relations (
                    id, source_fact_id, target_fact_id, relation_type,
                    confidence, created_by, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    relation_id,
                    source_fact_id,
                    target_fact_id,
                    relation_type.value,
                    confidence,
                    created_by,
                    json.dumps(metadata or {}),
                    now.isoformat(),
                ),
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

    def get_relations(
        self,
        fact_id: str,
        relation_type: Optional[FactRelationType] = None,
        as_source: bool = True,
        as_target: bool = True,
    ) -> list[FactRelation]:
        """Get relations for a fact.

        Args:
            fact_id: Fact ID
            relation_type: Optional filter by type
            as_source: Include relations where fact is source
            as_target: Include relations where fact is target

        Returns:
            List of relations
        """
        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            conditions = []
            params: list[Any] = []

            if as_source:
                conditions.append("source_fact_id = ?")
                params.append(fact_id)
            if as_target:
                conditions.append("target_fact_id = ?")
                params.append(fact_id)

            if not conditions:
                return []

            sql = f"SELECT * FROM fact_relations WHERE ({' OR '.join(conditions)})"

            if relation_type:
                sql += " AND relation_type = ?"
                params.append(relation_type.value)

            cursor.execute(sql, params)

            relations = []
            for row in cursor.fetchall():
                relations.append(
                    FactRelation(
                        id=row["id"],
                        source_fact_id=row["source_fact_id"],
                        target_fact_id=row["target_fact_id"],
                        relation_type=FactRelationType(row["relation_type"]),
                        confidence=row["confidence"],
                        created_by=row["created_by"] or "",
                        metadata=json.loads(row["metadata_json"] or "{}"),
                        created_at=datetime.fromisoformat(row["created_at"]),
                    )
                )
            return relations

    def delete_fact(self, fact_id: str) -> bool:
        """Delete a fact and its relations.

        Args:
            fact_id: Fact to delete

        Returns:
            True if deleted
        """
        with self.connection() as conn:
            cursor = conn.cursor()

            # Delete from FTS
            cursor.execute("DELETE FROM facts_fts WHERE fact_id = ?", (fact_id,))

            # Delete relations
            cursor.execute(
                "DELETE FROM fact_relations WHERE source_fact_id = ? OR target_fact_id = ?",
                (fact_id, fact_id),
            )

            # Delete fact
            cursor.execute("DELETE FROM facts WHERE id = ?", (fact_id,))

            return cursor.rowcount > 0

    def get_statistics(self, workspace_id: Optional[str] = None) -> dict[str, Any]:
        """Get store statistics.

        Args:
            workspace_id: Optional workspace filter

        Returns:
            Statistics dictionary
        """
        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            where = ""
            params: list[Any] = []
            if workspace_id:
                where = " WHERE workspace_id = ?"
                params = [workspace_id]

            # Total facts
            cursor.execute(f"SELECT COUNT(*) as count FROM facts{where}", params)
            total = cursor.fetchone()["count"]

            # By status
            cursor.execute(
                f"""
                SELECT validation_status, COUNT(*) as count
                FROM facts{where}
                GROUP BY validation_status
                """,
                params,
            )
            by_status = {row["validation_status"]: row["count"] for row in cursor.fetchall()}

            # Average confidence
            cursor.execute(
                f"SELECT AVG(confidence) as avg FROM facts{where}",
                params,
            )
            avg_confidence = cursor.fetchone()["avg"] or 0.0

            # Verified facts
            verified_statuses = (
                ValidationStatus.MAJORITY_AGREED.value,
                ValidationStatus.BYZANTINE_AGREED.value,
                ValidationStatus.FORMALLY_PROVEN.value,
            )
            placeholders = ",".join("?" * len(verified_statuses))
            cursor.execute(
                f"""
                SELECT COUNT(*) as count FROM facts
                WHERE validation_status IN ({placeholders})
                {' AND workspace_id = ?' if workspace_id else ''}
                """,
                list(verified_statuses) + ([workspace_id] if workspace_id else []),
            )
            verified = cursor.fetchone()["count"]

            # Relations count
            cursor.execute("SELECT COUNT(*) as count FROM fact_relations")
            relations = cursor.fetchone()["count"]

            return {
                "total_facts": total,
                "verified_facts": verified,
                "by_status": by_status,
                "average_confidence": avg_confidence,
                "total_relations": relations,
            }

    def _row_to_fact(self, row: sqlite3.Row) -> Fact:
        """Convert database row to Fact object."""
        return Fact(
            id=row["id"],
            statement=row["statement"],
            confidence=row["confidence"],
            evidence_ids=json.loads(row["evidence_ids_json"] or "[]"),
            consensus_proof_id=row["consensus_proof_id"],
            source_documents=json.loads(row["source_documents_json"] or "[]"),
            workspace_id=row["workspace_id"],
            validation_status=ValidationStatus(row["validation_status"]),
            topics=json.loads(row["topics_json"] or "[]"),
            metadata=json.loads(row["metadata_json"] or "{}"),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            superseded_by=row["superseded_by"],
        )


class InMemoryFactStore:
    """In-memory fact store for testing."""

    def __init__(self):
        """Initialize in-memory store."""
        self._facts: dict[str, Fact] = {}
        self._relations: dict[str, FactRelation] = {}
        self._statement_hashes: dict[str, str] = {}  # hash -> fact_id

    def _compute_hash(self, statement: str, workspace_id: str) -> str:
        """Compute statement hash for deduplication."""
        normalized = " ".join(statement.lower().split())
        return hashlib.sha256(f"{normalized}:{workspace_id}".encode()).hexdigest()[:32]

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
        """Add a fact to memory."""
        stmt_hash = self._compute_hash(statement, workspace_id)

        if deduplicate and stmt_hash in self._statement_hashes:
            return self._facts[self._statement_hashes[stmt_hash]]

        fact_id = f"fact_{uuid.uuid4().hex[:12]}"
        now = datetime.now()

        fact = Fact(
            id=fact_id,
            statement=statement,
            confidence=confidence,
            evidence_ids=evidence_ids or [],
            source_documents=source_documents or [],
            workspace_id=workspace_id,
            validation_status=validation_status,
            topics=topics or [],
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )

        self._facts[fact_id] = fact
        self._statement_hashes[stmt_hash] = fact_id

        return fact

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        """Get fact by ID."""
        return self._facts.get(fact_id)

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
        """Update a fact."""
        fact = self._facts.get(fact_id)
        if not fact:
            return None

        if confidence is not None:
            fact.confidence = confidence
        if validation_status is not None:
            fact.validation_status = validation_status
        if consensus_proof_id is not None:
            fact.consensus_proof_id = consensus_proof_id
        if evidence_ids is not None:
            fact.evidence_ids = evidence_ids
        if topics is not None:
            fact.topics = topics
        if metadata is not None:
            fact.metadata = metadata
        if superseded_by is not None:
            fact.superseded_by = superseded_by

        fact.updated_at = datetime.now()
        return fact

    def query_facts(
        self,
        query: str,
        filters: Optional[FactFilters] = None,
    ) -> list[Fact]:
        """Search facts by keyword."""
        filters = filters or FactFilters()
        query_lower = query.lower()

        results = []
        for fact in self._facts.values():
            if filters.workspace_id and fact.workspace_id != filters.workspace_id:
                continue
            if fact.confidence < filters.min_confidence:
                continue
            if filters.validation_status and fact.validation_status != filters.validation_status:
                continue
            if not filters.include_superseded and fact.superseded_by:
                continue

            # Keyword match
            if query_lower in fact.statement.lower():
                results.append(fact)
            elif any(query_lower in t.lower() for t in fact.topics):
                results.append(fact)

        results.sort(key=lambda f: f.confidence, reverse=True)
        return results[filters.offset : filters.offset + filters.limit]

    def list_facts(self, filters: Optional[FactFilters] = None) -> list[Fact]:
        """List facts."""
        filters = filters or FactFilters()

        results = []
        for fact in self._facts.values():
            if filters.workspace_id and fact.workspace_id != filters.workspace_id:
                continue
            if fact.confidence < filters.min_confidence:
                continue
            if filters.validation_status and fact.validation_status != filters.validation_status:
                continue
            if not filters.include_superseded and fact.superseded_by:
                continue

            results.append(fact)

        results.sort(key=lambda f: (f.confidence, f.created_at), reverse=True)
        return results[filters.offset : filters.offset + filters.limit]

    def get_contradictions(self, fact_id: str) -> list[Fact]:
        """Get contradicting facts."""
        contradictions = []
        for rel in self._relations.values():
            if rel.relation_type != FactRelationType.CONTRADICTS:
                continue
            if rel.source_fact_id == fact_id:
                target = self._facts.get(rel.target_fact_id)
                if target and not target.superseded_by:
                    contradictions.append(target)
            elif rel.target_fact_id == fact_id:
                source = self._facts.get(rel.source_fact_id)
                if source and not source.superseded_by:
                    contradictions.append(source)
        return contradictions

    def add_relation(
        self,
        source_fact_id: str,
        target_fact_id: str,
        relation_type: FactRelationType,
        confidence: float = 0.5,
        created_by: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> FactRelation:
        """Add a relation."""
        relation_id = f"rel_{uuid.uuid4().hex[:12]}"
        relation = FactRelation(
            id=relation_id,
            source_fact_id=source_fact_id,
            target_fact_id=target_fact_id,
            relation_type=relation_type,
            confidence=confidence,
            created_by=created_by,
            metadata=metadata or {},
            created_at=datetime.now(),
        )
        self._relations[relation_id] = relation
        return relation

    def get_relations(
        self,
        fact_id: str,
        relation_type: Optional[FactRelationType] = None,
        as_source: bool = True,
        as_target: bool = True,
    ) -> list[FactRelation]:
        """Get relations for a fact."""
        results = []
        for rel in self._relations.values():
            if relation_type and rel.relation_type != relation_type:
                continue
            if as_source and rel.source_fact_id == fact_id:
                results.append(rel)
            elif as_target and rel.target_fact_id == fact_id:
                results.append(rel)
        return results

    def delete_fact(self, fact_id: str) -> bool:
        """Delete a fact."""
        if fact_id not in self._facts:
            return False

        fact = self._facts.pop(fact_id)

        # Remove from hash index
        stmt_hash = self._compute_hash(fact.statement, fact.workspace_id)
        self._statement_hashes.pop(stmt_hash, None)

        # Remove relations
        to_remove = [
            rid
            for rid, rel in self._relations.items()
            if rel.source_fact_id == fact_id or rel.target_fact_id == fact_id
        ]
        for rid in to_remove:
            del self._relations[rid]

        return True

    def get_statistics(self, workspace_id: Optional[str] = None) -> dict[str, Any]:
        """Get statistics."""
        facts = list(self._facts.values())
        if workspace_id:
            facts = [f for f in facts if f.workspace_id == workspace_id]

        by_status: dict[str, int] = {}
        total_confidence = 0.0
        verified = 0

        for fact in facts:
            status = fact.validation_status.value
            by_status[status] = by_status.get(status, 0) + 1
            total_confidence += fact.confidence
            if fact.is_verified:
                verified += 1

        return {
            "total_facts": len(facts),
            "verified_facts": verified,
            "by_status": by_status,
            "average_confidence": total_confidence / len(facts) if facts else 0.0,
            "total_relations": len(self._relations),
        }

    def close(self) -> None:
        """No-op for in-memory store."""
        pass
