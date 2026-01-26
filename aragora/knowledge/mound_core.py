"""
Knowledge Mound - Unified knowledge storage with vector + graph capabilities.

Implements the "termite mound" architecture where agents contribute to a shared
superstructure of knowledge. Unifies ContinuumMemory, ConsensusMemory, and FactStore
into a coherent knowledge graph with semantic search.

Key concepts:
- KnowledgeNode: A unit of knowledge (fact, claim, memory, evidence, consensus)
- ProvenanceChain: Tracks origin and transformations of knowledge
- Graph relationships: supports, contradicts, derived_from for knowledge traversal

Usage:
    from aragora.knowledge.mound import KnowledgeMound, KnowledgeNode

    mound = KnowledgeMound(workspace_id="ws_123")
    await mound.initialize()

    # Add knowledge
    node = KnowledgeNode(
        node_type="fact",
        content="API keys should never be committed to version control",
        confidence=0.95,
    )
    node_id = await mound.add_node(node)

    # Query semantically
    results = await mound.query_semantic("security best practices", limit=10)

    # Traverse graph
    related = await mound.query_graph(node_id, "supports", depth=2)
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

from aragora.persistence.db_config import DatabaseType, get_db_path
from aragora.knowledge.types import (
    Fact,
    ValidationStatus,
)
from aragora.memory.tier_manager import MemoryTier
from aragora.storage.base_store import SQLiteStore
from aragora.utils.json_helpers import safe_json_loads

logger = logging.getLogger(__name__)


def _to_iso_string(value: Any) -> Optional[str]:
    """Safely convert datetime or string to ISO format string.

    Handles both datetime objects and ISO format strings to ensure
    consistent serialization regardless of how the value was stored.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value  # Already an ISO string
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _to_enum_value(value: Any) -> Any:
    """Safely extract value from enum or return string as-is.

    Handles both enum instances and raw string values to ensure
    consistent serialization regardless of how the value was stored.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value  # Already a string value
    if hasattr(value, "value"):
        return value.value
    return str(value)


# Type alias for node types
NodeType = Literal["fact", "claim", "memory", "evidence", "consensus", "entity"]

# Type alias for relationship types
RelationshipType = Literal["supports", "contradicts", "derived_from", "related_to", "supersedes"]


class ProvenanceType(Enum):
    """Source types for knowledge provenance."""

    DOCUMENT = "document"  # Extracted from document
    DEBATE = "debate"  # Result of multi-agent debate
    USER = "user"  # User-provided knowledge
    AGENT = "agent"  # Agent-generated inference
    INFERENCE = "inference"  # Derived from other knowledge
    MIGRATION = "migration"  # Migrated from legacy system


@dataclass
class ProvenanceChain:
    """
    Tracks the origin and transformations of knowledge.

    Provides full traceability for audit and compliance, recording
    where knowledge came from and how it has been modified.
    """

    source_type: ProvenanceType
    source_id: str  # ID of the source (document, debate, user, etc.)
    agent_id: Optional[str] = None
    debate_id: Optional[str] = None
    document_id: Optional[str] = None
    user_id: Optional[str] = None
    transformations: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def add_transformation(
        self,
        transform_type: str,
        agent_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record a transformation to this knowledge."""
        self.transformations.append(
            {
                "type": transform_type,
                "agent_id": agent_id,
                "details": details or {},
                "timestamp": datetime.now().isoformat(),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_type": self.source_type.value,
            "source_id": self.source_id,
            "agent_id": self.agent_id,
            "debate_id": self.debate_id,
            "document_id": self.document_id,
            "user_id": self.user_id,
            "transformations": self.transformations,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProvenanceChain":
        """Create from dictionary."""
        return cls(
            source_type=ProvenanceType(data["source_type"]),
            source_id=data["source_id"],
            agent_id=data.get("agent_id"),
            debate_id=data.get("debate_id"),
            document_id=data.get("document_id"),
            user_id=data.get("user_id"),
            transformations=data.get("transformations", []),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if isinstance(data.get("created_at"), str)
                else datetime.now()
            ),
        )


@dataclass
class KnowledgeNode:
    """
    A node in the knowledge mound (fact, claim, memory, evidence, consensus).

    Central data structure unifying the various memory systems. Each node
    can have relationships with other nodes, forming a knowledge graph.
    """

    id: str = ""
    node_type: NodeType = "fact"
    content: str = ""
    confidence: float = 0.5
    provenance: Optional[ProvenanceChain] = None
    tier: MemoryTier = MemoryTier.SLOW
    workspace_id: str = ""

    # Surprise-based learning (from ContinuumMemory)
    surprise_score: float = 0.0
    update_count: int = 1
    consolidation_score: float = 0.0

    # Validation (from FactStore)
    validation_status: ValidationStatus = ValidationStatus.UNVERIFIED
    consensus_proof_id: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    topics: list[str] = field(default_factory=list)
    embedding: Optional[list[float]] = None

    # Graph relationships (stored separately, referenced here for convenience)
    supports: list[str] = field(default_factory=list)  # Node IDs this supports
    contradicts: list[str] = field(default_factory=list)  # Node IDs this contradicts
    derived_from: list[str] = field(default_factory=list)  # Source node IDs

    def __post_init__(self) -> None:
        """Generate ID if not provided."""
        if not self.id:
            self.id = f"kn_{uuid.uuid4().hex[:16]}"

    @property
    def content_hash(self) -> str:
        """Hash of content for deduplication."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:32]

    @property
    def is_verified(self) -> bool:
        """Check if node has been verified."""
        return self.validation_status in (
            ValidationStatus.MAJORITY_AGREED,
            ValidationStatus.BYZANTINE_AGREED,
            ValidationStatus.FORMALLY_PROVEN,
        )

    @property
    def stability_score(self) -> float:
        """Inverse of surprise - how predictable this pattern is."""
        return 1.0 - self.surprise_score

    def should_promote(self) -> bool:
        """Check if this node should be promoted to a faster tier."""
        if self.tier == MemoryTier.FAST:
            return False
        # High surprise score indicates this knowledge is actively relevant
        return self.surprise_score > 0.7 and self.update_count > 5

    def should_demote(self) -> bool:
        """Check if this node should be demoted to a slower tier."""
        if self.tier == MemoryTier.GLACIAL:
            return False
        # Low surprise + high consolidation = stable knowledge
        return self.stability_score > 0.8 and self.consolidation_score > 0.5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "node_type": self.node_type,
            "content": self.content,
            "confidence": self.confidence,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "tier": self.tier.value,
            "workspace_id": self.workspace_id,
            "surprise_score": self.surprise_score,
            "update_count": self.update_count,
            "consolidation_score": self.consolidation_score,
            "validation_status": self.validation_status.value,
            "consensus_proof_id": self.consensus_proof_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "topics": self.topics,
            "supports": self.supports,
            "contradicts": self.contradicts,
            "derived_from": self.derived_from,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeNode":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            node_type=data.get("node_type", "fact"),
            content=data["content"],
            confidence=data.get("confidence", 0.5),
            provenance=(
                ProvenanceChain.from_dict(data["provenance"]) if data.get("provenance") else None
            ),
            tier=MemoryTier(data.get("tier", "slow")),
            workspace_id=data.get("workspace_id", ""),
            surprise_score=data.get("surprise_score", 0.0),
            update_count=data.get("update_count", 1),
            consolidation_score=data.get("consolidation_score", 0.0),
            validation_status=ValidationStatus(data.get("validation_status", "unverified")),
            consensus_proof_id=data.get("consensus_proof_id"),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if isinstance(data.get("created_at"), str)
                else datetime.now()
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if isinstance(data.get("updated_at"), str)
                else datetime.now()
            ),
            metadata=data.get("metadata", {}),
            topics=data.get("topics", []),
            supports=data.get("supports", []),
            contradicts=data.get("contradicts", []),
            derived_from=data.get("derived_from", []),
        )

    @classmethod
    def from_fact(cls, fact: Fact, workspace_id: str = "") -> "KnowledgeNode":
        """Create KnowledgeNode from existing Fact."""
        return cls(
            id=f"kn_{fact.id}",
            node_type="fact",
            content=fact.statement,
            confidence=fact.confidence,
            provenance=ProvenanceChain(
                source_type=ProvenanceType.DOCUMENT,
                source_id=fact.source_documents[0] if fact.source_documents else "",
                document_id=fact.source_documents[0] if fact.source_documents else None,
            ),
            workspace_id=workspace_id or fact.workspace_id,
            validation_status=fact.validation_status,
            consensus_proof_id=fact.consensus_proof_id,
            created_at=fact.created_at,
            updated_at=fact.updated_at,
            metadata=fact.metadata,
            topics=fact.topics,
        )


@dataclass
class KnowledgeRelationship:
    """
    A relationship between two knowledge nodes.

    Enables graph traversal and knowledge inference.
    """

    id: str = ""
    from_node_id: str = ""
    to_node_id: str = ""
    relationship_type: RelationshipType = "related_to"
    strength: float = 1.0
    created_by: str = ""  # Agent or user who established relation
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Generate ID if not provided."""
        if not self.id:
            self.id = f"kr_{uuid.uuid4().hex[:16]}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "from_node_id": self.from_node_id,
            "to_node_id": self.to_node_id,
            "relationship_type": self.relationship_type,
            "strength": self.strength,
            "created_by": self.created_by,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeRelationship":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            from_node_id=data["from_node_id"],
            to_node_id=data["to_node_id"],
            relationship_type=data.get("relationship_type", "related_to"),
            strength=data.get("strength", 1.0),
            created_by=data.get("created_by", ""),
            metadata=data.get("metadata", {}),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if isinstance(data.get("created_at"), str)
                else datetime.now()
            ),
        )


@dataclass
class KnowledgeQueryResult:
    """Result of querying the knowledge mound."""

    nodes: list[KnowledgeNode]
    total_count: int
    query: str
    processing_time_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeMoundMetaStore(SQLiteStore):
    """
    SQLite store for Knowledge Mound metadata and relationships.

    Handles fast queries, relationship traversal, and metadata storage.
    Vector embeddings are stored in Weaviate for semantic search.
    """

    SCHEMA_NAME = "knowledge_mound"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        -- Knowledge nodes (core metadata)
        CREATE TABLE IF NOT EXISTS knowledge_nodes (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL,
            node_type TEXT NOT NULL,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,
            tier TEXT DEFAULT 'slow',
            surprise_score REAL DEFAULT 0.0,
            update_count INTEGER DEFAULT 1,
            consolidation_score REAL DEFAULT 0.0,
            validation_status TEXT DEFAULT 'unverified',
            consensus_proof_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            metadata TEXT DEFAULT '{}'
        );

        -- Provenance tracking
        CREATE TABLE IF NOT EXISTS provenance (
            id TEXT PRIMARY KEY,
            node_id TEXT NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
            source_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            agent_id TEXT,
            debate_id TEXT,
            document_id TEXT,
            user_id TEXT,
            transformations TEXT DEFAULT '[]',
            created_at TEXT NOT NULL
        );

        -- Graph relationships
        CREATE TABLE IF NOT EXISTS relationships (
            id TEXT PRIMARY KEY,
            from_node_id TEXT NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
            to_node_id TEXT NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
            relationship_type TEXT NOT NULL,
            strength REAL DEFAULT 1.0,
            created_by TEXT,
            metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL
        );

        -- Topics for categorization
        CREATE TABLE IF NOT EXISTS node_topics (
            node_id TEXT NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
            topic TEXT NOT NULL,
            PRIMARY KEY (node_id, topic)
        );

        -- Indexes for efficient queries
        CREATE INDEX IF NOT EXISTS idx_nodes_workspace ON knowledge_nodes(workspace_id);
        CREATE INDEX IF NOT EXISTS idx_nodes_type ON knowledge_nodes(node_type);
        CREATE INDEX IF NOT EXISTS idx_nodes_tier ON knowledge_nodes(tier);
        CREATE INDEX IF NOT EXISTS idx_nodes_confidence ON knowledge_nodes(confidence DESC);
        CREATE INDEX IF NOT EXISTS idx_nodes_content_hash ON knowledge_nodes(content_hash);
        CREATE INDEX IF NOT EXISTS idx_nodes_validation ON knowledge_nodes(validation_status);
        CREATE INDEX IF NOT EXISTS idx_relationships_from ON relationships(from_node_id);
        CREATE INDEX IF NOT EXISTS idx_relationships_to ON relationships(to_node_id);
        CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relationship_type);
        CREATE INDEX IF NOT EXISTS idx_provenance_node ON provenance(node_id);
        CREATE INDEX IF NOT EXISTS idx_topics_topic ON node_topics(topic);
    """

    def save_node(self, node: KnowledgeNode) -> str:
        """Save a knowledge node."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO knowledge_nodes (
                    id, workspace_id, node_type, content, content_hash,
                    confidence, tier, surprise_score, update_count,
                    consolidation_score, validation_status, consensus_proof_id,
                    created_at, updated_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node.id,
                    node.workspace_id,
                    node.node_type,
                    node.content,
                    node.content_hash,
                    node.confidence,
                    _to_enum_value(node.tier),
                    node.surprise_score,
                    node.update_count,
                    node.consolidation_score,
                    _to_enum_value(node.validation_status),
                    node.consensus_proof_id,
                    _to_iso_string(node.created_at),
                    _to_iso_string(node.updated_at),
                    json.dumps(node.metadata),
                ),
            )

            # Save provenance
            if node.provenance:
                prov_id = f"prov_{node.id}"
                conn.execute(
                    """
                    INSERT OR REPLACE INTO provenance (
                        id, node_id, source_type, source_id, agent_id,
                        debate_id, document_id, user_id, transformations, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        prov_id,
                        node.id,
                        _to_enum_value(node.provenance.source_type),
                        node.provenance.source_id,
                        node.provenance.agent_id,
                        node.provenance.debate_id,
                        node.provenance.document_id,
                        node.provenance.user_id,
                        json.dumps(node.provenance.transformations),
                        _to_iso_string(node.provenance.created_at),
                    ),
                )

            # Save topics
            conn.execute("DELETE FROM node_topics WHERE node_id = ?", (node.id,))
            for topic in node.topics:
                conn.execute(
                    "INSERT OR IGNORE INTO node_topics (node_id, topic) VALUES (?, ?)",
                    (node.id, topic),
                )

        return node.id

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get a knowledge node by ID."""
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM knowledge_nodes WHERE id = ?", (node_id,)).fetchone()
            if not row:
                return None

            # Get provenance
            prov_row = conn.execute(
                "SELECT * FROM provenance WHERE node_id = ?", (node_id,)
            ).fetchone()
            provenance = None
            if prov_row:
                provenance = ProvenanceChain(
                    source_type=ProvenanceType(prov_row["source_type"]),
                    source_id=prov_row["source_id"],
                    agent_id=prov_row["agent_id"],
                    debate_id=prov_row["debate_id"],
                    document_id=prov_row["document_id"],
                    user_id=prov_row["user_id"],
                    transformations=safe_json_loads(prov_row["transformations"], []),
                    created_at=datetime.fromisoformat(prov_row["created_at"]),
                )

            # Get topics
            topics = [
                r["topic"]
                for r in conn.execute(
                    "SELECT topic FROM node_topics WHERE node_id = ?", (node_id,)
                ).fetchall()
            ]

            # Get relationships
            supports = [
                r["to_node_id"]
                for r in conn.execute(
                    "SELECT to_node_id FROM relationships WHERE from_node_id = ? AND relationship_type = 'supports'",
                    (node_id,),
                ).fetchall()
            ]
            contradicts = [
                r["to_node_id"]
                for r in conn.execute(
                    "SELECT to_node_id FROM relationships WHERE from_node_id = ? AND relationship_type = 'contradicts'",
                    (node_id,),
                ).fetchall()
            ]
            derived_from = [
                r["to_node_id"]
                for r in conn.execute(
                    "SELECT to_node_id FROM relationships WHERE from_node_id = ? AND relationship_type = 'derived_from'",
                    (node_id,),
                ).fetchall()
            ]

            return KnowledgeNode(
                id=row["id"],
                node_type=row["node_type"],
                content=row["content"],
                confidence=row["confidence"],
                provenance=provenance,
                tier=MemoryTier(row["tier"]),
                workspace_id=row["workspace_id"],
                surprise_score=row["surprise_score"],
                update_count=row["update_count"],
                consolidation_score=row["consolidation_score"],
                validation_status=ValidationStatus(row["validation_status"]),
                consensus_proof_id=row["consensus_proof_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                metadata=safe_json_loads(row["metadata"], {}),
                topics=topics,
                supports=supports,
                contradicts=contradicts,
                derived_from=derived_from,
            )

    def save_relationship(self, rel: KnowledgeRelationship) -> str:
        """Save a relationship between nodes."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO relationships (
                    id, from_node_id, to_node_id, relationship_type,
                    strength, created_by, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rel.id,
                    rel.from_node_id,
                    rel.to_node_id,
                    rel.relationship_type,
                    rel.strength,
                    rel.created_by,
                    json.dumps(rel.metadata),
                    rel.created_at.isoformat(),
                ),
            )
        return rel.id

    def get_relationships(
        self,
        node_id: str,
        relationship_type: Optional[RelationshipType] = None,
        direction: Literal["outgoing", "incoming", "both"] = "outgoing",
    ) -> list[KnowledgeRelationship]:
        """Get relationships for a node."""
        relationships = []
        with self.connection() as conn:
            if direction in ("outgoing", "both"):
                query = "SELECT * FROM relationships WHERE from_node_id = ?"
                params: list[Any] = [node_id]
                if relationship_type:
                    query += " AND relationship_type = ?"
                    params.append(relationship_type)
                for row in conn.execute(query, params).fetchall():
                    relationships.append(KnowledgeRelationship.from_dict(dict(row)))

            if direction in ("incoming", "both"):
                query = "SELECT * FROM relationships WHERE to_node_id = ?"
                params = [node_id]
                if relationship_type:
                    query += " AND relationship_type = ?"
                    params.append(relationship_type)
                for row in conn.execute(query, params).fetchall():
                    relationships.append(KnowledgeRelationship.from_dict(dict(row)))

        return relationships

    def query_nodes(
        self,
        workspace_id: Optional[str] = None,
        node_types: Optional[list[NodeType]] = None,
        min_confidence: float = 0.0,
        tier: Optional[MemoryTier] = None,
        validation_status: Optional[ValidationStatus] = None,
        topics: Optional[list[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[KnowledgeNode]:
        """Query nodes with filters."""
        conditions = ["1=1"]
        params: list[Any] = []

        if workspace_id:
            conditions.append("workspace_id = ?")
            params.append(workspace_id)
        if node_types:
            placeholders = ",".join("?" * len(node_types))
            conditions.append(f"node_type IN ({placeholders})")
            params.extend(node_types)
        if min_confidence > 0:
            conditions.append("confidence >= ?")
            params.append(min_confidence)
        if tier:
            conditions.append("tier = ?")
            params.append(tier.value)
        if validation_status:
            conditions.append("validation_status = ?")
            params.append(validation_status.value)

        where_clause = " AND ".join(conditions)

        # Handle topic filtering with join
        if topics:
            topic_placeholders = ",".join("?" * len(topics))
            query = f"""
                SELECT DISTINCT kn.id FROM knowledge_nodes kn
                JOIN node_topics nt ON kn.id = nt.node_id
                WHERE {where_clause} AND nt.topic IN ({topic_placeholders})
                ORDER BY kn.confidence DESC
                LIMIT ? OFFSET ?
            """
            params.extend(topics)
        else:
            query = f"""
                SELECT id FROM knowledge_nodes
                WHERE {where_clause}
                ORDER BY confidence DESC
                LIMIT ? OFFSET ?
            """

        params.extend([limit, offset])

        nodes = []
        with self.connection() as conn:
            for row in conn.execute(query, params).fetchall():
                node = self.get_node(row["id"])
                if node:
                    nodes.append(node)

        return nodes

    def find_by_content_hash(self, content_hash: str, workspace_id: str) -> Optional[KnowledgeNode]:
        """Find node by content hash (for deduplication)."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT id FROM knowledge_nodes WHERE content_hash = ? AND workspace_id = ?",
                (content_hash, workspace_id),
            ).fetchone()
            if row:
                return self.get_node(row["id"])
        return None

    def delete_node(self, node_id: str) -> bool:
        """Delete a knowledge node and its relationships."""
        with self.connection() as conn:
            # Delete relationships
            conn.execute(
                "DELETE FROM relationships WHERE from_node_id = ? OR to_node_id = ?",
                (node_id, node_id),
            )
            # Delete provenance
            conn.execute("DELETE FROM provenance WHERE node_id = ?", (node_id,))
            # Delete topics
            conn.execute("DELETE FROM node_topics WHERE node_id = ?", (node_id,))
            # Delete node
            cursor = conn.execute("DELETE FROM knowledge_nodes WHERE id = ?", (node_id,))
            return cursor.rowcount > 0

    def query_by_provenance(
        self,
        source_type: Optional[str] = None,
        source_id: Optional[str] = None,
        node_type: Optional[str] = None,
        workspace_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[str]:
        """Query nodes by provenance attributes."""
        conditions = ["1=1"]
        params: list[Any] = []

        if source_type:
            conditions.append("p.source_type = ?")
            params.append(source_type)
        if source_id:
            conditions.append("p.source_id = ?")
            params.append(source_id)
        if node_type:
            conditions.append("kn.node_type = ?")
            params.append(node_type)
        if workspace_id:
            conditions.append("kn.workspace_id = ?")
            params.append(workspace_id)

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT DISTINCT kn.id FROM knowledge_nodes kn
            JOIN provenance p ON kn.id = p.node_id
            WHERE {where_clause}
            ORDER BY kn.created_at DESC
            LIMIT ?
        """
        params.append(limit)

        with self.connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [row["id"] for row in rows]

    def get_stats(self, workspace_id: Optional[str] = None) -> dict[str, Any]:
        """Get statistics about the knowledge mound."""
        with self.connection() as conn:
            where = "WHERE workspace_id = ?" if workspace_id else ""
            params = [workspace_id] if workspace_id else []

            total = conn.execute(
                f"SELECT COUNT(*) as count FROM knowledge_nodes {where}", params
            ).fetchone()["count"]

            by_type = {}
            for row in conn.execute(
                f"SELECT node_type, COUNT(*) as count FROM knowledge_nodes {where} GROUP BY node_type",
                params,
            ).fetchall():
                by_type[row["node_type"]] = row["count"]

            by_tier = {}
            for row in conn.execute(
                f"SELECT tier, COUNT(*) as count FROM knowledge_nodes {where} GROUP BY tier",
                params,
            ).fetchall():
                by_tier[row["tier"]] = row["count"]

            by_validation = {}
            for row in conn.execute(
                f"SELECT validation_status, COUNT(*) as count FROM knowledge_nodes {where} GROUP BY validation_status",
                params,
            ).fetchall():
                by_validation[row["validation_status"]] = row["count"]

            avg_confidence = (
                conn.execute(
                    f"SELECT AVG(confidence) as avg FROM knowledge_nodes {where}", params
                ).fetchone()["avg"]
                or 0.0
            )

            relationship_count = conn.execute(
                "SELECT COUNT(*) as count FROM relationships"
            ).fetchone()["count"]

            return {
                "total_nodes": total,
                "by_type": by_type,
                "by_tier": by_tier,
                "by_validation_status": by_validation,
                "average_confidence": round(avg_confidence, 3),
                "total_relationships": relationship_count,
            }


class KnowledgeMound:
    """
    Unified knowledge storage with vector + graph capabilities.

    Combines:
    - SQLite for metadata, relationships, fast queries
    - Weaviate for vector embeddings and semantic search
    - Graph traversal for knowledge inference

    This is the "termite mound" - a shared superstructure where agents
    contribute knowledge that accumulates across tasks and sessions.
    """

    def __init__(
        self,
        workspace_id: str = "default",
        db_path: Optional[Union[str, Path]] = None,
        weaviate_config: Optional[dict[str, Any]] = None,
        embedding_fn: Optional[Callable[[str], list[float]]] = None,
    ):
        """
        Initialize the Knowledge Mound.

        Args:
            workspace_id: Workspace for multi-tenant isolation
            db_path: Path to SQLite database
            weaviate_config: Configuration for Weaviate vector store
            embedding_fn: Function to generate embeddings (defaults to internal)
        """
        self.workspace_id = workspace_id
        if db_path is None:
            knowledge_dir = get_db_path(DatabaseType.KNOWLEDGE).parent  # type: ignore[attr-defined]
            self._db_path = knowledge_dir / "mound.db"
        else:
            self._db_path = Path(db_path)
        self._weaviate_config = weaviate_config
        self._embedding_fn = embedding_fn

        # Initialize stores
        self._meta_store: Optional[KnowledgeMoundMetaStore] = None
        self._vector_store: Optional[Any] = None  # WeaviateStore when available
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the knowledge mound stores."""
        if self._initialized:
            return

        # Ensure directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite meta store
        self._meta_store = KnowledgeMoundMetaStore(self._db_path)

        # Try to initialize Weaviate if config provided
        if self._weaviate_config:
            try:
                from aragora.documents.indexing.weaviate_store import WeaviateStore, WeaviateConfig

                config = WeaviateConfig(**self._weaviate_config)
                self._vector_store = WeaviateStore(config)
                await self._vector_store.connect()
                logger.info("Knowledge Mound initialized with Weaviate vector store")
            except ImportError:
                logger.warning("Weaviate not available - using SQLite-only mode")
            except Exception as e:
                logger.warning(f"Failed to connect to Weaviate: {e} - using SQLite-only mode")

        self._initialized = True
        logger.info(f"Knowledge Mound initialized for workspace: {self.workspace_id}")

    def _ensure_initialized(self) -> None:
        """Ensure the mound is initialized."""
        if not self._initialized or not self._meta_store:
            raise RuntimeError("KnowledgeMound not initialized. Call initialize() first.")

    async def add_node(
        self,
        node: KnowledgeNode,
        deduplicate: bool = True,
    ) -> str:
        """
        Add a knowledge node to the mound.

        Args:
            node: The knowledge node to add
            deduplicate: If True, check for existing node with same content

        Returns:
            The node ID
        """
        self._ensure_initialized()
        assert self._meta_store is not None

        # Set workspace if not set
        if not node.workspace_id:
            node.workspace_id = self.workspace_id

        # Check for duplicates
        if deduplicate:
            existing = self._meta_store.find_by_content_hash(node.content_hash, node.workspace_id)
            if existing:
                # Update existing node
                existing.update_count += 1
                existing.updated_at = datetime.now()
                # Merge confidence (weighted average)
                existing.confidence = existing.confidence * 0.7 + node.confidence * 0.3
                node = existing

        # Save to SQLite
        self._meta_store.save_node(node)

        # Save embedding to Weaviate when vector store is available
        if self._vector_store and self._embedding_fn:
            try:
                embedding = self._embedding_fn(node.content)
                await self._vector_store.upsert(
                    id=node.id,
                    embedding=embedding,
                    content=node.content,
                    metadata={
                        "node_type": node.node_type,
                        "confidence": node.confidence,
                        "workspace_id": node.workspace_id,
                    },
                    namespace=node.workspace_id,
                )
            except Exception as e:
                logger.warning(f"Failed to save embedding to vector store: {e}")

        logger.debug(f"Added knowledge node: {node.id} ({node.node_type})")
        return node.id

    async def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get a knowledge node by ID."""
        self._ensure_initialized()
        assert self._meta_store is not None
        return self._meta_store.get_node(node_id)

    async def add_relationship(
        self,
        from_node_id: str,
        to_node_id: str,
        relationship_type: RelationshipType,
        strength: float = 1.0,
        created_by: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Add a relationship between two nodes.

        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID
            relationship_type: Type of relationship
            strength: Relationship strength (0-1)
            created_by: Agent/user who created relationship
            metadata: Additional metadata

        Returns:
            Relationship ID
        """
        self._ensure_initialized()
        assert self._meta_store is not None

        rel = KnowledgeRelationship(
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            relationship_type=relationship_type,
            strength=strength,
            created_by=created_by,
            metadata=metadata or {},
        )

        return self._meta_store.save_relationship(rel)

    async def query_semantic(
        self,
        query: str,
        limit: int = 10,
        node_types: Optional[list[NodeType]] = None,
        min_confidence: float = 0.0,
        workspace_id: Optional[str] = None,
    ) -> KnowledgeQueryResult:
        """
        Semantic search across the knowledge mound.

        Args:
            query: Natural language query
            limit: Maximum results
            node_types: Filter by node types
            min_confidence: Minimum confidence threshold
            workspace_id: Filter by workspace (defaults to self.workspace_id)

        Returns:
            Query result with matching nodes
        """
        self._ensure_initialized()
        assert self._meta_store is not None

        import time

        start = time.time()

        ws_id = workspace_id or self.workspace_id

        # Use Weaviate for semantic search when available
        if self._vector_store and self._embedding_fn:
            try:
                query_embedding = self._embedding_fn(query)
                if node_types:
                    # Note: Weaviate filters multiple types via OR, we'll filter post-search
                    pass

                vector_results = await self._vector_store.search(
                    embedding=query_embedding,
                    limit=limit * 2,  # Fetch extra for filtering
                    namespace=ws_id,
                    min_score=min_confidence,
                )

                # Get full nodes from SQLite and filter by type/confidence
                result_nodes = []
                for vr in vector_results:
                    node = self._meta_store.get_node(vr.id)
                    if node:
                        if node_types and node.node_type not in node_types:
                            continue
                        if node.confidence < min_confidence:
                            continue
                        result_nodes.append(node)
                        if len(result_nodes) >= limit:
                            break

                elapsed_ms = int((time.time() - start) * 1000)
                return KnowledgeQueryResult(
                    nodes=result_nodes,
                    total_count=len(result_nodes),
                    query=query,
                    processing_time_ms=elapsed_ms,
                )
            except Exception as e:
                logger.warning(f"Vector search failed, falling back to keyword: {e}")

        # Fall back to keyword-based search
        nodes = self._meta_store.query_nodes(
            workspace_id=ws_id,
            node_types=node_types,
            min_confidence=min_confidence,
            limit=limit,
        )

        # Simple keyword relevance scoring
        query_words = set(query.lower().split())
        scored_nodes = []
        for node in nodes:
            content_words = set(node.content.lower().split())
            overlap = len(query_words & content_words)
            score = overlap / max(len(query_words), 1)
            scored_nodes.append((score, node))

        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        result_nodes = [node for _, node in scored_nodes[:limit]]

        elapsed_ms = int((time.time() - start) * 1000)

        return KnowledgeQueryResult(
            nodes=result_nodes,
            total_count=len(result_nodes),
            query=query,
            processing_time_ms=elapsed_ms,
        )

    async def query_graph(
        self,
        start_node_id: str,
        relationship_type: Optional[RelationshipType] = None,
        depth: int = 2,
        direction: Literal["outgoing", "incoming", "both"] = "outgoing",
    ) -> list[KnowledgeNode]:
        """
        Graph traversal from a starting node.

        Args:
            start_node_id: Starting node ID
            relationship_type: Filter by relationship type
            depth: Maximum traversal depth
            direction: Direction of traversal

        Returns:
            List of connected nodes
        """
        self._ensure_initialized()
        assert self._meta_store is not None

        visited: set[str] = set()
        result: list[KnowledgeNode] = []

        async def traverse(node_id: str, current_depth: int) -> None:
            if current_depth > depth or node_id in visited:
                return

            visited.add(node_id)
            node = self._meta_store.get_node(node_id)
            if node:
                result.append(node)

            relationships = self._meta_store.get_relationships(
                node_id, relationship_type, direction
            )

            for rel in relationships:
                next_id = rel.to_node_id if direction != "incoming" else rel.from_node_id
                if next_id != node_id:
                    await traverse(next_id, current_depth + 1)

        await traverse(start_node_id, 0)
        return result

    async def query_nodes(
        self,
        node_types: Optional[list[NodeType]] = None,
        min_confidence: float = 0.0,
        tier: Optional[MemoryTier] = None,
        validation_status: Optional[ValidationStatus] = None,
        topics: Optional[list[str]] = None,
        limit: int = 100,
        offset: int = 0,
        workspace_id: Optional[str] = None,
    ) -> list[KnowledgeNode]:
        """
        Query nodes with filters.

        Args:
            node_types: Filter by node types
            min_confidence: Minimum confidence
            tier: Filter by memory tier
            validation_status: Filter by validation status
            topics: Filter by topics
            limit: Maximum results
            offset: Skip first N results
            workspace_id: Filter by workspace

        Returns:
            List of matching nodes
        """
        self._ensure_initialized()
        assert self._meta_store is not None

        return self._meta_store.query_nodes(
            workspace_id=workspace_id or self.workspace_id,
            node_types=node_types,
            min_confidence=min_confidence,
            tier=tier,
            validation_status=validation_status,
            topics=topics,
            limit=limit,
            offset=offset,
        )

    async def get_stats(self) -> dict[str, Any]:
        """Get statistics about the knowledge mound."""
        self._ensure_initialized()
        assert self._meta_store is not None
        return self._meta_store.get_stats(self.workspace_id)

    async def delete_node(self, node_id: str) -> bool:
        """
        Delete a knowledge node and its relationships.

        Args:
            node_id: The node ID to delete

        Returns:
            True if deleted, False if node not found
        """
        self._ensure_initialized()
        assert self._meta_store is not None

        # Delete from vector store if available
        if self._vector_store:
            try:
                await self._vector_store.delete(node_id)
            except Exception as e:
                logger.warning(f"Failed to delete node from vector store: {e}")

        return self._meta_store.delete_node(node_id)

    async def query_by_provenance(
        self,
        source_type: Optional[str] = None,
        source_id: Optional[str] = None,
        node_type: Optional[str] = None,
        limit: int = 100,
        workspace_id: Optional[str] = None,
    ) -> list[KnowledgeNode]:
        """
        Query nodes by provenance attributes.

        Args:
            source_type: Filter by provenance source type (e.g., "workflow_engine", "debate")
            source_id: Filter by provenance source ID (e.g., workflow_id, debate_id)
            node_type: Filter by node type
            limit: Maximum results to return
            workspace_id: Filter by workspace (defaults to self.workspace_id)

        Returns:
            List of matching KnowledgeNodes
        """
        self._ensure_initialized()
        assert self._meta_store is not None

        ws_id = workspace_id or self.workspace_id
        node_ids = self._meta_store.query_by_provenance(
            source_type=source_type,
            source_id=source_id,
            node_type=node_type,
            workspace_id=ws_id,
            limit=limit,
        )

        nodes = []
        for node_id in node_ids:
            node = self._meta_store.get_node(node_id)
            if node:
                nodes.append(node)

        return nodes

    async def get_relationships(
        self,
        node_id: str,
        relationship_type: Optional[RelationshipType] = None,
        direction: Literal["outgoing", "incoming", "both"] = "both",
    ) -> list[KnowledgeRelationship]:
        """
        Get relationships for a specific node.

        Args:
            node_id: The node ID to get relationships for
            relationship_type: Filter by relationship type (optional)
            direction: Direction of relationships ('outgoing', 'incoming', or 'both')

        Returns:
            List of relationships
        """
        self._ensure_initialized()
        assert self._meta_store is not None
        return self._meta_store.get_relationships(node_id, relationship_type, direction)

    async def merge_from_debate(
        self,
        debate_result: Any,  # DebateResult type
        extract_facts: bool = True,
    ) -> list[str]:
        """
        Extract and store knowledge from a debate outcome.

        Args:
            debate_result: Result from Arena.run()
            extract_facts: Whether to extract facts from messages

        Returns:
            List of created node IDs
        """
        self._ensure_initialized()
        created_ids: list[str] = []

        # Create consensus node
        if hasattr(debate_result, "consensus") and debate_result.consensus:
            consensus_node = KnowledgeNode(
                node_type="consensus",
                content=debate_result.consensus,
                confidence=getattr(debate_result, "confidence", 0.8),
                provenance=ProvenanceChain(
                    source_type=ProvenanceType.DEBATE,
                    source_id=getattr(debate_result, "debate_id", ""),
                    debate_id=getattr(debate_result, "debate_id", None),
                ),
                workspace_id=self.workspace_id,
                validation_status=ValidationStatus.MAJORITY_AGREED,
            )
            node_id = await self.add_node(consensus_node)
            created_ids.append(node_id)

        # Extract facts from debate messages when extract_facts=True
        if extract_facts and hasattr(debate_result, "messages"):
            messages = debate_result.messages
            for msg in messages:
                # Extract factual claims from agent messages
                content = getattr(msg, "content", "") if hasattr(msg, "content") else str(msg)
                agent_id = getattr(msg, "agent", "unknown") if hasattr(msg, "agent") else "unknown"

                # Skip short or non-substantive messages
                if len(content) < 50:
                    continue

                # Create a claim node from each substantive message
                claim_node = KnowledgeNode(
                    node_type="claim",
                    content=content[:2000],  # Truncate long content
                    confidence=0.6,  # Lower confidence for unverified claims
                    provenance=ProvenanceChain(
                        source_type=ProvenanceType.AGENT,
                        source_id=agent_id,
                        agent_id=agent_id,
                        debate_id=getattr(debate_result, "debate_id", None),
                    ),
                    workspace_id=self.workspace_id,
                    validation_status=ValidationStatus.PENDING,  # type: ignore[attr-defined]
                    metadata={
                        "agent": agent_id,
                        "debate_round": getattr(msg, "round", 0),
                    },
                )
                claim_id = await self.add_node(claim_node)
                created_ids.append(claim_id)

                # Link claim to consensus if one exists
                if created_ids and created_ids[0] != claim_id:
                    await self.add_relationship(
                        from_node_id=claim_id,
                        to_node_id=created_ids[0],  # Consensus node
                        relationship_type="supports",
                        strength=0.5,
                        created_by=agent_id,
                    )

        return created_ids

    async def export_graph_d3(
        self,
        start_node_id: Optional[str] = None,
        depth: int = 3,
        limit: int = 100,
    ) -> dict[str, Any]:
        """
        Export graph in D3.js-compatible format.

        Args:
            start_node_id: Starting node for traversal (None for all nodes)
            depth: Maximum traversal depth
            limit: Maximum number of nodes

        Returns:
            Dict with 'nodes' and 'links' arrays for D3 force-directed graph
        """
        self._ensure_initialized()
        assert self._meta_store is not None

        nodes: list[dict[str, Any]] = []
        links: list[dict[str, Any]] = []
        node_ids: set[str] = set()

        if start_node_id:
            # Traverse from starting node
            traversed = await self.query_graph(start_node_id, depth=depth, direction="both")
            for node in traversed[:limit]:
                if node.id not in node_ids:
                    node_ids.add(node.id)
                    nodes.append(
                        {
                            "id": node.id,
                            "label": node.content[:100] if node.content else "",
                            "type": node.node_type,
                            "confidence": node.confidence,
                            "tier": node.tier.value if node.tier else "medium",
                            "validation": (
                                node.validation_status.value
                                if node.validation_status
                                else "pending"
                            ),
                        }
                    )
        else:
            # Get all nodes up to limit
            all_nodes = await self.query_nodes(limit=limit)
            for node in all_nodes:
                node_ids.add(node.id)
                nodes.append(
                    {
                        "id": node.id,
                        "label": node.content[:100] if node.content else "",
                        "type": node.node_type,
                        "confidence": node.confidence,
                        "tier": node.tier.value if node.tier else "medium",
                        "validation": (
                            node.validation_status.value if node.validation_status else "pending"
                        ),
                    }
                )

        # Get relationships between collected nodes
        for node_id in node_ids:
            rels = self._meta_store.get_relationships(node_id, direction="outgoing")
            for rel in rels:
                if rel.to_node_id in node_ids:
                    links.append(
                        {
                            "source": rel.from_node_id,
                            "target": rel.to_node_id,
                            "type": (
                                rel.relationship_type.value  # type: ignore[union-attr]
                                if hasattr(rel.relationship_type, "value")
                                else str(rel.relationship_type)
                            ),
                            "strength": rel.strength,
                        }
                    )

        return {"nodes": nodes, "links": links}

    async def export_graph_graphml(
        self,
        start_node_id: Optional[str] = None,
        depth: int = 3,
        limit: int = 100,
    ) -> str:
        """
        Export graph in GraphML format.

        Args:
            start_node_id: Starting node for traversal (None for all nodes)
            depth: Maximum traversal depth
            limit: Maximum number of nodes

        Returns:
            GraphML XML string
        """
        d3_data = await self.export_graph_d3(start_node_id, depth, limit)

        # Build GraphML XML
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
            '  <key id="label" for="node" attr.name="label" attr.type="string"/>',
            '  <key id="type" for="node" attr.name="type" attr.type="string"/>',
            '  <key id="confidence" for="node" attr.name="confidence" attr.type="double"/>',
            '  <key id="rel_type" for="edge" attr.name="type" attr.type="string"/>',
            '  <key id="strength" for="edge" attr.name="strength" attr.type="double"/>',
            '  <graph id="knowledge_graph" edgedefault="directed">',
        ]

        # Add nodes
        for node in d3_data["nodes"]:
            # Escape XML special characters in label
            label = (
                (node.get("label", "") or "")
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
            )
            lines.append(f'    <node id="{node["id"]}">')
            lines.append(f'      <data key="label">{label}</data>')
            lines.append(f'      <data key="type">{node.get("type", "unknown")}</data>')
            lines.append(f'      <data key="confidence">{node.get("confidence", 0.0)}</data>')
            lines.append("    </node>")

        # Add edges
        for i, link in enumerate(d3_data["links"]):
            lines.append(
                f'    <edge id="e{i}" source="{link["source"]}" target="{link["target"]}">'
            )
            lines.append(f'      <data key="rel_type">{link.get("type", "related")}</data>')
            lines.append(f'      <data key="strength">{link.get("strength", 0.5)}</data>')
            lines.append("    </edge>")

        lines.append("  </graph>")
        lines.append("</graphml>")

        return "\n".join(lines)

    async def close(self) -> None:
        """Close connections."""
        if self._vector_store:
            try:
                await self._vector_store.close()
            except Exception as e:
                logger.debug(f"Error closing vector store: {e}")
        self._initialized = False
