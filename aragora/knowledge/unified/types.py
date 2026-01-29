"""
Type definitions for the Knowledge Mound system.

The Knowledge Mound provides a unified interface over multiple knowledge stores:
- ContinuumMemory: Multi-tier temporal learning
- ConsensusMemory: Debate outcomes and agreements
- FactStore: Verified facts from document analysis
- VectorStore: Semantic embeddings for similarity search

This module defines the shared types used across all Knowledge Mound components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional


class KnowledgeSource(str, Enum):
    """Source types for knowledge items."""

    # Core memory systems
    CONTINUUM = "continuum"  # ContinuumMemory entries
    CONSENSUS = "consensus"  # ConsensusMemory debate outcomes
    DEBATE = "debate"  # Debate orchestrator outcomes
    FACT = "fact"  # FactStore verified facts
    VECTOR = "vector"  # Vector store embeddings
    DOCUMENT = "document"  # Raw document chunks
    EXTERNAL = "external"  # External data sources
    EVIDENCE = "evidence"  # EvidenceStore snippets
    CRITIQUE = "critique"  # CritiqueStore patterns

    # Bidirectional integration sources
    PULSE = "pulse"  # Trending topics and scheduled debates
    INSIGHT = "insight"  # Debate insights and patterns
    FLIP = "flip"  # Position flip events (Trickster)
    ELO = "elo"  # Agent rankings and calibration
    BELIEF = "belief"  # Belief network nodes and cruxes
    PROVENANCE = "provenance"  # Evidence provenance chains
    COST = "cost"  # Cost patterns and anomalies
    RANKING = "ranking"  # Agent domain expertise profiles
    RLM = "rlm"  # RLM compression patterns
    EXTRACTION = "extraction"  # Extracted content from documents
    CALIBRATION = "calibration"  # Multi-party calibration fusion consensus


class RelationshipType(str, Enum):
    """Types of relationships between knowledge items."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    ELABORATES = "elaborates"
    SUPERSEDES = "supersedes"
    DERIVED_FROM = "derived_from"
    RELATED_TO = "related_to"
    CITES = "cites"


class ConfidenceLevel(str, Enum):
    """Confidence levels for knowledge items."""

    VERIFIED = "verified"  # Formally verified or highly confident
    HIGH = "high"  # Strong consensus or evidence
    MEDIUM = "medium"  # Moderate confidence
    LOW = "low"  # Weak evidence or contested
    UNVERIFIED = "unverified"  # Not yet verified


@dataclass
class KnowledgeItem:
    """
    A unified knowledge item that can represent content from any source.

    This is the common format returned by Knowledge Mound queries,
    abstracting away the underlying storage system.
    """

    id: str
    content: str
    source: KnowledgeSource
    source_id: str  # ID in the original store
    confidence: ConfidenceLevel
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    # Optional fields depending on source
    importance: float | None = None  # 0-1 importance score
    embedding: Optional[list[float]] = None  # Vector embedding

    # Cross-reference tracking
    cross_references: list[str] = field(default_factory=list)  # IDs of related items

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source.value,
            "source_id": self.source_id,
            "confidence": self.confidence.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "importance": self.importance,
            "cross_references": self.cross_references,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeItem":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            source=KnowledgeSource(data["source"]),
            source_id=data["source_id"],
            confidence=ConfidenceLevel(data["confidence"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
            importance=data.get("importance"),
            cross_references=data.get("cross_references", []),
        )


@dataclass
class KnowledgeLink:
    """
    A link between two knowledge items.

    Links enable the Knowledge Mound to function as a knowledge graph,
    tracking relationships between facts, memories, and documents.
    """

    id: str
    source_id: str  # Knowledge item ID
    target_id: str  # Knowledge item ID
    relationship: RelationshipType
    confidence: float  # 0-1 confidence in the relationship
    created_at: datetime
    created_by: str | None = None  # Agent or user that created the link
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship": self.relationship.value,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "metadata": self.metadata,
        }


@dataclass
class QueryFilters:
    """Filters for Knowledge Mound queries."""

    sources: Optional[list[KnowledgeSource]] = None  # Filter by source type
    min_confidence: ConfidenceLevel | None = None
    min_importance: float | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    workspace_id: str | None = None
    debate_id: str | None = None
    document_ids: Optional[list[str]] = None
    tags: Optional[list[str]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API serialization."""
        result: dict[str, Any] = {}
        if self.sources:
            result["sources"] = [s.value for s in self.sources]
        if self.min_confidence:
            result["min_confidence"] = self.min_confidence.value
        if self.min_importance is not None:
            result["min_importance"] = self.min_importance
        if self.created_after:
            result["created_after"] = self.created_after.isoformat()
        if self.created_before:
            result["created_before"] = self.created_before.isoformat()
        if self.workspace_id:
            result["workspace_id"] = self.workspace_id
        if self.debate_id:
            result["debate_id"] = self.debate_id
        if self.document_ids:
            result["document_ids"] = self.document_ids
        if self.tags:
            result["tags"] = self.tags
        return result


@dataclass
class QueryResult:
    """Result of a Knowledge Mound query."""

    items: list[KnowledgeItem]
    total_count: int  # Total matching items (may be more than returned)
    query: str
    filters: QueryFilters | None = None
    execution_time_ms: float = 0.0
    sources_queried: list[KnowledgeSource] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "items": [item.to_dict() for item in self.items],
            "total_count": self.total_count,
            "query": self.query,
            "filters": self.filters.to_dict() if self.filters else None,
            "execution_time_ms": self.execution_time_ms,
            "sources_queried": [s.value for s in self.sources_queried],
        }


@dataclass
class StoreResult:
    """Result of storing a knowledge item."""

    id: str
    source: KnowledgeSource
    success: bool
    cross_references_created: int = 0
    message: str | None = None


@dataclass
class LinkResult:
    """Result of creating a knowledge link."""

    id: str
    success: bool
    message: str | None = None


# Type aliases for commonly used parameter types
SourceFilter = Literal["all", "continuum", "consensus", "fact", "vector", "document"]
