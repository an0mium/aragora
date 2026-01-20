"""
Type definitions for the Knowledge Mound system.

Re-exports types from unified.types for backward compatibility,
plus additional types needed for the enhanced Knowledge Mound.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

# Re-export from unified types for compatibility
from aragora.knowledge.unified.types import (
    ConfidenceLevel,
    KnowledgeItem,
    KnowledgeLink,
    KnowledgeSource,
    LinkResult,
    QueryFilters,
    QueryResult,
    RelationshipType,
    SourceFilter,
    StoreResult,
)

# Alias for backward compatibility with adapters
SourceType = KnowledgeSource


class MoundBackend(str, Enum):
    """Backend storage options for Knowledge Mound."""

    SQLITE = "sqlite"
    POSTGRES = "postgres"
    HYBRID = "hybrid"  # Postgres + Redis caching


class CulturePatternType(str, Enum):
    """Types of organizational patterns tracked by the culture accumulator."""

    DECISION_STYLE = "decision_style"
    RISK_TOLERANCE = "risk_tolerance"
    DOMAIN_EXPERTISE = "domain_expertise"
    AGENT_PREFERENCES = "agent_preferences"
    DEBATE_DYNAMICS = "debate_dynamics"
    RESOLUTION_PATTERNS = "resolution_patterns"


class StalenessReason(str, Enum):
    """Reasons for knowledge staleness."""

    AGE = "age"
    CONTRADICTION = "contradiction"
    NEW_EVIDENCE = "new_evidence"
    CONSENSUS_CHANGE = "consensus_change"
    SCHEDULED = "scheduled"
    MANUAL = "manual"


# =============================================================================
# Visibility and Access Control Types (Phase 2)
# =============================================================================


class VisibilityLevel(str, Enum):
    """Visibility level for knowledge items.

    Controls who can see and access knowledge items:
    - PRIVATE: Only the creator and users with explicit access grants
    - WORKSPACE: All members of the workspace (default)
    - ORGANIZATION: All members of the organization
    - PUBLIC: Anyone, including unauthenticated users
    - SYSTEM: System-wide verified facts (global knowledge)
    """

    PRIVATE = "private"
    WORKSPACE = "workspace"
    ORGANIZATION = "organization"
    PUBLIC = "public"
    SYSTEM = "system"


class AccessGrantType(str, Enum):
    """Type of entity receiving an access grant."""

    USER = "user"
    ROLE = "role"
    WORKSPACE = "workspace"
    ORGANIZATION = "organization"


@dataclass
class AccessGrant:
    """Explicit access grant to a knowledge item.

    Allows fine-grained sharing of knowledge items with specific users,
    roles, workspaces, or organizations.
    """

    id: str
    item_id: str
    grantee_type: AccessGrantType
    grantee_id: str
    permissions: List[str] = field(default_factory=lambda: ["read"])
    granted_by: Optional[str] = None
    granted_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    def is_expired(self) -> bool:
        """Check if this grant has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def has_permission(self, permission: str) -> bool:
        """Check if this grant includes a specific permission."""
        return permission in self.permissions or "admin" in self.permissions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "item_id": self.item_id,
            "grantee_type": self.grantee_type.value,
            "grantee_id": self.grantee_id,
            "permissions": self.permissions,
            "granted_by": self.granted_by,
            "granted_at": self.granted_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass
class MoundConfig:
    """Configuration for the enhanced Knowledge Mound."""

    # Backend selection
    backend: MoundBackend = MoundBackend.SQLITE

    # PostgreSQL settings (production)
    postgres_url: Optional[str] = None
    postgres_pool_size: int = 10
    postgres_pool_max_overflow: int = 5

    # Redis settings (caching layer)
    redis_url: Optional[str] = None
    redis_cache_ttl: int = 300  # 5 minutes for queries
    redis_culture_ttl: int = 3600  # 1 hour for culture patterns

    # SQLite settings (development/testing)
    sqlite_path: Optional[str] = None

    # Weaviate settings (vector search)
    weaviate_url: Optional[str] = None
    weaviate_collection: str = "KnowledgeMound"
    weaviate_api_key: Optional[str] = None

    # Feature flags
    enable_staleness_detection: bool = True
    enable_culture_accumulator: bool = True
    enable_auto_revalidation: bool = False
    enable_deduplication: bool = True
    enable_provenance_tracking: bool = True

    # Bidirectional adapter flags
    enable_evidence_adapter: bool = True
    enable_pulse_adapter: bool = True
    enable_insights_adapter: bool = True
    enable_elo_adapter: bool = True
    enable_belief_adapter: bool = True
    enable_cost_adapter: bool = False  # Opt-in (sensitive data)

    # Adapter confidence thresholds
    evidence_min_reliability: float = 0.6
    pulse_min_quality: float = 0.6
    insight_min_confidence: float = 0.7
    crux_min_score: float = 0.3
    belief_min_confidence: float = 0.8

    # Multi-tenant
    default_workspace_id: str = "default"

    # Query settings
    default_query_limit: int = 20
    max_query_limit: int = 100
    parallel_queries: bool = True

    # Staleness settings
    staleness_check_interval: timedelta = timedelta(hours=1)
    staleness_age_threshold: timedelta = timedelta(days=7)
    staleness_revalidation_threshold: float = 0.8


@dataclass
class IngestionRequest:
    """Request to ingest new knowledge into the mound."""

    content: str
    workspace_id: str

    # Source identifiers (at least one recommended)
    source_type: KnowledgeSource = KnowledgeSource.FACT
    document_id: Optional[str] = None
    debate_id: Optional[str] = None
    agent_id: Optional[str] = None
    user_id: Optional[str] = None

    # Classification
    node_type: str = "fact"  # fact, claim, memory, evidence, consensus, entity
    confidence: float = 0.5
    tier: str = "slow"  # fast, medium, slow, glacial
    topics: List[str] = field(default_factory=list)

    # Relationships
    supports: List[str] = field(default_factory=list)
    contradicts: List[str] = field(default_factory=list)
    derived_from: List[str] = field(default_factory=list)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionResult:
    """Result of ingesting knowledge."""

    node_id: str
    success: bool
    deduplicated: bool = False
    existing_node_id: Optional[str] = None
    relationships_created: int = 0
    message: Optional[str] = None


@dataclass
class StalenessCheck:
    """Result of a staleness check."""

    node_id: str
    staleness_score: float  # 0.0 = fresh, 1.0 = stale
    reasons: List[StalenessReason] = field(default_factory=list)
    last_checked_at: datetime = field(default_factory=datetime.now)
    revalidation_recommended: bool = False
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CulturePattern:
    """An accumulated organizational pattern."""

    id: str
    workspace_id: str
    pattern_type: CulturePatternType
    pattern_key: str
    pattern_value: Dict[str, Any]
    observation_count: int
    confidence: float
    first_observed_at: datetime
    last_observed_at: datetime
    contributing_debates: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CultureProfile:
    """Aggregated culture profile for a workspace."""

    workspace_id: str
    patterns: Dict[CulturePatternType, List[CulturePattern]]
    generated_at: datetime
    total_observations: int
    dominant_traits: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphQueryResult:
    """Result of a graph traversal query."""

    nodes: List[KnowledgeItem]
    edges: List[KnowledgeLink]
    root_id: str
    depth: int
    total_nodes: int
    total_edges: int


@dataclass
class MoundStats:
    """Statistics about the Knowledge Mound."""

    total_nodes: int
    nodes_by_type: Dict[str, int]
    nodes_by_tier: Dict[str, int]
    nodes_by_validation: Dict[str, int]
    total_relationships: int
    relationships_by_type: Dict[str, int]
    average_confidence: float
    stale_nodes_count: int
    workspace_id: Optional[str] = None


@dataclass
class SyncResult:
    """Result of syncing from another memory system."""

    source: str
    nodes_synced: int
    nodes_updated: int
    nodes_skipped: int
    relationships_created: int
    duration_ms: int
    errors: List[str] = field(default_factory=list)


# =============================================================================
# Enhanced Types for Enterprise Control Plane (Phase 1)
# =============================================================================


@dataclass
class EnhancedKnowledgeItem(KnowledgeItem):
    """
    Extended knowledge item with semantic grounding and lineage.

    Extends the base KnowledgeItem with mandatory embedding support,
    tenant isolation, domain taxonomy, and belief lineage tracking.
    """

    # Mandatory embedding (semantic grounding)
    embedding: List[float] = field(default_factory=list)
    embedding_model: str = ""

    # Tenant isolation
    tenant_id: str = "default"

    # Domain taxonomy
    domain_path: List[str] = field(default_factory=list)  # ["legal", "contracts", "termination"]
    domain: str = "general"  # Flattened path: "legal/contracts/termination"

    # Lineage tracking
    predecessor_ids: List[str] = field(default_factory=list)  # Items this supersedes
    successor_id: Optional[str] = None  # Item that supersedes this

    # Coalescence tracking
    merged_from: List[str] = field(default_factory=list)  # Items merged into this

    # Retrieval metrics (for meta-learning)
    retrieval_count: int = 0
    last_retrieved_at: Optional[datetime] = None
    avg_retrieval_rank: float = 0.0  # Average position in search results

    # Visibility and access control (Phase 2)
    visibility: VisibilityLevel = VisibilityLevel.WORKSPACE
    visibility_set_by: Optional[str] = None
    access_grants: List[AccessGrant] = field(default_factory=list)
    is_discoverable: bool = True  # Whether item shows in search results

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            "embedding_model": self.embedding_model,
            "tenant_id": self.tenant_id,
            "domain_path": self.domain_path,
            "domain": self.domain,
            "predecessor_ids": self.predecessor_ids,
            "successor_id": self.successor_id,
            "merged_from": self.merged_from,
            "retrieval_count": self.retrieval_count,
            "last_retrieved_at": (
                self.last_retrieved_at.isoformat()
                if self.last_retrieved_at else None
            ),
            "avg_retrieval_rank": self.avg_retrieval_rank,
            # Visibility fields
            "visibility": self.visibility.value,
            "visibility_set_by": self.visibility_set_by,
            "access_grants": [g.to_dict() for g in self.access_grants],
            "is_discoverable": self.is_discoverable,
        })
        # Note: embedding intentionally excluded from dict to save space
        return base_dict


@dataclass
class BeliefLineageEntry:
    """
    Entry in a belief lineage chain.

    Tracks how knowledge evolved over time through debates and updates.
    """

    id: str
    current_id: str  # Current belief's Knowledge Mound ID
    predecessor_id: Optional[str]  # Previous version (None for origins)
    supersession_reason: Optional[str]  # Why it was superseded
    debate_id: Optional[str]  # Debate that caused supersession
    created_at: datetime
    tenant_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "current_id": self.current_id,
            "predecessor_id": self.predecessor_id,
            "supersession_reason": self.supersession_reason,
            "debate_id": self.debate_id,
            "created_at": self.created_at.isoformat(),
            "tenant_id": self.tenant_id,
        }


@dataclass
class DomainNode:
    """
    A node in the domain taxonomy tree.

    Represents a hierarchical domain category for knowledge organization.
    """

    id: str
    name: str
    full_path: str  # e.g., "legal/contracts/termination"
    parent_id: Optional[str]
    description: Optional[str]
    tenant_id: str
    created_at: datetime
    item_count: int = 0  # Number of items in this domain

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "full_path": self.full_path,
            "parent_id": self.parent_id,
            "description": self.description,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at.isoformat(),
            "item_count": self.item_count,
        }


@dataclass
class UnifiedQueryRequest:
    """
    Request for unified query across all knowledge sources.

    Supports semantic search, graph expansion, and domain filtering.
    """

    query: str
    tenant_id: str = "default"

    # Search mode
    search_mode: Literal["semantic", "keyword", "hybrid"] = "hybrid"

    # Filters
    sources: Optional[List[KnowledgeSource]] = None
    domain_filter: Optional[str] = None  # e.g., "legal" or "legal/contracts"
    min_confidence: Optional[float] = None
    min_importance: Optional[float] = None

    # Graph expansion
    include_graph: bool = False
    graph_depth: int = 1
    relationship_types: Optional[List[RelationshipType]] = None

    # Pagination
    limit: int = 20
    offset: int = 0


@dataclass
class UnifiedQueryResult:
    """
    Result of a unified query across all knowledge sources.

    Includes items, graph relationships, and query metadata.
    """

    items: List[Union[KnowledgeItem, "EnhancedKnowledgeItem"]]
    total_count: int
    query: str
    tenant_id: str
    execution_time_ms: float

    # Sources that were queried
    sources_queried: List[KnowledgeSource] = field(default_factory=list)

    # Graph data (if include_graph was True)
    graph_nodes: List[str] = field(default_factory=list)
    graph_edges: List[KnowledgeLink] = field(default_factory=list)

    # Lineage data (for items with predecessors/successors)
    lineage_chains: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "items": [item.to_dict() for item in self.items],
            "total_count": self.total_count,
            "query": self.query,
            "tenant_id": self.tenant_id,
            "execution_time_ms": self.execution_time_ms,
            "sources_queried": [s.value for s in self.sources_queried],
            "graph_nodes": self.graph_nodes,
            "graph_edges": [e.to_dict() for e in self.graph_edges],
            "lineage_chains": self.lineage_chains,
        }


__all__ = [
    # Re-exported types
    "ConfidenceLevel",
    "KnowledgeItem",
    "KnowledgeLink",
    "KnowledgeSource",
    "SourceType",  # Alias for KnowledgeSource
    "LinkResult",
    "QueryFilters",
    "QueryResult",
    "RelationshipType",
    "SourceFilter",
    "StoreResult",
    # New types
    "CulturePattern",
    "CulturePatternType",
    "CultureProfile",
    "GraphQueryResult",
    "IngestionRequest",
    "IngestionResult",
    "MoundBackend",
    "MoundConfig",
    "MoundStats",
    "StalenessCheck",
    "StalenessReason",
    "SyncResult",
    # Phase 1: Enhanced types for enterprise control plane
    "EnhancedKnowledgeItem",
    "BeliefLineageEntry",
    "DomainNode",
    "UnifiedQueryRequest",
    "UnifiedQueryResult",
    # Phase 2: Visibility and access control
    "VisibilityLevel",
    "AccessGrantType",
    "AccessGrant",
]
