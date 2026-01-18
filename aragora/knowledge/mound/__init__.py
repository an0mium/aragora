"""
Knowledge Mound - Unified enterprise knowledge storage.

The Knowledge Mound implements the "termite mound" architecture where all agents
contribute to and query from a shared knowledge superstructure. It provides:

- Unified API over multiple storage backends (SQLite, PostgreSQL, Redis)
- Cross-system queries across ContinuumMemory, ConsensusMemory, FactStore
- Provenance tracking for audit and compliance
- Staleness detection with automatic revalidation scheduling
- Culture accumulation for organizational learning
- Multi-tenant workspace isolation

Basic Usage:
    from aragora.knowledge.mound import KnowledgeMound, MoundConfig

    mound = KnowledgeMound(workspace_id="my_team")
    await mound.initialize()

    # Store knowledge
    result = await mound.store(IngestionRequest(
        content="Contracts require 90-day notice",
        source_type=KnowledgeSource.DEBATE,
        debate_id="debate_123",
        workspace_id="my_team",
    ))

    # Query
    results = await mound.query("contract notice requirements")

    # Check staleness
    stale = await mound.get_stale_knowledge(threshold=0.7)

Production Usage (PostgreSQL + Redis):
    config = MoundConfig(
        backend=MoundBackend.HYBRID,
        postgres_url="postgresql://user:pass@host/db",
        redis_url="redis://localhost:6379",
    )
    mound = KnowledgeMound(config, workspace_id="enterprise")
    await mound.initialize()
"""

# Re-export from core module for backward compatibility
from aragora.knowledge.mound_core import (
    KnowledgeMound as _LegacyKnowledgeMound,
    KnowledgeMoundMetaStore,
    KnowledgeNode,
    KnowledgeQueryResult,
    KnowledgeRelationship,
    NodeType,
    ProvenanceChain,
    ProvenanceType,
    RelationshipType as LegacyRelationshipType,
)

# New enhanced types
from aragora.knowledge.mound.types import (
    ConfidenceLevel,
    CulturePattern,
    CulturePatternType,
    CultureProfile,
    GraphQueryResult,
    IngestionRequest,
    IngestionResult,
    KnowledgeItem,
    KnowledgeLink,
    KnowledgeSource,
    LinkResult,
    MoundBackend,
    MoundConfig,
    MoundStats,
    QueryFilters,
    QueryResult,
    RelationshipType,
    SourceFilter,
    StalenessCheck,
    StalenessReason,
    StoreResult,
    SyncResult,
)

# New enhanced facade (import with alias to avoid confusion with legacy)
from aragora.knowledge.mound.facade import KnowledgeMound

# New Phase 1 components for enterprise control plane
from aragora.knowledge.mound.semantic_store import (
    SemanticStore,
    SemanticIndexEntry,
    SemanticSearchResult,
)
from aragora.knowledge.mound.graph_store import (
    KnowledgeGraphStore,
    GraphLink,
    LineageNode,
    GraphTraversalResult,
)
from aragora.knowledge.mound.taxonomy import (
    DomainTaxonomy,
    TaxonomyNode,
    DEFAULT_TAXONOMY,
    DOMAIN_KEYWORDS,
)
from aragora.knowledge.mound.meta_learner import (
    KnowledgeMoundMetaLearner,
    RetrievalMetrics,
    TierOptimizationRecommendation,
    CoalescenceResult,
)

__all__ = [
    # Enhanced facade (primary export)
    "KnowledgeMound",
    "MoundConfig",
    "MoundBackend",
    # Types
    "ConfidenceLevel",
    "CulturePattern",
    "CulturePatternType",
    "CultureProfile",
    "GraphQueryResult",
    "IngestionRequest",
    "IngestionResult",
    "KnowledgeItem",
    "KnowledgeLink",
    "KnowledgeSource",
    "LinkResult",
    "MoundStats",
    "QueryFilters",
    "QueryResult",
    "RelationshipType",
    "SourceFilter",
    "StalenessCheck",
    "StalenessReason",
    "StoreResult",
    "SyncResult",
    # Legacy exports (for backward compatibility)
    "KnowledgeMoundMetaStore",
    "KnowledgeNode",
    "KnowledgeQueryResult",
    "KnowledgeRelationship",
    "NodeType",
    "ProvenanceChain",
    "ProvenanceType",
    "LegacyRelationshipType",
    # Phase 1: Semantic Store (mandatory embeddings)
    "SemanticStore",
    "SemanticIndexEntry",
    "SemanticSearchResult",
    # Phase 1: Knowledge Graph Store (relationships and lineage)
    "KnowledgeGraphStore",
    "GraphLink",
    "LineageNode",
    "GraphTraversalResult",
    # Phase 1: Domain Taxonomy (hierarchical organization)
    "DomainTaxonomy",
    "TaxonomyNode",
    "DEFAULT_TAXONOMY",
    "DOMAIN_KEYWORDS",
    # Phase 1: Meta-Learner (cross-memory optimization)
    "KnowledgeMoundMetaLearner",
    "RetrievalMetrics",
    "TierOptimizationRecommendation",
    "CoalescenceResult",
]
