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
from aragora.knowledge.mound.revalidation_scheduler import (
    RevalidationScheduler,
    handle_revalidation_task,
)

# Metrics and observability
from aragora.knowledge.mound.metrics import (
    KMMetrics,
    OperationType,
    OperationStats,
    HealthStatus,
    HealthReport,
    get_metrics,
    set_metrics,
)

# Event batching for WebSocket efficiency
from aragora.knowledge.mound.event_batcher import (
    EventBatcher,
    EventBatch,
    BatchedEvent,
    AdapterEventBatcher,
)

# WebSocket bridge for real-time KM events
from aragora.knowledge.mound.websocket_bridge import (
    KMWebSocketBridge,
    KMSubscription,
    KMSubscriptionManager,
    get_km_bridge,
    set_km_bridge,
    create_km_bridge,
)

# Federated queries across adapters
from aragora.knowledge.mound.federated_query import (
    FederatedQueryAggregator,
    FederatedQueryResult,
    FederatedResult,
    QuerySource,
    EmbeddingRelevanceScorer,
)

# Persistence resilience (retry, transactions, health monitoring)
from aragora.knowledge.mound.resilience import (
    CacheInvalidationBus,
    CacheInvalidationEvent,
    ConnectionHealthMonitor,
    IntegrityCheckResult,
    IntegrityVerifier,
    ResilientPostgresStore,
    RetryConfig,
    RetryStrategy,
    TransactionConfig,
    TransactionIsolation,
    TransactionManager,
    get_invalidation_bus,
    with_retry,
)

# Checkpoint store for KM state persistence
from aragora.knowledge.mound.checkpoint import (
    KMCheckpointStore,
    KMCheckpointMetadata,
    KMCheckpointContent,
    RestoreResult,
    get_km_checkpoint_store,
    reset_km_checkpoint_store,
)

# Singleton instance
_knowledge_mound_instance: "KnowledgeMound" = None
_knowledge_mound_config: "MoundConfig" = None


def set_mound_config(config: "MoundConfig") -> None:
    """
    Set the global MoundConfig before creating the singleton.

    Call this early in application startup to configure the Knowledge Mound
    with environment-specific settings (database URLs, feature flags, etc.)
    before any code calls get_knowledge_mound().

    Args:
        config: MoundConfig with environment-specific settings

    Raises:
        RuntimeError: If called after singleton has been created
    """
    global _knowledge_mound_config, _knowledge_mound_instance

    if _knowledge_mound_instance is not None:
        raise RuntimeError(
            "Cannot set MoundConfig after KnowledgeMound singleton has been created. "
            "Call set_mound_config() before any calls to get_knowledge_mound()."
        )

    _knowledge_mound_config = config


def get_mound_config() -> "MoundConfig":
    """
    Get the current global MoundConfig.

    Returns:
        MoundConfig if set, or a default config otherwise
    """
    global _knowledge_mound_config

    if _knowledge_mound_config is None:
        return MoundConfig(backend=MoundBackend.SQLITE)

    return _knowledge_mound_config


def get_knowledge_mound(
    workspace_id: str = "default",
    config: "MoundConfig" = None,
    auto_initialize: bool = True,
) -> "KnowledgeMound":
    """
    Get or create the global KnowledgeMound singleton.

    This provides a lazy-loaded default Knowledge Mound instance that can be
    shared across the application. Use this when you need a KM instance but
    don't have one configured.

    Args:
        workspace_id: Workspace ID for multi-tenant isolation
        config: Optional MoundConfig for customization (or use set_mound_config())
        auto_initialize: If True, automatically initialize the mound

    Returns:
        KnowledgeMound instance
    """
    global _knowledge_mound_instance, _knowledge_mound_config

    if _knowledge_mound_instance is None:
        import logging
        logger = logging.getLogger(__name__)

        # Use provided config, global config, or default
        if config is None:
            config = _knowledge_mound_config or MoundConfig(backend=MoundBackend.SQLITE)

        logger.info(f"[knowledge_mound] Creating singleton instance (workspace={workspace_id}, backend={config.backend.value})")
        _knowledge_mound_instance = KnowledgeMound(
            config=config,
            workspace_id=workspace_id,
        )

        if auto_initialize:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Can't await in running loop, schedule for later
                    logger.debug("[knowledge_mound] Deferring initialization (event loop running)")
                else:
                    loop.run_until_complete(_knowledge_mound_instance.initialize())
                    logger.info("[knowledge_mound] Singleton initialized successfully")
            except RuntimeError:
                # No event loop available
                logger.debug("[knowledge_mound] No event loop, deferring initialization")

    return _knowledge_mound_instance


def reset_knowledge_mound() -> None:
    """Reset the global KnowledgeMound singleton (for testing)."""
    global _knowledge_mound_instance
    _knowledge_mound_instance = None


__all__ = [
    # Enhanced facade (primary export)
    "KnowledgeMound",
    "MoundConfig",
    "MoundBackend",
    # Singleton access
    "get_knowledge_mound",
    "reset_knowledge_mound",
    "set_mound_config",
    "get_mound_config",
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
    # Phase 1: Revalidation Scheduler (automatic staleness handling)
    "RevalidationScheduler",
    "handle_revalidation_task",
    # Metrics and observability
    "KMMetrics",
    "OperationType",
    "OperationStats",
    "HealthStatus",
    "HealthReport",
    "get_metrics",
    "set_metrics",
    # Event batching
    "EventBatcher",
    "EventBatch",
    "BatchedEvent",
    "AdapterEventBatcher",
    # WebSocket bridge
    "KMWebSocketBridge",
    "KMSubscription",
    "KMSubscriptionManager",
    "get_km_bridge",
    "set_km_bridge",
    "create_km_bridge",
    # Federated queries
    "FederatedQueryAggregator",
    "FederatedQueryResult",
    "FederatedResult",
    "QuerySource",
    "EmbeddingRelevanceScorer",
    # Checkpoint store
    "KMCheckpointStore",
    "KMCheckpointMetadata",
    "KMCheckpointContent",
    "RestoreResult",
    "get_km_checkpoint_store",
    "reset_km_checkpoint_store",
]
