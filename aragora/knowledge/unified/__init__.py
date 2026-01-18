"""
Unified Knowledge Store for Aragora.

This submodule provides a federated query interface across all Aragora
knowledge systems, complementing the main KnowledgeMound class in
aragora.knowledge.mound.

The unified_store module provides:
- Federated queries across ContinuumMemory, ConsensusMemory, FactStore, VectorStore
- Cross-system knowledge linking
- Unified query interface

For the main KnowledgeMound with graph and provenance support, use:
    from aragora.knowledge.mound import KnowledgeMound, KnowledgeNode

For federated queries across all systems, use:
    from aragora.knowledge.unified import UnifiedKnowledgeStore
"""

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
from aragora.knowledge.unified.unified_store import (
    KnowledgeMound as UnifiedKnowledgeStore,
    KnowledgeMoundConfig as UnifiedStoreConfig,
)

__all__ = [
    # Unified store (federated queries)
    "UnifiedKnowledgeStore",
    "UnifiedStoreConfig",
    # Types
    "ConfidenceLevel",
    "KnowledgeItem",
    "KnowledgeLink",
    "KnowledgeSource",
    "LinkResult",
    "QueryFilters",
    "QueryResult",
    "RelationshipType",
    "SourceFilter",
    "StoreResult",
]
