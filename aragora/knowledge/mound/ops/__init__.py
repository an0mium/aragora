"""
Knowledge Mound Operations mixins.

This module provides operational mixins for the KnowledgeMound facade:
- StalenessOperationsMixin: Staleness detection and revalidation
- CultureOperationsMixin: Culture accumulation and management
- SyncOperationsMixin: Cross-system synchronization
- GlobalKnowledgeMixin: Global/public knowledge operations
- KnowledgeSharingMixin: Cross-workspace knowledge sharing
- KnowledgeFederationMixin: Multi-region knowledge synchronization
- DedupOperationsMixin: Similarity-based deduplication
- PruningOperationsMixin: Automatic and manual pruning
"""

from aragora.knowledge.mound.ops.staleness import StalenessOperationsMixin
from aragora.knowledge.mound.ops.culture import CultureOperationsMixin
from aragora.knowledge.mound.ops.sync import SyncOperationsMixin
from aragora.knowledge.mound.ops.global_knowledge import GlobalKnowledgeMixin, SYSTEM_WORKSPACE_ID
from aragora.knowledge.mound.ops.sharing import KnowledgeSharingMixin
from aragora.knowledge.mound.ops.federation import (
    KnowledgeFederationMixin,
    FederationMode,
    SyncScope,
    FederatedRegion,
    SyncResult,
)
from aragora.knowledge.mound.ops.dedup import (
    DedupOperationsMixin,
    DuplicateCluster,
    DuplicateMatch,
    DedupReport,
    MergeResult,
)
from aragora.knowledge.mound.ops.pruning import (
    PruningOperationsMixin,
    PruningPolicy,
    PrunableItem,
    PruneResult,
    PruneHistory,
    PruningAction,
)

__all__ = [
    "StalenessOperationsMixin",
    "CultureOperationsMixin",
    "SyncOperationsMixin",
    "GlobalKnowledgeMixin",
    "KnowledgeSharingMixin",
    "KnowledgeFederationMixin",
    "FederationMode",
    "SyncScope",
    "FederatedRegion",
    "SyncResult",
    "SYSTEM_WORKSPACE_ID",
    # Dedup
    "DedupOperationsMixin",
    "DuplicateCluster",
    "DuplicateMatch",
    "DedupReport",
    "MergeResult",
    # Pruning
    "PruningOperationsMixin",
    "PruningPolicy",
    "PrunableItem",
    "PruneResult",
    "PruneHistory",
    "PruningAction",
]
