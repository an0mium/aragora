"""
Knowledge Mound Handler Module.

This module provides modular endpoint handlers for the Knowledge Mound API.
The handlers are split into mixins for better maintainability:

- NodeOperationsMixin: Node CRUD operations
- RelationshipOperationsMixin: Relationship operations
- GraphOperationsMixin: Graph traversal and export
- CultureOperationsMixin: Organization culture management
- StalenessOperationsMixin: Staleness detection and revalidation
- SyncOperationsMixin: Sync with legacy memory systems
- VisibilityOperationsMixin: Per-item visibility control
- SharingOperationsMixin: Cross-workspace knowledge sharing
- GlobalKnowledgeOperationsMixin: Global/public verified facts
- FederationOperationsMixin: Multi-region knowledge sync
- DedupOperationsMixin: Duplicate detection and merging
- PruningOperationsMixin: Knowledge pruning and archival
"""

from .handler import KnowledgeMoundHandler

__all__ = [
    "KnowledgeMoundHandler",
]
