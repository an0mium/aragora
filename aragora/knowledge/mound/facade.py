"""
Knowledge Mound Facade - Unified knowledge storage with production backends.

This is the main entry point for the enhanced Knowledge Mound system,
providing a unified API over multiple storage backends (SQLite, PostgreSQL, Redis)
and integrating staleness detection and culture accumulation.

The facade is composed from modular mixins:
- core.py: Initialization, lifecycle, storage adapters
- api/crud.py: Create, read, update, delete operations
- api/query.py: Query and search operations
- api/rlm.py: RLM (Recursive Language Models) integration
- ops/staleness.py: Staleness detection and revalidation
- ops/culture.py: Culture accumulation and management
- ops/sync.py: Cross-system synchronization

Usage:
    from aragora.knowledge.mound import KnowledgeMound, MoundConfig

    config = MoundConfig(
        backend=MoundBackend.POSTGRES,
        postgres_url="postgresql://user:pass@localhost/aragora",
        redis_url="redis://localhost:6379",
    )

    mound = KnowledgeMound(config, workspace_id="enterprise_team")
    await mound.initialize()

    # Store knowledge with provenance
    result = await mound.store(IngestionRequest(
        content="All contracts must have 90-day notice periods",
        source_type=KnowledgeSource.DEBATE,
        debate_id="debate_123",
        confidence=0.95,
        workspace_id="enterprise_team",
    ))

    # Query semantically
    results = await mound.query("contract notice requirements", limit=10)

    # Check staleness
    stale = await mound.get_stale_knowledge(threshold=0.7)

    # Get culture profile
    culture = await mound.get_culture_profile()
"""

from __future__ import annotations

from typing import Optional

from aragora.knowledge.mound.core import KnowledgeMoundCore
from aragora.knowledge.mound.api.crud import CRUDOperationsMixin
from aragora.knowledge.mound.api.query import QueryOperationsMixin
from aragora.knowledge.mound.api.rlm import RLMOperationsMixin
from aragora.knowledge.mound.ops.staleness import StalenessOperationsMixin
from aragora.knowledge.mound.ops.culture import CultureOperationsMixin
from aragora.knowledge.mound.ops.sync import SyncOperationsMixin
from aragora.knowledge.mound.ops.global_knowledge import GlobalKnowledgeMixin
from aragora.knowledge.mound.ops.sharing import KnowledgeSharingMixin
from aragora.knowledge.mound.ops.federation import KnowledgeFederationMixin
from aragora.knowledge.mound.ops.dedup import DedupOperationsMixin
from aragora.knowledge.mound.ops.pruning import PruningOperationsMixin
from aragora.knowledge.mound.types import MoundConfig


class KnowledgeMound(
    CRUDOperationsMixin,
    QueryOperationsMixin,
    RLMOperationsMixin,
    StalenessOperationsMixin,
    CultureOperationsMixin,
    SyncOperationsMixin,
    GlobalKnowledgeMixin,
    KnowledgeSharingMixin,
    KnowledgeFederationMixin,
    DedupOperationsMixin,
    PruningOperationsMixin,
    KnowledgeMoundCore,
):
    """
    Unified knowledge facade for the Aragora multi-agent control plane.

    The Knowledge Mound implements the "termite mound" architecture where
    all agents contribute to and query from a shared knowledge superstructure.

    Features:
    - Unified API across SQLite (dev), PostgreSQL (prod), and Redis (cache)
    - Cross-system queries across ContinuumMemory, ConsensusMemory, FactStore
    - Provenance tracking for audit and compliance
    - Staleness detection with automatic revalidation scheduling
    - Culture accumulation for organizational learning
    - Multi-tenant workspace isolation
    - RLM integration for hierarchical context navigation

    This class composes functionality from modular mixins:
    - CRUDOperationsMixin: store, get, update, delete, add, add_node, get_node
    - QueryOperationsMixin: query, query_semantic, query_graph, export_graph_*, query_with_visibility
    - RLMOperationsMixin: query_with_rlm, is_rlm_available
    - StalenessOperationsMixin: get_stale_knowledge, mark_validated, schedule_revalidation
    - CultureOperationsMixin: get_culture_profile, observe_debate, recommend_agents, org culture
    - SyncOperationsMixin: sync_from_*, sync_all
    - GlobalKnowledgeMixin: store_verified_fact, query_global_knowledge, promote_to_global
    - KnowledgeSharingMixin: share_with_workspace, share_with_user, get_shared_with_me, revoke_share
    - KnowledgeFederationMixin: register_federated_region, sync_to_region, pull_from_region
    - KnowledgeMoundCore: initialize, close, session, get_stats, storage adapters
    """

    def __init__(
        self,
        config: Optional[MoundConfig] = None,
        workspace_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the Knowledge Mound.

        Args:
            config: Mound configuration. Defaults to SQLite backend.
            workspace_id: Default workspace for queries. Overrides config.
        """
        super().__init__(config=config, workspace_id=workspace_id)
