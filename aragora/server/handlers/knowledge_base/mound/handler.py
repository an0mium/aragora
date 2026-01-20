"""
Main Knowledge Mound Handler.

Combines all mixins to provide the complete Knowledge Mound API:

Knowledge Mound API (unified knowledge storage):
- POST /api/knowledge/mound/query - Semantic query against knowledge mound
- POST /api/knowledge/mound/nodes - Add a knowledge node
- GET /api/knowledge/mound/nodes/:id - Get specific node
- GET /api/knowledge/mound/nodes/:id/relationships - Get relationships for a node
- GET /api/knowledge/mound/nodes/:id/visibility - Get node visibility
- PUT /api/knowledge/mound/nodes/:id/visibility - Set node visibility
- GET /api/knowledge/mound/nodes/:id/access - List access grants
- POST /api/knowledge/mound/nodes/:id/access - Grant access
- DELETE /api/knowledge/mound/nodes/:id/access - Revoke access
- GET /api/knowledge/mound/nodes - List/filter nodes
- POST /api/knowledge/mound/relationships - Add relationship between nodes
- GET /api/knowledge/mound/graph/:id - Get graph traversal from node
- GET /api/knowledge/mound/graph/:id/lineage - Get node lineage
- GET /api/knowledge/mound/graph/:id/related - Get related nodes
- GET /api/knowledge/mound/stats - Get mound statistics
- POST /api/knowledge/mound/index/repository - Index a repository
- GET /api/knowledge/mound/culture - Get culture profile
- POST /api/knowledge/mound/culture/documents - Add culture document
- POST /api/knowledge/mound/culture/promote - Promote knowledge to culture
- GET /api/knowledge/mound/stale - Get stale knowledge items
- POST /api/knowledge/mound/revalidate/:id - Revalidate specific node
- POST /api/knowledge/mound/schedule-revalidation - Schedule batch revalidation
- POST /api/knowledge/mound/sync/continuum - Sync from ContinuumMemory
- POST /api/knowledge/mound/sync/consensus - Sync from ConsensusMemory
- POST /api/knowledge/mound/sync/facts - Sync from FactStore
- GET /api/knowledge/mound/export/d3 - Export graph as D3 JSON
- GET /api/knowledge/mound/export/graphml - Export graph as GraphML

Sharing endpoints:
- POST /api/knowledge/mound/share - Share item with workspace/user
- GET /api/knowledge/mound/shared-with-me - Get items shared with me
- DELETE /api/knowledge/mound/share - Revoke a share
- PATCH /api/knowledge/mound/share - Update share permissions
- GET /api/knowledge/mound/my-shares - List items I've shared

Global knowledge endpoints:
- POST /api/knowledge/mound/global - Store verified fact (admin)
- GET /api/knowledge/mound/global - Query global knowledge
- POST /api/knowledge/mound/global/promote - Promote to global
- GET /api/knowledge/mound/global/facts - Get all system facts
- GET /api/knowledge/mound/global/workspace-id - Get system workspace ID

Federation endpoints:
- POST /api/knowledge/mound/federation/regions - Register federated region
- GET /api/knowledge/mound/federation/regions - List federated regions
- DELETE /api/knowledge/mound/federation/regions/:id - Unregister region
- POST /api/knowledge/mound/federation/sync/push - Sync to region
- POST /api/knowledge/mound/federation/sync/pull - Pull from region
- POST /api/knowledge/mound/federation/sync/all - Sync all regions
- GET /api/knowledge/mound/federation/status - Get federation status

Deduplication endpoints:
- GET /api/knowledge/mound/dedup/clusters - Find duplicate clusters
- GET /api/knowledge/mound/dedup/report - Generate dedup report
- POST /api/knowledge/mound/dedup/merge - Merge a duplicate cluster
- POST /api/knowledge/mound/dedup/auto-merge - Auto-merge exact duplicates

Pruning endpoints:
- GET /api/knowledge/mound/pruning/items - Get prunable items
- POST /api/knowledge/mound/pruning/execute - Prune specified items
- POST /api/knowledge/mound/pruning/auto - Run auto-prune with policy
- GET /api/knowledge/mound/pruning/history - Get pruning history
- POST /api/knowledge/mound/pruning/restore - Restore archived item
- POST /api/knowledge/mound/pruning/decay - Apply confidence decay
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from aragora.server.http_utils import run_async as _run_async

from ...base import (
    BaseHandler,
    HandlerResult,
    error_response,
)
from ...utils.rate_limit import RateLimiter, get_client_ip

from .culture import CultureOperationsMixin
from .dedup import DedupOperationsMixin
from .export import ExportOperationsMixin
from .federation import FederationOperationsMixin
from .global_knowledge import GlobalKnowledgeOperationsMixin
from .graph import GraphOperationsMixin
from .nodes import NodeOperationsMixin
from .pruning import PruningOperationsMixin
from .relationships import RelationshipOperationsMixin
from .sharing import SharingOperationsMixin
from .staleness import StalenessOperationsMixin
from .sync import SyncOperationsMixin
from .visibility import VisibilityOperationsMixin

if TYPE_CHECKING:
    from aragora.knowledge.mound import KnowledgeMound

logger = logging.getLogger(__name__)

# Rate limiter for knowledge endpoints (60 requests per minute)
_knowledge_limiter = RateLimiter(requests_per_minute=60)


class KnowledgeMoundHandler(
    NodeOperationsMixin,
    RelationshipOperationsMixin,
    GraphOperationsMixin,
    CultureOperationsMixin,
    StalenessOperationsMixin,
    SyncOperationsMixin,
    ExportOperationsMixin,
    VisibilityOperationsMixin,
    SharingOperationsMixin,
    GlobalKnowledgeOperationsMixin,
    FederationOperationsMixin,
    DedupOperationsMixin,
    PruningOperationsMixin,
    BaseHandler,
):
    """Handler for Knowledge Mound API endpoints (unified knowledge storage).

    Combines mixins for:
    - Node CRUD operations (NodeOperationsMixin)
    - Relationship management (RelationshipOperationsMixin)
    - Graph traversal (GraphOperationsMixin)
    - Culture management (CultureOperationsMixin)
    - Staleness detection (StalenessOperationsMixin)
    - Legacy sync (SyncOperationsMixin)
    - Graph export (ExportOperationsMixin)
    - Visibility control (VisibilityOperationsMixin)
    - Cross-workspace sharing (SharingOperationsMixin)
    - Global/public knowledge (GlobalKnowledgeOperationsMixin)
    - Multi-region federation (FederationOperationsMixin)
    - Deduplication (DedupOperationsMixin)
    - Pruning and archival (PruningOperationsMixin)
    """

    ROUTES = [
        "/api/knowledge/mound/query",
        "/api/knowledge/mound/nodes",
        "/api/knowledge/mound/relationships",
        "/api/knowledge/mound/stats",
        "/api/knowledge/mound/culture",
        "/api/knowledge/mound/culture/*",
        "/api/knowledge/mound/stale",
        "/api/knowledge/mound/revalidate/*",
        "/api/knowledge/mound/schedule-revalidation",
        "/api/knowledge/mound/sync/*",
        "/api/knowledge/mound/graph/*/lineage",
        "/api/knowledge/mound/graph/*/related",
        "/api/knowledge/mound/export/d3",
        "/api/knowledge/mound/export/graphml",
        # Deduplication
        "/api/knowledge/mound/dedup/clusters",
        "/api/knowledge/mound/dedup/report",
        "/api/knowledge/mound/dedup/merge",
        "/api/knowledge/mound/dedup/auto-merge",
        # Pruning
        "/api/knowledge/mound/pruning/items",
        "/api/knowledge/mound/pruning/execute",
        "/api/knowledge/mound/pruning/auto",
        "/api/knowledge/mound/pruning/history",
        "/api/knowledge/mound/pruning/restore",
        "/api/knowledge/mound/pruning/decay",
    ]

    def __init__(self, server_context: dict):
        """Initialize knowledge mound handler."""
        super().__init__(server_context)
        self._mound: Optional["KnowledgeMound"] = None
        self._mound_initialized = False

    def _get_mound(self) -> Optional["KnowledgeMound"]:
        """Get or create Knowledge Mound instance."""
        if self._mound is None:
            from aragora.knowledge.mound import KnowledgeMound

            self._mound = KnowledgeMound(workspace_id="default")
            try:
                _run_async(self._mound.initialize())
                self._mound_initialized = True
            except (RuntimeError, OSError, ValueError) as e:
                logger.exception("Failed to initialize Knowledge Mound: %s", e)
                self._mound = None
        return self._mound

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path.startswith("/api/knowledge/mound/")

    def handle(self, path: str, query_params: dict, handler: Any) -> Optional[HandlerResult]:
        """Route knowledge mound requests to appropriate methods."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _knowledge_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for mound endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # Semantic query
        if path == "/api/knowledge/mound/query":
            return self._handle_mound_query(handler)

        # Nodes listing/creation
        if path == "/api/knowledge/mound/nodes":
            method = getattr(handler, "command", "GET")
            if method == "POST":
                return self._handle_create_node(handler)
            return self._handle_list_nodes(query_params)

        # Relationships
        if path == "/api/knowledge/mound/relationships":
            return self._handle_create_relationship(handler)

        # Statistics
        if path == "/api/knowledge/mound/stats":
            return self._handle_mound_stats(query_params)

        # Dynamic routes
        if path.startswith("/api/knowledge/mound/nodes/"):
            return self._handle_node_routes(path, query_params, handler)

        if path.startswith("/api/knowledge/mound/graph/"):
            if "/lineage" in path:
                return self._handle_graph_lineage(path, query_params)
            if "/related" in path:
                return self._handle_graph_related(path, query_params)
            return self._handle_graph_traversal(path, query_params)

        if path == "/api/knowledge/mound/index/repository":
            return self._handle_index_repository(handler)

        # Culture endpoints
        if path == "/api/knowledge/mound/culture":
            return self._handle_get_culture(query_params)

        if path == "/api/knowledge/mound/culture/documents":
            return self._handle_add_culture_document(handler)

        if path == "/api/knowledge/mound/culture/promote":
            return self._handle_promote_to_culture(handler)

        # Staleness endpoints
        if path == "/api/knowledge/mound/stale":
            return self._handle_get_stale(query_params)

        if path.startswith("/api/knowledge/mound/revalidate/"):
            node_id = path.split("/")[-1]
            return self._handle_revalidate_node(node_id, handler)

        if path == "/api/knowledge/mound/schedule-revalidation":
            return self._handle_schedule_revalidation(handler)

        # Sync endpoints
        if path == "/api/knowledge/mound/sync/continuum":
            return self._handle_sync_continuum(handler)

        if path == "/api/knowledge/mound/sync/consensus":
            return self._handle_sync_consensus(handler)

        if path == "/api/knowledge/mound/sync/facts":
            return self._handle_sync_facts(handler)

        # Graph export endpoints
        if path == "/api/knowledge/mound/export/d3":
            return self._handle_export_d3(query_params)

        if path == "/api/knowledge/mound/export/graphml":
            return self._handle_export_graphml(query_params)

        # Sharing endpoints
        if path == "/api/knowledge/mound/share":
            method = getattr(handler, "command", "POST")
            if method == "POST":
                return self._handle_share_item(handler)
            elif method == "DELETE":
                return self._handle_revoke_share(handler)
            elif method == "PATCH":
                return self._handle_update_share(handler)

        if path == "/api/knowledge/mound/shared-with-me":
            return self._handle_shared_with_me(query_params, handler)

        if path == "/api/knowledge/mound/my-shares":
            return self._handle_my_shares(query_params, handler)

        # Global knowledge endpoints
        if path == "/api/knowledge/mound/global":
            method = getattr(handler, "command", "GET")
            if method == "POST":
                return self._handle_store_verified_fact(handler)
            return self._handle_query_global(query_params)

        if path == "/api/knowledge/mound/global/promote":
            return self._handle_promote_to_global(handler)

        if path == "/api/knowledge/mound/global/facts":
            return self._handle_get_system_facts(query_params)

        if path == "/api/knowledge/mound/global/workspace-id":
            return self._handle_get_system_workspace_id()

        # Federation endpoints
        if path == "/api/knowledge/mound/federation/regions":
            method = getattr(handler, "command", "GET")
            if method == "POST":
                return self._handle_register_region(handler)
            return self._handle_list_regions(query_params)

        if path.startswith("/api/knowledge/mound/federation/regions/"):
            region_id = path.split("/")[-1]
            method = getattr(handler, "command", "DELETE")
            if method == "DELETE":
                return self._handle_unregister_region(region_id, handler)

        if path == "/api/knowledge/mound/federation/sync/push":
            return self._handle_sync_to_region(handler)

        if path == "/api/knowledge/mound/federation/sync/pull":
            return self._handle_pull_from_region(handler)

        if path == "/api/knowledge/mound/federation/sync/all":
            return self._handle_sync_all_regions(handler)

        if path == "/api/knowledge/mound/federation/status":
            return self._handle_get_federation_status(query_params)

        # Deduplication endpoints
        if path == "/api/knowledge/mound/dedup/clusters":
            return self._handle_get_duplicate_clusters(query_params)

        if path == "/api/knowledge/mound/dedup/report":
            return self._handle_get_dedup_report(query_params)

        if path == "/api/knowledge/mound/dedup/merge":
            return self._handle_merge_duplicate_cluster(handler)

        if path == "/api/knowledge/mound/dedup/auto-merge":
            return self._handle_auto_merge_exact_duplicates(handler)

        # Pruning endpoints
        if path == "/api/knowledge/mound/pruning/items":
            return self._handle_get_prunable_items(query_params)

        if path == "/api/knowledge/mound/pruning/execute":
            return self._handle_execute_prune(handler)

        if path == "/api/knowledge/mound/pruning/auto":
            return self._handle_auto_prune(handler)

        if path == "/api/knowledge/mound/pruning/history":
            return self._handle_get_prune_history(query_params)

        if path == "/api/knowledge/mound/pruning/restore":
            return self._handle_restore_pruned_item(handler)

        if path == "/api/knowledge/mound/pruning/decay":
            return self._handle_apply_confidence_decay(handler)

        return None

    def _handle_node_routes(
        self, path: str, query_params: dict, handler: Any
    ) -> Optional[HandlerResult]:
        """Handle /api/knowledge/mound/nodes/:id routes."""
        parts = path.strip("/").split("/")
        if len(parts) >= 5:
            node_id = parts[4]
            method = getattr(handler, "command", "GET")

            # /api/knowledge/mound/nodes/:id/relationships
            if len(parts) >= 6 and parts[5] == "relationships":
                return self._handle_get_node_relationships(node_id, query_params)

            # /api/knowledge/mound/nodes/:id/visibility
            if len(parts) >= 6 and parts[5] == "visibility":
                if method == "PUT":
                    return self._handle_set_visibility(node_id, handler)
                return self._handle_get_visibility(node_id)

            # /api/knowledge/mound/nodes/:id/access
            if len(parts) >= 6 and parts[5] == "access":
                if method == "POST":
                    return self._handle_grant_access(node_id, handler)
                elif method == "DELETE":
                    return self._handle_revoke_access(node_id, handler)
                return self._handle_list_access_grants(node_id, query_params)

            # /api/knowledge/mound/nodes/:id - Get specific node
            return self._handle_get_node(node_id)
        return error_response("Invalid node path", 400)

    # -------------------------------------------------------------------------
    # Deduplication handler methods
    # -------------------------------------------------------------------------

    def _handle_get_duplicate_clusters(self, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/mound/dedup/clusters."""
        workspace_id = query_params.get("workspace_id", "default")
        similarity_threshold = float(query_params.get("similarity_threshold", 0.9))
        limit = int(query_params.get("limit", 100))
        return _run_async(
            self.get_duplicate_clusters(
                workspace_id=workspace_id,
                similarity_threshold=similarity_threshold,
                limit=limit,
            )
        )

    def _handle_get_dedup_report(self, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/mound/dedup/report."""
        workspace_id = query_params.get("workspace_id", "default")
        similarity_threshold = float(query_params.get("similarity_threshold", 0.9))
        return _run_async(
            self.get_dedup_report(
                workspace_id=workspace_id,
                similarity_threshold=similarity_threshold,
            )
        )

    def _handle_merge_duplicate_cluster(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/dedup/merge."""
        try:
            import json
            body = json.loads(handler.request.body.decode("utf-8")) if handler.request.body else {}
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as e:
            logger.warning("Failed to parse request body in merge_duplicate_cluster: %s", e)
            body = {}

        workspace_id = body.get("workspace_id", "default")
        cluster_id = body.get("cluster_id")
        primary_node_id = body.get("primary_node_id")
        archive = body.get("archive", True)

        if not cluster_id:
            return error_response("cluster_id is required", 400)

        return _run_async(
            self.merge_duplicate_cluster(
                workspace_id=workspace_id,
                cluster_id=cluster_id,
                primary_node_id=primary_node_id,
                archive=archive,
            )
        )

    def _handle_auto_merge_exact_duplicates(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/dedup/auto-merge."""
        try:
            import json
            body = json.loads(handler.request.body.decode("utf-8")) if handler.request.body else {}
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as e:
            logger.warning("Failed to parse request body in auto_merge_exact_duplicates: %s", e)
            body = {}

        workspace_id = body.get("workspace_id", "default")
        dry_run = body.get("dry_run", True)

        return _run_async(
            self.auto_merge_exact_duplicates(
                workspace_id=workspace_id,
                dry_run=dry_run,
            )
        )

    # -------------------------------------------------------------------------
    # Pruning handler methods
    # -------------------------------------------------------------------------

    def _handle_get_prunable_items(self, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/mound/pruning/items."""
        workspace_id = query_params.get("workspace_id", "default")
        staleness_threshold = float(query_params.get("staleness_threshold", 0.9))
        min_age_days = int(query_params.get("min_age_days", 30))
        limit = int(query_params.get("limit", 100))
        return _run_async(
            self.get_prunable_items(
                workspace_id=workspace_id,
                staleness_threshold=staleness_threshold,
                min_age_days=min_age_days,
                limit=limit,
            )
        )

    def _handle_execute_prune(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/pruning/execute."""
        try:
            import json
            body = json.loads(handler.request.body.decode("utf-8")) if handler.request.body else {}
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as e:
            logger.warning("Failed to parse request body in execute_prune: %s", e)
            body = {}

        workspace_id = body.get("workspace_id", "default")
        item_ids = body.get("item_ids", [])
        action = body.get("action", "archive")
        reason = body.get("reason", "manual_prune")

        if not item_ids:
            return error_response("item_ids is required", 400)

        return _run_async(
            self.execute_prune(
                workspace_id=workspace_id,
                item_ids=item_ids,
                action=action,
                reason=reason,
            )
        )

    def _handle_auto_prune(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/pruning/auto."""
        try:
            import json
            body = json.loads(handler.request.body.decode("utf-8")) if handler.request.body else {}
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as e:
            logger.warning("Failed to parse request body in auto_prune: %s", e)
            body = {}

        workspace_id = body.get("workspace_id", "default")
        policy_id = body.get("policy_id")
        staleness_threshold = float(body.get("staleness_threshold", 0.9))
        min_age_days = int(body.get("min_age_days", 30))
        action = body.get("action", "archive")
        dry_run = body.get("dry_run", True)

        return _run_async(
            self.auto_prune(
                workspace_id=workspace_id,
                policy_id=policy_id,
                staleness_threshold=staleness_threshold,
                min_age_days=min_age_days,
                action=action,
                dry_run=dry_run,
            )
        )

    def _handle_get_prune_history(self, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/mound/pruning/history."""
        workspace_id = query_params.get("workspace_id", "default")
        limit = int(query_params.get("limit", 50))
        since = query_params.get("since")
        return _run_async(
            self.get_prune_history(
                workspace_id=workspace_id,
                limit=limit,
                since=since,
            )
        )

    def _handle_restore_pruned_item(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/pruning/restore."""
        try:
            import json
            body = json.loads(handler.request.body.decode("utf-8")) if handler.request.body else {}
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as e:
            logger.warning("Failed to parse request body in restore_pruned_item: %s", e)
            body = {}

        workspace_id = body.get("workspace_id", "default")
        node_id = body.get("node_id")

        if not node_id:
            return error_response("node_id is required", 400)

        return _run_async(
            self.restore_pruned_item(
                workspace_id=workspace_id,
                node_id=node_id,
            )
        )

    def _handle_apply_confidence_decay(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/pruning/decay."""
        try:
            import json
            body = json.loads(handler.request.body.decode("utf-8")) if handler.request.body else {}
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as e:
            logger.warning("Failed to parse request body in apply_confidence_decay: %s", e)
            body = {}

        workspace_id = body.get("workspace_id", "default")
        decay_rate = float(body.get("decay_rate", 0.01))
        min_confidence = float(body.get("min_confidence", 0.1))

        return _run_async(
            self.apply_confidence_decay(
                workspace_id=workspace_id,
                decay_rate=decay_rate,
                min_confidence=min_confidence,
            )
        )
