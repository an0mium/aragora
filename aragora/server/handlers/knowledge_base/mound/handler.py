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

Phase A2 - Contradiction Detection endpoints:
- POST /api/knowledge/mound/contradictions/detect - Trigger contradiction scan
- GET /api/knowledge/mound/contradictions - List unresolved contradictions
- POST /api/knowledge/mound/contradictions/:id/resolve - Resolve a contradiction
- GET /api/knowledge/mound/contradictions/stats - Get contradiction statistics

Phase A2 - Governance (RBAC + Audit) endpoints:
- POST /api/knowledge/mound/governance/roles - Create a role
- POST /api/knowledge/mound/governance/roles/assign - Assign role to user
- POST /api/knowledge/mound/governance/roles/revoke - Revoke role from user
- GET /api/knowledge/mound/governance/permissions/:user_id - Get user permissions
- POST /api/knowledge/mound/governance/permissions/check - Check permission
- GET /api/knowledge/mound/governance/audit - Query audit trail
- GET /api/knowledge/mound/governance/audit/user/:user_id - Get user activity
- GET /api/knowledge/mound/governance/stats - Get governance stats

Phase A2 - Analytics endpoints:
- GET /api/knowledge/mound/analytics/coverage - Domain coverage analysis
- GET /api/knowledge/mound/analytics/usage - Usage pattern analysis
- POST /api/knowledge/mound/analytics/usage/record - Record usage event
- POST /api/knowledge/mound/analytics/quality/snapshot - Capture quality snapshot
- GET /api/knowledge/mound/analytics/quality/trend - Quality trend over time
- GET /api/knowledge/mound/analytics/stats - Analytics statistics

Phase A2 - Extraction endpoints:
- POST /api/knowledge/mound/extraction/debate - Extract from a debate
- POST /api/knowledge/mound/extraction/promote - Promote extracted claims
- GET /api/knowledge/mound/extraction/stats - Get extraction statistics

Phase A2 - Confidence Decay endpoints:
- POST /api/knowledge/mound/confidence/decay - Apply confidence decay
- POST /api/knowledge/mound/confidence/event - Record confidence event
- GET /api/knowledge/mound/confidence/history - Get adjustment history
- GET /api/knowledge/mound/confidence/stats - Get decay statistics
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

from .analytics import AnalyticsOperationsMixin
from .confidence_decay import ConfidenceDecayOperationsMixin
from .contradiction import ContradictionOperationsMixin
from .culture import CultureOperationsMixin
from .curation import CurationOperationsMixin
from .dashboard import DashboardOperationsMixin
from .dedup import DedupOperationsMixin
from .export import ExportOperationsMixin
from .extraction import ExtractionOperationsMixin
from .federation import FederationOperationsMixin
from .global_knowledge import GlobalKnowledgeOperationsMixin
from .governance import GovernanceOperationsMixin
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


class KnowledgeMoundHandler(  # type: ignore[misc]
    NodeOperationsMixin,
    RelationshipOperationsMixin,
    GraphOperationsMixin,
    CultureOperationsMixin,
    CurationOperationsMixin,
    StalenessOperationsMixin,
    SyncOperationsMixin,
    ExportOperationsMixin,
    VisibilityOperationsMixin,
    SharingOperationsMixin,
    GlobalKnowledgeOperationsMixin,
    FederationOperationsMixin,
    DedupOperationsMixin,
    PruningOperationsMixin,
    DashboardOperationsMixin,
    ContradictionOperationsMixin,
    GovernanceOperationsMixin,
    AnalyticsOperationsMixin,
    ExtractionOperationsMixin,
    ConfidenceDecayOperationsMixin,
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
        "/api/v1/knowledge/mound/query",
        "/api/v1/knowledge/mound/nodes",
        "/api/v1/knowledge/mound/relationships",
        "/api/v1/knowledge/mound/stats",
        "/api/v1/knowledge/mound/culture",
        "/api/v1/knowledge/mound/culture/*",
        "/api/v1/knowledge/mound/stale",
        "/api/v1/knowledge/mound/revalidate/*",
        "/api/v1/knowledge/mound/schedule-revalidation",
        "/api/v1/knowledge/mound/sync/*",
        "/api/v1/knowledge/mound/graph/*/lineage",
        "/api/v1/knowledge/mound/graph/*/related",
        "/api/v1/knowledge/mound/export/d3",
        "/api/v1/knowledge/mound/export/graphml",
        # Deduplication
        "/api/v1/knowledge/mound/dedup/clusters",
        "/api/v1/knowledge/mound/dedup/report",
        "/api/v1/knowledge/mound/dedup/merge",
        "/api/v1/knowledge/mound/dedup/auto-merge",
        # Pruning
        "/api/v1/knowledge/mound/pruning/items",
        "/api/v1/knowledge/mound/pruning/execute",
        "/api/v1/knowledge/mound/pruning/auto",
        "/api/v1/knowledge/mound/pruning/history",
        "/api/v1/knowledge/mound/pruning/restore",
        "/api/v1/knowledge/mound/pruning/decay",
        # Dashboard and metrics
        "/api/v1/knowledge/mound/dashboard/health",
        "/api/v1/knowledge/mound/dashboard/metrics",
        "/api/v1/knowledge/mound/dashboard/metrics/reset",
        "/api/v1/knowledge/mound/dashboard/adapters",
        "/api/v1/knowledge/mound/dashboard/queries",
        "/api/v1/knowledge/mound/dashboard/batcher",
        # Auto-curation (Phase 4)
        "/api/v1/knowledge/mound/curation/policy",
        "/api/v1/knowledge/mound/curation/status",
        "/api/v1/knowledge/mound/curation/run",
        "/api/v1/knowledge/mound/curation/history",
        "/api/v1/knowledge/mound/curation/scores",
        "/api/v1/knowledge/mound/curation/tiers",
        # Phase A2 - Contradiction detection
        "/api/v1/knowledge/mound/contradictions/detect",
        "/api/v1/knowledge/mound/contradictions",
        "/api/v1/knowledge/mound/contradictions/*/resolve",
        "/api/v1/knowledge/mound/contradictions/stats",
        # Phase A2 - Governance (RBAC + Audit)
        "/api/v1/knowledge/mound/governance/roles",
        "/api/v1/knowledge/mound/governance/roles/assign",
        "/api/v1/knowledge/mound/governance/roles/revoke",
        "/api/v1/knowledge/mound/governance/permissions/*",
        "/api/v1/knowledge/mound/governance/permissions/check",
        "/api/v1/knowledge/mound/governance/audit",
        "/api/v1/knowledge/mound/governance/audit/user/*",
        "/api/v1/knowledge/mound/governance/stats",
        # Phase A2 - Analytics
        "/api/v1/knowledge/mound/analytics/coverage",
        "/api/v1/knowledge/mound/analytics/usage",
        "/api/v1/knowledge/mound/analytics/usage/record",
        "/api/v1/knowledge/mound/analytics/quality/snapshot",
        "/api/v1/knowledge/mound/analytics/quality/trend",
        "/api/v1/knowledge/mound/analytics/stats",
        # Phase A2 - Extraction
        "/api/v1/knowledge/mound/extraction/debate",
        "/api/v1/knowledge/mound/extraction/promote",
        "/api/v1/knowledge/mound/extraction/stats",
        # Phase A2 - Confidence decay
        "/api/v1/knowledge/mound/confidence/decay",
        "/api/v1/knowledge/mound/confidence/event",
        "/api/v1/knowledge/mound/confidence/history",
        "/api/v1/knowledge/mound/confidence/stats",
    ]

    def __init__(self, server_context: dict):
        """Initialize knowledge mound handler."""
        super().__init__(server_context)  # type: ignore[arg-type]
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
        return path.startswith("/api/v1/knowledge/mound/")

    def handle(self, path: str, query_params: dict, handler: Any) -> Optional[HandlerResult]:
        """Route knowledge mound requests to appropriate methods."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _knowledge_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for mound endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # Require authentication for all knowledge mound endpoints
        try:
            user, err = self.require_auth_or_error(handler)
            if err:
                return err
        except Exception as e:
            logger.warning(f"Authentication failed for knowledge mound: {e}")
            return error_response("Authentication required", 401)

        # Semantic query
        if path == "/api/v1/knowledge/mound/query":
            return self._handle_mound_query(handler)

        # Nodes listing/creation
        if path == "/api/v1/knowledge/mound/nodes":
            method = getattr(handler, "command", "GET")
            if method == "POST":
                return self._handle_create_node(handler)
            return self._handle_list_nodes(query_params)

        # Relationships
        if path == "/api/v1/knowledge/mound/relationships":
            return self._handle_create_relationship(handler)

        # Statistics
        if path == "/api/v1/knowledge/mound/stats":
            return self._handle_mound_stats(query_params)

        # Dynamic routes
        if path.startswith("/api/v1/knowledge/mound/nodes/"):
            return self._handle_node_routes(path, query_params, handler)

        if path.startswith("/api/v1/knowledge/mound/graph/"):
            if "/lineage" in path:
                return self._handle_graph_lineage(path, query_params)
            if "/related" in path:
                return self._handle_graph_related(path, query_params)
            return self._handle_graph_traversal(path, query_params)

        if path == "/api/v1/knowledge/mound/index/repository":
            return self._handle_index_repository(handler)

        # Culture endpoints
        if path == "/api/v1/knowledge/mound/culture":
            return self._handle_get_culture(query_params)

        if path == "/api/v1/knowledge/mound/culture/documents":
            return self._handle_add_culture_document(handler)

        if path == "/api/v1/knowledge/mound/culture/promote":
            return self._handle_promote_to_culture(handler)

        # Staleness endpoints
        if path == "/api/v1/knowledge/mound/stale":
            return self._handle_get_stale(query_params)

        if path.startswith("/api/v1/knowledge/mound/revalidate/"):
            node_id = path.split("/")[-1]
            return self._handle_revalidate_node(node_id, handler)

        if path == "/api/v1/knowledge/mound/schedule-revalidation":
            return self._handle_schedule_revalidation(handler)

        # Auto-curation endpoints (Phase 4)
        if path.startswith("/api/v1/knowledge/mound/curation/"):
            return self._handle_curation_routes(path, query_params, handler)

        # Sync endpoints
        if path == "/api/v1/knowledge/mound/sync/continuum":
            return self._handle_sync_continuum(handler)

        if path == "/api/v1/knowledge/mound/sync/consensus":
            return self._handle_sync_consensus(handler)

        if path == "/api/v1/knowledge/mound/sync/facts":
            return self._handle_sync_facts(handler)

        # Graph export endpoints
        if path == "/api/v1/knowledge/mound/export/d3":
            return self._handle_export_d3(query_params)

        if path == "/api/v1/knowledge/mound/export/graphml":
            return self._handle_export_graphml(query_params)

        # Sharing endpoints
        if path == "/api/v1/knowledge/mound/share":
            method = getattr(handler, "command", "POST")
            if method == "POST":
                return self._handle_share_item(handler)
            elif method == "DELETE":
                return self._handle_revoke_share(handler)
            elif method == "PATCH":
                return self._handle_update_share(handler)

        if path == "/api/v1/knowledge/mound/shared-with-me":
            return self._handle_shared_with_me(query_params, handler)

        if path == "/api/v1/knowledge/mound/my-shares":
            return self._handle_my_shares(query_params, handler)

        # Global knowledge endpoints
        if path == "/api/v1/knowledge/mound/global":
            method = getattr(handler, "command", "GET")
            if method == "POST":
                return self._handle_store_verified_fact(handler)
            return self._handle_query_global(query_params)

        if path == "/api/v1/knowledge/mound/global/promote":
            return self._handle_promote_to_global(handler)

        if path == "/api/v1/knowledge/mound/global/facts":
            return self._handle_get_system_facts(query_params)

        if path == "/api/v1/knowledge/mound/global/workspace-id":
            return self._handle_get_system_workspace_id()

        # Federation endpoints
        if path == "/api/v1/knowledge/mound/federation/regions":
            method = getattr(handler, "command", "GET")
            if method == "POST":
                return self._handle_register_region(handler)
            return self._handle_list_regions(query_params)

        if path.startswith("/api/v1/knowledge/mound/federation/regions/"):
            region_id = path.split("/")[-1]
            method = getattr(handler, "command", "DELETE")
            if method == "DELETE":
                return self._handle_unregister_region(region_id, handler)

        if path == "/api/v1/knowledge/mound/federation/sync/push":
            return self._handle_sync_to_region(handler)

        if path == "/api/v1/knowledge/mound/federation/sync/pull":
            return self._handle_pull_from_region(handler)

        if path == "/api/v1/knowledge/mound/federation/sync/all":
            return self._handle_sync_all_regions(handler)

        if path == "/api/v1/knowledge/mound/federation/status":
            return self._handle_get_federation_status(query_params)

        # Deduplication endpoints
        if path == "/api/v1/knowledge/mound/dedup/clusters":
            return self._handle_get_duplicate_clusters(query_params)

        if path == "/api/v1/knowledge/mound/dedup/report":
            return self._handle_get_dedup_report(query_params)

        if path == "/api/v1/knowledge/mound/dedup/merge":
            return self._handle_merge_duplicate_cluster(handler)

        if path == "/api/v1/knowledge/mound/dedup/auto-merge":
            return self._handle_auto_merge_exact_duplicates(handler)

        # Pruning endpoints
        if path == "/api/v1/knowledge/mound/pruning/items":
            return self._handle_get_prunable_items(query_params)

        if path == "/api/v1/knowledge/mound/pruning/execute":
            return self._handle_execute_prune(handler)

        if path == "/api/v1/knowledge/mound/pruning/auto":
            return self._handle_auto_prune(handler)

        if path == "/api/v1/knowledge/mound/pruning/history":
            return self._handle_get_prune_history(query_params)

        if path == "/api/v1/knowledge/mound/pruning/restore":
            return self._handle_restore_pruned_item(handler)

        if path == "/api/v1/knowledge/mound/pruning/decay":
            return self._handle_apply_confidence_decay(handler)

        # Dashboard and metrics endpoints
        if path == "/api/v1/knowledge/mound/dashboard/health":
            return _run_async(self.handle_dashboard_health(handler.request))

        if path == "/api/v1/knowledge/mound/dashboard/metrics":
            return _run_async(self.handle_dashboard_metrics(handler.request))

        if path == "/api/v1/knowledge/mound/dashboard/metrics/reset":
            return _run_async(self.handle_dashboard_metrics_reset(handler.request))

        if path == "/api/v1/knowledge/mound/dashboard/adapters":
            return _run_async(self.handle_dashboard_adapters(handler.request))

        if path == "/api/v1/knowledge/mound/dashboard/queries":
            return _run_async(self.handle_dashboard_queries(handler.request))

        if path == "/api/v1/knowledge/mound/dashboard/batcher":
            return _run_async(self.handle_dashboard_batcher_stats(handler.request))

        # Phase A2 - Contradiction detection endpoints
        if path == "/api/v1/knowledge/mound/contradictions/detect":
            return self._handle_detect_contradictions(handler)  # type: ignore[attr-defined]

        if path == "/api/v1/knowledge/mound/contradictions":
            return self._handle_list_contradictions(query_params)  # type: ignore[attr-defined]

        if path.startswith("/api/v1/knowledge/mound/contradictions/") and path.endswith("/resolve"):
            contradiction_id = path.split("/")[-2]
            return self._handle_resolve_contradiction(contradiction_id, handler)  # type: ignore[attr-defined]

        if path == "/api/v1/knowledge/mound/contradictions/stats":
            return _run_async(self.get_contradiction_stats())

        # Phase A2 - Governance endpoints
        if path == "/api/v1/knowledge/mound/governance/roles":
            method = getattr(handler, "command", "GET")
            if method == "POST":
                return self._handle_create_role(handler)  # type: ignore[attr-defined]
            # GET would list roles - not implemented yet

        if path == "/api/v1/knowledge/mound/governance/roles/assign":
            return self._handle_assign_role(handler)  # type: ignore[attr-defined]

        if path == "/api/v1/knowledge/mound/governance/roles/revoke":
            return self._handle_revoke_role(handler)  # type: ignore[attr-defined]

        if path.startswith("/api/v1/knowledge/mound/governance/permissions/"):
            if path == "/api/v1/knowledge/mound/governance/permissions/check":
                return self._handle_check_permission(handler)  # type: ignore[attr-defined]
            else:
                user_id = path.split("/")[-1]
                return self._handle_get_user_permissions(user_id, query_params)  # type: ignore[attr-defined]

        if path == "/api/v1/knowledge/mound/governance/audit":
            return self._handle_query_audit(query_params)  # type: ignore[attr-defined]

        if path.startswith("/api/v1/knowledge/mound/governance/audit/user/"):
            user_id = path.split("/")[-1]
            return self._handle_get_user_activity(user_id, query_params)  # type: ignore[attr-defined]

        if path == "/api/v1/knowledge/mound/governance/stats":
            return _run_async(self.get_governance_stats())

        # Phase A2 - Analytics endpoints
        if path == "/api/v1/knowledge/mound/analytics/coverage":
            return self._handle_analyze_coverage(query_params)  # type: ignore[attr-defined]

        if path == "/api/v1/knowledge/mound/analytics/usage":
            return self._handle_analyze_usage(query_params)  # type: ignore[attr-defined]

        if path == "/api/v1/knowledge/mound/analytics/usage/record":
            return self._handle_record_usage_event(handler)  # type: ignore[attr-defined]

        if path == "/api/v1/knowledge/mound/analytics/quality/snapshot":
            return self._handle_capture_quality_snapshot(handler)  # type: ignore[attr-defined]

        if path == "/api/v1/knowledge/mound/analytics/quality/trend":
            return self._handle_get_quality_trend(query_params)  # type: ignore[attr-defined]

        if path == "/api/v1/knowledge/mound/analytics/stats":
            return _run_async(self.get_analytics_stats())

        # Phase A2 - Extraction endpoints
        if path == "/api/v1/knowledge/mound/extraction/debate":
            return self._handle_extract_from_debate(handler)  # type: ignore[attr-defined]

        if path == "/api/v1/knowledge/mound/extraction/promote":
            return self._handle_promote_extracted(handler)  # type: ignore[attr-defined]

        if path == "/api/v1/knowledge/mound/extraction/stats":
            return _run_async(self.get_extraction_stats())

        # Phase A2 - Confidence decay endpoints
        if path == "/api/v1/knowledge/mound/confidence/decay":
            return self._handle_apply_confidence_decay_new(handler)  # type: ignore[attr-defined]

        if path == "/api/v1/knowledge/mound/confidence/event":
            return self._handle_record_confidence_event(handler)  # type: ignore[attr-defined]

        if path == "/api/v1/knowledge/mound/confidence/history":
            return self._handle_get_confidence_history(query_params)  # type: ignore[attr-defined]

        if path == "/api/v1/knowledge/mound/confidence/stats":
            return _run_async(self.get_decay_stats())

        return None

    def _handle_node_routes(
        self, path: str, query_params: dict, handler: Any
    ) -> Optional[HandlerResult]:
        """Handle /api/v1/knowledge/mound/nodes/:id routes."""
        parts = path.strip("/").split("/")
        # Path: /api/v1/knowledge/mound/nodes/:id
        # parts: [api, v1, knowledge, mound, nodes, :id, ...]
        if len(parts) >= 6:
            node_id = parts[5]
            method = getattr(handler, "command", "GET")

            # /api/v1/knowledge/mound/nodes/:id/relationships (7 parts)
            if len(parts) >= 7 and parts[6] == "relationships":
                return self._handle_get_node_relationships(node_id, query_params)

            # /api/v1/knowledge/mound/nodes/:id/visibility (7 parts)
            if len(parts) >= 7 and parts[6] == "visibility":
                if method == "PUT":
                    return self._handle_set_visibility(node_id, handler)
                return self._handle_get_visibility(node_id)

            # /api/v1/knowledge/mound/nodes/:id/access (7 parts)
            if len(parts) >= 7 and parts[6] == "access":
                if method == "POST":
                    return self._handle_grant_access(node_id, handler)
                elif method == "DELETE":
                    return self._handle_revoke_access(node_id, handler)
                return self._handle_list_access_grants(node_id, query_params)

            # /api/v1/knowledge/mound/nodes/:id - Get specific node (6 parts)
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

    # -------------------------------------------------------------------------
    # Phase A2 - Contradiction Detection handler methods
    # -------------------------------------------------------------------------

    def _handle_detect_contradictions(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/contradictions/detect."""
        try:
            import json

            body = json.loads(handler.request.body.decode("utf-8")) if handler.request.body else {}
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as e:
            logger.warning("Failed to parse request body in detect_contradictions: %s", e)
            body = {}

        workspace_id = body.get("workspace_id", "default")
        item_ids = body.get("item_ids")

        return _run_async(
            self.detect_contradictions(
                workspace_id=workspace_id,
                item_ids=item_ids,
            )
        )

    def _handle_list_contradictions(self, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/mound/contradictions."""
        workspace_id = query_params.get("workspace_id")
        min_severity = query_params.get("min_severity")

        return _run_async(
            self.list_contradictions(
                workspace_id=workspace_id,
                min_severity=min_severity,
            )
        )

    def _handle_resolve_contradiction(self, contradiction_id: str, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/contradictions/:id/resolve."""
        try:
            import json

            body = json.loads(handler.request.body.decode("utf-8")) if handler.request.body else {}
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as e:
            logger.warning("Failed to parse request body in resolve_contradiction: %s", e)
            body = {}

        strategy = body.get("strategy")
        resolved_by = body.get("resolved_by")
        notes = body.get("notes", "")

        if not strategy:
            return error_response("strategy is required", 400)

        return _run_async(
            self.resolve_contradiction(
                contradiction_id=contradiction_id,
                strategy=strategy,
                resolved_by=resolved_by,
                notes=notes,
            )
        )

    # -------------------------------------------------------------------------
    # Phase A2 - Governance handler methods
    # -------------------------------------------------------------------------

    def _handle_create_role(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/governance/roles."""
        try:
            import json

            body = json.loads(handler.request.body.decode("utf-8")) if handler.request.body else {}
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as e:
            logger.warning("Failed to parse request body in create_role: %s", e)
            body = {}

        name = body.get("name")
        permissions = body.get("permissions", [])
        description = body.get("description", "")
        workspace_id = body.get("workspace_id")
        created_by = body.get("created_by")

        if not name:
            return error_response("name is required", 400)

        return _run_async(
            self.create_role(
                name=name,
                permissions=permissions,
                description=description,
                workspace_id=workspace_id,
                created_by=created_by,
            )
        )

    def _handle_assign_role(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/governance/roles/assign."""
        try:
            import json

            body = json.loads(handler.request.body.decode("utf-8")) if handler.request.body else {}
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as e:
            logger.warning("Failed to parse request body in assign_role: %s", e)
            body = {}

        user_id = body.get("user_id")
        role_id = body.get("role_id")
        workspace_id = body.get("workspace_id")
        assigned_by = body.get("assigned_by")

        if not user_id or not role_id:
            return error_response("user_id and role_id are required", 400)

        return _run_async(
            self.assign_role(
                user_id=user_id,
                role_id=role_id,
                workspace_id=workspace_id,
                assigned_by=assigned_by,
            )
        )

    def _handle_revoke_role(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/governance/roles/revoke."""
        try:
            import json

            body = json.loads(handler.request.body.decode("utf-8")) if handler.request.body else {}
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as e:
            logger.warning("Failed to parse request body in revoke_role: %s", e)
            body = {}

        user_id = body.get("user_id")
        role_id = body.get("role_id")
        workspace_id = body.get("workspace_id")

        if not user_id or not role_id:
            return error_response("user_id and role_id are required", 400)

        return _run_async(
            self.revoke_role(
                user_id=user_id,
                role_id=role_id,
                workspace_id=workspace_id,
            )
        )

    def _handle_get_user_permissions(self, user_id: str, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/mound/governance/permissions/:user_id."""
        workspace_id = query_params.get("workspace_id")

        return _run_async(
            self.get_user_permissions(
                user_id=user_id,
                workspace_id=workspace_id,
            )
        )

    def _handle_check_permission(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/governance/permissions/check."""
        try:
            import json

            body = json.loads(handler.request.body.decode("utf-8")) if handler.request.body else {}
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as e:
            logger.warning("Failed to parse request body in check_permission: %s", e)
            body = {}

        user_id = body.get("user_id")
        permission = body.get("permission")
        workspace_id = body.get("workspace_id")

        if not user_id or not permission:
            return error_response("user_id and permission are required", 400)

        return _run_async(
            self.check_permission(
                user_id=user_id,
                permission=permission,
                workspace_id=workspace_id,
            )
        )

    def _handle_query_audit(self, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/mound/governance/audit."""
        actor_id = query_params.get("actor_id")
        action = query_params.get("action")
        workspace_id = query_params.get("workspace_id")
        limit = int(query_params.get("limit", 100))

        return _run_async(
            self.query_audit_trail(
                actor_id=actor_id,
                action=action,
                workspace_id=workspace_id,
                limit=limit,
            )
        )

    def _handle_get_user_activity(self, user_id: str, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/mound/governance/audit/user/:user_id."""
        days = int(query_params.get("days", 30))

        return _run_async(
            self.get_user_activity(
                user_id=user_id,
                days=days,
            )
        )

    # -------------------------------------------------------------------------
    # Phase A2 - Analytics handler methods
    # -------------------------------------------------------------------------

    def _handle_analyze_coverage(self, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/mound/analytics/coverage."""
        workspace_id = query_params.get("workspace_id", "default")
        stale_threshold_days = int(query_params.get("stale_threshold_days", 90))

        return _run_async(
            self.analyze_coverage(
                workspace_id=workspace_id,
                stale_threshold_days=stale_threshold_days,
            )
        )

    def _handle_analyze_usage(self, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/mound/analytics/usage."""
        workspace_id = query_params.get("workspace_id", "default")
        days = int(query_params.get("days", 30))

        return _run_async(
            self.analyze_usage(
                workspace_id=workspace_id,
                days=days,
            )
        )

    def _handle_record_usage_event(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/analytics/usage/record."""
        try:
            import json

            body = json.loads(handler.request.body.decode("utf-8")) if handler.request.body else {}
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as e:
            logger.warning("Failed to parse request body in record_usage_event: %s", e)
            body = {}

        event_type = body.get("event_type")
        item_id = body.get("item_id")
        user_id = body.get("user_id")
        workspace_id = body.get("workspace_id")
        query = body.get("query")

        if not event_type:
            return error_response("event_type is required", 400)

        return _run_async(
            self.record_usage_event(
                event_type=event_type,
                item_id=item_id,
                user_id=user_id,
                workspace_id=workspace_id,
                query=query,
            )
        )

    def _handle_capture_quality_snapshot(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/analytics/quality/snapshot."""
        try:
            import json

            body = json.loads(handler.request.body.decode("utf-8")) if handler.request.body else {}
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as e:
            logger.warning("Failed to parse request body in capture_quality_snapshot: %s", e)
            body = {}

        workspace_id = body.get("workspace_id", "default")

        return _run_async(
            self.capture_quality_snapshot(
                workspace_id=workspace_id,
            )
        )

    def _handle_get_quality_trend(self, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/mound/analytics/quality/trend."""
        workspace_id = query_params.get("workspace_id", "default")
        days = int(query_params.get("days", 30))

        return _run_async(
            self.get_quality_trend(
                workspace_id=workspace_id,
                days=days,
            )
        )

    # -------------------------------------------------------------------------
    # Phase A2 - Extraction handler methods
    # -------------------------------------------------------------------------

    def _handle_extract_from_debate(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/extraction/debate."""
        try:
            import json

            body = json.loads(handler.request.body.decode("utf-8")) if handler.request.body else {}
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as e:
            logger.warning("Failed to parse request body in extract_from_debate: %s", e)
            body = {}

        debate_id = body.get("debate_id")
        messages = body.get("messages", [])
        consensus_text = body.get("consensus_text")
        topic = body.get("topic")

        if not debate_id:
            return error_response("debate_id is required", 400)

        if not messages:
            return error_response("messages list is required", 400)

        return _run_async(
            self.extract_from_debate(
                debate_id=debate_id,
                messages=messages,
                consensus_text=consensus_text,
                topic=topic,
            )
        )

    def _handle_promote_extracted(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/extraction/promote."""
        try:
            import json

            body = json.loads(handler.request.body.decode("utf-8")) if handler.request.body else {}
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as e:
            logger.warning("Failed to parse request body in promote_extracted: %s", e)
            body = {}

        workspace_id = body.get("workspace_id", "default")
        min_confidence = float(body.get("min_confidence", 0.6))

        return _run_async(
            self.promote_extracted_knowledge(
                workspace_id=workspace_id,
                min_confidence=min_confidence,
            )
        )

    # -------------------------------------------------------------------------
    # Phase A2 - Confidence Decay handler methods
    # -------------------------------------------------------------------------

    def _handle_apply_confidence_decay_new(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/confidence/decay."""
        try:
            import json

            body = json.loads(handler.request.body.decode("utf-8")) if handler.request.body else {}
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as e:
            logger.warning("Failed to parse request body in confidence decay: %s", e)
            body = {}

        workspace_id = body.get("workspace_id", "default")
        force = body.get("force", False)

        return _run_async(
            self.apply_confidence_decay_endpoint(
                workspace_id=workspace_id,
                force=force,
            )
        )

    def _handle_record_confidence_event(self, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/confidence/event."""
        try:
            import json

            body = json.loads(handler.request.body.decode("utf-8")) if handler.request.body else {}
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as e:
            logger.warning("Failed to parse request body in record_confidence_event: %s", e)
            body = {}

        item_id = body.get("item_id")
        event = body.get("event")
        reason = body.get("reason", "")

        if not item_id:
            return error_response("item_id is required", 400)

        if not event:
            return error_response("event is required", 400)

        return _run_async(
            self.record_confidence_event(
                item_id=item_id,
                event=event,
                reason=reason,
            )
        )

    def _handle_get_confidence_history(self, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/mound/confidence/history."""
        item_id = query_params.get("item_id")
        event_type = query_params.get("event_type")
        limit = int(query_params.get("limit", 100))

        return _run_async(
            self.get_confidence_history(
                item_id=item_id,
                event_type=event_type,
                limit=limit,
            )
        )
