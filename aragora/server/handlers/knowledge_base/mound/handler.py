"""
Main Knowledge Mound Handler.

Combines all mixins to provide the complete Knowledge Mound API:

Knowledge Mound API (unified knowledge storage):
- POST /api/knowledge/mound/query - Semantic query against knowledge mound
- POST /api/knowledge/mound/nodes - Add a knowledge node
- GET /api/knowledge/mound/nodes/:id - Get specific node
- GET /api/knowledge/mound/nodes/:id/relationships - Get relationships for a node
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
from .export import ExportOperationsMixin
from .graph import GraphOperationsMixin
from .nodes import NodeOperationsMixin
from .relationships import RelationshipOperationsMixin
from .staleness import StalenessOperationsMixin
from .sync import SyncOperationsMixin

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
            except Exception as e:
                logger.error(f"Failed to initialize Knowledge Mound: {e}")
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

        return None

    def _handle_node_routes(
        self, path: str, query_params: dict, handler: Any
    ) -> Optional[HandlerResult]:
        """Handle /api/knowledge/mound/nodes/:id routes."""
        parts = path.strip("/").split("/")
        if len(parts) >= 5:
            node_id = parts[4]
            if len(parts) >= 6 and parts[5] == "relationships":
                return self._handle_get_node_relationships(node_id, query_params)
            return self._handle_get_node(node_id)
        return error_response("Invalid node path", 400)
