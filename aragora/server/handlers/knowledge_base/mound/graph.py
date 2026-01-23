"""
Graph Operations Mixin for Knowledge Mound Handler.

Provides graph traversal operations:
- Basic graph traversal
- Node lineage (derived_from chain)
- Related nodes (immediate neighbors)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from aragora.knowledge.unified.types import RelationshipType
from aragora.server.http_utils import run_async as _run_async

from ...base import (
    HandlerResult,
    error_response,
    get_bounded_string_param,
    get_clamped_int_param,
    handle_errors,
    json_response,
)

if TYPE_CHECKING:
    from aragora.knowledge.mound import KnowledgeMound

logger = logging.getLogger(__name__)


class GraphHandlerProtocol(Protocol):
    """Protocol for handlers that use GraphOperationsMixin."""

    def _get_mound(self) -> "KnowledgeMound | None": ...


class GraphOperationsMixin:
    """Mixin providing graph operations for KnowledgeMoundHandler."""

    @handle_errors("graph traversal")
    def _handle_graph_traversal(
        self: GraphHandlerProtocol, path: str, query_params: dict
    ) -> HandlerResult:
        """Handle GET /api/knowledge/mound/graph/:id - Graph traversal."""

        parts = path.strip("/").split("/")
        if len(parts) < 5:
            return error_response("Node ID required", 400)

        node_id = parts[4]
        relationship_type = get_bounded_string_param(
            query_params, "relationship_type", None, max_length=50
        )
        depth = get_clamped_int_param(query_params, "depth", 2, min_val=1, max_val=5)
        direction = get_bounded_string_param(query_params, "direction", "outgoing", max_length=20)

        if direction not in ("outgoing", "incoming", "both"):
            return error_response("direction must be 'outgoing', 'incoming', or 'both'", 400)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            nodes = _run_async(
                mound.query_graph(  # type: ignore[call-arg]
                    start_node_id=node_id,
                    relationship_type=relationship_type,
                    depth=depth,
                    direction=direction,
                )
            )
        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            return error_response(f"Graph traversal failed: {e}", 500)

        return json_response(
            {
                "start_node_id": node_id,
                "depth": depth,
                "direction": direction,
                "relationship_type": relationship_type,
                "nodes": [n.to_dict() for n in nodes],  # type: ignore[attr-defined]
                "count": len(nodes),  # type: ignore[arg-type]
            }
        )

    @handle_errors("graph lineage")
    def _handle_graph_lineage(
        self: GraphHandlerProtocol, path: str, query_params: dict
    ) -> HandlerResult:
        """Handle GET /api/knowledge/mound/graph/:id/lineage - Get node lineage."""

        parts = path.strip("/").split("/")
        if len(parts) < 5:
            return error_response("Node ID required", 400)

        node_id = parts[4]
        depth = get_clamped_int_param(query_params, "depth", 5, min_val=1, max_val=10)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            result = _run_async(
                mound.query_graph(
                    start_id=node_id,
                    relationship_types=[RelationshipType.DERIVED_FROM],
                    depth=depth,
                )
            )
        except Exception as e:
            logger.error(f"Graph lineage failed: {e}")
            return error_response(f"Graph lineage failed: {e}", 500)

        return json_response(
            {
                "node_id": node_id,
                "lineage": {
                    "nodes": [n.to_dict() for n in result.nodes],
                    "edges": [e.to_dict() if hasattr(e, "to_dict") else e for e in result.edges],
                    "total_nodes": result.total_nodes,
                    "total_edges": result.total_edges,
                },
                "depth": depth,
            }
        )

    @handle_errors("graph related")
    def _handle_graph_related(
        self: GraphHandlerProtocol, path: str, query_params: dict
    ) -> HandlerResult:
        """Handle GET /api/knowledge/mound/graph/:id/related - Get related nodes."""

        parts = path.strip("/").split("/")
        if len(parts) < 5:
            return error_response("Node ID required", 400)

        node_id = parts[4]
        relationship_type = get_bounded_string_param(
            query_params, "relationship_type", None, max_length=50
        )
        limit = get_clamped_int_param(query_params, "limit", 20, min_val=1, max_val=100)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            rel_types = [relationship_type] if relationship_type else None
            result = _run_async(
                mound.query_graph(  # type: ignore[call-arg]
                    start_id=node_id,
                    relationship_types=rel_types,  # type: ignore[arg-type]
                    depth=1,
                    max_nodes=limit,
                )
            )
        except Exception as e:
            logger.error(f"Get related nodes failed: {e}")
            return error_response(f"Get related nodes failed: {e}", 500)

        return json_response(
            {
                "node_id": node_id,
                "related": [n.to_dict() for n in result.nodes if n.id != node_id],
                "relationship_type": relationship_type,
                "total": len(result.nodes) - 1,
            }
        )
