"""
Graph Operations Mixin for Knowledge Mound Handler.

Provides graph traversal operations:
- Basic graph traversal
- Node lineage (derived_from chain)
- Related nodes (immediate neighbors)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

from aragora.knowledge.unified.types import RelationshipType
from aragora.rbac.decorators import require_permission
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
    from aragora.knowledge.mound import GraphQueryResult, KnowledgeMound

logger = logging.getLogger(__name__)


def _parse_relationship_type(value: str | None) -> RelationshipType | None:
    """Parse a string into a RelationshipType enum value, or return None if invalid."""
    if value is None:
        return None
    try:
        return RelationshipType(value)
    except ValueError:
        # Try uppercase variant
        try:
            return RelationshipType[value.upper()]
        except KeyError:
            return None


class GraphHandlerProtocol(Protocol):
    """Protocol for handlers that use GraphOperationsMixin."""

    def _get_mound(self) -> KnowledgeMound | None: ...


class GraphOperationsMixin:
    """Mixin providing graph operations for KnowledgeMoundHandler."""

    @require_permission("knowledge:read")
    @handle_errors("graph traversal")
    def _handle_graph_traversal(
        self: GraphHandlerProtocol, path: str, query_params: dict[str, Any]
    ) -> HandlerResult:
        """Handle GET /api/knowledge/mound/graph/:id - Graph traversal."""

        parts = path.strip("/").split("/")
        if len(parts) < 5:
            return error_response("Node ID required", 400)

        node_id = parts[4]
        relationship_type_str = get_bounded_string_param(
            query_params, "relationship_type", None, max_length=50
        )
        depth = get_clamped_int_param(query_params, "depth", 2, min_val=1, max_val=5)
        max_nodes = get_clamped_int_param(query_params, "max_nodes", 50, min_val=1, max_val=1000)

        # Parse relationship type if provided
        relationship_type = _parse_relationship_type(relationship_type_str)
        relationship_types: list[RelationshipType] | None = None
        if relationship_type is not None:
            relationship_types = [relationship_type]

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            result: GraphQueryResult = _run_async(
                mound.query_graph(
                    start_id=node_id,
                    relationship_types=relationship_types,
                    depth=depth,
                    max_nodes=max_nodes,
                )
            )
        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            return error_response(f"Graph traversal failed: {e}", 500)

        return json_response(
            {
                "start_node_id": node_id,
                "depth": depth,
                "max_nodes": max_nodes,
                "relationship_type": relationship_type_str,
                "nodes": [n.to_dict() for n in result.nodes],
                "count": len(result.nodes),
            }
        )

    @require_permission("knowledge:read")
    @handle_errors("graph lineage")
    def _handle_graph_lineage(
        self: GraphHandlerProtocol, path: str, query_params: dict[str, Any]
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
            result: GraphQueryResult = _run_async(
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
                    "edges": [e.to_dict() for e in result.edges],
                    "total_nodes": result.total_nodes,
                    "total_edges": result.total_edges,
                },
                "depth": depth,
            }
        )

    @require_permission("knowledge:read")
    @handle_errors("graph related")
    def _handle_graph_related(
        self: GraphHandlerProtocol, path: str, query_params: dict[str, Any]
    ) -> HandlerResult:
        """Handle GET /api/knowledge/mound/graph/:id/related - Get related nodes."""

        parts = path.strip("/").split("/")
        if len(parts) < 5:
            return error_response("Node ID required", 400)

        node_id = parts[4]
        relationship_type_str = get_bounded_string_param(
            query_params, "relationship_type", None, max_length=50
        )
        limit = get_clamped_int_param(query_params, "limit", 20, min_val=1, max_val=100)

        # Parse relationship type if provided
        relationship_type = _parse_relationship_type(relationship_type_str)
        relationship_types: list[RelationshipType] | None = None
        if relationship_type is not None:
            relationship_types = [relationship_type]

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            result: GraphQueryResult = _run_async(
                mound.query_graph(
                    start_id=node_id,
                    relationship_types=relationship_types,
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
                "relationship_type": relationship_type_str,
                "total": len(result.nodes) - 1,
            }
        )
