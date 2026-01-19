"""
Relationship Operations Mixin for Knowledge Mound Handler.

Provides relationship management operations:
- Create relationships between nodes
- Get node relationships
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Protocol

from aragora.server.http_utils import run_async as _run_async

from ...base import (
    HandlerResult,
    error_response,
    get_bounded_string_param,
    handle_errors,
    json_response,
)

if TYPE_CHECKING:
    from aragora.knowledge.mound import KnowledgeMound

logger = logging.getLogger(__name__)


class RelationshipHandlerProtocol(Protocol):
    """Protocol for handlers that use RelationshipOperationsMixin."""

    def _get_mound(self) -> "KnowledgeMound | None": ...
    def require_auth_or_error(self, handler: Any) -> tuple[Any, HandlerResult | None]: ...


class RelationshipOperationsMixin:
    """Mixin providing relationship operations for KnowledgeMoundHandler."""

    @handle_errors("get node relationships")
    def _handle_get_node_relationships(
        self: RelationshipHandlerProtocol, node_id: str, query_params: dict
    ) -> HandlerResult:
        """Handle GET /api/knowledge/mound/nodes/:id/relationships - Get node relationships."""

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            node = _run_async(mound.get_node(node_id))
        except Exception as e:
            logger.error(f"Failed to get node: {e}")
            return error_response(f"Failed to get node: {e}", 500)

        if not node:
            return error_response(f"Node not found: {node_id}", 404)

        relationship_type = get_bounded_string_param(
            query_params, "relationship_type", None, max_length=50
        )
        direction = get_bounded_string_param(
            query_params, "direction", "both", max_length=20
        )

        if direction not in ("outgoing", "incoming", "both"):
            return error_response(
                "direction must be 'outgoing', 'incoming', or 'both'", 400
            )

        try:
            relationships = _run_async(
                mound.get_relationships(
                    node_id=node_id,
                    relationship_type=relationship_type,
                    direction=direction,
                )
            )
        except Exception as e:
            logger.error(f"Failed to get relationships: {e}")
            return error_response(f"Failed to get relationships: {e}", 500)

        return json_response({
            "node_id": node_id,
            "relationships": [
                {
                    "id": rel.id,
                    "from_node_id": rel.from_node_id,
                    "to_node_id": rel.to_node_id,
                    "relationship_type": rel.relationship_type,
                    "strength": rel.strength,
                    "created_at": rel.created_at.isoformat() if rel.created_at else None,
                    "created_by": rel.created_by,
                    "metadata": rel.metadata,
                }
                for rel in relationships
            ],
            "count": len(relationships),
            "direction": direction,
        })

    @handle_errors("create relationship")
    def _handle_create_relationship(self: RelationshipHandlerProtocol, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/relationships - Add relationship."""
        user, err = self.require_auth_or_error(handler)
        if err:
            return err

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                return error_response("Request body required", 400)
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        from_node_id = data.get("from_node_id")
        to_node_id = data.get("to_node_id")
        relationship_type = data.get("relationship_type")

        if not from_node_id:
            return error_response("from_node_id is required", 400)
        if not to_node_id:
            return error_response("to_node_id is required", 400)
        if not relationship_type:
            return error_response("relationship_type is required", 400)

        valid_types = ("supports", "contradicts", "derived_from", "related_to", "supersedes")
        if relationship_type not in valid_types:
            return error_response(f"Invalid relationship_type. Must be one of: {valid_types}", 400)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            rel_id = _run_async(
                mound.add_relationship(
                    from_node_id=from_node_id,
                    to_node_id=to_node_id,
                    relationship_type=relationship_type,
                    strength=data.get("strength", 1.0),
                    created_by=data.get("created_by", ""),
                    metadata=data.get("metadata"),
                )
            )
        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            return error_response(f"Failed to create relationship: {e}", 500)

        return json_response({
            "id": rel_id,
            "from_node_id": from_node_id,
            "to_node_id": to_node_id,
            "relationship_type": relationship_type,
        }, status=201)
