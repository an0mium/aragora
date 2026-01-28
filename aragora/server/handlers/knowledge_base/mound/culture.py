"""
Culture Operations Mixin for Knowledge Mound Handler.

Provides organization culture management:
- Get culture profile
- Add culture documents
- Promote knowledge to culture
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Protocol

from aragora.rbac.decorators import require_permission
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


class CultureHandlerProtocol(Protocol):
    """Protocol for handlers that use CultureOperationsMixin."""

    def _get_mound(self) -> "KnowledgeMound | None": ...


class CultureOperationsMixin:
    """Mixin providing culture operations for KnowledgeMoundHandler."""

    @require_permission("culture:read")
    @handle_errors("get culture")
    def _handle_get_culture(self: CultureHandlerProtocol, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/mound/culture - Get organization culture profile."""

        workspace_id = get_bounded_string_param(
            query_params, "workspace_id", "default", max_length=100
        )

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            culture = _run_async(mound.get_culture_profile(workspace_id))
        except Exception as e:
            logger.error(f"Failed to get culture profile: {e}")
            return error_response(f"Failed to get culture profile: {e}", 500)

        return json_response(
            {
                "workspace_id": culture.workspace_id,
                "patterns": {
                    k: v.to_dict() if hasattr(v, "to_dict") else v
                    for k, v in culture.patterns.items()
                },
                "generated_at": culture.generated_at.isoformat() if culture.generated_at else None,
                "total_observations": culture.total_observations,
            }
        )

    @require_permission("culture:write")
    @handle_errors("add culture document")
    def _handle_add_culture_document(self: CultureHandlerProtocol, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/culture/documents - Add culture document."""

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                return error_response("Request body required", 400)
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        content = data.get("content", "")
        if not content:
            return error_response("Content is required", 400)

        workspace_id = data.get("workspace_id", "default")
        document_type = data.get("document_type", "policy")
        metadata = data.get("metadata", {})

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            from aragora.knowledge.mound import KnowledgeNode, ProvenanceChain, ProvenanceType
            from aragora.memory.tier_manager import MemoryTier

            provenance = ProvenanceChain(
                source_type=ProvenanceType.USER,
                source_id="culture_document",
            )

            node = KnowledgeNode(  # type: ignore[call-arg]
                node_type="culture",  # type: ignore[arg-type]
                content=content,
                confidence=1.0,
                provenance=provenance,
                tier=MemoryTier.GLACIAL,
                workspace_id=workspace_id,
                topics=["culture", document_type],
                metadata={"document_type": document_type, **metadata},
            )

            node_id = _run_async(mound.add_node(node))  # type: ignore[misc]
        except Exception as e:
            logger.error(f"Failed to add culture document: {e}")
            return error_response(f"Failed to add culture document: {e}", 500)

        return json_response(
            {
                "node_id": node_id,
                "document_type": document_type,
                "workspace_id": workspace_id,
                "message": "Culture document added successfully",
            },
            status=201,
        )

    @require_permission("culture:write")
    @handle_errors("promote to culture")
    def _handle_promote_to_culture(self: CultureHandlerProtocol, handler: Any) -> HandlerResult:
        """Handle POST /api/knowledge/mound/culture/promote - Promote knowledge to culture."""

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                return error_response("Request body required", 400)
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        node_id = data.get("node_id")
        if not node_id:
            return error_response("node_id is required", 400)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            from aragora.memory.tier_manager import MemoryTier

            updated = _run_async(
                mound.update(  # type: ignore[misc]
                    node_id,
                    {
                        "node_type": "culture",
                        "tier": MemoryTier.GLACIAL.value,
                        "promoted_to_culture": True,
                    },
                )
            )

            if not updated:
                return error_response(f"Node not found: {node_id}", 404)

        except Exception as e:
            logger.error(f"Failed to promote to culture: {e}")
            return error_response(f"Failed to promote to culture: {e}", 500)

        return json_response(
            {
                "node_id": node_id,
                "promoted": True,
                "message": "Knowledge promoted to culture successfully",
            }
        )
