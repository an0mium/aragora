"""
Staleness Operations Mixin for Knowledge Mound Handler.

Provides staleness detection and revalidation:
- Get stale knowledge items
- Revalidate specific node
- Schedule batch revalidation
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
    get_bounded_float_param,
    get_bounded_string_param,
    get_clamped_int_param,
    handle_errors,
    json_response,
)

if TYPE_CHECKING:
    from aragora.knowledge.mound import KnowledgeMound

logger = logging.getLogger(__name__)


class StalenessHandlerProtocol(Protocol):
    """Protocol for handlers that use StalenessOperationsMixin."""

    def _get_mound(self) -> "KnowledgeMound | None": ...


class StalenessOperationsMixin:
    """Mixin providing staleness operations for KnowledgeMoundHandler."""

    @require_permission("knowledge:read")
    @handle_errors("get stale knowledge")
    def _handle_get_stale(self: StalenessHandlerProtocol, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/mound/stale - Get stale knowledge items."""

        workspace_id = get_bounded_string_param(
            query_params, "workspace_id", "default", max_length=100
        )
        threshold = get_bounded_float_param(
            query_params, "threshold", 0.5, min_val=0.0, max_val=1.0
        )
        limit = get_clamped_int_param(query_params, "limit", 50, min_val=1, max_val=200)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            stale_items = _run_async(
                mound.get_stale_knowledge(
                    threshold=threshold,
                    limit=limit,
                    workspace_id=workspace_id,
                )
            )
        except Exception as e:
            logger.error(f"Failed to get stale knowledge: {e}")
            return error_response(f"Failed to get stale knowledge: {e}", 500)

        return json_response(
            {
                "stale_items": [
                    {
                        "node_id": item.node_id,
                        "staleness_score": item.staleness_score,
                        "reasons": [r.value if hasattr(r, "value") else r for r in item.reasons],
                        "last_validated_at": (
                            item.last_validated_at.isoformat() if item.last_validated_at else None  # type: ignore[attr-defined]
                        ),
                        "recommended_action": item.recommended_action,  # type: ignore[attr-defined]
                    }
                    for item in stale_items
                ],
                "total": len(stale_items),
                "threshold": threshold,
                "workspace_id": workspace_id,
            }
        )

    @require_permission("knowledge:read")
    @handle_errors("revalidate node")
    def _handle_revalidate_node(
        self: StalenessHandlerProtocol, node_id: str, handler: Any
    ) -> HandlerResult:
        """Handle POST /api/knowledge/mound/revalidate/:id - Trigger revalidation."""

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                data = {}
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        validator = data.get("validator", "api")
        new_confidence = data.get("confidence")

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            _run_async(mound.mark_validated(node_id, validator, new_confidence))
        except Exception as e:
            logger.error(f"Failed to revalidate node: {e}")
            return error_response(f"Failed to revalidate node: {e}", 500)

        return json_response(
            {
                "node_id": node_id,
                "validated": True,
                "validator": validator,
                "new_confidence": new_confidence,
                "message": "Node revalidated successfully",
            }
        )

    @require_permission("knowledge:read")
    @handle_errors("schedule revalidation")
    def _handle_schedule_revalidation(
        self: StalenessHandlerProtocol, handler: Any
    ) -> HandlerResult:
        """Handle POST /api/knowledge/mound/schedule-revalidation - Schedule batch."""

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                return error_response("Request body required", 400)
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        node_ids = data.get("node_ids", [])
        if not node_ids:
            return error_response("node_ids is required", 400)

        priority = data.get("priority", "low")
        if priority not in ("low", "medium", "high"):
            return error_response("priority must be 'low', 'medium', or 'high'", 400)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            scheduled = _run_async(mound.schedule_revalidation(node_ids, priority))
        except Exception as e:
            logger.error(f"Failed to schedule revalidation: {e}")
            return error_response(f"Failed to schedule revalidation: {e}", 500)

        return json_response(
            {
                "scheduled": scheduled,
                "priority": priority,
                "count": len(scheduled),
                "message": f"Scheduled {len(scheduled)} nodes for revalidation",
            },
            status=202,
        )
