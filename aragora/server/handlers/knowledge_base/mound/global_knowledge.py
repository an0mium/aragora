"""
Global Knowledge Operations Mixin for Knowledge Mound Handler.

Provides global/system knowledge operations:
- Store verified facts (admin)
- Query global knowledge
- Promote to global
- List system facts
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Protocol

from aragora.server.http_utils import run_async as _run_async
from aragora.server.metrics import track_global_fact, track_global_query

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


class GlobalKnowledgeHandlerProtocol(Protocol):
    """Protocol for handlers that use GlobalKnowledgeOperationsMixin."""

    def _get_mound(self) -> "KnowledgeMound | None": ...
    def require_auth_or_error(self, handler: Any) -> tuple[Any, HandlerResult | None]: ...
    def require_admin_or_error(self, handler: Any) -> tuple[Any, HandlerResult | None]: ...


class GlobalKnowledgeOperationsMixin:
    """Mixin providing global knowledge operations for KnowledgeMoundHandler."""

    @handle_errors("store verified fact")
    def _handle_store_verified_fact(
        self: GlobalKnowledgeHandlerProtocol, handler: Any
    ) -> HandlerResult:
        """Handle POST /api/knowledge/mound/global - Store a verified fact (admin only)."""
        # Require admin for storing global facts
        user, err = self.require_admin_or_error(handler)
        if err:
            # Fall back to regular auth check for users with global_write permission
            user, err = self.require_auth_or_error(handler)
            if err:
                return err
            # Check for global_write permission
            permissions = getattr(user, "permissions", [])
            if "global_write" not in permissions and "admin" not in permissions:
                return error_response("Admin or global_write permission required", 403)

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            if content_length > 0:
                body = handler.rfile.read(content_length)
                data = json.loads(body.decode("utf-8"))
            else:
                return error_response("Request body required", 400)
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        content = data.get("content")
        source = data.get("source")

        if not content:
            return error_response("content is required", 400)
        if not source:
            return error_response("source is required", 400)

        confidence = data.get("confidence", 0.9)
        evidence_ids = data.get("evidence_ids", [])
        topics = data.get("topics", [])

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        user_id = getattr(user, "id", None) or getattr(user, "user_id", "unknown")

        try:
            node_id = _run_async(
                mound.store_verified_fact(
                    content=content,
                    source=source,
                    confidence=confidence,
                    evidence_ids=evidence_ids,
                    verified_by=user_id,
                    topics=topics,
                )
            )
            track_global_fact(action="store")
        except Exception as e:
            logger.error(f"Failed to store verified fact: {e}")
            return error_response(f"Failed to store verified fact: {e}", 500)

        return json_response(
            {
                "success": True,
                "node_id": node_id,
                "content": content,
                "source": source,
                "verified_by": user_id,
            },
            status=201,
        )

    @handle_errors("query global knowledge")
    def _handle_query_global(
        self: GlobalKnowledgeHandlerProtocol, query_params: dict
    ) -> HandlerResult:
        """Handle GET /api/knowledge/mound/global - Query global knowledge."""
        query = get_bounded_string_param(query_params, "query", "", max_length=1000)
        limit = get_clamped_int_param(query_params, "limit", 20, min_val=1, max_val=100)
        topics_str = get_bounded_string_param(query_params, "topics", None, max_length=500)
        topics = topics_str.split(",") if topics_str else None

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            if query:
                items = _run_async(
                    mound.query_global_knowledge(
                        query=query,
                        limit=limit,
                        topics=topics,
                    )
                )
            else:
                # If no query, get all system facts
                items = _run_async(mound.get_system_facts(limit=limit, topics=topics))
            track_global_query(has_results=len(items) > 0)
        except Exception as e:
            logger.error(f"Failed to query global knowledge: {e}")
            return error_response(f"Failed to query global knowledge: {e}", 500)

        return json_response(
            {
                "items": [
                    (
                        item.to_dict()
                        if hasattr(item, "to_dict")
                        else {
                            "id": getattr(item, "id", "unknown"),
                            "content": getattr(item, "content", ""),
                            "importance": getattr(item, "importance", 0.5),
                        }
                    )
                    for item in items
                ],
                "count": len(items),
                "query": query,
            }
        )

    @handle_errors("promote to global")
    def _handle_promote_to_global(
        self: GlobalKnowledgeHandlerProtocol, handler: Any
    ) -> HandlerResult:
        """Handle POST /api/knowledge/mound/global/promote - Promote item to global."""
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

        item_id = data.get("item_id")
        workspace_id = data.get("workspace_id")
        reason = data.get("reason")

        if not item_id:
            return error_response("item_id is required", 400)
        if not workspace_id:
            return error_response("workspace_id is required", 400)
        if not reason:
            return error_response("reason is required", 400)

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        user_id = getattr(user, "id", None) or getattr(user, "user_id", "unknown")

        try:
            global_id = _run_async(
                mound.promote_to_global(
                    item_id=item_id,
                    workspace_id=workspace_id,
                    promoted_by=user_id,
                    reason=reason,
                )
            )
            track_global_fact(action="promote")
        except ValueError as e:
            return error_response(str(e), 404)
        except Exception as e:
            logger.error(f"Failed to promote to global: {e}")
            return error_response(f"Failed to promote to global: {e}", 500)

        return json_response(
            {
                "success": True,
                "global_id": global_id,
                "original_id": item_id,
                "promoted_by": user_id,
                "reason": reason,
            },
            status=201,
        )

    @handle_errors("get system facts")
    def _handle_get_system_facts(
        self: GlobalKnowledgeHandlerProtocol, query_params: dict
    ) -> HandlerResult:
        """Handle GET /api/knowledge/mound/global/facts - Get all system facts."""
        limit = get_clamped_int_param(query_params, "limit", 100, min_val=1, max_val=500)
        offset = get_clamped_int_param(query_params, "offset", 0, min_val=0, max_val=10000)
        topics_str = get_bounded_string_param(query_params, "topics", None, max_length=500)
        topics = topics_str.split(",") if topics_str else None

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            facts = _run_async(mound.get_system_facts(limit=limit + offset, topics=topics))
        except Exception as e:
            logger.error(f"Failed to get system facts: {e}")
            return error_response(f"Failed to get system facts: {e}", 500)

        paginated_facts = facts[offset : offset + limit]

        return json_response(
            {
                "facts": [
                    (
                        fact.to_dict()
                        if hasattr(fact, "to_dict")
                        else {
                            "id": getattr(fact, "id", "unknown"),
                            "content": getattr(fact, "content", ""),
                        }
                    )
                    for fact in paginated_facts
                ],
                "count": len(paginated_facts),
                "total": len(facts),
                "limit": limit,
                "offset": offset,
            }
        )

    @handle_errors("get system workspace id")
    def _handle_get_system_workspace_id(
        self: GlobalKnowledgeHandlerProtocol,
    ) -> HandlerResult:
        """Handle GET /api/knowledge/mound/global/workspace-id - Get system workspace ID."""
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge Mound not available", 503)

        try:
            workspace_id = mound.get_system_workspace_id()
        except Exception as e:
            logger.error(f"Failed to get system workspace ID: {e}")
            return error_response(f"Failed to get system workspace ID: {e}", 500)

        return json_response(
            {
                "system_workspace_id": workspace_id,
            }
        )
