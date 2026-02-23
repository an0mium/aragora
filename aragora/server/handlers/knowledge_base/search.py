"""
Search Operations Mixin for Knowledge Handler.

Provides chunk search and statistics operations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from aragora.server.http_utils import run_async as _run_async
from aragora.rbac.decorators import require_permission

from ..base import (
    HandlerResult,
    error_response,
    get_bounded_string_param,
    get_clamped_int_param,
    handle_errors,
    json_response,
    ttl_cache,
)
from ..openapi_decorator import api_endpoint

if TYPE_CHECKING:
    from aragora.knowledge import DatasetQueryEngine, FactStore, SimpleQueryEngine

logger = logging.getLogger(__name__)

# Cache TTLs
CACHE_TTL_STATS = 300  # 5 minutes for statistics


class SearchHandlerProtocol(Protocol):
    """Protocol for handlers that use SearchOperationsMixin."""

    def _get_fact_store(self) -> FactStore: ...
    def _get_query_engine(self) -> DatasetQueryEngine | SimpleQueryEngine: ...


class SearchOperationsMixin:
    """Mixin providing search and stats operations for KnowledgeHandler."""

    @api_endpoint(
        method="GET",
        path="/api/v1/knowledge/search",
        summary="Search knowledge base chunks via embeddings",
        tags=["Knowledge Base"],
        parameters=[
            {
                "name": "q",
                "in": "query",
                "required": True,
                "schema": {"type": "string"},
                "description": "Search query string (max 500 chars)",
            },
            {
                "name": "workspace_id",
                "in": "query",
                "schema": {"type": "string", "default": "default"},
                "description": "Workspace to search within",
            },
            {
                "name": "limit",
                "in": "query",
                "schema": {"type": "integer", "default": 10},
                "description": "Maximum number of results (1-50)",
            },
        ],
        responses={
            "200": {
                "description": "Search results with matching chunks",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "workspace_id": {"type": "string"},
                                "results": {"type": "array", "items": {"type": "object"}},
                                "count": {"type": "integer"},
                            },
                        }
                    }
                },
            },
            "400": {"description": "Missing query parameter"},
            "401": {"description": "Unauthorized"},
            "403": {"description": "Forbidden"},
            "500": {"description": "Search failed"},
        },
    )
    @handle_errors("search chunks")
    @require_permission("knowledge:read")
    def _handle_search(self: SearchHandlerProtocol, query_params: dict) -> HandlerResult:
        """Handle GET /api/knowledge/search - Search chunks."""

        query = get_bounded_string_param(query_params, "q", "", max_length=500)
        if not query:
            return error_response("Query parameter 'q' is required", 400)

        workspace_id = get_bounded_string_param(
            query_params, "workspace_id", "default", max_length=100
        )
        limit = get_clamped_int_param(query_params, "limit", 10, min_val=1, max_val=50)

        engine = self._get_query_engine()
        if not hasattr(engine, "search"):
            raise TypeError("Query engine does not support search")

        try:
            results = _run_async(engine.search(query, workspace_id, limit))
        except (KeyError, ValueError, OSError, TypeError, RuntimeError) as e:
            logger.error("Search failed: %s", e)
            return error_response("Search operation failed", 500)

        return json_response(
            {
                "query": query,
                "workspace_id": workspace_id,
                "results": [r.to_dict() for r in results],
                "count": len(results),
            }
        )

    @api_endpoint(
        method="GET",
        path="/api/v1/knowledge/stats",
        summary="Get knowledge base statistics",
        tags=["Knowledge Base"],
        operation_id="get_knowledge_stats",
        parameters=[
            {
                "name": "workspace_id",
                "in": "query",
                "schema": {"type": "string"},
                "description": "Filter statistics by workspace ID",
            },
        ],
        responses={
            "200": {
                "description": "Knowledge base statistics",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "workspace_id": {"type": "string"},
                            },
                            "additionalProperties": True,
                        }
                    }
                },
            },
            "401": {"description": "Unauthorized"},
            "403": {"description": "Forbidden"},
        },
    )
    @ttl_cache(ttl_seconds=CACHE_TTL_STATS, key_prefix="knowledge_stats", skip_first=True)
    @handle_errors("get stats")
    def _handle_stats(self: SearchHandlerProtocol, workspace_id: str | None) -> HandlerResult:
        """Handle GET /api/knowledge/stats - Get statistics."""
        store = self._get_fact_store()
        stats = store.get_statistics(workspace_id)

        return json_response(
            {
                "workspace_id": workspace_id,
                **stats,
            }
        )
