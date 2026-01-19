"""
Search Operations Mixin for Knowledge Handler.

Provides chunk search and statistics operations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Protocol

from aragora.server.http_utils import run_async as _run_async

from ..base import (
    HandlerResult,
    error_response,
    get_bounded_string_param,
    get_clamped_int_param,
    handle_errors,
    json_response,
    ttl_cache,
)

if TYPE_CHECKING:
    from aragora.knowledge import DatasetQueryEngine, FactStore, SimpleQueryEngine

logger = logging.getLogger(__name__)

# Cache TTLs
CACHE_TTL_STATS = 300  # 5 minutes for statistics


class SearchHandlerProtocol(Protocol):
    """Protocol for handlers that use SearchOperationsMixin."""

    def _get_fact_store(self) -> "FactStore": ...
    def _get_query_engine(self) -> "DatasetQueryEngine | SimpleQueryEngine": ...


class SearchOperationsMixin:
    """Mixin providing search and stats operations for KnowledgeHandler."""

    @handle_errors("search chunks")
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

        try:
            results = _run_async(engine.search(query, workspace_id, limit))
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return error_response(f"Search failed: {e}", 500)

        return json_response(
            {
                "query": query,
                "workspace_id": workspace_id,
                "results": [r.to_dict() for r in results],
                "count": len(results),
            }
        )

    @ttl_cache(ttl_seconds=CACHE_TTL_STATS, key_prefix="knowledge_stats", skip_first=True)
    @handle_errors("get stats")
    def _handle_stats(self: SearchHandlerProtocol, workspace_id: Optional[str]) -> HandlerResult:
        """Handle GET /api/knowledge/stats - Get statistics."""
        store = self._get_fact_store()
        stats = store.get_statistics(workspace_id)

        return json_response(
            {
                "workspace_id": workspace_id,
                **stats,
            }
        )
