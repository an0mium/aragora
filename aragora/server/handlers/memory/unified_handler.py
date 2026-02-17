"""Unified Memory Gateway HTTP handler.

Exposes the MemoryGateway via a REST endpoint for cross-system
memory search with deduplication and ranking.
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.rbac.decorators import require_permission
from aragora.server.handlers.secure import SecureHandler

logger = logging.getLogger(__name__)

# Permissions for unified memory endpoints
MEMORY_READ_PERMISSION = "memory:read"


class UnifiedMemoryHandler(SecureHandler):
    """HTTP handler for unified memory operations.

    Wraps MemoryGateway to expose cross-system search via REST API.
    """

    def __init__(self, gateway: Any = None, ctx: dict[str, Any] | None = None) -> None:
        self._gateway = gateway
        self.ctx = ctx or {}

    @require_permission(MEMORY_READ_PERMISSION)
    async def handle_search(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Handle POST /api/v1/memory/unified/search.

        Args:
            request_data: JSON body with query parameters:
                - query (str, required): Search query
                - limit (int, optional): Max results (default 10)
                - min_confidence (float, optional): Min confidence filter
                - sources (list[str], optional): Source filter
                - dedup (bool, optional): Enable dedup (default True)

        Returns:
            JSON response with results, metadata, and errors
        """
        if not self._gateway:
            return {
                "error": "Unified memory gateway not configured",
                "results": [],
            }

        query_text = request_data.get("query", "")
        if not query_text:
            return {
                "error": "Missing required field: query",
                "results": [],
            }

        try:
            from aragora.memory.gateway import UnifiedMemoryQuery

            q = UnifiedMemoryQuery(
                query=query_text,
                limit=request_data.get("limit", 10),
                min_confidence=request_data.get("min_confidence", 0.0),
                sources=request_data.get("sources"),
                dedup=request_data.get("dedup", True),
            )

            response = await self._gateway.query(q)

            return {
                "results": [
                    {
                        "id": r.id,
                        "content": r.content,
                        "source_system": r.source_system,
                        "confidence": r.confidence,
                        "surprise_score": r.surprise_score,
                        "metadata": r.metadata,
                    }
                    for r in response.results
                ],
                "total_found": response.total_found,
                "sources_queried": response.sources_queried,
                "duplicates_removed": response.duplicates_removed,
                "query_time_ms": round(response.query_time_ms, 2),
                "errors": response.errors,
            }
        except (ImportError, RuntimeError, ValueError, TypeError) as e:
            logger.warning("Unified memory search failed: %s", e)
            return {
                "error": str(e),
                "results": [],
            }

    @require_permission(MEMORY_READ_PERMISSION)
    async def handle_stats(self) -> dict[str, Any]:
        """Handle GET /api/v1/memory/unified/stats.

        Returns:
            Gateway statistics
        """
        if not self._gateway:
            return {"error": "Unified memory gateway not configured"}

        return self._gateway.get_stats()
