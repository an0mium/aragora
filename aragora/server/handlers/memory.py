"""
Memory-related endpoint handlers.

Endpoints:
- GET /api/memory/continuum/retrieve - Retrieve memories from continuum
- POST /api/memory/continuum/consolidate - Trigger memory consolidation
"""

from typing import Optional

from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    get_int_param,
    get_bool_param,
)

# Optional import for memory functionality
try:
    from aragora.memory.continuum import ContinuumMemory, MemoryTier
    CONTINUUM_AVAILABLE = True
except ImportError:
    CONTINUUM_AVAILABLE = False
    ContinuumMemory = None
    MemoryTier = None


class MemoryHandler(BaseHandler):
    """Handler for memory-related endpoints."""

    ROUTES = [
        "/api/memory/continuum/retrieve",
        "/api/memory/continuum/consolidate",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route memory requests to appropriate handler methods."""
        if path == "/api/memory/continuum/retrieve":
            return self._get_continuum_memories(query_params)

        if path == "/api/memory/continuum/consolidate":
            return self._trigger_consolidation()

        return None

    def _get_continuum_memories(self, params: dict) -> HandlerResult:
        """Retrieve memories from the continuum memory system."""
        if not CONTINUUM_AVAILABLE:
            return error_response("Continuum memory system not available", 503)

        continuum = self.ctx.get("continuum_memory")
        if not continuum:
            return error_response("Continuum memory not initialized", 503)

        try:
            query = params.get("query", [""])[0] if isinstance(params.get("query"), list) else params.get("query", "")
            tiers_param = params.get("tiers", ["fast,medium,slow,glacial"])[0] if isinstance(params.get("tiers"), list) else params.get("tiers", "fast,medium,slow,glacial")
            limit = get_int_param(params, "limit", 10)
            min_importance = float(params.get("min_importance", ["0.0"])[0] if isinstance(params.get("min_importance"), list) else params.get("min_importance", "0.0"))

            # Parse tiers
            tier_names = [t.strip() for t in tiers_param.split(",")]
            tiers = []
            for name in tier_names:
                try:
                    tiers.append(MemoryTier[name.upper()])
                except KeyError:
                    continue

            if not tiers:
                tiers = list(MemoryTier)

            # Retrieve memories
            memories = continuum.retrieve(
                query=query,
                tiers=tiers,
                limit=limit,
                min_importance=min_importance,
            )

            return json_response({
                "memories": [
                    {
                        "id": m.id,
                        "tier": m.tier.name.lower(),
                        "content": m.content[:500] + "..." if len(m.content) > 500 else m.content,
                        "importance": m.importance,
                        "surprise_score": getattr(m, "surprise_score", 0.0),
                        "consolidation_score": getattr(m, "consolidation_score", 0.0),
                        "update_count": getattr(m, "update_count", 0),
                        "created_at": str(m.created_at) if hasattr(m, "created_at") else None,
                        "updated_at": str(m.updated_at) if hasattr(m, "updated_at") else None,
                    }
                    for m in memories
                ],
                "count": len(memories),
                "query": query,
                "tiers": [t.name.lower() for t in tiers],
            })
        except Exception as e:
            return error_response(f"Failed to retrieve memories: {e}", 500)

    def _trigger_consolidation(self) -> HandlerResult:
        """Trigger memory consolidation process."""
        if not CONTINUUM_AVAILABLE:
            return error_response("Continuum memory system not available", 503)

        continuum = self.ctx.get("continuum_memory")
        if not continuum:
            return error_response("Continuum memory not initialized", 503)

        try:
            import time
            start = time.time()

            # Run consolidation
            result = continuum.consolidate()

            duration = time.time() - start

            return json_response({
                "success": True,
                "entries_processed": result.get("processed", 0),
                "entries_promoted": result.get("promoted", 0),
                "entries_consolidated": result.get("consolidated", 0),
                "duration_seconds": round(duration, 2),
            })
        except Exception as e:
            return error_response(f"Failed to consolidate memories: {e}", 500)
