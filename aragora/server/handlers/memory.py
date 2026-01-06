"""
Memory-related endpoint handlers.

Endpoints:
- GET /api/memory/continuum/retrieve - Retrieve memories from continuum
- POST /api/memory/continuum/consolidate - Trigger memory consolidation
- POST /api/memory/continuum/cleanup - Cleanup expired memories
- GET /api/memory/tier-stats - Get tier statistics
"""

from typing import Optional

from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    get_int_param,
    get_float_param,
    get_string_param,
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
        "/api/memory/continuum/cleanup",
        "/api/memory/tier-stats",
        "/api/memory/archive-stats",
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

        if path == "/api/memory/continuum/cleanup":
            return self._trigger_cleanup(query_params)

        if path == "/api/memory/tier-stats":
            return self._get_tier_stats()

        if path == "/api/memory/archive-stats":
            return self._get_archive_stats()

        return None

    def _get_continuum_memories(self, params: dict) -> HandlerResult:
        """Retrieve memories from the continuum memory system."""
        if not CONTINUUM_AVAILABLE:
            return error_response("Continuum memory system not available", 503)

        continuum = self.ctx.get("continuum_memory")
        if not continuum:
            return error_response("Continuum memory not initialized", 503)

        try:
            query = get_string_param(params, "query", "")
            tiers_param = get_string_param(params, "tiers", "fast,medium,slow,glacial")
            limit = get_int_param(params, "limit", 10)
            min_importance = get_float_param(params, "min_importance", 0.0)

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

    def _trigger_cleanup(self, params: dict) -> HandlerResult:
        """Trigger memory cleanup with optional parameters."""
        if not CONTINUUM_AVAILABLE:
            return error_response("Continuum memory system not available", 503)

        continuum = self.ctx.get("continuum_memory")
        if not continuum:
            return error_response("Continuum memory not initialized", 503)

        try:
            import time
            start = time.time()

            # Parse parameters
            tier_param = get_string_param(params, "tier", "")
            archive_param = get_string_param(params, "archive", "true")
            max_age = get_float_param(params, "max_age_hours", 0)

            # Convert tier parameter
            tier = None
            if tier_param:
                try:
                    tier = MemoryTier[tier_param.upper()]
                except KeyError:
                    return error_response(f"Invalid tier: {tier_param}", 400)

            archive = archive_param.lower() == "true"

            # Run cleanup
            expired_result = continuum.cleanup_expired_memories(
                tier=tier,
                archive=archive,
                max_age_hours=max_age if max_age > 0 else None,
            )

            # Enforce tier limits
            limits_result = continuum.enforce_tier_limits(
                tier=tier,
                archive=archive,
            )

            duration = time.time() - start

            return json_response({
                "success": True,
                "expired": expired_result,
                "tier_limits": limits_result,
                "duration_seconds": round(duration, 2),
            })
        except Exception as e:
            return error_response(f"Cleanup failed: {e}", 500)

    def _get_tier_stats(self) -> HandlerResult:
        """Get statistics for each memory tier."""
        if not CONTINUUM_AVAILABLE:
            return error_response("Continuum memory system not available", 503)

        continuum = self.ctx.get("continuum_memory")
        if not continuum:
            return error_response("Continuum memory not initialized", 503)

        try:
            stats = continuum.get_stats()
            return json_response({
                "tiers": stats.get("by_tier", {}),
                "total_memories": stats.get("total_memories", 0),
                "transitions": stats.get("transitions", []),
            })
        except Exception as e:
            return error_response(f"Failed to get tier stats: {e}", 500)

    def _get_archive_stats(self) -> HandlerResult:
        """Get statistics for archived memories."""
        if not CONTINUUM_AVAILABLE:
            return error_response("Continuum memory system not available", 503)

        continuum = self.ctx.get("continuum_memory")
        if not continuum:
            return error_response("Continuum memory not initialized", 503)

        try:
            stats = continuum.get_archive_stats()
            return json_response(stats)
        except Exception as e:
            return error_response(f"Failed to get archive stats: {e}", 500)
