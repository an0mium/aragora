"""
Memory-related endpoint handlers.

Endpoints:
- GET /api/memory/continuum/retrieve - Retrieve memories from continuum
- POST /api/memory/continuum/consolidate - Trigger memory consolidation
- POST /api/memory/continuum/cleanup - Cleanup expired memories
- GET /api/memory/tier-stats - Get tier statistics
- GET /api/memory/archive-stats - Get archive statistics
- GET /api/memory/pressure - Get memory pressure and utilization
- DELETE /api/memory/continuum/{id} - Delete a memory by ID
- GET /api/memory/tiers - List all memory tiers with detailed stats
- GET /api/memory/search - Search memories across tiers
- GET /api/memory/critiques - Browse critique store entries
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, Type

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    get_bounded_float_param,
    get_bounded_string_param,
    get_clamped_int_param,
    handle_errors,
    json_response,
    safe_error_message,
)
from ..utils.rate_limit import RateLimiter, get_client_ip

# Rate limiters for memory endpoints
_retrieve_limiter = RateLimiter(requests_per_minute=60)  # Read operations
_stats_limiter = RateLimiter(requests_per_minute=30)  # Stats operations
_mutation_limiter = RateLimiter(requests_per_minute=10)  # State-changing operations

logger = logging.getLogger(__name__)

# Optional import for memory functionality
try:
    from aragora.memory.continuum import ContinuumMemory, MemoryTier

    CONTINUUM_AVAILABLE = True
except ImportError:
    CONTINUUM_AVAILABLE = False
    ContinuumMemory: Optional[Type[Any]] = None
    MemoryTier: Optional[Type[Any]] = None

# Optional import for critique store
try:
    from aragora.memory.store import CritiqueStore

    CRITIQUE_STORE_AVAILABLE = True
except ImportError:
    CRITIQUE_STORE_AVAILABLE = False
    CritiqueStore: Optional[Type[Any]] = None


class MemoryHandler(BaseHandler):
    """Handler for memory-related endpoints."""

    ROUTES = [
        "/api/v1/memory/continuum/retrieve",
        "/api/v1/memory/continuum/consolidate",
        "/api/v1/memory/continuum/cleanup",
        "/api/v1/memory/tier-stats",
        "/api/v1/memory/archive-stats",
        "/api/v1/memory/pressure",
        "/api/v1/memory/tiers",
        "/api/v1/memory/search",
        "/api/v1/memory/critiques",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path in self.ROUTES:
            return True
        # Handle /api/memory/continuum/{id} pattern for DELETE
        if path.startswith("/api/v1/memory/continuum/") and path.count("/") == 5:
            # Exclude known routes like /api/memory/continuum/retrieve
            segment = path.split("/")[-1]
            if segment not in ("retrieve", "consolidate", "cleanup"):
                return True
        return False

    def _get_user_store(self):
        """Get user store from context."""
        return self.ctx.get("user_store")

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route memory requests to appropriate handler methods."""
        client_ip = get_client_ip(handler)

        if path == "/api/v1/memory/continuum/retrieve":
            # Rate limit: 60/min for retrieve operations
            if not _retrieve_limiter.is_allowed(client_ip):
                return error_response("Rate limit exceeded. Please try again later.", 429)
            return self._get_continuum_memories(query_params)

        # POST-only endpoints - return 405 Method Not Allowed for GET
        if path in ("/api/v1/memory/continuum/consolidate", "/api/v1/memory/continuum/cleanup"):
            return error_response(
                "Use POST method for this endpoint",
                405,
                headers={"Allow": "POST"},
            )

        if path == "/api/v1/memory/tier-stats":
            # Rate limit: 30/min for stats operations
            if not _stats_limiter.is_allowed(client_ip):
                return error_response("Rate limit exceeded. Please try again later.", 429)
            return self._get_tier_stats()

        if path == "/api/v1/memory/archive-stats":
            # Rate limit: 30/min for stats operations
            if not _stats_limiter.is_allowed(client_ip):
                return error_response("Rate limit exceeded. Please try again later.", 429)
            return self._get_archive_stats()

        if path == "/api/v1/memory/pressure":
            # Rate limit: 30/min for stats operations
            if not _stats_limiter.is_allowed(client_ip):
                return error_response("Rate limit exceeded. Please try again later.", 429)
            return self._get_memory_pressure()

        if path == "/api/v1/memory/tiers":
            # Rate limit: 30/min for stats operations
            if not _stats_limiter.is_allowed(client_ip):
                return error_response("Rate limit exceeded. Please try again later.", 429)
            return self._get_all_tiers()

        if path == "/api/v1/memory/search":
            # Rate limit: 60/min for retrieve operations
            if not _retrieve_limiter.is_allowed(client_ip):
                return error_response("Rate limit exceeded. Please try again later.", 429)
            return self._search_memories(query_params)

        if path == "/api/v1/memory/critiques":
            # Rate limit: 30/min for stats operations
            if not _stats_limiter.is_allowed(client_ip):
                return error_response("Rate limit exceeded. Please try again later.", 429)
            return self._get_critiques(query_params)

        return None

    def handle_post(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route POST memory requests to state-mutating methods with auth."""
        from aragora.billing.jwt_auth import extract_user_from_request

        client_ip = get_client_ip(handler)

        if path == "/api/v1/memory/continuum/consolidate":
            # Rate limit: 10/min for mutation operations
            if not _mutation_limiter.is_allowed(client_ip):
                return error_response("Rate limit exceeded. Please try again later.", 429)
            # Require authentication for state mutation
            user_store = self._get_user_store()
            auth_ctx = extract_user_from_request(handler, user_store)
            if not auth_ctx.is_authenticated:
                return error_response("Authentication required", 401)
            return self._trigger_consolidation()

        if path == "/api/v1/memory/continuum/cleanup":
            # Rate limit: 10/min for mutation operations
            if not _mutation_limiter.is_allowed(client_ip):
                return error_response("Rate limit exceeded. Please try again later.", 429)
            # Require authentication for state mutation
            user_store = self._get_user_store()
            auth_ctx = extract_user_from_request(handler, user_store)
            if not auth_ctx.is_authenticated:
                return error_response("Authentication required", 401)
            return self._trigger_cleanup(query_params)

        return None

    @handle_errors("continuum memories retrieval")
    def _get_continuum_memories(self, params: dict) -> HandlerResult:
        """Retrieve memories from the continuum memory system."""
        if not CONTINUUM_AVAILABLE:
            return error_response("Continuum memory system not available", 503)

        continuum = self.ctx.get("continuum_memory")
        if not continuum:
            return error_response("Continuum memory not initialized", 503)

        query = get_bounded_string_param(params, "query", "", max_length=500)
        tiers_param = get_bounded_string_param(
            params, "tiers", "fast,medium,slow,glacial", max_length=100
        )
        limit = get_clamped_int_param(params, "limit", 10, min_val=1, max_val=100)
        min_importance = get_bounded_float_param(
            params, "min_importance", 0.0, min_val=0.0, max_val=1.0
        )

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

        return json_response(
            {
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
            }
        )

    @handle_errors("memory consolidation")
    def _trigger_consolidation(self) -> HandlerResult:
        """Trigger memory consolidation process."""
        if not CONTINUUM_AVAILABLE:
            return error_response("Continuum memory system not available", 503)

        continuum = self.ctx.get("continuum_memory")
        if not continuum:
            return error_response("Continuum memory not initialized", 503)

        start = time.time()

        # Run consolidation
        result = continuum.consolidate()

        duration = time.time() - start

        return json_response(
            {
                "success": True,
                "entries_processed": result.get("processed", 0),
                "entries_promoted": result.get("promoted", 0),
                "entries_consolidated": result.get("consolidated", 0),
                "duration_seconds": round(duration, 2),
            }
        )

    @handle_errors("memory cleanup")
    def _trigger_cleanup(self, params: dict) -> HandlerResult:
        """Trigger memory cleanup with optional parameters."""
        if not CONTINUUM_AVAILABLE:
            return error_response("Continuum memory system not available", 503)

        continuum = self.ctx.get("continuum_memory")
        if not continuum:
            return error_response("Continuum memory not initialized", 503)

        start = time.time()

        # Parse parameters
        tier_param = get_bounded_string_param(params, "tier", "", max_length=50)
        archive_param = get_bounded_string_param(params, "archive", "true", max_length=10)
        max_age = get_bounded_float_param(
            params, "max_age_hours", 0, min_val=0.0, max_val=8760.0
        )  # Max 1 year

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

        return json_response(
            {
                "success": True,
                "expired": expired_result,
                "tier_limits": limits_result,
                "duration_seconds": round(duration, 2),
            }
        )

    @handle_errors("tier stats retrieval")
    def _get_tier_stats(self) -> HandlerResult:
        """Get statistics for each memory tier."""
        if not CONTINUUM_AVAILABLE:
            return error_response("Continuum memory system not available", 503)

        continuum = self.ctx.get("continuum_memory")
        if not continuum:
            return error_response("Continuum memory not initialized", 503)

        stats = continuum.get_stats()
        return json_response(
            {
                "tiers": stats.get("by_tier", {}),
                "total_memories": stats.get("total_memories", 0),
                "transitions": stats.get("transitions", []),
            }
        )

    @handle_errors("archive stats retrieval")
    def _get_archive_stats(self) -> HandlerResult:
        """Get statistics for archived memories."""
        if not CONTINUUM_AVAILABLE:
            return error_response("Continuum memory system not available", 503)

        continuum = self.ctx.get("continuum_memory")
        if not continuum:
            return error_response("Continuum memory not initialized", 503)

        stats = continuum.get_archive_stats()
        return json_response(stats)

    @handle_errors("memory pressure retrieval")
    def _get_memory_pressure(self) -> HandlerResult:
        """Get current memory pressure and per-tier utilization.

        Returns:
            - pressure: Overall memory pressure (0.0-1.0)
            - status: "normal", "elevated", "high", or "critical"
            - tier_utilization: Dict of tier -> {count, limit, utilization}
            - auto_cleanup_triggered: Whether cleanup was auto-triggered

        Auto-triggers cleanup when pressure > 0.9.
        """
        if not CONTINUUM_AVAILABLE:
            return error_response("Continuum memory system not available", 503)

        continuum = self.ctx.get("continuum_memory")
        if not continuum:
            return error_response("Continuum memory not initialized", 503)

        # Get current pressure
        pressure = continuum.get_memory_pressure()

        # Get per-tier stats for utilization breakdown
        stats = continuum.get_stats()
        tier_stats = stats.get("by_tier", {})

        # Build tier utilization breakdown
        tier_utilization = {}
        tier_limits = {
            "FAST": 100,
            "MEDIUM": 500,
            "SLOW": 1000,
            "GLACIAL": 5000,
        }

        for tier_name, tier_data in tier_stats.items():
            count = tier_data.get("count", 0)
            limit = tier_limits.get(tier_name, 1000)
            utilization = count / limit if limit > 0 else 0.0
            tier_utilization[tier_name] = {
                "count": count,
                "limit": limit,
                "utilization": round(utilization, 3),
            }

        # Determine status
        if pressure < 0.5:
            status = "normal"
        elif pressure < 0.8:
            status = "elevated"
        elif pressure < 0.9:
            status = "high"
        else:
            status = "critical"

        # Note: GET endpoints should be idempotent - no auto-cleanup here.
        # Use POST /api/memory/continuum/cleanup to trigger cleanup explicitly.

        response_data = {
            "pressure": round(pressure, 3),
            "status": status,
            "tier_utilization": tier_utilization,
            "total_memories": stats.get("total_memories", 0),
            "cleanup_recommended": pressure > 0.9,  # Hint to caller
        }

        return json_response(response_data)

    def handle_delete(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route DELETE memory requests to appropriate methods with auth."""
        from aragora.billing.jwt_auth import extract_user_from_request

        from ..utils.rate_limit import RateLimiter, get_client_ip

        if path.startswith("/api/v1/memory/continuum/"):
            # Require authentication for state mutation
            user_store = self._get_user_store()
            auth_ctx = extract_user_from_request(handler, user_store)
            if not auth_ctx.is_authenticated:
                return error_response("Authentication required", 401)

            # Rate limit: 10 deletes per minute per IP
            if not hasattr(self, "_delete_limiter"):
                self._delete_limiter = RateLimiter(requests_per_minute=10)
            client_ip = get_client_ip(handler)
            if not self._delete_limiter.is_allowed(client_ip):
                return error_response("Rate limit exceeded. Please try again later.", 429)

            # Extract memory_id from /api/v1/memory/continuum/{id}
            # After strip().split("/") = ["api", "v1", "memory", "continuum", "{id}"]
            # memory_id is at index 4
            memory_id, err = self.extract_path_param(path, 4, "memory_id")
            if err:
                return err
            return self._delete_memory(memory_id)
        return None

    @handle_errors("memory deletion")
    def _delete_memory(self, memory_id: str) -> HandlerResult:
        """Delete a memory by ID."""
        if not CONTINUUM_AVAILABLE:
            return error_response("Continuum memory system not available", 503)

        continuum = self.ctx.get("continuum_memory")
        if not continuum:
            return error_response("Continuum memory not initialized", 503)

        # Check if delete method exists on continuum
        if not hasattr(continuum, "delete"):
            return error_response("Memory deletion not supported", 501)

        try:
            success = continuum.delete(memory_id)
            if success:
                return json_response(
                    {"success": True, "message": f"Memory {memory_id} deleted successfully"}
                )
            else:
                return error_response(f"Memory not found: {memory_id}", 404)
        except Exception as e:
            return error_response(safe_error_message(e, "delete memory"), 500)

    @handle_errors("get all tiers")
    def _get_all_tiers(self) -> HandlerResult:
        """Get comprehensive information about all memory tiers.

        Returns detailed stats for each tier including:
        - Name, description, and TTL
        - Current count and limit
        - Utilization percentage
        - Average importance and surprise scores
        - Recent activity count
        """
        if not CONTINUUM_AVAILABLE:
            return error_response("Continuum memory system not available", 503)

        continuum = self.ctx.get("continuum_memory")
        if not continuum:
            return error_response("Continuum memory not initialized", 503)

        # Get base stats
        stats = continuum.get_stats()
        tier_stats = stats.get("by_tier", {})

        # Tier metadata with explicit types
        tier_info: dict[str, dict[str, str | int]] = {
            "FAST": {
                "name": "Fast",
                "description": "Immediate context, very short-term",
                "ttl_seconds": 60,
                "limit": 100,
            },
            "MEDIUM": {
                "name": "Medium",
                "description": "Session memory, short-term",
                "ttl_seconds": 3600,
                "limit": 500,
            },
            "SLOW": {
                "name": "Slow",
                "description": "Cross-session learning, medium-term",
                "ttl_seconds": 86400,
                "limit": 1000,
            },
            "GLACIAL": {
                "name": "Glacial",
                "description": "Long-term patterns and insights",
                "ttl_seconds": 604800,
                "limit": 5000,
            },
        }

        # Build comprehensive tier data
        tiers = []
        for tier_name, info in tier_info.items():
            tier_data = tier_stats.get(tier_name, {})
            count = tier_data.get("count", 0)
            limit = int(info["limit"])
            ttl_seconds = int(info["ttl_seconds"])
            utilization = count / limit if limit > 0 else 0.0

            tiers.append(
                {
                    "id": tier_name.lower(),
                    "name": info["name"],
                    "description": info["description"],
                    "ttl_seconds": ttl_seconds,
                    "ttl_human": self._format_ttl(ttl_seconds),
                    "count": count,
                    "limit": limit,
                    "utilization": round(utilization, 3),
                    "avg_importance": tier_data.get("avg_importance", 0.0),
                    "avg_surprise": tier_data.get("avg_surprise", 0.0),
                }
            )

        return json_response(
            {
                "tiers": tiers,
                "total_memories": stats.get("total_memories", 0),
                "transitions_24h": len(stats.get("transitions", [])),
            }
        )

    def _format_ttl(self, seconds: int) -> str:
        """Format TTL in human-readable form."""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m"
        elif seconds < 86400:
            return f"{seconds // 3600}h"
        else:
            return f"{seconds // 86400}d"

    @handle_errors("search memories")
    def _search_memories(self, params: dict) -> HandlerResult:
        """Search memories across all tiers with filtering.

        Query params:
            q: Search query (required)
            tier: Filter by tier (optional, comma-separated)
            min_importance: Minimum importance score (0.0-1.0)
            limit: Maximum results (default: 20, max: 100)
            sort: Sort by 'relevance', 'importance', 'recency' (default: relevance)
        """
        if not CONTINUUM_AVAILABLE:
            return error_response("Continuum memory system not available", 503)

        continuum = self.ctx.get("continuum_memory")
        if not continuum:
            return error_response("Continuum memory not initialized", 503)

        query = get_bounded_string_param(params, "q", "", max_length=500)
        if not query:
            return error_response("Missing required parameter: q (search query)", 400)

        tier_param = get_bounded_string_param(params, "tier", "", max_length=100)
        limit = get_clamped_int_param(params, "limit", 20, min_val=1, max_val=100)
        min_importance = get_bounded_float_param(
            params, "min_importance", 0.0, min_val=0.0, max_val=1.0
        )
        sort_by = get_bounded_string_param(params, "sort", "relevance", max_length=20)

        # Parse tiers
        tiers = []
        if tier_param:
            for name in tier_param.split(","):
                try:
                    tiers.append(MemoryTier[name.strip().upper()])
                except KeyError:
                    continue
        if not tiers:
            tiers = list(MemoryTier)

        # Search memories
        memories = continuum.retrieve(
            query=query,
            tiers=tiers,
            limit=limit,
            min_importance=min_importance,
        )

        # Sort results
        if sort_by == "importance":
            memories.sort(key=lambda m: getattr(m, "importance", 0), reverse=True)
        elif sort_by == "recency":
            memories.sort(key=lambda m: getattr(m, "updated_at", 0), reverse=True)
        # 'relevance' is default from retrieve

        results = []
        for m in memories:
            results.append(
                {
                    "id": m.id,
                    "tier": m.tier.name.lower(),
                    "content": m.content[:300] + "..." if len(m.content) > 300 else m.content,
                    "importance": round(getattr(m, "importance", 0.0), 3),
                    "surprise_score": round(getattr(m, "surprise_score", 0.0), 3),
                    "created_at": str(m.created_at) if hasattr(m, "created_at") else None,
                    "updated_at": str(m.updated_at) if hasattr(m, "updated_at") else None,
                    "metadata": getattr(m, "metadata", {}),
                }
            )

        return json_response(
            {
                "query": query,
                "results": results,
                "count": len(results),
                "tiers_searched": [t.name.lower() for t in tiers],
                "filters": {
                    "min_importance": min_importance,
                    "sort": sort_by,
                },
            }
        )

    @handle_errors("get critiques")
    def _get_critiques(self, params: dict) -> HandlerResult:
        """Browse critique store entries.

        Query params:
            agent: Filter by agent name (optional)
            limit: Maximum results (default: 20, max: 100)
            offset: Skip first N results (default: 0)

        Note: debate_id filtering not currently supported by CritiqueStore.
        """
        if not CRITIQUE_STORE_AVAILABLE:
            return error_response("Critique store not available", 503)

        nomic_dir = self.ctx.get("nomic_dir")
        if not nomic_dir:
            return error_response("Critique store not configured", 503)

        agent = get_bounded_string_param(params, "agent", "", max_length=100)
        limit = get_clamped_int_param(params, "limit", 20, min_val=1, max_val=100)
        offset = get_clamped_int_param(params, "offset", 0, min_val=0, max_val=10000)

        try:
            store = CritiqueStore(str(nomic_dir))

            # Get recent critiques - CritiqueStore only supports get_recent()
            # Fetch extra to account for filtering and offset
            fetch_limit = limit + offset + 100 if agent else limit + offset
            all_critiques = store.get_recent(limit=fetch_limit)

            # Filter by agent if specified
            if agent:
                all_critiques = [c for c in all_critiques if c.agent == agent]

            # Apply offset and limit
            critiques = all_critiques[offset : offset + limit]

            results = []
            for c in critiques:
                # Build content from issues/suggestions since Critique has no 'content' field
                content_parts = []
                if c.issues:
                    content_parts.extend(c.issues[:2])
                if c.suggestions:
                    content_parts.extend(c.suggestions[:2])
                content = "; ".join(content_parts)[:300] if content_parts else ""

                results.append(
                    {
                        "id": None,  # Critique dataclass has no id field
                        "debate_id": None,  # Not available from get_recent()
                        "agent": c.agent,
                        "target_agent": c.target_agent,
                        "critique_type": None,  # Not available
                        "content": content,
                        "severity": c.severity,
                        "accepted": None,  # Not available
                        "created_at": None,  # Not available from Critique dataclass
                    }
                )

            # Get total count for pagination
            total = len(all_critiques) if agent else len(store.get_recent(limit=10000))

            return json_response(
                {
                    "critiques": results,
                    "count": len(results),
                    "total": total,
                    "offset": offset,
                    "limit": limit,
                    "filters": {
                        "agent": agent or None,
                    },
                }
            )

        except Exception as e:
            logger.error(f"Failed to get critiques: {e}")
            return error_response(safe_error_message(e, "get critiques"), 500)
