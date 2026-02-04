"""
Memory-related endpoint handlers.

Endpoints:
- GET /api/v1/memory/continuum/retrieve - Retrieve memories from continuum
- POST /api/v1/memory/continuum/consolidate - Trigger memory consolidation
- POST /api/v1/memory/continuum/cleanup - Cleanup expired memories
- GET /api/v1/memory/tier-stats - Get tier statistics
- GET /api/v1/memory/archive-stats - Get archive statistics
- GET /api/v1/memory/pressure - Get memory pressure and utilization
- DELETE /api/v1/memory/continuum/{id} - Delete a memory by ID
- GET /api/v1/memory/tiers - List all memory tiers with detailed stats
- GET /api/v1/memory/search - Search memories across tiers
- GET /api/v1/memory/critiques - Browse critique store entries

Note: `/api/memory/*` legacy routes are normalized to `/api/v1/memory/*`.
"""

from __future__ import annotations

from typing import Any

import logging
import math
import time

from aragora.events.handler_events import emit_handler_event, COMPLETED
from aragora.rbac.decorators import require_permission
from aragora.utils.async_utils import run_async

from ..base import (
    HandlerResult,
    error_response,
    get_bounded_float_param,
    get_bounded_string_param,
    get_clamped_int_param,
    handle_errors,
    json_response,
    safe_error_message,
)
from ..secure import SecureHandler
from ..utils.rate_limit import RateLimiter, get_client_ip, rate_limit

# Rate limiters for memory endpoints
_retrieve_limiter = RateLimiter(requests_per_minute=60)  # Read operations
_stats_limiter = RateLimiter(requests_per_minute=30)  # Stats operations
_mutation_limiter = RateLimiter(requests_per_minute=10)  # State-changing operations

logger = logging.getLogger(__name__)

# Permissions for memory endpoints
MEMORY_READ_PERMISSION = "memory:read"
MEMORY_WRITE_PERMISSION = "memory:write"
MEMORY_MANAGE_PERMISSION = "memory:manage"

# Pre-declare optional import names for type safety
MemoryTier: Any = None
CritiqueStore: Any = None

# Optional import for memory functionality
try:
    from aragora.memory.continuum import MemoryTier

    CONTINUUM_AVAILABLE = True
except ImportError:
    CONTINUUM_AVAILABLE = False

# Optional import for critique store - use canonical helper
from aragora.stores.canonical import get_critique_store, is_critique_store_available

CRITIQUE_STORE_AVAILABLE = is_critique_store_available()

# Optional import for critique store class (for direct fallback + test patching)
try:
    from aragora.memory.store import CritiqueStore
except ImportError:
    CritiqueStore = None


class MemoryHandler(SecureHandler):
    """Handler for memory-related endpoints.

    Requires authentication and memory:read/write permissions (RBAC).
    """

    def __init__(self, ctx: dict | None = None, server_context: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = server_context or ctx or {}

    ROUTES = [
        "/api/v1/memory/continuum/retrieve",
        "/api/v1/memory/continuum/consolidate",
        "/api/v1/memory/continuum/cleanup",
        "/api/v1/memory/tier-stats",
        "/api/v1/memory/archive-stats",
        "/api/v1/memory/pressure",
        "/api/v1/memory/tiers",
        "/api/v1/memory/search",
        "/api/v1/memory/search-index",
        "/api/v1/memory/search-timeline",
        "/api/v1/memory/entries",
        "/api/v1/memory/viewer",
        "/api/v1/memory/critiques",
    ]

    @staticmethod
    def _normalize_path(path: str) -> str:
        if path.startswith("/api/memory/"):
            return "/api/v1" + path[len("/api") :]
        return path

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        normalized = self._normalize_path(path)
        if normalized in self.ROUTES:
            return True
        # Handle /api/memory/continuum/{id} pattern for DELETE
        if normalized.startswith("/api/v1/memory/continuum/") and normalized.count("/") == 5:
            # Exclude known routes like /api/memory/continuum/retrieve
            segment = normalized.split("/")[-1]
            if segment not in ("retrieve", "consolidate", "cleanup"):
                return True
        return False

    def _get_user_store(self) -> Any:
        """Get user store from context."""
        return self.ctx.get("user_store")

    @require_permission("memory:read")
    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        """Route memory requests to appropriate handler methods."""
        path = self._normalize_path(path)
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

        if path == "/api/v1/memory/search-index":
            if not _retrieve_limiter.is_allowed(client_ip):
                return error_response("Rate limit exceeded. Please try again later.", 429)
            return self._search_index(query_params)

        if path == "/api/v1/memory/search-timeline":
            if not _retrieve_limiter.is_allowed(client_ip):
                return error_response("Rate limit exceeded. Please try again later.", 429)
            return self._search_timeline(query_params)

        if path == "/api/v1/memory/entries":
            if not _retrieve_limiter.is_allowed(client_ip):
                return error_response("Rate limit exceeded. Please try again later.", 429)
            return self._get_entries(query_params)

        if path == "/api/v1/memory/viewer":
            if not _retrieve_limiter.is_allowed(client_ip):
                return error_response("Rate limit exceeded. Please try again later.", 429)
            return self._render_viewer()

        if path == "/api/v1/memory/critiques":
            # Rate limit: 30/min for stats operations
            if not _stats_limiter.is_allowed(client_ip):
                return error_response("Rate limit exceeded. Please try again later.", 429)
            return self._get_critiques(query_params)

        return None

    @require_permission("memory:manage")
    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Route POST memory requests to admin-level state-mutating methods with auth."""
        path = self._normalize_path(path)
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

    @rate_limit(requests_per_minute=60, limiter_name="memory_read")
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

    @rate_limit(requests_per_minute=20, limiter_name="memory_write")
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

        emit_handler_event("memory", COMPLETED, {"entries_processed": result.get("processed", 0)})
        return json_response(
            {
                "success": True,
                "entries_processed": result.get("processed", 0),
                "entries_promoted": result.get("promoted", 0),
                "entries_consolidated": result.get("consolidated", 0),
                "duration_seconds": round(duration, 2),
            }
        )

    @rate_limit(requests_per_minute=10, limiter_name="memory_delete")
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

    @rate_limit(requests_per_minute=60, limiter_name="memory_read")
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

    @rate_limit(requests_per_minute=60, limiter_name="memory_read")
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

    @rate_limit(requests_per_minute=60, limiter_name="memory_read")
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

    def handle_delete(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Route DELETE memory requests to appropriate methods with auth."""
        path = self._normalize_path(path)
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
            # After split("/") = ["", "api", "v1", "memory", "continuum", "{id}"]
            # memory_id is at index 5
            memory_id, err = self.extract_path_param(path, 5, "memory_id")
            if err:
                return err
            return self._delete_memory(memory_id)
        return None

    @rate_limit(requests_per_minute=10, limiter_name="memory_delete")
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

    @rate_limit(requests_per_minute=60, limiter_name="memory_read")
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

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, int(math.ceil(len(text) / 4)))

    @staticmethod
    def _parse_bool_param(params: dict, name: str, default: bool = False) -> bool:
        raw = get_bounded_string_param(params, name, "", max_length=10)
        if raw == "":
            return default
        return raw.strip().lower() in ("1", "true", "yes", "y", "on")

    def _parse_tiers_param(self, params: dict, name: str = "tier") -> list["MemoryTier"]:
        tier_param = get_bounded_string_param(params, name, "", max_length=100)
        tiers: list[MemoryTier] = []
        if tier_param:
            for tier_name in tier_param.split(","):
                tier_name = tier_name.strip()
                if not tier_name:
                    continue
                try:
                    tiers.append(MemoryTier[tier_name.upper()])
                except KeyError:
                    continue
        return tiers or list(MemoryTier)

    def _format_entry_summary(
        self,
        entry: Any,
        *,
        preview_chars: int = 220,
        include_metadata: bool = False,
        include_content: bool = False,
    ) -> dict[str, Any]:
        def _get(attr: str, default: Any = None) -> Any:
            if isinstance(entry, dict):
                return entry.get(attr, default)
            return getattr(entry, attr, default)

        content = _get("content", "") or ""
        preview = content[:preview_chars].rstrip()
        if len(content) > preview_chars:
            preview = f"{preview}..."

        tier_value = _get("tier")
        tier_name = None
        if isinstance(tier_value, MemoryTier):
            tier_name = tier_value.name.lower()
        elif hasattr(tier_value, "name"):
            tier_name = str(getattr(tier_value, "name")).lower()
        elif tier_value is not None:
            tier_name = str(tier_value).lower()

        def _to_float(value: Any) -> float | None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        importance = _to_float(_get("importance"))
        surprise = _to_float(_get("surprise_score", _get("surprise")))

        created_at = _get("created_at")
        updated_at = _get("updated_at")

        result: dict[str, Any] = {
            "id": _get("id") or _get("memory_id"),
            "tier": tier_name,
            "preview": preview,
            "importance": round(importance, 3) if importance is not None else None,
            "surprise_score": round(surprise, 3) if surprise is not None else None,
            "created_at": str(created_at) if created_at is not None else None,
            "updated_at": str(updated_at) if updated_at is not None else None,
            "token_estimate": self._estimate_tokens(content),
        }

        red_line = _get("red_line", None)
        if red_line is not None:
            result["red_line"] = bool(red_line)
            red_line_reason = _get("red_line_reason", "")
            if red_line_reason:
                result["red_line_reason"] = red_line_reason

        if include_content:
            result["content"] = content
        if include_metadata:
            result["metadata"] = _get("metadata", {}) or {}

        return result

    def _format_entry_full(self, entry: Any) -> dict[str, Any]:
        result = self._format_entry_summary(
            entry,
            preview_chars=600,
            include_metadata=True,
            include_content=True,
        )
        result["token_estimate"] = self._estimate_tokens(result.get("content", ""))
        return result

    def _html_response(self, html: str, status: int = 200) -> HandlerResult:
        return HandlerResult(
            status_code=status,
            content_type="text/html",
            body=html.encode("utf-8"),
        )

    def _get_supermemory_adapter(self) -> Any | None:
        if hasattr(self, "_supermemory_client"):
            return getattr(self, "_supermemory_client")
        try:
            from aragora.connectors.supermemory import SupermemoryConfig, get_client
        except ImportError:
            return None

        config = SupermemoryConfig.from_env()
        if not config:
            return None

        try:
            client = get_client(config)
        except Exception as exc:  # pragma: no cover - external dependency
            logger.debug(f"Supermemory client init failed: {exc}")
            return None

        setattr(self, "_supermemory_client", client)
        setattr(self, "_supermemory_config", config)
        return client

    def _search_supermemory(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        client = self._get_supermemory_adapter()
        if not client:
            return []

        config = getattr(self, "_supermemory_config", None)
        container_tag = getattr(config, "container_tag", None)

        try:
            response = run_async(
                client.search(query=query, limit=limit, container_tag=container_tag)
            )
        except Exception as exc:  # pragma: no cover - external dependency
            logger.debug(f"Supermemory search failed: {exc}")
            return []

        results: list[dict[str, Any]] = []
        for idx, item in enumerate(getattr(response, "results", []) or []):
            content = getattr(item, "content", "") or ""
            preview = content[:220].rstrip()
            if len(content) > 220:
                preview = f"{preview}..."
            results.append(
                {
                    "id": getattr(item, "memory_id", None) or f"super_{idx}",
                    "source": "supermemory",
                    "preview": preview,
                    "score": round(float(getattr(item, "similarity", 0.0)), 4),
                    "token_estimate": self._estimate_tokens(content),
                    "metadata": getattr(item, "metadata", {}) or {},
                    "container_tag": getattr(item, "container_tag", None),
                }
            )
        return results

    def _search_claude_mem(
        self, query: str, limit: int = 10, project: str | None = None
    ) -> list[dict[str, Any]]:
        try:
            from aragora.connectors import ClaudeMemConnector, ClaudeMemConfig
        except ImportError:
            return []

        connector = ClaudeMemConnector(ClaudeMemConfig.from_env())
        try:
            evidence = run_async(connector.search(query, limit=limit, project=project))
        except Exception as exc:  # pragma: no cover - external dependency
            logger.debug(f"Claude-mem search failed: {exc}")
            return []

        results: list[dict[str, Any]] = []
        for item in evidence:
            content = getattr(item, "content", "") or ""
            preview = content[:220].rstrip()
            if len(content) > 220:
                preview = f"{preview}..."
            results.append(
                {
                    "id": getattr(item, "id", None),
                    "source": "claude-mem",
                    "preview": preview,
                    "token_estimate": self._estimate_tokens(content),
                    "metadata": getattr(item, "metadata", {}) or {},
                    "created_at": getattr(item, "created_at", None),
                }
            )
        return results

    @rate_limit(requests_per_minute=60, limiter_name="memory_read")
    @handle_errors("search index retrieval")
    def _search_index(self, params: dict) -> HandlerResult:
        """Progressive retrieval stage 1: compact index entries."""
        if not CONTINUUM_AVAILABLE:
            return error_response("Continuum memory system not available", 503)

        continuum = self.ctx.get("continuum_memory")
        if not continuum:
            return error_response("Continuum memory not initialized", 503)

        query = get_bounded_string_param(params, "q", "", max_length=500)
        if not query:
            return error_response("Missing required parameter: q (search query)", 400)

        limit = get_clamped_int_param(params, "limit", 20, min_val=1, max_val=100)
        min_importance = get_bounded_float_param(
            params, "min_importance", 0.0, min_val=0.0, max_val=1.0
        )
        tiers = self._parse_tiers_param(params)
        use_hybrid = self._parse_bool_param(params, "use_hybrid", False)

        results: list[dict[str, Any]] = []
        hybrid_used = False

        if use_hybrid and hasattr(continuum, "hybrid_search"):
            try:
                hybrid_results = run_async(
                    continuum.hybrid_search(
                        query=query,
                        limit=limit,
                        tiers=tiers,
                        min_importance=min_importance,
                    )
                )
                hybrid_used = True
            except Exception as exc:  # pragma: no cover - optional dependency
                logger.debug(f"Hybrid memory search failed, falling back: {exc}")
                hybrid_results = []

            ids = [r.memory_id for r in hybrid_results if getattr(r, "memory_id", None)]
            entries_by_id = {}
            if ids and hasattr(continuum, "get_many"):
                entries = continuum.get_many(ids)
                entries_by_id = {entry.id: entry for entry in entries}

            for result in hybrid_results:
                entry = entries_by_id.get(result.memory_id)
                summary = self._format_entry_summary(entry or result)
                summary["source"] = "continuum"
                summary["score"] = round(float(getattr(result, "combined_score", 0.0)), 4)
                summary["id"] = result.memory_id
                results.append(summary)

        if not hybrid_used:
            memories = continuum.retrieve(
                query=query,
                tiers=tiers,
                limit=limit,
                min_importance=min_importance,
            )
            for entry in memories:
                summary = self._format_entry_summary(entry)
                summary["source"] = "continuum"
                results.append(summary)

        include_external = self._parse_bool_param(params, "include_external", False)
        external_param = get_bounded_string_param(params, "external", "", max_length=200)
        external_results: list[dict[str, Any]] = []
        external_sources: list[str] = []

        if include_external:
            requested = []
            if external_param:
                for name in external_param.split(","):
                    name = name.strip().lower()
                    if not name:
                        continue
                    if name in ("claude_mem", "claudemem", "claude-mem"):
                        name = "claude-mem"
                    if name in ("supermemory", "super-memory", "sm"):
                        name = "supermemory"
                    requested.append(name)
            else:
                requested = ["supermemory", "claude-mem"]

            if "supermemory" in requested:
                external_results.extend(self._search_supermemory(query, limit=limit))
                external_sources.append("supermemory")
            if "claude-mem" in requested:
                project = get_bounded_string_param(params, "project", "", max_length=200) or None
                external_results.extend(
                    self._search_claude_mem(query, limit=limit, project=project)
                )
                external_sources.append("claude-mem")

        return json_response(
            {
                "query": query,
                "results": results,
                "count": len(results),
                "tiers": [tier.name.lower() for tier in tiers],
                "use_hybrid": hybrid_used,
                "external_sources": external_sources,
                "external_results": external_results,
            }
        )

    @rate_limit(requests_per_minute=60, limiter_name="memory_read")
    @handle_errors("timeline retrieval")
    def _search_timeline(self, params: dict) -> HandlerResult:
        """Progressive retrieval stage 2: timeline around an anchor."""
        if not CONTINUUM_AVAILABLE:
            return error_response("Continuum memory system not available", 503)

        continuum = self.ctx.get("continuum_memory")
        if not continuum:
            return error_response("Continuum memory not initialized", 503)

        anchor_id = get_bounded_string_param(params, "anchor_id", "", max_length=200)
        if not anchor_id:
            return error_response("Missing required parameter: anchor_id", 400)

        before = get_clamped_int_param(params, "before", 3, min_val=0, max_val=50)
        after = get_clamped_int_param(params, "after", 3, min_val=0, max_val=50)
        min_importance = get_bounded_float_param(
            params, "min_importance", 0.0, min_val=0.0, max_val=1.0
        )
        tiers = self._parse_tiers_param(params)

        if not hasattr(continuum, "get_timeline_entries"):
            return error_response("Timeline retrieval not supported", 501)

        timeline = continuum.get_timeline_entries(
            anchor_id=anchor_id,
            before=before,
            after=after,
            tiers=tiers,
            min_importance=min_importance,
        )
        if timeline is None:
            return error_response("Anchor memory not found", 404)

        return json_response(
            {
                "anchor_id": anchor_id,
                "anchor": self._format_entry_summary(timeline["anchor"], preview_chars=260),
                "before": [
                    self._format_entry_summary(entry, preview_chars=220)
                    for entry in timeline["before"]
                ],
                "after": [
                    self._format_entry_summary(entry, preview_chars=220)
                    for entry in timeline["after"]
                ],
            }
        )

    @rate_limit(requests_per_minute=60, limiter_name="memory_read")
    @handle_errors("entries retrieval")
    def _get_entries(self, params: dict) -> HandlerResult:
        """Progressive retrieval stage 3: full entries by ID."""
        if not CONTINUUM_AVAILABLE:
            return error_response("Continuum memory system not available", 503)

        continuum = self.ctx.get("continuum_memory")
        if not continuum:
            return error_response("Continuum memory not initialized", 503)

        ids_param = get_bounded_string_param(params, "ids", "", max_length=2000)
        if not ids_param:
            return error_response("Missing required parameter: ids", 400)

        ids = [entry_id.strip() for entry_id in ids_param.split(",") if entry_id.strip()]
        if not ids:
            return error_response("Missing required parameter: ids", 400)

        if not hasattr(continuum, "get_many"):
            return error_response("Bulk entry retrieval not supported", 501)

        entries = continuum.get_many(ids)

        return json_response(
            {
                "ids": ids,
                "count": len(entries),
                "entries": [self._format_entry_full(entry) for entry in entries],
            }
        )

    def _render_viewer(self) -> HandlerResult:
        """Render the memory viewer HTML UI."""
        html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Aragora Memory Viewer</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Space+Grotesk:wght@400;600;700&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg-1: #f6f1e9;
      --bg-2: #e7efe9;
      --ink: #111214;
      --muted: #5c6166;
      --accent: #0f766e;
      --accent-2: #c46f2b;
      --panel: #ffffff;
      --line: #e3e0da;
      --shadow: 0 24px 60px rgba(15, 23, 42, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Space Grotesk", system-ui, sans-serif;
      color: var(--ink);
      background: radial-gradient(circle at top left, #fef8ee 0%, var(--bg-1) 45%, var(--bg-2) 100%);
      min-height: 100vh;
    }
    .page {
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 20px 60px;
    }
    header {
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin-bottom: 24px;
    }
    header h1 {
      font-size: 32px;
      margin: 0;
      letter-spacing: -0.02em;
    }
    header p {
      margin: 0;
      color: var(--muted);
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) minmax(280px, 1.4fr) minmax(260px, 1fr);
      gap: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      animation: rise 0.5s ease both;
    }
    .panel h2 {
      font-size: 16px;
      margin: 0;
      letter-spacing: 0.02em;
      text-transform: uppercase;
      color: var(--muted);
    }
    label {
      font-size: 12px;
      text-transform: uppercase;
      color: var(--muted);
      letter-spacing: 0.08em;
    }
    input[type="text"],
    input[type="number"] {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 14px;
      font-family: "Space Grotesk", sans-serif;
      width: 100%;
    }
    .mono {
      font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
    }
    .tier-row, .toggle-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .tier-pill {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      font-size: 12px;
      background: #faf8f3;
    }
    .primary {
      background: var(--accent);
      color: #fff;
      border: none;
      padding: 10px 16px;
      border-radius: 10px;
      cursor: pointer;
      font-weight: 600;
      letter-spacing: 0.02em;
    }
    .primary:hover {
      background: #0b5d56;
    }
    .list {
      display: flex;
      flex-direction: column;
      gap: 10px;
      max-height: 520px;
      overflow: auto;
      padding-right: 6px;
    }
    .item {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      text-align: left;
      background: #fff;
      cursor: pointer;
      transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .item:hover {
      transform: translateY(-2px);
      border-color: var(--accent);
    }
    .item .meta {
      display: flex;
      gap: 8px;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
    }
    .item .preview {
      font-size: 13px;
      margin-top: 6px;
    }
    .detail {
      border-top: 1px dashed var(--line);
      padding-top: 12px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .detail h3 {
      margin: 0;
      font-size: 14px;
      color: var(--accent-2);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .detail pre {
      white-space: pre-wrap;
      background: #f9f7f2;
      border-radius: 10px;
      padding: 12px;
      margin: 0;
    }
    @keyframes rise {
      from { opacity: 0; transform: translateY(12px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @media (max-width: 1000px) {
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="page">
    <header>
      <h1>Memory Viewer</h1>
      <p>Progressive disclosure search across continuum memory, with optional external sources.</p>
    </header>
    <div class="grid">
      <section class="panel">
        <h2>Search</h2>
        <label for="query">Query</label>
        <input id="query" type="text" placeholder="Search memory content" />
        <label>Tiers</label>
        <div class="tier-row">
          <label class="tier-pill"><input type="checkbox" class="tier" value="fast" checked /> fast</label>
          <label class="tier-pill"><input type="checkbox" class="tier" value="medium" checked /> medium</label>
          <label class="tier-pill"><input type="checkbox" class="tier" value="slow" checked /> slow</label>
          <label class="tier-pill"><input type="checkbox" class="tier" value="glacial" checked /> glacial</label>
        </div>
        <label for="limit">Limit</label>
        <input id="limit" type="number" value="20" min="1" max="100" />
        <label for="minImportance">Min Importance</label>
        <input id="minImportance" type="number" value="0" step="0.05" min="0" max="1" />
        <div class="toggle-row">
          <label class="tier-pill"><input id="useHybrid" type="checkbox" /> hybrid search</label>
          <label class="tier-pill"><input id="includeExternal" type="checkbox" /> external</label>
        </div>
        <div class="toggle-row">
          <label class="tier-pill"><input id="extSupermemory" type="checkbox" /> supermemory</label>
          <label class="tier-pill"><input id="extClaudeMem" type="checkbox" /> claude-mem</label>
        </div>
        <button class="primary" id="searchBtn">Search</button>
        <div id="status" class="mono"></div>
      </section>

      <section class="panel">
        <h2>Index</h2>
        <div id="results" class="list"></div>
        <div class="detail">
          <h3>External</h3>
          <div id="external" class="list"></div>
        </div>
      </section>

      <section class="panel">
        <h2>Timeline</h2>
        <div id="timeline" class="list"></div>
        <div class="detail">
          <h3>Entry</h3>
          <pre id="entry">Select a memory to view full content.</pre>
        </div>
      </section>
    </div>
  </div>

  <script>
    const apiBase = "/api/v1/memory";
    const resultsEl = document.getElementById("results");
    const externalEl = document.getElementById("external");
    const timelineEl = document.getElementById("timeline");
    const entryEl = document.getElementById("entry");
    const statusEl = document.getElementById("status");

    const qs = (id) => document.getElementById(id);
    const tiers = () => Array.from(document.querySelectorAll(".tier"))
      .filter((el) => el.checked)
      .map((el) => el.value);

    function setStatus(text, isError = false) {
      statusEl.textContent = text;
      statusEl.style.color = isError ? "#b91c1c" : "#0f766e";
    }

    function clear(el) {
      while (el.firstChild) el.removeChild(el.firstChild);
    }

    function makeItem(item, onClick) {
      const button = document.createElement("button");
      button.className = "item";
      button.type = "button";
      button.addEventListener("click", onClick);

      const meta = document.createElement("div");
      meta.className = "meta";
      const tier = item.tier ? item.tier.toUpperCase() : (item.source || "external");
      meta.textContent = `${tier} | importance ${item.importance ?? "n/a"} | tokens ${item.token_estimate ?? "-"}`;

      const preview = document.createElement("div");
      preview.className = "preview";
      preview.textContent = item.preview || "(no preview)";

      button.appendChild(meta);
      button.appendChild(preview);
      return button;
    }

    async function fetchJson(url) {
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`Request failed: ${res.status}`);
      }
      return res.json();
    }

    async function searchIndex() {
      const query = qs("query").value.trim();
      if (!query) {
        setStatus("Enter a query.", true);
        return;
      }
      setStatus("Searching...");

      const params = new URLSearchParams();
      params.set("q", query);
      params.set("limit", qs("limit").value);
      params.set("min_importance", qs("minImportance").value);
      const tierValues = tiers();
      if (tierValues.length) {
        params.set("tier", tierValues.join(","));
      }
      if (qs("useHybrid").checked) {
        params.set("use_hybrid", "true");
      }
      if (qs("includeExternal").checked) {
        params.set("include_external", "true");
        const ext = [];
        if (qs("extSupermemory").checked) ext.push("supermemory");
        if (qs("extClaudeMem").checked) ext.push("claude-mem");
        if (ext.length) {
          params.set("external", ext.join(","));
        }
      }

      try {
        const data = await fetchJson(`${apiBase}/search-index?${params.toString()}`);
        renderResults(data.results || []);
        renderExternal(data.external_results || []);
        setStatus(`Found ${data.count} results${data.external_results?.length ? " + external" : ""}.`);
        if (data.results && data.results.length) {
          loadTimeline(data.results[0].id);
        }
      } catch (err) {
        setStatus(err.message || "Search failed.", true);
      }
    }

    function renderResults(items) {
      clear(resultsEl);
      if (!items.length) {
        resultsEl.textContent = "No results yet.";
        return;
      }
      items.forEach((item) => {
        const node = makeItem(item, () => loadTimeline(item.id));
        resultsEl.appendChild(node);
      });
    }

    function renderExternal(items) {
      clear(externalEl);
      if (!items.length) {
        externalEl.textContent = "No external results.";
        return;
      }
      items.forEach((item) => {
        const node = makeItem(item, () => showExternal(item));
        externalEl.appendChild(node);
      });
    }

    function showExternal(item) {
      const lines = [];
      lines.push(`source: ${item.source}`);
      if (item.metadata && Object.keys(item.metadata).length) {
        lines.push(`metadata: ${JSON.stringify(item.metadata, null, 2)}`);
      }
      lines.push("");
      lines.push(item.preview || "");
      entryEl.textContent = lines.join("\\n");
    }

    async function loadTimeline(anchorId) {
      if (!anchorId) return;
      const params = new URLSearchParams();
      params.set("anchor_id", anchorId);
      params.set("before", "3");
      params.set("after", "3");
      const tierValues = tiers();
      if (tierValues.length) {
        params.set("tier", tierValues.join(","));
      }
      try {
        const data = await fetchJson(`${apiBase}/search-timeline?${params.toString()}`);
        renderTimeline(data);
        if (data.anchor) {
          loadEntry(data.anchor.id);
        }
      } catch (err) {
        setStatus(err.message || "Timeline failed.", true);
      }
    }

    function renderTimeline(data) {
      clear(timelineEl);
      if (!data || !data.anchor) {
        timelineEl.textContent = "Timeline unavailable.";
        return;
      }
      const items = [...(data.before || []), data.anchor, ...(data.after || [])];
      items.forEach((item) => {
        const node = makeItem(item, () => loadEntry(item.id));
        timelineEl.appendChild(node);
      });
    }

    async function loadEntry(entryId) {
      if (!entryId) return;
      try {
        const data = await fetchJson(`${apiBase}/entries?ids=${encodeURIComponent(entryId)}`);
        const entry = data.entries && data.entries.length ? data.entries[0] : null;
        if (!entry) {
          entryEl.textContent = "Entry not found.";
          return;
        }
        const lines = [];
        lines.push(`id: ${entry.id}`);
        lines.push(`tier: ${entry.tier}`);
        lines.push(`importance: ${entry.importance}`);
        if (entry.red_line) {
          lines.push(`red_line: ${entry.red_line_reason || "true"}`);
        }
        lines.push("");
        lines.push(entry.content || "");
        entryEl.textContent = lines.join("\\n");
      } catch (err) {
        entryEl.textContent = err.message || "Failed to load entry.";
      }
    }

    document.getElementById("searchBtn").addEventListener("click", searchIndex);
  </script>
</body>
</html>"""
        return self._html_response(html)

    @rate_limit(requests_per_minute=60, limiter_name="memory_read")
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

    @rate_limit(requests_per_minute=60, limiter_name="memory_read")
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
            store = get_critique_store(nomic_dir)
            if store is None and CritiqueStore is not None:
                # Fallback for tests or environments where canonical DB is not present.
                store = CritiqueStore(nomic_dir)
            if store is None:
                return error_response("Critique store not available", 503)

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
