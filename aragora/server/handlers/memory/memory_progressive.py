"""Memory progressive retrieval mixin (MemoryProgressiveMixin).

Extracted from memory.py to reduce file size.
Contains progressive retrieval operations (search-index, timeline, entries).

Note: RBAC is handled in MemoryHandler.handle() which calls these mixin methods.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from aragora.rbac.decorators import require_permission  # noqa: F401 - Required for RBAC consistency
from aragora.utils.async_utils import run_async

# Permission constant - used by parent MemoryHandler
MEMORY_READ_PERMISSION = "memory:read"

from ..base import (
    HandlerResult,
    error_response,
    get_bounded_float_param,
    get_bounded_string_param,
    get_clamped_int_param,
    handle_errors,
    json_response,
)
from ..utils.rate_limit import rate_limit

if TYPE_CHECKING:
    from aragora.memory.continuum import MemoryTier

logger = logging.getLogger(__name__)


class MemoryProgressiveMixin:
    """Mixin providing progressive memory retrieval operations."""

    # These attributes/methods are defined in the main class or other mixins
    ctx: dict
    _search_supermemory: Any
    _search_claude_mem: Any
    _format_entry_summary: Any
    _format_entry_full: Any
    _parse_tiers_param: Any
    _parse_bool_param: Any
    _estimate_tokens: Any

    def _get_auth_context(self) -> Any:
        """Safely get auth context (may be None if not set)."""
        return getattr(self, "_auth_context", None)

    @rate_limit(requests_per_minute=60, limiter_name="memory_read")
    @handle_errors("search index retrieval")
    def _search_index(self, params: dict) -> HandlerResult:
        """Progressive retrieval stage 1: compact index entries."""
        from .memory import CONTINUUM_AVAILABLE

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

        try:
            from aragora.memory.access import (
                filter_entries,
                resolve_tenant_id,
                tenant_enforcement_enabled,
            )
        except Exception:
            logger.warning(
                "Memory access module unavailable, RBAC/tenant filtering disabled for search index",
                exc_info=True,
            )
            filter_entries = None  # type: ignore[assignment]
            resolve_tenant_id = None  # type: ignore[assignment]
            tenant_enforcement_enabled = None  # type: ignore[assignment]

        enforce_tenant = tenant_enforcement_enabled() if tenant_enforcement_enabled else False
        tenant_id = resolve_tenant_id(self._get_auth_context()) if resolve_tenant_id else None
        if enforce_tenant and not tenant_id:
            if self._get_auth_context() is None:
                enforce_tenant = False
            else:
                return error_response("Tenant/workspace context required for memory access", 400)
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
                entries = continuum.get_many(ids, tenant_id=tenant_id)
                if filter_entries:
                    entries = filter_entries(entries, self._get_auth_context())
                entries_by_id = {entry.id: entry for entry in entries}

            for result in hybrid_results:
                entry = entries_by_id.get(result.memory_id)
                if entry is None:
                    continue
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
                tenant_id=tenant_id,
                enforce_tenant_isolation=enforce_tenant,
            )
            if filter_entries:
                memories = filter_entries(memories, self._get_auth_context())
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
        from .memory import CONTINUUM_AVAILABLE

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

        try:
            from aragora.memory.access import resolve_tenant_id, tenant_enforcement_enabled
        except Exception:
            logger.warning(
                "Memory access module unavailable, RBAC/tenant filtering disabled for timeline",
                exc_info=True,
            )
            resolve_tenant_id = None  # type: ignore[assignment]
            tenant_enforcement_enabled = None  # type: ignore[assignment]

        enforce_tenant = tenant_enforcement_enabled() if tenant_enforcement_enabled else False
        tenant_id = resolve_tenant_id(self._get_auth_context()) if resolve_tenant_id else None
        if enforce_tenant and not tenant_id:
            if self._get_auth_context() is None:
                enforce_tenant = False
            else:
                return error_response("Tenant/workspace context required for memory access", 400)

        if not hasattr(continuum, "get_timeline_entries"):
            return error_response(
                "Timeline retrieval not supported by this continuum backend. "
                "Ensure the memory system is configured with a backend that supports timeline queries.",
                501,
            )

        timeline = continuum.get_timeline_entries(
            anchor_id=anchor_id,
            before=before,
            after=after,
            tiers=tiers,
            min_importance=min_importance,
            tenant_id=tenant_id,
            enforce_tenant_isolation=enforce_tenant,
        )
        if timeline is None:
            return error_response("Anchor memory not found", 404)

        try:
            from aragora.memory.access import filter_entries
        except Exception:
            logger.warning(
                "Memory access module unavailable, RBAC entry filtering disabled for timeline",
                exc_info=True,
            )
            filter_entries = None  # type: ignore[assignment]

        if filter_entries:
            # Enforce RBAC/tenant visibility on timeline entries
            anchor_list = filter_entries([timeline["anchor"]], self._get_auth_context())
            if not anchor_list:
                return error_response("Anchor memory not found", 404)
            timeline["anchor"] = anchor_list[0]
            timeline["before"] = filter_entries(timeline["before"], self._get_auth_context())
            timeline["after"] = filter_entries(timeline["after"], self._get_auth_context())

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
        from .memory import CONTINUUM_AVAILABLE

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
            return error_response(
                "Bulk entry retrieval not supported by this continuum backend. "
                "Ensure the memory system is configured with a backend that supports batch queries.",
                501,
            )

        try:
            from aragora.memory.access import (
                filter_entries,
                resolve_tenant_id,
                tenant_enforcement_enabled,
            )
        except Exception:
            logger.warning(
                "Memory access module unavailable, RBAC/tenant filtering disabled for bulk retrieval",
                exc_info=True,
            )
            filter_entries = None  # type: ignore[assignment]
            resolve_tenant_id = None  # type: ignore[assignment]
            tenant_enforcement_enabled = None  # type: ignore[assignment]

        enforce_tenant = tenant_enforcement_enabled() if tenant_enforcement_enabled else False
        tenant_id = resolve_tenant_id(self._get_auth_context()) if resolve_tenant_id else None
        if enforce_tenant and not tenant_id:
            if self._get_auth_context() is None:
                enforce_tenant = False
            else:
                return error_response("Tenant/workspace context required for memory access", 400)

        entries = continuum.get_many(ids, tenant_id=tenant_id)
        if filter_entries:
            entries = filter_entries(entries, self._get_auth_context())

        return json_response(
            {
                "ids": ids,
                "count": len(entries),
                "entries": [self._format_entry_full(entry) for entry in entries],
            }
        )

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
        from .memory import CONTINUUM_AVAILABLE, MemoryTier

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
        tiers: list[MemoryTier] = []
        if tier_param:
            for name in tier_param.split(","):
                try:
                    tiers.append(MemoryTier[name.strip().upper()])
                except KeyError:
                    continue
        if not tiers:
            tiers = list(MemoryTier)

        # Search memories (tenant-scoped when available)
        try:
            from aragora.memory.access import (
                filter_entries,
                resolve_tenant_id,
                tenant_enforcement_enabled,
            )
        except Exception:
            logger.warning(
                "Memory access module unavailable, RBAC/tenant filtering disabled for search",
                exc_info=True,
            )
            filter_entries = None  # type: ignore[assignment]
            resolve_tenant_id = None  # type: ignore[assignment]
            tenant_enforcement_enabled = None  # type: ignore[assignment]

        enforce_tenant = tenant_enforcement_enabled() if tenant_enforcement_enabled else False
        tenant_id = resolve_tenant_id(self._get_auth_context()) if resolve_tenant_id else None
        if enforce_tenant and not tenant_id:
            if self._get_auth_context() is None:
                enforce_tenant = False
            else:
                return error_response("Tenant/workspace context required for memory access", 400)
        memories = continuum.retrieve(
            query=query,
            tiers=tiers,
            limit=limit,
            min_importance=min_importance,
            tenant_id=tenant_id,
            enforce_tenant_isolation=enforce_tenant,
        )
        if filter_entries:
            memories = filter_entries(memories, self._get_auth_context())

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
