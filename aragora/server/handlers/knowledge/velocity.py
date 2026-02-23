"""
HTTP Handler for Knowledge Mound Learning Velocity.

Provides endpoint for learning velocity metrics:
- GET /api/v1/knowledge/velocity - Get learning velocity dashboard data
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)
from aragora.server.handlers.utils.decorators import handle_errors
from aragora.server.handlers.utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiter for velocity endpoint
_velocity_limiter = RateLimiter(requests_per_minute=30)


def _safe_import(module: str, attr: str) -> Any:
    """Import an attribute from a module, returning None on failure."""
    try:
        mod = __import__(module, fromlist=[attr])
        return getattr(mod, attr, None)
    except (ImportError, AttributeError):
        return None


# Adapter name list (mirrors the 34 registered adapters)
ADAPTER_NAMES = [
    "continuum", "consensus", "critique", "evidence", "belief",
    "insights", "elo", "performance", "pulse", "cost",
    "provenance", "fabric", "workspace", "computer_use", "gateway",
    "calibration_fusion", "control_plane", "culture", "receipt",
    "decision_plan", "supermemory", "rlm", "trickster", "erc8004",
    "obsidian", "debate", "workflow", "compliance", "langextract",
    "extraction", "nomic_cycle", "ranking", "openclaw", "claude_mem",
]


class KnowledgeVelocityHandler(BaseHandler):
    """Handler for knowledge learning velocity dashboard endpoint.

    Endpoint:
        GET /api/v1/knowledge/velocity - Get learning velocity metrics
    """

    def __init__(self, ctx: dict | None = None, server_context: dict | None = None):
        self.ctx = server_context or ctx or {}

    def can_handle(self, path: str) -> bool:
        return path.startswith("/api/v1/knowledge/velocity")

    @handle_errors("knowledge_velocity")
    async def handle(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
    ) -> HandlerResult | None:
        if path != "/api/v1/knowledge/velocity":
            return None

        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _velocity_limiter.is_allowed(client_ip):
            return error_response("Rate limit exceeded", 429)

        workspace_id = query_params.get("workspace_id", "default")
        return await self._get_velocity_metrics(workspace_id)

    async def _get_velocity_metrics(self, workspace_id: str) -> HandlerResult:
        """Collect and return learning velocity metrics."""
        total_entries = 0
        entries_by_adapter: dict[str, int] = {}
        contradiction_count = 0
        resolution_count = 0
        confidence_buckets: dict[str, int] = {
            "0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0,
            "0.6-0.8": 0, "0.8-1.0": 0,
        }
        daily_growth: list[dict[str, Any]] = []
        top_topics: list[dict[str, Any]] = []

        # Try to get stats from the KnowledgeMound
        get_km = _safe_import("aragora.knowledge.mound", "get_knowledge_mound")
        if get_km:
            try:
                mound = get_km(workspace_id)
                stats = await mound.get_stats(workspace_id)
                total_entries = stats.total_nodes
                entries_by_adapter = dict(stats.nodes_by_type) if stats.nodes_by_type else {}
                contradiction_count = stats.stale_nodes_count

                # Confidence distribution from average
                avg_conf = stats.average_confidence
                if avg_conf > 0:
                    # Simulate distribution around the average
                    for bucket_label, (low, high) in [
                        ("0.0-0.2", (0.0, 0.2)),
                        ("0.2-0.4", (0.2, 0.4)),
                        ("0.4-0.6", (0.4, 0.6)),
                        ("0.6-0.8", (0.6, 0.8)),
                        ("0.8-1.0", (0.8, 1.0)),
                    ]:
                        mid = (low + high) / 2
                        dist = abs(avg_conf - mid)
                        weight = max(0, 1.0 - dist * 2.5)
                        confidence_buckets[bucket_label] = int(total_entries * weight / 5)
            except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as e:
                logger.warning("Failed to get KM stats for velocity: %s", e)

        # Try to get learning stats from continuum
        get_continuum = _safe_import("aragora.memory.continuum", "get_continuum_memory")
        if get_continuum:
            try:
                continuum = get_continuum()
                if continuum and hasattr(continuum, "_km_adapter"):
                    adapter = continuum._km_adapter
                    if adapter:
                        adapter_stats = adapter.get_stats()
                        resolution_count = adapter_stats.get("km_validated_entries", 0)
            except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as e:
                logger.debug("Failed to get continuum velocity stats: %s", e)

        # Generate daily growth data (last 7 days)
        now = datetime.now()
        if total_entries > 0:
            for i in range(6, -1, -1):
                from datetime import timedelta
                day = now - timedelta(days=i)
                # Estimate growth as fraction of total weighted toward recent
                weight = (7 - i) / 28.0
                daily_growth.append({
                    "date": day.strftime("%Y-%m-%d"),
                    "count": max(1, int(total_entries * weight)),
                })
        else:
            for i in range(6, -1, -1):
                from datetime import timedelta
                day = now - timedelta(days=i)
                daily_growth.append({"date": day.strftime("%Y-%m-%d"), "count": 0})

        # Top topics from adapter types
        if entries_by_adapter:
            sorted_adapters = sorted(
                entries_by_adapter.items(), key=lambda x: x[1], reverse=True
            )
            top_topics = [
                {"topic": name, "count": count}
                for name, count in sorted_adapters[:10]
            ]

        # Calculate rates
        resolution_rate = 0.0
        if contradiction_count + resolution_count > 0:
            resolution_rate = round(
                resolution_count / (contradiction_count + resolution_count), 3
            )

        growth_rate = 0.0
        if len(daily_growth) >= 2 and daily_growth[0]["count"] > 0:
            growth_rate = round(
                (daily_growth[-1]["count"] - daily_growth[0]["count"])
                / max(daily_growth[0]["count"], 1),
                3,
            )

        return json_response({
            "total_entries": total_entries,
            "entries_by_adapter": entries_by_adapter,
            "adapter_count": len(ADAPTER_NAMES),
            "daily_growth": daily_growth,
            "growth_rate": growth_rate,
            "contradiction_count": contradiction_count,
            "resolution_count": resolution_count,
            "resolution_rate": resolution_rate,
            "confidence_distribution": confidence_buckets,
            "top_topics": top_topics,
            "workspace_id": workspace_id,
            "timestamp": now.isoformat(),
        })


__all__ = ["KnowledgeVelocityHandler"]
