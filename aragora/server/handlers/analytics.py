"""
Analytics and metrics endpoint handlers.

Endpoints:
- GET /api/analytics/disagreements - Get disagreement statistics
- GET /api/analytics/role-rotation - Get role rotation statistics
- GET /api/analytics/early-stops - Get early stopping statistics
- GET /api/ranking/stats - Get ranking statistics
- GET /api/memory/stats - Get memory statistics
"""

import logging
from typing import Optional

from aragora.config import (
    DB_INSIGHTS_PATH,
    CACHE_TTL_ANALYTICS,
    CACHE_TTL_ANALYTICS_RANKING,
    CACHE_TTL_ANALYTICS_DEBATES,
    CACHE_TTL_ANALYTICS_MEMORY,
)

logger = logging.getLogger(__name__)
from .base import BaseHandler, HandlerResult, json_response, error_response, get_int_param, ttl_cache


class AnalyticsHandler(BaseHandler):
    """Handler for analytics and metrics endpoints."""

    ROUTES = [
        "/api/analytics/disagreements",
        "/api/analytics/role-rotation",
        "/api/analytics/early-stops",
        "/api/ranking/stats",
        "/api/memory/stats",
        # Note: /api/memory/tier-stats moved to MemoryHandler for more specific handling
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route analytics requests to appropriate methods."""
        if path == "/api/analytics/disagreements":
            return self._get_disagreement_stats()

        if path == "/api/analytics/role-rotation":
            return self._get_role_rotation_stats()

        if path == "/api/analytics/early-stops":
            return self._get_early_stop_stats()

        if path == "/api/ranking/stats":
            return self._get_ranking_stats()

        if path == "/api/memory/stats":
            return self._get_memory_stats()

        return None

    @ttl_cache(ttl_seconds=CACHE_TTL_ANALYTICS, key_prefix="analytics_disagreement", skip_first=True)
    def _get_disagreement_stats(self) -> HandlerResult:
        """Get statistics about debate disagreements."""
        storage = self.get_storage()
        if not storage:
            return json_response({"stats": {}})

        try:
            debates = storage.list_debates(limit=100)

            stats = {
                "total_debates": len(debates),
                "with_disagreements": 0,
                "unanimous": 0,
                "disagreement_types": {},
            }

            for debate in debates:
                result = debate.get("result", {})
                report = result.get("disagreement_report")
                if report:
                    if report.get("unanimous_critiques"):
                        stats["with_disagreements"] += 1
                    else:
                        stats["unanimous"] += 1

                    dtype = result.get("uncertainty_metrics", {}).get("disagreement_type", "unknown")
                    stats["disagreement_types"][dtype] = stats["disagreement_types"].get(dtype, 0) + 1

            return json_response({"stats": stats})
        except Exception as e:
            return error_response(f"Failed to get disagreement stats: {e}", 500)

    @ttl_cache(ttl_seconds=CACHE_TTL_ANALYTICS, key_prefix="analytics_roles", skip_first=True)
    def _get_role_rotation_stats(self) -> HandlerResult:
        """Get statistics about cognitive role rotation."""
        storage = self.get_storage()
        if not storage:
            return json_response({"stats": {}})

        try:
            debates = storage.list_debates(limit=100)

            stats = {
                "total_debates": len(debates),
                "with_rotation": 0,
                "role_assignments": {},
            }

            for debate in debates:
                messages = debate.get("messages", [])
                for msg in messages:
                    role = msg.get("cognitive_role", msg.get("role", "unknown"))
                    stats["role_assignments"][role] = stats["role_assignments"].get(role, 0) + 1

            return json_response({"stats": stats})
        except Exception as e:
            return error_response(f"Failed to get role rotation stats: {e}", 500)

    @ttl_cache(ttl_seconds=CACHE_TTL_ANALYTICS, key_prefix="analytics_early_stop", skip_first=True)
    def _get_early_stop_stats(self) -> HandlerResult:
        """Get statistics about early debate stopping."""
        storage = self.get_storage()
        if not storage:
            return json_response({"stats": {}})

        try:
            debates = storage.list_debates(limit=100)

            stats = {
                "total_debates": len(debates),
                "early_stopped": 0,
                "full_rounds": 0,
                "average_rounds": 0,
            }

            total_rounds = 0
            for debate in debates:
                result = debate.get("result", {})
                rounds = result.get("rounds_used", 0)
                total_rounds += rounds

                if result.get("early_stopped"):
                    stats["early_stopped"] += 1
                else:
                    stats["full_rounds"] += 1

            if debates:
                stats["average_rounds"] = total_rounds / len(debates)

            return json_response({"stats": stats})
        except Exception as e:
            return error_response(f"Failed to get early stop stats: {e}", 500)

    @ttl_cache(ttl_seconds=CACHE_TTL_ANALYTICS_RANKING, key_prefix="analytics_ranking", skip_first=True)
    def _get_ranking_stats(self) -> HandlerResult:
        """Get ranking system statistics."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("Ranking system not available", 503)

        try:
            leaderboard = elo.get_leaderboard(limit=100)

            stats = {
                "total_agents": len(leaderboard),
                "total_matches": sum(a.total_debates for a in leaderboard) if leaderboard else 0,
                "avg_elo": sum(a.elo_rating for a in leaderboard) / len(leaderboard) if leaderboard else 1500,
                "top_agent": leaderboard[0].agent_name if leaderboard else None,
                "elo_range": {
                    "min": min(a.elo_rating for a in leaderboard) if leaderboard else 1500,
                    "max": max(a.elo_rating for a in leaderboard) if leaderboard else 1500,
                },
            }

            return json_response({"stats": stats})
        except Exception as e:
            return error_response(f"Failed to get ranking stats: {e}", 500)

    @ttl_cache(ttl_seconds=CACHE_TTL_ANALYTICS_DEBATES, key_prefix="analytics_debates", skip_first=True)
    def _get_cached_debates(self, limit: int = 100) -> list:
        """Cached helper for retrieving debates."""
        storage = self.get_storage()
        if not storage:
            return []
        try:
            return storage.list_debates(limit=limit)
        except Exception as e:
            logger.warning("Failed to list debates for analytics: %s: %s", type(e).__name__, e)
            return []

    @ttl_cache(ttl_seconds=CACHE_TTL_ANALYTICS_MEMORY, key_prefix="analytics_memory", skip_first=True)
    def _get_memory_stats(self) -> HandlerResult:
        """Get memory system statistics."""
        try:
            nomic_dir = self.get_nomic_dir()
            if not nomic_dir:
                return json_response({"stats": {}})

            stats = {
                "embeddings_db": False,
                "insights_db": False,
                "continuum_memory": False,
            }

            # Check for database files
            if (nomic_dir / "debate_embeddings.db").exists():
                stats["embeddings_db"] = True

            if (nomic_dir / DB_INSIGHTS_PATH).exists():
                stats["insights_db"] = True

            if (nomic_dir / "continuum_memory.db").exists():
                stats["continuum_memory"] = True

            return json_response({"stats": stats})
        except Exception as e:
            return error_response(f"Failed to get memory stats: {e}", 500)
