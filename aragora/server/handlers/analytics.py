"""
Analytics and metrics endpoint handlers.

Endpoints:
- GET /api/analytics/disagreements - Get disagreement statistics
- GET /api/analytics/role-rotation - Get role rotation statistics
- GET /api/analytics/early-stops - Get early stopping statistics
- GET /api/ranking/stats - Get ranking statistics
- GET /api/memory/stats - Get memory statistics
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from aragora.config import (
    DB_INSIGHTS_PATH,
    CACHE_TTL_ANALYTICS,
    CACHE_TTL_ANALYTICS_RANKING,
    CACHE_TTL_ANALYTICS_DEBATES,
    CACHE_TTL_ANALYTICS_MEMORY,
)

logger = logging.getLogger(__name__)
from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    get_int_param,
    ttl_cache,
    handle_errors,
)
from .utils.rate_limit import RateLimiter, get_client_ip

# Rate limiter for analytics endpoints (30 requests per minute - cached data)
_analytics_limiter = RateLimiter(requests_per_minute=30)


class AnalyticsHandler(BaseHandler):
    """Handler for analytics and metrics endpoints."""

    ROUTES = [
        "/api/analytics/disagreements",
        "/api/analytics/role-rotation",
        "/api/analytics/early-stops",
        "/api/analytics/consensus-quality",
        "/api/ranking/stats",
        "/api/memory/stats",
        # Note: /api/memory/tier-stats moved to MemoryHandler for more specific handling
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route analytics requests to appropriate methods."""
        logger.debug(f"Analytics request: {path}")

        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _analytics_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for analytics endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        if path == "/api/analytics/disagreements":
            return self._get_disagreement_stats()

        if path == "/api/analytics/role-rotation":
            return self._get_role_rotation_stats()

        if path == "/api/analytics/early-stops":
            return self._get_early_stop_stats()

        if path == "/api/analytics/consensus-quality":
            return self._get_consensus_quality()

        if path == "/api/ranking/stats":
            return self._get_ranking_stats()

        if path == "/api/memory/stats":
            return self._get_memory_stats()

        return None

    @ttl_cache(
        ttl_seconds=CACHE_TTL_ANALYTICS, key_prefix="analytics_disagreement", skip_first=True
    )
    @handle_errors("disagreement stats retrieval")
    def _get_disagreement_stats(self) -> HandlerResult:
        """Get statistics about debate disagreements."""
        storage = self.get_storage()
        if not storage:
            return json_response({"stats": {}})

        debates = storage.list_debates(limit=100)

        stats: dict[str, Any] = {
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

        logger.info(
            f"Disagreement stats: {stats['total_debates']} debates, {stats['with_disagreements']} with disagreements"
        )
        return json_response({"stats": stats})

    @ttl_cache(ttl_seconds=CACHE_TTL_ANALYTICS, key_prefix="analytics_roles", skip_first=True)
    @handle_errors("role rotation stats retrieval")
    def _get_role_rotation_stats(self) -> HandlerResult:
        """Get statistics about cognitive role rotation."""
        storage = self.get_storage()
        if not storage:
            return json_response({"stats": {}})

        debates = storage.list_debates(limit=100)

        stats: dict[str, Any] = {
            "total_debates": len(debates),
            "with_rotation": 0,
            "role_assignments": {},
        }

        for debate in debates:
            messages = debate.get("messages", [])
            for msg in messages:
                role = msg.get("cognitive_role", msg.get("role", "unknown"))
                stats["role_assignments"][role] = stats["role_assignments"].get(role, 0) + 1

        logger.info(
            f"Role rotation stats: {len(stats['role_assignments'])} roles across {stats['total_debates']} debates"
        )
        return json_response({"stats": stats})

    @ttl_cache(ttl_seconds=CACHE_TTL_ANALYTICS, key_prefix="analytics_early_stop", skip_first=True)
    @handle_errors("early stop stats retrieval")
    def _get_early_stop_stats(self) -> HandlerResult:
        """Get statistics about early debate stopping."""
        storage = self.get_storage()
        if not storage:
            return json_response({"stats": {}})

        debates = storage.list_debates(limit=100)

        stats = {
            "total_debates": len(debates),
            "early_stopped": 0,
            "full_rounds": 0,
            "average_rounds": 0.0,
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

        logger.info(
            f"Early stop stats: {stats['early_stopped']}/{stats['total_debates']} early stopped"
        )
        return json_response({"stats": stats})

    @ttl_cache(
        ttl_seconds=CACHE_TTL_ANALYTICS, key_prefix="analytics_consensus_quality", skip_first=True
    )
    @handle_errors("consensus quality stats retrieval")
    def _get_consensus_quality(self) -> HandlerResult:
        """Get consensus quality monitoring metrics.

        Tracks consensus confidence history across debates and detects declining trends.
        Returns quality metrics including:
        - confidence_history: Recent consensus confidence scores
        - trend: 'improving', 'stable', 'declining'
        - average_confidence: Mean confidence across recent debates
        - consensus_rate: Percentage of debates reaching consensus
        - quality_score: Overall quality score (0-100)
        - alert: Warning if quality is below threshold
        """
        storage = self.get_storage()
        if not storage:
            return json_response({"stats": {}, "quality_score": 0, "alert": None})

        debates = storage.list_debates(limit=50)

        # Extract confidence history
        confidence_history: list[dict] = []
        consensus_reached_count = 0

        for debate in debates:
            result = debate.get("result", {})
            confidence = result.get("confidence", 0.0)
            consensus = result.get("consensus_reached", False)
            debate_id = debate.get("id", "")
            timestamp = debate.get("timestamp", "")

            confidence_history.append(
                {
                    "debate_id": debate_id[:8] if debate_id else "",
                    "confidence": confidence,
                    "consensus_reached": consensus,
                    "timestamp": timestamp,
                }
            )

            if consensus:
                consensus_reached_count += 1

        # Calculate metrics
        total_debates = len(debates)
        if total_debates == 0:
            return json_response(
                {
                    "stats": {
                        "total_debates": 0,
                        "confidence_history": [],
                        "trend": "insufficient_data",
                        "average_confidence": 0.0,
                        "consensus_rate": 0.0,
                    },
                    "quality_score": 0,
                    "alert": None,
                }
            )

        confidences = [h["confidence"] for h in confidence_history]
        average_confidence = sum(confidences) / len(confidences)
        consensus_rate = consensus_reached_count / total_debates

        # Detect trend using simple linear regression
        trend = "stable"
        if len(confidences) >= 5:
            # Compare first half vs second half
            mid = len(confidences) // 2
            first_half_avg = sum(confidences[:mid]) / mid if mid > 0 else 0
            second_half_avg = sum(confidences[mid:]) / (len(confidences) - mid)

            diff = second_half_avg - first_half_avg
            if diff > 0.05:
                trend = "improving"
            elif diff < -0.05:
                trend = "declining"

        # Calculate quality score (0-100)
        # Weight: 50% average confidence, 30% consensus rate, 20% trend bonus
        trend_bonus = 10 if trend == "improving" else (-10 if trend == "declining" else 0)
        quality_score = min(
            100, max(0, int(average_confidence * 50 + consensus_rate * 30 + 20 + trend_bonus))
        )

        # Generate alert if quality is low
        alert = None
        if quality_score < 40:
            alert = {
                "level": "critical",
                "message": f"Consensus quality critically low ({quality_score}/100). Consider reviewing agent configurations.",
            }
        elif quality_score < 60:
            alert = {
                "level": "warning",
                "message": f"Consensus quality below target ({quality_score}/100). {trend.title()} trend detected.",
            }
        elif trend == "declining" and average_confidence < 0.7:
            alert = {
                "level": "info",
                "message": "Declining consensus trend detected. Monitor closely.",
            }

        return json_response(
            {
                "stats": {
                    "total_debates": total_debates,
                    "confidence_history": confidence_history[:20],  # Last 20 for UI
                    "trend": trend,
                    "average_confidence": round(average_confidence, 3),
                    "consensus_rate": round(consensus_rate, 3),
                    "consensus_reached_count": consensus_reached_count,
                },
                "quality_score": quality_score,
                "alert": alert,
            }
        )

    @ttl_cache(
        ttl_seconds=CACHE_TTL_ANALYTICS_RANKING, key_prefix="analytics_ranking", skip_first=True
    )
    @handle_errors("ranking stats retrieval")
    def _get_ranking_stats(self) -> HandlerResult:
        """Get ranking system statistics."""
        elo = self.get_elo_system()
        if not elo:
            return error_response("Ranking system not available", 503)

        leaderboard = elo.get_leaderboard(limit=100)

        stats = {
            "total_agents": len(leaderboard),
            "total_matches": sum(a.total_debates for a in leaderboard) if leaderboard else 0,
            "avg_elo": (
                sum(a.elo_rating for a in leaderboard) / len(leaderboard) if leaderboard else 1500
            ),
            "top_agent": leaderboard[0].agent_name if leaderboard else None,
            "elo_range": {
                "min": min(a.elo_rating for a in leaderboard) if leaderboard else 1500,
                "max": max(a.elo_rating for a in leaderboard) if leaderboard else 1500,
            },
        }

        return json_response({"stats": stats})

    @ttl_cache(
        ttl_seconds=CACHE_TTL_ANALYTICS_DEBATES, key_prefix="analytics_debates", skip_first=True
    )
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

    @ttl_cache(
        ttl_seconds=CACHE_TTL_ANALYTICS_MEMORY, key_prefix="analytics_memory", skip_first=True
    )
    @handle_errors("memory stats retrieval")
    def _get_memory_stats(self) -> HandlerResult:
        """Get memory system statistics."""
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
