"""
Debate Dashboard endpoint handler.

Provides a consolidated view of debate metrics for dashboard visualization.
Aggregates data from ELO, consensus, prometheus, and debate storage systems.
"""

import logging
import time
from typing import Optional

from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    get_int_param,
    ttl_cache,
)

logger = logging.getLogger(__name__)


class DashboardHandler(BaseHandler):
    """Handler for dashboard endpoint."""

    ROUTES = ["/api/dashboard/debates"]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route dashboard requests to appropriate methods."""
        if path == "/api/dashboard/debates":
            domain = query_params.get("domain")
            limit = get_int_param(query_params, "limit", 10)
            hours = get_int_param(query_params, "hours", 24)
            return self._get_debates_dashboard(domain, min(limit, 50), hours)
        return None

    @ttl_cache(ttl_seconds=600, key_prefix="dashboard_debates", skip_first=True)
    def _get_debates_dashboard(
        self, domain: Optional[str], limit: int, hours: int
    ) -> HandlerResult:
        """Get consolidated debate metrics for dashboard.

        Args:
            domain: Optional domain filter
            limit: Max items per list section
            hours: Time window for recent activity

        Returns:
            Aggregated dashboard metrics from all available subsystems
        """
        result = {
            "summary": {},
            "recent_activity": {},
            "agent_performance": {},
            "debate_patterns": {},
            "consensus_insights": {},
            "system_health": {},
            "generated_at": time.time(),
        }

        # Gather summary metrics
        result["summary"] = self._get_summary_metrics(domain)

        # Gather recent activity
        result["recent_activity"] = self._get_recent_activity(domain, hours)

        # Gather agent performance
        result["agent_performance"] = self._get_agent_performance(limit)

        # Gather debate patterns
        result["debate_patterns"] = self._get_debate_patterns()

        # Gather consensus insights
        result["consensus_insights"] = self._get_consensus_insights(domain)

        # Gather system health
        result["system_health"] = self._get_system_health()

        return json_response(result)

    def _get_summary_metrics(self, domain: Optional[str]) -> dict:
        """Get high-level summary metrics."""
        summary = {
            "total_debates": 0,
            "consensus_reached": 0,
            "consensus_rate": 0.0,
            "avg_confidence": 0.0,
            "avg_rounds": 0.0,
            "total_tokens_used": 0,
        }

        try:
            # Get debate counts from storage
            storage = self.get_storage()
            if storage:
                debates = storage.list_debates(limit=10000)
                if debates:
                    total = len(debates)
                    consensus_count = sum(
                        1 for d in debates if d.get("consensus_reached")
                    )
                    summary["total_debates"] = total
                    summary["consensus_reached"] = consensus_count
                    if total > 0:
                        summary["consensus_rate"] = round(consensus_count / total, 3)

                        # Average confidence
                        confidences = [
                            d.get("confidence", 0.5)
                            for d in debates
                            if d.get("confidence")
                        ]
                        if confidences:
                            summary["avg_confidence"] = round(
                                sum(confidences) / len(confidences), 3
                            )
        except Exception as e:
            logger.debug(f"Summary metrics error: {e}")

        return summary

    def _get_recent_activity(self, domain: Optional[str], hours: int) -> dict:
        """Get recent debate activity metrics."""
        activity = {
            "debates_last_period": 0,
            "consensus_last_period": 0,
            "domains_active": [],
            "most_active_domain": None,
            "period_hours": hours,
        }

        try:
            storage = self.get_storage()
            if storage:
                # Get recent debates
                from datetime import datetime, timedelta

                cutoff = datetime.now() - timedelta(hours=hours)
                debates = storage.list_debates(limit=1000)

                recent = []
                domain_counts = {}
                for d in debates:
                    created_at = d.get("created_at")
                    if created_at:
                        # Parse ISO timestamp
                        try:
                            dt = datetime.fromisoformat(
                                created_at.replace("Z", "+00:00")
                            )
                            if dt.replace(tzinfo=None) > cutoff:
                                recent.append(d)
                                d_domain = d.get("domain", "general")
                                domain_counts[d_domain] = domain_counts.get(d_domain, 0) + 1
                        except Exception:
                            pass

                activity["debates_last_period"] = len(recent)
                activity["consensus_last_period"] = sum(
                    1 for d in recent if d.get("consensus_reached")
                )
                activity["domains_active"] = list(domain_counts.keys())[:10]

                if domain_counts:
                    activity["most_active_domain"] = max(
                        domain_counts, key=domain_counts.get
                    )
        except Exception as e:
            logger.debug(f"Recent activity error: {e}")

        return activity

    def _get_agent_performance(self, limit: int) -> dict:
        """Get agent performance metrics."""
        performance = {
            "top_performers": [],
            "total_agents": 0,
            "avg_elo": 0,
        }

        try:
            elo = self.get_elo_system()
            if elo:
                # Get all ratings
                all_ratings = []
                for name in elo.list_agents():
                    rating = elo.get_rating(name)
                    if rating:
                        all_ratings.append(
                            {
                                "name": name,
                                "elo": rating.elo,
                                "wins": rating.wins,
                                "losses": rating.losses,
                                "draws": rating.draws,
                                "win_rate": rating.win_rate,
                                "debates_count": rating.debates_count,
                            }
                        )

                # Sort by ELO
                all_ratings.sort(key=lambda x: x["elo"], reverse=True)

                performance["top_performers"] = all_ratings[:limit]
                performance["total_agents"] = len(all_ratings)

                if all_ratings:
                    performance["avg_elo"] = round(
                        sum(r["elo"] for r in all_ratings) / len(all_ratings), 1
                    )
        except Exception as e:
            logger.debug(f"Agent performance error: {e}")

        return performance

    def _get_debate_patterns(self) -> dict:
        """Get debate pattern statistics."""
        patterns = {
            "disagreement_stats": {
                "with_disagreements": 0,
                "disagreement_types": {},
            },
            "early_stopping": {
                "early_stopped": 0,
                "full_duration": 0,
            },
        }

        try:
            storage = self.get_storage()
            if storage:
                debates = storage.list_debates(limit=1000)

                # Analyze disagreements
                with_disagreement = 0
                disagreement_types = {}
                early_stopped = 0
                full_duration = 0

                for d in debates:
                    # Check for disagreement reports
                    if d.get("disagreement_report"):
                        with_disagreement += 1
                        report = d.get("disagreement_report", {})
                        for dt in report.get("types", []):
                            disagreement_types[dt] = disagreement_types.get(dt, 0) + 1

                    # Check for early stopping
                    if d.get("early_stopped"):
                        early_stopped += 1
                    else:
                        full_duration += 1

                patterns["disagreement_stats"]["with_disagreements"] = with_disagreement
                patterns["disagreement_stats"]["disagreement_types"] = disagreement_types
                patterns["early_stopping"]["early_stopped"] = early_stopped
                patterns["early_stopping"]["full_duration"] = full_duration
        except Exception as e:
            logger.debug(f"Debate patterns error: {e}")

        return patterns

    def _get_consensus_insights(self, domain: Optional[str]) -> dict:
        """Get consensus memory insights."""
        insights = {
            "total_consensus_topics": 0,
            "high_confidence_count": 0,
            "avg_confidence": 0.0,
            "total_dissents": 0,
            "domains": [],
        }

        try:
            from aragora.memory.consensus import ConsensusMemory

            memory = ConsensusMemory()
            stats = memory.get_statistics()

            insights["total_consensus_topics"] = stats.get("total_consensus", 0)
            insights["total_dissents"] = stats.get("total_dissents", 0)
            insights["domains"] = list(stats.get("by_domain", {}).keys())

            # Get high confidence count from DB
            import sqlite3

            with sqlite3.connect(memory.db_path, timeout=30.0) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM consensus WHERE confidence >= 0.7"
                )
                insights["high_confidence_count"] = cursor.fetchone()[0]

                cursor.execute("SELECT AVG(confidence) FROM consensus")
                avg = cursor.fetchone()[0]
                insights["avg_confidence"] = round(avg, 3) if avg else 0.0

        except ImportError:
            logger.debug("Consensus memory not available")
        except Exception as e:
            logger.debug(f"Consensus insights error: {e}")

        return insights

    def _get_system_health(self) -> dict:
        """Get system health metrics."""
        health = {
            "uptime_seconds": 0,
            "cache_entries": 0,
            "active_websocket_connections": 0,
            "prometheus_available": False,
        }

        try:
            from aragora.server.prometheus import (
                is_prometheus_available,
                get_metrics_output,
            )

            health["prometheus_available"] = is_prometheus_available()

            # Get cache stats if available
            from .base import _cache

            if _cache:
                health["cache_entries"] = len(_cache)

        except Exception as e:
            logger.debug(f"System health error: {e}")

        return health
