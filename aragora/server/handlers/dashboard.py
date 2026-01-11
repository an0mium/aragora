"""
Debate Dashboard endpoint handler.

Provides a consolidated view of debate metrics for dashboard visualization.
Aggregates data from ELO, consensus, prometheus, and debate storage systems.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Optional

from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    get_int_param,
    ttl_cache,
)
from aragora.config import DB_TIMEOUT_SECONDS, CACHE_TTL_DASHBOARD_DEBATES

logger = logging.getLogger(__name__)


class DashboardHandler(BaseHandler):
    """Handler for dashboard endpoint."""

    ROUTES = ["/api/dashboard/debates", "/api/dashboard/quality-metrics"]

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
        elif path == "/api/dashboard/quality-metrics":
            return self._get_quality_metrics()
        return None

    @ttl_cache(ttl_seconds=CACHE_TTL_DASHBOARD_DEBATES, key_prefix="dashboard_debates", skip_first=True)
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
        request_start = time.perf_counter()
        logger.debug(
            "Dashboard request: domain=%s, limit=%d, hours=%d",
            domain, limit, hours
        )

        result: dict[str, Any] = {
            "summary": {},
            "recent_activity": {},
            "agent_performance": {},
            "debate_patterns": {},
            "consensus_insights": {},
            "system_health": {},
            "generated_at": time.time(),
        }

        # Use SQL aggregation for summary metrics (avoids loading 10K+ rows)
        storage = self.get_storage()
        if storage:
            result["summary"] = self._get_summary_metrics_sql(storage, domain)
            result["recent_activity"] = self._get_recent_activity_sql(storage, hours)
        else:
            result["summary"] = {
                "total_debates": 0,
                "consensus_reached": 0,
                "consensus_rate": 0.0,
                "avg_confidence": 0.0,
            }
            result["recent_activity"] = {
                "debates_last_period": 0,
                "consensus_last_period": 0,
                "period_hours": hours,
            }

        # Pattern metrics still require loading recent debates (limited set)
        result["debate_patterns"] = {"disagreement_stats": {}, "early_stopping": {}}

        # Gather agent performance
        result["agent_performance"] = self._get_agent_performance(limit)

        # Gather consensus insights
        result["consensus_insights"] = self._get_consensus_insights(domain)

        # Gather system health
        result["system_health"] = self._get_system_health()

        request_elapsed = time.perf_counter() - request_start
        summary = result.get("summary", {})
        total_debates = summary.get("total_debates", 0) if isinstance(summary, dict) else 0
        logger.debug(
            "Dashboard response: elapsed=%.3fs, total_debates=%d",
            request_elapsed, total_debates
        )
        return json_response(result)

    def _process_debates_single_pass(
        self, debates: list, domain: Optional[str], hours: int
    ) -> tuple[dict, dict, dict]:
        """Process all debate metrics in a single pass through the data.

        This optimization consolidates 3 separate loops into one, reducing
        iteration overhead for large debate lists.

        Args:
            debates: List of debate records
            domain: Optional domain filter
            hours: Time window for recent activity

        Returns:
            Tuple of (summary, activity, patterns) dicts
        """
        start_time = time.perf_counter()
        logger.debug(
            "Starting single-pass processing: debates=%d, domain=%s, hours=%d",
            len(debates), domain, hours
        )

        # Initialize summary metrics
        summary: dict[str, Any] = {
            "total_debates": 0,
            "consensus_reached": 0,
            "consensus_rate": 0.0,
            "avg_confidence": 0.0,
            "avg_rounds": 0.0,
            "total_tokens_used": 0,
        }

        # Initialize activity metrics
        activity: dict[str, Any] = {
            "debates_last_period": 0,
            "consensus_last_period": 0,
            "domains_active": [],
            "most_active_domain": None,
            "period_hours": hours,
        }

        # Initialize pattern metrics
        patterns: dict[str, dict[str, Any]] = {
            "disagreement_stats": {
                "with_disagreements": 0,
                "disagreement_types": {},
            },
            "early_stopping": {
                "early_stopped": 0,
                "full_duration": 0,
            },
        }

        if not debates:
            return summary, activity, patterns

        try:
            cutoff = datetime.now() - timedelta(hours=hours)

            # Accumulators for single-pass processing
            total = len(debates)
            consensus_count = 0
            confidences = []
            domain_counts: dict[str, int] = {}
            recent_count = 0
            recent_consensus = 0
            with_disagreement = 0
            disagreement_types: dict[str, int] = {}
            early_stopped = 0
            full_duration = 0

            for d in debates:
                # Summary metrics
                if d.get("consensus_reached"):
                    consensus_count += 1

                conf = d.get("confidence")
                if conf:
                    confidences.append(conf)

                # Activity metrics - check if recent
                created_at = d.get("created_at")
                if created_at:
                    try:
                        dt = datetime.fromisoformat(
                            created_at.replace("Z", "+00:00")
                        )
                        if dt.replace(tzinfo=None) > cutoff:
                            recent_count += 1
                            if d.get("consensus_reached"):
                                recent_consensus += 1
                            d_domain = d.get("domain", "general")
                            domain_counts[d_domain] = domain_counts.get(d_domain, 0) + 1
                    except (ValueError, KeyError) as e:
                        logger.debug(f"Skipping debate with invalid timestamp: {e}")

                # Pattern metrics
                if d.get("disagreement_report"):
                    with_disagreement += 1
                    report = d.get("disagreement_report", {})
                    for dt_type in report.get("types", []):
                        disagreement_types[dt_type] = disagreement_types.get(dt_type, 0) + 1

                if d.get("early_stopped"):
                    early_stopped += 1
                else:
                    full_duration += 1

            # Build summary
            summary["total_debates"] = total
            summary["consensus_reached"] = consensus_count
            if total > 0:
                summary["consensus_rate"] = round(consensus_count / total, 3)
            if confidences:
                summary["avg_confidence"] = round(
                    sum(confidences) / len(confidences), 3
                )

            # Build activity
            activity["debates_last_period"] = recent_count
            activity["consensus_last_period"] = recent_consensus
            activity["domains_active"] = list(domain_counts.keys())[:10]
            if domain_counts:
                activity["most_active_domain"] = max(
                    domain_counts, key=domain_counts.get
                )

            # Build patterns
            patterns["disagreement_stats"]["with_disagreements"] = with_disagreement
            patterns["disagreement_stats"]["disagreement_types"] = disagreement_types
            patterns["early_stopping"]["early_stopped"] = early_stopped
            patterns["early_stopping"]["full_duration"] = full_duration

        except Exception as e:
            logger.warning("Single-pass processing error: %s: %s", type(e).__name__, e)

        elapsed = time.perf_counter() - start_time
        logger.debug(
            "Completed single-pass processing: elapsed=%.3fs, total=%d, consensus=%d, recent=%d",
            elapsed, summary.get("total_debates", 0),
            summary.get("consensus_reached", 0),
            activity.get("debates_last_period", 0)
        )
        return summary, activity, patterns

    def _get_summary_metrics_sql(self, storage, domain: Optional[str]) -> dict:
        """Get summary metrics using SQL aggregation (O(1) memory)."""
        summary = {
            "total_debates": 0,
            "consensus_reached": 0,
            "consensus_rate": 0.0,
            "avg_confidence": 0.0,
        }

        try:
            with storage.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN consensus_reached THEN 1 ELSE 0 END) as consensus_count,
                        AVG(confidence) as avg_conf
                    FROM debates
                """)
                row = cursor.fetchone()
                if row:
                    total = row[0] or 0
                    consensus_count = row[1] or 0
                    avg_conf = row[2]

                    summary["total_debates"] = total
                    summary["consensus_reached"] = consensus_count
                    if total > 0:
                        summary["consensus_rate"] = round(consensus_count / total, 3)
                    if avg_conf is not None:
                        summary["avg_confidence"] = round(avg_conf, 3)
        except Exception as e:
            logger.warning("SQL summary metrics error: %s: %s", type(e).__name__, e)

        return summary

    def _get_recent_activity_sql(self, storage, hours: int) -> dict:
        """Get recent activity metrics using SQL aggregation."""
        activity = {
            "debates_last_period": 0,
            "consensus_last_period": 0,
            "period_hours": hours,
        }

        try:
            cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

            with storage.db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT
                        COUNT(*) as recent_total,
                        SUM(CASE WHEN consensus_reached THEN 1 ELSE 0 END) as recent_consensus
                    FROM debates
                    WHERE created_at >= ?
                """, (cutoff,))
                row = cursor.fetchone()
                if row:
                    activity["debates_last_period"] = row[0] or 0
                    activity["consensus_last_period"] = row[1] or 0
        except Exception as e:
            logger.warning("SQL recent activity error: %s: %s", type(e).__name__, e)

        return activity

    def _get_summary_metrics(self, domain: Optional[str], debates: list) -> dict:
        """Get high-level summary metrics (legacy, kept for compatibility)."""
        summary = {
            "total_debates": 0,
            "consensus_reached": 0,
            "consensus_rate": 0.0,
            "avg_confidence": 0.0,
            "avg_rounds": 0.0,
            "total_tokens_used": 0,
        }

        try:
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
            logger.warning("Summary metrics error: %s: %s", type(e).__name__, e)

        return summary

    def _get_recent_activity(self, domain: Optional[str], hours: int, debates: list) -> dict:
        """Get recent debate activity metrics."""
        activity = {
            "debates_last_period": 0,
            "consensus_last_period": 0,
            "domains_active": [],
            "most_active_domain": None,
            "period_hours": hours,
        }

        try:
            if debates:
                from datetime import datetime, timedelta

                cutoff = datetime.now() - timedelta(hours=hours)

                recent: list[dict] = []
                domain_counts: dict[str, int] = {}
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
                        except (ValueError, KeyError) as e:
                            logger.debug(f"Skipping debate with invalid datetime: {e}")

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
            logger.warning("Recent activity error: %s: %s", type(e).__name__, e)

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
                # Get all ratings in a single batch query (N+1 optimization)
                ratings = elo.get_all_ratings() if hasattr(elo, 'get_all_ratings') else []
                all_ratings = [
                    {
                        "name": rating.agent_name,
                        "elo": rating.elo,
                        "wins": rating.wins,
                        "losses": rating.losses,
                        "draws": rating.draws,
                        "win_rate": rating.win_rate,
                        "debates_count": rating.debates_count,
                    }
                    for rating in ratings
                ]
                # Already sorted by ELO descending from get_all_ratings()

                performance["top_performers"] = all_ratings[:limit]
                performance["total_agents"] = len(all_ratings)

                if all_ratings:
                    performance["avg_elo"] = round(
                        sum(r["elo"] for r in all_ratings) / len(all_ratings), 1
                    )
        except Exception as e:
            logger.warning("Agent performance error: %s: %s", type(e).__name__, e)

        return performance

    def _get_debate_patterns(self, debates: list) -> dict:
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
            if debates:
                # Analyze disagreements
                with_disagreement = 0
                disagreement_types: dict[str, int] = {}
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

                # Update patterns with computed stats
                disagree_stats = patterns["disagreement_stats"]
                if isinstance(disagree_stats, dict):
                    disagree_stats["with_disagreements"] = with_disagreement
                    disagree_stats["disagreement_types"] = disagreement_types
                early_stats = patterns["early_stopping"]
                if isinstance(early_stats, dict):
                    early_stats["early_stopped"] = early_stopped
                    early_stats["full_duration"] = full_duration
        except Exception as e:
            logger.warning("Debate patterns error: %s: %s", type(e).__name__, e)

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

            from aragora.storage.schema import get_wal_connection
            from aragora.config import DB_TIMEOUT_SECONDS

            memory = ConsensusMemory()
            stats = memory.get_statistics()

            insights["total_consensus_topics"] = stats.get("total_consensus", 0)
            insights["total_dissents"] = stats.get("total_dissents", 0)
            insights["domains"] = list(stats.get("by_domain", {}).keys())

            # Get high confidence count from DB
            with get_wal_connection(memory.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM consensus WHERE confidence >= 0.7"
                )
                row = cursor.fetchone()
                insights["high_confidence_count"] = row[0] if row else 0

                cursor.execute("SELECT AVG(confidence) FROM consensus")
                row = cursor.fetchone()
                avg = row[0] if row else None
                insights["avg_confidence"] = round(avg, 3) if avg else 0.0

        except ImportError:
            logger.debug("Consensus memory not available")
        except Exception as e:
            logger.warning("Consensus insights error: %s: %s", type(e).__name__, e)

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
            logger.warning("System health error: %s: %s", type(e).__name__, e)

        return health

    @ttl_cache(ttl_seconds=60, key_prefix="quality_metrics", skip_first=True)
    def _get_quality_metrics(self) -> HandlerResult:
        """Get unified quality metrics across all subsystems.

        Aggregates:
        - Agent calibration trends (over/underconfidence)
        - Performance metrics (latency, success rate)
        - Evolution progress (prompt versions)
        - Debate quality scores

        Returns:
            Consolidated quality metrics from available subsystems
        """
        result: dict[str, Any] = {
            "calibration": {},
            "performance": {},
            "evolution": {},
            "debate_quality": {},
            "generated_at": time.time(),
        }

        # Calibration metrics
        result["calibration"] = self._get_calibration_metrics()

        # Performance metrics
        result["performance"] = self._get_performance_metrics()

        # Evolution metrics
        result["evolution"] = self._get_evolution_metrics()

        # Debate quality metrics
        result["debate_quality"] = self._get_debate_quality_metrics()

        return json_response(result)

    def _get_calibration_metrics(self) -> dict:
        """Get comprehensive agent calibration metrics.

        Returns:
            Dict with calibration data including:
            - agents: Per-agent calibration summaries
            - overall_calibration: Average calibration score
            - overconfident_agents: Agents with high overconfidence bias
            - underconfident_agents: Agents with underconfidence bias
            - well_calibrated_agents: Agents with good calibration (|bias| < 0.1)
            - top_by_brier: Top 5 agents ranked by Brier score (lower is better)
            - calibration_curves: Bucket data for visualization (top 3 agents)
            - domain_breakdown: Per-domain stats for each agent
        """
        metrics: dict[str, Any] = {
            "agents": {},
            "overall_calibration": 0.0,
            "overconfident_agents": [],
            "underconfident_agents": [],
            "well_calibrated_agents": [],
            "top_by_brier": [],
            "calibration_curves": {},
            "domain_breakdown": {},
        }

        try:
            calibration_tracker = self._context.get("calibration_tracker")
            if not calibration_tracker:
                return metrics

            summary = calibration_tracker.get_calibration_summary()
            if summary:
                metrics["agents"] = summary.get("agents", {})
                metrics["overall_calibration"] = summary.get("overall", 0.0)

                # Categorize agents by calibration bias
                agent_brier_scores: list[tuple[str, float]] = []

                for agent, data in metrics["agents"].items():
                    bias = data.get("calibration_bias", 0)
                    brier = data.get("brier_score", 1.0)
                    agent_brier_scores.append((agent, brier))

                    if bias > 0.1:
                        metrics["overconfident_agents"].append(agent)
                    elif bias < -0.1:
                        metrics["underconfident_agents"].append(agent)
                    else:
                        metrics["well_calibrated_agents"].append(agent)

                # Get top agents by Brier score (lower is better)
                agent_brier_scores.sort(key=lambda x: x[1])
                metrics["top_by_brier"] = [
                    {"agent": agent, "brier_score": round(brier, 3)}
                    for agent, brier in agent_brier_scores[:5]
                ]

            # Get calibration curves for top 3 agents
            all_agents = calibration_tracker.get_all_agents()
            for agent in all_agents[:3]:
                try:
                    curve = calibration_tracker.get_calibration_curve(agent, num_buckets=10)
                    if curve:
                        metrics["calibration_curves"][agent] = [
                            {
                                "bucket": i,
                                "confidence_range": f"{bucket.range_start:.1f}-{bucket.range_end:.1f}",
                                "expected_accuracy": bucket.expected_accuracy,
                                "actual_accuracy": bucket.accuracy,
                                "count": bucket.total_predictions,
                            }
                            for i, bucket in enumerate(curve)
                        ]
                except Exception as e:
                    logger.debug(f"Calibration curve error for {agent}: {e}")

            # Get domain breakdown for agents with sufficient data
            for agent in all_agents[:5]:
                try:
                    domain_data = calibration_tracker.get_domain_breakdown(agent)
                    if domain_data:
                        metrics["domain_breakdown"][agent] = {
                            domain: {
                                "predictions": s.total_predictions,
                                "accuracy": round(s.accuracy, 3),
                                "brier_score": round(s.brier_score, 3),
                                "ece": round(s.ece, 3) if hasattr(s, 'ece') else None,
                            }
                            for domain, s in domain_data.items()
                        }
                except Exception as e:
                    logger.debug(f"Domain breakdown error for {agent}: {e}")

        except Exception as e:
            logger.warning("Calibration metrics error: %s", e)

        return metrics

    def _get_performance_metrics(self) -> dict:
        """Get agent performance metrics."""
        metrics: dict[str, Any] = {
            "agents": {},
            "avg_latency_ms": 0.0,
            "success_rate": 0.0,
            "total_calls": 0,
        }

        try:
            performance_monitor = self._context.get("performance_monitor")
            if performance_monitor:
                insights = performance_monitor.get_performance_insights()
                if insights:
                    metrics["agents"] = insights.get("agents", {})
                    metrics["avg_latency_ms"] = insights.get("avg_latency_ms", 0.0)
                    metrics["success_rate"] = insights.get("success_rate", 0.0)
                    metrics["total_calls"] = insights.get("total_calls", 0)
        except Exception as e:
            logger.warning("Performance metrics error: %s", e)

        return metrics

    def _get_evolution_metrics(self) -> dict:
        """Get prompt evolution progress."""
        metrics: dict[str, Any] = {
            "agents": {},
            "total_versions": 0,
            "patterns_extracted": 0,
            "last_evolution": None,
        }

        try:
            prompt_evolver = self._context.get("prompt_evolver")
            if prompt_evolver:
                # Get version counts per agent
                for agent_name in ["claude", "gemini", "codex", "grok"]:
                    try:
                        version = prompt_evolver.get_prompt_version(agent_name)
                        if version:
                            metrics["agents"][agent_name] = {
                                "current_version": version.version,
                                "performance_score": version.performance_score,
                                "debates_count": version.debates_count,
                            }
                            metrics["total_versions"] += version.version
                    except (AttributeError, KeyError) as e:
                        logger.debug(f"Skipping agent version with missing data: {e}")

                # Get pattern count
                patterns = prompt_evolver.get_top_patterns(limit=100)
                metrics["patterns_extracted"] = len(patterns) if patterns else 0
        except Exception as e:
            logger.warning("Evolution metrics error: %s", e)

        return metrics

    def _get_debate_quality_metrics(self) -> dict:
        """Get debate quality scores."""
        metrics: dict[str, Any] = {
            "avg_confidence": 0.0,
            "consensus_rate": 0.0,
            "avg_rounds": 0.0,
            "evidence_quality": 0.0,
            "recent_winners": [],
        }

        try:
            # Get from ELO system
            elo_system = self._context.get("elo_system")
            if elo_system:
                recent = elo_system.get_recent_matches(limit=10)
                if recent:
                    winners = [m.get("winner") for m in recent if m.get("winner")]
                    metrics["recent_winners"] = winners[:5]

                    # Calculate avg confidence from matches
                    confidences = [m.get("confidence", 0) for m in recent if m.get("confidence")]
                    if confidences:
                        metrics["avg_confidence"] = sum(confidences) / len(confidences)

            # Get from storage
            storage = self.get_storage()
            if storage:
                summary = self._get_summary_metrics_sql(storage, None)
                if summary:
                    metrics["consensus_rate"] = summary.get("consensus_rate", 0.0)
                    metrics["avg_rounds"] = summary.get("avg_rounds", 0.0)

        except Exception as e:
            logger.warning("Debate quality metrics error: %s", e)

        return metrics
