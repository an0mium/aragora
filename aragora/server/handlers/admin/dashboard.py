"""
Debate Dashboard endpoint handler.

Provides a consolidated view of debate metrics for dashboard visualization.
Aggregates data from ELO, consensus, prometheus, and debate storage systems.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from aragora.config import CACHE_TTL_DASHBOARD_DEBATES

from ..base import (
    HandlerResult,
    error_response,
    get_int_param,
    json_response,
    ttl_cache,
)
from ..secure import SecureHandler, ForbiddenError, UnauthorizedError
from ..utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiter for dashboard endpoints (60 requests per minute - frequently accessed)
_dashboard_limiter = RateLimiter(requests_per_minute=60)

# Permission required for dashboard access
DASHBOARD_PERMISSION = "dashboard.read"


class DashboardHandler(SecureHandler):
    """Handler for dashboard endpoint.

    Requires authentication and dashboard.read permission (RBAC).
    """

    ROUTES = [
        "/api/dashboard/debates",
        "/api/v1/dashboard",
        "/api/v1/dashboard/overview",
        "/api/v1/dashboard/debates",
        "/api/v1/dashboard/stats",
        "/api/v1/dashboard/stat-cards",
        "/api/v1/dashboard/team-performance",
        "/api/v1/dashboard/top-senders",
        "/api/v1/dashboard/labels",
        "/api/v1/dashboard/activity",
        "/api/v1/dashboard/inbox-summary",
        "/api/v1/dashboard/quick-actions",
        "/api/v1/dashboard/urgent",
        "/api/v1/dashboard/pending-actions",
        "/api/v1/dashboard/search",
        "/api/v1/dashboard/export",
        "/api/v1/dashboard/quality-metrics",
    ]
    ROUTE_PREFIXES = ["/api/v1/dashboard/"]
    RESOURCE_TYPE = "dashboard"

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path.startswith("/api/v1/dashboard/gastown/"):
            return False
        if path in self.ROUTES:
            return True
        if path.startswith("/api/v1/dashboard/debates/"):
            return len(path.split("/")) == 6
        if path.startswith("/api/v1/dashboard/team-performance/"):
            return len(path.split("/")) == 6
        if path.startswith("/api/v1/dashboard/quick-actions/"):
            return len(path.split("/")) == 6
        if path.startswith("/api/v1/dashboard/urgent/") and path.endswith("/dismiss"):
            return len(path.split("/")) == 7
        if path.startswith("/api/v1/dashboard/pending-actions/") and path.endswith("/complete"):
            return len(path.split("/")) == 7
        return False

    async def handle(  # type: ignore[override]
        self, path: str, query_params: dict, handler
    ) -> Optional[HandlerResult]:
        """Route dashboard requests to appropriate methods with RBAC."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _dashboard_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for dashboard endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # RBAC: Require authentication and dashboard.read permission
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
            self.check_permission(auth_context, DASHBOARD_PERMISSION)
        except UnauthorizedError as e:
            logger.warning(f"Dashboard auth error: {e}")
            return error_response("Authentication required", 401)
        except ForbiddenError as e:
            logger.warning(f"Dashboard access denied: {e}")
            return error_response(str(e), 403)

        if path == "/api/dashboard/debates":
            domain = query_params.get("domain")
            limit = get_int_param(query_params, "limit", 10)
            hours = get_int_param(query_params, "hours", 24)
            return self._get_debates_dashboard(domain, min(limit, 50), hours)

        if path in ("/api/v1/dashboard", "/api/v1/dashboard/overview"):
            return self._get_overview(query_params, handler)

        if path == "/api/v1/dashboard/debates":
            limit = get_int_param(query_params, "limit", 10)
            offset = get_int_param(query_params, "offset", 0)
            status = query_params.get("status")
            return self._get_dashboard_debates(min(limit, 50), max(offset, 0), status)

        if path.startswith("/api/v1/dashboard/debates/"):
            parts = path.split("/")
            if len(parts) == 6:
                return self._get_dashboard_debate(parts[5])

        if path == "/api/v1/dashboard/stats":
            return self._get_dashboard_stats()

        if path == "/api/v1/dashboard/stat-cards":
            return self._get_stat_cards()

        if path == "/api/v1/dashboard/team-performance":
            limit = get_int_param(query_params, "limit", 10)
            offset = get_int_param(query_params, "offset", 0)
            return self._get_team_performance(min(limit, 50), max(offset, 0))

        if path.startswith("/api/v1/dashboard/team-performance/"):
            parts = path.split("/")
            if len(parts) == 6:
                return self._get_team_performance_detail(parts[5])

        if path == "/api/v1/dashboard/top-senders":
            limit = get_int_param(query_params, "limit", 10)
            offset = get_int_param(query_params, "offset", 0)
            return self._get_top_senders(min(limit, 50), max(offset, 0))

        if path == "/api/v1/dashboard/labels":
            return self._get_labels()

        if path == "/api/v1/dashboard/activity":
            limit = get_int_param(query_params, "limit", 20)
            offset = get_int_param(query_params, "offset", 0)
            return self._get_activity(min(limit, 100), max(offset, 0))

        if path == "/api/v1/dashboard/inbox-summary":
            return self._get_inbox_summary()

        if path == "/api/v1/dashboard/quick-actions":
            return self._get_quick_actions()

        if path == "/api/v1/dashboard/urgent":
            limit = get_int_param(query_params, "limit", 20)
            offset = get_int_param(query_params, "offset", 0)
            return self._get_urgent_items(min(limit, 100), max(offset, 0))

        if path == "/api/v1/dashboard/pending-actions":
            limit = get_int_param(query_params, "limit", 20)
            offset = get_int_param(query_params, "offset", 0)
            return self._get_pending_actions(min(limit, 100), max(offset, 0))

        if path == "/api/v1/dashboard/search":
            query = query_params.get("q") or ""
            return self._search_dashboard(query)

        if path == "/api/v1/dashboard/quality-metrics":
            return self._get_quality_metrics()

        return None

    async def handle_post(  # type: ignore[override]
        self, path: str, body: dict, handler
    ) -> Optional[HandlerResult]:
        """Handle dashboard write actions (stub)."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _dashboard_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for dashboard endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # RBAC: Require authentication and dashboard.read permission
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
            self.check_permission(auth_context, DASHBOARD_PERMISSION)
        except UnauthorizedError as e:
            logger.warning(f"Dashboard auth error: {e}")
            return error_response("Authentication required", 401)
        except ForbiddenError as e:
            logger.warning(f"Dashboard access denied: {e}")
            return error_response(str(e), 403)

        if path.startswith("/api/v1/dashboard/quick-actions/"):
            parts = path.split("/")
            if len(parts) == 6:
                return self._execute_quick_action(parts[5])

        if path.startswith("/api/v1/dashboard/urgent/") and path.endswith("/dismiss"):
            parts = path.split("/")
            if len(parts) == 7:
                return self._dismiss_urgent_item(parts[5])

        if path.startswith("/api/v1/dashboard/pending-actions/") and path.endswith("/complete"):
            parts = path.split("/")
            if len(parts) == 7:
                return self._complete_pending_action(parts[5])

        if path == "/api/v1/dashboard/export":
            return self._export_dashboard_data()

        return None

    @ttl_cache(
        ttl_seconds=CACHE_TTL_DASHBOARD_DEBATES, key_prefix="dashboard_debates", skip_first=True
    )
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
        logger.debug("Dashboard request: domain=%s, limit=%d, hours=%d", domain, limit, hours)

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
            "Dashboard response: elapsed=%.3fs, total_debates=%d", request_elapsed, total_debates
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
            len(debates),
            domain,
            hours,
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
                        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
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
                summary["avg_confidence"] = round(sum(confidences) / len(confidences), 3)

            # Build activity
            activity["debates_last_period"] = recent_count
            activity["consensus_last_period"] = recent_consensus
            activity["domains_active"] = list(domain_counts.keys())[:10]
            if domain_counts:
                activity["most_active_domain"] = max(domain_counts, key=domain_counts.get)

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
            elapsed,
            summary.get("total_debates", 0),
            summary.get("consensus_reached", 0),
            activity.get("debates_last_period", 0),
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
                cursor.execute(
                    """
                    SELECT
                        COUNT(*) as recent_total,
                        SUM(CASE WHEN consensus_reached THEN 1 ELSE 0 END) as recent_consensus
                    FROM debates
                    WHERE created_at >= ?
                """,
                    (cutoff,),
                )
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
                consensus_count = sum(1 for d in debates if d.get("consensus_reached"))
                summary["total_debates"] = total
                summary["consensus_reached"] = consensus_count
                if total > 0:
                    summary["consensus_rate"] = round(consensus_count / total, 3)

                    # Average confidence
                    confidences = [d.get("confidence", 0.5) for d in debates if d.get("confidence")]
                    if confidences:
                        summary["avg_confidence"] = round(sum(confidences) / len(confidences), 3)
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
                            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
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
                    activity["most_active_domain"] = max(domain_counts, key=domain_counts.get)
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
                ratings = elo.get_all_ratings() if hasattr(elo, "get_all_ratings") else []
                all_ratings: list[dict[str, Any]] = [
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
            from aragora.config import DB_TIMEOUT_SECONDS
            from aragora.memory.consensus import ConsensusMemory
            from aragora.storage.schema import get_wal_connection

            memory = ConsensusMemory()
            stats = memory.get_statistics()

            insights["total_consensus_topics"] = stats.get("total_consensus", 0)
            insights["total_dissents"] = stats.get("total_dissents", 0)
            insights["domains"] = list(stats.get("by_domain", {}).keys())

            # Get high confidence count from DB
            with get_wal_connection(memory.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM consensus WHERE confidence >= 0.7")
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
        health: dict[str, Any] = {
            "uptime_seconds": 0,
            "cache_entries": 0,
            "active_websocket_connections": 0,
            "prometheus_available": False,
        }

        try:
            from aragora.server.prometheus import (
                is_prometheus_available,
            )

            health["prometheus_available"] = is_prometheus_available()

            # Get cache stats if available
            from ..base import _cache

            if _cache:
                health["cache_entries"] = len(_cache)

        except Exception as e:
            logger.warning("System health error: %s: %s", type(e).__name__, e)

        # Add connector health
        health["connector_health"] = self._get_connector_health()

        return health

    def _get_connector_health(self) -> dict:
        """Get connector health metrics for dashboard.

        Returns aggregated health from the sync scheduler including:
        - Summary stats (total, healthy, degraded, unhealthy, health_score)
        - Per-connector breakdown with status, sync metrics, and errors
        """
        result: dict[str, Any] = {
            "summary": {
                "total_connectors": 0,
                "healthy": 0,
                "degraded": 0,
                "unhealthy": 0,
                "health_score": 100,
                "scheduler_running": False,
                "running_syncs": 0,
                "success_rate": 1.0,
            },
            "connectors": [],
        }

        try:
            from aragora.server.handlers.connectors import get_scheduler

            scheduler = get_scheduler()
            stats = scheduler.get_stats()

            # Build summary from scheduler stats
            result["summary"]["total_connectors"] = stats.get("total_jobs", 0)
            result["summary"]["scheduler_running"] = scheduler._scheduler_task is not None
            result["summary"]["running_syncs"] = stats.get("running_syncs", 0)
            result["summary"]["success_rate"] = stats.get("success_rate", 1.0)

            # Build per-connector breakdown from jobs
            jobs = scheduler.list_jobs()
            healthy = degraded = unhealthy = 0

            for job in jobs:
                # Determine health: 3+ failures = unhealthy, 1-2 = degraded
                if job.consecutive_failures >= 3:
                    health = "unhealthy"
                    unhealthy += 1
                elif job.consecutive_failures >= 1:
                    health = "degraded"
                    degraded += 1
                else:
                    health = "healthy"
                    healthy += 1

                # Determine status
                if job.current_run_id:
                    status = "syncing"
                elif job.consecutive_failures >= 3:
                    status = "error"
                elif not job.schedule.enabled:
                    status = "disconnected"
                else:
                    status = "connected"

                # Calculate metrics from history
                history = scheduler.get_history(job_id=job.id, limit=100)
                total_syncs = len(history)
                failed = sum(1 for h in history if h.status.value == "failed")
                error_rate = (failed / total_syncs * 100) if total_syncs > 0 else 0.0
                avg_duration = (
                    sum(h.duration_seconds or 0 for h in history) / total_syncs
                    if total_syncs > 0
                    else 0.0
                )
                total_items = sum(h.items_synced for h in history)

                connector_name = job.connector_id
                connector_type = "unknown"
                if job.connector:
                    connector_name = getattr(job.connector, "name", job.connector_id)
                    connector_type = self._get_connector_type(job.connector)

                result["connectors"].append(
                    {
                        "connector_id": job.connector_id,
                        "connector_name": connector_name,
                        "connector_type": connector_type,
                        "status": status,
                        "health": health,
                        "uptime": round(100 - error_rate, 1),
                        "error_rate": round(error_rate, 1),
                        "last_sync": job.last_run.isoformat() if job.last_run else None,
                        "next_sync": job.next_run.isoformat() if job.next_run else None,
                        "items_synced": total_items,
                        "avg_sync_duration": round(avg_duration, 1),
                        "consecutive_failures": job.consecutive_failures,
                    }
                )

            result["summary"]["healthy"] = healthy
            result["summary"]["degraded"] = degraded
            result["summary"]["unhealthy"] = unhealthy
            total = healthy + degraded + unhealthy
            result["summary"]["health_score"] = round((healthy / total) * 100) if total > 0 else 100

        except ImportError:
            logger.debug("Connector scheduler not available")
        except Exception as e:
            logger.warning("Connector health error: %s: %s", type(e).__name__, e)

        return result

    def _get_connector_type(self, connector: Any) -> str:
        """Extract connector type from connector instance."""
        if not connector:
            return "unknown"
        class_name = type(connector).__name__.lower()
        type_mapping = {
            "githubenterpriseconnector": "github",
            "s3connector": "s3",
            "postgresqlconnector": "postgresql",
            "mongodbconnector": "mongodb",
            "fhirconnector": "fhir",
        }
        return type_mapping.get(class_name, class_name.replace("connector", ""))

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

    def _get_overview(self, query_params: dict, handler: Any) -> HandlerResult:
        """Return dashboard overview summary."""
        now = datetime.now(timezone.utc).isoformat()
        overview: dict[str, Any] = {
            "stats": [],
            "recent_debates": [],
            "active_debates": 0,
            "total_debates_today": 0,
            "consensus_rate": 0.0,
            "avg_debate_duration_ms": 0,
            "system_health": "healthy",
            "last_updated": now,
        }

        try:
            storage = self.get_storage()
            if storage:
                summary = self._get_summary_metrics_sql(storage, None)
                overview["consensus_rate"] = summary.get("consensus_rate", 0.0)

                # Count today's debates
                today_start = (
                    datetime.now(timezone.utc)
                    .replace(hour=0, minute=0, second=0, microsecond=0)
                    .isoformat()
                )
                try:
                    with storage.db.connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT COUNT(*) FROM debates WHERE created_at >= ?",
                            (today_start,),
                        )
                        row = cursor.fetchone()
                        overview["total_debates_today"] = row[0] if row else 0
                except Exception as e:
                    logger.debug(f"Could not get today's debates count: {e}")

            # Agent performance as stat cards
            perf = self._get_agent_performance(5)
            overview["stats"] = [
                {"label": "Total Agents", "value": perf.get("total_agents", 0)},
                {"label": "Avg ELO", "value": perf.get("avg_elo", 0)},
            ]
        except Exception as e:
            logger.warning("Overview error: %s: %s", type(e).__name__, e)

        return json_response(overview)

    def _get_dashboard_debates(self, limit: int, offset: int, status: Any) -> HandlerResult:
        """Return dashboard debate list from storage."""
        debates: list[dict[str, Any]] = []
        total = 0

        try:
            storage = self.get_storage()
            if storage:
                with storage.db.connection() as conn:
                    cursor = conn.cursor()
                    # Count total
                    if status:
                        cursor.execute("SELECT COUNT(*) FROM debates WHERE status = ?", (status,))
                    else:
                        cursor.execute("SELECT COUNT(*) FROM debates")
                    row = cursor.fetchone()
                    total = row[0] if row else 0

                    # Fetch page
                    if status:
                        cursor.execute(
                            "SELECT id, domain, status, consensus_reached, confidence, "
                            "created_at FROM debates WHERE status = ? "
                            "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                            (status, limit, offset),
                        )
                    else:
                        cursor.execute(
                            "SELECT id, domain, status, consensus_reached, confidence, "
                            "created_at FROM debates "
                            "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                            (limit, offset),
                        )

                    for row in cursor.fetchall():
                        debates.append(
                            {
                                "id": row[0],
                                "domain": row[1],
                                "status": row[2],
                                "consensus_reached": bool(row[3]),
                                "confidence": row[4],
                                "created_at": row[5],
                            }
                        )
        except Exception as e:
            logger.warning("Dashboard debates error: %s: %s", type(e).__name__, e)

        return json_response({"debates": debates, "total": total})

    def _get_dashboard_debate(self, debate_id: str) -> HandlerResult:
        """Return a single debate summary entry."""
        if not debate_id:
            return error_response("debate_id is required", 400)
        return json_response({"debate_id": debate_id})

    def _get_dashboard_stats(self) -> HandlerResult:
        """Return dashboard statistics aggregated from storage and ELO."""
        stats: dict[str, Any] = {
            "debates": {
                "total": 0,
                "today": 0,
                "this_week": 0,
                "this_month": 0,
                "by_status": {},
            },
            "agents": {"total": 0, "active": 0, "by_provider": {}},
            "performance": {
                "avg_response_time_ms": 0,
                "success_rate": 0.0,
                "consensus_rate": 0.0,
                "error_rate": 0.0,
            },
            "usage": {
                "api_calls_today": 0,
                "tokens_used_today": 0,
                "storage_used_bytes": 0,
            },
        }

        try:
            storage = self.get_storage()
            if storage:
                summary = self._get_summary_metrics_sql(storage, None)
                stats["debates"]["total"] = summary.get("total_debates", 0)
                stats["performance"]["consensus_rate"] = summary.get("consensus_rate", 0.0)

                now = datetime.now(timezone.utc)
                today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
                week_start = (now - timedelta(days=7)).isoformat()
                month_start = (now - timedelta(days=30)).isoformat()

                try:
                    with storage.db.connection() as conn:
                        cursor = conn.cursor()
                        # Today
                        cursor.execute(
                            "SELECT COUNT(*) FROM debates WHERE created_at >= ?",
                            (today_start,),
                        )
                        row = cursor.fetchone()
                        stats["debates"]["today"] = row[0] if row else 0

                        # This week
                        cursor.execute(
                            "SELECT COUNT(*) FROM debates WHERE created_at >= ?",
                            (week_start,),
                        )
                        row = cursor.fetchone()
                        stats["debates"]["this_week"] = row[0] if row else 0

                        # This month
                        cursor.execute(
                            "SELECT COUNT(*) FROM debates WHERE created_at >= ?",
                            (month_start,),
                        )
                        row = cursor.fetchone()
                        stats["debates"]["this_month"] = row[0] if row else 0

                        # By status
                        cursor.execute("SELECT status, COUNT(*) FROM debates GROUP BY status")
                        for row in cursor.fetchall():
                            if row[0]:
                                stats["debates"]["by_status"][row[0]] = row[1]
                except Exception as e:
                    logger.debug(f"Could not get debate stats: {e}")

            # Agent stats from ELO
            perf = self._get_agent_performance(100)
            stats["agents"]["total"] = perf.get("total_agents", 0)
            stats["agents"]["active"] = len(perf.get("top_performers", []))

            # Performance metrics
            pm = self._get_performance_metrics()
            stats["performance"]["avg_response_time_ms"] = pm.get("avg_latency_ms", 0.0)
            stats["performance"]["success_rate"] = pm.get("success_rate", 0.0)
            if stats["performance"]["success_rate"] > 0:
                stats["performance"]["error_rate"] = round(
                    1.0 - stats["performance"]["success_rate"], 3
                )
        except Exception as e:
            logger.warning("Dashboard stats error: %s: %s", type(e).__name__, e)

        return json_response(stats)

    def _get_stat_cards(self) -> HandlerResult:
        """Return stat cards summarizing key metrics."""
        cards: list[dict[str, Any]] = []

        try:
            storage = self.get_storage()
            if storage:
                summary = self._get_summary_metrics_sql(storage, None)
                cards.append(
                    {
                        "id": "total_debates",
                        "label": "Total Debates",
                        "value": summary.get("total_debates", 0),
                        "icon": "message-circle",
                    }
                )
                cards.append(
                    {
                        "id": "consensus_rate",
                        "label": "Consensus Rate",
                        "value": f"{summary.get('consensus_rate', 0) * 100:.1f}%",
                        "icon": "check-circle",
                    }
                )
                cards.append(
                    {
                        "id": "avg_confidence",
                        "label": "Avg Confidence",
                        "value": f"{summary.get('avg_confidence', 0):.2f}",
                        "icon": "trending-up",
                    }
                )

            perf = self._get_agent_performance(100)
            cards.append(
                {
                    "id": "active_agents",
                    "label": "Active Agents",
                    "value": perf.get("total_agents", 0),
                    "icon": "users",
                }
            )
            cards.append(
                {
                    "id": "avg_elo",
                    "label": "Avg ELO Rating",
                    "value": perf.get("avg_elo", 0),
                    "icon": "award",
                }
            )
        except Exception as e:
            logger.warning("Stat cards error: %s: %s", type(e).__name__, e)

        return json_response({"cards": cards})

    def _get_team_performance(self, limit: int, offset: int) -> HandlerResult:
        """Return team performance grouped by provider from ELO ratings."""
        teams: list[dict[str, Any]] = []

        try:
            perf = self._get_agent_performance(200)
            performers = perf.get("top_performers", [])

            # Group agents by provider prefix
            provider_groups: dict[str, list[dict]] = {}
            for agent in performers:
                name = agent.get("name", "")
                provider = name.split("-")[0] if "-" in name else name
                provider_groups.setdefault(provider, []).append(agent)

            for provider, agents in provider_groups.items():
                avg_elo = sum(a.get("elo", 1000) for a in agents) / len(agents) if agents else 0
                total_debates = sum(a.get("debates_count", 0) for a in agents)
                avg_win_rate = (
                    sum(a.get("win_rate", 0) for a in agents) / len(agents) if agents else 0
                )
                teams.append(
                    {
                        "team_id": provider,
                        "team_name": provider.title(),
                        "member_count": len(agents),
                        "avg_elo": round(avg_elo, 1),
                        "total_debates": total_debates,
                        "avg_win_rate": round(avg_win_rate, 3),
                    }
                )

            teams.sort(key=lambda t: t["avg_elo"], reverse=True)
        except Exception as e:
            logger.warning("Team performance error: %s: %s", type(e).__name__, e)

        paginated = teams[offset : offset + limit]
        return json_response({"teams": paginated, "total": len(teams)})

    def _get_team_performance_detail(self, team_id: str) -> HandlerResult:
        """Return team performance detail for a provider group."""
        if not team_id:
            return error_response("team_id is required", 400)

        detail: dict[str, Any] = {
            "team_id": team_id,
            "team_name": team_id.title(),
            "member_count": 0,
            "debates_participated": 0,
            "avg_response_time_ms": 0,
            "consensus_contribution_rate": 0.0,
            "quality_score": 0.0,
            "members": [],
        }

        try:
            perf = self._get_agent_performance(200)
            performers = perf.get("top_performers", [])

            members = [a for a in performers if a.get("name", "").startswith(team_id)]
            detail["member_count"] = len(members)
            detail["debates_participated"] = sum(a.get("debates_count", 0) for a in members)
            if members:
                avg_win = sum(a.get("win_rate", 0) for a in members) / len(members)
                detail["consensus_contribution_rate"] = round(avg_win, 3)
                avg_elo = sum(a.get("elo", 1000) for a in members) / len(members)
                detail["quality_score"] = round(avg_elo / 1000, 2)
            detail["members"] = members

            pm = self._get_performance_metrics()
            detail["avg_response_time_ms"] = pm.get("avg_latency_ms", 0.0)
        except Exception as e:
            logger.warning("Team detail error: %s: %s", type(e).__name__, e)

        return json_response(detail)

    def _get_top_senders(self, limit: int, offset: int) -> HandlerResult:
        """Return top debate initiators ranked by count."""
        senders: list[dict[str, Any]] = []

        try:
            storage = self.get_storage()
            if storage:
                with storage.db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT domain, COUNT(*) as cnt FROM debates "
                        "GROUP BY domain ORDER BY cnt DESC LIMIT ? OFFSET ?",
                        (limit, offset),
                    )
                    for row in cursor.fetchall():
                        senders.append(
                            {
                                "domain": row[0] or "general",
                                "debate_count": row[1],
                            }
                        )
        except Exception as e:
            logger.warning("Top senders error: %s: %s", type(e).__name__, e)

        return json_response({"senders": senders, "total": len(senders)})

    def _get_labels(self) -> HandlerResult:
        """Return label/domain counts from debate storage."""
        labels: list[dict[str, Any]] = []

        try:
            storage = self.get_storage()
            if storage:
                with storage.db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT domain, COUNT(*) as cnt FROM debates "
                        "GROUP BY domain ORDER BY cnt DESC LIMIT 20"
                    )
                    for row in cursor.fetchall():
                        labels.append(
                            {
                                "name": row[0] or "general",
                                "count": row[1],
                            }
                        )
        except Exception as e:
            logger.warning("Labels error: %s: %s", type(e).__name__, e)

        return json_response({"labels": labels})

    def _get_activity(self, limit: int, offset: int) -> HandlerResult:
        """Return recent activity feed from debate storage."""
        activity: list[dict[str, Any]] = []
        total = 0

        try:
            storage = self.get_storage()
            if storage:
                with storage.db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM debates")
                    row = cursor.fetchone()
                    total = row[0] if row else 0

                    cursor.execute(
                        "SELECT id, domain, consensus_reached, confidence, "
                        "created_at FROM debates "
                        "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                        (limit, offset),
                    )
                    for row in cursor.fetchall():
                        activity.append(
                            {
                                "type": "debate",
                                "debate_id": row[0],
                                "domain": row[1],
                                "consensus_reached": bool(row[2]),
                                "confidence": row[3],
                                "created_at": row[4],
                            }
                        )
        except Exception as e:
            logger.warning("Activity feed error: %s: %s", type(e).__name__, e)

        return json_response({"activity": activity, "total": total})

    def _get_inbox_summary(self) -> HandlerResult:
        """Return inbox summary derived from debate storage."""
        summary: dict[str, Any] = {
            "total_messages": 0,
            "unread_messages": 0,
            "urgent_count": 0,
            "today_count": 0,
            "by_label": [],
            "by_importance": {"high": 0, "medium": 0, "low": 0},
            "response_rate": 0.0,
            "avg_response_time_hours": 0.0,
        }

        try:
            storage = self.get_storage()
            if storage:
                sql_summary = self._get_summary_metrics_sql(storage, None)
                summary["total_messages"] = sql_summary.get("total_debates", 0)
                summary["response_rate"] = sql_summary.get("consensus_rate", 0.0)

                today_start = (
                    datetime.now(timezone.utc)
                    .replace(hour=0, minute=0, second=0, microsecond=0)
                    .isoformat()
                )
                try:
                    with storage.db.connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT COUNT(*) FROM debates WHERE created_at >= ?",
                            (today_start,),
                        )
                        row = cursor.fetchone()
                        summary["today_count"] = row[0] if row else 0
                except Exception as e:
                    logger.debug(f"Could not get today's inbox count: {e}")
        except Exception as e:
            logger.warning("Inbox summary error: %s: %s", type(e).__name__, e)

        return json_response(summary)

    def _get_quick_actions(self) -> HandlerResult:
        """Return quick actions list."""
        actions = [
            {
                "id": "archive_read",
                "name": "Archive All Read",
                "description": "Archive all read emails older than 24 hours",
                "icon": "archive",
                "available": True,
            },
            {
                "id": "snooze_low",
                "name": "Snooze Low Priority",
                "description": "Snooze all low priority emails until tomorrow",
                "icon": "clock",
                "available": True,
            },
            {
                "id": "mark_spam",
                "name": "Mark Bulk as Spam",
                "description": "Mark selected promotional emails as spam",
                "icon": "slash",
                "available": True,
            },
            {
                "id": "complete_actions",
                "name": "Complete Done Actions",
                "description": "Mark action items you've completed",
                "icon": "check-circle",
                "available": True,
            },
            {
                "id": "ai_respond",
                "name": "AI Auto-Respond",
                "description": "Let AI draft responses for simple emails",
                "icon": "sparkles",
                "available": True,
            },
            {
                "id": "sync_inbox",
                "name": "Sync Inbox",
                "description": "Force sync with email provider",
                "icon": "refresh",
                "available": True,
            },
        ]
        return json_response({"actions": actions, "total": len(actions)})

    def _execute_quick_action(self, action_id: str) -> HandlerResult:
        """Execute a quick action (stub)."""
        if not action_id:
            return error_response("action_id is required", 400)
        return json_response(
            {
                "success": True,
                "action_id": action_id,
                "executed_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def _get_urgent_items(self, limit: int, offset: int) -> HandlerResult:
        """Return urgent items: debates with low confidence or no consensus."""
        items: list[dict[str, Any]] = []

        try:
            storage = self.get_storage()
            if storage:
                with storage.db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT id, domain, confidence, created_at FROM debates "
                        "WHERE consensus_reached = 0 OR confidence < 0.3 "
                        "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                        (limit, offset),
                    )
                    for row in cursor.fetchall():
                        items.append(
                            {
                                "id": row[0],
                                "type": "low_consensus",
                                "domain": row[1],
                                "confidence": row[2],
                                "created_at": row[3],
                                "description": f"Debate in {row[1] or 'general'} needs attention",
                            }
                        )
        except Exception as e:
            logger.warning("Urgent items error: %s: %s", type(e).__name__, e)

        return json_response({"items": items, "total": len(items)})

    def _dismiss_urgent_item(self, item_id: str) -> HandlerResult:
        """Dismiss an urgent item (stub)."""
        if not item_id:
            return error_response("item_id is required", 400)
        return json_response({"success": True})

    def _get_pending_actions(self, limit: int, offset: int) -> HandlerResult:
        """Return pending actions: recent debates awaiting review."""
        actions: list[dict[str, Any]] = []

        try:
            storage = self.get_storage()
            if storage:
                with storage.db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT id, domain, created_at FROM debates "
                        "WHERE status = 'pending' OR status = 'in_progress' "
                        "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                        (limit, offset),
                    )
                    for row in cursor.fetchall():
                        actions.append(
                            {
                                "id": row[0],
                                "type": "review_debate",
                                "domain": row[1],
                                "created_at": row[2],
                                "description": f"Review debate in {row[1] or 'general'}",
                            }
                        )
        except Exception as e:
            logger.warning("Pending actions error: %s: %s", type(e).__name__, e)

        return json_response({"actions": actions, "total": len(actions)})

    def _complete_pending_action(self, action_id: str) -> HandlerResult:
        """Complete a pending action (stub)."""
        if not action_id:
            return error_response("action_id is required", 400)
        return json_response({"success": True})

    def _search_dashboard(self, query: str) -> HandlerResult:
        """Search dashboard data by domain or debate ID."""
        results: list[dict[str, Any]] = []

        if not query:
            return json_response({"results": [], "total": 0})

        try:
            storage = self.get_storage()
            if storage:
                with storage.db.connection() as conn:
                    cursor = conn.cursor()
                    like_query = f"%{query}%"
                    cursor.execute(
                        "SELECT id, domain, consensus_reached, confidence, "
                        "created_at FROM debates "
                        "WHERE id LIKE ? OR domain LIKE ? "
                        "ORDER BY created_at DESC LIMIT 20",
                        (like_query, like_query),
                    )
                    for row in cursor.fetchall():
                        results.append(
                            {
                                "id": row[0],
                                "domain": row[1],
                                "consensus_reached": bool(row[2]),
                                "confidence": row[3],
                                "created_at": row[4],
                            }
                        )
        except Exception as e:
            logger.warning("Dashboard search error: %s: %s", type(e).__name__, e)

        return json_response({"results": results, "total": len(results)})

    def _export_dashboard_data(self) -> HandlerResult:
        """Export dashboard data as a JSON snapshot."""
        export: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {},
            "agent_performance": {},
            "consensus_insights": {},
        }

        try:
            storage = self.get_storage()
            if storage:
                export["summary"] = self._get_summary_metrics_sql(storage, None)
            export["agent_performance"] = self._get_agent_performance(50)
            export["consensus_insights"] = self._get_consensus_insights(None)
        except Exception as e:
            logger.warning("Export error: %s: %s", type(e).__name__, e)

        return json_response(export)

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
            calibration_tracker = self.ctx.get("calibration_tracker")
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
                                "ece": round(s.ece, 3) if hasattr(s, "ece") else None,
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
            performance_monitor = self.ctx.get("performance_monitor")
            if performance_monitor:
                insights = performance_monitor.get_performance_insights()  # type: ignore[attr-defined]
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
            prompt_evolver = self.ctx.get("prompt_evolver")
            if prompt_evolver:
                # Get version counts per agent
                for agent_name in ["claude", "gemini", "codex", "grok"]:
                    try:
                        version = prompt_evolver.get_prompt_version(agent_name)  # type: ignore[attr-defined]
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
                patterns = prompt_evolver.get_top_patterns(limit=100)  # type: ignore[attr-defined]
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
            elo_system = self.ctx.get("elo_system")
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
