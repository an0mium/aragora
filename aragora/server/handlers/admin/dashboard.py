"""
Debate Dashboard endpoint handler.

Provides a consolidated view of debate metrics for dashboard visualization.
Aggregates data from ELO, consensus, prometheus, and debate storage systems.

View endpoints are in dashboard_views.py (DashboardViewsMixin).
Action/analytics endpoints are in dashboard_actions.py (DashboardActionsMixin).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from aragora.config import CACHE_TTL_DASHBOARD_DEBATES
from aragora.protocols import HTTPRequestHandler

from ..base import (
    HandlerResult,
    error_response,
    get_int_param,
    json_response,
    ttl_cache,
)
from ..secure import SecureHandler, ForbiddenError, UnauthorizedError
from ..utils.rate_limit import _get_limiter, get_client_ip

# Import extracted utility functions
from .dashboard_metrics import (
    get_debate_patterns,
    get_recent_activity_legacy,
    get_recent_activity_sql,
    get_summary_metrics_legacy,
    get_summary_metrics_sql,
    process_debates_single_pass,
)
from .dashboard_health import (
    get_connector_health,
    get_connector_type,
    get_system_health,
)
from .dashboard_views import DashboardViewsMixin
from .dashboard_actions import DashboardActionsMixin

logger = logging.getLogger(__name__)


def _call_bypassing_decorators(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Call underlying function bypassing @require_permission/@rate_limit wrappers."""
    inner = func
    while hasattr(inner, "__wrapped__"):
        inner = inner.__wrapped__
    return inner(*args, **kwargs)


# Rate limiter for dashboard endpoints (60 requests per minute - frequently accessed)
# Use _get_limiter to register in the global registry (cleared by clear_all_limiters)
_dashboard_limiter = _get_limiter("admin_dashboard", 60)

# RBAC Permission constants for admin dashboard endpoints
PERM_ADMIN_DASHBOARD_READ = "admin:dashboard:read"
PERM_ADMIN_DASHBOARD_WRITE = "admin:dashboard:write"
PERM_ADMIN_METRICS_READ = "admin:metrics:read"

# Legacy permission alias (kept for backward compatibility)
DASHBOARD_PERMISSION = PERM_ADMIN_DASHBOARD_READ


class DashboardHandler(DashboardActionsMixin, DashboardViewsMixin, SecureHandler):
    """Handler for dashboard endpoint.

    Requires authentication and dashboard.read permission (RBAC).
    """

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

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

    async def handle(
        self, path: str, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> HandlerResult | None:
        """Route dashboard requests to appropriate methods with RBAC."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _dashboard_limiter.is_allowed(client_ip):
            logger.warning("Rate limit exceeded for dashboard endpoint: %s", client_ip)
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # RBAC: Require authentication and admin:dashboard:read permission
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
            self.check_permission(auth_context, PERM_ADMIN_DASHBOARD_READ)
        except UnauthorizedError as e:
            logger.warning("Dashboard auth error: %s", e)
            return error_response("Authentication required", 401)
        except ForbiddenError as e:
            logger.warning("Dashboard access denied: %s", e)
            return error_response("Permission denied", 403)

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
            # Additional permission check for metrics endpoints
            try:
                self.check_permission(auth_context, PERM_ADMIN_METRICS_READ)
            except ForbiddenError as e:
                logger.warning("Metrics access denied: %s", e)
                return error_response("Permission denied", 403)
            return self._get_quality_metrics()

        return None

    async def handle_post(
        self, path: str, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> HandlerResult | None:
        """Handle dashboard write actions (stub)."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _dashboard_limiter.is_allowed(client_ip):
            logger.warning("Rate limit exceeded for dashboard endpoint: %s", client_ip)
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # RBAC: Require authentication and admin:dashboard:write permission for POST operations
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
            self.check_permission(auth_context, PERM_ADMIN_DASHBOARD_WRITE)
        except UnauthorizedError as e:
            logger.warning("Dashboard auth error: %s", e)
            return error_response("Authentication required", 401)
        except ForbiddenError as e:
            logger.warning("Dashboard access denied: %s", e)
            return error_response("Permission denied", 403)

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

    # ------------------------------------------------------------------
    # Data aggregation methods (kept in main handler - used by mixins)
    # ------------------------------------------------------------------

    @ttl_cache(
        ttl_seconds=CACHE_TTL_DASHBOARD_DEBATES, key_prefix="dashboard_debates", skip_first=True
    )
    def _get_debates_dashboard(self, domain: str | None, limit: int, hours: int) -> HandlerResult:
        """Get consolidated debate metrics for dashboard."""
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

        result["debate_patterns"] = {"disagreement_stats": {}, "early_stopping": {}}
        result["agent_performance"] = self._get_agent_performance(limit)
        result["consensus_insights"] = self._get_consensus_insights(domain)
        result["system_health"] = self._get_system_health()

        request_elapsed = time.perf_counter() - request_start
        summary = result.get("summary", {})
        total_debates = summary.get("total_debates", 0) if isinstance(summary, dict) else 0
        logger.debug(
            "Dashboard response: elapsed=%.3fs, total_debates=%d", request_elapsed, total_debates
        )
        return json_response(result)

    def _process_debates_single_pass(
        self, debates: list[Any], domain: str | None, hours: int
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Process all debate metrics in a single pass through the data."""
        return _call_bypassing_decorators(process_debates_single_pass, debates, domain, hours)

    def _get_summary_metrics_sql(self, storage: Any, domain: str | None) -> dict[str, Any]:
        """Get summary metrics using SQL aggregation (O(1) memory)."""
        return _call_bypassing_decorators(get_summary_metrics_sql, storage, domain)

    def _get_recent_activity_sql(self, storage: Any, hours: int) -> dict[str, Any]:
        """Get recent activity metrics using SQL aggregation."""
        return _call_bypassing_decorators(get_recent_activity_sql, storage, hours)

    def _get_summary_metrics(self, domain: str | None, debates: list[Any]) -> dict[str, Any]:
        """Get high-level summary metrics (legacy, kept for compatibility)."""
        return _call_bypassing_decorators(get_summary_metrics_legacy, domain, debates)

    def _get_recent_activity(
        self, domain: str | None, hours: int, debates: list[Any]
    ) -> dict[str, Any]:
        """Get recent debate activity metrics."""
        return _call_bypassing_decorators(get_recent_activity_legacy, domain, hours, debates)

    @ttl_cache(
        ttl_seconds=CACHE_TTL_DASHBOARD_DEBATES, key_prefix="agent_performance", skip_first=True
    )
    def _get_agent_performance(self, limit: int) -> dict[str, Any]:
        """Get agent performance metrics."""
        performance = {
            "top_performers": [],
            "total_agents": 0,
            "avg_elo": 0,
        }

        try:
            elo = self.get_elo_system()
            if elo:
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

                performance["top_performers"] = all_ratings[:limit]
                performance["total_agents"] = len(all_ratings)

                if all_ratings:
                    performance["avg_elo"] = round(
                        sum(r["elo"] for r in all_ratings) / len(all_ratings), 1
                    )
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError) as e:
            logger.warning("Agent performance error: %s: %s", type(e).__name__, e)

        return performance

    def _get_debate_patterns(self, debates: list[Any]) -> dict[str, Any]:
        """Get debate pattern statistics."""
        return _call_bypassing_decorators(get_debate_patterns, debates)

    @ttl_cache(
        ttl_seconds=CACHE_TTL_DASHBOARD_DEBATES, key_prefix="consensus_insights", skip_first=True
    )
    def _get_consensus_insights(self, domain: str | None) -> dict[str, Any]:
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
        except (KeyError, ValueError, OSError, TypeError, AttributeError) as e:
            logger.warning("Consensus insights error: %s: %s", type(e).__name__, e)

        return insights

    def _get_system_health(self) -> dict[str, Any]:
        """Get system health metrics."""
        health = get_system_health()
        health["connector_health"] = self._get_connector_health()
        return health

    def _get_connector_health(self) -> dict[str, Any]:
        """Get connector health metrics for dashboard."""
        return get_connector_health()

    def _get_connector_type(self, connector: Any) -> str:
        """Extract connector type from connector instance."""
        return get_connector_type(connector)

    # Remaining endpoint methods are provided by:
    # - DashboardViewsMixin (overview, debates, stats, teams, senders, labels, activity, inbox)
    # - DashboardActionsMixin (quick actions, urgent, pending, search, export, quality metrics)
