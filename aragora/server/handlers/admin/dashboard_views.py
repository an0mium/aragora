"""Dashboard view endpoint methods (mixin).

Contains read-only dashboard endpoints: overview, debates list/detail,
stats, stat cards, team performance, top senders, labels, activity feed,
and inbox summary.

Extracted from dashboard.py for maintainability.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from aragora.config import CACHE_TTL_DASHBOARD_DEBATES

from ..base import (
    HandlerResult,
    error_response,
    json_response,
    ttl_cache,
)
from ..openapi_decorator import api_endpoint

if TYPE_CHECKING:
    from aragora.ranking.elo import EloSystem

logger = logging.getLogger(__name__)


class DashboardViewsMixin:
    """Mixin providing dashboard view endpoints.

    Requires the host class to provide:
    - get_storage() -> storage instance
    - get_elo_system() -> ELO system instance
    - _get_summary_metrics_sql(storage, domain) -> dict
    - _get_agent_performance(limit) -> dict
    - _get_performance_metrics() -> dict
    """

    # Stub methods expected from the composing class
    def get_storage(self) -> Any: ...  # Returns storage with connection() method
    def get_elo_system(self) -> "EloSystem | None": ...
    def _get_summary_metrics_sql(self, storage: Any, domain: str | None) -> dict[str, Any]: ...
    def _get_agent_performance(self, limit: int) -> dict[str, Any]: ...
    def _get_performance_metrics(self) -> dict[str, Any]: ...

    @api_endpoint(
        method="GET",
        path="/api/v1/dashboard/overview",
        summary="Get dashboard overview",
        tags=["Dashboard"],
        responses={
            "200": {"description": "Dashboard overview data"},
            "401": {"description": "Unauthorized"},
            "403": {"description": "Forbidden - requires dashboard.read"},
        },
    )
    @ttl_cache(
        ttl_seconds=CACHE_TTL_DASHBOARD_DEBATES, key_prefix="dashboard_overview", skip_first=True
    )
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
                    with storage.connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT COUNT(*) FROM debates WHERE created_at >= ?",
                            (today_start,),
                        )
                        row = cursor.fetchone()
                        overview["total_debates_today"] = row[0] if row else 0
                except Exception as e:
                    logger.debug("Could not get today's debates count: %s", e)

            # Agent performance as stat cards
            perf = self._get_agent_performance(5)
            overview["stats"] = [
                {"label": "Total Agents", "value": perf.get("total_agents", 0)},
                {"label": "Avg ELO", "value": perf.get("avg_elo", 0)},
            ]
        except Exception as e:
            logger.warning("Overview error: %s: %s", type(e).__name__, e)

        return json_response(overview)

    @api_endpoint(
        method="GET",
        path="/api/v1/dashboard/debates",
        summary="List dashboard debates",
        tags=["Dashboard"],
        parameters=[
            {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 10}},
            {"name": "offset", "in": "query", "schema": {"type": "integer", "default": 0}},
            {"name": "status", "in": "query", "schema": {"type": "string"}},
        ],
        responses={
            "200": {
                "description": "Paginated list of debates",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "debates": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "domain": {"type": "string"},
                                            "status": {"type": "string"},
                                            "consensus_reached": {"type": "boolean"},
                                            "confidence": {"type": "number"},
                                            "created_at": {"type": "string"},
                                        },
                                    },
                                },
                                "total": {"type": "integer"},
                            },
                        }
                    }
                },
            },
            "401": {"description": "Unauthorized"},
            "403": {"description": "Forbidden"},
        },
    )
    def _get_dashboard_debates(self, limit: int, offset: int, status: Any) -> HandlerResult:
        """Return dashboard debate list from storage."""
        debates: list[dict[str, Any]] = []
        total = 0

        try:
            storage = self.get_storage()
            if storage:
                with storage.connection() as conn:
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

    @api_endpoint(
        method="GET",
        path="/api/v1/dashboard/debates/{debate_id}",
        summary="Get debate detail",
        tags=["Dashboard"],
        parameters=[
            {"name": "debate_id", "in": "path", "schema": {"type": "string"}, "required": True},
        ],
        responses={
            "200": {"description": "Debate detail returned"},
            "401": {"description": "Unauthorized"},
            "404": {"description": "Debate not found"},
        },
    )
    def _get_dashboard_debate(self, debate_id: str) -> HandlerResult:
        """Return a single debate summary entry."""
        if not debate_id:
            return error_response("debate_id is required", 400)
        return json_response({"debate_id": debate_id})

    @api_endpoint(
        method="GET",
        path="/api/v1/dashboard/stats",
        summary="Get dashboard statistics",
        tags=["Dashboard"],
        responses={
            "200": {"description": "Dashboard statistics"},
            "401": {"description": "Unauthorized"},
        },
    )
    @ttl_cache(
        ttl_seconds=CACHE_TTL_DASHBOARD_DEBATES, key_prefix="dashboard_stats", skip_first=True
    )
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
                    with storage.connection() as conn:
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
                    logger.debug("Could not get debate stats: %s", e)

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

    @api_endpoint(
        method="GET",
        path="/api/v1/dashboard/stat-cards",
        summary="Get dashboard stat cards",
        tags=["Dashboard"],
        responses={
            "200": {"description": "Stat card data for dashboard widgets"},
            "401": {"description": "Unauthorized"},
        },
    )
    @ttl_cache(ttl_seconds=CACHE_TTL_DASHBOARD_DEBATES, key_prefix="stat_cards", skip_first=True)
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

    @api_endpoint(
        method="GET",
        path="/api/v1/dashboard/team-performance",
        summary="Get team performance metrics",
        tags=["Dashboard"],
        parameters=[
            {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 10}},
            {"name": "offset", "in": "query", "schema": {"type": "integer", "default": 0}},
        ],
        responses={
            "200": {"description": "Team performance data"},
            "401": {"description": "Unauthorized"},
        },
    )
    @ttl_cache(
        ttl_seconds=CACHE_TTL_DASHBOARD_DEBATES, key_prefix="team_performance", skip_first=True
    )
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

    @api_endpoint(
        method="GET",
        path="/api/v1/dashboard/team-performance/{team_id}",
        summary="Get team performance detail",
        tags=["Dashboard"],
        parameters=[
            {"name": "team_id", "in": "path", "schema": {"type": "string"}, "required": True},
        ],
        responses={
            "200": {"description": "Detailed team performance"},
            "401": {"description": "Unauthorized"},
            "404": {"description": "Team not found"},
        },
    )
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

    @api_endpoint(
        method="GET",
        path="/api/v1/dashboard/top-senders",
        summary="Get top email senders",
        tags=["Dashboard"],
        parameters=[
            {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 10}},
            {"name": "offset", "in": "query", "schema": {"type": "integer", "default": 0}},
        ],
        responses={
            "200": {"description": "Top senders list"},
            "401": {"description": "Unauthorized"},
        },
    )
    def _get_top_senders(self, limit: int, offset: int) -> HandlerResult:
        """Return top debate initiators ranked by count."""
        senders: list[dict[str, Any]] = []

        try:
            storage = self.get_storage()
            if storage:
                with storage.connection() as conn:
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

    @api_endpoint(
        method="GET",
        path="/api/v1/dashboard/labels",
        summary="Get dashboard labels",
        tags=["Dashboard"],
        responses={
            "200": {"description": "Label categories and counts"},
            "401": {"description": "Unauthorized"},
        },
    )
    def _get_labels(self) -> HandlerResult:
        """Return label/domain counts from debate storage."""
        labels: list[dict[str, Any]] = []

        try:
            storage = self.get_storage()
            if storage:
                with storage.connection() as conn:
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

    @api_endpoint(
        method="GET",
        path="/api/v1/dashboard/activity",
        summary="Get recent activity feed",
        tags=["Dashboard"],
        parameters=[
            {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}},
            {"name": "offset", "in": "query", "schema": {"type": "integer", "default": 0}},
        ],
        responses={
            "200": {"description": "Activity feed entries"},
            "401": {"description": "Unauthorized"},
        },
    )
    def _get_activity(self, limit: int, offset: int) -> HandlerResult:
        """Return recent activity feed from debate storage."""
        activity: list[dict[str, Any]] = []
        total = 0

        try:
            storage = self.get_storage()
            if storage:
                with storage.connection() as conn:
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

    @api_endpoint(
        method="GET",
        path="/api/v1/dashboard/inbox-summary",
        summary="Get inbox summary",
        tags=["Dashboard"],
        responses={
            "200": {"description": "Inbox summary with counts by category"},
            "401": {"description": "Unauthorized"},
        },
    )
    @ttl_cache(ttl_seconds=CACHE_TTL_DASHBOARD_DEBATES, key_prefix="inbox_summary", skip_first=True)
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
                    with storage.connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT COUNT(*) FROM debates WHERE created_at >= ?",
                            (today_start,),
                        )
                        row = cursor.fetchone()
                        summary["today_count"] = row[0] if row else 0
                except Exception as e:
                    logger.debug("Could not get today's inbox count: %s", e)
        except Exception as e:
            logger.warning("Inbox summary error: %s: %s", type(e).__name__, e)

        return json_response(summary)
