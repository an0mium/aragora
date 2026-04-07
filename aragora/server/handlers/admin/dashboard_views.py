"""Dashboard view endpoint methods (mixin).

Contains read-only dashboard endpoints: overview, debates list/detail,
stats, stat cards, team performance, top senders, labels, activity feed,
and inbox summary.

Extracted from dashboard.py for maintainability.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, cast

from aragora.config import CACHE_TTL_DASHBOARD_DEBATES

from ..base import (
    HandlerResult,
    error_response,
    json_response,
    ttl_cache,
)
from ..openapi_decorator import api_endpoint
from .dashboard_metrics import (
    ACTIVE_DEBATE_STATUSES,
    find_debate_record,
    load_debate_records,
    summarize_debate_records,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _merge_summary_metrics(
    records: list[dict[str, Any]],
    summary_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    summary = summarize_debate_records(records)
    if summary_metrics:
        summary.update(summary_metrics)
    return summary


def _debate_list_entry(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record.get("id"),
        "domain": record.get("domain"),
        "status": record.get("status"),
        "consensus_reached": bool(record.get("consensus_reached")),
        "confidence": record.get("confidence"),
        "created_at": record.get("created_at"),
    }


def _overview_debate_entry(record: dict[str, Any]) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "debate_id": record.get("id"),
        "status": record.get("status"),
        "consensus_reached": bool(record.get("consensus_reached")),
        "created_at": record.get("created_at"),
    }
    if record.get("task"):
        entry["task"] = record["task"]
    if record.get("rounds_used") is not None:
        entry["round_count"] = record["rounds_used"]
    if record.get("duration_seconds") is not None:
        entry["duration_ms"] = round(float(record["duration_seconds"]) * 1000, 3)
    return entry


def _count_records_since(records: list[dict[str, Any]], cutoff: datetime) -> int:
    return sum(
        1
        for record in records
        if isinstance(record.get("_sort_created_at"), datetime)
        and record["_sort_created_at"] >= cutoff
    )


def _derive_system_health(summary: dict[str, Any]) -> str:
    if summary.get("needs_attention_debates", 0) > 0:
        return "degraded"
    if summary.get("open_debates", 0) > 0 and summary.get("consensus_rate", 0.0) < 0.5:
        return "degraded"
    return "healthy"


class DashboardViewsMixin:
    """Mixin providing dashboard view endpoints.

    Requires the host class to provide:
    - get_storage() -> storage instance
    - get_elo_system() -> ELO system instance
    - _get_summary_metrics_sql(storage, domain) -> dict
    - _get_agent_performance(limit) -> dict
    - _get_performance_metrics() -> dict
    """

    if TYPE_CHECKING:

        def get_storage(self) -> Any: ...

    def _get_summary_metrics_sql(self, storage: Any, domain: str | None) -> dict[str, Any]:
        raise NotImplementedError

    def _get_agent_performance(self, limit: int) -> dict[str, Any]:
        raise NotImplementedError

    def _get_performance_metrics(self) -> dict[str, Any]:
        raise NotImplementedError

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
            records: list[dict[str, Any]] = []
            summary: dict[str, Any] = {}
            if storage:
                try:
                    records = load_debate_records(storage)
                    summary = _merge_summary_metrics(
                        records, self._get_summary_metrics_sql(storage, None)
                    )
                except (KeyError, ValueError, OSError, TypeError) as e:
                    logger.warning("Overview summary error: %s: %s", type(e).__name__, e)
                    summary = summarize_debate_records(records)

            today_start = datetime.now(timezone.utc).replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
            overview["recent_debates"] = [_overview_debate_entry(record) for record in records[:5]]
            overview["active_debates"] = sum(
                1 for record in records if str(record.get("status") or "") in ACTIVE_DEBATE_STATUSES
            )
            overview["total_debates_today"] = _count_records_since(records, today_start)
            overview["consensus_rate"] = summary.get("consensus_rate", 0.0)
            overview["avg_debate_duration_ms"] = summary.get("avg_duration_ms", 0)
            overview["system_health"] = _derive_system_health(summary)
            overview["stats"] = [
                {"label": "Total Debates", "value": summary.get("total_debates", 0)},
                {"label": "Open Debates", "value": summary.get("open_debates", 0)},
                {
                    "label": "Consensus Rate",
                    "value": f"{summary.get('consensus_rate', 0.0) * 100:.1f}%",
                },
                {
                    "label": "Avg Confidence",
                    "value": round(summary.get("avg_confidence", 0.0), 2),
                },
            ]
        except (KeyError, ValueError, OSError, TypeError) as e:
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
                records = load_debate_records(storage)
                normalized_status = str(status).strip().lower() if status else None
                if normalized_status:
                    records = [
                        record
                        for record in records
                        if str(record.get("status") or "") == normalized_status
                    ]

                safe_offset = max(offset, 0)
                total = len(records)
                debates = [
                    _debate_list_entry(record)
                    for record in records[safe_offset : safe_offset + max(limit, 0)]
                ]
        except (KeyError, ValueError, OSError, TypeError) as e:
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

        try:
            storage = self.get_storage()
            if not storage:
                return error_response("Debate not found", 404)

            record = find_debate_record(storage, debate_id)
            if not record:
                return error_response("Debate not found", 404)

            detail = {
                "debate_id": record.get("id"),
                **_debate_list_entry(record),
                "needs_attention": bool(record.get("needs_attention")),
            }
            if record.get("task"):
                detail["task"] = record["task"]
            if record.get("rounds_used") is not None:
                detail["rounds_used"] = record["rounds_used"]
            if record.get("duration_seconds") is not None:
                detail["duration_seconds"] = round(float(record["duration_seconds"]), 3)
            return json_response(detail)
        except (KeyError, ValueError, OSError, TypeError) as e:
            logger.warning("Dashboard debate detail error: %s: %s", type(e).__name__, e)
            return error_response("Failed to load debate", 500)

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

        records: list[dict[str, Any]] = []

        try:
            storage = self.get_storage()
            if storage:
                records = load_debate_records(storage)
                summary = summarize_debate_records(records)
                try:
                    summary.update(self._get_summary_metrics_sql(storage, None) or {})
                except (KeyError, ValueError, OSError, TypeError) as e:
                    logger.warning("Dashboard stats summary error: %s: %s", type(e).__name__, e)

                now = datetime.now(timezone.utc)
                today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                week_start = now - timedelta(days=7)
                month_start = now - timedelta(days=30)

                stats["debates"]["total"] = summary.get("total_debates", 0)
                stats["debates"]["today"] = _count_records_since(records, today_start)
                stats["debates"]["this_week"] = _count_records_since(records, week_start)
                stats["debates"]["this_month"] = _count_records_since(records, month_start)
                stats["performance"]["consensus_rate"] = summary.get("consensus_rate", 0.0)

                by_status: dict[str, int] = {}
                for record in records:
                    status_name = str(record.get("status") or "")
                    if status_name:
                        by_status[status_name] = by_status.get(status_name, 0) + 1
                stats["debates"]["by_status"] = by_status

                today_records = [
                    record
                    for record in records
                    if isinstance(record.get("_sort_created_at"), datetime)
                    and record["_sort_created_at"] >= today_start
                ]
                stats["usage"]["api_calls_today"] = len(today_records)
                stats["usage"]["tokens_used_today"] = sum(
                    int(record.get("total_tokens") or 0) for record in today_records
                )
                stats["usage"]["storage_used_bytes"] = sum(
                    int(artifact_bytes)
                    for record in records
                    if isinstance((artifact_bytes := record.get("artifact_bytes")), (int, float))
                )
        except (KeyError, ValueError, OSError, TypeError) as e:
            logger.warning("Dashboard storage stats error: %s: %s", type(e).__name__, e)

        try:
            perf = self._get_agent_performance(100)
            stats["agents"]["total"] = perf.get("total_agents", 0)
            stats["agents"]["active"] = len(perf.get("top_performers", []))
        except (KeyError, ValueError, OSError, TypeError) as e:
            logger.warning("Dashboard agent stats error: %s: %s", type(e).__name__, e)

        try:
            pm = self._get_performance_metrics()
            stats["performance"]["avg_response_time_ms"] = pm.get("avg_latency_ms", 0.0)
            stats["performance"]["success_rate"] = pm.get("success_rate", 0.0)
            if stats["performance"]["success_rate"] > 0:
                stats["performance"]["error_rate"] = round(
                    1.0 - stats["performance"]["success_rate"], 3
                )
        except (KeyError, ValueError, OSError, TypeError) as e:
            logger.warning("Dashboard performance stats error: %s: %s", type(e).__name__, e)

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
                records = load_debate_records(storage)
                summary = summarize_debate_records(records)
                try:
                    summary.update(self._get_summary_metrics_sql(storage, None) or {})
                except (KeyError, ValueError, OSError, TypeError) as e:
                    logger.warning("Stat cards summary error: %s: %s", type(e).__name__, e)
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
        except (KeyError, ValueError, OSError, TypeError) as e:
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
        except (KeyError, ValueError, TypeError, ZeroDivisionError) as e:
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
        except (KeyError, ValueError, TypeError, ZeroDivisionError) as e:
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
        total = 0

        try:
            storage = self.get_storage()
            if storage:
                grouped: dict[str, list[dict[str, Any]]] = {}
                for record in load_debate_records(storage):
                    domain_name = str(record.get("domain_label") or "general")
                    grouped.setdefault(domain_name, []).append(record)

                total = len(grouped)
                senders = sorted(
                    (
                        {
                            "domain": domain_name,
                            "debate_count": len(entries),
                            "consensus_rate": round(
                                sum(1 for entry in entries if entry.get("consensus_reached"))
                                / len(entries),
                                3,
                            ),
                            "avg_confidence": round(
                                sum(
                                    float(entry["confidence"])
                                    for entry in entries
                                    if entry.get("confidence") is not None
                                )
                                / max(
                                    1,
                                    sum(
                                        1
                                        for entry in entries
                                        if entry.get("confidence") is not None
                                    ),
                                ),
                                3,
                            )
                            if any(entry.get("confidence") is not None for entry in entries)
                            else 0.0,
                            "open_debates": sum(
                                1
                                for entry in entries
                                if str(entry.get("status") or "") in ACTIVE_DEBATE_STATUSES
                            ),
                        }
                        for domain_name, entries in grouped.items()
                    ),
                    key=lambda sender: (
                        -cast(int, sender["debate_count"]),
                        str(sender["domain"]),
                    ),
                )[max(offset, 0) : max(offset, 0) + max(limit, 0)]
        except (KeyError, ValueError, OSError, TypeError) as e:
            logger.warning("Top senders error: %s: %s", type(e).__name__, e)

        return json_response({"senders": senders, "total": total})

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
                grouped: dict[str, int] = {}
                for record in load_debate_records(storage):
                    domain_name = str(record.get("domain_label") or "general")
                    grouped[domain_name] = grouped.get(domain_name, 0) + 1

                labels = [
                    {"name": domain_name, "count": count}
                    for domain_name, count in sorted(
                        grouped.items(),
                        key=lambda item: (-item[1], item[0]),
                    )[:20]
                ]
        except (KeyError, ValueError, OSError, TypeError) as e:
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
                records = load_debate_records(storage)
                total = len(records)
                activity = [
                    {
                        "type": "debate",
                        "debate_id": record.get("id"),
                        "domain": record.get("domain"),
                        "consensus_reached": bool(record.get("consensus_reached")),
                        "confidence": record.get("confidence"),
                        "created_at": record.get("created_at"),
                    }
                    for record in records[max(offset, 0) : max(offset, 0) + max(limit, 0)]
                ]
        except (KeyError, ValueError, OSError, TypeError) as e:
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
                records = load_debate_records(storage)
                sql_summary = summarize_debate_records(records)
                try:
                    sql_summary.update(self._get_summary_metrics_sql(storage, None) or {})
                except (KeyError, ValueError, OSError, TypeError) as e:
                    logger.warning("Inbox summary metrics error: %s: %s", type(e).__name__, e)
                summary["total_messages"] = sql_summary.get("total_debates", 0)
                summary["unread_messages"] = sql_summary.get("open_debates", 0)
                summary["urgent_count"] = sql_summary.get("needs_attention_debates", 0)
                summary["response_rate"] = sql_summary.get("consensus_rate", 0.0)

                today_start = datetime.now(timezone.utc).replace(
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
                summary["today_count"] = _count_records_since(records, today_start)

                grouped: dict[str, int] = {}
                by_importance = {"high": 0, "medium": 0, "low": 0}
                durations_hours: list[float] = []

                for record in records:
                    domain_name = str(record.get("domain_label") or "general")
                    grouped[domain_name] = grouped.get(domain_name, 0) + 1

                    confidence = record.get("confidence")
                    if confidence is None:
                        by_importance["medium"] += 1
                    elif float(confidence) >= 0.8:
                        by_importance["high"] += 1
                    elif float(confidence) >= 0.5:
                        by_importance["medium"] += 1
                    else:
                        by_importance["low"] += 1

                    if record.get("duration_seconds") is not None:
                        durations_hours.append(float(record["duration_seconds"]) / 3600.0)

                summary["by_label"] = [
                    {"name": domain_name, "count": count}
                    for domain_name, count in sorted(
                        grouped.items(),
                        key=lambda item: (-item[1], item[0]),
                    )[:10]
                ]
                summary["by_importance"] = by_importance
                if durations_hours:
                    summary["avg_response_time_hours"] = round(
                        sum(durations_hours) / len(durations_hours),
                        3,
                    )
        except (KeyError, ValueError, OSError, TypeError) as e:
            logger.warning("Inbox summary error: %s: %s", type(e).__name__, e)

        return json_response(summary)
