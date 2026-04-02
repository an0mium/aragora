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
    pass

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

    if TYPE_CHECKING:

        def get_storage(self) -> Any: ...

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
                except (OSError, ValueError, TypeError) as e:
                    logger.debug("Could not get today's debates count: %s", e)

            # Agent performance as stat cards
            perf = self._get_agent_performance(5)
            overview["stats"] = [
                {"label": "Total Agents", "value": perf.get("total_agents", 0)},
                {"label": "Avg ELO", "value": perf.get("avg_elo", 0)},
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
        except (KeyError, ValueError, OSError, TypeError) as e:
            logger.warning("Dashboard debates error: %s: %s", type(e).__name__, e)

        return json_response({"debates": debates, "total": total})

    @staticmethod
    def _coerce_dashboard_int(value: Any, fallback: int = 0) -> int:
        """Best-effort integer coercion for dashboard detail payloads."""
        if isinstance(value, bool):
            return fallback
        if isinstance(value, int):
            return max(value, 0)
        if isinstance(value, float):
            return max(int(value), 0)
        if isinstance(value, str):
            try:
                return max(int(value), 0)
            except ValueError:
                return fallback
        return fallback

    @staticmethod
    def _coerce_dashboard_float(value: Any, fallback: float = 0.0) -> float:
        """Best-effort float coercion for dashboard detail payloads."""
        if isinstance(value, bool):
            return fallback
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return fallback
        return fallback

    @staticmethod
    def _normalize_dashboard_string_list(values: Any) -> list[str]:
        """Keep only string entries from list-like values."""
        if not isinstance(values, list):
            return []
        return [item for item in values if isinstance(item, str)]

    def _build_dashboard_arguments(self, messages: Any) -> tuple[list[dict[str, Any]], int]:
        """Convert stored debate messages into dashboard-friendly argument entries."""
        if not isinstance(messages, list):
            return [], 0

        arguments: list[dict[str, Any]] = []
        rounds = 0
        for message in messages:
            if not isinstance(message, dict):
                continue

            round_num = self._coerce_dashboard_int(message.get("round"))
            rounds = max(rounds, round_num)

            agent = (
                message.get("agent") or message.get("author") or message.get("role") or "unknown"
            )
            if not isinstance(agent, str):
                agent = str(agent)

            position = message.get("position") or message.get("role") or ""
            if not isinstance(position, str):
                position = str(position)

            content = message.get("content", "")
            if not isinstance(content, str):
                content = str(content)

            arguments.append(
                {
                    "agent": agent,
                    "round": round_num,
                    "position": position,
                    "content": content,
                }
            )

        return arguments, rounds

    def _build_dashboard_cost_breakdown(self, per_agent_cost: Any) -> list[dict[str, Any]]:
        """Convert per-agent cost maps into the dashboard's array form."""
        if not isinstance(per_agent_cost, dict):
            return []

        breakdown: list[dict[str, Any]] = []
        for agent, cost in per_agent_cost.items():
            if not isinstance(agent, str):
                continue
            breakdown.append(
                {
                    "agent": agent,
                    "cost_usd": self._coerce_dashboard_float(cost),
                }
            )
        return breakdown

    def _derive_dashboard_verdict(
        self, status: str, consensus_reached: bool, confidence: float
    ) -> str:
        """Infer a coarse verdict when only debate storage data is available."""
        normalized_status = status.lower()
        if normalized_status == "timeout":
            return "TIMEOUT"
        if consensus_reached and confidence >= 0.7:
            return "APPROVED"
        if consensus_reached:
            return "APPROVED_WITH_CONDITIONS"
        if normalized_status in {"pending", "running", "in_progress", "queued"}:
            return "IN_PROGRESS"
        return "NEEDS_REVIEW"

    def _normalize_dashboard_detail(
        self,
        detail: dict[str, Any],
        *,
        debate_id: str,
        package_available: bool,
        detail_source: str,
    ) -> dict[str, Any]:
        """Add stable aliases and defaults for dashboard debate detail responses."""
        normalized = dict(detail)
        normalized["debate_id"] = debate_id
        normalized["id"] = debate_id

        question = normalized.get("question")
        if not isinstance(question, str):
            question = str(question or "")

        task = normalized.get("task")
        if not isinstance(task, str) or not task:
            task = question

        participants = self._normalize_dashboard_string_list(normalized.get("participants"))
        if not participants:
            participants = self._normalize_dashboard_string_list(normalized.get("agents"))

        status = normalized.get("status", "unknown")
        if not isinstance(status, str):
            status = str(status)

        confidence = self._coerce_dashboard_float(normalized.get("confidence"))
        consensus_reached = bool(normalized.get("consensus_reached", False))

        verdict = normalized.get("verdict")
        if not isinstance(verdict, str) or not verdict:
            verdict = self._derive_dashboard_verdict(status, consensus_reached, confidence)

        final_answer = normalized.get("final_answer", "")
        if not isinstance(final_answer, str):
            final_answer = str(final_answer or "")

        explanation_summary = normalized.get("explanation_summary", "")
        if not isinstance(explanation_summary, str):
            explanation_summary = str(explanation_summary or "")

        summary = normalized.get("summary")
        if not isinstance(summary, str) or not summary:
            summary = explanation_summary or final_answer or question

        rounds = self._coerce_dashboard_int(normalized.get("rounds"))

        created_at = normalized.get("created_at")
        if created_at is not None and not isinstance(created_at, str):
            created_at = str(created_at)

        duration_seconds = self._coerce_dashboard_float(normalized.get("duration_seconds"))

        cost = normalized.get("cost")
        if not isinstance(cost, dict):
            cost = {}
        per_agent_cost = cost.get("per_agent_cost")
        if not isinstance(per_agent_cost, dict):
            per_agent_cost = {}
        total_cost_usd = self._coerce_dashboard_float(cost.get("total_cost_usd"))

        normalized.update(
            {
                "question": question,
                "task": task,
                "participants": participants,
                "agents": participants,
                "status": status,
                "confidence": confidence,
                "consensus_reached": consensus_reached,
                "verdict": verdict,
                "final_answer": final_answer,
                "explanation_summary": explanation_summary,
                "summary": summary,
                "rounds": rounds,
                "created_at": created_at,
                "duration_seconds": duration_seconds,
                "cost": {
                    "total_cost_usd": total_cost_usd,
                    "per_agent_cost": per_agent_cost,
                },
                "cost_breakdown": normalized.get("cost_breakdown")
                if isinstance(normalized.get("cost_breakdown"), list)
                else self._build_dashboard_cost_breakdown(per_agent_cost),
                "arguments": normalized.get("arguments")
                if isinstance(normalized.get("arguments"), list)
                else [],
                "next_steps": normalized.get("next_steps")
                if isinstance(normalized.get("next_steps"), list)
                else [],
                "receipt": normalized.get("receipt")
                if isinstance(normalized.get("receipt"), dict) or normalized.get("receipt") is None
                else None,
                "argument_map": normalized.get("argument_map")
                if isinstance(normalized.get("argument_map"), dict)
                else None,
                "consensus": {
                    "reached": consensus_reached,
                    "confidence": confidence,
                    "verdict": verdict,
                },
                "package_available": package_available,
                "detail_source": detail_source,
            }
        )
        return normalized

    def _build_minimal_dashboard_detail(self, debate_id: str) -> dict[str, Any]:
        """Return a minimal, non-error payload when storage is unavailable."""
        return self._normalize_dashboard_detail(
            {
                "question": "",
                "status": "unknown",
                "participants": [],
                "rounds": 0,
                "arguments": [],
                "receipt": None,
                "cost": {"total_cost_usd": 0.0, "per_agent_cost": {}},
                "cost_breakdown": [],
                "next_steps": [],
            },
            debate_id=debate_id,
            package_available=False,
            detail_source="minimal_fallback",
        )

    def _build_storage_backed_dashboard_detail(
        self, debate_id: str, debate: dict[str, Any]
    ) -> dict[str, Any]:
        """Build dashboard detail directly from stored debate data."""
        result_data = debate.get("result")
        if not isinstance(result_data, dict):
            result_data = {}

        arguments, derived_rounds = self._build_dashboard_arguments(debate.get("messages"))
        participants = self._normalize_dashboard_string_list(result_data.get("participants"))
        if not participants:
            participants = self._normalize_dashboard_string_list(debate.get("agents"))

        status = debate.get("status", "unknown")
        if not isinstance(status, str):
            status = str(status)

        confidence = self._coerce_dashboard_float(
            result_data.get("confidence"),
            fallback=self._coerce_dashboard_float(debate.get("confidence")),
        )
        consensus_reached = bool(
            result_data.get("consensus_reached", debate.get("consensus_reached", False))
        )

        final_answer = result_data.get("final_answer", "")
        if not isinstance(final_answer, str):
            final_answer = str(final_answer or "")

        explanation_summary = result_data.get("explanation_summary", "")
        if not isinstance(explanation_summary, str):
            explanation_summary = str(explanation_summary or "")

        per_agent_cost = result_data.get("per_agent_cost")
        if not isinstance(per_agent_cost, dict):
            per_agent_cost = {}

        question = debate.get("question", "")
        if not isinstance(question, str):
            question = str(question or "")

        created_at = debate.get("created_at") or result_data.get("created_at")
        if created_at is not None and not isinstance(created_at, str):
            created_at = str(created_at)

        return self._normalize_dashboard_detail(
            {
                "question": question,
                "status": status,
                "confidence": confidence,
                "consensus_reached": consensus_reached,
                "verdict": self._derive_dashboard_verdict(status, consensus_reached, confidence),
                "final_answer": final_answer,
                "explanation_summary": explanation_summary,
                "participants": participants,
                "rounds": self._coerce_dashboard_int(
                    result_data.get("rounds"), fallback=derived_rounds
                ),
                "arguments": arguments,
                "receipt": None,
                "cost": {
                    "total_cost_usd": self._coerce_dashboard_float(
                        result_data.get("total_cost_usd")
                    ),
                    "per_agent_cost": per_agent_cost,
                },
                "cost_breakdown": self._build_dashboard_cost_breakdown(per_agent_cost),
                "argument_map": None,
                "next_steps": [],
                "created_at": created_at,
                "duration_seconds": self._coerce_dashboard_float(
                    result_data.get("duration_seconds"),
                    fallback=self._coerce_dashboard_float(debate.get("duration_seconds")),
                ),
            },
            debate_id=debate_id,
            package_available=False,
            detail_source="storage_fallback",
        )

    def _load_dashboard_detail_package(self, storage: Any, debate_id: str) -> dict[str, Any] | None:
        """Try to reuse the existing decision-package assembler for completed debates."""
        try:
            from aragora.server.handlers.debates.decision_package import DecisionPackageHandler

            package, err = DecisionPackageHandler(ctx={"storage": storage})._assemble_package(
                debate_id
            )
            if err is None and isinstance(package, dict):
                return self._normalize_dashboard_detail(
                    package,
                    debate_id=debate_id,
                    package_available=True,
                    detail_source="decision_package",
                )
            if err is not None:
                logger.debug(
                    "Dashboard debate detail falling back for %s: package unavailable (%s)",
                    debate_id,
                    err.status_code,
                )
        except (
            ImportError,
            KeyError,
            ValueError,
            OSError,
            TypeError,
            RuntimeError,
            AttributeError,
        ) as exc:
            logger.debug("Dashboard debate detail package load failed for %s: %s", debate_id, exc)
        return None

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
        """Return dashboard debate detail, preferring decision-package data when available."""
        if not debate_id:
            return error_response("debate_id is required", 400)

        try:
            storage = self.get_storage()
            if not storage:
                return json_response(self._build_minimal_dashboard_detail(debate_id))

            get_debate = getattr(storage, "get_debate", None)
            if not callable(get_debate):
                return json_response(self._build_minimal_dashboard_detail(debate_id))

            debate = get_debate(debate_id)
            if debate is None:
                return error_response("Debate not found", 404)
            if not isinstance(debate, dict):
                return json_response(self._build_minimal_dashboard_detail(debate_id))

            package_detail = self._load_dashboard_detail_package(storage, debate_id)
            if package_detail is not None:
                return json_response(package_detail)

            return json_response(self._build_storage_backed_dashboard_detail(debate_id, debate))
        except (KeyError, ValueError, OSError, TypeError, AttributeError) as e:
            logger.warning("Dashboard debate detail error: %s: %s", type(e).__name__, e)
            return json_response(self._build_minimal_dashboard_detail(debate_id))

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
                except (OSError, ValueError, TypeError) as e:
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
        except (KeyError, ValueError, OSError, TypeError) as e:
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
        except (KeyError, ValueError, OSError, TypeError) as e:
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
                except (OSError, ValueError, TypeError) as e:
                    logger.debug("Could not get today's inbox count: %s", e)
        except (KeyError, ValueError, OSError, TypeError) as e:
            logger.warning("Inbox summary error: %s: %s", type(e).__name__, e)

        return json_response(summary)
