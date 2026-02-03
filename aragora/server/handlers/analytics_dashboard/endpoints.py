"""Deliberation analytics: summary, channels, consensus, performance.

Endpoints handled:
- GET /api/v1/analytics/deliberations            - Deliberation summary (cached: 300s)
- GET /api/v1/analytics/deliberations/channels   - Deliberations by channel
- GET /api/v1/analytics/deliberations/consensus  - Consensus rates by team
- GET /api/v1/analytics/deliberations/performance - Performance metrics
"""

from __future__ import annotations

from typing import Any

from ._shared import (
    HandlerResult,
    cached_analytics_org,
    error_response,
    get_clamped_int_param,
    handle_errors,
    json_response,
    logger,
    require_user_auth,
    safe_error_message,
)


class DeliberationAnalyticsMixin:
    """Mixin providing deliberation analytics endpoint methods."""

    @require_user_auth
    @handle_errors("get deliberation summary")
    @cached_analytics_org("deliberations", org_key="org_id", days_key="days")
    def _get_deliberation_summary(
        self, query_params: dict[str, Any], handler: Any | None = None, user: Any | None = None
    ) -> HandlerResult:
        """
        Get deliberation analytics summary.

        Query params:
        - org_id: Organization ID (required)
        - days: Number of days to look back (default: 30)
        """
        org_id = query_params.get("org_id")
        if not org_id:
            return error_response("org_id is required", 400)

        days = get_clamped_int_param(query_params, "days", 30, min_val=1, max_val=365)

        try:
            from datetime import datetime, timedelta, timezone

            period_end = datetime.now(timezone.utc)
            period_start = period_end - timedelta(days=days)

            # Query deliberation metrics from database
            from aragora.memory.debate_store import get_debate_store

            store = get_debate_store()

            # Get deliberation statistics
            stats = store.get_deliberation_stats(
                org_id=org_id,
                start_time=period_start,
                end_time=period_end,
            )

            total = stats.get("total", 0)
            completed = stats.get("completed", 0)
            consensus_reached = stats.get("consensus_reached", 0)
            consensus_rate = (
                f"{(consensus_reached / completed * 100):.1f}%" if completed > 0 else "0%"
            )

            return json_response(
                {
                    "org_id": org_id,
                    "period": {
                        "start": period_start.isoformat(),
                        "end": period_end.isoformat(),
                        "days": days,
                    },
                    "total_deliberations": total,
                    "completed": completed,
                    "in_progress": stats.get("in_progress", 0),
                    "failed": stats.get("failed", 0),
                    "consensus_reached": consensus_reached,
                    "consensus_rate": consensus_rate,
                    "avg_rounds": round(stats.get("avg_rounds", 0), 1),
                    "avg_duration_seconds": round(stats.get("avg_duration_seconds", 0), 1),
                    "by_template": stats.get("by_template", {}),
                    "by_priority": stats.get("by_priority", {}),
                }
            )

        except (ImportError, RuntimeError, OSError, LookupError) as e:
            logger.exception(f"Error getting deliberation summary: {e}")
            return error_response(
                safe_error_message(e, "deliberation summary"), 500, code="INTERNAL_ERROR"
            )

    @require_user_auth
    @handle_errors("get deliberation by channel")
    def _get_deliberation_by_channel(
        self, query_params: dict[str, Any], handler: Any | None = None, user: Any | None = None
    ) -> HandlerResult:
        """
        Get deliberation breakdown by channel/platform.

        Query params:
        - org_id: Organization ID (required)
        - days: Number of days to look back (default: 30)
        """
        org_id = query_params.get("org_id")
        if not org_id:
            return error_response("org_id is required", 400, code="MISSING_ORG_ID")

        days = get_clamped_int_param(query_params, "days", 30, min_val=1, max_val=365)

        try:
            from datetime import datetime, timedelta, timezone

            period_end = datetime.now(timezone.utc)
            period_start = period_end - timedelta(days=days)

            from aragora.memory.debate_store import get_debate_store

            store = get_debate_store()

            # Get channel-level statistics
            channel_stats = store.get_deliberation_stats_by_channel(
                org_id=org_id,
                start_time=period_start,
                end_time=period_end,
            )

            # Aggregate by platform
            by_platform: dict = {}
            for ch in channel_stats:
                platform = ch.get("platform", "api")
                if platform not in by_platform:
                    by_platform[platform] = {
                        "count": 0,
                        "consensus_count": 0,
                        "total_duration": 0,
                    }
                by_platform[platform]["count"] += ch.get("total_deliberations", 0)
                by_platform[platform]["consensus_count"] += ch.get("consensus_reached", 0)
                by_platform[platform]["total_duration"] += ch.get("total_duration", 0)

            # Calculate platform-level rates
            platform_summary = {}
            for platform, data in by_platform.items():
                count = data["count"]
                consensus_rate = (
                    f"{(data['consensus_count'] / count * 100):.0f}%" if count > 0 else "0%"
                )
                platform_summary[platform] = {
                    "count": count,
                    "consensus_rate": consensus_rate,
                }

            return json_response(
                {
                    "org_id": org_id,
                    "period": {
                        "start": period_start.isoformat(),
                        "end": period_end.isoformat(),
                        "days": days,
                    },
                    "channels": channel_stats,
                    "by_platform": platform_summary,
                }
            )

        except (ImportError, RuntimeError, OSError, LookupError) as e:
            logger.exception(f"Error getting deliberation by channel: {e}")
            return error_response(
                safe_error_message(e, "deliberation channels"), 500, code="INTERNAL_ERROR"
            )

    @require_user_auth
    @handle_errors("get consensus rates")
    def _get_consensus_rates(
        self, query_params: dict[str, Any], handler: Any | None = None, user: Any | None = None
    ) -> HandlerResult:
        """
        Get consensus rates by agent team composition.

        Query params:
        - org_id: Organization ID (required)
        - days: Number of days to look back (default: 30)
        """
        org_id = query_params.get("org_id")
        if not org_id:
            return error_response("org_id is required", 400, code="MISSING_ORG_ID")

        days = get_clamped_int_param(query_params, "days", 30, min_val=1, max_val=365)

        try:
            from datetime import datetime, timedelta, timezone

            period_end = datetime.now(timezone.utc)
            period_start = period_end - timedelta(days=days)

            from aragora.memory.debate_store import get_debate_store

            store = get_debate_store()

            # Get consensus statistics
            consensus_stats = store.get_consensus_stats(
                org_id=org_id,
                start_time=period_start,
                end_time=period_end,
            )

            return json_response(
                {
                    "org_id": org_id,
                    "period": {
                        "start": period_start.isoformat(),
                        "end": period_end.isoformat(),
                        "days": days,
                    },
                    "overall_consensus_rate": consensus_stats.get("overall_consensus_rate", "0%"),
                    "by_team_size": consensus_stats.get("by_team_size", {}),
                    "by_agent": consensus_stats.get("by_agent", []),
                    "top_teams": consensus_stats.get("top_teams", []),
                }
            )

        except (ImportError, RuntimeError, OSError, LookupError) as e:
            logger.exception(f"Error getting consensus rates: {e}")
            return error_response(
                safe_error_message(e, "consensus rates"), 500, code="INTERNAL_ERROR"
            )

    @require_user_auth
    @handle_errors("get deliberation performance")
    def _get_deliberation_performance(
        self, query_params: dict[str, Any], handler: Any | None = None, user: Any | None = None
    ) -> HandlerResult:
        """
        Get deliberation performance metrics (latency, cost, efficiency).

        Query params:
        - org_id: Organization ID (required)
        - days: Number of days to look back (default: 30)
        - granularity: 'day' or 'week' (default: 'day')
        """
        org_id = query_params.get("org_id")
        if not org_id:
            return error_response("org_id is required", 400, code="MISSING_ORG_ID")

        days = get_clamped_int_param(query_params, "days", 30, min_val=1, max_val=365)

        granularity = query_params.get("granularity", "day")
        if granularity not in ("day", "week"):
            granularity = "day"

        try:
            from datetime import datetime, timedelta, timezone

            period_end = datetime.now(timezone.utc)
            period_start = period_end - timedelta(days=days)

            from aragora.memory.debate_store import get_debate_store

            store = get_debate_store()

            # Get performance statistics
            perf_stats = store.get_deliberation_performance(
                org_id=org_id,
                start_time=period_start,
                end_time=period_end,
                granularity=granularity,
            )

            return json_response(
                {
                    "org_id": org_id,
                    "period": {
                        "start": period_start.isoformat(),
                        "end": period_end.isoformat(),
                        "days": days,
                    },
                    "granularity": granularity,
                    "summary": perf_stats.get("summary", {}),
                    "by_template": perf_stats.get("by_template", []),
                    "trends": perf_stats.get("trends", []),
                    "cost_by_agent": perf_stats.get("cost_by_agent", {}),
                }
            )

        except (ImportError, RuntimeError, OSError, LookupError) as e:
            logger.exception(f"Error getting deliberation performance: {e}")
            return error_response(
                safe_error_message(e, "deliberation performance"), 500, code="INTERNAL_ERROR"
            )
