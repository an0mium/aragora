"""
Analytics Dashboard endpoint handlers.

Endpoints:
- GET  /api/analytics/summary           - Dashboard summary
- GET  /api/analytics/trends/findings   - Finding trends over time
- GET  /api/analytics/remediation       - Remediation metrics
- GET  /api/analytics/agents            - Agent performance metrics
- GET  /api/analytics/cost              - Cost analysis
- GET  /api/analytics/compliance        - Compliance scorecard
- GET  /api/analytics/heatmap           - Risk heatmap data
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from aragora.server.http_utils import run_async as _run_async

from .base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    require_user_auth,
    safe_error_message,
)
from .utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)


class AnalyticsDashboardHandler(BaseHandler):
    """Handler for analytics dashboard endpoints."""

    ROUTES = [
        "/api/analytics/summary",
        "/api/analytics/trends/findings",
        "/api/analytics/remediation",
        "/api/analytics/agents",
        "/api/analytics/cost",
        "/api/analytics/compliance",
        "/api/analytics/heatmap",
        "/api/analytics/tokens",
        "/api/analytics/tokens/trends",
        "/api/analytics/tokens/providers",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    @rate_limit(rpm=60)
    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route GET requests to appropriate methods."""
        if path == "/api/analytics/summary":
            return self._get_summary(query_params)
        elif path == "/api/analytics/trends/findings":
            return self._get_finding_trends(query_params)
        elif path == "/api/analytics/remediation":
            return self._get_remediation_metrics(query_params)
        elif path == "/api/analytics/agents":
            return self._get_agent_metrics(query_params)
        elif path == "/api/analytics/cost":
            return self._get_cost_metrics(query_params)
        elif path == "/api/analytics/compliance":
            return self._get_compliance_scorecard(query_params)
        elif path == "/api/analytics/heatmap":
            return self._get_risk_heatmap(query_params)
        elif path == "/api/analytics/tokens":
            return self._get_token_usage(query_params)
        elif path == "/api/analytics/tokens/trends":
            return self._get_token_trends(query_params)
        elif path == "/api/analytics/tokens/providers":
            return self._get_provider_breakdown(query_params)

        return None

    @require_user_auth
    @handle_errors("get analytics summary")
    def _get_summary(self, query_params: dict, user=None) -> HandlerResult:
        """
        Get dashboard summary with key metrics.

        Query params:
        - workspace_id: Workspace to analyze (required)
        - time_range: Time range (24h, 7d, 30d, 90d, 365d, all) - default 30d

        Response:
        {
            "workspace_id": "...",
            "time_range": "30d",
            "total_findings": 150,
            "open_findings": 45,
            "critical_findings": 5,
            "resolved_last_period": 23,
            "finding_trend": "down",
            "trend_percentage": -15.5,
            "top_categories": [...],
            "recent_critical": [...]
        }
        """
        workspace_id = query_params.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id is required", 400)

        time_range_str = query_params.get("time_range", "30d")

        try:
            from aragora.analytics import get_analytics_dashboard, TimeRange

            dashboard = get_analytics_dashboard()
            time_range = TimeRange(time_range_str)

            summary = _run_async(dashboard.get_summary(workspace_id, time_range))

            return json_response(summary.to_dict())

        except ValueError as e:
            logger.warning(f"Invalid analytics summary parameter: {e}")
            return error_response(f"Invalid time_range: {time_range_str}", 400)
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Data error in analytics summary: {e}")
            return error_response(safe_error_message(e, "analytics summary"), 400)
        except Exception as e:
            logger.exception(f"Unexpected error getting analytics summary: {e}")
            return error_response(safe_error_message(e, "analytics summary"), 500)

    @require_user_auth
    @handle_errors("get finding trends")
    def _get_finding_trends(self, query_params: dict, user=None) -> HandlerResult:
        """
        Get finding trends over time.

        Query params:
        - workspace_id: Workspace to analyze (required)
        - time_range: Time range (24h, 7d, 30d, 90d, 365d, all) - default 30d
        - granularity: Time bucket size (hour, day, week, month) - default day

        Response:
        {
            "trends": [
                {
                    "timestamp": "2024-01-15T00:00:00Z",
                    "total": 12,
                    "by_severity": {"critical": 1, "high": 5, ...},
                    "by_category": {...}
                },
                ...
            ]
        }
        """
        workspace_id = query_params.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id is required", 400)

        time_range_str = query_params.get("time_range", "30d")
        granularity_str = query_params.get("granularity", "day")

        try:
            from aragora.analytics import (
                get_analytics_dashboard,
                TimeRange,
                Granularity,
            )

            dashboard = get_analytics_dashboard()
            time_range = TimeRange(time_range_str)
            granularity = Granularity(granularity_str)

            trends = asyncio.run(
                dashboard.get_finding_trends(workspace_id, time_range, granularity)
            )

            return json_response(
                {
                    "workspace_id": workspace_id,
                    "time_range": time_range_str,
                    "granularity": granularity_str,
                    "trends": [t.to_dict() for t in trends],
                }
            )

        except ValueError as e:
            logger.warning(f"Invalid finding trends parameter: {e}")
            return error_response(f"Invalid parameter: {e}", 400)
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Data error in finding trends: {e}")
            return error_response(safe_error_message(e, "finding trends"), 400)
        except Exception as e:
            logger.exception(f"Unexpected error getting finding trends: {e}")
            return error_response(safe_error_message(e, "finding trends"), 500)

    @require_user_auth
    @handle_errors("get remediation metrics")
    def _get_remediation_metrics(self, query_params: dict, user=None) -> HandlerResult:
        """
        Get remediation performance metrics.

        Query params:
        - workspace_id: Workspace to analyze (required)
        - time_range: Time range - default 30d

        Response:
        {
            "total_resolved": 120,
            "total_open": 45,
            "mttr_hours": 48.5,
            "mttr_by_severity": {"critical": 4.2, "high": 24.1, ...},
            "false_positive_rate": 0.08,
            "accepted_risk_rate": 0.03
        }
        """
        workspace_id = query_params.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id is required", 400)

        time_range_str = query_params.get("time_range", "30d")

        try:
            from aragora.analytics import get_analytics_dashboard, TimeRange

            dashboard = get_analytics_dashboard()
            time_range = TimeRange(time_range_str)

            metrics = _run_async(dashboard.get_remediation_metrics(workspace_id, time_range))

            return json_response(
                {
                    "workspace_id": workspace_id,
                    "time_range": time_range_str,
                    **metrics.to_dict(),
                }
            )

        except ValueError as e:
            logger.warning(f"Invalid remediation metrics parameter: {e}")
            return error_response(f"Invalid parameter: {e}", 400)
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Data error in remediation metrics: {e}")
            return error_response(safe_error_message(e, "remediation metrics"), 400)
        except Exception as e:
            logger.exception(f"Unexpected error getting remediation metrics: {e}")
            return error_response(safe_error_message(e, "remediation metrics"), 500)

    @require_user_auth
    @handle_errors("get agent metrics")
    def _get_agent_metrics(self, query_params: dict, user=None) -> HandlerResult:
        """
        Get agent performance metrics.

        Query params:
        - workspace_id: Workspace to analyze (required)
        - time_range: Time range - default 30d

        Response:
        {
            "agents": [
                {
                    "agent_id": "claude-3-sonnet",
                    "agent_name": "Claude 3.5 Sonnet",
                    "total_findings": 45,
                    "agreement_rate": 0.92,
                    "precision": 0.87,
                    "finding_distribution": {...}
                },
                ...
            ]
        }
        """
        workspace_id = query_params.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id is required", 400)

        time_range_str = query_params.get("time_range", "30d")

        try:
            from aragora.analytics import get_analytics_dashboard, TimeRange

            dashboard = get_analytics_dashboard()
            time_range = TimeRange(time_range_str)

            metrics = _run_async(dashboard.get_agent_metrics(workspace_id, time_range))

            return json_response(
                {
                    "workspace_id": workspace_id,
                    "time_range": time_range_str,
                    "agents": [m.to_dict() for m in metrics],
                }
            )

        except ValueError as e:
            logger.warning(f"Invalid agent metrics parameter: {e}")
            return error_response(f"Invalid parameter: {e}", 400)
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Data error in agent metrics: {e}")
            return error_response(safe_error_message(e, "agent metrics"), 400)
        except Exception as e:
            logger.exception(f"Unexpected error getting agent metrics: {e}")
            return error_response(safe_error_message(e, "agent metrics"), 500)

    @require_user_auth
    @handle_errors("get cost metrics")
    def _get_cost_metrics(self, query_params: dict, user=None) -> HandlerResult:
        """
        Get cost analysis for audits.

        Query params:
        - workspace_id: Workspace to analyze (required)
        - time_range: Time range - default 30d

        Response:
        {
            "total_audits": 50,
            "total_cost_usd": 125.50,
            "avg_cost_per_audit": 2.51,
            "cost_by_type": {"security": 45.00, ...},
            "token_usage": {"input": 500000, "output": 100000}
        }
        """
        workspace_id = query_params.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id is required", 400)

        time_range_str = query_params.get("time_range", "30d")

        try:
            from aragora.analytics import get_analytics_dashboard, TimeRange

            dashboard = get_analytics_dashboard()
            time_range = TimeRange(time_range_str)

            metrics = _run_async(dashboard.get_cost_metrics(workspace_id, time_range))

            return json_response(
                {
                    "workspace_id": workspace_id,
                    "time_range": time_range_str,
                    **metrics.to_dict(),
                }
            )

        except ValueError as e:
            logger.warning(f"Invalid cost metrics parameter: {e}")
            return error_response(f"Invalid parameter: {e}", 400)
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Data error in cost metrics: {e}")
            return error_response(safe_error_message(e, "cost metrics"), 400)
        except Exception as e:
            logger.exception(f"Unexpected error getting cost metrics: {e}")
            return error_response(safe_error_message(e, "cost metrics"), 500)

    @require_user_auth
    @handle_errors("get compliance scorecard")
    def _get_compliance_scorecard(self, query_params: dict, user=None) -> HandlerResult:
        """
        Get compliance scorecard for specified frameworks.

        Query params:
        - workspace_id: Workspace to analyze (required)
        - frameworks: Comma-separated list (SOC2,GDPR,HIPAA,PCI-DSS)

        Response:
        {
            "scores": [
                {
                    "framework": "SOC2",
                    "score": 0.85,
                    "passing_controls": 17,
                    "failing_controls": 3,
                    "critical_gaps": ["access_control"]
                },
                ...
            ]
        }
        """
        workspace_id = query_params.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id is required", 400)

        frameworks_str = query_params.get("frameworks", "SOC2,GDPR,HIPAA,PCI-DSS")
        frameworks = [f.strip() for f in frameworks_str.split(",")]

        try:
            from aragora.analytics import get_analytics_dashboard

            dashboard = get_analytics_dashboard()

            scores = _run_async(dashboard.get_compliance_scorecard(workspace_id, frameworks))

            return json_response(
                {
                    "workspace_id": workspace_id,
                    "scores": [s.to_dict() for s in scores],
                }
            )

        except ValueError as e:
            logger.warning(f"Invalid compliance scorecard parameter: {e}")
            return error_response(f"Invalid parameter: {e}", 400)
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Data error in compliance scorecard: {e}")
            return error_response(safe_error_message(e, "compliance scorecard"), 400)
        except Exception as e:
            logger.exception(f"Unexpected error getting compliance scorecard: {e}")
            return error_response(safe_error_message(e, "compliance scorecard"), 500)

    @require_user_auth
    @handle_errors("get risk heatmap")
    def _get_risk_heatmap(self, query_params: dict, user=None) -> HandlerResult:
        """
        Get risk heatmap data (category x severity).

        Query params:
        - workspace_id: Workspace to analyze (required)
        - time_range: Time range - default 30d

        Response:
        {
            "cells": [
                {
                    "category": "security",
                    "severity": "critical",
                    "count": 5,
                    "trend": "down"
                },
                ...
            ]
        }
        """
        workspace_id = query_params.get("workspace_id")
        if not workspace_id:
            return error_response("workspace_id is required", 400)

        time_range_str = query_params.get("time_range", "30d")

        try:
            from aragora.analytics import get_analytics_dashboard, TimeRange

            dashboard = get_analytics_dashboard()
            time_range = TimeRange(time_range_str)

            cells = _run_async(dashboard.get_risk_heatmap(workspace_id, time_range))

            return json_response(
                {
                    "workspace_id": workspace_id,
                    "time_range": time_range_str,
                    "cells": [c.to_dict() for c in cells],
                }
            )

        except ValueError as e:
            logger.warning(f"Invalid risk heatmap parameter: {e}")
            return error_response(f"Invalid parameter: {e}", 400)
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Data error in risk heatmap: {e}")
            return error_response(safe_error_message(e, "risk heatmap"), 400)
        except Exception as e:
            logger.exception(f"Unexpected error getting risk heatmap: {e}")
            return error_response(safe_error_message(e, "risk heatmap"), 500)

    @require_user_auth
    @handle_errors("get token usage")
    def _get_token_usage(self, query_params: dict, user=None) -> HandlerResult:
        """
        Get token usage summary.

        Query params:
        - org_id: Organization ID (required)
        - days: Number of days to look back (default: 30)

        Response:
        {
            "org_id": "...",
            "period": {"start": "...", "end": "..."},
            "total_tokens_in": 500000,
            "total_tokens_out": 100000,
            "total_cost_usd": "125.50",
            "cost_by_provider": {"anthropic": "80.00", "openai": "45.50"},
            "top_models": [{"model": "claude-opus-4", "tokens": 400000, "cost": "60.00"}]
        }
        """
        org_id = query_params.get("org_id")
        if not org_id:
            return error_response("org_id is required", 400)

        try:
            days = int(query_params.get("days", "30"))
        except ValueError:
            days = 30

        try:
            from datetime import datetime, timedelta, timezone

            from aragora.billing.usage import UsageTracker

            tracker = UsageTracker()
            period_end = datetime.now(timezone.utc)
            period_start = period_end - timedelta(days=days)

            summary = tracker.get_summary(org_id, period_start, period_end)

            return json_response(
                {
                    "org_id": org_id,
                    "period": {
                        "start": period_start.isoformat(),
                        "end": period_end.isoformat(),
                        "days": days,
                    },
                    "total_tokens_in": summary.total_tokens_in,
                    "total_tokens_out": summary.total_tokens_out,
                    "total_tokens": summary.total_tokens_in + summary.total_tokens_out,
                    "total_cost_usd": str(summary.total_cost_usd),
                    "total_debates": summary.total_debates,
                    "total_agent_calls": summary.total_agent_calls,
                    "cost_by_provider": {k: str(v) for k, v in summary.cost_by_provider.items()},
                    "debates_by_day": summary.debates_by_day,
                }
            )

        except Exception as e:
            logger.exception(f"Unexpected error getting token usage: {e}")
            return error_response(safe_error_message(e, "token usage"), 500)

    @require_user_auth
    @handle_errors("get token trends")
    def _get_token_trends(self, query_params: dict, user=None) -> HandlerResult:
        """
        Get token usage trends over time.

        Query params:
        - org_id: Organization ID (required)
        - days: Number of days to look back (default: 30)
        - granularity: 'day' or 'hour' (default: 'day')

        Response:
        {
            "org_id": "...",
            "granularity": "day",
            "data_points": [
                {"date": "2026-01-15", "tokens_in": 10000, "tokens_out": 2000, "cost": "1.50"},
                ...
            ]
        }
        """
        org_id = query_params.get("org_id")
        if not org_id:
            return error_response("org_id is required", 400)

        try:
            days = int(query_params.get("days", "30"))
        except ValueError:
            days = 30

        granularity = query_params.get("granularity", "day")
        if granularity not in ("day", "hour"):
            granularity = "day"

        try:
            from datetime import datetime, timedelta, timezone

            from aragora.billing.usage import UsageTracker

            tracker = UsageTracker()
            period_end = datetime.now(timezone.utc)
            period_start = period_end - timedelta(days=days)

            data_points = []
            with tracker._connection() as conn:
                if granularity == "day":
                    date_format = "DATE(created_at)"
                else:
                    date_format = "strftime('%Y-%m-%d %H:00', created_at)"

                rows = conn.execute(
                    f"""
                    SELECT
                        {date_format} as period,
                        SUM(tokens_in) as tokens_in,
                        SUM(tokens_out) as tokens_out,
                        SUM(CAST(cost_usd AS REAL)) as cost,
                        COUNT(*) as event_count
                    FROM usage_events
                    WHERE org_id = ?
                        AND created_at >= ?
                        AND created_at <= ?
                    GROUP BY {date_format}
                    ORDER BY period
                    """,
                    (org_id, period_start.isoformat(), period_end.isoformat()),
                ).fetchall()

                for row in rows:
                    data_points.append(
                        {
                            "period": row["period"],
                            "tokens_in": row["tokens_in"] or 0,
                            "tokens_out": row["tokens_out"] or 0,
                            "total_tokens": (row["tokens_in"] or 0) + (row["tokens_out"] or 0),
                            "cost_usd": f"{row['cost'] or 0:.4f}",
                            "event_count": row["event_count"],
                        }
                    )

            return json_response(
                {
                    "org_id": org_id,
                    "granularity": granularity,
                    "period": {
                        "start": period_start.isoformat(),
                        "end": period_end.isoformat(),
                        "days": days,
                    },
                    "data_points": data_points,
                }
            )

        except Exception as e:
            logger.exception(f"Unexpected error getting token trends: {e}")
            return error_response(safe_error_message(e, "token trends"), 500)

    @require_user_auth
    @handle_errors("get provider breakdown")
    def _get_provider_breakdown(self, query_params: dict, user=None) -> HandlerResult:
        """
        Get detailed breakdown by provider and model.

        Query params:
        - org_id: Organization ID (required)
        - days: Number of days to look back (default: 30)

        Response:
        {
            "org_id": "...",
            "providers": [
                {
                    "provider": "anthropic",
                    "total_tokens": 500000,
                    "total_cost": "80.00",
                    "models": [
                        {"model": "claude-opus-4", "tokens_in": 400000, "tokens_out": 50000, "cost": "60.00"},
                        {"model": "claude-sonnet-4", "tokens_in": 40000, "tokens_out": 10000, "cost": "20.00"}
                    ]
                },
                ...
            ]
        }
        """
        org_id = query_params.get("org_id")
        if not org_id:
            return error_response("org_id is required", 400)

        try:
            days = int(query_params.get("days", "30"))
        except ValueError:
            days = 30

        try:
            from datetime import datetime, timedelta, timezone

            from aragora.billing.usage import UsageTracker

            tracker = UsageTracker()
            period_end = datetime.now(timezone.utc)
            period_start = period_end - timedelta(days=days)

            providers = {}
            with tracker._connection() as conn:
                rows = conn.execute(
                    """
                    SELECT
                        provider,
                        model,
                        SUM(tokens_in) as tokens_in,
                        SUM(tokens_out) as tokens_out,
                        SUM(CAST(cost_usd AS REAL)) as cost,
                        COUNT(*) as call_count
                    FROM usage_events
                    WHERE org_id = ?
                        AND created_at >= ?
                        AND created_at <= ?
                        AND provider IS NOT NULL
                        AND provider != ''
                    GROUP BY provider, model
                    ORDER BY cost DESC
                    """,
                    (org_id, period_start.isoformat(), period_end.isoformat()),
                ).fetchall()

                for row in rows:
                    provider = row["provider"] or "unknown"
                    if provider not in providers:
                        providers[provider] = {
                            "provider": provider,
                            "total_tokens_in": 0,
                            "total_tokens_out": 0,
                            "total_cost": 0.0,
                            "models": [],
                        }

                    tokens_in = row["tokens_in"] or 0
                    tokens_out = row["tokens_out"] or 0
                    cost = row["cost"] or 0.0

                    providers[provider]["total_tokens_in"] += tokens_in
                    providers[provider]["total_tokens_out"] += tokens_out
                    providers[provider]["total_cost"] += cost
                    providers[provider]["models"].append(
                        {
                            "model": row["model"] or "unknown",
                            "tokens_in": tokens_in,
                            "tokens_out": tokens_out,
                            "total_tokens": tokens_in + tokens_out,
                            "cost_usd": f"{cost:.4f}",
                            "call_count": row["call_count"],
                        }
                    )

            # Format totals
            result_providers = []
            for p in providers.values():
                p["total_tokens"] = p["total_tokens_in"] + p["total_tokens_out"]
                p["total_cost"] = f"{p['total_cost']:.4f}"
                result_providers.append(p)

            # Sort by total cost
            result_providers.sort(key=lambda x: float(x["total_cost"]), reverse=True)

            return json_response(
                {
                    "org_id": org_id,
                    "period": {
                        "start": period_start.isoformat(),
                        "end": period_end.isoformat(),
                        "days": days,
                    },
                    "providers": result_providers,
                }
            )

        except Exception as e:
            logger.exception(f"Unexpected error getting provider breakdown: {e}")
            return error_response(safe_error_message(e, "provider breakdown"), 500)
