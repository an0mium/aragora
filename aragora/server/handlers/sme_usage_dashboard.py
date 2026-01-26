"""
SME Usage Dashboard API Handlers.

Provides comprehensive usage visibility for SME users:
- GET /api/v1/usage/summary - Unified usage metrics
- GET /api/v1/usage/breakdown - Detailed breakdown by dimension
- GET /api/v1/usage/roi - ROI analysis
- GET /api/v1/usage/export - CSV/PDF/JSON export
- GET /api/v1/usage/budget-status - Budget utilization

Designed for the SME Starter Pack with focus on cost visibility and ROI.
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, Optional

from .base import (
    error_response,
    get_string_param,
    handle_errors,
    json_response,
)
from .utils.responses import HandlerResult
from .secure import SecureHandler
from .utils.decorators import require_permission
from .utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiter for usage dashboard (60 requests per minute)
_dashboard_limiter = RateLimiter(requests_per_minute=60)


class SMEUsageDashboardHandler(SecureHandler):
    """Handler for SME usage dashboard endpoints.

    Provides comprehensive usage tracking, ROI metrics, and budget
    visibility for SME tier customers. Focused on actionable insights
    and cost management.
    """

    RESOURCE_TYPE = "usage_dashboard"

    ROUTES = [
        "/api/v1/usage/summary",
        "/api/v1/usage/breakdown",
        "/api/v1/usage/roi",
        "/api/v1/usage/export",
        "/api/v1/usage/budget-status",
        "/api/v1/usage/forecast",
        "/api/v1/usage/benchmarks",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(
        self,
        path: str,
        query_params: dict,
        handler,
        method: str = "GET",
    ) -> Optional[HandlerResult]:
        """Route usage dashboard requests to appropriate methods."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _dashboard_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for usage dashboard: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # Determine HTTP method from handler if not provided
        if hasattr(handler, "command"):
            method = handler.command

        route_map = {
            "/api/v1/usage/summary": self._get_summary,
            "/api/v1/usage/breakdown": self._get_breakdown,
            "/api/v1/usage/roi": self._get_roi,
            "/api/v1/usage/export": self._export_usage,
            "/api/v1/usage/budget-status": self._get_budget_status,
            "/api/v1/usage/forecast": self._get_forecast,
            "/api/v1/usage/benchmarks": self._get_benchmarks,
        }

        if path in route_map and method == "GET":
            return route_map[path](handler, query_params)

        return error_response("Method not allowed", 405)

    def _get_user_and_org(self, handler, user):
        """Get user and organization from context."""
        user_store = self.ctx.get("user_store")
        if not user_store:
            return None, None, error_response("Service unavailable", 503)

        db_user = user_store.get_user_by_id(user.user_id)
        if not db_user:
            return None, None, error_response("User not found", 404)

        org = None
        if db_user.org_id:
            org = user_store.get_organization_by_id(db_user.org_id)

        if not org:
            return None, None, error_response("No organization found", 404)

        return db_user, org, None

    def _get_cost_tracker(self):
        """Get cost tracker instance."""
        from aragora.billing.cost_tracker import get_cost_tracker

        return get_cost_tracker()

    def _get_usage_tracker(self):
        """Get usage tracker instance."""
        from aragora.billing.usage import UsageTracker

        try:
            return UsageTracker()
        except Exception as e:
            logger.warning(f"Failed to get usage tracker: {e}")
            return None

    def _get_roi_calculator(self, benchmark: str = "sme"):
        """Get ROI calculator instance."""
        from aragora.billing.roi_calculator import (
            ROICalculator,
            IndustryBenchmark,
        )

        try:
            benchmark_enum = IndustryBenchmark(benchmark)
        except ValueError:
            benchmark_enum = IndustryBenchmark.SME

        return ROICalculator(benchmark=benchmark_enum)

    def _parse_period(self, handler) -> tuple[datetime, datetime, str]:
        """Parse period parameters from request."""
        period = get_string_param(handler, "period", "month")

        end_date = datetime.now(timezone.utc)

        if period == "hour":
            start_date = end_date - timedelta(hours=1)
        elif period == "day":
            start_date = end_date - timedelta(days=1)
        elif period == "week":
            start_date = end_date - timedelta(weeks=1)
        elif period == "month":
            start_date = end_date - timedelta(days=30)
        elif period == "quarter":
            start_date = end_date - timedelta(days=90)
        elif period == "year":
            start_date = end_date - timedelta(days=365)
        else:
            start_date = end_date - timedelta(days=30)

        # Check for custom date range
        start_str = get_string_param(handler, "start", None)
        end_str = get_string_param(handler, "end", None)

        if start_str:
            try:
                start_date = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            except ValueError:
                pass
        if end_str:
            try:
                end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            except ValueError:
                pass

        return start_date, end_date, period

    @handle_errors("get usage summary")
    @require_permission("org:usage:read")
    def _get_summary(
        self,
        handler,
        query_params: dict,
        user=None,
    ) -> HandlerResult:
        """
        Get unified usage summary for the SME dashboard.

        Query Parameters:
            period: Time period (hour, day, week, month, quarter, year)
            start: Custom start date (ISO format)
            end: Custom end date (ISO format)

        Returns:
            JSON response with usage summary:
            {
                "summary": {
                    "period": {...},
                    "debates": {
                        "total": 45,
                        "completed": 40,
                        "consensus_rate": 85.5
                    },
                    "costs": {
                        "total_usd": "12.50",
                        "avg_per_debate_usd": "0.28",
                        "by_provider": {...}
                    },
                    "tokens": {
                        "total": 750000,
                        "input": 500000,
                        "output": 250000
                    },
                    "activity": {
                        "active_days": 15,
                        "peak_day": "2025-01-15",
                        "debates_per_day": 3.0
                    }
                }
            }
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        start_date, end_date, period = self._parse_period(handler)

        # Get usage data
        cost_tracker = self._get_cost_tracker()
        workspace_stats = cost_tracker.get_workspace_stats(org.id)

        # Get usage tracker data
        usage_tracker = self._get_usage_tracker()
        usage_summary = None
        if usage_tracker:
            try:
                usage_summary = usage_tracker.get_summary(
                    org_id=org.id,
                    period_start=start_date,
                    period_end=end_date,
                )
            except Exception as e:
                logger.warning(f"Failed to get usage summary: {e}")

        # Build summary response
        total_cost = Decimal(workspace_stats.get("total_cost_usd", "0"))
        total_debates = usage_summary.total_debates if usage_summary else 0
        completed_debates = usage_summary.total_debates if usage_summary else 0

        avg_cost_per_debate = Decimal("0")
        if completed_debates > 0:
            avg_cost_per_debate = total_cost / completed_debates

        # Calculate consensus rate from completed debates
        # This is a placeholder - real implementation would query debate results
        consensus_rate = 85.0  # Placeholder

        # Calculate days in period
        days_in_period = max(1, (end_date - start_date).days)
        active_days = min(days_in_period, total_debates) if total_debates > 0 else 0
        debates_per_day = total_debates / days_in_period if days_in_period > 0 else 0

        summary = {
            "period": {
                "type": period,
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": days_in_period,
            },
            "debates": {
                "total": total_debates,
                "completed": completed_debates,
                "consensus_rate": consensus_rate,
            },
            "costs": {
                "total_usd": str(total_cost),
                "avg_per_debate_usd": str(avg_cost_per_debate.quantize(Decimal("0.01"))),
                "by_provider": (
                    {k: str(v) for k, v in usage_summary.cost_by_provider.items()}
                    if usage_summary and hasattr(usage_summary, "cost_by_provider")
                    else {}
                ),
            },
            "tokens": {
                "total": workspace_stats.get("total_tokens_in", 0)
                + workspace_stats.get("total_tokens_out", 0),
                "input": workspace_stats.get("total_tokens_in", 0),
                "output": workspace_stats.get("total_tokens_out", 0),
            },
            "activity": {
                "active_days": active_days,
                "debates_per_day": round(debates_per_day, 1),
                "api_calls": workspace_stats.get("total_api_calls", 0),
            },
        }

        return json_response({"summary": summary})

    @handle_errors("get usage breakdown")
    @require_permission("org:usage:read")
    def _get_breakdown(
        self,
        handler,
        query_params: dict,
        user=None,
    ) -> HandlerResult:
        """
        Get detailed usage breakdown by dimension.

        Query Parameters:
            dimension: Breakdown dimension (agent, model, day, debate)
            period: Time period
            start: Custom start date
            end: Custom end date

        Returns:
            JSON response with detailed breakdown
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        start_date, end_date, period = self._parse_period(handler)
        dimension = get_string_param(handler, "dimension", "agent")

        cost_tracker = self._get_cost_tracker()
        workspace_stats = cost_tracker.get_workspace_stats(org.id)

        # Build breakdown based on dimension
        breakdown_data: Dict[str, Any] = {
            "dimension": dimension,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "items": [],
        }

        if dimension == "agent":
            by_agent = workspace_stats.get("cost_by_agent", {})
            total_cost = Decimal(workspace_stats.get("total_cost_usd", "0"))

            for agent_name, cost_str in by_agent.items():
                cost = Decimal(cost_str)
                pct = float(cost / total_cost * 100) if total_cost > 0 else 0
                breakdown_data["items"].append(
                    {
                        "name": agent_name,
                        "cost_usd": str(cost),
                        "percentage": round(pct, 1),
                    }
                )

        elif dimension == "model":
            by_model = workspace_stats.get("cost_by_model", {})
            total_cost = Decimal(workspace_stats.get("total_cost_usd", "0"))

            for model_name, cost_str in by_model.items():
                cost = Decimal(cost_str)
                pct = float(cost / total_cost * 100) if total_cost > 0 else 0
                breakdown_data["items"].append(
                    {
                        "name": model_name,
                        "cost_usd": str(cost),
                        "percentage": round(pct, 1),
                    }
                )

        # Sort by cost descending
        breakdown_data["items"].sort(key=lambda x: Decimal(x["cost_usd"]), reverse=True)

        return json_response({"breakdown": breakdown_data})

    @handle_errors("get ROI metrics")
    @require_permission("org:usage:read")
    def _get_roi(
        self,
        handler,
        query_params: dict,
        user=None,
    ) -> HandlerResult:
        """
        Get ROI analysis for the organization.

        Query Parameters:
            benchmark: Industry benchmark (sme, enterprise, tech_startup, consulting)
            period: Time period
            hourly_rate: Override hourly rate assumption (USD)

        Returns:
            JSON response with ROI metrics:
            {
                "roi": {
                    "period": {...},
                    "time_savings": {...},
                    "cost": {...},
                    "roi": {...},
                    "quality": {...},
                    "productivity": {...},
                    "benchmark": {...}
                }
            }
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        start_date, end_date, period = self._parse_period(handler)
        benchmark = get_string_param(handler, "benchmark", "sme")

        # Get hourly rate override if provided
        hourly_rate_str = get_string_param(handler, "hourly_rate", None)
        hourly_rate = Decimal(hourly_rate_str) if hourly_rate_str else None

        # Get usage data
        cost_tracker = self._get_cost_tracker()
        workspace_stats = cost_tracker.get_workspace_stats(org.id)

        # Build debate ROI inputs from usage data
        # This is a simplified version - real implementation would query debate records
        from aragora.billing.roi_calculator import (
            DebateROIInput,
            IndustryBenchmark,
            ROICalculator,
        )

        try:
            benchmark_enum = IndustryBenchmark(benchmark)
            if hourly_rate:
                calculator = ROICalculator(
                    benchmark=benchmark_enum,
                    hourly_rate_override=hourly_rate,
                )
            else:
                calculator = self._get_roi_calculator(benchmark)
        except Exception:
            calculator = self._get_roi_calculator("sme")

        # Estimate debate count and costs from workspace stats
        total_cost = Decimal(workspace_stats.get("total_cost_usd", "0"))
        api_calls = workspace_stats.get("total_api_calls", 0)

        # Estimate debates (rough heuristic: ~10 API calls per debate)
        estimated_debates = max(1, api_calls // 10) if api_calls > 0 else 0

        # Build sample debate inputs
        debates = []
        if estimated_debates > 0:
            cost_per_debate = total_cost / estimated_debates
            for i in range(estimated_debates):
                debates.append(
                    DebateROIInput(
                        debate_id=f"estimated_{i}",
                        duration_seconds=300,  # Assume 5 min avg
                        cost_usd=cost_per_debate,
                        reached_consensus=True,
                        confidence_score=0.85,
                        agent_count=3,
                        round_count=3,
                        completed=True,
                    )
                )

        # Calculate ROI metrics
        metrics = calculator.calculate_period_roi(
            debates=debates,
            period_start=start_date,
            period_end=end_date,
        )

        return json_response({"roi": metrics.to_dict()})

    @handle_errors("get budget status")
    @require_permission("org:usage:read")
    def _get_budget_status(
        self,
        handler,
        query_params: dict,
        user=None,
    ) -> HandlerResult:
        """
        Get current budget utilization status.

        Returns:
            JSON response with budget status:
            {
                "budget": {
                    "monthly": {
                        "limit_usd": "100.00",
                        "spent_usd": "45.50",
                        "remaining_usd": "54.50",
                        "percent_used": 45.5,
                        "days_remaining": 15,
                        "projected_end_spend_usd": "91.00"
                    },
                    "daily": {...},
                    "alerts": [...]
                }
            }
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        cost_tracker = self._get_cost_tracker()
        budget = cost_tracker.get_budget(workspace_id=org.id, org_id=org.id)

        now = datetime.now(timezone.utc)

        if budget:
            # Calculate days remaining in month
            days_in_month = 30
            day_of_month = now.day
            days_remaining = max(0, days_in_month - day_of_month)

            # Calculate projected spend
            if day_of_month > 0:
                daily_rate = budget.current_monthly_spend / day_of_month
                projected_end_spend = budget.current_monthly_spend + (daily_rate * days_remaining)
            else:
                projected_end_spend = budget.current_monthly_spend

            monthly_percent = (
                float(budget.current_monthly_spend / budget.monthly_limit_usd * 100)
                if budget.monthly_limit_usd and budget.monthly_limit_usd > 0
                else 0
            )

            budget_status = {
                "monthly": {
                    "limit_usd": str(budget.monthly_limit_usd or "0"),
                    "spent_usd": str(budget.current_monthly_spend),
                    "remaining_usd": str(
                        max(
                            Decimal("0"),
                            (budget.monthly_limit_usd or Decimal("0"))
                            - budget.current_monthly_spend,
                        )
                    ),
                    "percent_used": round(monthly_percent, 1),
                    "days_remaining": days_remaining,
                    "projected_end_spend_usd": str(projected_end_spend.quantize(Decimal("0.01"))),
                },
                "daily": {
                    "limit_usd": str(budget.daily_limit_usd or "unlimited"),
                    "spent_usd": str(budget.current_daily_spend),
                },
                "alert_level": (
                    budget.check_alert_level().value if budget.check_alert_level() else None
                ),
            }
        else:
            # No budget set - return workspace stats
            workspace_stats = cost_tracker.get_workspace_stats(org.id)
            total_cost = Decimal(workspace_stats.get("total_cost_usd", "0"))

            budget_status = {
                "monthly": {
                    "limit_usd": "unlimited",
                    "spent_usd": str(total_cost),
                    "remaining_usd": "unlimited",
                    "percent_used": 0,
                    "days_remaining": 30 - now.day,
                    "projected_end_spend_usd": str(total_cost),
                },
                "daily": {
                    "limit_usd": "unlimited",
                    "spent_usd": str(total_cost),
                },
                "alert_level": None,
            }

        return json_response({"budget": budget_status})

    @handle_errors("get usage forecast")
    @require_permission("org:usage:read")
    def _get_forecast(
        self,
        handler,
        query_params: dict,
        user=None,
    ) -> HandlerResult:
        """
        Get usage forecast based on current patterns.

        Query Parameters:
            benchmark: Industry benchmark for ROI projection

        Returns:
            JSON response with forecast data
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        benchmark = get_string_param(handler, "benchmark", "sme")

        # Get current usage patterns
        cost_tracker = self._get_cost_tracker()
        workspace_stats = cost_tracker.get_workspace_stats(org.id)

        total_cost = Decimal(workspace_stats.get("total_cost_usd", "0"))
        api_calls = workspace_stats.get("total_api_calls", 0)

        # Estimate debates per month
        estimated_debates = max(1, api_calls // 10) if api_calls > 0 else 5
        cost_per_debate = (
            total_cost / estimated_debates if estimated_debates > 0 else Decimal("0.50")
        )

        # Get ROI calculator for projections
        calculator = self._get_roi_calculator(benchmark)
        projections = calculator.estimate_future_savings(
            projected_debates_per_month=estimated_debates,
            current_cost_per_debate=cost_per_debate,
        )

        return json_response({"forecast": projections})

    @handle_errors("get benchmarks")
    @require_permission("org:usage:read")
    def _get_benchmarks(
        self,
        handler,
        query_params: dict,
        user=None,
    ) -> HandlerResult:
        """
        Get industry benchmark comparison data.

        Returns:
            JSON response with all available benchmarks
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        calculator = self._get_roi_calculator()
        benchmarks = calculator.get_benchmark_comparison()

        return json_response({"benchmarks": benchmarks})

    @handle_errors("export usage")
    @require_permission("org:usage:read")
    def _export_usage(
        self,
        handler,
        query_params: dict,
        user=None,
    ) -> HandlerResult:
        """
        Export usage data in various formats.

        Query Parameters:
            format: Export format (csv, json, pdf)
            period: Time period
            start: Custom start date
            end: Custom end date
            include_roi: Include ROI metrics (true/false)

        Returns:
            File download or JSON response
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        start_date, end_date, period = self._parse_period(handler)
        export_format = get_string_param(handler, "format", "csv")
        include_roi = get_string_param(handler, "include_roi", "false") == "true"

        # Gather all data
        cost_tracker = self._get_cost_tracker()
        workspace_stats = cost_tracker.get_workspace_stats(org.id)

        # Build export data
        total_cost = Decimal(workspace_stats.get("total_cost_usd", "0"))
        total_tokens = workspace_stats.get("total_tokens_in", 0) + workspace_stats.get(
            "total_tokens_out", 0
        )
        api_calls = workspace_stats.get("total_api_calls", 0)
        by_agent = workspace_stats.get("cost_by_agent", {})
        by_model = workspace_stats.get("cost_by_model", {})

        if export_format == "json":
            export_data = {
                "organization": org.name if hasattr(org, "name") else str(org.id),
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
                "totals": {
                    "cost_usd": str(total_cost),
                    "tokens": total_tokens,
                    "api_calls": api_calls,
                },
                "by_agent": by_agent,
                "by_model": by_model,
            }

            if include_roi:
                calculator = self._get_roi_calculator()
                export_data["roi"] = calculator.get_benchmark_comparison()

            return json_response(export_data)

        # Build CSV
        output = io.StringIO()
        writer = csv.writer(output)

        org_name = org.name if hasattr(org, "name") else str(org.id)

        # Header
        writer.writerow(["SME Usage Dashboard Export"])
        writer.writerow(["Organization", org_name])
        writer.writerow(["Period Start", start_date.isoformat()])
        writer.writerow(["Period End", end_date.isoformat()])
        writer.writerow(["Generated", datetime.now(timezone.utc).isoformat()])
        writer.writerow([])

        # Totals
        writer.writerow(["Summary"])
        writer.writerow(["Total Cost (USD)", str(total_cost)])
        writer.writerow(["Total Tokens", total_tokens])
        writer.writerow(["Total API Calls", api_calls])
        writer.writerow([])

        # By agent
        writer.writerow(["Cost by Agent"])
        writer.writerow(["Agent", "Cost (USD)"])
        for agent, cost in by_agent.items():
            writer.writerow([agent, cost])
        writer.writerow([])

        # By model
        writer.writerow(["Cost by Model"])
        writer.writerow(["Model", "Cost (USD)"])
        for model, cost in by_model.items():
            writer.writerow([model, cost])
        writer.writerow([])

        if include_roi:
            writer.writerow(["ROI Comparison"])
            calculator = self._get_roi_calculator()
            benchmarks = calculator.get_benchmark_comparison()
            writer.writerow(["Benchmark", "Avg Decision Cost", "Hours/Decision", "Participants"])
            for name, data in benchmarks.get("benchmarks", {}).items():
                writer.writerow(
                    [
                        name,
                        data.get("avg_decision_cost_usd", ""),
                        data.get("avg_hours_per_decision", ""),
                        data.get("avg_participants", ""),
                    ]
                )

        csv_content = output.getvalue()
        output.close()

        # Generate filename
        org_slug = org.slug if hasattr(org, "slug") else org.id
        filename = f"sme_usage_{org_slug}_{datetime.now(timezone.utc).strftime('%Y%m%d')}.csv"

        return HandlerResult(
            status_code=200,
            content_type="text/csv",
            body=csv_content.encode("utf-8"),
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )


__all__ = ["SMEUsageDashboardHandler"]
