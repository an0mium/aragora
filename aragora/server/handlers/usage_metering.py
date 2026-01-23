"""
Usage Metering API Handlers.

Provides billing usage endpoints for ENTERPRISE_PLUS tier:
- GET /api/v1/billing/usage - Current usage summary
- GET /api/v1/billing/usage/breakdown - Detailed breakdown
- GET /api/v1/billing/limits - Current limits and usage %

Phase 4.3 Implementation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from aragora.billing.models import SubscriptionTier

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

# Rate limiter for usage endpoints (30 requests per minute)
_usage_limiter = RateLimiter(requests_per_minute=30)


class UsageMeteringHandler(SecureHandler):
    """Handler for usage metering endpoints.

    Provides comprehensive usage tracking and billing information
    for ENTERPRISE_PLUS tier customers.
    """

    RESOURCE_TYPE = "billing_usage"

    ROUTES = [
        "/api/v1/billing/usage",
        "/api/v1/billing/usage/breakdown",
        "/api/v1/billing/limits",
        "/api/v1/billing/usage/summary",
        "/api/v1/billing/usage/export",
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
        """Route usage metering requests to appropriate methods."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _usage_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for usage endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # Determine HTTP method from handler if not provided
        if hasattr(handler, "command"):
            method = handler.command

        if path == "/api/v1/billing/usage" and method == "GET":
            return self._get_usage(handler, query_params)

        if path == "/api/v1/billing/usage/summary" and method == "GET":
            return self._get_usage(handler, query_params)

        if path == "/api/v1/billing/usage/breakdown" and method == "GET":
            return self._get_usage_breakdown(handler, query_params)

        if path == "/api/v1/billing/limits" and method == "GET":
            return self._get_limits(handler, query_params)

        if path == "/api/v1/billing/usage/export" and method == "GET":
            return self._export_usage(handler, query_params)

        return error_response("Method not allowed", 405)

    def _get_user_store(self):
        """Get user store from context."""
        return self.ctx.get("user_store")

    def _get_usage_meter(self):
        """Get usage meter instance."""
        from aragora.services.usage_metering import get_usage_meter

        return get_usage_meter()

    def _get_org_tier(self, org) -> str:
        """Get organization tier as string."""
        if org is None:
            return "free"
        if isinstance(org.tier, SubscriptionTier):
            return org.tier.value
        return str(org.tier) if org.tier else "free"

    @handle_errors("get usage")
    @require_permission("org:billing")
    def _get_usage(
        self,
        handler,
        query_params: dict,
        user=None,
    ) -> HandlerResult:
        """
        Get current usage for the authenticated user's organization.

        Query Parameters:
            period: Billing period (hour, day, week, month, quarter, year)
                   Default: month

        Returns:
            JSON response with usage summary:
            {
                "usage": {
                    "period_start": "2025-01-01T00:00:00Z",
                    "period_end": "2025-01-31T23:59:59Z",
                    "period_type": "month",
                    "tokens": {
                        "input": 500000,
                        "output": 250000,
                        "total": 750000,
                        "cost": "12.50"
                    },
                    "counts": {
                        "debates": 45,
                        "api_calls": 1500
                    },
                    "by_provider": {...},
                    "limits": {...},
                    "usage_percent": {...}
                }
            }
        """
        import asyncio

        # Get user and organization
        user_store = self._get_user_store()
        if not user_store:
            return error_response("Service unavailable", 503)

        db_user = user_store.get_user_by_id(user.user_id)
        if not db_user:
            return error_response("User not found", 404)

        org = None
        if db_user.org_id:
            org = user_store.get_organization_by_id(db_user.org_id)

        if not org:
            return error_response("No organization found", 404)

        # Get query parameters
        period = get_string_param(handler, "period", "month")

        # Get tier
        tier = self._get_org_tier(org)

        # Get usage meter
        meter = self._get_usage_meter()

        # Get usage summary
        loop = asyncio.new_event_loop()
        try:
            summary = loop.run_until_complete(
                meter.get_usage_summary(
                    org_id=org.id,
                    period=period,
                    tier=tier,
                )
            )
        finally:
            loop.close()

        return json_response({"usage": summary.to_dict()})

    @handle_errors("get usage breakdown")
    @require_permission("org:billing")
    def _get_usage_breakdown(
        self,
        handler,
        query_params: dict,
        user=None,
    ) -> HandlerResult:
        """
        Get detailed usage breakdown for billing.

        Query Parameters:
            start: Start date (ISO format)
            end: End date (ISO format)

        Returns:
            JSON response with detailed breakdown:
            {
                "breakdown": {
                    "totals": {
                        "cost": "125.50",
                        "tokens": 5000000,
                        "debates": 150,
                        "api_calls": 5000
                    },
                    "by_model": [...],
                    "by_provider": [...],
                    "by_day": [...],
                    "by_user": [...]
                }
            }
        """
        import asyncio

        # Get user and organization
        user_store = self._get_user_store()
        if not user_store:
            return error_response("Service unavailable", 503)

        db_user = user_store.get_user_by_id(user.user_id)
        if not db_user:
            return error_response("User not found", 404)

        org = None
        if db_user.org_id:
            org = user_store.get_organization_by_id(db_user.org_id)

        if not org:
            return error_response("No organization found", 404)

        # Parse date parameters
        start_str = get_string_param(handler, "start", None)
        end_str = get_string_param(handler, "end", None)

        start_date = None
        end_date = None
        if start_str:
            try:
                start_date = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            except ValueError:
                return error_response("Invalid start date format", 400)
        if end_str:
            try:
                end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            except ValueError:
                return error_response("Invalid end date format", 400)

        # Get usage meter
        meter = self._get_usage_meter()

        # Get detailed breakdown
        loop = asyncio.new_event_loop()
        try:
            breakdown = loop.run_until_complete(
                meter.get_usage_breakdown(
                    org_id=org.id,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
        finally:
            loop.close()

        return json_response({"breakdown": breakdown.to_dict()})

    @handle_errors("get limits")
    @require_permission("org:billing")
    def _get_limits(
        self,
        handler,
        query_params: dict,
        user=None,
    ) -> HandlerResult:
        """
        Get current usage limits and utilization percentages.

        Returns:
            JSON response with limits and usage:
            {
                "limits": {
                    "tier": "enterprise_plus",
                    "limits": {
                        "tokens": 999999999,
                        "debates": 999999,
                        "api_calls": 999999
                    },
                    "used": {
                        "tokens": 750000,
                        "debates": 45,
                        "api_calls": 1500
                    },
                    "percent": {
                        "tokens": 0.075,
                        "debates": 0.0045,
                        "api_calls": 0.15
                    },
                    "exceeded": {
                        "tokens": false,
                        "debates": false,
                        "api_calls": false
                    }
                }
            }
        """
        import asyncio

        # Get user and organization
        user_store = self._get_user_store()
        if not user_store:
            return error_response("Service unavailable", 503)

        db_user = user_store.get_user_by_id(user.user_id)
        if not db_user:
            return error_response("User not found", 404)

        org = None
        if db_user.org_id:
            org = user_store.get_organization_by_id(db_user.org_id)

        if not org:
            return error_response("No organization found", 404)

        # Get tier
        tier = self._get_org_tier(org)

        # Get usage meter
        meter = self._get_usage_meter()

        # Get limits
        loop = asyncio.new_event_loop()
        try:
            limits = loop.run_until_complete(
                meter.get_usage_limits(
                    org_id=org.id,
                    tier=tier,
                )
            )
        finally:
            loop.close()

        return json_response({"limits": limits.to_dict()})

    @handle_errors("export usage")
    @require_permission("org:billing")
    def _export_usage(
        self,
        handler,
        query_params: dict,
        user=None,
    ) -> HandlerResult:
        """
        Export usage data as CSV.

        Query Parameters:
            start: Start date (ISO format)
            end: End date (ISO format)
            format: Export format (csv or json), default: csv

        Returns:
            CSV file download or JSON response
        """
        import asyncio
        import csv
        import io

        # Get user and organization
        user_store = self._get_user_store()
        if not user_store:
            return error_response("Service unavailable", 503)

        db_user = user_store.get_user_by_id(user.user_id)
        if not db_user:
            return error_response("User not found", 404)

        org = None
        if db_user.org_id:
            org = user_store.get_organization_by_id(db_user.org_id)

        if not org:
            return error_response("No organization found", 404)

        # Parse date parameters
        start_str = get_string_param(handler, "start", None)
        end_str = get_string_param(handler, "end", None)
        export_format = get_string_param(handler, "format", "csv")

        start_date = None
        end_date = None
        if start_str:
            try:
                start_date = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            except ValueError:
                return error_response("Invalid start date format", 400)
        if end_str:
            try:
                end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            except ValueError:
                return error_response("Invalid end date format", 400)

        # Get usage meter
        meter = self._get_usage_meter()

        # Get detailed breakdown
        loop = asyncio.new_event_loop()
        try:
            breakdown = loop.run_until_complete(
                meter.get_usage_breakdown(
                    org_id=org.id,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
        finally:
            loop.close()

        if export_format == "json":
            return json_response(breakdown.to_dict())

        # Build CSV
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(["Usage Export Report"])
        writer.writerow(["Organization", org.name])
        writer.writerow(["Period Start", breakdown.period_start.isoformat()])
        writer.writerow(["Period End", breakdown.period_end.isoformat()])
        writer.writerow([])

        # Totals
        writer.writerow(["Summary"])
        writer.writerow(["Total Cost (USD)", breakdown.total_cost])
        writer.writerow(["Total Tokens", breakdown.total_tokens])
        writer.writerow(["Total Debates", breakdown.total_debates])
        writer.writerow(["Total API Calls", breakdown.total_api_calls])
        writer.writerow([])

        # By model breakdown
        writer.writerow(["Usage by Model"])
        writer.writerow(
            ["Model", "Input Tokens", "Output Tokens", "Total Tokens", "Cost", "Requests"]
        )
        for item in breakdown.by_model:
            writer.writerow(
                [
                    item.get("model", ""),
                    item.get("input_tokens", 0),
                    item.get("output_tokens", 0),
                    item.get("total_tokens", 0),
                    item.get("cost", "0"),
                    item.get("requests", 0),
                ]
            )
        writer.writerow([])

        # By provider breakdown
        writer.writerow(["Usage by Provider"])
        writer.writerow(["Provider", "Total Tokens", "Cost", "Requests"])
        for item in breakdown.by_provider:
            writer.writerow(
                [
                    item.get("provider", ""),
                    item.get("total_tokens", 0),
                    item.get("cost", "0"),
                    item.get("requests", 0),
                ]
            )
        writer.writerow([])

        # Daily breakdown
        writer.writerow(["Daily Usage"])
        writer.writerow(["Date", "Tokens", "Cost", "Debates", "API Calls"])
        for item in breakdown.by_day:
            writer.writerow(
                [
                    item.get("day", ""),
                    item.get("total_tokens", 0),
                    item.get("cost", "0"),
                    item.get("debates", 0),
                    item.get("api_calls", 0),
                ]
            )

        csv_content = output.getvalue()
        output.close()

        # Return CSV file
        filename = f"usage_export_{org.slug}_{datetime.now(timezone.utc).strftime('%Y%m%d')}.csv"
        return HandlerResult(
            status_code=200,
            content_type="text/csv",
            body=csv_content.encode("utf-8"),
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )


__all__ = ["UsageMeteringHandler"]
