"""
Billing reporting handlers - Usage export, forecast, invoices, and audit log.

Extracted from core.py for maintainability. Contains reporting and
analytics-related billing endpoints.

This module is used as a mixin by BillingHandler in core.py.
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from typing import Any

from ..base import (
    HandlerResult,
    error_response,
    get_string_param,
    handle_errors,
    json_response,
)
from ..utils.decorators import require_permission

from .core_helpers import _safe_positive_int, _validate_iso_date, _MAX_EXPORT_ROWS


def _logger():
    """Resolve the logger from the core module for test mock compatibility."""
    core = sys.modules.get("aragora.server.handlers.billing.core")
    if core is not None:
        return core.logger
    import logging
    return logging.getLogger("aragora.server.handlers.billing.core")


class ReportingMixin:
    """Mixin providing reporting/analytics billing methods for BillingHandler."""

    @handle_errors("get audit log")
    @require_permission("admin:audit")
    def _get_audit_log(self, handler: Any, user: Any | None = None) -> HandlerResult:
        """Get billing audit log for organization (Enterprise feature).

        Requires admin:audit permission (admin/owner only).
        """
        user_store = self._get_user_store()
        if not user_store:
            return error_response("Service unavailable", 503)

        # Get organization and check tier
        db_user = user_store.get_user_by_id(user.user_id)
        if not db_user or not db_user.org_id:
            return error_response("No organization found", 404)

        org = user_store.get_organization_by_id(db_user.org_id)
        if not org:
            return error_response("Organization not found", 404)

        # Check if audit logs are enabled for this tier
        if not org.limits.audit_logs:
            return error_response("Audit logs require Enterprise tier", 403)

        # Only admins/owners can view audit logs
        if user.role not in ("owner", "admin"):
            return error_response("Insufficient permissions", 403)

        # Get query params with safe parsing
        limit = _safe_positive_int(get_string_param(handler, "limit", "50"), 50, 100)
        offset = _safe_positive_int(get_string_param(handler, "offset", "0"), 0, 100_000)
        action_filter = get_string_param(handler, "action", None)

        # Get audit entries
        entries = user_store.get_audit_log(
            org_id=org.id,
            action=action_filter,
            resource_type="subscription",
            limit=limit,
            offset=offset,
        )

        total = user_store.get_audit_log_count(
            org_id=org.id,
            action=action_filter,
            resource_type="subscription",
        )

        return json_response(
            {
                "entries": entries,
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        )

    @handle_errors("export usage CSV")
    @require_permission("org:billing")
    def _export_usage_csv(self, handler: Any, user: Any | None = None) -> HandlerResult:
        """Export usage data as CSV.

        Requires org:billing permission (owner only).
        """
        import csv
        import io

        user_store = self._get_user_store()
        if not user_store:
            return error_response("Service unavailable", 503)

        db_user = user_store.get_user_by_id(user.user_id)
        if not db_user or not db_user.org_id:
            return error_response("No organization found", 404)

        org = user_store.get_organization_by_id(db_user.org_id)
        if not org:
            return error_response("Organization not found", 404)

        # Get and validate date range from query params
        start_date = _validate_iso_date(get_string_param(handler, "start", None))
        end_date = _validate_iso_date(get_string_param(handler, "end", None))

        # Get usage events from store using parameterized queries
        usage_events = []
        if hasattr(user_store, "_transaction"):
            try:
                with user_store._transaction() as cursor:
                    query = (
                        "SELECT id, org_id, event_type, count, metadata, created_at "
                        "FROM usage_events WHERE org_id = ?"
                    )
                    params: list[Any] = [org.id]

                    if start_date:
                        query += " AND created_at >= ?"
                        params.append(start_date)
                    if end_date:
                        query += " AND created_at <= ?"
                        params.append(end_date)

                    query += " ORDER BY created_at DESC LIMIT ?"
                    params.append(_MAX_EXPORT_ROWS)
                    cursor.execute(query, params)
                    usage_events = cursor.fetchall()
            except (sqlite3.Error, OSError, ValueError) as e:
                _logger().error(f"Database error exporting usage data for org {org.id}: {e}")
                return error_response("Failed to export usage data", 500)

        # Build CSV
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Date", "Event Type", "Count", "Metadata"])

        for row in usage_events:
            writer.writerow(
                [
                    row[5],  # created_at
                    row[2],  # event_type
                    row[3],  # count
                    row[4],  # metadata
                ]
            )

        # Add summary row
        writer.writerow([])
        writer.writerow(["Summary"])
        writer.writerow(["Organization", org.name])
        writer.writerow(["Tier", org.tier.value])
        writer.writerow(["Debates Used", org.debates_used_this_month])
        writer.writerow(["Debates Limit", org.limits.debates_per_month])
        writer.writerow(["Billing Cycle Start", org.billing_cycle_start.isoformat()])

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

    @handle_errors("get usage forecast")
    @require_permission("org:billing")
    def _get_usage_forecast(self, handler: Any, user: Any | None = None) -> HandlerResult:
        """Get usage forecast and cost projection.

        Requires org:billing permission (owner only).
        """
        user_store = self._get_user_store()
        if not user_store:
            return error_response("Service unavailable", 503)

        db_user = user_store.get_user_by_id(user.user_id)
        if not db_user or not db_user.org_id:
            return error_response("No organization found", 404)

        org = user_store.get_organization_by_id(db_user.org_id)
        if not org:
            return error_response("Organization not found", 404)

        # Calculate days elapsed in billing cycle
        now = datetime.now(timezone.utc)
        days_elapsed = (now - org.billing_cycle_start).days
        days_in_cycle = 30  # Approximate

        if days_elapsed < 1:
            days_elapsed = 1  # Avoid division by zero

        # Calculate run rate
        debates_per_day = org.debates_used_this_month / days_elapsed
        days_remaining = max(0, days_in_cycle - days_elapsed)

        # Project usage
        projected_debates = org.debates_used_this_month + (debates_per_day * days_remaining)
        projected_debates = int(projected_debates)

        # Calculate if will hit limit
        will_hit_limit = projected_debates >= org.limits.debates_per_month
        debates_overage = max(0, projected_debates - org.limits.debates_per_month)

        # Get usage tracker for token/cost data
        usage_tracker = self._get_usage_tracker()
        projected_cost = 0.0
        tokens_per_day = 0

        if usage_tracker:
            summary = usage_tracker.get_summary(
                org_id=org.id,
                start_time=org.billing_cycle_start,
            )
            if summary and days_elapsed > 0:
                tokens_per_day = summary.total_tokens / days_elapsed
                cost_per_day = float(summary.total_cost) / days_elapsed
                projected_cost = float(summary.total_cost) + (cost_per_day * days_remaining)

        # Suggest tier upgrade if hitting limits
        tier_recommendation = None
        from aragora.billing.models import TIER_LIMITS, SubscriptionTier

        if will_hit_limit and org.tier != SubscriptionTier.ENTERPRISE:
            # Find next tier
            tier_order = [
                SubscriptionTier.FREE,
                SubscriptionTier.STARTER,
                SubscriptionTier.PROFESSIONAL,
                SubscriptionTier.ENTERPRISE,
            ]
            current_idx = tier_order.index(org.tier)
            if current_idx < len(tier_order) - 1:
                next_tier = tier_order[current_idx + 1]
                tier_recommendation = {
                    "recommended_tier": next_tier.value,
                    "debates_limit": TIER_LIMITS[next_tier].debates_per_month,
                    "price_monthly": f"${TIER_LIMITS[next_tier].price_monthly_cents / 100:.2f}",
                }

        return json_response(
            {
                "forecast": {
                    "current_usage": {
                        "debates": org.debates_used_this_month,
                        "debates_limit": org.limits.debates_per_month,
                    },
                    "projection": {
                        "debates_end_of_cycle": projected_debates,
                        "debates_per_day": round(debates_per_day, 2),
                        "tokens_per_day": int(tokens_per_day),
                        "cost_end_of_cycle_usd": round(projected_cost, 2),
                    },
                    "days_remaining": days_remaining,
                    "days_elapsed": days_elapsed,
                    "will_hit_limit": will_hit_limit,
                    "debates_overage": debates_overage,
                    "tier_recommendation": tier_recommendation,
                },
            }
        )

    @handle_errors("get invoices")
    @require_permission("org:billing")
    def _get_invoices(self, handler: Any, user: Any | None = None) -> HandlerResult:
        """Get invoice history from Stripe.

        Requires org:billing permission (owner only).
        """
        from aragora.billing.stripe_client import (
            StripeAPIError,
            StripeConfigError,
            StripeError,
        )

        # Resolve get_stripe_client through core module for test mock compatibility
        import aragora.server.handlers.billing.core as _core

        get_stripe_client = _core.get_stripe_client

        user_store = self._get_user_store()
        if not user_store:
            return error_response("Service unavailable", 503)

        db_user = user_store.get_user_by_id(user.user_id)
        if not db_user or not db_user.org_id:
            return error_response("No organization found", 404)

        org = user_store.get_organization_by_id(db_user.org_id)
        if not org or not org.stripe_customer_id:
            return error_response("No billing account found", 404)

        limit = _safe_positive_int(get_string_param(handler, "limit", "10"), 10, 100)

        try:
            stripe = get_stripe_client()
            invoices_data = stripe.list_invoices(
                customer_id=org.stripe_customer_id,
                limit=limit,
            )

            invoices = []
            for inv in invoices_data:
                invoices.append(
                    {
                        "id": inv.get("id"),
                        "number": inv.get("number"),
                        "status": inv.get("status"),
                        "amount_due": (inv.get("amount_due") or 0) / 100,
                        "amount_paid": (inv.get("amount_paid") or 0) / 100,
                        "currency": inv.get("currency", "usd").upper(),
                        "created": datetime.fromtimestamp(inv.get("created", 0)).isoformat(),
                        "period_start": (
                            datetime.fromtimestamp(inv.get("period_start", 0)).isoformat()
                            if inv.get("period_start")
                            else None
                        ),
                        "period_end": (
                            datetime.fromtimestamp(inv.get("period_end", 0)).isoformat()
                            if inv.get("period_end")
                            else None
                        ),
                        "hosted_invoice_url": inv.get("hosted_invoice_url"),
                        "invoice_pdf": inv.get("invoice_pdf"),
                    }
                )

            return json_response({"invoices": invoices})

        except StripeConfigError as e:
            _logger().error(f"Stripe invoices failed: {type(e).__name__}: {e}")
            return error_response("Payment service unavailable", 503)
        except StripeAPIError as e:
            _logger().error(f"Stripe API error getting invoices: {type(e).__name__}: {e}")
            return error_response("Failed to retrieve invoices from payment provider", 502)
        except StripeError as e:
            # Catch any other Stripe errors
            _logger().error(f"Stripe error getting invoices: {type(e).__name__}: {e}")
            return error_response("Payment service error", 500)
