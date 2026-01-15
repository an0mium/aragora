"""
Billing API Handlers.

Endpoints:
- GET /api/billing/plans - List available subscription plans
- GET /api/billing/usage - Get current usage for authenticated user
- GET /api/billing/subscription - Get current subscription
- POST /api/billing/checkout - Create checkout session for subscription
- POST /api/billing/portal - Create billing portal session
- POST /api/billing/cancel - Cancel subscription
- POST /api/billing/resume - Resume canceled subscription
- POST /api/webhooks/stripe - Handle Stripe webhooks
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from aragora.billing.jwt_auth import extract_user_from_request

# Module-level imports for test mocking compatibility
from aragora.billing.models import TIER_LIMITS, SubscriptionTier
from aragora.billing.stripe_client import (
    StripeAPIError,
    StripeConfigError,
    StripeError,
    get_stripe_client,
)
from aragora.server.validation.schema import CHECKOUT_SESSION_SCHEMA, validate_against_schema

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    get_string_param,
    handle_errors,
    json_response,
    log_request,
    require_permission,
)
from ..utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiter for billing endpoints (20 requests per minute - financial operations)
_billing_limiter = RateLimiter(requests_per_minute=20)

# Webhook idempotency tracking (persistent SQLite by default)
# Uses aragora.storage.webhook_store for persistence across restarts


def _is_duplicate_webhook(event_id: str) -> bool:
    """Check if webhook event was already processed."""
    from aragora.storage.webhook_store import get_webhook_store

    store = get_webhook_store()
    return store.is_processed(event_id)


def _mark_webhook_processed(event_id: str, result: str = "success") -> None:
    """Mark webhook event as processed."""
    from aragora.storage.webhook_store import get_webhook_store

    store = get_webhook_store()
    store.mark_processed(event_id, result)


class BillingHandler(BaseHandler):
    """Handler for billing and subscription endpoints."""

    ROUTES = [
        "/api/billing/plans",
        "/api/billing/usage",
        "/api/billing/subscription",
        "/api/billing/checkout",
        "/api/billing/portal",
        "/api/billing/cancel",
        "/api/billing/resume",
        "/api/billing/audit-log",
        "/api/billing/usage/export",
        "/api/billing/usage/forecast",
        "/api/billing/invoices",
        "/api/webhooks/stripe",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(
        self, path: str, query_params: dict, handler, method: str = "GET"
    ) -> Optional[HandlerResult]:
        """Route billing requests to appropriate methods."""
        # Rate limit check (skip for webhooks - they have their own idempotency)
        if path != "/api/webhooks/stripe":
            client_ip = get_client_ip(handler)
            if not _billing_limiter.is_allowed(client_ip):
                logger.warning(f"Rate limit exceeded for billing endpoint: {client_ip}")
                return error_response("Rate limit exceeded. Please try again later.", 429)

        # Determine HTTP method from handler if not provided
        if hasattr(handler, "command"):
            method = handler.command

        if path == "/api/billing/plans" and method == "GET":
            return self._get_plans()

        if path == "/api/billing/usage" and method == "GET":
            return self._get_usage(handler)

        if path == "/api/billing/subscription" and method == "GET":
            return self._get_subscription(handler)

        if path == "/api/billing/checkout" and method == "POST":
            return self._create_checkout(handler)

        if path == "/api/billing/portal" and method == "POST":
            return self._create_portal(handler)

        if path == "/api/billing/cancel" and method == "POST":
            return self._cancel_subscription(handler)

        if path == "/api/billing/resume" and method == "POST":
            return self._resume_subscription(handler)

        if path == "/api/billing/audit-log" and method == "GET":
            return self._get_audit_log(handler)

        if path == "/api/billing/usage/export" and method == "GET":
            return self._export_usage_csv(handler)

        if path == "/api/billing/usage/forecast" and method == "GET":
            return self._get_usage_forecast(handler)

        if path == "/api/billing/invoices" and method == "GET":
            return self._get_invoices(handler)

        if path == "/api/webhooks/stripe" and method == "POST":
            return self._handle_stripe_webhook(handler)

        return error_response("Method not allowed", 405)

    def _get_user_store(self):
        """Get user store from context."""
        return self.ctx.get("user_store")

    def _get_usage_tracker(self):
        """Get usage tracker from context."""
        return self.ctx.get("usage_tracker")

    @handle_errors("get plans")
    def _get_plans(self) -> HandlerResult:
        """Get available subscription plans."""
        plans = []
        for tier in SubscriptionTier:
            limits = TIER_LIMITS[tier]
            plans.append(
                {
                    "id": tier.value,
                    "name": tier.name.title(),
                    "price_monthly_cents": limits.price_monthly_cents,
                    "price_monthly": f"${limits.price_monthly_cents / 100:.2f}",
                    "features": {
                        "debates_per_month": limits.debates_per_month,
                        "users_per_org": limits.users_per_org,
                        "api_access": limits.api_access,
                        "all_agents": limits.all_agents,
                        "custom_agents": limits.custom_agents,
                        "sso_enabled": limits.sso_enabled,
                        "audit_logs": limits.audit_logs,
                        "priority_support": limits.priority_support,
                    },
                }
            )

        return json_response({"plans": plans})

    @handle_errors("get usage")
    @require_permission("org:billing")
    def _get_usage(self, handler, user=None) -> HandlerResult:
        """Get usage for authenticated user.

        Requires org:billing permission (owner only).
        """
        # Get user store
        user_store = self._get_user_store()
        if not user_store:
            return error_response("Service unavailable", 503)

        # Get user and organization from user context
        db_user = user_store.get_user_by_id(user.user_id)
        if not db_user:
            return error_response("User not found", 404)

        org = None
        if db_user.org_id:
            org = user_store.get_organization_by_id(db_user.org_id)

        # Get usage tracker
        usage_tracker = self._get_usage_tracker()

        usage_data: dict[str, Any] = {
            "debates_used": 0,
            "debates_limit": 10,
            "debates_remaining": 10,
            "tokens_used": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "estimated_cost_usd": 0.0,
            "cost_breakdown": None,
            "period_start": None,
            "period_end": None,
        }

        if org:
            usage_data["debates_used"] = org.debates_used_this_month
            usage_data["debates_limit"] = org.limits.debates_per_month
            usage_data["debates_remaining"] = org.debates_remaining
            usage_data["period_start"] = org.billing_cycle_start.isoformat()

        # Get detailed usage from tracker if available
        if usage_tracker and db_user.org_id:
            summary = usage_tracker.get_summary(
                org_id=db_user.org_id,
                period_start=org.billing_cycle_start if org else None,
            )
            if summary:
                usage_data["tokens_used"] = summary.total_tokens_in + summary.total_tokens_out
                usage_data["tokens_in"] = summary.total_tokens_in
                usage_data["tokens_out"] = summary.total_tokens_out
                usage_data["estimated_cost_usd"] = float(summary.total_cost_usd)

                # Calculate cost breakdown by token type
                # Using simplified pricing model (average across providers)
                input_cost = (summary.total_tokens_in / 1_000_000) * 2.0  # ~$2/M input
                output_cost = (summary.total_tokens_out / 1_000_000) * 8.0  # ~$8/M output
                usage_data["cost_breakdown"] = {
                    "input_cost": round(input_cost, 4),
                    "output_cost": round(output_cost, 4),
                    "total": round(float(summary.total_cost_usd), 2),
                }

                # Provider breakdown
                usage_data["cost_by_provider"] = {
                    k: str(v) for k, v in summary.cost_by_provider.items()
                }

        return json_response({"usage": usage_data})

    @handle_errors("get subscription")
    @require_permission("org:billing")
    def _get_subscription(self, handler, user=None) -> HandlerResult:
        """Get current subscription for authenticated user.

        Requires org:billing permission (owner only).
        """
        # Get user store
        user_store = self._get_user_store()
        if not user_store:
            return error_response("Service unavailable", 503)

        # Get user and organization from user context
        db_user = user_store.get_user_by_id(user.user_id)
        if not db_user:
            return error_response("User not found", 404)

        org = None
        if db_user.org_id:
            org = user_store.get_organization_by_id(db_user.org_id)

        subscription_data = {
            "tier": "free",
            "status": "active",
            "is_active": True,
        }

        if org:
            subscription_data["tier"] = org.tier.value
            subscription_data["organization"] = {
                "id": org.id,
                "name": org.name,
            }
            subscription_data["limits"] = org.limits.to_dict()

            # Get Stripe subscription if available
            if org.stripe_subscription_id:
                try:
                    stripe = get_stripe_client()
                    stripe_sub = stripe.get_subscription(org.stripe_subscription_id)
                    if stripe_sub:
                        subscription_data["status"] = stripe_sub.status
                        subscription_data["is_active"] = stripe_sub.status in (
                            "active",
                            "trialing",
                        )
                        subscription_data["current_period_end"] = (
                            stripe_sub.current_period_end.isoformat()
                        )
                        subscription_data["cancel_at_period_end"] = stripe_sub.cancel_at_period_end
                        # Include trial information
                        if stripe_sub.trial_start:
                            subscription_data["trial_start"] = stripe_sub.trial_start.isoformat()
                        if stripe_sub.trial_end:
                            subscription_data["trial_end"] = stripe_sub.trial_end.isoformat()
                        subscription_data["is_trialing"] = stripe_sub.is_trialing
                        # Check for payment failures (past_due status)
                        subscription_data["payment_failed"] = stripe_sub.status == "past_due"
                except StripeError as e:
                    # Log Stripe errors but continue with partial subscription data
                    # This allows the endpoint to degrade gracefully when Stripe is unavailable
                    logger.warning(f"Failed to get Stripe subscription: {type(e).__name__}: {e}")

        return json_response({"subscription": subscription_data})

    @handle_errors("create checkout")
    @log_request("create checkout session")
    @require_permission("org:billing")
    def _create_checkout(self, handler, user=None) -> HandlerResult:
        """Create Stripe checkout session.

        Requires org:billing permission (owner only).
        """
        from aragora.billing.stripe_client import StripeConfigError

        # User is authenticated and has billing permission

        # Parse request body
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        # Schema validation for input sanitization
        validation_result = validate_against_schema(body, CHECKOUT_SESSION_SCHEMA)
        if not validation_result.is_valid:
            return error_response(validation_result.error, 400)

        tier_str = body.get("tier", "").lower()
        success_url = body.get("success_url", "")
        cancel_url = body.get("cancel_url", "")

        # Validate tier
        try:
            tier = SubscriptionTier(tier_str)
        except ValueError:
            return error_response(f"Invalid tier: {tier_str}", 400)

        if tier == SubscriptionTier.FREE:
            return error_response("Cannot checkout free tier", 400)

        # Get user store and user
        user_store = self._get_user_store()
        if not user_store:
            return error_response("Service unavailable", 503)

        db_user = user_store.get_user_by_id(user.user_id)
        if not db_user:
            return error_response("User not found", 404)

        # Get or create Stripe customer
        org = None
        customer_id = None
        if db_user.org_id:
            org = user_store.get_organization_by_id(db_user.org_id)
            if org:
                customer_id = org.stripe_customer_id

        try:
            stripe = get_stripe_client()

            # Create checkout session
            session = stripe.create_checkout_session(
                tier=tier,
                customer_email=db_user.email,
                success_url=success_url,
                cancel_url=cancel_url,
                customer_id=customer_id,
                metadata={
                    "user_id": db_user.id,
                    "org_id": db_user.org_id or "",
                    "tier": tier.value,
                },
            )

            logger.info(f"Created checkout session {session.id} for user {db_user.email}")

            return json_response({"checkout": session.to_dict()})

        except StripeConfigError as e:
            logger.error(f"Stripe checkout failed: {type(e).__name__}: {e}")
            return error_response("Payment service unavailable", 503)

    @handle_errors("create portal")
    @require_permission("org:billing")
    def _create_portal(self, handler, user=None) -> HandlerResult:
        """Create Stripe billing portal session.

        Requires org:billing permission (owner only).
        """
        from aragora.billing.stripe_client import StripeConfigError

        # User is authenticated and has billing permission

        # Parse request body
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        return_url = body.get("return_url", "")
        if not return_url:
            return error_response("Return URL required", 400)

        # Get user store
        user_store = self._get_user_store()
        if not user_store:
            return error_response("Service unavailable", 503)

        # Get user and organization from user context
        db_user = user_store.get_user_by_id(user.user_id)
        if not db_user or not db_user.org_id:
            return error_response("No organization found", 404)

        org = user_store.get_organization_by_id(db_user.org_id)
        if not org or not org.stripe_customer_id:
            return error_response("No billing account found", 404)

        try:
            stripe = get_stripe_client()
            session = stripe.create_portal_session(
                customer_id=org.stripe_customer_id,
                return_url=return_url,
            )

            return json_response({"portal": session.to_dict()})

        except StripeConfigError as e:
            logger.error(f"Stripe portal failed: {type(e).__name__}: {e}")
            return error_response("Payment service unavailable", 503)

    def _log_audit(
        self,
        user_store,
        action: str,
        resource_type: str,
        resource_id: str = None,
        user_id: str = None,
        org_id: str = None,
        old_value: dict = None,
        new_value: dict = None,
        metadata: dict = None,
        handler=None,
    ) -> None:
        """Log an audit event for billing operations."""
        if not user_store or not hasattr(user_store, "log_audit_event"):
            return

        ip_address = None
        user_agent = None
        if handler:
            from aragora.server.middleware.auth import extract_client_ip

            ip_address = extract_client_ip(handler)
            user_agent = (
                handler.headers.get("User-Agent", "")[:200] if hasattr(handler, "headers") else None
            )

        try:
            user_store.log_audit_event(
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                user_id=user_id,
                org_id=org_id,
                old_value=old_value,
                new_value=new_value,
                metadata=metadata,
                ip_address=ip_address,
                user_agent=user_agent,
            )
        except Exception as e:
            logger.warning(f"Failed to log audit event: {e}")

    @handle_errors("get audit log")
    @require_permission("admin:audit")
    def _get_audit_log(self, handler, user=None) -> HandlerResult:
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

        # Get query params
        limit = min(int(get_string_param(handler, "limit", "50")), 100)
        offset = int(get_string_param(handler, "offset", "0"))
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
    def _export_usage_csv(self, handler) -> HandlerResult:
        """Export usage data as CSV."""
        import csv
        import io

        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        if not user_store:
            return error_response("Service unavailable", 503)

        user = user_store.get_user_by_id(auth_ctx.user_id)
        if not user or not user.org_id:
            return error_response("No organization found", 404)

        org = user_store.get_organization_by_id(user.org_id)
        if not org:
            return error_response("Organization not found", 404)

        # Get date range from query params
        start_date = get_string_param(handler, "start", None)
        end_date = get_string_param(handler, "end", None)

        # Get usage events from store
        usage_events = []
        if hasattr(user_store, "_transaction"):
            with user_store._transaction() as cursor:
                query = "SELECT * FROM usage_events WHERE org_id = ?"
                params = [org.id]

                if start_date:
                    query += " AND created_at >= ?"
                    params.append(start_date)
                if end_date:
                    query += " AND created_at <= ?"
                    params.append(end_date)

                query += " ORDER BY created_at DESC"
                cursor.execute(query, params)
                usage_events = cursor.fetchall()

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
        filename = f"usage_export_{org.slug}_{datetime.utcnow().strftime('%Y%m%d')}.csv"
        return HandlerResult(
            status_code=200,
            content_type="text/csv",
            body=csv_content.encode("utf-8"),
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )

    @handle_errors("get usage forecast")
    def _get_usage_forecast(self, handler) -> HandlerResult:
        """Get usage forecast and cost projection."""
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        if not user_store:
            return error_response("Service unavailable", 503)

        user = user_store.get_user_by_id(auth_ctx.user_id)
        if not user or not user.org_id:
            return error_response("No organization found", 404)

        org = user_store.get_organization_by_id(user.org_id)
        if not org:
            return error_response("Organization not found", 404)

        # Calculate days elapsed in billing cycle
        now = datetime.utcnow()
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
    def _get_invoices(self, handler) -> HandlerResult:
        """Get invoice history from Stripe."""
        from aragora.billing.stripe_client import StripeConfigError

        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        if not user_store:
            return error_response("Service unavailable", 503)

        user = user_store.get_user_by_id(auth_ctx.user_id)
        if not user or not user.org_id:
            return error_response("No organization found", 404)

        org = user_store.get_organization_by_id(user.org_id)
        if not org or not org.stripe_customer_id:
            return error_response("No billing account found", 404)

        limit = min(int(get_string_param(handler, "limit", "10")), 100)

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
                        "amount_due": inv.get("amount_due", 0) / 100,
                        "amount_paid": inv.get("amount_paid", 0) / 100,
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
            logger.error(f"Stripe invoices failed: {type(e).__name__}: {e}")
            return error_response("Payment service unavailable", 503)
        except StripeAPIError as e:
            logger.error(f"Stripe API error getting invoices: {type(e).__name__}: {e}")
            return error_response("Failed to retrieve invoices from payment provider", 502)
        except StripeError as e:
            # Catch any other Stripe errors
            logger.error(f"Stripe error getting invoices: {type(e).__name__}: {e}")
            return error_response("Payment service error", 500)

    @handle_errors("cancel subscription")
    @log_request("cancel subscription")
    @require_permission("org:billing")
    def _cancel_subscription(self, handler, user=None) -> HandlerResult:
        """Cancel subscription at end of billing period.

        Requires org:billing permission (owner only).
        """
        # User is authenticated and has billing permission
        user_store = self._get_user_store()
        if not user_store:
            return error_response("Service unavailable", 503)

        # Get organization from user context
        db_user = user_store.get_user_by_id(user.user_id)
        if not db_user or not db_user.org_id:
            return error_response("No organization found", 404)

        org = user_store.get_organization_by_id(db_user.org_id)
        if not org or not org.stripe_subscription_id:
            return error_response("No active subscription", 404)

        try:
            stripe = get_stripe_client()
            subscription = stripe.cancel_subscription(
                org.stripe_subscription_id,
                at_period_end=True,  # Cancel at end of period, not immediately
            )

            logger.info(f"Subscription canceled for org {org.id} (user: {db_user.email})")

            # Log audit event
            self._log_audit(
                user_store,
                action="subscription.canceled",
                resource_type="subscription",
                resource_id=org.stripe_subscription_id,
                user_id=user.user_id,
                org_id=org.id,
                old_value={"tier": org.tier.value, "status": "active"},
                new_value={"tier": org.tier.value, "status": "canceling"},
                handler=handler,
            )

            return json_response(
                {
                    "message": "Subscription will be canceled at period end",
                    "subscription": subscription.to_dict(),
                }
            )

        except StripeConfigError as e:
            logger.error(f"Stripe config error canceling subscription: {type(e).__name__}: {e}")
            return error_response("Payment service unavailable", 503)
        except StripeAPIError as e:
            logger.error(f"Stripe API error canceling subscription: {type(e).__name__}: {e}")
            return error_response("Failed to cancel subscription with payment provider", 502)
        except StripeError as e:
            logger.error(f"Stripe error canceling subscription: {type(e).__name__}: {e}")
            return error_response("Failed to cancel subscription", 500)

    @handle_errors("resume subscription")
    @require_permission("org:billing")
    def _resume_subscription(self, handler, user=None) -> HandlerResult:
        """Resume a canceled subscription.

        Requires org:billing permission (owner only).
        """
        # User is authenticated and has billing permission
        user_store = self._get_user_store()
        if not user_store:
            return error_response("Service unavailable", 503)

        # Get organization from user context
        db_user = user_store.get_user_by_id(user.user_id)
        if not db_user or not db_user.org_id:
            return error_response("No organization found", 404)

        org = user_store.get_organization_by_id(db_user.org_id)
        if not org or not org.stripe_subscription_id:
            return error_response("No subscription to resume", 404)

        try:
            stripe = get_stripe_client()
            subscription = stripe.resume_subscription(org.stripe_subscription_id)

            logger.info(f"Subscription resumed for org {org.id} (user: {db_user.email})")

            return json_response(
                {
                    "message": "Subscription resumed",
                    "subscription": subscription.to_dict(),
                }
            )

        except StripeConfigError as e:
            logger.error(f"Stripe config error resuming subscription: {type(e).__name__}: {e}")
            return error_response("Payment service unavailable", 503)
        except StripeAPIError as e:
            logger.error(f"Stripe API error resuming subscription: {type(e).__name__}: {e}")
            return error_response("Failed to resume subscription with payment provider", 502)
        except StripeError as e:
            logger.error(f"Stripe error resuming subscription: {type(e).__name__}: {e}")
            return error_response("Failed to resume subscription", 500)

    @handle_errors("stripe webhook")
    def _handle_stripe_webhook(self, handler) -> HandlerResult:
        """Handle Stripe webhook events."""
        from aragora.billing.stripe_client import (
            parse_webhook_event,
        )

        # Get raw body and signature (limit to 1MB for webhook payloads)
        MAX_WEBHOOK_SIZE = 1 * 1024 * 1024
        content_length = self.validate_content_length(handler, max_size=MAX_WEBHOOK_SIZE)
        if content_length is None:
            return error_response("Invalid or too large Content-Length", 400)
        try:
            payload = handler.rfile.read(content_length)
        except (ValueError, AttributeError):
            return error_response("Invalid request", 400)

        signature = handler.headers.get("Stripe-Signature", "")
        if not signature:
            return error_response("Missing signature", 400)

        # Parse and verify webhook
        event = parse_webhook_event(payload, signature)
        if not event:
            return error_response("Invalid webhook signature", 400)

        # Get event ID for idempotency check (use top-level Stripe event ID)
        event_id = event.event_id
        if not event_id:
            logger.warning("Webhook event missing ID, cannot check idempotency")
        elif _is_duplicate_webhook(event_id):
            logger.info(f"Skipping duplicate webhook event: {event_id}")
            return json_response({"received": True, "duplicate": True})

        logger.info(f"Received Stripe webhook: {event.type} (id={event_id})")

        # Get user store
        user_store = self._get_user_store()

        # Handle different event types
        result = None
        if event.type == "checkout.session.completed":
            result = self._handle_checkout_completed(event, user_store)

        elif event.type == "customer.subscription.created":
            result = self._handle_subscription_created(event, user_store)

        elif event.type == "customer.subscription.updated":
            result = self._handle_subscription_updated(event, user_store)

        elif event.type == "customer.subscription.deleted":
            result = self._handle_subscription_deleted(event, user_store)

        elif event.type == "invoice.payment_succeeded":
            result = self._handle_invoice_paid(event, user_store)

        elif event.type == "invoice.payment_failed":
            result = self._handle_invoice_failed(event, user_store)

        else:
            # Acknowledge unhandled events
            result = json_response({"received": True})

        # Mark event as processed (only for successful handling)
        if event_id and result and result.status_code < 400:
            _mark_webhook_processed(event_id)

        return result

    def _handle_checkout_completed(self, event, user_store) -> HandlerResult:
        """Handle checkout.session.completed event."""
        from aragora.billing.models import SubscriptionTier

        session = event.object
        metadata = event.metadata

        user_id = metadata.get("user_id")
        org_id = metadata.get("org_id")
        tier_str = metadata.get("tier", "starter")

        customer_id = session.get("customer")
        subscription_id = session.get("subscription")

        logger.info(
            f"Checkout completed: user={user_id}, org={org_id}, "
            f"customer={customer_id}, subscription={subscription_id}"
        )

        if user_store and org_id:
            org = user_store.get_organization_by_id(org_id)
            if org:
                old_tier = org.tier.value

                # Parse tier
                try:
                    tier = SubscriptionTier(tier_str)
                except ValueError:
                    tier = SubscriptionTier.STARTER

                # Update organization with Stripe IDs and tier
                user_store.update_organization(
                    org_id,
                    stripe_customer_id=customer_id,
                    stripe_subscription_id=subscription_id,
                    tier=tier,
                )
                logger.info(f"Updated org {org_id} with subscription, tier={tier.value}")

                # Log audit event
                self._log_audit(
                    user_store,
                    action="subscription.created",
                    resource_type="subscription",
                    resource_id=subscription_id,
                    user_id=user_id,
                    org_id=org_id,
                    old_value={"tier": old_tier},
                    new_value={"tier": tier.value, "subscription_id": subscription_id},
                    metadata={"checkout_session": session.get("id")},
                )

        return json_response({"received": True})

    def _handle_subscription_created(self, event, user_store) -> HandlerResult:
        """Handle customer.subscription.created event."""
        logger.info(f"Subscription created: {event.subscription_id}")
        return json_response({"received": True})

    def _handle_subscription_updated(self, event, user_store) -> HandlerResult:
        """Handle customer.subscription.updated event."""
        from aragora.billing.stripe_client import get_tier_from_price_id

        subscription = event.object
        subscription_id = subscription.get("id")
        status = subscription.get("status")
        cancel_at_period_end = subscription.get("cancel_at_period_end", False)

        # Get price ID from items
        items = subscription.get("items", {}).get("data", [])
        price_id = items[0].get("price", {}).get("id", "") if items else ""

        logger.info(
            f"Subscription updated: {subscription_id}, "
            f"status={status}, cancel_at_period_end={cancel_at_period_end}"
        )

        # Update organization tier if price changed
        if user_store and subscription_id:
            org = user_store.get_organization_by_subscription(subscription_id)
            if org:
                old_tier = org.tier.value
                updates = {}
                new_tier = None
                if price_id:
                    tier = get_tier_from_price_id(price_id)
                    if tier:
                        updates["tier"] = tier
                        new_tier = tier.value

                if updates:
                    user_store.update_organization(org.id, **updates)
                    logger.info(f"Updated org {org.id} tier from subscription update")

                    # Log audit event for tier change
                    if new_tier and new_tier != old_tier:
                        self._log_audit(
                            user_store,
                            action="subscription.tier_changed",
                            resource_type="subscription",
                            resource_id=subscription_id,
                            org_id=org.id,
                            old_value={"tier": old_tier},
                            new_value={"tier": new_tier, "status": status},
                        )

        return json_response({"received": True})

    def _handle_subscription_deleted(self, event, user_store) -> HandlerResult:
        """Handle customer.subscription.deleted event."""
        from aragora.billing.models import SubscriptionTier

        subscription = event.object
        subscription_id = subscription.get("id")

        logger.info(f"Subscription deleted: {subscription_id}")

        # Downgrade organization to free tier and clear subscription ID
        if user_store and subscription_id:
            org = user_store.get_organization_by_subscription(subscription_id)
            if org:
                old_tier = org.tier.value

                user_store.update_organization(
                    org.id,
                    tier=SubscriptionTier.FREE,
                    stripe_subscription_id=None,
                )
                logger.info(f"Downgraded org {org.id} to FREE tier after subscription deletion")

                # Log audit event
                self._log_audit(
                    user_store,
                    action="subscription.deleted",
                    resource_type="subscription",
                    resource_id=subscription_id,
                    org_id=org.id,
                    old_value={"tier": old_tier, "subscription_id": subscription_id},
                    new_value={"tier": "free", "subscription_id": None},
                )

        return json_response({"received": True})

    def _handle_invoice_paid(self, event, user_store) -> HandlerResult:
        """Handle invoice.payment_succeeded event."""
        from aragora.billing.payment_recovery import get_recovery_store

        invoice = event.object
        customer_id = invoice.get("customer")
        subscription_id = invoice.get("subscription")
        amount_paid = invoice.get("amount_paid", 0)

        logger.info(
            f"Invoice paid: customer={customer_id}, "
            f"subscription={subscription_id}, amount={amount_paid / 100:.2f}"
        )

        # Reset monthly usage counters on successful payment
        if user_store and customer_id:
            org = user_store.get_organization_by_stripe_customer(customer_id)
            if org:
                user_store.reset_org_usage(org.id)
                logger.info(f"Reset usage for org {org.id} after invoice payment")

                # Mark any active payment failure as recovered
                recovery_store = get_recovery_store()
                if recovery_store.mark_recovered(org.id):
                    logger.info(f"Payment recovered for org {org.id}")

        return json_response({"received": True})

    def _handle_invoice_failed(self, event, user_store) -> HandlerResult:
        """Handle invoice.payment_failed event.

        Records failure in recovery store and sends escalating notifications.
        Auto-downgrade is handled by background job checking grace periods.
        """
        from aragora.billing.notifications import get_billing_notifier
        from aragora.billing.payment_recovery import get_recovery_store

        invoice = event.object
        customer_id = invoice.get("customer")
        subscription_id = invoice.get("subscription")
        attempt_count = invoice.get("attempt_count", 1)
        invoice_id = invoice.get("id")
        hosted_invoice_url = invoice.get("hosted_invoice_url")

        logger.warning(
            f"Invoice payment failed: customer={customer_id}, "
            f"subscription={subscription_id}, attempt={attempt_count}"
        )

        # Record failure and get tracking info
        failure = None
        if user_store and customer_id:
            org = user_store.get_organization_by_stripe_customer(customer_id)
            if org:
                # Record in payment recovery store
                recovery_store = get_recovery_store()
                failure = recovery_store.record_failure(
                    org_id=org.id,
                    stripe_customer_id=customer_id,
                    stripe_subscription_id=subscription_id,
                    invoice_id=invoice_id,
                    invoice_url=hosted_invoice_url,
                )

                logger.info(
                    f"Payment failure recorded for org {org.id}: "
                    f"attempt={failure.attempt_count}, "
                    f"days_failing={failure.days_failing}, "
                    f"days_until_downgrade={failure.days_until_downgrade}"
                )

                # Send notification to organization owner
                owner = user_store.get_organization_owner(org.id)
                if owner and owner.email:
                    notifier = get_billing_notifier()

                    # Escalate notification severity based on attempt count
                    result = notifier.notify_payment_failed(
                        org_id=org.id,
                        org_name=org.name,
                        email=owner.email,
                        attempt_count=failure.attempt_count,
                        invoice_url=hosted_invoice_url,
                        days_until_downgrade=failure.days_until_downgrade,
                    )
                    logger.info(
                        f"Payment failure notification sent to {owner.email}: "
                        f"method={result.method}, success={result.success}"
                    )

                # Log warning if nearing grace period end
                if failure.days_until_downgrade <= 3:
                    logger.warning(
                        f"Org {org.id} payment grace period ending soon: "
                        f"{failure.days_until_downgrade} days until auto-downgrade"
                    )

        return json_response(
            {
                "received": True,
                "failure_tracked": failure is not None,
            }
        )


__all__ = ["BillingHandler"]
