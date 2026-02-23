"""
Billing API Handlers - Core Module.

Consolidated billing handlers for subscription management, billing operations,
and Stripe webhook handling.

Endpoints:
- GET /api/billing/plans - List available subscription plans
- GET /api/billing/usage - Get current usage for authenticated user
- GET /api/billing/subscription - Get current subscription
- POST /api/billing/checkout - Create checkout session for subscription
- POST /api/billing/portal - Create billing portal session
- POST /api/billing/cancel - Cancel subscription
- POST /api/billing/resume - Resume canceled subscription
- POST /api/webhooks/stripe - Handle Stripe webhooks

Migrated from admin/billing.py as part of handler consolidation.

Implementation is split across submodules for maintainability:
- core_helpers.py: Shared validation utilities (_validate_iso_date, _safe_positive_int)
- core_webhooks.py: Stripe webhook event handlers (WebhookMixin)
- core_reporting.py: Usage export, forecast, invoices, audit log (ReportingMixin)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from aragora.audit.unified import audit_admin, audit_data
from aragora.events.handler_events import emit_handler_event, CREATED, DELETED

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
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    log_request,
)
from ..secure import SecureHandler
from ..utils.decorators import require_permission
from ..utils.rate_limit import RateLimiter, get_client_ip

# Import submodule mixins
from .core_webhooks import WebhookMixin
from .core_reporting import ReportingMixin

# Re-export helpers for backward compatibility
from .core_helpers import (  # noqa: F401
    _ISO_DATE_RE,
    _MAX_EXPORT_ROWS,
    _get_admin_billing_callable,
    _is_duplicate_webhook,
    _mark_webhook_processed,
    _safe_positive_int,
    _validate_iso_date,
)

logger = logging.getLogger(__name__)

# Rate limiter for billing endpoints (20 requests per minute - financial operations)
_billing_limiter = RateLimiter(requests_per_minute=20)


def _get_billing_limiter() -> RateLimiter:
    """Return the billing rate limiter, allowing mock overrides."""
    admin_billing = sys.modules.get("aragora.server.handlers.admin.billing")
    if admin_billing is not None:
        limiter = getattr(admin_billing, "_billing_limiter", None)
        if limiter is not None and not isinstance(limiter, RateLimiter):
            return limiter
    return _billing_limiter


class BillingHandler(WebhookMixin, ReportingMixin, SecureHandler):
    """Handler for billing and subscription endpoints.

    Extends SecureHandler for JWT-based authentication, RBAC permission
    enforcement, and security audit logging.

    Implementation is split across mixins:
    - WebhookMixin: Stripe webhook event handlers (_handle_checkout_completed, etc.)
    - ReportingMixin: Audit log, CSV export, forecast, invoices
    """

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    # Resource type for audit logging
    RESOURCE_TYPE = "billing"

    ROUTES = [
        "/api/v1/billing/plans",
        "/api/v1/billing/usage",
        "/api/v1/billing/subscription",
        "/api/v1/billing/trial",
        "/api/v1/billing/trial/start",
        "/api/v1/billing/checkout",
        "/api/v1/billing/portal",
        "/api/v1/billing/cancel",
        "/api/v1/billing/resume",
        "/api/v1/billing/audit-log",
        "/api/v1/billing/usage/export",
        "/api/v1/billing/usage/forecast",
        "/api/v1/billing/invoices",
        "/api/v1/webhooks/stripe",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(
        self, path: str, query_params: dict[str, Any], handler: Any, method: str = "GET"
    ) -> HandlerResult | None:
        """Route billing requests to appropriate methods."""
        # Rate limit check (skip for webhooks - they have their own idempotency)
        if path != "/api/v1/webhooks/stripe":
            client_ip = get_client_ip(handler)
            test_name = os.environ.get("PYTEST_CURRENT_TEST")
            if test_name:
                client_ip = f"{client_ip}:{test_name}"
            if not _get_billing_limiter().is_allowed(client_ip):
                logger.warning("Rate limit exceeded for billing endpoint: %s", client_ip)
                return error_response("Rate limit exceeded. Please try again later.", 429)

        # Determine HTTP method from handler if not provided
        if hasattr(handler, "command"):
            method = handler.command

        if path == "/api/v1/billing/plans" and method == "GET":
            return self._get_plans()

        if path == "/api/v1/billing/usage" and method == "GET":
            return self._get_usage(handler)

        if path == "/api/v1/billing/subscription" and method == "GET":
            return self._get_subscription(handler)

        if path == "/api/v1/billing/trial" and method == "GET":
            return self._get_trial_status(handler)

        if path == "/api/v1/billing/trial/start" and method == "POST":
            return self._start_trial(handler)

        if path == "/api/v1/billing/checkout" and method == "POST":
            return self._create_checkout(handler)

        if path == "/api/v1/billing/portal" and method == "POST":
            return self._create_portal(handler)

        if path == "/api/v1/billing/cancel" and method == "POST":
            return self._cancel_subscription(handler)

        if path == "/api/v1/billing/resume" and method == "POST":
            return self._resume_subscription(handler)

        if path == "/api/v1/billing/audit-log" and method == "GET":
            return self._get_audit_log(handler)

        if path == "/api/v1/billing/usage/export" and method == "GET":
            return self._export_usage_csv(handler)

        if path == "/api/v1/billing/usage/forecast" and method == "GET":
            return self._get_usage_forecast(handler)

        if path == "/api/v1/billing/invoices" and method == "GET":
            return self._get_invoices(handler)

        if path == "/api/v1/webhooks/stripe" and method == "POST":
            return self._handle_stripe_webhook(handler)

        return error_response("Method not allowed", 405)

    def _get_user_store(self) -> Any | None:
        """Get user store from context."""
        return self.ctx.get("user_store")

    def _get_usage_tracker(self) -> Any | None:
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
    def _get_usage(self, handler: Any, user: Any | None = None) -> HandlerResult:
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
    def _get_subscription(self, handler: Any, user: Any | None = None) -> HandlerResult:
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
                    logger.warning("Failed to get Stripe subscription: %s: %s", type(e).__name__, e)

        return json_response({"subscription": subscription_data})

    @handle_errors("get trial status")
    @require_permission("org:billing")
    def _get_trial_status(self, handler: Any, user: Any | None = None) -> HandlerResult:
        """Get trial status for the organization.

        Requires org:billing permission (owner only).

        Returns:
            Trial status including days remaining, debates used, and expiration.
        """
        from aragora.billing.trial_manager import get_trial_manager

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

        # Get trial status from TrialManager
        trial_mgr = get_trial_manager()
        status = trial_mgr.get_trial_status(org)

        # Build response with conversion options
        trial_data = status.to_dict()

        # Add upgrade options if in trial or trial expired
        if status.is_active or status.is_expired:
            trial_data["upgrade_options"] = [
                {
                    "tier": "starter",
                    "name": "Starter",
                    "price": "$99/month",
                    "debates_per_month": 100,
                },
                {
                    "tier": "professional",
                    "name": "Professional",
                    "price": "$249/month",
                    "debates_per_month": 500,
                },
                {
                    "tier": "enterprise",
                    "name": "Enterprise",
                    "price": "Contact us",
                    "debates_per_month": "Unlimited",
                },
            ]

        # Add warning if trial expiring soon (3 days or less)
        if status.is_active and status.days_remaining <= 3:
            trial_data["warning"] = (
                f"Your trial expires in {status.days_remaining} day(s). "
                "Upgrade now to continue using all features."
            )

        return json_response({"trial": trial_data})

    @handle_errors("start trial")
    @log_request("start trial")
    def _start_trial(self, handler: Any, user: Any | None = None) -> HandlerResult:
        """Start a free trial for a new organization.

        This endpoint can be called by newly registered users to start
        their 7-day free trial with 10 free debates.
        """
        from aragora.billing.trial_manager import TrialManager, get_trial_manager

        user_store = self._get_user_store()
        if not user_store:
            return error_response("Service unavailable", 503)

        # Get user from JWT if available (for authenticated requests)
        # or create a new user from request body (for signup flow)
        db_user = None
        if user:
            db_user = user_store.get_user_by_id(user.user_id)

        if not db_user:
            # Check for signup flow - body should contain user info
            body = self.read_json_body(handler)
            if body is None:
                return error_response("User authentication required", 401)

            user_id = body.get("user_id")
            if user_id:
                db_user = user_store.get_user_by_id(user_id)

        if not db_user:
            return error_response("User not found", 404)

        org = None
        if db_user.org_id:
            org = user_store.get_organization_by_id(db_user.org_id)

        if not org:
            return error_response("No organization found", 404)

        # Check if trial already started
        if org.trial_started_at is not None:
            # Return current trial status instead of error
            trial_mgr = get_trial_manager()
            status = trial_mgr.get_trial_status(org)
            return json_response(
                {
                    "trial": status.to_dict(),
                    "message": "Trial already active",
                }
            )

        # Check if organization has a paid subscription
        if org.tier != SubscriptionTier.FREE:
            return error_response(
                "Cannot start trial - organization already has a paid subscription", 400
            )

        # Start the trial
        trial_mgr = TrialManager(user_store=user_store)
        status = trial_mgr.start_trial(org)

        logger.info("Started trial for org %s (user: %s)", org.id, db_user.email)
        audit_data(
            user_id=db_user.id,
            resource_type="trial",
            resource_id=org.id,
            action="start",
            org_id=org.id,
        )

        emit_handler_event(
            "billing",
            CREATED,
            {"action": "trial_started", "org_id": org.id},
            user_id=db_user.id,
        )

        return json_response(
            {
                "trial": status.to_dict(),
                "message": f"Trial started! You have {status.days_remaining} days and "
                f"{status.debates_remaining} debates.",
            }
        )

    @handle_errors("create checkout")
    @log_request("create checkout session")
    @require_permission("org:billing")
    def _create_checkout(self, handler: Any, user: Any | None = None) -> HandlerResult:
        """Create Stripe checkout session.

        Requires org:billing permission (owner only).
        """
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

            logger.info("Created checkout session %s for user %s", session.id, db_user.email)
            audit_data(
                user_id=db_user.id,
                resource_type="checkout_session",
                resource_id=session.id,
                action="create",
                tier=tier.value,
                org_id=db_user.org_id,
            )

            emit_handler_event(
                "billing",
                CREATED,
                {"action": "checkout", "session_id": session.id},
                user_id=db_user.id,
            )
            return json_response({"checkout": session.to_dict()})

        except StripeConfigError as e:
            logger.error("Stripe checkout failed: %s: %s", type(e).__name__, e)
            return error_response("Payment service unavailable", 503)
        except StripeAPIError as e:
            logger.error("Stripe API error during checkout: %s: %s", type(e).__name__, e)
            return error_response("Failed to create checkout session with payment provider", 502)
        except StripeError as e:
            logger.error("Stripe error during checkout: %s: %s", type(e).__name__, e)
            return error_response("Payment service error", 500)

    @handle_errors("create portal")
    @require_permission("org:billing")
    def _create_portal(self, handler: Any, user: Any | None = None) -> HandlerResult:
        """Create Stripe billing portal session.

        Requires org:billing permission (owner only).
        """
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
            logger.error("Stripe portal failed: %s: %s", type(e).__name__, e)
            return error_response("Payment service unavailable", 503)
        except StripeAPIError as e:
            logger.error("Stripe API error creating portal: %s: %s", type(e).__name__, e)
            return error_response("Failed to create billing portal with payment provider", 502)
        except StripeError as e:
            logger.error("Stripe error creating portal: %s: %s", type(e).__name__, e)
            return error_response("Payment service error", 500)

    def _log_audit(
        self,
        user_store: Any,
        action: str,
        resource_type: str,
        resource_id: str | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
        old_value: dict[str, Any] | None = None,
        new_value: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        handler: Any | None = None,
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
        except (AttributeError, OSError) as e:
            logger.warning("Failed to log audit event: %s", e)

    @handle_errors("cancel subscription")
    @log_request("cancel subscription")
    @require_permission("org:billing")
    def _cancel_subscription(self, handler: Any, user: Any | None = None) -> HandlerResult:
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

            logger.info("Subscription canceled for org %s (user: %s)", org.id, db_user.email)
            audit_admin(
                admin_id=user.user_id,
                action="cancel_subscription",
                target_type="subscription",
                target_id=org.stripe_subscription_id,
                org_id=org.id,
                tier=org.tier.value,
            )

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

            emit_handler_event(
                "billing", DELETED, {"action": "subscription_canceled"}, user_id=user.user_id
            )
            return json_response(
                {
                    "message": "Subscription will be canceled at period end",
                    "subscription": subscription.to_dict(),
                }
            )

        except StripeConfigError as e:
            logger.error("Stripe config error canceling subscription: %s: %s", type(e).__name__, e)
            return error_response("Payment service unavailable", 503)
        except StripeAPIError as e:
            logger.error("Stripe API error canceling subscription: %s: %s", type(e).__name__, e)
            return error_response("Failed to cancel subscription with payment provider", 502)
        except StripeError as e:
            logger.error("Stripe error canceling subscription: %s: %s", type(e).__name__, e)
            return error_response("Failed to cancel subscription", 500)

    @handle_errors("resume subscription")
    @require_permission("org:billing")
    def _resume_subscription(self, handler: Any, user: Any | None = None) -> HandlerResult:
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

            logger.info("Subscription resumed for org %s (user: %s)", org.id, db_user.email)
            audit_admin(
                admin_id=user.user_id,
                action="resume_subscription",
                target_type="subscription",
                target_id=org.stripe_subscription_id,
                org_id=org.id,
            )

            return json_response(
                {
                    "message": "Subscription resumed",
                    "subscription": subscription.to_dict(),
                }
            )

        except StripeConfigError as e:
            logger.error("Stripe config error resuming subscription: %s: %s", type(e).__name__, e)
            return error_response("Payment service unavailable", 503)
        except StripeAPIError as e:
            logger.error("Stripe API error resuming subscription: %s: %s", type(e).__name__, e)
            return error_response("Failed to resume subscription with payment provider", 502)
        except StripeError as e:
            logger.error("Stripe error resuming subscription: %s: %s", type(e).__name__, e)
            return error_response("Failed to resume subscription", 500)


__all__ = ["BillingHandler", "_billing_limiter"]
