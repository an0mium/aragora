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

import json
import logging
from datetime import datetime
from typing import Any, Optional

from .base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    log_request,
    get_string_param,
)

logger = logging.getLogger(__name__)


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
        "/api/webhooks/stripe",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(
        self, path: str, query_params: dict, handler, method: str = "GET"
    ) -> Optional[HandlerResult]:
        """Route billing requests to appropriate methods."""
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
        from aragora.billing.models import TIER_LIMITS, SubscriptionTier

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
    def _get_usage(self, handler) -> HandlerResult:
        """Get usage for authenticated user."""
        from aragora.billing.jwt_auth import extract_user_from_request

        # Get current user
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        # Get user store
        if not user_store:
            return error_response("Service unavailable", 503)

        # Get user and organization
        user = user_store.get_user_by_id(auth_ctx.user_id)
        if not user:
            return error_response("User not found", 404)

        org = None
        if user.org_id:
            org = user_store.get_organization_by_id(user.org_id)

        # Get usage tracker
        usage_tracker = self._get_usage_tracker()

        usage_data = {
            "debates_used": 0,
            "debates_limit": 10,
            "debates_remaining": 10,
            "tokens_used": 0,
            "estimated_cost_usd": 0.0,
            "period_start": None,
            "period_end": None,
        }

        if org:
            usage_data["debates_used"] = org.debates_used_this_month
            usage_data["debates_limit"] = org.limits.debates_per_month
            usage_data["debates_remaining"] = org.debates_remaining
            usage_data["period_start"] = org.billing_cycle_start.isoformat()

        # Get detailed usage from tracker if available
        if usage_tracker and user.org_id:
            summary = usage_tracker.get_summary(
                org_id=user.org_id,
                start_time=org.billing_cycle_start if org else None,
            )
            if summary:
                usage_data["tokens_used"] = summary.total_tokens
                usage_data["estimated_cost_usd"] = float(summary.total_cost)
                usage_data["debates_by_provider"] = summary.by_provider

        return json_response({"usage": usage_data})

    @handle_errors("get subscription")
    def _get_subscription(self, handler) -> HandlerResult:
        """Get current subscription for authenticated user."""
        from aragora.billing.jwt_auth import extract_user_from_request

        # Get current user
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        # Get user store
        if not user_store:
            return error_response("Service unavailable", 503)

        # Get user and organization
        user = user_store.get_user_by_id(auth_ctx.user_id)
        if not user:
            return error_response("User not found", 404)

        org = None
        if user.org_id:
            org = user_store.get_organization_by_id(user.org_id)

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
                from aragora.billing.stripe_client import get_stripe_client

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
                        subscription_data["cancel_at_period_end"] = (
                            stripe_sub.cancel_at_period_end
                        )
                except Exception as e:
                    logger.warning(f"Failed to get Stripe subscription: {e}")

        return json_response({"subscription": subscription_data})

    @handle_errors("create checkout")
    @log_request("create checkout session")
    def _create_checkout(self, handler) -> HandlerResult:
        """Create Stripe checkout session."""
        from aragora.billing.jwt_auth import extract_user_from_request
        from aragora.billing.models import SubscriptionTier
        from aragora.billing.stripe_client import (
            get_stripe_client,
            StripeConfigError,
        )

        # Get current user
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        # Parse request body
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        tier_str = body.get("tier", "").lower()
        success_url = body.get("success_url", "")
        cancel_url = body.get("cancel_url", "")

        if not tier_str:
            return error_response("Tier is required", 400)
        if not success_url or not cancel_url:
            return error_response("Success and cancel URLs required", 400)

        # Validate tier
        try:
            tier = SubscriptionTier(tier_str)
        except ValueError:
            return error_response(f"Invalid tier: {tier_str}", 400)

        if tier == SubscriptionTier.FREE:
            return error_response("Cannot checkout free tier", 400)

        # Get user store
        if not user_store:
            return error_response("Service unavailable", 503)

        # Get user
        user = user_store.get_user_by_id(auth_ctx.user_id)
        if not user:
            return error_response("User not found", 404)

        # Get or create Stripe customer
        org = None
        customer_id = None
        if user.org_id:
            org = user_store.get_organization_by_id(user.org_id)
            if org:
                customer_id = org.stripe_customer_id

        try:
            stripe = get_stripe_client()

            # Create checkout session
            session = stripe.create_checkout_session(
                tier=tier,
                customer_email=user.email,
                success_url=success_url,
                cancel_url=cancel_url,
                customer_id=customer_id,
                metadata={
                    "user_id": user.id,
                    "org_id": user.org_id or "",
                    "tier": tier.value,
                },
            )

            logger.info(
                f"Created checkout session {session.id} for user {user.email}"
            )

            return json_response({"checkout": session.to_dict()})

        except StripeConfigError as e:
            return error_response(str(e), 503)

    @handle_errors("create portal")
    def _create_portal(self, handler) -> HandlerResult:
        """Create Stripe billing portal session."""
        from aragora.billing.jwt_auth import extract_user_from_request
        from aragora.billing.stripe_client import (
            get_stripe_client,
            StripeConfigError,
        )

        # Get current user
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        # Parse request body
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        return_url = body.get("return_url", "")
        if not return_url:
            return error_response("Return URL required", 400)

        # Get user store
        if not user_store:
            return error_response("Service unavailable", 503)

        # Get user and organization
        user = user_store.get_user_by_id(auth_ctx.user_id)
        if not user or not user.org_id:
            return error_response("No organization found", 404)

        org = user_store.get_organization_by_id(user.org_id)
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
            return error_response(str(e), 503)

    @handle_errors("cancel subscription")
    @log_request("cancel subscription")
    def _cancel_subscription(self, handler) -> HandlerResult:
        """Cancel subscription at end of billing period."""
        from aragora.billing.jwt_auth import extract_user_from_request
        from aragora.billing.stripe_client import get_stripe_client

        # Get current user
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        # Only org owners/admins can cancel
        if auth_ctx.role not in ("owner", "admin"):
            return error_response("Only organization owners can cancel", 403)

        # Get user store
        if not user_store:
            return error_response("Service unavailable", 503)

        # Get organization
        user = user_store.get_user_by_id(auth_ctx.user_id)
        if not user or not user.org_id:
            return error_response("No organization found", 404)

        org = user_store.get_organization_by_id(user.org_id)
        if not org or not org.stripe_subscription_id:
            return error_response("No active subscription", 404)

        try:
            stripe = get_stripe_client()
            subscription = stripe.cancel_subscription(
                org.stripe_subscription_id,
                at_period_end=True,  # Cancel at end of period, not immediately
            )

            logger.info(
                f"Subscription canceled for org {org.id} "
                f"(user: {user.email})"
            )

            return json_response(
                {
                    "message": "Subscription will be canceled at period end",
                    "subscription": subscription.to_dict(),
                }
            )

        except Exception as e:
            logger.error(f"Failed to cancel subscription: {e}")
            return error_response("Failed to cancel subscription", 500)

    @handle_errors("resume subscription")
    def _resume_subscription(self, handler) -> HandlerResult:
        """Resume a canceled subscription."""
        from aragora.billing.jwt_auth import extract_user_from_request
        from aragora.billing.stripe_client import get_stripe_client

        # Get current user
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        if auth_ctx.role not in ("owner", "admin"):
            return error_response("Only organization owners can resume", 403)

        # Get user store
        user_store = self._get_user_store()
        if not user_store:
            return error_response("Service unavailable", 503)

        # Get organization
        user = user_store.get_user_by_id(auth_ctx.user_id)
        if not user or not user.org_id:
            return error_response("No organization found", 404)

        org = user_store.get_organization_by_id(user.org_id)
        if not org or not org.stripe_subscription_id:
            return error_response("No subscription to resume", 404)

        try:
            stripe = get_stripe_client()
            subscription = stripe.resume_subscription(org.stripe_subscription_id)

            logger.info(f"Subscription resumed for org {org.id}")

            return json_response(
                {
                    "message": "Subscription resumed",
                    "subscription": subscription.to_dict(),
                }
            )

        except Exception as e:
            logger.error(f"Failed to resume subscription: {e}")
            return error_response("Failed to resume subscription", 500)

    @handle_errors("stripe webhook")
    def _handle_stripe_webhook(self, handler) -> HandlerResult:
        """Handle Stripe webhook events."""
        from aragora.billing.stripe_client import (
            parse_webhook_event,
            get_tier_from_price_id,
        )
        from aragora.billing.models import SubscriptionTier

        # Get raw body and signature
        try:
            content_length = int(handler.headers.get("Content-Length", 0))
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

        logger.info(f"Received Stripe webhook: {event.type}")

        # Get user store
        user_store = self._get_user_store()

        # Handle different event types
        if event.type == "checkout.session.completed":
            return self._handle_checkout_completed(event, user_store)

        elif event.type == "customer.subscription.created":
            return self._handle_subscription_created(event, user_store)

        elif event.type == "customer.subscription.updated":
            return self._handle_subscription_updated(event, user_store)

        elif event.type == "customer.subscription.deleted":
            return self._handle_subscription_deleted(event, user_store)

        elif event.type == "invoice.payment_succeeded":
            return self._handle_invoice_paid(event, user_store)

        elif event.type == "invoice.payment_failed":
            return self._handle_invoice_failed(event, user_store)

        # Acknowledge unhandled events
        return json_response({"received": True})

    def _handle_checkout_completed(self, event, user_store) -> HandlerResult:
        """Handle checkout.session.completed event."""
        from aragora.billing.models import Organization, SubscriptionTier

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
                # Update organization with Stripe IDs and tier
                org.stripe_customer_id = customer_id
                org.stripe_subscription_id = subscription_id
                try:
                    org.tier = SubscriptionTier(tier_str)
                except ValueError:
                    org.tier = SubscriptionTier.STARTER
                org.updated_at = datetime.utcnow()
                user_store.save_organization(org)
                logger.info(f"Updated org {org_id} with subscription")

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
        if user_store and price_id:
            tier = get_tier_from_price_id(price_id)
            if tier:
                # Find org by subscription ID
                # (would need to iterate or have an index in production)
                pass

        return json_response({"received": True})

    def _handle_subscription_deleted(self, event, user_store) -> HandlerResult:
        """Handle customer.subscription.deleted event."""
        from aragora.billing.models import SubscriptionTier

        subscription = event.object
        subscription_id = subscription.get("id")

        logger.info(f"Subscription deleted: {subscription_id}")

        # In production, update organization to free tier
        # and clear subscription ID

        return json_response({"received": True})

    def _handle_invoice_paid(self, event, user_store) -> HandlerResult:
        """Handle invoice.payment_succeeded event."""
        invoice = event.object
        customer_id = invoice.get("customer")
        subscription_id = invoice.get("subscription")
        amount_paid = invoice.get("amount_paid", 0)

        logger.info(
            f"Invoice paid: customer={customer_id}, "
            f"subscription={subscription_id}, amount={amount_paid/100:.2f}"
        )

        # Reset monthly usage counters on successful payment
        if user_store:
            # Find org by customer ID and reset usage
            pass

        return json_response({"received": True})

    def _handle_invoice_failed(self, event, user_store) -> HandlerResult:
        """Handle invoice.payment_failed event."""
        invoice = event.object
        customer_id = invoice.get("customer")
        subscription_id = invoice.get("subscription")

        logger.warning(
            f"Invoice payment failed: customer={customer_id}, "
            f"subscription={subscription_id}"
        )

        # In production, send notification to user and possibly
        # downgrade access after grace period

        return json_response({"received": True})


__all__ = ["BillingHandler"]
