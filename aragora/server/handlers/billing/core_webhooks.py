"""
Billing webhook handlers - Stripe webhook event processing.

Extracted from core.py for maintainability. Contains the webhook
dispatch method and all individual event type handlers.

This module is used as a mixin by BillingHandler in core.py.
"""

from __future__ import annotations

import sys
from typing import Any

from ..base import HandlerResult, error_response, handle_errors, json_response
from .core_helpers import _get_admin_billing_callable, _is_duplicate_webhook, _mark_webhook_processed


def _logger():
    """Resolve the logger from the core module for test mock compatibility."""
    core = sys.modules.get("aragora.server.handlers.billing.core")
    if core is not None:
        return core.logger
    import logging
    return logging.getLogger("aragora.server.handlers.billing.core")


class WebhookMixin:
    """Mixin providing Stripe webhook handling methods for BillingHandler."""

    @handle_errors("stripe webhook")
    def _handle_stripe_webhook(self, handler: Any) -> HandlerResult:
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

        duplicate_checker = _get_admin_billing_callable(
            "_is_duplicate_webhook", _is_duplicate_webhook
        )
        mark_processed = _get_admin_billing_callable(
            "_mark_webhook_processed", _mark_webhook_processed
        )

        # Get event ID for idempotency check (use top-level Stripe event ID)
        event_id = event.event_id
        if not event_id:
            _logger().warning("Webhook event missing ID, cannot check idempotency")
        elif duplicate_checker(event_id):
            _logger().info(f"Skipping duplicate webhook event: {event_id}")
            return json_response({"received": True, "duplicate": True})

        _logger().info(f"Received Stripe webhook: {event.type} (id={event_id})")

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

        elif event.type == "invoice.finalized":
            result = self._handle_invoice_finalized(event, user_store)

        else:
            # Acknowledge unhandled events
            result = json_response({"received": True})

        # Mark event as processed (only for successful handling)
        if event_id and result and result.status_code < 400:
            mark_processed(event_id)

        return result

    def _handle_checkout_completed(self, event: Any, user_store: Any) -> HandlerResult:
        """Handle checkout.session.completed event."""
        from aragora.billing.models import SubscriptionTier

        session = event.object
        metadata = event.metadata

        user_id = metadata.get("user_id")
        org_id = metadata.get("org_id")
        tier_str = metadata.get("tier", "starter")

        customer_id = session.get("customer")
        subscription_id = session.get("subscription")

        _logger().info(
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
                _logger().info(f"Updated org {org_id} with subscription, tier={tier.value}")

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

    def _handle_subscription_created(self, event: Any, user_store: Any) -> HandlerResult:
        """Handle customer.subscription.created event."""
        _logger().info(f"Subscription created: {event.subscription_id}")
        return json_response({"received": True})

    def _handle_subscription_updated(self, event: Any, user_store: Any) -> HandlerResult:
        """Handle customer.subscription.updated event."""
        from aragora.billing.stripe_client import get_tier_from_price_id

        subscription = event.object
        subscription_id = subscription.get("id")
        status = subscription.get("status")
        cancel_at_period_end = subscription.get("cancel_at_period_end", False)

        # Get price ID from items
        items = subscription.get("items", {}).get("data", [])
        price_id = items[0].get("price", {}).get("id", "") if items else ""

        _logger().info(
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
                    _logger().info(f"Updated org {org.id} tier from subscription update")

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

    def _handle_subscription_deleted(self, event: Any, user_store: Any) -> HandlerResult:
        """Handle customer.subscription.deleted event."""
        from aragora.billing.models import SubscriptionTier

        subscription = event.object
        subscription_id = subscription.get("id")

        _logger().info(f"Subscription deleted: {subscription_id}")

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
                _logger().info(f"Downgraded org {org.id} to FREE tier after subscription deletion")

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

    def _handle_invoice_paid(self, event: Any, user_store: Any) -> HandlerResult:
        """Handle invoice.payment_succeeded event."""
        from aragora.billing.payment_recovery import get_recovery_store

        invoice = event.object
        customer_id = invoice.get("customer")
        subscription_id = invoice.get("subscription")
        amount_paid = invoice.get("amount_paid") or 0

        _logger().info(
            f"Invoice paid: customer={customer_id}, "
            f"subscription={subscription_id}, amount={amount_paid / 100:.2f}"
        )

        # Reset monthly usage counters on successful payment
        if user_store and customer_id:
            org = user_store.get_organization_by_stripe_customer(customer_id)
            if org:
                try:
                    user_store.reset_org_usage(org.id)
                    _logger().info(f"Reset usage for org {org.id} after invoice payment")
                except (AttributeError, ValueError, OSError) as e:
                    _logger().error(f"Failed to reset usage for org {org.id}: {e}")

                # Mark any active payment failure as recovered
                try:
                    recovery_store = get_recovery_store()
                    if recovery_store.mark_recovered(org.id):
                        _logger().info(f"Payment recovered for org {org.id}")
                except (AttributeError, ValueError, OSError) as e:
                    _logger().error(f"Failed to update recovery status for org {org.id}: {e}")

        return json_response({"received": True})

    def _handle_invoice_failed(self, event: Any, user_store: Any) -> HandlerResult:
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

        _logger().warning(
            f"Invoice payment failed: customer={customer_id}, "
            f"subscription={subscription_id}, attempt={attempt_count}"
        )

        # Record failure and get tracking info
        failure = None
        if user_store and customer_id:
            org = user_store.get_organization_by_stripe_customer(customer_id)
            if org:
                # Record in payment recovery store
                try:
                    recovery_store = get_recovery_store()
                    failure = recovery_store.record_failure(
                        org_id=org.id,
                        stripe_customer_id=customer_id,
                        stripe_subscription_id=subscription_id,
                        invoice_id=invoice_id,
                        invoice_url=hosted_invoice_url,
                    )
                except (AttributeError, IOError, OSError) as e:
                    _logger().error(f"Failed to record payment failure for org {org.id}: {e}")
                    failure = None

                if failure:
                    _logger().info(
                        f"Payment failure recorded for org {org.id}: "
                        f"attempt={failure.attempt_count}, "
                        f"days_failing={failure.days_failing}, "
                        f"days_until_downgrade={failure.days_until_downgrade}"
                    )

                    # Log warning if nearing grace period end
                    if failure.days_until_downgrade <= 3:
                        _logger().warning(
                            f"Org {org.id} payment grace period ending soon: "
                            f"{failure.days_until_downgrade} days until auto-downgrade"
                        )

                # Send notification to organization owner
                try:
                    owner = user_store.get_organization_owner(org.id)
                    if owner and owner.email:
                        notifier = get_billing_notifier()

                        # Escalate notification severity based on attempt count
                        notify_result = notifier.notify_payment_failed(
                            org_id=org.id,
                            org_name=org.name,
                            email=owner.email,
                            attempt_count=failure.attempt_count if failure else attempt_count,
                            invoice_url=hosted_invoice_url,
                            days_until_downgrade=(
                                failure.days_until_downgrade if failure else None
                            ),
                        )
                        _logger().info(
                            f"Payment failure notification sent to {owner.email}: "
                            f"method={notify_result.method}, success={notify_result.success}"
                        )
                except (AttributeError, IOError, OSError) as e:
                    _logger().error(
                        f"Failed to send payment failure notification for org {org.id}: {e}"
                    )

        return json_response(
            {
                "received": True,
                "failure_tracked": failure is not None,
            }
        )

    def _handle_invoice_finalized(self, event: Any, user_store: Any) -> HandlerResult:
        """Handle invoice.finalized event.

        Flushes any remainder usage that didn't meet the MIN_TOKENS_THRESHOLD
        during regular sync cycles. This ensures all usage is billed before
        the invoice is finalized.
        """
        from aragora.billing.usage_sync import get_usage_sync_service

        invoice = event.object
        customer_id = invoice.get("customer")
        subscription_id = invoice.get("subscription")

        _logger().info(f"Invoice finalized: customer={customer_id}, subscription={subscription_id}")

        # Flush remainder usage for the org
        flushed_records = []
        if user_store and customer_id:
            org = user_store.get_organization_by_stripe_customer(customer_id)
            if org:
                try:
                    usage_sync = get_usage_sync_service()
                    flushed_records = usage_sync.flush_period(org_id=org.id)
                    if flushed_records:
                        _logger().info(
                            f"Flushed {len(flushed_records)} usage records for org {org.id} "
                            f"on invoice finalize"
                        )
                except (AttributeError, IOError, OSError) as e:
                    _logger().error(f"Failed to flush usage on invoice finalize: {e}")

        return json_response(
            {
                "received": True,
                "usage_flushed": len(flushed_records),
            }
        )
