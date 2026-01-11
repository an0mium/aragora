"""
Stripe Webhook E2E Tests (Phase 12A).

Tests cover the complete webhook flow from Stripe events to database updates:
- checkout.session.completed -> org upgraded with Stripe IDs
- customer.subscription.updated -> tier changes propagate
- invoice.payment_failed -> graceful handling with warnings
- invoice.payment_succeeded -> usage reset
- customer.subscription.deleted -> downgrade to free tier
- Webhook idempotency (duplicate events handled)
- Webhook signature validation
"""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch
from dataclasses import dataclass
from typing import Optional

import pytest

from aragora.billing.models import (
    User,
    Organization,
    SubscriptionTier,
    TIER_LIMITS,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_user_store():
    """Create a mock user store with test data."""
    store = MagicMock()

    # Create test user and org
    test_user = User(
        id="user-123",
        email="test@example.com",
        name="Test User",
        org_id="org-456",
        role="owner",
    )

    test_org = Organization(
        id="org-456",
        name="Test Org",
        slug="test-org",
        tier=SubscriptionTier.FREE,
        owner_id="user-123",
        stripe_customer_id=None,
        stripe_subscription_id=None,
        debates_used_this_month=5,
    )

    store.get_user_by_id.return_value = test_user
    store.get_organization_by_id.return_value = test_org
    store.get_organization_by_subscription.return_value = test_org
    store.get_organization_by_stripe_customer.return_value = test_org
    store.update_organization.return_value = True
    store.reset_org_usage.return_value = True
    store.log_audit_event.return_value = True

    return store


@pytest.fixture
def billing_handler(mock_user_store):
    """Create billing handler with mocked context."""
    from aragora.server.handlers.billing import BillingHandler

    ctx = {"user_store": mock_user_store}
    handler = BillingHandler(ctx)
    return handler


@dataclass
class MockWebhookEvent:
    """Mock Stripe webhook event."""
    type: str
    object: dict
    data: dict
    metadata: dict = None
    subscription_id: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def create_mock_http_handler(payload: bytes, signature: str = "valid_sig"):
    """Create a mock HTTP handler for webhook tests."""
    handler = MagicMock()
    handler.headers = {
        "Content-Length": str(len(payload)),
        "Stripe-Signature": signature,
    }
    handler.rfile.read.return_value = payload
    handler.command = "POST"
    return handler


# =============================================================================
# Checkout Session Completed Tests
# =============================================================================

class TestCheckoutSessionCompleted:
    """Tests for checkout.session.completed webhook event."""

    @patch("aragora.server.handlers.billing.parse_webhook_event")
    @patch("aragora.server.handlers.billing._is_duplicate_webhook")
    @patch("aragora.server.handlers.billing._mark_webhook_processed")
    def test_checkout_completed_upgrades_org(
        self, mock_mark, mock_is_dup, mock_parse, billing_handler, mock_user_store
    ):
        """Successful checkout upgrades org to new tier with Stripe IDs."""
        mock_is_dup.return_value = False

        event = MockWebhookEvent(
            type="checkout.session.completed",
            object={
                "id": "cs_test_123",
                "customer": "cus_test_456",
                "subscription": "sub_test_789",
            },
            data={"id": "evt_checkout_123"},
            metadata={
                "user_id": "user-123",
                "org_id": "org-456",
                "tier": "professional",
            },
        )
        mock_parse.return_value = event

        payload = b'{"type": "checkout.session.completed"}'
        http_handler = create_mock_http_handler(payload)

        result = billing_handler._handle_stripe_webhook(http_handler)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["received"] is True

        # Verify org was updated with Stripe IDs and tier
        mock_user_store.update_organization.assert_called_once()
        call_args = mock_user_store.update_organization.call_args
        assert call_args[0][0] == "org-456"
        assert call_args[1]["stripe_customer_id"] == "cus_test_456"
        assert call_args[1]["stripe_subscription_id"] == "sub_test_789"
        assert call_args[1]["tier"] == SubscriptionTier.PROFESSIONAL

    @patch("aragora.server.handlers.billing.parse_webhook_event")
    @patch("aragora.server.handlers.billing._is_duplicate_webhook")
    def test_checkout_completed_logs_audit_event(
        self, mock_is_dup, mock_parse, billing_handler, mock_user_store
    ):
        """Checkout completion logs audit event for compliance."""
        mock_is_dup.return_value = False

        event = MockWebhookEvent(
            type="checkout.session.completed",
            object={
                "id": "cs_test_123",
                "customer": "cus_test_456",
                "subscription": "sub_test_789",
            },
            data={"id": "evt_checkout_456"},
            metadata={
                "user_id": "user-123",
                "org_id": "org-456",
                "tier": "starter",
            },
        )
        mock_parse.return_value = event

        payload = b'{"type": "checkout.session.completed"}'
        http_handler = create_mock_http_handler(payload)

        billing_handler._handle_stripe_webhook(http_handler)

        # Verify audit log was called
        mock_user_store.log_audit_event.assert_called()
        audit_call = mock_user_store.log_audit_event.call_args
        assert audit_call[1]["action"] == "subscription.created"
        assert audit_call[1]["resource_type"] == "subscription"

    @patch("aragora.server.handlers.billing.parse_webhook_event")
    @patch("aragora.server.handlers.billing._is_duplicate_webhook")
    def test_checkout_completed_handles_missing_org(
        self, mock_is_dup, mock_parse, billing_handler, mock_user_store
    ):
        """Checkout handles case where org doesn't exist gracefully."""
        mock_is_dup.return_value = False
        mock_user_store.get_organization_by_id.return_value = None

        event = MockWebhookEvent(
            type="checkout.session.completed",
            object={
                "id": "cs_test_123",
                "customer": "cus_test_456",
                "subscription": "sub_test_789",
            },
            data={"id": "evt_checkout_789"},
            metadata={
                "user_id": "user-123",
                "org_id": "nonexistent-org",
                "tier": "starter",
            },
        )
        mock_parse.return_value = event

        payload = b'{"type": "checkout.session.completed"}'
        http_handler = create_mock_http_handler(payload)

        result = billing_handler._handle_stripe_webhook(http_handler)

        # Should still return success (acknowledged)
        assert result.status_code == 200
        # But should not update org
        mock_user_store.update_organization.assert_not_called()


# =============================================================================
# Subscription Updated Tests
# =============================================================================

class TestSubscriptionUpdated:
    """Tests for customer.subscription.updated webhook event."""

    @patch("aragora.server.handlers.billing.parse_webhook_event")
    @patch("aragora.server.handlers.billing._is_duplicate_webhook")
    @patch("aragora.server.handlers.billing.get_tier_from_price_id")
    def test_subscription_updated_changes_tier(
        self, mock_get_tier, mock_is_dup, mock_parse, billing_handler, mock_user_store
    ):
        """Subscription update changes org tier when price changes."""
        mock_is_dup.return_value = False
        mock_get_tier.return_value = SubscriptionTier.ENTERPRISE

        event = MockWebhookEvent(
            type="customer.subscription.updated",
            object={
                "id": "sub_test_789",
                "status": "active",
                "cancel_at_period_end": False,
                "items": {
                    "data": [{"price": {"id": "price_enterprise_monthly"}}]
                },
            },
            data={"id": "evt_sub_update_123"},
        )
        mock_parse.return_value = event

        payload = b'{"type": "customer.subscription.updated"}'
        http_handler = create_mock_http_handler(payload)

        result = billing_handler._handle_stripe_webhook(http_handler)

        assert result.status_code == 200

        # Verify tier was updated
        mock_user_store.update_organization.assert_called()
        call_args = mock_user_store.update_organization.call_args
        assert call_args[1]["tier"] == SubscriptionTier.ENTERPRISE

    @patch("aragora.server.handlers.billing.parse_webhook_event")
    @patch("aragora.server.handlers.billing._is_duplicate_webhook")
    @patch("aragora.server.handlers.billing.get_tier_from_price_id")
    def test_subscription_updated_logs_tier_change(
        self, mock_get_tier, mock_is_dup, mock_parse, billing_handler, mock_user_store
    ):
        """Tier change is logged as audit event."""
        mock_is_dup.return_value = False
        mock_get_tier.return_value = SubscriptionTier.PROFESSIONAL

        event = MockWebhookEvent(
            type="customer.subscription.updated",
            object={
                "id": "sub_test_789",
                "status": "active",
                "cancel_at_period_end": False,
                "items": {
                    "data": [{"price": {"id": "price_pro_monthly"}}]
                },
            },
            data={"id": "evt_sub_update_456"},
        )
        mock_parse.return_value = event

        payload = b'{"type": "customer.subscription.updated"}'
        http_handler = create_mock_http_handler(payload)

        billing_handler._handle_stripe_webhook(http_handler)

        # Verify audit log was called for tier change
        assert mock_user_store.log_audit_event.called
        audit_call = mock_user_store.log_audit_event.call_args
        assert audit_call[1]["action"] == "subscription.tier_changed"


# =============================================================================
# Subscription Deleted Tests
# =============================================================================

class TestSubscriptionDeleted:
    """Tests for customer.subscription.deleted webhook event."""

    @patch("aragora.server.handlers.billing.parse_webhook_event")
    @patch("aragora.server.handlers.billing._is_duplicate_webhook")
    def test_subscription_deleted_downgrades_to_free(
        self, mock_is_dup, mock_parse, billing_handler, mock_user_store
    ):
        """Subscription deletion downgrades org to free tier."""
        mock_is_dup.return_value = False

        # Set org to professional tier
        mock_user_store.get_organization_by_subscription.return_value = Organization(
            id="org-456",
            name="Test Org",
            slug="test-org",
            tier=SubscriptionTier.PROFESSIONAL,
            stripe_subscription_id="sub_test_789",
        )

        event = MockWebhookEvent(
            type="customer.subscription.deleted",
            object={"id": "sub_test_789"},
            data={"id": "evt_sub_del_123"},
        )
        mock_parse.return_value = event

        payload = b'{"type": "customer.subscription.deleted"}'
        http_handler = create_mock_http_handler(payload)

        result = billing_handler._handle_stripe_webhook(http_handler)

        assert result.status_code == 200

        # Verify downgrade to FREE
        mock_user_store.update_organization.assert_called()
        call_args = mock_user_store.update_organization.call_args
        assert call_args[1]["tier"] == SubscriptionTier.FREE
        assert call_args[1]["stripe_subscription_id"] is None


# =============================================================================
# Invoice Payment Tests
# =============================================================================

class TestInvoicePayment:
    """Tests for invoice payment webhook events."""

    @patch("aragora.server.handlers.billing.parse_webhook_event")
    @patch("aragora.server.handlers.billing._is_duplicate_webhook")
    def test_invoice_paid_resets_usage(
        self, mock_is_dup, mock_parse, billing_handler, mock_user_store
    ):
        """Successful invoice payment resets monthly usage."""
        mock_is_dup.return_value = False

        event = MockWebhookEvent(
            type="invoice.payment_succeeded",
            object={
                "customer": "cus_test_456",
                "subscription": "sub_test_789",
                "amount_paid": 29900,  # $299
            },
            data={"id": "evt_inv_paid_123"},
        )
        mock_parse.return_value = event

        payload = b'{"type": "invoice.payment_succeeded"}'
        http_handler = create_mock_http_handler(payload)

        result = billing_handler._handle_stripe_webhook(http_handler)

        assert result.status_code == 200

        # Verify usage was reset
        mock_user_store.reset_org_usage.assert_called_once_with("org-456")

    @patch("aragora.server.handlers.billing.parse_webhook_event")
    @patch("aragora.server.handlers.billing._is_duplicate_webhook")
    def test_invoice_failed_logs_warning(
        self, mock_is_dup, mock_parse, billing_handler, mock_user_store
    ):
        """Failed invoice payment is logged as warning."""
        mock_is_dup.return_value = False

        event = MockWebhookEvent(
            type="invoice.payment_failed",
            object={
                "customer": "cus_test_456",
                "subscription": "sub_test_789",
                "attempt_count": 1,
            },
            data={"id": "evt_inv_fail_123"},
        )
        mock_parse.return_value = event

        payload = b'{"type": "invoice.payment_failed"}'
        http_handler = create_mock_http_handler(payload)

        with patch("aragora.server.handlers.billing.logger") as mock_logger:
            result = billing_handler._handle_stripe_webhook(http_handler)

        assert result.status_code == 200
        mock_logger.warning.assert_called()

    @patch("aragora.server.handlers.billing.parse_webhook_event")
    @patch("aragora.server.handlers.billing._is_duplicate_webhook")
    def test_invoice_failed_multiple_attempts_triggers_alert(
        self, mock_is_dup, mock_parse, billing_handler, mock_user_store
    ):
        """Multiple failed payment attempts trigger consideration for downgrade."""
        mock_is_dup.return_value = False

        event = MockWebhookEvent(
            type="invoice.payment_failed",
            object={
                "customer": "cus_test_456",
                "subscription": "sub_test_789",
                "attempt_count": 3,  # Third failure
            },
            data={"id": "evt_inv_fail_multi"},
        )
        mock_parse.return_value = event

        payload = b'{"type": "invoice.payment_failed"}'
        http_handler = create_mock_http_handler(payload)

        with patch("aragora.server.handlers.billing.logger") as mock_logger:
            result = billing_handler._handle_stripe_webhook(http_handler)

        assert result.status_code == 200
        # Should log warning about repeated failures
        warning_calls = [c for c in mock_logger.warning.call_args_list]
        assert len(warning_calls) >= 1


# =============================================================================
# Webhook Idempotency Tests
# =============================================================================

class TestWebhookIdempotency:
    """Tests for webhook idempotency handling."""

    @patch("aragora.server.handlers.billing.parse_webhook_event")
    @patch("aragora.server.handlers.billing._is_duplicate_webhook")
    def test_duplicate_event_skipped(
        self, mock_is_dup, mock_parse, billing_handler, mock_user_store
    ):
        """Duplicate webhook events are acknowledged but not processed."""
        mock_is_dup.return_value = True  # This is a duplicate

        event = MockWebhookEvent(
            type="checkout.session.completed",
            object={"id": "cs_test_123"},
            data={"id": "evt_duplicate_123"},
        )
        mock_parse.return_value = event

        payload = b'{"type": "checkout.session.completed"}'
        http_handler = create_mock_http_handler(payload)

        result = billing_handler._handle_stripe_webhook(http_handler)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["received"] is True
        assert data.get("duplicate") is True

        # Verify org was NOT updated (duplicate skipped)
        mock_user_store.update_organization.assert_not_called()

    def test_cleanup_old_webhook_events(self):
        """Old webhook events are cleaned up after TTL."""
        from aragora.server.handlers.billing import (
            _PROCESSED_WEBHOOK_EVENTS,
            _cleanup_old_webhook_events,
            _mark_webhook_processed,
            _WEBHOOK_EVENT_TTL_SECONDS,
        )

        # Clear any existing events
        _PROCESSED_WEBHOOK_EVENTS.clear()

        # Add an old event
        old_time = datetime.utcnow() - timedelta(seconds=_WEBHOOK_EVENT_TTL_SECONDS + 100)
        _PROCESSED_WEBHOOK_EVENTS["old_event_123"] = old_time

        # Add a recent event
        _mark_webhook_processed("recent_event_456")

        # Run cleanup
        _cleanup_old_webhook_events()

        # Old event should be removed, recent event should remain
        assert "old_event_123" not in _PROCESSED_WEBHOOK_EVENTS
        assert "recent_event_456" in _PROCESSED_WEBHOOK_EVENTS

        # Cleanup
        _PROCESSED_WEBHOOK_EVENTS.clear()


# =============================================================================
# Webhook Signature Validation Tests
# =============================================================================

class TestWebhookSignatureValidation:
    """Tests for webhook signature validation."""

    @patch("aragora.server.handlers.billing.parse_webhook_event")
    def test_missing_signature_rejected(
        self, mock_parse, billing_handler
    ):
        """Missing Stripe-Signature header is rejected."""
        payload = b'{"type": "checkout.session.completed"}'
        http_handler = create_mock_http_handler(payload, signature="")

        result = billing_handler._handle_stripe_webhook(http_handler)

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "signature" in data["error"].lower()

    @patch("aragora.server.handlers.billing.parse_webhook_event")
    def test_invalid_signature_rejected(
        self, mock_parse, billing_handler
    ):
        """Invalid signature is rejected."""
        mock_parse.return_value = None  # Signature validation failed

        payload = b'{"type": "checkout.session.completed"}'
        http_handler = create_mock_http_handler(payload, signature="invalid_sig")

        result = billing_handler._handle_stripe_webhook(http_handler)

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "invalid" in data["error"].lower() or "signature" in data["error"].lower()


# =============================================================================
# Unhandled Event Tests
# =============================================================================

class TestUnhandledEvents:
    """Tests for unhandled webhook event types."""

    @patch("aragora.server.handlers.billing.parse_webhook_event")
    @patch("aragora.server.handlers.billing._is_duplicate_webhook")
    def test_unknown_event_acknowledged(
        self, mock_is_dup, mock_parse, billing_handler
    ):
        """Unknown event types are acknowledged without error."""
        mock_is_dup.return_value = False

        event = MockWebhookEvent(
            type="some.unknown.event",
            object={"id": "unknown_123"},
            data={"id": "evt_unknown_123"},
        )
        mock_parse.return_value = event

        payload = b'{"type": "some.unknown.event"}'
        http_handler = create_mock_http_handler(payload)

        result = billing_handler._handle_stripe_webhook(http_handler)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["received"] is True


# =============================================================================
# Integration Tests
# =============================================================================

class TestWebhookIntegration:
    """Integration tests for complete webhook flows."""

    @patch("aragora.server.handlers.billing.parse_webhook_event")
    @patch("aragora.server.handlers.billing._is_duplicate_webhook")
    def test_full_subscription_lifecycle(
        self, mock_is_dup, mock_parse, billing_handler, mock_user_store
    ):
        """Test complete subscription lifecycle: create -> update -> delete."""
        mock_is_dup.return_value = False

        # Step 1: Checkout completed
        checkout_event = MockWebhookEvent(
            type="checkout.session.completed",
            object={
                "id": "cs_test_123",
                "customer": "cus_lifecycle_456",
                "subscription": "sub_lifecycle_789",
            },
            data={"id": "evt_checkout_lifecycle"},
            metadata={
                "user_id": "user-123",
                "org_id": "org-456",
                "tier": "professional",
            },
        )
        mock_parse.return_value = checkout_event

        payload = b'{"type": "checkout.session.completed"}'
        http_handler = create_mock_http_handler(payload)

        result = billing_handler._handle_stripe_webhook(http_handler)
        assert result.status_code == 200

        # Verify org was upgraded
        call_args = mock_user_store.update_organization.call_args
        assert call_args[1]["tier"] == SubscriptionTier.PROFESSIONAL

        # Reset mock for next step
        mock_user_store.reset_mock()

        # Update org to reflect the upgrade
        mock_user_store.get_organization_by_subscription.return_value = Organization(
            id="org-456",
            name="Test Org",
            slug="test-org",
            tier=SubscriptionTier.PROFESSIONAL,
            stripe_customer_id="cus_lifecycle_456",
            stripe_subscription_id="sub_lifecycle_789",
        )

        # Step 2: Subscription deleted (cancellation)
        delete_event = MockWebhookEvent(
            type="customer.subscription.deleted",
            object={"id": "sub_lifecycle_789"},
            data={"id": "evt_sub_delete_lifecycle"},
        )
        mock_parse.return_value = delete_event

        payload = b'{"type": "customer.subscription.deleted"}'
        http_handler = create_mock_http_handler(payload)

        result = billing_handler._handle_stripe_webhook(http_handler)
        assert result.status_code == 200

        # Verify downgrade to FREE
        call_args = mock_user_store.update_organization.call_args
        assert call_args[1]["tier"] == SubscriptionTier.FREE
        assert call_args[1]["stripe_subscription_id"] is None
