"""Comprehensive tests for billing webhook handlers (aragora/server/handlers/billing/core_webhooks.py).

Tests the WebhookMixin class and all its methods via the BillingHandler.handle() entrypoint:

- _handle_stripe_webhook: Main webhook dispatch
  - Content-Length validation
  - Missing/empty signature
  - Invalid webhook signature (parse_webhook_event returns None)
  - Idempotency: duplicate event skipping
  - Idempotency: missing event_id bypass
  - mark_webhook_processed on success
  - mark_webhook_processed NOT called on error
  - Unhandled event type acknowledgment
  - Payload read errors
  - Oversized payloads

- _handle_checkout_completed:
  - Updates org with Stripe IDs and tier
  - Invalid tier falls back to STARTER
  - Missing org gracefully continues
  - Missing user_store gracefully continues
  - Audit event logged on update

- _handle_subscription_created:
  - Simply acknowledges event

- _handle_subscription_updated:
  - Tier sync from price change
  - Status degradation (past_due, unpaid, incomplete) + audit
  - Canceled status -> downgrade to FREE
  - Re-activation restoring tier from price
  - No changes when price/status unchanged
  - No org found -> no update

- _handle_subscription_deleted:
  - Downgrades org to FREE
  - Clears stripe_subscription_id
  - Audit event logged
  - No org found -> no update

- _handle_invoice_paid:
  - Resets usage counters
  - Marks recovery as recovered
  - Handles recovery store errors gracefully
  - Handles reset_org_usage errors gracefully

- _handle_invoice_failed:
  - Records failure in recovery store
  - Sends notification to owner
  - Escalation when grace period near end
  - Handles recovery store errors gracefully
  - Handles notification errors gracefully
  - No owner -> no notification

- _handle_invoice_finalized:
  - Flushes remainder usage
  - Handles flush errors gracefully
  - No records flushed -> usage_flushed = 0

- Security tests:
  - Path traversal in event metadata
  - Injection attempts in event type
  - Oversized payloads rejected
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

from aragora.billing.models import SubscriptionTier
from aragora.server.handlers.billing.core import BillingHandler, _billing_limiter


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockTierLimits:
    """Mock tier limits for testing."""

    def __init__(self, debates_per_month: int = 10, **kwargs):
        self.debates_per_month = debates_per_month
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> dict:
        return {"debates_per_month": self.debates_per_month}


class MockUser:
    """Mock user for billing webhook tests."""

    def __init__(
        self,
        id: str,
        email: str,
        name: str = "Test User",
        role: str = "member",
        org_id: str | None = None,
    ):
        self.id = id
        self.user_id = id
        self.email = email
        self.name = name
        self.role = role
        self.org_id = org_id


class MockOrganization:
    """Mock organization for billing webhook tests."""

    def __init__(
        self,
        id: str,
        name: str,
        tier: SubscriptionTier = SubscriptionTier.FREE,
        debates_used_this_month: int = 0,
        stripe_customer_id: str | None = None,
        stripe_subscription_id: str | None = None,
        billing_cycle_start: datetime | None = None,
        limits: MockTierLimits | None = None,
    ):
        self.id = id
        self.name = name
        self.tier = tier
        self.debates_used_this_month = debates_used_this_month
        self.stripe_customer_id = stripe_customer_id
        self.stripe_subscription_id = stripe_subscription_id
        self.billing_cycle_start = billing_cycle_start or datetime.now(timezone.utc)
        self.limits = limits or MockTierLimits()

    @property
    def debates_remaining(self) -> int:
        return max(0, self.limits.debates_per_month - self.debates_used_this_month)


class MockUserStore:
    """Mock user store for webhook tests."""

    def __init__(self):
        self._users: dict[str, MockUser] = {}
        self._orgs: dict[str, MockOrganization] = {}
        self._orgs_by_subscription: dict[str, MockOrganization] = {}
        self._orgs_by_customer: dict[str, MockOrganization] = {}
        self._audit_entries: list[dict] = []

    def add_user(self, user: MockUser):
        self._users[user.id] = user

    def add_organization(self, org: MockOrganization):
        self._orgs[org.id] = org
        if org.stripe_subscription_id:
            self._orgs_by_subscription[org.stripe_subscription_id] = org
        if org.stripe_customer_id:
            self._orgs_by_customer[org.stripe_customer_id] = org

    def get_user_by_id(self, user_id: str) -> MockUser | None:
        return self._users.get(user_id)

    def get_organization_by_id(self, org_id: str) -> MockOrganization | None:
        return self._orgs.get(org_id)

    def get_organization_by_subscription(self, subscription_id: str) -> MockOrganization | None:
        return self._orgs_by_subscription.get(subscription_id)

    def get_organization_by_stripe_customer(self, customer_id: str) -> MockOrganization | None:
        return self._orgs_by_customer.get(customer_id)

    def get_organization_owner(self, org_id: str) -> MockUser | None:
        for u in self._users.values():
            if u.org_id == org_id and u.role == "owner":
                return u
        return None

    def update_organization(self, org_id: str, **kwargs) -> MockOrganization | None:
        org = self._orgs.get(org_id)
        if org:
            for key, value in kwargs.items():
                setattr(org, key, value)
        return org

    def reset_org_usage(self, org_id: str):
        org = self._orgs.get(org_id)
        if org:
            org.debates_used_this_month = 0

    def log_audit_event(self, **kwargs):
        self._audit_entries.append(kwargs)


class MockHTTPHandler:
    """Mock HTTP handler for webhook request simulation."""

    def __init__(
        self,
        body: dict | bytes | None = None,
        command: str = "POST",
        signature: str = "",
        content_length: str | None = None,
    ):
        self.command = command
        self.headers: dict[str, str] = {"User-Agent": "test-agent"}
        self.rfile = MagicMock()
        self.client_address = ("127.0.0.1", 12345)
        self.path = ""

        if isinstance(body, bytes):
            raw = body
        elif isinstance(body, dict):
            raw = json.dumps(body).encode()
        else:
            raw = b"{}"

        self.rfile.read.return_value = raw

        if content_length is not None:
            self.headers["Content-Length"] = content_length
        else:
            self.headers["Content-Length"] = str(len(raw))

        if signature:
            self.headers["Stripe-Signature"] = signature


class MockWebhookEvent:
    """Mock Stripe webhook event."""

    def __init__(
        self,
        event_id: str,
        event_type: str,
        object_data: dict | None = None,
        metadata: dict | None = None,
    ):
        self.event_id = event_id
        self.type = event_type
        self.object = object_data or {}
        self.metadata = metadata or {}
        self.subscription_id = object_data.get("id") if object_data else None


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


def _webhook_patches(event, is_duplicate=False, mark_processed_fn=None):
    """Context manager for common webhook test patches."""
    stack = ExitStack()
    stack.enter_context(
        patch("aragora.billing.stripe_client.parse_webhook_event", return_value=event)
    )

    _mark_calls = []

    def default_mark_processed(event_id, result="success"):
        _mark_calls.append((event_id, result))

    mark_fn = mark_processed_fn or default_mark_processed

    def mock_get_callable(name, fallback):
        if name == "_is_duplicate_webhook":
            return lambda event_id: is_duplicate
        if name == "_mark_webhook_processed":
            return mark_fn
        return fallback

    stack.enter_context(
        patch(
            "aragora.server.handlers.billing.core_webhooks._get_admin_billing_callable",
            side_effect=mock_get_callable,
        )
    )
    stack._mark_calls = _mark_calls  # type: ignore[attr-defined]
    return stack


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def user_store():
    """Create a user store with standard test data."""
    store = MockUserStore()
    auth_user = MockUser(id="test-user-001", email="test@example.com", role="owner", org_id="org_1")
    store.add_user(auth_user)

    org = MockOrganization(
        id="org_1",
        name="Test Org",
        tier=SubscriptionTier.STARTER,
        debates_used_this_month=5,
        stripe_customer_id="cus_test_123",
        stripe_subscription_id="sub_test_123",
        limits=MockTierLimits(debates_per_month=100),
    )
    store.add_organization(org)

    return store


@pytest.fixture
def handler(user_store):
    """Create a BillingHandler with a user store in context."""
    return BillingHandler(ctx={"user_store": user_store})


@pytest.fixture
def handler_no_store():
    """BillingHandler without a user store (service unavailable scenario)."""
    return BillingHandler(ctx={})


@pytest.fixture(autouse=True)
def _clear_rate_limiter():
    """Clear the rate limiter between tests to avoid cross-test pollution."""
    _billing_limiter._buckets.clear()
    yield
    _billing_limiter._buckets.clear()


def _make_webhook_http(
    signature: str = "valid_sig", content_length: str | None = None
) -> MockHTTPHandler:
    """Create a standard webhook HTTP handler."""
    return MockHTTPHandler(
        command="POST", signature=signature, content_length=content_length or "200"
    )


# ===========================================================================
# TestWebhookDispatch - Main _handle_stripe_webhook method
# ===========================================================================


class TestWebhookDispatch:
    """Tests for the main Stripe webhook dispatch method."""

    def test_webhook_returns_200_on_valid_event(self, handler):
        event = MockWebhookEvent("evt_001", "some.event")
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200
        assert _body(result)["received"] is True

    def test_missing_signature_returns_400(self, handler):
        http = MockHTTPHandler(command="POST", content_length="100")
        # No signature header set
        result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 400
        assert "signature" in _body(result).get("error", "").lower()

    def test_empty_signature_returns_400(self, handler):
        http = MockHTTPHandler(command="POST", signature="", content_length="100")
        # Signature is empty string
        result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 400

    def test_invalid_webhook_signature_returns_400(self, handler):
        http = _make_webhook_http(signature="invalid")
        with patch("aragora.billing.stripe_client.parse_webhook_event", return_value=None):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 400
        assert "signature" in _body(result).get("error", "").lower()

    def test_invalid_content_length_returns_400(self, handler):
        http = MockHTTPHandler(command="POST", signature="valid", content_length="not_a_number")
        result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 400

    def test_negative_content_length_returns_400(self, handler):
        http = MockHTTPHandler(command="POST", signature="valid", content_length="-1")
        result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 400

    def test_oversized_payload_returns_400(self, handler):
        # Payload larger than 1MB
        http = MockHTTPHandler(
            command="POST", signature="valid", content_length=str(2 * 1024 * 1024)
        )
        result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 400

    def test_exactly_1mb_payload_accepted(self, handler):
        event = MockWebhookEvent("evt_1mb", "some.event")
        http = MockHTTPHandler(
            command="POST", signature="valid", content_length=str(1 * 1024 * 1024)
        )
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200

    def test_zero_content_length_accepted(self, handler):
        # validate_content_length returns 0 which is valid (>=0 and <= max)
        event = MockWebhookEvent("evt_zero", "some.event")
        http = MockHTTPHandler(command="POST", signature="valid", content_length="0")
        # With 0 bytes read, signature will be checked next
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200

    def test_rfile_read_raises_value_error_returns_400(self, handler):
        http = MockHTTPHandler(command="POST", signature="valid", content_length="100")
        http.rfile.read.side_effect = ValueError("bad read")
        result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 400

    def test_rfile_read_raises_attribute_error_returns_400(self, handler):
        http = MockHTTPHandler(command="POST", signature="valid", content_length="100")
        http.rfile.read.side_effect = AttributeError("no rfile")
        result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 400

    def test_get_method_returns_405(self, handler):
        http = MockHTTPHandler(command="GET")
        result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="GET")
        assert _status(result) == 405

    def test_put_method_returns_405(self, handler):
        http = MockHTTPHandler(command="PUT")
        result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="PUT")
        assert _status(result) == 405

    def test_delete_method_returns_405(self, handler):
        http = MockHTTPHandler(command="DELETE")
        result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="DELETE")
        assert _status(result) == 405

    def test_patch_method_returns_405(self, handler):
        http = MockHTTPHandler(command="PATCH")
        result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="PATCH")
        assert _status(result) == 405


# ===========================================================================
# TestWebhookIdempotency - Duplicate detection
# ===========================================================================


class TestWebhookIdempotency:
    """Tests for webhook idempotency handling."""

    def test_duplicate_event_returns_received_with_duplicate_flag(self, handler):
        event = MockWebhookEvent("evt_dup_1", "checkout.session.completed")
        http = _make_webhook_http()
        with _webhook_patches(event, is_duplicate=True):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
            body = _body(result)
        assert _status(result) == 200
        assert body["received"] is True
        assert body["duplicate"] is True

    def test_duplicate_event_does_not_process_checkout(self, handler):
        """Duplicate events should not call _handle_checkout_completed."""
        event = MockWebhookEvent(
            "evt_dup_2",
            "checkout.session.completed",
            object_data={"customer": "cus_test_123", "subscription": "sub_new"},
            metadata={"org_id": "org_1", "tier": "professional"},
        )
        http = _make_webhook_http()
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        old_tier = org.tier

        with _webhook_patches(event, is_duplicate=True):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")

        # Tier should remain unchanged since duplicate was skipped
        assert org.tier == old_tier

    def test_missing_event_id_skips_idempotency_check(self, handler):
        """Events with no event_id should still be processed."""
        event = MockWebhookEvent("", "some.event")
        event.event_id = ""
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200
        assert _body(result)["received"] is True

    def test_none_event_id_skips_idempotency_check(self, handler):
        """Events with None event_id should still be processed."""
        event = MockWebhookEvent("", "some.event")
        event.event_id = None
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200

    def test_successful_event_marked_as_processed(self, handler):
        """After successful handling, event should be marked as processed."""
        mark_calls = []

        def track_mark(event_id, result="success"):
            mark_calls.append(event_id)

        event = MockWebhookEvent("evt_mark_1", "some.event")
        http = _make_webhook_http()
        with _webhook_patches(event, mark_processed_fn=track_mark):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")

        assert "evt_mark_1" in mark_calls

    def test_event_with_no_id_not_marked_as_processed(self, handler):
        """Events without event_id should not be marked as processed."""
        mark_calls = []

        def track_mark(event_id, result="success"):
            mark_calls.append(event_id)

        event = MockWebhookEvent("", "some.event")
        event.event_id = ""
        http = _make_webhook_http()
        with _webhook_patches(event, mark_processed_fn=track_mark):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")

        assert len(mark_calls) == 0


# ===========================================================================
# TestCheckoutCompleted
# ===========================================================================


class TestCheckoutCompleted:
    """Tests for _handle_checkout_completed."""

    def _make_checkout_event(
        self,
        event_id: str = "evt_checkout_1",
        customer: str = "cus_test_123",
        subscription: str = "sub_new_1",
        session_id: str = "cs_1",
        user_id: str = "test-user-001",
        org_id: str = "org_1",
        tier: str = "starter",
    ) -> MockWebhookEvent:
        return MockWebhookEvent(
            event_id=event_id,
            event_type="checkout.session.completed",
            object_data={
                "customer": customer,
                "subscription": subscription,
                "id": session_id,
            },
            metadata={
                "user_id": user_id,
                "org_id": org_id,
                "tier": tier,
            },
        )

    def test_checkout_completed_updates_org_tier(self, handler):
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        org.tier = SubscriptionTier.FREE

        event = self._make_checkout_event(tier="starter")
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200
        assert org.tier == SubscriptionTier.STARTER

    def test_checkout_completed_updates_stripe_ids(self, handler):
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")

        event = self._make_checkout_event(customer="cus_new", subscription="sub_new")
        http = _make_webhook_http()
        with _webhook_patches(event):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert org.stripe_customer_id == "cus_new"
        assert org.stripe_subscription_id == "sub_new"

    def test_checkout_completed_logs_audit_event(self, handler):
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        org.tier = SubscriptionTier.FREE

        event = self._make_checkout_event()
        http = _make_webhook_http()
        with _webhook_patches(event):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert len(store._audit_entries) >= 1
        audit = store._audit_entries[-1]
        assert audit["action"] == "subscription.created"
        assert audit["resource_type"] == "subscription"

    def test_checkout_completed_invalid_tier_falls_back_to_starter(self, handler):
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        org.tier = SubscriptionTier.FREE

        event = self._make_checkout_event(tier="nonexistent_tier")
        http = _make_webhook_http()
        with _webhook_patches(event):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert org.tier == SubscriptionTier.STARTER

    def test_checkout_completed_professional_tier(self, handler):
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        org.tier = SubscriptionTier.FREE

        event = self._make_checkout_event(tier="professional")
        http = _make_webhook_http()
        with _webhook_patches(event):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert org.tier == SubscriptionTier.PROFESSIONAL

    def test_checkout_completed_enterprise_tier(self, handler):
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        org.tier = SubscriptionTier.FREE

        event = self._make_checkout_event(tier="enterprise")
        http = _make_webhook_http()
        with _webhook_patches(event):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert org.tier == SubscriptionTier.ENTERPRISE

    def test_checkout_completed_missing_org_succeeds(self, handler):
        """Event for non-existent org should still return 200."""
        event = self._make_checkout_event(org_id="nonexistent_org")
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200
        assert _body(result)["received"] is True

    def test_checkout_completed_no_user_store(self, handler_no_store):
        """Event with no user_store should still return 200."""
        event = self._make_checkout_event()
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler_no_store.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200

    def test_checkout_completed_missing_org_id_in_metadata(self, handler):
        """Event without org_id in metadata should still succeed."""
        event = MockWebhookEvent(
            event_id="evt_no_org",
            event_type="checkout.session.completed",
            object_data={"customer": "cus_test_123", "subscription": "sub_new"},
            metadata={"user_id": "test-user-001"},
        )
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200

    def test_checkout_completed_default_tier_is_starter(self, handler):
        """When tier key is missing from metadata, default to 'starter'."""
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        org.tier = SubscriptionTier.FREE

        event = MockWebhookEvent(
            event_id="evt_no_tier",
            event_type="checkout.session.completed",
            object_data={"customer": "cus_test_123", "subscription": "sub_new"},
            metadata={"user_id": "test-user-001", "org_id": "org_1"},
        )
        http = _make_webhook_http()
        with _webhook_patches(event):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert org.tier == SubscriptionTier.STARTER

    def test_checkout_completed_audit_includes_old_and_new_tier(self, handler):
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        org.tier = SubscriptionTier.FREE

        event = self._make_checkout_event(tier="professional")
        http = _make_webhook_http()
        with _webhook_patches(event):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        audit = store._audit_entries[-1]
        assert audit["old_value"]["tier"] == "free"
        assert audit["new_value"]["tier"] == "professional"

    def test_checkout_completed_audit_includes_checkout_session(self, handler):
        store = handler.ctx["user_store"]
        event = self._make_checkout_event(session_id="cs_specific_session")
        http = _make_webhook_http()
        with _webhook_patches(event):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        audit = store._audit_entries[-1]
        assert audit["metadata"]["checkout_session"] == "cs_specific_session"


# ===========================================================================
# TestSubscriptionCreated
# ===========================================================================


class TestSubscriptionCreated:
    """Tests for _handle_subscription_created."""

    def test_subscription_created_acknowledged(self, handler):
        event = MockWebhookEvent(
            event_id="evt_sub_created",
            event_type="customer.subscription.created",
            object_data={"id": "sub_new"},
        )
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
            body = _body(result)
        assert _status(result) == 200
        assert body["received"] is True

    def test_subscription_created_does_not_modify_org(self, handler):
        """subscription.created just logs and acknowledges; no org changes."""
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        old_tier = org.tier

        event = MockWebhookEvent(
            event_id="evt_sub_created_2",
            event_type="customer.subscription.created",
            object_data={"id": "sub_test_123"},
        )
        http = _make_webhook_http()
        with _webhook_patches(event):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert org.tier == old_tier


# ===========================================================================
# TestSubscriptionUpdated
# ===========================================================================


class TestSubscriptionUpdated:
    """Tests for _handle_subscription_updated."""

    def _make_sub_updated_event(
        self,
        event_id: str = "evt_sub_upd",
        subscription_id: str = "sub_test_123",
        status: str = "active",
        cancel_at_period_end: bool = False,
        price_id: str = "",
    ) -> MockWebhookEvent:
        items = {"data": [{"price": {"id": price_id}}]} if price_id else {"data": []}
        return MockWebhookEvent(
            event_id=event_id,
            event_type="customer.subscription.updated",
            object_data={
                "id": subscription_id,
                "status": status,
                "cancel_at_period_end": cancel_at_period_end,
                "items": items,
            },
        )

    def test_tier_synced_from_price_change(self, handler):
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        org.tier = SubscriptionTier.FREE

        event = self._make_sub_updated_event(price_id="price_professional")
        http = _make_webhook_http()
        with (
            _webhook_patches(event),
            patch(
                "aragora.billing.stripe_client.get_tier_from_price_id",
                return_value=SubscriptionTier.PROFESSIONAL,
            ),
        ):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200
        assert org.tier == SubscriptionTier.PROFESSIONAL

    def test_tier_change_logs_audit_event(self, handler):
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        org.tier = SubscriptionTier.FREE

        event = self._make_sub_updated_event(price_id="price_starter")
        http = _make_webhook_http()
        with (
            _webhook_patches(event),
            patch(
                "aragora.billing.stripe_client.get_tier_from_price_id",
                return_value=SubscriptionTier.STARTER,
            ),
        ):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        audit = store._audit_entries[-1]
        assert audit["action"] == "subscription.tier_changed"
        assert audit["old_value"]["tier"] == "free"
        assert audit["new_value"]["tier"] == "starter"

    def test_no_tier_change_no_audit(self, handler):
        """If the tier doesn't actually change, no audit event should be logged."""
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        org.tier = SubscriptionTier.STARTER

        event = self._make_sub_updated_event(price_id="price_starter")
        http = _make_webhook_http()
        with (
            _webhook_patches(event),
            patch(
                "aragora.billing.stripe_client.get_tier_from_price_id",
                return_value=SubscriptionTier.STARTER,
            ),
        ):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        # No tier_changed audit since old == new
        tier_audits = [
            a for a in store._audit_entries if a["action"] == "subscription.tier_changed"
        ]
        assert len(tier_audits) == 0

    def test_past_due_status_logs_degradation_audit(self, handler):
        store = handler.ctx["user_store"]
        event = self._make_sub_updated_event(status="past_due")
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200
        degradation_audits = [
            a for a in store._audit_entries if a["action"] == "subscription.status_degraded"
        ]
        assert len(degradation_audits) == 1
        assert degradation_audits[0]["new_value"]["status"] == "past_due"

    def test_unpaid_status_logs_degradation_audit(self, handler):
        store = handler.ctx["user_store"]
        event = self._make_sub_updated_event(status="unpaid")
        http = _make_webhook_http()
        with _webhook_patches(event):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        degradation_audits = [
            a for a in store._audit_entries if a["action"] == "subscription.status_degraded"
        ]
        assert len(degradation_audits) == 1

    def test_incomplete_status_logs_degradation_audit(self, handler):
        store = handler.ctx["user_store"]
        event = self._make_sub_updated_event(status="incomplete")
        http = _make_webhook_http()
        with _webhook_patches(event):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        degradation_audits = [
            a for a in store._audit_entries if a["action"] == "subscription.status_degraded"
        ]
        assert len(degradation_audits) == 1

    def test_canceled_status_downgrades_to_free(self, handler):
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        org.tier = SubscriptionTier.PROFESSIONAL

        event = self._make_sub_updated_event(status="canceled")
        http = _make_webhook_http()
        with _webhook_patches(event):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert org.tier == SubscriptionTier.FREE
        assert org.stripe_subscription_id is None

    def test_canceled_status_logs_tier_changed_audit(self, handler):
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        org.tier = SubscriptionTier.PROFESSIONAL

        event = self._make_sub_updated_event(status="canceled")
        http = _make_webhook_http()
        with _webhook_patches(event):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        tier_audits = [
            a for a in store._audit_entries if a["action"] == "subscription.tier_changed"
        ]
        assert len(tier_audits) == 1
        assert tier_audits[0]["new_value"]["tier"] == "free"

    def test_active_reactivation_restores_tier(self, handler):
        """Active status on a free org restores tier from price."""
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        org.tier = SubscriptionTier.FREE

        event = self._make_sub_updated_event(status="active", price_id="price_starter")
        http = _make_webhook_http()
        with (
            _webhook_patches(event),
            patch(
                "aragora.billing.stripe_client.get_tier_from_price_id",
                return_value=SubscriptionTier.STARTER,
            ),
        ):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert org.tier == SubscriptionTier.STARTER

    def test_active_no_reactivation_when_already_paid(self, handler):
        """Active status on a non-free org with no price change: no tier change."""
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        org.tier = SubscriptionTier.PROFESSIONAL

        event = self._make_sub_updated_event(status="active")
        http = _make_webhook_http()
        with _webhook_patches(event):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        # Tier unchanged since org is not free
        assert org.tier == SubscriptionTier.PROFESSIONAL

    def test_no_org_found_still_returns_200(self, handler):
        event = self._make_sub_updated_event(subscription_id="sub_nonexistent")
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200

    def test_no_user_store_still_returns_200(self, handler_no_store):
        event = self._make_sub_updated_event()
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler_no_store.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200

    def test_no_subscription_id_still_returns_200(self, handler):
        """Event with no subscription ID in object should still return 200."""
        event = MockWebhookEvent(
            event_id="evt_no_sub_id",
            event_type="customer.subscription.updated",
            object_data={"status": "active", "items": {"data": []}},
        )
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200

    def test_empty_items_list_no_price_id(self, handler):
        """When items.data is empty, no price_id should be extracted."""
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        old_tier = org.tier

        event = MockWebhookEvent(
            event_id="evt_empty_items",
            event_type="customer.subscription.updated",
            object_data={
                "id": "sub_test_123",
                "status": "active",
                "items": {"data": []},
            },
        )
        http = _make_webhook_http()
        with _webhook_patches(event):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        # Tier should not change since no price ID
        assert org.tier == old_tier

    def test_price_id_with_no_tier_mapping_no_update(self, handler):
        """Price ID that doesn't map to a tier -> no tier update."""
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        old_tier = org.tier

        event = self._make_sub_updated_event(price_id="price_unknown")
        http = _make_webhook_http()
        with (
            _webhook_patches(event),
            patch(
                "aragora.billing.stripe_client.get_tier_from_price_id",
                return_value=None,
            ),
        ):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert org.tier == old_tier

    def test_degradation_audit_includes_cancel_at_period_end(self, handler):
        event = self._make_sub_updated_event(status="past_due", cancel_at_period_end=True)
        http = _make_webhook_http()
        store = handler.ctx["user_store"]
        with _webhook_patches(event):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        degradation_audits = [
            a for a in store._audit_entries if a["action"] == "subscription.status_degraded"
        ]
        assert degradation_audits[0]["metadata"]["cancel_at_period_end"] is True

    def test_reactivation_with_no_price_no_restore(self, handler):
        """Active + free org + no price_id -> no tier change."""
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        org.tier = SubscriptionTier.FREE

        event = self._make_sub_updated_event(status="active")
        http = _make_webhook_http()
        with _webhook_patches(event):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert org.tier == SubscriptionTier.FREE


# ===========================================================================
# TestSubscriptionDeleted
# ===========================================================================


class TestSubscriptionDeleted:
    """Tests for _handle_subscription_deleted."""

    def _make_deleted_event(
        self,
        event_id: str = "evt_del_1",
        subscription_id: str = "sub_test_123",
    ) -> MockWebhookEvent:
        return MockWebhookEvent(
            event_id=event_id,
            event_type="customer.subscription.deleted",
            object_data={"id": subscription_id},
        )

    def test_downgrades_org_to_free(self, handler):
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        org.tier = SubscriptionTier.PROFESSIONAL

        event = self._make_deleted_event()
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200
        assert org.tier == SubscriptionTier.FREE

    def test_clears_stripe_subscription_id(self, handler):
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        assert org.stripe_subscription_id is not None

        event = self._make_deleted_event()
        http = _make_webhook_http()
        with _webhook_patches(event):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert org.stripe_subscription_id is None

    def test_logs_audit_event(self, handler):
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        org.tier = SubscriptionTier.STARTER

        event = self._make_deleted_event()
        http = _make_webhook_http()
        with _webhook_patches(event):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        audit = store._audit_entries[-1]
        assert audit["action"] == "subscription.deleted"
        assert audit["old_value"]["tier"] == "starter"
        assert audit["new_value"]["tier"] == "free"
        assert audit["new_value"]["subscription_id"] is None

    def test_no_org_found_returns_200(self, handler):
        event = self._make_deleted_event(subscription_id="sub_nonexistent")
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200

    def test_no_user_store_returns_200(self, handler_no_store):
        event = self._make_deleted_event()
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler_no_store.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200

    def test_missing_subscription_id_returns_200(self, handler):
        event = MockWebhookEvent(
            event_id="evt_del_no_id",
            event_type="customer.subscription.deleted",
            object_data={},
        )
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200


# ===========================================================================
# TestInvoicePaid
# ===========================================================================


class TestInvoicePaid:
    """Tests for _handle_invoice_paid."""

    def _make_paid_event(
        self,
        event_id: str = "evt_inv_paid",
        customer: str = "cus_test_123",
        subscription: str = "sub_test_123",
        amount_paid: int = 9900,
    ) -> MockWebhookEvent:
        return MockWebhookEvent(
            event_id=event_id,
            event_type="invoice.payment_succeeded",
            object_data={
                "customer": customer,
                "subscription": subscription,
                "amount_paid": amount_paid,
            },
        )

    def test_resets_usage_counters(self, handler):
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        org.debates_used_this_month = 42

        event = self._make_paid_event()
        http = _make_webhook_http()
        with (
            _webhook_patches(event),
            patch("aragora.billing.payment_recovery.get_recovery_store") as mock_recovery,
        ):
            mock_recovery.return_value.mark_recovered.return_value = False
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200
        assert org.debates_used_this_month == 0

    def test_marks_recovery_as_recovered(self, handler):
        event = self._make_paid_event()
        http = _make_webhook_http()

        mock_recovery_store = MagicMock()
        mock_recovery_store.mark_recovered.return_value = True

        with (
            _webhook_patches(event),
            patch(
                "aragora.billing.payment_recovery.get_recovery_store",
                return_value=mock_recovery_store,
            ),
        ):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        mock_recovery_store.mark_recovered.assert_called_once_with("org_1")

    def test_recovery_store_error_handled_gracefully(self, handler):
        event = self._make_paid_event()
        http = _make_webhook_http()

        with (
            _webhook_patches(event),
            patch(
                "aragora.billing.payment_recovery.get_recovery_store",
                side_effect=OSError("connection failed"),
            ),
        ):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200

    def test_reset_usage_error_handled_gracefully(self, handler):
        """AttributeError/ValueError/OSError in reset_org_usage should not fail the webhook."""
        store = handler.ctx["user_store"]
        # Replace reset_org_usage with a version that raises
        original_reset = store.reset_org_usage
        store.reset_org_usage = MagicMock(side_effect=OSError("disk error"))

        event = self._make_paid_event()
        http = _make_webhook_http()

        with (
            _webhook_patches(event),
            patch("aragora.billing.payment_recovery.get_recovery_store") as mock_recovery,
        ):
            mock_recovery.return_value.mark_recovered.return_value = False
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200
        store.reset_org_usage = original_reset

    def test_no_customer_id_still_returns_200(self, handler):
        event = MockWebhookEvent(
            event_id="evt_no_cust",
            event_type="invoice.payment_succeeded",
            object_data={"subscription": "sub_test_123", "amount_paid": 9900},
        )
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200

    def test_no_org_for_customer_still_returns_200(self, handler):
        event = self._make_paid_event(customer="cus_unknown")
        http = _make_webhook_http()
        with _webhook_patches(event), patch("aragora.billing.payment_recovery.get_recovery_store"):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200

    def test_no_user_store_returns_200(self, handler_no_store):
        event = self._make_paid_event()
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler_no_store.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200

    def test_null_amount_paid_defaults_to_zero(self, handler):
        """amount_paid can be null/None; should default to 0."""
        event = MockWebhookEvent(
            event_id="evt_null_amt",
            event_type="invoice.payment_succeeded",
            object_data={"customer": "cus_test_123", "amount_paid": None},
        )
        http = _make_webhook_http()
        with (
            _webhook_patches(event),
            patch("aragora.billing.payment_recovery.get_recovery_store") as mock_recovery,
        ):
            mock_recovery.return_value.mark_recovered.return_value = False
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200


# ===========================================================================
# TestInvoiceFailed
# ===========================================================================


class TestInvoiceFailed:
    """Tests for _handle_invoice_failed."""

    def _make_failed_event(
        self,
        event_id: str = "evt_inv_fail",
        customer: str = "cus_test_123",
        subscription: str = "sub_test_123",
        attempt_count: int = 1,
        invoice_id: str = "inv_fail_1",
        hosted_invoice_url: str = "https://inv.stripe.com/pay",
    ) -> MockWebhookEvent:
        return MockWebhookEvent(
            event_id=event_id,
            event_type="invoice.payment_failed",
            object_data={
                "customer": customer,
                "subscription": subscription,
                "attempt_count": attempt_count,
                "id": invoice_id,
                "hosted_invoice_url": hosted_invoice_url,
            },
        )

    def _mock_failure(
        self, attempt_count: int = 1, days_failing: int = 3, days_until_downgrade: int = 11
    ):
        failure = MagicMock()
        failure.attempt_count = attempt_count
        failure.days_failing = days_failing
        failure.days_until_downgrade = days_until_downgrade
        return failure

    def test_records_failure_and_returns_tracked(self, handler):
        event = self._make_failed_event()
        http = _make_webhook_http()

        mock_recovery = MagicMock()
        mock_recovery.record_failure.return_value = self._mock_failure()
        mock_notifier = MagicMock()
        mock_notifier.notify_payment_failed.return_value = MagicMock(method="email", success=True)

        with (
            _webhook_patches(event),
            patch(
                "aragora.billing.payment_recovery.get_recovery_store", return_value=mock_recovery
            ),
            patch("aragora.billing.notifications.get_billing_notifier", return_value=mock_notifier),
        ):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
            body = _body(result)
        assert _status(result) == 200
        assert body["failure_tracked"] is True

    def test_sends_notification_to_owner(self, handler):
        event = self._make_failed_event()
        http = _make_webhook_http()

        mock_recovery = MagicMock()
        mock_recovery.record_failure.return_value = self._mock_failure()
        mock_notifier = MagicMock()
        mock_notifier.notify_payment_failed.return_value = MagicMock(method="email", success=True)

        with (
            _webhook_patches(event),
            patch(
                "aragora.billing.payment_recovery.get_recovery_store", return_value=mock_recovery
            ),
            patch("aragora.billing.notifications.get_billing_notifier", return_value=mock_notifier),
        ):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")

        mock_notifier.notify_payment_failed.assert_called_once()
        call_kwargs = mock_notifier.notify_payment_failed.call_args[1]
        assert call_kwargs["email"] == "test@example.com"
        assert call_kwargs["org_name"] == "Test Org"

    def test_notification_uses_failure_attempt_count(self, handler):
        event = self._make_failed_event(attempt_count=5)
        http = _make_webhook_http()

        failure = self._mock_failure(attempt_count=3)
        mock_recovery = MagicMock()
        mock_recovery.record_failure.return_value = failure
        mock_notifier = MagicMock()
        mock_notifier.notify_payment_failed.return_value = MagicMock(method="email", success=True)

        with (
            _webhook_patches(event),
            patch(
                "aragora.billing.payment_recovery.get_recovery_store", return_value=mock_recovery
            ),
            patch("aragora.billing.notifications.get_billing_notifier", return_value=mock_notifier),
        ):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")

        call_kwargs = mock_notifier.notify_payment_failed.call_args[1]
        # Should use failure.attempt_count (3), not event attempt_count (5)
        assert call_kwargs["attempt_count"] == 3

    def test_grace_period_near_end_warning(self, handler):
        """When days_until_downgrade <= 3, a warning should be logged."""
        event = self._make_failed_event()
        http = _make_webhook_http()

        failure = self._mock_failure(days_until_downgrade=2)
        mock_recovery = MagicMock()
        mock_recovery.record_failure.return_value = failure
        mock_notifier = MagicMock()
        mock_notifier.notify_payment_failed.return_value = MagicMock(method="email", success=True)

        with (
            _webhook_patches(event),
            patch(
                "aragora.billing.payment_recovery.get_recovery_store", return_value=mock_recovery
            ),
            patch("aragora.billing.notifications.get_billing_notifier", return_value=mock_notifier),
        ):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        # Should succeed regardless
        assert _status(result) == 200
        assert _body(result)["failure_tracked"] is True

    def test_recovery_store_error_failure_not_tracked(self, handler):
        event = self._make_failed_event()
        http = _make_webhook_http()

        mock_notifier = MagicMock()
        mock_notifier.notify_payment_failed.return_value = MagicMock(method="email", success=True)

        with (
            _webhook_patches(event),
            patch(
                "aragora.billing.payment_recovery.get_recovery_store",
                side_effect=OSError("store down"),
            ),
            patch("aragora.billing.notifications.get_billing_notifier", return_value=mock_notifier),
        ):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
            body = _body(result)
        assert _status(result) == 200
        assert body["failure_tracked"] is False

    def test_recovery_store_attribute_error_handled(self, handler):
        event = self._make_failed_event()
        http = _make_webhook_http()

        mock_notifier = MagicMock()
        mock_notifier.notify_payment_failed.return_value = MagicMock(method="email", success=True)

        with (
            _webhook_patches(event),
            patch(
                "aragora.billing.payment_recovery.get_recovery_store",
                side_effect=AttributeError("no store"),
            ),
            patch("aragora.billing.notifications.get_billing_notifier", return_value=mock_notifier),
        ):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200
        assert _body(result)["failure_tracked"] is False

    def test_notification_error_handled_gracefully(self, handler):
        event = self._make_failed_event()
        http = _make_webhook_http()

        mock_recovery = MagicMock()
        mock_recovery.record_failure.return_value = self._mock_failure()

        with (
            _webhook_patches(event),
            patch(
                "aragora.billing.payment_recovery.get_recovery_store", return_value=mock_recovery
            ),
            patch(
                "aragora.billing.notifications.get_billing_notifier",
                side_effect=OSError("notifier down"),
            ),
        ):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200
        assert _body(result)["failure_tracked"] is True

    def test_no_owner_no_notification(self, handler):
        """If org has no owner with email, notification should be skipped."""
        store = handler.ctx["user_store"]
        # Change the user so they're not an owner
        user = store._users["test-user-001"]
        user.role = "member"

        event = self._make_failed_event()
        http = _make_webhook_http()

        mock_recovery = MagicMock()
        mock_recovery.record_failure.return_value = self._mock_failure()
        mock_notifier = MagicMock()

        with (
            _webhook_patches(event),
            patch(
                "aragora.billing.payment_recovery.get_recovery_store", return_value=mock_recovery
            ),
            patch("aragora.billing.notifications.get_billing_notifier", return_value=mock_notifier),
        ):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200
        # Notifier should NOT have been called since no owner found
        mock_notifier.notify_payment_failed.assert_not_called()

    def test_no_customer_returns_untracked(self, handler):
        event = MockWebhookEvent(
            event_id="evt_fail_nocust",
            event_type="invoice.payment_failed",
            object_data={"subscription": "sub_test_123", "attempt_count": 1},
        )
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200
        assert _body(result)["failure_tracked"] is False

    def test_no_user_store_returns_untracked(self, handler_no_store):
        event = self._make_failed_event()
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler_no_store.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200
        assert _body(result)["failure_tracked"] is False

    def test_invoice_url_passed_to_notifier(self, handler):
        event = self._make_failed_event(hosted_invoice_url="https://pay.stripe.com/inv/xyz")
        http = _make_webhook_http()

        mock_recovery = MagicMock()
        mock_recovery.record_failure.return_value = self._mock_failure()
        mock_notifier = MagicMock()
        mock_notifier.notify_payment_failed.return_value = MagicMock(method="email", success=True)

        with (
            _webhook_patches(event),
            patch(
                "aragora.billing.payment_recovery.get_recovery_store", return_value=mock_recovery
            ),
            patch("aragora.billing.notifications.get_billing_notifier", return_value=mock_notifier),
        ):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")

        call_kwargs = mock_notifier.notify_payment_failed.call_args[1]
        assert call_kwargs["invoice_url"] == "https://pay.stripe.com/inv/xyz"

    def test_days_until_downgrade_passed_to_notifier(self, handler):
        event = self._make_failed_event()
        http = _make_webhook_http()

        failure = self._mock_failure(days_until_downgrade=5)
        mock_recovery = MagicMock()
        mock_recovery.record_failure.return_value = failure
        mock_notifier = MagicMock()
        mock_notifier.notify_payment_failed.return_value = MagicMock(method="email", success=True)

        with (
            _webhook_patches(event),
            patch(
                "aragora.billing.payment_recovery.get_recovery_store", return_value=mock_recovery
            ),
            patch("aragora.billing.notifications.get_billing_notifier", return_value=mock_notifier),
        ):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")

        call_kwargs = mock_notifier.notify_payment_failed.call_args[1]
        assert call_kwargs["days_until_downgrade"] == 5

    def test_no_failure_uses_event_attempt_count(self, handler):
        """When recovery store fails, notification should fall back to event attempt_count."""
        event = self._make_failed_event(attempt_count=7)
        http = _make_webhook_http()

        mock_recovery = MagicMock()
        mock_recovery.record_failure.side_effect = OSError("store down")
        mock_notifier = MagicMock()
        mock_notifier.notify_payment_failed.return_value = MagicMock(method="email", success=True)

        with (
            _webhook_patches(event),
            patch(
                "aragora.billing.payment_recovery.get_recovery_store", return_value=mock_recovery
            ),
            patch("aragora.billing.notifications.get_billing_notifier", return_value=mock_notifier),
        ):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")

        call_kwargs = mock_notifier.notify_payment_failed.call_args[1]
        # When failure is None, should use event attempt_count
        assert call_kwargs["attempt_count"] == 7
        assert call_kwargs["days_until_downgrade"] is None


# ===========================================================================
# TestInvoiceFinalized
# ===========================================================================


class TestInvoiceFinalized:
    """Tests for _handle_invoice_finalized."""

    def _make_finalized_event(
        self,
        event_id: str = "evt_fin_1",
        customer: str = "cus_test_123",
        subscription: str = "sub_test_123",
    ) -> MockWebhookEvent:
        return MockWebhookEvent(
            event_id=event_id,
            event_type="invoice.finalized",
            object_data={
                "customer": customer,
                "subscription": subscription,
            },
        )

    def test_flushes_usage_records(self, handler):
        event = self._make_finalized_event()
        http = _make_webhook_http()

        mock_sync = MagicMock()
        mock_sync.flush_period.return_value = [{"id": "rec_1"}, {"id": "rec_2"}]

        with (
            _webhook_patches(event),
            patch("aragora.billing.usage_sync.get_usage_sync_service", return_value=mock_sync),
        ):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
            body = _body(result)
        assert _status(result) == 200
        assert body["usage_flushed"] == 2

    def test_flush_with_correct_org_id(self, handler):
        event = self._make_finalized_event()
        http = _make_webhook_http()

        mock_sync = MagicMock()
        mock_sync.flush_period.return_value = []

        with (
            _webhook_patches(event),
            patch("aragora.billing.usage_sync.get_usage_sync_service", return_value=mock_sync),
        ):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        mock_sync.flush_period.assert_called_once_with(org_id="org_1")

    def test_no_records_flushed(self, handler):
        event = self._make_finalized_event()
        http = _make_webhook_http()

        mock_sync = MagicMock()
        mock_sync.flush_period.return_value = []

        with (
            _webhook_patches(event),
            patch("aragora.billing.usage_sync.get_usage_sync_service", return_value=mock_sync),
        ):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
            body = _body(result)
        assert body["usage_flushed"] == 0

    def test_flush_error_handled_gracefully(self, handler):
        event = self._make_finalized_event()
        http = _make_webhook_http()

        with (
            _webhook_patches(event),
            patch(
                "aragora.billing.usage_sync.get_usage_sync_service",
                side_effect=OSError("sync service down"),
            ),
        ):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
            body = _body(result)
        assert _status(result) == 200
        assert body["usage_flushed"] == 0

    def test_flush_attribute_error_handled(self, handler):
        event = self._make_finalized_event()
        http = _make_webhook_http()

        with (
            _webhook_patches(event),
            patch(
                "aragora.billing.usage_sync.get_usage_sync_service",
                side_effect=AttributeError("no flush"),
            ),
        ):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200
        assert _body(result)["usage_flushed"] == 0

    def test_no_customer_id_returns_unflushed(self, handler):
        event = MockWebhookEvent(
            event_id="evt_fin_nocust",
            event_type="invoice.finalized",
            object_data={"subscription": "sub_test_123"},
        )
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200
        assert _body(result)["usage_flushed"] == 0

    def test_no_org_for_customer_returns_unflushed(self, handler):
        event = self._make_finalized_event(customer="cus_unknown")
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200
        assert _body(result)["usage_flushed"] == 0

    def test_no_user_store_returns_unflushed(self, handler_no_store):
        event = self._make_finalized_event()
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler_no_store.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200
        assert _body(result)["usage_flushed"] == 0


# ===========================================================================
# TestUnhandledEvents
# ===========================================================================


class TestUnhandledEvents:
    """Tests for unhandled/unknown event types."""

    def test_unknown_event_type_acknowledged(self, handler):
        event = MockWebhookEvent("evt_unk_1", "some.unknown.event")
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
            body = _body(result)
        assert _status(result) == 200
        assert body["received"] is True

    def test_charge_succeeded_acknowledged(self, handler):
        event = MockWebhookEvent("evt_charge", "charge.succeeded")
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200

    def test_payment_intent_event_acknowledged(self, handler):
        event = MockWebhookEvent("evt_pi", "payment_intent.succeeded")
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200

    def test_unhandled_event_still_marked_processed(self, handler):
        mark_calls = []

        def track_mark(event_id, result="success"):
            mark_calls.append(event_id)

        event = MockWebhookEvent("evt_unk_mark", "some.other.event")
        http = _make_webhook_http()
        with _webhook_patches(event, mark_processed_fn=track_mark):
            handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert "evt_unk_mark" in mark_calls


# ===========================================================================
# TestWebhookSecurity
# ===========================================================================


class TestWebhookSecurity:
    """Security tests for webhook handling."""

    def test_path_traversal_in_event_metadata(self, handler):
        """Metadata with path traversal attempts should not cause issues."""
        event = MockWebhookEvent(
            event_id="evt_sec_1",
            event_type="checkout.session.completed",
            object_data={"customer": "cus_test_123", "subscription": "sub_new"},
            metadata={
                "user_id": "../../etc/passwd",
                "org_id": "org_1",
                "tier": "starter",
            },
        )
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        # Should succeed but not find a user with path traversal ID
        assert _status(result) == 200

    def test_injection_in_event_type(self, handler):
        """Event type with injection attempts should be safely handled."""
        event = MockWebhookEvent(
            "evt_sec_2",
            "checkout.session.completed; DROP TABLE users;--",
        )
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        # Unhandled event type, should be acknowledged
        assert _status(result) == 200

    def test_empty_event_type(self, handler):
        event = MockWebhookEvent("evt_sec_3", "")
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200

    def test_extremely_long_event_type(self, handler):
        event = MockWebhookEvent("evt_sec_4", "a" * 10000)
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200

    def test_null_bytes_in_metadata(self, handler):
        event = MockWebhookEvent(
            event_id="evt_sec_5",
            event_type="checkout.session.completed",
            object_data={"customer": "cus_test_123", "subscription": "sub_new"},
            metadata={"user_id": "user\x00evil", "org_id": "org_1", "tier": "starter"},
        )
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200

    def test_unicode_in_metadata(self, handler):
        event = MockWebhookEvent(
            event_id="evt_sec_6",
            event_type="checkout.session.completed",
            object_data={"customer": "cus_test_123", "subscription": "sub_new"},
            metadata={
                "user_id": "test-user-001",
                "org_id": "org_1",
                "tier": "\u0441\u0442\u0430\u0440\u0442\u0435\u0440",  # Cyrillic
            },
        )
        http = _make_webhook_http()
        with _webhook_patches(event):
            result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
        assert _status(result) == 200
        # Invalid tier should fall back to STARTER
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        assert org.tier == SubscriptionTier.STARTER

    def test_webhooks_skip_rate_limiting(self, handler):
        """Webhook endpoint should not be rate limited."""
        event = MockWebhookEvent("evt_rate", "some.event")
        # The rate limiter is checked in handler.handle() but skipped for webhook path
        for _ in range(30):  # Fire more than rate limit
            http = _make_webhook_http()
            with _webhook_patches(event):
                result = handler.handle("/api/v1/webhooks/stripe", {}, http, method="POST")
            assert _status(result) != 429


# ===========================================================================
# TestWebhookLogger - _logger() resolution
# ===========================================================================


class TestWebhookLogger:
    """Tests for _logger() resolution logic."""

    def test_logger_resolves_from_core_module(self):
        """When billing.core is in sys.modules, _logger() should use its logger."""
        from aragora.server.handlers.billing.core_webhooks import _logger

        logger = _logger()
        assert logger is not None

    def test_logger_fallback_when_core_not_in_modules(self):
        """When billing.core is not in sys.modules, should use logging.getLogger."""
        import sys
        from aragora.server.handlers.billing.core_webhooks import _logger

        # Save and remove the core module temporarily
        saved = sys.modules.pop("aragora.server.handlers.billing.core", None)
        try:
            logger = _logger()
            assert logger is not None
            assert logger.name == "aragora.server.handlers.billing.core"
        finally:
            if saved is not None:
                sys.modules["aragora.server.handlers.billing.core"] = saved


# ===========================================================================
# TestWebhookMixinIntegration - End-to-end scenarios
# ===========================================================================


class TestWebhookMixinIntegration:
    """Integration tests covering multi-step webhook scenarios."""

    def test_checkout_then_subscription_updated_to_higher_tier(self, handler):
        """Simulate checkout completing then upgrading to higher tier."""
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        org.tier = SubscriptionTier.FREE

        # Step 1: Checkout completed
        checkout_event = MockWebhookEvent(
            event_id="evt_int_1a",
            event_type="checkout.session.completed",
            object_data={
                "customer": "cus_test_123",
                "subscription": "sub_test_123",
                "id": "cs_1",
            },
            metadata={"user_id": "test-user-001", "org_id": "org_1", "tier": "starter"},
        )
        http1 = _make_webhook_http()
        with _webhook_patches(checkout_event):
            handler.handle("/api/v1/webhooks/stripe", {}, http1, method="POST")
        assert org.tier == SubscriptionTier.STARTER

        # Step 2: Subscription updated to professional
        sub_update_event = MockWebhookEvent(
            event_id="evt_int_1b",
            event_type="customer.subscription.updated",
            object_data={
                "id": "sub_test_123",
                "status": "active",
                "items": {"data": [{"price": {"id": "price_professional"}}]},
            },
        )
        http2 = _make_webhook_http()
        with (
            _webhook_patches(sub_update_event),
            patch(
                "aragora.billing.stripe_client.get_tier_from_price_id",
                return_value=SubscriptionTier.PROFESSIONAL,
            ),
        ):
            handler.handle("/api/v1/webhooks/stripe", {}, http2, method="POST")
        assert org.tier == SubscriptionTier.PROFESSIONAL

    def test_subscription_deleted_then_checkout_new_subscription(self, handler):
        """Simulate deletion followed by new checkout."""
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        org.tier = SubscriptionTier.PROFESSIONAL

        # Step 1: Subscription deleted
        del_event = MockWebhookEvent(
            event_id="evt_int_2a",
            event_type="customer.subscription.deleted",
            object_data={"id": "sub_test_123"},
        )
        http1 = _make_webhook_http()
        with _webhook_patches(del_event):
            handler.handle("/api/v1/webhooks/stripe", {}, http1, method="POST")
        assert org.tier == SubscriptionTier.FREE
        assert org.stripe_subscription_id is None

        # Step 2: New checkout
        checkout_event = MockWebhookEvent(
            event_id="evt_int_2b",
            event_type="checkout.session.completed",
            object_data={
                "customer": "cus_test_123",
                "subscription": "sub_new_2",
                "id": "cs_new",
            },
            metadata={"user_id": "test-user-001", "org_id": "org_1", "tier": "enterprise"},
        )
        http2 = _make_webhook_http()
        with _webhook_patches(checkout_event):
            handler.handle("/api/v1/webhooks/stripe", {}, http2, method="POST")
        assert org.tier == SubscriptionTier.ENTERPRISE
        assert org.stripe_subscription_id == "sub_new_2"

    def test_invoice_failed_then_paid_recovers(self, handler):
        """Simulate payment failure followed by recovery."""
        store = handler.ctx["user_store"]
        org = store.get_organization_by_id("org_1")
        org.debates_used_this_month = 50

        # Step 1: Invoice fails
        fail_event = MockWebhookEvent(
            event_id="evt_int_3a",
            event_type="invoice.payment_failed",
            object_data={
                "customer": "cus_test_123",
                "subscription": "sub_test_123",
                "attempt_count": 1,
                "id": "inv_1",
                "hosted_invoice_url": "https://inv.stripe.com/pay",
            },
        )
        http1 = _make_webhook_http()

        mock_recovery = MagicMock()
        mock_recovery.record_failure.return_value = MagicMock(
            attempt_count=1, days_failing=0, days_until_downgrade=14
        )
        mock_notifier = MagicMock()
        mock_notifier.notify_payment_failed.return_value = MagicMock(method="email", success=True)

        with (
            _webhook_patches(fail_event),
            patch(
                "aragora.billing.payment_recovery.get_recovery_store", return_value=mock_recovery
            ),
            patch("aragora.billing.notifications.get_billing_notifier", return_value=mock_notifier),
        ):
            result1 = handler.handle("/api/v1/webhooks/stripe", {}, http1, method="POST")
        assert _body(result1)["failure_tracked"] is True

        # Step 2: Invoice paid (recovers)
        paid_event = MockWebhookEvent(
            event_id="evt_int_3b",
            event_type="invoice.payment_succeeded",
            object_data={
                "customer": "cus_test_123",
                "subscription": "sub_test_123",
                "amount_paid": 9900,
            },
        )
        http2 = _make_webhook_http()

        mock_recovery2 = MagicMock()
        mock_recovery2.mark_recovered.return_value = True

        with (
            _webhook_patches(paid_event),
            patch(
                "aragora.billing.payment_recovery.get_recovery_store", return_value=mock_recovery2
            ),
        ):
            result2 = handler.handle("/api/v1/webhooks/stripe", {}, http2, method="POST")
        assert _status(result2) == 200
        assert org.debates_used_this_month == 0
        mock_recovery2.mark_recovered.assert_called_once_with("org_1")
