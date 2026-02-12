"""
Extended tests for BillingHandler (core billing operations).

Covers edge cases and error paths NOT tested in test_billing_core.py:
- Input validation helpers (_validate_iso_date, _safe_positive_int)
- _log_audit error handling
- Stripe error variants (StripeConfigError, StripeAPIError, StripeError)
- Checkout with invalid JSON body
- Portal with no org user
- Resume with no subscription
- Invoices pagination, no user store, Stripe errors
- Audit log non-admin rejection, pagination params
- Subscription with trialing and past_due states
- Usage with usage_tracker integration
- Webhook duplicate detection, unhandled events, invoice finalized
- Subscription updated webhook with tier changes
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.billing.core import (
    BillingHandler,
    _safe_positive_int,
    _validate_iso_date,
)


# ---------------------------------------------------------------------------
# Mock classes (same pattern as test_billing_core.py)
# ---------------------------------------------------------------------------


class FakeTier(Enum):
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@dataclass
class FakeUser:
    user_id: str = "user-123"
    email: str = "test@example.com"
    role: str = "owner"
    org_id: str = "org-123"


@dataclass
class FakeDbUser:
    id: str = "user-123"
    email: str = "test@example.com"
    org_id: str = "org-123"


@dataclass
class FakeTierLimits:
    debates_per_month: int = 100
    users_per_org: int = 10
    api_access: bool = True
    all_agents: bool = True
    custom_agents: bool = False
    sso_enabled: bool = False
    audit_logs: bool = False
    priority_support: bool = False
    price_monthly_cents: int = 2900

    def to_dict(self) -> dict[str, Any]:
        return {
            "debates_per_month": self.debates_per_month,
            "users_per_org": self.users_per_org,
            "api_access": self.api_access,
            "all_agents": self.all_agents,
            "custom_agents": self.custom_agents,
            "sso_enabled": self.sso_enabled,
            "audit_logs": self.audit_logs,
            "priority_support": self.priority_support,
        }


@dataclass
class FakeOrganization:
    id: str = "org-123"
    name: str = "Test Org"
    slug: str = "test-org"
    tier: FakeTier = field(default_factory=lambda: FakeTier.STARTER)
    limits: FakeTierLimits = field(default_factory=FakeTierLimits)
    stripe_customer_id: str | None = "cus_test123"
    stripe_subscription_id: str | None = "sub_test123"
    debates_used_this_month: int = 10
    debates_remaining: int = 90
    billing_cycle_start: datetime = field(
        default_factory=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc)
    )


@dataclass
class FakeSubscription:
    id: str = "sub_test123"
    status: str = "active"
    current_period_end: datetime = field(
        default_factory=lambda: datetime(2025, 2, 1, tzinfo=timezone.utc)
    )
    cancel_at_period_end: bool = False
    trial_start: datetime | None = None
    trial_end: datetime | None = None
    is_trialing: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "current_period_end": self.current_period_end.isoformat(),
            "cancel_at_period_end": self.cancel_at_period_end,
        }


class FakeHandler:
    """Mock HTTP handler for testing."""

    def __init__(
        self,
        method: str = "GET",
        body: dict | None = None,
        headers: dict | None = None,
        query_params: dict | None = None,
    ):
        import io

        self.command = method
        self._body = json.dumps(body).encode() if body else b"{}"
        self.headers = headers or {}
        self.client_address = ("127.0.0.1", 12345)
        self._query_params = query_params or {}

    @property
    def rfile(self):
        import io

        return io.BytesIO(self._body)

    def get(self, key: str, default: Any = None) -> Any:
        return self._query_params.get(key, default)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_user_store():
    store = MagicMock()
    store.get_user_by_id = MagicMock(return_value=FakeDbUser())
    store.get_organization_by_id = MagicMock(return_value=FakeOrganization())
    store.get_organization_by_subscription = MagicMock(return_value=FakeOrganization())
    store.get_organization_by_stripe_customer = MagicMock(return_value=FakeOrganization())
    store.update_organization = MagicMock()
    store.reset_org_usage = MagicMock()
    store.get_audit_log = MagicMock(return_value=[])
    store.get_audit_log_count = MagicMock(return_value=0)
    store.log_audit_event = MagicMock()
    store.get_organization_owner = MagicMock(return_value=MagicMock(email="owner@test.com"))
    return store


@pytest.fixture
def billing_handler(mock_user_store):
    return BillingHandler(ctx={"user_store": mock_user_store})


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    from aragora.server.handlers.billing.core import _billing_limiter

    _billing_limiter._requests.clear()
    yield
    _billing_limiter._requests.clear()


# ---------------------------------------------------------------------------
# Test _validate_iso_date
# ---------------------------------------------------------------------------


class TestValidateIsoDate:
    def test_valid_date(self):
        assert _validate_iso_date("2025-01-15") == "2025-01-15"

    def test_none_returns_none(self):
        assert _validate_iso_date(None) is None

    def test_empty_string_returns_none(self):
        assert _validate_iso_date("") is None

    def test_invalid_format_returns_none(self):
        assert _validate_iso_date("01-15-2025") is None

    def test_non_string_returns_none(self):
        assert _validate_iso_date(12345) is None  # type: ignore[arg-type]

    def test_impossible_date_returns_none(self):
        """Feb 30 matches the regex but is not a valid date."""
        assert _validate_iso_date("2025-02-30") is None

    def test_month_13_returns_none(self):
        assert _validate_iso_date("2025-13-01") is None


# ---------------------------------------------------------------------------
# Test _safe_positive_int
# ---------------------------------------------------------------------------


class TestSafePositiveInt:
    def test_valid_number(self):
        assert _safe_positive_int("10", 5, 100) == 10

    def test_non_numeric_returns_default(self):
        assert _safe_positive_int("abc", 5, 100) == 5

    def test_negative_returns_default(self):
        assert _safe_positive_int("-1", 5, 100) == 5

    def test_exceeds_maximum_is_clamped(self):
        assert _safe_positive_int("200", 5, 100) == 100

    def test_zero_is_valid(self):
        assert _safe_positive_int("0", 5, 100) == 0

    def test_none_value_returns_default(self):
        assert _safe_positive_int(None, 5, 100) == 5  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test _log_audit
# ---------------------------------------------------------------------------


class TestLogAudit:
    def test_skips_when_no_user_store(self, billing_handler):
        """No error when user_store is None."""
        billing_handler._log_audit(None, action="test", resource_type="sub")

    def test_skips_when_no_log_method(self, billing_handler):
        """No error when user_store lacks log_audit_event."""
        store = MagicMock(spec=[])  # no attributes
        billing_handler._log_audit(store, action="test", resource_type="sub")

    def test_handles_audit_logging_failure(self, billing_handler, mock_user_store):
        """IOError during audit logging is caught."""
        mock_user_store.log_audit_event.side_effect = OSError("disk full")
        # Should not raise
        billing_handler._log_audit(
            mock_user_store,
            action="subscription.created",
            resource_type="subscription",
            resource_id="sub_123",
        )

    def test_extracts_ip_and_user_agent_from_handler(self, billing_handler, mock_user_store):
        """Audit log captures IP and user-agent from HTTP handler."""
        handler = FakeHandler(headers={"User-Agent": "TestBrowser/1.0"})

        with patch("aragora.server.middleware.auth.extract_client_ip", return_value="10.0.0.1"):
            billing_handler._log_audit(
                mock_user_store,
                action="test",
                resource_type="sub",
                handler=handler,
            )

        call_kwargs = mock_user_store.log_audit_event.call_args
        assert call_kwargs.kwargs["ip_address"] == "10.0.0.1"
        assert call_kwargs.kwargs["user_agent"] == "TestBrowser/1.0"


# ---------------------------------------------------------------------------
# Test _create_checkout Stripe error paths
# ---------------------------------------------------------------------------


class TestCheckoutStripeErrors:
    def _call(self, billing_handler, handler, user, body):
        with patch.object(billing_handler, "read_json_body", return_value=body):
            fn = billing_handler._create_checkout.__wrapped__.__wrapped__.__wrapped__
            return fn(billing_handler, handler, user=user)

    def test_invalid_json_body_returns_400(self, billing_handler, mock_user_store):
        handler = FakeHandler(method="POST")
        user = FakeUser()

        with patch.object(billing_handler, "read_json_body", return_value=None):
            fn = billing_handler._create_checkout.__wrapped__.__wrapped__.__wrapped__
            result = fn(billing_handler, handler, user=user)

        assert result.status_code == 400

    def test_stripe_config_error_returns_503(self, billing_handler, mock_user_store):
        from aragora.billing.stripe_client import StripeConfigError

        handler = FakeHandler(method="POST")
        user = FakeUser()
        body = {
            "tier": "starter",
            "success_url": "https://example.com/ok",
            "cancel_url": "https://example.com/cancel",
        }

        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            side_effect=StripeConfigError("no key"),
        ):
            result = self._call(billing_handler, handler, user, body)

        assert result.status_code == 503

    def test_stripe_api_error_returns_502(self, billing_handler, mock_user_store):
        from aragora.billing.stripe_client import StripeAPIError

        handler = FakeHandler(method="POST")
        user = FakeUser()
        body = {
            "tier": "starter",
            "success_url": "https://example.com/ok",
            "cancel_url": "https://example.com/cancel",
        }
        mock_client = MagicMock()
        mock_client.create_checkout_session.side_effect = StripeAPIError("bad request")

        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            result = self._call(billing_handler, handler, user, body)

        assert result.status_code == 502

    def test_stripe_generic_error_returns_500(self, billing_handler, mock_user_store):
        from aragora.billing.stripe_client import StripeError

        handler = FakeHandler(method="POST")
        user = FakeUser()
        body = {
            "tier": "starter",
            "success_url": "https://example.com/ok",
            "cancel_url": "https://example.com/cancel",
        }
        mock_client = MagicMock()
        mock_client.create_checkout_session.side_effect = StripeError("unknown")

        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            result = self._call(billing_handler, handler, user, body)

        assert result.status_code == 500

    def test_checkout_user_not_found_returns_404(self, billing_handler, mock_user_store):
        mock_user_store.get_user_by_id.return_value = None
        handler = FakeHandler(method="POST")
        user = FakeUser()
        body = {
            "tier": "starter",
            "success_url": "https://example.com/ok",
            "cancel_url": "https://example.com/cancel",
        }
        result = self._call(billing_handler, handler, user, body)
        assert result.status_code == 404


# ---------------------------------------------------------------------------
# Test _create_portal Stripe error paths
# ---------------------------------------------------------------------------


class TestPortalStripeErrors:
    def _call(self, billing_handler, handler, user, body):
        with patch.object(billing_handler, "read_json_body", return_value=body):
            fn = billing_handler._create_portal.__wrapped__.__wrapped__
            return fn(billing_handler, handler, user=user)

    def test_invalid_json_returns_400(self, billing_handler, mock_user_store):
        handler = FakeHandler(method="POST")
        user = FakeUser()
        with patch.object(billing_handler, "read_json_body", return_value=None):
            fn = billing_handler._create_portal.__wrapped__.__wrapped__
            result = fn(billing_handler, handler, user=user)
        assert result.status_code == 400

    def test_user_without_org_returns_404(self, billing_handler, mock_user_store):
        mock_user_store.get_user_by_id.return_value = FakeDbUser(org_id=None)
        handler = FakeHandler(method="POST")
        user = FakeUser()
        body = {"return_url": "https://example.com/billing"}
        result = self._call(billing_handler, handler, user, body)
        assert result.status_code == 404

    def test_stripe_config_error_returns_503(self, billing_handler, mock_user_store):
        from aragora.billing.stripe_client import StripeConfigError

        handler = FakeHandler(method="POST")
        user = FakeUser()
        body = {"return_url": "https://example.com/billing"}

        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            side_effect=StripeConfigError("no key"),
        ):
            result = self._call(billing_handler, handler, user, body)

        assert result.status_code == 503

    def test_stripe_api_error_returns_502(self, billing_handler, mock_user_store):
        from aragora.billing.stripe_client import StripeAPIError

        handler = FakeHandler(method="POST")
        user = FakeUser()
        body = {"return_url": "https://example.com/billing"}
        mock_client = MagicMock()
        mock_client.create_portal_session.side_effect = StripeAPIError("err")

        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            result = self._call(billing_handler, handler, user, body)

        assert result.status_code == 502


# ---------------------------------------------------------------------------
# Test _cancel_subscription Stripe error paths
# ---------------------------------------------------------------------------


class TestCancelStripeErrors:
    def _call(self, billing_handler, handler, user):
        fn = billing_handler._cancel_subscription.__wrapped__.__wrapped__.__wrapped__
        return fn(billing_handler, handler, user=user)

    def test_stripe_config_error_returns_503(self, billing_handler, mock_user_store):
        from aragora.billing.stripe_client import StripeConfigError

        handler = FakeHandler(method="POST")
        user = FakeUser()

        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            side_effect=StripeConfigError("no key"),
        ):
            result = self._call(billing_handler, handler, user)

        assert result.status_code == 503

    def test_stripe_api_error_returns_502(self, billing_handler, mock_user_store):
        from aragora.billing.stripe_client import StripeAPIError

        handler = FakeHandler(method="POST")
        user = FakeUser()
        mock_client = MagicMock()
        mock_client.cancel_subscription.side_effect = StripeAPIError("err")

        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            result = self._call(billing_handler, handler, user)

        assert result.status_code == 502


# ---------------------------------------------------------------------------
# Test _resume_subscription error paths
# ---------------------------------------------------------------------------


class TestResumeErrors:
    def _call(self, billing_handler, handler, user):
        fn = billing_handler._resume_subscription.__wrapped__.__wrapped__
        return fn(billing_handler, handler, user=user)

    def test_no_subscription_to_resume_returns_404(self, billing_handler, mock_user_store):
        mock_user_store.get_organization_by_id.return_value = FakeOrganization(
            stripe_subscription_id=None
        )
        handler = FakeHandler(method="POST")
        user = FakeUser()
        result = self._call(billing_handler, handler, user)
        assert result.status_code == 404

    def test_stripe_config_error_returns_503(self, billing_handler, mock_user_store):
        from aragora.billing.stripe_client import StripeConfigError

        handler = FakeHandler(method="POST")
        user = FakeUser()

        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            side_effect=StripeConfigError("no key"),
        ):
            result = self._call(billing_handler, handler, user)

        assert result.status_code == 503

    def test_user_without_org_returns_404(self, billing_handler, mock_user_store):
        mock_user_store.get_user_by_id.return_value = FakeDbUser(org_id=None)
        handler = FakeHandler(method="POST")
        user = FakeUser()
        result = self._call(billing_handler, handler, user)
        assert result.status_code == 404


# ---------------------------------------------------------------------------
# Test _get_invoices error paths
# ---------------------------------------------------------------------------


class TestInvoicesErrors:
    def _call(self, billing_handler, handler, user):
        fn = billing_handler._get_invoices.__wrapped__.__wrapped__
        return fn(billing_handler, handler, user=user)

    def test_no_user_store_returns_503(self):
        handler_obj = BillingHandler(ctx={})
        handler = FakeHandler()
        fn = handler_obj._get_invoices.__wrapped__.__wrapped__
        result = fn(handler_obj, handler, user=FakeUser())
        assert result.status_code == 503

    def test_no_billing_account_returns_404(self, billing_handler, mock_user_store):
        mock_user_store.get_organization_by_id.return_value = FakeOrganization(
            stripe_customer_id=None
        )
        handler = FakeHandler()
        result = self._call(billing_handler, handler, FakeUser())
        assert result.status_code == 404

    def test_stripe_config_error_returns_503(self, billing_handler, mock_user_store):
        from aragora.billing.stripe_client import StripeConfigError

        handler = FakeHandler()
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            side_effect=StripeConfigError("no key"),
        ):
            result = self._call(billing_handler, handler, FakeUser())
        assert result.status_code == 503

    def test_stripe_api_error_returns_502(self, billing_handler, mock_user_store):
        from aragora.billing.stripe_client import StripeAPIError

        handler = FakeHandler()
        mock_client = MagicMock()
        mock_client.list_invoices.side_effect = StripeAPIError("err")
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            result = self._call(billing_handler, handler, FakeUser())
        assert result.status_code == 502

    def test_limit_param_is_parsed(self, billing_handler, mock_user_store):
        """Custom limit query param is forwarded to Stripe."""
        mock_client = MagicMock()
        mock_client.list_invoices.return_value = []
        handler = FakeHandler(query_params={"limit": "25"})
        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            result = self._call(billing_handler, handler, FakeUser())
        assert result.status_code == 200
        call_kwargs = mock_client.list_invoices.call_args
        assert call_kwargs.kwargs["limit"] == 25


# ---------------------------------------------------------------------------
# Test _get_audit_log edge cases
# ---------------------------------------------------------------------------


class TestAuditLogEdgeCases:
    def _call(self, billing_handler, handler, user):
        fn = billing_handler._get_audit_log.__wrapped__.__wrapped__
        return fn(billing_handler, handler, user=user)

    def test_non_admin_role_rejected(self, billing_handler, mock_user_store):
        """Members cannot view audit logs even if tier allows it."""
        org = FakeOrganization(limits=FakeTierLimits(audit_logs=True))
        mock_user_store.get_organization_by_id.return_value = org
        handler = FakeHandler()
        user = FakeUser(role="member")
        result = self._call(billing_handler, handler, user)
        assert result.status_code == 403

    def test_pagination_params_forwarded(self, billing_handler, mock_user_store):
        """Limit and offset query params are used."""
        org = FakeOrganization(limits=FakeTierLimits(audit_logs=True))
        mock_user_store.get_organization_by_id.return_value = org
        handler = FakeHandler(query_params={"limit": "20", "offset": "10"})
        user = FakeUser(role="admin")
        result = self._call(billing_handler, handler, user)
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["limit"] == 20
        assert data["offset"] == 10

    def test_no_user_store_returns_503(self):
        handler_obj = BillingHandler(ctx={})
        handler = FakeHandler()
        fn = handler_obj._get_audit_log.__wrapped__.__wrapped__
        result = fn(handler_obj, handler, user=FakeUser())
        assert result.status_code == 503


# ---------------------------------------------------------------------------
# Test _get_subscription edge cases
# ---------------------------------------------------------------------------


class TestSubscriptionEdgeCases:
    def _call(self, billing_handler, handler, user):
        fn = billing_handler._get_subscription.__wrapped__.__wrapped__
        return fn(billing_handler, handler, user=user)

    def test_user_without_org_returns_free_tier(self, billing_handler, mock_user_store):
        """User without org gets default free subscription."""
        mock_user_store.get_user_by_id.return_value = FakeDbUser(org_id=None)
        handler = FakeHandler()
        result = self._call(billing_handler, handler, FakeUser())
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["subscription"]["tier"] == "free"

    def test_past_due_subscription(self, billing_handler, mock_user_store):
        """Past due subscription is flagged correctly."""
        sub = FakeSubscription(status="past_due")
        mock_client = MagicMock()
        mock_client.get_subscription.return_value = sub
        handler = FakeHandler()

        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            result = self._call(billing_handler, handler, FakeUser())

        data = json.loads(result.body)
        assert data["subscription"]["payment_failed"] is True
        assert data["subscription"]["is_active"] is False

    def test_trialing_subscription(self, billing_handler, mock_user_store):
        """Trialing subscription includes trial dates."""
        trial_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        trial_end = datetime(2025, 1, 15, tzinfo=timezone.utc)
        sub = FakeSubscription(
            status="trialing",
            is_trialing=True,
            trial_start=trial_start,
            trial_end=trial_end,
        )
        mock_client = MagicMock()
        mock_client.get_subscription.return_value = sub
        handler = FakeHandler()

        with patch(
            "aragora.server.handlers.billing.core.get_stripe_client",
            return_value=mock_client,
        ):
            result = self._call(billing_handler, handler, FakeUser())

        data = json.loads(result.body)
        assert data["subscription"]["is_trialing"] is True
        assert data["subscription"]["trial_start"] == trial_start.isoformat()
        assert data["subscription"]["trial_end"] == trial_end.isoformat()
        assert data["subscription"]["is_active"] is True


# ---------------------------------------------------------------------------
# Test _get_usage with usage tracker
# ---------------------------------------------------------------------------


class TestUsageWithTracker:
    def _call(self, billing_handler, handler, user):
        fn = billing_handler._get_usage.__wrapped__.__wrapped__
        return fn(billing_handler, handler, user=user)

    def test_includes_token_breakdown(self, billing_handler, mock_user_store):
        """Usage includes token breakdown from tracker."""
        mock_tracker = MagicMock()
        mock_summary = MagicMock()
        mock_summary.total_tokens_in = 50_000
        mock_summary.total_tokens_out = 10_000
        mock_summary.total_cost_usd = 0.52
        mock_summary.cost_by_provider = {"anthropic": "0.40", "openai": "0.12"}
        mock_tracker.get_summary.return_value = mock_summary
        billing_handler.ctx["usage_tracker"] = mock_tracker

        handler = FakeHandler()
        result = self._call(billing_handler, handler, FakeUser())

        data = json.loads(result.body)
        usage = data["usage"]
        assert usage["tokens_in"] == 50_000
        assert usage["tokens_out"] == 10_000
        assert usage["cost_breakdown"] is not None
        assert "cost_by_provider" in usage

    def test_user_without_org_returns_defaults(self, billing_handler, mock_user_store):
        """User with no org returns default zero usage."""
        mock_user_store.get_user_by_id.return_value = FakeDbUser(org_id=None)
        handler = FakeHandler()
        result = self._call(billing_handler, handler, FakeUser())
        data = json.loads(result.body)
        assert data["usage"]["debates_used"] == 0
        assert data["usage"]["debates_limit"] == 10


# ---------------------------------------------------------------------------
# Test webhook edge cases
# ---------------------------------------------------------------------------


class TestWebhookEdgeCases:
    def test_duplicate_event_returns_early(self, billing_handler):
        """Duplicate webhook events are skipped."""
        event = MagicMock()
        event.type = "checkout.session.completed"
        event.event_id = "evt_dup123"

        handler = FakeHandler(method="POST", headers={"Stripe-Signature": "sig"})

        with (
            patch.object(billing_handler, "validate_content_length", return_value=10),
            patch(
                "aragora.billing.stripe_client.parse_webhook_event",
                return_value=event,
            ),
            patch(
                "aragora.server.handlers.billing.core._is_duplicate_webhook",
                return_value=True,
            ),
        ):
            fn = billing_handler._handle_stripe_webhook.__wrapped__
            result = fn(billing_handler, handler)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["duplicate"] is True

    def test_unhandled_event_type_acknowledged(self, billing_handler):
        """Unknown webhook event types return 200."""
        event = MagicMock()
        event.type = "some.unknown.event"
        event.event_id = "evt_unknown"

        handler = FakeHandler(method="POST", headers={"Stripe-Signature": "sig"})

        with (
            patch.object(billing_handler, "validate_content_length", return_value=10),
            patch(
                "aragora.billing.stripe_client.parse_webhook_event",
                return_value=event,
            ),
            patch(
                "aragora.server.handlers.billing.core._is_duplicate_webhook",
                return_value=False,
            ),
            patch("aragora.server.handlers.billing.core._mark_webhook_processed"),
        ):
            fn = billing_handler._handle_stripe_webhook.__wrapped__
            result = fn(billing_handler, handler)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["received"] is True

    def test_invalid_content_length_returns_400(self, billing_handler):
        """Invalid content length is rejected."""
        handler = FakeHandler(method="POST", headers={"Stripe-Signature": "sig"})

        with patch.object(billing_handler, "validate_content_length", return_value=None):
            fn = billing_handler._handle_stripe_webhook.__wrapped__
            result = fn(billing_handler, handler)

        assert result.status_code == 400


# ---------------------------------------------------------------------------
# Test _handle_subscription_updated
# ---------------------------------------------------------------------------


class TestSubscriptionUpdated:
    def test_updates_tier_on_price_change(self, billing_handler, mock_user_store):
        """Subscription update changes org tier when price changes."""
        event = MagicMock()
        event.object = {
            "id": "sub_test123",
            "status": "active",
            "cancel_at_period_end": False,
            "items": {"data": [{"price": {"id": "price_pro"}}]},
        }

        with patch(
            "aragora.billing.stripe_client.get_tier_from_price_id",
            return_value=FakeTier.PROFESSIONAL,
        ):
            result = billing_handler._handle_subscription_updated(event, mock_user_store)

        assert result.status_code == 200
        mock_user_store.update_organization.assert_called_once()

    def test_no_update_when_no_items(self, billing_handler, mock_user_store):
        """No tier update when subscription has no items."""
        event = MagicMock()
        event.object = {
            "id": "sub_test123",
            "status": "active",
            "cancel_at_period_end": False,
            "items": {"data": []},
        }

        with patch(
            "aragora.billing.stripe_client.get_tier_from_price_id",
            return_value=None,
        ):
            result = billing_handler._handle_subscription_updated(event, mock_user_store)

        assert result.status_code == 200
        mock_user_store.update_organization.assert_not_called()


# ---------------------------------------------------------------------------
# Test _handle_invoice_finalized
# ---------------------------------------------------------------------------


class TestInvoiceFinalized:
    def test_flushes_usage_on_finalize(self, billing_handler, mock_user_store):
        """Invoice finalization triggers usage flush."""
        event = MagicMock()
        event.object = {
            "customer": "cus_test123",
            "subscription": "sub_test",
        }

        with patch("aragora.billing.usage_sync.get_usage_sync_service") as mock_sync:
            mock_sync.return_value.flush_period.return_value = ["record1", "record2"]
            result = billing_handler._handle_invoice_finalized(event, mock_user_store)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["usage_flushed"] == 2

    def test_flush_failure_is_graceful(self, billing_handler, mock_user_store):
        """Usage flush failure doesn't crash."""
        event = MagicMock()
        event.object = {
            "customer": "cus_test123",
            "subscription": "sub_test",
        }

        with patch("aragora.billing.usage_sync.get_usage_sync_service") as mock_sync:
            mock_sync.return_value.flush_period.side_effect = OSError("store down")
            result = billing_handler._handle_invoice_finalized(event, mock_user_store)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["usage_flushed"] == 0

    def test_unknown_customer_returns_success(self, billing_handler, mock_user_store):
        """Unknown customer still returns success."""
        mock_user_store.get_organization_by_stripe_customer.return_value = None
        event = MagicMock()
        event.object = {"customer": "cus_unknown", "subscription": "sub_test"}

        result = billing_handler._handle_invoice_finalized(event, mock_user_store)
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["usage_flushed"] == 0
