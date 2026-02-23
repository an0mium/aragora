"""Tests for payment handler infrastructure (aragora/server/handlers/payments/handler.py).

Comprehensive test suite covering all components in the shared payment handler module:
- ROUTES and PaymentRoutesHandler route registry
- RBAC permission constants
- PaymentProvider and PaymentStatus enums
- PaymentRequest and PaymentResult data models
- Connector management (get_stripe_connector, get_authnet_connector)
- Provider resolution (_get_provider_from_request)
- Rate limiting (_check_rate_limit, _get_client_identifier)
- Webhook idempotency (_is_duplicate_webhook, _mark_webhook_processed)
- Resilient calls (_resilient_stripe_call, _resilient_authnet_call)
- Circuit breaker configuration
- Retry configuration

100+ tests covering all branches, error paths, and edge cases.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from aragora.server.handlers.payments.handler import (
    # Route registry
    ROUTES,
    PaymentRoutesHandler,
    # RBAC permission constants
    PERM_PAYMENTS_READ,
    PERM_PAYMENTS_CHARGE,
    PERM_PAYMENTS_AUTHORIZE,
    PERM_PAYMENTS_CAPTURE,
    PERM_PAYMENTS_REFUND,
    PERM_PAYMENTS_VOID,
    PERM_PAYMENTS_ADMIN,
    PERM_CUSTOMER_READ,
    PERM_CUSTOMER_CREATE,
    PERM_CUSTOMER_UPDATE,
    PERM_CUSTOMER_DELETE,
    PERM_SUBSCRIPTION_READ,
    PERM_SUBSCRIPTION_CREATE,
    PERM_SUBSCRIPTION_UPDATE,
    PERM_SUBSCRIPTION_CANCEL,
    PERM_WEBHOOK_STRIPE,
    PERM_WEBHOOK_AUTHNET,
    PERM_BILLING_DELETE,
    PERM_BILLING_CANCEL,
    # Data models
    PaymentProvider,
    PaymentStatus,
    PaymentRequest,
    PaymentResult,
    # Connector management
    get_stripe_connector,
    get_authnet_connector,
    _get_provider_from_request,
    # Rate limiting
    _check_rate_limit,
    _get_client_identifier,
    _payment_write_limiter,
    _payment_read_limiter,
    _webhook_limiter,
    # Webhook idempotency
    _is_duplicate_webhook,
    _mark_webhook_processed,
    # Resilient calls
    _resilient_stripe_call,
    _resilient_authnet_call,
    # Circuit breakers
    _stripe_cb,
    _authnet_cb,
    # Retry config
    _payment_retry_config,
)
from aragora.resilience import JitterMode, RetryStrategy


# ===========================================================================
# Helpers
# ===========================================================================

PKG = "aragora.server.handlers.payments.handler"


def _status(resp: web.Response) -> int:
    """Extract HTTP status code from aiohttp response."""
    return resp.status


def _body(resp: web.Response) -> dict[str, Any]:
    """Extract JSON body from aiohttp response."""
    return json.loads(resp.body)


def create_mock_request(
    body: dict[str, Any] | None = None,
    query: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    user_id: str | None = None,
    transport_peername: tuple[str, int] | None = ("127.0.0.1", 12345),
    transport: Any = "default",
) -> MagicMock:
    """Create a mock aiohttp request with the given parameters."""
    request = MagicMock(spec=web.Request)
    request.query = query or {}
    request.match_info = {}
    request.app = {}
    request.headers = headers or {}

    # user_id via request.get()
    _data: dict[str, Any] = {}
    if user_id:
        _data["user_id"] = user_id
    request.get = MagicMock(side_effect=lambda k, d=None: _data.get(k, d))

    # Transport for rate limiting
    if transport == "default":
        t = MagicMock()
        t.get_extra_info.return_value = transport_peername
        request.transport = t
    elif transport is None:
        request.transport = None
    else:
        request.transport = transport

    if body is not None:
        async def json_func():
            return body

        request.json = json_func

    return request


# ===========================================================================
# TestRoutes - Route Registry
# ===========================================================================


class TestRoutes:
    """Tests for module-level ROUTES list and PaymentRoutesHandler.ROUTES."""

    def test_routes_is_list(self):
        assert isinstance(ROUTES, list)

    def test_routes_non_empty(self):
        assert len(ROUTES) > 0

    def test_routes_all_strings(self):
        for route in ROUTES:
            assert isinstance(route, str), f"Route {route!r} should be a string"

    def test_routes_all_start_with_api(self):
        for route in ROUTES:
            assert route.startswith("/api/"), f"Route {route!r} should start with /api/"

    def test_routes_contain_charge(self):
        assert "/api/payments/charge" in ROUTES
        assert "/api/v1/payments/charge" in ROUTES

    def test_routes_contain_authorize(self):
        assert "/api/payments/authorize" in ROUTES
        assert "/api/v1/payments/authorize" in ROUTES

    def test_routes_contain_capture(self):
        assert "/api/payments/capture" in ROUTES
        assert "/api/v1/payments/capture" in ROUTES

    def test_routes_contain_refund(self):
        assert "/api/payments/refund" in ROUTES
        assert "/api/v1/payments/refund" in ROUTES

    def test_routes_contain_void(self):
        assert "/api/payments/void" in ROUTES
        assert "/api/v1/payments/void" in ROUTES

    def test_routes_contain_transaction(self):
        assert "/api/payments/transaction/*" in ROUTES
        assert "/api/v1/payments/transaction/*" in ROUTES

    def test_routes_contain_customer(self):
        assert "/api/payments/customer" in ROUTES
        assert "/api/v1/payments/customer" in ROUTES

    def test_routes_contain_subscription(self):
        assert "/api/payments/subscription" in ROUTES
        assert "/api/v1/payments/subscription" in ROUTES

    def test_routes_contain_webhooks(self):
        assert "/api/payments/webhook/stripe" in ROUTES
        assert "/api/payments/webhook/authnet" in ROUTES

    def test_payment_routes_handler_class(self):
        handler = PaymentRoutesHandler()
        assert hasattr(handler, "ROUTES")

    def test_payment_routes_handler_routes_are_v1(self):
        for route in PaymentRoutesHandler.ROUTES:
            assert route.startswith("/api/v1/"), f"Route {route!r} should be versioned"

    def test_payment_routes_handler_contains_charge(self):
        assert "/api/v1/payments/charge" in PaymentRoutesHandler.ROUTES

    def test_payment_routes_handler_contains_authorize(self):
        assert "/api/v1/payments/authorize" in PaymentRoutesHandler.ROUTES

    def test_payment_routes_handler_routes_count(self):
        assert len(PaymentRoutesHandler.ROUTES) == 7


# ===========================================================================
# TestPermissionConstants - RBAC Permission Constants
# ===========================================================================


class TestPermissionConstants:
    """Tests for RBAC permission constants."""

    def test_payments_read(self):
        assert PERM_PAYMENTS_READ == "payments:read"

    def test_payments_charge(self):
        assert PERM_PAYMENTS_CHARGE == "payments:charge"

    def test_payments_authorize(self):
        assert PERM_PAYMENTS_AUTHORIZE == "payments:authorize"

    def test_payments_capture(self):
        assert PERM_PAYMENTS_CAPTURE == "payments:capture"

    def test_payments_refund(self):
        assert PERM_PAYMENTS_REFUND == "payments:refund"

    def test_payments_void(self):
        assert PERM_PAYMENTS_VOID == "payments:void"

    def test_payments_admin(self):
        assert PERM_PAYMENTS_ADMIN == "payments:admin"

    def test_customer_read(self):
        assert PERM_CUSTOMER_READ == "payments:customer:read"

    def test_customer_create(self):
        assert PERM_CUSTOMER_CREATE == "payments:customer:create"

    def test_customer_update(self):
        assert PERM_CUSTOMER_UPDATE == "payments:customer:update"

    def test_customer_delete(self):
        assert PERM_CUSTOMER_DELETE == "payments:customer:delete"

    def test_subscription_read(self):
        assert PERM_SUBSCRIPTION_READ == "payments:subscription:read"

    def test_subscription_create(self):
        assert PERM_SUBSCRIPTION_CREATE == "payments:subscription:create"

    def test_subscription_update(self):
        assert PERM_SUBSCRIPTION_UPDATE == "payments:subscription:update"

    def test_subscription_cancel(self):
        assert PERM_SUBSCRIPTION_CANCEL == "payments:subscription:cancel"

    def test_webhook_stripe(self):
        assert PERM_WEBHOOK_STRIPE == "payments:webhook:stripe"

    def test_webhook_authnet(self):
        assert PERM_WEBHOOK_AUTHNET == "payments:webhook:authnet"

    def test_billing_delete(self):
        assert PERM_BILLING_DELETE == "billing:delete"

    def test_billing_cancel(self):
        assert PERM_BILLING_CANCEL == "billing:cancel"

    def test_all_permissions_are_strings(self):
        perms = [
            PERM_PAYMENTS_READ, PERM_PAYMENTS_CHARGE, PERM_PAYMENTS_AUTHORIZE,
            PERM_PAYMENTS_CAPTURE, PERM_PAYMENTS_REFUND, PERM_PAYMENTS_VOID,
            PERM_PAYMENTS_ADMIN, PERM_CUSTOMER_READ, PERM_CUSTOMER_CREATE,
            PERM_CUSTOMER_UPDATE, PERM_CUSTOMER_DELETE, PERM_SUBSCRIPTION_READ,
            PERM_SUBSCRIPTION_CREATE, PERM_SUBSCRIPTION_UPDATE,
            PERM_SUBSCRIPTION_CANCEL, PERM_WEBHOOK_STRIPE, PERM_WEBHOOK_AUTHNET,
            PERM_BILLING_DELETE, PERM_BILLING_CANCEL,
        ]
        for perm in perms:
            assert isinstance(perm, str)

    def test_all_permissions_use_colon_separator(self):
        perms = [
            PERM_PAYMENTS_READ, PERM_PAYMENTS_CHARGE, PERM_PAYMENTS_AUTHORIZE,
            PERM_PAYMENTS_CAPTURE, PERM_PAYMENTS_REFUND, PERM_PAYMENTS_VOID,
            PERM_PAYMENTS_ADMIN, PERM_CUSTOMER_READ, PERM_CUSTOMER_CREATE,
            PERM_CUSTOMER_UPDATE, PERM_CUSTOMER_DELETE, PERM_SUBSCRIPTION_READ,
            PERM_SUBSCRIPTION_CREATE, PERM_SUBSCRIPTION_UPDATE,
            PERM_SUBSCRIPTION_CANCEL, PERM_WEBHOOK_STRIPE, PERM_WEBHOOK_AUTHNET,
            PERM_BILLING_DELETE, PERM_BILLING_CANCEL,
        ]
        for perm in perms:
            assert ":" in perm, f"Permission {perm!r} should use colon separator"


# ===========================================================================
# TestPaymentProvider - PaymentProvider Enum
# ===========================================================================


class TestPaymentProvider:
    """Tests for PaymentProvider enum."""

    def test_stripe_value(self):
        assert PaymentProvider.STRIPE.value == "stripe"

    def test_authorize_net_value(self):
        assert PaymentProvider.AUTHORIZE_NET.value == "authorize_net"

    def test_member_count(self):
        assert len(PaymentProvider) == 2

    def test_from_value_stripe(self):
        assert PaymentProvider("stripe") is PaymentProvider.STRIPE

    def test_from_value_authorize_net(self):
        assert PaymentProvider("authorize_net") is PaymentProvider.AUTHORIZE_NET

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            PaymentProvider("paypal")


# ===========================================================================
# TestPaymentStatus - PaymentStatus Enum
# ===========================================================================


class TestPaymentStatus:
    """Tests for PaymentStatus enum."""

    def test_pending_value(self):
        assert PaymentStatus.PENDING.value == "pending"

    def test_approved_value(self):
        assert PaymentStatus.APPROVED.value == "approved"

    def test_declined_value(self):
        assert PaymentStatus.DECLINED.value == "declined"

    def test_error_value(self):
        assert PaymentStatus.ERROR.value == "error"

    def test_void_value(self):
        assert PaymentStatus.VOID.value == "void"

    def test_refunded_value(self):
        assert PaymentStatus.REFUNDED.value == "refunded"

    def test_member_count(self):
        assert len(PaymentStatus) == 6

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            PaymentStatus("cancelled")


# ===========================================================================
# TestPaymentRequest - PaymentRequest Dataclass
# ===========================================================================


class TestPaymentRequest:
    """Tests for PaymentRequest dataclass."""

    def test_minimal_construction(self):
        req = PaymentRequest(amount=Decimal("10.00"))
        assert req.amount == Decimal("10.00")

    def test_default_currency(self):
        req = PaymentRequest(amount=Decimal("5.00"))
        assert req.currency == "USD"

    def test_default_description(self):
        req = PaymentRequest(amount=Decimal("5.00"))
        assert req.description is None

    def test_default_customer_id(self):
        req = PaymentRequest(amount=Decimal("5.00"))
        assert req.customer_id is None

    def test_default_payment_method(self):
        req = PaymentRequest(amount=Decimal("5.00"))
        assert req.payment_method is None

    def test_default_metadata(self):
        req = PaymentRequest(amount=Decimal("5.00"))
        assert req.metadata == {}

    def test_default_provider(self):
        req = PaymentRequest(amount=Decimal("5.00"))
        assert req.provider is PaymentProvider.STRIPE

    def test_full_construction(self):
        req = PaymentRequest(
            amount=Decimal("99.99"),
            currency="EUR",
            description="Test payment",
            customer_id="cust_123",
            payment_method={"type": "card", "card": {"number": "4242"}},
            metadata={"order_id": "ord_789"},
            provider=PaymentProvider.AUTHORIZE_NET,
        )
        assert req.amount == Decimal("99.99")
        assert req.currency == "EUR"
        assert req.description == "Test payment"
        assert req.customer_id == "cust_123"
        assert req.payment_method["type"] == "card"
        assert req.metadata["order_id"] == "ord_789"
        assert req.provider is PaymentProvider.AUTHORIZE_NET

    def test_metadata_not_shared_between_instances(self):
        req1 = PaymentRequest(amount=Decimal("1.00"))
        req2 = PaymentRequest(amount=Decimal("2.00"))
        req1.metadata["key"] = "val"
        assert "key" not in req2.metadata


# ===========================================================================
# TestPaymentResult - PaymentResult Dataclass
# ===========================================================================


class TestPaymentResult:
    """Tests for PaymentResult dataclass."""

    def test_minimal_construction(self):
        result = PaymentResult(
            transaction_id="txn_123",
            provider=PaymentProvider.STRIPE,
            status=PaymentStatus.APPROVED,
            amount=Decimal("10.00"),
            currency="USD",
        )
        assert result.transaction_id == "txn_123"
        assert result.provider is PaymentProvider.STRIPE
        assert result.status is PaymentStatus.APPROVED
        assert result.amount == Decimal("10.00")
        assert result.currency == "USD"

    def test_default_optional_fields(self):
        result = PaymentResult(
            transaction_id="txn_123",
            provider=PaymentProvider.STRIPE,
            status=PaymentStatus.APPROVED,
            amount=Decimal("10.00"),
            currency="USD",
        )
        assert result.message is None
        assert result.auth_code is None
        assert result.avs_result is None
        assert result.cvv_result is None
        assert result.metadata == {}

    def test_created_at_default(self):
        result = PaymentResult(
            transaction_id="txn_123",
            provider=PaymentProvider.STRIPE,
            status=PaymentStatus.APPROVED,
            amount=Decimal("10.00"),
            currency="USD",
        )
        assert isinstance(result.created_at, datetime)
        assert result.created_at.tzinfo is not None

    def test_full_construction(self):
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = PaymentResult(
            transaction_id="txn_456",
            provider=PaymentProvider.AUTHORIZE_NET,
            status=PaymentStatus.DECLINED,
            amount=Decimal("50.00"),
            currency="EUR",
            message="Insufficient funds",
            auth_code="AUTH123",
            avs_result="Y",
            cvv_result="M",
            created_at=ts,
            metadata={"retry": True},
        )
        assert result.message == "Insufficient funds"
        assert result.auth_code == "AUTH123"
        assert result.avs_result == "Y"
        assert result.cvv_result == "M"
        assert result.created_at == ts
        assert result.metadata["retry"] is True

    def test_to_dict_keys(self):
        result = PaymentResult(
            transaction_id="txn_123",
            provider=PaymentProvider.STRIPE,
            status=PaymentStatus.APPROVED,
            amount=Decimal("10.00"),
            currency="USD",
        )
        d = result.to_dict()
        expected_keys = {
            "transaction_id", "provider", "status", "amount", "currency",
            "message", "auth_code", "avs_result", "cvv_result",
            "created_at", "metadata",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values(self):
        ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = PaymentResult(
            transaction_id="txn_abc",
            provider=PaymentProvider.AUTHORIZE_NET,
            status=PaymentStatus.REFUNDED,
            amount=Decimal("25.50"),
            currency="GBP",
            message="Refunded",
            auth_code="A123",
            avs_result="N",
            cvv_result="P",
            created_at=ts,
            metadata={"reason": "customer_request"},
        )
        d = result.to_dict()
        assert d["transaction_id"] == "txn_abc"
        assert d["provider"] == "authorize_net"
        assert d["status"] == "refunded"
        assert d["amount"] == "25.50"
        assert d["currency"] == "GBP"
        assert d["message"] == "Refunded"
        assert d["auth_code"] == "A123"
        assert d["avs_result"] == "N"
        assert d["cvv_result"] == "P"
        assert d["created_at"] == ts.isoformat()
        assert d["metadata"] == {"reason": "customer_request"}

    def test_to_dict_none_fields(self):
        result = PaymentResult(
            transaction_id="txn_x",
            provider=PaymentProvider.STRIPE,
            status=PaymentStatus.PENDING,
            amount=Decimal("0.01"),
            currency="USD",
        )
        d = result.to_dict()
        assert d["message"] is None
        assert d["auth_code"] is None
        assert d["avs_result"] is None
        assert d["cvv_result"] is None

    def test_to_dict_amount_is_string(self):
        result = PaymentResult(
            transaction_id="txn_x",
            provider=PaymentProvider.STRIPE,
            status=PaymentStatus.APPROVED,
            amount=Decimal("100.00"),
            currency="USD",
        )
        d = result.to_dict()
        assert isinstance(d["amount"], str)

    def test_metadata_not_shared_between_instances(self):
        r1 = PaymentResult(
            transaction_id="t1", provider=PaymentProvider.STRIPE,
            status=PaymentStatus.APPROVED, amount=Decimal("1"), currency="USD",
        )
        r2 = PaymentResult(
            transaction_id="t2", provider=PaymentProvider.STRIPE,
            status=PaymentStatus.APPROVED, amount=Decimal("2"), currency="USD",
        )
        r1.metadata["key"] = "val"
        assert "key" not in r2.metadata


# ===========================================================================
# TestGetClientIdentifier - Client Identification for Rate Limiting
# ===========================================================================


class TestGetClientIdentifier:
    """Tests for _get_client_identifier."""

    def test_user_id_takes_priority(self):
        request = create_mock_request(user_id="user_42")
        assert _get_client_identifier(request) == "user:user_42"

    def test_x_forwarded_for_when_no_user_id(self):
        request = create_mock_request(
            headers={"X-Forwarded-For": "10.0.0.1, 192.168.1.1"},
        )
        assert _get_client_identifier(request) == "10.0.0.1"

    def test_x_forwarded_for_single_ip(self):
        request = create_mock_request(
            headers={"X-Forwarded-For": "203.0.113.50"},
        )
        assert _get_client_identifier(request) == "203.0.113.50"

    def test_x_forwarded_for_strips_whitespace(self):
        request = create_mock_request(
            headers={"X-Forwarded-For": "  10.0.0.1 , 192.168.1.1"},
        )
        assert _get_client_identifier(request) == "10.0.0.1"

    def test_peer_ip_fallback(self):
        request = create_mock_request(transport_peername=("192.168.1.100", 54321))
        assert _get_client_identifier(request) == "192.168.1.100"

    def test_no_transport_returns_unknown(self):
        request = create_mock_request(transport=None)
        assert _get_client_identifier(request) == "unknown"

    def test_transport_no_peername_returns_unknown(self):
        transport = MagicMock()
        transport.get_extra_info.return_value = None
        request = create_mock_request(transport=transport)
        assert _get_client_identifier(request) == "unknown"

    def test_user_id_over_forwarded_for(self):
        request = create_mock_request(
            user_id="user_priority",
            headers={"X-Forwarded-For": "10.0.0.1"},
        )
        assert _get_client_identifier(request) == "user:user_priority"


# ===========================================================================
# TestCheckRateLimit - Rate Limit Checking
# ===========================================================================


class TestCheckRateLimit:
    """Tests for _check_rate_limit."""

    def test_allowed_returns_none(self):
        limiter = MagicMock()
        limiter.is_allowed.return_value = True
        request = create_mock_request(user_id="allowed_user")
        result = _check_rate_limit(request, limiter)
        assert result is None

    def test_denied_returns_429(self):
        limiter = MagicMock()
        limiter.is_allowed.return_value = False
        request = create_mock_request(user_id="blocked_user")
        result = _check_rate_limit(request, limiter)
        assert _status(result) == 429

    def test_denied_response_body(self):
        limiter = MagicMock()
        limiter.is_allowed.return_value = False
        request = create_mock_request(user_id="blocked_user")
        result = _check_rate_limit(request, limiter)
        body = _body(result)
        assert "error" in body
        assert "rate limit" in body["error"].lower()

    def test_denied_response_retry_after_header(self):
        limiter = MagicMock()
        limiter.is_allowed.return_value = False
        request = create_mock_request(user_id="blocked_user")
        result = _check_rate_limit(request, limiter)
        assert result.headers.get("Retry-After") == "60"

    def test_calls_limiter_with_client_id(self):
        limiter = MagicMock()
        limiter.is_allowed.return_value = True
        request = create_mock_request(user_id="test_user_99")
        _check_rate_limit(request, limiter)
        limiter.is_allowed.assert_called_once_with("user:test_user_99")


# ===========================================================================
# TestRateLimiterInstances - Pre-configured Rate Limiter Instances
# ===========================================================================


class TestRateLimiterInstances:
    """Tests for module-level rate limiter instances."""

    def test_write_limiter_exists(self):
        assert _payment_write_limiter is not None

    def test_read_limiter_exists(self):
        assert _payment_read_limiter is not None

    def test_webhook_limiter_exists(self):
        assert _webhook_limiter is not None

    def test_write_limiter_rpm(self):
        assert _payment_write_limiter.rpm == 10

    def test_read_limiter_rpm(self):
        assert _payment_read_limiter.rpm == 30

    def test_webhook_limiter_rpm(self):
        assert _webhook_limiter.rpm == 100

    def test_write_limiter_is_allowed_method(self):
        assert hasattr(_payment_write_limiter, "is_allowed")

    def test_read_limiter_is_allowed_method(self):
        assert hasattr(_payment_read_limiter, "is_allowed")


# ===========================================================================
# TestWebhookIdempotency - Webhook Deduplication
# ===========================================================================


class TestWebhookIdempotency:
    """Tests for webhook idempotency functions."""

    @patch("aragora.storage.webhook_store.get_webhook_store")
    def test_is_duplicate_returns_true(self, mock_get_store):
        store = MagicMock()
        store.is_processed.return_value = True
        mock_get_store.return_value = store
        assert _is_duplicate_webhook("evt_123") is True
        store.is_processed.assert_called_once_with("evt_123")

    @patch("aragora.storage.webhook_store.get_webhook_store")
    def test_is_duplicate_returns_false(self, mock_get_store):
        store = MagicMock()
        store.is_processed.return_value = False
        mock_get_store.return_value = store
        assert _is_duplicate_webhook("evt_456") is False

    @patch("aragora.storage.webhook_store.get_webhook_store")
    def test_mark_processed_default_result(self, mock_get_store):
        store = MagicMock()
        mock_get_store.return_value = store
        _mark_webhook_processed("evt_789")
        store.mark_processed.assert_called_once_with("evt_789", "success")

    @patch("aragora.storage.webhook_store.get_webhook_store")
    def test_mark_processed_custom_result(self, mock_get_store):
        store = MagicMock()
        mock_get_store.return_value = store
        _mark_webhook_processed("evt_abc", result="failed")
        store.mark_processed.assert_called_once_with("evt_abc", "failed")


# ===========================================================================
# TestGetProviderFromRequest - Provider Resolution
# ===========================================================================


class TestGetProviderFromRequest:
    """Tests for _get_provider_from_request."""

    def test_default_is_stripe(self):
        request = create_mock_request()
        result = _get_provider_from_request(request, {})
        assert result is PaymentProvider.STRIPE

    def test_explicit_stripe(self):
        request = create_mock_request()
        result = _get_provider_from_request(request, {"provider": "stripe"})
        assert result is PaymentProvider.STRIPE

    def test_authorize_net(self):
        request = create_mock_request()
        result = _get_provider_from_request(request, {"provider": "authorize_net"})
        assert result is PaymentProvider.AUTHORIZE_NET

    def test_authnet_alias(self):
        request = create_mock_request()
        result = _get_provider_from_request(request, {"provider": "authnet"})
        assert result is PaymentProvider.AUTHORIZE_NET

    def test_case_insensitive_stripe(self):
        request = create_mock_request()
        result = _get_provider_from_request(request, {"provider": "STRIPE"})
        assert result is PaymentProvider.STRIPE

    def test_case_insensitive_authnet(self):
        request = create_mock_request()
        result = _get_provider_from_request(request, {"provider": "AUTHNET"})
        assert result is PaymentProvider.AUTHORIZE_NET

    def test_case_insensitive_authorize_net(self):
        request = create_mock_request()
        result = _get_provider_from_request(request, {"provider": "AUTHORIZE_NET"})
        assert result is PaymentProvider.AUTHORIZE_NET

    def test_unknown_provider_defaults_stripe(self):
        request = create_mock_request()
        result = _get_provider_from_request(request, {"provider": "paypal"})
        assert result is PaymentProvider.STRIPE

    def test_empty_provider_defaults_stripe(self):
        request = create_mock_request()
        result = _get_provider_from_request(request, {"provider": ""})
        assert result is PaymentProvider.STRIPE


# ===========================================================================
# TestConnectorManagement - Stripe/Authnet Connector Management
# ===========================================================================


class TestGetStripeConnector:
    """Tests for get_stripe_connector."""

    @pytest.fixture(autouse=True)
    def reset_stripe_connector(self):
        """Reset global _stripe_connector before each test."""
        import aragora.server.handlers.payments.handler as handler_mod
        original = handler_mod._stripe_connector
        handler_mod._stripe_connector = None
        yield
        handler_mod._stripe_connector = original

    @patch(f"{PKG}.os.environ", {"STRIPE_SECRET_KEY": "sk_test_123", "STRIPE_WEBHOOK_SECRET": "whsec_test"})
    @patch(f"{PKG}.StripeConnector", create=True)
    @patch(f"{PKG}.StripeCredentials", create=True)
    async def test_creates_connector(self, mock_creds, mock_connector):
        """Connector is lazily initialized."""
        # Patch the import inside the function
        mock_stripe_module = MagicMock()
        mock_stripe_module.StripeConnector = mock_connector
        mock_stripe_module.StripeCredentials = mock_creds
        with patch.dict("sys.modules", {"aragora.connectors.payments.stripe": mock_stripe_module}):
            request = create_mock_request()
            result = await get_stripe_connector(request)
            assert result is not None

    async def test_import_error_returns_none(self):
        """Returns None if stripe module not importable."""
        with patch.dict("sys.modules", {"aragora.connectors.payments.stripe": None}):
            request = create_mock_request()
            result = await get_stripe_connector(request)
            assert result is None

    async def test_cached_connector_returned(self):
        """After first init, cached connector is returned without re-init."""
        import aragora.server.handlers.payments.handler as handler_mod
        mock_connector = MagicMock()
        handler_mod._stripe_connector = mock_connector
        request = create_mock_request()
        result = await get_stripe_connector(request)
        assert result is mock_connector


class TestGetAuthnetConnector:
    """Tests for get_authnet_connector."""

    @pytest.fixture(autouse=True)
    def reset_authnet_connector(self):
        """Reset global _authnet_connector before each test."""
        import aragora.server.handlers.payments.handler as handler_mod
        original = handler_mod._authnet_connector
        handler_mod._authnet_connector = None
        yield
        handler_mod._authnet_connector = original

    async def test_import_error_returns_none(self):
        """Returns None if authnet module not importable."""
        with patch.dict("sys.modules", {"aragora.connectors.payments.authorize_net": None}):
            request = create_mock_request()
            result = await get_authnet_connector(request)
            assert result is None

    async def test_cached_connector_returned(self):
        """After first init, cached connector is returned without re-init."""
        import aragora.server.handlers.payments.handler as handler_mod
        mock_connector = MagicMock()
        handler_mod._authnet_connector = mock_connector
        request = create_mock_request()
        result = await get_authnet_connector(request)
        assert result is mock_connector

    async def test_create_returns_none(self):
        """When create_authorize_net_connector returns None, stores None and returns None."""
        mock_authnet_module = MagicMock()
        mock_authnet_module.create_authorize_net_connector.return_value = None
        with patch.dict("sys.modules", {"aragora.connectors.payments.authorize_net": mock_authnet_module}):
            request = create_mock_request()
            result = await get_authnet_connector(request)
            assert result is None

    async def test_create_returns_connector(self):
        """When create_authorize_net_connector returns a connector, caches and returns it."""
        mock_connector = MagicMock()
        mock_authnet_module = MagicMock()
        mock_authnet_module.create_authorize_net_connector.return_value = mock_connector
        with patch.dict("sys.modules", {"aragora.connectors.payments.authorize_net": mock_authnet_module}):
            request = create_mock_request()
            result = await get_authnet_connector(request)
            assert result is mock_connector

    async def test_runtime_error_returns_none(self):
        """RuntimeError during init returns None."""
        mock_authnet_module = MagicMock()
        mock_authnet_module.create_authorize_net_connector.side_effect = RuntimeError("config error")
        with patch.dict("sys.modules", {"aragora.connectors.payments.authorize_net": mock_authnet_module}):
            request = create_mock_request()
            result = await get_authnet_connector(request)
            assert result is None


# ===========================================================================
# TestCircuitBreakerConfig - Circuit Breaker Configuration
# ===========================================================================


class TestCircuitBreakerConfig:
    """Tests for circuit breaker instances."""

    def test_stripe_cb_exists(self):
        assert _stripe_cb is not None

    def test_authnet_cb_exists(self):
        assert _authnet_cb is not None

    def test_stripe_cb_has_can_execute(self):
        assert hasattr(_stripe_cb, "can_execute")

    def test_stripe_cb_has_record_success(self):
        assert hasattr(_stripe_cb, "record_success")

    def test_stripe_cb_has_record_failure(self):
        assert hasattr(_stripe_cb, "record_failure")

    def test_authnet_cb_has_can_execute(self):
        assert hasattr(_authnet_cb, "can_execute")


# ===========================================================================
# TestRetryConfig - Retry Configuration
# ===========================================================================


class TestRetryConfig:
    """Tests for payment retry configuration."""

    def test_max_retries(self):
        assert _payment_retry_config.max_retries == 2

    def test_base_delay(self):
        assert _payment_retry_config.base_delay == 0.5

    def test_max_delay(self):
        assert _payment_retry_config.max_delay == 5.0

    def test_strategy(self):
        assert _payment_retry_config.strategy == RetryStrategy.EXPONENTIAL

    def test_jitter_mode(self):
        assert _payment_retry_config.jitter_mode == JitterMode.MULTIPLICATIVE

    def test_retryable_exceptions(self):
        assert ConnectionError in _payment_retry_config.retryable_exceptions
        assert TimeoutError in _payment_retry_config.retryable_exceptions
        assert OSError in _payment_retry_config.retryable_exceptions


# ===========================================================================
# TestResilientStripeCall - Resilient Stripe API Calls
# ===========================================================================


class TestResilientStripeCall:
    """Tests for _resilient_stripe_call."""

    async def test_success(self):
        """Successful call returns result."""
        mock_func = AsyncMock(return_value={"id": "pi_123"})
        with patch.object(_stripe_cb, "can_execute", return_value=True), \
             patch.object(_stripe_cb, "record_success"), \
             patch.object(_stripe_cb, "record_failure"):
            result = await _resilient_stripe_call("charge", mock_func)
            assert result == {"id": "pi_123"}

    async def test_success_records_success(self):
        """Successful call records success on circuit breaker."""
        mock_func = AsyncMock(return_value="ok")
        with patch.object(_stripe_cb, "can_execute", return_value=True), \
             patch.object(_stripe_cb, "record_success") as mock_record, \
             patch.object(_stripe_cb, "record_failure"):
            await _resilient_stripe_call("charge", mock_func)
            mock_record.assert_called_once()

    async def test_circuit_open_raises_connection_error(self):
        """When circuit is open, raises ConnectionError immediately."""
        mock_func = AsyncMock()
        with patch.object(_stripe_cb, "can_execute", return_value=False):
            with pytest.raises(ConnectionError, match="temporarily unavailable"):
                await _resilient_stripe_call("charge", mock_func)
        mock_func.assert_not_called()

    async def test_func_failure_records_failure(self):
        """When func raises, records failure on circuit breaker."""
        mock_func = AsyncMock(side_effect=ConnectionError("timeout"))
        with patch.object(_stripe_cb, "can_execute", return_value=True), \
             patch.object(_stripe_cb, "record_success"), \
             patch.object(_stripe_cb, "record_failure") as mock_failure:
            with pytest.raises(ConnectionError):
                await _resilient_stripe_call("charge", mock_func)
            mock_failure.assert_called_once()

    async def test_passes_args_to_func(self):
        """Arguments are forwarded to the wrapped function."""
        mock_func = AsyncMock(return_value="ok")
        with patch.object(_stripe_cb, "can_execute", return_value=True), \
             patch.object(_stripe_cb, "record_success"), \
             patch.object(_stripe_cb, "record_failure"):
            await _resilient_stripe_call("charge", mock_func, "arg1", key="val")
            mock_func.assert_called_once_with("arg1", key="val")

    async def test_timeout_error_records_failure(self):
        """TimeoutError records failure."""
        mock_func = AsyncMock(side_effect=TimeoutError("timed out"))
        with patch.object(_stripe_cb, "can_execute", return_value=True), \
             patch.object(_stripe_cb, "record_success"), \
             patch.object(_stripe_cb, "record_failure") as mock_failure:
            with pytest.raises(TimeoutError):
                await _resilient_stripe_call("charge", mock_func)
            mock_failure.assert_called_once()

    async def test_value_error_records_failure(self):
        """ValueError records failure."""
        mock_func = AsyncMock(side_effect=ValueError("bad data"))
        with patch.object(_stripe_cb, "can_execute", return_value=True), \
             patch.object(_stripe_cb, "record_success"), \
             patch.object(_stripe_cb, "record_failure") as mock_failure:
            with pytest.raises(ValueError):
                await _resilient_stripe_call("charge", mock_func)
            mock_failure.assert_called_once()

    async def test_runtime_error_records_failure(self):
        """RuntimeError records failure."""
        mock_func = AsyncMock(side_effect=RuntimeError("internal"))
        with patch.object(_stripe_cb, "can_execute", return_value=True), \
             patch.object(_stripe_cb, "record_success"), \
             patch.object(_stripe_cb, "record_failure") as mock_failure:
            with pytest.raises(RuntimeError):
                await _resilient_stripe_call("charge", mock_func)
            mock_failure.assert_called_once()


# ===========================================================================
# TestResilientAuthnetCall - Resilient Authorize.net API Calls
# ===========================================================================


class TestResilientAuthnetCall:
    """Tests for _resilient_authnet_call."""

    async def test_success(self):
        """Successful call returns result."""
        mock_func = AsyncMock(return_value={"id": "txn_456"})
        with patch.object(_authnet_cb, "can_execute", return_value=True), \
             patch.object(_authnet_cb, "record_success"), \
             patch.object(_authnet_cb, "record_failure"):
            result = await _resilient_authnet_call("charge", mock_func)
            assert result == {"id": "txn_456"}

    async def test_success_records_success(self):
        """Successful call records success on circuit breaker."""
        mock_func = AsyncMock(return_value="ok")
        with patch.object(_authnet_cb, "can_execute", return_value=True), \
             patch.object(_authnet_cb, "record_success") as mock_record, \
             patch.object(_authnet_cb, "record_failure"):
            await _resilient_authnet_call("authorize", mock_func)
            mock_record.assert_called_once()

    async def test_circuit_open_raises_connection_error(self):
        """When circuit is open, raises ConnectionError immediately."""
        mock_func = AsyncMock()
        with patch.object(_authnet_cb, "can_execute", return_value=False):
            with pytest.raises(ConnectionError, match="temporarily unavailable"):
                await _resilient_authnet_call("charge", mock_func)
        mock_func.assert_not_called()

    async def test_func_failure_records_failure(self):
        """When func raises, records failure on circuit breaker."""
        mock_func = AsyncMock(side_effect=ConnectionError("timeout"))
        with patch.object(_authnet_cb, "can_execute", return_value=True), \
             patch.object(_authnet_cb, "record_success"), \
             patch.object(_authnet_cb, "record_failure") as mock_failure:
            with pytest.raises(ConnectionError):
                await _resilient_authnet_call("refund", mock_func)
            mock_failure.assert_called_once()

    async def test_passes_args_to_func(self):
        """Arguments are forwarded to the wrapped function."""
        mock_func = AsyncMock(return_value="ok")
        with patch.object(_authnet_cb, "can_execute", return_value=True), \
             patch.object(_authnet_cb, "record_success"), \
             patch.object(_authnet_cb, "record_failure"):
            await _resilient_authnet_call("void", mock_func, "a", b="c")
            mock_func.assert_called_once_with("a", b="c")

    async def test_timeout_error_records_failure(self):
        """TimeoutError records failure."""
        mock_func = AsyncMock(side_effect=TimeoutError("timed out"))
        with patch.object(_authnet_cb, "can_execute", return_value=True), \
             patch.object(_authnet_cb, "record_success"), \
             patch.object(_authnet_cb, "record_failure") as mock_failure:
            with pytest.raises(TimeoutError):
                await _resilient_authnet_call("capture", mock_func)
            mock_failure.assert_called_once()

    async def test_value_error_records_failure(self):
        """ValueError records failure."""
        mock_func = AsyncMock(side_effect=ValueError("invalid amount"))
        with patch.object(_authnet_cb, "can_execute", return_value=True), \
             patch.object(_authnet_cb, "record_success"), \
             patch.object(_authnet_cb, "record_failure") as mock_failure:
            with pytest.raises(ValueError):
                await _resilient_authnet_call("charge", mock_func)
            mock_failure.assert_called_once()

    async def test_runtime_error_records_failure(self):
        """RuntimeError records failure."""
        mock_func = AsyncMock(side_effect=RuntimeError("internal"))
        with patch.object(_authnet_cb, "can_execute", return_value=True), \
             patch.object(_authnet_cb, "record_success"), \
             patch.object(_authnet_cb, "record_failure") as mock_failure:
            with pytest.raises(RuntimeError):
                await _resilient_authnet_call("charge", mock_func)
            mock_failure.assert_called_once()


# ===========================================================================
# TestGetStripeConnectorErrors - Error Handling Branches
# ===========================================================================


class TestGetStripeConnectorErrors:
    """Tests for error branches in get_stripe_connector."""

    @pytest.fixture(autouse=True)
    def reset_stripe_connector(self):
        """Reset global _stripe_connector before each test."""
        import aragora.server.handlers.payments.handler as handler_mod
        original = handler_mod._stripe_connector
        handler_mod._stripe_connector = None
        yield
        handler_mod._stripe_connector = original

    async def test_os_error_returns_none(self):
        """OSError during init returns None."""
        mock_stripe_module = MagicMock()
        mock_stripe_module.StripeConnector.side_effect = OSError("network unreachable")
        with patch.dict("sys.modules", {"aragora.connectors.payments.stripe": mock_stripe_module}):
            request = create_mock_request()
            result = await get_stripe_connector(request)
            assert result is None

    async def test_value_error_returns_none(self):
        """ValueError during init returns None."""
        mock_stripe_module = MagicMock()
        mock_stripe_module.StripeConnector.side_effect = ValueError("bad key")
        with patch.dict("sys.modules", {"aragora.connectors.payments.stripe": mock_stripe_module}):
            request = create_mock_request()
            result = await get_stripe_connector(request)
            assert result is None

    async def test_runtime_error_returns_none(self):
        """RuntimeError during init returns None."""
        mock_stripe_module = MagicMock()
        mock_stripe_module.StripeConnector.side_effect = RuntimeError("init failed")
        with patch.dict("sys.modules", {"aragora.connectors.payments.stripe": mock_stripe_module}):
            request = create_mock_request()
            result = await get_stripe_connector(request)
            assert result is None


# ===========================================================================
# TestEdgeCases - Edge Cases and Boundaries
# ===========================================================================


class TestEdgeCases:
    """Edge case tests for various functions."""

    def test_payment_request_zero_amount(self):
        req = PaymentRequest(amount=Decimal("0.00"))
        assert req.amount == Decimal("0.00")

    def test_payment_request_negative_amount(self):
        req = PaymentRequest(amount=Decimal("-5.00"))
        assert req.amount == Decimal("-5.00")

    def test_payment_request_large_amount(self):
        req = PaymentRequest(amount=Decimal("999999999.99"))
        assert req.amount == Decimal("999999999.99")

    def test_payment_result_to_dict_is_json_serializable(self):
        result = PaymentResult(
            transaction_id="txn_json",
            provider=PaymentProvider.STRIPE,
            status=PaymentStatus.APPROVED,
            amount=Decimal("10.50"),
            currency="USD",
        )
        d = result.to_dict()
        # Should not raise
        json.dumps(d)

    def test_payment_result_to_dict_created_at_iso_format(self):
        ts = datetime(2025, 3, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = PaymentResult(
            transaction_id="txn_ts",
            provider=PaymentProvider.STRIPE,
            status=PaymentStatus.PENDING,
            amount=Decimal("1.00"),
            currency="USD",
            created_at=ts,
        )
        d = result.to_dict()
        # Should parse back to same datetime
        parsed = datetime.fromisoformat(d["created_at"])
        assert parsed == ts

    def test_provider_from_request_mixed_case(self):
        request = create_mock_request()
        result = _get_provider_from_request(request, {"provider": "AuThNeT"})
        assert result is PaymentProvider.AUTHORIZE_NET

    def test_get_client_identifier_empty_forwarded_for(self):
        """Empty X-Forwarded-For header still extracts first part."""
        request = create_mock_request(headers={"X-Forwarded-For": ""})
        # Empty string is truthy for headers.get but split gives [""]
        ident = _get_client_identifier(request)
        # Falls through to empty string from split("")[0].strip() = ""
        # which is falsy, so falls to peername
        assert ident == "127.0.0.1" or ident == ""

    def test_payment_provider_enum_is_iterable(self):
        providers = list(PaymentProvider)
        assert len(providers) == 2
        assert PaymentProvider.STRIPE in providers
        assert PaymentProvider.AUTHORIZE_NET in providers

    def test_payment_status_enum_is_iterable(self):
        statuses = list(PaymentStatus)
        assert len(statuses) == 6

    def test_routes_no_duplicates(self):
        assert len(ROUTES) == len(set(ROUTES)), "ROUTES should not contain duplicates"

    def test_webhook_routes_not_in_v1(self):
        """Webhook routes are only in non-versioned prefix."""
        v1_routes = [r for r in ROUTES if r.startswith("/api/v1/") and "webhook" in r]
        assert len(v1_routes) == 0
