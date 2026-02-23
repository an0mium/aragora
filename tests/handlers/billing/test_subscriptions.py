"""Comprehensive tests for billing subscriptions handler.

Tests for aragora/server/handlers/billing/subscriptions.py (UsageMeteringHandler).

Covers every route and code path:
- can_handle() route matching for all 6 ROUTES
- GET /api/v1/billing/usage - Current usage summary
- GET /api/v1/billing/usage/summary - Alias for usage
- GET /api/v1/billing/usage/breakdown - Detailed breakdown
- GET /api/v1/billing/limits - Current limits and usage %
- GET /api/v1/billing/usage/export - CSV and JSON export
- GET /api/v1/quotas - Quota status via QuotaManager
- Rate limiting (429)
- Method not allowed (405)
- Error handling: no user store (503), user not found (404), no org (404)
- Date parsing validation (400)
- _get_org_tier with various org/tier types
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.billing.models import SubscriptionTier
from aragora.server.handlers.billing.subscriptions import (
    UsageMeteringHandler,
    _usage_limiter,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockUser:
    """Mock user for billing tests."""

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
    """Mock organization for billing tests."""

    def __init__(
        self,
        id: str,
        name: str,
        slug: str = "test-org",
        tier: SubscriptionTier = SubscriptionTier.ENTERPRISE_PLUS,
    ):
        self.id = id
        self.name = name
        self.slug = slug
        self.tier = tier


class MockUserStore:
    """Mock user store for usage metering tests."""

    def __init__(self):
        self._users: dict[str, MockUser] = {}
        self._orgs: dict[str, MockOrganization] = {}

    def add_user(self, user: MockUser):
        self._users[user.id] = user

    def add_organization(self, org: MockOrganization):
        self._orgs[org.id] = org

    def get_user_by_id(self, user_id: str) -> MockUser | None:
        return self._users.get(user_id)

    def get_organization_by_id(self, org_id: str) -> MockOrganization | None:
        return self._orgs.get(org_id)


class MockHTTPHandler:
    """Mock HTTP handler for request simulation."""

    def __init__(
        self,
        command: str = "GET",
        query_params: dict | None = None,
    ):
        self.command = command
        self.headers: dict[str, str] = {"User-Agent": "test-agent"}
        self.client_address = ("127.0.0.1", 12345)
        self.path = ""
        self._query_params = query_params or {}

    def get(self, key: str, default=None):
        """Support for get_string_param resolution."""
        return self._query_params.get(key, default)


class MockUsageSummary:
    """Mock usage summary returned by UsageMeter."""

    def __init__(self, **kwargs):
        self._data = {
            "org_id": "org_1",
            "period_start": datetime(2025, 1, 1, tzinfo=timezone.utc).isoformat(),
            "period_end": datetime(2025, 1, 31, 23, 59, 59, tzinfo=timezone.utc).isoformat(),
            "period_type": "month",
            "tokens": {
                "input": 500000,
                "output": 250000,
                "total": 750000,
                "cost": "12.50",
            },
            "counts": {"debates": 45, "api_calls": 1500},
            "by_model": {},
            "by_provider": {},
            "by_day": {},
            "limits": {"tokens": 999999, "debates": 999, "api_calls": 9999},
            "usage_percent": {"tokens": 0.075, "debates": 0.045, "api_calls": 0.15},
        }
        self._data.update(kwargs)

    def to_dict(self) -> dict:
        return self._data


class MockUsageBreakdown:
    """Mock usage breakdown returned by UsageMeter."""

    def __init__(self, **kwargs):
        self.period_start = kwargs.get("period_start", datetime(2025, 1, 1, tzinfo=timezone.utc))
        self.period_end = kwargs.get(
            "period_end", datetime(2025, 1, 31, 23, 59, 59, tzinfo=timezone.utc)
        )
        self.total_cost = kwargs.get("total_cost", Decimal("125.50"))
        self.total_tokens = kwargs.get("total_tokens", 5000000)
        self.total_debates = kwargs.get("total_debates", 150)
        self.total_api_calls = kwargs.get("total_api_calls", 5000)
        self.by_model = kwargs.get(
            "by_model",
            [
                {
                    "model": "claude-3-opus",
                    "input_tokens": 2000000,
                    "output_tokens": 1000000,
                    "total_tokens": 3000000,
                    "cost": "80.00",
                    "requests": 100,
                }
            ],
        )
        self.by_provider = kwargs.get(
            "by_provider",
            [
                {
                    "provider": "anthropic",
                    "total_tokens": 3000000,
                    "cost": "80.00",
                    "requests": 100,
                }
            ],
        )
        self.by_day = kwargs.get(
            "by_day",
            [
                {
                    "day": "2025-01-15",
                    "total_tokens": 200000,
                    "cost": "5.00",
                    "debates": 5,
                    "api_calls": 200,
                }
            ],
        )
        self.by_user = kwargs.get("by_user", [])

    def to_dict(self) -> dict:
        return {
            "org_id": "org_1",
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "totals": {
                "cost": str(self.total_cost),
                "tokens": self.total_tokens,
                "debates": self.total_debates,
                "api_calls": self.total_api_calls,
            },
            "by_model": self.by_model,
            "by_provider": self.by_provider,
            "by_day": self.by_day,
            "by_user": self.by_user,
        }


class MockUsageLimits:
    """Mock usage limits returned by UsageMeter."""

    def __init__(self, **kwargs):
        self._data = {
            "org_id": "org_1",
            "tier": "enterprise_plus",
            "limits": {"tokens": 999999999, "debates": 999999, "api_calls": 999999},
            "used": {"tokens": 750000, "debates": 45, "api_calls": 1500},
            "percent": {"tokens": 0.075, "debates": 0.0045, "api_calls": 0.15},
            "exceeded": {"tokens": False, "debates": False, "api_calls": False},
        }
        self._data.update(kwargs)

    def to_dict(self) -> dict:
        return self._data


class MockQuotaStatus:
    """Mock quota status."""

    def __init__(
        self,
        resource: str = "debates",
        limit: int = 100,
        current: int = 45,
        remaining: int = 55,
        percentage_used: float = 45.0,
        is_exceeded: bool = False,
        is_warning: bool = False,
        period_resets_at: datetime | None = None,
    ):
        self.resource = resource
        self.limit = limit
        self.current = current
        self.remaining = remaining
        self.percentage_used = percentage_used
        self.is_exceeded = is_exceeded
        self.is_warning = is_warning
        self.period_resets_at = period_resets_at
        # QuotaPeriod enum
        self.period = MagicMock()
        self.period.value = "day"


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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _inject_user_into_rbac_decorator(monkeypatch):
    """Ensure @require_permission from aragora.rbac.decorators injects user.

    The conftest auto-auth fixture handles the permission CHECK (via patched
    _get_context_from_args), but the aragora.rbac.decorators.require_permission
    does NOT inject the resolved context into the ``user`` kwarg the way
    aragora.server.handlers.utils.decorators.require_permission does.

    This fixture wraps each decorated async method on UsageMeteringHandler so
    that the auth context gets passed as ``user``.
    """
    from aragora.rbac.models import AuthorizationContext

    mock_user = AuthorizationContext(
        user_id="test-user-001",
        user_email="test@example.com",
        org_id="test-org-001",
        roles={"admin", "owner"},
        permissions={"*"},
    )

    # The methods decorated with @require_permission that need user injection
    method_names = [
        "_get_usage",
        "_get_usage_breakdown",
        "_get_limits",
        "_get_quota_status",
        "_export_usage",
    ]

    import functools
    import inspect

    for name in method_names:
        # Get the current (already-decorated) method from the class
        original = getattr(UsageMeteringHandler, name)

        @functools.wraps(original)
        async def wrapper(*args, _orig=original, **kwargs):
            # Inject user if the function accepts it and it wasn't already set
            sig = inspect.signature(_orig)
            if "user" in sig.parameters and "user" not in kwargs:
                kwargs["user"] = mock_user
            return await _orig(*args, **kwargs)

        monkeypatch.setattr(UsageMeteringHandler, name, wrapper)

    yield


@pytest.fixture
def user_store():
    """Create a user store with standard test data."""
    store = MockUserStore()

    # The conftest auto-auth context uses user_id="test-user-001"
    auth_user = MockUser(id="test-user-001", email="test@example.com", role="owner", org_id="org_1")
    store.add_user(auth_user)

    org = MockOrganization(
        id="org_1",
        name="Test Org",
        slug="test-org",
        tier=SubscriptionTier.ENTERPRISE_PLUS,
    )
    store.add_organization(org)

    return store


@pytest.fixture
def mock_meter():
    """Create a mock usage meter."""
    meter = AsyncMock()
    meter.get_usage_summary = AsyncMock(return_value=MockUsageSummary())
    meter.get_usage_breakdown = AsyncMock(return_value=MockUsageBreakdown())
    meter.get_usage_limits = AsyncMock(return_value=MockUsageLimits())
    return meter


@pytest.fixture
def handler(user_store, mock_meter):
    """Create a UsageMeteringHandler with a user store and mock meter."""
    h = UsageMeteringHandler(ctx={"user_store": user_store})
    h._get_usage_meter = MagicMock(return_value=mock_meter)
    return h


@pytest.fixture
def handler_no_store():
    """UsageMeteringHandler without a user store (service unavailable scenario)."""
    h = UsageMeteringHandler(ctx={})
    return h


@pytest.fixture(autouse=True)
def _clear_rate_limiter():
    """Clear the rate limiter between tests to avoid cross-test pollution."""
    _usage_limiter._buckets.clear()
    yield
    _usage_limiter._buckets.clear()


# ===========================================================================
# TestCanHandle - route matching
# ===========================================================================


class TestCanHandle:
    """Tests for route matching via can_handle()."""

    def test_billing_usage_route(self, handler):
        assert handler.can_handle("/api/v1/billing/usage")

    def test_billing_usage_breakdown_route(self, handler):
        assert handler.can_handle("/api/v1/billing/usage/breakdown")

    def test_billing_limits_route(self, handler):
        assert handler.can_handle("/api/v1/billing/limits")

    def test_billing_usage_summary_route(self, handler):
        assert handler.can_handle("/api/v1/billing/usage/summary")

    def test_billing_usage_export_route(self, handler):
        assert handler.can_handle("/api/v1/billing/usage/export")

    def test_quotas_route(self, handler):
        assert handler.can_handle("/api/v1/quotas")

    def test_unrelated_path_rejected(self, handler):
        assert not handler.can_handle("/api/v1/debates")

    def test_partial_billing_path_rejected(self, handler):
        assert not handler.can_handle("/api/v1/billing")

    def test_unversioned_path_rejected(self, handler):
        assert not handler.can_handle("/api/billing/usage")

    def test_empty_path_rejected(self, handler):
        assert not handler.can_handle("")

    def test_billing_plans_not_handled(self, handler):
        """Plans is a core billing route, not handled by this handler."""
        assert not handler.can_handle("/api/v1/billing/plans")

    def test_routes_list_has_6_entries(self, handler):
        assert len(handler.ROUTES) == 6


# ===========================================================================
# TestGetUsage
# ===========================================================================


class TestGetUsage:
    """Tests for GET /api/v1/billing/usage."""

    async def test_returns_usage_data(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage", {}, http, method="GET")
        body = _body(result)
        assert "usage" in body

    async def test_usage_summary_has_expected_fields(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage", {}, http, method="GET")
        body = _body(result)
        usage = body["usage"]
        assert "tokens" in usage
        assert "counts" in usage

    async def test_usage_tokens_breakdown(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage", {}, http, method="GET")
        body = _body(result)
        tokens = body["usage"]["tokens"]
        assert tokens["input"] == 500000
        assert tokens["output"] == 250000
        assert tokens["total"] == 750000

    async def test_usage_counts(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage", {}, http, method="GET")
        body = _body(result)
        counts = body["usage"]["counts"]
        assert counts["debates"] == 45
        assert counts["api_calls"] == 1500

    async def test_usage_with_period_param(self, handler, mock_meter):
        http = MockHTTPHandler(query_params={"period": "week"})
        result = await handler.handle("/api/v1/billing/usage", {}, http, method="GET")
        assert _status(result) == 200
        mock_meter.get_usage_summary.assert_awaited_once()
        call_kwargs = mock_meter.get_usage_summary.call_args[1]
        assert call_kwargs["period"] == "week"

    async def test_usage_default_period_is_month(self, handler, mock_meter):
        http = MockHTTPHandler()
        await handler.handle("/api/v1/billing/usage", {}, http, method="GET")
        call_kwargs = mock_meter.get_usage_summary.call_args[1]
        assert call_kwargs["period"] == "month"

    async def test_usage_passes_org_id(self, handler, mock_meter):
        http = MockHTTPHandler()
        await handler.handle("/api/v1/billing/usage", {}, http, method="GET")
        call_kwargs = mock_meter.get_usage_summary.call_args[1]
        assert call_kwargs["org_id"] == "org_1"

    async def test_usage_passes_tier(self, handler, mock_meter):
        http = MockHTTPHandler()
        await handler.handle("/api/v1/billing/usage", {}, http, method="GET")
        call_kwargs = mock_meter.get_usage_summary.call_args[1]
        assert call_kwargs["tier"] == "enterprise_plus"

    async def test_no_user_store_returns_503(self, handler_no_store):
        http = MockHTTPHandler()
        result = await handler_no_store.handle("/api/v1/billing/usage", {}, http, method="GET")
        assert _status(result) == 503

    async def test_unknown_user_returns_404(self):
        store = MockUserStore()
        h = UsageMeteringHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = await h.handle("/api/v1/billing/usage", {}, http, method="GET")
        assert _status(result) == 404

    async def test_user_without_org_returns_404(self):
        store = MockUserStore()
        store.add_user(MockUser(id="test-user-001", email="t@t.com", org_id=None))
        h = UsageMeteringHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = await h.handle("/api/v1/billing/usage", {}, http, method="GET")
        assert _status(result) == 404

    async def test_user_with_org_id_but_org_not_found_returns_404(self):
        store = MockUserStore()
        store.add_user(MockUser(id="test-user-001", email="t@t.com", org_id="nonexistent"))
        h = UsageMeteringHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = await h.handle("/api/v1/billing/usage", {}, http, method="GET")
        assert _status(result) == 404

    async def test_status_code_is_200(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage", {}, http, method="GET")
        assert _status(result) == 200


# ===========================================================================
# TestGetUsageSummary
# ===========================================================================


class TestGetUsageSummary:
    """Tests for GET /api/v1/billing/usage/summary (alias for usage)."""

    async def test_summary_returns_usage_data(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage/summary", {}, http, method="GET")
        body = _body(result)
        assert "usage" in body
        assert _status(result) == 200

    async def test_summary_routes_to_get_usage(self, handler, mock_meter):
        http = MockHTTPHandler()
        await handler.handle("/api/v1/billing/usage/summary", {}, http, method="GET")
        mock_meter.get_usage_summary.assert_awaited()

    async def test_summary_no_user_store_returns_503(self, handler_no_store):
        http = MockHTTPHandler()
        result = await handler_no_store.handle(
            "/api/v1/billing/usage/summary", {}, http, method="GET"
        )
        assert _status(result) == 503


# ===========================================================================
# TestGetUsageBreakdown
# ===========================================================================


class TestGetUsageBreakdown:
    """Tests for GET /api/v1/billing/usage/breakdown."""

    async def test_returns_breakdown(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage/breakdown", {}, http, method="GET")
        body = _body(result)
        assert "breakdown" in body

    async def test_breakdown_has_totals(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage/breakdown", {}, http, method="GET")
        body = _body(result)
        totals = body["breakdown"]["totals"]
        assert "cost" in totals
        assert "tokens" in totals
        assert "debates" in totals
        assert "api_calls" in totals

    async def test_breakdown_has_by_model(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage/breakdown", {}, http, method="GET")
        body = _body(result)
        assert "by_model" in body["breakdown"]

    async def test_breakdown_has_by_provider(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage/breakdown", {}, http, method="GET")
        body = _body(result)
        assert "by_provider" in body["breakdown"]

    async def test_breakdown_has_by_day(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage/breakdown", {}, http, method="GET")
        body = _body(result)
        assert "by_day" in body["breakdown"]

    async def test_breakdown_with_valid_start_date(self, handler, mock_meter):
        http = MockHTTPHandler(query_params={"start": "2025-01-01T00:00:00Z"})
        result = await handler.handle("/api/v1/billing/usage/breakdown", {}, http, method="GET")
        assert _status(result) == 200
        call_kwargs = mock_meter.get_usage_breakdown.call_args[1]
        assert call_kwargs["start_date"] is not None

    async def test_breakdown_with_valid_end_date(self, handler, mock_meter):
        http = MockHTTPHandler(query_params={"end": "2025-01-31T23:59:59Z"})
        result = await handler.handle("/api/v1/billing/usage/breakdown", {}, http, method="GET")
        assert _status(result) == 200
        call_kwargs = mock_meter.get_usage_breakdown.call_args[1]
        assert call_kwargs["end_date"] is not None

    async def test_breakdown_with_both_dates(self, handler, mock_meter):
        http = MockHTTPHandler(
            query_params={
                "start": "2025-01-01T00:00:00+00:00",
                "end": "2025-01-31T23:59:59+00:00",
            }
        )
        result = await handler.handle("/api/v1/billing/usage/breakdown", {}, http, method="GET")
        assert _status(result) == 200
        call_kwargs = mock_meter.get_usage_breakdown.call_args[1]
        assert call_kwargs["start_date"] is not None
        assert call_kwargs["end_date"] is not None

    async def test_breakdown_without_dates_passes_none(self, handler, mock_meter):
        http = MockHTTPHandler()
        await handler.handle("/api/v1/billing/usage/breakdown", {}, http, method="GET")
        call_kwargs = mock_meter.get_usage_breakdown.call_args[1]
        assert call_kwargs["start_date"] is None
        assert call_kwargs["end_date"] is None

    async def test_breakdown_invalid_start_date_returns_400(self, handler):
        http = MockHTTPHandler(query_params={"start": "not-a-date"})
        result = await handler.handle("/api/v1/billing/usage/breakdown", {}, http, method="GET")
        assert _status(result) == 400
        body = _body(result)
        assert "start date" in body.get("error", "").lower()

    async def test_breakdown_invalid_end_date_returns_400(self, handler):
        http = MockHTTPHandler(query_params={"end": "invalid"})
        result = await handler.handle("/api/v1/billing/usage/breakdown", {}, http, method="GET")
        assert _status(result) == 400
        body = _body(result)
        assert "end date" in body.get("error", "").lower()

    async def test_breakdown_no_user_store_returns_503(self, handler_no_store):
        http = MockHTTPHandler()
        result = await handler_no_store.handle(
            "/api/v1/billing/usage/breakdown", {}, http, method="GET"
        )
        assert _status(result) == 503

    async def test_breakdown_unknown_user_returns_404(self):
        store = MockUserStore()
        h = UsageMeteringHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = await h.handle("/api/v1/billing/usage/breakdown", {}, http, method="GET")
        assert _status(result) == 404

    async def test_breakdown_user_no_org_returns_404(self):
        store = MockUserStore()
        store.add_user(MockUser(id="test-user-001", email="t@t.com", org_id=None))
        h = UsageMeteringHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = await h.handle("/api/v1/billing/usage/breakdown", {}, http, method="GET")
        assert _status(result) == 404

    async def test_breakdown_status_code_is_200(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage/breakdown", {}, http, method="GET")
        assert _status(result) == 200


# ===========================================================================
# TestGetLimits
# ===========================================================================


class TestGetLimits:
    """Tests for GET /api/v1/billing/limits."""

    async def test_returns_limits(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/limits", {}, http, method="GET")
        body = _body(result)
        assert "limits" in body

    async def test_limits_has_tier(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/limits", {}, http, method="GET")
        body = _body(result)
        assert "tier" in body["limits"]

    async def test_limits_has_used(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/limits", {}, http, method="GET")
        body = _body(result)
        assert "used" in body["limits"]

    async def test_limits_has_percent(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/limits", {}, http, method="GET")
        body = _body(result)
        assert "percent" in body["limits"]

    async def test_limits_has_exceeded(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/limits", {}, http, method="GET")
        body = _body(result)
        assert "exceeded" in body["limits"]

    async def test_limits_passes_org_id_and_tier(self, handler, mock_meter):
        http = MockHTTPHandler()
        await handler.handle("/api/v1/billing/limits", {}, http, method="GET")
        call_kwargs = mock_meter.get_usage_limits.call_args[1]
        assert call_kwargs["org_id"] == "org_1"
        assert call_kwargs["tier"] == "enterprise_plus"

    async def test_limits_no_user_store_returns_503(self, handler_no_store):
        http = MockHTTPHandler()
        result = await handler_no_store.handle("/api/v1/billing/limits", {}, http, method="GET")
        assert _status(result) == 503

    async def test_limits_unknown_user_returns_404(self):
        store = MockUserStore()
        h = UsageMeteringHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = await h.handle("/api/v1/billing/limits", {}, http, method="GET")
        assert _status(result) == 404

    async def test_limits_user_no_org_returns_404(self):
        store = MockUserStore()
        store.add_user(MockUser(id="test-user-001", email="t@t.com", org_id=None))
        h = UsageMeteringHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = await h.handle("/api/v1/billing/limits", {}, http, method="GET")
        assert _status(result) == 404

    async def test_limits_status_code_is_200(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/limits", {}, http, method="GET")
        assert _status(result) == 200


# ===========================================================================
# TestExportUsage
# ===========================================================================


class TestExportUsage:
    """Tests for GET /api/v1/billing/usage/export."""

    async def test_export_csv_returns_200(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert _status(result) == 200

    async def test_export_csv_content_type(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert result.content_type == "text/csv"

    async def test_export_csv_has_content_disposition(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert "Content-Disposition" in result.headers
        assert "usage_export_" in result.headers["Content-Disposition"]
        assert "test-org" in result.headers["Content-Disposition"]

    async def test_export_csv_contains_header_row(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        csv_text = result.body.decode("utf-8")
        assert "Usage Export Report" in csv_text

    async def test_export_csv_contains_org_name(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        csv_text = result.body.decode("utf-8")
        assert "Test Org" in csv_text

    async def test_export_csv_contains_summary(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        csv_text = result.body.decode("utf-8")
        assert "Summary" in csv_text
        assert "Total Cost (USD)" in csv_text
        assert "Total Tokens" in csv_text
        assert "Total Debates" in csv_text
        assert "Total API Calls" in csv_text

    async def test_export_csv_contains_model_section(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        csv_text = result.body.decode("utf-8")
        assert "Usage by Model" in csv_text
        assert "claude-3-opus" in csv_text

    async def test_export_csv_contains_provider_section(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        csv_text = result.body.decode("utf-8")
        assert "Usage by Provider" in csv_text
        assert "anthropic" in csv_text

    async def test_export_csv_contains_daily_section(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        csv_text = result.body.decode("utf-8")
        assert "Daily Usage" in csv_text
        assert "2025-01-15" in csv_text

    async def test_export_json_format(self, handler):
        http = MockHTTPHandler(query_params={"format": "json"})
        result = await handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert _status(result) == 200
        body = _body(result)
        # JSON export returns the breakdown dict directly
        assert "totals" in body

    async def test_export_json_has_breakdown_fields(self, handler):
        http = MockHTTPHandler(query_params={"format": "json"})
        result = await handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        body = _body(result)
        assert "by_model" in body
        assert "by_provider" in body
        assert "by_day" in body

    async def test_export_with_valid_start_date(self, handler, mock_meter):
        http = MockHTTPHandler(query_params={"start": "2025-01-01T00:00:00Z"})
        result = await handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert _status(result) == 200
        call_kwargs = mock_meter.get_usage_breakdown.call_args[1]
        assert call_kwargs["start_date"] is not None

    async def test_export_with_valid_end_date(self, handler, mock_meter):
        http = MockHTTPHandler(query_params={"end": "2025-01-31T23:59:59Z"})
        result = await handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert _status(result) == 200
        call_kwargs = mock_meter.get_usage_breakdown.call_args[1]
        assert call_kwargs["end_date"] is not None

    async def test_export_invalid_start_date_returns_400(self, handler):
        http = MockHTTPHandler(query_params={"start": "yesterday"})
        result = await handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert _status(result) == 400

    async def test_export_invalid_end_date_returns_400(self, handler):
        http = MockHTTPHandler(query_params={"end": "tomorrow"})
        result = await handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert _status(result) == 400

    async def test_export_no_user_store_returns_503(self, handler_no_store):
        http = MockHTTPHandler()
        result = await handler_no_store.handle(
            "/api/v1/billing/usage/export", {}, http, method="GET"
        )
        assert _status(result) == 503

    async def test_export_unknown_user_returns_404(self):
        store = MockUserStore()
        h = UsageMeteringHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = await h.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert _status(result) == 404

    async def test_export_user_no_org_returns_404(self):
        store = MockUserStore()
        store.add_user(MockUser(id="test-user-001", email="t@t.com", org_id=None))
        h = UsageMeteringHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = await h.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert _status(result) == 404

    async def test_export_default_format_is_csv(self, handler):
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert result.content_type == "text/csv"

    async def test_export_csv_with_empty_breakdowns(self, handler, mock_meter):
        """Test CSV export when breakdowns have no entries."""
        mock_meter.get_usage_breakdown = AsyncMock(
            return_value=MockUsageBreakdown(by_model=[], by_provider=[], by_day=[])
        )
        http = MockHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage/export", {}, http, method="GET")
        assert _status(result) == 200
        csv_text = result.body.decode("utf-8")
        assert "Usage by Model" in csv_text


# ===========================================================================
# TestGetQuotaStatus
# ===========================================================================


class TestGetQuotaStatus:
    """Tests for GET /api/v1/quotas."""

    async def test_returns_quotas(self, handler):
        mock_status = MockQuotaStatus()
        mock_manager = AsyncMock()
        mock_manager.get_quota_status = AsyncMock(return_value=mock_status)
        with patch(
            "aragora.server.middleware.tier_enforcement.get_quota_manager",
            return_value=mock_manager,
        ):
            http = MockHTTPHandler()
            result = await handler.handle("/api/v1/quotas", {}, http, method="GET")
        body = _body(result)
        assert "quotas" in body
        assert _status(result) == 200

    async def test_quota_checks_all_resources(self, handler):
        mock_status = MockQuotaStatus()
        mock_manager = AsyncMock()
        mock_manager.get_quota_status = AsyncMock(return_value=mock_status)
        with patch(
            "aragora.server.middleware.tier_enforcement.get_quota_manager",
            return_value=mock_manager,
        ):
            http = MockHTTPHandler()
            await handler.handle("/api/v1/quotas", {}, http, method="GET")
        # Should check 5 resources
        assert mock_manager.get_quota_status.await_count == 5

    async def test_quota_resource_structure(self, handler):
        mock_status = MockQuotaStatus(limit=100, current=45, remaining=55, percentage_used=45.0)
        mock_manager = AsyncMock()
        mock_manager.get_quota_status = AsyncMock(return_value=mock_status)
        with patch(
            "aragora.server.middleware.tier_enforcement.get_quota_manager",
            return_value=mock_manager,
        ):
            http = MockHTTPHandler()
            result = await handler.handle("/api/v1/quotas", {}, http, method="GET")
        body = _body(result)
        # Pick one resource to check structure
        quotas = body["quotas"]
        assert len(quotas) > 0
        for resource_name, resource_data in quotas.items():
            assert "limit" in resource_data
            assert "current" in resource_data
            assert "remaining" in resource_data
            assert "period" in resource_data
            assert "percentage_used" in resource_data
            assert "is_exceeded" in resource_data
            assert "is_warning" in resource_data
            assert "resets_at" in resource_data

    async def test_quota_with_resets_at(self, handler):
        reset_time = datetime(2025, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
        mock_status = MockQuotaStatus(period_resets_at=reset_time)
        mock_manager = AsyncMock()
        mock_manager.get_quota_status = AsyncMock(return_value=mock_status)
        with patch(
            "aragora.server.middleware.tier_enforcement.get_quota_manager",
            return_value=mock_manager,
        ):
            http = MockHTTPHandler()
            result = await handler.handle("/api/v1/quotas", {}, http, method="GET")
        body = _body(result)
        for resource_data in body["quotas"].values():
            assert resource_data["resets_at"] == reset_time.isoformat()

    async def test_quota_with_no_resets_at(self, handler):
        mock_status = MockQuotaStatus(period_resets_at=None)
        mock_manager = AsyncMock()
        mock_manager.get_quota_status = AsyncMock(return_value=mock_status)
        with patch(
            "aragora.server.middleware.tier_enforcement.get_quota_manager",
            return_value=mock_manager,
        ):
            http = MockHTTPHandler()
            result = await handler.handle("/api/v1/quotas", {}, http, method="GET")
        body = _body(result)
        for resource_data in body["quotas"].values():
            assert resource_data["resets_at"] is None

    async def test_quota_resource_error_skipped_gracefully(self, handler):
        """When a resource check raises, it should be skipped."""
        call_count = 0

        async def flaky_quota_status(resource, tenant_id=None):
            nonlocal call_count
            call_count += 1
            if resource == "tokens":
                raise RuntimeError("Redis down")
            return MockQuotaStatus()

        mock_manager = AsyncMock()
        mock_manager.get_quota_status = flaky_quota_status
        with patch(
            "aragora.server.middleware.tier_enforcement.get_quota_manager",
            return_value=mock_manager,
        ):
            http = MockHTTPHandler()
            result = await handler.handle("/api/v1/quotas", {}, http, method="GET")
        body = _body(result)
        assert _status(result) == 200
        # tokens should be missing since it errored
        assert "tokens" not in body["quotas"]
        # Other resources should be present
        assert len(body["quotas"]) == 4

    async def test_quota_all_resources_error_returns_empty(self, handler):
        """When all resources fail, should return empty quotas."""
        mock_manager = AsyncMock()
        mock_manager.get_quota_status = AsyncMock(side_effect=RuntimeError("All down"))
        with patch(
            "aragora.server.middleware.tier_enforcement.get_quota_manager",
            return_value=mock_manager,
        ):
            http = MockHTTPHandler()
            result = await handler.handle("/api/v1/quotas", {}, http, method="GET")
        body = _body(result)
        assert body["quotas"] == {}
        assert _status(result) == 200

    async def test_quota_none_status_skipped(self, handler):
        """When get_quota_status returns None, resource should be skipped."""
        mock_manager = AsyncMock()
        mock_manager.get_quota_status = AsyncMock(return_value=None)
        with patch(
            "aragora.server.middleware.tier_enforcement.get_quota_manager",
            return_value=mock_manager,
        ):
            http = MockHTTPHandler()
            result = await handler.handle("/api/v1/quotas", {}, http, method="GET")
        body = _body(result)
        assert body["quotas"] == {}

    async def test_quota_no_user_store_returns_503(self, handler_no_store):
        http = MockHTTPHandler()
        result = await handler_no_store.handle("/api/v1/quotas", {}, http, method="GET")
        assert _status(result) == 503

    async def test_quota_unknown_user_returns_404(self):
        store = MockUserStore()
        h = UsageMeteringHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = await h.handle("/api/v1/quotas", {}, http, method="GET")
        assert _status(result) == 404

    async def test_quota_user_no_org_returns_404(self):
        store = MockUserStore()
        store.add_user(MockUser(id="test-user-001", email="t@t.com", org_id=None))
        h = UsageMeteringHandler(ctx={"user_store": store})
        http = MockHTTPHandler()
        result = await h.handle("/api/v1/quotas", {}, http, method="GET")
        assert _status(result) == 404

    async def test_quota_value_error_skipped(self, handler):
        """ValueError should also be caught gracefully."""

        async def value_error_status(resource, tenant_id=None):
            if resource == "debates":
                raise ValueError("Bad value")
            return MockQuotaStatus()

        mock_manager = AsyncMock()
        mock_manager.get_quota_status = value_error_status
        with patch(
            "aragora.server.middleware.tier_enforcement.get_quota_manager",
            return_value=mock_manager,
        ):
            http = MockHTTPHandler()
            result = await handler.handle("/api/v1/quotas", {}, http, method="GET")
        body = _body(result)
        assert "debates" not in body["quotas"]
        assert _status(result) == 200


# ===========================================================================
# TestMethodNotAllowed
# ===========================================================================


class TestMethodNotAllowed:
    """Tests for method-not-allowed responses (405)."""

    async def test_post_to_usage_returns_405(self, handler):
        http = MockHTTPHandler(command="POST")
        result = await handler.handle("/api/v1/billing/usage", {}, http, method="POST")
        assert _status(result) == 405

    async def test_post_to_breakdown_returns_405(self, handler):
        http = MockHTTPHandler(command="POST")
        result = await handler.handle("/api/v1/billing/usage/breakdown", {}, http, method="POST")
        assert _status(result) == 405

    async def test_post_to_limits_returns_405(self, handler):
        http = MockHTTPHandler(command="POST")
        result = await handler.handle("/api/v1/billing/limits", {}, http, method="POST")
        assert _status(result) == 405

    async def test_post_to_summary_returns_405(self, handler):
        http = MockHTTPHandler(command="POST")
        result = await handler.handle("/api/v1/billing/usage/summary", {}, http, method="POST")
        assert _status(result) == 405

    async def test_post_to_export_returns_405(self, handler):
        http = MockHTTPHandler(command="POST")
        result = await handler.handle("/api/v1/billing/usage/export", {}, http, method="POST")
        assert _status(result) == 405

    async def test_post_to_quotas_returns_405(self, handler):
        http = MockHTTPHandler(command="POST")
        result = await handler.handle("/api/v1/quotas", {}, http, method="POST")
        assert _status(result) == 405

    async def test_delete_to_usage_returns_405(self, handler):
        http = MockHTTPHandler(command="DELETE")
        result = await handler.handle("/api/v1/billing/usage", {}, http, method="DELETE")
        assert _status(result) == 405

    async def test_put_to_limits_returns_405(self, handler):
        http = MockHTTPHandler(command="PUT")
        result = await handler.handle("/api/v1/billing/limits", {}, http, method="PUT")
        assert _status(result) == 405


# ===========================================================================
# TestRateLimiting
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    async def test_rate_limit_returns_429(self, handler):
        mock_limiter = MagicMock()
        mock_limiter.is_allowed.return_value = False
        with patch(
            "aragora.server.handlers.billing.subscriptions._usage_limiter",
            mock_limiter,
        ):
            http = MockHTTPHandler()
            result = await handler.handle("/api/v1/billing/usage", {}, http, method="GET")
        assert _status(result) == 429

    async def test_rate_limit_error_message(self, handler):
        mock_limiter = MagicMock()
        mock_limiter.is_allowed.return_value = False
        with patch(
            "aragora.server.handlers.billing.subscriptions._usage_limiter",
            mock_limiter,
        ):
            http = MockHTTPHandler()
            result = await handler.handle("/api/v1/billing/usage", {}, http, method="GET")
        body = _body(result)
        assert "rate limit" in body.get("error", "").lower()

    async def test_rate_limit_applies_to_all_routes(self, handler):
        """Rate limiting should apply to all routes."""
        mock_limiter = MagicMock()
        mock_limiter.is_allowed.return_value = False
        routes = [
            "/api/v1/billing/usage",
            "/api/v1/billing/usage/breakdown",
            "/api/v1/billing/limits",
            "/api/v1/billing/usage/export",
            "/api/v1/quotas",
        ]
        with patch(
            "aragora.server.handlers.billing.subscriptions._usage_limiter",
            mock_limiter,
        ):
            for route in routes:
                http = MockHTTPHandler()
                result = await handler.handle(route, {}, http, method="GET")
                assert _status(result) == 429, f"Rate limit not applied to {route}"


# ===========================================================================
# TestHandlerInit
# ===========================================================================


class TestHandlerInit:
    """Tests for handler initialization and context."""

    def test_default_context_is_empty_dict(self):
        h = UsageMeteringHandler()
        assert h.ctx == {}

    def test_context_passed_through(self):
        ctx = {"user_store": MagicMock()}
        h = UsageMeteringHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_get_user_store_returns_from_context(self, handler, user_store):
        assert handler._get_user_store() is user_store

    def test_get_user_store_returns_none_when_missing(self, handler_no_store):
        assert handler_no_store._get_user_store() is None

    def test_resource_type_is_billing_usage(self, handler):
        assert handler.RESOURCE_TYPE == "billing_usage"


# ===========================================================================
# TestGetOrgTier
# ===========================================================================


class TestGetOrgTier:
    """Tests for _get_org_tier helper method."""

    def test_org_none_returns_free(self, handler):
        assert handler._get_org_tier(None) == "free"

    def test_org_with_subscription_tier_enum(self, handler):
        org = MockOrganization(id="o1", name="Test", tier=SubscriptionTier.ENTERPRISE_PLUS)
        assert handler._get_org_tier(org) == "enterprise_plus"

    def test_org_with_string_tier(self, handler):
        org = MagicMock()
        org.tier = "professional"
        assert handler._get_org_tier(org) == "professional"

    def test_org_with_none_tier(self, handler):
        org = MagicMock()
        org.tier = None
        assert handler._get_org_tier(org) == "free"

    def test_org_with_free_tier(self, handler):
        org = MockOrganization(id="o1", name="Free Org", tier=SubscriptionTier.FREE)
        assert handler._get_org_tier(org) == "free"

    def test_org_with_starter_tier(self, handler):
        org = MockOrganization(id="o1", name="Starter Org", tier=SubscriptionTier.STARTER)
        assert handler._get_org_tier(org) == "starter"

    def test_org_with_enterprise_tier(self, handler):
        org = MockOrganization(id="o1", name="Ent Org", tier=SubscriptionTier.ENTERPRISE)
        assert handler._get_org_tier(org) == "enterprise"


# ===========================================================================
# TestHandlerCommandOverride
# ===========================================================================


class TestHandlerCommandOverride:
    """Tests for HTTP method detection from handler.command attribute."""

    async def test_method_from_handler_command_attribute(self, handler):
        """When handler has command attribute, it should be used."""
        http = MockHTTPHandler(command="GET")
        result = await handler.handle("/api/v1/billing/usage", {}, http, method="POST")
        # handler.command="GET" overrides method="POST"
        assert _status(result) == 200

    async def test_method_not_overridden_when_no_command(self, handler):
        """When handler has no command attribute, method param is used."""
        http = MockHTTPHandler(command="GET")
        delattr(http, "command")
        result = await handler.handle("/api/v1/billing/usage", {}, http, method="GET")
        assert _status(result) == 200


# ===========================================================================
# TestEmitHandlerEvent
# ===========================================================================


class TestEmitHandlerEvent:
    """Tests for handler event emission."""

    async def test_usage_endpoint_emits_event(self, handler):
        with patch("aragora.server.handlers.billing.subscriptions.emit_handler_event") as mock_emit:
            http = MockHTTPHandler()
            await handler.handle("/api/v1/billing/usage", {}, http, method="GET")
        mock_emit.assert_called_once_with("billing", "queried", {"endpoint": "usage"})

    async def test_summary_endpoint_does_not_emit_event(self, handler):
        """The /usage/summary alias does NOT emit a handler event."""
        with patch("aragora.server.handlers.billing.subscriptions.emit_handler_event") as mock_emit:
            http = MockHTTPHandler()
            await handler.handle("/api/v1/billing/usage/summary", {}, http, method="GET")
        mock_emit.assert_not_called()
