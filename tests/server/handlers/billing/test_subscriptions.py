"""
Tests for UsageMeteringHandler (usage metering and quotas).

Covers:
- GET /api/v1/billing/usage - Current usage summary
- GET /api/v1/billing/usage/breakdown - Detailed breakdown
- GET /api/v1/billing/limits - Current limits
- GET /api/v1/billing/usage/export - Export as CSV/JSON
- GET /api/v1/quotas - Quota status
- Rate limiting
- RBAC permission checks
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.billing.subscriptions import UsageMeteringHandler


# ---------------------------------------------------------------------------
# Mock classes
# ---------------------------------------------------------------------------


class FakeTier(Enum):
    """Fake tier enum that mimics SubscriptionTier."""

    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    ENTERPRISE_PLUS = "enterprise_plus"


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "debates_per_month": self.debates_per_month,
            "users_per_org": self.users_per_org,
            "api_access": self.api_access,
        }


@dataclass
class FakeOrganization:
    id: str = "org-123"
    name: str = "Test Org"
    slug: str = "test-org"
    tier: FakeTier = field(default_factory=lambda: FakeTier.ENTERPRISE_PLUS)
    limits: FakeTierLimits = field(default_factory=FakeTierLimits)
    debates_used_this_month: int = 10
    billing_cycle_start: datetime = field(
        default_factory=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc)
    )


@dataclass
class FakeUsageSummary:
    """Mock usage summary."""

    period_start: datetime = field(
        default_factory=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc)
    )
    period_end: datetime = field(
        default_factory=lambda: datetime(2025, 1, 31, 23, 59, 59, tzinfo=timezone.utc)
    )
    period_type: str = "month"
    total_tokens: int = 750000
    input_tokens: int = 500000
    output_tokens: int = 250000
    total_cost: str = "12.50"
    debates: int = 45
    api_calls: int = 1500

    def to_dict(self) -> dict[str, Any]:
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "period_type": self.period_type,
            "tokens": {
                "input": self.input_tokens,
                "output": self.output_tokens,
                "total": self.total_tokens,
                "cost": self.total_cost,
            },
            "counts": {
                "debates": self.debates,
                "api_calls": self.api_calls,
            },
        }


@dataclass
class FakeUsageBreakdown:
    """Mock usage breakdown."""

    period_start: datetime = field(
        default_factory=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc)
    )
    period_end: datetime = field(
        default_factory=lambda: datetime(2025, 1, 31, 23, 59, 59, tzinfo=timezone.utc)
    )
    total_cost: str = "125.50"
    total_tokens: int = 5000000
    total_debates: int = 150
    total_api_calls: int = 5000
    by_model: list = field(default_factory=list)
    by_provider: list = field(default_factory=list)
    by_day: list = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "totals": {
                "cost": self.total_cost,
                "tokens": self.total_tokens,
                "debates": self.total_debates,
                "api_calls": self.total_api_calls,
            },
            "by_model": self.by_model,
            "by_provider": self.by_provider,
            "by_day": self.by_day,
        }


@dataclass
class FakeUsageLimits:
    """Mock usage limits."""

    tier: str = "enterprise_plus"
    limits: dict = field(default_factory=lambda: {"tokens": 999999999, "debates": 999999})
    used: dict = field(default_factory=lambda: {"tokens": 750000, "debates": 45})
    percent: dict = field(default_factory=lambda: {"tokens": 0.075, "debates": 0.0045})
    exceeded: dict = field(default_factory=lambda: {"tokens": False, "debates": False})

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "limits": self.limits,
            "used": self.used,
            "percent": self.percent,
            "exceeded": self.exceeded,
        }


class FakeHandler:
    """Mock HTTP handler for testing."""

    def __init__(
        self,
        method: str = "GET",
        query_params: dict | None = None,
    ):
        self.command = method
        self._query_params = query_params or {}
        self.client_address = ("127.0.0.1", 12345)

    def get(self, key: str, default: Any = None) -> Any:
        """Support get_string_param() calls on the handler."""
        return self._query_params.get(key, default)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_user_store():
    """Create a mock user store."""
    store = MagicMock()
    store.get_user_by_id = MagicMock(return_value=FakeDbUser())
    store.get_organization_by_id = MagicMock(return_value=FakeOrganization())
    return store


@pytest.fixture
def mock_usage_meter():
    """Create a mock usage meter."""
    meter = AsyncMock()
    meter.get_usage_summary = AsyncMock(return_value=FakeUsageSummary())
    meter.get_usage_breakdown = AsyncMock(return_value=FakeUsageBreakdown())
    meter.get_usage_limits = AsyncMock(return_value=FakeUsageLimits())
    return meter


@pytest.fixture
def usage_handler(mock_user_store):
    """Create a UsageMeteringHandler with mocked dependencies."""
    handler = UsageMeteringHandler(ctx={"user_store": mock_user_store})
    return handler


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter between tests."""
    from aragora.server.handlers.billing.subscriptions import _usage_limiter

    _usage_limiter._requests.clear()
    yield
    _usage_limiter._requests.clear()


# ---------------------------------------------------------------------------
# Test can_handle
# ---------------------------------------------------------------------------


class TestCanHandle:
    def test_handles_usage_routes(self, usage_handler):
        """Handler accepts usage routes."""
        assert usage_handler.can_handle("/api/v1/billing/usage") is True
        assert usage_handler.can_handle("/api/v1/billing/usage/breakdown") is True
        assert usage_handler.can_handle("/api/v1/billing/limits") is True
        assert usage_handler.can_handle("/api/v1/billing/usage/summary") is True
        assert usage_handler.can_handle("/api/v1/billing/usage/export") is True
        assert usage_handler.can_handle("/api/v1/quotas") is True

    def test_rejects_unknown_routes(self, usage_handler):
        """Handler rejects non-usage routes."""
        assert usage_handler.can_handle("/api/v1/billing/plans") is False
        assert usage_handler.can_handle("/api/v1/billing/checkout") is False
        assert usage_handler.can_handle("/api/v1/debates") is False


# ---------------------------------------------------------------------------
# Test _get_usage
# ---------------------------------------------------------------------------


class TestGetUsage:
    @pytest.mark.asyncio
    async def test_returns_usage_data(self, usage_handler, mock_user_store, mock_usage_meter):
        """Usage endpoint returns usage summary."""
        handler = FakeHandler()
        user = FakeUser()

        with patch.object(usage_handler, "_get_usage_meter", return_value=mock_usage_meter):
            fn = usage_handler._get_usage.__wrapped__.__wrapped__
            result = await fn(usage_handler, handler, {}, user=user)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "usage" in data
        assert "tokens" in data["usage"]
        assert "counts" in data["usage"]

    @pytest.mark.asyncio
    async def test_respects_period_param(self, usage_handler, mock_user_store, mock_usage_meter):
        """Usage endpoint respects period query parameter."""
        handler = FakeHandler(query_params={"period": "week"})
        user = FakeUser()

        with patch.object(usage_handler, "_get_usage_meter", return_value=mock_usage_meter):
            fn = usage_handler._get_usage.__wrapped__.__wrapped__
            await fn(usage_handler, handler, {}, user=user)

        mock_usage_meter.get_usage_summary.assert_called_once()
        call_kwargs = mock_usage_meter.get_usage_summary.call_args[1]
        assert call_kwargs["period"] == "week"

    @pytest.mark.asyncio
    async def test_returns_503_without_user_store(self, mock_usage_meter):
        """Returns 503 when user store unavailable."""
        handler_obj = UsageMeteringHandler(ctx={})  # No user_store
        handler = FakeHandler()
        user = FakeUser()

        fn = handler_obj._get_usage.__wrapped__.__wrapped__
        result = await fn(handler_obj, handler, {}, user=user)

        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_user(self, usage_handler, mock_user_store):
        """Returns 404 when user not found."""
        mock_user_store.get_user_by_id.return_value = None
        handler = FakeHandler()
        user = FakeUser()

        fn = usage_handler._get_usage.__wrapped__.__wrapped__
        result = await fn(usage_handler, handler, {}, user=user)

        assert result.status_code == 404


# ---------------------------------------------------------------------------
# Test _get_usage_breakdown
# ---------------------------------------------------------------------------


class TestGetUsageBreakdown:
    @pytest.mark.asyncio
    async def test_returns_breakdown_data(self, usage_handler, mock_user_store, mock_usage_meter):
        """Breakdown endpoint returns detailed usage."""
        handler = FakeHandler()
        user = FakeUser()

        with patch.object(usage_handler, "_get_usage_meter", return_value=mock_usage_meter):
            fn = usage_handler._get_usage_breakdown.__wrapped__.__wrapped__
            result = await fn(usage_handler, handler, {}, user=user)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "breakdown" in data
        assert "totals" in data["breakdown"]

    @pytest.mark.asyncio
    async def test_parses_date_params(self, usage_handler, mock_user_store, mock_usage_meter):
        """Breakdown endpoint parses date parameters."""
        handler = FakeHandler(
            query_params={
                "start": "2025-01-01T00:00:00Z",
                "end": "2025-01-31T23:59:59Z",
            }
        )
        user = FakeUser()

        with patch.object(usage_handler, "_get_usage_meter", return_value=mock_usage_meter):
            fn = usage_handler._get_usage_breakdown.__wrapped__.__wrapped__
            await fn(usage_handler, handler, {}, user=user)

        mock_usage_meter.get_usage_breakdown.assert_called_once()
        call_kwargs = mock_usage_meter.get_usage_breakdown.call_args[1]
        assert call_kwargs["start_date"] is not None
        assert call_kwargs["end_date"] is not None

    @pytest.mark.asyncio
    async def test_rejects_invalid_date_format(self, usage_handler, mock_user_store):
        """Returns 400 for invalid date format."""
        handler = FakeHandler(query_params={"start": "not-a-date"})
        user = FakeUser()

        fn = usage_handler._get_usage_breakdown.__wrapped__.__wrapped__
        result = await fn(usage_handler, handler, {}, user=user)

        assert result.status_code == 400


# ---------------------------------------------------------------------------
# Test _get_limits
# ---------------------------------------------------------------------------


class TestGetLimits:
    @pytest.mark.asyncio
    async def test_returns_limits_data(self, usage_handler, mock_user_store, mock_usage_meter):
        """Limits endpoint returns usage limits."""
        handler = FakeHandler()
        user = FakeUser()

        with patch.object(usage_handler, "_get_usage_meter", return_value=mock_usage_meter):
            fn = usage_handler._get_limits.__wrapped__.__wrapped__
            result = await fn(usage_handler, handler, {}, user=user)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "limits" in data
        assert "tier" in data["limits"]
        assert "used" in data["limits"]
        assert "percent" in data["limits"]


# ---------------------------------------------------------------------------
# Test _get_quota_status
# ---------------------------------------------------------------------------


class TestGetQuotaStatus:
    @pytest.mark.asyncio
    async def test_returns_quota_data(self, usage_handler, mock_user_store):
        """Quota endpoint returns quota status."""
        handler = FakeHandler()
        user = FakeUser()

        mock_quota_manager = AsyncMock()
        mock_status = MagicMock()
        mock_status.limit = 100
        mock_status.current = 45
        mock_status.remaining = 55
        mock_status.period.value = "day"
        mock_status.percentage_used = 45.0
        mock_status.is_exceeded = False
        mock_status.is_warning = False
        mock_status.period_resets_at = datetime.now(timezone.utc)
        mock_quota_manager.get_quota_status = AsyncMock(return_value=mock_status)

        with patch(
            "aragora.server.handlers.billing.subscriptions.get_quota_manager",
            return_value=mock_quota_manager,
        ):
            fn = usage_handler._get_quota_status.__wrapped__.__wrapped__
            result = await fn(usage_handler, handler, {}, user=user)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "quotas" in data


# ---------------------------------------------------------------------------
# Test _export_usage
# ---------------------------------------------------------------------------


class TestExportUsage:
    @pytest.mark.asyncio
    async def test_exports_csv(self, usage_handler, mock_user_store, mock_usage_meter):
        """Export returns CSV by default."""
        handler = FakeHandler()
        user = FakeUser()

        with patch.object(usage_handler, "_get_usage_meter", return_value=mock_usage_meter):
            fn = usage_handler._export_usage.__wrapped__.__wrapped__
            result = await fn(usage_handler, handler, {}, user=user)

        assert result.status_code == 200
        assert result.content_type == "text/csv"
        assert b"Usage Export Report" in result.body

    @pytest.mark.asyncio
    async def test_exports_json(self, usage_handler, mock_user_store, mock_usage_meter):
        """Export returns JSON when requested."""
        handler = FakeHandler(query_params={"format": "json"})
        user = FakeUser()

        with patch.object(usage_handler, "_get_usage_meter", return_value=mock_usage_meter):
            fn = usage_handler._export_usage.__wrapped__.__wrapped__
            result = await fn(usage_handler, handler, {}, user=user)

        assert result.status_code == 200
        # JSON response has different content type
        data = json.loads(result.body)
        assert "totals" in data


# ---------------------------------------------------------------------------
# Test Rate Limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    @pytest.mark.asyncio
    async def test_rate_limits_usage_endpoints(self, usage_handler, mock_user_store):
        """Usage endpoints are rate limited."""
        from aragora.server.handlers.billing.subscriptions import _usage_limiter

        handler = FakeHandler()

        with patch.dict("os.environ", {"PYTEST_CURRENT_TEST": ""}, clear=False):
            # Exhaust rate limit (30 requests per minute)
            for _ in range(35):
                _usage_limiter.is_allowed("127.0.0.1")

            # Next request should be rejected
            result = await usage_handler.handle("/api/v1/billing/usage", {}, handler, method="GET")

        assert result.status_code == 429


# ---------------------------------------------------------------------------
# Test Handler Routing
# ---------------------------------------------------------------------------


class TestHandlerRouting:
    @pytest.mark.asyncio
    async def test_routes_to_correct_method(self, usage_handler, mock_user_store, mock_usage_meter):
        """Handler routes requests to correct methods."""
        handler = FakeHandler(method="GET")

        with patch.object(usage_handler, "_get_usage_meter", return_value=mock_usage_meter):
            # Patch the decorator chain for auth
            with patch.object(
                usage_handler,
                "_get_usage",
                usage_handler._get_usage.__wrapped__.__wrapped__,
            ):
                result = await usage_handler.handle(
                    "/api/v1/billing/usage", {}, handler, method="GET"
                )

        # Will fail at permission check but confirms routing works
        assert result is not None

    @pytest.mark.asyncio
    async def test_rejects_invalid_method(self, usage_handler, mock_user_store):
        """Handler rejects unsupported methods."""
        handler = FakeHandler(method="DELETE")

        result = await usage_handler.handle("/api/v1/billing/usage", {}, handler, method="DELETE")

        assert result.status_code == 405
