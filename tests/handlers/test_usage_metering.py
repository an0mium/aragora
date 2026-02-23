"""Tests for usage metering handler endpoints.

Tests the UsageMeteringHandler covering:
- GET /api/v1/billing/usage - Current usage summary
- GET /api/v1/billing/usage/summary - Alias for usage summary
- GET /api/v1/billing/usage/breakdown - Detailed breakdown
- GET /api/v1/billing/usage/export - CSV/JSON export
- GET /api/v1/billing/limits - Usage limits and percentages
- GET /api/v1/quotas - Quota status
- Rate limiting
- Error handling, edge cases, input validation
"""

import json
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.usage_metering import UsageMeteringHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_body(result) -> dict:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


def body_text(result) -> str:
    """Decode body as UTF-8 text."""
    return result.body.decode("utf-8")


# ---------------------------------------------------------------------------
# Mock domain objects
# ---------------------------------------------------------------------------

from aragora.billing.models import SubscriptionTier


class MockQuotaPeriod(Enum):
    DAY = "day"
    MONTH = "month"


class MockUser:
    """Mock user (returned by user_store.get_user_by_id)."""

    def __init__(self, user_id: str = "u-1", org_id: str | None = "org-1"):
        self.user_id = user_id
        self.id = user_id
        self.org_id = org_id


class MockOrganization:
    """Mock organization (returned by user_store.get_organization_by_id)."""

    def __init__(
        self,
        org_id: str = "org-1",
        name: str = "TestOrg",
        slug: str = "test-org",
        tier: Any = None,
    ):
        self.id = org_id
        self.name = name
        self.slug = slug
        self.tier = tier if tier is not None else SubscriptionTier.ENTERPRISE_PLUS


class MockUsageSummary:
    """Mock return from meter.get_usage_summary."""

    def __init__(self, data: dict | None = None):
        self._data = data or {
            "period_start": "2025-01-01T00:00:00Z",
            "period_end": "2025-01-31T23:59:59Z",
            "period_type": "month",
            "tokens": {"input": 500000, "output": 250000, "total": 750000, "cost": "12.50"},
            "counts": {"debates": 45, "api_calls": 1500},
        }

    def to_dict(self) -> dict:
        return self._data


class MockUsageBreakdown:
    """Mock return from meter.get_usage_breakdown."""

    def __init__(self, data: dict | None = None):
        self._data = data or {
            "totals": {"cost": "125.50", "tokens": 5000000, "debates": 150, "api_calls": 5000},
            "by_model": [],
            "by_provider": [],
            "by_day": [],
        }
        self.period_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        self.period_end = datetime(2025, 1, 31, 23, 59, 59, tzinfo=timezone.utc)
        self.total_cost = "125.50"
        self.total_tokens = 5000000
        self.total_debates = 150
        self.total_api_calls = 5000
        self.by_model = self._data.get("by_model", [])
        self.by_provider = self._data.get("by_provider", [])
        self.by_day = self._data.get("by_day", [])

    def to_dict(self) -> dict:
        return self._data


class MockUsageLimits:
    """Mock return from meter.get_usage_limits."""

    def __init__(self, data: dict | None = None):
        self._data = data or {
            "tier": "enterprise_plus",
            "limits": {"tokens": 999999999, "debates": 999999, "api_calls": 999999},
            "used": {"tokens": 750000, "debates": 45, "api_calls": 1500},
            "percent": {"tokens": 0.075, "debates": 0.0045, "api_calls": 0.15},
            "exceeded": {"tokens": False, "debates": False, "api_calls": False},
        }

    def to_dict(self) -> dict:
        return self._data


class MockQuotaStatus:
    """Mock return from quota_manager.get_quota_status."""

    def __init__(
        self,
        limit: int = 100,
        current: int = 45,
        remaining: int = 55,
        period: MockQuotaPeriod = MockQuotaPeriod.DAY,
        percentage_used: float = 45.0,
        is_exceeded: bool = False,
        is_warning: bool = False,
        period_resets_at: datetime | None = None,
    ):
        self.limit = limit
        self.current = current
        self.remaining = remaining
        self.period = period
        self.percentage_used = percentage_used
        self.is_exceeded = is_exceeded
        self.is_warning = is_warning
        self.period_resets_at = period_resets_at


# ---------------------------------------------------------------------------
# Mock handler (simulates HTTP handler with query_params and client_address)
# ---------------------------------------------------------------------------

class FakeHTTPHandler:
    """Mimics what the server passes as 'handler'."""

    def __init__(self, query_params: dict | None = None, method: str = "GET"):
        self.query_params = query_params or {}
        self.command = method
        self.client_address = ("127.0.0.1", 54321)
        self.headers = {"Content-Length": "0"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

QUOTA_MANAGER_PATCH = "aragora.server.middleware.tier_enforcement.get_quota_manager"


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the module-level rate limiter before each test."""
    from aragora.server.handlers.usage_metering import _usage_limiter

    _usage_limiter._requests.clear()
    yield
    _usage_limiter._requests.clear()


@pytest.fixture(autouse=True)
def _reset_quota_manager_singleton():
    """Reset the cached QuotaManager singleton so patches take effect."""
    import aragora.server.middleware.tier_enforcement as te

    original = te._quota_manager
    te._quota_manager = None
    yield
    te._quota_manager = original


@pytest.fixture
def mock_user_store():
    store = MagicMock()
    store.get_user_by_id.return_value = MockUser()
    store.get_organization_by_id.return_value = MockOrganization()
    return store


@pytest.fixture
def mock_meter():
    meter = AsyncMock()
    meter.get_usage_summary.return_value = MockUsageSummary()
    meter.get_usage_breakdown.return_value = MockUsageBreakdown()
    meter.get_usage_limits.return_value = MockUsageLimits()
    return meter


@pytest.fixture
def handler(mock_user_store, mock_meter):
    """Create a UsageMeteringHandler with mocked dependencies."""
    h = UsageMeteringHandler(ctx={"user_store": mock_user_store})
    h._get_usage_meter = MagicMock(return_value=mock_meter)
    return h


@pytest.fixture
def http_handler():
    """Create a default FakeHTTPHandler."""
    return FakeHTTPHandler()


# ===========================================================================
# can_handle
# ===========================================================================


class TestCanHandle:
    """Tests for route matching."""

    def test_usage_route(self, handler):
        assert handler.can_handle("/api/v1/billing/usage") is True

    def test_usage_summary_route(self, handler):
        assert handler.can_handle("/api/v1/billing/usage/summary") is True

    def test_usage_breakdown_route(self, handler):
        assert handler.can_handle("/api/v1/billing/usage/breakdown") is True

    def test_usage_export_route(self, handler):
        assert handler.can_handle("/api/v1/billing/usage/export") is True

    def test_limits_route(self, handler):
        assert handler.can_handle("/api/v1/billing/limits") is True

    def test_quotas_route(self, handler):
        assert handler.can_handle("/api/v1/quotas") is True

    def test_unknown_route(self, handler):
        assert handler.can_handle("/api/v1/billing/unknown") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_partial_match(self, handler):
        assert handler.can_handle("/api/v1/billing") is False


# ===========================================================================
# GET /api/v1/billing/usage  (and /usage/summary alias)
# ===========================================================================


class TestGetUsage:
    """Tests for _get_usage endpoint."""

    @pytest.mark.asyncio
    async def test_success(self, handler, http_handler):
        result = await handler.handle("/api/v1/billing/usage", {}, http_handler)
        assert result.status_code == 200
        data = parse_body(result)
        assert "usage" in data

    @pytest.mark.asyncio
    async def test_summary_alias(self, handler, http_handler):
        result = await handler.handle("/api/v1/billing/usage/summary", {}, http_handler)
        assert result.status_code == 200
        data = parse_body(result)
        assert "usage" in data

    @pytest.mark.asyncio
    async def test_usage_data_content(self, handler, http_handler):
        result = await handler.handle("/api/v1/billing/usage", {}, http_handler)
        data = parse_body(result)
        usage = data["usage"]
        assert "period_type" in usage
        assert usage["period_type"] == "month"

    @pytest.mark.asyncio
    async def test_period_param_passed(self, handler, mock_meter):
        http = FakeHTTPHandler(query_params={"period": "week"})
        await handler.handle("/api/v1/billing/usage", {}, http)
        mock_meter.get_usage_summary.assert_called_once()
        call_kwargs = mock_meter.get_usage_summary.call_args
        assert call_kwargs.kwargs["period"] == "week"

    @pytest.mark.asyncio
    async def test_default_period_month(self, handler, mock_meter):
        http = FakeHTTPHandler()
        await handler.handle("/api/v1/billing/usage", {}, http)
        call_kwargs = mock_meter.get_usage_summary.call_args
        assert call_kwargs.kwargs["period"] == "month"

    @pytest.mark.asyncio
    async def test_tier_passed(self, handler, mock_meter):
        http = FakeHTTPHandler()
        await handler.handle("/api/v1/billing/usage", {}, http)
        call_kwargs = mock_meter.get_usage_summary.call_args
        assert call_kwargs.kwargs["tier"] == "enterprise_plus"

    @pytest.mark.asyncio
    async def test_no_user_store_503(self, mock_meter):
        h = UsageMeteringHandler(ctx={})
        h._get_usage_meter = MagicMock(return_value=mock_meter)
        http = FakeHTTPHandler()
        result = await h.handle("/api/v1/billing/usage", {}, http)
        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_user_not_found_404(self, handler, mock_user_store):
        mock_user_store.get_user_by_id.return_value = None
        http = FakeHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage", {}, http)
        assert result.status_code == 404
        assert "User not found" in body_text(result)

    @pytest.mark.asyncio
    async def test_no_org_id_404(self, handler, mock_user_store):
        mock_user_store.get_user_by_id.return_value = MockUser(org_id=None)
        http = FakeHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage", {}, http)
        assert result.status_code == 404
        assert "No organization found" in body_text(result)

    @pytest.mark.asyncio
    async def test_org_not_found_404(self, handler, mock_user_store):
        mock_user_store.get_organization_by_id.return_value = None
        http = FakeHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage", {}, http)
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_content_type_json(self, handler, http_handler):
        result = await handler.handle("/api/v1/billing/usage", {}, http_handler)
        assert result.content_type == "application/json"

    @pytest.mark.asyncio
    async def test_hour_period(self, handler, mock_meter):
        http = FakeHTTPHandler(query_params={"period": "hour"})
        await handler.handle("/api/v1/billing/usage", {}, http)
        assert mock_meter.get_usage_summary.call_args.kwargs["period"] == "hour"

    @pytest.mark.asyncio
    async def test_quarter_period(self, handler, mock_meter):
        http = FakeHTTPHandler(query_params={"period": "quarter"})
        await handler.handle("/api/v1/billing/usage", {}, http)
        assert mock_meter.get_usage_summary.call_args.kwargs["period"] == "quarter"

    @pytest.mark.asyncio
    async def test_year_period(self, handler, mock_meter):
        http = FakeHTTPHandler(query_params={"period": "year"})
        await handler.handle("/api/v1/billing/usage", {}, http)
        assert mock_meter.get_usage_summary.call_args.kwargs["period"] == "year"


# ===========================================================================
# GET /api/v1/billing/usage/breakdown
# ===========================================================================


class TestGetUsageBreakdown:
    """Tests for _get_usage_breakdown endpoint."""

    @pytest.mark.asyncio
    async def test_success(self, handler, http_handler):
        result = await handler.handle("/api/v1/billing/usage/breakdown", {}, http_handler)
        assert result.status_code == 200
        data = parse_body(result)
        assert "breakdown" in data

    @pytest.mark.asyncio
    async def test_breakdown_content(self, handler, http_handler):
        result = await handler.handle("/api/v1/billing/usage/breakdown", {}, http_handler)
        data = parse_body(result)
        bd = data["breakdown"]
        assert "totals" in bd

    @pytest.mark.asyncio
    async def test_with_start_date(self, handler, mock_meter):
        http = FakeHTTPHandler(query_params={"start": "2025-01-01T00:00:00Z"})
        await handler.handle("/api/v1/billing/usage/breakdown", {}, http)
        call_kwargs = mock_meter.get_usage_breakdown.call_args
        assert call_kwargs.kwargs["start_date"] is not None

    @pytest.mark.asyncio
    async def test_with_end_date(self, handler, mock_meter):
        http = FakeHTTPHandler(query_params={"end": "2025-01-31T23:59:59Z"})
        await handler.handle("/api/v1/billing/usage/breakdown", {}, http)
        call_kwargs = mock_meter.get_usage_breakdown.call_args
        assert call_kwargs.kwargs["end_date"] is not None

    @pytest.mark.asyncio
    async def test_with_both_dates(self, handler, mock_meter):
        http = FakeHTTPHandler(query_params={
            "start": "2025-01-01T00:00:00+00:00",
            "end": "2025-01-31T23:59:59+00:00",
        })
        await handler.handle("/api/v1/billing/usage/breakdown", {}, http)
        call_kwargs = mock_meter.get_usage_breakdown.call_args
        assert call_kwargs.kwargs["start_date"] is not None
        assert call_kwargs.kwargs["end_date"] is not None

    @pytest.mark.asyncio
    async def test_no_dates_passes_none(self, handler, mock_meter):
        http = FakeHTTPHandler()
        await handler.handle("/api/v1/billing/usage/breakdown", {}, http)
        call_kwargs = mock_meter.get_usage_breakdown.call_args
        assert call_kwargs.kwargs["start_date"] is None
        assert call_kwargs.kwargs["end_date"] is None

    @pytest.mark.asyncio
    async def test_invalid_start_date_400(self, handler):
        http = FakeHTTPHandler(query_params={"start": "not-a-date"})
        result = await handler.handle("/api/v1/billing/usage/breakdown", {}, http)
        assert result.status_code == 400
        assert "Invalid start date" in body_text(result)

    @pytest.mark.asyncio
    async def test_invalid_end_date_400(self, handler):
        http = FakeHTTPHandler(query_params={"end": "not-a-date"})
        result = await handler.handle("/api/v1/billing/usage/breakdown", {}, http)
        assert result.status_code == 400
        assert "Invalid end date" in body_text(result)

    @pytest.mark.asyncio
    async def test_invalid_start_valid_end_400(self, handler):
        http = FakeHTTPHandler(query_params={
            "start": "garbage",
            "end": "2025-01-31T00:00:00Z",
        })
        result = await handler.handle("/api/v1/billing/usage/breakdown", {}, http)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_valid_start_invalid_end_400(self, handler):
        http = FakeHTTPHandler(query_params={
            "start": "2025-01-01T00:00:00Z",
            "end": "garbage",
        })
        result = await handler.handle("/api/v1/billing/usage/breakdown", {}, http)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_no_user_store_503(self, mock_meter):
        h = UsageMeteringHandler(ctx={})
        h._get_usage_meter = MagicMock(return_value=mock_meter)
        http = FakeHTTPHandler()
        result = await h.handle("/api/v1/billing/usage/breakdown", {}, http)
        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_user_not_found_404(self, handler, mock_user_store):
        mock_user_store.get_user_by_id.return_value = None
        http = FakeHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage/breakdown", {}, http)
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_no_org_404(self, handler, mock_user_store):
        mock_user_store.get_user_by_id.return_value = MockUser(org_id=None)
        http = FakeHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage/breakdown", {}, http)
        assert result.status_code == 404


# ===========================================================================
# GET /api/v1/billing/limits
# ===========================================================================


class TestGetLimits:
    """Tests for _get_limits endpoint."""

    @pytest.mark.asyncio
    async def test_success(self, handler, http_handler):
        result = await handler.handle("/api/v1/billing/limits", {}, http_handler)
        assert result.status_code == 200
        data = parse_body(result)
        assert "limits" in data

    @pytest.mark.asyncio
    async def test_limits_content(self, handler, http_handler):
        result = await handler.handle("/api/v1/billing/limits", {}, http_handler)
        data = parse_body(result)
        limits = data["limits"]
        assert limits["tier"] == "enterprise_plus"

    @pytest.mark.asyncio
    async def test_tier_passed_to_meter(self, handler, mock_meter):
        http = FakeHTTPHandler()
        await handler.handle("/api/v1/billing/limits", {}, http)
        call_kwargs = mock_meter.get_usage_limits.call_args
        assert call_kwargs.kwargs["tier"] == "enterprise_plus"

    @pytest.mark.asyncio
    async def test_org_id_passed_to_meter(self, handler, mock_meter):
        http = FakeHTTPHandler()
        await handler.handle("/api/v1/billing/limits", {}, http)
        call_kwargs = mock_meter.get_usage_limits.call_args
        assert call_kwargs.kwargs["org_id"] == "org-1"

    @pytest.mark.asyncio
    async def test_no_user_store_503(self, mock_meter):
        h = UsageMeteringHandler(ctx={})
        h._get_usage_meter = MagicMock(return_value=mock_meter)
        http = FakeHTTPHandler()
        result = await h.handle("/api/v1/billing/limits", {}, http)
        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_user_not_found_404(self, handler, mock_user_store):
        mock_user_store.get_user_by_id.return_value = None
        http = FakeHTTPHandler()
        result = await handler.handle("/api/v1/billing/limits", {}, http)
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_org_not_found_404(self, handler, mock_user_store):
        mock_user_store.get_organization_by_id.return_value = None
        http = FakeHTTPHandler()
        result = await handler.handle("/api/v1/billing/limits", {}, http)
        assert result.status_code == 404


# ===========================================================================
# GET /api/v1/quotas
# ===========================================================================


class TestGetQuotaStatus:
    """Tests for _get_quota_status endpoint."""

    @pytest.mark.asyncio
    async def test_success(self, handler, http_handler):
        mock_manager = AsyncMock()
        mock_manager.get_quota_status.return_value = MockQuotaStatus()
        with patch(
            QUOTA_MANAGER_PATCH,
            return_value=mock_manager,
        ):
            result = await handler.handle("/api/v1/quotas", {}, http_handler)
        assert result.status_code == 200
        data = parse_body(result)
        assert "quotas" in data

    @pytest.mark.asyncio
    async def test_all_resources_returned(self, handler, http_handler):
        mock_manager = AsyncMock()
        mock_manager.get_quota_status.return_value = MockQuotaStatus()
        with patch(
            QUOTA_MANAGER_PATCH,
            return_value=mock_manager,
        ):
            result = await handler.handle("/api/v1/quotas", {}, http_handler)
        data = parse_body(result)
        quotas = data["quotas"]
        expected_resources = ["debates", "api_requests", "tokens", "storage_bytes", "knowledge_bytes"]
        for resource in expected_resources:
            assert resource in quotas

    @pytest.mark.asyncio
    async def test_quota_fields(self, handler, http_handler):
        mock_manager = AsyncMock()
        mock_manager.get_quota_status.return_value = MockQuotaStatus(
            limit=100, current=45, remaining=55,
            percentage_used=45.0, is_exceeded=False, is_warning=False,
        )
        with patch(
            QUOTA_MANAGER_PATCH,
            return_value=mock_manager,
        ):
            result = await handler.handle("/api/v1/quotas", {}, http_handler)
        data = parse_body(result)
        q = data["quotas"]["debates"]
        assert q["limit"] == 100
        assert q["current"] == 45
        assert q["remaining"] == 55
        assert q["percentage_used"] == 45.0
        assert q["is_exceeded"] is False
        assert q["is_warning"] is False

    @pytest.mark.asyncio
    async def test_quota_with_resets_at(self, handler, http_handler):
        reset_time = datetime(2025, 2, 1, tzinfo=timezone.utc)
        mock_manager = AsyncMock()
        mock_manager.get_quota_status.return_value = MockQuotaStatus(
            period_resets_at=reset_time,
        )
        with patch(
            QUOTA_MANAGER_PATCH,
            return_value=mock_manager,
        ):
            result = await handler.handle("/api/v1/quotas", {}, http_handler)
        data = parse_body(result)
        assert data["quotas"]["debates"]["resets_at"] == reset_time.isoformat()

    @pytest.mark.asyncio
    async def test_quota_resets_at_none(self, handler, http_handler):
        mock_manager = AsyncMock()
        mock_manager.get_quota_status.return_value = MockQuotaStatus(period_resets_at=None)
        with patch(
            QUOTA_MANAGER_PATCH,
            return_value=mock_manager,
        ):
            result = await handler.handle("/api/v1/quotas", {}, http_handler)
        data = parse_body(result)
        assert data["quotas"]["debates"]["resets_at"] is None

    @pytest.mark.asyncio
    async def test_quota_exceeded_flag(self, handler, http_handler):
        mock_manager = AsyncMock()
        mock_manager.get_quota_status.return_value = MockQuotaStatus(
            is_exceeded=True, is_warning=True,
        )
        with patch(
            QUOTA_MANAGER_PATCH,
            return_value=mock_manager,
        ):
            result = await handler.handle("/api/v1/quotas", {}, http_handler)
        data = parse_body(result)
        q = data["quotas"]["debates"]
        assert q["is_exceeded"] is True
        assert q["is_warning"] is True

    @pytest.mark.asyncio
    async def test_quota_resource_error_skipped(self, handler, http_handler):
        """When get_quota_status raises for a resource, it is skipped."""
        call_count = 0

        async def failing_status(resource, tenant_id=None):
            nonlocal call_count
            call_count += 1
            if resource == "debates":
                raise ValueError("Quota not configured")
            return MockQuotaStatus()

        mock_manager = AsyncMock()
        mock_manager.get_quota_status = failing_status
        with patch(
            QUOTA_MANAGER_PATCH,
            return_value=mock_manager,
        ):
            result = await handler.handle("/api/v1/quotas", {}, http_handler)
        assert result.status_code == 200
        data = parse_body(result)
        assert "debates" not in data["quotas"]
        # Other resources should still be present
        assert "api_requests" in data["quotas"]

    @pytest.mark.asyncio
    async def test_quota_returns_none_skipped(self, handler, http_handler):
        """When get_quota_status returns None for a resource, it is skipped."""
        async def sometimes_none(resource, tenant_id=None):
            if resource == "storage_bytes":
                return None
            return MockQuotaStatus()

        mock_manager = AsyncMock()
        mock_manager.get_quota_status = sometimes_none
        with patch(
            QUOTA_MANAGER_PATCH,
            return_value=mock_manager,
        ):
            result = await handler.handle("/api/v1/quotas", {}, http_handler)
        data = parse_body(result)
        assert "storage_bytes" not in data["quotas"]
        assert "debates" in data["quotas"]

    @pytest.mark.asyncio
    async def test_quota_all_errors_empty(self, handler, http_handler):
        """When all resources fail, quotas is empty dict."""
        mock_manager = AsyncMock()
        mock_manager.get_quota_status.side_effect = RuntimeError("total failure")
        with patch(
            QUOTA_MANAGER_PATCH,
            return_value=mock_manager,
        ):
            result = await handler.handle("/api/v1/quotas", {}, http_handler)
        assert result.status_code == 200
        data = parse_body(result)
        assert data["quotas"] == {}

    @pytest.mark.asyncio
    async def test_quota_no_user_store_503(self, mock_meter):
        h = UsageMeteringHandler(ctx={})
        h._get_usage_meter = MagicMock(return_value=mock_meter)
        http = FakeHTTPHandler()
        mock_manager = AsyncMock()
        with patch(
            QUOTA_MANAGER_PATCH,
            return_value=mock_manager,
        ):
            result = await h.handle("/api/v1/quotas", {}, http)
        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_quota_user_not_found_404(self, handler, mock_user_store):
        mock_user_store.get_user_by_id.return_value = None
        http = FakeHTTPHandler()
        mock_manager = AsyncMock()
        with patch(
            QUOTA_MANAGER_PATCH,
            return_value=mock_manager,
        ):
            result = await handler.handle("/api/v1/quotas", {}, http)
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_quota_no_org_404(self, handler, mock_user_store):
        mock_user_store.get_user_by_id.return_value = MockUser(org_id=None)
        http = FakeHTTPHandler()
        mock_manager = AsyncMock()
        with patch(
            QUOTA_MANAGER_PATCH,
            return_value=mock_manager,
        ):
            result = await handler.handle("/api/v1/quotas", {}, http)
        assert result.status_code == 404


# ===========================================================================
# GET /api/v1/billing/usage/export
# ===========================================================================


class TestExportUsage:
    """Tests for _export_usage endpoint (CSV and JSON)."""

    @pytest.mark.asyncio
    async def test_csv_default(self, handler, http_handler):
        result = await handler.handle("/api/v1/billing/usage/export", {}, http_handler)
        assert result.status_code == 200
        assert result.content_type == "text/csv"

    @pytest.mark.asyncio
    async def test_csv_has_content_disposition(self, handler, http_handler):
        result = await handler.handle("/api/v1/billing/usage/export", {}, http_handler)
        assert "Content-Disposition" in result.headers
        assert "attachment" in result.headers["Content-Disposition"]
        assert "test-org" in result.headers["Content-Disposition"]
        assert ".csv" in result.headers["Content-Disposition"]

    @pytest.mark.asyncio
    async def test_csv_content_has_header_row(self, handler, http_handler):
        result = await handler.handle("/api/v1/billing/usage/export", {}, http_handler)
        text = body_text(result)
        assert "Usage Export Report" in text

    @pytest.mark.asyncio
    async def test_csv_content_has_organization(self, handler, http_handler):
        result = await handler.handle("/api/v1/billing/usage/export", {}, http_handler)
        text = body_text(result)
        assert "TestOrg" in text

    @pytest.mark.asyncio
    async def test_csv_content_has_summary(self, handler, http_handler):
        result = await handler.handle("/api/v1/billing/usage/export", {}, http_handler)
        text = body_text(result)
        assert "Summary" in text
        assert "125.50" in text  # total_cost
        assert "5000000" in text  # total_tokens

    @pytest.mark.asyncio
    async def test_csv_sections(self, handler, http_handler):
        result = await handler.handle("/api/v1/billing/usage/export", {}, http_handler)
        text = body_text(result)
        assert "Usage by Model" in text
        assert "Usage by Provider" in text
        assert "Daily Usage" in text

    @pytest.mark.asyncio
    async def test_csv_with_model_data(self, handler, mock_meter, http_handler):
        breakdown = MockUsageBreakdown()
        breakdown.by_model = [
            {"model": "claude-3", "input_tokens": 100, "output_tokens": 50, "total_tokens": 150, "cost": "1.00", "requests": 10},
        ]
        mock_meter.get_usage_breakdown.return_value = breakdown
        result = await handler.handle("/api/v1/billing/usage/export", {}, http_handler)
        text = body_text(result)
        assert "claude-3" in text

    @pytest.mark.asyncio
    async def test_csv_with_provider_data(self, handler, mock_meter, http_handler):
        breakdown = MockUsageBreakdown()
        breakdown.by_provider = [
            {"provider": "anthropic", "total_tokens": 500, "cost": "5.00", "requests": 25},
        ]
        mock_meter.get_usage_breakdown.return_value = breakdown
        result = await handler.handle("/api/v1/billing/usage/export", {}, http_handler)
        text = body_text(result)
        assert "anthropic" in text

    @pytest.mark.asyncio
    async def test_csv_with_daily_data(self, handler, mock_meter, http_handler):
        breakdown = MockUsageBreakdown()
        breakdown.by_day = [
            {"day": "2025-01-15", "total_tokens": 1000, "cost": "2.00", "debates": 3, "api_calls": 50},
        ]
        mock_meter.get_usage_breakdown.return_value = breakdown
        result = await handler.handle("/api/v1/billing/usage/export", {}, http_handler)
        text = body_text(result)
        assert "2025-01-15" in text

    @pytest.mark.asyncio
    async def test_json_format(self, handler):
        http = FakeHTTPHandler(query_params={"format": "json"})
        result = await handler.handle("/api/v1/billing/usage/export", {}, http)
        assert result.status_code == 200
        assert result.content_type == "application/json"
        data = parse_body(result)
        assert "totals" in data

    @pytest.mark.asyncio
    async def test_explicit_csv_format(self, handler):
        http = FakeHTTPHandler(query_params={"format": "csv"})
        result = await handler.handle("/api/v1/billing/usage/export", {}, http)
        assert result.status_code == 200
        assert result.content_type == "text/csv"

    @pytest.mark.asyncio
    async def test_export_with_start_date(self, handler, mock_meter):
        http = FakeHTTPHandler(query_params={"start": "2025-01-01T00:00:00Z"})
        await handler.handle("/api/v1/billing/usage/export", {}, http)
        call_kwargs = mock_meter.get_usage_breakdown.call_args
        assert call_kwargs.kwargs["start_date"] is not None

    @pytest.mark.asyncio
    async def test_export_with_end_date(self, handler, mock_meter):
        http = FakeHTTPHandler(query_params={"end": "2025-01-31T00:00:00Z"})
        await handler.handle("/api/v1/billing/usage/export", {}, http)
        call_kwargs = mock_meter.get_usage_breakdown.call_args
        assert call_kwargs.kwargs["end_date"] is not None

    @pytest.mark.asyncio
    async def test_export_invalid_start_date_400(self, handler):
        http = FakeHTTPHandler(query_params={"start": "nope"})
        result = await handler.handle("/api/v1/billing/usage/export", {}, http)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_export_invalid_end_date_400(self, handler):
        http = FakeHTTPHandler(query_params={"end": "nope"})
        result = await handler.handle("/api/v1/billing/usage/export", {}, http)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_export_no_user_store_503(self, mock_meter):
        h = UsageMeteringHandler(ctx={})
        h._get_usage_meter = MagicMock(return_value=mock_meter)
        http = FakeHTTPHandler()
        result = await h.handle("/api/v1/billing/usage/export", {}, http)
        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_export_user_not_found_404(self, handler, mock_user_store):
        mock_user_store.get_user_by_id.return_value = None
        http = FakeHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage/export", {}, http)
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_export_no_org_404(self, handler, mock_user_store):
        mock_user_store.get_user_by_id.return_value = MockUser(org_id=None)
        http = FakeHTTPHandler()
        result = await handler.handle("/api/v1/billing/usage/export", {}, http)
        assert result.status_code == 404


# ===========================================================================
# Method not allowed (405)
# ===========================================================================


class TestMethodNotAllowed:
    """Tests for unsupported HTTP methods returning 405."""

    @pytest.mark.asyncio
    async def test_post_on_usage(self, handler):
        http = FakeHTTPHandler(method="POST")
        result = await handler.handle("/api/v1/billing/usage", {}, http, method="POST")
        assert result.status_code == 405

    @pytest.mark.asyncio
    async def test_put_on_usage(self, handler):
        http = FakeHTTPHandler(method="PUT")
        result = await handler.handle("/api/v1/billing/usage", {}, http, method="PUT")
        assert result.status_code == 405

    @pytest.mark.asyncio
    async def test_delete_on_limits(self, handler):
        http = FakeHTTPHandler(method="DELETE")
        result = await handler.handle("/api/v1/billing/limits", {}, http, method="DELETE")
        assert result.status_code == 405

    @pytest.mark.asyncio
    async def test_patch_on_breakdown(self, handler):
        http = FakeHTTPHandler(method="PATCH")
        result = await handler.handle("/api/v1/billing/usage/breakdown", {}, http, method="PATCH")
        assert result.status_code == 405

    @pytest.mark.asyncio
    async def test_post_on_quotas(self, handler):
        http = FakeHTTPHandler(method="POST")
        result = await handler.handle("/api/v1/quotas", {}, http, method="POST")
        assert result.status_code == 405

    @pytest.mark.asyncio
    async def test_post_on_export(self, handler):
        http = FakeHTTPHandler(method="POST")
        result = await handler.handle("/api/v1/billing/usage/export", {}, http, method="POST")
        assert result.status_code == 405


# ===========================================================================
# Rate limiting
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting on the handler."""

    @pytest.mark.asyncio
    async def test_rate_limit_eventually_triggers(self, handler):
        """Burst of requests eventually triggers rate limit (429)."""
        from aragora.server.handlers.usage_metering import _usage_limiter

        # The limiter allows 30 requests per minute. Exhaust them.
        http = FakeHTTPHandler()
        last_result = None
        for _ in range(35):
            last_result = await handler.handle("/api/v1/billing/usage", {}, http)
            if last_result.status_code == 429:
                break
        assert last_result.status_code == 429

    @pytest.mark.asyncio
    async def test_rate_limit_message(self, handler):
        http = FakeHTTPHandler()
        for _ in range(35):
            result = await handler.handle("/api/v1/billing/usage", {}, http)
            if result.status_code == 429:
                break
        assert "Rate limit" in body_text(result)


# ===========================================================================
# _get_org_tier helper
# ===========================================================================


class TestGetOrgTier:
    """Tests for the _get_org_tier helper method."""

    def test_none_org_returns_free(self, handler):
        assert handler._get_org_tier(None) == "free"

    def test_subscription_tier_enum(self, handler):
        from aragora.billing.models import SubscriptionTier

        org = MagicMock()
        org.tier = SubscriptionTier.ENTERPRISE
        result = handler._get_org_tier(org)
        assert result == SubscriptionTier.ENTERPRISE.value

    def test_string_tier(self, handler):
        org = MagicMock()
        org.tier = "professional"
        result = handler._get_org_tier(org)
        assert result == "professional"

    def test_none_tier_attr_returns_free(self, handler):
        org = MagicMock()
        org.tier = None
        result = handler._get_org_tier(org)
        assert result == "free"

    def test_real_subscription_tier_enum(self, handler):
        org = MockOrganization(tier=SubscriptionTier.STARTER)
        result = handler._get_org_tier(org)
        assert result == "starter"

    def test_enterprise_plus_tier(self, handler):
        org = MockOrganization(tier=SubscriptionTier.ENTERPRISE_PLUS)
        result = handler._get_org_tier(org)
        assert result == "enterprise_plus"


# ===========================================================================
# Constructor / context
# ===========================================================================


class TestConstructor:
    """Tests for handler initialization."""

    def test_default_ctx(self):
        h = UsageMeteringHandler()
        assert h.ctx == {}

    def test_ctx_passed(self):
        ctx = {"user_store": MagicMock()}
        h = UsageMeteringHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_resource_type(self):
        h = UsageMeteringHandler()
        assert h.RESOURCE_TYPE == "billing_usage"

    def test_routes_list(self):
        h = UsageMeteringHandler()
        assert len(h.ROUTES) == 6


# ===========================================================================
# _get_user_store
# ===========================================================================


class TestGetUserStore:
    """Tests for _get_user_store helper."""

    def test_returns_store_from_ctx(self, handler, mock_user_store):
        assert handler._get_user_store() is mock_user_store

    def test_returns_none_when_missing(self):
        h = UsageMeteringHandler(ctx={})
        assert h._get_user_store() is None

    def test_returns_none_when_ctx_empty(self):
        h = UsageMeteringHandler(ctx={"other": "stuff"})
        assert h._get_user_store() is None


# ===========================================================================
# Edge cases: handler.command attribute for method detection
# ===========================================================================


class TestMethodFromHandlerCommand:
    """Tests that handler.command overrides the method parameter."""

    @pytest.mark.asyncio
    async def test_command_override(self, handler):
        """If handler has .command attribute, it overrides method parameter."""
        http = FakeHTTPHandler(method="GET")
        # Pass method="POST" but handler.command is "GET"
        result = await handler.handle("/api/v1/billing/usage", {}, http, method="POST")
        # handler.command is "GET", so it should match the GET route
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_command_post_prevents_get(self, handler):
        """If handler.command is POST, GET routes should not match."""
        http = FakeHTTPHandler(method="POST")
        result = await handler.handle("/api/v1/billing/usage", {}, http, method="GET")
        # handler.command is "POST", overrides method="GET"
        assert result.status_code == 405
