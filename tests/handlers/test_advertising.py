"""Tests for advertising handler (aragora/server/handlers/features/advertising.py).

Covers all routes and behavior of the AdvertisingHandler class:
- can_handle() routing for all ROUTES
- GET    /api/v1/advertising/platforms            - List connected platforms
- POST   /api/v1/advertising/connect              - Connect a platform
- DELETE /api/v1/advertising/{platform}           - Disconnect platform
- GET    /api/v1/advertising/campaigns            - List all campaigns (cross-platform)
- GET    /api/v1/advertising/{platform}/campaigns - List platform campaigns
- POST   /api/v1/advertising/{platform}/campaigns - Create campaign
- PUT    /api/v1/advertising/{platform}/campaigns/{id} - Update campaign
- GET    /api/v1/advertising/{platform}/campaigns/{id} - Get campaign
- GET    /api/v1/advertising/performance          - Cross-platform performance
- GET    /api/v1/advertising/{platform}/performance - Platform performance
- POST   /api/v1/advertising/analyze              - Multi-agent performance analysis
- GET    /api/v1/advertising/budget-recommendations - Budget recommendations
- Error handling (not found, invalid data, missing params, bad JSON)
- Validation (budget, campaign name, dates, allowlists)
- Edge cases (empty state, disconnected platforms)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.advertising import (
    ALLOWED_ANALYSIS_TYPES,
    ALLOWED_BUDGET_OBJECTIVES,
    ALLOWED_CAMPAIGN_STATUSES,
    ALLOWED_CAMPAIGN_TYPES,
    ALLOWED_OBJECTIVES,
    MAX_ANALYSIS_DAYS,
    MAX_CAMPAIGN_NAME_LENGTH,
    MAX_DAILY_BUDGET,
    MAX_TOTAL_BUDGET,
    MIN_BUDGET,
    SUPPORTED_PLATFORMS,
    AdvertisingHandler,
    _platform_connectors,
    _platform_credentials,
    _validate_budget,
    _validate_campaign_name,
    _validate_date_iso,
    _validate_date_range,
    _validate_against_allowlist,
    get_advertising_circuit_breaker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: dict) -> dict:
    """Extract the body dict from a handler result."""
    return result.get("body", result)


def _status(result: dict) -> int:
    """Extract HTTP status code from a handler result."""
    return result.get("status_code", 200)


@dataclass
class MockRequest:
    """Mock async HTTP request for advertising handler tests."""

    method: str = "GET"
    path: str = "/"
    query: dict[str, str] = field(default_factory=dict)
    _body: dict[str, Any] | None = None
    content_length: int = 0

    def __post_init__(self):
        if self._body:
            self.content_length = len(json.dumps(self._body).encode())

    async def json(self) -> dict[str, Any]:
        return self._body or {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create an AdvertisingHandler with minimal server context."""
    return AdvertisingHandler({})


@pytest.fixture(autouse=True)
def _clear_global_state():
    """Clear global advertising state between tests."""
    _platform_credentials.clear()
    _platform_connectors.clear()
    yield
    _platform_credentials.clear()
    _platform_connectors.clear()


@pytest.fixture(autouse=True)
def _reset_circuit_breaker():
    """Reset the circuit breaker between tests."""
    cb = get_advertising_circuit_breaker()
    # Reset single-entity internal state
    cb._single_failures = 0
    cb._single_open_at = 0.0
    cb._single_successes = 0
    cb._single_half_open_calls = 0
    yield
    cb._single_failures = 0
    cb._single_open_at = 0.0
    cb._single_successes = 0
    cb._single_half_open_calls = 0


def _connect_platform(platform: str = "google_ads"):
    """Helper to mark a platform as connected in global state."""
    creds_map = {
        "google_ads": {
            "developer_token": "tok",
            "client_id": "cid",
            "client_secret": "csec",
            "refresh_token": "rtok",
            "customer_id": "cust",
        },
        "meta_ads": {"access_token": "tok", "ad_account_id": "acc"},
        "linkedin_ads": {"access_token": "tok", "ad_account_id": "acc"},
        "microsoft_ads": {
            "developer_token": "tok",
            "client_id": "cid",
            "client_secret": "csec",
            "refresh_token": "rtok",
            "account_id": "aid",
            "customer_id": "cust",
        },
    }
    _platform_credentials[platform] = {
        "credentials": creds_map.get(platform, {}),
        "connected_at": "2026-01-01T00:00:00+00:00",
    }


def _make_mock_connector():
    """Create a mock platform connector with standard methods."""
    connector = AsyncMock()
    connector.get_campaigns = AsyncMock(return_value=[])
    connector.get_campaign = AsyncMock()
    connector.create_campaign = AsyncMock(return_value="camp-123")
    connector.update_campaign_status = AsyncMock()
    connector.update_campaign = AsyncMock()
    connector.update_campaign_budget = AsyncMock()
    connector.get_campaign_performance = AsyncMock(return_value=[])
    connector.get_insights = AsyncMock(return_value=[])
    connector.get_account_analytics = AsyncMock()
    connector.close = AsyncMock()
    return connector


# ============================================================================
# can_handle routing
# ============================================================================


class TestCanHandle:
    """Verify that can_handle correctly accepts or rejects paths."""

    def test_platforms_path(self, handler):
        assert handler.can_handle("/api/v1/advertising/platforms")

    def test_connect_path(self, handler):
        assert handler.can_handle("/api/v1/advertising/connect")

    def test_disconnect_path(self, handler):
        assert handler.can_handle("/api/v1/advertising/google_ads")

    def test_campaigns_path(self, handler):
        assert handler.can_handle("/api/v1/advertising/campaigns")

    def test_platform_campaigns_path(self, handler):
        assert handler.can_handle("/api/v1/advertising/google_ads/campaigns")

    def test_campaign_id_path(self, handler):
        assert handler.can_handle("/api/v1/advertising/google_ads/campaigns/camp-123")

    def test_performance_path(self, handler):
        assert handler.can_handle("/api/v1/advertising/performance")

    def test_platform_performance_path(self, handler):
        assert handler.can_handle("/api/v1/advertising/meta_ads/performance")

    def test_analyze_path(self, handler):
        assert handler.can_handle("/api/v1/advertising/analyze")

    def test_budget_recommendations_path(self, handler):
        assert handler.can_handle("/api/v1/advertising/budget-recommendations")

    def test_rejects_unrelated_path(self, handler):
        assert not handler.can_handle("/api/v1/campaigns")

    def test_rejects_partial_prefix(self, handler):
        assert not handler.can_handle("/api/v1/advertising")

    def test_rejects_different_api_version(self, handler):
        assert not handler.can_handle("/api/v2/advertising/platforms")

    def test_rejects_empty_path(self, handler):
        assert not handler.can_handle("")

    def test_rejects_root(self, handler):
        assert not handler.can_handle("/")


# ============================================================================
# Initialization
# ============================================================================


class TestHandlerInit:
    """Test handler initialization."""

    def test_init_with_empty_context(self):
        h = AdvertisingHandler({})
        assert h.ctx == {}

    def test_init_with_server_context(self):
        ctx = {"some_key": "some_value"}
        h = AdvertisingHandler(server_context=ctx)
        assert h.ctx == ctx

    def test_init_with_ctx_kwarg(self):
        ctx = {"key": "val"}
        h = AdvertisingHandler(ctx=ctx)
        assert h.ctx == ctx

    def test_init_with_none_context(self):
        h = AdvertisingHandler(ctx=None, server_context=None)
        assert h.ctx == {}

    def test_resource_type(self, handler):
        assert handler.RESOURCE_TYPE == "advertising"

    def test_routes_defined(self, handler):
        assert len(handler.ROUTES) > 0
        assert "/api/v1/advertising/platforms" in handler.ROUTES


# ============================================================================
# Validation Helpers (unit tests)
# ============================================================================


class TestValidateBudget:
    """Test _validate_budget helper."""

    def test_none_value(self):
        val, err = _validate_budget(None, "budget", MAX_DAILY_BUDGET)
        assert val is None
        assert err is None

    def test_valid_float(self):
        val, err = _validate_budget(100.50, "budget", MAX_DAILY_BUDGET)
        assert val == 100.50
        assert err is None

    def test_valid_int(self):
        val, err = _validate_budget(100, "budget", MAX_DAILY_BUDGET)
        assert val == 100.0
        assert err is None

    def test_valid_string_number(self):
        val, err = _validate_budget("42.5", "budget", MAX_DAILY_BUDGET)
        assert val == 42.5
        assert err is None

    def test_negative_value(self):
        val, err = _validate_budget(-1, "budget", MAX_DAILY_BUDGET)
        assert val is None
        assert "non-negative" in err

    def test_exceeds_max(self):
        val, err = _validate_budget(MAX_DAILY_BUDGET + 1, "budget", MAX_DAILY_BUDGET)
        assert val is None
        assert "exceeds maximum" in err

    def test_non_numeric_string(self):
        val, err = _validate_budget("abc", "budget", MAX_DAILY_BUDGET)
        assert val is None
        assert "must be a number" in err

    def test_zero_budget(self):
        val, err = _validate_budget(0, "budget", MAX_DAILY_BUDGET)
        assert val == 0.0
        assert err is None

    def test_exact_max(self):
        val, err = _validate_budget(MAX_DAILY_BUDGET, "budget", MAX_DAILY_BUDGET)
        assert val == MAX_DAILY_BUDGET
        assert err is None


class TestValidateCampaignName:
    """Test _validate_campaign_name helper."""

    def test_valid_name(self):
        name, err = _validate_campaign_name("My Campaign 2026")
        assert name == "My Campaign 2026"
        assert err is None

    def test_none_name(self):
        name, err = _validate_campaign_name(None)
        assert name is None
        assert "required" in err

    def test_empty_string(self):
        name, err = _validate_campaign_name("")
        assert name is None
        assert "required" in err

    def test_non_string(self):
        name, err = _validate_campaign_name(123)
        assert name is None
        assert "required" in err

    def test_whitespace_only(self):
        name, err = _validate_campaign_name("   ")
        assert name is None
        assert "empty" in err

    def test_too_long(self):
        name, err = _validate_campaign_name("x" * (MAX_CAMPAIGN_NAME_LENGTH + 1))
        assert name is None
        assert "exceed" in err

    def test_max_length(self):
        name, err = _validate_campaign_name("x" * MAX_CAMPAIGN_NAME_LENGTH)
        assert name == "x" * MAX_CAMPAIGN_NAME_LENGTH
        assert err is None

    def test_strips_whitespace(self):
        name, err = _validate_campaign_name("  My Campaign  ")
        assert name == "My Campaign"
        assert err is None

    def test_special_allowed_chars(self):
        name, err = _validate_campaign_name("Campaign (Q1) - Brand & Growth #1")
        assert name == "Campaign (Q1) - Brand & Growth #1"
        assert err is None

    def test_invalid_chars(self):
        name, err = _validate_campaign_name("Campaign <script>")
        assert name is None
        assert "invalid characters" in err


class TestValidateDateIso:
    """Test _validate_date_iso helper."""

    def test_none_value(self):
        d, err = _validate_date_iso(None, "start_date")
        assert d is None
        assert err is None

    def test_valid_date(self):
        d, err = _validate_date_iso("2026-03-15", "start_date")
        assert d == date(2026, 3, 15)
        assert err is None

    def test_non_string(self):
        d, err = _validate_date_iso(20260315, "start_date")
        assert d is None
        assert "ISO format" in err

    def test_invalid_format(self):
        d, err = _validate_date_iso("03/15/2026", "start_date")
        assert d is None
        assert "valid ISO format" in err


class TestValidateDateRange:
    """Test _validate_date_range helper."""

    def test_valid_range(self):
        rng, err = _validate_date_range("2026-01-01", "2026-01-31")
        assert rng == (date(2026, 1, 1), date(2026, 1, 31))
        assert err is None

    def test_same_day(self):
        rng, err = _validate_date_range("2026-01-01", "2026-01-01")
        assert rng is not None
        assert err is None

    def test_start_after_end(self):
        rng, err = _validate_date_range("2026-02-01", "2026-01-01")
        assert rng is None
        assert "before or equal" in err

    def test_invalid_start(self):
        rng, err = _validate_date_range("bad", "2026-01-01")
        assert rng is None
        assert err is not None

    def test_invalid_end(self):
        rng, err = _validate_date_range("2026-01-01", "bad")
        assert rng is None
        assert err is not None


class TestValidateAgainstAllowlist:
    """Test _validate_against_allowlist helper."""

    def test_none_value(self):
        err = _validate_against_allowlist(None, {"a", "b"}, "field")
        assert err is None

    def test_valid_value(self):
        err = _validate_against_allowlist("a", {"a", "b"}, "field")
        assert err is None

    def test_invalid_value(self):
        err = _validate_against_allowlist("c", {"a", "b"}, "field")
        assert err is not None
        assert "Invalid" in err

    def test_non_string(self):
        err = _validate_against_allowlist(123, {"a", "b"}, "field")
        assert "must be a string" in err


# ============================================================================
# GET /api/v1/advertising/platforms
# ============================================================================


class TestListPlatforms:
    """Test listing supported advertising platforms."""

    @pytest.mark.asyncio
    async def test_list_platforms_none_connected(self, handler):
        request = MockRequest(method="GET", path="/api/v1/advertising/platforms")
        result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert "platforms" in body
        assert body["connected_count"] == 0
        assert len(body["platforms"]) == len(SUPPORTED_PLATFORMS)

    @pytest.mark.asyncio
    async def test_list_platforms_one_connected(self, handler):
        _connect_platform("google_ads")

        request = MockRequest(method="GET", path="/api/v1/advertising/platforms")
        result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["connected_count"] == 1

        # Verify the connected platform has connected=True
        google = next(p for p in body["platforms"] if p["id"] == "google_ads")
        assert google["connected"] is True

    @pytest.mark.asyncio
    async def test_list_platforms_all_connected(self, handler):
        for platform in SUPPORTED_PLATFORMS:
            _connect_platform(platform)

        request = MockRequest(method="GET", path="/api/v1/advertising/platforms")
        result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["connected_count"] == len(SUPPORTED_PLATFORMS)

    @pytest.mark.asyncio
    async def test_platform_includes_features(self, handler):
        request = MockRequest(method="GET", path="/api/v1/advertising/platforms")
        result = await handler.handle_request(request)

        body = _body(result)
        for platform in body["platforms"]:
            assert "features" in platform
            assert "name" in platform
            assert "description" in platform


# ============================================================================
# POST /api/v1/advertising/connect
# ============================================================================


class TestConnectPlatform:
    """Test connecting an advertising platform."""

    @pytest.mark.asyncio
    async def test_connect_google_ads(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/advertising/connect",
            _body={
                "platform": "google_ads",
                "credentials": {
                    "developer_token": "tok",
                    "client_id": "cid",
                    "client_secret": "csec",
                    "refresh_token": "rtok",
                    "customer_id": "cust",
                },
            },
        )

        # Mock _get_connector to avoid real imports
        with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
            result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["platform"] == "google_ads"
        assert "connected_at" in body
        assert "google_ads" in _platform_credentials

    @pytest.mark.asyncio
    async def test_connect_meta_ads(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/advertising/connect",
            _body={
                "platform": "meta_ads",
                "credentials": {"access_token": "tok", "ad_account_id": "acc"},
            },
        )

        with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
            result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["platform"] == "meta_ads"

    @pytest.mark.asyncio
    async def test_connect_missing_platform(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/advertising/connect",
            _body={"credentials": {"key": "val"}},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "required" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_connect_unsupported_platform(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/advertising/connect",
            _body={"platform": "tiktok_ads", "credentials": {"key": "val"}},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "Unsupported" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_connect_missing_credentials(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/advertising/connect",
            _body={"platform": "google_ads", "credentials": {}},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "credentials" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_connect_partial_credentials(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/advertising/connect",
            _body={
                "platform": "google_ads",
                "credentials": {"developer_token": "tok"},
            },
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "Missing required credentials" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_connect_empty_body(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/advertising/connect",
            _body={},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400


# ============================================================================
# DELETE /api/v1/advertising/{platform}
# ============================================================================


class TestDisconnectPlatform:
    """Test disconnecting an advertising platform."""

    @pytest.mark.asyncio
    async def test_disconnect_connected_platform(self, handler):
        _connect_platform("google_ads")

        request = MockRequest(method="DELETE", path="/api/v1/advertising/google_ads")
        result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["platform"] == "google_ads"
        assert "google_ads" not in _platform_credentials

    @pytest.mark.asyncio
    async def test_disconnect_not_connected(self, handler):
        request = MockRequest(method="DELETE", path="/api/v1/advertising/google_ads")
        result = await handler.handle_request(request)

        assert _status(result) == 404
        assert "not connected" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_disconnect_closes_connector(self, handler):
        _connect_platform("meta_ads")
        mock_conn = _make_mock_connector()
        _platform_connectors["meta_ads"] = mock_conn

        request = MockRequest(method="DELETE", path="/api/v1/advertising/meta_ads")
        result = await handler.handle_request(request)

        assert _status(result) == 200
        mock_conn.close.assert_awaited_once()
        assert "meta_ads" not in _platform_connectors

    @pytest.mark.asyncio
    async def test_disconnect_connector_without_close(self, handler):
        _connect_platform("google_ads")
        mock_conn = MagicMock(spec=[])  # No close method
        _platform_connectors["google_ads"] = mock_conn

        request = MockRequest(method="DELETE", path="/api/v1/advertising/google_ads")
        result = await handler.handle_request(request)

        assert _status(result) == 200


# ============================================================================
# GET /api/v1/advertising/campaigns
# ============================================================================


class TestListAllCampaigns:
    """Test listing campaigns across all platforms."""

    @pytest.mark.asyncio
    async def test_no_connected_platforms(self, handler):
        request = MockRequest(method="GET", path="/api/v1/advertising/campaigns")
        result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["campaigns"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_with_connected_platform(self, handler):
        _connect_platform("google_ads")
        mock_conn = _make_mock_connector()
        _platform_connectors["google_ads"] = mock_conn

        mock_campaign = MagicMock()
        mock_campaign.id = "camp-1"
        mock_campaign.name = "Test Campaign"
        mock_campaign.status = "ENABLED"
        mock_campaign.campaign_type = "SEARCH"
        mock_campaign.budget_micros = 50_000_000
        mock_campaign.start_date = None
        mock_campaign.end_date = None
        mock_campaign.bidding_strategy = "MAXIMIZE_CLICKS"
        mock_conn.get_campaigns.return_value = [mock_campaign]

        # Patch _get_connector to return our mock
        with patch.object(
            handler, "_get_connector", new_callable=AsyncMock, return_value=mock_conn
        ):
            request = MockRequest(method="GET", path="/api/v1/advertising/campaigns")
            result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] >= 1

    @pytest.mark.asyncio
    async def test_campaigns_with_limit(self, handler):
        request = MockRequest(
            method="GET",
            path="/api/v1/advertising/campaigns",
            query={"limit": "5"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_campaigns_fetch_error_handled(self, handler):
        """Verify that errors fetching from one platform don't crash the request."""
        _connect_platform("google_ads")

        # Patch _fetch_platform_campaigns to raise
        with patch.object(
            handler,
            "_fetch_platform_campaigns",
            new_callable=AsyncMock,
            side_effect=ConnectionError("API down"),
        ):
            request = MockRequest(method="GET", path="/api/v1/advertising/campaigns")
            result = await handler.handle_request(request)

        # Should still return 200 with empty campaigns (error logged, not raised)
        assert _status(result) == 200


# ============================================================================
# GET /api/v1/advertising/{platform}/campaigns
# ============================================================================


class TestListPlatformCampaigns:
    """Test listing campaigns for a specific platform."""

    @pytest.mark.asyncio
    async def test_platform_not_connected(self, handler):
        request = MockRequest(method="GET", path="/api/v1/advertising/google_ads/campaigns")
        result = await handler.handle_request(request)

        assert _status(result) == 404
        assert "not connected" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_platform_campaigns_empty(self, handler):
        _connect_platform("google_ads")

        with patch.object(
            handler,
            "_fetch_platform_campaigns",
            new_callable=AsyncMock,
            return_value=[],
        ):
            request = MockRequest(method="GET", path="/api/v1/advertising/google_ads/campaigns")
            result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["campaigns"] == []
        assert body["total"] == 0
        assert body["platform"] == "google_ads"

    @pytest.mark.asyncio
    async def test_platform_campaigns_with_status_filter(self, handler):
        _connect_platform("meta_ads")

        with patch.object(
            handler,
            "_fetch_platform_campaigns",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_fetch:
            request = MockRequest(
                method="GET",
                path="/api/v1/advertising/meta_ads/campaigns",
                query={"status": "ACTIVE"},
            )
            result = await handler.handle_request(request)

        assert _status(result) == 200
        # Verify the status_filter was passed
        mock_fetch.assert_awaited_once_with("meta_ads", "ACTIVE")


# ============================================================================
# GET /api/v1/advertising/{platform}/campaigns/{id}
# ============================================================================


class TestGetCampaign:
    """Test getting a specific campaign by ID."""

    @pytest.mark.asyncio
    async def test_platform_not_connected(self, handler):
        request = MockRequest(
            method="GET",
            path="/api/v1/advertising/google_ads/campaigns/camp-123",
        )
        result = await handler.handle_request(request)

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_campaign_success(self, handler):
        _connect_platform("google_ads")
        mock_conn = _make_mock_connector()

        mock_campaign = MagicMock()
        mock_campaign.id = "camp-123"
        mock_campaign.name = "Test"
        mock_campaign.status = "ENABLED"
        mock_campaign.campaign_type = "SEARCH"
        mock_campaign.budget_micros = 10_000_000
        mock_campaign.start_date = None
        mock_campaign.end_date = None
        mock_campaign.bidding_strategy = "CPC"
        mock_conn.get_campaign.return_value = mock_campaign

        with patch.object(
            handler, "_get_connector", new_callable=AsyncMock, return_value=mock_conn
        ):
            request = MockRequest(
                method="GET",
                path="/api/v1/advertising/google_ads/campaigns/camp-123",
            )
            result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == "camp-123"
        assert body["platform"] == "google_ads"

    @pytest.mark.asyncio
    async def test_get_campaign_not_found(self, handler):
        _connect_platform("google_ads")
        mock_conn = _make_mock_connector()
        mock_conn.get_campaign.side_effect = ValueError("Not found")

        with patch.object(
            handler, "_get_connector", new_callable=AsyncMock, return_value=mock_conn
        ):
            request = MockRequest(
                method="GET",
                path="/api/v1/advertising/google_ads/campaigns/nonexistent",
            )
            result = await handler.handle_request(request)

        assert _status(result) == 404
        assert "not found" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_get_campaign_no_connector(self, handler):
        _connect_platform("google_ads")

        with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
            request = MockRequest(
                method="GET",
                path="/api/v1/advertising/google_ads/campaigns/camp-123",
            )
            result = await handler.handle_request(request)

        assert _status(result) == 500


# ============================================================================
# POST /api/v1/advertising/{platform}/campaigns
# ============================================================================


class TestCreateCampaign:
    """Test creating a campaign on a platform."""

    @pytest.mark.asyncio
    async def test_platform_not_connected(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/advertising/google_ads/campaigns",
            _body={"name": "Test Campaign"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_create_google_campaign(self, handler):
        _connect_platform("google_ads")
        mock_conn = _make_mock_connector()
        mock_conn.create_campaign.return_value = "camp-new"

        with patch.object(
            handler, "_get_connector", new_callable=AsyncMock, return_value=mock_conn
        ):
            request = MockRequest(
                method="POST",
                path="/api/v1/advertising/google_ads/campaigns",
                _body={"name": "Test Campaign", "daily_budget": 100},
            )
            result = await handler.handle_request(request)

        assert _status(result) == 201
        body = _body(result)
        assert body["campaign_id"] == "camp-new"
        assert body["platform"] == "google_ads"

    @pytest.mark.asyncio
    async def test_create_microsoft_campaign(self, handler):
        _connect_platform("microsoft_ads")
        mock_conn = _make_mock_connector()
        mock_conn.create_campaign.return_value = "ms-camp-1"

        with patch.object(
            handler, "_get_connector", new_callable=AsyncMock, return_value=mock_conn
        ):
            request = MockRequest(
                method="POST",
                path="/api/v1/advertising/microsoft_ads/campaigns",
                _body={"name": "MS Campaign", "daily_budget": 50},
            )
            result = await handler.handle_request(request)

        assert _status(result) == 201
        body = _body(result)
        assert body["campaign_id"] == "ms-camp-1"

    @pytest.mark.asyncio
    async def test_create_meta_campaign(self, handler):
        _connect_platform("meta_ads")
        mock_conn = _make_mock_connector()

        mock_campaign = MagicMock()
        mock_campaign.id = "meta-camp-1"
        mock_campaign.name = "Meta Campaign"
        mock_campaign.status = "ACTIVE"
        mock_campaign.objective = "OUTCOME_TRAFFIC"
        mock_campaign.daily_budget = 100
        mock_campaign.lifetime_budget = None
        mock_campaign.start_time = None
        mock_campaign.stop_time = None
        mock_campaign.spend_cap = None
        mock_conn.create_campaign.return_value = mock_campaign

        with patch.object(
            handler, "_get_connector", new_callable=AsyncMock, return_value=mock_conn
        ):
            request = MockRequest(
                method="POST",
                path="/api/v1/advertising/meta_ads/campaigns",
                _body={"name": "Meta Campaign"},
            )
            result = await handler.handle_request(request)

        assert _status(result) == 201
        body = _body(result)
        assert body["platform"] == "meta_ads"

    @pytest.mark.asyncio
    async def test_create_linkedin_campaign_requires_group(self, handler):
        _connect_platform("linkedin_ads")
        mock_conn = _make_mock_connector()

        with patch.object(
            handler, "_get_connector", new_callable=AsyncMock, return_value=mock_conn
        ):
            request = MockRequest(
                method="POST",
                path="/api/v1/advertising/linkedin_ads/campaigns",
                _body={"name": "LinkedIn Campaign"},
            )
            result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "campaign_group_id" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_create_linkedin_campaign_with_group(self, handler):
        _connect_platform("linkedin_ads")
        mock_conn = _make_mock_connector()

        mock_campaign = MagicMock()
        mock_campaign.id = "li-camp-1"
        mock_campaign.name = "LinkedIn Campaign"
        mock_campaign.status = "ACTIVE"
        mock_campaign.objective_type = "WEBSITE_VISITS"
        mock_campaign.daily_budget = 50
        mock_campaign.total_budget = None
        mock_campaign.run_schedule_start = None
        mock_campaign.run_schedule_end = None
        mock_campaign.campaign_type = "SPONSORED_UPDATES"
        mock_conn.create_campaign.return_value = mock_campaign

        with patch.object(
            handler, "_get_connector", new_callable=AsyncMock, return_value=mock_conn
        ):
            request = MockRequest(
                method="POST",
                path="/api/v1/advertising/linkedin_ads/campaigns",
                _body={
                    "name": "LinkedIn Campaign",
                    "campaign_group_id": "grp-1",
                    "daily_budget": 50,
                },
            )
            result = await handler.handle_request(request)

        assert _status(result) == 201

    @pytest.mark.asyncio
    async def test_create_campaign_missing_name(self, handler):
        _connect_platform("google_ads")

        request = MockRequest(
            method="POST",
            path="/api/v1/advertising/google_ads/campaigns",
            _body={"daily_budget": 100},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "name" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_create_campaign_invalid_name(self, handler):
        _connect_platform("google_ads")

        request = MockRequest(
            method="POST",
            path="/api/v1/advertising/google_ads/campaigns",
            _body={"name": "<script>alert(1)</script>"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "invalid characters" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_create_campaign_negative_budget(self, handler):
        _connect_platform("google_ads")

        request = MockRequest(
            method="POST",
            path="/api/v1/advertising/google_ads/campaigns",
            _body={"name": "Test", "daily_budget": -10},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "non-negative" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_create_campaign_budget_exceeds_max(self, handler):
        _connect_platform("google_ads")

        request = MockRequest(
            method="POST",
            path="/api/v1/advertising/google_ads/campaigns",
            _body={"name": "Test", "daily_budget": MAX_DAILY_BUDGET + 1},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "exceeds" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_create_campaign_invalid_type(self, handler):
        _connect_platform("google_ads")

        request = MockRequest(
            method="POST",
            path="/api/v1/advertising/google_ads/campaigns",
            _body={"name": "Test", "campaign_type": "INVALID_TYPE"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "Invalid" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_create_campaign_invalid_objective(self, handler):
        _connect_platform("google_ads")

        request = MockRequest(
            method="POST",
            path="/api/v1/advertising/google_ads/campaigns",
            _body={"name": "Test", "objective": "INVALID_OBJ"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "Invalid" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_create_campaign_invalid_date_range(self, handler):
        _connect_platform("google_ads")

        request = MockRequest(
            method="POST",
            path="/api/v1/advertising/google_ads/campaigns",
            _body={
                "name": "Test",
                "start_date": "2026-03-01",
                "end_date": "2026-02-01",
            },
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "before" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_create_campaign_connector_error(self, handler):
        _connect_platform("google_ads")
        mock_conn = _make_mock_connector()
        mock_conn.create_campaign.side_effect = ConnectionError("API failure")

        with patch.object(
            handler, "_get_connector", new_callable=AsyncMock, return_value=mock_conn
        ):
            request = MockRequest(
                method="POST",
                path="/api/v1/advertising/google_ads/campaigns",
                _body={"name": "Test Campaign"},
            )
            result = await handler.handle_request(request)

        assert _status(result) == 500
        assert "Failed to create" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_create_campaign_no_connector(self, handler):
        _connect_platform("google_ads")

        with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
            request = MockRequest(
                method="POST",
                path="/api/v1/advertising/google_ads/campaigns",
                _body={"name": "Test Campaign"},
            )
            result = await handler.handle_request(request)

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_create_campaign_total_budget_validation(self, handler):
        _connect_platform("google_ads")

        request = MockRequest(
            method="POST",
            path="/api/v1/advertising/google_ads/campaigns",
            _body={"name": "Test", "total_budget": MAX_TOTAL_BUDGET + 1},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "exceeds" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_create_campaign_valid_campaign_type(self, handler):
        _connect_platform("google_ads")
        mock_conn = _make_mock_connector()
        mock_conn.create_campaign.return_value = "camp-typed"

        with patch.object(
            handler, "_get_connector", new_callable=AsyncMock, return_value=mock_conn
        ):
            request = MockRequest(
                method="POST",
                path="/api/v1/advertising/google_ads/campaigns",
                _body={"name": "Test", "campaign_type": "SEARCH"},
            )
            result = await handler.handle_request(request)

        assert _status(result) == 201


# ============================================================================
# PUT /api/v1/advertising/{platform}/campaigns/{id}
# ============================================================================


class TestUpdateCampaign:
    """Test updating a campaign."""

    @pytest.mark.asyncio
    async def test_platform_not_connected(self, handler):
        request = MockRequest(
            method="PUT",
            path="/api/v1/advertising/google_ads/campaigns/camp-123",
            _body={"status": "PAUSED"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_update_status(self, handler):
        _connect_platform("google_ads")
        mock_conn = _make_mock_connector()

        with patch.object(
            handler, "_get_connector", new_callable=AsyncMock, return_value=mock_conn
        ):
            request = MockRequest(
                method="PUT",
                path="/api/v1/advertising/google_ads/campaigns/camp-123",
                _body={"status": "PAUSED"},
            )
            result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["campaign_id"] == "camp-123"
        mock_conn.update_campaign_status.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_update_budget(self, handler):
        _connect_platform("google_ads")
        mock_conn = _make_mock_connector()

        with patch.object(
            handler, "_get_connector", new_callable=AsyncMock, return_value=mock_conn
        ):
            request = MockRequest(
                method="PUT",
                path="/api/v1/advertising/google_ads/campaigns/camp-123",
                _body={"daily_budget": 200},
            )
            result = await handler.handle_request(request)

        assert _status(result) == 200
        mock_conn.update_campaign_budget.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_update_invalid_status(self, handler):
        _connect_platform("google_ads")
        mock_conn = _make_mock_connector()

        with patch.object(
            handler, "_get_connector", new_callable=AsyncMock, return_value=mock_conn
        ):
            request = MockRequest(
                method="PUT",
                path="/api/v1/advertising/google_ads/campaigns/camp-123",
                _body={"status": "INVALID_STATUS"},
            )
            result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "Invalid" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_update_invalid_budget(self, handler):
        _connect_platform("google_ads")
        mock_conn = _make_mock_connector()

        with patch.object(
            handler, "_get_connector", new_callable=AsyncMock, return_value=mock_conn
        ):
            request = MockRequest(
                method="PUT",
                path="/api/v1/advertising/google_ads/campaigns/camp-123",
                _body={"daily_budget": -50},
            )
            result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "non-negative" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_update_invalid_name(self, handler):
        _connect_platform("google_ads")
        mock_conn = _make_mock_connector()

        with patch.object(
            handler, "_get_connector", new_callable=AsyncMock, return_value=mock_conn
        ):
            request = MockRequest(
                method="PUT",
                path="/api/v1/advertising/google_ads/campaigns/camp-123",
                _body={"name": "<bad>"},
            )
            result = await handler.handle_request(request)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_update_connector_error(self, handler):
        _connect_platform("google_ads")
        mock_conn = _make_mock_connector()
        mock_conn.update_campaign_status.side_effect = ConnectionError("Error")

        with patch.object(
            handler, "_get_connector", new_callable=AsyncMock, return_value=mock_conn
        ):
            request = MockRequest(
                method="PUT",
                path="/api/v1/advertising/google_ads/campaigns/camp-123",
                _body={"status": "PAUSED"},
            )
            result = await handler.handle_request(request)

        assert _status(result) == 500
        assert "Failed to update" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_update_no_connector(self, handler):
        _connect_platform("google_ads")

        with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
            request = MockRequest(
                method="PUT",
                path="/api/v1/advertising/google_ads/campaigns/camp-123",
                _body={"status": "PAUSED"},
            )
            result = await handler.handle_request(request)

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_update_meta_status(self, handler):
        _connect_platform("meta_ads")
        mock_conn = _make_mock_connector()

        with patch.object(
            handler, "_get_connector", new_callable=AsyncMock, return_value=mock_conn
        ):
            request = MockRequest(
                method="PUT",
                path="/api/v1/advertising/meta_ads/campaigns/camp-456",
                _body={"status": "PAUSED"},
            )
            result = await handler.handle_request(request)

        assert _status(result) == 200
        mock_conn.update_campaign.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_update_microsoft_budget(self, handler):
        _connect_platform("microsoft_ads")
        mock_conn = _make_mock_connector()

        with patch.object(
            handler, "_get_connector", new_callable=AsyncMock, return_value=mock_conn
        ):
            request = MockRequest(
                method="PUT",
                path="/api/v1/advertising/microsoft_ads/campaigns/ms-camp",
                _body={"daily_budget": 100},
            )
            result = await handler.handle_request(request)

        assert _status(result) == 200
        mock_conn.update_campaign_budget.assert_awaited_once()


# ============================================================================
# GET /api/v1/advertising/performance
# ============================================================================


class TestCrossPlatformPerformance:
    """Test cross-platform performance endpoint."""

    @pytest.mark.asyncio
    async def test_no_platforms_connected(self, handler):
        request = MockRequest(method="GET", path="/api/v1/advertising/performance")
        result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert "date_range" in body
        assert "totals" in body
        assert body["platforms"] == []

    @pytest.mark.asyncio
    async def test_with_days_param(self, handler):
        request = MockRequest(
            method="GET",
            path="/api/v1/advertising/performance",
            query={"days": "7"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        start = date.fromisoformat(body["date_range"]["start"])
        end = date.fromisoformat(body["date_range"]["end"])
        assert (end - start).days == 7

    @pytest.mark.asyncio
    async def test_with_connected_platform(self, handler):
        _connect_platform("google_ads")
        perf_data = {
            "platform": "google_ads",
            "impressions": 1000,
            "clicks": 50,
            "cost": 100.0,
            "conversions": 5,
            "conversion_value": 500.0,
        }

        with patch.object(
            handler,
            "_fetch_platform_performance",
            new_callable=AsyncMock,
            return_value=perf_data,
        ):
            request = MockRequest(method="GET", path="/api/v1/advertising/performance")
            result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert len(body["platforms"]) == 1
        assert body["totals"]["impressions"] == 1000

    @pytest.mark.asyncio
    async def test_totals_calculation(self, handler):
        """Verify CTR, CPC, CPM, ROAS are correctly calculated."""
        _connect_platform("google_ads")
        perf_data = {
            "platform": "google_ads",
            "impressions": 10000,
            "clicks": 200,
            "cost": 400.0,
            "conversions": 20,
            "conversion_value": 2000.0,
        }

        with patch.object(
            handler,
            "_fetch_platform_performance",
            new_callable=AsyncMock,
            return_value=perf_data,
        ):
            request = MockRequest(method="GET", path="/api/v1/advertising/performance")
            result = await handler.handle_request(request)

        totals = _body(result)["totals"]
        assert totals["ctr"] == round(200 / 10000 * 100, 2)
        assert totals["cpc"] == round(400.0 / 200, 2)
        assert totals["cpm"] == round(400.0 / 10000 * 1000, 2)
        assert totals["roas"] == round(2000.0 / 400.0, 2)


# ============================================================================
# GET /api/v1/advertising/{platform}/performance
# ============================================================================


class TestPlatformPerformance:
    """Test platform-specific performance endpoint."""

    @pytest.mark.asyncio
    async def test_platform_not_connected(self, handler):
        request = MockRequest(method="GET", path="/api/v1/advertising/google_ads/performance")
        result = await handler.handle_request(request)

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_platform_performance(self, handler):
        _connect_platform("google_ads")
        perf_data = {"platform": "google_ads", "impressions": 500}

        with patch.object(
            handler,
            "_fetch_platform_performance",
            new_callable=AsyncMock,
            return_value=perf_data,
        ):
            request = MockRequest(
                method="GET",
                path="/api/v1/advertising/google_ads/performance",
            )
            result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["impressions"] == 500

    @pytest.mark.asyncio
    async def test_platform_performance_with_days(self, handler):
        _connect_platform("meta_ads")
        perf_data = {"platform": "meta_ads", "impressions": 100}

        with patch.object(
            handler,
            "_fetch_platform_performance",
            new_callable=AsyncMock,
            return_value=perf_data,
        ):
            request = MockRequest(
                method="GET",
                path="/api/v1/advertising/meta_ads/performance",
                query={"days": "14"},
            )
            result = await handler.handle_request(request)

        assert _status(result) == 200


# ============================================================================
# POST /api/v1/advertising/analyze
# ============================================================================


class TestAnalyzePerformance:
    """Test performance analysis endpoint."""

    @pytest.mark.asyncio
    async def test_analyze_default(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/advertising/analyze",
            _body={},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert "analysis_id" in body
        assert body["type"] == "performance_review"
        assert body["status"] == "completed"

    @pytest.mark.asyncio
    async def test_analyze_with_type(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/advertising/analyze",
            _body={"type": "budget_optimization"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200
        assert _body(result)["type"] == "budget_optimization"

    @pytest.mark.asyncio
    async def test_analyze_invalid_type(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/advertising/analyze",
            _body={"type": "invalid_type"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "Invalid" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_analyze_with_platforms(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/advertising/analyze",
            _body={"platforms": ["google_ads", "meta_ads"]},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert "google_ads" in body["platforms_analyzed"]
        assert "meta_ads" in body["platforms_analyzed"]

    @pytest.mark.asyncio
    async def test_analyze_unsupported_platform(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/advertising/analyze",
            _body={"platforms": ["tiktok_ads"]},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "Unsupported" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_analyze_platforms_not_list(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/advertising/analyze",
            _body={"platforms": "google_ads"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "list" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_analyze_invalid_days(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/advertising/analyze",
            _body={"days": "not_a_number"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "integer" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_analyze_days_too_large(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/advertising/analyze",
            _body={"days": MAX_ANALYSIS_DAYS + 1},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400
        assert str(MAX_ANALYSIS_DAYS) in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_analyze_days_zero(self, handler):
        request = MockRequest(
            method="POST",
            path="/api/v1/advertising/analyze",
            _body={"days": 0},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_analyze_with_connected_platforms_data(self, handler):
        _connect_platform("google_ads")
        perf_data = {
            "platform": "google_ads",
            "impressions": 5000,
            "clicks": 100,
            "cost": 200,
            "conversions": 10,
            "conversion_value": 1000,
            "roas": 5.0,
            "cpc": 2.0,
        }

        with patch.object(
            handler,
            "_fetch_platform_performance",
            new_callable=AsyncMock,
            return_value=perf_data,
        ):
            request = MockRequest(
                method="POST",
                path="/api/v1/advertising/analyze",
                _body={"platforms": ["google_ads"]},
            )
            result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert "summary" in body
        assert "recommendations" in body
        assert "insights" in body

    @pytest.mark.asyncio
    async def test_analyze_all_valid_types(self, handler):
        """Verify all allowed analysis types are accepted."""
        for analysis_type in ALLOWED_ANALYSIS_TYPES:
            request = MockRequest(
                method="POST",
                path="/api/v1/advertising/analyze",
                _body={"type": analysis_type},
            )
            result = await handler.handle_request(request)
            assert _status(result) == 200, f"Failed for type: {analysis_type}"


# ============================================================================
# GET /api/v1/advertising/budget-recommendations
# ============================================================================


class TestBudgetRecommendations:
    """Test budget recommendations endpoint."""

    @pytest.mark.asyncio
    async def test_default_recommendations(self, handler):
        request = MockRequest(
            method="GET",
            path="/api/v1/advertising/budget-recommendations",
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert "total_budget" in body
        assert "objective" in body
        assert body["objective"] == "balanced"
        assert "rationale" in body

    @pytest.mark.asyncio
    async def test_budget_with_custom_amount(self, handler):
        request = MockRequest(
            method="GET",
            path="/api/v1/advertising/budget-recommendations",
            query={"budget": "50000"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 200
        assert _body(result)["total_budget"] == 50000.0

    @pytest.mark.asyncio
    async def test_budget_with_objective(self, handler):
        for obj in ALLOWED_BUDGET_OBJECTIVES:
            request = MockRequest(
                method="GET",
                path="/api/v1/advertising/budget-recommendations",
                query={"objective": obj},
            )
            result = await handler.handle_request(request)

            assert _status(result) == 200, f"Failed for objective: {obj}"
            assert _body(result)["objective"] == obj

    @pytest.mark.asyncio
    async def test_budget_invalid_objective(self, handler):
        request = MockRequest(
            method="GET",
            path="/api/v1/advertising/budget-recommendations",
            query={"objective": "invalid"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "Invalid" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_budget_negative_amount(self, handler):
        request = MockRequest(
            method="GET",
            path="/api/v1/advertising/budget-recommendations",
            query={"budget": "-100"},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "non-negative" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_budget_exceeds_max(self, handler):
        request = MockRequest(
            method="GET",
            path="/api/v1/advertising/budget-recommendations",
            query={"budget": str(MAX_TOTAL_BUDGET + 1)},
        )
        result = await handler.handle_request(request)

        assert _status(result) == 400
        assert "exceeds" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_budget_with_platforms(self, handler):
        _connect_platform("google_ads")
        _connect_platform("meta_ads")
        perf_data = {
            "platform": "test",
            "impressions": 1000,
            "clicks": 50,
            "cost": 100,
            "roas": 3.0,
        }

        with patch.object(
            handler,
            "_fetch_platform_performance",
            new_callable=AsyncMock,
            return_value=perf_data,
        ):
            request = MockRequest(
                method="GET",
                path="/api/v1/advertising/budget-recommendations",
                query={"budget": "20000"},
            )
            result = await handler.handle_request(request)

        assert _status(result) == 200
        body = _body(result)
        assert len(body["recommendations"]) == 2

    @pytest.mark.asyncio
    async def test_budget_rationale_by_objective(self, handler):
        """Verify different objectives produce different rationales."""
        rationales = set()
        for obj in ALLOWED_BUDGET_OBJECTIVES:
            request = MockRequest(
                method="GET",
                path="/api/v1/advertising/budget-recommendations",
                query={"objective": obj},
            )
            result = await handler.handle_request(request)
            rationales.add(_body(result)["rationale"])

        assert len(rationales) == len(ALLOWED_BUDGET_OBJECTIVES)


# ============================================================================
# Endpoint not found / Method routing
# ============================================================================


class TestEndpointNotFound:
    """Test fallback for unknown routes."""

    @pytest.mark.asyncio
    async def test_unknown_sub_path(self, handler):
        request = MockRequest(method="GET", path="/api/v1/advertising/unknown-endpoint")
        result = await handler.handle_request(request)

        assert _status(result) == 404
        assert "not found" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_wrong_method_for_connect(self, handler):
        request = MockRequest(method="GET", path="/api/v1/advertising/connect")
        result = await handler.handle_request(request)

        # GET on /connect doesn't match any route
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_wrong_method_for_analyze(self, handler):
        request = MockRequest(method="GET", path="/api/v1/advertising/analyze")
        result = await handler.handle_request(request)

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_post_to_platforms_list(self, handler):
        request = MockRequest(method="POST", path="/api/v1/advertising/platforms")
        result = await handler.handle_request(request)

        assert _status(result) == 404


# ============================================================================
# Internal helper methods
# ============================================================================


class TestHelperMethods:
    """Test internal helper and utility methods."""

    def test_json_response_format(self, handler):
        result = handler._json_response(200, {"key": "value"})
        assert result["status_code"] == 200
        assert result["headers"]["Content-Type"] == "application/json"
        assert result["body"]["key"] == "value"

    def test_error_response_format(self, handler):
        result = handler._error_response(400, "Bad request")
        assert result["status_code"] == 400
        assert result["body"]["error"] == "Bad request"

    def test_get_required_credentials_google(self, handler):
        creds = handler._get_required_credentials("google_ads")
        assert "developer_token" in creds
        assert "client_id" in creds
        assert "customer_id" in creds

    def test_get_required_credentials_meta(self, handler):
        creds = handler._get_required_credentials("meta_ads")
        assert "access_token" in creds
        assert "ad_account_id" in creds

    def test_get_required_credentials_unknown(self, handler):
        creds = handler._get_required_credentials("unknown_platform")
        assert creds == []

    def test_generate_performance_summary_empty(self, handler):
        summary = handler._generate_performance_summary({})
        assert summary["total_spend"] == 0
        assert summary["total_conversions"] == 0
        assert summary["platforms_analyzed"] == 0

    def test_generate_performance_summary_with_data(self, handler):
        data = {
            "google_ads": {"cost": 100, "conversions": 10, "roas": 5},
            "meta_ads": {"cost": 200, "conversions": 20, "roas": 3},
        }
        summary = handler._generate_performance_summary(data)
        assert summary["total_spend"] == 300
        assert summary["total_conversions"] == 30
        assert summary["platforms_analyzed"] == 2
        assert summary["best_performing_platform"] == "google_ads"

    def test_generate_recommendations_high_roas(self, handler):
        data = {"google_ads": {"roas": 5, "cpc": 1}}
        recs = handler._generate_recommendations(data)
        # High ROAS should trigger increase_budget recommendation
        assert any(r["type"] == "increase_budget" for r in recs)

    def test_generate_recommendations_low_roas(self, handler):
        data = {"google_ads": {"roas": 0.5, "cpc": 1}}
        recs = handler._generate_recommendations(data)
        assert any(r["type"] == "optimize" for r in recs)

    def test_generate_recommendations_high_cpc(self, handler):
        data = {"google_ads": {"roas": 2, "cpc": 10}}
        recs = handler._generate_recommendations(data)
        assert any(r["type"] == "reduce_cpc" for r in recs)

    def test_generate_insights_with_spend(self, handler):
        data = {
            "google_ads": {"cost": 300},
            "meta_ads": {"cost": 700},
        }
        insights = handler._generate_insights(data)
        assert len(insights) == 2
        # Verify percentage mentions
        assert any("google_ads" in i for i in insights)
        assert any("meta_ads" in i for i in insights)

    def test_generate_insights_no_spend(self, handler):
        data = {
            "google_ads": {"cost": 0},
            "meta_ads": {"cost": 0},
        }
        insights = handler._generate_insights(data)
        assert len(insights) == 0

    def test_calculate_budget_recommendations_balanced(self, handler):
        data = {
            "google_ads": {"roas": 4},
            "meta_ads": {"roas": 2},
        }
        recs = handler._calculate_budget_recommendations(data, 10000, "balanced")
        assert len(recs) == 2
        # Total recommended should approximately equal total budget
        total_rec = sum(r["recommended_budget"] for r in recs)
        assert abs(total_rec - 10000) < 1  # Allow rounding

    def test_calculate_budget_recommendations_awareness(self, handler):
        data = {
            "google_ads": {"roas": 4},
            "meta_ads": {"roas": 2},
        }
        recs = handler._calculate_budget_recommendations(data, 10000, "awareness")
        # Awareness distributes evenly
        assert len(recs) == 2
        assert recs[0]["recommended_budget"] == recs[1]["recommended_budget"]

    def test_generate_budget_rationale(self, handler):
        assert "conversions" in handler._generate_budget_rationale({}, "conversions").lower()
        assert "awareness" in handler._generate_budget_rationale({}, "awareness").lower()
        assert "balanced" in handler._generate_budget_rationale({}, "balanced").lower()


# ============================================================================
# Circuit breaker
# ============================================================================


class TestCircuitBreaker:
    """Test circuit breaker integration."""

    def test_get_circuit_breaker(self):
        cb = get_advertising_circuit_breaker()
        assert cb is not None
        assert cb.name == "advertising_handler"

    @pytest.mark.asyncio
    async def test_connector_uses_circuit_breaker(self, handler):
        """Verify that _get_connector checks the circuit breaker."""
        import time as _time

        _connect_platform("google_ads")
        cb = get_advertising_circuit_breaker()

        # Force circuit breaker open by setting single-entity state:
        # _single_open_at to a recent time and _single_failures above threshold
        cb._single_failures = cb.failure_threshold + 1
        cb._single_open_at = _time.time()

        connector = await handler._get_connector("google_ads")
        assert connector is None


# ============================================================================
# UnifiedCampaign and UnifiedPerformance dataclasses
# ============================================================================


class TestDataclasses:
    """Test dataclass serialization."""

    def test_unified_campaign_to_dict(self):
        from aragora.server.handlers.features.advertising import UnifiedCampaign

        camp = UnifiedCampaign(
            id="camp-1",
            platform="google_ads",
            name="Test",
            status="ENABLED",
            objective="SEARCH",
            daily_budget=100.0,
            total_budget=None,
            start_date=date(2026, 1, 1),
            end_date=date(2026, 12, 31),
            created_at=None,
        )
        d = camp.to_dict()
        assert d["id"] == "camp-1"
        assert d["start_date"] == "2026-01-01"
        assert d["end_date"] == "2026-12-31"
        assert d["created_at"] is None

    def test_unified_performance_to_dict(self):
        from aragora.server.handlers.features.advertising import UnifiedPerformance

        perf = UnifiedPerformance(
            platform="google_ads",
            campaign_id="camp-1",
            campaign_name="Test",
            date_range=(date(2026, 1, 1), date(2026, 1, 31)),
            impressions=10000,
            clicks=200,
            cost=400.0,
            conversions=20,
            conversion_value=2000.0,
            ctr=2.0,
            cpc=2.0,
            cpm=40.0,
            roas=5.0,
        )
        d = perf.to_dict()
        assert d["impressions"] == 10000
        assert d["date_range"]["start"] == "2026-01-01"
        assert d["cost"] == 400.0
        assert d["roas"] == 5.0
