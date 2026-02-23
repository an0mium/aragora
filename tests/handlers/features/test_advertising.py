"""Tests for advertising handler endpoints.

Tests the advertising API endpoints including:
- GET    /api/v1/advertising/platforms               - List connected platforms
- POST   /api/v1/advertising/connect                 - Connect a platform
- DELETE /api/v1/advertising/{platform}              - Disconnect platform
- GET    /api/v1/advertising/campaigns               - List all campaigns
- GET    /api/v1/advertising/{platform}/campaigns    - List platform campaigns
- POST   /api/v1/advertising/{platform}/campaigns    - Create campaign
- PUT    /api/v1/advertising/{platform}/campaigns/{id} - Update campaign
- GET    /api/v1/advertising/{platform}/campaigns/{id} - Get campaign
- GET    /api/v1/advertising/performance             - Cross-platform performance
- GET    /api/v1/advertising/{platform}/performance  - Platform performance
- POST   /api/v1/advertising/analyze                 - Multi-agent analysis
- GET    /api/v1/advertising/budget-recommendations  - Budget recommendations
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: dict) -> dict:
    """Extract the body dict from a handler response."""
    return result.get("body", result)


def _status(result: dict) -> int:
    """Extract the HTTP status code from a handler response."""
    return result.get("status_code", 200)


class MockRequest:
    """Lightweight mock for aiohttp-style request objects."""

    def __init__(
        self,
        method: str = "GET",
        path: str = "/",
        query: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
    ):
        self.method = method
        self.path = path
        self.query = query or {}
        self._body = body
        self.content_length = len(json.dumps(body).encode()) if body else 0

    async def json(self) -> dict[str, Any]:
        """Return the JSON body."""
        if self._body is None:
            raise ValueError("No body")
        return self._body

    async def read(self) -> bytes:
        """Return raw body bytes."""
        return json.dumps(self._body or {}).encode()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Reset module-level state between tests."""
    import aragora.server.handlers.features.advertising as adv_mod

    # Save originals
    orig_creds = dict(adv_mod._platform_credentials)
    orig_connectors = dict(adv_mod._platform_connectors)

    # Reset rate limiters
    try:
        from aragora.server.middleware.rate_limit.registry import reset_rate_limiters

        reset_rate_limiters()
    except ImportError:
        pass

    yield

    # Restore originals
    adv_mod._platform_credentials.clear()
    adv_mod._platform_credentials.update(orig_creds)
    adv_mod._platform_connectors.clear()
    adv_mod._platform_connectors.update(orig_connectors)

    try:
        from aragora.server.middleware.rate_limit.registry import reset_rate_limiters

        reset_rate_limiters()
    except ImportError:
        pass


@pytest.fixture
def handler():
    """Create an AdvertisingHandler instance."""
    from aragora.server.handlers.features.advertising import AdvertisingHandler

    return AdvertisingHandler(ctx={})


@pytest.fixture
def _seed_google_creds():
    """Seed Google Ads credentials into module state."""
    import aragora.server.handlers.features.advertising as adv_mod

    adv_mod._platform_credentials["google_ads"] = {
        "credentials": {
            "developer_token": "dev-tok",
            "client_id": "cid",
            "client_secret": "csec",
            "refresh_token": "rtok",
            "customer_id": "123",
        },
        "connected_at": "2026-01-01T00:00:00+00:00",
    }
    yield
    adv_mod._platform_credentials.pop("google_ads", None)
    adv_mod._platform_connectors.pop("google_ads", None)


@pytest.fixture
def _seed_meta_creds():
    """Seed Meta Ads credentials into module state."""
    import aragora.server.handlers.features.advertising as adv_mod

    adv_mod._platform_credentials["meta_ads"] = {
        "credentials": {
            "access_token": "tok",
            "ad_account_id": "act_123",
        },
        "connected_at": "2026-01-15T00:00:00+00:00",
    }
    yield
    adv_mod._platform_credentials.pop("meta_ads", None)
    adv_mod._platform_connectors.pop("meta_ads", None)


def _mock_connector(**overrides):
    """Build an AsyncMock connector with common campaign methods."""
    connector = AsyncMock()
    connector.get_campaigns = AsyncMock(return_value=[])
    connector.get_campaign = AsyncMock()
    connector.create_campaign = AsyncMock(return_value="camp-1")
    connector.update_campaign_status = AsyncMock()
    connector.update_campaign_budget = AsyncMock()
    connector.update_campaign = AsyncMock()
    connector.get_campaign_performance = AsyncMock(return_value=[])
    connector.get_insights = AsyncMock(return_value=[])
    connector.get_account_analytics = AsyncMock()
    connector.close = AsyncMock()
    for k, v in overrides.items():
        setattr(connector, k, v)
    return connector


# =============================================================================
# can_handle() Routing Tests
# =============================================================================


class TestCanHandle:
    """Tests for can_handle() route matching."""

    def test_handles_platforms_path(self, handler):
        assert handler.can_handle("/api/v1/advertising/platforms")

    def test_handles_connect_path(self, handler):
        assert handler.can_handle("/api/v1/advertising/connect")

    def test_handles_disconnect_path(self, handler):
        assert handler.can_handle("/api/v1/advertising/google_ads")

    def test_handles_campaigns_list(self, handler):
        assert handler.can_handle("/api/v1/advertising/campaigns")

    def test_handles_platform_campaigns(self, handler):
        assert handler.can_handle("/api/v1/advertising/google_ads/campaigns")

    def test_handles_platform_campaign_by_id(self, handler):
        assert handler.can_handle("/api/v1/advertising/google_ads/campaigns/camp-1")

    def test_handles_performance(self, handler):
        assert handler.can_handle("/api/v1/advertising/performance")

    def test_handles_platform_performance(self, handler):
        assert handler.can_handle("/api/v1/advertising/meta_ads/performance")

    def test_handles_analyze(self, handler):
        assert handler.can_handle("/api/v1/advertising/analyze")

    def test_handles_budget_recommendations(self, handler):
        assert handler.can_handle("/api/v1/advertising/budget-recommendations")

    def test_rejects_non_advertising_path(self, handler):
        assert not handler.can_handle("/api/v1/debates/123")

    def test_rejects_root_path(self, handler):
        assert not handler.can_handle("/api/v1/advertising")

    def test_rejects_unrelated_prefix(self, handler):
        assert not handler.can_handle("/api/v1/billing/plans")


# =============================================================================
# ROUTES attribute Tests
# =============================================================================


class TestRoutes:
    """Tests for ROUTES class attribute."""

    def test_routes_defined(self, handler):
        assert hasattr(handler, "ROUTES")
        assert len(handler.ROUTES) >= 10

    def test_routes_all_start_with_prefix(self, handler):
        for route in handler.ROUTES:
            assert route.startswith("/api/v1/advertising/")


# =============================================================================
# handle_request Routing Tests
# =============================================================================


class TestHandleRequestRouting:
    """Tests for handle_request() routing to correct sub-handlers."""

    @pytest.mark.asyncio
    async def test_routes_to_list_platforms(self, handler):
        req = MockRequest(method="GET", path="/api/v1/advertising/platforms")
        result = await handler.handle_request(req)
        assert _status(result) == 200
        assert "platforms" in _body(result)

    @pytest.mark.asyncio
    async def test_unknown_endpoint_returns_404(self, handler):
        req = MockRequest(method="GET", path="/api/v1/advertising/nonexistent-thing")
        result = await handler.handle_request(req)
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()


# =============================================================================
# GET /api/v1/advertising/platforms
# =============================================================================


class TestListPlatforms:
    """Tests for platform listing endpoint."""

    @pytest.mark.asyncio
    async def test_list_platforms_empty_connections(self, handler):
        req = MockRequest(method="GET", path="/api/v1/advertising/platforms")
        result = await handler.handle_request(req)
        body = _body(result)
        assert _status(result) == 200
        assert "platforms" in body
        assert body["connected_count"] == 0

    @pytest.mark.asyncio
    async def test_list_platforms_returns_all_supported(self, handler):
        req = MockRequest(method="GET", path="/api/v1/advertising/platforms")
        result = await handler.handle_request(req)
        body = _body(result)
        platform_ids = {p["id"] for p in body["platforms"]}
        assert "google_ads" in platform_ids
        assert "meta_ads" in platform_ids
        assert "linkedin_ads" in platform_ids
        assert "microsoft_ads" in platform_ids

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_list_platforms_shows_connected(self, handler):
        req = MockRequest(method="GET", path="/api/v1/advertising/platforms")
        result = await handler.handle_request(req)
        body = _body(result)
        assert body["connected_count"] == 1
        google = next(p for p in body["platforms"] if p["id"] == "google_ads")
        assert google["connected"] is True

    @pytest.mark.asyncio
    async def test_platform_entry_has_required_fields(self, handler):
        req = MockRequest(method="GET", path="/api/v1/advertising/platforms")
        result = await handler.handle_request(req)
        body = _body(result)
        for p in body["platforms"]:
            assert "id" in p
            assert "name" in p
            assert "description" in p
            assert "features" in p
            assert "connected" in p


# =============================================================================
# POST /api/v1/advertising/connect
# =============================================================================


class TestConnectPlatform:
    """Tests for platform connection endpoint."""

    @pytest.mark.asyncio
    async def test_connect_google_ads_success(self, handler):
        req = MockRequest(
            method="POST",
            path="/api/v1/advertising/connect",
            body={
                "platform": "google_ads",
                "credentials": {
                    "developer_token": "dev",
                    "client_id": "cid",
                    "client_secret": "csec",
                    "refresh_token": "rtok",
                    "customer_id": "123",
                },
            },
        )
        with patch.object(handler, "_get_connector", new=AsyncMock(return_value=None)):
            result = await handler.handle_request(req)
        assert _status(result) == 200
        assert "connected_at" in _body(result)

    @pytest.mark.asyncio
    async def test_connect_meta_ads_success(self, handler):
        req = MockRequest(
            method="POST",
            path="/api/v1/advertising/connect",
            body={
                "platform": "meta_ads",
                "credentials": {"access_token": "tok", "ad_account_id": "act_1"},
            },
        )
        with patch.object(handler, "_get_connector", new=AsyncMock(return_value=None)):
            result = await handler.handle_request(req)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_connect_missing_platform(self, handler):
        req = MockRequest(
            method="POST",
            path="/api/v1/advertising/connect",
            body={"credentials": {"token": "x"}},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "required" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_connect_unsupported_platform(self, handler):
        req = MockRequest(
            method="POST",
            path="/api/v1/advertising/connect",
            body={"platform": "tiktok_ads", "credentials": {"token": "x"}},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "unsupported" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_connect_missing_credentials(self, handler):
        req = MockRequest(
            method="POST",
            path="/api/v1/advertising/connect",
            body={"platform": "google_ads", "credentials": {}},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_connect_missing_required_credential_fields(self, handler):
        req = MockRequest(
            method="POST",
            path="/api/v1/advertising/connect",
            body={
                "platform": "google_ads",
                "credentials": {"developer_token": "dev"},  # missing others
            },
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "missing" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_connect_invalid_body(self, handler):
        req = MockRequest(method="POST", path="/api/v1/advertising/connect")
        result = await handler.handle_request(req)
        # Empty body should still parse to {} and miss platform
        assert _status(result) == 400


# =============================================================================
# DELETE /api/v1/advertising/{platform}
# =============================================================================


class TestDisconnectPlatform:
    """Tests for platform disconnection endpoint."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_disconnect_success(self, handler):
        req = MockRequest(method="DELETE", path="/api/v1/advertising/google_ads")
        result = await handler.handle_request(req)
        assert _status(result) == 200
        assert "disconnected" in _body(result).get("message", "").lower()

    @pytest.mark.asyncio
    async def test_disconnect_not_connected(self, handler):
        req = MockRequest(method="DELETE", path="/api/v1/advertising/google_ads")
        result = await handler.handle_request(req)
        assert _status(result) == 404
        assert "not connected" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_disconnect_closes_connector(self, handler):
        import aragora.server.handlers.features.advertising as adv_mod

        mock_conn = _mock_connector()
        adv_mod._platform_connectors["google_ads"] = mock_conn

        req = MockRequest(method="DELETE", path="/api/v1/advertising/google_ads")
        result = await handler.handle_request(req)
        assert _status(result) == 200
        mock_conn.close.assert_awaited_once()


# =============================================================================
# GET /api/v1/advertising/campaigns
# =============================================================================


class TestListAllCampaigns:
    """Tests for cross-platform campaign listing."""

    @pytest.mark.asyncio
    async def test_list_all_no_platforms_connected(self, handler):
        req = MockRequest(method="GET", path="/api/v1/advertising/campaigns")
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["campaigns"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_list_all_with_connected_platform(self, handler):
        mock_campaign = MagicMock()
        mock_campaign.id = "c1"
        mock_campaign.name = "Test Campaign"
        mock_campaign.status = "ENABLED"
        mock_campaign.campaign_type = "SEARCH"
        mock_campaign.budget_micros = 50_000_000
        mock_campaign.start_date = None
        mock_campaign.end_date = None
        mock_campaign.bidding_strategy = "MAXIMIZE_CLICKS"

        mock_conn = _mock_connector()
        mock_conn.get_campaigns = AsyncMock(return_value=[mock_campaign])

        with patch.object(handler, "_get_connector", new=AsyncMock(return_value=mock_conn)):
            req = MockRequest(method="GET", path="/api/v1/advertising/campaigns")
            result = await handler.handle_request(req)

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["campaigns"][0]["name"] == "Test Campaign"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_list_all_respects_limit(self, handler):
        req = MockRequest(
            method="GET",
            path="/api/v1/advertising/campaigns",
            query={"limit": "1"},
        )
        # Even if connector returns many, limit should cap it
        with patch.object(
            handler,
            "_fetch_platform_campaigns",
            new=AsyncMock(
                return_value=[
                    {"name": "A"},
                    {"name": "B"},
                    {"name": "C"},
                ]
            ),
        ):
            result = await handler.handle_request(req)
        body = _body(result)
        assert body["total"] == 1


# =============================================================================
# GET /api/v1/advertising/{platform}/campaigns
# =============================================================================


class TestListPlatformCampaigns:
    """Tests for platform-specific campaign listing."""

    @pytest.mark.asyncio
    async def test_platform_not_connected(self, handler):
        req = MockRequest(
            method="GET",
            path="/api/v1/advertising/google_ads/campaigns",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 404
        assert "not connected" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_platform_campaigns_success(self, handler):
        with patch.object(
            handler,
            "_fetch_platform_campaigns",
            new=AsyncMock(
                return_value=[
                    {"id": "c1", "name": "Campaign 1", "platform": "google_ads"},
                ]
            ),
        ):
            req = MockRequest(
                method="GET",
                path="/api/v1/advertising/google_ads/campaigns",
            )
            result = await handler.handle_request(req)

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["platform"] == "google_ads"


# =============================================================================
# GET /api/v1/advertising/{platform}/campaigns/{id}
# =============================================================================


class TestGetCampaign:
    """Tests for getting a specific campaign."""

    @pytest.mark.asyncio
    async def test_get_campaign_platform_not_connected(self, handler):
        req = MockRequest(
            method="GET",
            path="/api/v1/advertising/google_ads/campaigns/camp-1",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_get_campaign_no_connector(self, handler):
        with patch.object(handler, "_get_connector", new=AsyncMock(return_value=None)):
            req = MockRequest(
                method="GET",
                path="/api/v1/advertising/google_ads/campaigns/camp-1",
            )
            result = await handler.handle_request(req)
        assert _status(result) == 500

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_get_campaign_success_google(self, handler):
        mock_campaign = MagicMock()
        mock_campaign.id = "camp-1"
        mock_campaign.name = "My Campaign"
        mock_campaign.status = "ENABLED"
        mock_campaign.campaign_type = "SEARCH"
        mock_campaign.budget_micros = 10_000_000
        mock_campaign.start_date = None
        mock_campaign.end_date = None
        mock_campaign.bidding_strategy = "CPA"

        mock_conn = _mock_connector()
        mock_conn.get_campaign = AsyncMock(return_value=mock_campaign)

        with patch.object(handler, "_get_connector", new=AsyncMock(return_value=mock_conn)):
            req = MockRequest(
                method="GET",
                path="/api/v1/advertising/google_ads/campaigns/camp-1",
            )
            result = await handler.handle_request(req)

        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == "camp-1"
        assert body["platform"] == "google_ads"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_get_campaign_not_found(self, handler):
        mock_conn = _mock_connector()
        mock_conn.get_campaign = AsyncMock(side_effect=KeyError("not found"))

        with patch.object(handler, "_get_connector", new=AsyncMock(return_value=mock_conn)):
            req = MockRequest(
                method="GET",
                path="/api/v1/advertising/google_ads/campaigns/nonexistent",
            )
            result = await handler.handle_request(req)

        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()


# =============================================================================
# POST /api/v1/advertising/{platform}/campaigns
# =============================================================================


class TestCreateCampaign:
    """Tests for campaign creation endpoint."""

    @pytest.mark.asyncio
    async def test_create_platform_not_connected(self, handler):
        req = MockRequest(
            method="POST",
            path="/api/v1/advertising/google_ads/campaigns",
            body={"name": "Test"},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_create_google_campaign_success(self, handler):
        mock_conn = _mock_connector()
        mock_conn.create_campaign = AsyncMock(return_value="new-camp-1")

        with patch.object(handler, "_get_connector", new=AsyncMock(return_value=mock_conn)):
            req = MockRequest(
                method="POST",
                path="/api/v1/advertising/google_ads/campaigns",
                body={"name": "My Search Campaign", "daily_budget": 100.0},
            )
            result = await handler.handle_request(req)

        assert _status(result) == 201
        body = _body(result)
        assert body["campaign_id"] == "new-camp-1"
        assert body["platform"] == "google_ads"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_create_campaign_missing_name(self, handler):
        req = MockRequest(
            method="POST",
            path="/api/v1/advertising/google_ads/campaigns",
            body={"daily_budget": 100.0},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "name" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_create_campaign_empty_name(self, handler):
        req = MockRequest(
            method="POST",
            path="/api/v1/advertising/google_ads/campaigns",
            body={"name": "   "},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_create_campaign_name_too_long(self, handler):
        req = MockRequest(
            method="POST",
            path="/api/v1/advertising/google_ads/campaigns",
            body={"name": "A" * 300},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "256" in _body(result).get("error", "")

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_create_campaign_invalid_name_chars(self, handler):
        req = MockRequest(
            method="POST",
            path="/api/v1/advertising/google_ads/campaigns",
            body={"name": "Bad<script>"},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "invalid characters" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_create_campaign_negative_budget(self, handler):
        req = MockRequest(
            method="POST",
            path="/api/v1/advertising/google_ads/campaigns",
            body={"name": "Good Name", "daily_budget": -5.0},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "non-negative" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_create_campaign_budget_exceeds_max(self, handler):
        req = MockRequest(
            method="POST",
            path="/api/v1/advertising/google_ads/campaigns",
            body={"name": "Good Name", "daily_budget": 2_000_000.0},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "maximum" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_create_campaign_budget_not_a_number(self, handler):
        req = MockRequest(
            method="POST",
            path="/api/v1/advertising/google_ads/campaigns",
            body={"name": "Good Name", "daily_budget": "not-a-number"},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "number" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_create_campaign_invalid_campaign_type(self, handler):
        req = MockRequest(
            method="POST",
            path="/api/v1/advertising/google_ads/campaigns",
            body={"name": "Good Name", "campaign_type": "INVALID_TYPE"},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "invalid" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_create_campaign_valid_campaign_type(self, handler):
        mock_conn = _mock_connector()
        mock_conn.create_campaign = AsyncMock(return_value="c-1")

        with patch.object(handler, "_get_connector", new=AsyncMock(return_value=mock_conn)):
            req = MockRequest(
                method="POST",
                path="/api/v1/advertising/google_ads/campaigns",
                body={"name": "Display Campaign", "campaign_type": "DISPLAY"},
            )
            result = await handler.handle_request(req)

        assert _status(result) == 201

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_create_campaign_invalid_objective(self, handler):
        req = MockRequest(
            method="POST",
            path="/api/v1/advertising/google_ads/campaigns",
            body={"name": "Good Name", "objective": "NONEXISTENT"},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_create_campaign_invalid_date_range(self, handler):
        req = MockRequest(
            method="POST",
            path="/api/v1/advertising/google_ads/campaigns",
            body={
                "name": "Good Name",
                "start_date": "2026-03-01",
                "end_date": "2026-02-01",
            },
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "before" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_create_campaign_invalid_date_format(self, handler):
        req = MockRequest(
            method="POST",
            path="/api/v1/advertising/google_ads/campaigns",
            body={
                "name": "Good Name",
                "start_date": "not-a-date",
            },
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_create_campaign_connector_failure(self, handler):
        mock_conn = _mock_connector()
        mock_conn.create_campaign = AsyncMock(side_effect=ConnectionError("timeout"))

        with patch.object(handler, "_get_connector", new=AsyncMock(return_value=mock_conn)):
            req = MockRequest(
                method="POST",
                path="/api/v1/advertising/google_ads/campaigns",
                body={"name": "Good Name"},
            )
            result = await handler.handle_request(req)

        assert _status(result) == 500
        assert "failed" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_create_campaign_no_connector_available(self, handler):
        with patch.object(handler, "_get_connector", new=AsyncMock(return_value=None)):
            req = MockRequest(
                method="POST",
                path="/api/v1/advertising/google_ads/campaigns",
                body={"name": "Good Name"},
            )
            result = await handler.handle_request(req)
        assert _status(result) == 500

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_create_campaign_total_budget_exceeds_max(self, handler):
        req = MockRequest(
            method="POST",
            path="/api/v1/advertising/google_ads/campaigns",
            body={"name": "Good Name", "total_budget": 200_000_000.0},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_meta_creds")
    async def test_create_meta_campaign_success(self, handler):
        mock_campaign = MagicMock()
        mock_campaign.id = "meta-c1"
        mock_campaign.name = "Meta Campaign"
        mock_campaign.status = "ACTIVE"
        mock_campaign.objective = "OUTCOME_TRAFFIC"
        mock_campaign.daily_budget = 50.0
        mock_campaign.lifetime_budget = None
        mock_campaign.start_time = None
        mock_campaign.stop_time = None
        mock_campaign.spend_cap = None

        mock_conn = _mock_connector()
        mock_conn.create_campaign = AsyncMock(return_value=mock_campaign)

        with patch.object(handler, "_get_connector", new=AsyncMock(return_value=mock_conn)):
            req = MockRequest(
                method="POST",
                path="/api/v1/advertising/meta_ads/campaigns",
                body={"name": "Meta Campaign"},
            )
            result = await handler.handle_request(req)

        assert _status(result) == 201

    @pytest.mark.asyncio
    async def test_create_linkedin_requires_campaign_group_id(self, handler):
        import aragora.server.handlers.features.advertising as adv_mod

        adv_mod._platform_credentials["linkedin_ads"] = {
            "credentials": {"access_token": "tok", "ad_account_id": "acc_1"},
            "connected_at": "2026-01-01T00:00:00+00:00",
        }
        try:
            mock_conn = _mock_connector()
            with patch.object(handler, "_get_connector", new=AsyncMock(return_value=mock_conn)):
                req = MockRequest(
                    method="POST",
                    path="/api/v1/advertising/linkedin_ads/campaigns",
                    body={"name": "LinkedIn Campaign"},
                )
                result = await handler.handle_request(req)
            assert _status(result) == 400
            assert "campaign_group_id" in _body(result).get("error", "").lower()
        finally:
            adv_mod._platform_credentials.pop("linkedin_ads", None)


# =============================================================================
# PUT /api/v1/advertising/{platform}/campaigns/{id}
# =============================================================================


class TestUpdateCampaign:
    """Tests for campaign update endpoint."""

    @pytest.mark.asyncio
    async def test_update_platform_not_connected(self, handler):
        req = MockRequest(
            method="PUT",
            path="/api/v1/advertising/google_ads/campaigns/camp-1",
            body={"status": "PAUSED"},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_update_campaign_status_success(self, handler):
        mock_conn = _mock_connector()

        with patch.object(handler, "_get_connector", new=AsyncMock(return_value=mock_conn)):
            req = MockRequest(
                method="PUT",
                path="/api/v1/advertising/google_ads/campaigns/camp-1",
                body={"status": "PAUSED"},
            )
            result = await handler.handle_request(req)

        assert _status(result) == 200
        body = _body(result)
        assert body["campaign_id"] == "camp-1"
        mock_conn.update_campaign_status.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_update_campaign_invalid_status(self, handler):
        mock_conn = _mock_connector()

        with patch.object(handler, "_get_connector", new=AsyncMock(return_value=mock_conn)):
            req = MockRequest(
                method="PUT",
                path="/api/v1/advertising/google_ads/campaigns/camp-1",
                body={"status": "INVALID_STATUS"},
            )
            result = await handler.handle_request(req)

        assert _status(result) == 400

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_update_campaign_budget_success(self, handler):
        mock_conn = _mock_connector()

        with patch.object(handler, "_get_connector", new=AsyncMock(return_value=mock_conn)):
            req = MockRequest(
                method="PUT",
                path="/api/v1/advertising/google_ads/campaigns/camp-1",
                body={"daily_budget": 200.0},
            )
            result = await handler.handle_request(req)

        assert _status(result) == 200
        mock_conn.update_campaign_budget.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_update_campaign_invalid_budget(self, handler):
        mock_conn = _mock_connector()

        with patch.object(handler, "_get_connector", new=AsyncMock(return_value=mock_conn)):
            req = MockRequest(
                method="PUT",
                path="/api/v1/advertising/google_ads/campaigns/camp-1",
                body={"daily_budget": -10.0},
            )
            result = await handler.handle_request(req)

        assert _status(result) == 400

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_update_campaign_invalid_name(self, handler):
        mock_conn = _mock_connector()

        with patch.object(handler, "_get_connector", new=AsyncMock(return_value=mock_conn)):
            req = MockRequest(
                method="PUT",
                path="/api/v1/advertising/google_ads/campaigns/camp-1",
                body={"name": "<script>alert(1)</script>"},
            )
            result = await handler.handle_request(req)

        assert _status(result) == 400

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_update_campaign_connector_failure(self, handler):
        mock_conn = _mock_connector()
        mock_conn.update_campaign_status = AsyncMock(side_effect=TimeoutError("timed out"))

        with patch.object(handler, "_get_connector", new=AsyncMock(return_value=mock_conn)):
            req = MockRequest(
                method="PUT",
                path="/api/v1/advertising/google_ads/campaigns/camp-1",
                body={"status": "PAUSED"},
            )
            result = await handler.handle_request(req)

        assert _status(result) == 500
        assert "failed" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_update_campaign_no_connector(self, handler):
        with patch.object(handler, "_get_connector", new=AsyncMock(return_value=None)):
            req = MockRequest(
                method="PUT",
                path="/api/v1/advertising/google_ads/campaigns/camp-1",
                body={"status": "PAUSED"},
            )
            result = await handler.handle_request(req)

        assert _status(result) == 500


# =============================================================================
# GET /api/v1/advertising/performance
# =============================================================================


class TestCrossPlatformPerformance:
    """Tests for cross-platform performance endpoint."""

    @pytest.mark.asyncio
    async def test_performance_no_platforms(self, handler):
        req = MockRequest(
            method="GET",
            path="/api/v1/advertising/performance",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert "totals" in body
        assert "date_range" in body

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_performance_with_connected_platform(self, handler):
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
            new=AsyncMock(return_value=perf_data),
        ):
            req = MockRequest(
                method="GET",
                path="/api/v1/advertising/performance",
            )
            result = await handler.handle_request(req)

        assert _status(result) == 200
        body = _body(result)
        assert body["totals"]["impressions"] == 1000
        assert body["totals"]["clicks"] == 50

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_performance_custom_days_param(self, handler):
        with patch.object(
            handler,
            "_fetch_platform_performance",
            new=AsyncMock(
                return_value={
                    "platform": "google_ads",
                    "impressions": 0,
                    "clicks": 0,
                    "cost": 0,
                    "conversions": 0,
                    "conversion_value": 0,
                }
            ),
        ):
            req = MockRequest(
                method="GET",
                path="/api/v1/advertising/performance",
                query={"days": "7"},
            )
            result = await handler.handle_request(req)

        assert _status(result) == 200

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_performance_handles_fetch_exception(self, handler):
        """When a platform fetch raises, it is treated as an exception result."""
        with patch.object(
            handler,
            "_fetch_platform_performance",
            new=AsyncMock(side_effect=ConnectionError("boom")),
        ):
            req = MockRequest(
                method="GET",
                path="/api/v1/advertising/performance",
            )
            result = await handler.handle_request(req)

        # Should still succeed with empty totals
        assert _status(result) == 200


# =============================================================================
# GET /api/v1/advertising/{platform}/performance
# =============================================================================


class TestPlatformPerformance:
    """Tests for platform-specific performance endpoint."""

    @pytest.mark.asyncio
    async def test_platform_performance_not_connected(self, handler):
        req = MockRequest(
            method="GET",
            path="/api/v1/advertising/google_ads/performance",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_platform_performance_success(self, handler):
        perf_data = {
            "platform": "google_ads",
            "impressions": 500,
            "clicks": 25,
            "cost": 50.0,
        }

        with patch.object(
            handler,
            "_fetch_platform_performance",
            new=AsyncMock(return_value=perf_data),
        ):
            req = MockRequest(
                method="GET",
                path="/api/v1/advertising/google_ads/performance",
            )
            result = await handler.handle_request(req)

        assert _status(result) == 200
        body = _body(result)
        assert body["platform"] == "google_ads"


# =============================================================================
# POST /api/v1/advertising/analyze
# =============================================================================


class TestAnalyzePerformance:
    """Tests for multi-agent analysis endpoint."""

    @pytest.mark.asyncio
    async def test_analyze_success(self, handler):
        with patch.object(
            handler,
            "_fetch_platform_performance",
            new=AsyncMock(return_value={"roas": 2.5, "cpc": 1.5, "cost": 100}),
        ):
            req = MockRequest(
                method="POST",
                path="/api/v1/advertising/analyze",
                body={"type": "performance_review", "days": 30},
            )
            result = await handler.handle_request(req)

        assert _status(result) == 200
        body = _body(result)
        assert body["type"] == "performance_review"
        assert body["status"] == "completed"
        assert "analysis_id" in body

    @pytest.mark.asyncio
    async def test_analyze_invalid_type(self, handler):
        req = MockRequest(
            method="POST",
            path="/api/v1/advertising/analyze",
            body={"type": "invalid_type"},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_analyze_invalid_days(self, handler):
        req = MockRequest(
            method="POST",
            path="/api/v1/advertising/analyze",
            body={"type": "performance_review", "days": 0},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_analyze_days_exceeds_max(self, handler):
        req = MockRequest(
            method="POST",
            path="/api/v1/advertising/analyze",
            body={"type": "performance_review", "days": 800},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_analyze_days_not_integer(self, handler):
        req = MockRequest(
            method="POST",
            path="/api/v1/advertising/analyze",
            body={"type": "performance_review", "days": "abc"},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_analyze_invalid_platform_in_list(self, handler):
        req = MockRequest(
            method="POST",
            path="/api/v1/advertising/analyze",
            body={"platforms": ["google_ads", "tiktok_ads"]},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_analyze_platforms_not_a_list(self, handler):
        req = MockRequest(
            method="POST",
            path="/api/v1/advertising/analyze",
            body={"platforms": "google_ads"},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_analyze_budget_optimization(self, handler):
        with patch.object(
            handler,
            "_fetch_platform_performance",
            new=AsyncMock(return_value={"roas": 1.0, "cpc": 3.0, "cost": 50}),
        ):
            req = MockRequest(
                method="POST",
                path="/api/v1/advertising/analyze",
                body={"type": "budget_optimization"},
            )
            result = await handler.handle_request(req)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_analyze_audience_analysis(self, handler):
        req = MockRequest(
            method="POST",
            path="/api/v1/advertising/analyze",
            body={"type": "audience_analysis"},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_analyze_creative_review(self, handler):
        req = MockRequest(
            method="POST",
            path="/api/v1/advertising/analyze",
            body={"type": "creative_review"},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200


# =============================================================================
# GET /api/v1/advertising/budget-recommendations
# =============================================================================


class TestBudgetRecommendations:
    """Tests for budget recommendations endpoint."""

    @pytest.mark.asyncio
    async def test_budget_recommendations_default(self, handler):
        req = MockRequest(
            method="GET",
            path="/api/v1/advertising/budget-recommendations",
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert "total_budget" in body
        assert "objective" in body
        assert "recommendations" in body
        assert "rationale" in body

    @pytest.mark.asyncio
    async def test_budget_recommendations_custom_budget(self, handler):
        req = MockRequest(
            method="GET",
            path="/api/v1/advertising/budget-recommendations",
            query={"budget": "50000"},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["total_budget"] == 50000.0

    @pytest.mark.asyncio
    async def test_budget_recommendations_conversions_objective(self, handler):
        req = MockRequest(
            method="GET",
            path="/api/v1/advertising/budget-recommendations",
            query={"objective": "conversions"},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["objective"] == "conversions"
        assert "roas" in body["rationale"].lower()

    @pytest.mark.asyncio
    async def test_budget_recommendations_awareness_objective(self, handler):
        req = MockRequest(
            method="GET",
            path="/api/v1/advertising/budget-recommendations",
            query={"objective": "awareness"},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["objective"] == "awareness"

    @pytest.mark.asyncio
    async def test_budget_recommendations_invalid_objective(self, handler):
        req = MockRequest(
            method="GET",
            path="/api/v1/advertising/budget-recommendations",
            query={"objective": "invalid"},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_budget_recommendations_negative_budget(self, handler):
        req = MockRequest(
            method="GET",
            path="/api/v1/advertising/budget-recommendations",
            query={"budget": "-100"},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds", "_seed_meta_creds")
    async def test_budget_recommendations_with_platforms(self, handler):
        perf_google = {"platform": "google_ads", "roas": 3.0, "cost": 100}
        perf_meta = {"platform": "meta_ads", "roas": 2.0, "cost": 80}

        call_count = 0

        async def mock_fetch(platform, start, end):
            nonlocal call_count
            call_count += 1
            if platform == "google_ads":
                return perf_google
            return perf_meta

        with patch.object(handler, "_fetch_platform_performance", side_effect=mock_fetch):
            req = MockRequest(
                method="GET",
                path="/api/v1/advertising/budget-recommendations",
                query={"budget": "10000"},
            )
            result = await handler.handle_request(req)

        assert _status(result) == 200
        body = _body(result)
        assert len(body["recommendations"]) == 2


# =============================================================================
# Path Parameter Extraction Tests
# =============================================================================


class TestPathParameterExtraction:
    """Tests for extracting platform and campaign_id from paths."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_extracts_platform_from_campaigns_path(self, handler):
        with patch.object(handler, "_fetch_platform_campaigns", new=AsyncMock(return_value=[])):
            req = MockRequest(
                method="GET",
                path="/api/v1/advertising/google_ads/campaigns",
            )
            result = await handler.handle_request(req)
        assert _status(result) == 200
        assert _body(result)["platform"] == "google_ads"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_seed_google_creds")
    async def test_extracts_campaign_id_from_path(self, handler):
        mock_campaign = MagicMock()
        mock_campaign.id = "the-id"
        mock_campaign.name = "C"
        mock_campaign.status = "ENABLED"
        mock_campaign.campaign_type = "SEARCH"
        mock_campaign.budget_micros = 1_000_000
        mock_campaign.start_date = None
        mock_campaign.end_date = None
        mock_campaign.bidding_strategy = "CPA"

        mock_conn = _mock_connector()
        mock_conn.get_campaign = AsyncMock(return_value=mock_campaign)

        with patch.object(handler, "_get_connector", new=AsyncMock(return_value=mock_conn)):
            req = MockRequest(
                method="GET",
                path="/api/v1/advertising/google_ads/campaigns/the-id",
            )
            result = await handler.handle_request(req)

        assert _status(result) == 200
        assert _body(result)["id"] == "the-id"


# =============================================================================
# Validation Helper Unit Tests
# =============================================================================


class TestValidationHelpers:
    """Direct tests for validation helper functions."""

    def test_validate_budget_none(self):
        from aragora.server.handlers.features.advertising import _validate_budget

        val, err = _validate_budget(None, "budget", 100)
        assert val is None
        assert err is None

    def test_validate_budget_valid(self):
        from aragora.server.handlers.features.advertising import _validate_budget

        val, err = _validate_budget(50.0, "budget", 100)
        assert val == 50.0
        assert err is None

    def test_validate_budget_negative(self):
        from aragora.server.handlers.features.advertising import _validate_budget

        val, err = _validate_budget(-1, "budget", 100)
        assert val is None
        assert "non-negative" in err

    def test_validate_budget_exceeds_max(self):
        from aragora.server.handlers.features.advertising import _validate_budget

        val, err = _validate_budget(200, "budget", 100)
        assert val is None
        assert "maximum" in err

    def test_validate_budget_not_a_number(self):
        from aragora.server.handlers.features.advertising import _validate_budget

        val, err = _validate_budget("xyz", "budget", 100)
        assert val is None
        assert "number" in err

    def test_validate_campaign_name_valid(self):
        from aragora.server.handlers.features.advertising import _validate_campaign_name

        val, err = _validate_campaign_name("My Campaign 2026")
        assert val == "My Campaign 2026"
        assert err is None

    def test_validate_campaign_name_empty(self):
        from aragora.server.handlers.features.advertising import _validate_campaign_name

        val, err = _validate_campaign_name("")
        assert val is None
        assert err is not None

    def test_validate_campaign_name_not_string(self):
        from aragora.server.handlers.features.advertising import _validate_campaign_name

        val, err = _validate_campaign_name(12345)
        assert val is None

    def test_validate_campaign_name_too_long(self):
        from aragora.server.handlers.features.advertising import _validate_campaign_name

        val, err = _validate_campaign_name("X" * 300)
        assert val is None
        assert "256" in err

    def test_validate_campaign_name_invalid_chars(self):
        from aragora.server.handlers.features.advertising import _validate_campaign_name

        val, err = _validate_campaign_name("Test<>Campaign")
        assert val is None
        assert "invalid characters" in err.lower()

    def test_validate_campaign_name_strips_whitespace(self):
        from aragora.server.handlers.features.advertising import _validate_campaign_name

        val, err = _validate_campaign_name("  My Campaign  ")
        assert val == "My Campaign"
        assert err is None

    def test_validate_date_iso_valid(self):
        from aragora.server.handlers.features.advertising import _validate_date_iso
        from datetime import date

        val, err = _validate_date_iso("2026-03-15", "start")
        assert val == date(2026, 3, 15)
        assert err is None

    def test_validate_date_iso_none(self):
        from aragora.server.handlers.features.advertising import _validate_date_iso

        val, err = _validate_date_iso(None, "start")
        assert val is None
        assert err is None

    def test_validate_date_iso_invalid_format(self):
        from aragora.server.handlers.features.advertising import _validate_date_iso

        val, err = _validate_date_iso("15/03/2026", "start")
        assert val is None
        assert err is not None

    def test_validate_date_iso_not_string(self):
        from aragora.server.handlers.features.advertising import _validate_date_iso

        val, err = _validate_date_iso(20260315, "start")
        assert val is None
        assert "string" in err.lower()

    def test_validate_date_range_valid(self):
        from aragora.server.handlers.features.advertising import _validate_date_range
        from datetime import date

        val, err = _validate_date_range("2026-01-01", "2026-03-01")
        assert val == (date(2026, 1, 1), date(2026, 3, 1))
        assert err is None

    def test_validate_date_range_inverted(self):
        from aragora.server.handlers.features.advertising import _validate_date_range

        val, err = _validate_date_range("2026-06-01", "2026-01-01")
        assert val is None
        assert "before" in err.lower()

    def test_validate_against_allowlist_valid(self):
        from aragora.server.handlers.features.advertising import _validate_against_allowlist

        err = _validate_against_allowlist("SEARCH", {"SEARCH", "DISPLAY"}, "type")
        assert err is None

    def test_validate_against_allowlist_invalid(self):
        from aragora.server.handlers.features.advertising import _validate_against_allowlist

        err = _validate_against_allowlist("BAD", {"SEARCH", "DISPLAY"}, "type")
        assert err is not None
        assert "invalid" in err.lower()

    def test_validate_against_allowlist_none(self):
        from aragora.server.handlers.features.advertising import _validate_against_allowlist

        err = _validate_against_allowlist(None, {"A", "B"}, "type")
        assert err is None

    def test_validate_against_allowlist_not_string(self):
        from aragora.server.handlers.features.advertising import _validate_against_allowlist

        err = _validate_against_allowlist(123, {"A"}, "type")
        assert "string" in err.lower()


# =============================================================================
# Normalizer Method Tests
# =============================================================================


class TestNormalizeMethods:
    """Tests for campaign normalization methods."""

    def test_normalize_google_campaign(self, handler):
        campaign = MagicMock()
        campaign.id = "g1"
        campaign.name = "Google Campaign"
        campaign.status = "ENABLED"
        campaign.campaign_type = "SEARCH"
        campaign.budget_micros = 50_000_000
        campaign.start_date = None
        campaign.end_date = None
        campaign.bidding_strategy = "CPA"

        result = handler._normalize_google_campaign(campaign)
        assert result["id"] == "g1"
        assert result["platform"] == "google_ads"
        assert result["daily_budget"] == 50.0

    def test_normalize_google_campaign_enum_values(self, handler):
        campaign = MagicMock()
        campaign.id = "g2"
        campaign.name = "Enum Campaign"
        campaign.status.value = "ENABLED"
        campaign.campaign_type.value = "DISPLAY"
        campaign.budget_micros = None
        campaign.start_date = None
        campaign.end_date = None
        campaign.bidding_strategy.value = "MAXIMIZE_CLICKS"

        result = handler._normalize_google_campaign(campaign)
        assert result["status"] == "ENABLED"
        assert result["objective"] == "DISPLAY"
        assert result["daily_budget"] is None

    def test_normalize_meta_campaign(self, handler):
        campaign = MagicMock()
        campaign.id = "m1"
        campaign.name = "Meta Campaign"
        campaign.status = "ACTIVE"
        campaign.objective = "OUTCOME_TRAFFIC"
        campaign.daily_budget = 25.0
        campaign.lifetime_budget = 1000.0
        campaign.start_time = None
        campaign.stop_time = None
        campaign.spend_cap = 500.0

        result = handler._normalize_meta_campaign(campaign)
        assert result["id"] == "m1"
        assert result["platform"] == "meta_ads"
        assert result["total_budget"] == 1000.0

    def test_normalize_linkedin_campaign(self, handler):
        campaign = MagicMock()
        campaign.id = "l1"
        campaign.name = "LinkedIn Campaign"
        campaign.status = "ACTIVE"
        campaign.objective_type = "WEBSITE_VISITS"
        campaign.daily_budget = 50.0
        campaign.total_budget = 5000.0
        campaign.run_schedule_start = None
        campaign.run_schedule_end = None
        campaign.campaign_type = "SPONSORED_UPDATES"

        result = handler._normalize_linkedin_campaign(campaign)
        assert result["id"] == "l1"
        assert result["platform"] == "linkedin_ads"

    def test_normalize_microsoft_campaign(self, handler):
        campaign = MagicMock()
        campaign.id = "ms1"
        campaign.name = "MS Campaign"
        campaign.status = "ENABLED"
        campaign.campaign_type = "Search"
        campaign.daily_budget = 75.0
        campaign.start_date = None
        campaign.end_date = None
        campaign.bidding_scheme = "EnhancedCpc"

        result = handler._normalize_microsoft_campaign(campaign)
        assert result["id"] == "ms1"
        assert result["platform"] == "microsoft_ads"


# =============================================================================
# Summary / Recommendations / Insights Generation
# =============================================================================


class TestAnalysisGenerators:
    """Tests for the internal analysis generation methods."""

    def test_generate_performance_summary(self, handler):
        perf_data = {
            "google_ads": {"cost": 100, "conversions": 10, "roas": 3.0},
            "meta_ads": {"cost": 50, "conversions": 5, "roas": 2.0},
        }
        summary = handler._generate_performance_summary(perf_data)
        assert summary["total_spend"] == 150.0
        assert summary["total_conversions"] == 15
        assert summary["platforms_analyzed"] == 2

    def test_generate_recommendations_high_roas(self, handler):
        perf_data = {"google_ads": {"roas": 5.0, "cpc": 1.0}}
        recs = handler._generate_recommendations(perf_data)
        assert any(r["type"] == "increase_budget" for r in recs)

    def test_generate_recommendations_low_roas(self, handler):
        perf_data = {"google_ads": {"roas": 0.5, "cpc": 1.0}}
        recs = handler._generate_recommendations(perf_data)
        assert any(r["type"] == "optimize" for r in recs)

    def test_generate_recommendations_high_cpc(self, handler):
        perf_data = {"google_ads": {"roas": 2.0, "cpc": 10.0}}
        recs = handler._generate_recommendations(perf_data)
        assert any(r["type"] == "reduce_cpc" for r in recs)

    def test_generate_insights_spend_share(self, handler):
        perf_data = {
            "google_ads": {"cost": 60},
            "meta_ads": {"cost": 40},
        }
        insights = handler._generate_insights(perf_data)
        assert len(insights) == 2
        assert "google_ads" in insights[0]

    def test_generate_insights_no_spend(self, handler):
        perf_data = {
            "google_ads": {"cost": 0},
        }
        insights = handler._generate_insights(perf_data)
        assert insights == []

    def test_budget_rationale_conversions(self, handler):
        rationale = handler._generate_budget_rationale({}, "conversions")
        assert "roas" in rationale.lower()

    def test_budget_rationale_awareness(self, handler):
        rationale = handler._generate_budget_rationale({}, "awareness")
        assert "reach" in rationale.lower()

    def test_budget_rationale_balanced(self, handler):
        rationale = handler._generate_budget_rationale({}, "balanced")
        assert "balanced" in rationale.lower()


# =============================================================================
# Budget Recommendation Calculation
# =============================================================================


class TestBudgetCalculation:
    """Tests for budget allocation calculation."""

    def test_calculate_budget_roas_weighted(self, handler):
        perf_data = {
            "google_ads": {"roas": 4.0},
            "meta_ads": {"roas": 1.0},
        }
        recs = handler._calculate_budget_recommendations(perf_data, 10000, "conversions")
        assert len(recs) == 2
        google_rec = next(r for r in recs if r["platform"] == "google_ads")
        meta_rec = next(r for r in recs if r["platform"] == "meta_ads")
        assert google_rec["recommended_budget"] > meta_rec["recommended_budget"]

    def test_calculate_budget_awareness_even_split(self, handler):
        perf_data = {
            "google_ads": {"roas": 4.0},
            "meta_ads": {"roas": 1.0},
        }
        recs = handler._calculate_budget_recommendations(perf_data, 10000, "awareness")
        google_rec = next(r for r in recs if r["platform"] == "google_ads")
        meta_rec = next(r for r in recs if r["platform"] == "meta_ads")
        assert google_rec["recommended_budget"] == meta_rec["recommended_budget"]

    def test_calculate_budget_balanced(self, handler):
        perf_data = {
            "google_ads": {"roas": 3.0},
        }
        recs = handler._calculate_budget_recommendations(perf_data, 5000, "balanced")
        assert len(recs) == 1
        assert recs[0]["recommended_budget"] == 5000.0


# =============================================================================
# Required Credentials Tests
# =============================================================================


class TestRequiredCredentials:
    """Tests for _get_required_credentials."""

    def test_google_ads_credentials(self, handler):
        creds = handler._get_required_credentials("google_ads")
        assert "developer_token" in creds
        assert "client_id" in creds
        assert "customer_id" in creds
        assert len(creds) == 5

    def test_meta_ads_credentials(self, handler):
        creds = handler._get_required_credentials("meta_ads")
        assert "access_token" in creds
        assert "ad_account_id" in creds

    def test_linkedin_ads_credentials(self, handler):
        creds = handler._get_required_credentials("linkedin_ads")
        assert "access_token" in creds

    def test_microsoft_ads_credentials(self, handler):
        creds = handler._get_required_credentials("microsoft_ads")
        assert "developer_token" in creds
        assert len(creds) == 6

    def test_unknown_platform_returns_empty(self, handler):
        creds = handler._get_required_credentials("tiktok_ads")
        assert creds == []


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Tests for circuit breaker integration."""

    def test_get_circuit_breaker(self):
        from aragora.server.handlers.features.advertising import (
            get_advertising_circuit_breaker,
        )

        cb = get_advertising_circuit_breaker()
        assert cb is not None
        assert cb.name == "advertising_handler"


# =============================================================================
# UnifiedCampaign / UnifiedPerformance Dataclass Tests
# =============================================================================


class TestDataclasses:
    """Tests for dataclass serialization."""

    def test_unified_campaign_to_dict(self):
        from aragora.server.handlers.features.advertising import UnifiedCampaign
        from datetime import date, datetime, timezone

        campaign = UnifiedCampaign(
            id="c1",
            platform="google_ads",
            name="Test",
            status="ENABLED",
            objective="SEARCH",
            daily_budget=100.0,
            total_budget=None,
            start_date=date(2026, 1, 1),
            end_date=date(2026, 12, 31),
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            metrics={"impressions": 100},
        )
        d = campaign.to_dict()
        assert d["id"] == "c1"
        assert d["start_date"] == "2026-01-01"
        assert d["end_date"] == "2026-12-31"
        assert d["metrics"]["impressions"] == 100

    def test_unified_campaign_none_dates(self):
        from aragora.server.handlers.features.advertising import UnifiedCampaign

        campaign = UnifiedCampaign(
            id="c2",
            platform="meta_ads",
            name="No Dates",
            status="ACTIVE",
            objective=None,
            daily_budget=None,
            total_budget=None,
            start_date=None,
            end_date=None,
            created_at=None,
        )
        d = campaign.to_dict()
        assert d["start_date"] is None
        assert d["end_date"] is None
        assert d["created_at"] is None

    def test_unified_performance_to_dict(self):
        from aragora.server.handlers.features.advertising import UnifiedPerformance
        from datetime import date

        perf = UnifiedPerformance(
            platform="google_ads",
            campaign_id="c1",
            campaign_name="Test",
            date_range=(date(2026, 1, 1), date(2026, 1, 31)),
            impressions=10000,
            clicks=500,
            cost=250.0,
            conversions=50,
            conversion_value=2500.0,
            ctr=5.0,
            cpc=0.50,
            cpm=25.0,
            roas=10.0,
        )
        d = perf.to_dict()
        assert d["impressions"] == 10000
        assert d["date_range"]["start"] == "2026-01-01"
        assert d["cost"] == 250.0
        assert d["roas"] == 10.0


# =============================================================================
# Error Response Helpers
# =============================================================================


class TestResponseHelpers:
    """Tests for _json_response and _error_response."""

    def test_json_response_format(self, handler):
        result = handler._json_response(200, {"key": "value"})
        assert result["status_code"] == 200
        assert result["headers"]["Content-Type"] == "application/json"
        assert result["body"] == {"key": "value"}

    def test_error_response_format(self, handler):
        result = handler._error_response(400, "Bad request")
        assert result["status_code"] == 400
        assert result["body"]["error"] == "Bad request"

    def test_error_response_404(self, handler):
        result = handler._error_response(404, "Not found")
        assert result["status_code"] == 404

    def test_error_response_500(self, handler):
        result = handler._error_response(500, "Server error")
        assert result["status_code"] == 500
