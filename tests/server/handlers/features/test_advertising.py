"""
Tests for Advertising Platform Handler.

Comprehensive test suite covering:
- Platform configuration and validation
- Dataclass creation and serialization
- Handler routing and request handling
- Circuit breaker integration
- Rate limiting
- Cross-platform campaign management
- Performance metrics aggregation
- Budget recommendations
- Error handling and edge cases
"""

from __future__ import annotations

import pytest
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from aragora.server.handlers.features.advertising import (
    AdvertisingHandler,
    SUPPORTED_PLATFORMS,
    UnifiedCampaign,
    UnifiedPerformance,
    get_advertising_circuit_breaker,
    _platform_credentials,
    _platform_connectors,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def handler():
    """Create a handler instance for testing."""
    return AdvertisingHandler(server_context={})


@pytest.fixture
def mock_request():
    """Create a mock request object."""
    request = MagicMock()
    request.method = "GET"
    request.path = "/api/v1/advertising/platforms"
    request.query = {}
    return request


@pytest.fixture(autouse=True)
def clean_platform_state():
    """Clean platform state before each test."""
    _platform_credentials.clear()
    _platform_connectors.clear()
    yield
    _platform_credentials.clear()
    _platform_connectors.clear()


@pytest.fixture
def connected_platform():
    """Set up a connected platform for testing."""
    _platform_credentials["google_ads"] = {
        "credentials": {
            "developer_token": "test_token",
            "client_id": "test_client_id",
            "client_secret": "test_secret",
            "refresh_token": "test_refresh",
            "customer_id": "123456",
        },
        "connected_at": datetime.now(timezone.utc).isoformat(),
    }
    return "google_ads"


# =============================================================================
# Platform Configuration Tests
# =============================================================================


class TestSupportedPlatforms:
    """Tests for platform configuration."""

    def test_all_platforms_defined(self):
        """Test that all supported platforms are configured."""
        expected = ["google_ads", "meta_ads", "linkedin_ads", "microsoft_ads"]
        for platform in expected:
            assert platform in SUPPORTED_PLATFORMS

    def test_platform_has_required_fields(self):
        """Test that all platforms have required configuration."""
        for platform_id, config in SUPPORTED_PLATFORMS.items():
            assert "name" in config, f"{platform_id} missing name"
            assert "description" in config, f"{platform_id} missing description"
            assert "features" in config, f"{platform_id} missing features"
            assert isinstance(config["features"], list)

    def test_google_ads_features(self):
        """Test Google Ads has expected features."""
        config = SUPPORTED_PLATFORMS["google_ads"]
        assert "campaigns" in config["features"]
        assert "performance" in config["features"]

    def test_meta_ads_features(self):
        """Test Meta Ads has expected features."""
        config = SUPPORTED_PLATFORMS["meta_ads"]
        assert "campaigns" in config["features"]
        assert "audiences" in config["features"]

    def test_linkedin_ads_features(self):
        """Test LinkedIn Ads has expected features."""
        config = SUPPORTED_PLATFORMS["linkedin_ads"]
        assert "campaigns" in config["features"]
        assert "lead_gen" in config["features"]

    def test_microsoft_ads_features(self):
        """Test Microsoft Ads has expected features."""
        config = SUPPORTED_PLATFORMS["microsoft_ads"]
        assert "campaigns" in config["features"]
        assert "keywords" in config["features"]

    def test_platform_count(self):
        """Test expected number of platforms."""
        assert len(SUPPORTED_PLATFORMS) == 4


# =============================================================================
# UnifiedCampaign Tests
# =============================================================================


class TestUnifiedCampaign:
    """Tests for UnifiedCampaign dataclass."""

    def test_campaign_creation(self):
        """Test creating a unified campaign."""
        campaign = UnifiedCampaign(
            id="camp_123",
            platform="google_ads",
            name="Test Campaign",
            status="active",
            objective="conversions",
            daily_budget=100.0,
            total_budget=3000.0,
            start_date=date.today(),
            end_date=date.today() + timedelta(days=30),
            created_at=datetime.now(),
        )

        assert campaign.id == "camp_123"
        assert campaign.platform == "google_ads"
        assert campaign.status == "active"
        assert campaign.daily_budget == 100.0

    def test_campaign_defaults(self):
        """Test campaign with minimal fields."""
        campaign = UnifiedCampaign(
            id="camp_456",
            platform="meta_ads",
            name="Meta Campaign",
            status="paused",
            objective=None,
            daily_budget=None,
            total_budget=None,
            start_date=None,
            end_date=None,
            created_at=None,
        )

        assert campaign.id == "camp_456"
        assert campaign.objective is None

    def test_campaign_to_dict(self):
        """Test campaign serialization to dictionary."""
        start = date(2024, 1, 1)
        end = date(2024, 1, 31)
        created = datetime(2024, 1, 1, 12, 0, 0)

        campaign = UnifiedCampaign(
            id="camp_789",
            platform="linkedin_ads",
            name="LinkedIn Campaign",
            status="active",
            objective="brand_awareness",
            daily_budget=50.0,
            total_budget=1500.0,
            start_date=start,
            end_date=end,
            created_at=created,
        )

        result = campaign.to_dict()

        assert result["id"] == "camp_789"
        assert result["platform"] == "linkedin_ads"
        assert result["start_date"] == "2024-01-01"
        assert result["end_date"] == "2024-01-31"
        assert result["created_at"] == created.isoformat()

    def test_campaign_to_dict_with_none_dates(self):
        """Test campaign serialization with None dates."""
        campaign = UnifiedCampaign(
            id="camp_abc",
            platform="microsoft_ads",
            name="MS Campaign",
            status="draft",
            objective=None,
            daily_budget=None,
            total_budget=None,
            start_date=None,
            end_date=None,
            created_at=None,
        )

        result = campaign.to_dict()

        assert result["start_date"] is None
        assert result["end_date"] is None
        assert result["created_at"] is None

    def test_campaign_with_metrics(self):
        """Test campaign with metrics data."""
        campaign = UnifiedCampaign(
            id="camp_metrics",
            platform="google_ads",
            name="Metrics Campaign",
            status="active",
            objective="conversions",
            daily_budget=100.0,
            total_budget=3000.0,
            start_date=date.today(),
            end_date=date.today() + timedelta(days=30),
            created_at=datetime.now(),
            metrics={"impressions": 1000, "clicks": 50, "ctr": 0.05},
        )

        assert campaign.metrics is not None
        assert campaign.metrics["impressions"] == 1000


# =============================================================================
# UnifiedPerformance Tests
# =============================================================================


class TestUnifiedPerformance:
    """Tests for UnifiedPerformance dataclass."""

    def test_performance_creation(self):
        """Test creating performance metrics."""
        perf = UnifiedPerformance(
            platform="google_ads",
            campaign_id="camp_123",
            campaign_name="Test Campaign",
            date_range=(date(2024, 1, 1), date(2024, 1, 31)),
            impressions=100000,
            clicks=2500,
            cost=5000.0,
            conversions=125,
            conversion_value=15000.0,
            ctr=0.025,
            cpc=2.0,
            cpm=50.0,
            roas=3.0,
        )

        assert perf.platform == "google_ads"
        assert perf.impressions == 100000
        assert perf.roas == 3.0

    def test_performance_to_dict(self):
        """Test performance serialization to dictionary."""
        perf = UnifiedPerformance(
            platform="meta_ads",
            campaign_id="camp_456",
            campaign_name="Meta Campaign",
            date_range=(date(2024, 1, 1), date(2024, 1, 31)),
            impressions=50000,
            clicks=1000,
            cost=2500.0,
            conversions=50,
            conversion_value=7500.0,
            ctr=0.02,
            cpc=2.5,
            cpm=50.0,
            roas=3.0,
        )

        result = perf.to_dict()

        assert result["platform"] == "meta_ads"
        assert result["date_range"]["start"] == "2024-01-01"
        assert result["date_range"]["end"] == "2024-01-31"
        assert result["cost"] == 2500.0
        assert result["roas"] == 3.0

    def test_performance_without_campaign(self):
        """Test performance metrics at account level."""
        perf = UnifiedPerformance(
            platform="linkedin_ads",
            campaign_id=None,
            campaign_name=None,
            date_range=(date(2024, 1, 1), date(2024, 1, 31)),
            impressions=25000,
            clicks=500,
            cost=1250.0,
            conversions=25,
            conversion_value=5000.0,
            ctr=0.02,
            cpc=2.5,
            cpm=50.0,
            roas=4.0,
        )

        result = perf.to_dict()

        assert result["campaign_id"] is None
        assert result["campaign_name"] is None

    def test_performance_rounding(self):
        """Test that values are properly rounded in to_dict."""
        perf = UnifiedPerformance(
            platform="google_ads",
            campaign_id="camp_round",
            campaign_name="Rounding Test",
            date_range=(date(2024, 1, 1), date(2024, 1, 31)),
            impressions=100000,
            clicks=2500,
            cost=5000.123456,
            conversions=125,
            conversion_value=15000.987654,
            ctr=0.0250123,
            cpc=2.001234,
            cpm=50.005678,
            roas=3.001234,
        )

        result = perf.to_dict()

        assert result["cost"] == 5000.12
        assert result["conversion_value"] == 15000.99
        assert result["ctr"] == 0.0250
        assert result["cpc"] == 2.0


# =============================================================================
# Handler Tests
# =============================================================================


class TestAdvertisingHandler:
    """Tests for AdvertisingHandler class."""

    def test_handler_creation(self):
        """Test creating handler instance."""
        handler = AdvertisingHandler(server_context={})
        assert handler is not None

    def test_handler_creation_with_ctx(self):
        """Test creating handler with ctx parameter."""
        handler = AdvertisingHandler(ctx={"key": "value"})
        assert handler.ctx == {"key": "value"}

    def test_handler_has_routes(self):
        """Test that handler has route definitions."""
        handler = AdvertisingHandler(server_context={})
        assert hasattr(handler, "handle_request")
        assert hasattr(handler, "ROUTES")

    def test_can_handle_advertising_routes(self, handler):
        """Test that handler recognizes advertising routes."""
        assert handler.can_handle("/api/v1/advertising/platforms")
        assert handler.can_handle("/api/v1/advertising/connect")
        assert handler.can_handle("/api/v1/advertising/campaigns")
        assert handler.can_handle("/api/v1/advertising/google_ads/campaigns")

    def test_cannot_handle_other_routes(self, handler):
        """Test that handler rejects non-advertising routes."""
        assert not handler.can_handle("/api/v1/analytics/platforms")
        assert not handler.can_handle("/api/v1/debates")
        assert not handler.can_handle("/api/v1/other")

    def test_routes_list(self, handler):
        """Test that ROUTES contains expected endpoints."""
        routes = handler.ROUTES
        assert "/api/v1/advertising/platforms" in routes
        assert "/api/v1/advertising/connect" in routes
        assert "/api/v1/advertising/campaigns" in routes
        assert "/api/v1/advertising/performance" in routes
        assert "/api/v1/advertising/analyze" in routes


class TestAdvertisingHandlerListPlatforms:
    """Tests for listing platforms endpoint."""

    @pytest.mark.asyncio
    async def test_list_platforms_success(self, handler, mock_request):
        """Test successful platform listing."""
        mock_request.path = "/api/v1/advertising/platforms"
        mock_request.method = "GET"

        result = await handler.handle_request(mock_request)

        assert result["status_code"] == 200
        assert "platforms" in result["body"]
        assert len(result["body"]["platforms"]) == 4

    @pytest.mark.asyncio
    async def test_list_platforms_shows_connection_status(
        self, handler, mock_request, connected_platform
    ):
        """Test that platforms show correct connection status."""
        mock_request.path = "/api/v1/advertising/platforms"
        mock_request.method = "GET"

        result = await handler.handle_request(mock_request)

        platforms = result["body"]["platforms"]
        google_platform = next(p for p in platforms if p["id"] == "google_ads")
        meta_platform = next(p for p in platforms if p["id"] == "meta_ads")

        assert google_platform["connected"] is True
        assert meta_platform["connected"] is False

    @pytest.mark.asyncio
    async def test_list_platforms_connected_count(self, handler, mock_request, connected_platform):
        """Test connected_count in response."""
        mock_request.path = "/api/v1/advertising/platforms"
        mock_request.method = "GET"

        result = await handler.handle_request(mock_request)

        assert result["body"]["connected_count"] == 1


class TestAdvertisingHandlerConnect:
    """Tests for platform connection endpoint."""

    @pytest.mark.asyncio
    async def test_connect_missing_platform(self, handler, mock_request):
        """Test connect with missing platform."""
        mock_request.path = "/api/v1/advertising/connect"
        mock_request.method = "POST"

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {}
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
                mock_auth.return_value = MagicMock(user_id="test")
                with patch.object(handler, "check_permission"):
                    result = await handler.handle_request(mock_request)

        assert result["status_code"] == 400
        assert "Platform is required" in result["body"]["error"]

    @pytest.mark.asyncio
    async def test_connect_unsupported_platform(self, handler, mock_request):
        """Test connect with unsupported platform."""
        mock_request.path = "/api/v1/advertising/connect"
        mock_request.method = "POST"

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"platform": "unsupported_platform"}
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
                mock_auth.return_value = MagicMock(user_id="test")
                with patch.object(handler, "check_permission"):
                    result = await handler.handle_request(mock_request)

        assert result["status_code"] == 400
        assert "Unsupported platform" in result["body"]["error"]

    @pytest.mark.asyncio
    async def test_connect_missing_credentials(self, handler, mock_request):
        """Test connect with missing credentials."""
        mock_request.path = "/api/v1/advertising/connect"
        mock_request.method = "POST"

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"platform": "google_ads", "credentials": {}}
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
                mock_auth.return_value = MagicMock(user_id="test")
                with patch.object(handler, "check_permission"):
                    result = await handler.handle_request(mock_request)

        assert result["status_code"] == 400
        assert "Missing required credentials" in result["body"]["error"]

    @pytest.mark.asyncio
    async def test_connect_success(self, handler, mock_request):
        """Test successful platform connection."""
        mock_request.path = "/api/v1/advertising/connect"
        mock_request.method = "POST"

        credentials = {
            "developer_token": "test_token",
            "client_id": "test_client_id",
            "client_secret": "test_secret",
            "refresh_token": "test_refresh",
            "customer_id": "123456",
        }

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"platform": "google_ads", "credentials": credentials}
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
                mock_auth.return_value = MagicMock(user_id="test")
                with patch.object(handler, "check_permission"):
                    with patch.object(
                        handler, "_get_connector", new_callable=AsyncMock
                    ) as mock_conn:
                        mock_conn.return_value = None
                        result = await handler.handle_request(mock_request)

        assert result["status_code"] == 200
        assert "Successfully connected" in result["body"]["message"]
        assert "google_ads" in _platform_credentials


class TestAdvertisingHandlerDisconnect:
    """Tests for platform disconnection endpoint."""

    @pytest.mark.asyncio
    async def test_disconnect_not_connected(self, handler, mock_request):
        """Test disconnect when platform not connected."""
        mock_request.path = "/api/v1/advertising/google_ads"
        mock_request.method = "DELETE"

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(user_id="test")
            with patch.object(handler, "check_permission"):
                result = await handler.handle_request(mock_request)

        assert result["status_code"] == 404
        assert "not connected" in result["body"]["error"]

    @pytest.mark.asyncio
    async def test_disconnect_success(self, handler, mock_request, connected_platform):
        """Test successful platform disconnection."""
        mock_request.path = "/api/v1/advertising/google_ads"
        mock_request.method = "DELETE"

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(user_id="test")
            with patch.object(handler, "check_permission"):
                result = await handler.handle_request(mock_request)

        assert result["status_code"] == 200
        assert "Disconnected" in result["body"]["message"]
        assert "google_ads" not in _platform_credentials


class TestAdvertisingHandlerCampaigns:
    """Tests for campaign management endpoints."""

    @pytest.mark.asyncio
    async def test_list_all_campaigns_no_platforms(self, handler, mock_request):
        """Test listing campaigns with no connected platforms."""
        mock_request.path = "/api/v1/advertising/campaigns"
        mock_request.method = "GET"

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(user_id="test")
            with patch.object(handler, "check_permission"):
                result = await handler.handle_request(mock_request)

        assert result["status_code"] == 200
        assert result["body"]["campaigns"] == []
        assert result["body"]["total"] == 0

    @pytest.mark.asyncio
    async def test_list_platform_campaigns_not_connected(self, handler, mock_request):
        """Test listing campaigns for unconnected platform."""
        mock_request.path = "/api/v1/advertising/google_ads/campaigns"
        mock_request.method = "GET"

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(user_id="test")
            with patch.object(handler, "check_permission"):
                result = await handler.handle_request(mock_request)

        assert result["status_code"] == 404
        assert "not connected" in result["body"]["error"]

    @pytest.mark.asyncio
    async def test_create_campaign_missing_name(self, handler, mock_request, connected_platform):
        """Test creating campaign without name."""
        mock_request.path = "/api/v1/advertising/google_ads/campaigns"
        mock_request.method = "POST"

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {}
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
                mock_auth.return_value = MagicMock(user_id="test")
                with patch.object(handler, "check_permission"):
                    result = await handler.handle_request(mock_request)

        assert result["status_code"] == 400
        assert "name is required" in result["body"]["error"]


class TestAdvertisingHandlerPerformance:
    """Tests for performance metrics endpoints."""

    @pytest.mark.asyncio
    async def test_cross_platform_performance_no_platforms(self, handler, mock_request):
        """Test cross-platform performance with no connected platforms."""
        mock_request.path = "/api/v1/advertising/performance"
        mock_request.method = "GET"
        mock_request.query = {}

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(user_id="test")
            with patch.object(handler, "check_permission"):
                result = await handler.handle_request(mock_request)

        assert result["status_code"] == 200
        assert "date_range" in result["body"]
        assert result["body"]["platforms"] == []

    @pytest.mark.asyncio
    async def test_platform_performance_not_connected(self, handler, mock_request):
        """Test platform performance for unconnected platform."""
        mock_request.path = "/api/v1/advertising/google_ads/performance"
        mock_request.method = "GET"
        mock_request.query = {}

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(user_id="test")
            with patch.object(handler, "check_permission"):
                result = await handler.handle_request(mock_request)

        assert result["status_code"] == 404
        assert "not connected" in result["body"]["error"]


class TestAdvertisingHandlerAnalyze:
    """Tests for performance analysis endpoint."""

    @pytest.mark.asyncio
    async def test_analyze_performance(self, handler, mock_request):
        """Test performance analysis endpoint."""
        mock_request.path = "/api/v1/advertising/analyze"
        mock_request.method = "POST"

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"type": "performance_review", "days": 30}
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
                mock_auth.return_value = MagicMock(user_id="test")
                with patch.object(handler, "check_permission"):
                    result = await handler.handle_request(mock_request)

        assert result["status_code"] == 200
        assert "analysis_id" in result["body"]
        assert result["body"]["type"] == "performance_review"


class TestAdvertisingHandlerBudgetRecommendations:
    """Tests for budget recommendations endpoint."""

    @pytest.mark.asyncio
    async def test_budget_recommendations(self, handler, mock_request):
        """Test budget recommendations endpoint."""
        mock_request.path = "/api/v1/advertising/budget-recommendations"
        mock_request.method = "GET"
        mock_request.query = {"budget": "10000", "objective": "conversions"}

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(user_id="test")
            with patch.object(handler, "check_permission"):
                result = await handler.handle_request(mock_request)

        assert result["status_code"] == 200
        assert result["body"]["total_budget"] == 10000.0
        assert result["body"]["objective"] == "conversions"


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestAdvertisingCircuitBreaker:
    """Tests for circuit breaker integration."""

    def test_circuit_breaker_exists(self):
        """Test that circuit breaker is available."""
        cb = get_advertising_circuit_breaker()
        assert cb is not None
        assert cb.name == "advertising_handler"

    def test_circuit_breaker_configuration(self):
        """Test circuit breaker configuration."""
        cb = get_advertising_circuit_breaker()
        assert cb.failure_threshold == 5
        assert cb.cooldown_seconds == 60

    def test_circuit_breaker_can_execute(self):
        """Test circuit breaker allows execution when closed."""
        cb = get_advertising_circuit_breaker()
        cb.reset()
        assert cb.can_execute() is True

    def test_circuit_breaker_records_success(self):
        """Test circuit breaker records success."""
        cb = get_advertising_circuit_breaker()
        cb.reset()
        cb.record_success()
        assert cb.can_execute() is True

    def test_circuit_breaker_records_failure(self):
        """Test circuit breaker records failure."""
        cb = get_advertising_circuit_breaker()
        cb.reset()
        for _ in range(5):
            cb.record_failure()
        # After threshold failures, circuit should open
        assert cb.can_execute() is False


# =============================================================================
# Helper Method Tests
# =============================================================================


class TestAdvertisingHandlerHelpers:
    """Tests for handler helper methods."""

    def test_get_required_credentials_google(self, handler):
        """Test required credentials for Google Ads."""
        creds = handler._get_required_credentials("google_ads")
        assert "developer_token" in creds
        assert "client_id" in creds
        assert "customer_id" in creds

    def test_get_required_credentials_meta(self, handler):
        """Test required credentials for Meta Ads."""
        creds = handler._get_required_credentials("meta_ads")
        assert "access_token" in creds
        assert "ad_account_id" in creds

    def test_get_required_credentials_linkedin(self, handler):
        """Test required credentials for LinkedIn Ads."""
        creds = handler._get_required_credentials("linkedin_ads")
        assert "access_token" in creds
        assert "ad_account_id" in creds

    def test_get_required_credentials_microsoft(self, handler):
        """Test required credentials for Microsoft Ads."""
        creds = handler._get_required_credentials("microsoft_ads")
        assert "developer_token" in creds
        assert "account_id" in creds

    def test_get_required_credentials_unknown(self, handler):
        """Test required credentials for unknown platform."""
        creds = handler._get_required_credentials("unknown_platform")
        assert creds == []

    def test_json_response(self, handler):
        """Test JSON response creation."""
        response = handler._json_response(200, {"key": "value"})
        assert response["status_code"] == 200
        assert response["body"]["key"] == "value"
        assert response["headers"]["Content-Type"] == "application/json"

    def test_error_response(self, handler):
        """Test error response creation."""
        response = handler._error_response(400, "Bad request")
        assert response["status_code"] == 400
        assert response["body"]["error"] == "Bad request"


class TestAdvertisingHandlerNormalization:
    """Tests for campaign normalization methods."""

    def test_normalize_google_campaign(self, handler):
        """Test Google campaign normalization."""
        mock_campaign = MagicMock()
        mock_campaign.id = "123"
        mock_campaign.name = "Test Campaign"
        mock_campaign.status = "ENABLED"
        mock_campaign.campaign_type = "SEARCH"
        mock_campaign.budget_micros = 50000000
        mock_campaign.start_date = date(2024, 1, 1)
        mock_campaign.end_date = date(2024, 12, 31)
        mock_campaign.bidding_strategy = "MAXIMIZE_CONVERSIONS"

        result = handler._normalize_google_campaign(mock_campaign)

        assert result["id"] == "123"
        assert result["platform"] == "google_ads"
        assert result["daily_budget"] == 50.0

    def test_normalize_meta_campaign(self, handler):
        """Test Meta campaign normalization."""
        mock_campaign = MagicMock()
        mock_campaign.id = "456"
        mock_campaign.name = "Meta Campaign"
        mock_campaign.status = "ACTIVE"
        mock_campaign.objective = "CONVERSIONS"
        mock_campaign.daily_budget = 100.0
        mock_campaign.lifetime_budget = 3000.0
        mock_campaign.start_time = datetime(2024, 1, 1, 0, 0, 0)
        mock_campaign.stop_time = datetime(2024, 12, 31, 23, 59, 59)
        mock_campaign.spend_cap = 5000.0

        result = handler._normalize_meta_campaign(mock_campaign)

        assert result["id"] == "456"
        assert result["platform"] == "meta_ads"
        assert result["daily_budget"] == 100.0

    def test_normalize_linkedin_campaign(self, handler):
        """Test LinkedIn campaign normalization."""
        mock_campaign = MagicMock()
        mock_campaign.id = "789"
        mock_campaign.name = "LinkedIn Campaign"
        mock_campaign.status = "ACTIVE"
        mock_campaign.objective_type = "WEBSITE_VISITS"
        mock_campaign.daily_budget = 50.0
        mock_campaign.total_budget = 1500.0
        mock_campaign.run_schedule_start = datetime(2024, 1, 1)
        mock_campaign.run_schedule_end = datetime(2024, 6, 30)
        mock_campaign.campaign_type = "SPONSORED_UPDATES"

        result = handler._normalize_linkedin_campaign(mock_campaign)

        assert result["id"] == "789"
        assert result["platform"] == "linkedin_ads"

    def test_normalize_microsoft_campaign(self, handler):
        """Test Microsoft campaign normalization."""
        mock_campaign = MagicMock()
        mock_campaign.id = "abc"
        mock_campaign.name = "MS Campaign"
        mock_campaign.status = "Active"
        mock_campaign.campaign_type = "Search"
        mock_campaign.daily_budget = 75.0
        mock_campaign.start_date = date(2024, 1, 1)
        mock_campaign.end_date = date(2024, 12, 31)
        mock_campaign.bidding_scheme = "EnhancedCpc"

        result = handler._normalize_microsoft_campaign(mock_campaign)

        assert result["id"] == "abc"
        assert result["platform"] == "microsoft_ads"


class TestAdvertisingHandlerRecommendations:
    """Tests for recommendation generation methods."""

    def test_generate_performance_summary(self, handler):
        """Test performance summary generation."""
        performance_data = {
            "google_ads": {"cost": 1000, "conversions": 50, "roas": 3.0},
            "meta_ads": {"cost": 500, "conversions": 25, "roas": 2.5},
        }

        result = handler._generate_performance_summary(performance_data)

        assert result["total_spend"] == 1500.0
        assert result["total_conversions"] == 75
        assert result["platforms_analyzed"] == 2

    def test_generate_recommendations_high_roas(self, handler):
        """Test recommendations for high ROAS platform."""
        performance_data = {
            "google_ads": {"roas": 4.0, "cpc": 1.5},
        }

        result = handler._generate_recommendations(performance_data)

        assert any(r["type"] == "increase_budget" for r in result)

    def test_generate_recommendations_low_roas(self, handler):
        """Test recommendations for low ROAS platform."""
        performance_data = {
            "meta_ads": {"roas": 0.5, "cpc": 3.0},
        }

        result = handler._generate_recommendations(performance_data)

        assert any(r["type"] == "optimize" for r in result)

    def test_generate_recommendations_high_cpc(self, handler):
        """Test recommendations for high CPC platform."""
        performance_data = {
            "linkedin_ads": {"roas": 2.0, "cpc": 8.0},
        }

        result = handler._generate_recommendations(performance_data)

        assert any(r["type"] == "reduce_cpc" for r in result)

    def test_generate_insights(self, handler):
        """Test insights generation."""
        performance_data = {
            "google_ads": {"cost": 800},
            "meta_ads": {"cost": 200},
        }

        result = handler._generate_insights(performance_data)

        assert len(result) == 2
        assert "google_ads" in result[0]

    def test_calculate_budget_recommendations_conversions(self, handler):
        """Test budget recommendations for conversions objective."""
        performance_data = {
            "google_ads": {"roas": 3.0},
            "meta_ads": {"roas": 1.5},
        }

        result = handler._calculate_budget_recommendations(performance_data, 10000, "conversions")

        # Higher ROAS platform should get more budget
        google_rec = next(r for r in result if r["platform"] == "google_ads")
        meta_rec = next(r for r in result if r["platform"] == "meta_ads")
        assert google_rec["recommended_budget"] > meta_rec["recommended_budget"]

    def test_calculate_budget_recommendations_awareness(self, handler):
        """Test budget recommendations for awareness objective."""
        performance_data = {
            "google_ads": {"roas": 3.0},
            "meta_ads": {"roas": 1.5},
        }

        result = handler._calculate_budget_recommendations(performance_data, 10000, "awareness")

        # For awareness, budget should be more evenly distributed
        google_rec = next(r for r in result if r["platform"] == "google_ads")
        meta_rec = next(r for r in result if r["platform"] == "meta_ads")
        assert google_rec["recommended_budget"] == meta_rec["recommended_budget"]

    def test_generate_budget_rationale_conversions(self, handler):
        """Test budget rationale for conversions objective."""
        result = handler._generate_budget_rationale({}, "conversions")
        assert "ROAS" in result

    def test_generate_budget_rationale_awareness(self, handler):
        """Test budget rationale for awareness objective."""
        result = handler._generate_budget_rationale({}, "awareness")
        assert "reach" in result.lower()

    def test_generate_budget_rationale_balanced(self, handler):
        """Test budget rationale for balanced objective."""
        result = handler._generate_budget_rationale({}, "balanced")
        assert "Balanced" in result


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestAdvertisingHandlerErrors:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_unknown_endpoint(self, handler, mock_request):
        """Test handling of unknown endpoint."""
        mock_request.path = "/api/v1/advertising/unknown/endpoint"
        mock_request.method = "GET"

        result = await handler.handle_request(mock_request)

        assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_invalid_json_body(self, handler, mock_request):
        """Test handling of invalid JSON body."""
        mock_request.path = "/api/v1/advertising/connect"
        mock_request.method = "POST"

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.side_effect = ValueError("Invalid JSON")
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
                mock_auth.return_value = MagicMock(user_id="test")
                with patch.object(handler, "check_permission"):
                    result = await handler.handle_request(mock_request)

        assert result["status_code"] == 400
        assert "invalid" in result["body"]["error"].lower()


# =============================================================================
# Integration Tests
# =============================================================================


class TestAdvertisingHandlerIntegration:
    """Integration tests for advertising handler."""

    @pytest.mark.asyncio
    async def test_full_platform_lifecycle(self, handler, mock_request):
        """Test complete platform lifecycle: connect, list, disconnect."""
        credentials = {
            "access_token": "test_token",
            "ad_account_id": "123456",
        }

        # Connect platform
        mock_request.path = "/api/v1/advertising/connect"
        mock_request.method = "POST"

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"platform": "meta_ads", "credentials": credentials}
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
                mock_auth.return_value = MagicMock(user_id="test")
                with patch.object(handler, "check_permission"):
                    with patch.object(
                        handler, "_get_connector", new_callable=AsyncMock
                    ) as mock_conn:
                        mock_conn.return_value = None
                        result = await handler.handle_request(mock_request)

        assert result["status_code"] == 200
        assert "meta_ads" in _platform_credentials

        # List platforms
        mock_request.path = "/api/v1/advertising/platforms"
        mock_request.method = "GET"

        result = await handler.handle_request(mock_request)
        assert result["status_code"] == 200
        meta_platform = next(p for p in result["body"]["platforms"] if p["id"] == "meta_ads")
        assert meta_platform["connected"] is True

        # Disconnect platform
        mock_request.path = "/api/v1/advertising/meta_ads"
        mock_request.method = "DELETE"

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(user_id="test")
            with patch.object(handler, "check_permission"):
                result = await handler.handle_request(mock_request)

        assert result["status_code"] == 200
        assert "meta_ads" not in _platform_credentials
