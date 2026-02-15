"""
Tests for Analytics Platforms Handler.

Comprehensive test suite covering:
- Platform configuration and validation
- Dataclass creation and serialization
- Handler routing and request handling
- Circuit breaker integration
- Rate limiting
- Dashboard management
- Query execution
- Report generation
- Metrics aggregation
- Error handling and edge cases
"""

from __future__ import annotations

import pytest
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.features.analytics_platforms import (
    AnalyticsPlatformsHandler,
    SUPPORTED_PLATFORMS,
    UnifiedMetric,
    UnifiedDashboard,
    get_analytics_circuit_breaker,
    _platform_credentials,
    _platform_connectors,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def handler():
    """Create a handler instance for testing."""
    return AnalyticsPlatformsHandler(server_context={})


@pytest.fixture
def mock_request():
    """Create a mock request object."""
    request = MagicMock()
    request.method = "GET"
    request.path = "/api/v1/analytics/platforms"
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
def connected_metabase():
    """Set up a connected Metabase platform for testing."""
    _platform_credentials["metabase"] = {
        "credentials": {
            "base_url": "http://localhost:3000",
            "username": "admin@example.com",
            "password": "secret123",
        },
        "connected_at": datetime.now(timezone.utc).isoformat(),
    }
    return "metabase"


@pytest.fixture
def connected_google_analytics():
    """Set up a connected Google Analytics platform for testing."""
    _platform_credentials["google_analytics"] = {
        "credentials": {
            "property_id": "GA4-123456",
            "credentials_json": '{"type": "service_account"}',
        },
        "connected_at": datetime.now(timezone.utc).isoformat(),
    }
    return "google_analytics"


@pytest.fixture
def connected_mixpanel():
    """Set up a connected Mixpanel platform for testing."""
    _platform_credentials["mixpanel"] = {
        "credentials": {
            "project_id": "123456",
            "api_secret": "secret123",
        },
        "connected_at": datetime.now(timezone.utc).isoformat(),
    }
    return "mixpanel"


# =============================================================================
# Platform Configuration Tests
# =============================================================================


class TestSupportedPlatforms:
    """Tests for analytics platform configuration."""

    def test_all_platforms_defined(self):
        """Test that analytics platforms are configured."""
        assert "metabase" in SUPPORTED_PLATFORMS
        assert "google_analytics" in SUPPORTED_PLATFORMS
        assert "mixpanel" in SUPPORTED_PLATFORMS

    def test_platform_has_required_fields(self):
        """Test that all platforms have required configuration."""
        for platform_id, config in SUPPORTED_PLATFORMS.items():
            assert "name" in config, f"{platform_id} missing name"
            assert "description" in config, f"{platform_id} missing description"
            assert "features" in config, f"{platform_id} missing features"
            assert isinstance(config["features"], list)

    def test_metabase_features(self):
        """Test Metabase has expected features."""
        config = SUPPORTED_PLATFORMS["metabase"]
        assert "dashboards" in config["features"]
        assert "queries" in config["features"]

    def test_google_analytics_features(self):
        """Test Google Analytics has expected features."""
        config = SUPPORTED_PLATFORMS["google_analytics"]
        assert "reports" in config["features"]
        assert "realtime" in config["features"]

    def test_mixpanel_features(self):
        """Test Mixpanel has expected features."""
        config = SUPPORTED_PLATFORMS["mixpanel"]
        assert "funnels" in config["features"]
        assert "retention" in config["features"]

    def test_platform_count(self):
        """Test expected number of platforms."""
        assert len(SUPPORTED_PLATFORMS) == 3


# =============================================================================
# UnifiedMetric Tests
# =============================================================================


class TestUnifiedMetric:
    """Tests for UnifiedMetric dataclass."""

    def test_metric_creation(self):
        """Test creating a unified metric."""
        metric = UnifiedMetric(
            name="sessions",
            value=1000,
            platform="google_analytics",
            dimension="date",
            period="7d",
            change_percent=5.5,
        )

        assert metric.name == "sessions"
        assert metric.value == 1000
        assert metric.platform == "google_analytics"

    def test_metric_minimal(self):
        """Test metric with minimal fields."""
        metric = UnifiedMetric(
            name="pageviews",
            value=500,
            platform="metabase",
        )

        assert metric.name == "pageviews"
        assert metric.dimension is None
        assert metric.change_percent is None

    def test_metric_to_dict(self):
        """Test metric serialization to dictionary."""
        metric = UnifiedMetric(
            name="users",
            value=2500,
            platform="mixpanel",
            dimension="country",
            period="30d",
            change_percent=10.2,
        )

        result = metric.to_dict()

        assert result["name"] == "users"
        assert result["value"] == 2500
        assert result["platform"] == "mixpanel"
        assert result["dimension"] == "country"
        assert result["period"] == "30d"
        assert result["change_percent"] == 10.2

    def test_metric_with_float_value(self):
        """Test metric with float value."""
        metric = UnifiedMetric(
            name="bounce_rate",
            value=0.45,
            platform="google_analytics",
        )

        assert metric.value == 0.45

    def test_metric_with_negative_change(self):
        """Test metric with negative change percent."""
        metric = UnifiedMetric(
            name="conversions",
            value=100,
            platform="mixpanel",
            change_percent=-15.5,
        )

        assert metric.change_percent == -15.5


# =============================================================================
# UnifiedDashboard Tests
# =============================================================================


class TestUnifiedDashboard:
    """Tests for UnifiedDashboard dataclass."""

    def test_dashboard_creation(self):
        """Test creating a unified dashboard."""
        dashboard = UnifiedDashboard(
            id="dash_123",
            platform="metabase",
            name="Sales Dashboard",
            description="Overview of sales metrics",
            url="/dashboard/123",
            created_at=datetime.now(),
            cards_count=10,
        )

        assert dashboard.id == "dash_123"
        assert dashboard.platform == "metabase"
        assert dashboard.name == "Sales Dashboard"
        assert dashboard.cards_count == 10

    def test_dashboard_minimal(self):
        """Test dashboard with minimal fields."""
        dashboard = UnifiedDashboard(
            id="dash_456",
            platform="metabase",
            name="Test Dashboard",
            description=None,
            url=None,
            created_at=None,
        )

        assert dashboard.description is None
        assert dashboard.cards_count == 0

    def test_dashboard_to_dict(self):
        """Test dashboard serialization to dictionary."""
        created = datetime(2024, 1, 15, 10, 30, 0)

        dashboard = UnifiedDashboard(
            id="dash_789",
            platform="metabase",
            name="Marketing Dashboard",
            description="Marketing metrics and KPIs",
            url="/dashboard/789",
            created_at=created,
            cards_count=8,
        )

        result = dashboard.to_dict()

        assert result["id"] == "dash_789"
        assert result["platform"] == "metabase"
        assert result["name"] == "Marketing Dashboard"
        assert result["created_at"] == created.isoformat()
        assert result["cards_count"] == 8

    def test_dashboard_to_dict_with_none(self):
        """Test dashboard serialization with None values."""
        dashboard = UnifiedDashboard(
            id="dash_none",
            platform="metabase",
            name="Empty Dashboard",
            description=None,
            url=None,
            created_at=None,
        )

        result = dashboard.to_dict()

        assert result["description"] is None
        assert result["url"] is None
        assert result["created_at"] is None


# =============================================================================
# Handler Tests
# =============================================================================


class TestAnalyticsPlatformsHandler:
    """Tests for AnalyticsPlatformsHandler class."""

    def test_handler_creation(self):
        """Test creating handler instance."""
        handler = AnalyticsPlatformsHandler(server_context={})
        assert handler is not None

    def test_handler_creation_with_ctx(self):
        """Test creating handler with ctx parameter."""
        handler = AnalyticsPlatformsHandler(ctx={"key": "value"})
        assert handler.ctx == {"key": "value"}

    def test_handler_has_routes(self):
        """Test that handler has route definitions."""
        handler = AnalyticsPlatformsHandler(server_context={})
        assert hasattr(handler, "handle_request")
        assert hasattr(handler, "ROUTES")

    def test_can_handle_analytics_routes(self, handler):
        """Test that handler recognizes analytics routes."""
        assert handler.can_handle("/api/v1/analytics/platforms")
        assert handler.can_handle("/api/v1/analytics/connect")
        assert handler.can_handle("/api/v1/analytics/dashboards")
        assert handler.can_handle("/api/v1/analytics/metabase/dashboards")

    def test_cannot_handle_other_routes(self, handler):
        """Test that handler rejects non-analytics routes."""
        assert not handler.can_handle("/api/v1/advertising/platforms")
        assert not handler.can_handle("/api/v1/debates")
        assert not handler.can_handle("/api/v1/other")

    def test_routes_list(self, handler):
        """Test that ROUTES contains expected endpoints."""
        routes = handler.ROUTES
        assert "/api/v1/analytics/platforms" in routes
        assert "/api/v1/analytics/connect" in routes
        assert "/api/v1/analytics/dashboards" in routes
        assert "/api/v1/analytics/query" in routes
        assert "/api/v1/analytics/reports" in routes
        assert "/api/v1/analytics/metrics" in routes
        assert "/api/v1/analytics/realtime" in routes


class TestAnalyticsHandlerListPlatforms:
    """Tests for listing platforms endpoint."""

    @pytest.mark.asyncio
    async def test_list_platforms_success(self, handler, mock_request):
        """Test successful platform listing."""
        mock_request.path = "/api/v1/analytics/platforms"
        mock_request.method = "GET"

        result = await handler.handle_request(mock_request)

        assert result["status_code"] == 200
        assert "platforms" in result["body"]
        assert len(result["body"]["platforms"]) == 3

    @pytest.mark.asyncio
    async def test_list_platforms_shows_connection_status(
        self, handler, mock_request, connected_metabase
    ):
        """Test that platforms show correct connection status."""
        mock_request.path = "/api/v1/analytics/platforms"
        mock_request.method = "GET"

        result = await handler.handle_request(mock_request)

        platforms = result["body"]["platforms"]
        metabase = next(p for p in platforms if p["id"] == "metabase")
        mixpanel = next(p for p in platforms if p["id"] == "mixpanel")

        assert metabase["connected"] is True
        assert mixpanel["connected"] is False

    @pytest.mark.asyncio
    async def test_list_platforms_connected_count(
        self, handler, mock_request, connected_metabase, connected_google_analytics
    ):
        """Test connected_count in response."""
        mock_request.path = "/api/v1/analytics/platforms"
        mock_request.method = "GET"

        result = await handler.handle_request(mock_request)

        assert result["body"]["connected_count"] == 2


class TestAnalyticsHandlerConnect:
    """Tests for platform connection endpoint."""

    @pytest.mark.asyncio
    async def test_connect_missing_platform(self, handler, mock_request):
        """Test connect with missing platform."""
        mock_request.path = "/api/v1/analytics/connect"
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
        mock_request.path = "/api/v1/analytics/connect"
        mock_request.method = "POST"

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"platform": "unsupported"}
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
                mock_auth.return_value = MagicMock(user_id="test")
                with patch.object(handler, "check_permission"):
                    result = await handler.handle_request(mock_request)

        assert result["status_code"] == 400
        assert "Unsupported platform" in result["body"]["error"]

    @pytest.mark.asyncio
    async def test_connect_missing_credentials(self, handler, mock_request):
        """Test connect with missing credentials."""
        mock_request.path = "/api/v1/analytics/connect"
        mock_request.method = "POST"

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"platform": "metabase", "credentials": {}}
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
                mock_auth.return_value = MagicMock(user_id="test")
                with patch.object(handler, "check_permission"):
                    result = await handler.handle_request(mock_request)

        assert result["status_code"] == 400
        assert "Missing required credentials" in result["body"]["error"]

    @pytest.mark.asyncio
    async def test_connect_metabase_success(self, handler, mock_request):
        """Test successful Metabase connection."""
        mock_request.path = "/api/v1/analytics/connect"
        mock_request.method = "POST"

        credentials = {
            "base_url": "http://localhost:3000",
            "username": "admin@example.com",
            "password": "secret123",
        }

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"platform": "metabase", "credentials": credentials}
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
        assert "metabase" in _platform_credentials

    @pytest.mark.asyncio
    async def test_connect_google_analytics_success(self, handler, mock_request):
        """Test successful Google Analytics connection."""
        mock_request.path = "/api/v1/analytics/connect"
        mock_request.method = "POST"

        credentials = {
            "property_id": "GA4-123456",
            "credentials_json": '{"type": "service_account"}',
        }

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"platform": "google_analytics", "credentials": credentials}
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
                mock_auth.return_value = MagicMock(user_id="test")
                with patch.object(handler, "check_permission"):
                    with patch.object(
                        handler, "_get_connector", new_callable=AsyncMock
                    ) as mock_conn:
                        mock_conn.return_value = None
                        result = await handler.handle_request(mock_request)

        assert result["status_code"] == 200
        assert "google_analytics" in _platform_credentials

    @pytest.mark.asyncio
    async def test_connect_mixpanel_success(self, handler, mock_request):
        """Test successful Mixpanel connection."""
        mock_request.path = "/api/v1/analytics/connect"
        mock_request.method = "POST"

        credentials = {
            "project_id": "123456",
            "api_secret": "secret123",
        }

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"platform": "mixpanel", "credentials": credentials}
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
                mock_auth.return_value = MagicMock(user_id="test")
                with patch.object(handler, "check_permission"):
                    with patch.object(
                        handler, "_get_connector", new_callable=AsyncMock
                    ) as mock_conn:
                        mock_conn.return_value = None
                        result = await handler.handle_request(mock_request)

        assert result["status_code"] == 200
        assert "mixpanel" in _platform_credentials


class TestAnalyticsHandlerDisconnect:
    """Tests for platform disconnection endpoint."""

    @pytest.mark.asyncio
    async def test_disconnect_not_connected(self, handler, mock_request):
        """Test disconnect when platform not connected."""
        mock_request.path = "/api/v1/analytics/metabase"
        mock_request.method = "DELETE"

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(user_id="test")
            with patch.object(handler, "check_permission"):
                result = await handler.handle_request(mock_request)

        assert result["status_code"] == 404
        assert "not connected" in result["body"]["error"]

    @pytest.mark.asyncio
    async def test_disconnect_success(self, handler, mock_request, connected_metabase):
        """Test successful platform disconnection."""
        mock_request.path = "/api/v1/analytics/metabase"
        mock_request.method = "DELETE"

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(user_id="test")
            with patch.object(handler, "check_permission"):
                result = await handler.handle_request(mock_request)

        assert result["status_code"] == 200
        assert "Disconnected" in result["body"]["message"]
        assert "metabase" not in _platform_credentials


class TestAnalyticsHandlerDashboards:
    """Tests for dashboard endpoints."""

    @pytest.mark.asyncio
    async def test_list_all_dashboards_no_platforms(self, handler, mock_request):
        """Test listing dashboards with no connected platforms."""
        mock_request.path = "/api/v1/analytics/dashboards"
        mock_request.method = "GET"

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(user_id="test")
            with patch.object(handler, "check_permission"):
                result = await handler.handle_request(mock_request)

        assert result["status_code"] == 200
        assert result["body"]["dashboards"] == []
        assert result["body"]["total"] == 0

    @pytest.mark.asyncio
    async def test_list_platform_dashboards_not_connected(self, handler, mock_request):
        """Test listing dashboards for unconnected platform."""
        mock_request.path = "/api/v1/analytics/metabase/dashboards"
        mock_request.method = "GET"

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(user_id="test")
            with patch.object(handler, "check_permission"):
                result = await handler.handle_request(mock_request)

        assert result["status_code"] == 404
        assert "not connected" in result["body"]["error"]

    @pytest.mark.asyncio
    async def test_get_dashboard_not_connected(self, handler, mock_request):
        """Test getting a dashboard when platform not connected."""
        mock_request.path = "/api/v1/analytics/metabase/dashboards/123"
        mock_request.method = "GET"

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(user_id="test")
            with patch.object(handler, "check_permission"):
                result = await handler.handle_request(mock_request)

        assert result["status_code"] == 404
        assert "not connected" in result["body"]["error"]


class TestAnalyticsHandlerQuery:
    """Tests for query execution endpoint."""

    @pytest.mark.asyncio
    async def test_query_missing_platform(self, handler, mock_request):
        """Test query execution with missing platform."""
        mock_request.path = "/api/v1/analytics/query"
        mock_request.method = "POST"

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"query": "SELECT * FROM users"}
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
                mock_auth.return_value = MagicMock(user_id="test")
                with patch.object(handler, "check_permission"):
                    result = await handler.handle_request(mock_request)

        assert result["status_code"] == 400
        assert "platform is required" in result["body"]["error"]

    @pytest.mark.asyncio
    async def test_query_missing_query(self, handler, mock_request, connected_metabase):
        """Test query execution with missing query."""
        mock_request.path = "/api/v1/analytics/query"
        mock_request.method = "POST"

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"platform": "metabase"}
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
                mock_auth.return_value = MagicMock(user_id="test")
                with patch.object(handler, "check_permission"):
                    result = await handler.handle_request(mock_request)

        assert result["status_code"] == 400
        assert "Query is required" in result["body"]["error"]


class TestAnalyticsHandlerReports:
    """Tests for reports endpoints."""

    @pytest.mark.asyncio
    async def test_list_reports(self, handler, mock_request):
        """Test listing available reports."""
        mock_request.path = "/api/v1/analytics/reports"
        mock_request.method = "GET"
        mock_request.query = {}

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(user_id="test")
            with patch.object(handler, "check_permission"):
                result = await handler.handle_request(mock_request)

        assert result["status_code"] == 200
        assert "reports" in result["body"]
        assert result["body"]["total"] > 0

    @pytest.mark.asyncio
    async def test_list_reports_by_platform(self, handler, mock_request):
        """Test listing reports filtered by platform."""
        mock_request.path = "/api/v1/analytics/reports"
        mock_request.method = "GET"
        mock_request.query = {"platform": "mixpanel"}

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(user_id="test")
            with patch.object(handler, "check_permission"):
                result = await handler.handle_request(mock_request)

        assert result["status_code"] == 200
        for report in result["body"]["reports"]:
            assert "mixpanel" in report["platforms"]

    @pytest.mark.asyncio
    async def test_generate_report(self, handler, mock_request):
        """Test report generation endpoint."""
        mock_request.path = "/api/v1/analytics/reports/generate"
        mock_request.method = "POST"

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"type": "traffic_overview", "days": 30}
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
                mock_auth.return_value = MagicMock(user_id="test")
                with patch.object(handler, "check_permission"):
                    result = await handler.handle_request(mock_request)

        assert result["status_code"] == 200
        assert "report_id" in result["body"]
        assert result["body"]["type"] == "traffic_overview"


class TestAnalyticsHandlerMetrics:
    """Tests for metrics endpoints."""

    @pytest.mark.asyncio
    async def test_cross_platform_metrics_no_platforms(self, handler, mock_request):
        """Test cross-platform metrics with no connected platforms."""
        mock_request.path = "/api/v1/analytics/metrics"
        mock_request.method = "GET"
        mock_request.query = {}

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(user_id="test")
            with patch.object(handler, "check_permission"):
                result = await handler.handle_request(mock_request)

        assert result["status_code"] == 200
        assert "date_range" in result["body"]
        assert result["body"]["platforms"] == {}


class TestAnalyticsHandlerRealtime:
    """Tests for realtime metrics endpoint."""

    @pytest.mark.asyncio
    async def test_realtime_metrics_no_ga(self, handler, mock_request):
        """Test realtime metrics without Google Analytics connected."""
        mock_request.path = "/api/v1/analytics/realtime"
        mock_request.method = "GET"

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(user_id="test")
            with patch.object(handler, "check_permission"):
                result = await handler.handle_request(mock_request)

        assert result["status_code"] == 404
        assert "Google Analytics is not connected" in result["body"]["error"]


class TestAnalyticsHandlerEvents:
    """Tests for events endpoint."""

    @pytest.mark.asyncio
    async def test_get_events_not_connected(self, handler, mock_request):
        """Test getting events for unconnected platform."""
        mock_request.path = "/api/v1/analytics/mixpanel/events"
        mock_request.method = "GET"
        mock_request.query = {}

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(user_id="test")
            with patch.object(handler, "check_permission"):
                result = await handler.handle_request(mock_request)

        assert result["status_code"] == 404
        assert "not connected" in result["body"]["error"]


class TestAnalyticsHandlerFunnels:
    """Tests for funnels endpoint."""

    @pytest.mark.asyncio
    async def test_get_funnels_not_connected(self, handler, mock_request):
        """Test getting funnels for unconnected platform."""
        mock_request.path = "/api/v1/analytics/mixpanel/funnels"
        mock_request.method = "GET"
        mock_request.query = {}

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(user_id="test")
            with patch.object(handler, "check_permission"):
                result = await handler.handle_request(mock_request)

        assert result["status_code"] == 404
        assert "not connected" in result["body"]["error"]

    @pytest.mark.asyncio
    async def test_get_funnels_missing_id(self, handler, mock_request, connected_mixpanel):
        """Test getting funnels without funnel_id."""
        mock_request.path = "/api/v1/analytics/mixpanel/funnels"
        mock_request.method = "GET"
        mock_request.query = {}

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(user_id="test")
            with patch.object(handler, "check_permission"):
                result = await handler.handle_request(mock_request)

        assert result["status_code"] == 400
        assert "funnel_id is required" in result["body"]["error"]

    @pytest.mark.asyncio
    async def test_get_funnels_wrong_platform(self, handler, mock_request, connected_metabase):
        """Test getting funnels for non-Mixpanel platform."""
        mock_request.path = "/api/v1/analytics/metabase/funnels"
        mock_request.method = "GET"
        mock_request.query = {"funnel_id": "123"}

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(user_id="test")
            with patch.object(handler, "check_permission"):
                result = await handler.handle_request(mock_request)

        assert result["status_code"] == 400
        assert "only available for Mixpanel" in result["body"]["error"]


class TestAnalyticsHandlerRetention:
    """Tests for retention endpoint."""

    @pytest.mark.asyncio
    async def test_get_retention_not_connected(self, handler, mock_request):
        """Test getting retention for unconnected platform."""
        mock_request.path = "/api/v1/analytics/mixpanel/retention"
        mock_request.method = "GET"

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(user_id="test")
            with patch.object(handler, "check_permission"):
                result = await handler.handle_request(mock_request)

        assert result["status_code"] == 404
        assert "not connected" in result["body"]["error"]

    @pytest.mark.asyncio
    async def test_get_retention_wrong_platform(
        self, handler, mock_request, connected_google_analytics
    ):
        """Test getting retention for non-Mixpanel platform."""
        mock_request.path = "/api/v1/analytics/google_analytics/retention"
        mock_request.method = "GET"

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(user_id="test")
            with patch.object(handler, "check_permission"):
                result = await handler.handle_request(mock_request)

        assert result["status_code"] == 400
        assert "only available for Mixpanel" in result["body"]["error"]


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestAnalyticsCircuitBreaker:
    """Tests for circuit breaker integration."""

    def test_circuit_breaker_exists(self):
        """Test that circuit breaker is available."""
        cb = get_analytics_circuit_breaker()
        assert cb is not None
        assert cb.name == "analytics_platforms_handler"

    def test_circuit_breaker_configuration(self):
        """Test circuit breaker configuration."""
        cb = get_analytics_circuit_breaker()
        assert cb.failure_threshold == 5
        assert cb.cooldown_seconds == 60

    def test_circuit_breaker_can_execute(self):
        """Test circuit breaker allows execution when closed."""
        cb = get_analytics_circuit_breaker()
        cb.reset()
        assert cb.can_execute() is True

    def test_circuit_breaker_records_success(self):
        """Test circuit breaker records success."""
        cb = get_analytics_circuit_breaker()
        cb.reset()
        cb.record_success()
        assert cb.can_execute() is True

    def test_circuit_breaker_records_failure(self):
        """Test circuit breaker records failure."""
        cb = get_analytics_circuit_breaker()
        cb.reset()
        for _ in range(5):
            cb.record_failure()
        # After threshold failures, circuit should open
        assert cb.can_execute() is False


# =============================================================================
# Helper Method Tests
# =============================================================================


class TestAnalyticsHandlerHelpers:
    """Tests for handler helper methods."""

    def test_get_required_credentials_metabase(self, handler):
        """Test required credentials for Metabase."""
        creds = handler._get_required_credentials("metabase")
        assert "base_url" in creds
        assert "username" in creds
        assert "password" in creds

    def test_get_required_credentials_google_analytics(self, handler):
        """Test required credentials for Google Analytics."""
        creds = handler._get_required_credentials("google_analytics")
        assert "property_id" in creds
        assert "credentials_json" in creds

    def test_get_required_credentials_mixpanel(self, handler):
        """Test required credentials for Mixpanel."""
        creds = handler._get_required_credentials("mixpanel")
        assert "project_id" in creds
        assert "api_secret" in creds

    def test_get_required_credentials_unknown(self, handler):
        """Test required credentials for unknown platform."""
        creds = handler._get_required_credentials("unknown")
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


class TestAnalyticsHandlerNormalization:
    """Tests for normalization methods."""

    def test_normalize_metabase_dashboard(self, handler):
        """Test Metabase dashboard normalization."""
        mock_dashboard = MagicMock()
        mock_dashboard.id = 123
        mock_dashboard.name = "Sales Dashboard"
        mock_dashboard.description = "Sales metrics"
        mock_dashboard.created_at = datetime(2024, 1, 15)
        mock_dashboard.dashcards = [MagicMock(), MagicMock()]

        result = handler._normalize_metabase_dashboard(mock_dashboard)

        assert result["id"] == "123"
        assert result["platform"] == "metabase"
        assert result["name"] == "Sales Dashboard"
        assert result["cards_count"] == 2

    def test_normalize_metabase_card(self, handler):
        """Test Metabase card normalization."""
        mock_card = MagicMock()
        mock_card.id = 456
        mock_card.name = "Revenue Chart"
        mock_card.description = "Monthly revenue"
        mock_card.display = "line"
        mock_card.query_type = "native"

        result = handler._normalize_metabase_card(mock_card)

        assert result["id"] == "456"
        assert result["name"] == "Revenue Chart"
        assert result["display_type"] == "line"

    def test_normalize_ga_report(self, handler):
        """Test GA4 report normalization."""
        mock_report = MagicMock()
        mock_report.dimension_headers = [MagicMock(name="date"), MagicMock(name="country")]
        mock_report.metric_headers = [MagicMock(name="sessions"), MagicMock(name="users")]
        mock_report.rows = [
            MagicMock(
                dimension_values=[MagicMock(value="2024-01-15"), MagicMock(value="US")],
                metric_values=[MagicMock(value="100"), MagicMock(value="80")],
            )
        ]
        mock_report.row_count = 1

        result = handler._normalize_ga_report(mock_report)

        assert result["dimensions"] == ["date", "country"]
        assert result["metrics"] == ["sessions", "users"]
        assert result["row_count"] == 1

    def test_extract_ga_totals(self, handler):
        """Test GA4 totals extraction."""
        mock_report = MagicMock()
        mock_report.metric_headers = [
            MagicMock(name="totalUsers"),
            MagicMock(name="sessions"),
            MagicMock(name="eventCount"),
        ]
        mock_report.rows = [
            MagicMock(
                metric_values=[
                    MagicMock(value="1000"),
                    MagicMock(value="1500"),
                    MagicMock(value="5000"),
                ]
            )
        ]

        result = handler._extract_ga_totals(mock_report)

        assert result["users"] == 1000
        assert result["sessions"] == 1500
        assert result["events"] == 5000

    def test_extract_ga_totals_empty(self, handler):
        """Test GA4 totals extraction with empty report."""
        mock_report = MagicMock()
        mock_report.rows = []

        result = handler._extract_ga_totals(mock_report)

        assert result["users"] == 0
        assert result["sessions"] == 0
        assert result["events"] == 0

    def test_normalize_mixpanel_insight(self, handler):
        """Test Mixpanel insight normalization."""
        mock_insight = MagicMock()
        mock_insight.total = 5000
        mock_insight.series = [{"date": "2024-01-15", "value": 100}]
        mock_insight.breakdown = {"US": 3000, "UK": 2000}

        result = handler._normalize_mixpanel_insight(mock_insight)

        assert result["total"] == 5000
        assert len(result["series"]) == 1

    def test_normalize_mixpanel_event(self, handler):
        """Test Mixpanel event normalization."""
        mock_event = MagicMock()
        mock_event.name = "page_view"
        mock_event.distinct_id = "user_123"
        mock_event.timestamp = datetime(2024, 1, 15, 10, 30, 0)
        mock_event.properties = {"page": "/home"}

        result = handler._normalize_mixpanel_event(mock_event)

        assert result["event_name"] == "page_view"
        assert result["distinct_id"] == "user_123"

    def test_normalize_mixpanel_funnel(self, handler):
        """Test Mixpanel funnel normalization."""
        mock_funnel = MagicMock()
        mock_funnel.steps = [
            {"name": "Sign Up", "count": 1000},
            {"name": "Verify Email", "count": 800},
        ]
        mock_funnel.overall_conversion = 0.8
        mock_funnel.from_date = "2024-01-01"
        mock_funnel.to_date = "2024-01-31"

        result = handler._normalize_mixpanel_funnel(mock_funnel)

        assert len(result["steps"]) == 2
        assert result["conversion_rate"] == 0.8

    def test_normalize_mixpanel_retention(self, handler):
        """Test Mixpanel retention normalization."""
        mock_retention = MagicMock()
        mock_retention.cohorts = ["2024-01-01", "2024-01-08"]
        mock_retention.retention = [
            {"day": 0, "value": 100},
            {"day": 1, "value": 40},
        ]

        result = handler._normalize_mixpanel_retention(mock_retention)

        assert len(result["cohorts"]) == 2
        assert len(result["retention_by_day"]) == 2


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestAnalyticsHandlerErrors:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_unknown_endpoint(self, handler, mock_request):
        """Test handling of unknown endpoint."""
        mock_request.path = "/api/v1/analytics/unknown/endpoint"
        mock_request.method = "GET"

        result = await handler.handle_request(mock_request)

        assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_invalid_json_body(self, handler, mock_request):
        """Test handling of invalid JSON body."""
        mock_request.path = "/api/v1/analytics/connect"
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


class TestAnalyticsHandlerIntegration:
    """Integration tests for analytics handler."""

    @pytest.mark.asyncio
    async def test_full_platform_lifecycle(self, handler, mock_request):
        """Test complete platform lifecycle: connect, list, disconnect."""
        credentials = {
            "base_url": "http://localhost:3000",
            "username": "admin@example.com",
            "password": "secret123",
        }

        # Connect platform
        mock_request.path = "/api/v1/analytics/connect"
        mock_request.method = "POST"

        with patch.object(handler, "_get_json_body", new_callable=AsyncMock) as mock_body:
            mock_body.return_value = {"platform": "metabase", "credentials": credentials}
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
                mock_auth.return_value = MagicMock(user_id="test")
                with patch.object(handler, "check_permission"):
                    with patch.object(
                        handler, "_get_connector", new_callable=AsyncMock
                    ) as mock_conn:
                        mock_conn.return_value = None
                        result = await handler.handle_request(mock_request)

        assert result["status_code"] == 200
        assert "metabase" in _platform_credentials

        # List platforms
        mock_request.path = "/api/v1/analytics/platforms"
        mock_request.method = "GET"

        result = await handler.handle_request(mock_request)
        assert result["status_code"] == 200
        metabase = next(p for p in result["body"]["platforms"] if p["id"] == "metabase")
        assert metabase["connected"] is True

        # Disconnect platform
        mock_request.path = "/api/v1/analytics/metabase"
        mock_request.method = "DELETE"

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(user_id="test")
            with patch.object(handler, "check_permission"):
                result = await handler.handle_request(mock_request)

        assert result["status_code"] == 200
        assert "metabase" not in _platform_credentials
