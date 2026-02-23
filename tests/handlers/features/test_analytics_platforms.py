"""Tests for analytics platforms handler.

Tests the analytics platform API endpoints including:
- GET    /api/v1/analytics/platforms            - List connected platforms
- POST   /api/v1/analytics/connect              - Connect a platform
- DELETE /api/v1/analytics/{platform}           - Disconnect platform
- GET    /api/v1/analytics/dashboards           - List dashboards (cross-platform)
- GET    /api/v1/analytics/{platform}/dashboards - Platform dashboards
- GET    /api/v1/analytics/{platform}/dashboards/{dashboard_id} - Dashboard detail
- POST   /api/v1/analytics/query                - Execute unified query
- GET    /api/v1/analytics/reports              - Get available reports
- POST   /api/v1/analytics/reports/generate     - Generate custom report
- GET    /api/v1/analytics/metrics              - Cross-platform metrics overview
- GET    /api/v1/analytics/realtime             - Real-time metrics (GA4)
- GET    /api/v1/analytics/{platform}/events    - Platform events
- GET    /api/v1/analytics/{platform}/funnels   - Funnel analysis
- GET    /api/v1/analytics/{platform}/retention - Retention analysis
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.analytics_platforms import (
    AnalyticsPlatformsHandler,
    SUPPORTED_PLATFORMS,
    _platform_credentials,
    _platform_connectors,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class MockRequest:
    """Mock HTTP request for testing the analytics platforms handler."""

    path: str = "/api/v1/analytics/platforms"
    method: str = "GET"
    query: dict[str, Any] = field(default_factory=dict)
    _body: dict[str, Any] | None = None

    async def json(self) -> dict[str, Any]:
        return self._body or {}


def _status(result: dict[str, Any]) -> int:
    """Extract status code from handler response dict."""
    return result.get("status_code", 0)


def _body(result: dict[str, Any]) -> dict[str, Any]:
    """Extract body from handler response dict."""
    return result.get("body", {})


def _error(result: dict[str, Any]) -> str:
    """Extract error message from handler response dict."""
    return _body(result).get("error", "")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create an AnalyticsPlatformsHandler instance with empty context."""
    return AnalyticsPlatformsHandler(server_context={})


@pytest.fixture(autouse=True)
def reset_platform_state():
    """Reset global platform state before/after each test."""
    _platform_credentials.clear()
    _platform_connectors.clear()
    yield
    _platform_credentials.clear()
    _platform_connectors.clear()


@pytest.fixture(autouse=True)
def reset_rate_limiters():
    """Reset rate limiter state between tests."""
    try:
        from aragora.server.middleware.rate_limit.registry import (
            reset_rate_limiters as _reset,
        )

        _reset()
    except ImportError:
        pass
    yield
    try:
        from aragora.server.middleware.rate_limit.registry import (
            reset_rate_limiters as _reset,
        )

        _reset()
    except ImportError:
        pass


@pytest.fixture
def connected_metabase():
    """Pre-connect metabase platform."""
    _platform_credentials["metabase"] = {
        "credentials": {
            "base_url": "https://metabase.example.com",
            "username": "admin",
            "password": "secret",
        },
        "connected_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def connected_ga():
    """Pre-connect Google Analytics platform."""
    _platform_credentials["google_analytics"] = {
        "credentials": {
            "property_id": "123456789",
            "credentials_json": '{"type": "service_account"}',
        },
        "connected_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def connected_mixpanel():
    """Pre-connect Mixpanel platform."""
    _platform_credentials["mixpanel"] = {
        "credentials": {
            "project_id": "proj-123",
            "api_secret": "secret-abc",
        },
        "connected_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def mock_metabase_connector():
    """Create a mock Metabase connector."""
    connector = AsyncMock()
    connector.get_dashboards = AsyncMock(return_value=[])
    connector.get_dashboard = AsyncMock()
    connector.get_dashboard_cards = AsyncMock(return_value=[])
    connector.execute_query = AsyncMock()
    connector.close = AsyncMock()
    return connector


@pytest.fixture
def mock_ga_connector():
    """Create a mock Google Analytics connector."""
    connector = AsyncMock()
    connector.get_report = AsyncMock()
    connector.get_realtime_report = AsyncMock()
    connector.close = AsyncMock()
    return connector


@pytest.fixture
def mock_mixpanel_connector():
    """Create a mock Mixpanel connector."""
    connector = AsyncMock()
    connector.get_insights = AsyncMock()
    connector.get_events = AsyncMock(return_value=[])
    connector.get_funnel = AsyncMock()
    connector.get_retention = AsyncMock()
    connector.close = AsyncMock()
    return connector


# ---------------------------------------------------------------------------
# can_handle() Routing
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Test can_handle routing for all analytics paths."""

    def test_platforms_path(self, handler):
        assert handler.can_handle("/api/v1/analytics/platforms")

    def test_connect_path(self, handler):
        assert handler.can_handle("/api/v1/analytics/connect", "POST")

    def test_disconnect_path(self, handler):
        assert handler.can_handle("/api/v1/analytics/metabase", "DELETE")

    def test_dashboards_path(self, handler):
        assert handler.can_handle("/api/v1/analytics/dashboards")

    def test_platform_dashboards_path(self, handler):
        assert handler.can_handle("/api/v1/analytics/metabase/dashboards")

    def test_query_path(self, handler):
        assert handler.can_handle("/api/v1/analytics/query", "POST")

    def test_reports_path(self, handler):
        assert handler.can_handle("/api/v1/analytics/reports")

    def test_reports_generate_path(self, handler):
        assert handler.can_handle("/api/v1/analytics/reports/generate", "POST")

    def test_metrics_path(self, handler):
        assert handler.can_handle("/api/v1/analytics/metrics")

    def test_realtime_path(self, handler):
        assert handler.can_handle("/api/v1/analytics/realtime")

    def test_events_path(self, handler):
        assert handler.can_handle("/api/v1/analytics/mixpanel/events")

    def test_funnels_path(self, handler):
        assert handler.can_handle("/api/v1/analytics/mixpanel/funnels")

    def test_retention_path(self, handler):
        assert handler.can_handle("/api/v1/analytics/mixpanel/retention")

    def test_rejects_non_analytics_path(self, handler):
        assert not handler.can_handle("/api/v1/debates")

    def test_rejects_other_handler_path(self, handler):
        assert not handler.can_handle("/api/v1/users/me")

    def test_rejects_root(self, handler):
        assert not handler.can_handle("/")

    def test_routes_list_is_populated(self, handler):
        assert len(handler.ROUTES) >= 10


# ---------------------------------------------------------------------------
# GET /api/v1/analytics/platforms
# ---------------------------------------------------------------------------


class TestListPlatforms:
    """Test listing supported platforms."""

    @pytest.mark.asyncio
    async def test_list_platforms_success(self, handler):
        req = MockRequest(path="/api/v1/analytics/platforms", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert "platforms" in body
        assert len(body["platforms"]) == 3  # metabase, google_analytics, mixpanel

    @pytest.mark.asyncio
    async def test_list_platforms_none_connected(self, handler):
        req = MockRequest(path="/api/v1/analytics/platforms", method="GET")
        result = await handler.handle_request(req)
        body = _body(result)
        assert body["connected_count"] == 0
        for p in body["platforms"]:
            assert p["connected"] is False

    @pytest.mark.asyncio
    async def test_list_platforms_one_connected(self, handler, connected_metabase):
        req = MockRequest(path="/api/v1/analytics/platforms", method="GET")
        result = await handler.handle_request(req)
        body = _body(result)
        assert body["connected_count"] == 1

    @pytest.mark.asyncio
    async def test_list_platforms_all_connected(
        self, handler, connected_metabase, connected_ga, connected_mixpanel
    ):
        req = MockRequest(path="/api/v1/analytics/platforms", method="GET")
        result = await handler.handle_request(req)
        body = _body(result)
        assert body["connected_count"] == 3

    @pytest.mark.asyncio
    async def test_platform_metadata_present(self, handler):
        req = MockRequest(path="/api/v1/analytics/platforms", method="GET")
        result = await handler.handle_request(req)
        platforms = _body(result)["platforms"]
        for p in platforms:
            assert "id" in p
            assert "name" in p
            assert "description" in p
            assert "features" in p
            assert isinstance(p["features"], list)

    @pytest.mark.asyncio
    async def test_platform_connected_at_when_connected(self, handler, connected_metabase):
        req = MockRequest(path="/api/v1/analytics/platforms", method="GET")
        result = await handler.handle_request(req)
        platforms = _body(result)["platforms"]
        metabase_entry = next(p for p in platforms if p["id"] == "metabase")
        assert metabase_entry["connected"] is True
        assert metabase_entry["connected_at"] is not None

    @pytest.mark.asyncio
    async def test_platform_connected_at_none_when_disconnected(self, handler):
        req = MockRequest(path="/api/v1/analytics/platforms", method="GET")
        result = await handler.handle_request(req)
        platforms = _body(result)["platforms"]
        for p in platforms:
            assert p["connected_at"] is None


# ---------------------------------------------------------------------------
# POST /api/v1/analytics/connect
# ---------------------------------------------------------------------------


class TestConnectPlatform:
    """Test connecting an analytics platform."""

    @pytest.mark.asyncio
    async def test_connect_metabase_success(self, handler):
        req = MockRequest(path="/api/v1/analytics/connect", method="POST")
        body_data = {
            "platform": "metabase",
            "credentials": {
                "base_url": "https://metabase.example.com",
                "username": "admin",
                "password": "secret",
            },
        }
        with patch.object(handler, "_get_json_body", return_value=body_data):
            with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
                result = await handler.handle_request(req)
        assert _status(result) == 200
        assert "connected_at" in _body(result)
        assert _body(result)["platform"] == "metabase"

    @pytest.mark.asyncio
    async def test_connect_google_analytics_success(self, handler):
        req = MockRequest(path="/api/v1/analytics/connect", method="POST")
        body_data = {
            "platform": "google_analytics",
            "credentials": {
                "property_id": "123456",
                "credentials_json": '{"type": "service_account"}',
            },
        }
        with patch.object(handler, "_get_json_body", return_value=body_data):
            with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
                result = await handler.handle_request(req)
        assert _status(result) == 200
        assert _body(result)["platform"] == "google_analytics"

    @pytest.mark.asyncio
    async def test_connect_mixpanel_success(self, handler):
        req = MockRequest(path="/api/v1/analytics/connect", method="POST")
        body_data = {
            "platform": "mixpanel",
            "credentials": {
                "project_id": "proj-123",
                "api_secret": "secret-abc",
            },
        }
        with patch.object(handler, "_get_json_body", return_value=body_data):
            with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
                result = await handler.handle_request(req)
        assert _status(result) == 200
        assert _body(result)["platform"] == "mixpanel"

    @pytest.mark.asyncio
    async def test_connect_missing_platform(self, handler):
        req = MockRequest(path="/api/v1/analytics/connect", method="POST")
        with patch.object(handler, "_get_json_body", return_value={}):
            result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "required" in _error(result).lower()

    @pytest.mark.asyncio
    async def test_connect_unsupported_platform(self, handler):
        req = MockRequest(path="/api/v1/analytics/connect", method="POST")
        body_data = {"platform": "unknown_platform", "credentials": {"key": "val"}}
        with patch.object(handler, "_get_json_body", return_value=body_data):
            result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "Unsupported" in _error(result) or "unsupported" in _error(result).lower()

    @pytest.mark.asyncio
    async def test_connect_missing_credentials(self, handler):
        req = MockRequest(path="/api/v1/analytics/connect", method="POST")
        body_data = {"platform": "metabase", "credentials": {}}
        with patch.object(handler, "_get_json_body", return_value=body_data):
            result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_connect_partial_credentials(self, handler):
        req = MockRequest(path="/api/v1/analytics/connect", method="POST")
        body_data = {
            "platform": "metabase",
            "credentials": {"base_url": "https://metabase.example.com"},
        }
        with patch.object(handler, "_get_json_body", return_value=body_data):
            result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "Missing required credentials" in _error(result)

    @pytest.mark.asyncio
    async def test_connect_invalid_body(self, handler):
        req = MockRequest(path="/api/v1/analytics/connect", method="POST")
        with patch.object(handler, "_get_json_body", side_effect=ValueError("bad json")):
            result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_connect_stores_credentials(self, handler):
        req = MockRequest(path="/api/v1/analytics/connect", method="POST")
        body_data = {
            "platform": "metabase",
            "credentials": {
                "base_url": "https://metabase.example.com",
                "username": "admin",
                "password": "secret",
            },
        }
        with patch.object(handler, "_get_json_body", return_value=body_data):
            with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
                await handler.handle_request(req)
        assert "metabase" in _platform_credentials

    @pytest.mark.asyncio
    async def test_connect_connector_initialization_error(self, handler):
        """Connector init failure should not prevent connection."""
        req = MockRequest(path="/api/v1/analytics/connect", method="POST")
        body_data = {
            "platform": "metabase",
            "credentials": {
                "base_url": "https://metabase.example.com",
                "username": "admin",
                "password": "secret",
            },
        }
        with patch.object(handler, "_get_json_body", return_value=body_data):
            with patch.object(
                handler,
                "_get_connector",
                new_callable=AsyncMock,
                side_effect=ConnectionError("fail"),
            ):
                result = await handler.handle_request(req)
        # Connection still succeeds even if connector init fails
        assert _status(result) == 200
        assert "metabase" in _platform_credentials


# ---------------------------------------------------------------------------
# DELETE /api/v1/analytics/{platform}
# ---------------------------------------------------------------------------


class TestDisconnectPlatform:
    """Test disconnecting an analytics platform."""

    @pytest.mark.asyncio
    async def test_disconnect_metabase_success(self, handler, connected_metabase):
        req = MockRequest(path="/api/v1/analytics/metabase", method="DELETE")
        result = await handler.handle_request(req)
        assert _status(result) == 200
        assert "metabase" not in _platform_credentials

    @pytest.mark.asyncio
    async def test_disconnect_not_connected(self, handler):
        req = MockRequest(path="/api/v1/analytics/metabase", method="DELETE")
        result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_disconnect_closes_connector(self, handler, connected_metabase):
        mock_conn = AsyncMock()
        mock_conn.close = AsyncMock()
        _platform_connectors["metabase"] = mock_conn

        req = MockRequest(path="/api/v1/analytics/metabase", method="DELETE")
        result = await handler.handle_request(req)
        assert _status(result) == 200
        mock_conn.close.assert_awaited_once()
        assert "metabase" not in _platform_connectors

    @pytest.mark.asyncio
    async def test_disconnect_google_analytics(self, handler, connected_ga):
        req = MockRequest(path="/api/v1/analytics/google_analytics", method="DELETE")
        result = await handler.handle_request(req)
        assert _status(result) == 200
        assert "google_analytics" not in _platform_credentials

    @pytest.mark.asyncio
    async def test_disconnect_mixpanel(self, handler, connected_mixpanel):
        req = MockRequest(path="/api/v1/analytics/mixpanel", method="DELETE")
        result = await handler.handle_request(req)
        assert _status(result) == 200
        assert "mixpanel" not in _platform_credentials


# ---------------------------------------------------------------------------
# GET /api/v1/analytics/dashboards (cross-platform)
# ---------------------------------------------------------------------------


class TestListAllDashboards:
    """Test listing dashboards across all connected platforms."""

    @pytest.mark.asyncio
    async def test_list_all_dashboards_no_platforms(self, handler):
        req = MockRequest(path="/api/v1/analytics/dashboards", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["dashboards"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_all_dashboards_with_metabase(
        self, handler, connected_metabase, mock_metabase_connector
    ):
        dashboard_mock = MagicMock()
        dashboard_mock.id = 1
        dashboard_mock.name = "Sales Dashboard"
        dashboard_mock.description = "Monthly sales overview"
        dashboard_mock.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        dashboard_mock.dashcards = [MagicMock(), MagicMock()]

        mock_metabase_connector.get_dashboards.return_value = [dashboard_mock]
        _platform_connectors["metabase"] = mock_metabase_connector

        req = MockRequest(path="/api/v1/analytics/dashboards", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert body["dashboards"][0]["name"] == "Sales Dashboard"
        assert body["dashboards"][0]["platform"] == "metabase"

    @pytest.mark.asyncio
    async def test_list_all_dashboards_ga_returns_empty(
        self, handler, connected_ga, mock_ga_connector
    ):
        _platform_connectors["google_analytics"] = mock_ga_connector
        req = MockRequest(path="/api/v1/analytics/dashboards", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["dashboards"] == []

    @pytest.mark.asyncio
    async def test_list_all_dashboards_error_platform_skipped(self, handler, connected_metabase):
        """If a platform raises an error, it is skipped gracefully."""
        with patch.object(
            handler,
            "_fetch_platform_dashboards",
            new_callable=AsyncMock,
            side_effect=ConnectionError("timeout"),
        ):
            req = MockRequest(path="/api/v1/analytics/dashboards", method="GET")
            result = await handler.handle_request(req)
        assert _status(result) == 200
        assert _body(result)["total"] == 0


# ---------------------------------------------------------------------------
# GET /api/v1/analytics/{platform}/dashboards
# ---------------------------------------------------------------------------


class TestListPlatformDashboards:
    """Test listing dashboards for a specific platform."""

    @pytest.mark.asyncio
    async def test_list_metabase_dashboards(
        self, handler, connected_metabase, mock_metabase_connector
    ):
        mock_metabase_connector.get_dashboards.return_value = []
        _platform_connectors["metabase"] = mock_metabase_connector

        req = MockRequest(path="/api/v1/analytics/metabase/dashboards", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["platform"] == "metabase"
        assert body["dashboards"] == []

    @pytest.mark.asyncio
    async def test_list_dashboards_platform_not_connected(self, handler):
        req = MockRequest(path="/api/v1/analytics/metabase/dashboards", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 404


# ---------------------------------------------------------------------------
# GET /api/v1/analytics/{platform}/dashboards/{dashboard_id}
# ---------------------------------------------------------------------------


class TestGetDashboard:
    """Test getting a specific dashboard."""

    @pytest.mark.asyncio
    async def test_get_metabase_dashboard(
        self, handler, connected_metabase, mock_metabase_connector
    ):
        dashboard_mock = MagicMock()
        dashboard_mock.id = 42
        dashboard_mock.name = "Revenue Dashboard"
        dashboard_mock.description = "Revenue analysis"
        dashboard_mock.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        dashboard_mock.dashcards = []

        card_mock = MagicMock()
        card_mock.id = 1
        card_mock.name = "Revenue Chart"
        card_mock.description = "Revenue over time"
        card_mock.display = "line"
        card_mock.query_type = "native"

        mock_metabase_connector.get_dashboard.return_value = dashboard_mock
        mock_metabase_connector.get_dashboard_cards.return_value = [card_mock]
        _platform_connectors["metabase"] = mock_metabase_connector

        req = MockRequest(path="/api/v1/analytics/metabase/dashboards/42", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["name"] == "Revenue Dashboard"
        assert len(body["cards"]) == 1
        assert body["cards"][0]["name"] == "Revenue Chart"

    @pytest.mark.asyncio
    async def test_get_dashboard_platform_not_connected(self, handler):
        req = MockRequest(path="/api/v1/analytics/metabase/dashboards/42", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_dashboard_connector_unavailable(self, handler, connected_metabase):
        with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
            req = MockRequest(path="/api/v1/analytics/metabase/dashboards/42", method="GET")
            result = await handler.handle_request(req)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_get_dashboard_not_found(
        self, handler, connected_metabase, mock_metabase_connector
    ):
        mock_metabase_connector.get_dashboard.side_effect = ValueError("not found")
        _platform_connectors["metabase"] = mock_metabase_connector

        req = MockRequest(path="/api/v1/analytics/metabase/dashboards/999", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_dashboard_unsupported_platform(
        self, handler, connected_ga, mock_ga_connector
    ):
        """GA doesn't support individual dashboard fetching."""
        _platform_connectors["google_analytics"] = mock_ga_connector
        req = MockRequest(path="/api/v1/analytics/google_analytics/dashboards/1", method="GET")
        result = await handler.handle_request(req)
        # GA path falls through to "Unsupported platform" since only metabase is handled
        assert _status(result) == 400


# ---------------------------------------------------------------------------
# POST /api/v1/analytics/query
# ---------------------------------------------------------------------------


class TestExecuteQuery:
    """Test executing queries on analytics platforms."""

    @pytest.mark.asyncio
    async def test_query_metabase_success(
        self, handler, connected_metabase, mock_metabase_connector
    ):
        query_result = MagicMock()
        query_result.columns = ["id", "name", "total"]
        query_result.rows = [[1, "Product A", 100]]
        query_result.row_count = 1
        mock_metabase_connector.execute_query.return_value = query_result
        _platform_connectors["metabase"] = mock_metabase_connector

        req = MockRequest(path="/api/v1/analytics/query", method="POST")
        body_data = {
            "platform": "metabase",
            "query": "SELECT * FROM orders LIMIT 10",
        }
        with patch.object(handler, "_get_json_body", return_value=body_data):
            result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["platform"] == "metabase"
        assert body["columns"] == ["id", "name", "total"]
        assert body["row_count"] == 1

    @pytest.mark.asyncio
    async def test_query_metabase_with_database_id(
        self, handler, connected_metabase, mock_metabase_connector
    ):
        query_result = MagicMock()
        query_result.columns = ["count"]
        query_result.rows = [[42]]
        query_result.row_count = 1
        mock_metabase_connector.execute_query.return_value = query_result
        _platform_connectors["metabase"] = mock_metabase_connector

        req = MockRequest(path="/api/v1/analytics/query", method="POST")
        body_data = {
            "platform": "metabase",
            "query": "SELECT COUNT(*) FROM users",
            "database_id": 5,
        }
        with patch.object(handler, "_get_json_body", return_value=body_data):
            result = await handler.handle_request(req)
        assert _status(result) == 200
        mock_metabase_connector.execute_query.assert_awaited_once_with(
            5, "SELECT COUNT(*) FROM users"
        )

    @pytest.mark.asyncio
    async def test_query_google_analytics(self, handler, connected_ga, mock_ga_connector):
        report_mock = MagicMock()
        report_mock.dimension_headers = []
        report_mock.metric_headers = []
        report_mock.rows = []
        report_mock.row_count = 0
        mock_ga_connector.get_report.return_value = report_mock
        _platform_connectors["google_analytics"] = mock_ga_connector

        req = MockRequest(path="/api/v1/analytics/query", method="POST")
        body_data = {
            "platform": "google_analytics",
            "query": "report",
            "metrics": ["sessions"],
            "dimensions": ["date"],
        }
        with patch.object(handler, "_get_json_body", return_value=body_data):
            result = await handler.handle_request(req)
        assert _status(result) == 200
        assert _body(result)["platform"] == "google_analytics"

    @pytest.mark.asyncio
    async def test_query_mixpanel_with_event(
        self, handler, connected_mixpanel, mock_mixpanel_connector
    ):
        insight_mock = MagicMock()
        insight_mock.total = 500
        insight_mock.series = [100, 200, 200]
        insight_mock.breakdown = {}
        mock_mixpanel_connector.get_insights.return_value = insight_mock
        _platform_connectors["mixpanel"] = mock_mixpanel_connector

        req = MockRequest(path="/api/v1/analytics/query", method="POST")
        body_data = {
            "platform": "mixpanel",
            "query": "insights",
            "event": "signup",
            "from_date": "2026-01-01",
            "to_date": "2026-01-31",
        }
        with patch.object(handler, "_get_json_body", return_value=body_data):
            result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["platform"] == "mixpanel"
        assert body["result"]["total"] == 500

    @pytest.mark.asyncio
    async def test_query_missing_platform(self, handler):
        req = MockRequest(path="/api/v1/analytics/query", method="POST")
        with patch.object(handler, "_get_json_body", return_value={"query": "SELECT 1"}):
            result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_query_platform_not_connected(self, handler):
        req = MockRequest(path="/api/v1/analytics/query", method="POST")
        body_data = {"platform": "metabase", "query": "SELECT 1"}
        with patch.object(handler, "_get_json_body", return_value=body_data):
            result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_query_missing_query(self, handler, connected_metabase):
        req = MockRequest(path="/api/v1/analytics/query", method="POST")
        body_data = {"platform": "metabase"}
        with patch.object(handler, "_get_json_body", return_value=body_data):
            result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_query_connector_unavailable(self, handler, connected_metabase):
        req = MockRequest(path="/api/v1/analytics/query", method="POST")
        body_data = {"platform": "metabase", "query": "SELECT 1"}
        with patch.object(handler, "_get_json_body", return_value=body_data):
            with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
                result = await handler.handle_request(req)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_query_execution_error(
        self, handler, connected_metabase, mock_metabase_connector
    ):
        mock_metabase_connector.execute_query.side_effect = ConnectionError("timeout")
        _platform_connectors["metabase"] = mock_metabase_connector

        req = MockRequest(path="/api/v1/analytics/query", method="POST")
        body_data = {"platform": "metabase", "query": "SELECT 1"}
        with patch.object(handler, "_get_json_body", return_value=body_data):
            result = await handler.handle_request(req)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_query_invalid_body(self, handler):
        req = MockRequest(path="/api/v1/analytics/query", method="POST")
        with patch.object(handler, "_get_json_body", side_effect=ValueError("bad")):
            result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_query_rows_limited_to_1000(
        self, handler, connected_metabase, mock_metabase_connector
    ):
        """Metabase query results should be limited to 1000 rows."""
        big_rows = [[i] for i in range(2000)]
        query_result = MagicMock()
        query_result.columns = ["id"]
        query_result.rows = big_rows
        query_result.row_count = 2000
        mock_metabase_connector.execute_query.return_value = query_result
        _platform_connectors["metabase"] = mock_metabase_connector

        req = MockRequest(path="/api/v1/analytics/query", method="POST")
        body_data = {"platform": "metabase", "query": "SELECT * FROM big_table"}
        with patch.object(handler, "_get_json_body", return_value=body_data):
            result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert len(body["rows"]) == 1000
        assert body["row_count"] == 2000


# ---------------------------------------------------------------------------
# GET /api/v1/analytics/reports
# ---------------------------------------------------------------------------


class TestListReports:
    """Test listing available reports."""

    @pytest.mark.asyncio
    async def test_list_all_reports(self, handler):
        req = MockRequest(path="/api/v1/analytics/reports", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert "reports" in body
        assert body["total"] == 5

    @pytest.mark.asyncio
    async def test_list_reports_filtered_by_platform(self, handler):
        req = MockRequest(
            path="/api/v1/analytics/reports",
            method="GET",
            query={"platform": "mixpanel"},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        for report in body["reports"]:
            assert "mixpanel" in report["platforms"]

    @pytest.mark.asyncio
    async def test_list_reports_filtered_by_ga(self, handler):
        req = MockRequest(
            path="/api/v1/analytics/reports",
            method="GET",
            query={"platform": "google_analytics"},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        for report in body["reports"]:
            assert "google_analytics" in report["platforms"]

    @pytest.mark.asyncio
    async def test_list_reports_unknown_platform_returns_empty(self, handler):
        req = MockRequest(
            path="/api/v1/analytics/reports",
            method="GET",
            query={"platform": "unknown_platform"},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_report_structure(self, handler):
        req = MockRequest(path="/api/v1/analytics/reports", method="GET")
        result = await handler.handle_request(req)
        body = _body(result)
        for report in body["reports"]:
            assert "id" in report
            assert "name" in report
            assert "description" in report
            assert "platforms" in report
            assert "metrics" in report


# ---------------------------------------------------------------------------
# POST /api/v1/analytics/reports/generate
# ---------------------------------------------------------------------------


class TestGenerateReport:
    """Test generating custom reports."""

    @pytest.mark.asyncio
    async def test_generate_report_success(self, handler, connected_ga):
        report_mock = MagicMock()
        report_mock.dimension_headers = []
        report_mock.metric_headers = []
        report_mock.rows = []
        report_mock.row_count = 0

        req = MockRequest(path="/api/v1/analytics/reports/generate", method="POST")
        body_data = {
            "type": "traffic_overview",
            "platforms": ["google_analytics"],
            "days": 7,
        }
        with patch.object(handler, "_get_json_body", return_value=body_data):
            with patch.object(
                handler,
                "_fetch_report_data",
                new_callable=AsyncMock,
                return_value={"sessions": 100, "users": 50},
            ):
                result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert "report_id" in body
        assert "date_range" in body
        assert body["type"] == "traffic_overview"

    @pytest.mark.asyncio
    async def test_generate_report_default_type(self, handler):
        req = MockRequest(path="/api/v1/analytics/reports/generate", method="POST")
        with patch.object(handler, "_get_json_body", return_value={}):
            result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["type"] == "traffic_overview"

    @pytest.mark.asyncio
    async def test_generate_report_fetch_error(self, handler, connected_ga):
        req = MockRequest(path="/api/v1/analytics/reports/generate", method="POST")
        body_data = {"platforms": ["google_analytics"]}
        with patch.object(handler, "_get_json_body", return_value=body_data):
            with patch.object(
                handler,
                "_fetch_report_data",
                new_callable=AsyncMock,
                side_effect=ConnectionError("timeout"),
            ):
                result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["platforms"]["google_analytics"]["error"] == "Failed to fetch report data"

    @pytest.mark.asyncio
    async def test_generate_report_invalid_body(self, handler):
        req = MockRequest(path="/api/v1/analytics/reports/generate", method="POST")
        with patch.object(handler, "_get_json_body", side_effect=ValueError("bad")):
            result = await handler.handle_request(req)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_generate_report_skips_disconnected_platforms(self, handler):
        req = MockRequest(path="/api/v1/analytics/reports/generate", method="POST")
        body_data = {"platforms": ["metabase"]}  # not connected
        with patch.object(handler, "_get_json_body", return_value=body_data):
            result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        # metabase not in platforms because it is not connected
        assert "metabase" not in body["platforms"]


# ---------------------------------------------------------------------------
# GET /api/v1/analytics/metrics
# ---------------------------------------------------------------------------


class TestCrossPlatformMetrics:
    """Test cross-platform metrics overview."""

    @pytest.mark.asyncio
    async def test_metrics_no_connected_platforms(self, handler):
        req = MockRequest(path="/api/v1/analytics/metrics", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["summary"]["total_users"] == 0
        assert body["summary"]["total_sessions"] == 0
        assert body["summary"]["total_events"] == 0

    @pytest.mark.asyncio
    async def test_metrics_with_ga(self, handler, connected_ga, mock_ga_connector):
        # Build a GA report with metric values
        row_mock = MagicMock()
        val1 = MagicMock()
        val1.value = "100"
        val2 = MagicMock()
        val2.value = "200"
        val3 = MagicMock()
        val3.value = "50"
        row_mock.metric_values = [val1, val2, val3]

        report_mock = MagicMock()
        report_mock.rows = [row_mock]
        header1 = MagicMock()
        header1.name = "totalUsers"
        header2 = MagicMock()
        header2.name = "sessions"
        header3 = MagicMock()
        header3.name = "eventCount"
        report_mock.metric_headers = [header1, header2, header3]

        mock_ga_connector.get_report.return_value = report_mock
        _platform_connectors["google_analytics"] = mock_ga_connector

        req = MockRequest(path="/api/v1/analytics/metrics", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["summary"]["total_users"] == 100
        assert body["summary"]["total_sessions"] == 200
        assert body["summary"]["total_events"] == 50

    @pytest.mark.asyncio
    async def test_metrics_with_custom_days(self, handler):
        req = MockRequest(
            path="/api/v1/analytics/metrics",
            method="GET",
            query={"days": "30"},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert "date_range" in body

    @pytest.mark.asyncio
    async def test_metrics_platform_error(self, handler, connected_ga):
        with patch.object(
            handler,
            "_get_connector",
            new_callable=AsyncMock,
            side_effect=ConnectionError("fail"),
        ):
            req = MockRequest(path="/api/v1/analytics/metrics", method="GET")
            result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert "error" in body["platforms"].get("google_analytics", {})

    @pytest.mark.asyncio
    async def test_metrics_with_mixpanel(
        self, handler, connected_mixpanel, mock_mixpanel_connector
    ):
        insight_mock = MagicMock()
        insight_mock.total = 300
        mock_mixpanel_connector.get_insights.return_value = insight_mock
        _platform_connectors["mixpanel"] = mock_mixpanel_connector

        req = MockRequest(path="/api/v1/analytics/metrics", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["platforms"]["mixpanel"]["sessions"] == 300


# ---------------------------------------------------------------------------
# GET /api/v1/analytics/realtime
# ---------------------------------------------------------------------------


class TestRealtimeMetrics:
    """Test real-time metrics endpoint (GA4)."""

    @pytest.mark.asyncio
    async def test_realtime_ga_not_connected(self, handler):
        req = MockRequest(path="/api/v1/analytics/realtime", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_realtime_success(self, handler, connected_ga, mock_ga_connector):
        realtime_mock = MagicMock()
        realtime_mock.active_users = 42
        realtime_mock.rows = []
        mock_ga_connector.get_realtime_report.return_value = realtime_mock
        _platform_connectors["google_analytics"] = mock_ga_connector

        req = MockRequest(path="/api/v1/analytics/realtime", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["platform"] == "google_analytics"
        assert body["realtime"]["active_users"] == 42
        assert "timestamp" in body

    @pytest.mark.asyncio
    async def test_realtime_connector_unavailable(self, handler, connected_ga):
        with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
            req = MockRequest(path="/api/v1/analytics/realtime", method="GET")
            result = await handler.handle_request(req)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_realtime_api_error(self, handler, connected_ga, mock_ga_connector):
        mock_ga_connector.get_realtime_report.side_effect = ConnectionError("fail")
        _platform_connectors["google_analytics"] = mock_ga_connector

        req = MockRequest(path="/api/v1/analytics/realtime", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_realtime_with_rows(self, handler, connected_ga, mock_ga_connector):
        row_mock = MagicMock()
        dim_val = MagicMock()
        dim_val.value = "US"
        met_val = MagicMock()
        met_val.value = "15"
        row_mock.dimension_values = [dim_val]
        row_mock.metric_values = [met_val]

        realtime_mock = MagicMock()
        realtime_mock.active_users = 15
        realtime_mock.rows = [row_mock]
        mock_ga_connector.get_realtime_report.return_value = realtime_mock
        _platform_connectors["google_analytics"] = mock_ga_connector

        req = MockRequest(path="/api/v1/analytics/realtime", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert len(body["realtime"]["rows"]) == 1
        assert body["realtime"]["rows"][0]["dimensions"] == ["US"]
        assert body["realtime"]["rows"][0]["metrics"] == ["15"]


# ---------------------------------------------------------------------------
# GET /api/v1/analytics/{platform}/events
# ---------------------------------------------------------------------------


class TestGetEvents:
    """Test platform events endpoint."""

    @pytest.mark.asyncio
    async def test_events_mixpanel_success(
        self, handler, connected_mixpanel, mock_mixpanel_connector
    ):
        event_mock = MagicMock()
        event_mock.name = "signup"
        event_mock.distinct_id = "user-123"
        event_mock.timestamp = datetime(2026, 1, 15, tzinfo=timezone.utc)
        event_mock.properties = {"plan": "premium"}
        mock_mixpanel_connector.get_events.return_value = [event_mock]
        _platform_connectors["mixpanel"] = mock_mixpanel_connector

        req = MockRequest(path="/api/v1/analytics/mixpanel/events", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["platform"] == "mixpanel"
        assert len(body["events"]) == 1
        assert body["events"][0]["event_name"] == "signup"

    @pytest.mark.asyncio
    async def test_events_ga_success(self, handler, connected_ga, mock_ga_connector):
        report_mock = MagicMock()
        report_mock.dimension_headers = []
        report_mock.metric_headers = []
        report_mock.rows = []
        report_mock.row_count = 0
        mock_ga_connector.get_report.return_value = report_mock
        _platform_connectors["google_analytics"] = mock_ga_connector

        req = MockRequest(path="/api/v1/analytics/google_analytics/events", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 200
        assert _body(result)["platform"] == "google_analytics"

    @pytest.mark.asyncio
    async def test_events_platform_not_connected(self, handler):
        req = MockRequest(path="/api/v1/analytics/mixpanel/events", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_events_connector_unavailable(self, handler, connected_mixpanel):
        with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
            req = MockRequest(path="/api/v1/analytics/mixpanel/events", method="GET")
            result = await handler.handle_request(req)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_events_api_error(self, handler, connected_mixpanel, mock_mixpanel_connector):
        mock_mixpanel_connector.get_events.side_effect = ConnectionError("fail")
        _platform_connectors["mixpanel"] = mock_mixpanel_connector

        req = MockRequest(path="/api/v1/analytics/mixpanel/events", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_events_unsupported_platform(
        self, handler, connected_metabase, mock_metabase_connector
    ):
        _platform_connectors["metabase"] = mock_metabase_connector
        req = MockRequest(path="/api/v1/analytics/metabase/events", method="GET")
        result = await handler.handle_request(req)
        # metabase doesn't support events; falls through
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_events_with_query_params(
        self, handler, connected_mixpanel, mock_mixpanel_connector
    ):
        mock_mixpanel_connector.get_events.return_value = []
        _platform_connectors["mixpanel"] = mock_mixpanel_connector

        req = MockRequest(
            path="/api/v1/analytics/mixpanel/events",
            method="GET",
            query={"days": "14", "event": "purchase"},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200


# ---------------------------------------------------------------------------
# GET /api/v1/analytics/{platform}/funnels
# ---------------------------------------------------------------------------


class TestGetFunnels:
    """Test funnel analysis endpoint."""

    @pytest.mark.asyncio
    async def test_funnels_mixpanel_success(
        self, handler, connected_mixpanel, mock_mixpanel_connector
    ):
        funnel_mock = MagicMock()
        funnel_mock.steps = [
            {"name": "Visit", "count": 1000},
            {"name": "Signup", "count": 300},
        ]
        funnel_mock.overall_conversion = 0.30
        funnel_mock.from_date = "2026-01-01"
        funnel_mock.to_date = "2026-01-31"
        mock_mixpanel_connector.get_funnel.return_value = funnel_mock
        _platform_connectors["mixpanel"] = mock_mixpanel_connector

        req = MockRequest(
            path="/api/v1/analytics/mixpanel/funnels",
            method="GET",
            query={"funnel_id": "funnel-1"},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["platform"] == "mixpanel"
        assert body["funnel"]["conversion_rate"] == 0.30

    @pytest.mark.asyncio
    async def test_funnels_platform_not_connected(self, handler):
        req = MockRequest(
            path="/api/v1/analytics/mixpanel/funnels",
            method="GET",
            query={"funnel_id": "funnel-1"},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_funnels_non_mixpanel_platform(self, handler, connected_ga):
        req = MockRequest(
            path="/api/v1/analytics/google_analytics/funnels",
            method="GET",
            query={"funnel_id": "funnel-1"},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "only available for Mixpanel" in _error(result)

    @pytest.mark.asyncio
    async def test_funnels_missing_funnel_id(self, handler, connected_mixpanel):
        req = MockRequest(
            path="/api/v1/analytics/mixpanel/funnels",
            method="GET",
            query={},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "funnel_id" in _error(result)

    @pytest.mark.asyncio
    async def test_funnels_connector_unavailable(self, handler, connected_mixpanel):
        with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
            req = MockRequest(
                path="/api/v1/analytics/mixpanel/funnels",
                method="GET",
                query={"funnel_id": "funnel-1"},
            )
            result = await handler.handle_request(req)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_funnels_api_error(self, handler, connected_mixpanel, mock_mixpanel_connector):
        mock_mixpanel_connector.get_funnel.side_effect = ConnectionError("fail")
        _platform_connectors["mixpanel"] = mock_mixpanel_connector

        req = MockRequest(
            path="/api/v1/analytics/mixpanel/funnels",
            method="GET",
            query={"funnel_id": "funnel-1"},
        )
        result = await handler.handle_request(req)
        assert _status(result) == 500


# ---------------------------------------------------------------------------
# GET /api/v1/analytics/{platform}/retention
# ---------------------------------------------------------------------------


class TestGetRetention:
    """Test retention analysis endpoint."""

    @pytest.mark.asyncio
    async def test_retention_mixpanel_success(
        self, handler, connected_mixpanel, mock_mixpanel_connector
    ):
        retention_mock = MagicMock()
        retention_mock.cohorts = [{"date": "2026-01-01", "size": 100}]
        retention_mock.retention = [1.0, 0.8, 0.6]
        mock_mixpanel_connector.get_retention.return_value = retention_mock
        _platform_connectors["mixpanel"] = mock_mixpanel_connector

        req = MockRequest(path="/api/v1/analytics/mixpanel/retention", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 200
        body = _body(result)
        assert body["platform"] == "mixpanel"
        assert body["retention"]["cohorts"] == [{"date": "2026-01-01", "size": 100}]
        assert body["retention"]["retention_by_day"] == [1.0, 0.8, 0.6]

    @pytest.mark.asyncio
    async def test_retention_platform_not_connected(self, handler):
        req = MockRequest(path="/api/v1/analytics/mixpanel/retention", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_retention_non_mixpanel_platform(self, handler, connected_ga):
        req = MockRequest(path="/api/v1/analytics/google_analytics/retention", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 400
        assert "only available for Mixpanel" in _error(result)

    @pytest.mark.asyncio
    async def test_retention_connector_unavailable(self, handler, connected_mixpanel):
        with patch.object(handler, "_get_connector", new_callable=AsyncMock, return_value=None):
            req = MockRequest(path="/api/v1/analytics/mixpanel/retention", method="GET")
            result = await handler.handle_request(req)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_retention_api_error(self, handler, connected_mixpanel, mock_mixpanel_connector):
        mock_mixpanel_connector.get_retention.side_effect = ConnectionError("fail")
        _platform_connectors["mixpanel"] = mock_mixpanel_connector

        req = MockRequest(path="/api/v1/analytics/mixpanel/retention", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 500


# ---------------------------------------------------------------------------
# 404 / Unknown Routes
# ---------------------------------------------------------------------------


class TestUnknownRoutes:
    """Test that unknown routes return 404."""

    @pytest.mark.asyncio
    async def test_unknown_endpoint(self, handler):
        req = MockRequest(path="/api/v1/analytics/nonexistent", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_wrong_method_for_platforms(self, handler):
        req = MockRequest(path="/api/v1/analytics/platforms", method="POST")
        result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_wrong_method_for_connect(self, handler):
        req = MockRequest(path="/api/v1/analytics/connect", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_wrong_method_for_query(self, handler):
        req = MockRequest(path="/api/v1/analytics/query", method="GET")
        result = await handler.handle_request(req)
        assert _status(result) == 404


# ---------------------------------------------------------------------------
# Normalization Helpers
# ---------------------------------------------------------------------------


class TestNormalization:
    """Test data normalization helper methods."""

    def test_normalize_metabase_dashboard(self, handler):
        dashboard = MagicMock()
        dashboard.id = 5
        dashboard.name = "Test Dashboard"
        dashboard.description = "A test dashboard"
        dashboard.created_at = datetime(2026, 2, 1, tzinfo=timezone.utc)
        dashboard.dashcards = [MagicMock(), MagicMock(), MagicMock()]

        result = handler._normalize_metabase_dashboard(dashboard)
        assert result["id"] == "5"
        assert result["platform"] == "metabase"
        assert result["name"] == "Test Dashboard"
        assert result["description"] == "A test dashboard"
        assert result["url"] == "/dashboard/5"
        assert result["cards_count"] == 3
        assert result["created_at"] is not None

    def test_normalize_metabase_dashboard_no_created_at(self, handler):
        dashboard = MagicMock()
        dashboard.id = 1
        dashboard.name = "Dashboard"
        dashboard.description = None
        dashboard.created_at = None
        dashboard.dashcards = []

        result = handler._normalize_metabase_dashboard(dashboard)
        assert result["created_at"] is None
        assert result["cards_count"] == 0

    def test_normalize_metabase_card(self, handler):
        card = MagicMock()
        card.id = 10
        card.name = "Revenue Card"
        card.description = "Shows revenue"
        card.display = "bar"
        card.query_type = "native"

        result = handler._normalize_metabase_card(card)
        assert result["id"] == "10"
        assert result["name"] == "Revenue Card"
        assert result["display_type"] == "bar"
        assert result["query_type"] == "native"

    def test_normalize_metabase_card_enum_display(self, handler):
        card = MagicMock()
        card.id = 10
        card.name = "Card"
        card.description = None
        card.display.value = "line"
        card.query_type = "native"

        result = handler._normalize_metabase_card(card)
        assert result["display_type"] == "line"

    def test_normalize_ga_report_empty(self, handler):
        report = MagicMock(spec=[])
        result = handler._normalize_ga_report(report)
        assert result["dimensions"] == []
        assert result["metrics"] == []
        assert result["rows"] == []
        assert result["row_count"] == 0

    def test_normalize_ga_report_with_data(self, handler):
        dim_header = MagicMock()
        dim_header.name = "date"
        dim_header._mock_name = None
        met_header = MagicMock()
        met_header.name = "sessions"
        met_header._mock_name = None

        dim_val = MagicMock()
        dim_val.value = "2026-01-01"
        met_val = MagicMock()
        met_val.value = "100"

        row = MagicMock()
        row.dimension_values = [dim_val]
        row.metric_values = [met_val]

        report = MagicMock()
        report.dimension_headers = [dim_header]
        report.metric_headers = [met_header]
        report.rows = [row]
        report.row_count = 1

        result = handler._normalize_ga_report(report)
        assert result["row_count"] == 1
        assert len(result["rows"]) == 1
        assert result["rows"][0]["dimensions"] == ["2026-01-01"]
        assert result["rows"][0]["metrics"] == ["100"]

    def test_normalize_mixpanel_insight(self, handler):
        insight = MagicMock()
        insight.total = 500
        insight.series = [100, 200, 200]
        insight.breakdown = {"US": 300, "UK": 200}

        result = handler._normalize_mixpanel_insight(insight)
        assert result["total"] == 500
        assert result["series"] == [100, 200, 200]
        assert result["breakdown"] == {"US": 300, "UK": 200}

    def test_normalize_mixpanel_insight_empty(self, handler):
        insight = MagicMock(spec=[])
        result = handler._normalize_mixpanel_insight(insight)
        assert result["total"] == 0
        assert result["series"] == []
        assert result["breakdown"] == {}

    def test_normalize_mixpanel_event(self, handler):
        event = MagicMock()
        event.name = "purchase"
        event.distinct_id = "user-42"
        event.timestamp = datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc)
        event.properties = {"amount": 99.99}

        result = handler._normalize_mixpanel_event(event)
        assert result["event_name"] == "purchase"
        assert result["distinct_id"] == "user-42"
        assert result["timestamp"] is not None
        assert result["properties"] == {"amount": 99.99}

    def test_normalize_mixpanel_event_empty(self, handler):
        event = MagicMock(spec=[])
        result = handler._normalize_mixpanel_event(event)
        assert result["event_name"] == ""
        assert result["distinct_id"] == ""
        assert result["timestamp"] is None
        assert result["properties"] == {}

    def test_normalize_mixpanel_funnel(self, handler):
        funnel = MagicMock()
        funnel.steps = [{"name": "Visit", "count": 1000}]
        funnel.overall_conversion = 0.25
        funnel.from_date = "2026-01-01"
        funnel.to_date = "2026-01-31"

        result = handler._normalize_mixpanel_funnel(funnel)
        assert result["conversion_rate"] == 0.25
        assert result["date_range"]["start"] == "2026-01-01"
        assert result["date_range"]["end"] == "2026-01-31"

    def test_normalize_mixpanel_funnel_empty(self, handler):
        funnel = MagicMock(spec=[])
        result = handler._normalize_mixpanel_funnel(funnel)
        assert result["steps"] == []
        assert result["conversion_rate"] == 0

    def test_normalize_mixpanel_retention(self, handler):
        retention = MagicMock()
        retention.cohorts = [{"date": "2026-01-01", "size": 100}]
        retention.retention = [1.0, 0.5, 0.3]

        result = handler._normalize_mixpanel_retention(retention)
        assert result["cohorts"] == [{"date": "2026-01-01", "size": 100}]
        assert result["retention_by_day"] == [1.0, 0.5, 0.3]

    def test_normalize_mixpanel_retention_empty(self, handler):
        retention = MagicMock(spec=[])
        result = handler._normalize_mixpanel_retention(retention)
        assert result["cohorts"] == []
        assert result["retention_by_day"] == []

    def test_extract_ga_totals_no_rows(self, handler):
        report = MagicMock()
        report.rows = []
        result = handler._extract_ga_totals(report)
        assert result == {"users": 0, "sessions": 0, "events": 0}

    def test_extract_ga_totals_no_rows_attr(self, handler):
        report = MagicMock(spec=[])
        result = handler._extract_ga_totals(report)
        assert result == {"users": 0, "sessions": 0, "events": 0}


# ---------------------------------------------------------------------------
# Response Helpers
# ---------------------------------------------------------------------------


class TestResponseHelpers:
    """Test _json_response and _error_response helpers."""

    def test_json_response(self, handler):
        result = handler._json_response(200, {"key": "value"})
        assert result["status_code"] == 200
        assert result["headers"]["Content-Type"] == "application/json"
        assert result["body"]["key"] == "value"

    def test_error_response(self, handler):
        result = handler._error_response(404, "Not found")
        assert result["status_code"] == 404
        assert result["body"]["error"] == "Not found"


# ---------------------------------------------------------------------------
# Required Credentials
# ---------------------------------------------------------------------------


class TestRequiredCredentials:
    """Test _get_required_credentials helper."""

    def test_metabase_requires_base_url_username_password(self, handler):
        creds = handler._get_required_credentials("metabase")
        assert "base_url" in creds
        assert "username" in creds
        assert "password" in creds

    def test_google_analytics_requires_property_id(self, handler):
        creds = handler._get_required_credentials("google_analytics")
        assert "property_id" in creds
        assert "credentials_json" in creds

    def test_mixpanel_requires_project_id_api_secret(self, handler):
        creds = handler._get_required_credentials("mixpanel")
        assert "project_id" in creds
        assert "api_secret" in creds

    def test_unknown_platform_returns_empty(self, handler):
        creds = handler._get_required_credentials("unknown")
        assert creds == []


# ---------------------------------------------------------------------------
# UnifiedMetric and UnifiedDashboard dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Test the dataclass helper types."""

    def test_unified_metric_to_dict(self):
        from aragora.server.handlers.features.analytics_platforms import UnifiedMetric

        metric = UnifiedMetric(
            name="sessions",
            value=1000,
            platform="google_analytics",
            dimension="country",
            period="7d",
            change_percent=5.2,
        )
        d = metric.to_dict()
        assert d["name"] == "sessions"
        assert d["value"] == 1000
        assert d["platform"] == "google_analytics"
        assert d["dimension"] == "country"
        assert d["period"] == "7d"
        assert d["change_percent"] == 5.2

    def test_unified_metric_defaults(self):
        from aragora.server.handlers.features.analytics_platforms import UnifiedMetric

        metric = UnifiedMetric(name="users", value=50, platform="mixpanel")
        d = metric.to_dict()
        assert d["dimension"] is None
        assert d["period"] is None
        assert d["change_percent"] is None

    def test_unified_dashboard_to_dict(self):
        from aragora.server.handlers.features.analytics_platforms import UnifiedDashboard

        dashboard = UnifiedDashboard(
            id="dash-1",
            platform="metabase",
            name="Sales",
            description="Sales dashboard",
            url="/dashboard/1",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            cards_count=5,
        )
        d = dashboard.to_dict()
        assert d["id"] == "dash-1"
        assert d["platform"] == "metabase"
        assert d["name"] == "Sales"
        assert d["cards_count"] == 5
        assert d["created_at"] is not None

    def test_unified_dashboard_defaults(self):
        from aragora.server.handlers.features.analytics_platforms import UnifiedDashboard

        dashboard = UnifiedDashboard(
            id="dash-2",
            platform="ga",
            name="Traffic",
            description=None,
            url=None,
            created_at=None,
        )
        d = dashboard.to_dict()
        assert d["description"] is None
        assert d["url"] is None
        assert d["created_at"] is None
        assert d["cards_count"] == 0


# ---------------------------------------------------------------------------
# Supported Platforms Constants
# ---------------------------------------------------------------------------


class TestSupportedPlatforms:
    """Test the SUPPORTED_PLATFORMS constant."""

    def test_has_metabase(self):
        assert "metabase" in SUPPORTED_PLATFORMS

    def test_has_google_analytics(self):
        assert "google_analytics" in SUPPORTED_PLATFORMS

    def test_has_mixpanel(self):
        assert "mixpanel" in SUPPORTED_PLATFORMS

    def test_platform_has_name(self):
        for pid, meta in SUPPORTED_PLATFORMS.items():
            assert "name" in meta
            assert "description" in meta
            assert "features" in meta

    def test_exactly_three_platforms(self):
        assert len(SUPPORTED_PLATFORMS) == 3


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    """Test circuit breaker integration."""

    def test_get_analytics_circuit_breaker(self):
        from aragora.server.handlers.features.analytics_platforms import (
            get_analytics_circuit_breaker,
        )

        cb = get_analytics_circuit_breaker()
        assert cb is not None
        assert cb.name == "analytics_platforms_handler"

    def test_circuit_breaker_allows_execution(self):
        from aragora.server.handlers.features.analytics_platforms import (
            get_analytics_circuit_breaker,
        )

        cb = get_analytics_circuit_breaker()
        assert cb.can_execute()
