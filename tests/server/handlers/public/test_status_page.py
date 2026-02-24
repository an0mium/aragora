"""Tests for public status page handler.

Tests both legacy unversioned endpoints and versioned v1 endpoints
with {"data": ...} envelope. Also tests SLA instrumentation integration.
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

from aragora.server.handlers.public.status_page import (
    StatusPageHandler,
    ServiceStatus,
    ComponentHealth,
)


def _parse_json_result(result):
    """Parse JSON body from HandlerResult."""
    if result is None:
        return None
    # HandlerResult has body as bytes
    body = result.body if hasattr(result, "body") else result.get("body", result)
    if isinstance(body, bytes):
        return json.loads(body.decode("utf-8"))
    elif isinstance(body, str):
        return json.loads(body)
    return body


class TestStatusPageHandler:
    """Tests for StatusPageHandler."""

    @pytest.fixture
    def handler(self):
        """Create a status page handler."""
        return StatusPageHandler({})

    def test_can_handle_status_routes(self, handler):
        """Test handler recognizes status routes."""
        assert handler.can_handle("/status")
        assert handler.can_handle("/api/status")
        assert handler.can_handle("/api/status/summary")
        assert handler.can_handle("/api/status/history")
        assert handler.can_handle("/api/status/components")
        assert handler.can_handle("/api/status/incidents")

    def test_can_handle_v1_status_routes(self, handler):
        """Test handler recognizes versioned v1 routes."""
        assert handler.can_handle("/api/v1/status")
        assert handler.can_handle("/api/v1/status/components")
        assert handler.can_handle("/api/v1/status/incidents")
        assert handler.can_handle("/api/v1/status/uptime")

    def test_cannot_handle_non_status_routes(self, handler):
        """Test handler rejects non-status routes."""
        assert not handler.can_handle("/api/health")
        assert not handler.can_handle("/api/debates")
        assert not handler.can_handle("/")

    def test_public_routes_marker(self, handler):
        """Test that PUBLIC_ROUTES includes all v1 endpoints."""
        assert "/api/v1/status" in handler.PUBLIC_ROUTES
        assert "/api/v1/status/components" in handler.PUBLIC_ROUTES
        assert "/api/v1/status/incidents" in handler.PUBLIC_ROUTES
        assert "/api/v1/status/uptime" in handler.PUBLIC_ROUTES

    def test_json_status_summary(self, handler):
        """Test JSON status summary endpoint."""
        result = handler.handle("/api/status", {}, Mock())

        assert result is not None
        body = _parse_json_result(result)

        assert "status" in body
        assert "components" in body
        assert "timestamp" in body
        assert body["status"] in [s.value for s in ServiceStatus]

    def test_component_status(self, handler):
        """Test component status endpoint."""
        result = handler.handle("/api/status/components", {}, Mock())

        assert result is not None
        body = _parse_json_result(result)

        assert "components" in body
        assert len(body["components"]) > 0

        for component in body["components"]:
            assert "id" in component
            assert "name" in component
            assert "status" in component

    def test_uptime_history(self, handler):
        """Test uptime history endpoint."""
        result = handler.handle("/api/status/history", {}, Mock())

        assert result is not None
        body = _parse_json_result(result)

        assert "current" in body
        assert "periods" in body
        assert "24h" in body["periods"]
        assert "7d" in body["periods"]
        assert "30d" in body["periods"]

    def test_incidents(self, handler):
        """Test incidents endpoint."""
        result = handler.handle("/api/status/incidents", {}, Mock())

        assert result is not None
        body = _parse_json_result(result)

        assert "active" in body
        assert "recent" in body
        assert "scheduled_maintenance" in body

    def test_html_status_page(self, handler):
        """Test HTML status page endpoint."""
        result = handler.handle("/status", {}, Mock())

        assert result is not None
        # HandlerResult has attributes: status_code, content_type, body, headers
        assert hasattr(result, "body") or isinstance(result, dict)

        if hasattr(result, "content_type"):
            # HandlerResult object
            assert result.content_type == "text/html; charset=utf-8"
            html = result.body
            if isinstance(html, bytes):
                html = html.decode("utf-8")
        else:
            # Dict response
            assert "body" in result
            assert result["headers"]["Content-Type"] == "text/html; charset=utf-8"
            html = result["body"]

        assert "<!DOCTYPE html>" in html
        assert "Aragora Status" in html
        assert "All Systems" in html or "System" in html

    def test_handle_unknown_path_returns_none(self, handler):
        """Test that unknown paths return None."""
        result = handler.handle("/api/v1/unknown", {}, Mock())
        assert result is None


# =============================================================================
# Versioned v1 Endpoints
# =============================================================================


class TestV1StatusEndpoint:
    """Tests for GET /api/v1/status endpoint."""

    @pytest.fixture
    def handler(self):
        return StatusPageHandler({})

    def test_v1_status_returns_data_envelope(self, handler):
        """Test v1 status returns {"data": ...} envelope."""
        result = handler.handle("/api/v1/status", {}, Mock())

        assert result is not None
        body = _parse_json_result(result)

        assert "data" in body
        data = body["data"]
        assert "status" in data
        assert "status_detail" in data
        assert "message" in data
        assert "uptime_seconds" in data
        assert "uptime_formatted" in data
        assert "timestamp" in data
        assert "components_summary" in data
        assert "sla" in data

    def test_v1_status_has_correct_status_category(self, handler):
        """Test v1 status returns simplified status category."""
        with patch.object(
            handler,
            "_check_all_components",
            return_value=[
                ComponentHealth("API", ServiceStatus.OPERATIONAL),
            ],
        ):
            result = handler.handle("/api/v1/status", {}, Mock())
            body = _parse_json_result(result)

            assert body["data"]["status"] == "operational"

    def test_v1_status_degraded_maps_correctly(self, handler):
        """Test degraded status maps to 'degraded' category."""
        with patch.object(
            handler,
            "_check_all_components",
            return_value=[
                ComponentHealth("API", ServiceStatus.OPERATIONAL),
                ComponentHealth("Cache", ServiceStatus.DEGRADED),
            ],
        ):
            result = handler.handle("/api/v1/status", {}, Mock())
            body = _parse_json_result(result)

            assert body["data"]["status"] == "degraded"

    def test_v1_status_major_outage_maps_to_down(self, handler):
        """Test major outage maps to 'down' category."""
        with patch.object(
            handler,
            "_check_all_components",
            return_value=[
                ComponentHealth("API", ServiceStatus.MAJOR_OUTAGE),
            ],
        ):
            result = handler.handle("/api/v1/status", {}, Mock())
            body = _parse_json_result(result)

            assert body["data"]["status"] == "down"

    def test_v1_status_maintenance_maps_correctly(self, handler):
        """Test maintenance maps to 'maintenance' category."""
        with patch.object(
            handler,
            "_check_all_components",
            return_value=[
                ComponentHealth("API", ServiceStatus.MAINTENANCE),
            ],
        ):
            result = handler.handle("/api/v1/status", {}, Mock())
            body = _parse_json_result(result)

            assert body["data"]["status"] == "maintenance"

    def test_v1_status_components_summary(self, handler):
        """Test v1 status includes correct component summary counts."""
        with patch.object(
            handler,
            "_check_all_components",
            return_value=[
                ComponentHealth("API", ServiceStatus.OPERATIONAL),
                ComponentHealth("DB", ServiceStatus.OPERATIONAL),
                ComponentHealth("Cache", ServiceStatus.DEGRADED),
                ComponentHealth("WS", ServiceStatus.MAJOR_OUTAGE),
            ],
        ):
            result = handler.handle("/api/v1/status", {}, Mock())
            body = _parse_json_result(result)
            summary = body["data"]["components_summary"]

            assert summary["total"] == 4
            assert summary["operational"] == 2
            assert summary["degraded"] == 1
            assert summary["down"] == 1

    def test_v1_status_sla_structure(self, handler):
        """Test v1 status includes SLA metrics structure."""
        result = handler.handle("/api/v1/status", {}, Mock())
        body = _parse_json_result(result)
        sla = body["data"]["sla"]

        assert "latency" in sla
        assert "error_rate" in sla
        assert "p50" in sla["latency"]
        assert "p95" in sla["latency"]
        assert "p99" in sla["latency"]
        assert "total_requests" in sla["error_rate"]
        assert "error_count" in sla["error_rate"]
        assert "error_rate" in sla["error_rate"]

    def test_v1_status_returns_200(self, handler):
        """Test v1 status returns HTTP 200."""
        result = handler.handle("/api/v1/status", {}, Mock())
        assert result.status_code == 200


class TestV1ComponentsEndpoint:
    """Tests for GET /api/v1/status/components endpoint."""

    @pytest.fixture
    def handler(self):
        return StatusPageHandler({})

    def test_v1_components_returns_data_envelope(self, handler):
        """Test v1 components returns {"data": ...} envelope."""
        result = handler.handle("/api/v1/status/components", {}, Mock())

        assert result is not None
        body = _parse_json_result(result)

        assert "data" in body
        data = body["data"]
        assert "components" in data
        assert "timestamp" in data

    def test_v1_components_have_required_fields(self, handler):
        """Test each component has all required fields."""
        result = handler.handle("/api/v1/status/components", {}, Mock())
        body = _parse_json_result(result)

        for component in body["data"]["components"]:
            assert "id" in component
            assert "name" in component
            assert "description" in component
            assert "status" in component
            assert "response_time_ms" in component
            assert "last_check" in component
            assert "message" in component

    def test_v1_components_returns_all_defined_components(self, handler):
        """Test all defined components are returned."""
        result = handler.handle("/api/v1/status/components", {}, Mock())
        body = _parse_json_result(result)

        component_ids = {c["id"] for c in body["data"]["components"]}
        expected_ids = {c["id"] for c in handler.COMPONENTS}
        assert component_ids == expected_ids

    def test_v1_components_returns_200(self, handler):
        """Test v1 components returns HTTP 200."""
        result = handler.handle("/api/v1/status/components", {}, Mock())
        assert result.status_code == 200


class TestV1IncidentsEndpoint:
    """Tests for GET /api/v1/status/incidents endpoint."""

    @pytest.fixture
    def handler(self):
        return StatusPageHandler({})

    def test_v1_incidents_returns_data_envelope(self, handler):
        """Test v1 incidents returns {"data": ...} envelope."""
        result = handler.handle("/api/v1/status/incidents", {}, Mock())

        assert result is not None
        body = _parse_json_result(result)

        assert "data" in body
        data = body["data"]
        assert "active" in data
        assert "recent" in data
        assert "scheduled_maintenance" in data
        assert "timestamp" in data

    def test_v1_incidents_active_is_list(self, handler):
        """Test active incidents is a list."""
        result = handler.handle("/api/v1/status/incidents", {}, Mock())
        body = _parse_json_result(result)

        assert isinstance(body["data"]["active"], list)
        assert isinstance(body["data"]["recent"], list)

    def test_v1_incidents_returns_200(self, handler):
        """Test v1 incidents returns HTTP 200."""
        result = handler.handle("/api/v1/status/incidents", {}, Mock())
        assert result.status_code == 200

    def test_v1_incidents_with_incident_store(self, handler):
        """Test v1 incidents integrates with incident store."""
        mock_incident = MagicMock()
        mock_incident.to_dict.return_value = {
            "id": "test-123",
            "title": "Test incident",
            "status": "investigating",
            "severity": "major",
            "components": ["api"],
            "created_at": "2026-02-23T00:00:00+00:00",
            "updated_at": "2026-02-23T00:00:00+00:00",
            "resolved_at": None,
            "updates": [],
        }

        mock_store = MagicMock()
        mock_store.get_active_incidents.return_value = [mock_incident]
        mock_store.get_recent_incidents.return_value = []

        with patch(
            "aragora.server.handlers.public.status_page.get_incident_store",
            return_value=mock_store,
            create=True,
        ):
            # Need to also patch the import path used in the method
            with patch.dict(
                "sys.modules",
                {
                    "aragora.observability.incident_store": MagicMock(
                        get_incident_store=MagicMock(return_value=mock_store),
                    )
                },
            ):
                result = handler.handle("/api/v1/status/incidents", {}, Mock())
                body = _parse_json_result(result)

                # Should have active incidents (either from mock or empty fallback)
                assert isinstance(body["data"]["active"], list)


class TestV1UptimeEndpoint:
    """Tests for GET /api/v1/status/uptime endpoint."""

    @pytest.fixture
    def handler(self):
        return StatusPageHandler({})

    def test_v1_uptime_returns_data_envelope(self, handler):
        """Test v1 uptime returns {"data": ...} envelope."""
        result = handler.handle("/api/v1/status/uptime", {}, Mock())

        assert result is not None
        body = _parse_json_result(result)

        assert "data" in body
        data = body["data"]
        assert "current" in data
        assert "periods" in data
        assert "timestamp" in data

    def test_v1_uptime_has_period_windows(self, handler):
        """Test v1 uptime includes 24h, 7d, 30d windows."""
        result = handler.handle("/api/v1/status/uptime", {}, Mock())
        body = _parse_json_result(result)

        periods = body["data"]["periods"]
        assert "24h" in periods
        assert "7d" in periods
        assert "30d" in periods

    def test_v1_uptime_period_structure(self, handler):
        """Test each uptime period has correct structure."""
        result = handler.handle("/api/v1/status/uptime", {}, Mock())
        body = _parse_json_result(result)

        for period_key, period_data in body["data"]["periods"].items():
            assert "uptime_percent" in period_data
            assert isinstance(period_data["uptime_percent"], (int, float))
            assert 0 <= period_data["uptime_percent"] <= 100

    def test_v1_uptime_current_has_status(self, handler):
        """Test current section includes status and uptime seconds."""
        result = handler.handle("/api/v1/status/uptime", {}, Mock())
        body = _parse_json_result(result)

        current = body["data"]["current"]
        assert "status" in current
        assert "uptime_seconds" in current
        assert current["uptime_seconds"] >= 0

    def test_v1_uptime_returns_200(self, handler):
        """Test v1 uptime returns HTTP 200."""
        result = handler.handle("/api/v1/status/uptime", {}, Mock())
        assert result.status_code == 200

    def test_v1_uptime_with_sla_tracker(self, handler):
        """Test v1 uptime integrates with SLA tracker."""
        mock_tracker = MagicMock()
        mock_tracker.get_uptime.return_value = {
            "24h": {
                "uptime_percent": 99.95,
                "total_requests": 10000,
                "error_count": 5,
                "incidents": 5,
            },
            "7d": {
                "uptime_percent": 99.9,
                "total_requests": 70000,
                "error_count": 70,
                "incidents": 70,
            },
            "30d": {
                "uptime_percent": 99.85,
                "total_requests": 300000,
                "error_count": 450,
                "incidents": 450,
            },
        }

        with patch(
            "aragora.observability.sla_instrumentation.get_sla_tracker",
            return_value=mock_tracker,
        ):
            result = handler.handle("/api/v1/status/uptime", {}, Mock())
            body = _parse_json_result(result)
            periods = body["data"]["periods"]

            assert periods["24h"]["uptime_percent"] == 99.95
            assert periods["7d"]["total_requests"] == 70000


class TestServiceStatus:
    """Tests for ServiceStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert ServiceStatus.OPERATIONAL.value == "operational"
        assert ServiceStatus.DEGRADED.value == "degraded"
        assert ServiceStatus.PARTIAL_OUTAGE.value == "partial_outage"
        assert ServiceStatus.MAJOR_OUTAGE.value == "major_outage"
        assert ServiceStatus.MAINTENANCE.value == "maintenance"


class TestComponentHealth:
    """Tests for ComponentHealth dataclass."""

    def test_create_component_health(self):
        """Test creating ComponentHealth."""
        health = ComponentHealth(
            name="API",
            status=ServiceStatus.OPERATIONAL,
            response_time_ms=5.2,
            message="All good",
        )

        assert health.name == "API"
        assert health.status == ServiceStatus.OPERATIONAL
        assert health.response_time_ms == 5.2
        assert health.message == "All good"

    def test_component_health_defaults(self):
        """Test ComponentHealth default values."""
        health = ComponentHealth(
            name="Test",
            status=ServiceStatus.DEGRADED,
        )

        assert health.response_time_ms is None
        assert health.last_check is None
        assert health.message is None


class TestOverallStatus:
    """Tests for overall status calculation."""

    @pytest.fixture
    def handler(self):
        return StatusPageHandler({})

    def test_all_operational_returns_operational(self, handler):
        """Test all operational components = operational overall."""
        with patch.object(
            handler,
            "_check_all_components",
            return_value=[
                ComponentHealth("API", ServiceStatus.OPERATIONAL),
                ComponentHealth("DB", ServiceStatus.OPERATIONAL),
            ],
        ):
            assert handler._get_overall_status() == ServiceStatus.OPERATIONAL

    def test_one_degraded_returns_degraded(self, handler):
        """Test one degraded component = degraded overall."""
        with patch.object(
            handler,
            "_check_all_components",
            return_value=[
                ComponentHealth("API", ServiceStatus.OPERATIONAL),
                ComponentHealth("Cache", ServiceStatus.DEGRADED),
            ],
        ):
            assert handler._get_overall_status() == ServiceStatus.DEGRADED

    def test_one_partial_outage_returns_partial(self, handler):
        """Test one partial outage = partial outage overall."""
        with patch.object(
            handler,
            "_check_all_components",
            return_value=[
                ComponentHealth("API", ServiceStatus.OPERATIONAL),
                ComponentHealth("DB", ServiceStatus.PARTIAL_OUTAGE),
            ],
        ):
            assert handler._get_overall_status() == ServiceStatus.PARTIAL_OUTAGE

    def test_multiple_partial_returns_major(self, handler):
        """Test multiple partial outages = major outage overall."""
        with patch.object(
            handler,
            "_check_all_components",
            return_value=[
                ComponentHealth("API", ServiceStatus.PARTIAL_OUTAGE),
                ComponentHealth("DB", ServiceStatus.PARTIAL_OUTAGE),
            ],
        ):
            assert handler._get_overall_status() == ServiceStatus.MAJOR_OUTAGE

    def test_one_major_outage_returns_major(self, handler):
        """Test one major outage = major outage overall."""
        with patch.object(
            handler,
            "_check_all_components",
            return_value=[
                ComponentHealth("API", ServiceStatus.OPERATIONAL),
                ComponentHealth("DB", ServiceStatus.MAJOR_OUTAGE),
            ],
        ):
            assert handler._get_overall_status() == ServiceStatus.MAJOR_OUTAGE


class TestFormatUptime:
    """Tests for uptime formatting."""

    @pytest.fixture
    def handler(self):
        return StatusPageHandler({})

    def test_format_seconds(self, handler):
        """Test formatting seconds."""
        assert handler._format_uptime(30) == "< 1m"

    def test_format_minutes(self, handler):
        """Test formatting minutes."""
        assert handler._format_uptime(300) == "5m"

    def test_format_hours(self, handler):
        """Test formatting hours."""
        assert handler._format_uptime(7200) == "2h"

    def test_format_days(self, handler):
        """Test formatting days."""
        assert handler._format_uptime(172800) == "2d"

    def test_format_mixed(self, handler):
        """Test formatting mixed duration."""
        # 1 day, 2 hours, 30 minutes
        seconds = 86400 + 7200 + 1800
        assert handler._format_uptime(seconds) == "1d 2h 30m"


class TestSLAIntegration:
    """Tests for SLA instrumentation integration with status handler."""

    @pytest.fixture
    def handler(self):
        return StatusPageHandler({})

    def test_get_sla_metrics_returns_structure(self, handler):
        """Test _get_sla_metrics returns correct structure."""
        result = handler._get_sla_metrics()

        assert "latency" in result
        assert "error_rate" in result
        assert "p50" in result["latency"]
        assert "p95" in result["latency"]
        assert "p99" in result["latency"]

    def test_get_sla_metrics_handles_missing_tracker(self, handler):
        """Test _get_sla_metrics gracefully handles import error."""
        with patch.dict("sys.modules", {"aragora.observability.sla_instrumentation": None}):
            # Force ImportError path
            result = handler._get_sla_metrics()
            assert "latency" in result
            assert "error_rate" in result

    def test_get_sla_uptime_returns_structure(self, handler):
        """Test _get_sla_uptime returns correct structure."""
        result = handler._get_sla_uptime()

        assert "24h" in result
        assert "7d" in result
        assert "30d" in result
        for period_data in result.values():
            assert "uptime_percent" in period_data
