"""
Comprehensive tests for FastAPI health check route endpoints.

Covers:
- /healthz liveness probe
- /livez liveness probe (strict)
- /readyz readiness probe
- /api/v2/health detailed health status
- /api/v2/metrics/summary basic metrics
- Various subsystem states (healthy, degraded, not_initialized)
"""

from __future__ import annotations

import platform
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from aragora.server.fastapi import create_app
from aragora.server.fastapi.routes.health import router


@pytest.fixture
def app():
    """Create a test FastAPI app."""
    return create_app()


@pytest.fixture
def mock_storage():
    """Create a mock storage with debate counting."""
    storage = MagicMock()
    storage.count_debates.return_value = 42
    return storage


@pytest.fixture
def client(app, mock_storage):
    """Create a test client with fully initialized subsystems."""
    app.state.context = {
        "storage": mock_storage,
        "elo_system": MagicMock(),
        "user_store": MagicMock(),
        "rbac_checker": MagicMock(),
        "decision_service": MagicMock(),
    }
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def bare_client(app):
    """Create a test client with NO subsystems initialized."""
    app.state.context = {}
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def degraded_client(app):
    """Create a test client with degraded subsystems (storage=None)."""
    app.state.context = {
        "storage": None,
        "elo_system": None,
        "user_store": None,
        "rbac_checker": None,
        "decision_service": None,
    }
    return TestClient(app, raise_server_exceptions=False)


# =============================================================================
# /healthz
# =============================================================================


class TestHealthz:
    """Tests for GET /healthz."""

    def test_returns_200_ok(self, client):
        """/healthz always returns 200 with {"status": "ok"}."""
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_returns_200_even_without_subsystems(self, bare_client):
        """/healthz responds even when no subsystems are initialized."""
        response = bare_client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_includes_security_headers(self, client):
        """/healthz responses include security headers."""
        response = client.get("/healthz")
        assert response.headers.get("x-frame-options") == "DENY"
        assert response.headers.get("x-content-type-options") == "nosniff"

    def test_includes_trace_id(self, client):
        """/healthz responses include X-Trace-ID."""
        response = client.get("/healthz")
        assert "x-trace-id" in response.headers


# =============================================================================
# /livez
# =============================================================================


class TestLivez:
    """Tests for GET /livez."""

    def test_returns_200_alive(self, client):
        """/livez returns 200 with {"status": "alive"}."""
        response = client.get("/livez")
        assert response.status_code == 200
        assert response.json() == {"status": "alive"}

    def test_returns_200_without_subsystems(self, bare_client):
        """/livez responds without any subsystem initialization."""
        response = bare_client.get("/livez")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_endpoint_exists_in_router(self):
        """The /livez endpoint is registered in the health router."""
        route_paths = [route.path for route in router.routes]
        assert "/livez" in route_paths


# =============================================================================
# /readyz
# =============================================================================


class TestReadyz:
    """Tests for GET /readyz."""

    def test_returns_ready_when_storage_available(self, client):
        """/readyz returns 'ready' when storage is initialized."""
        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_returns_initializing_without_context(self, app):
        """/readyz returns 'initializing' when context is not set."""
        # Don't set app.state.context at all
        c = TestClient(app, raise_server_exceptions=False)
        response = c.get("/readyz")
        assert response.status_code == 200
        assert response.json()["status"] == "initializing"

    def test_returns_degraded_without_storage(self, degraded_client):
        """/readyz returns 'degraded' when storage is None."""
        response = degraded_client.get("/readyz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert "storage" in data.get("reason", "")


# =============================================================================
# /api/v2/health
# =============================================================================


class TestHealthDetailed:
    """Tests for GET /api/v2/health."""

    def test_returns_200_with_status(self, client):
        """Detailed health returns 200 with status."""
        response = client.get("/api/v2/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded")

    def test_includes_subsystems(self, client):
        """Detailed health includes subsystem statuses."""
        response = client.get("/api/v2/health")
        data = response.json()
        assert "subsystems" in data
        subs = data["subsystems"]
        assert "storage" in subs
        assert "elo_system" in subs
        assert "rbac" in subs

    def test_storage_shows_healthy_with_debate_count(self, client, mock_storage):
        """Storage subsystem shows debate count when healthy."""
        response = client.get("/api/v2/health")
        data = response.json()
        storage_status = data["subsystems"]["storage"]
        assert storage_status["status"] == "healthy"
        assert storage_status["debates_count"] == 42

    def test_storage_shows_not_initialized_when_absent(self, degraded_client):
        """Storage subsystem shows 'not_initialized' when None."""
        response = degraded_client.get("/api/v2/health")
        data = response.json()
        assert data["subsystems"]["storage"]["status"] == "not_initialized"

    def test_storage_shows_unhealthy_on_error(self, app):
        """Storage subsystem shows 'unhealthy' when count_debates raises."""
        broken_storage = MagicMock()
        broken_storage.count_debates.side_effect = RuntimeError("DB connection failed")
        app.state.context = {
            "storage": broken_storage,
            "elo_system": None,
            "rbac_checker": None,
        }
        c = TestClient(app, raise_server_exceptions=False)
        response = c.get("/api/v2/health")
        data = response.json()
        assert data["subsystems"]["storage"]["status"] == "unhealthy"

    def test_includes_version_info(self, client):
        """Detailed health includes version information."""
        response = client.get("/api/v2/health")
        data = response.json()
        assert "version" in data
        assert data["version"]["api"] == "2.0.0"
        assert data["version"]["server"] == "fastapi"
        assert data["version"]["python"] == platform.python_version()

    def test_includes_timestamp(self, client):
        """Detailed health includes ISO timestamp."""
        response = client.get("/api/v2/health")
        data = response.json()
        assert "timestamp" in data
        # Should be ISO format
        assert "T" in data["timestamp"]

    def test_includes_uptime(self, client):
        """Detailed health includes uptime string."""
        response = client.get("/api/v2/health")
        data = response.json()
        assert "uptime" in data
        # Should contain at least seconds
        assert "s" in data["uptime"]

    def test_includes_environment(self, client):
        """Detailed health includes environment."""
        response = client.get("/api/v2/health")
        data = response.json()
        assert "environment" in data

    def test_overall_healthy_when_all_subsystems_ok(self, client):
        """Overall status is 'healthy' when all subsystems are healthy or not_initialized."""
        response = client.get("/api/v2/health")
        data = response.json()
        assert data["status"] == "healthy"


# =============================================================================
# /api/v2/metrics/summary
# =============================================================================


class TestMetricsSummary:
    """Tests for GET /api/v2/metrics/summary."""

    def test_returns_200(self, client):
        """Metrics summary returns 200."""
        response = client.get("/api/v2/metrics/summary")
        assert response.status_code == 200

    def test_includes_uptime_seconds(self, client):
        """Metrics summary includes uptime_seconds."""
        response = client.get("/api/v2/metrics/summary")
        data = response.json()
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], int)
        assert data["uptime_seconds"] >= 0

    def test_includes_debate_count(self, client, mock_storage):
        """Metrics summary includes debates_total from storage."""
        response = client.get("/api/v2/metrics/summary")
        data = response.json()
        assert data.get("debates_total") == 42

    def test_works_without_storage(self, degraded_client):
        """Metrics summary works even without storage."""
        response = degraded_client.get("/api/v2/metrics/summary")
        assert response.status_code == 200
        data = response.json()
        assert "uptime_seconds" in data
        assert "debates_total" not in data


# =============================================================================
# Router Registration
# =============================================================================


class TestRouterRegistration:
    """Tests that health routes are properly registered."""

    def test_healthz_registered(self):
        route_paths = [route.path for route in router.routes]
        assert "/healthz" in route_paths

    def test_livez_registered(self):
        route_paths = [route.path for route in router.routes]
        assert "/livez" in route_paths

    def test_readyz_registered(self):
        route_paths = [route.path for route in router.routes]
        assert "/readyz" in route_paths

    def test_health_detail_registered(self):
        route_paths = [route.path for route in router.routes]
        assert "/api/v2/health" in route_paths

    def test_metrics_summary_registered(self):
        route_paths = [route.path for route in router.routes]
        assert "/api/v2/metrics/summary" in route_paths
