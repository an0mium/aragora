"""
Tests for HealthHandler endpoints.

Endpoints tested:
- GET /healthz - Kubernetes liveness probe
- GET /readyz - Kubernetes readiness probe
- GET /api/health - Comprehensive health check
- GET /api/health/detailed - Detailed health with observer metrics
- GET /api/health/deep - Deep health check with all external dependencies
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from aragora.server.handlers.admin.health import HealthHandler
from aragora.server.handlers.base import clear_cache


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_nomic_dir(tmp_path):
    """Create a mock nomic directory structure."""
    nomic_dir = tmp_path / ".nomic"
    nomic_dir.mkdir()
    return nomic_dir


@pytest.fixture
def health_handler(mock_nomic_dir):
    """Create a HealthHandler with mock dependencies."""
    ctx = {
        "storage": None,
        "elo_system": None,
        "nomic_dir": mock_nomic_dir,
    }
    return HealthHandler(ctx)


@pytest.fixture
def health_handler_no_nomic():
    """Create a HealthHandler without nomic_dir."""
    ctx = {
        "storage": None,
        "elo_system": None,
        "nomic_dir": None,
    }
    return HealthHandler(ctx)


@pytest.fixture
def mock_storage():
    """Create a mock storage that works properly."""
    storage = Mock()
    storage.list_recent = Mock(return_value=[])
    return storage


@pytest.fixture
def mock_elo_system():
    """Create a mock ELO system that works properly."""
    elo = Mock()
    elo.get_leaderboard = Mock(return_value=[])
    return elo


@pytest.fixture
def health_handler_with_deps(mock_nomic_dir, mock_storage, mock_elo_system):
    """Create a HealthHandler with working mock dependencies."""
    ctx = {
        "storage": mock_storage,
        "elo_system": mock_elo_system,
        "nomic_dir": mock_nomic_dir,
    }
    return HealthHandler(ctx)


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches before and after each test."""
    clear_cache()
    yield
    clear_cache()


# ============================================================================
# Route Matching Tests
# ============================================================================


class TestHealthRouting:
    """Tests for route matching."""

    def test_can_handle_healthz(self, health_handler):
        """Handler can handle /healthz."""
        assert health_handler.can_handle("/healthz") is True

    def test_can_handle_readyz(self, health_handler):
        """Handler can handle /readyz."""
        assert health_handler.can_handle("/readyz") is True

    def test_can_handle_api_health(self, health_handler):
        """Handler can handle /api/health."""
        assert health_handler.can_handle("/api/health") is True

    def test_can_handle_api_health_detailed(self, health_handler):
        """Handler can handle /api/health/detailed."""
        assert health_handler.can_handle("/api/health/detailed") is True

    def test_can_handle_api_health_deep(self, health_handler):
        """Handler can handle /api/health/deep."""
        assert health_handler.can_handle("/api/health/deep") is True

    def test_cannot_handle_unrelated_routes(self, health_handler):
        """Handler doesn't handle unrelated routes."""
        assert health_handler.can_handle("/api/debates") is False
        assert health_handler.can_handle("/api/agents") is False
        assert health_handler.can_handle("/healthz/extra") is False
        assert health_handler.can_handle("/api/health/unknown") is False


# ============================================================================
# GET /healthz Tests (Liveness Probe)
# ============================================================================


class TestLivenessProbe:
    """Tests for GET /healthz endpoint."""

    def test_liveness_returns_ok(self, health_handler):
        """Liveness probe returns ok status."""
        result = health_handler.handle("/healthz", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["status"] == "ok"

    def test_liveness_is_lightweight(self, health_handler):
        """Liveness probe doesn't check external dependencies."""
        # Even with broken dependencies, liveness should return ok
        result = health_handler.handle("/healthz", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["status"] == "ok"


# ============================================================================
# GET /readyz Tests (Readiness Probe)
# ============================================================================


class TestReadinessProbe:
    """Tests for GET /readyz endpoint."""

    def test_readiness_returns_ready_no_deps(self, health_handler):
        """Readiness probe returns ready when no dependencies configured."""
        result = health_handler.handle("/readyz", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["status"] == "ready"
        assert "checks" in data

    def test_readiness_returns_ready_with_deps(self, health_handler_with_deps):
        """Readiness probe returns ready when dependencies are healthy."""
        result = health_handler_with_deps.handle("/readyz", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["status"] == "ready"
        assert data["checks"]["storage"] is True
        assert data["checks"]["elo_system"] is True

    def test_readiness_returns_not_ready_on_storage_error(self, mock_nomic_dir):
        """Readiness probe returns not_ready when storage fails."""
        storage = Mock()
        storage.list_recent = Mock(side_effect=RuntimeError("Connection failed"))

        # Storage getter raises error
        ctx = {
            "storage": None,
            "elo_system": None,
            "nomic_dir": mock_nomic_dir,
        }
        handler = HealthHandler(ctx)

        # Patch get_storage to raise error
        with patch.object(handler, "get_storage", side_effect=RuntimeError("DB error")):
            result = handler.handle("/readyz", {}, None)

        assert result is not None
        assert result.status_code == 503
        data = json.loads(result.body)
        assert data["status"] == "not_ready"
        assert data["checks"]["storage"] is False


# ============================================================================
# GET /api/health Tests (Comprehensive Health)
# ============================================================================


class TestComprehensiveHealth:
    """Tests for GET /api/health endpoint."""

    def test_health_returns_status_and_checks(self, health_handler):
        """Health endpoint returns status and component checks."""
        result = health_handler.handle("/api/health", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)

        # Basic structure
        assert "status" in data
        assert "checks" in data
        assert "timestamp" in data
        assert "response_time_ms" in data
        assert "uptime_seconds" in data

    def test_health_includes_database_check(self, health_handler_with_deps):
        """Health endpoint includes database connectivity check."""
        result = health_handler_with_deps.handle("/api/health", {}, None)

        assert result is not None
        data = json.loads(result.body)
        assert "database" in data["checks"]
        assert data["checks"]["database"]["healthy"] is True

    def test_health_includes_elo_check(self, health_handler_with_deps):
        """Health endpoint includes ELO system check."""
        result = health_handler_with_deps.handle("/api/health", {}, None)

        assert result is not None
        data = json.loads(result.body)
        assert "elo_system" in data["checks"]
        assert data["checks"]["elo_system"]["healthy"] is True

    def test_health_includes_filesystem_check(self, health_handler):
        """Health endpoint includes filesystem write check."""
        result = health_handler.handle("/api/health", {}, None)

        assert result is not None
        data = json.loads(result.body)
        assert "filesystem" in data["checks"]
        # Filesystem should be healthy in test environment
        assert data["checks"]["filesystem"]["healthy"] is True

    def test_health_includes_ai_providers_check(self, health_handler):
        """Health endpoint includes AI providers availability check."""
        result = health_handler.handle("/api/health", {}, None)

        assert result is not None
        data = json.loads(result.body)
        assert "ai_providers" in data["checks"]
        assert "providers" in data["checks"]["ai_providers"]

    def test_health_returns_degraded_on_critical_failure(self, health_handler):
        """Health returns degraded status when critical service fails."""
        # Patch filesystem check to fail
        with patch.object(
            health_handler,
            "_check_filesystem_health",
            return_value={"healthy": False, "error": "Write failed"},
        ):
            result = health_handler.handle("/api/health", {}, None)

        assert result is not None
        assert result.status_code == 503
        data = json.loads(result.body)
        assert data["status"] == "degraded"

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-12345678901234567890"})
    def test_health_detects_api_keys(self, health_handler):
        """Health endpoint correctly detects configured API keys."""
        result = health_handler.handle("/api/health", {}, None)

        assert result is not None
        data = json.loads(result.body)
        assert data["checks"]["ai_providers"]["providers"]["anthropic"] is True
        assert data["checks"]["ai_providers"]["any_available"] is True


# ============================================================================
# GET /api/health/detailed Tests
# ============================================================================


class TestDetailedHealth:
    """Tests for GET /api/health/detailed endpoint."""

    def test_detailed_health_returns_components(self, health_handler):
        """Detailed health returns component status."""
        result = health_handler.handle("/api/health/detailed", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)

        assert "status" in data
        assert "components" in data
        assert "version" in data

    def test_detailed_health_includes_nomic_dir_status(self, health_handler):
        """Detailed health shows nomic_dir availability."""
        result = health_handler.handle("/api/health/detailed", {}, None)

        assert result is not None
        data = json.loads(result.body)
        assert "nomic_dir" in data["components"]
        assert data["components"]["nomic_dir"] is True

    def test_detailed_health_shows_nomic_dir_missing(self, health_handler_no_nomic):
        """Detailed health shows nomic_dir as false when not configured."""
        result = health_handler_no_nomic.handle("/api/health/detailed", {}, None)

        assert result is not None
        data = json.loads(result.body)
        assert data["components"]["nomic_dir"] is False

    def test_detailed_health_includes_warnings_array(self, health_handler):
        """Detailed health includes warnings array."""
        result = health_handler.handle("/api/health/detailed", {}, None)

        assert result is not None
        data = json.loads(result.body)
        assert "warnings" in data
        assert isinstance(data["warnings"], list)


# ============================================================================
# GET /api/health/deep Tests
# ============================================================================


class TestDeepHealth:
    """Tests for GET /api/health/deep endpoint."""

    def test_deep_health_returns_all_checks(self, health_handler_with_deps):
        """Deep health check returns comprehensive system status."""
        result = health_handler_with_deps.handle("/api/health/deep", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)

        # Should have many checks
        assert "status" in data
        assert "checks" in data
        assert "response_time_ms" in data
        assert "timestamp" in data

    def test_deep_health_includes_storage_check(self, health_handler_with_deps):
        """Deep health includes storage connectivity check."""
        result = health_handler_with_deps.handle("/api/health/deep", {}, None)

        assert result is not None
        data = json.loads(result.body)
        assert "storage" in data["checks"]
        assert data["checks"]["storage"]["healthy"] is True

    def test_deep_health_includes_elo_check(self, health_handler_with_deps):
        """Deep health includes ELO system check."""
        result = health_handler_with_deps.handle("/api/health/deep", {}, None)

        assert result is not None
        data = json.loads(result.body)
        assert "elo_system" in data["checks"]
        assert data["checks"]["elo_system"]["healthy"] is True

    def test_deep_health_includes_redis_check(self, health_handler):
        """Deep health includes Redis check (not configured by default)."""
        result = health_handler.handle("/api/health/deep", {}, None)

        assert result is not None
        data = json.loads(result.body)
        assert "redis" in data["checks"]
        # Redis not configured is ok
        assert data["checks"]["redis"]["healthy"] is True
        assert data["checks"]["redis"]["configured"] is False

    def test_deep_health_includes_ai_providers(self, health_handler):
        """Deep health includes AI provider availability."""
        result = health_handler.handle("/api/health/deep", {}, None)

        assert result is not None
        data = json.loads(result.body)
        assert "ai_providers" in data["checks"]

    def test_deep_health_includes_filesystem(self, health_handler):
        """Deep health includes filesystem check."""
        result = health_handler.handle("/api/health/deep", {}, None)

        assert result is not None
        data = json.loads(result.body)
        assert "filesystem" in data["checks"]
        assert data["checks"]["filesystem"]["healthy"] is True

    def test_deep_health_reports_warnings(self, health_handler):
        """Deep health reports warnings when applicable."""
        result = health_handler.handle("/api/health/deep", {}, None)

        assert result is not None
        data = json.loads(result.body)
        # Warnings may or may not be present depending on config
        if data.get("warnings"):
            assert isinstance(data["warnings"], list)


# ============================================================================
# Handle Routing Tests
# ============================================================================


class TestHealthHandleRouting:
    """Tests for handle() method routing."""

    def test_handle_routes_to_liveness(self, health_handler):
        """handle() correctly routes /healthz to liveness probe."""
        result = health_handler.handle("/healthz", {}, None)
        assert result is not None
        data = json.loads(result.body)
        assert data["status"] == "ok"

    def test_handle_routes_to_readiness(self, health_handler):
        """handle() correctly routes /readyz to readiness probe."""
        result = health_handler.handle("/readyz", {}, None)
        assert result is not None
        data = json.loads(result.body)
        assert "checks" in data

    def test_handle_routes_to_health(self, health_handler):
        """handle() correctly routes /api/health to health check."""
        result = health_handler.handle("/api/health", {}, None)
        assert result is not None
        data = json.loads(result.body)
        assert "uptime_seconds" in data

    def test_handle_returns_none_for_unknown(self, health_handler):
        """handle() returns None for unhandled routes."""
        result = health_handler.handle("/api/unknown", {}, None)
        assert result is None


# ============================================================================
# Handler Import Tests
# ============================================================================


class TestHealthHandlerImport:
    """Test HealthHandler import and export."""

    def test_handler_importable(self):
        """HealthHandler can be imported from handlers.admin.health module."""
        from aragora.server.handlers.admin.health import HealthHandler

        assert HealthHandler is not None

    def test_handler_has_routes(self):
        """HealthHandler has ROUTES class attribute."""
        from aragora.server.handlers.admin.health import HealthHandler

        assert hasattr(HealthHandler, "ROUTES")
        assert "/healthz" in HealthHandler.ROUTES
        assert "/readyz" in HealthHandler.ROUTES
        assert "/api/health" in HealthHandler.ROUTES


# ============================================================================
# GET /api/health/stores Tests (Database Stores Health)
# ============================================================================


class TestDatabaseStoresHealth:
    """Tests for GET /api/health/stores endpoint."""

    def test_stores_health_returns_response(self, health_handler):
        """Stores health endpoint returns a valid response."""
        result = health_handler.handle("/api/health/stores", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)

        assert "status" in data
        assert "stores" in data
        assert "summary" in data
        assert "elapsed_ms" in data

    def test_stores_health_includes_core_stores(self, health_handler_with_deps):
        """Stores health includes core database stores."""
        result = health_handler_with_deps.handle("/api/health/stores", {}, None)

        assert result is not None
        data = json.loads(result.body)
        stores = data["stores"]

        # Core stores that should always be checked
        assert "debate_storage" in stores
        assert "elo_system" in stores

    def test_stores_health_includes_new_stores(self, health_handler):
        """Stores health includes new stores (integration, gmail, sync, decision)."""
        result = health_handler.handle("/api/health/stores", {}, None)

        assert result is not None
        data = json.loads(result.body)
        stores = data["stores"]

        # New stores added for commercial viability
        assert "integration_store" in stores
        assert "gmail_token_store" in stores
        assert "sync_store" in stores
        assert "decision_result_store" in stores

    def test_stores_health_shows_module_not_available(self, health_handler):
        """Stores health gracefully handles missing modules."""
        # Patch imports to simulate missing modules
        with patch.dict("sys.modules", {"aragora.storage.integration_store": None}):
            result = health_handler.handle("/api/health/stores", {}, None)

        assert result is not None
        data = json.loads(result.body)
        # Even with missing modules, endpoint should return valid response
        assert "stores" in data

    def test_stores_health_summary_counts(self, health_handler):
        """Stores health summary has correct count fields."""
        result = health_handler.handle("/api/health/stores", {}, None)

        assert result is not None
        data = json.loads(result.body)
        summary = data["summary"]

        assert "total" in summary
        assert "healthy" in summary
        assert "connected" in summary
        assert "not_initialized" in summary

        # Total should equal healthy count (all should be healthy even if not initialized)
        assert summary["total"] == summary["healthy"]

    def test_stores_health_decision_store_has_metrics(self, health_handler, tmp_path):
        """Decision result store health check includes metrics."""
        import os

        os.environ["ARAGORA_DECISION_RESULTS_DB"] = str(tmp_path / "test_decisions.db")

        try:
            # Reset the singleton
            from aragora.storage import decision_result_store

            decision_result_store._decision_result_store = None

            result = health_handler.handle("/api/health/stores", {}, None)

            assert result is not None
            data = json.loads(result.body)

            if "decision_result_store" in data["stores"]:
                store_info = data["stores"]["decision_result_store"]
                if store_info.get("status") == "connected":
                    # Should have metrics
                    assert "total_entries" in store_info or "type" in store_info
        finally:
            if "ARAGORA_DECISION_RESULTS_DB" in os.environ:
                del os.environ["ARAGORA_DECISION_RESULTS_DB"]
            decision_result_store._decision_result_store = None


class TestStoresHealthRouting:
    """Tests for /api/health/stores route handling."""

    def test_can_handle_stores_route(self, health_handler):
        """Handler can handle /api/health/stores."""
        assert health_handler.can_handle("/api/health/stores") is True

    def test_handle_routes_to_stores_health(self, health_handler):
        """handle() correctly routes /api/health/stores."""
        result = health_handler.handle("/api/health/stores", {}, None)

        assert result is not None
        data = json.loads(result.body)
        assert "stores" in data
        assert "summary" in data
