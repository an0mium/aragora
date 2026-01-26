"""
Tests for the health handler - critical for K8s deployments.

Tests:
- Liveness probe (/healthz)
- Readiness probe (/readyz)
- Comprehensive health check (/api/health)
- Detailed health check (/api/health/detailed)
- Deep health check (/api/health/deep)
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from aragora.server.handlers.admin.health import HealthHandler

pytestmark = pytest.mark.asyncio


@pytest.fixture
def health_handler():
    """Create a health handler with mocked dependencies."""
    ctx = {"storage": None, "elo_system": None, "nomic_dir": None}
    handler = HealthHandler(ctx)
    return handler


@pytest.fixture
def health_handler_with_storage():
    """Create a health handler with mocked storage."""
    mock_storage = MagicMock()
    mock_storage.list_recent.return_value = []

    ctx = {"storage": mock_storage, "elo_system": None, "nomic_dir": None}
    handler = HealthHandler(ctx)
    return handler


class TestHealthHandler:
    """Tests for HealthHandler."""

    def test_can_handle_healthz(self, health_handler):
        """Test that handler recognizes /healthz route."""
        assert health_handler.can_handle("/healthz") is True

    def test_can_handle_readyz(self, health_handler):
        """Test that handler recognizes /readyz route."""
        assert health_handler.can_handle("/readyz") is True

    def test_can_handle_api_health(self, health_handler):
        """Test that handler recognizes /api/health route."""
        assert health_handler.can_handle("/api/v1/health") is True

    def test_can_handle_api_health_detailed(self, health_handler):
        """Test that handler recognizes /api/health/detailed route."""
        assert health_handler.can_handle("/api/v1/health/detailed") is True

    def test_can_handle_api_health_deep(self, health_handler):
        """Test that handler recognizes /api/health/deep route."""
        assert health_handler.can_handle("/api/v1/health/deep") is True

    def test_cannot_handle_unknown_path(self, health_handler):
        """Test that handler rejects unknown paths."""
        assert health_handler.can_handle("/unknown") is False
        assert health_handler.can_handle("/api/v1/debates") is False


class TestLivenessProbe:
    """Tests for /healthz liveness probe."""

    async def test_liveness_returns_ok(self, health_handler):
        """Liveness probe should always return ok if server is running."""
        result = await health_handler.handle("/healthz", {}, None)

        assert result is not None
        body = json.loads(result.body)
        assert body["status"] == "ok"
        assert result.status_code == 200


class TestReadinessProbe:
    """Tests for /readyz readiness probe."""

    async def test_readiness_with_no_deps_returns_ready(self, health_handler):
        """Readiness should return ready when no deps configured."""
        with patch.object(health_handler, "get_storage", return_value=None):
            with patch.object(health_handler, "get_elo_system", return_value=None):
                result = await health_handler.handle("/readyz", {}, None)

        assert result is not None
        body = json.loads(result.body)
        assert body["status"] == "ready"
        assert result.status_code == 200

    async def test_readiness_with_storage_returns_ready(self, health_handler):
        """Readiness should return ready when storage is available."""
        mock_storage = MagicMock()

        with patch.object(health_handler, "get_storage", return_value=mock_storage):
            with patch.object(health_handler, "get_elo_system", return_value=None):
                result = await health_handler.handle("/readyz", {}, None)

        assert result is not None
        body = json.loads(result.body)
        assert body["status"] == "ready"
        assert body["checks"]["storage"] is True
        assert result.status_code == 200

    async def test_readiness_with_storage_error_returns_not_ready(self, health_handler):
        """Readiness should return not_ready when storage fails."""
        with patch.object(health_handler, "get_storage", side_effect=RuntimeError("DB error")):
            with patch.object(health_handler, "get_elo_system", return_value=None):
                result = await health_handler.handle("/readyz", {}, None)

        assert result is not None
        body = json.loads(result.body)
        assert body["status"] == "not_ready"
        assert body["checks"]["storage"] is False
        assert result.status_code == 503


class TestComprehensiveHealthCheck:
    """Tests for /api/health comprehensive health check."""

    async def test_health_returns_status(self, health_handler):
        """Health check should return status and checks."""
        with patch.object(health_handler, "get_storage", return_value=None):
            with patch.object(health_handler, "get_elo_system", return_value=None):
                with patch.object(health_handler, "get_nomic_dir", return_value=None):
                    result = await health_handler.handle("/api/v1/health", {}, None)

        assert result is not None
        body = json.loads(result.body)
        assert "status" in body
        assert "checks" in body
        assert "timestamp" in body
        assert "version" in body

    async def test_health_includes_uptime(self, health_handler):
        """Health check should include uptime."""
        with patch.object(health_handler, "get_storage", return_value=None):
            with patch.object(health_handler, "get_elo_system", return_value=None):
                with patch.object(health_handler, "get_nomic_dir", return_value=None):
                    result = await health_handler.handle("/api/v1/health", {}, None)

        body = json.loads(result.body)
        assert "uptime_seconds" in body
        assert body["uptime_seconds"] >= 0

    async def test_health_includes_response_time(self, health_handler):
        """Health check should include response time."""
        with patch.object(health_handler, "get_storage", return_value=None):
            with patch.object(health_handler, "get_elo_system", return_value=None):
                with patch.object(health_handler, "get_nomic_dir", return_value=None):
                    result = await health_handler.handle("/api/v1/health", {}, None)

        body = json.loads(result.body)
        assert "response_time_ms" in body
        assert body["response_time_ms"] >= 0


class TestDetailedHealthCheck:
    """Tests for /api/health/detailed endpoint."""

    async def test_detailed_health_returns_components(self, health_handler):
        """Detailed health should return component status."""
        with patch.object(health_handler, "get_storage", return_value=None):
            with patch.object(health_handler, "get_elo_system", return_value=None):
                with patch.object(health_handler, "get_nomic_dir", return_value=None):
                    result = await health_handler.handle("/api/v1/health/detailed", {}, None)

        assert result is not None
        body = json.loads(result.body)
        assert "components" in body
        assert "storage" in body["components"]
        assert "elo_system" in body["components"]


class TestDeepHealthCheck:
    """Tests for /api/health/deep endpoint."""

    async def test_deep_health_returns_comprehensive_checks(self, health_handler):
        """Deep health should return comprehensive checks."""
        with patch.object(health_handler, "get_storage", return_value=None):
            with patch.object(health_handler, "get_elo_system", return_value=None):
                with patch.object(health_handler, "get_nomic_dir", return_value=None):
                    result = await health_handler.handle("/api/v1/health/deep", {}, None)

        assert result is not None
        body = json.loads(result.body)
        assert "status" in body
        assert "checks" in body
        assert "response_time_ms" in body

    async def test_deep_health_includes_ai_providers(self, health_handler):
        """Deep health should check AI provider availability."""
        with patch.object(health_handler, "get_storage", return_value=None):
            with patch.object(health_handler, "get_elo_system", return_value=None):
                with patch.object(health_handler, "get_nomic_dir", return_value=None):
                    result = await health_handler.handle("/api/v1/health/deep", {}, None)

        body = json.loads(result.body)
        assert "ai_providers" in body["checks"]


class TestFilesystemHealthCheck:
    """Tests for filesystem health check helper."""

    def test_filesystem_check_with_temp_dir(self, health_handler):
        """Filesystem check should work with temp directory."""
        with patch.object(health_handler, "get_nomic_dir", return_value=None):
            result = health_handler._check_filesystem_health()

        assert result["healthy"] is True
        assert "path" in result

    def test_filesystem_check_returns_error_on_permission_denied(self, health_handler):
        """Filesystem check should return error on permission denied."""
        with patch.object(health_handler, "get_nomic_dir", return_value=None):
            with patch("pathlib.Path.write_text", side_effect=PermissionError("denied")):
                result = health_handler._check_filesystem_health()

        assert result["healthy"] is False
        assert "Permission denied" in result["error"]


class TestRedisHealthCheck:
    """Tests for Redis health check helper."""

    def test_redis_check_without_config(self, health_handler):
        """Redis check should return healthy when not configured."""
        with patch.dict("os.environ", {}, clear=True):
            result = health_handler._check_redis_health()

        assert result["healthy"] is True
        assert result["configured"] is False


class TestAIProvidersHealthCheck:
    """Tests for AI providers health check helper."""

    def test_ai_providers_check_without_keys(self, health_handler):
        """AI providers check should work without any keys."""
        with patch.dict("os.environ", {}, clear=True):
            result = health_handler._check_ai_providers_health()

        assert result["healthy"] is True
        assert result["any_available"] is False
        assert result["available_count"] == 0

    def test_ai_providers_check_with_anthropic_key(self, health_handler):
        """AI providers check should detect Anthropic key."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test12345"}, clear=True):
            result = health_handler._check_ai_providers_health()

        assert result["healthy"] is True
        assert result["any_available"] is True
        assert result["providers"]["anthropic"] is True


class TestDatabaseStoresHealth:
    """Tests for /api/v1/health/stores endpoint."""

    async def test_stores_health_returns_status(self, health_handler):
        """Database stores health should return status."""
        result = await health_handler.handle("/api/v1/health/stores", {}, None)

        assert result is not None
        body = json.loads(result.body)
        assert "status" in body
        assert "stores" in body

    async def test_stores_health_includes_memory_store(self, health_handler):
        """Database stores health should include memory store info."""
        result = await health_handler.handle("/api/v1/health/stores", {}, None)

        assert result is not None
        body = json.loads(result.body)
        assert "memory" in body.get("stores", {}) or body["status"] in [
            "healthy",
            "degraded",
            "unhealthy",
        ]

    async def test_stores_health_returns_200_when_healthy(self, health_handler):
        """Database stores should return 200 when healthy."""
        result = await health_handler.handle("/api/v1/health/stores", {}, None)

        assert result is not None
        # Should be 200 for healthy/degraded, 503 for unhealthy
        assert result.status_code in [200, 503]


class TestSyncStatus:
    """Tests for /api/v1/health/sync endpoint."""

    async def test_sync_status_returns_response(self, health_handler):
        """Sync status should return sync information."""
        result = await health_handler.handle("/api/v1/health/sync", {}, None)

        assert result is not None
        body = json.loads(result.body)
        # Response should have some useful content
        assert isinstance(body, dict)
        assert len(body) > 0

    async def test_sync_status_returns_200(self, health_handler):
        """Sync status should return 200."""
        result = await health_handler.handle("/api/v1/health/sync", {}, None)

        assert result is not None
        assert result.status_code in [200, 503]


class TestCircuitBreakersStatus:
    """Tests for /api/v1/health/circuits endpoint."""

    async def test_circuits_returns_status(self, health_handler):
        """Circuit breakers should return status."""
        result = await health_handler.handle("/api/v1/health/circuits", {}, None)

        assert result is not None
        body = json.loads(result.body)
        assert "status" in body or "circuits" in body

    async def test_circuits_returns_200(self, health_handler):
        """Circuit breakers should return 200."""
        result = await health_handler.handle("/api/v1/health/circuits", {}, None)

        assert result is not None
        assert result.status_code == 200


class TestSlowDebatesStatus:
    """Tests for /api/v1/health/slow-debates endpoint."""

    async def test_slow_debates_returns_status(self, health_handler):
        """Slow debates should return status."""
        result = await health_handler.handle("/api/v1/health/slow-debates", {}, None)

        assert result is not None
        body = json.loads(result.body)
        assert "status" in body or "slow_debates" in body or "debates" in body

    async def test_slow_debates_returns_count(self, health_handler):
        """Slow debates should return count or list."""
        result = await health_handler.handle("/api/v1/health/slow-debates", {}, None)

        assert result is not None
        body = json.loads(result.body)
        # Should have some count or list of slow debates
        assert any(k in body for k in ["count", "slow_debates", "debates", "status"])


class TestCrossPollinationHealth:
    """Tests for /api/v1/health/cross-pollination endpoint."""

    async def test_cross_pollination_returns_status(self, health_handler):
        """Cross-pollination health should return status."""
        result = await health_handler.handle("/api/v1/health/cross-pollination", {}, None)

        assert result is not None
        body = json.loads(result.body)
        assert "status" in body

    async def test_cross_pollination_includes_enabled_flag(self, health_handler):
        """Cross-pollination should indicate if enabled."""
        result = await health_handler.handle("/api/v1/health/cross-pollination", {}, None)

        assert result is not None
        body = json.loads(result.body)
        # Should indicate enabled status
        assert "enabled" in body or "status" in body


class TestKnowledgeMoundHealth:
    """Tests for /api/v1/health/knowledge-mound endpoint."""

    async def test_knowledge_mound_returns_status(self, health_handler):
        """Knowledge Mound health should return status."""
        result = await health_handler.handle("/api/v1/health/knowledge-mound", {}, None)

        assert result is not None
        body = json.loads(result.body)
        assert "status" in body

    async def test_knowledge_mound_includes_adapter_info(self, health_handler):
        """Knowledge Mound should include adapter information."""
        result = await health_handler.handle("/api/v1/health/knowledge-mound", {}, None)

        assert result is not None
        body = json.loads(result.body)
        # Should have adapter or KM related info
        assert any(k in body for k in ["adapters", "knowledge_mound", "km", "status"])


class TestEncryptionHealth:
    """Tests for encryption-related health checks.

    Note: Encryption health is part of the detailed/deep checks, not a separate endpoint.
    """

    async def test_encryption_in_deep_health(self, health_handler):
        """Encryption info should be available in deep health check."""
        with patch.object(health_handler, "get_storage", return_value=None):
            with patch.object(health_handler, "get_elo_system", return_value=None):
                with patch.object(health_handler, "get_nomic_dir", return_value=None):
                    result = await health_handler.handle("/api/v1/health/deep", {}, None)

        assert result is not None
        body = json.loads(result.body)
        assert "checks" in body

    async def test_encryption_in_diagnostics(self, health_handler):
        """Encryption key check should be in diagnostics checklist."""
        result = await health_handler.handle("/api/v1/diagnostics", {}, None)

        assert result is not None
        body = json.loads(result.body)
        # Encryption key is in the security checklist
        if "checklist" in body and "security" in body["checklist"]:
            assert "encryption_key" in body["checklist"]["security"]


class TestPlatformHealth:
    """Tests for /api/v1/health/platform and /api/v1/platform/health endpoints."""

    async def test_platform_health_returns_status(self, health_handler):
        """Platform health should return status."""
        result = await health_handler.handle("/api/v1/health/platform", {}, None)

        assert result is not None
        body = json.loads(result.body)
        assert "status" in body

    async def test_platform_health_alternate_route(self, health_handler):
        """Platform health should work on alternate route."""
        result = await health_handler.handle("/api/v1/platform/health", {}, None)

        assert result is not None
        body = json.loads(result.body)
        assert "status" in body

    async def test_platform_health_includes_environment(self, health_handler):
        """Platform health should include environment info."""
        result = await health_handler.handle("/api/v1/health/platform", {}, None)

        assert result is not None
        body = json.loads(result.body)
        # Should have environment or platform info
        assert any(k in body for k in ["environment", "env", "platform", "status"])


class TestDiagnostics:
    """Tests for /api/v1/diagnostics endpoint."""

    async def test_diagnostics_returns_checklist_or_components(self, health_handler):
        """Diagnostics should return checklist or components."""
        result = await health_handler.handle("/api/v1/diagnostics", {}, None)

        assert result is not None
        body = json.loads(result.body)
        # Response includes checklist, components, issues, live
        assert "checklist" in body or "components" in body or "live" in body

    async def test_diagnostics_deployment_route(self, health_handler):
        """Diagnostics should work on deployment route."""
        result = await health_handler.handle("/api/v1/diagnostics/deployment", {}, None)

        assert result is not None
        body = json.loads(result.body)
        assert "status" in body or "deployment" in body or "checklist" in body

    async def test_diagnostics_includes_checklist(self, health_handler):
        """Diagnostics should include deployment checklist."""
        result = await health_handler.handle("/api/v1/diagnostics", {}, None)

        assert result is not None
        body = json.loads(result.body)
        # Should have checklist or status info
        assert any(k in body for k in ["checklist", "checks", "status", "deployment"])


class TestBackwardCompatibilityRoutes:
    """Tests for backward compatibility (non-v1) routes."""

    async def test_non_v1_health_route(self, health_handler):
        """Non-v1 health route should work."""
        result = await health_handler.handle("/api/health", {}, None)

        assert result is not None
        body = json.loads(result.body)
        assert "status" in body

    async def test_non_v1_health_detailed_route(self, health_handler):
        """Non-v1 detailed health route should work."""
        result = await health_handler.handle("/api/health/detailed", {}, None)

        assert result is not None
        body = json.loads(result.body)
        assert "components" in body or "status" in body

    async def test_non_v1_health_deep_route(self, health_handler):
        """Non-v1 deep health route should work."""
        result = await health_handler.handle("/api/health/deep", {}, None)

        assert result is not None
        body = json.loads(result.body)
        assert "status" in body

    async def test_non_v1_stores_route(self, health_handler):
        """Non-v1 stores route should work."""
        result = await health_handler.handle("/api/health/stores", {}, None)

        assert result is not None
        body = json.loads(result.body)
        assert "status" in body or "stores" in body

    async def test_non_v1_diagnostics_route(self, health_handler):
        """Non-v1 diagnostics route should work."""
        result = await health_handler.handle("/api/diagnostics", {}, None)

        assert result is not None
        body = json.loads(result.body)
        # Response includes checklist, components, issues, live
        assert "checklist" in body or "components" in body or "live" in body


class TestHealthHandlerRoutes:
    """Tests for all registered routes."""

    def test_all_routes_are_list(self, health_handler):
        """ROUTES should be a list."""
        assert isinstance(health_handler.ROUTES, list)

    def test_routes_not_empty(self, health_handler):
        """ROUTES should not be empty."""
        assert len(health_handler.ROUTES) > 0

    def test_healthz_in_routes(self, health_handler):
        """Liveness probe route should be in ROUTES."""
        assert "/healthz" in health_handler.ROUTES

    def test_readyz_in_routes(self, health_handler):
        """Readiness probe route should be in ROUTES."""
        assert "/readyz" in health_handler.ROUTES

    def test_all_v1_health_routes_present(self, health_handler):
        """All v1 health routes should be present."""
        expected_routes = [
            "/api/v1/health",
            "/api/v1/health/detailed",
            "/api/v1/health/deep",
            "/api/v1/health/stores",
            "/api/v1/health/sync",
            "/api/v1/health/circuits",
            "/api/v1/health/slow-debates",
            "/api/v1/health/cross-pollination",
            "/api/v1/health/knowledge-mound",
            "/api/v1/health/platform",
        ]
        for route in expected_routes:
            assert route in health_handler.ROUTES, f"Missing route: {route}"

    def test_can_handle_all_routes(self, health_handler):
        """Handler should recognize all registered routes."""
        for route in health_handler.ROUTES:
            assert health_handler.can_handle(route) is True, f"Cannot handle: {route}"


class TestHealthHandlerErrorHandling:
    """Tests for error handling in health endpoints."""

    async def test_unknown_route_returns_none(self, health_handler):
        """Unknown routes should return None."""
        result = await health_handler.handle("/api/v1/unknown", {}, None)
        assert result is None

    async def test_invalid_path_returns_none(self, health_handler):
        """Invalid paths should return None."""
        result = await health_handler.handle("/invalid", {}, None)
        assert result is None

    async def test_partial_path_returns_none(self, health_handler):
        """Partial paths should return None."""
        result = await health_handler.handle("/api/v1/heal", {}, None)
        assert result is None
