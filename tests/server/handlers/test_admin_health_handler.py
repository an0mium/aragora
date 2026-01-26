"""
Tests for the HealthHandler module.

Tests cover:
- Handler routing for health and readiness endpoints
- can_handle method
- ROUTES attribute
- Kubernetes probe endpoints
- Comprehensive health checks
- Database stores health
- Cross-pollination health
- Knowledge Mound health
- Deployment diagnostics
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from aragora.server.handlers.admin.health import HealthHandler

# Mark all tests in this module as async
pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_server_context():
    """Create mock server context for handler initialization."""
    return {"storage": None, "elo_system": None, "nomic_dir": None}


class TestHealthHandlerRouting:
    """Tests for handler routing."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return HealthHandler(mock_server_context)

    def test_can_handle_healthz(self, handler):
        """Handler can handle Kubernetes liveness probe."""
        assert handler.can_handle("/healthz")

    def test_can_handle_readyz(self, handler):
        """Handler can handle Kubernetes readiness probe."""
        assert handler.can_handle("/readyz")

    def test_can_handle_health_v1(self, handler):
        """Handler can handle versioned health endpoint."""
        assert handler.can_handle("/api/v1/health")

    def test_can_handle_health_detailed_v1(self, handler):
        """Handler can handle versioned detailed health endpoint."""
        assert handler.can_handle("/api/v1/health/detailed")

    def test_can_handle_health_deep_v1(self, handler):
        """Handler can handle versioned deep health endpoint."""
        assert handler.can_handle("/api/v1/health/deep")

    def test_can_handle_health_stores_v1(self, handler):
        """Handler can handle versioned stores health endpoint."""
        assert handler.can_handle("/api/v1/health/stores")

    def test_can_handle_health_sync_v1(self, handler):
        """Handler can handle versioned sync status endpoint."""
        assert handler.can_handle("/api/v1/health/sync")

    def test_can_handle_health_circuits_v1(self, handler):
        """Handler can handle versioned circuit breakers endpoint."""
        assert handler.can_handle("/api/v1/health/circuits")

    def test_can_handle_health_slow_debates_v1(self, handler):
        """Handler can handle versioned slow debates endpoint."""
        assert handler.can_handle("/api/v1/health/slow-debates")

    def test_can_handle_health_cross_pollination_v1(self, handler):
        """Handler can handle versioned cross-pollination health endpoint."""
        assert handler.can_handle("/api/v1/health/cross-pollination")

    def test_can_handle_health_knowledge_mound_v1(self, handler):
        """Handler can handle versioned knowledge mound health endpoint."""
        assert handler.can_handle("/api/v1/health/knowledge-mound")

    def test_can_handle_health_encryption_v1(self, handler):
        """Handler can handle versioned encryption health endpoint."""
        assert handler.can_handle("/api/v1/health/encryption")

    def test_can_handle_health_platform_v1(self, handler):
        """Handler can handle versioned platform health endpoint."""
        assert handler.can_handle("/api/v1/health/platform")

    def test_can_handle_platform_health_v1(self, handler):
        """Handler can handle versioned platform/health endpoint."""
        assert handler.can_handle("/api/v1/platform/health")

    def test_can_handle_diagnostics_v1(self, handler):
        """Handler can handle versioned diagnostics endpoint."""
        assert handler.can_handle("/api/v1/diagnostics")

    def test_can_handle_diagnostics_deployment_v1(self, handler):
        """Handler can handle versioned deployment diagnostics endpoint."""
        assert handler.can_handle("/api/v1/diagnostics/deployment")

    def test_can_handle_health_nonversioned(self, handler):
        """Handler can handle non-versioned health endpoint."""
        assert handler.can_handle("/api/health")

    def test_can_handle_health_detailed_nonversioned(self, handler):
        """Handler can handle non-versioned detailed health endpoint."""
        assert handler.can_handle("/api/health/detailed")

    def test_can_handle_health_deep_nonversioned(self, handler):
        """Handler can handle non-versioned deep health endpoint."""
        assert handler.can_handle("/api/health/deep")

    def test_can_handle_health_stores_nonversioned(self, handler):
        """Handler can handle non-versioned stores health endpoint."""
        assert handler.can_handle("/api/health/stores")

    def test_can_handle_diagnostics_nonversioned(self, handler):
        """Handler can handle non-versioned diagnostics endpoint."""
        assert handler.can_handle("/api/diagnostics")

    def test_cannot_handle_other_paths(self, handler):
        """Handler cannot handle unrelated paths."""
        assert not handler.can_handle("/api/v1/debates")
        assert not handler.can_handle("/api/v1/metrics")
        assert not handler.can_handle("/api/v1/other")
        assert not handler.can_handle("/")


class TestHealthHandlerRoutesAttribute:
    """Tests for ROUTES class attribute."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return HealthHandler(mock_server_context)

    def test_routes_contains_healthz(self, handler):
        """ROUTES contains Kubernetes liveness probe."""
        assert "/healthz" in handler.ROUTES

    def test_routes_contains_readyz(self, handler):
        """ROUTES contains Kubernetes readiness probe."""
        assert "/readyz" in handler.ROUTES

    def test_routes_contains_health_v1(self, handler):
        """ROUTES contains versioned health endpoint."""
        assert "/api/v1/health" in handler.ROUTES

    def test_routes_contains_health_detailed_v1(self, handler):
        """ROUTES contains versioned detailed health endpoint."""
        assert "/api/v1/health/detailed" in handler.ROUTES

    def test_routes_contains_health_deep_v1(self, handler):
        """ROUTES contains versioned deep health endpoint."""
        assert "/api/v1/health/deep" in handler.ROUTES

    def test_routes_contains_health_stores_v1(self, handler):
        """ROUTES contains versioned stores health endpoint."""
        assert "/api/v1/health/stores" in handler.ROUTES

    def test_routes_contains_diagnostics_v1(self, handler):
        """ROUTES contains versioned diagnostics endpoint."""
        assert "/api/v1/diagnostics" in handler.ROUTES

    def test_routes_contains_nonversioned_health(self, handler):
        """ROUTES contains non-versioned health endpoint."""
        assert "/api/health" in handler.ROUTES

    def test_routes_count_minimum(self, handler):
        """ROUTES has expected minimum number of endpoints."""
        # At least 20 routes based on handler inspection
        assert len(handler.ROUTES) >= 20


class TestHealthHandlerRouteDispatch:
    """Tests for route dispatch logic."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return HealthHandler(mock_server_context)

    async def test_handle_healthz_returns_result(self, handler):
        """Handle returns result for healthz endpoint."""
        mock_http = MagicMock()

        result = await handler.handle("/healthz", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

    async def test_handle_readyz_returns_result(self, handler):
        """Handle returns result for readyz endpoint."""
        mock_http = MagicMock()

        result = await handler.handle("/readyz", {}, mock_http)

        assert result is not None
        # May return 200 or 503 depending on system state
        assert result.status_code in (200, 503)

    async def test_handle_health_returns_result(self, handler):
        """Handle returns result for health endpoint."""
        mock_http = MagicMock()

        result = await handler.handle("/api/v1/health", {}, mock_http)

        assert result is not None
        # May return 200 or 503 depending on health
        assert result.status_code in (200, 503)

    async def test_handle_health_detailed_returns_result(self, handler):
        """Handle returns result for detailed health endpoint."""
        mock_http = MagicMock()

        result = await handler.handle("/api/v1/health/detailed", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

    async def test_handle_health_stores_returns_result(self, handler):
        """Handle returns result for stores health endpoint."""
        mock_http = MagicMock()

        result = await handler.handle("/api/v1/health/stores", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

    async def test_handle_health_sync_returns_result(self, handler):
        """Handle returns result for sync status endpoint."""
        mock_http = MagicMock()

        result = await handler.handle("/api/v1/health/sync", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

    async def test_handle_health_circuits_returns_result(self, handler):
        """Handle returns result for circuit breakers endpoint."""
        mock_http = MagicMock()

        result = await handler.handle("/api/v1/health/circuits", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

    async def test_handle_health_slow_debates_returns_result(self, handler):
        """Handle returns result for slow debates endpoint."""
        mock_http = MagicMock()

        result = await handler.handle("/api/v1/health/slow-debates", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

    async def test_handle_health_cross_pollination_returns_result(self, handler):
        """Handle returns result for cross-pollination health endpoint."""
        mock_http = MagicMock()

        result = await handler.handle("/api/v1/health/cross-pollination", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

    async def test_handle_health_knowledge_mound_returns_result(self, handler):
        """Handle returns result for knowledge mound health endpoint."""
        mock_http = MagicMock()

        result = await handler.handle("/api/v1/health/knowledge-mound", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

    async def test_handle_health_platform_returns_result(self, handler):
        """Handle returns result for platform health endpoint."""
        mock_http = MagicMock()

        result = await handler.handle("/api/v1/health/platform", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

    async def test_handle_unknown_returns_none(self, handler):
        """Handle returns None for unknown paths."""
        mock_http = MagicMock()

        result = await handler.handle("/api/v1/unknown", {}, mock_http)

        assert result is None


class TestHealthHandlerResponseFormat:
    """Tests for response format."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return HealthHandler(mock_server_context)

    async def test_healthz_response_is_json(self, handler):
        """Healthz endpoint returns JSON response."""
        mock_http = MagicMock()

        result = await handler.handle("/healthz", {}, mock_http)

        assert result is not None
        assert result.content_type == "application/json"

    async def test_readyz_response_is_json(self, handler):
        """Readyz endpoint returns JSON response."""
        mock_http = MagicMock()

        result = await handler.handle("/readyz", {}, mock_http)

        assert result is not None
        assert result.content_type == "application/json"

    async def test_health_response_includes_status(self, handler):
        """Health endpoint response includes status field."""
        mock_http = MagicMock()

        result = await handler.handle("/api/v1/health", {}, mock_http)

        assert result is not None
        import json

        body = json.loads(result.body)
        assert "status" in body

    async def test_healthz_response_includes_status(self, handler):
        """Healthz endpoint response includes status field."""
        mock_http = MagicMock()

        result = await handler.handle("/healthz", {}, mock_http)

        assert result is not None
        import json

        body = json.loads(result.body)
        assert "status" in body


class TestHealthHandlerLivenessProbe:
    """Tests for Kubernetes liveness probe behavior."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return HealthHandler(mock_server_context)

    async def test_liveness_returns_ok_status(self, handler):
        """Liveness probe returns ok status when server is running."""
        mock_http = MagicMock()

        result = await handler.handle("/healthz", {}, mock_http)

        assert result is not None
        import json

        body = json.loads(result.body)
        assert body["status"] == "ok"

    async def test_liveness_returns_200(self, handler):
        """Liveness probe returns 200 status code."""
        mock_http = MagicMock()

        result = await handler.handle("/healthz", {}, mock_http)

        assert result is not None
        assert result.status_code == 200


class TestHealthHandlerDetailedHealth:
    """Tests for detailed health endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return HealthHandler(mock_server_context)

    async def test_detailed_health_includes_components(self, handler):
        """Detailed health includes components section."""
        mock_http = MagicMock()

        result = await handler.handle("/api/v1/health/detailed", {}, mock_http)

        assert result is not None
        import json

        body = json.loads(result.body)
        assert "components" in body

    async def test_detailed_health_includes_version(self, handler):
        """Detailed health includes version field."""
        mock_http = MagicMock()

        result = await handler.handle("/api/v1/health/detailed", {}, mock_http)

        assert result is not None
        import json

        body = json.loads(result.body)
        assert "version" in body


class TestHealthHandlerStoresHealth:
    """Tests for database stores health endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return HealthHandler(mock_server_context)

    async def test_stores_health_includes_status(self, handler):
        """Stores health includes overall status."""
        mock_http = MagicMock()

        result = await handler.handle("/api/v1/health/stores", {}, mock_http)

        assert result is not None
        import json

        body = json.loads(result.body)
        assert "status" in body

    async def test_stores_health_includes_stores(self, handler):
        """Stores health includes stores section."""
        mock_http = MagicMock()

        result = await handler.handle("/api/v1/health/stores", {}, mock_http)

        assert result is not None
        import json

        body = json.loads(result.body)
        assert "stores" in body

    async def test_stores_health_includes_summary(self, handler):
        """Stores health includes summary section."""
        mock_http = MagicMock()

        result = await handler.handle("/api/v1/health/stores", {}, mock_http)

        assert result is not None
        import json

        body = json.loads(result.body)
        assert "summary" in body


class TestHealthHandlerCrossPollination:
    """Tests for cross-pollination health endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return HealthHandler(mock_server_context)

    async def test_cross_pollination_includes_status(self, handler):
        """Cross-pollination health includes status."""
        mock_http = MagicMock()

        result = await handler.handle("/api/v1/health/cross-pollination", {}, mock_http)

        assert result is not None
        import json

        body = json.loads(result.body)
        assert "status" in body

    async def test_cross_pollination_includes_features(self, handler):
        """Cross-pollination health includes features section."""
        mock_http = MagicMock()

        result = await handler.handle("/api/v1/health/cross-pollination", {}, mock_http)

        assert result is not None
        import json

        body = json.loads(result.body)
        assert "features" in body

    async def test_cross_pollination_includes_timestamp(self, handler):
        """Cross-pollination health includes timestamp."""
        mock_http = MagicMock()

        result = await handler.handle("/api/v1/health/cross-pollination", {}, mock_http)

        assert result is not None
        import json

        body = json.loads(result.body)
        assert "timestamp" in body


class TestHealthHandlerKnowledgeMound:
    """Tests for Knowledge Mound health endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return HealthHandler(mock_server_context)

    async def test_knowledge_mound_includes_status(self, handler):
        """Knowledge Mound health includes status."""
        mock_http = MagicMock()

        result = await handler.handle("/api/v1/health/knowledge-mound", {}, mock_http)

        assert result is not None
        import json

        body = json.loads(result.body)
        assert "status" in body

    async def test_knowledge_mound_includes_components(self, handler):
        """Knowledge Mound health includes components section."""
        mock_http = MagicMock()

        result = await handler.handle("/api/v1/health/knowledge-mound", {}, mock_http)

        assert result is not None
        import json

        body = json.loads(result.body)
        assert "components" in body

    async def test_knowledge_mound_includes_timestamp(self, handler):
        """Knowledge Mound health includes timestamp."""
        mock_http = MagicMock()

        result = await handler.handle("/api/v1/health/knowledge-mound", {}, mock_http)

        assert result is not None
        import json

        body = json.loads(result.body)
        assert "timestamp" in body


class TestHealthHandlerPathNormalization:
    """Tests for path normalization in handle method."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return HealthHandler(mock_server_context)

    async def test_v1_path_normalized_to_non_v1(self, handler):
        """Versioned paths are normalized to non-versioned internally."""
        mock_http = MagicMock()

        # Both versioned and non-versioned should work
        result_v1 = await handler.handle("/api/v1/health", {}, mock_http)
        result_nonv1 = await handler.handle("/api/health", {}, mock_http)

        assert result_v1 is not None
        assert result_nonv1 is not None
        # Both should return similar results (both are health checks)
        assert result_v1.status_code == result_nonv1.status_code
