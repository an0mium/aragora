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
        assert health_handler.can_handle("/api/health") is True

    def test_can_handle_api_health_detailed(self, health_handler):
        """Test that handler recognizes /api/health/detailed route."""
        assert health_handler.can_handle("/api/health/detailed") is True

    def test_can_handle_api_health_deep(self, health_handler):
        """Test that handler recognizes /api/health/deep route."""
        assert health_handler.can_handle("/api/health/deep") is True

    def test_cannot_handle_unknown_path(self, health_handler):
        """Test that handler rejects unknown paths."""
        assert health_handler.can_handle("/unknown") is False
        assert health_handler.can_handle("/api/debates") is False


class TestLivenessProbe:
    """Tests for /healthz liveness probe."""

    def test_liveness_returns_ok(self, health_handler):
        """Liveness probe should always return ok if server is running."""
        result = health_handler.handle("/healthz", {}, None)

        assert result is not None
        body = json.loads(result.body)
        assert body["status"] == "ok"
        assert result.status_code == 200


class TestReadinessProbe:
    """Tests for /readyz readiness probe."""

    def test_readiness_with_no_deps_returns_ready(self, health_handler):
        """Readiness should return ready when no deps configured."""
        with patch.object(health_handler, 'get_storage', return_value=None):
            with patch.object(health_handler, 'get_elo_system', return_value=None):
                result = health_handler.handle("/readyz", {}, None)

        assert result is not None
        body = json.loads(result.body)
        assert body["status"] == "ready"
        assert result.status_code == 200

    def test_readiness_with_storage_returns_ready(self, health_handler):
        """Readiness should return ready when storage is available."""
        mock_storage = MagicMock()

        with patch.object(health_handler, 'get_storage', return_value=mock_storage):
            with patch.object(health_handler, 'get_elo_system', return_value=None):
                result = health_handler.handle("/readyz", {}, None)

        assert result is not None
        body = json.loads(result.body)
        assert body["status"] == "ready"
        assert body["checks"]["storage"] is True
        assert result.status_code == 200

    def test_readiness_with_storage_error_returns_not_ready(self, health_handler):
        """Readiness should return not_ready when storage fails."""
        with patch.object(health_handler, 'get_storage', side_effect=RuntimeError("DB error")):
            with patch.object(health_handler, 'get_elo_system', return_value=None):
                result = health_handler.handle("/readyz", {}, None)

        assert result is not None
        body = json.loads(result.body)
        assert body["status"] == "not_ready"
        assert body["checks"]["storage"] is False
        assert result.status_code == 503


class TestComprehensiveHealthCheck:
    """Tests for /api/health comprehensive health check."""

    def test_health_returns_status(self, health_handler):
        """Health check should return status and checks."""
        with patch.object(health_handler, 'get_storage', return_value=None):
            with patch.object(health_handler, 'get_elo_system', return_value=None):
                with patch.object(health_handler, 'get_nomic_dir', return_value=None):
                    result = health_handler.handle("/api/health", {}, None)

        assert result is not None
        body = json.loads(result.body)
        assert "status" in body
        assert "checks" in body
        assert "timestamp" in body
        assert "version" in body

    def test_health_includes_uptime(self, health_handler):
        """Health check should include uptime."""
        with patch.object(health_handler, 'get_storage', return_value=None):
            with patch.object(health_handler, 'get_elo_system', return_value=None):
                with patch.object(health_handler, 'get_nomic_dir', return_value=None):
                    result = health_handler.handle("/api/health", {}, None)

        body = json.loads(result.body)
        assert "uptime_seconds" in body
        assert body["uptime_seconds"] >= 0

    def test_health_includes_response_time(self, health_handler):
        """Health check should include response time."""
        with patch.object(health_handler, 'get_storage', return_value=None):
            with patch.object(health_handler, 'get_elo_system', return_value=None):
                with patch.object(health_handler, 'get_nomic_dir', return_value=None):
                    result = health_handler.handle("/api/health", {}, None)

        body = json.loads(result.body)
        assert "response_time_ms" in body
        assert body["response_time_ms"] >= 0


class TestDetailedHealthCheck:
    """Tests for /api/health/detailed endpoint."""

    def test_detailed_health_returns_components(self, health_handler):
        """Detailed health should return component status."""
        with patch.object(health_handler, 'get_storage', return_value=None):
            with patch.object(health_handler, 'get_elo_system', return_value=None):
                with patch.object(health_handler, 'get_nomic_dir', return_value=None):
                    result = health_handler.handle("/api/health/detailed", {}, None)

        assert result is not None
        body = json.loads(result.body)
        assert "components" in body
        assert "storage" in body["components"]
        assert "elo_system" in body["components"]


class TestDeepHealthCheck:
    """Tests for /api/health/deep endpoint."""

    def test_deep_health_returns_comprehensive_checks(self, health_handler):
        """Deep health should return comprehensive checks."""
        with patch.object(health_handler, 'get_storage', return_value=None):
            with patch.object(health_handler, 'get_elo_system', return_value=None):
                with patch.object(health_handler, 'get_nomic_dir', return_value=None):
                    result = health_handler.handle("/api/health/deep", {}, None)

        assert result is not None
        body = json.loads(result.body)
        assert "status" in body
        assert "checks" in body
        assert "response_time_ms" in body

    def test_deep_health_includes_ai_providers(self, health_handler):
        """Deep health should check AI provider availability."""
        with patch.object(health_handler, 'get_storage', return_value=None):
            with patch.object(health_handler, 'get_elo_system', return_value=None):
                with patch.object(health_handler, 'get_nomic_dir', return_value=None):
                    result = health_handler.handle("/api/health/deep", {}, None)

        body = json.loads(result.body)
        assert "ai_providers" in body["checks"]


class TestFilesystemHealthCheck:
    """Tests for filesystem health check helper."""

    def test_filesystem_check_with_temp_dir(self, health_handler):
        """Filesystem check should work with temp directory."""
        with patch.object(health_handler, 'get_nomic_dir', return_value=None):
            result = health_handler._check_filesystem_health()

        assert result["healthy"] is True
        assert "path" in result

    def test_filesystem_check_returns_error_on_permission_denied(self, health_handler):
        """Filesystem check should return error on permission denied."""
        with patch.object(health_handler, 'get_nomic_dir', return_value=None):
            with patch('pathlib.Path.write_text', side_effect=PermissionError("denied")):
                result = health_handler._check_filesystem_health()

        assert result["healthy"] is False
        assert "Permission denied" in result["error"]


class TestRedisHealthCheck:
    """Tests for Redis health check helper."""

    def test_redis_check_without_config(self, health_handler):
        """Redis check should return healthy when not configured."""
        with patch.dict('os.environ', {}, clear=True):
            result = health_handler._check_redis_health()

        assert result["healthy"] is True
        assert result["configured"] is False


class TestAIProvidersHealthCheck:
    """Tests for AI providers health check helper."""

    def test_ai_providers_check_without_keys(self, health_handler):
        """AI providers check should work without any keys."""
        with patch.dict('os.environ', {}, clear=True):
            result = health_handler._check_ai_providers_health()

        assert result["healthy"] is True
        assert result["any_available"] is False
        assert result["available_count"] == 0

    def test_ai_providers_check_with_anthropic_key(self, health_handler):
        """AI providers check should detect Anthropic key."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'sk-ant-test12345'}, clear=True):
            result = health_handler._check_ai_providers_health()

        assert result["healthy"] is True
        assert result["any_available"] is True
        assert result["providers"]["anthropic"] is True
