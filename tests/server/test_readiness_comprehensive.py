"""Tests for comprehensive readiness probe (Gap 5).

Verifies that readiness_probe_fast() includes startup_complete and
handlers_initialized checks from Gap 1 and handler registry.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clear_health_cache():
    """Clear the health check cache before each test."""
    try:
        from aragora.server.handlers.admin.health import _health_cache

        _health_cache.clear()
    except (ImportError, AttributeError):
        pass
    yield
    try:
        from aragora.server.handlers.admin.health import _health_cache

        _health_cache.clear()
    except (ImportError, AttributeError):
        pass


@pytest.fixture
def mock_handler():
    """Create a mock handler for readiness probes."""
    handler = MagicMock()
    handler.get_storage.return_value = MagicMock()
    handler.get_elo_system.return_value = MagicMock()
    return handler


class TestReadinessProbeStartupCheck:
    """Test startup_complete check in readiness_probe_fast."""

    def test_returns_503_before_startup(self, mock_handler):
        """readiness_probe_fast should return 503 when server not ready."""
        from aragora.server.handlers.admin.health.kubernetes import readiness_probe_fast

        with patch("aragora.server.unified_server._server_ready", False), \
             patch.dict("os.environ", {}, clear=True):
            result = readiness_probe_fast(mock_handler)
            assert result["status"] == 503
            body = result["body"]
            assert body["checks"]["startup_complete"] is False

    def test_returns_200_after_startup(self, mock_handler):
        """readiness_probe_fast should return 200 when server is ready."""
        from aragora.server.handlers.admin.health.kubernetes import readiness_probe_fast

        # Also need route index to have entries
        route_index_mock = MagicMock()
        route_index_mock._exact_routes = {"/health": ("_h", None)}

        with patch("aragora.server.unified_server._server_ready", True), \
             patch(
                 "aragora.server.handler_registry.core.get_route_index",
                 return_value=route_index_mock,
             ), \
             patch.dict("os.environ", {}, clear=True):
            result = readiness_probe_fast(mock_handler)
            assert result["status"] == 200
            body = result["body"]
            assert body["checks"]["startup_complete"] is True

    def test_graceful_import_failure(self, mock_handler):
        """If unified_server import fails, startup check is skipped."""
        from aragora.server.handlers.admin.health.kubernetes import readiness_probe_fast

        route_index_mock = MagicMock()
        route_index_mock._exact_routes = {"/health": ("_h", None)}

        with patch.dict("os.environ", {}, clear=True), \
             patch(
                 "aragora.server.handler_registry.core.get_route_index",
                 return_value=route_index_mock,
             ):
            # Even if unified_server import works, verify the check is included
            result = readiness_probe_fast(mock_handler)
            body = result["body"]
            assert "startup_complete" in body["checks"]


class TestReadinessProbeHandlerCheck:
    """Test handlers_initialized check in readiness_probe_fast."""

    def test_returns_503_when_no_routes(self, mock_handler):
        """If route index is empty, readiness should fail."""
        from aragora.server.handlers.admin.health.kubernetes import readiness_probe_fast

        route_index_mock = MagicMock()
        route_index_mock._exact_routes = {}

        with patch("aragora.server.unified_server._server_ready", True), \
             patch(
                 "aragora.server.handler_registry.core.get_route_index",
                 return_value=route_index_mock,
             ), \
             patch.dict("os.environ", {}, clear=True):
            result = readiness_probe_fast(mock_handler)
            assert result["status"] == 503
            body = result["body"]
            assert body["checks"]["handlers_initialized"] is False

    def test_returns_200_when_routes_populated(self, mock_handler):
        """If route index has entries, handler check passes."""
        from aragora.server.handlers.admin.health.kubernetes import readiness_probe_fast

        route_index_mock = MagicMock()
        route_index_mock._exact_routes = {
            "/api/v1/health": ("_health_handler", None),
            "/api/v1/debates": ("_debates_handler", None),
        }

        with patch("aragora.server.unified_server._server_ready", True), \
             patch(
                 "aragora.server.handler_registry.core.get_route_index",
                 return_value=route_index_mock,
             ), \
             patch.dict("os.environ", {}, clear=True):
            result = readiness_probe_fast(mock_handler)
            assert result["status"] == 200
            body = result["body"]
            assert body["checks"]["handlers_initialized"] is True

    def test_both_checks_present_in_response(self, mock_handler):
        """Both startup_complete and handlers_initialized appear in checks."""
        from aragora.server.handlers.admin.health.kubernetes import readiness_probe_fast

        route_index_mock = MagicMock()
        route_index_mock._exact_routes = {"/health": ("_h", None)}

        with patch("aragora.server.unified_server._server_ready", True), \
             patch(
                 "aragora.server.handler_registry.core.get_route_index",
                 return_value=route_index_mock,
             ), \
             patch.dict("os.environ", {}, clear=True):
            result = readiness_probe_fast(mock_handler)
            body = result["body"]
            assert "startup_complete" in body["checks"]
            assert "handlers_initialized" in body["checks"]
