"""Tests for comprehensive readiness probe (Gap 5).

Verifies that readiness_probe_fast() includes startup_complete and
handlers_initialized checks from Gap 1 and handler registry.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset health cache, server_ready flag, and degraded mode before each test."""
    import aragora.server.unified_server as usrv

    original_ready = usrv._server_ready
    usrv._server_ready = False

    # Clear degraded mode (may be set by other tests)
    try:
        from aragora.server.degraded_mode import clear_degraded

        clear_degraded()
    except ImportError:
        pass

    try:
        from aragora.server.handlers.admin.health import (
            _HEALTH_CACHE,
            _HEALTH_CACHE_TIMESTAMPS,
        )

        _HEALTH_CACHE.clear()
        _HEALTH_CACHE_TIMESTAMPS.clear()
    except (ImportError, AttributeError):
        pass

    yield

    usrv._server_ready = original_ready

    try:
        from aragora.server.degraded_mode import clear_degraded

        clear_degraded()
    except ImportError:
        pass

    try:
        from aragora.server.handlers.admin.health import (
            _HEALTH_CACHE,
            _HEALTH_CACHE_TIMESTAMPS,
        )

        _HEALTH_CACHE.clear()
        _HEALTH_CACHE_TIMESTAMPS.clear()
    except (ImportError, AttributeError):
        pass


@pytest.fixture
def mock_handler():
    """Create a mock handler for readiness probes."""
    handler = MagicMock()
    handler.get_storage.return_value = MagicMock()
    handler.get_elo_system.return_value = MagicMock()
    return handler


def _parse_probe_result(result):
    """Parse a HandlerResult dataclass into (status_code, body_dict)."""
    body = json.loads(result.body.decode("utf-8"))
    return result.status_code, body


class TestReadinessProbeStartupCheck:
    """Test startup_complete check in readiness_probe_fast."""

    def test_returns_503_before_startup(self, mock_handler):
        """readiness_probe_fast should return 503 when server not ready."""
        import aragora.server.unified_server as usrv
        from aragora.server.handlers.admin.health.kubernetes import readiness_probe_fast

        usrv._server_ready = False
        with patch.dict("os.environ", {}, clear=True):
            result = readiness_probe_fast(mock_handler)
            status, body = _parse_probe_result(result)
            assert status == 503
            assert body["checks"]["startup_complete"] is False

    def test_returns_200_after_startup(self, mock_handler):
        """readiness_probe_fast should return 200 when server is ready."""
        import aragora.server.unified_server as usrv
        from aragora.server.handlers.admin.health.kubernetes import readiness_probe_fast

        usrv._server_ready = True

        route_index_mock = MagicMock()
        route_index_mock._exact_routes = {"/health": ("_h", None)}

        with (
            patch(
                "aragora.server.handler_registry.core.get_route_index",
                return_value=route_index_mock,
            ),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = readiness_probe_fast(mock_handler)
            status, body = _parse_probe_result(result)
            assert status == 200
            assert body["checks"]["startup_complete"] is True

    def test_graceful_import_failure(self, mock_handler):
        """Startup check is included in the response."""
        import aragora.server.unified_server as usrv
        from aragora.server.handlers.admin.health.kubernetes import readiness_probe_fast

        usrv._server_ready = True

        route_index_mock = MagicMock()
        route_index_mock._exact_routes = {"/health": ("_h", None)}

        with (
            patch(
                "aragora.server.handler_registry.core.get_route_index",
                return_value=route_index_mock,
            ),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = readiness_probe_fast(mock_handler)
            _, body = _parse_probe_result(result)
            assert "startup_complete" in body["checks"]


class TestReadinessProbeHandlerCheck:
    """Test handlers_initialized check in readiness_probe_fast."""

    def test_returns_503_when_no_routes(self, mock_handler):
        """If route index is empty, readiness should fail."""
        import aragora.server.unified_server as usrv
        from aragora.server.handlers.admin.health.kubernetes import readiness_probe_fast

        usrv._server_ready = True

        route_index_mock = MagicMock()
        route_index_mock._exact_routes = {}

        with (
            patch(
                "aragora.server.handler_registry.core.get_route_index",
                return_value=route_index_mock,
            ),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = readiness_probe_fast(mock_handler)
            status, body = _parse_probe_result(result)
            assert status == 503
            assert body["checks"]["handlers_initialized"] is False

    def test_returns_200_when_routes_populated(self, mock_handler):
        """If route index has entries, handler check passes."""
        import aragora.server.unified_server as usrv
        from aragora.server.handlers.admin.health.kubernetes import readiness_probe_fast

        usrv._server_ready = True

        route_index_mock = MagicMock()
        route_index_mock._exact_routes = {
            "/api/v1/health": ("_health_handler", None),
            "/api/v1/debates": ("_debates_handler", None),
        }

        with (
            patch(
                "aragora.server.handler_registry.core.get_route_index",
                return_value=route_index_mock,
            ),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = readiness_probe_fast(mock_handler)
            status, body = _parse_probe_result(result)
            assert status == 200
            assert body["checks"]["handlers_initialized"] is True

    def test_both_checks_present_in_response(self, mock_handler):
        """Both startup_complete and handlers_initialized appear in checks."""
        import aragora.server.unified_server as usrv
        from aragora.server.handlers.admin.health.kubernetes import readiness_probe_fast

        usrv._server_ready = True

        route_index_mock = MagicMock()
        route_index_mock._exact_routes = {"/health": ("_h", None)}

        with (
            patch(
                "aragora.server.handler_registry.core.get_route_index",
                return_value=route_index_mock,
            ),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = readiness_probe_fast(mock_handler)
            _, body = _parse_probe_result(result)
            assert "startup_complete" in body["checks"]
            assert "handlers_initialized" in body["checks"]
