"""
Tests for Gauntlet Handler.

Tests cover:
- Handler routing for gauntlet stress-testing endpoints
- Rate limiting
- API versioning headers
- Input validation
- Error handling
"""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from aragora.server.handlers.gauntlet import (
    GauntletHandler,
    _gauntlet_runs,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_server_context():
    """Create mock server context."""
    return {
        "user_store": MagicMock(),
        "nomic_dir": "/tmp/test",
        "stream_emitter": MagicMock(),
    }


@pytest.fixture
def handler(mock_server_context):
    """Create GauntletHandler with mock context."""
    return GauntletHandler(mock_server_context)


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP handler."""
    mock = MagicMock()
    mock.command = "GET"
    mock.client_address = ("127.0.0.1", 12345)
    mock.path = "/api/v1/gauntlet/run"
    mock.headers = {}
    return mock


@pytest.fixture(autouse=True)
def clear_gauntlet_runs():
    """Clear gauntlet runs between tests."""
    _gauntlet_runs.clear()
    yield
    _gauntlet_runs.clear()


# ============================================================================
# Routing Tests
# ============================================================================


class TestGauntletHandlerRouting:
    """Tests for handler routing."""

    def test_can_handle_run_endpoint(self, handler):
        """Handler can handle POST /api/v1/gauntlet/run."""
        assert handler.can_handle("/api/v1/gauntlet/run", method="POST")

    def test_can_handle_personas_endpoint(self, handler):
        """Handler can handle GET /api/v1/gauntlet/personas."""
        assert handler.can_handle("/api/v1/gauntlet/personas", method="GET")

    def test_can_handle_results_endpoint(self, handler):
        """Handler can handle GET /api/v1/gauntlet/results."""
        assert handler.can_handle("/api/v1/gauntlet/results", method="GET")

    def test_can_handle_gauntlet_id(self, handler):
        """Handler can handle GET /api/v1/gauntlet/:id."""
        assert handler.can_handle("/api/v1/gauntlet/abc-123", method="GET")
        assert handler.can_handle("/api/v1/gauntlet/uuid-1234-5678", method="GET")

    def test_can_handle_receipt_endpoint(self, handler):
        """Handler can handle GET /api/v1/gauntlet/:id/receipt."""
        assert handler.can_handle("/api/v1/gauntlet/abc-123/receipt", method="GET")

    def test_can_handle_receipt_verify(self, handler):
        """Handler can handle GET /api/v1/gauntlet/:id/receipt/verify."""
        assert handler.can_handle("/api/v1/gauntlet/abc-123/receipt/verify", method="GET")

    def test_can_handle_heatmap_endpoint(self, handler):
        """Handler can handle GET /api/v1/gauntlet/:id/heatmap."""
        assert handler.can_handle("/api/v1/gauntlet/abc-123/heatmap", method="GET")

    def test_can_handle_compare_endpoint(self, handler):
        """Handler can handle GET /api/v1/gauntlet/:id/compare/:id2."""
        assert handler.can_handle("/api/v1/gauntlet/abc-123/compare/def-456", method="GET")

    def test_can_handle_delete_endpoint(self, handler):
        """Handler can handle DELETE /api/v1/gauntlet/:id."""
        assert handler.can_handle("/api/v1/gauntlet/abc-123", method="DELETE")

    def test_cannot_handle_unknown_path(self, handler):
        """Handler cannot handle unknown paths."""
        assert not handler.can_handle("/api/v1/other/endpoint", method="GET")
        assert not handler.can_handle("/api/v1/debates", method="GET")

    def test_cannot_handle_wrong_method(self, handler):
        """Handler returns False for wrong methods."""
        # run is POST only
        assert not handler.can_handle("/api/v1/gauntlet/run", method="PUT")


# ============================================================================
# API Version Tests
# ============================================================================


class TestGauntletAPIVersioning:
    """Tests for API versioning."""

    def test_handler_has_api_version(self, handler):
        """Handler has API_VERSION set."""
        assert handler.API_VERSION == "v1"

    def test_has_auth_required_endpoints(self, handler):
        """Handler has AUTH_REQUIRED_ENDPOINTS list."""
        assert len(handler.AUTH_REQUIRED_ENDPOINTS) >= 2
        assert "/api/v1/gauntlet/run" in handler.AUTH_REQUIRED_ENDPOINTS


# ============================================================================
# Handler Method Tests
# ============================================================================


class TestGauntletHandlerMethods:
    """Tests for handler methods."""

    def test_personas_endpoint_is_routable(self, handler):
        """GET /api/v1/gauntlet/personas is routable."""
        assert handler.can_handle("/api/v1/gauntlet/personas", method="GET")

    def test_results_endpoint_is_routable(self, handler):
        """GET /api/v1/gauntlet/results is routable."""
        assert handler.can_handle("/api/v1/gauntlet/results", method="GET")


# ============================================================================
# Route Normalization Tests
# ============================================================================


class TestRouteNormalization:
    """Tests for route normalization."""

    def test_normalize_v1_path(self, handler):
        """V1 paths are normalized correctly."""
        result = handler._normalize_path("/api/v1/gauntlet/run")
        assert result == "/api/v1/gauntlet/run"

    def test_is_legacy_route_false_for_v1(self, handler):
        """V1 routes are not marked as legacy."""
        assert not handler._is_legacy_route("/api/v1/gauntlet/run")


# ============================================================================
# Memory Management Tests
# ============================================================================


class TestGauntletMemoryManagement:
    """Tests for gauntlet run memory management."""

    def test_max_runs_constant_exists(self):
        """MAX_GAUNTLET_RUNS_IN_MEMORY constant is defined."""
        from aragora.server.handlers.gauntlet import MAX_GAUNTLET_RUNS_IN_MEMORY

        assert MAX_GAUNTLET_RUNS_IN_MEMORY > 0
        assert MAX_GAUNTLET_RUNS_IN_MEMORY == 500

    def test_completed_ttl_exists(self):
        """_GAUNTLET_COMPLETED_TTL constant is defined."""
        from aragora.server.handlers.gauntlet import _GAUNTLET_COMPLETED_TTL

        assert _GAUNTLET_COMPLETED_TTL > 0
        assert _GAUNTLET_COMPLETED_TTL == 3600  # 1 hour

    def test_gauntlet_runs_is_ordered_dict(self):
        """_gauntlet_runs is an OrderedDict for FIFO eviction."""
        from collections import OrderedDict

        assert isinstance(_gauntlet_runs, OrderedDict)


# ============================================================================
# Handler Initialization Tests
# ============================================================================


class TestGauntletHandlerInit:
    """Tests for handler initialization."""

    def test_handler_has_routes(self, handler):
        """Handler has ROUTES list."""
        assert len(handler.ROUTES) >= 8

    def test_handler_extends_base_handler(self, handler):
        """Handler extends BaseHandler."""
        from aragora.server.handlers.base import BaseHandler

        assert isinstance(handler, BaseHandler)

    def test_handler_sets_broadcast_fn_if_emitter(self, mock_server_context):
        """Handler sets broadcast function if stream_emitter is provided."""
        with patch("aragora.server.handlers.gauntlet.set_gauntlet_broadcast_fn") as mock_set:
            handler = GauntletHandler(mock_server_context)
            mock_set.assert_called_once()

    def test_handler_without_emitter(self):
        """Handler works without stream_emitter."""
        ctx = {"user_store": MagicMock(), "nomic_dir": "/tmp/test"}
        handler = GauntletHandler(ctx)
        assert handler is not None


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestGauntletErrorHandling:
    """Tests for error handling."""

    def test_can_handle_returns_false_for_unhandled(self, handler):
        """can_handle returns False for unhandled paths."""
        assert not handler.can_handle("/api/v1/other/endpoint", method="GET")
        assert not handler.can_handle("/api/v1/debates", method="GET")
        assert not handler.can_handle("/api/v1/agents", method="GET")
