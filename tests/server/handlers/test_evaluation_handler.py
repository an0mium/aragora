"""
Tests for the EvaluationHandler module.

Tests cover:
- Handler initialization and routing
- Rate limiter configuration
- Route handling and can_handle method
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from aragora.server.handlers.evaluation import EvaluationHandler


@pytest.fixture
def mock_server_context():
    """Create mock server context for handler initialization."""
    return {"storage": None, "elo_system": None, "nomic_dir": None}


class TestEvaluationHandlerRouting:
    """Tests for handler routing."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return EvaluationHandler(mock_server_context)

    def test_can_handle_evaluate(self, handler):
        """Handler can handle evaluate endpoint."""
        assert handler.can_handle("/api/evaluate")

    def test_can_handle_compare(self, handler):
        """Handler can handle compare endpoint."""
        assert handler.can_handle("/api/evaluate/compare")

    def test_can_handle_dimensions(self, handler):
        """Handler can handle dimensions endpoint."""
        assert handler.can_handle("/api/evaluate/dimensions")

    def test_can_handle_profiles(self, handler):
        """Handler can handle profiles endpoint."""
        assert handler.can_handle("/api/evaluate/profiles")

    def test_cannot_handle_unknown(self, handler):
        """Handler cannot handle unknown paths."""
        assert not handler.can_handle("/api/other")
        assert not handler.can_handle("/api/evaluate/unknown")

    def test_routes_list_complete(self, handler):
        """ROUTES list contains all expected endpoints."""
        assert len(handler.ROUTES) == 4


class TestEvaluationHandlerRateLimiting:
    """Tests for rate limiting."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return EvaluationHandler(mock_server_context)

    def test_rate_limiter_configured(self, handler):
        """Rate limiter is configured for evaluation."""
        from aragora.server.handlers.evaluation import _evaluation_limiter

        # Rate limiter should exist
        assert _evaluation_limiter is not None


class TestEvaluationHandlerRouteDispatch:
    """Tests for route dispatch logic."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return EvaluationHandler(mock_server_context)

    def test_handle_dispatches_to_dimensions(self, handler):
        """Handle dispatches /api/evaluate/dimensions to _list_dimensions."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/evaluate/dimensions", {}, mock_http)

        # Result should be returned (either rate limit, service unavailable, or success)
        assert result is not None

    def test_handle_dispatches_to_profiles(self, handler):
        """Handle dispatches /api/evaluate/profiles to _list_profiles."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/evaluate/profiles", {}, mock_http)

        assert result is not None


class TestEvaluationHandlerUnknownPath:
    """Tests for unknown path handling."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return EvaluationHandler(mock_server_context)

    def test_unknown_get_path_returns_none(self, handler):
        """Unknown GET path returns None."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/other", {}, mock_http)

        assert result is None

    def test_unknown_post_path_returns_none(self, handler):
        """Unknown POST path returns None."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/other", {}, mock_http)

        assert result is None
