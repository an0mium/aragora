"""
Tests for aragora.server.handlers.evolution - Evolution A/B testing handlers.

Tests cover:
- EvolutionABTestingHandler routing
- Module unavailability handling
- Error responses
"""

from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# Test Fixtures
# ===========================================================================


def make_mock_handler(
    body: dict | None = None,
    method: str = "GET",
    headers: dict | None = None,
):
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.command = method
    handler.headers = headers or {}
    handler.client_address = ("127.0.0.1", 12345)

    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        handler.headers["Content-Length"] = str(len(body_bytes))
        handler.rfile = BytesIO(body_bytes)
        handler.request_body = body_bytes
    else:
        handler.rfile = BytesIO(b"")
        handler.headers["Content-Length"] = "0"
        handler.request_body = b"{}"

    return handler


def get_status(result) -> int:
    """Extract status code from HandlerResult or tuple."""
    if hasattr(result, "status_code"):
        return result.status_code
    return result[1]


def get_body(result) -> dict:
    """Extract body from HandlerResult or tuple."""
    if hasattr(result, "body"):
        body = result.body
        if isinstance(body, bytes):
            return json.loads(body.decode("utf-8"))
        return json.loads(body)
    body = result[0]
    if isinstance(body, dict):
        return body
    return json.loads(body)


@pytest.fixture
def evolution_handler():
    """Create EvolutionABTestingHandler."""
    from aragora.server.handlers.evolution.ab_testing import EvolutionABTestingHandler

    ctx = {"ab_tests_db": ":memory:"}
    return EvolutionABTestingHandler(ctx)


# ===========================================================================
# Test Routing
# ===========================================================================


class TestEvolutionHandlerRouting:
    """Tests for EvolutionABTestingHandler routing."""

    def test_can_handle_evolution_paths(self, evolution_handler):
        """Test handler recognizes evolution paths."""
        assert evolution_handler.can_handle("/api/v1/evolution/ab-tests") is True
        assert evolution_handler.can_handle("/api/v1/evolution/ab-tests/") is True
        assert evolution_handler.can_handle("/api/v1/evolution/ab-tests/test-123") is True
        assert evolution_handler.can_handle("/api/v1/evolution/ab-tests/gpt-4/active") is True

    def test_cannot_handle_non_evolution_paths(self, evolution_handler):
        """Test handler rejects non-evolution paths."""
        assert evolution_handler.can_handle("/api/v1/debates") is False
        assert evolution_handler.can_handle("/api/v1/admin/users") is False
        assert evolution_handler.can_handle("/api/v1/memory/tiers") is False


# ===========================================================================
# Test Module Not Available
# ===========================================================================


class TestModuleNotAvailable:
    """Tests for when A/B testing module is not available."""

    def test_get_returns_503_when_unavailable(self):
        """Test GET returns 503 when module not available."""
        from aragora.server.handlers.evolution.ab_testing import EvolutionABTestingHandler

        with patch("aragora.server.handlers.evolution.ab_testing.AB_TESTING_AVAILABLE", False):
            handler_obj = EvolutionABTestingHandler({})
            http_handler = make_mock_handler()

            result = handler_obj.handle("/api/v1/evolution/ab-tests", {}, http_handler)

            assert result is not None
            assert get_status(result) == 503
            data = get_body(result)
            assert "not available" in data["error"].lower()

    def test_post_returns_503_when_unavailable(self):
        """Test POST returns 503 when module not available."""
        from aragora.server.handlers.evolution.ab_testing import EvolutionABTestingHandler

        with patch("aragora.server.handlers.evolution.ab_testing.AB_TESTING_AVAILABLE", False):
            handler_obj = EvolutionABTestingHandler({})
            http_handler = make_mock_handler(body={}, method="POST")

            result = handler_obj.handle_post("/api/v1/evolution/ab-tests", {}, http_handler)

            assert result is not None
            assert get_status(result) == 503

    def test_delete_returns_503_when_unavailable(self):
        """Test DELETE returns 503 when module not available."""
        from aragora.server.handlers.evolution.ab_testing import EvolutionABTestingHandler

        with patch("aragora.server.handlers.evolution.ab_testing.AB_TESTING_AVAILABLE", False):
            handler_obj = EvolutionABTestingHandler({})
            http_handler = make_mock_handler(method="DELETE")

            result = handler_obj.handle_delete(
                "/api/v1/evolution/ab-tests/test-1", {}, http_handler
            )

            assert result is not None
            assert get_status(result) == 503


# ===========================================================================
# Test Invalid Path Segments
# ===========================================================================


class TestPathValidation:
    """Tests for path segment validation."""

    def test_invalid_test_id_rejected(self, evolution_handler):
        """Test invalid test ID is rejected."""
        http_handler = make_mock_handler()

        # Path traversal attempt
        result = evolution_handler.handle(
            "/api/v1/evolution/ab-tests/../../../etc/passwd", {}, http_handler
        )

        # Should return 400 for invalid path segment
        assert result is not None
        assert get_status(result) == 400

    def test_empty_test_id_returns_list(self, evolution_handler):
        """Test empty path returns list endpoint."""
        http_handler = make_mock_handler()

        # This should be handled by the list_tests endpoint
        result = evolution_handler.handle("/api/v1/evolution/ab-tests/", {}, http_handler)

        # Should return tests list (or error if not configured)
        assert result is not None


# ===========================================================================
# Test Manager Property
# ===========================================================================


class TestManagerProperty:
    """Tests for lazy manager initialization."""

    def test_manager_lazy_loaded(self):
        """Test manager is lazy-loaded on first access."""
        from aragora.server.handlers.evolution.ab_testing import EvolutionABTestingHandler

        handler = EvolutionABTestingHandler({"ab_tests_db": ":memory:"})

        # Manager should not be initialized yet
        assert handler._manager is None

        # Access manager property
        with patch("aragora.server.handlers.evolution.ab_testing.AB_TESTING_AVAILABLE", True):
            with patch("aragora.server.handlers.evolution.ab_testing.ABTestManager") as mock_class:
                mock_class.return_value = MagicMock()
                _ = handler.manager

                # Manager should now be initialized
                mock_class.assert_called_once_with(db_path=":memory:")

    def test_manager_returns_none_when_unavailable(self):
        """Test manager returns None when module unavailable."""
        from aragora.server.handlers.evolution.ab_testing import EvolutionABTestingHandler

        with patch("aragora.server.handlers.evolution.ab_testing.AB_TESTING_AVAILABLE", False):
            handler = EvolutionABTestingHandler({})

            # Manager should be None when module unavailable
            assert handler.manager is None
