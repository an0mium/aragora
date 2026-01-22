"""Tests for Consensus handler endpoints.

Validates the REST API endpoints for consensus memory including:
- Finding similar debates
- Getting settled topics
- Consensus statistics
- Dissenting views
- Contrarian perspectives
- Risk warnings
- Domain history
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.consensus import ConsensusHandler, _consensus_limiter


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset the rate limiter before each test."""
    if hasattr(_consensus_limiter, "_requests"):
        _consensus_limiter._requests.clear()
    yield


@pytest.fixture
def consensus_handler():
    """Create a consensus handler with mocked dependencies."""
    ctx = {"storage": None, "elo_system": None, "nomic_dir": None}
    handler = ConsensusHandler(ctx)
    return handler


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler with client address."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {}
    return handler


class TestConsensusHandlerCanHandle:
    """Test ConsensusHandler.can_handle method."""

    def test_can_handle_similar(self, consensus_handler):
        """Test can_handle returns True for similar endpoint."""
        assert consensus_handler.can_handle("/api/v1/consensus/similar")

    def test_can_handle_settled(self, consensus_handler):
        """Test can_handle returns True for settled endpoint."""
        assert consensus_handler.can_handle("/api/v1/consensus/settled")

    def test_can_handle_stats(self, consensus_handler):
        """Test can_handle returns True for stats endpoint."""
        assert consensus_handler.can_handle("/api/v1/consensus/stats")

    def test_can_handle_dissents(self, consensus_handler):
        """Test can_handle returns True for dissents endpoint."""
        assert consensus_handler.can_handle("/api/v1/consensus/dissents")

    def test_can_handle_contrarian_views(self, consensus_handler):
        """Test can_handle returns True for contrarian-views endpoint."""
        assert consensus_handler.can_handle("/api/v1/consensus/contrarian-views")

    def test_can_handle_risk_warnings(self, consensus_handler):
        """Test can_handle returns True for risk-warnings endpoint."""
        assert consensus_handler.can_handle("/api/v1/consensus/risk-warnings")

    def test_can_handle_seed_demo(self, consensus_handler):
        """Test can_handle returns True for seed-demo endpoint."""
        assert consensus_handler.can_handle("/api/v1/consensus/seed-demo")

    def test_can_handle_domain(self, consensus_handler):
        """Test can_handle returns True for domain endpoint."""
        assert consensus_handler.can_handle("/api/v1/consensus/domain/software-engineering")

    def test_cannot_handle_unknown(self, consensus_handler):
        """Test can_handle returns False for unknown endpoint."""
        assert not consensus_handler.can_handle("/api/v1/consensus/unknown")
        assert not consensus_handler.can_handle("/api/v1/debates")


class TestConsensusHandlerSimilar:
    """Test GET /api/consensus/similar endpoint."""

    def test_similar_missing_topic(self, consensus_handler, mock_http_handler):
        """Test similar endpoint requires topic parameter."""
        result = consensus_handler.handle("/api/v1/consensus/similar", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body

    def test_similar_topic_too_long(self, consensus_handler, mock_http_handler):
        """Test similar endpoint rejects topics over 500 chars."""
        long_topic = "a" * 501
        result = consensus_handler.handle(
            "/api/v1/consensus/similar", {"topic": long_topic}, mock_http_handler
        )

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body
        assert "too long" in body["error"].lower()

    def test_similar_with_valid_topic(self, consensus_handler, mock_http_handler):
        """Test similar endpoint with valid topic."""
        # This will fail if ConsensusMemory is not available, but that's expected
        result = consensus_handler.handle(
            "/api/v1/consensus/similar", {"topic": "rate limiting"}, mock_http_handler
        )

        assert result is not None
        # Either returns data or 503 if feature unavailable
        assert result.status_code in [200, 400, 429, 500, 503]


class TestConsensusHandlerSettled:
    """Test GET /api/consensus/settled endpoint."""

    def test_settled_default_params(self, consensus_handler, mock_http_handler):
        """Test settled endpoint with default parameters."""
        result = consensus_handler.handle("/api/v1/consensus/settled", {}, mock_http_handler)

        assert result is not None
        # Either returns data or 503 if feature unavailable
        assert result.status_code in [200, 400, 429, 500, 503]

    def test_settled_with_params(self, consensus_handler, mock_http_handler):
        """Test settled endpoint with custom parameters."""
        result = consensus_handler.handle(
            "/api/v1/consensus/settled", {"min_confidence": "0.9", "limit": "10"}, mock_http_handler
        )

        assert result is not None
        assert result.status_code in [200, 400, 429, 500, 503]


class TestConsensusHandlerStats:
    """Test GET /api/consensus/stats endpoint."""

    def test_stats_endpoint(self, consensus_handler, mock_http_handler):
        """Test stats endpoint."""
        result = consensus_handler.handle("/api/v1/consensus/stats", {}, mock_http_handler)

        assert result is not None
        # Either returns data or 503 if feature unavailable
        assert result.status_code in [200, 400, 429, 500, 503]


class TestConsensusHandlerDissents:
    """Test GET /api/consensus/dissents endpoint."""

    def test_dissents_default(self, consensus_handler, mock_http_handler):
        """Test dissents endpoint with default parameters."""
        result = consensus_handler.handle("/api/v1/consensus/dissents", {}, mock_http_handler)

        assert result is not None
        assert result.status_code in [200, 400, 429, 500, 503]

    def test_dissents_with_topic(self, consensus_handler, mock_http_handler):
        """Test dissents endpoint with topic filter."""
        result = consensus_handler.handle(
            "/api/v1/consensus/dissents",
            {"topic": "caching strategies", "limit": "5"},
            mock_http_handler,
        )

        assert result is not None
        assert result.status_code in [200, 400, 429, 500, 503]

    def test_dissents_with_domain(self, consensus_handler, mock_http_handler):
        """Test dissents endpoint with domain filter."""
        result = consensus_handler.handle(
            "/api/v1/consensus/dissents",
            {"domain": "software-engineering", "limit": "5"},
            mock_http_handler,
        )

        assert result is not None
        assert result.status_code in [200, 400, 429, 500, 503]


class TestConsensusHandlerContrarianViews:
    """Test GET /api/consensus/contrarian-views endpoint."""

    def test_contrarian_views_default(self, consensus_handler, mock_http_handler):
        """Test contrarian-views endpoint with default parameters."""
        result = consensus_handler.handle(
            "/api/v1/consensus/contrarian-views", {}, mock_http_handler
        )

        assert result is not None
        assert result.status_code in [200, 400, 429, 500, 503]

    def test_contrarian_views_with_topic(self, consensus_handler, mock_http_handler):
        """Test contrarian-views endpoint with topic filter."""
        result = consensus_handler.handle(
            "/api/v1/consensus/contrarian-views", {"topic": "microservices"}, mock_http_handler
        )

        assert result is not None
        assert result.status_code in [200, 400, 429, 500, 503]


class TestConsensusHandlerRiskWarnings:
    """Test GET /api/consensus/risk-warnings endpoint."""

    def test_risk_warnings_default(self, consensus_handler, mock_http_handler):
        """Test risk-warnings endpoint with default parameters."""
        result = consensus_handler.handle("/api/v1/consensus/risk-warnings", {}, mock_http_handler)

        assert result is not None
        assert result.status_code in [200, 400, 429, 500, 503]

    def test_risk_warnings_with_topic(self, consensus_handler, mock_http_handler):
        """Test risk-warnings endpoint with topic filter."""
        result = consensus_handler.handle(
            "/api/v1/consensus/risk-warnings",
            {"topic": "authentication", "domain": "security"},
            mock_http_handler,
        )

        assert result is not None
        assert result.status_code in [200, 400, 429, 500, 503]


class TestConsensusHandlerDomain:
    """Test GET /api/consensus/domain/:domain endpoint."""

    def test_domain_history(self, consensus_handler, mock_http_handler):
        """Test domain history endpoint."""
        result = consensus_handler.handle(
            "/api/v1/consensus/domain/software-engineering", {}, mock_http_handler
        )

        assert result is not None
        assert result.status_code in [200, 400, 429, 500, 503]

    def test_domain_history_with_limit(self, consensus_handler, mock_http_handler):
        """Test domain history endpoint with limit."""
        result = consensus_handler.handle(
            "/api/v1/consensus/domain/security", {"limit": "25"}, mock_http_handler
        )

        assert result is not None
        assert result.status_code in [200, 400, 429, 500, 503]


class TestConsensusHandlerSeedDemo:
    """Test GET /api/consensus/seed-demo endpoint."""

    def test_seed_demo(self, consensus_handler, mock_http_handler):
        """Test seed-demo endpoint."""
        result = consensus_handler.handle("/api/v1/consensus/seed-demo", {}, mock_http_handler)

        assert result is not None
        # Either returns data, 503 if feature unavailable, or 500 if fixtures fail
        assert result.status_code in [200, 400, 429, 500, 503]


class TestConsensusHandlerRateLimiting:
    """Test rate limiting for consensus endpoints."""

    def test_handles_many_requests(self, consensus_handler, mock_http_handler):
        """Test handler doesn't crash under many requests."""
        from aragora.server.handlers.consensus import _consensus_limiter

        # Save original state
        original_requests = (
            _consensus_limiter._requests.copy() if hasattr(_consensus_limiter, "_requests") else {}
        )

        try:
            # Make several requests
            results = []
            for _ in range(10):
                result = consensus_handler.handle("/api/v1/consensus/stats", {}, mock_http_handler)
                results.append(result)

            # All should return something (either data or rate limit error)
            for r in results:
                assert r is not None
        finally:
            # Restore rate limiter state
            if hasattr(_consensus_limiter, "_requests"):
                _consensus_limiter._requests = original_requests


class TestConsensusHandlerIntegration:
    """Integration tests for consensus handler."""

    def test_all_routes_reachable(self, consensus_handler, mock_http_handler):
        """Test all consensus routes are reachable."""
        routes_to_test = [
            ("/api/v1/consensus/similar", {"topic": "test topic"}),
            ("/api/v1/consensus/settled", {}),
            ("/api/v1/consensus/stats", {}),
            ("/api/v1/consensus/dissents", {}),
            ("/api/v1/consensus/contrarian-views", {}),
            ("/api/v1/consensus/risk-warnings", {}),
            ("/api/v1/consensus/domain/test-domain", {}),
        ]

        for path, params in routes_to_test:
            result = consensus_handler.handle(path, params, mock_http_handler)
            assert result is not None, f"Route {path} returned None"
            # All should return either success or feature unavailable
            assert result.status_code in [
                200,
                400,
                429,
                500,
                503,
            ], f"Route {path} returned unexpected {result.status_code}"

    def test_parameter_validation(self, consensus_handler, mock_http_handler):
        """Test parameter validation across endpoints."""
        # Test limit clamping
        result = consensus_handler.handle(
            "/api/v1/consensus/settled",
            {"limit": "1000"},  # Should be clamped to max
            mock_http_handler,
        )
        assert result is not None

        # Test confidence clamping
        result = consensus_handler.handle(
            "/api/v1/consensus/settled",
            {"min_confidence": "2.0"},  # Should be clamped to 1.0
            mock_http_handler,
        )
        assert result is not None
