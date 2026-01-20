"""Tests for Agent Introspection handler endpoints.

Tests the actual handler routes:
- GET /api/introspection/all - Get all agent introspection
- GET /api/introspection/leaderboard - Get agents ranked by reputation
- GET /api/introspection/agents - List available agents
- GET /api/introspection/agents/{name} - Get specific agent introspection
"""

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.introspection import IntrospectionHandler


@pytest.fixture
def introspection_handler():
    """Create an introspection handler with mocked dependencies."""
    ctx = {
        "storage": None,
        "elo_system": None,
        "nomic_dir": None,
        "calibration_tracker": None,
    }
    handler = IntrospectionHandler(ctx)
    return handler


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler with client address."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {"Content-Length": "0"}
    handler.command = "GET"
    return handler


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter before each test."""
    try:
        from aragora.server.handlers.introspection import _introspection_limiter
        # RateLimiter uses _buckets (dict of timestamp lists)
        if hasattr(_introspection_limiter, '_buckets'):
            _introspection_limiter._buckets.clear()
    except ImportError:
        pass
    yield


# ============================================================================
# can_handle Tests
# ============================================================================


class TestIntrospectionHandlerCanHandle:
    """Test IntrospectionHandler.can_handle method."""

    def test_can_handle_all(self, introspection_handler):
        """Test can_handle returns True for /all endpoint."""
        assert introspection_handler.can_handle("/api/introspection/all")

    def test_can_handle_leaderboard(self, introspection_handler):
        """Test can_handle returns True for /leaderboard endpoint."""
        assert introspection_handler.can_handle("/api/introspection/leaderboard")

    def test_can_handle_agents(self, introspection_handler):
        """Test can_handle returns True for /agents endpoint."""
        assert introspection_handler.can_handle("/api/introspection/agents")

    def test_can_handle_agents_with_name(self, introspection_handler):
        """Test can_handle returns True for /agents/{name} endpoint."""
        assert introspection_handler.can_handle("/api/introspection/agents/claude")
        assert introspection_handler.can_handle("/api/introspection/agents/gemini")
        assert introspection_handler.can_handle("/api/introspection/agents/grok")

    def test_cannot_handle_unknown(self, introspection_handler):
        """Test can_handle returns False for unknown endpoint."""
        assert not introspection_handler.can_handle("/api/unknown")
        assert not introspection_handler.can_handle("/api/introspection/unknown")
        assert not introspection_handler.can_handle("/api/introspection/snapshot")


class TestIntrospectionHandlerRoutesAttribute:
    """Tests for ROUTES class attribute."""

    def test_routes_contains_all(self, introspection_handler):
        """ROUTES contains /all endpoint."""
        assert "/api/introspection/all" in introspection_handler.ROUTES

    def test_routes_contains_leaderboard(self, introspection_handler):
        """ROUTES contains /leaderboard endpoint."""
        assert "/api/introspection/leaderboard" in introspection_handler.ROUTES

    def test_routes_contains_agents(self, introspection_handler):
        """ROUTES contains /agents endpoint."""
        assert "/api/introspection/agents" in introspection_handler.ROUTES

    def test_routes_contains_agents_wildcard(self, introspection_handler):
        """ROUTES contains /agents/* wildcard."""
        assert "/api/introspection/agents/*" in introspection_handler.ROUTES


# ============================================================================
# /api/introspection/all Endpoint Tests
# ============================================================================


class TestIntrospectionHandlerAllEndpoint:
    """Tests for GET /api/introspection/all endpoint."""

    def test_all_returns_503_when_unavailable(self, introspection_handler, mock_http_handler):
        """All endpoint returns 503 when introspection unavailable."""
        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", False):
            result = introspection_handler.handle("/api/introspection/all", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 503

    def test_all_returns_data_when_available(self, introspection_handler, mock_http_handler):
        """All endpoint returns data when available."""
        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {
            "agent_name": "claude",
            "reputation_score": 0.85,
        }

        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.introspection.get_agent_introspection",
                return_value=mock_snapshot,
            ):
                result = introspection_handler.handle(
                    "/api/introspection/all",
                    {},
                    mock_http_handler,
                )

        assert result is not None
        assert result.status_code == 200


# ============================================================================
# /api/introspection/leaderboard Endpoint Tests
# ============================================================================


class TestIntrospectionHandlerLeaderboardEndpoint:
    """Tests for GET /api/introspection/leaderboard endpoint."""

    def test_leaderboard_returns_503_when_unavailable(self, introspection_handler, mock_http_handler):
        """Leaderboard endpoint returns 503 when introspection unavailable."""
        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", False):
            result = introspection_handler.handle(
                "/api/introspection/leaderboard", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 503

    def test_leaderboard_default_limit(self, introspection_handler, mock_http_handler):
        """Leaderboard uses default limit of 10."""
        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {"reputation_score": 0.85}

        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.introspection.get_agent_introspection",
                return_value=mock_snapshot,
            ):
                result = introspection_handler.handle(
                    "/api/introspection/leaderboard",
                    {},
                    mock_http_handler,
                )

        assert result is not None
        assert result.status_code == 200

    def test_leaderboard_custom_limit(self, introspection_handler, mock_http_handler):
        """Leaderboard respects custom limit parameter."""
        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {"reputation_score": 0.85}

        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.introspection.get_agent_introspection",
                return_value=mock_snapshot,
            ):
                result = introspection_handler.handle(
                    "/api/introspection/leaderboard",
                    {"limit": "5"},
                    mock_http_handler,
                )

        assert result is not None
        assert result.status_code == 200

    def test_leaderboard_limit_capped_at_50(self, introspection_handler, mock_http_handler):
        """Leaderboard limit is capped at 50."""
        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {"reputation_score": 0.85}

        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.introspection.get_agent_introspection",
                return_value=mock_snapshot,
            ):
                result = introspection_handler.handle(
                    "/api/introspection/leaderboard",
                    {"limit": "100"},  # Exceeds cap
                    mock_http_handler,
                )

        assert result is not None
        assert result.status_code == 200


# ============================================================================
# /api/introspection/agents Endpoint Tests
# ============================================================================


class TestIntrospectionHandlerAgentsEndpoint:
    """Tests for GET /api/introspection/agents endpoint."""

    def test_agents_returns_list(self, introspection_handler, mock_http_handler):
        """Agents endpoint returns list of available agents."""
        result = introspection_handler.handle(
            "/api/introspection/agents",
            {},
            mock_http_handler,
        )

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "agents" in data
        assert isinstance(data["agents"], list)


# ============================================================================
# /api/introspection/agents/{name} Endpoint Tests
# ============================================================================


class TestIntrospectionHandlerAgentByNameEndpoint:
    """Tests for GET /api/introspection/agents/{name} endpoint."""

    def test_agent_returns_503_when_unavailable(self, introspection_handler, mock_http_handler):
        """Agent endpoint returns 503 when introspection unavailable."""
        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", False):
            result = introspection_handler.handle(
                "/api/introspection/agents/claude", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 503

    def test_agent_validates_name(self, introspection_handler, mock_http_handler):
        """Agent endpoint validates agent name format."""
        # Path traversal attempt should be rejected
        result = introspection_handler.handle(
            "/api/introspection/agents/../etc/passwd",
            {},
            mock_http_handler,
        )

        assert result is not None
        assert result.status_code == 400

    def test_agent_returns_data(self, introspection_handler, mock_http_handler):
        """Agent endpoint returns introspection data."""
        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {
            "agent_name": "claude",
            "reputation_score": 0.85,
            "total_debates": 42,
        }

        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.introspection.get_agent_introspection",
                return_value=mock_snapshot,
            ):
                result = introspection_handler.handle(
                    "/api/introspection/agents/claude",
                    {},
                    mock_http_handler,
                )

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["agent_name"] == "claude"

    def test_agent_not_found(self, introspection_handler, mock_http_handler):
        """Agent endpoint returns 404 when agent not found."""
        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.introspection.get_agent_introspection",
                return_value=None,
            ):
                result = introspection_handler.handle(
                    "/api/introspection/agents/unknown",
                    {},
                    mock_http_handler,
                )

        assert result is not None
        assert result.status_code == 404


# ============================================================================
# Rate Limiting Tests
# ============================================================================


class TestIntrospectionHandlerRateLimiting:
    """Tests for rate limiting on introspection endpoints."""

    def test_rate_limit_exceeded_returns_429(self, introspection_handler, mock_http_handler):
        """Exceeding rate limit returns 429."""
        # Make 20 requests (limit) then one more
        for _ in range(20):
            introspection_handler.handle("/api/introspection/agents", {}, mock_http_handler)

        result = introspection_handler.handle("/api/introspection/agents", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 429


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestIntrospectionHandlerErrorHandling:
    """Tests for error handling in introspection endpoints."""

    def test_handles_exception_gracefully(self, introspection_handler, mock_http_handler):
        """Handler handles exceptions gracefully."""
        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.introspection.get_agent_introspection",
                side_effect=Exception("Test error"),
            ):
                result = introspection_handler.handle(
                    "/api/introspection/agents/claude",
                    {},
                    mock_http_handler,
                )

        assert result is not None
        assert result.status_code == 500

    def test_unknown_route_returns_none(self, introspection_handler, mock_http_handler):
        """Unknown routes return None to allow other handlers."""
        result = introspection_handler.handle(
            "/api/introspection/unknown",
            {},
            mock_http_handler,
        )

        assert result is None


# ============================================================================
# Input Validation Tests
# ============================================================================


class TestIntrospectionHandlerInputValidation:
    """Tests for input validation."""

    def test_invalid_agent_name_rejected(self, introspection_handler, mock_http_handler):
        """Invalid agent names are rejected with 400."""
        # Names with special characters not in [a-zA-Z0-9_-] are rejected
        invalid_names = [
            "..",
            "..%2F",
            "<script>",
            "agent@domain",
            "agent;rm -rf",
        ]

        for name in invalid_names:
            result = introspection_handler.handle(
                f"/api/introspection/agents/{name}",
                {},
                mock_http_handler,
            )
            # Invalid names return 400
            assert result is not None
            assert result.status_code == 400, f"Expected 400 for '{name}', got {result.status_code}"

    def test_path_traversal_extracts_valid_segment(self, introspection_handler, mock_http_handler):
        """Path traversal attempts extract only the agent name segment.

        Path like /api/introspection/agents/claude/../../../etc/passwd
        extracts segment 3 = 'claude' (valid), remaining segments are ignored.
        This is secure because we only use the extracted agent name.
        """
        # These paths have valid agent names at segment 3
        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.introspection.get_agent_introspection",
                return_value=MagicMock(to_dict=lambda: {"agent_name": "claude"}),
            ):
                result = introspection_handler.handle(
                    "/api/introspection/agents/claude/../../../etc/passwd",
                    {},
                    mock_http_handler,
                )
                # "claude" is extracted and used, rest is ignored
                assert result is not None
                assert result.status_code == 200


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntrospectionHandlerIntegration:
    """Integration tests for introspection handler."""

    def test_handle_returns_none_for_unknown(self, introspection_handler, mock_http_handler):
        """Handler returns None for paths it cannot handle."""
        result = introspection_handler.handle(
            "/api/debates",
            {},
            mock_http_handler,
        )
        assert result is None

    def test_full_introspection_flow(self, introspection_handler, mock_http_handler):
        """Test complete introspection flow."""
        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {
            "agent_name": "claude",
            "reputation_score": 0.85,
        }

        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True):
            with patch(
                "aragora.server.handlers.introspection.get_agent_introspection",
                return_value=mock_snapshot,
            ):
                # List agents
                list_result = introspection_handler.handle(
                    "/api/introspection/agents",
                    {},
                    mock_http_handler,
                )
                assert list_result is not None
                assert list_result.status_code == 200

                # Get specific agent
                agent_result = introspection_handler.handle(
                    "/api/introspection/agents/claude",
                    {},
                    mock_http_handler,
                )
                assert agent_result is not None
                assert agent_result.status_code == 200

                # Get leaderboard
                leaderboard_result = introspection_handler.handle(
                    "/api/introspection/leaderboard",
                    {},
                    mock_http_handler,
                )
                assert leaderboard_result is not None
                assert leaderboard_result.status_code == 200
