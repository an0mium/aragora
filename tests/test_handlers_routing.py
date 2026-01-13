"""
Tests for the Routing Handler endpoints.

Covers:
- GET /api/routing/best-teams - Get best-performing team combinations
- POST /api/routing/recommendations - Get agent recommendations for a task
- POST /api/routing/auto-route - Auto-route task with domain detection
- POST /api/routing/detect-domain - Detect domain from task text
- GET /api/routing/domain-leaderboard - Get agents ranked by domain
"""

from __future__ import annotations

import io
import json
from unittest.mock import Mock, patch, MagicMock

import pytest

from aragora.server.handlers.routing import RoutingHandler


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def routing_handler(handler_context):
    """Create a RoutingHandler with mock context."""
    return RoutingHandler(handler_context)


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler object."""
    handler = Mock()
    handler.headers = {"Content-Type": "application/json"}
    handler.command = "GET"
    return handler


@pytest.fixture
def mock_http_handler_post():
    """Create a mock HTTP handler for POST requests."""

    def create_handler(body: dict):
        handler = Mock()
        handler.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(json.dumps(body))),
        }
        handler.command = "POST"
        handler.rfile = io.BytesIO(json.dumps(body).encode())
        return handler

    return create_handler


# =============================================================================
# can_handle Tests
# =============================================================================


class TestCanHandle:
    """Test route matching for RoutingHandler."""

    def test_can_handle_best_teams_route(self, routing_handler):
        """Test matching best-teams endpoint."""
        assert routing_handler.can_handle("/api/routing/best-teams")

    def test_can_handle_recommendations_route(self, routing_handler):
        """Test matching recommendations endpoint."""
        assert routing_handler.can_handle("/api/routing/recommendations")

    def test_can_handle_auto_route_route(self, routing_handler):
        """Test matching auto-route endpoint."""
        assert routing_handler.can_handle("/api/routing/auto-route")

    def test_can_handle_detect_domain_route(self, routing_handler):
        """Test matching detect-domain endpoint."""
        assert routing_handler.can_handle("/api/routing/detect-domain")

    def test_can_handle_domain_leaderboard_route(self, routing_handler):
        """Test matching domain-leaderboard endpoint."""
        assert routing_handler.can_handle("/api/routing/domain-leaderboard")

    def test_cannot_handle_unknown_route(self, routing_handler):
        """Test rejection of unknown routes."""
        assert not routing_handler.can_handle("/api/unknown")
        assert not routing_handler.can_handle("/api/routing")
        assert not routing_handler.can_handle("/api/routing/unknown")


# =============================================================================
# Best Teams Endpoint Tests
# =============================================================================


class TestBestTeamsEndpoint:
    """Test /api/routing/best-teams endpoint."""

    def test_best_teams_routing_unavailable_returns_503(self, routing_handler, mock_http_handler):
        """Test error when routing module unavailable."""
        with patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", False):
            result = routing_handler.handle("/api/routing/best-teams", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 503

    def test_best_teams_with_parameters(self, routing_handler, mock_http_handler):
        """Test best-teams endpoint with query parameters."""
        mock_selector = MagicMock()
        mock_selector.get_best_team_combinations.return_value = [
            {"team": ["claude", "gpt4"], "score": 0.9},
            {"team": ["claude", "gemini"], "score": 0.85},
        ]

        with (
            patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True),
            patch("aragora.server.handlers.routing.AgentSelector", return_value=mock_selector),
        ):
            result = routing_handler.handle(
                "/api/routing/best-teams",
                {"min_debates": ["5"], "limit": ["10"]},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["min_debates"] == 5
        assert len(body["combinations"]) == 2

    def test_best_teams_limit_clamped(self, routing_handler, mock_http_handler):
        """Test limit parameter is clamped to valid range."""
        mock_selector = MagicMock()
        mock_selector.get_best_team_combinations.return_value = []

        with (
            patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True),
            patch("aragora.server.handlers.routing.AgentSelector", return_value=mock_selector),
        ):
            # Should not error on extreme values
            result = routing_handler.handle(
                "/api/routing/best-teams",
                {"limit": ["1000"]},  # Will be clamped to 50
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 200


# =============================================================================
# Domain Leaderboard Endpoint Tests
# =============================================================================


class TestDomainLeaderboardEndpoint:
    """Test /api/routing/domain-leaderboard endpoint."""

    def test_leaderboard_routing_unavailable_returns_503(self, routing_handler, mock_http_handler):
        """Test error when routing module unavailable."""
        with patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", False):
            result = routing_handler.handle(
                "/api/routing/domain-leaderboard", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 503

    def test_leaderboard_with_domain_parameter(self, routing_handler, mock_http_handler):
        """Test leaderboard endpoint with domain parameter."""
        mock_selector = MagicMock()
        mock_selector.get_domain_leaderboard.return_value = [
            {"agent": "claude", "score": 1300},
            {"agent": "gpt4", "score": 1250},
        ]

        with (
            patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True),
            patch("aragora.server.handlers.routing.AgentSelector") as MockSelector,
        ):
            MockSelector.create_with_defaults.return_value = mock_selector
            result = routing_handler.handle(
                "/api/routing/domain-leaderboard",
                {"domain": ["coding"], "limit": ["5"]},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["domain"] == "coding"
        assert len(body["leaderboard"]) == 2

    def test_leaderboard_default_domain(self, routing_handler, mock_http_handler):
        """Test leaderboard endpoint with default domain."""
        mock_selector = MagicMock()
        mock_selector.get_domain_leaderboard.return_value = []

        with (
            patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True),
            patch("aragora.server.handlers.routing.AgentSelector") as MockSelector,
        ):
            MockSelector.create_with_defaults.return_value = mock_selector
            result = routing_handler.handle(
                "/api/routing/domain-leaderboard", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["domain"] == "general"


# =============================================================================
# Recommendations Endpoint Tests (POST)
# =============================================================================


class TestRecommendationsEndpoint:
    """Test /api/routing/recommendations endpoint."""

    def test_recommendations_routing_unavailable_returns_503(
        self, routing_handler, mock_http_handler_post
    ):
        """Test error when routing module unavailable."""
        handler = mock_http_handler_post({"primary_domain": "coding"})

        with patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", False):
            result = routing_handler.handle_post("/api/routing/recommendations", {}, handler)

        assert result is not None
        assert result.status_code == 503

    def test_recommendations_invalid_json_returns_400(self, routing_handler):
        """Test error on invalid JSON body."""
        handler = Mock()
        handler.headers = {"Content-Type": "application/json", "Content-Length": "10"}
        handler.rfile = io.BytesIO(b"invalid json")

        with patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True):
            result = routing_handler.handle_post("/api/routing/recommendations", {}, handler)

        assert result is not None
        assert result.status_code == 400

    def test_recommendations_success(self, routing_handler, mock_http_handler_post):
        """Test successful recommendations retrieval."""
        handler = mock_http_handler_post(
            {
                "primary_domain": "coding",
                "secondary_domains": ["analysis"],
                "required_traits": ["analytical"],
                "limit": 3,
            }
        )

        mock_selector = MagicMock()
        mock_selector.get_recommendations.return_value = [
            {"agent": "claude", "score": 0.9},
            {"agent": "gpt4", "score": 0.85},
        ]

        with (
            patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True),
            patch("aragora.server.handlers.routing.AgentSelector", return_value=mock_selector),
            patch("aragora.server.handlers.routing.TaskRequirements"),
        ):
            result = routing_handler.handle_post("/api/routing/recommendations", {}, handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["primary_domain"] == "coding"
        assert len(body["recommendations"]) == 2


# =============================================================================
# Auto-Route Endpoint Tests (POST)
# =============================================================================


class TestAutoRouteEndpoint:
    """Test /api/routing/auto-route endpoint."""

    def test_auto_route_routing_unavailable_returns_503(
        self, routing_handler, mock_http_handler_post
    ):
        """Test error when routing module unavailable."""
        handler = mock_http_handler_post({"task": "Build a REST API"})

        with patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", False):
            result = routing_handler.handle_post("/api/routing/auto-route", {}, handler)

        assert result is not None
        assert result.status_code == 503

    def test_auto_route_missing_task_returns_400(self, routing_handler, mock_http_handler_post):
        """Test error when task field missing."""
        handler = mock_http_handler_post({})

        with (
            patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True),
            patch("aragora.server.handlers.routing.AgentSelector"),
            patch("aragora.server.handlers.routing.DomainDetector"),
        ):
            result = routing_handler.handle_post("/api/routing/auto-route", {}, handler)

        assert result is not None
        assert result.status_code == 400

    def test_auto_route_success(self, routing_handler, mock_http_handler_post):
        """Test successful auto-routing."""
        handler = mock_http_handler_post(
            {"task": "Build a REST API with authentication", "exclude": ["gpt4"]}
        )

        mock_agent = MagicMock()
        mock_agent.name = "claude"
        mock_agent.expertise = {"coding": 0.9}

        mock_team = MagicMock()
        mock_team.task_id = "task_123"
        mock_team.agents = [mock_agent]
        mock_team.roles = {"claude": "lead"}
        mock_team.expected_quality = 0.85
        mock_team.diversity_score = 0.7
        mock_team.rationale = "Best for API development"

        mock_selector = MagicMock()
        mock_selector.auto_route.return_value = mock_team

        with (
            patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True),
            patch("aragora.server.handlers.routing.AgentSelector") as MockSelector,
            patch("aragora.server.handlers.routing.DomainDetector"),
        ):
            MockSelector.create_with_defaults.return_value = mock_selector
            result = routing_handler.handle_post("/api/routing/auto-route", {}, handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["task_id"] == "task_123"
        assert body["team"]["agents"] == ["claude"]
        assert body["rationale"] == "Best for API development"


# =============================================================================
# Detect Domain Endpoint Tests (POST)
# =============================================================================


class TestDetectDomainEndpoint:
    """Test /api/routing/detect-domain endpoint."""

    def test_detect_domain_routing_unavailable_returns_503(
        self, routing_handler, mock_http_handler_post
    ):
        """Test error when routing module unavailable."""
        handler = mock_http_handler_post({"task": "Build a REST API"})

        with patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", False):
            result = routing_handler.handle_post("/api/routing/detect-domain", {}, handler)

        assert result is not None
        assert result.status_code == 503

    def test_detect_domain_missing_task_returns_400(self, routing_handler, mock_http_handler_post):
        """Test error when task field missing."""
        handler = mock_http_handler_post({})

        with (
            patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True),
            patch("aragora.server.handlers.routing.DomainDetector"),
        ):
            result = routing_handler.handle_post("/api/routing/detect-domain", {}, handler)

        assert result is not None
        assert result.status_code == 400

    def test_detect_domain_success(self, routing_handler, mock_http_handler_post):
        """Test successful domain detection."""
        handler = mock_http_handler_post(
            {"task": "Build a machine learning model for image classification", "top_n": 3}
        )

        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            ("machine_learning", 0.9),
            ("coding", 0.7),
            ("data_science", 0.6),
        ]

        with (
            patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True),
            patch("aragora.server.handlers.routing.DomainDetector", return_value=mock_detector),
        ):
            result = routing_handler.handle_post("/api/routing/detect-domain", {}, handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["primary_domain"] == "machine_learning"
        assert len(body["domains"]) == 3
        assert body["domains"][0]["domain"] == "machine_learning"
        assert body["domains"][0]["confidence"] == 0.9

    def test_detect_domain_truncates_long_task(self, routing_handler, mock_http_handler_post):
        """Test task text is truncated in response."""
        long_task = "x" * 300
        handler = mock_http_handler_post({"task": long_task})

        mock_detector = MagicMock()
        mock_detector.detect.return_value = [("general", 0.5)]

        with (
            patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True),
            patch("aragora.server.handlers.routing.DomainDetector", return_value=mock_detector),
        ):
            result = routing_handler.handle_post("/api/routing/detect-domain", {}, handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["task"].endswith("...")
        assert len(body["task"]) == 203  # 200 chars + "..."


# =============================================================================
# handle_post Routing Tests
# =============================================================================


class TestHandlePostRouting:
    """Test POST request routing."""

    def test_handle_post_unknown_route_returns_none(self, routing_handler, mock_http_handler_post):
        """Test unknown POST route returns None."""
        handler = mock_http_handler_post({"task": "test"})

        result = routing_handler.handle_post("/api/routing/unknown", {}, handler)

        assert result is None
