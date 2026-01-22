"""
Tests for Routing Handler.

Tests the agent routing and team selection API endpoints including:
- Best team combinations
- Agent recommendations
- Auto-routing with domain detection
- Domain detection
- Domain leaderboard
- Rate limiting
"""

import pytest
import json
from io import BytesIO
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any

from aragora.server.handlers.routing import RoutingHandler


@dataclass
class MockAgent:
    """Mock agent for testing."""

    name: str
    expertise: Dict[str, float]


@dataclass
class MockTeam:
    """Mock team composition for testing."""

    task_id: str
    agents: List[MockAgent]
    roles: Dict[str, str]
    expected_quality: float
    diversity_score: float
    rationale: str


class TestRoutingHandlerRouting:
    """Test RoutingHandler routing logic."""

    def test_routes_defined(self):
        """Handler should define expected routes."""
        assert "/api/routing/best-teams" in RoutingHandler.ROUTES
        assert "/api/routing/recommendations" in RoutingHandler.ROUTES
        assert "/api/routing/auto-route" in RoutingHandler.ROUTES
        assert "/api/routing/detect-domain" in RoutingHandler.ROUTES
        assert "/api/routing/domain-leaderboard" in RoutingHandler.ROUTES

    def test_can_handle_routes(self):
        """Should handle defined routes."""
        handler = RoutingHandler({})
        for route in RoutingHandler.ROUTES:
            assert handler.can_handle(route) is True

    def test_cannot_handle_unknown_routes(self):
        """Should not handle unknown routes."""
        handler = RoutingHandler({})
        assert handler.can_handle("/api/unknown") is False
        assert handler.can_handle("/api/routing/unknown") is False


class TestBestTeamCombinations:
    """Test best team combinations endpoint."""

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", False)
    def test_routing_unavailable(self):
        """Should return 503 when routing module not available."""
        handler = RoutingHandler({})
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/routing/best-teams", {}, mock_http)

        assert result is not None
        assert result.status_code == 503

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.AgentSelector")
    def test_best_teams_success(self, mock_selector_class):
        """Should return team combinations when successful."""
        mock_selector = MagicMock()
        mock_selector.get_best_team_combinations.return_value = [
            {"agents": ["claude", "gpt-4"], "success_rate": 0.85, "debates": 10},
        ]
        mock_selector_class.return_value = mock_selector

        handler = RoutingHandler({})
        handler.get_elo_system = MagicMock(return_value=MagicMock())
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle(
            "/api/routing/best-teams", {"min_debates": ["5"], "limit": ["10"]}, mock_http
        )

        assert result is not None
        assert result.status_code == 200
        mock_selector.get_best_team_combinations.assert_called_once()


class TestRecommendations:
    """Test agent recommendations endpoint."""

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", False)
    def test_routing_unavailable(self):
        """Should return 503 when routing module not available."""
        handler = RoutingHandler({})
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)
        handler.read_json_body = MagicMock(return_value={"primary_domain": "code"})

        result = handler.handle_post("/api/routing/recommendations", {}, mock_http)

        assert result is not None
        assert result.status_code == 503

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.AgentSelector")
    @patch("aragora.server.handlers.routing.TaskRequirements")
    def test_recommendations_success(self, mock_requirements, mock_selector_class):
        """Should return recommendations when successful."""
        mock_selector = MagicMock()
        mock_selector.get_recommendations.return_value = [
            {"agent": "claude", "score": 0.95, "domain_match": 0.9},
        ]
        mock_selector_class.return_value = mock_selector

        handler = RoutingHandler({})
        handler.get_elo_system = MagicMock(return_value=MagicMock())
        handler.read_json_body = MagicMock(
            return_value={"primary_domain": "code", "task": "Write a function"}
        )
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/routing/recommendations", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

    def test_invalid_json_body(self):
        """Should return 400 for invalid JSON body."""
        handler = RoutingHandler({})
        handler.read_json_body = MagicMock(return_value=None)
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/routing/recommendations", {}, mock_http)

        assert result is not None
        assert result.status_code == 400


class TestAutoRoute:
    """Test auto-route endpoint."""

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", False)
    def test_routing_unavailable(self):
        """Should return 503 when routing module not available."""
        handler = RoutingHandler({})
        handler.read_json_body = MagicMock(return_value={"task": "Test task"})
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/routing/auto-route", {}, mock_http)

        assert result is not None
        assert result.status_code == 503

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.AgentSelector")
    @patch("aragora.server.handlers.routing.DomainDetector")
    def test_auto_route_success(self, mock_detector, mock_selector_class):
        """Should auto-route task successfully."""
        mock_agent = MockAgent(name="claude", expertise={"code": 0.9})
        mock_team = MockTeam(
            task_id="task-123",
            agents=[mock_agent],
            roles={"claude": "lead"},
            expected_quality=0.85,
            diversity_score=0.7,
            rationale="Selected for code expertise",
        )

        mock_selector = MagicMock()
        mock_selector.auto_route.return_value = mock_team
        mock_selector_class.create_with_defaults.return_value = mock_selector

        handler = RoutingHandler({})
        handler.get_elo_system = MagicMock(return_value=MagicMock())
        handler.read_json_body = MagicMock(return_value={"task": "Write a Python function"})
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/routing/auto-route", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

    def test_missing_task_field(self):
        """Should return 400 when task field is missing."""
        handler = RoutingHandler({})
        handler.read_json_body = MagicMock(return_value={"exclude": []})
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        with patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True):
            with patch("aragora.server.handlers.routing.AgentSelector"):
                with patch("aragora.server.handlers.routing.DomainDetector"):
                    result = handler.handle_post("/api/routing/auto-route", {}, mock_http)

        assert result is not None
        assert result.status_code == 400

    def test_invalid_json_body(self):
        """Should return 400 for invalid JSON body."""
        handler = RoutingHandler({})
        handler.read_json_body = MagicMock(return_value=None)
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/routing/auto-route", {}, mock_http)

        assert result is not None
        assert result.status_code == 400


class TestDetectDomain:
    """Test domain detection endpoint."""

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", False)
    def test_detection_unavailable(self):
        """Should return 503 when domain detection not available."""
        handler = RoutingHandler({})
        handler.read_json_body = MagicMock(return_value={"task": "Test task"})
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/routing/detect-domain", {}, mock_http)

        assert result is not None
        assert result.status_code == 503

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.DomainDetector")
    def test_detect_domain_success(self, mock_detector_class):
        """Should detect domain successfully."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [("code", 0.9), ("math", 0.7), ("general", 0.5)]
        mock_detector_class.return_value = mock_detector

        handler = RoutingHandler({})
        handler.read_json_body = MagicMock(return_value={"task": "Write a Python function"})
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/routing/detect-domain", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

    def test_missing_task_field(self):
        """Should return 400 when task field is missing."""
        handler = RoutingHandler({})
        handler.read_json_body = MagicMock(return_value={"top_n": 3})
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        with patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True):
            with patch("aragora.server.handlers.routing.DomainDetector"):
                result = handler.handle_post("/api/routing/detect-domain", {}, mock_http)

        assert result is not None
        assert result.status_code == 400


class TestDomainLeaderboard:
    """Test domain leaderboard endpoint."""

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", False)
    def test_routing_unavailable(self):
        """Should return 503 when routing module not available."""
        handler = RoutingHandler({})
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/routing/domain-leaderboard", {"domain": ["code"]}, mock_http)

        assert result is not None
        assert result.status_code == 503

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.AgentSelector")
    def test_leaderboard_success(self, mock_selector_class):
        """Should return domain leaderboard when successful."""
        mock_selector = MagicMock()
        mock_selector.get_domain_leaderboard.return_value = [
            {"agent": "claude", "score": 0.95, "debates_in_domain": 50},
            {"agent": "gpt-4", "score": 0.92, "debates_in_domain": 45},
        ]
        mock_selector_class.create_with_defaults.return_value = mock_selector

        handler = RoutingHandler({})
        handler.get_elo_system = MagicMock(return_value=MagicMock())
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle(
            "/api/routing/domain-leaderboard",
            {"domain": ["code"], "limit": ["10"]},
            mock_http,
        )

        assert result is not None
        assert result.status_code == 200


class TestRateLimiting:
    """Test rate limiting behavior."""

    @patch("aragora.server.handlers.routing._routing_limiter")
    def test_rate_limit_exceeded(self, mock_limiter):
        """Should return 429 when rate limit exceeded."""
        mock_limiter.is_allowed.return_value = False

        handler = RoutingHandler({})
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/routing/best-teams", {}, mock_http)

        assert result is not None
        assert result.status_code == 429


class TestParameterValidation:
    """Test query parameter validation."""

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.AgentSelector")
    def test_min_debates_clamped(self, mock_selector_class):
        """Should clamp min_debates parameter."""
        mock_selector = MagicMock()
        mock_selector.get_best_team_combinations.return_value = []
        mock_selector_class.return_value = mock_selector

        handler = RoutingHandler({})
        handler.get_elo_system = MagicMock(return_value=MagicMock())
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        # min_debates should be clamped to max of 20
        result = handler.handle("/api/routing/best-teams", {"min_debates": ["100"]}, mock_http)

        assert result is not None
        assert result.status_code == 200

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.AgentSelector")
    def test_limit_clamped(self, mock_selector_class):
        """Should clamp limit parameter."""
        mock_selector = MagicMock()
        mock_selector.get_best_team_combinations.return_value = []
        mock_selector_class.return_value = mock_selector

        handler = RoutingHandler({})
        handler.get_elo_system = MagicMock(return_value=MagicMock())
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        # limit should be clamped to max of 50
        result = handler.handle("/api/routing/best-teams", {"limit": ["1000"]}, mock_http)

        assert result is not None
        assert result.status_code == 200
