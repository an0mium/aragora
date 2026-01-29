"""
Tests for Routing Handler.

Tests the agent routing and team selection API endpoints including:
- Best team combinations
- Agent recommendations
- Auto-routing with domain detection
- Domain detection
- Domain leaderboard
- Rate limiting
- Response body validation
- Parameter defaults and validation
- Version prefix handling
"""

import pytest
import json
from io import BytesIO
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any

from aragora.server.handlers.routing import RoutingHandler, _routing_limiter


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
        # ROUTES uses normalized paths without version prefix
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
        assert handler.can_handle("/api/v1/unknown") is False
        assert handler.can_handle("/api/v1/routing/unknown") is False


class TestBestTeamCombinations:
    """Test best team combinations endpoint."""

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", False)
    def test_routing_unavailable(self):
        """Should return 503 when routing module not available."""
        handler = RoutingHandler({})
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle("/api/v1/routing/best-teams", {}, mock_http)

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
            "/api/v1/routing/best-teams", {"min_debates": ["5"], "limit": ["10"]}, mock_http
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

        result = handler.handle_post("/api/v1/routing/recommendations", {}, mock_http)

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

        result = handler.handle_post("/api/v1/routing/recommendations", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

    def test_invalid_json_body(self):
        """Should return 400 for invalid JSON body."""
        handler = RoutingHandler({})
        handler.read_json_body = MagicMock(return_value=None)
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/v1/routing/recommendations", {}, mock_http)

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

        result = handler.handle_post("/api/v1/routing/auto-route", {}, mock_http)

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

        result = handler.handle_post("/api/v1/routing/auto-route", {}, mock_http)

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
                    result = handler.handle_post("/api/v1/routing/auto-route", {}, mock_http)

        assert result is not None
        assert result.status_code == 400

    def test_invalid_json_body(self):
        """Should return 400 for invalid JSON body."""
        handler = RoutingHandler({})
        handler.read_json_body = MagicMock(return_value=None)
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/v1/routing/auto-route", {}, mock_http)

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

        result = handler.handle_post("/api/v1/routing/detect-domain", {}, mock_http)

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

        result = handler.handle_post("/api/v1/routing/detect-domain", {}, mock_http)

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
                result = handler.handle_post("/api/v1/routing/detect-domain", {}, mock_http)

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

        result = handler.handle(
            "/api/v1/routing/domain-leaderboard", {"domain": ["code"]}, mock_http
        )

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
            "/api/v1/routing/domain-leaderboard",
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

        result = handler.handle("/api/v1/routing/best-teams", {}, mock_http)

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
        result = handler.handle("/api/v1/routing/best-teams", {"min_debates": ["100"]}, mock_http)

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
        result = handler.handle("/api/v1/routing/best-teams", {"limit": ["1000"]}, mock_http)

        assert result is not None
        assert result.status_code == 200


class TestResponseBodyValidation:
    """Test response body JSON structure."""

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.AgentSelector")
    def test_best_teams_response_structure(self, mock_selector_class):
        """Best teams response should have expected JSON fields."""
        mock_selector = MagicMock()
        mock_selector.get_best_team_combinations.return_value = [
            {"agents": ["claude", "gpt-4"], "success_rate": 0.85, "debates": 10},
        ]
        mock_selector_class.return_value = mock_selector

        handler = RoutingHandler({})
        handler.get_elo_system = MagicMock(return_value=MagicMock())
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)
        _routing_limiter._buckets.clear()

        result = handler.handle("/api/v1/routing/best-teams", {}, mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "min_debates" in body
        assert "combinations" in body
        assert "count" in body
        assert body["count"] == 1

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.AgentSelector")
    @patch("aragora.server.handlers.routing.TaskRequirements")
    def test_recommendations_response_structure(self, mock_requirements, mock_selector_class):
        """Recommendations response should have expected JSON fields."""
        mock_selector = MagicMock()
        mock_selector.get_recommendations.return_value = [
            {"agent": "claude", "score": 0.95, "domain_match": 0.9},
        ]
        mock_selector_class.return_value = mock_selector

        handler = RoutingHandler({})
        handler.get_elo_system = MagicMock(return_value=MagicMock())
        handler.read_json_body = MagicMock(
            return_value={"primary_domain": "code", "task_id": "test-123"}
        )
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/v1/routing/recommendations", {}, mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "task_id" in body
        assert "primary_domain" in body
        assert "recommendations" in body
        assert "count" in body
        assert body["task_id"] == "test-123"
        assert body["primary_domain"] == "code"

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.AgentSelector")
    @patch("aragora.server.handlers.routing.DomainDetector")
    def test_auto_route_response_structure(self, mock_detector, mock_selector_class):
        """Auto-route response should have expected JSON fields."""
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

        result = handler.handle_post("/api/v1/routing/auto-route", {}, mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "task_id" in body
        assert "detected_domain" in body
        assert "team" in body
        assert "rationale" in body
        assert "agents" in body["team"]
        assert "roles" in body["team"]
        assert "expected_quality" in body["team"]
        assert "diversity_score" in body["team"]

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.DomainDetector")
    def test_detect_domain_response_structure(self, mock_detector_class):
        """Detect domain response should have expected JSON fields."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [("code", 0.9), ("math", 0.7)]
        mock_detector_class.return_value = mock_detector

        handler = RoutingHandler({})
        handler.read_json_body = MagicMock(return_value={"task": "Write a Python function"})
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/v1/routing/detect-domain", {}, mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "task" in body
        assert "domains" in body
        assert "primary_domain" in body
        assert body["primary_domain"] == "code"
        assert len(body["domains"]) == 2
        assert body["domains"][0]["domain"] == "code"
        assert body["domains"][0]["confidence"] == 0.9

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.AgentSelector")
    def test_domain_leaderboard_response_structure(self, mock_selector_class):
        """Domain leaderboard response should have expected JSON fields."""
        mock_selector = MagicMock()
        mock_selector.get_domain_leaderboard.return_value = [
            {"agent": "claude", "score": 0.95},
        ]
        mock_selector_class.create_with_defaults.return_value = mock_selector

        handler = RoutingHandler({})
        handler.get_elo_system = MagicMock(return_value=MagicMock())
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)
        _routing_limiter._buckets.clear()

        result = handler.handle(
            "/api/v1/routing/domain-leaderboard",
            {"domain": ["code"]},
            mock_http,
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "domain" in body
        assert "leaderboard" in body
        assert "count" in body
        assert body["domain"] == "code"


class TestEmptyTaskValidation:
    """Test empty task string validation."""

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.AgentSelector")
    @patch("aragora.server.handlers.routing.DomainDetector")
    def test_auto_route_empty_task(self, mock_detector, mock_selector_class):
        """Should return 400 when task is empty string."""
        handler = RoutingHandler({})
        handler.read_json_body = MagicMock(return_value={"task": ""})
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/v1/routing/auto-route", {}, mock_http)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.DomainDetector")
    def test_detect_domain_empty_task(self, mock_detector_class):
        """Should return 400 when task is empty string."""
        handler = RoutingHandler({})
        handler.read_json_body = MagicMock(return_value={"task": ""})
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/v1/routing/detect-domain", {}, mock_http)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body


class TestDetectDomainInvalidBody:
    """Test detect-domain endpoint with invalid JSON body."""

    def test_detect_domain_invalid_json_body(self):
        """Should return 400 for invalid JSON body."""
        handler = RoutingHandler({})
        handler.read_json_body = MagicMock(return_value=None)
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/v1/routing/detect-domain", {}, mock_http)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body


class TestTaskTextTruncation:
    """Test task text truncation in responses."""

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.DomainDetector")
    def test_long_task_text_truncated(self, mock_detector_class):
        """Should truncate long task text in response."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [("general", 0.5)]
        mock_detector_class.return_value = mock_detector

        handler = RoutingHandler({})
        long_task = "A" * 300  # Longer than 200 characters
        handler.read_json_body = MagicMock(return_value={"task": long_task})
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/v1/routing/detect-domain", {}, mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert len(body["task"]) == 203  # 200 chars + "..."
        assert body["task"].endswith("...")

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.DomainDetector")
    def test_short_task_text_not_truncated(self, mock_detector_class):
        """Should not truncate short task text."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [("general", 0.5)]
        mock_detector_class.return_value = mock_detector

        handler = RoutingHandler({})
        short_task = "Write a function"
        handler.read_json_body = MagicMock(return_value={"task": short_task})
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/v1/routing/detect-domain", {}, mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["task"] == short_task
        assert not body["task"].endswith("...")


class TestVersionPrefixHandling:
    """Test version prefix stripping in routes."""

    def test_can_handle_v1_routes(self):
        """Should handle routes with v1 prefix."""
        handler = RoutingHandler({})
        assert handler.can_handle("/api/v1/routing/best-teams") is True
        assert handler.can_handle("/api/v1/routing/recommendations") is True
        assert handler.can_handle("/api/v1/routing/auto-route") is True
        assert handler.can_handle("/api/v1/routing/detect-domain") is True
        assert handler.can_handle("/api/v1/routing/domain-leaderboard") is True

    def test_can_handle_v2_routes(self):
        """Should handle routes with v2 prefix."""
        handler = RoutingHandler({})
        assert handler.can_handle("/api/v2/routing/best-teams") is True
        assert handler.can_handle("/api/v2/routing/recommendations") is True

    def test_can_handle_routes_without_version(self):
        """Should handle routes without version prefix."""
        handler = RoutingHandler({})
        assert handler.can_handle("/api/routing/best-teams") is True
        assert handler.can_handle("/api/routing/recommendations") is True


class TestDefaultParameterValues:
    """Test default parameter values."""

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.AgentSelector")
    def test_best_teams_default_min_debates(self, mock_selector_class):
        """Should use default min_debates of 3."""
        mock_selector = MagicMock()
        mock_selector.get_best_team_combinations.return_value = []
        mock_selector_class.return_value = mock_selector

        handler = RoutingHandler({})
        handler.get_elo_system = MagicMock(return_value=MagicMock())
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)
        _routing_limiter._buckets.clear()

        result = handler.handle("/api/v1/routing/best-teams", {}, mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["min_debates"] == 3

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.AgentSelector")
    def test_domain_leaderboard_default_domain(self, mock_selector_class):
        """Should use default domain of 'general'."""
        mock_selector = MagicMock()
        mock_selector.get_domain_leaderboard.return_value = []
        mock_selector_class.create_with_defaults.return_value = mock_selector

        handler = RoutingHandler({})
        handler.get_elo_system = MagicMock(return_value=MagicMock())
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)
        _routing_limiter._buckets.clear()

        result = handler.handle("/api/v1/routing/domain-leaderboard", {}, mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["domain"] == "general"

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.AgentSelector")
    @patch("aragora.server.handlers.routing.TaskRequirements")
    def test_recommendations_default_task_id(self, mock_requirements, mock_selector_class):
        """Should use default task_id of 'ad-hoc'."""
        mock_selector = MagicMock()
        mock_selector.get_recommendations.return_value = []
        mock_selector_class.return_value = mock_selector

        handler = RoutingHandler({})
        handler.get_elo_system = MagicMock(return_value=MagicMock())
        handler.read_json_body = MagicMock(return_value={"primary_domain": "code"})
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/v1/routing/recommendations", {}, mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["task_id"] == "ad-hoc"


class TestRecommendationsWithAllParameters:
    """Test recommendations endpoint with all parameters."""

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.AgentSelector")
    @patch("aragora.server.handlers.routing.TaskRequirements")
    def test_recommendations_with_secondary_domains(self, mock_requirements, mock_selector_class):
        """Should accept secondary_domains parameter."""
        mock_selector = MagicMock()
        mock_selector.get_recommendations.return_value = []
        mock_selector_class.return_value = mock_selector

        handler = RoutingHandler({})
        handler.get_elo_system = MagicMock(return_value=MagicMock())
        handler.read_json_body = MagicMock(
            return_value={
                "primary_domain": "code",
                "secondary_domains": ["math", "logic"],
                "task": "Implement algorithm",
            }
        )
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/v1/routing/recommendations", {}, mock_http)

        assert result.status_code == 200
        mock_requirements.assert_called_once()
        call_kwargs = mock_requirements.call_args
        assert call_kwargs.kwargs.get("secondary_domains") == ["math", "logic"]

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.AgentSelector")
    @patch("aragora.server.handlers.routing.TaskRequirements")
    def test_recommendations_with_required_traits(self, mock_requirements, mock_selector_class):
        """Should accept required_traits parameter."""
        mock_selector = MagicMock()
        mock_selector.get_recommendations.return_value = []
        mock_selector_class.return_value = mock_selector

        handler = RoutingHandler({})
        handler.get_elo_system = MagicMock(return_value=MagicMock())
        handler.read_json_body = MagicMock(
            return_value={
                "primary_domain": "code",
                "required_traits": ["analytical", "creative"],
            }
        )
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/v1/routing/recommendations", {}, mock_http)

        assert result.status_code == 200
        mock_requirements.assert_called_once()
        call_kwargs = mock_requirements.call_args
        assert call_kwargs.kwargs.get("required_traits") == ["analytical", "creative"]

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.AgentSelector")
    @patch("aragora.server.handlers.routing.TaskRequirements")
    def test_recommendations_limit_clamped_to_20(self, mock_requirements, mock_selector_class):
        """Should clamp limit to maximum of 20."""
        mock_selector = MagicMock()
        mock_selector.get_recommendations.return_value = []
        mock_selector_class.return_value = mock_selector

        handler = RoutingHandler({})
        handler.get_elo_system = MagicMock(return_value=MagicMock())
        handler.read_json_body = MagicMock(
            return_value={"primary_domain": "code", "limit": 100}
        )
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/v1/routing/recommendations", {}, mock_http)

        assert result.status_code == 200
        mock_selector.get_recommendations.assert_called_once()
        call_kwargs = mock_selector.get_recommendations.call_args
        assert call_kwargs.kwargs.get("limit") == 20


class TestAutoRouteWithAllParameters:
    """Test auto-route endpoint with all parameters."""

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.AgentSelector")
    @patch("aragora.server.handlers.routing.DomainDetector")
    def test_auto_route_with_exclude(self, mock_detector, mock_selector_class):
        """Should accept exclude parameter."""
        mock_agent = MockAgent(name="gpt-4", expertise={"code": 0.85})
        mock_team = MockTeam(
            task_id="task-456",
            agents=[mock_agent],
            roles={"gpt-4": "lead"},
            expected_quality=0.8,
            diversity_score=0.6,
            rationale="Selected after excluding claude",
        )

        mock_selector = MagicMock()
        mock_selector.auto_route.return_value = mock_team
        mock_selector_class.create_with_defaults.return_value = mock_selector

        handler = RoutingHandler({})
        handler.get_elo_system = MagicMock(return_value=MagicMock())
        handler.read_json_body = MagicMock(
            return_value={"task": "Write code", "exclude": ["claude"]}
        )
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/v1/routing/auto-route", {}, mock_http)

        assert result.status_code == 200
        mock_selector.auto_route.assert_called_once()
        call_kwargs = mock_selector.auto_route.call_args
        assert call_kwargs.kwargs.get("exclude") == ["claude"]

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.AgentSelector")
    @patch("aragora.server.handlers.routing.DomainDetector")
    def test_auto_route_with_task_id(self, mock_detector, mock_selector_class):
        """Should accept task_id parameter."""
        mock_agent = MockAgent(name="claude", expertise={"code": 0.9})
        mock_team = MockTeam(
            task_id="custom-task-id",
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
        handler.read_json_body = MagicMock(
            return_value={"task": "Write code", "task_id": "custom-task-id"}
        )
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/v1/routing/auto-route", {}, mock_http)

        assert result.status_code == 200
        mock_selector.auto_route.assert_called_once()
        call_kwargs = mock_selector.auto_route.call_args
        assert call_kwargs.kwargs.get("task_id") == "custom-task-id"


class TestDetectDomainTopN:
    """Test detect-domain top_n parameter."""

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.DomainDetector")
    def test_detect_domain_custom_top_n(self, mock_detector_class):
        """Should accept custom top_n parameter."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            ("code", 0.9),
            ("math", 0.7),
            ("logic", 0.6),
            ("writing", 0.4),
            ("general", 0.3),
        ]
        mock_detector_class.return_value = mock_detector

        handler = RoutingHandler({})
        handler.read_json_body = MagicMock(return_value={"task": "Test", "top_n": 5})
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/v1/routing/detect-domain", {}, mock_http)

        assert result.status_code == 200
        mock_detector.detect.assert_called_once()
        call_kwargs = mock_detector.detect.call_args
        assert call_kwargs.kwargs.get("top_n") == 5

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.DomainDetector")
    def test_detect_domain_top_n_clamped_to_10(self, mock_detector_class):
        """Should clamp top_n to maximum of 10."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [("code", 0.9)]
        mock_detector_class.return_value = mock_detector

        handler = RoutingHandler({})
        handler.read_json_body = MagicMock(return_value={"task": "Test", "top_n": 100})
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/v1/routing/detect-domain", {}, mock_http)

        assert result.status_code == 200
        mock_detector.detect.assert_called_once()
        call_kwargs = mock_detector.detect.call_args
        assert call_kwargs.kwargs.get("top_n") == 10


class TestHandleReturnsNone:
    """Test handle methods returning None for non-matching paths."""

    def test_handle_returns_none_for_unknown_get_path(self):
        """Handle should return None for unknown GET paths."""
        handler = RoutingHandler({})
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)
        _routing_limiter._buckets.clear()

        result = handler.handle("/api/v1/unknown", {}, mock_http)

        assert result is None

    def test_handle_post_returns_none_for_unknown_path(self):
        """Handle_post should return None for unknown POST paths."""
        handler = RoutingHandler({})
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/v1/unknown", {}, mock_http)

        assert result is None

    def test_handle_returns_none_for_get_on_post_route(self):
        """Handle (GET) should return None for POST-only routes."""
        handler = RoutingHandler({})
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)
        _routing_limiter._buckets.clear()

        # recommendations is POST-only
        result = handler.handle("/api/v1/routing/recommendations", {}, mock_http)

        assert result is None


class TestEmptyResults:
    """Test handling of empty results."""

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.AgentSelector")
    def test_best_teams_empty_combinations(self, mock_selector_class):
        """Should return empty list when no combinations found."""
        mock_selector = MagicMock()
        mock_selector.get_best_team_combinations.return_value = []
        mock_selector_class.return_value = mock_selector

        handler = RoutingHandler({})
        handler.get_elo_system = MagicMock(return_value=MagicMock())
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)
        _routing_limiter._buckets.clear()

        result = handler.handle("/api/v1/routing/best-teams", {}, mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["combinations"] == []
        assert body["count"] == 0

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.DomainDetector")
    def test_detect_domain_empty_results(self, mock_detector_class):
        """Should handle empty domain detection results."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = []
        mock_detector_class.return_value = mock_detector

        handler = RoutingHandler({})
        handler.read_json_body = MagicMock(return_value={"task": "Unknown task"})
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/v1/routing/detect-domain", {}, mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["domains"] == []
        assert body["primary_domain"] == "general"


class TestWithPersonaManager:
    """Test handler with persona_manager in context."""

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.AgentSelector")
    def test_best_teams_with_persona_manager(self, mock_selector_class):
        """Should pass persona_manager to AgentSelector."""
        mock_selector = MagicMock()
        mock_selector.get_best_team_combinations.return_value = []
        mock_selector_class.return_value = mock_selector

        mock_persona_manager = MagicMock()
        handler = RoutingHandler({"persona_manager": mock_persona_manager})
        handler.get_elo_system = MagicMock(return_value=MagicMock())
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)
        _routing_limiter._buckets.clear()

        result = handler.handle("/api/v1/routing/best-teams", {}, mock_http)

        assert result.status_code == 200
        mock_selector_class.assert_called_once()
        call_kwargs = mock_selector_class.call_args
        assert call_kwargs.kwargs.get("persona_manager") == mock_persona_manager

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.AgentSelector")
    @patch("aragora.server.handlers.routing.TaskRequirements")
    def test_recommendations_with_persona_manager(self, mock_requirements, mock_selector_class):
        """Should pass persona_manager to AgentSelector for recommendations."""
        mock_selector = MagicMock()
        mock_selector.get_recommendations.return_value = []
        mock_selector_class.return_value = mock_selector

        mock_persona_manager = MagicMock()
        handler = RoutingHandler({"persona_manager": mock_persona_manager})
        handler.get_elo_system = MagicMock(return_value=MagicMock())
        handler.read_json_body = MagicMock(return_value={"primary_domain": "code"})
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/v1/routing/recommendations", {}, mock_http)

        assert result.status_code == 200
        mock_selector_class.assert_called_once()
        call_kwargs = mock_selector_class.call_args
        assert call_kwargs.kwargs.get("persona_manager") == mock_persona_manager


class TestAutoRouteWithEmptyAgents:
    """Test auto-route handling when team has empty agents."""

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.AgentSelector")
    @patch("aragora.server.handlers.routing.DomainDetector")
    def test_auto_route_empty_agents(self, mock_detector, mock_selector_class):
        """Should handle team with empty agents list."""
        mock_team = MockTeam(
            task_id="task-123",
            agents=[],
            roles={},
            expected_quality=0.0,
            diversity_score=0.0,
            rationale="No suitable agents found",
        )

        mock_selector = MagicMock()
        mock_selector.auto_route.return_value = mock_team
        mock_selector_class.create_with_defaults.return_value = mock_selector

        handler = RoutingHandler({})
        handler.get_elo_system = MagicMock(return_value=MagicMock())
        handler.read_json_body = MagicMock(return_value={"task": "Write code"})
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        result = handler.handle_post("/api/v1/routing/auto-route", {}, mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["team"]["agents"] == []
        assert body["detected_domain"] == {}


class TestRateLimiterClear:
    """Test rate limiter clearing between tests."""

    @patch("aragora.server.handlers.routing.ROUTING_AVAILABLE", True)
    @patch("aragora.server.handlers.routing.AgentSelector")
    def test_rate_limiter_allows_after_clear(self, mock_selector_class):
        """Rate limiter should allow requests after clearing buckets."""
        mock_selector = MagicMock()
        mock_selector.get_best_team_combinations.return_value = []
        mock_selector_class.return_value = mock_selector

        handler = RoutingHandler({})
        handler.get_elo_system = MagicMock(return_value=MagicMock())
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)

        # Clear the rate limiter
        _routing_limiter._buckets.clear()

        result = handler.handle("/api/v1/routing/best-teams", {}, mock_http)

        assert result.status_code == 200
