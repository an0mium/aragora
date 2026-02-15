"""
Tests for aragora.server.handlers.routing - Routing HTTP Handlers.

Tests cover:
- RoutingHandler: instantiation, ROUTES, can_handle
- GET /api/routing/best-teams: success, routing unavailable
- GET /api/routing/domain-leaderboard: success, routing unavailable
- POST /api/routing/recommendations: success, body validation, routing unavailable
- POST /api/routing/auto-route: success, missing task, routing unavailable
- POST /api/routing/detect-domain: success, missing task, routing unavailable
- handle routing: rate limiting, returns None for unmatched paths
- handle_post routing: returns None for unmatched paths
- Version prefix stripping (v1 paths)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.routing import RoutingHandler
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helpers
# ===========================================================================


def _parse_body(result: HandlerResult) -> dict[str, Any]:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body)


def _make_mock_handler(
    method: str = "GET",
    body: bytes = b"",
    content_type: str = "application/json",
) -> MagicMock:
    """Create a mock HTTP handler object."""
    handler = MagicMock()
    handler.command = method
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": content_type,
        "Host": "localhost:8080",
    }
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = body
    return handler


# ===========================================================================
# Mock Routing Objects
# ===========================================================================


class MockAgent:
    """Mock agent for team composition."""

    def __init__(self, name: str = "claude"):
        self.name = name
        self.expertise = {"general": 0.9}


class MockTeam:
    """Mock team result from auto_route."""

    def __init__(self):
        self.task_id = "task-001"
        self.agents = [MockAgent("claude"), MockAgent("gpt-4")]
        self.roles = {"proposer": "claude", "critic": "gpt-4"}
        self.expected_quality = 0.85
        self.diversity_score = 0.7
        self.rationale = "Best team for general tasks"


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the rate limiter between tests."""
    from aragora.server.handlers.routing import _routing_limiter

    _routing_limiter._buckets.clear()


@pytest.fixture
def handler():
    """Create a RoutingHandler."""
    return RoutingHandler(ctx={})


# ===========================================================================
# Test Instantiation and Basics
# ===========================================================================


class TestRoutingHandlerBasics:
    """Basic instantiation and attribute tests."""

    def test_instantiation(self, handler):
        assert handler is not None
        assert isinstance(handler, RoutingHandler)

    def test_routes(self, handler):
        assert "/api/routing/best-teams" in handler.ROUTES
        assert "/api/routing/recommendations" in handler.ROUTES
        assert "/api/routing/auto-route" in handler.ROUTES
        assert "/api/routing/detect-domain" in handler.ROUTES
        assert "/api/routing/domain-leaderboard" in handler.ROUTES

    def test_can_handle_best_teams(self, handler):
        assert handler.can_handle("/api/routing/best-teams") is True

    def test_can_handle_recommendations(self, handler):
        assert handler.can_handle("/api/routing/recommendations") is True

    def test_can_handle_auto_route(self, handler):
        assert handler.can_handle("/api/routing/auto-route") is True

    def test_can_handle_detect_domain(self, handler):
        assert handler.can_handle("/api/routing/detect-domain") is True

    def test_can_handle_domain_leaderboard(self, handler):
        assert handler.can_handle("/api/routing/domain-leaderboard") is True

    def test_can_handle_versioned_path(self, handler):
        assert handler.can_handle("/api/v1/routing/best-teams") is True

    def test_cannot_handle_other_path(self, handler):
        assert handler.can_handle("/api/debates") is False

    def test_default_context(self):
        h = RoutingHandler()
        assert h.ctx == {}


# ===========================================================================
# Test GET /api/routing/best-teams
# ===========================================================================


class TestGetBestTeams:
    """Tests for the best teams endpoint."""

    def test_best_teams_success(self, handler):
        mock_selector = MagicMock()
        mock_selector.get_best_team_combinations.return_value = [
            {"agents": ["claude", "gpt-4"], "win_rate": 0.85}
        ]
        mock_selector_cls = MagicMock(return_value=mock_selector)

        with patch.multiple(
            "aragora.server.handlers.routing",
            ROUTING_AVAILABLE=True,
            AgentSelector=mock_selector_cls,
        ):
            with patch.object(handler, "get_elo_system", return_value=MagicMock()):
                result = handler._get_best_team_combinations(3, 10)
                assert result.status_code == 200
                data = _parse_body(result)
                assert "combinations" in data
                assert data["min_debates"] == 3

    def test_best_teams_routing_unavailable(self, handler):
        with patch.multiple(
            "aragora.server.handlers.routing",
            ROUTING_AVAILABLE=False,
            AgentSelector=None,
        ):
            result = handler._get_best_team_combinations(3, 10)
            assert result.status_code == 503


# ===========================================================================
# Test GET /api/routing/domain-leaderboard
# ===========================================================================


class TestGetDomainLeaderboard:
    """Tests for the domain leaderboard endpoint."""

    def test_domain_leaderboard_success(self, handler):
        mock_selector = MagicMock()
        mock_selector.get_domain_leaderboard.return_value = [{"agent": "claude", "score": 1500}]
        mock_selector_cls = MagicMock()
        mock_selector_cls.create_with_defaults.return_value = mock_selector

        with patch.multiple(
            "aragora.server.handlers.routing",
            ROUTING_AVAILABLE=True,
            AgentSelector=mock_selector_cls,
        ):
            with patch.object(handler, "get_elo_system", return_value=MagicMock()):
                result = handler._get_domain_leaderboard("general", 10)
                assert result.status_code == 200
                data = _parse_body(result)
                assert data["domain"] == "general"
                assert "leaderboard" in data

    def test_domain_leaderboard_routing_unavailable(self, handler):
        with patch.multiple(
            "aragora.server.handlers.routing",
            ROUTING_AVAILABLE=False,
            AgentSelector=None,
        ):
            result = handler._get_domain_leaderboard("general", 10)
            assert result.status_code == 503


# ===========================================================================
# Test POST /api/routing/recommendations
# ===========================================================================


class TestGetRecommendations:
    """Tests for the recommendations endpoint."""

    def test_recommendations_success(self, handler):
        mock_selector = MagicMock()
        mock_selector.get_recommendations.return_value = [{"agent": "claude", "score": 0.92}]
        mock_selector_cls = MagicMock(return_value=mock_selector)
        mock_requirements_cls = MagicMock()

        mock_handler = _make_mock_handler("POST")
        body = {
            "primary_domain": "code",
            "description": "Build a REST API",
            "task_id": "task-001",
        }

        with patch.multiple(
            "aragora.server.handlers.routing",
            ROUTING_AVAILABLE=True,
            AgentSelector=mock_selector_cls,
            TaskRequirements=mock_requirements_cls,
        ):
            with patch.object(handler, "read_json_body", return_value=body):
                with patch.object(handler, "get_elo_system", return_value=MagicMock()):
                    result = handler._get_recommendations(mock_handler)
                    assert result.status_code == 200
                    data = _parse_body(result)
                    assert data["task_id"] == "task-001"
                    assert data["primary_domain"] == "code"

    def test_recommendations_invalid_body(self, handler):
        mock_handler = _make_mock_handler("POST")
        mock_selector_cls = MagicMock()
        mock_requirements_cls = MagicMock()

        with patch.multiple(
            "aragora.server.handlers.routing",
            ROUTING_AVAILABLE=True,
            AgentSelector=mock_selector_cls,
            TaskRequirements=mock_requirements_cls,
        ):
            with patch.object(handler, "read_json_body", return_value=None):
                result = handler._get_recommendations(mock_handler)
                assert result.status_code == 400

    def test_recommendations_routing_unavailable(self, handler):
        mock_handler = _make_mock_handler("POST")
        with patch.multiple(
            "aragora.server.handlers.routing",
            ROUTING_AVAILABLE=False,
            AgentSelector=None,
            TaskRequirements=None,
        ):
            result = handler._get_recommendations(mock_handler)
            assert result.status_code == 503


# ===========================================================================
# Test POST /api/routing/auto-route
# ===========================================================================


class TestAutoRoute:
    """Tests for the auto-route endpoint."""

    def test_auto_route_success(self, handler):
        mock_team = MockTeam()
        mock_selector = MagicMock()
        mock_selector.auto_route.return_value = mock_team
        mock_selector_cls = MagicMock()
        mock_selector_cls.create_with_defaults.return_value = mock_selector

        mock_handler = _make_mock_handler("POST")

        with patch.multiple(
            "aragora.server.handlers.routing",
            ROUTING_AVAILABLE=True,
            AgentSelector=mock_selector_cls,
            DomainDetector=MagicMock(),
        ):
            with patch.object(
                handler,
                "read_json_body",
                return_value={"task": "Design a rate limiter"},
            ):
                with patch.object(handler, "get_elo_system", return_value=MagicMock()):
                    result = handler._auto_route(mock_handler)
                    assert result.status_code == 200
                    data = _parse_body(result)
                    assert data["task_id"] == "task-001"
                    assert "team" in data

    def test_auto_route_missing_task(self, handler):
        mock_handler = _make_mock_handler("POST")
        mock_selector_cls = MagicMock()

        with patch.multiple(
            "aragora.server.handlers.routing",
            ROUTING_AVAILABLE=True,
            AgentSelector=mock_selector_cls,
            DomainDetector=MagicMock(),
        ):
            with patch.object(handler, "read_json_body", return_value={"task": ""}):
                result = handler._auto_route(mock_handler)
                assert result.status_code == 400

    def test_auto_route_invalid_body(self, handler):
        mock_handler = _make_mock_handler("POST")
        mock_selector_cls = MagicMock()

        with patch.multiple(
            "aragora.server.handlers.routing",
            ROUTING_AVAILABLE=True,
            AgentSelector=mock_selector_cls,
            DomainDetector=MagicMock(),
        ):
            with patch.object(handler, "read_json_body", return_value=None):
                result = handler._auto_route(mock_handler)
                assert result.status_code == 400

    def test_auto_route_routing_unavailable(self, handler):
        mock_handler = _make_mock_handler("POST")
        with patch.multiple(
            "aragora.server.handlers.routing",
            ROUTING_AVAILABLE=False,
            AgentSelector=None,
            DomainDetector=None,
        ):
            result = handler._auto_route(mock_handler)
            assert result.status_code == 503


# ===========================================================================
# Test POST /api/routing/detect-domain
# ===========================================================================


class TestDetectDomain:
    """Tests for the domain detection endpoint."""

    def test_detect_domain_success(self, handler):
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            ("code", 0.9),
            ("general", 0.5),
        ]
        mock_detector_cls = MagicMock(return_value=mock_detector)

        mock_handler = _make_mock_handler("POST")

        with patch.multiple(
            "aragora.server.handlers.routing",
            ROUTING_AVAILABLE=True,
            DomainDetector=mock_detector_cls,
        ):
            with patch.object(
                handler,
                "read_json_body",
                return_value={"task": "Build a REST API"},
            ):
                result = handler._detect_domain(mock_handler)
                assert result.status_code == 200
                data = _parse_body(result)
                assert data["primary_domain"] == "code"
                assert len(data["domains"]) == 2

    def test_detect_domain_missing_task(self, handler):
        mock_handler = _make_mock_handler("POST")

        with patch.multiple(
            "aragora.server.handlers.routing",
            ROUTING_AVAILABLE=True,
            DomainDetector=MagicMock(),
        ):
            with patch.object(handler, "read_json_body", return_value={"task": ""}):
                result = handler._detect_domain(mock_handler)
                assert result.status_code == 400

    def test_detect_domain_routing_unavailable(self, handler):
        mock_handler = _make_mock_handler("POST")
        with patch.multiple(
            "aragora.server.handlers.routing",
            ROUTING_AVAILABLE=False,
            DomainDetector=None,
        ):
            result = handler._detect_domain(mock_handler)
            assert result.status_code == 503

    def test_detect_domain_truncates_long_task(self, handler):
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [("general", 0.5)]
        mock_detector_cls = MagicMock(return_value=mock_detector)
        mock_handler = _make_mock_handler("POST")
        long_task = "x" * 300

        with patch.multiple(
            "aragora.server.handlers.routing",
            ROUTING_AVAILABLE=True,
            DomainDetector=mock_detector_cls,
        ):
            with patch.object(
                handler,
                "read_json_body",
                return_value={"task": long_task},
            ):
                result = handler._detect_domain(mock_handler)
                assert result.status_code == 200
                data = _parse_body(result)
                assert data["task"].endswith("...")
                assert len(data["task"]) == 203  # 200 + "..."


# ===========================================================================
# Test handle() Routing (GET)
# ===========================================================================


class TestHandleRouting:
    """Tests for the top-level handle() method routing."""

    def test_handle_best_teams(self, handler):
        mock_handler = _make_mock_handler()
        mock_selector = MagicMock()
        mock_selector.get_best_team_combinations.return_value = []
        mock_selector_cls = MagicMock(return_value=mock_selector)

        with patch.multiple(
            "aragora.server.handlers.routing",
            ROUTING_AVAILABLE=True,
            AgentSelector=mock_selector_cls,
        ):
            with patch.object(handler, "get_elo_system", return_value=MagicMock()):
                result = handler.handle("/api/routing/best-teams", {}, mock_handler)
                assert result is not None
                assert result.status_code == 200

    def test_handle_domain_leaderboard(self, handler):
        mock_handler = _make_mock_handler()
        mock_selector = MagicMock()
        mock_selector.get_domain_leaderboard.return_value = []
        mock_selector_cls = MagicMock()
        mock_selector_cls.create_with_defaults.return_value = mock_selector

        with patch.multiple(
            "aragora.server.handlers.routing",
            ROUTING_AVAILABLE=True,
            AgentSelector=mock_selector_cls,
        ):
            with patch.object(handler, "get_elo_system", return_value=MagicMock()):
                result = handler.handle(
                    "/api/routing/domain-leaderboard",
                    {"domain": ["code"]},
                    mock_handler,
                )
                assert result is not None
                assert result.status_code == 200

    def test_handle_unmatched_returns_none(self, handler):
        mock_handler = _make_mock_handler()
        result = handler.handle("/api/routing/unknown", {}, mock_handler)
        assert result is None

    def test_handle_rate_limited(self, handler):
        from aragora.server.handlers.routing import _routing_limiter

        mock_handler = _make_mock_handler()
        with patch.object(_routing_limiter, "is_allowed", return_value=False):
            result = handler.handle("/api/routing/best-teams", {}, mock_handler)
            assert result.status_code == 429


# ===========================================================================
# Test handle_post() Routing
# ===========================================================================


class TestHandlePostRouting:
    """Tests for the handle_post() method routing."""

    def test_handle_post_recommendations(self, handler):
        mock_handler = _make_mock_handler("POST")
        mock_selector = MagicMock()
        mock_selector.get_recommendations.return_value = []
        mock_selector_cls = MagicMock(return_value=mock_selector)
        mock_requirements_cls = MagicMock()

        with patch.multiple(
            "aragora.server.handlers.routing",
            ROUTING_AVAILABLE=True,
            AgentSelector=mock_selector_cls,
            TaskRequirements=mock_requirements_cls,
        ):
            with patch.object(
                handler,
                "read_json_body",
                return_value={"primary_domain": "general"},
            ):
                with patch.object(handler, "get_elo_system", return_value=MagicMock()):
                    result = handler.handle_post("/api/routing/recommendations", {}, mock_handler)
                    assert result is not None
                    assert result.status_code == 200

    def test_handle_post_detect_domain(self, handler):
        mock_handler = _make_mock_handler("POST")
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [("general", 0.5)]
        mock_detector_cls = MagicMock(return_value=mock_detector)

        with patch.multiple(
            "aragora.server.handlers.routing",
            ROUTING_AVAILABLE=True,
            DomainDetector=mock_detector_cls,
        ):
            with patch.object(
                handler,
                "read_json_body",
                return_value={"task": "Something"},
            ):
                result = handler.handle_post("/api/routing/detect-domain", {}, mock_handler)
                assert result is not None
                assert result.status_code == 200

    def test_handle_post_unmatched_returns_none(self, handler):
        mock_handler = _make_mock_handler("POST")
        result = handler.handle_post("/api/routing/unknown", {}, mock_handler)
        assert result is None

    def test_handle_post_versioned_path(self, handler):
        """Verify versioned paths are stripped and routed correctly."""
        mock_handler = _make_mock_handler("POST")
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [("general", 0.5)]
        mock_detector_cls = MagicMock(return_value=mock_detector)

        with patch.multiple(
            "aragora.server.handlers.routing",
            ROUTING_AVAILABLE=True,
            DomainDetector=mock_detector_cls,
        ):
            with patch.object(
                handler,
                "read_json_body",
                return_value={"task": "Something"},
            ):
                result = handler.handle_post("/api/v1/routing/detect-domain", {}, mock_handler)
                assert result is not None
                assert result.status_code == 200
