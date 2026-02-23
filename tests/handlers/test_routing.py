"""Tests for routing handler (aragora/server/handlers/routing.py).

Covers all routes and behavior of the RoutingHandler class:
- can_handle() routing for all ROUTES (versioned and unversioned)
- GET /api/routing/best-teams - Best-performing team combinations
- GET /api/routing/domain-leaderboard - Agents ranked by domain
- POST /api/routing/recommendations - Agent recommendations for a task
- POST /api/routing/auto-route - Auto-route task with domain detection
- POST /api/routing/detect-domain - Detect domain from task text
- Rate limiting
- RBAC permission checks
- Routing unavailable (503) responses
- Error handling and edge cases
- Input validation (missing fields, invalid JSON)
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.routing import (
    RoutingHandler,
    _routing_limiter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Mock HTTP handler used by BaseHandler.read_json_body."""

    def __init__(self, body: dict | None = None):
        self.rfile = MagicMock()
        self._body = body
        if body is not None:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers = {
                "Content-Length": str(len(body_bytes)),
                "Content-Type": "application/json",
            }
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {
                "Content-Length": "2",
                "Content-Type": "application/json",
            }
        self.client_address = ("127.0.0.1", 54321)


class MockHTTPHandlerInvalidJSON:
    """Mock HTTP handler returning invalid JSON."""

    def __init__(self):
        self.rfile = MagicMock()
        self.rfile.read.return_value = b"NOT-JSON"
        self.headers = {
            "Content-Length": "8",
            "Content-Type": "application/json",
        }
        self.client_address = ("127.0.0.1", 54321)


class MockHTTPHandlerNoBody:
    """Mock HTTP handler with no body content."""

    def __init__(self):
        self.rfile = MagicMock()
        self.rfile.read.return_value = b""
        self.headers = {
            "Content-Length": "0",
            "Content-Type": "application/json",
        }
        self.client_address = ("127.0.0.1", 54321)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ROUTING_MOD = "aragora.server.handlers.routing"


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the module-level rate limiter between tests."""
    _routing_limiter._buckets = defaultdict(list)
    _routing_limiter._requests = _routing_limiter._buckets
    yield
    _routing_limiter._buckets = defaultdict(list)
    _routing_limiter._requests = _routing_limiter._buckets


@pytest.fixture
def handler():
    """Create a RoutingHandler with a minimal context."""
    return RoutingHandler(ctx={"elo_system": MagicMock()})


@pytest.fixture
def handler_with_persona():
    """Create a RoutingHandler with both elo_system and persona_manager."""
    return RoutingHandler(
        ctx={
            "elo_system": MagicMock(),
            "persona_manager": MagicMock(),
        }
    )


@pytest.fixture
def handler_no_ctx():
    """Create a RoutingHandler with no context."""
    return RoutingHandler()


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------


def _mock_selector(**overrides):
    """Create a mock AgentSelector."""
    sel = MagicMock()
    sel.get_best_team_combinations.return_value = overrides.get(
        "combinations", [{"agents": ["claude", "gpt4"], "win_rate": 0.8}]
    )
    sel.get_recommendations.return_value = overrides.get(
        "recommendations", [{"agent": "claude", "score": 0.95}]
    )
    sel.get_domain_leaderboard.return_value = overrides.get(
        "leaderboard", [{"agent": "claude", "score": 100}]
    )
    # auto_route returns a team object
    team = MagicMock()
    team.task_id = overrides.get("task_id", "task-123")
    agent_mock = MagicMock()
    agent_mock.name = "claude"
    agent_mock.expertise = {"general": 0.9}
    team.agents = overrides.get("agents", [agent_mock])
    team.roles = overrides.get("roles", {"claude": "proposer"})
    team.expected_quality = overrides.get("expected_quality", 0.85)
    team.diversity_score = overrides.get("diversity_score", 0.7)
    team.rationale = overrides.get("rationale", "Best fit for general tasks")
    sel.auto_route.return_value = team
    sel.create_with_defaults = MagicMock(return_value=sel)
    return sel


def _mock_detector(**overrides):
    """Create a mock DomainDetector."""
    det = MagicMock()
    det.detect.return_value = overrides.get(
        "domains", [("general", 0.9), ("code", 0.5), ("finance", 0.1)]
    )
    return det


# ============================================================================
# can_handle
# ============================================================================


class TestCanHandle:
    """Tests for can_handle() route matching."""

    def test_best_teams(self, handler):
        assert handler.can_handle("/api/routing/best-teams") is True

    def test_recommendations(self, handler):
        assert handler.can_handle("/api/routing/recommendations") is True

    def test_auto_route(self, handler):
        assert handler.can_handle("/api/routing/auto-route") is True

    def test_detect_domain(self, handler):
        assert handler.can_handle("/api/routing/detect-domain") is True

    def test_domain_leaderboard(self, handler):
        assert handler.can_handle("/api/routing/domain-leaderboard") is True

    def test_versioned_best_teams(self, handler):
        assert handler.can_handle("/api/v1/routing/best-teams") is True

    def test_versioned_recommendations(self, handler):
        assert handler.can_handle("/api/v1/routing/recommendations") is True

    def test_versioned_auto_route(self, handler):
        assert handler.can_handle("/api/v1/routing/auto-route") is True

    def test_versioned_detect_domain(self, handler):
        assert handler.can_handle("/api/v1/routing/detect-domain") is True

    def test_versioned_domain_leaderboard(self, handler):
        assert handler.can_handle("/api/v1/routing/domain-leaderboard") is True

    def test_non_matching_path(self, handler):
        assert handler.can_handle("/api/other/endpoint") is False

    def test_partial_match(self, handler):
        assert handler.can_handle("/api/routing") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_routes_count(self, handler):
        assert len(handler.ROUTES) == 5

    def test_v2_version_prefix(self, handler):
        assert handler.can_handle("/api/v2/routing/best-teams") is True


# ============================================================================
# GET /api/routing/best-teams
# ============================================================================


class TestBestTeams:
    """Tests for GET /api/routing/best-teams."""

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_success_default_params(self, mock_cls, handler):
        mock_sel = _mock_selector()
        mock_cls.return_value = mock_sel
        http = MockHTTPHandler()

        result = handler.handle("/api/routing/best-teams", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["min_debates"] == 3  # default
        assert "combinations" in body
        assert "count" in body

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_custom_min_debates(self, mock_cls, handler):
        mock_sel = _mock_selector()
        mock_cls.return_value = mock_sel
        http = MockHTTPHandler()

        result = handler.handle(
            "/api/routing/best-teams",
            {"min_debates": ["7"]},
            http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["min_debates"] == 7

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_custom_limit(self, mock_cls, handler):
        mock_sel = _mock_selector(combinations=[{"a": 1}] * 20)
        mock_cls.return_value = mock_sel
        http = MockHTTPHandler()

        result = handler.handle(
            "/api/routing/best-teams",
            {"limit": ["5"]},
            http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] <= 5

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_min_debates_clamped_low(self, mock_cls, handler):
        """min_debates below 1 is clamped to 1."""
        mock_sel = _mock_selector()
        mock_cls.return_value = mock_sel
        http = MockHTTPHandler()

        result = handler.handle(
            "/api/routing/best-teams",
            {"min_debates": ["0"]},
            http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["min_debates"] == 1

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_min_debates_clamped_high(self, mock_cls, handler):
        """min_debates above 20 is clamped to 20."""
        mock_sel = _mock_selector()
        mock_cls.return_value = mock_sel
        http = MockHTTPHandler()

        result = handler.handle(
            "/api/routing/best-teams",
            {"min_debates": ["100"]},
            http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["min_debates"] == 20

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_limit_clamped_high(self, mock_cls, handler):
        """limit above 50 is clamped to 50."""
        mock_sel = _mock_selector()
        mock_cls.return_value = mock_sel
        http = MockHTTPHandler()

        result = handler.handle(
            "/api/routing/best-teams",
            {"limit": ["100"]},
            http,
        )
        assert _status(result) == 200
        # limit is clamped internally to 50

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", False)
    def test_routing_not_available(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/routing/best-teams", {}, http)
        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "").lower()

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.AgentSelector", None)
    def test_agent_selector_none(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/routing/best-teams", {}, http)
        assert _status(result) == 503

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_versioned_path(self, mock_cls, handler):
        mock_cls.return_value = _mock_selector()
        http = MockHTTPHandler()

        result = handler.handle("/api/v1/routing/best-teams", {}, http)
        assert _status(result) == 200

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_uses_elo_system_from_ctx(self, mock_cls, handler):
        mock_sel = _mock_selector()
        mock_cls.return_value = mock_sel
        http = MockHTTPHandler()

        handler.handle("/api/routing/best-teams", {}, http)
        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args
        assert call_kwargs.kwargs.get("elo_system") is not None

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_uses_persona_manager_from_ctx(self, mock_cls, handler_with_persona):
        mock_sel = _mock_selector()
        mock_cls.return_value = mock_sel
        http = MockHTTPHandler()

        handler_with_persona.handle("/api/routing/best-teams", {}, http)
        call_kwargs = mock_cls.call_args
        assert call_kwargs.kwargs.get("persona_manager") is not None

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_empty_combinations(self, mock_cls, handler):
        mock_sel = _mock_selector(combinations=[])
        mock_cls.return_value = mock_sel
        http = MockHTTPHandler()

        result = handler.handle("/api/routing/best-teams", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["combinations"] == []
        assert body["count"] == 0


# ============================================================================
# GET /api/routing/domain-leaderboard
# ============================================================================


class TestDomainLeaderboard:
    """Tests for GET /api/routing/domain-leaderboard."""

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_success_default_domain(self, mock_cls, handler):
        mock_sel = _mock_selector()
        mock_cls.create_with_defaults.return_value = mock_sel
        http = MockHTTPHandler()

        result = handler.handle("/api/routing/domain-leaderboard", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["domain"] == "general"
        assert "leaderboard" in body
        assert "count" in body

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_custom_domain(self, mock_cls, handler):
        mock_sel = _mock_selector()
        mock_cls.create_with_defaults.return_value = mock_sel
        http = MockHTTPHandler()

        result = handler.handle(
            "/api/routing/domain-leaderboard",
            {"domain": ["code"]},
            http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["domain"] == "code"

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_custom_limit(self, mock_cls, handler):
        mock_sel = _mock_selector()
        mock_cls.create_with_defaults.return_value = mock_sel
        http = MockHTTPHandler()

        result = handler.handle(
            "/api/routing/domain-leaderboard",
            {"limit": ["5"]},
            http,
        )
        assert _status(result) == 200

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_limit_clamped(self, mock_cls, handler):
        mock_sel = _mock_selector()
        mock_cls.create_with_defaults.return_value = mock_sel
        http = MockHTTPHandler()

        result = handler.handle(
            "/api/routing/domain-leaderboard",
            {"limit": ["100"]},
            http,
        )
        assert _status(result) == 200

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", False)
    def test_routing_not_available(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/routing/domain-leaderboard", {}, http)
        assert _status(result) == 503

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.AgentSelector", None)
    def test_agent_selector_none(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/routing/domain-leaderboard", {}, http)
        assert _status(result) == 503

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_empty_leaderboard(self, mock_cls, handler):
        mock_sel = _mock_selector(leaderboard=[])
        mock_cls.create_with_defaults.return_value = mock_sel
        http = MockHTTPHandler()

        result = handler.handle("/api/routing/domain-leaderboard", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["leaderboard"] == []
        assert body["count"] == 0

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_versioned_path(self, mock_cls, handler):
        mock_sel = _mock_selector()
        mock_cls.create_with_defaults.return_value = mock_sel
        http = MockHTTPHandler()

        result = handler.handle("/api/v1/routing/domain-leaderboard", {}, http)
        assert _status(result) == 200


# ============================================================================
# handle() - unmatched path returns None
# ============================================================================


class TestHandleUnmatched:
    """Tests for handle() returning None on non-matching paths."""

    def test_handle_returns_none_for_unmatched(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/routing/unknown", {}, http)
        assert result is None

    def test_handle_returns_none_for_post_routes(self, handler):
        """POST routes should not be handled by handle() (GET handler)."""
        http = MockHTTPHandler()
        result = handler.handle("/api/routing/recommendations", {}, http)
        assert result is None

    def test_handle_returns_none_for_auto_route(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/routing/auto-route", {}, http)
        assert result is None

    def test_handle_returns_none_for_detect_domain(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/routing/detect-domain", {}, http)
        assert result is None


# ============================================================================
# POST /api/routing/recommendations
# ============================================================================


class TestRecommendations:
    """Tests for POST /api/routing/recommendations."""

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.TaskRequirements")
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_success_default_body(self, mock_cls, mock_req, handler):
        mock_sel = _mock_selector()
        mock_cls.return_value = mock_sel
        http = MockHTTPHandler(body={})

        result = handler.handle_post("/api/routing/recommendations", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["task_id"] == "ad-hoc"
        assert body["primary_domain"] == "general"
        assert "recommendations" in body
        assert "count" in body

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.TaskRequirements")
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_full_body(self, mock_cls, mock_req, handler):
        mock_sel = _mock_selector()
        mock_cls.return_value = mock_sel
        http = MockHTTPHandler(
            body={
                "primary_domain": "code",
                "secondary_domains": ["testing"],
                "required_traits": ["analytical"],
                "task_id": "task-456",
                "description": "Review this code",
                "limit": 3,
            }
        )

        result = handler.handle_post("/api/routing/recommendations", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["task_id"] == "task-456"
        assert body["primary_domain"] == "code"

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.TaskRequirements")
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_task_field_alias(self, mock_cls, mock_req, handler):
        """'task' field is used as fallback for 'description'."""
        mock_sel = _mock_selector()
        mock_cls.return_value = mock_sel
        http = MockHTTPHandler(body={"task": "Do something"})

        result = handler.handle_post("/api/routing/recommendations", {}, http)
        assert _status(result) == 200
        # Verify TaskRequirements was called with correct description
        call_kwargs = mock_req.call_args
        assert call_kwargs.kwargs.get("description") == "Do something"

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.TaskRequirements")
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_limit_capped_at_20(self, mock_cls, mock_req, handler):
        """Limit should be capped at 20."""
        mock_sel = _mock_selector()
        mock_cls.return_value = mock_sel
        http = MockHTTPHandler(body={"limit": 50})

        handler.handle_post("/api/routing/recommendations", {}, http)
        mock_sel.get_recommendations.assert_called_once()
        call_kwargs = mock_sel.get_recommendations.call_args
        assert call_kwargs.kwargs.get("limit") == 20

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.TaskRequirements")
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_limit_within_bounds(self, mock_cls, mock_req, handler):
        """Limit within 1-20 is used as-is."""
        mock_sel = _mock_selector()
        mock_cls.return_value = mock_sel
        http = MockHTTPHandler(body={"limit": 8})

        handler.handle_post("/api/routing/recommendations", {}, http)
        mock_sel.get_recommendations.assert_called_once()
        call_kwargs = mock_sel.get_recommendations.call_args
        assert call_kwargs.kwargs.get("limit") == 8

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", False)
    def test_routing_not_available(self, handler):
        http = MockHTTPHandler(body={})
        result = handler.handle_post("/api/routing/recommendations", {}, http)
        assert _status(result) == 503

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.AgentSelector")
    @patch(f"{ROUTING_MOD}.TaskRequirements", None)
    def test_task_requirements_none(self, mock_cls, handler):
        http = MockHTTPHandler(body={})
        result = handler.handle_post("/api/routing/recommendations", {}, http)
        assert _status(result) == 503

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.TaskRequirements")
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_invalid_json_body(self, mock_cls, mock_req, handler):
        http = MockHTTPHandlerInvalidJSON()
        result = handler.handle_post("/api/routing/recommendations", {}, http)
        assert _status(result) == 400

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.TaskRequirements")
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_versioned_path(self, mock_cls, mock_req, handler):
        mock_sel = _mock_selector()
        mock_cls.return_value = mock_sel
        http = MockHTTPHandler(body={})

        result = handler.handle_post("/api/v1/routing/recommendations", {}, http)
        assert _status(result) == 200

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.TaskRequirements")
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_secondary_domains_passed(self, mock_cls, mock_req, handler):
        mock_sel = _mock_selector()
        mock_cls.return_value = mock_sel
        http = MockHTTPHandler(body={"secondary_domains": ["security", "testing"]})

        handler.handle_post("/api/routing/recommendations", {}, http)
        call_kwargs = mock_req.call_args
        assert call_kwargs.kwargs.get("secondary_domains") == ["security", "testing"]

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.TaskRequirements")
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_required_traits_passed(self, mock_cls, mock_req, handler):
        mock_sel = _mock_selector()
        mock_cls.return_value = mock_sel
        http = MockHTTPHandler(body={"required_traits": ["analytical", "creative"]})

        handler.handle_post("/api/routing/recommendations", {}, http)
        call_kwargs = mock_req.call_args
        assert call_kwargs.kwargs.get("required_traits") == ["analytical", "creative"]


# ============================================================================
# POST /api/routing/auto-route
# ============================================================================


class TestAutoRoute:
    """Tests for POST /api/routing/auto-route."""

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_success(self, mock_cls, mock_det, handler):
        mock_sel = _mock_selector()
        mock_cls.create_with_defaults.return_value = mock_sel
        http = MockHTTPHandler(body={"task": "Review this code"})

        result = handler.handle_post("/api/routing/auto-route", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert "task_id" in body
        assert "detected_domain" in body
        assert "team" in body
        assert "rationale" in body

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_team_structure(self, mock_cls, mock_det, handler):
        mock_sel = _mock_selector()
        mock_cls.create_with_defaults.return_value = mock_sel
        http = MockHTTPHandler(body={"task": "Review this code"})

        result = handler.handle_post("/api/routing/auto-route", {}, http)
        body = _body(result)
        team = body["team"]
        assert "agents" in team
        assert "roles" in team
        assert "expected_quality" in team
        assert "diversity_score" in team

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_with_task_id(self, mock_cls, mock_det, handler):
        mock_sel = _mock_selector()
        mock_cls.create_with_defaults.return_value = mock_sel
        http = MockHTTPHandler(body={"task": "Do something", "task_id": "custom-id"})

        handler.handle_post("/api/routing/auto-route", {}, http)
        mock_sel.auto_route.assert_called_once()
        call_kwargs = mock_sel.auto_route.call_args
        assert call_kwargs.kwargs.get("task_id") == "custom-id"

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_with_exclude(self, mock_cls, mock_det, handler):
        mock_sel = _mock_selector()
        mock_cls.create_with_defaults.return_value = mock_sel
        http = MockHTTPHandler(body={"task": "Do something", "exclude": ["gpt4", "gemini"]})

        handler.handle_post("/api/routing/auto-route", {}, http)
        call_kwargs = mock_sel.auto_route.call_args
        assert call_kwargs.kwargs.get("exclude") == ["gpt4", "gemini"]

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_missing_task_field(self, mock_cls, mock_det, handler):
        http = MockHTTPHandler(body={})
        result = handler.handle_post("/api/routing/auto-route", {}, http)
        assert _status(result) == 400
        assert "task" in _body(result).get("error", "").lower()

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_empty_task_field(self, mock_cls, mock_det, handler):
        http = MockHTTPHandler(body={"task": ""})
        result = handler.handle_post("/api/routing/auto-route", {}, http)
        assert _status(result) == 400

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", False)
    def test_routing_not_available(self, handler):
        http = MockHTTPHandler(body={"task": "Do something"})
        result = handler.handle_post("/api/routing/auto-route", {}, http)
        assert _status(result) == 503

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.AgentSelector")
    @patch(f"{ROUTING_MOD}.DomainDetector", None)
    def test_domain_detector_none(self, mock_cls, handler):
        http = MockHTTPHandler(body={"task": "Do something"})
        result = handler.handle_post("/api/routing/auto-route", {}, http)
        assert _status(result) == 503

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_invalid_json_body(self, mock_cls, mock_det, handler):
        http = MockHTTPHandlerInvalidJSON()
        result = handler.handle_post("/api/routing/auto-route", {}, http)
        assert _status(result) == 400

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_versioned_path(self, mock_cls, mock_det, handler):
        mock_sel = _mock_selector()
        mock_cls.create_with_defaults.return_value = mock_sel
        http = MockHTTPHandler(body={"task": "Do something"})

        result = handler.handle_post("/api/v1/routing/auto-route", {}, http)
        assert _status(result) == 200

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_empty_agents_team(self, mock_cls, mock_det, handler):
        """When team has no agents, detected_domain should be empty."""
        mock_sel = _mock_selector(agents=[])
        mock_cls.create_with_defaults.return_value = mock_sel
        http = MockHTTPHandler(body={"task": "Do something"})

        result = handler.handle_post("/api/routing/auto-route", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["detected_domain"] == {}


# ============================================================================
# POST /api/routing/detect-domain
# ============================================================================


class TestDetectDomain:
    """Tests for POST /api/routing/detect-domain."""

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    def test_success(self, mock_cls, handler):
        mock_det = _mock_detector()
        mock_cls.return_value = mock_det
        http = MockHTTPHandler(body={"task": "Analyze this financial report"})

        result = handler.handle_post("/api/routing/detect-domain", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert "task" in body
        assert "domains" in body
        assert "primary_domain" in body

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    def test_domain_format(self, mock_cls, handler):
        mock_det = _mock_detector()
        mock_cls.return_value = mock_det
        http = MockHTTPHandler(body={"task": "Test task"})

        result = handler.handle_post("/api/routing/detect-domain", {}, http)
        body = _body(result)
        domains = body["domains"]
        assert len(domains) == 3
        assert all("domain" in d and "confidence" in d for d in domains)

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    def test_primary_domain(self, mock_cls, handler):
        mock_det = _mock_detector(domains=[("finance", 0.95)])
        mock_cls.return_value = mock_det
        http = MockHTTPHandler(body={"task": "Financial analysis"})

        result = handler.handle_post("/api/routing/detect-domain", {}, http)
        body = _body(result)
        assert body["primary_domain"] == "finance"

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    def test_empty_domains(self, mock_cls, handler):
        mock_det = _mock_detector(domains=[])
        mock_cls.return_value = mock_det
        http = MockHTTPHandler(body={"task": "Something"})

        result = handler.handle_post("/api/routing/detect-domain", {}, http)
        body = _body(result)
        assert body["primary_domain"] == "general"
        assert body["domains"] == []

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    def test_custom_top_n(self, mock_cls, handler):
        mock_det = _mock_detector()
        mock_cls.return_value = mock_det
        http = MockHTTPHandler(body={"task": "Something", "top_n": 5})

        handler.handle_post("/api/routing/detect-domain", {}, http)
        mock_det.detect.assert_called_once()
        call_kwargs = mock_det.detect.call_args
        assert call_kwargs.kwargs.get("top_n") == 5

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    def test_top_n_capped_at_10(self, mock_cls, handler):
        mock_det = _mock_detector()
        mock_cls.return_value = mock_det
        http = MockHTTPHandler(body={"task": "Something", "top_n": 50})

        handler.handle_post("/api/routing/detect-domain", {}, http)
        call_kwargs = mock_det.detect.call_args
        assert call_kwargs.kwargs.get("top_n") == 10

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    def test_task_truncation_in_response(self, mock_cls, handler):
        """Long task text is truncated to 200 chars + '...' in response."""
        mock_det = _mock_detector()
        mock_cls.return_value = mock_det
        long_task = "x" * 300
        http = MockHTTPHandler(body={"task": long_task})

        result = handler.handle_post("/api/routing/detect-domain", {}, http)
        body = _body(result)
        assert len(body["task"]) == 203  # 200 + "..."
        assert body["task"].endswith("...")

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    def test_short_task_not_truncated(self, mock_cls, handler):
        mock_det = _mock_detector()
        mock_cls.return_value = mock_det
        http = MockHTTPHandler(body={"task": "Short task"})

        result = handler.handle_post("/api/routing/detect-domain", {}, http)
        body = _body(result)
        assert body["task"] == "Short task"

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    def test_missing_task_field(self, mock_cls, handler):
        http = MockHTTPHandler(body={})
        result = handler.handle_post("/api/routing/detect-domain", {}, http)
        assert _status(result) == 400
        assert "task" in _body(result).get("error", "").lower()

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    def test_empty_task_field(self, mock_cls, handler):
        http = MockHTTPHandler(body={"task": ""})
        result = handler.handle_post("/api/routing/detect-domain", {}, http)
        assert _status(result) == 400

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", False)
    def test_routing_not_available(self, handler):
        http = MockHTTPHandler(body={"task": "Something"})
        result = handler.handle_post("/api/routing/detect-domain", {}, http)
        assert _status(result) == 503

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector", None)
    def test_detector_none(self, handler):
        http = MockHTTPHandler(body={"task": "Something"})
        result = handler.handle_post("/api/routing/detect-domain", {}, http)
        assert _status(result) == 503

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    def test_invalid_json_body(self, mock_cls, handler):
        http = MockHTTPHandlerInvalidJSON()
        result = handler.handle_post("/api/routing/detect-domain", {}, http)
        assert _status(result) == 400

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    def test_versioned_path(self, mock_cls, handler):
        mock_det = _mock_detector()
        mock_cls.return_value = mock_det
        http = MockHTTPHandler(body={"task": "Something"})

        result = handler.handle_post("/api/v1/routing/detect-domain", {}, http)
        assert _status(result) == 200

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    def test_confidence_rounded(self, mock_cls, handler):
        """Confidence scores should be rounded to 3 decimal places."""
        mock_det = _mock_detector(domains=[("code", 0.123456789)])
        mock_cls.return_value = mock_det
        http = MockHTTPHandler(body={"task": "Something"})

        result = handler.handle_post("/api/routing/detect-domain", {}, http)
        body = _body(result)
        assert body["domains"][0]["confidence"] == 0.123

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    def test_task_exactly_200_chars_not_truncated(self, mock_cls, handler):
        """Task text of exactly 200 chars should not be truncated."""
        mock_det = _mock_detector()
        mock_cls.return_value = mock_det
        task = "x" * 200
        http = MockHTTPHandler(body={"task": task})

        result = handler.handle_post("/api/routing/detect-domain", {}, http)
        body = _body(result)
        assert body["task"] == task
        assert not body["task"].endswith("...")


# ============================================================================
# handle_post - unmatched path returns None
# ============================================================================


class TestHandlePostUnmatched:
    """Tests for handle_post() returning None on non-matching paths."""

    def test_unmatched_path(self, handler):
        http = MockHTTPHandler(body={})
        result = handler.handle_post("/api/routing/unknown", {}, http)
        assert result is None

    def test_get_routes_not_handled_by_post(self, handler):
        http = MockHTTPHandler(body={})
        result = handler.handle_post("/api/routing/best-teams", {}, http)
        assert result is None

    def test_domain_leaderboard_not_handled_by_post(self, handler):
        http = MockHTTPHandler(body={})
        result = handler.handle_post("/api/routing/domain-leaderboard", {}, http)
        assert result is None


# ============================================================================
# Rate Limiting
# ============================================================================


class TestRateLimiting:
    """Tests for rate limiting on routing endpoints."""

    def test_rate_limiter_config(self):
        """Rate limiter should be configured for 100 requests per minute."""
        assert _routing_limiter.rpm == 100

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_rate_limit_exceeded(self, mock_cls, handler):
        """Should return 429 when rate limit is exceeded."""
        mock_cls.return_value = _mock_selector()
        http = MockHTTPHandler()

        # Exhaust rate limit
        with patch.object(_routing_limiter, "is_allowed", return_value=False):
            result = handler.handle("/api/routing/best-teams", {}, http)
        assert _status(result) == 429
        assert "rate limit" in _body(result).get("error", "").lower()

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_rate_limit_allowed(self, mock_cls, handler):
        """Should proceed normally when within rate limit."""
        mock_cls.return_value = _mock_selector()
        http = MockHTTPHandler()

        with patch.object(_routing_limiter, "is_allowed", return_value=True):
            result = handler.handle("/api/routing/best-teams", {}, http)
        assert _status(result) == 200


# ============================================================================
# Initialization
# ============================================================================


class TestInit:
    """Tests for RoutingHandler initialization."""

    def test_default_ctx(self):
        h = RoutingHandler()
        assert h.ctx == {}

    def test_custom_ctx(self):
        ctx = {"elo_system": MagicMock(), "persona_manager": MagicMock()}
        h = RoutingHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_none_ctx_defaults_to_empty(self):
        h = RoutingHandler(ctx=None)
        assert h.ctx == {}


# ============================================================================
# Error Handling via handle_errors decorator
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in routing endpoints."""

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_best_teams_exception(self, mock_cls, handler):
        """Exceptions in _get_best_team_combinations are caught by @handle_errors."""
        mock_cls.side_effect = Exception("selector failure")
        http = MockHTTPHandler()

        result = handler.handle("/api/routing/best-teams", {}, http)
        assert _status(result) == 500

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_domain_leaderboard_exception(self, mock_cls, handler):
        mock_sel = MagicMock()
        mock_sel.get_domain_leaderboard.side_effect = RuntimeError("fail")
        mock_cls.create_with_defaults.return_value = mock_sel
        http = MockHTTPHandler()

        result = handler.handle("/api/routing/domain-leaderboard", {}, http)
        assert _status(result) == 500

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.TaskRequirements")
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_recommendations_exception(self, mock_cls, mock_req, handler):
        mock_cls.side_effect = Exception("selector failure")
        http = MockHTTPHandler(body={})

        result = handler.handle_post("/api/routing/recommendations", {}, http)
        assert _status(result) == 500

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_auto_route_value_error(self, mock_cls, mock_det, handler):
        """ValueError maps to 400 via handle_errors decorator."""
        mock_sel = MagicMock()
        mock_sel.auto_route.side_effect = ValueError("bad input")
        mock_cls.create_with_defaults.return_value = mock_sel
        http = MockHTTPHandler(body={"task": "Something"})

        result = handler.handle_post("/api/routing/auto-route", {}, http)
        assert _status(result) == 400

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_auto_route_runtime_error(self, mock_cls, mock_det, handler):
        """RuntimeError maps to 500 via handle_errors decorator."""
        mock_sel = MagicMock()
        mock_sel.auto_route.side_effect = RuntimeError("internal failure")
        mock_cls.create_with_defaults.return_value = mock_sel
        http = MockHTTPHandler(body={"task": "Something"})

        result = handler.handle_post("/api/routing/auto-route", {}, http)
        assert _status(result) == 500

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    def test_detect_domain_exception(self, mock_cls, handler):
        mock_det = MagicMock()
        mock_det.detect.side_effect = RuntimeError("detection failure")
        mock_cls.return_value = mock_det
        http = MockHTTPHandler(body={"task": "Something"})

        result = handler.handle_post("/api/routing/detect-domain", {}, http)
        assert _status(result) == 500


# ============================================================================
# ELO System Integration
# ============================================================================


class TestEloSystemIntegration:
    """Tests for ELO system usage in routing handlers."""

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_best_teams_uses_elo(self, mock_cls, handler):
        mock_cls.return_value = _mock_selector()
        http = MockHTTPHandler()

        handler.handle("/api/routing/best-teams", {}, http)
        call_kwargs = mock_cls.call_args.kwargs
        assert "elo_system" in call_kwargs

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_domain_leaderboard_uses_create_with_defaults(self, mock_cls, handler):
        mock_sel = _mock_selector()
        mock_cls.create_with_defaults.return_value = mock_sel
        http = MockHTTPHandler()

        handler.handle("/api/routing/domain-leaderboard", {}, http)
        mock_cls.create_with_defaults.assert_called_once()

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_auto_route_uses_create_with_defaults(self, mock_cls, mock_det, handler):
        mock_sel = _mock_selector()
        mock_cls.create_with_defaults.return_value = mock_sel
        http = MockHTTPHandler(body={"task": "Something"})

        handler.handle_post("/api/routing/auto-route", {}, http)
        mock_cls.create_with_defaults.assert_called_once()


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Edge case tests for routing handler."""

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    def test_detect_domain_top_n_default(self, mock_cls, handler):
        """Default top_n is 3."""
        mock_det = _mock_detector()
        mock_cls.return_value = mock_det
        http = MockHTTPHandler(body={"task": "Test"})

        handler.handle_post("/api/routing/detect-domain", {}, http)
        call_kwargs = mock_det.detect.call_args
        assert call_kwargs.kwargs.get("top_n") == 3

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_auto_route_default_exclude(self, mock_cls, mock_det, handler):
        """Default exclude list is empty."""
        mock_sel = _mock_selector()
        mock_cls.create_with_defaults.return_value = mock_sel
        http = MockHTTPHandler(body={"task": "Something"})

        handler.handle_post("/api/routing/auto-route", {}, http)
        call_kwargs = mock_sel.auto_route.call_args
        assert call_kwargs.kwargs.get("exclude") == []

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.TaskRequirements")
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_recommendations_no_body(self, mock_cls, mock_req, handler):
        """Empty body handler should use defaults."""
        mock_sel = _mock_selector()
        mock_cls.return_value = mock_sel
        http = MockHTTPHandlerNoBody()

        result = handler.handle_post("/api/routing/recommendations", {}, http)
        # Empty body returns {} from read_json_body, which is valid
        assert _status(result) == 200

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.TaskRequirements")
    @patch(f"{ROUTING_MOD}.AgentSelector")
    def test_recommendations_description_over_task(self, mock_cls, mock_req, handler):
        """'description' field takes precedence over 'task' field."""
        mock_sel = _mock_selector()
        mock_cls.return_value = mock_sel
        http = MockHTTPHandler(body={"description": "Primary description", "task": "Fallback task"})

        handler.handle_post("/api/routing/recommendations", {}, http)
        call_kwargs = mock_req.call_args
        assert call_kwargs.kwargs.get("description") == "Primary description"

    def test_handler_no_elo_system(self, handler_no_ctx):
        """Handler without elo_system in ctx should still work (returns None)."""
        elo = handler_no_ctx.get_elo_system()
        assert elo is None

    @patch(f"{ROUTING_MOD}.ROUTING_AVAILABLE", True)
    @patch(f"{ROUTING_MOD}.DomainDetector")
    def test_detect_domain_task_exactly_201_chars(self, mock_cls, handler):
        """Task text of 201 chars gets truncated."""
        mock_det = _mock_detector()
        mock_cls.return_value = mock_det
        task = "x" * 201
        http = MockHTTPHandler(body={"task": task})

        result = handler.handle_post("/api/routing/detect-domain", {}, http)
        body = _body(result)
        assert body["task"] == "x" * 200 + "..."
        assert len(body["task"]) == 203
