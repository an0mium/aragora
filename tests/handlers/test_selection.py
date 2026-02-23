"""Tests for selection handler (aragora/server/handlers/selection.py).

Covers all routes and behavior of the SelectionHandler class:
- can_handle() routing for all ROUTES and PREFIX_ROUTES
- GET  /api/v1/selection/plugins               List all plugins
- GET  /api/v1/selection/defaults              Get default plugin config
- GET  /api/v1/selection/scorers/<name>        Get scorer details
- GET  /api/v1/selection/team-selectors/<name> Get team selector details
- GET  /api/v1/selection/role-assigners/<name> Get role assigner details
- GET  /api/v1/agent-selection/history         Get selection history
- POST /api/v1/selection/score                 Score agents for a task
- POST /api/v1/selection/team                  Select a team
- POST /api/v1/selection/assign-roles          Assign roles
- SDK agent-selection alias routes
- Rate limiting behavior
- Error handling (missing params, invalid JSON, unknown plugins, not found)
- Path normalization (agent-selection -> selection)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.selection import (
    SelectionHandler,
    _create_agent_pool,
    _selection_limiter,
    get_team_builder,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class _MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed to SelectionHandler."""

    def __init__(
        self,
        method: str = "GET",
        body: dict[str, Any] | None = None,
        client_address: tuple[str, int] | None = None,
    ):
        self.command = method
        self.headers = {"Content-Length": "0"}
        self.client_address = client_address or ("127.0.0.1", 12345)

        if body is not None:
            raw = json.dumps(body).encode()
            self.request_body = raw
            self.headers = {"Content-Length": str(len(raw))}
        else:
            self.request_body = b"{}"
            self.headers = {"Content-Length": "2"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a SelectionHandler with minimal context."""
    return SelectionHandler({})


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the rate limiter between tests."""
    _selection_limiter._requests.clear()
    yield
    _selection_limiter._requests.clear()


@pytest.fixture(autouse=True)
def _reset_team_builder():
    """Reset the global team builder between tests."""
    import aragora.server.handlers.selection as sel_module

    sel_module._team_builder = None
    yield
    sel_module._team_builder = None


# ============================================================================
# can_handle Routing
# ============================================================================


class TestCanHandle:
    """Verify can_handle correctly accepts or rejects paths."""

    def test_plugins_path(self, handler):
        assert handler.can_handle("/api/v1/selection/plugins")

    def test_defaults_path(self, handler):
        assert handler.can_handle("/api/v1/selection/defaults")

    def test_score_path(self, handler):
        assert handler.can_handle("/api/v1/selection/score")

    def test_team_path(self, handler):
        assert handler.can_handle("/api/v1/selection/team")

    def test_team_selection_path(self, handler):
        assert handler.can_handle("/api/v1/team-selection")

    def test_sdk_plugins_alias(self, handler):
        assert handler.can_handle("/api/v1/agent-selection/plugins")

    def test_sdk_defaults_alias(self, handler):
        assert handler.can_handle("/api/v1/agent-selection/defaults")

    def test_sdk_score_alias(self, handler):
        assert handler.can_handle("/api/v1/agent-selection/score")

    def test_sdk_best_alias(self, handler):
        assert handler.can_handle("/api/v1/agent-selection/best")

    def test_sdk_select_team_alias(self, handler):
        assert handler.can_handle("/api/v1/agent-selection/select-team")

    def test_sdk_assign_roles_alias(self, handler):
        assert handler.can_handle("/api/v1/agent-selection/assign-roles")

    def test_sdk_history_alias(self, handler):
        assert handler.can_handle("/api/v1/agent-selection/history")

    def test_scorer_prefix(self, handler):
        assert handler.can_handle("/api/v1/selection/scorers/elo-weighted")

    def test_team_selector_prefix(self, handler):
        assert handler.can_handle("/api/v1/selection/team-selectors/diverse")

    def test_role_assigner_prefix(self, handler):
        assert handler.can_handle("/api/v1/selection/role-assigners/domain-based")

    def test_rejects_unrelated_path(self, handler):
        assert not handler.can_handle("/api/v1/debates")

    def test_rejects_partial_path(self, handler):
        assert not handler.can_handle("/api/v1/selection")

    def test_rejects_v2_path(self, handler):
        assert not handler.can_handle("/api/v2/selection/plugins")


# ============================================================================
# Path Normalization
# ============================================================================


class TestPathNormalization:
    """Test agent-selection to selection path normalization."""

    def test_normalizes_agent_selection_to_selection(self, handler):
        normalized = handler._normalize_path("/api/v1/agent-selection/plugins")
        assert normalized == "/api/v1/selection/plugins"

    def test_normalizes_agent_selection_score(self, handler):
        normalized = handler._normalize_path("/api/v1/agent-selection/score")
        assert normalized == "/api/v1/selection/score"

    def test_normalizes_agent_selection_best(self, handler):
        normalized = handler._normalize_path("/api/v1/agent-selection/best")
        assert normalized == "/api/v1/selection/best"

    def test_normalizes_agent_selection_history(self, handler):
        normalized = handler._normalize_path("/api/v1/agent-selection/history")
        assert normalized == "/api/v1/selection/history"

    def test_no_change_for_selection_path(self, handler):
        normalized = handler._normalize_path("/api/v1/selection/plugins")
        assert normalized == "/api/v1/selection/plugins"


# ============================================================================
# Handler Initialization
# ============================================================================


class TestInit:
    """Test handler initialization."""

    def test_default_context(self):
        handler = SelectionHandler()
        assert handler.ctx == {}

    def test_custom_context(self):
        ctx = {"key": "value"}
        handler = SelectionHandler(ctx)
        assert handler.ctx == ctx

    def test_extends_base_handler(self, handler):
        from aragora.server.handlers.base import BaseHandler
        assert isinstance(handler, BaseHandler)

    def test_has_routes(self, handler):
        assert len(handler.ROUTES) >= 12

    def test_has_prefix_routes(self, handler):
        assert len(handler.PREFIX_ROUTES) == 3


# ============================================================================
# GET /api/v1/selection/plugins
# ============================================================================


class TestListPlugins:
    """Tests for listing all available selection plugins."""

    def test_returns_200(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/plugins", {}, mock_h)
        assert _status(result) == 200

    def test_response_has_scorers(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/plugins", {}, mock_h)
        body = _body(result)
        assert "scorers" in body

    def test_response_has_team_selectors(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/plugins", {}, mock_h)
        body = _body(result)
        assert "team_selectors" in body

    def test_response_has_role_assigners(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/plugins", {}, mock_h)
        body = _body(result)
        assert "role_assigners" in body

    def test_elo_weighted_scorer_listed(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/plugins", {}, mock_h)
        body = _body(result)
        scorer_names = [s["name"] for s in body["scorers"]]
        assert "elo-weighted" in scorer_names

    def test_diverse_team_selector_listed(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/plugins", {}, mock_h)
        body = _body(result)
        selector_names = [s["name"] for s in body["team_selectors"]]
        assert "diverse" in selector_names

    def test_domain_based_role_assigner_listed(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/plugins", {}, mock_h)
        body = _body(result)
        assigner_names = [a["name"] for a in body["role_assigners"]]
        assert "domain-based" in assigner_names

    def test_sdk_alias_works(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/agent-selection/plugins", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert "scorers" in body


# ============================================================================
# GET /api/v1/selection/defaults
# ============================================================================


class TestGetDefaults:
    """Tests for getting default plugin configuration."""

    def test_returns_200(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/defaults", {}, mock_h)
        assert _status(result) == 200

    def test_has_scorer_default(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/defaults", {}, mock_h)
        body = _body(result)
        assert "scorer" in body
        assert body["scorer"] == "elo-weighted"

    def test_has_team_selector_default(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/defaults", {}, mock_h)
        body = _body(result)
        assert "team_selector" in body
        assert body["team_selector"] == "diverse"

    def test_has_role_assigner_default(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/defaults", {}, mock_h)
        body = _body(result)
        assert "role_assigner" in body
        assert body["role_assigner"] == "domain-based"

    def test_sdk_alias_works(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/agent-selection/defaults", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        assert body["scorer"] == "elo-weighted"


# ============================================================================
# GET /api/v1/selection/scorers/<name>
# ============================================================================


class TestGetScorer:
    """Tests for getting scorer details."""

    def test_known_scorer_returns_200(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/scorers/elo-weighted", {}, mock_h)
        assert _status(result) == 200

    def test_known_scorer_has_name(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/scorers/elo-weighted", {}, mock_h)
        body = _body(result)
        assert body["name"] == "elo-weighted"

    def test_known_scorer_has_description(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/scorers/elo-weighted", {}, mock_h)
        body = _body(result)
        assert "description" in body

    def test_known_scorer_has_is_default(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/scorers/elo-weighted", {}, mock_h)
        body = _body(result)
        assert body["is_default"] is True

    def test_unknown_scorer_returns_404(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/scorers/nonexistent", {}, mock_h)
        assert _status(result) == 404

    def test_unknown_scorer_error_message(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/scorers/nonexistent", {}, mock_h)
        body = _body(result)
        assert "nonexistent" in body.get("error", "")


# ============================================================================
# GET /api/v1/selection/team-selectors/<name>
# ============================================================================


class TestGetTeamSelector:
    """Tests for getting team selector details."""

    def test_known_selector_returns_200(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/team-selectors/diverse", {}, mock_h)
        assert _status(result) == 200

    def test_known_selector_has_name(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/team-selectors/diverse", {}, mock_h)
        body = _body(result)
        assert body["name"] == "diverse"

    def test_known_selector_has_description(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/team-selectors/diverse", {}, mock_h)
        body = _body(result)
        assert "description" in body

    def test_greedy_selector_returns_200(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/team-selectors/greedy", {}, mock_h)
        assert _status(result) == 200

    def test_random_selector_returns_200(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/team-selectors/random", {}, mock_h)
        assert _status(result) == 200

    def test_unknown_selector_returns_404(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/team-selectors/nonexistent", {}, mock_h)
        assert _status(result) == 404

    def test_unknown_selector_error_message(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/team-selectors/nonexistent", {}, mock_h)
        body = _body(result)
        assert "nonexistent" in body.get("error", "")


# ============================================================================
# GET /api/v1/selection/role-assigners/<name>
# ============================================================================


class TestGetRoleAssigner:
    """Tests for getting role assigner details."""

    def test_known_assigner_returns_200(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/role-assigners/domain-based", {}, mock_h)
        assert _status(result) == 200

    def test_known_assigner_has_name(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/role-assigners/domain-based", {}, mock_h)
        body = _body(result)
        assert body["name"] == "domain-based"

    def test_known_assigner_has_description(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/role-assigners/domain-based", {}, mock_h)
        body = _body(result)
        assert "description" in body

    def test_ahmad_assigner_returns_200(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/role-assigners/ahmad", {}, mock_h)
        assert _status(result) == 200

    def test_simple_assigner_returns_200(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/role-assigners/simple", {}, mock_h)
        assert _status(result) == 200

    def test_unknown_assigner_returns_404(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/role-assigners/nonexistent", {}, mock_h)
        assert _status(result) == 404

    def test_unknown_assigner_error_message(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/role-assigners/nonexistent", {}, mock_h)
        body = _body(result)
        assert "nonexistent" in body.get("error", "")


# ============================================================================
# GET /api/v1/selection/history (via agent-selection alias)
# ============================================================================


class TestGetHistory:
    """Tests for getting selection history."""

    def test_returns_200(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/agent-selection/history", {}, mock_h)
        assert _status(result) == 200

    def test_empty_history(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/agent-selection/history", {}, mock_h)
        body = _body(result)
        assert body["selections"] == []
        assert body["total"] == 0

    def test_default_pagination(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/agent-selection/history", {}, mock_h)
        body = _body(result)
        assert body["limit"] == 20
        assert body["offset"] == 0

    def test_custom_limit(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle(
            "/api/v1/agent-selection/history", {"limit": "5"}, mock_h
        )
        body = _body(result)
        assert body["limit"] == 5

    def test_custom_offset(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle(
            "/api/v1/agent-selection/history", {"offset": "10"}, mock_h
        )
        body = _body(result)
        assert body["offset"] == 10

    def test_limit_clamped_to_max(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle(
            "/api/v1/agent-selection/history", {"limit": "9999"}, mock_h
        )
        body = _body(result)
        assert body["limit"] == 500

    def test_limit_clamped_to_min(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle(
            "/api/v1/agent-selection/history", {"limit": "0"}, mock_h
        )
        body = _body(result)
        assert body["limit"] == 1

    def test_offset_clamped_to_min(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle(
            "/api/v1/agent-selection/history", {"offset": "-5"}, mock_h
        )
        body = _body(result)
        assert body["offset"] == 0

    def test_history_with_mocked_data(self, handler):
        """History with team builder data returns properly."""
        tb = get_team_builder()
        tb._selection_history = [
            {"task_id": "t1", "selected": ["claude"], "timestamp": "2026-01-01T00:00:00"},
            {"task_id": "t2", "selected": ["grok"], "timestamp": "2026-01-02T00:00:00"},
        ]
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/agent-selection/history", {}, mock_h)
        body = _body(result)
        assert body["total"] == 2
        assert len(body["selections"]) == 2

    def test_history_pagination_offset(self, handler):
        """Offset skips items correctly."""
        tb = get_team_builder()
        tb._selection_history = [
            {"task_id": f"t{i}", "selected": ["claude"], "timestamp": f"2026-01-{i:02d}T00:00:00"}
            for i in range(1, 6)
        ]
        mock_h = _MockHTTPHandler()
        result = handler.handle(
            "/api/v1/agent-selection/history", {"offset": "3", "limit": "10"}, mock_h
        )
        body = _body(result)
        assert body["total"] == 5
        # offset=3 from a reverse-sorted list of 5 items should return 2
        assert len(body["selections"]) == 2


# ============================================================================
# POST /api/v1/selection/score
# ============================================================================


class TestScoreAgents:
    """Tests for scoring agents for a task."""

    def test_returns_200(self, handler):
        mock_h = _MockHTTPHandler(body={"task_description": "Build a security audit tool"})
        result = handler.handle_post("/api/v1/selection/score", {}, mock_h)
        assert _status(result) == 200

    def test_response_has_agents(self, handler):
        mock_h = _MockHTTPHandler(body={"task_description": "Build a security audit tool"})
        result = handler.handle_post("/api/v1/selection/score", {}, mock_h)
        body = _body(result)
        assert "agents" in body
        assert len(body["agents"]) > 0

    def test_agents_have_required_fields(self, handler):
        mock_h = _MockHTTPHandler(body={"task_description": "Build a security audit tool"})
        result = handler.handle_post("/api/v1/selection/score", {}, mock_h)
        body = _body(result)
        agent = body["agents"][0]
        assert "name" in agent
        assert "type" in agent
        assert "score" in agent
        assert "domain_expertise" in agent
        assert "elo_rating" in agent

    def test_agents_sorted_by_score_desc(self, handler):
        mock_h = _MockHTTPHandler(body={"task_description": "Build a security audit tool"})
        result = handler.handle_post("/api/v1/selection/score", {}, mock_h)
        body = _body(result)
        scores = [a["score"] for a in body["agents"]]
        assert scores == sorted(scores, reverse=True)

    def test_response_has_scorer_used(self, handler):
        mock_h = _MockHTTPHandler(body={"task_description": "Build a security audit tool"})
        result = handler.handle_post("/api/v1/selection/score", {}, mock_h)
        body = _body(result)
        assert "scorer_used" in body

    def test_response_has_task_id(self, handler):
        mock_h = _MockHTTPHandler(body={"task_description": "Build a security audit tool"})
        result = handler.handle_post("/api/v1/selection/score", {}, mock_h)
        body = _body(result)
        assert "task_id" in body
        assert body["task_id"].startswith("score-")

    def test_custom_primary_domain(self, handler):
        mock_h = _MockHTTPHandler(
            body={"task_description": "Build something", "primary_domain": "security"}
        )
        result = handler.handle_post("/api/v1/selection/score", {}, mock_h)
        body = _body(result)
        assert _status(result) == 200
        # Agents should have security domain expertise
        for agent in body["agents"]:
            assert "domain_expertise" in agent

    def test_auto_detect_domain(self, handler):
        mock_h = _MockHTTPHandler(body={"task_description": "Analyze database performance"})
        result = handler.handle_post("/api/v1/selection/score", {}, mock_h)
        assert _status(result) == 200

    def test_missing_task_description_returns_400(self, handler):
        mock_h = _MockHTTPHandler(body={})
        result = handler.handle_post("/api/v1/selection/score", {}, mock_h)
        assert _status(result) == 400

    def test_empty_task_description_returns_400(self, handler):
        mock_h = _MockHTTPHandler(body={"task_description": ""})
        result = handler.handle_post("/api/v1/selection/score", {}, mock_h)
        assert _status(result) == 400

    def test_invalid_json_returns_400(self, handler):
        mock_h = _MockHTTPHandler()
        mock_h.request_body = b"not valid json"
        result = handler.handle_post("/api/v1/selection/score", {}, mock_h)
        assert _status(result) == 400

    def test_invalid_scorer_returns_400(self, handler):
        mock_h = _MockHTTPHandler(
            body={"task_description": "Build a tool", "scorer": "nonexistent"}
        )
        result = handler.handle_post("/api/v1/selection/score", {}, mock_h)
        assert _status(result) == 400

    def test_sdk_alias_score(self, handler):
        mock_h = _MockHTTPHandler(body={"task_description": "Build a security tool"})
        result = handler.handle_post("/api/v1/agent-selection/score", {}, mock_h)
        assert _status(result) == 200

    def test_sdk_alias_best(self, handler):
        mock_h = _MockHTTPHandler(body={"task_description": "Build a security tool"})
        result = handler.handle_post("/api/v1/agent-selection/best", {}, mock_h)
        assert _status(result) == 200

    def test_with_required_traits(self, handler):
        mock_h = _MockHTTPHandler(
            body={
                "task_description": "Analyze a security issue",
                "required_traits": ["analytical", "detail-oriented"],
            }
        )
        result = handler.handle_post("/api/v1/selection/score", {}, mock_h)
        assert _status(result) == 200

    def test_with_secondary_domains(self, handler):
        mock_h = _MockHTTPHandler(
            body={
                "task_description": "Security and performance review",
                "secondary_domains": ["performance", "testing"],
            }
        )
        result = handler.handle_post("/api/v1/selection/score", {}, mock_h)
        assert _status(result) == 200

    def test_task_description_truncated(self, handler):
        """Long task descriptions are truncated to 500 chars."""
        long_desc = "x" * 1000
        mock_h = _MockHTTPHandler(body={"task_description": long_desc})
        result = handler.handle_post("/api/v1/selection/score", {}, mock_h)
        assert _status(result) == 200


# ============================================================================
# POST /api/v1/selection/team
# ============================================================================


class TestSelectTeam:
    """Tests for selecting a team for a task."""

    def test_returns_200(self, handler):
        mock_h = _MockHTTPHandler(body={"task_description": "Design a REST API"})
        result = handler.handle_post("/api/v1/selection/team", {}, mock_h)
        assert _status(result) == 200

    def test_response_has_team_id(self, handler):
        mock_h = _MockHTTPHandler(body={"task_description": "Design a REST API"})
        result = handler.handle_post("/api/v1/selection/team", {}, mock_h)
        body = _body(result)
        assert "team_id" in body
        assert body["team_id"].startswith("team-")

    def test_response_has_task_id(self, handler):
        mock_h = _MockHTTPHandler(body={"task_description": "Design a REST API"})
        result = handler.handle_post("/api/v1/selection/team", {}, mock_h)
        body = _body(result)
        assert "task_id" in body
        assert body["task_id"].startswith("team-")

    def test_response_has_agents(self, handler):
        mock_h = _MockHTTPHandler(body={"task_description": "Design a REST API"})
        result = handler.handle_post("/api/v1/selection/team", {}, mock_h)
        body = _body(result)
        assert "agents" in body
        assert len(body["agents"]) >= 2

    def test_agents_have_required_fields(self, handler):
        mock_h = _MockHTTPHandler(body={"task_description": "Design a REST API"})
        result = handler.handle_post("/api/v1/selection/team", {}, mock_h)
        body = _body(result)
        for agent in body["agents"]:
            assert "name" in agent
            assert "type" in agent
            assert "role" in agent
            assert "score" in agent
            assert "expertise" in agent
            assert "elo_rating" in agent

    def test_response_has_quality_metrics(self, handler):
        mock_h = _MockHTTPHandler(body={"task_description": "Design a REST API"})
        result = handler.handle_post("/api/v1/selection/team", {}, mock_h)
        body = _body(result)
        assert "expected_quality" in body
        assert "expected_cost" in body
        assert "diversity_score" in body
        assert isinstance(body["expected_quality"], float)
        assert isinstance(body["expected_cost"], float)
        assert isinstance(body["diversity_score"], float)

    def test_response_has_rationale(self, handler):
        mock_h = _MockHTTPHandler(body={"task_description": "Design a REST API"})
        result = handler.handle_post("/api/v1/selection/team", {}, mock_h)
        body = _body(result)
        assert "rationale" in body
        assert len(body["rationale"]) > 0

    def test_response_has_plugins_used(self, handler):
        mock_h = _MockHTTPHandler(body={"task_description": "Design a REST API"})
        result = handler.handle_post("/api/v1/selection/team", {}, mock_h)
        body = _body(result)
        assert "plugins_used" in body
        plugins = body["plugins_used"]
        assert "scorer" in plugins
        assert "team_selector" in plugins
        assert "role_assigner" in plugins

    def test_missing_task_description_returns_400(self, handler):
        mock_h = _MockHTTPHandler(body={})
        result = handler.handle_post("/api/v1/selection/team", {}, mock_h)
        assert _status(result) == 400

    def test_empty_task_description_returns_400(self, handler):
        mock_h = _MockHTTPHandler(body={"task_description": ""})
        result = handler.handle_post("/api/v1/selection/team", {}, mock_h)
        assert _status(result) == 400

    def test_invalid_json_returns_400(self, handler):
        mock_h = _MockHTTPHandler()
        mock_h.request_body = b"{broken"
        result = handler.handle_post("/api/v1/selection/team", {}, mock_h)
        assert _status(result) == 400

    def test_custom_team_size(self, handler):
        mock_h = _MockHTTPHandler(
            body={
                "task_description": "Design a REST API",
                "min_agents": 3,
                "max_agents": 4,
            }
        )
        result = handler.handle_post("/api/v1/selection/team", {}, mock_h)
        body = _body(result)
        assert _status(result) == 200
        assert 3 <= len(body["agents"]) <= 4

    def test_exclude_agents(self, handler):
        mock_h = _MockHTTPHandler(
            body={
                "task_description": "Design a REST API",
                "exclude_agents": ["claude", "grok"],
            }
        )
        result = handler.handle_post("/api/v1/selection/team", {}, mock_h)
        body = _body(result)
        assert _status(result) == 200
        agent_names = [a["name"] for a in body["agents"]]
        assert "claude" not in agent_names
        assert "grok" not in agent_names

    def test_custom_quality_priority(self, handler):
        mock_h = _MockHTTPHandler(
            body={
                "task_description": "Design a REST API",
                "quality_priority": 0.9,
            }
        )
        result = handler.handle_post("/api/v1/selection/team", {}, mock_h)
        assert _status(result) == 200

    def test_custom_diversity_preference(self, handler):
        mock_h = _MockHTTPHandler(
            body={
                "task_description": "Design a REST API",
                "diversity_preference": 0.8,
            }
        )
        result = handler.handle_post("/api/v1/selection/team", {}, mock_h)
        assert _status(result) == 200

    def test_custom_primary_domain(self, handler):
        mock_h = _MockHTTPHandler(
            body={
                "task_description": "Build something",
                "primary_domain": "security",
            }
        )
        result = handler.handle_post("/api/v1/selection/team", {}, mock_h)
        assert _status(result) == 200

    def test_custom_secondary_domains(self, handler):
        mock_h = _MockHTTPHandler(
            body={
                "task_description": "Security review with testing focus",
                "secondary_domains": ["testing", "debugging"],
            }
        )
        result = handler.handle_post("/api/v1/selection/team", {}, mock_h)
        assert _status(result) == 200

    def test_custom_scorer_plugin(self, handler):
        mock_h = _MockHTTPHandler(
            body={
                "task_description": "Design a REST API",
                "scorer": "elo-weighted",
            }
        )
        result = handler.handle_post("/api/v1/selection/team", {}, mock_h)
        body = _body(result)
        assert body["plugins_used"]["scorer"] == "elo-weighted"

    def test_custom_team_selector_plugin(self, handler):
        mock_h = _MockHTTPHandler(
            body={
                "task_description": "Design a REST API",
                "team_selector": "greedy",
            }
        )
        result = handler.handle_post("/api/v1/selection/team", {}, mock_h)
        body = _body(result)
        assert body["plugins_used"]["team_selector"] == "greedy"

    def test_custom_role_assigner_plugin(self, handler):
        mock_h = _MockHTTPHandler(
            body={
                "task_description": "Design a REST API",
                "role_assigner": "simple",
            }
        )
        result = handler.handle_post("/api/v1/selection/team", {}, mock_h)
        body = _body(result)
        assert body["plugins_used"]["role_assigner"] == "simple"

    def test_invalid_scorer_returns_400(self, handler):
        mock_h = _MockHTTPHandler(
            body={
                "task_description": "Design a REST API",
                "scorer": "nonexistent-scorer",
            }
        )
        result = handler.handle_post("/api/v1/selection/team", {}, mock_h)
        assert _status(result) == 400

    def test_invalid_team_selector_returns_400(self, handler):
        mock_h = _MockHTTPHandler(
            body={
                "task_description": "Design a REST API",
                "team_selector": "nonexistent-selector",
            }
        )
        result = handler.handle_post("/api/v1/selection/team", {}, mock_h)
        assert _status(result) == 400

    def test_invalid_role_assigner_returns_400(self, handler):
        mock_h = _MockHTTPHandler(
            body={
                "task_description": "Design a REST API",
                "role_assigner": "nonexistent-assigner",
            }
        )
        result = handler.handle_post("/api/v1/selection/team", {}, mock_h)
        assert _status(result) == 400

    def test_sdk_alias_select_team(self, handler):
        mock_h = _MockHTTPHandler(body={"task_description": "Design a REST API"})
        result = handler.handle_post("/api/v1/agent-selection/select-team", {}, mock_h)
        assert _status(result) == 200

    def test_team_selection_alias(self, handler):
        mock_h = _MockHTTPHandler(body={"task_description": "Design a REST API"})
        result = handler.handle_post("/api/v1/team-selection", {}, mock_h)
        assert _status(result) == 200

    def test_diversity_score_range(self, handler):
        mock_h = _MockHTTPHandler(body={"task_description": "Design a REST API"})
        result = handler.handle_post("/api/v1/selection/team", {}, mock_h)
        body = _body(result)
        assert 0.0 <= body["diversity_score"] <= 1.0

    def test_expected_quality_range(self, handler):
        mock_h = _MockHTTPHandler(body={"task_description": "Design a REST API"})
        result = handler.handle_post("/api/v1/selection/team", {}, mock_h)
        body = _body(result)
        assert 0.0 <= body["expected_quality"] <= 1.0


# ============================================================================
# POST /api/v1/selection/assign-roles
# ============================================================================


class TestAssignRoles:
    """Tests for assigning roles to agents."""

    def test_returns_200(self, handler):
        mock_h = _MockHTTPHandler(
            body={
                "task_description": "Build a security tool",
                "agents": ["claude", "codex"],
            }
        )
        result = handler.handle_post("/api/v1/selection/assign-roles", {}, mock_h)
        assert _status(result) == 200

    def test_via_sdk_alias(self, handler):
        mock_h = _MockHTTPHandler(
            body={
                "task_description": "Build a security tool",
                "agents": ["claude", "codex"],
            }
        )
        result = handler.handle_post("/api/v1/agent-selection/assign-roles", {}, mock_h)
        assert _status(result) == 200

    def test_response_has_task_id(self, handler):
        mock_h = _MockHTTPHandler(
            body={
                "task_description": "Build a security tool",
                "agents": ["claude", "codex"],
            }
        )
        result = handler.handle_post("/api/v1/selection/assign-roles", {}, mock_h)
        body = _body(result)
        assert "task_id" in body
        assert body["task_id"].startswith("roles-")

    def test_response_has_assignments(self, handler):
        mock_h = _MockHTTPHandler(
            body={
                "task_description": "Build a security tool",
                "agents": ["claude", "codex"],
            }
        )
        result = handler.handle_post("/api/v1/selection/assign-roles", {}, mock_h)
        body = _body(result)
        assert "assignments" in body
        assert len(body["assignments"]) == 2

    def test_assignments_have_agent_and_role(self, handler):
        mock_h = _MockHTTPHandler(
            body={
                "task_description": "Build a security tool",
                "agents": ["claude", "codex"],
            }
        )
        result = handler.handle_post("/api/v1/selection/assign-roles", {}, mock_h)
        body = _body(result)
        for assignment in body["assignments"]:
            assert "agent" in assignment
            assert "role" in assignment

    def test_response_has_role_assigner(self, handler):
        mock_h = _MockHTTPHandler(
            body={
                "task_description": "Build a security tool",
                "agents": ["claude", "codex"],
            }
        )
        result = handler.handle_post("/api/v1/selection/assign-roles", {}, mock_h)
        body = _body(result)
        assert "role_assigner" in body

    def test_custom_role_assigner(self, handler):
        mock_h = _MockHTTPHandler(
            body={
                "task_description": "Build a security tool",
                "agents": ["claude", "codex"],
                "role_assigner": "simple",
            }
        )
        result = handler.handle_post("/api/v1/selection/assign-roles", {}, mock_h)
        body = _body(result)
        assert body["role_assigner"] == "simple"

    def test_missing_task_description_returns_400(self, handler):
        mock_h = _MockHTTPHandler(body={"agents": ["claude"]})
        result = handler.handle_post("/api/v1/selection/assign-roles", {}, mock_h)
        assert _status(result) == 400

    def test_missing_agents_returns_400(self, handler):
        mock_h = _MockHTTPHandler(body={"task_description": "Build a tool"})
        result = handler.handle_post("/api/v1/selection/assign-roles", {}, mock_h)
        assert _status(result) == 400

    def test_empty_agents_returns_400(self, handler):
        mock_h = _MockHTTPHandler(
            body={"task_description": "Build a tool", "agents": []}
        )
        result = handler.handle_post("/api/v1/selection/assign-roles", {}, mock_h)
        assert _status(result) == 400

    def test_invalid_agent_names_returns_400(self, handler):
        mock_h = _MockHTTPHandler(
            body={
                "task_description": "Build a tool",
                "agents": ["nonexistent_agent1", "nonexistent_agent2"],
            }
        )
        result = handler.handle_post("/api/v1/selection/assign-roles", {}, mock_h)
        assert _status(result) == 400

    def test_invalid_json_returns_400(self, handler):
        mock_h = _MockHTTPHandler()
        mock_h.request_body = b"not-json"
        result = handler.handle_post("/api/v1/selection/assign-roles", {}, mock_h)
        assert _status(result) == 400

    def test_invalid_role_assigner_returns_400(self, handler):
        mock_h = _MockHTTPHandler(
            body={
                "task_description": "Build a security tool",
                "agents": ["claude", "codex"],
                "role_assigner": "nonexistent",
            }
        )
        result = handler.handle_post("/api/v1/selection/assign-roles", {}, mock_h)
        assert _status(result) == 400

    def test_mixed_valid_invalid_agents(self, handler):
        """Only valid agents get roles; if at least one valid, succeeds."""
        mock_h = _MockHTTPHandler(
            body={
                "task_description": "Build a tool",
                "agents": ["claude", "nonexistent_agent"],
            }
        )
        result = handler.handle_post("/api/v1/selection/assign-roles", {}, mock_h)
        assert _status(result) == 200
        body = _body(result)
        agent_names = [a["agent"] for a in body["assignments"]]
        assert "claude" in agent_names

    def test_custom_primary_domain(self, handler):
        mock_h = _MockHTTPHandler(
            body={
                "task_description": "Build a tool",
                "agents": ["claude", "codex"],
                "primary_domain": "security",
            }
        )
        result = handler.handle_post("/api/v1/selection/assign-roles", {}, mock_h)
        assert _status(result) == 200

    def test_all_known_agents(self, handler):
        mock_h = _MockHTTPHandler(
            body={
                "task_description": "Build a full-stack app",
                "agents": ["claude", "codex", "gemini", "grok", "deepseek"],
            }
        )
        result = handler.handle_post("/api/v1/selection/assign-roles", {}, mock_h)
        body = _body(result)
        assert len(body["assignments"]) == 5


# ============================================================================
# Rate Limiting
# ============================================================================


class TestRateLimiting:
    """Tests for rate limiting on selection endpoints."""

    def test_rate_limit_returns_429(self, handler):
        """Exceeding rate limit returns 429."""
        mock_h = _MockHTTPHandler()
        # Patch is_allowed to return False
        with patch.object(_selection_limiter, "is_allowed", return_value=False):
            result = handler.handle("/api/v1/selection/plugins", {}, mock_h)
            assert _status(result) == 429

    def test_rate_limit_error_message(self, handler):
        mock_h = _MockHTTPHandler()
        with patch.object(_selection_limiter, "is_allowed", return_value=False):
            result = handler.handle("/api/v1/selection/plugins", {}, mock_h)
            body = _body(result)
            assert "rate limit" in body.get("error", "").lower()

    def test_allowed_request_succeeds(self, handler):
        """Normal requests pass rate limiting."""
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/selection/plugins", {}, mock_h)
        assert _status(result) == 200


# ============================================================================
# Unmatched Routes
# ============================================================================


class TestUnmatchedRoutes:
    """Tests for unmatched route handling."""

    def test_handle_returns_none_for_unmatched_get(self, handler):
        mock_h = _MockHTTPHandler()
        result = handler.handle("/api/v1/other/endpoint", {}, mock_h)
        assert result is None

    def test_handle_post_returns_none_for_unmatched(self, handler):
        mock_h = _MockHTTPHandler(body={"task_description": "test"})
        result = handler.handle_post("/api/v1/other/endpoint", {}, mock_h)
        assert result is None


# ============================================================================
# _create_agent_pool Helper
# ============================================================================


class TestCreateAgentPool:
    """Tests for the _create_agent_pool helper function."""

    def test_returns_dict(self):
        pool = _create_agent_pool()
        assert isinstance(pool, dict)

    def test_contains_known_agents(self):
        pool = _create_agent_pool()
        assert "claude" in pool
        assert "codex" in pool
        assert "gemini" in pool
        assert "grok" in pool
        assert "deepseek" in pool

    def test_agents_have_name(self):
        pool = _create_agent_pool()
        for name, profile in pool.items():
            assert profile.name == name

    def test_agents_have_agent_type(self):
        pool = _create_agent_pool()
        for name, profile in pool.items():
            assert profile.agent_type == name

    def test_agents_have_expertise(self):
        pool = _create_agent_pool()
        for profile in pool.values():
            assert len(profile.expertise) > 0


# ============================================================================
# get_team_builder Helper
# ============================================================================


class TestGetTeamBuilder:
    """Tests for the get_team_builder helper function."""

    def test_returns_team_builder(self):
        from aragora.routing.team_builder import TeamBuilder
        tb = get_team_builder()
        assert isinstance(tb, TeamBuilder)

    def test_returns_singleton(self):
        tb1 = get_team_builder()
        tb2 = get_team_builder()
        assert tb1 is tb2


# ============================================================================
# Error Handling
# ============================================================================


class TestErrorHandling:
    """Tests for error handling edge cases."""

    def test_handler_without_request_body_attribute(self, handler):
        """Handler without request_body returns empty dict from _get_json_body."""
        mock_h = MagicMock()
        mock_h.client_address = ("127.0.0.1", 12345)
        del mock_h.request_body  # Simulate missing attribute
        # For score endpoint, missing task_description should return 400
        result = handler.handle_post("/api/v1/selection/score", {}, mock_h)
        assert _status(result) == 400

    def test_score_with_attribute_error(self, handler):
        """AttributeError during body parsing returns 400."""
        mock_h = MagicMock()
        mock_h.client_address = ("127.0.0.1", 12345)
        mock_h.request_body = None  # Will cause TypeError in decode
        result = handler.handle_post("/api/v1/selection/score", {}, mock_h)
        assert _status(result) == 400

    def test_select_team_with_attribute_error(self, handler):
        """AttributeError during body parsing in select_team returns 400."""
        mock_h = MagicMock()
        mock_h.client_address = ("127.0.0.1", 12345)
        mock_h.request_body = None
        result = handler.handle_post("/api/v1/selection/team", {}, mock_h)
        assert _status(result) == 400
