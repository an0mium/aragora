"""
Tests for SelectionHandler endpoints.

Endpoints tested:
- GET  /api/selection/plugins               List all available plugins
- GET  /api/selection/defaults              Get default plugin configuration
- GET  /api/selection/scorers/<name>        Get scorer details
- GET  /api/selection/team-selectors/<name> Get team selector details
- GET  /api/selection/role-assigners/<name> Get role assigner details
- POST /api/selection/score                 Score agents for a task
- POST /api/selection/team                  Select a team for a task
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from aragora.server.handlers.selection import SelectionHandler
from aragora.server.handlers.base import clear_cache


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def selection_handler():
    """Create a SelectionHandler with minimal context."""
    ctx = {
        "storage": None,
        "elo_system": None,
    }
    return SelectionHandler(ctx)


@pytest.fixture
def mock_handler():
    """Create a mock HTTP handler for POST requests."""
    handler = Mock()
    handler.headers = {"Content-Type": "application/json"}
    handler.client_address = ("127.0.0.1", 12345)
    return handler


@pytest.fixture
def mock_handler_with_body(mock_handler):
    """Factory to create mock handler with JSON body."""

    def create_with_body(body_dict):
        body_bytes = json.dumps(body_dict).encode("utf-8")
        mock_handler.request_body = body_bytes
        return mock_handler

    return create_with_body


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches before and after each test."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter between tests."""
    from aragora.server.handlers.selection import _selection_limiter

    _selection_limiter.clear()
    yield
    _selection_limiter.clear()


# ============================================================================
# Route Matching Tests
# ============================================================================


class TestSelectionRouting:
    """Tests for route matching."""

    def test_can_handle_plugins_list(self, selection_handler):
        """Handler can handle /api/selection/plugins."""
        assert selection_handler.can_handle("/api/selection/plugins") is True

    def test_can_handle_defaults(self, selection_handler):
        """Handler can handle /api/selection/defaults."""
        assert selection_handler.can_handle("/api/selection/defaults") is True

    def test_can_handle_score(self, selection_handler):
        """Handler can handle /api/selection/score."""
        assert selection_handler.can_handle("/api/selection/score") is True

    def test_can_handle_team(self, selection_handler):
        """Handler can handle /api/selection/team."""
        assert selection_handler.can_handle("/api/selection/team") is True

    def test_can_handle_scorer_detail(self, selection_handler):
        """Handler can handle /api/selection/scorers/<name>."""
        assert selection_handler.can_handle("/api/selection/scorers/weighted") is True
        assert selection_handler.can_handle("/api/selection/scorers/elo-based") is True

    def test_can_handle_team_selector_detail(self, selection_handler):
        """Handler can handle /api/selection/team-selectors/<name>."""
        assert selection_handler.can_handle("/api/selection/team-selectors/top-k") is True
        assert selection_handler.can_handle("/api/selection/team-selectors/diverse") is True

    def test_can_handle_role_assigner_detail(self, selection_handler):
        """Handler can handle /api/selection/role-assigners/<name>."""
        assert selection_handler.can_handle("/api/selection/role-assigners/round-robin") is True

    def test_cannot_handle_unrelated_routes(self, selection_handler):
        """Handler doesn't handle unrelated routes."""
        assert selection_handler.can_handle("/api/debates") is False
        assert selection_handler.can_handle("/api/agents") is False
        assert selection_handler.can_handle("/api/selection") is False
        assert selection_handler.can_handle("/api/plugins") is False


# ============================================================================
# GET /api/selection/plugins Tests
# ============================================================================


class TestListPlugins:
    """Tests for GET /api/selection/plugins endpoint."""

    def test_list_plugins_success(self, selection_handler, mock_handler):
        """Returns list of available plugins."""
        result = selection_handler.handle("/api/selection/plugins", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)

        # Should have scorers, team_selectors, and role_assigners
        assert "scorers" in data
        assert "team_selectors" in data
        assert "role_assigners" in data

        # Each should be a list
        assert isinstance(data["scorers"], list)
        assert isinstance(data["team_selectors"], list)
        assert isinstance(data["role_assigners"], list)


# ============================================================================
# GET /api/selection/defaults Tests
# ============================================================================


class TestGetDefaults:
    """Tests for GET /api/selection/defaults endpoint."""

    def test_get_defaults_success(self, selection_handler, mock_handler):
        """Returns default plugin configuration."""
        result = selection_handler.handle("/api/selection/defaults", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)

        # Should have default scorer, team_selector, and role_assigner
        assert "scorer" in data
        assert "team_selector" in data
        assert "role_assigner" in data

        # Each should be a string (plugin name)
        assert isinstance(data["scorer"], str)
        assert isinstance(data["team_selector"], str)
        assert isinstance(data["role_assigner"], str)


# ============================================================================
# GET /api/selection/scorers/<name> Tests
# ============================================================================


class TestGetScorer:
    """Tests for GET /api/selection/scorers/<name> endpoint."""

    def test_get_scorer_success(self, selection_handler, mock_handler):
        """Returns scorer info for valid scorer name."""
        # Get defaults to find a valid scorer name
        defaults_result = selection_handler.handle("/api/selection/defaults", {}, mock_handler)
        defaults = json.loads(defaults_result.body)
        scorer_name = defaults["scorer"]

        result = selection_handler.handle(f"/api/selection/scorers/{scorer_name}", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "name" in data

    def test_get_scorer_not_found(self, selection_handler, mock_handler):
        """Returns 404 for unknown scorer."""
        result = selection_handler.handle(
            "/api/selection/scorers/nonexistent-scorer", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 404
        data = json.loads(result.body)
        assert "error" in data
        assert "nonexistent-scorer" in data["error"].lower() or "unknown" in data["error"].lower()


# ============================================================================
# GET /api/selection/team-selectors/<name> Tests
# ============================================================================


class TestGetTeamSelector:
    """Tests for GET /api/selection/team-selectors/<name> endpoint."""

    def test_get_team_selector_success(self, selection_handler, mock_handler):
        """Returns team selector info for valid name."""
        # Get defaults to find a valid team selector name
        defaults_result = selection_handler.handle("/api/selection/defaults", {}, mock_handler)
        defaults = json.loads(defaults_result.body)
        selector_name = defaults["team_selector"]

        result = selection_handler.handle(
            f"/api/selection/team-selectors/{selector_name}", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "name" in data

    def test_get_team_selector_not_found(self, selection_handler, mock_handler):
        """Returns 404 for unknown team selector."""
        result = selection_handler.handle(
            "/api/selection/team-selectors/nonexistent", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 404
        data = json.loads(result.body)
        assert "error" in data


# ============================================================================
# GET /api/selection/role-assigners/<name> Tests
# ============================================================================


class TestGetRoleAssigner:
    """Tests for GET /api/selection/role-assigners/<name> endpoint."""

    def test_get_role_assigner_success(self, selection_handler, mock_handler):
        """Returns role assigner info for valid name."""
        # Get defaults to find a valid role assigner name
        defaults_result = selection_handler.handle("/api/selection/defaults", {}, mock_handler)
        defaults = json.loads(defaults_result.body)
        assigner_name = defaults["role_assigner"]

        result = selection_handler.handle(
            f"/api/selection/role-assigners/{assigner_name}", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "name" in data

    def test_get_role_assigner_not_found(self, selection_handler, mock_handler):
        """Returns 404 for unknown role assigner."""
        result = selection_handler.handle(
            "/api/selection/role-assigners/nonexistent", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 404
        data = json.loads(result.body)
        assert "error" in data


# ============================================================================
# POST /api/selection/score Tests
# ============================================================================


class TestScoreAgents:
    """Tests for POST /api/selection/score endpoint."""

    def test_score_agents_success(self, selection_handler, mock_handler_with_body):
        """Successfully scores agents for a task."""
        handler = mock_handler_with_body(
            {"task_description": "Build a REST API with authentication"}
        )

        result = selection_handler.handle_post("/api/selection/score", {}, handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)

        assert "scorer_used" in data
        assert "agents" in data
        assert "task_id" in data
        assert isinstance(data["agents"], list)

        # Each agent should have required fields
        if data["agents"]:
            agent = data["agents"][0]
            assert "name" in agent
            assert "score" in agent
            assert isinstance(agent["score"], (int, float))

    def test_score_agents_with_domains(self, selection_handler, mock_handler_with_body):
        """Scores agents with explicit domain specification."""
        handler = mock_handler_with_body(
            {
                "task_description": "Create machine learning pipeline",
                "primary_domain": "ml",
                "secondary_domains": ["python", "data"],
                "required_traits": ["research"],
            }
        )

        result = selection_handler.handle_post("/api/selection/score", {}, handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "agents" in data

    def test_score_agents_missing_task(self, selection_handler, mock_handler_with_body):
        """Returns 400 when task_description is missing."""
        handler = mock_handler_with_body({})

        result = selection_handler.handle_post("/api/selection/score", {}, handler)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "error" in data
        assert "task_description" in data["error"].lower()

    def test_score_agents_invalid_json(self, selection_handler, mock_handler):
        """Returns 400 for invalid JSON body."""
        mock_handler.request_body = b"not valid json"

        result = selection_handler.handle_post("/api/selection/score", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "error" in data

    def test_score_agents_invalid_scorer(self, selection_handler, mock_handler_with_body):
        """Returns 400 for unknown scorer."""
        handler = mock_handler_with_body(
            {"task_description": "Test task", "scorer": "nonexistent-scorer"}
        )

        result = selection_handler.handle_post("/api/selection/score", {}, handler)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "error" in data


# ============================================================================
# POST /api/selection/team Tests
# ============================================================================


class TestSelectTeam:
    """Tests for POST /api/selection/team endpoint."""

    def test_select_team_success(self, selection_handler, mock_handler_with_body):
        """Successfully selects a team for a task."""
        handler = mock_handler_with_body(
            {"task_description": "Design a distributed system for real-time analytics"}
        )

        result = selection_handler.handle_post("/api/selection/team", {}, handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)

        assert "team_id" in data
        assert "task_id" in data
        assert "agents" in data
        assert "expected_quality" in data
        assert "expected_cost" in data
        assert "diversity_score" in data
        assert "rationale" in data
        assert "plugins_used" in data

        # Agents should be a list
        assert isinstance(data["agents"], list)

        # Each team member should have required fields
        if data["agents"]:
            member = data["agents"][0]
            assert "name" in member
            assert "role" in member
            assert "score" in member

    def test_select_team_with_constraints(self, selection_handler, mock_handler_with_body):
        """Selects team with constraints."""
        handler = mock_handler_with_body(
            {
                "task_description": "Optimize database queries",
                "min_agents": 2,
                "max_agents": 4,
                "quality_priority": 0.8,
                "diversity_preference": 0.3,
                "exclude_agents": ["gpt4"],
            }
        )

        result = selection_handler.handle_post("/api/selection/team", {}, handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)

        # Team size should respect constraints
        assert len(data["agents"]) >= 2
        assert len(data["agents"]) <= 4

        # Excluded agents should not be in team
        agent_names = [a["name"] for a in data["agents"]]
        assert "gpt4" not in agent_names

    def test_select_team_missing_task(self, selection_handler, mock_handler_with_body):
        """Returns 400 when task_description is missing."""
        handler = mock_handler_with_body({"min_agents": 3})

        result = selection_handler.handle_post("/api/selection/team", {}, handler)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "error" in data
        assert "task_description" in data["error"].lower()

    def test_select_team_invalid_json(self, selection_handler, mock_handler):
        """Returns 400 for invalid JSON body."""
        mock_handler.request_body = b"{invalid json"

        result = selection_handler.handle_post("/api/selection/team", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "error" in data

    def test_select_team_invalid_plugin(self, selection_handler, mock_handler_with_body):
        """Returns 400 for unknown plugin."""
        handler = mock_handler_with_body(
            {"task_description": "Test task", "team_selector": "nonexistent-selector"}
        )

        result = selection_handler.handle_post("/api/selection/team", {}, handler)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "error" in data


# ============================================================================
# Rate Limiting Tests
# ============================================================================


class TestRateLimiting:
    """Tests for rate limiting on selection endpoints."""

    def test_rate_limit_exceeded(self, selection_handler, mock_handler):
        """Returns 429 when rate limit is exceeded."""
        from aragora.server.handlers.selection import _selection_limiter

        # Fill up the rate limiter
        client_ip = "127.0.0.1"
        for _ in range(101):  # Exceed 100 requests per minute limit
            _selection_limiter.is_allowed(client_ip)

        result = selection_handler.handle("/api/selection/plugins", {}, mock_handler)

        assert result is not None
        assert result.status_code == 429
        data = json.loads(result.body)
        assert "error" in data
        assert "rate limit" in data["error"].lower()


# ============================================================================
# Handler Routing Tests
# ============================================================================


class TestHandlerRouting:
    """Tests for request routing."""

    def test_handle_returns_none_for_unhandled(self, selection_handler, mock_handler):
        """Returns None for unhandled GET routes."""
        result = selection_handler.handle("/api/other/endpoint", {}, mock_handler)
        assert result is None

    def test_handle_post_returns_none_for_unhandled(self, selection_handler, mock_handler):
        """Returns None for unhandled POST routes."""
        result = selection_handler.handle_post("/api/other/endpoint", {}, mock_handler)
        assert result is None

    def test_handle_routes_to_list_plugins(self, selection_handler, mock_handler):
        """GET /api/selection/plugins routes correctly."""
        result = selection_handler.handle("/api/selection/plugins", {}, mock_handler)
        assert result is not None
        assert result.status_code == 200

    def test_handle_routes_to_defaults(self, selection_handler, mock_handler):
        """GET /api/selection/defaults routes correctly."""
        result = selection_handler.handle("/api/selection/defaults", {}, mock_handler)
        assert result is not None
        assert result.status_code == 200

    def test_handle_post_routes_to_score(self, selection_handler, mock_handler_with_body):
        """POST /api/selection/score routes correctly."""
        handler = mock_handler_with_body({"task_description": "Test"})
        result = selection_handler.handle_post("/api/selection/score", {}, handler)
        assert result is not None

    def test_handle_post_routes_to_team(self, selection_handler, mock_handler_with_body):
        """POST /api/selection/team routes correctly."""
        handler = mock_handler_with_body({"task_description": "Test"})
        result = selection_handler.handle_post("/api/selection/team", {}, handler)
        assert result is not None


# ============================================================================
# Handler Import Tests
# ============================================================================


class TestSelectionHandlerImport:
    """Test SelectionHandler import and export."""

    def test_handler_importable(self):
        """SelectionHandler can be imported from handlers package."""
        from aragora.server.handlers import SelectionHandler

        assert SelectionHandler is not None

    def test_handler_in_all_handlers(self):
        """SelectionHandler is in ALL_HANDLERS registry."""
        from aragora.server.handlers import ALL_HANDLERS, SelectionHandler

        assert SelectionHandler in ALL_HANDLERS
