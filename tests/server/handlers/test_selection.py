"""
Tests for the selection handler - agent selection plugin API.

Tests:
- Route handling (can_handle)
- List plugins endpoint
- Get defaults endpoint
- Get specific plugin info (scorers, team selectors, role assigners)
- Score agents endpoint
- Select team endpoint
- Rate limiting
- Error handling
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from aragora.server.handlers.selection import SelectionHandler, _selection_limiter


@pytest.fixture
def selection_handler():
    """Create a selection handler with mocked dependencies."""
    ctx = {"storage": None, "elo_system": None}
    handler = SelectionHandler(ctx)
    return handler


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler for POST requests."""
    mock = MagicMock()
    mock.client_address = ("127.0.0.1", 12345)
    mock.headers = {"Content-Type": "application/json"}
    return mock


def make_post_handler(body: dict) -> MagicMock:
    """Create a mock handler with a JSON body."""
    mock = MagicMock()
    mock.client_address = ("127.0.0.1", 12345)
    body_bytes = json.dumps(body).encode()
    mock.headers = {"Content-Type": "application/json", "Content-Length": str(len(body_bytes))}
    mock.request_body = body_bytes
    mock.rfile = MagicMock()
    mock.rfile.read.return_value = body_bytes
    return mock


class TestSelectionHandlerRouting:
    """Tests for SelectionHandler route matching."""

    def test_can_handle_plugins_list(self, selection_handler):
        """Test that handler recognizes /api/selection/plugins route."""
        assert selection_handler.can_handle("/api/v1/selection/plugins") is True

    def test_can_handle_defaults(self, selection_handler):
        """Test that handler recognizes /api/selection/defaults route."""
        assert selection_handler.can_handle("/api/v1/selection/defaults") is True

    def test_can_handle_score(self, selection_handler):
        """Test that handler recognizes /api/selection/score route."""
        assert selection_handler.can_handle("/api/v1/selection/score") is True

    def test_can_handle_team(self, selection_handler):
        """Test that handler recognizes /api/selection/team route."""
        assert selection_handler.can_handle("/api/v1/selection/team") is True

    def test_can_handle_scorers_prefix(self, selection_handler):
        """Test that handler recognizes /api/selection/scorers/<name> routes."""
        assert selection_handler.can_handle("/api/v1/selection/scorers/default") is True
        assert selection_handler.can_handle("/api/v1/selection/scorers/composite") is True

    def test_can_handle_team_selectors_prefix(self, selection_handler):
        """Test that handler recognizes /api/selection/team-selectors/<name> routes."""
        assert selection_handler.can_handle("/api/v1/selection/team-selectors/default") is True
        assert selection_handler.can_handle("/api/v1/selection/team-selectors/diverse") is True

    def test_can_handle_role_assigners_prefix(self, selection_handler):
        """Test that handler recognizes /api/selection/role-assigners/<name> routes."""
        assert selection_handler.can_handle("/api/v1/selection/role-assigners/default") is True
        assert selection_handler.can_handle("/api/v1/selection/role-assigners/adaptive") is True

    def test_cannot_handle_unknown_path(self, selection_handler):
        """Test that handler rejects unknown paths."""
        assert selection_handler.can_handle("/api/v1/unknown") is False
        assert selection_handler.can_handle("/api/v1/debates") is False
        assert selection_handler.can_handle("/api/v1/selection") is False  # No trailing /


class TestListPlugins:
    """Tests for /api/selection/plugins endpoint."""

    def test_list_plugins_returns_all_categories(self, selection_handler, mock_http_handler):
        """List plugins should return scorers, team_selectors, and role_assigners."""
        # Reset rate limiter for clean test
        _selection_limiter._buckets.clear()

        result = selection_handler.handle("/api/v1/selection/plugins", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "scorers" in body
        assert "team_selectors" in body
        assert "role_assigners" in body

    def test_list_plugins_includes_default_plugins(self, selection_handler, mock_http_handler):
        """List plugins should include built-in default plugins."""
        _selection_limiter._buckets.clear()

        result = selection_handler.handle("/api/v1/selection/plugins", {}, mock_http_handler)

        body = json.loads(result.body)
        # Check that at least default plugins are present
        assert len(body["scorers"]) >= 1
        assert len(body["team_selectors"]) >= 1
        assert len(body["role_assigners"]) >= 1


class TestGetDefaults:
    """Tests for /api/selection/defaults endpoint."""

    def test_get_defaults_returns_default_plugins(self, selection_handler, mock_http_handler):
        """Get defaults should return the default plugin for each category."""
        _selection_limiter._buckets.clear()

        result = selection_handler.handle("/api/v1/selection/defaults", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "scorer" in body
        assert "team_selector" in body
        assert "role_assigner" in body


class TestGetScorerInfo:
    """Tests for /api/selection/scorers/<name> endpoint."""

    def test_get_scorer_info_valid(self, selection_handler, mock_http_handler):
        """Get scorer info for existing scorer should return details."""
        _selection_limiter._buckets.clear()

        # Get the default scorer name first
        defaults_result = selection_handler.handle(
            "/api/v1/selection/defaults", {}, mock_http_handler
        )
        defaults = json.loads(defaults_result.body)
        scorer_name = defaults["scorer"]

        result = selection_handler.handle(
            f"/api/v1/selection/scorers/{scorer_name}", {}, mock_http_handler
        )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "name" in body

    def test_get_scorer_info_not_found(self, selection_handler, mock_http_handler):
        """Get scorer info for unknown scorer should return 404."""
        _selection_limiter._buckets.clear()

        result = selection_handler.handle(
            "/api/v1/selection/scorers/nonexistent_scorer", {}, mock_http_handler
        )

        assert result is not None
        assert result.status_code == 404
        body = json.loads(result.body)
        assert "error" in body


class TestGetTeamSelectorInfo:
    """Tests for /api/selection/team-selectors/<name> endpoint."""

    def test_get_team_selector_info_valid(self, selection_handler, mock_http_handler):
        """Get team selector info for existing selector should return details."""
        _selection_limiter._buckets.clear()

        defaults_result = selection_handler.handle(
            "/api/v1/selection/defaults", {}, mock_http_handler
        )
        defaults = json.loads(defaults_result.body)
        selector_name = defaults["team_selector"]

        result = selection_handler.handle(
            f"/api/v1/selection/team-selectors/{selector_name}", {}, mock_http_handler
        )

        assert result is not None
        assert result.status_code == 200

    def test_get_team_selector_info_not_found(self, selection_handler, mock_http_handler):
        """Get team selector info for unknown selector should return 404."""
        _selection_limiter._buckets.clear()

        result = selection_handler.handle(
            "/api/v1/selection/team-selectors/nonexistent", {}, mock_http_handler
        )

        assert result is not None
        assert result.status_code == 404


class TestGetRoleAssignerInfo:
    """Tests for /api/selection/role-assigners/<name> endpoint."""

    def test_get_role_assigner_info_valid(self, selection_handler, mock_http_handler):
        """Get role assigner info for existing assigner should return details."""
        _selection_limiter._buckets.clear()

        defaults_result = selection_handler.handle(
            "/api/v1/selection/defaults", {}, mock_http_handler
        )
        defaults = json.loads(defaults_result.body)
        assigner_name = defaults["role_assigner"]

        result = selection_handler.handle(
            f"/api/v1/selection/role-assigners/{assigner_name}", {}, mock_http_handler
        )

        assert result is not None
        assert result.status_code == 200

    def test_get_role_assigner_info_not_found(self, selection_handler, mock_http_handler):
        """Get role assigner info for unknown assigner should return 404."""
        _selection_limiter._buckets.clear()

        result = selection_handler.handle(
            "/api/v1/selection/role-assigners/nonexistent", {}, mock_http_handler
        )

        assert result is not None
        assert result.status_code == 404


class TestScoreAgents:
    """Tests for POST /api/selection/score endpoint."""

    def test_score_agents_success(self, selection_handler):
        """Score agents with valid request should return scored list."""
        _selection_limiter._buckets.clear()

        handler = make_post_handler(
            {"task_description": "Implement a new feature for user authentication"}
        )

        result = selection_handler.handle_post("/api/v1/selection/score", {}, handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "agents" in body
        assert "scorer_used" in body
        assert "task_id" in body
        assert len(body["agents"]) > 0
        # Check agent structure
        agent = body["agents"][0]
        assert "name" in agent
        assert "score" in agent

    def test_score_agents_with_custom_domain(self, selection_handler):
        """Score agents with explicit domain should use that domain."""
        _selection_limiter._buckets.clear()

        handler = make_post_handler(
            {
                "task_description": "Build a REST API",
                "primary_domain": "software_engineering",
                "secondary_domains": ["architecture"],
                "required_traits": ["analytical"],
            }
        )

        result = selection_handler.handle_post("/api/v1/selection/score", {}, handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert len(body["agents"]) > 0

    def test_score_agents_missing_task(self, selection_handler):
        """Score agents without task_description should return 400."""
        _selection_limiter._buckets.clear()

        handler = make_post_handler({})

        result = selection_handler.handle_post("/api/v1/selection/score", {}, handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body

    def test_score_agents_invalid_scorer(self, selection_handler):
        """Score agents with invalid scorer name should return 400."""
        _selection_limiter._buckets.clear()

        handler = make_post_handler(
            {"task_description": "Test task", "scorer": "nonexistent_scorer"}
        )

        result = selection_handler.handle_post("/api/v1/selection/score", {}, handler)

        assert result is not None
        assert result.status_code == 400


class TestSelectTeam:
    """Tests for POST /api/selection/team endpoint."""

    def test_select_team_success(self, selection_handler):
        """Select team with valid request should return team."""
        _selection_limiter._buckets.clear()

        handler = make_post_handler(
            {"task_description": "Design a distributed system", "team_size": 3}
        )

        result = selection_handler.handle_post("/api/v1/selection/team", {}, handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "team" in body or "selected_agents" in body or "agents" in body

    def test_select_team_missing_task(self, selection_handler):
        """Select team without task_description should return 400."""
        _selection_limiter._buckets.clear()

        handler = make_post_handler({"team_size": 3})

        result = selection_handler.handle_post("/api/v1/selection/team", {}, handler)

        assert result is not None
        assert result.status_code == 400


class TestRateLimiting:
    """Tests for rate limiting on selection endpoints."""

    def test_rate_limit_exceeded(self, selection_handler, mock_http_handler):
        """Exceeding rate limit should return 429."""
        # Fill up the rate limiter
        _selection_limiter._buckets.clear()

        # Simulate many requests from same IP
        for _ in range(105):  # Over the 100 limit
            _selection_limiter.is_allowed("127.0.0.1")

        result = selection_handler.handle("/api/v1/selection/plugins", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 429
        body = json.loads(result.body)
        assert "Rate limit" in body.get("error", "")


class TestHandleReturnsNone:
    """Tests for handle returning None for non-matching paths."""

    def test_handle_returns_none_for_non_matching_get(self, selection_handler, mock_http_handler):
        """Handle should return None for non-matching GET paths."""
        _selection_limiter._buckets.clear()

        result = selection_handler.handle("/api/v1/other", {}, mock_http_handler)
        assert result is None

    def test_handle_post_returns_none_for_non_matching(self, selection_handler, mock_http_handler):
        """Handle_post should return None for non-matching POST paths."""
        result = selection_handler.handle_post("/api/v1/other", {}, mock_http_handler)
        assert result is None
