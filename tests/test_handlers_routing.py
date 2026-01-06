"""
Tests for RoutingHandler endpoints.

Endpoints tested:
- GET /api/routing/best-teams - Get best-performing team combinations
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch

from aragora.server.handlers.routing import RoutingHandler
from aragora.server.handlers.base import clear_cache


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_agent_selector():
    """Create a mock AgentSelector."""
    selector = Mock()
    selector.get_best_team_combinations.return_value = [
        {
            "agents": ["claude", "gpt4"],
            "wins": 15,
            "losses": 5,
            "draws": 2,
            "win_rate": 0.68,
            "total_debates": 22,
        },
        {
            "agents": ["gemini", "gpt4"],
            "wins": 12,
            "losses": 6,
            "draws": 3,
            "win_rate": 0.57,
            "total_debates": 21,
        },
    ]
    return selector


@pytest.fixture
def mock_elo_system():
    """Create a mock ELO system."""
    return Mock()


@pytest.fixture
def routing_handler(mock_elo_system):
    """Create a RoutingHandler with mock dependencies."""
    ctx = {
        "storage": None,
        "elo_system": mock_elo_system,
        "persona_manager": None,
        "nomic_dir": None,
    }
    return RoutingHandler(ctx)


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches before and after each test."""
    clear_cache()
    yield
    clear_cache()


# ============================================================================
# Route Matching Tests
# ============================================================================

class TestRoutingRouting:
    """Tests for route matching."""

    def test_can_handle_best_teams(self, routing_handler):
        assert routing_handler.can_handle("/api/routing/best-teams") is True

    def test_can_handle_recommendations(self, routing_handler):
        # POST endpoint now handled by this handler
        assert routing_handler.can_handle("/api/routing/recommendations") is True

    def test_cannot_handle_unrelated_routes(self, routing_handler):
        assert routing_handler.can_handle("/api/routing") is False
        assert routing_handler.can_handle("/api/agents") is False
        assert routing_handler.can_handle("/api/routing/other") is False


# ============================================================================
# GET /api/routing/best-teams Tests
# ============================================================================

class TestBestTeams:
    """Tests for GET /api/routing/best-teams endpoint."""

    def test_best_teams_module_unavailable(self, routing_handler):
        import aragora.server.handlers.routing as mod
        original = mod.ROUTING_AVAILABLE
        mod.ROUTING_AVAILABLE = False
        try:
            result = routing_handler.handle("/api/routing/best-teams", {}, None)
            assert result is not None
            assert result.status_code == 503
            data = json.loads(result.body)
            assert "not available" in data["error"].lower()
        finally:
            mod.ROUTING_AVAILABLE = original

    def test_best_teams_success(self, routing_handler, mock_agent_selector):
        import aragora.server.handlers.routing as mod

        if not mod.ROUTING_AVAILABLE:
            pytest.skip("Routing module not available")

        with patch.object(mod, 'AgentSelector', return_value=mock_agent_selector):
            result = routing_handler.handle("/api/routing/best-teams", {}, None)

            assert result is not None
            assert result.status_code == 200
            data = json.loads(result.body)
            assert "combinations" in data
            assert data["count"] == 2
            assert len(data["combinations"]) == 2

    def test_best_teams_with_min_debates(self, routing_handler, mock_agent_selector):
        import aragora.server.handlers.routing as mod

        if not mod.ROUTING_AVAILABLE:
            pytest.skip("Routing module not available")

        with patch.object(mod, 'AgentSelector', return_value=mock_agent_selector):
            result = routing_handler.handle("/api/routing/best-teams", {"min_debates": "5"}, None)

            assert result is not None
            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["min_debates"] == 5

    def test_best_teams_min_debates_clamped(self, routing_handler, mock_agent_selector):
        import aragora.server.handlers.routing as mod

        if not mod.ROUTING_AVAILABLE:
            pytest.skip("Routing module not available")

        with patch.object(mod, 'AgentSelector', return_value=mock_agent_selector):
            # Request 0, should be clamped to 1
            result = routing_handler.handle("/api/routing/best-teams", {"min_debates": "0"}, None)

            assert result is not None
            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["min_debates"] == 1

    def test_best_teams_with_limit(self, routing_handler, mock_agent_selector):
        import aragora.server.handlers.routing as mod

        if not mod.ROUTING_AVAILABLE:
            pytest.skip("Routing module not available")

        with patch.object(mod, 'AgentSelector', return_value=mock_agent_selector):
            result = routing_handler.handle("/api/routing/best-teams", {"limit": "5"}, None)

            assert result is not None
            assert result.status_code == 200

    def test_best_teams_limit_capped(self, routing_handler, mock_agent_selector):
        import aragora.server.handlers.routing as mod

        if not mod.ROUTING_AVAILABLE:
            pytest.skip("Routing module not available")

        with patch.object(mod, 'AgentSelector', return_value=mock_agent_selector):
            # Request 100, should be capped at 50
            result = routing_handler.handle("/api/routing/best-teams", {"limit": "100"}, None)

            assert result is not None
            assert result.status_code == 200


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestRoutingErrorHandling:
    """Tests for error handling."""

    def test_handle_returns_none_for_unhandled_route(self, routing_handler):
        result = routing_handler.handle("/api/other/endpoint", {}, None)
        assert result is None

    def test_best_teams_exception(self, routing_handler):
        import aragora.server.handlers.routing as mod

        if not mod.ROUTING_AVAILABLE:
            pytest.skip("Routing module not available")

        mock_selector = Mock()
        mock_selector.get_best_team_combinations.side_effect = Exception("Database error")

        with patch.object(mod, 'AgentSelector', return_value=mock_selector):
            result = routing_handler.handle("/api/routing/best-teams", {}, None)

            assert result is not None
            assert result.status_code == 500


# ============================================================================
# Edge Cases
# ============================================================================

class TestRoutingEdgeCases:
    """Tests for edge cases."""

    def test_invalid_min_debates_param(self, routing_handler, mock_agent_selector):
        import aragora.server.handlers.routing as mod

        if not mod.ROUTING_AVAILABLE:
            pytest.skip("Routing module not available")

        with patch.object(mod, 'AgentSelector', return_value=mock_agent_selector):
            # Invalid param should use default
            result = routing_handler.handle("/api/routing/best-teams", {"min_debates": "invalid"}, None)

            assert result is not None
            assert result.status_code == 200

    def test_best_teams_empty_result(self, routing_handler, mock_agent_selector):
        import aragora.server.handlers.routing as mod

        if not mod.ROUTING_AVAILABLE:
            pytest.skip("Routing module not available")

        mock_agent_selector.get_best_team_combinations.return_value = []

        with patch.object(mod, 'AgentSelector', return_value=mock_agent_selector):
            result = routing_handler.handle("/api/routing/best-teams", {}, None)

            assert result is not None
            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["combinations"] == []
            assert data["count"] == 0
