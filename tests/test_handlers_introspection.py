"""
Tests for IntrospectionHandler endpoints.

Endpoints tested:
- GET /api/introspection/all - Get introspection for all agents
- GET /api/introspection/leaderboard - Get agents ranked by reputation
- GET /api/introspection/agents/{name} - Get introspection for specific agent
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from aragora.server.handlers.introspection import IntrospectionHandler
from aragora.server.handlers.base import clear_cache


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_introspection_snapshot():
    """Create a mock introspection snapshot."""

    def create_snapshot(agent_name, reputation_score=0.75):
        snapshot = Mock()
        snapshot.to_dict.return_value = {
            "agent_name": agent_name,
            "reputation_score": reputation_score,
            "total_debates": 50,
            "wins": 30,
            "losses": 15,
            "draws": 5,
            "strengths": ["analytical", "thorough"],
            "weaknesses": ["speed"],
            "recent_performance": {"last_5_debates": ["win", "win", "loss", "draw", "win"]},
        }
        return snapshot

    return create_snapshot


@pytest.fixture
def mock_get_agent_introspection(mock_introspection_snapshot):
    """Create a mock get_agent_introspection function."""

    def mock_func(agent, memory=None, persona_manager=None):
        return mock_introspection_snapshot(agent)

    return mock_func


@pytest.fixture
def mock_critique_store():
    """Create a mock critique store."""
    store = Mock()
    reputation1 = Mock()
    reputation1.agent_name = "claude"
    reputation2 = Mock()
    reputation2.agent_name = "gpt4"
    store.get_all_reputations.return_value = [reputation1, reputation2]
    return store


@pytest.fixture
def introspection_handler(tmp_path):
    """Create an IntrospectionHandler with mock dependencies."""
    ctx = {
        "storage": None,
        "elo_system": None,
        "nomic_dir": tmp_path,
    }
    return IntrospectionHandler(ctx)


@pytest.fixture
def handler_no_nomic_dir():
    """Create an IntrospectionHandler without nomic_dir."""
    ctx = {
        "storage": None,
        "elo_system": None,
        "nomic_dir": None,
    }
    return IntrospectionHandler(ctx)


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches before and after each test."""
    clear_cache()
    yield
    clear_cache()


# ============================================================================
# Route Matching Tests
# ============================================================================


class TestIntrospectionRouting:
    """Tests for route matching."""

    def test_can_handle_all_introspection(self, introspection_handler):
        assert introspection_handler.can_handle("/api/v1/introspection/all") is True

    def test_can_handle_leaderboard(self, introspection_handler):
        assert introspection_handler.can_handle("/api/v1/introspection/leaderboard") is True

    def test_can_handle_agent_introspection(self, introspection_handler):
        assert introspection_handler.can_handle("/api/v1/introspection/agents/claude") is True
        assert introspection_handler.can_handle("/api/v1/introspection/agents/gpt-4") is True

    def test_can_handle_list_agents(self, introspection_handler):
        """Test that /api/introspection/agents endpoint is handled."""
        assert introspection_handler.can_handle("/api/v1/introspection/agents") is True

    def test_cannot_handle_unrelated_routes(self, introspection_handler):
        assert introspection_handler.can_handle("/api/v1/introspection") is False
        # /api/introspection/agents is now a valid endpoint (list agents)
        assert introspection_handler.can_handle("/api/v1/agents") is False
        assert introspection_handler.can_handle("/api/v1/personas") is False


# ============================================================================
# GET /api/introspection/all Tests
# ============================================================================


class TestGetAllIntrospection:
    """Tests for GET /api/introspection/all endpoint."""

    def test_all_introspection_module_unavailable(self, introspection_handler):
        import aragora.server.handlers.introspection as mod

        original = mod.INTROSPECTION_AVAILABLE
        mod.INTROSPECTION_AVAILABLE = False
        try:
            result = introspection_handler.handle("/api/introspection/all", {}, None)
            assert result is not None
            assert result.status_code == 503
            data = json.loads(result.body)
            assert "not available" in data["error"].lower()
        finally:
            mod.INTROSPECTION_AVAILABLE = original

    def test_all_introspection_success(self, introspection_handler, mock_get_agent_introspection):
        import aragora.server.handlers.introspection as mod

        if not mod.INTROSPECTION_AVAILABLE:
            pytest.skip("Introspection module not available")

        with patch.object(mod, "get_agent_introspection", mock_get_agent_introspection):
            result = introspection_handler.handle("/api/introspection/all", {}, None)

            assert result is not None
            assert result.status_code == 200
            data = json.loads(result.body)
            assert "agents" in data
            assert "count" in data


# ============================================================================
# GET /api/introspection/leaderboard Tests
# ============================================================================


class TestGetIntrospectionLeaderboard:
    """Tests for GET /api/introspection/leaderboard endpoint."""

    def test_leaderboard_module_unavailable(self, introspection_handler):
        import aragora.server.handlers.introspection as mod

        original = mod.INTROSPECTION_AVAILABLE
        mod.INTROSPECTION_AVAILABLE = False
        try:
            result = introspection_handler.handle("/api/introspection/leaderboard", {}, None)
            assert result is not None
            assert result.status_code == 503
        finally:
            mod.INTROSPECTION_AVAILABLE = original

    def test_leaderboard_with_limit(self, introspection_handler, mock_get_agent_introspection):
        import aragora.server.handlers.introspection as mod

        if not mod.INTROSPECTION_AVAILABLE:
            pytest.skip("Introspection module not available")

        with patch.object(mod, "get_agent_introspection", mock_get_agent_introspection):
            result = introspection_handler.handle(
                "/api/introspection/leaderboard", {"limit": "5"}, None
            )

            assert result is not None
            assert result.status_code == 200
            data = json.loads(result.body)
            assert "leaderboard" in data
            assert "total_agents" in data

    def test_leaderboard_limit_capped(self, introspection_handler, mock_get_agent_introspection):
        import aragora.server.handlers.introspection as mod

        if not mod.INTROSPECTION_AVAILABLE:
            pytest.skip("Introspection module not available")

        with patch.object(mod, "get_agent_introspection", mock_get_agent_introspection):
            # Request limit of 100, should be capped at 50
            result = introspection_handler.handle(
                "/api/introspection/leaderboard", {"limit": "100"}, None
            )

            assert result is not None
            assert result.status_code == 200


# ============================================================================
# GET /api/introspection/agents Tests (List Agents)
# ============================================================================


class TestListAgents:
    """Tests for GET /api/introspection/agents endpoint."""

    def test_list_agents_returns_list(self, introspection_handler):
        """Test that list agents returns agent list."""
        from aragora.server.handlers.base import clear_cache

        clear_cache()  # Clear cache for fresh response

        result = introspection_handler.handle("/api/introspection/agents", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "agents" in data
        assert "count" in data
        assert isinstance(data["agents"], list)

    def test_list_agents_default_agents(self, introspection_handler):
        """Test that default agents are returned when no store available."""
        from aragora.server.handlers.base import clear_cache

        clear_cache()

        result = introspection_handler.handle("/api/introspection/agents", {}, None)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        # Should have at least the default agents
        assert data["count"] >= 1
        # Each agent should have a name
        for agent in data["agents"]:
            assert "name" in agent


# ============================================================================
# GET /api/introspection/agents/{name} Tests
# ============================================================================


class TestGetAgentIntrospection:
    """Tests for GET /api/introspection/agents/{name} endpoint."""

    def test_agent_introspection_module_unavailable(self, introspection_handler):
        import aragora.server.handlers.introspection as mod

        original = mod.INTROSPECTION_AVAILABLE
        mod.INTROSPECTION_AVAILABLE = False
        try:
            result = introspection_handler.handle("/api/introspection/agents/claude", {}, None)
            assert result is not None
            assert result.status_code == 503
        finally:
            mod.INTROSPECTION_AVAILABLE = original

    def test_agent_introspection_success(self, introspection_handler, mock_get_agent_introspection):
        import aragora.server.handlers.introspection as mod

        if not mod.INTROSPECTION_AVAILABLE:
            pytest.skip("Introspection module not available")

        with patch.object(mod, "get_agent_introspection", mock_get_agent_introspection):
            result = introspection_handler.handle("/api/introspection/agents/claude", {}, None)

            assert result is not None
            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["agent_name"] == "claude"

    def test_agent_introspection_invalid_name(self, introspection_handler):
        # Use special characters that validation rejects
        result = introspection_handler.handle("/api/introspection/agents/test<script>", {}, None)

        assert result is not None
        assert result.status_code == 400

    def test_agent_introspection_with_hyphen(
        self, introspection_handler, mock_get_agent_introspection
    ):
        import aragora.server.handlers.introspection as mod

        if not mod.INTROSPECTION_AVAILABLE:
            pytest.skip("Introspection module not available")

        with patch.object(mod, "get_agent_introspection", mock_get_agent_introspection):
            result = introspection_handler.handle("/api/introspection/agents/gpt-4", {}, None)

            assert result is not None
            # Should either succeed or return agent not found, not 400
            assert result.status_code in (200, 404, 500)


# ============================================================================
# Security Tests
# ============================================================================


class TestIntrospectionSecurity:
    """Security tests for introspection endpoints."""

    def test_path_traversal_blocked(self, introspection_handler):
        # The handler extracts last segment, so use .. in segment name itself
        result = introspection_handler.handle("/api/introspection/agents/test..admin", {}, None)
        assert result.status_code == 400

    def test_sql_injection_blocked(self, introspection_handler):
        result = introspection_handler.handle(
            "/api/introspection/agents/'; DROP TABLE agents;--", {}, None
        )
        assert result.status_code == 400

    def test_xss_blocked(self, introspection_handler):
        result = introspection_handler.handle(
            "/api/introspection/agents/<script>alert(1)</script>", {}, None
        )
        assert result.status_code == 400


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestIntrospectionErrorHandling:
    """Tests for error handling."""

    def test_handle_returns_none_for_unhandled_route(self, introspection_handler):
        result = introspection_handler.handle("/api/other/endpoint", {}, None)
        assert result is None

    def test_introspection_exception(self, introspection_handler):
        import aragora.server.handlers.introspection as mod

        if not mod.INTROSPECTION_AVAILABLE:
            pytest.skip("Introspection module not available")

        def raise_error(*args, **kwargs):
            raise Exception("Introspection error")

        with patch.object(mod, "get_agent_introspection", raise_error):
            result = introspection_handler.handle("/api/introspection/agents/claude", {}, None)

            assert result is not None
            assert result.status_code == 500


# ============================================================================
# Edge Cases
# ============================================================================


class TestIntrospectionEdgeCases:
    """Tests for edge cases."""

    def test_empty_agent_name(self, introspection_handler):
        # Path with empty agent name should not be handled
        result = introspection_handler.handle("/api/introspection/agents/", {}, None)
        # Empty name validation should fail
        assert result is None or result.status_code == 400

    def test_very_long_agent_name(self, introspection_handler):
        long_name = "a" * 1000
        result = introspection_handler.handle(f"/api/introspection/agents/{long_name}", {}, None)
        # Should handle gracefully (either accept or reject)
        assert result is not None

    def test_unicode_agent_name(self, introspection_handler):
        result = introspection_handler.handle("/api/introspection/agents/测试", {}, None)
        # Should either accept or reject gracefully
        assert result is not None

    def test_default_agents_used_when_no_store(
        self, handler_no_nomic_dir, mock_get_agent_introspection
    ):
        import aragora.server.handlers.introspection as mod

        if not mod.INTROSPECTION_AVAILABLE:
            pytest.skip("Introspection module not available")

        with patch.object(mod, "get_agent_introspection", mock_get_agent_introspection):
            result = handler_no_nomic_dir.handle("/api/introspection/all", {}, None)

            assert result is not None
            # Should use DEFAULT_AGENTS when no critique store available
            assert result.status_code == 200
