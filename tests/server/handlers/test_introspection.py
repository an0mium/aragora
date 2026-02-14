"""
Tests for aragora.server.handlers.introspection - Introspection Endpoint Handlers.

Tests cover:
- IntrospectionHandler: instantiation, ROUTES, can_handle routing
- GET /api/introspection/all: get all agent introspection
- GET /api/introspection/leaderboard: ranked agents
- GET /api/introspection/agents: list agents
- GET /api/introspection/agents/availability: credential status
- GET /api/introspection/agents/{name}: specific agent introspection
- handle() routing: dispatches to correct method
- Error paths: module unavailable, rate limiting
- _get_critique_store, _get_persona_manager, _get_known_agents
- DEFAULT_AGENTS constant
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.introspection import IntrospectionHandler
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helpers
# ===========================================================================


def _parse_body(result: HandlerResult) -> dict[str, Any]:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body)


def _make_mock_handler(
    method: str = "GET",
    client_ip: str = "127.0.0.1",
) -> MagicMock:
    """Create a mock HTTP handler object."""
    handler = MagicMock()
    handler.command = method
    handler.client_address = (client_ip, 12345)
    handler.headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer test-token",
    }
    return handler


class MockIntrospectionSnapshot:
    """Mock introspection snapshot."""

    def __init__(self, agent: str = "claude", score: float = 0.85):
        self.agent_name = agent
        self.reputation_score = score

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "reputation_score": self.reputation_score,
            "strengths": ["reasoning", "clarity"],
            "weaknesses": [],
        }


class MockReputation:
    """Mock reputation object."""

    def __init__(self, agent_name: str, score: float = 0.7, critiques: int = 10):
        self.agent_name = agent_name
        self.score = score
        self.total_critiques = critiques


class MockCredentialStatus:
    """Mock credential status."""

    def __init__(self, available: bool = True):
        self.is_available = available
        self.required_vars = ["API_KEY"]
        self.missing_vars = [] if available else ["API_KEY"]
        self.available_via = "environment" if available else None


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def handler():
    """Create IntrospectionHandler with clean state."""
    h = IntrospectionHandler(ctx={})
    return h


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the rate limiter between tests."""
    import aragora.server.handlers.introspection as mod
    mod._introspection_limiter = mod.RateLimiter(requests_per_minute=1000)
    yield


@pytest.fixture(autouse=True)
def _clear_ttl_caches(handler):
    """Clear TTL caches between tests to avoid stale data."""
    # The ttl_cache decorator stores cache on the function; clear if present
    for method_name in ("_list_agents", "_get_all_introspection", "_get_introspection_leaderboard"):
        method = getattr(handler, method_name, None)
        if method and hasattr(method, "cache_clear"):
            method.cache_clear()
    yield


# ===========================================================================
# Test Basics
# ===========================================================================


class TestIntrospectionHandlerBasics:
    """Basic instantiation and routing tests."""

    def test_instantiation(self, handler):
        assert handler is not None
        assert isinstance(handler, IntrospectionHandler)

    def test_routes_attribute(self, handler):
        assert "/api/introspection/all" in handler.ROUTES
        assert "/api/introspection/leaderboard" in handler.ROUTES
        assert "/api/introspection/agents" in handler.ROUTES
        assert "/api/introspection/agents/availability" in handler.ROUTES
        assert "/api/introspection/agents/*" in handler.ROUTES

    def test_default_agents(self, handler):
        assert "claude" in handler.DEFAULT_AGENTS
        assert "gemini" in handler.DEFAULT_AGENTS
        assert len(handler.DEFAULT_AGENTS) == 5

    def test_can_handle_all(self, handler):
        assert handler.can_handle("/api/introspection/all") is True

    def test_can_handle_leaderboard(self, handler):
        assert handler.can_handle("/api/introspection/leaderboard") is True

    def test_can_handle_agents_list(self, handler):
        assert handler.can_handle("/api/introspection/agents") is True

    def test_can_handle_availability(self, handler):
        assert handler.can_handle("/api/introspection/agents/availability") is True

    def test_can_handle_specific_agent(self, handler):
        assert handler.can_handle("/api/introspection/agents/claude") is True

    def test_cannot_handle_other(self, handler):
        assert handler.can_handle("/api/debates") is False

    def test_cannot_handle_root_introspection(self, handler):
        assert handler.can_handle("/api/introspection") is False

    def test_can_handle_versioned_path(self, handler):
        assert handler.can_handle("/api/v1/introspection/all") is True


# ===========================================================================
# Test _list_agents
# ===========================================================================


class TestListAgents:
    """Tests for GET /api/introspection/agents."""

    def test_list_agents_defaults(self, handler):
        with patch.object(handler, "_get_critique_store", return_value=None):
            result = handler._list_agents()
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["count"] == 5
            assert len(data["agents"]) == 5

    def test_list_agents_with_reputations(self, handler):
        mock_store = MagicMock()
        mock_store.get_all_reputations.return_value = [
            MockReputation("claude", 0.9, 50),
            MockReputation("gemini", 0.8, 30),
        ]
        mock_store.get_agent_reputation.side_effect = lambda name: (
            MockReputation(name, 0.9 if name == "claude" else 0.8)
        )

        with patch.object(handler, "_get_critique_store", return_value=mock_store):
            result = handler._list_agents()
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["count"] == 2


# ===========================================================================
# Test _get_agent_introspection
# ===========================================================================


class TestGetAgentIntrospection:
    """Tests for GET /api/introspection/agents/{name}."""

    def test_get_agent_success(self, handler):
        mock_snapshot = MockIntrospectionSnapshot("claude")
        with patch(
            "aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True
        ), patch(
            "aragora.server.handlers.introspection.get_agent_introspection",
            return_value=mock_snapshot,
        ), patch.object(
            handler, "_get_critique_store", return_value=None
        ), patch.object(
            handler, "_get_persona_manager", return_value=None
        ):
            result = handler._get_agent_introspection("claude")
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["agent_name"] == "claude"

    def test_get_agent_not_found(self, handler):
        with patch(
            "aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True
        ), patch(
            "aragora.server.handlers.introspection.get_agent_introspection",
            return_value=None,
        ), patch.object(
            handler, "_get_critique_store", return_value=None
        ), patch.object(
            handler, "_get_persona_manager", return_value=None
        ):
            result = handler._get_agent_introspection("nonexistent")
            assert result.status_code == 404

    def test_get_agent_module_unavailable(self, handler):
        with patch(
            "aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", False
        ):
            result = handler._get_agent_introspection("claude")
            assert result.status_code == 503


# ===========================================================================
# Test _get_all_introspection
# ===========================================================================


class TestGetAllIntrospection:
    """Tests for GET /api/introspection/all."""

    def test_get_all_success(self, handler):
        mock_snapshot = MockIntrospectionSnapshot("claude")
        with patch(
            "aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True
        ), patch(
            "aragora.server.handlers.introspection.get_agent_introspection",
            return_value=mock_snapshot,
        ), patch.object(
            handler, "_get_critique_store", return_value=None
        ), patch.object(
            handler, "_get_persona_manager", return_value=None
        ):
            result = handler._get_all_introspection()
            assert result.status_code == 200
            data = _parse_body(result)
            assert "agents" in data
            assert data["count"] > 0

    def test_get_all_module_unavailable(self, handler):
        with patch(
            "aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", False
        ):
            result = handler._get_all_introspection()
            assert result.status_code == 503


# ===========================================================================
# Test _get_introspection_leaderboard
# ===========================================================================


class TestGetLeaderboard:
    """Tests for GET /api/introspection/leaderboard."""

    def test_leaderboard_success(self, handler):
        mock_snapshot = MockIntrospectionSnapshot("claude", 0.85)
        with patch(
            "aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True
        ), patch(
            "aragora.server.handlers.introspection.get_agent_introspection",
            return_value=mock_snapshot,
        ), patch.object(
            handler, "_get_critique_store", return_value=None
        ), patch.object(
            handler, "_get_persona_manager", return_value=None
        ):
            result = handler._get_introspection_leaderboard(10)
            assert result.status_code == 200
            data = _parse_body(result)
            assert "leaderboard" in data
            assert "total_agents" in data

    def test_leaderboard_respects_limit(self, handler):
        mock_snapshot = MockIntrospectionSnapshot("claude")
        with patch(
            "aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True
        ), patch(
            "aragora.server.handlers.introspection.get_agent_introspection",
            return_value=mock_snapshot,
        ), patch.object(
            handler, "_get_critique_store", return_value=None
        ), patch.object(
            handler, "_get_persona_manager", return_value=None
        ):
            result = handler._get_introspection_leaderboard(2)
            assert result.status_code == 200
            data = _parse_body(result)
            assert len(data["leaderboard"]) <= 2

    def test_leaderboard_module_unavailable(self, handler):
        with patch(
            "aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", False
        ):
            result = handler._get_introspection_leaderboard(10)
            assert result.status_code == 503


# ===========================================================================
# Test _get_agent_availability
# ===========================================================================


class TestGetAgentAvailability:
    """Tests for GET /api/introspection/agents/availability."""

    def test_availability_success(self, handler):
        with patch(
            "aragora.server.handlers.introspection.get_agent_credential_status",
            return_value={
                "claude": MockCredentialStatus(True),
                "gemini": MockCredentialStatus(False),
            },
        ):
            result = handler._get_agent_availability()
            assert result.status_code == 200
            data = _parse_body(result)
            assert "claude" in data["available"]
            assert "gemini" in data["missing"]
            assert "claude" in data["details"]

    def test_availability_module_unavailable(self, handler):
        with patch(
            "aragora.server.handlers.introspection.get_agent_credential_status",
            side_effect=ImportError("not available"),
        ), patch.dict(
            "sys.modules",
            {"aragora.agents.credential_validator": None},
        ):
            # The handler catches ImportError at the module level
            # We need to simulate the import failure inside the method
            result = handler._get_agent_availability()
            # Should handle gracefully
            assert result is not None


# ===========================================================================
# Test handle() Routing
# ===========================================================================


class TestHandleRouting:
    """Tests for the top-level handle() method routing."""

    def test_handle_all(self, handler):
        mock_handler = _make_mock_handler()
        mock_snapshot = MockIntrospectionSnapshot("claude")
        with patch(
            "aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True
        ), patch(
            "aragora.server.handlers.introspection.get_agent_introspection",
            return_value=mock_snapshot,
        ), patch.object(
            handler, "_get_critique_store", return_value=None
        ), patch.object(
            handler, "_get_persona_manager", return_value=None
        ):
            result = handler.handle.__wrapped__(handler, "/api/introspection/all", {}, mock_handler)
            assert result is not None
            assert result.status_code == 200

    def test_handle_leaderboard(self, handler):
        mock_handler = _make_mock_handler()
        mock_snapshot = MockIntrospectionSnapshot("claude")
        with patch(
            "aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True
        ), patch(
            "aragora.server.handlers.introspection.get_agent_introspection",
            return_value=mock_snapshot,
        ), patch.object(
            handler, "_get_critique_store", return_value=None
        ), patch.object(
            handler, "_get_persona_manager", return_value=None
        ):
            result = handler.handle.__wrapped__(handler, "/api/introspection/leaderboard", {}, mock_handler)
            assert result is not None
            assert result.status_code == 200

    def test_handle_agents_list(self, handler):
        mock_handler = _make_mock_handler()
        with patch.object(handler, "_get_critique_store", return_value=None):
            result = handler.handle.__wrapped__(handler, "/api/introspection/agents", {}, mock_handler)
            assert result is not None
            assert result.status_code == 200

    def test_handle_agent_availability(self, handler):
        mock_handler = _make_mock_handler()
        with patch(
            "aragora.server.handlers.introspection.get_agent_credential_status",
            return_value={"claude": MockCredentialStatus(True)},
        ):
            result = handler.handle.__wrapped__(handler, "/api/introspection/agents/availability", {}, mock_handler)
            assert result is not None
            assert result.status_code == 200

    def test_handle_specific_agent(self, handler):
        mock_handler = _make_mock_handler()
        mock_snapshot = MockIntrospectionSnapshot("claude")
        with patch(
            "aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True
        ), patch(
            "aragora.server.handlers.introspection.get_agent_introspection",
            return_value=mock_snapshot,
        ), patch.object(
            handler, "_get_critique_store", return_value=None
        ), patch.object(
            handler, "_get_persona_manager", return_value=None
        ):
            result = handler.handle.__wrapped__(
                handler, "/api/introspection/agents/claude", {}, mock_handler
            )
            assert result is not None
            assert result.status_code == 200

    def test_handle_unmatched_returns_none(self, handler):
        mock_handler = _make_mock_handler()
        result = handler.handle.__wrapped__(
            handler, "/api/introspection/unknown", {}, mock_handler
        )
        assert result is None

    def test_handle_rate_limited(self, handler):
        import aragora.server.handlers.introspection as mod
        mod._introspection_limiter = mod.RateLimiter(requests_per_minute=0)

        mock_handler = _make_mock_handler()
        with patch.object(mod._introspection_limiter, "is_allowed", return_value=False):
            result = handler.handle.__wrapped__(
                handler, "/api/introspection/all", {}, mock_handler
            )
            assert result is not None
            assert result.status_code == 429


# ===========================================================================
# Test _get_critique_store and _get_persona_manager
# ===========================================================================


class TestDependencyGetters:
    """Tests for dependency getter methods."""

    def test_get_critique_store_unavailable(self, handler):
        with patch(
            "aragora.server.handlers.introspection.CRITIQUE_STORE_AVAILABLE", False
        ):
            assert handler._get_critique_store() is None

    def test_get_persona_manager_unavailable(self, handler):
        with patch(
            "aragora.server.handlers.introspection.PERSONA_MANAGER_AVAILABLE", False
        ):
            assert handler._get_persona_manager() is None

    def test_get_known_agents_with_store(self, handler):
        mock_store = MagicMock()
        mock_store.get_all_reputations.return_value = [
            MockReputation("agent-a"),
            MockReputation("agent-b"),
        ]
        agents = handler._get_known_agents(mock_store)
        assert agents == ["agent-a", "agent-b"]

    def test_get_known_agents_without_store(self, handler):
        agents = handler._get_known_agents(None)
        assert agents == handler.DEFAULT_AGENTS

    def test_get_known_agents_empty_reputations(self, handler):
        mock_store = MagicMock()
        mock_store.get_all_reputations.return_value = []
        agents = handler._get_known_agents(mock_store)
        assert agents == handler.DEFAULT_AGENTS

    def test_get_known_agents_store_error(self, handler):
        mock_store = MagicMock()
        mock_store.get_all_reputations.side_effect = RuntimeError("DB error")
        agents = handler._get_known_agents(mock_store)
        assert agents == handler.DEFAULT_AGENTS
