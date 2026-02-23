"""Tests for introspection handler endpoints.

Tests the introspection API endpoints including:
- GET /api/introspection/all - Get introspection for all agents
- GET /api/introspection/leaderboard - Get agents ranked by reputation
- GET /api/introspection/agents - List available agents
- GET /api/introspection/agents/availability - Get agent credential availability
- GET /api/introspection/agents/{name} - Get introspection for specific agent
"""

import json
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


def parse_body(result) -> dict:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


class MockHandler:
    """Mock HTTP handler for tests."""

    def __init__(self):
        self.client_address = ("127.0.0.1", 12345)
        self.headers = {}


class MockReputation:
    """Mock reputation object."""

    def __init__(self, agent_name: str, score: float = 0.5, total_critiques: int = 10):
        self.agent_name = agent_name
        self.score = score
        self.total_critiques = total_critiques


class MockSnapshot:
    """Mock introspection snapshot."""

    def __init__(self, agent_name: str, reputation_score: float = 0.75, data: dict | None = None):
        self.agent_name = agent_name
        self.reputation_score = reputation_score
        self._data = data or {
            "agent_name": agent_name,
            "reputation_score": reputation_score,
            "self_assessment": "I am performing well",
            "strengths": ["reasoning", "analysis"],
            "weaknesses": ["creativity"],
        }

    def to_dict(self) -> dict:
        return self._data


class MockCredentialStatus:
    """Mock credential status for agent availability."""

    def __init__(
        self,
        is_available: bool = True,
        required_vars: list[str] | None = None,
        missing_vars: list[str] | None = None,
        available_via: str = "env",
    ):
        self.is_available = is_available
        self.required_vars = required_vars or ["API_KEY"]
        self.missing_vars = missing_vars or []
        self.available_via = available_via


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP handler."""
    return MockHandler()


@pytest.fixture
def handler():
    """Create IntrospectionHandler for testing."""
    from aragora.server.handlers.introspection import IntrospectionHandler

    return IntrospectionHandler({})


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset the rate limiter before each test so tests don't interfere."""
    with patch("aragora.server.handlers.introspection._introspection_limiter") as mock_limiter:
        mock_limiter.is_allowed.return_value = True
        yield mock_limiter


@pytest.fixture(autouse=True)
def clear_ttl_cache():
    """Clear TTL caches before each test to avoid stale cached results."""
    from aragora.server.handlers.admin.cache import clear_cache

    clear_cache()
    yield
    # Clear after test to prevent cross-test contamination
    clear_cache()


# ============================================================================
# Routing Tests
# ============================================================================


class TestIntrospectionRouting:
    """Test route matching via can_handle."""

    def test_can_handle_all(self, handler):
        """Test matching /api/introspection/all."""
        assert handler.can_handle("/api/introspection/all") is True

    def test_can_handle_all_with_version(self, handler):
        """Test matching /api/v1/introspection/all strips version prefix."""
        assert handler.can_handle("/api/v1/introspection/all") is True

    def test_can_handle_leaderboard(self, handler):
        """Test matching /api/introspection/leaderboard."""
        assert handler.can_handle("/api/introspection/leaderboard") is True

    def test_can_handle_leaderboard_with_version(self, handler):
        """Test matching /api/v1/introspection/leaderboard."""
        assert handler.can_handle("/api/v1/introspection/leaderboard") is True

    def test_can_handle_agents_list(self, handler):
        """Test matching /api/introspection/agents."""
        assert handler.can_handle("/api/introspection/agents") is True

    def test_can_handle_agents_availability(self, handler):
        """Test matching /api/introspection/agents/availability."""
        assert handler.can_handle("/api/introspection/agents/availability") is True

    def test_can_handle_specific_agent(self, handler):
        """Test matching /api/introspection/agents/{name}."""
        assert handler.can_handle("/api/introspection/agents/claude") is True
        assert handler.can_handle("/api/introspection/agents/gpt-4") is True
        assert handler.can_handle("/api/introspection/agents/deepseek") is True

    def test_cannot_handle_unrelated(self, handler):
        """Test rejecting unrelated routes."""
        assert handler.can_handle("/api/debates") is False
        assert handler.can_handle("/api/agents") is False
        assert handler.can_handle("/api/health") is False

    def test_cannot_handle_partial_prefix(self, handler):
        """Test rejecting paths that only partially match."""
        assert handler.can_handle("/api/introspection") is False
        assert handler.can_handle("/api/introspect") is False

    def test_handle_returns_none_for_unknown_route(self, handler, mock_http_handler):
        """Test that handle returns None for unmatched paths."""
        result = handler.handle("/api/unknown", {}, mock_http_handler)
        assert result is None


# ============================================================================
# List Agents Tests
# ============================================================================


class TestListAgents:
    """Test GET /api/introspection/agents endpoint."""

    def test_list_agents_default_agents(self, handler, mock_http_handler):
        """Test listing agents falls back to defaults when no store."""
        with patch.object(handler, "_get_critique_store", return_value=None):
            result = handler.handle("/api/introspection/agents", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert "agents" in body
        assert "count" in body
        assert body["count"] == 5  # DEFAULT_AGENTS has 5
        agent_names = [a["name"] for a in body["agents"]]
        assert "claude" in agent_names
        assert "gemini" in agent_names

    def test_list_agents_from_store(self, handler, mock_http_handler):
        """Test listing agents from critique store reputations."""
        mock_store = MagicMock()
        mock_store.get_all_reputations.return_value = [
            MockReputation("alice", score=0.9),
            MockReputation("bob", score=0.7),
        ]
        mock_store.get_agent_reputation.side_effect = lambda name: (
            MockReputation("alice", score=0.9) if name == "alice"
            else MockReputation("bob", score=0.7)
        )

        with patch.object(handler, "_get_critique_store", return_value=mock_store):
            result = handler.handle("/api/introspection/agents", {}, mock_http_handler)

        assert result.status_code == 200
        body = parse_body(result)
        assert body["count"] == 2
        # Sorted by reputation descending
        assert body["agents"][0]["name"] == "alice"
        assert body["agents"][0]["reputation_score"] == 0.9

    def test_list_agents_with_reputation_scores(self, handler, mock_http_handler):
        """Test that reputation scores are included when available."""
        mock_store = MagicMock()
        rep = MockReputation("claude", score=0.85, total_critiques=42)
        mock_store.get_all_reputations.return_value = [rep]
        mock_store.get_agent_reputation.return_value = rep

        with patch.object(handler, "_get_critique_store", return_value=mock_store):
            result = handler.handle("/api/introspection/agents", {}, mock_http_handler)

        assert result.status_code == 200
        body = parse_body(result)
        agent = body["agents"][0]
        assert agent["reputation_score"] == 0.85
        assert agent["total_critiques"] == 42

    def test_list_agents_reputation_error_handled(self, handler, mock_http_handler):
        """Test that individual reputation errors are gracefully handled."""
        mock_store = MagicMock()
        mock_store.get_all_reputations.return_value = [
            MockReputation("alice"),
            MockReputation("bob"),
        ]
        mock_store.get_agent_reputation.side_effect = ValueError("DB error")

        with patch.object(handler, "_get_critique_store", return_value=mock_store):
            result = handler.handle("/api/introspection/agents", {}, mock_http_handler)

        assert result.status_code == 200
        body = parse_body(result)
        # Both agents present but without reputation data
        assert body["count"] == 2

    def test_list_agents_sorted_descending_by_reputation(self, handler, mock_http_handler):
        """Test agents sorted by reputation score descending."""
        mock_store = MagicMock()
        mock_store.get_all_reputations.return_value = [
            MockReputation("low", score=0.2),
            MockReputation("high", score=0.95),
            MockReputation("mid", score=0.6),
        ]
        mock_store.get_agent_reputation.side_effect = lambda name: {
            "low": MockReputation("low", score=0.2),
            "high": MockReputation("high", score=0.95),
            "mid": MockReputation("mid", score=0.6),
        }[name]

        with patch.object(handler, "_get_critique_store", return_value=mock_store):
            result = handler.handle("/api/introspection/agents", {}, mock_http_handler)

        body = parse_body(result)
        scores = [a.get("reputation_score", 0) for a in body["agents"]]
        assert scores == sorted(scores, reverse=True)

    def test_list_agents_empty_reputations_falls_back(self, handler, mock_http_handler):
        """Test that empty reputations list falls back to default agents."""
        mock_store = MagicMock()
        mock_store.get_all_reputations.return_value = []
        # get_agent_reputation returns None for default agents
        mock_store.get_agent_reputation.return_value = None

        with patch.object(handler, "_get_critique_store", return_value=mock_store):
            result = handler.handle("/api/introspection/agents", {}, mock_http_handler)

        assert result.status_code == 200
        body = parse_body(result)
        assert body["count"] == 5  # DEFAULT_AGENTS

    def test_list_agents_reputations_oserror_falls_back(self, handler, mock_http_handler):
        """Test that OSError in reputations falls back to default agents."""
        mock_store = MagicMock()
        mock_store.get_all_reputations.side_effect = OSError("disk error")
        # When reputation lookup also fails, it should be handled gracefully
        mock_store.get_agent_reputation.side_effect = OSError("disk error")

        with patch.object(handler, "_get_critique_store", return_value=mock_store):
            result = handler.handle("/api/introspection/agents", {}, mock_http_handler)

        assert result.status_code == 200
        body = parse_body(result)
        assert body["count"] == 5


# ============================================================================
# Agent Introspection Tests
# ============================================================================


class TestAgentIntrospection:
    """Test GET /api/introspection/agents/{name} endpoint."""

    @patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True)
    @patch("aragora.server.handlers.introspection.get_agent_introspection")
    def test_get_agent_success(self, mock_get_intro, handler, mock_http_handler):
        """Test successful agent introspection retrieval."""
        mock_get_intro.return_value = MockSnapshot("claude", 0.85)

        with patch.object(handler, "_get_critique_store", return_value=None), \
             patch.object(handler, "_get_persona_manager", return_value=None):
            result = handler.handle(
                "/api/introspection/agents/claude", {}, mock_http_handler
            )

        assert result.status_code == 200
        body = parse_body(result)
        assert body["agent_name"] == "claude"
        assert body["reputation_score"] == 0.85

    @patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True)
    @patch("aragora.server.handlers.introspection.get_agent_introspection")
    def test_get_agent_not_found(self, mock_get_intro, handler, mock_http_handler):
        """Test agent introspection for non-existent agent."""
        mock_get_intro.return_value = None

        with patch.object(handler, "_get_critique_store", return_value=None), \
             patch.object(handler, "_get_persona_manager", return_value=None):
            result = handler.handle(
                "/api/introspection/agents/nonexistent", {}, mock_http_handler
            )

        assert result.status_code == 404

    @patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", False)
    def test_get_agent_module_not_available(self, handler, mock_http_handler):
        """Test returns 503 when introspection module is not available."""
        result = handler.handle(
            "/api/introspection/agents/claude", {}, mock_http_handler
        )

        assert result.status_code == 503
        body = parse_body(result)
        assert "not available" in body["error"].lower()

    @patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True)
    @patch("aragora.server.handlers.introspection.get_agent_introspection")
    def test_get_agent_error_handling(self, mock_get_intro, handler, mock_http_handler):
        """Test error handling during introspection retrieval."""
        mock_get_intro.side_effect = ValueError("introspection error")

        with patch.object(handler, "_get_critique_store", return_value=None), \
             patch.object(handler, "_get_persona_manager", return_value=None):
            result = handler.handle(
                "/api/introspection/agents/claude", {}, mock_http_handler
            )

        assert result.status_code == 500

    @patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True)
    @patch("aragora.server.handlers.introspection.get_agent_introspection")
    def test_get_agent_attribute_error(self, mock_get_intro, handler, mock_http_handler):
        """Test AttributeError handling in agent introspection."""
        mock_get_intro.side_effect = AttributeError("missing attr")

        with patch.object(handler, "_get_critique_store", return_value=None), \
             patch.object(handler, "_get_persona_manager", return_value=None):
            result = handler.handle(
                "/api/introspection/agents/claude", {}, mock_http_handler
            )

        assert result.status_code == 500

    @patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True)
    @patch("aragora.server.handlers.introspection.get_agent_introspection")
    def test_get_agent_passes_memory_and_persona(self, mock_get_intro, handler, mock_http_handler):
        """Test that memory and persona_manager are passed to introspection."""
        mock_get_intro.return_value = MockSnapshot("claude")
        mock_store = MagicMock()
        mock_persona = MagicMock()

        with patch.object(handler, "_get_critique_store", return_value=mock_store), \
             patch.object(handler, "_get_persona_manager", return_value=mock_persona):
            handler.handle(
                "/api/introspection/agents/claude", {}, mock_http_handler
            )

        mock_get_intro.assert_called_once_with(
            "claude", memory=mock_store, persona_manager=mock_persona
        )

    def test_get_agent_invalid_name_rejected(self, handler, mock_http_handler):
        """Test that invalid agent names are rejected by path param validation."""
        result = handler.handle(
            "/api/introspection/agents/../../etc/passwd", {}, mock_http_handler
        )
        assert result is not None
        assert result.status_code == 400

    def test_get_agent_name_with_spaces_rejected(self, handler, mock_http_handler):
        """Test that agent names with spaces are rejected."""
        result = handler.handle(
            "/api/introspection/agents/bad agent", {}, mock_http_handler
        )
        # The path wouldn't match as a valid route or would fail validation
        assert result is None or result.status_code == 400

    def test_get_agent_empty_name_returns_agents_list_or_none(self, handler, mock_http_handler):
        """Test /api/introspection/agents/ (trailing slash) handling."""
        # This path ends with / so the extracted agent name would be empty
        result = handler.handle("/api/introspection/agents/", {}, mock_http_handler)
        # Empty segment - should be rejected
        if result is not None:
            assert result.status_code == 400

    def test_get_agent_name_too_long_rejected(self, handler, mock_http_handler):
        """Test that excessively long agent names are rejected."""
        long_name = "a" * 100  # exceeds 32 char limit
        result = handler.handle(
            f"/api/introspection/agents/{long_name}", {}, mock_http_handler
        )
        assert result is not None
        assert result.status_code == 400

    def test_get_agent_hyphenated_name(self, handler, mock_http_handler):
        """Test that hyphenated agent names are accepted."""
        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True), \
             patch("aragora.server.handlers.introspection.get_agent_introspection") as mock_intro:
            mock_intro.return_value = MockSnapshot("gpt-4")
            with patch.object(handler, "_get_critique_store", return_value=None), \
                 patch.object(handler, "_get_persona_manager", return_value=None):
                result = handler.handle(
                    "/api/introspection/agents/gpt-4", {}, mock_http_handler
                )
            assert result.status_code == 200

    def test_get_agent_underscore_name(self, handler, mock_http_handler):
        """Test that underscore agent names are accepted."""
        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True), \
             patch("aragora.server.handlers.introspection.get_agent_introspection") as mock_intro:
            mock_intro.return_value = MockSnapshot("my_agent")
            with patch.object(handler, "_get_critique_store", return_value=None), \
                 patch.object(handler, "_get_persona_manager", return_value=None):
                result = handler.handle(
                    "/api/introspection/agents/my_agent", {}, mock_http_handler
                )
            assert result.status_code == 200


# ============================================================================
# All Introspection Tests
# ============================================================================


class TestAllIntrospection:
    """Test GET /api/introspection/all endpoint."""

    @patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True)
    @patch("aragora.server.handlers.introspection.get_agent_introspection")
    def test_get_all_success(self, mock_get_intro, handler, mock_http_handler):
        """Test successful retrieval of all agents' introspection."""
        mock_get_intro.side_effect = lambda name, **kw: MockSnapshot(name)

        with patch.object(handler, "_get_critique_store", return_value=None), \
             patch.object(handler, "_get_persona_manager", return_value=None):
            result = handler.handle("/api/introspection/all", {}, mock_http_handler)

        assert result.status_code == 200
        body = parse_body(result)
        assert "agents" in body
        assert "count" in body
        assert body["count"] == 5  # DEFAULT_AGENTS

    @patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True)
    @patch("aragora.server.handlers.introspection.get_agent_introspection")
    def test_get_all_partial_failures(self, mock_get_intro, handler, mock_http_handler):
        """Test that individual agent failures are handled gracefully."""
        def side_effect(name, **kw):
            if name == "gemini":
                raise ValueError("introspection error")
            return MockSnapshot(name)

        mock_get_intro.side_effect = side_effect

        with patch.object(handler, "_get_critique_store", return_value=None), \
             patch.object(handler, "_get_persona_manager", return_value=None):
            result = handler.handle("/api/introspection/all", {}, mock_http_handler)

        assert result.status_code == 200
        body = parse_body(result)
        # Should have 4 out of 5 (gemini failed)
        assert body["count"] == 4
        assert "gemini" not in body["agents"]

    @patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", False)
    def test_get_all_module_not_available(self, handler, mock_http_handler):
        """Test returns 503 when introspection module unavailable."""
        result = handler.handle("/api/introspection/all", {}, mock_http_handler)

        assert result.status_code == 503

    @patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True)
    @patch("aragora.server.handlers.introspection.get_agent_introspection")
    def test_get_all_with_custom_agents(self, mock_get_intro, handler, mock_http_handler):
        """Test all introspection uses agents from store when available."""
        mock_store = MagicMock()
        mock_store.get_all_reputations.return_value = [
            MockReputation("alpha"),
            MockReputation("beta"),
        ]
        mock_get_intro.side_effect = lambda name, **kw: MockSnapshot(name)

        with patch.object(handler, "_get_critique_store", return_value=mock_store), \
             patch.object(handler, "_get_persona_manager", return_value=None):
            result = handler.handle("/api/introspection/all", {}, mock_http_handler)

        assert result.status_code == 200
        body = parse_body(result)
        assert body["count"] == 2
        assert "alpha" in body["agents"]
        assert "beta" in body["agents"]

    @patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True)
    @patch("aragora.server.handlers.introspection.get_agent_introspection")
    def test_get_all_total_failure(self, mock_get_intro, handler, mock_http_handler):
        """Test all introspection when every agent fails."""
        mock_get_intro.side_effect = TypeError("total failure")

        with patch.object(handler, "_get_critique_store", return_value=None), \
             patch.object(handler, "_get_persona_manager", return_value=None):
            result = handler.handle("/api/introspection/all", {}, mock_http_handler)

        assert result.status_code == 200
        body = parse_body(result)
        assert body["count"] == 0

    @patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True)
    @patch("aragora.server.handlers.introspection.get_agent_introspection")
    def test_get_all_contains_agent_data(self, mock_get_intro, handler, mock_http_handler):
        """Test that all introspection contains correct snapshot data."""
        mock_get_intro.side_effect = lambda name, **kw: MockSnapshot(
            name, data={"agent_name": name, "reputation_score": 0.8}
        )

        with patch.object(handler, "_get_critique_store", return_value=None), \
             patch.object(handler, "_get_persona_manager", return_value=None):
            result = handler.handle("/api/introspection/all", {}, mock_http_handler)

        body = parse_body(result)
        for agent_name, snapshot in body["agents"].items():
            assert snapshot["agent_name"] == agent_name


# ============================================================================
# Leaderboard Tests
# ============================================================================


class TestLeaderboard:
    """Test GET /api/introspection/leaderboard endpoint."""

    @patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True)
    @patch("aragora.server.handlers.introspection.get_agent_introspection")
    def test_leaderboard_success(self, mock_get_intro, handler, mock_http_handler):
        """Test successful leaderboard retrieval."""
        scores = {"gemini": 0.9, "claude": 0.85, "codex": 0.7, "grok": 0.6, "deepseek": 0.5}
        mock_get_intro.side_effect = lambda name, **kw: MockSnapshot(
            name, data={"agent_name": name, "reputation_score": scores.get(name, 0.5)}
        )

        with patch.object(handler, "_get_critique_store", return_value=None), \
             patch.object(handler, "_get_persona_manager", return_value=None):
            result = handler.handle(
                "/api/introspection/leaderboard", {}, mock_http_handler
            )

        assert result.status_code == 200
        body = parse_body(result)
        assert "leaderboard" in body
        assert "total_agents" in body
        assert body["total_agents"] == 5

    @patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True)
    @patch("aragora.server.handlers.introspection.get_agent_introspection")
    def test_leaderboard_sorted_descending(self, mock_get_intro, handler, mock_http_handler):
        """Test leaderboard is sorted by reputation descending."""
        scores = {"gemini": 0.3, "claude": 0.95, "codex": 0.7, "grok": 0.1, "deepseek": 0.5}
        mock_get_intro.side_effect = lambda name, **kw: MockSnapshot(
            name, data={"agent_name": name, "reputation_score": scores.get(name, 0.5)}
        )

        with patch.object(handler, "_get_critique_store", return_value=None), \
             patch.object(handler, "_get_persona_manager", return_value=None):
            result = handler.handle(
                "/api/introspection/leaderboard", {}, mock_http_handler
            )

        body = parse_body(result)
        lb_scores = [entry["reputation_score"] for entry in body["leaderboard"]]
        assert lb_scores == sorted(lb_scores, reverse=True)

    @patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True)
    @patch("aragora.server.handlers.introspection.get_agent_introspection")
    def test_leaderboard_default_limit(self, mock_get_intro, handler, mock_http_handler):
        """Test leaderboard uses default limit of 10."""
        mock_get_intro.side_effect = lambda name, **kw: MockSnapshot(name)

        with patch.object(handler, "_get_critique_store", return_value=None), \
             patch.object(handler, "_get_persona_manager", return_value=None):
            result = handler.handle(
                "/api/introspection/leaderboard", {}, mock_http_handler
            )

        body = parse_body(result)
        # Default agents is 5, which is less than limit of 10
        assert len(body["leaderboard"]) <= 10

    @patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True)
    @patch("aragora.server.handlers.introspection.get_agent_introspection")
    def test_leaderboard_custom_limit(self, mock_get_intro, handler, mock_http_handler):
        """Test leaderboard respects custom limit param."""
        mock_get_intro.side_effect = lambda name, **kw: MockSnapshot(name)

        with patch.object(handler, "_get_critique_store", return_value=None), \
             patch.object(handler, "_get_persona_manager", return_value=None):
            result = handler.handle(
                "/api/introspection/leaderboard",
                {"limit": ["3"]},
                mock_http_handler,
            )

        body = parse_body(result)
        assert len(body["leaderboard"]) <= 3

    @patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True)
    @patch("aragora.server.handlers.introspection.get_agent_introspection")
    def test_leaderboard_limit_capped_at_50(self, mock_get_intro, handler, mock_http_handler):
        """Test leaderboard limit is capped at 50."""
        mock_get_intro.side_effect = lambda name, **kw: MockSnapshot(name)

        with patch.object(handler, "_get_critique_store", return_value=None), \
             patch.object(handler, "_get_persona_manager", return_value=None):
            result = handler.handle(
                "/api/introspection/leaderboard",
                {"limit": ["999"]},
                mock_http_handler,
            )

        # The limit should be min(999, 50) = 50
        # With only 5 default agents, we get 5 entries
        body = parse_body(result)
        assert len(body["leaderboard"]) <= 50

    @patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", False)
    def test_leaderboard_module_not_available(self, handler, mock_http_handler):
        """Test leaderboard returns 503 when module unavailable."""
        result = handler.handle(
            "/api/introspection/leaderboard", {}, mock_http_handler
        )

        assert result.status_code == 503

    @patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True)
    @patch("aragora.server.handlers.introspection.get_agent_introspection")
    def test_leaderboard_partial_failures(self, mock_get_intro, handler, mock_http_handler):
        """Test leaderboard handles individual agent failures."""
        call_count = 0

        def side_effect(name, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise OSError("agent error")
            return MockSnapshot(name)

        mock_get_intro.side_effect = side_effect

        with patch.object(handler, "_get_critique_store", return_value=None), \
             patch.object(handler, "_get_persona_manager", return_value=None):
            result = handler.handle(
                "/api/introspection/leaderboard", {}, mock_http_handler
            )

        assert result.status_code == 200
        body = parse_body(result)
        assert body["total_agents"] == 4  # 5 - 1 failure


# ============================================================================
# Agent Availability Tests
# ============================================================================


class TestAgentAvailability:
    """Test GET /api/introspection/agents/availability endpoint."""

    @patch("aragora.server.handlers.introspection.IntrospectionHandler._get_agent_availability")
    def test_availability_success(self, mock_avail, handler, mock_http_handler):
        """Test successful availability check."""
        from aragora.server.handlers.utils.responses import json_response

        mock_avail.return_value = json_response({
            "available": ["claude", "gemini"],
            "missing": ["grok"],
            "details": {
                "claude": {"available": True, "required_vars": ["ANTHROPIC_API_KEY"],
                           "missing_vars": [], "available_via": "env"},
                "gemini": {"available": True, "required_vars": ["GEMINI_API_KEY"],
                           "missing_vars": [], "available_via": "env"},
                "grok": {"available": False, "required_vars": ["GROK_API_KEY"],
                         "missing_vars": ["GROK_API_KEY"], "available_via": ""},
            },
        })

        result = handler.handle(
            "/api/introspection/agents/availability", {}, mock_http_handler
        )

        assert result.status_code == 200

    def test_availability_import_error(self, handler, mock_http_handler):
        """Test availability when credential_validator module not importable."""
        with patch(
            "aragora.server.handlers.introspection.IntrospectionHandler._get_agent_availability"
        ) as mock_method:
            from aragora.server.handlers.utils.responses import error_response

            mock_method.return_value = error_response(
                "Credential validator not available", 503
            )
            result = handler.handle(
                "/api/introspection/agents/availability", {}, mock_http_handler
            )

        assert result.status_code == 503

    def test_availability_direct_import_error(self, handler, mock_http_handler):
        """Test availability via _get_agent_availability when import fails."""
        with patch.dict("sys.modules", {"aragora.agents.credential_validator": None}):
            result = handler._get_agent_availability()

        assert result.status_code == 503

    def test_availability_direct_success(self, handler, mock_http_handler):
        """Test direct _get_agent_availability with mocked credential validator."""
        mock_status_claude = MockCredentialStatus(
            is_available=True,
            required_vars=["ANTHROPIC_API_KEY"],
            missing_vars=[],
            available_via="env",
        )
        mock_status_grok = MockCredentialStatus(
            is_available=False,
            required_vars=["GROK_API_KEY"],
            missing_vars=["GROK_API_KEY"],
            available_via="",
        )

        with patch(
            "aragora.agents.credential_validator.get_agent_credential_status",
            return_value={"claude": mock_status_claude, "grok": mock_status_grok},
        ):
            result = handler._get_agent_availability()

        assert result.status_code == 200
        body = parse_body(result)
        assert "claude" in body["available"]
        assert "grok" in body["missing"]
        assert body["details"]["claude"]["available"] is True
        assert body["details"]["grok"]["available"] is False

    def test_availability_direct_error(self, handler, mock_http_handler):
        """Test _get_agent_availability handles errors in credential status."""
        with patch(
            "aragora.agents.credential_validator.get_agent_credential_status",
            side_effect=TypeError("bad status"),
        ):
            result = handler._get_agent_availability()

        assert result.status_code == 500


# ============================================================================
# Rate Limiting Tests
# ============================================================================


class TestRateLimiting:
    """Test rate limiting on introspection endpoints."""

    def test_rate_limit_exceeded(self, handler, mock_http_handler, reset_rate_limiter):
        """Test that rate limited requests return 429."""
        reset_rate_limiter.is_allowed.return_value = False

        result = handler.handle("/api/introspection/agents", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 429
        body = parse_body(result)
        assert "rate limit" in body["error"].lower()

    def test_rate_limit_applies_to_all_routes(self, handler, mock_http_handler, reset_rate_limiter):
        """Test rate limiting applies to all introspection routes."""
        reset_rate_limiter.is_allowed.return_value = False

        for path in [
            "/api/introspection/all",
            "/api/introspection/leaderboard",
            "/api/introspection/agents",
            "/api/introspection/agents/availability",
            "/api/introspection/agents/claude",
        ]:
            result = handler.handle(path, {}, mock_http_handler)
            assert result is not None
            assert result.status_code == 429, f"Rate limit not enforced for {path}"

    def test_rate_limit_uses_client_ip(self, handler, reset_rate_limiter):
        """Test that rate limiter uses client IP for identification."""
        mock_handler_obj = MockHandler()
        mock_handler_obj.client_address = ("10.0.0.1", 54321)

        with patch.object(handler, "_get_critique_store", return_value=None):
            handler.handle("/api/introspection/agents", {}, mock_handler_obj)

        reset_rate_limiter.is_allowed.assert_called()


# ============================================================================
# Version Prefix Tests
# ============================================================================


class TestVersionPrefix:
    """Test version prefix stripping."""

    def test_handle_with_v1_prefix(self, handler, mock_http_handler):
        """Test handle strips /api/v1/ prefix."""
        with patch.object(handler, "_get_critique_store", return_value=None):
            result = handler.handle(
                "/api/v1/introspection/agents", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 200

    def test_handle_with_v2_prefix(self, handler, mock_http_handler):
        """Test handle strips /api/v2/ prefix."""
        with patch.object(handler, "_get_critique_store", return_value=None):
            result = handler.handle(
                "/api/v2/introspection/agents", {}, mock_http_handler
            )

        assert result is not None
        assert result.status_code == 200


# ============================================================================
# Critique Store Tests
# ============================================================================


class TestCritiqueStore:
    """Test _get_critique_store internal method."""

    def test_returns_none_when_module_unavailable(self, handler):
        """Test returns None when CritiqueStore not importable."""
        with patch(
            "aragora.server.handlers.introspection.CRITIQUE_STORE_AVAILABLE", False
        ):
            result = handler._get_critique_store()
        assert result is None

    def test_returns_none_when_no_nomic_dir(self, handler):
        """Test returns None when nomic_dir not configured."""
        with patch(
            "aragora.server.handlers.introspection.CRITIQUE_STORE_AVAILABLE", True
        ), patch.object(handler, "get_nomic_dir", return_value=None):
            result = handler._get_critique_store()
        assert result is None

    def test_returns_none_when_db_missing(self, handler, tmp_path):
        """Test returns None when debates.db doesn't exist."""
        with patch(
            "aragora.server.handlers.introspection.CRITIQUE_STORE_AVAILABLE", True
        ), patch.object(handler, "get_nomic_dir", return_value=tmp_path):
            result = handler._get_critique_store()
        assert result is None

    def test_returns_store_when_db_exists(self, handler, tmp_path):
        """Test returns CritiqueStore when debates.db exists."""
        db_path = tmp_path / "debates.db"
        db_path.touch()

        mock_store = MagicMock()
        with patch(
            "aragora.server.handlers.introspection.CRITIQUE_STORE_AVAILABLE", True
        ), patch(
            "aragora.server.handlers.introspection.CritiqueStore", return_value=mock_store
        ), patch.object(handler, "get_nomic_dir", return_value=tmp_path):
            result = handler._get_critique_store()

        assert result is mock_store


# ============================================================================
# Persona Manager Tests
# ============================================================================


class TestPersonaManager:
    """Test _get_persona_manager internal method."""

    def test_returns_none_when_module_unavailable(self, handler):
        """Test returns None when PersonaManager not importable."""
        with patch(
            "aragora.server.handlers.introspection.PERSONA_MANAGER_AVAILABLE", False
        ):
            result = handler._get_persona_manager()
        assert result is None

    def test_returns_none_when_no_nomic_dir(self, handler):
        """Test returns None when nomic_dir not configured."""
        with patch(
            "aragora.server.handlers.introspection.PERSONA_MANAGER_AVAILABLE", True
        ), patch.object(handler, "get_nomic_dir", return_value=None):
            result = handler._get_persona_manager()
        assert result is None

    def test_returns_none_when_persona_db_missing(self, handler, tmp_path):
        """Test returns None when persona DB doesn't exist."""
        with patch(
            "aragora.server.handlers.introspection.PERSONA_MANAGER_AVAILABLE", True
        ), patch.object(handler, "get_nomic_dir", return_value=tmp_path), \
             patch(
                 "aragora.server.handlers.introspection.get_db_path",
                 return_value=tmp_path / "personas.db",
             ):
            result = handler._get_persona_manager()
        assert result is None

    def test_returns_manager_when_db_exists(self, handler, tmp_path):
        """Test returns PersonaManager when persona DB exists."""
        persona_db = tmp_path / "personas.db"
        persona_db.touch()

        mock_manager = MagicMock()
        with patch(
            "aragora.server.handlers.introspection.PERSONA_MANAGER_AVAILABLE", True
        ), patch(
            "aragora.server.handlers.introspection.PersonaManager", return_value=mock_manager
        ), patch.object(handler, "get_nomic_dir", return_value=tmp_path), \
             patch(
                 "aragora.server.handlers.introspection.get_db_path",
                 return_value=persona_db,
             ):
            result = handler._get_persona_manager()

        assert result is mock_manager


# ============================================================================
# Known Agents Tests
# ============================================================================


class TestGetKnownAgents:
    """Test _get_known_agents internal method."""

    def test_returns_defaults_when_no_store(self, handler):
        """Test falls back to DEFAULT_AGENTS when store is None."""
        result = handler._get_known_agents(None)
        assert result == handler.DEFAULT_AGENTS

    def test_returns_agents_from_store(self, handler):
        """Test returns agents from store reputations."""
        mock_store = MagicMock()
        mock_store.get_all_reputations.return_value = [
            MockReputation("alpha"),
            MockReputation("beta"),
        ]
        result = handler._get_known_agents(mock_store)
        assert result == ["alpha", "beta"]

    def test_returns_defaults_on_empty_reputations(self, handler):
        """Test falls back to defaults when store returns empty list."""
        mock_store = MagicMock()
        mock_store.get_all_reputations.return_value = []
        result = handler._get_known_agents(mock_store)
        assert result == handler.DEFAULT_AGENTS

    def test_returns_defaults_on_store_error(self, handler):
        """Test falls back to defaults on store KeyError."""
        mock_store = MagicMock()
        mock_store.get_all_reputations.side_effect = KeyError("missing")
        result = handler._get_known_agents(mock_store)
        assert result == handler.DEFAULT_AGENTS

    def test_returns_defaults_on_oserror(self, handler):
        """Test falls back to defaults on OSError."""
        mock_store = MagicMock()
        mock_store.get_all_reputations.side_effect = OSError("disk")
        result = handler._get_known_agents(mock_store)
        assert result == handler.DEFAULT_AGENTS

    def test_returns_defaults_on_type_error(self, handler):
        """Test falls back to defaults on TypeError."""
        mock_store = MagicMock()
        mock_store.get_all_reputations.side_effect = TypeError("bad type")
        result = handler._get_known_agents(mock_store)
        assert result == handler.DEFAULT_AGENTS


# ============================================================================
# Security Tests
# ============================================================================


class TestSecurityValidation:
    """Test security-related input validation."""

    def test_path_traversal_in_agent_name(self, handler, mock_http_handler):
        """Test path traversal attempt in agent name."""
        result = handler.handle(
            "/api/introspection/agents/../../etc", {}, mock_http_handler
        )
        # Should be rejected
        if result is not None:
            assert result.status_code == 400

    def test_special_chars_in_agent_name(self, handler, mock_http_handler):
        """Test special characters in agent name are rejected."""
        for bad_name in ["<script>", "agent;rm", "agent&cmd", "agent|pipe"]:
            result = handler.handle(
                f"/api/introspection/agents/{bad_name}", {}, mock_http_handler
            )
            if result is not None:
                assert result.status_code == 400, f"Should reject agent name: {bad_name}"

    def test_sql_injection_in_agent_name(self, handler, mock_http_handler):
        """Test SQL injection attempt in agent name."""
        result = handler.handle(
            "/api/introspection/agents/'; DROP TABLE--", {}, mock_http_handler
        )
        if result is not None:
            assert result.status_code == 400

    def test_null_bytes_in_agent_name(self, handler, mock_http_handler):
        """Test null byte injection in agent name."""
        result = handler.handle(
            "/api/introspection/agents/agent%00evil", {}, mock_http_handler
        )
        # URL-decoded %00 would be null byte - path matching may reject this
        if result is not None:
            assert result.status_code in (400, 404)

    def test_unicode_agent_name(self, handler, mock_http_handler):
        """Test unicode characters in agent name are rejected."""
        result = handler.handle(
            "/api/introspection/agents/\u00e9\u00e8\u00ea", {}, mock_http_handler
        )
        if result is not None:
            assert result.status_code == 400


# ============================================================================
# Handler Context Tests
# ============================================================================


class TestHandlerContext:
    """Test handler initialization and context."""

    def test_default_context(self):
        """Test handler initializes with empty context."""
        from aragora.server.handlers.introspection import IntrospectionHandler

        handler = IntrospectionHandler()
        assert handler.ctx == {}

    def test_custom_context(self):
        """Test handler uses provided context."""
        from aragora.server.handlers.introspection import IntrospectionHandler

        ctx = {"nomic_dir": "/tmp/test"}
        handler = IntrospectionHandler(ctx)
        assert handler.ctx == ctx

    def test_default_agents_constant(self):
        """Test DEFAULT_AGENTS contains expected values."""
        from aragora.server.handlers.introspection import IntrospectionHandler

        agents = IntrospectionHandler.DEFAULT_AGENTS
        assert len(agents) == 5
        assert "claude" in agents
        assert "gemini" in agents
        assert "codex" in agents
        assert "grok" in agents
        assert "deepseek" in agents

    def test_routes_defined(self):
        """Test ROUTES class attribute is populated."""
        from aragora.server.handlers.introspection import IntrospectionHandler

        assert len(IntrospectionHandler.ROUTES) >= 4


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test various edge cases."""

    def test_leaderboard_limit_zero(self, handler, mock_http_handler):
        """Test leaderboard with limit=0."""
        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True), \
             patch("aragora.server.handlers.introspection.get_agent_introspection") as mock_intro:
            mock_intro.side_effect = lambda name, **kw: MockSnapshot(name)

            with patch.object(handler, "_get_critique_store", return_value=None), \
                 patch.object(handler, "_get_persona_manager", return_value=None):
                result = handler.handle(
                    "/api/introspection/leaderboard",
                    {"limit": ["0"]},
                    mock_http_handler,
                )

        assert result.status_code == 200
        body = parse_body(result)
        assert len(body["leaderboard"]) == 0

    def test_leaderboard_negative_limit(self, handler, mock_http_handler):
        """Test leaderboard with negative limit defaults gracefully."""
        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True), \
             patch("aragora.server.handlers.introspection.get_agent_introspection") as mock_intro:
            mock_intro.side_effect = lambda name, **kw: MockSnapshot(name)

            with patch.object(handler, "_get_critique_store", return_value=None), \
                 patch.object(handler, "_get_persona_manager", return_value=None):
                result = handler.handle(
                    "/api/introspection/leaderboard",
                    {"limit": ["-5"]},
                    mock_http_handler,
                )

        # Negative limit should still work (empty list or default)
        assert result.status_code == 200

    def test_leaderboard_non_numeric_limit(self, handler, mock_http_handler):
        """Test leaderboard with non-numeric limit uses default."""
        with patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True), \
             patch("aragora.server.handlers.introspection.get_agent_introspection") as mock_intro:
            mock_intro.side_effect = lambda name, **kw: MockSnapshot(name)

            with patch.object(handler, "_get_critique_store", return_value=None), \
                 patch.object(handler, "_get_persona_manager", return_value=None):
                result = handler.handle(
                    "/api/introspection/leaderboard",
                    {"limit": ["abc"]},
                    mock_http_handler,
                )

        # Non-numeric should fall back to default 10
        assert result.status_code == 200

    def test_handle_with_none_handler(self, handler):
        """Test handle with None HTTP handler (rate limiter gets 'unknown' IP)."""
        # get_client_ip handles None by returning "unknown"
        with patch.object(handler, "_get_critique_store", return_value=None):
            result = handler.handle("/api/introspection/agents", {}, None)
        assert result is not None
        assert result.status_code == 200

    def test_availability_route_does_not_match_agent(self, handler, mock_http_handler):
        """Test that /availability is matched as a specific route, not as agent name."""
        # The route /api/introspection/agents/availability should NOT be treated
        # as an agent named "availability"
        with patch.object(handler, "_get_agent_availability") as mock_avail:
            from aragora.server.handlers.utils.responses import json_response
            mock_avail.return_value = json_response({"available": [], "missing": [], "details": {}})

            result = handler.handle(
                "/api/introspection/agents/availability", {}, mock_http_handler
            )

        mock_avail.assert_called_once()

    @patch("aragora.server.handlers.introspection.INTROSPECTION_AVAILABLE", True)
    @patch("aragora.server.handlers.introspection.get_agent_introspection")
    def test_all_introspection_oserror(self, mock_get_intro, handler, mock_http_handler):
        """Test _get_all_introspection handles individual OSError per agent."""
        def side_effect(name, **kw):
            if name == "claude":
                raise OSError("file not found")
            return MockSnapshot(name)

        mock_get_intro.side_effect = side_effect

        with patch.object(handler, "_get_critique_store", return_value=None), \
             patch.object(handler, "_get_persona_manager", return_value=None):
            result = handler.handle("/api/introspection/all", {}, mock_http_handler)

        assert result.status_code == 200
        body = parse_body(result)
        assert "claude" not in body["agents"]

    def test_list_agents_store_no_get_agent_reputation(self, handler, mock_http_handler):
        """Test list agents when store lacks get_agent_reputation method."""
        mock_store = MagicMock(spec=[])  # No methods at all
        mock_store.get_all_reputations = MagicMock(return_value=[
            MockReputation("test_agent"),
        ])

        with patch.object(handler, "_get_critique_store", return_value=mock_store):
            result = handler.handle("/api/introspection/agents", {}, mock_http_handler)

        assert result.status_code == 200
        body = parse_body(result)
        # Agent should be listed but without reputation_score
        assert body["count"] == 1
        agent = body["agents"][0]
        assert agent["name"] == "test_agent"
        assert "reputation_score" not in agent
