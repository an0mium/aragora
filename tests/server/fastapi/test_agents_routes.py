"""
Tests for FastAPI agent route endpoints.

Covers:
- List agents
- Get agent details
- Get agent leaderboard
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from aragora.server.fastapi import create_app


@pytest.fixture
def app():
    """Create a test FastAPI app."""
    return create_app()


@pytest.fixture
def mock_elo_system():
    """Create a mock ELO system with sample data."""
    elo = MagicMock()
    elo.get_leaderboard = MagicMock(
        return_value=[
            {
                "name": "claude",
                "elo": 1650.5,
                "matches": 120,
                "wins": 80,
                "losses": 40,
            },
            {
                "name": "codex",
                "elo": 1580.2,
                "matches": 95,
                "wins": 55,
                "losses": 40,
            },
            {
                "name": "gemini",
                "elo": 1520.0,
                "matches": 75,
                "wins": 40,
                "losses": 35,
            },
        ]
    )
    elo.get_agent = MagicMock(return_value=None)
    return elo


@pytest.fixture
def client(app, mock_elo_system):
    """Create a test client with mocked context."""
    app.state.context = {
        "storage": MagicMock(),
        "elo_system": mock_elo_system,
        "user_store": None,
        "rbac_checker": MagicMock(),
        "decision_service": MagicMock(),
    }
    return TestClient(app, raise_server_exceptions=False)


class TestListAgents:
    """Tests for GET /api/v2/agents."""

    def test_list_agents_returns_200(self, client):
        """List agents returns 200 with agents."""
        response = client.get("/api/v2/agents")
        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert "total" in data
        assert data["total"] > 0

    def test_list_agents_without_stats(self, client):
        """List agents without stats returns agent names only."""
        response = client.get("/api/v2/agents")
        assert response.status_code == 200
        data = response.json()
        assert len(data["agents"]) == 3
        # Without stats, elo defaults to 1500
        for agent in data["agents"]:
            assert "name" in agent

    def test_list_agents_with_stats(self, client):
        """List agents with stats includes ELO data."""
        response = client.get("/api/v2/agents?include_stats=true")
        assert response.status_code == 200
        data = response.json()
        assert len(data["agents"]) == 3
        # With stats, should have real ELO values
        claude = next(a for a in data["agents"] if a["name"] == "claude")
        assert claude["elo"] == 1650.5
        assert claude["matches"] == 120

    def test_list_agents_fallback_when_no_elo(self, app):
        """List agents falls back to config when ELO unavailable."""
        app.state.context = {
            "storage": MagicMock(),
            "elo_system": None,
            "user_store": None,
            "rbac_checker": MagicMock(),
            "decision_service": MagicMock(),
        }
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/api/v2/agents")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] > 0


class TestGetAgent:
    """Tests for GET /api/v2/agents/{agent_id}."""

    def test_get_agent_from_leaderboard(self, client):
        """Get agent found in leaderboard returns details."""
        response = client.get("/api/v2/agents/claude")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "claude"
        assert data["elo"] == 1650.5
        assert data["matches"] == 120

    def test_get_agent_not_in_elo_but_known(self, client, mock_elo_system):
        """Get agent not in ELO but in known types returns basic info."""
        # Return empty leaderboard for search
        mock_elo_system.get_leaderboard.return_value = []

        with patch(
            "aragora.server.fastapi.routes.agents._get_known_agents",
            return_value=["claude", "codex", "gemini", "grok", "deepseek"],
        ):
            response = client.get("/api/v2/agents/grok")
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "grok"
            assert data["elo"] == 1500.0  # Default ELO

    def test_get_agent_not_found(self, client, mock_elo_system):
        """Get unknown agent returns 404."""
        mock_elo_system.get_leaderboard.return_value = []

        with patch(
            "aragora.server.fastapi.routes.agents._get_known_agents",
            return_value=["claude", "codex"],
        ):
            response = client.get("/api/v2/agents/nonexistent-agent")
            assert response.status_code == 404


class TestLeaderboard:
    """Tests for GET /api/v2/agents/leaderboard."""

    def test_leaderboard_returns_200(self, client):
        """Leaderboard returns ranked agents."""
        response = client.get("/api/v2/agents/leaderboard")
        assert response.status_code == 200
        data = response.json()
        assert "leaderboard" in data
        assert "total" in data
        assert data["total"] == 3

    def test_leaderboard_has_ranks(self, client):
        """Leaderboard entries have sequential ranks."""
        response = client.get("/api/v2/agents/leaderboard")
        assert response.status_code == 200
        data = response.json()
        ranks = [entry["rank"] for entry in data["leaderboard"]]
        assert ranks == [1, 2, 3]

    def test_leaderboard_ordered_by_elo(self, client):
        """Leaderboard is ordered by ELO (highest first)."""
        response = client.get("/api/v2/agents/leaderboard")
        assert response.status_code == 200
        data = response.json()
        elos = [entry["elo"] for entry in data["leaderboard"]]
        assert elos == sorted(elos, reverse=True)

    def test_leaderboard_with_limit(self, client, mock_elo_system):
        """Leaderboard passes limit to ELO system."""
        response = client.get("/api/v2/agents/leaderboard?limit=2")
        assert response.status_code == 200
        # Verify the limit was passed to the ELO system
        mock_elo_system.get_leaderboard.assert_called_with(limit=2)

    def test_leaderboard_with_domain(self, client):
        """Leaderboard supports domain filter."""
        response = client.get("/api/v2/agents/leaderboard?domain=security")
        assert response.status_code == 200
        data = response.json()
        assert data["domain"] == "security"

    def test_leaderboard_includes_win_rate(self, client):
        """Leaderboard entries include computed win rate."""
        response = client.get("/api/v2/agents/leaderboard")
        assert response.status_code == 200
        data = response.json()
        for entry in data["leaderboard"]:
            assert "win_rate" in entry
            if entry["matches"] > 0:
                expected_win_rate = round(entry["wins"] / entry["matches"], 3)
                assert entry["win_rate"] == expected_win_rate

    def test_leaderboard_empty_when_no_elo(self, app):
        """Leaderboard is empty when ELO system is unavailable."""
        app.state.context = {
            "storage": MagicMock(),
            "elo_system": None,
            "user_store": None,
            "rbac_checker": MagicMock(),
            "decision_service": MagicMock(),
        }
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/api/v2/agents/leaderboard")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["leaderboard"] == []

    def test_leaderboard_limit_validation(self, client):
        """Leaderboard limit must be between 1 and 100."""
        response = client.get("/api/v2/agents/leaderboard?limit=0")
        assert response.status_code == 422

        response = client.get("/api/v2/agents/leaderboard?limit=101")
        assert response.status_code == 422


class TestAgentLeaderboardRouteOrder:
    """Tests to verify /agents/leaderboard routes before /agents/{agent_id}."""

    def test_leaderboard_not_treated_as_agent_id(self, client):
        """Ensure /agents/leaderboard is not matched as /agents/{agent_id}='leaderboard'."""
        response = client.get("/api/v2/agents/leaderboard")
        assert response.status_code == 200
        data = response.json()
        # Should be a leaderboard response, not an agent detail response
        assert "leaderboard" in data
        assert "name" not in data


class TestRankings:
    """Tests for GET /api/v2/agents/rankings (alias for leaderboard)."""

    def test_rankings_returns_200(self, client):
        """Rankings returns ranked agents (alias for leaderboard)."""
        response = client.get("/api/v2/agents/rankings")
        assert response.status_code == 200
        data = response.json()
        assert "leaderboard" in data
        assert "total" in data
        assert data["total"] == 3

    def test_rankings_has_same_data_as_leaderboard(self, client):
        """Rankings returns the same data as leaderboard."""
        rankings_resp = client.get("/api/v2/agents/rankings")
        leaderboard_resp = client.get("/api/v2/agents/leaderboard")
        assert rankings_resp.json() == leaderboard_resp.json()

    def test_rankings_with_domain(self, client):
        """Rankings supports domain filter."""
        response = client.get("/api/v2/agents/rankings?domain=security")
        assert response.status_code == 200
        data = response.json()
        assert data["domain"] == "security"

    def test_rankings_not_treated_as_agent_id(self, client):
        """Ensure /agents/rankings is not matched as /agents/{agent_id}='rankings'."""
        response = client.get("/api/v2/agents/rankings")
        assert response.status_code == 200
        data = response.json()
        assert "leaderboard" in data
        assert "name" not in data


class TestRegisterAgent:
    """Tests for POST /api/v2/agents."""

    def _override_auth(self, client):
        """Override auth for write operations."""
        from aragora.server.fastapi.dependencies.auth import require_authenticated
        from aragora.rbac.models import AuthorizationContext

        auth_ctx = AuthorizationContext(
            user_id="user-1",
            org_id="org-1",
            workspace_id="ws-1",
            roles={"admin"},
            permissions={"agents:write"},
        )
        client.app.dependency_overrides[require_authenticated] = lambda: auth_ctx

    def test_register_returns_201(self, client):
        """Register creates a new agent."""
        self._override_auth(client)

        with patch(
            "aragora.server.fastapi.routes.agents._get_known_agents",
            return_value=["claude", "codex"],
        ):
            response = client.post(
                "/api/v2/agents",
                json={"name": "custom-agent", "type": "custom"},
            )
        client.app.dependency_overrides.clear()

        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["agent"]["name"] == "custom-agent"
        assert data["agent"]["type"] == "custom"
        assert data["agent"]["elo"] == 1500.0

    def test_register_duplicate_returns_409(self, client):
        """Register existing agent returns 409."""
        self._override_auth(client)

        with patch(
            "aragora.server.fastapi.routes.agents._get_known_agents",
            return_value=["claude", "codex", "gemini"],
        ):
            response = client.post(
                "/api/v2/agents",
                json={"name": "claude", "type": "api"},
            )
        client.app.dependency_overrides.clear()

        assert response.status_code == 409

    def test_register_requires_name(self, client):
        """Register without name returns 422."""
        self._override_auth(client)

        response = client.post(
            "/api/v2/agents",
            json={"type": "custom"},
        )
        client.app.dependency_overrides.clear()

        assert response.status_code == 422

    def test_register_requires_auth(self, client):
        """Register without auth returns 401."""
        response = client.post(
            "/api/v2/agents",
            json={"name": "unauthorized-agent"},
        )
        assert response.status_code == 401
