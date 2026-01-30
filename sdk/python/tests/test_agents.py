"""Tests for Agents namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient


class TestAgentsList:
    """Tests for listing agents."""

    def test_list_agents(self) -> None:
        """List all available agents."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "agents": [
                    {"name": "claude", "provider": "anthropic"},
                    {"name": "gpt-4", "provider": "openai"},
                ]
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.agents.list()

            mock_request.assert_called_once_with("GET", "/api/v1/agents")
            assert len(result["agents"]) == 2
            client.close()


class TestAgentsGet:
    """Tests for getting agent details."""

    def test_get_agent(self) -> None:
        """Get details for a specific agent."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "name": "claude",
                "provider": "anthropic",
                "capabilities": ["reasoning", "coding"],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.agents.get("claude")

            mock_request.assert_called_once_with("GET", "/api/v1/agents/claude")
            assert result["name"] == "claude"
            client.close()


class TestAgentsPerformance:
    """Tests for agent performance metrics."""

    def test_get_performance(self) -> None:
        """Get performance metrics for an agent."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "elo_rating": 1650,
                "win_rate": 0.65,
                "debates_participated": 100,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.agents.get_performance("claude")

            mock_request.assert_called_once_with("GET", "/api/v1/agents/claude/performance")
            assert result["elo_rating"] == 1650
            assert result["win_rate"] == 0.65
            client.close()


class TestAgentsCalibration:
    """Tests for agent calibration data."""

    def test_get_calibration(self) -> None:
        """Get calibration data for an agent."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "calibration_score": 0.85,
                "overconfidence_ratio": 0.12,
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.agents.get_calibration("claude")

            mock_request.assert_called_once_with("GET", "/api/v1/agents/claude/calibration")
            assert result["calibration_score"] == 0.85
            client.close()

    def test_get_calibration_curve(self) -> None:
        """Get calibration curve data."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "points": [
                    {"confidence": 0.5, "accuracy": 0.48},
                    {"confidence": 0.9, "accuracy": 0.85},
                ]
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.agents.get_calibration_curve("claude")

            mock_request.assert_called_once_with("GET", "/api/v1/agent/claude/calibration-curve")
            assert len(result["points"]) == 2
            client.close()

    def test_get_calibration_summary(self) -> None:
        """Get calibration summary."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"well_calibrated": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.get_calibration_summary("claude")

            mock_request.assert_called_once_with("GET", "/api/v1/agent/claude/calibration-summary")
            client.close()


class TestAgentsRelationships:
    """Tests for agent relationship data."""

    def test_get_relationships(self) -> None:
        """Get relationship data for an agent."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "allies": ["gemini"],
                "rivals": ["gpt-4"],
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.get_relationships("claude")

            mock_request.assert_called_once_with("GET", "/api/v1/agents/claude/relationships")
            client.close()

    def test_get_allies(self) -> None:
        """Get agents that frequently agree."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"allies": ["gemini", "mistral"]}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.get_allies("claude")

            mock_request.assert_called_once_with("GET", "/api/v1/agent/claude/allies")
            client.close()

    def test_get_rivals(self) -> None:
        """Get agents that frequently disagree."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"rivals": ["gpt-4"]}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.get_rivals("claude")

            mock_request.assert_called_once_with("GET", "/api/v1/agent/claude/rivals")
            client.close()

    def test_get_network(self) -> None:
        """Get agent network graph."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"nodes": [], "edges": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.get_network("claude")

            mock_request.assert_called_once_with("GET", "/api/v1/agent/claude/network")
            client.close()


class TestAgentsHistory:
    """Tests for agent history."""

    def test_get_history_default_pagination(self) -> None:
        """Get agent history with default pagination."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"debates": [], "total": 50}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.get_history("claude")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/agents/claude/history",
                params={"limit": 20, "offset": 0},
            )
            client.close()

    def test_get_history_custom_pagination(self) -> None:
        """Get agent history with custom pagination."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"debates": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.get_history("claude", limit=50, offset=25)

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/agents/claude/history",
                params={"limit": 50, "offset": 25},
            )
            client.close()

    def test_get_flips(self) -> None:
        """Get instances where agent changed position."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"flips": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.get_flips("claude", limit=10)

            mock_request.assert_called_once_with(
                "GET", "/api/v1/agent/claude/flips", params={"limit": 10}
            )
            client.close()

    def test_get_moments(self) -> None:
        """Get agent's notable debate moments."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"moments": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.get_moments("claude", limit=5)

            mock_request.assert_called_once_with(
                "GET", "/api/v1/agent/claude/moments", params={"limit": 5}
            )
            client.close()


class TestAgentsComparison:
    """Tests for agent comparison."""

    def test_compare_agents(self) -> None:
        """Compare two agents' performance."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "agent1": {"name": "claude", "elo": 1650},
                "agent2": {"name": "gpt-4", "elo": 1600},
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.compare("claude", "gpt-4")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/agents/compare",
                params={"agent1": "claude", "agent2": "gpt-4"},
            )
            client.close()

    def test_compare_agents_via_singular_endpoint(self) -> None:
        """Compare agents using /agent/compare endpoint."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"comparison": {}}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.compare_agents("claude", "gpt-4")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/agent/compare",
                params={"agent1": "claude", "agent2": "gpt-4"},
            )
            client.close()


class TestAgentsTeamSelection:
    """Tests for team selection."""

    def test_select_team_default(self) -> None:
        """Select optimal team with defaults."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "team": ["claude", "gpt-4", "gemini"],
                "rationale": "Balanced skill set",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.agents.select_team(task="Complex reasoning task")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/agents/select-team",
                json={
                    "task": "Complex reasoning task",
                    "team_size": 3,
                    "strategy": "balanced",
                },
            )
            assert len(result["team"]) == 3
            client.close()

    def test_select_team_custom(self) -> None:
        """Select team with custom parameters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"team": ["claude", "gpt-4", "gemini", "mistral", "llama"]}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.select_team(
                task="Adversarial debate", team_size=5, strategy="competitive"
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/agents/select-team",
                json={
                    "task": "Adversarial debate",
                    "team_size": 5,
                    "strategy": "competitive",
                },
            )
            client.close()


class TestAgentsProfile:
    """Tests for agent profile operations."""

    def test_get_profile(self) -> None:
        """Get agent's full profile."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "name": "claude",
                "bio": "AI assistant by Anthropic",
            }

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.agents.get_profile("claude")

            mock_request.assert_called_once_with("GET", "/api/v1/agent/claude/profile")
            assert result["name"] == "claude"
            client.close()

    def test_get_persona(self) -> None:
        """Get agent's persona configuration."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"persona": "analytical"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.get_persona("claude")

            mock_request.assert_called_once_with("GET", "/api/v1/agent/claude/persona")
            client.close()

    def test_delete_persona(self) -> None:
        """Delete agent's custom persona."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"deleted": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.delete_persona("claude")

            mock_request.assert_called_once_with("DELETE", "/api/v1/agent/claude/persona")
            client.close()

    def test_get_identity_prompt(self) -> None:
        """Get agent's identity prompt."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"prompt": "You are Claude..."}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.get_identity_prompt("claude")

            mock_request.assert_called_once_with("GET", "/api/v1/agent/claude/identity-prompt")
            client.close()


class TestAgentsAnalytics:
    """Tests for agent analytics."""

    def test_get_accuracy(self) -> None:
        """Get agent's accuracy metrics."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"accuracy": 0.92}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.agents.get_accuracy("claude")

            mock_request.assert_called_once_with("GET", "/api/v1/agent/claude/accuracy")
            assert result["accuracy"] == 0.92
            client.close()

    def test_get_consistency(self) -> None:
        """Get agent's consistency metrics."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"consistency_score": 0.88}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.get_consistency("claude")

            mock_request.assert_called_once_with("GET", "/api/v1/agent/claude/consistency")
            client.close()

    def test_get_reputation(self) -> None:
        """Get agent's reputation score."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"reputation": 95}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.get_reputation("claude")

            mock_request.assert_called_once_with("GET", "/api/v1/agent/claude/reputation")
            client.close()

    def test_get_domains(self) -> None:
        """Get domains the agent specializes in."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"domains": ["coding", "reasoning", "writing"]}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.agents.get_domains("claude")

            mock_request.assert_called_once_with("GET", "/api/v1/agent/claude/domains")
            assert "coding" in result["domains"]
            client.close()


class TestAsyncAgents:
    """Tests for async agents API."""

    @pytest.mark.asyncio
    async def test_async_list_agents(self) -> None:
        """List agents asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"agents": []}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.agents.list()

                mock_request.assert_called_once_with("GET", "/api/v1/agents")

    @pytest.mark.asyncio
    async def test_async_get_agent(self) -> None:
        """Get agent details asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"name": "claude"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.agents.get("claude")

                mock_request.assert_called_once_with("GET", "/api/v1/agents/claude")
                assert result["name"] == "claude"

    @pytest.mark.asyncio
    async def test_async_get_performance(self) -> None:
        """Get agent performance asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"elo_rating": 1650}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.agents.get_performance("claude")

                mock_request.assert_called_once_with("GET", "/api/v1/agents/claude/performance")
                assert result["elo_rating"] == 1650

    @pytest.mark.asyncio
    async def test_async_select_team(self) -> None:
        """Select team asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"team": ["claude", "gpt-4"]}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.agents.select_team(task="Test task", team_size=2)

                assert len(result["team"]) == 2
