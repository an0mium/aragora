"""Tests for Agents namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


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


class TestAgentsCalibration:
    """Tests for agent calibration data."""

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


# =========================================================================
# Health & Availability
# =========================================================================


class TestAgentsHealth:
    """Tests for agent health and availability."""

    def test_list_health(self) -> None:
        """List health status for all agents."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"agents": [{"name": "claude", "healthy": True}]}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.agents.list_health()

            mock_request.assert_called_once_with("GET", "/api/agents/health")
            assert result["agents"][0]["healthy"] is True
            client.close()

    def test_list_availability(self) -> None:
        """List availability for all agents."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"agents": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.list_availability()

            mock_request.assert_called_once_with("GET", "/api/agents/availability")
            client.close()

    def test_list_local_agents(self) -> None:
        """List locally available agents."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"agents": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.list_local_agents()

            mock_request.assert_called_once_with("GET", "/api/agents/local")
            client.close()

    def test_get_local_status(self) -> None:
        """Get local agent provider status."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"ollama": "running"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.get_local_status()

            mock_request.assert_called_once_with("GET", "/api/agents/local/status")
            client.close()


# =========================================================================
# Agent Details (Extended)
# =========================================================================


class TestAgentsDetailsExtended:
    """Tests for extended agent detail methods."""

    def test_get_head_to_head(self) -> None:
        """Get head-to-head stats."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"wins": 15, "losses": 8}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.agents.get_head_to_head("claude", "gpt-4")

            mock_request.assert_called_once_with("GET", "/api/agent/claude/head-to-head/gpt-4")
            assert result["wins"] == 15
            client.close()

    def test_get_opponent_briefing(self) -> None:
        """Get opponent briefing."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"strategy": "analytical"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.get_opponent_briefing("claude", "gemini")

            mock_request.assert_called_once_with(
                "GET", "/api/agent/claude/opponent-briefing/gemini"
            )
            client.close()

    def test_get_positions(self) -> None:
        """Get agent positions."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"positions": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.get_positions("claude")

            mock_request.assert_called_once_with("GET", "/api/agent/claude/positions")
            client.close()

    def test_get_introspection(self) -> None:
        """Get agent introspection data."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"self_awareness": {}}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.get_introspection("claude")

            mock_request.assert_called_once_with("GET", "/api/agent/claude/introspect")
            client.close()


# =========================================================================
# Leaderboard & Analytics (Extended)
# =========================================================================


class TestAgentsLeaderboard:
    """Tests for leaderboard and analytics methods."""

    def test_get_leaderboard_default(self) -> None:
        """Get leaderboard with default view."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"rankings": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.get_leaderboard()

            mock_request.assert_called_once_with(
                "GET", "/api/leaderboard", params={"view": "overall"}
            )
            client.close()

    def test_get_leaderboard_custom_view(self) -> None:
        """Get leaderboard with specific view."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"rankings": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.get_leaderboard(view="coding")

            mock_request.assert_called_once_with(
                "GET", "/api/leaderboard", params={"view": "coding"}
            )
            client.close()

    def test_get_recent_matches(self) -> None:
        """Get recent matches."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"matches": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.get_recent_matches(limit=5)

            mock_request.assert_called_once_with("GET", "/api/matches/recent", params={"limit": 5})
            client.close()

    def test_get_recent_flips(self) -> None:
        """Get recent flips across agents."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"flips": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.get_recent_flips(limit=10)

            mock_request.assert_called_once_with("GET", "/api/flips/recent", params={"limit": 10})
            client.close()

    def test_get_flips_summary(self) -> None:
        """Get aggregate flip stats."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"total_flips": 42}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.agents.get_flips_summary()

            mock_request.assert_called_once_with("GET", "/api/flips/summary")
            assert result["total_flips"] == 42
            client.close()

    def test_get_calibration_leaderboard(self) -> None:
        """Get calibration leaderboard."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"rankings": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.get_calibration_leaderboard()

            mock_request.assert_called_once_with("GET", "/api/calibration/leaderboard")
            client.close()

    def test_get_rankings(self) -> None:
        """Get rankings with filters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"rankings": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.agents.get_rankings(domain="coding", period="weekly", limit=10)

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/rankings",
                params={"limit": 10, "domain": "coding", "period": "weekly"},
            )
            client.close()


# =========================================================================
# Agent Lifecycle
# =========================================================================

# =========================================================================
# Agent Management
# =========================================================================

# =========================================================================
# Async Tests (Extended)
# =========================================================================


class TestAsyncAgentsExtended:
    """Async tests for new agent methods."""

    @pytest.mark.asyncio
    async def test_async_list_health(self) -> None:
        """List health asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"agents": []}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.agents.list_health()

                mock_request.assert_called_once_with("GET", "/api/agents/health")

    @pytest.mark.asyncio
    async def test_async_get_leaderboard(self) -> None:
        """Get leaderboard asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"rankings": []}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.agents.get_leaderboard(view="coding")

                mock_request.assert_called_once_with(
                    "GET", "/api/leaderboard", params={"view": "coding"}
                )
