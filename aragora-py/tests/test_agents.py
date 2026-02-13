"""Tests for Aragora SDK Agents API.

Comprehensive tests covering:
- AgentsAPI listing and retrieval
- Agent history, rivals, and allies
- Calibration and performance metrics
- Head-to-head statistics
- ELO ratings and leaderboards
- Agent comparison and relationships
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora_client.client import AgentsAPI, AragoraClient
from aragora_client.types import AgentProfile

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock AragoraClient."""
    client = MagicMock(spec=AragoraClient)
    client._get = AsyncMock()
    client._post = AsyncMock()
    return client


@pytest.fixture
def agents_api(mock_client: MagicMock) -> AgentsAPI:
    """Create AgentsAPI with mock client."""
    return AgentsAPI(mock_client)


@pytest.fixture
def agent_profile_response() -> dict[str, Any]:
    """Standard agent profile response."""
    return {
        "id": "claude",
        "name": "Claude",
        "provider": "anthropic",
        "elo_rating": 1650.5,
        "matches_played": 250,
        "win_rate": 0.62,
        "specialties": ["reasoning", "coding", "analysis"],
        "metadata": {"version": "claude-3-opus"},
    }


@pytest.fixture
def match_history_response() -> dict[str, Any]:
    """Standard match history response."""
    return {
        "matches": [
            {
                "id": "match-1",
                "debate_id": "debate-123",
                "opponent": "gpt4",
                "result": "win",
                "elo_change": 12.5,
                "timestamp": "2026-01-01T00:00:00Z",
            },
            {
                "id": "match-2",
                "debate_id": "debate-124",
                "opponent": "gemini",
                "result": "loss",
                "elo_change": -8.3,
                "timestamp": "2026-01-02T00:00:00Z",
            },
        ]
    }


@pytest.fixture
def rivals_response() -> dict[str, Any]:
    """Standard rivals response."""
    return {
        "rivals": [
            {"agent_id": "gpt4", "matches": 45, "win_rate_vs": 0.58},
            {"agent_id": "gemini", "matches": 30, "win_rate_vs": 0.45},
        ]
    }


@pytest.fixture
def calibration_response() -> dict[str, Any]:
    """Standard calibration data response."""
    return {
        "agent_id": "claude",
        "confidence_calibration": 0.85,
        "overconfidence_rate": 0.12,
        "underconfidence_rate": 0.03,
        "brier_score": 0.18,
        "sample_size": 500,
    }


@pytest.fixture
def performance_response() -> dict[str, Any]:
    """Standard performance metrics response."""
    return {
        "agent_id": "claude",
        "avg_response_time_ms": 1250,
        "avg_token_count": 450,
        "consensus_contribution_rate": 0.35,
        "argument_quality_score": 0.82,
        "citation_accuracy": 0.91,
    }


@pytest.fixture
def head_to_head_response() -> dict[str, Any]:
    """Standard head-to-head response."""
    return {
        "agent_a": "claude",
        "agent_b": "gpt4",
        "total_matches": 45,
        "wins_a": 26,
        "wins_b": 19,
        "win_rate_a": 0.578,
        "recent_trend": "claude_improving",
    }


@pytest.fixture
def elo_response() -> dict[str, Any]:
    """Standard ELO response."""
    return {
        "agent_id": "claude",
        "current_rating": 1650.5,
        "peak_rating": 1720.3,
        "lowest_rating": 1480.2,
        "history": [
            {"date": "2026-01-01", "rating": 1620.0},
            {"date": "2026-01-15", "rating": 1650.5},
        ],
    }


# =============================================================================
# Basic Agent Operations Tests
# =============================================================================


class TestAgentsAPIList:
    """Tests for AgentsAPI.list()."""

    @pytest.mark.asyncio
    async def test_list_agents(
        self,
        agents_api: AgentsAPI,
        mock_client: MagicMock,
        agent_profile_response: dict[str, Any],
    ) -> None:
        """Test listing all agents."""
        mock_client._get.return_value = {"agents": [agent_profile_response]}

        result = await agents_api.list()

        mock_client._get.assert_called_once_with("/api/v1/agents")
        assert len(result) == 1
        assert isinstance(result[0], AgentProfile)
        assert result[0].id == "claude"
        assert result[0].elo_rating == 1650.5

    @pytest.mark.asyncio
    async def test_list_empty(
        self, agents_api: AgentsAPI, mock_client: MagicMock
    ) -> None:
        """Test listing when no agents available."""
        mock_client._get.return_value = {"agents": []}

        result = await agents_api.list()

        assert result == []


class TestAgentsAPIGet:
    """Tests for AgentsAPI.get()."""

    @pytest.mark.asyncio
    async def test_get_agent(
        self,
        agents_api: AgentsAPI,
        mock_client: MagicMock,
        agent_profile_response: dict[str, Any],
    ) -> None:
        """Test getting a single agent."""
        mock_client._get.return_value = agent_profile_response

        result = await agents_api.get("claude")

        mock_client._get.assert_called_once_with("/api/v1/agents/claude")
        assert isinstance(result, AgentProfile)
        assert result.name == "Claude"
        assert result.provider == "anthropic"
        assert "reasoning" in result.specialties


# =============================================================================
# History and Relationship Tests
# =============================================================================


class TestAgentsAPIHistory:
    """Tests for AgentsAPI.history()."""

    @pytest.mark.asyncio
    async def test_get_history(
        self,
        agents_api: AgentsAPI,
        mock_client: MagicMock,
        match_history_response: dict[str, Any],
    ) -> None:
        """Test getting match history."""
        mock_client._get.return_value = match_history_response

        result = await agents_api.history("claude")

        mock_client._get.assert_called_once()
        assert len(result) == 2
        assert result[0]["result"] == "win"
        assert result[1]["result"] == "loss"

    @pytest.mark.asyncio
    async def test_get_history_with_limit(
        self, agents_api: AgentsAPI, mock_client: MagicMock
    ) -> None:
        """Test getting history with limit."""
        mock_client._get.return_value = {"matches": []}

        await agents_api.history("claude", limit=10)

        call_args = mock_client._get.call_args
        assert call_args[1]["params"]["limit"] == 10


class TestAgentsAPIRivals:
    """Tests for AgentsAPI.rivals()."""

    @pytest.mark.asyncio
    async def test_get_rivals(
        self,
        agents_api: AgentsAPI,
        mock_client: MagicMock,
        rivals_response: dict[str, Any],
    ) -> None:
        """Test getting agent rivals."""
        mock_client._get.return_value = rivals_response

        result = await agents_api.rivals("claude")

        mock_client._get.assert_called_once_with("/api/v1/agents/claude/rivals")
        assert len(result) == 2
        assert result[0]["agent_id"] == "gpt4"


class TestAgentsAPIAllies:
    """Tests for AgentsAPI.allies()."""

    @pytest.mark.asyncio
    async def test_get_allies(
        self, agents_api: AgentsAPI, mock_client: MagicMock
    ) -> None:
        """Test getting agent allies."""
        mock_client._get.return_value = {
            "allies": [{"agent_id": "mistral", "collaboration_score": 0.85}]
        }

        result = await agents_api.allies("claude")

        mock_client._get.assert_called_once_with("/api/v1/agents/claude/allies")
        assert len(result) == 1


# =============================================================================
# Calibration and Performance Tests
# =============================================================================


class TestAgentsAPICalibration:
    """Tests for AgentsAPI.get_calibration()."""

    @pytest.mark.asyncio
    async def test_get_calibration(
        self,
        agents_api: AgentsAPI,
        mock_client: MagicMock,
        calibration_response: dict[str, Any],
    ) -> None:
        """Test getting calibration data."""
        mock_client._get.return_value = calibration_response

        result = await agents_api.get_calibration("claude")

        mock_client._get.assert_called_once_with("/api/v1/agents/claude/calibration")
        assert result["confidence_calibration"] == 0.85
        assert result["brier_score"] == 0.18


class TestAgentsAPIPerformance:
    """Tests for AgentsAPI.get_performance()."""

    @pytest.mark.asyncio
    async def test_get_performance(
        self,
        agents_api: AgentsAPI,
        mock_client: MagicMock,
        performance_response: dict[str, Any],
    ) -> None:
        """Test getting performance metrics."""
        mock_client._get.return_value = performance_response

        result = await agents_api.get_performance("claude")

        mock_client._get.assert_called_once_with("/api/v1/agents/claude/performance")
        assert result["avg_response_time_ms"] == 1250
        assert result["argument_quality_score"] == 0.82


# =============================================================================
# Head-to-Head and Opponent Analysis Tests
# =============================================================================


class TestAgentsAPIHeadToHead:
    """Tests for AgentsAPI.get_head_to_head()."""

    @pytest.mark.asyncio
    async def test_get_head_to_head(
        self,
        agents_api: AgentsAPI,
        mock_client: MagicMock,
        head_to_head_response: dict[str, Any],
    ) -> None:
        """Test getting head-to-head statistics."""
        mock_client._get.return_value = head_to_head_response

        result = await agents_api.get_head_to_head("claude", "gpt4")

        mock_client._get.assert_called_once_with(
            "/api/v1/agents/claude/head-to-head/gpt4"
        )
        assert result["total_matches"] == 45
        assert result["wins_a"] == 26


class TestAgentsAPIOpponentBriefing:
    """Tests for AgentsAPI.get_opponent_briefing()."""

    @pytest.mark.asyncio
    async def test_get_opponent_briefing(
        self, agents_api: AgentsAPI, mock_client: MagicMock
    ) -> None:
        """Test getting opponent briefing."""
        mock_client._get.return_value = {
            "opponent": "gpt4",
            "strengths": ["logical reasoning", "code generation"],
            "weaknesses": ["creative writing"],
            "recommended_strategy": "Focus on creative arguments",
        }

        result = await agents_api.get_opponent_briefing("claude", "gpt4")

        mock_client._get.assert_called_once_with(
            "/api/v1/agents/claude/opponent-briefing/gpt4"
        )
        assert "strengths" in result
        assert "recommended_strategy" in result


# =============================================================================
# Consistency and Position Tests
# =============================================================================


class TestAgentsAPIConsistency:
    """Tests for AgentsAPI.get_consistency()."""

    @pytest.mark.asyncio
    async def test_get_consistency(
        self, agents_api: AgentsAPI, mock_client: MagicMock
    ) -> None:
        """Test getting consistency metrics."""
        mock_client._get.return_value = {
            "consistency_score": 0.92,
            "position_drift_rate": 0.08,
            "self_contradiction_rate": 0.02,
        }

        result = await agents_api.get_consistency("claude")

        mock_client._get.assert_called_once_with("/api/v1/agents/claude/consistency")
        assert result["consistency_score"] == 0.92


class TestAgentsAPIFlips:
    """Tests for AgentsAPI.get_flips()."""

    @pytest.mark.asyncio
    async def test_get_flips(
        self, agents_api: AgentsAPI, mock_client: MagicMock
    ) -> None:
        """Test getting position flips."""
        mock_client._get.return_value = {
            "flips": [
                {
                    "debate_id": "debate-123",
                    "topic": "microservices",
                    "from_position": "for",
                    "to_position": "against",
                    "reason": "new evidence presented",
                }
            ]
        }

        result = await agents_api.get_flips("claude")

        assert len(result) == 1
        assert result[0]["from_position"] == "for"

    @pytest.mark.asyncio
    async def test_get_flips_pagination(
        self, agents_api: AgentsAPI, mock_client: MagicMock
    ) -> None:
        """Test getting flips with pagination."""
        mock_client._get.return_value = {"flips": []}

        await agents_api.get_flips("claude", limit=5, offset=10)

        call_args = mock_client._get.call_args
        assert call_args[1]["params"]["limit"] == 5
        assert call_args[1]["params"]["offset"] == 10


class TestAgentsAPIPositions:
    """Tests for AgentsAPI.get_positions()."""

    @pytest.mark.asyncio
    async def test_get_positions(
        self, agents_api: AgentsAPI, mock_client: MagicMock
    ) -> None:
        """Test getting agent positions."""
        mock_client._get.return_value = {
            "positions": [
                {"topic": "microservices", "stance": "for", "confidence": 0.85}
            ]
        }

        result = await agents_api.get_positions("claude")

        assert len(result) == 1
        assert result[0]["topic"] == "microservices"

    @pytest.mark.asyncio
    async def test_get_positions_by_topic(
        self, agents_api: AgentsAPI, mock_client: MagicMock
    ) -> None:
        """Test getting positions filtered by topic."""
        mock_client._get.return_value = {"positions": []}

        await agents_api.get_positions("claude", topic="kubernetes")

        call_args = mock_client._get.call_args
        assert call_args[1]["params"]["topic"] == "kubernetes"


# =============================================================================
# Network and Moments Tests
# =============================================================================


class TestAgentsAPINetwork:
    """Tests for AgentsAPI.get_network()."""

    @pytest.mark.asyncio
    async def test_get_network(
        self, agents_api: AgentsAPI, mock_client: MagicMock
    ) -> None:
        """Test getting agent network graph."""
        mock_client._get.return_value = {
            "nodes": [{"id": "claude"}, {"id": "gpt4"}],
            "edges": [{"source": "claude", "target": "gpt4", "weight": 45}],
        }

        result = await agents_api.get_network("claude")

        mock_client._get.assert_called_once_with("/api/v1/agents/claude/network")
        assert "nodes" in result
        assert "edges" in result


class TestAgentsAPIMoments:
    """Tests for AgentsAPI.get_moments()."""

    @pytest.mark.asyncio
    async def test_get_moments(
        self, agents_api: AgentsAPI, mock_client: MagicMock
    ) -> None:
        """Test getting notable moments."""
        mock_client._get.return_value = {
            "moments": [
                {
                    "debate_id": "debate-123",
                    "type": "brilliant_argument",
                    "description": "Novel insight about distributed systems",
                    "timestamp": "2026-01-01T00:00:00Z",
                }
            ]
        }

        result = await agents_api.get_moments("claude")

        assert len(result) == 1
        assert result[0]["type"] == "brilliant_argument"

    @pytest.mark.asyncio
    async def test_get_moments_by_type(
        self, agents_api: AgentsAPI, mock_client: MagicMock
    ) -> None:
        """Test getting moments filtered by type."""
        mock_client._get.return_value = {"moments": []}

        await agents_api.get_moments("claude", type="upset_victory")

        call_args = mock_client._get.call_args
        assert call_args[1]["params"]["type"] == "upset_victory"


# =============================================================================
# Domain and ELO Tests
# =============================================================================


class TestAgentsAPIDomains:
    """Tests for AgentsAPI.get_domains()."""

    @pytest.mark.asyncio
    async def test_get_domains(
        self, agents_api: AgentsAPI, mock_client: MagicMock
    ) -> None:
        """Test getting domain expertise ratings."""
        mock_client._get.return_value = {
            "domains": [
                {"domain": "software_engineering", "rating": 1720, "matches": 80},
                {"domain": "philosophy", "rating": 1580, "matches": 25},
            ]
        }

        result = await agents_api.get_domains("claude")

        mock_client._get.assert_called_once_with("/api/v1/agents/claude/domains")
        assert len(result) == 2
        assert result[0]["domain"] == "software_engineering"


class TestAgentsAPIElo:
    """Tests for AgentsAPI.get_elo()."""

    @pytest.mark.asyncio
    async def test_get_elo(
        self,
        agents_api: AgentsAPI,
        mock_client: MagicMock,
        elo_response: dict[str, Any],
    ) -> None:
        """Test getting ELO rating and history."""
        mock_client._get.return_value = elo_response

        result = await agents_api.get_elo("claude")

        mock_client._get.assert_called_once_with("/api/v1/agents/claude/elo")
        assert result["current_rating"] == 1650.5
        assert result["peak_rating"] == 1720.3
        assert len(result["history"]) == 2


# =============================================================================
# Leaderboard and Comparison Tests
# =============================================================================


class TestAgentsAPILeaderboard:
    """Tests for AgentsAPI.get_leaderboard()."""

    @pytest.mark.asyncio
    async def test_get_leaderboard(
        self,
        agents_api: AgentsAPI,
        mock_client: MagicMock,
        agent_profile_response: dict[str, Any],
    ) -> None:
        """Test getting the leaderboard."""
        agent2 = agent_profile_response.copy()
        agent2["id"] = "gpt4"
        agent2["name"] = "GPT-4"
        agent2["elo_rating"] = 1620.0

        mock_client._get.return_value = {"agents": [agent_profile_response, agent2]}

        result = await agents_api.get_leaderboard()

        mock_client._get.assert_called_once_with("/api/v1/leaderboard")
        assert len(result) == 2
        assert isinstance(result[0], AgentProfile)
        # First agent should have higher ELO
        assert result[0].elo_rating >= result[1].elo_rating


class TestAgentsAPICompare:
    """Tests for AgentsAPI.compare()."""

    @pytest.mark.asyncio
    async def test_compare_agents(
        self, agents_api: AgentsAPI, mock_client: MagicMock
    ) -> None:
        """Test comparing multiple agents."""
        mock_client._get.return_value = {
            "comparison": {
                "claude": {"elo": 1650, "win_rate": 0.62},
                "gpt4": {"elo": 1620, "win_rate": 0.58},
                "gemini": {"elo": 1580, "win_rate": 0.52},
            },
            "head_to_head": {
                "claude_vs_gpt4": {"wins": 26, "losses": 19},
                "claude_vs_gemini": {"wins": 18, "losses": 12},
            },
        }

        result = await agents_api.compare(["claude", "gpt4", "gemini"])

        mock_client._get.assert_called_once()
        call_args = mock_client._get.call_args
        assert call_args[1]["params"]["agents"] == "claude,gpt4,gemini"
        assert "comparison" in result


class TestAgentsAPIRelationship:
    """Tests for AgentsAPI.get_relationship()."""

    @pytest.mark.asyncio
    async def test_get_relationship(
        self, agents_api: AgentsAPI, mock_client: MagicMock
    ) -> None:
        """Test getting relationship analysis between two agents."""
        mock_client._get.return_value = {
            "agent_a": "claude",
            "agent_b": "gpt4",
            "relationship_type": "rival",
            "collaboration_history": 12,
            "conflict_history": 33,
            "similarity_score": 0.72,
        }

        result = await agents_api.get_relationship("claude", "gpt4")

        mock_client._get.assert_called_once_with(
            "/api/v1/agents/claude/relationship/gpt4"
        )
        assert result["relationship_type"] == "rival"
        assert result["similarity_score"] == 0.72
