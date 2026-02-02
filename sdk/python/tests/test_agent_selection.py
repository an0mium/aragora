"""Tests for Agent Selection namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestPluginDiscovery:
    """Tests for plugin discovery methods."""

    def test_list_plugins(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "plugins": [{"name": "elo_scorer", "type": "scorer", "enabled": True}]
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.agent_selection.list_plugins()
            mock_request.assert_called_once_with("GET", "/api/v1/agent-selection/plugins")
            assert result["plugins"][0]["name"] == "elo_scorer"
            client.close()

    def test_get_defaults(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "scorer": "elo_scorer",
                "team_selector": "balanced",
                "role_assigner": "capability_match",
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.agent_selection.get_defaults()
            mock_request.assert_called_once_with("GET", "/api/v1/agent-selection/defaults")
            assert result["scorer"] == "elo_scorer"
            client.close()


class TestAgentScoring:
    """Tests for agent scoring methods."""

    def test_score_agents_minimal(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"scores": [{"agent": "claude", "score": 0.95}]}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.agent_selection.score_agents(agents=["claude", "gpt-4"])
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/agent-selection/score",
                json={"agents": ["claude", "gpt-4"]},
            )
            assert result["scores"][0]["agent"] == "claude"
            client.close()

    def test_score_agents_with_all_options(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"scores": []}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.agent_selection.score_agents(
                agents=["claude", "gpt-4", "gemini"],
                context="security code review",
                dimensions=["accuracy", "speed", "cost"],
                scorer="elo_scorer",
                weights={"accuracy": 0.6, "speed": 0.2, "cost": 0.2},
                top_k=2,
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/agent-selection/score",
                json={
                    "agents": ["claude", "gpt-4", "gemini"],
                    "context": "security code review",
                    "dimensions": ["accuracy", "speed", "cost"],
                    "scorer": "elo_scorer",
                    "weights": {"accuracy": 0.6, "speed": 0.2, "cost": 0.2},
                    "top_k": 2,
                },
            )
            client.close()

    def test_score_agents_omits_none_values(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"scores": []}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.agent_selection.score_agents(
                agents=["claude"],
                context="data analysis",
                top_k=1,
            )
            call_json = mock_request.call_args[1]["json"]
            assert "dimensions" not in call_json
            assert "scorer" not in call_json
            assert "weights" not in call_json
            client.close()

    def test_get_best_agent_minimal(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"agent": "claude", "score": 0.92}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.agent_selection.get_best_agent(
                pool=["claude", "gpt-4", "gemini"],
                task_type="code_review",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/agent-selection/best",
                json={"pool": ["claude", "gpt-4", "gemini"], "task_type": "code_review"},
            )
            assert result["agent"] == "claude"
            client.close()

    def test_get_best_agent_with_context(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"agent": "gpt-4", "score": 0.88}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.agent_selection.get_best_agent(
                pool=["claude", "gpt-4"],
                task_type="creative",
                context="Write marketing copy for a SaaS product",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/agent-selection/best",
                json={
                    "pool": ["claude", "gpt-4"],
                    "task_type": "creative",
                    "context": "Write marketing copy for a SaaS product",
                },
            )
            client.close()


class TestTeamSelection:
    """Tests for team selection and role assignment methods."""

    def test_select_team_minimal(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "team": [{"agent": "claude", "role": "lead"}],
                "team_score": 0.91,
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.agent_selection.select_team(
                pool=["claude", "gpt-4", "gemini", "mistral"],
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/agent-selection/select-team",
                json={"pool": ["claude", "gpt-4", "gemini", "mistral"]},
            )
            assert result["team_score"] == 0.91
            client.close()

    def test_select_team_with_full_options(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"team": [], "team_score": 0.85}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.agent_selection.select_team(
                pool=["claude", "gpt-4", "gemini", "mistral", "deepseek"],
                task_requirements={"domain": "security", "complexity": "high"},
                team_size=3,
                constraints={"max_cost": 100},
                required_roles=["lead", "reviewer"],
                excluded_agents=["deepseek"],
                diversity_weight=0.7,
                selector="balanced",
                role_assigner="capability_match",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/agent-selection/select-team",
                json={
                    "pool": ["claude", "gpt-4", "gemini", "mistral", "deepseek"],
                    "task_requirements": {"domain": "security", "complexity": "high"},
                    "team_size": 3,
                    "constraints": {"max_cost": 100},
                    "required_roles": ["lead", "reviewer"],
                    "excluded_agents": ["deepseek"],
                    "diversity_weight": 0.7,
                    "selector": "balanced",
                    "role_assigner": "capability_match",
                },
            )
            client.close()

    def test_select_team_with_size_range(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"team": []}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.agent_selection.select_team(
                pool=["claude", "gpt-4", "gemini"],
                min_team_size=2,
                max_team_size=4,
            )
            call_json = mock_request.call_args[1]["json"]
            assert call_json["min_team_size"] == 2
            assert call_json["max_team_size"] == 4
            assert "team_size" not in call_json
            client.close()

    def test_assign_roles_minimal(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "assignments": [
                    {"agent": "claude", "role": "lead"},
                    {"agent": "gpt-4", "role": "reviewer"},
                ]
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.agent_selection.assign_roles(
                members=["claude", "gpt-4"],
                roles=["lead", "reviewer"],
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/agent-selection/assign-roles",
                json={
                    "members": ["claude", "gpt-4"],
                    "roles": ["lead", "reviewer"],
                },
            )
            assert len(result["assignments"]) == 2
            client.close()

    def test_assign_roles_with_context_and_assigner(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"assignments": []}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.agent_selection.assign_roles(
                members=["claude", "gpt-4", "gemini"],
                roles=["lead", "critic", "synthesizer"],
                task_context="Debate on API design patterns",
                assigner="capability_match",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/agent-selection/assign-roles",
                json={
                    "members": ["claude", "gpt-4", "gemini"],
                    "roles": ["lead", "critic", "synthesizer"],
                    "task_context": "Debate on API design patterns",
                    "assigner": "capability_match",
                },
            )
            client.close()


class TestSelectionHistory:
    """Tests for selection history methods."""

    def test_get_selection_history_no_params(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"history": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.agent_selection.get_selection_history()
            mock_request.assert_called_once_with(
                "GET", "/api/v1/agent-selection/history", params={}
            )
            assert result["total"] == 0
            client.close()

    def test_get_selection_history_with_filters(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"history": [{"id": "sel_1", "team_size": 3}]}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.agent_selection.get_selection_history(
                limit=10,
                since="2025-06-01T00:00:00Z",
            )
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/agent-selection/history",
                params={"limit": 10, "since": "2025-06-01T00:00:00Z"},
            )
            assert result["history"][0]["id"] == "sel_1"
            client.close()


class TestAsyncAgentSelection:
    """Tests for async agent selection methods."""

    @pytest.mark.asyncio
    async def test_list_plugins(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"plugins": [{"name": "elo_scorer"}]}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.agent_selection.list_plugins()
            mock_request.assert_called_once_with("GET", "/api/v1/agent-selection/plugins")
            assert result["plugins"][0]["name"] == "elo_scorer"
            await client.close()

    @pytest.mark.asyncio
    async def test_score_agents(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"scores": [{"agent": "claude", "score": 0.95}]}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.agent_selection.score_agents(
                agents=["claude", "gpt-4"],
                context="data analysis",
                dimensions=["accuracy", "speed"],
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/agent-selection/score",
                json={
                    "agents": ["claude", "gpt-4"],
                    "context": "data analysis",
                    "dimensions": ["accuracy", "speed"],
                },
            )
            assert result["scores"][0]["score"] == 0.95
            await client.close()

    @pytest.mark.asyncio
    async def test_select_team(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "team": [{"agent": "claude", "role": "lead"}],
                "team_score": 0.9,
            }
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.agent_selection.select_team(
                pool=["claude", "gpt-4", "gemini"],
                task_requirements={"domain": "security"},
                team_size=2,
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/agent-selection/select-team",
                json={
                    "pool": ["claude", "gpt-4", "gemini"],
                    "task_requirements": {"domain": "security"},
                    "team_size": 2,
                },
            )
            assert result["team_score"] == 0.9
            await client.close()

    @pytest.mark.asyncio
    async def test_get_selection_history(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"history": [{"id": "sel_1"}]}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.agent_selection.get_selection_history(limit=5)
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/agent-selection/history",
                params={"limit": 5},
            )
            assert len(result["history"]) == 1
            await client.close()
