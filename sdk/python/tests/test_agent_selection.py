"""Tests for Agent Selection SDK namespace."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock client."""
    return MagicMock()


@pytest.fixture
def mock_async_client() -> MagicMock:
    """Create a mock async client."""
    client = MagicMock()
    client.request = AsyncMock()
    return client


class TestAgentSelectionAPI:
    """Test synchronous AgentSelectionAPI."""

    def test_init(self, mock_client: MagicMock) -> None:
        """Test API initialization."""
        from aragora.namespaces.agent_selection import AgentSelectionAPI

        api = AgentSelectionAPI(mock_client)
        assert api._client is mock_client

    # =========================================================================
    # Plugin Discovery Tests
    # =========================================================================

    def test_list_plugins(self, mock_client: MagicMock) -> None:
        """Test list_plugins calls correct endpoint."""
        from aragora.namespaces.agent_selection import AgentSelectionAPI

        mock_client.request.return_value = {
            "plugins": [
                {
                    "name": "elo_scorer",
                    "type": "scorer",
                    "description": "ELO-based agent scoring",
                    "version": "1.0.0",
                    "enabled": True,
                }
            ]
        }

        api = AgentSelectionAPI(mock_client)
        result = api.list_plugins()

        mock_client.request.assert_called_once_with("GET", "/api/v1/agent-selection/plugins")
        assert len(result["plugins"]) == 1
        assert result["plugins"][0]["name"] == "elo_scorer"

    def test_get_defaults(self, mock_client: MagicMock) -> None:
        """Test get_defaults calls correct endpoint."""
        from aragora.namespaces.agent_selection import AgentSelectionAPI

        mock_client.request.return_value = {
            "scorer": "elo_scorer",
            "team_selector": "diversity_selector",
            "role_assigner": "capability_assigner",
            "weights": {"accuracy": 0.4, "speed": 0.3, "cost": 0.3},
        }

        api = AgentSelectionAPI(mock_client)
        result = api.get_defaults()

        mock_client.request.assert_called_once_with("GET", "/api/v1/agent-selection/defaults")
        assert result["scorer"] == "elo_scorer"
        assert result["weights"]["accuracy"] == 0.4

    # =========================================================================
    # Agent Scoring Tests
    # =========================================================================

    def test_score_agents_minimal(self, mock_client: MagicMock) -> None:
        """Test score_agents with minimal required arguments."""
        from aragora.namespaces.agent_selection import AgentSelectionAPI

        mock_client.request.return_value = {
            "scores": [
                {"agent": "claude", "score": 0.95, "confidence": 0.88},
                {"agent": "gpt-4", "score": 0.92, "confidence": 0.85},
            ]
        }

        api = AgentSelectionAPI(mock_client)
        result = api.score_agents(agents=["claude", "gpt-4"])

        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/agent-selection/score")
        assert call_args[1]["json"] == {"agents": ["claude", "gpt-4"]}
        assert len(result["scores"]) == 2

    def test_score_agents_with_all_options(self, mock_client: MagicMock) -> None:
        """Test score_agents with all optional arguments."""
        from aragora.namespaces.agent_selection import AgentSelectionAPI

        mock_client.request.return_value = {"scores": []}

        api = AgentSelectionAPI(mock_client)
        api.score_agents(
            agents=["claude", "gpt-4", "gemini"],
            context="security code review",
            dimensions=["accuracy", "speed", "cost"],
            scorer="elo_scorer",
            weights={"accuracy": 0.5, "speed": 0.3, "cost": 0.2},
            top_k=2,
        )

        call_args = mock_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["agents"] == ["claude", "gpt-4", "gemini"]
        assert json_body["context"] == "security code review"
        assert json_body["dimensions"] == ["accuracy", "speed", "cost"]
        assert json_body["scorer"] == "elo_scorer"
        assert json_body["weights"] == {"accuracy": 0.5, "speed": 0.3, "cost": 0.2}
        assert json_body["top_k"] == 2

    def test_score_agents_with_partial_options(self, mock_client: MagicMock) -> None:
        """Test score_agents with some optional arguments."""
        from aragora.namespaces.agent_selection import AgentSelectionAPI

        mock_client.request.return_value = {"scores": []}

        api = AgentSelectionAPI(mock_client)
        api.score_agents(
            agents=["claude"],
            context="data analysis",
            top_k=1,
        )

        call_args = mock_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["agents"] == ["claude"]
        assert json_body["context"] == "data analysis"
        assert json_body["top_k"] == 1
        assert "dimensions" not in json_body
        assert "scorer" not in json_body
        assert "weights" not in json_body

    def test_get_best_agent_minimal(self, mock_client: MagicMock) -> None:
        """Test get_best_agent with minimal required arguments."""
        from aragora.namespaces.agent_selection import AgentSelectionAPI

        mock_client.request.return_value = {
            "agent": "claude",
            "score": 0.95,
            "reasoning": "Best for code review tasks",
        }

        api = AgentSelectionAPI(mock_client)
        result = api.get_best_agent(
            pool=["claude", "gpt-4", "gemini"],
            task_type="code_review",
        )

        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/agent-selection/best")
        assert call_args[1]["json"] == {
            "pool": ["claude", "gpt-4", "gemini"],
            "task_type": "code_review",
        }
        assert result["agent"] == "claude"

    def test_get_best_agent_with_context(self, mock_client: MagicMock) -> None:
        """Test get_best_agent with optional context."""
        from aragora.namespaces.agent_selection import AgentSelectionAPI

        mock_client.request.return_value = {"agent": "gpt-4", "score": 0.92}

        api = AgentSelectionAPI(mock_client)
        api.get_best_agent(
            pool=["claude", "gpt-4"],
            task_type="analysis",
            context="Financial data with complex spreadsheets",
        )

        call_args = mock_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["pool"] == ["claude", "gpt-4"]
        assert json_body["task_type"] == "analysis"
        assert json_body["context"] == "Financial data with complex spreadsheets"

    # =========================================================================
    # Team Selection Tests
    # =========================================================================

    def test_select_team_minimal(self, mock_client: MagicMock) -> None:
        """Test select_team with minimal required arguments."""
        from aragora.namespaces.agent_selection import AgentSelectionAPI

        mock_client.request.return_value = {
            "team": [
                {"agent": "claude", "role": "lead"},
                {"agent": "gpt-4", "role": "reviewer"},
            ],
            "team_score": 0.91,
            "diversity_score": 0.85,
        }

        api = AgentSelectionAPI(mock_client)
        result = api.select_team(pool=["claude", "gpt-4", "gemini", "mistral"])

        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/agent-selection/select-team")
        assert call_args[1]["json"] == {"pool": ["claude", "gpt-4", "gemini", "mistral"]}
        assert len(result["team"]) == 2

    def test_select_team_with_all_options(self, mock_client: MagicMock) -> None:
        """Test select_team with all optional arguments."""
        from aragora.namespaces.agent_selection import AgentSelectionAPI

        mock_client.request.return_value = {"team": [], "team_score": 0.0}

        api = AgentSelectionAPI(mock_client)
        api.select_team(
            pool=["claude", "gpt-4", "gemini", "mistral", "deepseek"],
            task_requirements={"domain": "security", "complexity": "high"},
            team_size=3,
            constraints={"max_cost": 100.0},
            min_team_size=2,
            max_team_size=4,
            required_roles=["lead", "reviewer"],
            excluded_agents=["deepseek"],
            diversity_weight=0.7,
            selector="diversity_selector",
            role_assigner="capability_assigner",
        )

        call_args = mock_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["pool"] == ["claude", "gpt-4", "gemini", "mistral", "deepseek"]
        assert json_body["task_requirements"] == {"domain": "security", "complexity": "high"}
        assert json_body["team_size"] == 3
        assert json_body["constraints"] == {"max_cost": 100.0}
        assert json_body["min_team_size"] == 2
        assert json_body["max_team_size"] == 4
        assert json_body["required_roles"] == ["lead", "reviewer"]
        assert json_body["excluded_agents"] == ["deepseek"]
        assert json_body["diversity_weight"] == 0.7
        assert json_body["selector"] == "diversity_selector"
        assert json_body["role_assigner"] == "capability_assigner"

    def test_select_team_with_size_range(self, mock_client: MagicMock) -> None:
        """Test select_team with min/max team size range."""
        from aragora.namespaces.agent_selection import AgentSelectionAPI

        mock_client.request.return_value = {"team": []}

        api = AgentSelectionAPI(mock_client)
        api.select_team(
            pool=["claude", "gpt-4", "gemini"],
            min_team_size=2,
            max_team_size=5,
        )

        call_args = mock_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["min_team_size"] == 2
        assert json_body["max_team_size"] == 5
        assert "team_size" not in json_body

    def test_assign_roles_minimal(self, mock_client: MagicMock) -> None:
        """Test assign_roles with minimal required arguments."""
        from aragora.namespaces.agent_selection import AgentSelectionAPI

        mock_client.request.return_value = {
            "assignments": [
                {"agent": "claude", "role": "lead", "reasoning": "Strongest overall"},
                {"agent": "gpt-4", "role": "reviewer", "reasoning": "Good at critique"},
            ]
        }

        api = AgentSelectionAPI(mock_client)
        result = api.assign_roles(
            members=["claude", "gpt-4"],
            roles=["lead", "reviewer"],
        )

        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/agent-selection/assign-roles")
        assert call_args[1]["json"] == {
            "members": ["claude", "gpt-4"],
            "roles": ["lead", "reviewer"],
        }
        assert len(result["assignments"]) == 2

    def test_assign_roles_with_all_options(self, mock_client: MagicMock) -> None:
        """Test assign_roles with all optional arguments."""
        from aragora.namespaces.agent_selection import AgentSelectionAPI

        mock_client.request.return_value = {"assignments": []}

        api = AgentSelectionAPI(mock_client)
        api.assign_roles(
            members=["claude", "gpt-4", "gemini"],
            roles=["lead", "reviewer", "analyst"],
            task_context="Security audit for financial application",
            assigner="capability_assigner",
        )

        call_args = mock_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["members"] == ["claude", "gpt-4", "gemini"]
        assert json_body["roles"] == ["lead", "reviewer", "analyst"]
        assert json_body["task_context"] == "Security audit for financial application"
        assert json_body["assigner"] == "capability_assigner"

    # =========================================================================
    # History Tests
    # =========================================================================

    def test_get_selection_history_no_params(self, mock_client: MagicMock) -> None:
        """Test get_selection_history without any parameters."""
        from aragora.namespaces.agent_selection import AgentSelectionAPI

        mock_client.request.return_value = {
            "history": [
                {
                    "id": "sel-123",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "team": ["claude", "gpt-4"],
                    "task_type": "code_review",
                }
            ],
            "total": 1,
        }

        api = AgentSelectionAPI(mock_client)
        result = api.get_selection_history()

        mock_client.request.assert_called_once_with(
            "GET", "/api/v1/agent-selection/history", params={}
        )
        assert len(result["history"]) == 1
        assert result["history"][0]["id"] == "sel-123"

    def test_get_selection_history_with_limit(self, mock_client: MagicMock) -> None:
        """Test get_selection_history with limit parameter."""
        from aragora.namespaces.agent_selection import AgentSelectionAPI

        mock_client.request.return_value = {"history": [], "total": 0}

        api = AgentSelectionAPI(mock_client)
        api.get_selection_history(limit=50)

        mock_client.request.assert_called_once_with(
            "GET", "/api/v1/agent-selection/history", params={"limit": 50}
        )

    def test_get_selection_history_with_since(self, mock_client: MagicMock) -> None:
        """Test get_selection_history with since parameter."""
        from aragora.namespaces.agent_selection import AgentSelectionAPI

        mock_client.request.return_value = {"history": []}

        api = AgentSelectionAPI(mock_client)
        api.get_selection_history(since="2024-01-01T00:00:00Z")

        mock_client.request.assert_called_once_with(
            "GET",
            "/api/v1/agent-selection/history",
            params={"since": "2024-01-01T00:00:00Z"},
        )

    def test_get_selection_history_with_all_params(self, mock_client: MagicMock) -> None:
        """Test get_selection_history with all parameters."""
        from aragora.namespaces.agent_selection import AgentSelectionAPI

        mock_client.request.return_value = {"history": []}

        api = AgentSelectionAPI(mock_client)
        api.get_selection_history(limit=100, since="2024-01-01T00:00:00Z")

        mock_client.request.assert_called_once_with(
            "GET",
            "/api/v1/agent-selection/history",
            params={"limit": 100, "since": "2024-01-01T00:00:00Z"},
        )


class TestAsyncAgentSelectionAPI:
    """Test asynchronous AsyncAgentSelectionAPI."""

    def test_init(self, mock_async_client: MagicMock) -> None:
        """Test API initialization."""
        from aragora.namespaces.agent_selection import AsyncAgentSelectionAPI

        api = AsyncAgentSelectionAPI(mock_async_client)
        assert api._client is mock_async_client

    # =========================================================================
    # Plugin Discovery Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_list_plugins(self, mock_async_client: MagicMock) -> None:
        """Test list_plugins calls correct endpoint."""
        from aragora.namespaces.agent_selection import AsyncAgentSelectionAPI

        mock_async_client.request.return_value = {
            "plugins": [{"name": "elo_scorer", "type": "scorer", "enabled": True}]
        }

        api = AsyncAgentSelectionAPI(mock_async_client)
        result = await api.list_plugins()

        mock_async_client.request.assert_called_once_with("GET", "/api/v1/agent-selection/plugins")
        assert len(result["plugins"]) == 1

    @pytest.mark.asyncio
    async def test_get_defaults(self, mock_async_client: MagicMock) -> None:
        """Test get_defaults calls correct endpoint."""
        from aragora.namespaces.agent_selection import AsyncAgentSelectionAPI

        mock_async_client.request.return_value = {
            "scorer": "elo_scorer",
            "team_selector": "diversity_selector",
        }

        api = AsyncAgentSelectionAPI(mock_async_client)
        result = await api.get_defaults()

        mock_async_client.request.assert_called_once_with("GET", "/api/v1/agent-selection/defaults")
        assert result["scorer"] == "elo_scorer"

    # =========================================================================
    # Agent Scoring Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_score_agents_minimal(self, mock_async_client: MagicMock) -> None:
        """Test score_agents with minimal required arguments."""
        from aragora.namespaces.agent_selection import AsyncAgentSelectionAPI

        mock_async_client.request.return_value = {"scores": [{"agent": "claude", "score": 0.95}]}

        api = AsyncAgentSelectionAPI(mock_async_client)
        result = await api.score_agents(agents=["claude"])

        mock_async_client.request.assert_called_once()
        call_args = mock_async_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/agent-selection/score")
        assert call_args[1]["json"] == {"agents": ["claude"]}
        assert result["scores"][0]["agent"] == "claude"

    @pytest.mark.asyncio
    async def test_score_agents_with_options(self, mock_async_client: MagicMock) -> None:
        """Test score_agents with optional arguments."""
        from aragora.namespaces.agent_selection import AsyncAgentSelectionAPI

        mock_async_client.request.return_value = {"scores": []}

        api = AsyncAgentSelectionAPI(mock_async_client)
        await api.score_agents(
            agents=["claude", "gpt-4"],
            context="code review",
            dimensions=["accuracy", "speed"],
            scorer="elo_scorer",
            weights={"accuracy": 0.6, "speed": 0.4},
            top_k=1,
        )

        call_args = mock_async_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["agents"] == ["claude", "gpt-4"]
        assert json_body["context"] == "code review"
        assert json_body["dimensions"] == ["accuracy", "speed"]
        assert json_body["scorer"] == "elo_scorer"
        assert json_body["weights"] == {"accuracy": 0.6, "speed": 0.4}
        assert json_body["top_k"] == 1

    @pytest.mark.asyncio
    async def test_get_best_agent(self, mock_async_client: MagicMock) -> None:
        """Test get_best_agent calls correct endpoint."""
        from aragora.namespaces.agent_selection import AsyncAgentSelectionAPI

        mock_async_client.request.return_value = {
            "agent": "claude",
            "score": 0.95,
            "reasoning": "Best for this task",
        }

        api = AsyncAgentSelectionAPI(mock_async_client)
        result = await api.get_best_agent(
            pool=["claude", "gpt-4"],
            task_type="analysis",
        )

        mock_async_client.request.assert_called_once()
        call_args = mock_async_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/agent-selection/best")
        assert result["agent"] == "claude"

    @pytest.mark.asyncio
    async def test_get_best_agent_with_context(self, mock_async_client: MagicMock) -> None:
        """Test get_best_agent with optional context."""
        from aragora.namespaces.agent_selection import AsyncAgentSelectionAPI

        mock_async_client.request.return_value = {"agent": "gpt-4"}

        api = AsyncAgentSelectionAPI(mock_async_client)
        await api.get_best_agent(
            pool=["claude", "gpt-4"],
            task_type="creative",
            context="Writing marketing copy",
        )

        call_args = mock_async_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["context"] == "Writing marketing copy"

    # =========================================================================
    # Team Selection Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_select_team_minimal(self, mock_async_client: MagicMock) -> None:
        """Test select_team with minimal required arguments."""
        from aragora.namespaces.agent_selection import AsyncAgentSelectionAPI

        mock_async_client.request.return_value = {
            "team": [{"agent": "claude", "role": "lead"}],
            "team_score": 0.9,
        }

        api = AsyncAgentSelectionAPI(mock_async_client)
        result = await api.select_team(pool=["claude", "gpt-4"])

        mock_async_client.request.assert_called_once()
        call_args = mock_async_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/agent-selection/select-team")
        assert call_args[1]["json"] == {"pool": ["claude", "gpt-4"]}
        assert len(result["team"]) == 1

    @pytest.mark.asyncio
    async def test_select_team_with_all_options(self, mock_async_client: MagicMock) -> None:
        """Test select_team with all optional arguments."""
        from aragora.namespaces.agent_selection import AsyncAgentSelectionAPI

        mock_async_client.request.return_value = {"team": []}

        api = AsyncAgentSelectionAPI(mock_async_client)
        await api.select_team(
            pool=["claude", "gpt-4", "gemini"],
            task_requirements={"domain": "ml", "complexity": "medium"},
            team_size=2,
            constraints={"budget": 50},
            min_team_size=1,
            max_team_size=3,
            required_roles=["lead"],
            excluded_agents=["gemini"],
            diversity_weight=0.5,
            selector="cost_optimizer",
            role_assigner="random_assigner",
        )

        call_args = mock_async_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["pool"] == ["claude", "gpt-4", "gemini"]
        assert json_body["task_requirements"] == {"domain": "ml", "complexity": "medium"}
        assert json_body["team_size"] == 2
        assert json_body["constraints"] == {"budget": 50}
        assert json_body["min_team_size"] == 1
        assert json_body["max_team_size"] == 3
        assert json_body["required_roles"] == ["lead"]
        assert json_body["excluded_agents"] == ["gemini"]
        assert json_body["diversity_weight"] == 0.5
        assert json_body["selector"] == "cost_optimizer"
        assert json_body["role_assigner"] == "random_assigner"

    @pytest.mark.asyncio
    async def test_assign_roles_minimal(self, mock_async_client: MagicMock) -> None:
        """Test assign_roles with minimal required arguments."""
        from aragora.namespaces.agent_selection import AsyncAgentSelectionAPI

        mock_async_client.request.return_value = {
            "assignments": [
                {"agent": "claude", "role": "lead"},
            ]
        }

        api = AsyncAgentSelectionAPI(mock_async_client)
        result = await api.assign_roles(
            members=["claude"],
            roles=["lead"],
        )

        mock_async_client.request.assert_called_once()
        call_args = mock_async_client.request.call_args
        assert call_args[0] == ("POST", "/api/v1/agent-selection/assign-roles")
        assert call_args[1]["json"] == {
            "members": ["claude"],
            "roles": ["lead"],
        }
        assert len(result["assignments"]) == 1

    @pytest.mark.asyncio
    async def test_assign_roles_with_options(self, mock_async_client: MagicMock) -> None:
        """Test assign_roles with optional arguments."""
        from aragora.namespaces.agent_selection import AsyncAgentSelectionAPI

        mock_async_client.request.return_value = {"assignments": []}

        api = AsyncAgentSelectionAPI(mock_async_client)
        await api.assign_roles(
            members=["claude", "gpt-4"],
            roles=["lead", "reviewer"],
            task_context="Code review session",
            assigner="capability_assigner",
        )

        call_args = mock_async_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["task_context"] == "Code review session"
        assert json_body["assigner"] == "capability_assigner"

    # =========================================================================
    # History Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_get_selection_history_no_params(self, mock_async_client: MagicMock) -> None:
        """Test get_selection_history without any parameters."""
        from aragora.namespaces.agent_selection import AsyncAgentSelectionAPI

        mock_async_client.request.return_value = {
            "history": [{"id": "sel-123", "team": ["claude"]}],
            "total": 1,
        }

        api = AsyncAgentSelectionAPI(mock_async_client)
        result = await api.get_selection_history()

        mock_async_client.request.assert_called_once_with(
            "GET", "/api/v1/agent-selection/history", params={}
        )
        assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_get_selection_history_with_limit(self, mock_async_client: MagicMock) -> None:
        """Test get_selection_history with limit parameter."""
        from aragora.namespaces.agent_selection import AsyncAgentSelectionAPI

        mock_async_client.request.return_value = {"history": []}

        api = AsyncAgentSelectionAPI(mock_async_client)
        await api.get_selection_history(limit=25)

        mock_async_client.request.assert_called_once_with(
            "GET", "/api/v1/agent-selection/history", params={"limit": 25}
        )

    @pytest.mark.asyncio
    async def test_get_selection_history_with_since(self, mock_async_client: MagicMock) -> None:
        """Test get_selection_history with since parameter."""
        from aragora.namespaces.agent_selection import AsyncAgentSelectionAPI

        mock_async_client.request.return_value = {"history": []}

        api = AsyncAgentSelectionAPI(mock_async_client)
        await api.get_selection_history(since="2024-06-01T00:00:00Z")

        mock_async_client.request.assert_called_once_with(
            "GET",
            "/api/v1/agent-selection/history",
            params={"since": "2024-06-01T00:00:00Z"},
        )

    @pytest.mark.asyncio
    async def test_get_selection_history_with_all_params(
        self, mock_async_client: MagicMock
    ) -> None:
        """Test get_selection_history with all parameters."""
        from aragora.namespaces.agent_selection import AsyncAgentSelectionAPI

        mock_async_client.request.return_value = {"history": []}

        api = AsyncAgentSelectionAPI(mock_async_client)
        await api.get_selection_history(limit=50, since="2024-03-15T12:00:00Z")

        mock_async_client.request.assert_called_once_with(
            "GET",
            "/api/v1/agent-selection/history",
            params={"limit": 50, "since": "2024-03-15T12:00:00Z"},
        )
