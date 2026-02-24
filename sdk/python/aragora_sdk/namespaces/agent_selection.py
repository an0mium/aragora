"""
Agent Selection Namespace API

Provides methods for agent team selection operations:
- Plugin discovery and configuration
- Agent scoring for specific tasks
- Team selection with role assignment
- Selection history tracking
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class AgentSelectionAPI:
    """
    Synchronous Agent Selection API.

    Provides methods for agent team selection and scoring:
    - List available selection plugins
    - Get default plugin configurations
    - Score agents for specific tasks
    - Select optimal teams with role assignment
    - Track selection history

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> plugins = client.agent_selection.list_plugins()
        >>> scores = client.agent_selection.score_agents(
        ...     agents=["claude", "gpt-4", "gemini"],
        ...     context="security code review",
        ...     dimensions=["accuracy", "speed", "cost"],
        ... )
        >>> team = client.agent_selection.select_team(
        ...     pool=["claude", "gpt-4", "gemini", "mistral"],
        ...     task_requirements={"domain": "security", "complexity": "high"},
        ...     team_size=3,
        ... )
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Plugin Discovery
    # ===========================================================================

    def list_plugins(self) -> dict[str, Any]:
        """
        List all available selection plugins.

        Returns plugins for scoring, team selection, and role assignment.

        Returns:
            Dict with plugins array containing plugin metadata including
            name, type, description, version, enabled status, and config schema
        """
        return self._client.request("GET", "/api/v1/agent-selection/plugins")

    def get_defaults(self) -> dict[str, Any]:
        """
        Get default plugin configuration.

        Returns the default scorer, team selector, role assigner,
        and scorer weight settings.

        Returns:
            Dict with default plugin names and weight configurations
        """
        return self._client.request("GET", "/api/v1/agent-selection/defaults")

    # ===========================================================================
    # Agent Scoring
    # ===========================================================================

    def score_agents(
        self,
        agents: list[str],
        context: str | None = None,
        dimensions: list[str] | None = None,
        scorer: str | None = None,
        weights: dict[str, float] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        Score agents for a specific task or context.

        Evaluates agents across multiple dimensions (e.g., accuracy, speed, cost)
        and returns ranked scores with reasoning.

        Args:
            agents: List of agent identifiers to score
            context: Task context or description for scoring
            dimensions: Scoring dimensions (e.g., ["accuracy", "speed", "cost"])
            scorer: Specific scorer plugin to use (optional)
            weights: Custom weights for each dimension (optional)
            top_k: Return only top K agents (optional)

        Returns:
            Dict with scores array containing agent scores, dimension breakdowns,
            confidence levels, and reasoning for each agent
        """
        data: dict[str, Any] = {"agents": agents}
        if context is not None:
            data["context"] = context
        if dimensions is not None:
            data["dimensions"] = dimensions
        if scorer is not None:
            data["scorer"] = scorer
        if weights is not None:
            data["weights"] = weights
        if top_k is not None:
            data["top_k"] = top_k
        return self._client.request("POST", "/api/v1/agent-selection/score", json=data)

    def get_best_agent(
        self,
        pool: list[str],
        task_type: str,
        context: str | None = None,
    ) -> dict[str, Any]:
        """
        Get the best agent for a specific task from a pool.

        Convenience method that returns the single best-suited agent
        for the given task type and context.

        Args:
            pool: List of candidate agent identifiers
            task_type: Type of task (e.g., "code_review", "analysis", "creative")
            context: Additional context for selection (optional)

        Returns:
            Dict with the best agent details including score and reasoning
        """
        data: dict[str, Any] = {
            "pool": pool,
            "task_type": task_type,
        }
        if context is not None:
            data["context"] = context
        return self._client.request("POST", "/api/v1/agent-selection/best", json=data)

    # ===========================================================================
    # Team Selection
    # ===========================================================================

    def select_team(
        self,
        pool: list[str],
        task_requirements: dict[str, Any] | None = None,
        team_size: int | None = None,
        constraints: dict[str, Any] | None = None,
        min_team_size: int | None = None,
        max_team_size: int | None = None,
        required_roles: list[str] | None = None,
        excluded_agents: list[str] | None = None,
        diversity_weight: float | None = None,
        selector: str | None = None,
        role_assigner: str | None = None,
    ) -> dict[str, Any]:
        """
        Select an optimal team of agents for a task.

        Analyzes the agent pool and selects a team optimized for
        the given task requirements, with automatic role assignment.

        Args:
            pool: List of candidate agent identifiers
            task_requirements: Dict describing task needs (domain, complexity, etc.)
            team_size: Exact team size (optional, use min/max for range)
            constraints: Additional selection constraints (optional)
            min_team_size: Minimum team size (optional)
            max_team_size: Maximum team size (optional)
            required_roles: Roles that must be filled (optional)
            excluded_agents: Agents to exclude from selection (optional)
            diversity_weight: Weight for team diversity (0.0-1.0, optional)
            selector: Specific team selector plugin to use (optional)
            role_assigner: Specific role assigner plugin to use (optional)

        Returns:
            Dict with team array containing selected members with roles,
            team score, diversity score, coverage score, and selection metadata
        """
        data: dict[str, Any] = {"pool": pool}
        if task_requirements is not None:
            data["task_requirements"] = task_requirements
        if team_size is not None:
            data["team_size"] = team_size
        if constraints is not None:
            data["constraints"] = constraints
        if min_team_size is not None:
            data["min_team_size"] = min_team_size
        if max_team_size is not None:
            data["max_team_size"] = max_team_size
        if required_roles is not None:
            data["required_roles"] = required_roles
        if excluded_agents is not None:
            data["excluded_agents"] = excluded_agents
        if diversity_weight is not None:
            data["diversity_weight"] = diversity_weight
        if selector is not None:
            data["selector"] = selector
        if role_assigner is not None:
            data["role_assigner"] = role_assigner
        return self._client.request("POST", "/api/v1/agent-selection/select-team", json=data)

    def assign_roles(
        self,
        members: list[str],
        roles: list[str],
        task_context: str | None = None,
        assigner: str | None = None,
    ) -> dict[str, Any]:
        """
        Assign roles to a set of team members.

        Given a list of agents and available roles, determines the
        optimal role assignment based on agent capabilities.

        Args:
            members: List of agent identifiers to assign roles to
            roles: List of roles to assign
            task_context: Context for role assignment decisions (optional)
            assigner: Specific role assigner plugin to use (optional)

        Returns:
            Dict with assignments mapping agents to roles, with reasoning
        """
        data: dict[str, Any] = {
            "members": members,
            "roles": roles,
        }
        if task_context is not None:
            data["task_context"] = task_context
        if assigner is not None:
            data["assigner"] = assigner
        return self._client.request("POST", "/api/v1/agent-selection/assign-roles", json=data)

    # ===========================================================================
    # History
    # ===========================================================================

    def get_selection_history(
        self,
        limit: int | None = None,
        since: str | None = None,
    ) -> dict[str, Any]:
        """
        Get agent selection history.

        Returns past selection decisions for analysis and auditing.

        Args:
            limit: Maximum number of history entries to return (optional)
            since: ISO 8601 timestamp to filter history from (optional)

        Returns:
            Dict with history array containing past selection decisions
        """
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if since is not None:
            params["since"] = since
        return self._client.request("GET", "/api/v1/agent-selection/history", params=params)

    def team_selection(self, **kwargs: Any) -> dict[str, Any]:
        """
        Get team selection overview.

        GET /api/v1/team-selection

        Returns:
            Dict with team selection data
        """
        return self._client.request("GET", "/api/v1/team-selection", params=kwargs or None)


class AsyncAgentSelectionAPI:
    """
    Asynchronous Agent Selection API.

    Provides async methods for agent team selection and scoring:
    - List available selection plugins
    - Get default plugin configurations
    - Score agents for specific tasks
    - Select optimal teams with role assignment
    - Track selection history

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     plugins = await client.agent_selection.list_plugins()
        ...     scores = await client.agent_selection.score_agents(
        ...         agents=["claude", "gpt-4", "gemini"],
        ...         context="security code review",
        ...         dimensions=["accuracy", "speed", "cost"],
        ...     )
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # Plugin Discovery
    # ===========================================================================

    async def list_plugins(self) -> dict[str, Any]:
        """List all available selection plugins."""
        return await self._client.request("GET", "/api/v1/agent-selection/plugins")

    async def get_defaults(self) -> dict[str, Any]:
        """Get default plugin configuration."""
        return await self._client.request("GET", "/api/v1/agent-selection/defaults")

    # ===========================================================================
    # Agent Scoring
    # ===========================================================================

    async def score_agents(
        self,
        agents: list[str],
        context: str | None = None,
        dimensions: list[str] | None = None,
        scorer: str | None = None,
        weights: dict[str, float] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Score agents for a specific task or context."""
        data: dict[str, Any] = {"agents": agents}
        if context is not None:
            data["context"] = context
        if dimensions is not None:
            data["dimensions"] = dimensions
        if scorer is not None:
            data["scorer"] = scorer
        if weights is not None:
            data["weights"] = weights
        if top_k is not None:
            data["top_k"] = top_k
        return await self._client.request("POST", "/api/v1/agent-selection/score", json=data)

    async def get_best_agent(
        self,
        pool: list[str],
        task_type: str,
        context: str | None = None,
    ) -> dict[str, Any]:
        """Get the best agent for a specific task from a pool."""
        data: dict[str, Any] = {
            "pool": pool,
            "task_type": task_type,
        }
        if context is not None:
            data["context"] = context
        return await self._client.request("POST", "/api/v1/agent-selection/best", json=data)

    # ===========================================================================
    # Team Selection
    # ===========================================================================

    async def select_team(
        self,
        pool: list[str],
        task_requirements: dict[str, Any] | None = None,
        team_size: int | None = None,
        constraints: dict[str, Any] | None = None,
        min_team_size: int | None = None,
        max_team_size: int | None = None,
        required_roles: list[str] | None = None,
        excluded_agents: list[str] | None = None,
        diversity_weight: float | None = None,
        selector: str | None = None,
        role_assigner: str | None = None,
    ) -> dict[str, Any]:
        """Select an optimal team of agents for a task."""
        data: dict[str, Any] = {"pool": pool}
        if task_requirements is not None:
            data["task_requirements"] = task_requirements
        if team_size is not None:
            data["team_size"] = team_size
        if constraints is not None:
            data["constraints"] = constraints
        if min_team_size is not None:
            data["min_team_size"] = min_team_size
        if max_team_size is not None:
            data["max_team_size"] = max_team_size
        if required_roles is not None:
            data["required_roles"] = required_roles
        if excluded_agents is not None:
            data["excluded_agents"] = excluded_agents
        if diversity_weight is not None:
            data["diversity_weight"] = diversity_weight
        if selector is not None:
            data["selector"] = selector
        if role_assigner is not None:
            data["role_assigner"] = role_assigner
        return await self._client.request("POST", "/api/v1/agent-selection/select-team", json=data)

    async def assign_roles(
        self,
        members: list[str],
        roles: list[str],
        task_context: str | None = None,
        assigner: str | None = None,
    ) -> dict[str, Any]:
        """Assign roles to a set of team members."""
        data: dict[str, Any] = {
            "members": members,
            "roles": roles,
        }
        if task_context is not None:
            data["task_context"] = task_context
        if assigner is not None:
            data["assigner"] = assigner
        return await self._client.request("POST", "/api/v1/agent-selection/assign-roles", json=data)

    # ===========================================================================
    # History
    # ===========================================================================

    async def get_selection_history(
        self,
        limit: int | None = None,
        since: str | None = None,
    ) -> dict[str, Any]:
        """Get agent selection history."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if since is not None:
            params["since"] = since
        return await self._client.request("GET", "/api/v1/agent-selection/history", params=params)

    async def team_selection(self, **kwargs: Any) -> dict[str, Any]:
        """Get team selection overview. GET /api/v1/team-selection"""
        return await self._client.request("GET", "/api/v1/team-selection", params=kwargs or None)
