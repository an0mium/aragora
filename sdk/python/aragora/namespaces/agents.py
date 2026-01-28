"""
Agents Namespace API

Provides methods for listing agents and viewing their performance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class AgentsAPI:
    """
    Synchronous Agents API.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> agents = client.agents.list()
        >>> performance = client.agents.get_performance("claude")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def list(self) -> dict[str, Any]:
        """
        List all available agents.

        Returns:
            List of agents with their capabilities
        """
        return self._client.request("GET", "/api/v1/agents")

    def get(self, agent_name: str) -> dict[str, Any]:
        """
        Get details for a specific agent.

        Args:
            agent_name: The agent's name

        Returns:
            Agent details
        """
        return self._client.request("GET", f"/api/v1/agents/{agent_name}")

    def get_performance(self, agent_name: str) -> dict[str, Any]:
        """
        Get performance metrics for an agent.

        Args:
            agent_name: The agent's name

        Returns:
            Performance metrics including ELO rating, win rate, etc.
        """
        return self._client.request("GET", f"/api/v1/agents/{agent_name}/performance")

    def get_calibration(self, agent_name: str) -> dict[str, Any]:
        """
        Get calibration data for an agent.

        Args:
            agent_name: The agent's name

        Returns:
            Calibration metrics
        """
        return self._client.request("GET", f"/api/v1/agents/{agent_name}/calibration")

    def get_relationships(self, agent_name: str) -> dict[str, Any]:
        """
        Get relationship data between agents.

        Args:
            agent_name: The agent's name

        Returns:
            Relationship data with other agents
        """
        return self._client.request("GET", f"/api/v1/agents/{agent_name}/relationships")

    def get_history(
        self,
        agent_name: str,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Get debate history for an agent.

        Args:
            agent_name: The agent's name
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            Debate history
        """
        return self._client.request(
            "GET",
            f"/api/v1/agents/{agent_name}/history",
            params={"limit": limit, "offset": offset},
        )

    def compare(self, agent1: str, agent2: str) -> dict[str, Any]:
        """
        Compare two agents' performance.

        Args:
            agent1: First agent name
            agent2: Second agent name

        Returns:
            Comparison data
        """
        return self._client.request(
            "GET",
            "/api/v1/agents/compare",
            params={"agent1": agent1, "agent2": agent2},
        )

    def select_team(
        self,
        task: str,
        team_size: int = 3,
        strategy: str = "balanced",
    ) -> dict[str, Any]:
        """
        Select an optimal team of agents for a task.

        Args:
            task: The task description
            team_size: Number of agents to select
            strategy: Selection strategy (balanced, competitive, etc.)

        Returns:
            Selected team with rationale
        """
        return self._client.request(
            "POST",
            "/api/v1/agents/select-team",
            json={"task": task, "team_size": team_size, "strategy": strategy},
        )

    # =========================================================================
    # Agent Profile & Identity (singular /agent/ endpoints)
    # =========================================================================

    def get_profile(self, name: str) -> dict[str, Any]:
        """Get agent's full profile."""
        return self._client.request("GET", f"/api/v1/agent/{name}/profile")

    def get_persona(self, name: str) -> dict[str, Any]:
        """Get agent's persona configuration."""
        return self._client.request("GET", f"/api/v1/agent/{name}/persona")

    def delete_persona(self, name: str) -> dict[str, Any]:
        """Delete agent's custom persona."""
        return self._client.request("DELETE", f"/api/v1/agent/{name}/persona")

    def get_grounded_persona(self, name: str) -> dict[str, Any]:
        """Get agent's grounded (evidence-based) persona."""
        return self._client.request("GET", f"/api/v1/agent/{name}/grounded-persona")

    def get_identity_prompt(self, name: str) -> dict[str, Any]:
        """Get agent's identity prompt."""
        return self._client.request("GET", f"/api/v1/agent/{name}/identity-prompt")

    # =========================================================================
    # Agent Analytics
    # =========================================================================

    def get_accuracy(self, name: str) -> dict[str, Any]:
        """Get agent's accuracy metrics."""
        return self._client.request("GET", f"/api/v1/agent/{name}/accuracy")

    def get_consistency(self, name: str) -> dict[str, Any]:
        """Get agent's consistency metrics."""
        return self._client.request("GET", f"/api/v1/agent/{name}/consistency")

    def get_calibration_curve(self, name: str) -> dict[str, Any]:
        """Get agent's calibration curve data."""
        return self._client.request("GET", f"/api/v1/agent/{name}/calibration-curve")

    def get_calibration_summary(self, name: str) -> dict[str, Any]:
        """Get agent's calibration summary."""
        return self._client.request("GET", f"/api/v1/agent/{name}/calibration-summary")

    def get_reputation(self, name: str) -> dict[str, Any]:
        """Get agent's reputation score."""
        return self._client.request("GET", f"/api/v1/agent/{name}/reputation")

    def get_domains(self, name: str) -> dict[str, Any]:
        """Get domains the agent specializes in."""
        return self._client.request("GET", f"/api/v1/agent/{name}/domains")

    # =========================================================================
    # Agent Relationships & Network
    # =========================================================================

    def get_allies(self, name: str) -> dict[str, Any]:
        """Get agents that frequently agree with this agent."""
        return self._client.request("GET", f"/api/v1/agent/{name}/allies")

    def get_rivals(self, name: str) -> dict[str, Any]:
        """Get agents that frequently disagree with this agent."""
        return self._client.request("GET", f"/api/v1/agent/{name}/rivals")

    def get_network(self, name: str) -> dict[str, Any]:
        """Get agent's relationship network graph."""
        return self._client.request("GET", f"/api/v1/agent/{name}/network")

    # =========================================================================
    # Agent History & Events
    # =========================================================================

    def get_flips(self, name: str, limit: int = 20) -> dict[str, Any]:
        """Get instances where agent changed position."""
        return self._client.request("GET", f"/api/v1/agent/{name}/flips", params={"limit": limit})

    def get_moments(self, name: str, limit: int = 10) -> dict[str, Any]:
        """Get agent's notable debate moments."""
        return self._client.request("GET", f"/api/v1/agent/{name}/moments", params={"limit": limit})

    def compare_agents(self, agent1: str, agent2: str) -> dict[str, Any]:
        """Compare two agents using singular endpoint."""
        return self._client.request(
            "GET", "/api/v1/agent/compare", params={"agent1": agent1, "agent2": agent2}
        )


class AsyncAgentsAPI:
    """
    Asynchronous Agents API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     agents = await client.agents.list()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list(self) -> dict[str, Any]:
        """List all available agents."""
        return await self._client.request("GET", "/api/v1/agents")

    async def get(self, agent_name: str) -> dict[str, Any]:
        """Get details for a specific agent."""
        return await self._client.request("GET", f"/api/v1/agents/{agent_name}")

    async def get_performance(self, agent_name: str) -> dict[str, Any]:
        """Get performance metrics for an agent."""
        return await self._client.request("GET", f"/api/v1/agents/{agent_name}/performance")

    async def get_calibration(self, agent_name: str) -> dict[str, Any]:
        """Get calibration data for an agent."""
        return await self._client.request("GET", f"/api/v1/agents/{agent_name}/calibration")

    async def get_relationships(self, agent_name: str) -> dict[str, Any]:
        """Get relationship data between agents."""
        return await self._client.request("GET", f"/api/v1/agents/{agent_name}/relationships")

    async def get_history(
        self,
        agent_name: str,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Get debate history for an agent."""
        return await self._client.request(
            "GET",
            f"/api/v1/agents/{agent_name}/history",
            params={"limit": limit, "offset": offset},
        )

    async def compare(self, agent1: str, agent2: str) -> dict[str, Any]:
        """Compare two agents' performance."""
        return await self._client.request(
            "GET",
            "/api/v1/agents/compare",
            params={"agent1": agent1, "agent2": agent2},
        )

    async def select_team(
        self,
        task: str,
        team_size: int = 3,
        strategy: str = "balanced",
    ) -> dict[str, Any]:
        """Select an optimal team of agents for a task."""
        return await self._client.request(
            "POST",
            "/api/v1/agents/select-team",
            json={"task": task, "team_size": team_size, "strategy": strategy},
        )

    # Agent Profile & Identity
    async def get_profile(self, name: str) -> dict[str, Any]:
        """Get agent's full profile."""
        return await self._client.request("GET", f"/api/v1/agent/{name}/profile")

    async def get_persona(self, name: str) -> dict[str, Any]:
        """Get agent's persona configuration."""
        return await self._client.request("GET", f"/api/v1/agent/{name}/persona")

    async def delete_persona(self, name: str) -> dict[str, Any]:
        """Delete agent's custom persona."""
        return await self._client.request("DELETE", f"/api/v1/agent/{name}/persona")

    async def get_grounded_persona(self, name: str) -> dict[str, Any]:
        """Get agent's grounded persona."""
        return await self._client.request("GET", f"/api/v1/agent/{name}/grounded-persona")

    async def get_identity_prompt(self, name: str) -> dict[str, Any]:
        """Get agent's identity prompt."""
        return await self._client.request("GET", f"/api/v1/agent/{name}/identity-prompt")

    # Agent Analytics
    async def get_accuracy(self, name: str) -> dict[str, Any]:
        """Get agent's accuracy metrics."""
        return await self._client.request("GET", f"/api/v1/agent/{name}/accuracy")

    async def get_consistency(self, name: str) -> dict[str, Any]:
        """Get agent's consistency metrics."""
        return await self._client.request("GET", f"/api/v1/agent/{name}/consistency")

    async def get_calibration_curve(self, name: str) -> dict[str, Any]:
        """Get agent's calibration curve data."""
        return await self._client.request("GET", f"/api/v1/agent/{name}/calibration-curve")

    async def get_calibration_summary(self, name: str) -> dict[str, Any]:
        """Get agent's calibration summary."""
        return await self._client.request("GET", f"/api/v1/agent/{name}/calibration-summary")

    async def get_reputation(self, name: str) -> dict[str, Any]:
        """Get agent's reputation score."""
        return await self._client.request("GET", f"/api/v1/agent/{name}/reputation")

    async def get_domains(self, name: str) -> dict[str, Any]:
        """Get domains the agent specializes in."""
        return await self._client.request("GET", f"/api/v1/agent/{name}/domains")

    # Agent Relationships & Network
    async def get_allies(self, name: str) -> dict[str, Any]:
        """Get agents that frequently agree with this agent."""
        return await self._client.request("GET", f"/api/v1/agent/{name}/allies")

    async def get_rivals(self, name: str) -> dict[str, Any]:
        """Get agents that frequently disagree with this agent."""
        return await self._client.request("GET", f"/api/v1/agent/{name}/rivals")

    async def get_network(self, name: str) -> dict[str, Any]:
        """Get agent's relationship network graph."""
        return await self._client.request("GET", f"/api/v1/agent/{name}/network")

    # Agent History & Events
    async def get_flips(self, name: str, limit: int = 20) -> dict[str, Any]:
        """Get instances where agent changed position."""
        return await self._client.request(
            "GET", f"/api/v1/agent/{name}/flips", params={"limit": limit}
        )

    async def get_moments(self, name: str, limit: int = 10) -> dict[str, Any]:
        """Get agent's notable debate moments."""
        return await self._client.request(
            "GET", f"/api/v1/agent/{name}/moments", params={"limit": limit}
        )

    async def compare_agents(self, agent1: str, agent2: str) -> dict[str, Any]:
        """Compare two agents using singular endpoint."""
        return await self._client.request(
            "GET", "/api/v1/agent/compare", params={"agent1": agent1, "agent2": agent2}
        )
