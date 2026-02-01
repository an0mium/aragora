"""
Agents Namespace API

Provides methods for listing agents and viewing their performance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient
    from ..pagination import AsyncPaginator, SyncPaginator


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

    def list_all(self, page_size: int = 20) -> "SyncPaginator":
        """
        Iterate through all agents with automatic pagination.

        Args:
            page_size: Number of agents per page (default 20)

        Returns:
            SyncPaginator yielding agent dictionaries

        Example::

            for agent in client.agents.list_all():
                print(agent["name"])
        """
        from ..pagination import SyncPaginator

        return SyncPaginator(self._client, "/api/v1/agents", {}, page_size)

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

    # =========================================================================
    # Health & Availability
    # =========================================================================

    def list_health(self) -> dict[str, Any]:
        """Get health status for all agents."""
        return self._client.request("GET", "/api/agents/health")

    def list_availability(self) -> dict[str, Any]:
        """Get availability status for all agents."""
        return self._client.request("GET", "/api/agents/availability")

    def list_local_agents(self) -> dict[str, Any]:
        """List locally available agents (Ollama, etc.)."""
        return self._client.request("GET", "/api/agents/local")

    def get_local_status(self) -> dict[str, Any]:
        """Get status of local agent providers."""
        return self._client.request("GET", "/api/agents/local/status")

    # =========================================================================
    # Agent Details
    # =========================================================================

    def get_elo(self, name: str) -> dict[str, Any]:
        """Get agent's ELO rating."""
        return self._client.request("GET", f"/api/agent/{name}/elo")

    def get_head_to_head(self, name: str, opponent: str) -> dict[str, Any]:
        """Get head-to-head statistics against another agent."""
        return self._client.request("GET", f"/api/agent/{name}/head-to-head/{opponent}")

    def get_opponent_briefing(self, name: str, opponent: str) -> dict[str, Any]:
        """Get strategic briefing about an opponent."""
        return self._client.request("GET", f"/api/agent/{name}/opponent-briefing/{opponent}")

    def get_positions(self, name: str) -> dict[str, Any]:
        """Get agent's position history across debates."""
        return self._client.request("GET", f"/api/agent/{name}/positions")

    def get_introspection(self, name: str) -> dict[str, Any]:
        """Get agent's self-awareness data."""
        return self._client.request("GET", f"/api/agent/{name}/introspect")

    # =========================================================================
    # Leaderboard & Analytics
    # =========================================================================

    def get_leaderboard(self, view: str = "overall") -> dict[str, Any]:
        """Get agent leaderboard."""
        return self._client.request("GET", "/api/leaderboard", params={"view": view})

    def get_recent_matches(self, limit: int = 20) -> dict[str, Any]:
        """Get recent debate matches."""
        return self._client.request("GET", "/api/matches/recent", params={"limit": limit})

    def get_recent_flips(self, limit: int = 20) -> dict[str, Any]:
        """Get recent position flips across all agents."""
        return self._client.request("GET", "/api/flips/recent", params={"limit": limit})

    def get_flips_summary(self) -> dict[str, Any]:
        """Get aggregate flip statistics."""
        return self._client.request("GET", "/api/flips/summary")

    def get_calibration_leaderboard(self) -> dict[str, Any]:
        """Get calibration leaderboard."""
        return self._client.request("GET", "/api/calibration/leaderboard")

    def get_rankings(
        self,
        domain: str | None = None,
        period: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get agent rankings with optional filters."""
        params: dict[str, Any] = {"limit": limit}
        if domain:
            params["domain"] = domain
        if period:
            params["period"] = period
        return self._client.request("GET", "/api/v1/rankings", params=params)

    # =========================================================================
    # Agent Lifecycle (Control Plane)
    # =========================================================================

    def register(self, agent_id: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Register an agent with the control plane."""
        payload: dict[str, Any] = {"agent_id": agent_id}
        if options:
            payload.update(options)
        return self._client.request("POST", "/api/v1/control-plane/agents", json=payload)

    def unregister(self, agent_id: str) -> dict[str, Any]:
        """Unregister an agent from the control plane."""
        return self._client.request("DELETE", f"/api/v1/control-plane/agents/{agent_id}")

    def heartbeat(self, agent_id: str, status: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send heartbeat for an agent."""
        payload: dict[str, Any] = {}
        if status:
            payload["status"] = status
        return self._client.request(
            "POST", f"/api/v1/control-plane/agents/{agent_id}/heartbeat", json=payload
        )

    # =========================================================================
    # Agent Management
    # =========================================================================

    def enable(self, name: str) -> dict[str, Any]:
        """Enable an agent for participation."""
        return self._client.request("POST", f"/api/v1/agents/{name}/enable", json={})

    def disable(self, name: str) -> dict[str, Any]:
        """Disable an agent from participation."""
        return self._client.request("POST", f"/api/v1/agents/{name}/disable", json={})

    def calibrate(self, name: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Trigger calibration for an agent."""
        return self._client.request("POST", f"/api/v1/agents/{name}/calibrate", json=options or {})

    def get_quota(self, name: str) -> dict[str, Any]:
        """Get rate quota for an agent."""
        return self._client.request("GET", f"/api/v1/agents/{name}/quota")

    def set_quota(self, name: str, quota: dict[str, Any]) -> dict[str, Any]:
        """Set rate quota for an agent."""
        return self._client.request("PUT", f"/api/v1/agents/{name}/quota", json=quota)

    def update_elo(self, name: str, change: float, reason: str | None = None) -> dict[str, Any]:
        """Manually update agent's ELO rating."""
        payload: dict[str, Any] = {"change": change}
        if reason:
            payload["reason"] = reason
        return self._client.request("POST", f"/api/v1/agents/{name}/elo", json=payload)

    def get_stats(self) -> dict[str, Any]:
        """Get aggregate statistics for all agents."""
        return self._client.request("GET", "/api/v1/agents/stats")


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

    def list_all(self, page_size: int = 20) -> "AsyncPaginator":
        """
        Iterate through all agents with automatic pagination.

        Args:
            page_size: Number of agents per page (default 20)

        Returns:
            AsyncPaginator yielding agent dictionaries

        Example::

            async for agent in client.agents.list_all():
                print(agent["name"])
        """
        from ..pagination import AsyncPaginator

        return AsyncPaginator(self._client, "/api/v1/agents", {}, page_size)

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

    # =========================================================================
    # Health & Availability
    # =========================================================================

    async def list_health(self) -> dict[str, Any]:
        """Get health status for all agents."""
        return await self._client.request("GET", "/api/agents/health")

    async def list_availability(self) -> dict[str, Any]:
        """Get availability status for all agents."""
        return await self._client.request("GET", "/api/agents/availability")

    async def list_local_agents(self) -> dict[str, Any]:
        """List locally available agents (Ollama, etc.)."""
        return await self._client.request("GET", "/api/agents/local")

    async def get_local_status(self) -> dict[str, Any]:
        """Get status of local agent providers."""
        return await self._client.request("GET", "/api/agents/local/status")

    # =========================================================================
    # Agent Details
    # =========================================================================

    async def get_elo(self, name: str) -> dict[str, Any]:
        """Get agent's ELO rating."""
        return await self._client.request("GET", f"/api/agent/{name}/elo")

    async def get_head_to_head(self, name: str, opponent: str) -> dict[str, Any]:
        """Get head-to-head statistics against another agent."""
        return await self._client.request("GET", f"/api/agent/{name}/head-to-head/{opponent}")

    async def get_opponent_briefing(self, name: str, opponent: str) -> dict[str, Any]:
        """Get strategic briefing about an opponent."""
        return await self._client.request("GET", f"/api/agent/{name}/opponent-briefing/{opponent}")

    async def get_positions(self, name: str) -> dict[str, Any]:
        """Get agent's position history across debates."""
        return await self._client.request("GET", f"/api/agent/{name}/positions")

    async def get_introspection(self, name: str) -> dict[str, Any]:
        """Get agent's self-awareness data."""
        return await self._client.request("GET", f"/api/agent/{name}/introspect")

    # =========================================================================
    # Leaderboard & Analytics
    # =========================================================================

    async def get_leaderboard(self, view: str = "overall") -> dict[str, Any]:
        """Get agent leaderboard."""
        return await self._client.request("GET", "/api/leaderboard", params={"view": view})

    async def get_recent_matches(self, limit: int = 20) -> dict[str, Any]:
        """Get recent debate matches."""
        return await self._client.request("GET", "/api/matches/recent", params={"limit": limit})

    async def get_recent_flips(self, limit: int = 20) -> dict[str, Any]:
        """Get recent position flips across all agents."""
        return await self._client.request("GET", "/api/flips/recent", params={"limit": limit})

    async def get_flips_summary(self) -> dict[str, Any]:
        """Get aggregate flip statistics."""
        return await self._client.request("GET", "/api/flips/summary")

    async def get_calibration_leaderboard(self) -> dict[str, Any]:
        """Get calibration leaderboard."""
        return await self._client.request("GET", "/api/calibration/leaderboard")

    async def get_rankings(
        self,
        domain: str | None = None,
        period: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get agent rankings with optional filters."""
        params: dict[str, Any] = {"limit": limit}
        if domain:
            params["domain"] = domain
        if period:
            params["period"] = period
        return await self._client.request("GET", "/api/v1/rankings", params=params)

    # =========================================================================
    # Agent Lifecycle (Control Plane)
    # =========================================================================

    async def register(
        self, agent_id: str, options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Register an agent with the control plane."""
        payload: dict[str, Any] = {"agent_id": agent_id}
        if options:
            payload.update(options)
        return await self._client.request("POST", "/api/v1/control-plane/agents", json=payload)

    async def unregister(self, agent_id: str) -> dict[str, Any]:
        """Unregister an agent from the control plane."""
        return await self._client.request("DELETE", f"/api/v1/control-plane/agents/{agent_id}")

    async def heartbeat(
        self, agent_id: str, status: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send heartbeat for an agent."""
        payload: dict[str, Any] = {}
        if status:
            payload["status"] = status
        return await self._client.request(
            "POST", f"/api/v1/control-plane/agents/{agent_id}/heartbeat", json=payload
        )

    # =========================================================================
    # Agent Management
    # =========================================================================

    async def enable(self, name: str) -> dict[str, Any]:
        """Enable an agent for participation."""
        return await self._client.request("POST", f"/api/v1/agents/{name}/enable", json={})

    async def disable(self, name: str) -> dict[str, Any]:
        """Disable an agent from participation."""
        return await self._client.request("POST", f"/api/v1/agents/{name}/disable", json={})

    async def calibrate(self, name: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Trigger calibration for an agent."""
        return await self._client.request(
            "POST", f"/api/v1/agents/{name}/calibrate", json=options or {}
        )

    async def get_quota(self, name: str) -> dict[str, Any]:
        """Get rate quota for an agent."""
        return await self._client.request("GET", f"/api/v1/agents/{name}/quota")

    async def set_quota(self, name: str, quota: dict[str, Any]) -> dict[str, Any]:
        """Set rate quota for an agent."""
        return await self._client.request("PUT", f"/api/v1/agents/{name}/quota", json=quota)

    async def update_elo(
        self, name: str, change: float, reason: str | None = None
    ) -> dict[str, Any]:
        """Manually update agent's ELO rating."""
        payload: dict[str, Any] = {"change": change}
        if reason:
            payload["reason"] = reason
        return await self._client.request("POST", f"/api/v1/agents/{name}/elo", json=payload)

    async def get_stats(self) -> dict[str, Any]:
        """Get aggregate statistics for all agents."""
        return await self._client.request("GET", "/api/v1/agents/stats")
