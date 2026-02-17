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

    def list_all(self, page_size: int = 20) -> SyncPaginator:
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
    # Agent Configurations (YAML-based)
    # =========================================================================

    def list_configs(
        self,
        priority: str | None = None,
        role: str | None = None,
    ) -> dict[str, Any]:
        """
        List available YAML agent configurations.

        Args:
            priority: Filter by priority level (low, normal, high, critical)
            role: Filter by role (proposer, critic, synthesizer, judge)

        Returns:
            List of configuration summaries
        """
        params: dict[str, Any] = {}
        if priority:
            params["priority"] = priority
        if role:
            params["role"] = role
        return self._client.request("GET", "/api/v1/agents/configs", params=params or None)

    def get_config(self, config_name: str) -> dict[str, Any]:
        """
        Get a specific agent configuration by name.

        Args:
            config_name: Name of the agent configuration

        Returns:
            Agent configuration details including expertise, capabilities, and model settings
        """
        return self._client.request("GET", f"/api/v1/agents/configs/{config_name}")

    def search_configs(
        self,
        query: str | None = None,
        expertise: str | None = None,
        capability: str | None = None,
        tag: str | None = None,
    ) -> dict[str, Any]:
        """
        Search agent configurations by expertise, capability, or tag.

        Args:
            query: Free-text search query
            expertise: Filter by expertise area
            capability: Filter by capability
            tag: Filter by tag

        Returns:
            Matching configurations
        """
        params: dict[str, Any] = {}
        if query:
            params["q"] = query
        if expertise:
            params["expertise"] = expertise
        if capability:
            params["capability"] = capability
        if tag:
            params["tag"] = tag
        return self._client.request("GET", "/api/v1/agents/configs/search", params=params or None)

    def create_from_config(self, config_name: str) -> dict[str, Any]:
        """
        Create an agent from a named YAML configuration.

        Args:
            config_name: Name of the agent configuration to instantiate

        Returns:
            Created agent details
        """
        return self._client.request("POST", f"/api/v1/agents/configs/{config_name}/create")

    def reload_configs(self) -> dict[str, Any]:
        """
        Reload all agent configurations from disk.

        Requires admin role.

        Returns:
            Reload result with count of loaded configurations
        """
        return self._client.request("POST", "/api/v1/agents/configs/reload")

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

    def compare_leaderboard(self, agent1: str, agent2: str) -> dict[str, Any]:
        """Compare two agents on the leaderboard."""
        return self._client.request("GET", "/api/v1/leaderboard/compare", params={"agent1": agent1, "agent2": agent2})

    def get_leaderboard_domains(self) -> dict[str, Any]:
        """Get leaderboard domain breakdown."""
        return self._client.request("GET", "/api/v1/leaderboard/domains")

    def get_leaderboard_movers(self, period: str = "7d") -> dict[str, Any]:
        """Get biggest movers on the leaderboard."""
        return self._client.request("GET", "/api/v1/leaderboard/movers", params={"period": period})


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

    def list_all(self, page_size: int = 20) -> AsyncPaginator:
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

    # Agent Configurations (YAML-based)
    async def list_configs(
        self,
        priority: str | None = None,
        role: str | None = None,
    ) -> dict[str, Any]:
        """List available YAML agent configurations."""
        params: dict[str, Any] = {}
        if priority:
            params["priority"] = priority
        if role:
            params["role"] = role
        return await self._client.request("GET", "/api/v1/agents/configs", params=params or None)

    async def get_config(self, config_name: str) -> dict[str, Any]:
        """Get a specific agent configuration by name."""
        return await self._client.request("GET", f"/api/v1/agents/configs/{config_name}")

    async def search_configs(
        self,
        query: str | None = None,
        expertise: str | None = None,
        capability: str | None = None,
        tag: str | None = None,
    ) -> dict[str, Any]:
        """Search agent configurations by expertise, capability, or tag."""
        params: dict[str, Any] = {}
        if query:
            params["q"] = query
        if expertise:
            params["expertise"] = expertise
        if capability:
            params["capability"] = capability
        if tag:
            params["tag"] = tag
        return await self._client.request(
            "GET", "/api/v1/agents/configs/search", params=params or None
        )

    async def create_from_config(self, config_name: str) -> dict[str, Any]:
        """Create an agent from a named YAML configuration."""
        return await self._client.request("POST", f"/api/v1/agents/configs/{config_name}/create")

    async def reload_configs(self) -> dict[str, Any]:
        """Reload all agent configurations from disk. Requires admin role."""
        return await self._client.request("POST", "/api/v1/agents/configs/reload")

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

    async def compare_leaderboard(self, agent1: str, agent2: str) -> dict[str, Any]:
        """Compare two agents on the leaderboard."""
        return await self._client.request("GET", "/api/v1/leaderboard/compare", params={"agent1": agent1, "agent2": agent2})

    async def get_leaderboard_domains(self) -> dict[str, Any]:
        """Get leaderboard domain breakdown."""
        return await self._client.request("GET", "/api/v1/leaderboard/domains")

    async def get_leaderboard_movers(self, period: str = "7d") -> dict[str, Any]:
        """Get biggest movers on the leaderboard."""
        return await self._client.request("GET", "/api/v1/leaderboard/movers", params={"period": period})
