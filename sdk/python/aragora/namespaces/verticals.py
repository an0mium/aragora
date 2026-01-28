"""
Verticals Namespace API

Provides domain-specific vertical management:
- List and configure industry verticals
- Get vertical-specific tools and compliance
- Create specialized agents and debates

Features:
- Industry-specific configurations
- Compliance framework management
- Specialized agent creation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


VerticalId = Literal[
    "software",
    "legal",
    "healthcare",
    "accounting",
    "finance",
    "research",
    "education",
    "marketing",
    "operations",
    "general",
]

ComplianceLevel = Literal["advisory", "warning", "enforced"]
ConsensusType = Literal["majority", "unanimous", "weighted"]


class VerticalsAPI:
    """
    Synchronous Verticals API.

    Provides methods for managing industry verticals:
    - List and configure verticals
    - Get vertical-specific tools and compliance frameworks
    - Create specialized agents and debates

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai", api_key="...")
        >>> verticals = client.verticals.list()
        >>> tools = client.verticals.get_tools("legal")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def list(self, keyword: str | None = None) -> dict[str, Any]:
        """
        List all available verticals.

        Args:
            keyword: Optional keyword filter

        Returns:
            Dict with verticals list
        """
        params = {"keyword": keyword} if keyword else None
        return self._client.request("GET", "/api/verticals", params=params)

    def get(self, vertical_id: VerticalId) -> dict[str, Any]:
        """
        Get vertical details.

        Args:
            vertical_id: The vertical ID

        Returns:
            Dict with full vertical details
        """
        return self._client.request("GET", f"/api/verticals/{vertical_id}")

    def get_tools(self, vertical_id: VerticalId) -> dict[str, Any]:
        """
        Get tools available for a vertical.

        Args:
            vertical_id: The vertical ID

        Returns:
            Dict with tools list
        """
        return self._client.request("GET", f"/api/verticals/{vertical_id}/tools")

    def get_compliance(
        self,
        vertical_id: VerticalId,
        level: ComplianceLevel | None = None,
    ) -> dict[str, Any]:
        """
        Get compliance frameworks for a vertical.

        Args:
            vertical_id: The vertical ID
            level: Filter by compliance level

        Returns:
            Dict with frameworks list
        """
        params = {"level": level} if level else None
        return self._client.request(
            "GET", f"/api/verticals/{vertical_id}/compliance", params=params
        )

    def suggest(self, task: str) -> dict[str, Any]:
        """
        Suggest the best vertical for a task.

        Args:
            task: Task description

        Returns:
            Dict with suggestions ranked by confidence
        """
        return self._client.request("GET", "/api/verticals/suggest", params={"task": task})

    def create_agent(
        self,
        vertical_id: VerticalId,
        name: str,
        model: str | None = None,
        role: str | None = None,
        traits: list[str] | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a vertical-specialized agent.

        Args:
            vertical_id: The vertical ID
            name: Agent name
            model: Model to use
            role: Agent role
            traits: Personality traits
            description: Agent description

        Returns:
            Dict with created agent details
        """
        data: dict[str, Any] = {"name": name}
        if model:
            data["model"] = model
        if role:
            data["role"] = role
        if traits:
            data["traits"] = traits
        if description:
            data["description"] = description
        return self._client.request("POST", f"/api/verticals/{vertical_id}/agent", json=data)

    def create_debate(
        self,
        vertical_id: VerticalId,
        topic: str,
        agent_name: str | None = None,
        rounds: int | None = None,
        consensus: ConsensusType | None = None,
        additional_agents: list[str] | None = None,
        context: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a vertical-specialized debate.

        Args:
            vertical_id: The vertical ID
            topic: Debate topic
            agent_name: Primary agent
            rounds: Number of rounds
            consensus: Consensus type
            additional_agents: Additional agents
            context: Context information

        Returns:
            Dict with created debate details
        """
        data: dict[str, Any] = {"topic": topic}
        if agent_name:
            data["agent_name"] = agent_name
        if rounds:
            data["rounds"] = rounds
        if consensus:
            data["consensus"] = consensus
        if additional_agents:
            data["additional_agents"] = additional_agents
        if context:
            data["context"] = context
        return self._client.request("POST", f"/api/verticals/{vertical_id}/debate", json=data)

    def update_config(
        self,
        vertical_id: VerticalId,
        enabled: bool | None = None,
        model_config: dict[str, Any] | None = None,
        default_traits: list[str] | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """
        Update vertical configuration.

        Args:
            vertical_id: The vertical ID
            enabled: Enable/disable vertical
            model_config: Model configuration
            default_traits: Default agent traits
            description: Updated description

        Returns:
            Dict with updated vertical details
        """
        data: dict[str, Any] = {}
        if enabled is not None:
            data["enabled"] = enabled
        if model_config:
            data["model_config"] = model_config
        if default_traits:
            data["default_traits"] = default_traits
        if description:
            data["description"] = description
        return self._client.request("PUT", f"/api/verticals/{vertical_id}/config", json=data)


class AsyncVerticalsAPI:
    """
    Asynchronous Verticals API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     verticals = await client.verticals.list()
        ...     debate = await client.verticals.create_debate("healthcare", topic="...")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def list(self, keyword: str | None = None) -> dict[str, Any]:
        """List all available verticals."""
        params = {"keyword": keyword} if keyword else None
        return await self._client.request("GET", "/api/verticals", params=params)

    async def get(self, vertical_id: VerticalId) -> dict[str, Any]:
        """Get vertical details."""
        return await self._client.request("GET", f"/api/verticals/{vertical_id}")

    async def get_tools(self, vertical_id: VerticalId) -> dict[str, Any]:
        """Get tools available for a vertical."""
        return await self._client.request("GET", f"/api/verticals/{vertical_id}/tools")

    async def get_compliance(
        self,
        vertical_id: VerticalId,
        level: ComplianceLevel | None = None,
    ) -> dict[str, Any]:
        """Get compliance frameworks for a vertical."""
        params = {"level": level} if level else None
        return await self._client.request(
            "GET", f"/api/verticals/{vertical_id}/compliance", params=params
        )

    async def suggest(self, task: str) -> dict[str, Any]:
        """Suggest the best vertical for a task."""
        return await self._client.request("GET", "/api/verticals/suggest", params={"task": task})

    async def create_agent(
        self,
        vertical_id: VerticalId,
        name: str,
        model: str | None = None,
        role: str | None = None,
        traits: list[str] | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Create a vertical-specialized agent."""
        data: dict[str, Any] = {"name": name}
        if model:
            data["model"] = model
        if role:
            data["role"] = role
        if traits:
            data["traits"] = traits
        if description:
            data["description"] = description
        return await self._client.request("POST", f"/api/verticals/{vertical_id}/agent", json=data)

    async def create_debate(
        self,
        vertical_id: VerticalId,
        topic: str,
        agent_name: str | None = None,
        rounds: int | None = None,
        consensus: ConsensusType | None = None,
        additional_agents: list[str] | None = None,
        context: str | None = None,
    ) -> dict[str, Any]:
        """Create a vertical-specialized debate."""
        data: dict[str, Any] = {"topic": topic}
        if agent_name:
            data["agent_name"] = agent_name
        if rounds:
            data["rounds"] = rounds
        if consensus:
            data["consensus"] = consensus
        if additional_agents:
            data["additional_agents"] = additional_agents
        if context:
            data["context"] = context
        return await self._client.request("POST", f"/api/verticals/{vertical_id}/debate", json=data)

    async def update_config(
        self,
        vertical_id: VerticalId,
        enabled: bool | None = None,
        model_config: dict[str, Any] | None = None,
        default_traits: list[str] | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Update vertical configuration."""
        data: dict[str, Any] = {}
        if enabled is not None:
            data["enabled"] = enabled
        if model_config:
            data["model_config"] = model_config
        if default_traits:
            data["default_traits"] = default_traits
        if description:
            data["description"] = description
        return await self._client.request("PUT", f"/api/verticals/{vertical_id}/config", json=data)
