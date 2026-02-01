"""
Hybrid Debates Namespace API.

Provides methods for managing hybrid debates that combine external
and internal agents for consensus-driven decisions.

Endpoints:
    POST /api/v1/debates/hybrid          - Start a hybrid debate
    GET  /api/v1/debates/hybrid          - List hybrid debates
    GET  /api/v1/debates/hybrid/{id}     - Get hybrid debate result
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class HybridDebatesAPI:
    """
    Synchronous Hybrid Debates API.

    Provides methods for starting and managing hybrid debates that
    coordinate external agents (e.g., CrewAI, LangGraph) with internal
    verification agents to produce consensus-driven decisions.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> # Start a hybrid debate
        >>> debate = client.hybrid_debates.create(
        ...     task="Should we migrate to Kubernetes?",
        ...     external_agent="crewai-infra-team",
        ...     consensus_threshold=0.8,
        ...     max_rounds=5,
        ... )
        >>> # Get debate result
        >>> result = client.hybrid_debates.get(debate["debate_id"])
        >>> # List hybrid debates
        >>> debates = client.hybrid_debates.list(status="completed")
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    def create(
        self,
        task: str,
        external_agent: str,
        consensus_threshold: float = 0.7,
        max_rounds: int = 3,
        verification_agents: list[str] | None = None,
        domain: str = "general",
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Start a hybrid debate.

        Creates a new hybrid debate combining an external agent with
        internal verification agents for consensus-driven decisions.

        Args:
            task: The question or topic to debate.
            external_agent: Name of the registered external agent.
            consensus_threshold: Consensus threshold between 0.0 and 1.0.
            max_rounds: Maximum number of debate rounds (1-10).
            verification_agents: Internal verification agent names.
            domain: Domain context for the debate.
            config: Additional configuration.

        Returns:
            Dict with debate details including debate_id, status, and result.
        """
        data: dict[str, Any] = {
            "task": task,
            "external_agent": external_agent,
            "consensus_threshold": consensus_threshold,
            "max_rounds": max_rounds,
            "domain": domain,
        }
        if verification_agents:
            data["verification_agents"] = verification_agents
        if config:
            data["config"] = config

        return self._client.request("POST", "/api/v1/debates/hybrid", json=data)

    def get(self, debate_id: str) -> dict[str, Any]:
        """
        Get a hybrid debate result.

        Args:
            debate_id: Hybrid debate identifier.

        Returns:
            Dict with full debate details including result.
        """
        return self._client.request("GET", f"/api/v1/debates/hybrid/{debate_id}")

    def list(
        self,
        status: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """
        List hybrid debates.

        Args:
            status: Filter by debate status.
            limit: Maximum number of results (1-100).

        Returns:
            Dict with debates array and total count.
        """
        params: dict[str, Any] = {}
        if status:
            params["status"] = status
        if limit != 20:
            params["limit"] = limit

        return self._client.request("GET", "/api/v1/debates/hybrid", params=params)


class AsyncHybridDebatesAPI:
    """Asynchronous Hybrid Debates API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def create(
        self,
        task: str,
        external_agent: str,
        consensus_threshold: float = 0.7,
        max_rounds: int = 3,
        verification_agents: list[str] | None = None,
        domain: str = "general",
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Start a hybrid debate."""
        data: dict[str, Any] = {
            "task": task,
            "external_agent": external_agent,
            "consensus_threshold": consensus_threshold,
            "max_rounds": max_rounds,
            "domain": domain,
        }
        if verification_agents:
            data["verification_agents"] = verification_agents
        if config:
            data["config"] = config

        return await self._client.request("POST", "/api/v1/debates/hybrid", json=data)

    async def get(self, debate_id: str) -> dict[str, Any]:
        """Get a hybrid debate result."""
        return await self._client.request("GET", f"/api/v1/debates/hybrid/{debate_id}")

    async def list(
        self,
        status: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """List hybrid debates."""
        params: dict[str, Any] = {}
        if status:
            params["status"] = status
        if limit != 20:
            params["limit"] = limit

        return await self._client.request("GET", "/api/v1/debates/hybrid", params=params)
