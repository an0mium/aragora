"""AgentsAPI resource for the Aragora client."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from ..models import AgentProfile

if TYPE_CHECKING:
    from ..client import AragoraClient


class AgentsAPI:
    """API interface for agents."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    def list(self) -> list[AgentProfile]:
        """
        List all available agents.

        Returns:
            List of AgentProfile objects.
        """
        response = self._client._get("/api/agents")
        agents = response.get("agents", response) if isinstance(response, dict) else response
        return [AgentProfile(**a) for a in agents]

    async def list_async(self) -> List[AgentProfile]:
        """Async version of list()."""
        response = await self._client._get_async("/api/agents")
        agents = response.get("agents", response) if isinstance(response, dict) else response
        return [AgentProfile(**a) for a in agents]

    def get(self, agent_id: str) -> AgentProfile:
        """
        Get agent profile by ID.

        Args:
            agent_id: The agent ID.

        Returns:
            AgentProfile with details.
        """
        response = self._client._get(f"/api/agent/{agent_id}")
        return AgentProfile(**response)

    async def get_async(self, agent_id: str) -> AgentProfile:
        """Async version of get()."""
        response = await self._client._get_async(f"/api/agent/{agent_id}")
        return AgentProfile(**response)
