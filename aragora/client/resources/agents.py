"""AgentsAPI resource for the Aragora client."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from ..models import (
    AgentCalibration,
    AgentConsistency,
    AgentFlip,
    AgentMoment,
    AgentNetwork,
    AgentPerformance,
    AgentPosition,
    AgentProfile,
    DomainRating,
    HeadToHeadStats,
    OpponentBriefing,
)

if TYPE_CHECKING:
    from ..client import AragoraClient


class AgentsAPI:
    """API interface for agents."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    def list(self) -> List[AgentProfile]:
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

    def get_profile(self, agent_id: str) -> AgentProfile:
        """
        Get full agent profile with rating, rank, wins.

        Args:
            agent_id: The agent ID.

        Returns:
            AgentProfile with full details.
        """
        response = self._client._get(f"/api/v1/agent/{agent_id}/profile")
        return AgentProfile(**response)

    async def get_profile_async(self, agent_id: str) -> AgentProfile:
        """Async version of get_profile()."""
        response = await self._client._get_async(f"/api/v1/agent/{agent_id}/profile")
        return AgentProfile(**response)

    def get_calibration(self, agent_id: str) -> AgentCalibration:
        """
        Get calibration scores by domain.

        Args:
            agent_id: The agent ID.

        Returns:
            AgentCalibration with domain scores.
        """
        response = self._client._get(f"/api/v1/agent/{agent_id}/calibration")
        return AgentCalibration(**response)

    async def get_calibration_async(self, agent_id: str) -> AgentCalibration:
        """Async version of get_calibration()."""
        response = await self._client._get_async(f"/api/v1/agent/{agent_id}/calibration")
        return AgentCalibration(**response)

    def get_performance(self, agent_id: str) -> AgentPerformance:
        """
        Get win rates, ELO trend.

        Args:
            agent_id: The agent ID.

        Returns:
            AgentPerformance with win/loss rates and trends.
        """
        response = self._client._get(f"/api/v1/agent/{agent_id}/performance")
        return AgentPerformance(**response)

    async def get_performance_async(self, agent_id: str) -> AgentPerformance:
        """Async version of get_performance()."""
        response = await self._client._get_async(f"/api/v1/agent/{agent_id}/performance")
        return AgentPerformance(**response)

    def get_head_to_head(self, agent_id: str, opponent_id: str) -> HeadToHeadStats:
        """
        Get head-to-head statistics against an opponent.

        Args:
            agent_id: The agent ID.
            opponent_id: The opponent agent ID.

        Returns:
            HeadToHeadStats with matchup history.
        """
        response = self._client._get(f"/api/v1/agent/{agent_id}/head-to-head/{opponent_id}")
        return HeadToHeadStats(**response)

    async def get_head_to_head_async(self, agent_id: str, opponent_id: str) -> HeadToHeadStats:
        """Async version of get_head_to_head()."""
        response = await self._client._get_async(
            f"/api/v1/agent/{agent_id}/head-to-head/{opponent_id}"
        )
        return HeadToHeadStats(**response)

    def get_opponent_briefing(self, agent_id: str, opponent_id: str) -> OpponentBriefing:
        """
        Get strategic briefing against an opponent.

        Args:
            agent_id: The agent ID.
            opponent_id: The opponent agent ID.

        Returns:
            OpponentBriefing with strategy recommendations.
        """
        response = self._client._get(f"/api/v1/agent/{agent_id}/opponent-briefing/{opponent_id}")
        return OpponentBriefing(**response)

    async def get_opponent_briefing_async(
        self, agent_id: str, opponent_id: str
    ) -> OpponentBriefing:
        """Async version of get_opponent_briefing()."""
        response = await self._client._get_async(
            f"/api/v1/agent/{agent_id}/opponent-briefing/{opponent_id}"
        )
        return OpponentBriefing(**response)

    def get_consistency(self, agent_id: str) -> AgentConsistency:
        """
        Get position consistency score.

        Args:
            agent_id: The agent ID.

        Returns:
            AgentConsistency with stability metrics.
        """
        response = self._client._get(f"/api/v1/agent/{agent_id}/consistency")
        return AgentConsistency(**response)

    async def get_consistency_async(self, agent_id: str) -> AgentConsistency:
        """Async version of get_consistency()."""
        response = await self._client._get_async(f"/api/v1/agent/{agent_id}/consistency")
        return AgentConsistency(**response)

    def get_flips(self, agent_id: str, limit: int = 20, offset: int = 0) -> List[AgentFlip]:
        """
        Get position flip history.

        Args:
            agent_id: The agent ID.
            limit: Maximum number of flips to return.
            offset: Number of flips to skip.

        Returns:
            List of AgentFlip events.
        """
        response = self._client._get(
            f"/api/v1/agent/{agent_id}/flips", params={"limit": limit, "offset": offset}
        )
        flips = response.get("flips", response) if isinstance(response, dict) else response
        return [AgentFlip(**f) for f in flips]

    async def get_flips_async(
        self, agent_id: str, limit: int = 20, offset: int = 0
    ) -> List[AgentFlip]:
        """Async version of get_flips()."""
        response = await self._client._get_async(
            f"/api/v1/agent/{agent_id}/flips", params={"limit": limit, "offset": offset}
        )
        flips = response.get("flips", response) if isinstance(response, dict) else response
        return [AgentFlip(**f) for f in flips]

    def get_network(self, agent_id: str) -> AgentNetwork:
        """
        Get rivals and allies network.

        Args:
            agent_id: The agent ID.

        Returns:
            AgentNetwork with relationship data.
        """
        response = self._client._get(f"/api/v1/agent/{agent_id}/network")
        return AgentNetwork(**response)

    async def get_network_async(self, agent_id: str) -> AgentNetwork:
        """Async version of get_network()."""
        response = await self._client._get_async(f"/api/v1/agent/{agent_id}/network")
        return AgentNetwork(**response)

    def get_moments(
        self, agent_id: str, moment_type: Optional[str] = None, limit: int = 20, offset: int = 0
    ) -> List[AgentMoment]:
        """
        Get significant moments.

        Args:
            agent_id: The agent ID.
            moment_type: Optional filter by type (breakthrough, comeback, etc).
            limit: Maximum number of moments to return.
            offset: Number of moments to skip.

        Returns:
            List of AgentMoment events.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if moment_type:
            params["type"] = moment_type
        response = self._client._get(f"/api/v1/agent/{agent_id}/moments", params=params)
        moments = response.get("moments", response) if isinstance(response, dict) else response
        return [AgentMoment(**m) for m in moments]

    async def get_moments_async(
        self, agent_id: str, moment_type: Optional[str] = None, limit: int = 20, offset: int = 0
    ) -> List[AgentMoment]:
        """Async version of get_moments()."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if moment_type:
            params["type"] = moment_type
        response = await self._client._get_async(f"/api/v1/agent/{agent_id}/moments", params=params)
        moments = response.get("moments", response) if isinstance(response, dict) else response
        return [AgentMoment(**m) for m in moments]

    def get_positions(
        self, agent_id: str, topic: Optional[str] = None, limit: int = 20, offset: int = 0
    ) -> List[AgentPosition]:
        """
        Get position history.

        Args:
            agent_id: The agent ID.
            topic: Optional filter by topic.
            limit: Maximum number of positions to return.
            offset: Number of positions to skip.

        Returns:
            List of AgentPosition records.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if topic:
            params["topic"] = topic
        response = self._client._get(f"/api/v1/agent/{agent_id}/positions", params=params)
        positions = response.get("positions", response) if isinstance(response, dict) else response
        return [AgentPosition(**p) for p in positions]

    async def get_positions_async(
        self, agent_id: str, topic: Optional[str] = None, limit: int = 20, offset: int = 0
    ) -> List[AgentPosition]:
        """Async version of get_positions()."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if topic:
            params["topic"] = topic
        response = await self._client._get_async(
            f"/api/v1/agent/{agent_id}/positions", params=params
        )
        positions = response.get("positions", response) if isinstance(response, dict) else response
        return [AgentPosition(**p) for p in positions]

    def get_domains(self, agent_id: str) -> List[DomainRating]:
        """
        Get domain-specific ELO ratings.

        Args:
            agent_id: The agent ID.

        Returns:
            List of DomainRating for each domain.
        """
        response = self._client._get(f"/api/v1/agent/{agent_id}/domains")
        domains = response.get("domains", response) if isinstance(response, dict) else response
        return [DomainRating(**d) for d in domains]

    async def get_domains_async(self, agent_id: str) -> List[DomainRating]:
        """Async version of get_domains()."""
        response = await self._client._get_async(f"/api/v1/agent/{agent_id}/domains")
        domains = response.get("domains", response) if isinstance(response, dict) else response
        return [DomainRating(**d) for d in domains]
