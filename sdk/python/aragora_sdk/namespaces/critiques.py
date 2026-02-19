"""
Critiques Namespace API.

Provides a namespaced interface for critique patterns and agent reputation data.
Enables analysis of critique effectiveness and agent performance tracking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class CritiquesAPI:
    """
    Synchronous Critiques API.

    Provides methods for critique pattern retrieval and agent reputation queries:
    - Critique pattern discovery with effectiveness metrics
    - Archive statistics for historical analysis
    - Agent reputation scores and performance data
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Critique Patterns
    # =========================================================================

    def list_patterns(
        self,
        limit: int = 10,
        min_success: float = 0.5,
    ) -> dict[str, Any]:
        """
        Get high-impact critique patterns.

        Retrieves patterns that have been effective in improving debate outcomes.

        Args:
            limit: Maximum number of patterns to return (1-50, default 10).
            min_success: Minimum success rate filter (0.0-1.0, default 0.5).

        Returns:
            List of critique patterns with:
            - issue_type: Category of issue addressed
            - pattern_text: The critique pattern template
            - success_rate: Historical effectiveness (0.0-1.0)
            - usage_count: Number of times pattern was applied
        """
        params: dict[str, Any] = {
            "limit": limit,
            "min_success": min_success,
        }
        return self._client._request("GET", "/api/v1/critiques/patterns", params=params)

    def get_archive_stats(self) -> dict[str, Any]:
        """
        Get archive statistics for critique data.

        Returns:
            Archive statistics including:
            - archived: Total archived critiques
            - by_type: Breakdown by issue type
            - date_range: Time span of archived data
        """
        return self._client._request("GET", "/api/v1/critiques/archive")

    # =========================================================================
    # Agent Reputation
    # =========================================================================

    def list_reputations(self) -> dict[str, Any]:
        """
        Get reputation data for all agents.

        Returns:
            List of agent reputations with:
            - agent_name: Agent identifier
            - reputation_score: Overall reputation (0.0-1.0)
            - vote_weight: Voting influence multiplier
            - proposal_acceptance_rate: Success rate for proposals
            - critique_value: Quality of critiques provided
            - debates_participated: Total debate count
        """
        return self._client._request("GET", "/api/v1/reputation/all")

    def get_agent_reputation(self, agent_name: str) -> dict[str, Any]:
        """
        Get reputation data for a specific agent.

        Args:
            agent_name: The agent identifier (e.g., "claude", "gpt-4").

        Returns:
            Agent reputation data or null if agent not found.
        """
        return self._client._request("GET", f"/api/v1/agent/{agent_name}/reputation")


class AsyncCritiquesAPI:
    """
    Asynchronous Critiques API.

    Provides async methods for critique pattern retrieval and agent reputation queries.
    """

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    # =========================================================================
    # Critique Patterns
    # =========================================================================

    async def list_patterns(
        self,
        limit: int = 10,
        min_success: float = 0.5,
    ) -> dict[str, Any]:
        """
        Get high-impact critique patterns.

        Args:
            limit: Maximum number of patterns to return (1-50, default 10).
            min_success: Minimum success rate filter (0.0-1.0, default 0.5).

        Returns:
            List of critique patterns with effectiveness metrics.
        """
        params: dict[str, Any] = {
            "limit": limit,
            "min_success": min_success,
        }
        return await self._client._request("GET", "/api/v1/critiques/patterns", params=params)

    async def get_archive_stats(self) -> dict[str, Any]:
        """Get archive statistics for critique data."""
        return await self._client._request("GET", "/api/v1/critiques/archive")

    # =========================================================================
    # Agent Reputation
    # =========================================================================

    async def list_reputations(self) -> dict[str, Any]:
        """Get reputation data for all agents."""
        return await self._client._request("GET", "/api/v1/reputation/all")

    async def get_agent_reputation(self, agent_name: str) -> dict[str, Any]:
        """
        Get reputation data for a specific agent.

        Args:
            agent_name: The agent identifier (e.g., "claude", "gpt-4").

        Returns:
            Agent reputation data or null if agent not found.
        """
        return await self._client._request("GET", f"/api/v1/agent/{agent_name}/reputation")
