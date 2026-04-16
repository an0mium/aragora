"""
Evolution Namespace API

Provides access to agent evolution and A/B testing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class EvolutionAPI:
    """Synchronous Evolution API for agent evolution tracking."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_history(self, agent: str) -> dict[str, Any]:
        """Get evolution history for an agent.

        Args:
            agent: Agent identifier.

        Returns:
            Evolution history with versions and metrics.
        """
        return self._client.request("GET", f"/api/v1/evolution/{agent}/history")

    def get_prompt(self, agent: str) -> dict[str, Any]:
        """Get current prompt for an agent.

        Args:
            agent: Agent identifier.

        Returns:
            Agent prompt configuration.
        """
        return self._client.request("GET", f"/api/v1/evolution/{agent}/prompt")

    def list_ab_tests(self) -> dict[str, Any]:
        """List active A/B tests.

        Returns:
            List of A/B tests with configurations.
        """
        return self._client.request("GET", "/api/v1/evolution/ab-tests")

    def get_patterns(self) -> dict[str, Any]:
        """Get evolution patterns.

        Returns:
            Detected evolution patterns.
        """
        return self._client.request("GET", "/api/v1/evolution/patterns")

    def get_summary(self) -> dict[str, Any]:
        """Get evolution summary.

        Returns:
            Summary of evolution metrics.
        """
        return self._client.request("GET", "/api/v1/evolution/summary")

    def get_overview(self) -> dict[str, Any]:
        """Get evolution overview.

        Returns:
            Dict with overall evolution status and metrics.
        """
        return self._client.request("GET", "/api/v1/evolution")

    def get_agent_evolution_timeline(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Get agent evolution dashboard timeline events."""
        return self._client.request(
            "GET",
            "/api/v1/agent-evolution/timeline",
            params={"limit": limit, "offset": offset},
        )

    def get_agent_evolution_elo_trends(self, period: str = "7d") -> dict[str, Any]:
        """Get agent evolution ELO trend data."""
        return self._client.request(
            "GET",
            "/api/v1/agent-evolution/elo-trends",
            params={"period": period},
        )

    def get_agent_evolution_pending(self) -> dict[str, Any]:
        """Get pending agent evolution changes."""
        return self._client.request("GET", "/api/v1/agent-evolution/pending")

    def approve_agent_evolution_change(self, change_id: str) -> dict[str, Any]:
        """Approve a pending agent evolution change."""
        return self._client.request(
            "POST",
            f"/api/v1/agent-evolution/pending/{change_id}/approve",
        )

    def reject_agent_evolution_change(self, change_id: str) -> dict[str, Any]:
        """Reject a pending agent evolution change."""
        return self._client.request(
            "POST",
            f"/api/v1/agent-evolution/pending/{change_id}/reject",
        )


class AsyncEvolutionAPI:
    """Asynchronous Evolution API for agent evolution tracking."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_history(self, agent: str) -> dict[str, Any]:
        """Get evolution history for an agent.

        Args:
            agent: Agent identifier.

        Returns:
            Evolution history with versions and metrics.
        """
        return await self._client.request("GET", f"/api/v1/evolution/{agent}/history")

    async def get_prompt(self, agent: str) -> dict[str, Any]:
        """Get current prompt for an agent.

        Args:
            agent: Agent identifier.

        Returns:
            Agent prompt configuration.
        """
        return await self._client.request("GET", f"/api/v1/evolution/{agent}/prompt")

    async def list_ab_tests(self) -> dict[str, Any]:
        """List active A/B tests.

        Returns:
            List of A/B tests with configurations.
        """
        return await self._client.request("GET", "/api/v1/evolution/ab-tests")

    async def get_patterns(self) -> dict[str, Any]:
        """Get evolution patterns.

        Returns:
            Detected evolution patterns.
        """
        return await self._client.request("GET", "/api/v1/evolution/patterns")

    async def get_summary(self) -> dict[str, Any]:
        """Get evolution summary.

        Returns:
            Summary of evolution metrics.
        """
        return await self._client.request("GET", "/api/v1/evolution/summary")

    async def get_overview(self) -> dict[str, Any]:
        """Get evolution overview."""
        return await self._client.request("GET", "/api/v1/evolution")

    async def get_agent_evolution_timeline(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Get agent evolution dashboard timeline events."""
        return await self._client.request(
            "GET",
            "/api/v1/agent-evolution/timeline",
            params={"limit": limit, "offset": offset},
        )

    async def get_agent_evolution_elo_trends(self, period: str = "7d") -> dict[str, Any]:
        """Get agent evolution ELO trend data."""
        return await self._client.request(
            "GET",
            "/api/v1/agent-evolution/elo-trends",
            params={"period": period},
        )

    async def get_agent_evolution_pending(self) -> dict[str, Any]:
        """Get pending agent evolution changes."""
        return await self._client.request("GET", "/api/v1/agent-evolution/pending")

    async def approve_agent_evolution_change(self, change_id: str) -> dict[str, Any]:
        """Approve a pending agent evolution change."""
        return await self._client.request(
            "POST",
            f"/api/v1/agent-evolution/pending/{change_id}/approve",
        )

    async def reject_agent_evolution_change(self, change_id: str) -> dict[str, Any]:
        """Reject a pending agent evolution change."""
        return await self._client.request(
            "POST",
            f"/api/v1/agent-evolution/pending/{change_id}/reject",
        )
