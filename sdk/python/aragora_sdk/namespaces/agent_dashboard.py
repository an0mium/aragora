"""
Agent Dashboard Namespace API

Provides methods for monitoring and managing agents via the dashboard:
- List and inspect agents with metrics
- Pause and resume agents
- View dashboard health and queue status
- Stream real-time agent events
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class AgentDashboardAPI:
    """
    Synchronous Agent Dashboard API.

    Provides a monitoring view over agents with metrics, health checks,
    and queue management.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> agents = client.agent_dashboard.list_agents()
        >>> health = client.agent_dashboard.get_health()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # =========================================================================
    # Agents
    # =========================================================================

    def list_agents(self) -> dict[str, Any]:
        """
        List all agents visible in the dashboard.

        Returns:
            Dict with list of agents and their summary metrics.
        """
        return self._client.request("GET", "/api/agent-dashboard/agents")

    def get_agent_metrics(self, agent_id: str) -> dict[str, Any]:
        """
        Get detailed metrics for a specific agent.

        Args:
            agent_id: The agent identifier.

        Returns:
            Dict with response time, throughput, error rate, and utilization.
        """
        return self._client.request("GET", f"/api/agent-dashboard/agents/{agent_id}/metrics")

    def pause_agent(self, agent_id: str) -> dict[str, Any]:
        """
        Pause an agent so it stops accepting new tasks.

        Args:
            agent_id: The agent identifier.

        Returns:
            Dict with updated agent status.
        """
        return self._client.request("POST", f"/api/agent-dashboard/agents/{agent_id}/pause")

    def resume_agent(self, agent_id: str) -> dict[str, Any]:
        """
        Resume a paused agent.

        Args:
            agent_id: The agent identifier.

        Returns:
            Dict with updated agent status.
        """
        return self._client.request("POST", f"/api/agent-dashboard/agents/{agent_id}/resume")

    # =========================================================================
    # Health & Metrics
    # =========================================================================

    def get_health(self) -> dict[str, Any]:
        """
        Get overall agent dashboard health status.

        Returns:
            Dict with system health, active agents, and alert counts.
        """
        return self._client.request("GET", "/api/agent-dashboard/health")

    def get_metrics(self) -> dict[str, Any]:
        """
        Get aggregate dashboard metrics.

        Returns:
            Dict with total agents, tasks processed, average latency, etc.
        """
        return self._client.request("GET", "/api/agent-dashboard/metrics")

    # =========================================================================
    # Queue
    # =========================================================================

    def get_queue(self) -> dict[str, Any]:
        """
        Get the current task queue state.

        Returns:
            Dict with pending tasks, queue depth, and processing stats.
        """
        return self._client.request("GET", "/api/agent-dashboard/queue")

    def prioritize_queue(self, **kwargs: Any) -> dict[str, Any]:
        """
        Reprioritize tasks in the queue.

        Args:
            **kwargs: Prioritization parameters (task_id, priority, etc.)

        Returns:
            Dict with updated queue state.
        """
        return self._client.request("POST", "/api/agent-dashboard/queue/prioritize", json=kwargs)

    # =========================================================================
    # Stream
    # =========================================================================

    def get_stream_info(self) -> dict[str, Any]:
        """
        Get SSE/WebSocket stream connection info for real-time agent events.

        Returns:
            Dict with stream URL and supported event types.
        """
        return self._client.request("GET", "/api/agent-dashboard/stream")


class AsyncAgentDashboardAPI:
    """
    Asynchronous Agent Dashboard API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     agents = await client.agent_dashboard.list_agents()
        ...     health = await client.agent_dashboard.get_health()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # Agents
    async def list_agents(self) -> dict[str, Any]:
        """List all agents visible in the dashboard."""
        return await self._client.request("GET", "/api/agent-dashboard/agents")

    async def get_agent_metrics(self, agent_id: str) -> dict[str, Any]:
        """Get detailed metrics for a specific agent."""
        return await self._client.request("GET", f"/api/agent-dashboard/agents/{agent_id}/metrics")

    async def pause_agent(self, agent_id: str) -> dict[str, Any]:
        """Pause an agent so it stops accepting new tasks."""
        return await self._client.request("POST", f"/api/agent-dashboard/agents/{agent_id}/pause")

    async def resume_agent(self, agent_id: str) -> dict[str, Any]:
        """Resume a paused agent."""
        return await self._client.request("POST", f"/api/agent-dashboard/agents/{agent_id}/resume")

    # Health & Metrics
    async def get_health(self) -> dict[str, Any]:
        """Get overall agent dashboard health status."""
        return await self._client.request("GET", "/api/agent-dashboard/health")

    async def get_metrics(self) -> dict[str, Any]:
        """Get aggregate dashboard metrics."""
        return await self._client.request("GET", "/api/agent-dashboard/metrics")

    # Queue
    async def get_queue(self) -> dict[str, Any]:
        """Get the current task queue state."""
        return await self._client.request("GET", "/api/agent-dashboard/queue")

    async def prioritize_queue(self, **kwargs: Any) -> dict[str, Any]:
        """Reprioritize tasks in the queue."""
        return await self._client.request(
            "POST", "/api/agent-dashboard/queue/prioritize", json=kwargs
        )

    # Stream
    async def get_stream_info(self) -> dict[str, Any]:
        """Get SSE/WebSocket stream connection info for real-time agent events."""
        return await self._client.request("GET", "/api/agent-dashboard/stream")
