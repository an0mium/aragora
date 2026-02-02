"""
A2A (Agent-to-Agent) Namespace API

Provides methods for the Agent-to-Agent protocol:
- Agent discovery and metadata
- Task submission and execution
- Streaming task responses
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class A2AAPI:
    """
    Synchronous A2A (Agent-to-Agent) API.

    Implements the Agent-to-Agent protocol for inter-agent communication
    and task delegation.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> agents = client.a2a.list_agents()
        >>> task = client.a2a.submit_task(agent="claude", task="Analyze this document")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Agent Discovery
    # ===========================================================================

    def get_agent_card(self) -> dict[str, Any]:
        """
        Get the agent card (well-known metadata).

        Returns:
            Agent card with capabilities, name, description, and supported tasks
        """
        return self._client.request("GET", "/api/v1/a2a/.well-known/agent.json")

    def list_agents(self) -> dict[str, Any]:
        """
        List available A2A agents.

        Returns:
            Dict with agents array
        """
        return self._client.request("GET", "/api/v1/a2a/agents")

    def get_agent(self, name: str) -> dict[str, Any]:
        """
        Get details for a specific A2A agent.

        Args:
            name: Agent name

        Returns:
            Agent details with capabilities
        """
        return self._client.request("GET", f"/api/v1/a2a/agents/{name}")

    def get_openapi_spec(self) -> dict[str, Any]:
        """
        Get the A2A OpenAPI specification.

        Returns:
            OpenAPI spec for the A2A protocol
        """
        return self._client.request("GET", "/api/v1/a2a/openapi.json")

    # ===========================================================================
    # Task Management
    # ===========================================================================

    def submit_task(
        self,
        task: str,
        agent: str | None = None,
        context: dict[str, Any] | None = None,
        timeout: int | None = None,
        priority: str | None = None,
    ) -> dict[str, Any]:
        """
        Submit a task to an A2A agent.

        Args:
            task: Task description or prompt
            agent: Target agent name (uses best available if not specified)
            context: Additional context for the task
            timeout: Task timeout in seconds
            priority: Task priority (low, normal, high)

        Returns:
            Task submission result with task_id
        """
        data: dict[str, Any] = {"task": task}
        if agent:
            data["agent"] = agent
        if context:
            data["context"] = context
        if timeout:
            data["timeout"] = timeout
        if priority:
            data["priority"] = priority

        return self._client.request("POST", "/api/v1/a2a/tasks", json=data)

    def get_task(self, task_id: str) -> dict[str, Any]:
        """
        Get the status and result of a task.

        Args:
            task_id: Task ID

        Returns:
            Task details with status and result
        """
        return self._client.request("GET", f"/api/v1/a2a/tasks/{task_id}")

    def stream_task(
        self,
        task_id: str,
        from_sequence: int | None = None,
    ) -> dict[str, Any]:
        """
        Stream task output (for long-running tasks).

        Args:
            task_id: Task ID
            from_sequence: Start from this sequence number

        Returns:
            Streaming endpoint information
        """
        data: dict[str, Any] = {}
        if from_sequence is not None:
            data["from_sequence"] = from_sequence

        return self._client.request("POST", f"/api/v1/a2a/tasks/{task_id}/stream", json=data)


class AsyncA2AAPI:
    """
    Asynchronous A2A (Agent-to-Agent) API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     agents = await client.a2a.list_agents()
        ...     task = await client.a2a.submit_task(agent="claude", task="Analyze this")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # Agent Discovery
    # ===========================================================================

    async def get_agent_card(self) -> dict[str, Any]:
        """Get the agent card (well-known metadata)."""
        return await self._client.request("GET", "/api/v1/a2a/.well-known/agent.json")

    async def list_agents(self) -> dict[str, Any]:
        """List available A2A agents."""
        return await self._client.request("GET", "/api/v1/a2a/agents")

    async def get_agent(self, name: str) -> dict[str, Any]:
        """Get details for a specific A2A agent."""
        return await self._client.request("GET", f"/api/v1/a2a/agents/{name}")

    async def get_openapi_spec(self) -> dict[str, Any]:
        """Get the A2A OpenAPI specification."""
        return await self._client.request("GET", "/api/v1/a2a/openapi.json")

    # ===========================================================================
    # Task Management
    # ===========================================================================

    async def submit_task(
        self,
        task: str,
        agent: str | None = None,
        context: dict[str, Any] | None = None,
        timeout: int | None = None,
        priority: str | None = None,
    ) -> dict[str, Any]:
        """Submit a task to an A2A agent."""
        data: dict[str, Any] = {"task": task}
        if agent:
            data["agent"] = agent
        if context:
            data["context"] = context
        if timeout:
            data["timeout"] = timeout
        if priority:
            data["priority"] = priority

        return await self._client.request("POST", "/api/v1/a2a/tasks", json=data)

    async def get_task(self, task_id: str) -> dict[str, Any]:
        """Get the status and result of a task."""
        return await self._client.request("GET", f"/api/v1/a2a/tasks/{task_id}")

    async def stream_task(
        self,
        task_id: str,
        from_sequence: int | None = None,
    ) -> dict[str, Any]:
        """Stream task output (for long-running tasks)."""
        data: dict[str, Any] = {}
        if from_sequence is not None:
            data["from_sequence"] = from_sequence

        return await self._client.request("POST", f"/api/v1/a2a/tasks/{task_id}/stream", json=data)
