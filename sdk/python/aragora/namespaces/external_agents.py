"""
External Agents Namespace API

Provides methods for external agent operations:
- Adapter discovery and health monitoring
- Task submission, tracking, and cancellation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class ExternalAgentsAPI:
    """
    Synchronous External Agents API.

    Provides methods for managing external agent adapters and tasks:
    - List available adapters
    - Check adapter health
    - Submit, track, and cancel tasks

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> adapters = client.external_agents.list_adapters()
        >>> task = client.external_agents.submit_task(
        ...     task_type="code_review",
        ...     prompt="Review this pull request for security issues",
        ... )
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Adapter Discovery
    # ===========================================================================

    def list_adapters(self) -> dict[str, Any]:
        """
        List available external agent adapters.

        Returns:
            Dict with adapters array and their configurations
        """
        return self._client.request("GET", "/api/v1/external-agents/adapters")

    def get_adapter_health(
        self,
        adapter_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Get health status for external agent adapters.

        Args:
            adapter_name: Filter by specific adapter name (optional)

        Returns:
            Dict with adapter health status information
        """
        params: dict[str, Any] = {}
        if adapter_name:
            params["adapter"] = adapter_name
        return self._client.request("GET", "/api/v1/external-agents/health", params=params)

    # ===========================================================================
    # Task Management
    # ===========================================================================

    def submit_task(
        self,
        task_type: str,
        prompt: str,
        adapter: str = "openhands",
        tool_permissions: list[str] | None = None,
        timeout_seconds: float = 3600.0,
        max_steps: int = 100,
        context: dict[str, Any] | None = None,
        workspace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Submit a task to an external agent.

        Args:
            task_type: Type of task to execute
            prompt: The prompt or instructions for the agent
            adapter: Adapter to use (default: "openhands")
            tool_permissions: List of tool permissions granted to the agent
            timeout_seconds: Maximum execution time in seconds (default: 3600.0)
            max_steps: Maximum number of steps the agent can take (default: 100)
            context: Additional context for the task
            workspace_id: Workspace to execute the task in
            metadata: Additional metadata

        Returns:
            Dict with task_id and submission status
        """
        data: dict[str, Any] = {
            "task_type": task_type,
            "prompt": prompt,
            "adapter": adapter,
            "timeout_seconds": timeout_seconds,
            "max_steps": max_steps,
            "metadata": metadata or {},
        }
        if tool_permissions is not None:
            data["tool_permissions"] = tool_permissions
        if context is not None:
            data["context"] = context
        if workspace_id is not None:
            data["workspace_id"] = workspace_id
        return self._client.request("POST", "/api/v1/external-agents/tasks", json=data)

    def get_task(self, task_id: str) -> dict[str, Any]:
        """
        Get details for a specific external agent task.

        Args:
            task_id: Task ID

        Returns:
            Dict with task details including status, result, and metadata
        """
        return self._client.request("GET", f"/api/v1/external-agents/tasks/{task_id}")

    def cancel_task(self, task_id: str) -> dict[str, Any]:
        """
        Cancel a running external agent task.

        Args:
            task_id: Task ID

        Returns:
            Dict with cancellation status
        """
        return self._client.request("DELETE", f"/api/v1/external-agents/tasks/{task_id}")


class AsyncExternalAgentsAPI:
    """
    Asynchronous External Agents API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     adapters = await client.external_agents.list_adapters()
        ...     task = await client.external_agents.submit_task(
        ...         task_type="code_review",
        ...         prompt="Review this pull request for security issues",
        ...     )
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # Adapter Discovery
    # ===========================================================================

    async def list_adapters(self) -> dict[str, Any]:
        """List available external agent adapters."""
        return await self._client.request("GET", "/api/v1/external-agents/adapters")

    async def get_adapter_health(
        self,
        adapter_name: str | None = None,
    ) -> dict[str, Any]:
        """Get health status for external agent adapters."""
        params: dict[str, Any] = {}
        if adapter_name:
            params["adapter"] = adapter_name
        return await self._client.request("GET", "/api/v1/external-agents/health", params=params)

    # ===========================================================================
    # Task Management
    # ===========================================================================

    async def submit_task(
        self,
        task_type: str,
        prompt: str,
        adapter: str = "openhands",
        tool_permissions: list[str] | None = None,
        timeout_seconds: float = 3600.0,
        max_steps: int = 100,
        context: dict[str, Any] | None = None,
        workspace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a task to an external agent."""
        data: dict[str, Any] = {
            "task_type": task_type,
            "prompt": prompt,
            "adapter": adapter,
            "timeout_seconds": timeout_seconds,
            "max_steps": max_steps,
            "metadata": metadata or {},
        }
        if tool_permissions is not None:
            data["tool_permissions"] = tool_permissions
        if context is not None:
            data["context"] = context
        if workspace_id is not None:
            data["workspace_id"] = workspace_id
        return await self._client.request("POST", "/api/v1/external-agents/tasks", json=data)

    async def get_task(self, task_id: str) -> dict[str, Any]:
        """Get external agent task details."""
        return await self._client.request("GET", f"/api/v1/external-agents/tasks/{task_id}")

    async def cancel_task(self, task_id: str) -> dict[str, Any]:
        """Cancel a running external agent task."""
        return await self._client.request("DELETE", f"/api/v1/external-agents/tasks/{task_id}")
