"""Control plane API for the Aragora SDK."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from aragora_client.client import AragoraClient


# =============================================================================
# Types
# =============================================================================


class AgentStatus(str):
    """Agent status values."""

    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    DRAINING = "draining"
    OFFLINE = "offline"
    FAILED = "failed"


class TaskStatus(str):
    """Task status values."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(str):
    """Task priority values."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class RegisteredAgent(BaseModel):
    """A registered agent in the control plane."""

    agent_id: str
    capabilities: list[str] = Field(default_factory=list)
    status: str = "ready"
    model: str = "unknown"
    provider: str = "unknown"
    is_available: bool = True
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_latency_ms: float = 0.0
    region_id: str = "default"


class AgentHealth(BaseModel):
    """Health status for an agent."""

    agent_id: str
    status: str
    is_available: bool
    last_heartbeat: float
    heartbeat_age_seconds: float
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_latency_ms: float = 0.0
    current_task_id: str | None = None
    health_check: dict[str, Any] | None = None


class Task(BaseModel):
    """A task in the control plane."""

    task_id: str
    task_type: str
    status: str
    priority: str = "normal"
    created_at: float
    assigned_at: float | None = None
    started_at: float | None = None
    completed_at: float | None = None
    assigned_agent: str | None = None
    retries: int = 0
    max_retries: int = 3
    timeout_seconds: float = 300.0
    result: dict[str, Any] | None = None
    error: str | None = None
    is_timed_out: bool = False


class ControlPlaneStatus(BaseModel):
    """Overall control plane status."""

    status: str
    registry: dict[str, Any] = Field(default_factory=dict)
    scheduler: dict[str, Any] = Field(default_factory=dict)
    health_monitor: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    knowledge_mound: dict[str, Any] | None = None


class ResourceUtilization(BaseModel):
    """Resource utilization metrics."""

    queue_depths: dict[str, int] = Field(default_factory=dict)
    agents: dict[str, Any] = Field(default_factory=dict)
    tasks_by_type: dict[str, int] = Field(default_factory=dict)
    tasks_by_priority: dict[str, int] = Field(default_factory=dict)


# =============================================================================
# API Client
# =============================================================================


class ControlPlaneAPI:
    """
    API for control plane operations.

    Provides agent management, task submission, and health monitoring
    for the Aragora control plane.

    Example:
        >>> client = AragoraClient("http://localhost:8080")
        >>> # Register an agent
        >>> agent = await client.control_plane.register_agent(
        ...     agent_id="my-agent",
        ...     capabilities=["debate", "code"],
        ...     model="claude-3-opus",
        ... )
        >>> # Submit a task
        >>> task_id = await client.control_plane.submit_task(
        ...     task_type="debate",
        ...     payload={"question": "Should we use microservices?"},
        ... )
        >>> # Wait for completion
        >>> task = await client.control_plane.wait_for_task(task_id)
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Agent Operations
    # =========================================================================

    async def register_agent(
        self,
        agent_id: str,
        capabilities: list[str] | None = None,
        *,
        model: str = "unknown",
        provider: str = "unknown",
    ) -> RegisteredAgent:
        """
        Register an agent with the control plane.

        Args:
            agent_id: Unique identifier for the agent
            capabilities: List of capabilities (e.g., ["debate", "code"])
            model: Model name (e.g., "claude-3-opus")
            provider: Provider name (e.g., "anthropic")

        Returns:
            RegisteredAgent with registration details
        """
        data = await self._client._post(
            "/api/v1/control-plane/agents",
            {
                "agent_id": agent_id,
                "capabilities": capabilities or ["debate"],
                "model": model,
                "provider": provider,
            },
        )
        return RegisteredAgent.model_validate(data.get("agent", data))

    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the control plane.

        Args:
            agent_id: Agent to unregister

        Returns:
            True if unregistered successfully
        """
        data = await self._client._post(
            f"/api/v1/control-plane/agents/{agent_id}/unregister",
            {},
        )
        return data.get("success", False)

    async def list_agents(
        self,
        *,
        capability: str | None = None,
        only_available: bool = True,
    ) -> list[RegisteredAgent]:
        """
        List all registered agents.

        Args:
            capability: Optional capability filter
            only_available: Only return available agents

        Returns:
            List of registered agents
        """
        params: dict[str, Any] = {"only_available": only_available}
        if capability:
            params["capability"] = capability
        data = await self._client._get("/api/v1/control-plane/agents", params=params)
        return [RegisteredAgent.model_validate(a) for a in data.get("agents", [])]

    async def get_agent_health(self, agent_id: str) -> AgentHealth:
        """
        Get detailed health status for an agent.

        Args:
            agent_id: Agent to query

        Returns:
            AgentHealth with detailed status
        """
        data = await self._client._get(
            f"/api/v1/control-plane/agents/{agent_id}/health"
        )
        return AgentHealth.model_validate(data)

    # =========================================================================
    # Task Operations
    # =========================================================================

    async def submit_task(
        self,
        task_type: str,
        payload: dict[str, Any],
        *,
        required_capabilities: list[str] | None = None,
        priority: str = "normal",
        timeout_seconds: float = 300.0,
    ) -> str:
        """
        Submit a task to the control plane.

        Args:
            task_type: Type of task (e.g., "debate", "code_review")
            payload: Task data
            required_capabilities: Required agent capabilities
            priority: Task priority (low, normal, high, urgent)
            timeout_seconds: Task timeout

        Returns:
            Task ID
        """
        data = await self._client._post(
            "/api/v1/control-plane/tasks",
            {
                "task_type": task_type,
                "payload": payload,
                "required_capabilities": required_capabilities,
                "priority": priority,
                "timeout_seconds": timeout_seconds,
            },
        )
        return data["task_id"]

    async def get_task_status(self, task_id: str) -> Task:
        """
        Get task status.

        Args:
            task_id: Task to query

        Returns:
            Task with current status
        """
        data = await self._client._get(f"/api/v1/control-plane/tasks/{task_id}")
        return Task.model_validate(data)

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending or running task.

        Args:
            task_id: Task to cancel

        Returns:
            True if cancelled successfully
        """
        data = await self._client._post(
            f"/api/v1/control-plane/tasks/{task_id}/cancel",
            {},
        )
        return data.get("success", False)

    async def list_pending_tasks(
        self,
        *,
        task_type: str | None = None,
        limit: int = 20,
    ) -> list[Task]:
        """
        List tasks in the pending queue.

        Args:
            task_type: Optional filter by task type
            limit: Maximum tasks to return

        Returns:
            List of pending tasks
        """
        params: dict[str, Any] = {"limit": limit, "status": "pending"}
        if task_type:
            params["task_type"] = task_type
        data = await self._client._get("/api/v1/control-plane/tasks", params=params)
        return [Task.model_validate(t) for t in data.get("tasks", [])]

    async def wait_for_task(
        self,
        task_id: str,
        *,
        poll_interval: float = 1.0,
        timeout: float = 300.0,
    ) -> Task:
        """
        Wait for a task to complete.

        Args:
            task_id: Task to wait for
            poll_interval: Polling interval in seconds
            timeout: Maximum wait time in seconds

        Returns:
            Completed task

        Raises:
            AragoraTimeoutError: If task doesn't complete within timeout
        """
        from aragora_client.exceptions import AragoraTimeoutError

        elapsed = 0.0
        while elapsed < timeout:
            task = await self.get_task_status(task_id)
            if task.status in ("completed", "failed", "cancelled", "timeout"):
                return task
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise AragoraTimeoutError(f"Task {task_id} did not complete within {timeout}s")

    # =========================================================================
    # Health & Status
    # =========================================================================

    async def get_status(self) -> ControlPlaneStatus:
        """
        Get overall control plane status.

        Returns:
            ControlPlaneStatus with system health
        """
        data = await self._client._get("/api/v1/control-plane/status")
        return ControlPlaneStatus.model_validate(data)

    async def trigger_health_check(self, agent_id: str | None = None) -> dict[str, Any]:
        """
        Trigger a health check.

        Args:
            agent_id: Specific agent to check, or None for all agents

        Returns:
            Health check results
        """
        if agent_id:
            return await self._client._post(
                f"/api/v1/control-plane/agents/{agent_id}/health-check",
                {},
            )
        else:
            return await self._client._post(
                "/api/v1/control-plane/health-check",
                {},
            )

    async def get_resource_utilization(self) -> ResourceUtilization:
        """
        Get resource utilization metrics.

        Returns:
            ResourceUtilization with queue depths and agent stats
        """
        data = await self._client._get("/api/v1/control-plane/utilization")
        return ResourceUtilization.model_validate(data)
