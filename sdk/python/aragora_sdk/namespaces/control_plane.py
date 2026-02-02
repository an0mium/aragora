"""
Control Plane Namespace API

Provides methods for enterprise control plane operations:
- Agent registration and discovery
- Task submission and scheduling
- Deliberation requests (vetted decisionmaking)
- Health monitoring and circuit breakers
- Audit logs and policy violation management
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class ControlPlaneAPI:
    """
    Synchronous Control Plane API.

    Provides methods for enterprise orchestration:
    - Register and manage agents
    - Submit and track tasks
    - Run deliberations (vetted decisionmaking)
    - Monitor system health
    - Query audit logs
    - Manage policy violations

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> health = client.control_plane.get_system_health()
        >>> agents = client.control_plane.list_agents()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Agent Management
    # ===========================================================================

    def list_agents(
        self,
        capability: str | None = None,
        available: bool = True,
    ) -> dict[str, Any]:
        """
        List registered agents.

        Args:
            capability: Filter by capability
            available: Only return available agents (default: True)

        Returns:
            Dict with agents array and total count
        """
        params: dict[str, Any] = {"available": str(available).lower()}
        if capability:
            params["capability"] = capability
        return self._client.request("GET", "/api/v1/control-plane/agents", params=params)

    def register_agent(
        self,
        agent_id: str,
        capabilities: list[str] | None = None,
        model: str = "unknown",
        provider: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Register a new agent with the control plane.

        Args:
            agent_id: Unique agent identifier
            capabilities: List of agent capabilities
            model: Model name
            provider: Provider name
            metadata: Additional metadata

        Returns:
            Registered agent info
        """
        data: dict[str, Any] = {
            "agent_id": agent_id,
            "capabilities": capabilities or [],
            "model": model,
            "provider": provider,
            "metadata": metadata or {},
        }
        return self._client.request("POST", "/api/v1/control-plane/agents", json=data)

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        """
        Get agent details.

        Args:
            agent_id: Agent ID

        Returns:
            Agent details
        """
        return self._client.request("GET", f"/api/v1/control-plane/agents/{agent_id}")

    def unregister_agent(self, agent_id: str) -> dict[str, Any]:
        """
        Unregister an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Dict with unregistered status
        """
        return self._client.request("DELETE", f"/api/v1/control-plane/agents/{agent_id}")

    def heartbeat(
        self,
        agent_id: str,
        status: str | None = None,
    ) -> dict[str, Any]:
        """
        Send agent heartbeat.

        Args:
            agent_id: Agent ID
            status: Optional status update

        Returns:
            Dict with acknowledged status
        """
        data: dict[str, Any] = {}
        if status:
            data["status"] = status
        return self._client.request(
            "POST", f"/api/v1/control-plane/agents/{agent_id}/heartbeat", json=data
        )

    # ===========================================================================
    # Task Management
    # ===========================================================================

    def submit_task(
        self,
        task_type: str,
        payload: dict[str, Any] | None = None,
        required_capabilities: list[str] | None = None,
        priority: str = "normal",
        timeout_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Submit a new task.

        Args:
            task_type: Type of task
            payload: Task payload
            required_capabilities: Required agent capabilities
            priority: Task priority (low, normal, high, critical)
            timeout_seconds: Task timeout
            metadata: Additional metadata

        Returns:
            Dict with task_id
        """
        data: dict[str, Any] = {
            "task_type": task_type,
            "payload": payload or {},
            "required_capabilities": required_capabilities or [],
            "priority": priority,
            "metadata": metadata or {},
        }
        if timeout_seconds is not None:
            data["timeout_seconds"] = timeout_seconds
        return self._client.request("POST", "/api/v1/control-plane/tasks", json=data)

    def get_task(self, task_id: str) -> dict[str, Any]:
        """
        Get task details.

        Args:
            task_id: Task ID

        Returns:
            Task details
        """
        return self._client.request("GET", f"/api/v1/control-plane/tasks/{task_id}")

    def claim_task(
        self,
        agent_id: str,
        capabilities: list[str] | None = None,
        block_ms: int = 5000,
    ) -> dict[str, Any]:
        """
        Claim a task for an agent.

        Args:
            agent_id: Agent ID claiming the task
            capabilities: Agent capabilities
            block_ms: Time to block waiting for task (default: 5000)

        Returns:
            Dict with task or None
        """
        data: dict[str, Any] = {
            "agent_id": agent_id,
            "capabilities": capabilities or [],
            "block_ms": block_ms,
        }
        return self._client.request("POST", "/api/v1/control-plane/tasks/claim", json=data)

    def complete_task(
        self,
        task_id: str,
        result: Any = None,
        agent_id: str | None = None,
        latency_ms: int | None = None,
    ) -> dict[str, Any]:
        """
        Mark task as completed.

        Args:
            task_id: Task ID
            result: Task result
            agent_id: Agent that completed the task
            latency_ms: Execution latency in milliseconds

        Returns:
            Dict with completed status
        """
        data: dict[str, Any] = {}
        if result is not None:
            data["result"] = result
        if agent_id:
            data["agent_id"] = agent_id
        if latency_ms is not None:
            data["latency_ms"] = latency_ms
        return self._client.request(
            "POST", f"/api/v1/control-plane/tasks/{task_id}/complete", json=data
        )

    def fail_task(
        self,
        task_id: str,
        error: str = "Unknown error",
        agent_id: str | None = None,
        latency_ms: int | None = None,
        requeue: bool = True,
    ) -> dict[str, Any]:
        """
        Mark task as failed.

        Args:
            task_id: Task ID
            error: Error message
            agent_id: Agent that failed
            latency_ms: Execution latency
            requeue: Whether to requeue the task (default: True)

        Returns:
            Dict with failed status
        """
        data: dict[str, Any] = {"error": error, "requeue": requeue}
        if agent_id:
            data["agent_id"] = agent_id
        if latency_ms is not None:
            data["latency_ms"] = latency_ms
        return self._client.request(
            "POST", f"/api/v1/control-plane/tasks/{task_id}/fail", json=data
        )

    def cancel_task(self, task_id: str) -> dict[str, Any]:
        """
        Cancel a task.

        Args:
            task_id: Task ID

        Returns:
            Dict with cancelled status
        """
        return self._client.request("POST", f"/api/v1/control-plane/tasks/{task_id}/cancel")

    # ===========================================================================
    # Deliberations
    # ===========================================================================

    def submit_deliberation(
        self,
        content: str,
        async_mode: bool = False,
        priority: str = "normal",
        required_capabilities: list[str] | None = None,
        timeout_seconds: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Submit a deliberation request (vetted decisionmaking).

        Args:
            content: The question or topic to deliberate
            async_mode: Run asynchronously (default: False)
            priority: Priority level (low, normal, high, critical)
            required_capabilities: Required agent capabilities
            timeout_seconds: Timeout for deliberation
            context: Additional context

        Returns:
            For sync: Full deliberation result
            For async: Dict with task_id and request_id
        """
        data: dict[str, Any] = {
            "content": content,
            "async": async_mode,
            "priority": priority,
            "required_capabilities": required_capabilities or ["deliberation"],
        }
        if timeout_seconds is not None:
            data["timeout_seconds"] = timeout_seconds
        if context:
            data["context"] = context
        return self._client.request("POST", "/api/v1/control-plane/deliberations", json=data)

    def get_deliberation(self, request_id: str) -> dict[str, Any]:
        """
        Get deliberation result.

        Args:
            request_id: Request ID

        Returns:
            Deliberation result
        """
        return self._client.request("GET", f"/api/v1/control-plane/deliberations/{request_id}")

    def get_deliberation_status(self, request_id: str) -> dict[str, Any]:
        """
        Get deliberation status for polling.

        Args:
            request_id: Request ID

        Returns:
            Status info
        """
        return self._client.request(
            "GET", f"/api/v1/control-plane/deliberations/{request_id}/status"
        )

    # ===========================================================================
    # Health Monitoring
    # ===========================================================================

    def get_system_health(self) -> dict[str, Any]:
        """
        Get system health status.

        Returns:
            Dict with status and agent health info
        """
        return self._client.request("GET", "/api/v1/control-plane/health")

    def get_detailed_health(self) -> dict[str, Any]:
        """
        Get detailed system health with component status.

        Returns:
            Dict with status, uptime, version, and components
        """
        return self._client.request("GET", "/api/v1/control-plane/health/detailed")

    def get_agent_health(self, agent_id: str) -> dict[str, Any]:
        """
        Get health status for a specific agent.

        Args:
            agent_id: Agent ID

        Returns:
            Agent health details
        """
        return self._client.request("GET", f"/api/v1/control-plane/health/{agent_id}")

    def get_circuit_breakers(self) -> dict[str, Any]:
        """
        Get circuit breaker states.

        Returns:
            Dict with breakers array
        """
        return self._client.request("GET", "/api/v1/control-plane/breakers")

    # ===========================================================================
    # Statistics and Metrics
    # ===========================================================================

    def get_stats(self) -> dict[str, Any]:
        """
        Get control plane statistics.

        Returns:
            Control plane stats
        """
        return self._client.request("GET", "/api/v1/control-plane/stats")

    def get_queue(self, limit: int = 50) -> dict[str, Any]:
        """
        Get current job queue (pending and running tasks).

        Args:
            limit: Maximum jobs to return (default: 50)

        Returns:
            Dict with jobs array and total
        """
        return self._client.request("GET", "/api/v1/control-plane/queue", params={"limit": limit})

    def get_queue_metrics(self) -> dict[str, Any]:
        """
        Get task queue performance metrics.

        Returns:
            Dict with queue metrics (pending, running, throughput, etc.)
        """
        return self._client.request("GET", "/api/v1/control-plane/queue/metrics")

    def get_metrics(self) -> dict[str, Any]:
        """
        Get control plane metrics for dashboard.

        Returns:
            Dict with active_jobs, queued_jobs, agents, etc.
        """
        return self._client.request("GET", "/api/v1/control-plane/metrics")

    # ===========================================================================
    # Notifications
    # ===========================================================================

    def get_notifications(self) -> dict[str, Any]:
        """
        Get recent notification history.

        Returns:
            Dict with notifications and stats
        """
        return self._client.request("GET", "/api/v1/control-plane/notifications")

    def get_notification_stats(self) -> dict[str, Any]:
        """
        Get notification statistics.

        Returns:
            Dict with total_sent, successful, failed, by_channel
        """
        return self._client.request("GET", "/api/v1/control-plane/notifications/stats")

    # ===========================================================================
    # Audit Logs
    # ===========================================================================

    def query_audit_logs(
        self,
        start_time: str | None = None,
        end_time: str | None = None,
        actions: list[str] | None = None,
        actor_types: list[str] | None = None,
        actor_ids: list[str] | None = None,
        resource_types: list[str] | None = None,
        workspace_ids: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Query audit logs with filtering.

        Args:
            start_time: Start time (ISO 8601)
            end_time: End time (ISO 8601)
            actions: Filter by action types
            actor_types: Filter by actor types
            actor_ids: Filter by actor IDs
            resource_types: Filter by resource types
            workspace_ids: Filter by workspace IDs
            limit: Maximum results (default: 100)
            offset: Pagination offset

        Returns:
            Dict with entries array and total
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        if actions:
            params["actions"] = ",".join(actions)
        if actor_types:
            params["actor_types"] = ",".join(actor_types)
        if actor_ids:
            params["actor_ids"] = ",".join(actor_ids)
        if resource_types:
            params["resource_types"] = ",".join(resource_types)
        if workspace_ids:
            params["workspace_ids"] = ",".join(workspace_ids)
        return self._client.request("GET", "/api/v1/control-plane/audit", params=params)

    def get_audit_stats(self) -> dict[str, Any]:
        """
        Get audit log statistics.

        Returns:
            Dict with total_entries and storage_backend
        """
        return self._client.request("GET", "/api/v1/control-plane/audit/stats")

    def verify_audit_integrity(
        self,
        start_seq: int = 0,
        end_seq: int | None = None,
    ) -> dict[str, Any]:
        """
        Verify audit log integrity.

        Args:
            start_seq: Start sequence number (default: 0)
            end_seq: End sequence number (optional)

        Returns:
            Dict with valid status and message
        """
        params: dict[str, Any] = {"start_seq": start_seq}
        if end_seq is not None:
            params["end_seq"] = end_seq
        return self._client.request("GET", "/api/v1/control-plane/audit/verify", params=params)

    # ===========================================================================
    # Policy Violations
    # ===========================================================================

    def list_policy_violations(
        self,
        policy_id: str | None = None,
        violation_type: str | None = None,
        status: str | None = None,
        workspace_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List policy violations.

        Args:
            policy_id: Filter by policy ID
            violation_type: Filter by violation type
            status: Filter by status (open, investigating, resolved, false_positive)
            workspace_id: Filter by workspace
            limit: Maximum results (default: 100)
            offset: Pagination offset

        Returns:
            Dict with violations array and total
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if policy_id:
            params["policy_id"] = policy_id
        if violation_type:
            params["violation_type"] = violation_type
        if status:
            params["status"] = status
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request(
            "GET", "/api/v1/control-plane/policies/violations", params=params
        )

    def get_policy_violation(self, violation_id: str) -> dict[str, Any]:
        """
        Get a specific policy violation.

        Args:
            violation_id: Violation ID

        Returns:
            Dict with violation details
        """
        return self._client.request(
            "GET", f"/api/v1/control-plane/policies/violations/{violation_id}"
        )

    def get_policy_violation_stats(self) -> dict[str, Any]:
        """
        Get policy violation statistics.

        Returns:
            Dict with total, open, resolved, by_type
        """
        return self._client.request("GET", "/api/v1/control-plane/policies/violations/stats")

    def update_policy_violation(
        self,
        violation_id: str,
        status: str,
        resolution_notes: str | None = None,
    ) -> dict[str, Any]:
        """
        Update a policy violation status.

        Args:
            violation_id: Violation ID
            status: New status (open, investigating, resolved, false_positive)
            resolution_notes: Optional resolution notes

        Returns:
            Dict with updated status
        """
        data: dict[str, Any] = {"status": status}
        if resolution_notes:
            data["resolution_notes"] = resolution_notes
        return self._client.request(
            "PATCH", f"/api/v1/control-plane/policies/violations/{violation_id}", json=data
        )


class AsyncControlPlaneAPI:
    """
    Asynchronous Control Plane API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     health = await client.control_plane.get_system_health()
        ...     agents = await client.control_plane.list_agents()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # Agent Management
    # ===========================================================================

    async def list_agents(
        self,
        capability: str | None = None,
        available: bool = True,
    ) -> dict[str, Any]:
        """List registered agents."""
        params: dict[str, Any] = {"available": str(available).lower()}
        if capability:
            params["capability"] = capability
        return await self._client.request("GET", "/api/v1/control-plane/agents", params=params)

    async def register_agent(
        self,
        agent_id: str,
        capabilities: list[str] | None = None,
        model: str = "unknown",
        provider: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register a new agent."""
        data: dict[str, Any] = {
            "agent_id": agent_id,
            "capabilities": capabilities or [],
            "model": model,
            "provider": provider,
            "metadata": metadata or {},
        }
        return await self._client.request("POST", "/api/v1/control-plane/agents", json=data)

    async def get_agent(self, agent_id: str) -> dict[str, Any]:
        """Get agent details."""
        return await self._client.request("GET", f"/api/v1/control-plane/agents/{agent_id}")

    async def unregister_agent(self, agent_id: str) -> dict[str, Any]:
        """Unregister an agent."""
        return await self._client.request("DELETE", f"/api/v1/control-plane/agents/{agent_id}")

    async def heartbeat(self, agent_id: str, status: str | None = None) -> dict[str, Any]:
        """Send agent heartbeat."""
        data: dict[str, Any] = {}
        if status:
            data["status"] = status
        return await self._client.request(
            "POST", f"/api/v1/control-plane/agents/{agent_id}/heartbeat", json=data
        )

    # ===========================================================================
    # Task Management
    # ===========================================================================

    async def submit_task(
        self,
        task_type: str,
        payload: dict[str, Any] | None = None,
        required_capabilities: list[str] | None = None,
        priority: str = "normal",
        timeout_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a new task."""
        data: dict[str, Any] = {
            "task_type": task_type,
            "payload": payload or {},
            "required_capabilities": required_capabilities or [],
            "priority": priority,
            "metadata": metadata or {},
        }
        if timeout_seconds is not None:
            data["timeout_seconds"] = timeout_seconds
        return await self._client.request("POST", "/api/v1/control-plane/tasks", json=data)

    async def get_task(self, task_id: str) -> dict[str, Any]:
        """Get task details."""
        return await self._client.request("GET", f"/api/v1/control-plane/tasks/{task_id}")

    async def claim_task(
        self,
        agent_id: str,
        capabilities: list[str] | None = None,
        block_ms: int = 5000,
    ) -> dict[str, Any]:
        """Claim a task for an agent."""
        data: dict[str, Any] = {
            "agent_id": agent_id,
            "capabilities": capabilities or [],
            "block_ms": block_ms,
        }
        return await self._client.request("POST", "/api/v1/control-plane/tasks/claim", json=data)

    async def complete_task(
        self,
        task_id: str,
        result: Any = None,
        agent_id: str | None = None,
        latency_ms: int | None = None,
    ) -> dict[str, Any]:
        """Mark task as completed."""
        data: dict[str, Any] = {}
        if result is not None:
            data["result"] = result
        if agent_id:
            data["agent_id"] = agent_id
        if latency_ms is not None:
            data["latency_ms"] = latency_ms
        return await self._client.request(
            "POST", f"/api/v1/control-plane/tasks/{task_id}/complete", json=data
        )

    async def fail_task(
        self,
        task_id: str,
        error: str = "Unknown error",
        agent_id: str | None = None,
        latency_ms: int | None = None,
        requeue: bool = True,
    ) -> dict[str, Any]:
        """Mark task as failed."""
        data: dict[str, Any] = {"error": error, "requeue": requeue}
        if agent_id:
            data["agent_id"] = agent_id
        if latency_ms is not None:
            data["latency_ms"] = latency_ms
        return await self._client.request(
            "POST", f"/api/v1/control-plane/tasks/{task_id}/fail", json=data
        )

    async def cancel_task(self, task_id: str) -> dict[str, Any]:
        """Cancel a task."""
        return await self._client.request("POST", f"/api/v1/control-plane/tasks/{task_id}/cancel")

    # ===========================================================================
    # Deliberations
    # ===========================================================================

    async def submit_deliberation(
        self,
        content: str,
        async_mode: bool = False,
        priority: str = "normal",
        required_capabilities: list[str] | None = None,
        timeout_seconds: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a deliberation request."""
        data: dict[str, Any] = {
            "content": content,
            "async": async_mode,
            "priority": priority,
            "required_capabilities": required_capabilities or ["deliberation"],
        }
        if timeout_seconds is not None:
            data["timeout_seconds"] = timeout_seconds
        if context:
            data["context"] = context
        return await self._client.request("POST", "/api/v1/control-plane/deliberations", json=data)

    async def get_deliberation(self, request_id: str) -> dict[str, Any]:
        """Get deliberation result."""
        return await self._client.request(
            "GET", f"/api/v1/control-plane/deliberations/{request_id}"
        )

    async def get_deliberation_status(self, request_id: str) -> dict[str, Any]:
        """Get deliberation status."""
        return await self._client.request(
            "GET", f"/api/v1/control-plane/deliberations/{request_id}/status"
        )

    # ===========================================================================
    # Health Monitoring
    # ===========================================================================

    async def get_system_health(self) -> dict[str, Any]:
        """Get system health status."""
        return await self._client.request("GET", "/api/v1/control-plane/health")

    async def get_detailed_health(self) -> dict[str, Any]:
        """Get detailed system health."""
        return await self._client.request("GET", "/api/v1/control-plane/health/detailed")

    async def get_agent_health(self, agent_id: str) -> dict[str, Any]:
        """Get agent health."""
        return await self._client.request("GET", f"/api/v1/control-plane/health/{agent_id}")

    async def get_circuit_breakers(self) -> dict[str, Any]:
        """Get circuit breaker states."""
        return await self._client.request("GET", "/api/v1/control-plane/breakers")

    # ===========================================================================
    # Statistics and Metrics
    # ===========================================================================

    async def get_stats(self) -> dict[str, Any]:
        """Get control plane statistics."""
        return await self._client.request("GET", "/api/v1/control-plane/stats")

    async def get_queue(self, limit: int = 50) -> dict[str, Any]:
        """Get current job queue."""
        return await self._client.request(
            "GET", "/api/v1/control-plane/queue", params={"limit": limit}
        )

    async def get_queue_metrics(self) -> dict[str, Any]:
        """Get task queue metrics."""
        return await self._client.request("GET", "/api/v1/control-plane/queue/metrics")

    async def get_metrics(self) -> dict[str, Any]:
        """Get dashboard metrics."""
        return await self._client.request("GET", "/api/v1/control-plane/metrics")

    # ===========================================================================
    # Notifications
    # ===========================================================================

    async def get_notifications(self) -> dict[str, Any]:
        """Get notifications."""
        return await self._client.request("GET", "/api/v1/control-plane/notifications")

    async def get_notification_stats(self) -> dict[str, Any]:
        """Get notification stats."""
        return await self._client.request("GET", "/api/v1/control-plane/notifications/stats")

    # ===========================================================================
    # Audit Logs
    # ===========================================================================

    async def query_audit_logs(
        self,
        start_time: str | None = None,
        end_time: str | None = None,
        actions: list[str] | None = None,
        actor_types: list[str] | None = None,
        actor_ids: list[str] | None = None,
        resource_types: list[str] | None = None,
        workspace_ids: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Query audit logs."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        if actions:
            params["actions"] = ",".join(actions)
        if actor_types:
            params["actor_types"] = ",".join(actor_types)
        if actor_ids:
            params["actor_ids"] = ",".join(actor_ids)
        if resource_types:
            params["resource_types"] = ",".join(resource_types)
        if workspace_ids:
            params["workspace_ids"] = ",".join(workspace_ids)
        return await self._client.request("GET", "/api/v1/control-plane/audit", params=params)

    async def get_audit_stats(self) -> dict[str, Any]:
        """Get audit stats."""
        return await self._client.request("GET", "/api/v1/control-plane/audit/stats")

    async def verify_audit_integrity(
        self,
        start_seq: int = 0,
        end_seq: int | None = None,
    ) -> dict[str, Any]:
        """Verify audit integrity."""
        params: dict[str, Any] = {"start_seq": start_seq}
        if end_seq is not None:
            params["end_seq"] = end_seq
        return await self._client.request(
            "GET", "/api/v1/control-plane/audit/verify", params=params
        )

    # ===========================================================================
    # Policy Violations
    # ===========================================================================

    async def list_policy_violations(
        self,
        policy_id: str | None = None,
        violation_type: str | None = None,
        status: str | None = None,
        workspace_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List policy violations."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if policy_id:
            params["policy_id"] = policy_id
        if violation_type:
            params["violation_type"] = violation_type
        if status:
            params["status"] = status
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request(
            "GET", "/api/v1/control-plane/policies/violations", params=params
        )

    async def get_policy_violation(self, violation_id: str) -> dict[str, Any]:
        """Get violation details."""
        return await self._client.request(
            "GET", f"/api/v1/control-plane/policies/violations/{violation_id}"
        )

    async def get_policy_violation_stats(self) -> dict[str, Any]:
        """Get violation stats."""
        return await self._client.request("GET", "/api/v1/control-plane/policies/violations/stats")

    async def update_policy_violation(
        self,
        violation_id: str,
        status: str,
        resolution_notes: str | None = None,
    ) -> dict[str, Any]:
        """Update violation status."""
        data: dict[str, Any] = {"status": status}
        if resolution_notes:
            data["resolution_notes"] = resolution_notes
        return await self._client.request(
            "PATCH", f"/api/v1/control-plane/policies/violations/{violation_id}", json=data
        )
