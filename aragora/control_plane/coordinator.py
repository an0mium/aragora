"""
Control Plane Coordinator for Aragora.

Provides a unified high-level API for the control plane, coordinating
between the AgentRegistry, TaskScheduler, and HealthMonitor.

This is the main entry point for control plane operations.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from aragora.control_plane.health import HealthCheck, HealthMonitor, HealthStatus
from aragora.control_plane.registry import (
    AgentCapability,
    AgentInfo,
    AgentRegistry,
    AgentStatus,
)
from aragora.control_plane.scheduler import Task, TaskPriority, TaskScheduler, TaskStatus

logger = logging.getLogger(__name__)


@dataclass
class ControlPlaneConfig:
    """Configuration for the control plane."""

    redis_url: str = "redis://localhost:6379"
    key_prefix: str = "aragora:cp:"
    heartbeat_timeout: float = 30.0
    heartbeat_interval: float = 10.0
    probe_interval: float = 30.0
    probe_timeout: float = 10.0
    task_timeout: float = 300.0
    max_task_retries: int = 3
    cleanup_interval: float = 60.0

    @classmethod
    def from_env(cls) -> "ControlPlaneConfig":
        """Create config from environment variables."""
        return cls(
            redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379"),
            key_prefix=os.environ.get("CONTROL_PLANE_PREFIX", "aragora:cp:"),
            heartbeat_timeout=float(os.environ.get("HEARTBEAT_TIMEOUT", "30")),
            heartbeat_interval=float(os.environ.get("HEARTBEAT_INTERVAL", "10")),
            probe_interval=float(os.environ.get("PROBE_INTERVAL", "30")),
            probe_timeout=float(os.environ.get("PROBE_TIMEOUT", "10")),
            task_timeout=float(os.environ.get("TASK_TIMEOUT", "300")),
            max_task_retries=int(os.environ.get("MAX_TASK_RETRIES", "3")),
            cleanup_interval=float(os.environ.get("CLEANUP_INTERVAL", "60")),
        )


class ControlPlaneCoordinator:
    """
    Unified coordinator for the Aragora control plane.

    Provides high-level operations that coordinate between:
    - AgentRegistry: Service discovery and agent management
    - TaskScheduler: Task distribution and lifecycle
    - HealthMonitor: Health tracking and circuit breakers

    Usage:
        # Create and connect
        coordinator = await ControlPlaneCoordinator.create()

        # Register agents
        await coordinator.register_agent(
            agent_id="claude-3",
            capabilities=["debate", "code"],
            model="claude-3-opus",
        )

        # Submit tasks
        task_id = await coordinator.submit_task(
            task_type="debate",
            payload={"question": "..."},
            required_capabilities=["debate"],
        )

        # Wait for completion
        result = await coordinator.wait_for_result(task_id, timeout=60.0)

        # Shutdown
        await coordinator.shutdown()
    """

    def __init__(
        self,
        config: Optional[ControlPlaneConfig] = None,
        registry: Optional[AgentRegistry] = None,
        scheduler: Optional[TaskScheduler] = None,
        health_monitor: Optional[HealthMonitor] = None,
    ):
        """
        Initialize the coordinator.

        Args:
            config: Control plane configuration
            registry: Optional pre-configured AgentRegistry
            scheduler: Optional pre-configured TaskScheduler
            health_monitor: Optional pre-configured HealthMonitor
        """
        self._config = config or ControlPlaneConfig.from_env()

        self._registry = registry or AgentRegistry(
            redis_url=self._config.redis_url,
            key_prefix=f"{self._config.key_prefix}agents:",
            heartbeat_timeout=self._config.heartbeat_timeout,
            cleanup_interval=self._config.cleanup_interval,
        )

        self._scheduler = scheduler or TaskScheduler(
            redis_url=self._config.redis_url,
            key_prefix=f"{self._config.key_prefix}tasks:",
            stream_prefix=f"{self._config.key_prefix}stream:",
        )

        self._health_monitor = health_monitor or HealthMonitor(
            registry=self._registry,
            probe_interval=self._config.probe_interval,
            probe_timeout=self._config.probe_timeout,
        )

        self._connected = False
        self._result_waiters: Dict[str, asyncio.Event] = {}

    @classmethod
    async def create(
        cls,
        config: Optional[ControlPlaneConfig] = None,
    ) -> "ControlPlaneCoordinator":
        """
        Create and connect a coordinator.

        Args:
            config: Optional configuration

        Returns:
            Connected ControlPlaneCoordinator
        """
        coordinator = cls(config)
        await coordinator.connect()
        return coordinator

    async def connect(self) -> None:
        """Connect to Redis and start background services."""
        if self._connected:
            return

        await self._registry.connect()
        await self._scheduler.connect()
        await self._health_monitor.start()

        self._connected = True
        logger.info("ControlPlaneCoordinator connected")

    async def shutdown(self) -> None:
        """Shutdown the coordinator and all services."""
        if not self._connected:
            return

        await self._health_monitor.stop()
        await self._scheduler.close()
        await self._registry.close()

        self._connected = False
        logger.info("ControlPlaneCoordinator shutdown complete")

    # =========================================================================
    # Agent Operations
    # =========================================================================

    async def register_agent(
        self,
        agent_id: str,
        capabilities: List[str | AgentCapability],
        model: str = "unknown",
        provider: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
        health_probe: Optional[Callable[[], bool]] = None,
    ) -> AgentInfo:
        """
        Register an agent with the control plane.

        Args:
            agent_id: Unique agent identifier
            capabilities: Agent capabilities
            model: Model name
            provider: Provider name
            metadata: Additional metadata
            health_probe: Optional health check function

        Returns:
            AgentInfo for the registered agent
        """
        agent = await self._registry.register(
            agent_id=agent_id,
            capabilities=capabilities,
            model=model,
            provider=provider,
            metadata=metadata,
        )

        # Register health probe if provided
        if health_probe:
            self._health_monitor.register_probe(agent_id, health_probe)

        return agent

    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the control plane.

        Args:
            agent_id: Agent to unregister

        Returns:
            True if unregistered, False if not found
        """
        self._health_monitor.unregister_probe(agent_id)
        return await self._registry.unregister(agent_id)

    async def heartbeat(
        self,
        agent_id: str,
        status: Optional[AgentStatus] = None,
    ) -> bool:
        """
        Send agent heartbeat.

        Args:
            agent_id: Agent sending heartbeat
            status: Optional status update

        Returns:
            True if recorded, False if agent not found
        """
        return await self._registry.heartbeat(agent_id, status)

    async def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """
        Get agent information.

        Args:
            agent_id: Agent to look up

        Returns:
            AgentInfo if found
        """
        return await self._registry.get(agent_id)

    async def list_agents(
        self,
        capability: Optional[str | AgentCapability] = None,
        only_available: bool = True,
    ) -> List[AgentInfo]:
        """
        List registered agents.

        Args:
            capability: Optional capability filter
            only_available: Only return available agents

        Returns:
            List of matching agents
        """
        if capability:
            return await self._registry.find_by_capability(
                capability, only_available=only_available
            )
        return await self._registry.list_all(include_offline=not only_available)

    async def select_agent(
        self,
        capabilities: List[str | AgentCapability],
        strategy: str = "least_loaded",
        exclude: Optional[List[str]] = None,
    ) -> Optional[AgentInfo]:
        """
        Select an agent for a task.

        Args:
            capabilities: Required capabilities
            strategy: Selection strategy
            exclude: Agent IDs to exclude

        Returns:
            Selected agent or None
        """
        # Also exclude unhealthy agents
        all_excluded = set(exclude or [])

        for agent_id in list(self._health_monitor._health_checks.keys()):
            if not self._health_monitor.is_agent_available(agent_id):
                all_excluded.add(agent_id)

        return await self._registry.select_agent(
            capabilities=capabilities,
            strategy=strategy,
            exclude=list(all_excluded),
        )

    # =========================================================================
    # Task Operations
    # =========================================================================

    async def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        required_capabilities: Optional[List[str]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Submit a task for execution.

        Args:
            task_type: Type of task
            payload: Task data
            required_capabilities: Required agent capabilities
            priority: Task priority
            timeout_seconds: Task timeout (uses config default if not specified)
            metadata: Additional metadata

        Returns:
            Task ID
        """
        return await self._scheduler.submit(
            task_type=task_type,
            payload=payload,
            required_capabilities=required_capabilities,
            priority=priority,
            timeout_seconds=timeout_seconds or self._config.task_timeout,
            max_retries=self._config.max_task_retries,
            metadata=metadata,
        )

    async def claim_task(
        self,
        agent_id: str,
        capabilities: List[str],
        block_ms: int = 5000,
    ) -> Optional[Task]:
        """
        Claim a task for an agent.

        Args:
            agent_id: Agent claiming the task
            capabilities: Agent's capabilities
            block_ms: Time to block waiting

        Returns:
            Task if claimed, None otherwise
        """
        task = await self._scheduler.claim(
            worker_id=agent_id,
            capabilities=capabilities,
            block_ms=block_ms,
        )

        if task:
            # Update agent status
            await self._registry.heartbeat(
                agent_id,
                status=AgentStatus.BUSY,
                current_task_id=task.id,
            )

        return task

    async def complete_task(
        self,
        task_id: str,
        result: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        latency_ms: Optional[float] = None,
    ) -> bool:
        """
        Mark a task as completed.

        Args:
            task_id: Task to complete
            result: Task result
            agent_id: Agent that completed the task
            latency_ms: Execution time

        Returns:
            True if completed, False if not found
        """
        success = await self._scheduler.complete(task_id, result)

        if success and agent_id:
            # Update agent metrics
            await self._registry.record_task_completion(
                agent_id,
                success=True,
                latency_ms=latency_ms or 0.0,
            )

            # Notify waiters
            if task_id in self._result_waiters:
                self._result_waiters[task_id].set()

        return success

    async def fail_task(
        self,
        task_id: str,
        error: str,
        agent_id: Optional[str] = None,
        latency_ms: Optional[float] = None,
        requeue: bool = True,
    ) -> bool:
        """
        Mark a task as failed.

        Args:
            task_id: Task that failed
            error: Error message
            agent_id: Agent that failed
            latency_ms: Execution time
            requeue: Whether to requeue for retry

        Returns:
            True if processed, False if not found
        """
        success = await self._scheduler.fail(task_id, error, requeue)

        if success and agent_id:
            await self._registry.record_task_completion(
                agent_id,
                success=False,
                latency_ms=latency_ms or 0.0,
            )

        # Notify waiters if not requeued
        task = await self._scheduler.get(task_id)
        if task and task.status in (TaskStatus.FAILED, TaskStatus.CANCELLED):
            if task_id in self._result_waiters:
                self._result_waiters[task_id].set()

        return success

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.

        Args:
            task_id: Task to cancel

        Returns:
            True if cancelled, False otherwise
        """
        success = await self._scheduler.cancel(task_id)

        if success and task_id in self._result_waiters:
            self._result_waiters[task_id].set()

        return success

    async def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get task by ID.

        Args:
            task_id: Task to retrieve

        Returns:
            Task if found
        """
        return await self._scheduler.get(task_id)

    async def wait_for_result(
        self,
        task_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[Task]:
        """
        Wait for a task to complete.

        Args:
            task_id: Task to wait for
            timeout: Maximum wait time in seconds

        Returns:
            Completed task, or None if timeout/not found
        """
        task = await self._scheduler.get(task_id)
        if not task:
            return None

        # Already completed?
        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            return task

        # Create waiter
        if task_id not in self._result_waiters:
            self._result_waiters[task_id] = asyncio.Event()

        try:
            await asyncio.wait_for(
                self._result_waiters[task_id].wait(),
                timeout=timeout or self._config.task_timeout,
            )
            return await self._scheduler.get(task_id)
        except asyncio.TimeoutError:
            return None
        finally:
            self._result_waiters.pop(task_id, None)

    # =========================================================================
    # Health Operations
    # =========================================================================

    def get_agent_health(self, agent_id: str) -> Optional[HealthCheck]:
        """
        Get health status for an agent.

        Args:
            agent_id: Agent to query

        Returns:
            HealthCheck if available
        """
        return self._health_monitor.get_agent_health(agent_id)

    def get_system_health(self) -> HealthStatus:
        """
        Get overall system health.

        Returns:
            System HealthStatus
        """
        return self._health_monitor.get_system_health()

    def is_agent_available(self, agent_id: str) -> bool:
        """
        Check if an agent is available.

        Args:
            agent_id: Agent to check

        Returns:
            True if agent is available for tasks
        """
        return self._health_monitor.is_agent_available(agent_id)

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive control plane statistics.

        Returns:
            Dict with registry, scheduler, and health stats
        """
        return {
            "registry": await self._registry.get_stats(),
            "scheduler": await self._scheduler.get_stats(),
            "health": self._health_monitor.get_stats(),
            "config": {
                "redis_url": self._config.redis_url,
                "heartbeat_timeout": self._config.heartbeat_timeout,
                "task_timeout": self._config.task_timeout,
            },
        }


async def create_control_plane(
    config: Optional[ControlPlaneConfig] = None,
) -> ControlPlaneCoordinator:
    """
    Convenience function to create a connected control plane.

    Args:
        config: Optional configuration

    Returns:
        Connected ControlPlaneCoordinator
    """
    return await ControlPlaneCoordinator.create(config)
