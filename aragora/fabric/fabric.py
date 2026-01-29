"""
Agent Fabric - Unified facade for high-scale agent orchestration.

The AgentFabric class provides a single entry point for:
- Spawning and managing agents (50+ concurrent)
- Agent pools with auto-scaling
- Scheduling and executing tasks
- Enforcing policies and budgets
- Monitoring health and resource usage

This is the foundation for both Gastown (developer orchestration)
and Moltbot (consumer device) parity extensions.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from .models import (
    AgentConfig,
    AgentHandle,
    AgentInfo,
    BudgetConfig,
    BudgetStatus,
    HealthStatus,
    Policy,
    PolicyContext,
    PolicyDecision,
    Priority,
    Task,
    TaskHandle,
    Usage,
    UsageReport,
)
from .scheduler import AgentScheduler
from .lifecycle import LifecycleManager
from .policy import PolicyEngine
from .budget import BudgetManager

logger = logging.getLogger(__name__)


@dataclass
class FabricConfig:
    """Configuration for the Agent Fabric."""

    # Scheduler settings
    max_queue_depth: int = 1000
    default_timeout_seconds: float = 300.0

    # Lifecycle settings
    heartbeat_interval_seconds: float = 30.0
    heartbeat_timeout_seconds: float = 90.0
    drain_timeout_seconds: float = 30.0

    # Concurrency settings
    max_concurrent_agents: int = 100
    max_concurrent_tasks_per_agent: int = 5

    # Default budgets
    default_tokens_per_day: int | None = None
    default_cost_per_day_usd: float | None = None


@dataclass
class FabricStats:
    """Aggregated statistics from all fabric components."""

    # Agent stats
    agents_active: int = 0
    agents_healthy: int = 0
    agents_degraded: int = 0
    agents_unhealthy: int = 0
    agents_spawned: int = 0
    agents_terminated: int = 0
    agents_pooled: int = 0

    # Task stats
    tasks_pending: int = 0
    tasks_running: int = 0
    tasks_scheduled: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_cancelled: int = 0

    # Policy stats
    policies_active: int = 0
    decisions_allowed: int = 0
    decisions_denied: int = 0
    pending_approvals: int = 0

    # Budget stats
    entities_tracked: int = 0
    alerts_triggered: int = 0


@dataclass
class AgentPool:
    """A pool of agents with shared configuration."""

    id: str
    name: str
    model: str
    min_agents: int = 0
    max_agents: int = 10
    current_agents: list[str] = field(default_factory=list)
    budget: BudgetConfig | None = None
    policies: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


class AgentFabric:
    """
    Unified facade for high-scale agent orchestration.

    Supports 50+ concurrent agents with:
    - Agent pools for auto-scaling
    - Priority-based task scheduling
    - Policy-gated tool access with approvals
    - Per-agent budget tracking and enforcement
    - Health monitoring with circuit breakers

    Usage:
        async with AgentFabric() as fabric:
            # Create a pool
            pool = await fabric.create_pool("workers", "claude-3-opus", min_agents=5)

            # Schedule work to the pool
            task = Task(id="task-1", type="debate", payload={...})
            handle = await fabric.schedule_to_pool(task, pool.id)

            # Wait for result
            result = await fabric.wait_for_task(handle.task_id)
    """

    def __init__(
        self,
        config: FabricConfig | None = None,
        scheduler: AgentScheduler | None = None,
        lifecycle: LifecycleManager | None = None,
        policy: PolicyEngine | None = None,
        budget: BudgetManager | None = None,
    ) -> None:
        """
        Initialize the Agent Fabric.

        Args:
            config: Fabric configuration (uses defaults if None)
            scheduler: Custom scheduler (creates default if None)
            lifecycle: Custom lifecycle manager (creates default if None)
            policy: Custom policy engine (creates default if None)
            budget: Custom budget manager (creates default if None)
        """
        self._config = config or FabricConfig()
        self.scheduler = scheduler or AgentScheduler(
            max_queue_depth=self._config.max_queue_depth,
            default_timeout_seconds=self._config.default_timeout_seconds,
        )
        self.lifecycle = lifecycle or LifecycleManager(
            heartbeat_interval_seconds=self._config.heartbeat_interval_seconds,
            heartbeat_timeout_seconds=self._config.heartbeat_timeout_seconds,
            drain_timeout_seconds=self._config.drain_timeout_seconds,
        )
        self.policy = policy or PolicyEngine()
        self.budget = budget or BudgetManager()

        # Agent pools
        self._pools: dict[str, AgentPool] = {}

        # Task executors
        self._executors: dict[str, Callable[[Task, AgentHandle], Coroutine[Any, Any, Any]]] = {}

        self._started = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the Agent Fabric and all components."""
        if self._started:
            return
        await self.lifecycle.start()
        self._started = True
        logger.info("Agent Fabric started")

    async def stop(self) -> None:
        """Stop the Agent Fabric and all components."""
        if not self._started:
            return
        await self.lifecycle.stop()
        self._started = False
        logger.info("Agent Fabric stopped")

    async def __aenter__(self) -> "AgentFabric":
        """Context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        await self.stop()

    # =========================================================================
    # Agent Lifecycle
    # =========================================================================

    async def spawn(
        self,
        config: AgentConfig,
    ) -> AgentHandle:
        """
        Spawn a new agent.

        Args:
            config: Agent configuration

        Returns:
            Handle to the spawned agent
        """
        handle = await self.lifecycle.spawn(config)

        if config.budget:
            await self.budget.set_budget(config.id, config.budget)

        return handle

    async def terminate(
        self,
        agent_id: str,
        graceful: bool = True,
    ) -> bool:
        """Terminate an agent."""
        return await self.lifecycle.terminate(agent_id, graceful=graceful)

    async def get_agent(self, agent_id: str) -> AgentHandle | None:
        """Get an agent by ID."""
        return await self.lifecycle.get_agent(agent_id)

    async def list_agents(
        self,
        status: HealthStatus | None = None,
        model: str | None = None,
    ) -> list[AgentInfo]:
        """List agents with optional filters."""
        return await self.lifecycle.list_agents(status=status, model=model)

    async def heartbeat(self, agent_id: str) -> bool:
        """Record agent heartbeat."""
        return await self.lifecycle.heartbeat(agent_id)

    # =========================================================================
    # Pool Management
    # =========================================================================

    async def create_pool(
        self,
        name: str,
        model: str,
        min_agents: int = 0,
        max_agents: int = 10,
        budget: BudgetConfig | None = None,
        policies: list[str] | None = None,
    ) -> AgentPool:
        """
        Create an agent pool with optional warm agents.

        Args:
            name: Pool name
            model: Model for agents in this pool
            min_agents: Minimum agents to keep warm
            max_agents: Maximum agents allowed
            budget: Budget config for pool agents
            policies: Policy IDs to apply

        Returns:
            Created pool
        """
        import uuid

        pool_id = f"pool-{uuid.uuid4().hex[:8]}"
        pool = AgentPool(
            id=pool_id,
            name=name,
            model=model,
            min_agents=min_agents,
            max_agents=max_agents,
            budget=budget,
            policies=policies or [],
        )

        async with self._lock:
            self._pools[pool_id] = pool

        # Spawn minimum agents
        for i in range(min_agents):
            config = AgentConfig(
                id=f"{pool_id}-agent-{i}",
                model=model,
                pool_id=pool_id,
                budget=budget or BudgetConfig(),
                policies=policies or [],
            )
            await self.spawn(config)
            pool.current_agents.append(config.id)

        logger.info(f"Created pool {pool_id} ({name}) with {min_agents} agents")
        return pool

    async def get_pool(self, pool_id: str) -> AgentPool | None:
        """Get a pool by ID."""
        return self._pools.get(pool_id)

    async def list_pools(self) -> list[AgentPool]:
        """List all pools."""
        return list(self._pools.values())

    async def scale_pool(self, pool_id: str, target_agents: int) -> int:
        """
        Scale a pool to target number of agents.

        Args:
            pool_id: Pool to scale
            target_agents: Target agent count

        Returns:
            Actual agent count after scaling
        """
        pool = self._pools.get(pool_id)
        if not pool:
            raise ValueError(f"Pool {pool_id} not found")

        target = max(pool.min_agents, min(pool.max_agents, target_agents))
        current = len(pool.current_agents)

        if target > current:
            # Scale up
            for i in range(target - current):
                config = AgentConfig(
                    id=f"{pool_id}-agent-{current + i}",
                    model=pool.model,
                    pool_id=pool_id,
                    budget=pool.budget or BudgetConfig(),
                    policies=pool.policies,
                )
                await self.spawn(config)
                pool.current_agents.append(config.id)
        elif target < current:
            # Scale down
            agents_to_remove = pool.current_agents[target:]
            for agent_id in agents_to_remove:
                await self.terminate(agent_id, graceful=True)
            pool.current_agents = pool.current_agents[:target]

        logger.info(f"Scaled pool {pool_id} to {len(pool.current_agents)} agents")
        return len(pool.current_agents)

    async def delete_pool(self, pool_id: str) -> bool:
        """Delete a pool and terminate its agents."""
        pool = self._pools.get(pool_id)
        if not pool:
            return False

        # Terminate all agents
        for agent_id in list(pool.current_agents):
            await self.terminate(agent_id, graceful=True)

        async with self._lock:
            del self._pools[pool_id]

        logger.info(f"Deleted pool {pool_id}")
        return True

    async def schedule_to_pool(
        self,
        task: Task,
        pool_id: str,
        priority: Priority = Priority.NORMAL,
    ) -> TaskHandle:
        """
        Schedule a task to the best available agent in a pool.

        Uses least-loaded routing to distribute work evenly.

        Args:
            task: Task to schedule
            pool_id: Target pool
            priority: Task priority

        Returns:
            Task handle
        """
        pool = self._pools.get(pool_id)
        if not pool:
            raise ValueError(f"Pool {pool_id} not found")

        if not pool.current_agents:
            raise ValueError(f"Pool {pool_id} has no agents")

        # Find least-loaded agent
        best_agent = None
        best_pending = float("inf")

        for agent_id in pool.current_agents:
            pending = await self.scheduler.list_pending(agent_id)
            if len(pending) < best_pending:
                best_pending = len(pending)
                best_agent = agent_id

        if not best_agent:
            raise ValueError(f"No available agents in pool {pool_id}")

        return await self.schedule(task, best_agent, priority=priority)

    async def wait_for_task(
        self,
        task_id: str,
        timeout_seconds: float | None = None,
    ) -> TaskHandle | None:
        """
        Wait for a task to complete.

        Args:
            task_id: Task to wait for
            timeout_seconds: Maximum wait time

        Returns:
            Task handle or None if timeout
        """
        event = asyncio.Event()

        async def on_complete(handle: TaskHandle) -> None:
            event.set()

        self.scheduler.on_complete(task_id, on_complete)

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_seconds)
            return await self.scheduler.get_handle(task_id)
        except asyncio.TimeoutError:
            return None

    # =========================================================================
    # Task Scheduling
    # =========================================================================

    async def schedule(
        self,
        task: Task,
        agent_id: str,
        priority: Priority = Priority.NORMAL,
        depends_on: list[str] | None = None,
    ) -> TaskHandle:
        """
        Schedule a task for an agent.

        Args:
            task: Task to schedule
            agent_id: Target agent
            priority: Task priority
            depends_on: Task dependencies

        Returns:
            Task handle
        """
        agent = await self.lifecycle.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        return await self.scheduler.schedule(
            task=task,
            agent_id=agent_id,
            priority=priority,
            depends_on=depends_on,
        )

    async def get_task(self, task_id: str) -> TaskHandle | None:
        """Get a task by ID."""
        return await self.scheduler.get_handle(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        return await self.scheduler.cancel(task_id)

    async def pop_next_task(self, agent_id: str) -> Task | None:
        """Get the next task for an agent to execute."""
        return await self.scheduler.pop_next(agent_id)

    async def complete_task(
        self,
        task_id: str,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        """Mark a task as complete."""
        await self.scheduler.complete_task(task_id, result=result, error=error)

        handle = await self.scheduler.get_handle(task_id)
        if handle:
            await self.lifecycle.update_task_stats(
                handle.agent_id,
                completed=1 if not error else 0,
                failed=1 if error else 0,
            )

    # =========================================================================
    # Policy Enforcement
    # =========================================================================

    async def check_policy(
        self,
        action: str,
        context: PolicyContext,
    ) -> PolicyDecision:
        """Check if an action is allowed by policy."""
        return await self.policy.check(action, context)

    async def add_policy(self, policy: Policy) -> None:
        """Add a policy."""
        await self.policy.add_policy(policy)

    async def remove_policy(self, policy_id: str) -> bool:
        """Remove a policy."""
        return await self.policy.remove_policy(policy_id)

    async def request_approval(
        self,
        action: str,
        context: PolicyContext,
        approvers: list[str],
    ) -> bool:
        """Request approval for an action."""
        result = await self.policy.require_approval(action, context, approvers)
        return result.approved

    async def approve(self, request_id: str, approver_id: str) -> bool:
        """Approve a pending request."""
        return await self.policy.approve(request_id, approver_id)

    async def deny(self, request_id: str, denier_id: str) -> bool:
        """Deny a pending request."""
        return await self.policy.deny(request_id, denier_id)

    # =========================================================================
    # Budget Management
    # =========================================================================

    async def track_usage(self, usage: Usage) -> BudgetStatus:
        """Track resource usage."""
        return await self.budget.track(usage)

    async def check_budget(
        self,
        agent_id: str,
        estimated_tokens: int = 0,
        estimated_cost_usd: float = 0.0,
    ) -> tuple[bool, BudgetStatus]:
        """Check if operation is within budget."""
        return await self.budget.check_budget(
            agent_id,
            estimated_tokens=estimated_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )

    async def get_usage_report(
        self,
        entity_id: str,
        period_days: int = 1,
    ) -> UsageReport:
        """Get usage report for an entity."""
        return await self.budget.get_usage(entity_id, period_days=period_days)

    async def set_budget(self, entity_id: str, config: BudgetConfig) -> None:
        """Set budget for an entity."""
        await self.budget.set_budget(entity_id, config)

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics as a dict."""
        scheduler_stats = await self.scheduler.get_stats()
        lifecycle_stats = await self.lifecycle.get_stats()
        policy_stats = await self.policy.get_stats()
        budget_stats = await self.budget.get_stats()

        stats = {
            "scheduler": scheduler_stats,
            "lifecycle": lifecycle_stats,
            "policy": policy_stats,
            "budget": budget_stats,
            "pools": len(self._pools),
        }

        # Record to Prometheus metrics
        self._record_prometheus_metrics(stats)

        return stats

    def _record_prometheus_metrics(self, stats: dict[str, Any]) -> None:
        """Record stats to Prometheus metrics (graceful fallback if unavailable)."""
        try:
            from aragora.observability.metrics.fabric import record_fabric_stats

            record_fabric_stats(stats)
        except ImportError:
            pass  # Metrics not available

    async def get_fabric_stats(self) -> FabricStats:
        """Get aggregated statistics as FabricStats object."""
        lifecycle_stats = await self.lifecycle.get_stats()
        scheduler_stats = await self.scheduler.get_stats()
        policy_stats = await self.policy.get_stats()
        budget_stats = await self.budget.get_stats()

        return FabricStats(
            # Agent stats
            agents_active=lifecycle_stats["agents_active"],
            agents_healthy=lifecycle_stats["agents_healthy"],
            agents_degraded=lifecycle_stats["agents_degraded"],
            agents_unhealthy=lifecycle_stats["agents_unhealthy"],
            agents_spawned=lifecycle_stats["agents_spawned"],
            agents_terminated=lifecycle_stats["agents_terminated"],
            agents_pooled=lifecycle_stats["agents_pooled"],
            # Task stats
            tasks_pending=scheduler_stats["tasks_pending"],
            tasks_running=scheduler_stats["tasks_running"],
            tasks_scheduled=scheduler_stats["tasks_scheduled"],
            tasks_completed=scheduler_stats["tasks_completed"],
            tasks_failed=scheduler_stats["tasks_failed"],
            tasks_cancelled=scheduler_stats["tasks_cancelled"],
            # Policy stats
            policies_active=policy_stats["policies"],
            decisions_allowed=policy_stats["decisions_allowed"],
            decisions_denied=policy_stats["decisions_denied"],
            pending_approvals=policy_stats["pending_approvals"],
            # Budget stats
            entities_tracked=budget_stats["entities_tracked"],
            alerts_triggered=budget_stats["alerts_triggered"],
        )

    # =========================================================================
    # Task Execution
    # =========================================================================

    def register_executor(
        self,
        task_type: str,
        executor: Callable[[Task, AgentHandle], Coroutine[Any, Any, Any]],
    ) -> None:
        """
        Register an executor for a task type.

        Args:
            task_type: Task type to handle
            executor: Async function that executes the task
        """
        self._executors[task_type] = executor
        logger.debug(f"Registered executor for task type: {task_type}")
