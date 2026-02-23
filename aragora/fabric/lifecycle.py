"""
Agent Lifecycle Manager - Spawn, heartbeat, and termination.

Manages the lifecycle of agents including creation, health monitoring,
and graceful shutdown.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from .models import (
    AgentConfig,
    AgentHandle,
    AgentInfo,
    HealthStatus,
)

logger = logging.getLogger(__name__)


class LifecycleManager:
    """
    Manages agent lifecycle including spawn, heartbeat, and termination.

    Features:
    - Agent spawning with configuration
    - Heartbeat monitoring
    - Graceful shutdown with task draining
    - Agent pooling for fast spawn
    - Health status tracking
    """

    def __init__(
        self,
        heartbeat_interval_seconds: float = 30.0,
        heartbeat_timeout_seconds: float = 90.0,
        drain_timeout_seconds: float = 30.0,
    ) -> None:
        self._heartbeat_interval = heartbeat_interval_seconds
        self._heartbeat_timeout = heartbeat_timeout_seconds
        self._drain_timeout = drain_timeout_seconds

        self._agents: dict[str, AgentHandle] = {}
        self._pools: dict[str, list[AgentHandle]] = {}
        self._shutting_down: set[str] = set()
        self._health_checker: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._agents_spawned = 0
        self._agents_terminated = 0

    async def start(self) -> None:
        """Start the lifecycle manager background tasks."""
        if self._health_checker is None:
            self._health_checker = asyncio.create_task(self._health_check_loop())
            logger.info("Lifecycle manager started")

    async def stop(self) -> None:
        """Stop the lifecycle manager and terminate all agents."""
        if self._health_checker:
            self._health_checker.cancel()
            try:
                await self._health_checker
            except asyncio.CancelledError:
                pass
            self._health_checker = None

        agent_ids = list(self._agents.keys())
        for agent_id in agent_ids:
            await self.terminate(agent_id, graceful=False)

        logger.info("Lifecycle manager stopped")

    async def spawn(self, config: AgentConfig) -> AgentHandle:
        """Spawn a new agent with the given configuration."""
        async with self._lock:
            if config.id in self._agents:
                raise ValueError(f"Agent {config.id} already exists")

            pool_id = config.pool_id
            if pool_id and pool_id in self._pools and self._pools[pool_id]:
                handle = self._pools[pool_id].pop()
                handle.agent_id = config.id
                handle.config = config
                handle.status = HealthStatus.HEALTHY
                handle.last_heartbeat = datetime.now(timezone.utc)
                self._agents[config.id] = handle
                logger.debug("Reused agent from pool %s as %s", pool_id, config.id)
                return handle

            handle = AgentHandle(
                agent_id=config.id,
                config=config,
                spawned_at=datetime.now(timezone.utc),
                status=HealthStatus.HEALTHY,
                last_heartbeat=datetime.now(timezone.utc),
            )

            self._agents[config.id] = handle
            self._agents_spawned += 1
            logger.info("Spawned agent %s (model=%s)", config.id, config.model)
            return handle

    async def terminate(
        self,
        agent_id: str,
        graceful: bool = True,
        drain_timeout: float | None = None,
    ) -> bool:
        """Terminate an agent."""
        async with self._lock:
            if agent_id not in self._agents:
                return False
            if agent_id in self._shutting_down:
                return True
            self._shutting_down.add(agent_id)

        try:
            handle = self._agents[agent_id]

            if graceful:
                timeout = drain_timeout or self._drain_timeout
                logger.debug("Draining agent %s (timeout=%ss)", agent_id, timeout)
                await asyncio.sleep(0.1)

            pool_id = handle.config.pool_id
            if pool_id:
                async with self._lock:
                    if pool_id not in self._pools:
                        self._pools[pool_id] = []
                    self._pools[pool_id].append(handle)
                    del self._agents[agent_id]
                    logger.debug("Returned agent %s to pool %s", agent_id, pool_id)
            else:
                async with self._lock:
                    del self._agents[agent_id]
                    self._agents_terminated += 1
                    logger.info("Terminated agent %s", agent_id)

            return True
        finally:
            self._shutting_down.discard(agent_id)

    async def heartbeat(self, agent_id: str) -> bool:
        """Record a heartbeat from an agent."""
        async with self._lock:
            if agent_id not in self._agents:
                return False
            handle = self._agents[agent_id]
            handle.last_heartbeat = datetime.now(timezone.utc)
            if handle.status == HealthStatus.DEGRADED:
                handle.status = HealthStatus.HEALTHY
            return True

    async def get_health(self, agent_id: str) -> HealthStatus | None:
        """Get the health status of an agent."""
        handle = self._agents.get(agent_id)
        return handle.status if handle else None

    async def get_agent(self, agent_id: str) -> AgentHandle | None:
        """Get an agent handle."""
        return self._agents.get(agent_id)

    async def list_agents(
        self,
        status: HealthStatus | None = None,
        model: str | None = None,
    ) -> list[AgentInfo]:
        """List agents with optional filters."""
        results = []
        async with self._lock:
            for handle in self._agents.values():
                if status and handle.status != status:
                    continue
                if model and handle.config.model != model:
                    continue

                info = AgentInfo(
                    agent_id=handle.agent_id,
                    model=handle.config.model,
                    status=handle.status,
                    spawned_at=handle.spawned_at,
                    last_heartbeat=handle.last_heartbeat,
                    tasks_pending=0,
                    tasks_running=0,
                    tasks_completed=handle.tasks_completed,
                    tasks_failed=handle.tasks_failed,
                    budget_usage_percent=0.0,
                )
                results.append(info)
        return results

    async def update_task_stats(
        self,
        agent_id: str,
        completed: int = 0,
        failed: int = 0,
    ) -> None:
        """Update task statistics for an agent."""
        async with self._lock:
            if agent_id in self._agents:
                handle = self._agents[agent_id]
                handle.tasks_completed += completed
                handle.tasks_failed += failed

    async def _health_check_loop(self) -> None:
        """Background loop to check agent health."""
        while True:
            try:
                await asyncio.sleep(self._heartbeat_interval)
                await self._check_all_health()
            except asyncio.CancelledError:
                break
            except (RuntimeError, OSError, ValueError) as e:
                logger.error("Health check error: %s", e)

    async def _check_all_health(self) -> None:
        """Check health of all agents."""
        now = datetime.now(timezone.utc)
        timeout_threshold = now - timedelta(seconds=self._heartbeat_timeout)
        degraded_threshold = now - timedelta(seconds=self._heartbeat_interval * 2)

        async with self._lock:
            for agent_id, handle in self._agents.items():
                if agent_id in self._shutting_down:
                    continue
                if handle.last_heartbeat is None:
                    continue

                if handle.last_heartbeat < timeout_threshold:
                    if handle.status != HealthStatus.UNHEALTHY:
                        handle.status = HealthStatus.UNHEALTHY
                        logger.warning("Agent %s marked UNHEALTHY", agent_id)
                elif handle.last_heartbeat < degraded_threshold:
                    if handle.status == HealthStatus.HEALTHY:
                        handle.status = HealthStatus.DEGRADED
                        logger.debug("Agent %s marked DEGRADED", agent_id)

    async def get_stats(self) -> dict[str, Any]:
        """Get lifecycle manager statistics."""
        async with self._lock:
            healthy = sum(1 for h in self._agents.values() if h.status == HealthStatus.HEALTHY)
            degraded = sum(1 for h in self._agents.values() if h.status == HealthStatus.DEGRADED)
            unhealthy = sum(1 for h in self._agents.values() if h.status == HealthStatus.UNHEALTHY)
            pooled = sum(len(p) for p in self._pools.values())

            return {
                "agents_spawned": self._agents_spawned,
                "agents_terminated": self._agents_terminated,
                "agents_active": len(self._agents),
                "agents_healthy": healthy,
                "agents_degraded": degraded,
                "agents_unhealthy": unhealthy,
                "agents_pooled": pooled,
                "pools": len(self._pools),
            }
