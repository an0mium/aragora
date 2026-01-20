"""
Health Monitor for the Aragora Control Plane.

Provides health monitoring with:
- Periodic health probes for agents
- Circuit breaker integration
- Cascading failure detection
- Health metrics for observability

The health monitor works with the AgentRegistry to track
agent health and automatically remove unhealthy agents from
the active pool.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from aragora.resilience import CircuitBreaker, get_circuit_breaker
from aragora.server.prometheus_control_plane import (
    record_control_plane_agent_health,
    record_control_plane_agent_latency,
)

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status for agents and the control plane."""

    HEALTHY = "healthy"  # All systems operational
    DEGRADED = "degraded"  # Some issues but operational
    UNHEALTHY = "unhealthy"  # Significant issues
    CRITICAL = "critical"  # System is failing


@dataclass
class HealthCheck:
    """
    Result of a health check.

    Attributes:
        agent_id: Agent that was checked
        status: Health status
        latency_ms: Check latency in milliseconds
        timestamp: When the check was performed
        error: Error message if unhealthy
        metadata: Additional health data
    """

    agent_id: str
    status: HealthStatus
    latency_ms: float
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "error": self.error,
            "metadata": self.metadata,
        }


class HealthMonitor:
    """
    Health monitoring system for the control plane.

    Monitors agent health through periodic probes and integrates
    with circuit breakers for automatic failure handling.

    Usage:
        from aragora.control_plane.registry import AgentRegistry

        registry = AgentRegistry()
        monitor = HealthMonitor(registry)

        # Register a health probe for an agent
        monitor.register_probe("claude-3", probe_func)

        # Start monitoring
        await monitor.start()

        # Get health status
        status = monitor.get_agent_health("claude-3")

        # Stop monitoring
        await monitor.stop()
    """

    def __init__(
        self,
        registry: Optional[Any] = None,  # AgentRegistry, optional to avoid circular import
        probe_interval: float = 30.0,
        probe_timeout: float = 10.0,
        unhealthy_threshold: int = 3,
        recovery_threshold: int = 2,
    ):
        """
        Initialize the health monitor.

        Args:
            registry: AgentRegistry for agent status updates
            probe_interval: Seconds between health probes
            probe_timeout: Timeout for health probe calls
            unhealthy_threshold: Consecutive failures before marking unhealthy
            recovery_threshold: Consecutive successes before marking healthy
        """
        self._registry = registry
        self._probe_interval = probe_interval
        self._probe_timeout = probe_timeout
        self._unhealthy_threshold = unhealthy_threshold
        self._recovery_threshold = recovery_threshold

        # Health probes: agent_id -> probe function
        self._probes: Dict[str, Callable[[], bool]] = {}

        # Health state tracking
        self._health_checks: Dict[str, HealthCheck] = {}
        self._failure_counts: Dict[str, int] = {}
        self._success_counts: Dict[str, int] = {}

        # Circuit breakers per agent
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Monitoring task
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

        # Event callbacks
        self._on_unhealthy: List[Callable[[str, HealthCheck], None]] = []
        self._on_recovered: List[Callable[[str, HealthCheck], None]] = []

    async def start(self) -> None:
        """Start the health monitoring loop."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"HealthMonitor started (interval={self._probe_interval}s)")

    async def stop(self) -> None:
        """Stop the health monitoring loop."""
        self._running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("HealthMonitor stopped")

    def register_probe(
        self,
        agent_id: str,
        probe: Callable[[], bool],
        circuit_breaker_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a health probe for an agent.

        Args:
            agent_id: Agent to monitor
            probe: Function that returns True if healthy, False otherwise
            circuit_breaker_config: Optional circuit breaker configuration
        """
        self._probes[agent_id] = probe
        self._failure_counts[agent_id] = 0
        self._success_counts[agent_id] = 0

        # Create circuit breaker for this agent
        cb_config = circuit_breaker_config or {}
        self._circuit_breakers[agent_id] = get_circuit_breaker(
            f"health_{agent_id}",
            failure_threshold=cb_config.get("failure_threshold", 3),
            cooldown_seconds=cb_config.get("cooldown_seconds", 60.0),
        )

        logger.debug(f"Registered health probe for agent: {agent_id}")

    def unregister_probe(self, agent_id: str) -> None:
        """
        Unregister a health probe.

        Args:
            agent_id: Agent to stop monitoring
        """
        self._probes.pop(agent_id, None)
        self._failure_counts.pop(agent_id, None)
        self._success_counts.pop(agent_id, None)
        self._health_checks.pop(agent_id, None)

        logger.debug(f"Unregistered health probe for agent: {agent_id}")

    def get_agent_health(self, agent_id: str) -> Optional[HealthCheck]:
        """
        Get the latest health check for an agent.

        Args:
            agent_id: Agent to query

        Returns:
            HealthCheck if available, None otherwise
        """
        return self._health_checks.get(agent_id)

    def get_all_health(self) -> Dict[str, HealthCheck]:
        """
        Get health status for all monitored agents.

        Returns:
            Dict mapping agent_id to HealthCheck
        """
        return self._health_checks.copy()

    def get_system_health(self) -> HealthStatus:
        """
        Get overall system health status.

        Returns:
            HealthStatus based on aggregate agent health
        """
        if not self._health_checks:
            return HealthStatus.HEALTHY

        statuses = [hc.status for hc in self._health_checks.values()]

        critical_count = sum(1 for s in statuses if s == HealthStatus.CRITICAL)
        unhealthy_count = sum(1 for s in statuses if s == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for s in statuses if s == HealthStatus.DEGRADED)

        total = len(statuses)

        # Critical if more than 50% are critical or unhealthy
        if (critical_count + unhealthy_count) / total > 0.5:
            return HealthStatus.CRITICAL

        # Unhealthy if more than 25% are unhealthy
        if unhealthy_count / total > 0.25:
            return HealthStatus.UNHEALTHY

        # Degraded if any agents are degraded or unhealthy
        if degraded_count + unhealthy_count > 0:
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    def on_unhealthy(self, callback: Callable[[str, HealthCheck], None]) -> None:
        """
        Register callback for when an agent becomes unhealthy.

        Args:
            callback: Function called with (agent_id, health_check)
        """
        self._on_unhealthy.append(callback)

    def on_recovered(self, callback: Callable[[str, HealthCheck], None]) -> None:
        """
        Register callback for when an agent recovers.

        Args:
            callback: Function called with (agent_id, health_check)
        """
        self._on_recovered.append(callback)

    def get_circuit_breaker(self, agent_id: str) -> Optional[CircuitBreaker]:
        """
        Get circuit breaker for an agent.

        Args:
            agent_id: Agent to query

        Returns:
            CircuitBreaker if registered, None otherwise
        """
        return self._circuit_breakers.get(agent_id)

    def is_agent_available(self, agent_id: str) -> bool:
        """
        Check if an agent is available (healthy and circuit closed).

        Args:
            agent_id: Agent to check

        Returns:
            True if agent is available for tasks
        """
        # Check circuit breaker
        cb = self._circuit_breakers.get(agent_id)
        if cb and not cb.can_proceed():
            return False

        # Check health status
        hc = self._health_checks.get(agent_id)
        if hc and hc.status in (HealthStatus.UNHEALTHY, HealthStatus.CRITICAL):
            return False

        return True

    def get_stats(self) -> Dict[str, Any]:
        """
        Get health monitoring statistics.

        Returns:
            Dict with health metrics
        """
        status_counts = {s.value: 0 for s in HealthStatus}
        total_latency = 0.0
        latency_count = 0

        for hc in self._health_checks.values():
            status_counts[hc.status.value] += 1
            if hc.latency_ms > 0:
                total_latency += hc.latency_ms
                latency_count += 1

        circuit_breaker_stats = {}
        for agent_id, cb in self._circuit_breakers.items():
            circuit_breaker_stats[agent_id] = {
                "status": cb.get_status(),
                "failures": cb.failures,
            }

        return {
            "system_health": self.get_system_health().value,
            "monitored_agents": len(self._probes),
            "by_status": status_counts,
            "avg_latency_ms": total_latency / latency_count if latency_count > 0 else 0.0,
            "circuit_breakers": circuit_breaker_stats,
            "probe_interval": self._probe_interval,
            "unhealthy_threshold": self._unhealthy_threshold,
        }

    async def _monitor_loop(self) -> None:
        """Background loop for health probing."""
        while self._running:
            try:
                await self._probe_all_agents()
                await asyncio.sleep(self._probe_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
                await asyncio.sleep(self._probe_interval)

    async def _probe_all_agents(self) -> None:
        """Run health probes for all registered agents."""
        tasks = []

        for agent_id, probe in self._probes.items():
            tasks.append(self._probe_agent(agent_id, probe))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _probe_agent(
        self,
        agent_id: str,
        probe: Callable[[], bool],
    ) -> None:
        """
        Run a health probe for a single agent.

        Args:
            agent_id: Agent to probe
            probe: Health probe function
        """
        start_time = time.time()
        error_msg = None
        is_healthy = False

        cb = self._circuit_breakers.get(agent_id)

        try:
            # Check circuit breaker
            if cb and not cb.can_proceed():
                # Circuit is open, don't probe
                return

            # Run probe with timeout
            is_healthy = await asyncio.wait_for(
                asyncio.to_thread(probe),
                timeout=self._probe_timeout,
            )

            if cb:
                if is_healthy:
                    cb.record_success()
                else:
                    cb.record_failure()

        except asyncio.TimeoutError:
            error_msg = "Health probe timeout"
            if cb:
                cb.record_failure()
        except Exception as e:
            error_msg = str(e)
            if cb:
                cb.record_failure()

        latency_ms = (time.time() - start_time) * 1000

        # Update health state
        await self._update_health_state(agent_id, is_healthy, latency_ms, error_msg)

    async def _update_health_state(
        self,
        agent_id: str,
        is_healthy: bool,
        latency_ms: float,
        error: Optional[str],
    ) -> None:
        """
        Update agent health state based on probe result.

        Args:
            agent_id: Agent that was probed
            is_healthy: Whether probe succeeded
            latency_ms: Probe latency
            error: Error message if probe failed
        """
        prev_check = self._health_checks.get(agent_id)
        was_healthy = prev_check is None or prev_check.status == HealthStatus.HEALTHY

        if is_healthy:
            self._failure_counts[agent_id] = 0
            self._success_counts[agent_id] = self._success_counts.get(agent_id, 0) + 1

            # Determine status based on latency
            if latency_ms > 5000:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY

            # Check recovery threshold
            if not was_healthy and self._success_counts[agent_id] >= self._recovery_threshold:
                status = HealthStatus.HEALTHY
                logger.info(f"Agent {agent_id} recovered")

                # Notify recovery callbacks
                check = HealthCheck(
                    agent_id=agent_id,
                    status=status,
                    latency_ms=latency_ms,
                )
                for callback in self._on_recovered:
                    try:
                        callback(agent_id, check)
                    except Exception as e:
                        logger.error(f"Recovery callback error: {e}")

        else:
            self._success_counts[agent_id] = 0
            self._failure_counts[agent_id] = self._failure_counts.get(agent_id, 0) + 1

            failure_count = self._failure_counts[agent_id]

            if failure_count >= self._unhealthy_threshold:
                status = HealthStatus.UNHEALTHY
            else:
                status = HealthStatus.DEGRADED

            # Notify unhealthy callbacks
            if was_healthy and status in (HealthStatus.UNHEALTHY, HealthStatus.CRITICAL):
                logger.warning(f"Agent {agent_id} became unhealthy: {error}")

                check = HealthCheck(
                    agent_id=agent_id,
                    status=status,
                    latency_ms=latency_ms,
                    error=error,
                )
                for callback in self._on_unhealthy:
                    try:
                        callback(agent_id, check)
                    except Exception as e:
                        logger.error(f"Unhealthy callback error: {e}")

        # Store health check
        self._health_checks[agent_id] = HealthCheck(
            agent_id=agent_id,
            status=status,
            latency_ms=latency_ms,
            error=error,
        )

        # Record Prometheus metrics
        health_value = {
            HealthStatus.HEALTHY: 2,
            HealthStatus.DEGRADED: 1,
            HealthStatus.UNHEALTHY: 0,
            HealthStatus.CRITICAL: 0,
        }.get(status, 0)
        record_control_plane_agent_health(agent_id, health_value)
        record_control_plane_agent_latency(agent_id, latency_ms / 1000.0)  # Convert to seconds

        # Update registry if available
        if self._registry:
            try:
                from aragora.control_plane.registry import AgentStatus

                if status == HealthStatus.UNHEALTHY:
                    await self._registry.update_status(agent_id, AgentStatus.FAILED)
                elif status == HealthStatus.HEALTHY and was_healthy is False:
                    await self._registry.update_status(agent_id, AgentStatus.READY)
            except Exception as e:
                logger.debug(f"Could not update registry for {agent_id}: {e}")
