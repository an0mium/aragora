"""
Control Plane Integration Module.

Bridges the enterprise ControlPlaneCoordinator with the SharedControlPlaneState
to provide unified agent/task visibility across both systems.

This enables:
- Agents registered via coordinator are visible in shared state (UI dashboard)
- Tasks submitted via coordinator are visible in shared state
- Events from coordinator are broadcast through shared state streams
- Metrics are aggregated from both systems

Usage:
    from aragora.control_plane.integration import (
        setup_control_plane_integration,
        IntegratedControlPlane,
    )

    # Set up integration at startup
    integrated = await setup_control_plane_integration()

    # Use as single entry point
    await integrated.register_agent(...)
    await integrated.submit_task(...)

    # Both systems are kept in sync automatically
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from aragora.control_plane.coordinator import (
    ControlPlaneConfig,
    ControlPlaneCoordinator,
)
from aragora.control_plane.registry import AgentCapability, AgentInfo, AgentStatus
from aragora.control_plane.scheduler import Task, TaskPriority
from aragora.control_plane.shared_state import (
    SharedControlPlaneState,
    set_shared_state,
)

logger = logging.getLogger(__name__)


class IntegratedControlPlane:
    """
    Unified control plane that keeps coordinator and shared state in sync.

    Wraps ControlPlaneCoordinator and automatically syncs state changes
    to SharedControlPlaneState for UI visibility.

    All operations are performed on the coordinator (source of truth),
    with changes mirrored to shared state for dashboards and streaming.
    """

    def __init__(
        self,
        coordinator: ControlPlaneCoordinator,
        shared_state: SharedControlPlaneState,
        sync_interval: float = 5.0,
    ):
        """
        Initialize integrated control plane.

        Args:
            coordinator: Enterprise control plane coordinator
            shared_state: Shared state for UI/dashboard persistence
            sync_interval: Interval for periodic state sync (seconds)
        """
        self._coordinator = coordinator
        self._shared_state = shared_state
        self._sync_interval = sync_interval
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False

    @property
    def coordinator(self) -> ControlPlaneCoordinator:
        """Access underlying coordinator."""
        return self._coordinator

    @property
    def shared_state(self) -> SharedControlPlaneState:
        """Access underlying shared state."""
        return self._shared_state

    async def start(self) -> None:
        """Start integration sync."""
        if self._running:
            return

        self._running = True
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info("IntegratedControlPlane started")

    async def stop(self) -> None:
        """Stop integration sync."""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        logger.info("IntegratedControlPlane stopped")

    async def _sync_loop(self) -> None:
        """Periodically sync state from coordinator to shared state."""
        while self._running:
            try:
                await self._sync_agents()
                await asyncio.sleep(self._sync_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
                await asyncio.sleep(self._sync_interval)

    async def _sync_agents(self) -> None:
        """Sync agents from coordinator to shared state."""
        try:
            agents = await self._coordinator.list_agents(only_available=False)
            for agent in agents:
                await self._sync_agent_to_shared_state(agent)
        except Exception as e:
            logger.debug(f"Agent sync failed: {e}")

    async def _sync_agent_to_shared_state(self, agent: AgentInfo) -> None:
        """Sync a single agent to shared state."""
        try:
            # Map AgentInfo to shared state format
            agent_data = {
                "id": agent.agent_id,
                "name": agent.agent_id,
                "type": agent.provider,
                "model": agent.model,
                "status": self._map_agent_status(agent.status),
                "capabilities": [str(c) for c in agent.capabilities],
                "tasks_completed": agent.tasks_completed,
                "avg_response_time": agent.avg_latency_ms,
                "error_rate": agent.tasks_failed / (agent.tasks_completed + agent.tasks_failed)
                if (agent.tasks_completed + agent.tasks_failed) > 0
                else 0.0,
                "last_active": datetime.fromtimestamp(
                    agent.last_heartbeat, tz=timezone.utc
                ).isoformat()
                if agent.last_heartbeat
                else None,
                "metadata": agent.metadata or {},
            }
            await self._shared_state.register_agent(agent_data)
        except Exception as e:
            logger.debug(f"Failed to sync agent {agent.agent_id}: {e}")

    def _map_agent_status(self, status: AgentStatus) -> str:
        """Map AgentStatus enum to shared state status string."""
        mapping = {
            AgentStatus.STARTING: "idle",
            AgentStatus.READY: "active",
            AgentStatus.BUSY: "active",
            AgentStatus.DRAINING: "paused",
            AgentStatus.OFFLINE: "offline",
            AgentStatus.FAILED: "offline",
        }
        return mapping.get(status, "idle")

    # =========================================================================
    # Agent Operations (sync to shared state)
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
        Register an agent with automatic sync to shared state.

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
        # Register with coordinator
        agent = await self._coordinator.register_agent(
            agent_id=agent_id,
            capabilities=capabilities,
            model=model,
            provider=provider,
            metadata=metadata,
            health_probe=health_probe,
        )

        # Sync to shared state
        await self._sync_agent_to_shared_state(agent)

        # Broadcast event
        await self._shared_state._broadcast_event(
            {
                "type": "agent_registered",
                "agent_id": agent_id,
                "model": model,
                "provider": provider,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        return agent

    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent.

        Args:
            agent_id: Agent to unregister

        Returns:
            True if unregistered
        """
        result = await self._coordinator.unregister_agent(agent_id)

        if result:
            # Update shared state
            await self._shared_state.update_agent_status(agent_id, "offline")
            await self._shared_state._broadcast_event(
                {
                    "type": "agent_unregistered",
                    "agent_id": agent_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        return result

    async def pause_agent(self, agent_id: str) -> bool:
        """
        Pause an agent.

        Args:
            agent_id: Agent to pause

        Returns:
            True if paused
        """
        # Update coordinator status (DRAINING means completing current task, no new tasks)
        result = await self._coordinator.heartbeat(
            agent_id,
            status=AgentStatus.DRAINING,
        )

        if result:
            # Sync to shared state
            await self._shared_state.update_agent_status(agent_id, "paused")

        return result

    async def resume_agent(self, agent_id: str) -> bool:
        """
        Resume a paused agent.

        Args:
            agent_id: Agent to resume

        Returns:
            True if resumed
        """
        result = await self._coordinator.heartbeat(
            agent_id,
            status=AgentStatus.READY,
        )

        if result:
            await self._shared_state.update_agent_status(agent_id, "active")

        return result

    async def list_agents(
        self,
        capability: Optional[str | AgentCapability] = None,
        only_available: bool = True,
    ) -> List[AgentInfo]:
        """List agents from coordinator."""
        return await self._coordinator.list_agents(
            capability=capability,
            only_available=only_available,
        )

    async def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """Get agent from coordinator."""
        return await self._coordinator.get_agent(agent_id)

    # =========================================================================
    # Task Operations (sync to shared state)
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
        Submit a task with sync to shared state.

        Args:
            task_type: Type of task
            payload: Task data
            required_capabilities: Required agent capabilities
            priority: Task priority
            timeout_seconds: Task timeout
            metadata: Additional metadata

        Returns:
            Task ID
        """
        task_id = await self._coordinator.submit_task(
            task_type=task_type,
            payload=payload,
            required_capabilities=required_capabilities,
            priority=priority,
            timeout_seconds=timeout_seconds,
            metadata=metadata,
        )

        # Sync to shared state
        priority_str = {
            TaskPriority.HIGH: "high",
            TaskPriority.NORMAL: "normal",
            TaskPriority.LOW: "low",
        }.get(priority, "normal")

        await self._shared_state.add_task(
            {
                "id": task_id,
                "type": task_type,
                "priority": priority_str,
                "status": "pending",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "payload": payload,
                "metadata": metadata or {},
            }
        )

        return task_id

    async def complete_task(
        self,
        task_id: str,
        result: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        latency_ms: Optional[float] = None,
    ) -> bool:
        """
        Complete a task with sync to shared state.

        Args:
            task_id: Task to complete
            result: Task result
            agent_id: Agent that completed the task
            latency_ms: Execution time

        Returns:
            True if completed
        """
        success = await self._coordinator.complete_task(
            task_id=task_id,
            result=result,
            agent_id=agent_id,
            latency_ms=latency_ms,
        )

        if success:
            # Update shared state
            await self._shared_state.update_task_priority(task_id, "normal")

            # Record activity for agent
            if agent_id:
                await self._shared_state.record_agent_activity(
                    agent_id,
                    tasks_completed=1,
                    response_time_ms=latency_ms,
                )

            # Broadcast completion
            await self._shared_state._broadcast_event(
                {
                    "type": "task_completed",
                    "task_id": task_id,
                    "agent_id": agent_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

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
        Fail a task with sync to shared state.

        Args:
            task_id: Task that failed
            error: Error message
            agent_id: Agent that failed
            latency_ms: Execution time
            requeue: Whether to requeue for retry

        Returns:
            True if processed
        """
        success = await self._coordinator.fail_task(
            task_id=task_id,
            error=error,
            agent_id=agent_id,
            latency_ms=latency_ms,
            requeue=requeue,
        )

        if success and agent_id:
            await self._shared_state.record_agent_activity(
                agent_id,
                response_time_ms=latency_ms,
                error=True,
            )

        return success

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task from coordinator."""
        return await self._coordinator.get_task(task_id)

    async def wait_for_result(
        self,
        task_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[Task]:
        """Wait for task completion."""
        return await self._coordinator.wait_for_result(task_id, timeout)

    # =========================================================================
    # Deliberation Operations (first-class task type)
    # =========================================================================

    async def submit_deliberation(
        self,
        question: str,
        context: Optional[str] = None,
        agents: Optional[List[str]] = None,
        priority: str = "normal",
        timeout_seconds: float = 300.0,
        max_rounds: int = 5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Submit a deliberation as a first-class control plane task.

        Deliberations are routed through the scheduler with SLA tracking
        and ELO updates on completion.

        Args:
            question: The question to deliberate
            context: Optional context for the deliberation
            agents: Specific agents to use (optional)
            priority: Task priority (low, normal, high, urgent)
            timeout_seconds: SLA timeout
            max_rounds: Maximum debate rounds
            metadata: Additional metadata

        Returns:
            Task ID for tracking
        """
        from aragora.control_plane.deliberation import (
            DeliberationManager,
        )

        # Create deliberation manager with ELO callback
        manager = DeliberationManager(
            coordinator=self._coordinator,
            elo_callback=self._create_elo_callback(),
            notification_callback=self._create_notification_callback(),
        )

        task_id = await manager.submit_deliberation(
            question=question,
            context=context,
            agents=agents,
            priority=priority,
            timeout_seconds=timeout_seconds,
            max_rounds=max_rounds,
            metadata=metadata,
        )

        # Broadcast deliberation started event
        await self._shared_state._broadcast_event(
            {
                "type": "deliberation_started",
                "task_id": task_id,
                "question_preview": question[:100],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        return task_id

    async def wait_for_deliberation(
        self,
        task_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Wait for a deliberation to complete.

        Args:
            task_id: Deliberation task ID
            timeout: Optional timeout override

        Returns:
            Deliberation outcome dict or None
        """
        from aragora.control_plane.deliberation import DeliberationManager

        manager = DeliberationManager(
            coordinator=self._coordinator,
            elo_callback=self._create_elo_callback(),
        )

        outcome = await manager.wait_for_outcome(task_id, timeout)
        if outcome:
            return {
                "task_id": outcome.task_id,
                "success": outcome.success,
                "consensus_reached": outcome.consensus_reached,
                "consensus_confidence": outcome.consensus_confidence,
                "winning_position": outcome.winning_position,
                "duration_seconds": outcome.duration_seconds,
                "sla_compliant": outcome.sla_compliant,
            }
        return None

    def _create_elo_callback(self) -> Callable[[Any], None]:
        """Create callback to update ELO on deliberation completion."""

        def elo_callback(outcome: Any) -> None:
            """Feed deliberation outcome to ELO system."""
            try:
                from aragora.ranking.elo import get_elo_store

                elo_system = get_elo_store()
                if not elo_system or not outcome.agent_performances:
                    return

                # Extract participating agents and their performance
                agents = list(outcome.agent_performances.keys())
                if len(agents) < 2:
                    return

                # Calculate scores based on consensus contribution
                scores: Dict[str, float] = {}
                for agent_id, perf in outcome.agent_performances.items():
                    score = 0.5  # Base score
                    if perf.contributed_to_consensus:
                        score += 0.3
                    if perf.final_position_correct:
                        score += 0.2
                    scores[agent_id] = score

                # Record match with confidence weighting
                confidence_weight = outcome.consensus_confidence or 0.5
                elo_system.record_match(
                    debate_id=outcome.task_id,
                    participants=agents,
                    scores=scores,
                    domain="deliberation",
                    confidence_weight=confidence_weight,
                )

                logger.debug(f"Updated ELO for deliberation {outcome.task_id}: {scores}")

            except ImportError:
                logger.debug("ELO system not available")
            except Exception as e:
                logger.error(f"Failed to update ELO: {e}")

        return elo_callback

    def _create_notification_callback(self) -> Callable[[str, Dict[str, Any]], None]:
        """Create callback for deliberation notifications."""

        def notification_callback(event_type: str, data: Dict[str, Any]) -> None:
            """Handle deliberation SLA notifications."""
            try:
                from aragora.control_plane.notifications import (
                    create_notification_dispatcher,
                )
                from aragora.control_plane.channels import (
                    create_deliberation_consensus_notification,
                    NotificationEventType,
                    NotificationPriority,
                )

                # Create dispatcher (will use any configured channels)
                dispatcher = create_notification_dispatcher()

                if event_type == "sla_warning":
                    asyncio.create_task(
                        dispatcher.dispatch(
                            event_type=NotificationEventType.SLA_WARNING,
                            title="Vetted decisionmaking SLA Warning",
                            body=f"Task {data['task_id'][:8]}... approaching timeout "
                            f"({data['elapsed_seconds']:.0f}s / {data['timeout_seconds']:.0f}s)",
                            priority=NotificationPriority.HIGH,
                            metadata=data,
                        )
                    )
                elif event_type == "sla_violated":
                    asyncio.create_task(
                        dispatcher.dispatch(
                            event_type=NotificationEventType.SLA_VIOLATION,
                            title="Vetted decisionmaking SLA Violation",
                            body=f"Task {data['task_id'][:8]}... exceeded timeout "
                            f"({data['elapsed_seconds']:.0f}s > {data['timeout_seconds']:.0f}s)",
                            priority=NotificationPriority.URGENT,
                            metadata=data,
                        )
                    )
                elif event_type == "consensus_reached":
                    message = create_deliberation_consensus_notification(
                        task_id=data.get("task_id", ""),
                        question=data.get("question", "Unknown question"),
                        answer=data.get("answer") or "No answer provided",
                        confidence=float(data.get("confidence") or 0.0),
                    )
                    asyncio.create_task(
                        dispatcher.dispatch(
                            event_type=message.event_type,
                            title=message.title,
                            body=message.body,
                            priority=message.priority,
                            metadata=message.metadata,
                            workspace_id=data.get("workspace_id"),
                            link_url=message.link_url,
                            link_text=message.link_text,
                        )
                    )
                elif event_type == "no_consensus":
                    asyncio.create_task(
                        dispatcher.dispatch(
                            event_type=NotificationEventType.DELIBERATION_FAILED,
                            title="Vetted decisionmaking completed without consensus",
                            body=f"Task {data['task_id'][:8]}... completed without consensus.",
                            priority=NotificationPriority.NORMAL,
                            metadata=data,
                            workspace_id=data.get("workspace_id"),
                        )
                    )
                elif event_type == "deliberation_failed":
                    asyncio.create_task(
                        dispatcher.dispatch(
                            event_type=NotificationEventType.DELIBERATION_FAILED,
                            title="Vetted decisionmaking failed",
                            body=f"Task {data['task_id'][:8]}... failed: {data.get('error', 'Unknown error')}",
                            priority=NotificationPriority.HIGH,
                            metadata=data,
                            workspace_id=data.get("workspace_id"),
                        )
                    )

            except ImportError:
                logger.debug("Notification dispatcher not available")
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")

        return notification_callback

    # =========================================================================
    # Metrics (aggregated from both systems)
    # =========================================================================

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive stats from both systems.

        Returns:
            Combined stats from coordinator and shared state
        """
        coordinator_stats = await self._coordinator.get_stats()
        shared_stats = await self._shared_state.get_metrics()

        return {
            "coordinator": coordinator_stats,
            "shared_state": shared_stats,
            "integrated": {
                "sync_interval": self._sync_interval,
                "persistent_backend": self._shared_state.is_persistent,
            },
        }


# Module-level singleton
_integrated: Optional[IntegratedControlPlane] = None


async def setup_control_plane_integration(
    config: Optional[ControlPlaneConfig] = None,
    redis_url: str = "redis://localhost:6379",
    sync_interval: float = 5.0,
) -> IntegratedControlPlane:
    """
    Set up integrated control plane with both coordinator and shared state.

    This is the recommended way to initialize the control plane for production.
    It ensures both systems are connected and kept in sync.

    Args:
        config: Optional coordinator config (uses env vars if not provided)
        redis_url: Redis URL for shared state
        sync_interval: State sync interval in seconds

    Returns:
        IntegratedControlPlane instance

    Usage:
        from aragora.control_plane.integration import setup_control_plane_integration

        # At application startup
        integrated = await setup_control_plane_integration()

        # Register agents
        await integrated.register_agent("claude-3", ["debate", "reasoning"])

        # Submit tasks
        task_id = await integrated.submit_task("debate", {"question": "..."})

        # At shutdown
        await integrated.stop()
    """
    global _integrated

    if _integrated is not None:
        return _integrated

    # Create coordinator
    coordinator = await ControlPlaneCoordinator.create(config)

    # Create shared state
    shared_state = SharedControlPlaneState(redis_url=redis_url)
    await shared_state.connect()
    set_shared_state(shared_state)

    # Create integrated instance
    _integrated = IntegratedControlPlane(
        coordinator=coordinator,
        shared_state=shared_state,
        sync_interval=sync_interval,
    )
    await _integrated.start()

    logger.info("Control plane integration set up successfully")
    return _integrated


def get_integrated_control_plane() -> Optional[IntegratedControlPlane]:
    """
    Get the global integrated control plane instance.

    Returns:
        IntegratedControlPlane or None if not initialized
    """
    return _integrated


async def shutdown_control_plane() -> None:
    """Shutdown the global integrated control plane."""
    global _integrated

    if _integrated:
        await _integrated.stop()
        await _integrated.coordinator.shutdown()
        await _integrated.shared_state.close()
        _integrated = None
        logger.info("Control plane integration shut down")


__all__ = [
    "IntegratedControlPlane",
    "setup_control_plane_integration",
    "get_integrated_control_plane",
    "shutdown_control_plane",
]
