"""
Control Plane Coordinator for Aragora.

Provides a unified high-level API for the control plane, coordinating
between the AgentRegistry, TaskScheduler, and HealthMonitor.

This is the main entry point for control plane operations.
"""

from __future__ import annotations

import asyncio
import os
import time
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

# Observability
from aragora.observability import (
    get_logger,
    create_span,
    add_span_attributes,
)

# Policy imports (optional - graceful fallback if not available)
try:
    from aragora.control_plane.policy import (
        ControlPlanePolicyManager,
        PolicyViolation,
        PolicyViolationError,
        EnforcementLevel,
    )

    HAS_POLICY = True
except ImportError:
    HAS_POLICY = False
    ControlPlanePolicyManager = None  # type: ignore
    PolicyViolation = None  # type: ignore
    PolicyViolationError = None  # type: ignore
    EnforcementLevel = None  # type: ignore

# Optional KM integration
try:
    from aragora.knowledge.mound.adapters.control_plane_adapter import (
        ControlPlaneAdapter,
        TaskOutcome,
    )

    HAS_KM_ADAPTER = True
except ImportError:
    HAS_KM_ADAPTER = False
    ControlPlaneAdapter = None  # type: ignore
    TaskOutcome = None  # type: ignore

# Optional Arena Bridge
try:
    from aragora.control_plane.arena_bridge import (  # noqa: F401
        ArenaControlPlaneBridge,
        get_arena_bridge,
        init_arena_bridge,
    )
    from aragora.control_plane.deliberation import (  # noqa: F401
        DELIBERATION_TASK_TYPE,
        AgentPerformance,
        DeliberationOutcome,
        DeliberationTask,
    )

    HAS_ARENA_BRIDGE = True
except ImportError:
    HAS_ARENA_BRIDGE = False
    ArenaControlPlaneBridge = None  # type: ignore
    DeliberationTask = None  # type: ignore
    DeliberationOutcome = None  # type: ignore
    DELIBERATION_TASK_TYPE = "deliberation"

logger = get_logger(__name__)


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

    # Knowledge Mound integration
    enable_km_integration: bool = True
    km_workspace_id: str = "default"

    # Policy sync from compliance store
    enable_policy_sync: bool = True
    policy_sync_workspace: Optional[str] = None

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
            enable_km_integration=os.environ.get("CP_ENABLE_KM", "true").lower() == "true",
            km_workspace_id=os.environ.get("CP_KM_WORKSPACE", "default"),
            enable_policy_sync=os.environ.get("CP_ENABLE_POLICY_SYNC", "true").lower() == "true",
            policy_sync_workspace=os.environ.get("CP_POLICY_SYNC_WORKSPACE") or None,
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
        km_adapter: Optional["ControlPlaneAdapter"] = None,
        knowledge_mound: Optional[Any] = None,
        arena_bridge: Optional["ArenaControlPlaneBridge"] = None,
        stream_server: Optional[Any] = None,
        shared_state: Optional[Any] = None,
        policy_manager: Optional["ControlPlanePolicyManager"] = None,
    ):
        """
        Initialize the coordinator.

        Args:
            config: Control plane configuration
            registry: Optional pre-configured AgentRegistry
            scheduler: Optional pre-configured TaskScheduler
            health_monitor: Optional pre-configured HealthMonitor
            km_adapter: Optional pre-configured ControlPlaneAdapter
            knowledge_mound: Optional KnowledgeMound for auto-creating adapter
            arena_bridge: Optional ArenaControlPlaneBridge for debate execution
            stream_server: Optional ControlPlaneStreamServer for event broadcasting
            shared_state: Optional SharedControlPlaneState for persistence
            policy_manager: Optional ControlPlanePolicyManager for policy enforcement
        """
        self._config = config or ControlPlaneConfig.from_env()
        self._stream_server = stream_server
        self._shared_state = shared_state
        self._policy_manager: Optional["ControlPlanePolicyManager"] = None
        if policy_manager:
            self._policy_manager = policy_manager
        elif HAS_POLICY:
            self._policy_manager = ControlPlanePolicyManager(
                violation_callback=self._handle_policy_violation
            )

        # Auto-sync policies from compliance store
        if self._policy_manager and self._config.enable_policy_sync:
            try:
                synced = self._policy_manager.sync_from_compliance_store(
                    workspace_id=self._config.policy_sync_workspace,
                    enabled_only=True,
                )
                if synced > 0:
                    logger.info(
                        "control_plane_policy_sync",
                        synced_policies=synced,
                        workspace=self._config.policy_sync_workspace,
                    )
            except Exception as e:
                logger.debug(f"Policy sync skipped: {e}")

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
            policy_manager=self._policy_manager,
        )

        self._health_monitor = health_monitor or HealthMonitor(
            registry=self._registry,
            probe_interval=self._config.probe_interval,
            probe_timeout=self._config.probe_timeout,
        )

        # Knowledge Mound integration
        self._km_adapter: Optional["ControlPlaneAdapter"] = None
        if self._config.enable_km_integration and HAS_KM_ADAPTER:
            if km_adapter:
                self._km_adapter = km_adapter
            elif knowledge_mound:
                self._km_adapter = ControlPlaneAdapter(
                    coordinator=self,
                    knowledge_mound=knowledge_mound,
                    workspace_id=self._config.km_workspace_id,
                )

        # Arena Bridge integration for unified debate execution
        self._arena_bridge: Optional["ArenaControlPlaneBridge"] = None
        if HAS_ARENA_BRIDGE:
            if arena_bridge:
                self._arena_bridge = arena_bridge
            elif stream_server or shared_state:
                # Auto-create bridge if streaming/state components provided
                self._arena_bridge = ArenaControlPlaneBridge(
                    stream_server=stream_server,
                    shared_state=shared_state,
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

        with create_span(
            "control_plane.connect",
            {"redis_url": self._config.redis_url},
        ) as span:
            start = time.monotonic()

            await self._registry.connect()
            await self._scheduler.connect()
            await self._health_monitor.start()

            self._connected = True
            latency_ms = (time.monotonic() - start) * 1000
            add_span_attributes(span, {"latency_ms": latency_ms, "success": True})
            logger.info(
                "control_plane_connected",
                latency_ms=latency_ms,
                redis_url=self._config.redis_url,
            )

    async def shutdown(self) -> None:
        """Shutdown the coordinator and all services."""
        if not self._connected:
            return

        with create_span("control_plane.shutdown") as span:
            start = time.monotonic()

            await self._health_monitor.stop()
            await self._scheduler.close()
            await self._registry.close()

            self._connected = False
            latency_ms = (time.monotonic() - start) * 1000
            add_span_attributes(span, {"latency_ms": latency_ms})
            logger.info("control_plane_shutdown", latency_ms=latency_ms)

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _schedule_async(self, coro: Any) -> None:
        """Schedule a coroutine regardless of sync/async context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                from aragora.utils.async_utils import run_async

                run_async(coro)
            except Exception as e:
                logger.debug("async_schedule_failed", error=str(e))
        else:
            loop.create_task(coro)

    def _handle_policy_violation(self, violation: "PolicyViolation") -> None:
        """Handle policy violations with audit logging and notifications."""
        try:
            enforcement = getattr(violation.enforcement_level, "value", "hard")
            decision = "warn" if enforcement == "warn" else "deny"
        except Exception:
            decision = "deny"

        async def _log_violation() -> None:
            try:
                from aragora.control_plane.audit import log_policy_decision
            except ImportError:
                return

            await log_policy_decision(
                policy_id=violation.policy_id,
                decision=decision,
                task_type=violation.task_type or "unknown",
                reason=violation.description,
                workspace_id=violation.workspace_id,
                task_id=violation.task_id,
                agent_id=violation.agent_id,
                violations=[violation.violation_type],
                metadata={
                    "policy_name": violation.policy_name,
                    "region": violation.region,
                },
            )

        self._schedule_async(_log_violation())

        try:
            from aragora.control_plane.notifications import get_default_notification_dispatcher
            from aragora.control_plane.channels import (
                NotificationEventType,
                NotificationPriority,
            )

            dispatcher = get_default_notification_dispatcher()
            if dispatcher:
                title = f"Policy {'Warning' if decision == 'warn' else 'Violation'}: {violation.policy_name}"
                body = (
                    f"Policy `{violation.policy_id}` blocked task `{(violation.task_id or '')[:8]}...` "
                    f"for agent `{violation.agent_id or 'unknown'}`.\n\n"
                    f"Reason: {violation.description}"
                )
                self._schedule_async(
                    dispatcher.dispatch(
                        event_type=NotificationEventType.POLICY_VIOLATION,
                        title=title,
                        body=body,
                        priority=NotificationPriority.HIGH,
                        metadata=violation.to_dict(),
                        workspace_id=violation.workspace_id,
                    )
                )
        except Exception as e:
            logger.debug("policy_notification_failed", error=str(e))

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
        with create_span(
            "control_plane.register_agent",
            {
                "agent_id": agent_id,
                "model": model,
                "provider": provider,
                "capability_count": len(capabilities),
            },
        ):
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

            logger.info(
                "agent_registered",
                agent_id=agent_id,
                model=model,
                provider=provider,
                capabilities=[str(c) for c in capabilities],
            )
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
        task_type: Optional[str] = None,
        use_km_recommendations: bool = True,
    ) -> Optional[AgentInfo]:
        """
        Select an agent for a task.

        Args:
            capabilities: Required capabilities
            strategy: Selection strategy
            exclude: Agent IDs to exclude
            task_type: Task type for KM-based recommendations
            use_km_recommendations: Whether to use KM history for weighting

        Returns:
            Selected agent or None
        """
        # Also exclude unhealthy agents
        all_excluded = set(exclude or [])

        for agent_id in list(self._health_monitor._health_checks.keys()):
            if not self._health_monitor.is_agent_available(agent_id):
                all_excluded.add(agent_id)

        # If KM integration enabled and task_type provided, use KM recommendations
        if use_km_recommendations and self._km_adapter and task_type and HAS_KM_ADAPTER:
            return await self._select_agent_with_km(
                capabilities=capabilities,
                task_type=task_type,
                exclude=list(all_excluded),
            )

        return await self._registry.select_agent(
            capabilities=capabilities,
            strategy=strategy,
            exclude=list(all_excluded),
        )

    async def _select_agent_with_km(
        self,
        capabilities: List[str | AgentCapability],
        task_type: str,
        exclude: Optional[List[str]] = None,
    ) -> Optional[AgentInfo]:
        """
        Select an agent using KM-based historical recommendations.

        Queries the Knowledge Mound for agent success rates on similar
        tasks and uses this to weight the selection.

        Args:
            capabilities: Required capabilities
            task_type: Type of task
            exclude: Agent IDs to exclude

        Returns:
            Selected agent or None
        """
        # Get available agents with required capabilities
        available_agents = []
        for cap in capabilities:
            agents = await self._registry.find_by_capability(cap, only_available=True)
            for agent in agents:
                if agent.agent_id not in (exclude or []):
                    available_agents.append(agent)

        if not available_agents:
            return None

        # Deduplicate
        agent_map = {a.agent_id: a for a in available_agents}
        agent_ids = list(agent_map.keys())

        # Get KM recommendations
        try:
            cap_strings = [str(c) for c in capabilities]
            recommendations = await self._km_adapter.get_agent_recommendations_for_task(
                task_type=task_type,
                available_agents=agent_ids,
                required_capabilities=cap_strings,
                top_n=len(agent_ids),
            )

            if recommendations:
                # Select the highest-scoring agent
                best_rec = recommendations[0]
                selected_id = best_rec["agent_id"]

                logger.debug(
                    "km_agent_selection",
                    task_type=task_type,
                    selected=selected_id,
                    score=best_rec.get("combined_score", 0),
                    km_recommendations=[r["agent_id"] for r in recommendations[:3]],
                )

                return agent_map.get(selected_id)

        except Exception as e:
            logger.debug(f"KM recommendation failed, using fallback: {e}")

        # Fallback to registry selection
        return await self._registry.select_agent(
            capabilities=capabilities,
            strategy="least_loaded",
            exclude=exclude,
        )

    async def get_agent_recommendations_from_km(
        self,
        task_type: str,
        capabilities: List[str],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get agent recommendations from Knowledge Mound.

        Public method for querying KM for agent recommendations without
        selecting an agent.

        Args:
            task_type: Type of task
            capabilities: Required capabilities
            limit: Maximum recommendations to return

        Returns:
            List of agent recommendations with scores
        """
        if not self._km_adapter or not HAS_KM_ADAPTER:
            return []

        # Get available agents
        available_agents = []
        for cap in capabilities:
            agents = await self._registry.find_by_capability(cap, only_available=True)
            available_agents.extend([a.agent_id for a in agents])

        # Deduplicate
        available_agents = list(set(available_agents))

        if not available_agents:
            return []

        try:
            return await self._km_adapter.get_agent_recommendations_for_task(
                task_type=task_type,
                available_agents=available_agents,
                required_capabilities=capabilities,
                top_n=limit,
            )
        except Exception as e:
            logger.debug(f"Failed to get KM recommendations: {e}")
            return []

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
        workspace_id: Optional[str] = None,
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
            workspace_id: Optional workspace ID for policy scoping

        Returns:
            Task ID

        Raises:
            PolicyViolationError: If task violates HARD enforcement policy
        """
        with create_span(
            "control_plane.submit_task",
            {
                "task_type": task_type,
                "priority": priority.value,
                "required_capabilities": str(required_capabilities or []),
            },
        ) as span:
            start = time.monotonic()

            task_id = await self._scheduler.submit(
                task_type=task_type,
                payload=payload,
                required_capabilities=required_capabilities,
                priority=priority,
                timeout_seconds=timeout_seconds or self._config.task_timeout,
                max_retries=self._config.max_task_retries,
                metadata=metadata,
                workspace_id=workspace_id,
            )

            latency_ms = (time.monotonic() - start) * 1000
            add_span_attributes(span, {"task_id": task_id, "latency_ms": latency_ms})
            logger.info(
                "task_submitted",
                task_id=task_id,
                task_type=task_type,
                priority=priority.value,
                latency_ms=latency_ms,
            )

            # Emit task submitted notification
            try:
                from aragora.control_plane.task_events import emit_task_submitted

                await emit_task_submitted(
                    task_id=task_id,
                    task_type=task_type,
                    priority=priority.name,
                    workspace_id=workspace_id
                    or (metadata.get("workspace_id") if metadata else None),
                    metadata=metadata,
                )
            except Exception:
                pass  # Don't fail submission on notification error

            return task_id

    async def claim_task(
        self,
        agent_id: str,
        capabilities: List[str],
        block_ms: int = 5000,
        agent_region: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> Optional[Task]:
        """
        Claim a task for an agent.

        Args:
            agent_id: Agent claiming the task
            capabilities: Agent's capabilities
            block_ms: Time to block waiting
            agent_region: Region where the agent is located (for policy checks)
            workspace_id: Workspace ID for policy scoping

        Returns:
            Task if claimed, None otherwise
        """
        task = await self._scheduler.claim(
            worker_id=agent_id,
            capabilities=capabilities,
            block_ms=block_ms,
            worker_region=agent_region,
            workspace_id=workspace_id,
        )

        if task:
            # Update agent status
            await self._registry.heartbeat(
                agent_id,
                status=AgentStatus.BUSY,
                current_task_id=task.id,
            )

            # Emit task claimed notification
            try:
                from aragora.control_plane.task_events import emit_task_claimed

                await emit_task_claimed(
                    task_id=task.id,
                    task_type=task.task_type,
                    agent_id=agent_id,
                    workspace_id=task.metadata.get("workspace_id") if task.metadata else None,
                )
            except Exception:
                pass  # Don't fail claim on notification error

        return task

    async def complete_task(
        self,
        task_id: str,
        result: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        latency_ms: Optional[float] = None,
        sla_policy_id: Optional[str] = None,
    ) -> bool:
        """
        Mark a task as completed.

        Args:
            task_id: Task to complete
            result: Task result
            agent_id: Agent that completed the task
            latency_ms: Execution time
            sla_policy_id: Optional policy ID to check SLA compliance against

        Returns:
            True if completed, False if not found
        """
        with create_span(
            "control_plane.complete_task",
            {
                "task_id": task_id,
                "agent_id": agent_id or "unknown",
                "execution_latency_ms": latency_ms or 0.0,
            },
        ) as span:
            # Get task details before completing (for KM storage)
            task = await self._scheduler.get(task_id)
            if task:
                add_span_attributes(span, {"task_type": task.task_type})

            success = await self._scheduler.complete(task_id, result)
            add_span_attributes(span, {"success": success})

            if success and agent_id:
                # Update agent metrics
                await self._registry.record_task_completion(
                    agent_id,
                    success=True,
                    latency_ms=latency_ms or 0.0,
                )

                # SLA compliance check (if policy manager and policy_id provided)
                if self._policy_manager and sla_policy_id and task and HAS_POLICY:
                    execution_seconds = (latency_ms or 0.0) / 1000.0
                    queue_seconds = None
                    if task.assigned_at and task.created_at:
                        queue_seconds = task.assigned_at - task.created_at

                    sla_result = self._policy_manager.evaluate_sla_compliance(
                        policy_id=sla_policy_id,
                        execution_seconds=execution_seconds,
                        queue_seconds=queue_seconds,
                        task_id=task_id,
                        task_type=task.task_type,
                        agent_id=agent_id,
                        workspace=task.metadata.get("workspace_id") if task.metadata else None,
                    )

                    if not sla_result.allowed:
                        if sla_result.enforcement_level == EnforcementLevel.WARN:
                            logger.warning(
                                "sla_warning_on_complete",
                                task_id=task_id,
                                agent_id=agent_id,
                                reason=sla_result.reason,
                                policy_id=sla_policy_id,
                                execution_seconds=execution_seconds,
                            )
                        else:
                            logger.error(
                                "sla_violation_on_complete",
                                task_id=task_id,
                                agent_id=agent_id,
                                reason=sla_result.reason,
                                policy_id=sla_policy_id,
                                execution_seconds=execution_seconds,
                            )
                        add_span_attributes(
                            span,
                            {
                                "sla_compliant": False,
                                "sla_violation_reason": sla_result.reason,
                            },
                        )
                    else:
                        add_span_attributes(span, {"sla_compliant": True})

                # Store outcome in Knowledge Mound
                if self._km_adapter and task and HAS_KM_ADAPTER:
                    try:
                        outcome = TaskOutcome(
                            task_id=task_id,
                            task_type=task.task_type,
                            agent_id=agent_id,
                            success=True,
                            duration_seconds=(latency_ms or 0.0) / 1000.0,
                            workspace_id=self._config.km_workspace_id,
                            metadata=task.metadata or {},
                        )
                        await self._km_adapter.store_task_outcome(outcome)
                    except Exception as e:
                        logger.debug("km_store_failed", error=str(e), task_id=task_id)

                # Notify waiters
                if task_id in self._result_waiters:
                    self._result_waiters[task_id].set()

                logger.info(
                    "task_completed",
                    task_id=task_id,
                    agent_id=agent_id,
                    task_type=task.task_type if task else "unknown",
                    latency_ms=latency_ms or 0.0,
                )

                # Emit task completed notification
                try:
                    from aragora.control_plane.task_events import emit_task_completed

                    await emit_task_completed(
                        task_id=task_id,
                        task_type=task.task_type if task else "unknown",
                        agent_id=agent_id,
                        duration_seconds=(latency_ms or 0.0) / 1000.0,
                        workspace_id=task.metadata.get("workspace_id")
                        if task and task.metadata
                        else None,
                    )
                except Exception:
                    pass  # Don't fail completion on notification error

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
        with create_span(
            "control_plane.fail_task",
            {
                "task_id": task_id,
                "agent_id": agent_id or "unknown",
                "requeue": requeue,
                "error_message": error[:200],  # Truncate long errors
            },
        ) as span:
            # Get task details before failing (for KM storage)
            task = await self._scheduler.get(task_id)
            if task:
                add_span_attributes(span, {"task_type": task.task_type})

            success = await self._scheduler.fail(task_id, error, requeue)
            add_span_attributes(span, {"success": success})

            if success and agent_id:
                await self._registry.record_task_completion(
                    agent_id,
                    success=False,
                    latency_ms=latency_ms or 0.0,
                )

                # Store failure outcome in Knowledge Mound (only if not requeuing)
                if self._km_adapter and task and not requeue and HAS_KM_ADAPTER:
                    try:
                        outcome = TaskOutcome(
                            task_id=task_id,
                            task_type=task.task_type,
                            agent_id=agent_id,
                            success=False,
                            duration_seconds=(latency_ms or 0.0) / 1000.0,
                            workspace_id=self._config.km_workspace_id,
                            error_message=error,
                            metadata=task.metadata or {},
                        )
                        await self._km_adapter.store_task_outcome(outcome)
                    except Exception as e:
                        logger.debug("km_store_failed", error=str(e), task_id=task_id)

            # Notify waiters if not requeued
            task = await self._scheduler.get(task_id)
            if task and task.status in (TaskStatus.FAILED, TaskStatus.CANCELLED):
                if task_id in self._result_waiters:
                    self._result_waiters[task_id].set()

            logger.warning(
                "task_failed",
                task_id=task_id,
                agent_id=agent_id,
                task_type=task.task_type if task else "unknown",
                error=error[:200],
                requeued=requeue,
            )

            # Emit task failed notification
            try:
                from aragora.control_plane.task_events import emit_task_failed

                await emit_task_failed(
                    task_id=task_id,
                    task_type=task.task_type if task else "unknown",
                    agent_id=agent_id,
                    error=error,
                    will_retry=requeue,
                    workspace_id=task.metadata.get("workspace_id")
                    if task and task.metadata
                    else None,
                )
            except Exception:
                pass  # Don't fail on notification error

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
        stats = {
            "registry": await self._registry.get_stats(),
            "scheduler": await self._scheduler.get_stats(),
            "health": self._health_monitor.get_stats(),
            "config": {
                "redis_url": self._config.redis_url,
                "heartbeat_timeout": self._config.heartbeat_timeout,
                "task_timeout": self._config.task_timeout,
            },
        }

        # Add KM adapter stats if available
        if self._km_adapter:
            stats["knowledge_mound"] = self._km_adapter.get_stats()

        # Add policy manager stats if available
        if self._policy_manager and HAS_POLICY:
            stats["policy"] = self._policy_manager.get_metrics()

        return stats

    # =========================================================================
    # Policy Manager Integration
    # =========================================================================

    @property
    def policy_manager(self) -> Optional["ControlPlanePolicyManager"]:
        """Get the Policy Manager if configured."""
        return self._policy_manager

    def set_policy_manager(self, manager: "ControlPlanePolicyManager") -> None:
        """
        Set the Policy Manager.

        Also updates the scheduler's policy manager reference.

        Args:
            manager: ControlPlanePolicyManager instance
        """
        self._policy_manager = manager
        # Also update the scheduler's policy manager
        self._scheduler._policy_manager = manager

    # =========================================================================
    # Knowledge Mound Integration
    # =========================================================================

    @property
    def km_adapter(self) -> Optional["ControlPlaneAdapter"]:
        """Get the Knowledge Mound adapter if configured."""
        return self._km_adapter

    def set_km_adapter(self, adapter: "ControlPlaneAdapter") -> None:
        """
        Set the Knowledge Mound adapter.

        Args:
            adapter: ControlPlaneAdapter instance
        """
        self._km_adapter = adapter

    async def get_agent_recommendations(
        self,
        capability: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get agent recommendations from Knowledge Mound.

        Uses historical performance data to recommend agents for a capability.

        Args:
            capability: Capability to query
            limit: Maximum recommendations

        Returns:
            List of agent recommendation dicts with success rates
        """
        if not self._km_adapter:
            return []

        try:
            records = await self._km_adapter.get_capability_recommendations(capability, limit=limit)
            return [
                {
                    "agent_id": r.agent_id,
                    "capability": r.capability,
                    "success_rate": r.success_count / max(1, r.success_count + r.failure_count),
                    "avg_duration_seconds": r.avg_duration_seconds,
                    "confidence": r.confidence,
                }
                for r in records
            ]
        except Exception as e:
            logger.debug(f"Failed to get agent recommendations: {e}")
            return []

    # =========================================================================
    # Arena Bridge Integration
    # =========================================================================

    @property
    def arena_bridge(self) -> Optional["ArenaControlPlaneBridge"]:
        """Get the Arena Bridge if configured."""
        return self._arena_bridge

    def set_arena_bridge(self, bridge: "ArenaControlPlaneBridge") -> None:
        """
        Set the Arena Bridge.

        Args:
            bridge: ArenaControlPlaneBridge instance
        """
        self._arena_bridge = bridge

    async def execute_deliberation(
        self,
        task: "DeliberationTask",
        agents: Optional[List[Any]] = None,
        workspace_id: Optional[str] = None,
    ) -> Optional["DeliberationOutcome"]:
        """
        Execute a deliberation using the Arena Bridge.

        This provides unified debate orchestration with SLA tracking and
        real-time event streaming through the control plane.

        Args:
            task: DeliberationTask to execute
            agents: Optional list of Agent instances (if not provided, selects from registry)
            workspace_id: Optional workspace for knowledge mound scoping

        Returns:
            DeliberationOutcome if bridge is configured, None otherwise
        """
        if not self._arena_bridge or not HAS_ARENA_BRIDGE:
            logger.warning(
                "arena_bridge_not_configured",
                task_id=task.task_id if hasattr(task, "task_id") else "unknown",
            )
            return None

        with create_span(
            "control_plane.execute_deliberation",
            {
                "task_id": task.task_id,
                "question_preview": task.question[:100] if hasattr(task, "question") else "",
                "agent_count": len(agents) if agents else 0,
            },
        ) as span:
            start = time.monotonic()

            # If no agents provided, select from registry
            if not agents:
                capabilities = (
                    task.required_capabilities
                    if hasattr(task, "required_capabilities")
                    else ["debate"]
                )
                selected_agents: list[Any] = []
                for _ in range(task.sla.min_agents if hasattr(task, "sla") else 2):
                    agent_info = await self.select_agent(
                        capabilities=capabilities,
                        exclude=[a.name if hasattr(a, "name") else str(a) for a in selected_agents],
                    )
                    if agent_info:
                        # We need to convert AgentInfo to actual Agent instances
                        # This is a bridge - the caller should provide agents
                        pass
                logger.warning(
                    "deliberation_no_agents",
                    task_id=task.task_id,
                    msg="No agents provided and auto-selection not yet implemented",
                )

            try:
                outcome = await self._arena_bridge.execute_via_arena(
                    task=task,
                    agents=agents or [],
                    workspace_id=workspace_id or self._config.km_workspace_id,
                )

                latency_ms = (time.monotonic() - start) * 1000
                add_span_attributes(
                    span,
                    {
                        "success": outcome.success,
                        "consensus_reached": outcome.consensus_reached,
                        "latency_ms": latency_ms,
                    },
                )

                logger.info(
                    "deliberation_completed",
                    task_id=task.task_id,
                    success=outcome.success,
                    consensus_reached=outcome.consensus_reached,
                    duration_seconds=outcome.duration_seconds,
                    sla_compliant=outcome.sla_compliant,
                )

                return outcome

            except Exception as e:
                latency_ms = (time.monotonic() - start) * 1000
                add_span_attributes(span, {"error": str(e), "latency_ms": latency_ms})
                logger.error(
                    "deliberation_failed",
                    task_id=task.task_id,
                    error=str(e),
                )
                raise


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
