"""
Control Plane Coordinator Core.

Provides a unified high-level API for the control plane, coordinating
between the StateManager, PolicyEnforcer, and SchedulerBridge.

This is the main entry point for control plane operations.
"""

from __future__ import annotations

import time
from typing import (
    TYPE_CHECKING,
    Any,
)
from collections.abc import Callable

from aragora.control_plane.health import HealthCheck, HealthStatus
from aragora.control_plane.registry import (
    AgentCapability,
    AgentInfo,
    AgentStatus,
)
from aragora.control_plane.scheduler import Task, TaskPriority

from aragora.observability import (
    get_logger,
    create_span,
    add_span_attributes,
)

from aragora.resilience.retry import (
    PROVIDER_RETRY_POLICIES,
    with_retry,
)

from aragora.control_plane.coordinator.state_manager import (
    ControlPlaneConfig,
    StateManager,
)
from aragora.control_plane.coordinator.policy_enforcer import PolicyEnforcer
from aragora.control_plane.coordinator.scheduler_bridge import SchedulerBridge

if TYPE_CHECKING:
    import asyncio

    from aragora.control_plane.policy import ControlPlanePolicyManager
    from aragora.control_plane.registry import AgentRegistry
    from aragora.control_plane.scheduler import TaskScheduler
    from aragora.control_plane.health import HealthMonitor
    from aragora.knowledge.mound.adapters.control_plane_adapter import ControlPlaneAdapter
    from aragora.control_plane.watchdog import ThreeTierWatchdog, WatchdogIssue
    from aragora.control_plane.agent_factory import AgentFactory

# Optional Arena Bridge
ArenaControlPlaneBridge: Any = None
DeliberationTask: Any = None
DeliberationOutcome: Any = None
DELIBERATION_TASK_TYPE = "deliberation"
try:
    from aragora.control_plane.arena_bridge import ArenaControlPlaneBridge
    from aragora.control_plane.deliberation import (
        DELIBERATION_TASK_TYPE as _DELIBERATION_TASK_TYPE,
        DeliberationOutcome,
        DeliberationTask,
    )

    DELIBERATION_TASK_TYPE = _DELIBERATION_TASK_TYPE  # Use real value if available

    HAS_ARENA_BRIDGE = True
except ImportError:
    HAS_ARENA_BRIDGE = False

# Optional Watchdog
HAS_WATCHDOG = False
IssueSeverityType: Any = None
try:
    from aragora.control_plane.watchdog import IssueSeverity as IssueSeverityType

    HAS_WATCHDOG = True
except ImportError:
    pass

logger = get_logger(__name__)

# Retry configuration for control plane operations
_CP_RETRY_CONFIG = PROVIDER_RETRY_POLICIES["control_plane"]


class ControlPlaneCoordinator:
    """
    Unified coordinator for the Aragora control plane.

    Provides high-level operations that coordinate between:
    - StateManager: Agent registry, health monitoring, and KM integration
    - PolicyEnforcer: Policy evaluation and enforcement
    - SchedulerBridge: Task distribution and lifecycle

    This class uses composition to delegate to specialized submodules.

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
        config: ControlPlaneConfig | None = None,
        # New modular API
        state_manager: StateManager | None = None,
        policy_enforcer: PolicyEnforcer | None = None,
        scheduler_bridge: SchedulerBridge | None = None,
        # Backward compatibility - old API (deprecated but still supported)
        registry: Any | None = None,
        scheduler: Any | None = None,
        health_monitor: Any | None = None,
        # Common parameters
        km_adapter: ControlPlaneAdapter | None = None,
        knowledge_mound: Any | None = None,
        arena_bridge: ArenaControlPlaneBridge | None = None,
        stream_server: Any | None = None,
        shared_state: Any | None = None,
        policy_manager: ControlPlanePolicyManager | None = None,
    ):
        """
        Initialize the coordinator.

        Args:
            config: Control plane configuration
            state_manager: Optional pre-configured StateManager
            policy_enforcer: Optional pre-configured PolicyEnforcer
            scheduler_bridge: Optional pre-configured SchedulerBridge
            registry: (Deprecated) Optional pre-configured AgentRegistry
            scheduler: (Deprecated) Optional pre-configured TaskScheduler
            health_monitor: (Deprecated) Optional pre-configured HealthMonitor
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

        # Initialize policy enforcer
        self._policy_enforcer = policy_enforcer or PolicyEnforcer(
            policy_manager=policy_manager,
            violation_callback=self._handle_policy_violation,
            enable_policy_sync=self._config.enable_policy_sync,
            policy_sync_workspace=self._config.policy_sync_workspace,
        )

        # Initialize state manager (with backward compatibility for registry/health_monitor)
        self._state_manager = state_manager or StateManager(
            config=self._config,
            registry=registry,
            health_monitor=health_monitor,
            km_adapter=km_adapter,
            knowledge_mound=knowledge_mound,
            stream_server=stream_server,
        )

        # Set up KM adapter if knowledge_mound provided
        if knowledge_mound and not km_adapter:
            try:
                from aragora.knowledge.mound.adapters.control_plane_adapter import (
                    ControlPlaneAdapter,
                )

                adapter = ControlPlaneAdapter(
                    coordinator=self,
                    knowledge_mound=knowledge_mound,
                    workspace_id=self._config.km_workspace_id,
                )
                self._state_manager.set_km_adapter(adapter)
            except ImportError:
                pass

        # Initialize scheduler bridge (with backward compatibility for scheduler param)
        self._scheduler_bridge = scheduler_bridge or SchedulerBridge(
            config=self._config,
            state_manager=self._state_manager,
            policy_enforcer=self._policy_enforcer,
            scheduler=scheduler,  # Backward compatibility
        )

        # Register watchdog handlers if available
        if self._state_manager.watchdog and HAS_WATCHDOG:
            from aragora.control_plane.watchdog import WatchdogTier

            self._state_manager.register_watchdog_handler(
                WatchdogTier.MECHANICAL,
                self._handle_watchdog_issue,
            )
            self._state_manager.register_watchdog_handler(
                WatchdogTier.BOOT_AGENT,
                self._handle_watchdog_issue,
            )
            self._state_manager.register_watchdog_handler(
                WatchdogTier.DEACON,
                self._handle_watchdog_issue,
            )

        # Arena Bridge integration for unified debate execution
        self._arena_bridge: ArenaControlPlaneBridge | None = None
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

    def _handle_policy_violation(self, violation: Any) -> None:
        """Handle policy violations - delegate to policy enforcer."""
        # The policy enforcer already handles this internally
        pass

    async def _handle_watchdog_issue(self, issue: WatchdogIssue) -> None:
        """Handle watchdog issues with control plane actions.

        This method is called by the ThreeTierWatchdog when issues are detected
        across any tier. It triggers appropriate control plane responses:

        - CRITICAL issues: May trigger agent quarantine or circuit breaker
        - ERROR issues: Logged and may affect scheduling priority
        - WARNING issues: Logged for monitoring
        """
        if not HAS_WATCHDOG:
            return

        try:
            # Log issue with structured attributes
            logger.info(
                "watchdog_issue_detected",
                issue_id=issue.id,
                severity=issue.severity.name,
                category=issue.category.value,
                agent=issue.agent,
                issue_message=issue.message,
                detected_by=issue.detected_by.value if issue.detected_by else None,
            )

            # Take action based on severity
            if issue.severity >= IssueSeverityType.CRITICAL:
                # Critical issues may trigger agent status change
                if issue.agent:
                    # Update health status
                    agent_info = await self._state_manager.get_agent(issue.agent)
                    if agent_info and agent_info.status != AgentStatus.FAILED:
                        logger.warning(
                            "watchdog_critical_agent",
                            agent=issue.agent,
                            issue_category=issue.category.value,
                            issue_message=issue.message,
                        )
                        # Circuit breaker may already be handling this via health monitor

            # Broadcast issue event if stream server available
            if self._stream_server:
                try:
                    await self._stream_server.broadcast_event(
                        {
                            "type": "watchdog_issue",
                            "issue": issue.to_dict(),
                        }
                    )
                except (ConnectionError, OSError, AttributeError) as e:
                    logger.debug(f"Failed to broadcast watchdog issue: {e}")

            # Record in KM if adapter available
            km_adapter = self._state_manager.km_adapter
            if km_adapter and issue.severity >= IssueSeverityType.ERROR:
                try:
                    # KM adapter can track operational incidents
                    pass  # KM tracking is optional - no-op if not needed
                except (ConnectionError, OSError, AttributeError) as e:
                    logger.debug(f"Failed to record watchdog issue in KM: {e}")

        except (RuntimeError, ValueError, KeyError) as e:
            logger.warning(f"Error handling watchdog issue: {e}")

    @classmethod
    async def create(
        cls,
        config: ControlPlaneConfig | None = None,
    ) -> ControlPlaneCoordinator:
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

    @with_retry(_CP_RETRY_CONFIG)
    async def connect(self) -> None:
        """Connect to Redis and start background services."""
        if self._connected:
            return

        with create_span(
            "control_plane.connect",
            {
                "redis_url": self._config.redis_url,
                "redis_ha_mode": self._config.redis_ha_mode,
            },
        ) as span:
            start = time.monotonic()

            # Connect state manager (registry, health monitor, watchdog)
            await self._state_manager.connect()

            # Connect scheduler
            await self._scheduler_bridge.connect()

            # Sync policies from compliance store
            self._sync_policies_from_store()

            self._connected = True
            latency_ms = (time.monotonic() - start) * 1000
            add_span_attributes(
                span,
                {
                    "latency_ms": latency_ms,
                    "success": True,
                    "redis_ha_enabled": self._config.redis_ha_enabled,
                },
            )

            # Log Redis HA status
            if self._config.redis_ha_enabled:
                ha_mode = self._config.redis_ha_mode
                logger.info(
                    "control_plane_connected",
                    latency_ms=latency_ms,
                    redis_mode=ha_mode,
                    redis_ha_enabled=True,
                )
            else:
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

            await self._state_manager.shutdown()
            await self._scheduler_bridge.close()

            self._connected = False
            latency_ms = (time.monotonic() - start) * 1000
            add_span_attributes(span, {"latency_ms": latency_ms})
            logger.info("control_plane_shutdown", latency_ms=latency_ms)

    # =========================================================================
    # Agent Operations (delegated to StateManager)
    # =========================================================================

    async def register_agent(
        self,
        agent_id: str,
        capabilities: list[str | AgentCapability],
        model: str = "unknown",
        provider: str = "unknown",
        metadata: dict[str, Any] | None = None,
        health_probe: Callable[[], bool] | None = None,
    ) -> AgentInfo:
        """Register an agent with the control plane."""
        return await self._state_manager.register_agent(
            agent_id=agent_id,
            capabilities=capabilities,
            model=model,
            provider=provider,
            metadata=metadata,
            health_probe=health_probe,
        )

    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the control plane."""
        return await self._state_manager.unregister_agent(agent_id)

    async def heartbeat(
        self,
        agent_id: str,
        status: AgentStatus | None = None,
    ) -> bool:
        """Send agent heartbeat."""
        return await self._state_manager.heartbeat(agent_id, status)

    async def get_agent(self, agent_id: str) -> AgentInfo | None:
        """Get agent information."""
        return await self._state_manager.get_agent(agent_id)

    async def list_agents(
        self,
        capability: str | AgentCapability | None = None,
        only_available: bool = True,
    ) -> list[AgentInfo]:
        """List registered agents."""
        return await self._state_manager.list_agents(capability, only_available)

    async def select_agent(
        self,
        capabilities: list[str | AgentCapability],
        strategy: str = "least_loaded",
        exclude: list[str] | None = None,
        task_type: str | None = None,
        use_km_recommendations: bool = True,
    ) -> AgentInfo | None:
        """Select an agent for a task."""
        return await self._state_manager.select_agent(
            capabilities=capabilities,
            strategy=strategy,
            exclude=exclude,
            task_type=task_type,
            use_km_recommendations=use_km_recommendations,
        )

    async def get_agent_recommendations_from_km(
        self,
        task_type: str,
        capabilities: list[str],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get agent recommendations from Knowledge Mound."""
        return await self._state_manager.get_agent_recommendations_from_km(
            task_type=task_type,
            capabilities=capabilities,
            limit=limit,
        )

    # =========================================================================
    # Task Operations (delegated to SchedulerBridge)
    # =========================================================================

    async def submit_task(
        self,
        task_type: str,
        payload: dict[str, Any],
        required_capabilities: list[str] | None = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
        workspace_id: str | None = None,
    ) -> str:
        """Submit a task for execution."""
        return await self._scheduler_bridge.submit_task(
            task_type=task_type,
            payload=payload,
            required_capabilities=required_capabilities,
            priority=priority,
            timeout_seconds=timeout_seconds,
            metadata=metadata,
            workspace_id=workspace_id,
        )

    async def claim_task(
        self,
        agent_id: str,
        capabilities: list[str],
        block_ms: int = 5000,
        agent_region: str | None = None,
        workspace_id: str | None = None,
    ) -> Task | None:
        """Claim a task for an agent."""
        return await self._scheduler_bridge.claim_task(
            agent_id=agent_id,
            capabilities=capabilities,
            block_ms=block_ms,
            agent_region=agent_region,
            workspace_id=workspace_id,
        )

    async def complete_task(
        self,
        task_id: str,
        result: dict[str, Any] | None = None,
        agent_id: str | None = None,
        latency_ms: float | None = None,
        sla_policy_id: str | None = None,
    ) -> bool:
        """Mark a task as completed."""
        return await self._scheduler_bridge.complete_task(
            task_id=task_id,
            result=result,
            agent_id=agent_id,
            latency_ms=latency_ms,
            sla_policy_id=sla_policy_id,
        )

    async def fail_task(
        self,
        task_id: str,
        error: str,
        agent_id: str | None = None,
        latency_ms: float | None = None,
        requeue: bool = True,
    ) -> bool:
        """Mark a task as failed."""
        return await self._scheduler_bridge.fail_task(
            task_id=task_id,
            error=error,
            agent_id=agent_id,
            latency_ms=latency_ms,
            requeue=requeue,
        )

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        return await self._scheduler_bridge.cancel_task(task_id)

    async def get_task(self, task_id: str) -> Task | None:
        """Get task by ID."""
        return await self._scheduler_bridge.get_task(task_id)

    async def wait_for_result(
        self,
        task_id: str,
        timeout: float | None = None,
    ) -> Task | None:
        """Wait for a task to complete."""
        return await self._scheduler_bridge.wait_for_result(task_id, timeout)

    # =========================================================================
    # Health Operations (delegated to StateManager)
    # =========================================================================

    def get_agent_health(self, agent_id: str) -> HealthCheck | None:
        """Get health status for an agent."""
        return self._state_manager.get_agent_health(agent_id)

    def get_system_health(self) -> HealthStatus:
        """Get overall system health."""
        return self._state_manager.get_system_health()

    def is_agent_available(self, agent_id: str) -> bool:
        """Check if an agent is available."""
        return self._state_manager.is_agent_available(agent_id)

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_stats(self) -> dict[str, Any]:
        """Get comprehensive control plane statistics."""
        stats = {
            **await self._state_manager.get_stats(),
            "scheduler": await self._scheduler_bridge.get_stats(),
            "config": {
                "redis_url": self._config.redis_url,
                "heartbeat_timeout": self._config.heartbeat_timeout,
                "task_timeout": self._config.task_timeout,
            },
        }

        # Add policy manager stats if available
        if self._policy_enforcer.policy_manager:
            stats["policy"] = self._policy_enforcer.get_metrics()

        return stats

    # =========================================================================
    # Watchdog Integration
    # =========================================================================

    @property
    def watchdog(self) -> ThreeTierWatchdog | None:
        """Get the Three-Tier Watchdog if configured."""
        return self._state_manager.watchdog

    def record_request(
        self,
        agent_id: str,
        success: bool,
        latency_ms: float,
    ) -> None:
        """Record a request to an agent for watchdog monitoring."""
        self._state_manager.record_request(agent_id, success, latency_ms)

    # =========================================================================
    # Policy Manager Integration
    # =========================================================================

    @property
    def policy_manager(self) -> ControlPlanePolicyManager | None:
        """Get the Policy Manager if configured."""
        return self._policy_enforcer.policy_manager

    def set_policy_manager(self, manager: ControlPlanePolicyManager) -> None:
        """Set the Policy Manager."""
        self._policy_enforcer.set_policy_manager(manager)
        # Also update the scheduler's policy manager
        self._scheduler_bridge.update_policy_manager(manager)

    def _sync_policies_from_store(self) -> int:
        """Backward compatibility: sync policies from compliance store.

        Delegates to the policy enforcer.
        """
        return self._policy_enforcer.sync_policies_from_store()

    def _should_sync_policies_from_store(self) -> bool:
        """Backward compatibility: check if policy sync should occur.

        Delegates to the policy enforcer.
        """
        return self._policy_enforcer._should_sync_policies_from_store()

    # =========================================================================
    # Knowledge Mound Integration
    # =========================================================================

    @property
    def km_adapter(self) -> ControlPlaneAdapter | None:
        """Get the Knowledge Mound adapter if configured."""
        return self._state_manager.km_adapter

    def set_km_adapter(self, adapter: ControlPlaneAdapter) -> None:
        """Set the Knowledge Mound adapter."""
        self._state_manager.set_km_adapter(adapter)

    async def get_agent_recommendations(
        self,
        capability: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get agent recommendations from Knowledge Mound."""
        return await self._state_manager.get_agent_recommendations(capability, limit)

    # =========================================================================
    # Arena Bridge Integration
    # =========================================================================

    @property
    def arena_bridge(self) -> ArenaControlPlaneBridge | None:
        """Get the Arena Bridge if configured."""
        return self._arena_bridge

    def set_arena_bridge(self, bridge: ArenaControlPlaneBridge) -> None:
        """Set the Arena Bridge."""
        self._arena_bridge = bridge

    async def execute_deliberation(
        self,
        task: DeliberationTask,
        agents: list[Any] | None = None,
        workspace_id: str | None = None,
    ) -> DeliberationOutcome | None:
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

            # If no agents provided, select from registry and convert to Agent instances
            if not agents:
                # Get required_capabilities from task, defaulting to ["debate"]
                raw_capabilities: list[str] = getattr(task, "required_capabilities", ["debate"])
                capabilities: list[str | AgentCapability] = list(raw_capabilities)
                selected_infos: list[AgentInfo] = []
                min_agents = task.sla.min_agents if hasattr(task, "sla") else 2

                for _ in range(min_agents):
                    agent_info = await self.select_agent(
                        capabilities=capabilities,
                        exclude=[a.agent_id for a in selected_infos],
                    )
                    if agent_info:
                        selected_infos.append(agent_info)

                # Convert AgentInfo -> concrete Agent instances via factory
                agent_factory = self._state_manager.agent_factory
                if selected_infos and agent_factory:
                    try:
                        agents = await agent_factory.create_agents(
                            selected_infos,
                            role="proposer",
                            min_agents=0,  # Don't raise, log warning instead
                        )
                    except (RuntimeError, ValueError, KeyError) as e:
                        logger.warning(
                            "deliberation_agent_creation_failed",
                            task_id=task.task_id,
                            error=str(e),
                            selected_count=len(selected_infos),
                        )

                if not agents:
                    logger.warning(
                        "deliberation_no_agents",
                        task_id=task.task_id,
                        msg="No agents could be created from registry selection",
                        selected_count=len(selected_infos),
                        factory_available=agent_factory is not None,
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

            except (OSError, ConnectionError, RuntimeError, ValueError) as e:
                latency_ms = (time.monotonic() - start) * 1000
                add_span_attributes(span, {"error": str(e), "latency_ms": latency_ms})
                logger.error(
                    "deliberation_failed",
                    task_id=task.task_id,
                    error=str(e),
                )
                raise

    # =========================================================================
    # Internal Component Access (for backward compatibility)
    # =========================================================================

    @property
    def _registry(self) -> AgentRegistry:
        """Backward compatibility: access to registry."""
        return self._state_manager.registry

    @property
    def _scheduler(self) -> TaskScheduler:
        """Backward compatibility: access to scheduler."""
        return self._scheduler_bridge.scheduler

    @property
    def _health_monitor(self) -> HealthMonitor:
        """Backward compatibility: access to health monitor."""
        return self._state_manager.health_monitor

    @property
    def _km_adapter(self) -> ControlPlaneAdapter | None:
        """Backward compatibility: access to KM adapter."""
        return self._state_manager.km_adapter

    @property
    def _watchdog(self) -> ThreeTierWatchdog | None:
        """Backward compatibility: access to watchdog."""
        return self._state_manager.watchdog

    @property
    def _agent_factory(self) -> AgentFactory | None:
        """Backward compatibility: access to agent factory."""
        return self._state_manager.agent_factory

    @property
    def _result_waiters(self) -> dict[str, asyncio.Event]:
        """Backward compatibility: access to result waiters."""
        return self._scheduler_bridge._result_waiters


async def create_control_plane(
    config: ControlPlaneConfig | None = None,
) -> ControlPlaneCoordinator:
    """
    Convenience function to create a connected control plane.

    Args:
        config: Optional configuration

    Returns:
        Connected ControlPlaneCoordinator
    """
    return await ControlPlaneCoordinator.create(config)
