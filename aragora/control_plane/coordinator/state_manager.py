"""
State Manager for Control Plane Coordinator.

Handles state tracking, persistence, and coordination between
the AgentRegistry, HealthMonitor, and Knowledge Mound adapter.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
)

from aragora.control_plane.health import HealthCheck, HealthMonitor, HealthStatus
from aragora.control_plane.registry import (
    AgentCapability,
    AgentInfo,
    AgentRegistry,
    AgentStatus,
)

from aragora.observability import (
    get_logger,
    create_span,
    add_span_attributes,
)

from aragora.resilience.retry import (
    PROVIDER_RETRY_POLICIES,
    with_retry,
)

if TYPE_CHECKING:
    from aragora.knowledge.mound.adapters.control_plane_adapter import (
        ControlPlaneAdapter,
    )

# Optional KM integration
HAS_KM_ADAPTER = False
try:
    from aragora.knowledge.mound.adapters.control_plane_adapter import TaskOutcome

    HAS_KM_ADAPTER = True
except ImportError:
    TaskOutcome = None

# Optional Watchdog support (Gastown three-tier monitoring)
ThreeTierWatchdog: Any = None
WatchdogConfig: Any = None
WatchdogTier: Any = None
WatchdogIssue: Any = None
get_watchdog: Any = None
try:
    from aragora.control_plane.watchdog import (
        ThreeTierWatchdog,
        WatchdogConfig,
        WatchdogTier,
        WatchdogIssue,
        get_watchdog,
    )

    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False

# Optional AgentFactory for auto-creating agents from registry
AgentFactory: Any = None
get_agent_factory: Any = None
try:
    from aragora.control_plane.agent_factory import (
        AgentFactory,
        get_agent_factory,
    )

    HAS_AGENT_FACTORY = True
except ImportError:
    HAS_AGENT_FACTORY = False

# Optional Redis HA support
RedisHASettings: Any = None
get_redis_ha_config: Any = None
try:
    from aragora.config.redis import RedisHASettings, get_redis_ha_config

    HAS_REDIS_HA = True
except ImportError:
    HAS_REDIS_HA = False

logger = get_logger(__name__)

# Retry configuration for control plane operations
_CP_RETRY_CONFIG = PROVIDER_RETRY_POLICIES["control_plane"]


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
    policy_sync_workspace: str | None = None

    # Redis HA configuration
    redis_ha_enabled: bool = False
    redis_ha_mode: str = "standalone"
    redis_ha_settings: Optional["RedisHASettings"] = None

    # Three-tier watchdog (Gastown pattern)
    enable_watchdog: bool = True
    watchdog_heartbeat_timeout: float = 30.0
    watchdog_check_interval: float = 5.0
    watchdog_auto_escalate: bool = True

    @classmethod
    def from_env(cls) -> "ControlPlaneConfig":
        """Create config from environment variables.

        This method checks for Redis HA configuration and uses the HA client
        if configured. The Redis HA module provides Sentinel and Cluster support
        for high availability deployments.

        Environment Variables:
            REDIS_URL: Standard Redis URL (fallback)
            ARAGORA_REDIS_MODE: Redis mode (standalone, sentinel, cluster)
            ARAGORA_REDIS_SENTINEL_HOSTS: Comma-separated Sentinel hosts
            ARAGORA_REDIS_CLUSTER_NODES: Comma-separated Cluster nodes

        See docs/ENVIRONMENT.md for full Redis HA configuration reference.
        """
        # Check for Redis HA configuration
        redis_ha_enabled = False
        redis_ha_mode = "standalone"
        redis_ha_settings = None
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")

        if HAS_REDIS_HA and get_redis_ha_config is not None:
            try:
                ha_config = get_redis_ha_config()
                if ha_config.is_configured or ha_config.enabled:
                    redis_ha_enabled = True
                    redis_ha_mode = ha_config.mode.value
                    redis_ha_settings = ha_config

                    # Use the URL from HA config if available
                    if ha_config.url:
                        redis_url = ha_config.url
                    elif ha_config.mode.value == "standalone":
                        redis_url = f"redis://{ha_config.host}:{ha_config.port}/{ha_config.db}"

                    logger.info(
                        "redis_ha_config_loaded",
                        mode=redis_ha_mode,
                        enabled=redis_ha_enabled,
                    )
            except (AttributeError, ValueError, KeyError, OSError) as e:
                logger.debug(f"Redis HA config not available: {e}")

        return cls(
            redis_url=redis_url,
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
            # Support both ARAGORA_POLICY_SYNC_ON_STARTUP and CP_ENABLE_POLICY_SYNC for backward compat
            enable_policy_sync=os.environ.get(
                "ARAGORA_POLICY_SYNC_ON_STARTUP",
                os.environ.get("CP_ENABLE_POLICY_SYNC", "true"),
            ).lower()
            == "true",
            policy_sync_workspace=os.environ.get("CP_POLICY_SYNC_WORKSPACE") or None,
            # Redis HA settings
            redis_ha_enabled=redis_ha_enabled,
            redis_ha_mode=redis_ha_mode,
            redis_ha_settings=redis_ha_settings,
            # Watchdog settings
            enable_watchdog=os.environ.get("CP_ENABLE_WATCHDOG", "true").lower() == "true",
            watchdog_heartbeat_timeout=float(os.environ.get("CP_WATCHDOG_HEARTBEAT_TIMEOUT", "30")),
            watchdog_check_interval=float(os.environ.get("CP_WATCHDOG_CHECK_INTERVAL", "5")),
            watchdog_auto_escalate=os.environ.get("CP_WATCHDOG_AUTO_ESCALATE", "true").lower()
            == "true",
        )


class StateManager:
    """
    Manages state tracking and persistence for the control plane.

    Coordinates between:
    - AgentRegistry: Service discovery and agent management
    - HealthMonitor: Health tracking and circuit breakers
    - KnowledgeMound: Historical data and recommendations

    This class handles agent registration, heartbeats, agent selection,
    and Knowledge Mound integration.
    """

    def __init__(
        self,
        config: ControlPlaneConfig,
        registry: AgentRegistry | None = None,
        health_monitor: HealthMonitor | None = None,
        km_adapter: Optional["ControlPlaneAdapter"] = None,
        knowledge_mound: Any | None = None,
        stream_server: Any | None = None,
    ):
        """
        Initialize the state manager.

        Args:
            config: Control plane configuration
            registry: Optional pre-configured AgentRegistry
            health_monitor: Optional pre-configured HealthMonitor
            km_adapter: Optional pre-configured ControlPlaneAdapter
            knowledge_mound: Optional KnowledgeMound for auto-creating adapter
            stream_server: Optional ControlPlaneStreamServer for event broadcasting
        """
        self._config = config
        self._stream_server = stream_server

        self._registry = registry or AgentRegistry(
            redis_url=self._config.redis_url,
            key_prefix=f"{self._config.key_prefix}agents:",
            heartbeat_timeout=self._config.heartbeat_timeout,
            cleanup_interval=self._config.cleanup_interval,
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
                # Will be set up by coordinator once it's initialized
                pass

        # Three-tier watchdog integration (Gastown pattern)
        self._watchdog: Optional["ThreeTierWatchdog"] = None
        if self._config.enable_watchdog and HAS_WATCHDOG:
            self._watchdog = get_watchdog()

            # Configure watchdog tiers based on control plane config
            self._watchdog.configure_tier(
                WatchdogConfig(
                    tier=WatchdogTier.MECHANICAL,
                    heartbeat_timeout_seconds=self._config.watchdog_heartbeat_timeout,
                    check_interval_seconds=self._config.watchdog_check_interval,
                    auto_escalate=self._config.watchdog_auto_escalate,
                )
            )

            logger.info(
                "watchdog_initialized",
                heartbeat_timeout=self._config.watchdog_heartbeat_timeout,
                check_interval=self._config.watchdog_check_interval,
                auto_escalate=self._config.watchdog_auto_escalate,
            )

        # Agent factory for auto-creating agents from registry selection
        self._agent_factory: Optional["AgentFactory"] = None
        if HAS_AGENT_FACTORY:
            self._agent_factory = get_agent_factory()

        self._connected = False

    @property
    def registry(self) -> AgentRegistry:
        """Get the agent registry."""
        return self._registry

    @property
    def health_monitor(self) -> HealthMonitor:
        """Get the health monitor."""
        return self._health_monitor

    @property
    def km_adapter(self) -> Optional["ControlPlaneAdapter"]:
        """Get the Knowledge Mound adapter if configured."""
        return self._km_adapter

    @property
    def watchdog(self) -> Optional["ThreeTierWatchdog"]:
        """Get the Three-Tier Watchdog if configured."""
        return self._watchdog

    @property
    def agent_factory(self) -> Optional["AgentFactory"]:
        """Get the agent factory if configured."""
        return self._agent_factory

    @property
    def config(self) -> ControlPlaneConfig:
        """Get the configuration."""
        return self._config

    @property
    def connected(self) -> bool:
        """Check if state manager is connected."""
        return self._connected

    def set_km_adapter(self, adapter: "ControlPlaneAdapter") -> None:
        """Set the Knowledge Mound adapter."""
        self._km_adapter = adapter

    def register_watchdog_handler(
        self,
        tier: "WatchdogTier",
        handler: Callable[["WatchdogIssue"], Any],
    ) -> None:
        """Register a handler for watchdog issues."""
        if self._watchdog:
            self._watchdog.register_handler(tier, handler)

    @with_retry(_CP_RETRY_CONFIG)
    async def connect(self) -> None:
        """Connect to Redis and start background services."""
        if self._connected:
            return

        with create_span(
            "state_manager.connect",
            {
                "redis_url": self._config.redis_url,
                "redis_ha_mode": self._config.redis_ha_mode,
            },
        ) as span:
            start = time.monotonic()

            await self._registry.connect()
            await self._health_monitor.start()

            # Start three-tier watchdog monitoring
            if self._watchdog:
                await self._watchdog.start()
                # Register existing agents with watchdog
                agents = await self._registry.list_all()
                for agent in agents:
                    self._watchdog.register_agent(agent.agent_id)
                logger.debug(
                    "watchdog_started",
                    monitored_agents=len(agents),
                )

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

            logger.info(
                "state_manager_connected",
                latency_ms=latency_ms,
                redis_ha_enabled=self._config.redis_ha_enabled,
            )

    async def shutdown(self) -> None:
        """Shutdown the state manager."""
        if not self._connected:
            return

        with create_span("state_manager.shutdown") as span:
            start = time.monotonic()

            # Stop watchdog first
            if self._watchdog:
                await self._watchdog.stop()

            await self._health_monitor.stop()
            await self._registry.close()

            self._connected = False
            latency_ms = (time.monotonic() - start) * 1000
            add_span_attributes(span, {"latency_ms": latency_ms})
            logger.info("state_manager_shutdown", latency_ms=latency_ms)

    # =========================================================================
    # Agent Operations
    # =========================================================================

    @with_retry(_CP_RETRY_CONFIG)
    async def register_agent(
        self,
        agent_id: str,
        capabilities: list[str | AgentCapability],
        model: str = "unknown",
        provider: str = "unknown",
        metadata: Optional[dict[str, Any]] = None,
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
            "state_manager.register_agent",
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

            # Register with watchdog for multi-tier monitoring
            if self._watchdog:
                self._watchdog.register_agent(agent_id)

            logger.info(
                "agent_registered",
                agent_id=agent_id,
                model=model,
                provider=provider,
                capabilities=[str(c) for c in capabilities],
            )
            return agent

    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the control plane."""
        self._health_monitor.unregister_probe(agent_id)

        # Unregister from watchdog
        if self._watchdog:
            self._watchdog.unregister_agent(agent_id)

        return await self._registry.unregister(agent_id)

    async def heartbeat(
        self,
        agent_id: str,
        status: AgentStatus | None = None,
        current_task_id: str | None = None,
    ) -> bool:
        """Send agent heartbeat."""
        result = await self._registry.heartbeat(agent_id, status, current_task_id)

        # Record heartbeat with watchdog for three-tier monitoring
        if result and self._watchdog:
            self._watchdog.record_heartbeat(agent_id)

        return result

    async def get_agent(self, agent_id: str) -> AgentInfo | None:
        """Get agent information."""
        return await self._registry.get(agent_id)

    async def list_agents(
        self,
        capability: str | AgentCapability | None = None,
        only_available: bool = True,
    ) -> list[AgentInfo]:
        """List registered agents."""
        if capability:
            return await self._registry.find_by_capability(
                capability, only_available=only_available
            )
        return await self._registry.list_all(include_offline=not only_available)

    async def select_agent(
        self,
        capabilities: list[str | AgentCapability],
        strategy: str = "least_loaded",
        exclude: Optional[list[str]] = None,
        task_type: str | None = None,
        use_km_recommendations: bool = True,
    ) -> AgentInfo | None:
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
        capabilities: list[str | AgentCapability],
        task_type: str,
        exclude: Optional[list[str]] = None,
    ) -> AgentInfo | None:
        """Select an agent using KM-based historical recommendations."""
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

        except (KeyError, AttributeError, TypeError) as e:
            logger.debug(f"KM recommendation failed, using fallback: {e}")

        # Fallback to registry selection
        return await self._registry.select_agent(
            capabilities=capabilities,
            strategy="least_loaded",
            exclude=exclude,
        )

    async def record_task_completion(
        self,
        agent_id: str,
        success: bool,
        latency_ms: float,
    ) -> None:
        """Record task completion metrics for an agent."""
        await self._registry.record_task_completion(
            agent_id,
            success=success,
            latency_ms=latency_ms,
        )

    # =========================================================================
    # Health Operations
    # =========================================================================

    def get_agent_health(self, agent_id: str) -> HealthCheck | None:
        """Get health status for an agent."""
        return self._health_monitor.get_agent_health(agent_id)

    def get_system_health(self) -> HealthStatus:
        """Get overall system health."""
        return self._health_monitor.get_system_health()

    def is_agent_available(self, agent_id: str) -> bool:
        """Check if an agent is available."""
        return self._health_monitor.is_agent_available(agent_id)

    # =========================================================================
    # Watchdog Operations
    # =========================================================================

    def record_request(
        self,
        agent_id: str,
        success: bool,
        latency_ms: float,
    ) -> None:
        """Record a request to an agent for watchdog monitoring."""
        if self._watchdog:
            self._watchdog.record_request(agent_id, success, latency_ms)

    # =========================================================================
    # Knowledge Mound Operations
    # =========================================================================

    async def get_agent_recommendations(
        self,
        capability: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get agent recommendations from Knowledge Mound."""
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
        except (AttributeError, TypeError, ZeroDivisionError) as e:
            logger.debug(f"Failed to get agent recommendations: {e}")
            return []

    async def get_agent_recommendations_from_km(
        self,
        task_type: str,
        capabilities: list[str],
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get agent recommendations from Knowledge Mound for a task type."""
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
        except (KeyError, AttributeError, TypeError) as e:
            logger.debug(f"Failed to get KM recommendations: {e}")
            return []

    async def store_task_outcome(
        self,
        task_id: str,
        task_type: str,
        agent_id: str,
        success: bool,
        duration_seconds: float,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a task outcome in Knowledge Mound."""
        if not self._km_adapter or not HAS_KM_ADAPTER:
            return

        try:
            outcome = TaskOutcome(
                task_id=task_id,
                task_type=task_type,
                agent_id=agent_id,
                success=success,
                duration_seconds=duration_seconds,
                workspace_id=self._config.km_workspace_id,
                error_message=error_message,
                metadata=metadata or {},
            )
            await self._km_adapter.store_task_outcome(outcome)
        except (AttributeError, TypeError, KeyError) as e:
            logger.debug("km_store_failed", error=str(e), task_id=task_id)

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_stats(self) -> dict[str, Any]:
        """Get state manager statistics."""
        stats = {
            "registry": await self._registry.get_stats(),
            "health": self._health_monitor.get_stats(),
        }

        # Add KM adapter stats if available
        if self._km_adapter:
            stats["knowledge_mound"] = self._km_adapter.get_stats()

        # Add watchdog stats if available
        if self._watchdog and HAS_WATCHDOG:
            stats["watchdog"] = self._watchdog.get_stats()

        return stats
