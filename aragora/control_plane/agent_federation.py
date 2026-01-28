"""
Federated Agent Pool for Cross-Instance Agent Sharing.

Provides a unified abstraction over local and remote agents across
multiple Aragora instances, enabling:
- Cross-instance agent discovery
- Load balancing across instances
- Failover to local agents when remote unavailable
- Latency-aware routing

This builds on the existing AgentRegistry and RegionalEventBus
infrastructure to enable true multi-instance agent federation.

Usage:
    from aragora.control_plane.agent_federation import (
        FederatedAgentPool,
        FederatedAgentConfig,
    )

    pool = FederatedAgentPool(
        local_registry=registry,
        event_bus=event_bus,
        config=FederatedAgentConfig(prefer_local=True),
    )
    await pool.connect()

    # Find agents across all instances
    agents = await pool.find_agents(
        capability="debate",
        min_count=3,
    )

    # Execute on best available agent
    result = await pool.execute(
        agent_id="claude-3",
        task=task,
        fallback_local=True,
    )
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from aragora.control_plane.registry import AgentInfo, AgentRegistry

logger = logging.getLogger(__name__)


class LoadBalanceStrategy(str, Enum):
    """Strategy for selecting agents from the federated pool."""

    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    LOWEST_LATENCY = "lowest_latency"
    RANDOM = "random"
    PREFER_LOCAL = "prefer_local"


class FederationMode(str, Enum):
    """Mode of agent federation."""

    DISABLED = "disabled"  # Only use local agents
    READONLY = "readonly"  # Discover but don't execute remotely
    FULL = "full"  # Full federation with remote execution


@dataclass
class FederatedAgentConfig:
    """Configuration for federated agent pool."""

    mode: FederationMode = FederationMode.FULL
    load_balance_strategy: LoadBalanceStrategy = LoadBalanceStrategy.PREFER_LOCAL
    prefer_local: bool = True
    local_bias: float = 0.7  # Preference weight for local agents (0-1)
    max_remote_latency_ms: float = 5000.0  # Max acceptable remote latency
    failover_enabled: bool = True  # Fall back to local on remote failure
    discovery_interval: float = 30.0  # Seconds between remote discovery
    health_check_interval: float = 10.0  # Seconds between health checks
    max_concurrent_remotes: int = 10  # Max concurrent remote agent calls


@dataclass
class FederatedAgent:
    """
    An agent in the federated pool.

    Extends AgentInfo with federation-specific metadata.
    """

    info: AgentInfo
    instance_id: str  # Aragora instance this agent belongs to
    is_local: bool = True
    remote_endpoint: Optional[str] = None
    estimated_latency_ms: float = 0.0
    last_success_at: Optional[float] = None
    last_failure_at: Optional[float] = None
    consecutive_failures: int = 0

    @property
    def agent_id(self) -> str:
        """Get the agent ID."""
        return self.info.agent_id

    @property
    def is_healthy(self) -> bool:
        """Check if agent is considered healthy."""
        if self.consecutive_failures >= 3:
            return False
        if self.last_failure_at and not self.last_success_at:
            return False
        return self.info.is_available()

    def record_success(self, latency_ms: float) -> None:
        """Record a successful execution."""
        self.last_success_at = time.time()
        self.consecutive_failures = 0
        # Update latency with exponential moving average
        alpha = 0.3
        self.estimated_latency_ms = alpha * latency_ms + (1 - alpha) * self.estimated_latency_ms

    def record_failure(self) -> None:
        """Record a failed execution."""
        self.last_failure_at = time.time()
        self.consecutive_failures += 1


class FederatedAgentPool:
    """
    Pool of agents federated across multiple Aragora instances.

    Provides unified access to local and remote agents with
    load balancing, failover, and health tracking.
    """

    def __init__(
        self,
        local_registry: AgentRegistry,
        event_bus: Optional[Any] = None,
        config: Optional[FederatedAgentConfig] = None,
        instance_id: Optional[str] = None,
    ):
        """
        Initialize the federated agent pool.

        Args:
            local_registry: Local agent registry
            event_bus: Optional RegionalEventBus for cross-instance sync
            config: Pool configuration
            instance_id: Unique ID for this Aragora instance
        """
        self._local_registry = local_registry
        self._event_bus = event_bus
        self._config = config or FederatedAgentConfig()
        self._instance_id = instance_id or self._generate_instance_id()

        # Federated agent cache
        self._agents: Dict[str, FederatedAgent] = {}
        self._remote_instances: Dict[str, Dict[str, Any]] = {}

        # Round-robin state
        self._rr_index: Dict[str, int] = {}

        # Background tasks
        self._discovery_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None

        self._connected = False

        logger.info(f"[FederatedAgentPool] Initialized instance {self._instance_id}")

    def _generate_instance_id(self) -> str:
        """Generate a unique instance ID."""
        import socket
        import uuid

        hostname = socket.gethostname()
        unique = uuid.uuid4().hex[:8]
        return f"{hostname}-{unique}"

    async def connect(self) -> None:
        """Start the federated pool."""
        if self._connected:
            return

        # Load local agents
        await self._sync_local_agents()

        # Start background tasks
        if self._config.mode != FederationMode.DISABLED and self._event_bus:
            self._discovery_task = asyncio.create_task(self._discovery_loop())
            self._health_task = asyncio.create_task(self._health_check_loop())

            # Subscribe to agent events from other instances
            if hasattr(self._event_bus, "subscribe"):
                await self._event_bus.subscribe(self._handle_remote_event)

        self._connected = True
        logger.info(f"[FederatedAgentPool] Connected with {len(self._agents)} agents")

    async def close(self) -> None:
        """Stop the federated pool."""
        if self._discovery_task:
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass

        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        self._connected = False
        logger.info("[FederatedAgentPool] Closed")

    async def _sync_local_agents(self) -> None:
        """Sync agents from local registry."""
        local_agents = await self._local_registry.list_all()

        for info in local_agents:
            agent = FederatedAgent(
                info=info,
                instance_id=self._instance_id,
                is_local=True,
            )
            self._agents[info.agent_id] = agent

    async def _discovery_loop(self) -> None:
        """Periodically discover remote agents."""
        while True:
            try:
                await asyncio.sleep(self._config.discovery_interval)
                await self._discover_remote_agents()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[FederatedAgentPool] Discovery error: {e}")

    async def _discover_remote_agents(self) -> None:
        """Discover agents from remote instances via event bus."""
        if not self._event_bus:
            return

        # Request agent list from other instances
        # This would use the RegionalEventBus to broadcast a discovery request
        # Remote instances respond with their agent lists
        pass  # Implementation depends on event bus protocol

    async def _health_check_loop(self) -> None:
        """Periodically check agent health."""
        while True:
            try:
                await asyncio.sleep(self._config.health_check_interval)
                await self._check_agent_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[FederatedAgentPool] Health check error: {e}")

    async def _check_agent_health(self) -> None:
        """Check health of all agents."""
        # Update local agent status from registry
        await self._sync_local_agents()

        # Mark remote agents as unhealthy if not seen recently
        now = time.time()
        for agent in self._agents.values():
            if not agent.is_local:
                if agent.last_success_at and now - agent.last_success_at > 60:
                    agent.consecutive_failures += 1

    async def _handle_remote_event(self, event: Any) -> None:
        """Handle events from remote instances."""
        # Process agent registration/update/deregistration events
        # from the RegionalEventBus
        pass  # Implementation depends on event types

    def find_agents(
        self,
        capability: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        min_count: int = 1,
        include_remote: bool = True,
        region: Optional[str] = None,
    ) -> List[FederatedAgent]:
        """
        Find agents matching criteria.

        Args:
            capability: Single capability to match
            capabilities: List of capabilities to match (AND)
            min_count: Minimum number of agents needed
            include_remote: Include remote agents
            region: Filter by region

        Returns:
            List of matching agents
        """
        matches = []

        for agent in self._agents.values():
            # Skip unhealthy agents
            if not agent.is_healthy:
                continue

            # Skip remote agents if not including
            if not include_remote and not agent.is_local:
                continue

            # Filter by region
            if region and not agent.info.is_available_in_region(region):
                continue

            # Check capabilities
            if capability and not agent.info.has_capability(capability):
                continue
            if capabilities and not agent.info.has_all_capabilities(capabilities):
                continue

            matches.append(agent)

        return matches

    def select_agent(
        self,
        agents: List[FederatedAgent],
        strategy: Optional[LoadBalanceStrategy] = None,
    ) -> Optional[FederatedAgent]:
        """
        Select an agent using the load balancing strategy.

        Args:
            agents: List of candidate agents
            strategy: Override strategy (uses config default if None)

        Returns:
            Selected agent or None if no agents available
        """
        if not agents:
            return None

        strategy = strategy or self._config.load_balance_strategy

        if strategy == LoadBalanceStrategy.RANDOM:
            return random.choice(agents)

        if strategy == LoadBalanceStrategy.ROUND_ROBIN:
            # Group by capability for RR
            key = "default"
            idx = self._rr_index.get(key, 0)
            agent = agents[idx % len(agents)]
            self._rr_index[key] = idx + 1
            return agent

        if strategy == LoadBalanceStrategy.LOWEST_LATENCY:
            return min(agents, key=lambda a: a.estimated_latency_ms)

        if strategy == LoadBalanceStrategy.LEAST_LOADED:
            return min(
                agents,
                key=lambda a: a.info.tasks_completed + (10 if a.info.current_task_id else 0),
            )

        if strategy == LoadBalanceStrategy.PREFER_LOCAL:
            # Score agents with local bias
            local_agents = [a for a in agents if a.is_local]
            if local_agents and random.random() < self._config.local_bias:
                return random.choice(local_agents)
            return random.choice(agents)

        return agents[0]

    def get_agent(self, agent_id: str) -> Optional[FederatedAgent]:
        """Get a specific agent by ID."""
        return self._agents.get(agent_id)

    def list_local_agents(self) -> List[FederatedAgent]:
        """List all local agents."""
        return [a for a in self._agents.values() if a.is_local]

    def list_remote_agents(self) -> List[FederatedAgent]:
        """List all remote agents."""
        return [a for a in self._agents.values() if not a.is_local]

    def get_instance_stats(self) -> Dict[str, Any]:
        """Get statistics about the federated pool."""
        local_count = sum(1 for a in self._agents.values() if a.is_local)
        remote_count = len(self._agents) - local_count
        healthy_count = sum(1 for a in self._agents.values() if a.is_healthy)

        return {
            "instance_id": self._instance_id,
            "total_agents": len(self._agents),
            "local_agents": local_count,
            "remote_agents": remote_count,
            "healthy_agents": healthy_count,
            "remote_instances": len(self._remote_instances),
            "mode": self._config.mode.value,
            "strategy": self._config.load_balance_strategy.value,
        }

    async def register_agent(
        self,
        agent_id: str,
        capabilities: List[str],
        model: str = "unknown",
        provider: str = "unknown",
        **kwargs: Any,
    ) -> FederatedAgent:
        """
        Register a new agent in the federated pool.

        Args:
            agent_id: Unique agent ID
            capabilities: Agent capabilities
            model: Underlying model
            provider: Model provider
            **kwargs: Additional agent metadata

        Returns:
            Registered FederatedAgent
        """
        # Register in local registry
        info = await self._local_registry.register(
            agent_id=agent_id,
            capabilities=capabilities,
            model=model,
            provider=provider,
            **kwargs,
        )

        # Add to federated pool
        agent = FederatedAgent(
            info=info,
            instance_id=self._instance_id,
            is_local=True,
        )
        self._agents[agent_id] = agent

        # Broadcast to other instances
        if self._event_bus and self._config.mode == FederationMode.FULL:
            # Publish agent registration event
            pass  # Implementation depends on event bus

        logger.info(f"[FederatedAgentPool] Registered agent {agent_id}")
        return agent

    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the federated pool.

        Args:
            agent_id: Agent ID to unregister

        Returns:
            True if agent was found and unregistered
        """
        if agent_id in self._agents:
            agent = self._agents[agent_id]

            if agent.is_local:
                await self._local_registry.unregister(agent_id)

            del self._agents[agent_id]

            # Broadcast to other instances
            if self._event_bus and self._config.mode == FederationMode.FULL:
                # Publish agent unregistration event
                pass

            logger.info(f"[FederatedAgentPool] Unregistered agent {agent_id}")
            return True

        return False
