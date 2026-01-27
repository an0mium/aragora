"""
Mayor Coordinator: Bridges Leader Election with Agent Hierarchy.

This module connects the distributed leader election system with the Gas Town
agent hierarchy, ensuring the elected leader becomes the MAYOR and losers
demote to WITNESS roles.

Key concepts:
- MayorCoordinator: Bridges LeaderElection with AgentHierarchy
- Automatic role promotion/demotion based on leadership
- Regional mayors for convoy sharding (optional)

Usage:
    from aragora.nomic.mayor_coordinator import MayorCoordinator

    coordinator = MayorCoordinator(
        hierarchy=agent_hierarchy,
        node_id="node-001",
    )
    await coordinator.start()

    # Check if this node is the mayor
    is_mayor = coordinator.is_mayor

    # Get current mayor info
    mayor_info = coordinator.get_mayor_info()
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from aragora.nomic.agent_roles import AgentHierarchy, AgentRole

logger = logging.getLogger(__name__)


@dataclass
class MayorInfo:
    """Information about the current mayor."""

    node_id: str
    agent_id: str
    became_mayor_at: datetime
    region: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "node_id": self.node_id,
            "agent_id": self.agent_id,
            "became_mayor_at": self.became_mayor_at.isoformat(),
            "region": self.region,
            "metadata": self.metadata,
        }


class MayorCoordinator:
    """
    Coordinates Mayor role with distributed leader election.

    When this node wins leader election, it promotes itself to MAYOR role.
    When it loses leadership, it demotes to WITNESS role.
    """

    def __init__(
        self,
        hierarchy: AgentHierarchy,
        node_id: Optional[str] = None,
        region: Optional[str] = None,
        on_become_mayor: Optional[Callable[[], Any]] = None,
        on_lose_mayor: Optional[Callable[[], Any]] = None,
    ):
        """
        Initialize the mayor coordinator.

        Args:
            hierarchy: Agent hierarchy for role management
            node_id: This node's identifier (auto-generated if not provided)
            region: Optional region for regional leader election
            on_become_mayor: Callback when this node becomes mayor
            on_lose_mayor: Callback when this node loses mayor role
        """
        self.hierarchy = hierarchy
        self.node_id = node_id or f"node-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        self.region = region
        self._agent_id = f"mayor-{self.node_id}"

        self._election: Optional[Any] = None  # LeaderElection instance
        self._is_mayor = False
        self._mayor_info: Optional[MayorInfo] = None
        self._started = False
        self._lock = asyncio.Lock()

        # Callbacks
        self._on_become_mayor = on_become_mayor
        self._on_lose_mayor = on_lose_mayor

        # Track current mayor (even if it's another node)
        self._current_mayor_node: Optional[str] = None

    @property
    def is_mayor(self) -> bool:
        """Check if this node is the current mayor."""
        return self._is_mayor

    @property
    def is_started(self) -> bool:
        """Check if the coordinator is running."""
        return self._started

    def get_mayor_info(self) -> Optional[MayorInfo]:
        """Get information about this node's mayor role (if mayor)."""
        return self._mayor_info if self._is_mayor else None

    def get_current_mayor_node(self) -> Optional[str]:
        """Get the node ID of the current mayor (may be another node)."""
        return self._current_mayor_node

    async def start(self) -> bool:
        """
        Start the mayor coordinator and leader election.

        Returns:
            True if started successfully, False otherwise
        """
        if self._started:
            return True

        try:
            from aragora.control_plane.leader import LeaderElection, LeaderConfig

            # Configure leader election
            config = LeaderConfig(
                node_id=self.node_id,
                lock_ttl_seconds=30.0,
                heartbeat_interval=10.0,
                key_prefix=f"aragora:mayor:{self.region or 'global'}:",
            )

            self._election = LeaderElection(config=config)

            # Register callbacks
            self._election.on_become_leader(self._handle_become_leader)
            self._election.on_lose_leader(self._handle_lose_leader)
            self._election.on_leader_change(self._handle_leader_change)

            # Start election
            await self._election.start()

            self._started = True
            logger.info(
                f"Mayor coordinator started for node {self.node_id} "
                f"(region={self.region or 'global'})"
            )
            return True

        except ImportError as e:
            logger.warning(f"Leader election not available: {e}")
            # Fall back to single-node mode - this node is always mayor
            await self._promote_to_mayor()
            self._started = True
            return True

        except Exception as e:
            logger.error(f"Failed to start mayor coordinator: {e}")
            return False

    async def stop(self) -> None:
        """Stop the mayor coordinator and resign leadership."""
        if not self._started:
            return

        if self._election:
            await self._election.stop()

        if self._is_mayor:
            await self._demote_from_mayor()

        self._started = False
        logger.info(f"Mayor coordinator stopped for node {self.node_id}")

    async def _handle_become_leader(self) -> None:
        """Called when this node wins leader election."""
        await self._promote_to_mayor()

    async def _handle_lose_leader(self) -> None:
        """Called when this node loses leader election."""
        await self._demote_from_mayor()

    async def _handle_leader_change(self, new_leader: Optional[str]) -> None:
        """Called when any leader change occurs."""
        self._current_mayor_node = new_leader
        logger.info(f"Mayor changed to node: {new_leader or 'none'}")

    async def _promote_to_mayor(self) -> None:
        """Promote this node to MAYOR role."""
        async with self._lock:
            if self._is_mayor:
                return

            try:
                # Register as MAYOR in hierarchy
                await self.hierarchy.register_agent(
                    agent_id=self._agent_id,
                    role=AgentRole.MAYOR,
                    metadata={
                        "node_id": self.node_id,
                        "region": self.region,
                        "promoted_at": datetime.now(timezone.utc).isoformat(),
                    },
                )

                self._is_mayor = True
                self._current_mayor_node = self.node_id
                self._mayor_info = MayorInfo(
                    node_id=self.node_id,
                    agent_id=self._agent_id,
                    became_mayor_at=datetime.now(timezone.utc),
                    region=self.region,
                )

                logger.info(f"Node {self.node_id} promoted to MAYOR (agent={self._agent_id})")

                # Call callback
                if self._on_become_mayor:
                    try:
                        result = self._on_become_mayor()
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.error(f"on_become_mayor callback error: {e}")

            except Exception as e:
                logger.error(f"Failed to promote to MAYOR: {e}")

    async def _demote_from_mayor(self) -> None:
        """Demote this node from MAYOR to WITNESS role."""
        async with self._lock:
            if not self._is_mayor:
                return

            try:
                # Unregister as MAYOR
                await self.hierarchy.unregister_agent(self._agent_id)

                # Register as WITNESS instead
                witness_agent_id = f"witness-{self.node_id}"
                await self.hierarchy.register_agent(
                    agent_id=witness_agent_id,
                    role=AgentRole.WITNESS,
                    metadata={
                        "node_id": self.node_id,
                        "region": self.region,
                        "demoted_at": datetime.now(timezone.utc).isoformat(),
                        "former_mayor": True,
                    },
                )

                self._is_mayor = False
                self._mayor_info = None

                logger.info(
                    f"Node {self.node_id} demoted from MAYOR to WITNESS (agent={witness_agent_id})"
                )

                # Call callback
                if self._on_lose_mayor:
                    try:
                        result = self._on_lose_mayor()
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.error(f"on_lose_mayor callback error: {e}")

            except Exception as e:
                logger.error(f"Failed to demote from MAYOR: {e}")


# Global coordinator instance
_mayor_coordinator: Optional[MayorCoordinator] = None


def get_mayor_coordinator() -> Optional[MayorCoordinator]:
    """Get the global mayor coordinator instance."""
    return _mayor_coordinator


async def init_mayor_coordinator(
    hierarchy: AgentHierarchy,
    node_id: Optional[str] = None,
    region: Optional[str] = None,
) -> Optional[MayorCoordinator]:
    """
    Initialize and start the global mayor coordinator.

    Args:
        hierarchy: Agent hierarchy for role management
        node_id: This node's identifier
        region: Optional region for regional leader election

    Returns:
        The initialized MayorCoordinator, or None if initialization failed
    """
    global _mayor_coordinator

    coordinator = MayorCoordinator(
        hierarchy=hierarchy,
        node_id=node_id,
        region=region,
    )

    if await coordinator.start():
        _mayor_coordinator = coordinator
        return coordinator

    return None
