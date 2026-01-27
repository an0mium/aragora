"""
Convoy Coordinator: Intelligent Bead Distribution.

This module provides intelligent distribution of beads within convoys to agents,
considering agent capabilities, current load, and delegation strategies.

Key concepts:
- ConvoyCoordinator: Assigns beads to agents based on multiple factors
- BeadAssignment: Tracks who is assigned what bead and when
- RebalancePolicy: When and how to redistribute work
- LoadTracker: Monitors agent load from HookQueue depth

Usage:
    from aragora.nomic.convoy_coordinator import ConvoyCoordinator

    coordinator = ConvoyCoordinator(
        convoy_manager=convoy_manager,
        hierarchy=agent_hierarchy,
        hook_queue=hook_queue,
    )
    await coordinator.initialize()

    # Distribute beads in a convoy
    assignments = await coordinator.distribute_convoy(convoy_id)

    # Check for rebalancing needs
    await coordinator.check_rebalance(convoy_id)

    # Handle agent failure
    await coordinator.handle_agent_failure(agent_id)
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.nomic.beads import BeadStore
    from aragora.nomic.convoys import Convoy, ConvoyManager
    from aragora.nomic.hook_queue import HookQueue

from aragora.nomic.agent_roles import (
    AgentHierarchy,
    AgentRole,
    RoleBasedRouter,
    RoleCapability,
)

logger = logging.getLogger(__name__)


class AssignmentStatus(str, Enum):
    """Status of a bead assignment."""

    PENDING = "pending"  # Assigned but not started
    ACTIVE = "active"  # Agent is working on it
    COMPLETED = "completed"  # Work finished
    FAILED = "failed"  # Work failed
    REASSIGNED = "reassigned"  # Moved to another agent


class RebalanceReason(str, Enum):
    """Reasons for triggering rebalance."""

    AGENT_FAILURE = "agent_failure"
    AGENT_OVERLOADED = "agent_overloaded"
    AGENT_IDLE = "agent_idle"
    PROGRESS_STALLED = "progress_stalled"
    MANUAL = "manual"
    PRIORITY_CHANGE = "priority_change"


@dataclass
class BeadAssignment:
    """
    Tracks assignment of a bead to an agent.

    Records who is assigned what, when, and tracks progress.
    """

    id: str
    bead_id: str
    agent_id: str
    convoy_id: str
    status: AssignmentStatus
    assigned_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration_minutes: int = 30
    actual_duration_minutes: Optional[int] = None
    priority: int = 50
    previous_agents: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "bead_id": self.bead_id,
            "agent_id": self.agent_id,
            "convoy_id": self.convoy_id,
            "status": self.status.value,
            "assigned_at": self.assigned_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "estimated_duration_minutes": self.estimated_duration_minutes,
            "actual_duration_minutes": self.actual_duration_minutes,
            "priority": self.priority,
            "previous_agents": self.previous_agents,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BeadAssignment":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            bead_id=data["bead_id"],
            agent_id=data["agent_id"],
            convoy_id=data["convoy_id"],
            status=AssignmentStatus(data["status"]),
            assigned_at=datetime.fromisoformat(data["assigned_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            started_at=(
                datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
            ),
            estimated_duration_minutes=data.get("estimated_duration_minutes", 30),
            actual_duration_minutes=data.get("actual_duration_minutes"),
            priority=data.get("priority", 50),
            previous_agents=data.get("previous_agents", []),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AgentLoad:
    """Tracks current load for an agent."""

    agent_id: str
    active_beads: int = 0
    pending_beads: int = 0
    completed_today: int = 0
    failed_today: int = 0
    avg_completion_minutes: float = 30.0
    last_heartbeat: Optional[datetime] = None
    is_available: bool = True

    @property
    def total_assigned(self) -> int:
        """Total beads currently assigned."""
        return self.active_beads + self.pending_beads

    @property
    def capacity_score(self) -> float:
        """Score from 0-1 indicating remaining capacity (1 = fully available)."""
        max_concurrent = 3  # Default max concurrent beads
        if self.total_assigned >= max_concurrent:
            return 0.0
        return 1.0 - (self.total_assigned / max_concurrent)


@dataclass
class RebalancePolicy:
    """
    Policy for when to rebalance work.

    Configures thresholds and behaviors for work redistribution.
    """

    # Load thresholds
    max_beads_per_agent: int = 3
    min_capacity_threshold: float = 0.2  # Trigger rebalance below this

    # Stall detection
    stall_threshold_minutes: int = 10
    max_reassignments: int = 3

    # Timing
    rebalance_interval_seconds: int = 60
    cooldown_seconds: int = 30

    # Behaviors
    prefer_persistent_agents: bool = True  # Prefer CREW over POLECAT
    spawn_polecats_on_demand: bool = True
    allow_cross_convoy_help: bool = False

    def should_rebalance(
        self,
        agent_load: AgentLoad,
        assignment: BeadAssignment,
    ) -> Optional[RebalanceReason]:
        """
        Determine if assignment should be rebalanced.

        Returns:
            RebalanceReason if rebalance needed, None otherwise
        """
        # Agent not available
        if not agent_load.is_available:
            return RebalanceReason.AGENT_FAILURE

        # Agent overloaded
        if agent_load.total_assigned > self.max_beads_per_agent:
            return RebalanceReason.AGENT_OVERLOADED

        # Capacity too low
        if agent_load.capacity_score < self.min_capacity_threshold:
            return RebalanceReason.AGENT_OVERLOADED

        # Check for stalled progress
        if assignment.started_at:
            minutes_active = (
                datetime.now(timezone.utc) - assignment.started_at
            ).total_seconds() / 60
            if minutes_active > self.stall_threshold_minutes:
                expected = assignment.estimated_duration_minutes
                if minutes_active > expected * 2:
                    return RebalanceReason.PROGRESS_STALLED

        return None


class ConvoyCoordinator:
    """
    Coordinates bead distribution within convoys.

    Manages assignment of beads to agents based on:
    - Agent capabilities and roles
    - Current load from hook queue
    - Delegation strategies
    - Convoy priorities
    """

    def __init__(
        self,
        convoy_manager: "ConvoyManager",
        hierarchy: AgentHierarchy,
        hook_queue: Optional["HookQueue"] = None,
        bead_store: Optional["BeadStore"] = None,
        storage_dir: Optional[Path] = None,
        policy: Optional[RebalancePolicy] = None,
    ):
        """
        Initialize the coordinator.

        Args:
            convoy_manager: Manager for convoy operations
            hierarchy: Agent role hierarchy
            hook_queue: Optional hook queue for load tracking
            bead_store: Optional bead store for bead operations
            storage_dir: Directory for persistence
            policy: Rebalance policy configuration
        """
        self.convoy_manager = convoy_manager
        self.hierarchy = hierarchy
        self.hook_queue = hook_queue
        self.bead_store = bead_store or convoy_manager.bead_store
        self.storage_dir = storage_dir or Path(".convoys")
        self.policy = policy or RebalancePolicy()

        self._assignments: Dict[str, BeadAssignment] = {}  # assignment_id -> assignment
        self._bead_assignments: Dict[str, str] = {}  # bead_id -> assignment_id
        self._agent_loads: Dict[str, AgentLoad] = {}
        self._lock = asyncio.Lock()
        self._initialized = False

        # Router for role-based task routing
        self.router = RoleBasedRouter(hierarchy)

    async def initialize(self) -> None:
        """Initialize the coordinator, loading existing assignments."""
        if self._initialized:
            return

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        await self._load_assignments()
        await self._refresh_agent_loads()
        self._initialized = True
        logger.info(f"ConvoyCoordinator initialized with {len(self._assignments)} assignments")

    async def _load_assignments(self) -> None:
        """Load assignments from storage."""
        assignments_file = self.storage_dir / "assignments.jsonl"
        if not assignments_file.exists():
            return

        try:
            with open(assignments_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        assignment = BeadAssignment.from_dict(data)
                        self._assignments[assignment.id] = assignment
                        self._bead_assignments[assignment.bead_id] = assignment.id
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Invalid assignment data: {e}")
        except Exception as e:
            logger.error(f"Failed to load assignments: {e}")

    async def _save_assignments(self) -> None:
        """Save assignments to storage."""
        assignments_file = self.storage_dir / "assignments.jsonl"
        temp_file = assignments_file.with_suffix(".tmp")

        try:
            with open(temp_file, "w") as f:
                for assignment in self._assignments.values():
                    f.write(json.dumps(assignment.to_dict()) + "\n")
            temp_file.rename(assignments_file)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            logger.error(f"Failed to save assignments: {e}")

    async def _refresh_agent_loads(self) -> None:
        """Refresh agent load information."""
        # Get all registered agents
        for role in AgentRole:
            agents = await self.hierarchy.get_agents_by_role(role)
            for agent_assignment in agents:
                agent_id = agent_assignment.agent_id
                if agent_id not in self._agent_loads:
                    self._agent_loads[agent_id] = AgentLoad(agent_id=agent_id)

        # Count active assignments per agent
        for agent_id in self._agent_loads:
            load = self._agent_loads[agent_id]
            load.active_beads = 0
            load.pending_beads = 0

        for assignment in self._assignments.values():
            if assignment.agent_id not in self._agent_loads:
                self._agent_loads[assignment.agent_id] = AgentLoad(agent_id=assignment.agent_id)

            load = self._agent_loads[assignment.agent_id]
            if assignment.status == AssignmentStatus.ACTIVE:
                load.active_beads += 1
            elif assignment.status == AssignmentStatus.PENDING:
                load.pending_beads += 1

        # Update from hook queue if available
        if self.hook_queue:
            await self._update_loads_from_hook_queue()

    async def _update_loads_from_hook_queue(self) -> None:
        """Update agent loads from hook queue depth."""
        if not self.hook_queue:
            return

        # Hook queue provides additional load signals
        try:
            queue_stats = await self.hook_queue.get_statistics()
            # Update agent availability based on queue depth
            for agent_id, stats in queue_stats.get("by_agent", {}).items():
                if agent_id in self._agent_loads:
                    load = self._agent_loads[agent_id]
                    # High queue depth indicates agent is busy
                    queue_depth = stats.get("pending", 0)
                    load.pending_beads = max(load.pending_beads, queue_depth)
        except Exception as e:
            logger.warning(f"Failed to update loads from hook queue: {e}")

    async def distribute_convoy(
        self,
        convoy_id: str,
        agent_ids: Optional[List[str]] = None,
        strategy: str = "balanced",
    ) -> List[BeadAssignment]:
        """
        Distribute beads in a convoy to agents.

        Args:
            convoy_id: Convoy to distribute
            agent_ids: Specific agents to use (None = auto-select)
            strategy: Distribution strategy (balanced, round_robin, priority)

        Returns:
            List of created bead assignments
        """
        async with self._lock:
            convoy = await self.convoy_manager.get_convoy(convoy_id)
            if not convoy:
                raise ValueError(f"Convoy {convoy_id} not found")

            # Get agents to distribute to
            agents = await self._select_distribution_agents(convoy, agent_ids)
            if not agents:
                raise ValueError("No available agents for distribution")

            # Get beads that need assignment
            unassigned_beads = await self._get_unassigned_beads(convoy)
            if not unassigned_beads:
                logger.info(f"All beads in convoy {convoy_id} already assigned")
                return []

            # Distribute based on strategy
            assignments = []
            if strategy == "round_robin":
                assignments = await self._distribute_round_robin(convoy, unassigned_beads, agents)
            elif strategy == "priority":
                assignments = await self._distribute_by_priority(convoy, unassigned_beads, agents)
            else:  # balanced (default)
                assignments = await self._distribute_balanced(convoy, unassigned_beads, agents)

            await self._save_assignments()
            logger.info(
                f"Distributed {len(assignments)} beads in convoy {convoy_id} to {len(agents)} agents"
            )
            return assignments

    async def _select_distribution_agents(
        self,
        convoy: "Convoy",
        agent_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Select agents for distribution."""
        if agent_ids:
            # Verify agents exist and are available
            available = []
            for agent_id in agent_ids:
                assignment = await self.hierarchy.get_assignment(agent_id)
                if assignment and assignment.has_capability(RoleCapability.CLAIM_BEAD):
                    load = self._agent_loads.get(agent_id)
                    if not load or load.is_available:
                        available.append(agent_id)
            return available

        # Auto-select agents with CLAIM_BEAD capability
        capable_agents = await self.hierarchy.get_agents_by_capability(RoleCapability.CLAIM_BEAD)

        # Filter by availability and load
        available = []
        for agent_assignment in capable_agents:
            agent_id = agent_assignment.agent_id
            load = self._agent_loads.get(agent_id)
            if not load:
                load = AgentLoad(agent_id=agent_id)
                self._agent_loads[agent_id] = load

            if load.is_available and load.capacity_score > 0:
                available.append(agent_id)

        # Prefer CREW over POLECAT if configured
        if self.policy.prefer_persistent_agents:
            crew = [
                a.agent_id
                for a in capable_agents
                if a.role == AgentRole.CREW and a.agent_id in available
            ]
            if crew:
                return crew

        return available

    async def _get_unassigned_beads(self, convoy: "Convoy") -> List[str]:
        """Get bead IDs that need assignment."""
        unassigned = []
        for bead_id in convoy.bead_ids:
            if bead_id not in self._bead_assignments:
                unassigned.append(bead_id)
            else:
                # Check if existing assignment is terminal
                assignment_id = self._bead_assignments[bead_id]
                assignment = self._assignments.get(assignment_id)
                if assignment and assignment.status in (
                    AssignmentStatus.COMPLETED,
                    AssignmentStatus.FAILED,
                ):
                    # Could be retried
                    if assignment.status == AssignmentStatus.FAILED:
                        unassigned.append(bead_id)
        return unassigned

    async def _distribute_balanced(
        self,
        convoy: "Convoy",
        bead_ids: List[str],
        agent_ids: List[str],
    ) -> List[BeadAssignment]:
        """Distribute beads evenly across agents considering load."""
        import uuid

        assignments = []
        now = datetime.now(timezone.utc)

        # Sort agents by capacity (most available first)
        sorted_agents = sorted(
            agent_ids,
            key=lambda a: self._agent_loads.get(a, AgentLoad(agent_id=a)).capacity_score,
            reverse=True,
        )

        # Distribute beads
        agent_idx = 0
        for bead_id in bead_ids:
            # Find next available agent
            attempts = 0
            while attempts < len(sorted_agents):
                agent_id = sorted_agents[agent_idx % len(sorted_agents)]
                load = self._agent_loads.get(agent_id, AgentLoad(agent_id=agent_id))

                if load.total_assigned < self.policy.max_beads_per_agent:
                    break

                agent_idx += 1
                attempts += 1
            else:
                # All agents at capacity, spawn polecat if allowed
                if self.policy.spawn_polecats_on_demand:
                    agent_id = await self._spawn_polecat_for_bead(bead_id)
                else:
                    logger.warning(f"No capacity for bead {bead_id}, skipping")
                    continue

            # Create assignment
            assignment = BeadAssignment(
                id=str(uuid.uuid4()),
                bead_id=bead_id,
                agent_id=agent_id,
                convoy_id=convoy.id,
                status=AssignmentStatus.PENDING,
                assigned_at=now,
                updated_at=now,
                priority=convoy.priority.value,
            )

            self._assignments[assignment.id] = assignment
            self._bead_assignments[bead_id] = assignment.id

            # Update load tracking
            load = self._agent_loads.get(agent_id, AgentLoad(agent_id=agent_id))
            load.pending_beads += 1
            self._agent_loads[agent_id] = load

            assignments.append(assignment)
            agent_idx += 1

        return assignments

    async def _distribute_round_robin(
        self,
        convoy: "Convoy",
        bead_ids: List[str],
        agent_ids: List[str],
    ) -> List[BeadAssignment]:
        """Distribute beads in round-robin fashion."""
        import uuid

        assignments = []
        now = datetime.now(timezone.utc)

        for i, bead_id in enumerate(bead_ids):
            agent_id = agent_ids[i % len(agent_ids)]

            assignment = BeadAssignment(
                id=str(uuid.uuid4()),
                bead_id=bead_id,
                agent_id=agent_id,
                convoy_id=convoy.id,
                status=AssignmentStatus.PENDING,
                assigned_at=now,
                updated_at=now,
                priority=convoy.priority.value,
            )

            self._assignments[assignment.id] = assignment
            self._bead_assignments[bead_id] = assignment.id

            load = self._agent_loads.get(agent_id, AgentLoad(agent_id=agent_id))
            load.pending_beads += 1
            self._agent_loads[agent_id] = load

            assignments.append(assignment)

        return assignments

    async def _distribute_by_priority(
        self,
        convoy: "Convoy",
        bead_ids: List[str],
        agent_ids: List[str],
    ) -> List[BeadAssignment]:
        """Distribute beads prioritizing high-priority to best agents."""
        import uuid

        assignments = []
        now = datetime.now(timezone.utc)

        # Get bead priorities
        bead_priorities = []
        for bead_id in bead_ids:
            bead = await self.bead_store.get(bead_id)
            priority = bead.priority.value if bead else 50
            bead_priorities.append((bead_id, priority))

        # Sort beads by priority (highest first)
        bead_priorities.sort(key=lambda x: x[1], reverse=True)

        # Sort agents by capacity
        sorted_agents = sorted(
            agent_ids,
            key=lambda a: self._agent_loads.get(a, AgentLoad(agent_id=a)).capacity_score,
            reverse=True,
        )

        # Assign highest priority beads to best agents
        for i, (bead_id, priority) in enumerate(bead_priorities):
            agent_id = sorted_agents[min(i, len(sorted_agents) - 1)]

            assignment = BeadAssignment(
                id=str(uuid.uuid4()),
                bead_id=bead_id,
                agent_id=agent_id,
                convoy_id=convoy.id,
                status=AssignmentStatus.PENDING,
                assigned_at=now,
                updated_at=now,
                priority=priority,
            )

            self._assignments[assignment.id] = assignment
            self._bead_assignments[bead_id] = assignment.id

            load = self._agent_loads.get(agent_id, AgentLoad(agent_id=agent_id))
            load.pending_beads += 1
            self._agent_loads[agent_id] = load

            assignments.append(assignment)

        return assignments

    async def _spawn_polecat_for_bead(self, bead_id: str) -> str:
        """Spawn an ephemeral polecat worker for a bead."""
        bead = await self.bead_store.get(bead_id)
        description = bead.title if bead else f"Work on bead {bead_id}"

        # Find a supervisor (Mayor or Witness)
        supervisor = await self.router.route_to_mayor()
        if not supervisor:
            supervisor = await self.router.route_to_witness()

        if not supervisor:
            raise ValueError("No supervisor available to spawn Polecat")

        assignment = await self.hierarchy.spawn_polecat(
            supervised_by=supervisor,
            task_description=description,
        )

        # Initialize load tracking
        self._agent_loads[assignment.agent_id] = AgentLoad(agent_id=assignment.agent_id)

        logger.info(f"Spawned Polecat {assignment.agent_id} for bead {bead_id}")
        return assignment.agent_id

    async def check_rebalance(self, convoy_id: Optional[str] = None) -> List[BeadAssignment]:
        """
        Check assignments and rebalance if needed.

        Args:
            convoy_id: Specific convoy to check (None = all)

        Returns:
            List of reassigned assignments
        """
        async with self._lock:
            await self._refresh_agent_loads()

            reassigned = []
            assignments_to_check = list(self._assignments.values())

            if convoy_id:
                assignments_to_check = [a for a in assignments_to_check if a.convoy_id == convoy_id]

            for assignment in assignments_to_check:
                if assignment.status not in (AssignmentStatus.PENDING, AssignmentStatus.ACTIVE):
                    continue

                load = self._agent_loads.get(
                    assignment.agent_id,
                    AgentLoad(agent_id=assignment.agent_id),
                )

                reason = self.policy.should_rebalance(load, assignment)
                if reason:
                    new_assignment = await self._reassign_bead(assignment, reason)
                    if new_assignment:
                        reassigned.append(new_assignment)

            if reassigned:
                await self._save_assignments()
                logger.info(f"Rebalanced {len(reassigned)} assignments")

            return reassigned

    async def _reassign_bead(
        self,
        assignment: BeadAssignment,
        reason: RebalanceReason,
    ) -> Optional[BeadAssignment]:
        """Reassign a bead to a different agent."""
        import uuid

        # Check reassignment limit
        if len(assignment.previous_agents) >= self.policy.max_reassignments:
            logger.warning(
                f"Bead {assignment.bead_id} hit max reassignments ({self.policy.max_reassignments})"
            )
            return None

        # Find new agent
        current_agent = assignment.agent_id
        agents = await self._select_distribution_agents(
            await self.convoy_manager.get_convoy(assignment.convoy_id),
            None,
        )
        agents = [a for a in agents if a != current_agent]

        if not agents:
            if self.policy.spawn_polecats_on_demand:
                new_agent = await self._spawn_polecat_for_bead(assignment.bead_id)
            else:
                logger.warning(f"No agents available to reassign bead {assignment.bead_id}")
                return None
        else:
            # Pick agent with most capacity
            new_agent = max(
                agents,
                key=lambda a: self._agent_loads.get(a, AgentLoad(agent_id=a)).capacity_score,
            )

        # Mark old assignment as reassigned
        assignment.status = AssignmentStatus.REASSIGNED
        assignment.updated_at = datetime.now(timezone.utc)
        assignment.metadata["reassign_reason"] = reason.value

        # Create new assignment
        now = datetime.now(timezone.utc)
        new_assignment = BeadAssignment(
            id=str(uuid.uuid4()),
            bead_id=assignment.bead_id,
            agent_id=new_agent,
            convoy_id=assignment.convoy_id,
            status=AssignmentStatus.PENDING,
            assigned_at=now,
            updated_at=now,
            priority=assignment.priority,
            previous_agents=assignment.previous_agents + [current_agent],
            metadata={"reassigned_from": assignment.id, "reason": reason.value},
        )

        self._assignments[new_assignment.id] = new_assignment
        self._bead_assignments[assignment.bead_id] = new_assignment.id

        # Update loads
        old_load = self._agent_loads.get(current_agent)
        if old_load:
            if assignment.status == AssignmentStatus.ACTIVE:
                old_load.active_beads = max(0, old_load.active_beads - 1)
            else:
                old_load.pending_beads = max(0, old_load.pending_beads - 1)

        new_load = self._agent_loads.get(new_agent, AgentLoad(agent_id=new_agent))
        new_load.pending_beads += 1
        self._agent_loads[new_agent] = new_load

        logger.info(
            f"Reassigned bead {assignment.bead_id} from {current_agent} to {new_agent} "
            f"(reason: {reason.value})"
        )
        return new_assignment

    async def handle_agent_failure(self, agent_id: str) -> List[BeadAssignment]:
        """
        Handle agent failure by reassigning its beads.

        Args:
            agent_id: ID of failed agent

        Returns:
            List of reassigned assignments
        """
        async with self._lock:
            # Mark agent as unavailable
            load = self._agent_loads.get(agent_id, AgentLoad(agent_id=agent_id))
            load.is_available = False
            self._agent_loads[agent_id] = load

            # Find active assignments
            agent_assignments = [
                a
                for a in self._assignments.values()
                if a.agent_id == agent_id
                and a.status in (AssignmentStatus.PENDING, AssignmentStatus.ACTIVE)
            ]

            reassigned = []
            for assignment in agent_assignments:
                new_assignment = await self._reassign_bead(
                    assignment, RebalanceReason.AGENT_FAILURE
                )
                if new_assignment:
                    reassigned.append(new_assignment)

            if reassigned:
                await self._save_assignments()

            logger.info(f"Handled failure of agent {agent_id}, reassigned {len(reassigned)} beads")
            return reassigned

    async def update_assignment_status(
        self,
        bead_id: str,
        status: AssignmentStatus,
        error_message: Optional[str] = None,
    ) -> Optional[BeadAssignment]:
        """
        Update assignment status for a bead.

        Args:
            bead_id: Bead to update
            status: New status
            error_message: Optional error message for failures

        Returns:
            Updated assignment or None
        """
        async with self._lock:
            assignment_id = self._bead_assignments.get(bead_id)
            if not assignment_id:
                return None

            assignment = self._assignments.get(assignment_id)
            if not assignment:
                return None

            now = datetime.now(timezone.utc)
            old_status = assignment.status
            assignment.status = status
            assignment.updated_at = now

            if status == AssignmentStatus.ACTIVE and not assignment.started_at:
                assignment.started_at = now

            if status in (AssignmentStatus.COMPLETED, AssignmentStatus.FAILED):
                assignment.completed_at = now
                if assignment.started_at:
                    duration = (now - assignment.started_at).total_seconds() / 60
                    assignment.actual_duration_minutes = int(duration)

            if error_message:
                assignment.error_message = error_message

            # Update load tracking
            load = self._agent_loads.get(assignment.agent_id)
            if load:
                if old_status == AssignmentStatus.PENDING:
                    load.pending_beads = max(0, load.pending_beads - 1)
                elif old_status == AssignmentStatus.ACTIVE:
                    load.active_beads = max(0, load.active_beads - 1)

                if status == AssignmentStatus.ACTIVE:
                    load.active_beads += 1
                elif status == AssignmentStatus.COMPLETED:
                    load.completed_today += 1
                elif status == AssignmentStatus.FAILED:
                    load.failed_today += 1

            await self._save_assignments()
            return assignment

    async def get_assignment(self, bead_id: str) -> Optional[BeadAssignment]:
        """Get current assignment for a bead."""
        assignment_id = self._bead_assignments.get(bead_id)
        if not assignment_id:
            return None
        return self._assignments.get(assignment_id)

    async def get_agent_assignments(
        self,
        agent_id: str,
        status: Optional[AssignmentStatus] = None,
    ) -> List[BeadAssignment]:
        """Get assignments for an agent."""
        assignments = [a for a in self._assignments.values() if a.agent_id == agent_id]
        if status:
            assignments = [a for a in assignments if a.status == status]
        return assignments

    async def get_convoy_assignments(self, convoy_id: str) -> List[BeadAssignment]:
        """Get all assignments for a convoy."""
        return [a for a in self._assignments.values() if a.convoy_id == convoy_id]

    async def get_statistics(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        assignments = list(self._assignments.values())
        by_status: Dict[str, int] = {}
        by_agent: Dict[str, int] = {}

        for assignment in assignments:
            status_key = assignment.status.value
            by_status[status_key] = by_status.get(status_key, 0) + 1

            agent_key = assignment.agent_id
            by_agent[agent_key] = by_agent.get(agent_key, 0) + 1

        return {
            "total_assignments": len(assignments),
            "by_status": by_status,
            "by_agent": by_agent,
            "agent_loads": {
                agent_id: {
                    "active": load.active_beads,
                    "pending": load.pending_beads,
                    "capacity": load.capacity_score,
                    "available": load.is_available,
                }
                for agent_id, load in self._agent_loads.items()
            },
        }


# Singleton instance
_default_coordinator: Optional[ConvoyCoordinator] = None


async def get_convoy_coordinator(
    convoy_manager: "ConvoyManager",
    hierarchy: AgentHierarchy,
    hook_queue: Optional["HookQueue"] = None,
) -> ConvoyCoordinator:
    """Get the default convoy coordinator instance."""
    global _default_coordinator
    if _default_coordinator is None:
        _default_coordinator = ConvoyCoordinator(
            convoy_manager=convoy_manager,
            hierarchy=hierarchy,
            hook_queue=hook_queue,
        )
        await _default_coordinator.initialize()
    return _default_coordinator


def reset_convoy_coordinator() -> None:
    """Reset the default coordinator (for testing)."""
    global _default_coordinator
    _default_coordinator = None
