"""
Convoys: Grouped Work Orders.

Inspired by Gastown's Convoy pattern, this module provides batch assignment
and tracking of related beads. Convoys wrap multiple beads into a single
work order that can be assigned to agents.

Key concepts:
- Convoy: A grouped work order containing related beads
- ConvoyManager: Orchestrates convoy lifecycle and assignment
- ConvoyStatus: Tracks convoy progress (PENDING, ACTIVE, COMPLETED, FAILED)

Usage:
    manager = ConvoyManager(bead_store)

    # Create convoy from beads
    convoy = await manager.create_convoy(
        title="Implement auth feature",
        bead_ids=["bead-1", "bead-2", "bead-3"],
        priority=1,
    )

    # Assign to agents
    await manager.assign_convoy(convoy.id, agent_ids=["agent-001", "agent-002"])

    # Track progress
    status = await manager.get_convoy_progress(convoy.id)
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from aragora.nomic.beads import Bead, BeadPriority, BeadStatus, BeadStore, BeadType

logger = logging.getLogger(__name__)


class ConvoyStatus(str, Enum):
    """Lifecycle status of a convoy."""

    PENDING = "pending"  # Not yet started
    ACTIVE = "active"  # Work in progress
    COMPLETED = "completed"  # All beads completed
    FAILED = "failed"  # One or more beads failed
    CANCELLED = "cancelled"  # Convoy cancelled
    PARTIAL = "partial"  # Some beads completed, some failed


class ConvoyPriority(int, Enum):
    """Priority levels for convoys."""

    LOW = 0
    NORMAL = 50
    HIGH = 75
    URGENT = 100


@dataclass
class ConvoyProgress:
    """Progress tracking for a convoy."""

    total_beads: int
    pending_beads: int
    running_beads: int
    completed_beads: int
    failed_beads: int
    completion_percentage: float

    @property
    def is_complete(self) -> bool:
        """Check if convoy is complete (all beads finished)."""
        return self.pending_beads == 0 and self.running_beads == 0


@dataclass
class Convoy:
    """
    Grouped work order wrapping related beads.

    Convoys provide a way to batch assign and track related work items.
    They support dependencies between convoys for complex workflows.
    """

    id: str
    title: str
    description: str
    bead_ids: List[str]  # Beads in this convoy
    status: ConvoyStatus
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_to: List[str] = field(default_factory=list)  # agent_ids
    priority: ConvoyPriority = ConvoyPriority.NORMAL
    dependencies: List[str] = field(default_factory=list)  # Convoy IDs
    parent_id: Optional[str] = None  # For nested convoys
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    @classmethod
    def create(
        cls,
        title: str,
        bead_ids: List[str],
        description: str = "",
        priority: ConvoyPriority = ConvoyPriority.NORMAL,
        dependencies: Optional[List[str]] = None,
        parent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Convoy":
        """Create a new convoy with generated ID and timestamps."""
        now = datetime.now(timezone.utc)
        return cls(
            id=str(uuid.uuid4()),
            title=title,
            description=description,
            bead_ids=bead_ids,
            status=ConvoyStatus.PENDING,
            created_at=now,
            updated_at=now,
            priority=priority,
            dependencies=dependencies or [],
            parent_id=parent_id,
            tags=tags or [],
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize convoy to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "bead_ids": self.bead_ids,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "assigned_to": self.assigned_to,
            "priority": self.priority.value,
            "dependencies": self.dependencies,
            "parent_id": self.parent_id,
            "tags": self.tags,
            "metadata": self.metadata,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Convoy":
        """Deserialize convoy from dictionary."""
        return cls(
            id=data["id"],
            title=data["title"],
            description=data.get("description", ""),
            bead_ids=data["bead_ids"],
            status=ConvoyStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            started_at=(
                datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
            ),
            assigned_to=data.get("assigned_to", []),
            priority=ConvoyPriority(data.get("priority", ConvoyPriority.NORMAL.value)),
            dependencies=data.get("dependencies", []),
            parent_id=data.get("parent_id"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            error_message=data.get("error_message"),
        )

    def can_start(self, completed_convoy_ids: Set[str]) -> bool:
        """Check if this convoy can start (all dependencies completed)."""
        return all(dep_id in completed_convoy_ids for dep_id in self.dependencies)


class ConvoyManager:
    """
    Manages convoy lifecycle and assignment.

    Coordinates with BeadStore for bead operations and provides
    higher-level convoy orchestration.
    """

    def __init__(
        self,
        bead_store: BeadStore,
        convoy_dir: Optional[Path] = None,
    ):
        """
        Initialize the convoy manager.

        Args:
            bead_store: The bead store for bead operations
            convoy_dir: Directory for convoy storage (defaults to bead_store's dir)
        """
        self.bead_store = bead_store
        self.convoy_dir = convoy_dir or bead_store.bead_dir
        self.convoy_file = self.convoy_dir / "convoys.jsonl"
        self._lock = asyncio.Lock()
        self._convoys: Dict[str, Convoy] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the manager, loading existing convoys."""
        if self._initialized:
            return

        self.convoy_dir.mkdir(parents=True, exist_ok=True)
        await self._load_convoys()
        self._initialized = True
        logger.info(f"ConvoyManager initialized with {len(self._convoys)} convoys")

    async def _load_convoys(self) -> None:
        """Load convoys from JSONL file."""
        if not self.convoy_file.exists():
            return

        try:
            with open(self.convoy_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        convoy = Convoy.from_dict(data)
                        self._convoys[convoy.id] = convoy
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Invalid convoy data: {e}")
        except Exception as e:
            logger.error(f"Failed to load convoys: {e}")

    async def _save_convoy(self, convoy: Convoy) -> None:
        """Save a convoy to the JSONL file."""
        # For simplicity, rewrite the file (could optimize)
        temp_file = self.convoy_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w") as f:
                for c in self._convoys.values():
                    f.write(json.dumps(c.to_dict()) + "\n")
            temp_file.rename(self.convoy_file)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise e

    async def create_convoy(
        self,
        title: str,
        bead_ids: List[str],
        description: str = "",
        priority: ConvoyPriority = ConvoyPriority.NORMAL,
        dependencies: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Convoy:
        """
        Create a new convoy from existing beads.

        Args:
            title: Convoy title
            bead_ids: List of bead IDs to include
            description: Optional description
            priority: Convoy priority
            dependencies: Other convoy IDs that must complete first
            tags: Optional tags
            metadata: Optional metadata

        Returns:
            The created convoy
        """
        async with self._lock:
            # Verify all beads exist
            for bead_id in bead_ids:
                bead = await self.bead_store.get(bead_id)
                if not bead:
                    raise ValueError(f"Bead {bead_id} not found")

            convoy = Convoy.create(
                title=title,
                bead_ids=bead_ids,
                description=description,
                priority=priority,
                dependencies=dependencies,
                tags=tags,
                metadata=metadata,
            )

            self._convoys[convoy.id] = convoy
            await self._save_convoy(convoy)

            logger.info(f"Created convoy: {convoy.id} ({convoy.title}) with {len(bead_ids)} beads")
            return convoy

    async def create_convoy_from_subtasks(
        self,
        title: str,
        subtasks: List[Dict[str, Any]],
        priority: ConvoyPriority = ConvoyPriority.NORMAL,
    ) -> Convoy:
        """
        Create a convoy by first creating beads from subtask definitions.

        Args:
            title: Convoy title
            subtasks: List of subtask dicts with title, description, dependencies
            priority: Convoy priority

        Returns:
            The created convoy with all beads
        """
        async with self._lock:
            bead_ids = []
            subtask_id_map: Dict[str, str] = {}  # subtask_id -> bead_id

            # First pass: create beads without dependencies
            for subtask in subtasks:
                # Convert ConvoyPriority to BeadPriority (same underlying values)
                bead_priority = BeadPriority(priority.value)
                bead = Bead.create(
                    bead_type=BeadType.TASK,
                    title=subtask.get("title", "Untitled"),
                    description=subtask.get("description", ""),
                    priority=bead_priority,
                    tags=subtask.get("tags", []),
                    metadata=subtask.get("metadata", {}),
                )
                await self.bead_store.create(bead)
                bead_ids.append(bead.id)
                subtask_id_map[subtask.get("id", bead.id)] = bead.id

            # Second pass: update dependencies
            for i, subtask in enumerate(subtasks):
                subtask_deps = subtask.get("dependencies", [])
                if subtask_deps:
                    bead_id = bead_ids[i]
                    bead = await self.bead_store.get(bead_id)
                    if bead:
                        bead.dependencies = [subtask_id_map.get(dep, dep) for dep in subtask_deps]
                        await self.bead_store.update(bead)

            # Create convoy
            convoy = Convoy.create(
                title=title,
                bead_ids=bead_ids,
                priority=priority,
            )

            self._convoys[convoy.id] = convoy
            await self._save_convoy(convoy)

            logger.info(f"Created convoy from subtasks: {convoy.id} with {len(bead_ids)} beads")
            return convoy

    async def get_convoy(self, convoy_id: str) -> Optional[Convoy]:
        """Get a convoy by ID."""
        return self._convoys.get(convoy_id)

    async def assign_convoy(
        self,
        convoy_id: str,
        agent_ids: List[str],
    ) -> bool:
        """
        Assign a convoy to agents.

        Args:
            convoy_id: The convoy to assign
            agent_ids: List of agent IDs to assign

        Returns:
            True if assigned successfully
        """
        async with self._lock:
            convoy = self._convoys.get(convoy_id)
            if not convoy:
                raise ValueError(f"Convoy {convoy_id} not found")

            if convoy.status != ConvoyStatus.PENDING:
                return False

            convoy.assigned_to = agent_ids
            convoy.status = ConvoyStatus.ACTIVE
            convoy.started_at = datetime.now(timezone.utc)
            convoy.updated_at = convoy.started_at

            await self._save_convoy(convoy)

            logger.info(f"Assigned convoy {convoy_id} to agents: {agent_ids}")
            return True

    async def get_convoy_progress(self, convoy_id: str) -> ConvoyProgress:
        """
        Get progress information for a convoy.

        Args:
            convoy_id: The convoy to check

        Returns:
            ConvoyProgress with bead counts and completion percentage
        """
        convoy = self._convoys.get(convoy_id)
        if not convoy:
            raise ValueError(f"Convoy {convoy_id} not found")

        pending = 0
        running = 0
        completed = 0
        failed = 0

        for bead_id in convoy.bead_ids:
            bead = await self.bead_store.get(bead_id)
            if not bead:
                continue

            if bead.status == BeadStatus.PENDING:
                pending += 1
            elif bead.status in (BeadStatus.CLAIMED, BeadStatus.RUNNING):
                running += 1
            elif bead.status == BeadStatus.COMPLETED:
                completed += 1
            elif bead.status in (BeadStatus.FAILED, BeadStatus.CANCELLED):
                failed += 1

        total = len(convoy.bead_ids)
        completion_pct = (completed / total * 100) if total > 0 else 0.0

        return ConvoyProgress(
            total_beads=total,
            pending_beads=pending,
            running_beads=running,
            completed_beads=completed,
            failed_beads=failed,
            completion_percentage=completion_pct,
        )

    async def update_convoy_status(self, convoy_id: str) -> ConvoyStatus:
        """
        Update convoy status based on bead states.

        Args:
            convoy_id: The convoy to update

        Returns:
            The new convoy status
        """
        async with self._lock:
            progress = await self.get_convoy_progress(convoy_id)
            convoy = self._convoys[convoy_id]

            if progress.is_complete:
                if progress.failed_beads == 0:
                    convoy.status = ConvoyStatus.COMPLETED
                elif progress.completed_beads == 0:
                    convoy.status = ConvoyStatus.FAILED
                else:
                    convoy.status = ConvoyStatus.PARTIAL
                convoy.completed_at = datetime.now(timezone.utc)
            elif progress.running_beads > 0 or progress.completed_beads > 0:
                convoy.status = ConvoyStatus.ACTIVE

            convoy.updated_at = datetime.now(timezone.utc)
            await self._save_convoy(convoy)

            return convoy.status

    async def list_convoys(
        self,
        status: Optional[ConvoyStatus] = None,
        agent_id: Optional[str] = None,
    ) -> List[Convoy]:
        """
        List convoys with optional filtering.

        Args:
            status: Filter by status
            agent_id: Filter by assigned agent

        Returns:
            List of matching convoys
        """
        convoys = list(self._convoys.values())

        if status:
            convoys = [c for c in convoys if c.status == status]

        if agent_id:
            convoys = [c for c in convoys if agent_id in c.assigned_to]

        return convoys

    async def list_pending_runnable(self) -> List[Convoy]:
        """List pending convoys that can be started (dependencies met)."""
        completed_ids = {c.id for c in self._convoys.values() if c.status == ConvoyStatus.COMPLETED}
        return [
            c
            for c in self._convoys.values()
            if c.status == ConvoyStatus.PENDING and c.can_start(completed_ids)
        ]

    async def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about convoys."""
        convoys = list(self._convoys.values())
        by_status: Dict[str, int] = {}
        total_beads = 0

        for convoy in convoys:
            by_status[convoy.status.value] = by_status.get(convoy.status.value, 0) + 1
            total_beads += len(convoy.bead_ids)

        return {
            "total_convoys": len(convoys),
            "total_beads": total_beads,
            "by_status": by_status,
            "avg_beads_per_convoy": total_beads / len(convoys) if convoys else 0,
        }


# Singleton instance
_default_manager: Optional[ConvoyManager] = None


async def get_convoy_manager(bead_store: BeadStore) -> ConvoyManager:
    """Get the default convoy manager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ConvoyManager(bead_store)
        await _default_manager.initialize()
    return _default_manager


def reset_convoy_manager() -> None:
    """Reset the default manager (for testing)."""
    global _default_manager
    _default_manager = None
