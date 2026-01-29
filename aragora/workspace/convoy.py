"""
Convoy Tracker - Work batch lifecycle management.

A convoy bundles multiple beads (work items) into a tracked batch.
It manages the lifecycle of the batch from creation through assignment,
execution, merge, and completion.

This is the Gastown equivalent of a bundled work order.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ConvoyStatus(Enum):
    """Convoy lifecycle status."""

    CREATED = "created"
    ASSIGNING = "assigning"
    EXECUTING = "executing"
    MERGING = "merging"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Convoy:
    """A batch of related work items tracked as a unit."""

    convoy_id: str
    workspace_id: str
    rig_id: str
    name: str = ""
    description: str = ""
    status: ConvoyStatus = ConvoyStatus.CREATED
    bead_ids: list[str] = field(default_factory=list)
    assigned_agents: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    merge_result: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def total_beads(self) -> int:
        """Total number of beads in this convoy."""
        return len(self.bead_ids)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Convoy:
        """Deserialize from dictionary."""
        data = dict(data)
        if isinstance(data.get("status"), str):
            data["status"] = ConvoyStatus(data["status"])
        return cls(**data)


class ConvoyTracker:
    """
    Tracks convoy lifecycle from creation through completion.

    Features:
    - Create convoys with a set of beads
    - Track convoy progress (beads completed vs total)
    - State machine: created → assigning → executing → merging → done
    - Automatic status transitions based on bead completion
    """

    def __init__(self) -> None:
        self._convoys: dict[str, Convoy] = {}

    async def create_convoy(
        self,
        workspace_id: str,
        rig_id: str,
        name: str = "",
        description: str = "",
        bead_ids: list[str] | None = None,
        convoy_id: str | None = None,
    ) -> Convoy:
        """Create a new convoy."""
        if convoy_id is None:
            import hashlib

            convoy_id = f"cv-{hashlib.sha256(str(time.time()).encode()).hexdigest()[:5]}"

        convoy = Convoy(
            convoy_id=convoy_id,
            workspace_id=workspace_id,
            rig_id=rig_id,
            name=name,
            description=description,
            bead_ids=bead_ids or [],
        )
        self._convoys[convoy_id] = convoy
        logger.info(f"Created convoy {convoy_id} with {len(convoy.bead_ids)} beads")
        return convoy

    async def get_convoy(self, convoy_id: str) -> Convoy | None:
        """Get a convoy by ID."""
        return self._convoys.get(convoy_id)

    async def add_beads(self, convoy_id: str, bead_ids: list[str]) -> Convoy | None:
        """Add beads to a convoy."""
        convoy = self._convoys.get(convoy_id)
        if not convoy:
            return None
        convoy.bead_ids.extend(bead_ids)
        convoy.updated_at = time.time()
        return convoy

    async def start_assigning(self, convoy_id: str) -> Convoy | None:
        """Transition convoy to ASSIGNING state."""
        convoy = self._convoys.get(convoy_id)
        if not convoy:
            return None
        if convoy.status != ConvoyStatus.CREATED:
            logger.warning(f"Cannot start assigning convoy {convoy_id} in state {convoy.status}")
            return convoy
        convoy.status = ConvoyStatus.ASSIGNING
        convoy.updated_at = time.time()
        return convoy

    async def start_executing(
        self,
        convoy_id: str,
        assigned_agents: list[str] | None = None,
    ) -> Convoy | None:
        """Transition convoy to EXECUTING state."""
        convoy = self._convoys.get(convoy_id)
        if not convoy:
            return None
        convoy.status = ConvoyStatus.EXECUTING
        convoy.started_at = time.time()
        convoy.updated_at = time.time()
        if assigned_agents:
            convoy.assigned_agents = assigned_agents
        return convoy

    async def start_merging(self, convoy_id: str) -> Convoy | None:
        """Transition convoy to MERGING state."""
        convoy = self._convoys.get(convoy_id)
        if not convoy:
            return None
        convoy.status = ConvoyStatus.MERGING
        convoy.updated_at = time.time()
        return convoy

    async def complete_convoy(
        self,
        convoy_id: str,
        merge_result: dict[str, Any] | None = None,
    ) -> Convoy | None:
        """Mark a convoy as done."""
        convoy = self._convoys.get(convoy_id)
        if not convoy:
            return None
        convoy.status = ConvoyStatus.DONE
        convoy.completed_at = time.time()
        convoy.updated_at = time.time()
        convoy.merge_result = merge_result
        logger.info(f"Convoy {convoy_id} completed")
        return convoy

    async def fail_convoy(self, convoy_id: str, error: str) -> Convoy | None:
        """Mark a convoy as failed."""
        convoy = self._convoys.get(convoy_id)
        if not convoy:
            return None
        convoy.status = ConvoyStatus.FAILED
        convoy.completed_at = time.time()
        convoy.updated_at = time.time()
        convoy.error = error
        logger.error(f"Convoy {convoy_id} failed: {error}")
        return convoy

    async def cancel_convoy(self, convoy_id: str) -> Convoy | None:
        """Cancel a convoy."""
        convoy = self._convoys.get(convoy_id)
        if not convoy:
            return None
        convoy.status = ConvoyStatus.CANCELLED
        convoy.completed_at = time.time()
        convoy.updated_at = time.time()
        return convoy

    async def list_convoys(
        self,
        workspace_id: str | None = None,
        rig_id: str | None = None,
        status: ConvoyStatus | None = None,
    ) -> list[Convoy]:
        """List convoys with optional filters."""
        results = []
        for convoy in self._convoys.values():
            if workspace_id and convoy.workspace_id != workspace_id:
                continue
            if rig_id and convoy.rig_id != rig_id:
                continue
            if status and convoy.status != status:
                continue
            results.append(convoy)
        return results

    async def get_stats(self) -> dict[str, Any]:
        """Get convoy tracker statistics."""
        by_status: dict[str, int] = {}
        for convoy in self._convoys.values():
            key = convoy.status.value
            by_status[key] = by_status.get(key, 0) + 1
        return {
            "total_convoys": len(self._convoys),
            "by_status": by_status,
        }
