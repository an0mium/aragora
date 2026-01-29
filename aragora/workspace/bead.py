"""
Bead Manager - Atomic work units with JSONL-backed tracking.

Beads are the Gastown equivalent of atomic tasks: each bead represents
a single unit of work with a prefix + 5-character ID. Beads are tracked
via JSONL files for crash-resilient state management.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class BeadStatus(Enum):
    """Bead lifecycle status."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Bead:
    """An atomic unit of work (Gastown bead equivalent)."""

    bead_id: str  # prefix + 5-char ID
    convoy_id: str
    workspace_id: str
    title: str = ""
    description: str = ""
    status: BeadStatus = BeadStatus.PENDING
    assigned_agent: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    git_ref: str | None = None
    depends_on: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Bead:
        """Deserialize from dictionary."""
        data = dict(data)
        if isinstance(data.get("status"), str):
            data["status"] = BeadStatus(data["status"])
        return cls(**data)


def generate_bead_id(prefix: str = "bd") -> str:
    """Generate a bead ID with prefix + 5-char hash (Gastown convention)."""
    h = hashlib.sha256(f"{time.time()}-{id(prefix)}".encode()).hexdigest()[:5]
    return f"{prefix}-{h}"


class BeadManager:
    """
    Manages beads (atomic work units) with JSONL-backed persistence.

    Features:
    - Create beads with auto-generated IDs
    - Track bead lifecycle (pending → assigned → running → done/failed)
    - Persist bead state as JSONL for crash recovery
    - List and filter beads by status, convoy, agent
    - Dependency tracking between beads
    """

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        self._beads: dict[str, Bead] = {}
        self._storage_dir = Path(storage_dir) if storage_dir else None
        if self._storage_dir:
            self._storage_dir.mkdir(parents=True, exist_ok=True)

    async def create_bead(
        self,
        convoy_id: str,
        workspace_id: str,
        title: str = "",
        description: str = "",
        payload: dict[str, Any] | None = None,
        depends_on: list[str] | None = None,
        prefix: str = "bd",
    ) -> Bead:
        """Create a new bead."""
        bead_id = generate_bead_id(prefix)
        bead = Bead(
            bead_id=bead_id,
            convoy_id=convoy_id,
            workspace_id=workspace_id,
            title=title,
            description=description,
            payload=payload or {},
            depends_on=depends_on or [],
        )
        self._beads[bead_id] = bead
        self._persist(bead)
        return bead

    async def get_bead(self, bead_id: str) -> Bead | None:
        """Get a bead by ID."""
        return self._beads.get(bead_id)

    async def assign_bead(self, bead_id: str, agent_id: str) -> Bead | None:
        """Assign a bead to an agent."""
        bead = self._beads.get(bead_id)
        if not bead:
            return None
        bead.assigned_agent = agent_id
        bead.status = BeadStatus.ASSIGNED
        bead.updated_at = time.time()
        self._persist(bead)
        return bead

    async def start_bead(self, bead_id: str) -> Bead | None:
        """Mark a bead as running."""
        bead = self._beads.get(bead_id)
        if not bead:
            return None
        bead.status = BeadStatus.RUNNING
        bead.started_at = time.time()
        bead.updated_at = time.time()
        self._persist(bead)
        return bead

    async def complete_bead(
        self,
        bead_id: str,
        result: dict[str, Any] | None = None,
    ) -> Bead | None:
        """Mark a bead as done with an optional result."""
        bead = self._beads.get(bead_id)
        if not bead:
            return None
        bead.status = BeadStatus.DONE
        bead.result = result
        bead.completed_at = time.time()
        bead.updated_at = time.time()
        self._persist(bead)
        return bead

    async def fail_bead(self, bead_id: str, error: str) -> Bead | None:
        """Mark a bead as failed."""
        bead = self._beads.get(bead_id)
        if not bead:
            return None
        bead.status = BeadStatus.FAILED
        bead.error = error
        bead.completed_at = time.time()
        bead.updated_at = time.time()
        self._persist(bead)
        return bead

    async def list_beads(
        self,
        convoy_id: str | None = None,
        status: BeadStatus | None = None,
        agent_id: str | None = None,
    ) -> list[Bead]:
        """List beads with optional filters."""
        results = []
        for bead in self._beads.values():
            if convoy_id and bead.convoy_id != convoy_id:
                continue
            if status and bead.status != status:
                continue
            if agent_id and bead.assigned_agent != agent_id:
                continue
            results.append(bead)
        return results

    async def get_ready_beads(self, convoy_id: str) -> list[Bead]:
        """Get beads that are ready to execute (dependencies met)."""
        ready = []
        for bead in self._beads.values():
            if bead.convoy_id != convoy_id:
                continue
            if bead.status != BeadStatus.PENDING:
                continue
            # Check dependencies
            deps_met = all(
                self._beads.get(dep_id) and self._beads[dep_id].status == BeadStatus.DONE
                for dep_id in bead.depends_on
            )
            if deps_met:
                ready.append(bead)
        return ready

    def _persist(self, bead: Bead) -> None:
        """Persist bead state to JSONL."""
        if not self._storage_dir:
            return
        path = self._storage_dir / f"{bead.convoy_id}.jsonl"
        line = json.dumps(bead.to_dict(), default=str)
        with open(path, "a") as f:
            f.write(line + "\n")
