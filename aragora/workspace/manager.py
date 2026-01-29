"""
Workspace Manager - Gastown-style project management.

Coordinates rigs (per-repo containers), convoys (work batches), and
beads (atomic work units) to provide developer orchestration capabilities.

This is the top-level entry point for the Gastown extension, tying
together the Agent Fabric, Hook Manager, Convoy Tracker, and Bead
Manager into a unified workspace management interface.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

from aragora.workspace.bead import Bead, BeadManager, BeadStatus
from aragora.workspace.convoy import Convoy, ConvoyStatus, ConvoyTracker
from aragora.workspace.rig import Rig, RigConfig, RigStatus

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """
    Top-level workspace management for Gastown parity.

    Manages the lifecycle of rigs (projects), convoys (work batches),
    and beads (work items). Integrates with the Agent Fabric for
    agent scheduling and the Hook Manager for crash-resilient persistence.

    Usage:
        ws = WorkspaceManager(workspace_root="/path/to/projects")
        rig = await ws.create_rig("my-project", config=RigConfig(repo_url="..."))
        convoy = await ws.create_convoy(rig.rig_id, beads=[...])
        await ws.start_convoy(convoy.convoy_id)
    """

    def __init__(
        self,
        workspace_root: str = ".",
        workspace_id: str = "default",
    ) -> None:
        self._workspace_root = Path(workspace_root)
        self._workspace_id = workspace_id

        # Sub-managers
        self._convoy_tracker = ConvoyTracker()
        self._bead_manager = BeadManager(storage_dir=self._workspace_root / ".aragora_beads")

        # Rig storage
        self._rigs: dict[str, Rig] = {}

    # =========================================================================
    # Rig management
    # =========================================================================

    async def create_rig(
        self,
        name: str,
        config: RigConfig | None = None,
        rig_id: str | None = None,
    ) -> Rig:
        """
        Create a new rig (per-repo project container).

        Args:
            name: Human-readable rig name.
            config: Rig configuration.
            rig_id: Optional custom ID (auto-generated if None).

        Returns:
            The created Rig.
        """
        if rig_id is None:
            rig_id = f"rig-{hashlib.sha256(f'{name}-{time.time()}'.encode()).hexdigest()[:5]}"

        rig = Rig(
            rig_id=rig_id,
            name=name,
            workspace_id=self._workspace_id,
            config=config or RigConfig(),
            status=RigStatus.READY,
        )
        self._rigs[rig_id] = rig
        logger.info(f"Created rig {rig_id} ({name})")
        return rig

    async def get_rig(self, rig_id: str) -> Rig | None:
        """Get a rig by ID."""
        return self._rigs.get(rig_id)

    async def list_rigs(
        self,
        status: RigStatus | None = None,
    ) -> list[Rig]:
        """List all rigs with optional status filter."""
        results = []
        for rig in self._rigs.values():
            if status and rig.status != status:
                continue
            results.append(rig)
        return results

    async def assign_agent_to_rig(self, rig_id: str, agent_id: str) -> Rig | None:
        """Assign an agent to a rig."""
        rig = self._rigs.get(rig_id)
        if not rig:
            return None
        if len(rig.assigned_agents) >= rig.config.max_agents:
            raise ValueError(f"Rig {rig_id} at max capacity ({rig.config.max_agents} agents)")
        if agent_id not in rig.assigned_agents:
            rig.assigned_agents.append(agent_id)
            rig.updated_at = time.time()
        return rig

    async def remove_agent_from_rig(self, rig_id: str, agent_id: str) -> Rig | None:
        """Remove an agent from a rig."""
        rig = self._rigs.get(rig_id)
        if not rig:
            return None
        if agent_id in rig.assigned_agents:
            rig.assigned_agents.remove(agent_id)
            rig.updated_at = time.time()
        return rig

    async def stop_rig(self, rig_id: str) -> Rig | None:
        """Stop a rig and drain its agents."""
        rig = self._rigs.get(rig_id)
        if not rig:
            return None
        rig.status = RigStatus.DRAINING
        rig.updated_at = time.time()
        return rig

    async def delete_rig(self, rig_id: str) -> bool:
        """Delete a rig."""
        if rig_id in self._rigs:
            del self._rigs[rig_id]
            return True
        return False

    # =========================================================================
    # Convoy management
    # =========================================================================

    async def create_convoy(
        self,
        rig_id: str,
        name: str = "",
        description: str = "",
        bead_specs: list[dict[str, Any]] | None = None,
    ) -> Convoy:
        """
        Create a convoy with optional bead specifications.

        Args:
            rig_id: Rig this convoy belongs to.
            name: Convoy name.
            description: Convoy description.
            bead_specs: List of bead specifications (title, description, payload).

        Returns:
            The created Convoy with beads.
        """
        rig = self._rigs.get(rig_id)
        if not rig:
            raise ValueError(f"Rig {rig_id} not found")

        # Create beads first
        bead_ids = []
        convoy_id = f"cv-{hashlib.sha256(str(time.time()).encode()).hexdigest()[:5]}"

        for spec in bead_specs or []:
            bead = await self._bead_manager.create_bead(
                convoy_id=convoy_id,
                workspace_id=self._workspace_id,
                title=spec.get("title", ""),
                description=spec.get("description", ""),
                payload=spec.get("payload", {}),
                depends_on=spec.get("depends_on", []),
            )
            bead_ids.append(bead.bead_id)

        # Create convoy
        convoy = await self._convoy_tracker.create_convoy(
            workspace_id=self._workspace_id,
            rig_id=rig_id,
            name=name,
            description=description,
            bead_ids=bead_ids,
            convoy_id=convoy_id,
        )

        # Track convoy in rig
        rig.active_convoys.append(convoy.convoy_id)

        return convoy

    async def get_convoy(self, convoy_id: str) -> Convoy | None:
        """Get a convoy by ID."""
        return await self._convoy_tracker.get_convoy(convoy_id)

    async def get_convoy_status(self, convoy_id: str) -> dict[str, Any] | None:
        """Get detailed convoy status including bead progress."""
        convoy = await self._convoy_tracker.get_convoy(convoy_id)
        if not convoy:
            return None

        beads = await self._bead_manager.list_beads(convoy_id=convoy_id)
        done = sum(1 for b in beads if b.status == BeadStatus.DONE)
        failed = sum(1 for b in beads if b.status == BeadStatus.FAILED)
        running = sum(1 for b in beads if b.status == BeadStatus.RUNNING)
        pending = sum(1 for b in beads if b.status in (BeadStatus.PENDING, BeadStatus.ASSIGNED))

        return {
            "convoy_id": convoy.convoy_id,
            "status": convoy.status.value,
            "name": convoy.name,
            "total_beads": len(beads),
            "done": done,
            "failed": failed,
            "running": running,
            "pending": pending,
            "progress_percent": (done / len(beads) * 100) if beads else 0,
            "assigned_agents": convoy.assigned_agents,
        }

    async def start_convoy(self, convoy_id: str) -> Convoy | None:
        """Start executing a convoy."""
        return await self._convoy_tracker.start_executing(convoy_id)

    async def complete_convoy(
        self,
        convoy_id: str,
        merge_result: dict[str, Any] | None = None,
    ) -> Convoy | None:
        """Mark a convoy as completed."""
        return await self._convoy_tracker.complete_convoy(convoy_id, merge_result)

    async def list_convoys(
        self,
        rig_id: str | None = None,
        status: ConvoyStatus | None = None,
    ) -> list[Convoy]:
        """List convoys."""
        return await self._convoy_tracker.list_convoys(
            workspace_id=self._workspace_id,
            rig_id=rig_id,
            status=status,
        )

    # =========================================================================
    # Bead management (pass-through to BeadManager)
    # =========================================================================

    async def get_bead(self, bead_id: str) -> Bead | None:
        """Get a bead by ID."""
        return await self._bead_manager.get_bead(bead_id)

    async def assign_bead(self, bead_id: str, agent_id: str) -> Bead | None:
        """Assign a bead to an agent."""
        return await self._bead_manager.assign_bead(bead_id, agent_id)

    async def complete_bead(
        self,
        bead_id: str,
        result: dict[str, Any] | None = None,
    ) -> Bead | None:
        """Complete a bead."""
        return await self._bead_manager.complete_bead(bead_id, result)

    async def fail_bead(self, bead_id: str, error: str) -> Bead | None:
        """Fail a bead."""
        return await self._bead_manager.fail_bead(bead_id, error)

    async def get_ready_beads(self, convoy_id: str) -> list[Bead]:
        """Get beads ready for execution (dependencies met)."""
        return await self._bead_manager.get_ready_beads(convoy_id)

    # =========================================================================
    # Stats
    # =========================================================================

    async def get_stats(self) -> dict[str, Any]:
        """Get workspace manager statistics."""
        convoy_stats = await self._convoy_tracker.get_stats()
        return {
            "workspace_id": self._workspace_id,
            "total_rigs": len(self._rigs),
            "rigs_by_status": {
                status.value: sum(1 for r in self._rigs.values() if r.status == status)
                for status in RigStatus
            },
            "convoys": convoy_stats,
        }
