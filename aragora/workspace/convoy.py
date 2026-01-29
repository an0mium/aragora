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

from datetime import datetime, timezone

from aragora.nomic.beads import BeadStore as NomicBeadStore
from aragora.nomic.convoys import (
    Convoy as NomicConvoy,
    ConvoyManager as NomicConvoyManager,
    ConvoyStatus as NomicConvoyStatus,
)

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

    def __init__(
        self,
        bead_store: NomicBeadStore | None = None,
        use_nomic_store: bool | None = None,
    ) -> None:
        self._convoys: dict[str, Convoy] = {}
        self._use_nomic_store = use_nomic_store if use_nomic_store is not None else bool(bead_store)
        self._nomic_manager = NomicConvoyManager(bead_store) if bead_store else None
        self._nomic_initialized = False

    async def _ensure_nomic_manager(self) -> None:
        if self._nomic_manager and not self._nomic_initialized:
            await self._nomic_manager.bead_store.initialize()
            await self._nomic_manager.initialize()
            self._nomic_initialized = True

    def _to_nomic_status(self, status: ConvoyStatus) -> NomicConvoyStatus:
        return {
            ConvoyStatus.CREATED: NomicConvoyStatus.PENDING,
            ConvoyStatus.ASSIGNING: NomicConvoyStatus.ACTIVE,
            ConvoyStatus.EXECUTING: NomicConvoyStatus.ACTIVE,
            ConvoyStatus.MERGING: NomicConvoyStatus.ACTIVE,
            ConvoyStatus.DONE: NomicConvoyStatus.COMPLETED,
            ConvoyStatus.FAILED: NomicConvoyStatus.FAILED,
            ConvoyStatus.CANCELLED: NomicConvoyStatus.CANCELLED,
        }.get(status, NomicConvoyStatus.PENDING)

    def _from_nomic_status(
        self, status: NomicConvoyStatus, metadata: dict[str, Any]
    ) -> ConvoyStatus:
        stored = metadata.get("workspace_status")
        if isinstance(stored, str):
            try:
                return ConvoyStatus(stored)
            except ValueError:
                pass
        return {
            NomicConvoyStatus.PENDING: ConvoyStatus.CREATED,
            NomicConvoyStatus.ACTIVE: ConvoyStatus.EXECUTING,
            NomicConvoyStatus.COMPLETED: ConvoyStatus.DONE,
            NomicConvoyStatus.FAILED: ConvoyStatus.FAILED,
            NomicConvoyStatus.CANCELLED: ConvoyStatus.CANCELLED,
            NomicConvoyStatus.PARTIAL: ConvoyStatus.EXECUTING,
        }.get(status, ConvoyStatus.CREATED)

    def _to_nomic_metadata(
        self,
        workspace_id: str,
        rig_id: str,
        name: str,
        description: str,
        status: ConvoyStatus,
        merge_result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "workspace_id": workspace_id,
            "rig_id": rig_id,
            "workspace_name": name,
            "workspace_description": description,
            "workspace_status": status.value,
        }
        if merge_result is not None:
            metadata["merge_result"] = merge_result
        if error:
            metadata["error"] = error
        return metadata

    def _from_nomic_convoy(self, convoy: NomicConvoy) -> Convoy:
        metadata = convoy.metadata or {}
        status = self._from_nomic_status(convoy.status, metadata)
        return Convoy(
            convoy_id=convoy.id,
            workspace_id=metadata.get("workspace_id", ""),
            rig_id=metadata.get("rig_id", ""),
            name=metadata.get("workspace_name", convoy.title),
            description=metadata.get("workspace_description", convoy.description),
            status=status,
            bead_ids=list(convoy.bead_ids),
            assigned_agents=list(convoy.assigned_to),
            created_at=convoy.created_at.timestamp(),
            updated_at=convoy.updated_at.timestamp(),
            started_at=convoy.started_at.timestamp() if convoy.started_at else None,
            completed_at=convoy.completed_at.timestamp() if convoy.completed_at else None,
            merge_result=metadata.get("merge_result"),
            error=convoy.error_message or metadata.get("error"),
            metadata=metadata if isinstance(metadata, dict) else {},
        )

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
        if self._use_nomic_store and self._nomic_manager:
            await self._ensure_nomic_manager()
            metadata = self._to_nomic_metadata(
                workspace_id=workspace_id,
                rig_id=rig_id,
                name=name,
                description=description,
                status=ConvoyStatus.CREATED,
            )
            nomic_convoy = await self._nomic_manager.create_convoy(
                title=name or "Untitled Convoy",
                bead_ids=bead_ids or [],
                description=description,
                metadata=metadata,
                convoy_id=convoy_id,
            )
            return self._from_nomic_convoy(nomic_convoy)

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
        if self._use_nomic_store and self._nomic_manager:
            await self._ensure_nomic_manager()
            nomic_convoy = await self._nomic_manager.get_convoy(convoy_id)
            return self._from_nomic_convoy(nomic_convoy) if nomic_convoy else None
        return self._convoys.get(convoy_id)

    async def add_beads(self, convoy_id: str, bead_ids: list[str]) -> Convoy | None:
        """Add beads to a convoy."""
        if self._use_nomic_store and self._nomic_manager:
            await self._ensure_nomic_manager()
            nomic_convoy = await self._nomic_manager.get_convoy(convoy_id)
            if not nomic_convoy:
                return None
            updated = list(nomic_convoy.bead_ids) + list(bead_ids)
            nomic_convoy = await self._nomic_manager.update_convoy(
                convoy_id,
                bead_ids=updated,
            )
            return self._from_nomic_convoy(nomic_convoy)

        convoy = self._convoys.get(convoy_id)
        if not convoy:
            return None
        convoy.bead_ids.extend(bead_ids)
        convoy.updated_at = time.time()
        return convoy

    async def start_assigning(self, convoy_id: str) -> Convoy | None:
        """Transition convoy to ASSIGNING state."""
        if self._use_nomic_store and self._nomic_manager:
            await self._ensure_nomic_manager()
            nomic_convoy = await self._nomic_manager.get_convoy(convoy_id)
            if not nomic_convoy:
                return None
            metadata = dict(nomic_convoy.metadata or {})
            metadata["workspace_status"] = ConvoyStatus.ASSIGNING.value
            nomic_convoy = await self._nomic_manager.update_convoy(
                convoy_id,
                metadata_updates=metadata,
            )
            return self._from_nomic_convoy(nomic_convoy)

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
        if self._use_nomic_store and self._nomic_manager:
            await self._ensure_nomic_manager()
            nomic_convoy = await self._nomic_manager.get_convoy(convoy_id)
            if not nomic_convoy:
                return None
            metadata = dict(nomic_convoy.metadata or {})
            metadata["workspace_status"] = ConvoyStatus.EXECUTING.value
            started_at = datetime.now(timezone.utc)
            nomic_convoy = await self._nomic_manager.update_convoy(
                convoy_id,
                status=NomicConvoyStatus.ACTIVE,
                metadata_updates=metadata,
                assigned_to=assigned_agents or nomic_convoy.assigned_to,
                started_at=started_at,
            )
            return self._from_nomic_convoy(nomic_convoy)

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
        if self._use_nomic_store and self._nomic_manager:
            await self._ensure_nomic_manager()
            nomic_convoy = await self._nomic_manager.get_convoy(convoy_id)
            if not nomic_convoy:
                return None
            metadata = dict(nomic_convoy.metadata or {})
            metadata["workspace_status"] = ConvoyStatus.MERGING.value
            nomic_convoy = await self._nomic_manager.update_convoy(
                convoy_id,
                metadata_updates=metadata,
            )
            return self._from_nomic_convoy(nomic_convoy)

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
        if self._use_nomic_store and self._nomic_manager:
            await self._ensure_nomic_manager()
            nomic_convoy = await self._nomic_manager.get_convoy(convoy_id)
            if not nomic_convoy:
                return None
            metadata = dict(nomic_convoy.metadata or {})
            metadata["workspace_status"] = ConvoyStatus.DONE.value
            if merge_result is not None:
                metadata["merge_result"] = merge_result
            completed_at = datetime.now(timezone.utc)
            nomic_convoy = await self._nomic_manager.update_convoy(
                convoy_id,
                status=NomicConvoyStatus.COMPLETED,
                metadata_updates=metadata,
                completed_at=completed_at,
            )
            logger.info(f"Convoy {convoy_id} completed")
            return self._from_nomic_convoy(nomic_convoy)

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
        if self._use_nomic_store and self._nomic_manager:
            await self._ensure_nomic_manager()
            nomic_convoy = await self._nomic_manager.get_convoy(convoy_id)
            if not nomic_convoy:
                return None
            metadata = dict(nomic_convoy.metadata or {})
            metadata["workspace_status"] = ConvoyStatus.FAILED.value
            metadata["error"] = error
            completed_at = datetime.now(timezone.utc)
            nomic_convoy = await self._nomic_manager.update_convoy(
                convoy_id,
                status=NomicConvoyStatus.FAILED,
                metadata_updates=metadata,
                completed_at=completed_at,
                error_message=error,
            )
            logger.error(f"Convoy {convoy_id} failed: {error}")
            return self._from_nomic_convoy(nomic_convoy)

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
        if self._use_nomic_store and self._nomic_manager:
            await self._ensure_nomic_manager()
            nomic_convoy = await self._nomic_manager.get_convoy(convoy_id)
            if not nomic_convoy:
                return None
            metadata = dict(nomic_convoy.metadata or {})
            metadata["workspace_status"] = ConvoyStatus.CANCELLED.value
            completed_at = datetime.now(timezone.utc)
            nomic_convoy = await self._nomic_manager.update_convoy(
                convoy_id,
                status=NomicConvoyStatus.CANCELLED,
                metadata_updates=metadata,
                completed_at=completed_at,
            )
            return self._from_nomic_convoy(nomic_convoy)

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
        if self._use_nomic_store and self._nomic_manager:
            await self._ensure_nomic_manager()
            convoys = await self._nomic_manager.list_convoys()
            results: list[Convoy] = []
            for nomic_convoy in convoys:
                workspace_convoy = self._from_nomic_convoy(nomic_convoy)
                if workspace_id and workspace_convoy.workspace_id != workspace_id:
                    continue
                if rig_id and workspace_convoy.rig_id != rig_id:
                    continue
                if status and workspace_convoy.status != status:
                    continue
                results.append(workspace_convoy)
            return results

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
        if self._use_nomic_store and self._nomic_manager:
            await self._ensure_nomic_manager()
            convoys = await self._nomic_manager.list_convoys()
            by_status: dict[str, int] = {}
            for convoy in convoys:
                workspace_convoy = self._from_nomic_convoy(convoy)
                key = workspace_convoy.status.value
                by_status[key] = by_status.get(key, 0) + 1
            return {"total_convoys": len(convoys), "by_status": by_status}

        local_by_status: dict[str, int] = {}
        for convoy in self._convoys.values():
            key = convoy.status.value
            local_by_status[key] = local_by_status.get(key, 0) + 1
        return {
            "total_convoys": len(self._convoys),
            "by_status": local_by_status,
        }
