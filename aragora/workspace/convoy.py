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

from aragora.nomic.convoys import ConvoyManager
from aragora.nomic.stores import (
    BeadStore as NomicBeadStore,
    Convoy as NomicConvoy,
    ConvoyStatus as NomicConvoyStatus,
)
from aragora.nomic.stores.paths import should_use_canonical_store
from aragora.nomic.stores.adapters.workspace import (
    nomic_convoy_to_workspace,
    resolve_workspace_convoy_status,
    workspace_convoy_metadata,
    workspace_convoy_status_to_nomic,
)
from aragora.stores.canonical import WorkspaceStores, get_canonical_workspace_stores

logger = logging.getLogger(__name__)


class ConvoyStatus(Enum):
    """Workspace convoy lifecycle status.

    Maps to canonical ``aragora.nomic.stores.ConvoyStatus``:
        CREATED   -> NomicConvoyStatus.PENDING
        ASSIGNING -> NomicConvoyStatus.ACTIVE
        EXECUTING -> NomicConvoyStatus.ACTIVE
        MERGING   -> NomicConvoyStatus.ACTIVE
        DONE      -> NomicConvoyStatus.COMPLETED
        FAILED    -> NomicConvoyStatus.FAILED
        CANCELLED -> NomicConvoyStatus.CANCELLED

    Note: Workspace has finer-grained ACTIVE states (ASSIGNING, EXECUTING, MERGING)
    that all map to NomicConvoyStatus.ACTIVE. The specific sub-state is preserved
    in metadata["workspace_status"].
    """

    CREATED = "created"
    ASSIGNING = "assigning"
    EXECUTING = "executing"
    MERGING = "merging"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"

    def to_nomic(self) -> NomicConvoyStatus:
        """Convert to canonical nomic status."""
        mapping = {
            ConvoyStatus.CREATED: NomicConvoyStatus.PENDING,
            ConvoyStatus.ASSIGNING: NomicConvoyStatus.ACTIVE,
            ConvoyStatus.EXECUTING: NomicConvoyStatus.ACTIVE,
            ConvoyStatus.MERGING: NomicConvoyStatus.ACTIVE,
            ConvoyStatus.DONE: NomicConvoyStatus.COMPLETED,
            ConvoyStatus.FAILED: NomicConvoyStatus.FAILED,
            ConvoyStatus.CANCELLED: NomicConvoyStatus.CANCELLED,
        }
        return mapping[self]

    @classmethod
    def from_nomic(
        cls,
        nomic_status: NomicConvoyStatus,
        workspace_status: str | None = None,
    ) -> "ConvoyStatus":
        """Convert from canonical nomic status.

        If workspace_status is provided (from metadata), use that for
        finer-grained ACTIVE sub-states.
        """
        if workspace_status:
            try:
                return cls(workspace_status)
            except ValueError:
                pass
        mapping = {
            NomicConvoyStatus.PENDING: cls.CREATED,
            NomicConvoyStatus.ACTIVE: cls.EXECUTING,
            NomicConvoyStatus.COMPLETED: cls.DONE,
            NomicConvoyStatus.FAILED: cls.FAILED,
            NomicConvoyStatus.CANCELLED: cls.CANCELLED,
            NomicConvoyStatus.PARTIAL: cls.EXECUTING,
        }
        return mapping.get(nomic_status, cls.CREATED)


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

    # ConvoyRecord protocol properties (cross-layer compatibility)
    # Note: convoy_id field already exists, so we alias to convoy_record_id
    @property
    def convoy_record_id(self) -> str:
        """Protocol: convoy identifier (alias for convoy_id field)."""
        return object.__getattribute__(self, "convoy_id")

    @property
    def convoy_title(self) -> str:
        """Protocol: convoy title (maps from name)."""
        return self.name

    @property
    def convoy_description(self) -> str:
        """Protocol: convoy description."""
        return self.description

    @property
    def convoy_bead_ids(self) -> list[str]:
        """Protocol: bead IDs in convoy."""
        return self.bead_ids

    @property
    def convoy_status_value(self) -> str:
        """Protocol: status enum value."""
        return self.status.value

    @property
    def convoy_created_at(self) -> datetime:
        """Protocol: creation timestamp (converted from float)."""
        from datetime import datetime, timezone

        return datetime.fromtimestamp(self.created_at, tz=timezone.utc)

    @property
    def convoy_updated_at(self) -> datetime:
        """Protocol: last update timestamp (converted from float)."""
        from datetime import datetime, timezone

        return datetime.fromtimestamp(self.updated_at, tz=timezone.utc)

    @property
    def convoy_assigned_agents(self) -> list[str]:
        """Protocol: assigned agent IDs."""
        return self.assigned_agents

    @property
    def convoy_error(self) -> str | None:
        """Protocol: error message if failed."""
        return self.error

    @property
    def convoy_metadata(self) -> dict[str, Any]:
        """Protocol: metadata dictionary."""
        return dict(self.metadata)  # Convert str values to Any

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
        canonical_stores: WorkspaceStores | None = None,
    ) -> None:
        default_use = should_use_canonical_store(default=True) or bool(bead_store)
        self._use_nomic_store = use_nomic_store if use_nomic_store is not None else default_use
        if not self._use_nomic_store:
            logger.warning(
                "Workspace ConvoyTracker fallback store is deprecated; using canonical store."
            )
            self._use_nomic_store = True
        self._canonical_stores = canonical_stores
        self._bead_store = bead_store
        if self._use_nomic_store and self._canonical_stores is None:
            self._canonical_stores = get_canonical_workspace_stores(
                git_enabled=False,
                auto_commit=False,
                bead_store=self._bead_store,
            )
        self._nomic_manager: ConvoyManager | None = None
        self._nomic_initialized = False

    async def _ensure_nomic_manager(self) -> None:
        if self._nomic_manager is None:
            if self._canonical_stores is not None:
                self._nomic_manager = await self._canonical_stores.convoy_manager()
            elif self._bead_store is not None:
                from aragora.nomic.convoys import get_convoy_manager

                self._nomic_manager = await get_convoy_manager(self._bead_store)
        if self._nomic_manager and not self._nomic_initialized:
            await self._nomic_manager.bead_store.initialize()
            await self._nomic_manager.initialize()
            self._nomic_initialized = True

    def _to_nomic_status(self, status: ConvoyStatus) -> NomicConvoyStatus:
        return workspace_convoy_status_to_nomic(status)

    def _from_nomic_status(
        self, status: NomicConvoyStatus, metadata: dict[str, Any]
    ) -> ConvoyStatus:
        return resolve_workspace_convoy_status(status, metadata, ConvoyStatus)

    def _to_nomic_metadata(
        self,
        workspace_id: str,
        rig_id: str,
        name: str,
        description: str,
        status: ConvoyStatus,
        merge_result: dict[str, Any] | None = None,
        error: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return workspace_convoy_metadata(
            workspace_id=workspace_id,
            rig_id=rig_id,
            name=name,
            description=description,
            status=status,
            merge_result=merge_result,
            error=error,
            extra_metadata=extra_metadata,
        )

    def _from_nomic_convoy(self, convoy: NomicConvoy) -> Convoy:
        return nomic_convoy_to_workspace(
            convoy,
            convoy_cls=Convoy,
            status_cls=ConvoyStatus,
        )

    async def create_convoy(
        self,
        workspace_id: str,
        rig_id: str,
        name: str = "",
        description: str = "",
        bead_ids: list[str] | None = None,
        convoy_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        assigned_agents: list[str] | None = None,
    ) -> Convoy:
        """Create a new convoy."""
        await self._ensure_nomic_manager()
        assert self._nomic_manager is not None
        metadata = self._to_nomic_metadata(
            workspace_id=workspace_id,
            rig_id=rig_id,
            name=name,
            description=description,
            status=ConvoyStatus.CREATED,
            extra_metadata=metadata,
        )
        nomic_convoy = await self._nomic_manager.create_convoy(
            title=name or "Untitled Convoy",
            bead_ids=bead_ids or [],
            description=description,
            metadata=metadata,
            convoy_id=convoy_id,
        )
        if assigned_agents:
            nomic_convoy = await self._nomic_manager.update_convoy(
                nomic_convoy.id,
                assigned_to=assigned_agents,
            )
        return self._from_nomic_convoy(nomic_convoy)

    async def get_convoy(self, convoy_id: str) -> Convoy | None:
        """Get a convoy by ID."""
        await self._ensure_nomic_manager()
        assert self._nomic_manager is not None
        nomic_convoy = await self._nomic_manager.get_convoy(convoy_id)
        return self._from_nomic_convoy(nomic_convoy) if nomic_convoy else None

    async def add_beads(self, convoy_id: str, bead_ids: list[str]) -> Convoy | None:
        """Add beads to a convoy."""
        await self._ensure_nomic_manager()
        assert self._nomic_manager is not None
        nomic_convoy = await self._nomic_manager.get_convoy(convoy_id)
        if not nomic_convoy:
            return None
        updated = list(nomic_convoy.bead_ids) + list(bead_ids)
        nomic_convoy = await self._nomic_manager.update_convoy(
            convoy_id,
            bead_ids=updated,
        )
        return self._from_nomic_convoy(nomic_convoy)

    async def start_assigning(self, convoy_id: str) -> Convoy | None:
        """Transition convoy to ASSIGNING state."""
        await self._ensure_nomic_manager()
        assert self._nomic_manager is not None
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

    async def start_executing(
        self,
        convoy_id: str,
        assigned_agents: list[str] | None = None,
    ) -> Convoy | None:
        """Transition convoy to EXECUTING state."""
        await self._ensure_nomic_manager()
        assert self._nomic_manager is not None
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

    async def start_merging(self, convoy_id: str) -> Convoy | None:
        """Transition convoy to MERGING state."""
        await self._ensure_nomic_manager()
        assert self._nomic_manager is not None
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

    async def complete_convoy(
        self,
        convoy_id: str,
        merge_result: dict[str, Any] | None = None,
    ) -> Convoy | None:
        """Mark a convoy as done."""
        await self._ensure_nomic_manager()
        assert self._nomic_manager is not None
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

    async def fail_convoy(self, convoy_id: str, error: str) -> Convoy | None:
        """Mark a convoy as failed."""
        await self._ensure_nomic_manager()
        assert self._nomic_manager is not None
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

    async def cancel_convoy(self, convoy_id: str) -> Convoy | None:
        """Cancel a convoy."""
        await self._ensure_nomic_manager()
        assert self._nomic_manager is not None
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

    async def list_convoys(
        self,
        workspace_id: str | None = None,
        rig_id: str | None = None,
        status: ConvoyStatus | None = None,
        agent_id: str | None = None,
    ) -> list[Convoy]:
        """List convoys with optional filters."""
        await self._ensure_nomic_manager()
        assert self._nomic_manager is not None
        convoys = await self._nomic_manager.list_convoys(agent_id=agent_id)
        results: list[Convoy] = []
        for nomic_convoy in convoys:
            workspace_convoy = self._from_nomic_convoy(nomic_convoy)
            if workspace_id and workspace_convoy.workspace_id != workspace_id:
                continue
            if rig_id and workspace_convoy.rig_id != rig_id:
                continue
            if status and workspace_convoy.status != status:
                continue
            if agent_id and agent_id not in workspace_convoy.assigned_agents:
                continue
            results.append(workspace_convoy)
        return results

    async def update_metadata(
        self,
        convoy_id: str,
        metadata_updates: dict[str, Any] | None = None,
        assigned_agents: list[str] | None = None,
    ) -> Convoy | None:
        """Update convoy metadata and assigned agents."""
        await self._ensure_nomic_manager()
        assert self._nomic_manager is not None
        nomic_convoy = await self._nomic_manager.update_convoy(
            convoy_id,
            metadata_updates=metadata_updates,
            assigned_to=assigned_agents,
        )
        return self._from_nomic_convoy(nomic_convoy)

    async def get_stats(self) -> dict[str, Any]:
        """Get convoy tracker statistics."""
        await self._ensure_nomic_manager()
        assert self._nomic_manager is not None
        convoys = await self._nomic_manager.list_convoys()
        by_status: dict[str, int] = {}
        for convoy in convoys:
            workspace_convoy = self._from_nomic_convoy(convoy)
            key = workspace_convoy.status.value
            by_status[key] = by_status.get(key, 0) + 1
        return {"total_convoys": len(convoys), "by_status": by_status}
