"""
Canonical Nomic APIs for Convoys and Beads.

These facades provide a stable API surface for orchestration primitives.
Adapters (e.g., Gastown) should call into these APIs rather than
reimplementing core behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from aragora.nomic.beads import (
    Bead,
    BeadPriority,
    BeadStatus,
    BeadStore,
    BeadType,
    create_bead_store,
)
from aragora.nomic.convoys import (
    Convoy,
    ConvoyManager,
    ConvoyPriority,
    ConvoyStatus,
    get_convoy_manager,
)


@runtime_checkable
class BeadAPI(Protocol):
    async def create_bead(
        self,
        *,
        bead_type: BeadType,
        title: str,
        description: str = "",
        parent_id: str | None = None,
        dependencies: list[str] | None = None,
        priority: BeadPriority = BeadPriority.NORMAL,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Bead: ...

    async def get_bead(self, bead_id: str) -> Bead | None: ...
    async def update_bead(self, bead: Bead) -> Bead: ...
    async def list_beads(self, *, status: BeadStatus | None = None) -> list[Bead]: ...


@runtime_checkable
class ConvoyAPI(Protocol):
    async def create_convoy(
        self,
        *,
        title: str,
        bead_ids: list[str],
        description: str = "",
        priority: ConvoyPriority = ConvoyPriority.NORMAL,
        dependencies: list[str] | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        convoy_id: str | None = None,
    ) -> Convoy: ...

    async def assign_convoy(self, convoy_id: str, agent_ids: list[str]) -> bool: ...
    async def get_convoy(self, convoy_id: str) -> Convoy | None: ...
    async def list_convoys(
        self,
        *,
        status: ConvoyStatus | None = None,
        agent_id: str | None = None,
    ) -> list[Convoy]: ...


@dataclass
class NomicBeadAPI(BeadAPI):
    """Canonical bead API backed by BeadStore."""

    bead_store: BeadStore | None = None
    bead_dir: str = ".beads"
    git_enabled: bool = True
    auto_commit: bool = False

    async def _store(self) -> BeadStore:
        if self.bead_store is None:
            self.bead_store = await create_bead_store(
                bead_dir=self.bead_dir,
                git_enabled=self.git_enabled,
                auto_commit=self.auto_commit,
            )
        return self.bead_store

    async def create_bead(
        self,
        *,
        bead_type: BeadType,
        title: str,
        description: str = "",
        parent_id: str | None = None,
        dependencies: list[str] | None = None,
        priority: BeadPriority = BeadPriority.NORMAL,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Bead:
        bead = Bead.create(
            bead_type=bead_type,
            title=title,
            description=description,
            parent_id=parent_id,
            dependencies=dependencies,
            priority=priority,
            tags=tags,
            metadata=metadata,
        )
        store = await self._store()
        await store.create(bead)
        return bead

    async def get_bead(self, bead_id: str) -> Bead | None:
        store = await self._store()
        return await store.get(bead_id)

    async def update_bead(self, bead: Bead) -> Bead:
        store = await self._store()
        return await store.update(bead)

    async def list_beads(self, *, status: BeadStatus | None = None) -> list[Bead]:
        store = await self._store()
        if status is None:
            return await store.list_all()
        return await store.list_by_status(status)


@dataclass
class NomicConvoyAPI(ConvoyAPI):
    """Canonical convoy API backed by ConvoyManager."""

    bead_store: BeadStore | None = None
    convoy_manager: ConvoyManager | None = None
    bead_dir: str = ".beads"
    git_enabled: bool = True
    auto_commit: bool = False

    async def _manager(self) -> ConvoyManager:
        if self.convoy_manager is None:
            if self.bead_store is None:
                self.bead_store = await create_bead_store(
                    bead_dir=self.bead_dir,
                    git_enabled=self.git_enabled,
                    auto_commit=self.auto_commit,
                )
            self.convoy_manager = await get_convoy_manager(self.bead_store)
        return self.convoy_manager

    async def create_convoy(
        self,
        *,
        title: str,
        bead_ids: list[str],
        description: str = "",
        priority: ConvoyPriority = ConvoyPriority.NORMAL,
        dependencies: list[str] | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        convoy_id: str | None = None,
    ) -> Convoy:
        manager = await self._manager()
        return await manager.create_convoy(
            title=title,
            bead_ids=bead_ids,
            description=description,
            priority=priority,
            dependencies=dependencies,
            tags=tags,
            metadata=metadata,
            convoy_id=convoy_id,
        )

    async def assign_convoy(self, convoy_id: str, agent_ids: list[str]) -> bool:
        manager = await self._manager()
        return await manager.assign_convoy(convoy_id, agent_ids)

    async def get_convoy(self, convoy_id: str) -> Convoy | None:
        manager = await self._manager()
        return await manager.get_convoy(convoy_id)

    async def list_convoys(
        self,
        *,
        status: ConvoyStatus | None = None,
        agent_id: str | None = None,
    ) -> list[Convoy]:
        manager = await self._manager()
        return await manager.list_convoys(status=status, agent_id=agent_id)
