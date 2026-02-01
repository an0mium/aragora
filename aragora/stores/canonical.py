"""
Canonical store access helpers.

This module centralizes access to the canonical persistence layers for:
- Convoys/Beads/Workspaces (Nomic stores)
- Gateway/Inbox (Gateway store + Unified Inbox store)

Use these helpers instead of constructing ad-hoc store instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aragora.gateway.persistence import GatewayStore
    from aragora.nomic.beads import BeadStore
    from aragora.nomic.convoys import ConvoyManager
    from aragora.storage.unified_inbox_store import UnifiedInboxStoreBackend


@runtime_checkable
class WorkspaceStores(Protocol):
    async def bead_store(self) -> BeadStore: ...
    async def convoy_manager(self) -> ConvoyManager: ...


@runtime_checkable
class GatewayStores(Protocol):
    def gateway_store(self) -> GatewayStore: ...
    def inbox_store(self) -> UnifiedInboxStoreBackend: ...


@dataclass
class CanonicalWorkspaceStores(WorkspaceStores):
    """Canonical stores for convoys/beads/workspaces."""

    bead_dir: str | None = None
    convoy_dir: str | None = None
    git_enabled: bool = True
    auto_commit: bool = False
    _bead_store: BeadStore | None = None
    _convoy_manager: ConvoyManager | None = None

    async def bead_store(self) -> BeadStore:
        if self._bead_store is None:
            from aragora.nomic.beads import create_bead_store

            self._bead_store = await create_bead_store(
                bead_dir=self.bead_dir,
                git_enabled=self.git_enabled,
                auto_commit=self.auto_commit,
            )
        return self._bead_store

    async def convoy_manager(self) -> ConvoyManager:
        if self._convoy_manager is None:
            bead_store = await self.bead_store()
            if self.convoy_dir:
                from aragora.nomic.convoys import ConvoyManager

                self._convoy_manager = ConvoyManager(
                    bead_store=bead_store,
                    convoy_dir=Path(self.convoy_dir),
                )
                await self._convoy_manager.initialize()
            else:
                from aragora.nomic.convoys import get_convoy_manager

                self._convoy_manager = await get_convoy_manager(bead_store)
        return self._convoy_manager


@dataclass
class CanonicalGatewayStores(GatewayStores):
    """Canonical stores for gateway/inbox persistence."""

    _gateway_store: GatewayStore | None = None
    _inbox_store: UnifiedInboxStoreBackend | None = None

    def gateway_store(self) -> GatewayStore:
        if self._gateway_store is None:
            from aragora.gateway.persistence import get_gateway_store_from_env

            self._gateway_store = get_gateway_store_from_env(
                backend_env="ARAGORA_GATEWAY_STORE",
                fallback_backend_env="ARAGORA_GATEWAY_SESSION_STORE",
                path_env="ARAGORA_GATEWAY_STORE_PATH",
                fallback_path_env="ARAGORA_GATEWAY_SESSION_PATH",
                redis_env="ARAGORA_GATEWAY_STORE_REDIS_URL",
                fallback_redis_env="ARAGORA_GATEWAY_SESSION_REDIS_URL",
                default_backend="auto",
                allow_disabled=False,
            )
        return self._gateway_store

    def inbox_store(self) -> UnifiedInboxStoreBackend:
        if self._inbox_store is None:
            from aragora.storage.unified_inbox_store import get_unified_inbox_store

            self._inbox_store = get_unified_inbox_store()
        return self._inbox_store


def get_canonical_workspace_stores(
    *,
    bead_dir: str | None = None,
    convoy_dir: str | None = None,
    git_enabled: bool = True,
    auto_commit: bool = False,
) -> CanonicalWorkspaceStores:
    """Return a canonical workspace stores accessor."""
    return CanonicalWorkspaceStores(
        bead_dir=bead_dir,
        convoy_dir=convoy_dir,
        git_enabled=git_enabled,
        auto_commit=auto_commit,
    )


def get_canonical_gateway_stores() -> CanonicalGatewayStores:
    """Return a canonical gateway stores accessor."""
    return CanonicalGatewayStores()


__all__ = [
    "CanonicalGatewayStores",
    "CanonicalWorkspaceStores",
    "GatewayStores",
    "WorkspaceStores",
    "get_canonical_gateway_stores",
    "get_canonical_workspace_stores",
]
