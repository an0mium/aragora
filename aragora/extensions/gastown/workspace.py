"""
Workspace Manager - Project and Rig Management.

Manages workspaces and rigs for multi-agent development workflows.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from aragora.nomic.stores.paths import resolve_store_dir
from aragora.workspace.manager import WorkspaceManager as CoreWorkspaceManager
from aragora.workspace.rig import Rig as CoreRig
from aragora.workspace.rig import RigConfig as CoreRigConfig
from aragora.workspace.rig import RigStatus as CoreRigStatus

from .models import (
    Rig,
    RigConfig,
    Workspace,
    WorkspaceConfig,
)

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """
    Manages workspaces and rigs for developer orchestration.

    A workspace is a root container for projects, and rigs are per-repository
    contexts where agents operate.
    """

    def __init__(self, storage_path: str | Path | None = None) -> None:
        """
        Initialize the workspace manager.

        Args:
            storage_path: Path for workspace metadata storage
        """
        base_path = Path(storage_path) if storage_path else resolve_store_dir()
        self._storage_path = base_path
        self._state_path = self._storage_path / "state.json"
        self._workspaces: dict[str, Workspace] = {}
        self._rigs: dict[str, Rig] = {}
        self._workspace_managers: dict[str, CoreWorkspaceManager] = {}
        self._rig_index: dict[str, str] = {}
        self._lock = asyncio.Lock()

        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._load_state()

    def _workspace_to_dict(self, workspace: Workspace) -> dict[str, Any]:
        data = asdict(workspace)
        data["created_at"] = workspace.created_at.isoformat()
        data["updated_at"] = workspace.updated_at.isoformat()
        return data

    def _workspace_from_dict(self, data: dict[str, Any]) -> Workspace:
        payload = dict(data)
        created_at = payload.get("created_at")
        updated_at = payload.get("updated_at")
        if isinstance(created_at, str):
            payload["created_at"] = datetime.fromisoformat(created_at)
        if isinstance(updated_at, str):
            payload["updated_at"] = datetime.fromisoformat(updated_at)
        config = payload.get("config")
        if isinstance(config, dict):
            payload["config"] = WorkspaceConfig(**config)
        return Workspace(**payload)

    def _rig_to_dict(self, rig: Rig) -> dict[str, Any]:
        data = asdict(rig)
        data["created_at"] = rig.created_at.isoformat()
        data["updated_at"] = rig.updated_at.isoformat()
        if rig.last_sync:
            data["last_sync"] = rig.last_sync.isoformat()
        return data

    def _rig_from_dict(self, data: dict[str, Any]) -> Rig:
        payload = dict(data)
        created_at = payload.get("created_at")
        updated_at = payload.get("updated_at")
        last_sync = payload.get("last_sync")
        if isinstance(created_at, str):
            payload["created_at"] = datetime.fromisoformat(created_at)
        if isinstance(updated_at, str):
            payload["updated_at"] = datetime.fromisoformat(updated_at)
        if isinstance(last_sync, str):
            payload["last_sync"] = datetime.fromisoformat(last_sync)
        config = payload.get("config")
        if isinstance(config, dict):
            payload["config"] = RigConfig(**config)
        return Rig(**payload)

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            payload = json.loads(self._state_path.read_text())
        except json.JSONDecodeError:
            logger.warning("Gastown workspace state is corrupted; ignoring %s", self._state_path)
            return
        for ws_data in payload.get("workspaces", []):
            workspace = self._workspace_from_dict(ws_data)
            self._workspaces[workspace.id] = workspace
        for rig_data in payload.get("rigs", []):
            rig = self._rig_from_dict(rig_data)
            self._rigs[rig.id] = rig
            self._rig_index[rig.id] = rig.workspace_id

    def _save_state(self) -> None:
        payload = {
            "workspaces": [self._workspace_to_dict(w) for w in self._workspaces.values()],
            "rigs": [self._rig_to_dict(r) for r in self._rigs.values()],
        }
        self._state_path.write_text(json.dumps(payload, indent=2))

    def _ensure_core_manager(self, workspace: Workspace) -> CoreWorkspaceManager:
        manager = self._workspace_managers.get(workspace.id)
        if manager:
            return manager
        manager = CoreWorkspaceManager(
            workspace_root=workspace.config.root_path,
            workspace_id=workspace.id,
        )
        self._workspace_managers[workspace.id] = manager
        return manager

    def _to_core_config(self, config: RigConfig) -> CoreRigConfig:
        return CoreRigConfig(
            repo_path=config.repo_path,
            branch=config.branch,
            max_agents=config.max_agents,
            allowed_agent_types=list(config.tools),
        )

    def _to_gastown_status(self, status: CoreRigStatus) -> Literal["active", "paused", "archived"]:
        mapping: dict[CoreRigStatus, Literal["active", "paused", "archived"]] = {
            CoreRigStatus.ACTIVE: "active",
            CoreRigStatus.READY: "active",
            CoreRigStatus.INITIALIZING: "active",
            CoreRigStatus.DRAINING: "paused",
            CoreRigStatus.STOPPED: "archived",
            CoreRigStatus.ERROR: "archived",
        }
        return mapping.get(status, "active")

    def _to_gastown_rig(self, core_rig: CoreRig) -> Rig:
        meta = core_rig.metadata or {}
        config_data = meta.get("gastown_config")
        config: RigConfig
        if isinstance(config_data, dict):
            try:
                config = RigConfig(**config_data)
            except TypeError:
                config = RigConfig(
                    name=core_rig.name,
                    repo_path=core_rig.config.repo_path,
                    branch=core_rig.config.branch,
                    max_agents=core_rig.config.max_agents,
                    tools=list(core_rig.config.allowed_agent_types),
                    metadata=dict(meta.get("gastown_metadata", {})),
                    description=str(meta.get("gastown_description", "")),
                    worktree_path=meta.get("gastown_worktree_path"),
                )
        else:
            config = RigConfig(
                name=core_rig.name,
                repo_path=core_rig.config.repo_path,
                branch=core_rig.config.branch,
                max_agents=core_rig.config.max_agents,
                tools=list(core_rig.config.allowed_agent_types),
                metadata=dict(meta.get("gastown_metadata", {})),
                description=str(meta.get("gastown_description", "")),
                worktree_path=meta.get("gastown_worktree_path"),
            )

        return Rig(
            id=core_rig.rig_id,
            workspace_id=core_rig.workspace_id,
            config=config,
            created_at=datetime.utcfromtimestamp(core_rig.created_at),
            updated_at=datetime.utcfromtimestamp(core_rig.updated_at),
            status=self._to_gastown_status(core_rig.status),
            agents=list(core_rig.assigned_agents),
            active_convoys=list(core_rig.active_convoys),
            last_sync=meta.get("gastown_last_sync"),
        )

    async def create_workspace(
        self,
        config: WorkspaceConfig,
        owner_id: str = "",
        tenant_id: str | None = None,
    ) -> Workspace:
        """
        Create a new workspace.

        Args:
            config: Workspace configuration
            owner_id: Owner user ID
            tenant_id: Tenant ID for multi-tenancy

        Returns:
            Created workspace
        """
        async with self._lock:
            workspace_id = str(uuid.uuid4())

            workspace = Workspace(
                id=workspace_id,
                config=config,
                owner_id=owner_id,
                tenant_id=tenant_id,
            )

            # Create workspace directory
            workspace_path = Path(config.root_path)
            workspace_path.mkdir(parents=True, exist_ok=True)

            self._workspaces[workspace_id] = workspace
            self._workspace_managers[workspace_id] = CoreWorkspaceManager(
                workspace_root=config.root_path,
                workspace_id=workspace_id,
            )
            self._save_state()
            logger.info(f"Created workspace {config.name} ({workspace_id})")

            return workspace

    async def get_workspace(self, workspace_id: str) -> Workspace | None:
        """Get a workspace by ID."""
        return self._workspaces.get(workspace_id)

    async def list_workspaces(
        self,
        owner_id: str | None = None,
        tenant_id: str | None = None,
    ) -> list[Workspace]:
        """List workspaces with optional filters."""
        workspaces = list(self._workspaces.values())

        if owner_id:
            workspaces = [w for w in workspaces if w.owner_id == owner_id]
        if tenant_id:
            workspaces = [w for w in workspaces if w.tenant_id == tenant_id]

        return workspaces

    async def update_workspace(
        self,
        workspace_id: str,
        config: WorkspaceConfig | None = None,
        status: Literal["active", "archived", "suspended"] | None = None,
    ) -> Workspace | None:
        """Update a workspace."""
        async with self._lock:
            workspace = self._workspaces.get(workspace_id)
            if not workspace:
                return None

            if config:
                workspace.config = config
            if status:
                workspace.status = status
            workspace.updated_at = datetime.now(timezone.utc)
            self._save_state()

            return workspace

    async def delete_workspace(
        self,
        workspace_id: str,
        force: bool = False,
    ) -> bool:
        """
        Delete a workspace.

        Args:
            workspace_id: Workspace to delete
            force: Delete even if rigs exist

        Returns:
            True if deleted
        """
        async with self._lock:
            workspace = self._workspaces.get(workspace_id)
            if not workspace:
                return False

            if workspace.rigs and not force:
                raise ValueError(f"Workspace has {len(workspace.rigs)} rigs, use force=True")

            # Delete associated rigs
            for rig_id in workspace.rigs:
                if rig_id in self._rigs:
                    del self._rigs[rig_id]
                self._rig_index.pop(rig_id, None)

            self._workspace_managers.pop(workspace_id, None)

            del self._workspaces[workspace_id]
            self._save_state()
            logger.info(f"Deleted workspace {workspace_id}")
            return True

    async def create_rig(
        self,
        workspace_id: str,
        config: RigConfig,
    ) -> Rig:
        """
        Create a new rig in a workspace.

        Args:
            workspace_id: Parent workspace
            config: Rig configuration

        Returns:
            Created rig
        """
        async with self._lock:
            workspace = self._workspaces.get(workspace_id)
            if not workspace:
                raise ValueError(f"Workspace {workspace_id} not found")

            if len(workspace.rigs) >= workspace.config.max_rigs:
                raise ValueError(f"Workspace at max rigs ({workspace.config.max_rigs})")

            rig_id = str(uuid.uuid4())

            # Verify repo path exists
            repo_path = Path(config.repo_path)
            if not repo_path.exists():
                raise ValueError(f"Repository path does not exist: {repo_path}")

            # Set up worktree if specified
            if config.worktree_path:
                worktree = Path(config.worktree_path)
                worktree.mkdir(parents=True, exist_ok=True)

            core_manager = self._ensure_core_manager(workspace)
            core_config = self._to_core_config(config)
            core_rig = await core_manager.create_rig(
                config.name,
                config=core_config,
                rig_id=rig_id,
            )
            core_rig.metadata.setdefault("gastown_config", asdict(config))
            core_rig.metadata.setdefault("gastown_description", config.description)
            core_rig.metadata.setdefault("gastown_worktree_path", config.worktree_path)
            core_rig.metadata.setdefault("gastown_metadata", dict(config.metadata))

            rig = self._to_gastown_rig(core_rig)
            self._rigs[rig_id] = rig
            self._rig_index[rig_id] = workspace_id
            workspace.rigs.append(rig_id)
            workspace.updated_at = datetime.now(timezone.utc)
            self._save_state()

            logger.info(f"Created rig {config.name} ({rig_id}) in workspace {workspace_id}")
            return rig

    async def get_rig(self, rig_id: str) -> Rig | None:
        """Get a rig by ID."""
        workspace_id = self._rig_index.get(rig_id)
        if workspace_id:
            workspace = self._workspaces.get(workspace_id)
            if workspace:
                core_manager = self._ensure_core_manager(workspace)
                core_rig = await core_manager.get_rig(rig_id)
                if core_rig:
                    rig = self._to_gastown_rig(core_rig)
                    self._rigs[rig_id] = rig
                    return rig
        return self._rigs.get(rig_id)

    async def list_rigs(
        self,
        workspace_id: str | None = None,
        status: str | None = None,
    ) -> list[Rig]:
        """List rigs with optional filters."""
        rigs: list[Rig] = []
        workspaces = (
            [self._workspaces[workspace_id]]
            if workspace_id and workspace_id in self._workspaces
            else list(self._workspaces.values())
        )
        for ws in workspaces:
            core_manager = self._ensure_core_manager(ws)
            core_rigs = await core_manager.list_rigs()
            for core_rig in core_rigs:
                rig = self._to_gastown_rig(core_rig)
                self._rigs[rig.id] = rig
                self._rig_index[rig.id] = rig.workspace_id
                rigs.append(rig)

        if status:
            rigs = [r for r in rigs if r.status == status]
        return rigs

    async def update_rig(
        self,
        rig_id: str,
        config: RigConfig | None = None,
        status: Literal["active", "paused", "archived"] | None = None,
    ) -> Rig | None:
        """Update a rig."""
        async with self._lock:
            rig = await self.get_rig(rig_id)
            if not rig:
                return None

            if config:
                rig.config = config
            if status:
                rig.status = status
            rig.updated_at = datetime.now(timezone.utc)
            self._rigs[rig_id] = rig

            workspace_id = self._rig_index.get(rig_id)
            if workspace_id:
                workspace = self._workspaces.get(workspace_id)
                if workspace:
                    core_manager = self._ensure_core_manager(workspace)
                    core_rig = await core_manager.get_rig(rig_id)
                    if core_rig:
                        if config:
                            core_rig.config = self._to_core_config(config)
                            core_rig.metadata["gastown_config"] = asdict(config)
                            core_rig.metadata["gastown_description"] = config.description
                            core_rig.metadata["gastown_worktree_path"] = config.worktree_path
                            core_rig.metadata["gastown_metadata"] = dict(config.metadata)
                        if status:
                            core_rig.status = {
                                "active": CoreRigStatus.ACTIVE,
                                "paused": CoreRigStatus.DRAINING,
                                "archived": CoreRigStatus.STOPPED,
                            }.get(status, CoreRigStatus.ACTIVE)
                        core_rig.updated_at = datetime.now(timezone.utc).timestamp()

            self._save_state()
            return rig

    async def delete_rig(
        self,
        rig_id: str,
        force: bool = False,
    ) -> bool:
        """
        Delete a rig.

        Args:
            rig_id: Rig to delete
            force: Delete even if active convoys exist

        Returns:
            True if deleted
        """
        async with self._lock:
            rig = await self.get_rig(rig_id)
            if not rig:
                return False

            if rig.active_convoys and not force:
                raise ValueError(f"Rig has {len(rig.active_convoys)} active convoys")

            # Remove from workspace
            workspace = self._workspaces.get(rig.workspace_id)
            if workspace and rig_id in workspace.rigs:
                workspace.rigs.remove(rig_id)
                workspace.updated_at = datetime.now(timezone.utc)

            del self._rigs[rig_id]
            self._rig_index.pop(rig_id, None)
            if workspace:
                core_manager = self._ensure_core_manager(workspace)
                await core_manager.delete_rig(rig_id)
            self._save_state()
            logger.info(f"Deleted rig {rig_id}")
            return True

    async def assign_agent(
        self,
        rig_id: str,
        agent_id: str,
    ) -> bool:
        """Assign an agent to a rig."""
        async with self._lock:
            rig = await self.get_rig(rig_id)
            if not rig:
                return False

            if len(rig.agents) >= rig.config.max_agents:
                raise ValueError(f"Rig at max agents ({rig.config.max_agents})")

            workspace_id = self._rig_index.get(rig_id)
            if workspace_id:
                workspace = self._workspaces.get(workspace_id)
                if workspace:
                    core_manager = self._ensure_core_manager(workspace)
                    await core_manager.assign_agent_to_rig(rig_id, agent_id)

            rig = await self.get_rig(rig_id)
            self._save_state()
            return rig is not None and agent_id in rig.agents

    async def unassign_agent(
        self,
        rig_id: str,
        agent_id: str,
    ) -> bool:
        """Remove an agent from a rig."""
        async with self._lock:
            rig = await self.get_rig(rig_id)
            if not rig:
                return False

            workspace_id = self._rig_index.get(rig_id)
            if workspace_id:
                workspace = self._workspaces.get(workspace_id)
                if workspace:
                    core_manager = self._ensure_core_manager(workspace)
                    await core_manager.remove_agent_from_rig(rig_id, agent_id)

            rig = await self.get_rig(rig_id)
            self._save_state()
            return rig is not None and agent_id not in rig.agents

    async def sync_rig(self, rig_id: str) -> bool:
        """
        Sync a rig with its repository.

        Updates the rig's last_sync timestamp and optionally
        pulls latest changes.
        """
        async with self._lock:
            rig = await self.get_rig(rig_id)
            if not rig:
                return False

            # In a real implementation, this would:
            # 1. Check git status
            # 2. Pull latest changes if clean
            # 3. Update worktree if applicable

            now = datetime.now(timezone.utc)
            rig.last_sync = now
            rig.updated_at = now
            self._rigs[rig_id] = rig

            workspace_id = self._rig_index.get(rig_id)
            if workspace_id:
                workspace = self._workspaces.get(workspace_id)
                if workspace:
                    core_manager = self._ensure_core_manager(workspace)
                    core_rig = await core_manager.get_rig(rig_id)
                    if core_rig:
                        core_rig.metadata["gastown_last_sync"] = now
                        core_rig.updated_at = now.timestamp()

            self._save_state()
            logger.debug(f"Synced rig {rig_id}")
            return True

    async def get_stats(self) -> dict[str, Any]:
        """Get workspace manager statistics."""
        async with self._lock:
            rigs = await self.list_rigs()
            active_workspaces = sum(1 for w in self._workspaces.values() if w.status == "active")
            active_rigs = sum(1 for r in rigs if r.status == "active")
            total_agents = sum(len(r.agents) for r in rigs)

            return {
                "workspaces_total": len(self._workspaces),
                "workspaces_active": active_workspaces,
                "rigs_total": len(rigs),
                "rigs_active": active_rigs,
                "agents_assigned": total_agents,
            }
