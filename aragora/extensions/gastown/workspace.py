"""
Workspace Manager - Project and Rig Management.

Manages workspaces and rigs for multi-agent development workflows.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

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
        self._storage_path = Path(storage_path) if storage_path else None
        self._workspaces: dict[str, Workspace] = {}
        self._rigs: dict[str, Rig] = {}
        self._lock = asyncio.Lock()

        if self._storage_path:
            self._storage_path.mkdir(parents=True, exist_ok=True)

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
        status: str | None = None,
    ) -> Workspace | None:
        """Update a workspace."""
        async with self._lock:
            workspace = self._workspaces.get(workspace_id)
            if not workspace:
                return None

            if config:
                workspace.config = config
            if status:
                workspace.status = status  # type: ignore[attr-defined]
            workspace.updated_at = datetime.utcnow()

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

            del self._workspaces[workspace_id]
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

            rig = Rig(
                id=rig_id,
                workspace_id=workspace_id,
                config=config,
            )

            # Verify repo path exists
            repo_path = Path(config.repo_path)
            if not repo_path.exists():
                raise ValueError(f"Repository path does not exist: {repo_path}")

            # Set up worktree if specified
            if config.worktree_path:
                worktree = Path(config.worktree_path)
                worktree.mkdir(parents=True, exist_ok=True)

            self._rigs[rig_id] = rig
            workspace.rigs.append(rig_id)
            workspace.updated_at = datetime.utcnow()

            logger.info(f"Created rig {config.name} ({rig_id}) in workspace {workspace_id}")
            return rig

    async def get_rig(self, rig_id: str) -> Rig | None:
        """Get a rig by ID."""
        return self._rigs.get(rig_id)

    async def list_rigs(
        self,
        workspace_id: str | None = None,
        status: str | None = None,
    ) -> list[Rig]:
        """List rigs with optional filters."""
        rigs = list(self._rigs.values())

        if workspace_id:
            rigs = [r for r in rigs if r.workspace_id == workspace_id]
        if status:
            rigs = [r for r in rigs if r.status == status]

        return rigs

    async def update_rig(
        self,
        rig_id: str,
        config: RigConfig | None = None,
        status: str | None = None,
    ) -> Rig | None:
        """Update a rig."""
        async with self._lock:
            rig = self._rigs.get(rig_id)
            if not rig:
                return None

            if config:
                rig.config = config
            if status:
                rig.status = status  # type: ignore[attr-defined]
            rig.updated_at = datetime.utcnow()

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
            rig = self._rigs.get(rig_id)
            if not rig:
                return False

            if rig.active_convoys and not force:
                raise ValueError(f"Rig has {len(rig.active_convoys)} active convoys")

            # Remove from workspace
            workspace = self._workspaces.get(rig.workspace_id)
            if workspace and rig_id in workspace.rigs:
                workspace.rigs.remove(rig_id)
                workspace.updated_at = datetime.utcnow()

            del self._rigs[rig_id]
            logger.info(f"Deleted rig {rig_id}")
            return True

    async def assign_agent(
        self,
        rig_id: str,
        agent_id: str,
    ) -> bool:
        """Assign an agent to a rig."""
        async with self._lock:
            rig = self._rigs.get(rig_id)
            if not rig:
                return False

            if len(rig.agents) >= rig.config.max_agents:
                raise ValueError(f"Rig at max agents ({rig.config.max_agents})")

            if agent_id not in rig.agents:
                rig.agents.append(agent_id)
                rig.updated_at = datetime.utcnow()

            return True

    async def unassign_agent(
        self,
        rig_id: str,
        agent_id: str,
    ) -> bool:
        """Remove an agent from a rig."""
        async with self._lock:
            rig = self._rigs.get(rig_id)
            if not rig:
                return False

            if agent_id in rig.agents:
                rig.agents.remove(agent_id)
                rig.updated_at = datetime.utcnow()

            return True

    async def sync_rig(self, rig_id: str) -> bool:
        """
        Sync a rig with its repository.

        Updates the rig's last_sync timestamp and optionally
        pulls latest changes.
        """
        async with self._lock:
            rig = self._rigs.get(rig_id)
            if not rig:
                return False

            # In a real implementation, this would:
            # 1. Check git status
            # 2. Pull latest changes if clean
            # 3. Update worktree if applicable

            rig.last_sync = datetime.utcnow()
            rig.updated_at = datetime.utcnow()

            logger.debug(f"Synced rig {rig_id}")
            return True

    async def get_stats(self) -> dict[str, Any]:
        """Get workspace manager statistics."""
        async with self._lock:
            active_workspaces = sum(1 for w in self._workspaces.values() if w.status == "active")
            active_rigs = sum(1 for r in self._rigs.values() if r.status == "active")
            total_agents = sum(len(r.agents) for r in self._rigs.values())

            return {
                "workspaces_total": len(self._workspaces),
                "workspaces_active": active_workspaces,
                "rigs_total": len(self._rigs),
                "rigs_active": active_rigs,
                "agents_assigned": total_agents,
            }
