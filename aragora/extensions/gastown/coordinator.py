"""
Coordinator - Mayor-Style Orchestration Interface.

Provides high-level orchestration for multi-agent development workflows.
Inspired by Gastown's "Mayor" pattern for developer orchestration.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

from aragora.nomic.stores.paths import resolve_runtime_store_dir

# Type aliases matching LedgerEntry fields
LedgerEntryType = Literal["issue", "task", "decision", "note"]
LedgerEntryStatus = Literal["open", "in_progress", "resolved", "closed"]

from .models import (
    Convoy,
    ConvoyStatus,
    HookType,
    LedgerEntry,
    Rig,
    RigConfig,
    Workspace,
    WorkspaceConfig,
)
from .workspace import WorkspaceManager
from .convoy import ConvoyTracker
from .hooks import HookRunner

logger = logging.getLogger(__name__)


class Coordinator:
    """
    High-level orchestrator for multi-agent development workflows.

    The Coordinator (Mayor) manages the lifecycle of workspaces, rigs,
    and convoys, providing a unified interface for agent orchestration.
    """

    def __init__(
        self,
        storage_path: str | Path | None = None,
        auto_persist: bool = True,
    ) -> None:
        """
        Initialize the coordinator.

        Args:
            storage_path: Path for all state storage
            auto_persist: Auto-persist state changes
        """
        base_path = Path(storage_path) if storage_path else resolve_runtime_store_dir()
        self._storage_path = base_path
        self._auto_persist = auto_persist

        # Initialize subsystems
        self._workspace_manager = WorkspaceManager(storage_path=self._storage_path / "workspaces")
        self._convoy_tracker = ConvoyTracker(storage_path=self._storage_path / "convoys")
        self._hook_runner = HookRunner(
            storage_path=self._storage_path / "hooks",
            auto_commit=auto_persist,
        )

        # Ledger for issue/task tracking
        self._ledger: dict[str, LedgerEntry] = {}
        self._lock = asyncio.Lock()

        self._storage_path.mkdir(parents=True, exist_ok=True)

    @property
    def workspaces(self) -> WorkspaceManager:
        """Access workspace manager."""
        return self._workspace_manager

    @property
    def convoys(self) -> ConvoyTracker:
        """Access convoy tracker."""
        return self._convoy_tracker

    @property
    def hooks(self) -> HookRunner:
        """Access hook runner."""
        return self._hook_runner

    # ========== Workspace Operations ==========

    async def create_workspace(
        self,
        name: str,
        root_path: str,
        description: str = "",
        owner_id: str = "",
        tenant_id: str | None = None,
        max_rigs: int = 10,
    ) -> Workspace:
        """Create a new workspace with default configuration."""
        config = WorkspaceConfig(
            name=name,
            root_path=root_path,
            description=description,
            max_rigs=max_rigs,
        )
        return await self._workspace_manager.create_workspace(config, owner_id, tenant_id)

    async def setup_rig(
        self,
        workspace_id: str,
        name: str,
        repo_path: str,
        branch: str = "main",
        create_worktree: bool = False,
    ) -> Rig:
        """
        Set up a new rig in a workspace.

        Args:
            workspace_id: Parent workspace
            name: Rig name
            repo_path: Repository path
            branch: Git branch
            create_worktree: Create a worktree for isolation

        Returns:
            Created rig
        """
        worktree_path = None

        if create_worktree:
            # Create isolated worktree for the rig
            workspace = await self._workspace_manager.get_workspace(workspace_id)
            if workspace:
                import uuid

                worktree_name = f"{name}-{uuid.uuid4().hex[:8]}"
                worktree_path = str(Path(workspace.config.root_path) / ".worktrees" / worktree_name)
                result = await self._hook_runner.create_worktree(repo_path, worktree_path, branch)
                if not result["success"]:
                    raise ValueError(f"Failed to create worktree: {result.get('error')}")

        config = RigConfig(
            name=name,
            repo_path=repo_path,
            branch=branch,
            worktree_path=worktree_path,
        )

        rig = await self._workspace_manager.create_rig(workspace_id, config)

        # Set up default hooks for the rig
        await self._setup_rig_hooks(rig)

        return rig

    async def _setup_rig_hooks(self, rig: Rig) -> None:
        """Set up default hooks for a rig."""
        hooks_dir = Path(rig.config.repo_path) / ".git" / "hooks"

        # Pre-commit hook for state persistence
        await self._hook_runner.create_hook(
            rig_id=rig.id,
            hook_type=HookType.PRE_COMMIT,
            path=str(hooks_dir / "pre-commit"),
            content="#!/bin/bash\n# Aragora pre-commit hook\nexit 0\n",
        )

        # Post-commit hook for convoy tracking
        await self._hook_runner.create_hook(
            rig_id=rig.id,
            hook_type=HookType.POST_COMMIT,
            path=str(hooks_dir / "post-commit"),
            content="#!/bin/bash\n# Aragora post-commit hook\nexit 0\n",
        )

    # ========== Convoy Operations ==========

    async def start_work(
        self,
        rig_id: str,
        title: str,
        description: str = "",
        agent_id: str | None = None,
        issue_ref: str | None = None,
    ) -> Convoy:
        """
        Start new work in a rig.

        Creates a convoy and optionally assigns an agent immediately.

        Args:
            rig_id: Rig to work in
            title: Work title
            description: Work description
            agent_id: Agent to assign (starts work if provided)
            issue_ref: External issue reference

        Returns:
            Created (and optionally started) convoy
        """
        # Create the convoy
        convoy = await self._convoy_tracker.create_convoy(
            rig_id=rig_id,
            title=title,
            description=description,
            issue_ref=issue_ref,
        )

        # Update rig with active convoy
        rig = await self._workspace_manager.get_rig(rig_id)
        if rig:
            rig.active_convoys.append(convoy.id)

        # Start work if agent provided
        if agent_id:
            updated = await self._convoy_tracker.start_convoy(convoy.id, agent_id)
            if updated:
                convoy = updated

        return convoy

    async def complete_work(
        self,
        convoy_id: str,
        result: dict[str, Any] | None = None,
    ) -> Convoy | None:
        """
        Complete work on a convoy.

        Args:
            convoy_id: Convoy to complete
            result: Final result data

        Returns:
            Completed convoy
        """
        convoy = await self._convoy_tracker.complete_convoy(convoy_id, result)

        if convoy:
            # Remove from rig's active convoys
            rig = await self._workspace_manager.get_rig(convoy.rig_id)
            if rig and convoy_id in rig.active_convoys:
                rig.active_convoys.remove(convoy_id)

            # Persist state
            if self._auto_persist:
                await self._persist_convoy_result(convoy)

        return convoy

    async def _persist_convoy_result(self, convoy: Convoy) -> None:
        """Persist convoy result to hooks storage."""
        await self._hook_runner.persist_state(
            rig_id=convoy.rig_id,
            state={
                "convoy_id": convoy.id,
                "title": convoy.title,
                "status": convoy.status.value,
                "result": convoy.result,
                "artifacts": convoy.artifacts,
                "completed_at": convoy.completed_at.isoformat() if convoy.completed_at else None,
            },
            message=f"Completed convoy: {convoy.title}",
        )

    # ========== Ledger Operations ==========

    async def create_ledger_entry(
        self,
        workspace_id: str,
        entry_type: LedgerEntryType,
        title: str,
        body: str = "",
        convoy_id: str | None = None,
        created_by: str = "",
        labels: list[str] | None = None,
    ) -> LedgerEntry:
        """
        Create a ledger entry (issue, task, decision, note).

        Args:
            workspace_id: Parent workspace
            entry_type: Type (issue, task, decision, note)
            title: Entry title
            body: Entry body
            convoy_id: Related convoy
            created_by: Creator ID
            labels: Labels

        Returns:
            Created entry
        """
        import uuid

        async with self._lock:
            entry_id = str(uuid.uuid4())

            entry = LedgerEntry(
                id=entry_id,
                workspace_id=workspace_id,
                type=entry_type,
                title=title,
                body=body,
                convoy_id=convoy_id,
                created_by=created_by,
                labels=labels or [],
            )

            self._ledger[entry_id] = entry
            logger.info(f"Created ledger entry: {title}")

            return entry

    async def get_ledger_entry(self, entry_id: str) -> LedgerEntry | None:
        """Get a ledger entry by ID."""
        return self._ledger.get(entry_id)

    async def list_ledger_entries(
        self,
        workspace_id: str | None = None,
        entry_type: str | None = None,
        status: str | None = None,
        convoy_id: str | None = None,
    ) -> list[LedgerEntry]:
        """List ledger entries with filters."""
        entries = list(self._ledger.values())

        if workspace_id:
            entries = [e for e in entries if e.workspace_id == workspace_id]
        if entry_type:
            entries = [e for e in entries if e.type == entry_type]
        if status:
            entries = [e for e in entries if e.status == status]
        if convoy_id:
            entries = [e for e in entries if e.convoy_id == convoy_id]

        return entries

    async def resolve_ledger_entry(
        self,
        entry_id: str,
        resolution: str = "",
    ) -> LedgerEntry | None:
        """Resolve a ledger entry."""
        async with self._lock:
            entry = self._ledger.get(entry_id)
            if not entry:
                return None

            entry.status = "resolved"
            entry.resolved_at = datetime.utcnow()
            entry.updated_at = datetime.utcnow()
            if resolution:
                entry.metadata["resolution"] = resolution

            return entry

    # ========== Agent Orchestration ==========

    async def assign_agent_to_rig(
        self,
        rig_id: str,
        agent_id: str,
    ) -> bool:
        """Assign an agent to a rig."""
        return await self._workspace_manager.assign_agent(rig_id, agent_id)

    async def handoff_to_agent(
        self,
        convoy_id: str,
        from_agent: str,
        to_agent: str,
        notes: str = "",
    ) -> Convoy | None:
        """Hand off a convoy to another agent."""
        convoy = await self._convoy_tracker.handoff_convoy(convoy_id, from_agent, to_agent, notes)

        if convoy:
            # Ensure to_agent is assigned to the rig
            await self._workspace_manager.assign_agent(convoy.rig_id, to_agent)

        return convoy

    # ========== Status and Statistics ==========

    async def get_workspace_status(self, workspace_id: str) -> dict[str, Any] | None:
        """Get comprehensive status for a workspace."""
        workspace = await self._workspace_manager.get_workspace(workspace_id)
        if not workspace:
            return None

        rigs = await self._workspace_manager.list_rigs(workspace_id=workspace_id)
        rig_statuses = []

        for rig in rigs:
            convoys = await self._convoy_tracker.list_convoys(rig_id=rig.id)
            active_convoys = [c for c in convoys if c.status == ConvoyStatus.IN_PROGRESS]
            blocked_convoys = [c for c in convoys if c.status == ConvoyStatus.BLOCKED]

            rig_statuses.append(
                {
                    "id": rig.id,
                    "name": rig.config.name,
                    "status": rig.status,
                    "agents": len(rig.agents),
                    "convoys_total": len(convoys),
                    "convoys_active": len(active_convoys),
                    "convoys_blocked": len(blocked_convoys),
                }
            )

        ledger_entries = await self.list_ledger_entries(workspace_id=workspace_id)
        open_entries = [e for e in ledger_entries if e.status == "open"]

        return {
            "workspace": {
                "id": workspace.id,
                "name": workspace.config.name,
                "status": workspace.status,
            },
            "rigs": rig_statuses,
            "ledger": {
                "total": len(ledger_entries),
                "open": len(open_entries),
            },
        }

    async def get_stats(self) -> dict[str, Any]:
        """Get comprehensive coordinator statistics."""
        workspace_stats = await self._workspace_manager.get_stats()
        convoy_stats = await self._convoy_tracker.get_stats()
        hook_stats = await self._hook_runner.get_stats()

        async with self._lock:
            ledger_by_type: dict[str, int] = {}
            for entry in self._ledger.values():
                ledger_by_type[entry.type] = ledger_by_type.get(entry.type, 0) + 1

        return {
            "workspaces": workspace_stats,
            "convoys": convoy_stats,
            "hooks": hook_stats,
            "ledger": {
                "total": len(self._ledger),
                "by_type": ledger_by_type,
            },
        }

    # ========== Disaster Recovery ==========

    async def persist_all(self) -> dict[str, Any]:
        """Persist all state for disaster recovery."""
        if not self._storage_path:
            return {"success": False, "error": "No storage path configured"}

        results = {}

        # Persist ledger
        import json

        ledger_path = self._storage_path / "ledger.json"
        ledger_data = {
            entry_id: {
                "id": entry.id,
                "workspace_id": entry.workspace_id,
                "type": entry.type,
                "title": entry.title,
                "body": entry.body,
                "status": entry.status,
                "created_at": entry.created_at.isoformat(),
            }
            for entry_id, entry in self._ledger.items()
        }
        ledger_path.write_text(json.dumps(ledger_data, indent=2))
        results["ledger"] = str(ledger_path)

        logger.info("Persisted all coordinator state")
        return {"success": True, "paths": results}

    async def restore_all(self) -> dict[str, Any]:
        """Restore all state from storage."""
        if not self._storage_path:
            return {"success": False, "error": "No storage path configured"}

        results = {}

        # Restore ledger
        import json

        ledger_path = self._storage_path / "ledger.json"
        if ledger_path.exists():
            ledger_data = json.loads(ledger_path.read_text())
            for entry_id, data in ledger_data.items():
                entry = LedgerEntry(
                    id=data["id"],
                    workspace_id=data["workspace_id"],
                    type=cast(LedgerEntryType, data["type"]),
                    title=data["title"],
                    body=data.get("body", ""),
                    status=cast(LedgerEntryStatus, data.get("status", "open")),
                )
                self._ledger[entry_id] = entry
            results["ledger"] = len(ledger_data)

        logger.info("Restored coordinator state")
        return {"success": True, "restored": results}
