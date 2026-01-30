"""
Convoy Tracker - Work Unit Management.

Manages convoys (work tracking units) with artifacts and handoffs.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from aragora.nomic.stores import BeadStore as NomicBeadStore
from aragora.nomic.stores import ConvoyManager as NomicConvoyManager
from aragora.nomic.stores.paths import resolve_store_dir
from aragora.workspace.convoy import (
    Convoy as WorkspaceConvoy,
    ConvoyStatus as WorkspaceConvoyStatus,
    ConvoyTracker as WorkspaceConvoyTracker,
)

from .models import (
    Convoy,
    ConvoyArtifact,
    ConvoyStatus,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Convoy",
    "ConvoyArtifact",
    "ConvoyStatus",
    "ConvoyTracker",
    "NomicBeadStore",
    "NomicConvoyManager",
]


class ConvoyTracker:
    """
    Tracks convoys (work units) through their lifecycle.

    A convoy represents a unit of work that flows through the system,
    collecting artifacts and tracking state across agent handoffs.
    """

    def __init__(
        self,
        storage_path: str | Path | None = None,
        use_nomic_store: bool | None = None,
    ) -> None:
        """
        Initialize the convoy tracker.

        Args:
            storage_path: Path for convoy metadata storage
        """
        self._storage_path = Path(storage_path) if storage_path else None
        self._artifacts: dict[str, ConvoyArtifact] = {}
        self._lock = asyncio.Lock()
        self._use_nomic_store = (
            use_nomic_store if use_nomic_store is not None else bool(storage_path)
        )
        self._bead_store: NomicBeadStore | None = None
        if self._use_nomic_store and self._storage_path is None:
            self._storage_path = resolve_store_dir()
        if self._storage_path:
            self._storage_path.mkdir(parents=True, exist_ok=True)
            if self._use_nomic_store:
                self._bead_store = NomicBeadStore(
                    self._storage_path,
                    git_enabled=False,
                    auto_commit=False,
                )
        self._workspace_tracker = WorkspaceConvoyTracker(
            bead_store=self._bead_store,
            use_nomic_store=self._use_nomic_store,
        )
        self._default_workspace_id = "gastown"

    def _to_workspace_status(self, status: ConvoyStatus) -> WorkspaceConvoyStatus:
        return WorkspaceConvoyStatus(status.to_workspace_status())

    def _to_workspace_metadata(self, convoy: Convoy) -> dict[str, Any]:
        return {
            "issue_ref": convoy.issue_ref,
            "parent_convoy": convoy.parent_convoy,
            "priority": convoy.priority,
            "tags": list(convoy.tags),
            "metadata": dict(convoy.metadata),
            "assigned_agents": list(convoy.assigned_agents),
            "current_agent": convoy.current_agent,
            "handoff_count": convoy.handoff_count,
            "artifacts": list(convoy.artifacts),
            "gastown_status": convoy.status.value,
            "error": convoy.error,
            "result": convoy.result,
            "depends_on": list(convoy.depends_on),
        }

    def _from_workspace_convoy(self, convoy: WorkspaceConvoy) -> Convoy:
        metadata = convoy.metadata or {}
        stored = metadata.get("gastown_status")
        status = ConvoyStatus.from_workspace_status(convoy.status.value)
        if isinstance(stored, str):
            try:
                status = ConvoyStatus(stored)
            except ValueError:
                pass
        created_at = (
            datetime.utcfromtimestamp(convoy.created_at) if convoy.created_at else datetime.utcnow()
        )
        updated_at = (
            datetime.utcfromtimestamp(convoy.updated_at) if convoy.updated_at else created_at
        )
        started_at = datetime.utcfromtimestamp(convoy.started_at) if convoy.started_at else None
        completed_at = (
            datetime.utcfromtimestamp(convoy.completed_at) if convoy.completed_at else None
        )
        reserved_keys = {
            "issue_ref",
            "parent_convoy",
            "priority",
            "tags",
            "metadata",
            "assigned_agents",
            "current_agent",
            "handoff_count",
            "artifacts",
            "gastown_status",
            "error",
            "result",
            "depends_on",
            "workspace_id",
            "workspace_name",
            "workspace_description",
            "workspace_status",
            "rig_id",
            "merge_result",
        }
        raw_user_metadata = metadata.get("metadata")
        user_metadata: dict[str, Any] = (
            dict(raw_user_metadata) if isinstance(raw_user_metadata, dict) else {}
        )
        for key, value in metadata.items():
            if key in reserved_keys:
                continue
            user_metadata.setdefault(key, value)

        return Convoy(
            id=convoy.convoy_id,
            rig_id=convoy.rig_id,
            title=convoy.name,
            description=convoy.description,
            status=status,
            created_at=created_at,
            updated_at=updated_at,
            started_at=started_at,
            completed_at=completed_at,
            issue_ref=metadata.get("issue_ref"),
            parent_convoy=metadata.get("parent_convoy"),
            depends_on=list(metadata.get("depends_on", [])),
            assigned_agents=list(convoy.assigned_agents),
            current_agent=metadata.get("current_agent"),
            handoff_count=int(metadata.get("handoff_count", 0)),
            artifacts=list(metadata.get("artifacts", [])),
            result=dict(metadata.get("result") or {}),
            error=metadata.get("error"),
            priority=int(metadata.get("priority", 0)),
            tags=list(metadata.get("tags", [])),
            metadata=user_metadata,
        )

    async def create_convoy(
        self,
        rig_id: str,
        title: str,
        description: str = "",
        issue_ref: str | None = None,
        parent_convoy: str | None = None,
        priority: int = 0,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Convoy:
        """
        Create a new convoy.

        Args:
            rig_id: Parent rig ID
            title: Convoy title
            description: Detailed description
            issue_ref: External issue reference
            parent_convoy: Parent convoy ID for sub-tasks
            priority: Priority level (higher = more important)
            tags: Tags for categorization
            metadata: Additional metadata

        Returns:
            Created convoy
        """
        async with self._lock:
            convoy_id = str(uuid.uuid4())

            convoy = Convoy(
                id=convoy_id,
                rig_id=rig_id,
                title=title,
                description=description,
                issue_ref=issue_ref,
                parent_convoy=parent_convoy,
                priority=priority,
                tags=tags or [],
                metadata=metadata or {},
            )

            workspace_id = (
                (metadata or {}).get("workspace_id") if isinstance(metadata, dict) else None
            ) or self._default_workspace_id
            ws_metadata = self._to_workspace_metadata(convoy)
            ws_convoy = await self._workspace_tracker.create_convoy(
                workspace_id=workspace_id,
                rig_id=rig_id,
                name=title,
                description=description,
                bead_ids=[],
                convoy_id=convoy_id,
                metadata=ws_metadata,
                assigned_agents=convoy.assigned_agents,
            )
            logger.info(f"Created convoy {title} ({convoy_id})")
            return self._from_workspace_convoy(ws_convoy)

    async def get_convoy(self, convoy_id: str) -> Convoy | None:
        """Get a convoy by ID."""
        ws_convoy = await self._workspace_tracker.get_convoy(convoy_id)
        if not ws_convoy:
            return None
        return self._from_workspace_convoy(ws_convoy)

    async def list_convoys(
        self,
        rig_id: str | None = None,
        status: ConvoyStatus | None = None,
        agent_id: str | None = None,
    ) -> list[Convoy]:
        """List convoys with optional filters."""
        ws_status = self._to_workspace_status(status) if status else None
        ws_convoys = await self._workspace_tracker.list_convoys(
            rig_id=rig_id,
            status=ws_status,
            agent_id=agent_id,
        )
        convoys = [self._from_workspace_convoy(c) for c in ws_convoys]
        if agent_id:
            convoys = [c for c in convoys if agent_id in c.assigned_agents]
        return convoys

    async def start_convoy(self, convoy_id: str, agent_id: str) -> Convoy | None:
        """
        Start work on a convoy.

        Args:
            convoy_id: Convoy to start
            agent_id: Agent starting work

        Returns:
            Updated convoy or None if not found
        """
        async with self._lock:
            convoy = await self.get_convoy(convoy_id)
            if not convoy:
                return None

            if convoy.status != ConvoyStatus.PENDING:
                raise ValueError(f"Convoy {convoy_id} is not pending")

            assigned_agents = list({*convoy.assigned_agents, agent_id})
            ws_convoy = await self._workspace_tracker.start_executing(
                convoy_id,
                assigned_agents=assigned_agents,
            )
            metadata_updates = {
                "current_agent": agent_id,
                "handoff_count": convoy.handoff_count,
                "assigned_agents": assigned_agents,
                "gastown_status": ConvoyStatus.IN_PROGRESS.value,
            }
            ws_convoy = await self._workspace_tracker.update_metadata(
                convoy_id,
                metadata_updates=metadata_updates,
                assigned_agents=assigned_agents,
            )
            logger.info(f"Started convoy {convoy_id} with agent {agent_id}")
            if not ws_convoy:
                return None
            return self._from_workspace_convoy(ws_convoy)

    async def handoff_convoy(
        self,
        convoy_id: str,
        from_agent: str,
        to_agent: str,
        notes: str = "",
    ) -> Convoy | None:
        """
        Hand off a convoy to another agent.

        Args:
            convoy_id: Convoy to hand off
            from_agent: Current agent
            to_agent: Target agent
            notes: Handoff notes

        Returns:
            Updated convoy or None if not found
        """
        async with self._lock:
            convoy = await self.get_convoy(convoy_id)
            if not convoy:
                return None

            if convoy.current_agent != from_agent:
                raise ValueError(f"Agent {from_agent} is not current owner")

            assigned_agents = list({*convoy.assigned_agents, to_agent})
            handoffs = list(convoy.metadata.get("handoffs", []))
            handoffs.append(
                {
                    "from": from_agent,
                    "to": to_agent,
                    "notes": notes,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            metadata_updates = {
                "current_agent": to_agent,
                "handoff_count": convoy.handoff_count + 1,
                "handoffs": handoffs,
                "assigned_agents": assigned_agents,
            }
            ws_convoy = await self._workspace_tracker.update_metadata(
                convoy_id,
                metadata_updates=metadata_updates,
                assigned_agents=assigned_agents,
            )
            logger.info(f"Handed off convoy {convoy_id} from {from_agent} to {to_agent}")
            if not ws_convoy:
                return None
            return self._from_workspace_convoy(ws_convoy)

    async def block_convoy(
        self,
        convoy_id: str,
        reason: str,
        depends_on: list[str] | None = None,
    ) -> Convoy | None:
        """
        Mark a convoy as blocked.

        Args:
            convoy_id: Convoy to block
            reason: Reason for blocking
            depends_on: IDs of blocking convoys/issues

        Returns:
            Updated convoy or None if not found
        """
        async with self._lock:
            convoy = await self.get_convoy(convoy_id)
            if not convoy:
                return None

            await self._workspace_tracker.fail_convoy(convoy_id, reason)
            depends = list(convoy.depends_on)
            if depends_on:
                depends.extend(depends_on)
            metadata_updates = {
                "block_reason": reason,
                "depends_on": depends,
                "gastown_status": ConvoyStatus.BLOCKED.value,
            }
            ws_convoy = await self._workspace_tracker.update_metadata(
                convoy_id,
                metadata_updates=metadata_updates,
                assigned_agents=convoy.assigned_agents,
            )
            logger.info(f"Blocked convoy {convoy_id}: {reason}")
            if not ws_convoy:
                return None
            return self._from_workspace_convoy(ws_convoy)

    async def unblock_convoy(self, convoy_id: str) -> Convoy | None:
        """Resume a blocked convoy."""
        async with self._lock:
            convoy = await self.get_convoy(convoy_id)
            if not convoy:
                return None

            if convoy.status != ConvoyStatus.BLOCKED:
                raise ValueError(f"Convoy {convoy_id} is not blocked")

            ws_convoy = await self._workspace_tracker.start_executing(
                convoy_id,
                assigned_agents=convoy.assigned_agents,
            )
            metadata_updates = {
                "block_reason": None,
                "gastown_status": ConvoyStatus.IN_PROGRESS.value,
            }
            ws_convoy = await self._workspace_tracker.update_metadata(
                convoy_id,
                metadata_updates=metadata_updates,
                assigned_agents=convoy.assigned_agents,
            )
            logger.info(f"Unblocked convoy {convoy_id}")
            if not ws_convoy:
                return None
            return self._from_workspace_convoy(ws_convoy)

    async def submit_for_review(self, convoy_id: str) -> Convoy | None:
        """Submit a convoy for review."""
        async with self._lock:
            convoy = await self.get_convoy(convoy_id)
            if not convoy:
                return None

            ws_convoy = await self._workspace_tracker.start_merging(convoy_id)
            ws_convoy = await self._workspace_tracker.update_metadata(
                convoy_id,
                metadata_updates={"gastown_status": ConvoyStatus.REVIEW.value},
                assigned_agents=convoy.assigned_agents,
            )
            logger.info(f"Submitted convoy {convoy_id} for review")
            if not ws_convoy:
                return None
            return self._from_workspace_convoy(ws_convoy)

    async def complete_convoy(
        self,
        convoy_id: str,
        result: dict[str, Any] | None = None,
    ) -> Convoy | None:
        """
        Mark a convoy as completed.

        Args:
            convoy_id: Convoy to complete
            result: Final result data

        Returns:
            Updated convoy or None if not found
        """
        async with self._lock:
            convoy = await self.get_convoy(convoy_id)
            if not convoy:
                return None

            ws_convoy = await self._workspace_tracker.complete_convoy(
                convoy_id,
                merge_result=result,
            )
            ws_convoy = await self._workspace_tracker.update_metadata(
                convoy_id,
                metadata_updates={
                    "result": result or {},
                    "gastown_status": ConvoyStatus.COMPLETED.value,
                },
                assigned_agents=convoy.assigned_agents,
            )
            logger.info(f"Completed convoy {convoy_id}")
            if not ws_convoy:
                return None
            return self._from_workspace_convoy(ws_convoy)

    async def cancel_convoy(
        self,
        convoy_id: str,
        reason: str = "",
    ) -> Convoy | None:
        """Cancel a convoy."""
        async with self._lock:
            convoy = await self.get_convoy(convoy_id)
            if not convoy:
                return None

            await self._workspace_tracker.cancel_convoy(convoy_id)
            ws_convoy = await self._workspace_tracker.update_metadata(
                convoy_id,
                metadata_updates={
                    "error": reason,
                    "gastown_status": ConvoyStatus.CANCELLED.value,
                },
                assigned_agents=convoy.assigned_agents,
            )
            logger.info(f"Cancelled convoy {convoy_id}: {reason}")
            if not ws_convoy:
                return None
            return self._from_workspace_convoy(ws_convoy)

    async def add_artifact(
        self,
        convoy_id: str,
        artifact_type: str,
        path: str,
        content_hash: str = "",
        size_bytes: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> ConvoyArtifact | None:
        """
        Add an artifact to a convoy.

        Args:
            convoy_id: Parent convoy
            artifact_type: Type (file, diff, log, report, receipt)
            path: Artifact path
            content_hash: Hash of content
            size_bytes: Size in bytes
            metadata: Additional metadata

        Returns:
            Created artifact or None if convoy not found
        """
        async with self._lock:
            convoy = await self.get_convoy(convoy_id)
            if not convoy:
                return None

            artifact_id = str(uuid.uuid4())

            artifact = ConvoyArtifact(
                id=artifact_id,
                convoy_id=convoy_id,
                type=artifact_type,
                path=path,
                content_hash=content_hash,
                size_bytes=size_bytes,
                metadata=metadata or {},
            )

            self._artifacts[artifact_id] = artifact
            artifacts = list(convoy.artifacts)
            artifacts.append(artifact_id)

            logger.debug(f"Added artifact {artifact_type} to convoy {convoy_id}")
            await self._workspace_tracker.update_metadata(
                convoy_id,
                metadata_updates={"artifacts": artifacts},
                assigned_agents=convoy.assigned_agents,
            )
            return artifact

    async def get_artifact(self, artifact_id: str) -> ConvoyArtifact | None:
        """Get an artifact by ID."""
        return self._artifacts.get(artifact_id)

    async def list_artifacts(self, convoy_id: str) -> list[ConvoyArtifact]:
        """List artifacts for a convoy."""
        convoy = await self.get_convoy(convoy_id)
        if not convoy:
            return []

        return [self._artifacts[aid] for aid in convoy.artifacts if aid in self._artifacts]

    async def get_stats(self) -> dict[str, Any]:
        """Get convoy tracker statistics."""
        async with self._lock:
            convoys = await self.list_convoys()
            by_status: dict[str, int] = {}
            for convoy in convoys:
                status = convoy.status.value
                by_status[status] = by_status.get(status, 0) + 1

            total_handoffs = sum(c.handoff_count for c in convoys)

            return {
                "convoys_total": len(convoys),
                "convoys_by_status": by_status,
                "artifacts_total": len(self._artifacts),
                "total_handoffs": total_handoffs,
            }
