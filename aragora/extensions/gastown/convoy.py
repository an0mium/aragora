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

from .models import (
    Convoy,
    ConvoyArtifact,
    ConvoyStatus,
)

logger = logging.getLogger(__name__)


class ConvoyTracker:
    """
    Tracks convoys (work units) through their lifecycle.

    A convoy represents a unit of work that flows through the system,
    collecting artifacts and tracking state across agent handoffs.
    """

    def __init__(self, storage_path: str | Path | None = None) -> None:
        """
        Initialize the convoy tracker.

        Args:
            storage_path: Path for convoy metadata storage
        """
        self._storage_path = Path(storage_path) if storage_path else None
        self._convoys: dict[str, Convoy] = {}
        self._artifacts: dict[str, ConvoyArtifact] = {}
        self._lock = asyncio.Lock()

        if self._storage_path:
            self._storage_path.mkdir(parents=True, exist_ok=True)

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

            self._convoys[convoy_id] = convoy
            logger.info(f"Created convoy {title} ({convoy_id})")

            return convoy

    async def get_convoy(self, convoy_id: str) -> Convoy | None:
        """Get a convoy by ID."""
        return self._convoys.get(convoy_id)

    async def list_convoys(
        self,
        rig_id: str | None = None,
        status: ConvoyStatus | None = None,
        agent_id: str | None = None,
    ) -> list[Convoy]:
        """List convoys with optional filters."""
        convoys = list(self._convoys.values())

        if rig_id:
            convoys = [c for c in convoys if c.rig_id == rig_id]
        if status:
            convoys = [c for c in convoys if c.status == status]
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
            convoy = self._convoys.get(convoy_id)
            if not convoy:
                return None

            if convoy.status != ConvoyStatus.PENDING:
                raise ValueError(f"Convoy {convoy_id} is not pending")

            convoy.status = ConvoyStatus.IN_PROGRESS
            convoy.started_at = datetime.utcnow()
            convoy.current_agent = agent_id
            if agent_id not in convoy.assigned_agents:
                convoy.assigned_agents.append(agent_id)
            convoy.updated_at = datetime.utcnow()

            logger.info(f"Started convoy {convoy_id} with agent {agent_id}")
            return convoy

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
            convoy = self._convoys.get(convoy_id)
            if not convoy:
                return None

            if convoy.current_agent != from_agent:
                raise ValueError(f"Agent {from_agent} is not current owner")

            convoy.current_agent = to_agent
            if to_agent not in convoy.assigned_agents:
                convoy.assigned_agents.append(to_agent)
            convoy.handoff_count += 1
            convoy.updated_at = datetime.utcnow()

            # Store handoff notes in metadata
            handoffs = convoy.metadata.get("handoffs", [])
            handoffs.append(
                {
                    "from": from_agent,
                    "to": to_agent,
                    "notes": notes,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            convoy.metadata["handoffs"] = handoffs

            logger.info(f"Handed off convoy {convoy_id} from {from_agent} to {to_agent}")
            return convoy

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
            convoy = self._convoys.get(convoy_id)
            if not convoy:
                return None

            convoy.status = ConvoyStatus.BLOCKED
            convoy.metadata["block_reason"] = reason
            if depends_on:
                convoy.depends_on.extend(depends_on)
            convoy.updated_at = datetime.utcnow()

            logger.info(f"Blocked convoy {convoy_id}: {reason}")
            return convoy

    async def unblock_convoy(self, convoy_id: str) -> Convoy | None:
        """Resume a blocked convoy."""
        async with self._lock:
            convoy = self._convoys.get(convoy_id)
            if not convoy:
                return None

            if convoy.status != ConvoyStatus.BLOCKED:
                raise ValueError(f"Convoy {convoy_id} is not blocked")

            convoy.status = ConvoyStatus.IN_PROGRESS
            convoy.metadata.pop("block_reason", None)
            convoy.updated_at = datetime.utcnow()

            logger.info(f"Unblocked convoy {convoy_id}")
            return convoy

    async def submit_for_review(self, convoy_id: str) -> Convoy | None:
        """Submit a convoy for review."""
        async with self._lock:
            convoy = self._convoys.get(convoy_id)
            if not convoy:
                return None

            convoy.status = ConvoyStatus.REVIEW
            convoy.updated_at = datetime.utcnow()

            logger.info(f"Submitted convoy {convoy_id} for review")
            return convoy

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
            convoy = self._convoys.get(convoy_id)
            if not convoy:
                return None

            convoy.status = ConvoyStatus.COMPLETED
            convoy.completed_at = datetime.utcnow()
            if result:
                convoy.result = result
            convoy.updated_at = datetime.utcnow()

            logger.info(f"Completed convoy {convoy_id}")
            return convoy

    async def cancel_convoy(
        self,
        convoy_id: str,
        reason: str = "",
    ) -> Convoy | None:
        """Cancel a convoy."""
        async with self._lock:
            convoy = self._convoys.get(convoy_id)
            if not convoy:
                return None

            convoy.status = ConvoyStatus.CANCELLED
            convoy.error = reason
            convoy.updated_at = datetime.utcnow()

            logger.info(f"Cancelled convoy {convoy_id}: {reason}")
            return convoy

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
            convoy = self._convoys.get(convoy_id)
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
            convoy.artifacts.append(artifact_id)
            convoy.updated_at = datetime.utcnow()

            logger.debug(f"Added artifact {artifact_type} to convoy {convoy_id}")
            return artifact

    async def get_artifact(self, artifact_id: str) -> ConvoyArtifact | None:
        """Get an artifact by ID."""
        return self._artifacts.get(artifact_id)

    async def list_artifacts(self, convoy_id: str) -> list[ConvoyArtifact]:
        """List artifacts for a convoy."""
        convoy = self._convoys.get(convoy_id)
        if not convoy:
            return []

        return [self._artifacts[aid] for aid in convoy.artifacts if aid in self._artifacts]

    async def get_stats(self) -> dict[str, Any]:
        """Get convoy tracker statistics."""
        async with self._lock:
            by_status = {}
            for convoy in self._convoys.values():
                status = convoy.status.value
                by_status[status] = by_status.get(status, 0) + 1

            total_handoffs = sum(c.handoff_count for c in self._convoys.values())

            return {
                "convoys_total": len(self._convoys),
                "convoys_by_status": by_status,
                "artifacts_total": len(self._artifacts),
                "total_handoffs": total_handoffs,
            }
