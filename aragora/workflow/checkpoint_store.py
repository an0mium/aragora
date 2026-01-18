"""
Checkpoint Store implementations for Workflow Engine.

Provides persistent storage for workflow checkpoints:
- KnowledgeMoundCheckpointStore: Stores checkpoints in KnowledgeMound
- FileCheckpointStore: Stores checkpoints as local files (fallback)

Checkpoints enable:
- Crash recovery and resume
- Long-running workflow persistence
- Audit trail of workflow progress
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from aragora.workflow.types import WorkflowCheckpoint

logger = logging.getLogger(__name__)


class CheckpointStore(Protocol):
    """Protocol for checkpoint storage backends."""

    async def save(self, checkpoint: WorkflowCheckpoint) -> str:
        """Save a checkpoint and return its ID."""
        ...

    async def load(self, checkpoint_id: str) -> Optional[WorkflowCheckpoint]:
        """Load a checkpoint by ID."""
        ...

    async def load_latest(self, workflow_id: str) -> Optional[WorkflowCheckpoint]:
        """Load the most recent checkpoint for a workflow."""
        ...

    async def list_checkpoints(self, workflow_id: str) -> List[str]:
        """List all checkpoint IDs for a workflow."""
        ...

    async def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        ...


class KnowledgeMoundCheckpointStore:
    """
    Stores workflow checkpoints in KnowledgeMound.

    Checkpoints are stored as KnowledgeNodes with:
    - node_type: "workflow_checkpoint"
    - content: Serialized checkpoint state
    - provenance: Workflow ID, step ID, timestamp
    - tier: MEDIUM (balance between persistence and cleanup)

    This enables:
    - Unified storage with other knowledge
    - Cross-workspace checkpoint access
    - Automatic tier-based cleanup
    - Semantic search over checkpoints

    Usage:
        from aragora.knowledge.mound import KnowledgeMound
        from aragora.workflow.checkpoint_store import KnowledgeMoundCheckpointStore

        mound = KnowledgeMound(workspace_id="my_workspace")
        store = KnowledgeMoundCheckpointStore(mound)

        # Save checkpoint
        checkpoint_id = await store.save(checkpoint)

        # Resume from checkpoint
        checkpoint = await store.load_latest("workflow_123")
    """

    def __init__(
        self,
        mound: "KnowledgeMound",
        workspace_id: Optional[str] = None,
    ):
        """
        Initialize checkpoint store with KnowledgeMound backend.

        Args:
            mound: KnowledgeMound instance for storage
            workspace_id: Optional workspace override (defaults to mound's workspace)
        """
        self.mound = mound
        self.workspace_id = workspace_id or getattr(mound, "_workspace_id", "default")

    async def save(self, checkpoint: WorkflowCheckpoint) -> str:
        """
        Save a checkpoint to KnowledgeMound.

        Args:
            checkpoint: WorkflowCheckpoint to save

        Returns:
            Checkpoint node ID
        """
        try:
            from aragora.knowledge.mound import KnowledgeNode, MemoryTier, ProvenanceChain

            # Serialize checkpoint to JSON
            checkpoint_dict = self._checkpoint_to_dict(checkpoint)
            content = json.dumps(checkpoint_dict, indent=2, default=str)

            # Build provenance chain
            provenance = ProvenanceChain(
                source_type="workflow_engine",
                source_id=checkpoint.workflow_id,
                timestamp=datetime.now().isoformat(),
                chain=[
                    {
                        "workflow_id": checkpoint.workflow_id,
                        "step_id": checkpoint.current_step_id,
                        "status": checkpoint.status,
                    }
                ],
                metadata={
                    "checkpoint_type": "workflow",
                    "steps_completed": len(checkpoint.completed_steps),
                },
            )

            # Create knowledge node
            node = KnowledgeNode(
                node_type="workflow_checkpoint",
                content=content,
                confidence=1.0,  # Checkpoints are authoritative
                provenance=provenance,
                tier=MemoryTier.MEDIUM,  # Balance persistence and cleanup
                workspace_id=self.workspace_id,
            )

            # Store in mound
            node_id = await self.mound.add_node(node)
            logger.info(
                f"Saved workflow checkpoint: workflow={checkpoint.workflow_id}, "
                f"step={checkpoint.current_step_id}, node_id={node_id}"
            )
            return node_id

        except Exception as e:
            logger.error(f"Failed to save checkpoint to KnowledgeMound: {e}")
            raise

    async def load(self, checkpoint_id: str) -> Optional[WorkflowCheckpoint]:
        """
        Load a checkpoint by its node ID.

        Args:
            checkpoint_id: KnowledgeNode ID

        Returns:
            WorkflowCheckpoint or None if not found
        """
        try:
            node = await self.mound.get_node(checkpoint_id)
            if node is None:
                return None

            if node.node_type != "workflow_checkpoint":
                logger.warning(f"Node {checkpoint_id} is not a checkpoint")
                return None

            checkpoint_dict = json.loads(node.content)
            return self._dict_to_checkpoint(checkpoint_dict)

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None

    async def load_latest(self, workflow_id: str) -> Optional[WorkflowCheckpoint]:
        """
        Load the most recent checkpoint for a workflow.

        Args:
            workflow_id: Workflow ID to find checkpoint for

        Returns:
            Most recent WorkflowCheckpoint or None
        """
        try:
            # Query for checkpoints with this workflow ID
            nodes = await self.mound.query_by_provenance(
                source_type="workflow_engine",
                source_id=workflow_id,
                node_type="workflow_checkpoint",
                limit=1,
            )

            if not nodes:
                return None

            # Get the most recent (should be first due to ordering)
            latest_node = nodes[0]
            checkpoint_dict = json.loads(latest_node.content)
            return self._dict_to_checkpoint(checkpoint_dict)

        except Exception as e:
            logger.error(f"Failed to load latest checkpoint for {workflow_id}: {e}")
            return None

    async def list_checkpoints(self, workflow_id: str) -> List[str]:
        """
        List all checkpoint IDs for a workflow.

        Args:
            workflow_id: Workflow ID

        Returns:
            List of checkpoint node IDs
        """
        try:
            nodes = await self.mound.query_by_provenance(
                source_type="workflow_engine",
                source_id=workflow_id,
                node_type="workflow_checkpoint",
                limit=100,
            )
            return [node.id for node in nodes]

        except Exception as e:
            logger.error(f"Failed to list checkpoints for {workflow_id}: {e}")
            return []

    async def delete(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.

        Args:
            checkpoint_id: Checkpoint node ID to delete

        Returns:
            True if deleted, False otherwise
        """
        try:
            return await self.mound.delete_node(checkpoint_id)
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False

    def _checkpoint_to_dict(self, checkpoint: WorkflowCheckpoint) -> Dict[str, Any]:
        """Convert checkpoint to dictionary for serialization."""
        if hasattr(checkpoint, "to_dict"):
            return checkpoint.to_dict()
        elif hasattr(checkpoint, "__dataclass_fields__"):
            return asdict(checkpoint)
        else:
            return {
                "workflow_id": checkpoint.workflow_id,
                "definition_id": checkpoint.definition_id,
                "current_step_id": checkpoint.current_step_id,
                "completed_steps": list(checkpoint.completed_steps),
                "step_outputs": dict(checkpoint.step_outputs),
                "state": dict(checkpoint.state),
                "status": checkpoint.status,
                "created_at": checkpoint.created_at,
                "updated_at": checkpoint.updated_at,
            }

    def _dict_to_checkpoint(self, data: Dict[str, Any]) -> WorkflowCheckpoint:
        """Convert dictionary back to WorkflowCheckpoint."""
        return WorkflowCheckpoint(
            workflow_id=data.get("workflow_id", ""),
            definition_id=data.get("definition_id", ""),
            current_step_id=data.get("current_step_id"),
            completed_steps=set(data.get("completed_steps", [])),
            step_outputs=data.get("step_outputs", {}),
            state=data.get("state", {}),
            status=data.get("status", "pending"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


class FileCheckpointStore:
    """
    Fallback checkpoint store using local files.

    Stores checkpoints as JSON files in a specified directory.
    Useful for development or when KnowledgeMound is unavailable.

    Usage:
        store = FileCheckpointStore("/path/to/checkpoints")
        checkpoint_id = await store.save(checkpoint)
    """

    def __init__(self, checkpoint_dir: str = ".checkpoints"):
        """
        Initialize file-based checkpoint store.

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    async def save(self, checkpoint: WorkflowCheckpoint) -> str:
        """Save a checkpoint to a file."""
        checkpoint_id = f"{checkpoint.workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        file_path = self.checkpoint_dir / f"{checkpoint_id}.json"

        checkpoint_dict = self._checkpoint_to_dict(checkpoint)
        checkpoint_dict["checkpoint_id"] = checkpoint_id

        file_path.write_text(json.dumps(checkpoint_dict, indent=2, default=str))
        logger.info(f"Saved checkpoint to file: {file_path}")
        return checkpoint_id

    async def load(self, checkpoint_id: str) -> Optional[WorkflowCheckpoint]:
        """Load a checkpoint from a file."""
        file_path = self.checkpoint_dir / f"{checkpoint_id}.json"
        if not file_path.exists():
            return None

        data = json.loads(file_path.read_text())
        return self._dict_to_checkpoint(data)

    async def load_latest(self, workflow_id: str) -> Optional[WorkflowCheckpoint]:
        """Load the most recent checkpoint for a workflow."""
        matching_files = sorted(
            self.checkpoint_dir.glob(f"{workflow_id}_*.json"),
            reverse=True,
        )
        if not matching_files:
            return None

        data = json.loads(matching_files[0].read_text())
        return self._dict_to_checkpoint(data)

    async def list_checkpoints(self, workflow_id: str) -> List[str]:
        """List all checkpoint IDs for a workflow."""
        return [
            f.stem
            for f in self.checkpoint_dir.glob(f"{workflow_id}_*.json")
        ]

    async def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint file."""
        file_path = self.checkpoint_dir / f"{checkpoint_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def _checkpoint_to_dict(self, checkpoint: WorkflowCheckpoint) -> Dict[str, Any]:
        """Convert checkpoint to dictionary."""
        if hasattr(checkpoint, "to_dict"):
            return checkpoint.to_dict()
        elif hasattr(checkpoint, "__dataclass_fields__"):
            return asdict(checkpoint)
        else:
            return {
                "workflow_id": checkpoint.workflow_id,
                "definition_id": checkpoint.definition_id,
                "current_step_id": checkpoint.current_step_id,
                "completed_steps": list(checkpoint.completed_steps),
                "step_outputs": dict(checkpoint.step_outputs),
                "state": dict(checkpoint.state),
                "status": checkpoint.status,
                "created_at": checkpoint.created_at,
                "updated_at": checkpoint.updated_at,
            }

    def _dict_to_checkpoint(self, data: Dict[str, Any]) -> WorkflowCheckpoint:
        """Convert dictionary to WorkflowCheckpoint."""
        return WorkflowCheckpoint(
            workflow_id=data.get("workflow_id", ""),
            definition_id=data.get("definition_id", ""),
            current_step_id=data.get("current_step_id"),
            completed_steps=set(data.get("completed_steps", [])),
            step_outputs=data.get("step_outputs", {}),
            state=data.get("state", {}),
            status=data.get("status", "pending"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


def get_checkpoint_store(
    mound: Optional["KnowledgeMound"] = None,
    fallback_dir: str = ".checkpoints",
) -> CheckpointStore:
    """
    Get the appropriate checkpoint store based on availability.

    Args:
        mound: Optional KnowledgeMound instance
        fallback_dir: Fallback directory for file-based storage

    Returns:
        CheckpointStore implementation
    """
    if mound is not None:
        return KnowledgeMoundCheckpointStore(mound)
    return FileCheckpointStore(fallback_dir)
