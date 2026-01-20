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
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol

if TYPE_CHECKING:
    from aragora.knowledge.mound import KnowledgeMound

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
                        "step_id": checkpoint.current_step,
                        "steps_completed": len(checkpoint.completed_steps),
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
                f"step={checkpoint.current_step}, node_id={node_id}"
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
            # Fallback for objects without to_dict - matches WorkflowCheckpoint fields
            return {
                "id": getattr(checkpoint, "id", ""),
                "workflow_id": checkpoint.workflow_id,
                "definition_id": checkpoint.definition_id,
                "current_step": checkpoint.current_step,
                "completed_steps": list(checkpoint.completed_steps),
                "step_outputs": dict(checkpoint.step_outputs),
                "context_state": dict(getattr(checkpoint, "context_state", {})),
                "created_at": (
                    checkpoint.created_at.isoformat()
                    if hasattr(checkpoint.created_at, "isoformat")
                    else str(checkpoint.created_at)
                ),
                "checksum": getattr(checkpoint, "checksum", ""),
            }

    def _dict_to_checkpoint(self, data: Dict[str, Any]) -> WorkflowCheckpoint:
        """Convert dictionary back to WorkflowCheckpoint."""
        from datetime import datetime

        created_at = data.get("created_at", "")
        if isinstance(created_at, str) and created_at:
            try:
                created_at = datetime.fromisoformat(created_at)
            except ValueError:
                created_at = datetime.now()
        elif not isinstance(created_at, datetime):
            created_at = datetime.now()

        return WorkflowCheckpoint(
            id=data.get("id", ""),
            workflow_id=data.get("workflow_id", ""),
            definition_id=data.get("definition_id", ""),
            current_step=data.get("current_step", ""),
            completed_steps=list(data.get("completed_steps", [])),
            step_outputs=data.get("step_outputs", {}),
            context_state=data.get("context_state", {}),
            created_at=created_at,
            checksum=data.get("checksum", ""),
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
        return [f.stem for f in self.checkpoint_dir.glob(f"{workflow_id}_*.json")]

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
            # Fallback for objects without to_dict - matches WorkflowCheckpoint fields
            return {
                "id": getattr(checkpoint, "id", ""),
                "workflow_id": checkpoint.workflow_id,
                "definition_id": checkpoint.definition_id,
                "current_step": checkpoint.current_step,
                "completed_steps": list(checkpoint.completed_steps),
                "step_outputs": dict(checkpoint.step_outputs),
                "context_state": dict(getattr(checkpoint, "context_state", {})),
                "created_at": (
                    checkpoint.created_at.isoformat()
                    if hasattr(checkpoint.created_at, "isoformat")
                    else str(checkpoint.created_at)
                ),
                "checksum": getattr(checkpoint, "checksum", ""),
            }

    def _dict_to_checkpoint(self, data: Dict[str, Any]) -> WorkflowCheckpoint:
        """Convert dictionary to WorkflowCheckpoint."""
        from datetime import datetime

        created_at = data.get("created_at", "")
        if isinstance(created_at, str) and created_at:
            try:
                created_at = datetime.fromisoformat(created_at)
            except ValueError:
                created_at = datetime.now()
        elif not isinstance(created_at, datetime):
            created_at = datetime.now()

        return WorkflowCheckpoint(
            id=data.get("id", ""),
            workflow_id=data.get("workflow_id", ""),
            definition_id=data.get("definition_id", ""),
            current_step=data.get("current_step", ""),
            completed_steps=list(data.get("completed_steps", [])),
            step_outputs=data.get("step_outputs", {}),
            context_state=data.get("context_state", {}),
            created_at=created_at,
            checksum=data.get("checksum", ""),
        )


# Module-level default KnowledgeMound for checkpoint storage
_default_mound: Optional["KnowledgeMound"] = None


def set_default_knowledge_mound(mound: "KnowledgeMound") -> None:
    """
    Set the default KnowledgeMound for checkpoint storage.

    When set, get_checkpoint_store() will use KnowledgeMoundCheckpointStore
    instead of FileCheckpointStore, enabling durable checkpoint persistence
    with the Knowledge Mound backend.

    Usage:
        from aragora.knowledge.mound import KnowledgeMound
        from aragora.workflow.checkpoint_store import set_default_knowledge_mound

        mound = KnowledgeMound(workspace_id="production")
        await mound.initialize()
        set_default_knowledge_mound(mound)

    Args:
        mound: KnowledgeMound instance to use for checkpoints
    """
    global _default_mound
    _default_mound = mound
    logger.info("Set default KnowledgeMound for workflow checkpoints")


def get_default_knowledge_mound() -> Optional["KnowledgeMound"]:
    """Get the default KnowledgeMound for checkpoint storage."""
    return _default_mound


def get_checkpoint_store(
    mound: Optional["KnowledgeMound"] = None,
    fallback_dir: str = ".checkpoints",
    use_default_mound: bool = True,
) -> CheckpointStore:
    """
    Get the appropriate checkpoint store based on availability.

    Priority order:
    1. Explicitly provided KnowledgeMound
    2. Default KnowledgeMound (if set via set_default_knowledge_mound)
    3. FileCheckpointStore (persistent file-based fallback)

    Args:
        mound: Optional KnowledgeMound instance (highest priority)
        fallback_dir: Fallback directory for file-based storage
        use_default_mound: Whether to use the default mound if no mound provided

    Returns:
        CheckpointStore implementation
    """
    # Use explicitly provided mound
    if mound is not None:
        logger.debug("Using provided KnowledgeMound for checkpoints")
        return KnowledgeMoundCheckpointStore(mound)

    # Try default mound
    if use_default_mound and _default_mound is not None:
        logger.debug("Using default KnowledgeMound for checkpoints")
        return KnowledgeMoundCheckpointStore(_default_mound)

    # Fall back to file-based storage
    logger.debug(f"Using FileCheckpointStore in {fallback_dir}")
    return FileCheckpointStore(fallback_dir)
