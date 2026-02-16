"""
KnowledgeMound-backed checkpoint store implementation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

from aragora.workflow.types import WorkflowCheckpoint

if TYPE_CHECKING:
    from aragora.knowledge.mound import KnowledgeMound

logger = logging.getLogger(__name__)


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
        mound: KnowledgeMound,
        workspace_id: str | None = None,
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
            # Import KM types dynamically - these may not be available at type-check time
            # as the KM module has multiple implementations
            km_module = __import__(
                "aragora.knowledge.mound",
                fromlist=["KnowledgeNode", "MemoryTier", "ProvenanceChain"],
            )
            KnowledgeNode: Any = getattr(km_module, "KnowledgeNode")
            MemoryTier: Any = getattr(km_module, "MemoryTier")
            ProvenanceChain: Any = getattr(km_module, "ProvenanceChain")

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

            # Store in mound - mound interface is duck-typed
            add_node_method: Any = getattr(self.mound, "add_node")
            node_id: str = await add_node_method(node)
            logger.info(
                f"Saved workflow checkpoint: workflow={checkpoint.workflow_id}, "
                f"step={checkpoint.current_step}, node_id={node_id}"
            )
            return node_id

        except (ImportError, AttributeError, RuntimeError, ValueError, TypeError, OSError) as e:
            logger.error(f"Failed to save checkpoint to KnowledgeMound: {e}")
            raise

    async def load(self, checkpoint_id: str) -> WorkflowCheckpoint | None:
        """
        Load a checkpoint by its node ID.

        Args:
            checkpoint_id: KnowledgeNode ID

        Returns:
            WorkflowCheckpoint or None if not found
        """
        try:
            # Duck-typed mound interface - get_node may have varying signatures
            get_node: Callable[..., Any] = getattr(self.mound, "get_node")
            node: Any = await get_node(checkpoint_id)
            if node is None:
                return None

            if getattr(node, "node_type", None) != "workflow_checkpoint":
                logger.warning(f"Node {checkpoint_id} is not a checkpoint")
                return None

            checkpoint_dict = json.loads(node.content)
            return self._dict_to_checkpoint(checkpoint_dict)

        except (AttributeError, RuntimeError, ValueError, TypeError, OSError, KeyError) as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None

    async def load_latest(self, workflow_id: str) -> WorkflowCheckpoint | None:
        """
        Load the most recent checkpoint for a workflow.

        Args:
            workflow_id: Workflow ID to find checkpoint for

        Returns:
            Most recent WorkflowCheckpoint or None
        """
        try:
            # Query for checkpoints with this workflow ID
            # Duck-typed mound interface - query_by_provenance may not exist on all impls
            query_method: Callable[..., Any] = getattr(self.mound, "query_by_provenance")
            nodes: list[Any] = await query_method(
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

        except (AttributeError, RuntimeError, ValueError, TypeError, OSError, KeyError) as e:
            logger.error(f"Failed to load latest checkpoint for {workflow_id}: {e}")
            return None

    async def list_checkpoints(self, workflow_id: str) -> list[str]:
        """
        List all checkpoint IDs for a workflow.

        Args:
            workflow_id: Workflow ID

        Returns:
            List of checkpoint node IDs
        """
        try:
            # Duck-typed mound interface
            query_method: Callable[..., Any] = getattr(self.mound, "query_by_provenance")
            nodes: list[Any] = await query_method(
                source_type="workflow_engine",
                source_id=workflow_id,
                node_type="workflow_checkpoint",
                limit=100,
            )
            return [getattr(node, "id", "") for node in nodes]

        except (AttributeError, RuntimeError, ValueError, TypeError, OSError) as e:
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
            # Duck-typed mound interface
            delete_method: Callable[..., Any] = getattr(self.mound, "delete_node")
            result: bool = await delete_method(checkpoint_id)
            return result
        except (AttributeError, RuntimeError, ValueError, TypeError, OSError) as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False

    def _checkpoint_to_dict(self, checkpoint: WorkflowCheckpoint) -> dict[str, Any]:
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

    def _dict_to_checkpoint(self, data: dict[str, Any]) -> WorkflowCheckpoint:
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
