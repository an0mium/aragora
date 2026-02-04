"""
Checkpoint store protocol definition.
"""

from __future__ import annotations

from typing import Protocol

from aragora.workflow.types import WorkflowCheckpoint


class CheckpointStore(Protocol):
    """Protocol for checkpoint storage backends."""

    async def save(self, checkpoint: WorkflowCheckpoint) -> str:
        """Save a checkpoint and return its ID."""
        ...

    async def load(self, checkpoint_id: str) -> WorkflowCheckpoint | None:
        """Load a checkpoint by ID."""
        ...

    async def load_latest(self, workflow_id: str) -> WorkflowCheckpoint | None:
        """Load the most recent checkpoint for a workflow."""
        ...

    async def list_checkpoints(self, workflow_id: str) -> list[str]:
        """List all checkpoint IDs for a workflow."""
        ...

    async def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        ...
