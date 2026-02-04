"""
File-based checkpoint store implementation (fallback).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from aragora.workflow.types import WorkflowCheckpoint

logger = logging.getLogger(__name__)


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

    async def load(self, checkpoint_id: str) -> WorkflowCheckpoint | None:
        """Load a checkpoint from a file."""
        file_path = self.checkpoint_dir / f"{checkpoint_id}.json"
        if not file_path.exists():
            return None

        data = json.loads(file_path.read_text())
        return self._dict_to_checkpoint(data)

    async def load_latest(self, workflow_id: str) -> WorkflowCheckpoint | None:
        """Load the most recent checkpoint for a workflow."""
        matching_files = sorted(
            self.checkpoint_dir.glob(f"{workflow_id}_*.json"),
            reverse=True,
        )
        if not matching_files:
            return None

        data = json.loads(matching_files[0].read_text())
        return self._dict_to_checkpoint(data)

    async def list_checkpoints(self, workflow_id: str) -> list[str]:
        """List all checkpoint IDs for a workflow."""
        return [f.stem for f in self.checkpoint_dir.glob(f"{workflow_id}_*.json")]

    async def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint file."""
        file_path = self.checkpoint_dir / f"{checkpoint_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def _checkpoint_to_dict(self, checkpoint: WorkflowCheckpoint) -> dict[str, Any]:
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

    def _dict_to_checkpoint(self, data: dict[str, Any]) -> WorkflowCheckpoint:
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
