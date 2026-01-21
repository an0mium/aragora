"""
Nomic Loop Checkpoint System.

Provides persistence for state machine state, enabling:
- Crash recovery (resume from last checkpoint)
- Manual pause/resume
- Audit trail of cycle progress
"""

import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Safe pattern for checkpoint identifiers (alphanumeric, hyphens, underscores)
_SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def _sanitize_checkpoint_id(value: str, param_name: str) -> str:
    """Sanitize a checkpoint identifier to prevent path traversal.

    Args:
        value: The value to sanitize
        param_name: Name of parameter (for error messages)

    Returns:
        The validated value

    Raises:
        ValueError: If value contains unsafe characters
    """
    if not value:
        return value
    if not _SAFE_ID_PATTERN.match(value):
        raise ValueError(
            f"Invalid {param_name}: must contain only alphanumeric, hyphen, underscore"
        )
    return value


# Checkpoint file naming
CHECKPOINT_PREFIX = "checkpoint"
CHECKPOINT_EXT = ".json"
LATEST_CHECKPOINT_NAME = "latest.json"


def get_checkpoint_path(checkpoint_dir: str, cycle_id: str, suffix: str = "") -> Path:
    """
    Get the path for a checkpoint file.

    Args:
        checkpoint_dir: Base directory for checkpoints
        cycle_id: The cycle ID (alphanumeric, hyphens, underscores only)
        suffix: Optional suffix (e.g., state name)

    Returns:
        Path to the checkpoint file

    Raises:
        ValueError: If cycle_id or suffix contains unsafe characters
    """
    # Sanitize inputs to prevent path traversal
    safe_cycle_id = _sanitize_checkpoint_id(cycle_id, "cycle_id")
    safe_suffix = _sanitize_checkpoint_id(suffix, "suffix")

    filename = f"{CHECKPOINT_PREFIX}_{safe_cycle_id}"
    if safe_suffix:
        filename += f"_{safe_suffix}"
    filename += CHECKPOINT_EXT
    return Path(checkpoint_dir) / filename


def save_checkpoint(
    data: Dict[str, Any],
    checkpoint_dir: str,
    cycle_id: str,
    suffix: str = "",
) -> str:
    """
    Save a checkpoint to disk.

    Args:
        data: The checkpoint data to save
        checkpoint_dir: Directory to save checkpoints
        cycle_id: The cycle ID
        suffix: Optional suffix for the filename

    Returns:
        Path to the saved checkpoint file
    """
    # Ensure directory exists
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Add metadata
    data["_checkpoint_meta"] = {
        "cycle_id": cycle_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "suffix": suffix,
    }

    # Write to temp file first (atomic write)
    filepath = get_checkpoint_path(checkpoint_dir, cycle_id, suffix)
    temp_path = filepath.with_suffix(".tmp")

    try:
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        # Atomic rename
        shutil.move(str(temp_path), str(filepath))

        # Also update 'latest' symlink/copy
        latest_path = checkpoint_path / LATEST_CHECKPOINT_NAME
        shutil.copy2(str(filepath), str(latest_path))

        logger.debug(f"Saved checkpoint: {filepath}")
        return str(filepath)

    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        # Clean up temp file if it exists
        if temp_path.exists():
            temp_path.unlink()
        raise


def load_checkpoint(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """
    Load a checkpoint from disk.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        The checkpoint data, or None if not found/invalid
    """
    path = Path(checkpoint_path)

    if not path.exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return None

    try:
        with open(path) as f:
            data = json.load(f)

        logger.debug(f"Loaded checkpoint: {checkpoint_path}")
        return data

    except json.JSONDecodeError as e:
        logger.error(f"Invalid checkpoint JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return None


def load_latest_checkpoint(checkpoint_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load the most recent checkpoint from a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        The latest checkpoint data, or None if none exist
    """
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        return None

    # Try 'latest' first
    latest_path = checkpoint_path / LATEST_CHECKPOINT_NAME
    if latest_path.exists():
        return load_checkpoint(str(latest_path))

    # Otherwise, find most recent checkpoint file
    checkpoints = list(checkpoint_path.glob(f"{CHECKPOINT_PREFIX}_*{CHECKPOINT_EXT}"))
    if not checkpoints:
        return None

    # Sort by modification time, most recent first
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return load_checkpoint(str(checkpoints[0]))


def list_checkpoints(checkpoint_dir: str) -> List[Dict[str, Any]]:
    """
    List all checkpoints in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        List of checkpoint metadata (without full data)
    """
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        return []

    checkpoints = []
    for filepath in checkpoint_path.glob(f"{CHECKPOINT_PREFIX}_*{CHECKPOINT_EXT}"):
        if filepath.name == LATEST_CHECKPOINT_NAME:
            continue

        try:
            with open(filepath) as f:
                data = json.load(f)

            meta = data.get("_checkpoint_meta", {})
            context = data.get("context", {})

            checkpoints.append(
                {
                    "path": str(filepath),
                    "cycle_id": meta.get("cycle_id", "unknown"),
                    "saved_at": meta.get("saved_at"),
                    "state": context.get("current_state", "unknown"),
                    "size_bytes": filepath.stat().st_size,
                }
            )
        except Exception as e:
            logger.warning(f"Could not read checkpoint {filepath}: {e}")

    # Sort by save time, most recent first
    checkpoints.sort(key=lambda c: c.get("saved_at", ""), reverse=True)
    return checkpoints


def delete_checkpoint(checkpoint_path: str) -> bool:
    """
    Delete a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        True if deleted, False otherwise
    """
    path = Path(checkpoint_path)

    if not path.exists():
        return False

    try:
        path.unlink()
        logger.debug(f"Deleted checkpoint: {checkpoint_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete checkpoint: {e}")
        return False


def cleanup_old_checkpoints(
    checkpoint_dir: str,
    keep_count: int = 10,
    keep_days: int = 7,
) -> int:
    """
    Clean up old checkpoints, keeping only recent ones.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_count: Maximum number of checkpoints to keep
        keep_days: Delete checkpoints older than this many days

    Returns:
        Number of checkpoints deleted
    """
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        return 0

    checkpoints = list(checkpoint_path.glob(f"{CHECKPOINT_PREFIX}_*{CHECKPOINT_EXT}"))
    if not checkpoints:
        return 0

    # Sort by modification time, oldest first
    checkpoints.sort(key=lambda p: p.stat().st_mtime)

    deleted = 0
    cutoff_time = datetime.now(timezone.utc).timestamp() - (keep_days * 24 * 3600)

    # Delete old checkpoints
    for filepath in checkpoints[:-keep_count]:  # Keep at least keep_count
        if filepath.name == LATEST_CHECKPOINT_NAME:
            continue

        # Check if old enough to delete
        if filepath.stat().st_mtime < cutoff_time:
            try:
                filepath.unlink()
                deleted += 1
                logger.debug(f"Cleaned up old checkpoint: {filepath}")
            except Exception as e:
                logger.warning(f"Could not delete checkpoint {filepath}: {e}")

    return deleted


class CheckpointManager:
    """
    High-level checkpoint management for the nomic loop.

    Provides automatic checkpoint rotation, compression,
    and recovery suggestions.
    """

    def __init__(
        self,
        checkpoint_dir: str = ".nomic/checkpoints",
        max_checkpoints: int = 20,
        auto_cleanup: bool = True,
    ):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoints
            max_checkpoints: Maximum checkpoints to keep
            auto_cleanup: Whether to auto-clean old checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.auto_cleanup = auto_cleanup

        # Ensure directory exists
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def save(
        self,
        data: Dict[str, Any],
        cycle_id: str,
        state_name: str = "",
    ) -> str:
        """
        Save a checkpoint.

        Args:
            data: Checkpoint data
            cycle_id: The cycle ID
            state_name: Current state name (for suffix)

        Returns:
            Path to saved checkpoint
        """
        path = save_checkpoint(data, self.checkpoint_dir, cycle_id, state_name)

        # Auto cleanup if enabled
        if self.auto_cleanup:
            cleanup_old_checkpoints(
                self.checkpoint_dir,
                keep_count=self.max_checkpoints,
            )

        return path

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint."""
        return load_latest_checkpoint(self.checkpoint_dir)

    def load(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """Load a specific checkpoint."""
        return load_checkpoint(checkpoint_path)

    def list_all(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        return list_checkpoints(self.checkpoint_dir)

    def get_recovery_options(self) -> List[Dict[str, Any]]:
        """
        Get available recovery options based on checkpoints.

        Returns:
            List of recovery options with recommendations
        """
        checkpoints = self.list_all()

        if not checkpoints:
            return [
                {
                    "option": "start_fresh",
                    "description": "No checkpoints found. Start a new cycle.",
                    "recommended": True,
                }
            ]

        options = []

        # Latest checkpoint
        latest = checkpoints[0]
        options.append(
            {
                "option": "resume_latest",
                "description": f"Resume from {latest['state']} (cycle {latest['cycle_id']})",
                "checkpoint_path": latest["path"],
                "recommended": True,
            }
        )

        # Previous checkpoints (up to 3 more)
        for i, cp in enumerate(checkpoints[1:4], start=1):
            options.append(
                {
                    "option": f"resume_older_{i}",
                    "description": f"Resume from {cp['state']} (cycle {cp['cycle_id']})",
                    "checkpoint_path": cp["path"],
                    "recommended": False,
                }
            )

        # Start fresh option
        options.append(
            {
                "option": "start_fresh",
                "description": "Discard checkpoints and start a new cycle",
                "recommended": False,
            }
        )

        return options

    def cleanup(self, keep_count: Optional[int] = None) -> int:
        """
        Manually clean up old checkpoints.

        Args:
            keep_count: Number to keep (uses default if None)

        Returns:
            Number of checkpoints deleted
        """
        return cleanup_old_checkpoints(
            self.checkpoint_dir,
            keep_count=keep_count or self.max_checkpoints,
        )
