"""
Checkpoint module for crash recovery.

Provides atomic save/load operations for implementation progress,
enabling resumption after crashes or interruptions.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from .types import ImplementProgress


PROGRESS_FILENAME = "implement_progress.json"


def get_progress_path(repo_path: Path) -> Path:
    """Get the path to the progress file."""
    return repo_path / ".nomic" / PROGRESS_FILENAME


def save_progress(progress: ImplementProgress, repo_path: Path) -> None:
    """
    Atomically save progress to disk.

    Uses write-to-temp + rename for atomicity to prevent corruption
    if the process is killed mid-write.
    """
    progress_path = get_progress_path(repo_path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file first
    fd, temp_path = tempfile.mkstemp(
        dir=progress_path.parent,
        prefix=".progress_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(progress.to_dict(), f, indent=2)
        # Atomic rename
        os.rename(temp_path, progress_path)
    except Exception:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def load_progress(repo_path: Path) -> Optional[ImplementProgress]:
    """
    Load progress from disk if it exists.

    Returns None if no progress file exists or if it's corrupted.
    """
    progress_path = get_progress_path(repo_path)

    if not progress_path.exists():
        return None

    try:
        with open(progress_path) as f:
            data = json.load(f)
        return ImplementProgress.from_dict(data)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        # Corrupted file - log and return None
        print(f"Warning: Corrupted progress file, starting fresh: {e}")
        return None


def clear_progress(repo_path: Path) -> None:
    """
    Clear progress file after successful completion.
    """
    progress_path = get_progress_path(repo_path)
    if progress_path.exists():
        progress_path.unlink()


def update_current_task(repo_path: Path, task_id: str) -> None:
    """
    Update just the current task in progress (lightweight update).
    """
    progress = load_progress(repo_path)
    if progress:
        progress.current_task = task_id
        save_progress(progress, repo_path)
