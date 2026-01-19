"""
Replay Storage - Manages debate replay persistence and retrieval.

Handles storage, indexing, and pruning of debate recordings. Each recording
is stored as a directory containing metadata (meta.json) and events
(events.jsonl).

Usage:
    from aragora.replay.storage import ReplayStorage

    storage = ReplayStorage(storage_dir=".nomic/replays")

    # List available recordings
    recordings = storage.list_recordings(limit=50)

    # Load a specific recording
    meta, events = storage.load("debate-123")

    # Prune old recordings
    storage.prune(keep_last=100)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ReplayStorage:
    """Manages replay storage and indexing."""

    def __init__(self, storage_dir: str = ".nomic/replays"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def list_recordings(self, limit: int = 50) -> List[Dict[str, Any]]:
        recordings = []
        try:
            dir_entries = list(self.storage_dir.iterdir())
        except OSError as e:
            logger.warning(f"Failed to list replay directory {self.storage_dir}: {e}")
            return []

        for session_dir in dir_entries:
            try:
                if not session_dir.is_dir():
                    continue
                meta_path = session_dir / "meta.json"
                if not meta_path.exists():
                    continue
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                recordings.append(
                    {
                        "id": meta.get("debate_id"),
                        "topic": meta.get("topic"),
                        "status": meta.get("status"),
                        "event_count": meta.get("event_count"),
                        "started_at": meta.get("started_at"),
                    }
                )
            except (OSError, json.JSONDecodeError) as e:
                logger.debug(f"Skipping invalid replay {session_dir}: {e}")
        recordings.sort(key=lambda x: x.get("started_at", ""), reverse=True)
        return recordings[:limit]

    def prune(self, keep_last: int = 100) -> int:
        """Prune old recordings, keeping only the most recent ones.

        Returns:
            Number of recordings successfully removed.
        """
        import shutil

        recordings = self.list_recordings()
        if len(recordings) <= keep_last:
            return 0

        removed = 0
        to_remove = recordings[keep_last:]
        for rec in to_remove:
            rec_id = rec.get("id")
            if not rec_id:
                continue
            session_dir = self.storage_dir / rec_id
            try:
                if session_dir.exists():
                    shutil.rmtree(session_dir)
                    removed += 1
            except OSError as e:
                logger.warning(f"Failed to remove replay {rec_id}: {e}")
        return removed
