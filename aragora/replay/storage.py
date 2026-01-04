import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ReplayStorage:
    """Manages replay storage and indexing."""
    
    def __init__(self, storage_dir: str = ".nomic/replays"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def list_recordings(self, limit: int = 50) -> List[Dict[str, Any]]:
        recordings = []
        for session_dir in self.storage_dir.iterdir():
            if session_dir.is_dir():
                meta_path = session_dir / "meta.json"
                if meta_path.exists():
                    try:
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)
                        recordings.append({
                            "id": meta.get("debate_id"),
                            "topic": meta.get("topic"),
                            "status": meta.get("status"),
                            "event_count": meta.get("event_count"),
                            "started_at": meta.get("started_at")
                        })
                    except Exception as e:
                        logger.debug(f"Skipping invalid replay metadata {meta_path}: {e}")
        recordings.sort(key=lambda x: x.get("started_at", ""), reverse=True)
        return recordings[:limit]
    
    def prune(self, keep_last: int = 100) -> None:
        recordings = self.list_recordings()
        if len(recordings) > keep_last:
            to_remove = recordings[keep_last:]
            for rec in to_remove:
                session_dir = self.storage_dir / rec["id"]
                if session_dir.exists():
                    import shutil
                    shutil.rmtree(session_dir)