"""
Debate recording and replay functionality.

Enables persistence and playback of past debates for review and learning.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from aragora.core import DebateResult

logger = logging.getLogger(__name__)


class DebateRecorder:
    """Records and persists debate results for later replay."""

    def __init__(self, storage_dir: str = "replays"):
        """Initialize recorder with storage directory.

        Args:
            storage_dir: Directory to store debate recordings
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

    def save_debate(self, result: DebateResult, metadata: Optional[Dict] = None) -> str:
        """Save a debate result to storage.

        Args:
            result: The DebateResult to save
            metadata: Optional additional metadata (e.g., agent names, protocol settings)

        Returns:
            The filename of the saved debate
        """
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debate_{timestamp}_{result.id[:8]}.json"
        filepath = self.storage_dir / filename

        # Prepare data for serialization
        data = {
            "debate_result": result.__dict__,
            "metadata": metadata or {},
            "recorded_at": datetime.now().isoformat(),
        }

        # Convert any non-serializable objects to strings
        data["debate_result"] = self._make_serializable(data["debate_result"])

        # Save to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return str(filepath)

    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, "__dict__"):
            # For dataclasses and objects, serialize their __dict__
            return self._make_serializable(obj.__dict__)
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # Convert other types to string
            return str(obj)


class DebateReplayer:
    """Loads and replays recorded debates."""

    def __init__(self, storage_dir: str = "replays"):
        """Initialize replayer with storage directory.

        Args:
            storage_dir: Directory containing debate recordings
        """
        self.storage_dir = Path(storage_dir)

    def list_debates(self) -> List[Dict]:
        """List all recorded debates with metadata.

        Returns:
            List of debate metadata dictionaries
        """
        debates: list[dict[str, Any]] = []
        if not self.storage_dir.exists():
            return debates

        for file_path in self.storage_dir.glob("debate_*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                debate_info = {
                    "filename": file_path.name,
                    "filepath": str(file_path),
                    "task": data.get("debate_result", {}).get("task", "Unknown"),
                    "consensus_reached": data.get("debate_result", {}).get(
                        "consensus_reached", False
                    ),
                    "confidence": data.get("debate_result", {}).get("confidence", 0.0),
                    "duration_seconds": data.get("debate_result", {}).get("duration_seconds", 0.0),
                    "rounds_used": data.get("debate_result", {}).get("rounds_used", 0),
                    "recorded_at": data.get("recorded_at", "Unknown"),
                    "metadata": data.get("metadata", {}),
                }
                debates.append(debate_info)
            except Exception as e:
                logger.warning("Could not load debate %s: %s", file_path, e)

        # Sort by recorded_at descending (newest first)
        debates.sort(key=lambda x: x.get("recorded_at", ""), reverse=True)
        return debates

    def load_debate(self, filename: str) -> Optional[DebateResult]:
        """Load a specific debate by filename.

        Args:
            filename: Name of the debate file (e.g., "debate_20231201_120000_abc12345.json")

        Returns:
            DebateResult if found, None otherwise
        """
        filepath = self.storage_dir / filename
        if not filepath.exists():
            return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Reconstruct DebateResult from dict
            result_data = data["debate_result"]
            result = DebateResult(**result_data)
            return result
        except Exception as e:
            logger.error("Error loading debate %s: %s", filename, e)
            return None

    def replay_debate(self, filename: str, speed: float = 1.0) -> Optional[DebateResult]:
        """Replay a debate with optional speed control.

        Args:
            filename: Debate file to replay
            speed: Playback speed multiplier (1.0 = normal speed)

        Returns:
            The debate result, or None if not found
        """
        result = self.load_debate(filename)
        if not result:
            return None

        print(f"\n{'='*80}")
        print(f"REPLAYING DEBATE: {result.task[:80]}...")
        print(f"Recorded: {len(result.messages)} messages, {result.rounds_used} rounds")
        print(f"Consensus: {'Yes' if result.consensus_reached else 'No'} ({result.confidence:.0%})")
        print(f"{'='*80}\n")

        # Replay messages in sequence
        import time

        for i, message in enumerate(result.messages, 1):
            # Handle both dict and Message objects (dicts come from JSON loading)
            if isinstance(message, dict):
                agent = message.get("agent", "unknown")
                content = message.get("content", "")
            else:
                agent = message.agent
                content = message.content
            print(f"[{i:2d}] {agent}: {content}")
            if speed < 10.0:  # Don't delay for very fast replays
                time.sleep(0.5 / speed)  # Brief pause between messages

        print(f"\nFinal Answer: {result.final_answer}")
        print(f"Duration: {result.duration_seconds:.1f}s")
        return result
