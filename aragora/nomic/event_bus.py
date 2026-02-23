"""Cross-Worktree Event Bus.

File-based IPC for coordinating multiple worktrees. Each event is written
as a JSON file to the `.aragora_events/` directory. Worktrees poll this
directory to discover events from other tracks.

Event types:
- task_claimed: "I'm working on X, don't touch these files"
- task_completed: "X is done, ready for merge"
- conflict_detected: "My changes overlap with track Y"
- sync_requested: "Please rebase me onto latest main"

Usage:
    from aragora.nomic.event_bus import EventBus, WorktreeEvent

    bus = EventBus()
    bus.publish("task_claimed", track="core", data={"files": ["orchestrator.py"]})
    events = bus.poll(since_minutes=30)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WorktreeEvent:
    """An event published to the cross-worktree event bus."""

    event_type: str  # task_claimed | task_completed | conflict_detected | sync_requested
    track: str
    timestamp: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    event_id: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.event_id:
            ts = int(time.time() * 1000)
            self.event_id = f"{self.track}-{self.event_type}-{ts}"


# Valid event types
VALID_EVENT_TYPES = frozenset({
    "task_claimed",
    "task_completed",
    "conflict_detected",
    "sync_requested",
    "merge_ready",
    "merge_completed",
    "error",
})


class EventBus:
    """File-based event bus for cross-worktree communication.

    Events are stored as individual JSON files in the events directory.
    Each file is named with a timestamp for easy ordering and cleanup.
    """

    def __init__(
        self,
        events_dir: Path | None = None,
        repo_root: Path | None = None,
        max_age_hours: float = 24.0,
    ):
        self.repo_root = repo_root or Path.cwd()
        self.events_dir = events_dir or (self.repo_root / ".aragora_events")
        self.max_age_hours = max_age_hours

    def _ensure_dir(self) -> None:
        """Ensure events directory exists."""
        self.events_dir.mkdir(parents=True, exist_ok=True)

    def publish(
        self,
        event_type: str,
        track: str,
        data: dict[str, Any] | None = None,
    ) -> WorktreeEvent:
        """Publish an event to the bus.

        Args:
            event_type: Type of event (e.g., "task_claimed")
            track: Track publishing the event
            data: Event payload data

        Returns:
            The published WorktreeEvent
        """
        if event_type not in VALID_EVENT_TYPES:
            raise ValueError(
                f"Invalid event type: {event_type}. "
                f"Valid types: {', '.join(sorted(VALID_EVENT_TYPES))}"
            )

        event = WorktreeEvent(
            event_type=event_type,
            track=track,
            data=data or {},
        )

        self._ensure_dir()

        # Write event as JSON file
        ts = int(time.time() * 1000)
        filename = f"{ts}_{track}_{event_type}.json"
        event_path = self.events_dir / filename

        with open(event_path, "w") as f:
            json.dump(asdict(event), f, indent=2)

        logger.info(
            "event_published type=%s track=%s id=%s",
            event_type, track, event.event_id,
        )
        return event

    def poll(
        self,
        since_minutes: float = 60.0,
        event_type: str | None = None,
        exclude_track: str | None = None,
    ) -> list[WorktreeEvent]:
        """Poll for recent events.

        Args:
            since_minutes: Only return events from the last N minutes
            event_type: Filter by event type
            exclude_track: Exclude events from this track

        Returns:
            List of matching events, oldest first
        """
        if not self.events_dir.exists():
            return []

        cutoff = time.time() - (since_minutes * 60)
        events: list[WorktreeEvent] = []

        for event_file in sorted(self.events_dir.glob("*.json")):
            # Check file modification time for quick cutoff
            if event_file.stat().st_mtime < cutoff:
                continue

            try:
                with open(event_file) as f:
                    data = json.load(f)

                event = WorktreeEvent(
                    event_type=data.get("event_type", ""),
                    track=data.get("track", ""),
                    timestamp=data.get("timestamp", ""),
                    data=data.get("data", {}),
                    event_id=data.get("event_id", ""),
                )

                # Apply filters
                if event_type and event.event_type != event_type:
                    continue
                if exclude_track and event.track == exclude_track:
                    continue

                events.append(event)

            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.debug("Skipping invalid event file %s: %s", event_file, e)
                continue

        return events

    def get_claimed_files(self) -> dict[str, list[str]]:
        """Get currently claimed files by track.

        Returns:
            Mapping of track -> list of claimed file paths
        """
        claims: dict[str, list[str]] = {}

        # Get the latest task_claimed event per track
        events = self.poll(
            since_minutes=self.max_age_hours * 60,
            event_type="task_claimed",
        )

        for event in events:
            files = event.data.get("files", [])
            if files:
                claims[event.track] = files

        return claims

    def get_completed_tracks(self) -> list[str]:
        """Get tracks that have published task_completed events.

        Returns:
            List of track names that have completed their work
        """
        events = self.poll(event_type="task_completed")
        return list({e.track for e in events})

    def get_merge_ready(self) -> list[WorktreeEvent]:
        """Get tracks that are ready for merge.

        Returns:
            List of merge_ready events
        """
        return self.poll(event_type="merge_ready")

    def cleanup(self, max_age_hours: float | None = None) -> int:
        """Remove old event files.

        Args:
            max_age_hours: Maximum event age (default: self.max_age_hours)

        Returns:
            Number of events cleaned up
        """
        if not self.events_dir.exists():
            return 0

        cutoff = time.time() - ((max_age_hours or self.max_age_hours) * 3600)
        cleaned = 0

        for event_file in self.events_dir.glob("*.json"):
            if event_file.stat().st_mtime < cutoff:
                event_file.unlink(missing_ok=True)
                cleaned += 1

        if cleaned:
            logger.info("event_cleanup removed=%d", cleaned)

        return cleaned

    def clear_all(self) -> int:
        """Remove all events. Use for testing or reset.

        Returns:
            Number of events removed
        """
        if not self.events_dir.exists():
            return 0

        count = 0
        for event_file in self.events_dir.glob("*.json"):
            event_file.unlink(missing_ok=True)
            count += 1
        return count


__all__ = [
    "EventBus",
    "WorktreeEvent",
    "VALID_EVENT_TYPES",
]
