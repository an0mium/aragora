"""
Replay Reader for loading and navigating debate recordings.

Provides filtering, seeking, and integrity validation for replay data.
"""

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

from .schema import ReplayEvent, ReplayMeta

logger = logging.getLogger(__name__)


class ReplayReader:
    """
    Loads and reads replay data with filtering, seeking, and validation.

    Usage:
        reader = ReplayReader("/path/to/session")

        # Basic iteration
        for event in reader.iter_events():
            print(event.content)

        # Filtered iteration
        for event in reader.filter_by_type("turn"):
            print(event.content)

        # Seek to offset
        for event in reader.seek_to_offset(5000):  # 5 seconds in
            print(event.content)

        # Get statistics
        stats = reader.get_stats()
        print(f"Total events: {stats['total_events']}")

        # Validate integrity
        is_valid, errors = reader.validate_integrity()
    """

    def __init__(self, session_dir: str):
        """
        Initialize the replay reader.

        Args:
            session_dir: Path to the session directory containing meta.json and events.jsonl
        """
        self.session_dir = Path(session_dir)
        self.meta_path = self.session_dir / "meta.json"
        self.events_path = self.session_dir / "events.jsonl"
        self.meta: Optional[ReplayMeta] = None
        self._load_error: Optional[str] = None
        self._event_index: Optional[List[Tuple[int, int, str]]] = (
            None  # (offset_ms, file_pos, event_id)
        )

        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load replay metadata from meta.json."""
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = ReplayMeta.from_json(f.read())
        except FileNotFoundError:
            self._load_error = f"Replay metadata not found: {self.meta_path}"
            logger.warning(self._load_error)
        except json.JSONDecodeError as e:
            self._load_error = f"Corrupted replay metadata: {e.msg} at pos {e.pos}"
            logger.warning(self._load_error)
        except Exception as e:
            self._load_error = f"Failed to load replay: {type(e).__name__}: {e}"
            logger.warning(self._load_error)

    @property
    def is_valid(self) -> bool:
        """Check if the replay was loaded successfully."""
        return self._load_error is None and self.meta is not None

    @property
    def load_error(self) -> Optional[str]:
        """Get the load error message if any."""
        return self._load_error

    def iter_events(self) -> Iterator[ReplayEvent]:
        """
        Iterate over all events in the replay.

        Yields:
            ReplayEvent objects in chronological order

        Note:
            Corrupted events are skipped with a warning logged.
        """
        if not self.events_path.exists():
            return
        try:
            with open(self.events_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        yield ReplayEvent.from_jsonl(line)
                    except (json.JSONDecodeError, ValueError, TypeError) as e:
                        logger.warning(f"Skipping corrupted event at line {line_num}: {e}")
                        continue
        except Exception as e:
            logger.error(f"Failed to read events file: {type(e).__name__}: {e}")

    def filter_by_type(self, event_type: str) -> Iterator[ReplayEvent]:
        """
        Filter events by event type.

        Args:
            event_type: Event type to filter for ('turn', 'vote', 'audience_input',
                       'phase_change', 'system')

        Yields:
            ReplayEvent objects matching the specified type
        """
        for event in self.iter_events():
            if event.event_type == event_type:
                yield event

    def filter_by_types(self, event_types: Set[str]) -> Iterator[ReplayEvent]:
        """
        Filter events by multiple event types.

        Args:
            event_types: Set of event types to include

        Yields:
            ReplayEvent objects matching any of the specified types
        """
        for event in self.iter_events():
            if event.event_type in event_types:
                yield event

    def filter_by_agent(self, agent_id: str) -> Iterator[ReplayEvent]:
        """
        Filter events by agent/source ID.

        Args:
            agent_id: Agent or source ID to filter for

        Yields:
            ReplayEvent objects from the specified agent
        """
        for event in self.iter_events():
            if event.source == agent_id:
                yield event

    def filter_by_agents(self, agent_ids: Set[str]) -> Iterator[ReplayEvent]:
        """
        Filter events by multiple agents.

        Args:
            agent_ids: Set of agent/source IDs to include

        Yields:
            ReplayEvent objects from any of the specified agents
        """
        for event in self.iter_events():
            if event.source in agent_ids:
                yield event

    def filter(
        self,
        predicate: Callable[[ReplayEvent], bool],
    ) -> Iterator[ReplayEvent]:
        """
        Filter events using a custom predicate function.

        Args:
            predicate: Function that returns True for events to include

        Yields:
            ReplayEvent objects for which predicate returns True
        """
        for event in self.iter_events():
            if predicate(event):
                yield event

    def seek_to_offset(self, offset_ms: int) -> Iterator[ReplayEvent]:
        """
        Seek to a specific offset and iterate from there.

        Args:
            offset_ms: Offset in milliseconds from debate start

        Yields:
            ReplayEvent objects starting from the specified offset
        """
        for event in self.iter_events():
            if event.offset_ms >= offset_ms:
                yield event

    def seek_to_event(self, event_id: str) -> Iterator[ReplayEvent]:
        """
        Seek to a specific event by ID and iterate from there.

        Args:
            event_id: The event ID to seek to

        Yields:
            ReplayEvent objects starting from the specified event (inclusive)
        """
        found = False
        for event in self.iter_events():
            if event.event_id == event_id:
                found = True
            if found:
                yield event

    def get_events_in_range(
        self,
        start_ms: int,
        end_ms: int,
    ) -> Iterator[ReplayEvent]:
        """
        Get events within a time range.

        Args:
            start_ms: Start offset in milliseconds (inclusive)
            end_ms: End offset in milliseconds (inclusive)

        Yields:
            ReplayEvent objects within the specified time range
        """
        for event in self.iter_events():
            if start_ms <= event.offset_ms <= end_ms:
                yield event
            elif event.offset_ms > end_ms:
                break  # Events are chronological, no need to continue

    def get_event_by_id(self, event_id: str) -> Optional[ReplayEvent]:
        """
        Get a specific event by ID.

        Args:
            event_id: The event ID to find

        Returns:
            The ReplayEvent if found, None otherwise
        """
        for event in self.iter_events():
            if event.event_id == event_id:
                return event
        return None

    def get_event_count(self) -> int:
        """
        Get the total number of events.

        Returns:
            Total event count (from metadata if available, otherwise counted)
        """
        if self.meta and self.meta.event_count > 0:
            return self.meta.event_count
        return sum(1 for _ in self.iter_events())

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the replay.

        Returns:
            Dictionary with statistics including:
            - total_events: Total number of events
            - duration_ms: Total duration in milliseconds
            - event_types: Count by event type
            - agents: Count by agent/source
            - events_per_second: Average events per second
        """
        event_types: Dict[str, int] = {}
        agents: Dict[str, int] = {}
        total_events = 0
        min_offset = float("inf")
        max_offset = 0

        for event in self.iter_events():
            total_events += 1
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
            agents[event.source] = agents.get(event.source, 0) + 1
            min_offset = min(min_offset, event.offset_ms)
            max_offset = max(max_offset, event.offset_ms)

        duration_ms = max_offset - min_offset if total_events > 0 else 0
        events_per_second = (total_events / (duration_ms / 1000)) if duration_ms > 0 else 0

        return {
            "total_events": total_events,
            "duration_ms": duration_ms,
            "event_types": event_types,
            "agents": agents,
            "events_per_second": round(events_per_second, 2),
            "first_offset_ms": int(min_offset) if min_offset != float("inf") else 0,
            "last_offset_ms": max_offset,
        }

    def validate_integrity(self) -> Tuple[bool, List[str]]:
        """
        Validate the integrity of the replay data.

        Checks:
        - Metadata file exists and is valid JSON
        - Events file exists and is valid JSONL
        - Event IDs are unique
        - Events are in chronological order
        - Required fields are present in events

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors: List[str] = []

        # Check metadata
        if not self.meta_path.exists():
            errors.append(f"Metadata file not found: {self.meta_path}")
        elif self._load_error:
            errors.append(f"Metadata error: {self._load_error}")

        # Check events file
        if not self.events_path.exists():
            errors.append(f"Events file not found: {self.events_path}")
            return len(errors) == 0, errors

        # Validate events
        seen_ids: Set[str] = set()
        last_offset = -1
        line_num = 0

        try:
            with open(self.events_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        event = ReplayEvent.from_jsonl(line)

                        # Check for duplicate IDs
                        if event.event_id in seen_ids:
                            errors.append(
                                f"Duplicate event ID at line {line_num}: {event.event_id}"
                            )
                        seen_ids.add(event.event_id)

                        # Check chronological order
                        if event.offset_ms < last_offset:
                            errors.append(
                                f"Events out of order at line {line_num}: "
                                f"offset {event.offset_ms} < previous {last_offset}"
                            )
                        last_offset = event.offset_ms

                        # Check required fields
                        if not event.event_id:
                            errors.append(f"Missing event_id at line {line_num}")
                        if not event.event_type:
                            errors.append(f"Missing event_type at line {line_num}")
                        if not event.source:
                            errors.append(f"Missing source at line {line_num}")

                    except (json.JSONDecodeError, ValueError, TypeError) as e:
                        errors.append(f"Corrupted event at line {line_num}: {e}")

        except Exception as e:
            errors.append(f"Failed to read events file: {type(e).__name__}: {e}")

        # Verify event count matches metadata
        if self.meta and self.meta.event_count > 0:
            actual_count = len(seen_ids)
            if actual_count != self.meta.event_count:
                errors.append(
                    f"Event count mismatch: metadata says {self.meta.event_count}, "
                    f"found {actual_count}"
                )

        return len(errors) == 0, errors

    def compute_checksum(self) -> str:
        """
        Compute a checksum of the replay data for integrity verification.

        Returns:
            SHA-256 hash of the events file content
        """
        if not self.events_path.exists():
            return ""

        hasher = hashlib.sha256()
        try:
            with open(self.events_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Failed to compute checksum: {e}")
            return ""

    def to_bundle(self) -> Dict[str, Any]:
        """
        Export the complete replay as a single dictionary.

        Returns:
            Dictionary with 'meta', 'events', and optionally 'error' keys
        """
        if self._load_error:
            return {"error": self._load_error, "meta": None, "events": []}
        events = [asdict(e) for e in self.iter_events()]
        return {"meta": asdict(self.meta) if self.meta else None, "events": events}

    def __len__(self) -> int:
        """Return the number of events."""
        return self.get_event_count()

    def __iter__(self) -> Iterator[ReplayEvent]:
        """Iterate over all events."""
        return self.iter_events()

    def __repr__(self) -> str:
        status = "valid" if self.is_valid else f"error: {self._load_error}"
        return f"ReplayReader({self.session_dir}, {status})"
