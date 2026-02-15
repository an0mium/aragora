"""Tests for aragora.nomic.event_bus."""

from __future__ import annotations

import json
import time

import pytest

from aragora.nomic.event_bus import (
    VALID_EVENT_TYPES,
    EventBus,
    WorktreeEvent,
)


@pytest.fixture
def event_dir(tmp_path):
    """Create a temporary event directory."""
    return tmp_path / ".aragora_events"


@pytest.fixture
def bus(tmp_path, event_dir):
    """Create an EventBus with a temporary directory."""
    return EventBus(events_dir=event_dir, repo_root=tmp_path)


class TestWorktreeEvent:
    """Tests for WorktreeEvent dataclass."""

    def test_auto_timestamp(self):
        event = WorktreeEvent(event_type="task_claimed", track="core")
        assert event.timestamp
        assert "T" in event.timestamp  # ISO format

    def test_auto_event_id(self):
        event = WorktreeEvent(event_type="task_claimed", track="core")
        assert event.event_id
        assert "core" in event.event_id
        assert "task_claimed" in event.event_id

    def test_custom_data(self):
        event = WorktreeEvent(
            event_type="task_claimed",
            track="core",
            data={"files": ["orchestrator.py"]},
        )
        assert event.data["files"] == ["orchestrator.py"]


class TestEventBusPublish:
    """Tests for publishing events."""

    def test_publish_creates_file(self, bus, event_dir):
        bus.publish("task_claimed", "core", data={"files": ["a.py"]})
        assert event_dir.exists()
        files = list(event_dir.glob("*.json"))
        assert len(files) == 1

    def test_publish_returns_event(self, bus):
        event = bus.publish("task_completed", "qa")
        assert isinstance(event, WorktreeEvent)
        assert event.event_type == "task_completed"
        assert event.track == "qa"

    def test_publish_invalid_type_raises(self, bus):
        with pytest.raises(ValueError, match="Invalid event type"):
            bus.publish("invalid_type", "core")

    def test_publish_all_valid_types(self, bus):
        for event_type in VALID_EVENT_TYPES:
            event = bus.publish(event_type, "core")
            assert event.event_type == event_type

    def test_publish_with_data(self, bus, event_dir):
        bus.publish("task_claimed", "core", data={"files": ["a.py", "b.py"]})
        files = list(event_dir.glob("*.json"))
        with open(files[0]) as f:
            data = json.load(f)
        assert data["data"]["files"] == ["a.py", "b.py"]


class TestEventBusPoll:
    """Tests for polling events."""

    def test_poll_empty(self, bus):
        events = bus.poll()
        assert events == []

    def test_poll_returns_published_events(self, bus):
        bus.publish("task_claimed", "core")
        bus.publish("task_completed", "qa")
        events = bus.poll(since_minutes=5)
        assert len(events) == 2

    def test_poll_filter_by_type(self, bus):
        bus.publish("task_claimed", "core")
        bus.publish("task_completed", "qa")
        events = bus.poll(event_type="task_claimed")
        assert len(events) == 1
        assert events[0].event_type == "task_claimed"

    def test_poll_exclude_track(self, bus):
        bus.publish("task_claimed", "core")
        bus.publish("task_claimed", "qa")
        events = bus.poll(exclude_track="core")
        assert len(events) == 1
        assert events[0].track == "qa"

    def test_poll_ordering(self, bus):
        bus.publish("error", "a")
        time.sleep(0.01)  # Ensure different timestamps
        bus.publish("error", "b")
        events = bus.poll()
        assert events[0].track == "a"
        assert events[1].track == "b"


class TestEventBusClaimed:
    """Tests for claimed files tracking."""

    def test_get_claimed_files_empty(self, bus):
        claims = bus.get_claimed_files()
        assert claims == {}

    def test_get_claimed_files(self, bus):
        bus.publish("task_claimed", "core", data={"files": ["a.py", "b.py"]})
        bus.publish("task_claimed", "qa", data={"files": ["tests/c.py"]})
        claims = bus.get_claimed_files()
        assert "core" in claims
        assert "qa" in claims
        assert claims["core"] == ["a.py", "b.py"]


class TestEventBusCompleted:
    """Tests for completed track tracking."""

    def test_get_completed_empty(self, bus):
        assert bus.get_completed_tracks() == []

    def test_get_completed_tracks(self, bus):
        bus.publish("task_completed", "core")
        bus.publish("task_completed", "qa")
        completed = bus.get_completed_tracks()
        assert set(completed) == {"core", "qa"}


class TestEventBusCleanup:
    """Tests for event cleanup."""

    def test_cleanup_no_events(self, bus):
        assert bus.cleanup() == 0

    def test_cleanup_old_events(self, bus, event_dir):
        # Create an old event file manually
        event_dir.mkdir(parents=True, exist_ok=True)
        old_file = event_dir / "old_event.json"
        old_file.write_text('{"event_type": "old"}')
        # Set modification time to 48 hours ago
        import os
        old_time = time.time() - (48 * 3600)
        os.utime(old_file, (old_time, old_time))

        cleaned = bus.cleanup(max_age_hours=24)
        assert cleaned == 1
        assert not old_file.exists()

    def test_clear_all(self, bus):
        bus.publish("task_claimed", "a")
        bus.publish("task_claimed", "b")
        count = bus.clear_all()
        assert count == 2
        assert bus.poll() == []


class TestEventBusMergeReady:
    """Tests for merge-ready tracking."""

    def test_get_merge_ready_empty(self, bus):
        assert bus.get_merge_ready() == []

    def test_get_merge_ready(self, bus):
        bus.publish("merge_ready", "core", data={"branch": "dev/core-track"})
        ready = bus.get_merge_ready()
        assert len(ready) == 1
        assert ready[0].track == "core"
