"""
Tests for the ReplayReader module.

Tests loading, filtering, seeking, and validation of replay data.
"""

import json
import tempfile
import time
from pathlib import Path

import pytest

from aragora.replay.reader import ReplayReader
from aragora.replay.schema import ReplayEvent, ReplayMeta


class TestReplayReader:
    """Tests for ReplayReader class."""

    @pytest.fixture
    def temp_session(self):
        """Create a temporary session with test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "test-session"
            session_dir.mkdir()

            # Create metadata
            meta = ReplayMeta(
                debate_id="test-debate",
                topic="Test Topic",
                proposal="Test Proposal",
                agents=[{"id": "claude", "name": "Claude"}, {"id": "gpt", "name": "GPT"}],
                started_at="2024-01-01T00:00:00Z",
                ended_at="2024-01-01T00:05:00Z",
                duration_ms=300000,
                status="completed",
                final_verdict="approve",
                vote_tally={"approve": 2},
                event_count=5,
            )
            with open(session_dir / "meta.json", "w") as f:
                f.write(meta.to_json())

            # Create events
            events = [
                ReplayEvent("e1", time.time(), 0, "system", "system", "Start"),
                ReplayEvent("e2", time.time(), 1000, "turn", "claude", "Proposal"),
                ReplayEvent("e3", time.time(), 2000, "turn", "gpt", "Counter"),
                ReplayEvent(
                    "e4", time.time(), 3000, "vote", "claude", "approve", {"reasoning": "Good"}
                ),
                ReplayEvent(
                    "e5", time.time(), 4000, "vote", "gpt", "approve", {"reasoning": "Agree"}
                ),
            ]
            with open(session_dir / "events.jsonl", "w") as f:
                for event in events:
                    f.write(event.to_jsonl() + "\n")

            yield str(session_dir)

    @pytest.fixture
    def reader(self, temp_session):
        """Create a ReplayReader with test data."""
        return ReplayReader(temp_session)

    def test_init_loads_metadata(self, reader):
        """Test that initialization loads metadata."""
        assert reader.is_valid
        assert reader.meta is not None
        assert reader.meta.debate_id == "test-debate"
        assert reader.meta.topic == "Test Topic"

    def test_is_valid_property(self, reader):
        """Test is_valid property."""
        assert reader.is_valid is True
        assert reader.load_error is None

    def test_iter_events(self, reader):
        """Test iterating over all events."""
        events = list(reader.iter_events())

        assert len(events) == 5
        assert events[0].event_type == "system"
        assert events[1].event_type == "turn"

    def test_filter_by_type(self, reader):
        """Test filtering events by type."""
        turns = list(reader.filter_by_type("turn"))

        assert len(turns) == 2
        assert all(e.event_type == "turn" for e in turns)

    def test_filter_by_types(self, reader):
        """Test filtering by multiple types."""
        events = list(reader.filter_by_types({"turn", "vote"}))

        assert len(events) == 4
        assert all(e.event_type in {"turn", "vote"} for e in events)

    def test_filter_by_agent(self, reader):
        """Test filtering events by agent."""
        claude_events = list(reader.filter_by_agent("claude"))

        assert len(claude_events) == 2
        assert all(e.source == "claude" for e in claude_events)

    def test_filter_by_agents(self, reader):
        """Test filtering by multiple agents."""
        events = list(reader.filter_by_agents({"claude", "gpt"}))

        assert len(events) == 4
        assert all(e.source in {"claude", "gpt"} for e in events)

    def test_filter_with_predicate(self, reader):
        """Test filtering with custom predicate."""
        votes = list(reader.filter(lambda e: e.event_type == "vote" and e.content == "approve"))

        assert len(votes) == 2

    def test_seek_to_offset(self, reader):
        """Test seeking to a specific offset."""
        events = list(reader.seek_to_offset(2000))

        assert len(events) == 3
        assert events[0].offset_ms >= 2000

    def test_seek_to_event(self, reader):
        """Test seeking to a specific event ID."""
        events = list(reader.seek_to_event("e3"))

        assert len(events) == 3
        assert events[0].event_id == "e3"

    def test_get_events_in_range(self, reader):
        """Test getting events within a time range."""
        events = list(reader.get_events_in_range(1000, 3000))

        assert len(events) == 3
        assert all(1000 <= e.offset_ms <= 3000 for e in events)

    def test_get_event_by_id(self, reader):
        """Test getting a specific event by ID."""
        event = reader.get_event_by_id("e2")

        assert event is not None
        assert event.event_id == "e2"
        assert event.source == "claude"

    def test_get_event_by_id_not_found(self, reader):
        """Test getting a non-existent event."""
        event = reader.get_event_by_id("nonexistent")

        assert event is None

    def test_get_event_count(self, reader):
        """Test getting event count."""
        count = reader.get_event_count()

        assert count == 5

    def test_get_stats(self, reader):
        """Test getting replay statistics."""
        stats = reader.get_stats()

        assert stats["total_events"] == 5
        assert stats["duration_ms"] == 4000
        assert "turn" in stats["event_types"]
        assert stats["event_types"]["turn"] == 2
        assert "claude" in stats["agents"]
        assert stats["agents"]["claude"] == 2

    def test_validate_integrity_valid(self, reader):
        """Test integrity validation for valid data."""
        is_valid, errors = reader.validate_integrity()

        assert is_valid is True
        assert len(errors) == 0

    def test_compute_checksum(self, reader):
        """Test computing checksum."""
        checksum = reader.compute_checksum()

        assert checksum != ""
        assert len(checksum) == 64  # SHA-256 hex length

    def test_to_bundle(self, reader):
        """Test exporting as a bundle."""
        bundle = reader.to_bundle()

        assert "meta" in bundle
        assert "events" in bundle
        assert bundle["meta"]["debate_id"] == "test-debate"
        assert len(bundle["events"]) == 5

    def test_len(self, reader):
        """Test __len__ method."""
        assert len(reader) == 5

    def test_iter(self, reader):
        """Test __iter__ method."""
        events = list(reader)

        assert len(events) == 5

    def test_repr(self, reader):
        """Test __repr__ method."""
        repr_str = repr(reader)

        assert "ReplayReader" in repr_str
        assert "valid" in repr_str


class TestReplayReaderInvalid:
    """Tests for ReplayReader with invalid data."""

    def test_missing_metadata(self):
        """Test handling missing metadata file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "missing-meta"
            session_dir.mkdir()

            # Create events but no metadata
            with open(session_dir / "events.jsonl", "w") as f:
                f.write(
                    '{"event_id": "e1", "timestamp": 0, "offset_ms": 0, "event_type": "test", "source": "test", "content": "test", "metadata": {}}\n'
                )

            reader = ReplayReader(str(session_dir))

            assert reader.is_valid is False
            assert "not found" in reader.load_error

    def test_corrupted_metadata(self):
        """Test handling corrupted metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "corrupted-meta"
            session_dir.mkdir()

            with open(session_dir / "meta.json", "w") as f:
                f.write("not valid json")

            reader = ReplayReader(str(session_dir))

            assert reader.is_valid is False
            assert "Corrupted" in reader.load_error or "Invalid" in reader.load_error

    def test_missing_events_file(self):
        """Test handling missing events file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "missing-events"
            session_dir.mkdir()

            meta = ReplayMeta(debate_id="test")
            with open(session_dir / "meta.json", "w") as f:
                f.write(meta.to_json())

            reader = ReplayReader(str(session_dir))

            # Should still be valid (metadata exists)
            assert reader.is_valid is True

            # But events should be empty
            events = list(reader.iter_events())
            assert len(events) == 0

    def test_corrupted_event(self):
        """Test handling corrupted event in events file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "corrupted-event"
            session_dir.mkdir()

            meta = ReplayMeta(debate_id="test", event_count=3)
            with open(session_dir / "meta.json", "w") as f:
                f.write(meta.to_json())

            with open(session_dir / "events.jsonl", "w") as f:
                f.write(
                    '{"event_id": "e1", "timestamp": 0, "offset_ms": 0, "event_type": "test", "source": "test", "content": "test", "metadata": {}}\n'
                )
                f.write("not valid json\n")
                f.write(
                    '{"event_id": "e3", "timestamp": 0, "offset_ms": 2000, "event_type": "test", "source": "test", "content": "test", "metadata": {}}\n'
                )

            reader = ReplayReader(str(session_dir))

            # Should skip corrupted event and continue
            events = list(reader.iter_events())
            assert len(events) == 2

    def test_validate_integrity_duplicate_ids(self):
        """Test integrity validation detects duplicate IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "duplicate-ids"
            session_dir.mkdir()

            meta = ReplayMeta(debate_id="test", event_count=2)
            with open(session_dir / "meta.json", "w") as f:
                f.write(meta.to_json())

            with open(session_dir / "events.jsonl", "w") as f:
                f.write(
                    '{"event_id": "e1", "timestamp": 0, "offset_ms": 0, "event_type": "test", "source": "test", "content": "test", "metadata": {}}\n'
                )
                f.write(
                    '{"event_id": "e1", "timestamp": 0, "offset_ms": 1000, "event_type": "test", "source": "test", "content": "test2", "metadata": {}}\n'
                )

            reader = ReplayReader(str(session_dir))
            is_valid, errors = reader.validate_integrity()

            assert is_valid is False
            assert any("Duplicate" in e for e in errors)

    def test_validate_integrity_out_of_order(self):
        """Test integrity validation detects out-of-order events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "out-of-order"
            session_dir.mkdir()

            meta = ReplayMeta(debate_id="test", event_count=2)
            with open(session_dir / "meta.json", "w") as f:
                f.write(meta.to_json())

            with open(session_dir / "events.jsonl", "w") as f:
                f.write(
                    '{"event_id": "e1", "timestamp": 0, "offset_ms": 2000, "event_type": "test", "source": "test", "content": "test", "metadata": {}}\n'
                )
                f.write(
                    '{"event_id": "e2", "timestamp": 0, "offset_ms": 1000, "event_type": "test", "source": "test", "content": "test2", "metadata": {}}\n'
                )

            reader = ReplayReader(str(session_dir))
            is_valid, errors = reader.validate_integrity()

            assert is_valid is False
            assert any("out of order" in e for e in errors)

    def test_validate_integrity_event_count_mismatch(self):
        """Test integrity validation detects event count mismatch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "count-mismatch"
            session_dir.mkdir()

            meta = ReplayMeta(debate_id="test", event_count=5)  # Claims 5 events
            with open(session_dir / "meta.json", "w") as f:
                f.write(meta.to_json())

            # But only write 2
            with open(session_dir / "events.jsonl", "w") as f:
                f.write(
                    '{"event_id": "e1", "timestamp": 0, "offset_ms": 0, "event_type": "test", "source": "test", "content": "test", "metadata": {}}\n'
                )
                f.write(
                    '{"event_id": "e2", "timestamp": 0, "offset_ms": 1000, "event_type": "test", "source": "test", "content": "test2", "metadata": {}}\n'
                )

            reader = ReplayReader(str(session_dir))
            is_valid, errors = reader.validate_integrity()

            assert is_valid is False
            assert any("count mismatch" in e for e in errors)

    def test_to_bundle_with_error(self):
        """Test to_bundle when load failed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "error-session"
            session_dir.mkdir()
            # No files created

            reader = ReplayReader(str(session_dir))
            bundle = reader.to_bundle()

            assert "error" in bundle
            assert bundle["meta"] is None
            assert bundle["events"] == []
