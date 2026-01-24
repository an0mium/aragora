"""
Tests for the ReplayRecorder module.

Tests non-blocking debate event recording.
"""

import tempfile
import time
from pathlib import Path

import pytest

from aragora.replay.recorder import ReplayRecorder
from aragora.replay.schema import ReplayEvent, ReplayMeta


class TestReplayRecorder:
    """Tests for ReplayRecorder class."""

    @pytest.fixture
    def temp_storage(self):
        """Create a temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def recorder(self, temp_storage):
        """Create a ReplayRecorder with temp storage."""
        return ReplayRecorder(
            debate_id="test-debate-123",
            topic="Test Topic",
            proposal="Test Proposal",
            agents=[
                {"id": "claude", "name": "Claude"},
                {"id": "gpt", "name": "GPT"},
            ],
            storage_dir=temp_storage,
        )

    def test_init_creates_session_dir(self, recorder, temp_storage):
        """Test that initialization creates session directory."""
        session_dir = Path(temp_storage) / "test-debate-123"
        assert session_dir.exists()
        assert session_dir.is_dir()

    def test_init_sets_metadata(self, recorder):
        """Test that initialization sets metadata correctly."""
        assert recorder.debate_id == "test-debate-123"
        assert recorder.meta.topic == "Test Topic"
        assert recorder.meta.proposal == "Test Proposal"
        assert len(recorder.meta.agents) == 2

    def test_start_activates_recorder(self, recorder):
        """Test that start() activates the recorder."""
        recorder.start()
        try:
            assert recorder._is_active is True
            assert recorder._start_time is not None
            assert recorder._writer_thread is not None
            assert recorder._writer_thread.is_alive()
        finally:
            recorder.abort()

    def test_start_writes_meta(self, recorder):
        """Test that start() writes metadata file."""
        recorder.start()
        try:
            assert recorder.meta_path.exists()
            with open(recorder.meta_path) as f:
                meta = ReplayMeta.from_json(f.read())
            assert meta.debate_id == "test-debate-123"
        finally:
            recorder.abort()

    def test_record_turn(self, recorder):
        """Test recording a turn event."""
        recorder.start()
        try:
            recorder.record_turn(
                agent_id="claude",
                content="My proposal is...",
                round_num=1,
            )
            time.sleep(0.2)  # Allow writer thread to flush
        finally:
            path = recorder.finalize("approve", {"approve": 2})

        # Verify event was written
        events_path = Path(path) / "events.jsonl"
        assert events_path.exists()

        with open(events_path) as f:
            lines = f.readlines()
        assert len(lines) >= 1

        event = ReplayEvent.from_jsonl(lines[0])
        assert event.event_type == "turn"
        assert event.source == "claude"
        assert event.content == "My proposal is..."
        assert event.metadata["round"] == 1

    def test_record_vote(self, recorder):
        """Test recording a vote event."""
        recorder.start()
        try:
            recorder.record_vote(
                agent_id="gpt",
                vote="approve",
                reasoning="Good proposal",
            )
            time.sleep(0.2)
        finally:
            path = recorder.finalize("approve", {"approve": 1})

        events_path = Path(path) / "events.jsonl"
        with open(events_path) as f:
            event = ReplayEvent.from_jsonl(f.readline())

        assert event.event_type == "vote"
        assert event.source == "gpt"
        assert event.content == "approve"
        assert event.metadata["reasoning"] == "Good proposal"

    def test_record_audience_input(self, recorder):
        """Test recording audience input event."""
        recorder.start()
        try:
            recorder.record_audience_input(
                user_id="user-123",
                message="What about edge cases?",
                loop_id="loop-1",
            )
            time.sleep(0.2)
        finally:
            path = recorder.finalize("approve", {})

        events_path = Path(path) / "events.jsonl"
        with open(events_path) as f:
            event = ReplayEvent.from_jsonl(f.readline())

        assert event.event_type == "audience_input"
        assert event.source == "user-123"
        assert event.metadata["user_id"] == "user-123"
        assert event.metadata["loop_id"] == "loop-1"

    def test_record_phase_change(self, recorder):
        """Test recording phase change event."""
        recorder.start()
        try:
            recorder.record_phase_change("voting")
            time.sleep(0.2)
        finally:
            path = recorder.finalize("approve", {})

        events_path = Path(path) / "events.jsonl"
        with open(events_path) as f:
            event = ReplayEvent.from_jsonl(f.readline())

        assert event.event_type == "phase_change"
        assert event.source == "system"
        assert event.content == "voting"

    def test_record_system(self, recorder):
        """Test recording system event."""
        recorder.start()
        try:
            recorder.record_system("Debate started")
            time.sleep(0.2)
        finally:
            path = recorder.finalize("approve", {})

        events_path = Path(path) / "events.jsonl"
        with open(events_path) as f:
            event = ReplayEvent.from_jsonl(f.readline())

        assert event.event_type == "system"
        assert event.source == "system"
        assert event.content == "Debate started"

    def test_finalize_updates_meta(self, recorder):
        """Test that finalize() updates metadata."""
        recorder.start()
        recorder.record_turn("claude", "Hello", 1)
        time.sleep(0.2)
        path = recorder.finalize("approve", {"approve": 2, "reject": 0})

        meta_path = Path(path) / "meta.json"
        with open(meta_path) as f:
            meta = ReplayMeta.from_json(f.read())

        assert meta.status == "completed"
        assert meta.final_verdict == "approve"
        assert meta.vote_tally == {"approve": 2, "reject": 0}
        assert meta.ended_at is not None
        assert meta.duration_ms is not None

    def test_abort_sets_crashed_status(self, recorder):
        """Test that abort() sets status to crashed."""
        recorder.start()
        recorder.record_turn("claude", "Hello", 1)
        recorder.abort()

        with open(recorder.meta_path) as f:
            meta = ReplayMeta.from_json(f.read())

        assert meta.status == "crashed"

    def test_not_active_after_finalize(self, recorder):
        """Test that recorder is not active after finalize."""
        recorder.start()
        recorder.finalize("approve", {})

        assert recorder._is_active is False

    def test_record_when_not_active(self, recorder):
        """Test that recording when not active is a no-op."""
        # Don't call start()
        recorder.record_turn("claude", "Hello", 1)

        # Should not create events file
        assert not recorder.events_path.exists()

    def test_elapsed_ms(self, recorder):
        """Test elapsed milliseconds calculation."""
        recorder.start()
        time.sleep(0.1)
        elapsed = recorder._elapsed_ms()
        recorder.abort()

        assert elapsed >= 100
        assert elapsed < 200

    def test_multiple_events(self, recorder):
        """Test recording multiple events."""
        recorder.start()
        try:
            recorder.record_system("Start")
            recorder.record_phase_change("proposal")
            recorder.record_turn("claude", "Proposal 1", 1)
            recorder.record_turn("gpt", "Counter", 1)
            recorder.record_phase_change("voting")
            recorder.record_vote("claude", "approve", "Good")
            recorder.record_vote("gpt", "approve", "Agree")
            time.sleep(0.3)
        finally:
            path = recorder.finalize("approve", {"approve": 2})

        events_path = Path(path) / "events.jsonl"
        with open(events_path) as f:
            lines = f.readlines()

        assert len(lines) >= 7

    def test_event_offset_increases(self, recorder):
        """Test that event offsets increase over time."""
        recorder.start()
        try:
            recorder.record_turn("claude", "First", 1)
            time.sleep(0.1)
            recorder.record_turn("gpt", "Second", 1)
            time.sleep(0.2)
        finally:
            path = recorder.finalize("approve", {})

        events_path = Path(path) / "events.jsonl"
        with open(events_path) as f:
            lines = f.readlines()

        event1 = ReplayEvent.from_jsonl(lines[0])
        event2 = ReplayEvent.from_jsonl(lines[1])

        assert event2.offset_ms > event1.offset_ms


class TestReplayRecorderEdgeCases:
    """Edge case tests for ReplayRecorder."""

    @pytest.fixture
    def temp_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_empty_debate(self, temp_storage):
        """Test finalizing a debate with no events."""
        recorder = ReplayRecorder(
            debate_id="empty-debate",
            topic="Empty",
            proposal="Nothing",
            agents=[],
            storage_dir=temp_storage,
        )
        recorder.start()
        path = recorder.finalize("none", {})

        # Should still create valid files
        assert Path(path).exists()
        assert (Path(path) / "meta.json").exists()

    def test_long_content(self, temp_storage):
        """Test recording events with very long content."""
        recorder = ReplayRecorder(
            debate_id="long-content",
            topic="Long",
            proposal="Test",
            agents=[{"id": "a", "name": "A"}],
            storage_dir=temp_storage,
        )
        recorder.start()
        try:
            long_content = "A" * 10000
            recorder.record_turn("a", long_content, 1)
            time.sleep(0.2)
        finally:
            path = recorder.finalize("approve", {})

        events_path = Path(path) / "events.jsonl"
        with open(events_path) as f:
            event = ReplayEvent.from_jsonl(f.readline())

        assert len(event.content) == 10000

    def test_unicode_content(self, temp_storage):
        """Test recording events with unicode content."""
        recorder = ReplayRecorder(
            debate_id="unicode-debate",
            topic="Unicode \u4e2d\u6587",
            proposal="Test \U0001f680",
            agents=[{"id": "agent", "name": "\u4ee3\u7406"}],
            storage_dir=temp_storage,
        )
        recorder.start()
        try:
            recorder.record_turn("agent", "Message with \u00e9\u00e0\u00fc", 1)
            time.sleep(0.2)
        finally:
            path = recorder.finalize("approve", {})

        events_path = Path(path) / "events.jsonl"
        with open(events_path) as f:
            event = ReplayEvent.from_jsonl(f.readline())

        assert "\u00e9" in event.content
