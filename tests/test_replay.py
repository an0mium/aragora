"""
Tests for debate replay functionality.

This module tests two replay systems:
1. DebateRecorder/DebateReplayer (replay.py) - file-based replay of completed debates
2. Stream-based replay (reader.py, recorder.py, schema.py, storage.py) - live event recording
"""

import json
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from aragora.core import DebateResult, Environment, Message, Vote
from aragora.replay.replay import DebateRecorder, DebateReplayer
from aragora.replay.schema import ReplayEvent, ReplayMeta, SCHEMA_VERSION
from aragora.replay.reader import ReplayReader
from aragora.replay.recorder import ReplayRecorder
from aragora.replay.storage import ReplayStorage


class TestDebateRecorder:
    """Test the DebateRecorder class."""

    def test_save_debate(self):
        """Test saving a debate result."""
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = DebateRecorder(temp_dir)

            # Create a mock debate result
            result = DebateResult(
                id="test-123",
                task="Test debate task",
                final_answer="Test answer",
                confidence=0.8,
                consensus_reached=True,
                messages=[
                    Message(role="proposer", agent="agent1", content="Proposal 1"),
                    Message(role="critic", agent="agent2", content="Critique of proposal 1"),
                ],
                votes=[
                    Vote(agent="agent1", choice="accept", confidence=0.9, reasoning="Good proposal")
                ],
                rounds_used=2,
                duration_seconds=10.5,
            )

            # Save the debate
            filepath = recorder.save_debate(result, {"test": "metadata"})

            # Verify file was created
            assert Path(filepath).exists()

            # Verify content
            with open(filepath, "r") as f:
                data = json.load(f)

            assert data["debate_result"]["id"] == "test-123"
            assert data["debate_result"]["task"] == "Test debate task"
            assert data["metadata"]["test"] == "metadata"
            assert "recorded_at" in data

    def test_make_serializable(self):
        """Test converting objects to serializable format."""
        recorder = DebateRecorder()

        # Test with dict containing dataclass
        test_dict = {"result": DebateResult(id="test")}
        result = recorder._make_serializable(test_dict)

        assert isinstance(result, dict)
        assert result["result"]["id"] == "test"

        # Test with list
        test_list = [1, "string", {"key": "value"}]
        result = recorder._make_serializable(test_list)
        assert result == test_list


class TestDebateReplayer:
    """Test the DebateReplayer class."""

    def test_list_debates_empty(self):
        """Test listing debates when none exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            replayer = DebateReplayer(temp_dir)
            debates = replayer.list_debates()
            assert debates == []

    def test_list_debates_with_data(self):
        """Test listing debates with saved data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # First save a debate
            recorder = DebateRecorder(temp_dir)
            result = DebateResult(
                id="test-123",
                task="Test debate about AI safety",
                consensus_reached=True,
                confidence=0.9,
                duration_seconds=15.0,
                rounds_used=3,
            )
            recorder.save_debate(result)

            # Now list debates
            replayer = DebateReplayer(temp_dir)
            debates = replayer.list_debates()

            assert len(debates) == 1
            debate = debates[0]
            assert debate["task"] == "Test debate about AI safety"
            assert debate["consensus_reached"] is True
            assert debate["confidence"] == 0.9
            assert debate["duration_seconds"] == 15.0
            assert debate["rounds_used"] == 3

    def test_load_debate(self):
        """Test loading a specific debate."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save a debate
            recorder = DebateRecorder(temp_dir)
            original_result = DebateResult(
                id="test-456",
                task="Test loading functionality",
                final_answer="Loaded successfully",
            )
            filepath = recorder.save_debate(original_result)
            filename = Path(filepath).name

            # Load the debate
            replayer = DebateReplayer(temp_dir)
            loaded_result = replayer.load_debate(filename)

            assert loaded_result is not None
            assert loaded_result.id == "test-456"
            assert loaded_result.task == "Test loading functionality"
            assert loaded_result.final_answer == "Loaded successfully"

    def test_load_debate_not_found(self):
        """Test loading a non-existent debate."""
        with tempfile.TemporaryDirectory() as temp_dir:
            replayer = DebateReplayer(temp_dir)
            result = replayer.load_debate("nonexistent.json")
            assert result is None

    def test_replay_debate(self, capsys):
        """Test replaying a debate (captures output)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save a debate with messages
            recorder = DebateRecorder(temp_dir)
            result = DebateResult(
                id="replay-test",
                task="Test replay functionality",
                final_answer="Replay complete",
                messages=[
                    Message(role="proposer", agent="Agent1", content="First message"),
                    Message(role="critic", agent="Agent2", content="Second message"),
                ],
                consensus_reached=True,
                confidence=1.0,
                duration_seconds=5.0,
                rounds_used=1,
            )
            filepath = recorder.save_debate(result)
            filename = Path(filepath).name

            # Replay the debate
            replayer = DebateReplayer(temp_dir)
            replayed_result = replayer.replay_debate(filename, speed=10.0)  # Fast replay

            assert replayed_result is not None
            assert replayed_result.id == "replay-test"

            # Check output
            captured = capsys.readouterr()
            assert "REPLAYING DEBATE" in captured.out
            assert "Test replay functionality" in captured.out
            assert "[ 1] Agent1: First message" in captured.out
            assert "[ 2] Agent2: Second message" in captured.out
            assert "Final Answer: Replay complete" in captured.out


# =============================================================================
# Stream-based Replay System Tests (reader.py, recorder.py, schema.py, storage.py)
# =============================================================================


class TestReplayEvent:
    """Tests for ReplayEvent dataclass."""

    def test_create_event(self):
        """Event is created with all fields."""
        event = ReplayEvent(
            event_id="abc123",
            timestamp=1704067200.0,
            offset_ms=5000,
            event_type="turn",
            source="agent-1",
            content="Hello world",
            metadata={"round": 1},
        )
        assert event.event_id == "abc123"
        assert event.timestamp == 1704067200.0
        assert event.offset_ms == 5000
        assert event.event_type == "turn"
        assert event.source == "agent-1"
        assert event.content == "Hello world"
        assert event.metadata == {"round": 1}

    def test_default_metadata(self):
        """Metadata defaults to empty dict."""
        event = ReplayEvent(
            event_id="abc123",
            timestamp=1704067200.0,
            offset_ms=0,
            event_type="turn",
            source="agent-1",
            content="test",
        )
        assert event.metadata == {}

    def test_to_jsonl(self):
        """Event serializes to valid JSON line."""
        event = ReplayEvent(
            event_id="abc123",
            timestamp=1704067200.0,
            offset_ms=5000,
            event_type="turn",
            source="agent-1",
            content="Hello world",
            metadata={"round": 1},
        )
        jsonl = event.to_jsonl()
        parsed = json.loads(jsonl)
        assert parsed["event_id"] == "abc123"
        assert parsed["timestamp"] == 1704067200.0
        assert parsed["offset_ms"] == 5000
        assert parsed["event_type"] == "turn"
        assert parsed["source"] == "agent-1"
        assert parsed["content"] == "Hello world"
        assert parsed["metadata"] == {"round": 1}

    def test_to_jsonl_unicode(self):
        """Event correctly handles unicode content."""
        event = ReplayEvent(
            event_id="abc123",
            timestamp=1704067200.0,
            offset_ms=0,
            event_type="turn",
            source="agent-1",
            content="Unicode: \u4e2d\u6587 \U0001f680",
        )
        jsonl = event.to_jsonl()
        assert "\u4e2d\u6587" in jsonl

    def test_from_jsonl(self):
        """Event deserializes from JSON line."""
        jsonl = '{"event_id": "abc123", "timestamp": 1704067200.0, "offset_ms": 5000, "event_type": "turn", "source": "agent-1", "content": "Hello", "metadata": {}}'
        event = ReplayEvent.from_jsonl(jsonl)
        assert event.event_id == "abc123"
        assert event.timestamp == 1704067200.0
        assert event.event_type == "turn"
        assert event.source == "agent-1"
        assert event.content == "Hello"

    def test_from_jsonl_with_newline(self):
        """Event deserializes with trailing newline."""
        jsonl = '{"event_id": "abc123", "timestamp": 1704067200.0, "offset_ms": 0, "event_type": "turn", "source": "agent-1", "content": "Hello", "metadata": {}}\n'
        event = ReplayEvent.from_jsonl(jsonl)
        assert event.event_id == "abc123"

    def test_from_jsonl_invalid_json(self):
        """Invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            ReplayEvent.from_jsonl("not valid json")

    def test_roundtrip(self):
        """Event survives roundtrip serialization."""
        original = ReplayEvent(
            event_id="test123",
            timestamp=1704067200.0,
            offset_ms=12345,
            event_type="vote",
            source="agent-2",
            content="approve",
            metadata={"reasoning": "Good argument"},
        )
        jsonl = original.to_jsonl()
        restored = ReplayEvent.from_jsonl(jsonl)
        assert original == restored


class TestReplayMeta:
    """Tests for ReplayMeta dataclass."""

    def test_create_meta_defaults(self):
        """Meta is created with default values."""
        meta = ReplayMeta()
        assert meta.schema_version == SCHEMA_VERSION
        assert meta.debate_id == ""
        assert meta.topic == ""
        assert meta.proposal == ""
        assert meta.agents == []
        assert meta.started_at == ""
        assert meta.ended_at is None
        assert meta.duration_ms is None
        assert meta.status == "in_progress"
        assert meta.final_verdict is None
        assert meta.vote_tally == {}
        assert meta.event_count == 0
        assert meta.tags == []

    def test_create_meta_with_values(self):
        """Meta is created with specified values."""
        meta = ReplayMeta(
            debate_id="debate-123",
            topic="AI Safety",
            proposal="AI systems should be regulated",
            agents=[{"id": "agent-1", "name": "Claude"}],
            started_at="2024-01-01T00:00:00",
            status="completed",
        )
        assert meta.debate_id == "debate-123"
        assert meta.topic == "AI Safety"
        assert meta.proposal == "AI systems should be regulated"
        assert meta.agents == [{"id": "agent-1", "name": "Claude"}]

    def test_to_json(self):
        """Meta serializes to valid JSON."""
        meta = ReplayMeta(debate_id="debate-123", topic="Test topic", status="completed")
        json_str = meta.to_json()
        parsed = json.loads(json_str)
        assert parsed["debate_id"] == "debate-123"
        assert parsed["topic"] == "Test topic"
        assert parsed["status"] == "completed"
        assert parsed["schema_version"] == SCHEMA_VERSION

    def test_from_json(self):
        """Meta deserializes from JSON."""
        json_str = '{"schema_version": "1.0", "debate_id": "debate-123", "topic": "Test", "proposal": "", "agents": [], "started_at": "", "ended_at": null, "duration_ms": null, "status": "completed", "final_verdict": "approved", "vote_tally": {"approve": 2, "reject": 1}, "event_count": 10, "tags": []}'
        meta = ReplayMeta.from_json(json_str)
        assert meta.debate_id == "debate-123"
        assert meta.status == "completed"
        assert meta.final_verdict == "approved"
        assert meta.vote_tally == {"approve": 2, "reject": 1}
        assert meta.event_count == 10

    def test_from_json_invalid(self):
        """Invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            ReplayMeta.from_json("{invalid json}")

    def test_roundtrip(self):
        """Meta survives roundtrip serialization."""
        original = ReplayMeta(
            debate_id="debate-456",
            topic="Climate Change",
            proposal="Carbon tax proposal",
            agents=[{"id": "a1"}, {"id": "a2"}],
            started_at="2024-01-01T12:00:00",
            ended_at="2024-01-01T13:00:00",
            duration_ms=3600000,
            status="completed",
            final_verdict="approved",
            vote_tally={"approve": 3, "reject": 1},
            event_count=50,
            tags=["climate", "policy"],
        )
        json_str = original.to_json()
        restored = ReplayMeta.from_json(json_str)
        assert original == restored


class TestReplayReader:
    """Tests for ReplayReader class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)

    @pytest.fixture
    def sample_replay(self, temp_dir):
        """Create sample replay in temp directory."""
        session_dir = Path(temp_dir) / "test-debate"
        session_dir.mkdir()

        meta = ReplayMeta(
            debate_id="test-debate",
            topic="Sample topic",
            proposal="Test proposal",
            agents=[{"id": "agent-1"}],
            started_at="2024-01-01T00:00:00",
            status="completed",
            event_count=3,
        )
        (session_dir / "meta.json").write_text(meta.to_json())

        events = [
            ReplayEvent("e1", 1704067200.0, 0, "turn", "agent-1", "First message"),
            ReplayEvent("e2", 1704067201.0, 1000, "turn", "agent-2", "Second message"),
            ReplayEvent(
                "e3", 1704067202.0, 2000, "vote", "agent-1", "approve", {"reasoning": "Good"}
            ),
        ]
        events_file = session_dir / "events.jsonl"
        events_file.write_text("\n".join(e.to_jsonl() for e in events) + "\n")

        return str(session_dir)

    def test_read_valid_replay(self, sample_replay):
        """Reader loads valid replay successfully."""
        reader = ReplayReader(sample_replay)
        assert reader.meta is not None
        assert reader.meta.debate_id == "test-debate"
        assert reader.meta.topic == "Sample topic"
        assert reader._load_error is None

    def test_read_events(self, sample_replay):
        """Reader iterates through events."""
        reader = ReplayReader(sample_replay)
        events = list(reader.iter_events())
        assert len(events) == 3
        assert events[0].event_id == "e1"
        assert events[1].event_id == "e2"
        assert events[2].event_id == "e3"
        assert events[2].event_type == "vote"

    def test_to_bundle(self, sample_replay):
        """Reader creates valid bundle."""
        reader = ReplayReader(sample_replay)
        bundle = reader.to_bundle()
        assert bundle["meta"] is not None
        assert bundle["meta"]["debate_id"] == "test-debate"
        assert len(bundle["events"]) == 3
        assert "error" not in bundle or bundle["error"] is None

    def test_missing_meta_file(self, temp_dir):
        """Reader handles missing meta file gracefully."""
        session_dir = Path(temp_dir) / "missing-meta"
        session_dir.mkdir()
        # No meta.json created

        reader = ReplayReader(str(session_dir))
        assert reader.meta is None
        assert reader._load_error is not None
        assert "not found" in reader._load_error

    def test_corrupted_meta_file(self, temp_dir):
        """Reader handles corrupted meta file gracefully."""
        session_dir = Path(temp_dir) / "corrupted-meta"
        session_dir.mkdir()
        (session_dir / "meta.json").write_text("{invalid json}")

        reader = ReplayReader(str(session_dir))
        assert reader.meta is None
        assert reader._load_error is not None

    def test_missing_events_file(self, temp_dir):
        """Reader handles missing events file gracefully."""
        session_dir = Path(temp_dir) / "no-events"
        session_dir.mkdir()
        meta = ReplayMeta(debate_id="no-events", topic="Test")
        (session_dir / "meta.json").write_text(meta.to_json())
        # No events.jsonl created

        reader = ReplayReader(str(session_dir))
        assert reader.meta is not None
        events = list(reader.iter_events())
        assert events == []

    def test_corrupted_event_line(self, temp_dir):
        """Reader stops at corrupted event lines (ValueError from from_jsonl)."""
        session_dir = Path(temp_dir) / "corrupted-event"
        session_dir.mkdir()
        meta = ReplayMeta(debate_id="corrupted-event", topic="Test")
        (session_dir / "meta.json").write_text(meta.to_json())

        event1 = ReplayEvent("e1", 1704067200.0, 0, "turn", "agent-1", "Valid")
        events_content = event1.to_jsonl() + "\n{invalid json}\n"
        event2 = ReplayEvent("e2", 1704067201.0, 1000, "turn", "agent-2", "Also valid")
        events_content += event2.to_jsonl() + "\n"
        (session_dir / "events.jsonl").write_text(events_content)

        reader = ReplayReader(str(session_dir))
        events = list(reader.iter_events())
        # Reader stops at corrupted line (ValueError caught by outer handler)
        assert len(events) == 1
        assert events[0].event_id == "e1"

    def test_to_bundle_with_error(self, temp_dir):
        """Bundle includes error when load failed."""
        session_dir = Path(temp_dir) / "error-bundle"
        session_dir.mkdir()
        # No meta.json

        reader = ReplayReader(str(session_dir))
        bundle = reader.to_bundle()
        assert bundle["error"] is not None
        assert bundle["meta"] is None
        assert bundle["events"] == []


class TestStreamReplayRecorder:
    """Tests for stream-based ReplayRecorder class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_init_creates_directory(self, temp_dir):
        """Recorder creates session directory on init."""
        storage_dir = Path(temp_dir) / "replays"
        recorder = ReplayRecorder(
            debate_id="test-123",
            topic="Test topic",
            proposal="Test proposal",
            agents=[{"id": "a1"}],
            storage_dir=str(storage_dir),
        )
        assert recorder.session_dir.exists()
        assert recorder.debate_id == "test-123"

    def test_start_writes_meta(self, temp_dir):
        """Start writes initial metadata."""
        storage_dir = Path(temp_dir) / "replays"
        recorder = ReplayRecorder(
            debate_id="test-123",
            topic="Test topic",
            proposal="Test proposal",
            agents=[{"id": "a1", "name": "Agent 1"}],
            storage_dir=str(storage_dir),
        )
        recorder.start()
        try:
            assert recorder.meta_path.exists()
            meta_content = json.loads(recorder.meta_path.read_text())
            assert meta_content["debate_id"] == "test-123"
            assert meta_content["topic"] == "Test topic"
            assert meta_content["status"] == "in_progress"
        finally:
            recorder.abort()

    def test_record_turn(self, temp_dir):
        """Records turn events."""
        storage_dir = Path(temp_dir) / "replays"
        recorder = ReplayRecorder(
            debate_id="test-456",
            topic="Turn test",
            proposal="Test",
            agents=[],
            storage_dir=str(storage_dir),
        )
        recorder.start()
        recorder.record_turn("agent-1", "Hello world", round_num=1)
        recorder.record_turn("agent-2", "Reply here", round_num=1)
        time.sleep(0.3)  # Let writer thread process
        recorder.finalize("approved", {"approve": 2})

        reader = ReplayReader(str(recorder.session_dir))
        events = list(reader.iter_events())
        assert len(events) == 2
        assert events[0].event_type == "turn"
        assert events[0].source == "agent-1"
        assert events[0].content == "Hello world"
        assert events[0].metadata["round"] == 1

    def test_record_vote(self, temp_dir):
        """Records vote events."""
        storage_dir = Path(temp_dir) / "replays"
        recorder = ReplayRecorder(
            debate_id="test-vote",
            topic="Vote test",
            proposal="Test",
            agents=[],
            storage_dir=str(storage_dir),
        )
        recorder.start()
        recorder.record_vote("agent-1", "approve", "Convincing argument")
        time.sleep(0.3)
        recorder.finalize("approved", {"approve": 1})

        reader = ReplayReader(str(recorder.session_dir))
        events = list(reader.iter_events())
        assert len(events) == 1
        assert events[0].event_type == "vote"
        assert events[0].content == "approve"
        assert events[0].metadata["reasoning"] == "Convincing argument"

    def test_record_audience_input(self, temp_dir):
        """Records audience input events."""
        storage_dir = Path(temp_dir) / "replays"
        recorder = ReplayRecorder(
            debate_id="test-audience",
            topic="Audience test",
            proposal="Test",
            agents=[],
            storage_dir=str(storage_dir),
        )
        recorder.start()
        recorder.record_audience_input("user-123", "Great question!", loop_id="loop-1")
        time.sleep(0.3)
        recorder.finalize("approved", {})

        reader = ReplayReader(str(recorder.session_dir))
        events = list(reader.iter_events())
        assert len(events) == 1
        assert events[0].event_type == "audience_input"
        assert events[0].source == "user-123"
        assert events[0].metadata["loop_id"] == "loop-1"

    def test_record_phase_change(self, temp_dir):
        """Records phase change events."""
        storage_dir = Path(temp_dir) / "replays"
        recorder = ReplayRecorder(
            debate_id="test-phase",
            topic="Phase test",
            proposal="Test",
            agents=[],
            storage_dir=str(storage_dir),
        )
        recorder.start()
        recorder.record_phase_change("voting")
        time.sleep(0.3)
        recorder.finalize("approved", {})

        reader = ReplayReader(str(recorder.session_dir))
        events = list(reader.iter_events())
        assert len(events) == 1
        assert events[0].event_type == "phase_change"
        assert events[0].source == "system"
        assert events[0].content == "voting"

    def test_record_system_message(self, temp_dir):
        """Records system messages."""
        storage_dir = Path(temp_dir) / "replays"
        recorder = ReplayRecorder(
            debate_id="test-system",
            topic="System test",
            proposal="Test",
            agents=[],
            storage_dir=str(storage_dir),
        )
        recorder.start()
        recorder.record_system("Debate initialized")
        time.sleep(0.3)
        recorder.finalize("approved", {})

        reader = ReplayReader(str(recorder.session_dir))
        events = list(reader.iter_events())
        assert len(events) == 1
        assert events[0].event_type == "system"
        assert events[0].content == "Debate initialized"

    def test_finalize_updates_meta(self, temp_dir):
        """Finalize updates metadata with final status."""
        storage_dir = Path(temp_dir) / "replays"
        recorder = ReplayRecorder(
            debate_id="test-finalize",
            topic="Finalize test",
            proposal="Test",
            agents=[],
            storage_dir=str(storage_dir),
        )
        recorder.start()
        recorder.record_turn("agent-1", "Test message", round_num=1)
        time.sleep(0.3)
        result_path = recorder.finalize("approved", {"approve": 3, "reject": 1})

        meta = ReplayMeta.from_json(recorder.meta_path.read_text())
        assert meta.status == "completed"
        assert meta.final_verdict == "approved"
        assert meta.vote_tally == {"approve": 3, "reject": 1}
        assert meta.ended_at is not None
        assert meta.duration_ms is not None
        assert result_path == str(recorder.session_dir)

    def test_abort_sets_crashed_status(self, temp_dir):
        """Abort sets status to crashed."""
        storage_dir = Path(temp_dir) / "replays"
        recorder = ReplayRecorder(
            debate_id="test-abort",
            topic="Abort test",
            proposal="Test",
            agents=[],
            storage_dir=str(storage_dir),
        )
        recorder.start()
        recorder.abort()

        meta = ReplayMeta.from_json(recorder.meta_path.read_text())
        assert meta.status == "crashed"

    def test_no_events_when_inactive(self, temp_dir):
        """Events are not recorded when recorder is inactive."""
        storage_dir = Path(temp_dir) / "replays"
        recorder = ReplayRecorder(
            debate_id="test-inactive",
            topic="Inactive test",
            proposal="Test",
            agents=[],
            storage_dir=str(storage_dir),
        )
        # Don't call start()
        recorder.record_turn("agent-1", "Should not be recorded", round_num=1)
        # Events should not be recorded since recorder is not active
        assert recorder._write_queue.empty()


class TestReplayStorage:
    """Tests for ReplayStorage class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)

    @pytest.fixture
    def populated_storage(self, temp_dir):
        """Create storage with sample recordings."""
        storage_dir = Path(temp_dir) / "replays"
        storage_dir.mkdir()

        # Create 5 sample recordings
        for i in range(5):
            session_dir = storage_dir / f"debate-{i}"
            session_dir.mkdir()
            meta = ReplayMeta(
                debate_id=f"debate-{i}",
                topic=f"Topic {i}",
                started_at=f"2024-01-0{i+1}T00:00:00",
                status="completed",
                event_count=i * 10,
            )
            (session_dir / "meta.json").write_text(meta.to_json())

        return str(storage_dir)

    def test_init_creates_directory(self, temp_dir):
        """Storage creates directory on init."""
        storage_dir = Path(temp_dir) / "new_replays"
        storage = ReplayStorage(str(storage_dir))
        assert storage_dir.exists()

    def test_list_recordings_empty(self, temp_dir):
        """Empty storage returns empty list."""
        storage_dir = Path(temp_dir) / "empty_replays"
        storage = ReplayStorage(str(storage_dir))
        recordings = storage.list_recordings()
        assert recordings == []

    def test_list_recordings_populated(self, populated_storage):
        """Populated storage returns recordings."""
        storage = ReplayStorage(populated_storage)
        recordings = storage.list_recordings()
        assert len(recordings) == 5

    def test_list_recordings_sorted_by_date(self, populated_storage):
        """Recordings are sorted by date descending."""
        storage = ReplayStorage(populated_storage)
        recordings = storage.list_recordings()
        # debate-4 has latest date (2024-01-05)
        assert recordings[0]["id"] == "debate-4"
        assert recordings[-1]["id"] == "debate-0"

    def test_list_recordings_limit(self, populated_storage):
        """List respects limit parameter."""
        storage = ReplayStorage(populated_storage)
        recordings = storage.list_recordings(limit=3)
        assert len(recordings) == 3

    def test_list_recordings_includes_fields(self, populated_storage):
        """Recordings include expected fields."""
        storage = ReplayStorage(populated_storage)
        recordings = storage.list_recordings()
        rec = recordings[0]
        assert "id" in rec
        assert "topic" in rec
        assert "status" in rec
        assert "event_count" in rec
        assert "started_at" in rec

    def test_list_recordings_skips_non_dirs(self, populated_storage):
        """List skips non-directory entries."""
        # Add a file to storage directory
        (Path(populated_storage) / "random_file.txt").write_text("test")

        storage = ReplayStorage(populated_storage)
        recordings = storage.list_recordings()
        assert len(recordings) == 5  # Only the 5 directories

    def test_list_recordings_skips_missing_meta(self, temp_dir):
        """List skips directories without meta.json."""
        storage_dir = Path(temp_dir) / "replays"
        storage_dir.mkdir()

        # Directory with meta
        valid_dir = storage_dir / "valid"
        valid_dir.mkdir()
        meta = ReplayMeta(debate_id="valid", topic="Test", started_at="2024-01-01T00:00:00")
        (valid_dir / "meta.json").write_text(meta.to_json())

        # Directory without meta
        invalid_dir = storage_dir / "invalid"
        invalid_dir.mkdir()
        # No meta.json

        storage = ReplayStorage(str(storage_dir))
        recordings = storage.list_recordings()
        assert len(recordings) == 1
        assert recordings[0]["id"] == "valid"

    def test_list_recordings_skips_corrupted_meta(self, temp_dir):
        """List skips directories with corrupted meta.json."""
        storage_dir = Path(temp_dir) / "replays"
        storage_dir.mkdir()

        # Directory with valid meta
        valid_dir = storage_dir / "valid"
        valid_dir.mkdir()
        meta = ReplayMeta(debate_id="valid", topic="Test", started_at="2024-01-01T00:00:00")
        (valid_dir / "meta.json").write_text(meta.to_json())

        # Directory with corrupted meta
        corrupted_dir = storage_dir / "corrupted"
        corrupted_dir.mkdir()
        (corrupted_dir / "meta.json").write_text("{invalid json}")

        storage = ReplayStorage(str(storage_dir))
        recordings = storage.list_recordings()
        assert len(recordings) == 1
        assert recordings[0]["id"] == "valid"

    def test_prune_removes_old_recordings(self, temp_dir):
        """Prune removes oldest recordings."""
        storage_dir = Path(temp_dir) / "replays"
        storage_dir.mkdir()

        # Create 10 recordings
        for i in range(10):
            session_dir = storage_dir / f"debate-{i}"
            session_dir.mkdir()
            meta = ReplayMeta(
                debate_id=f"debate-{i}",
                topic=f"Topic {i}",
                started_at=f"2024-01-{i+1:02d}T00:00:00",
            )
            (session_dir / "meta.json").write_text(meta.to_json())

        storage = ReplayStorage(str(storage_dir))
        removed = storage.prune(keep_last=5)

        assert removed == 5
        recordings = storage.list_recordings()
        assert len(recordings) == 5
        # Should keep debate-5 through debate-9 (most recent)
        ids = [r["id"] for r in recordings]
        assert "debate-9" in ids
        assert "debate-5" in ids
        assert "debate-0" not in ids

    def test_prune_no_op_when_under_limit(self, populated_storage):
        """Prune does nothing when under limit."""
        storage = ReplayStorage(populated_storage)
        removed = storage.prune(keep_last=10)

        assert removed == 0
        recordings = storage.list_recordings()
        assert len(recordings) == 5

    def test_prune_returns_count(self, temp_dir):
        """Prune returns count of removed recordings."""
        storage_dir = Path(temp_dir) / "replays"
        storage_dir.mkdir()

        for i in range(8):
            session_dir = storage_dir / f"debate-{i}"
            session_dir.mkdir()
            meta = ReplayMeta(
                debate_id=f"debate-{i}",
                topic=f"Topic {i}",
                started_at=f"2024-01-{i+1:02d}T00:00:00",
            )
            (session_dir / "meta.json").write_text(meta.to_json())

        storage = ReplayStorage(str(storage_dir))
        removed = storage.prune(keep_last=3)

        assert removed == 5


class TestStreamReplayIntegration:
    """Integration tests for stream-based replay workflow."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_full_workflow(self, temp_dir):
        """Test complete record -> read workflow."""
        storage_dir = Path(temp_dir) / "replays"

        # Record a debate
        recorder = ReplayRecorder(
            debate_id="full-test",
            topic="Climate Change",
            proposal="Implement carbon tax",
            agents=[{"id": "claude", "name": "Claude"}, {"id": "gpt", "name": "GPT-4"}],
            storage_dir=str(storage_dir),
        )
        recorder.start()

        # Simulate debate events
        recorder.record_system("Debate started")
        recorder.record_turn("claude", "I support the carbon tax because...", round_num=1)
        recorder.record_turn("gpt", "I have concerns about...", round_num=1)
        recorder.record_turn("claude", "Addressing your concerns...", round_num=2)
        recorder.record_turn("gpt", "You make a good point...", round_num=2)
        recorder.record_phase_change("voting")
        recorder.record_vote("claude", "approve", "Strong economic case")
        recorder.record_vote("gpt", "approve", "Convinced by arguments")

        time.sleep(0.5)  # Let writer catch up
        session_path = recorder.finalize("approved", {"approve": 2, "reject": 0})

        # Read the replay
        reader = ReplayReader(session_path)
        assert reader.meta is not None
        assert reader.meta.debate_id == "full-test"
        assert reader.meta.topic == "Climate Change"
        assert reader.meta.status == "completed"
        assert reader.meta.final_verdict == "approved"
        assert len(reader.meta.agents) == 2

        events = list(reader.iter_events())
        assert len(events) == 8
        event_types = [e.event_type for e in events]
        assert event_types.count("turn") == 4
        assert event_types.count("vote") == 2
        assert event_types.count("phase_change") == 1
        assert event_types.count("system") == 1

        # Verify bundle
        bundle = reader.to_bundle()
        assert bundle["meta"]["final_verdict"] == "approved"
        assert len(bundle["events"]) == 8

    def test_storage_lists_recorded_debates(self, temp_dir):
        """Storage correctly lists recorded debates."""
        storage_dir = Path(temp_dir) / "replays"

        # Record multiple debates
        for i in range(3):
            recorder = ReplayRecorder(
                debate_id=f"debate-{i}",
                topic=f"Topic {i}",
                proposal=f"Proposal {i}",
                agents=[],
                storage_dir=str(storage_dir),
            )
            recorder.start()
            recorder.record_turn("agent", f"Message for debate {i}", round_num=1)
            time.sleep(0.2)
            recorder.finalize("approved", {})

        storage = ReplayStorage(str(storage_dir))
        recordings = storage.list_recordings()
        assert len(recordings) == 3
        ids = {r["id"] for r in recordings}
        assert ids == {"debate-0", "debate-1", "debate-2"}
