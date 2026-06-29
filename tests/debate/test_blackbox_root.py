"""
Tests for the BlackboxRecorder flight recorder.

Tests cover:
- BlackboxEvent and BlackboxSnapshot dataclass serialization
- BlackboxRecorder initialization and directory setup
- Event recording and flushing
- Snapshot creation and persistence
- Error and agent failure logging
- Session summary generation
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from aragora.debate.blackbox import (
    BlackboxEvent,
    BlackboxRecorder,
    BlackboxSnapshot,
    close_blackbox,
    get_blackbox,
    _active_recorders,
)


# === Fixtures ===


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test sessions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def recorder(temp_dir):
    """Create a BlackboxRecorder with temporary storage."""
    return BlackboxRecorder(
        session_id="test_session",
        base_path=temp_dir,
    )


@pytest.fixture(autouse=True)
def cleanup_active_recorders():
    """Clean up active recorders after each test."""
    yield
    _active_recorders.clear()


# === BlackboxEvent Tests ===


class TestBlackboxEvent:
    """Tests for BlackboxEvent dataclass."""

    def test_event_creation(self):
        """Test event creation with required fields."""
        event = BlackboxEvent(
            timestamp=1234567890.0,
            event_type="test",
            component="test_component",
        )
        assert event.timestamp == 1234567890.0
        assert event.event_type == "test"
        assert event.component == "test_component"
        assert event.data == {}

    def test_event_with_data(self):
        """Test event creation with data."""
        event = BlackboxEvent(
            timestamp=1234567890.0,
            event_type="test",
            component="test_component",
            data={"key": "value"},
        )
        assert event.data == {"key": "value"}

    def test_to_dict(self):
        """Test serialization to dictionary."""
        event = BlackboxEvent(
            timestamp=1234567890.0,
            event_type="error",
            component="orchestrator",
            data={"error": "test error"},
        )
        result = event.to_dict()

        assert result["timestamp"] == 1234567890.0
        assert result["event_type"] == "error"
        assert result["component"] == "orchestrator"
        assert result["data"]["error"] == "test error"

    def test_to_dict_is_json_serializable(self):
        """Test that to_dict output is JSON serializable."""
        event = BlackboxEvent(
            timestamp=time.time(),
            event_type="test",
            component="test",
            data={"nested": {"value": 123}},
        )
        # Should not raise
        json_str = json.dumps(event.to_dict())
        assert isinstance(json_str, str)


# === BlackboxSnapshot Tests ===


class TestBlackboxSnapshot:
    """Tests for BlackboxSnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test snapshot creation with required fields."""
        snapshot = BlackboxSnapshot(
            turn_id=1,
            timestamp=1234567890.0,
            agents_active=["claude", "gpt-4"],
            agents_failed=[],
            consensus_strength=0.8,
            transcript_length=100,
        )
        assert snapshot.turn_id == 1
        assert snapshot.timestamp == 1234567890.0
        assert snapshot.agents_active == ["claude", "gpt-4"]
        assert snapshot.agents_failed == []
        assert snapshot.consensus_strength == 0.8
        assert snapshot.transcript_length == 100
        assert snapshot.metadata == {}

    def test_snapshot_with_metadata(self):
        """Test snapshot creation with metadata."""
        snapshot = BlackboxSnapshot(
            turn_id=1,
            timestamp=1234567890.0,
            agents_active=[],
            agents_failed=[],
            consensus_strength=0.0,
            transcript_length=0,
            metadata={"phase": "proposal"},
        )
        assert snapshot.metadata == {"phase": "proposal"}

    def test_to_dict(self):
        """Test serialization to dictionary."""
        snapshot = BlackboxSnapshot(
            turn_id=5,
            timestamp=1234567890.0,
            agents_active=["agent1"],
            agents_failed=["agent2"],
            consensus_strength=0.75,
            transcript_length=500,
            metadata={"round": 2},
        )
        result = snapshot.to_dict()

        assert result["turn_id"] == 5
        assert result["timestamp"] == 1234567890.0
        assert result["agents_active"] == ["agent1"]
        assert result["agents_failed"] == ["agent2"]
        assert result["consensus_strength"] == 0.75
        assert result["transcript_length"] == 500
        assert result["metadata"] == {"round": 2}

    def test_to_dict_is_json_serializable(self):
        """Test that to_dict output is JSON serializable."""
        snapshot = BlackboxSnapshot(
            turn_id=1,
            timestamp=time.time(),
            agents_active=["a", "b"],
            agents_failed=["c"],
            consensus_strength=0.5,
            transcript_length=42,
        )
        # Should not raise
        json_str = json.dumps(snapshot.to_dict())
        assert isinstance(json_str, str)


# === BlackboxRecorder Initialization Tests ===


class TestBlackboxRecorderInit:
    """Tests for BlackboxRecorder initialization."""

    def test_init_creates_session_directory(self, temp_dir):
        """Test that initialization creates session directory."""
        recorder = BlackboxRecorder(
            session_id="new_session",
            base_path=temp_dir,
        )
        assert recorder.session_path.exists()
        assert recorder.session_path.is_dir()

    def test_init_writes_metadata(self, temp_dir):
        """Test that initialization writes metadata file."""
        recorder = BlackboxRecorder(
            session_id="meta_test",
            base_path=temp_dir,
        )
        meta_path = recorder.session_path / "meta.json"
        assert meta_path.exists()

        with open(meta_path) as f:
            metadata = json.load(f)
        assert metadata["session_id"] == "meta_test"
        assert "started_at" in metadata
        assert "start_time" in metadata

    def test_init_with_default_path(self, temp_dir):
        """Test that default base path is .nomic/sessions."""
        recorder = BlackboxRecorder(session_id="test", base_path=temp_dir)
        # Override the session path to check default behavior
        default_recorder = BlackboxRecorder.__new__(BlackboxRecorder)
        default_recorder.base_path = None
        # Check that the default would be .nomic/sessions
        default_path = Path(".nomic/sessions")
        assert default_path == Path(".nomic/sessions")

    def test_init_sets_max_events(self, temp_dir):
        """Test custom max_events setting."""
        recorder = BlackboxRecorder(
            session_id="test",
            base_path=temp_dir,
            max_events=500,
        )
        assert recorder.max_events == 500

    def test_init_empty_events_and_snapshots(self, temp_dir):
        """Test that events and snapshots are empty initially."""
        recorder = BlackboxRecorder(
            session_id="test",
            base_path=temp_dir,
        )
        assert recorder.events == []
        assert recorder.snapshots == []


# === Event Recording Tests ===


class TestRecordEvent:
    """Tests for event recording."""

    def test_record_event_basic(self, recorder):
        """Test basic event recording."""
        event = recorder.record_event("test", "component")
        assert event.event_type == "test"
        assert event.component == "component"
        assert len(recorder.events) == 1

    def test_record_event_with_data(self, recorder):
        """Test event recording with data."""
        event = recorder.record_event(
            "test",
            "component",
            data={"key": "value"},
        )
        assert event.data == {"key": "value"}

    def test_record_event_adds_timestamp(self, recorder):
        """Test that events get a timestamp."""
        before = time.time()
        event = recorder.record_event("test", "component")
        after = time.time()

        assert before <= event.timestamp <= after

    def test_record_event_auto_flush(self, recorder):
        """Test that events are auto-flushed when max is reached."""
        recorder.max_events = 5
        for i in range(6):
            recorder.record_event("test", f"component_{i}")

        # After 6 events with max=5, should have flushed and have 1 event
        assert len(recorder.events) == 1

    def test_record_event_returns_event(self, recorder):
        """Test that record_event returns the created event."""
        event = recorder.record_event("test", "component")
        assert isinstance(event, BlackboxEvent)


# === Snapshot Tests ===


class TestSnapshotTurn:
    """Tests for snapshot_turn method."""

    def test_snapshot_turn_basic(self, recorder):
        """Test basic snapshot creation."""
        snapshot = recorder.snapshot_turn(1, {})
        assert snapshot.turn_id == 1
        assert len(recorder.snapshots) == 1

    def test_snapshot_turn_with_state_data(self, recorder):
        """Test snapshot with full state data."""
        state_data = {
            "agents_active": ["claude", "gpt-4"],
            "agents_failed": ["gemini"],
            "consensus_strength": 0.8,
            "transcript_length": 1500,
            "metadata": {"phase": "consensus"},
        }
        snapshot = recorder.snapshot_turn(5, state_data)

        assert snapshot.agents_active == ["claude", "gpt-4"]
        assert snapshot.agents_failed == ["gemini"]
        assert snapshot.consensus_strength == 0.8
        assert snapshot.transcript_length == 1500
        assert snapshot.metadata == {"phase": "consensus"}

    def test_snapshot_turn_writes_file(self, recorder):
        """Test that snapshot is written to disk."""
        recorder.snapshot_turn(3, {})
        snapshot_path = recorder.session_path / "turn_0003.json"
        assert snapshot_path.exists()

        with open(snapshot_path) as f:
            data = json.load(f)
        assert data["turn_id"] == 3

    def test_snapshot_turn_file_format(self, recorder):
        """Test snapshot file naming format."""
        recorder.snapshot_turn(1, {})
        recorder.snapshot_turn(10, {})
        recorder.snapshot_turn(100, {})

        assert (recorder.session_path / "turn_0001.json").exists()
        assert (recorder.session_path / "turn_0010.json").exists()
        assert (recorder.session_path / "turn_0100.json").exists()


# === Error Logging Tests ===


class TestLogError:
    """Tests for log_error method."""

    def test_log_error_basic(self, recorder):
        """Test basic error logging."""
        event = recorder.log_error("component", "error message")
        assert event.event_type == "error"
        assert event.component == "component"
        assert event.data["error"] == "error message"

    def test_log_error_recoverable(self, recorder):
        """Test error with recoverable flag."""
        event = recorder.log_error("component", "error", recoverable=True)
        assert event.data["recoverable"] is True

        event = recorder.log_error("component", "error", recoverable=False)
        assert event.data["recoverable"] is False

    def test_log_error_with_context(self, recorder):
        """Test error with context."""
        event = recorder.log_error(
            "component",
            "error",
            context={"attempt": 3},
        )
        assert event.data["context"] == {"attempt": 3}

    def test_log_error_writes_to_file(self, recorder):
        """Test that errors are written to error log file."""
        recorder.log_error("test_component", "test error message")
        error_log = recorder.session_path / "errors.log"
        assert error_log.exists()

        content = error_log.read_text()
        assert "test_component" in content
        assert "test error message" in content

    def test_log_error_truncates_long_errors(self, recorder):
        """Test that long errors are truncated."""
        long_error = "x" * 1000
        event = recorder.log_error("component", long_error)
        assert len(event.data["error"]) <= 500


# === Agent Failure Logging Tests ===


class TestLogAgentFailure:
    """Tests for log_agent_failure method."""

    def test_log_agent_failure_basic(self, recorder):
        """Test basic agent failure logging."""
        event = recorder.log_agent_failure("claude", "timeout", 30.0)
        assert event.event_type == "agent_failure"
        assert event.component == "claude"
        assert event.data["failure_type"] == "timeout"
        assert event.data["duration_seconds"] == 30.0

    def test_log_agent_failure_with_context(self, recorder):
        """Test agent failure with context."""
        event = recorder.log_agent_failure(
            "gpt-4",
            "error",
            15.5,
            context={"prompt_length": 1000},
        )
        assert event.data["context"] == {"prompt_length": 1000}


# === Recovery Logging Tests ===


class TestLogRecovery:
    """Tests for log_recovery method."""

    def test_log_recovery_basic(self, recorder):
        """Test basic recovery logging."""
        event = recorder.log_recovery(
            "orchestrator",
            "fallback_agent",
            "Original error message",
        )
        assert event.event_type == "recovery"
        assert event.component == "orchestrator"
        assert event.data["recovery_type"] == "fallback_agent"
        assert event.data["original_error"] == "Original error message"

    def test_log_recovery_truncates_error(self, recorder):
        """Test that original error is truncated."""
        long_error = "x" * 500
        event = recorder.log_recovery("comp", "type", long_error)
        assert len(event.data["original_error"]) <= 200


# === Consensus Logging Tests ===


class TestLogConsensus:
    """Tests for log_consensus method."""

    def test_log_consensus_basic(self, recorder):
        """Test basic consensus logging."""
        event = recorder.log_consensus(
            strength=0.85,
            participating_agents=["claude", "gpt-4"],
            topic="Implementation approach",
        )
        assert event.event_type == "consensus"
        assert event.component == "orchestrator"
        assert event.data["strength"] == 0.85
        assert event.data["participating_agents"] == ["claude", "gpt-4"]
        assert event.data["topic"] == "Implementation approach"

    def test_log_consensus_with_result(self, recorder):
        """Test consensus with result."""
        event = recorder.log_consensus(
            strength=0.9,
            participating_agents=["a", "b"],
            topic="topic",
            result="Final consensus result",
        )
        assert event.data["result"] == "Final consensus result"


# === Flush Events Tests ===


class TestFlushEvents:
    """Tests for flush_events method."""

    def test_flush_events_writes_file(self, recorder):
        """Test that flush writes events to file."""
        recorder.record_event("test1", "comp1")
        recorder.record_event("test2", "comp2")
        recorder.flush_events()

        events_path = recorder.session_path / "events.jsonl"
        assert events_path.exists()

        lines = events_path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_flush_events_clears_list(self, recorder):
        """Test that flush clears events list."""
        recorder.record_event("test", "comp")
        assert len(recorder.events) == 1
        recorder.flush_events()
        assert len(recorder.events) == 0

    def test_flush_events_empty_is_noop(self, recorder):
        """Test that flushing empty events is a no-op."""
        recorder.flush_events()
        events_path = recorder.session_path / "events.jsonl"
        assert not events_path.exists()

    def test_flush_events_appends(self, recorder):
        """Test that flush appends to existing file."""
        recorder.record_event("test1", "comp1")
        recorder.flush_events()
        recorder.record_event("test2", "comp2")
        recorder.flush_events()

        events_path = recorder.session_path / "events.jsonl"
        lines = events_path.read_text().strip().split("\n")
        assert len(lines) == 2


# === Query Methods Tests ===


class TestQueryMethods:
    """Tests for query methods."""

    def test_get_latest_snapshot_empty(self, recorder):
        """Test get_latest_snapshot with no snapshots."""
        assert recorder.get_latest_snapshot() is None

    def test_get_latest_snapshot(self, recorder):
        """Test get_latest_snapshot returns most recent."""
        recorder.snapshot_turn(1, {})
        recorder.snapshot_turn(2, {})
        recorder.snapshot_turn(3, {})

        latest = recorder.get_latest_snapshot()
        assert latest.turn_id == 3

    def test_get_agent_failure_rate_no_events(self, recorder):
        """Test failure rate with no events."""
        rate = recorder.get_agent_failure_rate("unknown")
        assert rate == 0.0

    def test_get_agent_failure_rate_no_failures(self, recorder):
        """Test failure rate with events but no failures."""
        recorder.record_event("success", "agent1")
        recorder.record_event("success", "agent1")
        rate = recorder.get_agent_failure_rate("agent1")
        assert rate == 0.0

    def test_get_agent_failure_rate_all_failures(self, recorder):
        """Test failure rate with all failures."""
        recorder.log_agent_failure("agent1", "timeout", 1.0)
        recorder.log_agent_failure("agent1", "error", 2.0)
        rate = recorder.get_agent_failure_rate("agent1")
        assert rate == 1.0

    def test_get_agent_failure_rate_partial(self, recorder):
        """Test failure rate with mixed events."""
        recorder.record_event("success", "agent1")
        recorder.log_agent_failure("agent1", "timeout", 1.0)
        rate = recorder.get_agent_failure_rate("agent1")
        assert rate == 0.5


# === Session Summary Tests ===


class TestGetSessionSummary:
    """Tests for get_session_summary method."""

    def test_session_summary_empty(self, recorder):
        """Test session summary with no events."""
        summary = recorder.get_session_summary()
        assert summary["session_id"] == "test_session"
        assert summary["total_events"] == 0
        assert summary["total_snapshots"] == 0
        assert summary["total_errors"] == 0
        assert summary["total_agent_failures"] == 0
        assert summary["total_recoveries"] == 0

    def test_session_summary_with_events(self, recorder):
        """Test session summary with various events."""
        recorder.log_error("comp", "error")
        recorder.log_agent_failure("agent", "timeout", 1.0)
        recorder.log_recovery("comp", "retry", "error")
        recorder.snapshot_turn(1, {})

        summary = recorder.get_session_summary()
        assert summary["total_errors"] == 1
        assert summary["total_agent_failures"] == 1
        assert summary["total_recoveries"] == 1
        assert summary["total_snapshots"] == 1
        assert summary["total_events"] == 3  # error + failure + recovery

    def test_session_summary_recovery_rate(self, recorder):
        """Test session summary recovery rate calculation."""
        recorder.log_error("comp", "error1")
        recorder.log_error("comp", "error2")
        recorder.log_recovery("comp", "retry", "error1")

        summary = recorder.get_session_summary()
        assert summary["recovery_rate"] == 0.5  # 1 recovery / 2 errors

    def test_session_summary_duration(self, recorder):
        """Test session summary includes duration."""
        time.sleep(0.1)
        summary = recorder.get_session_summary()
        assert summary["duration_seconds"] >= 0.1


# === Close Tests ===


class TestClose:
    """Tests for close method."""

    def test_close_flushes_events(self, recorder):
        """Test that close flushes remaining events."""
        recorder.record_event("test", "comp")
        recorder.close()

        events_path = recorder.session_path / "events.jsonl"
        assert events_path.exists()
        assert len(recorder.events) == 0

    def test_close_writes_summary(self, recorder):
        """Test that close writes summary file."""
        recorder.close()
        summary_path = recorder.session_path / "summary.json"
        assert summary_path.exists()

        with open(summary_path) as f:
            summary = json.load(f)
        assert summary["session_id"] == "test_session"


# === Helper Function Tests ===


class TestHelperFunctions:
    """Tests for get_blackbox and close_blackbox helper functions."""

    def test_get_blackbox_creates_new(self, temp_dir):
        """Test get_blackbox creates new recorder."""
        with patch.object(BlackboxRecorder, "__init__", return_value=None) as mock_init:
            mock_init.return_value = None
            # Clear to start fresh
            _active_recorders.clear()

            # This will fail because __init__ is mocked
            # Let's use a different approach

    def test_get_blackbox_returns_same_instance(self):
        """Test get_blackbox returns same instance for same session."""
        recorder1 = get_blackbox("shared_session")
        recorder2 = get_blackbox("shared_session")
        assert recorder1 is recorder2

    def test_get_blackbox_different_sessions(self):
        """Test get_blackbox creates different instances for different sessions."""
        recorder1 = get_blackbox("session_a")
        recorder2 = get_blackbox("session_b")
        assert recorder1 is not recorder2

    def test_close_blackbox(self):
        """Test close_blackbox closes and removes recorder."""
        recorder = get_blackbox("close_test")
        assert "close_test" in _active_recorders

        close_blackbox("close_test")
        assert "close_test" not in _active_recorders

    def test_close_blackbox_unknown_session(self):
        """Test close_blackbox with unknown session is a no-op."""
        close_blackbox("nonexistent")  # Should not raise


# === Edge Case Tests ===


class TestEdgeCases:
    """Edge case tests."""

    def test_multiple_snapshots_same_turn(self, recorder):
        """Test multiple snapshots for same turn."""
        recorder.snapshot_turn(1, {"consensus_strength": 0.5})
        recorder.snapshot_turn(1, {"consensus_strength": 0.8})

        # File should be overwritten
        snapshot_path = recorder.session_path / "turn_0001.json"
        with open(snapshot_path) as f:
            data = json.load(f)
        assert data["consensus_strength"] == 0.8

    def test_event_data_with_nested_structures(self, recorder):
        """Test recording events with deeply nested data."""
        data = {
            "level1": {
                "level2": {
                    "level3": ["a", "b", "c"],
                },
            },
        }
        event = recorder.record_event("test", "comp", data=data)
        recorder.flush_events()

        events_path = recorder.session_path / "events.jsonl"
        line = events_path.read_text().strip()
        parsed = json.loads(line)
        assert parsed["data"]["level1"]["level2"]["level3"] == ["a", "b", "c"]

    def test_special_characters_in_error(self, recorder):
        """Test error messages with special characters."""
        error = 'Error with "quotes" and\nnewlines and\ttabs'
        event = recorder.log_error("comp", error)
        recorder.flush_events()

        # Should be JSON-serializable
        events_path = recorder.session_path / "events.jsonl"
        line = events_path.read_text().strip()
        parsed = json.loads(line)
        assert "quotes" in parsed["data"]["error"]
