"""
Tests for Blackbox Protocol module.

Tests cover:
- BlackboxEvent dataclass operations
- BlackboxSnapshot dataclass operations
- BlackboxRecorder initialization and lifecycle
- Event recording and logging
- Turn snapshots and persistence
- Error and recovery logging
- Session summaries and metrics
- Global recorder management
"""

import json
import time
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from aragora.debate.blackbox import (
    BlackboxEvent,
    BlackboxSnapshot,
    BlackboxRecorder,
    get_blackbox,
    close_blackbox,
    _active_recorders,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for blackbox storage."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def blackbox(temp_dir):
    """Create a blackbox recorder with temp storage."""
    recorder = BlackboxRecorder(
        session_id="test_session_123",
        base_path=temp_dir,
    )
    yield recorder
    # Cleanup
    recorder.close()


@pytest.fixture(autouse=True)
def cleanup_global_recorders():
    """Clean up global recorders before and after each test."""
    _active_recorders.clear()
    yield
    _active_recorders.clear()


# ============================================================================
# BlackboxEvent Tests
# ============================================================================


class TestBlackboxEvent:
    """Tests for BlackboxEvent dataclass."""

    def test_create_event(self):
        """Test creating a blackbox event."""
        event = BlackboxEvent(
            timestamp=1234567890.0,
            event_type="turn",
            component="orchestrator",
            data={"turn_id": 5},
        )

        assert event.timestamp == 1234567890.0
        assert event.event_type == "turn"
        assert event.component == "orchestrator"
        assert event.data["turn_id"] == 5

    def test_to_dict(self):
        """Test converting event to dictionary."""
        event = BlackboxEvent(
            timestamp=time.time(),
            event_type="error",
            component="agent",
            data={"error": "timeout"},
        )
        data = event.to_dict()

        assert isinstance(data, dict)
        assert "timestamp" in data
        assert data["event_type"] == "error"
        assert data["component"] == "agent"
        assert data["data"]["error"] == "timeout"

    def test_event_with_empty_data(self):
        """Test event with no data."""
        event = BlackboxEvent(
            timestamp=time.time(),
            event_type="test",
            component="test",
        )
        data = event.to_dict()

        assert data["data"] == {}


# ============================================================================
# BlackboxSnapshot Tests
# ============================================================================


class TestBlackboxSnapshot:
    """Tests for BlackboxSnapshot dataclass."""

    def test_create_snapshot(self):
        """Test creating a snapshot."""
        snapshot = BlackboxSnapshot(
            turn_id=3,
            timestamp=time.time(),
            agents_active=["claude", "gpt4"],
            agents_failed=["gemini"],
            consensus_strength=0.75,
            transcript_length=150,
        )

        assert snapshot.turn_id == 3
        assert len(snapshot.agents_active) == 2
        assert "gemini" in snapshot.agents_failed
        assert snapshot.consensus_strength == 0.75

    def test_to_dict(self):
        """Test converting snapshot to dictionary."""
        snapshot = BlackboxSnapshot(
            turn_id=1,
            timestamp=1234567890.0,
            agents_active=["a1"],
            agents_failed=[],
            consensus_strength=0.5,
            transcript_length=10,
            metadata={"topic": "AI"},
        )
        data = snapshot.to_dict()

        assert isinstance(data, dict)
        assert data["turn_id"] == 1
        assert data["metadata"]["topic"] == "AI"


# ============================================================================
# BlackboxRecorder Initialization Tests
# ============================================================================


class TestBlackboxRecorderInit:
    """Tests for BlackboxRecorder initialization."""

    def test_initialization(self, temp_dir):
        """Test recorder initialization."""
        recorder = BlackboxRecorder(
            session_id="test_123",
            base_path=temp_dir,
        )

        assert recorder.session_id == "test_123"
        assert recorder.session_path.exists()
        assert len(recorder.events) == 0
        assert len(recorder.snapshots) == 0

    def test_creates_session_directory(self, temp_dir):
        """Test that session directory is created."""
        recorder = BlackboxRecorder(
            session_id="my_session",
            base_path=temp_dir,
        )

        expected_path = temp_dir / "my_session"
        assert expected_path.exists()

    def test_writes_metadata(self, temp_dir):
        """Test that metadata file is written."""
        recorder = BlackboxRecorder(
            session_id="meta_test",
            base_path=temp_dir,
        )

        meta_path = temp_dir / "meta_test" / "meta.json"
        assert meta_path.exists()

        with open(meta_path) as f:
            metadata = json.load(f)

        assert metadata["session_id"] == "meta_test"
        assert "started_at" in metadata


# ============================================================================
# Event Recording Tests
# ============================================================================


class TestEventRecording:
    """Tests for event recording."""

    def test_record_event(self, blackbox):
        """Test recording a generic event."""
        event = blackbox.record_event(
            event_type="turn",
            component="orchestrator",
            data={"turn_id": 1},
        )

        assert event.event_type == "turn"
        assert len(blackbox.events) == 1

    def test_record_multiple_events(self, blackbox):
        """Test recording multiple events."""
        for i in range(5):
            blackbox.record_event("turn", "orchestrator", {"turn_id": i})

        assert len(blackbox.events) == 5

    def test_auto_flush_on_max_events(self, temp_dir):
        """Test automatic flush when max events reached."""
        recorder = BlackboxRecorder(
            session_id="flush_test",
            base_path=temp_dir,
            max_events=10,
        )

        # Record 15 events (should trigger flush at 10)
        for i in range(15):
            recorder.record_event("turn", "test", {"i": i})

        # After flush, only remaining events should be in memory
        assert len(recorder.events) == 5

        # Events should be persisted
        events_path = recorder.session_path / "events.jsonl"
        assert events_path.exists()


# ============================================================================
# Snapshot Tests
# ============================================================================


class TestSnapshots:
    """Tests for turn snapshots."""

    def test_snapshot_turn(self, blackbox):
        """Test creating a turn snapshot."""
        snapshot = blackbox.snapshot_turn(
            turn_id=1,
            state_data={
                "agents_active": ["claude", "gpt4"],
                "agents_failed": [],
                "consensus_strength": 0.8,
                "transcript_length": 50,
            },
        )

        assert snapshot.turn_id == 1
        assert len(snapshot.agents_active) == 2
        assert len(blackbox.snapshots) == 1

    def test_snapshot_persists_to_disk(self, blackbox):
        """Test that snapshots are written to disk."""
        blackbox.snapshot_turn(
            turn_id=3,
            state_data={"agents_active": ["a1"], "agents_failed": []},
        )

        snapshot_path = blackbox.session_path / "turn_0003.json"
        assert snapshot_path.exists()

        with open(snapshot_path) as f:
            data = json.load(f)
        assert data["turn_id"] == 3

    def test_get_latest_snapshot(self, blackbox):
        """Test getting the latest snapshot."""
        blackbox.snapshot_turn(1, {"agents_active": []})
        blackbox.snapshot_turn(2, {"agents_active": []})
        blackbox.snapshot_turn(3, {"agents_active": ["final"]})

        latest = blackbox.get_latest_snapshot()
        assert latest.turn_id == 3

    def test_get_latest_snapshot_empty(self, blackbox):
        """Test getting latest snapshot when none exist."""
        assert blackbox.get_latest_snapshot() is None


# ============================================================================
# Error Logging Tests
# ============================================================================


class TestErrorLogging:
    """Tests for error logging."""

    def test_log_error(self, blackbox):
        """Test logging an error."""
        event = blackbox.log_error(
            component="calibration",
            error="Division by zero",
            recoverable=True,
        )

        assert event.event_type == "error"
        assert event.data["error"] == "Division by zero"
        assert event.data["recoverable"] is True

    def test_log_error_truncates_long_message(self, blackbox):
        """Test that long error messages are truncated."""
        long_error = "x" * 1000
        event = blackbox.log_error("test", long_error)

        assert len(event.data["error"]) <= 500

    def test_log_error_writes_to_file(self, blackbox):
        """Test that errors are written to error log file."""
        blackbox.log_error("test", "Test error message")

        error_log = blackbox.session_path / "errors.log"
        assert error_log.exists()

        content = error_log.read_text()
        assert "Test error message" in content


# ============================================================================
# Agent Failure Tests
# ============================================================================


class TestAgentFailureLogging:
    """Tests for agent failure logging."""

    def test_log_agent_failure(self, blackbox):
        """Test logging an agent failure."""
        event = blackbox.log_agent_failure(
            agent_name="claude",
            failure_type="timeout",
            duration_seconds=90.0,
        )

        assert event.event_type == "agent_failure"
        assert event.component == "claude"
        assert event.data["failure_type"] == "timeout"
        assert event.data["duration_seconds"] == 90.0

    def test_get_agent_failure_rate(self, blackbox):
        """Test calculating agent failure rate."""
        # Record some events for agent
        blackbox.record_event("turn", "claude", {})
        blackbox.record_event("turn", "claude", {})
        blackbox.log_agent_failure("claude", "timeout", 30.0)

        rate = blackbox.get_agent_failure_rate("claude")
        assert rate == 1 / 3  # 1 failure out of 3 events

    def test_get_agent_failure_rate_no_events(self, blackbox):
        """Test failure rate with no events."""
        rate = blackbox.get_agent_failure_rate("unknown_agent")
        assert rate == 0.0


# ============================================================================
# Recovery Logging Tests
# ============================================================================


class TestRecoveryLogging:
    """Tests for recovery logging."""

    def test_log_recovery(self, blackbox):
        """Test logging a recovery."""
        event = blackbox.log_recovery(
            component="claude",
            recovery_type="fallback",
            original_error="API timeout",
        )

        assert event.event_type == "recovery"
        assert event.data["recovery_type"] == "fallback"
        assert event.data["original_error"] == "API timeout"

    def test_log_recovery_truncates_error(self, blackbox):
        """Test that long original errors are truncated."""
        long_error = "e" * 500
        event = blackbox.log_recovery("test", "retry", long_error)

        assert len(event.data["original_error"]) <= 200


# ============================================================================
# Consensus Logging Tests
# ============================================================================


class TestConsensusLogging:
    """Tests for consensus logging."""

    def test_log_consensus(self, blackbox):
        """Test logging a consensus event."""
        event = blackbox.log_consensus(
            strength=0.85,
            participating_agents=["claude", "gpt4", "gemini"],
            topic="AI safety guidelines",
            result="We should prioritize transparency",
        )

        assert event.event_type == "consensus"
        assert event.data["strength"] == 0.85
        assert len(event.data["participating_agents"]) == 3


# ============================================================================
# Session Summary Tests
# ============================================================================


class TestSessionSummary:
    """Tests for session summary generation."""

    def test_get_session_summary(self, blackbox):
        """Test getting session summary."""
        # Generate some activity
        blackbox.record_event("turn", "orchestrator", {})
        blackbox.log_error("agent", "test error")
        blackbox.log_agent_failure("claude", "timeout", 30.0)
        blackbox.log_recovery("claude", "fallback", "timeout")

        summary = blackbox.get_session_summary()

        assert summary["session_id"] == "test_session_123"
        assert summary["total_events"] == 4
        assert summary["total_errors"] == 1
        assert summary["total_agent_failures"] == 1
        assert summary["total_recoveries"] == 1

    def test_recovery_rate_calculation(self, blackbox):
        """Test recovery rate in summary."""
        blackbox.log_error("test", "e1")
        blackbox.log_error("test", "e2")
        blackbox.log_recovery("test", "retry", "e1")

        summary = blackbox.get_session_summary()
        assert summary["recovery_rate"] == 0.5  # 1 recovery / 2 errors


# ============================================================================
# Flush and Close Tests
# ============================================================================


class TestFlushAndClose:
    """Tests for flush and close operations."""

    def test_flush_events(self, blackbox):
        """Test flushing events to disk."""
        blackbox.record_event("turn", "test", {"n": 1})
        blackbox.record_event("turn", "test", {"n": 2})

        blackbox.flush_events()

        # Events should be cleared from memory
        assert len(blackbox.events) == 0

        # Events should be on disk
        events_path = blackbox.session_path / "events.jsonl"
        assert events_path.exists()

        lines = events_path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_flush_empty_events(self, blackbox):
        """Test flushing when no events."""
        # Should not raise
        blackbox.flush_events()

    def test_close_writes_summary(self, blackbox):
        """Test that close writes summary file."""
        blackbox.record_event("turn", "test", {})
        blackbox.close()

        summary_path = blackbox.session_path / "summary.json"
        assert summary_path.exists()


# ============================================================================
# Global Recorder Management Tests
# ============================================================================


class TestGlobalRecorderManagement:
    """Tests for global recorder functions."""

    def test_get_blackbox_creates_new(self, temp_dir):
        """Test get_blackbox creates new recorder."""
        # Use unique session ID to avoid conflicts
        session_id = f"test_new_{time.time()}"
        recorder = get_blackbox(session_id)

        assert session_id in _active_recorders
        assert recorder.session_id == session_id

        # Cleanup
        close_blackbox(session_id)

    def test_get_blackbox_returns_existing(self):
        """Test get_blackbox returns existing recorder."""
        # Create a recorder manually
        recorder1 = BlackboxRecorder.__new__(BlackboxRecorder)
        recorder1.session_id = "existing"
        recorder1.events = []
        recorder1.snapshots = []
        _active_recorders["existing"] = recorder1

        recorder2 = get_blackbox("existing")
        assert recorder2 is recorder1

    def test_close_blackbox(self, temp_dir):
        """Test closing a blackbox recorder."""
        recorder = BlackboxRecorder(
            session_id="to_close",
            base_path=temp_dir,
        )
        _active_recorders["to_close"] = recorder

        close_blackbox("to_close")

        assert "to_close" not in _active_recorders

    def test_close_blackbox_nonexistent(self):
        """Test closing nonexistent recorder doesn't raise."""
        # Should not raise
        close_blackbox("nonexistent_session")
