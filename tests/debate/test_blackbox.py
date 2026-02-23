"""Tests for aragora.debate.blackbox — flight recorder for debate sessions."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from aragora.debate.blackbox import (
    BlackboxEvent,
    BlackboxRecorder,
    BlackboxSnapshot,
    _active_recorders,
    close_blackbox,
    get_blackbox,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def recorder(tmp_path: Path) -> BlackboxRecorder:
    """Create a BlackboxRecorder rooted in a temporary directory."""
    return BlackboxRecorder("test-session", base_path=tmp_path)


@pytest.fixture(autouse=True)
def _clear_active_recorders():
    """Ensure global registry is clean before/after each test."""
    _active_recorders.clear()
    yield
    _active_recorders.clear()


# ---------------------------------------------------------------------------
# BlackboxEvent
# ---------------------------------------------------------------------------


class TestBlackboxEvent:
    def test_to_dict_contains_all_fields(self):
        event = BlackboxEvent(
            timestamp=1000.0,
            event_type="error",
            component="orchestrator",
            data={"key": "value"},
        )
        d = event.to_dict()
        assert d == {
            "timestamp": 1000.0,
            "event_type": "error",
            "component": "orchestrator",
            "data": {"key": "value"},
        }

    def test_to_dict_default_data(self):
        event = BlackboxEvent(timestamp=1.0, event_type="turn", component="agent")
        d = event.to_dict()
        assert d["data"] == {}


# ---------------------------------------------------------------------------
# BlackboxSnapshot
# ---------------------------------------------------------------------------


class TestBlackboxSnapshot:
    def test_to_dict_contains_all_fields(self):
        snap = BlackboxSnapshot(
            turn_id=3,
            timestamp=2000.0,
            agents_active=["claude", "gpt"],
            agents_failed=["gemini"],
            consensus_strength=0.85,
            transcript_length=42,
            metadata={"phase": "critique"},
        )
        d = snap.to_dict()
        assert d["turn_id"] == 3
        assert d["agents_active"] == ["claude", "gpt"]
        assert d["agents_failed"] == ["gemini"]
        assert d["consensus_strength"] == 0.85
        assert d["transcript_length"] == 42
        assert d["metadata"] == {"phase": "critique"}

    def test_to_dict_default_metadata(self):
        snap = BlackboxSnapshot(
            turn_id=0,
            timestamp=0.0,
            agents_active=[],
            agents_failed=[],
            consensus_strength=0.0,
            transcript_length=0,
        )
        assert snap.to_dict()["metadata"] == {}


# ---------------------------------------------------------------------------
# BlackboxRecorder — init
# ---------------------------------------------------------------------------


class TestBlackboxRecorderInit:
    def test_creates_session_directory(self, tmp_path: Path):
        rec = BlackboxRecorder("sess-1", base_path=tmp_path)
        assert rec.session_path.is_dir()
        assert rec.session_path == tmp_path / "sess-1"

    def test_writes_meta_json(self, tmp_path: Path):
        rec = BlackboxRecorder("sess-2", base_path=tmp_path)
        meta_path = rec.session_path / "meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["session_id"] == "sess-2"
        assert "started_at" in meta
        assert "start_time" in meta

    def test_default_max_events(self, tmp_path: Path):
        rec = BlackboxRecorder("sess-3", base_path=tmp_path)
        assert rec.max_events == 10000


# ---------------------------------------------------------------------------
# BlackboxRecorder — record_event
# ---------------------------------------------------------------------------


class TestRecordEvent:
    def test_creates_event_with_correct_fields(self, recorder: BlackboxRecorder):
        event = recorder.record_event("turn", "orchestrator", data={"round": 1})
        assert event.event_type == "turn"
        assert event.component == "orchestrator"
        assert event.data == {"round": 1}
        assert isinstance(event.timestamp, float)

    def test_appends_event_to_list(self, recorder: BlackboxRecorder):
        recorder.record_event("turn", "agent")
        assert len(recorder.events) == 1

    def test_data_defaults_to_empty_dict(self, recorder: BlackboxRecorder):
        event = recorder.record_event("turn", "agent")
        assert event.data == {}

    def test_auto_flush_at_max_events(self, tmp_path: Path):
        rec = BlackboxRecorder("flush-sess", base_path=tmp_path, max_events=5)
        for i in range(5):
            rec.record_event("turn", "agent", data={"i": i})
        # After hitting max_events, events should have been flushed
        assert len(rec.events) == 0
        events_path = rec.session_path / "events.jsonl"
        assert events_path.exists()
        lines = events_path.read_text().strip().split("\n")
        assert len(lines) == 5


# ---------------------------------------------------------------------------
# BlackboxRecorder — snapshot_turn
# ---------------------------------------------------------------------------


class TestSnapshotTurn:
    def test_creates_snapshot_with_state_data(self, recorder: BlackboxRecorder):
        state = {
            "agents_active": ["claude"],
            "agents_failed": ["gpt"],
            "consensus_strength": 0.7,
            "transcript_length": 100,
            "metadata": {"topic": "rates"},
        }
        snap = recorder.snapshot_turn(1, state)
        assert snap.turn_id == 1
        assert snap.agents_active == ["claude"]
        assert snap.agents_failed == ["gpt"]
        assert snap.consensus_strength == 0.7
        assert snap.transcript_length == 100
        assert snap.metadata == {"topic": "rates"}

    def test_writes_turn_file_to_disk(self, recorder: BlackboxRecorder):
        recorder.snapshot_turn(3, {"agents_active": ["a"]})
        turn_file = recorder.session_path / "turn_0003.json"
        assert turn_file.exists()
        data = json.loads(turn_file.read_text())
        assert data["turn_id"] == 3

    def test_handles_write_error_gracefully(self, recorder: BlackboxRecorder):
        # Make session_path read-only to trigger OSError on write
        with patch("builtins.open", side_effect=OSError("disk full")):
            snap = recorder.snapshot_turn(99, {"agents_active": []})
        # Should still return the snapshot without raising
        assert snap.turn_id == 99


# ---------------------------------------------------------------------------
# BlackboxRecorder — log_error
# ---------------------------------------------------------------------------


class TestLogError:
    def test_records_error_event(self, recorder: BlackboxRecorder):
        event = recorder.log_error("calibration", "Division by zero")
        assert event.event_type == "error"
        assert event.component == "calibration"
        assert event.data["error"] == "Division by zero"
        assert event.data["recoverable"] is True
        assert event.data["context"] == {}

    def test_writes_to_errors_log(self, recorder: BlackboxRecorder):
        recorder.log_error("agent", "timeout", context={"retry": 2})
        error_log = recorder.session_path / "errors.log"
        assert error_log.exists()
        content = error_log.read_text()
        assert "agent: timeout" in content

    def test_truncates_long_error_to_500_chars(self, recorder: BlackboxRecorder):
        long_msg = "x" * 1000
        event = recorder.log_error("comp", long_msg)
        assert len(event.data["error"]) == 500

    def test_non_recoverable_flag(self, recorder: BlackboxRecorder):
        event = recorder.log_error("comp", "fatal", recoverable=False)
        assert event.data["recoverable"] is False


# ---------------------------------------------------------------------------
# BlackboxRecorder — log_agent_failure
# ---------------------------------------------------------------------------


class TestLogAgentFailure:
    def test_records_failure_event(self, recorder: BlackboxRecorder):
        event = recorder.log_agent_failure("claude", "timeout", 90.5)
        assert event.event_type == "agent_failure"
        assert event.component == "claude"
        assert event.data["failure_type"] == "timeout"
        assert event.data["duration_seconds"] == 90.5
        assert event.data["context"] == {}

    def test_with_context(self, recorder: BlackboxRecorder):
        ctx = {"round": 3, "retries": 2}
        event = recorder.log_agent_failure("gpt", "error", 10.0, context=ctx)
        assert event.data["context"] == ctx


# ---------------------------------------------------------------------------
# BlackboxRecorder — log_recovery
# ---------------------------------------------------------------------------


class TestLogRecovery:
    def test_records_recovery_event(self, recorder: BlackboxRecorder):
        event = recorder.log_recovery("agent", "fallback", "timeout error")
        assert event.event_type == "recovery"
        assert event.component == "agent"
        assert event.data["recovery_type"] == "fallback"
        assert event.data["original_error"] == "timeout error"
        assert event.data["context"] == {}

    def test_truncates_original_error_to_200(self, recorder: BlackboxRecorder):
        long_err = "e" * 500
        event = recorder.log_recovery("comp", "retry", long_err)
        assert len(event.data["original_error"]) == 200


# ---------------------------------------------------------------------------
# BlackboxRecorder — log_consensus
# ---------------------------------------------------------------------------


class TestLogConsensus:
    def test_records_consensus_event(self, recorder: BlackboxRecorder):
        event = recorder.log_consensus(
            strength=0.92,
            participating_agents=["claude", "gpt"],
            topic="rate limiting",
            result="use token bucket",
        )
        assert event.event_type == "consensus"
        assert event.component == "orchestrator"
        assert event.data["strength"] == 0.92
        assert event.data["participating_agents"] == ["claude", "gpt"]
        assert event.data["topic"] == "rate limiting"
        assert event.data["result"] == "use token bucket"

    def test_result_none(self, recorder: BlackboxRecorder):
        event = recorder.log_consensus(0.5, ["a"], "topic")
        assert event.data["result"] is None


# ---------------------------------------------------------------------------
# BlackboxRecorder — flush_events
# ---------------------------------------------------------------------------


class TestFlushEvents:
    def test_writes_events_to_jsonl(self, recorder: BlackboxRecorder):
        recorder.record_event("turn", "agent", data={"n": 1})
        recorder.record_event("error", "agent", data={"n": 2})
        recorder.flush_events()

        events_path = recorder.session_path / "events.jsonl"
        assert events_path.exists()
        lines = events_path.read_text().strip().split("\n")
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["event_type"] == "turn"

    def test_clears_event_list(self, recorder: BlackboxRecorder):
        recorder.record_event("turn", "agent")
        recorder.flush_events()
        assert len(recorder.events) == 0

    def test_empty_list_is_noop(self, recorder: BlackboxRecorder):
        recorder.flush_events()
        events_path = recorder.session_path / "events.jsonl"
        assert not events_path.exists()

    def test_appends_on_subsequent_flush(self, recorder: BlackboxRecorder):
        recorder.record_event("turn", "a")
        recorder.flush_events()
        recorder.record_event("error", "b")
        recorder.flush_events()
        events_path = recorder.session_path / "events.jsonl"
        lines = events_path.read_text().strip().split("\n")
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# BlackboxRecorder — get_latest_snapshot
# ---------------------------------------------------------------------------


class TestGetLatestSnapshot:
    def test_returns_none_when_empty(self, recorder: BlackboxRecorder):
        assert recorder.get_latest_snapshot() is None

    def test_returns_last_snapshot(self, recorder: BlackboxRecorder):
        recorder.snapshot_turn(1, {"agents_active": ["a"]})
        recorder.snapshot_turn(2, {"agents_active": ["b"]})
        snap = recorder.get_latest_snapshot()
        assert snap is not None
        assert snap.turn_id == 2


# ---------------------------------------------------------------------------
# BlackboxRecorder — get_agent_failure_rate
# ---------------------------------------------------------------------------


class TestGetAgentFailureRate:
    def test_zero_events_returns_zero(self, recorder: BlackboxRecorder):
        assert recorder.get_agent_failure_rate("claude") == 0.0

    def test_calculates_correctly(self, recorder: BlackboxRecorder):
        # 2 failures out of 4 events for "claude"
        recorder.log_agent_failure("claude", "timeout", 1.0)
        recorder.log_agent_failure("claude", "error", 2.0)
        recorder.record_event("turn", "claude")
        recorder.record_event("turn", "claude")
        rate = recorder.get_agent_failure_rate("claude")
        assert rate == pytest.approx(0.5)

    def test_ignores_other_agents(self, recorder: BlackboxRecorder):
        recorder.log_agent_failure("claude", "timeout", 1.0)
        recorder.record_event("turn", "gpt")
        assert recorder.get_agent_failure_rate("gpt") == 0.0


# ---------------------------------------------------------------------------
# BlackboxRecorder — get_session_summary
# ---------------------------------------------------------------------------


class TestGetSessionSummary:
    def test_counts_events_by_type(self, recorder: BlackboxRecorder):
        recorder.log_error("a", "err1")
        recorder.log_error("b", "err2")
        recorder.log_agent_failure("c", "timeout", 1.0)
        recorder.log_recovery("a", "retry", "err1")
        summary = recorder.get_session_summary()
        assert summary["session_id"] == "test-session"
        assert summary["total_events"] == 4
        assert summary["total_errors"] == 2
        assert summary["total_agent_failures"] == 1
        assert summary["total_recoveries"] == 1

    def test_duration_is_positive(self, recorder: BlackboxRecorder):
        summary = recorder.get_session_summary()
        assert summary["duration_seconds"] >= 0

    def test_recovery_rate(self, recorder: BlackboxRecorder):
        recorder.log_error("a", "e1")
        recorder.log_error("a", "e2")
        recorder.log_recovery("a", "retry", "e1")
        summary = recorder.get_session_summary()
        # 1 recovery / 2 errors = 0.5
        assert summary["recovery_rate"] == pytest.approx(0.5)

    def test_recovery_rate_no_errors(self, recorder: BlackboxRecorder):
        # No errors: denominator is max(0, 1) = 1
        summary = recorder.get_session_summary()
        assert summary["recovery_rate"] == 0.0


# ---------------------------------------------------------------------------
# BlackboxRecorder — close
# ---------------------------------------------------------------------------


class TestClose:
    def test_flushes_events_and_writes_summary(self, recorder: BlackboxRecorder):
        recorder.record_event("turn", "agent")
        recorder.close()
        # Events flushed
        events_path = recorder.session_path / "events.jsonl"
        assert events_path.exists()
        # Summary written
        summary_path = recorder.session_path / "summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary["session_id"] == "test-session"


# ---------------------------------------------------------------------------
# Global functions
# ---------------------------------------------------------------------------


class TestGlobalFunctions:
    def test_get_blackbox_creates_new_recorder(self, tmp_path: Path):
        with patch(
            "aragora.debate.blackbox.BlackboxRecorder",
            wraps=BlackboxRecorder,
        ) as mock_cls:
            mock_cls.side_effect = lambda sid, **kw: BlackboxRecorder(sid, base_path=tmp_path)
            rec = get_blackbox("new-sess")
            assert rec is not None
            assert "new-sess" in _active_recorders

    def test_get_blackbox_returns_existing(self, tmp_path: Path):
        # Manually insert a recorder so we don't trigger import
        rec1 = BlackboxRecorder("reuse-sess", base_path=tmp_path)
        _active_recorders["reuse-sess"] = rec1
        rec2 = get_blackbox("reuse-sess")
        assert rec2 is rec1

    def test_close_blackbox_closes_and_removes(self, tmp_path: Path):
        rec = BlackboxRecorder("close-sess", base_path=tmp_path)
        _active_recorders["close-sess"] = rec
        close_blackbox("close-sess")
        assert "close-sess" not in _active_recorders
        # Summary should exist after close
        summary_path = rec.session_path / "summary.json"
        assert summary_path.exists()

    def test_close_blackbox_nonexistent_is_noop(self):
        # Should not raise
        close_blackbox("does-not-exist")
