"""Tests for the debate InterventionManager.

Covers:
- Pause/resume state transitions
- Nudge delivery
- Challenge injection
- Evidence injection
- Intervention log tracking
- Thread safety
- WebSocket event emission
- Module-level registry
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.intervention import (
    DebateInterventionState,
    InterventionEntry,
    InterventionLog,
    InterventionManager,
    InterventionType,
    _reset_managers,
    get_intervention_manager,
    list_intervention_managers,
    remove_intervention_manager,
)


@pytest.fixture(autouse=True)
def _reset():
    """Reset the global manager registry between tests."""
    _reset_managers()
    yield
    _reset_managers()


# ============================================================================
# State Transitions
# ============================================================================


class TestPauseResume:
    """Pause/resume state transitions."""

    def test_initial_state_is_running(self):
        mgr = InterventionManager(debate_id="d1")
        assert mgr.state == DebateInterventionState.RUNNING
        assert mgr.is_running is True
        assert mgr.is_paused is False

    def test_pause_transitions_to_paused(self):
        mgr = InterventionManager(debate_id="d1")
        entry = mgr.pause(user_id="user-1")
        assert mgr.state == DebateInterventionState.PAUSED
        assert mgr.is_paused is True
        assert entry.intervention_type == InterventionType.PAUSE
        assert entry.user_id == "user-1"

    def test_resume_transitions_to_running(self):
        mgr = InterventionManager(debate_id="d1")
        mgr.pause()
        entry = mgr.resume(user_id="user-2")
        assert mgr.state == DebateInterventionState.RUNNING
        assert mgr.is_running is True
        assert entry.intervention_type == InterventionType.RESUME
        assert entry.user_id == "user-2"

    def test_cannot_pause_when_paused(self):
        mgr = InterventionManager(debate_id="d1")
        mgr.pause()
        with pytest.raises(ValueError, match="Cannot pause"):
            mgr.pause()

    def test_cannot_resume_when_running(self):
        mgr = InterventionManager(debate_id="d1")
        with pytest.raises(ValueError, match="Cannot resume"):
            mgr.resume()

    def test_cannot_pause_when_completed(self):
        mgr = InterventionManager(debate_id="d1")
        mgr.mark_completed()
        assert mgr.is_completed is True
        with pytest.raises(ValueError, match="Cannot pause"):
            mgr.pause()

    def test_cannot_resume_when_completed(self):
        mgr = InterventionManager(debate_id="d1")
        mgr.mark_completed()
        with pytest.raises(ValueError, match="Cannot resume"):
            mgr.resume()

    def test_multiple_pause_resume_cycles(self):
        mgr = InterventionManager(debate_id="d1")
        for _ in range(5):
            mgr.pause()
            assert mgr.is_paused
            mgr.resume()
            assert mgr.is_running


# ============================================================================
# Nudge
# ============================================================================


class TestNudge:
    """Nudge delivery."""

    def test_nudge_while_running(self):
        mgr = InterventionManager(debate_id="d1")
        entry = mgr.nudge("Consider the cost", user_id="u1")
        assert entry.intervention_type == InterventionType.NUDGE
        assert entry.message == "Consider the cost"
        assert entry.user_id == "u1"

    def test_nudge_while_paused(self):
        mgr = InterventionManager(debate_id="d1")
        mgr.pause()
        entry = mgr.nudge("Think about this", user_id="u1")
        assert entry.intervention_type == InterventionType.NUDGE

    def test_nudge_with_target_agent(self):
        mgr = InterventionManager(debate_id="d1")
        entry = mgr.nudge("Focus on security", target_agent="claude")
        assert entry.target_agent == "claude"

    def test_nudge_empty_message_raises(self):
        mgr = InterventionManager(debate_id="d1")
        with pytest.raises(ValueError, match="empty"):
            mgr.nudge("")

    def test_nudge_whitespace_only_raises(self):
        mgr = InterventionManager(debate_id="d1")
        with pytest.raises(ValueError, match="empty"):
            mgr.nudge("   ")

    def test_nudge_when_completed_raises(self):
        mgr = InterventionManager(debate_id="d1")
        mgr.mark_completed()
        with pytest.raises(ValueError, match="completed"):
            mgr.nudge("test")


# ============================================================================
# Challenge
# ============================================================================


class TestChallenge:
    """Challenge injection."""

    def test_challenge_while_running(self):
        mgr = InterventionManager(debate_id="d1")
        entry = mgr.challenge("What about the counterargument?", user_id="u1")
        assert entry.intervention_type == InterventionType.CHALLENGE
        assert entry.message == "What about the counterargument?"

    def test_challenge_empty_raises(self):
        mgr = InterventionManager(debate_id="d1")
        with pytest.raises(ValueError, match="empty"):
            mgr.challenge("")

    def test_challenge_when_completed_raises(self):
        mgr = InterventionManager(debate_id="d1")
        mgr.mark_completed()
        with pytest.raises(ValueError, match="completed"):
            mgr.challenge("test")


# ============================================================================
# Evidence Injection
# ============================================================================


class TestInjectEvidence:
    """Evidence injection."""

    def test_inject_evidence(self):
        mgr = InterventionManager(debate_id="d1")
        entry = mgr.inject_evidence(
            evidence="Studies show 80% improvement",
            source="https://example.com/study",
            user_id="u1",
        )
        assert entry.intervention_type == InterventionType.INJECT_EVIDENCE
        assert entry.message == "Studies show 80% improvement"
        assert entry.source == "https://example.com/study"
        assert entry.user_id == "u1"

    def test_inject_evidence_no_source(self):
        mgr = InterventionManager(debate_id="d1")
        entry = mgr.inject_evidence("Some fact")
        assert entry.source is None

    def test_inject_evidence_empty_raises(self):
        mgr = InterventionManager(debate_id="d1")
        with pytest.raises(ValueError, match="empty"):
            mgr.inject_evidence("")

    def test_inject_evidence_when_completed_raises(self):
        mgr = InterventionManager(debate_id="d1")
        mgr.mark_completed()
        with pytest.raises(ValueError, match="completed"):
            mgr.inject_evidence("test")


# ============================================================================
# Intervention Log
# ============================================================================


class TestInterventionLog:
    """Intervention log tracking."""

    def test_empty_log(self):
        mgr = InterventionManager(debate_id="d1")
        log = mgr.get_log()
        assert log.debate_id == "d1"
        assert len(log.entries) == 0

    def test_log_records_all_interventions(self):
        mgr = InterventionManager(debate_id="d1")
        mgr.pause(user_id="u1")
        mgr.nudge("hint", user_id="u1")
        mgr.resume(user_id="u1")
        mgr.challenge("counter", user_id="u2")
        mgr.inject_evidence("fact", source="src")

        log = mgr.get_log()
        assert len(log.entries) == 5
        types = [e.intervention_type for e in log.entries]
        assert types == [
            InterventionType.PAUSE,
            InterventionType.NUDGE,
            InterventionType.RESUME,
            InterventionType.CHALLENGE,
            InterventionType.INJECT_EVIDENCE,
        ]

    def test_log_entries_have_timestamps(self):
        mgr = InterventionManager(debate_id="d1")
        before = time.time()
        mgr.nudge("hint")
        after = time.time()

        log = mgr.get_log()
        assert len(log.entries) == 1
        assert before <= log.entries[0].timestamp <= after

    def test_log_is_a_copy(self):
        """Modifying the returned log should not affect the manager's log."""
        mgr = InterventionManager(debate_id="d1")
        mgr.nudge("a")
        log = mgr.get_log()
        log.entries.clear()
        assert len(mgr.get_log().entries) == 1

    def test_log_to_dict(self):
        mgr = InterventionManager(debate_id="d1")
        mgr.nudge("hint")
        log = mgr.get_log()
        d = log.to_dict()
        assert d["debate_id"] == "d1"
        assert d["entry_count"] == 1
        assert len(d["entries"]) == 1
        assert d["entries"][0]["type"] == "nudge"


# ============================================================================
# Thread Safety
# ============================================================================


class TestThreadSafety:
    """Thread safety of InterventionManager."""

    def test_concurrent_nudges(self):
        """Multiple threads can nudge without data corruption."""
        mgr = InterventionManager(debate_id="d1")
        errors: list[Exception] = []

        def nudge_n_times(n: int):
            try:
                for i in range(n):
                    mgr.nudge(f"nudge-{threading.current_thread().name}-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=nudge_n_times, args=(20,)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        assert len(mgr.get_log().entries) == 100

    def test_concurrent_pause_resume(self):
        """Pause/resume under contention does not corrupt state."""
        mgr = InterventionManager(debate_id="d1")
        successes = {"pause": 0, "resume": 0}
        lock = threading.Lock()

        def toggle(action: str, count: int):
            for _ in range(count):
                try:
                    if action == "pause":
                        mgr.pause()
                        with lock:
                            successes["pause"] += 1
                    else:
                        mgr.resume()
                        with lock:
                            successes["resume"] += 1
                except ValueError:
                    pass  # Expected when state doesn't match

        # Interleave pause and resume attempts
        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=toggle, args=("pause", 10)))
            threads.append(threading.Thread(target=toggle, args=("resume", 10)))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # State must be consistent: either running or paused
        assert mgr.state in (
            DebateInterventionState.RUNNING,
            DebateInterventionState.PAUSED,
        )


# ============================================================================
# WebSocket Event Emission
# ============================================================================


class TestWebSocketEmission:
    """WebSocket event emission on interventions."""

    def test_pause_emits_event(self):
        emitter = MagicMock()
        mgr = InterventionManager(debate_id="d1", emitter=emitter)
        mgr.pause()
        emitter.emit.assert_called_once()
        event = emitter.emit.call_args[0][0]
        assert event.data["event"] == "debate_paused"
        assert event.data["debate_id"] == "d1"

    def test_resume_emits_event(self):
        emitter = MagicMock()
        mgr = InterventionManager(debate_id="d1", emitter=emitter)
        mgr.pause()
        emitter.emit.reset_mock()
        mgr.resume()
        emitter.emit.assert_called_once()
        event = emitter.emit.call_args[0][0]
        assert event.data["event"] == "debate_resumed"

    def test_nudge_emits_event(self):
        emitter = MagicMock()
        mgr = InterventionManager(debate_id="d1", emitter=emitter)
        mgr.nudge("test hint")
        emitter.emit.assert_called_once()
        event = emitter.emit.call_args[0][0]
        assert event.data["event"] == "debate_nudge"

    def test_challenge_emits_event(self):
        emitter = MagicMock()
        mgr = InterventionManager(debate_id="d1", emitter=emitter)
        mgr.challenge("counter")
        emitter.emit.assert_called_once()
        event = emitter.emit.call_args[0][0]
        assert event.data["event"] == "debate_challenge"

    def test_inject_evidence_emits_event(self):
        emitter = MagicMock()
        mgr = InterventionManager(debate_id="d1", emitter=emitter)
        mgr.inject_evidence("fact", source="src")
        emitter.emit.assert_called_once()
        event = emitter.emit.call_args[0][0]
        assert event.data["event"] == "debate_evidence_injected"

    def test_no_emitter_does_not_raise(self):
        """When emitter is None, operations succeed without errors."""
        mgr = InterventionManager(debate_id="d1", emitter=None)
        mgr.pause()
        mgr.resume()
        mgr.nudge("test")
        assert len(mgr.get_log().entries) == 3


# ============================================================================
# Module-Level Registry
# ============================================================================


class TestManagerRegistry:
    """Module-level registry for active intervention managers."""

    def test_get_creates_manager(self):
        mgr = get_intervention_manager("d1")
        assert mgr is not None
        assert mgr.debate_id == "d1"

    def test_get_returns_same_instance(self):
        mgr1 = get_intervention_manager("d1")
        mgr2 = get_intervention_manager("d1")
        assert mgr1 is mgr2

    def test_get_with_create_false_returns_none(self):
        result = get_intervention_manager("missing", create=False)
        assert result is None

    def test_remove_manager(self):
        get_intervention_manager("d1")
        removed = remove_intervention_manager("d1")
        assert removed is not None
        assert get_intervention_manager("d1", create=False) is None

    def test_remove_nonexistent_returns_none(self):
        assert remove_intervention_manager("nope") is None

    def test_list_managers(self):
        get_intervention_manager("d1")
        get_intervention_manager("d2")
        managers = list_intervention_managers()
        assert "d1" in managers
        assert "d2" in managers
        assert len(managers) == 2


# ============================================================================
# Entry / Log Serialization
# ============================================================================


class TestSerialization:
    """InterventionEntry and InterventionLog serialization."""

    def test_entry_to_dict(self):
        entry = InterventionEntry(
            intervention_type=InterventionType.NUDGE,
            timestamp=1700000000.0,
            user_id="u1",
            message="hint",
            target_agent="claude",
        )
        d = entry.to_dict()
        assert d["type"] == "nudge"
        assert d["timestamp"] == 1700000000.0
        assert d["user_id"] == "u1"
        assert d["message"] == "hint"
        assert d["target_agent"] == "claude"

    def test_log_to_dict_empty(self):
        log = InterventionLog(debate_id="d1")
        d = log.to_dict()
        assert d["debate_id"] == "d1"
        assert d["entry_count"] == 0
        assert d["entries"] == []

    def test_get_state_dict(self):
        mgr = InterventionManager(debate_id="d1")
        mgr.nudge("test")
        state = mgr.get_state_dict()
        assert state["debate_id"] == "d1"
        assert state["state"] == "running"
        assert state["intervention_count"] == 1
