"""Tests for WorktreeWatchdog - stall detection and recovery."""

import os
import signal
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.nomic.worktree_watchdog import (
    HealthReport,
    WatchdogConfig,
    WorktreeSession,
    WorktreeWatchdog,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def watchdog(tmp_path):
    """Create a watchdog with short timeouts for testing."""
    config = WatchdogConfig(
        stall_timeout_seconds=1.0,
        abandon_timeout_seconds=3.0,
        auto_kill_stalled=True,
        auto_cleanup_abandoned=True,
        emit_events=False,  # Disable event bus in tests
    )
    return WorktreeWatchdog(repo_path=tmp_path, config=config)


@pytest.fixture
def watchdog_no_kill(tmp_path):
    """Create a watchdog with auto-kill disabled."""
    config = WatchdogConfig(
        stall_timeout_seconds=1.0,
        abandon_timeout_seconds=3.0,
        auto_kill_stalled=False,
        auto_cleanup_abandoned=False,
        emit_events=False,
    )
    return WorktreeWatchdog(repo_path=tmp_path, config=config)


@pytest.fixture
def worktree_path(tmp_path):
    """Create a mock worktree directory."""
    wt = tmp_path / ".worktrees" / "dev-sme-1"
    wt.mkdir(parents=True)
    return wt


# =============================================================================
# Session Registration Tests
# =============================================================================


class TestSessionRegistration:
    """Tests for session registration."""

    def test_register_session_returns_unique_id(self, watchdog, worktree_path):
        """Each registration should return a unique session ID."""
        id1 = watchdog.register_session(
            branch_name="dev/sme-1",
            worktree_path=worktree_path,
            track="sme",
        )
        id2 = watchdog.register_session(
            branch_name="dev/qa-1",
            worktree_path=worktree_path,
            track="qa",
        )

        assert id1 != id2
        assert id1.startswith("wt-")
        assert id2.startswith("wt-")

    def test_register_session_sets_active_status(self, watchdog, worktree_path):
        """Newly registered sessions should be active."""
        session_id = watchdog.register_session(
            branch_name="dev/sme-1",
            worktree_path=worktree_path,
            track="sme",
        )

        session = watchdog.get_session(session_id)
        assert session is not None
        assert session.status == "active"
        assert session.branch_name == "dev/sme-1"
        assert session.track == "sme"

    def test_register_session_default_pid(self, watchdog, worktree_path):
        """Should use current PID when none specified."""
        session_id = watchdog.register_session(
            branch_name="dev/core-1",
            worktree_path=worktree_path,
            track="core",
        )

        session = watchdog.get_session(session_id)
        assert session.pid == os.getpid()

    def test_register_session_custom_pid(self, watchdog, worktree_path):
        """Should accept custom PID."""
        session_id = watchdog.register_session(
            branch_name="dev/qa-1",
            worktree_path=worktree_path,
            track="qa",
            pid=99999,
        )

        session = watchdog.get_session(session_id)
        assert session.pid == 99999

    def test_list_sessions(self, watchdog, worktree_path):
        """Should list all registered sessions."""
        watchdog.register_session(
            branch_name="dev/sme-1",
            worktree_path=worktree_path,
            track="sme",
        )
        watchdog.register_session(
            branch_name="dev/qa-1",
            worktree_path=worktree_path,
            track="qa",
        )

        sessions = watchdog.list_sessions()
        assert len(sessions) == 2
        tracks = {s.track for s in sessions}
        assert tracks == {"sme", "qa"}


# =============================================================================
# Heartbeat Tests
# =============================================================================


class TestHeartbeat:
    """Tests for heartbeat recording."""

    def test_heartbeat_increments_count(self, watchdog, worktree_path):
        """Heartbeat should increment the counter."""
        session_id = watchdog.register_session(
            branch_name="dev/sme-1",
            worktree_path=worktree_path,
            track="sme",
        )

        assert watchdog.heartbeat(session_id) is True
        assert watchdog.heartbeat(session_id) is True

        session = watchdog.get_session(session_id)
        assert session.heartbeat_count == 2

    def test_heartbeat_unknown_session(self, watchdog):
        """Heartbeat for unknown session should return False."""
        assert watchdog.heartbeat("nonexistent-id") is False

    def test_heartbeat_revives_stalled_session(self, watchdog, worktree_path):
        """Heartbeat should revive a stalled session."""
        session_id = watchdog.register_session(
            branch_name="dev/sme-1",
            worktree_path=worktree_path,
            track="sme",
        )

        # Manually set stalled status
        session = watchdog.get_session(session_id)
        session.status = "stalled"

        # Heartbeat should revive it
        assert watchdog.heartbeat(session_id) is True

        session = watchdog.get_session(session_id)
        assert session.status == "active"

    def test_heartbeat_updates_timestamp(self, watchdog, worktree_path):
        """Heartbeat should update the last_heartbeat timestamp."""
        session_id = watchdog.register_session(
            branch_name="dev/sme-1",
            worktree_path=worktree_path,
            track="sme",
        )

        session = watchdog.get_session(session_id)
        first_hb = session.last_heartbeat

        time.sleep(0.01)  # Small delay to ensure different timestamp
        watchdog.heartbeat(session_id)

        session = watchdog.get_session(session_id)
        assert session.last_heartbeat > first_hb


# =============================================================================
# Session Completion Tests
# =============================================================================


class TestSessionCompletion:
    """Tests for session completion."""

    def test_complete_session(self, watchdog, worktree_path):
        """Should mark session as completed."""
        session_id = watchdog.register_session(
            branch_name="dev/sme-1",
            worktree_path=worktree_path,
            track="sme",
        )

        assert watchdog.complete_session(session_id) is True

        session = watchdog.get_session(session_id)
        assert session.status == "completed"

    def test_complete_unknown_session(self, watchdog):
        """Should return False for unknown session."""
        assert watchdog.complete_session("nonexistent-id") is False

    def test_unregister_session(self, watchdog, worktree_path):
        """Should remove session from tracking."""
        session_id = watchdog.register_session(
            branch_name="dev/sme-1",
            worktree_path=worktree_path,
            track="sme",
        )

        assert watchdog.unregister_session(session_id) is True
        assert watchdog.get_session(session_id) is None
        assert watchdog.unregister_session(session_id) is False


# =============================================================================
# Stall Detection Tests
# =============================================================================


class TestStallDetection:
    """Tests for stall detection via check_health."""

    def test_active_session_stays_active(self, watchdog, worktree_path):
        """Session with recent heartbeat should remain active."""
        watchdog.register_session(
            branch_name="dev/sme-1",
            worktree_path=worktree_path,
            track="sme",
        )

        report = watchdog.check_health()
        assert report.active_sessions == 1
        assert report.stalled_sessions == 0

    def test_stall_detection_after_timeout(self, watchdog, worktree_path):
        """Session without heartbeat should be marked stalled after timeout."""
        session_id = watchdog.register_session(
            branch_name="dev/sme-1",
            worktree_path=worktree_path,
            track="sme",
        )

        # Push heartbeat into the past (beyond 1s stall timeout)
        session = watchdog.get_session(session_id)
        session.last_heartbeat = time.monotonic() - 2.0

        report = watchdog.check_health()
        assert report.stalled_sessions == 1
        assert report.active_sessions == 0

    def test_abandon_detection_after_timeout(self, watchdog, worktree_path):
        """Session without heartbeat should be marked abandoned after long timeout."""
        session_id = watchdog.register_session(
            branch_name="dev/sme-1",
            worktree_path=worktree_path,
            track="sme",
        )

        # Push heartbeat into the past (beyond 3s abandon timeout)
        session = watchdog.get_session(session_id)
        session.last_heartbeat = time.monotonic() - 5.0

        report = watchdog.check_health()
        assert report.abandoned_sessions == 1
        assert report.stalled_sessions == 0  # Abandoned supersedes stalled

    def test_completed_session_not_marked_stalled(self, watchdog, worktree_path):
        """Completed sessions should not be reclassified."""
        session_id = watchdog.register_session(
            branch_name="dev/sme-1",
            worktree_path=worktree_path,
            track="sme",
        )

        watchdog.complete_session(session_id)

        # Push heartbeat into past
        session = watchdog.get_session(session_id)
        session.last_heartbeat = time.monotonic() - 10.0

        report = watchdog.check_health()
        assert report.completed_sessions == 1
        assert report.stalled_sessions == 0
        assert report.abandoned_sessions == 0

    def test_health_report_structure(self, watchdog, worktree_path):
        """Health report should have correct structure."""
        watchdog.register_session(
            branch_name="dev/sme-1",
            worktree_path=worktree_path,
            track="sme",
        )

        report = watchdog.check_health()

        assert isinstance(report, HealthReport)
        assert report.total_sessions == 1
        assert report.timestamp  # ISO format string
        assert len(report.sessions) == 1
        assert "session_id" in report.sessions[0]
        assert "branch_name" in report.sessions[0]
        assert "status" in report.sessions[0]
        assert "last_heartbeat_ago_seconds" in report.sessions[0]


# =============================================================================
# Auto-Recovery Tests
# =============================================================================


class TestAutoRecovery:
    """Tests for stall recovery behavior."""

    def test_recover_stalled_dead_process(self, watchdog, worktree_path):
        """Should recover session when process is already dead."""
        session_id = watchdog.register_session(
            branch_name="dev/sme-1",
            worktree_path=worktree_path,
            track="sme",
            pid=99999,  # Unlikely to be a real PID
        )

        # Push heartbeat into the past
        session = watchdog.get_session(session_id)
        session.last_heartbeat = time.monotonic() - 2.0

        with patch.object(watchdog, "_is_process_alive", return_value=False):
            recovered = watchdog.recover_stalled()

        assert session_id in recovered
        session = watchdog.get_session(session_id)
        assert session.status == "recovered"

    def test_recover_stalled_kills_alive_process(self, watchdog, worktree_path):
        """Should send kill signal to alive stalled process."""
        session_id = watchdog.register_session(
            branch_name="dev/sme-1",
            worktree_path=worktree_path,
            track="sme",
            pid=99999,
        )

        session = watchdog.get_session(session_id)
        session.last_heartbeat = time.monotonic() - 2.0

        with patch.object(watchdog, "_is_process_alive", return_value=True), \
             patch("os.kill") as mock_kill:
            recovered = watchdog.recover_stalled()

        assert session_id in recovered
        mock_kill.assert_called_once_with(99999, signal.SIGTERM)

    def test_recover_respects_auto_kill_disabled(self, watchdog_no_kill, worktree_path):
        """Should not kill when auto_kill_stalled is disabled."""
        session_id = watchdog_no_kill.register_session(
            branch_name="dev/sme-1",
            worktree_path=worktree_path,
            track="sme",
            pid=99999,
        )

        session = watchdog_no_kill.get_session(session_id)
        session.last_heartbeat = time.monotonic() - 2.0

        with patch("os.kill") as mock_kill:
            recovered = watchdog_no_kill.recover_stalled()

        assert len(recovered) == 0
        mock_kill.assert_not_called()

    def test_recover_handles_no_pid(self, watchdog, worktree_path):
        """Should handle session with no PID gracefully."""
        session_id = watchdog.register_session(
            branch_name="dev/sme-1",
            worktree_path=worktree_path,
            track="sme",
        )

        session = watchdog.get_session(session_id)
        session.pid = None
        session.last_heartbeat = time.monotonic() - 2.0

        recovered = watchdog.recover_stalled()
        assert len(recovered) == 0


# =============================================================================
# Abandoned Worktree Cleanup Tests
# =============================================================================


class TestCleanupAbandoned:
    """Tests for abandoned worktree cleanup."""

    def test_cleanup_abandoned_removes_worktree(self, watchdog, worktree_path):
        """Should remove abandoned worktree via git."""
        session_id = watchdog.register_session(
            branch_name="dev/sme-1",
            worktree_path=worktree_path,
            track="sme",
            pid=99999,
        )

        session = watchdog.get_session(session_id)
        session.last_heartbeat = time.monotonic() - 5.0  # Beyond abandon timeout

        with patch.object(watchdog, "_is_process_alive", return_value=False), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            cleaned = watchdog.cleanup_abandoned()

        assert session_id in cleaned
        # Should have called git worktree remove
        calls = mock_run.call_args_list
        assert any("worktree" in str(c) and "remove" in str(c) for c in calls)

    def test_cleanup_skips_alive_process(self, watchdog, worktree_path):
        """Should not clean up worktree if process is still alive."""
        session_id = watchdog.register_session(
            branch_name="dev/sme-1",
            worktree_path=worktree_path,
            track="sme",
            pid=99999,
        )

        session = watchdog.get_session(session_id)
        session.last_heartbeat = time.monotonic() - 5.0

        with patch.object(watchdog, "_is_process_alive", return_value=True):
            cleaned = watchdog.cleanup_abandoned()

        assert len(cleaned) == 0

    def test_cleanup_respects_auto_cleanup_disabled(self, watchdog_no_kill, worktree_path):
        """Should not clean up when auto_cleanup_abandoned is disabled."""
        session_id = watchdog_no_kill.register_session(
            branch_name="dev/sme-1",
            worktree_path=worktree_path,
            track="sme",
            pid=99999,
        )

        session = watchdog_no_kill.get_session(session_id)
        session.last_heartbeat = time.monotonic() - 5.0

        with patch.object(watchdog_no_kill, "_is_process_alive", return_value=False):
            cleaned = watchdog_no_kill.cleanup_abandoned()

        assert len(cleaned) == 0

    def test_cleanup_handles_missing_worktree_dir(self, watchdog, tmp_path):
        """Should handle case where worktree directory is already gone."""
        missing_path = tmp_path / ".worktrees" / "already-gone"
        # Do NOT create the directory

        session_id = watchdog.register_session(
            branch_name="dev/sme-1",
            worktree_path=missing_path,
            track="sme",
            pid=99999,
        )

        session = watchdog.get_session(session_id)
        session.last_heartbeat = time.monotonic() - 5.0

        with patch.object(watchdog, "_is_process_alive", return_value=False), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            cleaned = watchdog.cleanup_abandoned()

        # Session should be cleaned (removed from tracking) even if dir missing
        assert session_id in cleaned


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for concurrent session handling."""

    def test_concurrent_registrations(self, watchdog, worktree_path):
        """Multiple threads should be able to register sessions concurrently."""
        session_ids: list[str] = []
        lock = threading.Lock()

        def register_session(track_name):
            sid = watchdog.register_session(
                branch_name=f"dev/{track_name}-1",
                worktree_path=worktree_path,
                track=track_name,
            )
            with lock:
                session_ids.append(sid)

        threads = [
            threading.Thread(target=register_session, args=(f"track-{i}",))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(session_ids) == 10
        assert len(set(session_ids)) == 10  # All unique

    def test_concurrent_heartbeats(self, watchdog, worktree_path):
        """Multiple threads should be able to heartbeat concurrently."""
        session_id = watchdog.register_session(
            branch_name="dev/sme-1",
            worktree_path=worktree_path,
            track="sme",
        )

        def heartbeat_loop():
            for _ in range(50):
                watchdog.heartbeat(session_id)

        threads = [threading.Thread(target=heartbeat_loop) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        session = watchdog.get_session(session_id)
        assert session.heartbeat_count == 250  # 5 threads * 50 heartbeats

    def test_concurrent_health_checks(self, watchdog, worktree_path):
        """Health checks should be safe under concurrent access."""
        for i in range(5):
            watchdog.register_session(
                branch_name=f"dev/track-{i}",
                worktree_path=worktree_path,
                track=f"track-{i}",
            )

        reports: list[HealthReport] = []
        lock = threading.Lock()

        def check():
            report = watchdog.check_health()
            with lock:
                reports.append(report)

        threads = [threading.Thread(target=check) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(reports) == 10
        for report in reports:
            assert report.total_sessions == 5


# =============================================================================
# Event Emission Tests
# =============================================================================


class TestEventEmission:
    """Tests for event bus integration."""

    def test_event_emission_on_register(self, tmp_path, worktree_path):
        """Should emit task_claimed event on registration."""
        config = WatchdogConfig(emit_events=True)
        watchdog = WorktreeWatchdog(repo_path=tmp_path, config=config)

        mock_bus = MagicMock()
        watchdog._event_bus = mock_bus

        watchdog.register_session(
            branch_name="dev/sme-1",
            worktree_path=worktree_path,
            track="sme",
        )

        mock_bus.publish.assert_called_once()
        call_kwargs = mock_bus.publish.call_args
        assert call_kwargs[1]["event_type"] == "task_claimed"
        assert call_kwargs[1]["track"] == "sme"

    def test_event_emission_on_complete(self, tmp_path, worktree_path):
        """Should emit task_completed event on session completion."""
        config = WatchdogConfig(emit_events=True)
        watchdog = WorktreeWatchdog(repo_path=tmp_path, config=config)

        mock_bus = MagicMock()
        watchdog._event_bus = mock_bus

        session_id = watchdog.register_session(
            branch_name="dev/sme-1",
            worktree_path=worktree_path,
            track="sme",
        )

        mock_bus.publish.reset_mock()
        watchdog.complete_session(session_id)

        mock_bus.publish.assert_called_once()
        call_kwargs = mock_bus.publish.call_args
        assert call_kwargs[1]["event_type"] == "task_completed"

    def test_no_event_when_disabled(self, tmp_path, worktree_path):
        """Should not emit events when emit_events is False."""
        config = WatchdogConfig(emit_events=False)
        watchdog = WorktreeWatchdog(repo_path=tmp_path, config=config)

        # _event_bus should remain None
        watchdog.register_session(
            branch_name="dev/sme-1",
            worktree_path=worktree_path,
            track="sme",
        )

        assert watchdog._event_bus is None


# =============================================================================
# Process Alive Detection Tests
# =============================================================================


class TestProcessAlive:
    """Tests for process liveness detection."""

    def test_current_process_is_alive(self, watchdog):
        """Current process should be detected as alive."""
        assert watchdog._is_process_alive(os.getpid()) is True

    def test_nonexistent_pid_is_not_alive(self, watchdog):
        """Non-existent PID should be detected as not alive."""
        # Use a very high PID unlikely to exist
        assert watchdog._is_process_alive(4194304) is False
