"""
Tests for the Session Health Monitor.

Tests cover:
- Session tracking lifecycle
- Expired session detection
- Session sweep cleanup
- Metrics accuracy
- Hijacking detection
- Concurrent access safety
- Singleton management
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from aragora.auth.session_monitor import (
    DEFAULT_SWEEP_INTERVAL_SECONDS,
    SessionHealthMonitor,
    SessionHealthStatus,
    SessionMetrics,
    SessionState,
    TrackedSession,
    get_session_monitor,
    reset_session_monitor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_monitor():
    """Reset global singleton before and after each test."""
    reset_session_monitor()
    yield
    reset_session_monitor()


@pytest.fixture
def monitor():
    """Create a fresh monitor with short TTLs for testing."""
    return SessionHealthMonitor(
        sweep_interval_seconds=1.0,
        session_ttl_seconds=10.0,
        max_ips_per_session=3,
    )


# ===========================================================================
# TrackedSession
# ===========================================================================


class TestTrackedSession:
    """Tests for the TrackedSession dataclass."""

    def test_default_expires_at(self):
        """Test that default expiration is set to 8 hours from creation."""
        now = time.time()
        session = TrackedSession(session_id="s1", user_id="u1")
        # Should expire ~8 hours from creation
        assert session.expires_at > now
        assert session.expires_at - session.created_at == pytest.approx(8 * 3600, abs=1)

    def test_custom_expires_at(self):
        """Test that custom expiration overrides default."""
        future = time.time() + 60
        session = TrackedSession(session_id="s1", user_id="u1", expires_at=future)
        assert session.expires_at == future

    def test_is_expired_false_for_active(self):
        """Test that a fresh session is not expired."""
        session = TrackedSession(session_id="s1", user_id="u1")
        assert session.is_expired is False

    def test_is_expired_true_for_past(self):
        """Test that a session with past expiry is detected."""
        session = TrackedSession(
            session_id="s1",
            user_id="u1",
            expires_at=time.time() - 10,
        )
        assert session.is_expired is True

    def test_ip_tracked_on_init(self):
        """Test that initial IP is added to seen_ips."""
        session = TrackedSession(session_id="s1", user_id="u1", ip_address="1.2.3.4")
        assert "1.2.3.4" in session.seen_ips

    def test_to_dict(self):
        """Test dict serialization."""
        session = TrackedSession(session_id="s1", user_id="u1")
        d = session.to_dict()
        assert d["session_id"] == "s1"
        assert d["user_id"] == "u1"
        assert d["state"] == "active"
        assert "is_expired" in d
        assert "duration_seconds" in d

    def test_duration_seconds(self):
        """Test duration calculation."""
        now = time.time()
        session = TrackedSession(
            session_id="s1",
            user_id="u1",
            created_at=now - 100,
            expires_at=now + 100,
        )
        assert session.duration_seconds >= 99


# ===========================================================================
# SessionHealthMonitor - Tracking
# ===========================================================================


class TestSessionTracking:
    """Tests for session tracking lifecycle."""

    def test_track_new_session(self, monitor: SessionHealthMonitor):
        """Test tracking a new session."""
        session = monitor.track_session("s1", user_id="u1", ip_address="1.2.3.4")
        assert session.session_id == "s1"
        assert session.user_id == "u1"
        assert session.state == SessionState.ACTIVE

    def test_track_session_returns_tracked_session(self, monitor: SessionHealthMonitor):
        """Test that tracked session has correct type."""
        session = monitor.track_session("s1", user_id="u1")
        assert isinstance(session, TrackedSession)

    def test_record_activity_updates_last_activity(self, monitor: SessionHealthMonitor):
        """Test that activity recording updates timestamps."""
        monitor.track_session("s1", user_id="u1")
        time.sleep(0.01)
        status = monitor.record_activity("s1")
        assert status == SessionHealthStatus.HEALTHY

    def test_record_activity_unknown_session(self, monitor: SessionHealthMonitor):
        """Test activity on an unknown session returns UNKNOWN."""
        status = monitor.record_activity("nonexistent")
        assert status == SessionHealthStatus.UNKNOWN

    def test_revoke_session(self, monitor: SessionHealthMonitor):
        """Test revoking a tracked session."""
        monitor.track_session("s1", user_id="u1")
        result = monitor.revoke_session("s1")
        assert result is True

        # Verify state changed
        health = monitor.check_session_health("s1")
        assert health["session"]["state"] == "revoked"

    def test_revoke_nonexistent_session(self, monitor: SessionHealthMonitor):
        """Test revoking a session that doesn't exist."""
        result = monitor.revoke_session("nonexistent")
        assert result is False


# ===========================================================================
# SessionHealthMonitor - Expired Detection
# ===========================================================================


class TestExpiredDetection:
    """Tests for expired session detection."""

    def test_check_expired_session(self, monitor: SessionHealthMonitor):
        """Test that expired sessions are detected in health check."""
        monitor.track_session("s1", user_id="u1", ttl_seconds=0.01)
        time.sleep(0.02)

        health = monitor.check_session_health("s1")
        assert health["status"] == "expired"

    def test_record_activity_detects_expired(self, monitor: SessionHealthMonitor):
        """Test that recording activity on expired session returns EXPIRED."""
        monitor.track_session("s1", user_id="u1", ttl_seconds=0.01)
        time.sleep(0.02)

        status = monitor.record_activity("s1")
        assert status == SessionHealthStatus.EXPIRED

    def test_unknown_session_health(self, monitor: SessionHealthMonitor):
        """Test health check for non-tracked session."""
        health = monitor.check_session_health("nonexistent")
        assert health["status"] == "unknown"


# ===========================================================================
# SessionHealthMonitor - Sweep
# ===========================================================================


class TestSweepCleanup:
    """Tests for the sweep_expired cleanup."""

    def test_sweep_removes_expired(self, monitor: SessionHealthMonitor):
        """Test that sweep removes expired sessions."""
        monitor.track_session("s1", user_id="u1", ttl_seconds=0.01)
        monitor.track_session("s2", user_id="u1", ttl_seconds=100)
        time.sleep(0.02)

        removed = monitor.sweep_expired()
        assert removed == 1

        # s2 should still be tracked
        health = monitor.check_session_health("s2")
        assert health["status"] == "healthy"

        # s1 should be gone
        health = monitor.check_session_health("s1")
        assert health["status"] == "unknown"

    def test_sweep_removes_revoked(self, monitor: SessionHealthMonitor):
        """Test that sweep removes revoked sessions."""
        monitor.track_session("s1", user_id="u1")
        monitor.revoke_session("s1")

        removed = monitor.sweep_expired()
        assert removed == 1

    def test_sweep_empty(self, monitor: SessionHealthMonitor):
        """Test sweep with no sessions to remove."""
        monitor.track_session("s1", user_id="u1")
        removed = monitor.sweep_expired()
        assert removed == 0

    def test_sweep_updates_metrics(self, monitor: SessionHealthMonitor):
        """Test that sweep updates sweep count metrics."""
        monitor.sweep_expired()
        metrics = monitor.get_metrics()
        assert metrics.sweep_count == 1
        assert metrics.last_sweep_at is not None

    def test_should_sweep_initially(self, monitor: SessionHealthMonitor):
        """Test that should_sweep returns True initially."""
        assert monitor.should_sweep() is True

    def test_should_sweep_after_sweep(self, monitor: SessionHealthMonitor):
        """Test that should_sweep returns False right after sweep."""
        monitor.sweep_expired()
        assert monitor.should_sweep() is False


# ===========================================================================
# SessionHealthMonitor - Metrics
# ===========================================================================


class TestMetrics:
    """Tests for session metrics accuracy."""

    def test_metrics_empty(self, monitor: SessionHealthMonitor):
        """Test metrics with no sessions."""
        metrics = monitor.get_metrics()
        assert metrics.active_sessions == 0
        assert metrics.total_tracked == 0
        assert metrics.auth_failure_rate == 0.0

    def test_metrics_active_count(self, monitor: SessionHealthMonitor):
        """Test active session count in metrics."""
        monitor.track_session("s1", user_id="u1")
        monitor.track_session("s2", user_id="u2")

        metrics = monitor.get_metrics()
        assert metrics.active_sessions == 2
        assert metrics.total_tracked == 2

    def test_metrics_auth_success(self, monitor: SessionHealthMonitor):
        """Test auth success counter."""
        monitor.record_auth_success()
        monitor.record_auth_success()
        monitor.record_auth_success()

        metrics = monitor.get_metrics()
        assert metrics.auth_success_count == 3

    def test_metrics_auth_failure(self, monitor: SessionHealthMonitor):
        """Test auth failure counter."""
        monitor.record_auth_failure()
        monitor.record_auth_failure()

        metrics = monitor.get_metrics()
        assert metrics.auth_failure_count == 2

    def test_metrics_failure_rate(self, monitor: SessionHealthMonitor):
        """Test auth failure rate calculation."""
        for _ in range(7):
            monitor.record_auth_success()
        for _ in range(3):
            monitor.record_auth_failure()

        metrics = monitor.get_metrics()
        assert metrics.auth_failure_rate == pytest.approx(30.0, abs=0.1)

    def test_metrics_to_dict(self, monitor: SessionHealthMonitor):
        """Test metrics dict serialization."""
        metrics = monitor.get_metrics()
        d = metrics.to_dict()
        assert "active_sessions" in d
        assert "auth_failure_rate" in d
        assert "hijack_attempts_detected" in d

    def test_metrics_avg_duration(self, monitor: SessionHealthMonitor):
        """Test average session duration metric."""
        now = time.time()
        # Session started 100 seconds ago
        session = monitor.track_session("s1", user_id="u1")
        # Override created_at for test
        session.created_at = now - 100
        session.expires_at = now + 100

        metrics = monitor.get_metrics()
        assert metrics.avg_session_duration >= 99.0


# ===========================================================================
# SessionHealthMonitor - Hijacking Detection
# ===========================================================================


class TestHijackingDetection:
    """Tests for session hijacking detection."""

    def test_multiple_ips_triggers_suspicious(self, monitor: SessionHealthMonitor):
        """Test that too many IPs flags session as suspicious."""
        monitor.track_session("s1", user_id="u1", ip_address="1.1.1.1")
        monitor.record_activity("s1", ip_address="2.2.2.2")
        monitor.record_activity("s1", ip_address="3.3.3.3")

        # 4th IP should trigger (max_ips=3)
        status = monitor.record_activity("s1", ip_address="4.4.4.4")
        assert status == SessionHealthStatus.HIJACK_SUSPECT

    def test_same_ip_no_hijack(self, monitor: SessionHealthMonitor):
        """Test that repeated same IP does not trigger hijacking."""
        monitor.track_session("s1", user_id="u1", ip_address="1.1.1.1")
        for _ in range(10):
            status = monitor.record_activity("s1", ip_address="1.1.1.1")
        assert status == SessionHealthStatus.HEALTHY

    def test_hijack_detection_increments_metric(self, monitor: SessionHealthMonitor):
        """Test that hijack detection increments the metric."""
        monitor.track_session("s1", user_id="u1", ip_address="1.1.1.1")
        monitor.record_activity("s1", ip_address="2.2.2.2")
        monitor.record_activity("s1", ip_address="3.3.3.3")
        monitor.record_activity("s1", ip_address="4.4.4.4")

        metrics = monitor.get_metrics()
        assert metrics.hijack_attempts_detected >= 1

    def test_get_suspicious_sessions(self, monitor: SessionHealthMonitor):
        """Test retrieval of suspicious sessions."""
        monitor.track_session("s1", user_id="u1", ip_address="1.1.1.1")
        monitor.record_activity("s1", ip_address="2.2.2.2")
        monitor.record_activity("s1", ip_address="3.3.3.3")
        monitor.record_activity("s1", ip_address="4.4.4.4")

        suspicious = monitor.get_suspicious_sessions()
        assert len(suspicious) == 1
        assert suspicious[0]["session_id"] == "s1"


# ===========================================================================
# SessionHealthMonitor - Concurrent Access
# ===========================================================================


class TestConcurrentAccess:
    """Tests for thread safety under concurrent access."""

    def test_concurrent_track(self, monitor: SessionHealthMonitor):
        """Test concurrent session tracking."""
        errors: list[Exception] = []

        def track_batch(start: int):
            try:
                for i in range(50):
                    monitor.track_session(f"s-{start + i}", user_id=f"u-{start + i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=track_batch, args=(i * 50,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        metrics = monitor.get_metrics()
        assert metrics.total_tracked == 200

    def test_concurrent_record_activity(self, monitor: SessionHealthMonitor):
        """Test concurrent activity recording."""
        monitor.track_session("s1", user_id="u1", ip_address="1.1.1.1")
        errors: list[Exception] = []

        def record_batch():
            try:
                for _ in range(50):
                    monitor.record_activity("s1", ip_address="1.1.1.1")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_batch) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_sweep(self, monitor: SessionHealthMonitor):
        """Test concurrent sweep operations."""
        for i in range(20):
            monitor.track_session(f"s-{i}", user_id="u1", ttl_seconds=0.01)
        time.sleep(0.02)

        errors: list[Exception] = []

        def sweep():
            try:
                monitor.sweep_expired()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=sweep) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ===========================================================================
# Singleton / Global Access
# ===========================================================================


class TestSingleton:
    """Tests for get/reset_session_monitor singleton."""

    def test_get_session_monitor_returns_same_instance(self):
        """Test that get_session_monitor returns the same instance."""
        m1 = get_session_monitor()
        m2 = get_session_monitor()
        assert m1 is m2

    def test_reset_clears_instance(self):
        """Test that reset allows creating a new instance."""
        m1 = get_session_monitor()
        reset_session_monitor()
        m2 = get_session_monitor()
        assert m1 is not m2

    def test_monitor_reset_clears_state(self):
        """Test that calling reset on a monitor clears all state."""
        monitor = SessionHealthMonitor()
        monitor.track_session("s1", user_id="u1")
        monitor.record_auth_failure()
        monitor.reset()

        metrics = monitor.get_metrics()
        assert metrics.total_tracked == 0
        assert metrics.auth_failure_count == 0


# ===========================================================================
# SessionMetrics
# ===========================================================================


class TestSessionMetrics:
    """Tests for SessionMetrics dataclass."""

    def test_failure_rate_zero_when_no_attempts(self):
        """Test failure rate is 0 when there are no auth attempts."""
        m = SessionMetrics()
        assert m.auth_failure_rate == 0.0

    def test_failure_rate_calculation(self):
        """Test failure rate calculation."""
        m = SessionMetrics(auth_failure_count=25, auth_success_count=75)
        assert m.auth_failure_rate == 25.0

    def test_failure_rate_all_failures(self):
        """Test failure rate when all are failures."""
        m = SessionMetrics(auth_failure_count=10, auth_success_count=0)
        assert m.auth_failure_rate == 100.0


# ===========================================================================
# User-scoped session listing
# ===========================================================================


class TestUserSessions:
    """Tests for get_sessions_for_user."""

    def test_get_sessions_for_user(self, monitor: SessionHealthMonitor):
        """Test listing sessions for a specific user."""
        monitor.track_session("s1", user_id="u1")
        monitor.track_session("s2", user_id="u1")
        monitor.track_session("s3", user_id="u2")

        sessions = monitor.get_sessions_for_user("u1")
        assert len(sessions) == 2
        assert all(s["user_id"] == "u1" for s in sessions)

    def test_get_sessions_excludes_revoked(self, monitor: SessionHealthMonitor):
        """Test that revoked sessions are excluded from user listing."""
        monitor.track_session("s1", user_id="u1")
        monitor.track_session("s2", user_id="u1")
        monitor.revoke_session("s2")

        sessions = monitor.get_sessions_for_user("u1")
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "s1"

    def test_get_sessions_empty(self, monitor: SessionHealthMonitor):
        """Test listing sessions for a user with none."""
        sessions = monitor.get_sessions_for_user("nobody")
        assert sessions == []
