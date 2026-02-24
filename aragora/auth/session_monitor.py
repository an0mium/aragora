"""
Session Health Monitor for auth reliability.

Tracks session lifecycle, detects anomalies (expired, orphaned, hijacking),
and provides metrics and automated cleanup.

Usage:
    from aragora.auth.session_monitor import get_session_monitor

    monitor = get_session_monitor()

    # Track a new session
    monitor.track_session("session-id", user_id="user-1", ip="1.2.3.4")

    # Check health of a session
    health = monitor.check_session_health("session-id")

    # Get aggregate metrics
    metrics = monitor.get_metrics()

    # Sweep expired sessions
    removed = monitor.sweep_expired()
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SessionState(str, Enum):
    """Possible states for a tracked session."""

    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPICIOUS = "suspicious"


class SessionHealthStatus(str, Enum):
    """Health check result for a session."""

    HEALTHY = "healthy"
    EXPIRED = "expired"
    ORPHANED = "orphaned"
    HIJACK_SUSPECT = "hijack_suspect"
    UNKNOWN = "unknown"


@dataclass
class TrackedSession:
    """A session being monitored for health."""

    session_id: str
    user_id: str
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    expires_at: float = 0.0
    ip_address: str | None = None
    user_agent: str | None = None
    state: SessionState = SessionState.ACTIVE
    # Track IP addresses seen for this session (for hijacking detection)
    seen_ips: set[str] = field(default_factory=set)
    activity_count: int = 0

    def __post_init__(self) -> None:
        if self.expires_at == 0.0:
            # Default 8-hour session
            self.expires_at = self.created_at + 8 * 3600
        if self.ip_address:
            self.seen_ips.add(self.ip_address)

    @property
    def is_expired(self) -> bool:
        """Check if the session has expired."""
        return time.time() > self.expires_at

    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds."""
        end = min(time.time(), self.expires_at)
        return end - self.created_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "expires_at": self.expires_at,
            "ip_address": self.ip_address,
            "state": self.state.value,
            "activity_count": self.activity_count,
            "is_expired": self.is_expired,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class SessionMetrics:
    """Aggregate session metrics."""

    active_sessions: int = 0
    expired_sessions: int = 0
    revoked_sessions: int = 0
    suspicious_sessions: int = 0
    total_tracked: int = 0
    avg_session_duration: float = 0.0
    auth_failure_count: int = 0
    auth_success_count: int = 0
    hijack_attempts_detected: int = 0
    sweep_count: int = 0
    last_sweep_at: float | None = None
    last_sweep_removed: int = 0

    @property
    def auth_failure_rate(self) -> float:
        """Calculate auth failure rate as a percentage."""
        total = self.auth_success_count + self.auth_failure_count
        if total == 0:
            return 0.0
        return (self.auth_failure_count / total) * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "active_sessions": self.active_sessions,
            "expired_sessions": self.expired_sessions,
            "revoked_sessions": self.revoked_sessions,
            "suspicious_sessions": self.suspicious_sessions,
            "total_tracked": self.total_tracked,
            "avg_session_duration": round(self.avg_session_duration, 2),
            "auth_failure_count": self.auth_failure_count,
            "auth_success_count": self.auth_success_count,
            "auth_failure_rate": round(self.auth_failure_rate, 2),
            "hijack_attempts_detected": self.hijack_attempts_detected,
            "sweep_count": self.sweep_count,
            "last_sweep_at": self.last_sweep_at,
            "last_sweep_removed": self.last_sweep_removed,
        }


# Maximum sessions to track in memory (prevent unbounded growth)
_MAX_TRACKED_SESSIONS = 50_000

# Default sweep interval: 15 minutes
DEFAULT_SWEEP_INTERVAL_SECONDS = 15 * 60

# Maximum IP addresses per session before flagging as suspicious
_MAX_IPS_PER_SESSION = 5


class SessionHealthMonitor:
    """
    Monitors session health, detects anomalies, and provides metrics.

    Thread-safe implementation using a threading lock for all mutable state.

    Features:
    - Track session creation, activity, and expiration
    - Detect expired and orphaned sessions
    - Detect potential session hijacking (multiple IPs)
    - Provide aggregate metrics (active count, avg duration, failure rate)
    - Auto-cleanup of expired sessions via configurable sweep
    """

    def __init__(
        self,
        sweep_interval_seconds: float = DEFAULT_SWEEP_INTERVAL_SECONDS,
        session_ttl_seconds: float = 8 * 3600,
        max_ips_per_session: int = _MAX_IPS_PER_SESSION,
    ) -> None:
        """
        Initialize the session health monitor.

        Args:
            sweep_interval_seconds: How often auto-sweep runs (seconds).
            session_ttl_seconds: Default session lifetime (seconds).
            max_ips_per_session: Max distinct IPs before flagging suspicious.
        """
        self._sessions: dict[str, TrackedSession] = {}
        self._lock = threading.Lock()
        self._sweep_interval = sweep_interval_seconds
        self._session_ttl = session_ttl_seconds
        self._max_ips = max_ips_per_session

        # Counters for metrics
        self._auth_failures = 0
        self._auth_successes = 0
        self._hijack_detections = 0
        self._sweep_count = 0
        self._last_sweep_at: float | None = None
        self._last_sweep_removed = 0

    # =========================================================================
    # Session tracking
    # =========================================================================

    def track_session(
        self,
        session_id: str,
        user_id: str,
        ip_address: str | None = None,
        user_agent: str | None = None,
        ttl_seconds: float | None = None,
    ) -> TrackedSession:
        """
        Begin tracking a new session.

        Args:
            session_id: Unique session identifier.
            user_id: User who owns the session.
            ip_address: Client IP address.
            user_agent: Client user agent string.
            ttl_seconds: Override default TTL for this session.

        Returns:
            The tracked session object.
        """
        now = time.time()
        ttl = ttl_seconds if ttl_seconds is not None else self._session_ttl

        session = TrackedSession(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_activity=now,
            expires_at=now + ttl,
            ip_address=ip_address,
            user_agent=user_agent,
            state=SessionState.ACTIVE,
            activity_count=1,
        )

        with self._lock:
            # Evict oldest sessions if at capacity
            if len(self._sessions) >= _MAX_TRACKED_SESSIONS:
                self._evict_oldest_locked()
            self._sessions[session_id] = session

        logger.debug(
            "Session tracked: session_id=%s user_id=%s ip=%s",
            session_id,
            user_id,
            ip_address,
        )
        return session

    def record_activity(
        self,
        session_id: str,
        ip_address: str | None = None,
    ) -> SessionHealthStatus:
        """
        Record activity on a session and check for anomalies.

        Args:
            session_id: The session ID.
            ip_address: Current client IP address.

        Returns:
            Health status after recording activity.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return SessionHealthStatus.UNKNOWN

            session.last_activity = time.time()
            session.activity_count += 1

            # Check expiration
            if session.is_expired:
                session.state = SessionState.EXPIRED
                return SessionHealthStatus.EXPIRED

            # Check for IP change (potential hijacking)
            if ip_address:
                session.seen_ips.add(ip_address)
                if len(session.seen_ips) > self._max_ips:
                    session.state = SessionState.SUSPICIOUS
                    self._hijack_detections += 1
                    logger.warning(
                        "Potential session hijacking detected: session_id=%s "
                        "user_id=%s ips=%d",
                        session_id,
                        session.user_id,
                        len(session.seen_ips),
                    )
                    return SessionHealthStatus.HIJACK_SUSPECT

            return SessionHealthStatus.HEALTHY

    def revoke_session(self, session_id: str) -> bool:
        """
        Mark a session as revoked.

        Args:
            session_id: The session to revoke.

        Returns:
            True if the session was found and revoked.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            session.state = SessionState.REVOKED
            logger.info("Session revoked: session_id=%s", session_id)
            return True

    # =========================================================================
    # Health checks
    # =========================================================================

    def check_session_health(self, session_id: str) -> dict[str, Any]:
        """
        Check the health of a specific session.

        Args:
            session_id: The session to check.

        Returns:
            Dict with health status and session details.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return {
                    "session_id": session_id,
                    "status": SessionHealthStatus.UNKNOWN.value,
                    "message": "Session not found or not tracked",
                }

            status = SessionHealthStatus.HEALTHY

            if session.state == SessionState.REVOKED:
                status = SessionHealthStatus.UNKNOWN
            elif session.is_expired:
                status = SessionHealthStatus.EXPIRED
                session.state = SessionState.EXPIRED
            elif session.state == SessionState.SUSPICIOUS:
                status = SessionHealthStatus.HIJACK_SUSPECT
            elif time.time() - session.last_activity > self._session_ttl:
                status = SessionHealthStatus.ORPHANED

            return {
                "session_id": session_id,
                "status": status.value,
                "session": session.to_dict(),
            }

    def get_sessions_for_user(self, user_id: str) -> list[dict[str, Any]]:
        """
        Get all tracked sessions for a user.

        Args:
            user_id: The user ID.

        Returns:
            List of session dictionaries.
        """
        with self._lock:
            return [
                s.to_dict()
                for s in self._sessions.values()
                if s.user_id == user_id and s.state == SessionState.ACTIVE
            ]

    # =========================================================================
    # Auth event recording (for failure rate metrics)
    # =========================================================================

    def record_auth_success(self) -> None:
        """Record a successful authentication."""
        with self._lock:
            self._auth_successes += 1

    def record_auth_failure(self) -> None:
        """Record a failed authentication attempt."""
        with self._lock:
            self._auth_failures += 1

    # =========================================================================
    # Metrics
    # =========================================================================

    def get_metrics(self) -> SessionMetrics:
        """
        Get aggregate session metrics.

        Returns:
            SessionMetrics with current counts and rates.
        """
        with self._lock:
            active = 0
            expired = 0
            revoked = 0
            suspicious = 0
            total_duration = 0.0
            duration_count = 0

            for session in self._sessions.values():
                # Refresh expired status
                if session.state == SessionState.ACTIVE and session.is_expired:
                    session.state = SessionState.EXPIRED

                if session.state == SessionState.ACTIVE:
                    active += 1
                    total_duration += session.duration_seconds
                    duration_count += 1
                elif session.state == SessionState.EXPIRED:
                    expired += 1
                    total_duration += session.duration_seconds
                    duration_count += 1
                elif session.state == SessionState.REVOKED:
                    revoked += 1
                elif session.state == SessionState.SUSPICIOUS:
                    suspicious += 1
                    total_duration += session.duration_seconds
                    duration_count += 1

            avg_duration = total_duration / duration_count if duration_count > 0 else 0.0

            return SessionMetrics(
                active_sessions=active,
                expired_sessions=expired,
                revoked_sessions=revoked,
                suspicious_sessions=suspicious,
                total_tracked=len(self._sessions),
                avg_session_duration=avg_duration,
                auth_failure_count=self._auth_failures,
                auth_success_count=self._auth_successes,
                hijack_attempts_detected=self._hijack_detections,
                sweep_count=self._sweep_count,
                last_sweep_at=self._last_sweep_at,
                last_sweep_removed=self._last_sweep_removed,
            )

    # =========================================================================
    # Cleanup
    # =========================================================================

    def sweep_expired(self) -> int:
        """
        Remove expired and revoked sessions from tracking.

        Returns:
            Number of sessions removed.
        """
        now = time.time()
        removed = 0

        with self._lock:
            to_remove = []
            for sid, session in self._sessions.items():
                if session.state == SessionState.REVOKED:
                    to_remove.append(sid)
                elif session.is_expired:
                    to_remove.append(sid)

            for sid in to_remove:
                del self._sessions[sid]
                removed += 1

            self._sweep_count += 1
            self._last_sweep_at = now
            self._last_sweep_removed = removed

        if removed > 0:
            logger.info("Session sweep completed: removed=%d", removed)
        else:
            logger.debug("Session sweep completed: nothing to remove")

        return removed

    def should_sweep(self) -> bool:
        """
        Check if enough time has passed since last sweep.

        Returns:
            True if a sweep should be triggered.
        """
        if self._last_sweep_at is None:
            return True
        return time.time() - self._last_sweep_at >= self._sweep_interval

    def _evict_oldest_locked(self) -> None:
        """Evict the oldest sessions when at capacity. Must hold _lock."""
        # Sort by last_activity, remove the oldest 10%
        sorted_sessions = sorted(
            self._sessions.items(), key=lambda kv: kv[1].last_activity
        )
        evict_count = max(1, len(sorted_sessions) // 10)
        for sid, _ in sorted_sessions[:evict_count]:
            del self._sessions[sid]
        logger.debug("Evicted %d oldest sessions (capacity limit)", evict_count)

    # =========================================================================
    # Admin utilities
    # =========================================================================

    def get_suspicious_sessions(self) -> list[dict[str, Any]]:
        """
        Get all sessions flagged as suspicious.

        Returns:
            List of suspicious session dictionaries.
        """
        with self._lock:
            return [
                s.to_dict()
                for s in self._sessions.values()
                if s.state == SessionState.SUSPICIOUS
            ]

    def reset(self) -> None:
        """Reset all state (for testing)."""
        with self._lock:
            self._sessions.clear()
            self._auth_failures = 0
            self._auth_successes = 0
            self._hijack_detections = 0
            self._sweep_count = 0
            self._last_sweep_at = None
            self._last_sweep_removed = 0


# =============================================================================
# Global singleton
# =============================================================================

_session_monitor: SessionHealthMonitor | None = None
_monitor_lock = threading.Lock()


def get_session_monitor(
    sweep_interval_seconds: float = DEFAULT_SWEEP_INTERVAL_SECONDS,
    session_ttl_seconds: float = 8 * 3600,
) -> SessionHealthMonitor:
    """
    Get or create the global SessionHealthMonitor instance.

    Args:
        sweep_interval_seconds: Sweep interval (only used on first call).
        session_ttl_seconds: Session TTL (only used on first call).

    Returns:
        The global SessionHealthMonitor instance.
    """
    global _session_monitor

    if _session_monitor is not None:
        return _session_monitor

    with _monitor_lock:
        if _session_monitor is None:
            _session_monitor = SessionHealthMonitor(
                sweep_interval_seconds=sweep_interval_seconds,
                session_ttl_seconds=session_ttl_seconds,
            )

    return _session_monitor


def reset_session_monitor() -> None:
    """Reset the global session monitor (for testing)."""
    global _session_monitor
    with _monitor_lock:
        _session_monitor = None


__all__ = [
    "SessionHealthMonitor",
    "SessionHealthStatus",
    "SessionMetrics",
    "SessionState",
    "TrackedSession",
    "get_session_monitor",
    "reset_session_monitor",
    "DEFAULT_SWEEP_INTERVAL_SECONDS",
]
