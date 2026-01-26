"""
JWT Session Management.

Tracks active JWT sessions per user with activity timestamps, device info,
and provides methods to list, track, and revoke sessions.

This complements the token blacklist by providing session-level management:
- List all active sessions for a user
- Track last activity time per session
- Store session metadata (device, IP, user agent)
- Revoke individual sessions or all sessions

Usage:
    from aragora.billing.auth.sessions import get_session_manager

    manager = get_session_manager()

    # Track a new session on login
    session = manager.create_session(
        user_id="user-123",
        token_jti="unique-token-id",
        ip_address="192.168.1.1",
        user_agent="Mozilla/5.0...",
    )

    # Update activity on each request
    manager.touch_session("user-123", "unique-token-id")

    # List user's active sessions
    sessions = manager.list_sessions("user-123")

    # Revoke a specific session
    manager.revoke_session("user-123", "unique-token-id")

    # Revoke all sessions (logout everywhere)
    manager.revoke_all_sessions("user-123")
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Configuration via environment
_SESSION_TTL = int(os.getenv("ARAGORA_JWT_SESSION_TTL", "2592000"))  # 30 days default
_MAX_SESSIONS_PER_USER = int(os.getenv("ARAGORA_MAX_SESSIONS_PER_USER", "10"))
_INACTIVITY_TIMEOUT = int(os.getenv("ARAGORA_SESSION_INACTIVITY_TIMEOUT", "86400"))  # 24 hours


@dataclass
class JWTSession:
    """Represents an active JWT session."""

    session_id: str  # Unique identifier (typically JTI from JWT)
    user_id: str
    created_at: float  # Unix timestamp
    last_activity: float  # Unix timestamp
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    device_name: Optional[str] = None  # Derived from user agent
    expires_at: Optional[float] = None  # Token expiration time

    # MFA tracking for step-up authentication
    mfa_verified_at: Optional[float] = None  # Unix timestamp of last MFA verification
    mfa_methods_used: Optional[List[str]] = None  # ['totp', 'backup_code', etc.]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": datetime.fromtimestamp(self.created_at, timezone.utc).isoformat(),
            "last_activity": datetime.fromtimestamp(self.last_activity, timezone.utc).isoformat(),
            "ip_address": self.ip_address,
            "device_name": self.device_name or _parse_device_name(self.user_agent),
            "is_current": False,  # Set by caller if needed
            "expires_at": (
                datetime.fromtimestamp(self.expires_at, timezone.utc).isoformat()
                if self.expires_at
                else None
            ),
            "mfa_verified_at": (
                datetime.fromtimestamp(self.mfa_verified_at, timezone.utc).isoformat()
                if self.mfa_verified_at
                else None
            ),
            "mfa_methods_used": self.mfa_methods_used or [],
        }

    def is_expired(self) -> bool:
        """Check if session has expired."""
        if self.expires_at and time.time() > self.expires_at:
            return True
        return False

    def is_inactive(self, timeout: int = _INACTIVITY_TIMEOUT) -> bool:
        """Check if session is inactive beyond timeout."""
        return time.time() - self.last_activity > timeout

    def is_mfa_fresh(self, max_age_seconds: int = 900) -> bool:
        """
        Check if MFA verification is fresh (within max_age_seconds).

        Args:
            max_age_seconds: Maximum age of MFA verification (default: 15 minutes)

        Returns:
            True if MFA was verified within max_age_seconds, False otherwise
        """
        if self.mfa_verified_at is None:
            return False
        return (time.time() - self.mfa_verified_at) <= max_age_seconds

    def mfa_age_seconds(self) -> Optional[int]:
        """Get seconds since last MFA verification, or None if never verified."""
        if self.mfa_verified_at is None:
            return None
        return int(time.time() - self.mfa_verified_at)

    def record_mfa_verification(self, methods: Optional[List[str]] = None) -> None:
        """
        Record that MFA was successfully verified for this session.

        Args:
            methods: List of MFA methods used (e.g., ['totp'], ['backup_code'])
        """
        self.mfa_verified_at = time.time()
        self.mfa_methods_used = methods or ["totp"]


def _parse_device_name(user_agent: Optional[str]) -> str:
    """Parse a human-readable device name from user agent string."""
    if not user_agent:
        return "Unknown Device"

    ua_lower = user_agent.lower()

    # Mobile devices
    if "iphone" in ua_lower:
        return "iPhone"
    if "ipad" in ua_lower:
        return "iPad"
    if "android" in ua_lower:
        if "mobile" in ua_lower:
            return "Android Phone"
        return "Android Tablet"

    # Desktop browsers
    if "macintosh" in ua_lower or "mac os" in ua_lower:
        if "chrome" in ua_lower:
            return "Chrome on Mac"
        if "safari" in ua_lower:
            return "Safari on Mac"
        if "firefox" in ua_lower:
            return "Firefox on Mac"
        return "Mac"

    if "windows" in ua_lower:
        # Check Edge before Chrome since Edge UA contains "Chrome"
        if "edg/" in ua_lower or "edge/" in ua_lower:
            return "Edge on Windows"
        if "chrome" in ua_lower:
            return "Chrome on Windows"
        if "firefox" in ua_lower:
            return "Firefox on Windows"
        return "Windows"

    if "linux" in ua_lower:
        if "chrome" in ua_lower:
            return "Chrome on Linux"
        if "firefox" in ua_lower:
            return "Firefox on Linux"
        return "Linux"

    # CLI/API clients
    if "curl" in ua_lower:
        return "cURL"
    if "python" in ua_lower:
        return "Python Client"
    if "postman" in ua_lower:
        return "Postman"

    return "Unknown Device"


class JWTSessionManager:
    """
    Manages active JWT sessions with activity tracking.

    Thread-safe in-memory implementation with LRU eviction.
    For distributed deployments, use Redis-backed implementation.
    """

    def __init__(
        self,
        session_ttl: int = _SESSION_TTL,
        max_sessions_per_user: int = _MAX_SESSIONS_PER_USER,
        inactivity_timeout: int = _INACTIVITY_TIMEOUT,
    ):
        """Initialize session manager.

        Args:
            session_ttl: Maximum session lifetime in seconds
            max_sessions_per_user: Maximum concurrent sessions per user
            inactivity_timeout: Inactivity timeout in seconds
        """
        self.session_ttl = session_ttl
        self.max_sessions_per_user = max_sessions_per_user
        self.inactivity_timeout = inactivity_timeout

        # user_id -> OrderedDict of session_id -> JWTSession
        self._sessions: Dict[str, OrderedDict[str, JWTSession]] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes

    def create_session(
        self,
        user_id: str,
        token_jti: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        expires_at: Optional[float] = None,
    ) -> JWTSession:
        """Create and track a new session.

        Args:
            user_id: User identifier
            token_jti: JWT token identifier (JTI claim)
            ip_address: Client IP address
            user_agent: Client user agent string
            expires_at: Token expiration timestamp

        Returns:
            Created JWTSession instance
        """
        now = time.time()

        session = JWTSession(
            session_id=token_jti,
            user_id=user_id,
            created_at=now,
            last_activity=now,
            ip_address=ip_address,
            user_agent=user_agent,
            device_name=_parse_device_name(user_agent),
            expires_at=expires_at,
        )

        with self._lock:
            self._maybe_cleanup()

            if user_id not in self._sessions:
                self._sessions[user_id] = OrderedDict()

            user_sessions = self._sessions[user_id]

            # Enforce max sessions per user (evict oldest)
            while len(user_sessions) >= self.max_sessions_per_user:
                oldest_id = next(iter(user_sessions))
                del user_sessions[oldest_id]
                logger.debug(f"Evicted oldest session {oldest_id[:8]}... for user {user_id}")

            user_sessions[token_jti] = session

        logger.info(
            f"Created session {token_jti[:8]}... for user {user_id} from {ip_address or 'unknown'}"
        )
        return session

    def touch_session(self, user_id: str, token_jti: str) -> bool:
        """Update last activity time for a session.

        Args:
            user_id: User identifier
            token_jti: Session/token identifier

        Returns:
            True if session was updated, False if not found
        """
        with self._lock:
            user_sessions = self._sessions.get(user_id)
            if not user_sessions:
                return False

            session = user_sessions.get(token_jti)
            if not session:
                return False

            session.last_activity = time.time()

            # Move to end of OrderedDict (most recently active)
            user_sessions.move_to_end(token_jti)
            return True

    def update_mfa_verification(
        self, user_id: str, token_jti: str, methods: Optional[List[str]] = None
    ) -> bool:
        """Record MFA verification for a session (step-up auth)."""
        with self._lock:
            user_sessions = self._sessions.get(user_id)
            if not user_sessions:
                return False
            session = user_sessions.get(token_jti)
            if not session:
                return False
            session.record_mfa_verification(methods)
            logger.debug(f"MFA verification recorded for session {token_jti[:8]}...")
            return True

    def get_mfa_freshness(self, user_id: str, token_jti: str) -> Optional[int]:
        """Get seconds since last MFA verification."""
        with self._lock:
            user_sessions = self._sessions.get(user_id)
            if not user_sessions:
                return None
            session = user_sessions.get(token_jti)
            if not session:
                return None
            return session.mfa_age_seconds()

    def is_session_mfa_fresh(
        self, user_id: str, token_jti: str, max_age_seconds: int = 900
    ) -> bool:
        """Check if MFA verification is fresh for a session."""
        with self._lock:
            user_sessions = self._sessions.get(user_id)
            if not user_sessions:
                return False
            session = user_sessions.get(token_jti)
            if not session:
                return False
            return session.is_mfa_fresh(max_age_seconds)

    def get_session(self, user_id: str, token_jti: str) -> Optional[JWTSession]:
        """Get a specific session.

        Args:
            user_id: User identifier
            token_jti: Session/token identifier

        Returns:
            JWTSession if found and valid, None otherwise
        """
        with self._lock:
            user_sessions = self._sessions.get(user_id)
            if not user_sessions:
                return None

            session = user_sessions.get(token_jti)
            if session and (session.is_expired() or session.is_inactive(self.inactivity_timeout)):
                # Clean up expired/inactive session
                del user_sessions[token_jti]
                return None

            return session

    def list_sessions(
        self,
        user_id: str,
        include_inactive: bool = False,
    ) -> List[JWTSession]:
        """List all active sessions for a user.

        Args:
            user_id: User identifier
            include_inactive: Include sessions past inactivity timeout

        Returns:
            List of JWTSession instances
        """
        with self._lock:
            user_sessions = self._sessions.get(user_id)
            if not user_sessions:
                return []

            result = []
            to_remove = []

            for session_id, session in user_sessions.items():
                if session.is_expired():
                    to_remove.append(session_id)
                    continue

                if not include_inactive and session.is_inactive(self.inactivity_timeout):
                    continue

                result.append(session)

            # Clean up expired sessions
            for session_id in to_remove:
                del user_sessions[session_id]

            return result

    def revoke_session(self, user_id: str, token_jti: str) -> bool:
        """Revoke a specific session.

        Note: This only removes tracking. The actual token should also
        be added to the blacklist via the token revocation system.

        Args:
            user_id: User identifier
            token_jti: Session/token identifier

        Returns:
            True if session was revoked, False if not found
        """
        with self._lock:
            user_sessions = self._sessions.get(user_id)
            if not user_sessions:
                return False

            if token_jti not in user_sessions:
                return False

            del user_sessions[token_jti]
            logger.info(f"Revoked session {token_jti[:8]}... for user {user_id}")
            return True

    def revoke_all_sessions(self, user_id: str, except_jti: Optional[str] = None) -> int:
        """Revoke all sessions for a user.

        Args:
            user_id: User identifier
            except_jti: Optional session ID to exclude (current session)

        Returns:
            Number of sessions revoked
        """
        with self._lock:
            user_sessions = self._sessions.get(user_id)
            if not user_sessions:
                return 0

            if except_jti:
                # Keep current session, revoke others
                to_revoke = [sid for sid in user_sessions if sid != except_jti]
                for sid in to_revoke:
                    del user_sessions[sid]
                count = len(to_revoke)
            else:
                count = len(user_sessions)
                user_sessions.clear()

            logger.info(f"Revoked {count} sessions for user {user_id}")
            return count

    def get_session_count(self, user_id: str) -> int:
        """Get number of active sessions for a user."""
        with self._lock:
            user_sessions = self._sessions.get(user_id)
            return len(user_sessions) if user_sessions else 0

    def _maybe_cleanup(self) -> None:
        """Periodic cleanup of expired sessions."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = now
        total_removed = 0

        for user_id, user_sessions in list(self._sessions.items()):
            to_remove = [
                sid
                for sid, session in user_sessions.items()
                if session.is_expired() or session.is_inactive(self.inactivity_timeout)
            ]
            for sid in to_remove:
                del user_sessions[sid]
                total_removed += 1

            # Remove empty user entries
            if not user_sessions:
                del self._sessions[user_id]

        if total_removed > 0:
            logger.debug(f"Cleaned up {total_removed} expired/inactive sessions")


# Singleton instance
_session_manager: Optional[JWTSessionManager] = None
_manager_lock = threading.Lock()


def get_session_manager() -> JWTSessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        with _manager_lock:
            if _session_manager is None:
                _session_manager = JWTSessionManager()
    return _session_manager


def reset_session_manager() -> None:
    """Reset the session manager (for testing)."""
    global _session_manager
    with _manager_lock:
        _session_manager = None


__all__ = [
    "JWTSession",
    "JWTSessionManager",
    "get_session_manager",
    "reset_session_manager",
]
