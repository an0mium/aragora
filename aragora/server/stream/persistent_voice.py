# mypy: ignore-errors
"""
Persistent Voice Manager.

Manages long-lived voice sessions with reconnection support for:
- 24-hour session TTL (configurable)
- Reconnection within grace period after disconnect
- Heartbeat monitoring
- Session state persistence

Usage:
    from aragora.server.stream.persistent_voice import PersistentVoiceManager

    manager = PersistentVoiceManager()

    # Create persistent session
    session = await manager.create_session(debate_id, user_id, persistent=True)

    # Handle disconnect with reconnect token
    token = await manager.handle_disconnect(session)

    # Reconnect using token
    session = await manager.reconnect(token, ws)
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# Configuration from environment
VOICE_MAX_SESSION_SECONDS = int(os.environ.get("ARAGORA_VOICE_MAX_SESSION", "86400"))  # 24h
VOICE_RECONNECT_WINDOW_SECONDS = int(os.environ.get("ARAGORA_VOICE_RECONNECT_WINDOW", "300"))  # 5m
VOICE_HEARTBEAT_INTERVAL_SECONDS = int(os.environ.get("ARAGORA_VOICE_HEARTBEAT_INTERVAL", "30"))
VOICE_HEARTBEAT_TIMEOUT_SECONDS = int(os.environ.get("ARAGORA_VOICE_HEARTBEAT_TIMEOUT", "90"))


@dataclass
class PersistentVoiceSession:
    """
    A long-lived voice session with reconnection support.

    Attributes:
        session_id: Unique session identifier
        user_id: Associated user
        debate_id: Associated debate (optional)
        is_persistent: Whether session supports reconnection
        audio_format: Audio format (pcm_16khz, opus, etc.)
        state: Session state (active, disconnected, expired)
        reconnect_token: Token for reconnection
        reconnect_expires_at: When reconnect token expires
        last_heartbeat: Last heartbeat timestamp
        created_at: Session creation timestamp
        expires_at: Session expiration timestamp
        metadata: Additional session metadata
    """

    session_id: str
    user_id: str
    debate_id: Optional[str] = None
    is_persistent: bool = True
    audio_format: str = "pcm_16khz"

    # State
    state: str = "active"  # active, disconnected, reconnecting, expired
    reconnect_token: Optional[str] = None
    reconnect_expires_at: Optional[float] = None

    # Heartbeat
    last_heartbeat: float = field(default_factory=time.time)

    # Timestamps
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + VOICE_MAX_SESSION_SECONDS)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Audio state
    audio_buffer: bytes = b""
    transcript_buffer: str = ""

    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return self.state == "active" and time.time() < self.expires_at

    @property
    def is_disconnected(self) -> bool:
        """Check if session is disconnected but reconnectable."""
        return self.state == "disconnected" and self.can_reconnect

    @property
    def can_reconnect(self) -> bool:
        """Check if session can be reconnected."""
        if not self.is_persistent:
            return False
        if self.reconnect_expires_at is None:
            return False
        return time.time() < self.reconnect_expires_at

    @property
    def time_until_expiry(self) -> float:
        """Time until session expires in seconds."""
        return max(0, self.expires_at - time.time())

    @property
    def time_until_reconnect_expiry(self) -> float:
        """Time until reconnect window closes in seconds."""
        if self.reconnect_expires_at is None:
            return 0
        return max(0, self.reconnect_expires_at - time.time())

    def touch(self) -> None:
        """Update last heartbeat timestamp."""
        self.last_heartbeat = time.time()

    def extend(self, seconds: float) -> None:
        """Extend session expiration."""
        self.expires_at = time.time() + seconds

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "debate_id": self.debate_id,
            "is_persistent": self.is_persistent,
            "audio_format": self.audio_format,
            "state": self.state,
            "can_reconnect": self.can_reconnect,
            "time_until_expiry": self.time_until_expiry,
            "time_until_reconnect_expiry": self.time_until_reconnect_expiry,
            "created_at": datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat(),
            "expires_at": datetime.fromtimestamp(self.expires_at, tz=timezone.utc).isoformat(),
            "metadata": self.metadata,
        }


class PersistentVoiceManager:
    """
    Manages persistent voice sessions with reconnection support.

    Features:
    - Long-lived sessions (up to 24 hours)
    - Reconnection within grace period after disconnect
    - Heartbeat monitoring for session health
    - Session state persistence (Redis-backed for scaling)
    """

    def __init__(self):
        """Initialize the voice manager."""
        self._sessions: Dict[str, PersistentVoiceSession] = {}
        self._reconnect_tokens: Dict[str, str] = {}  # token -> session_id
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        # Callbacks
        self._on_session_expired: Optional[Callable[[PersistentVoiceSession], None]] = None
        self._on_heartbeat_timeout: Optional[Callable[[PersistentVoiceSession], None]] = None

    # ==========================================================================
    # Lifecycle
    # ==========================================================================

    async def start(self) -> None:
        """Start the voice manager background tasks."""
        if self._running:
            return

        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
        self._cleanup_task = asyncio.create_task(self._cleanup_expired())
        logger.info("Persistent voice manager started")

    async def stop(self) -> None:
        """Stop the voice manager."""
        self._running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("Persistent voice manager stopped")

    # ==========================================================================
    # Session Management
    # ==========================================================================

    async def create_session(
        self,
        user_id: str,
        debate_id: Optional[str] = None,
        persistent: bool = True,
        ttl_hours: float = 24,
        audio_format: str = "pcm_16khz",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PersistentVoiceSession:
        """
        Create a new voice session.

        Args:
            user_id: User identifier
            debate_id: Optional debate to attach to
            persistent: Whether to enable reconnection
            ttl_hours: Session TTL in hours
            audio_format: Audio format
            metadata: Additional metadata

        Returns:
            Created PersistentVoiceSession
        """
        session_id = f"voice_{secrets.token_hex(16)}"
        ttl_seconds = ttl_hours * 3600

        session = PersistentVoiceSession(
            session_id=session_id,
            user_id=user_id,
            debate_id=debate_id,
            is_persistent=persistent,
            audio_format=audio_format,
            expires_at=time.time() + ttl_seconds,
            metadata=metadata or {},
        )

        self._sessions[session_id] = session

        # Store in session store for persistence
        await self._persist_session(session)

        logger.info(
            f"Created voice session {session_id} for user {user_id} "
            f"(persistent={persistent}, ttl={ttl_hours}h)"
        )
        return session

    async def get_session(self, session_id: str) -> Optional[PersistentVoiceSession]:
        """Get a session by ID."""
        session = self._sessions.get(session_id)

        # Try to load from persistent store if not in memory
        if not session:
            session = await self._load_session(session_id)
            if session:
                self._sessions[session_id] = session

        return session

    async def get_user_sessions(self, user_id: str) -> List[PersistentVoiceSession]:
        """Get all sessions for a user."""
        return [s for s in self._sessions.values() if s.user_id == user_id]

    async def terminate_session(self, session_id: str) -> bool:
        """
        Terminate a session immediately.

        Args:
            session_id: Session to terminate

        Returns:
            True if session was terminated
        """
        session = self._sessions.pop(session_id, None)
        if session:
            session.state = "expired"
            await self._remove_session(session_id)

            # Clean up reconnect token
            if session.reconnect_token:
                self._reconnect_tokens.pop(session.reconnect_token, None)

            logger.info(f"Terminated voice session {session_id}")
            return True

        return False

    # ==========================================================================
    # Heartbeat
    # ==========================================================================

    async def heartbeat(self, session_id: str) -> bool:
        """
        Record a heartbeat for a session.

        Args:
            session_id: Session to heartbeat

        Returns:
            True if heartbeat recorded
        """
        session = await self.get_session(session_id)
        if not session:
            return False

        session.touch()
        return True

    async def _heartbeat_monitor(self) -> None:
        """Background task to monitor heartbeats."""
        while self._running:
            try:
                await asyncio.sleep(VOICE_HEARTBEAT_INTERVAL_SECONDS)

                now = time.time()
                timed_out: List[str] = []

                for session_id, session in list(self._sessions.items()):
                    if session.state != "active":
                        continue

                    time_since_heartbeat = now - session.last_heartbeat
                    if time_since_heartbeat > VOICE_HEARTBEAT_TIMEOUT_SECONDS:
                        timed_out.append(session_id)

                # Handle timed out sessions
                for session_id in timed_out:
                    session = self._sessions.get(session_id)
                    if session:
                        logger.warning(
                            f"Heartbeat timeout for session {session_id}, initiating disconnect"
                        )
                        await self.handle_disconnect(session)

                        if self._on_heartbeat_timeout:
                            self._on_heartbeat_timeout(session)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")

    # ==========================================================================
    # Disconnect and Reconnect
    # ==========================================================================

    async def handle_disconnect(self, session: PersistentVoiceSession) -> Optional[str]:
        """
        Handle client disconnect.

        For persistent sessions, generates a reconnect token.

        Args:
            session: Session that disconnected

        Returns:
            Reconnect token if session is persistent, None otherwise
        """
        if not session.is_persistent:
            session.state = "expired"
            return None

        # Generate reconnect token
        reconnect_token = secrets.token_urlsafe(32)
        reconnect_expires_at = time.time() + VOICE_RECONNECT_WINDOW_SECONDS

        session.state = "disconnected"
        session.reconnect_token = reconnect_token
        session.reconnect_expires_at = reconnect_expires_at

        # Store token mapping
        self._reconnect_tokens[reconnect_token] = session.session_id

        # Persist state
        await self._persist_session(session)

        logger.info(
            f"Session {session.session_id} disconnected, "
            f"reconnect window: {VOICE_RECONNECT_WINDOW_SECONDS}s"
        )
        return reconnect_token

    async def reconnect(self, token: str) -> Optional[PersistentVoiceSession]:
        """
        Reconnect to a session using a reconnect token.

        Args:
            token: Reconnect token

        Returns:
            Session if reconnected successfully, None otherwise
        """
        session_id = self._reconnect_tokens.get(token)
        if not session_id:
            logger.warning("Invalid reconnect token")
            return None

        session = await self.get_session(session_id)
        if not session:
            logger.warning(f"Session {session_id} not found for reconnect")
            return None

        if not session.can_reconnect:
            logger.warning(f"Session {session_id} reconnect window expired")
            return None

        # Restore session
        session.state = "active"
        session.reconnect_token = None
        session.reconnect_expires_at = None
        session.touch()

        # Clean up token
        self._reconnect_tokens.pop(token, None)

        # Persist state
        await self._persist_session(session)

        logger.info(f"Session {session_id} reconnected successfully")
        return session

    # ==========================================================================
    # Cleanup
    # ==========================================================================

    async def _cleanup_expired(self) -> None:
        """Background task to clean up expired sessions."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute

                now = time.time()
                expired: List[str] = []
                reconnect_expired: List[str] = []

                for session_id, session in list(self._sessions.items()):
                    # Check session expiry
                    if now >= session.expires_at:
                        expired.append(session_id)
                        continue

                    # Check reconnect window expiry
                    if session.state == "disconnected":
                        if session.reconnect_expires_at and now >= session.reconnect_expires_at:
                            reconnect_expired.append(session_id)

                # Clean up expired sessions
                for session_id in expired:
                    session = self._sessions.pop(session_id, None)
                    if session:
                        session.state = "expired"
                        logger.info(f"Session {session_id} expired")

                        if self._on_session_expired:
                            self._on_session_expired(session)

                # Clean up reconnect-expired sessions
                for session_id in reconnect_expired:
                    session = self._sessions.pop(session_id, None)
                    if session:
                        session.state = "expired"
                        logger.info(f"Session {session_id} reconnect window expired")

                        if session.reconnect_token:
                            self._reconnect_tokens.pop(session.reconnect_token, None)

                # Clean up stale reconnect tokens
                stale_tokens = [
                    token
                    for token, sid in self._reconnect_tokens.items()
                    if sid not in self._sessions
                ]
                for token in stale_tokens:
                    self._reconnect_tokens.pop(token, None)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")

    # ==========================================================================
    # Persistence
    # ==========================================================================

    async def _persist_session(self, session: PersistentVoiceSession) -> None:
        """Persist session to session store."""
        try:
            from aragora.server.session_store import VoiceSession, get_session_store

            store = get_session_store()

            voice_session = VoiceSession(
                session_id=session.session_id,
                user_id=session.user_id,
                debate_id=session.debate_id,
                reconnect_token=session.reconnect_token,
                reconnect_expires_at=session.reconnect_expires_at,
                is_persistent=session.is_persistent,
                audio_format=session.audio_format,
                last_heartbeat=session.last_heartbeat,
                created_at=session.created_at,
                expires_at=session.expires_at,
                metadata=session.metadata,
            )

            store.set_voice_session(voice_session)

        except ImportError:
            logger.debug("Session store not available for persistence")
        except Exception as e:
            logger.warning(f"Failed to persist session: {e}")

    async def _load_session(self, session_id: str) -> Optional[PersistentVoiceSession]:
        """Load session from session store."""
        try:
            from aragora.server.session_store import get_session_store

            store = get_session_store()
            voice_session = store.get_voice_session(session_id)

            if not voice_session:
                return None

            return PersistentVoiceSession(
                session_id=voice_session.session_id,
                user_id=voice_session.user_id,
                debate_id=voice_session.debate_id,
                is_persistent=voice_session.is_persistent,
                audio_format=voice_session.audio_format,
                state="active" if not voice_session.reconnect_token else "disconnected",
                reconnect_token=voice_session.reconnect_token,
                reconnect_expires_at=voice_session.reconnect_expires_at,
                last_heartbeat=voice_session.last_heartbeat,
                created_at=voice_session.created_at,
                expires_at=voice_session.expires_at,
                metadata=voice_session.metadata,
            )

        except ImportError:
            return None
        except Exception as e:
            logger.warning(f"Failed to load session: {e}")
            return None

    async def _remove_session(self, session_id: str) -> None:
        """Remove session from session store."""
        try:
            from aragora.server.session_store import get_session_store

            store = get_session_store()
            store.delete_voice_session(session_id)

        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to remove session: {e}")

    # ==========================================================================
    # Callbacks
    # ==========================================================================

    def on_session_expired(self, callback: Callable[[PersistentVoiceSession], None]) -> None:
        """Set callback for session expiration."""
        self._on_session_expired = callback

    def on_heartbeat_timeout(self, callback: Callable[[PersistentVoiceSession], None]) -> None:
        """Set callback for heartbeat timeout."""
        self._on_heartbeat_timeout = callback

    # ==========================================================================
    # Statistics
    # ==========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        active = sum(1 for s in self._sessions.values() if s.state == "active")
        disconnected = sum(1 for s in self._sessions.values() if s.state == "disconnected")

        return {
            "total_sessions": len(self._sessions),
            "active_sessions": active,
            "disconnected_sessions": disconnected,
            "pending_reconnects": len(self._reconnect_tokens),
            "running": self._running,
        }


# ==========================================================================
# Global Instance
# ==========================================================================

_manager: Optional[PersistentVoiceManager] = None


def get_persistent_voice_manager() -> PersistentVoiceManager:
    """Get the global persistent voice manager."""
    global _manager
    if _manager is None:
        _manager = PersistentVoiceManager()
    return _manager
