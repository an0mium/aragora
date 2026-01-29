"""
Gateway Protocol Adapter.

Provides a minimal session/presence protocol layer that can be used to
match Moltbot-style gateway behavior without requiring full protocol parity.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from aragora.gateway.server import GatewayConfig, LocalGateway

@dataclass
class GatewaySession:
    """Session state for a connected client/device."""

    session_id: str
    user_id: str
    device_id: str
    status: str = "active"  # active, paused, ended
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    end_reason: str | None = None

class GatewayProtocolAdapter:
    """
    Minimal protocol adapter for gateway session + presence semantics.

    This is intentionally lightweight: it provides session lifecycle and
    presence tracking that can be extended into full Moltbot parity.
    """

    def __init__(self, gateway: LocalGateway):
        self._gateway = gateway
        self._sessions: dict[str, GatewaySession] = {}

    def get_config(self) -> GatewayConfig:
        """Return the active gateway configuration."""
        return self._gateway._config

    async def open_session(
        self,
        user_id: str,
        device_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> GatewaySession:
        """Create a new session."""
        session_id = f"sess-{uuid.uuid4().hex[:12]}"
        session = GatewaySession(
            session_id=session_id,
            user_id=user_id,
            device_id=device_id,
            metadata=metadata or {},
        )
        self._sessions[session_id] = session
        return session

    async def close_session(
        self, session_id: str, reason: str = "ended"
    ) -> GatewaySession | None:
        """End a session and record a reason."""
        session = self._sessions.get(session_id)
        if not session:
            return None
        session.status = "ended"
        session.end_reason = reason
        session.last_seen = time.time()
        return session

    async def update_presence(
        self,
        session_id: str,
        status: str = "active",
    ) -> GatewaySession | None:
        """Update presence for a session."""
        session = self._sessions.get(session_id)
        if not session:
            return None
        session.status = status
        session.last_seen = time.time()
        return session

    async def get_session(self, session_id: str) -> GatewaySession | None:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    async def list_sessions(
        self,
        user_id: str | None = None,
        device_id: str | None = None,
        status: str | None = None,
    ) -> list[GatewaySession]:
        """List sessions with optional filters."""
        sessions = list(self._sessions.values())
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        if device_id:
            sessions = [s for s in sessions if s.device_id == device_id]
        if status:
            sessions = [s for s in sessions if s.status == status]
        return sessions

    async def get_config_for_session(self, session_id: str) -> GatewayConfig | None:
        """Return gateway config scoped to a session.

        Validates the session exists and is not ended before returning
        the active gateway configuration.
        """
        session = self._sessions.get(session_id)
        if not session or session.status == "ended":
            return None
        return self.get_config()

    async def resume_session(self, session_id: str, device_id: str) -> GatewaySession | None:
        """Resume a paused session from the same device.

        Restores the session to ``active`` status and refreshes
        ``last_seen``.  Returns ``None`` if the session does not exist,
        is not paused, or the device_id does not match.
        """
        session = self._sessions.get(session_id)
        if not session:
            return None
        if session.status != "paused":
            return None
        if session.device_id != device_id:
            return None
        session.status = "active"
        session.last_seen = time.time()
        return session

    async def bind_device_to_session(self, session_id: str, device_id: str) -> bool:
        """Bind a device to an existing active session.

        Allows device migration (e.g. reconnecting from a new browser tab).
        The session must be active.  Returns ``True`` on success.
        """
        session = self._sessions.get(session_id)
        if not session or session.status != "active":
            return False
        session.device_id = device_id
        session.last_seen = time.time()
        return True
