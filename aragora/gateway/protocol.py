"""
Gateway Protocol Adapter.

Provides a minimal session/presence protocol layer that can be used to
match Moltbot-style gateway behavior without requiring full protocol parity.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
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

    async def close_session(self, session_id: str, reason: str = "ended") -> GatewaySession | None:
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


class GatewayWebSocketProtocol:
    """Minimal WebSocket protocol handler for gateway session/presence parity."""

    def __init__(self, adapter: GatewayProtocolAdapter) -> None:
        self._adapter = adapter

    async def handle_message(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Handle a WebSocket message payload and return a response payload."""
        msg_type = payload.get("type")
        request_id = payload.get("request_id") or payload.get("requestId")

        if msg_type == "ping":
            return self._wrap_response({"type": "pong"}, request_id)

        if msg_type == "session.open":
            user_id = payload.get("user_id")
            device_id = payload.get("device_id")
            if not user_id or not device_id:
                return self._error(
                    "missing_fields",
                    "session.open requires user_id and device_id",
                    request_id,
                )
            session = await self._adapter.open_session(
                user_id=user_id,
                device_id=device_id,
                metadata=payload.get("metadata"),
            )
            return self._wrap_response(
                {"type": "session.opened", "session": self._serialize_session(session)},
                request_id,
            )

        if msg_type == "session.close":
            session_id = payload.get("session_id")
            if not session_id:
                return self._error(
                    "missing_fields",
                    "session.close requires session_id",
                    request_id,
                )
            session = await self._adapter.close_session(
                session_id,
                reason=payload.get("reason", "ended"),
            )
            if session is None:
                return self._error("not_found", "session not found", request_id)
            return self._wrap_response(
                {"type": "session.closed", "session": self._serialize_session(session)},
                request_id,
            )

        if msg_type == "session.get":
            session_id = payload.get("session_id")
            if not session_id:
                return self._error(
                    "missing_fields",
                    "session.get requires session_id",
                    request_id,
                )
            session = await self._adapter.get_session(session_id)
            if session is None:
                return self._error("not_found", "session not found", request_id)
            return self._wrap_response(
                {"type": "session", "session": self._serialize_session(session)},
                request_id,
            )

        if msg_type == "session.list":
            sessions = await self._adapter.list_sessions(
                user_id=payload.get("user_id"),
                device_id=payload.get("device_id"),
                status=payload.get("status"),
            )
            return self._wrap_response(
                {
                    "type": "session.list",
                    "sessions": [self._serialize_session(s) for s in sessions],
                },
                request_id,
            )

        if msg_type == "presence.update":
            session_id = payload.get("session_id")
            status = payload.get("status", "active")
            if not session_id:
                return self._error(
                    "missing_fields",
                    "presence.update requires session_id",
                    request_id,
                )
            session = await self._adapter.update_presence(session_id, status=status)
            if session is None:
                return self._error("not_found", "session not found", request_id)
            return self._wrap_response(
                {"type": "presence.updated", "session": self._serialize_session(session)},
                request_id,
            )

        if msg_type == "session.resume":
            session_id = payload.get("session_id")
            device_id = payload.get("device_id")
            if not session_id or not device_id:
                return self._error(
                    "missing_fields",
                    "session.resume requires session_id and device_id",
                    request_id,
                )
            session = await self._adapter.resume_session(session_id, device_id=device_id)
            if session is None:
                return self._error("not_found", "session not resumable", request_id)
            return self._wrap_response(
                {"type": "session.resumed", "session": self._serialize_session(session)},
                request_id,
            )

        if msg_type == "session.bind":
            session_id = payload.get("session_id")
            device_id = payload.get("device_id")
            if not session_id or not device_id:
                return self._error(
                    "missing_fields",
                    "session.bind requires session_id and device_id",
                    request_id,
                )
            ok = await self._adapter.bind_device_to_session(session_id, device_id=device_id)
            if not ok:
                return self._error("not_found", "session not bindable", request_id)
            session = await self._adapter.get_session(session_id)
            return self._wrap_response(
                {"type": "session.bound", "session": self._serialize_session(session)},
                request_id,
            )

        if msg_type == "config.get":
            session_id = payload.get("session_id")
            if session_id:
                config = await self._adapter.get_config_for_session(session_id)
                if config is None:
                    return self._error("not_found", "session not found", request_id)
            else:
                config = self._adapter.get_config()
            return self._wrap_response(
                {"type": "config", "config": self._serialize_config(config)},
                request_id,
            )

        if msg_type is None:
            return self._error("missing_type", "message type is required", request_id)

        return self._error("unknown_type", f"unsupported type: {msg_type}", request_id)

    def _wrap_response(self, payload: dict[str, Any], request_id: str | None) -> dict[str, Any]:
        if request_id:
            payload["request_id"] = request_id
        return payload

    def _error(self, code: str, message: str, request_id: str | None) -> dict[str, Any]:
        return self._wrap_response(
            {"type": "error", "error": {"code": code, "message": message}}, request_id
        )

    def _serialize_session(self, session: GatewaySession | None) -> dict[str, Any] | None:
        if session is None:
            return None
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "device_id": session.device_id,
            "status": session.status,
            "created_at": session.created_at,
            "last_seen": session.last_seen,
            "metadata": dict(session.metadata),
            "end_reason": session.end_reason,
        }

    def _serialize_config(self, config: GatewayConfig) -> dict[str, Any]:
        return asdict(config)
