"""
Gateway Protocol Adapter.

Provides a minimal session/presence protocol layer that can be used to
match OpenClaw-style gateway behavior without requiring full protocol parity.

Supports optional policy intercept hooks for enterprise security:
- Session creation policy checks
- Action execution policy enforcement
- Message filtering and audit logging
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from aragora.gateway.persistence import GatewayStore
from aragora.gateway.server import GatewayConfig, LocalGateway

if TYPE_CHECKING:
    from aragora.gateway.openclaw_policy import OpenClawPolicy
    from aragora.gateway.openclaw_proxy import OpenClawSecureProxy

OPENCLAW_PROTOCOL_VERSION = 3

logger = logging.getLogger(__name__)


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


@dataclass
class CloseRequest:
    """Request to close the WebSocket connection."""

    code: int
    reason: str


@dataclass
class PolicyInterceptResult:
    """Result of a policy intercept check."""

    allowed: bool
    reason: str | None = None
    modified_payload: dict[str, Any] | None = None
    audit_data: dict[str, Any] | None = None


class GatewayProtocolAdapter:
    """
    Minimal protocol adapter for gateway session + presence semantics.

    This is intentionally lightweight: it provides session lifecycle and
    presence tracking that can be extended into full Moltbot parity.

    Supports optional policy hooks for enterprise security:
    - pre_session_hook: Called before session creation
    - post_session_hook: Called after session creation
    - action_policy_hook: Called to evaluate actions
    - audit_hook: Called for audit logging
    """

    def __init__(
        self,
        gateway: LocalGateway,
        store: GatewayStore | None = None,
        policy: "OpenClawPolicy | None" = None,
        secure_proxy: "OpenClawSecureProxy | None" = None,
        pre_session_hook: Callable[[str, str, dict[str, Any] | None], PolicyInterceptResult]
        | None = None,
        post_session_hook: Callable[[GatewaySession], None] | None = None,
        action_policy_hook: Callable[[str, str, dict[str, Any]], PolicyInterceptResult]
        | None = None,
        audit_hook: Callable[[dict[str, Any]], None] | None = None,
    ):
        self._gateway = gateway
        self._sessions: dict[str, GatewaySession] = {}
        self._store = store
        self._store_loaded = False
        self._store_lock = asyncio.Lock()

        # Policy integration
        self._policy = policy
        self._secure_proxy = secure_proxy
        self._pre_session_hook = pre_session_hook
        self._post_session_hook = post_session_hook
        self._action_policy_hook = action_policy_hook
        self._audit_hook = audit_hook

    async def _ensure_loaded(self) -> None:
        if self._store is None or self._store_loaded:
            return
        async with self._store_lock:
            if self._store_loaded:
                return
            records = await self._store.load_sessions()
            for record in records:
                session = self._record_to_session(record)
                if session:
                    self._sessions[session.session_id] = session
            self._store_loaded = True

    async def _persist_session(self, session: GatewaySession) -> None:
        if self._store is None:
            return
        await self._store.save_session(self._session_to_record(session))

    def _record_to_session(self, record: dict[str, Any]) -> GatewaySession | None:
        session_id = record.get("session_id")
        user_id = record.get("user_id")
        device_id = record.get("device_id")
        if not session_id or not user_id or not device_id:
            return None
        return GatewaySession(
            session_id=session_id,
            user_id=user_id,
            device_id=device_id,
            status=record.get("status", "active"),
            created_at=record.get("created_at", time.time()),
            last_seen=record.get("last_seen", record.get("created_at", time.time())),
            metadata=record.get("metadata", {}) or {},
            end_reason=record.get("end_reason"),
        )

    def _session_to_record(self, session: GatewaySession) -> dict[str, Any]:
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

    def _emit_audit(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an audit event via the audit hook."""
        if self._audit_hook:
            try:
                self._audit_hook(
                    {
                        "event_type": event_type,
                        "timestamp": time.time(),
                        "source": "gateway_protocol",
                        **data,
                    }
                )
            except Exception as e:
                logger.warning(f"Audit hook failed: {e}")

    def set_policy(self, policy: "OpenClawPolicy") -> None:
        """Set the policy engine for this adapter."""
        self._policy = policy

    def set_secure_proxy(self, proxy: "OpenClawSecureProxy") -> None:
        """Set the secure proxy for action routing."""
        self._secure_proxy = proxy

    async def get_stats(self) -> dict[str, Any]:
        """Return gateway stats for protocol responses."""
        return await self._gateway.get_stats()

    def get_config(self) -> GatewayConfig:
        """Return the active gateway configuration."""
        return self._gateway._config

    async def open_session(
        self,
        user_id: str,
        device_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> GatewaySession | PolicyInterceptResult:
        """Create a new session.

        If a pre_session_hook is configured and returns allowed=False,
        returns a PolicyInterceptResult instead of a session.
        """
        await self._ensure_loaded()

        # Check pre-session policy hook
        if self._pre_session_hook:
            result = self._pre_session_hook(user_id, device_id, metadata)
            if not result.allowed:
                self._emit_audit(
                    "session_denied",
                    {
                        "user_id": user_id,
                        "device_id": device_id,
                        "reason": result.reason,
                    },
                )
                return result

        session_id = f"sess-{uuid.uuid4().hex[:12]}"
        session = GatewaySession(
            session_id=session_id,
            user_id=user_id,
            device_id=device_id,
            metadata=metadata or {},
        )
        self._sessions[session_id] = session
        await self._persist_session(session)

        # Call post-session hook
        if self._post_session_hook:
            try:
                self._post_session_hook(session)
            except Exception as e:
                logger.warning(f"Post-session hook failed: {e}")

        self._emit_audit(
            "session_created",
            {
                "session_id": session_id,
                "user_id": user_id,
                "device_id": device_id,
            },
        )

        return session

    async def evaluate_action(
        self,
        session_id: str,
        action_type: str,
        action_params: dict[str, Any],
    ) -> PolicyInterceptResult:
        """Evaluate an action against policy.

        Used to check if an action should be allowed before execution.
        Returns PolicyInterceptResult with allowed=True/False.
        """
        session = self._sessions.get(session_id)
        if not session:
            return PolicyInterceptResult(
                allowed=False,
                reason="Session not found",
            )

        # Use action policy hook if available
        if self._action_policy_hook:
            result = self._action_policy_hook(session_id, action_type, action_params)
            self._emit_audit(
                "action_evaluated",
                {
                    "session_id": session_id,
                    "action_type": action_type,
                    "allowed": result.allowed,
                    "reason": result.reason,
                },
            )
            return result

        # If secure proxy is configured, use it for evaluation
        if self._secure_proxy:
            # Map to proxy action types
            proxy_session = self._secure_proxy.get_session(session_id)
            if not proxy_session:
                return PolicyInterceptResult(
                    allowed=False,
                    reason="No proxy session",
                )
            # Proxy will handle policy evaluation
            return PolicyInterceptResult(allowed=True)

        # No policy configured - allow by default
        return PolicyInterceptResult(allowed=True)

    async def close_session(self, session_id: str, reason: str = "ended") -> GatewaySession | None:
        """End a session and record a reason."""
        await self._ensure_loaded()
        session = self._sessions.get(session_id)
        if not session:
            return None
        session.status = "ended"
        session.end_reason = reason
        session.last_seen = time.time()
        await self._persist_session(session)

        self._emit_audit(
            "session_closed",
            {
                "session_id": session_id,
                "user_id": session.user_id,
                "reason": reason,
                "duration_seconds": time.time() - session.created_at,
            },
        )

        return session

    async def update_presence(
        self,
        session_id: str,
        status: str = "active",
    ) -> GatewaySession | None:
        """Update presence for a session."""
        await self._ensure_loaded()
        session = self._sessions.get(session_id)
        if not session:
            return None
        session.status = status
        session.last_seen = time.time()
        await self._persist_session(session)
        return session

    async def get_session(self, session_id: str) -> GatewaySession | None:
        """Get a session by ID."""
        await self._ensure_loaded()
        return self._sessions.get(session_id)

    async def list_sessions(
        self,
        user_id: str | None = None,
        device_id: str | None = None,
        status: str | None = None,
    ) -> list[GatewaySession]:
        """List sessions with optional filters."""
        await self._ensure_loaded()
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
        await self._ensure_loaded()
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
        await self._ensure_loaded()
        session = self._sessions.get(session_id)
        if not session:
            return None
        if session.status != "paused":
            return None
        if session.device_id != device_id:
            return None
        session.status = "active"
        session.last_seen = time.time()
        await self._persist_session(session)
        return session

    async def bind_device_to_session(self, session_id: str, device_id: str) -> bool:
        """Bind a device to an existing active session.

        Allows device migration (e.g. reconnecting from a new browser tab).
        The session must be active.  Returns ``True`` on success.
        """
        await self._ensure_loaded()
        session = self._sessions.get(session_id)
        if not session or session.status != "active":
            return False
        session.device_id = device_id
        session.last_seen = time.time()
        await self._persist_session(session)
        return True


class GatewayWebSocketProtocol:
    """Minimal WebSocket protocol handler for gateway session/presence parity."""

    def __init__(
        self,
        adapter: GatewayProtocolAdapter,
        *,
        protocol_version: int = OPENCLAW_PROTOCOL_VERSION,
        tick_interval_ms: int = 15000,
        require_connect_first: bool = True,
    ) -> None:
        self._adapter = adapter
        self._protocol_version = protocol_version
        self._tick_interval_ms = tick_interval_ms
        self._require_connect_first = require_connect_first
        self._connected = False
        self._session_id: str | None = None
        self._close_request: CloseRequest | None = None
        self._challenge_nonce: str | None = None

    def create_challenge_event(self) -> dict[str, Any]:
        """Create an OpenClaw-style connect challenge event."""
        nonce = uuid.uuid4().hex
        self._challenge_nonce = nonce
        return {
            "type": "event",
            "event": "connect.challenge",
            "payload": {
                "nonce": nonce,
                "issued_at": time.time(),
                "protocol": {"min": 1, "max": self._protocol_version},
            },
        }

    def consume_close_request(self) -> CloseRequest | None:
        """Return and clear any pending close request."""
        close = self._close_request
        self._close_request = None
        return close

    async def handle_message(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Handle a WebSocket message payload and return a response payload."""
        msg_type = payload.get("type")
        request_id = payload.get("request_id") or payload.get("requestId")

        if msg_type == "req":
            return await self._handle_openclaw_request(payload)

        if msg_type == "event":
            return None

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

    async def _handle_openclaw_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = payload.get("id")
        method = payload.get("method")
        params = payload.get("params") or {}

        if self._require_connect_first and not self._connected and method != "connect":
            self._close_request = CloseRequest(1008, "connect required before other requests")
            return self._openclaw_error(
                request_id,
                "invalid_state",
                "connect required before other requests",
            )

        if method == "connect":
            return await self._handle_connect(request_id, params)

        if method == "presence.update":
            session_id = params.get("session_id") or params.get("sessionId") or self._session_id
            status = params.get("status", "active")
            if not session_id:
                return self._openclaw_error(request_id, "missing_fields", "session_id required")
            session = await self._adapter.update_presence(session_id, status=status)
            if session is None:
                return self._openclaw_error(request_id, "not_found", "session not found")
            return self._openclaw_ok(
                request_id,
                {
                    "type": "presence.updated",
                    "session": self._serialize_session(session),
                },
            )

        if method == "system.presence":
            sessions = await self._adapter.list_sessions()
            return self._openclaw_ok(
                request_id,
                {
                    "type": "system.presence",
                    "sessions": [self._serialize_session(s) for s in sessions],
                },
            )

        if method in {"health", "status"}:
            stats = await self._adapter.get_stats()
            return self._openclaw_ok(request_id, {"type": method, "stats": stats})

        return self._openclaw_error(request_id, "unknown_method", f"unsupported method: {method}")

    async def _handle_connect(
        self,
        request_id: str | None,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        min_protocol = params.get("minProtocol", 1)
        max_protocol = params.get("maxProtocol", self._protocol_version)
        try:
            min_protocol = int(min_protocol)
            max_protocol = int(max_protocol)
        except (TypeError, ValueError):
            return self._openclaw_error(
                request_id,
                "invalid_protocol",
                "minProtocol/maxProtocol must be integers",
            )

        if min_protocol > self._protocol_version:
            self._close_request = CloseRequest(1008, "protocol unsupported")
            return self._openclaw_error(
                request_id,
                "protocol_unsupported",
                "protocol version too new",
            )

        negotiated = min(max_protocol, self._protocol_version)
        user_id = params.get("user_id") or params.get("userId") or "unknown"
        device_id = None
        device = params.get("device") or {}
        if isinstance(device, dict):
            device_id = device.get("id")
        device_id = device_id or params.get("device_id") or params.get("deviceId") or "unknown"

        metadata = {
            "role": params.get("role"),
            "scopes": params.get("scopes", []),
            "client": params.get("client"),
            "commands": params.get("commands"),
            "permissions": params.get("permissions"),
            "auth": params.get("auth"),
            "nonce": params.get("nonce") or self._challenge_nonce,
        }

        session = await self._adapter.open_session(
            user_id=user_id,
            device_id=device_id,
            metadata=metadata,
        )
        self._connected = True
        self._session_id = session.session_id

        payload = {
            "type": "hello-ok",
            "protocol": {"version": negotiated},
            "session": self._serialize_session(session),
            "policy": {"tickIntervalMs": self._tick_interval_ms},
        }
        return self._openclaw_ok(request_id, payload)

    def _wrap_response(self, payload: dict[str, Any], request_id: str | None) -> dict[str, Any]:
        if request_id:
            payload["request_id"] = request_id
        return payload

    def _error(self, code: str, message: str, request_id: str | None) -> dict[str, Any]:
        return self._wrap_response(
            {"type": "error", "error": {"code": code, "message": message}}, request_id
        )

    def _openclaw_ok(self, request_id: str | None, payload: dict[str, Any]) -> dict[str, Any]:
        return {"type": "res", "id": request_id, "ok": True, "payload": payload}

    def _openclaw_error(self, request_id: str | None, code: str, message: str) -> dict[str, Any]:
        return {
            "type": "res",
            "id": request_id,
            "ok": False,
            "error": {"code": code, "message": message},
        }

    def _serialize_session(
        self, session: GatewaySession | PolicyInterceptResult | None
    ) -> dict[str, Any] | None:
        if session is None:
            return None
        # Handle PolicyInterceptResult - it doesn't have GatewaySession fields
        if isinstance(session, PolicyInterceptResult):
            return {"blocked": True, "reason": session.reason}
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
