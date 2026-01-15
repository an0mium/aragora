"""
WebSocket-only debate streaming server using the websockets library.

Provides real-time debate event streaming to connected clients without HTTP API.
For combined HTTP+WebSocket, use AiohttpUnifiedServer instead.

Usage:
    server = DebateStreamServer(port=8765)
    hooks = create_arena_hooks(server.emitter)
    arena = Arena(env, agents, event_hooks=hooks)

    # In async context:
    asyncio.create_task(server.start())
    await arena.run()
"""

import asyncio
import json
import logging
import os
import secrets
import threading
import time
from typing import Optional

from .emitter import TokenBucket
from .events import AudienceMessage, StreamEvent, StreamEventType
from .server_base import ServerBase
from .state_manager import LoopInstance

logger = logging.getLogger(__name__)

# Import centralized config
from aragora.config import WS_MAX_MESSAGE_SIZE

# Import auth for WebSocket authentication
from aragora.server.auth import auth_config

# Centralized CORS configuration
from aragora.server.cors_config import WS_ALLOWED_ORIGINS

# Trusted proxies for X-Forwarded-For header validation
TRUSTED_PROXIES = frozenset(
    p.strip() for p in os.getenv("ARAGORA_TRUSTED_PROXIES", "127.0.0.1,::1,localhost").split(",")
)

# Connection rate limiting per IP
WS_CONNECTIONS_PER_IP_PER_MINUTE = int(os.getenv("ARAGORA_WS_CONN_RATE", "30"))

# Token revalidation interval for long-lived connections (5 minutes)
WS_TOKEN_REVALIDATION_INTERVAL = 300.0

# Maximum connections per IP (concurrent)
WS_MAX_CONNECTIONS_PER_IP = int(os.getenv("ARAGORA_WS_MAX_PER_IP", "10"))

# Per-connection message rate limiting (messages per second)
WS_MESSAGES_PER_SECOND = int(os.getenv("ARAGORA_WS_MSG_RATE", "10"))
WS_MESSAGE_BURST_SIZE = int(os.getenv("ARAGORA_WS_MSG_BURST", "20"))


class WebSocketMessageRateLimiter:
    """Per-connection message rate limiter using token bucket algorithm.

    Limits the rate of incoming messages from a single WebSocket connection
    to prevent DoS attacks via message flooding.
    """

    def __init__(
        self,
        messages_per_second: float = WS_MESSAGES_PER_SECOND,
        burst_size: int = WS_MESSAGE_BURST_SIZE,
    ):
        """Initialize rate limiter.

        Args:
            messages_per_second: Maximum sustained message rate
            burst_size: Maximum burst of messages allowed
        """
        self.messages_per_second = messages_per_second
        self.burst_size = burst_size
        self._tokens = float(burst_size)
        self._last_update = time.time()

    def allow_message(self) -> bool:
        """Check if a message is allowed under the rate limit.

        Returns:
            True if message is allowed, False if rate limited
        """
        now = time.time()
        elapsed = now - self._last_update
        self._last_update = now

        # Refill tokens based on elapsed time
        self._tokens = min(self.burst_size, self._tokens + elapsed * self.messages_per_second)

        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


class DebateStreamServer(ServerBase):
    """
    WebSocket server broadcasting debate events to connected clients.

    Supports multiple concurrent nomic loop instances with view switching.
    Inherits common functionality from ServerBase including rate limiting,
    debate state caching, and active loops tracking.

    This server uses the pure websockets library for WebSocket handling.
    For combined HTTP+WebSocket support, use AiohttpUnifiedServer instead.

    Usage:
        server = DebateStreamServer(port=8765)
        hooks = create_arena_hooks(server.emitter)
        arena = Arena(env, agents, event_hooks=hooks)

        # In async context:
        asyncio.create_task(server.start())
        await arena.run()
    """

    # Cleanup interval for rate limiters
    _CLEANUP_INTERVAL = 100

    def __init__(self, host: str = "localhost", port: int = 8765):
        # Initialize base class with common functionality
        super().__init__()

        self.host = host
        self.port = port
        self.current_debate: Optional[dict] = None

        # WebSocket-specific: connection rate limiting per IP
        self._ws_conn_rate: dict[str, list[float]] = {}  # ip -> list of connection timestamps
        self._ws_conn_rate_lock = threading.Lock()
        self._ws_conn_per_ip: dict[str, int] = {}  # ip -> current connection count

        # Token revalidation tracking for long-lived connections
        self._ws_token_validated: dict[int, float] = {}  # ws_id -> last validation time

        # Per-connection message rate limiters
        self._ws_msg_limiters: dict[int, WebSocketMessageRateLimiter] = {}  # ws_id -> rate limiter

        # Async lock for client set operations (prevents race conditions during broadcast)
        self._clients_lock = asyncio.Lock()

    def _cleanup_stale_rate_limiters(self) -> None:
        """Remove rate limiters not accessed within TTL period."""
        self.cleanup_rate_limiters()

    def _extract_ws_token(self, websocket) -> Optional[str]:
        """Extract authentication token from WebSocket connection.

        Attempts to extract token from Authorization header only.
        Query parameter tokens are not accepted for security reasons
        (they appear in logs and browser history).

        Args:
            websocket: The WebSocket connection object

        Returns:
            The extracted token or None if not found
        """
        try:
            # Try newer websockets API first (websockets 10+)
            if hasattr(websocket, "request") and hasattr(websocket.request, "headers"):
                headers = websocket.request.headers
            elif hasattr(websocket, "request_headers"):
                headers = websocket.request_headers
            else:
                return None

            # Only accept Authorization: Bearer header (not query params)
            auth_header = headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                return auth_header[7:]

            return None
        except Exception as e:
            logger.debug(f"Could not extract WebSocket token: {e}")
            return None

    def _validate_ws_auth(self, websocket, loop_id: str = "") -> bool:
        """Validate WebSocket authentication.

        Args:
            websocket: The WebSocket connection object
            loop_id: Optional loop_id for token validation

        Returns:
            True if authenticated or auth is disabled, False otherwise
        """
        if not auth_config.enabled:
            return True

        token = self._extract_ws_token(websocket)
        if not token:
            return False

        return auth_config.validate_token(token, loop_id)

    def _extract_ws_ip(self, websocket) -> str:
        """Extract client IP address from WebSocket connection.

        Handles X-Forwarded-For header from trusted proxies.

        Args:
            websocket: The WebSocket connection object

        Returns:
            Client IP address string
        """
        try:
            # Get direct connection IP
            if hasattr(websocket, "remote_address"):
                direct_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"
            else:
                direct_ip = "unknown"

            # Check X-Forwarded-For if from trusted proxy
            if direct_ip in TRUSTED_PROXIES:
                headers = None
                if hasattr(websocket, "request") and hasattr(websocket.request, "headers"):
                    headers = websocket.request.headers
                elif hasattr(websocket, "request_headers"):
                    headers = websocket.request_headers

                if headers:
                    xff = headers.get("X-Forwarded-For", "")
                    if xff:
                        # Take first IP (original client)
                        return xff.split(",")[0].strip()

            return direct_ip
        except Exception as e:
            logger.debug(f"Could not extract client IP: {e}")
            return "unknown"

    def _check_ws_connection_rate(self, ip: str) -> tuple[bool, str]:
        """Check if IP is within WebSocket connection rate limit.

        Args:
            ip: Client IP address

        Returns:
            Tuple of (allowed: bool, error_message: str)
        """
        if ip == "unknown":
            return True, ""  # Can't rate limit unknown IPs

        now = time.time()
        window_start = now - 60.0  # 1 minute window

        with self._ws_conn_rate_lock:
            # Clean old timestamps
            if ip in self._ws_conn_rate:
                self._ws_conn_rate[ip] = [ts for ts in self._ws_conn_rate[ip] if ts > window_start]
            else:
                self._ws_conn_rate[ip] = []

            # Check rate
            if len(self._ws_conn_rate[ip]) >= WS_CONNECTIONS_PER_IP_PER_MINUTE:
                return (
                    False,
                    f"Connection rate limit exceeded ({WS_CONNECTIONS_PER_IP_PER_MINUTE}/min)",
                )

            # Check concurrent connections
            current_count = self._ws_conn_per_ip.get(ip, 0)
            if current_count >= WS_MAX_CONNECTIONS_PER_IP:
                return False, f"Max concurrent connections exceeded ({WS_MAX_CONNECTIONS_PER_IP})"

            # Record connection
            self._ws_conn_rate[ip].append(now)
            self._ws_conn_per_ip[ip] = current_count + 1

            return True, ""

    def _release_ws_connection(self, ip: str) -> None:
        """Release a WebSocket connection slot for an IP.

        Args:
            ip: Client IP address
        """
        if ip == "unknown":
            return

        with self._ws_conn_rate_lock:
            current = self._ws_conn_per_ip.get(ip, 0)
            if current > 0:
                self._ws_conn_per_ip[ip] = current - 1

    def _should_revalidate_token(self, ws_id: int) -> bool:
        """Check if token should be revalidated for a connection.

        Args:
            ws_id: WebSocket connection ID

        Returns:
            True if token needs revalidation
        """
        last_validated = self._ws_token_validated.get(ws_id, 0)
        return (time.time() - last_validated) > WS_TOKEN_REVALIDATION_INTERVAL

    def _mark_token_validated(self, ws_id: int) -> None:
        """Mark token as validated for a connection.

        Args:
            ws_id: WebSocket connection ID
        """
        self._ws_token_validated[ws_id] = time.time()

    def _cleanup_stale_entries(self) -> None:
        """Remove stale entries from all tracking dicts.

        Delegates to ServerBase.cleanup_all() and adds server-specific cleanup.
        """
        results = self.cleanup_all()
        total = sum(results.values())
        if total > 0:
            logger.debug(f"Cleaned up {total} stale entries")

    def _update_debate_state(self, event: StreamEvent) -> None:  # type: ignore[override]
        """Update cached debate state based on emitted events.

        Overrides ServerBase._update_debate_state with StreamEvent-specific handling.
        """
        loop_id = event.loop_id
        with self._debate_states_lock:
            if event.type == StreamEventType.DEBATE_START:
                # Enforce max size with LRU eviction (only evict ended debates)
                if len(self.debate_states) >= self.config.max_debate_states:
                    # Find oldest ended debate to evict
                    ended_states = [
                        (k, self._debate_states_last_access.get(k, 0))
                        for k, v in self.debate_states.items()
                        if v.get("ended")
                    ]
                    if ended_states:
                        oldest = min(ended_states, key=lambda x: x[1])[0]
                        self.debate_states.pop(oldest, None)
                        self._debate_states_last_access.pop(oldest, None)
                self.debate_states[loop_id] = {
                    "id": loop_id,
                    "task": event.data.get("task", ""),
                    "agents": event.data.get("agents", []),
                    "messages": [],
                    "consensus_reached": False,
                    "consensus_confidence": 0.0,
                    "consensus_answer": "",
                    "started_at": event.timestamp,
                    "rounds": 0,
                    "ended": False,
                    "duration": 0.0,
                }
                self._debate_states_last_access[loop_id] = time.time()
            elif event.type == StreamEventType.AGENT_MESSAGE:
                if loop_id in self.debate_states:
                    state = self.debate_states[loop_id]
                    state["messages"].append(
                        {
                            "agent": event.agent,
                            "role": event.data.get("role", "agent"),
                            "round": event.round,
                            "content": event.data.get("content", ""),
                        }
                    )
                    # Cap at last 1000 messages to allow full debate history without truncation
                    if len(state["messages"]) > 1000:
                        state["messages"] = state["messages"][-1000:]
                    self._debate_states_last_access[loop_id] = time.time()
            elif event.type == StreamEventType.CONSENSUS:
                if loop_id in self.debate_states:
                    state = self.debate_states[loop_id]
                    state["consensus_reached"] = event.data.get("reached", False)
                    state["consensus_confidence"] = event.data.get("confidence", 0.0)
                    state["consensus_answer"] = event.data.get("answer", "")
                    self._debate_states_last_access[loop_id] = time.time()
            elif event.type == StreamEventType.DEBATE_END:
                if loop_id in self.debate_states:
                    state = self.debate_states[loop_id]
                    state["ended"] = True
                    state["duration"] = event.data.get("duration", 0.0)
                    state["rounds"] = event.data.get("rounds", 0)
                    self._debate_states_last_access[loop_id] = time.time()
            elif event.type == StreamEventType.LOOP_UNREGISTER:
                self.debate_states.pop(loop_id, None)
                self._debate_states_last_access.pop(loop_id, None)

        # Update loop state for cycle/phase events (outside debate_states_lock)
        if event.type == StreamEventType.CYCLE_START:
            self.update_loop_state(loop_id, cycle=event.data.get("cycle"))
        elif event.type == StreamEventType.PHASE_START:
            self.update_loop_state(loop_id, phase=event.data.get("phase"))

    async def broadcast(self, event: StreamEvent) -> None:
        """Send event to all connected clients."""
        if not self.clients:
            return

        message = event.to_json()
        disconnected = set()

        # Take snapshot under lock to prevent race conditions
        async with self._clients_lock:
            clients_snapshot = list(self.clients)

        for client in clients_snapshot:
            try:
                await client.send(message)
            except Exception as e:
                logger.debug(f"Client disconnected during broadcast: {e}")
                disconnected.add(client)

        if disconnected:
            async with self._clients_lock:
                self.clients -= disconnected

    async def broadcast_batch(self, events: list[StreamEvent]) -> None:
        """Send multiple events to all connected clients in a single message.

        Batching reduces WebSocket overhead by sending events as a JSON array
        instead of individual messages. Frontends should handle both single
        events and arrays for backward compatibility.

        Args:
            events: List of events to broadcast together
        """
        if not self.clients or not events:
            return

        # Send as JSON array for batching efficiency
        message = json.dumps([e.to_dict() for e in events])
        disconnected = set()

        # Take snapshot under lock to prevent race conditions
        async with self._clients_lock:
            clients_snapshot = list(self.clients)

        for client in clients_snapshot:
            try:
                await client.send(message)
            except Exception as e:
                logger.debug(f"Client disconnected during batch broadcast: {e}")
                disconnected.add(client)

        if disconnected:
            async with self._clients_lock:
                self.clients -= disconnected

    def _group_events_by_agent(self, events: list[StreamEvent]) -> list[StreamEvent]:
        """Reorder events to group TOKEN_DELTA by agent.

        Non-token events maintain their relative order.
        Token events are grouped by agent for smoother frontend rendering,
        reducing visual interleaving during parallel agent generation.
        """
        result: list[StreamEvent] = []
        token_groups: dict[str, list[StreamEvent]] = {}

        for event in events:
            if event.type == StreamEventType.TOKEN_DELTA and event.agent:
                # Buffer token events by agent
                if event.agent not in token_groups:
                    token_groups[event.agent] = []
                token_groups[event.agent].append(event)
            else:
                # Flush buffered tokens before non-token event
                for agent_tokens in token_groups.values():
                    result.extend(agent_tokens)
                token_groups.clear()
                result.append(event)

        # Flush remaining tokens (grouped by agent)
        for agent_tokens in token_groups.values():
            result.extend(agent_tokens)

        return result

    async def _drain_loop(self) -> None:
        """Background task that drains the emitter queue and broadcasts.

        Uses batching to send multiple events in a single WebSocket message,
        reducing overhead and context switches by 5-10x for high-frequency
        event streams. Events are grouped by agent to prevent visual
        interleaving during parallel generation.
        """
        while self._running:
            # Collect all pending events into a batch
            # Use large batch size to prevent queue buildup with many concurrent agents
            events = list(self._emitter.drain(max_batch_size=1000))
            if events:
                # Group token events by agent for smoother streaming display
                grouped = self._group_events_by_agent(events)
                await self.broadcast_batch(grouped)
            await asyncio.sleep(0.03)  # Faster drain rate for high-volume debates

    def register_loop(self, loop_id: str, name: str, path: str = "") -> None:
        """Register a new nomic loop instance."""
        # Trigger periodic cleanup using base class config
        self._rate_limiter_cleanup_counter += 1
        if self._rate_limiter_cleanup_counter >= self.config.rate_limiter_cleanup_interval:
            self._rate_limiter_cleanup_counter = 0
            self._cleanup_stale_entries()

        instance = LoopInstance(
            loop_id=loop_id,
            name=name,
            started_at=time.time(),
            path=path,
        )
        # Use base class method for active loop management
        self.set_active_loop(loop_id, instance)
        with self._active_loops_lock:
            loop_count = len(self.active_loops)
        # Emit registration event
        self._emitter.emit(
            StreamEvent(
                type=StreamEventType.LOOP_REGISTER,
                data={
                    "loop_id": loop_id,
                    "name": name,
                    "started_at": instance.started_at,
                    "path": path,
                    "active_loops": loop_count,
                },
            )
        )

    def unregister_loop(self, loop_id: str) -> None:
        """Unregister a nomic loop instance."""
        removed = self.remove_active_loop(loop_id)
        if removed is None:
            return  # Loop not found, nothing to unregister
        with self._active_loops_lock:
            loop_count = len(self.active_loops)
        # Emit unregistration event
        self._emitter.emit(
            StreamEvent(
                type=StreamEventType.LOOP_UNREGISTER,
                data={
                    "loop_id": loop_id,
                    "active_loops": loop_count,
                },
            )
        )

    def update_loop_state(
        self, loop_id: str, cycle: int | None = None, phase: str | None = None
    ) -> None:
        """Update the state of an active loop instance."""
        with self._active_loops_lock:
            if loop_id in self.active_loops:
                if cycle is not None:
                    self.active_loops[loop_id].cycle = cycle
                if phase is not None:
                    self.active_loops[loop_id].phase = phase

    def get_loop_list(self) -> list[dict]:
        """Get list of active loops for client sync."""
        with self._active_loops_lock:
            return [
                {
                    "loop_id": loop.loop_id,
                    "name": loop.name,
                    "started_at": loop.started_at,
                    "cycle": loop.cycle,
                    "phase": loop.phase,
                    "path": loop.path,
                }
                for loop in self.active_loops.values()
            ]

    def _extract_ws_origin(self, websocket) -> str:
        """Extract Origin header from websocket (handles different library versions)."""
        try:
            if hasattr(websocket, "request") and hasattr(websocket.request, "headers"):
                return websocket.request.headers.get("Origin", "")
            elif hasattr(websocket, "request_headers"):
                return websocket.request_headers.get("Origin", "")
            return ""
        except Exception as e:
            logger.debug(f"Could not extract origin header: {e}")
            return ""

    def _validate_audience_payload(self, data: dict) -> tuple[Optional[dict], Optional[str]]:
        """Validate audience message payload.

        Returns:
            Tuple of (validated_payload, error_message). If error, payload is None.
        """
        payload = data.get("payload", {})
        if not isinstance(payload, dict):
            return None, "Invalid payload format"

        # Limit payload size to 10KB (DoS protection)
        try:
            payload_str = json.dumps(payload)
            if len(payload_str) > 10240:
                return None, "Payload too large (max 10KB)"
        except (TypeError, ValueError):
            return None, "Invalid payload structure"

        return payload, None

    def _process_audience_message(
        self,
        msg_type: str,
        loop_id: str,
        payload: dict,
        client_id: str,
    ) -> None:
        """Process validated audience vote/suggestion message."""
        audience_msg = AudienceMessage(
            type="vote" if msg_type == "user_vote" else "suggestion",
            loop_id=loop_id,
            payload=payload,
            user_id=client_id,
        )
        self.audience_inbox.put(audience_msg)

        # Emit event for dashboard visibility
        event_type = (
            StreamEventType.USER_VOTE
            if msg_type == "user_vote"
            else StreamEventType.USER_SUGGESTION
        )
        self._emitter.emit(
            StreamEvent(
                type=event_type,
                data=audience_msg.payload,
                loop_id=loop_id,
            )
        )

        # Emit updated audience metrics after each vote
        if msg_type == "user_vote":
            metrics = self.audience_inbox.get_summary(loop_id=loop_id)
            self._emitter.emit(
                StreamEvent(
                    type=StreamEventType.AUDIENCE_METRICS,
                    data=metrics,
                    loop_id=loop_id,
                )
            )

    async def _setup_connection(self, websocket) -> tuple[bool, str, str, int, bool, str | None]:
        """Set up WebSocket connection with validation.

        Returns:
            Tuple of (success, client_ip, client_id, ws_id, is_authenticated, ws_token)
            If success is False, connection has been closed with error.
        """
        client_ip = self._extract_ws_ip(websocket)

        # Check connection rate limit
        rate_allowed, rate_error = self._check_ws_connection_rate(client_ip)
        if not rate_allowed:
            logger.warning(f"[ws] Connection rejected for {client_ip}: {rate_error}")
            await websocket.close(4029, rate_error)
            return (False, client_ip, "", 0, False, None)

        # Validate origin
        origin = self._extract_ws_origin(websocket)
        if origin and origin not in WS_ALLOWED_ORIGINS:
            self._release_ws_connection(client_ip)
            await websocket.close(4003, "Origin not allowed")
            return (False, client_ip, "", 0, False, None)

        # Validate authentication
        is_authenticated = self._validate_ws_auth(websocket)
        ws_token = self._extract_ws_token(websocket)

        # Generate secure client ID
        ws_id = id(websocket)
        client_id = secrets.token_urlsafe(16)
        if len(self._client_ids) >= self.config.max_client_ids:
            self._client_ids.popitem(last=False)
        self._client_ids[ws_id] = client_id

        # NOTE: Client is added to self.clients AFTER _send_initial_state() in handler()
        # to ensure client receives consistent state before broadcasts

        if is_authenticated:
            self._mark_token_validated(ws_id)

        # Create per-connection message rate limiter
        self._ws_msg_limiters[ws_id] = WebSocketMessageRateLimiter()

        return (True, client_ip, client_id, ws_id, is_authenticated, ws_token)

    async def _send_initial_state(self, websocket, client_id: str, is_authenticated: bool) -> None:
        """Send initial connection state to client."""
        await websocket.send(
            json.dumps(
                {
                    "type": "connection_info",
                    "data": {
                        "authenticated": is_authenticated,
                        "client_id": client_id[:8] + "...",
                        "write_access": is_authenticated or not auth_config.enabled,
                    },
                }
            )
        )

        await websocket.send(
            json.dumps(
                {
                    "type": "loop_list",
                    "data": {
                        "loops": self.get_loop_list(),
                        "count": len(self.active_loops),
                    },
                }
            )
        )

        # Take snapshot under lock to prevent race conditions during iteration
        with self._debate_states_lock:
            states_snapshot = dict(self.debate_states)

        for loop_id, state in states_snapshot.items():
            await websocket.send(json.dumps({"type": "sync", "data": state}))

    async def _send_debate_state(self, websocket, debate_id: str) -> None:
        """Send current debate state to a newly subscribed client.

        Args:
            websocket: The WebSocket connection
            debate_id: The debate ID to send state for
        """
        # Get state under lock
        with self._debate_states_lock:
            state = self.debate_states.get(debate_id)

        if state:
            await websocket.send(json.dumps({
                "type": "sync",
                "data": state,
                "debate_id": debate_id,
            }))
            # If debate is in progress, resend debate_start for UI consistency
            if not state.get("ended"):
                await websocket.send(json.dumps({
                    "type": "debate_start",
                    "loop_id": debate_id,
                    "data": {
                        "debate_id": debate_id,
                        "status": "in_progress",
                        "task": state.get("task", ""),
                        "agents": state.get("agents", []),
                    }
                }))
        else:
            # No state yet, send waiting status
            await websocket.send(json.dumps({
                "type": "sync",
                "data": {"debate_id": debate_id, "status": "waiting"},
                "debate_id": debate_id,
            }))

    async def _parse_message(self, message: str) -> dict | None:
        """Parse and validate incoming WebSocket message.

        Returns parsed data dict, or None if invalid.
        """
        if len(message) > WS_MAX_MESSAGE_SIZE:
            logger.warning(f"[ws] Message too large from client: {len(message)} bytes")
            return None

        try:
            loop = asyncio.get_running_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, json.loads, message), timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.warning("[ws] JSON parsing timed out - possible DoS attempt")
        except json.JSONDecodeError as e:
            logger.warning(f"[ws] Invalid JSON from client: {e}")
        except RuntimeError as e:
            logger.error(f"[ws] Event loop error during JSON parsing: {e}")
        return None

    async def _handle_user_action(
        self,
        websocket,
        data: dict,
        ws_id: int,
        ws_token: str | None,
        is_authenticated: bool,
        client_id: str,
    ) -> bool:
        """Handle user vote or suggestion message.

        Returns:
            Updated is_authenticated value (may change if token invalidated).
        """
        msg_type = data.get("type")

        # Require authentication for write operations
        if not is_authenticated and auth_config.enabled:
            await websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "data": {
                            "message": "Authentication required for voting/suggestions",
                            "code": 401,
                        },
                    }
                )
            )
            return is_authenticated

        # Periodic token revalidation
        if is_authenticated and ws_token and self._should_revalidate_token(ws_id):
            if not auth_config.validate_token(ws_token):
                logger.warning(f"[ws] Token invalidated for client {client_id[:8]}...")
                await websocket.send(
                    json.dumps(
                        {
                            "type": "auth_revoked",
                            "data": {"message": "Token has been revoked or expired", "code": 401},
                        }
                    )
                )
                return False
            self._mark_token_validated(ws_id)

        stored_client_id = self._client_ids.get(ws_id, secrets.token_urlsafe(16))
        loop_id = data.get("loop_id", "")

        # Validate loop_id
        if not loop_id or loop_id not in self.active_loops:
            await websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "data": {"message": f"Invalid or inactive loop_id: {loop_id}"},
                    }
                )
            )
            return is_authenticated

        # Validate token for specific loop_id
        if auth_config.enabled and ws_token:
            is_valid, err_msg = auth_config.validate_token_for_loop(ws_token, loop_id)
            if not is_valid:
                await websocket.send(
                    json.dumps({"type": "error", "data": {"message": err_msg, "code": 403}})
                )
                return is_authenticated

        # Validate payload
        payload, error = self._validate_audience_payload(data)
        if error:
            await websocket.send(json.dumps({"type": "error", "data": {"message": error}}))
            return is_authenticated

        # Rate limiting
        should_cleanup = False
        with self._rate_limiters_lock:
            if stored_client_id not in self._rate_limiters:
                self._rate_limiters[stored_client_id] = TokenBucket(
                    rate_per_minute=10.0, burst_size=5
                )
            self._rate_limiter_last_access[stored_client_id] = time.time()
            rate_limiter = self._rate_limiters[stored_client_id]
            self._rate_limiter_cleanup_counter += 1
            if self._rate_limiter_cleanup_counter >= self._CLEANUP_INTERVAL:
                self._rate_limiter_cleanup_counter = 0
                should_cleanup = True
        if should_cleanup:
            self._cleanup_stale_rate_limiters()

        if not rate_limiter.consume(1):
            await websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "data": {"message": "Rate limited. Please wait before submitting again."},
                    }
                )
            )
            return is_authenticated

        # Process the message
        self._process_audience_message(msg_type, loop_id, payload, stored_client_id)
        await websocket.send(
            json.dumps(
                {"type": "ack", "data": {"message": "Message received", "msg_type": msg_type}}
            )
        )

        return is_authenticated

    async def _handle_wisdom_submission(
        self,
        websocket,
        data: dict,
        ws_id: int,
        ws_token: str | None,
        is_authenticated: bool,
        client_id: str,
    ) -> None:
        """Handle wisdom submission from audience.

        Wisdom submissions are stored and can be injected as fallback
        when AI agents fail to respond.
        """
        # Require authentication if enabled
        if not is_authenticated and auth_config.enabled:
            await websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "data": {
                            "message": "Authentication required for wisdom submissions",
                            "code": 401,
                        },
                    }
                )
            )
            return

        loop_id = data.get("loop_id", "")

        # Validate loop_id
        if not loop_id or loop_id not in self.active_loops:
            await websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "data": {"message": f"Invalid or inactive loop_id: {loop_id}"},
                    }
                )
            )
            return

        # Extract wisdom text
        text = data.get("text", "").strip()
        if not text or len(text) < 10:
            await websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "data": {"message": "Wisdom text must be at least 10 characters"},
                    }
                )
            )
            return

        if len(text) > 280:
            text = text[:280]

        # Get wisdom store
        try:
            from aragora.insights.store import InsightStore

            wisdom_store = InsightStore()

            wisdom_id = wisdom_store.add_wisdom_submission(
                loop_id=loop_id,
                wisdom_data={
                    "text": text,
                    "submitter_id": client_id[:16],
                    "context_tags": data.get("tags", []),
                },
            )

            await websocket.send(
                json.dumps(
                    {
                        "type": "wisdom_confirmed",
                        "data": {"wisdom_id": wisdom_id, "message": "Wisdom received"},
                    }
                )
            )

            logger.info(f"[wisdom] Stored submission {wisdom_id} for loop {loop_id}")

        except Exception as e:
            logger.error(f"[wisdom] Failed to store submission: {e}")
            await websocket.send(
                json.dumps(
                    {"type": "error", "data": {"message": "Failed to store wisdom submission"}}
                )
            )

    async def _cleanup_connection(self, client_ip: str, client_id: str, ws_id: int, websocket) -> None:
        """Clean up resources after connection closes."""
        async with self._clients_lock:
            self.clients.discard(websocket)
        logger.info(
            f"[ws] Client {client_id[:8]}... disconnected from {client_ip} "
            f"(remaining_clients={len(self.clients)})"
        )

        stored_client_id = self._client_ids.pop(ws_id, None)
        if stored_client_id:
            with self._rate_limiters_lock:
                self._rate_limiters.pop(stored_client_id, None)
                self._rate_limiter_last_access.pop(stored_client_id, None)

        self._release_ws_connection(client_ip)
        self._ws_token_validated.pop(ws_id, None)
        self._ws_msg_limiters.pop(ws_id, None)

    async def handler(self, websocket) -> None:
        """Handle a WebSocket connection with origin validation."""
        # Set up connection with validation
        (
            success,
            client_ip,
            client_id,
            ws_id,
            is_authenticated,
            ws_token,
        ) = await self._setup_connection(websocket)
        if not success:
            return

        logger.info(
            f"[ws] Client {client_id[:8]}... connected from {client_ip} "
            f"(authenticated={is_authenticated}, total_clients={len(self.clients)})"
        )
        try:
            # Send initial connection state
            await self._send_initial_state(websocket, client_id, is_authenticated)

            # Add client to broadcast set AFTER initial sync completes
            # This prevents receiving broadcasts before state is synchronized
            async with self._clients_lock:
                self.clients.add(websocket)

            # Handle incoming messages
            async for message in websocket:
                # Per-connection message rate limiting
                msg_limiter = self._ws_msg_limiters.get(ws_id)
                if msg_limiter and not msg_limiter.allow_message():
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "data": {
                                    "message": "Message rate limit exceeded. Please slow down.",
                                    "code": 429,
                                },
                            }
                        )
                    )
                    continue

                data = await self._parse_message(message)
                if data is None:
                    await websocket.send(
                        json.dumps({"type": "error", "data": {"message": "Invalid message format"}})
                    )
                    continue

                msg_type = data.get("type")

                if msg_type == "get_loops":
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "loop_list",
                                "data": {
                                    "loops": self.get_loop_list(),
                                    "count": len(self.active_loops),
                                },
                            }
                        )
                    )
                elif msg_type in ("user_vote", "user_suggestion"):
                    is_authenticated = await self._handle_user_action(
                        websocket, data, ws_id, ws_token, is_authenticated, client_id
                    )
                elif msg_type == "wisdom_submission":
                    await self._handle_wisdom_submission(
                        websocket, data, ws_id, ws_token, is_authenticated, client_id
                    )
                elif msg_type == "subscribe":
                    # Handle explicit subscribe to a specific debate
                    debate_id = data.get("debate_id")
                    if debate_id:
                        await self._send_debate_state(websocket, debate_id)
                    else:
                        await websocket.send(
                            json.dumps({
                                "type": "error",
                                "data": {"message": "subscribe requires debate_id"}
                            })
                        )

        except Exception as e:
            error_name = type(e).__name__
            if "ConnectionClosed" not in error_name and "ConnectionClosedOK" not in error_name:
                logger.error(
                    f"[ws] Unexpected error for client {client_id[:8]}...: {error_name}: {e}"
                )
        finally:
            await self._cleanup_connection(client_ip, client_id, ws_id, websocket)

    async def start(self) -> None:
        """Start the WebSocket server."""
        try:
            import websockets
        except ImportError:
            raise ImportError("websockets package required. Install with: pip install websockets")

        self._running = True
        # Store task reference and add error callback to prevent silent failures
        self._drain_task = asyncio.create_task(self._drain_loop())
        self._drain_task.add_done_callback(self._handle_drain_task_error)

        async with websockets.serve(
            self.handler,
            self.host,
            self.port,
            max_size=WS_MAX_MESSAGE_SIZE,
            ping_interval=30,  # Send ping every 30s
            ping_timeout=10,  # Close connection if no pong within 10s
            compression="deflate",  # Enable permessage-deflate for reduced bandwidth
        ):
            logger.info(
                f"WebSocket server: ws://{self.host}:{self.port} (max message size: {WS_MAX_MESSAGE_SIZE} bytes)"
            )
            await asyncio.Future()  # Run forever

    def stop(self) -> None:
        """Stop the server."""
        self._running = False

    def _handle_drain_task_error(self, task: asyncio.Task) -> None:
        """Handle errors from the drain loop task."""
        try:
            exc = task.exception()
            if exc is not None:
                logger.error(f"Drain loop task failed with exception: {exc}")
        except asyncio.CancelledError:
            pass  # Task was cancelled, not an error

    async def graceful_shutdown(self) -> None:
        """Gracefully close all client connections."""
        self._running = False
        # Close all connected clients under lock
        async with self._clients_lock:
            if self.clients:
                close_tasks = []
                for client in list(self.clients):
                    try:
                        close_tasks.append(client.close())
                    except Exception as e:
                        logger.debug(f"Error closing WebSocket client: {e}")
                if close_tasks:
                    await asyncio.gather(*close_tasks, return_exceptions=True)
                self.clients.clear()


__all__ = ["DebateStreamServer"]
