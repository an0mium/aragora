"""
WebSocket client for real-time debate streaming.

Provides async iteration over debate events with automatic
reconnection and heartbeat management.

Usage:
    from aragora.client import AragoraClient
    from aragora.client.websocket import DebateStream, stream_debate

    # Class-based API
    stream = DebateStream('ws://localhost:8765/ws', 'debate-123')
    await stream.connect()

    stream.on('agent_message', lambda event: print(event))
    stream.on('consensus', lambda event: print('Consensus!', event))

    # Or async iterator
    async for event in stream_debate('ws://localhost:8765/ws', 'debate-123'):
        print(event.type, event.data)
        if event.type == 'debate_end':
            break
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional
import time

logger = logging.getLogger(__name__)

# Event queue limits
MAX_EVENT_QUEUE_SIZE = 1000  # Prevent unbounded memory growth from event flooding


class DebateEventType(str, Enum):
    """Types of WebSocket stream events (matches server StreamEventType)."""

    # Control messages
    CONNECTION_INFO = "connection_info"
    LOOP_LIST = "loop_list"
    SYNC = "sync"
    ACK = "ack"
    AUTH_REVOKED = "auth_revoked"

    # Debate events
    DEBATE_START = "debate_start"
    ROUND_START = "round_start"
    AGENT_MESSAGE = "agent_message"
    CRITIQUE = "critique"
    VOTE = "vote"
    CONSENSUS = "consensus"
    DEBATE_END = "debate_end"

    # Token streaming events
    TOKEN_START = "token_start"
    TOKEN_DELTA = "token_delta"
    TOKEN_END = "token_end"

    # Nomic loop events
    CYCLE_START = "cycle_start"
    CYCLE_END = "cycle_end"
    PHASE_START = "phase_start"
    PHASE_END = "phase_end"
    TASK_START = "task_start"
    TASK_COMPLETE = "task_complete"
    TASK_RETRY = "task_retry"
    VERIFICATION_START = "verification_start"
    VERIFICATION_RESULT = "verification_result"
    COMMIT = "commit"
    BACKUP_CREATED = "backup_created"
    BACKUP_RESTORED = "backup_restored"
    ERROR = "error"
    LOG_MESSAGE = "log_message"

    # Multi-loop management
    LOOP_REGISTER = "loop_register"
    LOOP_UNREGISTER = "loop_unregister"

    # Audience participation
    USER_VOTE = "user_vote"
    USER_SUGGESTION = "user_suggestion"
    AUDIENCE_SUMMARY = "audience_summary"
    AUDIENCE_METRICS = "audience_metrics"
    AUDIENCE_DRAIN = "audience_drain"

    # Memory/learning events
    MEMORY_RECALL = "memory_recall"
    INSIGHT_EXTRACTED = "insight_extracted"

    # Rankings/leaderboard events
    MATCH_RECORDED = "match_recorded"
    LEADERBOARD_UPDATE = "leaderboard_update"
    GROUNDED_VERDICT = "grounded_verdict"
    MOMENT_DETECTED = "moment_detected"
    AGENT_ELO_UPDATED = "agent_elo_updated"

    # Claim verification events
    CLAIM_VERIFICATION_RESULT = "claim_verification_result"
    FORMAL_VERIFICATION_RESULT = "formal_verification_result"

    # Memory tier events
    MEMORY_TIER_PROMOTION = "memory_tier_promotion"
    MEMORY_TIER_DEMOTION = "memory_tier_demotion"

    # Graph debate events
    GRAPH_NODE_ADDED = "graph_node_added"
    GRAPH_BRANCH_CREATED = "graph_branch_created"
    GRAPH_BRANCH_MERGED = "graph_branch_merged"

    # Position tracking events
    FLIP_DETECTED = "flip_detected"

    # Feature integration events
    TRAIT_EMERGED = "trait_emerged"
    RISK_WARNING = "risk_warning"
    EVIDENCE_FOUND = "evidence_found"
    CALIBRATION_UPDATE = "calibration_update"
    GENESIS_EVOLUTION = "genesis_evolution"
    TRAINING_DATA_EXPORTED = "training_data_exported"

    # Rhetorical analysis events
    RHETORICAL_OBSERVATION = "rhetorical_observation"

    # Trickster/hollow consensus events
    HOLLOW_CONSENSUS = "hollow_consensus"
    TRICKSTER_INTERVENTION = "trickster_intervention"

    # Human intervention breakpoint events
    BREAKPOINT = "breakpoint"
    BREAKPOINT_RESOLVED = "breakpoint_resolved"

    # Mood/sentiment events
    MOOD_DETECTED = "mood_detected"
    MOOD_SHIFT = "mood_shift"
    DEBATE_ENERGY = "debate_energy"

    # Capability probe events
    PROBE_START = "probe_start"
    PROBE_RESULT = "probe_result"
    PROBE_COMPLETE = "probe_complete"

    # Deep audit events
    AUDIT_START = "audit_start"
    AUDIT_ROUND = "audit_round"
    AUDIT_FINDING = "audit_finding"
    AUDIT_CROSS_EXAM = "audit_cross_exam"
    AUDIT_VERDICT = "audit_verdict"

    # Telemetry events
    TELEMETRY_THOUGHT = "telemetry_thought"
    TELEMETRY_CAPABILITY = "telemetry_capability"
    TELEMETRY_REDACTION = "telemetry_redaction"
    TELEMETRY_DIAGNOSTIC = "telemetry_diagnostic"

    # Gauntlet events
    GAUNTLET_START = "gauntlet_start"
    GAUNTLET_PHASE = "gauntlet_phase"
    GAUNTLET_AGENT_ACTIVE = "gauntlet_agent_active"
    GAUNTLET_ATTACK = "gauntlet_attack"
    GAUNTLET_FINDING = "gauntlet_finding"
    GAUNTLET_PROBE = "gauntlet_probe"
    GAUNTLET_VERIFICATION = "gauntlet_verification"
    GAUNTLET_RISK = "gauntlet_risk"
    GAUNTLET_PROGRESS = "gauntlet_progress"
    GAUNTLET_VERDICT = "gauntlet_verdict"
    GAUNTLET_COMPLETE = "gauntlet_complete"

    # Analytics events
    UNCERTAINTY_ANALYSIS = "uncertainty_analysis"

    # Client keepalive / legacy
    PING = "ping"
    PONG = "pong"
    ROUND_END = "round_end"


@dataclass
class DebateEvent:
    """A debate event from WebSocket stream."""

    type: DebateEventType
    data: Dict[str, Any] = field(default_factory=dict)
    debate_id: str = ""
    timestamp: float = field(default_factory=time.time)
    round: int = 0
    agent: str = ""
    loop_id: str = ""
    seq: int = 0
    agent_seq: int = 0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DebateEvent":
        """Create from dictionary."""
        event_type = d.get("type", "error")
        try:
            event_type = DebateEventType(event_type)
        except ValueError:
            event_type = DebateEventType.ERROR
        data = d.get("data", {})
        if not isinstance(data, dict):
            data = {}

        timestamp = d.get("timestamp")
        if isinstance(timestamp, (int, float)):
            ts = float(timestamp)
        elif isinstance(timestamp, str):
            try:
                ts = datetime.fromisoformat(timestamp).timestamp()
            except ValueError:
                ts = time.time()
        else:
            ts = time.time()

        loop_id = d.get("loop_id", "") or data.get("loop_id", "")
        debate_id = d.get("debate_id", "") or data.get("debate_id", "") or loop_id

        return cls(
            type=event_type,
            debate_id=debate_id,
            timestamp=ts,
            data=data,
            round=int(d.get("round") or 0),
            agent=d.get("agent", "") or "",
            loop_id=loop_id,
            seq=int(d.get("seq") or 0),
            agent_seq=int(d.get("agent_seq") or 0),
        )


@dataclass
class WebSocketOptions:
    """Options for WebSocket connection."""

    reconnect: bool = True
    reconnect_interval: float = 1.0
    max_reconnect_attempts: int = 5
    heartbeat_interval: float = 30.0
    connect_timeout: float = 10.0


EventCallback = Callable[[DebateEvent], None]
ErrorCallback = Callable[[Exception], None]
CloseCallback = Callable[[int, str], None]


class DebateStream:
    """
    WebSocket client for streaming debate events.

    Provides both callback-based and async iterator APIs
    for consuming debate events in real-time.

    Example:
        stream = DebateStream('ws://localhost:8765/ws', 'debate-123')
        await stream.connect()

        # Callback API
        stream.on('agent_message', lambda e: print(e.data))

        # Or iterate
        async for event in stream:
            print(event)
    """

    def __init__(
        self,
        base_url: str,
        debate_id: str,
        options: Optional[WebSocketOptions] = None,
    ):
        """
        Initialize debate stream.

        Args:
            base_url: WebSocket server URL (ws:// or wss://)
            debate_id: ID of debate to stream
            options: Connection options
        """
        self.debate_id = debate_id
        self.options = options or WebSocketOptions()
        self.url = self._build_url(base_url, debate_id)

        self._ws: Optional[Any] = None
        self._reconnect_attempts = 0
        self._reconnect_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._is_closing = False

        self._event_callbacks: Dict[str, List[EventCallback]] = {}
        self._error_callbacks: List[ErrorCallback] = []
        self._close_callbacks: List[CloseCallback] = []

        self._event_queue: asyncio.Queue[DebateEvent] = asyncio.Queue(maxsize=MAX_EVENT_QUEUE_SIZE)

    def _build_url(self, base_url: str, debate_id: str) -> str:
        """Build WebSocket URL."""
        ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://").rstrip("/")
        if ws_url.endswith("/ws"):
            return ws_url
        return f"{ws_url}/ws"

    def _event_loop_id(self, event: DebateEvent) -> str:
        """Resolve loop/debate id for filtering."""
        if event.loop_id:
            return event.loop_id
        if event.debate_id:
            return event.debate_id
        data = event.data if isinstance(event.data, dict) else {}
        loop_id = data.get("loop_id") or data.get("debate_id") or ""
        return loop_id if isinstance(loop_id, str) else ""

    def _should_emit(self, event: DebateEvent) -> bool:
        if not self.debate_id:
            return True
        event_loop_id = self._event_loop_id(event)
        return not event_loop_id or event_loop_id == self.debate_id

    async def connect(self) -> None:
        """
        Connect to the WebSocket server.

        Raises:
            ConnectionError: If connection fails
        """
        try:
            import websockets
        except ImportError:
            raise ImportError("websockets package required. Install with: pip install websockets")

        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(
                    self.url,
                    ping_interval=self.options.heartbeat_interval,
                    ping_timeout=10,
                    close_timeout=5,
                ),
                timeout=self.options.connect_timeout,
            )
            self._reconnect_attempts = 0
            self._is_closing = False

            # Start message receiver
            asyncio.create_task(self._receive_loop())

            logger.debug(f"Connected to {self.url}")

        except asyncio.TimeoutError:
            raise ConnectionError(f"Connection timeout to {self.url}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self.url}: {e}")

    async def disconnect(self) -> None:
        """Disconnect from the WebSocket server."""
        self._is_closing = True

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None

        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None

        if self._ws:
            await self._ws.close()
            self._ws = None

    def on(self, event_type: str, callback: EventCallback) -> "DebateStream":
        """
        Subscribe to an event type.

        Args:
            event_type: Event type to subscribe to, or '*' for all events
            callback: Callback function receiving DebateEvent

        Returns:
            Self for chaining
        """
        if event_type not in self._event_callbacks:
            self._event_callbacks[event_type] = []
        self._event_callbacks[event_type].append(callback)
        return self

    def on_error(self, callback: ErrorCallback) -> "DebateStream":
        """Subscribe to errors."""
        self._error_callbacks.append(callback)
        return self

    def on_close(self, callback: CloseCallback) -> "DebateStream":
        """Subscribe to connection close."""
        self._close_callbacks.append(callback)
        return self

    def off(self, event_type: str, callback: EventCallback) -> "DebateStream":
        """Unsubscribe from an event type."""
        if event_type in self._event_callbacks:
            try:
                self._event_callbacks[event_type].remove(callback)
            except ValueError:
                pass
        return self

    async def send(self, data: Dict[str, Any]) -> None:
        """Send a message to the server."""
        if self._ws:
            await self._ws.send(json.dumps(data))

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._ws is not None and self._ws.open

    async def _receive_loop(self) -> None:
        """Receive messages from WebSocket."""
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                    event = DebateEvent.from_dict(data)

                    if not self._should_emit(event):
                        continue

                    # Emit to callbacks
                    self._emit_event(event)

                    # Add to queue for async iteration (drop if full to prevent blocking)
                    try:
                        self._event_queue.put_nowait(event)
                    except asyncio.QueueFull:
                        logger.warning("Event queue full, dropping event")

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON: {message[:100]}")
                except Exception as e:
                    logger.warning(f"Error processing message: {e}")

        except Exception as e:
            if not self._is_closing:
                self._emit_error(e)
                self._emit_close(1006, str(e))

                if self.options.reconnect:
                    await self._attempt_reconnect()

    def _emit_event(self, event: DebateEvent) -> None:
        """Emit event to callbacks."""
        # Type-specific callbacks
        type_key = event.type.value if isinstance(event.type, Enum) else event.type
        if type_key in self._event_callbacks:
            for callback in self._event_callbacks[type_key]:
                try:
                    callback(event)
                except Exception as e:
                    logger.warning(f"Callback error: {e}")

        # Wildcard callbacks
        if "*" in self._event_callbacks:
            for callback in self._event_callbacks["*"]:
                try:
                    callback(event)
                except Exception as e:
                    logger.warning(f"Callback error: {e}")

    def _emit_error(self, error: Exception) -> None:
        """Emit error to callbacks."""
        for callback in self._error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.warning(f"Error callback error: {e}")

    def _emit_close(self, code: int, reason: str) -> None:
        """Emit close to callbacks."""
        for callback in self._close_callbacks:
            try:
                callback(code, reason)
            except Exception as e:
                logger.warning(f"Close callback error: {e}")

    async def _attempt_reconnect(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        if self._reconnect_attempts >= self.options.max_reconnect_attempts:
            self._emit_error(ConnectionError("Max reconnect attempts reached"))
            return

        self._reconnect_attempts += 1
        delay = self.options.reconnect_interval * (2 ** (self._reconnect_attempts - 1))

        logger.info(f"Reconnecting in {delay}s (attempt {self._reconnect_attempts})")
        await asyncio.sleep(delay)

        try:
            await self.connect()
        except Exception as e:
            logger.warning(f"Reconnect failed: {e}")
            await self._attempt_reconnect()

    def __aiter__(self):
        """Async iterator support."""
        return self

    async def __anext__(self) -> DebateEvent:
        """Get next event."""
        if self._is_closing and self._event_queue.empty():
            raise StopAsyncIteration

        try:
            event = await asyncio.wait_for(
                self._event_queue.get(),
                timeout=self.options.heartbeat_interval * 2,
            )
            return event
        except asyncio.TimeoutError:
            if not self.is_connected:
                raise StopAsyncIteration
            # Return a synthetic ping event to indicate stream is alive
            return DebateEvent(
                type=DebateEventType.PING,
                debate_id=self.debate_id,
                timestamp=time.time(),
                data={"keepalive": True},
                loop_id=self.debate_id,
            )


async def stream_debate(
    base_url: str,
    debate_id: str,
    options: Optional[WebSocketOptions] = None,
) -> AsyncGenerator[DebateEvent, None]:
    """
    Stream debate events as an async generator.

    This is a convenience function that handles connection
    management automatically.

    Args:
        base_url: WebSocket server URL
        debate_id: ID of debate to stream
        options: Connection options

    Yields:
        DebateEvent for each event from the debate

    Example:
        async for event in stream_debate('ws://localhost:8765/ws', 'debate-123'):
            print(event.type, event.data)
            if event.type == DebateEventType.DEBATE_END:
                break
    """
    stream = DebateStream(base_url, debate_id, options)

    try:
        await stream.connect()

        async for event in stream:
            yield event

            if event.type == DebateEventType.DEBATE_END:
                break

    finally:
        await stream.disconnect()


__all__ = [
    "DebateEventType",
    "DebateEvent",
    "WebSocketOptions",
    "DebateStream",
    "stream_debate",
]
