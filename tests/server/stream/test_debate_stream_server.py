"""
Tests for DebateStreamServer WebSocket streaming.

Tests cover:
- Server initialization and configuration
- WebSocket message rate limiting
- Client connection rate limiting
- Authentication validation
- Event broadcasting with subscription filtering
- Debate state updates
- Loop registration/unregistration
- Audience message processing
- Token grouping for batch delivery
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from aragora.events.types import StreamEvent, StreamEventType
from aragora.server.stream.debate_stream_server import (
    DebateStreamServer,
    WebSocketMessageRateLimiter,
)


# ===========================================================================
# Test WebSocketMessageRateLimiter
# ===========================================================================


class TestWebSocketMessageRateLimiter:
    """Tests for the per-connection message rate limiter."""

    def test_allows_initial_burst(self):
        """Should allow burst_size messages initially."""
        limiter = WebSocketMessageRateLimiter(
            messages_per_second=10.0,
            burst_size=5,
        )
        # Should allow all 5 burst messages
        for _ in range(5):
            assert limiter.allow_message() is True

    def test_rejects_after_burst_exhausted(self):
        """Should reject messages after burst is exhausted."""
        limiter = WebSocketMessageRateLimiter(
            messages_per_second=10.0,
            burst_size=3,
        )
        # Exhaust burst
        for _ in range(3):
            limiter.allow_message()
        # Next should be rejected
        assert limiter.allow_message() is False

    def test_tokens_replenish_over_time(self):
        """Should replenish tokens over time."""
        limiter = WebSocketMessageRateLimiter(
            messages_per_second=100.0,  # Fast replenishment for test
            burst_size=1,
        )
        # Use the one token
        assert limiter.allow_message() is True
        assert limiter.allow_message() is False

        # Simulate time passing (manually set _last_update)
        limiter._last_update = time.time() - 0.1  # 0.1 seconds ago

        # Now should have replenished ~10 tokens (100 * 0.1)
        assert limiter.allow_message() is True

    def test_tokens_capped_at_burst_size(self):
        """Should not accumulate more than burst_size tokens."""
        limiter = WebSocketMessageRateLimiter(
            messages_per_second=1000.0,
            burst_size=5,
        )
        # Even after a long time, only burst_size tokens available
        limiter._last_update = time.time() - 100  # 100 seconds ago

        # Check allows burst_size
        for _ in range(5):
            assert limiter.allow_message() is True
        assert limiter.allow_message() is False

    @patch.dict("os.environ", {"ARAGORA_DISABLE_ALL_RATE_LIMITS": "true"})
    def test_bypassed_when_rate_limits_disabled(self):
        """Should always allow when rate limiting is globally disabled."""
        # Need to reimport to pick up the env change
        from importlib import reload

        import aragora.server.stream.debate_stream_server as module

        reload(module)
        limiter = module.WebSocketMessageRateLimiter(
            messages_per_second=0.001,  # Very restrictive
            burst_size=1,
        )
        # Should still allow due to global disable
        for _ in range(100):
            assert limiter.allow_message() is True


# ===========================================================================
# Test DebateStreamServer Initialization
# ===========================================================================


class TestDebateStreamServerInit:
    """Tests for server initialization."""

    def test_default_initialization(self):
        """Should initialize with default host and port."""
        server = DebateStreamServer()
        assert server.host == "localhost"
        assert server.port == 8765
        assert server.current_debate is None

    def test_custom_host_port(self):
        """Should accept custom host and port."""
        server = DebateStreamServer(host="0.0.0.0", port=9000)
        assert server.host == "0.0.0.0"
        assert server.port == 9000

    def test_initializes_tracking_dicts(self):
        """Should initialize all tracking dictionaries."""
        server = DebateStreamServer()
        assert server._ws_conn_rate == {}
        assert server._ws_conn_per_ip == {}
        assert server._ws_token_validated == {}
        assert server._ws_msg_limiters == {}
        assert server._client_subscriptions == {}


# ===========================================================================
# Test Connection Rate Limiting
# ===========================================================================


class TestConnectionRateLimiting:
    """Tests for IP-based connection rate limiting."""

    @patch.dict("os.environ", {"ARAGORA_DISABLE_ALL_RATE_LIMITS": ""})
    def test_allows_first_connection(self, monkeypatch):
        """Should allow first connection from any IP."""
        # Ensure rate limiting is enabled for this test
        monkeypatch.setattr(
            "aragora.server.stream.debate_stream_server.WS_RATE_LIMITING_DISABLED",
            False,
        )
        server = DebateStreamServer()
        allowed, error = server._check_ws_connection_rate("192.168.1.100")
        assert allowed is True
        assert error == ""

    @patch.dict("os.environ", {"ARAGORA_DISABLE_ALL_RATE_LIMITS": ""})
    def test_tracks_connection_timestamps(self, monkeypatch):
        """Should record connection timestamps per IP."""
        monkeypatch.setattr(
            "aragora.server.stream.debate_stream_server.WS_RATE_LIMITING_DISABLED",
            False,
        )
        server = DebateStreamServer()
        server._check_ws_connection_rate("192.168.1.100")
        assert "192.168.1.100" in server._ws_conn_rate
        assert len(server._ws_conn_rate["192.168.1.100"]) == 1

    @patch.dict("os.environ", {"ARAGORA_DISABLE_ALL_RATE_LIMITS": ""})
    def test_tracks_concurrent_connections(self, monkeypatch):
        """Should track concurrent connection count per IP."""
        monkeypatch.setattr(
            "aragora.server.stream.debate_stream_server.WS_RATE_LIMITING_DISABLED",
            False,
        )
        server = DebateStreamServer()
        server._check_ws_connection_rate("192.168.1.100")
        assert server._ws_conn_per_ip.get("192.168.1.100") == 1

    @patch.dict("os.environ", {"ARAGORA_DISABLE_ALL_RATE_LIMITS": ""})
    def test_releases_connection_slot(self, monkeypatch):
        """Should release connection slot when called."""
        monkeypatch.setattr(
            "aragora.server.stream.debate_stream_server.WS_RATE_LIMITING_DISABLED",
            False,
        )
        server = DebateStreamServer()
        server._check_ws_connection_rate("192.168.1.100")
        assert server._ws_conn_per_ip.get("192.168.1.100") == 1

        server._release_ws_connection("192.168.1.100")
        assert server._ws_conn_per_ip.get("192.168.1.100") == 0

    def test_ignores_unknown_ip(self):
        """Should allow 'unknown' IPs without tracking."""
        server = DebateStreamServer()
        allowed, error = server._check_ws_connection_rate("unknown")
        assert allowed is True
        assert "unknown" not in server._ws_conn_rate


# ===========================================================================
# Test Token Revalidation
# ===========================================================================


class TestTokenRevalidation:
    """Tests for long-lived connection token revalidation."""

    def test_should_revalidate_when_never_validated(self):
        """Should require revalidation for new connections."""
        server = DebateStreamServer()
        assert server._should_revalidate_token(12345) is True

    def test_should_not_revalidate_recently_validated(self):
        """Should not require revalidation for recently validated tokens."""
        server = DebateStreamServer()
        ws_id = 12345
        server._mark_token_validated(ws_id)
        assert server._should_revalidate_token(ws_id) is False

    def test_should_revalidate_after_interval(self):
        """Should require revalidation after interval expires."""
        server = DebateStreamServer()
        ws_id = 12345
        # Simulate validation that happened 10 minutes ago
        server._ws_token_validated[ws_id] = time.time() - 600
        assert server._should_revalidate_token(ws_id) is True


# ===========================================================================
# Test Authentication
# ===========================================================================


class TestWebSocketAuthentication:
    """Tests for WebSocket authentication."""

    def test_extract_token_from_bearer_header(self):
        """Should extract token from Authorization: Bearer header."""
        server = DebateStreamServer()
        mock_ws = MagicMock()
        mock_ws.request = MagicMock()
        mock_ws.request.headers = {"Authorization": "Bearer test-token-123"}

        token = server._extract_ws_token(mock_ws)
        assert token == "test-token-123"

    def test_extract_token_from_legacy_headers(self):
        """Should extract token from legacy request_headers attribute."""
        server = DebateStreamServer()
        mock_ws = MagicMock()
        del mock_ws.request  # Remove new API
        mock_ws.request_headers = {"Authorization": "Bearer legacy-token"}

        token = server._extract_ws_token(mock_ws)
        assert token == "legacy-token"

    def test_returns_none_without_auth_header(self):
        """Should return None when no Authorization header."""
        server = DebateStreamServer()
        mock_ws = MagicMock()
        mock_ws.request = MagicMock()
        mock_ws.request.headers = {}

        token = server._extract_ws_token(mock_ws)
        assert token is None

    def test_returns_none_for_non_bearer_auth(self):
        """Should return None for non-Bearer authentication."""
        server = DebateStreamServer()
        mock_ws = MagicMock()
        mock_ws.request = MagicMock()
        mock_ws.request.headers = {"Authorization": "Basic dXNlcjpwYXNz"}

        token = server._extract_ws_token(mock_ws)
        assert token is None

    @patch("aragora.server.stream.debate_stream_server.auth_config")
    def test_validate_auth_returns_true_when_disabled(self, mock_auth):
        """Should return True when auth is disabled."""
        mock_auth.enabled = False
        server = DebateStreamServer()
        mock_ws = MagicMock()

        result = server._validate_ws_auth(mock_ws)
        assert result is True

    @patch("aragora.server.stream.debate_stream_server.auth_config")
    def test_validate_auth_checks_token_when_enabled(self, mock_auth):
        """Should validate token when auth is enabled."""
        mock_auth.enabled = True
        mock_auth.validate_token = MagicMock(return_value=True)

        server = DebateStreamServer()
        mock_ws = MagicMock()
        mock_ws.request = MagicMock()
        mock_ws.request.headers = {"Authorization": "Bearer valid-token"}

        result = server._validate_ws_auth(mock_ws)
        assert result is True
        mock_auth.validate_token.assert_called_once_with("valid-token", "")


# ===========================================================================
# Test IP Extraction
# ===========================================================================


class TestIPExtraction:
    """Tests for client IP address extraction."""

    def test_extracts_direct_ip(self):
        """Should extract IP from remote_address."""
        server = DebateStreamServer()
        mock_ws = MagicMock()
        mock_ws.remote_address = ("192.168.1.50", 54321)

        ip = server._extract_ws_ip(mock_ws)
        assert ip == "192.168.1.50"

    def test_uses_xff_from_trusted_proxy(self):
        """Should use X-Forwarded-For from trusted proxy."""
        server = DebateStreamServer()
        mock_ws = MagicMock()
        mock_ws.remote_address = ("127.0.0.1", 54321)  # Trusted proxy
        mock_ws.request = MagicMock()
        mock_ws.request.headers = {"X-Forwarded-For": "203.0.113.50, 10.0.0.1"}

        ip = server._extract_ws_ip(mock_ws)
        assert ip == "203.0.113.50"  # First IP (original client)

    def test_ignores_xff_from_untrusted_source(self):
        """Should ignore X-Forwarded-For from untrusted sources."""
        server = DebateStreamServer()
        mock_ws = MagicMock()
        mock_ws.remote_address = ("192.168.1.50", 54321)  # Not a trusted proxy
        mock_ws.request = MagicMock()
        mock_ws.request.headers = {"X-Forwarded-For": "spoofed.ip.address"}

        ip = server._extract_ws_ip(mock_ws)
        assert ip == "192.168.1.50"  # Uses direct IP, not XFF


# ===========================================================================
# Test Debate State Updates
# ===========================================================================


class TestDebateStateUpdates:
    """Tests for debate state caching and updates."""

    def test_updates_state_on_debate_start(self):
        """Should create debate state on DEBATE_START event."""
        server = DebateStreamServer()
        event = StreamEvent(
            type=StreamEventType.DEBATE_START,
            data={
                "task": "Test question",
                "agents": ["claude", "gpt4"],
            },
            loop_id="debate-1",
        )

        server._update_debate_state(event)

        assert "debate-1" in server.debate_states
        state = server.debate_states["debate-1"]
        assert state["task"] == "Test question"
        assert state["agents"] == ["claude", "gpt4"]
        assert state["ended"] is False

    def test_appends_messages_on_agent_message(self):
        """Should append messages to debate state."""
        server = DebateStreamServer()
        # First create the debate
        server._update_debate_state(
            StreamEvent(
                type=StreamEventType.DEBATE_START,
                data={"task": "Test", "agents": []},
                loop_id="debate-1",
            )
        )

        # Then add a message
        server._update_debate_state(
            StreamEvent(
                type=StreamEventType.AGENT_MESSAGE,
                data={"content": "Hello world", "role": "proposer"},
                loop_id="debate-1",
                agent="claude",
                round=1,
            )
        )

        state = server.debate_states["debate-1"]
        assert len(state["messages"]) == 1
        assert state["messages"][0]["agent"] == "claude"
        assert state["messages"][0]["content"] == "Hello world"

    def test_updates_consensus_on_consensus_event(self):
        """Should update consensus info on CONSENSUS event."""
        server = DebateStreamServer()
        # Create debate
        server._update_debate_state(
            StreamEvent(
                type=StreamEventType.DEBATE_START,
                data={"task": "Test", "agents": []},
                loop_id="debate-1",
            )
        )

        # Add consensus
        server._update_debate_state(
            StreamEvent(
                type=StreamEventType.CONSENSUS,
                data={
                    "reached": True,
                    "confidence": 0.95,
                    "answer": "The consensus answer",
                },
                loop_id="debate-1",
            )
        )

        state = server.debate_states["debate-1"]
        assert state["consensus_reached"] is True
        assert state["consensus_confidence"] == 0.95
        assert state["consensus_answer"] == "The consensus answer"

    def test_marks_ended_on_debate_end(self):
        """Should mark debate as ended on DEBATE_END event."""
        server = DebateStreamServer()
        # Create debate
        server._update_debate_state(
            StreamEvent(
                type=StreamEventType.DEBATE_START,
                data={"task": "Test", "agents": []},
                loop_id="debate-1",
            )
        )

        # End debate
        server._update_debate_state(
            StreamEvent(
                type=StreamEventType.DEBATE_END,
                data={"duration": 45.0, "rounds": 3},
                loop_id="debate-1",
            )
        )

        state = server.debate_states["debate-1"]
        assert state["ended"] is True
        assert state["duration"] == 45.0
        assert state["rounds"] == 3

    def test_removes_state_on_loop_unregister(self):
        """Should remove debate state on LOOP_UNREGISTER event."""
        server = DebateStreamServer()
        # Create debate
        server._update_debate_state(
            StreamEvent(
                type=StreamEventType.DEBATE_START,
                data={"task": "Test", "agents": []},
                loop_id="debate-1",
            )
        )
        assert "debate-1" in server.debate_states

        # Unregister
        server._update_debate_state(
            StreamEvent(
                type=StreamEventType.LOOP_UNREGISTER,
                data={},
                loop_id="debate-1",
            )
        )

        assert "debate-1" not in server.debate_states


# ===========================================================================
# Test Broadcasting
# ===========================================================================


class TestBroadcasting:
    """Tests for event broadcasting to WebSocket clients."""

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_all_clients(self):
        """Should send event to all connected clients."""
        server = DebateStreamServer()

        # Create mock clients
        client1 = AsyncMock()
        client2 = AsyncMock()
        server.clients = {client1, client2}
        server._clients_lock = asyncio.Lock()

        event = StreamEvent(
            type=StreamEventType.DEBATE_START,
            data={"task": "Test"},
            loop_id="debate-1",
        )

        await server.broadcast(event)

        # Both clients should receive the message
        assert client1.send.called
        assert client2.send.called

    @pytest.mark.asyncio
    async def test_broadcast_filters_by_subscription(self):
        """Should only send to clients subscribed to the debate."""
        server = DebateStreamServer()

        # Create mock clients
        client1 = AsyncMock()
        client2 = AsyncMock()
        server.clients = {client1, client2}
        server._clients_lock = asyncio.Lock()

        # Subscribe client1 to debate-1, client2 to debate-2
        server._client_subscriptions[id(client1)] = "debate-1"
        server._client_subscriptions[id(client2)] = "debate-2"

        event = StreamEvent(
            type=StreamEventType.AGENT_MESSAGE,
            data={"content": "Test"},
            loop_id="debate-1",
        )

        await server.broadcast(event)

        # Only client1 should receive (subscribed to debate-1)
        assert client1.send.called
        assert not client2.send.called

    @pytest.mark.asyncio
    async def test_broadcast_removes_disconnected_clients(self):
        """Should remove clients that fail to receive."""
        server = DebateStreamServer()

        # Create mock clients - one that fails
        client1 = AsyncMock()
        client2 = AsyncMock()
        client2.send.side_effect = ConnectionError("Disconnected")

        server.clients = {client1, client2}
        server._clients_lock = asyncio.Lock()

        event = StreamEvent(
            type=StreamEventType.DEBATE_START,
            data={},
            loop_id="debate-1",
        )

        await server.broadcast(event)

        # Failed client should be removed
        assert client1 in server.clients
        assert client2 not in server.clients


# ===========================================================================
# Test Token Grouping
# ===========================================================================


class TestTokenGrouping:
    """Tests for grouping TOKEN_DELTA events by agent."""

    def test_groups_tokens_by_agent(self):
        """Should group token events by agent for smooth rendering."""
        server = DebateStreamServer()

        events = [
            StreamEvent(
                type=StreamEventType.TOKEN_DELTA,
                data={"token": "Hello"},
                agent="claude",
                agent_seq=1,
            ),
            StreamEvent(
                type=StreamEventType.TOKEN_DELTA,
                data={"token": "World"},
                agent="gpt4",
                agent_seq=1,
            ),
            StreamEvent(
                type=StreamEventType.TOKEN_DELTA,
                data={"token": " there"},
                agent="claude",
                agent_seq=2,
            ),
        ]

        grouped = server._group_events_by_agent(events)

        # Claude's tokens should be together
        claude_indices = [i for i, e in enumerate(grouped) if e.agent == "claude"]
        assert claude_indices == [0, 1] or claude_indices == [1, 2]

    def test_non_token_events_flush_buffer(self):
        """Should flush token buffer when non-token event encountered."""
        server = DebateStreamServer()

        events = [
            StreamEvent(
                type=StreamEventType.TOKEN_DELTA,
                data={"token": "A"},
                agent="claude",
                agent_seq=1,
            ),
            StreamEvent(
                type=StreamEventType.ROUND_START,  # Non-token event
                data={"round": 1},
            ),
            StreamEvent(
                type=StreamEventType.TOKEN_DELTA,
                data={"token": "B"},
                agent="claude",
                agent_seq=2,
            ),
        ]

        grouped = server._group_events_by_agent(events)

        # ROUND_START should be between the two token events
        assert grouped[0].type == StreamEventType.TOKEN_DELTA
        assert grouped[1].type == StreamEventType.ROUND_START
        assert grouped[2].type == StreamEventType.TOKEN_DELTA

    def test_sorts_tokens_by_agent_seq(self):
        """Should sort token events within a group by agent_seq."""
        server = DebateStreamServer()

        events = [
            StreamEvent(
                type=StreamEventType.TOKEN_DELTA,
                data={"token": "3"},
                agent="claude",
                agent_seq=3,
            ),
            StreamEvent(
                type=StreamEventType.TOKEN_DELTA,
                data={"token": "1"},
                agent="claude",
                agent_seq=1,
            ),
            StreamEvent(
                type=StreamEventType.TOKEN_DELTA,
                data={"token": "2"},
                agent="claude",
                agent_seq=2,
            ),
        ]

        grouped = server._group_events_by_agent(events)

        # Should be ordered 1, 2, 3
        assert grouped[0].data["token"] == "1"
        assert grouped[1].data["token"] == "2"
        assert grouped[2].data["token"] == "3"


# ===========================================================================
# Test Loop Registration
# ===========================================================================


class TestLoopRegistration:
    """Tests for nomic loop registration and management."""

    def test_register_loop_adds_to_active_loops(self):
        """Should add loop to active loops on registration."""
        server = DebateStreamServer()
        # Mock emitter to avoid actual event emission
        server._emitter = MagicMock()

        server.register_loop("loop-1", "Test Loop", "/path/to/loop")

        assert "loop-1" in server.active_loops
        loop = server.active_loops["loop-1"]
        assert loop.name == "Test Loop"
        assert loop.path == "/path/to/loop"

    def test_unregister_loop_removes_from_active_loops(self):
        """Should remove loop from active loops on unregistration."""
        server = DebateStreamServer()
        server._emitter = MagicMock()

        server.register_loop("loop-1", "Test Loop")
        assert "loop-1" in server.active_loops

        server.unregister_loop("loop-1")
        assert "loop-1" not in server.active_loops

    def test_get_loop_list_returns_all_active(self):
        """Should return list of all active loops."""
        server = DebateStreamServer()
        server._emitter = MagicMock()

        server.register_loop("loop-1", "Loop 1")
        server.register_loop("loop-2", "Loop 2")

        loops = server.get_loop_list()
        assert len(loops) == 2
        loop_ids = {loop["loop_id"] for loop in loops}
        assert loop_ids == {"loop-1", "loop-2"}


# ===========================================================================
# Test Audience Payload Validation
# ===========================================================================


class TestAudiencePayloadValidation:
    """Tests for audience message payload validation."""

    def test_validates_valid_payload(self):
        """Should accept valid payload."""
        server = DebateStreamServer()
        data = {"payload": {"vote": "agree", "confidence": 0.8}}

        payload, error = server._validate_audience_payload(data)
        assert payload is not None
        assert error is None
        assert payload["vote"] == "agree"

    def test_rejects_non_dict_payload(self):
        """Should reject non-dict payload."""
        server = DebateStreamServer()
        data = {"payload": "not a dict"}

        payload, error = server._validate_audience_payload(data)
        assert payload is None
        assert error == "Invalid payload format"

    def test_rejects_oversized_payload(self):
        """Should reject payload larger than 10KB."""
        server = DebateStreamServer()
        # Create a large payload
        data = {"payload": {"data": "x" * 20000}}

        payload, error = server._validate_audience_payload(data)
        assert payload is None
        assert "too large" in error.lower()
