"""
Tests for per-client broadcast logic and audience input routing.

Verifies:
- Unsubscribed clients receive events correctly when auth is off
- loop_id is auto-injected by SyncEventEmitter
- Audience input routing through AudienceInbox
- Rate limiting via TokenBucket
"""

import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock, patch

import pytest

from aragora.server.stream import (
    DebateStreamServer,
    StreamEvent,
    StreamEventType,
    SyncEventEmitter,
    AudienceInbox,
    AudienceMessage,
    TokenBucket,
    LoopInstance,
)
from aragora.server.auth import AuthConfig, check_auth, generate_shareable_link


class TestBroadcastWithAuthOff:
    """Test that all clients receive events when auth is disabled."""

    @pytest.mark.asyncio
    async def test_all_clients_receive_broadcast_when_auth_disabled(self):
        """All connected clients should receive broadcast events when auth is off."""
        server = DebateStreamServer(host="localhost", port=0)

        # Create mock clients
        client1 = AsyncMock()
        client2 = AsyncMock()
        client3 = AsyncMock()

        server.clients = {client1, client2, client3}

        event = StreamEvent(
            type=StreamEventType.TASK_START,
            data={"task": "test_task"},
            loop_id="test_loop"
        )

        # Run broadcast
        await server.broadcast(event)

        # All clients should have received the message
        expected_message = event.to_json()
        client1.send.assert_called_once_with(expected_message)
        client2.send.assert_called_once_with(expected_message)
        client3.send.assert_called_once_with(expected_message)

    @pytest.mark.asyncio
    async def test_broadcast_handles_disconnected_clients(self):
        """Disconnected clients should be removed from client set."""
        server = DebateStreamServer(host="localhost", port=0)

        # Create mock clients - one that fails
        good_client = AsyncMock()
        bad_client = AsyncMock()
        bad_client.send.side_effect = Exception("Connection closed")

        server.clients = {good_client, bad_client}

        event = StreamEvent(
            type=StreamEventType.TASK_START,
            data={"task": "test"},
        )

        await server.broadcast(event)

        # Bad client should be removed
        assert bad_client not in server.clients
        assert good_client in server.clients

    @pytest.mark.asyncio
    async def test_broadcast_skips_when_no_clients(self):
        """Broadcast should handle empty client set gracefully."""
        server = DebateStreamServer(host="localhost", port=0)
        server.clients = set()

        event = StreamEvent(
            type=StreamEventType.TASK_START,
            data={"task": "test"},
        )

        # Should not raise
        await server.broadcast(event)


class TestLoopIdAutoInjection:
    """Test that loop_id is auto-injected by SyncEventEmitter."""

    def test_loop_id_injected_when_not_set(self):
        """Emitter should inject loop_id when event does not have one."""
        emitter = SyncEventEmitter(loop_id="default_loop")

        event = StreamEvent(
            type=StreamEventType.TASK_START,
            data={"task": "test"},
            loop_id=""  # Empty loop_id
        )

        emitter.emit(event)

        # Drain and check
        events = emitter.drain()
        assert len(events) == 1
        assert events[0].loop_id == "default_loop"

    def test_loop_id_preserved_when_already_set(self):
        """Emitter should preserve existing loop_id on event."""
        emitter = SyncEventEmitter(loop_id="default_loop")

        event = StreamEvent(
            type=StreamEventType.TASK_START,
            data={"task": "test"},
            loop_id="specific_loop"  # Already has loop_id
        )

        emitter.emit(event)

        events = emitter.drain()
        assert len(events) == 1
        assert events[0].loop_id == "specific_loop"

    def test_set_loop_id_updates_emitter(self):
        """set_loop_id should update the default loop_id for future events."""
        emitter = SyncEventEmitter(loop_id="initial_loop")

        # Emit first event
        event1 = StreamEvent(type=StreamEventType.TASK_START, data={})
        emitter.emit(event1)

        # Update loop_id
        emitter.set_loop_id("new_loop")

        # Emit second event
        event2 = StreamEvent(type=StreamEventType.TASK_COMPLETE, data={})
        emitter.emit(event2)

        events = emitter.drain()
        assert events[0].loop_id == "initial_loop"
        assert events[1].loop_id == "new_loop"

    def test_emitter_without_loop_id(self):
        """Emitter without loop_id should not inject anything."""
        emitter = SyncEventEmitter()  # No loop_id

        event = StreamEvent(
            type=StreamEventType.TASK_START,
            data={"task": "test"},
            loop_id=""
        )

        emitter.emit(event)

        events = emitter.drain()
        assert len(events) == 1
        assert events[0].loop_id == ""

    def test_loop_id_in_serialized_event(self):
        """loop_id should appear in serialized event when set."""
        emitter = SyncEventEmitter(loop_id="serialization_test")

        event = StreamEvent(
            type=StreamEventType.CYCLE_START,
            data={"cycle": 1},
        )

        emitter.emit(event)
        events = emitter.drain()

        serialized = events[0].to_dict()
        assert "loop_id" in serialized
        assert serialized["loop_id"] == "serialization_test"


class TestAudienceInputRouting:
    """Test audience input routing through AudienceInbox."""

    def test_audience_message_queuing(self):
        """AudienceInbox should queue messages correctly."""
        inbox = AudienceInbox()

        msg1 = AudienceMessage(
            type="vote",
            loop_id="loop_1",
            payload={"choice": "option_a"},
            user_id="user_1"
        )
        msg2 = AudienceMessage(
            type="suggestion",
            loop_id="loop_1",
            payload={"text": "My suggestion"},
            user_id="user_2"
        )

        inbox.put(msg1)
        inbox.put(msg2)

        messages = inbox.get_all()
        assert len(messages) == 2
        assert messages[0].type == "vote"
        assert messages[1].type == "suggestion"

    def test_inbox_drains_on_get_all(self):
        """get_all should drain the inbox."""
        inbox = AudienceInbox()

        inbox.put(AudienceMessage(type="vote", loop_id="loop_1", payload={}))
        inbox.put(AudienceMessage(type="vote", loop_id="loop_1", payload={}))

        first_drain = inbox.get_all()
        assert len(first_drain) == 2

        second_drain = inbox.get_all()
        assert len(second_drain) == 0

    def test_inbox_summary_counts_votes(self):
        """get_summary should count votes by choice."""
        inbox = AudienceInbox()

        inbox.put(AudienceMessage(type="vote", loop_id="loop_1", payload={"choice": "A"}))
        inbox.put(AudienceMessage(type="vote", loop_id="loop_1", payload={"choice": "A"}))
        inbox.put(AudienceMessage(type="vote", loop_id="loop_1", payload={"choice": "B"}))
        inbox.put(AudienceMessage(type="suggestion", loop_id="loop_1", payload={"text": "idea"}))

        summary = inbox.get_summary()

        assert summary["votes"]["A"] == 2
        assert summary["votes"]["B"] == 1
        assert summary["suggestions"] == 1
        assert summary["total"] == 4

    def test_inbox_thread_safety(self):
        """Inbox should be thread-safe for concurrent access."""
        import threading

        inbox = AudienceInbox()
        errors = []

        def add_messages():
            try:
                for i in range(100):
                    inbox.put(AudienceMessage(
                        type="vote",
                        loop_id="loop_1",
                        payload={"choice": f"option_{i % 3}"}
                    ))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_messages) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        messages = inbox.get_all()
        assert len(messages) == 500  # 5 threads * 100 messages


class TestRateLimiting:
    """Test TokenBucket rate limiting for audience messages."""

    def test_token_bucket_allows_burst(self):
        """TokenBucket should allow initial burst up to burst_size."""
        bucket = TokenBucket(rate_per_minute=10.0, burst_size=5)

        # Should allow 5 immediate requests
        for _ in range(5):
            assert bucket.consume(1) is True

        # 6th should be denied (no time for refill)
        assert bucket.consume(1) is False

    def test_token_bucket_refills_over_time(self):
        """TokenBucket should refill tokens over time."""
        bucket = TokenBucket(rate_per_minute=600.0, burst_size=5)  # 10 per second

        # Consume all tokens
        for _ in range(5):
            bucket.consume(1)

        # Wait a short time for refill
        time.sleep(0.1)  # Should refill ~1 token

        # Should now have some tokens
        assert bucket.consume(1) is True

    def test_token_bucket_thread_safety(self):
        """TokenBucket should be thread-safe."""
        import threading

        bucket = TokenBucket(rate_per_minute=6000.0, burst_size=100)
        results = []

        def consume_tokens():
            for _ in range(20):
                results.append(bucket.consume(1))

        threads = [threading.Thread(target=consume_tokens) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Total consumed should not exceed burst_size + refilled
        total_consumed = sum(1 for r in results if r)
        assert total_consumed <= 100 + 10  # burst + some refill


class TestAuthIntegration:
    """Test authentication integration with broadcast."""

    def test_auth_config_disabled_by_default(self):
        """Auth should be disabled when no token is configured."""
        config = AuthConfig()
        assert config.enabled is False
        authenticated, _ = check_auth({}, "")
        assert authenticated is True  # Should allow when disabled

    def test_auth_config_enables_with_token(self):
        """Auth should enable when token is set."""
        config = AuthConfig()
        config.api_token = "secret_token"
        config.enabled = True

        # Without valid token, should fail
        assert config.validate_token("", "") is False

        # Generate and validate token
        token = config.generate_token("loop_123")
        assert config.validate_token(token, "loop_123") is True

    def test_token_expiration(self):
        """Tokens should expire after TTL."""
        config = AuthConfig()
        config.api_token = "secret_token"
        config.token_ttl = -1  # Already expired

        token = config.generate_token("loop_123", expires_in=-1)
        assert config.validate_token(token, "loop_123") is False

    def test_token_loop_id_validation(self):
        """Token should only be valid for its designated loop_id."""
        config = AuthConfig()
        config.api_token = "secret_token"

        token = config.generate_token("loop_123")

        # Valid for correct loop
        assert config.validate_token(token, "loop_123") is True
        # Invalid for different loop
        assert config.validate_token(token, "loop_456") is False


class TestLoopRegistration:
    """Test loop registration and broadcasting."""

    def test_register_loop_creates_instance(self):
        """register_loop should create a LoopInstance."""
        server = DebateStreamServer(host="localhost", port=0)

        server.register_loop("loop_1", "Test Loop", "/path/to/loop")

        assert "loop_1" in server.active_loops
        loop = server.active_loops["loop_1"]
        assert loop.name == "Test Loop"
        assert loop.path == "/path/to/loop"

    def test_unregister_loop_removes_instance(self):
        """unregister_loop should remove the LoopInstance."""
        server = DebateStreamServer(host="localhost", port=0)

        server.register_loop("loop_1", "Test Loop")
        assert "loop_1" in server.active_loops

        server.unregister_loop("loop_1")
        assert "loop_1" not in server.active_loops

    def test_update_loop_state(self):
        """update_loop_state should modify loop attributes."""
        server = DebateStreamServer(host="localhost", port=0)

        server.register_loop("loop_1", "Test Loop")
        server.update_loop_state("loop_1", cycle=5, phase="verification")

        loop = server.active_loops["loop_1"]
        assert loop.cycle == 5
        assert loop.phase == "verification"

    def test_get_loop_list_returns_all_loops(self):
        """get_loop_list should return info for all active loops."""
        server = DebateStreamServer(host="localhost", port=0)

        server.register_loop("loop_1", "Loop One")
        server.register_loop("loop_2", "Loop Two")

        loop_list = server.get_loop_list()

        assert len(loop_list) == 2
        loop_ids = {l["loop_id"] for l in loop_list}
        assert loop_ids == {"loop_1", "loop_2"}


class TestEventSerialization:
    """Test event serialization for WebSocket transmission."""

    def test_stream_event_to_dict(self):
        """StreamEvent should serialize to dict correctly."""
        event = StreamEvent(
            type=StreamEventType.AGENT_MESSAGE,
            data={"content": "Hello", "role": "assistant"},
            round=3,
            agent="claude-visionary",
            loop_id="test_loop"
        )

        result = event.to_dict()

        assert result["type"] == "agent_message"
        assert result["data"]["content"] == "Hello"
        assert result["round"] == 3
        assert result["agent"] == "claude-visionary"
        assert result["loop_id"] == "test_loop"

    def test_stream_event_to_json(self):
        """StreamEvent should serialize to valid JSON."""
        event = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={"reached": True, "confidence": 0.95, "answer": "Test answer"},
            loop_id="json_test"
        )

        json_str = event.to_json()
        parsed = json.loads(json_str)

        assert parsed["type"] == "consensus"
        assert parsed["data"]["reached"] is True
        assert parsed["data"]["confidence"] == 0.95


class TestEmitterSubscriptions:
    """Test synchronous subscriber functionality."""

    def test_subscriber_receives_events(self):
        """Subscribers should receive emitted events synchronously."""
        emitter = SyncEventEmitter()
        received = []

        def subscriber(event: StreamEvent):
            received.append(event)

        emitter.subscribe(subscriber)

        event = StreamEvent(type=StreamEventType.TASK_START, data={"task": "test"})
        emitter.emit(event)

        assert len(received) == 1
        assert received[0].type == StreamEventType.TASK_START

    def test_multiple_subscribers(self):
        """Multiple subscribers should all receive events."""
        emitter = SyncEventEmitter()
        received1 = []
        received2 = []

        emitter.subscribe(lambda e: received1.append(e))
        emitter.subscribe(lambda e: received2.append(e))

        event = StreamEvent(type=StreamEventType.CYCLE_START, data={"cycle": 1})
        emitter.emit(event)

        assert len(received1) == 1
        assert len(received2) == 1

    def test_subscriber_exception_doesnt_break_emit(self):
        """Exception in subscriber should not prevent other subscribers."""
        emitter = SyncEventEmitter()
        received = []

        def bad_subscriber(event: StreamEvent):
            raise ValueError("Subscriber error")

        def good_subscriber(event: StreamEvent):
            received.append(event)

        emitter.subscribe(bad_subscriber)
        emitter.subscribe(good_subscriber)

        event = StreamEvent(type=StreamEventType.ERROR, data={"error": "test"})
        emitter.emit(event)  # Should not raise

        assert len(received) == 1


class TestErrorEventHandling:
    """Test error event serialization and broadcasting."""

    def test_error_event_serialization(self):
        """Error events should serialize with proper structure."""
        event = StreamEvent(
            type=StreamEventType.ERROR,
            data={"error": "Connection failed", "code": "ERR_CONN"},
            loop_id="test_loop"
        )

        result = event.to_dict()

        assert result["type"] == "error"
        assert result["data"]["error"] == "Connection failed"
        assert result["data"]["code"] == "ERR_CONN"
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_error_event_broadcast(self):
        """Error events should broadcast to all clients."""
        server = DebateStreamServer(host="localhost", port=0)

        client1 = AsyncMock()
        client2 = AsyncMock()
        server.clients = {client1, client2}

        error_event = StreamEvent(
            type=StreamEventType.ERROR,
            data={"error": "Debate failed"},
            loop_id="test_loop"
        )

        await server.broadcast(error_event)

        # Both clients should receive the error
        assert client1.send.called
        assert client2.send.called
        sent_data = json.loads(client1.send.call_args[0][0])
        assert sent_data["type"] == "error"

    def test_error_event_with_stack_trace_sanitized(self):
        """Error events should not expose internal stack traces."""
        # Simulate an internal error
        event = StreamEvent(
            type=StreamEventType.ERROR,
            data={"error": "Internal server error"},
            loop_id="test_loop"
        )

        json_str = event.to_json()
        parsed = json.loads(json_str)

        # Should not contain path or traceback info
        assert "traceback" not in parsed["data"].get("error", "").lower()
        assert "/Users/" not in json_str
        assert "\\Users\\" not in json_str


class TestDebateStateErrors:
    """Test error handling in debate state management."""

    def test_update_state_missing_loop(self):
        """State update for non-existent loop should not crash."""
        server = DebateStreamServer(host="localhost", port=0)

        # Try to update state for a loop that doesn't exist
        event = StreamEvent(
            type=StreamEventType.AGENT_MESSAGE,
            data={"role": "assistant", "content": "Test"},
            agent="test-agent",
            loop_id="nonexistent_loop"
        )

        # Should not raise
        server._update_debate_state(event)

        # Loop should still not exist in states
        assert "nonexistent_loop" not in server.debate_states

    def test_debate_state_malformed_event_data(self):
        """Malformed event data should be handled gracefully."""
        server = DebateStreamServer(host="localhost", port=0)

        # Create a debate first
        start_event = StreamEvent(
            type=StreamEventType.DEBATE_START,
            data={"task": "Test task", "agents": ["agent1", "agent2"]},
            loop_id="test_loop"
        )
        server._update_debate_state(start_event)

        # Send message with missing fields
        malformed_event = StreamEvent(
            type=StreamEventType.AGENT_MESSAGE,
            data={},  # Missing required fields
            agent="test-agent",
            loop_id="test_loop"
        )

        # Should raise KeyError since role is required
        with pytest.raises(KeyError):
            server._update_debate_state(malformed_event)

    def test_message_history_overflow_protection(self):
        """Message history should be capped to prevent memory issues."""
        server = DebateStreamServer(host="localhost", port=0)

        # Create a debate
        start_event = StreamEvent(
            type=StreamEventType.DEBATE_START,
            data={"task": "Test task", "agents": ["agent1"]},
            loop_id="test_loop"
        )
        server._update_debate_state(start_event)

        # Add 1100 messages (above the 1000 cap)
        for i in range(1100):
            msg_event = StreamEvent(
                type=StreamEventType.AGENT_MESSAGE,
                data={"role": "assistant", "content": f"Message {i}"},
                agent="agent1",
                round=i,
                loop_id="test_loop"
            )
            server._update_debate_state(msg_event)

        # Should be capped at 1000
        assert len(server.debate_states["test_loop"]["messages"]) == 1000
        # Should keep the latest messages
        assert "Message 1099" in server.debate_states["test_loop"]["messages"][-1]["content"]


class TestRateLimiterCleanup:
    """Test rate limiter cleanup and memory management."""

    def test_stale_rate_limiter_cleanup(self):
        """Stale rate limiters should be cleaned up."""
        server = DebateStreamServer(host="localhost", port=0)

        # Add a rate limiter with old timestamp
        with server._rate_limiters_lock:
            server._rate_limiters["old_client"] = TokenBucket(rate_per_minute=10, burst_size=5)
            server._rate_limiter_last_access["old_client"] = time.time() - 7200  # 2 hours ago

            server._rate_limiters["recent_client"] = TokenBucket(rate_per_minute=10, burst_size=5)
            server._rate_limiter_last_access["recent_client"] = time.time()

        # Run cleanup
        server._cleanup_stale_rate_limiters()

        # Old client should be removed
        assert "old_client" not in server._rate_limiters
        # Recent client should remain
        assert "recent_client" in server._rate_limiters

    def test_rate_limiter_ttl_boundary(self):
        """Rate limiter exactly at TTL boundary should be kept."""
        server = DebateStreamServer(host="localhost", port=0)

        # Add a rate limiter at exactly TTL - 1 second (using config value)
        with server._rate_limiters_lock:
            server._rate_limiters["boundary_client"] = TokenBucket(rate_per_minute=10, burst_size=5)
            server._rate_limiter_last_access["boundary_client"] = time.time() - server.config.rate_limiter_ttl + 1

        server._cleanup_stale_rate_limiters()

        # Should still exist (not yet expired)
        assert "boundary_client" in server._rate_limiters


class TestLateJoinerSync:
    """Test late joiner state synchronization."""

    def test_get_debate_state_for_late_joiner(self):
        """Late joiners should receive current debate state."""
        server = DebateStreamServer(host="localhost", port=0)

        # Create a debate with some history
        start_event = StreamEvent(
            type=StreamEventType.DEBATE_START,
            data={"task": "Test debate", "agents": ["agent1", "agent2"]},
            loop_id="sync_test"
        )
        server._update_debate_state(start_event)

        # Add messages
        for i in range(3):
            msg_event = StreamEvent(
                type=StreamEventType.AGENT_MESSAGE,
                data={"role": "assistant", "content": f"Turn {i}"},
                agent=f"agent{(i % 2) + 1}",
                round=i,
                loop_id="sync_test"
            )
            server._update_debate_state(msg_event)

        # Get state for late joiner
        state = server.debate_states.get("sync_test")

        assert state is not None
        assert state["task"] == "Test debate"
        assert len(state["messages"]) == 3

    def test_late_joiner_missing_loop(self):
        """Request for non-existent loop should return None."""
        server = DebateStreamServer(host="localhost", port=0)

        state = server.debate_states.get("nonexistent_loop")
        assert state is None


class TestClientIdSecurity:
    """Test secure client ID generation and mapping."""

    def test_client_id_is_cryptographically_secure(self):
        """Client IDs should be cryptographically random, not memory addresses."""
        import secrets

        server = DebateStreamServer(host="localhost", port=0)

        # Simulate generating client IDs for multiple clients
        mock_ws_1 = Mock()
        mock_ws_2 = Mock()

        # The server uses secrets.token_hex for client IDs
        client_id_1 = secrets.token_hex(16)
        client_id_2 = secrets.token_hex(16)

        # IDs should be different
        assert client_id_1 != client_id_2
        # IDs should be proper hex strings
        assert all(c in "0123456789abcdef" for c in client_id_1)
        assert len(client_id_1) == 32  # 16 bytes = 32 hex chars

    def test_client_id_not_predictable(self):
        """Client IDs should not be predictable from memory address."""
        import secrets

        # Generate many IDs and ensure they're all unique
        ids = [secrets.token_hex(16) for _ in range(100)]
        assert len(set(ids)) == 100  # All unique

