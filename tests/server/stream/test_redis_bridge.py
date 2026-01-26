"""Tests for Redis Pub/Sub broadcast bridge."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.stream.redis_bridge import (
    RedisBroadcastBridge,
    get_broadcast_bridge,
    reset_broadcast_bridge,
    REDIS_AVAILABLE,
)


class TestRedisBroadcastBridge:
    """Tests for RedisBroadcastBridge."""

    @pytest.fixture
    def mock_broadcaster(self):
        """Create a mock broadcaster."""
        mock = MagicMock()
        mock.broadcast_to_debate = AsyncMock()
        mock.broadcast_to_loop = AsyncMock()
        mock.broadcast_to_all = AsyncMock()
        return mock

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = AsyncMock()
        mock.ping = AsyncMock(return_value=True)
        mock.publish = AsyncMock()
        mock.close = AsyncMock()
        mock.pubsub = MagicMock()
        return mock

    @pytest.mark.asyncio
    async def test_bridge_init(self, mock_broadcaster):
        """Test bridge initialization."""
        bridge = RedisBroadcastBridge(mock_broadcaster)

        assert bridge._broadcaster == mock_broadcaster
        assert bridge._connected is False
        assert bridge._running is False

    @pytest.mark.asyncio
    async def test_bridge_connect_success(self, mock_broadcaster, mock_redis):
        """Test successful connection."""
        with patch("aragora.server.stream.redis_bridge.REDIS_AVAILABLE", True):
            with patch("aragora.server.stream.redis_bridge.aioredis") as mock_aioredis:
                mock_aioredis.from_url.return_value = mock_redis

                # Mock pubsub
                mock_pubsub = AsyncMock()
                mock_pubsub.subscribe = AsyncMock()
                mock_pubsub.listen = AsyncMock(return_value=iter([]))
                mock_redis.pubsub.return_value = mock_pubsub

                bridge = RedisBroadcastBridge(mock_broadcaster)
                result = await bridge.connect()

                assert result is True
                assert bridge.is_connected is True
                mock_redis.ping.assert_called_once()

                # Cleanup
                bridge._running = False
                if bridge._listener_task:
                    bridge._listener_task.cancel()
                    try:
                        await bridge._listener_task
                    except asyncio.CancelledError:
                        pass

    @pytest.mark.asyncio
    async def test_bridge_connect_without_redis(self, mock_broadcaster):
        """Test connection fails gracefully without Redis."""
        with patch("aragora.server.stream.redis_bridge.REDIS_AVAILABLE", False):
            bridge = RedisBroadcastBridge(mock_broadcaster)
            result = await bridge.connect()

            assert result is False
            assert bridge.is_connected is False

    @pytest.mark.asyncio
    async def test_bridge_connect_failure(self, mock_broadcaster, mock_redis):
        """Test connection failure handling."""
        with patch("aragora.server.stream.redis_bridge.REDIS_AVAILABLE", True):
            with patch("aragora.server.stream.redis_bridge.aioredis") as mock_aioredis:
                mock_redis.ping.side_effect = ConnectionError("Connection refused")
                mock_aioredis.from_url.return_value = mock_redis

                bridge = RedisBroadcastBridge(mock_broadcaster)
                result = await bridge.connect()

                assert result is False
                assert bridge.is_connected is False

    @pytest.mark.asyncio
    async def test_publish_debate_event(self, mock_broadcaster, mock_redis):
        """Test publishing debate events."""
        with patch("aragora.server.stream.redis_bridge.REDIS_AVAILABLE", True):
            with patch("aragora.server.stream.redis_bridge.aioredis") as mock_aioredis:
                mock_aioredis.from_url.return_value = mock_redis

                bridge = RedisBroadcastBridge(mock_broadcaster)
                bridge._redis = mock_redis
                bridge._connected = True

                await bridge.publish_debate_event(
                    "debate-123",
                    "message",
                    {"content": "Hello world"},
                )

                mock_redis.publish.assert_called_once()
                call_args = mock_redis.publish.call_args
                channel = call_args[0][0]
                message = json.loads(call_args[0][1])

                assert channel == "aragora:broadcast:debates"
                assert message["type"] == "message"
                assert message["payload"]["debate_id"] == "debate-123"
                assert message["payload"]["content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_publish_loop_event(self, mock_broadcaster, mock_redis):
        """Test publishing loop events."""
        with patch("aragora.server.stream.redis_bridge.REDIS_AVAILABLE", True):
            bridge = RedisBroadcastBridge(mock_broadcaster)
            bridge._redis = mock_redis
            bridge._connected = True

            await bridge.publish_loop_event(
                "loop-456",
                "iteration",
                {"round": 3},
            )

            call_args = mock_redis.publish.call_args
            channel = call_args[0][0]
            message = json.loads(call_args[0][1])

            assert channel == "aragora:broadcast:loops"
            assert message["type"] == "iteration"
            assert message["payload"]["loop_id"] == "loop-456"

    @pytest.mark.asyncio
    async def test_publish_global_event(self, mock_broadcaster, mock_redis):
        """Test publishing global events."""
        bridge = RedisBroadcastBridge(mock_broadcaster)
        bridge._redis = mock_redis
        bridge._connected = True

        await bridge.publish_global_event(
            "system_status",
            {"status": "healthy"},
        )

        call_args = mock_redis.publish.call_args
        channel = call_args[0][0]
        message = json.loads(call_args[0][1])

        assert channel == "aragora:broadcast:global"
        assert message["type"] == "system_status"

    @pytest.mark.asyncio
    async def test_publish_skipped_when_disconnected(self, mock_broadcaster, mock_redis):
        """Test publishing is skipped when not connected."""
        bridge = RedisBroadcastBridge(mock_broadcaster)
        bridge._connected = False

        await bridge.publish_debate_event("debate-123", "message", {})

        mock_redis.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_message_ignores_own_instance(self, mock_broadcaster):
        """Test that messages from own instance are ignored."""
        bridge = RedisBroadcastBridge(mock_broadcaster, instance_id="instance-1")
        bridge._connected = True

        message = {
            "type": "message",
            "channel": "aragora:broadcast:debates",
            "data": json.dumps(
                {
                    "type": "debate_message",
                    "payload": {"debate_id": "test"},
                    "instance_id": "instance-1",  # Same as bridge
                }
            ),
        }

        await bridge._handle_message(message)

        # Should not relay to broadcaster
        mock_broadcaster.broadcast_to_debate.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_message_relays_remote_events(self, mock_broadcaster):
        """Test that messages from other instances are relayed."""
        bridge = RedisBroadcastBridge(mock_broadcaster, instance_id="instance-1")
        bridge._connected = True
        bridge._broadcaster = mock_broadcaster

        message = {
            "type": "message",
            "channel": "aragora:broadcast:debates",
            "data": json.dumps(
                {
                    "type": "debate_message",
                    "payload": {"debate_id": "test-debate"},
                    "instance_id": "instance-2",  # Different from bridge
                }
            ),
        }

        await bridge._handle_message(message)

        # Should relay to broadcaster
        mock_broadcaster.broadcast_to_debate.assert_called_once()
        call_args = mock_broadcaster.broadcast_to_debate.call_args
        assert call_args[0][0] == "test-debate"

    @pytest.mark.asyncio
    async def test_handle_message_invalid_json(self, mock_broadcaster):
        """Test handling of invalid JSON messages."""
        bridge = RedisBroadcastBridge(mock_broadcaster)
        bridge._connected = True

        message = {
            "type": "message",
            "channel": "aragora:broadcast:debates",
            "data": "not valid json",
        }

        # Should not raise, just log warning
        await bridge._handle_message(message)
        mock_broadcaster.broadcast_to_debate.assert_not_called()

    @pytest.mark.asyncio
    async def test_relay_to_local_debate_channel(self, mock_broadcaster):
        """Test relaying to local broadcaster for debate channel."""
        bridge = RedisBroadcastBridge(mock_broadcaster)

        await bridge._relay_to_local(
            "aragora:broadcast:debates",
            "message",
            {"debate_id": "test-123", "content": "hello"},
        )

        mock_broadcaster.broadcast_to_debate.assert_called_once_with(
            "test-123",
            {"type": "message", "debate_id": "test-123", "content": "hello"},
        )

    @pytest.mark.asyncio
    async def test_relay_to_local_loop_channel(self, mock_broadcaster):
        """Test relaying to local broadcaster for loop channel."""
        bridge = RedisBroadcastBridge(mock_broadcaster)

        await bridge._relay_to_local(
            "aragora:broadcast:loops",
            "iteration",
            {"loop_id": "loop-456", "round": 2},
        )

        mock_broadcaster.broadcast_to_loop.assert_called_once_with(
            "loop-456",
            {"type": "iteration", "loop_id": "loop-456", "round": 2},
        )

    @pytest.mark.asyncio
    async def test_relay_to_local_global_channel(self, mock_broadcaster):
        """Test relaying to local broadcaster for global channel."""
        bridge = RedisBroadcastBridge(mock_broadcaster)

        await bridge._relay_to_local(
            "aragora:broadcast:global",
            "status",
            {"status": "healthy"},
        )

        mock_broadcaster.broadcast_to_all.assert_called_once_with(
            {"type": "status", "status": "healthy"},
        )

    @pytest.mark.asyncio
    async def test_health_check_connected(self, mock_broadcaster, mock_redis):
        """Test health check when connected."""
        bridge = RedisBroadcastBridge(mock_broadcaster, instance_id="test-instance")
        bridge._redis = mock_redis
        bridge._connected = True
        bridge._listener_task = MagicMock()
        bridge._listener_task.done.return_value = False

        health = await bridge.health_check()

        assert health["connected"] is True
        assert health["instance_id"] == "test-instance"
        assert health["listener_running"] is True
        assert "ping_ms" in health

    @pytest.mark.asyncio
    async def test_health_check_disconnected(self, mock_broadcaster):
        """Test health check when disconnected."""
        bridge = RedisBroadcastBridge(mock_broadcaster, instance_id="test-instance")
        bridge._connected = False

        health = await bridge.health_check()

        assert health["connected"] is False
        assert health["instance_id"] == "test-instance"
        assert health["listener_running"] is False

    @pytest.mark.asyncio
    async def test_disconnect(self, mock_broadcaster, mock_redis):
        """Test disconnect cleanup."""
        bridge = RedisBroadcastBridge(mock_broadcaster)
        bridge._redis = mock_redis
        bridge._connected = True
        bridge._running = True

        # Create mock pubsub
        mock_pubsub = AsyncMock()
        mock_pubsub.unsubscribe = AsyncMock()
        mock_pubsub.close = AsyncMock()
        bridge._pubsub = mock_pubsub

        # No listener task for this test - just test pubsub cleanup
        bridge._listener_task = None

        await bridge.disconnect()

        assert bridge._connected is False
        assert bridge._running is False
        mock_pubsub.unsubscribe.assert_called_once()
        mock_pubsub.close.assert_called_once()
        mock_redis.close.assert_called_once()


class TestBridgeSingleton:
    """Tests for global bridge singleton functions."""

    @pytest.mark.asyncio
    async def test_get_broadcast_bridge_creates_new(self):
        """Test get_broadcast_bridge creates new instance."""
        await reset_broadcast_bridge()

        mock_broadcaster = MagicMock()
        mock_broadcaster.broadcast_to_debate = AsyncMock()

        with patch("aragora.server.stream.redis_bridge.REDIS_AVAILABLE", False):
            bridge = await get_broadcast_bridge(mock_broadcaster)

            assert bridge is not None
            assert bridge._broadcaster == mock_broadcaster

        await reset_broadcast_bridge()

    @pytest.mark.asyncio
    async def test_get_broadcast_bridge_requires_broadcaster_first_time(self):
        """Test that first call requires broadcaster."""
        await reset_broadcast_bridge()

        with pytest.raises(ValueError, match="broadcaster required"):
            await get_broadcast_bridge()

        await reset_broadcast_bridge()

    @pytest.mark.asyncio
    async def test_reset_broadcast_bridge(self):
        """Test reset_broadcast_bridge cleans up."""
        await reset_broadcast_bridge()

        mock_broadcaster = MagicMock()
        with patch("aragora.server.stream.redis_bridge.REDIS_AVAILABLE", False):
            bridge = await get_broadcast_bridge(mock_broadcaster)
            assert bridge is not None

        await reset_broadcast_bridge()

        # After reset, should require broadcaster again
        with pytest.raises(ValueError, match="broadcaster required"):
            await get_broadcast_bridge()
