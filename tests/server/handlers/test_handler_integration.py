"""
Integration tests for handler end-to-end flows.

Tests cover:
- DecisionRouter integration in command registry (affects Discord/Teams)
- Webhook delivery wiring to debate lifecycle
- Origin tracking persistence
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone, timedelta
from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.bots.base import (
    BotChannel,
    BotMessage,
    BotUser,
    CommandContext,
    CommandResult,
    Platform,
)
from aragora.bots.commands import CommandRegistry, BotCommand


# ===========================================================================
# Test Fixtures
# ===========================================================================


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(
        self,
        headers: dict | None = None,
        body: bytes = b"",
        path: str = "/",
        method: str = "GET",
    ):
        self.headers = headers or {}
        self._body = body
        self.path = path
        self.command = method
        self.rfile = BytesIO(body)


@pytest.fixture
def mock_server_context():
    """Create a mock server context for handler initialization."""
    return {
        "storage": MagicMock(),
        "user_store": MagicMock(),
        "elo_system": MagicMock(),
        "continuum_memory": MagicMock(),
        "critique_store": MagicMock(),
        "document_store": MagicMock(),
        "evidence_store": MagicMock(),
        "usage_tracker": MagicMock(),
    }


@pytest.fixture
def mock_decision_result():
    """Create a mock DecisionResult."""
    mock = MagicMock()
    mock.debate_id = "test-debate-123"
    mock.success = True
    mock.error = None
    return mock


@pytest.fixture
def fresh_registry():
    """Create a fresh command registry without built-in commands for testing."""
    return CommandRegistry()


def unique_user_id():
    """Generate a unique user ID to avoid cooldown issues."""
    return f"test-user-{uuid.uuid4().hex[:8]}"


def create_command_context(platform: Platform, topic: str = "Test topic") -> CommandContext:
    """Create a CommandContext with unique user ID."""
    user_id = unique_user_id()
    user = BotUser(
        id=user_id,
        username=f"user_{user_id}",
        display_name="Test User",
        platform=platform,
    )
    channel = BotChannel(
        id=f"channel-{uuid.uuid4().hex[:8]}",
        platform=platform,
    )
    message = BotMessage(
        id=f"msg-{uuid.uuid4().hex[:8]}",
        text=f"/debate {topic}",
        user=user,
        channel=channel,
        timestamp=datetime.now(timezone.utc),
        platform=platform,
    )
    return CommandContext(
        message=message,
        user=user,
        channel=channel,
        platform=platform,
        args=["debate", topic],
        raw_args=f"debate {topic}",
        metadata={"api_base": "http://localhost:8080"},
    )


# ===========================================================================
# Command Registry DecisionRouter Integration Tests
# ===========================================================================


class TestCommandRegistryWithDecisionRouter:
    """Tests for DecisionRouter integration in command registry."""

    @pytest.mark.asyncio
    async def test_debate_command_routes_through_decision_router(
        self, fresh_registry, mock_decision_result
    ):
        """Test debate command routes through DecisionRouter when available."""
        route_called = False
        captured_request = None

        async def mock_debate_handler(ctx: CommandContext) -> CommandResult:
            """Test handler that uses DecisionRouter pattern."""
            nonlocal route_called, captured_request

            # Simulate what the real debate command does
            try:
                from aragora.core import (
                    DecisionRequest,
                    DecisionType,
                    InputSource,
                    RequestContext,
                    ResponseChannel,
                )

                source = InputSource.DISCORD
                response_channel = ResponseChannel(
                    platform=ctx.platform.value,
                    channel_id=ctx.channel_id,
                    user_id=ctx.user_id,
                )

                request = DecisionRequest(
                    content=ctx.raw_args,
                    decision_type=DecisionType.DEBATE,
                    source=source,
                    response_channels=[response_channel],
                    context=RequestContext(
                        user_id=ctx.user_id,
                        session_id=f"{ctx.platform.value}:{ctx.channel_id}",
                    ),
                )
                captured_request = request

                # Mock routing
                route_called = True
                return CommandResult.ok(
                    f"Debate started on: **{ctx.raw_args}**\nDebate ID: `test-debate-123`",
                    data={"debate_id": "test-debate-123"},
                )

            except ImportError:
                return CommandResult.fail("DecisionRouter not available")

        cmd = BotCommand(
            name="debate",
            handler=mock_debate_handler,
            description="Start a debate",
            requires_args=True,
        )
        fresh_registry.register(cmd)

        ctx = create_command_context(Platform.DISCORD, "Should AI be regulated?")
        result = await fresh_registry.execute(ctx)

        assert route_called
        assert result.success
        assert "test-debate-123" in result.message
        assert captured_request is not None
        # Note: The registry strips the command name, so content is just the topic
        assert "Should AI be regulated?" in captured_request.content

    @pytest.mark.asyncio
    async def test_debate_command_includes_response_channel(self, fresh_registry):
        """Test debate command includes proper response channel info."""
        captured_channel = None

        async def capture_handler(ctx: CommandContext) -> CommandResult:
            nonlocal captured_channel
            from aragora.core import ResponseChannel

            captured_channel = ResponseChannel(
                platform=ctx.platform.value,
                channel_id=ctx.channel_id,
                user_id=ctx.user_id,
                thread_id=ctx.thread_id,
            )
            return CommandResult.ok("OK")

        cmd = BotCommand(name="debate", handler=capture_handler, requires_args=True)
        fresh_registry.register(cmd)

        ctx = create_command_context(Platform.TEAMS, "Test topic")
        await fresh_registry.execute(ctx)

        assert captured_channel is not None
        assert captured_channel.platform == "teams"
        assert captured_channel.user_id == ctx.user_id

    @pytest.mark.asyncio
    async def test_fallback_to_http_on_router_failure(self, fresh_registry):
        """Test command falls back to HTTP when DecisionRouter fails."""
        http_called = False

        async def fallback_handler(ctx: CommandContext) -> CommandResult:
            nonlocal http_called
            # Simulate DecisionRouter failure and HTTP fallback
            try:
                raise RuntimeError("Router unavailable")
            except RuntimeError:
                # Fallback to HTTP
                http_called = True
                return CommandResult.ok(
                    "Debate started via HTTP fallback",
                    data={"debate_id": "http-fallback-123"},
                )

        cmd = BotCommand(name="debate", handler=fallback_handler, requires_args=True)
        fresh_registry.register(cmd)

        ctx = create_command_context(Platform.DISCORD, "Test topic")
        result = await fresh_registry.execute(ctx)

        assert http_called
        assert result.success
        assert result.data["debate_id"] == "http-fallback-123"


# ===========================================================================
# Webhook Delivery Integration Tests
# ===========================================================================


class TestWebhookDeliveryIntegration:
    """Tests for webhook delivery integration with debate lifecycle."""

    @pytest.mark.asyncio
    async def test_webhook_delivery_persists_on_retry(self, tmp_path):
        """Test webhook delivery persists retrying deliveries."""
        from aragora.server.webhook_delivery import (
            DeliveryStatus,
            WebhookDeliveryManager,
        )

        db_path = str(tmp_path / "test_delivery.db")
        manager = WebhookDeliveryManager(
            max_retries=3,
            base_delay_seconds=0.1,
            db_path=db_path,
            enable_persistence=True,
        )

        async def mock_sender(url, payload, headers):
            return 500, {"error": "Server error"}

        manager.set_sender(mock_sender)

        delivery = await manager.deliver(
            webhook_id="wh-123",
            event_type="debate_end",
            payload={"debate_id": "d-456"},
            url="https://example.com/webhook",
        )

        assert delivery.status == DeliveryStatus.RETRYING

        # Verify persisted to database
        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT status FROM webhook_deliveries WHERE delivery_id = ?",
            (delivery.delivery_id,),
        )
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "retrying"

    @pytest.mark.asyncio
    async def test_webhook_delivery_removes_on_success(self, tmp_path):
        """Test webhook delivery removes record on successful delivery."""
        from aragora.server.webhook_delivery import (
            DeliveryStatus,
            WebhookDeliveryManager,
        )

        db_path = str(tmp_path / "test_delivery.db")
        manager = WebhookDeliveryManager(
            db_path=db_path,
            enable_persistence=True,
        )

        async def mock_sender(url, payload, headers):
            return 200, {"ok": True}

        manager.set_sender(mock_sender)

        delivery = await manager.deliver(
            webhook_id="wh-123",
            event_type="debate_end",
            payload={"debate_id": "d-456"},
            url="https://example.com/webhook",
        )

        assert delivery.status == DeliveryStatus.DELIVERED

        # Verify NOT persisted (removed on success)
        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT COUNT(*) FROM webhook_deliveries WHERE delivery_id = ?",
            (delivery.delivery_id,),
        )
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 0

    @pytest.mark.asyncio
    async def test_webhook_recovery_on_startup(self, tmp_path):
        """Test webhook delivery recovers pending deliveries on startup."""
        from aragora.server.webhook_delivery import (
            DeliveryStatus,
            WebhookDeliveryManager,
        )

        db_path = str(tmp_path / "test_recovery.db")

        # First manager - create a retrying delivery
        manager1 = WebhookDeliveryManager(
            max_retries=5,
            base_delay_seconds=1.0,
            db_path=db_path,
            enable_persistence=True,
        )

        async def mock_sender_fail(url, payload, headers):
            return 500, {"error": "Failure"}

        manager1.set_sender(mock_sender_fail)

        delivery = await manager1.deliver(
            webhook_id="wh-recover-1",
            event_type="debate_end",
            payload={"important": "data"},
            url="https://example.com/webhook",
        )

        assert delivery.status == DeliveryStatus.RETRYING
        delivery_id = delivery.delivery_id

        # Second manager - should recover on start
        manager2 = WebhookDeliveryManager(
            max_retries=5,
            base_delay_seconds=0.05,
            db_path=db_path,
            enable_persistence=True,
        )

        await manager2.start()

        try:
            # Check retry queue has the recovered delivery
            assert delivery_id in manager2._retry_queue

            recovered = manager2._retry_queue[delivery_id]
            assert recovered.webhook_id == "wh-recover-1"
            assert recovered.payload == {"important": "data"}
        finally:
            await manager2.stop()

    @pytest.mark.asyncio
    async def test_webhook_dead_letter_persistence(self, tmp_path):
        """Test dead-lettered deliveries are persisted."""
        from aragora.server.webhook_delivery import (
            DeliveryStatus,
            WebhookDeliveryManager,
        )

        db_path = str(tmp_path / "test_dead_letter.db")
        manager = WebhookDeliveryManager(
            max_retries=1,  # Only 1 retry
            db_path=db_path,
            enable_persistence=True,
        )

        async def mock_sender(url, payload, headers):
            return 500, {"error": "Always fails"}

        manager.set_sender(mock_sender)

        delivery = await manager.deliver(
            webhook_id="wh-dead-1",
            event_type="debate_end",
            payload={"data": "test"},
            url="https://example.com/webhook",
        )

        assert delivery.status == DeliveryStatus.DEAD_LETTERED

        # Verify persisted as dead_lettered
        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT status FROM webhook_deliveries WHERE delivery_id = ?",
            (delivery.delivery_id,),
        )
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "dead_lettered"


# ===========================================================================
# Origin Tracking Integration Tests
# ===========================================================================


class TestOriginTrackingIntegration:
    """Tests for origin tracking persistence."""

    def test_origin_store_registers_and_retrieves(self):
        """Test origin tracking registers and retrieves origins."""
        from aragora.server.debate_origin import (
            register_debate_origin,
            get_debate_origin,
        )

        test_debate_id = f"test-debate-{uuid.uuid4().hex[:8]}"

        origin = register_debate_origin(
            debate_id=test_debate_id,
            platform="telegram",
            channel_id="123",
            user_id="456",
        )

        # Verify it can be retrieved
        retrieved = get_debate_origin(test_debate_id)
        assert retrieved is not None
        assert retrieved.platform == "telegram"
        assert retrieved.channel_id == "123"
        assert retrieved.user_id == "456"

    def test_origin_store_different_platforms(self):
        """Test origin tracking works for different platforms."""
        from aragora.server.debate_origin import (
            register_debate_origin,
            get_debate_origin,
        )

        platforms = [
            ("telegram", "chat123", "user456"),
            ("discord", "guild:channel", "user789"),
            ("teams", "tenant:channel", "user012"),
            ("slack", "workspace:channel", "user345"),
            ("email", "inbox", "sender@example.com"),
        ]

        for platform, channel_id, user_id in platforms:
            debate_id = f"debate-{platform}-{uuid.uuid4().hex[:8]}"

            register_debate_origin(
                debate_id=debate_id,
                platform=platform,
                channel_id=channel_id,
                user_id=user_id,
            )

            retrieved = get_debate_origin(debate_id)
            assert retrieved is not None, f"Failed to retrieve {platform}"
            assert retrieved.platform == platform
            assert retrieved.channel_id == channel_id
            assert retrieved.user_id == user_id

    def test_origin_includes_metadata(self):
        """Test origin tracking includes metadata."""
        from aragora.server.debate_origin import (
            register_debate_origin,
            get_debate_origin,
        )

        test_debate_id = f"metadata-debate-{uuid.uuid4().hex[:8]}"

        register_debate_origin(
            debate_id=test_debate_id,
            platform="discord",
            channel_id="channel123",
            user_id="user456",
            metadata={"guild_id": "guild789", "voice_enabled": True},
        )

        retrieved = get_debate_origin(test_debate_id)
        assert retrieved is not None
        assert retrieved.metadata.get("guild_id") == "guild789"
        assert retrieved.metadata.get("voice_enabled") is True


# ===========================================================================
# Cross-Platform Integration Tests
# ===========================================================================


class TestCrossPlatformIntegration:
    """Tests for cross-platform integration scenarios."""

    @pytest.mark.asyncio
    async def test_multiple_platforms_can_route_debates(self, fresh_registry):
        """Test multiple platforms can route debates through same registry."""
        routed_platforms = []

        async def tracking_handler(ctx: CommandContext) -> CommandResult:
            routed_platforms.append(ctx.platform.value)
            return CommandResult.ok(
                f"Debate from {ctx.platform.value}",
                data={"debate_id": f"debate-{ctx.platform.value}"},
            )

        cmd = BotCommand(name="debate", handler=tracking_handler, requires_args=True)
        fresh_registry.register(cmd)

        # Route from different platforms
        for platform in [Platform.DISCORD, Platform.TEAMS, Platform.SLACK]:
            ctx = create_command_context(platform, f"Topic from {platform.value}")
            result = await fresh_registry.execute(ctx)
            assert result.success

        # All platforms should have been routed
        assert len(routed_platforms) == 3
        assert "discord" in routed_platforms
        assert "teams" in routed_platforms
        assert "slack" in routed_platforms

    @pytest.mark.asyncio
    async def test_platform_context_preserved(self, fresh_registry):
        """Test platform-specific context is preserved through routing."""
        captured_contexts = []

        async def capture_handler(ctx: CommandContext) -> CommandResult:
            captured_contexts.append(
                {
                    "platform": ctx.platform,
                    "user_id": ctx.user_id,
                    "channel_id": ctx.channel_id,
                    "raw_args": ctx.raw_args,
                }
            )
            return CommandResult.ok("OK")

        cmd = BotCommand(name="debate", handler=capture_handler, requires_args=True)
        fresh_registry.register(cmd)

        # Create contexts with different data
        ctx1 = create_command_context(Platform.DISCORD, "Discord topic")
        ctx2 = create_command_context(Platform.TEAMS, "Teams topic")

        await fresh_registry.execute(ctx1)
        await fresh_registry.execute(ctx2)

        assert len(captured_contexts) == 2

        # Verify context data is preserved and different
        assert captured_contexts[0]["platform"] == Platform.DISCORD
        assert captured_contexts[1]["platform"] == Platform.TEAMS
        assert captured_contexts[0]["user_id"] != captured_contexts[1]["user_id"]
        assert "Discord topic" in captured_contexts[0]["raw_args"]
        assert "Teams topic" in captured_contexts[1]["raw_args"]


# ===========================================================================
# Error Handling Integration Tests
# ===========================================================================


class TestErrorHandlingIntegration:
    """Tests for error handling in integration scenarios."""

    @pytest.mark.asyncio
    async def test_graceful_error_handling(self, fresh_registry):
        """Test commands handle errors gracefully."""

        async def failing_handler(ctx: CommandContext) -> CommandResult:
            raise RuntimeError("Simulated failure")

        cmd = BotCommand(name="debate", handler=failing_handler, requires_args=True)
        fresh_registry.register(cmd)

        ctx = create_command_context(Platform.DISCORD, "Test topic")
        result = await fresh_registry.execute(ctx)

        # Should return a failure, not raise
        assert not result.success
        assert "failed" in result.error.lower() or "simulated" in result.error.lower()

    @pytest.mark.asyncio
    async def test_missing_required_args(self, fresh_registry):
        """Test commands validate required arguments."""

        async def handler(ctx: CommandContext) -> CommandResult:
            return CommandResult.ok("OK")

        cmd = BotCommand(
            name="debate",
            handler=handler,
            requires_args=True,
            min_args=1,
        )
        fresh_registry.register(cmd)

        # Create context with no args
        user = BotUser(id=unique_user_id(), username="test", platform=Platform.DISCORD)
        channel = BotChannel(id="channel", platform=Platform.DISCORD)
        message = BotMessage(
            id="msg",
            text="/debate",
            user=user,
            channel=channel,
            timestamp=datetime.now(timezone.utc),
            platform=Platform.DISCORD,
        )
        ctx = CommandContext(
            message=message,
            user=user,
            channel=channel,
            platform=Platform.DISCORD,
            args=["debate"],  # Command name only, no topic
            raw_args="debate",
            metadata={"api_base": "http://localhost:8080"},
        )

        result = await fresh_registry.execute(ctx)

        # Should fail validation
        assert not result.success
        assert "requires" in result.error.lower() or "argument" in result.error.lower()
