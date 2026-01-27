"""
Tests for Channel Dock system.

Tests the unified channel abstraction for multi-platform message delivery.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from aragora.channels.dock import (
    ChannelDock,
    ChannelCapability,
    MessageType,
    SendResult,
)
from aragora.channels.normalized import (
    NormalizedMessage,
    MessageFormat,
    MessageButton,
    MessageAttachment,
)
from aragora.channels.registry import DockRegistry


# =============================================================================
# ChannelCapability Tests
# =============================================================================


class TestChannelCapability:
    """Tests for ChannelCapability flags."""

    def test_no_capabilities(self):
        """Test empty capabilities."""
        cap = ChannelCapability.NONE
        assert not cap

    def test_single_capability(self):
        """Test single capability."""
        cap = ChannelCapability.RICH_TEXT
        assert cap
        assert cap & ChannelCapability.RICH_TEXT

    def test_multiple_capabilities(self):
        """Test combining capabilities."""
        cap = ChannelCapability.RICH_TEXT | ChannelCapability.BUTTONS
        assert cap & ChannelCapability.RICH_TEXT
        assert cap & ChannelCapability.BUTTONS
        assert not (cap & ChannelCapability.VOICE)

    def test_capability_check(self):
        """Test checking for specific capability."""
        cap = ChannelCapability.RICH_TEXT | ChannelCapability.THREADS

        assert bool(cap & ChannelCapability.RICH_TEXT)
        assert bool(cap & ChannelCapability.THREADS)
        assert not bool(cap & ChannelCapability.VOICE)


# =============================================================================
# SendResult Tests
# =============================================================================


class TestSendResult:
    """Tests for SendResult dataclass."""

    def test_ok_result(self):
        """Test successful result."""
        result = SendResult.ok(
            message_id="msg-123",
            platform="slack",
            channel_id="C123",
        )
        assert result.success is True
        assert result.message_id == "msg-123"
        assert result.error is None

    def test_fail_result(self):
        """Test failed result."""
        result = SendResult.fail(
            error="Connection timeout",
            platform="slack",
            channel_id="C123",
        )
        assert result.success is False
        assert result.error == "Connection timeout"
        assert result.message_id is None

    def test_result_with_metadata(self):
        """Test result with additional metadata."""
        result = SendResult.ok(
            platform="slack",
            channel_id="C123",
            thread_ts="1234567890.123456",
        )
        assert result.metadata.get("thread_ts") == "1234567890.123456"


# =============================================================================
# NormalizedMessage Tests
# =============================================================================


class TestNormalizedMessage:
    """Tests for NormalizedMessage dataclass."""

    def test_basic_message(self):
        """Test creating a basic message."""
        msg = NormalizedMessage(
            content="Hello, world!",
            message_type=MessageType.NOTIFICATION,
        )
        assert msg.content == "Hello, world!"
        assert msg.format == MessageFormat.PLAIN

    def test_message_with_buttons(self):
        """Test message with buttons."""
        msg = NormalizedMessage(content="Click below")
        msg.with_button("Click me", "https://example.com", "primary")

        assert msg.has_buttons()
        assert len(msg.buttons) == 1
        assert msg.buttons[0].label == "Click me"

    def test_message_with_attachment(self):
        """Test message with attachment."""
        msg = NormalizedMessage(content="See attached")
        msg.with_attachment("image", url="https://example.com/image.png")

        assert msg.has_attachments()
        assert len(msg.attachments) == 1

    def test_to_plain_text(self):
        """Test plain text conversion."""
        msg = NormalizedMessage(
            content="**Bold** and *italic*",
            format=MessageFormat.MARKDOWN,
        )
        plain = msg.to_plain_text()
        assert "**" not in plain
        assert "*" not in plain

    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        original = NormalizedMessage(
            content="Test message",
            message_type=MessageType.RESULT,
            format=MessageFormat.MARKDOWN,
            title="Test Title",
            thread_id="thread-123",
        )
        original.with_button("Click", "https://example.com")

        data = original.to_dict()
        restored = NormalizedMessage.from_dict(data)

        assert restored.content == original.content
        assert restored.message_type == original.message_type
        assert restored.title == original.title
        assert len(restored.buttons) == 1

    def test_get_audio_attachment(self):
        """Test getting audio attachment."""
        msg = NormalizedMessage(content="Voice")
        msg.with_attachment("image", url="image.png")
        msg.with_attachment("audio", data=b"audio_data")

        audio = msg.get_audio_attachment()
        assert audio is not None
        assert audio.type == "audio"


# =============================================================================
# ChannelDock Tests
# =============================================================================


class ConcreteTestDock(ChannelDock):
    """Concrete implementation for testing."""

    PLATFORM = "test"
    CAPABILITIES = ChannelCapability.RICH_TEXT | ChannelCapability.BUTTONS

    async def send_message(self, channel_id, message, **kwargs):
        return SendResult.ok(
            message_id="test-msg-123",
            platform=self.PLATFORM,
            channel_id=channel_id,
        )


class TestChannelDock:
    """Tests for ChannelDock abstract class."""

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test dock initialization."""
        dock = ConcreteTestDock()
        assert not dock.is_initialized

        result = await dock.initialize()
        assert result is True
        assert dock.is_initialized

    def test_supports_capability(self):
        """Test capability checking."""
        dock = ConcreteTestDock()
        assert dock.supports(ChannelCapability.RICH_TEXT)
        assert dock.supports(ChannelCapability.BUTTONS)
        assert not dock.supports(ChannelCapability.VOICE)

    @pytest.mark.asyncio
    async def test_send_result(self):
        """Test sending a debate result."""
        dock = ConcreteTestDock()
        result = await dock.send_result(
            channel_id="C123",
            result={"decision": "Approve", "confidence": 0.85},
        )
        assert result.success

    @pytest.mark.asyncio
    async def test_send_receipt(self):
        """Test sending a receipt."""
        dock = ConcreteTestDock()
        result = await dock.send_receipt(
            channel_id="C123",
            summary="Decision reached",
            receipt_url="https://example.com/receipt/123",
        )
        assert result.success

    @pytest.mark.asyncio
    async def test_send_error(self):
        """Test sending an error message."""
        dock = ConcreteTestDock()
        result = await dock.send_error(
            channel_id="C123",
            error_message="Something went wrong",
        )
        assert result.success

    @pytest.mark.asyncio
    async def test_send_voice_unsupported(self):
        """Test voice send on dock without voice support."""
        dock = ConcreteTestDock()
        result = await dock.send_voice(
            channel_id="C123",
            audio_data=b"audio_bytes",
        )
        # Should fail because ConcreteTestDock doesn't support voice
        assert result.success is False
        assert "does not support voice" in result.error


# =============================================================================
# DockRegistry Tests
# =============================================================================


class TestDockRegistry:
    """Tests for DockRegistry."""

    def test_register_and_get(self):
        """Test registering and retrieving a dock."""
        registry = DockRegistry()
        registry.register(ConcreteTestDock)

        dock = registry.get_dock("test")
        assert dock is not None
        assert dock.PLATFORM == "test"

    def test_get_unregistered_dock(self):
        """Test getting an unregistered dock."""
        registry = DockRegistry()
        dock = registry.get_dock("nonexistent")
        assert dock is None

    def test_has_dock(self):
        """Test checking dock registration."""
        registry = DockRegistry()
        registry.register(ConcreteTestDock)

        assert registry.has_dock("test")
        assert not registry.has_dock("nonexistent")

    def test_get_platforms(self):
        """Test getting list of platforms."""
        registry = DockRegistry()
        registry.register(ConcreteTestDock)

        platforms = registry.get_platforms()
        assert "test" in platforms

    def test_get_platforms_with_capability(self):
        """Test filtering platforms by capability."""
        registry = DockRegistry()
        registry.register(ConcreteTestDock)

        # ConcreteTestDock has RICH_TEXT
        rich_platforms = registry.get_platforms_with_capability(ChannelCapability.RICH_TEXT)
        assert "test" in rich_platforms

        # ConcreteTestDock doesn't have VOICE
        voice_platforms = registry.get_platforms_with_capability(ChannelCapability.VOICE)
        assert "test" not in voice_platforms

    def test_unregister(self):
        """Test unregistering a dock."""
        registry = DockRegistry()
        registry.register(ConcreteTestDock)
        assert registry.has_dock("test")

        registry.unregister("test")
        assert not registry.has_dock("test")

    def test_dock_caching(self):
        """Test that dock instances are cached."""
        registry = DockRegistry()
        registry.register(ConcreteTestDock)

        dock1 = registry.get_dock("test")
        dock2 = registry.get_dock("test")

        assert dock1 is dock2

    def test_config_override_creates_new_instance(self):
        """Test that config override creates a new instance."""
        registry = DockRegistry()
        registry.register(ConcreteTestDock)

        dock1 = registry.get_dock("test")
        dock2 = registry.get_dock("test", config={"custom": "config"})

        # With config override, should be different instance
        assert dock1 is not dock2


# =============================================================================
# Integration Tests
# =============================================================================


class TestDockIntegration:
    """Integration tests for dock system."""

    @pytest.mark.asyncio
    async def test_full_message_workflow(self):
        """Test complete message workflow."""
        # Create dock
        dock = ConcreteTestDock()
        await dock.initialize()

        # Create message
        message = NormalizedMessage(
            content="Debate complete! The decision is to **approve**.",
            message_type=MessageType.RESULT,
            format=MessageFormat.MARKDOWN,
            title="Decision Reached",
        )
        message.with_button("View Details", "https://example.com/details")

        # Send message
        result = await dock.send_message("channel-123", message)

        assert result.success
        assert result.platform == "test"
        assert result.message_id is not None

    def test_registry_workflow(self):
        """Test registry-based dock lookup."""
        registry = DockRegistry()
        registry.register(ConcreteTestDock)

        # Lookup by platform
        dock = registry.get_dock("test")
        assert dock is not None

        # Check capabilities
        assert dock.supports(ChannelCapability.RICH_TEXT)


# =============================================================================
# Slack/Telegram Dock Tests (if available)
# =============================================================================


class TestSlackDock:
    """Tests for SlackDock implementation."""

    def test_capabilities(self):
        """Test Slack dock capabilities."""
        from aragora.channels.docks.slack import SlackDock

        assert SlackDock.PLATFORM == "slack"
        assert SlackDock.CAPABILITIES & ChannelCapability.RICH_TEXT
        assert SlackDock.CAPABILITIES & ChannelCapability.BUTTONS
        assert SlackDock.CAPABILITIES & ChannelCapability.THREADS

    @pytest.mark.asyncio
    async def test_initialize_without_token(self):
        """Test initialization without token."""
        from aragora.channels.docks.slack import SlackDock

        dock = SlackDock({})
        with patch.dict("os.environ", {}, clear=True):
            result = await dock.initialize()
            # Should fail without token
            assert result is False

    def test_build_payload(self):
        """Test Slack payload building."""
        from aragora.channels.docks.slack import SlackDock

        dock = SlackDock({"token": "test-token"})

        message = NormalizedMessage(
            content="Test message",
            format=MessageFormat.MARKDOWN,
            title="Test Title",
        )
        message.with_button("Click", "https://example.com")

        payload = dock._build_payload("C123", message)

        assert payload["channel"] == "C123"
        assert "blocks" in payload
        assert payload["text"]  # Plain text fallback


class TestTelegramDock:
    """Tests for TelegramDock implementation."""

    def test_capabilities(self):
        """Test Telegram dock capabilities."""
        from aragora.channels.docks.telegram import TelegramDock

        assert TelegramDock.PLATFORM == "telegram"
        assert TelegramDock.CAPABILITIES & ChannelCapability.RICH_TEXT
        assert TelegramDock.CAPABILITIES & ChannelCapability.VOICE

    def test_build_payload(self):
        """Test Telegram payload building."""
        from aragora.channels.docks.telegram import TelegramDock

        dock = TelegramDock({"token": "test-token"})

        message = NormalizedMessage(
            content="Test message",
            format=MessageFormat.MARKDOWN,
        )
        message.with_button("Click", "https://example.com")

        payload = dock._build_payload("123456789", message)

        assert payload["chat_id"] == "123456789"
        assert payload["text"] == "Test message"
        assert "reply_markup" in payload  # Inline keyboard for button
