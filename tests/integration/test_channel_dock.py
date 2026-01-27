"""
Integration tests for Channel Dock routing system.

Tests the end-to-end flow of:
- Channel router routing to correct docks
- Platform-specific dock handling
- Debate result delivery to platforms
- Retry and error handling
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from dataclasses import dataclass
from typing import Any, Optional

from aragora.channels.dock import ChannelDock, ChannelCapability, SendResult
from aragora.channels.normalized import NormalizedMessage, MessageFormat
from aragora.channels.registry import DockRegistry
from aragora.channels.router import ChannelRouter
from aragora.server.debate_origin import (
    DebateOrigin,
    register_debate_origin,
    get_debate_origin,
)


# =============================================================================
# Mock Dock Implementation
# =============================================================================


class MockDock(ChannelDock):
    """Mock dock for testing."""

    PLATFORM = "test"
    CAPABILITIES = ChannelCapability.RICH_TEXT | ChannelCapability.BUTTONS

    def __init__(self):
        super().__init__()
        self._initialized = False
        self.sent_messages = []
        self.sent_results = []
        self.sent_errors = []

    async def initialize(self) -> bool:
        self._initialized = True
        return True

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def send_message(
        self,
        channel_id: str,
        message: NormalizedMessage,
        **kwargs: Any,
    ) -> SendResult:
        self.sent_messages.append(
            {
                "channel_id": channel_id,
                "message": message,
                "kwargs": kwargs,
            }
        )
        return SendResult.ok(
            message_id=f"msg-{len(self.sent_messages)}",
            platform=self.PLATFORM,
            channel_id=channel_id,
        )

    async def send_result(
        self,
        channel_id: str,
        result: dict[str, Any],
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        self.sent_results.append(
            {
                "channel_id": channel_id,
                "result": result,
                "thread_id": thread_id,
                "kwargs": kwargs,
            }
        )
        return SendResult.ok(
            message_id=f"result-{len(self.sent_results)}",
            platform=self.PLATFORM,
            channel_id=channel_id,
        )

    async def send_error(
        self,
        channel_id: str,
        error_message: str,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        self.sent_errors.append(
            {
                "channel_id": channel_id,
                "error_message": error_message,
                "thread_id": thread_id,
            }
        )
        return SendResult.ok(
            platform=self.PLATFORM,
            channel_id=channel_id,
        )


class FailingDock(MockDock):
    """Dock that fails on first attempt."""

    PLATFORM = "failing"

    def __init__(self, fail_count: int = 1):
        super().__init__()
        self.fail_count = fail_count
        self.attempt_count = 0

    async def send_result(
        self,
        channel_id: str,
        result: dict[str, Any],
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        self.attempt_count += 1
        if self.attempt_count <= self.fail_count:
            return SendResult.fail(
                error="Temporary failure",
                platform=self.PLATFORM,
                channel_id=channel_id,
            )
        return await super().send_result(channel_id, result, thread_id, **kwargs)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_dock():
    """Create a mock dock."""
    return MockDock()


@pytest.fixture
def failing_dock():
    """Create a failing dock."""
    return FailingDock(fail_count=1)


@pytest.fixture
def registry(mock_dock):
    """Create a registry with mock dock."""
    reg = DockRegistry()
    reg._dock_instances["test"] = mock_dock
    mock_dock._initialized = True
    return reg


@pytest.fixture
def router(registry):
    """Create a router with mock registry."""
    return ChannelRouter(registry=registry)


@pytest.fixture
def sample_debate_result():
    """Sample debate result for testing."""
    return {
        "debate_id": "debate-123",
        "final_answer": "The proposed solution is optimal.",
        "confidence": 0.85,
        "consensus_reached": True,
        "winning_agent": "claude-opus",
        "rounds": 3,
        "votes": {"for": 2, "against": 1},
    }


@pytest.fixture
def debate_origin():
    """Create a sample debate origin."""
    return DebateOrigin(
        debate_id="debate-123",
        platform="test",
        channel_id="channel-456",
        user_id="user-789",
        thread_id="thread-111",
    )


# =============================================================================
# Router Tests
# =============================================================================


class TestChannelRouterRouting:
    """Tests for channel router routing to correct docks."""

    @pytest.mark.asyncio
    async def test_routes_to_correct_platform(self, router, mock_dock, sample_debate_result):
        """Verify router routes to the correct platform dock."""
        result = await router.route_result(
            platform="test",
            channel_id="channel-123",
            result=sample_debate_result,
        )

        assert result.success is True
        assert len(mock_dock.sent_results) == 1
        assert mock_dock.sent_results[0]["channel_id"] == "channel-123"

    @pytest.mark.asyncio
    async def test_routes_with_thread_id(self, router, mock_dock, sample_debate_result):
        """Verify router preserves thread_id."""
        await router.route_result(
            platform="test",
            channel_id="channel-123",
            result=sample_debate_result,
            thread_id="thread-456",
        )

        assert mock_dock.sent_results[0]["thread_id"] == "thread-456"

    @pytest.mark.asyncio
    async def test_routes_error_to_channel(self, router, mock_dock):
        """Verify router routes errors correctly."""
        result = await router.route_error(
            platform="test",
            channel_id="channel-123",
            error_message="Something went wrong",
            debate_id="debate-123",
        )

        assert result.success is True
        assert len(mock_dock.sent_errors) == 1
        assert mock_dock.sent_errors[0]["error_message"] == "Something went wrong"

    @pytest.mark.asyncio
    async def test_unknown_platform_fails(self, router):
        """Verify routing to unknown platform returns failure."""
        result = await router.route_result(
            platform="nonexistent",
            channel_id="channel-123",
            result={"test": "data"},
        )

        assert result.success is False
        assert "not found" in result.error.lower() or "unknown" in result.error.lower()


# =============================================================================
# Platform Dock Tests
# =============================================================================


class TestTelegramDockReceivesResult:
    """Tests for Telegram dock receiving debate results."""

    @pytest.mark.asyncio
    async def test_telegram_dock_mock(self, registry, sample_debate_result):
        """Test Telegram dock (mocked) receives results correctly."""
        telegram_dock = MockDock()
        telegram_dock.PLATFORM = "telegram"
        telegram_dock._initialized = True
        registry._dock_instances["telegram"] = telegram_dock

        router = ChannelRouter(registry=registry)

        result = await router.route_result(
            platform="telegram",
            channel_id="chat-123",
            result=sample_debate_result,
            thread_id="reply-456",
        )

        assert result.success is True
        assert result.platform == "telegram"
        assert len(telegram_dock.sent_results) == 1


class TestWhatsAppDockReceivesResult:
    """Tests for WhatsApp dock receiving debate results."""

    @pytest.mark.asyncio
    async def test_whatsapp_dock_mock(self, registry, sample_debate_result):
        """Test WhatsApp dock (mocked) receives results correctly."""
        whatsapp_dock = MockDock()
        whatsapp_dock.PLATFORM = "whatsapp"
        whatsapp_dock._initialized = True
        registry._dock_instances["whatsapp"] = whatsapp_dock

        router = ChannelRouter(registry=registry)

        result = await router.route_result(
            platform="whatsapp",
            channel_id="+1234567890",
            result=sample_debate_result,
        )

        assert result.success is True
        assert result.platform == "whatsapp"


class TestSlackDockReceivesResult:
    """Tests for Slack dock receiving debate results."""

    @pytest.mark.asyncio
    async def test_slack_dock_mock(self, registry, sample_debate_result):
        """Test Slack dock (mocked) receives results correctly."""
        slack_dock = MockDock()
        slack_dock.PLATFORM = "slack"
        slack_dock.CAPABILITIES = (
            ChannelCapability.RICH_TEXT | ChannelCapability.BUTTONS | ChannelCapability.THREADS
        )
        slack_dock._initialized = True
        registry._dock_instances["slack"] = slack_dock

        router = ChannelRouter(registry=registry)

        result = await router.route_result(
            platform="slack",
            channel_id="C12345678",
            result=sample_debate_result,
            thread_id="1234567890.123456",
        )

        assert result.success is True
        assert result.platform == "slack"


# =============================================================================
# Retry and Error Handling Tests
# =============================================================================


class TestChannelDockRetryOnFailure:
    """Tests for dock retry behavior on failure."""

    @pytest.mark.asyncio
    async def test_dock_returns_failure(self, failing_dock, sample_debate_result):
        """Test dock returns failure result."""
        result = await failing_dock.send_result(
            channel_id="channel-123",
            result=sample_debate_result,
        )

        assert result.success is False
        assert "failure" in result.error.lower()

    @pytest.mark.asyncio
    async def test_dock_succeeds_after_retry(self, failing_dock, sample_debate_result):
        """Test dock succeeds on retry after initial failure."""
        # First attempt fails
        result1 = await failing_dock.send_result(
            channel_id="channel-123",
            result=sample_debate_result,
        )
        assert result1.success is False

        # Second attempt succeeds
        result2 = await failing_dock.send_result(
            channel_id="channel-123",
            result=sample_debate_result,
        )
        assert result2.success is True

    @pytest.mark.asyncio
    async def test_router_handles_dock_failure(self, registry, sample_debate_result):
        """Test router handles dock failure gracefully."""
        failing = FailingDock(fail_count=100)  # Always fails
        failing._initialized = True
        registry._dock_instances["failing"] = failing

        router = ChannelRouter(registry=registry)

        result = await router.route_result(
            platform="failing",
            channel_id="channel-123",
            result=sample_debate_result,
        )

        assert result.success is False


# =============================================================================
# Debate Origin Integration Tests
# =============================================================================


class TestDebateOriginRouting:
    """Tests for debate origin tracking and result routing."""

    def test_register_debate_origin(self):
        """Test registering a debate origin."""
        origin = register_debate_origin(
            debate_id="debate-new",
            platform="telegram",
            channel_id="chat-123",
            user_id="user-456",
            thread_id="reply-789",
        )

        assert origin.debate_id == "debate-new"
        assert origin.platform == "telegram"
        assert origin.channel_id == "chat-123"

    def test_get_debate_origin(self):
        """Test retrieving a debate origin."""
        register_debate_origin(
            debate_id="debate-lookup",
            platform="slack",
            channel_id="C12345",
            user_id="U67890",
        )

        origin = get_debate_origin("debate-lookup")

        assert origin is not None
        assert origin.platform == "slack"

    def test_unknown_debate_origin_returns_none(self):
        """Test unknown debate returns None."""
        origin = get_debate_origin("nonexistent-debate")
        assert origin is None

    @pytest.mark.asyncio
    async def test_route_result_to_origin(self, registry, mock_dock, sample_debate_result):
        """Test routing result back to debate origin."""
        # Register origin
        register_debate_origin(
            debate_id="debate-123",
            platform="test",
            channel_id="channel-origin",
            user_id="user-123",
            thread_id="thread-origin",
        )

        router = ChannelRouter(registry=registry)
        origin = get_debate_origin("debate-123")

        result = await router.route_result(
            platform=origin.platform,
            channel_id=origin.channel_id,
            result=sample_debate_result,
            thread_id=origin.thread_id,
        )

        assert result.success is True
        assert mock_dock.sent_results[0]["channel_id"] == "channel-origin"
        assert mock_dock.sent_results[0]["thread_id"] == "thread-origin"


# =============================================================================
# Capability Tests
# =============================================================================


class TestDockCapabilities:
    """Tests for dock capability detection."""

    def test_supports_capability(self, mock_dock):
        """Test checking dock capabilities."""
        assert mock_dock.supports(ChannelCapability.RICH_TEXT) is True
        assert mock_dock.supports(ChannelCapability.BUTTONS) is True
        assert mock_dock.supports(ChannelCapability.VOICE) is False

    def test_registry_finds_capable_docks(self, registry, mock_dock):
        """Test registry finds docks with specific capabilities."""
        platforms = registry.get_platforms_with_capability(ChannelCapability.RICH_TEXT)
        assert "test" in platforms

        platforms = registry.get_platforms_with_capability(ChannelCapability.VOICE)
        assert "test" not in platforms


# =============================================================================
# Normalized Message Tests
# =============================================================================


class TestNormalizedMessage:
    """Tests for normalized message creation and conversion."""

    def test_create_simple_message(self):
        """Test creating a simple normalized message."""
        msg = NormalizedMessage(
            content="Hello, world!",
            message_type="notification",
        )

        assert msg.content == "Hello, world!"
        assert msg.format == MessageFormat.PLAIN

    def test_message_with_buttons(self):
        """Test creating message with buttons."""
        msg = NormalizedMessage(content="Choose an option")
        msg = msg.with_button("Yes", "action:yes", style="primary")
        msg = msg.with_button("No", "action:no", style="danger")

        assert msg.has_buttons() is True
        assert len(msg.buttons) == 2

    def test_message_to_plain_text(self):
        """Test converting message to plain text."""
        msg = NormalizedMessage(
            content="**Bold** and _italic_",
            format=MessageFormat.MARKDOWN,
        )

        plain = msg.to_plain_text()
        assert isinstance(plain, str)

    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        msg = NormalizedMessage(
            content="Test message",
            title="Test Title",
            format=MessageFormat.MARKDOWN,
        )

        d = msg.to_dict()
        assert d["content"] == "Test message"
        assert d["title"] == "Test Title"
        assert d["format"] == "markdown"


# =============================================================================
# Send Result Tests
# =============================================================================


class TestSendResult:
    """Tests for SendResult creation and handling."""

    def test_ok_result(self):
        """Test creating successful result."""
        result = SendResult.ok(
            message_id="msg-123",
            platform="test",
            channel_id="channel-456",
        )

        assert result.success is True
        assert result.message_id == "msg-123"
        assert result.error is None

    def test_fail_result(self):
        """Test creating failure result."""
        result = SendResult.fail(
            error="Connection timeout",
            platform="test",
            channel_id="channel-456",
        )

        assert result.success is False
        assert result.error == "Connection timeout"
        assert result.message_id is None
