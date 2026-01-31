"""
Tests for Moltbot InboxManager - Unified Multi-Channel Message Aggregation.

Tests channel management, message handling, threading, and intent detection.
"""

import asyncio
import pytest
from pathlib import Path

from aragora.extensions.moltbot import (
    InboxManager,
    ChannelConfig,
    ChannelType,
    InboxMessageStatus,
)


class TestChannelManagement:
    """Tests for channel management."""

    @pytest.fixture
    def inbox(self, tmp_path: Path) -> InboxManager:
        """Create an inbox manager for testing."""
        return InboxManager(storage_path=tmp_path / "inbox")

    @pytest.mark.asyncio
    async def test_register_channel(self, inbox: InboxManager):
        """Test registering a channel."""
        config = ChannelConfig(
            type=ChannelType.SMS,
            name="My SMS Channel",
        )

        channel = await inbox.register_channel(
            config=config,
            user_id="user-1",
        )

        assert channel is not None
        assert channel.id is not None
        assert channel.config.type == ChannelType.SMS
        assert channel.config.name == "My SMS Channel"
        assert channel.user_id == "user-1"
        assert channel.status == "active"

    @pytest.mark.asyncio
    async def test_register_channel_with_tenant(self, inbox: InboxManager):
        """Test registering channel with tenant."""
        config = ChannelConfig(type=ChannelType.EMAIL, name="Email")

        channel = await inbox.register_channel(
            config=config,
            user_id="user-1",
            tenant_id="tenant-1",
        )

        assert channel.tenant_id == "tenant-1"

    @pytest.mark.asyncio
    async def test_get_channel(self, inbox: InboxManager):
        """Test getting a channel by ID."""
        config = ChannelConfig(type=ChannelType.WHATSAPP, name="WhatsApp")
        registered = await inbox.register_channel(config=config, user_id="user-1")

        channel = await inbox.get_channel(registered.id)

        assert channel is not None
        assert channel.id == registered.id
        assert channel.config.name == "WhatsApp"

    @pytest.mark.asyncio
    async def test_get_nonexistent_channel(self, inbox: InboxManager):
        """Test getting nonexistent channel."""
        channel = await inbox.get_channel("nonexistent")
        assert channel is None

    @pytest.mark.asyncio
    async def test_list_channels(self, inbox: InboxManager):
        """Test listing channels."""
        for ch_type in [ChannelType.SMS, ChannelType.EMAIL, ChannelType.TELEGRAM]:
            config = ChannelConfig(type=ch_type, name=ch_type.value)
            await inbox.register_channel(config=config, user_id="user-1")

        channels = await inbox.list_channels()
        assert len(channels) == 3

    @pytest.mark.asyncio
    async def test_list_channels_by_user(self, inbox: InboxManager):
        """Test listing channels by user."""
        config1 = ChannelConfig(type=ChannelType.SMS, name="User1 SMS")
        config2 = ChannelConfig(type=ChannelType.SMS, name="User2 SMS")

        await inbox.register_channel(config=config1, user_id="user-1")
        await inbox.register_channel(config=config2, user_id="user-2")

        user1_channels = await inbox.list_channels(user_id="user-1")
        assert len(user1_channels) == 1
        assert user1_channels[0].user_id == "user-1"

    @pytest.mark.asyncio
    async def test_list_channels_by_type(self, inbox: InboxManager):
        """Test listing channels by type."""
        sms_config = ChannelConfig(type=ChannelType.SMS, name="SMS")
        email_config = ChannelConfig(type=ChannelType.EMAIL, name="Email")

        await inbox.register_channel(config=sms_config, user_id="user-1")
        await inbox.register_channel(config=email_config, user_id="user-1")

        sms_channels = await inbox.list_channels(channel_type=ChannelType.SMS)
        assert len(sms_channels) == 1
        assert sms_channels[0].config.type == ChannelType.SMS

    @pytest.mark.asyncio
    async def test_update_channel_status(self, inbox: InboxManager):
        """Test updating channel status."""
        config = ChannelConfig(type=ChannelType.SLACK, name="Slack")
        channel = await inbox.register_channel(config=config, user_id="user-1")

        updated = await inbox.update_channel_status(channel.id, "paused")

        assert updated is not None
        assert updated.status == "paused"

    @pytest.mark.asyncio
    async def test_unregister_channel(self, inbox: InboxManager):
        """Test unregistering a channel."""
        config = ChannelConfig(type=ChannelType.DISCORD, name="Discord")
        channel = await inbox.register_channel(config=config, user_id="user-1")

        result = await inbox.unregister_channel(channel.id)
        assert result is True

        retrieved = await inbox.get_channel(channel.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_channel(self, inbox: InboxManager):
        """Test unregistering nonexistent channel."""
        result = await inbox.unregister_channel("nonexistent")
        assert result is False


class TestMessageReceiving:
    """Tests for receiving messages."""

    @pytest.fixture
    def inbox(self, tmp_path: Path) -> InboxManager:
        """Create an inbox manager for testing."""
        return InboxManager(storage_path=tmp_path / "inbox")

    @pytest.fixture
    async def channel(self, inbox: InboxManager):
        """Create a test channel."""
        config = ChannelConfig(type=ChannelType.TELEGRAM, name="Test Channel")
        return await inbox.register_channel(config=config, user_id="owner")

    @pytest.mark.asyncio
    async def test_receive_message(self, inbox: InboxManager, channel):
        """Test receiving a message."""
        message = await inbox.receive_message(
            channel_id=channel.id,
            user_id="sender-1",
            content="Hello, world!",
        )

        assert message is not None
        assert message.id is not None
        assert message.channel_id == channel.id
        assert message.user_id == "sender-1"
        assert message.content == "Hello, world!"
        assert message.direction == "inbound"
        assert message.content_type == "text"

    @pytest.mark.asyncio
    async def test_receive_message_creates_thread(self, inbox: InboxManager, channel):
        """Test receiving message creates thread."""
        message = await inbox.receive_message(
            channel_id=channel.id,
            user_id="sender-1",
            content="First message",
        )

        assert message.thread_id is not None
        # Message ID should be thread ID for first message
        assert message.thread_id == message.id

    @pytest.mark.asyncio
    async def test_receive_message_with_external_id(self, inbox: InboxManager, channel):
        """Test receiving message with external ID."""
        message = await inbox.receive_message(
            channel_id=channel.id,
            user_id="sender-1",
            content="External message",
            external_id="provider-msg-12345",
        )

        assert message.external_id == "provider-msg-12345"

    @pytest.mark.asyncio
    async def test_receive_message_with_metadata(self, inbox: InboxManager, channel):
        """Test receiving message with metadata."""
        message = await inbox.receive_message(
            channel_id=channel.id,
            user_id="sender-1",
            content="Metadata message",
            metadata={"source": "mobile", "version": "2.0"},
        )

        assert message.metadata["source"] == "mobile"
        assert message.metadata["version"] == "2.0"

    @pytest.mark.asyncio
    async def test_receive_message_invalid_channel(self, inbox: InboxManager):
        """Test receiving message on invalid channel."""
        with pytest.raises(ValueError, match="not found"):
            await inbox.receive_message(
                channel_id="nonexistent",
                user_id="sender-1",
                content="Test",
            )

    @pytest.mark.asyncio
    async def test_receive_message_updates_channel_stats(self, inbox: InboxManager, channel):
        """Test receiving message updates channel stats."""
        initial_count = channel.message_count

        await inbox.receive_message(
            channel_id=channel.id,
            user_id="sender-1",
            content="Message 1",
        )
        await inbox.receive_message(
            channel_id=channel.id,
            user_id="sender-1",
            content="Message 2",
        )

        updated_channel = await inbox.get_channel(channel.id)
        assert updated_channel.message_count == initial_count + 2
        assert updated_channel.last_message_at is not None


class TestMessageSending:
    """Tests for sending messages."""

    @pytest.fixture
    def inbox(self, tmp_path: Path) -> InboxManager:
        """Create an inbox manager for testing."""
        return InboxManager(storage_path=tmp_path / "inbox")

    @pytest.fixture
    async def channel(self, inbox: InboxManager):
        """Create a test channel."""
        config = ChannelConfig(type=ChannelType.WHATSAPP, name="Send Channel")
        return await inbox.register_channel(config=config, user_id="owner")

    @pytest.mark.asyncio
    async def test_send_message(self, inbox: InboxManager, channel):
        """Test sending a message."""
        message = await inbox.send_message(
            channel_id=channel.id,
            user_id="recipient-1",
            content="Hello from the system!",
        )

        assert message is not None
        assert message.id is not None
        assert message.direction == "outbound"
        assert message.content == "Hello from the system!"
        assert message.status == InboxMessageStatus.DELIVERED

    @pytest.mark.asyncio
    async def test_send_message_with_thread(self, inbox: InboxManager, channel):
        """Test sending message in a thread."""
        # First receive a message to create thread
        received = await inbox.receive_message(
            channel_id=channel.id,
            user_id="user-1",
            content="Question",
        )

        # Send reply
        sent = await inbox.send_message(
            channel_id=channel.id,
            user_id="user-1",
            content="Answer",
            thread_id=received.thread_id,
        )

        assert sent.thread_id == received.thread_id

    @pytest.mark.asyncio
    async def test_send_message_reply(self, inbox: InboxManager, channel):
        """Test sending message as reply."""
        original = await inbox.receive_message(
            channel_id=channel.id,
            user_id="user-1",
            content="Original",
        )

        reply = await inbox.send_message(
            channel_id=channel.id,
            user_id="user-1",
            content="Reply",
            reply_to=original.id,
        )

        assert reply.reply_to == original.id
        assert reply.thread_id == original.thread_id

    @pytest.mark.asyncio
    async def test_send_message_invalid_channel(self, inbox: InboxManager):
        """Test sending message on invalid channel."""
        with pytest.raises(ValueError, match="not found"):
            await inbox.send_message(
                channel_id="nonexistent",
                user_id="recipient",
                content="Test",
            )


class TestMessageRetrieval:
    """Tests for message retrieval."""

    @pytest.fixture
    def inbox(self, tmp_path: Path) -> InboxManager:
        """Create an inbox manager for testing."""
        return InboxManager(storage_path=tmp_path / "inbox")

    @pytest.fixture
    async def channel(self, inbox: InboxManager):
        """Create a test channel."""
        config = ChannelConfig(type=ChannelType.SMS, name="Retrieval Channel")
        return await inbox.register_channel(config=config, user_id="owner")

    @pytest.mark.asyncio
    async def test_get_message(self, inbox: InboxManager, channel):
        """Test getting a message by ID."""
        created = await inbox.receive_message(
            channel_id=channel.id,
            user_id="user-1",
            content="Test message",
        )

        message = await inbox.get_message(created.id)

        assert message is not None
        assert message.id == created.id
        assert message.content == "Test message"

    @pytest.mark.asyncio
    async def test_get_nonexistent_message(self, inbox: InboxManager):
        """Test getting nonexistent message."""
        message = await inbox.get_message("nonexistent")
        assert message is None

    @pytest.mark.asyncio
    async def test_list_messages(self, inbox: InboxManager, channel):
        """Test listing messages."""
        for i in range(5):
            await inbox.receive_message(
                channel_id=channel.id,
                user_id="user-1",
                content=f"Message {i}",
            )

        messages = await inbox.list_messages()
        assert len(messages) == 5

    @pytest.mark.asyncio
    async def test_list_messages_by_channel(self, inbox: InboxManager):
        """Test listing messages by channel."""
        config1 = ChannelConfig(type=ChannelType.SMS, name="Channel 1")
        config2 = ChannelConfig(type=ChannelType.EMAIL, name="Channel 2")

        channel1 = await inbox.register_channel(config=config1, user_id="owner")
        channel2 = await inbox.register_channel(config=config2, user_id="owner")

        await inbox.receive_message(channel_id=channel1.id, user_id="u1", content="C1 Msg")
        await inbox.receive_message(channel_id=channel2.id, user_id="u1", content="C2 Msg")
        await inbox.receive_message(channel_id=channel2.id, user_id="u1", content="C2 Msg 2")

        ch1_messages = await inbox.list_messages(channel_id=channel1.id)
        ch2_messages = await inbox.list_messages(channel_id=channel2.id)

        assert len(ch1_messages) == 1
        assert len(ch2_messages) == 2

    @pytest.mark.asyncio
    async def test_list_messages_by_direction(self, inbox: InboxManager, channel):
        """Test listing messages by direction."""
        await inbox.receive_message(channel_id=channel.id, user_id="u1", content="Inbound")
        await inbox.send_message(channel_id=channel.id, user_id="u1", content="Outbound")

        inbound = await inbox.list_messages(direction="inbound")
        outbound = await inbox.list_messages(direction="outbound")

        assert len(inbound) == 1
        assert inbound[0].direction == "inbound"
        assert len(outbound) == 1
        assert outbound[0].direction == "outbound"

    @pytest.mark.asyncio
    async def test_list_messages_with_limit(self, inbox: InboxManager, channel):
        """Test listing messages with limit."""
        for i in range(10):
            await inbox.receive_message(
                channel_id=channel.id,
                user_id="user-1",
                content=f"Message {i}",
            )

        messages = await inbox.list_messages(limit=5)
        assert len(messages) == 5

    @pytest.mark.asyncio
    async def test_list_messages_with_offset(self, inbox: InboxManager, channel):
        """Test listing messages with offset."""
        for i in range(10):
            await inbox.receive_message(
                channel_id=channel.id,
                user_id="user-1",
                content=f"Message {i}",
            )

        messages = await inbox.list_messages(limit=5, offset=5)
        assert len(messages) == 5


class TestThreading:
    """Tests for message threading."""

    @pytest.fixture
    def inbox(self, tmp_path: Path) -> InboxManager:
        """Create an inbox manager for testing."""
        return InboxManager(storage_path=tmp_path / "inbox")

    @pytest.fixture
    async def channel(self, inbox: InboxManager):
        """Create a test channel."""
        config = ChannelConfig(type=ChannelType.TELEGRAM, name="Thread Channel")
        return await inbox.register_channel(config=config, user_id="owner")

    @pytest.mark.asyncio
    async def test_get_thread(self, inbox: InboxManager, channel):
        """Test getting thread messages."""
        # Create thread with multiple messages
        first = await inbox.receive_message(
            channel_id=channel.id,
            user_id="user-1",
            content="First message",
        )

        await inbox.receive_message(
            channel_id=channel.id,
            user_id="user-1",
            content="Second message",
            thread_id=first.thread_id,
        )

        await inbox.send_message(
            channel_id=channel.id,
            user_id="user-1",
            content="Response",
            thread_id=first.thread_id,
        )

        thread = await inbox.get_thread(first.thread_id)

        assert len(thread) == 3
        # Should be sorted by created_at
        assert thread[0].content == "First message"

    @pytest.mark.asyncio
    async def test_auto_thread_on_reply(self, inbox: InboxManager, channel):
        """Test auto-threading when replying."""
        original = await inbox.receive_message(
            channel_id=channel.id,
            user_id="user-1",
            content="Original",
        )

        reply = await inbox.receive_message(
            channel_id=channel.id,
            user_id="user-2",
            content="Reply",
            reply_to=original.id,
        )

        # Reply should inherit thread from original
        assert reply.thread_id == original.thread_id

    @pytest.mark.asyncio
    async def test_empty_thread(self, inbox: InboxManager):
        """Test getting empty/nonexistent thread."""
        thread = await inbox.get_thread("nonexistent-thread")
        assert thread == []


class TestIntentDetection:
    """Tests for message intent detection."""

    @pytest.fixture
    def inbox(self, tmp_path: Path) -> InboxManager:
        """Create an inbox manager for testing."""
        return InboxManager(storage_path=tmp_path / "inbox")

    @pytest.fixture
    async def channel(self, inbox: InboxManager):
        """Create a test channel."""
        config = ChannelConfig(type=ChannelType.WEB, name="Intent Channel")
        return await inbox.register_channel(config=config, user_id="owner")

    @pytest.mark.asyncio
    async def test_detect_help_intent(self, inbox: InboxManager, channel):
        """Test detecting help request intent."""
        message = await inbox.receive_message(
            channel_id=channel.id,
            user_id="user-1",
            content="I need help with my account",
        )

        assert message.intent == "help_request"

    @pytest.mark.asyncio
    async def test_detect_purchase_intent(self, inbox: InboxManager, channel):
        """Test detecting purchase intent."""
        message = await inbox.receive_message(
            channel_id=channel.id,
            user_id="user-1",
            content="I want to buy the premium plan",
        )

        assert message.intent == "purchase_intent"

    @pytest.mark.asyncio
    async def test_detect_cancellation_intent(self, inbox: InboxManager, channel):
        """Test detecting cancellation intent."""
        message = await inbox.receive_message(
            channel_id=channel.id,
            user_id="user-1",
            content="Please cancel my subscription",
        )

        assert message.intent == "cancellation"

    @pytest.mark.asyncio
    async def test_detect_question_intent(self, inbox: InboxManager, channel):
        """Test detecting question intent."""
        message = await inbox.receive_message(
            channel_id=channel.id,
            user_id="user-1",
            content="How do I reset my password?",
        )

        assert message.intent == "question"

    @pytest.mark.asyncio
    async def test_detect_general_intent(self, inbox: InboxManager, channel):
        """Test detecting general intent."""
        message = await inbox.receive_message(
            channel_id=channel.id,
            user_id="user-1",
            content="Thanks for the update",
        )

        assert message.intent == "general"


class TestMessageStatus:
    """Tests for message status management."""

    @pytest.fixture
    def inbox(self, tmp_path: Path) -> InboxManager:
        """Create an inbox manager for testing."""
        return InboxManager(storage_path=tmp_path / "inbox")

    @pytest.fixture
    async def channel(self, inbox: InboxManager):
        """Create a test channel."""
        config = ChannelConfig(type=ChannelType.SLACK, name="Status Channel")
        return await inbox.register_channel(config=config, user_id="owner")

    @pytest.mark.asyncio
    async def test_mark_read(self, inbox: InboxManager, channel):
        """Test marking message as read."""
        message = await inbox.receive_message(
            channel_id=channel.id,
            user_id="user-1",
            content="Test message",
        )

        updated = await inbox.mark_read(message.id)

        assert updated is not None
        assert updated.status == InboxMessageStatus.READ
        assert updated.read_at is not None

    @pytest.mark.asyncio
    async def test_mark_read_nonexistent(self, inbox: InboxManager):
        """Test marking nonexistent message as read."""
        result = await inbox.mark_read("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_archive_message(self, inbox: InboxManager, channel):
        """Test archiving a message."""
        message = await inbox.receive_message(
            channel_id=channel.id,
            user_id="user-1",
            content="To archive",
        )

        updated = await inbox.archive_message(message.id)

        assert updated is not None
        assert updated.status == InboxMessageStatus.ARCHIVED

    @pytest.mark.asyncio
    async def test_archive_nonexistent(self, inbox: InboxManager):
        """Test archiving nonexistent message."""
        result = await inbox.archive_message("nonexistent")
        assert result is None


class TestMessageHandlers:
    """Tests for message handlers."""

    @pytest.fixture
    def inbox(self, tmp_path: Path) -> InboxManager:
        """Create an inbox manager for testing."""
        return InboxManager(storage_path=tmp_path / "inbox")

    @pytest.mark.asyncio
    async def test_register_handler(self, inbox: InboxManager):
        """Test registering a message handler."""
        handled_messages = []

        async def sms_handler(message, channel):
            handled_messages.append(message)

        inbox.register_handler(ChannelType.SMS, sms_handler)

        config = ChannelConfig(type=ChannelType.SMS, name="SMS")
        channel = await inbox.register_channel(config=config, user_id="owner")

        await inbox.send_message(
            channel_id=channel.id,
            user_id="recipient",
            content="Test SMS",
        )

        assert len(handled_messages) == 1
        assert handled_messages[0].content == "Test SMS"


class TestInboxStats:
    """Tests for inbox statistics."""

    @pytest.fixture
    def inbox(self, tmp_path: Path) -> InboxManager:
        """Create an inbox manager for testing."""
        return InboxManager(storage_path=tmp_path / "inbox")

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, inbox: InboxManager):
        """Test getting stats with no data."""
        stats = await inbox.get_stats()

        assert stats["channels_total"] == 0
        assert stats["channels_active"] == 0
        assert stats["messages_total"] == 0
        assert stats["messages_inbound"] == 0
        assert stats["messages_outbound"] == 0
        assert stats["threads_total"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_with_data(self, inbox: InboxManager):
        """Test getting stats with data."""
        # Create channels
        sms_config = ChannelConfig(type=ChannelType.SMS, name="SMS")
        email_config = ChannelConfig(type=ChannelType.EMAIL, name="Email")

        sms_channel = await inbox.register_channel(config=sms_config, user_id="owner")
        email_channel = await inbox.register_channel(config=email_config, user_id="owner")

        # Create messages
        await inbox.receive_message(channel_id=sms_channel.id, user_id="u1", content="In 1")
        await inbox.receive_message(channel_id=sms_channel.id, user_id="u1", content="In 2")
        await inbox.send_message(channel_id=email_channel.id, user_id="u1", content="Out 1")

        stats = await inbox.get_stats()

        assert stats["channels_total"] == 2
        assert stats["channels_active"] == 2
        assert stats["messages_total"] == 3
        assert stats["messages_inbound"] == 2
        assert stats["messages_outbound"] == 1
        assert stats["by_channel"]["sms"] == 2
        assert stats["by_channel"]["email"] == 1
