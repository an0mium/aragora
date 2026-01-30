"""
Tests for chat platform data models.

Tests cover:
- MessageType, InteractionType, UserRole, ChannelType enums
- ChatUser dataclass and enrichment
- ChatChannel dataclass and enrichment
- ChatMessage serialization
- BotCommand parsing
- UserInteraction handling
- MessageBlock, MessageButton, FileAttachment structures
- VoiceMessage data
- SendMessageRequest/Response
- WebhookEvent processing
- ChatEvidence creation and scoring
- ChannelContext aggregation
- MetadataCache TTL functionality
- build_chat_context helper
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch


class TestMessageTypeEnum:
    """Tests for MessageType enum."""

    def test_text_type(self):
        """Should have TEXT type."""
        from aragora.connectors.chat.models import MessageType

        assert MessageType.TEXT.value == "text"

    def test_rich_type(self):
        """Should have RICH type for formatted messages."""
        from aragora.connectors.chat.models import MessageType

        assert MessageType.RICH.value == "rich"

    def test_file_type(self):
        """Should have FILE type."""
        from aragora.connectors.chat.models import MessageType

        assert MessageType.FILE.value == "file"

    def test_voice_type(self):
        """Should have VOICE type."""
        from aragora.connectors.chat.models import MessageType

        assert MessageType.VOICE.value == "voice"

    def test_command_type(self):
        """Should have COMMAND type."""
        from aragora.connectors.chat.models import MessageType

        assert MessageType.COMMAND.value == "command"

    def test_enum_is_string(self):
        """Should inherit from str for easy serialization."""
        from aragora.connectors.chat.models import MessageType

        assert isinstance(MessageType.TEXT, str)
        assert MessageType.TEXT == "text"


class TestInteractionTypeEnum:
    """Tests for InteractionType enum."""

    def test_button_click(self):
        """Should have BUTTON_CLICK type."""
        from aragora.connectors.chat.models import InteractionType

        assert InteractionType.BUTTON_CLICK.value == "button_click"

    def test_select_menu(self):
        """Should have SELECT_MENU type."""
        from aragora.connectors.chat.models import InteractionType

        assert InteractionType.SELECT_MENU.value == "select_menu"

    def test_modal_submit(self):
        """Should have MODAL_SUBMIT type."""
        from aragora.connectors.chat.models import InteractionType

        assert InteractionType.MODAL_SUBMIT.value == "modal_submit"

    def test_shortcut(self):
        """Should have SHORTCUT type."""
        from aragora.connectors.chat.models import InteractionType

        assert InteractionType.SHORTCUT.value == "shortcut"


class TestUserRoleEnum:
    """Tests for UserRole enum."""

    def test_role_hierarchy(self):
        """Should have all role levels."""
        from aragora.connectors.chat.models import UserRole

        assert UserRole.OWNER.value == "owner"
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.MODERATOR.value == "moderator"
        assert UserRole.MEMBER.value == "member"
        assert UserRole.GUEST.value == "guest"
        assert UserRole.UNKNOWN.value == "unknown"


class TestChannelTypeEnum:
    """Tests for ChannelType enum."""

    def test_channel_types(self):
        """Should have all channel types."""
        from aragora.connectors.chat.models import ChannelType

        assert ChannelType.PUBLIC.value == "public"
        assert ChannelType.PRIVATE.value == "private"
        assert ChannelType.DM.value == "dm"
        assert ChannelType.GROUP_DM.value == "group_dm"
        assert ChannelType.THREAD.value == "thread"
        assert ChannelType.UNKNOWN.value == "unknown"


class TestChatUser:
    """Tests for ChatUser dataclass."""

    def test_minimal_init(self):
        """Should initialize with minimal required fields."""
        from aragora.connectors.chat.models import ChatUser

        user = ChatUser(id="U123", platform="slack")

        assert user.id == "U123"
        assert user.platform == "slack"
        assert user.username is None
        assert user.is_bot is False

    def test_full_init(self):
        """Should initialize with all fields."""
        from aragora.connectors.chat.models import ChatUser, UserRole

        user = ChatUser(
            id="U123",
            platform="slack",
            username="johndoe",
            display_name="John Doe",
            email="john@example.com",
            avatar_url="https://example.com/avatar.png",
            is_bot=False,
            timezone="America/New_York",
            language="en",
            locale="en-US",
            role=UserRole.ADMIN,
            status="online",
            metadata={"custom": "data"},
        )

        assert user.display_name == "John Doe"
        assert user.timezone == "America/New_York"
        assert user.role == UserRole.ADMIN

    def test_is_enriched_false(self):
        """Should report not enriched when no enrichment data."""
        from aragora.connectors.chat.models import ChatUser

        user = ChatUser(id="U123", platform="slack")

        assert user.is_enriched is False

    def test_is_enriched_with_timezone(self):
        """Should report enriched when timezone is set."""
        from aragora.connectors.chat.models import ChatUser

        user = ChatUser(id="U123", platform="slack", timezone="UTC")

        assert user.is_enriched is True

    def test_is_enriched_with_language(self):
        """Should report enriched when language is set."""
        from aragora.connectors.chat.models import ChatUser

        user = ChatUser(id="U123", platform="slack", language="en")

        assert user.is_enriched is True

    def test_to_context_dict(self):
        """Should export context dict for prompts."""
        from aragora.connectors.chat.models import ChatUser, UserRole

        user = ChatUser(
            id="U123",
            platform="slack",
            username="johndoe",
            display_name="John Doe",
            timezone="America/New_York",
            language="en",
            locale="en-US",
            role=UserRole.MEMBER,
            status="online",
            is_bot=False,
        )

        context = user.to_context_dict()

        assert context["user_id"] == "U123"
        assert context["username"] == "johndoe"
        assert context["display_name"] == "John Doe"
        assert context["timezone"] == "America/New_York"
        assert context["language"] == "en"
        assert context["role"] == "member"
        assert context["is_bot"] is False


class TestChatChannel:
    """Tests for ChatChannel dataclass."""

    def test_minimal_init(self):
        """Should initialize with minimal fields."""
        from aragora.connectors.chat.models import ChatChannel

        channel = ChatChannel(id="C123", platform="slack")

        assert channel.id == "C123"
        assert channel.platform == "slack"
        assert channel.is_private is False
        assert channel.is_dm is False

    def test_full_init(self):
        """Should initialize with all fields."""
        from aragora.connectors.chat.models import ChatChannel, ChannelType

        channel = ChatChannel(
            id="C123",
            platform="slack",
            name="general",
            is_private=False,
            is_dm=False,
            team_id="T123",
            channel_type=ChannelType.PUBLIC,
            topic="General discussion",
            description="The main channel",
            member_count=50,
        )

        assert channel.name == "general"
        assert channel.topic == "General discussion"
        assert channel.member_count == 50

    def test_is_enriched_false(self):
        """Should report not enriched when no enrichment data."""
        from aragora.connectors.chat.models import ChatChannel

        channel = ChatChannel(id="C123", platform="slack")

        assert channel.is_enriched is False

    def test_is_enriched_with_topic(self):
        """Should report enriched when topic is set."""
        from aragora.connectors.chat.models import ChatChannel

        channel = ChatChannel(id="C123", platform="slack", topic="Discussion")

        assert channel.is_enriched is True

    def test_is_enriched_with_member_count(self):
        """Should report enriched when member_count is set."""
        from aragora.connectors.chat.models import ChatChannel

        channel = ChatChannel(id="C123", platform="slack", member_count=10)

        assert channel.is_enriched is True

    def test_to_context_dict(self):
        """Should export context dict for prompts."""
        from aragora.connectors.chat.models import ChatChannel, ChannelType

        channel = ChatChannel(
            id="C123",
            platform="slack",
            name="general",
            channel_type=ChannelType.PUBLIC,
            topic="General discussion",
            member_count=50,
            team_id="T123",
            is_private=False,
            is_dm=False,
        )

        context = channel.to_context_dict()

        assert context["channel_id"] == "C123"
        assert context["channel_name"] == "general"
        assert context["channel_type"] == "public"
        assert context["topic"] == "General discussion"
        assert context["member_count"] == 50


class TestChatMessage:
    """Tests for ChatMessage dataclass."""

    def test_init_with_required_fields(self):
        """Should initialize with required fields."""
        from aragora.connectors.chat.models import ChatMessage, ChatUser, ChatChannel

        user = ChatUser(id="U123", platform="slack")
        channel = ChatChannel(id="C123", platform="slack")

        message = ChatMessage(
            id="M123",
            platform="slack",
            channel=channel,
            author=user,
            content="Hello, World!",
        )

        assert message.id == "M123"
        assert message.content == "Hello, World!"
        assert message.author.id == "U123"
        assert message.channel.id == "C123"

    def test_to_dict_serialization(self):
        """Should serialize to dictionary."""
        from aragora.connectors.chat.models import (
            ChatMessage,
            ChatUser,
            ChatChannel,
            MessageType,
        )

        user = ChatUser(id="U123", platform="slack", username="john", display_name="John")
        channel = ChatChannel(id="C123", platform="slack", name="general")

        message = ChatMessage(
            id="M123",
            platform="slack",
            channel=channel,
            author=user,
            content="Hello!",
            message_type=MessageType.TEXT,
            thread_id="T456",
        )

        data = message.to_dict()

        assert data["id"] == "M123"
        assert data["platform"] == "slack"
        assert data["content"] == "Hello!"
        assert data["message_type"] == "text"
        assert data["thread_id"] == "T456"
        assert data["channel"]["id"] == "C123"
        assert data["author"]["id"] == "U123"
        assert "timestamp" in data

    def test_default_message_type(self):
        """Should default to TEXT message type."""
        from aragora.connectors.chat.models import (
            ChatMessage,
            ChatUser,
            ChatChannel,
            MessageType,
        )

        user = ChatUser(id="U123", platform="slack")
        channel = ChatChannel(id="C123", platform="slack")

        message = ChatMessage(
            id="M123",
            platform="slack",
            channel=channel,
            author=user,
            content="Test",
        )

        assert message.message_type == MessageType.TEXT


class TestBotCommand:
    """Tests for BotCommand dataclass."""

    def test_init(self):
        """Should initialize bot command."""
        from aragora.connectors.chat.models import BotCommand

        cmd = BotCommand(
            name="debate",
            text="/debate Should we use microservices?",
            args=["Should", "we", "use", "microservices?"],
            options={"rounds": 3},
            platform="slack",
        )

        assert cmd.name == "debate"
        assert cmd.args == ["Should", "we", "use", "microservices?"]
        assert cmd.options["rounds"] == 3

    def test_default_values(self):
        """Should have sensible defaults."""
        from aragora.connectors.chat.models import BotCommand

        cmd = BotCommand(name="test", text="/test")

        assert cmd.args == []
        assert cmd.options == {}
        assert cmd.platform == ""
        assert cmd.response_url is None


class TestUserInteraction:
    """Tests for UserInteraction dataclass."""

    def test_button_interaction(self):
        """Should handle button click interaction."""
        from aragora.connectors.chat.models import UserInteraction, InteractionType

        interaction = UserInteraction(
            id="I123",
            interaction_type=InteractionType.BUTTON_CLICK,
            action_id="approve_btn",
            value="approved",
            platform="slack",
        )

        assert interaction.interaction_type == InteractionType.BUTTON_CLICK
        assert interaction.action_id == "approve_btn"
        assert interaction.value == "approved"

    def test_select_menu_interaction(self):
        """Should handle select menu interaction with multiple values."""
        from aragora.connectors.chat.models import UserInteraction, InteractionType

        interaction = UserInteraction(
            id="I123",
            interaction_type=InteractionType.SELECT_MENU,
            action_id="agent_select",
            values=["claude", "gpt4", "gemini"],
            platform="slack",
        )

        assert interaction.interaction_type == InteractionType.SELECT_MENU
        assert len(interaction.values) == 3


class TestMessageBlock:
    """Tests for MessageBlock dataclass."""

    def test_section_block(self):
        """Should create section block."""
        from aragora.connectors.chat.models import MessageBlock

        block = MessageBlock(
            type="section",
            text="*Bold text*",
            fields=[{"type": "mrkdwn", "text": "Field 1"}],
        )

        assert block.type == "section"
        assert block.text == "*Bold text*"
        assert len(block.fields) == 1


class TestMessageButton:
    """Tests for MessageButton dataclass."""

    def test_default_button(self):
        """Should create default style button."""
        from aragora.connectors.chat.models import MessageButton

        button = MessageButton(
            text="Click me",
            action_id="btn_click",
            value="clicked",
        )

        assert button.text == "Click me"
        assert button.style == "default"
        assert button.url is None

    def test_danger_button(self):
        """Should create danger style button."""
        from aragora.connectors.chat.models import MessageButton

        button = MessageButton(
            text="Delete",
            action_id="btn_delete",
            style="danger",
            confirm={"title": "Are you sure?"},
        )

        assert button.style == "danger"
        assert button.confirm is not None


class TestFileAttachment:
    """Tests for FileAttachment dataclass."""

    def test_file_attachment(self):
        """Should create file attachment."""
        from aragora.connectors.chat.models import FileAttachment

        attachment = FileAttachment(
            id="F123",
            filename="document.pdf",
            content_type="application/pdf",
            size=1024,
            url="https://example.com/document.pdf",
        )

        assert attachment.filename == "document.pdf"
        assert attachment.size == 1024
        assert attachment.content is None


class TestVoiceMessage:
    """Tests for VoiceMessage dataclass."""

    def test_voice_message(self):
        """Should create voice message."""
        from aragora.connectors.chat.models import (
            VoiceMessage,
            ChatChannel,
            ChatUser,
            FileAttachment,
        )

        channel = ChatChannel(id="C123", platform="slack")
        user = ChatUser(id="U123", platform="slack")
        file = FileAttachment(
            id="F123",
            filename="audio.ogg",
            content_type="audio/ogg",
            size=50000,
        )

        voice = VoiceMessage(
            id="V123",
            channel=channel,
            author=user,
            duration_seconds=30.5,
            file=file,
            transcription="Hello, this is a test.",
            platform="slack",
        )

        assert voice.duration_seconds == 30.5
        assert voice.transcription == "Hello, this is a test."


class TestSendMessageRequestResponse:
    """Tests for SendMessageRequest and SendMessageResponse."""

    def test_send_request_minimal(self):
        """Should create minimal send request."""
        from aragora.connectors.chat.models import SendMessageRequest

        request = SendMessageRequest(
            channel_id="C123",
            text="Hello!",
        )

        assert request.channel_id == "C123"
        assert request.text == "Hello!"
        assert request.ephemeral is False

    def test_send_request_ephemeral(self):
        """Should create ephemeral message request."""
        from aragora.connectors.chat.models import SendMessageRequest

        request = SendMessageRequest(
            channel_id="C123",
            text="Only you can see this",
            ephemeral=True,
            ephemeral_user_id="U456",
        )

        assert request.ephemeral is True
        assert request.ephemeral_user_id == "U456"

    def test_send_response_success(self):
        """Should create success response."""
        from aragora.connectors.chat.models import SendMessageResponse

        response = SendMessageResponse(
            success=True,
            message_id="M123",
            channel_id="C123",
            timestamp="1234567890.123456",
        )

        assert response.success is True
        assert response.error is None

    def test_send_response_failure(self):
        """Should create failure response."""
        from aragora.connectors.chat.models import SendMessageResponse

        response = SendMessageResponse(
            success=False,
            error="channel_not_found",
        )

        assert response.success is False
        assert response.error == "channel_not_found"

    def test_message_send_result_alias(self):
        """Should have MessageSendResult alias for backwards compatibility."""
        from aragora.connectors.chat.models import (
            SendMessageResponse,
            MessageSendResult,
        )

        assert MessageSendResult is SendMessageResponse


class TestWebhookEvent:
    """Tests for WebhookEvent dataclass."""

    def test_message_event(self):
        """Should create message webhook event."""
        from aragora.connectors.chat.models import (
            WebhookEvent,
            ChatMessage,
            ChatUser,
            ChatChannel,
        )

        user = ChatUser(id="U123", platform="slack")
        channel = ChatChannel(id="C123", platform="slack")
        message = ChatMessage(
            id="M123",
            platform="slack",
            channel=channel,
            author=user,
            content="Hello!",
        )

        event = WebhookEvent(
            platform="slack",
            event_type="message",
            message=message,
            raw_payload={"type": "message"},
        )

        assert event.platform == "slack"
        assert event.message is not None
        assert event.is_verification is False

    def test_verification_event(self):
        """Should identify verification challenge."""
        from aragora.connectors.chat.models import WebhookEvent

        event = WebhookEvent(
            platform="slack",
            event_type="url_verification",
            challenge="abc123challenge",
        )

        assert event.is_verification is True
        assert event.challenge == "abc123challenge"

    def test_command_event(self):
        """Should create command webhook event."""
        from aragora.connectors.chat.models import WebhookEvent, BotCommand

        cmd = BotCommand(name="debate", text="/debate topic")

        event = WebhookEvent(
            platform="slack",
            event_type="slash_command",
            command=cmd,
        )

        assert event.command is not None
        assert event.command.name == "debate"


class TestChatEvidence:
    """Tests for ChatEvidence dataclass."""

    def test_init(self):
        """Should initialize chat evidence."""
        from aragora.connectors.chat.models import ChatEvidence

        evidence = ChatEvidence(
            id="E123",
            source_id="M123",
            platform="slack",
            channel_id="C123",
            content="This is important information.",
            author_id="U123",
            relevance_score=0.9,
            confidence=0.8,
            freshness=0.95,
        )

        assert evidence.id == "E123"
        assert evidence.source_type == "chat"
        assert evidence.relevance_score == 0.9

    def test_reliability_score(self):
        """Should compute reliability score from factors."""
        from aragora.connectors.chat.models import ChatEvidence

        evidence = ChatEvidence(
            id="E123",
            relevance_score=1.0,
            confidence=1.0,
            freshness=1.0,
        )

        # Formula: 0.5 * relevance + 0.3 * freshness + 0.2 * confidence
        assert evidence.reliability_score == pytest.approx(1.0)

    def test_reliability_score_weighted(self):
        """Should weight reliability factors correctly."""
        from aragora.connectors.chat.models import ChatEvidence

        evidence = ChatEvidence(
            id="E123",
            relevance_score=0.8,
            confidence=0.5,
            freshness=0.6,
        )

        # 0.5 * 0.8 + 0.3 * 0.6 + 0.2 * 0.5 = 0.4 + 0.18 + 0.1 = 0.68
        assert evidence.reliability_score == pytest.approx(0.68)

    def test_source_url_from_metadata(self):
        """Should get source URL from metadata permalink."""
        from aragora.connectors.chat.models import ChatEvidence

        evidence = ChatEvidence(
            id="E123",
            metadata={"permalink": "https://slack.com/archives/C123/p123"},
        )

        assert evidence.source_url == "https://slack.com/archives/C123/p123"

    def test_source_url_none(self):
        """Should return None when no permalink in metadata."""
        from aragora.connectors.chat.models import ChatEvidence

        evidence = ChatEvidence(id="E123")

        assert evidence.source_url is None

    def test_to_dict(self):
        """Should serialize to dictionary."""
        from aragora.connectors.chat.models import ChatEvidence

        evidence = ChatEvidence(
            id="E123",
            source_id="M123",
            platform="slack",
            channel_id="C123",
            channel_name="general",
            content="Test content",
            title="Test",
            author_id="U123",
            relevance_score=0.9,
        )

        data = evidence.to_dict()

        assert data["id"] == "E123"
        assert data["source_type"] == "chat"
        assert data["platform"] == "slack"
        assert data["relevance_score"] == 0.9
        assert "reliability_score" in data
        assert "timestamp" in data

    def test_from_message(self):
        """Should create evidence from ChatMessage."""
        from aragora.connectors.chat.models import (
            ChatEvidence,
            ChatMessage,
            ChatUser,
            ChatChannel,
        )

        user = ChatUser(id="U123", platform="slack", username="john", display_name="John")
        channel = ChatChannel(id="C123", platform="slack", name="general")
        message = ChatMessage(
            id="M123",
            platform="slack",
            channel=channel,
            author=user,
            content="Important discussion point",
            thread_id="T456",
        )

        evidence = ChatEvidence.from_message(
            message=message,
            query="discussion",
            relevance_score=0.85,
        )

        assert evidence.source_id == "M123"
        assert evidence.platform == "slack"
        assert evidence.channel_id == "C123"
        assert evidence.channel_name == "general"
        assert evidence.content == "Important discussion point"
        assert evidence.author_id == "U123"
        assert evidence.author_name == "John"
        assert evidence.relevance_score == 0.85
        assert evidence.source_message is message
        assert evidence.metadata.get("query") == "discussion"


class TestChannelContext:
    """Tests for ChannelContext dataclass."""

    def test_init(self):
        """Should initialize channel context."""
        from aragora.connectors.chat.models import ChannelContext, ChatChannel

        channel = ChatChannel(id="C123", platform="slack", name="general")

        context = ChannelContext(
            channel=channel,
            message_count=100,
            participant_count=25,
        )

        assert context.channel.id == "C123"
        assert context.message_count == 100
        assert context.messages == []

    def test_to_context_string(self):
        """Should generate context string for prompts."""
        from aragora.connectors.chat.models import (
            ChannelContext,
            ChatChannel,
            ChatMessage,
            ChatUser,
        )

        channel = ChatChannel(id="C123", platform="slack", name="general")
        user = ChatUser(id="U123", platform="slack", display_name="Alice")
        messages = [
            ChatMessage(
                id="M1",
                platform="slack",
                channel=channel,
                author=user,
                content="First message",
            ),
            ChatMessage(
                id="M2",
                platform="slack",
                channel=channel,
                author=user,
                content="Second message",
            ),
        ]

        context = ChannelContext(
            channel=channel,
            messages=messages,
            participants=[user],
        )

        result = context.to_context_string()

        assert "# Channel Context: general" in result
        assert "Platform: slack" in result
        assert "Messages: 2" in result
        assert "First message" in result
        assert "Second message" in result

    def test_to_dict(self):
        """Should serialize to dictionary."""
        from aragora.connectors.chat.models import (
            ChannelContext,
            ChatChannel,
            ChatUser,
        )

        channel = ChatChannel(id="C123", platform="slack", name="general")
        user = ChatUser(id="U123", platform="slack", username="alice")

        context = ChannelContext(
            channel=channel,
            participants=[user],
            warnings=["Rate limited"],
        )

        data = context.to_dict()

        assert data["channel"]["id"] == "C123"
        assert len(data["participants"]) == 1
        assert data["warnings"] == ["Rate limited"]
        assert "fetched_at" in data

    def test_from_message_static_method(self):
        """Should create evidence via static from_message method."""
        from aragora.connectors.chat.models import (
            ChannelContext,
            ChatChannel,
            ChatMessage,
            ChatUser,
        )

        channel = ChatChannel(id="C123", platform="slack", name="general")
        user = ChatUser(id="U123", platform="slack", display_name="Bob")
        message = ChatMessage(
            id="M123",
            platform="slack",
            channel=channel,
            author=user,
            content="Test message content",
        )

        # ChannelContext has a static from_message that creates ChatEvidence
        evidence = ChannelContext.from_message(message, relevance_score=0.7)

        assert evidence.source_id == "M123"
        assert evidence.channel_id == "C123"
        assert evidence.relevance_score == 0.7


class TestMetadataCacheEntry:
    """Tests for MetadataCacheEntry dataclass."""

    def test_init(self):
        """Should initialize cache entry."""
        from aragora.connectors.chat.models import MetadataCacheEntry

        entry = MetadataCacheEntry(
            data={"timezone": "UTC"},
            enriched_at=datetime.utcnow(),
            ttl_seconds=3600,
        )

        assert entry.data["timezone"] == "UTC"
        assert entry.ttl_seconds == 3600

    def test_is_expired_false(self):
        """Should report not expired for fresh entry."""
        from aragora.connectors.chat.models import MetadataCacheEntry

        entry = MetadataCacheEntry(
            data={},
            enriched_at=datetime.utcnow(),
            ttl_seconds=3600,
        )

        assert entry.is_expired is False

    def test_is_expired_true(self):
        """Should report expired for old entry."""
        from aragora.connectors.chat.models import MetadataCacheEntry

        entry = MetadataCacheEntry(
            data={},
            enriched_at=datetime.utcnow() - timedelta(seconds=3601),
            ttl_seconds=3600,
        )

        assert entry.is_expired is True


class TestMetadataCache:
    """Tests for MetadataCache class."""

    def test_init_default_ttl(self):
        """Should initialize with default TTL."""
        from aragora.connectors.chat.models import MetadataCache

        cache = MetadataCache()

        assert cache.default_ttl == 3600

    def test_init_custom_ttl(self):
        """Should accept custom TTL."""
        from aragora.connectors.chat.models import MetadataCache

        cache = MetadataCache(default_ttl=7200)

        assert cache.default_ttl == 7200

    def test_set_and_get_user(self):
        """Should cache and retrieve user metadata."""
        from aragora.connectors.chat.models import MetadataCache

        cache = MetadataCache()

        cache.set_user("U123", "slack", {"timezone": "America/New_York"})
        result = cache.get_user("U123", "slack")

        assert result is not None
        assert result["timezone"] == "America/New_York"

    def test_get_user_miss(self):
        """Should return None for cache miss."""
        from aragora.connectors.chat.models import MetadataCache

        cache = MetadataCache()

        result = cache.get_user("U999", "slack")

        assert result is None

    def test_set_and_get_channel(self):
        """Should cache and retrieve channel metadata."""
        from aragora.connectors.chat.models import MetadataCache

        cache = MetadataCache()

        cache.set_channel("C123", "slack", {"topic": "General"})
        result = cache.get_channel("C123", "slack")

        assert result is not None
        assert result["topic"] == "General"

    def test_get_channel_miss(self):
        """Should return None for channel cache miss."""
        from aragora.connectors.chat.models import MetadataCache

        cache = MetadataCache()

        result = cache.get_channel("C999", "slack")

        assert result is None

    def test_invalidate_user(self):
        """Should invalidate user cache entry."""
        from aragora.connectors.chat.models import MetadataCache

        cache = MetadataCache()
        cache.set_user("U123", "slack", {"timezone": "UTC"})

        cache.invalidate_user("U123", "slack")
        result = cache.get_user("U123", "slack")

        assert result is None

    def test_invalidate_channel(self):
        """Should invalidate channel cache entry."""
        from aragora.connectors.chat.models import MetadataCache

        cache = MetadataCache()
        cache.set_channel("C123", "slack", {"topic": "Test"})

        cache.invalidate_channel("C123", "slack")
        result = cache.get_channel("C123", "slack")

        assert result is None

    def test_clear(self):
        """Should clear all cache entries."""
        from aragora.connectors.chat.models import MetadataCache

        cache = MetadataCache()
        cache.set_user("U123", "slack", {"timezone": "UTC"})
        cache.set_channel("C123", "slack", {"topic": "Test"})

        cache.clear()

        assert cache.get_user("U123", "slack") is None
        assert cache.get_channel("C123", "slack") is None

    def test_stats(self):
        """Should return cache statistics."""
        from aragora.connectors.chat.models import MetadataCache

        cache = MetadataCache(default_ttl=1800)
        cache.set_user("U1", "slack", {})
        cache.set_user("U2", "slack", {})
        cache.set_channel("C1", "slack", {})

        stats = cache.stats()

        assert stats["user_entries"] == 2
        assert stats["channel_entries"] == 1
        assert stats["default_ttl_seconds"] == 1800

    def test_expired_entry_cleanup_on_get(self):
        """Should clean up expired entries on get."""
        from aragora.connectors.chat.models import MetadataCache

        cache = MetadataCache(default_ttl=1)

        cache.set_user("U123", "slack", {"test": "data"})

        # Simulate time passing by modifying the entry's enriched_at
        key = "slack:U123"
        cache._user_cache[key].enriched_at = datetime.utcnow() - timedelta(seconds=2)

        result = cache.get_user("U123", "slack")

        assert result is None
        assert key not in cache._user_cache


class TestGetMetadataCache:
    """Tests for get_metadata_cache global function."""

    def test_returns_cache_instance(self):
        """Should return MetadataCache instance."""
        from aragora.connectors.chat.models import get_metadata_cache, MetadataCache

        cache = get_metadata_cache()

        assert isinstance(cache, MetadataCache)

    def test_returns_same_instance(self):
        """Should return same singleton instance."""
        from aragora.connectors.chat.models import get_metadata_cache

        cache1 = get_metadata_cache()
        cache2 = get_metadata_cache()

        assert cache1 is cache2


class TestBuildChatContext:
    """Tests for build_chat_context helper function."""

    def test_build_with_user_and_channel(self):
        """Should build context from user and channel."""
        from aragora.connectors.chat.models import (
            build_chat_context,
            ChatUser,
            ChatChannel,
            UserRole,
            ChannelType,
        )

        user = ChatUser(
            id="U123",
            platform="slack",
            username="john",
            timezone="America/New_York",
            role=UserRole.MEMBER,
        )
        channel = ChatChannel(
            id="C123",
            platform="slack",
            name="general",
            channel_type=ChannelType.PUBLIC,
            topic="Discussion",
        )

        context = build_chat_context(user=user, channel=channel)

        assert "user" in context
        assert "channel" in context
        assert context["user"]["user_id"] == "U123"
        assert context["channel"]["channel_id"] == "C123"

    def test_build_user_only(self):
        """Should build context with user only."""
        from aragora.connectors.chat.models import build_chat_context, ChatUser

        user = ChatUser(id="U123", platform="slack", username="john")

        context = build_chat_context(user=user, include_channel=False)

        assert "user" in context
        assert "channel" not in context

    def test_build_channel_only(self):
        """Should build context with channel only."""
        from aragora.connectors.chat.models import build_chat_context, ChatChannel

        channel = ChatChannel(id="C123", platform="slack", name="general")

        context = build_chat_context(channel=channel, include_user=False)

        assert "channel" in context
        assert "user" not in context

    def test_build_filters_none_values(self):
        """Should filter out None values from context."""
        from aragora.connectors.chat.models import build_chat_context, ChatUser

        user = ChatUser(id="U123", platform="slack")  # username is None

        context = build_chat_context(user=user)

        # username should not be in the filtered context since it's None
        assert "username" not in context.get("user", {})

    def test_build_empty_context(self):
        """Should return empty context when nothing provided."""
        from aragora.connectors.chat.models import build_chat_context

        context = build_chat_context()

        assert context == {}
