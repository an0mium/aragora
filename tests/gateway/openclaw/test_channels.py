"""
Tests for OpenClaw channel formatters and message types.
"""

import pytest
from datetime import datetime, timezone

from aragora.gateway.openclaw.adapter import (
    ActionResult,
    ActionStatus,
    ChannelFormatter,
    ChannelMapping,
    DiscordFormatter,
    OpenClawAction,
    OpenClawActionType,
    OpenClawChannel,
    OpenClawMessage,
    OpenClawSession,
    SessionState,
    SlackFormatter,
    TelegramFormatter,
    WhatsAppFormatter,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestOpenClawChannel:
    """Test OpenClawChannel enum."""

    def test_all_channels(self):
        assert OpenClawChannel.WHATSAPP == "whatsapp"
        assert OpenClawChannel.TELEGRAM == "telegram"
        assert OpenClawChannel.SLACK == "slack"
        assert OpenClawChannel.DISCORD == "discord"
        assert OpenClawChannel.SMS == "sms"
        assert OpenClawChannel.EMAIL == "email"
        assert OpenClawChannel.WEB == "web"
        assert OpenClawChannel.VOICE == "voice"
        assert OpenClawChannel.TEAMS == "teams"
        assert OpenClawChannel.MATRIX == "matrix"

    def test_channel_count(self):
        assert len(OpenClawChannel) == 10


class TestOpenClawActionType:
    """Test OpenClawActionType enum."""

    def test_browser_actions(self):
        assert OpenClawActionType.BROWSER_NAVIGATE == "browser_navigate"
        assert OpenClawActionType.BROWSER_CLICK == "browser_click"
        assert OpenClawActionType.BROWSER_SCREENSHOT == "browser_screenshot"

    def test_file_actions(self):
        assert OpenClawActionType.FILE_READ == "file_read"
        assert OpenClawActionType.FILE_WRITE == "file_write"

    def test_code_actions(self):
        assert OpenClawActionType.CODE_RUN == "code_run"
        assert OpenClawActionType.CODE_EVAL == "code_eval"

    def test_custom_action(self):
        assert OpenClawActionType.CUSTOM == "custom"


class TestSessionState:
    """Test SessionState enum."""

    def test_all_states(self):
        assert SessionState.CREATED == "created"
        assert SessionState.ACTIVE == "active"
        assert SessionState.PAUSED == "paused"
        assert SessionState.EXPIRED == "expired"
        assert SessionState.TERMINATED == "terminated"


class TestActionStatus:
    """Test ActionStatus enum."""

    def test_all_statuses(self):
        assert ActionStatus.PENDING == "pending"
        assert ActionStatus.RUNNING == "running"
        assert ActionStatus.COMPLETED == "completed"
        assert ActionStatus.FAILED == "failed"
        assert ActionStatus.TIMEOUT == "timeout"
        assert ActionStatus.CANCELLED == "cancelled"


# =============================================================================
# OpenClawMessage Tests
# =============================================================================


class TestOpenClawMessage:
    """Test OpenClawMessage dataclass."""

    def test_text_message(self):
        msg = OpenClawMessage(
            message_id="msg-1",
            type="text",
            content="Hello world",
            channel=OpenClawChannel.SLACK,
        )
        assert msg.type == "text"
        assert msg.content == "Hello world"
        assert msg.channel == OpenClawChannel.SLACK

    def test_to_dict(self):
        msg = OpenClawMessage(
            message_id="msg-1",
            type="text",
            content="Hello",
            channel=OpenClawChannel.TELEGRAM,
            sender_id="user-1",
        )
        d = msg.to_dict()
        assert d["message_id"] == "msg-1"
        assert d["type"] == "text"
        assert d["channel"] == "telegram"
        assert d["sender_id"] == "user-1"

    def test_to_dict_string_channel(self):
        msg = OpenClawMessage(
            message_id="msg-1",
            type="text",
            content="Hello",
            channel="custom_channel",
        )
        d = msg.to_dict()
        assert d["channel"] == "custom_channel"

    def test_from_dict(self):
        data = {
            "message_id": "msg-1",
            "type": "image",
            "content": "https://example.com/img.png",
            "channel": "whatsapp",
            "sender_id": "user-1",
        }
        msg = OpenClawMessage.from_dict(data)
        assert msg.message_id == "msg-1"
        assert msg.type == "image"
        assert msg.channel == OpenClawChannel.WHATSAPP

    def test_from_dict_unknown_channel(self):
        data = {"type": "text", "content": "hi", "channel": "unknown_platform"}
        msg = OpenClawMessage.from_dict(data)
        assert msg.channel == "unknown_platform"

    def test_roundtrip_serialization(self):
        original = OpenClawMessage(
            message_id="msg-1",
            type="text",
            content="Hello",
            channel=OpenClawChannel.DISCORD,
            reply_to="msg-0",
            thread_id="thread-1",
        )
        d = original.to_dict()
        restored = OpenClawMessage.from_dict(d)
        assert restored.message_id == original.message_id
        assert restored.type == original.type
        assert restored.content == original.content
        assert restored.reply_to == original.reply_to
        assert restored.thread_id == original.thread_id


# =============================================================================
# OpenClawAction Tests
# =============================================================================


class TestOpenClawAction:
    """Test OpenClawAction dataclass."""

    def test_basic_action(self):
        action = OpenClawAction(
            action_type=OpenClawActionType.BROWSER_NAVIGATE,
            parameters={"url": "https://example.com"},
        )
        assert action.action_type == OpenClawActionType.BROWSER_NAVIGATE
        assert action.timeout == 60  # default

    def test_to_dict(self):
        action = OpenClawAction(
            action_type=OpenClawActionType.CODE_RUN,
            parameters={"code": "print('hello')"},
            timeout=30,
        )
        d = action.to_dict()
        assert d["action_type"] == "code_run"
        assert d["timeout"] == 30

    def test_from_dict(self):
        data = {
            "action_type": "file_read",
            "parameters": {"path": "/tmp/test.txt"},
            "timeout": 10,
        }
        action = OpenClawAction.from_dict(data)
        assert action.action_type == OpenClawActionType.FILE_READ
        assert action.parameters["path"] == "/tmp/test.txt"

    def test_from_dict_custom_action(self):
        data = {"action_type": "my_custom_action", "parameters": {}}
        action = OpenClawAction.from_dict(data)
        assert action.action_type == "my_custom_action"

    def test_roundtrip_serialization(self):
        original = OpenClawAction(
            action_type=OpenClawActionType.HTTP_REQUEST,
            parameters={"url": "https://api.example.com", "method": "POST"},
            timeout=15,
            retry_count=2,
            callback_url="https://hooks.example.com/cb",
        )
        d = original.to_dict()
        restored = OpenClawAction.from_dict(d)
        assert restored.action_type == original.action_type
        assert restored.parameters == original.parameters
        assert restored.timeout == original.timeout
        assert restored.retry_count == original.retry_count


# =============================================================================
# OpenClawSession Tests
# =============================================================================


class TestOpenClawSession:
    """Test OpenClawSession dataclass."""

    def test_new_session(self):
        session = OpenClawSession(
            session_id="sess-1",
            user_id="user-1",
            channel=OpenClawChannel.SLACK,
        )
        assert session.state == SessionState.CREATED
        assert session.tenant_id is None

    def test_is_active_when_active(self):
        session = OpenClawSession(
            session_id="sess-1",
            user_id="user-1",
            channel=OpenClawChannel.WEB,
            state=SessionState.ACTIVE,
        )
        assert session.is_active() is True

    def test_is_active_when_created(self):
        session = OpenClawSession(
            session_id="sess-1",
            user_id="user-1",
            channel=OpenClawChannel.WEB,
            state=SessionState.CREATED,
        )
        assert session.is_active() is False

    def test_is_active_when_expired_by_time(self):
        session = OpenClawSession(
            session_id="sess-1",
            user_id="user-1",
            channel=OpenClawChannel.WEB,
            state=SessionState.ACTIVE,
            expires_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        assert session.is_active() is False

    def test_update_activity(self):
        session = OpenClawSession(
            session_id="sess-1",
            user_id="user-1",
            channel=OpenClawChannel.WEB,
        )
        old_activity = session.last_activity
        session.update_activity()
        assert session.last_activity >= old_activity

    def test_to_dict(self):
        session = OpenClawSession(
            session_id="sess-1",
            user_id="user-1",
            channel=OpenClawChannel.TELEGRAM,
            state=SessionState.ACTIVE,
            tenant_id="tenant-1",
        )
        d = session.to_dict()
        assert d["session_id"] == "sess-1"
        assert d["channel"] == "telegram"
        assert d["state"] == "active"
        assert d["tenant_id"] == "tenant-1"

    def test_from_dict(self):
        data = {
            "session_id": "sess-1",
            "user_id": "user-1",
            "channel": "slack",
            "state": "active",
            "tenant_id": "t-1",
        }
        session = OpenClawSession.from_dict(data)
        assert session.channel == OpenClawChannel.SLACK
        assert session.state == SessionState.ACTIVE

    def test_from_dict_unknown_channel(self):
        data = {"session_id": "s1", "user_id": "u1", "channel": "custom"}
        session = OpenClawSession.from_dict(data)
        assert session.channel == "custom"

    def test_roundtrip_serialization(self):
        original = OpenClawSession(
            session_id="sess-1",
            user_id="user-1",
            channel=OpenClawChannel.DISCORD,
            state=SessionState.ACTIVE,
            tenant_id="t-1",
            metadata={"theme": "dark"},
            context={"history": []},
        )
        d = original.to_dict()
        restored = OpenClawSession.from_dict(d)
        assert restored.session_id == original.session_id
        assert restored.channel == original.channel
        assert restored.state == original.state
        assert restored.metadata == original.metadata


# =============================================================================
# ChannelMapping Tests
# =============================================================================


class TestChannelMapping:
    """Test ChannelMapping dataclass."""

    def test_basic_mapping(self):
        mapping = ChannelMapping(
            aragora_channel="slack",
            openclaw_channel=OpenClawChannel.SLACK,
        )
        assert mapping.response_routing == "direct"  # default

    def test_to_dict(self):
        mapping = ChannelMapping(
            aragora_channel="whatsapp",
            openclaw_channel=OpenClawChannel.WHATSAPP,
            formatter="whatsapp_formatter",
            response_routing="async",
        )
        d = mapping.to_dict()
        assert d["aragora_channel"] == "whatsapp"
        assert d["openclaw_channel"] == "whatsapp"
        assert d["formatter"] == "whatsapp_formatter"


# =============================================================================
# ActionResult Tests
# =============================================================================


class TestActionResult:
    """Test ActionResult dataclass."""

    def test_success_result(self):
        result = ActionResult(
            action_id="act-1",
            status=ActionStatus.COMPLETED,
            result={"data": "hello"},
            execution_time_ms=150,
        )
        assert result.status == ActionStatus.COMPLETED
        assert result.error is None

    def test_failure_result(self):
        result = ActionResult(
            action_id="act-1",
            status=ActionStatus.FAILED,
            error="Permission denied",
        )
        assert result.status == ActionStatus.FAILED

    def test_to_dict(self):
        result = ActionResult(
            action_id="act-1",
            status=ActionStatus.COMPLETED,
            result={"output": "done"},
            execution_time_ms=250,
        )
        d = result.to_dict()
        assert d["action_id"] == "act-1"
        assert d["status"] == "completed"
        assert d["result"]["output"] == "done"


# =============================================================================
# Channel Formatter Tests
# =============================================================================


class TestChannelFormatter:
    """Test base ChannelFormatter."""

    @pytest.fixture
    def session(self):
        return OpenClawSession(
            session_id="sess-1",
            user_id="user-1",
            channel=OpenClawChannel.WEB,
        )

    def test_default_format_outgoing(self, session):
        formatter = ChannelFormatter()
        msg = OpenClawMessage(
            message_id="msg-1",
            type="text",
            content="Hello",
            channel=OpenClawChannel.WEB,
        )
        result = formatter.format_outgoing(msg, session)
        assert result["content"] == "Hello"

    def test_default_parse_incoming(self, session):
        formatter = ChannelFormatter()
        raw = {"type": "text", "content": "Hello", "channel": "web"}
        msg = formatter.parse_incoming(raw, session)
        assert msg.type == "text"
        assert msg.content == "Hello"


class TestWhatsAppFormatter:
    """Test WhatsApp message formatting."""

    @pytest.fixture
    def session(self):
        return OpenClawSession(
            session_id="sess-1",
            user_id="user-1",
            channel=OpenClawChannel.WHATSAPP,
            metadata={"phone_number": "+1234567890"},
        )

    @pytest.fixture
    def formatter(self):
        return WhatsAppFormatter()

    def test_text_message(self, formatter, session):
        msg = OpenClawMessage(
            message_id="msg-1",
            type="text",
            content="Hello WhatsApp!",
            channel=OpenClawChannel.WHATSAPP,
        )
        result = formatter.format_outgoing(msg, session)
        assert result["messaging_product"] == "whatsapp"
        assert result["to"] == "+1234567890"
        assert result["type"] == "text"
        assert result["text"]["body"] == "Hello WhatsApp!"

    def test_image_message(self, formatter, session):
        msg = OpenClawMessage(
            message_id="msg-1",
            type="image",
            content="https://example.com/img.png",
            channel=OpenClawChannel.WHATSAPP,
        )
        result = formatter.format_outgoing(msg, session)
        assert result["type"] == "image"
        assert result["image"]["link"] == "https://example.com/img.png"

    def test_video_message(self, formatter, session):
        msg = OpenClawMessage(
            message_id="msg-1",
            type="video",
            content="https://example.com/video.mp4",
            channel=OpenClawChannel.WHATSAPP,
        )
        result = formatter.format_outgoing(msg, session)
        assert result["type"] == "video"
        assert result["video"]["link"] == "https://example.com/video.mp4"

    def test_file_message(self, formatter, session):
        msg = OpenClawMessage(
            message_id="msg-1",
            type="file",
            content="https://example.com/doc.pdf",
            channel=OpenClawChannel.WHATSAPP,
        )
        result = formatter.format_outgoing(msg, session)
        assert result["type"] == "document"
        assert result["document"]["link"] == "https://example.com/doc.pdf"

    def test_type_mapping(self, formatter):
        assert formatter._map_message_type("text") == "text"
        assert formatter._map_message_type("image") == "image"
        assert formatter._map_message_type("file") == "document"
        assert formatter._map_message_type("unknown") == "text"


class TestTelegramFormatter:
    """Test Telegram message formatting."""

    @pytest.fixture
    def session(self):
        return OpenClawSession(
            session_id="sess-1",
            user_id="user-1",
            channel=OpenClawChannel.TELEGRAM,
            metadata={"chat_id": 123456789},
        )

    @pytest.fixture
    def formatter(self):
        return TelegramFormatter()

    def test_text_message(self, formatter, session):
        msg = OpenClawMessage(
            message_id="msg-1",
            type="text",
            content="Hello Telegram!",
            channel=OpenClawChannel.TELEGRAM,
        )
        result = formatter.format_outgoing(msg, session)
        assert result["chat_id"] == 123456789
        assert result["text"] == "Hello Telegram!"
        assert result["parse_mode"] == "HTML"

    def test_image_message(self, formatter, session):
        msg = OpenClawMessage(
            message_id="msg-1",
            type="image",
            content="https://example.com/img.png",
            channel=OpenClawChannel.TELEGRAM,
        )
        result = formatter.format_outgoing(msg, session)
        assert result["photo"] == "https://example.com/img.png"

    def test_reply_to_message(self, formatter, session):
        msg = OpenClawMessage(
            message_id="msg-2",
            type="text",
            content="Reply",
            channel=OpenClawChannel.TELEGRAM,
            reply_to="msg-1",
        )
        result = formatter.format_outgoing(msg, session)
        assert result["reply_to_message_id"] == "msg-1"

    def test_no_reply_to(self, formatter, session):
        msg = OpenClawMessage(
            message_id="msg-1",
            type="text",
            content="Hello",
            channel=OpenClawChannel.TELEGRAM,
        )
        result = formatter.format_outgoing(msg, session)
        assert "reply_to_message_id" not in result


class TestSlackFormatter:
    """Test Slack message formatting."""

    @pytest.fixture
    def session(self):
        return OpenClawSession(
            session_id="sess-1",
            user_id="user-1",
            channel=OpenClawChannel.SLACK,
            metadata={"channel_id": "C1234567890"},
        )

    @pytest.fixture
    def formatter(self):
        return SlackFormatter()

    def test_text_message(self, formatter, session):
        msg = OpenClawMessage(
            message_id="msg-1",
            type="text",
            content="Hello Slack!",
            channel=OpenClawChannel.SLACK,
        )
        result = formatter.format_outgoing(msg, session)
        assert result["channel"] == "C1234567890"
        assert result["text"] == "Hello Slack!"

    def test_text_with_blocks(self, formatter, session):
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "*Bold*"}}]
        msg = OpenClawMessage(
            message_id="msg-1",
            type="text",
            content="Bold text",
            channel=OpenClawChannel.SLACK,
            metadata={"blocks": blocks},
        )
        result = formatter.format_outgoing(msg, session)
        assert result["blocks"] == blocks

    def test_image_attachment(self, formatter, session):
        msg = OpenClawMessage(
            message_id="msg-1",
            type="image",
            content="https://example.com/img.png",
            channel=OpenClawChannel.SLACK,
            metadata={"alt_text": "A screenshot"},
        )
        result = formatter.format_outgoing(msg, session)
        assert result["text"] == "A screenshot"
        assert len(result["attachments"]) == 1

    def test_thread_reply(self, formatter, session):
        msg = OpenClawMessage(
            message_id="msg-1",
            type="text",
            content="Thread reply",
            channel=OpenClawChannel.SLACK,
            thread_id="1234567890.123456",
        )
        result = formatter.format_outgoing(msg, session)
        assert result["thread_ts"] == "1234567890.123456"


class TestDiscordFormatter:
    """Test Discord message formatting."""

    @pytest.fixture
    def session(self):
        return OpenClawSession(
            session_id="sess-1",
            user_id="user-1",
            channel=OpenClawChannel.DISCORD,
        )

    @pytest.fixture
    def formatter(self):
        return DiscordFormatter()

    def test_text_message(self, formatter, session):
        msg = OpenClawMessage(
            message_id="msg-1",
            type="text",
            content="Hello Discord!",
            channel=OpenClawChannel.DISCORD,
        )
        result = formatter.format_outgoing(msg, session)
        assert result["content"] == "Hello Discord!"

    def test_text_with_embeds(self, formatter, session):
        embeds = [{"title": "Test", "description": "A test embed"}]
        msg = OpenClawMessage(
            message_id="msg-1",
            type="text",
            content="See embed",
            channel=OpenClawChannel.DISCORD,
            metadata={"embeds": embeds},
        )
        result = formatter.format_outgoing(msg, session)
        assert result["embeds"] == embeds

    def test_image_embed(self, formatter, session):
        msg = OpenClawMessage(
            message_id="msg-1",
            type="image",
            content="https://example.com/img.png",
            channel=OpenClawChannel.DISCORD,
        )
        result = formatter.format_outgoing(msg, session)
        assert result["embeds"][0]["image"]["url"] == "https://example.com/img.png"

    def test_file_attachment(self, formatter, session):
        msg = OpenClawMessage(
            message_id="msg-1",
            type="file",
            content="https://example.com/doc.pdf",
            channel=OpenClawChannel.DISCORD,
            metadata={"description": "Important doc"},
        )
        result = formatter.format_outgoing(msg, session)
        assert result["content"] == "Important doc"
        assert result["attachments"][0]["url"] == "https://example.com/doc.pdf"
