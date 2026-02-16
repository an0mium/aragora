"""
Tests for all channel dock implementations.

Tests Discord, Email, Google Chat, Teams, Telegram, and WhatsApp docks
for message delivery, payload building, and error handling.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from aragora.channels.dock import ChannelDock, ChannelCapability, SendResult, MessageType
from aragora.channels.normalized import (
    NormalizedMessage,
    MessageFormat,
    MessageButton,
    MessageAttachment,
)


# =============================================================================
# Shared helpers
# =============================================================================


def _make_message(**kwargs):
    """Create a NormalizedMessage for testing."""
    return NormalizedMessage(
        content=kwargs.get("content", "Test message"),
        message_type=kwargs.get("message_type", "notification"),
        format=kwargs.get("format", MessageFormat.MARKDOWN),
        title=kwargs.get("title", None),
        thread_id=kwargs.get("thread_id", None),
        reply_to=kwargs.get("reply_to", None),
        metadata=kwargs.get("metadata", {}),
    )


def _make_httpx_response(status_code=200, json_data=None, text="OK"):
    """Create a mock httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.content = text.encode() if text else b""
    resp.json.return_value = json_data or {}
    return resp


# =============================================================================
# Discord Dock Tests
# =============================================================================


class TestDiscordDock:
    """Tests for DiscordDock."""

    def _make_dock(self, config=None):
        from aragora.channels.docks.discord import DiscordDock

        return DiscordDock(config)

    def test_platform(self):
        """Test platform identifier."""
        dock = self._make_dock()
        assert dock.PLATFORM == "discord"

    def test_capabilities(self):
        """Test supported capabilities."""
        dock = self._make_dock()
        assert dock.supports(ChannelCapability.RICH_TEXT)
        assert dock.supports(ChannelCapability.THREADS)
        assert dock.supports(ChannelCapability.FILES)
        assert dock.supports(ChannelCapability.REACTIONS)
        assert dock.supports(ChannelCapability.INLINE_IMAGES)
        assert not dock.supports(ChannelCapability.VOICE)
        assert not dock.supports(ChannelCapability.BUTTONS)

    @pytest.mark.asyncio
    async def test_initialize_with_config_token(self):
        """Test initialization with token in config."""
        dock = self._make_dock({"token": "test-token"})
        result = await dock.initialize()
        assert result is True
        assert dock.is_initialized

    @pytest.mark.asyncio
    async def test_initialize_with_env_token(self):
        """Test initialization with env var token."""
        dock = self._make_dock()
        with patch.dict("os.environ", {"DISCORD_BOT_TOKEN": "env-token"}):
            result = await dock.initialize()
            assert result is True

    @pytest.mark.asyncio
    async def test_initialize_no_token(self):
        """Test initialization fails without token."""
        dock = self._make_dock()
        with patch.dict("os.environ", {}, clear=True):
            result = await dock.initialize()
            assert result is False

    @pytest.mark.asyncio
    async def test_send_message_no_token(self):
        """Test send fails when no token configured."""
        dock = self._make_dock()
        msg = _make_message()
        result = await dock.send_message("channel-1", msg)
        assert result.success is False
        assert "not configured" in result.error

    @pytest.mark.asyncio
    async def test_send_message_success(self):
        """Test successful message send."""
        dock = self._make_dock({"token": "test-token"})
        await dock.initialize()
        msg = _make_message(content="Hello Discord")

        mock_resp = _make_httpx_response(200, {"id": "msg-123"})
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await dock.send_message("channel-1", msg)
            assert result.success is True
            assert result.message_id == "msg-123"

    @pytest.mark.asyncio
    async def test_send_message_api_error(self):
        """Test handling API error."""
        dock = self._make_dock({"token": "test-token"})
        await dock.initialize()
        msg = _make_message()

        mock_resp = _make_httpx_response(403, {"message": "Forbidden"})
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await dock.send_message("channel-1", msg)
            assert result.success is False

    @pytest.mark.asyncio
    async def test_send_message_exception(self):
        """Test handling network exception."""
        dock = self._make_dock({"token": "test-token"})
        await dock.initialize()
        msg = _make_message()

        mock_client = AsyncMock()
        mock_client.post.side_effect = OSError("Connection failed")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await dock.send_message("channel-1", msg)
            assert result.success is False
            assert "Connection failed" in result.error

    def test_build_payload_plain_text(self):
        """Test payload building with plain text."""
        dock = self._make_dock({"token": "t"})
        msg = _make_message(content="Simple text", format=MessageFormat.PLAIN)
        payload = dock._build_payload("ch-1", msg)
        assert payload["content"] == "Simple text"

    def test_build_payload_with_title(self):
        """Test payload building with title."""
        dock = self._make_dock({"token": "t"})
        msg = _make_message(content="Body text", title="My Title")
        payload = dock._build_payload("ch-1", msg)
        assert "**My Title**" in payload["content"]

    def test_build_payload_content_truncation(self):
        """Test content is truncated to 2000 chars."""
        dock = self._make_dock({"token": "t"})
        long_content = "A" * 3000
        msg = _make_message(content=long_content)
        payload = dock._build_payload("ch-1", msg)
        assert len(payload["content"]) <= 2000

    def test_build_payload_reply_to(self):
        """Test reply_to creates message_reference."""
        dock = self._make_dock({"token": "t"})
        msg = _make_message(reply_to="orig-msg-id")
        payload = dock._build_payload("ch-1", msg)
        assert payload["message_reference"]["message_id"] == "orig-msg-id"

    def test_build_payload_embeds_with_title(self):
        """Test embeds are created when title is present."""
        dock = self._make_dock({"token": "t"})
        msg = _make_message(content="Content", title="Title")
        payload = dock._build_payload("ch-1", msg)
        assert "embeds" in payload
        assert payload["embeds"][0]["title"] == "Title"

    def test_build_payload_embeds_with_image(self):
        """Test image attachments in embeds."""
        dock = self._make_dock({"token": "t"})
        msg = _make_message(title="Photo")
        msg.with_attachment("image", url="https://example.com/img.png")
        payload = dock._build_payload("ch-1", msg)
        assert "embeds" in payload
        assert payload["embeds"][0]["image"]["url"] == "https://example.com/img.png"

    @pytest.mark.asyncio
    async def test_send_voice_no_token(self):
        """Test voice send fails without token."""
        dock = self._make_dock()
        result = await dock.send_voice("ch-1", b"audio-data")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_send_voice_success(self):
        """Test successful voice send."""
        dock = self._make_dock({"token": "test-token"})
        await dock.initialize()

        mock_resp = _make_httpx_response(200, {"id": "voice-123"})
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await dock.send_voice("ch-1", b"audio-data", text="Transcript")
            assert result.success is True
            assert result.message_id == "voice-123"

    def test_repr(self):
        """Test string representation."""
        dock = self._make_dock()
        assert "DiscordDock" in repr(dock)
        assert "discord" in repr(dock)


# =============================================================================
# Email Dock Tests
# =============================================================================


class TestEmailDock:
    """Tests for EmailDock."""

    def _make_dock(self, config=None):
        from aragora.channels.docks.email import EmailDock

        return EmailDock(config)

    def test_platform(self):
        """Test platform identifier."""
        dock = self._make_dock()
        assert dock.PLATFORM == "email"

    def test_capabilities(self):
        """Test supported capabilities."""
        dock = self._make_dock()
        assert dock.supports(ChannelCapability.RICH_TEXT)
        assert dock.supports(ChannelCapability.FILES)
        assert not dock.supports(ChannelCapability.VOICE)
        assert not dock.supports(ChannelCapability.BUTTONS)

    @pytest.mark.asyncio
    async def test_initialize_succeeds(self):
        """Test initialization always succeeds."""
        dock = self._make_dock()
        result = await dock.initialize()
        assert result is True
        assert dock.is_initialized

    @pytest.mark.asyncio
    async def test_send_message_no_notification_system(self):
        """Test send fails when notification system unavailable."""
        dock = self._make_dock()
        dock._notification_system_available = False
        dock._initialized = True
        msg = _make_message()
        result = await dock.send_message("user@example.com", msg)
        assert result.success is False
        assert "not available" in result.error

    def test_build_email_body_plain(self):
        """Test email body with plain text message."""
        dock = self._make_dock()
        msg = _make_message(content="Plain content", format=MessageFormat.PLAIN)
        body = dock._build_email_body(msg)
        assert "<pre>" in body
        assert "Plain content" in body

    def test_build_email_body_html(self):
        """Test email body with HTML message."""
        dock = self._make_dock()
        msg = _make_message(content="<b>Bold</b>", format=MessageFormat.HTML)
        body = dock._build_email_body(msg)
        assert "<b>Bold</b>" in body

    def test_build_email_body_markdown(self):
        """Test email body with markdown message."""
        dock = self._make_dock()
        msg = _make_message(content="**Bold** text", format=MessageFormat.MARKDOWN)
        body = dock._build_email_body(msg)
        assert "<strong>" in body

    def test_build_email_body_with_title(self):
        """Test email body includes title."""
        dock = self._make_dock()
        msg = _make_message(content="Body", title="Email Title")
        body = dock._build_email_body(msg)
        assert "<h2>" in body
        assert "Email Title" in body

    def test_build_email_body_with_buttons(self):
        """Test email body renders buttons as links."""
        dock = self._make_dock()
        msg = _make_message(content="Click below")
        msg.with_button("View Report", "https://example.com/report")
        body = dock._build_email_body(msg)
        assert 'href="https://example.com/report"' in body
        assert "View Report" in body

    def test_build_email_body_button_non_http_skipped(self):
        """Test non-HTTP button actions are skipped."""
        dock = self._make_dock()
        msg = _make_message(content="Body")
        msg.with_button("Action", "do_something")
        body = dock._build_email_body(msg)
        assert "do_something" not in body

    def test_escape_html(self):
        """Test HTML escaping."""
        dock = self._make_dock()
        result = dock._escape_html('<script>alert("xss")</script>')
        assert "&lt;script&gt;" in result
        assert "&quot;" in result

    def test_markdown_to_html_bold(self):
        """Test markdown bold conversion."""
        dock = self._make_dock()
        result = dock._markdown_to_html("**bold**")
        assert "<strong>bold</strong>" in result

    def test_markdown_to_html_italic(self):
        """Test markdown italic conversion."""
        dock = self._make_dock()
        result = dock._markdown_to_html("*italic*")
        assert "<em>italic</em>" in result

    def test_markdown_to_html_code(self):
        """Test markdown inline code conversion."""
        dock = self._make_dock()
        result = dock._markdown_to_html("`code`")
        assert "<code>code</code>" in result

    def test_markdown_to_html_link(self):
        """Test markdown link conversion."""
        dock = self._make_dock()
        result = dock._markdown_to_html("[Click](https://example.com)")
        assert '<a href="https://example.com">Click</a>' in result

    def test_markdown_to_html_line_breaks(self):
        """Test markdown line break conversion."""
        dock = self._make_dock()
        result = dock._markdown_to_html("Line1\nLine2")
        assert "<br>" in result

    def test_markdown_to_html_paragraphs(self):
        """Test markdown paragraph conversion."""
        dock = self._make_dock()
        result = dock._markdown_to_html("Para1\n\nPara2")
        assert "</p><p>" in result

    def test_repr(self):
        """Test string representation."""
        dock = self._make_dock()
        assert "EmailDock" in repr(dock)


# =============================================================================
# Google Chat Dock Tests
# =============================================================================


class TestGoogleChatDock:
    """Tests for GoogleChatDock."""

    def _make_dock(self, config=None):
        from aragora.channels.docks.google_chat import GoogleChatDock

        return GoogleChatDock(config)

    def test_platform(self):
        """Test platform identifier."""
        dock = self._make_dock()
        assert dock.PLATFORM == "google_chat"

    def test_capabilities(self):
        """Test supported capabilities."""
        dock = self._make_dock()
        assert dock.supports(ChannelCapability.RICH_TEXT)
        assert dock.supports(ChannelCapability.BUTTONS)
        assert dock.supports(ChannelCapability.THREADS)
        assert dock.supports(ChannelCapability.FILES)
        assert dock.supports(ChannelCapability.CARDS)
        assert not dock.supports(ChannelCapability.VOICE)

    @pytest.mark.asyncio
    async def test_send_message_no_connector(self):
        """Test send fails when connector is not available."""
        dock = self._make_dock()
        msg = _make_message()
        result = await dock.send_message("spaces/test", msg)
        assert result.success is False
        assert "not configured" in result.error

    def test_build_card_sections_basic(self):
        """Test card sections from basic message."""
        dock = self._make_dock()
        msg = _make_message(content="Hello")
        sections = dock._build_card_sections(msg)
        # Should have content section
        assert len(sections) >= 1
        has_text = any("textParagraph" in w for s in sections for w in s.get("widgets", []))
        assert has_text

    def test_build_card_sections_with_title(self):
        """Test card sections include header for titled message."""
        dock = self._make_dock()
        msg = _make_message(content="Body", title="My Title")
        sections = dock._build_card_sections(msg)
        headers = [s.get("header") for s in sections if "header" in s]
        assert "My Title" in headers

    def test_build_card_sections_with_http_buttons(self):
        """Test card sections include URL buttons."""
        dock = self._make_dock()
        msg = _make_message(content="Click")
        msg.with_button("Visit", "https://example.com")
        sections = dock._build_card_sections(msg)
        button_sections = [
            s for s in sections if any("buttonList" in w for w in s.get("widgets", []))
        ]
        assert len(button_sections) == 1

    def test_build_card_sections_with_action_buttons(self):
        """Test card sections include action buttons."""
        dock = self._make_dock()
        msg = _make_message(content="Do action")
        msg.with_button("Act", "do_action")
        sections = dock._build_card_sections(msg)
        button_sections = [
            s for s in sections if any("buttonList" in w for w in s.get("widgets", []))
        ]
        assert len(button_sections) == 1

    @pytest.mark.asyncio
    async def test_send_message_success(self):
        """Test successful send via connector."""
        dock = self._make_dock()
        connector = AsyncMock()
        connector.send_message.return_value = MagicMock(success=True, message_id="gchat-msg-1")
        dock._connector = connector
        dock._initialized = True

        msg = _make_message(content="Hello GChat")
        result = await dock.send_message("spaces/ABC", msg)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_send_message_connector_failure(self):
        """Test handling connector failure."""
        dock = self._make_dock()
        connector = AsyncMock()
        connector.send_message.return_value = MagicMock(success=False, error="API error")
        dock._connector = connector
        dock._initialized = True

        msg = _make_message(content="Hello")
        result = await dock.send_message("spaces/ABC", msg)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_send_message_exception(self):
        """Test handling exception during send."""
        dock = self._make_dock()
        connector = AsyncMock()
        connector.send_message.side_effect = OSError("Network error")
        dock._connector = connector
        dock._initialized = True

        msg = _make_message(content="Hello")
        result = await dock.send_message("spaces/ABC", msg)
        assert result.success is False
        assert "Network error" in result.error

    @pytest.mark.asyncio
    async def test_send_result_no_connector(self):
        """Test send_result fails when no connector."""
        dock = self._make_dock()
        result = await dock.send_result("spaces/ABC", {"decision": "Yes"})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_send_result_success(self):
        """Test successful debate result send."""
        dock = self._make_dock()
        connector = AsyncMock()
        connector.send_message.return_value = MagicMock(success=True, message_id="res-1")
        dock._connector = connector
        dock._initialized = True

        debate_result = {
            "consensus_reached": True,
            "final_answer": "Proceed with plan A",
            "confidence": 0.85,
            "task": "Review plan",
            "id": "debate-1",
        }
        result = await dock.send_result("spaces/ABC", debate_result)
        assert result.success is True

    def test_repr(self):
        """Test string representation."""
        dock = self._make_dock()
        assert "GoogleChatDock" in repr(dock)


# =============================================================================
# Teams Dock Tests
# =============================================================================


class TestTeamsDock:
    """Tests for TeamsDock."""

    def _make_dock(self, config=None):
        from aragora.channels.docks.teams import TeamsDock

        return TeamsDock(config)

    def test_platform(self):
        """Test platform identifier."""
        dock = self._make_dock()
        assert dock.PLATFORM == "teams"

    def test_capabilities(self):
        """Test supported capabilities."""
        dock = self._make_dock()
        assert dock.supports(ChannelCapability.RICH_TEXT)
        assert dock.supports(ChannelCapability.BUTTONS)
        assert dock.supports(ChannelCapability.THREADS)
        assert dock.supports(ChannelCapability.FILES)
        assert dock.supports(ChannelCapability.CARDS)
        assert not dock.supports(ChannelCapability.VOICE)

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test initialization."""
        dock = self._make_dock({"webhook_url": "https://webhook.example.com"})
        result = await dock.initialize()
        assert result is True
        assert dock.is_initialized

    @pytest.mark.asyncio
    async def test_send_message_no_webhook(self):
        """Test send fails without webhook URL."""
        dock = self._make_dock()
        await dock.initialize()
        msg = _make_message()
        result = await dock.send_message("not-a-url", msg)
        assert result.success is False
        assert "not configured" in result.error

    @pytest.mark.asyncio
    async def test_send_message_success(self):
        """Test successful message send."""
        dock = self._make_dock({"webhook_url": "https://webhook.example.com"})
        await dock.initialize()
        msg = _make_message(content="Hello Teams")

        mock_resp = _make_httpx_response(200)
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await dock.send_message("ch-1", msg)
            assert result.success is True

    @pytest.mark.asyncio
    async def test_send_message_api_error(self):
        """Test handling API error."""
        dock = self._make_dock({"webhook_url": "https://webhook.example.com"})
        await dock.initialize()
        msg = _make_message()

        mock_resp = _make_httpx_response(400, text="Bad Request")
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await dock.send_message("ch-1", msg)
            assert result.success is False

    @pytest.mark.asyncio
    async def test_send_message_exception(self):
        """Test handling network exception."""
        dock = self._make_dock({"webhook_url": "https://webhook.example.com"})
        await dock.initialize()
        msg = _make_message()

        mock_client = AsyncMock()
        mock_client.post.side_effect = OSError("Timeout")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await dock.send_message("ch-1", msg)
            assert result.success is False

    def test_build_payload_basic(self):
        """Test basic payload structure."""
        dock = self._make_dock()
        msg = _make_message(content="Hello")
        payload = dock._build_payload("ch-1", msg)
        assert payload["type"] == "message"
        assert "attachments" in payload
        card = payload["attachments"][0]["content"]
        assert card["type"] == "AdaptiveCard"

    def test_build_payload_with_title(self):
        """Test payload includes title block."""
        dock = self._make_dock()
        msg = _make_message(content="Body", title="Card Title")
        payload = dock._build_payload("ch-1", msg)
        card = payload["attachments"][0]["content"]
        title_blocks = [b for b in card["body"] if "Card Title" in b.get("text", "")]
        assert len(title_blocks) >= 1

    def test_build_payload_with_buttons(self):
        """Test payload includes action buttons."""
        dock = self._make_dock()
        msg = _make_message(content="Content")
        msg.with_button("Open", "https://example.com/page")
        payload = dock._build_payload("ch-1", msg)
        card = payload["attachments"][0]["content"]
        assert "actions" in card
        assert card["actions"][0]["url"] == "https://example.com/page"

    def test_build_payload_content_truncation(self):
        """Test content truncation to 3000 chars."""
        dock = self._make_dock()
        long = "A" * 5000
        msg = _make_message(content=long)
        payload = dock._build_payload("ch-1", msg)
        card = payload["attachments"][0]["content"]
        text_blocks = [b for b in card["body"] if b.get("type") == "TextBlock" and b.get("wrap")]
        for block in text_blocks:
            assert len(block["text"]) <= 3000

    @pytest.mark.asyncio
    async def test_send_result(self):
        """Test sending a debate result."""
        dock = self._make_dock({"webhook_url": "https://webhook.example.com"})
        await dock.initialize()

        mock_resp = _make_httpx_response(200)
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await dock.send_result(
                "ch-1",
                {"consensus_reached": True, "confidence": 0.8, "final_answer": "Go ahead"},
            )
            assert result.success is True

    def test_repr(self):
        """Test string representation."""
        dock = self._make_dock()
        assert "TeamsDock" in repr(dock)


# =============================================================================
# Telegram Dock Tests
# =============================================================================


class TestTelegramDock:
    """Tests for TelegramDock."""

    def _make_dock(self, config=None):
        from aragora.channels.docks.telegram import TelegramDock

        return TelegramDock(config)

    def test_platform(self):
        """Test platform identifier."""
        dock = self._make_dock()
        assert dock.PLATFORM == "telegram"

    def test_capabilities(self):
        """Test supported capabilities."""
        dock = self._make_dock()
        assert dock.supports(ChannelCapability.RICH_TEXT)
        assert dock.supports(ChannelCapability.BUTTONS)
        assert dock.supports(ChannelCapability.VOICE)
        assert dock.supports(ChannelCapability.FILES)
        assert dock.supports(ChannelCapability.INLINE_IMAGES)
        assert not dock.supports(ChannelCapability.CARDS)

    @pytest.mark.asyncio
    async def test_initialize_with_config_token(self):
        """Test initialization with config token."""
        dock = self._make_dock({"token": "123456:ABC"})
        result = await dock.initialize()
        assert result is True
        assert dock.is_initialized

    @pytest.mark.asyncio
    async def test_initialize_with_env_token(self):
        """Test initialization with env var token."""
        dock = self._make_dock()
        with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "env-token"}):
            result = await dock.initialize()
            assert result is True

    @pytest.mark.asyncio
    async def test_initialize_no_token(self):
        """Test initialization fails without token."""
        dock = self._make_dock()
        with patch.dict("os.environ", {}, clear=True):
            result = await dock.initialize()
            assert result is False

    @pytest.mark.asyncio
    async def test_send_message_no_token(self):
        """Test send fails without token."""
        dock = self._make_dock()
        msg = _make_message()
        result = await dock.send_message("12345", msg)
        assert result.success is False
        assert "not configured" in result.error

    @pytest.mark.asyncio
    async def test_send_message_success(self):
        """Test successful text message send."""
        dock = self._make_dock({"token": "123456:ABC"})
        await dock.initialize()
        msg = _make_message(content="Hello Telegram")

        mock_resp = _make_httpx_response(200, {"ok": True, "result": {"message_id": 42}})
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await dock.send_message("12345", msg)
            assert result.success is True
            assert result.message_id == "42"

    @pytest.mark.asyncio
    async def test_send_message_api_not_ok(self):
        """Test handling Telegram API not-ok response."""
        dock = self._make_dock({"token": "123456:ABC"})
        await dock.initialize()
        msg = _make_message()

        mock_resp = _make_httpx_response(200, {"ok": False, "description": "Chat not found"})
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await dock.send_message("12345", msg)
            assert result.success is False

    @pytest.mark.asyncio
    async def test_send_message_http_error(self):
        """Test handling HTTP error."""
        dock = self._make_dock({"token": "123456:ABC"})
        await dock.initialize()
        msg = _make_message()

        mock_resp = _make_httpx_response(500)
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await dock.send_message("12345", msg)
            assert result.success is False

    @pytest.mark.asyncio
    async def test_send_message_exception(self):
        """Test handling exception."""
        dock = self._make_dock({"token": "123456:ABC"})
        await dock.initialize()
        msg = _make_message()

        mock_client = AsyncMock()
        mock_client.post.side_effect = OSError("DNS failure")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await dock.send_message("12345", msg)
            assert result.success is False
            assert "DNS failure" in result.error

    def test_build_payload_markdown(self):
        """Test payload for markdown messages."""
        dock = self._make_dock({"token": "t"})
        dock._token = "t"
        msg = _make_message(content="**bold**", format=MessageFormat.MARKDOWN)
        payload = dock._build_payload("12345", msg)
        assert payload["chat_id"] == "12345"
        assert payload["parse_mode"] == "Markdown"
        assert "**bold**" in payload["text"]

    def test_build_payload_html(self):
        """Test payload for HTML messages."""
        dock = self._make_dock({"token": "t"})
        dock._token = "t"
        msg = _make_message(content="<b>bold</b>", format=MessageFormat.HTML)
        payload = dock._build_payload("12345", msg)
        assert payload["parse_mode"] == "HTML"

    def test_build_payload_with_title(self):
        """Test title is bolded in Telegram format."""
        dock = self._make_dock({"token": "t"})
        dock._token = "t"
        msg = _make_message(content="Body", title="My Title")
        payload = dock._build_payload("12345", msg)
        assert "*My Title*" in payload["text"]

    def test_build_payload_text_truncation(self):
        """Test text truncation to 4096 chars."""
        dock = self._make_dock({"token": "t"})
        dock._token = "t"
        long = "A" * 5000
        msg = _make_message(content=long)
        payload = dock._build_payload("12345", msg)
        assert len(payload["text"]) <= 4096

    def test_build_payload_reply(self):
        """Test reply_to_message_id in payload."""
        dock = self._make_dock({"token": "t"})
        dock._token = "t"
        msg = _make_message(reply_to="999")
        payload = dock._build_payload("12345", msg)
        assert payload["reply_to_message_id"] == "999"

    def test_build_payload_inline_keyboard(self):
        """Test inline keyboard from buttons."""
        dock = self._make_dock({"token": "t"})
        dock._token = "t"
        msg = _make_message(content="Choose")
        msg.with_button("URL Button", "https://example.com")
        msg.with_button("Action Button", "do_something")
        payload = dock._build_payload("12345", msg)
        keyboard = payload["reply_markup"]["inline_keyboard"]
        assert len(keyboard) == 2
        # URL button
        assert "url" in keyboard[0][0]
        # Action button
        assert "callback_data" in keyboard[1][0]

    @pytest.mark.asyncio
    async def test_send_voice_message(self):
        """Test voice message triggers voice send path."""
        dock = self._make_dock({"token": "123456:ABC"})
        await dock.initialize()

        msg = _make_message(content="Voice caption")
        msg.with_attachment("audio", data=b"voice-audio-data")

        mock_resp = _make_httpx_response(200, {"ok": True, "result": {"message_id": 55}})
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await dock.send_message("12345", msg)
            assert result.success is True

    def test_repr(self):
        """Test string representation."""
        dock = self._make_dock()
        assert "TelegramDock" in repr(dock)


# =============================================================================
# WhatsApp Dock Tests
# =============================================================================


class TestWhatsAppDock:
    """Tests for WhatsAppDock."""

    def _make_dock(self, config=None):
        from aragora.channels.docks.whatsapp import WhatsAppDock

        return WhatsAppDock(config)

    def test_platform(self):
        """Test platform identifier."""
        dock = self._make_dock()
        assert dock.PLATFORM == "whatsapp"

    def test_capabilities(self):
        """Test supported capabilities."""
        dock = self._make_dock()
        assert dock.supports(ChannelCapability.VOICE)
        assert dock.supports(ChannelCapability.FILES)
        assert not dock.supports(ChannelCapability.RICH_TEXT)
        assert not dock.supports(ChannelCapability.BUTTONS)

    @pytest.mark.asyncio
    async def test_initialize_with_config(self):
        """Test initialization with config credentials."""
        dock = self._make_dock({"access_token": "tok", "phone_number_id": "123"})
        result = await dock.initialize()
        assert result is True
        assert dock.is_initialized

    @pytest.mark.asyncio
    async def test_initialize_with_env(self):
        """Test initialization with env var credentials."""
        dock = self._make_dock()
        with patch.dict(
            "os.environ",
            {"WHATSAPP_ACCESS_TOKEN": "tok", "WHATSAPP_PHONE_NUMBER_ID": "123"},
        ):
            result = await dock.initialize()
            assert result is True

    @pytest.mark.asyncio
    async def test_initialize_no_credentials(self):
        """Test initialization fails without credentials."""
        dock = self._make_dock()
        with patch.dict("os.environ", {}, clear=True):
            result = await dock.initialize()
            assert result is False

    @pytest.mark.asyncio
    async def test_send_message_no_credentials(self):
        """Test send fails without credentials."""
        dock = self._make_dock()
        msg = _make_message()
        result = await dock.send_message("+1234567890", msg)
        assert result.success is False
        assert "not configured" in result.error

    @pytest.mark.asyncio
    async def test_send_message_success(self):
        """Test successful text message send."""
        dock = self._make_dock({"access_token": "tok", "phone_number_id": "num-1"})
        await dock.initialize()
        msg = _make_message(content="Hello WhatsApp")

        mock_resp = _make_httpx_response(200, {"messages": [{"id": "wamid.abc123"}]})
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await dock.send_message("+1234567890", msg)
            assert result.success is True
            assert result.message_id == "wamid.abc123"

    @pytest.mark.asyncio
    async def test_send_message_api_error(self):
        """Test handling API error."""
        dock = self._make_dock({"access_token": "tok", "phone_number_id": "num-1"})
        await dock.initialize()
        msg = _make_message()

        mock_resp = _make_httpx_response(400, {"error": {"message": "Invalid phone number"}})
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await dock.send_message("+invalid", msg)
            assert result.success is False

    @pytest.mark.asyncio
    async def test_send_message_exception(self):
        """Test handling network exception."""
        dock = self._make_dock({"access_token": "tok", "phone_number_id": "num-1"})
        await dock.initialize()
        msg = _make_message()

        mock_client = AsyncMock()
        mock_client.post.side_effect = OSError("Timeout")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await dock.send_message("+1234567890", msg)
            assert result.success is False

    def test_build_payload(self):
        """Test WhatsApp API payload structure."""
        dock = self._make_dock({"access_token": "tok", "phone_number_id": "num-1"})
        msg = _make_message(content="Hello")
        payload = dock._build_payload("+1234567890", msg)
        assert payload["messaging_product"] == "whatsapp"
        assert payload["to"] == "+1234567890"
        assert payload["type"] == "text"
        assert "Hello" in payload["text"]["body"]

    def test_build_payload_with_title(self):
        """Test payload includes bolded title."""
        dock = self._make_dock({"access_token": "tok", "phone_number_id": "num-1"})
        msg = _make_message(content="Body", title="Alert")
        payload = dock._build_payload("+1234567890", msg)
        assert "*Alert*" in payload["text"]["body"]

    def test_build_payload_text_truncation(self):
        """Test text truncation to 4096 chars."""
        dock = self._make_dock({"access_token": "tok", "phone_number_id": "num-1"})
        long = "A" * 5000
        msg = _make_message(content=long)
        payload = dock._build_payload("+1234567890", msg)
        assert len(payload["text"]["body"]) <= 4096

    @pytest.mark.asyncio
    async def test_send_voice_success(self):
        """Test successful voice message send."""
        dock = self._make_dock({"access_token": "tok", "phone_number_id": "num-1"})
        await dock.initialize()

        # Mock upload + send
        upload_resp = _make_httpx_response(200, {"id": "media-123"})
        send_resp = _make_httpx_response(200, {"messages": [{"id": "voice-msg-1"}]})

        mock_client = AsyncMock()
        mock_client.post.side_effect = [upload_resp, send_resp]
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await dock.send_voice("+1234567890", b"audio-data", text="Hello")
            assert result.success is True

    @pytest.mark.asyncio
    async def test_send_voice_upload_failure(self):
        """Test voice send fails on upload error."""
        dock = self._make_dock({"access_token": "tok", "phone_number_id": "num-1"})
        await dock.initialize()

        upload_resp = _make_httpx_response(500)
        mock_client = AsyncMock()
        mock_client.post.return_value = upload_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await dock.send_voice("+1234567890", b"audio-data")
            assert result.success is False

    @pytest.mark.asyncio
    async def test_send_voice_no_audio_data(self):
        """Test voice send fails without audio data."""
        dock = self._make_dock({"access_token": "tok", "phone_number_id": "num-1"})
        await dock.initialize()
        msg = _make_message(content="No audio")
        msg.attachments.append(MessageAttachment(type="audio", data=None))
        result = await dock._send_voice_message("+1234567890", msg.attachments[0], msg)
        assert result.success is False
        assert "No audio data" in result.error

    def test_repr(self):
        """Test string representation."""
        dock = self._make_dock()
        assert "WhatsAppDock" in repr(dock)
