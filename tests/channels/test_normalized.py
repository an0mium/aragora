"""
Tests for NormalizedMessage and related classes.

Tests the cross-platform message representation used by all channel docks.
"""

import pytest

from aragora.channels.normalized import (
    NormalizedMessage,
    MessageFormat,
    MessageButton,
    MessageAttachment,
)


# =============================================================================
# MessageFormat Tests
# =============================================================================


class TestMessageFormat:
    """Tests for MessageFormat enum."""

    def test_plain_format(self):
        """Test plain text format value."""
        assert MessageFormat.PLAIN.value == "plain"

    def test_markdown_format(self):
        """Test markdown format value."""
        assert MessageFormat.MARKDOWN.value == "markdown"

    def test_html_format(self):
        """Test HTML format value."""
        assert MessageFormat.HTML.value == "html"

    def test_adaptive_format(self):
        """Test adaptive card format value."""
        assert MessageFormat.ADAPTIVE.value == "adaptive"

    def test_format_from_value(self):
        """Test creating format from string value."""
        assert MessageFormat("plain") == MessageFormat.PLAIN
        assert MessageFormat("markdown") == MessageFormat.MARKDOWN
        assert MessageFormat("html") == MessageFormat.HTML


# =============================================================================
# MessageButton Tests
# =============================================================================


class TestMessageButton:
    """Tests for MessageButton dataclass."""

    def test_basic_button(self):
        """Test creating a basic button."""
        btn = MessageButton(label="Click", action="https://example.com")
        assert btn.label == "Click"
        assert btn.action == "https://example.com"
        assert btn.style == "default"
        assert btn.metadata == {}

    def test_button_with_style(self):
        """Test creating a button with a custom style."""
        btn = MessageButton(label="Delete", action="delete_item", style="danger")
        assert btn.style == "danger"

    def test_button_with_metadata(self):
        """Test creating a button with metadata."""
        btn = MessageButton(label="Vote", action="vote_yes", metadata={"debate_id": "123"})
        assert btn.metadata == {"debate_id": "123"}

    def test_button_to_dict(self):
        """Test button serialization to dict."""
        btn = MessageButton(
            label="OK",
            action="confirm",
            style="primary",
            metadata={"key": "val"},
        )
        d = btn.to_dict()
        assert d == {
            "label": "OK",
            "action": "confirm",
            "style": "primary",
            "metadata": {"key": "val"},
        }

    def test_button_to_dict_defaults(self):
        """Test button serialization with default values."""
        btn = MessageButton(label="Go", action="go")
        d = btn.to_dict()
        assert d["style"] == "default"
        assert d["metadata"] == {}


# =============================================================================
# MessageAttachment Tests
# =============================================================================


class TestMessageAttachment:
    """Tests for MessageAttachment dataclass."""

    def test_basic_attachment(self):
        """Test creating a basic attachment."""
        att = MessageAttachment(type="file")
        assert att.type == "file"
        assert att.data is None
        assert att.url is None
        assert att.filename is None
        assert att.mimetype is None
        assert att.metadata == {}

    def test_attachment_with_url(self):
        """Test attachment with URL."""
        att = MessageAttachment(
            type="image",
            url="https://example.com/img.png",
            filename="img.png",
            mimetype="image/png",
        )
        assert att.url == "https://example.com/img.png"
        assert att.filename == "img.png"
        assert att.mimetype == "image/png"

    def test_attachment_with_data(self):
        """Test attachment with binary data."""
        data = b"audio-bytes"
        att = MessageAttachment(type="audio", data=data)
        assert att.data == data

    def test_attachment_to_dict(self):
        """Test attachment serialization excludes binary data."""
        att = MessageAttachment(
            type="file",
            data=b"binary-data",
            url="https://example.com/file.txt",
            filename="file.txt",
            mimetype="text/plain",
        )
        d = att.to_dict()
        assert d == {
            "type": "file",
            "url": "https://example.com/file.txt",
            "filename": "file.txt",
            "mimetype": "text/plain",
            "metadata": {},
        }
        # Binary data should NOT be in the dict
        assert "data" not in d

    def test_attachment_to_dict_with_metadata(self):
        """Test attachment serialization with metadata."""
        att = MessageAttachment(type="video", metadata={"duration": 30, "resolution": "1080p"})
        d = att.to_dict()
        assert d["metadata"] == {"duration": 30, "resolution": "1080p"}


# =============================================================================
# NormalizedMessage Tests
# =============================================================================


class TestNormalizedMessage:
    """Tests for NormalizedMessage dataclass."""

    def test_basic_message(self):
        """Test creating a basic message."""
        msg = NormalizedMessage(content="Hello")
        assert msg.content == "Hello"
        assert msg.message_type == "notification"
        assert msg.format == MessageFormat.PLAIN
        assert msg.title is None
        assert msg.buttons == []
        assert msg.attachments == []
        assert msg.thread_id is None
        assert msg.reply_to is None
        assert msg.metadata == {}

    def test_full_message(self):
        """Test creating a fully specified message."""
        msg = NormalizedMessage(
            content="Result text",
            message_type="result",
            format=MessageFormat.MARKDOWN,
            title="Debate Result",
            thread_id="thread-1",
            reply_to="msg-42",
            metadata={"debate_id": "d-99"},
        )
        assert msg.content == "Result text"
        assert msg.message_type == "result"
        assert msg.format == MessageFormat.MARKDOWN
        assert msg.title == "Debate Result"
        assert msg.thread_id == "thread-1"
        assert msg.reply_to == "msg-42"
        assert msg.metadata == {"debate_id": "d-99"}

    def test_with_button_fluent(self):
        """Test fluent button API."""
        msg = NormalizedMessage(content="Choose")
        result = msg.with_button("Yes", "yes_action", style="primary")
        # Returns self for chaining
        assert result is msg
        assert len(msg.buttons) == 1
        assert isinstance(msg.buttons[0], MessageButton)
        assert msg.buttons[0].label == "Yes"
        assert msg.buttons[0].action == "yes_action"
        assert msg.buttons[0].style == "primary"

    def test_with_button_chaining(self):
        """Test chaining multiple button calls."""
        msg = (
            NormalizedMessage(content="Vote")
            .with_button("Agree", "agree")
            .with_button("Disagree", "disagree")
        )
        assert len(msg.buttons) == 2
        assert msg.buttons[0].label == "Agree"
        assert msg.buttons[1].label == "Disagree"

    def test_with_attachment_fluent(self):
        """Test fluent attachment API."""
        msg = NormalizedMessage(content="File attached")
        result = msg.with_attachment("file", url="https://example.com/f.pdf", filename="report.pdf")
        assert result is msg
        assert len(msg.attachments) == 1
        att = msg.attachments[0]
        assert isinstance(att, MessageAttachment)
        assert att.type == "file"
        assert att.url == "https://example.com/f.pdf"
        assert att.filename == "report.pdf"

    def test_with_attachment_data(self):
        """Test attachment with binary data via fluent API."""
        data = b"audio-content"
        msg = NormalizedMessage(content="Audio")
        msg.with_attachment("audio", data=data)
        assert msg.attachments[0].data == data

    def test_to_plain_text_from_plain(self):
        """Test plain text conversion from plain content."""
        msg = NormalizedMessage(content="Hello world")
        assert msg.to_plain_text() == "Hello world"

    def test_to_plain_text_strips_markdown(self):
        """Test plain text conversion strips markdown markers."""
        msg = NormalizedMessage(content="**Bold** and *italic* and `code`")
        text = msg.to_plain_text()
        assert "**" not in text
        assert "*" not in text
        assert "`" not in text
        assert "Bold" in text
        assert "italic" in text
        assert "code" in text

    def test_to_plain_text_strips_code_blocks(self):
        """Test plain text conversion strips triple backticks."""
        msg = NormalizedMessage(content="```python\nprint('hi')\n```")
        text = msg.to_plain_text()
        assert "```" not in text
        assert "print('hi')" in text

    def test_to_markdown_from_markdown(self):
        """Test markdown returns content as-is when already markdown."""
        msg = NormalizedMessage(content="**Bold** text", format=MessageFormat.MARKDOWN)
        assert msg.to_markdown() == "**Bold** text"

    def test_to_markdown_from_plain(self):
        """Test markdown from plain text returns content unchanged."""
        msg = NormalizedMessage(content="Plain text", format=MessageFormat.PLAIN)
        assert msg.to_markdown() == "Plain text"

    def test_to_markdown_from_html(self):
        """Test markdown from HTML strips to plain text."""
        msg = NormalizedMessage(content="<b>Bold</b> text", format=MessageFormat.HTML)
        result = msg.to_markdown()
        # HTML format goes through to_plain_text
        assert "Bold" in result

    def test_has_buttons_false(self):
        """Test has_buttons when no buttons."""
        msg = NormalizedMessage(content="No buttons")
        assert msg.has_buttons() is False

    def test_has_buttons_true(self):
        """Test has_buttons when buttons present."""
        msg = NormalizedMessage(content="With buttons")
        msg.with_button("Click", "click")
        assert msg.has_buttons() is True

    def test_has_attachments_false(self):
        """Test has_attachments when no attachments."""
        msg = NormalizedMessage(content="No attachments")
        assert msg.has_attachments() is False

    def test_has_attachments_true(self):
        """Test has_attachments when attachments present."""
        msg = NormalizedMessage(content="With attachment")
        msg.with_attachment("file", url="https://example.com/f")
        assert msg.has_attachments() is True

    def test_get_audio_attachment_none(self):
        """Test get_audio_attachment when no audio."""
        msg = NormalizedMessage(content="No audio")
        msg.with_attachment("image", url="https://example.com/img.png")
        assert msg.get_audio_attachment() is None

    def test_get_audio_attachment_from_dataclass(self):
        """Test get_audio_attachment with MessageAttachment."""
        msg = NormalizedMessage(content="Audio")
        msg.with_attachment("audio", data=b"audio-bytes")
        audio = msg.get_audio_attachment()
        assert audio is not None
        assert isinstance(audio, MessageAttachment)
        assert audio.type == "audio"

    def test_get_audio_attachment_from_dict(self):
        """Test get_audio_attachment with dict attachment."""
        msg = NormalizedMessage(content="Audio")
        msg.attachments.append({"type": "audio", "data": b"bytes"})
        audio = msg.get_audio_attachment()
        assert audio is not None
        assert audio["type"] == "audio"

    def test_to_dict(self):
        """Test full dictionary serialization."""
        msg = NormalizedMessage(
            content="Test",
            message_type="result",
            format=MessageFormat.MARKDOWN,
            title="Title",
            thread_id="t1",
            reply_to="r1",
            metadata={"key": "val"},
        )
        msg.with_button("Click", "https://example.com")
        msg.with_attachment("image", url="https://example.com/img.png")

        d = msg.to_dict()
        assert d["content"] == "Test"
        assert d["message_type"] == "result"
        assert d["format"] == "markdown"
        assert d["title"] == "Title"
        assert d["thread_id"] == "t1"
        assert d["reply_to"] == "r1"
        assert d["metadata"] == {"key": "val"}
        assert len(d["buttons"]) == 1
        assert d["buttons"][0]["label"] == "Click"
        assert len(d["attachments"]) == 1
        assert d["attachments"][0]["type"] == "image"

    def test_to_dict_with_dict_attachments(self):
        """Test serialization handles dict attachments."""
        msg = NormalizedMessage(content="Test")
        msg.attachments.append({"type": "file", "url": "https://example.com/f"})
        d = msg.to_dict()
        assert d["attachments"][0] == {"type": "file", "url": "https://example.com/f"}

    def test_from_dict_basic(self):
        """Test deserialization from dict."""
        data = {
            "content": "Hello",
            "message_type": "notification",
            "format": "plain",
        }
        msg = NormalizedMessage.from_dict(data)
        assert msg.content == "Hello"
        assert msg.message_type == "notification"
        assert msg.format == MessageFormat.PLAIN

    def test_from_dict_with_buttons(self):
        """Test deserialization with buttons."""
        data = {
            "content": "Choose",
            "buttons": [
                {"label": "Yes", "action": "yes"},
                {"label": "No", "action": "no"},
            ],
        }
        msg = NormalizedMessage.from_dict(data)
        assert len(msg.buttons) == 2
        assert isinstance(msg.buttons[0], MessageButton)
        assert msg.buttons[0].label == "Yes"

    def test_from_dict_with_attachments(self):
        """Test deserialization with attachments."""
        data = {
            "content": "File",
            "attachments": [
                {"type": "image", "url": "https://example.com/img.png"},
            ],
        }
        msg = NormalizedMessage.from_dict(data)
        assert len(msg.attachments) == 1
        assert isinstance(msg.attachments[0], MessageAttachment)
        assert msg.attachments[0].type == "image"

    def test_from_dict_defaults(self):
        """Test deserialization provides defaults for missing fields."""
        data = {}
        msg = NormalizedMessage.from_dict(data)
        assert msg.content == ""
        assert msg.message_type == "notification"
        assert msg.format == MessageFormat.PLAIN
        assert msg.title is None
        assert msg.buttons == []
        assert msg.attachments == []

    def test_from_dict_format_as_enum(self):
        """Test deserialization with format already as enum."""
        data = {
            "content": "Test",
            "format": MessageFormat.MARKDOWN,
        }
        msg = NormalizedMessage.from_dict(data)
        assert msg.format == MessageFormat.MARKDOWN

    def test_roundtrip_serialization(self):
        """Test to_dict -> from_dict roundtrip preserves data."""
        original = NormalizedMessage(
            content="Roundtrip test",
            message_type="receipt",
            format=MessageFormat.MARKDOWN,
            title="Receipt",
            thread_id="t42",
            reply_to="r99",
            metadata={"key": "value"},
        )
        original.with_button("View", "https://example.com/view", style="primary")
        original.with_attachment("file", url="https://example.com/f.pdf", filename="report.pdf")

        d = original.to_dict()
        restored = NormalizedMessage.from_dict(d)

        assert restored.content == original.content
        assert restored.message_type == original.message_type
        assert restored.format == original.format
        assert restored.title == original.title
        assert restored.thread_id == original.thread_id
        assert restored.reply_to == original.reply_to
        assert restored.metadata == original.metadata
        assert len(restored.buttons) == 1
        assert restored.buttons[0].label == "View"
        assert len(restored.attachments) == 1
        assert restored.attachments[0].type == "file"
