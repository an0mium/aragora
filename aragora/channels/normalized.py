"""
Normalized Message - Cross-platform message representation.

Provides a unified message format that can be adapted to any platform's
specific requirements by the channel docks.

Example:
    from aragora.channels.normalized import NormalizedMessage, MessageFormat

    message = NormalizedMessage(
        content="Hello, world!",
        format=MessageFormat.MARKDOWN,
        buttons=[{"label": "Click me", "action": "click"}],
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

__all__ = [
    "NormalizedMessage",
    "MessageFormat",
    "MessageButton",
    "MessageAttachment",
]


class MessageFormat(Enum):
    """Message content format."""

    PLAIN = "plain"  # Plain text
    MARKDOWN = "markdown"  # Markdown formatted
    HTML = "html"  # HTML formatted
    ADAPTIVE = "adaptive"  # Adaptive Card (Teams)


@dataclass
class MessageButton:
    """An interactive button that can be included in a message."""

    label: str
    action: str  # URL or action identifier
    style: str = "default"  # default, primary, danger
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "label": self.label,
            "action": self.action,
            "style": self.style,
            "metadata": self.metadata,
        }


@dataclass
class MessageAttachment:
    """An attachment (file, image, audio) that can be included in a message."""

    type: str  # file, image, audio, video
    data: Optional[bytes] = None  # Raw data
    url: Optional[str] = None  # URL to resource
    filename: Optional[str] = None
    mimetype: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excluding binary data)."""
        return {
            "type": self.type,
            "url": self.url,
            "filename": self.filename,
            "mimetype": self.mimetype,
            "metadata": self.metadata,
        }


@dataclass
class NormalizedMessage:
    """
    A platform-independent message representation.

    This class provides a unified format for messages that can be
    adapted by each platform dock to its specific requirements.

    Attributes:
        content: The main message text
        message_type: Type of message (result, receipt, error, etc.)
        format: Text format (plain, markdown, html)
        title: Optional title/header
        buttons: Interactive buttons (if supported)
        attachments: File/media attachments
        thread_id: Thread to reply in (if supported)
        reply_to: Message ID to reply to
        metadata: Additional platform-specific data
    """

    content: str
    message_type: str = "notification"
    format: MessageFormat = MessageFormat.PLAIN
    title: Optional[str] = None
    buttons: list[MessageButton] = field(default_factory=list)
    attachments: list[MessageAttachment | dict[str, Any]] = field(default_factory=list)
    thread_id: Optional[str] = None
    reply_to: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_button(
        self,
        label: str,
        action: str,
        style: str = "default",
    ) -> "NormalizedMessage":
        """
        Add a button to the message (fluent interface).

        Args:
            label: Button label text
            action: URL or action identifier
            style: Button style (default, primary, danger)

        Returns:
            self for chaining
        """
        self.buttons.append(MessageButton(label=label, action=action, style=style))
        return self

    def with_attachment(
        self,
        type: str,
        data: Optional[bytes] = None,
        url: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> "NormalizedMessage":
        """
        Add an attachment to the message (fluent interface).

        Args:
            type: Attachment type (file, image, audio)
            data: Raw attachment data
            url: URL to the attachment
            filename: Filename for the attachment

        Returns:
            self for chaining
        """
        self.attachments.append(
            MessageAttachment(
                type=type,
                data=data,
                url=url,
                filename=filename,
            )
        )
        return self

    def to_plain_text(self) -> str:
        """
        Convert message to plain text format.

        Strips markdown/HTML formatting.

        Returns:
            Plain text content
        """
        # Simple markdown stripping
        text = self.content
        # Remove bold/italic markers
        for marker in ["**", "*", "__", "_", "`", "```"]:
            text = text.replace(marker, "")
        return text

    def to_markdown(self) -> str:
        """
        Get message as markdown format.

        Returns:
            Markdown formatted content
        """
        if self.format == MessageFormat.MARKDOWN:
            return self.content
        elif self.format == MessageFormat.PLAIN:
            # Escape special markdown characters
            return self.content
        else:
            # HTML to markdown would need conversion
            return self.to_plain_text()

    def has_buttons(self) -> bool:
        """Check if message has interactive buttons."""
        return len(self.buttons) > 0

    def has_attachments(self) -> bool:
        """Check if message has attachments."""
        return len(self.attachments) > 0

    def get_audio_attachment(self) -> Optional[MessageAttachment | dict[str, Any]]:
        """Get the first audio attachment if present."""
        for att in self.attachments:
            att_type = att.type if isinstance(att, MessageAttachment) else att.get("type")
            if att_type == "audio":
                return att
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "content": self.content,
            "message_type": self.message_type,
            "format": self.format.value,
            "title": self.title,
            "buttons": [b.to_dict() if isinstance(b, MessageButton) else b for b in self.buttons],
            "attachments": [
                a.to_dict() if isinstance(a, MessageAttachment) else a for a in self.attachments
            ],
            "thread_id": self.thread_id,
            "reply_to": self.reply_to,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NormalizedMessage":
        """Create from dictionary representation."""
        buttons = [
            MessageButton(**b) if isinstance(b, dict) else b for b in data.get("buttons", [])
        ]
        attachments = [
            MessageAttachment(**a) if isinstance(a, dict) and "type" in a else a
            for a in data.get("attachments", [])
        ]

        format_val = data.get("format", "plain")
        if isinstance(format_val, str):
            format_val = MessageFormat(format_val)

        return cls(
            content=data.get("content", ""),
            message_type=data.get("message_type", "notification"),
            format=format_val,
            title=data.get("title"),
            buttons=buttons,
            attachments=attachments,
            thread_id=data.get("thread_id"),
            reply_to=data.get("reply_to"),
            metadata=data.get("metadata", {}),
        )
