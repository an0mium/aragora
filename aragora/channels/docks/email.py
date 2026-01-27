"""
Email Dock - Channel dock implementation for Email.

Handles message delivery via email using the notification system.

Example:
    from aragora.channels.docks.email import EmailDock

    dock = EmailDock()
    await dock.initialize()
    await dock.send_message(email_address, message)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from aragora.channels.dock import ChannelDock, ChannelCapability, SendResult

if TYPE_CHECKING:
    from aragora.channels.normalized import NormalizedMessage

logger = logging.getLogger(__name__)

__all__ = ["EmailDock"]


class EmailDock(ChannelDock):
    """
    Email platform dock.

    Supports HTML formatting and file attachments via email.
    Uses the existing email notification system.
    """

    PLATFORM = "email"
    CAPABILITIES = ChannelCapability.RICH_TEXT | ChannelCapability.FILES

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """
        Initialize Email dock.

        Config options:
            sender: Default sender email address
            smtp_host: SMTP server host
            smtp_port: SMTP server port
        """
        super().__init__(config)
        self._notification_system_available = False

    async def initialize(self) -> bool:
        """Initialize the Email dock."""
        from importlib.util import find_spec

        if find_spec("aragora.server.handlers.social.notifications") is None:
            logger.warning("Email notification system not available")
            self._initialized = True  # Still initialized, but will return errors
            return True

        self._notification_system_available = True
        self._initialized = True
        return True

    async def send_message(
        self,
        channel_id: str,
        message: "NormalizedMessage",
        **kwargs: Any,
    ) -> SendResult:
        """
        Send a message via email.

        Args:
            channel_id: Recipient email address
            message: The normalized message to send
            **kwargs: Additional options (subject, from_address, etc.)

        Returns:
            SendResult indicating success or failure
        """
        if not self._notification_system_available:
            return SendResult.fail(
                error="Email notification system not available",
                platform=self.PLATFORM,
                channel_id=channel_id,
            )

        try:
            from aragora.server.handlers.social.notifications import send_email_notification  # type: ignore[attr-defined]

            email = kwargs.get("email") or channel_id
            subject = kwargs.get("subject") or message.title or "Aragora Notification"

            # Build HTML body
            body = self._build_email_body(message)

            await send_email_notification(
                to_email=email,
                subject=subject,
                body=body,
            )

            logger.info(f"Email sent to {email}")
            return SendResult.ok(
                platform=self.PLATFORM,
                channel_id=channel_id,
            )

        except Exception as e:
            logger.error(f"Email send error: {e}")
            return SendResult.fail(
                error=str(e),
                platform=self.PLATFORM,
                channel_id=channel_id,
            )

    def _build_email_body(self, message: "NormalizedMessage") -> str:
        """Build HTML email body from normalized message."""
        from aragora.channels.normalized import MessageFormat

        parts = []

        # Add title
        if message.title:
            parts.append(f"<h2>{self._escape_html(message.title)}</h2>")

        # Add content
        if message.content:
            if message.format == MessageFormat.HTML:
                # Already HTML
                parts.append(f"<div>{message.content}</div>")
            elif message.format == MessageFormat.MARKDOWN:
                # Convert markdown to HTML (basic conversion)
                html_content = self._markdown_to_html(message.content)
                parts.append(f"<div>{html_content}</div>")
            else:
                # Plain text - wrap in pre to preserve formatting
                parts.append(f"<pre>{self._escape_html(message.content)}</pre>")

        # Add buttons as links
        if message.has_buttons():
            parts.append("<div style='margin-top: 20px;'>")
            for button in message.buttons:
                if isinstance(button, dict):
                    label = button.get("label", "Click")
                    action = button.get("action", "#")
                else:
                    label = getattr(button, "label", "Click")
                    action = getattr(button, "action", "#")

                if action.startswith("http"):
                    parts.append(
                        f'<a href="{action}" style="display: inline-block; padding: 10px 20px; '
                        f"background-color: #007bff; color: white; text-decoration: none; "
                        f'border-radius: 4px; margin-right: 10px;">{self._escape_html(label)}</a>'
                    )
            parts.append("</div>")

        return "\n".join(parts)

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    def _markdown_to_html(self, text: str) -> str:
        """Basic markdown to HTML conversion."""
        import re

        html = self._escape_html(text)

        # Bold: **text** or __text__
        html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
        html = re.sub(r"__(.+?)__", r"<strong>\1</strong>", html)

        # Italic: *text* or _text_
        html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)
        html = re.sub(r"_(.+?)_", r"<em>\1</em>", html)

        # Code: `text`
        html = re.sub(r"`(.+?)`", r"<code>\1</code>", html)

        # Links: [text](url)
        html = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', html)

        # Line breaks
        html = html.replace("\n\n", "</p><p>")
        html = html.replace("\n", "<br>")
        html = f"<p>{html}</p>"

        return html

    async def send_result(
        self,
        channel_id: str,
        result: dict[str, Any],
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SendResult:
        """Send a debate result via email with rich formatting."""
        from aragora.channels.normalized import NormalizedMessage, MessageFormat
        from aragora.channels.dock import MessageType

        consensus = result.get("consensus_reached", False)
        confidence = result.get("confidence", 0)
        answer = result.get("final_answer", "No conclusion reached.")
        task = result.get("task", "Unknown topic")

        # Build HTML content
        status = "Consensus Reached" if consensus else "No Consensus"
        confidence_pct = f"{confidence:.0%}"

        content = f"""
        <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>Topic:</strong></td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">{self._escape_html(task[:500])}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>Status:</strong></td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">{status}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>Confidence:</strong></td>
                <td style="padding: 10px; border-bottom: 1px solid #ddd;">{confidence_pct}</td>
            </tr>
        </table>
        <h3>Conclusion</h3>
        <p>{self._escape_html(answer[:3000])}</p>
        """

        message = NormalizedMessage(
            content=content,
            message_type=MessageType.RESULT,
            format=MessageFormat.HTML,
            title="Aragora Debate Complete",
        )

        # Add view details button
        receipt_url = result.get("receipt_url") or kwargs.get("receipt_url")
        if receipt_url:
            message.with_button("View Full Details", receipt_url)

        return await self.send_message(
            channel_id,
            message,
            subject="Aragora Debate Complete",
            **kwargs,
        )
