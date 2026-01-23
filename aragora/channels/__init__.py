"""
Channel-specific formatters for delivering decision receipts.

Supports formatting receipts for:
- Slack (Block Kit)
- Teams (Adaptive Cards)
- Discord (Embeds)
- Email (HTML)
"""

from .formatter import ReceiptFormatter, format_receipt_for_channel
from .slack_formatter import SlackReceiptFormatter
from .teams_formatter import TeamsReceiptFormatter
from .discord_formatter import DiscordReceiptFormatter
from .email_formatter import EmailReceiptFormatter

__all__ = [
    "ReceiptFormatter",
    "format_receipt_for_channel",
    "SlackReceiptFormatter",
    "TeamsReceiptFormatter",
    "DiscordReceiptFormatter",
    "EmailReceiptFormatter",
]
