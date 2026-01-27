"""
Channel-specific formatters and docks for delivering decision receipts and messages.

This module provides:
1. Receipt formatters (legacy) - format receipts for specific platforms
2. Channel docks (new) - unified message delivery abstraction

Supports platforms:
- Slack (Block Kit)
- Teams (Adaptive Cards)
- Discord (Embeds)
- Telegram
- WhatsApp
- Email (HTML)
- Google Chat

Receipt Formatters (Legacy):
    from aragora.channels import format_receipt_for_channel
    formatted = format_receipt_for_channel("slack", receipt)

Channel Docks (New):
    from aragora.channels import get_dock_registry, NormalizedMessage
    registry = get_dock_registry()
    dock = registry.get_dock("slack")
    await dock.send_message(channel_id, message)
"""

# Legacy receipt formatters
from .formatter import ReceiptFormatter, format_receipt_for_channel
from .slack_formatter import SlackReceiptFormatter
from .teams_formatter import TeamsReceiptFormatter
from .discord_formatter import DiscordReceiptFormatter
from .email_formatter import EmailReceiptFormatter

# New channel dock system
from .dock import (
    ChannelDock,
    ChannelCapability,
    MessageType,
    SendResult,
)
from .normalized import NormalizedMessage, MessageFormat
from .registry import DockRegistry, get_dock_registry

__all__ = [
    # Legacy formatters
    "ReceiptFormatter",
    "format_receipt_for_channel",
    "SlackReceiptFormatter",
    "TeamsReceiptFormatter",
    "DiscordReceiptFormatter",
    "EmailReceiptFormatter",
    # New dock system
    "ChannelDock",
    "ChannelCapability",
    "MessageType",
    "SendResult",
    "NormalizedMessage",
    "MessageFormat",
    "DockRegistry",
    "get_dock_registry",
]
