"""
Telegram Bot Connector - Backwards Compatibility Module.

This file is maintained for backwards compatibility.
The implementation has been refactored into the telegram/ subpackage.

Import from aragora.connectors.chat.telegram instead.
"""

from __future__ import annotations

# Re-export from the refactored module for backwards compatibility
from aragora.connectors.chat.telegram import (
    TelegramBotManagementMixin,
    TelegramConnector,
    TelegramConnectorBase,
    TelegramFilesMixin,
    TelegramInlineMixin,
    TelegramMediaMixin,
    TelegramMessagesMixin,
    TelegramWebhooksMixin,
)

__all__ = [
    "TelegramConnector",
    "TelegramConnectorBase",
    "TelegramMessagesMixin",
    "TelegramFilesMixin",
    "TelegramWebhooksMixin",
    "TelegramMediaMixin",
    "TelegramInlineMixin",
    "TelegramBotManagementMixin",
]
