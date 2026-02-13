"""
Telegram Bot Connector - Backwards Compatibility Module.

This file is maintained for backwards compatibility.
The implementation has been refactored into the telegram/ subpackage.

Import from aragora.connectors.chat.telegram instead.
"""

from __future__ import annotations

# Re-export from the refactored package for backwards compatibility.
from aragora.connectors.chat.telegram.bot_management import TelegramBotManagementMixin
from aragora.connectors.chat.telegram.client import TelegramConnectorBase
from aragora.connectors.chat.telegram.files import TelegramFilesMixin
from aragora.connectors.chat.telegram.inline import TelegramInlineMixin
from aragora.connectors.chat.telegram.media import TelegramMediaMixin
from aragora.connectors.chat.telegram.messages import TelegramMessagesMixin
from aragora.connectors.chat.telegram.webhooks import TelegramWebhooksMixin


class TelegramConnector(
    TelegramMessagesMixin,
    TelegramFilesMixin,
    TelegramWebhooksMixin,
    TelegramMediaMixin,
    TelegramInlineMixin,
    TelegramBotManagementMixin,
    TelegramConnectorBase,
):
    """Compatibility wrapper exposing the full Telegram connector API."""

    pass

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
