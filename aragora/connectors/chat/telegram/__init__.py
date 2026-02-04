"""Telegram chat connector package."""

from __future__ import annotations

from aragora.connectors.chat.telegram.client import TelegramConnectorBase
from aragora.connectors.chat.telegram.files import TelegramFilesMixin
from aragora.connectors.chat.telegram.messages import TelegramMessagesMixin
from aragora.connectors.chat.telegram.webhooks import TelegramWebhooksMixin


class TelegramConnector(
    TelegramConnectorBase,
    TelegramMessagesMixin,
    TelegramFilesMixin,
    TelegramWebhooksMixin,
):
    """Telegram connector with messaging, files, and webhook support."""


__all__ = [
    "TelegramConnector",
    "TelegramConnectorBase",
    "TelegramMessagesMixin",
    "TelegramFilesMixin",
    "TelegramWebhooksMixin",
]
