"""
Telegram Bot Connector.

Implements ChatPlatformConnector for Telegram using the Bot API.
Includes circuit breaker protection for fault tolerance.

Environment Variables:
- TELEGRAM_BOT_TOKEN: Bot API token from @BotFather
- TELEGRAM_WEBHOOK_URL: Webhook URL for receiving updates

This module is split into focused submodules:
- client.py: Core API request handling and base class
- messages.py: Basic messaging (send, update, delete)
- files.py: File upload/download and voice messages
- webhooks.py: Webhook handling, parsing, verification
- media.py: Rich media (photos, videos, animations)
- inline.py: Inline query support
- bot_management.py: Bot commands and info
"""

from __future__ import annotations

try:
    import httpx  # noqa: F401

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from aragora.connectors.chat.telegram.bot_management import TelegramBotManagementMixin
from aragora.connectors.chat.telegram.client import TelegramConnectorBase
from aragora.connectors.chat.telegram.client import _classify_telegram_error
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
    """
    Full Telegram connector combining all functionality.

    Supports:
    - Sending messages with Markdown/HTML formatting
    - Inline keyboards (buttons)
    - File uploads (documents, photos, voice)
    - Reply messages (threads)
    - Callback queries (button interactions)
    - Webhook and long-polling
    - Rich media (photos, videos, animations, media groups)
    - Inline queries
    - Bot management (commands, info)

    All HTTP operations include circuit breaker protection for fault tolerance.
    """

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
    "_classify_telegram_error",
]
