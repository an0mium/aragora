"""Platform-specific sender implementations for debate result routing.

Each submodule provides send functions for a specific chat platform
(result, receipt, error, and voice messages).
"""

from .slack import _send_slack_result, _send_slack_receipt, _send_slack_error
from .teams import _send_teams_result, _send_teams_receipt, _send_teams_error
from .telegram import (
    _send_telegram_result,
    _send_telegram_receipt,
    _send_telegram_error,
    _send_telegram_voice,
)
from .discord import (
    _send_discord_result,
    _send_discord_receipt,
    _send_discord_error,
    _send_discord_voice,
)
from .whatsapp import _send_whatsapp_result, _send_whatsapp_voice
from .email import _send_email_result
from .google_chat import _send_google_chat_result, _send_google_chat_receipt

__all__ = [
    "_send_slack_result",
    "_send_slack_receipt",
    "_send_slack_error",
    "_send_teams_result",
    "_send_teams_receipt",
    "_send_teams_error",
    "_send_telegram_result",
    "_send_telegram_receipt",
    "_send_telegram_error",
    "_send_telegram_voice",
    "_send_discord_result",
    "_send_discord_receipt",
    "_send_discord_error",
    "_send_discord_voice",
    "_send_whatsapp_result",
    "_send_whatsapp_voice",
    "_send_email_result",
    "_send_google_chat_result",
    "_send_google_chat_receipt",
]
