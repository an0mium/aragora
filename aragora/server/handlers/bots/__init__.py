"""
Bot webhook handlers for chat platform integrations.

Provides HTTP endpoints for receiving webhooks from Discord, Teams, Telegram,
WhatsApp, and Zoom - enabling bidirectional human-AI communication.

Aragora is omnivorous by design: query from any channel, get multi-agent consensus.
"""

from aragora.server.handlers.bots.discord import DiscordHandler
from aragora.server.handlers.bots.email_webhook import EmailWebhookHandler
from aragora.server.handlers.bots.teams import TeamsHandler
from aragora.server.handlers.bots.telegram import TelegramHandler
from aragora.server.handlers.bots.whatsapp import WhatsAppHandler
from aragora.server.handlers.bots.zoom import ZoomHandler

__all__ = [
    "DiscordHandler",
    "EmailWebhookHandler",
    "TeamsHandler",
    "TelegramHandler",
    "WhatsAppHandler",
    "ZoomHandler",
]
