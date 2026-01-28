"""
Bot integrations for Aragora control plane.

Aragora delivers deliberation results wherever your team works:
Discord, Teams, Telegram, WhatsApp, Zoom, and more.

Query from any channel, get defensible decisions with multi-agent consensus.
"""

from aragora.server.handlers.bots.base import BotHandlerMixin
from aragora.server.handlers.bots.discord import DiscordHandler
from aragora.server.handlers.bots.email_webhook import EmailWebhookHandler
from aragora.server.handlers.bots.google_chat import GoogleChatHandler
from aragora.server.handlers.bots.teams import TeamsHandler
from aragora.server.handlers.bots.telegram import TelegramHandler
from aragora.server.handlers.bots.whatsapp import WhatsAppHandler
from aragora.server.handlers.bots.zoom import ZoomHandler

__all__ = [
    "BotHandlerMixin",
    "DiscordHandler",
    "EmailWebhookHandler",
    "GoogleChatHandler",
    "TeamsHandler",
    "TelegramHandler",
    "WhatsAppHandler",
    "ZoomHandler",
]
