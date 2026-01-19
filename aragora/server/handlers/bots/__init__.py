"""
Bot webhook handlers for chat platform integrations.

Provides HTTP endpoints for receiving webhooks from Discord, Teams, and Zoom.
"""

from aragora.server.handlers.bots.discord import DiscordHandler
from aragora.server.handlers.bots.teams import TeamsHandler
from aragora.server.handlers.bots.zoom import ZoomHandler

__all__ = [
    "DiscordHandler",
    "TeamsHandler",
    "ZoomHandler",
]
