"""
Unified bot framework for chat platform integrations.

Provides a common interface for building bots that interact with Aragora
across different platforms (Slack, Discord, Teams, Zoom).

Usage:
    from aragora.bots import CommandRegistry, BotCommand, Platform

    # Register a command
    registry = CommandRegistry()

    @registry.command("debate", description="Start a multi-agent debate")
    async def handle_debate(ctx: CommandContext, topic: str):
        # Start debate and return result
        return await start_debate(topic)

    # Process incoming message
    result = await registry.execute("debate", ctx, "Should AI be regulated?")
"""

from aragora.bots.base import (
    Platform,
    CommandContext,
    CommandResult,
    BotConfig,
    BotMessage,
    BotUser,
    BotChannel,
    BotEventHandler,
    DefaultBotEventHandler,
)
from aragora.bots.commands import (
    BotCommand,
    CommandRegistry,
    command,
    get_default_registry,
)


# Platform-specific bots (lazy imports to avoid dependency issues)
def get_discord_bot():
    """Get Discord bot class (requires discord.py)."""
    from aragora.bots.discord_bot import AragoraDiscordBot, create_discord_bot

    return AragoraDiscordBot, create_discord_bot


def get_teams_bot():
    """Get Teams bot class (requires botbuilder-core)."""
    from aragora.bots.teams_bot import AragoraTeamsBot, create_teams_bot

    return AragoraTeamsBot, create_teams_bot


def get_zoom_bot():
    """Get Zoom bot class (requires aiohttp)."""
    from aragora.bots.zoom_bot import AragoraZoomBot, create_zoom_bot

    return AragoraZoomBot, create_zoom_bot


def get_slack_bot():
    """Get Slack bot class (requires slack-bolt)."""
    from aragora.bots.slack_bot import AragoraSlackBot, create_slack_bot

    return AragoraSlackBot, create_slack_bot


__all__ = [
    # Base classes
    "Platform",
    "CommandContext",
    "CommandResult",
    "BotConfig",
    "BotMessage",
    "BotUser",
    "BotChannel",
    "BotEventHandler",
    "DefaultBotEventHandler",
    # Command framework
    "BotCommand",
    "CommandRegistry",
    "command",
    "get_default_registry",
    # Bot getters (lazy)
    "get_discord_bot",
    "get_teams_bot",
    "get_zoom_bot",
    "get_slack_bot",
]
