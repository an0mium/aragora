"""Aragora integrations for external services."""

from aragora.integrations.webhooks import (
    WebhookDispatcher,
    WebhookConfig,
    AragoraJSONEncoder,
    DEFAULT_EVENT_TYPES,
)
from aragora.integrations.slack import (
    SlackIntegration,
    SlackConfig,
    SlackMessage,
)
from aragora.integrations.discord import (
    DiscordIntegration,
    DiscordConfig,
    DiscordWebhookManager,
    discord_manager,
    create_discord_integration,
)

__all__ = [
    # Webhooks
    "WebhookDispatcher",
    "WebhookConfig",
    "AragoraJSONEncoder",
    "DEFAULT_EVENT_TYPES",
    # Slack
    "SlackIntegration",
    "SlackConfig",
    "SlackMessage",
    # Discord
    "DiscordIntegration",
    "DiscordConfig",
    "DiscordWebhookManager",
    "discord_manager",
    "create_discord_integration",
]