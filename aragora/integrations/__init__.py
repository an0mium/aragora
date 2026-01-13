"""Aragora integrations for external services."""

from aragora.integrations.discord import (
    DiscordConfig,
    DiscordIntegration,
    DiscordWebhookManager,
    create_discord_integration,
    discord_manager,
)
from aragora.integrations.email import (
    EmailConfig,
    EmailIntegration,
    EmailRecipient,
)
from aragora.integrations.slack import (
    SlackConfig,
    SlackIntegration,
    SlackMessage,
)
from aragora.integrations.telegram import (
    InlineButton,
    TelegramConfig,
    TelegramIntegration,
    TelegramMessage,
)
from aragora.integrations.webhooks import (
    DEFAULT_EVENT_TYPES,
    AragoraJSONEncoder,
    WebhookConfig,
    WebhookDispatcher,
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
    # Telegram
    "TelegramIntegration",
    "TelegramConfig",
    "TelegramMessage",
    "InlineButton",
    # Email
    "EmailIntegration",
    "EmailConfig",
    "EmailRecipient",
]
