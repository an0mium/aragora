"""Aragora integrations for external services."""

from aragora.integrations.base import (
    BaseIntegration,
    FormattedConsensusData,
    FormattedDebateData,
    FormattedErrorData,
    FormattedLeaderboardData,
)
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
    EmailProvider,
    EmailRecipient,
)
from aragora.integrations.matrix import (
    MatrixConfig,
    MatrixIntegration,
)
from aragora.integrations.slack import (
    SlackConfig,
    SlackIntegration,
    SlackMessage,
)
from aragora.integrations.teams import (
    AdaptiveCard,
    TeamsConfig,
    TeamsIntegration,
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
from aragora.integrations.whatsapp import (
    WhatsAppConfig,
    WhatsAppIntegration,
    WhatsAppProvider,
)
from aragora.integrations.zoom import (
    ZoomConfig,
    ZoomIntegration,
    ZoomMeetingInfo,
    ZoomWebhookEvent,
)

__all__ = [
    # Base
    "BaseIntegration",
    "FormattedDebateData",
    "FormattedConsensusData",
    "FormattedErrorData",
    "FormattedLeaderboardData",
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
    "EmailProvider",
    "EmailRecipient",
    # Microsoft Teams
    "TeamsIntegration",
    "TeamsConfig",
    "AdaptiveCard",
    # WhatsApp
    "WhatsAppIntegration",
    "WhatsAppConfig",
    "WhatsAppProvider",
    # Matrix/Element
    "MatrixIntegration",
    "MatrixConfig",
    # Zoom
    "ZoomIntegration",
    "ZoomConfig",
    "ZoomMeetingInfo",
    "ZoomWebhookEvent",
]
