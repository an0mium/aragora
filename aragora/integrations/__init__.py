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
from aragora.integrations.receipt_webhooks import (
    ReceiptWebhookNotifier,
    ReceiptWebhookPayload,
    get_receipt_notifier,
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

# External automation platforms
from aragora.integrations.zapier import (
    ZapierApp,
    ZapierIntegration,
    ZapierTrigger,
    get_zapier_integration,
)
from aragora.integrations.make import (
    MakeConnection,
    MakeIntegration,
    MakeWebhook,
    get_make_integration,
)
from aragora.integrations.n8n import (
    N8nCredential,
    N8nIntegration,
    N8nWebhook,
    N8nResourceType,
    N8nOperation,
    get_n8n_integration,
)
from aragora.integrations.langchain import (
    AragoraTool,
    AragoraRetriever,
    AragoraCallbackHandler,
    is_langchain_available,
    LANGCHAIN_AVAILABLE,
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
    # Receipt Webhooks
    "ReceiptWebhookNotifier",
    "ReceiptWebhookPayload",
    "get_receipt_notifier",
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
    # Zapier
    "ZapierIntegration",
    "ZapierApp",
    "ZapierTrigger",
    "get_zapier_integration",
    # Make (Integromat)
    "MakeIntegration",
    "MakeConnection",
    "MakeWebhook",
    "get_make_integration",
    # n8n
    "N8nIntegration",
    "N8nCredential",
    "N8nWebhook",
    "N8nResourceType",
    "N8nOperation",
    "get_n8n_integration",
    # LangChain
    "AragoraTool",
    "AragoraRetriever",
    "AragoraCallbackHandler",
    "is_langchain_available",
    "LANGCHAIN_AVAILABLE",
]
