"""
Integration handlers for external automation platforms.

Provides HTTP handlers for:
- Webhook subscription management
- Event dispatch configuration
- Platform-specific endpoints (Zapier, n8n)
- Integration management (Slack, Teams, Discord, Email)
"""

from aragora.server.handlers.integrations.automation import AutomationHandler
from aragora.server.handlers.integrations.email_webhook import (
    EmailWebhookHandler,
    register_email_webhook_routes,
)

# Re-export IntegrationsHandler from the renamed module for backward compatibility
from aragora.server.handlers.integration_management import IntegrationsHandler

__all__ = [
    "AutomationHandler",
    "IntegrationsHandler",
    "EmailWebhookHandler",
    "register_email_webhook_routes",
]
