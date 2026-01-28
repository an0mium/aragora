"""
Integration handlers for external automation platforms.

Provides HTTP handlers for:
- Webhook subscription management
- Event dispatch configuration
- Platform-specific endpoints (Zapier, n8n)
- Integration management (Slack, Teams, Discord, Email)
"""

from aragora.server.handlers.integrations.automation import AutomationHandler

# Import IntegrationsHandler from features module
try:
    from aragora.server.handlers.features.integrations import IntegrationsHandler
except ImportError:
    IntegrationsHandler = None  # type: ignore

__all__ = ["AutomationHandler", "IntegrationsHandler"]
