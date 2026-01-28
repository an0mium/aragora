"""
Integration handlers for external automation platforms.

Provides HTTP handlers for:
- Webhook subscription management
- Event dispatch configuration
- Platform-specific endpoints (Zapier, n8n)
"""

from aragora.server.handlers.integrations.automation import AutomationHandler

__all__ = ["AutomationHandler"]
