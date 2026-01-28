"""
Automation platform connectors for Zapier, n8n, and similar services.

These connectors enable Aragora to integrate with workflow automation platforms
as both a trigger (event source) and an action (API endpoint).

Supported platforms:
- Zapier: Webhook-based triggers and REST actions
- n8n: Self-hosted workflow automation with node discovery

Usage:
    from aragora.connectors.automation import (
        ZapierConnector,
        N8NConnector,
        AutomationEventType,
        WebhookSubscription,
    )

    # Register a Zapier webhook
    connector = ZapierConnector()
    subscription = await connector.subscribe(
        webhook_url="https://hooks.zapier.com/...",
        events=[AutomationEventType.DEBATE_COMPLETED],
    )

    # Trigger event dispatch
    await connector.dispatch_event(
        AutomationEventType.CONSENSUS_REACHED,
        payload={"debate_id": "...", "consensus": "..."},
    )
"""

from aragora.connectors.automation.base import (
    AutomationConnector,
    AutomationEventType,
    WebhookSubscription,
    WebhookDeliveryResult,
)
from aragora.connectors.automation.zapier import ZapierConnector
from aragora.connectors.automation.n8n import N8NConnector

__all__ = [
    "AutomationConnector",
    "AutomationEventType",
    "WebhookSubscription",
    "WebhookDeliveryResult",
    "ZapierConnector",
    "N8NConnector",
]
