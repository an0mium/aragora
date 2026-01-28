"""
Zapier Connector for Aragora.

Enables Aragora to integrate with Zapier as both a trigger (event source)
and an action (API endpoint).

Zapier Integration Pattern:
- Triggers: Aragora dispatches events to Zapier webhook URLs
- Actions: Zapier calls Aragora API endpoints via REST

Webhook Signature:
- Uses HMAC-SHA256 for signature verification
- Signature header: X-Aragora-Signature
- Timestamp header: X-Aragora-Timestamp (prevents replay attacks)

Usage:
    from aragora.connectors.automation import ZapierConnector, AutomationEventType

    connector = ZapierConnector()

    # Subscribe to events
    sub = await connector.subscribe(
        webhook_url="https://hooks.zapier.com/hooks/catch/123/abc",
        events=[AutomationEventType.DEBATE_COMPLETED],
    )

    # Dispatch event
    results = await connector.dispatch_event(
        AutomationEventType.DEBATE_COMPLETED,
        payload={"debate_id": "...", "consensus": "..."},
    )
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

from aragora.connectors.automation.base import (
    AutomationConnector,
    AutomationEventType,
    WebhookSubscription,
)

logger = logging.getLogger(__name__)


class ZapierConnector(AutomationConnector):
    """
    Zapier-specific webhook connector.

    Formats payloads in Zapier's expected format and handles
    Zapier-specific authentication and verification.
    """

    PLATFORM_NAME = "zapier"

    # Zapier supports all event types
    SUPPORTED_EVENTS: Set[AutomationEventType] = set(AutomationEventType)

    # Zapier has specific payload requirements
    MAX_PAYLOAD_SIZE = 6 * 1024 * 1024  # 6MB limit

    def __init__(self, http_client: Optional[Any] = None):
        """Initialize Zapier connector."""
        super().__init__(http_client)
        logger.info("[Zapier] Connector initialized")

    async def format_payload(
        self,
        event_type: AutomationEventType,
        payload: Dict[str, Any],
        subscription: WebhookSubscription,
    ) -> Dict[str, Any]:
        """
        Format event payload for Zapier.

        Zapier expects a flat or nested JSON object that maps to trigger fields.
        We wrap the event in a standard envelope with metadata.

        Args:
            event_type: Type of event
            payload: Raw event data
            subscription: Target subscription

        Returns:
            Zapier-formatted payload
        """
        now = datetime.now(timezone.utc)

        # Build Zapier payload envelope
        formatted = {
            # Event metadata (available as trigger fields in Zapier)
            "id": f"{event_type.value}_{now.timestamp()}",
            "event_type": event_type.value,
            "event_category": event_type.value.split(".")[0],
            "timestamp": now.isoformat(),
            "timestamp_unix": int(now.timestamp()),
            # Subscription context
            "subscription_id": subscription.id,
            "workspace_id": subscription.workspace_id,
            # Event payload (flattened for easy Zapier field mapping)
            **self._flatten_for_zapier(payload),
        }

        return formatted

    def _flatten_for_zapier(
        self,
        data: Dict[str, Any],
        prefix: str = "",
    ) -> Dict[str, Any]:
        """
        Flatten nested dictionaries for Zapier field mapping.

        Zapier works best with flat field structures. This flattens
        nested dicts to dot-notation keys while preserving arrays.

        Args:
            data: Data to flatten
            prefix: Key prefix for nested items

        Returns:
            Flattened dictionary
        """
        result = {}

        for key, value in data.items():
            flat_key = f"{prefix}{key}" if prefix else key

            if isinstance(value, dict):
                # Recurse into nested dicts
                result.update(self._flatten_for_zapier(value, f"{flat_key}_"))
            elif isinstance(value, list):
                # Keep arrays as-is (Zapier can handle them)
                result[flat_key] = value
                # Also add count for convenience
                result[f"{flat_key}_count"] = len(value)
            else:
                result[flat_key] = value

        return result

    def generate_signature(
        self,
        payload: bytes,
        secret: str,
        timestamp: int,
    ) -> str:
        """
        Generate HMAC-SHA256 signature for Zapier webhook verification.

        Signature is computed over: timestamp + "." + payload

        Args:
            payload: JSON payload bytes
            secret: Subscription secret
            timestamp: Unix timestamp

        Returns:
            HMAC-SHA256 signature as hex string
        """
        # Construct signed payload (similar to Stripe's approach)
        signed_payload = f"{timestamp}.".encode() + payload

        signature = hmac.new(
            secret.encode(),
            signed_payload,
            hashlib.sha256,
        ).hexdigest()

        return signature

    async def test_subscription(
        self,
        subscription: WebhookSubscription,
    ) -> bool:
        """
        Send a test event to verify subscription.

        Zapier expects a test event during setup to validate the webhook.

        Args:
            subscription: Subscription to test

        Returns:
            True if test was successful
        """
        test_payload = {
            "test": True,
            "message": "This is a test event from Aragora",
            "subscription_id": subscription.id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        results = await self.dispatch_event(
            AutomationEventType.TEST_EVENT,
            test_payload,
            workspace_id=subscription.workspace_id,
        )

        # Find result for this subscription
        for result in results:
            if result.subscription_id == subscription.id:
                if result.success:
                    subscription.verified = True
                    logger.info(f"[Zapier] Subscription {subscription.id} verified")
                return result.success

        return False

    async def get_sample_data(
        self,
        event_type: AutomationEventType,
    ) -> Dict[str, Any]:
        """
        Get sample data for Zapier trigger setup.

        Zapier needs sample data to configure field mappings.

        Args:
            event_type: Event type to get sample for

        Returns:
            Sample event payload
        """
        samples = {
            AutomationEventType.DEBATE_COMPLETED: {
                "debate_id": "deb_sample123",
                "task": "Should we adopt microservices?",
                "consensus_reached": True,
                "final_answer": "Yes, with careful planning and gradual migration.",
                "confidence": 0.85,
                "rounds_completed": 3,
                "agents": ["claude", "gpt-4", "gemini"],
                "duration_seconds": 45,
            },
            AutomationEventType.CONSENSUS_REACHED: {
                "debate_id": "deb_sample123",
                "consensus_type": "supermajority",
                "confidence": 0.85,
                "supporting_agents": ["claude", "gpt-4"],
                "dissenting_agents": ["gemini"],
                "summary": "Agents agreed on the core recommendation.",
            },
            AutomationEventType.KNOWLEDGE_ADDED: {
                "knowledge_id": "kn_sample456",
                "title": "Q3 Sales Report",
                "source": "document",
                "content_preview": "Total revenue increased by 15%...",
                "confidence": 0.92,
                "tags": ["sales", "quarterly", "2024"],
            },
            AutomationEventType.DECISION_MADE: {
                "decision_id": "dec_sample789",
                "question": "Approve budget increase?",
                "decision": "approved",
                "rationale": "ROI projections justify the investment.",
                "decision_maker": "finance_team",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }

        default: Dict[str, Any] = {
            "event_type": event_type.value,
            "sample": True,
            "message": f"Sample payload for {event_type.value}",
        }
        return samples.get(event_type, default)
