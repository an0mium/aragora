"""
Base classes for automation platform connectors.

Provides common abstractions for webhook-based automation integrations.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class AutomationEventType(str, Enum):
    """Events that can trigger automation workflows."""

    # Debate lifecycle events
    DEBATE_STARTED = "debate.started"
    DEBATE_ROUND_COMPLETED = "debate.round_completed"
    DEBATE_COMPLETED = "debate.completed"
    DEBATE_FAILED = "debate.failed"

    # Consensus events
    CONSENSUS_REACHED = "consensus.reached"
    CONSENSUS_FAILED = "consensus.failed"

    # Agent events
    AGENT_RESPONSE = "agent.response"
    AGENT_CRITIQUE = "agent.critique"
    AGENT_VOTE = "agent.vote"

    # Knowledge events
    KNOWLEDGE_ADDED = "knowledge.added"
    KNOWLEDGE_UPDATED = "knowledge.updated"
    KNOWLEDGE_QUERY = "knowledge.query"

    # Decision events
    DECISION_MADE = "decision.made"
    RECEIPT_GENERATED = "receipt.generated"

    # User events
    USER_FEEDBACK = "user.feedback"
    USER_VOTE = "user.vote"

    # System events
    HEALTH_CHECK = "health.check"
    TEST_EVENT = "test.event"


@dataclass
class WebhookSubscription:
    """A webhook subscription for automation events."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    webhook_url: str = ""
    events: set[AutomationEventType] = field(default_factory=set)
    secret: str = field(default_factory=lambda: hashlib.sha256(uuid.uuid4().bytes).hexdigest()[:32])

    # Metadata
    platform: str = "generic"  # zapier, n8n, custom
    workspace_id: str | None = None
    user_id: str | None = None
    name: str | None = None
    description: str | None = None

    # Status
    enabled: bool = True
    verified: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_delivery_at: datetime | None = None
    delivery_count: int = 0
    failure_count: int = 0

    # Configuration
    retry_count: int = 3
    timeout_seconds: int = 30
    headers: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "webhook_url": self.webhook_url,
            "events": [e.value for e in self.events],
            "platform": self.platform,
            "workspace_id": self.workspace_id,
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "verified": self.verified,
            "created_at": self.created_at.isoformat(),
            "last_delivery_at": self.last_delivery_at.isoformat()
            if self.last_delivery_at
            else None,
            "delivery_count": self.delivery_count,
            "failure_count": self.failure_count,
            "retry_count": self.retry_count,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WebhookSubscription:
        """Deserialize from dictionary."""
        events = {AutomationEventType(e) for e in data.get("events", [])}
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            webhook_url=data.get("webhook_url", ""),
            events=events,
            secret=data.get("secret", hashlib.sha256(uuid.uuid4().bytes).hexdigest()[:32]),
            platform=data.get("platform", "generic"),
            workspace_id=data.get("workspace_id"),
            user_id=data.get("user_id"),
            name=data.get("name"),
            description=data.get("description"),
            enabled=data.get("enabled", True),
            verified=data.get("verified", False),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(timezone.utc),
            last_delivery_at=datetime.fromisoformat(data["last_delivery_at"])
            if data.get("last_delivery_at")
            else None,
            delivery_count=data.get("delivery_count", 0),
            failure_count=data.get("failure_count", 0),
            retry_count=data.get("retry_count", 3),
            timeout_seconds=data.get("timeout_seconds", 30),
            headers=data.get("headers", {}),
        )


@dataclass
class WebhookDeliveryResult:
    """Result of a webhook delivery attempt."""

    subscription_id: str
    event_type: AutomationEventType
    success: bool
    status_code: int | None = None
    response_body: str | None = None
    error: str | None = None
    duration_ms: float = 0
    attempt: int = 1
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AutomationConnector(ABC):
    """
    Base class for automation platform connectors.

    Subclasses implement platform-specific webhook handling and event formatting.
    """

    PLATFORM_NAME: str = "generic"
    SUPPORTED_EVENTS: set[AutomationEventType] = set(AutomationEventType)

    def __init__(self, http_client: Any | None = None):
        """
        Initialize the connector.

        Args:
            http_client: Optional HTTP client (httpx.AsyncClient or similar)
        """
        self._http_client = http_client
        self._subscriptions: dict[str, WebhookSubscription] = {}

    @abstractmethod
    async def format_payload(
        self,
        event_type: AutomationEventType,
        payload: dict[str, Any],
        subscription: WebhookSubscription,
    ) -> dict[str, Any]:
        """
        Format event payload for the target platform.

        Args:
            event_type: Type of event
            payload: Raw event data
            subscription: Target subscription

        Returns:
            Formatted payload for the platform
        """
        pass

    @abstractmethod
    def generate_signature(
        self,
        payload: bytes,
        secret: str,
        timestamp: int,
    ) -> str:
        """
        Generate webhook signature for verification.

        Args:
            payload: JSON payload bytes
            secret: Subscription secret
            timestamp: Unix timestamp

        Returns:
            HMAC signature string
        """
        pass

    def verify_signature(
        self,
        payload: bytes,
        signature: str,
        secret: str,
        timestamp: int,
    ) -> bool:
        """
        Verify webhook signature.

        Args:
            payload: JSON payload bytes
            signature: Provided signature
            secret: Subscription secret
            timestamp: Unix timestamp

        Returns:
            True if signature is valid
        """
        expected = self.generate_signature(payload, secret, timestamp)
        return hmac.compare_digest(expected, signature)

    async def subscribe(
        self,
        webhook_url: str,
        events: list[AutomationEventType],
        workspace_id: str | None = None,
        user_id: str | None = None,
        name: str | None = None,
    ) -> WebhookSubscription:
        """
        Create a new webhook subscription.

        Args:
            webhook_url: URL to deliver events to
            events: List of event types to subscribe to
            workspace_id: Optional workspace ID
            user_id: Optional user ID
            name: Optional subscription name

        Returns:
            Created subscription
        """
        # Validate events
        unsupported = set(events) - self.SUPPORTED_EVENTS
        if unsupported:
            raise ValueError(f"Unsupported events for {self.PLATFORM_NAME}: {unsupported}")

        # SECURITY: Validate webhook URL to prevent SSRF attacks
        try:
            from aragora.security.ssrf_protection import validate_webhook_url

            validate_webhook_url(webhook_url)
        except ImportError:
            logger.warning("SSRF protection module not available, validating URL scheme only")
            from urllib.parse import urlparse

            parsed = urlparse(webhook_url)
            if parsed.scheme not in ("http", "https"):
                raise ValueError(f"Webhook URL must use http/https scheme, got: {parsed.scheme}")

        subscription = WebhookSubscription(
            webhook_url=webhook_url,
            events=set(events),
            platform=self.PLATFORM_NAME,
            workspace_id=workspace_id,
            user_id=user_id,
            name=name,
        )

        self._subscriptions[subscription.id] = subscription
        logger.info(
            "[%s] Created subscription %s for %s events", self.PLATFORM_NAME, subscription.id, len(events)
        )

        return subscription

    async def unsubscribe(self, subscription_id: str) -> bool:
        """
        Remove a webhook subscription.

        Args:
            subscription_id: ID of subscription to remove

        Returns:
            True if subscription was found and removed
        """
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            logger.info("[%s] Removed subscription %s", self.PLATFORM_NAME, subscription_id)
            return True
        return False

    async def dispatch_event(
        self,
        event_type: AutomationEventType,
        payload: dict[str, Any],
        workspace_id: str | None = None,
    ) -> list[WebhookDeliveryResult]:
        """
        Dispatch event to all matching subscriptions.

        Args:
            event_type: Type of event
            payload: Event payload
            workspace_id: Optional workspace filter

        Returns:
            List of delivery results
        """
        results = []

        for subscription in self._subscriptions.values():
            # Check if subscription matches
            if not subscription.enabled:
                continue
            if event_type not in subscription.events:
                continue
            if (
                workspace_id
                and subscription.workspace_id
                and subscription.workspace_id != workspace_id
            ):
                continue

            # Deliver webhook
            result = await self._deliver_webhook(subscription, event_type, payload)
            results.append(result)

        return results

    async def _deliver_webhook(
        self,
        subscription: WebhookSubscription,
        event_type: AutomationEventType,
        payload: dict[str, Any],
    ) -> WebhookDeliveryResult:
        """
        Deliver webhook to a subscription.

        Args:
            subscription: Target subscription
            event_type: Event type
            payload: Event payload

        Returns:
            Delivery result
        """
        import json

        start_time = time.time()

        try:
            # SECURITY: Defense-in-depth SSRF check at delivery time
            # (subscriptions may be deserialized from storage, bypassing subscribe())
            try:
                from aragora.security.ssrf_protection import validate_webhook_url

                validate_webhook_url(subscription.webhook_url)
            except ImportError:
                pass  # Best-effort; primary check is in subscribe()
            except (ValueError, OSError) as ssrf_err:
                duration_ms = (time.time() - start_time) * 1000
                logger.warning(
                    "[%s] Blocked SSRF attempt to %s: %s", self.PLATFORM_NAME, subscription.webhook_url, ssrf_err
                )
                return WebhookDeliveryResult(
                    subscription_id=subscription.id,
                    event_type=event_type,
                    success=False,
                    status_code=0,
                    response_body=f"SSRF validation failed: {ssrf_err}",
                    duration_ms=duration_ms,
                )

            # Format payload
            formatted = await self.format_payload(event_type, payload, subscription)
            payload_bytes = json.dumps(formatted).encode()

            # Generate signature
            timestamp = int(time.time())
            signature = self.generate_signature(payload_bytes, subscription.secret, timestamp)

            # Build headers
            headers = {
                "Content-Type": "application/json",
                "X-Aragora-Signature": signature,
                "X-Aragora-Timestamp": str(timestamp),
                "X-Aragora-Event": event_type.value,
                "X-Aragora-Subscription-Id": subscription.id,
                **subscription.headers,
            }

            # Send request
            if self._http_client:
                response = await self._http_client.post(
                    subscription.webhook_url,
                    content=payload_bytes,
                    headers=headers,
                    timeout=subscription.timeout_seconds,
                )
                status_code = response.status_code
                response_body = response.text[:1000] if response.text else None
                success = 200 <= status_code < 300
            else:
                # Dry run mode
                logger.info(
                    "[%s] Dry run: would deliver to %s", self.PLATFORM_NAME, subscription.webhook_url
                )
                status_code = 200
                response_body = None
                success = True

            # Update subscription stats
            subscription.delivery_count += 1
            subscription.last_delivery_at = datetime.now(timezone.utc)
            if not success:
                subscription.failure_count += 1

            duration_ms = (time.time() - start_time) * 1000

            return WebhookDeliveryResult(
                subscription_id=subscription.id,
                event_type=event_type,
                success=success,
                status_code=status_code,
                response_body=response_body,
                duration_ms=duration_ms,
            )

        except (OSError, ValueError, RuntimeError, TypeError) as e:
            duration_ms = (time.time() - start_time) * 1000
            subscription.failure_count += 1

            logger.warning("[%s] Webhook delivery failed: %s", self.PLATFORM_NAME, e)

            return WebhookDeliveryResult(
                subscription_id=subscription.id,
                event_type=event_type,
                success=False,
                error="Webhook delivery failed",
                duration_ms=duration_ms,
            )

    def get_subscription(self, subscription_id: str) -> WebhookSubscription | None:
        """Get a subscription by ID."""
        return self._subscriptions.get(subscription_id)

    def list_subscriptions(
        self,
        workspace_id: str | None = None,
        event_type: AutomationEventType | None = None,
    ) -> list[WebhookSubscription]:
        """
        List subscriptions with optional filters.

        Args:
            workspace_id: Optional workspace filter
            event_type: Optional event type filter

        Returns:
            Matching subscriptions
        """
        results = []
        for sub in self._subscriptions.values():
            if workspace_id and sub.workspace_id != workspace_id:
                continue
            if event_type and event_type not in sub.events:
                continue
            results.append(sub)
        return results
