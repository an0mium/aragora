"""
Zapier integration for Aragora.

Provides Zapier-compatible triggers and actions for workflow automation.
Implements Zapier's REST API webhook format with authentication and polling.

Triggers (webhook-based):
- debate_completed: Fires when a debate finishes
- consensus_reached: Fires when consensus is achieved
- gauntlet_completed: Fires when stress-test completes
- decision_made: Fires when a decision is finalized

Actions:
- start_debate: Start a new debate
- get_debate_status: Get current debate status
- submit_evidence: Submit evidence to a debate
"""

import hashlib
import hmac
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aragora.integrations.base import BaseIntegration

logger = logging.getLogger(__name__)


# =============================================================================
# Zapier Data Models
# =============================================================================


@dataclass
class ZapierTrigger:
    """Configuration for a Zapier trigger subscription."""

    id: str
    trigger_type: str
    webhook_url: str
    api_key: str
    created_at: float = field(default_factory=time.time)
    last_fired_at: Optional[float] = None
    fire_count: int = 0

    # Filtering
    workspace_id: Optional[str] = None
    debate_tags: Optional[List[str]] = None
    min_confidence: Optional[float] = None

    def matches_event(self, event: Dict[str, Any]) -> bool:
        """Check if an event matches this trigger's filters."""
        # Check workspace
        if self.workspace_id:
            if event.get("workspace_id") != self.workspace_id:
                return False

        # Check tags
        if self.debate_tags:
            event_tags = event.get("tags", [])
            if not any(tag in event_tags for tag in self.debate_tags):
                return False

        # Check confidence
        if self.min_confidence is not None:
            confidence = event.get("confidence", 0)
            if confidence < self.min_confidence:
                return False

        return True


@dataclass
class ZapierApp:
    """Zapier app configuration for an Aragora workspace."""

    id: str
    workspace_id: str
    api_key: str
    api_secret: str
    created_at: float = field(default_factory=time.time)
    triggers: Dict[str, ZapierTrigger] = field(default_factory=dict)
    active: bool = True

    # Usage tracking
    action_count: int = 0
    trigger_count: int = 0
    last_action_at: Optional[float] = None


# =============================================================================
# Zapier Integration
# =============================================================================


class ZapierIntegration(BaseIntegration):
    """
    Zapier integration for Aragora workflows.

    Supports:
    - REST Hook triggers (instant, webhook-based)
    - Polling triggers (for Zapier apps without instant triggers)
    - Action endpoints for debate management
    - Authentication via API key
    """

    # Supported trigger types
    TRIGGER_TYPES = {
        "debate_completed": "Fires when a debate finishes",
        "consensus_reached": "Fires when agents reach consensus",
        "gauntlet_completed": "Fires when stress-test completes",
        "decision_made": "Fires when a final decision is recorded",
        "agent_joined": "Fires when an agent joins a debate",
        "evidence_submitted": "Fires when evidence is added",
        "breakpoint_hit": "Fires when human intervention is needed",
    }

    # Supported action types
    ACTION_TYPES = {
        "start_debate": "Start a new multi-agent debate",
        "get_debate": "Get debate status and results",
        "submit_evidence": "Submit evidence to an active debate",
        "add_constraint": "Add a constraint to a debate",
        "request_synthesis": "Request a synthesis of current positions",
        "trigger_gauntlet": "Run stress-test on a decision",
    }

    def __init__(self, api_base: str = "https://aragora.ai"):
        """Initialize Zapier integration.

        Args:
            api_base: Base URL for API endpoints
        """
        super().__init__()
        self.api_base = api_base
        self._apps: Dict[str, ZapierApp] = {}

    @property
    def is_configured(self) -> bool:
        """Check if integration has any configured apps."""
        return len(self._apps) > 0

    async def send_message(self, content: str, **kwargs: Any) -> bool:
        """Send message to Zapier webhook (for triggers)."""
        webhook_url = kwargs.get("webhook_url")
        if not webhook_url:
            logger.warning("No webhook URL provided for Zapier message")
            return False

        session = await self._get_session()
        try:
            async with session.post(
                webhook_url,
                json={"message": content, **kwargs.get("data", {})},
                timeout=10,
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Failed to send Zapier webhook: {e}")
            return False

    # =========================================================================
    # App Management
    # =========================================================================

    def create_app(self, workspace_id: str) -> ZapierApp:
        """Create a new Zapier app configuration for a workspace.

        Args:
            workspace_id: Workspace to create app for

        Returns:
            New ZapierApp with generated credentials
        """
        app_id = f"zapier_{workspace_id}_{secrets.token_hex(8)}"
        api_key = f"zap_{secrets.token_urlsafe(32)}"
        api_secret = secrets.token_urlsafe(48)

        app = ZapierApp(
            id=app_id,
            workspace_id=workspace_id,
            api_key=api_key,
            api_secret=api_secret,
        )

        self._apps[app_id] = app
        logger.info(f"Created Zapier app {app_id} for workspace {workspace_id}")
        return app

    def get_app(self, app_id: str) -> Optional[ZapierApp]:
        """Get Zapier app by ID."""
        return self._apps.get(app_id)

    def get_app_by_key(self, api_key: str) -> Optional[ZapierApp]:
        """Get Zapier app by API key."""
        for app in self._apps.values():
            if app.api_key == api_key:
                return app
        return None

    def list_apps(self, workspace_id: Optional[str] = None) -> List[ZapierApp]:
        """List all Zapier apps, optionally filtered by workspace."""
        apps = list(self._apps.values())
        if workspace_id:
            apps = [a for a in apps if a.workspace_id == workspace_id]
        return apps

    def delete_app(self, app_id: str) -> bool:
        """Delete a Zapier app."""
        if app_id in self._apps:
            del self._apps[app_id]
            logger.info(f"Deleted Zapier app {app_id}")
            return True
        return False

    # =========================================================================
    # Trigger Management
    # =========================================================================

    def subscribe_trigger(
        self,
        app_id: str,
        trigger_type: str,
        webhook_url: str,
        workspace_id: Optional[str] = None,
        debate_tags: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
    ) -> Optional[ZapierTrigger]:
        """Subscribe to a trigger (REST Hook subscription).

        Args:
            app_id: Zapier app ID
            trigger_type: Type of trigger to subscribe to
            webhook_url: URL to send trigger events to
            workspace_id: Optional workspace filter
            debate_tags: Optional tag filter
            min_confidence: Optional minimum confidence filter

        Returns:
            ZapierTrigger if successful, None otherwise
        """
        app = self._apps.get(app_id)
        if not app:
            logger.warning(f"Zapier app not found: {app_id}")
            return None

        if trigger_type not in self.TRIGGER_TYPES:
            logger.warning(f"Invalid trigger type: {trigger_type}")
            return None

        trigger_id = f"trigger_{secrets.token_hex(8)}"
        trigger = ZapierTrigger(
            id=trigger_id,
            trigger_type=trigger_type,
            webhook_url=webhook_url,
            api_key=app.api_key,
            workspace_id=workspace_id,
            debate_tags=debate_tags,
            min_confidence=min_confidence,
        )

        app.triggers[trigger_id] = trigger
        logger.info(f"Subscribed trigger {trigger_id} ({trigger_type}) for app {app_id}")
        return trigger

    def unsubscribe_trigger(self, app_id: str, trigger_id: str) -> bool:
        """Unsubscribe from a trigger (REST Hook unsubscription).

        Args:
            app_id: Zapier app ID
            trigger_id: Trigger ID to unsubscribe

        Returns:
            True if unsubscribed, False otherwise
        """
        app = self._apps.get(app_id)
        if not app:
            return False

        if trigger_id in app.triggers:
            del app.triggers[trigger_id]
            logger.info(f"Unsubscribed trigger {trigger_id} from app {app_id}")
            return True
        return False

    def list_triggers(self, app_id: str) -> List[ZapierTrigger]:
        """List all triggers for a Zapier app."""
        app = self._apps.get(app_id)
        if not app:
            return []
        return list(app.triggers.values())

    # =========================================================================
    # Trigger Dispatch
    # =========================================================================

    async def fire_trigger(
        self,
        trigger_type: str,
        event_data: Dict[str, Any],
    ) -> int:
        """Fire a trigger for all matching subscriptions.

        Args:
            trigger_type: Type of trigger to fire
            event_data: Event data to send

        Returns:
            Number of triggers fired
        """
        fired_count = 0

        for app in self._apps.values():
            if not app.active:
                continue

            for trigger in app.triggers.values():
                if trigger.trigger_type != trigger_type:
                    continue

                if not trigger.matches_event(event_data):
                    continue

                # Fire the trigger
                success = await self._fire_single_trigger(trigger, event_data)
                if success:
                    trigger.last_fired_at = time.time()
                    trigger.fire_count += 1
                    app.trigger_count += 1
                    fired_count += 1

        if fired_count > 0:
            logger.info(f"Fired {fired_count} Zapier triggers for {trigger_type}")

        return fired_count

    async def _fire_single_trigger(
        self,
        trigger: ZapierTrigger,
        event_data: Dict[str, Any],
    ) -> bool:
        """Fire a single trigger webhook.

        Args:
            trigger: Trigger to fire
            event_data: Event data to send

        Returns:
            True if successful, False otherwise
        """
        # Format payload for Zapier
        payload = self._format_trigger_payload(trigger, event_data)

        session = await self._get_session()
        try:
            async with session.post(
                trigger.webhook_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-Aragora-Trigger": trigger.trigger_type,
                    "X-Aragora-Trigger-Id": trigger.id,
                },
                timeout=10,
            ) as response:
                if response.status == 200:
                    return True
                else:
                    logger.warning(
                        f"Zapier trigger {trigger.id} failed: {response.status}"
                    )
                    return False
        except Exception as e:
            logger.error(f"Failed to fire Zapier trigger {trigger.id}: {e}")
            return False

    def _format_trigger_payload(
        self,
        trigger: ZapierTrigger,
        event_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Format event data for Zapier trigger payload.

        Zapier expects a list of objects, even for single events.
        """
        # Ensure required fields
        formatted = {
            "id": event_data.get("id", f"evt_{secrets.token_hex(8)}"),
            "trigger_type": trigger.trigger_type,
            "timestamp": event_data.get("timestamp", time.time()),
            **event_data,
        }

        # Zapier expects a list for REST Hooks
        return [formatted]

    # =========================================================================
    # Polling Support
    # =========================================================================

    def get_polling_data(
        self,
        app_id: str,
        trigger_type: str,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get polling data for a trigger (for non-instant triggers).

        This is used by Zapier's polling mechanism when REST Hooks aren't available.

        Args:
            app_id: Zapier app ID
            trigger_type: Type of trigger to poll
            since: Unix timestamp to get events since
            limit: Maximum number of events to return

        Returns:
            List of events for polling
        """
        # This would typically query a database of events
        # For now, return empty list (implement with event store)
        return []

    # =========================================================================
    # Authentication
    # =========================================================================

    def verify_signature(
        self,
        payload: bytes,
        signature: str,
        api_secret: str,
    ) -> bool:
        """Verify webhook signature from Zapier.

        Args:
            payload: Raw request body
            signature: X-Zapier-Signature header value
            api_secret: App's API secret

        Returns:
            True if signature is valid
        """
        expected = hmac.new(
            api_secret.encode("utf-8"),
            payload,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(signature, expected)

    def authenticate_request(
        self,
        api_key: str,
    ) -> Optional[ZapierApp]:
        """Authenticate a request using API key.

        Args:
            api_key: API key from request header

        Returns:
            ZapierApp if authenticated, None otherwise
        """
        return self.get_app_by_key(api_key)


# =============================================================================
# Module-level singleton
# =============================================================================

_zapier_integration: Optional[ZapierIntegration] = None


def get_zapier_integration() -> ZapierIntegration:
    """Get or create the global Zapier integration instance."""
    global _zapier_integration
    if _zapier_integration is None:
        _zapier_integration = ZapierIntegration()
    return _zapier_integration


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ZapierIntegration",
    "ZapierApp",
    "ZapierTrigger",
    "get_zapier_integration",
]
