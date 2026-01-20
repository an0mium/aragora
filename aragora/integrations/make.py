"""
Make (Integromat) integration for Aragora.

Provides Make-compatible modules for workflow automation.
Implements Make's webhook format with instant triggers and actions.

Triggers (webhook-based):
- Watch Debates: Monitors for new or completed debates
- Watch Consensus: Monitors for consensus events
- Watch Decisions: Monitors for final decisions
- Watch Gauntlet: Monitors stress-test completions

Actions:
- Create Debate: Start a new debate
- Get Debate: Retrieve debate details
- Submit Evidence: Add evidence to a debate
- Get Agents: List available agents
- Run Gauntlet: Execute stress-test
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
# Make Data Models
# =============================================================================


@dataclass
class MakeWebhook:
    """Configuration for a Make webhook subscription."""

    id: str
    module_type: str
    webhook_url: str
    created_at: float = field(default_factory=time.time)
    last_triggered_at: Optional[float] = None
    trigger_count: int = 0

    # Filtering options
    workspace_id: Optional[str] = None
    event_filter: Optional[Dict[str, Any]] = None

    def matches_event(self, event: Dict[str, Any]) -> bool:
        """Check if an event matches this webhook's filters."""
        if self.workspace_id:
            if event.get("workspace_id") != self.workspace_id:
                return False

        if self.event_filter:
            for key, value in self.event_filter.items():
                if event.get(key) != value:
                    return False

        return True


@dataclass
class MakeConnection:
    """Make connection configuration for an Aragora workspace."""

    id: str
    workspace_id: str
    api_key: str
    created_at: float = field(default_factory=time.time)
    webhooks: Dict[str, MakeWebhook] = field(default_factory=dict)
    active: bool = True

    # Usage tracking
    total_operations: int = 0
    last_operation_at: Optional[float] = None


# =============================================================================
# Make Integration
# =============================================================================


class MakeIntegration(BaseIntegration):
    """
    Make (Integromat) integration for Aragora workflows.

    Supports:
    - Instant triggers via webhooks
    - Scheduled triggers via polling
    - Action modules for debate management
    - Connection testing and authentication
    """

    # Module types (triggers and actions)
    MODULE_TYPES = {
        # Instant triggers (webhook-based)
        "watch_debates": {
            "type": "trigger",
            "instant": True,
            "description": "Triggers when a debate starts or completes",
        },
        "watch_consensus": {
            "type": "trigger",
            "instant": True,
            "description": "Triggers when consensus is reached",
        },
        "watch_decisions": {
            "type": "trigger",
            "instant": True,
            "description": "Triggers when a final decision is made",
        },
        "watch_gauntlet": {
            "type": "trigger",
            "instant": True,
            "description": "Triggers when a gauntlet stress-test completes",
        },
        "watch_agents": {
            "type": "trigger",
            "instant": True,
            "description": "Triggers when agent status changes",
        },
        # Actions
        "create_debate": {
            "type": "action",
            "description": "Create and start a new debate",
        },
        "get_debate": {
            "type": "action",
            "description": "Get debate details and status",
        },
        "list_debates": {
            "type": "action",
            "description": "List recent debates",
        },
        "submit_evidence": {
            "type": "action",
            "description": "Submit evidence to an active debate",
        },
        "get_agents": {
            "type": "action",
            "description": "List available agents",
        },
        "run_gauntlet": {
            "type": "action",
            "description": "Run a gauntlet stress-test",
        },
        "get_decision_receipt": {
            "type": "action",
            "description": "Get a decision receipt for audit",
        },
    }

    def __init__(self, api_base: str = "https://aragora.ai"):
        """Initialize Make integration.

        Args:
            api_base: Base URL for API endpoints
        """
        super().__init__()
        self.api_base = api_base
        self._connections: Dict[str, MakeConnection] = {}

    @property
    def is_configured(self) -> bool:
        """Check if integration has any configured connections."""
        return len(self._connections) > 0

    async def send_message(self, content: str, **kwargs: Any) -> bool:
        """Send message to Make webhook."""
        webhook_url = kwargs.get("webhook_url")
        if not webhook_url:
            logger.warning("No webhook URL provided for Make message")
            return False

        session = await self._get_session()
        try:
            async with session.post(
                webhook_url,
                json={"content": content, **kwargs.get("data", {})},
                timeout=10,
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Failed to send Make webhook: {e}")
            return False

    # =========================================================================
    # Connection Management
    # =========================================================================

    def create_connection(self, workspace_id: str) -> MakeConnection:
        """Create a new Make connection for a workspace.

        Args:
            workspace_id: Workspace to create connection for

        Returns:
            New MakeConnection with generated credentials
        """
        conn_id = f"make_{workspace_id}_{secrets.token_hex(8)}"
        api_key = f"make_{secrets.token_urlsafe(32)}"

        connection = MakeConnection(
            id=conn_id,
            workspace_id=workspace_id,
            api_key=api_key,
        )

        self._connections[conn_id] = connection
        logger.info(f"Created Make connection {conn_id} for workspace {workspace_id}")
        return connection

    def get_connection(self, conn_id: str) -> Optional[MakeConnection]:
        """Get Make connection by ID."""
        return self._connections.get(conn_id)

    def get_connection_by_key(self, api_key: str) -> Optional[MakeConnection]:
        """Get Make connection by API key."""
        for conn in self._connections.values():
            if conn.api_key == api_key:
                return conn
        return None

    def list_connections(self, workspace_id: Optional[str] = None) -> List[MakeConnection]:
        """List all Make connections, optionally filtered by workspace."""
        connections = list(self._connections.values())
        if workspace_id:
            connections = [c for c in connections if c.workspace_id == workspace_id]
        return connections

    def delete_connection(self, conn_id: str) -> bool:
        """Delete a Make connection."""
        if conn_id in self._connections:
            del self._connections[conn_id]
            logger.info(f"Deleted Make connection {conn_id}")
            return True
        return False

    # =========================================================================
    # Webhook Management
    # =========================================================================

    def register_webhook(
        self,
        conn_id: str,
        module_type: str,
        webhook_url: str,
        workspace_id: Optional[str] = None,
        event_filter: Optional[Dict[str, Any]] = None,
    ) -> Optional[MakeWebhook]:
        """Register a webhook for a trigger module.

        Args:
            conn_id: Make connection ID
            module_type: Type of module (must be an instant trigger)
            webhook_url: URL to send events to
            workspace_id: Optional workspace filter
            event_filter: Optional event filter

        Returns:
            MakeWebhook if successful, None otherwise
        """
        connection = self._connections.get(conn_id)
        if not connection:
            logger.warning(f"Make connection not found: {conn_id}")
            return None

        module_config = self.MODULE_TYPES.get(module_type)
        if not module_config:
            logger.warning(f"Invalid module type: {module_type}")
            return None

        if module_config.get("type") != "trigger":
            logger.warning(f"Module {module_type} is not a trigger")
            return None

        webhook_id = f"webhook_{secrets.token_hex(8)}"
        webhook = MakeWebhook(
            id=webhook_id,
            module_type=module_type,
            webhook_url=webhook_url,
            workspace_id=workspace_id,
            event_filter=event_filter,
        )

        connection.webhooks[webhook_id] = webhook
        logger.info(f"Registered webhook {webhook_id} ({module_type}) for connection {conn_id}")
        return webhook

    def unregister_webhook(self, conn_id: str, webhook_id: str) -> bool:
        """Unregister a webhook.

        Args:
            conn_id: Make connection ID
            webhook_id: Webhook ID to unregister

        Returns:
            True if unregistered, False otherwise
        """
        connection = self._connections.get(conn_id)
        if not connection:
            return False

        if webhook_id in connection.webhooks:
            del connection.webhooks[webhook_id]
            logger.info(f"Unregistered webhook {webhook_id} from connection {conn_id}")
            return True
        return False

    def list_webhooks(self, conn_id: str) -> List[MakeWebhook]:
        """List all webhooks for a Make connection."""
        connection = self._connections.get(conn_id)
        if not connection:
            return []
        return list(connection.webhooks.values())

    # =========================================================================
    # Trigger Dispatch
    # =========================================================================

    async def trigger_webhooks(
        self,
        module_type: str,
        event_data: Dict[str, Any],
    ) -> int:
        """Trigger webhooks for all matching connections.

        Args:
            module_type: Type of trigger module
            event_data: Event data to send

        Returns:
            Number of webhooks triggered
        """
        triggered_count = 0

        for connection in self._connections.values():
            if not connection.active:
                continue

            for webhook in connection.webhooks.values():
                if webhook.module_type != module_type:
                    continue

                if not webhook.matches_event(event_data):
                    continue

                # Trigger the webhook
                success = await self._trigger_single_webhook(webhook, event_data)
                if success:
                    webhook.last_triggered_at = time.time()
                    webhook.trigger_count += 1
                    connection.total_operations += 1
                    connection.last_operation_at = time.time()
                    triggered_count += 1

        if triggered_count > 0:
            logger.info(f"Triggered {triggered_count} Make webhooks for {module_type}")

        return triggered_count

    async def _trigger_single_webhook(
        self,
        webhook: MakeWebhook,
        event_data: Dict[str, Any],
    ) -> bool:
        """Trigger a single webhook.

        Args:
            webhook: Webhook to trigger
            event_data: Event data to send

        Returns:
            True if successful, False otherwise
        """
        # Format payload for Make
        payload = self._format_webhook_payload(webhook, event_data)

        session = await self._get_session()
        try:
            async with session.post(
                webhook.webhook_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-Aragora-Module": webhook.module_type,
                    "X-Aragora-Webhook-Id": webhook.id,
                },
                timeout=10,
            ) as response:
                if response.status == 200:
                    return True
                else:
                    logger.warning(
                        f"Make webhook {webhook.id} failed: {response.status}"
                    )
                    return False
        except Exception as e:
            logger.error(f"Failed to trigger Make webhook {webhook.id}: {e}")
            return False

    def _format_webhook_payload(
        self,
        webhook: MakeWebhook,
        event_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Format event data for Make webhook payload."""
        return {
            "event_id": event_data.get("id", f"evt_{secrets.token_hex(8)}"),
            "module_type": webhook.module_type,
            "timestamp": event_data.get("timestamp", time.time()),
            "data": event_data,
        }

    # =========================================================================
    # Action Execution
    # =========================================================================

    async def execute_action(
        self,
        conn_id: str,
        action_type: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a Make action.

        Args:
            conn_id: Make connection ID
            action_type: Type of action to execute
            parameters: Action parameters

        Returns:
            Action result
        """
        connection = self._connections.get(conn_id)
        if not connection:
            return {"error": "Connection not found"}

        module_config = self.MODULE_TYPES.get(action_type)
        if not module_config:
            return {"error": f"Invalid action type: {action_type}"}

        if module_config.get("type") != "action":
            return {"error": f"Module {action_type} is not an action"}

        # Track operation
        connection.total_operations += 1
        connection.last_operation_at = time.time()

        # Execute action (would call actual API)
        result = await self._execute_action_internal(action_type, parameters)
        return result

    async def _execute_action_internal(
        self,
        action_type: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Internal action execution.

        This would typically make API calls to execute the action.
        """
        # Placeholder - actual implementation would call Aragora API
        return {
            "success": True,
            "action": action_type,
            "parameters": parameters,
            "timestamp": time.time(),
        }

    # =========================================================================
    # Authentication
    # =========================================================================

    def authenticate_request(
        self,
        api_key: str,
    ) -> Optional[MakeConnection]:
        """Authenticate a request using API key.

        Args:
            api_key: API key from request header

        Returns:
            MakeConnection if authenticated, None otherwise
        """
        return self.get_connection_by_key(api_key)

    def test_connection(self, conn_id: str) -> Dict[str, Any]:
        """Test a Make connection.

        Args:
            conn_id: Connection ID to test

        Returns:
            Test result
        """
        connection = self._connections.get(conn_id)
        if not connection:
            return {"success": False, "error": "Connection not found"}

        return {
            "success": True,
            "connection_id": conn_id,
            "workspace_id": connection.workspace_id,
            "webhooks_count": len(connection.webhooks),
            "total_operations": connection.total_operations,
        }


# =============================================================================
# Module-level singleton
# =============================================================================

_make_integration: Optional[MakeIntegration] = None


def get_make_integration() -> MakeIntegration:
    """Get or create the global Make integration instance."""
    global _make_integration
    if _make_integration is None:
        _make_integration = MakeIntegration()
    return _make_integration


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MakeIntegration",
    "MakeConnection",
    "MakeWebhook",
    "get_make_integration",
]
