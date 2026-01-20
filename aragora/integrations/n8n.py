"""
n8n integration for Aragora.

Provides n8n-compatible nodes for workflow automation.
Implements n8n's webhook format and credential management.

Trigger Nodes:
- Aragora Trigger: Webhook-based trigger for debate events
- Aragora Poll Trigger: Polling-based trigger for events

Regular Nodes:
- Aragora: CRUD operations on debates, agents, evidence
- Aragora Gauntlet: Stress-testing operations
"""

import hashlib
import hmac
import logging
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from aragora.integrations.base import BaseIntegration

logger = logging.getLogger(__name__)


# =============================================================================
# n8n Data Models
# =============================================================================


class N8nResourceType(str, Enum):
    """n8n resource types for Aragora node."""

    DEBATE = "debate"
    AGENT = "agent"
    EVIDENCE = "evidence"
    DECISION = "decision"
    GAUNTLET = "gauntlet"
    KNOWLEDGE = "knowledge"


class N8nOperation(str, Enum):
    """n8n operations for Aragora node."""

    CREATE = "create"
    GET = "get"
    GET_ALL = "getAll"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"


@dataclass
class N8nWebhook:
    """Configuration for an n8n webhook subscription."""

    id: str
    webhook_path: str  # n8n uses path-based webhooks
    events: List[str]
    created_at: float = field(default_factory=time.time)
    last_triggered_at: Optional[float] = None
    trigger_count: int = 0

    # n8n workflow metadata
    workflow_id: Optional[str] = None
    node_id: Optional[str] = None

    # Filtering
    workspace_id: Optional[str] = None

    def matches_event(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """Check if an event matches this webhook's filters."""
        if event_type not in self.events and "*" not in self.events:
            return False

        if self.workspace_id:
            if event_data.get("workspace_id") != self.workspace_id:
                return False

        return True


@dataclass
class N8nCredential:
    """n8n credential configuration for Aragora."""

    id: str
    workspace_id: str
    api_key: str
    api_url: str
    created_at: float = field(default_factory=time.time)
    webhooks: Dict[str, N8nWebhook] = field(default_factory=dict)
    active: bool = True

    # Usage tracking
    operation_count: int = 0
    last_operation_at: Optional[float] = None


# =============================================================================
# n8n Node Definitions
# =============================================================================


# Node properties that define the Aragora node in n8n
N8N_NODE_DEFINITION = {
    "displayName": "Aragora",
    "name": "aragora",
    "icon": "file:aragora.svg",
    "group": ["transform"],
    "version": 1,
    "description": "Interact with Aragora multi-agent decision engine",
    "defaults": {
        "name": "Aragora",
    },
    "inputs": ["main"],
    "outputs": ["main"],
    "credentials": [
        {
            "name": "aragoraApi",
            "required": True,
        }
    ],
    "properties": [
        {
            "displayName": "Resource",
            "name": "resource",
            "type": "options",
            "noDataExpression": True,
            "options": [
                {"name": "Debate", "value": "debate"},
                {"name": "Agent", "value": "agent"},
                {"name": "Evidence", "value": "evidence"},
                {"name": "Decision", "value": "decision"},
                {"name": "Gauntlet", "value": "gauntlet"},
                {"name": "Knowledge", "value": "knowledge"},
            ],
            "default": "debate",
        },
        {
            "displayName": "Operation",
            "name": "operation",
            "type": "options",
            "noDataExpression": True,
            "displayOptions": {
                "show": {
                    "resource": ["debate"],
                },
            },
            "options": [
                {"name": "Create", "value": "create", "action": "Create a debate"},
                {"name": "Get", "value": "get", "action": "Get a debate"},
                {"name": "Get All", "value": "getAll", "action": "Get all debates"},
                {"name": "Delete", "value": "delete", "action": "Delete a debate"},
            ],
            "default": "create",
        },
    ],
}

N8N_TRIGGER_NODE_DEFINITION = {
    "displayName": "Aragora Trigger",
    "name": "aragoraTrigger",
    "icon": "file:aragora.svg",
    "group": ["trigger"],
    "version": 1,
    "description": "Starts the workflow when Aragora events occur",
    "defaults": {
        "name": "Aragora Trigger",
    },
    "inputs": [],
    "outputs": ["main"],
    "credentials": [
        {
            "name": "aragoraApi",
            "required": True,
        }
    ],
    "webhooks": [
        {
            "name": "default",
            "httpMethod": "POST",
            "responseMode": "onReceived",
            "path": "webhook",
        }
    ],
    "properties": [
        {
            "displayName": "Events",
            "name": "events",
            "type": "multiOptions",
            "options": [
                {"name": "Debate Started", "value": "debate_start"},
                {"name": "Debate Completed", "value": "debate_end"},
                {"name": "Consensus Reached", "value": "consensus"},
                {"name": "Decision Made", "value": "decision_made"},
                {"name": "Gauntlet Completed", "value": "gauntlet_complete"},
                {"name": "Evidence Submitted", "value": "evidence_submitted"},
                {"name": "Breakpoint Hit", "value": "breakpoint"},
            ],
            "default": ["debate_end"],
            "required": True,
        },
    ],
}


# =============================================================================
# n8n Integration
# =============================================================================


class N8nIntegration(BaseIntegration):
    """
    n8n integration for Aragora workflows.

    Supports:
    - Webhook triggers for instant notifications
    - Poll triggers for scheduled checks
    - CRUD operations on debates, agents, evidence
    - Gauntlet stress-testing integration
    """

    # Available event types for triggers
    EVENT_TYPES = {
        "debate_start": "Debate has started",
        "debate_end": "Debate has completed",
        "consensus": "Consensus reached",
        "decision_made": "Final decision recorded",
        "gauntlet_start": "Gauntlet stress-test started",
        "gauntlet_complete": "Gauntlet stress-test completed",
        "evidence_submitted": "Evidence submitted to debate",
        "agent_message": "Agent sent a message",
        "vote": "Vote cast in debate",
        "breakpoint": "Human intervention needed",
        "breakpoint_resolved": "Breakpoint resolved",
    }

    def __init__(self, api_base: str = "https://aragora.ai"):
        """Initialize n8n integration.

        Args:
            api_base: Base URL for API endpoints
        """
        super().__init__()
        self.api_base = api_base
        self._credentials: Dict[str, N8nCredential] = {}
        self._webhook_path_map: Dict[str, N8nWebhook] = {}  # path -> webhook

    @property
    def is_configured(self) -> bool:
        """Check if integration has any configured credentials."""
        return len(self._credentials) > 0

    async def send_message(self, content: str, **kwargs: Any) -> bool:
        """Send message to n8n webhook."""
        webhook_url = kwargs.get("webhook_url")
        if not webhook_url:
            logger.warning("No webhook URL provided for n8n message")
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
            logger.error(f"Failed to send n8n webhook: {e}")
            return False

    # =========================================================================
    # Credential Management
    # =========================================================================

    def create_credential(
        self,
        workspace_id: str,
        api_url: Optional[str] = None,
    ) -> N8nCredential:
        """Create a new n8n credential for a workspace.

        Args:
            workspace_id: Workspace to create credential for
            api_url: Optional custom API URL

        Returns:
            New N8nCredential with generated API key
        """
        cred_id = f"n8n_{workspace_id}_{secrets.token_hex(8)}"
        api_key = f"n8n_{secrets.token_urlsafe(32)}"

        credential = N8nCredential(
            id=cred_id,
            workspace_id=workspace_id,
            api_key=api_key,
            api_url=api_url or self.api_base,
        )

        self._credentials[cred_id] = credential
        logger.info(f"Created n8n credential {cred_id} for workspace {workspace_id}")
        return credential

    def get_credential(self, cred_id: str) -> Optional[N8nCredential]:
        """Get n8n credential by ID."""
        return self._credentials.get(cred_id)

    def get_credential_by_key(self, api_key: str) -> Optional[N8nCredential]:
        """Get n8n credential by API key."""
        for cred in self._credentials.values():
            if cred.api_key == api_key:
                return cred
        return None

    def list_credentials(self, workspace_id: Optional[str] = None) -> List[N8nCredential]:
        """List all n8n credentials, optionally filtered by workspace."""
        credentials = list(self._credentials.values())
        if workspace_id:
            credentials = [c for c in credentials if c.workspace_id == workspace_id]
        return credentials

    def delete_credential(self, cred_id: str) -> bool:
        """Delete an n8n credential."""
        if cred_id in self._credentials:
            # Also remove associated webhooks from path map
            cred = self._credentials[cred_id]
            for webhook in cred.webhooks.values():
                if webhook.webhook_path in self._webhook_path_map:
                    del self._webhook_path_map[webhook.webhook_path]

            del self._credentials[cred_id]
            logger.info(f"Deleted n8n credential {cred_id}")
            return True
        return False

    # =========================================================================
    # Webhook Management
    # =========================================================================

    def register_webhook(
        self,
        cred_id: str,
        events: List[str],
        workflow_id: Optional[str] = None,
        node_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> Optional[N8nWebhook]:
        """Register a webhook for n8n trigger node.

        Args:
            cred_id: n8n credential ID
            events: List of event types to subscribe to
            workflow_id: Optional n8n workflow ID
            node_id: Optional n8n node ID
            workspace_id: Optional workspace filter

        Returns:
            N8nWebhook if successful, None otherwise
        """
        credential = self._credentials.get(cred_id)
        if not credential:
            logger.warning(f"n8n credential not found: {cred_id}")
            return None

        # Validate events
        invalid_events = [e for e in events if e != "*" and e not in self.EVENT_TYPES]
        if invalid_events:
            logger.warning(f"Invalid event types: {invalid_events}")
            return None

        webhook_id = f"webhook_{secrets.token_hex(8)}"
        webhook_path = f"/n8n/webhook/{webhook_id}"

        webhook = N8nWebhook(
            id=webhook_id,
            webhook_path=webhook_path,
            events=events,
            workflow_id=workflow_id,
            node_id=node_id,
            workspace_id=workspace_id,
        )

        credential.webhooks[webhook_id] = webhook
        self._webhook_path_map[webhook_path] = webhook

        logger.info(
            f"Registered n8n webhook {webhook_id} for credential {cred_id}, "
            f"events: {events}"
        )
        return webhook

    def unregister_webhook(self, cred_id: str, webhook_id: str) -> bool:
        """Unregister an n8n webhook.

        Args:
            cred_id: n8n credential ID
            webhook_id: Webhook ID to unregister

        Returns:
            True if unregistered, False otherwise
        """
        credential = self._credentials.get(cred_id)
        if not credential:
            return False

        if webhook_id in credential.webhooks:
            webhook = credential.webhooks[webhook_id]
            if webhook.webhook_path in self._webhook_path_map:
                del self._webhook_path_map[webhook.webhook_path]
            del credential.webhooks[webhook_id]
            logger.info(f"Unregistered n8n webhook {webhook_id} from credential {cred_id}")
            return True
        return False

    def get_webhook_by_path(self, path: str) -> Optional[N8nWebhook]:
        """Get webhook by its path."""
        return self._webhook_path_map.get(path)

    def list_webhooks(self, cred_id: str) -> List[N8nWebhook]:
        """List all webhooks for an n8n credential."""
        credential = self._credentials.get(cred_id)
        if not credential:
            return []
        return list(credential.webhooks.values())

    # =========================================================================
    # Event Dispatch
    # =========================================================================

    async def dispatch_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        n8n_instance_url: Optional[str] = None,
    ) -> int:
        """Dispatch an event to all matching n8n webhooks.

        Args:
            event_type: Type of event
            event_data: Event data to send
            n8n_instance_url: Optional n8n instance URL override

        Returns:
            Number of webhooks triggered
        """
        triggered_count = 0

        for credential in self._credentials.values():
            if not credential.active:
                continue

            for webhook in credential.webhooks.values():
                if not webhook.matches_event(event_type, event_data):
                    continue

                # Dispatch to webhook
                success = await self._dispatch_to_webhook(
                    webhook,
                    event_type,
                    event_data,
                    n8n_instance_url or credential.api_url,
                )

                if success:
                    webhook.last_triggered_at = time.time()
                    webhook.trigger_count += 1
                    credential.operation_count += 1
                    credential.last_operation_at = time.time()
                    triggered_count += 1

        if triggered_count > 0:
            logger.info(f"Dispatched {triggered_count} n8n webhooks for {event_type}")

        return triggered_count

    async def _dispatch_to_webhook(
        self,
        webhook: N8nWebhook,
        event_type: str,
        event_data: Dict[str, Any],
        base_url: str,
    ) -> bool:
        """Dispatch event to a single n8n webhook.

        Args:
            webhook: Webhook to trigger
            event_type: Type of event
            event_data: Event data to send
            base_url: Base URL for the n8n instance

        Returns:
            True if successful, False otherwise
        """
        # Format payload for n8n
        payload = self._format_webhook_payload(webhook, event_type, event_data)

        # n8n webhooks typically use the workflow's webhook URL
        webhook_url = f"{base_url}{webhook.webhook_path}"

        session = await self._get_session()
        try:
            async with session.post(
                webhook_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-Aragora-Event": event_type,
                    "X-Aragora-Webhook-Id": webhook.id,
                },
                timeout=10,
            ) as response:
                if response.status == 200:
                    return True
                else:
                    logger.warning(
                        f"n8n webhook {webhook.id} failed: {response.status}"
                    )
                    return False
        except Exception as e:
            logger.error(f"Failed to dispatch to n8n webhook {webhook.id}: {e}")
            return False

    def _format_webhook_payload(
        self,
        webhook: N8nWebhook,
        event_type: str,
        event_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Format event data for n8n webhook payload."""
        return {
            "event": event_type,
            "timestamp": event_data.get("timestamp", time.time()),
            "webhook_id": webhook.id,
            "workflow_id": webhook.workflow_id,
            "node_id": webhook.node_id,
            "data": event_data,
        }

    # =========================================================================
    # Node Operations
    # =========================================================================

    async def execute_operation(
        self,
        cred_id: str,
        resource: N8nResourceType,
        operation: N8nOperation,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute an n8n node operation.

        Args:
            cred_id: n8n credential ID
            resource: Resource type (debate, agent, etc.)
            operation: Operation type (create, get, etc.)
            parameters: Operation parameters

        Returns:
            Operation result
        """
        credential = self._credentials.get(cred_id)
        if not credential:
            return {"success": False, "error": "Credential not found"}

        # Track operation
        credential.operation_count += 1
        credential.last_operation_at = time.time()

        # Execute operation based on resource and operation type
        result = await self._execute_operation_internal(resource, operation, parameters)
        return result

    async def _execute_operation_internal(
        self,
        resource: N8nResourceType,
        operation: N8nOperation,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Internal operation execution.

        This would typically make API calls to execute the operation.
        """
        # Placeholder - actual implementation would call Aragora API
        return {
            "success": True,
            "resource": resource.value,
            "operation": operation.value,
            "parameters": parameters,
            "timestamp": time.time(),
        }

    # =========================================================================
    # Authentication
    # =========================================================================

    def authenticate_request(
        self,
        api_key: str,
    ) -> Optional[N8nCredential]:
        """Authenticate a request using API key.

        Args:
            api_key: API key from request header

        Returns:
            N8nCredential if authenticated, None otherwise
        """
        return self.get_credential_by_key(api_key)

    def verify_webhook_signature(
        self,
        payload: bytes,
        signature: str,
        webhook_id: str,
    ) -> bool:
        """Verify webhook signature from n8n.

        Args:
            payload: Raw request body
            signature: Signature header value
            webhook_id: Webhook ID

        Returns:
            True if signature is valid
        """
        # Find the credential containing this webhook
        for cred in self._credentials.values():
            if webhook_id in cred.webhooks:
                expected = hmac.new(
                    cred.api_key.encode("utf-8"),
                    payload,
                    hashlib.sha256,
                ).hexdigest()
                return hmac.compare_digest(signature, expected)
        return False

    # =========================================================================
    # Node Definitions Export
    # =========================================================================

    def get_node_definition(self) -> Dict[str, Any]:
        """Get the n8n node definition for Aragora."""
        return N8N_NODE_DEFINITION

    def get_trigger_node_definition(self) -> Dict[str, Any]:
        """Get the n8n trigger node definition for Aragora."""
        return N8N_TRIGGER_NODE_DEFINITION

    def get_credential_definition(self) -> Dict[str, Any]:
        """Get the n8n credential definition for Aragora API."""
        return {
            "name": "aragoraApi",
            "displayName": "Aragora API",
            "documentationUrl": "https://docs.aragora.ai/integrations/n8n",
            "properties": [
                {
                    "displayName": "API Key",
                    "name": "apiKey",
                    "type": "string",
                    "typeOptions": {"password": True},
                    "default": "",
                    "required": True,
                },
                {
                    "displayName": "API URL",
                    "name": "apiUrl",
                    "type": "string",
                    "default": "https://api.aragora.ai",
                    "required": True,
                },
            ],
        }


# =============================================================================
# Module-level singleton
# =============================================================================

_n8n_integration: Optional[N8nIntegration] = None


def get_n8n_integration() -> N8nIntegration:
    """Get or create the global n8n integration instance."""
    global _n8n_integration
    if _n8n_integration is None:
        _n8n_integration = N8nIntegration()
    return _n8n_integration


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "N8nIntegration",
    "N8nCredential",
    "N8nWebhook",
    "N8nResourceType",
    "N8nOperation",
    "get_n8n_integration",
    "N8N_NODE_DEFINITION",
    "N8N_TRIGGER_NODE_DEFINITION",
]
