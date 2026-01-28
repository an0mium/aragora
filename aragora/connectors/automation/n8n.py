"""
n8n Connector for Aragora.

Enables Aragora to integrate with n8n (self-hosted workflow automation)
as both a trigger (event source) and node (API actions).

n8n Integration Pattern:
- Triggers: Aragora dispatches events to n8n webhook URLs
- Nodes: n8n executes Aragora API operations
- Community Node: Can be packaged as n8n community node

Node Discovery:
n8n supports community nodes that describe available operations.
This connector provides node definition metadata for n8n discovery.

Usage:
    from aragora.connectors.automation import N8NConnector, AutomationEventType

    connector = N8NConnector()

    # Subscribe to events
    sub = await connector.subscribe(
        webhook_url="http://localhost:5678/webhook/aragora",
        events=[AutomationEventType.DEBATE_COMPLETED],
    )

    # Get node definition for n8n
    node_def = connector.get_node_definition()
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from aragora.connectors.automation.base import (
    AutomationConnector,
    AutomationEventType,
    WebhookSubscription,
)

logger = logging.getLogger(__name__)


class N8NConnector(AutomationConnector):
    """
    n8n-specific webhook connector.

    Provides n8n-compatible webhook handling and node definition
    for community node integration.
    """

    PLATFORM_NAME = "n8n"

    # n8n supports all event types
    SUPPORTED_EVENTS: Set[AutomationEventType] = set(AutomationEventType)

    def __init__(
        self,
        http_client: Optional[Any] = None,
        aragora_base_url: str = "http://localhost:8080",
    ):
        """
        Initialize n8n connector.

        Args:
            http_client: Optional HTTP client
            aragora_base_url: Base URL for Aragora API (used in node definition)
        """
        super().__init__(http_client)
        self.aragora_base_url = aragora_base_url
        logger.info("[n8n] Connector initialized")

    async def format_payload(
        self,
        event_type: AutomationEventType,
        payload: Dict[str, Any],
        subscription: WebhookSubscription,
    ) -> Dict[str, Any]:
        """
        Format event payload for n8n.

        n8n works well with structured JSON. We provide a clean
        envelope with typed fields for easy workflow building.

        Args:
            event_type: Type of event
            payload: Raw event data
            subscription: Target subscription

        Returns:
            n8n-formatted payload
        """
        now = datetime.now(timezone.utc)

        # Build n8n payload with clear structure
        formatted = {
            # Event metadata
            "meta": {
                "event_id": f"{event_type.value}_{int(now.timestamp())}",
                "event_type": event_type.value,
                "category": event_type.value.split(".")[0],
                "action": event_type.value.split(".")[-1]
                if "." in event_type.value
                else event_type.value,
                "timestamp": now.isoformat(),
                "source": "aragora",
            },
            # Subscription context
            "context": {
                "subscription_id": subscription.id,
                "workspace_id": subscription.workspace_id,
                "user_id": subscription.user_id,
            },
            # Event data (preserved structure)
            "data": payload,
        }

        return formatted

    def generate_signature(
        self,
        payload: bytes,
        secret: str,
        timestamp: int,
    ) -> str:
        """
        Generate HMAC-SHA256 signature for n8n webhook verification.

        n8n supports custom header-based authentication.

        Args:
            payload: JSON payload bytes
            secret: Subscription secret
            timestamp: Unix timestamp

        Returns:
            HMAC-SHA256 signature as hex string
        """
        # Use timestamp in signature for replay protection
        signed_payload = f"v0:{timestamp}:".encode() + payload

        signature = hmac.new(
            secret.encode(),
            signed_payload,
            hashlib.sha256,
        ).hexdigest()

        return f"v0={signature}"

    def get_node_definition(self) -> Dict[str, Any]:
        """
        Get n8n community node definition.

        Returns the node definition that describes Aragora's capabilities
        for n8n's node discovery and UI building.

        Returns:
            n8n node definition
        """
        return {
            "name": "Aragora",
            "displayName": "Aragora",
            "description": "Multi-agent debate and decision-making platform",
            "icon": "file:aragora.svg",
            "group": ["transform"],
            "version": 1,
            "subtitle": '={{$parameter["operation"]}}',
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
            "properties": self._get_node_properties(),
        }

    def _get_node_properties(self) -> List[Dict[str, Any]]:
        """Get n8n node properties definition."""
        return [
            {
                "displayName": "Resource",
                "name": "resource",
                "type": "options",
                "noDataExpression": True,
                "options": [
                    {"name": "Debate", "value": "debate"},
                    {"name": "Knowledge", "value": "knowledge"},
                    {"name": "Agent", "value": "agent"},
                    {"name": "Decision", "value": "decision"},
                ],
                "default": "debate",
            },
            # Debate operations
            {
                "displayName": "Operation",
                "name": "operation",
                "type": "options",
                "noDataExpression": True,
                "displayOptions": {
                    "show": {"resource": ["debate"]},
                },
                "options": [
                    {"name": "Start Debate", "value": "start", "action": "Start a new debate"},
                    {"name": "Get Status", "value": "status", "action": "Get debate status"},
                    {"name": "Get Result", "value": "result", "action": "Get debate result"},
                    {"name": "List Debates", "value": "list", "action": "List debates"},
                ],
                "default": "start",
            },
            # Knowledge operations
            {
                "displayName": "Operation",
                "name": "operation",
                "type": "options",
                "noDataExpression": True,
                "displayOptions": {
                    "show": {"resource": ["knowledge"]},
                },
                "options": [
                    {"name": "Query", "value": "query", "action": "Query knowledge base"},
                    {"name": "Add", "value": "add", "action": "Add to knowledge base"},
                    {"name": "Search", "value": "search", "action": "Search knowledge"},
                ],
                "default": "query",
            },
            # Debate parameters
            {
                "displayName": "Task",
                "name": "task",
                "type": "string",
                "default": "",
                "required": True,
                "displayOptions": {
                    "show": {
                        "resource": ["debate"],
                        "operation": ["start"],
                    },
                },
                "description": "The question or task to debate",
            },
            {
                "displayName": "Agents",
                "name": "agents",
                "type": "multiOptions",
                "default": ["claude", "gpt-4"],
                "displayOptions": {
                    "show": {
                        "resource": ["debate"],
                        "operation": ["start"],
                    },
                },
                "options": [
                    {"name": "Claude", "value": "claude"},
                    {"name": "GPT-4", "value": "gpt-4"},
                    {"name": "Gemini", "value": "gemini"},
                    {"name": "Grok", "value": "grok"},
                    {"name": "Mistral", "value": "mistral"},
                ],
                "description": "AI agents to participate in the debate",
            },
            # Knowledge parameters
            {
                "displayName": "Query",
                "name": "query",
                "type": "string",
                "default": "",
                "required": True,
                "displayOptions": {
                    "show": {
                        "resource": ["knowledge"],
                        "operation": ["query", "search"],
                    },
                },
                "description": "Search query for knowledge base",
            },
        ]

    def get_credentials_definition(self) -> Dict[str, Any]:
        """
        Get n8n credentials definition for Aragora API.

        Returns:
            n8n credentials definition
        """
        return {
            "name": "aragoraApi",
            "displayName": "Aragora API",
            "documentationUrl": "https://docs.aragora.ai/api",
            "properties": [
                {
                    "displayName": "API URL",
                    "name": "apiUrl",
                    "type": "string",
                    "default": self.aragora_base_url,
                    "description": "Base URL of your Aragora instance",
                },
                {
                    "displayName": "API Token",
                    "name": "apiToken",
                    "type": "string",
                    "typeOptions": {"password": True},
                    "default": "",
                    "description": "Your Aragora API token",
                },
            ],
            "authenticate": {
                "type": "generic",
                "properties": {
                    "headers": {
                        "Authorization": "=Bearer {{$credentials.apiToken}}",
                    },
                },
            },
        }

    def get_trigger_definition(self) -> Dict[str, Any]:
        """
        Get n8n trigger node definition for Aragora events.

        Returns:
            n8n trigger node definition
        """
        return {
            "name": "AragoraTrigger",
            "displayName": "Aragora Trigger",
            "description": "Triggers workflow on Aragora events",
            "icon": "file:aragora.svg",
            "group": ["trigger"],
            "version": 1,
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
                    "path": "aragora",
                }
            ],
            "properties": [
                {
                    "displayName": "Events",
                    "name": "events",
                    "type": "multiOptions",
                    "default": [],
                    "options": [
                        {"name": "Debate Completed", "value": "debate.completed"},
                        {"name": "Consensus Reached", "value": "consensus.reached"},
                        {"name": "Decision Made", "value": "decision.made"},
                        {"name": "Knowledge Added", "value": "knowledge.added"},
                        {"name": "Agent Response", "value": "agent.response"},
                    ],
                    "description": "Events to trigger on",
                },
            ],
        }
