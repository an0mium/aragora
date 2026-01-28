"""
HTTP Handler for Automation Platform Webhooks.

Provides REST endpoints for managing webhook subscriptions with
automation platforms like Zapier and n8n.

Endpoints:
- POST /api/v1/webhooks/subscribe - Create new subscription
- DELETE /api/v1/webhooks/{id} - Remove subscription
- GET /api/v1/webhooks - List subscriptions
- GET /api/v1/webhooks/{id} - Get subscription details
- POST /api/v1/webhooks/{id}/test - Test subscription
- GET /api/v1/webhooks/events - List available event types
- POST /api/v1/webhooks/dispatch - Dispatch event (internal)

Supports:
- Zapier webhooks
- n8n webhooks
- Generic webhook subscriptions
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from aragora.connectors.automation import (
    AutomationEventType,
    N8NConnector,
    ZapierConnector,
)
from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)


class AutomationHandler(BaseHandler):
    """
    HTTP handler for automation webhook management.

    Manages webhook subscriptions for Zapier, n8n, and other
    automation platforms.
    """

    HANDLED_PATHS = [
        "/api/v1/webhooks",
        "/api/v1/webhooks/subscribe",
        "/api/v1/webhooks/events",
        "/api/v1/webhooks/dispatch",
        "/api/v1/webhooks/platforms",
        # n8n-specific endpoints
        "/api/v1/n8n/node",
        "/api/v1/n8n/credentials",
        "/api/v1/n8n/trigger",
    ]

    def __init__(self) -> None:
        """Initialize automation handler."""
        super().__init__()
        self._zapier = ZapierConnector()
        self._n8n = N8NConnector()
        self._connectors = {
            "zapier": self._zapier,
            "n8n": self._n8n,
            "generic": self._zapier,  # Default to Zapier-style
        }
        logger.info("[AutomationHandler] Initialized")

    def handle_get(
        self,
        path: str,
        query_params: Dict[str, Any],
        handler: Any,
    ) -> Optional[HandlerResult]:
        """Handle GET requests."""
        # List webhooks
        if path == "/api/v1/webhooks":
            return self._list_webhooks(query_params)

        # Get specific webhook
        if path.startswith("/api/v1/webhooks/") and not path.endswith("/test"):
            webhook_id = path.split("/")[-1]
            if webhook_id not in ["events", "subscribe", "dispatch", "platforms"]:
                return self._get_webhook(webhook_id)

        # List available events
        if path == "/api/v1/webhooks/events":
            return self._list_events()

        # List supported platforms
        if path == "/api/v1/webhooks/platforms":
            return self._list_platforms()

        # n8n node definition
        if path == "/api/v1/n8n/node":
            return self._get_n8n_node()

        # n8n credentials definition
        if path == "/api/v1/n8n/credentials":
            return self._get_n8n_credentials()

        # n8n trigger definition
        if path == "/api/v1/n8n/trigger":
            return self._get_n8n_trigger()

        return None

    async def handle_post(
        self,
        path: str,
        query_params: Dict[str, Any],
        handler: Any,
    ) -> Optional[HandlerResult]:
        """Handle POST requests."""
        # Subscribe to webhooks
        if path in ["/api/v1/webhooks", "/api/v1/webhooks/subscribe"]:
            return await self._subscribe(handler)

        # Test a webhook
        if path.startswith("/api/v1/webhooks/") and path.endswith("/test"):
            webhook_id = path.split("/")[-2]
            return await self._test_webhook(webhook_id)

        # Dispatch event (internal use)
        if path == "/api/v1/webhooks/dispatch":
            return await self._dispatch_event(handler)

        return None

    def handle_delete(
        self,
        path: str,
        query_params: Dict[str, Any],
        handler: Any,
    ) -> Optional[HandlerResult]:
        """Handle DELETE requests."""
        if path.startswith("/api/v1/webhooks/"):
            webhook_id = path.split("/")[-1]
            return self._unsubscribe(webhook_id)
        return None

    def _list_webhooks(self, query_params: Dict[str, Any]) -> HandlerResult:
        """List all webhook subscriptions."""
        platform = query_params.get("platform")
        workspace_id = query_params.get("workspace_id")
        event = query_params.get("event")

        event_type = AutomationEventType(event) if event else None

        all_subs: List[Dict[str, Any]] = []
        for name, connector in self._connectors.items():
            if platform and name != platform:
                continue
            subs = connector.list_subscriptions(
                workspace_id=workspace_id,
                event_type=event_type,
            )
            all_subs.extend([s.to_dict() for s in subs])

        return success_response(
            {
                "subscriptions": all_subs,
                "count": len(all_subs),
            }
        )

    def _get_webhook(self, webhook_id: str) -> HandlerResult:
        """Get a specific webhook subscription."""
        for connector in self._connectors.values():
            sub = connector.get_subscription(webhook_id)
            if sub:
                return success_response(sub.to_dict())

        return error_response(f"Webhook {webhook_id} not found", status_code=404)

    def _list_events(self) -> HandlerResult:
        """List available automation event types."""
        events = []
        for event in AutomationEventType:
            category, action = (
                event.value.split(".") if "." in event.value else (event.value, event.value)
            )
            events.append(
                {
                    "type": event.value,
                    "category": category,
                    "action": action,
                    "name": event.name,
                }
            )

        # Group by category
        categories: Dict[str, List[Dict[str, Any]]] = {}
        for event in events:
            cat = event["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(event)

        return success_response(
            {
                "events": events,
                "categories": categories,
                "count": len(events),
            }
        )

    def _list_platforms(self) -> HandlerResult:
        """List supported automation platforms."""
        platforms = [
            {
                "id": "zapier",
                "name": "Zapier",
                "description": "Connect Aragora to 5000+ apps",
                "documentation": "https://docs.aragora.ai/integrations/zapier",
                "features": ["triggers", "actions", "instant_triggers"],
            },
            {
                "id": "n8n",
                "name": "n8n",
                "description": "Self-hosted workflow automation",
                "documentation": "https://docs.aragora.ai/integrations/n8n",
                "features": ["triggers", "actions", "community_node"],
            },
            {
                "id": "generic",
                "name": "Generic Webhook",
                "description": "Custom webhook integration",
                "documentation": "https://docs.aragora.ai/integrations/webhooks",
                "features": ["triggers"],
            },
        ]

        return success_response(
            {
                "platforms": platforms,
                "count": len(platforms),
            }
        )

    async def _subscribe(self, handler: Any) -> HandlerResult:
        """Create a new webhook subscription."""
        try:
            body = self._get_request_body(handler)
        except Exception as e:
            return error_response(f"Invalid request body: {e}", status_code=400)

        webhook_url = body.get("webhook_url") or body.get("url")
        if not webhook_url:
            return error_response("webhook_url is required", status_code=400)

        events_raw = body.get("events", [])
        if not events_raw:
            return error_response("events list is required", status_code=400)

        # Parse events
        try:
            events = [AutomationEventType(e) for e in events_raw]
        except ValueError as e:
            return error_response(f"Invalid event type: {e}", status_code=400)

        platform = body.get("platform", "generic")
        connector = self._connectors.get(platform)
        if not connector:
            return error_response(f"Unknown platform: {platform}", status_code=400)

        workspace_id = body.get("workspace_id")
        user_id = body.get("user_id")
        name = body.get("name")

        try:
            subscription = await connector.subscribe(
                webhook_url=webhook_url,
                events=events,
                workspace_id=workspace_id,
                user_id=user_id,
                name=name,
            )

            logger.info(f"[AutomationHandler] Created subscription {subscription.id}")

            return success_response(
                {
                    "subscription": subscription.to_dict(),
                    "secret": subscription.secret,  # Only returned on creation
                    "message": "Webhook subscription created successfully",
                },
                status_code=201,
            )

        except Exception as e:
            logger.error(f"[AutomationHandler] Subscribe failed: {e}")
            return error_response(f"Failed to create subscription: {e}", status_code=500)

    def _unsubscribe(self, webhook_id: str) -> HandlerResult:
        """Remove a webhook subscription."""
        import asyncio

        for connector in self._connectors.values():
            # Run async unsubscribe in event loop
            loop = asyncio.new_event_loop()
            try:
                removed = loop.run_until_complete(connector.unsubscribe(webhook_id))
            finally:
                loop.close()

            if removed:
                logger.info(f"[AutomationHandler] Removed subscription {webhook_id}")
                return success_response(
                    {
                        "message": f"Webhook {webhook_id} removed successfully",
                    }
                )

        return error_response(f"Webhook {webhook_id} not found", status_code=404)

    async def _test_webhook(self, webhook_id: str) -> HandlerResult:
        """Test a webhook subscription."""
        for name, connector in self._connectors.items():
            sub = connector.get_subscription(webhook_id)
            if sub:
                if name == "zapier":
                    success = await self._zapier.test_subscription(sub)
                else:
                    # Generic test
                    results = await connector.dispatch_event(
                        AutomationEventType.TEST_EVENT,
                        {"test": True, "message": "Test from Aragora"},
                        workspace_id=sub.workspace_id,
                    )
                    success = any(r.success for r in results if r.subscription_id == webhook_id)

                if success:
                    return success_response(
                        {
                            "success": True,
                            "message": "Webhook test successful",
                            "verified": sub.verified,
                        }
                    )
                else:
                    return success_response(
                        {
                            "success": False,
                            "message": "Webhook test failed",
                            "verified": False,
                        }
                    )

        return error_response(f"Webhook {webhook_id} not found", status_code=404)

    async def _dispatch_event(self, handler: Any) -> HandlerResult:
        """Dispatch an event to matching subscriptions (internal)."""
        try:
            body = self._get_request_body(handler)
        except Exception as e:
            return error_response(f"Invalid request body: {e}", status_code=400)

        event_raw = body.get("event_type") or body.get("event")
        if not event_raw:
            return error_response("event_type is required", status_code=400)

        try:
            event_type = AutomationEventType(event_raw)
        except ValueError:
            return error_response(f"Invalid event type: {event_raw}", status_code=400)

        payload = body.get("payload", {})
        workspace_id = body.get("workspace_id")

        all_results = []
        for connector in self._connectors.values():
            results = await connector.dispatch_event(
                event_type,
                payload,
                workspace_id=workspace_id,
            )
            all_results.extend(results)

        success_count = sum(1 for r in all_results if r.success)
        failure_count = len(all_results) - success_count

        return success_response(
            {
                "dispatched": len(all_results),
                "success": success_count,
                "failed": failure_count,
                "results": [
                    {
                        "subscription_id": r.subscription_id,
                        "success": r.success,
                        "status_code": r.status_code,
                        "error": r.error,
                        "duration_ms": r.duration_ms,
                    }
                    for r in all_results
                ],
            }
        )

    def _get_n8n_node(self) -> HandlerResult:
        """Get n8n community node definition."""
        return success_response(self._n8n.get_node_definition())

    def _get_n8n_credentials(self) -> HandlerResult:
        """Get n8n credentials definition."""
        return success_response(self._n8n.get_credentials_definition())

    def _get_n8n_trigger(self) -> HandlerResult:
        """Get n8n trigger node definition."""
        return success_response(self._n8n.get_trigger_definition())

    def _get_request_body(self, handler: Any) -> Dict[str, Any]:
        """Extract request body from handler."""
        if hasattr(handler, "request") and hasattr(handler.request, "body"):
            body = handler.request.body
            if isinstance(body, bytes):
                return json.loads(body.decode("utf-8"))
            return json.loads(body) if isinstance(body, str) else body
        return {}
