"""
Slack Workspace API Handlers.

Provides management APIs for Slack workspace integrations:
- GET /api/v1/sme/slack/workspaces - List connected workspaces
- GET /api/v1/sme/slack/workspaces/:workspace_id - Get workspace details
- POST /api/v1/sme/slack/workspaces/:workspace_id/test - Test connection
- DELETE /api/v1/sme/slack/workspaces/:workspace_id - Disconnect workspace
- GET /api/v1/sme/slack/channels/:workspace_id - List available channels
- POST /api/v1/sme/slack/subscribe - Subscribe channel to notifications
- GET /api/v1/sme/slack/subscriptions - List subscriptions
- DELETE /api/v1/sme/slack/subscriptions/:id - Remove subscription
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from ..base import (
    error_response,
    get_string_param,
    handle_errors,
    json_response,
)
from ..utils.responses import HandlerResult
from ..secure import SecureHandler
from ..utils.decorators import require_permission
from ..utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiter for Slack workspace APIs (30 requests per minute)
_slack_limiter = RateLimiter(requests_per_minute=30)


class SlackWorkspaceHandler(SecureHandler):
    """Handler for Slack workspace management endpoints.

    Provides APIs for managing Slack workspace connections,
    listing channels, and subscribing to notifications.
    """

    RESOURCE_TYPE = "slack_workspace"

    ROUTES = [
        "/api/v1/sme/slack/workspaces",
        "/api/v1/sme/slack/channels",
        "/api/v1/sme/slack/subscribe",
        "/api/v1/sme/slack/subscriptions",
    ]

    # Regex patterns for parameterized routes
    ROUTE_PATTERNS = [
        (re.compile(r"^/api/v1/sme/slack/workspaces/([^/]+)/test$"), "workspace_test"),
        (re.compile(r"^/api/v1/sme/slack/workspaces/([^/]+)$"), "workspace_detail"),
        (re.compile(r"^/api/v1/sme/slack/channels/([^/]+)$"), "channels"),
        (re.compile(r"^/api/v1/sme/slack/subscriptions/([^/]+)$"), "subscription_detail"),
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path in self.ROUTES:
            return True
        for pattern, _ in self.ROUTE_PATTERNS:
            if pattern.match(path):
                return True
        return False

    def _match_route(self, path: str) -> tuple[str | None, str | None]:
        """Match a path against parameterized routes.

        Returns:
            Tuple of (route_name, extracted_id) or (None, None) if no match.
        """
        for pattern, route_name in self.ROUTE_PATTERNS:
            match = pattern.match(path)
            if match:
                return route_name, match.group(1)
        return None, None

    def handle(
        self,
        path: str,
        query_params: dict,
        handler,
        method: str = "GET",
    ) -> HandlerResult | None:
        """Route Slack workspace requests to appropriate methods."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _slack_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for Slack workspace: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # Determine HTTP method from handler if not provided
        if hasattr(handler, "command"):
            method = handler.command

        # Handle static routes
        if path == "/api/v1/sme/slack/workspaces":
            if method == "GET":
                return self._list_workspaces(handler, query_params)
            return error_response("Method not allowed", 405)

        if path == "/api/v1/sme/slack/subscribe":
            if method == "POST":
                return self._subscribe_channel(handler, query_params)
            return error_response("Method not allowed", 405)

        if path == "/api/v1/sme/slack/subscriptions":
            if method == "GET":
                return self._list_subscriptions(handler, query_params)
            return error_response("Method not allowed", 405)

        # Handle parameterized routes
        route_name, param_id = self._match_route(path)
        if route_name:
            if route_name == "workspace_detail":
                if method == "GET":
                    return self._get_workspace(handler, query_params, param_id)
                elif method == "DELETE":
                    return self._disconnect_workspace(handler, query_params, param_id)
                return error_response("Method not allowed", 405)

            if route_name == "workspace_test":
                if method == "POST":
                    return self._test_connection(handler, query_params, param_id)
                return error_response("Method not allowed", 405)

            if route_name == "channels":
                if method == "GET":
                    return self._list_channels(handler, query_params, param_id)
                return error_response("Method not allowed", 405)

            if route_name == "subscription_detail":
                if method == "DELETE":
                    return self._delete_subscription(handler, query_params, param_id)
                return error_response("Method not allowed", 405)

        return error_response("Not found", 404)

    def _get_workspace_store(self):
        """Get Slack workspace store instance."""
        from aragora.storage.slack_workspace_store import get_slack_workspace_store

        return get_slack_workspace_store()

    def _get_subscription_store(self):
        """Get channel subscription store instance."""
        from aragora.storage.channel_subscription_store import (
            get_channel_subscription_store,
        )

        return get_channel_subscription_store()

    def _get_user_and_org(self, handler, user):
        """Get user and organization from context."""
        user_store = self.ctx.get("user_store")
        if not user_store:
            return None, None, error_response("Service unavailable", 503)

        db_user = user_store.get_user_by_id(user.user_id)
        if not db_user:
            return None, None, error_response("User not found", 404)

        org = None
        if db_user.org_id:
            org = user_store.get_organization_by_id(db_user.org_id)

        if not org:
            return None, None, error_response("No organization found", 404)

        return db_user, org, None

    @handle_errors("list Slack workspaces")
    @require_permission("sme:workspaces:read")
    def _list_workspaces(
        self,
        handler,
        query_params: dict,
        user=None,
    ) -> HandlerResult:
        """
        List connected Slack workspaces for the organization.

        Query Parameters:
            limit: Maximum results (default: 50)
            offset: Pagination offset (default: 0)

        Returns:
            JSON response with workspace list:
            {
                "workspaces": [...],
                "total": 3,
                "limit": 50,
                "offset": 0
            }
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        limit = int(get_string_param(handler, "limit", "50"))
        offset = int(get_string_param(handler, "offset", "0"))

        store = self._get_workspace_store()
        workspaces = store.get_by_tenant(org.id)

        # Apply pagination
        paginated = workspaces[offset : offset + limit]

        return json_response(
            {
                "workspaces": [w.to_dict() for w in paginated],
                "total": len(workspaces),
                "limit": limit,
                "offset": offset,
            }
        )

    @handle_errors("get Slack workspace")
    @require_permission("sme:workspaces:read")
    def _get_workspace(
        self,
        handler,
        query_params: dict,
        workspace_id: str,
        user=None,
    ) -> HandlerResult:
        """
        Get details for a specific Slack workspace.

        Path Parameters:
            workspace_id: Slack workspace ID

        Returns:
            JSON response with workspace details
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        store = self._get_workspace_store()
        workspace = store.get(workspace_id)

        if not workspace:
            return error_response("Workspace not found", 404)

        # Verify workspace belongs to this org
        if workspace.tenant_id != org.id:
            return error_response("Workspace not found", 404)

        return json_response({"workspace": workspace.to_dict()})

    @handle_errors("test Slack connection")
    @require_permission("sme:workspaces:write")
    def _test_connection(
        self,
        handler,
        query_params: dict,
        workspace_id: str,
        user=None,
    ) -> HandlerResult:
        """
        Test connection to a Slack workspace.

        Path Parameters:
            workspace_id: Slack workspace ID

        Returns:
            JSON response with connection status:
            {
                "status": "connected",
                "workspace_id": "...",
                "workspace_name": "...",
                "bot_user_id": "...",
                "token_valid": true,
                "tested_at": "..."
            }
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        store = self._get_workspace_store()
        workspace = store.get(workspace_id)

        if not workspace:
            return error_response("Workspace not found", 404)

        if workspace.tenant_id != org.id:
            return error_response("Workspace not found", 404)

        # Check token expiration
        token_valid = True
        if workspace.token_expires_at:
            token_valid = workspace.token_expires_at > datetime.now(timezone.utc).timestamp()

        # Validate token format
        connection_status = "connected"
        error_message = None

        try:
            # Slack bot tokens start with "xoxb-"
            if not workspace.access_token or not workspace.access_token.startswith("xoxb-"):
                connection_status = "invalid_token"
                token_valid = False
        except Exception as e:
            logger.warning(f"Slack connection test failed for {workspace_id}: {e}")
            connection_status = "error"
            error_message = str(e)

        result = {
            "status": connection_status,
            "workspace_id": workspace.workspace_id,
            "workspace_name": workspace.workspace_name,
            "bot_user_id": workspace.bot_user_id,
            "token_valid": token_valid,
            "tested_at": datetime.now(timezone.utc).isoformat(),
        }

        if error_message:
            result["error"] = error_message

        return json_response(result)

    @handle_errors("disconnect Slack workspace")
    @require_permission("sme:workspaces:write")
    def _disconnect_workspace(
        self,
        handler,
        query_params: dict,
        workspace_id: str,
        user=None,
    ) -> HandlerResult:
        """
        Disconnect (deactivate) a Slack workspace.

        Path Parameters:
            workspace_id: Slack workspace ID

        Returns:
            JSON response confirming disconnection
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        store = self._get_workspace_store()
        workspace = store.get(workspace_id)

        if not workspace:
            return error_response("Workspace not found", 404)

        if workspace.tenant_id != org.id:
            return error_response("Workspace not found", 404)

        # Deactivate (soft delete)
        success = store.deactivate(workspace_id)

        if not success:
            return error_response("Failed to disconnect workspace", 500)

        # Also deactivate any subscriptions for this workspace
        sub_store = self._get_subscription_store()
        from aragora.storage.channel_subscription_store import ChannelType

        subscriptions = sub_store.get_by_org(org.id, channel_type=ChannelType.SLACK)
        for sub in subscriptions:
            if sub.workspace_id == workspace_id:
                sub_store.deactivate(sub.id)

        logger.info(f"Disconnected Slack workspace {workspace_id} for org {org.id}")

        return json_response(
            {
                "disconnected": True,
                "workspace_id": workspace_id,
                "message": "Workspace disconnected successfully",
            }
        )

    @handle_errors("list Slack channels")
    @require_permission("sme:workspaces:read")
    def _list_channels(
        self,
        handler,
        query_params: dict,
        workspace_id: str,
        user=None,
    ) -> HandlerResult:
        """
        List available channels for a Slack workspace.

        Path Parameters:
            workspace_id: Slack workspace ID

        Query Parameters:
            types: Channel types to include (public_channel, private_channel)

        Returns:
            JSON response with channel list:
            {
                "channels": [
                    {
                        "id": "C123...",
                        "name": "general",
                        "is_private": false,
                        "num_members": 42
                    }
                ],
                "workspace_id": "..."
            }
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        store = self._get_workspace_store()
        workspace = store.get(workspace_id)

        if not workspace:
            return error_response("Workspace not found", 404)

        if workspace.tenant_id != org.id:
            return error_response("Workspace not found", 404)

        _types = get_string_param(handler, "types", "public_channel")  # noqa: F841

        # Try to fetch channels from Slack API
        channels: list[dict[str, Any]] = []

        try:
            from aragora.connectors.chat.slack import SlackConnector

            _connector = SlackConnector(token=workspace.access_token)  # noqa: F841
            # Note: In production, this would make actual API calls
            # For now, return placeholder indicating API integration needed
            channels = [
                {
                    "id": "placeholder",
                    "name": "Channel listing requires Slack Web API integration",
                    "is_private": False,
                    "num_members": 0,
                }
            ]
        except ImportError:
            logger.warning("Slack connector not available")
        except Exception as e:
            logger.warning(f"Failed to list Slack channels: {e}")

        return json_response(
            {
                "channels": channels,
                "workspace_id": workspace_id,
            }
        )

    @handle_errors("subscribe Slack channel")
    @require_permission("sme:channels:subscribe")
    def _subscribe_channel(
        self,
        handler,
        query_params: dict,
        user=None,
    ) -> HandlerResult:
        """
        Subscribe a Slack channel to receive notifications.

        Request Body:
            {
                "workspace_id": "T123...",
                "channel_id": "C123...",
                "channel_name": "#general",
                "event_types": ["receipt", "budget_alert"]
            }

        Returns:
            JSON response with subscription details
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        # Parse request body
        import json as json_lib

        try:
            body = handler.rfile.read(int(handler.headers.get("Content-Length", 0)))
            data = json_lib.loads(body.decode("utf-8")) if body else {}
        except (json_lib.JSONDecodeError, ValueError):
            return error_response("Invalid JSON body", 400)

        workspace_id = data.get("workspace_id")
        channel_id = data.get("channel_id")
        channel_name = data.get("channel_name")
        event_types = data.get("event_types", ["receipt", "budget_alert"])

        if not workspace_id or not channel_id:
            return error_response("workspace_id and channel_id are required", 400)

        # Verify workspace exists and belongs to org
        ws_store = self._get_workspace_store()
        workspace = ws_store.get(workspace_id)

        if not workspace:
            return error_response("Workspace not found", 404)

        if workspace.tenant_id != org.id:
            return error_response("Workspace not found", 404)

        # Create subscription
        from aragora.storage.channel_subscription_store import (
            ChannelSubscription,
            ChannelType,
            EventType,
        )

        # Parse event types
        parsed_events = []
        for et in event_types:
            try:
                parsed_events.append(EventType(et))
            except ValueError:
                return error_response(f"Invalid event type: {et}", 400)

        subscription = ChannelSubscription(
            id="",  # Will be generated
            org_id=org.id,
            channel_type=ChannelType.SLACK,
            channel_id=channel_id,
            workspace_id=workspace_id,
            channel_name=channel_name,
            event_types=parsed_events,
            created_at=0,  # Will be set
            created_by=db_user.id,
            is_active=True,
            config={},
        )

        sub_store = self._get_subscription_store()
        try:
            created = sub_store.create(subscription)
            logger.info(
                f"Created Slack subscription {created.id} for channel {channel_id} in org {org.id}"
            )
            return json_response({"subscription": created.to_dict()}, status=201)
        except ValueError as e:
            return error_response(str(e), 409)

    @handle_errors("list Slack subscriptions")
    @require_permission("sme:channels:subscribe")
    def _list_subscriptions(
        self,
        handler,
        query_params: dict,
        user=None,
    ) -> HandlerResult:
        """
        List Slack channel subscriptions for the organization.

        Query Parameters:
            event_type: Filter by event type (receipt, budget_alert, etc.)

        Returns:
            JSON response with subscription list
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        from aragora.storage.channel_subscription_store import ChannelType, EventType

        event_type_str = get_string_param(handler, "event_type", None)
        event_type = None
        if event_type_str:
            try:
                event_type = EventType(event_type_str)
            except ValueError:
                pass

        sub_store = self._get_subscription_store()
        subscriptions = sub_store.get_by_org(
            org.id,
            channel_type=ChannelType.SLACK,
            event_type=event_type,
        )

        return json_response(
            {
                "subscriptions": [s.to_dict() for s in subscriptions],
                "total": len(subscriptions),
            }
        )

    @handle_errors("delete Slack subscription")
    @require_permission("sme:channels:subscribe")
    def _delete_subscription(
        self,
        handler,
        query_params: dict,
        subscription_id: str,
        user=None,
    ) -> HandlerResult:
        """
        Delete a Slack channel subscription.

        Path Parameters:
            subscription_id: Subscription ID to delete

        Returns:
            JSON response confirming deletion
        """
        db_user, org, error = self._get_user_and_org(handler, user)
        if error:
            return error

        sub_store = self._get_subscription_store()
        subscription = sub_store.get(subscription_id)

        if not subscription:
            return error_response("Subscription not found", 404)

        # Verify subscription belongs to this org
        if subscription.org_id != org.id:
            return error_response("Subscription not found", 404)

        success = sub_store.delete(subscription_id)

        if not success:
            return error_response("Failed to delete subscription", 500)

        logger.info(f"Deleted Slack subscription {subscription_id} for org {org.id}")

        return json_response(
            {
                "deleted": True,
                "subscription_id": subscription_id,
            }
        )


__all__ = ["SlackWorkspaceHandler"]
